"""Controlled, local categorical-circuit noise tests for Stage B.

These deliberately generic channels are not an IBM calibration model.  A
depolarizing channel follows every compiled one-/two-qubit gate, including
``rz``.  All-to-all compilation does not model routing or a hardware topology.
Noisy evolution uses an exact full-space density matrix; independent symmetric
readout flips are then applied to its complete probability vector.  Feasible
probabilities are unconditional, so no postselection or fallback is hidden.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .categorical_ibm import build_categorical_qaoa_circuit
from .categorical_qaoa import categorical_qaoa_probabilities, encode_onehot


@dataclass(frozen=True)
class NoiseParameters:
    label: str
    single_qubit_depolarizing: float
    two_qubit_depolarizing: float
    readout_bitflip: float

    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("Noise label must be nonempty text")
        values = (self.single_qubit_depolarizing, self.two_qubit_depolarizing,
                  self.readout_bitflip)
        if any(isinstance(value, (bool, np.bool_)) or not np.isfinite(value)
               or not 0 <= value <= 1 for value in values):
            raise ValueError("Channel parameters must be finite numbers in [0, 1]")


NOISE_LEVELS = {
    "ideal": NoiseParameters("ideal", 0.0, 0.0, 0.0),
    "low": NoiseParameters("low", 0.0001, 0.001, 0.005),
    "high": NoiseParameters("high", 0.001, 0.01, 0.02),
}
BASIS_GATES = ("rz", "sx", "x", "cx")
TRANSPILE_SEED = 17
OPTIMIZATION_LEVEL = 1
MAX_DENSITY_QUBITS = 12


@dataclass(frozen=True)
class PreparedNoiseCircuit:
    circuit: object
    feasible_indices: np.ndarray
    ideal_physical_probabilities: np.ndarray
    metadata: dict


def _validated_probabilities(probabilities):
    """Remove floating-point roundoff only, never condition on feasibility."""
    result = np.asarray(probabilities, dtype=float).copy()
    if result.ndim != 1 or result.size < 2 or result.size & (result.size - 1):
        raise ValueError("A complete power-of-two physical probability vector is required")
    if not np.all(np.isfinite(result)) or np.min(result) < -1e-12:
        raise ValueError("Nonfinite or materially negative physical probability")
    total = float(np.sum(result))
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Physical probability mass is not one: {total}")
    negative_roundoff = float(-np.sum(result[result < 0]))
    np.maximum(result, 0.0, out=result)
    result /= np.sum(result)
    return result, {"raw_probability_mass": total,
                    "negative_roundoff_removed": negative_roundoff}


def apply_readout_bitflips(probabilities, probability: float):
    """Apply the exact independent binary-symmetric channel on all bits.

    This includes transfers both into and out of the feasible subspace.  It
    does not multiply by a no-error probability or discard malformed strings.
    Array index bits follow Qiskit's least-significant-qubit convention.
    """
    if (isinstance(probability, (bool, np.bool_)) or not np.isfinite(probability)
            or not 0 <= probability <= 1):
        raise ValueError("Readout flip probability must lie in [0, 1]")
    result, _ = _validated_probabilities(probabilities)
    for qubit in range(result.size.bit_length() - 1):
        stride = 1 << qubit
        blocks = result.reshape(-1, 2, stride)
        zero = blocks[:, 0, :].copy()
        one = blocks[:, 1, :].copy()
        blocks[:, 0, :] = (1 - probability) * zero + probability * one
        blocks[:, 1, :] = probability * zero + (1 - probability) * one
    return result


def prepare_noise_circuit(problem, gammas, betas, *, max_qubits=MAX_DENSITY_QUBITS):
    """Compile and audit a bounded-width circuit without contacting hardware."""
    from qiskit import transpile
    from qiskit.quantum_info import Statevector

    if (isinstance(max_qubits, bool) or not isinstance(max_qubits, (int, np.integer))
            or not 1 <= max_qubits <= MAX_DENSITY_QUBITS):
        raise ValueError(f"max_qubits must be between 1 and {MAX_DENSITY_QUBITS}")
    if problem.num_qubits > max_qubits:
        raise ValueError(f"Exact noise simulation capped at {max_qubits} physical qubits")
    started = perf_counter()
    logical = build_categorical_qaoa_circuit(problem, gammas, betas, measure=False)
    compiled = transpile(logical, basis_gates=list(BASIS_GATES),
                         optimization_level=OPTIMIZATION_LEVEL,
                         seed_transpiler=TRANSPILE_SEED)
    compiled_seconds = perf_counter() - started
    if compiled.num_qubits != problem.num_qubits:
        raise ValueError("Unrouted compilation unexpectedly changed circuit width")
    if set(compiled.count_ops()) - set(BASIS_GATES):
        raise ValueError("Unexpected gate outside the frozen noise basis")
    audit_start = perf_counter()
    physical, roundoff = _validated_probabilities(Statevector.from_instruction(compiled).probabilities())
    indices = np.array([int(encode_onehot(state, problem.state_counts), 2)
                        for state in problem.states], dtype=np.int64)
    expected = categorical_qaoa_probabilities(problem, gammas, betas)
    error = float(np.max(np.abs(physical[indices] - expected)))
    ideal_invalid = max(0.0, 1.0 - float(physical[indices].sum()))
    if error > 1e-10 or ideal_invalid > 1e-10:
        raise ValueError(f"Compiled ideal distribution audit failed: {error=}, {ideal_invalid=}")
    metadata = {
        "num_qubits": int(compiled.num_qubits), "qaoa_depth": len(gammas),
        "basis_gates": list(BASIS_GATES), "connectivity": "all-to-all; no routing",
        "optimization_level": OPTIMIZATION_LEVEL, "transpile_seed": TRANSPILE_SEED,
        "compiled_depth": int(compiled.depth()),
        "compiled_gate_counts": dict(compiled.count_ops()),
        "compilation_seconds": compiled_seconds,
        "ideal_audit_seconds": perf_counter() - audit_start,
        "ideal_audit_max_absolute_probability_error": error,
        "ideal_audit_invalid_probability": ideal_invalid,
        "ideal_audit_roundoff": roundoff,
        "physical_basis_order": "Qiskit integer: qubit zero is least-significant bit",
        "feasible_basis_order": "same as problem.states; time-major mixed radix",
    }
    return PreparedNoiseCircuit(compiled, indices, physical, metadata)


def simulate_prepared_noise(prepared: PreparedNoiseCircuit, noise="ideal", *, max_parallel_threads=4):
    """Return exact all-shot probabilities and explicit invalid mass.

    Ideal simulation reuses the independently checked physical statevector.
    Every nonzero gate-noise case evolves a density matrix with no trajectories
    or Monte Carlo shots.  Numerical normalization changes only full-space
    roundoff, with the uncorrected mass archived; it never renormalizes feasible
    outcomes.  Sampling repeats should be performed by the caller afterwards.
    """
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    parameters = NOISE_LEVELS[noise] if isinstance(noise, str) else noise
    if not isinstance(parameters, NoiseParameters):
        raise ValueError("noise must be a named noise level or NoiseParameters")
    if (isinstance(max_parallel_threads, bool)
            or not isinstance(max_parallel_threads, (int, np.integer))
            or max_parallel_threads < 1):
        raise ValueError("max_parallel_threads must be a positive integer")
    started = perf_counter()
    model = NoiseModel()
    if parameters.single_qubit_depolarizing:
        model.add_all_qubit_quantum_error(
            depolarizing_error(parameters.single_qubit_depolarizing, 1), ["rz", "sx", "x"])
    if parameters.two_qubit_depolarizing:
        model.add_all_qubit_quantum_error(
            depolarizing_error(parameters.two_qubit_depolarizing, 2), ["cx"])
    if parameters.single_qubit_depolarizing or parameters.two_qubit_depolarizing:
        circuit = prepared.circuit.copy()
        circuit.save_probabilities(list(range(circuit.num_qubits)), label="physical_probabilities")
        simulator = AerSimulator(method="density_matrix", noise_model=model,
                                 max_parallel_threads=int(max_parallel_threads),
                                 max_parallel_experiments=1, max_parallel_shots=1)
        result = simulator.run(circuit, shots=None).result()
        if not result.success or not result.results[0].success:
            raise RuntimeError(f"Exact density-matrix simulation failed: {result.status}")
        physical, roundoff = _validated_probabilities(result.data(0)["physical_probabilities"])
        method = "exact_density_matrix"
    else:
        physical = prepared.ideal_physical_probabilities.copy()
        roundoff = {"raw_probability_mass": float(physical.sum()), "negative_roundoff_removed": 0.0}
        method = "exact_statevector_ideal_audit"
    physical = apply_readout_bitflips(physical, parameters.readout_bitflip)
    physical, final_roundoff = _validated_probabilities(physical)
    feasible = physical[prepared.feasible_indices].copy()
    invalid = max(0.0, 1 - float(np.sum(feasible)))
    return {
        "feasible_probabilities": feasible,
        "physical_probabilities": physical,
        "feasible_physical_indices": prepared.feasible_indices.copy(),
        "feasible_probability": float(np.sum(feasible)),
        "invalid_probability": invalid,
        "noise_parameters": asdict(parameters),
        "noise_model_description": "Depolarizing after every compiled rz/sx/x/cx; independent symmetric readout flips",
        "noise_model_is_hardware_calibration": False,
        "probabilities_conditioned_on_feasibility": False,
        "classical_fallback_used": False,
        "simulation_method": method,
        "simulation_seconds": perf_counter() - started,
        "max_parallel_threads": int(max_parallel_threads),
        "pre_readout_roundoff": roundoff, "final_roundoff": final_roundoff,
        "circuit": dict(prepared.metadata),
    }


def simulate_categorical_noise(problem, gammas, betas, noise="ideal", *,
                               max_qubits=MAX_DENSITY_QUBITS, max_parallel_threads=4):
    """Convenience wrapper; reuse preparation explicitly for a noise sweep."""
    prepared = prepare_noise_circuit(problem, gammas, betas, max_qubits=max_qubits)
    return simulate_prepared_noise(prepared, noise, max_parallel_threads=max_parallel_threads)
