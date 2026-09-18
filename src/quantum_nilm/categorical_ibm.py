"""Feasibility-preserving categorical Q.NILM circuits and explicit shot decoding.

This module does not load IBM accounts or submit jobs. Physical qubits are
time-major one-hot registers; within a register, category r is physical bit r.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class CategoricalCircuitTemplate:
    """One p=1 W/XY circuit with only cost rotations left symbolic."""

    circuit: object
    state_counts: tuple[int, ...]
    n_segments: int
    beta: float
    parameters: tuple
    edges: tuple[tuple[int, int], ...]


def prepare_uniform_onehot(circuit, qubits: Sequence[int]) -> None:
    """Prepare a positive-amplitude W state from an all-zero register.

    Each CRY/CX pair leaves 1/sqrt(m) on the current excitation and transfers
    the remaining amplitude to the next qubit. This is unitary state
    preparation; it uses no reset, measurement, or postselection.
    """
    qubits = list(qubits)
    if not qubits or len(set(qubits)) != len(qubits):
        raise ValueError("One-hot preparation requires distinct, nonempty qubits")
    circuit.x(qubits[0])
    for position in range(len(qubits) - 1):
        theta = 2 * np.arccos(1 / np.sqrt(len(qubits) - position))
        circuit.cry(float(theta), qubits[position], qubits[position + 1])
        circuit.cx(qubits[position + 1], qubits[position])


def _validate_encoding(problem) -> None:
    sizes = tuple(problem.register_sizes)
    expected = tuple(problem.state_counts) * len(problem.aggregate)
    if (sizes != expected or not sizes or any(size < 1 for size in sizes)
            or sum(sizes) != problem.num_qubits):
        raise ValueError("Expected nonempty time-major one-hot register sizes")
    offsets = tuple(np.cumsum((0,) + sizes[:-1]).tolist())
    if tuple(problem.register_offsets) != offsets:
        raise ValueError("Register offsets do not match the one-hot encoding")
    if np.shape(problem.linear) != (problem.num_qubits,) or not np.all(np.isfinite(problem.linear)):
        raise ValueError("Invalid categorical linear coefficients")
    if not np.isfinite(problem.scale) or problem.scale <= 0:
        raise ValueError("Cost phase scale must be positive and finite")
    for (left, right), value in problem.quadratic.items():
        if not 0 <= left < right < problem.num_qubits or not np.isfinite(value):
            raise ValueError("Expected finite upper-triangle quadratic coefficients")


def build_categorical_qaoa_circuit(problem, gammas, betas, *, measure: bool = True):
    """Build the same cost/W/chain-XY QAOA used by the categorical simulator.

    Each mixer pair is RXX(beta) followed by RYY(beta), implementing
    exp[-i beta(XX+YY)/2]. Pairs are applied in ascending register order and
    ascending adjacent-category order, not an unspecified simultaneous XY
    Hamiltonian. A finite Trotter chain is part of the algorithm definition.

    The supplied QUBO agrees with the categorical objective in the one-hot
    subspace; one-hot penalties are unnecessary. Hardware-invalid shots must
    be counted explicitly by score_categorical_counts, never silently repaired.
    """
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    _validate_encoding(problem)
    gamma_values = np.asarray(gammas, dtype=float)
    beta_values = np.asarray(betas, dtype=float)
    if (gamma_values.ndim != 1 or beta_values.ndim != 1 or not gamma_values.size
            or gamma_values.size != beta_values.size
            or not np.all(np.isfinite(gamma_values)) or not np.all(np.isfinite(beta_values))):
        raise ValueError("gammas and betas must be nonempty, finite vectors of equal length")
    if not isinstance(measure, bool):
        raise ValueError("measure must be boolean")
    quantum = QuantumRegister(problem.num_qubits, "q")
    circuit = QuantumCircuit(quantum, name="Q.NILM-categorical-QAOA")
    if measure:
        circuit.add_register(ClassicalRegister(problem.num_qubits, "meas"))
    for offset, size in zip(problem.register_offsets, problem.register_sizes):
        prepare_uniform_onehot(circuit, range(int(offset), int(offset + size)))
    # x=(1-Z)/2. The omitted scalar Ising offset is only a global phase.
    fields = -np.asarray(problem.linear, dtype=float) / 2
    for (left, right), value in problem.quadratic.items():
        fields[left] -= value / 4
        fields[right] -= value / 4
    for gamma, beta in zip(gamma_values, beta_values):
        for qubit, field in enumerate(fields):
            if field:
                circuit.rz(float(2 * gamma * field / problem.scale), qubit)
        for (left, right), value in sorted(problem.quadratic.items()):
            if value:
                circuit.rzz(float(gamma * value / (2 * problem.scale)), left, right)
        for offset, size in zip(problem.register_offsets, problem.register_sizes):
            for pair in range(int(size) - 1):
                left = int(offset + pair)
                circuit.rxx(float(beta), left, left + 1)
                circuit.ryy(float(beta), left, left + 1)
    if measure:
        circuit.measure(quantum, circuit.cregs[0])
    circuit.metadata = {
        "algorithm": "Q.NILM categorical W-state/chain-XY QAOA",
        "qaoa_depth": int(gamma_values.size),
        "gammas": gamma_values.tolist(), "betas": beta_values.tolist(),
        "phase_scale": float(problem.scale),
        "state_counts": [int(value) for value in problem.state_counts],
        "register_order": "time-major, channel-major; category r is offset+r",
        "mixer": "ascending chain; RXX(beta) then RYY(beta) per pair",
        "invalid_shot_policy": "report infeasible shots; no repair and no classical fallback",
    }
    return circuit


def build_categorical_parameterized_template(state_counts, n_segments: int, beta: float):
    """Build a reusable p=1 template for one or two categorical intervals.

    Compile this circuit once for each interval count. New aggregate readings,
    weights, gamma, and method-own preceding states only change cost parameter
    bindings. Beta and register topology are fixed before compilation.
    """
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit import ParameterVector

    sizes = tuple(state_counts)
    if (not sizes or any(isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral)
                         or size < 1 for size in sizes)
            or isinstance(n_segments, (bool, np.bool_)) or not isinstance(n_segments, Integral)
            or n_segments not in (1, 2)
            or not np.isfinite(beta)):
        raise ValueError("Require positive state counts, one/two intervals, and finite beta")
    sizes = tuple(int(size) for size in sizes)
    widths = sizes * n_segments
    if sum(widths) > 256:
        raise ValueError("Template exceeds 256 physical qubits")
    offsets = np.cumsum((0,) + widths[:-1]).tolist()
    edges = []
    for segment in range(n_segments):
        for left_channel, left_size in enumerate(sizes):
            left_offset = offsets[segment * len(sizes) + left_channel]
            for right_channel in range(left_channel + 1, len(sizes)):
                right_offset = offsets[segment * len(sizes) + right_channel]
                edges.extend((left_offset + a, right_offset + b)
                             for a in range(left_size) for b in range(sizes[right_channel]))
    for segment in range(1, n_segments):
        for channel, size in enumerate(sizes):
            left_offset = offsets[(segment - 1) * len(sizes) + channel]
            right_offset = offsets[segment * len(sizes) + channel]
            edges.extend((left_offset + category, right_offset + category) for category in range(size))
    edges = tuple(sorted(edges))
    quantum = QuantumRegister(sum(widths), "q")
    circuit = QuantumCircuit(quantum, ClassicalRegister(sum(widths), "meas"),
                             name=f"Q.NILM-categorical-p1-K{n_segments}")
    parameters = tuple(ParameterVector("cost_angle", sum(widths) + len(edges)))
    for offset, size in zip(offsets, widths):
        prepare_uniform_onehot(circuit, range(offset, offset + size))
    for qubit in range(sum(widths)):
        circuit.rz(parameters[qubit], qubit)
    for position, (left, right) in enumerate(edges):
        circuit.rzz(parameters[sum(widths) + position], left, right)
    for offset, size in zip(offsets, widths):
        for pair in range(size - 1):
            circuit.rxx(float(beta), offset + pair, offset + pair + 1)
            circuit.ryy(float(beta), offset + pair, offset + pair + 1)
    circuit.measure(quantum, circuit.cregs[0])
    circuit.metadata = {
        "algorithm": "Q.NILM categorical W-state/chain-XY QAOA", "qaoa_depth": 1,
        "state_counts": list(sizes), "n_segments": n_segments, "beta": float(beta),
        "parameter_rule": "field RZ then sorted-edge RZZ angles; coefficient-only normalized cost",
        "register_order": "time-major, channel-major; category r is offset+r",
        "invalid_shot_policy": "report all infeasible shots; never silently repair or use classical fallback",
    }
    return CategoricalCircuitTemplate(circuit, sizes, int(n_segments), float(beta), parameters, edges)


def categorical_template_parameter_values(template, problem, gamma: float, *, parameter_order=None):
    """Return cost values in template order or a compiled circuit's order.

    Pass ``tuple(prepared.parameters)`` as parameter_order for an IBM Sampler
    parameterized PUB. Exact unique parameter names also support loading a QPY
    checkpoint and reconstructing the semantic template in a later process.
    """
    _validate_encoding(problem)
    if (tuple(problem.state_counts) != template.state_counts
            or len(problem.aggregate) != template.n_segments or not np.isfinite(gamma)):
        raise ValueError("Problem topology/gamma does not match the categorical template")
    edge_set = set(template.edges)
    if any(edge not in edge_set for edge in problem.quadratic):
        raise ValueError("Problem has a quadratic edge outside the template topology")
    fields = -np.asarray(problem.linear, dtype=float) / 2
    for (left, right), value in problem.quadratic.items():
        fields[left] -= value / 4
        fields[right] -= value / 4
    values = np.concatenate((2 * gamma * fields / problem.scale,
                             np.array([gamma * problem.quadratic.get(edge, 0.0) / (2 * problem.scale)
                                       for edge in template.edges], dtype=float)))
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite template cost rotation")
    if parameter_order is None:
        return values
    mapping = dict(zip(template.parameters, values))
    by_name = {str(parameter): value for parameter, value in mapping.items()}
    try:
        return np.array([mapping[parameter] if parameter in mapping else by_name[str(parameter)]
                         for parameter in parameter_order], dtype=float)
    except KeyError as error:
        raise ValueError("Compiled circuit contains an unknown non-cost parameter") from error


def bind_categorical_template(template, problem, gamma: float, *, compiled_circuit=None):
    """Bind a template or its compiled version, with no second transpilation."""
    circuit = template.circuit if compiled_circuit is None else compiled_circuit
    metadata = circuit.metadata or {}
    if (metadata.get("state_counts") != list(template.state_counts)
            or metadata.get("n_segments") != template.n_segments
            or metadata.get("beta") != template.beta):
        raise ValueError("Compiled template metadata does not match topology/frozen beta")
    order = tuple(circuit.parameters)
    values = categorical_template_parameter_values(template, problem, gamma, parameter_order=order)
    bound = circuit.assign_parameters(dict(zip(order, values)), inplace=False)
    if bound.num_parameters:
        raise ValueError("Binding left unresolved circuit parameters")
    return bound


def decode_categorical_counts(counts: Mapping[str, int], state_counts, n_segments: int) -> dict:
    """Decode every raw shot, retaining invalid patterns and their counts.

    Qiskit strings print the highest classical bit first. Reversal before
    register slicing is therefore mandatory. A shot is feasible only when
    every appliance/segment register has exactly one 1.
    """
    sizes = tuple(state_counts)
    if (not sizes or any(isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral)
                         or size < 1 for size in sizes)
            or isinstance(n_segments, (bool, np.bool_)) or not isinstance(n_segments, Integral)
            or n_segments < 1):
        raise ValueError("Positive integer state counts and segment count are required")
    if not isinstance(counts, Mapping) or not counts:
        raise ValueError("counts must be a nonempty mapping")
    widths = sizes * n_segments
    offsets = np.cumsum((0,) + widths[:-1]).tolist()
    n_qubits = sum(widths)
    feasible = []
    invalid = {}
    for bitstring, count in sorted(counts.items(), key=lambda item: str(item[0])):
        if (not isinstance(bitstring, str) or len(bitstring) != n_qubits
                or any(bit not in "01" for bit in bitstring)):
            raise ValueError(f"Count keys must have exactly {n_qubits} binary digits")
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count <= 0:
            raise ValueError("Each shot count must be a positive integer")
        low_first = bitstring[::-1]
        registers = [low_first[offset:offset + width] for offset, width in zip(offsets, widths)]
        if any(register.count("1") != 1 for register in registers):
            invalid[bitstring] = int(count)
            continue
        categories = [register.index("1") for register in registers]
        state_id, stride = 0, 1
        for category, width in zip(categories, widths):
            state_id += category * stride
            stride *= width
        feasible.append({"bitstring": bitstring, "count": int(count),
                         "feasible_basis_index": int(state_id),
                         "states": np.array(categories).reshape(n_segments, len(sizes)).tolist()})
    total = sum(int(count) for count in counts.values())
    feasible_shots = sum(record["count"] for record in feasible)
    return {"total_shots": total, "feasible_shots": feasible_shots,
            "infeasible_shots": total - feasible_shots,
            "feasible_fraction": feasible_shots / total,
            "feasible_records": feasible, "invalid_counts": invalid,
            "raw_counts_are_postselected": False}


def score_categorical_counts(problem, counts: Mapping[str, int], *, atol: float = 1e-8) -> dict:
    """Select the lowest-cost feasible observed state without a fallback.

    If every shot violates one-hot constraints, the returned prediction is
    explicitly missing. The caller must record a failed quantum inference,
    not replace it with an exact/classical prediction under a quantum label.
    """
    _validate_encoding(problem)
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    result = decode_categorical_counts(counts, problem.state_counts, len(problem.aggregate))
    minimum = float(np.min(problem.energies))
    result.update({"prediction_status": "no_feasible_samples", "best_states": None,
                   "best_feasible_objective": None, "best_feasible_bitstring": None,
                   "mean_objective_conditional_on_feasibility": None,
                   "exact_objective": minimum, "exact_optimum_shots": 0,
                   "exact_optimum_probability_all_shots": 0.0,
                   "classical_fallback_used": False})
    records = result["feasible_records"]
    if not records:
        return result
    for record in records:
        record["objective"] = float(problem.energies[record["feasible_basis_index"]])
    best = min(records, key=lambda record: (record["objective"], record["feasible_basis_index"]))
    optimum_shots = sum(record["count"] for record in records
                        if np.isclose(record["objective"], minimum, rtol=0, atol=atol))
    result.update({"prediction_status": "feasible_sample_selected", "best_states": best["states"],
                   "best_feasible_objective": best["objective"],
                   "best_feasible_bitstring": best["bitstring"],
                   "mean_objective_conditional_on_feasibility":
                       sum(record["count"] * record["objective"] for record in records) / result["feasible_shots"],
                   "exact_optimum_shots": optimum_shots,
                   "exact_optimum_probability_all_shots": optimum_shots / result["total_shots"]})
    return result
