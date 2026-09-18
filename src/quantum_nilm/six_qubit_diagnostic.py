"""Offline, full-Hilbert-space diagnostics for small categorical Q.NILM.

No accounts, network requests or hardware submission capabilities. Calibration
models are approximations and coherent injections are declared sensitivity
scenarios, not fitted estimates of IBM's physical errors.
"""
from __future__ import annotations

import numpy as np

from .categorical_ibm import build_categorical_qaoa_circuit, prepare_uniform_onehot
from .categorical_qaoa import prepare_categorical_problem, categorical_qaoa_probabilities

BASIS = ("rz", "sx", "x", "cz", "measure")
MAX_QUBITS = 12
PLACEMENTS = {
    "suspect": [128, 129, 130, 131, 132, 133, 127, 126, 125, 124, 123, 122],
    "comparison": list(range(140, 152)),
}
CASES = {
    "six_p1": (2, 1, 1, 1),
    "six_p2": (2, 1, 2, 1),
    "six_p3": (2, 1, 3, 1),
    "six_cz3": (2, 1, 1, 3),
    "six_cz5": (2, 1, 1, 5),
    "nine_p1": (3, 1, 1, 1),
    "twelve_p1": (2, 2, 1, 1),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def make_problem(channels=2, intervals=1):
    require(channels in (2, 3) and intervals in (1, 2), "Unsupported diagnostic shape")
    levels = [[0., 400., 900.], [0., 120., 280.], [0., 55., 175.]][:channels]
    truth = np.array([[1, 2, 1], [2, 1, 2]], dtype=int)[:intervals, :channels]
    aggregate = [sum(levels[i][state] for i, state in enumerate(row)) for row in truth]
    problem = prepare_categorical_problem(aggregate, levels, [1600., 400., 100.][:channels])
    return problem, truth


def problem_receipt(problem, truth):
    return {"aggregate_w": problem.aggregate.tolist(), "levels_w": [x.tolist() for x in problem.levels],
            "switch_penalty_w2": problem.switch_penalty.tolist(),
            "segment_weights": problem.segment_weights.tolist(), "known_states": truth.tolist(),
            "num_qubits": problem.num_qubits, "num_feasible_states": problem.num_feasible_states,
            "scope": "Synthetic controlled sum of only the modeled loads; not whole-house R1Hz accuracy"}


def independent_objective_check(problem):
    """Direct state-by-state arithmetic, including the full QUBO constant."""
    largest = 0.
    for index, states in enumerate(problem.states):
        measured_cost = 0.
        bits = np.zeros(problem.num_qubits)
        for t, row in enumerate(states):
            prediction = sum(float(problem.levels[i][s]) for i, s in enumerate(row))
            measured_cost += float(problem.segment_weights[t]) * (float(problem.aggregate[t]) - prediction) ** 2
            if t:
                measured_cost += sum(float(problem.switch_penalty[i]) for i, s in enumerate(row) if s != states[t - 1, i])
            for i, s in enumerate(row):
                bits[problem.register_offsets[t * len(problem.levels) + i] + s] = 1
        qubo_cost = problem.constant + sum(float(a * b) for a, b in zip(problem.linear, bits))
        qubo_cost += sum(float(value * bits[a] * bits[b]) for (a, b), value in problem.quadratic.items())
        largest = max(largest, abs(measured_cost - float(problem.energies[index])), abs(qubo_cost - measured_cost))
    require(largest < 1e-7, "Direct objective / QUBO mismatch")
    return {"states_checked": problem.num_feasible_states, "max_absolute_error_w2": largest,
            "minimum_objective_w2": float(min(problem.energies))}


def feasible_indices(problem):
    # Independent integer construction; no bitstring decoder or encoder call.
    return np.array([sum(1 << (offset + int(s)) for offset, s in zip(problem.register_offsets, state.ravel()))
                     for state in problem.states], dtype=np.int64)


def logical_circuit(problem, gammas, betas, component="full", phase=None):
    """Same ordered cost/XY sequence; optional diagonal phase at cost end."""
    from qiskit import QuantumCircuit
    require(component in ("W", "W_cost", "W_mixer", "full"), "Unknown component")
    require(len(gammas) == len(betas) and len(gammas) > 0, "Angle dimensions differ")
    circuit = QuantumCircuit(problem.num_qubits, problem.num_qubits)
    for offset, size in zip(problem.register_offsets, problem.register_sizes):
        prepare_uniform_onehot(circuit, range(offset, offset + size))
    fields = -problem.linear.copy() / 2
    for (a, b), coefficient in problem.quadratic.items():
        fields[a] -= coefficient / 4
        fields[b] -= coefficient / 4
    for gamma, beta in zip(gammas, betas):
        if component in ("W_cost", "full"):
            for q, field in enumerate(fields):
                if field:
                    circuit.rz(float(2 * gamma * field / problem.scale), q)
            for (a, b), coefficient in sorted(problem.quadratic.items()):
                if coefficient:
                    circuit.rzz(float(gamma * coefficient / (2 * problem.scale)), a, b)
            if phase is not None:
                circuit.rz(float(phase), 2)
        if component in ("W_mixer", "full"):
            for offset, size in zip(problem.register_offsets, problem.register_sizes):
                for a in range(offset, offset + size - 1):
                    circuit.rxx(float(beta), a, a + 1)
                    circuit.ryy(float(beta), a, a + 1)
    circuit.measure(range(problem.num_qubits), range(problem.num_qubits))
    return circuit


def reference_probabilities(problem, gammas, betas, component):
    values = categorical_qaoa_probabilities(problem,
        [0.] * len(gammas) if component in ("W", "W_mixer") else gammas,
        [0.] * len(betas) if component in ("W", "W_cost") else betas)
    physical = np.zeros(1 << problem.num_qubits)
    physical[feasible_indices(problem)] = values
    return physical


def load_calibration(snapshot):
    from qiskit_ibm_runtime.models import BackendConfiguration, BackendProperties
    from qiskit_ibm_runtime.utils.backend_converter import convert_to_target
    properties = BackendProperties.from_dict(snapshot["backend_properties"])
    target = convert_to_target(configuration=BackendConfiguration.from_dict(snapshot["backend_configuration"]),
                               properties=properties, include_control_flow=False, include_fractional_gates=False)
    return target, properties


def compact_target(full_target, physical):
    from copy import deepcopy
    from qiskit.transpiler import Target
    target = Target(num_qubits=len(physical), dt=full_target.dt,
                    qubit_properties=[full_target.qubit_properties[q] for q in physical])
    reverse = {q: i for i, q in enumerate(physical)}
    for name in BASIS:
        properties = {tuple(reverse[q] for q in qargs): deepcopy(info)
                      for qargs, info in full_target[name].items()
                      if qargs is not None and all(q in reverse for q in qargs)}
        require(properties, f"No native {name} instructions on selected subgraph")
        target.add_instruction(full_target.operation_from_name(name), properties)
    return target


def compile_circuit(logical, placement, full_target, fold=1):
    from qiskit import transpile
    require(logical.num_qubits <= MAX_QUBITS, "Local memory limit exceeded")
    require(fold in (1, 3, 5), "CZ fold must be positive odd prespecified factor")
    if placement == "all_to_all":
        target, physical = None, None
        compiled = transpile(logical, basis_gates=list(BASIS[:-1]), optimization_level=1, seed_transpiler=17)
    else:
        physical = PLACEMENTS[placement][:logical.num_qubits]
        target = compact_target(full_target, physical)
        compiled = transpile(logical, target=target, initial_layout=list(range(logical.num_qubits)),
                             layout_method="trivial", routing_method="sabre", optimization_level=1, seed_transpiler=17)
    require(compiled.num_qubits == logical.num_qubits, "Unexpected routing ancilla")
    if fold != 1:
        result = compiled.copy_empty_like()
        for item in compiled.data:
            for _ in range(fold if item.operation.name == "cz" else 1):
                result.append(item.operation, item.qubits, item.clbits)
        compiled = result
    output_map = measurement_map(compiled)
    return compiled, target, {"placement": placement, "physical_qubits": physical,
        "logical_measurement_to_compact": output_map, "native_gate_counts": dict(compiled.count_ops()),
        "compiled_depth": compiled.depth(), "cz_fold": fold,
        "routing": "no routing" if target is None else "restricted induced ibm_fez subgraph; fixed initial layout; SABRE seed 17",
        "actual_hardware_execution": False}


def measurement_map(circuit):
    mapping = {}
    for item in circuit.data:
        if item.operation.name == "measure":
            bit = circuit.find_bit(item.clbits[0]).index
            require(bit not in mapping, "Repeated measurement")
            mapping[bit] = circuit.find_bit(item.qubits[0]).index
    require(set(mapping) == set(range(circuit.num_qubits)), "Incomplete measurement map")
    require(len(set(mapping.values())) == circuit.num_qubits, "Non-bijective measurement")
    return [mapping[i] for i in range(circuit.num_qubits)]


def reorder_probabilities(probabilities, mapping):
    require(len(probabilities) == 1 << len(mapping), "Probability dimensions differ")
    require(sorted(mapping) == list(range(len(mapping))), "Invalid bit permutation")
    indices = np.arange(len(probabilities))
    output = sum(((indices >> physical) & 1) << bit for bit, physical in enumerate(mapping))
    return np.bincount(output, weights=probabilities, minlength=len(probabilities))


def validate_probabilities(values):
    p = np.asarray(values, dtype=float).copy()
    require(p.ndim == 1 and len(p) >= 2 and not len(p) & (len(p) - 1), "Full power-of-two distribution required")
    require(np.isfinite(p).all() and p.min() >= -1e-12, "Invalid probability")
    mass = float(p.sum())
    require(abs(mass - 1.) < 1e-8, "Probability mass changed")
    p[p < 0] = 0.
    p /= p.sum()
    return p, mass


def ideal_probabilities(circuit):
    from qiskit.quantum_info import Statevector
    p = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False)).probabilities()
    return validate_probabilities(reorder_probabilities(p, measurement_map(circuit)))[0]


def apply_readout(probabilities, errors):
    """Exact independent asymmetric assignment channel in physical-bit order."""
    p, _ = validate_probabilities(probabilities)
    require(len(errors) == len(p).bit_length() - 1, "Readout dimensions differ")
    for q, (p10, p01) in enumerate(errors):
        require(np.isfinite([p10, p01]).all() and 0 <= p10 <= 1 and 0 <= p01 <= 1, "Invalid assignment rate")
        blocks = p.reshape(-1, 2, 1 << q)
        zero, one = blocks[:, 0].copy(), blocks[:, 1].copy()
        blocks[:, 0] = (1 - p10) * zero + p01 * one
        blocks[:, 1] = p10 * zero + (1 - p01) * one
    return p


def calibration_channels(properties, physical, mode):
    from qiskit_aer.noise.device import basic_device_gate_errors
    include_depol = mode in ("depolarizing", "combined")
    include_thermal = mode in ("relaxation", "combined")
    if not (include_depol or include_thermal):
        return {}
    reverse = {q: i for i, q in enumerate(physical)}
    return {(name, tuple(reverse[q] for q in qs)): error.to_quantumchannel().to_instruction()
            for name, qs, error in basic_device_gate_errors(properties, gate_error=include_depol,
                thermal_relaxation=include_thermal, temperature=0)
            if all(q in reverse for q in qs)}


def noisy_circuit(compiled, target, properties, physical, model):
    """Schedule a gate DAG ASAP; add independent idle/gate relaxation explicitly.

    This transparent local schedule is not a reconstruction of IBM's pulse
    schedule. Gate channels use calibrated combined infidelity without adding
    a second copy of gate-time relaxation. Measurement error is applied later.
    """
    from qiskit import QuantumCircuit
    from qiskit_aer.noise import depolarizing_error, thermal_relaxation_error
    n = compiled.num_qubits
    name = model["name"]
    generic = name == "generic_combined"
    mode = model.get("calibration", "none")
    require(not generic or mode == "none", "Generic and calibration channels cannot be mixed")
    require(mode == "none" or physical is not None, "Calibration needs real physical placement")
    channels = calibration_channels(properties, physical, mode) if mode != "none" else {}
    thermal = generic or mode in ("relaxation", "combined")
    clocks = np.zeros(n)
    output = QuantumCircuit(n)
    idle_total, gate_total, operations = 0., 0., 0
    t1s = [100e-6] * n if generic else [properties.t1(q) for q in physical] if thermal else []
    t2s = [80e-6] * n if generic else [min(properties.t2(q), 2 * properties.t1(q)) for q in physical] if thermal else []
    def relaxation(q, seconds):
        if seconds > 1e-15 and thermal:
            output.append(thermal_relaxation_error(t1s[q], t2s[q], seconds).to_quantumchannel().to_instruction(), [q])
    for item in compiled.data:
        operation = item.operation
        if operation.name == "measure":
            continue
        require(operation.name in BASIS[:-1], "Unexpected native instruction")
        qs = tuple(compiled.find_bit(q).index for q in item.qubits)
        start = max(clocks[list(qs)])
        duration = (0. if operation.name == "rz" else 80e-9 if operation.name == "cz" else 40e-9) if generic or target is None else target[operation.name][qs].duration
        require(duration is not None and duration >= 0, "Missing gate duration")
        for q in qs:
            waiting = max(0., start - clocks[q])
            idle_total += waiting
            relaxation(q, waiting)
        output.append(operation, qs)
        if generic and operation.name != "rz":
            probability = .001 if len(qs) == 2 else .0001
            output.append(depolarizing_error(probability, len(qs)).to_quantumchannel().to_instruction(), qs)
        elif (operation.name, qs) in channels:
            output.append(channels[(operation.name, qs)], qs)
        if generic:
            for q in qs:
                relaxation(q, duration)
        if operation.name == "cz" and model.get("phase_kind"):
            angle = model["phase_rad"]
            if model["phase_kind"] == "zz":
                output.rzz(angle, *qs)
            elif model["phase_kind"] == "z":
                output.rz(angle, qs[-1])
            else:
                raise ValueError("Unknown coherent phase injection")
            operations += 1
        gate_total += duration * len(qs)
        clocks[list(qs)] = start + duration
    makespan = max(clocks)
    for q in range(n):
        waiting = max(0., makespan - clocks[q])
        idle_total += waiting
        relaxation(q, waiting)
    # Readout errors are applied after exact probabilities, including invalid
    # states. Do not add measurement-time T1 again to assignment calibration.
    if generic:
        errors = [(.01, .01)] * n
    elif model.get("readout"):
        errors = [(properties.qubit_property(q, "prob_meas1_prep0")[0],
                   properties.qubit_property(q, "prob_meas0_prep1")[0]) for q in physical]
    else:
        errors = [(0., 0.)] * n
    return output, errors, {"schedule": "ASAP gate-DAG model, terminal synchronized readout; not actual pulse schedule",
        "gate_makespan_s": float(makespan), "summed_idle_qubit_s": float(idle_total),
        "summed_active_qubit_s": float(gate_total), "coherent_injections": operations,
        "initialization": "ideal |0>; no modeled reset or leakage",
        "measurement_relaxation": "not separately added; assignment channel applied once"}


def simulate(noisy, errors, mapping):
    from qiskit_aer import AerSimulator
    require(noisy.num_qubits <= MAX_QUBITS, "Exact density simulation memory cap")
    circuit = noisy.copy()
    circuit.save_probabilities(list(range(circuit.num_qubits)), label="p")
    backend = AerSimulator(method="density_matrix", max_parallel_threads=4,
                           max_parallel_experiments=1, max_parallel_shots=1)
    result = backend.run(circuit, shots=None).result()
    require(result.success and result.results[0].success, "Density simulation failed")
    p, mass = validate_probabilities(result.data(0)["p"])
    p = apply_readout(p, errors)
    return validate_probabilities(reorder_probabilities(p, mapping))[0], mass


def score(problem, probabilities, ideal, shots=256):
    p, mass = validate_probabilities(probabilities)
    expected, _ = validate_probabilities(ideal)
    indices = feasible_indices(problem)
    weights = p[indices]
    valid = float(weights.sum())
    optimal = np.isclose(problem.energies, min(problem.energies), atol=1e-8, rtol=0)
    optimum_p = float(weights[optimal].sum())
    no_valid = float(max(0., 1. - valid) ** shots)
    hit = float(-np.expm1(shots * np.log1p(-optimum_p))) if optimum_p < 1 else 1.
    return {"probability_mass": mass, "valid_probability": valid,
        "invalid_probability": max(0., 1 - valid), "optimal_probability_per_raw_shot": optimum_p,
        "conditional_mean_objective_w2": float(weights @ problem.energies / valid) if valid > 1e-15 else None,
        "full_distribution_tv_to_own_ideal": float(abs(p - expected).sum() / 2),
        "conditional_feasible_tv_to_own_ideal": float(abs(weights / valid - expected[indices] / expected[indices].sum()).sum() / 2) if valid > 1e-15 else None,
        "shots_for_analytic_projection": shots, "optimal_hit_probability": hit,
        "no_valid_sample_probability": no_valid,
        "metric_scope": "Exact model probabilities; iid shot projection; no postselection, fallback or held-out MAE"}


def models():
    return [{"name": "ideal"}, {"name": "readout", "readout": True},
        {"name": "depolarizing", "calibration": "depolarizing"},
        {"name": "relaxation", "calibration": "relaxation"},
        {"name": "calibration_combined", "calibration": "combined", "readout": True},
        *[{"name": f"coherent_{kind}_{sign}", "phase_kind": kind, "phase_rad": value}
          for kind in ("z", "zz") for sign, value in (("negative", -.02), ("positive", .02))],
        {"name": "generic_combined"}]
