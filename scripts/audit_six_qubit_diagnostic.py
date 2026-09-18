#!/usr/bin/env python3
"""Independent offline audit; imports neither the diagnostic nor its runner."""
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path

import numpy as np
from qiskit import qpy
from qiskit.quantum_info import DensityMatrix, Statevector

ROOT = Path(__file__).resolve().parents[1]


def require(value, reason):
    if not value:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def circuit_at(path):
    with path.open("rb") as handle:
        circuits = qpy.load(handle)
    require(len(circuits) == 1, "Unexpected QPY contents")
    return circuits[0]


def problem_data(info):
    """Reconstruct all assignments, direct energies and coefficient-only scale."""
    levels = info["levels_w"]
    intervals, channels = len(info["aggregate_w"]), len(levels)
    sizes = [len(x) for x in levels] * intervals
    offsets = np.cumsum([0] + sizes[:-1]).tolist()
    states, indices, energies = [], [], []
    for k in range(int(np.prod(sizes))):
        digits, value = [], k
        for size in sizes:
            digits.append(value % size)
            value //= size
        state = np.array(digits).reshape(intervals, channels)
        index = sum(2 ** (offset + digit) for offset, digit in zip(offsets, digits))
        energy = 0.
        for t in range(intervals):
            predicted = sum(levels[i][state[t, i]] for i in range(channels))
            energy += info["segment_weights"][t] * (info["aggregate_w"][t] - predicted) ** 2
            if t:
                energy += sum(info["switch_penalty_w2"][i] for i in range(channels) if state[t, i] != state[t-1, i])
        states.append(state); indices.append(index); energies.append(energy)
    linear = []
    for t in range(intervals):
        for values in levels:
            linear.extend(info["segment_weights"][t] * (v*v - 2*info["aggregate_w"][t]*v) for v in values)
    quadratic_abs = 0.
    for t in range(intervals):
        for i in range(channels):
            for j in range(i+1, channels):
                quadratic_abs += sum(abs(2*info["segment_weights"][t]*a*b) for a in levels[i] for b in levels[j])
    quadratic_abs += (intervals-1)*sum(info["switch_penalty_w2"][i]*len(levels[i]) for i in range(channels))
    scale = max(1., sum(abs(x) for x in linear) + quadratic_abs)
    energies = np.array(energies)
    require(abs(min(energies) - info["minimum_objective_w2"]) < 1e-7, "Direct optimum differs")
    require(np.array_equal(states[int(np.argmin(energies))], info["known_states"]), "Known solution differs from optimum")
    return np.array(states), np.array(indices), energies, sizes, scale


def reference(info, gammas, betas, component, phase=None):
    states, indices, energies, sizes, scale = problem_data(info)
    state = np.ones(len(indices), dtype=complex) / np.sqrt(len(indices))
    for gamma, beta in zip(gammas, betas):
        if component in ("W_cost", "full"):
            state *= np.exp(-1j * gamma * energies / scale)
            if phase is not None:
                state *= np.exp(-.5j * phase * (1 - 2*((indices >> 2) & 1)))
        if component in ("W_mixer", "full"):
            stride = 1
            for size in sizes:
                for category in range(size-1):
                    for k in range(len(state)):
                        if k // stride % size == category:
                            other = k + stride
                            left, right = state[k], state[other]
                            state[k] = np.cos(beta)*left - 1j*np.sin(beta)*right
                            state[other] = -1j*np.sin(beta)*left + np.cos(beta)*right
                stride *= size
    physical = np.zeros(1 << sum(sizes))
    physical[indices] = abs(state) ** 2
    return physical / physical.sum()


def measurement_order(circuit):
    mapping = {}
    for op in circuit.data:
        if op.operation.name == "measure":
            bit = circuit.find_bit(op.clbits[0]).index
            require(bit not in mapping, "Repeated classical bit")
            mapping[bit] = circuit.find_bit(op.qubits[0]).index
    require(sorted(mapping) == list(range(circuit.num_qubits)), "Missing output bit")
    return [mapping[i] for i in range(len(mapping))]


def permute(p, mapping):
    output = np.zeros(len(p))
    for i, probability in enumerate(p):
        bits = format(i, f"0{len(mapping)}b")[::-1]
        classical = "".join(bits[q] for q in mapping)[::-1]
        output[int(classical, 2)] += probability
    return output


def assignment(p, errors):
    """Dense Kronecker assignment matrix, independent of production reshape."""
    matrix = np.array([[1.]])
    for p10, p01 in reversed(errors):
        matrix = np.kron(matrix, [[1-p10, p01], [p10, 1-p01]])
    return matrix @ p


def compare_metrics(info, p, ideal, stored):
    _, indices, energy, _, _ = problem_data(info)
    require(np.isfinite(p).all() and p.min() >= -1e-12 and abs(p.sum()-1) < 1e-8, "Probability array invalid")
    valid = sum(float(p[i]) for i in indices)
    optimum = sum(float(p[i]) for i, e in zip(indices, energy) if abs(e-min(energy)) < 1e-8)
    expectation = sum(float(p[i]*e) for i, e in zip(indices, energy))/valid if valid > 1e-15 else None
    actual = {"valid_probability": valid, "invalid_probability": max(0., 1-valid),
        "optimal_probability_per_raw_shot": optimum, "conditional_mean_objective_w2": expectation,
        "full_distribution_tv_to_own_ideal": float(sum(abs(p-ideal))/2),
        "conditional_feasible_tv_to_own_ideal": float(sum(abs(p[indices]/valid-ideal[indices]/sum(ideal[indices])))/2) if valid > 1e-15 else None,
        "no_valid_sample_probability": max(0., 1-valid)**256,
        "optimal_hit_probability": 1 - max(0., 1-optimum)**256}
    for key, value in actual.items():
        if value is None:
            require(stored[key] is None, "Invalid conditional metric")
        else:
            tolerance = 1e-6 if key.endswith("w2") else 1e-8
            require(abs(stored[key] - value) < tolerance, f"Metric differs: {key}")


def audit(folder, output):
    require(not output.exists(), "Never overwrite an audit")
    plan, summary, training = [read(folder / name) for name in ("plan.json", "summary.json", "training.json")]
    require(summary["plan_sha256"] == digest(folder/"plan.json"), "Summary plan binding differs")
    for name, expected in plan["source_sha256"].items():
        require(digest(ROOT/name) == expected, "Frozen source changed")
    for name, expected in plan["files_sha256"].items():
        require(digest(folder/name) == expected, "Frozen artifact changed")
    require(digest(ROOT/plan["snapshot_relative_path"]) == plan["snapshot_sha256"], "Snapshot changed")
    require(summary["hardware_jobs"] == summary["qpu_seconds"] == 0, "Not an offline campaign")
    require(training["evaluations"] == 1024, "Training budget differs")
    ideal_errors, noisy_errors, spec_by_id = [], [], {s["id"]:s for s in plan["circuits"]}
    require(len(spec_by_id) == 30 and len(plan["logical_phase_controls"]) == 12, "Ideal coverage differs")
    expected_rows = set()
    for spec in spec_by_id.values():
        info = plan["problems"][spec["case"]]
        expected = reference(info, spec["gammas"], spec["betas"], spec["component"])
        circuit = circuit_at(folder/f"{spec['id']}.qpy")
        mapping = measurement_order(circuit)
        require(mapping == spec["logical_measurement_to_compact"], "Measurement map differs")
        actual = permute(Statevector.from_instruction(circuit.remove_final_measurements(inplace=False)).probabilities(), mapping)
        saved = np.load(folder/f"{spec['id']}_ideal.npy", allow_pickle=False)
        error = max(float(max(abs(actual-expected))), float(max(abs(saved-expected))))
        require(error < 1e-8, "Independent ideal check failed")
        ideal_errors.append(error)
        compare_metrics(info, saved, expected, spec["ideal_metrics"])
        if spec["placement"] == "all_to_all":
            names = {"generic_combined"}
        elif spec["case"] == "six_p1":
            names = {"readout", "depolarizing", "relaxation", "calibration_combined", "coherent_z_negative",
                     "coherent_z_positive", "coherent_zz_negative", "coherent_zz_positive", "generic_combined"}
        else:
            names = {"generic_combined", "calibration_combined"}
        expected_rows.update(f"{spec['id']}__{name}" for name in names)
    for control in plan["logical_phase_controls"]:
        info = plan["problems"]["six_p1"]
        gamma, beta = training["gammas"], training["betas"]
        expected = reference(info, gamma, beta, control["component"], control["phase_rad"])
        saved = np.load(folder/f"{control['id']}.npy", allow_pickle=False)
        circuit = circuit_at(folder/f"{control['id']}.qpy")
        actual = permute(Statevector.from_instruction(circuit.remove_final_measurements(inplace=False)).probabilities(), measurement_order(circuit))
        error = max(float(max(abs(expected-saved))), float(max(abs(expected-actual))))
        require(error < 1e-8, "Logical phase control differs")
        ideal_errors.append(error)
        compare_metrics(info, saved, reference(info, gamma, beta, control["component"]), control["metrics"])
    rows = summary["rows"]
    require(len(rows) == len(expected_rows) == 106 and {r["id"] for r in rows} == expected_rows, "Noise coverage differs")
    for number, row in enumerate(rows):
        require(row == read(folder/f"{row['id']}.json"), "Summary row differs from checkpoint")
        require(row["plan_sha256"] == digest(folder/"plan.json"), "Noisy row plan binding differs")
        for name, expected in row["files_sha256"].items():
            require(digest(folder/name) == expected, "Noisy artifact changed")
        spec = spec_by_id[row["spec_id"]]
        info = plan["problems"][spec["case"]]
        measured = np.load(folder/f"{row['id']}.npy", allow_pickle=False)
        ideal = reference(info, spec["gammas"], spec["betas"], spec["component"])
        compare_metrics(info, measured, ideal, row["metrics"])
        n = info["num_qubits"]
        timing = row["timing_model"]
        require(abs(timing["summed_idle_qubit_s"] + timing["summed_active_qubit_s"] - n*timing["gate_makespan_s"]) < 1e-12,
                "ASAP idle/active accounting does not reconcile")
        if n == 6:
            noisy = circuit_at(folder/f"{row['id']}.qpy")
            p = DensityMatrix.from_instruction(noisy).probabilities()
            p = assignment(p, row["readout_errors_physical_order"])
            p = permute(p, spec["logical_measurement_to_compact"])
            error = float(max(abs(p-measured)))
            require(error < 1e-8, "Independent density evolution differs")
            noisy_errors.append(error)
        if (number+1) % 10 == 0:
            print(f"Audited {number+1}/106 noisy records", flush=True)
    receipt = {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "ideal_checks": len(ideal_errors), "noisy_metric_checks": len(rows),
        "independent_six_qubit_density_checks": len(noisy_errors),
        "max_ideal_probability_error": max(ideal_errors), "max_independent_density_probability_error": max(noisy_errors),
        "plan_sha256": digest(folder/"plan.json"), "summary_sha256": digest(folder/"summary.json"),
        "audit_code_sha256": digest(Path(__file__)),
        "limitations": ["Qiskit DensityMatrix cross-checks evolution of saved injected channels, not physical correctness of the noise model",
            "Nine- and twelve-qubit noisy distributions receive independent metric/provenance checks, not a second dense evolution",
            "No hardware execution or causal attribution of historical IBM failure"]}
    with output.open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.folder.resolve(), args.output.resolve())
