#!/usr/bin/env python3
"""Offline p=1/p=2 controls using the archived IBM Fez calibration.

There is deliberately no IBM service, account, or submission code in this file.
It reuses the corrected diagnostic pilot, not independent held-out NILM data.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from quantum_nilm.ibm import build_qaoa_circuit, counts_to_metrics
from quantum_nilm.qaoa import phase_scale_for, qaoa_probabilities
from quantum_nilm.qubo import build_binary_temporal_qubo


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    from qiskit_ibm_runtime import RuntimeEncoder

    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, cls=RuntimeEncoder, allow_nan=False)
        handle.write("\n")


def depth_angles(gamma: float, beta: float, depth: int) -> tuple[list, list]:
    """Split total rotation angles equally; this is not p=2 optimization."""
    if depth not in (1, 2) or not np.isfinite(gamma) or not np.isfinite(beta):
        raise ValueError("Expected finite pilot angles and depth 1 or 2")
    return [float(gamma / depth)] * depth, [float(beta / depth)] * depth


def compact_circuit(circuit):
    """Remove idle physical registers, preserving classical measurement indices.

    Every operation on an active physical qubit is copied without optimization.
    The returned ordered list maps compact indices back to physical indices.
    """
    from qiskit import QuantumCircuit

    active = sorted({circuit.find_bit(q).index for item in circuit.data
                     if item.operation.name != "barrier" for q in item.qubits})
    mapping = {physical: compact for compact, physical in enumerate(active)}
    compact = QuantumCircuit(len(active), circuit.num_clbits)
    compact.global_phase = circuit.global_phase
    for item in circuit.data:
        if item.operation.name == "barrier":
            continue
        compact.append(item.operation,
                       [mapping[circuit.find_bit(q).index] for q in item.qubits],
                       [circuit.find_bit(c).index for c in item.clbits])
    return compact, active


def compact_noise_dictionary(noise: dict, active: list[int]) -> dict:
    """Remap public NoiseModel serialization to the same compact registers.

    Errors referring exclusively to unused gates/qubits cannot act on this
    circuit and are dropped. Partially overlapping multi-qubit errors are also
    dropped: their corresponding gate cannot occur in the compact circuit.
    """
    if not active or len(set(active)) != len(active):
        raise ValueError("active physical indices must be nonempty and unique")
    mapping = {physical: compact for compact, physical in enumerate(active)}
    result = copy.deepcopy(noise)
    result["errors"] = []
    for error in noise.get("errors", []):
        remapped = copy.deepcopy(error)
        if "gate_qubits" in error:
            remapped["gate_qubits"] = [[mapping[q] for q in qubits]
                                        for qubits in error["gate_qubits"]
                                        if all(q in mapping for q in qubits)]
            if not remapped["gate_qubits"]:
                continue
        result["errors"].append(remapped)
    return result


def classical_order_probabilities(circuit) -> np.ndarray:
    """Exact measurement distribution, including final physical routing."""
    from qiskit.quantum_info import Statevector

    measurements = [(circuit.find_bit(item.qubits[0]).index,
                     circuit.find_bit(item.clbits[0]).index)
                    for item in circuit.data if item.operation.name == "measure"]
    if (len(measurements) != circuit.num_clbits
            or len({c for _, c in measurements}) != circuit.num_clbits):
        raise ValueError("Expected exactly one final measurement per classical bit")
    raw = Statevector.from_instruction(
        circuit.remove_final_measurements(inplace=False)).probabilities()
    states = np.arange(raw.size, dtype=np.uint64)
    measured = np.zeros(raw.size, dtype=np.uint64)
    for quantum, classical in measurements:
        measured |= ((states >> quantum) & 1) << classical
    return np.bincount(measured.astype(np.int64), weights=raw,
                       minlength=1 << circuit.num_clbits)


def circuit_stats(circuit) -> dict:
    compact, active = compact_circuit(circuit)
    return {
        "allocated_qubits": circuit.num_qubits, "active_qubits": len(active),
        "active_physical_indices": active, "depth": circuit.depth(),
        "two_qubit_gates": sum(len(item.qubits) == 2 for item in compact.data),
        "gate_counts": dict(circuit.count_ops()),
        "logical_to_final_physical": (circuit.layout.final_index_layout()
                                       if circuit.layout else None),
    }


def load_problem(source: dict):
    if source.get("aggregate_mode") != "signed":
        raise ValueError("This follow-up requires the corrected signed aggregate")
    if source.get("qaoa_depth") != 1 or not 1 <= source["qubits"] <= 12:
        raise ValueError("Expected a corrected depth-one pilot with at most 12 qubits")
    channels = source["channels"]
    powers = np.array([source["nominal_incremental_power_w"][key] for key in channels])
    penalties = np.array([source["switch_penalty"][key] for key in channels])
    aggregate = np.array(source["aggregate_segment_power_w"])
    weights = np.array(source["selected_segment_durations_blocks"])
    qubo = build_binary_temporal_qubo(aggregate, powers, penalties, weights)
    if qubo.n_variables != source["qubits"] or not np.isclose(
            phase_scale_for(qubo), source["phase_scale"], rtol=1e-12, atol=0):
        raise ValueError("Frozen pilot dimensions or phase scale do not match")
    return qubo, dict(reference_states=np.array(source["reference_states"]),
                     aggregate=aggregate, powers=powers, segment_weights=weights)


def archived_noise_model(properties):
    """Build gate/readout noise without an unscheduled delay relaxation pass.

    Fez does not archive frequency values. Aer 0.17.2's high-level properties
    constructor nevertheless requires them in its delay pass. The public device
    builders handle missing frequencies at zero temperature without inventing
    values. No delay instructions are present in this experiment.
    """
    from qiskit_aer.noise import NoiseModel
    from qiskit_aer.noise.device import basic_device_gate_errors, basic_device_readout_errors

    noise = NoiseModel(basis_gates=sorted({gate.gate for gate in properties.gates}))
    for qubits, error in basic_device_readout_errors(properties):
        noise.add_readout_error(error, qubits)
    for name, qubits, error in basic_device_gate_errors(
            properties, gate_error=True, thermal_relaxation=True, temperature=0):
        noise.add_quantum_error(error, name, qubits)
    return noise


def run(args) -> None:
    from qiskit import qasm3, qpy
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime import RuntimeDecoder
    from qiskit_ibm_runtime.models import BackendConfiguration, BackendProperties
    from qiskit_ibm_runtime.utils.backend_converter import convert_to_target

    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory; existing experiments are immutable")
    source = json.loads(args.pilot_summary.read_text())
    qubo, metric_inputs = load_problem(source)
    configuration_path = args.calibration_dir / "backend_configuration.json"
    properties_path = args.calibration_dir / "backend_properties.json"
    configuration = BackendConfiguration.from_dict(
        json.loads(configuration_path.read_text(), cls=RuntimeDecoder))
    properties = BackendProperties.from_dict(
        json.loads(properties_path.read_text(), cls=RuntimeDecoder))
    archived_plan = json.loads((args.calibration_dir / "plan.json").read_text())
    for path in (configuration_path, properties_path):
        if checksum(path) != archived_plan["files_sha256"][path.name]:
            raise ValueError("Archived calibration checksum failed")
    # Conversion is a version-pinned local runtime utility, not a service call.
    target = convert_to_target(configuration=configuration, properties=properties,
                               include_control_flow=False, include_fractional_gates=False)
    full_noise = archived_noise_model(properties)
    _, energies = qubo.energies()
    optimum_mask = np.isclose(energies, np.min(energies), rtol=0, atol=1e-8)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "input_summary.json", source)
    rows = []
    for depth in (1, 2):
        gammas, betas = depth_angles(source["gamma"], source["beta"], depth)
        logical = build_qaoa_circuit(qubo, gammas, betas, source["phase_scale"])
        expected = qaoa_probabilities(qubo, gammas, betas, source["phase_scale"])
        logical_error = float(np.max(np.abs(expected - classical_order_probabilities(logical))))
        if logical_error > 1e-12:
            raise ValueError("Logical Qiskit circuit failed the independent NumPy audit")
        for seed in args.seeds:
            directory = args.output_dir / f"p{depth}_seed{seed}"
            directory.mkdir()
            started = perf_counter()
            prepared = generate_preset_pass_manager(
                target=target, optimization_level=3, seed_transpiler=seed).run(logical)
            compilation_seconds = perf_counter() - started
            compact, active = compact_circuit(prepared)
            if "delay" in compact.count_ops():
                raise ValueError("Delay gates require a separately validated idle-noise model")
            if len(active) > 12:
                raise ValueError("More than 12 active simulation qubits; refuse excessive local memory")
            actual = classical_order_probabilities(compact)
            compiled_error = float(np.max(np.abs(expected - actual)))
            if compiled_error > 1e-6:
                raise ValueError("Compiled circuit failed the measurement-order probability audit")
            # Keep complex Kraus arrays for reconstruction; RuntimeEncoder handles
            # their separate archived JSON encoding. Aer from_dict cannot decode
            # the nested real/imaginary lists from to_dict(serializable=True).
            remapped_noise = compact_noise_dictionary(full_noise.to_dict(), active)
            noise_model = NoiseModel.from_dict(remapped_noise)
            write_json(directory / "compact_noise_model.json", remapped_noise)
            with (directory / "prepared.qpy").open("xb") as handle:
                qpy.dump(prepared, handle)
            with (directory / "prepared.qasm").open("x") as handle:
                handle.write(qasm3.dumps(prepared))
            row = {
                "qaoa_depth": depth, "transpiler_seed": seed, "simulator_seed": seed,
                "gammas": gammas, "betas": betas, "shots_per_mode": args.shots,
                "phase_scale": source["phase_scale"],
                "parameter_rule": ("frozen corrected p=1 classical noiseless grid optimum" if depth == 1
                                   else "equal split of frozen p=1 total angles; no p=2 optimization"),
                "depth_comparison_is_optimized": False,
                "compiled_circuit": circuit_stats(prepared),
                "simulation_compact_to_physical": active,
                "compilation_wall_time_s": compilation_seconds,
                "logical_probability_max_abs_error": logical_error,
                "compiled_probability_max_abs_error": compiled_error,
                "theoretical_ideal_optimum_probability": float(expected[optimum_mask].sum()),
                "theoretical_ideal_mean_objective": float(expected @ energies),
                "modes": {},
            }
            for mode, noise in (("ideal", None), ("calibration_matched_noisy", noise_model)):
                simulator = AerSimulator(method="density_matrix", noise_model=noise,
                                         max_parallel_threads=4)
                started = perf_counter()
                result = simulator.run(compact, shots=args.shots, seed_simulator=seed).result()
                elapsed = perf_counter() - started
                if not result.success:
                    raise RuntimeError("Local Aer experiment failed")
                counts = result.get_counts()
                write_json(directory / f"counts_{mode}.json", counts)
                metrics = counts_to_metrics(qubo, counts, **metric_inputs)
                metrics.update({"local_simulation_wall_time_s": elapsed,
                                "aer_metadata": result.results[0].metadata,
                                "counts_sha256": checksum(directory / f"counts_{mode}.json")})
                row["modes"][mode] = metrics
            write_json(directory / "summary.json", row)
            rows.append(row)
            print(json.dumps({"depth": depth, "seed": seed,
                              "active_qubits": len(active),
                              "ideal_p_opt": row["modes"]["ideal"]["exact_optimum_probability"],
                              "noisy_p_opt": row["modes"]["calibration_matched_noisy"]["exact_optimum_probability"]}),
                  flush=True)
    uniform_probability = float(np.mean(optimum_mask))
    summary = {
        "algorithm": "Q.NILM", "status": "completed offline simulator controls; no new QPU jobs",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim": "circuit/depth/layout/noise sensitivity on a retrospective diagnostic pilot; no quantum advantage",
        "problem_scope": "corrected selected-channel proxy pilot, not held-out whole-house NILM",
        "noise_scope": "archived Fez gate depolarization, thermal relaxation, and local readout approximation; not live hardware",
        "noise_construction": "Aer public basic_device_gate_errors and basic_device_readout_errors, temperature=0; no synthetic frequency values",
        "noise_exclusions": ["calibration drift", "correlated readout errors", "crosstalk",
                             "non-Markovian noise", "idle-time effects absent from the unscheduled circuit"],
        "archived_backend": configuration.backend_name,
        "calibration_last_update": properties.last_update_date,
        "source_sha256": {str(args.pilot_summary): checksum(args.pilot_summary),
                          str(configuration_path): checksum(configuration_path),
                          str(properties_path): checksum(properties_path)},
        "versions": {name: version(name) for name in
                     ("numpy", "qiskit", "qiskit-aer", "qiskit-ibm-runtime")},
        "uniform_optimum_probability_per_shot": uniform_probability,
        "uniform_probability_at_least_one_optimum_at_this_shot_budget":
            float(1 - (1 - uniform_probability) ** args.shots),
        "uniform_expected_objective": float(np.mean(energies)),
        "classical_angle_optimization_cost": {
            "p1": "inherited 41 x 31 exact-statevector grid (1271 objective evaluations); not rerun/timed here",
            "p2": "no optimization; algebraic split of p1 angles",
            "scalability_limitation": "exact-statevector parameter search is itself classical exponential work"},
        "rows": rows,
    }
    write_json(args.output_dir / "summary.json", summary)
    plan = {
        "status": "offline resource plan only; submission unsupported by this runner",
        "submission_supported": False, "required_instance_plan": "open",
        "paid_execution_allowed": False, "remote_account_checked": False,
        "calibration_current_at_submission": False,
        "stage_a": {"scope": "corrected pilot, two depths, three transpiler seeds",
                    "circuits": len(rows), "shots_per_circuit": args.shots,
                    "total_shots": len(rows) * args.shots,
                    "max_execution_time_s_for_single_initial_job": 60,
                    "planning_qpu_seconds_range": [18, 36],
                    "estimate_basis": "six circuits times historical 3 s usage; up to 2x allowance for deeper circuits; excludes uncertain job overhead; not an IBM estimate/guarantee"},
        "stage_b": {"scope": "30 untouched preselected windows x 2 depths x 3 layouts x 3 calibration dates",
                    "circuits": 540, "shots_per_circuit": args.shots,
                    "total_shots": 540 * args.shots,
                    "planning_qpu_seconds_range": [1620, 3240],
                    "execution_requires": "fresh allocation check and explicit campaign budget; do not assume Open Plan capacity"},
        "pre_submission_gates": [
            "Read-only check of saved account active instance plan and remaining allowance; require Open Plan",
            "Select live operational backend, archive fresh calibration, recompile and audit circuits",
            "Freeze windows, model, angles, metrics, seeds, maximum total jobs/shots and stopping rule before execution",
            "For p=2 implement and test a submission path; the existing pilot runner only accepts frozen p=1 input",
            "Reject non-Open plans; no paid override; maximum execution time 60 s for initial job",
            "Write exclusive submission record before sending; reconcile ambiguous failures; never blind resubmit",
            "Do not submit stage B until held-out model and strong classical baselines pass quality checks"],
        "compiled_resources": [{"depth": row["qaoa_depth"], "seed": row["transpiler_seed"],
                                **row["compiled_circuit"]} for row in rows],
    }
    write_json(args.output_dir / "hardware_plan.json", plan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-summary", type=Path, default=Path("results/pilot_corrected/summary.json"))
    parser.add_argument("--calibration-dir", type=Path, default=Path("results/ibm_qpu_pilot"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ibm_followup_controls/run_001"))
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    args = parser.parse_args()
    if not 1 <= args.shots <= 10000 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Require 1-10000 shots and distinct integer seeds")
    run(args)


if __name__ == "__main__":
    main()
