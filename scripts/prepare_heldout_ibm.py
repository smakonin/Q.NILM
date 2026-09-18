#!/usr/bin/env python3
"""Compile categorical IBM templates and estimate resources offline only.

No IBM service/account calls and no quantum job submission are implemented.
The synthetic aggregate probe is not a held-out NILM evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np

from quantum_nilm.categorical_ibm import (
    bind_categorical_template, build_categorical_parameterized_template,
)
from quantum_nilm.categorical_qaoa import prepare_categorical_problem


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    from qiskit_ibm_runtime import RuntimeEncoder

    with path.open("x") as handle:
        json.dump(value, handle, cls=RuntimeEncoder, indent=2, allow_nan=False)
        handle.write("\n")


def audit_live_templates(folder, output):
    """Audit every feasible probability after physical routing, offline only."""
    from qiskit import QuantumCircuit, qpy
    from qiskit_aer import AerSimulator
    from quantum_nilm.categorical_qaoa import categorical_qaoa_probabilities

    root = folder.parent
    if output.exists():
        raise FileExistsError("Audit output exists; choose an unused file")
    model = json.loads((root / "model.json").read_text())
    trained = model["models"]["multistate"]
    levels = [np.array(values, dtype=float) for values in trained["levels_w"]]
    rho = model["selected_parameters"]["multistate_compressed"]["rho"]
    penalties = rho * np.array(trained["ranges_w"]) ** 2
    angles = json.loads((root / "angles.json").read_text())
    windows = json.loads((root / "test_inputs.json").read_text())
    sources = [root / name for name in ("model.json", "angles.json", "test_inputs.json")]
    sources += [folder / name for name in ("plan.json", "backend_properties.json", "backend_configuration.json",
                                           "template_1.qpy", "template_2.qpy", "template_1.qasm", "template_2.qasm")]
    hashes_before = {str(path): checksum(path) for path in sources}
    rows = []
    for intervals in (1, 2):
        window, chunk = next((window, chunk) for window in windows for chunk in window["chunks"]
                             if len(chunk["weights"]) == intervals)
        problem = prepare_categorical_problem(chunk["aggregate"], levels, penalties, chunk["weights"],
                                             previous_states=None)
        template = build_categorical_parameterized_template(problem.state_counts, intervals, angles["betas"][0])
        with (folder / f"template_{intervals}.qpy").open("rb") as handle:
            prepared = qpy.load(handle)[0]
        bound = bind_categorical_template(template, problem, angles["gammas"][0], compiled_circuit=prepared)
        active = sorted({bound.find_bit(qubit).index for item in bound.data
                         if item.operation.name != "barrier" for qubit in item.qubits})
        mapping = {physical: compact for compact, physical in enumerate(active)}
        compact = QuantumCircuit(len(active))
        compact.global_phase = bound.global_phase
        measured = {}
        for item in bound.data:
            if item.operation.name == "barrier":
                continue
            if item.operation.name == "measure":
                classical = bound.find_bit(item.clbits[0]).index
                if classical in measured:
                    raise ValueError("Repeated classical measurement bit")
                measured[classical] = mapping[bound.find_bit(item.qubits[0]).index]
            else:
                if item.clbits:
                    raise ValueError("Unexpected classical operation in unitary circuit audit")
                compact.append(item.operation, [mapping[bound.find_bit(q).index] for q in item.qubits])
        if set(measured) != set(range(problem.num_qubits)) or len(set(measured.values())) != problem.num_qubits:
            raise ValueError("Measurements are not one-to-one across all logical bits")
        unmeasured = sorted(set(range(len(active))) - set(measured.values()))
        indices = np.zeros(problem.num_feasible_states, dtype=np.uint64)
        for register, offset in enumerate(problem.register_offsets):
            categories = problem.states.reshape(problem.num_feasible_states, -1)[:, register]
            indices |= np.array([1 << measured[int(offset + category)] for category in categories], dtype=np.uint64)
        marginal_indices = np.empty((problem.num_feasible_states, 1 << len(unmeasured)), dtype=np.uint64)
        for alternative in range(1 << len(unmeasured)):
            ancillary_bits = sum(((alternative >> bit) & 1) << qubit for bit, qubit in enumerate(unmeasured))
            marginal_indices[:, alternative] = indices | ancillary_bits
        compact.save_matrix_product_state(label="mps")
        simulator = AerSimulator(method="matrix_product_state", max_parallel_threads=4,
                                 matrix_product_state_truncation_threshold=0.0,
                                 matrix_product_state_max_bond_dimension=None)
        started = perf_counter()
        result = simulator.run(compact, shots=1).result()
        if not result.success:
            raise RuntimeError("Local full-circuit MPS audit failed")
        data = result.data(0)
        gammas, lambdas = data["mps"]
        # Aer 0.17.2's MPS save_amplitudes can expose the internal routing
        # permutation. Its exported MPS tensors are in canonical qubit order;
        # contract these directly, cross-checked against a full statevector.
        # This avoids allocating the full 2**25 amplitude vector.
        requested = marginal_indices.reshape(-1)
        amplitudes = np.ones((len(requested), 1), dtype=complex)
        environment = np.ones((1, 1), dtype=complex)
        for site, pair in enumerate(gammas):
            factor = np.asarray(lambdas[site]) if site < len(lambdas) else np.ones(1)
            matrices = [np.asarray(matrix) * factor[None, :] for matrix in pair]
            ones = ((requested >> site) & 1).astype(bool)
            amplitudes = np.where(ones[:, None], amplitudes @ matrices[1], amplitudes @ matrices[0])
            environment = sum(matrix.conj().T @ environment @ matrix for matrix in matrices)
        actual = np.sum(np.abs(amplitudes[:, 0].reshape(marginal_indices.shape)) ** 2, axis=1)
        norm = float(np.real(environment[0, 0]))
        expected = categorical_qaoa_probabilities(problem, angles["gammas"], angles["betas"])
        error = float(np.max(np.abs(actual - expected)))
        feasible_mass = float(np.sum(actual))
        residual = norm - feasible_mass
        if error > 1e-7 or abs(norm - 1) > 1e-8 or abs(residual) > 1e-7:
            raise ValueError(f"Compiled circuit audit exceeded tolerance: error={error}, norm={norm}, residual={residual}")
        row = {"intervals": intervals, "window_id": window["window"]["id"], "chunk_index": chunk["chunk"],
               "aggregate": chunk["aggregate"], "weights": chunk["weights"], "previous_states": None,
               "boundary_note": "No preceding state for this algebraic template audit; not a reconstructed hardware trajectory",
               "active_qubits": len(active), "logical_measured_qubits": problem.num_qubits,
               "feasible_states": problem.num_feasible_states,
               "active_compact_to_physical": active,
               "logical_measurement_bit_to_compact_qubit": [measured[bit] for bit in range(problem.num_qubits)],
               "unmeasured_routing_ancillas_compact": unmeasured,
               "all_unmeasured_ancilla_assignments_marginalized": True,
               "mps_total_norm": norm, "feasible_probability_mass": feasible_mass,
               "infeasible_mass_signed_roundoff_residual": residual,
               "infeasible_mass_nonnegative": max(0., residual),
               "max_absolute_feasible_probability_error": error,
               "total_variation_distance_including_residual": float((np.sum(np.abs(actual - expected)) + max(0., residual)) / 2),
               "local_audit_wall_time_s": perf_counter() - started,
               "mps_truncation_threshold": 0., "mps_max_bond_dimension_limit": None,
               "amplitude_extraction": "direct canonical exported-MPS tensor contraction; no save_amplitudes routing-order ambiguity",
               "maximum_observed_bond_dimension": max(len(value) for value in lambdas),
               "aer_metadata": result.results[0].metadata}
        rows.append(row)
        print(json.dumps({key: row[key] for key in ("intervals", "active_qubits", "feasible_probability_mass",
                                                    "max_absolute_feasible_probability_error", "local_audit_wall_time_s")}), flush=True)
    hashes_after = {str(path): checksum(path) for path in sources}
    if hashes_after != hashes_before:
        raise ValueError("Audit inputs changed while the audit was running")
    payload = {"status": "passed; full compiled ideal probability audit; no hardware/account/network calls",
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "hardware_folder": str(folder), "source_sha256": hashes_before,
               "source_hashes_unchanged_during_audit": True,
               "gammas": angles["gammas"], "betas": angles["betas"], "rows": rows,
               "note": "Input aggregates/durations only; no test labels or post-test parameter tuning. QPY templates were bound but not recompiled.",
               "versions": {name: version(name) for name in ("numpy", "qiskit", "qiskit-aer")}}
    write_json(output, payload)


def run(args):
    from qiskit import qasm3, qpy
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import RuntimeDecoder
    from qiskit_ibm_runtime.models import BackendConfiguration, BackendProperties
    from qiskit_ibm_runtime.utils.backend_converter import convert_to_target

    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory; prepared resources are immutable")
    model = json.loads(args.frozen_models.read_text())
    trained = model["models"]["multistate"]
    levels = [np.array(values, dtype=float) for values in trained["levels_w"]]
    state_counts = tuple(len(values) for values in levels)
    rho = model["selected_parameters"]["multistate_compressed"]["rho"]
    penalties = rho * np.array(trained["ranges_w"]) ** 2
    rows = json.loads(args.test_windows.read_text())
    windows = [row for row in rows if row["model"] == "multistate_compressed" and row["scope"] == "mains"]
    if not windows or len({row["window_id"] for row in windows}) != len(windows):
        raise ValueError("Expected distinct frozen multistate-compressed mains windows")
    chunks = {1: sum(row["segments"] % 2 for row in windows),
              2: sum(row["segments"] // 2 for row in windows)}
    stages = max(ceil(row["segments"] / 2) for row in windows)
    configuration_path = args.calibration_dir / "backend_configuration.json"
    properties_path = args.calibration_dir / "backend_properties.json"
    saved_plan = json.loads((args.calibration_dir / "plan.json").read_text())
    for path in (configuration_path, properties_path):
        if checksum(path) != saved_plan["files_sha256"][path.name]:
            raise ValueError("Cached backend configuration/properties checksum mismatch")
    configuration = BackendConfiguration.from_dict(json.loads(configuration_path.read_text(), cls=RuntimeDecoder))
    properties = BackendProperties.from_dict(json.loads(properties_path.read_text(), cls=RuntimeDecoder))
    target = convert_to_target(configuration=configuration, properties=properties,
                               include_control_flow=False, include_fractional_gates=False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    resources = []
    for intervals in (1, 2):
        template = build_categorical_parameterized_template(state_counts, intervals, args.beta)
        started = perf_counter()
        compiled = generate_preset_pass_manager(target=target, optimization_level=3,
                                                seed_transpiler=args.seed).run(template.circuit)
        compile_time = perf_counter() - started
        # Aggregate/weights are declared synthetic structural probes, not data
        # chosen to improve the held-out result or determine trained angles.
        problem = prepare_categorical_problem([600., 800.][:intervals], levels, penalties,
                                             [10., 20.][:intervals], previous_states=np.zeros(len(levels), dtype=int))
        bound = bind_categorical_template(template, problem, args.gamma, compiled_circuit=compiled)
        duration = float(bound.estimate_duration(target, unit="s"))
        active = sorted({bound.find_bit(qubit).index for item in bound.data for qubit in item.qubits})
        prefix = args.output_dir / f"k{intervals}"
        with prefix.with_suffix(".qpy").open("xb") as handle:
            qpy.dump(compiled, handle)
        with prefix.with_suffix(".qasm").open("x") as handle:
            handle.write(qasm3.dumps(compiled))
        record = {"intervals": intervals, "logical_qubits": problem.num_qubits,
                  "feasible_states": problem.num_feasible_states,
                  "allocated_qubits": compiled.num_qubits, "active_qubits": len(active),
                  "active_physical_indices": active,
                  "logical_to_final_physical": compiled.layout.final_index_layout(),
                  "gate_counts": dict(compiled.count_ops()), "depth": compiled.depth(),
                  "cost_parameters": compiled.num_parameters,
                  "compiled_parameter_order": [str(parameter) for parameter in compiled.parameters],
                  "compile_wall_time_s": compile_time,
                  "critical_path_duration_s": duration,
                  "default_rep_delay_s": configuration.default_rep_delay,
                  "shot_duration_plus_reset_s": duration + configuration.default_rep_delay,
                  "campaign_circuits_of_this_width": chunks[intervals]}
        resources.append(record)
    historical = json.loads((args.calibration_dir / "summary.json").read_text())
    timing = historical["job_metrics"]
    overhead = max(0., timing["usage"]["qpu_charge_time_seconds"] - timing["circuits_execution_time_ns"] / 1e9)
    estimates = []
    for shots in sorted(set((256, 512, args.shots))):
        execution = sum(record["campaign_circuits_of_this_width"] * record["shot_duration_plus_reset_s"] * shots
                        for record in resources)
        estimate = execution + stages * overhead
        estimates.append({"shots_per_chunk": shots, "total_shots": sum(chunks.values()) * shots,
                          "estimated_circuit_and_reset_seconds": execution,
                          "assumed_per_stage_accounting_overhead_s": overhead,
                          "estimated_stage_overhead_s": stages * overhead,
                          "planning_total_s": estimate,
                          "planning_total_with_25_percent_margin_s": estimate * 1.25,
                          "under_540s_with_margin": estimate * 1.25 <= 540,
                          "is_provider_estimate_or_guarantee": False})
    summary = {
        "status": "offline parameterized template preparation only; no submitted jobs",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": configuration.backend_name, "calibration_last_update": properties.last_update_date,
        "source_sha256": {str(path): checksum(path) for path in
                          (args.frozen_models, args.test_windows, configuration_path, properties_path)},
        "synthetic_probe": {"aggregate_w": [600., 800.], "weights": [10., 20.],
                            "previous_states": [0] * len(levels),
                            "is_heldout_evaluation": False},
        "gamma": args.gamma, "beta": args.beta,
        "angle_status": "user-supplied resource probe; not evidence of train-only selection",
        "seed_transpiler": args.seed,
        "schedule": {"test_days": len(windows), "blocks": sum(row["blocks"] for row in windows),
                     "compressed_intervals": sum(row["segments"] for row in windows),
                     "one_interval_chunks": chunks[1], "two_interval_chunks": chunks[2],
                     "total_circuits": sum(chunks.values()), "sequential_adaptive_stages": stages},
        "resources": resources, "usage_planning_estimates": estimates,
        "timing_method": "Qiskit estimate_duration critical path + decoded cached default_rep_delay; historical accounted-minus-circuit usage per adaptive stage",
        "limitations": ["Not a server QPU usage quote or live-allowance check",
                        "No queue/client, circuit-loading or variable stage overhead model",
                        "Cached calibration and probe angles may differ from final frozen execution",
                        "A stage depends on each method's own preceding measured state, so all chunks cannot be submitted independently",
                        "Readout/gate errors can violate one-hot constraints; raw invalid counts and failure handling must be reported"],
        "submission_supported": False,
        "versions": {name: version(name) for name in ("numpy", "qiskit", "qiskit-ibm-runtime")},
    }
    write_json(args.output_dir / "resources.json", summary)
    print(json.dumps({"schedule": summary["schedule"], "estimates": estimates,
                      "resources": [{key: row[key] for key in
                                     ("intervals", "active_qubits", "gate_counts", "depth", "cost_parameters")}
                                    for row in resources]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-models", type=Path, default=Path("results/heldout_campaign/frozen_models.json"))
    parser.add_argument("--test-windows", type=Path, default=Path("results/heldout_campaign/test_windows.json"))
    parser.add_argument("--calibration-dir", type=Path, default=Path("results/ibm_corrected_qpu_check"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/categorical_ibm_preparation"))
    parser.add_argument("--gamma", type=float, default=1.)
    parser.add_argument("--beta", type=float, default=.3)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--audit-live-folder", type=Path)
    parser.add_argument("--audit-output", type=Path, default=Path("results/categorical_ibm_preparation/live_template_audit.json"))
    args = parser.parse_args()
    if not np.isfinite(args.gamma) or not np.isfinite(args.beta) or not 1 <= args.shots <= 10000:
        parser.error("Require finite angles and 1-10000 shots")
    if args.audit_live_folder:
        audit_live_templates(args.audit_live_folder, args.audit_output)
    else:
        run(args)


if __name__ == "__main__":
    main()
