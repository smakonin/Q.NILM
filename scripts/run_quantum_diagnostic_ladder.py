#!/usr/bin/env python3
"""Immutable, one-job, free-only categorical circuit diagnostic ladder.

Preparation is offline. Submission needs --allow-free-submission; collection
only retrieves the recorded job. Unknown submission outcomes never retry.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm3, qpy
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import RuntimeDecoder, RuntimeEncoder, SamplerV2
from qiskit_ibm_runtime.models import BackendConfiguration, BackendProperties
from qiskit_ibm_runtime.utils.backend_converter import convert_to_target

from quantum_nilm.categorical_ibm import (build_categorical_parameterized_template,
    categorical_template_parameter_values, score_categorical_counts)
from quantum_nilm.categorical_qaoa import prepare_categorical_problem
from run_quantum_heldout_ibm import account, accounted_usage, circuit_info, remaining, utc
from run_quantum_heldout import model_arrays, sha


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/quantum_heldout"
CONDITIONS = ("W", "W_mixer", "W_cost", "full")
SHOTS, CAP, RESERVE = 1024, 60, 20


def read(path):
    return json.loads(Path(path).read_text(), cls=RuntimeDecoder)


def save(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, cls=RuntimeEncoder, indent=2, allow_nan=False)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def component_circuit(template, condition):
    """Keep phase barriers, including a barrier before terminal measurement.

    This is a new diagnostic circuit family, not a matched-duration ablation.
    Cost parameters must survive transpilation; prepare fails otherwise.
    """
    require(condition in CONDITIONS, "Unknown ladder condition")
    n = template.circuit.num_qubits
    circuit = QuantumCircuit(QuantumRegister(n, "q"), ClassicalRegister(n, "meas"),
                             name=f"ladder_K{template.n_segments}_{condition}")
    phases = [{"x", "cry", "cx"}]
    if condition in ("W_cost", "full"):
        phases.append({"rz", "rzz"})
    if condition in ("W_mixer", "full"):
        phases.append({"rxx", "ryy"})
    for names in phases:
        for item in template.circuit.data:
            if item.operation.name in names:
                circuit.append(item.operation, [template.circuit.find_bit(q).index for q in item.qubits])
        circuit.barrier()
    circuit.measure(range(n), range(n))
    circuit.metadata = {**template.circuit.metadata, "diagnostic_condition": condition,
                        "phase_barriers": True, "no_fallback": True}
    return circuit


def prepare(folder):
    require(not folder.exists(), "Output folder already exists; no overwrite")
    source_hashes = read(SOURCE / "input_hashes.json")
    for name in ("model.json", "training_chunks.json"):
        require(sha(SOURCE / name) == source_hashes[name], "Frozen training/model source changed")
    original_plan = read(SOURCE / "hardware/plan.json")
    for name, checksum in original_plan["files_sha256"].items():
        require(sha(SOURCE / "hardware" / name) == checksum, "Original prepared hardware artifact changed")
    angles = read(SOURCE / "angles.json")
    require(sha(SOURCE / "angles.json") == original_plan["angles_sha256"], "Frozen angles changed")
    examples = sorted(read(SOURCE / "training_chunks.json"), key=lambda x: (x["window_id"], x["chunk"]))[:3]
    require(len(examples) == 3 and all(x["window_id"].startswith("train-") and len(x["weights"]) == 2 for x in examples),
            "Need three frozen two-interval training chunks")
    levels, penalties, _ = model_arrays(read(SOURCE / "model.json"))
    configuration = BackendConfiguration.from_dict(read(SOURCE / "hardware/backend_configuration.json"))
    properties = BackendProperties.from_dict(read(SOURCE / "hardware/backend_properties.json"))
    target = convert_to_target(configuration=configuration, properties=properties,
                               include_control_flow=False, include_fractional_gates=False)
    folder.mkdir(parents=True)
    save(folder / "examples.json", examples)
    templates, circuits, resources = {}, {}, {}
    for width in (1, 2):
        template = build_categorical_parameterized_template([len(x) for x in levels], width, angles["betas"][0])
        templates[width] = template
        with (SOURCE / f"hardware/template_{width}.qpy").open("rb") as handle:
            original = qpy.load(handle)[0]
        layout = original.layout.initial_index_layout(filter_ancillas=True)
        for condition in CONDITIONS:
            key = f"K{width}_{condition}"
            logical = component_circuit(template, condition)
            circuit = generate_preset_pass_manager(target=target, initial_layout=layout,
                optimization_level=3, approximation_degree=1.0, seed_transpiler=7).run(logical)
            require(set(map(str, circuit.parameters)) == set(map(str, logical.parameters)),
                    "Compiler eliminated cost parameters; component comparison is unsafe")
            mappings = [(circuit.find_bit(i.clbits[0]).index, circuit.find_bit(i.qubits[0]).index)
                        for i in circuit.data if i.operation.name == "measure"]
            require(sorted(c for c, _ in mappings) == list(range(sum(template.state_counts) * width)),
                    "Incomplete measurement mapping")
            with (folder / f"{key}.qpy").open("xb") as handle:
                qpy.dump(circuit, handle)
            with (folder / f"{key}.qasm").open("x") as handle:
                handle.write(qasm3.dumps(circuit))
            circuits[key] = circuit
            probe = prepare_categorical_problem(examples[0]["aggregate"][:width], levels, penalties,
                examples[0]["weights"][:width], previous_states=None)
            values = categorical_template_parameter_values(template, probe, angles["gammas"][0],
                parameter_order=tuple(circuit.parameters)) if circuit.num_parameters else []
            numeric = circuit.assign_parameters(values)
            resources[key] = {"width": width, "condition": condition, "initial_layout": layout,
                "logical_gate_counts": dict(logical.count_ops()), "statistics": circuit_info(circuit),
                "parameter_order": list(map(str, circuit.parameters)),
                "logical_measurement_bit_to_physical": [p for _, p in sorted(mappings)],
                "duration_s": float(numeric.estimate_duration(target, unit="s")),
                "cost_parameters_retained": circuit.num_parameters if condition in ("W_cost", "full") else None}
    pubs = []
    for example_index, example in enumerate(examples):
        for width in (1, 2):
            problem = prepare_categorical_problem(example["aggregate"][:width], levels, penalties,
                                                  example["weights"][:width], previous_states=None)
            for condition in CONDITIONS:
                key = f"K{width}_{condition}"
                circuit = circuits[key]
                values = categorical_template_parameter_values(templates[width], problem, angles["gammas"][0],
                    parameter_order=tuple(circuit.parameters)).tolist() if circuit.num_parameters else []
                pubs.append({"example_index": example_index, "template": key, "width": width,
                             "condition": condition, "parameter_values": values})
    order = np.random.default_rng(1709).permutation(len(pubs)).tolist()
    pubs = [{"pub_index": i, **pubs[j]} for i, j in enumerate(order)]
    save(folder / "pubs.json", pubs)
    sources = [SOURCE / name for name in ("model.json", "training_chunks.json", "angles.json", "input_hashes.json",
        "hardware/plan.json", "hardware/backend_properties.json", "hardware/backend_configuration.json",
        "hardware/template_1.qpy", "hardware/template_2.qpy")]
    codes = [Path(__file__), ROOT / "src/quantum_nilm/categorical_ibm.py", ROOT / "src/quantum_nilm/categorical_qaoa.py",
             ROOT / "scripts/run_quantum_heldout_ibm.py", ROOT / "scripts/run_quantum_heldout.py"]
    estimate = 2 + SHOTS * sum(resources[p["template"]]["duration_s"] + original_plan["rep_delay_s"] for p in pubs)
    require(estimate < CAP - 10, "Conservative one-job estimate does not fit cap")
    plan = {"created_utc": utc(), "backend": "ibm_fez", "instance_plan": "open", "paid_allowed": False,
        "shots_per_circuit": SHOTS, "circuits": len(pubs), "total_shots": SHOTS * len(pubs),
        "campaign_cap_s": CAP, "free_reserve_s": RESERVE, "max_execution_time_s": CAP - 10,
        "estimated_accounted_s": estimate, "rep_delay_s": original_plan["rep_delay_s"],
        "conditions": list(CONDITIONS), "resources": resources, "gammas": angles["gammas"], "betas": angles["betas"],
        "selection": "first3 sorted frozen training chunks; K1 uses first interval; standalone previous_states=None",
        "order": "seed1709 fixed random permutation of24 PUBs; W/W_mixer repeat across three input examples",
        "compiler": "archived target, original initial placements, symbolic costs, phase barriers, opt3 seed7 approximation_degree1.0",
        "options": {"max_execution_time": CAP - 10, "default_shots": SHOTS,
            "dynamical_decoupling": {"enable": False}, "twirling": {"enable_gates": False, "enable_measure": False},
            "execution": {"rep_delay": original_plan["rep_delay_s"], "init_qubits": True},
            "environment": {"job_tags": ["qnilm", "training-diagnostic-ladder"]}},
        "no_fallback": True, "raw_counts_postselected": False,
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in codes},
        "files_sha256": {p.name: sha(p) for p in folder.iterdir() if p.is_file()},
        "limitations": ["Same initial placement is not matched final routing, gate count, duration or noise",
            "W+cost is ideally indistinguishable from W in measurement probabilities; barriers retain the physical cost burden",
            "New barrier-separated diagnostic family, not a rerun of the historical compiled circuits",
            "Three PUB executions per condition in one job; not independent date/layout replicas; no NILM accuracy or quantum advantage claim",
            "max_execution_time excludes some accounting overhead; reported usage checked against60s; no further jobs"]}
    save(folder / "plan.json", plan)
    print(json.dumps({"prepared": str(folder), "circuits": len(pubs), "estimated_accounted_s": estimate}), flush=True)


def check(folder):
    plan = read(folder / "plan.json")
    require(plan["instance_plan"] == "open" and plan["paid_allowed"] is False and plan["campaign_cap_s"] == CAP
            and plan["free_reserve_s"] == RESERVE and plan["shots_per_circuit"] == SHOTS and plan["circuits"] == 24
            and plan["total_shots"] == 24 * SHOTS and plan["max_execution_time_s"] == CAP - 10,
            "Diagnostic budget or protocol changed")
    require(plan["options"]["max_execution_time"] == CAP - 10
            and plan["options"]["default_shots"] == SHOTS
            and plan["options"]["dynamical_decoupling"]["enable"] is False
            and plan["options"]["twirling"] == {"enable_gates": False, "enable_measure": False}
            and plan["options"]["execution"] == {"rep_delay": plan["rep_delay_s"], "init_qubits": True},
            "Frozen execution options differ")
    for group in ("source_sha256", "code_sha256"):
        for name, checksum in plan[group].items():
            path = (ROOT / name).resolve()
            require(path.is_relative_to(ROOT) and sha(path) == checksum, "Frozen source/code checksum mismatch")
    for name, checksum in plan["files_sha256"].items():
        require(Path(name).name == name and sha(folder / name) == checksum, "Prepared artifact checksum mismatch")
    pubs = read(folder / "pubs.json")
    expected = {(example, width, condition) for example in range(3) for width in (1, 2) for condition in CONDITIONS}
    require(len(pubs) == 24 and {(p["example_index"], p["width"], p["condition"]) for p in pubs} == expected
            and [p["pub_index"] for p in pubs] == list(range(24)), "Actual PUB schedule differs")
    require(set(plan["resources"]) == {f"K{width}_{condition}" for width in (1, 2) for condition in CONDITIONS},
            "Actual template set differs")
    for key, resource in plan["resources"].items():
        with (folder / f"{key}.qpy").open("rb") as handle:
            circuit = qpy.load(handle)[0]
        require(list(map(str, circuit.parameters)) == resource["parameter_order"], "QPY parameter order differs")
        for pub in [p for p in pubs if p["template"] == key]:
            require(key == f"K{pub['width']}_{pub['condition']}" and len(pub["parameter_values"]) == circuit.num_parameters
                    and np.all(np.isfinite(pub["parameter_values"])), "Actual PUB parameters/topology differ")
    require(all(p["template"] in plan["resources"] for p in pubs), "PUB uses unknown template")
    return plan


def submit(folder, allowed):
    require(allowed, "Submission requires explicit --allow-free-submission")
    plan = check(folder)
    require(not (folder / "STOP").exists(), "STOP marker present")
    require(not (folder / "intent.json").exists(), "Prior intent exists; collect known job only; never resubmit")
    service = account()
    allowance = remaining(service)
    require(allowance >= CAP + RESERVE, "Free allowance does not cover cap plus reserve")
    backend = service.backend(plan["backend"])
    require(backend.status().operational, "Backend is not operational")
    # Execution-era public calibration is evidence only; never recompile or
    # substitute it for the already frozen archived-target circuits.
    save(folder / "submission_calibration.json", {"retrieved_utc": utc(), "backend": plan["backend"],
        "plan_sha256": sha(folder / "plan.json"), "backend_properties": backend.properties().to_dict(),
        "backend_configuration": backend.configuration().to_dict()})
    circuits = {}
    for key in plan["resources"]:
        with (folder / f"{key}.qpy").open("rb") as handle:
            circuits[key] = qpy.load(handle)[0]
    pubs = [(circuits[p["template"]], p["parameter_values"]) for p in read(folder / "pubs.json")]
    # Immutable intent is created before the only submission call. If receipt
    # saving fails or submit raises, there is no automatic retry or new job.
    require(not (folder / "STOP").exists(), "STOP marker present before submission")
    save(folder / "intent.json", {"created_utc": utc(), "plan_sha256": sha(folder / "plan.json"),
        "pubs_sha256": sha(folder / "pubs.json"), "free_allowance_before_s": allowance, "options": plan["options"],
        "submission_calibration_sha256": sha(folder / "submission_calibration.json"),
        "circuits": 24, "shots_per_circuit": SHOTS, "submission_state": "started; receipt required to recover"})
    job = SamplerV2(mode=backend, options=plan["options"]).run(pubs, shots=SHOTS)
    save(folder / "job.json", {"job_id": job.job_id(), "submitted_utc": utc(),
        "plan_sha256": sha(folder / "plan.json"), "intent_sha256": sha(folder / "intent.json")})
    print(json.dumps({"submitted_job_id": job.job_id(), "collect_separately": True}), flush=True)


def collect(folder):
    plan = check(folder)
    require(not (folder / "result.json").exists(), "Result already exists; audit it without overwriting")
    receipt, intent = read(folder / "job.json"), read(folder / "intent.json")
    require(receipt["plan_sha256"] == intent["plan_sha256"] == sha(folder / "plan.json")
            and receipt["intent_sha256"] == sha(folder / "intent.json")
            and intent["pubs_sha256"] == sha(folder / "pubs.json"), "Submission anchors differ")
    raw_path, metrics_path = folder / "runtime.json.gz", folder / "metrics.json"
    if not raw_path.exists() or not metrics_path.exists():
        service = account()
        job = service.job(receipt["job_id"])
        if not raw_path.exists():
            result = job.result()
            # Save via exclusive temporary file, then atomic rename. An
            # interrupted incomplete temporary is never mistaken for raw data.
            temporary = folder / "runtime.json.gz.pending"
            with temporary.open("xb") as raw, gzip.open(raw, "wt") as handle:
                json.dump(result, handle, cls=RuntimeEncoder, allow_nan=False)
            require(not raw_path.exists(), "Concurrent collector wrote raw result")
            temporary.rename(raw_path)
        if not metrics_path.exists():
            metrics = job.metrics()
            require(accounted_usage(metrics) is not None, "Accounting not finalized; collect later, no new job")
            save(metrics_path, metrics)
    with gzip.open(raw_path, "rt") as handle:
        result = json.load(handle, cls=RuntimeDecoder)
    metrics = read(metrics_path)
    usage = accounted_usage(metrics)
    require(usage is not None and usage <= CAP, "Missing accounting or campaign usage exceeded cap")
    pubs, examples = read(folder / "pubs.json"), read(folder / "examples.json")
    require(len(result) == len(pubs) == 24, "Wrong returned PUB count")
    levels, penalties, _ = model_arrays(read(SOURCE / "model.json"))
    rows, grouped = [], {}
    for pub, specification in zip(result, pubs):
        example, width = examples[specification["example_index"]], specification["width"]
        problem = prepare_categorical_problem(example["aggregate"][:width], levels, penalties,
                                              example["weights"][:width], previous_states=None)
        counts = pub.data.meas.get_counts()
        require(sum(counts.values()) == SHOTS, "Unexpected shot count")
        score = score_categorical_counts(problem, counts)
        require(not score["classical_fallback_used"], "Unexpected fallback")
        rows.append({**specification, "raw_counts": counts, "sample_score": score})
        group = grouped.setdefault(specification["template"], {"circuits": 0, "shots": 0, "feasible_shots": 0})
        group["circuits"] += 1
        group["shots"] += SHOTS
        group["feasible_shots"] += score["feasible_shots"]
    for group in grouped.values():
        group["feasible_fraction"] = group["feasible_shots"] / group["shots"]
    save(folder / "result.json", {"completed_utc": utc(), "job_id": receipt["job_id"],
        "plan_sha256": sha(folder / "plan.json"), "runtime_sha256": sha(raw_path), "metrics_sha256": sha(metrics_path),
        "accounted_s": usage, "total_shots": 24 * SHOTS, "rows": rows, "groups": grouped,
        "no_fallback": True, "postselected_raw_counts": False})
    print(json.dumps({"accounted_s": usage, "groups": grouped}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "submit", "collect"), required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/quantum_diagnostics/ladder")
    parser.add_argument("--allow-free-submission", action="store_true")
    args = parser.parse_args()
    try:
        if args.mode == "submit":
            submit(args.output_dir.resolve(), args.allow_free_submission)
        else:
            {"prepare": prepare, "collect": collect}[args.mode](args.output_dir.resolve())
    except Exception as error:
        print(json.dumps({"status": "stopped safely", "mode": args.mode, "error_type": type(error).__name__}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
