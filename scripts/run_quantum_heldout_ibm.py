#!/usr/bin/env python3
"""Prepare, execute, recover and score all fixed held-out categorical circuits.

Only --mode run can submit. Open Plan only; no paid override. Adaptive jobs
process the next two intervals of all unfinished days in parallel. Immutable
per-stage submission intents prevent blind duplicate jobs after a failure.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from qiskit import qpy, qasm3
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeEncoder, SamplerV2

from quantum_nilm.categorical_ibm import (build_categorical_parameterized_template,
    categorical_template_parameter_values, score_categorical_counts)
from quantum_nilm.categorical_qaoa import prepare_categorical_problem
from quantum_nilm.evaluation import pool_window_metrics
from run_quantum_heldout import (CHANNELS, check_inputs, checked_window, expand_predictions,
    model_arrays, read_json, save_json, score_prediction, sha)

ROOT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, cls=RuntimeEncoder, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def account():
    service = QiskitRuntimeService(name="qnilm")
    active = service.active_instance()
    plan = next((item.get("plan") for item in service.instances() if item["crn"] == active), None)
    if plan != "open":
        raise RuntimeError("Only the Open Plan is permitted; no paid override")
    return service


def remaining(service):
    usage = service.usage()
    value = usage.get("usage_remaining_seconds")
    if value is None or not np.isfinite(value) or value < 0 or usage.get("usage_limit_reached"):
        raise RuntimeError("Free allowance unavailable or exhausted")
    return float(value)


def circuit_info(circuit):
    active = {circuit.find_bit(q).index for item in circuit.data if item.operation.name != "barrier" for q in item.qubits}
    return {"allocated_qubits": circuit.num_qubits, "active_qubits": len(active),
            "active_physical_indices": sorted(active), "depth": circuit.depth(),
            "gate_counts": dict(circuit.count_ops()),
            "logical_to_final_physical": circuit.layout.final_index_layout() if circuit.layout else None}


def accounted_usage(metrics):
    """Read current IBM billing field or its legacy alias; fail on conflict."""
    usage = metrics.get("usage") or {}
    primary = usage.get("qpu_charge_time_seconds")
    legacy = usage.get("quantum_seconds")
    for value in (primary, legacy):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not np.isfinite(value) or value < 0):
            raise RuntimeError("Invalid IBM accounted-usage value")
    if primary is not None and legacy is not None and not math.isclose(primary, legacy, rel_tol=0, abs_tol=1e-9):
        raise RuntimeError("Conflicting IBM accounted-usage fields")
    return primary if primary is not None else legacy


def prepared_problem(chunk, levels, penalties, previous):
    return prepare_categorical_problem(chunk["aggregate"], levels, penalties, chunk["weights"], previous_states=previous)


def prepare(output):
    check_inputs(output)
    folder = output / "hardware"
    if folder.exists():
        raise FileExistsError("Hardware preparation already exists")
    angles = read_json(output / "angles.json")
    config = read_json(output / "protocol.json")
    windows = read_json(output / "test_inputs.json")
    levels, penalties, _ = model_arrays(read_json(output / "model.json"))
    service = account()
    allowance = remaining(service)
    backend = service.backend("ibm_fez")
    if not backend.status().operational:
        raise RuntimeError("Prepared backend is not operational")
    folder.mkdir()
    started = perf_counter()
    atomic_json(folder / "backend_properties.json", backend.properties().to_dict())
    atomic_json(folder / "backend_configuration.json", backend.configuration().to_dict())
    templates = {}
    for length in (1, 2):
        template = build_categorical_parameterized_template([len(x) for x in levels], length, angles["betas"][0])
        manager = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=7)
        compiled = manager.run(template.circuit)
        with (folder / f"template_{length}.qpy").open("wb") as handle:
            qpy.dump(compiled, handle)
        (folder / f"template_{length}.qasm").write_text(qasm3.dumps(compiled))
        chunk = next(c for window in windows for c in window["chunks"] if len(c["weights"]) == length)
        problem = prepared_problem(chunk, levels, penalties, None)
        values = categorical_template_parameter_values(template, problem, angles["gammas"][0], parameter_order=tuple(compiled.parameters))
        numeric = compiled.assign_parameters(dict(zip(compiled.parameters, values)))
        duration = float(numeric.estimate_duration(backend.target, unit="s"))
        templates[str(length)] = {"parameter_order": [str(p) for p in compiled.parameters],
                                 "duration_s": duration, "statistics": circuit_info(compiled)}
    shots = config["shots_per_chunk"]
    rep_delay = float(backend.default_rep_delay)
    stages = max(len(w["chunks"]) for w in windows)
    circuit_execution = sum(templates[str(len(c["weights"]))]["duration_s"] + rep_delay
                            for w in windows for c in w["chunks"]) * shots
    plan = {"created_utc": utc(), "backend": backend.name, "instance_plan": "open",
            "shots": shots, "maximum_campaign_usage_s": config["hardware_maximum_campaign_usage_s"],
            "free_allowance_at_preparation_s": allowance, "days": len(windows),
            "circuits": sum(len(w["chunks"]) for w in windows), "adaptive_stages": stages,
            "templates": templates, "rep_delay_s": rep_delay,
            "estimated_shot_execution_s": circuit_execution,
            "estimated_accounted_usage_s": circuit_execution + 2 * stages,
            "estimate_note": "critical-path duration plus repetition delay and approximate2s/job overhead; actual subjobs and accounting may differ",
            "preparation_wall_time_s": perf_counter() - started,
            "angles_sha256": sha(output / "angles.json"), "protocol_sha256": sha(output / "protocol.json"),
            "versions": {name: version(name) for name in ("qiskit", "qiskit-ibm-runtime", "numpy")},
            "files_sha256": {p.name: sha(p) for p in folder.iterdir() if p.is_file()},
            "code_sha256": {p: sha(ROOT / p) for p in ("scripts/run_quantum_heldout_ibm.py", "src/quantum_nilm/categorical_ibm.py", "src/quantum_nilm/categorical_qaoa.py")},
            "fallback_policy": config["hardware_zero_valid_policy"],
            "suppression": "dynamical decoupling and gate/measurement twirling disabled",
            "timing": "job/accounted usage distinct from circuit time and end-to-end workflow"}
    save_json(folder / "plan.json", plan)
    print(json.dumps({k: plan[k] for k in ("circuits", "adaptive_stages", "estimated_accounted_usage_s", "free_allowance_at_preparation_s", "maximum_campaign_usage_s")}, indent=2), flush=True)


def check_plan(output):
    check_inputs(output)
    plan = read_json(output / "hardware/plan.json")
    for name, expected in plan["files_sha256"].items():
        if Path(name).name != name or sha(output / "hardware" / name) != expected:
            raise RuntimeError("Prepared hardware file hash mismatch")
    if sha(output / "angles.json") != plan["angles_sha256"] or sha(output / "protocol.json") != plan["protocol_sha256"]:
        raise RuntimeError("Frozen angle or protocol changed")
    expected_code = plan["code_sha256"]
    for path in sorted((output / "hardware").glob("code_revision_*.json")):
        revision = read_json(path)
        if (revision["plan_sha256"] != sha(output / "hardware/plan.json")
                or revision["previous_code_sha256"] != expected_code
                or set(revision["code_sha256"]) != set(expected_code)):
            raise RuntimeError("Invalid execution-code revision chain")
        expected_code = revision["code_sha256"]
    for name, expected in expected_code.items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("Audited hardware code changed after preparation")
    return plan


def decode_stage(plan, current, result, job_metrics, elapsed):
    if len(result) != len(current):
        raise RuntimeError("Returned PUB count differs from requested day count")
    rows = []
    for item, pub in zip(current, result):
        counts = pub.data.meas.get_counts()
        if sum(counts.values()) != plan["shots"]:
            raise RuntimeError("Unexpected hardware shot count")
        score = score_categorical_counts(item["problem"], counts)
        states = score["best_states"]
        fallback = states is None
        if fallback:
            prior = item["previous"]
            state = np.zeros(len(item["problem"].state_counts), dtype=int) if prior is None else prior
            states = np.tile(state, (len(item["problem"].aggregate), 1)).tolist()
        rows.append({"window_id": item["window_id"], "chunk": item["chunk"]["chunk"],
                     "previous_states": None if item["previous"] is None else item["previous"].tolist(),
                     "states": states, "fallback_used": fallback, "raw_counts": counts, "sample_score": score})
    return {"completed_utc": utc(), "rows": rows, "job_metrics": job_metrics,
            "result_wait_wall_time_s": elapsed}


def run(output):
    plan = check_plan(output)
    folder = output / "hardware"
    service = account()
    backend = service.backend(plan["backend"])
    if not backend.status().operational:
        raise RuntimeError("Backend is not operational")
    config, angles = read_json(output / "protocol.json"), read_json(output / "angles.json")
    windows = read_json(output / "test_inputs.json")
    levels, penalties, _ = model_arrays(read_json(output / "model.json"))
    templates, compiled = {}, {}
    for length in (1, 2):
        templates[length] = build_categorical_parameterized_template([len(x) for x in levels], length, angles["betas"][0])
        with (folder / f"template_{length}.qpy").open("rb") as handle:
            compiled[length] = qpy.load(handle)[0]
        if [str(p) for p in compiled[length].parameters] != plan["templates"][str(length)]["parameter_order"]:
            raise RuntimeError("Prepared parameter order changed")
    start_path = folder / "start.json"
    if not start_path.exists():
        save_json(start_path, {"created_utc": utc(), "initial_free_allowance_s": remaining(service),
                               "plan_sha256": sha(folder / "plan.json")})
    start = read_json(start_path)
    if start["plan_sha256"] != sha(folder / "plan.json"):
        raise RuntimeError("Started hardware plan hash changed")
    previous = {w["window"]["id"]: None for w in windows}
    accounted = 0.0
    for stage in range(plan["adaptive_stages"]):
        stage_started = perf_counter()
        if (folder / "STOP").exists():
            print("Stop marker found; no further jobs submitted", flush=True)
            return
        name = f"stage_{stage:03d}"
        result_path, intent_path = folder / f"{name}_result.json", folder / f"{name}_job.json"
        if result_path.exists():
            saved = read_json(result_path)
            expected_days = [w["window"]["id"] for w in windows if stage < len(w["chunks"])]
            if (saved.get("stage") != stage or saved.get("plan_sha256") != sha(folder / "plan.json")
                    or [row["window_id"] for row in saved["rows"]] != expected_days):
                raise RuntimeError("Saved stage result does not match frozen schedule")
        else:
            current, pubs = [], []
            for window in windows:
                if stage >= len(window["chunks"]):
                    continue
                chunk, wid = window["chunks"][stage], window["window"]["id"]
                prior = None if chunk["reset"] else previous[wid]
                problem = prepared_problem(chunk, levels, penalties, prior)
                length = len(chunk["weights"])
                values = categorical_template_parameter_values(templates[length], problem, angles["gammas"][0], parameter_order=tuple(compiled[length].parameters))
                pubs.append((compiled[length], values))
                current.append({"window_id": wid, "chunk": chunk, "previous": prior, "problem": problem})
            if intent_path.exists():
                intent = read_json(intent_path)
                expected_prior = {x["window_id"]: None if x["previous"] is None else x["previous"].tolist() for x in current}
                if (intent.get("plan_sha256") != sha(folder / "plan.json") or intent.get("stage") != stage
                        or intent.get("days") != [x["window_id"] for x in current]
                        or intent.get("shots_per_circuit") != plan["shots"]
                        or intent.get("previous_states") != expected_prior
                        or intent.get("parameter_values") != [np.asarray(pub[1]).tolist() for pub in pubs]):
                    raise RuntimeError("Saved submission intent does not match reconstructed stage")
                if not intent.get("job_id"):
                    raise RuntimeError("Ambiguous earlier submission; reconcile manually, never resubmit")
                job = service.job(intent["job_id"])
                print(f"Recovering stage {stage + 1}: {intent['job_id']}", flush=True)
            else:
                allowance = remaining(service)
                consumed = max(accounted, start["initial_free_allowance_s"] - allowance)
                budget = min(plan["maximum_campaign_usage_s"] - consumed, allowance - 20)
                estimate = 2 + plan["shots"] * sum(plan["templates"][str(len(x["chunk"]["weights"]))]["duration_s"] + plan["rep_delay_s"] for x in current)
                cap = int(math.ceil(1.5 * estimate + 2))
                if budget < cap:
                    atomic_json(folder / "budget_stop.json", {"stopped_utc": utc(), "next_stage": stage,
                                "accounted_s": accounted, "conservative_consumed_s": consumed,
                                "free_remaining_s": allowance, "required_next_job_cap_s": cap})
                    print("Stopped before submission: campaign/free-allowance guard", flush=True)
                    return
                request = {"created_utc": utc(), "stage": stage, "plan_sha256": sha(folder / "plan.json"),
                           "days": [x["window_id"] for x in current], "shots_per_circuit": plan["shots"],
                           "max_execution_time_s": cap, "estimated_usage_s": estimate,
                           "previous_states": {x["window_id"]: None if x["previous"] is None else x["previous"].tolist() for x in current},
                           "parameter_values": [np.asarray(pub[1]).tolist() for pub in pubs],
                           "status": "submission_started"}
                with intent_path.open("x") as handle:
                    json.dump(request, handle, indent=2, allow_nan=False)
                sampler = SamplerV2(mode=backend, options={"max_execution_time": cap,
                    "dynamical_decoupling": {"enable": False},
                    "twirling": {"enable_gates": False, "enable_measure": False},
                    "execution": {"rep_delay": plan["rep_delay_s"], "init_qubits": True},
                    "environment": {"job_tags": ["qnilm", "heldout-categorical", f"stage-{stage:03d}"]}})
                job = sampler.run(pubs, shots=plan["shots"])
                request.update({"job_id": job.job_id(), "submitted_utc": utc(), "status": "submitted"})
                atomic_json(intent_path, request)
                print(f"Submitted stage {stage + 1}/{plan['adaptive_stages']}: {len(pubs)} circuits, job {job.job_id()}", flush=True)
            wait_start = perf_counter()
            result = job.result()
            with gzip.open(folder / f"{name}_runtime.json.gz", "wt") as handle:
                json.dump(result, handle, cls=RuntimeEncoder, allow_nan=False)
            saved = decode_stage(plan, current, result, job.metrics(), perf_counter() - wait_start)
            saved.update({"stage": stage, "job_id": job.job_id(), "plan_sha256": sha(folder / "plan.json"),
                          "stage_preparation_submit_wait_wall_time_s": perf_counter() - stage_started})
            save_json(result_path, saved)
        usage = accounted_usage(saved["job_metrics"])
        if usage is None:
            refreshed = service.job(saved["job_id"]).metrics()
            usage = accounted_usage(refreshed)
            if usage is not None:
                # Retain the original result; billing can finalize after counts arrive.
                refresh_path = folder / f"{name}_finalized_metrics.json"
                if not refresh_path.exists():
                    save_json(refresh_path, {"refreshed_utc": utc(), "job_metrics": refreshed})
        if usage is None or not np.isfinite(usage) or usage < 0:
            raise RuntimeError("Accounted usage not yet available; recover this stage before proceeding")
        accounted += float(usage)
        for row in saved["rows"]:
            previous[row["window_id"]] = np.asarray(row["states"][-1], dtype=int)
        feasible = sum(row["sample_score"]["feasible_shots"] for row in saved["rows"])
        total = len(saved["rows"]) * plan["shots"]
        print(f"Completed stage {stage + 1}/{plan['adaptive_stages']}; accounted {accounted:.1f}s; valid shots {feasible}/{total}", flush=True)
    save_json(folder / "completed.json", {"completed_utc": utc(), "stages": plan["adaptive_stages"],
              "circuits": plan["circuits"], "accounted_quantum_seconds": accounted,
              "free_remaining_s": remaining(service), "plan_sha256": sha(folder / "plan.json")})


def score(output):
    plan = check_plan(output)
    folder = output / "hardware"
    if not (folder / "completed.json").exists():
        raise RuntimeError("Full hardware evaluation is incomplete; do not report complete MAE")
    config = read_json(output / "protocol.json")
    windows = read_json(output / "test_inputs.json")
    levels, _, _ = model_arrays(read_json(output / "model.json"))
    quality = read_json(Path(config["original_dir"]) / "data_quality.json")
    raw = {w["window"]["id"]: [] for w in windows}
    totals = {"raw_shots": 0, "feasible_shots": 0, "fallback_chunks": 0, "fallback_blocks": 0}
    for stage in range(plan["adaptive_stages"]):
        for row in read_json(folder / f"stage_{stage:03d}_result.json")["rows"]:
            raw[row["window_id"]].append(row)
            totals["raw_shots"] += row["sample_score"]["total_shots"]
            totals["feasible_shots"] += row["sample_score"]["feasible_shots"]
            totals["fallback_chunks"] += int(row["fallback_used"])
    records = []
    for window in windows:
        rows = raw[window["window"]["id"]]
        prediction = expand_predictions(window, rows, levels)
        loaded = checked_window(config, quality, window["window"], ["main", *CHANNELS])
        record = score_prediction(window, loaded, prediction, config["state_proxy_thresholds_w"], "qaoa_ibm_with_declared_fallback", None, None)
        record["fallback_blocks"] = sum(c["block_stop"] - c["block_start"] for c, r in zip(window["chunks"], rows) if r["fallback_used"])
        totals["fallback_blocks"] += record["fallback_blocks"]
        records.append(record)
    save_json(folder / "test_windows.json", records)
    appliances = pool_window_metrics(records, CHANNELS)
    blocks = sum(r["blocks"] for r in records)
    summary = {"execution": "physical IBM QPU, adaptive categorical p1, declared no-valid-sample fallback",
               "windows": len(records), "blocks": blocks, "circuits": plan["circuits"], **totals,
               "feasible_fraction": totals["feasible_shots"] / totals["raw_shots"],
               "appliances": appliances, "macro_appliance_mae_w": float(np.mean([a["mae_w"] for a in appliances.values()])),
               "aggregate_mae_w": sum(r["raw_aggregate_mae_w"] * r["blocks"] for r in records) / blocks,
               "completion": read_json(folder / "completed.json"), "limitations": config["limitations"],
               "hardware_limitations": ["one adaptive campaign, not independent date/layout replicas", "infeasible shots are discarded and charged; all-invalid chunks use declared carry-state/low-state fallback"]}
    save_json(folder / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("windows", "blocks", "macro_appliance_mae_w", "feasible_fraction", "fallback_chunks")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("prepare", "run", "score"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/quantum_heldout")
    args = parser.parse_args()
    try:
        {"prepare": prepare, "run": run, "score": score}[args.mode](args.output_dir)
    except Exception as error:
        # Never print provider exceptions that might contain account identifiers.
        print(json.dumps({"status": "stopped safely", "error_type": type(error).__name__, "mode": args.mode}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
