#!/usr/bin/env python3
"""Additive, retrospective matched-objective MILP/DP benchmark; no QPU use.

This does not reproduce either published Balletti or Li NILM pipeline.
Every prediction is archived before reference values are read for scoring.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.run_quantum_heldout import check_inputs, checked_window, make_chunks
from scripts.run_stage_d_components import (
    CHANNELS, assemble_runs, direct_objective, model_values, paired_bootstrap,
    pooled, score_trace,
)
from quantum_nilm.multistate import solve_multistate_temporal_exact
from quantum_nilm.stage_e_mip import solve_categorical_milp

SOURCE = ROOT / "results/quantum_heldout"
ARMS = ("milp", "exact_dp")
CODE = (
    "scripts/run_stage_e_mip.py", "src/quantum_nilm/stage_e_mip.py",
    "tests/test_stage_e_runner.py", "tests/test_stage_e_mip.py",
    "docs/stage_e_mip_protocol.md", "scripts/run_quantum_heldout.py",
    "scripts/run_stage_d_components.py", "src/quantum_nilm/evaluation.py",
    "src/quantum_nilm/multistate.py", "src/quantum_nilm/heldout_data.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize before exclusive creation, so a nonfinite value leaves no file.
    data = json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n"
    with path.open("x") as handle:
        handle.write(data)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def digest_value(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True,
                                   separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validation_selection(windows):
    candidates = [w for w in windows if w["split"] in ("validation", "val")]
    candidates.sort(key=lambda w: (w["start_unix"], w["id"]))
    require(len(candidates) >= 3, "Three original validation windows are required")
    return candidates[:3]


def rederive_window(metadata, loaded, threshold):
    """Accept only timestamps/mains; no appliance reference argument."""
    values = np.asarray(loaded["values"])
    require(values.ndim == 2 and values.shape[1] == 1,
            "Inference preparation requires mains-only data")
    chunks = make_chunks(values[:, 0], loaded["timestamps"], threshold)
    require(chunks, "A fixed selected window has no valid runs")
    return {"window": metadata, "blocks": len(loaded["timestamps"]),
            "source_window_sha256": loaded["source_window_sha256"], "chunks": chunks}


def select_candidate(candidates):
    require(len(candidates) == 2 and {r["presolve"] for r in candidates} == {True, False},
            "Exactly the frozen presolve-on/off candidates are required")
    def key(row):
        gap = row["sum_certified_relative_gap"]
        return (row["no_incumbent_runs"], row["not_solver_optimal_runs"],
                float("inf") if gap is None else gap,
                row["formulation_plus_solve_time_s"], not row["presolve"])
    return min(candidates, key=key)


def prepare(output):
    output = Path(output)
    require(not output.exists(), "Use a new archive; existing experiments are immutable")
    check_inputs(SOURCE)
    config, model = read(SOURCE / "protocol.json"), read(SOURCE / "model.json")
    levels, ranges, rho = model_values(model)
    require([len(x) for x in levels] == [4, 3, 2, 3], "Frozen register sizes changed")
    windows = read(SOURCE / "test_inputs.json")
    require(len(windows) == 30 and sum(w["blocks"] for w in windows) == 86396,
            "Original test population changed")
    require(sum(len(assemble_runs(w)) for w in windows) == 34
            and sum(len(r["weights"]) for w in windows for r in assemble_runs(w)) == 4881,
            "Original test compression changed")
    original = Path(config["original_dir"])
    quality = read(original / "data_quality.json")
    validation = validation_selection(config["windows"])
    artifacts = [SOURCE / p for p in ("protocol.json", "model.json", "test_inputs.json",
                 "input_hashes.json", "simulation/summary.json", "simulation/test_windows.json")]
    artifacts.extend(original / p for p in ("protocol.json", "frozen_models.json", "data_quality.json"))
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "Stage E generic matched-objective MILP; post-exposure original R1Hz windows",
        "source_directory": str(SOURCE), "source": config["source"],
        "source_artifacts_sha256": {str(p): sha(p) for p in artifacts},
        "source_code_sha256": {p: sha(ROOT / p) for p in CODE},
        "arms": list(ARMS), "windows": 30, "blocks": 86396, "runs": 34, "segments": 4881,
        "validation_windows": [w["id"] for w in validation],
        "candidate_presolve": [True, False], "validation_time_limit_s": 15.0,
        "test_time_limit_s": 30.0, "mip_rel_gap": 1e-8,
        "selection_order": ["fewest missing validated incumbents", "fewest not solver-optimal runs",
                            "smallest summed relative objective gap to exact DP",
                            "smallest total formulation plus backend solve seconds", "presolve on"],
        "certified_relative_gap": "max(0, incumbent-direct-DP) / max(1, abs(direct-DP)); missing incumbent is infinite in ranking",
        "rho": rho, "penalties": (rho * ranges**2).tolist(),
        "event_threshold_w": model["models"]["multistate"]["event_threshold_w"],
        "bootstrap": {"replicates": 2000, "seed": 9301, "unit": "paired original test window"},
        "no_reference_inference": True, "no_dp_warm_start": True,
        "test_arm_order": "even window index: MILP then DP; odd: DP then MILP",
        "failure_policy": "retain no-incumbent runs without fallback; no full-population MAE unless every original block covered",
        "timing_definition": "aggregate-only compression including equality/run checks + solver call (including build/decode or DP setup/traceback) + expansion; excludes source reads, separate objective/archive verification, scoring and archive I/O",
        "solver_threads": "SciPy/HiGHS default; not exposed or overridden by this runner",
        "solver_seed": "SciPy/HiGHS default; not exposed or overridden by this runner",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform()},
        "limitations": ["generic matched-objective solver, not Balletti or Li reproduction",
                        "single retrospective cohort and one timing execution per arm; no stable speed claim",
                        "expanded 72-joint-state formulation; not a scalability claim for arbitrary appliance counts",
                        "no new quantum execution or quantum-advantage evidence",
                        "complete benchmark does not complete all of Stage E"],
    }
    output.mkdir(parents=True)
    save(output / "protocol.json", protocol)
    for name in CODE:
        path = output / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    save(output / "frozen_inputs.json", {"config": config, "model": model, "windows": windows,
         "source_quality": quality, "validation_windows": validation})
    validation_inputs, validation_files = [], {}
    for metadata in validation:
        loaded = checked_window(config, quality, metadata, ["main"])
        window = rederive_window(metadata, loaded, protocol["event_threshold_w"])
        validation_inputs.append(window)
        path = output / "validation_data" / f"{metadata['id']}.npz"
        save_npz(path, timestamps=loaded["timestamps"], mains_w=loaded["values"][:, 0])
        validation_files[str(path.relative_to(output))] = sha(path)
    save(output / "validation_inputs.json", validation_inputs)
    freeze = {"protocol_sha256": sha(output / "protocol.json"),
              "frozen_inputs_sha256": sha(output / "frozen_inputs.json"),
              "validation_inputs_sha256": sha(output / "validation_inputs.json"),
              "validation_data_sha256": validation_files}
    save(output / "freeze.json", freeze)
    check_frozen(output)
    print(json.dumps({"status": "frozen_before_tuning", "protocol_sha256": freeze["protocol_sha256"]}), flush=True)


def check_frozen(output):
    output = Path(output)
    freeze, protocol = read(output / "freeze.json"), read(output / "protocol.json")
    check_environment(protocol["environment"])
    for key, name in (("protocol_sha256", "protocol.json"),
                      ("frozen_inputs_sha256", "frozen_inputs.json"),
                      ("validation_inputs_sha256", "validation_inputs.json")):
        require(sha(output / name) == freeze[key], f"Frozen artifact changed: {name}")
    for name, expected in freeze["validation_data_sha256"].items():
        require(sha(output / name) == expected, f"Validation data changed: {name}")
    for name, expected in protocol["source_artifacts_sha256"].items():
        require(sha(name) == expected, f"Original artifact changed: {name}")
    for name, expected in protocol["source_code_sha256"].items():
        require(sha(ROOT / name) == expected and sha(output / "source_snapshot" / name) == expected,
                f"Frozen code changed: {name}")
    check_inputs(Path(protocol["source_directory"]))
    return protocol, read(output / "frozen_inputs.json"), freeze["protocol_sha256"]


def check_environment(expected):
    actual = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    require(all(expected.get(name) == value for name, value in actual.items()),
            "Frozen Python/NumPy/SciPy versions changed; create a separate explicitly documented replay archive")


def checked_states(states, run, levels):
    labels = np.asarray(states)
    require(labels.shape == (len(run["weights"]), len(levels)) and labels.dtype.kind in "iu",
            "Invalid solver state trajectory")
    require(all(np.all((labels[:, i] >= 0) & (labels[:, i] < len(values)))
                for i, values in enumerate(levels)), "Solver category out of range")
    return labels


def solve_run(run, levels, penalties, arm, presolve, time_limit_s, mip_rel_gap):
    started = perf_counter()
    if arm == "exact_dp":
        result = solve_multistate_temporal_exact(run["aggregate"], levels, penalties, run["weights"])
    else:
        result = solve_categorical_milp(run["aggregate"], levels, penalties, run["weights"],
                    presolve=presolve, time_limit_s=time_limit_s, mip_rel_gap=mip_rel_gap)
    call_time = perf_counter() - started
    solver = jsonable(asdict(result))
    record = {k: run[k] for k in ("run", "block_start", "block_stop")}
    record.update({"solver": solver, "solver_call_wall_time_s": call_time,
                   "states": None, "incumbent_feasible": False,
                   "solver_wall_time_s": solver.get("solve_time_s", solver["wall_time_s"]),
                   "build_time_s": solver.get("build_time_s"),
                   "solver_certified_optimal": arm == "exact_dp" or
                       (solver["status_code"] == 0 and solver["success"] and solver["incumbent_feasible"]
                        and solver["dual_bound"] is not None
                        and solver["bound_validation"] in ("passed", "negative_within_roundoff"))})
    if result.states is None:
        record["failure"] = "No validated incumbent; no fallback substituted"
        return record
    labels = checked_states(result.states, run, levels)
    objective, reconstruction, transition, switches = direct_objective(
        run["aggregate"], run["weights"], labels, levels, penalties)
    require(np.isclose(objective, result.energy, rtol=1e-10, atol=1e-4),
            "Decoded direct objective differs from solver-returned energy")
    require(arm == "exact_dp" or solver["incumbent_feasible"], "Unvalidated MILP incumbent")
    record.update({"states": labels.tolist(), "incumbent_feasible": True,
        "native_segment_objective": objective, "common_segment_objective": objective,
        "native_reconstruction": reconstruction, "common_segment_reconstruction": reconstruction,
        "native_transition": transition, "common_transition": transition,
        "switch_counts": switches.tolist()})
    return record


def expand_partial(window, records, levels):
    prediction = np.full((window["blocks"], len(levels)), np.nan)
    means = np.full(window["blocks"], np.nan)
    runs = assemble_runs(window)
    require(len(records) == len(runs), "Some run outcomes are missing")
    for run, record in zip(runs, records):
        require(all(record[k] == run[k] for k in ("run", "block_start", "block_stop")), "Run identity changed")
        part = slice(run["block_start"], run["block_stop"])
        means[part] = np.repeat(run["aggregate"], run["weights"])
        if record["states"] is not None:
            labels = checked_states(record["states"], run, levels)
            powers = np.column_stack([v[labels[:, i]] for i, v in enumerate(levels)])
            prediction[part] = np.repeat(powers, run["weights"], axis=0)
    require(np.all(np.isfinite(means)), "Compression did not cover the fixed window")
    return prediction, means, np.all(np.isfinite(prediction), axis=1)


def infer_window(window, loaded, levels, penalties, threshold, arm, *, presolve=True,
                 time_limit_s=30.0, mip_rel_gap=1e-8, checkpoint=None, binding=None):
    started = perf_counter()
    fresh = rederive_window(window["window"], loaded, threshold)
    require(fresh == window, "Fresh aggregate-only compression differs from frozen inputs")
    runs = assemble_runs(fresh)
    compression_time = perf_counter() - started
    records, checkpoint_hashes = [], {}
    for number, run in enumerate(runs):
        path = Path(checkpoint) / f"run_{number:03d}.json" if checkpoint is not None else None
        if path is not None and path.exists():
            record = read(path)
            require(record["binding"] == binding and record["input_sha256"] == digest_value(run),
                    "Run checkpoint provenance changed")
        else:
            if path is not None:
                intent = path.with_suffix(".intent.json")
                require(not intent.exists(), "Unfinished local solve intent requires reconciliation; not silently retried")
                save(intent, {"binding": binding, "input_sha256": digest_value(run),
                              "started_utc": datetime.now(timezone.utc).isoformat()})
            record = solve_run(run, levels, penalties, arm, presolve, time_limit_s, mip_rel_gap)
            record.update({"binding": binding, "input_sha256": digest_value(run)})
            if path is not None:
                save(path, record)
        if path is not None:
            checkpoint_hashes[path.name] = sha(path)
        records.append(record)
    started = perf_counter()
    prediction, means, mask = expand_partial(fresh, records, levels)
    expansion_time = perf_counter() - started
    trace = {"window_id": window["window"]["id"], "arm": arm, "runs": records,
             "complete": bool(np.all(mask)), "covered_blocks": int(mask.sum()),
             "blocks": window["blocks"], "compression_wall_time_s": compression_time,
             "expansion_wall_time_s": expansion_time,
             "inference_wall_time_s": compression_time + expansion_time + sum(r["solver_call_wall_time_s"] for r in records),
             "solver_wall_time_s": sum(r["solver_wall_time_s"] for r in records),
             "formulation_wall_time_s": None if arm == "exact_dp" else sum(r["build_time_s"] for r in records),
             "checkpoint_sha256": checkpoint_hashes, "binding": binding,
             "fresh_compression_sha256": digest_value(fresh), "presolve": None if arm == "exact_dp" else presolve}
    return trace, prediction, means, mask


def readonly_model(frozen):
    levels, ranges, rho = model_values(frozen)
    penalties = rho * ranges**2
    for values in [*levels, penalties]:
        values.setflags(write=False)
    return levels, penalties


def load_validation(output, window):
    with np.load(Path(output) / "validation_data" / f"{window['window']['id']}.npz", allow_pickle=False) as data:
        return {"timestamps": data["timestamps"].copy(), "values": data["mains_w"][:, None].copy(),
                "source_window_sha256": window["source_window_sha256"]}


def checked_trace(path, checkpoint, binding):
    trace = read(path)
    require(trace["binding"] == binding, "Trace binding changed")
    for name, expected in trace["checkpoint_sha256"].items():
        require(sha(Path(checkpoint) / name) == expected, "Run checkpoint changed after trace was saved")
    return trace


def tune(output):
    output = Path(output)
    protocol, frozen, protocol_hash = check_frozen(output)
    if (output / "selection.json").exists():
        return check_selection(output, protocol_hash)
    require(not (output / "traces").exists(), "Test inference must not precede frozen setting selection")
    levels, penalties = readonly_model(frozen["model"])
    windows = read(output / "validation_inputs.json")
    traces, artifacts = {}, {}
    for label, presolve in (("exact_dp", True), ("presolve_on", True), ("presolve_off", False)):
        traces[label] = []
        for window in windows:
            name = window["window"]["id"]
            path = output / "tuning/traces" / label / f"{name}.json"
            checkpoint = output / "tuning/run_records" / label / name
            binding = {"protocol_sha256": protocol_hash, "phase": "validation", "candidate": label}
            if path.exists():
                trace = checked_trace(path, checkpoint, binding)
            else:
                trace, _, _, _ = infer_window(window, load_validation(output, window), levels, penalties,
                    protocol["event_threshold_w"], "exact_dp" if label == "exact_dp" else "milp",
                    presolve=presolve, time_limit_s=protocol["validation_time_limit_s"],
                    mip_rel_gap=protocol["mip_rel_gap"], checkpoint=checkpoint, binding=binding)
                save(path, trace)
            traces[label].append(trace)
            artifacts[str(path.relative_to(output))] = sha(path)
            for leaf, digest in trace["checkpoint_sha256"].items():
                artifacts[str((checkpoint / leaf).relative_to(output))] = digest
            print(f"Validation {label}: {name}", flush=True)
    candidates = []
    exact = [r for t in traces["exact_dp"] for r in t["runs"]]
    for label, presolve in (("presolve_on", True), ("presolve_off", False)):
        rows = [r for t in traces[label] for r in t["runs"]]
        require(len(rows) == len(exact), "Candidate validation coverage differs")
        missing = sum(not row["incumbent_feasible"] for row in rows)
        gaps, certificates = [], []
        for row, oracle in zip(rows, exact):
            require(row["input_sha256"] == oracle["input_sha256"], "Validation candidates have different inputs")
            optimum = oracle["common_segment_objective"]
            gap = None if not row["incumbent_feasible"] else row["common_segment_objective"] - optimum
            if gap is not None:
                require(gap >= -max(1e-4, 1e-10 * abs(optimum)), "MILP beats the exact validation certificate")
                gaps.append(max(0.0, gap) / max(1.0, abs(optimum)))
            certificates.append({"input_sha256": row["input_sha256"], "exact_dp_objective": optimum,
                                 "milp_minus_dp_objective": gap,
                                 "certified_relative_gap": None if gap is None else gaps[-1]})
        candidates.append({"label": label, "presolve": presolve, "runs": len(rows),
            "no_incumbent_runs": missing,
            "not_solver_optimal_runs": sum(not row["solver_certified_optimal"] for row in rows),
            "sum_certified_relative_gap": None if missing else sum(gaps),
            "formulation_plus_solve_time_s": sum(row["build_time_s"] + row["solver_wall_time_s"] for row in rows),
            "certificates": certificates})
    result = {"protocol_sha256": protocol_hash, "candidates": candidates,
              "artifact_sha256": artifacts, "appliance_labels_used": False,
              "selected_presolves": [select_candidate(candidates)["presolve"]]}
    path = output / "candidates.json"
    if path.exists():
        require(read(path) == result, "Existing candidate archive differs")
    else:
        save(path, result)
    check_frozen(output)
    selection = {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol_hash,
                 "candidates_sha256": sha(path), "presolve": select_candidate(candidates)["presolve"],
                 "selected_before_test_inference": True, "appliance_labels_used": False}
    save(output / "selection.json", selection)
    print(json.dumps(selection), flush=True)
    return selection


def check_selection(output, protocol_hash):
    output = Path(output)
    selection, candidates = read(output / "selection.json"), read(output / "candidates.json")
    require(selection["protocol_sha256"] == protocol_hash and candidates["protocol_sha256"] == protocol_hash,
            "Setting-selection protocol changed")
    require(sha(output / "candidates.json") == selection["candidates_sha256"], "Candidate archive changed")
    for name, expected in candidates["artifact_sha256"].items():
        require(sha(output / name) == expected, "Validation candidate or certificate changed")
    require(select_candidate(candidates["candidates"])["presolve"] == selection["presolve"],
            "Selected presolve does not follow the frozen rule")
    return selection


def test_binding(protocol_hash, selection_hash):
    return {"protocol_sha256": protocol_hash, "selection_sha256": selection_hash, "phase": "test"}


def run(output):
    output = Path(output)
    protocol, frozen, protocol_hash = check_frozen(output)
    selection = check_selection(output, protocol_hash)
    binding = test_binding(protocol_hash, sha(output / "selection.json"))
    levels, penalties = readonly_model(frozen["model"])
    started = perf_counter()
    for number, window in enumerate(frozen["windows"]):
        name = window["window"]["id"]
        read_started = perf_counter()
        loaded = checked_window(frozen["config"], frozen["source_quality"], window["window"], ["main"])
        read_elapsed = perf_counter() - read_started
        receipt_path = output / "source_reads" / f"{name}.json"
        if not receipt_path.exists():
            save(receipt_path, {"binding": binding, "window_id": name,
                 "source_window_sha256": loaded["source_window_sha256"], "blocks": len(loaded["timestamps"]),
                 "source_read_wall_time_s": read_elapsed})
        receipt = read(receipt_path)
        require(receipt["binding"] == binding and receipt["source_window_sha256"] == loaded["source_window_sha256"]
                and receipt["blocks"] == window["blocks"], "Saved source-read receipt changed")
        order = ARMS if number % 2 == 0 else tuple(reversed(ARMS))
        traces, prediction_hashes = {}, {}
        for arm in order:
            trace_path = output / "traces" / arm / f"{name}.json"
            prediction_path = output / "predictions" / arm / f"{name}.npz"
            checkpoint = output / "run_records" / arm / name
            if trace_path.exists():
                trace = checked_trace(trace_path, checkpoint, binding)
                require(trace["arm"] == arm and trace["window_id"] == name, "Archived trace identity differs")
                prediction, means, mask = expand_partial(window, trace["runs"], levels)
            else:
                trace, prediction, means, mask = infer_window(window, loaded, levels, penalties,
                    protocol["event_threshold_w"], arm, presolve=selection["presolve"],
                    time_limit_s=protocol["test_time_limit_s"], mip_rel_gap=protocol["mip_rel_gap"],
                    checkpoint=checkpoint, binding=binding)
                trace["window_index"] = number
                trace["arm_order"] = list(order)
                trace["protocol_sha256"] = protocol_hash
                save(trace_path, trace)
            if not prediction_path.exists():
                save_npz(prediction_path, prediction_w=prediction, compressed_mains_w=means,
                         timestamps=loaded["timestamps"], covered_mask=mask)
            with np.load(prediction_path, allow_pickle=False) as data:
                require(np.array_equal(data["prediction_w"], prediction, equal_nan=True)
                        and np.array_equal(data["compressed_mains_w"], means)
                        and np.array_equal(data["timestamps"], loaded["timestamps"])
                        and np.array_equal(data["covered_mask"], mask), "Archived prediction differs from trace")
            traces[arm], prediction_hashes[arm] = trace, sha(prediction_path)
        # Both arms' predictions (including explicit failed runs) now exist.
        refs = None
        for arm in ARMS:
            trace, path = traces[arm], output / "records" / arm / f"{name}.json"
            provenance = {"protocol_sha256": protocol_hash, "selection_sha256": binding["selection_sha256"],
                "trace_sha256": sha(output / "traces" / arm / f"{name}.json"),
                "prediction_sha256": prediction_hashes[arm], "source_read_sha256": sha(receipt_path),
                "source_window_sha256": loaded["source_window_sha256"]}
            if path.exists():
                row = read(path)
                require(all(row.get(k) == v for k, v in provenance.items()), "Scored checkpoint provenance changed")
                continue
            if trace["complete"]:
                if refs is None:
                    refs = checked_window(frozen["config"], frozen["source_quality"], window["window"], ["main", *CHANNELS])
                row = score_trace(window, trace, refs, levels, frozen["config"]["state_proxy_thresholds_w"])
                row["complete"] = True
            else:
                row = {"window_id": name, "model": arm, "complete": False,
                       "blocks": window["blocks"], "covered_blocks": trace["covered_blocks"],
                       "reason": "No validated incumbent for at least one run; no full-window metrics or DP fallback",
                       "appliances": None, "full_window_metrics": None}
            row.update(provenance)
            save(path, row)
        print(f"Test window {number + 1}/{len(frozen['windows'])}: {name}; MILP complete={traces['milp']['complete']}", flush=True)
    check_frozen(output)
    check_selection(output, protocol_hash)
    if not (output / "execution.json").exists():
        save(output / "execution.json", {"status": "all_scheduled_windows_attempted",
             "completed_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol_hash,
             "current_process_wall_time_s": perf_counter() - started,
             "note": "May resume earlier records; authoritative inference time is the sum of saved components"})
    return summarize(output)


def summarize(output):
    output = Path(output)
    protocol, frozen, protocol_hash = check_frozen(output)
    check_selection(output, protocol_hash)
    selection_hash = sha(output / "selection.json")
    binding = test_binding(protocol_hash, selection_hash)
    records, traces, hashes, arms = {}, {}, {}, {}
    for arm in ARMS:
        records[arm], traces[arm] = [], []
        for window in frozen["windows"]:
            name = window["window"]["id"]
            path = output / "records" / arm / f"{name}.json"
            row = read(path)
            trace_path = output / "traces" / arm / f"{name}.json"
            trace = checked_trace(trace_path, output / "run_records" / arm / name, binding)
            require(trace["arm"] == arm and trace["window_id"] == name, "Trace identity differs")
            require(row["model"] == arm and row["window_id"] == name and row["protocol_sha256"] == protocol_hash
                    and row["selection_sha256"] == selection_hash and row["trace_sha256"] == sha(trace_path)
                    and row["prediction_sha256"] == sha(output / "predictions" / arm / f"{name}.npz")
                    and row["source_read_sha256"] == sha(output / "source_reads" / f"{name}.json"),
                    "Result provenance differs from the completed archive")
            require(row["complete"] == trace["complete"], "Coverage flag disagrees")
            records[arm].append(row)
            traces[arm].append(trace)
            hashes[str(path.relative_to(output))] = sha(path)
        complete = all(row["complete"] for row in records[arm])
        available = [row for row in records[arm] if row["complete"]]
        runs = [r for trace in traces[arm] for r in trace["runs"]]
        arms[arm] = {"full_coverage": complete, "scheduled_windows": len(records[arm]),
            "complete_windows": len(available), "scheduled_blocks": protocol["blocks"],
            "covered_blocks": sum(t["covered_blocks"] for t in traces[arm]),
            "no_incumbent_runs": sum(not r["incumbent_feasible"] for r in runs),
            "solver_optimal_runs": sum(r["solver_certified_optimal"] for r in runs),
            "missing_dual_bound_runs": 0 if arm == "exact_dp" else sum(r["solver"]["dual_bound"] is None for r in runs),
            "inconsistent_dual_bound_runs": 0 if arm == "exact_dp" else sum(r["solver"]["bound_validation"] == "dual_bound_exceeds_incumbent" for r in runs),
            "solver_status_counts": ({"exact_dp": len(runs)} if arm == "exact_dp" else
                {status: sum(r["solver"]["status"] == status for r in runs)
                 for status in sorted({r["solver"]["status"] for r in runs})}),
            "full_coverage_metrics": pooled(available) if complete else None,
            "complete_window_subset_metrics": None if complete or not available else pooled(available),
            "subset_warning": None if complete else "Subset metrics are not full-population accuracy and exclude incomplete windows",
            "inference_wall_time_s": sum(t["inference_wall_time_s"] for t in traces[arm]),
            "compression_wall_time_s": sum(t["compression_wall_time_s"] for t in traces[arm]),
            "expansion_wall_time_s": sum(t["expansion_wall_time_s"] for t in traces[arm]),
            "solver_call_wall_time_s": sum(r["solver_call_wall_time_s"] for r in runs),
            "backend_solve_wall_time_s": sum(r["solver_wall_time_s"] for r in runs),
            "formulation_wall_time_s": None if arm == "exact_dp" else sum(r["build_time_s"] for r in runs)}
    certificates = []
    for left, right in zip(traces["milp"], traces["exact_dp"]):
        require(left["window_id"] == right["window_id"], "Unpaired window traces")
        for candidate, oracle in zip(left["runs"], right["runs"]):
            require(candidate["input_sha256"] == oracle["input_sha256"], "Solvers used different inputs")
            optimum = oracle["common_segment_objective"]
            gap = candidate.get("common_segment_objective")
            gap = None if gap is None else gap - optimum
            if gap is not None:
                require(gap >= -max(1e-4, 1e-10 * abs(optimum)), "Test incumbent beats exact DP")
            certificates.append({"window_id": left["window_id"], "run": candidate["run"],
                "exact_dp_objective": optimum, "milp_minus_dp_objective": gap,
                "relative_gap_to_dp": None if gap is None else max(0.0, gap) / max(1.0, abs(optimum)),
                "milp_solver_certified_optimal": candidate["solver_certified_optimal"],
                "milp_status": candidate["solver"]["status"],
                "milp_dual_bound": candidate["solver"]["dual_bound"]})
    full = all(row["full_coverage"] for row in arms.values())
    comparison = paired_bootstrap(records["milp"], records["exact_dp"], seed=9301, replicates=2000) if full else None
    source_time = sum(read(output / "source_reads" / f"{w['window']['id']}.json")["source_read_wall_time_s"] for w in frozen["windows"])
    result = {"status": "complete" if full else "all_attempted_incomplete_prediction_coverage",
              "full_coverage": full, "protocol_sha256": protocol_hash, "selection_sha256": selection_hash,
              "selected_presolve": read(output / "selection.json")["presolve"], "arms": arms,
              "paired_milp_minus_dp_macro_mae": comparison, "run_certificates": certificates,
              "common_source_read_wall_time_s": source_time,
              "record_sha256": hashes, "limitations": protocol["limitations"],
              "stage_e_overall_status": "partial: source-faithful published NILM comparisons remain pending"}
    path = output / "summary.json"
    if path.exists():
        require(read(path) == result, "Saved summary differs from deterministic reconstruction")
    else:
        save(path, result)
    print(json.dumps({"status": result["status"], "arms": arms}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "tune", "run", "summarize"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage_e/mip/run_001")
    args = parser.parse_args()
    {"prepare": prepare, "tune": tune, "run": run, "summarize": summarize}[args.mode](args.output.resolve())


if __name__ == "__main__":
    main()
