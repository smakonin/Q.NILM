#!/usr/bin/env python3
"""Frozen, local exact-DP duration/penalty ablations on the original R1Hz cohort.

This additive retrospective programme neither trains a model nor submits jobs.
The original experiment and inference modules remain immutable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.run_quantum_heldout import check_inputs, checked_window
from quantum_nilm.evaluation import contiguous_slices, pool_window_metrics, score_power
from quantum_nilm.multistate import solve_multistate_temporal_exact

CHANNELS = ("dryr", "frdg", "vacu")
ARMS = ("duration_normalized", "unit_normalized", "duration_global_max", "duration_global_mean")
BOOTSTRAP_SEED = 7031
BOOTSTRAP_REPLICATES = 2000
SOURCE = ROOT / "results/quantum_heldout"
CODE = ("scripts/run_stage_d_components.py", "tests/test_stage_d_components.py",
        "docs/stage_d_components_protocol.md", "scripts/run_quantum_heldout.py",
        "src/quantum_nilm/evaluation.py", "src/quantum_nilm/multistate.py",
        "src/quantum_nilm/heldout_data.py")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def model_values(frozen):
    model = frozen["models"]["multistate"]
    chosen = frozen["selected_parameters"]["multistate_compressed"]
    require(chosen["family"] == "multistate" and chosen["compressed"] is True,
            "Original selection is not compressed multistate")
    rho = float(chosen["rho"])
    require(rho == .1, "Original rho differs from the approved frozen .1 setting")
    levels = [np.asarray(x, dtype=float) for x in model["levels_w"]]
    ranges = np.asarray(model["ranges_w"], dtype=float)
    require(len(levels) == 4 and ranges.shape == (4,), "Expected three appliances plus background")
    require(np.all(np.isfinite(ranges)) and np.all(ranges > 0), "Invalid power ranges")
    require(np.allclose(ranges, [np.ptp(x) for x in levels], rtol=0, atol=1e-9),
            "Frozen ranges disagree with frozen levels")
    return levels, ranges, rho


def arm_penalties(ranges, rho, arm):
    require(arm in ARMS, "Unknown frozen arm")
    normalized = rho * np.asarray(ranges, dtype=float)**2
    if arm == "duration_global_max":
        return np.full_like(normalized, np.max(normalized))
    if arm == "duration_global_mean":
        return np.full_like(normalized, np.mean(normalized))
    return normalized


def assemble_runs(window):
    """Reassemble the frozen two-interval chunks without joining missing-data runs."""
    runs, cursor = [], 0
    for index, chunk in enumerate(window["chunks"]):
        weights = np.asarray(chunk["weights"])
        aggregate = np.asarray(chunk["aggregate"], dtype=float)
        require(chunk["chunk"] == index and chunk["block_start"] == cursor,
                "Chunk sequence is noncontiguous or duplicated")
        require(weights.ndim == 1 and len(weights) in (1, 2) and weights.dtype.kind in "iu"
                and np.all(weights > 0) and aggregate.shape == weights.shape
                and np.all(np.isfinite(aggregate)), "Invalid frozen chunk values")
        require(chunk["block_stop"] - cursor == int(weights.sum()), "Durations do not cover chunk")
        if chunk["reset"]:
            require(not runs or chunk["run"] != runs[-1]["run"], "Reset repeated inside a run")
            runs.append({"run": chunk["run"], "block_start": cursor,
                         "aggregate": [], "weights": []})
        else:
            require(runs and chunk["run"] == runs[-1]["run"], "Missing reset at changed run")
        runs[-1]["aggregate"].extend(aggregate.tolist())
        runs[-1]["weights"].extend(weights.tolist())
        runs[-1]["block_stop"] = chunk["block_stop"]
        cursor = chunk["block_stop"]
    require(runs and cursor == window["blocks"], "Frozen chunks do not cover every valid block")
    return runs


def direct_objective(aggregate, weights, states, levels, penalties):
    states = np.asarray(states, dtype=int)
    prediction = np.column_stack([values[states[:, i]] for i, values in enumerate(levels)])
    reconstruction = float(np.sum(np.asarray(weights) * (np.asarray(aggregate) - prediction.sum(axis=1))**2))
    switches = np.sum(states[1:] != states[:-1], axis=0).astype(int)
    transition = float(np.asarray(penalties) @ switches)
    return reconstruction + transition, reconstruction, transition, switches


def infer_window(window, levels, ranges, rho, arm):
    """No reference values or target error enter inference."""
    penalties = arm_penalties(ranges, rho, arm)
    original_penalties = rho * ranges**2
    records = []
    started = perf_counter()
    for run in assemble_runs(window):
        actual_weights = np.asarray(run["weights"], dtype=int)
        fit_weights = np.ones_like(actual_weights) if arm == "unit_normalized" else actual_weights
        result = solve_multistate_temporal_exact(np.asarray(run["aggregate"]), levels, penalties, fit_weights)
        native, reconstruction, transitions, switches = direct_objective(
            run["aggregate"], fit_weights, result.states, levels, penalties)
        common, common_reconstruction, common_transitions, _ = direct_objective(
            run["aggregate"], actual_weights, result.states, levels, original_penalties)
        require(np.isclose(native, result.energy, rtol=1e-11, atol=1e-5), "Direct native objective differs from DP")
        records.append({"run": run["run"], "block_start": run["block_start"],
                        "block_stop": run["block_stop"], "states": result.states.tolist(),
                        "native_segment_objective": native, "native_reconstruction": reconstruction,
                        "native_transition": transitions, "common_segment_objective": common,
                        "common_segment_reconstruction": common_reconstruction,
                        "common_transition": common_transitions, "switch_counts": switches.tolist(),
                        "solver_wall_time_s": result.wall_time_s})
    return {"window_id": window["window"]["id"], "arm": arm, "runs": records,
            "inference_wall_time_s": perf_counter() - started,
            "solver_wall_time_s": sum(r["solver_wall_time_s"] for r in records)}


def expand_trace(window, trace, levels):
    runs = assemble_runs(window)
    require(len(runs) == len(trace["runs"]), "Incomplete trace runs")
    prediction = np.full((window["blocks"], len(levels)), np.nan)
    means = np.full(window["blocks"], np.nan)
    for run, record in zip(runs, trace["runs"]):
        require(all(run[k] == record[k] for k in ("run", "block_start", "block_stop")), "Trace run mismatch")
        states = np.asarray(record["states"])
        require(states.shape == (len(run["weights"]), len(levels)) and states.dtype.kind in "iu",
                "Invalid state trajectory")
        require(all(np.all((states[:, i] >= 0) & (states[:, i] < len(values)))
                    for i, values in enumerate(levels)), "Category outside frozen levels")
        powers = np.column_stack([values[states[:, i]] for i, values in enumerate(levels)])
        part = slice(run["block_start"], run["block_stop"])
        prediction[part] = np.repeat(powers, run["weights"], axis=0)
        means[part] = np.repeat(run["aggregate"], run["weights"])
    require(np.all(np.isfinite(prediction)) and np.all(np.isfinite(means)), "Incomplete expanded coverage")
    return prediction, means


def score_trace(window, trace, loaded, levels, thresholds):
    prediction, means = expand_trace(window, trace, levels)
    require(len(loaded["timestamps"]) == window["blocks"], "Source coverage changed")
    actual_runs = [(p.start, p.stop) for p in contiguous_slices(loaded["timestamps"])]
    require(actual_runs == [(r["block_start"], r["block_stop"]) for r in trace["runs"]],
            "Archived runs do not respect source-data gaps")
    residual = float(np.sum((loaded["values"][:, 0] - means)**2))
    raw_reconstruction = float(np.sum((loaded["values"][:, 0] - prediction.sum(axis=1))**2))
    segment_reconstruction = sum(r["common_segment_reconstruction"] for r in trace["runs"])
    require(np.isclose(raw_reconstruction, segment_reconstruction + residual, rtol=1e-10, atol=1e-4),
            "Compressed means plus residual do not equal raw-block reconstruction")
    common_transition = sum(r["common_transition"] for r in trace["runs"])
    return {"window_id": window["window"]["id"], "model": trace["arm"], "seed": None,
            "blocks": window["blocks"], "runs": len(trace["runs"]),
            "segments": sum(len(r["states"]) for r in trace["runs"]),
            "appliances": score_power(loaded["values"][:, 1:], prediction[:, :3], loaded["timestamps"], thresholds, CHANNELS),
            "raw_aggregate_mae_w": float(np.mean(np.abs(loaded["values"][:, 0] - prediction.sum(axis=1)))),
            "native_segment_objective": sum(r["native_segment_objective"] for r in trace["runs"]),
            "common_segment_objective": sum(r["common_segment_objective"] for r in trace["runs"]),
            "compression_residual_constant": residual, "raw_block_reconstruction": raw_reconstruction,
            "common_full_block_objective": raw_reconstruction + common_transition,
            "switch_counts": np.sum([r["switch_counts"] for r in trace["runs"]], axis=0).tolist(),
            "solver_wall_time_s": trace["solver_wall_time_s"],
            "inference_wall_time_s": trace["inference_wall_time_s"]}


def pooled(records):
    appliances = pool_window_metrics(records, CHANNELS)
    blocks = sum(r["blocks"] for r in records)
    return {"windows": len(records), "blocks": blocks, "runs": sum(r["runs"] for r in records),
            "segments": sum(r["segments"] for r in records), "appliances": appliances,
            "macro_appliance_mae_w": float(np.mean([a["mae_w"] for a in appliances.values()])),
            "aggregate_mae_w": sum(r["raw_aggregate_mae_w"] * r["blocks"] for r in records) / blocks,
            "native_segment_objective": sum(r["native_segment_objective"] for r in records),
            "common_full_block_objective": sum(r["common_full_block_objective"] for r in records),
            "compression_residual_constant": sum(r["compression_residual_constant"] for r in records),
            "switch_counts": np.sum([r["switch_counts"] for r in records], axis=0).tolist(),
            "solver_wall_time_s": sum(r["solver_wall_time_s"] for r in records),
            "inference_wall_time_s": sum(r["inference_wall_time_s"] for r in records)}


def paired_bootstrap(left, right, *, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES):
    a, b = {r["window_id"]: r for r in left}, {r["window_id"]: r for r in right}
    require(len(a) == len(left) and len(b) == len(right) and a.keys() == b.keys(), "Unpaired or duplicate windows")
    differences, denominators = [], []
    for name in sorted(a):
        require(a[name]["blocks"] == b[name]["blocks"], "Paired block counts differ")
        differences.append(sum(a[name]["appliances"][c]["absolute_error_sum_w"]
                               - b[name]["appliances"][c]["absolute_error_sum_w"] for c in CHANNELS))
        denominators.append(3 * a[name]["blocks"])
    difference, denominator = np.asarray(differences), np.asarray(denominators)
    draws = np.random.default_rng(seed).integers(0, len(a), size=(replicates, len(a)))
    estimates = difference[draws].sum(axis=1) / denominator[draws].sum(axis=1)
    return {"difference_w": float(difference.sum() / denominator.sum()),
            "paired_window_bootstrap_95_interval_w": np.quantile(estimates, [.025, .975]).tolist(),
            "seed": seed, "replicates": replicates, "windows": len(a),
            "definition": "left minus right macro MAE; ratio of paired resampled error sums to three times resampled block counts",
            "interpretation": "descriptive within-home post-exposure interval, not multiplicity-adjusted or a cross-home result"}


def prepare(output):
    require(not output.exists(), "Use a new output directory; existing experiments are immutable")
    check_inputs(SOURCE)
    config, frozen = read(SOURCE / "protocol.json"), read(SOURCE / "model.json")
    _, ranges, rho = model_values(frozen)
    inputs = read(SOURCE / "test_inputs.json")
    require(len(inputs) == 30 and sum(w["blocks"] for w in inputs) == 86396, "Original cohort changed")
    expected = read(SOURCE / "simulation/summary.json")["pooled_results"]["exact_full/None"]
    require(abs(expected["macro_appliance_mae_w"] - 26.0414278901) < 1e-8, "Original baseline changed")
    original = Path(config["original_dir"])
    artifacts = [SOURCE / p for p in ("protocol.json", "model.json", "test_inputs.json", "input_hashes.json",
                 "simulation/summary.json", "simulation/test_windows.json")] + [original / "data_quality.json"]
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "Stage D fixed-model duration/penalty ablations; original exposed R1Hz test cohort",
        "source_directory": str(SOURCE), "source": config["source"],
        "source_artifacts_sha256": {str(p): sha(p) for p in artifacts},
        "source_code_sha256": {p: sha(ROOT / p) for p in CODE},
        "arms": list(ARMS), "rho": rho,
        "penalties_by_arm": {arm: arm_penalties(ranges, rho, arm).tolist() for arm in ARMS},
        "primary_metric": "pooled macro MAE of dryer/fridge/vacuum on all 86396 valid 30-second blocks",
        "cohort_windows": [w["window"]["id"] for w in inputs], "windows": 30, "blocks": 86396,
        "duration_rule": "unit_normalized fits one per compressed interval; all other arms fit actual durations; all expand using actual durations",
        "solver": "exact full-valid-run categorical temporal DP; never transition across a window boundary or missing-data gap",
        "cross_score": "original actual-duration reconstruction and channel-normalized penalties, plus within-segment raw-block residual constant",
        "native_objective": "separately reported segment-mean objective under each arm; not directly comparable across arms",
        "baseline_gate": {"expected_macro_mae_w": expected["macro_appliance_mae_w"], "absolute_tolerance_w": 1e-8,
                          "compare_per_window_appliance_and_aggregate_errors": True,
                          "must_pass_before_ablated_arms": True},
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "unit": "paired fixed 24-hour window",
                      "estimator": "ratio of resampled absolute-error sums to three times resampled block counts",
                      "contrasts": [[arm, ARMS[0]] for arm in ARMS[1:]] + [[ARMS[2], ARMS[3]]]},
        "retuning": False, "hardware_jobs": False,
        "limits": ["retrospective same-home fixed-setting ablations, not a fresh blind test",
                   "global maximum penalty changes total penalty magnitude; global mean is a scale-matched sensitivity control",
                   "unit weights change reconstruction-to-transition balance; no arm receives fresh tuning",
                   "proxy thresholds are not verified physical ON labels",
                   "these classical modeling ablations are not quantum advantage or a mixer comparison"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}}
    output.mkdir(parents=True)
    save(output / "protocol.json", protocol)
    for relative in CODE:
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    save(output / "frozen_inputs.json", {"config": config, "model": frozen, "windows": inputs,
         "source_quality": read(original / "data_quality.json"), "expected_baseline": expected})
    save(output / "freeze.json", {"protocol_sha256": sha(output / "protocol.json"),
                                  "frozen_inputs_sha256": sha(output / "frozen_inputs.json")})
    print(json.dumps({"status": "frozen_before_inference", "protocol_sha256": sha(output / "protocol.json")}), flush=True)


def check_frozen(output):
    freeze = read(output / "freeze.json")
    require(sha(output / "protocol.json") == freeze["protocol_sha256"], "Protocol changed")
    require(sha(output / "frozen_inputs.json") == freeze["frozen_inputs_sha256"], "Frozen inputs changed")
    protocol = read(output / "protocol.json")
    for path, expected in protocol["source_artifacts_sha256"].items():
        require(sha(path) == expected, f"Source artifact changed: {path}")
    for path, expected in protocol["source_code_sha256"].items():
        require(sha(ROOT / path) == expected and sha(output / "source_snapshot" / path) == expected,
                f"Frozen implementation changed: {path}")
    check_inputs(SOURCE)
    return protocol, read(output / "frozen_inputs.json"), freeze["protocol_sha256"]


def baseline_gate(records, protocol, expected):
    actual = pooled(records)
    require(actual["windows"] == 30 and actual["blocks"] == 86396, "Baseline coverage incomplete")
    delta = actual["macro_appliance_mae_w"] - protocol["baseline_gate"]["expected_macro_mae_w"]
    require(abs(delta) <= protocol["baseline_gate"]["absolute_tolerance_w"], "Baseline MAE reproduction gate failed")
    original_rows = {r["window_id"]: r for r in read(SOURCE / "simulation/test_windows.json") if r["model"] == "exact_full"}
    for row in records:
        prior = original_rows[row["window_id"]]
        require(row["blocks"] == prior["blocks"], "Baseline per-window coverage changed")
        for channel in CHANNELS:
            require(np.isclose(row["appliances"][channel]["absolute_error_sum_w"],
                               prior["appliances"][channel]["absolute_error_sum_w"], rtol=1e-12, atol=1e-7),
                    "Baseline per-window appliance reproduction failed")
        require(np.isclose(row["raw_aggregate_mae_w"], prior["raw_aggregate_mae_w"], rtol=1e-12, atol=1e-9),
                "Baseline per-window aggregate reproduction failed")
    return {"status": "passed", "macro_mae_w": actual["macro_appliance_mae_w"], "difference_w": delta,
            "windows": actual["windows"], "blocks": actual["blocks"], "per_window_checks": True}


def run(output):
    protocol, frozen, protocol_hash = check_frozen(output)
    levels, ranges, rho = model_values(frozen["model"])
    references, all_records = {}, {}
    started = perf_counter()
    for arm in ARMS:
        if arm != ARMS[0]:
            require(read(output / "baseline_gate.json")["status"] == "passed", "Baseline must pass first")
        rows = []
        for number, window in enumerate(frozen["windows"]):
            name = window["window"]["id"]
            trace_path, row_path = output / "traces" / arm / f"{name}.json", output / "records" / arm / f"{name}.json"
            if row_path.exists():
                row = read(row_path)
                require(row["protocol_sha256"] == protocol_hash and row["trace_sha256"] == sha(trace_path), "Checkpoint provenance changed")
            else:
                if trace_path.exists():
                    trace = read(trace_path)
                    require(trace["protocol_sha256"] == protocol_hash, "Trace protocol differs")
                else:
                    trace = infer_window(window, levels, ranges, rho, arm)
                    trace["protocol_sha256"] = protocol_hash
                    save(trace_path, trace)
                # Inference has no reference argument; save its trace before scoring.
                if name not in references:
                    references[name] = checked_window(frozen["config"], frozen["source_quality"], window["window"], ["main", *CHANNELS])
                row = score_trace(window, trace, references[name], levels, frozen["config"]["state_proxy_thresholds_w"])
                row.update({"protocol_sha256": protocol_hash, "trace_sha256": sha(trace_path),
                            "source_window_sha256": references[name]["source_window_sha256"]})
                save(row_path, row)
            rows.append(row)
            if (number + 1) % 10 == 0:
                print(f"{arm}: {number + 1}/30 fixed windows", flush=True)
        all_records[arm] = rows
        if arm == ARMS[0]:
            gate = baseline_gate(rows, protocol, frozen["expected_baseline"])
            if not (output / "baseline_gate.json").exists():
                save(output / "baseline_gate.json", gate)
            print(json.dumps({"baseline_gate": gate}), flush=True)
    check_frozen(output)
    if not (output / "execution.json").exists():
        save(output / "execution.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
             "status": "complete", "protocol_sha256": protocol_hash, "records": 120,
             "current_process_wall_time_s": perf_counter() - started,
             "timing_note": "Current process may resume earlier records; per-record inference timings are authoritative"})
    return summarize(output)


def summarize(output):
    protocol, frozen, protocol_hash = check_frozen(output)
    records, summaries, record_hashes = {}, {}, {}
    for arm in ARMS:
        paths = [output / "records" / arm / f"{window['window']['id']}.json" for window in frozen["windows"]]
        require(set((output / "records" / arm).glob("*.json")) == set(paths), "Unexpected or missing result records")
        records[arm] = [read(path) for path in paths]
        for path, row in zip(paths, records[arm]):
            require(row["model"] == arm and row["protocol_sha256"] == protocol_hash, "Record identity mismatch")
            require(sha(output / "traces" / arm / f"{row['window_id']}.json") == row["trace_sha256"], "Trace changed after scoring")
            record_hashes[str(path.relative_to(output))] = sha(path)
        summaries[arm] = pooled(records[arm])
    gate = baseline_gate(records[ARMS[0]], protocol, frozen["expected_baseline"])
    for arm in ARMS:
        summary = summaries[arm]
        summary["common_objective_gap_from_baseline"] = summary["common_full_block_objective"] - summaries[ARMS[0]]["common_full_block_objective"]
        require(summary["common_objective_gap_from_baseline"] >= -1e-3, "An ablated prediction beats certified common-objective baseline")
    comparisons = [{"left": left, "right": right, **paired_bootstrap(records[left], records[right])}
                   for left, right in protocol["bootstrap"]["contrasts"]]
    result = {"status": "complete", "protocol_sha256": protocol_hash, "baseline_gate": gate,
              "arms": summaries, "paired_comparisons": comparisons, "record_sha256": record_hashes,
              "limits": protocol["limits"]}
    if (output / "summary.json").exists():
        require(read(output / "summary.json") == result, "Existing summary differs from deterministic reconstruction")
    else:
        save(output / "summary.json", result)
    print(json.dumps({arm: row["macro_appliance_mae_w"] for arm, row in summaries.items()}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "summarize"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage_d/components/run_001")
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize}[args.mode](args.output.resolve())


if __name__ == "__main__":
    main()
