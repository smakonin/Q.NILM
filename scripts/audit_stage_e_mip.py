#!/usr/bin/env python3
"""Independent dense-Bellman, solver-bound and raw-metric audit for Stage E.

Only the established R1Hz source reader is reused, solely for extraction and
row hashes. No experiment inference, objective, scoring or bootstrap helper
is imported. The audit never modifies experimental artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quantum_nilm.heldout_data import read_block_window

CHANNELS = ("dryr", "frdg", "vacu")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def close(actual, expected, label, *, atol=1e-6, rtol=1e-10):
    if expected is None:
        require(actual is None, f"{label}: expected missing value")
        return
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b)),
            f"{label}: nonfinite value or shape mismatch")
    require(np.all(np.abs(a - b) <= atol + rtol * np.abs(b)), f"{label}: numeric mismatch")


def direct_energy(aggregate, weights, states, levels, penalties):
    categories = np.asarray(states)
    measured, duration = np.asarray(aggregate, dtype=float), np.asarray(weights, dtype=float)
    require(categories.shape == (len(measured), len(levels)) and categories.dtype.kind in "iu",
            "Decoded states must be an integer interval-by-channel array")
    require(duration.shape == measured.shape and np.all(duration > 0), "Invalid objective duration")
    require(np.all(np.isfinite(measured)) and np.all(np.isfinite(duration)), "Nonfinite objective input")
    for channel, values in enumerate(levels):
        require(np.all((categories[:, channel] >= 0) & (categories[:, channel] < len(values))),
                "Decoded category outside frozen model")
    powers = np.array([[levels[i][int(state)] for i, state in enumerate(row)] for row in categories])
    reconstruction = math.fsum(float(w) * (float(y) - math.fsum(row)) ** 2
                               for y, w, row in zip(measured, duration, powers))
    switches = [sum(int(a[i] != b[i]) for a, b in zip(categories[:-1], categories[1:]))
                for i in range(len(levels))]
    transition = math.fsum(float(p) * n for p, n in zip(penalties, switches))
    return {"energy": reconstruction + transition, "reconstruction": reconstruction,
            "transition": transition, "switch_counts": switches, "prediction_w": powers}


def dense_oracle(aggregate, weights, levels, penalties):
    """O(K*S*S) Bellman oracle, not the production separable O(K*N*S) code."""
    states = np.array(list(itertools.product(*(range(len(values)) for values in levels))), dtype=int)
    require(len(states) <= 4096, "Independent dense oracle state cap exceeded")
    powers = np.array([math.fsum(float(levels[i][c]) for i, c in enumerate(row)) for row in states])
    transition = np.zeros((len(states), len(states)))
    for i, penalty in enumerate(penalties):
        transition += float(penalty) * (states[:, i, None] != states[None, :, i])
    emissions = np.array([float(w) * (float(y) - powers) ** 2 for y, w in zip(aggregate, weights)])
    costs = emissions[0].copy()
    for emission in emissions[1:]:
        costs = np.min(costs[:, None] + transition, axis=0) + emission
    minima = np.min(emissions, axis=1)
    shifted = emissions - minima[:, None]
    scale = max(1.0, float(np.max(shifted)) / 10000, float(np.max(penalties)) / 10000)
    return {"optimum": float(np.min(costs)), "objective_offset": float(np.sum(minima)),
            "objective_scale": scale, "joint_states": len(states)}


def audit_solver(solver, direct, oracle):
    """Check restored bounds without equating tied trajectories."""
    optimum = oracle["optimum"]
    tolerance = 1e-4 + 1e-8 * max(1.0, abs(optimum))
    close(solver["objective_offset"], oracle["objective_offset"], "MILP offset")
    close(solver["objective_scale"], oracle["objective_scale"], "MILP scale")
    require(solver["states_per_segment"] == oracle["joint_states"], "MILP joint-state count")
    # The runner preserves every field exposed by the frozen solver API.
    for raw, restored in (("raw_solver_objective", "solver_objective"),
                          ("raw_solver_dual_bound", "dual_bound")):
        require(raw in solver, f"Missing raw {raw} for restoration audit")
        expected = None if solver[raw] is None else solver["objective_offset"] + solver["objective_scale"] * solver[raw]
        close(solver[restored], expected, f"Restored {restored}")
    bound = solver["dual_bound"]
    if bound is not None:
        require(math.isfinite(bound) and bound <= optimum + tolerance,
                "Restored lower bound exceeds independent optimum")
    if direct is None:
        require(not solver["incumbent_feasible"] and solver["energy"] is None and solver["states"] is None,
                "Missing trajectory masquerades as validated incumbent")
        require(solver["status_code"] != 0, "Optimal solver status without a validated incumbent")
        return {"incumbent": False, "gap_to_oracle": None, "certified_relative_gap": None}
    require(solver["incumbent_feasible"], "Decoded path lacks solver validation")
    for field in ("max_integrality_violation", "max_bound_violation", "max_constraint_violation"):
        require(solver[field] is not None and 0 <= solver[field] <= 1e-6,
                f"Accepted incumbent violates {field}")
    close(solver["energy"], direct["energy"], "Decoded MILP energy")
    require(direct["energy"] >= optimum - tolerance, "Decoded MILP path beats independent optimum")
    require(solver["solver_objective"] is not None and direct["energy"] <= solver["solver_objective"] + tolerance,
            "Decoded path exceeds restored incumbent objective")
    if solver["status_code"] == 0:
        require(solver["success"], "Optimal status disagrees with solver success")
        close(solver["solver_objective"], direct["energy"], "Optimal incumbent objective", atol=tolerance, rtol=0)
    if bound is not None:
        gap = direct["energy"] - bound
        close(solver["absolute_gap"], gap, "Restored absolute bound gap")
        close(solver["relative_gap"], gap / max(1.0, abs(direct["energy"])), "Restored relative bound gap")
        require(solver["bound_validation"] in ("passed", "negative_within_roundoff"),
                "Solver reports a materially invalid incumbent bound")
        # This label refers to the solver's archived arithmetic, not the
        # independent fsum reconstruction's last-bit rounding.
        archived_gap = solver["energy"] - bound
        roundoff = 128 * np.finfo(float).eps * max(1., abs(solver["energy"]), abs(bound))
        expected_label = "passed" if archived_gap >= 0 else "negative_within_roundoff" if archived_gap >= -roundoff else "dual_bound_exceeds_incumbent"
        require(solver["bound_validation"] == expected_label, "Bound-validation label hides its signed gap")
    else:
        require(solver["absolute_gap"] is None and solver["relative_gap"] is None, "Gap without a lower bound")
    return {"incumbent": True, "gap_to_oracle": max(0.0, direct["energy"] - optimum),
            "certified_relative_gap": max(0.0, direct["energy"] - optimum) / max(1.0, abs(optimum))}


def reconstruct_runs(window):
    """Assemble immutable intervals independently of runner assembly helpers."""
    runs, cursor = [], 0
    for number, chunk in enumerate(window["chunks"]):
        require(chunk["chunk"] == number and chunk["block_start"] == cursor, "Missing or reordered chunk")
        weights = np.asarray(chunk["weights"])
        require(weights.ndim == 1 and weights.dtype.kind in "iu" and np.all(weights > 0), "Invalid chunk duration")
        require(len(weights) == len(chunk["aggregate"]) and chunk["block_stop"] == cursor + sum(weights),
                "Chunk duration/coverage mismatch")
        if chunk["reset"]:
            require(not runs or runs[-1]["run"] != chunk["run"], "Repeated reset inside a run")
            runs.append({"run": chunk["run"], "block_start": cursor, "weights": [], "aggregate": []})
        require(runs and runs[-1]["run"] == chunk["run"], "A changed run lacks a reset")
        runs[-1]["weights"].extend(chunk["weights"])
        runs[-1]["aggregate"].extend(chunk["aggregate"])
        runs[-1]["block_stop"] = chunk["block_stop"]
        cursor = chunk["block_stop"]
    require(cursor == window["blocks"], "Runs do not cover all selected blocks")
    return runs


def check_compression(window, timestamps, mains, threshold):
    """Re-detect aggregate-only events without the production event helper."""
    timestamps, mains = np.asarray(timestamps), np.asarray(mains)
    require(timestamps.shape == mains.shape == (window["blocks"],), "Raw compression coverage mismatch")
    require(np.all(np.diff(timestamps) > 0) and np.all(np.isfinite(mains)), "Invalid raw compression input")
    runs = reconstruct_runs(window)
    starts = [0] + [i for i in range(1, len(timestamps)) if timestamps[i] != timestamps[i - 1] + 30]
    stops = starts[1:] + [len(timestamps)]
    require(len(starts) == len(runs), "Raw data gaps disagree with frozen reset runs")
    for run, start, stop in zip(runs, starts, stops):
        require((run["block_start"], run["block_stop"]) == (start, stop), "Frozen run crosses a raw gap")
        values = mains[start:stop]
        edges = []
        candidates = [i for i in range(1, len(values)) if abs(values[i] - values[i - 1]) >= threshold]
        groups = []
        for index in candidates:
            if not groups or index != groups[-1][-1] + 1:
                groups.append([])
            groups[-1].append(index)
        for group in groups:
            edges.append(max(group, key=lambda i: abs(values[i] - values[i - 1])))
        boundaries = [0, *edges, len(values)]
        durations = [b - a for a, b in zip(boundaries[:-1], boundaries[1:])]
        means = [math.fsum(map(float, values[a:b])) / (b - a) for a, b in zip(boundaries[:-1], boundaries[1:])]
        require(durations == run["weights"], "Frozen event durations do not reproduce from mains")
        close(run["aggregate"], means, "Frozen event means", atol=1e-8, rtol=1e-12)
    return runs


def basic_metrics(reference, predicted):
    truth, estimate = np.asarray(reference), np.asarray(predicted)
    require(truth.shape == estimate.shape and truth.ndim == 2 and truth.shape[1] == 3 and len(truth) > 0,
            "Raw appliance metric shape mismatch")
    require(np.all(np.isfinite(truth)) and np.all(np.isfinite(estimate)), "Nonfinite raw appliance metric")
    absolute = [math.fsum(abs(float(a) - float(b)) for a, b in zip(truth[:, i], estimate[:, i])) for i in range(3)]
    return {"blocks": len(truth), "absolute_error_sum_w": absolute,
            "mae_w": [x / len(truth) for x in absolute], "macro_mae_w": math.fsum(absolute) / (3 * len(truth))}


def classification(tp, tn, fp, fn):
    n = tp + tn + fp + fn
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "bit_accuracy": (tp + tn) / n if n else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "mcc": (tp * tn - fp * fn) / math.sqrt(denominator) if denominator else None}


def raw_appliance_metrics(reference, predicted, timestamps, thresholds):
    """Independent score arithmetic, including the frozen high-state proxies."""
    basic = basic_metrics(reference, predicted)
    result = {}
    for i, name in enumerate(CHANNELS):
        truth = np.asarray(reference)[:, i]
        error = np.asarray(predicted)[:, i] - truth
        actual = truth > thresholds[i]
        estimate = np.asarray(predicted)[:, i] > thresholds[i]
        tp = int(np.count_nonzero(actual & estimate))
        tn = int(np.count_nonzero(~actual & ~estimate))
        fp = int(np.count_nonzero(~actual & estimate))
        fn = int(np.count_nonzero(actual & ~estimate))
        found = spurious = missed = 0
        starts = [0] + [j for j in range(1, len(timestamps)) if timestamps[j] != timestamps[j - 1] + 30]
        for start, stop in zip(starts, starts[1:] + [len(timestamps)]):
            for sign in (-1, 1):
                observed_events = [j for j in range(start + 1, stop) if int(actual[j]) - int(actual[j - 1]) == sign]
                inferred_events = [j for j in range(start + 1, stop) if int(estimate[j]) - int(estimate[j - 1]) == sign]
                a = b = 0
                while a < len(observed_events) and b < len(inferred_events):
                    if inferred_events[b] < observed_events[a] - 1:
                        spurious += 1
                        b += 1
                    elif observed_events[a] < inferred_events[b] - 1:
                        missed += 1
                        a += 1
                    else:
                        found += 1
                        a += 1
                        b += 1
                missed += len(observed_events) - a
                spurious += len(inferred_events) - b
        signed = math.fsum(map(float, error))
        squared = math.fsum(float(x) ** 2 for x in error)
        power = math.fsum(map(float, truth))
        power_squared = math.fsum(float(x) ** 2 for x in truth)
        event_denominator = 2 * found + spurious + missed
        result[name] = {"n_blocks": basic["blocks"], "mae_w": basic["mae_w"][i],
            "absolute_error_sum_w": basic["absolute_error_sum_w"][i], "squared_error_sum_w2": squared,
            "true_power_sum_w": power, "true_squared_power_sum_w2": power_squared,
            "signed_power_error_sum_w": signed, "energy_error_wh": signed / 120,
            "normalized_disaggregation_error": squared / power_squared if power_squared else None,
            "signal_aggregate_error": abs(signed) / abs(power) if power else None,
            "active_blocks": int(np.count_nonzero(actual)), "state": classification(tp, tn, fp, fn),
            "event_counts": {"tp": found, "fp": spurious, "fn": missed},
            "event_f1": 2 * found / event_denominator if event_denominator else None}
    return result


def check_mapping(actual, expected, label):
    require(set(actual) == set(expected), f"{label}: field coverage mismatch")
    for field, value in expected.items():
        if isinstance(value, dict):
            check_mapping(actual[field], value, f"{label}/{field}")
        elif value is None or isinstance(value, (bool, str, int)):
            require(actual[field] == value, f"{label}/{field}: value mismatch")
        else:
            close(actual[field], value, f"{label}/{field}")


def paired_interval(left, right, *, seed=9301, replicates=2000):
    require(set(left) == set(right) and left, "Unpaired bootstrap cohort")
    names = sorted(left)
    denominators, differences = [], []
    for name in names:
        require(left[name]["blocks"] == right[name]["blocks"], "Paired bootstrap coverage mismatch")
        denominators.append(3 * left[name]["blocks"])
        differences.append(math.fsum(left[name]["absolute_error_sum_w"]) - math.fsum(right[name]["absolute_error_sum_w"]))
    draws = np.random.default_rng(seed).integers(0, len(names), size=(replicates, len(names)))
    d, n = np.asarray(differences), np.asarray(denominators)
    distribution = d[draws].sum(axis=1) / n[draws].sum(axis=1)
    return {"difference_w": math.fsum(differences) / sum(denominators),
            "paired_window_bootstrap_95_interval_w": np.quantile(distribution, [0.025, 0.975]).tolist()}


def candidate_key(records, presolve):
    absent = sum(not r["incumbent"] for r in records)
    return (absent, sum(not r["solver_optimal"] for r in records),
            math.inf if absent else math.fsum(r["certified_relative_gap"] for r in records),
            math.fsum(r["formulation_plus_solve_time_s"] for r in records), 0 if presolve else 1)


def digest_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def audit_trace(archive, window, trace, levels, penalties, oracles, binding, *, label, validation=False):
    runs = reconstruct_runs(window)
    name = window["window"]["id"]
    require(trace["binding"] == binding and trace["window_id"] == name, "Trace provenance/identity mismatch")
    require(trace["fresh_compression_sha256"] == digest_value(window), "Trace fresh-input digest mismatch")
    require(len(trace["runs"]) == len(runs) == len(oracles), "Missing or extra run outcome")
    prediction = np.full((window["blocks"], len(levels)), np.nan)
    means = np.empty(window["blocks"])
    folder = archive / ("tuning/run_records" if validation else "run_records") / label / name
    expected_leaves = {f"run_{i:03d}.json" for i in range(len(runs))}
    require(set(trace["checkpoint_sha256"]) == expected_leaves, "Checkpoint hash coverage differs")
    require({p.name for p in folder.glob("run_*.json") if not p.name.endswith(".intent.json")} == expected_leaves,
            "Missing or extra run checkpoint")
    evaluated = []
    for index, (run, row, oracle) in enumerate(zip(runs, trace["runs"], oracles)):
        for field in ("run", "block_start", "block_stop"):
            require(row[field] == run[field], "Trace run identity differs from frozen input")
        leaf = f"run_{index:03d}.json"
        checkpoint = folder / leaf
        require(sha(checkpoint) == trace["checkpoint_sha256"][leaf], "Checkpoint hash differs")
        require(json.loads(checkpoint.read_text()) == row, "Trace differs from saved solver checkpoint")
        require(row["binding"] == binding and row["input_sha256"] == digest_value(run), "Checkpoint input provenance mismatch")
        intent = json.loads(checkpoint.with_suffix(".intent.json").read_text())
        require(intent["binding"] == binding and intent["input_sha256"] == row["input_sha256"], "Solve intent input/binding mismatch")
        part = slice(run["block_start"], run["block_stop"])
        means[part] = np.repeat(run["aggregate"], run["weights"])
        direct = None if row["states"] is None else direct_energy(run["aggregate"], run["weights"], row["states"], levels, penalties)
        require(row["incumbent_feasible"] == (direct is not None), "Run incumbent flag differs from trace")
        if direct is not None:
            require(np.array_equal(row["states"], row["solver"]["states"]), "Decoded states differ between trace and solver")
            close(row["solver"]["predicted_power"], direct["prediction_w"], "Solver's decoded powers")
            for field, expected in (("native_segment_objective", direct["energy"]),
                                    ("common_segment_objective", direct["energy"]),
                                    ("native_reconstruction", direct["reconstruction"]),
                                    ("common_segment_reconstruction", direct["reconstruction"]),
                                    ("native_transition", direct["transition"]),
                                    ("common_transition", direct["transition"]),
                                    ("switch_counts", direct["switch_counts"])):
                close(row[field], expected, f"{name}/{label}/{index}/{field}")
            prediction[part] = np.repeat(direct["prediction_w"], run["weights"], axis=0)
        if label == "exact_dp":
            require(direct is not None and row["solver_certified_optimal"], "Exact DP failed or uncertified")
            close(direct["energy"], oracle["optimum"], "Dense/separable DP optimum", atol=1e-4, rtol=1e-10)
            close(row["solver"]["energy"], direct["energy"], "DP returned objective")
            result = {"incumbent": True, "gap_to_oracle": max(0., direct["energy"] - oracle["optimum"]),
                      "certified_relative_gap": 0.}
        else:
            result = audit_solver(row["solver"], direct, oracle)
            expected_optimal = (row["solver"]["status_code"] == 0 and row["solver"]["success"]
                                and row["solver"]["incumbent_feasible"] and row["solver"]["dual_bound"] is not None
                                and row["solver"]["bound_validation"] in ("passed", "negative_within_roundoff"))
            require(row["solver_certified_optimal"] == expected_optimal, "Solver optimality status is overstated")
            k, s, n = len(run["weights"]), oracle["joint_states"], len(levels)
            counts = sum(len(v) for v in levels)
            require(row["solver"]["binary_variables"] == k * s
                    and row["solver"]["continuous_variables"] == (k - 1) * n
                    and row["solver"]["constraints"] == k + (k - 1) * counts
                    and row["solver"]["constraint_nonzeros"] == k * s + (k - 1) * (2 * n * s + counts),
                    "Reported expanded MILP dimensions are wrong")
            close(row["build_time_s"], row["solver"]["build_time_s"], "Formulation time copy")
            close(row["solver_wall_time_s"], row["solver"]["solve_time_s"], "Backend time copy")
        require(row["solver_call_wall_time_s"] >= row["solver"]["wall_time_s"] - 1e-8,
                "Solver call timing excludes measured solver work")
        result.update({"solver_optimal": row["solver_certified_optimal"], "optimum": oracle["optimum"],
                       "energy": None if direct is None else direct["energy"], "row": row, "started_utc": intent["started_utc"],
                       "formulation_plus_solve_time_s": (row["build_time_s"] or 0.) + row["solver_wall_time_s"]})
        evaluated.append(result)
    mask = np.all(np.isfinite(prediction), axis=1)
    require(trace["complete"] == bool(np.all(mask)) and trace["covered_blocks"] == int(mask.sum())
            and trace["blocks"] == window["blocks"], "Trace coverage mismatch")
    for field in ("compression_wall_time_s", "expansion_wall_time_s", "inference_wall_time_s", "solver_wall_time_s"):
        require(math.isfinite(trace[field]) and trace[field] >= 0, f"Invalid {field}")
    close(trace["inference_wall_time_s"], trace["compression_wall_time_s"] + trace["expansion_wall_time_s"]
          + math.fsum(r["solver_call_wall_time_s"] for r in trace["runs"]), "Inference timing sum", atol=1e-8)
    close(trace["solver_wall_time_s"], math.fsum(r["solver_wall_time_s"] for r in trace["runs"]), "Backend timing sum", atol=1e-8)
    return {"prediction": prediction, "means": means, "mask": mask, "runs": evaluated}


def pooled_appliances(rows):
    pooled = {}
    for channel in CHANNELS:
        values = [row["appliances"][channel] for row in rows]
        n = sum(v["n_blocks"] for v in values)
        sums = {key: math.fsum(v[key] for v in values) for key in (
            "absolute_error_sum_w", "squared_error_sum_w2", "signed_power_error_sum_w",
            "true_power_sum_w", "true_squared_power_sum_w2")}
        events = {key: sum(v["event_counts"][key] for v in values) for key in ("tp", "fp", "fn")}
        counts = {key: sum(v["state"][key] for v in values) for key in ("tp", "tn", "fp", "fn")}
        denominator = 2 * events["tp"] + events["fp"] + events["fn"]
        pooled[channel] = {"n_blocks": n, "mae_w": sums["absolute_error_sum_w"] / n,
            "rmse_w": math.sqrt(sums["squared_error_sum_w2"] / n),
            "energy_error_wh": sums["signed_power_error_sum_w"] / 120,
            "signal_aggregate_error": abs(sums["signed_power_error_sum_w"]) / abs(sums["true_power_sum_w"]) if sums["true_power_sum_w"] else None,
            "normalized_disaggregation_error": sums["squared_error_sum_w2"] / sums["true_squared_power_sum_w2"] if sums["true_squared_power_sum_w2"] else None,
            "active_blocks": sum(v["active_blocks"] for v in values),
            "active_windows": sum(v["active_blocks"] > 0 for v in values),
            "state": classification(**counts), "event_counts": events,
            "event_f1": 2 * events["tp"] / denominator if denominator else None}
    return pooled


def audit(archive):
    archive = Path(archive).resolve()
    started = time.perf_counter()
    observed_hashes = {}

    def read(name):
        path = archive / name
        observed_hashes[str(name)] = sha(path)
        return json.loads(path.read_text())

    protocol, frozen, freeze, validation, selection, candidates, summary = [read(name) for name in (
        "protocol.json", "frozen_inputs.json", "freeze.json", "validation_inputs.json", "selection.json", "candidates.json", "summary.json")]
    protocol_hash = observed_hashes["protocol.json"]
    require(protocol["arms"] == ["milp", "exact_dp"] and protocol["no_reference_inference"] and protocol["no_dp_warm_start"],
            "Unexpected or unsupported experiment design")
    require((protocol["windows"], protocol["blocks"], protocol["runs"], protocol["segments"]) == (30, 86396, 34, 4881),
            "Original cohort scope changed")
    require(protocol["validation_time_limit_s"] == 15 and protocol["test_time_limit_s"] == 30
            and protocol["mip_rel_gap"] == 1e-8, "Frozen solver budgets changed")

    def frozen_check():
        for filename in ("protocol", "frozen_inputs", "validation_inputs"):
            require(sha(archive / f"{filename}.json") == freeze[f"{filename}_sha256"], "Frozen archive changed")
        for path, expected in protocol["source_artifacts_sha256"].items():
            require(sha(path) == expected, f"Original artifact changed: {path}")
        for path, expected in protocol["source_code_sha256"].items():
            require(sha(ROOT / path) == expected == sha(archive / "source_snapshot" / path), f"Frozen source changed: {path}")
        for path, expected in freeze["validation_data_sha256"].items():
            require(sha(archive / path) == expected, f"Frozen validation array changed: {path}")
        for path, expected in observed_hashes.items():
            require(sha(archive / path) == expected, f"Artifact changed during audit: {path}")

    frozen_check()
    for artifact in (selection, candidates, summary):
        require(artifact["protocol_sha256"] == protocol_hash, "Summary/selection protocol binding mismatch")
    require(selection["candidates_sha256"] == observed_hashes["candidates.json"]
            and summary["selection_sha256"] == observed_hashes["selection.json"], "Selection hash mismatch")
    require(selection["selected_before_test_inference"] and not selection["appliance_labels_used"]
            and not candidates["appliance_labels_used"], "Invalid validation-selection provenance")
    for name, expected in candidates["artifact_sha256"].items():
        require(sha(archive / name) == expected, "Validation candidate hash changed")
    for name, expected in summary["record_sha256"].items():
        require(sha(archive / name) == expected, "Scored record hash changed")
    model = frozen["model"]["models"]["multistate"]
    original = Path(protocol["source_directory"])
    require(frozen["model"] == json.loads((original / "model.json").read_text())
            and frozen["config"] == json.loads((original / "protocol.json").read_text())
            and frozen["windows"] == json.loads((original / "test_inputs.json").read_text()),
            "Frozen model/config/test arrays differ from original anchored archive")
    levels = [np.asarray(values, dtype=float) for values in model["levels_w"]]
    require([len(v) for v in levels] == [4, 3, 2, 3] and protocol["rho"] == .1, "Frozen model changed")
    ranges = [float(max(v) - min(v)) for v in levels]
    close(model["ranges_w"], ranges, "Frozen model power ranges")
    penalties = [.1 * value ** 2 for value in ranges]
    close(protocol["penalties"], penalties, "Frozen transition penalties")
    require(model["event_threshold_w"] == protocol["event_threshold_w"], "Frozen event threshold changed")
    expected_validation = sorted([w for w in frozen["config"]["windows"] if w["split"] in ("val", "validation")],
                                 key=lambda w: (w["start_unix"], w["id"]))[:3]
    require([w["window"] for w in validation] == expected_validation == frozen["validation_windows"], "Validation selection changed")
    require(protocol["validation_windows"] == [w["id"] for w in expected_validation], "Validation protocol IDs changed")
    windows = frozen["windows"]
    require(len(windows) == 30 and len({w["window"]["id"] for w in windows}) == 30
            and sum(w["blocks"] for w in windows) == 86396, "Test coverage changed")
    require(sum(len(reconstruct_runs(w)) for w in windows) == 34
            and sum(len(r["weights"]) for w in windows for r in reconstruct_runs(w)) == 4881, "Test runs/intervals changed")
    ids = {w["window"]["id"] for w in windows}
    for arm in ("milp", "exact_dp"):
        for folder, suffix in (("traces", ".json"), ("records", ".json"), ("predictions", ".npz")):
            require({p.stem for p in (archive / folder / arm).glob(f"*{suffix}")} == ids,
                    f"Unexpected or missing {arm}/{folder} window")
    selection_time = datetime.fromisoformat(selection["created_utc"])
    validation_rows = {label: [] for label in ("exact_dp", "presolve_on", "presolve_off")}
    source_hashes = {}
    for window in validation:
        name = window["window"]["id"]
        observed = read_block_window(protocol["source"], window["window"]["start_unix"], window["window"]["end_unix"],
                                     channels=("main",), exclude_markers=("s", "+"))
        require(observed["source_window_sha256"] == window["source_window_sha256"], "Validation raw-source hash mismatch")
        source_hashes[name] = observed["source_window_sha256"]
        runs = check_compression(window, observed["timestamps"], observed["values"][:, 0], protocol["event_threshold_w"])
        with np.load(archive / "validation_data" / f"{name}.npz", allow_pickle=False) as data:
            require(set(data.files) == {"timestamps", "mains_w"}, "Validation archive contains non-input channels")
            require(np.array_equal(data["timestamps"], observed["timestamps"]), "Validation timestamps changed")
            close(data["mains_w"], observed["values"][:, 0], "Validation mains copy")
        oracles = [dense_oracle(r["aggregate"], r["weights"], levels, penalties) for r in runs]
        for label in validation_rows:
            trace = read(f"tuning/traces/{label}/{name}.json")
            binding = {"protocol_sha256": protocol_hash, "phase": "validation", "candidate": label}
            checked = audit_trace(archive, window, trace, levels, penalties, oracles, binding, label=label, validation=True)
            require(all(datetime.fromisoformat(r["started_utc"]) <= selection_time for r in checked["runs"]),
                    "Validation solve started after setting selection")
            validation_rows[label].extend(checked["runs"])
            if label != "exact_dp":
                for result in checked["runs"]:
                    require(result["row"]["solver"]["options"] == {"time_limit": 15., "mip_rel_gap": 1e-8,
                            "presolve": label == "presolve_on"}, "Validation solver options changed")
    require({c["label"] for c in candidates["candidates"]} == {"presolve_on", "presolve_off"}, "Candidate coverage mismatch")
    rankings = []
    for candidate in candidates["candidates"]:
        label, presolve = candidate["label"], candidate["presolve"]
        require(presolve == (label == "presolve_on"), "Candidate label/presolve mismatch")
        rows = validation_rows[label]
        key = candidate_key(rows, presolve)
        require(candidate["runs"] == len(rows) and candidate["no_incumbent_runs"] == key[0]
                and candidate["not_solver_optimal_runs"] == key[1], "Validation candidate count mismatch")
        # Recorded rank gaps use the directly reconstructed exact-DP path cost;
        # all such costs have separately passed the dense optimum audit.
        certificate_gaps = []
        require(len(candidate["certificates"]) == len(rows), "Missing validation certificates")
        for row, exact, certificate in zip(rows, validation_rows["exact_dp"], candidate["certificates"]):
            optimum = exact["energy"]
            gap = None if row["energy"] is None else row["energy"] - optimum
            relative = None if gap is None else max(0., gap) / max(1., abs(optimum))
            close(certificate["exact_dp_objective"], optimum, "Validation exact certificate")
            close(certificate["milp_minus_dp_objective"], gap, "Validation primal certificate")
            close(certificate["certified_relative_gap"], relative, "Validation relative certificate", atol=1e-12, rtol=1e-8)
            require(certificate["input_sha256"] == row["row"]["input_sha256"] == exact["row"]["input_sha256"],
                    "Validation certificate inputs mismatch")
            certificate_gaps.append(relative)
        gap_sum = None if key[0] else math.fsum(certificate_gaps)
        close(candidate["sum_certified_relative_gap"], gap_sum, "Validation gap total", atol=1e-12, rtol=1e-8)
        close(candidate["formulation_plus_solve_time_s"], key[3], "Validation timing total", atol=1e-8)
        # Match the frozen ranking's native floating-point sum exactly after
        # independently checking each term, so near-zero roundoff ties do not
        # substitute a new candidate-selection policy after the experiment.
        rank = (key[0], key[1], math.inf if gap_sum is None else candidate["sum_certified_relative_gap"],
                candidate["formulation_plus_solve_time_s"], 0 if presolve else 1)
        rankings.append((rank, presolve))
    require(selection["presolve"] == min(rankings)[1] == summary["selected_presolve"]
            and candidates["selected_presolves"] == [selection["presolve"]], "Frozen setting does not follow validation ranking")
    binding = {"protocol_sha256": protocol_hash, "selection_sha256": observed_hashes["selection.json"], "phase": "test"}
    checked_by_arm, metrics_by_arm, rows_by_arm, traces_by_arm = ({a: [] for a in ("milp", "exact_dp")} for _ in range(4))
    basic_by_arm = {a: {} for a in ("milp", "exact_dp")}
    common_read_seconds = 0.
    for number, window in enumerate(windows):
        name = window["window"]["id"]
        observed = read_block_window(protocol["source"], window["window"]["start_unix"], window["window"]["end_unix"],
                                     channels=("main", *CHANNELS), exclude_markers=("s", "+"))
        require(observed["source_window_sha256"] == window["source_window_sha256"], "Test raw-source hash mismatch")
        source_hashes[name] = observed["source_window_sha256"]
        runs = check_compression(window, observed["timestamps"], observed["values"][:, 0], protocol["event_threshold_w"])
        oracles = [dense_oracle(r["aggregate"], r["weights"], levels, penalties) for r in runs]
        source_receipt = read(f"source_reads/{name}.json")
        require(source_receipt["binding"] == binding and source_receipt["blocks"] == window["blocks"]
                and source_receipt["source_window_sha256"] == window["source_window_sha256"], "Source-read receipt mismatch")
        common_read_seconds += source_receipt["source_read_wall_time_s"]
        for arm in ("milp", "exact_dp"):
            trace = read(f"traces/{arm}/{name}.json")
            row = read(f"records/{arm}/{name}.json")
            require(trace["window_index"] == number and trace["arm_order"] == (["milp", "exact_dp"] if number % 2 == 0 else ["exact_dp", "milp"]),
                    "Prespecified alternating arm order changed")
            checked = audit_trace(archive, window, trace, levels, penalties, oracles, binding, label=arm)
            require(all(datetime.fromisoformat(r["started_utc"]) >= selection_time for r in checked["runs"]),
                    "Test solve started before frozen setting selection")
            checked_by_arm[arm].extend(checked["runs"])
            traces_by_arm[arm].append(trace)
            rows_by_arm[arm].append(row)
            prediction_path = archive / "predictions" / arm / f"{name}.npz"
            require(row["trace_sha256"] == observed_hashes[f"traces/{arm}/{name}.json"]
                    and row["prediction_sha256"] == sha(prediction_path)
                    and row["source_read_sha256"] == observed_hashes[f"source_reads/{name}.json"]
                    and row["source_window_sha256"] == window["source_window_sha256"]
                    and row["protocol_sha256"] == protocol_hash and row["selection_sha256"] == binding["selection_sha256"],
                    "Scored row provenance mismatch")
            require(row["window_id"] == name and row["model"] == arm and row["complete"] == trace["complete"], "Scored row identity/coverage mismatch")
            with np.load(prediction_path, allow_pickle=False) as data:
                require(set(data.files) == {"prediction_w", "compressed_mains_w", "timestamps", "covered_mask"}, "Unexpected prediction array fields")
                require(np.array_equal(data["prediction_w"], checked["prediction"], equal_nan=True)
                        and np.array_equal(data["covered_mask"], checked["mask"])
                        and np.array_equal(data["timestamps"], observed["timestamps"]), "Archived prediction does not expand saved trace")
                close(data["compressed_mains_w"], checked["means"], "Archived compressed mains expansion")
            for result in checked["runs"]:
                if arm == "milp":
                    require(result["row"]["solver"]["options"] == {"time_limit": 30., "mip_rel_gap": 1e-8,
                            "presolve": selection["presolve"]}, "Test solver options changed after selection")
            if not row["complete"]:
                require(row["appliances"] is None and row["full_window_metrics"] is None
                        and row["covered_blocks"] == int(checked["mask"].sum()), "Partial coverage is presented as full accuracy")
                continue
            powers = checked["prediction"]
            expected_metrics = raw_appliance_metrics(observed["values"][:, 1:], powers[:, :3], observed["timestamps"],
                                                      frozen["config"]["state_proxy_thresholds_w"])
            check_mapping(row["appliances"], expected_metrics, f"{name}/{arm}/appliance metrics")
            basic = basic_metrics(observed["values"][:, 1:], powers[:, :3])
            basic_by_arm[arm][name] = basic
            aggregate = observed["values"][:, 0]
            raw_sse = math.fsum((float(y) - math.fsum(p)) ** 2 for y, p in zip(aggregate, powers))
            residual = math.fsum((float(a) - float(b)) ** 2 for a, b in zip(aggregate, checked["means"]))
            aggregate_mae = math.fsum(abs(float(y) - math.fsum(p)) for y, p in zip(aggregate, powers)) / len(aggregate)
            transitions = math.fsum(r["native_transition"] for r in trace["runs"])
            native = math.fsum(r["energy"] for r in checked["runs"])
            for field, expected in (("raw_aggregate_mae_w", aggregate_mae), ("raw_block_reconstruction", raw_sse),
                                    ("compression_residual_constant", residual), ("common_full_block_objective", raw_sse + transitions),
                                    ("native_segment_objective", native), ("common_segment_objective", native)):
                close(row[field], expected, f"{name}/{arm}/{field}", atol=1e-4)
            close(raw_sse + transitions, native + residual, "Compressed/raw objective constant identity", atol=1e-4)
            require(row["blocks"] == window["blocks"] and row["runs"] == len(runs)
                    and row["segments"] == sum(len(r["weights"]) for r in runs), "Scored count mismatch")
            metrics_by_arm[arm].append({**row, "appliances": expected_metrics, "raw_aggregate_mae_w": aggregate_mae})
        if (number + 1) % 10 == 0:
            print(f"Independent Stage E raw metrics and dense certificates: {number + 1}/30 windows", flush=True)
    for arm in ("milp", "exact_dp"):
        result = summary["arms"][arm]
        traces, runs, rows = traces_by_arm[arm], checked_by_arm[arm], metrics_by_arm[arm]
        complete = len(rows) == len(windows)
        require(result["full_coverage"] == complete and result["scheduled_windows"] == 30
                and result["complete_windows"] == len(rows) and result["scheduled_blocks"] == 86396
                and result["covered_blocks"] == sum(t["covered_blocks"] for t in traces)
                and result["no_incumbent_runs"] == sum(not r["incumbent"] for r in runs)
                and result["solver_optimal_runs"] == sum(r["solver_optimal"] for r in runs), "Summary arm coverage/status counts mismatch")
        missing_bounds = 0 if arm == "exact_dp" else sum(r["row"]["solver"]["dual_bound"] is None for r in runs)
        inconsistent_bounds = 0 if arm == "exact_dp" else sum(r["row"]["solver"]["bound_validation"] == "dual_bound_exceeds_incumbent" for r in runs)
        statuses = ["exact_dp" if arm == "exact_dp" else r["row"]["solver"]["status"] for r in runs]
        require(result["missing_dual_bound_runs"] == missing_bounds
                and result["inconsistent_dual_bound_runs"] == inconsistent_bounds
                and result["solver_status_counts"] == {s: statuses.count(s) for s in set(statuses)},
                "Summary solver bound/status diagnostics mismatch")
        pooled = result["full_coverage_metrics"] if complete else result["complete_window_subset_metrics"]
        require((result["full_coverage_metrics"] is not None) == complete, "Partial cohort given full-population metrics")
        if rows:
            require(pooled is not None, "Missing evaluated-window metrics")
            expected_pool = pooled_appliances(rows)
            check_mapping(pooled["appliances"], expected_pool, f"{arm}/pooled metrics")
            close(pooled["macro_appliance_mae_w"], math.fsum(v["mae_w"] for v in expected_pool.values()) / 3, "Pooled macro MAE")
            total_blocks = sum(r["blocks"] for r in rows)
            close(pooled["aggregate_mae_w"], math.fsum(r["raw_aggregate_mae_w"] * r["blocks"] for r in rows) / total_blocks, "Pooled aggregate MAE")
            require(pooled["windows"] == len(rows) and pooled["blocks"] == total_blocks, "Pooled coverage mismatch")
            for field in ("native_segment_objective", "common_full_block_objective", "compression_residual_constant",
                          "solver_wall_time_s", "inference_wall_time_s"):
                close(pooled[field], math.fsum(r[field] for r in rows), f"Pooled {field}", atol=1e-4)
        else:
            require(pooled is None, "Empty evaluated cohort received invented metrics")
        for field, expected in (("inference_wall_time_s", math.fsum(t["inference_wall_time_s"] for t in traces)),
                                ("compression_wall_time_s", math.fsum(t["compression_wall_time_s"] for t in traces)),
                                ("expansion_wall_time_s", math.fsum(t["expansion_wall_time_s"] for t in traces)),
                                ("solver_call_wall_time_s", math.fsum(r["row"]["solver_call_wall_time_s"] for r in runs)),
                                ("backend_solve_wall_time_s", math.fsum(r["row"]["solver_wall_time_s"] for r in runs))):
            close(result[field], expected, f"{arm}/{field}", atol=1e-8)
        close(result["formulation_wall_time_s"], None if arm == "exact_dp" else math.fsum(r["row"]["build_time_s"] for r in runs),
              "Summary formulation timing", atol=1e-8)
    certificates = summary["run_certificates"]
    require(len(certificates) == 34, "Missing test objective certificates")
    for candidate, exact, certificate in zip(checked_by_arm["milp"], checked_by_arm["exact_dp"], certificates):
        gap = None if candidate["energy"] is None else candidate["energy"] - exact["energy"]
        relative = None if gap is None else max(0., gap) / max(1., abs(exact["energy"]))
        close(certificate["exact_dp_objective"], exact["energy"], "Summary exact objective certificate")
        close(certificate["milp_minus_dp_objective"], gap, "Summary primal objective gap")
        close(certificate["relative_gap_to_dp"], relative, "Summary relative objective gap", atol=1e-12, rtol=1e-8)
        close(certificate["milp_dual_bound"], candidate["row"]["solver"]["dual_bound"], "Summary restored dual bound")
        require(certificate["milp_solver_certified_optimal"] == candidate["solver_optimal"]
                and certificate["milp_status"] == candidate["row"]["solver"]["status"], "Summary certificate status mismatch")
    full = all(summary["arms"][arm]["full_coverage"] for arm in ("milp", "exact_dp"))
    require(summary["full_coverage"] == full and summary["status"] == ("complete" if full else "all_attempted_incomplete_prediction_coverage"),
            "Summary completion status overstates coverage")
    if full:
        comparison = paired_interval(basic_by_arm["milp"], basic_by_arm["exact_dp"])
        for field, value in comparison.items():
            close(summary["paired_milp_minus_dp_macro_mae"][field], value, "Paired window bootstrap")
        require(summary["paired_milp_minus_dp_macro_mae"]["seed"] == 9301
                and summary["paired_milp_minus_dp_macro_mae"]["replicates"] == 2000, "Bootstrap design changed")
    else:
        require(summary["paired_milp_minus_dp_macro_mae"] is None, "Incomplete cohort has a full-population comparison")
    close(summary["common_source_read_wall_time_s"], common_read_seconds, "Common source-read timing")
    frozen_check()
    return {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": protocol_hash, "summary_sha256": observed_hashes["summary.json"],
            "selection_sha256": observed_hashes["selection.json"], "audit_script_sha256": sha(__file__),
            "full_prediction_coverage": full, "source_window_sha256": source_hashes,
            "coverage": {"test_windows": 30, "test_blocks": 86396, "test_runs_per_arm": 34,
                         "validation_windows": len(validation), "validation_runs_per_candidate": len(validation_rows["exact_dp"]),
                         "test_intervals": 4881},
            "checks": ["frozen code, model, inputs, source windows and result hash bindings",
                       "every validation and test run independently certified by dense S-by-S Bellman recursion",
                       "decoded categories, direct energies, restored solver bounds, infeasible/absent outcomes",
                       "validation candidate coverage, certificates and prespecified selection ranking",
                       "raw mains event compression, gaps and duration expansion reproduced independently",
                       "raw appliance and aggregate metrics, pooled errors and paired-window bootstrap",
                       "all coverage, status counts and recorded timing sums reconciled; no path-tie equality requirement"],
            "limitations": ["Source extraction reuses the established raw-window reader, not metric or solver helpers",
                            "Full solver candidate vectors are not archived: discrete feasibility and reported scalar validation diagnostics are checked",
                            "Passing audit is not evidence of quantum advantage or completion of published-method Stage E comparisons"],
            "artifact_sha256": observed_hashes, "audit_wall_time_s": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/stage_e/mip/run_001")
    args = parser.parse_args()
    result = audit(args.run)
    path = args.run / "independent_audit.json"
    with path.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "full_prediction_coverage": result["full_prediction_coverage"],
                      "coverage": result["coverage"], "receipt": str(path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
