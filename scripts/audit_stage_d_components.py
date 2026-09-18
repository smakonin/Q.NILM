#!/usr/bin/env python3
"""Independent objectives, dense-DP certificates and metrics for Stage D.

The established source-window reader is reused solely to extract/hash R1Hz
blocks. No production inference, scoring, pooling, or bootstrap helper is
called. Scored artifacts are read-only; the receipt is create-new.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from quantum_nilm.heldout_data import read_block_window

CHANNELS = ("dryr", "frdg", "vacu")
ARMS = ("duration_normalized", "unit_normalized", "duration_global_max", "duration_global_mean")


def require(value, message):
    if not value:
        raise ValueError(message)


def same(actual, expected, label, atol=1e-7, rtol=1e-11):
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f"{label}: field coverage")
        for key in expected:
            same(actual[key], expected[key], f"{label}/{key}", atol, rtol)
    elif expected is None or isinstance(expected, (str, bool, int)):
        require(actual == expected, f"{label}: {actual} != {expected}")
    else:
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        require(a.shape == b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b)), f"{label}: nonfinite or shape")
        require(np.all(np.abs(a-b) <= atol+rtol*np.abs(b)), f"{label}: mismatch {np.max(np.abs(a-b))}")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def class_metrics(tp, tn, fp, fn):
    tp, tn, fp, fn = map(int, (tp, tn, fp, fn))
    total = tp+tn+fp+fn
    denominator = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                bit_accuracy=(tp+tn)/total if total else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                mcc=(tp*tn-fp*fn)/math.sqrt(denominator) if denominator else None)


def independent_events(actual, inferred, timestamps):
    cutoffs = [0]+(np.where(np.diff(timestamps) != 30)[0]+1).tolist()+[len(timestamps)]
    found = spurious = missed = 0
    for lo, hi in zip(cutoffs[:-1], cutoffs[1:]):
        da = np.diff(actual[lo:hi].astype(int))
        dp = np.diff(inferred[lo:hi].astype(int))
        for sign in (-1, 1):
            reference = np.where(da == sign)[0].tolist()
            predicted = np.where(dp == sign)[0].tolist()
            i = j = 0
            while i < len(reference) and j < len(predicted):
                if predicted[j] < reference[i]-1:
                    spurious += 1
                    j += 1
                elif reference[i] < predicted[j]-1:
                    missed += 1
                    i += 1
                else:
                    found += 1
                    i += 1
                    j += 1
            missed += len(reference)-i
            spurious += len(predicted)-j
    return dict(tp=found, fp=spurious, fn=missed)


def independently_score(reference, predicted, timestamps, thresholds):
    result = {}
    for i, channel in enumerate(CHANNELS):
        actual, estimated = reference[:, i], predicted[:, i]
        error = estimated-actual
        a, p = actual > thresholds[i], estimated > thresholds[i]
        events = independent_events(a, p, timestamps)
        denominator = 2*events["tp"]+events["fp"]+events["fn"]
        totals = dict(absolute=float(np.linalg.norm(error, ord=1)),
                      squared=float(np.dot(error, error)),
                      signed=float(np.sum(error)), power=float(np.sum(actual)),
                      power_squared=float(np.dot(actual, actual)))
        result[channel] = dict(n_blocks=len(actual), mae_w=totals["absolute"]/len(actual),
            absolute_error_sum_w=totals["absolute"], squared_error_sum_w2=totals["squared"],
            true_power_sum_w=totals["power"], true_squared_power_sum_w2=totals["power_squared"],
            signed_power_error_sum_w=totals["signed"], energy_error_wh=totals["signed"]/120,
            normalized_disaggregation_error=totals["squared"]/totals["power_squared"] if totals["power_squared"] else None,
            signal_aggregate_error=abs(totals["signed"])/abs(totals["power"]) if totals["power"] else None,
            active_blocks=int(np.count_nonzero(a)),
            state=class_metrics(np.count_nonzero(a&p), np.count_nonzero(~a&~p), np.count_nonzero(~a&p), np.count_nonzero(a&~p)),
            event_counts=events, event_f1=2*events["tp"]/denominator if denominator else None)
    return result


def independently_pool(records):
    result = {}
    for channel in CHANNELS:
        rows = [r["appliances"][channel] for r in records]
        n = sum(x["n_blocks"] for x in rows)
        sums = {field:sum(r[field] for r in rows) for field in (
            "absolute_error_sum_w", "signed_power_error_sum_w", "squared_error_sum_w2",
            "true_power_sum_w", "true_squared_power_sum_w2")}
        events = {field:sum(r["event_counts"][field] for r in rows) for field in ("tp", "fp", "fn")}
        states = {field:sum(r["state"][field] for r in rows) for field in ("tp", "tn", "fp", "fn")}
        denom = 2*events["tp"]+events["fp"]+events["fn"]
        result[channel] = dict(n_blocks=n, mae_w=sums["absolute_error_sum_w"]/n,
            rmse_w=math.sqrt(sums["squared_error_sum_w2"]/n),
            energy_error_wh=sums["signed_power_error_sum_w"]/120,
            signal_aggregate_error=abs(sums["signed_power_error_sum_w"])/abs(sums["true_power_sum_w"]) if sums["true_power_sum_w"] else None,
            normalized_disaggregation_error=sums["squared_error_sum_w2"]/sums["true_squared_power_sum_w2"] if sums["true_squared_power_sum_w2"] else None,
            active_blocks=sum(r["active_blocks"] for r in rows), active_windows=sum(r["active_blocks"] > 0 for r in rows),
            state=class_metrics(**states), event_counts=events, event_f1=2*events["tp"]/denom if denom else None)
    return result


def audit(archive):
    started = time.perf_counter()
    hashes = {}

    def read(path):
        hashes[str(path.relative_to(archive))] = sha(path)
        return json.loads(path.read_text())

    protocol, frozen, freeze, summary = [read(archive/name) for name in (
        "protocol.json", "frozen_inputs.json", "freeze.json", "summary.json")]
    require(freeze["protocol_sha256"] == hashes["protocol.json"] == summary["protocol_sha256"], "Protocol hashes differ")
    require(freeze["frozen_inputs_sha256"] == hashes["frozen_inputs.json"], "Frozen inputs changed")
    for path, expected in protocol["source_code_sha256"].items():
        require(sha(ROOT/path) == expected == sha(archive/"source_snapshot"/path), f"Source changed: {path}")
    for path, expected in protocol["source_artifacts_sha256"].items():
        require(sha(path) == expected, f"Parent archive changed: {path}")
    require(protocol["arms"] == list(ARMS) and protocol["retuning"] is False and protocol["hardware_jobs"] is False, "Unexpected scientific design")
    require(summary["status"] == "complete" and summary["baseline_gate"]["status"] == "passed", "Execution/gate incomplete")
    windows = frozen["windows"]
    identifiers = [w["window"]["id"] for w in windows]
    require(len(set(identifiers)) == len(identifiers) == 30 and identifiers == protocol["cohort_windows"], "Cohort coverage")
    require(sum(w["blocks"] for w in windows) == protocol["blocks"] == 86396, "Block coverage")
    model = frozen["model"]["models"]["multistate"]
    levels = [np.asarray(x) for x in model["levels_w"]]
    normalized = .1*np.array([np.max(x)-np.min(x) for x in levels])**2
    penalties = dict(duration_normalized=normalized, unit_normalized=normalized,
                     duration_global_max=np.repeat(normalized.max(), len(levels)),
                     duration_global_mean=np.repeat(normalized.mean(), len(levels)))
    for arm in ARMS:
        same(protocol["penalties_by_arm"][arm], penalties[arm], f"{arm}: penalty definition")
        for folder in ("traces", "records"):
            require({p.stem for p in (archive/folder/arm).glob("*.json")} == set(identifiers), f"{arm}: {folder} coverage")
    joint = np.array(list(itertools.product(*(range(len(x)) for x in levels))))
    joint_power = sum(levels[i][joint[:, i]] for i in range(len(levels)))
    transitions = {arm:np.sum((joint[:, None, :] != joint[None, :, :])*penalties[arm], axis=2) for arm in ARMS}
    reference_baseline = {r["window_id"]:r for r in json.loads((Path(protocol["source_directory"])/"simulation/test_windows.json").read_text()) if r["model"] == "exact_full"}
    by_arm = {arm:[] for arm in ARMS}
    native_max_gap = common_max_identity_error = 0.
    certified_runs = inferred_categories = 0
    source_hashes = {}

    for window in windows:
        identifier = window["window"]["id"]
        observed = read_block_window(protocol["source"], window["window"]["start_unix"],
            window["window"]["end_unix"], channels=("main", *CHANNELS), exclude_markers=("s", "+"))
        require(observed["source_window_sha256"] == window["source_window_sha256"], f"{identifier}: source window hash")
        source_hashes[identifier] = observed["source_window_sha256"]
        values, timestamps = observed["values"], observed["timestamps"]
        require(len(values) == window["blocks"], f"{identifier}: source block count")
        chunks = window["chunks"]
        runs = []
        cursor = 0
        for number, chunk in enumerate(chunks):
            require(chunk["chunk"] == number and chunk["block_start"] == cursor, f"{identifier}: chunk continuity")
            if chunk["reset"]:
                runs.append(dict(run=chunk["run"], start=cursor, means=[], weights=[]))
            require(runs and runs[-1]["run"] == chunk["run"], f"{identifier}: missing reset")
            runs[-1]["means"].extend(chunk["aggregate"])
            runs[-1]["weights"].extend(chunk["weights"])
            cursor = chunk["block_stop"]
            runs[-1]["stop"] = cursor
        require(cursor == len(values), f"{identifier}: final chunk coverage")
        source_edges = [0]+(np.flatnonzero(np.diff(timestamps) != 30)+1).tolist()+[len(timestamps)]
        require([(r["start"], r["stop"]) for r in runs] == list(zip(source_edges[:-1], source_edges[1:])), f"{identifier}: gap resets")
        for run in runs:
            position = run["start"]
            for mean, weight in zip(run["means"], run["weights"]):
                require(isinstance(weight, int) and weight > 0, f"{identifier}: invalid duration")
                same(mean, float(values[position:position+weight, 0].mean()), f"{identifier}: segment mean", atol=1e-8)
                position += weight
            require(position == run["stop"], f"{identifier}: duration coverage")

        for arm in ARMS:
            trace = read(archive/"traces"/arm/(identifier+".json"))
            row = read(archive/"records"/arm/(identifier+".json"))
            require(trace["arm"] == row["model"] == arm and trace["window_id"] == row["window_id"] == identifier,
                    f"{arm}/{identifier}: identity")
            require(trace["protocol_sha256"] == row["protocol_sha256"] == freeze["protocol_sha256"], "Trace/record protocol")
            require(row["trace_sha256"] == hashes[f"traces/{arm}/{identifier}.json"], "Trace hash mismatch")
            require(row["source_window_sha256"] == observed["source_window_sha256"], "Record source hash")
            require(len(trace["runs"]) == len(runs), "Trace run coverage")
            prediction = np.zeros((len(values), len(levels)))
            block_means = np.zeros(len(values))
            native = common = common_transition = 0.
            switches = np.zeros(len(levels), dtype=np.int64)
            for run, segment in zip(runs, trace["runs"]):
                require((segment["run"], segment["block_start"], segment["block_stop"]) == (run["run"], run["start"], run["stop"]), "Trace reset or endpoints")
                states = np.asarray(segment["states"], dtype=int)
                means, weights = np.asarray(run["means"]), np.asarray(run["weights"])
                require(states.shape == (len(means), len(levels)), "State trajectory dimensions")
                require(all(np.all((states[:, i] >= 0)&(states[:, i] < len(x))) for i,x in enumerate(levels)), "State outside levels")
                powers = np.column_stack([x[states[:, i]] for i,x in enumerate(levels)])
                native_weights = np.ones(len(weights)) if arm == "unit_normalized" else weights
                reconstruction = (means-powers.sum(axis=1))**2
                run_switches = np.sum(states[1:] != states[:-1], axis=0)
                native_reconstruction = float(native_weights@reconstruction)
                native_transition = float(penalties[arm]@run_switches)
                common_reconstruction = float(weights@reconstruction)
                transition = float(normalized@run_switches)
                objective = native_reconstruction+native_transition
                cross_objective = common_reconstruction+transition
                # Independent ordinary dense Viterbi recurrence, not the factorized production solver.
                costs = native_weights[0]*(means[0]-joint_power)**2
                for mean, weight in zip(means[1:], native_weights[1:]):
                    costs = np.min(costs[:, None]+transitions[arm], axis=0)+weight*(mean-joint_power)**2
                optimum = float(np.min(costs))
                native_max_gap = max(native_max_gap, abs(objective-optimum))
                same(objective, optimum, f"{arm}/{identifier}: independent dense-DP certificate", atol=1e-3, rtol=1e-12)
                for field, expected in (("native_segment_objective", objective), ("native_reconstruction", native_reconstruction),
                                        ("native_transition", native_transition), ("common_segment_objective", cross_objective),
                                        ("common_segment_reconstruction", common_reconstruction), ("common_transition", transition)):
                    same(segment[field], expected, f"{arm}/{identifier}/{field}", atol=1e-3, rtol=1e-12)
                same(segment["switch_counts"], run_switches, "Per-run switches", atol=0, rtol=0)
                prediction[run["start"]:run["stop"]] = np.repeat(powers, weights, axis=0)
                block_means[run["start"]:run["stop"]] = np.repeat(means, weights)
                native += objective
                common += cross_objective
                common_transition += transition
                switches += run_switches
                certified_runs += 1
                inferred_categories += states.size
            measured = independently_score(values[:, 1:], prediction[:, :3], timestamps,
                                             frozen["config"]["state_proxy_thresholds_w"])
            same(row["appliances"], measured, f"{arm}/{identifier}: all raw-block appliance metrics")
            residual = float(np.dot(values[:, 0]-block_means, values[:, 0]-block_means))
            raw_residual = values[:, 0]-prediction.sum(axis=1)
            raw_reconstruction = float(np.dot(raw_residual, raw_residual))
            common_full = raw_reconstruction+common_transition
            common_max_identity_error = max(common_max_identity_error, abs(common_full-common-residual))
            same(common_full, common+residual, "Raw-block compression identity", atol=1e-3, rtol=1e-12)
            expected = dict(window_id=identifier, model=arm, seed=None, blocks=len(values), runs=len(runs),
                            segments=sum(len(r["means"]) for r in runs), appliances=measured,
                            raw_aggregate_mae_w=float(np.linalg.norm(raw_residual, ord=1)/len(values)),
                            native_segment_objective=native, common_segment_objective=common,
                            compression_residual_constant=residual, raw_block_reconstruction=raw_reconstruction,
                            common_full_block_objective=common_full, switch_counts=switches.tolist())
            for field, actual in expected.items():
                same(row[field], actual, f"{arm}/{identifier}: {field}", atol=1e-3 if "objective" in field or "reconstruction" in field or "constant" in field else 1e-7, rtol=1e-12)
            require(trace["solver_wall_time_s"] == row["solver_wall_time_s"] and trace["inference_wall_time_s"] == row["inference_wall_time_s"] and
                    trace["inference_wall_time_s"] >= trace["solver_wall_time_s"] >= 0, "Timing reconciliation")
            if arm == ARMS[0]:
                old = reference_baseline[identifier]
                same(row["appliances"], old["appliances"], f"{identifier}: original complete baseline metrics")
                same(row["raw_aggregate_mae_w"], old["raw_aggregate_mae_w"], f"{identifier}: original aggregate baseline")
            by_arm[arm].append(row)
        print(f"Independently checked {identifier}: all four arms", flush=True)

    pooled_mae = {}
    for arm, rows in by_arm.items():
        result = summary["arms"][arm]
        appliance = independently_pool(rows)
        same(result["appliances"], appliance, f"{arm}: pooled metrics")
        macro = sum(x["mae_w"] for x in appliance.values())/3
        pooled_mae[arm] = macro
        same(result["macro_appliance_mae_w"], macro, f"{arm}: macro MAE")
        for field in ("blocks", "runs", "segments", "native_segment_objective", "common_full_block_objective",
                      "compression_residual_constant", "solver_wall_time_s", "inference_wall_time_s"):
            same(result[field], sum(r[field] for r in rows), f"{arm}: pooled {field}", atol=1e-3 if "objective" in field or "constant" in field else 1e-7)
        same(result["aggregate_mae_w"], sum(r["raw_aggregate_mae_w"]*r["blocks"] for r in rows)/86396, f"{arm}: pooled aggregate")
        same(result["switch_counts"], np.sum([r["switch_counts"] for r in rows], axis=0), f"{arm}: pooled switches", atol=0, rtol=0)
        same(result["common_objective_gap_from_baseline"], result["common_full_block_objective"]-summary["arms"][ARMS[0]]["common_full_block_objective"], f"{arm}: common objective gap", atol=1e-3)
        require(result["common_objective_gap_from_baseline"] >= -1e-3, "Ablation beats common exact certificate")
    same(pooled_mae[ARMS[0]], frozen["expected_baseline"]["macro_appliance_mae_w"], "Original pooled baseline", atol=1e-8, rtol=0)
    require(len(summary["paired_comparisons"]) == len(protocol["bootstrap"]["contrasts"]) == 4, "Contrast count")
    for result, (left, right) in zip(summary["paired_comparisons"], protocol["bootstrap"]["contrasts"]):
        require(result["left"] == left and result["right"] == right, "Contrast direction/coverage")
        a = {r["window_id"]:r for r in by_arm[left]}
        b = {r["window_id"]:r for r in by_arm[right]}
        differences = np.array([sum(a[w]["appliances"][c]["absolute_error_sum_w"]-b[w]["appliances"][c]["absolute_error_sum_w"] for c in CHANNELS) for w in sorted(a)])
        denominators = np.array([3*a[w]["blocks"] for w in sorted(a)])
        indices = np.random.default_rng(protocol["bootstrap"]["seed"]).integers(0, 30, (protocol["bootstrap"]["replicates"], 30))
        bootstrapped = np.sum(differences[indices], axis=1)/np.sum(denominators[indices], axis=1)
        same(result["difference_w"], differences.sum()/denominators.sum(), f"{left}/{right}: pooled difference")
        same(result["paired_window_bootstrap_95_interval_w"], np.percentile(bootstrapped, [2.5, 97.5]), f"{left}/{right}: paired CI")
    require(set(summary["record_sha256"]) == {f"records/{arm}/{w}.json" for arm in ARMS for w in identifiers}, "Summary receipt coverage")
    for path, expected in summary["record_sha256"].items():
        require(hashes[path] == expected, "Summary record hash")
    return dict(status="passed_independent_full_component_audit", audited_utc=datetime.now(timezone.utc).isoformat(),
                audit_script_sha256=sha(Path(__file__)), protocol_sha256=freeze["protocol_sha256"],
                arms_checked=4, windows_per_arm=30, valid_blocks_per_arm=86396, arm_window_records_checked=120,
                independently_certified_valid_runs=certified_runs, inferred_categories_checked=inferred_categories,
                maximum_native_dense_dp_certificate_discrepancy=native_max_gap,
                maximum_raw_block_compression_identity_discrepancy=common_max_identity_error,
                paired_macro_mae_bootstrap_intervals_checked=4, macro_mae_w=pooled_mae,
                source_window_sha256=source_hashes, artifact_sha256=hashes,
                scope="Reused hashed source-window extraction only; independent dense temporal DP, all raw-block power/state/event metrics, native/common objectives, expansion/reset coverage, pooling and paired confidence intervals",
                limitations="Already-examined single-home data; fixed settings without retuning; classical component comparisons, not quantum advantage",
                runtime_seconds=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT/"results/stage_d/components/run_001")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    archive = args.archive.resolve()
    result = audit(archive)
    if not args.check_only:
        with (archive/"independent_audit.json").open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("artifact_sha256", "source_window_sha256")}, indent=2))


if __name__ == "__main__":
    main()
