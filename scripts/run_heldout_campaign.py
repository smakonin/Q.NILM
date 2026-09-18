#!/usr/bin/env python3
"""First timestamp-selected held-out R1Hz campaign; no remote execution.

Freeze the protocol before reading measurements, train only on the first
partition, select penalties on validation, then read the held-out test once.
All results are retained, including poor and inactive-appliance windows.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from quantum_nilm.heldout_data import source_bounds, make_chronological_manifest, read_block_window
from quantum_nilm.multistate import fit_power_levels
from quantum_nilm.evaluation import predict_temporal, score_power, pool_window_metrics
from quantum_nilm.fhmm import fit_fhmm, predict_fhmm

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ["dryr", "frdg", "vacu"]
SOURCE_CHANNELS = ["main", *CHANNELS]
RHOS = (0.0, 0.02, 0.1)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_partition(source, manifest, split):
    result = []
    windows = [w for w in manifest["windows"] if w["split"] == split]
    for index, window in enumerate(windows):
        loaded = read_block_window(source, window["start_unix"], window["end_unix"],
                                   channels=SOURCE_CHANNELS, exclude_markers=("s", "+"))
        result.append({**loaded, "window": window})
        if (index + 1) % 10 == 0 or index + 1 == len(windows):
            print(f"{split}: extracted {index + 1}/{len(windows)} fixed days", flush=True)
    return result


def quality_record(window):
    return {"window": window["window"], "quality": window["quality"],
            "source_window_sha256": window["source_window_sha256"],
            "valid_blocks": len(window["timestamps"])}


def fit_models(training):
    usable = [w["values"] for w in training if len(w["values"])]
    if not usable:
        raise ValueError("No valid training blocks")
    values = np.concatenate(usable)
    background = values[:, 0] - values[:, 1:].sum(axis=1)
    background_levels = fit_power_levels(background, 3)
    models = {}
    for name, counts in (("binary", [2, 2, 2]), ("multistate", [4, 3, 2])):
        levels = [fit_power_levels(values[:, i + 1], count) for i, count in enumerate(counts)]
        models[name] = {
            "levels": [*levels, background_levels],
            "ranges": np.array([max(float(np.ptp(x)), 1.0) for x in [*levels, background_levels]]),
            "requested_appliance_states": counts,
        }
    thresholds = np.array([
        (x[0] + x[-1]) / 2 if len(x) > 1 else x[0] + 1
        for x in models["binary"]["levels"][:3]
    ])
    common_event_threshold = 0.01 * max(models["binary"]["ranges"][:3])
    for model in models.values():
        model["event_threshold_w"] = float(common_event_threshold)
    return models, thresholds


def predict_window(window, model, rho, compressed, scope):
    values, times = window["values"], window["timestamps"]
    mains = scope == "mains"
    levels = model["levels"] if mains else model["levels"][:3]
    ranges = model["ranges"] if mains else model["ranges"][:3]
    aggregate = values[:, 0] if mains else values[:, 1:].sum(axis=1)
    # All modeled powers are raw absolute centroids. This is algebraically
    # equivalent to signed centering and incremental powers, with no clipping.
    threshold = model["event_threshold_w"] if compressed else None
    return predict_temporal(aggregate, times, levels, rho * ranges**2, threshold)


def score_window(window, prediction, info, thresholds, model_id, scope):
    values, times = window["values"], window["timestamps"]
    aggregate = values[:, 0] if scope == "mains" else values[:, 1:].sum(axis=1)
    return {
        "window_id": window["window"]["id"], "model": model_id, "scope": scope,
        "appliances": score_power(values[:, 1:], prediction[:, :3], times, thresholds, CHANNELS),
        "raw_aggregate_mae_w": float(np.abs(aggregate - prediction.sum(axis=1)).mean()),
        **info,
    }


def select_penalties(validation, models):
    selections, records = {}, []
    for family, model in models.items():
        for compressed in (False, True):
            model_id = family + ("_compressed" if compressed else "_regular")
            candidates = []
            for rho in RHOS:
                abs_error, blocks, elapsed = 0.0, 0, 0.0
                for window in validation:
                    if not len(window["timestamps"]):
                        continue
                    prediction, info = predict_window(window, model, rho, compressed, "mains")
                    abs_error += float(np.abs(prediction[:, :3] - window["values"][:, 1:]).sum())
                    blocks += len(prediction)
                    elapsed += info["solver_wall_time_s"]
                if not blocks:
                    raise ValueError("No valid validation blocks")
                record = {"model": model_id, "rho": rho, "macro_appliance_mae_w": abs_error / (3 * blocks),
                          "valid_blocks": blocks, "solver_wall_time_s": elapsed}
                candidates.append(record)
                records.append(record)
                print(f"validation: {model_id} rho={rho:g} MAE={record['macro_appliance_mae_w']:.3f} W", flush=True)
            # Equal validation loss prefers the smaller penalty, independent of test.
            chosen = min(candidates, key=lambda r: (r["macro_appliance_mae_w"], r["rho"]))
            selections[model_id] = {"family": family, "compressed": compressed, "rho": chosen["rho"]}
    return selections, records


def aggregate_results(records):
    grouped = {}
    for model_id, scope in sorted({(r["model"], r["scope"]) for r in records}):
        rows = [r for r in records if r["model"] == model_id and r["scope"] == scope]
        appliances = pool_window_metrics(rows, CHANNELS)
        valid = sum(r["blocks"] for r in rows)
        grouped[scope + "/" + model_id] = {
            "scope": scope, "model": model_id, "n_windows": len(rows), "valid_blocks": valid,
            "appliances": appliances,
            "macro_appliance_mae_w": float(np.mean([r["mae_w"] for r in appliances.values()])),
            "raw_aggregate_mae_w": sum(r["raw_aggregate_mae_w"] * r["blocks"] for r in rows) / valid,
            "solver_wall_time_s": sum(r["solver_wall_time_s"] for r in rows),
            "segments": sum(r["segments"] for r in rows),
        }
    return grouped


def paired_bootstrap(records, left, right, scope="mains"):
    left_rows = {r["window_id"]: r for r in records if r["model"] == left and r["scope"] == scope}
    right_rows = {r["window_id"]: r for r in records if r["model"] == right and r["scope"] == scope}
    if left_rows.keys() != right_rows.keys() or not left_rows:
        raise ValueError("Paired inference requires identical nonempty windows")
    names = sorted(left_rows)
    error_left, error_right, counts = [], [], []
    for name in names:
        a, b = left_rows[name], right_rows[name]
        if a["blocks"] != b["blocks"]:
            raise ValueError("Paired windows have different block counts")
        error_left.append(sum(x["absolute_error_sum_w"] for x in a["appliances"].values()))
        error_right.append(sum(x["absolute_error_sum_w"] for x in b["appliances"].values()))
        counts.append(a["blocks"] * 3)
    difference = np.asarray(error_left) - np.asarray(error_right)
    counts = np.asarray(counts)
    rng = np.random.default_rng(7301)
    draws = rng.integers(0, len(names), (2000, len(names)))
    boot = difference[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "left": left, "right": right, "scope": scope, "n_paired_days": len(names),
        "metric": "left minus right macro appliance MAE W; negative favors left",
        "difference_w": float(difference.sum() / counts.sum()),
        "paired_day_bootstrap_95_interval_w": np.quantile(boot, [0.025, 0.975]).tolist(),
        "bootstrap_replicates": 2000, "seed": 7301,
        "interpretation": "descriptive paired day bootstrap within one home; not cross-home uncertainty or a quantum comparison",
    }


def markdown(summary):
    lines = ["# Q.NILM first held-out R1Hz campaign", "",
             "Classical exact optimization and model evaluation, not quantum-hardware accuracy or quantum advantage.", "",
             "The protocol, timestamp-only day list, state counts, metrics and candidate penalties were frozen before measurement reads. "
             "Power levels used training data only. Penalties were selected by validation appliance MAE before any test day was read. "
             "This is a sampled within-home campaign, not a full-dataset or cross-home evaluation.", "",
             "## Held-out outcomes", "",
             "| Input | Model | Days | Dryer MAE W | Fridge MAE W | Vacuum MAE W | Macro MAE W | Aggregate MAE W |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["pooled_results"].values():
        maes = [row["appliances"][ch]["mae_w"] for ch in CHANNELS]
        lines.append(f"| {row['scope']} | {row['model']} | {row['n_windows']} | " +
                     " | ".join(f"{value:.3f}" for value in [*maes, row["macro_appliance_mae_w"], row["raw_aggregate_mae_w"]]) + " |")
    lines += ["", "Mains input includes a three-state background component learned from training-only mains minus the selected circuits. "
              "Selected-circuit input is a diagnostic control unavailable to a deployed NILM system. "
              "Both use the same appliance centroids and validation-selected penalties. "
              "All errors above are against original valid 30-second block measurements, not compressed interval means.", "",
              "## Paired model comparisons", "",
              "| Comparison (mains) | MAE difference W | Descriptive 95% day-bootstrap interval W |",
              "|---|---:|---:|"]
    for row in summary["paired_comparisons"]:
        low, high = row["paired_day_bootstrap_95_interval_w"]
        lines.append(f"| {row['left']} minus {row['right']} | {row['difference_w']:.3f} | [{low:.3f}, {high:.3f}] |")
    lines += ["", "Negative differences favor the left method. These are prespecified descriptive comparisons, "
              "not multiple-testing-adjusted superiority claims.", "", "## Activity and state labels", "",
              "High-state proxy thresholds come from training-only binary centroids and are shared by all models. "
              "These high/low circuit proxies are not manual physical-state annotations. Continuous appliance-power MAE is primary. "
              "Event F1 uses a fixed +/-30-second tolerance with sign-matched one-to-one events and no events across gaps. "
              "An undefined F1 or MCC is null, never silently perfect.", "",
              "| Appliance | Above-threshold test blocks | Windows with above-threshold blocks | High-state proxy threshold W |",
              "|---|---:|---:|---:|"]
    reference = summary["pooled_results"]["mains/binary_regular"]["appliances"]
    for ch, threshold in zip(CHANNELS, summary["state_proxy_thresholds_w"]):
        row = reference[ch]
        lines.append(f"| {ch} | {row['active_blocks']} | {row['active_windows']} | {threshold:.3f} |")
    lines += ["", "## Limitations", "", *["- " + x for x in summary["limitations"]], "",
              "See `protocol.json`, `frozen_models.json`, `data_quality.json`, `validation.json`, and `test_windows.json` "
              "for exact windows, source hashes, exclusions, train-only model values, every candidate validation score, and every test result.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/Users/stephen/Documents/Research/Datasets/R1Hz/recovered/power.csv"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/heldout_campaign")
    parser.add_argument("--counts", type=int, nargs=3, default=(90, 30, 30))
    parser.add_argument("--window-seconds", type=int, default=86400)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory already exists; refusing to overwrite or silently retest")
    started = perf_counter()
    first, end = source_bounds(args.source)
    manifest = make_chronological_manifest(first, end, counts=tuple(args.counts), window_seconds=args.window_seconds)
    args.output_dir.mkdir(parents=True)
    protocol = {
        "campaign": "first timestamp-selected chronological within-home model campaign",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source), "source_size_bytes": args.source.stat().st_size,
        "source_mtime_ns": args.source.stat().st_mtime_ns,
        "dataset_doi": "10.7910/DVN/RCB5VJ", "manifest": manifest,
        "source_channels": SOURCE_CHANNELS, "exclude_markers": ["s", "+"],
        "block_seconds": 30, "incomplete_or_nonfinite_blocks": "excluded; gaps reset all temporal inference",
        "model_states": {"binary": [2, 2, 2], "multistate": [4, 3, 2], "background": 3},
        "centering": "raw absolute centroids, algebraically equivalent to signed centering; no clipping",
        "penalty_rho_candidates": list(RHOS), "event_threshold_ratio": 0.01,
        "event_threshold_basis": "common training binary maximum appliance power range for both families",
        "validation_selection": "lowest pooled macro appliance power MAE on actual mains; ties prefer lower rho",
        "test_scopes": ["mains", "selected_circuits_control"],
        "baselines": ["train_low_power_constant", "exact_MAP_factorial_HMM"],
        "primary_metric": "appliance-power MAE on original valid 30-second blocks, equal appliance weights",
        "descriptive_comparisons": [["multistate_regular", "binary_regular"], ["multistate_compressed", "binary_compressed"]],
        "bootstrap": {"unit": "fixed day window", "replicates": 2000, "seed": 7301},
        "event_tolerance_seconds": 30, "missing_test_days": "report, do not replace after observing activity/results",
        "quantum_jobs_authorized_by_this_script": False,
    }
    # This immutable design record is written BEFORE reading train/validation/test measurements.
    write_json(args.output_dir / "protocol.json", protocol)
    protocol_hash = sha(args.output_dir / "protocol.json")
    print("Protocol frozen; reading TRAIN only", flush=True)
    training = load_partition(args.source, manifest, "train")
    models, thresholds = fit_models(training)
    fhmm_models = {family: fit_fhmm([w for w in training if len(w["timestamps"])], model["levels"])
                   for family, model in models.items()}
    print("Train-only levels fitted; reading VALIDATION only", flush=True)
    validation = load_partition(args.source, manifest, "validation")
    selections, validation_records = select_penalties(validation, models)
    frozen = {
        "frozen_utc_before_test_read": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_hash, "selected_parameters": selections,
        "state_proxy_thresholds_w": thresholds.tolist(),
        "models": {family: {"levels_w": [x.tolist() for x in model["levels"]],
                            "ranges_w": model["ranges"].tolist(),
                            "event_threshold_w": model["event_threshold_w"],
                            "requested_appliance_states": model["requested_appliance_states"]}
                   for family, model in models.items()},
        "fhmm": {family: {"transition_costs": [x.tolist() for x in model.transition_costs],
                          "initial_costs": [x.tolist() for x in model.initial_costs],
                          "noise_variance_w2": float(model.noise_variance)} for family, model in fhmm_models.items()},
    }
    write_json(args.output_dir / "validation.json", validation_records)
    write_json(args.output_dir / "frozen_models.json", frozen)
    frozen_hash = sha(args.output_dir / "frozen_models.json")
    print("Validation complete; parameters frozen. Reading TEST for first evaluation", flush=True)
    testing = load_partition(args.source, manifest, "test")
    records, missing = [], []
    for index, window in enumerate(testing):
        if not len(window["timestamps"]):
            missing.append(window["window"])
            continue
        for scope in ("mains", "selected_circuits_control"):
            for model_id, selection in selections.items():
                prediction, info = predict_window(window, models[selection["family"]], selection["rho"], selection["compressed"], scope)
                records.append(score_window(window, prediction, info, thresholds, model_id, scope))
            aggregate = window["values"][:, 0] if scope == "mains" else window["values"][:, 1:].sum(axis=1)
            for family, model in fhmm_models.items():
                prediction, info = predict_fhmm(aggregate, window["timestamps"], model, include_background=scope == "mains")
                records.append(score_window(window, prediction, info, thresholds, family + "_fhmm", scope))
            levels = models["binary"]["levels"][:4 if scope == "mains" else 3]
            baseline = np.tile([x[0] for x in levels], (len(window["timestamps"]), 1))
            records.append(score_window(window, baseline,
                {"blocks": len(baseline), "segments": len(baseline), "solver_wall_time_s": 0.0,
                 "timing_note": "constant baseline timing not measured, not a timing comparator"},
                thresholds, "train_low_power_constant", scope))
        print(f"test: scored fixed day {index + 1}/{len(testing)}", flush=True)
    if not records:
        raise RuntimeError("No valid held-out observations; frozen protocol retained")
    write_json(args.output_dir / "test_windows.json", records)
    write_json(args.output_dir / "data_quality.json", {
        split: [quality_record(w) for w in windows] for split, windows in
        (("train", training), ("validation", validation), ("test", testing))})
    if args.source.stat().st_size != protocol["source_size_bytes"] or args.source.stat().st_mtime_ns != protocol["source_mtime_ns"]:
        raise RuntimeError("Source changed during campaign")
    if sha(args.output_dir / "protocol.json") != protocol_hash or sha(args.output_dir / "frozen_models.json") != frozen_hash:
        raise RuntimeError("Frozen experiment inputs changed during test")
    summary = {
        "algorithm": "Q.NILM", "status": "held-out classical model evaluation, not quantum advantage",
        "protocol_sha256": protocol_hash, "frozen_models_sha256": frozen_hash,
        "state_proxy_thresholds_w": thresholds.tolist(), "selected_parameters": selections,
        "pooled_results": aggregate_results(records),
        "paired_comparisons": [paired_bootstrap(records, a, b) for a, b in protocol["descriptive_comparisons"]],
        "empty_test_windows": missing,
        "valid_blocks_by_partition": {split: sum(len(w["timestamps"]) for w in windows) for split, windows in
            (("train", training), ("validation", validation), ("test", testing))},
        "total_campaign_wall_time_s": perf_counter() - started,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "limitations": [
            "One household and a timestamp-selected subset of days, not every day in the 757-day release or a cross-home result.",
            "Training is supervised using non-mixed circuit channels. Test inference sees only the stated aggregate, never test circuit labels.",
            "The background component is a coarse three-state approximation to unmodeled loads and meter residuals, not a fully resolved appliance inventory.",
            "State F1/MCC refer to shared train-derived high/low proxies; multistate improvements are judged primarily by continuous appliance power error.",
            "Selected-circuit totals are diagnostic controls and are not a deployable input available from mains alone.",
            "Calendar selection avoids result-based cherry-picking but does not guarantee 30 active days for rare appliances; coverage is reported.",
            "Bootstrap intervals describe this one-home day sample and are not corrected simultaneous intervals or proof of general superiority.",
            "Exact classical DP and exact MAP FHMM are included; tuned MIP and resource-matched quantum scaling across independent homes remain outstanding.",
            "No quantum circuit was run by this campaign; better multistate accuracy would be a classical modeling result, not quantum advantage.",
        ],
    }
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "summary.md").write_text(markdown(summary))
    tracked = sorted(p for p in args.output_dir.iterdir() if p.is_file())
    (args.output_dir / "checksums.sha256").write_text("".join(f"{sha(p)}  {p.name}\n" for p in tracked))
    print(markdown(summary), flush=True)


if __name__ == "__main__":
    main()
