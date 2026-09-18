#!/usr/bin/env python3
"""Prospective REFIT evaluation of fixed Q.NILM settings; strictly local."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from quantum_nilm.evaluation import contiguous_slices, pool_window_metrics, predict_temporal, score_power
from quantum_nilm.heldout_data import make_chronological_manifest
from quantum_nilm.multistate import fit_power_levels
from quantum_nilm.refit import source_bounds, read_block_window
from scripts.run_quantum_heldout import make_chunks, infer_chunks, expand_predictions, sha, save_json

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ["washing", "cooling", "dishwasher"]
# Order: washing machine, one stable identified cooling channel, dishwasher.
HOMES = {1: (5, 1, 6), 2: (2, 1, 3), 3: (6, 2, 5), 5: (3, 1, 4),
         6: (2, 1, 3), 7: (5, 1, 6), 9: (3, 1, 4), 10: (5, 4, 6),
         11: (3, 2, 4), 15: (3, 1, 4), 16: (5, 1, 6), 18: (5, 3, 6),
         20: (4, 1, 5), 21: (3, 1, 4)}
METHODS = ("qaoa_ideal", "uniform", "exact_chunk", "exact_full", "training_mean")
SEEDS = (1907, 2907, 3907)
CODE = ("scripts/run_stage_g_refit.py", "src/quantum_nilm/refit.py",
        "scripts/run_quantum_heldout.py", "src/quantum_nilm/categorical_qaoa.py",
        "src/quantum_nilm/multistate.py", "src/quantum_nilm/evaluation.py",
        "src/quantum_nilm/heldout_data.py", "docs/stage_g_refit_protocol.md")


def read(path):
    return json.loads(Path(path).read_text())


def freeze(output, source):
    if output.exists():
        raise FileExistsError("Use a fresh archive; no silent retest or overwrite")
    homes = []
    for house, selected in HOMES.items():
        path = source / f"CLEAN_House{house}.csv"
        raw_first, raw_end = source_bounds(path)
        first, end = math.ceil(raw_first), math.floor(raw_end)
        homes.append({"house": house, "source": str(path.resolve()),
                      "source_sha256": sha(path), "source_size_bytes": path.stat().st_size,
                      "channels": [f"Appliance{i}" for i in selected],
                      "raw_supported_bounds": [raw_first, raw_end],
                      "integer_interior_bounds": [first, end],
                      "manifest": make_chronological_manifest(first, end, counts=(30, 0, 30))})
        print(f"Boundary-only source freeze: house {house}", flush=True)
    angles = ROOT / "results/quantum_heldout/angles.json"
    original_angles = read(angles)
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "study": "external calibrated-home frozen-setting ideal simulation",
                "dataset_doi": "10.15129/9ab14b0e-19ac-4279-938f-27f643078cec",
                "homes": homes, "target_order": CHANNELS, "requested_counts": [4, 3, 2, 3],
                "rho": .1, "event_threshold_ratio": .01, "shots": 256, "seeds": list(SEEDS),
                "methods": list(METHODS), "depth": 1, "maximum_intervals": 2,
                "transferred_angles": {"path": str(angles), "sha256": sha(angles),
                                       "gammas": original_angles["gammas"], "betas": original_angles["betas"]},
                "day_bootstrap": {"replicates": 2000, "seed_plus_house": 8101},
                "home_bootstrap": {"replicates": 10000, "seed": 9101},
                "code_sha256": {p: sha(ROOT / p) for p in CODE},
                "source_metadata_sha256": sha(ROOT / "data/external/refit/source.json"),
                "data_contract": "30 s fully supported left-hold blocks, consecutive source spacing <=16 s; Issues/nonfinite/negative exclusions; no extrapolation",
                "primary_metric": "equal-home mean macro appliance MAE, mean stochastic error sums over three seeds",
                "selection": "metadata-only eligible homes; timestamp-only 30 train and 30 test days, no replacements",
                "limitations": ["within-home supervised calibration, not zero-shot unseen-home transfer",
                                "cleaned REFIT includes unflagged imputations and asynchronous appliance observations",
                                "coarse two-level dishwasher and three-level residual background",
                                "ideal simulation, not QPU execution or quantum speed advantage",
                                "descriptive non-population bootstrap, no multiplicity-adjusted superiority claim"]}
    save_json(output / "protocol.json", protocol)
    print(f"Frozen protocol SHA256 {sha(output / 'protocol.json')}", flush=True)


def check(output, sources=False):
    protocol = read(output / "protocol.json")
    for path, digest in protocol["code_sha256"].items():
        if sha(ROOT / path) != digest:
            raise RuntimeError(f"Frozen source changed: {path}")
    angles = protocol["transferred_angles"]
    if sha(angles["path"]) != angles["sha256"]:
        raise RuntimeError("Transferred angles changed")
    if sources:
        for home in protocol["homes"]:
            if sha(home["source"]) != home["source_sha256"]:
                raise RuntimeError(f"REFIT source changed: {home['house']}")
    return protocol


def load(home, window):
    return read_block_window(home["source"], window["start_unix"], window["end_unix"],
                             channels=tuple(home["channels"]))


def fit_model(training_values):
    values = np.asarray(training_values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or not len(values):
        raise ValueError("Training needs nonempty aggregate and three-channel values")
    background = values[:, 0] - values[:, 1:].sum(axis=1)
    levels = [fit_power_levels(values[:, i + 1], n) for i, n in enumerate((4, 3, 2))]
    levels.append(fit_power_levels(background, 3))
    binary = [fit_power_levels(values[:, i + 1], 2) for i in range(3)]
    ranges = [max(float(np.ptp(x)), 1.) for x in levels]
    return {"levels": [x.tolist() for x in levels], "ranges": ranges,
            "penalties": (.1 * np.square(ranges)).tolist(),
            "thresholds": [(float(x[0]) + float(x[-1])) / 2 if len(x) > 1 else float(x[0]) + 1 for x in binary],
            "event_threshold": .01 * max(max(float(np.ptp(x)), 1.) for x in binary),
            "training_mean": [*values[:, 1:].mean(axis=0).tolist(), float(background.mean())],
            "training_blocks": len(values), "actual_counts": [len(x) for x in levels]}


def fit(output):
    config = check(output, sources=True)
    frozen = {"protocol_sha256": sha(output / "protocol.json"), "homes": {}}
    for home in config["homes"]:
        started = perf_counter()
        loaded = [(w, load(home, w)) for w in home["manifest"]["windows"] if w["split"] == "train"]
        usable = [x["values"] for _, x in loaded if len(x["timestamps"])]
        if not usable:
            raise ValueError(f"House {home['house']} has no supported training block; no adaptive replacement")
        quality = [{"window": w, "quality": x["quality"], "valid_blocks": len(x["timestamps"]),
                    "source_window_sha256": x["source_window_sha256"]} for w, x in loaded]
        extraction_s = perf_counter() - started
        started = perf_counter()
        model = fit_model(np.concatenate(usable))
        model.update({"extraction_wall_time_s": extraction_s, "fitting_wall_time_s": perf_counter() - started,
                      "quality": quality})
        frozen["homes"][str(home["house"])] = model
        print(f"Training only: house {home['house']}, {model['training_blocks']} blocks", flush=True)
    save_json(output / "frozen_models.json", frozen)
    check(output)
    print("All home models frozen before any REFIT test read", flush=True)


def predict(mains, timestamps, model, angles, house, day, shots=256):
    """Aggregate-only API: no held-out appliance references are accepted."""
    levels = [np.asarray(x) for x in model["levels"]]
    chunks = make_chunks(mains, timestamps, model["event_threshold"])
    window = {"chunks": chunks, "blocks": len(timestamps)}
    pending, traces = [], []
    for method in METHODS:
        seeds = SEEDS if method in ("qaoa_ideal", "uniform") else (None,)
        for seed in seeds:
            started = perf_counter()
            if method in ("qaoa_ideal", "uniform", "exact_chunk"):
                effective_seed = (seed or SEEDS[0]) + house * 1_000_000 + day * 10_000
                trace, elapsed = infer_chunks(chunks, levels, model["penalties"], angles["gammas"],
                                             angles["betas"], method, shots, effective_seed)
                prediction = expand_predictions(window, trace, levels)
                detail = {"records": trace, "effective_seed": effective_seed}
            elif method == "exact_full":
                prediction, detail = predict_temporal(mains, timestamps, levels, model["penalties"], model["event_threshold"])
                elapsed = detail["solver_wall_time_s"]
            else:
                prediction = np.tile(model["training_mean"], (len(timestamps), 1))
                elapsed = perf_counter() - started
                detail = {"training_mean": model["training_mean"]}
            pending.append((method, seed, prediction, elapsed))
            traces.append({"model": method, "seed": seed, "detail": detail, "solver_wall_time_s": elapsed})
    return pending, {"chunks": chunks, "traces": traces}


def evaluate(output):
    config = check(output, sources=True)
    models = read(output / "frozen_models.json")
    if models["protocol_sha256"] != sha(output / "protocol.json"):
        raise RuntimeError("Model/protocol mismatch")
    directory = output / "evaluation"
    if directory.exists():
        raise FileExistsError("Evaluation already started; preserve previous attempt")
    directory.mkdir()
    save_json(directory / "freeze.json", {"protocol_sha256": sha(output / "protocol.json"),
              "models_sha256": sha(output / "frozen_models.json"), "started_utc": datetime.now(timezone.utc).isoformat()})
    started = perf_counter()
    records, quality = [], []
    for home in config["homes"]:
        house = home["house"]
        model = models["homes"][str(house)]
        for day, window in enumerate(home["manifest"]["splits"]["test"]["windows"]):
            extracted = perf_counter()
            loaded = load(home, window)
            q = {"house": house, "window": window, "quality": loaded["quality"],
                 "valid_blocks": len(loaded["timestamps"]), "source_window_sha256": loaded["source_window_sha256"],
                 "extraction_wall_time_s": perf_counter() - extracted}
            quality.append(q)
            if not len(loaded["timestamps"]):
                continue
            pending, trace = predict(loaded["values"][:, 0], loaded["timestamps"], model,
                                     config["transferred_angles"], house, day, config["shots"])
            stem = f"house-{house:02d}_{window['id']}"
            # Save all predictions before calculating a test error. Appliance
            # values were loaded for common quality screening, never inference.
            with (directory / (stem + '.npz')).open('xb') as handle:
                np.savez_compressed(handle, timestamps=loaded["timestamps"],
                                    **{f"{m}_{s}": p for m, s, p, _ in pending})
            with gzip.open(directory / (stem + '.json.gz'), 'xt') as handle:
                json.dump(trace, handle, allow_nan=False)
            local = []
            for method, seed, prediction, elapsed in pending:
                local.append({"house": house, "window_id": window["id"], "model": method, "seed": seed,
                              "blocks": len(prediction), "runs": len(contiguous_slices(loaded["timestamps"])),
                              "chunks": len(trace["chunks"]), "solver_wall_time_s": elapsed,
                              "appliances": score_power(loaded["values"][:, 1:], prediction[:, :3],
                                                        loaded["timestamps"], model["thresholds"], CHANNELS),
                              "raw_aggregate_mae_w": float(np.abs(loaded["values"][:, 0] - prediction.sum(axis=1)).mean())})
            save_json(directory / (stem + '_scores.json'), {"quality": q, "records": local,
                      "predictions_sha256": sha(directory / (stem + '.npz')),
                      "trace_sha256": sha(directory / (stem + '.json.gz'))})
            records.extend(local)
            if (day + 1) % 10 == 0:
                print(f"REFIT test house {house}: {day + 1}/30 fixed windows", flush=True)
    save_json(output / "test_windows.json", records)
    save_json(output / "test_quality.json", quality)
    save_json(output / "timing.json", {"evaluation_wall_time_s": perf_counter() - started,
              "python": platform.python_version(), "numpy": np.__version__, "execution": "local ideal simulation and classical controls"})
    check(output, sources=True)
    if sha(output / "frozen_models.json") != read(directory / "freeze.json")["models_sha256"]:
        raise RuntimeError("Models changed during evaluation")


def errors_by_day(records, house, method):
    rows = [r for r in records if r["house"] == house and r["model"] == method]
    by_day = {}
    for name in sorted({r["window_id"] for r in rows}):
        group = [r for r in rows if r["window_id"] == name]
        expected = set(SEEDS) if method in ("qaoa_ideal", "uniform") else {None}
        if {r["seed"] for r in group} != expected or len(group) != len(expected):
            raise ValueError("Incomplete or duplicate stochastic replicate")
        if len({r["blocks"] for r in group}) != 1:
            raise ValueError("Different seed coverage")
        by_day[name] = {"error": float(np.mean([sum(a["absolute_error_sum_w"] for a in r["appliances"].values()) for r in group])),
                        "denominator": 3 * group[0]["blocks"]}
    return by_day


def paired_days(left, right, seed):
    if left.keys() != right.keys() or not left:
        raise ValueError("Require matched nonempty day sets")
    names = sorted(left)
    if any(left[k]["denominator"] != right[k]["denominator"] for k in names):
        raise ValueError("Coverage differs across methods")
    difference = np.array([left[k]["error"] - right[k]["error"] for k in names])
    counts = np.array([left[k]["denominator"] for k in names])
    draws = np.random.default_rng(seed).integers(0, len(names), (2000, len(names)))
    sample = difference[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"difference_w": float(difference.sum() / counts.sum()),
            "descriptive_95_interval_w": np.quantile(sample, [.025, .975]).tolist(),
            "days": len(names), "replicates": 2000, "seed": seed}


def summarize_records(records):
    homes = sorted({r["house"] for r in records})
    if not homes:
        return {"status": "no_evaluable_homes", "overall": {m: {
            "equal_home_macro_mae_w": None, "block_pooled_macro_mae_w": None,
            "evaluable_homes": 0, "blocks": 0, "windows": 0} for m in METHODS},
            "per_home": {}, "paired_day_comparisons": [], "paired_home_comparisons": [],
            "nonevaluable_homes": sorted(HOMES)}
    per_home = {}
    comparisons = []
    for house in homes:
        methods = {}
        for method in METHODS:
            days = errors_by_day(records, house, method)
            rows = [r for r in records if r["house"] == house and r["model"] == method]
            seed_metrics = []
            for seed in (SEEDS if method in ("qaoa_ideal", "uniform") else (None,)):
                subset = [r for r in rows if r["seed"] == seed]
                n = sum(r["blocks"] for r in subset)
                seed_metrics.append({"seed": seed, "appliances": pool_window_metrics(subset, CHANNELS),
                                     "aggregate_mae_w": sum(r["raw_aggregate_mae_w"] * r["blocks"] for r in subset) / n,
                                     "solver_wall_time_s": sum(r["solver_wall_time_s"] for r in subset)})
            total_error = sum(v["error"] for v in days.values())
            denom = sum(v["denominator"] for v in days.values())
            methods[method] = {"macro_mae_w": total_error / denom, "absolute_error_sum_w": total_error,
                               "blocks": denom // 3, "windows": len(days), "seed_metrics": seed_metrics,
                               "aggregate_mae_w": float(np.mean([r["aggregate_mae_w"] for r in seed_metrics])),
                               "appliance_mae_w": {c: float(np.mean([r["appliances"][c]["mae_w"] for r in seed_metrics])) for c in CHANNELS}}
        per_home[str(house)] = methods
        for right in METHODS[1:]:
            comparisons.append({"house": house, "left": "qaoa_ideal", "right": right,
                                **paired_days(errors_by_day(records, house, "qaoa_ideal"),
                                              errors_by_day(records, house, right), 8101 + house)})
    overall = {}
    for method in METHODS:
        rows = [per_home[str(h)][method] for h in homes]
        overall[method] = {"equal_home_macro_mae_w": float(np.mean([r["macro_mae_w"] for r in rows])),
                           "block_pooled_macro_mae_w": sum(r["absolute_error_sum_w"] for r in rows) / (3 * sum(r["blocks"] for r in rows)),
                           "evaluable_homes": len(rows), "blocks": sum(r["blocks"] for r in rows),
                           "windows": sum(r["windows"] for r in rows)}
    home_comparisons = []
    for right in METHODS[1:]:
        difference = np.array([per_home[str(h)]["qaoa_ideal"]["macro_mae_w"] - per_home[str(h)][right]["macro_mae_w"] for h in homes])
        draw = np.random.default_rng(9101).integers(0, len(homes), (10000, len(homes)))
        home_comparisons.append({"left": "qaoa_ideal", "right": right, "difference_w": float(difference.mean()),
                                 "descriptive_95_interval_w": np.quantile(difference[draw].mean(axis=1), [.025, .975]).tolist(),
                                 "homes": len(homes), "replicates": 10000, "seed": 9101})
    return {"status": "complete", "overall": overall, "per_home": per_home, "paired_day_comparisons": comparisons,
            "paired_home_comparisons": home_comparisons,
            "nonevaluable_homes": sorted(set(HOMES) - set(homes))}


def validate_coverage(config, quality, records):
    expected = {(h["house"], w["id"]): w for h in config["homes"]
                for w in h["manifest"]["splits"]["test"]["windows"]}
    keyed = {(q["house"], q["window"]["id"]): q for q in quality}
    if len(keyed) != len(quality) or keyed.keys() != expected.keys():
        raise ValueError("Quality archive must contain every selected window exactly once")
    if any(q["window"] != expected[k] for k, q in keyed.items()):
        raise ValueError("Quality archive timestamp manifest mismatch")
    expected_arms = {(m, s) for m in METHODS for s in
                     (SEEDS if m in ("qaoa_ideal", "uniform") else (None,))}
    if any((r["house"], r["window_id"]) not in keyed for r in records):
        raise ValueError("Unselected score window")
    per_home = {}
    for key, q in keyed.items():
        rows = [r for r in records if (r["house"], r["window_id"]) == key]
        n = q["valid_blocks"]
        if n == 0 and rows:
            raise ValueError("Empty window cannot have score rows")
        if n and (len(rows) != len(expected_arms) or {(r["model"], r["seed"]) for r in rows} != expected_arms):
            raise ValueError("Every nonempty window needs exactly nine method/seed records")
        if any(r["blocks"] != n or any(a["n_blocks"] != n for a in r["appliances"].values()) for r in rows):
            raise ValueError("Prediction/reference coverage mismatch")
        h = per_home.setdefault(str(key[0]), {"selected_windows": 0, "nonempty_windows": 0,
             "expected_blocks": 0, "valid_blocks": 0, "empty_windows": [],
             "issues_rows": 0, "large_gap_seconds": 0., "invalid_supported_seconds": 0.,
             "valid_source_rows_including_support": 0, "all_appliances_zero_source_rows_including_support": 0,
             "all_appliances_zero_retained_blocks": 0, "zero_retained_blocks_by_channel": {}})
        h["selected_windows"] += 1
        h["nonempty_windows"] += int(n > 0)
        h["expected_blocks"] += q["quality"]["expected_complete_blocks"]
        h["valid_blocks"] += n
        if not n:
            h["empty_windows"].append(key[1])
        for target, source in (("issues_rows", "rows_issues"),
                               ("large_gap_seconds", "unsupported_large_gap_seconds"),
                               ("invalid_supported_seconds", "supported_invalid_seconds"),
                               ("valid_source_rows_including_support", "valid_source_rows"),
                               ("all_appliances_zero_source_rows_including_support", "all_selected_appliances_zero_source_rows"),
                               ("all_appliances_zero_retained_blocks", "all_selected_appliances_zero_retained_blocks")):
            h[target] += q["quality"][source]
        for c, count in q["quality"]["zero_retained_blocks_by_channel"].items():
            h["zero_retained_blocks_by_channel"][c] = h["zero_retained_blocks_by_channel"].get(c, 0) + count
    for h in per_home.values():
        h["coverage_fraction"] = h["valid_blocks"] / h["expected_blocks"]
        h["all_appliances_zero_retained_fraction"] = h["all_appliances_zero_retained_blocks"] / h["valid_blocks"] if h["valid_blocks"] else None
    return per_home


def summarize(output):
    config = check(output)
    records = read(output / "test_windows.json")
    coverage = validate_coverage(config, read(output / "test_quality.json"), records)
    result = summarize_records(records)
    result.update({"protocol_sha256": sha(output / "protocol.json"), "limitations": config["limitations"],
                   "timing": read(output / "timing.json"), "coverage": coverage})
    save_json(output / "summary.json", result)
    print(json.dumps(result["overall"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("freeze", "fit", "evaluate", "summarize"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/stage_g/refit/run_001")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/external/refit/cleaned")
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args.output_dir, args.source_dir)
    else:
        {"fit": fit, "evaluate": evaluate, "summarize": summarize}[args.mode](args.output_dir)


if __name__ == "__main__":
    main()
