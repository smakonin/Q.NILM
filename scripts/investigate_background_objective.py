#!/usr/bin/env python3
"""Reproducible background/objective diagnostics; TRAIN and VALIDATION only."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scripts import background_objective_core as core
from quantum_nilm.heldout_data import read_block_window, source_bounds
from quantum_nilm.multistate import fit_power_levels
from quantum_nilm.evaluation import predict_temporal

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/heldout_campaign"
PREV = ROOT / "results/quantum_diagnostics/training_validation_001"
CODE = ["scripts/background_objective_core.py", "scripts/investigate_background_objective.py",
        "tests/test_background_objective.py", "docs/background_objective_protocol.md",
        "src/quantum_nilm/heldout_data.py", "src/quantum_nilm/multistate.py", "src/quantum_nilm/evaluation.py"]


def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f: json.dump(value, f, indent=2, allow_nan=False); f.write("\n")
def npz(p, **arrays):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f: np.savez_compressed(f, **arrays)


def candidates():
    rows = []
    for k in (1, 3, 6, 12):
        rows.append(dict(id=f"sse_k{k}", family="sse", k=k, weight=0, prior_scope="all"))
        for w in (100, 1000, 10000):
            rows.append(dict(id=f"prior_all_k{k}_w{w}", family="sse", k=k, weight=w, prior_scope="all"))
    for scope in ("appliances", "background"):
        for w in (100, 1000, 10000):
            rows.append(dict(id=f"prior_{scope}_k3_w{w}", family="sse", k=3, weight=w, prior_scope=scope))
    for k in (3, 12):
        for w in (0, 1):
            rows.append(dict(id=f"mixture_k{k}_prior{w}", family="mixture", k=k, weight=w))
    for rho in (.02, .1):
        for bg in (False, True):
            rows.append(dict(id=f"temporal_rho{rho:g}_bg{int(bg)}", family="temporal", k=3, rho=rho, background_switch=bg))
    return rows


def freeze(folder):
    core.require(not folder.exists(), "Use a new named archive")
    old = read(OLD/"protocol.json"); vm = read(PREV/"validation_data.json"); oldmodel = read(OLD/"frozen_models.json")
    core.require(oldmodel["protocol_sha256"] == sha(OLD/"protocol.json"), "Original model lineage")
    core.require(vm["plan_sha256"] == sha(PREV/"plan.json"), "Validation lineage")
    paths = [OLD/"protocol.json", OLD/"frozen_models.json", OLD/"data_quality.json", PREV/"plan.json", PREV/"validation_data.json"]
    for item in vm["windows"]:
        core.require(item["window"]["split"] == "validation" and sha(PREV/item["file"]) == item["sha256"], "Validation input only")
        paths.append(PREV/item["file"])
    source = Path(old["source"])
    core.require(source.stat().st_size == old["source_size_bytes"] and source.stat().st_mtime_ns == old["source_mtime_ns"], "R1Hz changed")
    plan = {"created_utc": now(), "scope": "Exploratory training/validation diagnosis only; no final-test loading or scoring, no production/hardware/paper changes",
            "source": str(source), "source_size_bytes": source.stat().st_size, "source_mtime_ns": source.stat().st_mtime_ns,
            "source_bounds": list(source_bounds(source)), "source_global_ordering": "assumed; checked only in selected source windows",
            "source_sha256": {p: sha(ROOT/p) for p in CODE},
            "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
            "versions": {p: version(p) for p in ("numpy", "scipy")},
            "training_windows": old["manifest"]["splits"]["train"]["windows"],
            "validation_windows": vm["windows"], "candidate_grid": candidates(),
            "model": oldmodel["models"]["multistate"],
            "selection": "Lowest pooled three-appliance measured MAE among deployable-input diagnostic candidates; ties lexicographic ID; no final confirmation",
            "controls": ["raw_mains", "quantized_targets_true_background", "measured_targets_quantized_background", "quantized_targets_quantized_background", "known_background", "representation_oracle", "low_power_constant"],
            "hardware_jobs": 0}
    save(folder/"plan.json", plan)
    print(json.dumps({"plan_sha256": sha(folder/"plan.json"), "candidates": len(candidates()), "training_days": 90, "validation_days": 10}), flush=True)


def verify(folder):
    plan = read(folder/"plan.json")
    for key in ("source_sha256", "input_sha256"):
        for path, digest in plan[key].items(): core.require(sha(ROOT/path) == digest, f"Frozen input changed: {path}")
    for p, v in plan["versions"].items(): core.require(version(p) == v, "Version changed")
    source = Path(plan["source"])
    core.require(source.stat().st_size == plan["source_size_bytes"] and source.stat().st_mtime_ns == plan["source_mtime_ns"], "Source changed")
    return plan


def extract(folder):
    plan = verify(folder); old = {r["window"]["id"]: r for r in read(OLD/"data_quality.json")["train"]}
    records = []; started = perf_counter()
    core.require(not (folder/"training_data.json").exists(), "Extraction already complete")
    for i, window in enumerate(plan["training_windows"]):
        receipt = folder/"training"/(window["id"]+".json"); file = receipt.with_suffix(".npz")
        if receipt.exists():
            r = read(receipt); core.require(sha(file) == r["sha256"], "Training cache changed")
        else:
            core.require(window["split"] == "train", "Training only")
            loaded = read_block_window(plan["source"], window["start"], window["end"])
            expected = old[window["id"]]
            core.require(loaded["source_window_sha256"] == expected["source_window_sha256"] and len(loaded["values"]) == expected["valid_blocks"], "Training source mismatch")
            npz(file, values=loaded["values"], timestamps=loaded["timestamps"])
            r = {"window": window, "file": str(file.relative_to(folder)), "sha256": sha(file),
                 "source_window_sha256": loaded["source_window_sha256"], "quality": loaded["quality"], "blocks": len(loaded["values"])}
            save(receipt, r)
        records.append(r)
        if (i+1) % 10 == 0: print(f"Training extraction {i+1}/90", flush=True)
    save(folder/"training_data.json", {"created_utc": now(), "plan_sha256": sha(folder/"plan.json"), "windows": records,
                                      "blocks": sum(r["blocks"] for r in records), "runtime_s": perf_counter()-started})


def load_training(folder):
    receipt = read(folder/"training_data.json")
    core.require(receipt["plan_sha256"] == sha(folder/"plan.json"), "Training binding")
    for r in receipt["windows"]: core.require(sha(folder/r["file"]) == r["sha256"], "Training checksum")
    return np.concatenate([np.load(folder/r["file"], allow_pickle=False)["values"] for r in receipt["windows"]])


def fit(folder):
    plan = verify(folder); values = load_training(folder); bg = values[:, 0]-values[:, 1:].sum(axis=1)
    levels = [np.asarray(x) for x in plan["model"]["levels_w"]]; models = {}; failures = []
    for k in (1, 3, 6, 12):
        try: centres = fit_power_levels(bg, k)
        except ValueError as exc:
            if k == 3: raise
            failures.append({"k": k, "error": str(exc)}); continue
        if k == 3: core.require(np.allclose(centres, levels[3], rtol=0, atol=1e-8), "Original background fit not reproduced")
        ls = [*levels[:3], centres]; prior = core.fit_frequencies(np.column_stack((values[:, 1:], bg)), ls)
        models[str(k)] = {"levels": [x.tolist() for x in ls], "priors": [x.tolist() for x in prior],
                          "mixture": core.fit_background_mixture(bg, centres)}
    thresholds = [(levels[0][1]+levels[0][2])/2, (levels[1][0]+levels[1][1])/2, (levels[2][0]+levels[2][1])/2]
    save(folder/"fitted.json", {"created_utc": now(), "plan_sha256": sha(folder/"plan.json"),
         "training_data_sha256": sha(folder/"training_data.json"), "models": models, "failed_fits": failures,
         "active_thresholds_w": thresholds, "training_blocks": len(values), "training_background_quantiles_w": core.quantiles(bg),
         "training_active_blocks": (values[:, 1:] > thresholds).sum(axis=0).tolist(),
         "training_negative_background_blocks": int((bg < 0).sum())})


def profile(values, model):
    bg = values[:, 0]-values[:, 1:].sum(axis=1); lev = np.asarray(model["levels"][3]); q = lev[core.nearest(bg, lev)]
    error = bg-q
    return {"blocks": len(values), "background_quantiles_w": core.quantiles(bg), "quantization_error_quantiles_w": core.quantiles(error),
            "background_quantization_mae_w": float(abs(error).mean()), "background_quantization_rmse_w": float(np.sqrt(np.mean(error**2))),
            "below_min_centroid_blocks": int((bg < lev.min()).sum()), "above_max_centroid_blocks": int((bg > lev.max()).sum()),
            "negative_background_blocks": int((bg < 0).sum()), "background_category_counts": np.bincount(core.nearest(bg, lev), minlength=len(lev)).tolist()}


def infer_candidate(values, times, spec, model):
    levels, priors = model["levels"], model["priors"]
    if spec["family"] == "sse":
        pred, _, cost = core.infer_static(values[:, 0], levels, priors, weight_w2=spec["weight"], prior_scope=spec["prior_scope"])
        return pred, {"objective_sum": float(cost.sum()), "objective_units": "W2"}
    if spec["family"] == "mixture":
        pred, _, cost = core.infer_marginal(values[:, 0], levels[:3], model["mixture"], priors[:3], spec["weight"])
        return pred, {"objective_sum": float(cost.sum()), "objective_units": "negative log-density plus log-prior"}
    penalty = spec["rho"]*np.array([max(float(np.ptp(x)), 1)**2 for x in levels])
    if not spec["background_switch"]: penalty[3] = 0
    pred, info = predict_temporal(values[:, 0], times, levels, penalty)
    return pred, {"objective_sum": info["full_block_objective_energy"], "objective_units": "W2", "penalty_w2": penalty.tolist()}


def control_predictions(values, levels):
    truth = values[:, 1:]; bg = values[:, 0]-truth.sum(axis=1)
    q = np.column_stack([np.asarray(x)[core.nearest(truth[:, i], x)] for i, x in enumerate(levels[:3])])
    qbg = np.asarray(levels[3])[core.nearest(bg, levels[3])]
    signals = {"raw_mains": values[:, 0], "quantized_targets_true_background": q.sum(axis=1)+bg,
               "measured_targets_quantized_background": truth.sum(axis=1)+qbg,
               "quantized_targets_quantized_background": q.sum(axis=1)+qbg}
    result = {name: core.infer_static(signal, levels)[0] for name, signal in signals.items()}
    selected = core.infer_static(truth.sum(axis=1), levels[:3])[0]
    result["known_background"] = np.column_stack((selected, bg))
    result["representation_oracle"] = np.column_stack((q, bg))
    result["low_power_constant"] = np.tile([x[0] for x in levels], (len(values), 1))
    return result


def evaluate(folder):
    plan = verify(folder); fitted = read(folder/"fitted.json")
    core.require(fitted["plan_sha256"] == sha(folder/"plan.json"), "Fitted binding")
    core.require(not (folder/"results.json").exists(), "Results already complete")
    rows = []; controls = []; values_all = []; predictions_all = {}; day_ids = []
    for item in plan["validation_windows"]:
        path = PREV/item["file"]; core.require(sha(path) == item["sha256"], "Validation checksum")
        data = np.load(path, allow_pickle=False); values, times = data["values"], data["timestamps"]
        window = item["window"]; day_ids.extend([window["id"]]*len(values)); values_all.append(values)
        output = folder/"validation"/(window["id"]+".json"); binary = output.with_suffix(".npz")
        if output.exists():
            receipt = read(output); core.require(receipt["fitted_sha256"] == sha(folder/"fitted.json") and sha(binary) == receipt["predictions_sha256"], "Window binding")
            arrays = np.load(binary, allow_pickle=False)
            rowset, controlset = receipt["candidates"], receipt["controls"]
        else:
            arrays = {}; rowset = []; controlset = []
            for spec in plan["candidate_grid"]:
                if str(spec["k"]) not in fitted["models"]: continue
                started = perf_counter(); model = fitted["models"][str(spec["k"])]; pred, extra = infer_candidate(values, times, spec, model)
                arrays[spec["id"]] = pred
                rowset.append({"id": spec["id"], "window": window["id"], "runtime_s": perf_counter()-started, "objective": extra,
                               "metrics": core.metrics(values, pred, model["levels"], fitted["active_thresholds_w"])})
            for name, pred in control_predictions(values, fitted["models"]["3"]["levels"]).items():
                arrays["control_"+name] = pred
                controlset.append({"id": name, "window": window["id"], "metrics": core.metrics(values, pred, fitted["models"]["3"]["levels"], fitted["active_thresholds_w"])})
            npz(binary, **arrays)
            save(output, {"plan_sha256": sha(folder/"plan.json"), "fitted_sha256": sha(folder/"fitted.json"), "source_npz_sha256": sha(path),
                          "predictions_sha256": sha(binary), "candidates": rowset, "controls": controlset})
        for key in arrays: predictions_all.setdefault(key, []).append(arrays[key])
        rows.extend(rowset); controls.extend(controlset)
        print(f"Validated {window['id']}: {len(rowset)} candidates and {len(controlset)} labelled controls", flush=True)
    values = np.concatenate(values_all); predictions = {k: np.concatenate(v) for k, v in predictions_all.items()}
    pooled = {name: core.pool([r["metrics"] for r in rows if r["id"] == name]) for name in sorted({r["id"] for r in rows})}
    cp = {name: core.pool([r["metrics"] for r in controls if r["id"] == name]) for name in sorted({r["id"] for r in controls})}
    levels = fitted["models"]["3"]["levels"]; baseline = predictions["sse_k3"]; truth = values[:, 1:]
    bg = values[:, 0]-truth.sum(axis=1); qbg = np.asarray(levels[3])[core.nearest(bg, levels[3])]
    qerror = bg-qbg; edges = [-np.inf, -100, -25, 25, 100, np.inf]; bins = []
    oracleerr = abs(predictions["control_representation_oracle"][:, :3]-truth).mean(axis=1)
    baseerr = abs(baseline[:, :3]-truth).mean(axis=1)
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (qerror >= a) & (qerror < b)
        bins.append({"lower_inclusive_w": None if not np.isfinite(a) else a, "upper_exclusive_w": None if not np.isfinite(b) else b,
                     "blocks": int(mask.sum()), "baseline_absolute_error_sum_w": float(baseerr[mask].sum()),
                     "oracle_absolute_error_sum_w": float(oracleerr[mask].sum()),
                     "excess_mae_w": float((baseerr[mask]-oracleerr[mask]).mean()) if mask.any() else None})
    alltrain = load_training(folder)
    profiles = {k: {"train": profile(alltrain, m), "validation": profile(values, m)} for k, m in fitted["models"].items()}
    examples = []
    for i in np.argsort(-(baseerr-oracleerr), kind="stable")[:10]:
        examples.append({"pooled_index": int(i), "day": day_ids[i], "observed_mains_targets_w": values[i].tolist(),
                         "baseline_prediction_w": baseline[i].tolist(), "true_background_w": float(bg[i]), "nearest_background_w": float(qbg[i]),
                         "oracle_target_w": predictions["control_representation_oracle"][i, :3].tolist()})
    winner = min(pooled, key=lambda k: (pooled[k]["macro_mae_w"], k))
    staticwinner = min((s["id"] for s in plan["candidate_grid"] if s["family"] == "sse" and s["id"] in pooled), key=lambda k: (pooled[k]["macro_mae_w"], k))
    save(folder/"results.json", {"completed_utc": now(), "plan_sha256": sha(folder/"plan.json"), "fitted_sha256": sha(folder/"fitted.json"),
          "validation_blocks": len(values), "validation_days": 10, "candidate_results": pooled, "control_results": cp,
          "candidate_window_results": rows, "control_window_results": controls, "profiles": profiles,
          "ambiguity": core.ambiguity(values, levels), "background_quantization_error_bins": bins, "largest_excess_error_examples": examples,
          "exploratory_validation_winner": winner, "quadratic_static_validation_winner": staticwinner,
          "validation_active_blocks": (truth > fitted["active_thresholds_w"]).sum(axis=0).tolist(),
          "limits": [plan["scope"], "Ten reused development days from one home; no independent held-out claim",
                     "Label-based interventions are diagnostic controls, not deployable models", "Temporal arms use whole-day future mains, not two-interval or causal-online inference",
                     "Mixture likelihood is not asserted to be QUBO compatible", "Source ordering verified within windows only; no new final-test data accessed"]})
    print(json.dumps({"validation_winner": winner, "static_quadratic_winner": staticwinner, "baseline_mae_w": pooled["sse_k3"]["macro_mae_w"],
                      "winner_mae_w": pooled[winner]["macro_mae_w"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--mode", choices=("freeze", "verify", "extract", "fit", "evaluate"), required=True)
    args = parser.parse_args(); folder = args.folder.resolve()
    core.require(folder.is_relative_to(ROOT/"results/quantum_diagnostics") and folder != ROOT/"results/quantum_diagnostics", "Named archive required")
    {"freeze": freeze, "verify": verify, "extract": extract, "fit": fit, "evaluate": evaluate}[args.mode](folder)
