#!/usr/bin/env python3
"""Independent numerical cross-check of the development-only diagnostic."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import itertools
import json
from pathlib import Path
import numpy as np
from scripts.audit_training_validation import independent_window

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT/"results/quantum_diagnostics/training_validation_001"


def read(p): return json.loads(Path(p).read_text())
def sha(p): return sha256(Path(p).read_bytes()).hexdigest()
def require(ok, message):
    if not ok: raise ValueError(message)
def close(a, b, message, atol=1e-7):
    require(np.allclose(a, b, rtol=1e-10, atol=atol), message)
def reference(levels):
    states = np.array([x[::-1] for x in itertools.product(*[range(len(x)) for x in levels[::-1]])])
    powers = np.array([[levels[i][s[i]] for i in range(len(levels))] for s in states])
    return states, powers
def labels(values, levels):
    return np.column_stack([np.searchsorted((np.asarray(x)[1:]+np.asarray(x)[:-1])/2, values[:, i], side="left") for i, x in enumerate(levels)])


def direct_static(values, spec, model):
    levels = model["levels"]
    state, power = reference(levels if spec["family"] == "sse" else levels[:3])
    if spec["family"] == "sse":
        chosen = range(4) if spec["prior_scope"] == "all" else (range(3) if spec["prior_scope"] == "appliances" else [3])
        penalty = np.array([sum(-np.log(model["priors"][i][s[i]]) for i in chosen) for s in state])
        score = (values[:, 0, None]-power.sum(axis=1))**2+spec["weight"]*penalty
        ids = score.argmin(axis=1)
        return power[ids], score[np.arange(len(values)), ids]
    mix = model["mixture"]; b, sd, w = [np.asarray(mix[k]) for k in ("centres_w", "sd_w", "weights")]
    residual = values[:, 0, None]-power.sum(axis=1)
    terms = np.log(w/sd/np.sqrt(2*np.pi))-.5*((residual[:, :, None]-b)/sd)**2
    peak = terms.max(axis=2); scaled = np.exp(terms-peak[:, :, None]); logdensity = peak+np.log(scaled.sum(axis=2))
    penalty = np.array([sum(-np.log(model["priors"][i][s[i]]) for i in range(3)) for s in state])
    scores = -logdensity+spec["weight"]*penalty; ids = scores.argmin(axis=1); row = np.arange(len(values))
    posterior = scaled[row, ids]/scaled[row, ids].sum(axis=1)[:, None]
    return np.column_stack((power[ids], posterior@b)), scores[row, ids]


def temporal_optimum(values, times, levels, spec):
    states, powers = reference(levels)
    penalties = spec["rho"]*np.array([max(np.ptp(x), 1)**2 for x in levels])
    if not spec["background_switch"]: penalties[3] = 0
    transition = np.sum((states[:, None, :] != states[None, :, :])*penalties, axis=2)
    energy = (values[:, 0, None]-powers.sum(axis=1))**2
    edges = np.r_[0, np.flatnonzero(np.diff(times) != 30)+1, len(times)]
    optimum = 0.
    for a, b in zip(edges[:-1], edges[1:]):
        costs = energy[a].copy()
        for i in range(a+1, b): costs = np.min(costs[:, None]+transition, axis=0)+energy[i]
        optimum += costs.min()
    return optimum, penalties


def audit_metrics(values, prediction, levels, thresholds, saved):
    true, predicted = values[:, 1:4], prediction[:, :3]
    err = abs(true-predicted); n = len(true)
    close(err.sum(axis=0), saved["absolute_error_sum_w"], "Error sums")
    close(err.mean(axis=0), saved["per_appliance_mae_w"], "Per-channel errors")
    close(err.mean(), saved["macro_mae_w"], "Macro error")
    tstate, pstate = labels(true, levels[:3]), labels(predicted, levels[:3])
    ta, pa = true > thresholds, predicted > thresholds
    checks = {"blocks": n, "category_correct": (tstate==pstate).sum(axis=0), "all_categories_correct": np.all(tstate==pstate, axis=1).sum(),
              "active_blocks": ta.sum(axis=0), "predicted_active_blocks": pa.sum(axis=0), "tp": (ta&pa).sum(axis=0),
              "fp": (~ta&pa).sum(axis=0), "fn": (ta&~pa).sum(axis=0)}
    for key, value in checks.items(): require(np.array_equal(value, saved[key]), key)
    close((err*ta).sum(axis=0), saved["active_absolute_error_sum_w"], "Active errors")
    close((err*~ta).sum(axis=0), saved["inactive_absolute_error_sum_w"], "Inactive errors")
    for i, x in enumerate(levels[:3]):
        confusion = [[int(((tstate[:, i] == a)&(pstate[:, i] == b)).sum()) for b in range(len(x))] for a in range(len(x))]
        require(confusion == saved["confusion"][i], "Confusion")
    close(np.square(values[:, 0]-prediction.sum(axis=1)).sum(), saved["aggregate_squared_error_sum_w2"], "Aggregate error")
    close(abs(prediction[:, 3]-(values[:, 0]-true.sum(axis=1))).sum(), saved["background_absolute_error_sum_w"], "Background error")


def main(folder):
    plan = read(folder/"plan.json"); fitted = read(folder/"fitted.json"); result = read(folder/"results.json")
    require(result["plan_sha256"] == fitted["plan_sha256"] == sha(folder/"plan.json"), "Plan binding")
    require(result["fitted_sha256"] == sha(folder/"fitted.json"), "Fitted binding")
    for group in ("source_sha256", "input_sha256"):
        for name, digest in plan[group].items(): require(sha(ROOT/name) == digest, "Frozen source/input")
    training = read(folder/"training_data.json"); require(fitted["training_data_sha256"] == sha(folder/"training_data.json"), "Training receipt")
    train = []; reaggregated = 0; nsource = 0
    for i, entry in enumerate(training["windows"]):
        require(sha(folder/entry["file"]) == entry["sha256"], "Training file")
        arr = np.load(folder/entry["file"], allow_pickle=False); train.append(arr["values"])
        require(entry["window"]["split"] == "train", "Training scope")
        if i in (0, 29, 59, 89):
            w = entry["window"]; t, v, digest, _ = independent_window(plan["source"], w["start"], w["end"])
            require(np.array_equal(t, arr["timestamps"]) and digest == entry["source_window_sha256"], "Independent training source")
            close(v, arr["values"], "Training aggregation", 1e-9); reaggregated += 1; nsource += len(v)
    train = np.concatenate(train); bg = train[:, 0]-train[:, 1:].sum(axis=1)
    for model in fitted["models"].values():
        ls = model["levels"]; lbl = labels(np.column_stack((train[:, 1:], bg)), ls)
        for i, centres in enumerate(ls):
            counts = np.array([(lbl[:, i]==j).sum() for j in range(len(centres))])
            close((counts+1)/(len(train)+len(centres)), model["priors"][i], "Independent priors", 1e-12)
            if i == 3:
                for j, centre in enumerate(centres):
                    close(bg[lbl[:, 3] == j].mean(), centre, "Background centre stationary", 1e-8)
                    close(max(1, bg[lbl[:, 3] == j].std()), model["mixture"]["sd_w"][j], "Background SD", 1e-8)
    specs = {x["id"]: x for x in plan["candidate_grid"]}; preds = {}; values_all = []; rows = []; controls = []
    nstatic = ntemporal = 0; maxdiscrepancy = 0.; max_temporal_gap = 0.
    for entry in plan["validation_windows"]:
        w = entry["window"]; require(w["split"] == "validation", "Validation only")
        t, values, digest, _ = independent_window(plan["source"], w["start"], w["end"])
        saved = np.load(PREV/entry["file"], allow_pickle=False)
        require(np.array_equal(t, saved["timestamps"]) and digest == entry["source_window_sha256"], "Validation raw data")
        close(values, saved["values"], "Validation aggregation", 1e-9); reaggregated += 1; nsource += len(t)
        receipt = read(folder/"validation"/(w["id"]+".json")); binary = folder/"validation"/(w["id"]+".npz")
        require(sha(binary) == receipt["predictions_sha256"] and receipt["fitted_sha256"] == sha(folder/"fitted.json"), "Prediction binding")
        output = np.load(binary, allow_pickle=False); levels = fitted["models"]["3"]["levels"]
        for row in receipt["candidates"]:
            spec = specs[row["id"]]; model = fitted["models"][str(spec["k"])]; p = output[row["id"]]
            audit_metrics(values, p, model["levels"], fitted["active_thresholds_w"], row["metrics"])
            if spec["family"] != "temporal":
                expected, score = direct_static(values, spec, model)
                close(expected, p, "Independent static/mixture prediction", 1e-8)
                close(score.sum(), row["objective"]["objective_sum"], "Independent static objective")
                maxdiscrepancy = max(maxdiscrepancy, float(abs(expected-p).max())); nstatic += 1
            else:
                optimum, penalty = temporal_optimum(values, t, model["levels"], spec)
                adjacent = np.diff(t) == 30
                observed = np.square(values[:, 0]-p.sum(axis=1)).sum()+np.sum((p[1:] != p[:-1])*adjacent[:, None]*penalty)
                for channel, ls in enumerate(model["levels"]): require(np.isin(p[:, channel], ls).all(), "Temporal feasibility")
                close(optimum, observed, "Independent temporal optimum", 1e-3)
                close(observed, row["objective"]["objective_sum"], "Temporal objective", 1e-3)
                max_temporal_gap = max(max_temporal_gap, abs(float(observed-optimum))); ntemporal += 1
            preds.setdefault(row["id"], []).append(p)
        truth = values[:, 1:]; bg = values[:, 0]-truth.sum(axis=1)
        lab = labels(truth, levels[:3]); q = np.column_stack([np.asarray(ls)[lab[:, i]] for i, ls in enumerate(levels[:3])])
        qbg = np.asarray(levels[3])[labels(bg[:, None], [levels[3]])[:, 0]]
        signal = {"raw_mains": values[:, 0], "quantized_targets_true_background": q.sum(axis=1)+bg,
                  "measured_targets_quantized_background": truth.sum(axis=1)+qbg,
                  "quantized_targets_quantized_background": q.sum(axis=1)+qbg}
        _, power = reference(levels)
        for row in receipt["controls"]:
            name = row["id"]; p = output["control_"+name]
            audit_metrics(values, p, levels, fitted["active_thresholds_w"], row["metrics"])
            if name in signal:
                ids = abs(signal[name][:, None]-power.sum(axis=1)).argmin(axis=1); close(p, power[ids], "Control prediction")
            elif name == "known_background":
                _, target = reference(levels[:3]); ids = abs(truth.sum(axis=1)[:, None]-target.sum(axis=1)).argmin(axis=1)
                close(p, np.column_stack((target[ids], bg)), "Known-background control")
            elif name == "representation_oracle": close(p, np.column_stack((q, bg)), "Oracle control")
            else: close(p, np.tile([ls[0] for ls in levels], (len(values), 1)), "Constant baseline")
            preds.setdefault("control_"+name, []).append(p)
        values_all.append(values); rows.extend(receipt["candidates"]); controls.extend(receipt["controls"])
        print(f"Independently checked {w['id']}", flush=True)
    values = np.concatenate(values_all)
    for prefix, table in [("", "candidate_results"), ("control_", "control_results")]:
        for name, score in result[table].items():
            p = np.concatenate(preds[prefix+name]); error = abs(p[:, :3]-values[:, 1:])
            close(error.mean(axis=0), score["per_appliance_mae_w"], "Pooled appliance MAE")
            close(error.mean(), score["macro_mae_w"], "Pooled macro MAE")
    require(rows == result["candidate_window_results"] and controls == result["control_window_results"], "Window coverage")
    require(len(values) == result["validation_blocks"], "Block denominator")
    winner = min(result["candidate_results"], key=lambda k: (result["candidate_results"][k]["macro_mae_w"], k))
    require(winner == result["exploratory_validation_winner"], "Selection")
    for key, model in fitted["models"].items():
        for split, v in [("train", train), ("validation", values)]:
            residual = v[:, 0]-v[:, 1:].sum(axis=1); ls = np.asarray(model["levels"][3]); ids = labels(residual[:, None], [ls])[:, 0]
            p = result["profiles"][key][split]
            close(abs(residual-ls[ids]).mean(), p["background_quantization_mae_w"], "Background quantization")
            close(np.sqrt(np.square(residual-ls[ids]).mean()), p["background_quantization_rmse_w"], "Background quantization RMSE")
    bins = result["background_quantization_error_bins"]
    require(sum(b["blocks"] for b in bins) == len(values), "Excess-error bin coverage")
    close(sum(b["baseline_absolute_error_sum_w"]-b["oracle_absolute_error_sum_w"] for b in bins)/len(values),
          result["candidate_results"]["sse_k3"]["macro_mae_w"]-result["control_results"]["representation_oracle"]["macro_mae_w"], "Excess-error reconciliation")
    report = {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": sha(folder/"plan.json"),
              "results_sha256": sha(folder/"results.json"), "audit_sha256": sha(__file__),
              "independent_source_parser_sha256": sha(ROOT/"scripts/audit_training_validation.py"),
              "source_windows_independently_reaggregated": reaggregated, "source_blocks_independently_reaggregated": nsource,
              "static_or_mixture_window_checks": nstatic, "temporal_global_optimality_window_checks": ntemporal,
              "label_control_window_checks": len(controls), "max_prediction_discrepancy_w": maxdiscrepancy,
              "max_temporal_total_objective_discrepancy_w2": max_temporal_gap,
              "limits": ["Four of ninety training days and all ten validation days independently reaggregated; all training artifact hashes checked",
                         "Background fit stationarity and statistics checked, not an independent k-means initialization rerun",
                         "No final-test or physical-hardware evaluation"]}
    with (folder/"independent_audit.json").open("x") as f: json.dump(report, f, indent=2); f.write("\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--folder", type=Path, required=True)
    main(parser.parse_args().folder.resolve())
