#!/usr/bin/env python3
"""Post-hoc interpretation, not additional fitting or test-set selection."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from scripts.background_objective_core import basis, nearest

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT/"results/quantum_diagnostics/training_validation_001"
def read(p): return json.loads(Path(p).read_text())
def sha(p): return sha256(Path(p).read_bytes()).hexdigest()


def main(folder):
    result = read(folder/"results.json"); fitted = read(folder/"fitted.json"); plan = read(folder/"plan.json")
    levels = fitted["models"]["3"]["levels"]; _, targets = basis(levels[:3]); bg = np.asarray(levels[3])
    values = np.concatenate([np.load(PREV/r["file"], allow_pickle=False)["values"] for r in plan["validation_windows"]])
    opt = []; counts = []; activecorrect = 0; sensible_rank_optimum = 0
    thresholds = np.asarray(fitted["active_thresholds_w"])
    for a in range(0, len(values), 1024):
        v = values[a:a+1024]; costs = np.min((v[:, 0, None, None]-targets.sum(axis=1)[None, :, None]-bg)**2, axis=2)
        index = costs.argmin(axis=1); winning = targets[index]
        near = np.sqrt(costs) <= np.sqrt(costs.min(axis=1))[:, None]+10
        meaningfully_different = np.max(abs(targets[None, :, :]-winning[:, None, :]), axis=2) >= 25
        counts.extend((near&meaningfully_different).sum(axis=1).tolist())
        actual_active = v[:, 1:] > thresholds
        activecorrect += int(np.all((winning > thresholds) == actual_active, axis=1).sum())
        closest = np.column_stack([np.asarray(ls)[nearest(v[:, i+1], ls)] for i, ls in enumerate(levels[:3])])
        equivalent = np.max(abs(targets[None, :, :]-closest[:, None, :]), axis=2) <= 2
        sensible_rank_optimum += int(np.any(equivalent & np.isclose(costs, costs.min(axis=1)[:, None], rtol=0, atol=1e-8), axis=1).sum())
        opt.extend(index.tolist())
    derived = {}
    for name, row in result["candidate_results"].items():
        active = np.asarray(row["active_blocks"]); n = row["blocks"]
        recall = [row["tp"][i]/active[i] if active[i] else None for i in range(3)]
        active_mae = [row["active_absolute_error_sum_w"][i]/active[i] if active[i] else None for i in range(3)]
        inactive_mae = [row["inactive_absolute_error_sum_w"][i]/(n-active[i]) if n > active[i] else None for i in range(3)]
        f1 = [2*row["tp"][i]/(2*row["tp"][i]+row["fp"][i]+row["fn"][i]) if 2*row["tp"][i]+row["fp"][i]+row["fn"][i] else None for i in range(3)]
        days = [r for r in result["candidate_window_results"] if r["id"] == name]
        baseline_days = {r["window"]:r["metrics"]["macro_mae_w"] for r in result["candidate_window_results"] if r["id"] == "sse_k3"}
        derived[name] = {"macro_mae_w": row["macro_mae_w"], "active_recall": recall, "active_mae_w": active_mae,
                         "inactive_mae_w": inactive_mae, "active_f1": f1,
                         "days_lower_mae_than_baseline": sum(r["metrics"]["macro_mae_w"] < baseline_days[r["window"]] for r in days)}
    b = result["candidate_results"]["sse_k3"]
    stricter = [name for name, row in result["candidate_results"].items() if name != "sse_k3" and row["macro_mae_w"] < b["macro_mae_w"]
                and all(row["fn"][i] <= b["fn"][i] for i in (0, 1)) and all(row["fp"][i] <= b["fp"][i] for i in range(3))]
    summaries = result["background_quantization_error_bins"]
    totalexcess = sum(x["baseline_absolute_error_sum_w"]-x["oracle_absolute_error_sum_w"] for x in summaries)
    for x in summaries:
        x["share_of_total_excess_error"] = (x["baseline_absolute_error_sum_w"]-x["oracle_absolute_error_sum_w"])/totalexcess
    ar = result["candidate_window_results"]
    out = {"created_utc": datetime.now(timezone.utc).isoformat(), "results_sha256": sha(folder/"results.json"),
           "code_sha256": sha(__file__), "scope": "Post-hoc descriptive interpretation of frozen development comparisons; no extra fitting or final-test access",
           "derived_candidate_metrics": derived,
           "ambiguity_qualified": {"blocks": len(values), "near_optimal_meaningfully_different_blocks": int((np.array(counts)>0).sum()),
                                   "definition": "At least one alternative within 10 W of optimal absolute aggregate residual and differing by at least 25 W in an appliance",
                                   "near_alternative_count_median": float(np.median(counts)),
                                   "oracle_target_with_2w_equivalence_is_optimum_blocks": sensible_rank_optimum,
                                   "baseline_all_activity_proxies_correct_blocks": activecorrect,
                                   "off_state_caveat": "Dryer centres 0.003 and 0.980 W are distinct categories but physically similar; raw category ambiguity exaggerates meaningful uncertainty"},
           "posthoc_no_worse_false_positive_and_missed_active_candidates": stricter,
           "background_error_bins": summaries,
           "validation_days_with_active_proxies": [sum(r["metrics"]["active_blocks"][i] > 0 for r in ar if r["id"] == "sse_k3") for i in range(3)],
           "appliance_order": ["dryer", "fridge", "vacuum"],
           "limits": ["No new model selected by these post-hoc metrics", "No active vacuum examples; null active recall/MAE, not zero error", "Activity states are train-centroid power proxies, not event annotations"]}
    with (folder/"interpretation.json").open("x") as f: json.dump(out, f, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps({k:v for k,v in out.items() if k not in ("derived_candidate_metrics", "background_error_bins")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--folder", type=Path, required=True)
    main(parser.parse_args().folder.resolve())
