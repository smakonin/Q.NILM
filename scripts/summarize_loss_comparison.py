#!/usr/bin/env python3
"""Aggregate every paired trial without choosing winners or retuning parameters."""
import argparse
from collections import defaultdict
from hashlib import sha256
import gzip
import json
from pathlib import Path
import numpy as np


def sha(path): return sha256(path.read_bytes()).hexdigest()


def stats(values):
    a = np.array(values, float)
    return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()), "values": a.tolist()}


def summarize(folder):
    path = folder/"comparison_tables.json"
    if path.exists(): raise ValueError("Never overwrite tables")
    result = json.loads((folder/"summary.json").read_text())
    audit = json.loads((folder/"independent_audit.json").read_text())
    assert audit["status"]=="passed" and audit["summary_sha256"]==sha(folder/"summary.json")
    grouped = defaultdict(list)
    for row in result["ideal_rows"]:
        if row["role"]=="training": grouped[(row["config"],row["loss"])].append(row)
    primary = []
    for (config, loss), rows in sorted(grouped.items()):
        rows.sort(key=lambda r:r["trial"])
        primary.append({"config": config, "loss": loss, "trials": len(rows),
            "mean_cost_w2": stats([r["metrics"]["conditional_mean_cost_w2"] for r in rows]),
            "raw_optimum_percent": stats([100*r["metrics"]["raw_optimum_probability"] for r in rows]),
            "best16_cost_w2": stats([r["metrics"]["budgets"]["16"]["conditional_best_cost_w2"] for r in rows]),
            "mae16_w": stats([r["metrics"]["budgets"]["16"]["conditional_selected_macro_mae_w"] for r in rows]),
            "hit16_percent": stats([100*r["metrics"]["budgets"]["16"]["optimum_hit_probability"] for r in rows]),
            "mae256_w": stats([r["metrics"]["budgets"]["256"]["conditional_selected_macro_mae_w"] for r in rows])})
    transfer_group = defaultdict(list)
    for r in result["ideal_rows"]:
        if r["role"]=="synthetic_transfer": transfer_group[(r["config"],r["loss"],r["trial"])].append(r)
    by_loss = defaultdict(list)
    for (config,loss,trial), rows in transfer_group.items():
        assert len(rows)==8
        by_loss[(config,loss)].append({"trial":trial,
            "mae16_w": float(np.mean([r["metrics"]["budgets"]["16"]["conditional_selected_macro_mae_w"] for r in rows])),
            "mae256_w": float(np.mean([r["metrics"]["budgets"]["256"]["conditional_selected_macro_mae_w"] for r in rows])),
            "hit16_percent": float(np.mean([100*r["metrics"]["budgets"]["16"]["optimum_hit_probability"] for r in rows]))})
    transfer = []
    for (config,loss), rows in sorted(by_loss.items()):
        rows.sort(key=lambda r:r["trial"])
        transfer.append({"config":config,"loss":loss,"trials":len(rows),
                         **{key:stats([r[key] for r in rows]) for key in ("mae16_w","mae256_w","hit16_percent")}})
    noise_group = defaultdict(list)
    for row in result["noise_rows"]: noise_group[(row["placement"],row["model"]["name"],row["loss"])].append(row)
    noise = []
    for (placement,model,loss), rows in sorted(noise_group.items()):
        rows.sort(key=lambda r:r["trial"])
        noise.append({"placement":placement,"model":model,"loss":loss,"trials":len(rows),
            "raw_valid_percent":stats([100*r["metrics"]["raw_valid_probability"] for r in rows]),
            "raw_optimum_percent":stats([100*r["metrics"]["raw_optimum_probability"] for r in rows]),
            "hit16_percent":stats([100*r["metrics"]["budgets"]["16"]["optimum_hit_probability"] for r in rows]),
            "no_valid16_probability":stats([r["metrics"]["budgets"]["16"]["no_valid_probability"] for r in rows]),
            "conditional_mae16_w":stats([r["metrics"]["budgets"]["16"]["conditional_selected_macro_mae_w"] for r in rows]),
            "conditional_cost16_w2":stats([r["metrics"]["budgets"]["16"]["conditional_best_cost_w2"] for r in rows])})
    paired = []
    for config in ("exact8","sampled8","exact16"):
        baseline = next(r for r in primary if r["config"]==config and r["loss"]=="mean")
        for arm in [r for r in primary if r["config"]==config and r["loss"]!="mean"]:
            diff = np.array(arm["best16_cost_w2"]["values"])-baseline["best16_cost_w2"]["values"]
            paired.append({"config":config,"loss":arm["loss"],"cost16_difference_w2":stats(diff),
                           "lower_cost_trials":int(sum(diff < -1e-8)), "higher_cost_trials":int(sum(diff > 1e-8))})
    zeros = defaultdict(int); angles = defaultdict(list)
    for f in result["fits"]:
        with gzip.open(folder/f["file"],"rt") as handle: fit = json.load(handle)["fit"]
        if fit["selected_training_loss"]==0: zeros[(f["config"],f["loss"])] += 1
        angles[(f["config"],f["loss"])].append(fit["angles"][0])
    output = {"summary_sha256":sha(folder/"summary.json"),"audit_sha256":sha(folder/"independent_audit.json"),
        "summary_code_sha256":sha(Path(__file__)), "aggregation":"Equal-weight means of five paired trial expectations; min/max are seed variation, not confidence intervals. Transfer averages eight cases within each trial first.",
        "primary":primary,"transfer":transfer,"noise":noise,"paired_to_mean":paired,
        "selected_zero_training_loss":[{"config":c,"loss":l,"trials":n} for (c,l),n in sorted(zeros.items())],
        "gamma_ranges":[{"config":c,"loss":l,**stats(v)} for (c,l),v in sorted(angles.items())]}
    with path.open("x") as handle:
        json.dump(output,handle,indent=2,allow_nan=False); handle.write("\n")
    print(json.dumps({"primary_groups":len(primary),"noise_groups":len(noise),"transfer_groups":len(transfer)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,required=True)
    summarize(parser.parse_args().folder.resolve())
