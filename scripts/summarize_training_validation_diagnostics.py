#!/usr/bin/env python3
"""Post-hoc explanation of the frozen cohort; never selects or changes a fit."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np


def read(p): return json.loads(p.read_text())
def sha(p): return sha256(p.read_bytes()).hexdigest()


def summarize(folder):
    output=folder/"diagnostic_breakdown.json"
    if output.exists(): raise ValueError("No overwrite")
    plan=read(folder/"plan.json"); summary=read(folder/"summary.json"); receipt=read(folder/"test_data.json")
    values=np.concatenate([np.load(folder/r["file"],allow_pickle=False)["values"] for r in receipt["windows"]])
    activity=[]
    for i,name in enumerate(("dryr","frdg","vacu")):
        levels=np.array(plan["model"]["levels_w"][i])
        categories=np.argmin(abs(values[:,i+1,None]-levels[None,:]),axis=1)
        activity.append({"appliance":name,"measured_power_quantiles_w":dict(zip(("min","median","p95","max"),np.quantile(values[:,i+1],[0,.5,.95,1]).tolist())),
                         "nearest_centroid_counts":np.bincount(categories,minlength=len(levels)).tolist(),"levels_w":levels.tolist()})
    def result(name,budget): return next(r for r in summary["real_test"] if (r["method"],r["budget"])==(name,budget))
    exact=result("exact","exact"); oracle=result("oracle","oracle"); selected=result("exact_selected_circuits","exact")
    residual=values[:,0]-values[:,1:].sum(axis=1)
    breakdown={"scope":"Post-hoc descriptive checks of the already frozen test cohort; no model/parameter selection or primary-metric change",
        "source_sha256":{n:sha(folder/n) for n in ("plan.json","summary.json","test_data.json")},"code_sha256":sha(Path(__file__)),
        "activity":activity,"mains_minus_target_power_quantiles_w":dict(zip(("min","median","p95","max"),np.quantile(residual,[0,.5,.95,1]).tolist())),
        "exact_minus_representation_oracle_mae_w":exact["macro_mae_w"]-oracle["macro_mae_w"],
        "exact_minus_oracle_per_appliance_w":(np.array(exact["per_appliance_mae_w"])-oracle["per_appliance_mae_w"]).tolist(),
        "mains_exact_minus_privileged_selected_circuit_exact_mae_w":exact["macro_mae_w"]-selected["macro_mae_w"],
        "two_nonconstant_target_descriptive_mae_w":[{"method":r["method"],"budget":r["budget"],"dryer_fridge_mean_w":float(np.mean(r["per_appliance_mae_w"][:2]))} for r in summary["real_test"]],
        "limitations":["No high-state vacuum examples occur in these reserved test days; vacuum activation performance is not evaluated",
                       "The dryer/fridge-only average is post-hoc and does not replace the predeclared three-appliance metric",
                       "Exact-versus-oracle separates representation floor from excess error, not a unique causal mechanism",
                       "Selected-circuit control removes background and changes the observation; its improvement is not a pure additive causal attribution"]}
    with output.open("x") as f: json.dump(breakdown,f,indent=2,allow_nan=False); f.write("\n")
    print(json.dumps({"test_blocks":len(values),"vacuum_state_counts":activity[2]["nearest_centroid_counts"],"exact_minus_oracle_w":breakdown["exact_minus_representation_oracle_mae_w"]}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--folder",type=Path,required=True)
    summarize(parser.parse_args().folder.resolve())
