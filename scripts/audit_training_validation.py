#!/usr/bin/env python3
"""Independent numerical/source audit; does not import the new runner or module."""
import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def require(x, message):
    if not x: raise ValueError(message)
def read(p): return json.loads(Path(p).read_text())
def sha(p): return sha256(Path(p).read_bytes()).hexdigest()
def close(a, b, name, atol=1e-6):
    if not np.allclose(a, b, rtol=1e-9, atol=atol, equal_nan=True): raise ValueError("Mismatch: "+name)


def reference(aggregate, levels):
    states = np.array([x[::-1] for x in product(*[range(len(x)) for x in levels[::-1]])])
    powers = np.array([[levels[i][s] for i,s in enumerate(row)] for row in states])
    y = np.array(aggregate, float)
    energy = np.square(y[:,None]-powers.sum(axis=1)[None,:])
    scale = np.ones(len(y))
    for j,z in enumerate(y):
        linear = sum(abs(v*v-2*z*v) for lev in levels for v in lev)
        quadratic = sum(abs(2*a*b) for i, lev in enumerate(levels) for other in levels[i+1:] for a in lev for b in other)
        scale[j] = max(1.,linear+quadratic)
    return {"levels":levels,"states":states,"powers":powers,"energy":energy,"scale":scale}


def distribution(ref, angles):
    states = ref["states"]; width = len(states)
    v = np.exp(-1j*angles[0]*ref["energy"]/ref["scale"][:,None])/np.sqrt(width)
    c, s = np.cos(angles[1]), -1j*np.sin(angles[1])
    for channel, lev in enumerate(ref["levels"]):
        stride = int(np.prod([len(x) for x in ref["levels"][:channel]]))
        for category in range(len(lev)-1):
            left_indices = np.flatnonzero(states[:,channel] == category)
            right_indices = left_indices+stride
            a,b = v[:,left_indices].copy(),v[:,right_indices].copy()
            v[:,left_indices] = c*a+s*b; v[:,right_indices] = s*a+c*b
    p = abs(v)**2
    return p/p.sum(axis=1)[:,None]


def redraw(p, rng, shots):
    uniform = rng.random((len(p),shots)); cumulative = p.cumsum(axis=1); cumulative[:,-1]=1.
    return np.array([np.searchsorted(c,u,side="right") for c,u in zip(cumulative,uniform)],dtype=np.uint8)


def loss(ref, samples, name):
    per_case = []
    for i, row in enumerate(samples):
        values = ref["energy"][i,np.asarray(row)]
        if name=="cvar50": values = np.sort(values)[:len(values)//2]
        per_case.append(float(np.sum(values)/len(values)/ref["scale"][i]))
    return float(np.mean(per_case))


def metric_arrays(ref, truth, mode, angles=None):
    e = ref["energy"]; states = ref["states"]; powers = ref["powers"]
    truth = np.asarray(truth); dim=truth.shape[1]
    errors = abs(truth[:,None,:]-powers[None,:,:dim])
    nearest = np.column_stack([np.argmin(abs(truth[:,i,None]-np.array(lev)[None,:]),axis=1) for i,lev in enumerate(ref["levels"][:dim])])
    category = states[None,:,:dim] == nearest[:,None,:]
    if mode=="oracle":
        floor = np.column_stack([np.min(abs(truth[:,i,None]-np.array(lev)[None,:]),axis=1) for i,lev in enumerate(ref["levels"][:dim])])
        return {"oracle":{"absolute_error":floor,"category_correct":np.ones_like(floor),"best_cost":None,"hit_optimum":None}}
    if mode in ("exact","low_power"):
        ix = np.argmin(e,axis=1) if mode=="exact" else np.zeros(len(e),dtype=int)
        return {mode:{"absolute_error":errors[np.arange(len(e)),ix],"category_correct":category[np.arange(len(e)),ix].astype(float),
                      "best_cost":e[np.arange(len(e)),ix],"hit_optimum":(e[np.arange(len(e)),ix]==e.min(axis=1)).astype(float)}}
    p = distribution(ref,angles) if mode=="qaoa" else np.ones_like(e)/e.shape[1]
    # Each possible winner excludes all lower-cost states and lower-index ties.
    better_mass = np.zeros_like(p)
    for i in range(e.shape[1]):
        better = (e<e[:,i,None]) | ((e==e[:,i,None]) & (np.arange(e.shape[1])[None,:]<i))
        better_mass[:,i]=(p*better).sum(axis=1)
    popt = np.sum(p*(e==e.min(axis=1)[:,None]),axis=1)
    result = {}
    for k in (16,256):
        win=np.clip(1-better_mass,0,1)**k-np.clip(1-better_mass-p,0,1)**k
        close(win.sum(axis=1),np.ones(len(e)),"Winner mass",1e-8)
        result[str(k)]={"absolute_error":np.sum(win[:,:,None]*errors,axis=1),
                       "category_correct":np.sum(win[:,:,None]*category,axis=1),
                       "best_cost":np.sum(win*e,axis=1),"hit_optimum":1-(1-popt)**k}
    return result


def independent_window(source, start, end):
    """Independent CSV parsing/block assembly; timestamp bisection assumes order."""
    digest=sha256(); groups={}; previous=start-1; total=excluded=nonnumeric=0
    with Path(source).open("rb") as f:
        header=next(csv.reader([f.readline().decode("utf-8-sig").strip()]))
        positions=[header.index(x) for x in ("main","dryr","frdg","vacu")]
        marker=header.index("marker"); first=f.tell(); f.seek(0,2); high=f.tell(); low=first
        while high-low>1000000:
            midpoint=(low+high)//2; f.seek(midpoint); f.readline(); pos=f.tell(); raw=f.readline()
            if not raw: high=midpoint; continue
            timestamp=int(raw.split(b",",1)[0])
            if timestamp<start: low=pos
            else: high=midpoint
        f.seek(low)
        for raw in f:
            if not raw.strip(): continue
            timestamp=int(raw.split(b",",1)[0])
            if timestamp<start: continue
            if timestamp>=end: break
            require(timestamp>previous,"Independent duplicate/order check")
            previous=timestamp; digest.update(raw); total+=1
            index=(timestamp-start)//30
            try:
                row=next(csv.reader([raw.decode("utf-8")]))
                require(len(row)==len(header),"Column count")
                if row[marker] in ("s","+"): excluded+=1; continue
                values=[float(row[i]) for i in positions]
                if not np.isfinite(values).all(): raise ValueError("Nonfinite")
            except (ValueError,IndexError,UnicodeDecodeError,csv.Error): nonnumeric+=1; continue
            groups.setdefault(index,[]).append((timestamp,values))
    times=[]; values=[]
    for i in range((end-start)//30):
        rows=groups.get(i,[])
        if [r[0] for r in rows] == list(range(start+30*i,start+30*(i+1))):
            times.append(start+30*i); values.append(np.array([r[1] for r in rows]).mean(axis=0))
    return np.array(times,dtype=np.int64),np.array(values).reshape(-1,4),digest.hexdigest(),dict(total_rows=total,excluded_marker=excluded,nonnumeric=nonnumeric)


def audit_data(folder, plan):
    arrays={}; checks=blocks=0
    for split in ("train","validation","test"):
        receipt=read(folder/(split+"_data.json"))
        require(receipt["plan_sha256"]==sha(folder/"plan.json"),"Data plan")
        require([x["window"] for x in receipt["windows"]]==plan["windows"][split],"Window coverage")
        for item in receipt["windows"]:
            require(sha(folder/item["file"])==item["sha256"],"Stored data hash")
            w=item["window"]; expected=independent_window(plan["source"],w["start"],w["end"])
            with np.load(folder/item["file"],allow_pickle=False) as saved:
                require(np.array_equal(saved["timestamps"],expected[0]),"Source block timestamps")
                close(saved["values"],expected[1],"Raw submeter/mains means",1e-9)
                arrays[w["id"]]=saved["values"].copy()
            require(expected[2]==item["source_window_sha256"],"Raw window checksum")
            require(len(expected[0])==item["blocks"]==item["quality"]["accepted_blocks"],"Block count")
            for k,v in expected[3].items(): require(item["quality"][k]==v,"Source quality "+k)
            blocks+=len(expected[0]); checks+=1
        require(sum(x["blocks"] for x in receipt["windows"])==receipt["total_blocks"],"Partition block count")
        print(f"Independently checked {split} extraction",flush=True)
    return arrays,checks,blocks


def audit_fit(record, folder, ref, seeds, bound):
    fit=record["fit"]; require(fit["seeds"]==seeds and fit["gamma_max"]==bound,"Seed/bound binding")
    require(fit["evaluations"]==1024 and len(fit["trace"])==1024 and fit["training_cases"]==len(ref["energy"]),"Fit coverage")
    path=folder/"fits"/record["samples_file"]; require(sha(path)==record["samples_sha256"],"Shot archive hash")
    with np.load(path,allow_pickle=False) as f: raw=f["search"]; recheck=f["recheck"]
    require(raw.shape==(1024,len(ref["energy"]),256) and recheck.shape==(5,len(ref["energy"]),4096),"Raw-shot dimensions")
    max_error=0.
    for j, restart in enumerate(fit["restarts"]):
        require(restart["seed"]==seeds[j] and restart["offset"]==512*j and restart["evaluations"]==512,"Restart metadata")
        initial=np.vstack([np.zeros(2),np.random.default_rng(seeds[j]).uniform([0,0],[bound,np.pi],size=(31,2))])
        require(np.array_equal(initial,restart["initial_candidates"]),"Paired candidate pool")
        require(np.array_equal(initial,[r["angles"] for r in fit["trace"][512*j:512*j+32]]),"Initial trace")
        require(sum(x["calls"]+x["padding_calls"] for x in restart["refinements"])+32==512,"Powell accounting")
        rng=np.random.default_rng(seeds[j]+1000000)
        for i in range(512*j,512*(j+1)):
            row=fit["trace"][i]; a=np.array(row["angles"])
            require(np.all(a>=0) and np.all(a<=[bound,np.pi]),"Fit angle domain")
            require(np.array_equal(redraw(distribution(ref,a),rng,256),raw[i]),"Measurement replay")
            error=abs(loss(ref,raw[i],fit["loss"])-row["loss"]); require(error<1e-10,"Training-loss replay")
            max_error=max(max_error,error)
    ranked=np.argsort([r["loss"] for r in fit["trace"]],kind="stable")
    unique=[]; seen=set()
    for i in ranked:
        angle=tuple(fit["trace"][i]["angles"])
        if angle not in seen: unique.append(int(i)); seen.add(angle)
        if len(unique)==5: break
    require(fit["shortlist_indices"]==unique and fit["precheck_training_index"]==int(ranked[0]),"Observed-loss shortlist")
    require(fit["precheck_angles"]==fit["trace"][ranked[0]]["angles"],"Before-recheck angles")
    rng=np.random.default_rng(seeds[0]+2000000)
    for j,i in enumerate(unique):
        row=fit["rechecks"][j]
        require(row["training_index"]==i and row["angles"]==fit["trace"][i]["angles"],"Recheck candidate")
        require(np.array_equal(redraw(distribution(ref,row["angles"]),rng,4096),recheck[j]),"Independent-recheck sample replay")
        error=abs(loss(ref,recheck[j],fit["loss"])-row["loss"]); require(error<1e-10,"Recheck loss")
        max_error=max(error,max_error)
    selected=int(np.argmin([r["loss"] for r in fit["rechecks"]]))
    require(fit["selected_recheck_index"]==selected and fit["angles"]==fit["rechecks"][selected]["angles"],"Rechecked final selection")
    require(fit["training_shots"]==raw.size and fit["recheck_shots"]==recheck.size
            and fit["total_measurements"]==raw.size+recheck.size,"Complete measurement budget")
    return raw.size,recheck.size,max_error


def check_window_metrics(values, levels, row, mode, angles=None, selected=False):
    require(row["score"]["blocks"]==len(values),"Day blocks")
    sums={}
    for start in range(0,len(values),256):
        block=values[start:start+256]
        ref=reference(block[:,1:].sum(axis=1) if selected else block[:,0],levels[:3] if selected else levels)
        result=metric_arrays(ref,block[:,1:],mode,angles)
        for k,m in result.items():
            dest=sums.setdefault(k,{"absolute_error_sum":np.zeros(3),"category_correct_sum":np.zeros(3),"best_cost_sum":0.,"hit_optimum_sum":0.})
            for key,saved in (("absolute_error","absolute_error_sum"),("category_correct","category_correct_sum"),("best_cost","best_cost_sum"),("hit_optimum","hit_optimum_sum")):
                if m[key] is not None: dest[saved]+=np.sum(m[key],axis=0)
    require(set(sums)==set(row["score"]["budgets"]),"Reported budgets")
    for k,m in sums.items():
        saved=row["score"]["budgets"][k]
        require(saved["blocks"]==len(values),"Budget blocks")
        for key,v in m.items():
            if mode=="oracle" and key in ("best_cost_sum","hit_optimum_sum"): require(saved[key] is None,"No fabricated oracle objective")
            else: close(v,saved[key],"Day "+key,1e-5)


def audit(folder):
    output=folder/"independent_audit.json"; require(not output.exists(),"Audit receipt exists")
    plan=read(folder/"plan.json"); binding=sha(folder/"plan.json")
    summary=read(folder/"summary.json"); require(summary["plan_sha256"]==binding,"Summary plan")
    for group in ("source_sha256","input_sha256"):
        for path,value in plan[group].items(): require(sha(ROOT/path)==value,"Frozen source/input "+path)
    for path,value in summary["source_results_sha256"].items(): require(sha(folder/path)==value,"Result binding")
    for split in ("validation","test"):
        windows=plan["windows"][split]
        for w in windows:
            require(w["end"]-w["start"]==86400,"Full day interval")
            require(all(w["end"]<=a-86400 or w["start"]>=b+86400 for a,b in plan["exposed_ranges"]),"Historical exposure guard")
        require(all(a["end"]<=b["start"] for a,b in zip(windows,windows[1:])),"New window overlap")
    require(plan["windows"]["validation"][-1]["end"]<plan["windows"]["test"][0]["start"],"Chronological validation/test split")
    selection=read(folder/"selection.json"); data=read(folder/"test_data.json")
    require(selection["validation_sha256"]==sha(folder/"validation.json") and data["selection_sha256_before_read"]==sha(folder/"selection.json"),"Test selection gate")
    require(datetime.fromisoformat(selection["frozen_utc_before_test_read"])<datetime.fromisoformat(data["extracted_utc"]),"Test after selection")
    arrays,source_checks,blocks=audit_data(folder,plan)
    train=np.concatenate([arrays[x["id"]] for x in plan["windows"]["train"]])
    real_ref=reference(train[:,0],plan["model"]["levels_w"])
    synthetic_ref=reference([680.],[[0.,400.,900.],[0.,120.,280.]])
    fits={}; search_shots=recheck_shots=0; max_error=0.
    expected={f"synthetic_{bound}_{loss}_{i:02}": (synthetic_ref,seeds,maximum)
              for bound,maximum in plan["synthetic_gamma_bounds"].items() for i,seeds in enumerate(plan["synthetic_seed_pairs"]) for loss in plan["losses"]}
    expected.update({f"r1hz_{loss}_{i:02}":(real_ref,seeds,plan["real_gamma_max"]) for i,seeds in enumerate(plan["real_seed_pairs"]) for loss in plan["losses"]})
    require({p.stem for p in (folder/"fits").glob("*.json")}==set(expected),"100-fit coverage")
    for i,(identifier,(ref,seeds,bound)) in enumerate(expected.items()):
        record=read(folder/"fits"/(identifier+".json")); require(record["plan_sha256"]==binding and record["id"]==identifier,"Fit identity")
        a,b,c=audit_fit(record,folder,ref,seeds,bound); search_shots+=a; recheck_shots+=b; max_error=max(max_error,c)
        fits[identifier]=record["fit"]
        if (i+1)%10==0: print(f"Replayed {i+1}/100 complete training and recheck traces",flush=True)
    synth=read(folder/"synthetic.json"); powers=synthetic_ref["powers"]; ref=reference(powers.sum(axis=1),synthetic_ref["levels"])
    require(synth["training_case_index"]==7 and np.array_equal(synth["cases"],powers),"Synthetic case order")
    require(len(synth["rows"])==160,"Synthetic result coverage")
    for row in synth["rows"]:
        fit=fits[row["id"]]; angle=fit["angles" if row["phase"]=="after_recheck" else "precheck_angles"]
        require(row["angles"]==angle,"Synthetic evaluation angles")
        metrics=metric_arrays(ref,powers,"qaoa",angle)
        for k,m in metrics.items():
            for key,value in m.items(): close(value,row["metrics"][k][key],"Synthetic metric")
    for k,m in metric_arrays(ref,powers,"uniform").items():
        for key,value in m.items(): close(value,synth["uniform"][k][key],"Uniform metric")
    validation=read(folder/"validation.json"); training=read(folder/"real_training.json")
    require(len(validation["rows"])==200 and len(training["fits"])==20,"Validation coverage")
    for row in validation["rows"]:
        check_window_metrics(arrays[row["window_id"]],plan["model"]["levels_w"],row,"qaoa",fits[row["fit_id"]]["angles"])
    for row in validation["summaries"]:
        observed=[r["score"]["budgets"]["16"] for r in validation["rows"] if r["fit_id"]==row["id"] and r["score"]["blocks"]]
        n=sum(x["blocks"] for x in observed); error=np.sum([x["absolute_error_sum"] for x in observed],axis=0)/n
        close(error,row["per_appliance_mae_w"],"Validation aggregate"); close(error.mean(),row["macro_mae_w"],"Validation criterion")
        require(row["angles"]==fits[row["id"]]["angles"],"Validation fit angles")
    for name in plan["losses"]:
        chosen=min([s for s in validation["summaries"] if s["loss"]==name],key=lambda s:(s["macro_mae_w"],s["trial"]))
        require(chosen==selection["selected_per_loss"][name],"Validation-only model selection")
    require(selection["validation_winner"]==min(plan["losses"],key=lambda n:selection["selected_per_loss"][n]["macro_mae_w"]),"Validation winner")
    print("All validation day metrics independently checked",flush=True)
    testing=read(folder/"test.json"); require(len(testing["rows"])==140,"Test method/day coverage")
    for row in testing["rows"]:
        name=row["method"]; mode="qaoa" if name in plan["losses"] else "exact" if name=="exact_selected_circuits" else name
        angles=selection["selected_per_loss"][name]["angles"] if mode=="qaoa" else None
        check_window_metrics(arrays[row["window_id"]],plan["model"]["levels_w"],row,mode,angles,name=="exact_selected_circuits")
    for row in summary["real_test"]:
        values=[r["score"]["budgets"][row["budget"]] for r in testing["rows"] if r["method"]==row["method"] and r["score"]["blocks"]]
        count=sum(x["blocks"] for x in values); errors=np.sum([x["absolute_error_sum"] for x in values],axis=0)/count
        require(row["blocks"]==count,"Pooled test coverage"); close(row["per_appliance_mae_w"],errors,"Pooled measured MAE")
        close(row["macro_mae_w"],errors.mean(),"Pooled macro MAE")
    for row in summary["paired_contrasts"]:
        left=[]; right=[]; counts=[]
        for w in plan["windows"]["test"]:
            a=next(x for x in testing["rows"] if x["window_id"]==w["id"] and x["method"]==row["method"])
            b=next(x for x in testing["rows"] if x["window_id"]==w["id"] and x["method"]==row["control"])
            if not a["score"]["blocks"]: continue
            av=a["score"]["budgets"][row["budget"]]; bv=b["score"]["budgets"]["oracle" if row["control"]=="oracle" else row["budget"]]
            left.append(np.mean(av["absolute_error_sum"])); right.append(np.mean(bv["absolute_error_sum"])); counts.append(av["blocks"])
        delta=np.array(left)-right; counts=np.array(counts); rng=np.random.default_rng(plan["bootstrap"]["seed"])
        boot=[]
        for _ in range(2000):
            ids=rng.integers(0,len(delta),len(delta)); boot.append(delta[ids].sum()/counts[ids].sum())
        close(row["difference_w"],delta.sum()/counts.sum(),"Paired contrast")
        close(row["descriptive_day_bootstrap_ci95"],np.percentile(boot,[2.5,97.5]),"Paired bootstrap")
    for row in summary["synthetic"]:
        values=[r for r in synth["rows"] if (r["bound"],r["loss"],r["phase"])==(row["bound"],row["loss"],row["phase"])]
        k=str(row["shots"])
        for key,x in (("training_mae_w",[np.mean(v["metrics"][k]["absolute_error"][7]) for v in values]),
                      ("transfer_mae_w",[np.mean([a for i,a in enumerate(v["metrics"][k]["absolute_error"]) if i!=7]) for v in values]),
                      ("training_best_cost_w2",[v["metrics"][k]["best_cost"][7] for v in values])):
            close(row[key]["values"],x,"Synthetic trial aggregation"); close(row[key]["mean"],np.mean(x),"Synthetic mean")
    # Explicit physical statevectors check every final synthetic fit and all real
    # fitted angles on three prespecified training cases. No device model is used.
    from qiskit.quantum_info import Statevector
    from quantum_nilm.categorical_qaoa import prepare_categorical_problem
    from quantum_nilm.six_qubit_diagnostic import logical_circuit, feasible_indices
    circuit_checks=0; max_probability_error=0.
    for identifier,fit in fits.items():
        checks=[(680.,synthetic_ref["levels"])] if identifier.startswith("synthetic") else [(float(train[j,0]),plan["model"]["levels_w"]) for j in (0,len(train)//2,len(train)-1)]
        for y,lev in checks:
            p=prepare_categorical_problem([y],lev,[0]*len(lev))
            circuit=logical_circuit(p,[fit["angles"][0]],[fit["angles"][1]]).remove_final_measurements(inplace=False)
            actual=Statevector.from_instruction(circuit).probabilities()
            expected=distribution(reference([y],lev),fit["angles"])[0]
            error=float(np.max(abs(actual[feasible_indices(p)]-expected)))
            require(error<1e-9 and abs(actual[feasible_indices(p)].sum()-1)<1e-9,"Physical statevector equivalence")
            max_probability_error=max(max_probability_error,error); circuit_checks+=1
    require(summary["training_evaluations"]==102400 and summary["training_measurements"]==search_shots
            and summary["independent_recheck_measurements"]==recheck_shots
            and summary["total_classically_sampled_measurements"]==search_shots+recheck_shots,"Campaign budget")
    receipt={"status":"passed","created_utc":datetime.now(timezone.utc).isoformat(),"plan_sha256":binding,
             "summary_sha256":sha(folder/"summary.json"),"audit_source_sha256":sha(Path(__file__)),
             "source_windows_independently_reaggregated":source_checks,"source_valid_blocks_checked":blocks,
             "training_calls_recomputed":102400,"shortlist_rechecks_recomputed":500,
             "classically_sampled_measurements_reproduced":search_shots+recheck_shots,
             "synthetic_case_metric_sets":160*9*2+9*2,"validation_method_days":200,"test_method_days":140,
             "physical_statevector_checks":circuit_checks,"max_training_loss_discrepancy":max_error,
             "max_physical_probability_discrepancy":max_probability_error,
             "limitations":["No global source ordering or no-ever-inspected claim","Optimizer trajectories checked, not independently rerun",
                            "Static within-home validation, not temporal/full IBM performance or hardware evidence"]}
    with output.open("x") as f: json.dump(receipt,f,indent=2,allow_nan=False); f.write("\n")
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--folder",type=Path,required=True)
    audit(parser.parse_args().folder.resolve())
