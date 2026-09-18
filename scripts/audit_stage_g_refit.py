#!/usr/bin/env python3
"""Independent REFIT result/trace/statistics audit; no new experiment execution.

The frozen REFIT source reader is reused as input extraction only. Scoring,
objective certificates, sample-only selection and uncertainty are independently
recomputed. QAOA draw replay is preselected to the earliest nonempty test day
per home; uniform draws and recorded-sample correctness are checked everywhere.
Full audit requires the complete summary. Optional prechecks inspect only
already-saved window artifacts after all models are frozen and test execution
has begun; hash-bound receipts avoid repeating those checks at completion.
"""
from __future__ import annotations

import argparse
import gzip
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
from quantum_nilm.refit import read_block_window
from audit_stage_b import close, require, digest, direct_ideal_probabilities
from audit_stage_d_components import independently_score, independently_pool, same as compare_value

CHANNELS = ("washing", "cooling", "dishwasher")
OLD_CHANNELS = ("dryr", "frdg", "vacu")
METHODS = ("qaoa_ideal", "uniform", "exact_chunk", "exact_full", "training_mean")
SEEDS = (1907, 2907, 3907)
ARMS = tuple((method, seed) for method in METHODS for seed in
             (SEEDS if method in ("qaoa_ideal", "uniform") else (None,)))


def same(actual, expected, label, atol=1e-7, rtol=1e-11):
    """Strict recursive comparison, including lists of per-seed dictionaries."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), label+": field coverage")
        for key in expected:
            same(actual[key], expected[key], f"{label}/{key}", atol, rtol)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label+": list coverage")
        for index,(a,b) in enumerate(zip(actual, expected)):
            same(a,b,f"{label}/{index}",atol,rtol)
    else:
        compare_value(actual,expected,label,atol,rtol)


def score(reference, predicted, timestamps, thresholds):
    result = independently_score(reference, predicted, timestamps, thresholds)
    return {new:result[old] for new,old in zip(CHANNELS, OLD_CHANNELS)}


def pool(rows):
    mapped = [dict(appliances={old:r["appliances"][new] for old,new in zip(OLD_CHANNELS, CHANNELS)}) for r in rows]
    result = independently_pool(mapped)
    return {new:result[old] for new,old in zip(CHANNELS, OLD_CHANNELS)}


def categorical_basis(counts, intervals):
    radices = tuple(counts)*intervals
    values = [x[::-1] for x in itertools.product(*(range(n) for n in radices[::-1]))]
    return np.array(values, dtype=int).reshape(-1, intervals, len(counts))


def chunk_objective(chunk, model, previous, basis):
    levels = [np.asarray(x) for x in model["levels"]]
    penalties = np.asarray(model["penalties"])
    powers = sum(values[basis[:, :, c]] for c,values in enumerate(levels))
    aggregate, weights = np.asarray(chunk["aggregate"]), np.asarray(chunk["weights"])
    energy = np.sum(weights*(aggregate-powers)**2, axis=1)
    if len(weights) > 1:
        energy += np.sum(penalties*(basis[:, 1:] != basis[:, :-1]), axis=(1,2))
    if previous is not None:
        energy += np.sum(penalties*(basis[:, 0] != previous), axis=1)
    return energy


def phase_scale(chunk, model, previous):
    """Independent coefficient-only scale including the inferred prior state."""
    levels = [np.asarray(x) for x in model["levels"]]
    total = 0.
    for t,(measurement,weight) in enumerate(zip(chunk["aggregate"], chunk["weights"])):
        for channel,values in enumerate(levels):
            linear = weight*(values**2-2*measurement*values)
            if t == 0 and previous is not None:
                linear = linear.copy()
                linear[previous[channel]] -= model["penalties"][channel]
            total += np.abs(linear).sum()
            for other in range(channel+1,len(levels)):
                total += np.abs(2*weight*np.outer(values,levels[other])).sum()
    if len(chunk["weights"]) > 1:
        total += sum(len(values)*p for values,p in zip(levels,model["penalties"]))
    return max(1., float(total))


def check_chunks(chunks, mains, timestamps, threshold=None):
    """Verify compression coverage and actual interval means, without references."""
    cursor = 0
    previous_run = None
    source_edges = [0]+(np.where(np.diff(timestamps) != 30)[0]+1).tolist()+[len(timestamps)]
    starts, runs = [], []
    for number,chunk in enumerate(chunks):
        require(chunk["chunk"] == number and chunk["block_start"] == cursor, "Chunk ordering/coverage")
        require(len(chunk["weights"]) in (1,2) and len(chunk["aggregate"]) == len(chunk["weights"]), "Chunk interval count")
        if chunk["reset"]:
            require(previous_run is None or previous_run != chunk["run"], "Repeated reset inside run")
            starts.append(cursor)
            runs.append([])
        else:
            require(previous_run == chunk["run"], "Changed run without reset")
        position = cursor
        for weight,mean in zip(chunk["weights"],chunk["aggregate"]):
            require(isinstance(weight,int) and weight > 0, "Nonpositive/inexact duration")
            close(mean, np.mean(mains[position:position+weight]), "Raw aggregate segment mean", atol=1e-8)
            runs[-1].append((position,position+weight,float(mean),weight))
            position += weight
        require(position == chunk["block_stop"], "Duration expansion mismatch")
        require(chunk["start_unix"] == timestamps[cursor], "Chunk timestamp mismatch")
        cursor,previous_run = position,chunk["run"]
    require(cursor == len(mains) and starts == source_edges[:-1], "Incomplete/gap-crossing compression")
    if threshold is not None:
        for run,(lo,hi) in zip(runs,zip(source_edges[:-1],source_edges[1:])):
            differences = np.abs(np.diff(mains[lo:hi]))
            candidates = np.where(differences >= threshold)[0]
            groups = np.split(candidates,np.where(np.diff(candidates) > 1)[0]+1)
            events = [int(group[np.argmax(differences[group])])+lo+1 for group in groups if len(group)]
            boundaries = [lo]+events+[hi]
            require([(s,e) for s,e,_,_ in run] == list(zip(boundaries[:-1],boundaries[1:])),
                    "Compression boundaries differ from frozen aggregate-only threshold")
    return runs


def check_chunk_trace(trace, chunks, prediction, model, angles, method, seed, house, day,
                      replay_qaoa=False):
    levels = [np.asarray(x) for x in model["levels"]]
    counts = tuple(map(len,levels))
    basis = {k:categorical_basis(counts,k) for k in (1,2)}
    detail = trace["detail"]
    effective = (seed if seed is not None else SEEDS[0])+house*1_000_000+day*10_000
    require(detail["effective_seed"] == effective, "Effective RNG seed mismatch")
    rng = np.random.default_rng(effective)
    previous = None
    max_energy_error = 0.
    draws_checked = 0
    require(len(detail["records"]) == len(chunks), "Chunk trace count mismatch")
    for chunk,row in zip(chunks,detail["records"]):
        if chunk["reset"]:
            previous = None
        require(row["chunk"] == chunk["chunk"] and row["previous_states"] ==
                (None if previous is None else previous.tolist()), "Method does not carry its own state/reset")
        states = basis[len(chunk["weights"])]
        energies = chunk_objective(chunk,model,previous,states)
        chosen = row["chosen_index"]
        require(isinstance(chosen,int) and 0 <= chosen < len(states), "Chosen categorical index out of range")
        require(row["states"] == states[chosen].tolist(), "Chosen index/state mismatch")
        raw_counts = row["raw_feasible_counts"]
        ids = [x[0] for x in raw_counts]
        require(ids == sorted(set(ids)) and all(0 <= x < len(states) for x in ids), "Invalid sampled feasible indices")
        require(all(isinstance(n,int) and n > 0 for _,n in raw_counts), "Invalid sample counts")
        require(sum(n for _,n in raw_counts) == (1 if method == "exact_chunk" else 256), "Shot count differs from protocol")
        require(chosen in ids, "Inserted unsampled solution")
        observed_minimum = float(energies[ids].min())
        exact_minimum = float(energies.min())
        close(energies[chosen], observed_minimum, "Not a sampled minimum", atol=1e-5, rtol=1e-11)
        if method == "exact_chunk":
            close(energies[chosen], exact_minimum, "Matched exact certificate", atol=1e-5, rtol=1e-11)
            require(raw_counts == [[chosen,1]] and row["distribution_expected_energy"] is None, "Exact trace scope")
        close(row["best_observed_energy"], energies[chosen], "Best direct objective", atol=1e-5, rtol=1e-11)
        close(row["conditional_exact_energy"], exact_minimum, "Conditional exact objective", atol=1e-5, rtol=1e-11)
        max_energy_error = max(max_energy_error,abs(row["conditional_exact_energy"]-exact_minimum))
        require(row["logical_qubits"] == sum(counts)*len(chunk["weights"]) and row["feasible_states"] == len(states), "Circuit size mismatch")
        if method == "uniform" or (method == "qaoa_ideal" and replay_qaoa):
            if method == "uniform":
                probabilities = np.ones(len(states))/len(states)
            else:
                probabilities = direct_ideal_probabilities(states,energies,phase_scale(chunk,model,previous),
                    counts,angles["gammas"],angles["betas"])
            close(row["distribution_expected_energy"], probabilities@energies, "Independent circuit expectation", atol=1e-5, rtol=1e-11)
            sampled = rng.choice(len(states),256,p=probabilities)
            unique,frequencies = np.unique(sampled,return_counts=True)
            require(raw_counts == np.column_stack((unique,frequencies)).tolist(), "Independent RNG replay differs")
            draws_checked += 256
        powers = np.column_stack([values[states[chosen,:,i]] for i,values in enumerate(levels)])
        expanded = np.repeat(powers,chunk["weights"],axis=0)
        close(prediction[chunk["block_start"]:chunk["block_stop"]],expanded,"Saved chunk prediction expansion",atol=0,rtol=0)
        previous = states[chosen,-1].copy()
    return max_energy_error, draws_checked


def check_full_dp(trace, runs, mains, prediction, model):
    levels = [np.asarray(x) for x in model["levels"]]
    penalties = np.asarray(model["penalties"])
    joint = np.array(list(itertools.product(*(range(len(x)) for x in levels))))
    joint_powers = sum(x[joint[:,i]] for i,x in enumerate(levels))
    transition = np.sum((joint[:,None,:] != joint[None,:,:])*penalties,axis=2)
    segment_total = residual = 0.
    max_gap = 0.
    for run in runs:
        previous = None
        actual_objective = 0.
        costs = None
        for start,stop,mean,weight in run:
            powers = prediction[start]
            close(prediction[start:stop],np.tile(powers,(stop-start,1)),"Full-DP interval constancy",atol=0,rtol=0)
            states = np.array([int(np.argmin(np.abs(x-powers[i]))) for i,x in enumerate(levels)])
            close(powers,[x[states[i]] for i,x in enumerate(levels)],"Full-DP centroid membership",atol=1e-8)
            actual_objective += weight*(mean-powers.sum())**2
            if previous is not None:
                actual_objective += float(penalties@(states != previous))
            previous = states
            reconstruction = weight*(mean-joint_powers)**2
            costs = reconstruction if costs is None else np.min(costs[:,None]+transition,axis=0)+reconstruction
            residual += float(np.sum((mains[start:stop]-mean)**2))
        minimum = float(costs.min())
        max_gap = max(max_gap,abs(actual_objective-minimum))
        close(actual_objective,minimum,"Independent full temporal DP certificate",atol=1e-3,rtol=1e-11)
        segment_total += actual_objective
    detail = trace["detail"]
    close(detail["objective_energy"],segment_total,"Full-DP segment objective",atol=1e-3,rtol=1e-11)
    close(detail["within_segment_residual_constant"],residual,"Compression residual",atol=1e-3,rtol=1e-11)
    close(detail["full_block_objective_energy"],segment_total+residual,"Full-DP raw-block objective",atol=1e-3,rtol=1e-11)
    require(detail["segments"] == sum(len(x) for x in runs) and detail["blocks"] == len(mains), "Full-DP coverage metadata")
    return max_gap


def audit_statistics(records, protocol, summary):
    homes = sorted({r["house"] for r in records})
    require(summary["status"] == ("complete" if homes else "no_evaluable_homes"), "Summary completion status")
    per_home,day_errors = {},{}
    for house in homes:
        per_home[str(house)] = {}
        for method in METHODS:
            rows = [r for r in records if r["house"] == house and r["model"] == method]
            seed_metrics = []
            for seed in (SEEDS if method in ("qaoa_ideal","uniform") else (None,)):
                subset = [r for r in rows if r["seed"] == seed]
                blocks = sum(r["blocks"] for r in subset)
                seed_metrics.append(dict(seed=seed,appliances=pool(subset),
                    aggregate_mae_w=sum(r["raw_aggregate_mae_w"]*r["blocks"] for r in subset)/blocks,
                    solver_wall_time_s=sum(r["solver_wall_time_s"] for r in subset)))
            days = {}
            for name in sorted({r["window_id"] for r in rows}):
                group = [r for r in rows if r["window_id"] == name]
                errors = [sum(r["appliances"][c]["absolute_error_sum_w"] for c in CHANNELS) for r in group]
                days[name] = (float(np.mean(errors)),3*group[0]["blocks"])
            error = sum(x[0] for x in days.values())
            denom = sum(x[1] for x in days.values())
            result = dict(macro_mae_w=error/denom,absolute_error_sum_w=error,blocks=denom//3,windows=len(days),
                seed_metrics=seed_metrics,aggregate_mae_w=float(np.mean([x["aggregate_mae_w"] for x in seed_metrics])),
                appliance_mae_w={c:float(np.mean([x["appliances"][c]["mae_w"] for x in seed_metrics])) for c in CHANNELS})
            same(summary["per_home"][str(house)][method],result,f"Home {house}/{method}: all pooled metrics")
            per_home[str(house)][method] = result
            day_errors[house,method] = days
    require(set(summary["per_home"]) == set(per_home),"Per-home summary coverage")
    nonevaluable = sorted({h["house"] for h in protocol["homes"]}-set(homes))
    require(summary["nonevaluable_homes"] == nonevaluable,"Nonevaluable home denominator")
    for method in METHODS:
        rows = [per_home[str(h)][method] for h in homes]
        expected = dict(equal_home_macro_mae_w=float(np.mean([x["macro_mae_w"] for x in rows])) if rows else None,
            block_pooled_macro_mae_w=sum(x["absolute_error_sum_w"] for x in rows)/(3*sum(x["blocks"] for x in rows)) if rows else None,
            evaluable_homes=len(rows),blocks=sum(x["blocks"] for x in rows),windows=sum(x["windows"] for x in rows))
        same(summary["overall"][method],expected,f"{method}: equal-home versus block-pooled estimands")
    require(len(summary["paired_day_comparisons"]) == len(homes)*4,"Paired day contrast coverage")
    seen = set()
    for item in summary["paired_day_comparisons"]:
        house,left,right = item["house"],item["left"],item["right"]
        require(left == "qaoa_ideal" and right in METHODS[1:] and (house,right) not in seen,"Duplicate/invalid day contrast")
        seen.add((house,right))
        a,b = day_errors[house,left],day_errors[house,right]
        require(a.keys() == b.keys(),"Methods use different days")
        names = sorted(a)
        require(all(a[k][1] == b[k][1] for k in names),"Paired day block denominators differ")
        difference = np.array([a[k][0]-b[k][0] for k in names])
        denominator = np.array([a[k][1] for k in names])
        seed = protocol["day_bootstrap"]["seed_plus_house"]+house
        reps = protocol["day_bootstrap"]["replicates"]
        draws = np.random.default_rng(seed).integers(0,len(names),(reps,len(names)))
        values = difference[draws].sum(axis=1)/denominator[draws].sum(axis=1)
        close(item["difference_w"],difference.sum()/denominator.sum(),"Paired day difference")
        close(item["descriptive_95_interval_w"],np.percentile(values,[2.5,97.5]),"Paired day confidence interval")
        require(item["days"] == len(names) and item["replicates"] == reps and item["seed"] == seed,"Day bootstrap metadata")
    require(len(summary["paired_home_comparisons"]) == (4 if homes else 0),"Home contrast count")
    for item,right in zip(summary["paired_home_comparisons"],METHODS[1:]):
        require(item["left"] == "qaoa_ideal" and item["right"] == right,"Home contrast identity")
        difference = np.array([per_home[str(h)]["qaoa_ideal"]["macro_mae_w"]-per_home[str(h)][right]["macro_mae_w"] for h in homes])
        seed,reps = protocol["home_bootstrap"]["seed"],protocol["home_bootstrap"]["replicates"]
        draws = np.random.default_rng(seed).integers(0,len(homes),(reps,len(homes)))
        close(item["difference_w"],difference.mean(),"Paired equal-home difference")
        close(item["descriptive_95_interval_w"],np.percentile(difference[draws].mean(axis=1),[2.5,97.5]),"Paired home confidence interval")
        require(item["homes"] == len(homes) and item["replicates"] == reps and item["seed"] == seed,"Home bootstrap metadata")
    return len(summary["paired_day_comparisons"])+len(summary["paired_home_comparisons"])


def reload_window(home,window,q):
    loaded = read_block_window(home["source"],window["start_unix"],window["end_unix"],channels=tuple(home["channels"]))
    require(loaded["source_window_sha256"] == q["source_window_sha256"],"Window/source support hash")
    same(q["quality"],loaded["quality"],"Re-read Issues/gap/zero/source coverage counters",atol=1e-9,rtol=0)
    values = loaded["values"]
    require(len(values) == q["valid_blocks"],"Re-read valid block count")
    require(np.all(np.isfinite(values)),"Source nonfinite values escaped screening")
    for i,name in enumerate(loaded["channels"]):
        require(int(np.count_nonzero(values[:,i] == 0)) == q["quality"]["zero_retained_blocks_by_channel"][name],"Independent retained-zero prevalence")
    require(int(np.count_nonzero(np.all(values[:,1:] == 0,axis=1))) == q["quality"]["all_selected_appliances_zero_retained_blocks"],"Independent all-zero block prevalence")
    return loaded


def window_paths(archive,house,identifier):
    stem=f"house-{house:02d}_{identifier}"
    return [archive/"evaluation"/(stem+suffix) for suffix in ("_scores.json",".npz",".json.gz")]


def window_binding(archive,home,identifier,replay_qaoa):
    paths=window_paths(archive,home["house"],identifier)
    return dict(audit_script_sha256=digest(Path(__file__)),
        utility_sha256={p:digest(ROOT/p) for p in ("scripts/audit_stage_b.py","scripts/audit_stage_d_components.py")},
        protocol_sha256=digest(archive/"protocol.json"),models_sha256=digest(archive/"frozen_models.json"),
        evaluation_freeze_sha256=digest(archive/"evaluation/freeze.json"),
        source_sha256=home["source_sha256"],replay_qaoa=replay_qaoa,
        artifact_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths})


def audit_saved_window(archive,home,model,protocol,window,day,record_file,replay_qaoa):
    """All source/prediction/trace checks for one completed nonempty window."""
    house=home["house"]
    q=record_file["quality"]
    require(q["house"] == house and q["window"] == window,"Completed window identity")
    loaded=reload_window(home,window,q)
    values,timestamps=loaded["values"],loaded["timestamps"]
    _,predictions_path,trace_path=window_paths(archive,house,window["id"])
    require(digest(predictions_path) == record_file["predictions_sha256"] and digest(trace_path) == record_file["trace_sha256"],"Prediction/trace hashes")
    with gzip.open(trace_path,"rt") as handle:
        trace=json.load(handle)
    chunks=trace["chunks"]
    runs=check_chunks(chunks,values[:,0],timestamps,model["event_threshold"])
    traces={(x["model"],x["seed"]):x for x in trace["traces"]}
    require(len(trace["traces"]) == len(traces) == 9 and set(traces) == set(ARMS),"Nine trace arms")
    rows={(x["model"],x["seed"]):x for x in record_file["records"]}
    require(len(record_file["records"]) == len(rows) == 9 and set(rows) == set(ARMS),"Nine scored arms")
    max_chunk_error=max_full_gap=0.
    tested_chunks=replayed_draws=0
    with np.load(predictions_path,allow_pickle=False) as npz:
        require(set(npz.files) == {"timestamps",*(f"{m}_{s}" for m,s in ARMS)},"Saved prediction keys")
        require(np.array_equal(npz["timestamps"],timestamps),"Prediction/source timestamps")
        for method,seed in ARMS:
            row=rows[method,seed]
            require(row["house"] == house and row["window_id"] == window["id"],"Scored arm identity")
            prediction=npz[f"{method}_{seed}"]
            require(prediction.shape == (len(values),4) and np.all(np.isfinite(prediction)),"Complete finite predictions")
            measured=score(values[:,1:],prediction[:,:3],timestamps,model["thresholds"])
            same(row["appliances"],measured,f"House {house}/{window['id']}/{method}/{seed}: all appliance metrics")
            close(row["raw_aggregate_mae_w"],np.mean(np.abs(values[:,0]-prediction.sum(axis=1))),"Raw aggregate MAE",atol=1e-7)
            require(row["blocks"] == len(values) and row["runs"] == len(runs) and row["chunks"] == len(chunks),"Common evaluation coverage")
            require(row["solver_wall_time_s"] == traces[method,seed]["solver_wall_time_s"] and row["solver_wall_time_s"] >= 0,"Inference timing provenance")
            if method in ("qaoa_ideal","uniform","exact_chunk"):
                error,draws=check_chunk_trace(traces[method,seed],chunks,prediction,model,
                    protocol["transferred_angles"],method,seed,house,day,replay_qaoa=replay_qaoa)
                max_chunk_error=max(max_chunk_error,error)
                replayed_draws+=draws
                tested_chunks+=len(chunks)
            elif method == "exact_full":
                max_full_gap=max(max_full_gap,check_full_dp(traces[method,seed],runs,values[:,0],prediction,model))
            else:
                close(prediction,np.tile(model["training_mean"],(len(values),1)),"Training-mean baseline",atol=0,rtol=0)
    return dict(valid_source_blocks=len(values),tested_chunks=tested_chunks,replayed_draws=replayed_draws,
                max_chunk_error=max_chunk_error,max_full_gap=max_full_gap)


def saved_or_checked_window(archive,home,model,protocol,window,day,record_file,replay_qaoa,allow_write):
    """Only reuse a complete receipt with exact matching sources/code/artifacts."""
    binding=window_binding(archive,home,window["id"],replay_qaoa)
    suffix="_qaoa-replay" if replay_qaoa else ""
    cache=archive/"independent_prechecks"/f"house-{home['house']:02d}_{window['id']}{suffix}.json"
    if cache.is_file():
        previous=json.loads(cache.read_text())
        require(previous["status"] == "passed" and previous["binding"] == binding,"Stale/tampered independent window precheck")
        return previous["checks"],binding,True
    checks=audit_saved_window(archive,home,model,protocol,window,day,record_file,replay_qaoa)
    if allow_write:
        cache.parent.mkdir(exist_ok=True)
        with cache.open("x") as handle:
            json.dump(dict(status="passed",binding=binding,checks=checks),handle,indent=2,allow_nan=False)
            handle.write("\n")
    return checks,binding,False


def precheck_available(archive):
    """Audit already-completed window files; do not infer pending/empty days."""
    protocol=json.loads((archive/"protocol.json").read_text())
    models=json.loads((archive/"frozen_models.json").read_text())
    freeze=json.loads((archive/"evaluation/freeze.json").read_text())
    require(freeze["protocol_sha256"] == models["protocol_sha256"] == digest(archive/"protocol.json"),"Protocol precheck provenance")
    require(freeze["models_sha256"] == digest(archive/"frozen_models.json"),"Model precheck provenance")
    require(set(models["homes"]) == {str(h["house"]) for h in protocol["homes"]},"All home models required before precheck")
    for path,expected in protocol["code_sha256"].items():
        require(digest(ROOT/path) == expected,"Frozen code changed during precheck")
    checked=reused=0
    for home in protocol["homes"]:
        house=home["house"]
        require(digest(home["source"]) == home["source_sha256"],"Source changed during precheck")
        # Restrict QAOA replay here to literal day 0. If it is empty, the final
        # audit determines the earliest nonempty day from complete quality.
        for day,window in enumerate(home["manifest"]["splits"]["test"]["windows"]):
            score_path=window_paths(archive,house,window["id"])[0]
            if not score_path.is_file():
                continue
            try:
                record_file=json.loads(score_path.read_text())
            except json.JSONDecodeError:
                continue  # producer may still be closing this particular file
            _,_,cached=saved_or_checked_window(archive,home,models["homes"][str(house)],protocol,
                window,day,record_file,replay_qaoa=day==0,allow_write=True)
            checked+=1
            reused+=int(cached)
            if checked%10==0:
                print(f"Independent completed-window precheck: {checked} ({reused} previously checked)",flush=True)
    return dict(status="partial_prechecks_only",completed_windows_checked=checked,reused_windows=reused)


def audit(archive):
    started = time.perf_counter()
    # Completion gate is evaluated before opening any external test data.
    require((archive/"summary.json").is_file(),"Wait for the complete evaluation summary before auditing test data")
    hashes = {}

    def read(path):
        path = Path(path)
        hashes[str(path.relative_to(ROOT))] = digest(path)
        return json.loads(path.read_text())

    protocol,models,summary,quality,records,freeze = [read(archive/path) for path in (
        "protocol.json","frozen_models.json","summary.json","test_quality.json","test_windows.json","evaluation/freeze.json")]
    protocol_sha = digest(archive/"protocol.json")
    require(protocol_sha == models["protocol_sha256"] == summary["protocol_sha256"] == freeze["protocol_sha256"],"Protocol provenance mismatch")
    require(digest(archive/"frozen_models.json") == freeze["models_sha256"],"Models changed after test start")
    for path,expected in protocol["code_sha256"].items():
        require(digest(ROOT/path) == expected,f"Frozen source changed: {path}")
    require(digest(ROOT/"data/external/refit/source.json") == protocol["source_metadata_sha256"],"Official source metadata changed")
    require(digest(protocol["transferred_angles"]["path"]) == protocol["transferred_angles"]["sha256"],"Transferred angles changed")
    require(protocol["methods"] == list(METHODS) and protocol["seeds"] == list(SEEDS) and protocol["shots"] == 256,"Frozen inference settings")
    expected = {(h["house"],w["id"]):w for h in protocol["homes"] for w in h["manifest"]["splits"]["test"]["windows"]}
    actual = {(q["house"],q["window"]["id"]):q for q in quality}
    require(len(expected) == len(actual) == len(quality) == 420 and expected.keys() == actual.keys(),"Every selected quality window must be retained")
    row_map = {}
    for row in records:
        key = row["house"],row["window_id"],row["model"],row["seed"]
        require(key not in row_map,"Duplicate model/seed/window record")
        row_map[key] = row
    expected_row_keys = {(*key,method,seed) for key,q in actual.items() if q["valid_blocks"] for method,seed in ARMS}
    require(row_map.keys() == expected_row_keys,"Nonempty windows require all nine arms; empty windows no arms")
    require(set(models["homes"]) == {str(h["house"]) for h in protocol["homes"]},"All home models must have been fitted before evaluation")
    checked_records = []
    source_blocks = tested_chunks = replayed_draws = empty_windows = cached_windows = 0
    max_chunk_error = max_full_gap = 0.
    selected_qaoa_days = {}
    for home in protocol["homes"]:
        house = home["house"]
        require(digest(home["source"]) == home["source_sha256"],f"Official CSV changed: house {house}")
        model = models["homes"][str(house)]
        train = home["manifest"]["splits"]["train"]["windows"]
        require([q["window"] for q in model["quality"]] == train and len(train) == 30,"Training-only model quality coverage")
        require(model["training_blocks"] == sum(q["valid_blocks"] for q in model["quality"]),"Training block denominator")
        close(model["penalties"],.1*np.square(model["ranges"]),"Frozen per-home transition scaling")
        require(model["actual_counts"] == list(map(len,model["levels"])) and all(1 <= n <= cap for n,cap in zip(model["actual_counts"],protocol["requested_counts"])),"Fitted state count scope")
        test = home["manifest"]["splits"]["test"]["windows"]
        selected_qaoa_days[house] = next((w["id"] for w in test if actual[house,w["id"]]["valid_blocks"]),None)
        for day,window in enumerate(test):
            identifier = window["id"]
            q = actual[house,identifier]
            require(q["window"] == window,"Quality timestamp mismatch")
            stem = f"house-{house:02d}_{identifier}"
            if not q["valid_blocks"]:
                reload_window(home,window,q)
                empty_windows += 1
                require(not (archive/"evaluation"/(stem+"_scores.json")).exists(),"Empty window received scores")
                continue
            record_file = read(archive/"evaluation"/(stem+"_scores.json"))
            require(record_file["quality"] == q,"Window score/quality mismatch")
            checks,binding,cached=saved_or_checked_window(archive,home,model,protocol,window,day,record_file,
                replay_qaoa=identifier == selected_qaoa_days[house],allow_write=False)
            hashes.update(binding["artifact_sha256"])
            source_blocks+=checks["valid_source_blocks"]
            tested_chunks+=checks["tested_chunks"]
            replayed_draws+=checks["replayed_draws"]
            max_chunk_error=max(max_chunk_error,checks["max_chunk_error"])
            max_full_gap=max(max_full_gap,checks["max_full_gap"])
            cached_windows+=int(cached)
            local_rows = {(x["model"],x["seed"]):x for x in record_file["records"]}
            for method,seed in ARMS:
                row = row_map[house,identifier,method,seed]
                require(local_rows[method,seed] == row,"Consolidated/window score mismatch")
                checked_records.append(row)
            print(f"Audited REFIT house {house}: {day+1}/30 selected windows",flush=True)
    require(len(checked_records) == len(records),"Some records not independently scored")
    intervals = audit_statistics(checked_records,protocol,summary)
    # Reconcile the public coverage receipt independently from the raw quality rows.
    require(set(summary["coverage"]) == {str(h["house"]) for h in protocol["homes"]},"Coverage includes all selected homes")
    for home in protocol["homes"]:
        house = home["house"]
        rows = [q for q in quality if q["house"] == house]
        expected = dict(selected_windows=30,nonempty_windows=sum(q["valid_blocks"] > 0 for q in rows),
            expected_blocks=sum(q["quality"]["expected_complete_blocks"] for q in rows),valid_blocks=sum(q["valid_blocks"] for q in rows),
            empty_windows=[q["window"]["id"] for q in rows if not q["valid_blocks"]],
            issues_rows=sum(q["quality"]["rows_issues"] for q in rows),
            large_gap_seconds=sum(q["quality"]["unsupported_large_gap_seconds"] for q in rows),
            invalid_supported_seconds=sum(q["quality"]["supported_invalid_seconds"] for q in rows),
            valid_source_rows_including_support=sum(q["quality"]["valid_source_rows"] for q in rows),
            all_appliances_zero_source_rows_including_support=sum(q["quality"]["all_selected_appliances_zero_source_rows"] for q in rows),
            all_appliances_zero_retained_blocks=sum(q["quality"]["all_selected_appliances_zero_retained_blocks"] for q in rows),
            zero_retained_blocks_by_channel={name:sum(q["quality"]["zero_retained_blocks_by_channel"][name] for q in rows) for name in rows[0]["quality"]["zero_retained_blocks_by_channel"]})
        expected["coverage_fraction"] = expected["valid_blocks"]/expected["expected_blocks"]
        expected["all_appliances_zero_retained_fraction"] = expected["all_appliances_zero_retained_blocks"]/expected["valid_blocks"] if expected["valid_blocks"] else None
        same(summary["coverage"][str(house)],expected,"Reported per-home quality/coverage")
    return dict(status="passed",audit_type="independent_refit_evaluation_audit",audited_utc=datetime.now(timezone.utc).isoformat(),
                audit_script_sha256=digest(Path(__file__)),audit_utilities_sha256={p:digest(ROOT/p) for p in
                    ("scripts/audit_stage_b.py","scripts/audit_stage_d_components.py")},
                protocol_sha256=protocol_sha,summary_sha256=digest(archive/"summary.json"),
                frozen_models_sha256=digest(archive/"frozen_models.json"),
                coverage=dict(selected_homes=len(protocol["homes"]),selected_test_windows=420,
                    nonempty_test_windows=420-empty_windows,valid_source_blocks=source_blocks,
                    model_seed_window_records=len(checked_records)),
                checks=dict(all_source_quality_windows=True,all_saved_prediction_metrics=True,
                    all_sample_only_minima_and_own_history=True,all_full_dp_certificates=True,
                    all_reported_pooling_and_bootstrap_intervals=True),
                selected_home_count=len(protocol["homes"]),selected_test_windows_checked=420,
                empty_windows_retained=empty_windows,valid_source_blocks_re_read=source_blocks,
                complete_model_seed_window_records_checked=len(checked_records),
                hash_bound_prechecked_windows_reused=cached_windows,
                own_history_and_observed_sample_chunk_decisions_checked=tested_chunks,
                quantum_replay_day_per_home=selected_qaoa_days,independently_replayed_uniform_and_selected_quantum_draws=replayed_draws,
                maximum_conditional_exact_objective_discrepancy=max_chunk_error,
                maximum_full_dp_dense_certificate_discrepancy=max_full_gap,
                paired_day_and_home_intervals_recomputed=intervals,source_sha256=hashes,
                scope="Every selected test window re-read using the frozen source adapter; every saved prediction, raw-block metric, sample-only choice, history/reset and full-DP certificate checked; uniform draws replayed everywhere, quantum draws replayed on the earliest nonempty test day per home; all pooling and bootstrap calculations independent",
                limitations="Source extraction implementation reused and independently hand-tested, not a second full-data parser; training fits hash/coverage checked but not refitted; quantum draw replay is a timestamp-selected audit sample, not every circuit; cleaned-channel within-home calibrated external evaluation, not quantum advantage",
                runtime_seconds=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,default=ROOT/"results/stage_g/refit/run_001")
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--precheck-available",action="store_true",
                        help="Check only already-saved windows after test execution has started")
    args = parser.parse_args()
    if args.precheck_available:
        print(json.dumps(precheck_available(args.archive.resolve()),indent=2))
        return
    result = audit(args.archive.resolve())
    if not args.check_only:
        with (args.archive/"independent_audit.json").open("x") as handle:
            json.dump(result,handle,indent=2,allow_nan=False)
            handle.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k != "source_sha256"},indent=2))


if __name__ == "__main__":
    main()
