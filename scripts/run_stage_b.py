#!/usr/bin/env python3
"""Frozen, resumable categorical Stage B controlled simulation programme.

Preparation fixes the design and source hashes before scored execution. This
is local classical simulation, not QPU execution or a speedup experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quantum_nilm.stage_b import make_stage_b_instance, optimize_stage_b_qaoa
from quantum_nilm.stage_b_noise import prepare_noise_circuit, simulate_prepared_noise

SHAPES = [([2, 2], 1), ([2, 2], 2), ([3, 2], 2), ([3, 3], 2),
          ([4, 3, 2, 3], 1), ([4, 3, 2, 3], 2)]
SOURCES = ["scripts/run_stage_b.py", "src/quantum_nilm/stage_b.py",
           "src/quantum_nilm/stage_b_noise.py", "src/quantum_nilm/categorical_qaoa.py",
           "src/quantum_nilm/categorical_ibm.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        def numpy_json(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, np.generic):
                return x.item()
            raise TypeError(f"Unsupported archive type: {type(x).__name__}")
        json.dump(value, stream, indent=2, allow_nan=False, default=numpy_json)
        stream.write("\n")


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def problem_id(counts, segments, seed, sigma):
    return f"m{'-'.join(map(str, counts))}_k{segments}_s{seed}_n{sigma:g}"


def shots99(probability):
    if probability <= 0:
        return None  # mathematically infinite, not zero or unmeasured
    if probability >= 1:
        return 1
    return int(math.ceil(math.log(0.01) / math.log1p(-probability)))


def endpoints(problem, truth, probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != problem.energies.shape or np.any(probabilities < -1e-12):
        raise ValueError("Invalid unconditional feasible probabilities")
    feasible = float(probabilities.sum())
    if not np.isfinite(feasible) or feasible > 1 + 1e-9 or feasible < 0:
        raise ValueError("Invalid feasible probability mass")
    energies = problem.energies
    optimum = float(energies.min())
    span = float(np.ptp(energies))
    tolerance = 1e-10 * max(1.0, span)
    mask = energies <= optimum + tolerance
    popt = min(1.0, float(probabilities[mask].sum()))
    canonical = int(np.flatnonzero(mask)[0])
    category = np.mean(problem.states[canonical] == truth)
    bit = 1 - 2 * np.count_nonzero(problem.states[canonical] != truth) / problem.num_qubits
    expected = float(probabilities @ energies / feasible) if feasible > 0 else None
    ngap = (expected - optimum) / span if expected is not None and span else 0.0 if feasible else None
    return dict(feasible_probability=min(1.0, feasible), invalid_probability=max(0.0, 1-feasible),
                optimum_probability=popt, shots_to_optimum_99=shots99(popt),
                exact_cost=optimum, energy_span=span, optimum_tolerance=tolerance,
                optimum_count=int(mask.sum()), canonical_exact_index=canonical,
                exact_truth_category_accuracy=float(category), exact_truth_bit_accuracy=float(bit),
                conditional_expected_cost=expected, conditional_expected_normalized_gap=ngap)


def sample_trials(problem, truth, probabilities, identifier, budgets, repeats):
    p = np.maximum(0, np.asarray(probabilities, dtype=float))
    invalid = max(0.0, 1.0-float(p.sum()))
    distribution = np.append(p, invalid)
    distribution /= distribution.sum()  # roundoff only; invalid stays in denominator
    meta = endpoints(problem, truth, p)
    canonical = problem.states[meta["canonical_exact_index"]]
    for shots in budgets:
        for repeat in range(repeats):
            # Common random numbers across samplers/depths/noise on a matched
            # latent instance; repeat and shot-budget streams are independent.
            seed = stable_seed(identifier.split("_p")[0], shots, repeat, "stage-b-measurements")
            draws = np.random.default_rng(seed).choice(len(distribution), shots, p=distribution)
            feasible_draws = draws[draws < len(p)]
            best = None
            gap = None
            truth_category = truth_bit = exact_category = exact_bit = 0.0
            if feasible_draws.size:
                observed = np.unique(feasible_draws)
                best = int(observed[np.argmin(problem.energies[observed])])
                gap = max(0.0, float(problem.energies[best] - meta["exact_cost"]))
                state = problem.states[best]
                truth_category = float(np.mean(state == truth))
                truth_bit = 1 - 2*np.count_nonzero(state != truth)/problem.num_qubits
                exact_category = float(np.mean(state == canonical))
                exact_bit = 1 - 2*np.count_nonzero(state != canonical)/problem.num_qubits
            normalized = gap / meta["energy_span"] if gap is not None and meta["energy_span"] else 0.0 if gap is not None else None
            yield dict(distribution_id=identifier, shots=shots, repeat=repeat, sample_seed=seed,
                       feasible_shots=int(feasible_draws.size), no_feasible=int(best is None),
                       best_index=best, best_cost=float(problem.energies[best]) if best is not None else None,
                       certified_gap=gap, normalized_gap=normalized,
                       failure_penalized_normalized_gap=1.0 if normalized is None else normalized,
                       hit_optimum=int(gap is not None and gap <= meta["optimum_tolerance"]),
                       truth_category_accuracy=truth_category, truth_bit_accuracy=truth_bit,
                       canonical_exact_category_accuracy=exact_category, canonical_exact_bit_accuracy=exact_bit)


def prepare(out, noise_max_qubits):
    import scipy, qiskit, qiskit_aer
    design = dict(schema=1, prepared_utc=datetime.now(timezone.utc).isoformat(),
        scope="Local categorical controlled simulation; not QPU results or quantum advantage",
        shapes=[dict(state_counts=c, segments=k) for c,k in SHAPES],
        instance_seeds=list(range(1001,1009)), measurement_noise=[0.0,0.01,0.05],
        depths=[1,2], optimizer_restart_seeds=[7001,7002], evaluations_per_restart=512,
        angle_bounds=dict(gamma=[0,8*math.pi], beta=[0,math.pi]),
        optimization="Ideal exact expected cost, independently optimized at each depth; no label or exact optimum in selection",
        noise_max_qubits=noise_max_qubits, noise_levels=["low","high"],
        noise_shapes=[dict(state_counts=c, segments=k) for c,k in SHAPES if sum(c)*k <= noise_max_qubits],
        noise_design={"low":dict(one_qubit_depolarization=0.0001,two_qubit_depolarization=0.001,readout_flip=0.005),
                      "high":dict(one_qubit_depolarization=0.001,two_qubit_depolarization=0.01,readout_flip=0.02)},
        noise_scope="Full-connectivity fixed-basis circuit, no routing/calibration/drift/thermal model; freeze ideal-trained angles",
        shots=[32,128,512,2048], sampling_repeats=32,
        sample_randomness="Common RNG seeds across depths, noise and uniform within an instance; independent streams for each shot budget and repeat; saved seeds permit raw draws to be regenerated",
        baseline="Uniform feasible sampling at every shot budget, no gate noise applied to this classical baseline",
        failures="Invalid physical shots are an extra outcome; no feasible sample means abstention, hit=0, accuracy=0, penalized normalized gap=1; raw gap missing",
        exact="Enumerate feasible states; optimum tolerance=1e-10*max(1,energy span); ties choose lowest observed feasible index",
        uncertainty="95% paired percentile bootstrap over 8 base-instance seeds; all noise levels/depths/repeats share a cluster; descriptive intervals, no multiplicity-adjusted significance claim",
        primary_comparison="p2 minus p1 ideal failure-penalized normalized best-sample gap at 2048 shots, separately by shape and measurement noise",
        bootstrap_replicates=10000, bootstrap_seed=88201,
        reporting="Category accuracy primary, size-sensitive one-hot bit accuracy secondary. Instance-level summaries, probabilities, optimization histories, raw trial metrics and timings; every planned condition must finish",
        source_sha256={name:sha(ROOT/name) for name in SOURCES},
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,qiskit=qiskit.__version__,aer=qiskit_aer.__version__,platform=platform.platform(),machine=platform.machine()))
    design["expected_instances"] = len(SHAPES)*8*3
    small = sum(sum(c)*k <= noise_max_qubits for c,k in SHAPES)
    design["expected_optimizations"] = design["expected_instances"]*2
    design["expected_distributions"] = design["expected_instances"]*3 + small*8*3*2*2
    dump_new(out/"protocol.json", design)
    print(json.dumps({k:design[k] for k in ("expected_instances","expected_optimizations","expected_distributions","noise_max_qubits")}))


def check_sources(protocol):
    for name, digest in protocol["source_sha256"].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError(f"Frozen source changed: {name}; use a new archive, do not mutate a scored run")


def run(out):
    protocol = json.loads((out/"protocol.json").read_text())
    check_sources(protocol)
    protocol_sha = sha(out/"protocol.json")
    for shape in protocol["shapes"]:
        for seed in protocol["instance_seeds"]:
            for sigma in protocol["measurement_noise"]:
                counts, k = shape["state_counts"], shape["segments"]
                iid = problem_id(counts,k,seed,sigma)
                tick = time.perf_counter()
                instance = make_stage_b_instance(counts,k,seed,sigma)
                prep_time = time.perf_counter()-tick
                problem, truth = instance.problem, instance.truth
                input_path = out/"instances"/(iid+".json")
                if not input_path.exists():
                    dump_new(input_path, dict(id=iid,state_counts=counts,segments=k,seed=seed,measurement_noise=sigma,
                        truth=truth.tolist(),aggregate=problem.aggregate.tolist(),levels=[x.tolist() for x in problem.levels],
                        switch_penalty=problem.switch_penalty.tolist(),segment_weights=problem.segment_weights.tolist(),
                        scale=problem.scale,qubits=problem.num_qubits,feasible_states=problem.num_feasible_states,
                        metadata=instance.metadata,preparation_seconds=prep_time,protocol_sha256=protocol_sha))
                common = dict(instance_id=iid,state_counts=counts,segments=k,seed=seed,measurement_noise=sigma,
                              qubits=problem.num_qubits,feasible_states=problem.num_feasible_states,protocol_sha256=protocol_sha)
                def save_distribution(label, depth, probabilities, extra):
                    did = f"{iid}_p{depth}_{label}"
                    path = out/"distributions"/(did+".json")
                    if path.exists():
                        return
                    payload=dict(common,distribution_id=did,depth=depth,noise=label,**endpoints(problem,truth,probabilities),**extra)
                    payload["probabilities"] = np.asarray(probabilities).tolist()
                    tick_trials = time.perf_counter()
                    rows=list(sample_trials(problem,truth,probabilities,did,protocol["shots"],protocol["sampling_repeats"]))
                    payload["sampling_seconds"] = time.perf_counter()-tick_trials
                    payload["trials"] = rows
                    dump_new(path,payload)
                    print(f"Completed {did}",flush=True)
                save_distribution("uniform",0,np.full(problem.num_feasible_states,1/problem.num_feasible_states),{})
                for depth in protocol["depths"]:
                    opt_path = out/"optimizations"/f"{iid}_p{depth}.json"
                    if not opt_path.exists():
                        optimized = optimize_stage_b_qaoa(problem,depth,restart_seeds=tuple(protocol["optimizer_restart_seeds"]),evaluations_per_restart=protocol["evaluations_per_restart"])
                        dump_new(opt_path,dict(common,**optimized))
                    optimized=json.loads(opt_path.read_text())
                    save_distribution("ideal",depth,optimized["probabilities"],dict(optimization_seconds=optimized["runtime_seconds"],optimizer_evaluations=optimized["evaluations"]))
                    if problem.num_qubits <= protocol["noise_max_qubits"]:
                        pending=[level for level in protocol["noise_levels"] if not (out/"distributions"/f"{iid}_p{depth}_{level}.json").exists()]
                        if pending:
                            prepared=prepare_noise_circuit(problem,optimized["gammas"],optimized["betas"],max_qubits=protocol["noise_max_qubits"])
                            for level in pending:
                                noisy=simulate_prepared_noise(prepared,noise=level)
                                # The adapter supplies all-shot feasible probabilities, never postselected probabilities.
                                probabilities=noisy.pop("feasible_probabilities")
                                noisy.pop("invalid_probability",None)
                                save_distribution(level,depth,probabilities,dict(noise_metadata=noisy))
    print("All planned computations complete. Run --mode summarize to audit coverage and create summaries.",flush=True)


def bootstrap(values, seed, count):
    values=np.asarray(values,dtype=float)
    draws=np.random.default_rng(seed).integers(0,len(values),size=(count,len(values)))
    lo,hi=np.quantile(values[draws].mean(axis=1),[.025,.975])
    return dict(mean=float(values.mean()),ci95=[float(lo),float(hi)],n_instances=len(values))


def summarize(out):
    protocol=json.loads((out/"protocol.json").read_text())
    check_sources(protocol)
    records=[json.loads(p.read_text()) for p in sorted((out/"distributions").glob("*.json"))]
    if len(records)!=protocol["expected_distributions"]:
        raise RuntimeError(f"Incomplete: {len(records)}/{protocol['expected_distributions']} distributions")
    expected_ids=set()
    for shape in protocol["shapes"]:
        for seed in protocol["instance_seeds"]:
            for sigma in protocol["measurement_noise"]:
                iid=problem_id(shape["state_counts"],shape["segments"],seed,sigma)
                expected_ids.add(f"{iid}_p0_uniform")
                for depth in protocol["depths"]:
                    expected_ids.add(f"{iid}_p{depth}_ideal")
                    if sum(shape["state_counts"])*shape["segments"]<=protocol["noise_max_qubits"]:
                        expected_ids.update(f"{iid}_p{depth}_{noise}" for noise in protocol["noise_levels"])
    if {r["distribution_id"] for r in records} != expected_ids:
        raise RuntimeError("Distribution IDs differ from the frozen matrix")
    grouped={}
    flat=[]
    for record in records:
        expected_trials=len(protocol["shots"])*protocol["sampling_repeats"]
        if len(record["trials"]) != expected_trials:
            raise RuntimeError("Missing trials")
        if {(t["shots"],t["repeat"]) for t in record["trials"]}!={(s,r) for s in protocol["shots"] for r in range(protocol["sampling_repeats"])}:
            raise RuntimeError("Duplicate or missing trial keys")
        key=(tuple(record["state_counts"]),record["segments"],record["measurement_noise"],record["depth"],record["noise"])
        grouped.setdefault(key,[]).append(record)
        for trial in record["trials"]:
            flat.append(dict(instance_id=record["instance_id"],qubits=record["qubits"],seed=record["seed"],measurement_noise=record["measurement_noise"],depth=record["depth"],noise=record["noise"],**trial))
    groups=[]
    for key, rows in sorted(grouped.items()):
        if sorted(r["seed"] for r in rows)!=protocol["instance_seeds"]:
            raise RuntimeError("Missing or duplicate base-instance seeds")
        counts,k,sigma,depth,noise=key
        group=dict(state_counts=counts,segments=k,measurement_noise=sigma,depth=depth,noise=noise,qubits=rows[0]["qubits"],n_instances=len(rows))
        for field in ("feasible_probability","optimum_probability","conditional_expected_normalized_gap","exact_truth_category_accuracy","exact_truth_bit_accuracy"):
            group[field]=bootstrap([r[field] for r in rows],stable_seed(protocol["bootstrap_seed"],key,field),protocol["bootstrap_replicates"])
        finite=[r["shots_to_optimum_99"] for r in rows if r["shots_to_optimum_99"] is not None]
        group["shots99"]=dict(median=float(np.median(finite)) if finite else None,min=min(finite) if finite else None,max=max(finite) if finite else None,infinite_instances=len(rows)-len(finite))
        group["shot_results"]={}
        for shots in protocol["shots"]:
            trials=[[t for t in r["trials"] if t["shots"]==shots] for r in rows]
            metrics={field:bootstrap([np.mean([t[field] for t in trial]) for trial in trials],stable_seed(protocol["bootstrap_seed"],key,shots,field),protocol["bootstrap_replicates"]) for field in ("hit_optimum","no_feasible","failure_penalized_normalized_gap","truth_category_accuracy","truth_bit_accuracy")}
            metrics["analytic_hit_probability_mean"]=float(np.mean([-np.expm1(shots*np.log1p(-min(1-1e-16,r["optimum_probability"]))) if r["optimum_probability"]<1 else 1 for r in rows]))
            group["shot_results"][str(shots)]=metrics
        groups.append(group)
    paired=[]
    for shape in protocol["shapes"]:
        for sigma in protocol["measurement_noise"]:
            for noise in ["ideal"]+protocol["noise_levels"]:
                keys=[(tuple(shape["state_counts"]),shape["segments"],sigma,p,noise) for p in (1,2)]
                if any(key not in grouped for key in keys):
                    continue
                maps=[{r["seed"]:r for r in grouped[key]} for key in keys]
                differences=[]
                for seed in protocol["instance_seeds"]:
                    means=[np.mean([t["failure_penalized_normalized_gap"] for t in m[seed]["trials"] if t["shots"]==2048]) for m in maps]
                    differences.append(float(means[1]-means[0]))
                paired.append(dict(**shape,measurement_noise=sigma,noise=noise,metric="p2_minus_p1_penalized_normalized_gap_at_2048",per_seed=differences,**bootstrap(differences,stable_seed(protocol["bootstrap_seed"],keys,"paired"),protocol["bootstrap_replicates"])))
    summary=dict(status="complete",protocol_sha256=sha(out/"protocol.json"),instances=protocol["expected_instances"],optimizations=protocol["expected_optimizations"],distributions=len(records),sampling_trials=len(flat),groups=groups,paired_depth_comparisons=paired)
    dump_new(out/"summary.json",summary)
    with (out/"trials.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]))
        writer.writeheader();writer.writerows(flat)
    print(json.dumps({k:summary[k] for k in ("status","instances","optimizations","distributions","sampling_trials")}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["prepare","run","summarize"],required=True)
    parser.add_argument("--output",type=Path,default=ROOT/"results/stage_b/run_002")
    parser.add_argument("--noise-max-qubits",type=int,choices=[8,10,12],default=10)
    args=parser.parse_args()
    if args.mode=="prepare":prepare(args.output,args.noise_max_qubits)
    elif args.mode=="run":run(args.output)
    else:summarize(args.output)


if __name__=="__main__":main()
