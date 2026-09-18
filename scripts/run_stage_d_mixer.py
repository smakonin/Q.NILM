#!/usr/bin/env python3
"""Frozen, resumable ideal Stage D architecture/mixer/initialization controls."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quantum_nilm.categorical_ibm import build_categorical_qaoa_circuit
from quantum_nilm.categorical_qaoa import prepare_categorical_problem
from quantum_nilm.stage_b import make_stage_b_instance
from quantum_nilm.stage_d_mixer import (
    ARMS, audit_penalty_problem, build_penalty_mixer_circuit, compile_and_audit,
    optimize_penalty_mixer, penalty_mixer_probabilities, prepare_penalty_problem,
)
from run_stage_b import bootstrap, dump_new, endpoints, sample_trials, sha, stable_seed


SOURCES = ["scripts/run_stage_d_mixer.py", "src/quantum_nilm/stage_d_mixer.py",
           "scripts/run_stage_b.py", "src/quantum_nilm/stage_b.py",
           "src/quantum_nilm/categorical_qaoa.py", "src/quantum_nilm/categorical_ibm.py"]


def read(path):
    return json.loads(Path(path).read_text())


def reconstruct(record):
    return prepare_categorical_problem(record["aggregate"], record["levels"],
                                       record["switch_penalty"], record["segment_weights"])


def preflight_audits():
    checks = []
    for counts in ((2, 2), (3, 2)):
        problem = prepare_penalty_problem(make_stage_b_instance(counts, 1, 99999, 0.01).problem)
        penalty_audit = audit_penalty_problem(problem)
        for depth in (1, 2):
            gammas, betas = [1.37, 2.11][:depth], [0.29, 0.61][:depth]
            for initialization in ("h", "w"):
                circuit = build_penalty_mixer_circuit(problem, gammas, betas, initialization)
                expected = penalty_mixer_probabilities(problem, gammas, betas, initialization)
                checks.append(dict(state_counts=list(counts), segments=1, seed=99999, depth=depth,
                                   initialization=initialization, penalty_audit=penalty_audit,
                                   circuit_audit=compile_and_audit(circuit, expected)))
    return checks


def prepare(out, stage_b):
    import scipy, qiskit
    if (out / "protocol.json").exists():
        raise FileExistsError("An existing Stage D protocol must not be replaced")
    b_protocol, b_summary = read(stage_b / "protocol.json"), read(stage_b / "summary.json")
    if b_summary["status"] != "complete" or b_summary["protocol_sha256"] != sha(stage_b / "protocol.json"):
        raise ValueError("Completed, hash-matched Stage B archive required")
    selected, manifest = [], {}
    for path in sorted((stage_b / "instances").glob("*.json")):
        value = read(path)
        if value["qubits"] > 12:
            continue
        if value["protocol_sha256"] != b_summary["protocol_sha256"]:
            raise ValueError("Source input has a different protocol hash")
        relative = str(path.relative_to(ROOT))
        item = {"id": value["id"], "instance_path": relative, "xy": {}}
        manifest[relative] = sha(path)
        for depth in b_protocol["depths"]:
            opt = stage_b / "optimizations" / f"{value['id']}_p{depth}.json"
            distribution = stage_b / "distributions" / f"{value['id']}_p{depth}_ideal.json"
            for source in (opt, distribution):
                manifest[str(source.relative_to(ROOT))] = sha(source)
            item["xy"][str(depth)] = {"optimization_path": str(opt.relative_to(ROOT)), "distribution_path": str(distribution.relative_to(ROOT))}
        selected.append(item)
    if len(selected) != 120:
        raise ValueError("Expected all 120 Stage B inputs at <=12 qubits")
    protocol = {
        "schema": 1, "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Ideal controlled architecture/mixer/initialization ablation; not QPU execution",
        "stage_b_root": str(stage_b.relative_to(ROOT)),
        "stage_b_protocol_sha256": sha(stage_b / "protocol.json"),
        "stage_b_summary_sha256": sha(stage_b / "summary.json"),
        "stage_b_results_seen_before_design": True,
        "shapes": [value for value in b_protocol["shapes"] if sum(value["state_counts"]) * value["segments"] <= 12],
        "instance_seeds": b_protocol["instance_seeds"], "measurement_noise": b_protocol["measurement_noise"],
        "depths": [1, 2], "arms": list(ARMS),
        "initialization": {"w_xy": "uniform positive one-hot W states", "h_x_penalty": "H on every physical qubit", "w_x_penalty": "same W preparation as w_xy"},
        "mixer": {"w_xy": "ascending adjacent RXX(beta) RYY(beta)", "h_x_penalty": "RX(2 beta) on every physical qubit", "w_x_penalty": "RX(2 beta) on every physical qubit"},
        "barrier": "A=1+2S, S=sum(abs(base linear))+sum(abs(base quadratic)); add A*sum_register(occupancy-1)^2",
        "barrier_proof": "base energy in [c-S,c+S], invalid penalty>=A =>invalid energy>=c+S+1>every feasible energy",
        "phase_normalization": "XY uses original coefficient L1 scale; penalty arms use aggregated augmented coefficient L1 scale, excluding constants",
        "training_objective": "XY ideal expected feasible objective; X arms ideal full-binary penalized expected cost; no planted label or minimizing state used in selection",
        "optimizer_restart_seeds": [7001, 7002], "evaluations_per_restart": 512,
        "search": "32 candidates including zero angles; bounded Powell from best two distinct candidates,240calls each, unused calls filled by seeded random candidates; no lower-depth warm start",
        "gamma_bounds": [0, 8 * np.pi], "beta_bounds": [0, np.pi],
        "boundary_roundoff": "Snap <=8*eps*max(1,abs(bounds)) only; reject larger violations, archive corrections",
        "shots": [32, 128, 512, 2048], "sampling_repeats": 32,
        "sampling": "Reuse Stage B common random seed function and observed-feasible-only selection, with invalid probability kept as an additional outcome",
        "failure": "No feasible draw means abstention: hit=0, accuracy=0, raw gap missing, failure-penalized normalized gap=1; no fallback",
        "certification": "Enumerate full binary extension and feasible states; verify every invalid energy exceeds every feasible energy and all feasible costs match original objective",
        "resource_audit": "Every selected arm/depth circuit compiled all-to-all into rz/sx/x/cx,opt1,seed17; full compiled Qiskit statevector compared with NumPy probabilities",
        "primary_contrasts": ["h_x_penalty minus w_xy", "w_x_penalty minus w_xy"],
        "secondary_contrast": "h_x_penalty minus w_x_penalty isolates initialization within the penalized X architecture",
        "comparison_metric": "failure_penalized_normalized_best_sample_gap_at_2048; separately by shape,depth,measurement noise",
        "uncertainty": "Average 32 repeats within each instance; paired percentile bootstrap over8base seeds,10000replicates; exploratory descriptive intervals, no multiplicity-adjusted significance",
        "bootstrap_replicates": 10000, "bootstrap_seed": 88301,
        "interpretation": "Architectures differ in mixer, penalties and scaling; W/X controls initialization only. Conservative sufficient penalty is not an optimally tuned penalty baseline. Equal depth/call budgets do not mean equal gate or simulation cost.",
        "expected_instances": 120, "expected_new_optimizations": 480, "expected_reused_xy_optimizations": 240,
        "expected_distributions": 720, "expected_sampling_trials": 92160,
        "selected_inputs": selected, "stage_b_source_sha256": manifest,
        "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
        "preflight_controls": preflight_audits(),
        "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "qiskit": qiskit.__version__, "platform": platform.platform()},
    }
    dump_new(out / "protocol.json", protocol)
    print(json.dumps({key: protocol[key] for key in ("expected_instances", "expected_new_optimizations", "expected_reused_xy_optimizations", "expected_distributions", "expected_sampling_trials")}))


def check_sources(protocol):
    for collection in ("source_sha256", "stage_b_source_sha256"):
        for name, digest in protocol[collection].items():
            if sha(ROOT / name) != digest:
                raise ValueError(f"Frozen source changed: {name}; preserve the archive and use a new run")


def run(out):
    protocol = read(out / "protocol.json")
    check_sources(protocol)
    protocol_sha = sha(out / "protocol.json")
    for source in protocol["selected_inputs"]:
        raw = read(ROOT / source["instance_path"])
        tick = time.perf_counter()
        base = reconstruct(raw)
        problem = prepare_penalty_problem(base)
        certified = audit_penalty_problem(problem)
        preparation_seconds = time.perf_counter() - tick
        truth = np.asarray(raw["truth"], dtype=int)
        common = {key: raw[key] for key in ("state_counts", "segments", "seed", "measurement_noise", "qubits", "feasible_states")}
        common.update(instance_id=source["id"], protocol_sha256=protocol_sha)
        instance_path = out / "instances" / f"{source['id']}.json"
        if not instance_path.exists():
            dump_new(instance_path, dict(common, stage_b_input_path=source["instance_path"], stage_b_input_sha256=sha(ROOT / source["instance_path"]),
                                          penalty_audit=certified, preparation_seconds=preparation_seconds))
        for depth in protocol["depths"]:
            for arm in protocol["arms"]:
                identifier = f"{source['id']}_p{depth}_{arm}"
                path = out / "distributions" / f"{identifier}.json"
                if path.exists():
                    continue
                xy_source = source["xy"][str(depth)]
                if arm == "w_xy":
                    optimized = read(ROOT / xy_source["optimization_path"])
                    stage_b_distribution = read(ROOT / xy_source["distribution_path"])
                    feasible_probabilities = np.asarray(stage_b_distribution["probabilities"])
                    full_probabilities = np.zeros(len(problem.energies))
                    full_probabilities[problem.feasible_indices] = feasible_probabilities
                    circuit = build_categorical_qaoa_circuit(base, optimized["gammas"], optimized["betas"], measure=False)
                    training = {"reused": True, "source_path": xy_source["optimization_path"],
                                "source_sha256": sha(ROOT / xy_source["optimization_path"]),
                                "distribution_source_path": xy_source["distribution_path"],
                                "distribution_source_sha256": sha(ROOT / xy_source["distribution_path"]),
                                "evaluations": optimized["evaluations"], "runtime_seconds": optimized["runtime_seconds"]}
                else:
                    initialization = "h" if arm == "h_x_penalty" else "w"
                    opt_path = out / "optimizations" / f"{identifier}.json"
                    if not opt_path.exists():
                        optimized = optimize_penalty_mixer(problem, depth, initialization,
                                                            tuple(protocol["optimizer_restart_seeds"]), protocol["evaluations_per_restart"])
                        dump_new(opt_path, dict(common, arm=arm, **optimized))
                    optimized = read(opt_path)
                    full_probabilities = np.asarray(optimized["probabilities"])
                    feasible_probabilities = full_probabilities[problem.feasible_indices]
                    circuit = build_penalty_mixer_circuit(problem, optimized["gammas"], optimized["betas"], initialization)
                    training = {"reused": False, "source_path": str(opt_path.relative_to(ROOT)),
                                "source_sha256": sha(opt_path), "evaluations": optimized["evaluations"], "runtime_seconds": optimized["runtime_seconds"]}
                resources = compile_and_audit(circuit, full_probabilities)
                tick = time.perf_counter()
                trials = list(sample_trials(base, truth, feasible_probabilities, identifier, protocol["shots"], protocol["sampling_repeats"]))
                sampling_seconds = time.perf_counter() - tick
                payload = dict(common, distribution_id=identifier, arm=arm, depth=depth,
                               **endpoints(base, truth, feasible_probabilities),
                               probabilities=feasible_probabilities.tolist(), gammas=optimized["gammas"], betas=optimized["betas"],
                               full_training_expected_cost=optimized["expected_cost"], training=training,
                               penalty_audit=certified, resources=resources, sampling_seconds=sampling_seconds, trials=trials)
                dump_new(path, payload)
                print(f"Completed {identifier}", flush=True)
    print("All mixer/initialization conditions complete; run --mode summarize.", flush=True)


def group_key(record):
    return (tuple(record["state_counts"]), record["segments"], record["measurement_noise"], record["depth"], record["arm"])


def summarize(out):
    protocol = read(out / "protocol.json")
    check_sources(protocol)
    rows = [read(path) for path in sorted((out / "distributions").glob("*.json"))]
    expected = {f"{value['id']}_p{depth}_{arm}" for value in protocol["selected_inputs"] for depth in protocol["depths"] for arm in protocol["arms"]}
    if len(rows) != len(expected) or {value["distribution_id"] for value in rows} != expected:
        raise ValueError("Incomplete or duplicate distribution matrix")
    groups, flat = {}, []
    for value in rows:
        if value["protocol_sha256"] != sha(out / "protocol.json"):
            raise ValueError("Record protocol mismatch")
        expected_trials = {(shots, repeat) for shots in protocol["shots"] for repeat in range(protocol["sampling_repeats"])}
        if len(value["trials"]) != len(expected_trials) or {(trial["shots"], trial["repeat"]) for trial in value["trials"]} != expected_trials:
            raise ValueError("Incomplete or duplicate trial matrix")
        groups.setdefault(group_key(value), []).append(value)
        for trial in value["trials"]:
            flat.append(dict(instance_id=value["instance_id"], arm=value["arm"], depth=value["depth"], qubits=value["qubits"], seed=value["seed"], measurement_noise=value["measurement_noise"], **trial))
    summarized = []
    for key, members in sorted(groups.items()):
        if sorted(value["seed"] for value in members) != protocol["instance_seeds"]:
            raise ValueError("Missing or duplicate base-instance seeds")
        counts, segments, sigma, depth, arm = key
        result = dict(state_counts=counts, segments=segments, measurement_noise=sigma, depth=depth,
                      arm=arm, qubits=members[0]["qubits"], n_instances=len(members))
        for field in ("feasible_probability", "optimum_probability", "conditional_expected_normalized_gap", "exact_truth_category_accuracy"):
            result[field] = bootstrap([value[field] for value in members], stable_seed(protocol["bootstrap_seed"], key, field), protocol["bootstrap_replicates"])
        result["shot_results"] = {}
        for shots in protocol["shots"]:
            result["shot_results"][str(shots)] = {}
            for field in ("hit_optimum", "no_feasible", "failure_penalized_normalized_gap", "truth_category_accuracy", "truth_bit_accuracy"):
                means = [float(np.mean([trial[field] for trial in value["trials"] if trial["shots"] == shots])) for value in members]
                result["shot_results"][str(shots)][field] = bootstrap(means, stable_seed(protocol["bootstrap_seed"], key, shots, field), protocol["bootstrap_replicates"])
        result["mean_compiled_depth"] = float(np.mean([value["resources"]["compiled_depth"] for value in members]))
        result["mean_compiled_cx"] = float(np.mean([value["resources"]["compiled_gate_counts"].get("cx", 0) for value in members]))
        result["mean_penalty_to_xy_scale_ratio"] = float(np.mean([value["penalty_audit"]["penalty_to_xy_scale_ratio"] for value in members]))
        summarized.append(result)
    contrasts = []
    for shape in protocol["shapes"]:
        for sigma in protocol["measurement_noise"]:
            for depth in protocol["depths"]:
                prefix = (tuple(shape["state_counts"]), shape["segments"], sigma, depth)
                maps = {arm: {value["seed"]: value for value in groups[prefix + (arm,)]} for arm in protocol["arms"]}
                for left, right in (("h_x_penalty", "w_xy"), ("w_x_penalty", "w_xy"), ("h_x_penalty", "w_x_penalty")):
                    differences = []
                    for seed in protocol["instance_seeds"]:
                        pair = [float(np.mean([trial["failure_penalized_normalized_gap"] for trial in maps[arm][seed]["trials"] if trial["shots"] == 2048])) for arm in (left, right)]
                        differences.append(pair[0] - pair[1])
                    contrasts.append(dict(**shape, measurement_noise=sigma, depth=depth, contrast=f"{left}_minus_{right}",
                                          per_seed=differences, **bootstrap(differences, stable_seed(protocol["bootstrap_seed"], prefix, left, right), protocol["bootstrap_replicates"])))
    inputs = [read(path) for path in sorted((out / "instances").glob("*.json"))]
    opts = [read(path) for path in sorted((out / "optimizations").glob("*.json"))]
    if len(inputs) != protocol["expected_instances"] or len(opts) != protocol["expected_new_optimizations"]:
        raise ValueError("Input or optimization matrix incomplete")
    if any(value["evaluations"] != 1024 or value["status"] != "completed_fixed_budget" for value in opts):
        raise ValueError("Training budget incomplete")
    reused = [value["training"] for value in rows if value["arm"] == "w_xy"]
    timing = dict(input_preparation_seconds=sum(value["preparation_seconds"] for value in inputs),
                  new_training_seconds=sum(value["runtime_seconds"] for value in opts),
                  new_training_evaluations=sum(value["evaluations"] for value in opts),
                  reused_xy_training_seconds=sum(value["runtime_seconds"] for value in reused),
                  reused_xy_training_evaluations=sum(value["evaluations"] for value in reused),
                  compilation_seconds=sum(value["resources"]["compilation_seconds"] for value in rows),
                  independent_circuit_audit_seconds=sum(value["resources"]["audit_seconds"] for value in rows),
                  sampling_seconds=sum(value["sampling_seconds"] for value in rows))
    audit = dict(max_circuit_probability_error=max(value["resources"]["audit_max_probability_error"] for value in rows),
                 minimum_invalid_minus_worst_feasible=min(value["penalty_audit"]["invalid_minus_worst_feasible"] for value in inputs),
                 max_feasible_objective_error=max(value["penalty_audit"]["feasible_objective_max_absolute_error"] for value in inputs),
                 max_augmented_qubo_error=max(value["penalty_audit"]["augmented_qubo_max_absolute_error"] for value in inputs))
    summary = dict(status="complete", protocol_sha256=sha(out / "protocol.json"), instances=len(inputs),
                   new_optimizations=len(opts), reused_xy_optimizations=len(reused), distributions=len(rows), sampling_trials=len(flat),
                   groups=summarized, paired_contrasts=contrasts, timing=timing, audit=audit)
    dump_new(out / "summary.json", summary)
    with (out / "trials.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = ["# Stage D: ideal mixer/architecture and initialization controls", "", "The frozen three-arm comparison is complete. This is a local simulation, not new IBM evidence or a quantum-advantage test.", "",
             "All five Stage B shapes at <=12 qubits, eight base seeds and all three measurement-noise levels are included. Prior Stage B results had been seen when this additive design was frozen. X arms were subsequently trained with equal objective-call budgets; W/XY training is reused and fully charged in the resource accounting.", "",
             "## Results at 1% measurement noise and 512 shots", "",
             "Entries are means across eight instances, with 32 shot-repeat outcomes averaged within each instance. P(opt) and feasibility include all physical shots. Category accuracy assigns zero to abstentions. CX counts and depths are compiled all-to-all resources, not hardware routing forecasts.", "",
             "| Shape | p | Arm | Feasible (%) | P(opt) (%) | Hit (%) | Category (%) | CX | Depth |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for value in summarized:
        if value["measurement_noise"] == 0.01:
            shots = value["shot_results"]["512"]
            lines.append(f"| {value['state_counts']}, K={value['segments']} | {value['depth']} | {value['arm']} | {100*value['feasible_probability']['mean']:.2f} | {100*value['optimum_probability']['mean']:.4f} | {100*shots['hit_optimum']['mean']:.2f} | {100*shots['truth_category_accuracy']['mean']:.2f} | {value['mean_compiled_cx']:.1f} | {value['mean_compiled_depth']:.1f} |")
    lines += ["", "## Interpretation and limits", "",
              "The W/XY versus H/X comparison changes initialization, feasibility preservation, objective penalties and phase scaling; it is not a pure mixer comparison. W/X holds initialization fixed relative to W/XY, but still changes the mixer, penalty and normalization. H/X versus W/X isolates initialization within that architecture.", "",
              "The coefficient-only barrier is a conservative sufficient bound, not a tuned best-performing penalty baseline. It makes every invalid state energetically worse than every feasible state, yet finite-depth circuits may still emit invalid shots. Its larger normalization suppresses feasible objective phase differences. Equal p and equal expectation-call budgets do not imply equal gate count, classical training time or optimal parameter quality.", "",
              "All outcomes are retained. An all-invalid trial abstains: hit=0, accuracy=0 and separately named failure-penalized normalized gap=1; there is no classical recovery or insertion of an unobserved state. The exact feasible objective can disagree with planted truth, as already shown in Stage B.", "",
              "Paired 2,048-shot contrasts and 95% percentile bootstrap intervals over eight base seeds are in summary.json. They are exploratory descriptive comparisons without multiplicity-adjusted significance or population-generalization claims. All noise variants remain paired; shot repeats are not extra independent instances.", "", "## Recorded computation and checks", "", "```json", json.dumps(dict(timing=timing, audit=audit), indent=2), "```", "",
              "Reused XY training time is previous computation, not newly elapsed time. Component times exclude startup, I/O and summary/bootstrap generation; none are QPU times or speedup evidence.", "", "## Provenance", "",
              f"- [Frozen protocol](protocol.json), SHA-256 `{summary['protocol_sha256']}`.",
              "- [Complete summary, every condition and paired contrast](summary.json).", "- [Every sampled-trial metric](trials.csv).",
              "- [Per-input penalty certificates](instances/), [new optimization histories and full binary probabilities](optimizations/), [arm distributions and compiled-resource audits](distributions/).",
              "- Stage B input/optimization/distribution paths and SHA-256 hashes are frozen individually in protocol.json; the original archives are unchanged.", ""]
    with (out / "report.md").open("x") as stream:
        stream.write("\n".join(lines))
    print(json.dumps({key: summary[key] for key in ("status", "instances", "new_optimizations", "reused_xy_optimizations", "distributions", "sampling_trials", "timing", "audit")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["prepare", "run", "summarize"])
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage_d/mixer/run_001")
    parser.add_argument("--stage-b", type=Path, default=ROOT / "results/stage_b/run_002")
    args = parser.parse_args()
    out, stage_b = args.output.resolve(), args.stage_b.resolve()
    if args.mode == "prepare":
        prepare(out, stage_b)
    elif args.mode == "run":
        run(out)
    else:
        summarize(out)


if __name__ == "__main__":
    main()
