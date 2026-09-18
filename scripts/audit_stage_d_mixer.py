#!/usr/bin/env python3
"""Independent full-space/finite-shot verification of the Stage D mixer archive.

Only independent audit utilities are reused; no scored experiment helpers are
imported. The auditor reconstructs augmented objectives and selected X/XY
probabilities, resamples every finite-shot trial, and recomputes uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import time
from datetime import datetime, timezone

import numpy as np

from audit_stage_b import (close, require, digest, seed_from_parts, enumerate_direct,
                           direct_ideal_probabilities, check_bootstrap)

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("w_xy", "h_x_penalty", "w_x_penalty")
CONTRASTS = (("h_x_penalty", "w_xy"), ("w_x_penalty", "w_xy"), ("h_x_penalty", "w_x_penalty"))
METRICS = ("hit_optimum", "no_feasible", "failure_penalized_normalized_gap",
           "truth_category_accuracy", "truth_bit_accuracy")


def augmented_objective(raw):
    """Independently construct the feasible-equivalent extension and barrier."""
    states, exact, indices, base_scale = enumerate_direct(raw)
    counts, intervals = tuple(raw["state_counts"]), raw["segments"]
    registers = counts*intervals
    offsets = np.r_[0, np.cumsum(registers)[:-1]].astype(int)
    levels = [np.asarray(x) for x in raw["levels"]]
    n = raw["qubits"]
    bits = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(float)
    linear = np.zeros(n)
    quadratic = {}
    constant = 0.
    for t, (measurement, weight) in enumerate(zip(raw["aggregate"], raw["segment_weights"])):
        constant += weight*measurement**2
        for channel, values in enumerate(levels):
            start = offsets[t*len(counts)+channel]
            linear[start:start+len(values)] += weight*(values**2-2*measurement*values)
            for other in range(channel+1, len(counts)):
                other_start = offsets[t*len(counts)+other]
                for a, power_a in enumerate(values):
                    for b, power_b in enumerate(levels[other]):
                        coefficient = 2*weight*power_a*power_b
                        if coefficient:
                            quadratic[start+a, other_start+b] = coefficient
    for t in range(1, intervals):
        for channel, penalty in enumerate(raw["switch_penalty"]):
            constant += penalty
            before, after = offsets[(t-1)*len(counts)+channel], offsets[t*len(counts)+channel]
            for category in range(counts[channel]):
                if penalty:
                    quadratic[before+category, after+category] = -penalty
    s = float(np.abs(linear).sum()+sum(abs(x) for x in quadratic.values()))
    close(base_scale, max(1., s), "Base coefficient-only scale", atol=1e-7)
    original = constant+bits@linear
    for (left, right), coefficient in quadratic.items():
        original += coefficient*bits[:, left]*bits[:, right]
    barrier = 1.+2*s
    violations = np.zeros(len(bits))
    for offset, width in zip(offsets, registers):
        violations += (bits[:, offset:offset+width].sum(axis=1)-1)**2
        constant += barrier
        linear[offset:offset+width] -= barrier
        for left in range(offset, offset+width):
            for right in range(left+1, offset+width):
                quadratic[left, right] = quadratic.get((left, right), 0.)+2*barrier
    scale = max(1., float(np.abs(linear).sum()+sum(abs(x) for x in quadratic.values())))
    penalized = original+barrier*violations
    expanded = constant+bits@linear
    for (left, right), coefficient in quadratic.items():
        expanded += coefficient*bits[:, left]*bits[:, right]
    tolerance = 128*np.finfo(float).eps*max(1., abs(constant)+scale)
    close(penalized, expanded, "Augmented full-binary expansion", atol=tolerance, rtol=0)
    close(penalized[indices], exact, "Feasible objective equivalence", atol=tolerance, rtol=0)
    require(set(np.flatnonzero(violations == 0)) == set(indices), "Independent one-hot feasibility")
    require(penalized[violations > 0].min() > penalized[indices].max(), "Penalty fails strict feasibility dominance")
    certificate = dict(barrier=barrier, base_coefficient_l1=s,
        invalid_minimum=float(penalized[violations > 0].min()),
        feasible_maximum=float(penalized[indices].max()), feasible_minimum=float(penalized[indices].min()),
        invalid_minus_best_feasible=float(penalized[violations > 0].min()-penalized[indices].min()),
        invalid_minus_worst_feasible=float(penalized[violations > 0].min()-penalized[indices].max()),
        xy_coefficient_scale=base_scale, penalty_coefficient_scale=scale,
        penalty_to_xy_scale_ratio=scale/base_scale,
        xy_feasible_phase_range_per_unit_gamma=float(np.ptp(exact)/base_scale),
        penalty_feasible_phase_range_per_unit_gamma=float(np.ptp(exact)/scale))
    return states, exact, indices, base_scale, penalized, scale, certificate


def independent_x_probabilities(energies, scale, indices, gammas, betas, initialization):
    n = int(math.log2(len(energies)))
    if initialization == "h":
        vector = np.ones(len(energies), dtype=complex)/math.sqrt(len(energies))
    else:
        vector = np.zeros(len(energies), dtype=complex)
        vector[indices] = 1/math.sqrt(len(indices))
    # Bit-index swapping rather than the production reshape/block implementation.
    basis = np.arange(len(vector))
    for gamma, beta in zip(gammas, betas):
        vector *= np.exp(-1j*gamma*energies/scale)
        for qubit in range(n):
            vector = math.cos(beta)*vector-1j*math.sin(beta)*vector[basis^(1 << qubit)]
    p = vector.real**2+vector.imag**2
    close(p.sum(), 1., "Independent X norm", atol=1e-12)
    return p/p.sum()


def audit(archive):
    started = time.perf_counter()
    hashes = {}

    def read(path):
        path = Path(path)
        hashes[str(path.relative_to(ROOT))] = digest(path)
        return json.loads(path.read_text())

    protocol, summary = read(archive/"protocol.json"), read(archive/"summary.json")
    protocol_sha = digest(archive/"protocol.json")
    require(summary["status"] == "complete" and summary["protocol_sha256"] == protocol_sha, "Incomplete/protocol mismatch")
    require(protocol["arms"] == list(ARMS), "Architecture-arm mismatch")
    for field in ("source_sha256", "stage_b_source_sha256"):
        for path, expected in protocol[field].items():
            require(digest(ROOT/path) == expected, f"Frozen source changed: {path}")
    parent = ROOT/protocol["stage_b_root"]
    require(digest(parent/"protocol.json") == protocol["stage_b_protocol_sha256"] and
            digest(parent/"summary.json") == protocol["stage_b_summary_sha256"], "Stage B provenance changed")
    expected_inputs = {x["id"] for x in protocol["selected_inputs"]}
    expected_rows = {f"{iid}_p{p}_{arm}" for iid in expected_inputs for p in protocol["depths"] for arm in ARMS}
    expected_opts = {x for x in expected_rows if not x.endswith("_w_xy")}
    for folder, expected in (("instances", expected_inputs), ("distributions", expected_rows), ("optimizations", expected_opts)):
        require({x.stem for x in (archive/folder).glob("*.json")} == expected, f"{folder}: exact matrix coverage")
    require(len(expected_inputs) == summary["instances"] == 120 and len(expected_opts) == summary["new_optimizations"] == 480 and len(expected_rows) == summary["distributions"] == 720, "Coverage totals")
    require(len(protocol["preflight_controls"]) == 8, "Preflight count")
    for check in protocol["preflight_controls"]:
        require(check["circuit_audit"]["audit_max_probability_error"] <= 1e-10 and check["penalty_audit"]["status"] == "passed", "Preflight failed")
    instances, certificates, opts, data = {}, {}, {}, {}
    source_by_id = {x["id"]:x for x in protocol["selected_inputs"]}
    max_rebuilt_error = 0.
    history_entries = 0
    for iid in sorted(expected_inputs):
        raw = read(ROOT/source_by_id[iid]["instance_path"])
        archived = read(archive/"instances"/(iid+".json"))
        require(archived["instance_id"] == iid and archived["protocol_sha256"] == protocol_sha, "Input identity")
        require(archived["stage_b_input_sha256"] == digest(ROOT/source_by_id[iid]["instance_path"]), "Input source hash")
        computed = augmented_objective(raw)
        for field, expected in computed[-1].items():
            close(archived["penalty_audit"][field], expected, f"{iid}: penalty certificate/{field}", atol=1e-4, rtol=1e-11)
        require(archived["penalty_audit"]["status"] == "passed" and archived["penalty_audit"]["barrier_uses_exact_or_truth"] is False, "Penalty audit scope")
        data[iid], instances[iid], certificates[iid] = computed, raw, archived
    for oid in sorted(expected_opts):
        record = read(archive/"optimizations"/(oid+".json"))
        iid, arm, depth = record["instance_id"], record["arm"], record["depth"]
        require(oid == f"{iid}_p{depth}_{arm}" and record["protocol_sha256"] == protocol_sha, "Optimizer identity")
        states, exact, indices, base_scale, energy, scale, certificate = data[iid]
        initialization = "h" if arm == "h_x_penalty" else "w"
        require(record["initialization"] == initialization and len(record["gammas"]) == len(record["betas"]) == depth, "Optimizer circuit definition")
        require(record["evaluations"] == 1024 and record["status"] == "completed_fixed_budget", "Optimizer budget/status")
        require([x["seed"] for x in record["restarts"]] == protocol["optimizer_restart_seeds"], "Restart seeds")
        for restart in record["restarts"]:
            history = np.asarray(restart["objective_history"])
            require(len(history) == restart["evaluations"] == 512 and np.all(np.isfinite(history)), "Restart history coverage")
            require(restart["best_evaluation_index"] == int(np.argmin(history)), "Best counted evaluation")
            close(restart["expected_cost"], history.min()*scale, "Best restart expectation", atol=1e-5)
            close(restart["normalized_expected_cost"], history.min(), "Best normalized expectation")
            zero = energy.mean() if initialization == "h" else energy[indices].mean()
            close(history[0], zero/scale, "First candidate initialization")
            require(restart["initial_evaluations"] == 32 and sum(x["evaluations"]+x["padding_evaluations"] for x in restart["refinements"])+32 == 512, "Call ledger")
            for refinement in restart["refinements"]:
                require(refinement["evaluations"]+refinement["padding_evaluations"] == refinement["allowance"], "Powell allotment")
            require(restart["runtime_seconds"] >= restart["statevector_seconds"] >= 0, "Optimizer timing")
        selected = int(np.argmin([x["normalized_expected_cost"] for x in record["restarts"]]))
        require(record["chosen_restart_index"] == selected and record["chosen_seed"] == record["restarts"][selected]["seed"], "Restart selection")
        require(record["gammas"] == record["restarts"][selected]["gammas"] and record["betas"] == record["restarts"][selected]["betas"], "Selected angles")
        for angles, bounds in ((record["gammas"], protocol["gamma_bounds"]), (record["betas"], protocol["beta_bounds"])):
            require(all(bounds[0] <= x <= bounds[1] and math.isfinite(x) for x in angles), "Selected angle bounds")
        rebuilt = independent_x_probabilities(energy, scale, indices, record["gammas"], record["betas"], initialization)
        full = np.asarray(record["probabilities"])
        close(full, rebuilt, f"{oid}: independently reconstructed full X distribution", atol=1e-11, rtol=1e-9)
        max_rebuilt_error = max(max_rebuilt_error, float(np.max(np.abs(full-rebuilt))))
        close(record["expected_cost"], float(full@energy), "Full penalized expectation", atol=1e-5)
        close(record["normalized_expected_cost"], float(full@energy)/scale, "Normalized full expectation")
        require(record["runtime_seconds"] >= record["statevector_seconds"] >= 0, "Optimizer total timing")
        opts[oid] = record
        history_entries += record["evaluations"]

    records, groups = {}, {}
    trial_count = draw_count = 0
    for did in sorted(expected_rows):
        row = read(archive/"distributions"/(did+".json"))
        iid, arm, depth = row["instance_id"], row["arm"], row["depth"]
        require(row["distribution_id"] == did == f"{iid}_p{depth}_{arm}" and row["protocol_sha256"] == protocol_sha, "Distribution identity")
        states, energies, indices, base_scale, full_energies, penalty_scale, certificate = data[iid]
        raw = instances[iid]
        for field in ("state_counts", "segments", "seed", "measurement_noise", "qubits", "feasible_states"):
            require(row[field] == raw[field], f"{did}: source input field/{field}")
        training = row["training"]
        require(digest(ROOT/training["source_path"]) == training["source_sha256"], "Training provenance")
        if arm == "w_xy":
            previous = read(ROOT/source_by_id[iid]["xy"][str(depth)]["optimization_path"])
            previous_distribution = read(ROOT/source_by_id[iid]["xy"][str(depth)]["distribution_path"])
            require(training["reused"] is True and digest(ROOT/training["distribution_source_path"]) == training["distribution_source_sha256"], "Reused XY provenance")
            p = direct_ideal_probabilities(states, energies, base_scale, raw["state_counts"], row["gammas"], row["betas"])
            close(row["probabilities"], previous_distribution["probabilities"], "Reused XY distribution")
            full_expected_cost = previous["expected_cost"]
        else:
            previous = opts[did]
            require(training["reused"] is False, "New optimization labelled reused")
            p = np.asarray(previous["probabilities"])[indices]
            full_expected_cost = previous["expected_cost"]
        require(row["gammas"] == previous["gammas"] and row["betas"] == previous["betas"], "Frozen selected circuit angles")
        require(training["evaluations"] == 1024 and training["runtime_seconds"] == previous["runtime_seconds"], "Training-cost accounting")
        close(row["probabilities"], p, "All-shot feasible distribution", atol=1e-11, rtol=1e-9)
        close(row["full_training_expected_cost"], full_expected_cost, "Training expectation retained", atol=1e-5)
        p = np.asarray(row["probabilities"], dtype=float)
        require(p.shape == energies.shape and np.all(np.isfinite(p)) and np.all(p >= 0), "Feasible probability bounds")
        feasible = float(p.sum())
        require(feasible <= 1+1e-10, "Feasibility mass >1")
        optimum, span = float(energies.min()), float(np.ptp(energies))
        tolerance = 1e-10*max(1., span)
        minimizers = np.where(energies <= optimum+tolerance)[0]
        canonical = states[minimizers[0]]
        truth = np.asarray(raw["truth"])
        popt = min(1., float(p[minimizers].sum()))
        conditional = float(p@energies/feasible) if feasible else None
        expected_endpoints = dict(feasible_probability=min(1., feasible), invalid_probability=max(0., 1-feasible),
            optimum_probability=popt, exact_cost=optimum, energy_span=span, optimum_tolerance=tolerance,
            conditional_expected_cost=conditional,
            conditional_expected_normalized_gap=(conditional-optimum)/span if feasible and span else 0. if feasible else None,
            exact_truth_category_accuracy=float(np.mean(canonical == truth)),
            exact_truth_bit_accuracy=1-2*np.count_nonzero(canonical != truth)/raw["qubits"])
        for field, expected in expected_endpoints.items():
            close(row[field], expected, f"{did}: endpoint/{field}", atol=1e-7 if field in ("exact_cost", "energy_span", "conditional_expected_cost") else 1e-10)
        require(row["canonical_exact_index"] == minimizers[0] and row["optimum_count"] == len(minimizers), "Exact minimizer set")
        n99 = None if popt == 0 else 1 if popt == 1 else math.ceil(math.log(.01)/math.log1p(-popt))
        require(row["shots_to_optimum_99"] == n99, "Analytic shots99")
        for field, expected in certificate.items():
            close(row["penalty_audit"][field], expected, "Retained penalty certificate", atol=1e-4)
        resource = row["resources"]
        require(resource["num_qubits"] == raw["qubits"] and resource["audit_max_probability_error"] <= 1e-10 and
                resource["audit_probability_mass_error"] <= 1e-10 and resource["basis_gates"] == ["rz", "sx", "x", "cx"] and
                resource["optimization_level"] == 1 and resource["transpile_seed"] == 17, "Circuit audit or compilation contract")
        require(all(isinstance(n, int) and n >= 0 for n in resource["compiled_gate_counts"].values()), "Invalid gate counts")
        distribution = np.r_[p, max(0., 1-feasible)]
        distribution /= distribution.sum()
        expected_keys = set(itertools.product(protocol["shots"], range(protocol["sampling_repeats"])))
        require(len(row["trials"]) == len(expected_keys) and {(t["shots"], t["repeat"]) for t in row["trials"]} == expected_keys, "Trial coverage")
        for trial in row["trials"]:
            shots, repeat = trial["shots"], trial["repeat"]
            seed = seed_from_parts(iid, shots, repeat, "stage-b-measurements")
            require(trial["sample_seed"] == seed and trial["distribution_id"] == did, "Matched sampling seed")
            draws = np.random.default_rng(seed).choice(len(distribution), shots, p=distribution)
            valid = draws[draws < len(p)]
            candidates = sorted(set(valid.tolist()))
            best = min(candidates, key=lambda x:(energies[x], x)) if candidates else None
            gap = float(energies[best]-optimum) if best is not None else None
            normalized = gap/span if gap is not None and span else 0. if gap is not None else None
            expected = dict(feasible_shots=len(valid), no_feasible=int(best is None), best_index=best,
                best_cost=float(energies[best]) if best is not None else None, certified_gap=gap, normalized_gap=normalized,
                failure_penalized_normalized_gap=1. if normalized is None else normalized,
                hit_optimum=int(gap is not None and gap <= tolerance),
                truth_category_accuracy=float(np.mean(states[best] == truth)) if best is not None else 0.,
                truth_bit_accuracy=1-2*np.count_nonzero(states[best] != truth)/raw["qubits"] if best is not None else 0.,
                canonical_exact_category_accuracy=float(np.mean(states[best] == canonical)) if best is not None else 0.,
                canonical_exact_bit_accuracy=1-2*np.count_nonzero(states[best] != canonical)/raw["qubits"] if best is not None else 0.)
            for field, value in expected.items():
                if field in ("feasible_shots", "no_feasible", "best_index", "hit_optimum"):
                    require(trial[field] == value, f"{did}: sampled {field}")
                else:
                    close(trial[field], value, f"{did}: sampled {field}", atol=1e-7 if field in ("best_cost", "certified_gap") else 1e-10)
            trial_count += 1
            draw_count += shots
        key = (tuple(raw["state_counts"]), raw["segments"], raw["measurement_noise"], depth, arm)
        groups.setdefault(key, []).append(row)
        records[did] = row
    require(trial_count == summary["sampling_trials"] == 92160, "Total trial count")
    require(len(summary["groups"]) == len(groups) == 90, "Group coverage")
    bootstrap_count = 0
    keys_seen = set()
    for result in summary["groups"]:
        key = (tuple(result["state_counts"]), result["segments"], result["measurement_noise"], result["depth"], result["arm"])
        require(key not in keys_seen and key in groups, "Duplicate/missing group key")
        keys_seen.add(key)
        members = groups[key]
        require([x["seed"] for x in members] == protocol["instance_seeds"], "Eight ordered base seeds")
        for field in ("feasible_probability", "optimum_probability", "conditional_expected_normalized_gap", "exact_truth_category_accuracy"):
            check_bootstrap(result[field], [x[field] for x in members], seed_from_parts(protocol["bootstrap_seed"], key, field), protocol["bootstrap_replicates"], f"{key}/{field}")
            bootstrap_count += 1
        for shots in protocol["shots"]:
            for field in METRICS:
                means = [np.mean([t[field] for t in x["trials"] if t["shots"] == shots]) for x in members]
                check_bootstrap(result["shot_results"][str(shots)][field], means, seed_from_parts(protocol["bootstrap_seed"], key, shots, field), protocol["bootstrap_replicates"], f"{key}/{shots}/{field}")
                bootstrap_count += 1
        for field, values in (("mean_compiled_depth", [x["resources"]["compiled_depth"] for x in members]),
                               ("mean_compiled_cx", [x["resources"]["compiled_gate_counts"].get("cx", 0) for x in members]),
                               ("mean_penalty_to_xy_scale_ratio", [x["penalty_audit"]["penalty_to_xy_scale_ratio"] for x in members])):
            close(result[field], np.mean(values), f"{key}: resource summary/{field}")
    expected_contrasts = {(tuple(s["state_counts"]), s["segments"], sigma, depth, f"{left}_minus_{right}")
                          for s, sigma, depth, (left,right) in itertools.product(protocol["shapes"], protocol["measurement_noise"], protocol["depths"], CONTRASTS)}
    actual_contrasts = set()
    for result in summary["paired_contrasts"]:
        prefix = (tuple(result["state_counts"]), result["segments"], result["measurement_noise"], result["depth"])
        key = prefix+(result["contrast"],)
        require(key in expected_contrasts and key not in actual_contrasts, "Contrast coverage")
        actual_contrasts.add(key)
        left, right = next(pair for pair in CONTRASTS if result["contrast"] == f"{pair[0]}_minus_{pair[1]}")
        maps = [{r["seed"]:r for r in groups[prefix+(arm,)]} for arm in (left, right)]
        differences = [float(np.mean([t["failure_penalized_normalized_gap"] for t in maps[0][seed]["trials"] if t["shots"] == 2048])-
                             np.mean([t["failure_penalized_normalized_gap"] for t in maps[1][seed]["trials"] if t["shots"] == 2048])) for seed in protocol["instance_seeds"]]
        close(result["per_seed"], differences, "Paired seed differences")
        check_bootstrap(result, differences, seed_from_parts(protocol["bootstrap_seed"], prefix, left, right), protocol["bootstrap_replicates"], str(key))
        bootstrap_count += 1
    require(actual_contrasts == expected_contrasts, "Missing contrasts")
    reused = [r["training"] for r in records.values() if r["arm"] == "w_xy"]
    expected_timing = dict(input_preparation_seconds=sum(x["preparation_seconds"] for x in certificates.values()),
        new_training_seconds=sum(x["runtime_seconds"] for x in opts.values()), new_training_evaluations=history_entries,
        reused_xy_training_seconds=sum(x["runtime_seconds"] for x in reused), reused_xy_training_evaluations=sum(x["evaluations"] for x in reused),
        compilation_seconds=sum(x["resources"]["compilation_seconds"] for x in records.values()),
        independent_circuit_audit_seconds=sum(x["resources"]["audit_seconds"] for x in records.values()),
        sampling_seconds=sum(x["sampling_seconds"] for x in records.values()))
    for field, value in expected_timing.items():
        close(summary["timing"][field], value, "Component timing ledger")
    require(len(reused) == summary["reused_xy_optimizations"] == 240, "Reused XY count")
    csv_keys = set()
    with (archive/"trials.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["distribution_id"], int(row["shots"]), int(row["repeat"]))
            require(key not in csv_keys and key[0] in records, "CSV trial identity")
            csv_keys.add(key)
            original = records[key[0]]
            trial = next(t for t in original["trials"] if (t["shots"], t["repeat"]) == key[1:])
            expected = {k:original[k] for k in ("instance_id", "arm", "depth", "qubits", "seed", "measurement_noise")}
            expected.update(trial)
            require(set(row) == set(expected), "CSV field coverage")
            for field, value in expected.items():
                if value is None:
                    require(row[field] == "", "CSV missing sentinel")
                elif isinstance(value, str):
                    require(row[field] == value, "CSV string value")
                elif isinstance(value, int):
                    require(int(row[field]) == value, "CSV integer value")
                else:
                    close(float(row[field]), value, "CSV numerical value", atol=1e-7)
    require(len(csv_keys) == trial_count, "CSV row total")
    hashes[str((archive/"trials.csv").relative_to(ROOT))] = digest(archive/"trials.csv")
    return dict(status="passed_independent_full_mixer_audit", audited_utc=datetime.now(timezone.utc).isoformat(),
                audit_script_sha256=digest(Path(__file__)), independent_utility_sha256=digest(ROOT/"scripts/audit_stage_b.py"),
                protocol_sha256=protocol_sha, inputs_checked=120, full_binary_penalty_certificates_checked=120,
                new_optimizations_checked=480, new_restart_histories_checked=960,
                new_counted_objective_evaluations_checked=history_entries, reused_xy_optimizations_checked=240,
                independently_reconstructed_selected_circuits=720, maximum_independent_full_x_probability_error=max_rebuilt_error,
                distributions_checked=720, regenerated_trials=trial_count, regenerated_draws=draw_count,
                grouped_summaries_checked=90, paired_contrasts_checked=len(actual_contrasts),
                bootstrap_intervals_recomputed=bootstrap_count, csv_rows_checked=len(csv_keys),
                source_sha256=hashes, timing=expected_timing,
                scope="Independent augmented-cost expansion and dominance proofs, full X and feasible XY selected-circuit probabilities, all sampling outcomes, endpoint metrics, grouped and paired CIs, training/timing and source provenance; compiled gate-count metadata checked but compilation not rerun",
                limitations="Conservative fixed penalty and differing phase scales; architecture comparison rather than isolated mixer effect; bounded ideal synthetic follow-up, not quantum advantage",
                runtime_seconds=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT/"results/stage_d/mixer/run_001")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = audit(args.archive.resolve())
    if not args.check_only:
        with (args.archive/"independent_audit.json").open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
