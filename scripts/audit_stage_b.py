#!/usr/bin/env python3
"""Independent, read-only verification of a completed categorical Stage B run.

Imports no experiment helpers. Objectives, ideal distributions, one-hot
encodings, finite-shot selections, and bootstrap summaries are reconstructed
here from the immutable archive. The optional audit receipt is create-new.
No hardware service is contacted, and no scored artifact is modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRIAL_METRICS = (
    "hit_optimum", "no_feasible", "failure_penalized_normalized_gap",
    "truth_category_accuracy", "truth_bit_accuracy",
)
GROUP_METRICS = (
    "feasible_probability", "optimum_probability",
    "conditional_expected_normalized_gap", "exact_truth_category_accuracy",
    "exact_truth_bit_accuracy",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name, atol=1e-10, rtol=1e-10):
    if expected is None:
        require(actual is None, f"{name}: expected missing, got {actual}")
        return
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b)),
            f"{name}: nonfinite values or incompatible shapes")
    require(np.all(np.abs(a-b) <= atol + rtol*np.abs(b)),
            f"{name}: numerical mismatch (max {np.max(np.abs(a-b))})")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_from_parts(*parts):
    # This is the archived seed contract, independently implemented here.
    message = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(message).digest()[:8], byteorder="little")


def instance_name(counts, segments, seed, sigma):
    return f"m{'-'.join(str(c) for c in counts)}_k{segments}_s{seed}_n{sigma:g}"


def enumerate_direct(instance):
    """Cartesian enumeration and original NILM objective, with no QUBO helper."""
    counts = tuple(instance["state_counts"])
    intervals = instance["segments"]
    registers = counts * intervals
    combinations = itertools.product(*(range(n) for n in registers[::-1]))
    states = np.asarray([row[::-1] for row in combinations], dtype=np.int64)
    states = states.reshape(-1, intervals, len(counts))
    levels = [np.asarray(x, dtype=float) for x in instance["levels"]]
    aggregate = np.asarray(instance["aggregate"], dtype=float)
    weights = np.asarray(instance["segment_weights"], dtype=float)
    penalties = np.asarray(instance["switch_penalty"], dtype=float)
    require(len(levels) == len(counts) and all(len(v) == n for v, n in zip(levels, counts)),
            "Level/register mismatch")
    require(aggregate.shape == weights.shape == (intervals,) and np.all(weights > 0),
            "Invalid aggregate or duration weights")
    require(penalties.shape == (len(counts),) and np.all(penalties >= 0),
            "Invalid transition penalties")
    prediction = np.zeros((len(states), intervals))
    for channel, values in enumerate(levels):
        prediction += values[states[:, :, channel]]
    energies = np.einsum("st,t->s", (prediction-aggregate)**2, weights)
    if intervals > 1:
        energies += np.sum((states[:, 1:, :] != states[:, :-1, :])*penalties, axis=(1, 2))

    # Independently reconstruct the coefficient-only normalization scale.
    coefficient_sum = 0.0
    for measurement, weight in zip(aggregate, weights):
        for channel, values in enumerate(levels):
            coefficient_sum += np.abs(weight*(values*values-2*measurement*values)).sum()
            for other in range(channel+1, len(levels)):
                coefficient_sum += np.abs(2*weight*np.outer(values, levels[other])).sum()
    coefficient_sum += (intervals-1)*sum(n*p for n, p in zip(counts, penalties))
    scale = max(1.0, float(coefficient_sum))
    close(instance["scale"], scale, "coefficient normalization", atol=1e-7)
    require(instance["qubits"] == sum(registers), "Physical width mismatch")
    require(instance["feasible_states"] == len(states), "Feasible state count mismatch")

    physical = np.zeros(len(states), dtype=np.int64)
    offset = 0
    for register, width in enumerate(registers):
        categories = states.reshape(len(states), -1)[:, register]
        physical += np.left_shift(np.int64(1), offset + categories)
        offset += width
    require(len(np.unique(physical)) == len(states), "One-hot encoding collision")
    truth = np.asarray(instance["truth"], dtype=np.int64)
    require(truth.shape == states.shape[1:], "Planted truth shape mismatch")
    for channel, width in enumerate(counts):
        require(np.all((truth[:, channel] >= 0) & (truth[:, channel] < width)),
                "Planted category out of range")
    return states, energies, physical, scale


def direct_ideal_probabilities(states, energies, scale, counts, gammas, betas):
    """Independent pair-index circuit application in categorical basis order."""
    flat_states = states.reshape(len(states), -1)
    radices = tuple(counts)*states.shape[1]
    pair_indices = []
    place = 1
    for register, width in enumerate(radices):
        for category in range(width-1):
            left = np.flatnonzero(flat_states[:, register] == category)
            pair_indices.append((left, left+place))
        place *= width
    vector = np.ones(len(states), dtype=np.complex128)/math.sqrt(len(states))
    for gamma, beta in zip(gammas, betas):
        vector *= np.exp(-1j*gamma*energies/scale)
        cosine, sine = math.cos(beta), -1j*math.sin(beta)
        for left, right in pair_indices:
            a, b = vector[left].copy(), vector[right].copy()
            vector[left] = cosine*a+sine*b
            vector[right] = sine*a+cosine*b
    probabilities = vector.real**2+vector.imag**2
    close(probabilities.sum(), 1.0, "independent ideal norm", atol=1e-12)
    return probabilities/probabilities.sum()


def independently_bootstrap(values, seed, count):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and len(values) == 8 and np.all(np.isfinite(values)),
            "Bootstrap must contain eight finite independent instance summaries")
    indices = np.random.default_rng(seed).integers(low=0, high=len(values),
                                                size=(count, len(values)))
    sample_means = np.sum(values[indices], axis=1)/len(values)
    bounds = np.percentile(sample_means, [2.5, 97.5])
    return {"mean": float(np.sum(values)/len(values)),
            "ci95": bounds.tolist(), "n_instances": len(values)}


def check_bootstrap(actual, values, seed, count, label):
    expected = independently_bootstrap(values, seed, count)
    require(actual["n_instances"] == expected["n_instances"], f"{label}: cluster count")
    close(actual["mean"], expected["mean"], f"{label}: mean")
    close(actual["ci95"], expected["ci95"], f"{label}: paired percentile bounds")


def audit(archive):
    started = time.perf_counter()
    source_hashes = {}

    def read(path):
        path = Path(path)
        require(path.is_file(), f"Required artifact missing: {path}")
        source_hashes[str(path.relative_to(archive))] = digest(path)
        return json.loads(path.read_text())

    protocol = read(archive/"protocol.json")
    summary = read(archive/"summary.json")
    protocol_sha = source_hashes["protocol.json"]
    for path, expected in protocol["source_sha256"].items():
        require(digest(ROOT/path) == expected, f"Frozen source differs: {path}")
    predecessor = None
    if archive.name == "run_002":
        previous = archive.parent/"run_001"
        old_protocol = json.loads((previous/"protocol.json").read_text())
        old_failure = json.loads((previous/"failure.json").read_text())
        exempt = {"prepared_utc", "source_sha256"}
        require({k:v for k,v in old_protocol.items() if k not in exempt} ==
                {k:v for k,v in protocol.items() if k not in exempt},
                "Scientific protocol changed after the implementation abort")
        require(old_failure["status"] == "aborted_numerical_boundary_check" and
                old_failure["failed_instance_id"] == "m2-2_k1_s1006_n0.01" and
                old_failure["failed_depth"] == 2,
                "Original implementation abort is not retained as documented")
        for path, expected in old_protocol["source_sha256"].items():
            require(digest(previous/"frozen_source"/Path(path).name) == expected,
                    f"Prior frozen source not retained: {path}")
        changed = [path for path, value in protocol["source_sha256"].items()
                   if old_protocol["source_sha256"][path] != value]
        require(set(changed) == {"scripts/run_stage_b.py", "src/quantum_nilm/stage_b.py"},
                "Unexpected scored-source change after abort")
        require((previous/"frozen_source/run_stage_b.py").read_text().replace(
                    'default=ROOT/"results/stage_b/run_001"',
                    'default=ROOT/"results/stage_b/run_002"') ==
                (ROOT/"scripts/run_stage_b.py").read_text(),
                "Runner changed beyond the default output directory after abort")
        predecessor = dict(protocol_sha256=digest(previous/"protocol.json"),
                           failure_sha256=digest(previous/"failure.json"),
                           retained_status=old_failure["status"],
                           scientific_design_unchanged=True,
                           changed_source=changed)
    require(summary["status"] == "complete", "Run is not marked complete")
    require(summary["protocol_sha256"] == protocol_sha, "Summary/protocol provenance mismatch")
    expected_instances, expected_optimizers, expected_distributions = set(), set(), set()
    for shape, seed, sigma in itertools.product(protocol["shapes"],
                                                protocol["instance_seeds"],
                                                protocol["measurement_noise"]):
        name = instance_name(shape["state_counts"], shape["segments"], seed, sigma)
        expected_instances.add(name)
        expected_distributions.add(name+"_p0_uniform")
        for depth in protocol["depths"]:
            expected_optimizers.add(f"{name}_p{depth}")
            expected_distributions.add(f"{name}_p{depth}_ideal")
            if sum(shape["state_counts"])*shape["segments"] <= protocol["noise_max_qubits"]:
                for noise in protocol["noise_levels"]:
                    expected_distributions.add(f"{name}_p{depth}_{noise}")
    for folder, expected in (("instances", expected_instances),
                             ("optimizations", expected_optimizers),
                             ("distributions", expected_distributions)):
        actual = {p.stem for p in (archive/folder).glob("*.json")}
        require(actual == expected, f"{folder} matrix mismatch: missing {expected-actual}, extra {actual-expected}")
    require(len(expected_instances) == protocol["expected_instances"] == summary["instances"], "Instance count mismatch")
    require(len(expected_optimizers) == protocol["expected_optimizations"] == summary["optimizations"], "Optimizer count mismatch")
    require(len(expected_distributions) == protocol["expected_distributions"] == summary["distributions"], "Distribution count mismatch")

    instances, direct, paired_base = {}, {}, {}
    for iid in sorted(expected_instances):
        record = read(archive/"instances"/(iid+".json"))
        require(record["id"] == iid and record["protocol_sha256"] == protocol_sha, f"{iid}: identity/provenance")
        states, energies, physical, scale = enumerate_direct(record)
        metadata = record["metadata"]
        require(metadata["previous_states"] is None and metadata["clipped"] is False,
                f"{iid}: unexpected previous reference state or clipping")
        maxima = np.asarray(metadata["channel_maxima_watts"])
        for values, recorded in zip(record["levels"], metadata["levels_watts"]):
            close(values, recorded, f"{iid}: levels metadata")
        for values, width, maximum in zip(record["levels"], record["state_counts"], maxima):
            close(values, np.linspace(0, maximum, width), f"{iid}: evenly spaced levels")
        close(record["switch_penalty"], .02*maxima**2, f"{iid}: normalized penalty")
        noise = record["measurement_noise"]*maxima.max()*np.asarray(metadata["standard_normal_noise"])
        clean = sum(np.asarray(values)[np.asarray(record["truth"])[:, i]] for i, values in enumerate(record["levels"]))
        close(metadata["clean_aggregate_watts"], clean, f"{iid}: clean aggregate")
        close(record["aggregate"], clean+noise, f"{iid}: paired measurement noise")
        close(metadata["noise_watts"], noise, f"{iid}: stored noise")
        base_key = (tuple(record["state_counts"]), record["segments"], record["seed"])
        base = {field:record[field] for field in ("truth", "levels", "segment_weights", "switch_penalty")}
        base["standard_normal_noise"] = metadata["standard_normal_noise"]
        if base_key in paired_base:
            require(base == paired_base[base_key], f"{iid}: sigma variants changed latent instance")
        else:
            paired_base[base_key] = base
        require(record["preparation_seconds"] >= 0, f"{iid}: negative preparation time")
        instances[iid] = record
        direct[iid] = (states, energies, physical, scale)

    optimizers = {}
    evaluations = 0
    max_ideal_error = 0.0
    boundary_corrected_calls = 0
    max_boundary_correction = 0.0
    for oid in sorted(expected_optimizers):
        record = read(archive/"optimizations"/(oid+".json"))
        iid = record["instance_id"]
        require(oid == f"{iid}_p{record['depth']}" and record["protocol_sha256"] == protocol_sha,
                f"{oid}: identity/provenance")
        states, energies, _, scale = direct[iid]
        depth = record["depth"]
        require(len(record["gammas"]) == len(record["betas"]) == depth, f"{oid}: depth mismatch")
        for angles, bounds in ((record["gammas"], protocol["angle_bounds"]["gamma"]),
                               (record["betas"], protocol["angle_bounds"]["beta"])):
            require(all(np.isfinite(x) and bounds[0] <= x <= bounds[1] for x in angles), f"{oid}: angle bounds")
        require(record["status"] == "completed_fixed_budget", f"{oid}: optimizer unfinished")
        expected_evaluations = len(protocol["optimizer_restart_seeds"])*protocol["evaluations_per_restart"]
        require(record["evaluations"] == expected_evaluations, f"{oid}: evaluation count")
        require([r["seed"] for r in record["restarts"]] == protocol["optimizer_restart_seeds"], f"{oid}: restart seeds")
        for restart in record["restarts"]:
            history = np.asarray(restart["objective_history"], dtype=float)
            require(len(history) == restart["evaluations"] == protocol["evaluations_per_restart"], f"{oid}: history coverage")
            require(np.all(np.isfinite(history)), f"{oid}: nonfinite history")
            require(restart["best_evaluation_index"] == int(np.argmin(history)), f"{oid}: best history index")
            close(restart["normalized_expected_cost"], min(history), f"{oid}: restart best")
            close(restart["expected_cost"], scale*min(history), f"{oid}: restart expectation", atol=1e-7)
            close(history[0], np.mean(energies)/scale, f"{oid}: uniform first candidate")
            require(restart["initial_evaluations"] + sum(r["evaluations"]+r["padding_evaluations"] for r in restart["refinements"]) == len(history), f"{oid}: call ledger")
            require(restart["padding_evaluations"] == sum(r["padding_evaluations"] for r in restart["refinements"]), f"{oid}: padding ledger")
            for refinement in restart["refinements"]:
                require(refinement["evaluations"]+refinement["padding_evaluations"] == refinement["allowance"], f"{oid}: refinement budget")
            require(restart["runtime_seconds"] >= restart["statevector_seconds"] >= 0, f"{oid}: restart timing")
            if "boundary_roundoff_evaluations" in restart:
                bounds = ([protocol["angle_bounds"]["gamma"]]*depth+
                          [protocol["angle_bounds"]["beta"]]*depth)
                expected_tolerance = [8*np.finfo(float).eps*max(1., abs(a), abs(b)) for a,b in bounds]
                close(record["optimizer"]["boundary_roundoff_tolerance"], expected_tolerance,
                      f"{oid}: boundary correction tolerance", atol=0, rtol=1e-14)
                require(0 <= restart["boundary_roundoff_evaluations"] <= len(history) and
                        restart["boundary_roundoff_evaluations"] <= restart["boundary_roundoff_coordinates"] <= 2*depth*restart["boundary_roundoff_evaluations"],
                        f"{oid}: boundary correction count ledger")
                require(0 <= restart["boundary_max_correction"] <= max(expected_tolerance),
                        f"{oid}: non-roundoff boundary correction")
                boundary_corrected_calls += restart["boundary_roundoff_evaluations"]
                max_boundary_correction = max(max_boundary_correction, restart["boundary_max_correction"])
        chosen = int(np.argmin([r["normalized_expected_cost"] for r in record["restarts"]]))
        require(chosen == record["chosen_restart_index"], f"{oid}: restart selection")
        require(record["chosen_seed"] == record["restarts"][chosen]["seed"], f"{oid}: selected seed")
        require(record["gammas"] == record["restarts"][chosen]["gammas"] and record["betas"] == record["restarts"][chosen]["betas"], f"{oid}: selected angles")
        expected = direct_ideal_probabilities(states, energies, scale, instances[iid]["state_counts"], record["gammas"], record["betas"])
        close(record["probabilities"], expected, f"{oid}: independent ideal circuit", atol=1e-11, rtol=1e-9)
        max_ideal_error = max(max_ideal_error, float(np.max(np.abs(expected-np.asarray(record["probabilities"])))))
        close(record["expected_cost"], expected@energies, f"{oid}: selected expectation", atol=1e-7)
        close(record["normalized_expected_cost"], expected@energies/scale, f"{oid}: normalized selected expectation")
        require(record["expected_cost"] <= np.mean(energies)+1e-7, f"{oid}: worse than evaluated uniform candidate")
        require(record["runtime_seconds"] >= record["statevector_seconds"] >= 0, f"{oid}: total timing")
        evaluations += record["evaluations"]
        optimizers[oid] = record

    records, groups, independent_trials = {}, defaultdict(list), {}
    trials_checked = shots_regenerated = noise_checked = 0
    max_physical_audit_error = 0.0
    for did in sorted(expected_distributions):
        record = read(archive/"distributions"/(did+".json"))
        iid = record["instance_id"]
        instance = instances[iid]
        states, energies, physical_indices, _ = direct[iid]
        require(record["distribution_id"] == did and record["protocol_sha256"] == protocol_sha, f"{did}: identity/provenance")
        require(did == f"{iid}_p{record['depth']}_{record['noise']}", f"{did}: label mismatch")
        for field in ("state_counts", "segments", "seed", "measurement_noise", "qubits", "feasible_states"):
            require(record[field] == instance[field], f"{did}: {field} mismatch")
        p = np.asarray(record["probabilities"], dtype=float)
        require(p.shape == energies.shape and np.all(np.isfinite(p)) and np.all(p >= 0), f"{did}: invalid probabilities")
        feasible = float(np.sum(p))
        require(feasible <= 1+1e-10, f"{did}: excessive probability mass")
        close(record["feasible_probability"], min(1., feasible), f"{did}: all-shot feasible mass")
        close(record["invalid_probability"], max(0., 1-feasible), f"{did}: invalid mass")
        optimum, span = float(np.min(energies)), float(np.max(energies)-np.min(energies))
        tolerance = 1e-10*max(1., span)
        minimizers = np.flatnonzero(energies <= optimum+tolerance)
        popt = min(1., float(np.sum(p[minimizers])))
        for field, value in (("exact_cost", optimum), ("energy_span", span), ("optimum_tolerance", tolerance), ("optimum_probability", popt)):
            close(record[field], value, f"{did}: {field}", atol=1e-7 if field in ("exact_cost", "energy_span") else 1e-10)
        require(record["optimum_count"] == len(minimizers) and record["canonical_exact_index"] == minimizers[0], f"{did}: optimum-set convention")
        n99 = None if popt == 0 else 1 if popt == 1 else math.ceil(math.log(.01)/math.log1p(-popt))
        require(record["shots_to_optimum_99"] == n99, f"{did}: 99% shots")
        conditional = float(p@energies/feasible) if feasible else None
        ngap = (conditional-optimum)/span if conditional is not None and span else 0. if feasible else None
        close(record["conditional_expected_cost"], conditional, f"{did}: conditional energy", atol=1e-7)
        close(record["conditional_expected_normalized_gap"], ngap, f"{did}: conditional gap")
        canonical = states[minimizers[0]]
        truth = np.asarray(instance["truth"])
        close(record["exact_truth_category_accuracy"], np.mean(canonical == truth), f"{did}: exact/truth categories")
        close(record["exact_truth_bit_accuracy"], 1-2*np.count_nonzero(canonical != truth)/record["qubits"], f"{did}: exact/truth bits")
        if record["noise"] == "uniform":
            close(p, np.ones(len(p))/len(p), f"{did}: uniform reference")
            require(record["depth"] == 0, f"{did}: reference depth")
        elif record["noise"] == "ideal":
            close(p, optimizers[f"{iid}_p{record['depth']}"]["probabilities"], f"{did}: frozen ideal probabilities")
        else:
            noise = record["noise_metadata"]
            full = np.asarray(noise["physical_probabilities"], dtype=float)
            require(full.shape == (2**record["qubits"],) and np.all(np.isfinite(full)) and np.all(full >= 0), f"{did}: physical probability bounds")
            close(full.sum(), 1., f"{did}: physical normalization", atol=1e-12)
            require(noise["feasible_physical_indices"] == physical_indices.tolist(), f"{did}: independently encoded one-hot mapping")
            close(p, full[physical_indices], f"{did}: unconditional physical subset", atol=1e-12)
            require(noise["simulation_method"] == "exact_density_matrix" and noise["probabilities_conditioned_on_feasibility"] is False and noise["classical_fallback_used"] is False and noise["noise_model_is_hardware_calibration"] is False, f"{did}: simulation scope flags")
            parameters = noise["noise_parameters"]
            design = protocol["noise_design"][record["noise"]]
            for field, registered in (("single_qubit_depolarizing", "one_qubit_depolarization"), ("two_qubit_depolarizing", "two_qubit_depolarization"), ("readout_bitflip", "readout_flip")):
                close(parameters[field], design[registered], f"{did}: registered noise strength")
            circuit = noise["circuit"]
            require(circuit["num_qubits"] == record["qubits"] and circuit["qaoa_depth"] == record["depth"], f"{did}: physical circuit dimensions")
            require(circuit["ideal_audit_max_absolute_probability_error"] <= 1e-10 and circuit["ideal_audit_invalid_probability"] <= 1e-10, f"{did}: original physical/ideal audit failed")
            max_physical_audit_error = max(max_physical_audit_error, circuit["ideal_audit_max_absolute_probability_error"])
            require(noise["simulation_seconds"] >= 0 and circuit["compilation_seconds"] >= 0 and circuit["ideal_audit_seconds"] >= 0, f"{did}: negative circuit timings")
            noise_checked += 1

        expected_keys = set(itertools.product(protocol["shots"], range(protocol["sampling_repeats"])))
        actual_keys = [(r["shots"], r["repeat"]) for r in record["trials"]]
        require(len(actual_keys) == len(expected_keys) and set(actual_keys) == expected_keys, f"{did}: trial coverage")
        distribution = np.concatenate((p, [max(0., 1-float(p.sum()))]))
        distribution /= distribution.sum()
        recomputed = []
        for trial in record["trials"]:
            shots, repeat = trial["shots"], trial["repeat"]
            seed = seed_from_parts(iid, shots, repeat, "stage-b-measurements")
            require(trial["distribution_id"] == did and trial["sample_seed"] == seed, f"{did}: trial seed or identity")
            draws = np.random.default_rng(seed).choice(len(distribution), size=shots, p=distribution)
            observed, frequencies = np.unique(draws[draws < len(p)], return_counts=True)
            best = int(observed[np.argmin(energies[observed])]) if len(observed) else None
            gap = max(0., float(energies[best]-optimum)) if best is not None else None
            normal = gap/span if gap is not None and span else 0. if gap is not None else None
            expected = dict(distribution_id=did, shots=shots, repeat=repeat, sample_seed=seed,
                            feasible_shots=int(frequencies.sum()), no_feasible=int(best is None), best_index=best,
                            best_cost=float(energies[best]) if best is not None else None,
                            certified_gap=gap, normalized_gap=normal,
                            failure_penalized_normalized_gap=1. if normal is None else normal,
                            hit_optimum=int(gap is not None and gap <= tolerance),
                            truth_category_accuracy=float(np.mean(states[best] == truth)) if best is not None else 0.,
                            truth_bit_accuracy=1-2*np.count_nonzero(states[best] != truth)/record["qubits"] if best is not None else 0.,
                            canonical_exact_category_accuracy=float(np.mean(states[best] == canonical)) if best is not None else 0.,
                            canonical_exact_bit_accuracy=1-2*np.count_nonzero(states[best] != canonical)/record["qubits"] if best is not None else 0.)
            require(set(trial) == set(expected), f"{did}: unexpected trial fields")
            for field, value in expected.items():
                if isinstance(value, str) or field in ("sample_seed", "best_index", "shots", "repeat", "feasible_shots", "no_feasible", "hit_optimum"):
                    require(trial[field] == value, f"{did}/{shots}/{repeat}: {field}")
                else:
                    close(trial[field], value, f"{did}/{shots}/{repeat}: {field}", atol=1e-7 if field in ("best_cost", "certified_gap") else 1e-10)
            recomputed.append(expected)
            trials_checked += 1
            shots_regenerated += shots
        independent_trials[did] = recomputed
        key = (tuple(record["state_counts"]), record["segments"], record["measurement_noise"], record["depth"], record["noise"])
        groups[key].append(record)
        records[did] = record

    require(trials_checked == summary["sampling_trials"], "Trial total differs from summary")
    require(len(summary["groups"]) == len(groups), "Summary group count mismatch")
    summary_keys = set()
    bootstrap_checks = 0
    for result in summary["groups"]:
        key = (tuple(result["state_counts"]), result["segments"], result["measurement_noise"], result["depth"], result["noise"])
        require(key in groups and key not in summary_keys, "Duplicate or unexpected summary group")
        summary_keys.add(key)
        rows = groups[key]
        require([r["seed"] for r in rows] == protocol["instance_seeds"], "Group base-seed order or coverage")
        for field in GROUP_METRICS:
            check_bootstrap(result[field], [r[field] for r in rows], seed_from_parts(protocol["bootstrap_seed"], key, field), protocol["bootstrap_replicates"], f"{key}/{field}")
            bootstrap_checks += 1
        finite = [r["shots_to_optimum_99"] for r in rows if r["shots_to_optimum_99"] is not None]
        expected99 = dict(median=float(np.median(finite)) if finite else None,
                          min=min(finite) if finite else None, max=max(finite) if finite else None,
                          infinite_instances=len(rows)-len(finite))
        require(result["shots99"] == expected99, f"{key}: per-instance shots99 summary")
        require(set(result["shot_results"]) == {str(s) for s in protocol["shots"]}, f"{key}: shot summary coverage")
        for shots in protocol["shots"]:
            point = result["shot_results"][str(shots)]
            for field in TRIAL_METRICS:
                values = [np.mean([t[field] for t in independent_trials[r["distribution_id"]] if t["shots"] == shots]) for r in rows]
                check_bootstrap(point[field], values, seed_from_parts(protocol["bootstrap_seed"], key, shots, field), protocol["bootstrap_replicates"], f"{key}/{shots}/{field}")
                bootstrap_checks += 1
            analytic = [1.-(1.-r["optimum_probability"])**shots for r in rows]
            close(point["analytic_hit_probability_mean"], np.mean(analytic), f"{key}/{shots}: analytic hit curve", atol=1e-12)

    expected_pairs = {(tuple(s["state_counts"]), s["segments"], sigma, noise)
                      for s, sigma in itertools.product(protocol["shapes"], protocol["measurement_noise"])
                      for noise in (["ideal"]+protocol["noise_levels"] if sum(s["state_counts"])*s["segments"] <= protocol["noise_max_qubits"] else ["ideal"])}
    actual_pairs = set()
    for result in summary["paired_depth_comparisons"]:
        pair = (tuple(result["state_counts"]), result["segments"], result["measurement_noise"], result["noise"])
        require(pair in expected_pairs and pair not in actual_pairs, "Duplicate or unexpected depth contrast")
        actual_pairs.add(pair)
        counts, segments, sigma, noise = pair
        keys = [(counts, segments, sigma, depth, noise) for depth in (1, 2)]
        differences = []
        for seed in protocol["instance_seeds"]:
            means = []
            for depth in (1, 2):
                did = instance_name(counts, segments, seed, sigma)+f"_p{depth}_{noise}"
                means.append(np.mean([t["failure_penalized_normalized_gap"] for t in independent_trials[did] if t["shots"] == 2048]))
            differences.append(float(means[1]-means[0]))
        require(result["metric"] == "p2_minus_p1_penalized_normalized_gap_at_2048", "Unknown paired endpoint")
        close(result["per_seed"], differences, f"{pair}: paired differences")
        check_bootstrap(result, differences, seed_from_parts(protocol["bootstrap_seed"], keys, "paired"), protocol["bootstrap_replicates"], f"{pair}: paired depth contrast")
        bootstrap_checks += 1
    require(actual_pairs == expected_pairs, "Missing depth contrast")

    csv_path = archive/"trials.csv"
    require(csv_path.is_file(), "Flat trial export missing")
    source_hashes["trials.csv"] = digest(csv_path)
    csv_keys = set()
    with csv_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["distribution_id"], int(row["shots"]), int(row["repeat"]))
            require(key not in csv_keys, "Duplicate trial in CSV export")
            csv_keys.add(key)
            record = records[key[0]]
            trial = next(t for t in independent_trials[key[0]] if (t["shots"], t["repeat"]) == key[1:])
            expected = {field:record[field] for field in ("instance_id", "qubits", "seed", "measurement_noise", "depth", "noise")}
            expected.update(trial)
            require(set(row) == set(expected), "Flat trial export fields differ")
            for field, value in expected.items():
                if value is None:
                    require(row[field] == "", f"CSV missing sentinel {key}/{field}")
                elif isinstance(value, str):
                    require(row[field] == value, f"CSV string {key}/{field}")
                elif isinstance(value, (int, np.integer)):
                    require(int(row[field]) == value, f"CSV integer {key}/{field}")
                else:
                    close(float(row[field]), value, f"CSV numeric {key}/{field}", atol=1e-7 if field in ("best_cost", "certified_gap") else 1e-10)
    require(len(csv_keys) == trials_checked, "CSV trial coverage mismatch")
    return dict(status="passed_independent_full_archive_audit",
                audited_utc=datetime.now(timezone.utc).isoformat(),
                audit_script_sha256=digest(Path(__file__)), protocol_sha256=protocol_sha,
                instances_checked=len(instances), optimizations_checked=len(optimizers),
                optimizer_restarts_checked=sum(len(x["restarts"]) for x in optimizers.values()),
                objective_history_entries_checked=evaluations,
                boundary_roundoff_corrected_evaluations=boundary_corrected_calls,
                maximum_boundary_roundoff_correction=max_boundary_correction,
                preserved_predecessor=predecessor,
                ideal_circuits_independently_reconstructed=len(optimizers),
                maximum_independent_ideal_probability_error=max_ideal_error,
                distributions_checked=len(records), full_physical_noise_distributions_checked=noise_checked,
                maximum_archived_physical_ideal_audit_error=max_physical_audit_error,
                finite_shot_trials_regenerated=trials_checked, individual_shots_regenerated=shots_regenerated,
                grouped_summaries_checked=len(groups), paired_depth_contrasts_checked=len(actual_pairs),
                bootstrap_intervals_recomputed=bootstrap_checks, flat_csv_rows_checked=len(csv_keys),
                scope="All archived inputs, direct enumerated objectives, independent ideal circuits, physical feasibility mappings, every finite-shot trial and grouped/paired summaries; no independent rerun of noisy density evolution or optimizer search histories",
                limitations="Synthetic instances only; eight instance clusters per cell; noise is generic, not calibrated hardware; no quantum-advantage conclusion",
                source_sha256=source_hashes, runtime_seconds=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT/"results/stage_b/run_002")
    parser.add_argument("--check-only", action="store_true", help="Do not create an audit receipt")
    parser.add_argument("--output", type=Path, help="Create-new receipt path; defaults to archive/independent_audit.json")
    args = parser.parse_args()
    result = audit(args.archive.resolve())
    if not args.check_only:
        output = args.output or args.archive/"independent_audit.json"
        with output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({k:v for k, v in result.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
