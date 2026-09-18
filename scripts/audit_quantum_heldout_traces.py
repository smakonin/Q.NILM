#!/usr/bin/env python3
"""Audit every saved categorical QAOA/uniform/exact-chunk trace offline.

This independently evaluates the direct categorical cost without importing the
optimization, QAOA, IBM, or campaign inference helpers. No source measurements,
accounts, network, retraining, or new predictions are used. The original traces
and summaries are never modified. Full-run exact DP has no saved trajectory and
is explicitly outside the full-trace audit; its score is reconciled separately.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(np.isfinite(actual) and np.isfinite(expected), f"Nonfinite value: {message}")
    delta = abs(float(actual) - float(expected))
    require(delta <= 1e-5 + 1e-12 * max(abs(float(actual)), abs(float(expected))),
            f"Direct arithmetic mismatch: {message}; difference={delta}")
    return delta


def basis_states(counts, intervals):
    sizes = list(counts) * intervals
    indices = np.arange(int(np.prod(sizes)))
    digits, stride = [], 1
    for size in sizes:
        digits.append((indices // stride) % size)
        stride *= size
    return np.column_stack(digits).reshape(-1, intervals, len(counts))


def validated_states(value, counts, intervals, context):
    states = np.asarray(value)
    require(states.shape == (intervals, len(counts)) and states.dtype.kind in "iu",
            f"Invalid categorical state shape/type: {context}")
    for channel, size in enumerate(counts):
        require(np.all((states[:, channel] >= 0) & (states[:, channel] < size)),
                f"Out-of-range categorical state: {context}")
    return states


def direct_base_cost(states, aggregate, weights, levels, penalties):
    # State-independent residual constants are included in this segment-mean
    # objective. The within-segment raw-block variance is not available here.
    powers = np.stack([level[states[:, :, channel]] for channel, level in enumerate(levels)], axis=2)
    reconstruction = np.sum(weights * (aggregate - np.sum(powers, axis=2)) ** 2, axis=1)
    transitions = np.sum((states[:, 1:] != states[:, :-1]) * penalties, axis=(1, 2))
    return reconstruction + transitions


def validate_chunks(window):
    position, run, segments, resets = 0, -1, 0, 0
    for index, chunk in enumerate(window["chunks"]):
        require(chunk["chunk"] == index, "Chunk IDs are not sequential")
        weights = np.asarray(chunk["weights"])
        aggregate = np.asarray(chunk["aggregate"])
        require(weights.ndim == 1 and len(weights) in (1, 2) and weights.dtype.kind in "iu"
                and np.all(weights > 0), "Invalid duration weights")
        require(aggregate.shape == weights.shape and np.all(np.isfinite(aggregate)), "Invalid aggregate shape/values")
        require(chunk["block_start"] == position, "Overlapping or uncovered block ranges")
        require(chunk["block_stop"] - chunk["block_start"] == int(weights.sum()), "Block duration mismatch")
        if chunk["reset"]:
            require(chunk["run"] == run + 1, "A reset must start exactly the next valid run")
            run, resets = chunk["run"], resets + 1
        else:
            require(chunk["run"] == run and run >= 0, "Prior state would cross a gap or uninitialized run")
        if index:
            previous = window["chunks"][index - 1]
            elapsed = chunk["start_unix"] - previous["start_unix"]
            duration = 30 * sum(previous["weights"])
            require(elapsed > duration if chunk["reset"] else elapsed == duration,
                    "Gap/reset policy disagrees with physical timestamps")
        position, segments = chunk["block_stop"], segments + len(weights)
    require(position == window["blocks"], "Window does not cover all frozen valid blocks")
    return segments, resets


def audit(campaign):
    started = perf_counter()
    config = read(campaign / "protocol.json")
    frozen = read(campaign / "model.json")
    windows = read(campaign / "test_inputs.json")
    input_hashes = read(campaign / "input_hashes.json")
    for name, digest in input_hashes.items():
        require(sha(campaign / name) == digest, f"Frozen input hash mismatch: {name}")
    freeze = read(campaign / "simulation/freeze.json")
    require(sha(campaign / "angles.json") == freeze["angles_sha256"], "Frozen angles changed")
    require(sha(campaign / "protocol.json") == freeze["protocol_sha256"], "Frozen protocol changed")
    levels = [np.asarray(x, dtype=float) for x in frozen["models"]["multistate"]["levels_w"]]
    counts = [len(x) for x in levels]
    penalties = config["rho"] * np.asarray(frozen["models"]["multistate"]["ranges_w"]) ** 2
    bases = {length: basis_states(counts, length) for length in (1, 2)}
    variants = [("exact_chunk", config["simulation_seeds"][0])]
    variants += [(method, seed) for method in ("uniform", "qaoa_ideal") for seed in config["simulation_seeds"]]
    summaries = {f"{method}/{seed}": {"method": method, "seed": seed, "windows": 0,
                  "chunks": 0, "intervals": 0, "blocks": 0, "raw_draws": 0,
                  "exact_optimum_draws_at_absolute_1e_minus_8": 0,
                  "conditional_best_matches_exact": 0, "direct_segment_objective_sum": 0.,
                  "empirical_sample_energy_sum": 0.} for method, seed in variants}
    source_paths = [campaign / name for name in ("protocol.json", "model.json", "angles.json", "input_hashes.json", "test_inputs.json")]
    source_paths.append(campaign / "simulation/freeze.json")
    verified_chunks = verified_samples = verified_links = resets_total = 0
    numerical_ties, max_error = [], 0.
    coverage = []
    expected_files = set()
    for wi, window in enumerate(windows):
        wid = window["window"]["id"]
        segments, resets = validate_chunks(window)
        coverage.append({"window_id": wid, "blocks": window["blocks"], "intervals": segments,
                         "chunks": len(window["chunks"]), "valid_runs": resets})
        traces, priors, direct_cost_sums, saved_cost_sums, generators = {}, {}, {}, {}, {}
        for method, seed in variants:
            key = f"{method}/{seed}"
            path = campaign / "simulation" / f"{wid}_{method}_{seed}.json.gz"
            expected_files.add(path.name)
            with gzip.open(path, "rt") as handle:
                trace = json.load(handle)
            require((trace["window_id"], trace["method"], trace["seed"]) == (wid, method, seed), "Trace metadata mismatch")
            require(len(trace["records"]) == len(window["chunks"]), "Incomplete trace coverage")
            traces[key], priors[key] = trace["records"], None
            direct_cost_sums[key], saved_cost_sums[key] = 0., 0.
            generators[key] = np.random.default_rng(seed + wi * 10000)
            source_paths.append(path)
        for ci, chunk in enumerate(window["chunks"]):
            length = len(chunk["weights"])
            states = bases[length]
            base_cost = direct_base_cost(states, np.asarray(chunk["aggregate"]), np.asarray(chunk["weights"]), levels, penalties)
            for method, seed in variants:
                key, context = f"{method}/{seed}", f"{wid}/{method}/{seed}/chunk-{ci}"
                row = traces[key][ci]
                require(row["chunk"] == ci, f"Trace chunk order mismatch: {context}")
                previous = None if chunk["reset"] else priors[key]
                require(row["previous_states"] == (None if previous is None else previous.tolist()), f"Own-state/reset mismatch: {context}")
                recovered = validated_states(row["states"], counts, length, context)
                energy = base_cost if previous is None else base_cost + np.sum(penalties * (states[:, 0] != previous), axis=1)
                require(row["logical_qubits"] == sum(counts) * length and row["feasible_states"] == len(states), f"State-space metadata mismatch: {context}")
                samples = np.asarray(row["raw_feasible_counts"])
                require(samples.ndim == 2 and samples.shape[1] == 2 and samples.dtype.kind in "iu", f"Invalid saved counts shape: {context}")
                ids, frequencies = samples[:, 0], samples[:, 1]
                require(len(ids) > 0 and len(np.unique(ids)) == len(ids) and np.all(ids[:-1] < ids[1:])
                        and np.all((ids >= 0) & (ids < len(states))) and np.all(frequencies > 0), f"Invalid feasible counts: {context}")
                expected_shots = 1 if method == "exact_chunk" else config["shots_per_chunk"]
                require(int(frequencies.sum()) == expected_shots, f"Shot count mismatch: {context}")
                chosen = row["chosen_index"]
                require(isinstance(chosen, int) and chosen in ids and np.array_equal(recovered, states[chosen]), f"Chosen state absent from raw samples: {context}")
                observed_best = int(ids[np.argmin(energy[ids])])
                max_error = max(max_error, close(energy[chosen], energy[observed_best], context + "/sample-best"),
                                close(energy[chosen], row["best_observed_energy"], context + "/chosen-cost"),
                                close(np.min(energy), row["conditional_exact_energy"], context + "/exact-cost"))
                if chosen != observed_best:
                    numerical_ties.append({"context": context, "stored_index": chosen, "direct_index": observed_best,
                                           "direct_difference": float(energy[chosen] - energy[observed_best])})
                if method == "exact_chunk":
                    max_error = max(max_error, close(energy[chosen], np.min(energy), context + "/exact-trajectory"))
                    require(row["distribution_expected_energy"] is None, "Exact control has unexpected stochastic expectation")
                else:
                    expected_energy = row["distribution_expected_energy"]
                    require(np.isfinite(expected_energy) and float(np.min(energy)) - 1e-5 <= expected_energy <= float(np.max(energy)) + 1e-5,
                            f"Distribution expectation outside cost bounds: {context}")
                    if method == "uniform":
                        max_error = max(max_error, close(np.mean(energy), expected_energy, context + "/uniform-mean"))
                        sampled = generators[key].choice(len(states), size=expected_shots, p=np.full(len(states), 1 / len(states)))
                        regenerated_ids, regenerated_counts = np.unique(sampled, return_counts=True)
                        require(np.array_equal(ids, regenerated_ids) and np.array_equal(frequencies, regenerated_counts), f"Uniform RNG replay mismatch: {context}")
                priors[key] = recovered[-1].copy()
                direct_cost_sums[key] += float(energy[chosen])
                saved_cost_sums[key] += row["best_observed_energy"]
                summary = summaries[key]
                summary["chunks"] += 1
                summary["intervals"] += length
                summary["raw_draws"] += expected_shots
                summary["exact_optimum_draws_at_absolute_1e_minus_8"] += int(frequencies[np.isclose(energy[ids], np.min(energy), rtol=0, atol=1e-8)].sum())
                summary["conditional_best_matches_exact"] += int(np.isclose(energy[chosen], np.min(energy), rtol=0, atol=1e-8))
                summary["empirical_sample_energy_sum"] += float(np.dot(energy[ids], frequencies))
                verified_chunks, verified_samples, verified_links = verified_chunks + 1, verified_samples + expected_shots, verified_links + 1
                resets_total += int(chunk["reset"])
        for key in traces:
            max_error = max(max_error, close(direct_cost_sums[key], saved_cost_sums[key], f"{wid}/{key}/window-cost"))
            summaries[key]["windows"] += 1
            summaries[key]["blocks"] += window["blocks"]
            summaries[key]["direct_segment_objective_sum"] += direct_cost_sums[key]
        print(f"Audited all saved traces for {wid}: {wi + 1}/{len(windows)} windows", flush=True)
    observed_files = {p.name for p in (campaign / "simulation").glob("test-*.json.gz")}
    require(observed_files == expected_files, "Unexpected extra/missing stochastic or exact-chunk trace files")
    for summary in summaries.values():
        summary["empirical_mean_objective_per_draw"] = summary.pop("empirical_sample_energy_sum") / summary["raw_draws"]
        summary["conditional_optimum_draw_fraction"] = summary["exact_optimum_draws_at_absolute_1e_minus_8"] / summary["raw_draws"]
    original_path = Path(config["original_dir"]) / "summary.json"
    original = read(original_path)["pooled_results"]["mains/multistate_compressed"]
    replay_path = campaign / "simulation/summary.json"
    full = read(replay_path)["pooled_results"]["exact_full/None"]
    require(full["macro_appliance_mae_w"] == original["macro_appliance_mae_w"], "Full-DP score no longer matches original Table VI")
    source_paths.extend([original_path, replay_path])
    return {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
            "execution": "offline direct-cost and trajectory audit; no hardware or source-data access",
            "trace_files_audited": len(expected_files), "unique_windows": len(windows),
            "unique_valid_blocks": sum(w["blocks"] for w in windows),
            "chunks_audited_including_replicates": verified_chunks,
            "raw_draws_audited_including_exact_certificates": verified_samples,
            "prior_state_links_checked": verified_links, "gap_or_window_resets_checked": resets_total,
            "methods_and_seeds": summaries, "window_coverage": coverage,
            "maximum_absolute_direct_cost_discrepancy": max_error,
            "numeric_tolerance": "absolute 1e-5 plus relative 1e-12; conditional hit counts separately use absolute 1e-8 only",
            "numerical_tie_selection_cases": numerical_ties,
            "exact_full": {"trace_status": "no full-run state trajectory saved; not included in trajectory audit",
                           "macro_appliance_mae_matches_original_table_vi": True,
                           "macro_appliance_mae_w": full["macro_appliance_mae_w"]},
            "limitations": ["Does not regenerate every ideal QAOA probability vector or QAOA RNG sequence; their saved samples and conditional costs are audited.",
                            "Uniform RNG sequences are independently replayed; exact-chunk trajectories are certified by direct feasible-state enumeration.",
                            "No source measurements are reread, so this audit does not independently recompute appliance-power errors from raw measurements.",
                            "Conditional optimum rates pool different method-specific prior states and are descriptive, not a shared-instance advantage comparison.",
                            "All costs use duration-weighted segment means; the state-independent within-segment raw-block variance is unavailable in these inputs."],
            "elapsed_wall_time_s": perf_counter() - started,
            "provenance_sha256": {str(p.relative_to(campaign)) if p.is_relative_to(campaign) else str(p): sha(p) for p in source_paths},
            "audit_script_sha256": sha(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=ROOT / "results/quantum_heldout")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    campaign = args.campaign_dir.resolve()
    output = args.output.resolve() if args.output else campaign / "simulation/full_trace_audit.json"
    if output.exists():
        parser.error("Audit output already exists; choose a new --output path")
    result = audit(campaign)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "trace_files_audited", "unique_windows", "unique_valid_blocks",
                      "chunks_audited_including_replicates", "raw_draws_audited_including_exact_certificates", "elapsed_wall_time_s")}, indent=2))


if __name__ == "__main__":
    main()
