#!/usr/bin/env python3
"""Offline, post-exposure replay at each circuit's observed usable-shot budget.

No account access, submission, fitting, or original-artifact mutation occurs.
The observed hardware feasibility counts are endogenous to that hardware run;
this is a conditional diagnostic, not an identified causal noise ablation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from quantum_nilm.categorical_qaoa import prepare_categorical_problem, categorical_qaoa_probabilities
from run_quantum_heldout import (CHANNELS, check_inputs, checked_window, expand_predictions,
                                model_arrays, read_json, save_json, score_prediction, sha)
from summarize_quantum_heldout import summarize_method, paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]


def check_hashes(expected):
    for name, digest in expected.items():
        if sha(Path(name)) != digest:
            raise RuntimeError(f"Diagnostic source/code hash changed: {name}")


def count_raw_feasibility(raw_counts, register_sizes):
    """Count raw one-hot validity directly, including zero- and multi-hot shots."""
    histograms = [Counter() for _ in register_sizes]
    full_valid, total = 0, 0
    for bitstring, count in raw_counts.items():
        if (not isinstance(bitstring, str) or len(bitstring) != sum(register_sizes)
                or set(bitstring) - {"0", "1"} or isinstance(count, bool)
                or not isinstance(count, int) or count <= 0):
            raise ValueError("Invalid raw bitstring/count")
        offset, valid = 0, True
        for register, size in enumerate(register_sizes):
            hamming = bitstring[::-1][offset:offset + size].count("1")
            histograms[register][hamming] += count
            valid &= hamming == 1
            offset += size
        total += count
        full_valid += count * valid
    if total <= 0:
        raise ValueError("Missing raw hardware shots")
    return int(full_valid), total, histograms


def build_hardware_profile(campaign, windows, config, frozen):
    lookup = {w["window"]["id"]: w for w in windows}
    budgets = {wid: [None] * len(w["chunks"]) for wid, w in lookup.items()}
    sizes = [len(x) for x in frozen["models"]["multistate"]["levels_w"]]
    grouped = {k: {"valid": [], "raw": 0, "histograms": [Counter() for _ in sizes * k]}
               for k in (1, 2)}
    vacuum = Counter()
    paths = sorted((campaign / "hardware").glob("stage_*_result.json"))
    completed = read_json(campaign / "hardware/completed.json")
    if len(paths) != completed["stages"]:
        raise ValueError("Incomplete hardware stages")
    for stage, path in enumerate(paths):
        if path.name != f"stage_{stage:03d}_result.json":
            raise ValueError("Hardware stage sequence is incomplete")
        for row in read_json(path)["rows"]:
            wid, index = row["window_id"], row["chunk"]
            if wid not in lookup or not 0 <= index < len(budgets[wid]) or budgets[wid][index] is not None:
                raise ValueError("Duplicate or unexpected hardware circuit")
            chunk = lookup[wid]["chunks"][index]
            k = len(chunk["weights"])
            valid, total, histograms = count_raw_feasibility(row["raw_counts"], sizes * k)
            if total != config["shots_per_chunk"] or valid != row["sample_score"]["feasible_shots"]:
                raise ValueError("Raw hardware feasibility does not match saved count")
            if bool(row["fallback_used"]) != (valid == 0):
                raise ValueError("Hardware fallback disagrees with zero feasible count")
            budgets[wid][index] = {"chunk": index, "effective_shots": valid, "hardware_raw_shots": total,
                                   "intervals": k, "reset": chunk["reset"]}
            group = grouped[k]
            group["valid"].append(valid)
            group["raw"] += total
            for dest, source in zip(group["histograms"], histograms):
                dest.update(source)
            for state, weight in zip(row["states"], chunk["weights"]):
                high = state[2] == 1
                vacuum["prediction_blocks"] += weight
                vacuum["predicted_high_blocks"] += weight * high
                vacuum["fallback_blocks"] += weight * row["fallback_used"]
                vacuum["fallback_high_blocks"] += weight * high * row["fallback_used"]
                vacuum["nonfallback_high_blocks"] += weight * high * (not row["fallback_used"])
    if any(value is None for rows in budgets.values() for value in rows):
        raise ValueError("Missing circuit shot budgets")
    profile = {}
    for k, group in grouped.items():
        valid = np.asarray(group["valid"], dtype=int)
        profile[str(k)] = {"intervals": k, "qubits": sum(sizes) * k,
            "circuits": len(valid), "raw_shots": group["raw"], "feasible_shots": int(valid.sum()),
            "feasible_fraction": float(valid.sum() / group["raw"]),
            "mean_usable_shots": float(valid.mean()), "median_usable_shots": float(np.median(valid)),
            "usable_shots_quartiles": np.quantile(valid, [.25, .75]).tolist(),
            "minimum_usable_shots": int(valid.min()), "maximum_usable_shots": int(valid.max()),
            "zero_usable_circuits": int(np.count_nonzero(valid == 0)),
            "at_most_one_usable_circuits": int(np.count_nonzero(valid <= 1)),
            "at_most_three_usable_circuits": int(np.count_nonzero(valid <= 3)),
            "usable_count_histogram": {str(a): b for a, b in sorted(Counter(valid.tolist()).items())},
            "register_hamming_weight_counts": [{str(a): b for a, b in sorted(h.items())} for h in group["histograms"]]}
    hardware_records = read_json(campaign / "hardware/test_windows.json")
    total_blocks = sum(r["blocks"] for r in hardware_records)
    confusion = {key: sum(r["appliances"]["vacu"]["state"][key] for r in hardware_records)
                 for key in ("tp", "tn", "fp", "fn")}
    true_sum = sum(r["appliances"]["vacu"]["true_power_sum_w"] for r in hardware_records)
    bias_sum = sum(r["appliances"]["vacu"]["signed_power_error_sum_w"] for r in hardware_records)
    if vacuum["prediction_blocks"] != total_blocks or vacuum["predicted_high_blocks"] != confusion["tp"] + confusion["fp"]:
        raise ValueError("Predicted vacuum states do not reconcile with scored coverage")
    vacuum.update(confusion)
    profile = {"by_intervals": profile, "vacuum": {**vacuum,
        "high_state_power_w": frozen["models"]["multistate"]["levels_w"][2][1],
        "reference_high_threshold_w": config["state_proxy_thresholds_w"][2],
        "reference_high_blocks": confusion["tp"] + confusion["fn"],
        "actual_mean_power_w": true_sum / total_blocks,
        "predicted_mean_power_w": (true_sum + bias_sum) / total_blocks,
        "predicted_high_fraction": vacuum["predicted_high_blocks"] / total_blocks,
        "reference_high_fraction": (confusion["tp"] + confusion["fn"]) / total_blocks},
        "caveat": "K=1 circuits are nonrandomized tails, not a controlled circuit-size experiment; high-state proxies are not manual ON annotations"}
    return profile, budgets, paths


def freeze(output, campaign):
    if output.exists():
        raise FileExistsError("Use a new diagnostic directory; earlier artifacts are immutable")
    check_inputs(campaign)
    config, frozen, angles, windows = (read_json(campaign / name) for name in
                                      ("protocol.json", "model.json", "angles.json", "test_inputs.json"))
    profile, budgets, hardware_paths = build_hardware_profile(campaign, windows, config, frozen)
    source_paths = [campaign / name for name in ("protocol.json", "model.json", "angles.json", "test_inputs.json",
                    "input_hashes.json", "simulation/test_windows.json", "simulation/summary.json", "hardware/test_windows.json",
                    "hardware/summary.json", "hardware/completed.json")]
    source_paths += hardware_paths + [Path(config["original_dir"]) / "data_quality.json"]
    code_paths = [Path(__file__), ROOT / "scripts/run_quantum_heldout.py", ROOT / "scripts/summarize_quantum_heldout.py",
                  ROOT / "src/quantum_nilm/categorical_qaoa.py", ROOT / "src/quantum_nilm/evaluation.py",
                  ROOT / "src/quantum_nilm/heldout_data.py"]
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "campaign": str(campaign),
        "study": "post-exposure effective-usable-shot conditional diagnostic",
        "budget_rule": "each fixed chunk receives its observed hardware feasible-shot count; no count-based exclusion",
        "methods": ["qaoa_effective", "uniform_effective"], "simulation_seeds": config["simulation_seeds"],
        "rng_rule": "base seed plus original window index times 10000; zero-budget chunks consume no random draws",
        "boundary_policy": "each replay carries only its own last output, reset at every original window or timestamp gap",
        "zero_budget_policy": "repeat own previous state; at reset use the lowest-power state in each register; report fallback explicitly",
        "nonzero_decoding": "minimum direct objective among sampled feasible states, ties lower feasible index; never insert an unsampled optimum",
        "model_and_angles": "unchanged original training-only centroids, rho, compression threshold and frozen QAOA angles; no fitting",
        "expected_windows": len(windows), "expected_valid_blocks": sum(w["blocks"] for w in windows),
        "expected_chunks_per_seed": sum(len(w["chunks"]) for w in windows),
        "effective_draws_per_seed": sum(b["effective_shots"] for rows in budgets.values() for b in rows),
        "expected_zero_budget_chunks_per_seed": sum(b["effective_shots"] == 0 for rows in budgets.values() for b in rows),
        "statistical_description": "average absolute-error sums over three fixed seeds within each of 30 windows; paired window bootstrap 2000 draws, seed7301; descriptive only",
        "limitations": ["Hardware feasible counts are observed, endogenous outcomes of a different noisy state history; replay does not identify a causal effect or fraction of hardware error attributable to lost shots.",
                        "Replay uses ideal or uniform feasible distributions, not hardware-noisy conditional distributions, and carries its own history rather than the hardware history.",
                        "This is post-exposure diagnostic reuse of one household, not a newly blind accuracy test, hardware experiment, or quantum advantage claim.",
                        "Zero-budget fallbacks are classical outputs and are included in all-block accuracy; they are not quantum samples.",
                        "Original high-state proxies and rare-load limitations remain; no test-label model or angle tuning is performed."],
        "source_sha256": {str(path.resolve()): sha(path) for path in source_paths},
        "code_sha256": {str(path.resolve()): sha(path) for path in code_paths}}
    output.mkdir(parents=True)
    save_json(output / "protocol.json", protocol)
    save_json(output / "hardware_profile.json", profile)
    save_json(output / "effective_budgets.json", budgets)
    save_json(output / "frozen_artifacts.json", {name: sha(output / name) for name in
              ("protocol.json", "hardware_profile.json", "effective_budgets.json")})
    print(json.dumps({"status": "diagnostic frozen before replay", "effective_draws_per_seed": protocol["effective_draws_per_seed"],
                      "zero_budget_chunks_per_seed": protocol["expected_zero_budget_chunks_per_seed"]}, indent=2), flush=True)


def replay_chunks(chunks, budgets, levels, penalties, gammas, betas, method, seed):
    """No ground truth or hardware previous state is accepted by this function."""
    if method not in ("qaoa_effective", "uniform_effective"):
        raise ValueError("Unknown replay method")
    if len(chunks) != len(budgets):
        raise ValueError("Budget and chunk coverage differ")
    for index, (chunk, budget) in enumerate(zip(chunks, budgets)):
        n = budget["effective_shots"]
        if (chunk["chunk"] != index or budget["chunk"] != index or isinstance(n, (bool, np.bool_))
                or not isinstance(n, (int, np.integer)) or n < 0
                or n > budget["hardware_raw_shots"]
                or budget.get("reset", chunk["reset"]) != chunk["reset"]
                or budget.get("intervals", len(chunk["weights"])) != len(chunk["weights"])):
            raise ValueError("Invalid effective budget or chunk order")
    started, rng, previous, records = perf_counter(), np.random.default_rng(seed), None, []
    for chunk, budget in zip(chunks, budgets):
        if chunk["reset"]:
            previous = None
        problem = prepare_categorical_problem(chunk["aggregate"], levels, penalties,
                                             chunk["weights"], previous_states=previous)
        n = budget["effective_shots"]
        if n == 0:
            low = np.array([int(np.argmin(level)) for level in levels]) if previous is None else previous
            states = np.tile(low, (len(chunk["weights"]), 1))
            chosen, ids, frequencies, expected = None, [], [], None
            prediction = np.array([sum(levels[i][row[i]] for i in range(len(levels))) for row in states])
            energy = float(np.sum(np.asarray(chunk["weights"]) * (np.asarray(chunk["aggregate"]) - prediction) ** 2))
            energy += float(np.sum(penalties * (states[1:] != states[:-1])))
            if previous is not None:
                energy += float(np.sum(penalties * (states[0] != previous)))
        else:
            probabilities = (categorical_qaoa_probabilities(problem, gammas, betas)
                             if method == "qaoa_effective" else np.full(len(problem.energies), 1 / len(problem.energies)))
            sampled = rng.choice(len(problem.energies), size=n, p=probabilities)
            ids, frequencies = np.unique(sampled, return_counts=True)
            chosen = int(ids[np.argmin(problem.energies[ids])])
            states, energy = problem.states[chosen], float(problem.energies[chosen])
            expected = float(probabilities @ problem.energies)
        records.append({"chunk": chunk["chunk"], "effective_shots": int(n),
            "previous_states": None if previous is None else previous.tolist(), "states": states.tolist(),
            "fallback_used": n == 0, "selected_state_origin": "declared_zero_budget_fallback" if n == 0 else "sampled_feasible_state",
            "chosen_index": chosen, "output_objective": energy, "conditional_exact_energy": float(np.min(problem.energies)),
            "distribution_expected_energy": expected,
            "raw_feasible_counts": [[int(i), int(c)] for i, c in zip(ids, frequencies)]})
        previous = states[-1].copy()
    return records, perf_counter() - started


def verify_frozen(output):
    for name, digest in read_json(output / "frozen_artifacts.json").items():
        if sha(output / name) != digest:
            raise RuntimeError("Frozen diagnostic artifact changed")
    protocol = read_json(output / "protocol.json")
    check_hashes(protocol["source_sha256"])
    check_hashes(protocol["code_sha256"])
    return protocol


def replay(output):
    protocol = verify_frozen(output)
    if (output / "traces").exists() or (output / "summary.json").exists():
        raise FileExistsError("Replay already started; refuse silent overwrite/retest")
    campaign = Path(protocol["campaign"])
    config, frozen, angles, windows = (read_json(campaign / name) for name in
                                      ("protocol.json", "model.json", "angles.json", "test_inputs.json"))
    budgets = read_json(output / "effective_budgets.json")
    quality = read_json(Path(config["original_dir"]) / "data_quality.json")
    levels, penalties, _ = model_arrays(frozen)
    (output / "traces").mkdir()
    save_json(output / "replay_started.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
              "frozen_artifacts_sha256": sha(output / "frozen_artifacts.json")})
    started, records = perf_counter(), []
    for index, window in enumerate(windows):
        wid, pending = window["window"]["id"], []
        for method in protocol["methods"]:
            for seed in protocol["simulation_seeds"]:
                raw, elapsed = replay_chunks(window["chunks"], budgets[wid], levels, penalties,
                    angles["gammas"], angles["betas"], method, seed + index * 10000)
                with gzip.open(output / "traces" / f"{wid}_{method}_{seed}.json.gz", "wt") as handle:
                    json.dump({"window_id": wid, "method": method, "seed": seed, "effective_rng_seed": seed + index * 10000,
                               "solver_wall_time_s": elapsed, "records": raw}, handle, allow_nan=False)
                prediction = expand_predictions(window, raw, levels)
                fallback_blocks = sum(c["block_stop"] - c["block_start"] for c, r in zip(window["chunks"], raw) if r["fallback_used"])
                pending.append((method, seed, prediction, elapsed, fallback_blocks))
        # New predictions are fixed before reading references for this window.
        loaded = checked_window(config, quality, window["window"], ["main", *CHANNELS])
        for method, seed, prediction, elapsed, fallback_blocks in pending:
            result = score_prediction(window, loaded, prediction, config["state_proxy_thresholds_w"], method, seed, elapsed)
            result.update({"fallback_blocks": fallback_blocks,
                "zero_budget_chunks": sum(b["effective_shots"] == 0 for b in budgets[wid]),
                "effective_draws": sum(b["effective_shots"] for b in budgets[wid]),
                "source_window_sha256": loaded["source_window_sha256"]})
            records.append(result)
        print(f"Effective-shot diagnostic scored {index + 1}/{len(windows)} fixed windows", flush=True)
    save_json(output / "test_windows.json", records)
    expected = {w["window"]["id"]: w["blocks"] for w in windows}
    methods = {name: summarize_method([r for r in records if r["model"] == name], expected) for name in protocol["methods"]}
    original = read_json(campaign / "simulation/test_windows.json") + read_json(campaign / "hardware/test_windows.json")
    for name in ("qaoa_ideal", "uniform", "exact_chunk", "exact_full", "qaoa_ibm_with_declared_fallback"):
        methods[name] = summarize_method([r for r in original if r["model"] == name], expected)
    comparisons = [paired_bootstrap(methods[a], methods[b], a, b) for a, b in (
        ("qaoa_effective", "qaoa_ideal"), ("uniform_effective", "uniform"),
        ("qaoa_effective", "uniform_effective"),
        ("qaoa_ibm_with_declared_fallback", "qaoa_effective"),
        ("qaoa_ibm_with_declared_fallback", "uniform_effective"))]
    per_seed = {}
    for method in protocol["methods"]:
        for seed in protocol["simulation_seeds"]:
            rows = [r for r in records if r["model"] == method and r["seed"] == seed]
            per_seed[f"{method}/{seed}"] = {"windows": len(rows), "blocks": sum(r["blocks"] for r in rows),
                "effective_draws": sum(r["effective_draws"] for r in rows),
                "fallback_chunks": sum(r["zero_budget_chunks"] for r in rows),
                "fallback_blocks": sum(r["fallback_blocks"] for r in rows)}
            if per_seed[f"{method}/{seed}"]["effective_draws"] != protocol["effective_draws_per_seed"]:
                raise RuntimeError("Effective-shot coverage mismatch")
    verify_frozen(output)
    summary = {"completed_utc": datetime.now(timezone.utc).isoformat(), "execution": "offline conditional effective-shot replay, not hardware",
        "protocol_sha256": sha(output / "protocol.json"), "methods": methods, "paired_comparisons": comparisons,
        "coverage_per_seed": per_seed, "wall_time_s": perf_counter() - started,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "source_and_code_integrity_verified_before_and_after": True, "limitations": protocol["limitations"]}
    save_json(output / "summary.json", summary)
    lines = ["# Effective usable-shot diagnostic", "",
        "This is an offline post-exposure diagnostic, not new hardware execution or a causal attribution experiment. "
        "Each replay receives exactly the observed hardware feasible count for the corresponding chunk, including zero. "
        "Each method carries its own prior state and uses the declared fallback at zero budget; no hardware prior states or test labels guide inference.", "",
        "| Method | Dryer MAE W | Fridge MAE W | Vacuum MAE W | Macro MAE W | Aggregate MAE W |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, row in methods.items():
        values = [row["appliance_mae_w"][ch] for ch in CHANNELS] + [row["macro_appliance_mae_w"], row["aggregate_mae_w"]]
        lines.append(f"| {name} | " + " | ".join(f"{x:.3f}" for x in values) + " |")
    lines += ["", "New replay and original ideal/uniform rows average three fixed seeds; the hardware row is one adaptive campaign. "
        "Every row covers the same 30 windows and 86,396 valid blocks. Each new replay seed uses 8,770 feasible draws over 2,451 chunks; "
        "131 zero-budget chunks / 4,310 blocks use explicit classical fallback. These fallback counts match the budget map, not necessarily the hardware's chosen fallback states.", "",
        "## Descriptive paired comparisons", "", "| Left minus right | Macro MAE difference W | 95% window-bootstrap interval W |", "|---|---:|---:|"]
    for row in comparisons:
        low, high = row["paired_window_bootstrap_95_interval_w"]
        lines.append(f"| {row['left']} minus {row['right']} | {row['difference_w']:.3f} | [{low:.3f}, {high:.3f}] |")
    lines += ["", "Intervals use 2,000 paired-window resamples, seed 7301, after averaging error sums across simulation seeds. "
              "They are descriptive and are not multiplicity-adjusted or independent hardware-date uncertainty.", "", "## Interpretation limits", ""]
    lines += ["- " + item for item in protocol["limitations"]]
    lines += ["", "The gap between a 256-shot control and an effective-shot replay diagnoses sensitivity to a reduced usable sample budget. "
              "It must not be called the causal fraction of hardware error caused by invalid shots: observed counts, noisy distributions, and state histories are not independently randomized. "
              "`hardware_profile.json` preserves the full raw-count-derived K=1/K=2 breakdown and vacuum false-positive evidence. "
              "`protocol.json` records every source/code hash; `effective_budgets.json`, per-window traces, and `test_windows.json` retain the complete audit trail.", ""]
    (output / "summary.md").write_text("\n".join(lines))
    tracked = sorted(path for path in output.rglob("*") if path.is_file())
    save_json(output / "checksums.json", {str(path.relative_to(output)): sha(path) for path in tracked})
    print(json.dumps({name: row["macro_appliance_mae_w"] for name, row in methods.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("freeze", "replay"))
    parser.add_argument("--campaign-dir", type=Path, default=ROOT / "results/quantum_heldout")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/quantum_diagnostics/effective_shots")
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args.output_dir.resolve(), args.campaign_dir.resolve())
    else:
        replay(args.output_dir.resolve())


if __name__ == "__main__":
    main()
