#!/usr/bin/env python3
"""Independently reconcile immutable QAOA campaign records and paired intervals.

This script reads no source measurements and neither trains nor submits jobs.
After hardware scoring completes it automatically includes those records; use a
new --output-dir to preserve this first simulation-only analysis.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ("dryr", "frdg", "vacu")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def summarize_method(records, expected):
    """Average each window's error sums over seeds, then pool valid blocks."""
    seeds = sorted({r["seed"] for r in records}, key=str)
    windows, seen = {}, set()
    for record in records:
        key = (record["window_id"], record["seed"])
        if key in seen or record["window_id"] not in expected:
            raise ValueError("Duplicate or unexpected method/window/seed record")
        seen.add(key)
        if record["blocks"] != expected[record["window_id"]]:
            raise ValueError("Method coverage differs from the frozen valid-block cohort")
        windows.setdefault(record["window_id"], []).append(record)
    if set(windows) != set(expected):
        raise ValueError("Incomplete window coverage; do not report a full-campaign MAE")
    averaged = {}
    for name, rows in sorted(windows.items()):
        if {r["seed"] for r in rows} != set(seeds):
            raise ValueError("Stochastic seed coverage differs across windows")
        for row in rows:
            for channel in CHANNELS:
                if row["appliances"][channel]["n_blocks"] != expected[name]:
                    raise ValueError("Appliance denominator differs from window block count")
        errors = {channel: float(np.mean([r["appliances"][channel]["absolute_error_sum_w"] for r in rows]))
                  for channel in CHANNELS}
        averaged[name] = {"blocks": expected[name], "absolute_error_sum_w": errors,
                          "aggregate_absolute_error_sum_w": float(np.mean([
                              r["raw_aggregate_mae_w"] * r["blocks"] for r in rows]))}
    blocks = sum(expected.values())
    maes = {channel: sum(w["absolute_error_sum_w"][channel] for w in averaged.values()) / blocks
            for channel in CHANNELS}
    timed = [r["solver_wall_time_s"] for r in records if r.get("solver_wall_time_s") is not None]
    return {"windows": len(expected), "unique_valid_blocks": blocks, "seeds": seeds,
            "stochastic_replication_note": "same windows repeated; not additional independent household observations",
            "appliance_mae_w": maes, "macro_appliance_mae_w": float(np.mean(list(maes.values()))),
            "aggregate_mae_w": sum(w["aggregate_absolute_error_sum_w"] for w in averaged.values()) / blocks,
            "total_recorded_solver_wall_time_s": sum(timed) if timed else None,
            "mean_solver_wall_time_per_seed_s": sum(timed) / len(seeds) if timed else None,
            "window_seed_mean_errors": averaged}


def paired_bootstrap(left, right, left_name, right_name):
    """Ratio of resampled sums, pairing windows after fixed-seed averaging."""
    a, b = left["window_seed_mean_errors"], right["window_seed_mean_errors"]
    if a.keys() != b.keys():
        raise ValueError("Paired methods must cover identical windows")
    names = sorted(a)
    differences, denominators = [], []
    for name in names:
        if a[name]["blocks"] != b[name]["blocks"]:
            raise ValueError("Paired block counts differ")
        differences.append(sum(a[name]["absolute_error_sum_w"].values()) -
                           sum(b[name]["absolute_error_sum_w"].values()))
        denominators.append(3 * a[name]["blocks"])
    differences, denominators = np.asarray(differences), np.asarray(denominators)
    draws = np.random.default_rng(7301).integers(0, len(names), size=(2000, len(names)))
    estimates = differences[draws].sum(axis=1) / denominators[draws].sum(axis=1)
    return {"left": left_name, "right": right_name, "n_paired_windows": len(names),
            "difference_w": float(differences.sum() / denominators.sum()),
            "paired_window_bootstrap_95_interval_w": np.quantile(estimates, [.025, .975]).tolist(),
            "replicates": 2000, "seed": 7301,
            "definition": "left minus right macro MAE; negative favors left; average fixed-seed error sums within window before resampling",
            "limits": "descriptive within-home, post-exposure comparisons; not multiplicity-adjusted or across-day hardware replication"}


def audit_raw_samples(folder, inputs, frozen, angles):
    """Spot-check twelve chunks using independently expanded direct energies.

    All prior-state links in six selected traces are checked. This is a scoped
    sample, not an audit of every shot file or a physical hardware validation.
    """
    levels = [np.asarray(x) for x in frozen["models"]["multistate"]["levels_w"]]
    penalties = .1 * np.asarray(frozen["models"]["multistate"]["ranges_w"]) ** 2
    windows = {w["window"]["id"]: w for w in inputs}
    chosen_files = ["test-000_qaoa_ideal_1907.json.gz", "test-004_qaoa_ideal_2907.json.gz",
                    "test-029_qaoa_ideal_3907.json.gz", "test-010_uniform_1907.json.gz",
                    "test-012_uniform_2907.json.gz", "test-019_exact_chunk_1907.json.gz"]
    rng, checks, links = np.random.default_rng(71123), [], 0
    for filename in chosen_files:
        with gzip.open(folder / filename, "rt") as handle:
            trace = json.load(handle)
        chunks = windows[trace["window_id"]]["chunks"]
        if len(chunks) != len(trace["records"]):
            raise ValueError("Incomplete sampled trace")
        prior = None
        for chunk, row in zip(chunks, trace["records"]):
            expected_prior = None if chunk["reset"] else prior
            if row["previous_states"] != expected_prior or chunk["chunk"] != row["chunk"]:
                raise ValueError("Own-prior-state chain mismatch")
            prior = row["states"][-1]
            links += 1
        for index in sorted(rng.choice(len(chunks), size=min(2, len(chunks)), replace=False).tolist()):
            chunk, row = chunks[index], trace["records"][index]
            sizes = [len(x) for x in levels] * len(chunk["weights"])
            states = np.empty((int(np.prod(sizes)), len(chunk["weights"]), len(levels)), dtype=int)
            basis, stride = np.arange(len(states)), 1
            for register, size in enumerate(sizes):
                states[:, register // len(levels), register % len(levels)] = (basis // stride) % size
                stride *= size
            prediction = sum(level[states[:, :, channel]] for channel, level in enumerate(levels))
            energy = np.sum(np.asarray(chunk["weights"]) * (np.asarray(chunk["aggregate"]) - prediction) ** 2, axis=1)
            energy += np.sum(penalties * (states[:, 1:] != states[:, :-1]), axis=(1, 2))
            if row["previous_states"] is not None:
                energy += np.sum(penalties * (states[:, 0] != row["previous_states"]), axis=1)
            ids, counts = np.asarray(row["raw_feasible_counts"], dtype=int).T
            if len(np.unique(ids)) != len(ids) or np.any(counts <= 0) or np.any(ids < 0) or np.any(ids >= len(states)):
                raise ValueError("Invalid saved feasible sample counts")
            chosen = int(ids[np.lexsort((ids, energy[ids]))[0]])
            if chosen != row["chosen_index"] or not np.array_equal(states[chosen], row["states"]):
                raise ValueError("Chosen trajectory was not the best observed sample")
            for computed, recorded in ((energy[chosen], row["best_observed_energy"]),
                                       (np.min(energy), row["conditional_exact_energy"])):
                if not np.isclose(computed, recorded, rtol=1e-12, atol=1e-5):
                    raise ValueError("Direct objective reconciliation failed")
            expected_shots = 1 if trace["method"] == "exact_chunk" else 256
            if int(counts.sum()) != expected_shots:
                raise ValueError("Unexpected sampled shot total")
            # An independent full enumeration supplies the exact optimum;
            # sample means here use saved counts, not an unsampled optimum.
            optimum_count = int(np.sum(counts[np.isclose(energy[ids], np.min(energy), rtol=0, atol=1e-8)]))
            if trace["method"] == "uniform" and not np.isclose(float(np.mean(energy)), row["distribution_expected_energy"], rtol=1e-12):
                raise ValueError("Uniform expected objective reconciliation failed")
            checks.append({"file": filename, "chunk": index, "shots": int(counts.sum()),
                           "direct_best_sample_energy": float(energy[chosen]),
                           "direct_exact_energy": float(np.min(energy)),
                           "empirical_mean_energy": float(np.average(energy[ids], weights=counts)),
                           "exact_optimum_shots": optimum_count, "sample_only_selection_verified": True})
    return {"method": "independent direct categorical energy enumeration; saved samples only",
            "selection_seed": 71123, "trace_files": len(chosen_files), "prior_links_checked": links,
            "chunks_checked": len(checks), "checks": checks,
            "limits": "does not independently regenerate ideal quantum probability vectors or audit every trace"}


def make_markdown(analysis):
    labels = {"exact_full": "Full-run exact DP (compressed)", "exact_chunk": "Matched two-interval exact",
              "uniform": "Uniform feasible sampling", "qaoa_ideal": "Ideal QAOA simulation",
              "qaoa_ibm_with_declared_fallback": "IBM QPU + declared fallback"}
    lines = ["# Full-coverage Q.NILM extension: numerical analysis", "",
             "This reuses the original 30 already-examined test windows; it is a frozen post-exposure extension, not a newly blind test. "
             "All methods use mains input, the original multistate/background model, and all 86,396 valid 30-second blocks. "
             "The quantum circuit uses training-only frozen angles. Ideal simulation is not physical-QPU evidence.", "",
             "## Appliance power error", "",
             "| Method | Runs / replication | Dryer MAE W | Fridge MAE W | Vacuum MAE W | Macro MAE W | Aggregate MAE W |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, method in analysis["methods"].items():
        values = [method["appliance_mae_w"][c] for c in CHANNELS] + [method["macro_appliance_mae_w"], method["aggregate_mae_w"]]
        replication = "1 hardware campaign" if name == "qaoa_ibm_with_declared_fallback" else (f"{len(method['seeds'])} fixed seeds" if len(method['seeds']) > 1 else "1 exact solve per window")
        lines.append(f"| {labels.get(name, name)} | {replication} | " + " | ".join(f"{x:.3f}" for x in values) + " |")
    lines += ["", "The ideal-QAOA and uniform-simulation rows average absolute-error sums across three fixed seeds within each window, then pool blocks; "
              "the hardware row is one adaptive campaign, not three simulated seeds or independent hardware repetitions. "
              "The same observations are not counted as three independent datasets. The full-run compressed DP reproduces the original "
              "Table VI result exactly. The two-interval exact method may have slightly lower appliance MAE despite a less globally optimized "
              "aggregate objective: these are different loss functions.", "", "## Paired 24-hour-window comparisons", "",
              "| Left minus right | Macro MAE difference W | Descriptive 95% interval W |",
              "|---|---:|---:|"]
    for comparison in analysis["paired_comparisons"]:
        low, high = comparison["paired_window_bootstrap_95_interval_w"]
        lines.append(f"| {comparison['left']} minus {comparison['right']} | {comparison['difference_w']:.3f} | [{low:.3f}, {high:.3f}] |")
    lines += ["", "Negative differences favor the left method. The bootstrap resamples 30 paired windows 2,000 times with seed 7301, "
              "after averaging stochastic errors across the fixed seeds. These are descriptive, post-exposure within-home intervals, "
              "not multiplicity-adjusted claims or uncertainty across independent hardware dates.", "", "## Coverage and interpretation", "",
              "- 4,881 compressed intervals across 34 valid runs; 2,451 circuits per stochastic seed: 2,430 two-interval circuits and 21 single-interval tails.",
              "- Ideal QAOA uses 256 shots per circuit, 627,456 per seed and 1,882,368 over the three seeds. Every ideal shot lies in the feasible one-hot subspace; there is no ideal-simulation fallback.",
              "- Every method carries its own previous inferred state and resets at gaps. No classical exact solution is used as a QAOA warm start.",
              "- Ideal simulated QAOA improves over uniform sampling but remains worse than both exact classical controls on macro appliance MAE. This statement does not describe hardware performance; no quantum advantage is established.",
              "- State/event metrics elsewhere use high-state proxies, not verified physical ON states. Rare vacuum activity and coarse background attribution remain limitations.",
              "- The original model is unchanged; no appliance MAE was used to choose the QAOA angles. Classical enumeration, simulation, and angle training remain part of resource accounting.", ""]
    hardware = analysis["hardware"]
    if hardware["status"] == "complete_scored":
        lines += ["## Physical hardware coverage", "",
                  f"Raw hardware shots: {hardware['raw_shots']:,}; feasible shots: {hardware['feasible_shots']:,} "
                  f"({100 * hardware['feasible_fraction']:.3f}%). Declared fallback was used for {hardware['fallback_chunks']:,} chunks / "
                  f"{hardware['fallback_blocks']:,} blocks. The hardware MAE includes this fallback; it is not raw feasible-only QPU accuracy. "
                  "Infeasible shots are charged and reported, not repaired or hidden.", ""]
    else:
        lines += ["Physical hardware results were not complete and scored when this analysis was created; no full hardware MAE is inferred from partial coverage.", ""]
    lines += ["## Audit and provenance", "",
              f"All saved per-seed pooled MAEs reconcile with window-level error sums. A scoped audit checked "
              f"{analysis['raw_sample_audit']['prior_links_checked']} prior-state links in six traces and directly enumerated twelve "
              "selected chunk objectives, confirming that chosen outputs occur in the saved samples. This is not a complete raw-shot audit. "
              "`analysis.json` records exact numbers, source hashes, per-window seed-mean errors, bootstrap definitions, and individual audit checks.", "",
              "The original protocol, model, counts, prediction traces and result summaries are unchanged. "
              "Any regenerated analysis must use a new output directory. " +
              ("The separately labelled hardware row is included above." if hardware["status"] == "complete_scored" else
               "A future analysis can include hardware only after its complete scored records are available."), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=ROOT / "results/quantum_heldout")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    campaign = args.campaign_dir.resolve()
    output = args.output_dir.resolve() if args.output_dir else campaign / "simulation"
    if (output / "analysis.json").exists() or (output / "summary.md").exists():
        parser.error("Analysis already exists; choose a new --output-dir rather than overwriting")
    config, inputs, frozen, angles = (read(campaign / name) for name in
                                     ("protocol.json", "test_inputs.json", "model.json", "angles.json"))
    expected = {w["window"]["id"]: w["blocks"] for w in inputs}
    if len(expected) != 30 or sum(expected.values()) != 86396:
        raise ValueError("This analysis targets the frozen 30-window, 86,396-block campaign")
    source_paths = [campaign / name for name in ("protocol.json", "model.json", "angles.json", "input_hashes.json", "test_inputs.json")]
    for name, digest in read(campaign / "input_hashes.json").items():
        if sha(campaign / name) != digest:
            raise ValueError("Frozen input hash mismatch")
    freeze = read(campaign / "simulation/freeze.json")
    if sha(campaign / "angles.json") != freeze["angles_sha256"] or sha(campaign / "protocol.json") != freeze["protocol_sha256"]:
        raise ValueError("Simulation angle/protocol freeze mismatch")
    records = read(campaign / "simulation/test_windows.json")
    source_summary = read(campaign / "simulation/summary.json")
    source_paths += [campaign / "simulation" / name for name in ("freeze.json", "summary.json", "test_windows.json")]
    methods = {name: summarize_method([r for r in records if r["model"] == name], expected)
               for name in ("exact_full", "exact_chunk", "uniform", "qaoa_ideal")}
    for key, row in source_summary["pooled_results"].items():
        recovered = summarize_method([r for r in records if r["model"] == row["model"] and r["seed"] == row["seed"]], expected)
        if not np.isclose(recovered["macro_appliance_mae_w"], row["macro_appliance_mae_w"], rtol=0, atol=1e-12):
            raise ValueError(f"Per-seed summary reconciliation failed: {key}")
        for channel in CHANNELS:
            if not np.isclose(recovered["appliance_mae_w"][channel], row["appliances"][channel]["mae_w"], rtol=0, atol=1e-12):
                raise ValueError(f"Per-seed appliance reconciliation failed: {key}/{channel}")
    original_path = Path(config["original_dir"]) / "summary.json"
    original = read(original_path)["pooled_results"]["mains/multistate_compressed"]
    if methods["exact_full"]["macro_appliance_mae_w"] != original["macro_appliance_mae_w"]:
        raise ValueError("Full DP does not reproduce the original compressed comparison")
    source_paths.append(original_path)
    comparisons = [paired_bootstrap(methods[a], methods[b], a, b) for a, b in
                   (("qaoa_ideal", "uniform"), ("qaoa_ideal", "exact_chunk"),
                    ("qaoa_ideal", "exact_full"), ("exact_chunk", "exact_full"))]
    hardware = {"status": "not_complete_and_scored", "raw_feasibility": None, "fallback_coverage": None}
    hardware_files = [campaign / "hardware" / name for name in ("completed.json", "summary.json", "test_windows.json")]
    if all(path.exists() for path in hardware_files):
        hardware_records, hardware_summary = read(hardware_files[2]), read(hardware_files[1])
        name = "qaoa_ibm_with_declared_fallback"
        methods[name] = summarize_method(hardware_records, expected)
        if not np.isclose(methods[name]["macro_appliance_mae_w"], hardware_summary["macro_appliance_mae_w"], rtol=0, atol=1e-12):
            raise ValueError("Hardware summary reconciliation failed")
        hardware = {"status": "complete_scored", **{key: hardware_summary[key] for key in
                    ("raw_shots", "feasible_shots", "feasible_fraction", "fallback_chunks", "fallback_blocks")},
                    "coverage": "all original blocks, including explicitly declared fallback",
                    "completion": read(hardware_files[0])}
        comparisons += [paired_bootstrap(methods[name], methods[other], name, other)
                        for other in ("qaoa_ideal", "exact_chunk", "exact_full")]
        source_paths += hardware_files
    audit = audit_raw_samples(campaign / "simulation", inputs, frozen, angles)
    source_paths += sorted((campaign / "simulation").glob("*.json.gz"))
    analysis = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "study": config["study"], "methods": methods, "paired_comparisons": comparisons,
                "hardware": hardware, "raw_sample_audit": audit,
                "original_compressed_exact_mae_reproduced_w": original["macro_appliance_mae_w"],
                "classical_angle_training_wall_time_s": angles["classical_training_wall_time_s"],
                "angle_objective_evaluations": angles["objective_evaluations"],
                "provenance_sha256": {str(path.relative_to(campaign)) if path.is_relative_to(campaign) else str(path): sha(path) for path in source_paths},
                "analysis_script_sha256": sha(Path(__file__)), "limitations": config["limitations"]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    (output / "summary.md").write_text(make_markdown(analysis))
    print(json.dumps({"methods": {name: row["macro_appliance_mae_w"] for name, row in methods.items()},
                      "paired_comparisons": comparisons, "hardware_status": hardware["status"]}, indent=2))


if __name__ == "__main__":
    main()
