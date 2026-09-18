#!/usr/bin/env python3
"""Make a source-backed Markdown companion to a completed Stage B archive.

This post-processing script does not train circuits, run simulations, alter
the frozen programme, or overwrite an existing artifact. It reads the frozen
summary and raw timing records, deduplicating circuit preparation metadata
shared between low/high noise distributions. Metrics retain their registered
instance-level aggregation; no confidence interval treats shot repeats or
measurement-noise variants as independent experimental instances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def key_for(value):
    return (tuple(value["state_counts"]), value["segments"], value["measurement_noise"], value["depth"], value["noise"])


def numeric(value, precision=4):
    if value is None:
        return "not finite"
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Non-finite report value")
    return f"{value:.{precision}f}"


def percent(value, precision=2):
    return numeric(100 * float(value), precision) + "%"


def ci(value, precision=6):
    return f"{numeric(value['mean'], precision)} [{numeric(value['ci95'][0], precision)}, {numeric(value['ci95'][1], precision)}]"


def shape_label(counts, segments):
    return "(" + ",".join(map(str, counts)) + f"), K={segments}"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def summarize_timings(instances, optimizations, distributions):
    """Count each measured component once, including reused preparation."""
    compiled = {}
    noisy = [value for value in distributions if "noise_metadata" in value]
    for record in noisy:
        metadata = record["noise_metadata"]["circuit"]
        # The same prepared circuit metadata is copied to low/high records.
        # A resumed process could prepare it again; differing recorded timing
        # metadata denotes a distinct recorded preparation and is kept.
        signature = (record["instance_id"], record["depth"], json.dumps(metadata, sort_keys=True))
        compiled[signature] = metadata
    components = {
        "input_preparation_seconds": sum(value["preparation_seconds"] for value in instances),
        "optimization_seconds": sum(value["runtime_seconds"] for value in optimizations),
        "compilation_seconds": sum(value["compilation_seconds"] for value in compiled.values()),
        "compiled_ideal_audit_seconds": sum(value["ideal_audit_seconds"] for value in compiled.values()),
        "noisy_simulation_seconds": sum(value["noise_metadata"]["simulation_seconds"] for value in noisy),
        "sampling_seconds": sum(value["sampling_seconds"] for value in distributions),
    }
    return {
        **components,
        "accounted_component_seconds": sum(components.values()),
        "training_statevector_seconds_subset_of_optimization": sum(value["statevector_seconds"] for value in optimizations),
        "optimizer_objective_evaluations": sum(value["evaluations"] for value in optimizations),
        "unique_recorded_circuit_preparations": len(compiled),
        "noisy_density_matrix_simulations": len(noisy),
        "max_compiled_ideal_probability_error": max((value["ideal_audit_max_absolute_probability_error"] for value in compiled.values()), default=None),
        "max_compiled_ideal_invalid_probability": max((value["ideal_audit_invalid_probability"] for value in compiled.values()), default=None),
        "boundary_roundoff_evaluations": sum(restart.get("boundary_roundoff_evaluations", 0) for value in optimizations for restart in value["restarts"]),
        "boundary_roundoff_coordinates": sum(restart.get("boundary_roundoff_coordinates", 0) for value in optimizations for restart in value["restarts"]),
        "max_boundary_correction": max((restart.get("boundary_max_correction", 0) for value in optimizations for restart in value["restarts"]), default=0),
        "interpretation": "Sum of recorded local component durations; excludes imports, archive I/O, bootstrap/reporting and unrecorded restart overhead. Not end-to-end wall time or quantum time. Training-statevector duration is already inside optimization and is not added twice. Shared low/high compilation/audit metadata is counted once.",
    }


def check_archive(out, protocol, summary, instances, optimizations, distributions):
    if summary.get("status") != "complete":
        raise ValueError("A completed summary is required")
    protocol_sha = sha(out / "protocol.json")
    if summary["protocol_sha256"] != protocol_sha:
        raise ValueError("Summary protocol hash does not match the frozen protocol")
    expected_counts = {
        "instances": (len(instances), protocol["expected_instances"]),
        "optimizations": (len(optimizations), protocol["expected_optimizations"]),
        "distributions": (len(distributions), protocol["expected_distributions"]),
    }
    for label, (actual, expected) in expected_counts.items():
        if actual != expected or summary[label] != expected:
            raise ValueError(f"Incomplete {label}: {actual}/{expected}")
    for value in instances + optimizations + distributions:
        if value["protocol_sha256"] != protocol_sha:
            raise ValueError("Archive record belongs to a different frozen protocol")
    expected_calls = len(protocol["optimizer_restart_seeds"]) * protocol["evaluations_per_restart"]
    for value in optimizations:
        if value["evaluations"] != expected_calls or value["status"] != "completed_fixed_budget":
            raise ValueError("Incomplete optimization or changed call budget")
    if sum(len(value["trials"]) for value in distributions) != summary["sampling_trials"]:
        raise ValueError("Raw trial count differs from summary")
    if len({value["id"] for value in instances}) != len(instances):
        raise ValueError("Duplicate instance ID")
    if len({value["distribution_id"] for value in distributions}) != len(distributions):
        raise ValueError("Duplicate distribution ID")
    for value in summary["groups"]:
        if value["n_instances"] != len(protocol["instance_seeds"]):
            raise ValueError("Group uses an unexpected number of base-instance seeds")


def build_report(out):
    protocol = read_json(out / "protocol.json")
    summary = read_json(out / "summary.json")
    instances = [read_json(path) for path in sorted((out / "instances").glob("*.json"))]
    optimizations = [read_json(path) for path in sorted((out / "optimizations").glob("*.json"))]
    distributions = [read_json(path) for path in sorted((out / "distributions").glob("*.json"))]
    check_archive(out, protocol, summary, instances, optimizations, distributions)
    groups = {key_for(value): value for value in summary["groups"]}
    if len(groups) != len(summary["groups"]):
        raise ValueError("Duplicate summarized condition")
    timings = summarize_timings(instances, optimizations, distributions)
    provenance = {
        "protocol_path": str((out / "protocol.json").resolve()),
        "protocol_sha256": sha(out / "protocol.json"),
        "summary_path": str((out / "summary.json").resolve()),
        "summary_sha256": sha(out / "summary.json"),
        "report_generator_path": str(Path(__file__).resolve()),
        "report_generator_sha256": sha(__file__),
        "scope": "Frozen synthetic Stage B programme; local classical simulation, not IBM hardware observations",
    }
    coverage = {key: summary[key] for key in ("instances", "optimizations", "distributions", "sampling_trials")}
    coverage["independent_base_seeds_per_shape"] = len(protocol["instance_seeds"])
    coverage["optimizer_restarts"] = sum(len(value["restarts"]) for value in optimizations)
    coverage["noise_distribution_count"] = sum("noise_metadata" in value for value in distributions)
    baseline_labels = (("uniform", 0, "Uniform"), ("ideal", 1, "Q.NILM p=1"), ("ideal", 2, "Q.NILM p=2"))

    def group(shape, sigma, noise, depth):
        return groups[(tuple(shape["state_counts"]), shape["segments"], sigma, depth, noise)]

    ideal_rows, ideal_records = [], []
    for shape in protocol["shapes"]:
        for noise, depth, label in baseline_labels:
            value = group(shape, 0.01, noise, depth)
            shots = value["shot_results"]["512"]
            record = {
                "state_counts": shape["state_counts"], "segments": shape["segments"],
                "qubits": value["qubits"], "method": label,
                "optimum_probability": value["optimum_probability"]["mean"],
                "median_shots99": value["shots99"]["median"],
                "infinite_shots99_instances": value["shots99"]["infinite_instances"],
                "hit_at_512": shots["hit_optimum"]["mean"],
                "truth_category_accuracy_at_512": shots["truth_category_accuracy"]["mean"],
                "expected_normalized_gap": value["conditional_expected_normalized_gap"]["mean"],
            }
            ideal_records.append(record)
            shots99 = numeric(record["median_shots99"], 1)
            if record["infinite_shots99_instances"]:
                shots99 += f" ({record['infinite_shots99_instances']} infinite)"
            ideal_rows.append([
                shape_label(shape["state_counts"], shape["segments"]), value["qubits"], label,
                percent(record["optimum_probability"]), shots99,
                percent(record["hit_at_512"]), percent(record["truth_category_accuracy_at_512"]), numeric(record["expected_normalized_gap"]),
            ])

    noisy_rows, noisy_records = [], []
    for shape in protocol["noise_shapes"]:
        for depth in protocol["depths"]:
            for noise in protocol["noise_levels"]:
                value = group(shape, 0.01, noise, depth)
                record = {
                    "state_counts": shape["state_counts"], "segments": shape["segments"],
                    "qubits": value["qubits"], "depth": depth, "noise": noise,
                    "feasible_probability": value["feasible_probability"]["mean"],
                    "all_shot_optimum_probability": value["optimum_probability"]["mean"],
                    "hit_at_32": value["shot_results"]["32"]["hit_optimum"]["mean"],
                    "hit_at_512": value["shot_results"]["512"]["hit_optimum"]["mean"],
                    "no_feasible_at_32": value["shot_results"]["32"]["no_feasible"]["mean"],
                }
                noisy_records.append(record)
                noisy_rows.append([shape_label(shape["state_counts"], shape["segments"]), value["qubits"], depth, noise,
                                   percent(record["feasible_probability"]), percent(record["all_shot_optimum_probability"]),
                                   percent(record["hit_at_32"]), percent(record["hit_at_512"]), percent(record["no_feasible_at_32"])])

    truth_rows, truth_records = [], []
    for shape in protocol["shapes"]:
        accuracies = {sigma: group(shape, sigma, "uniform", 0)["exact_truth_category_accuracy"]["mean"] for sigma in protocol["measurement_noise"]}
        record = {"state_counts": shape["state_counts"], "segments": shape["segments"],
                  "exact_truth_category_accuracy": {str(sigma): value for sigma, value in accuracies.items()},
                  "change_5pct_minus_1pct_percentage_points": 100 * (accuracies[0.05] - accuracies[0.01])}
        truth_records.append(record)
        truth_rows.append([shape_label(shape["state_counts"], shape["segments"]),
                           percent(accuracies[0.0]), percent(accuracies[0.01]), percent(accuracies[0.05]),
                           f"{record['change_5pct_minus_1pct_percentage_points']:+.2f}"])

    aggregate_rows, aggregate_records = [], []
    for sigma in protocol["measurement_noise"]:
        for noise, depth, label in baseline_labels:
            selected = [group(shape, sigma, noise, depth) for shape in protocol["shapes"]]
            record = {
                "measurement_noise": sigma, "method": label,
                "shape_count": len(selected), "base_seeds_per_shape": len(protocol["instance_seeds"]),
                "mean_optimum_probability": float(np.mean([value["optimum_probability"]["mean"] for value in selected])),
                "mean_hit_at_512": float(np.mean([value["shot_results"]["512"]["hit_optimum"]["mean"] for value in selected])),
                "mean_truth_category_accuracy_at_512": float(np.mean([value["shot_results"]["512"]["truth_category_accuracy"]["mean"] for value in selected])),
                "mean_expected_normalized_gap": float(np.mean([value["conditional_expected_normalized_gap"]["mean"] for value in selected])),
            }
            aggregate_records.append(record)
            aggregate_rows.append([percent(sigma, 0), label, percent(record["mean_optimum_probability"]),
                                   percent(record["mean_hit_at_512"]), percent(record["mean_truth_category_accuracy_at_512"]),
                                   numeric(record["mean_expected_normalized_gap"])])

    paired_ideal = [value for value in summary["paired_depth_comparisons"] if value["noise"] == "ideal"]
    paired_rows = [[shape_label(value["state_counts"], value["segments"]), percent(value["measurement_noise"], 0), ci(value)] for value in paired_ideal]
    largest = max(protocol["shapes"], key=lambda value: math.prod(value["state_counts"]) ** value["segments"])
    largest_primary = [value for value in paired_ideal if value["state_counts"] == largest["state_counts"] and value["segments"] == largest["segments"]]
    largest_lines = "\n".join(f"- Measurement noise {percent(value['measurement_noise'], 0)}: {ci(value)}." for value in largest_primary)

    timing_rows = [
        ["Input preparation", coverage["instances"], numeric(timings["input_preparation_seconds"], 3)],
        ["Ideal angle training (both restarts)", coverage["optimizations"], numeric(timings["optimization_seconds"], 3)],
        ["Noise-circuit compilation (deduplicated)", timings["unique_recorded_circuit_preparations"], numeric(timings["compilation_seconds"], 3)],
        ["Independent compiled ideal audits (deduplicated)", timings["unique_recorded_circuit_preparations"], numeric(timings["compiled_ideal_audit_seconds"], 3)],
        ["Density-matrix noise/readout simulation", timings["noisy_density_matrix_simulations"], numeric(timings["noisy_simulation_seconds"], 3)],
        ["Sampling and per-trial metrics", coverage["sampling_trials"], numeric(timings["sampling_seconds"], 3)],
        ["Recorded component total", "—", numeric(timings["accounted_component_seconds"], 3)],
    ]
    evidence = {
        "provenance": provenance, "coverage": coverage, "timings": timings,
        "ideal_measurement_1pct": ideal_records, "noise_measurement_1pct": noisy_records,
        "exact_truth_measurement_comparison": truth_records, "all_shape_descriptive_means": aggregate_records,
        "paired_ideal_depth_comparisons": paired_ideal,
    }
    report = f"""# Q.NILM Stage B: controlled categorical simulation results

The frozen Stage B matrix is complete: {coverage['instances']} synthetic inputs,
{coverage['optimizations']} independently trained depth settings ({coverage['optimizer_restarts']} optimizer restarts),
{coverage['distributions']} probability distributions, and {coverage['sampling_trials']} sampled trials.
These are local classical simulations of specified quantum circuits, not additional IBM hardware results or evidence of quantum advantage.

## Scope and interpretation

The six fixed categorical shapes span 4–24 physical qubits and 4–5,184 feasible states.
Each shape uses eight base-instance seeds, paired across 0%, 1%, and 5% Gaussian measurement-noise levels.
Noise percentages denote standard deviation relative to the largest channel maximum, not relative to the measured aggregate.
Ideal p=1 and p=2 each receive two 512-call restarts; p=2 is optimized independently and is not an angle-split p=1 circuit.
The objective is expected cost, not planted-label accuracy or optimum probability.

All probability and accuracy entries below are means across eight instance-level values unless stated otherwise.
Shot-based values first average 32 independent sampling repeats within each instance.
`P(opt)` is the probability of a certified minimum-cost feasible state; exact optimization need not recover the planted truth under noise or objective ambiguity.
Category accuracy is the fraction of correctly inferred channel/interval categories.
`E-gap` is the expected objective gap divided by that instance's feasible energy range.
`S99` is the median of per-instance analytic 99%-success shot counts, not a shot count calculated from the mean probability.

## Ideal depth and uniform comparisons at 1% measurement noise

The 1% table is a fixed reporting slice; every registered measurement-noise condition remains in the summary and the descriptive tables below.
Hit and category-accuracy columns use 512 shots. Uniform denotes a classical feasible sampler with the identical observed-sample selection rule.

{table(['Shape (categories), intervals', 'Qubits', 'Method', 'P(opt)', 'Median S99', 'Hit / 512', 'Category / 512', 'E-gap'], ideal_rows)}

## Controlled gate/readout noise at 1% measurement noise

Exact density-matrix simulation covers the preselected 4-, 8-, and 10-qubit shapes.
Ideal-trained angles are frozen before applying noise. Low noise uses one-/two-qubit depolarizing strengths 0.0001/0.001 and readout flip 0.005;
high noise uses 0.001/0.01 and readout flip 0.02. Depolarization follows every compiled one-/two-qubit gate, including RZ.
Compilation assumes all-to-all connectivity, so these are controlled synthetic error models, not IBM calibration or routed-hardware forecasts.

The optimum probability uses **all physical shots**, including invalid one-hot outcomes.
No-feasible trials are retained as abstentions: hit=0, category accuracy=0, and failure-penalized normalized best-sample gap=1. There is no classical fallback.

{table(['Shape', 'Qubits', 'Depth', 'Noise', 'Feasible shots', 'All-shot P(opt)', 'Hit / 32', 'Hit / 512', 'Abstain / 32'], noisy_rows)}

## Measurement noise changes the target-recovery problem

This comparison uses the canonical exact minimum-cost trajectory, not a quantum sampler; it distinguishes objective/planted-truth disagreement from sampler error.
Each objective contributes once per base instance, without duplicating it across circuit depths or sampling repeats.
The change column is descriptive percentage points (5% noise minus 1% noise).

{table(['Shape', 'Exact category, 0%', 'Exact category, 1%', 'Exact category, 5%', 'Change (pp)'], truth_rows)}

## Paired optimized-depth endpoint at 2,048 shots

The registered endpoint is p=2 minus p=1 failure-penalized normalized best-sample gap; negative values favour p=2.
Intervals are 95% paired percentile bootstrap intervals over eight base-instance seeds (10,000 resamples), not over shot repeats.
This is an exploratory, small-instance comparison; intervals are descriptive and are not multiplicity-adjusted significance tests.
A zero interval can reflect all sampled trials reaching an optimum and does not establish general equivalence.

For the largest (24-qubit, 5,184-feasible-state) shape:

{largest_lines}

All ideal shape/noise comparisons in the registered endpoint family are retained here:

{table(['Shape', 'Measurement noise', 'p2 − p1 gap [95% paired interval]'], paired_rows)}

## Descriptive coverage of every measurement-noise level

These are equally weighted arithmetic averages of the six shape means, offered only as a compact inventory of the complete sweep.
The shapes are heterogeneous fixed cases, not a sampled population; no pooled confidence interval, p-value, or enlarged independent sample size is attached.
Each underlying shape still has eight paired base seeds.

{table(['Measurement noise', 'Method', 'Mean P(opt)', 'Mean hit / 512', 'Mean category / 512', 'Mean E-gap'], aggregate_rows)}

## Local computational accounting

{table(['Recorded component', 'Count', 'Seconds'], timing_rows)}

There were {timings['optimizer_objective_evaluations']} counted objective evaluations.
The {numeric(timings['training_statevector_seconds_subset_of_optimization'], 3)} seconds spent inside the feasible-subspace simulator are already included in angle-training time, not added again.
Shared circuit compilation and ideal-audit metadata copied into low/high noise records are counted once.
The maximum independently audited ideal probability discrepancy was {timings['max_compiled_ideal_probability_error']:.3g}.
These component sums exclude imports, archive I/O, summary/bootstrap generation and unrecorded restart overhead; they are not end-to-end wall-clock time or quantum execution time.
No classical-versus-quantum speedup follows from these timings.

## Numerical recovery and provenance

The incomplete Run 001 archive is preserved. Its bounded Powell optimizer generated gamma = −4.440892098500626e−16 at a nominal zero boundary.
The replacement run retains the same scientific matrix, seeds and evaluation budgets, with a regression-tested numerical correction only:
boundary excursions no larger than 8 machine epsilons times the coordinate bound scale are snapped to the boundary; larger violations still fail.
The completed archive records {timings['boundary_roundoff_evaluations']} corrected evaluations ({timings['boundary_roundoff_coordinates']} coordinates),
with maximum displacement {timings['max_boundary_correction']:.3g}. This repair does not change the search domain or insert a favourable outcome.

- [Frozen protocol](protocol.json), SHA-256 `{provenance['protocol_sha256']}`.
- [Complete machine-readable summary](summary.json), SHA-256 `{provenance['summary_sha256']}`.
- [Raw sampling-trial metrics](trials.csv).
- [Input archives](instances/), [optimizer histories and timing](optimizations/), and [probability distributions and raw trial metrics](distributions/).
- [Exact report values and timing accounting](report-data.json).
- [Report-generation source](../../../scripts/summarize_stage_b_report.py), SHA-256 `{provenance['report_generator_sha256']}`.

Stage B supports bounded circuit/noise/shot-budget conclusions only. It does not replace the held-out R1Hz evaluation, the pending core Stage D ablations, or Stage G external-dataset validation.
"""
    return report, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/stage_b/run_002")
    args = parser.parse_args()
    out = args.run.resolve()
    destinations = (out / "report.md", out / "report-data.json")
    if any(path.exists() for path in destinations):
        raise FileExistsError("Report artifacts already exist; refusing to overwrite")
    report, evidence = build_report(out)
    with destinations[0].open("x") as stream:
        stream.write(report)
    with destinations[1].open("x") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(destinations[0]), "data": str(destinations[1]), "coverage": evidence["coverage"], "timings": evidence["timings"]}, indent=2))


if __name__ == "__main__":
    main()
