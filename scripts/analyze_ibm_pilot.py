#!/usr/bin/env python3
"""Reproduce the manuscript's IBM pilot statistics without contacting IBM."""

import argparse
import hashlib
import json
from math import sqrt
from pathlib import Path

import numpy as np

from quantum_nilm.ibm import counts_to_metrics
from quantum_nilm.qaoa import qaoa_probabilities
from quantum_nilm.qubo import build_binary_temporal_qubo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/ibm_pilot_statistics.json"))
    args = parser.parse_args()
    source_path = args.results_dir / "pilot" / "summary.json"
    source = json.loads(source_path.read_text())
    channels = source["channels"]
    powers = np.array([source["nominal_incremental_power_w"][c] for c in channels])
    penalties = np.array([source["switch_penalty"][c] for c in channels])
    aggregate = np.array(source["aggregate_segment_power_w"])
    weights = np.array(source["selected_segment_durations_blocks"])
    qubo = build_binary_temporal_qubo(aggregate, powers, penalties, weights)
    _, costs = qubo.energies()
    optimum = float(costs.min())
    mask = np.isclose(costs, optimum, rtol=0, atol=1e-8)
    probabilities = qaoa_probabilities(qubo, [source["gamma"]], [source["beta"]], source["phase_scale"])
    theoretical_mean = float(probabilities @ costs)
    uniform_p = float(mask.mean())
    records = {}
    for name in ("ibm_ideal", "ibm_noisy", "ibm_qpu_pilot"):
        directory = args.results_dir / name
        counts_path = directory / "counts.json"
        counts = json.loads(counts_path.read_text())
        metrics = counts_to_metrics(qubo, counts, np.array(source["reference_states"]),
                                    aggregate, powers, weights)
        saved = json.loads((directory / "summary.json").read_text())
        assert all(saved[k] == v for k, v in metrics.items()), f"Metric mismatch: {name}"
        n = metrics["shots"]
        hits = sum(count for bitstring, count in counts.items() if mask[int(bitstring, 2)])
        rate = hits / n
        z = 1.959963984540054
        denominator = 1 + z*z/n
        center = (rate + z*z/(2*n)) / denominator
        half = z * sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / denominator
        records[name] = {
            "counts_sha256": hashlib.sha256(counts_path.read_bytes()).hexdigest(),
            "optimum_hits": hits,
            "wilson_95_interval": [center-half, center+half],
            "percent_mean_above_exact_ideal_qaoa_expectation":
                100 * (metrics["mean_objective"] / theoretical_mean - 1),
            **metrics,
        }
    report = {
        "source_summary_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "interval_method": "Wilson score interval, z=1.959963984540054",
        "interval_scope": "Descriptive within-run iid shot model; excludes job, layout, date and window variability",
        "number_of_hardware_jobs": 1,
        "number_of_nilm_instances": 1,
        "number_of_exact_optima": int(mask.sum()),
        "exact_minimum_objective": optimum,
        "theoretical_ideal_qaoa_mean_objective": theoretical_mean,
        "theoretical_ideal_qaoa_optimum_probability": float(probabilities[mask].sum()),
        "uniform_mean_objective": float(costs.mean()),
        "uniform_optimum_probability": uniform_p,
        "uniform_success_probability_4096_draws": float(-np.expm1(4096 * np.log1p(-uniform_p))),
        "runs": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Verified three saved runs; manuscript statistics written to {args.output}")


if __name__ == "__main__":
    main()
