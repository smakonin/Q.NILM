#!/usr/bin/env python3
"""Run a deterministic small-instance benchmark for the Q.NILM circuit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from quantum_nilm.qaoa import optimize_qaoa_p1, sample_best
from quantum_nilm.qubo import build_binary_temporal_qubo


SCENARIOS = ((2, 3), (3, 3), (3, 4))
BASE_POWERS = np.array([90.0, 420.0, 1250.0])


def make_instance(
    n_appliances: int, n_segments: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    powers = BASE_POWERS[:n_appliances] * rng.uniform(0.9, 1.1, n_appliances)
    states = np.empty((n_segments, n_appliances), dtype=np.int8)
    states[0] = rng.integers(0, 2, n_appliances)
    for segment in range(1, n_segments):
        flips = rng.random(n_appliances) < 0.35
        states[segment] = np.logical_xor(states[segment - 1], flips)
    noise = rng.normal(0.0, 0.01 * np.max(powers), n_segments)
    aggregate = np.maximum(states @ powers + noise, 0.0)
    weights = rng.integers(1, 9, n_segments).astype(float)
    return aggregate, powers, states, weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--shots", type=int, default=2000)
    args = parser.parse_args()

    records: list[dict[str, float | int | bool]] = []
    for n_appliances, n_segments in SCENARIOS:
        for seed in range(args.seeds):
            aggregate, powers, truth, weights = make_instance(
                n_appliances, n_segments, seed
            )
            penalties = 0.02 * powers**2
            qubo = build_binary_temporal_qubo(
                aggregate, powers, penalties, segment_weights=weights
            )
            bits, costs = qubo.energies()
            exact_index = int(np.argmin(costs))
            exact_bits = bits[exact_index]
            exact_cost = float(costs[exact_index])
            qaoa = optimize_qaoa_p1(qubo, gamma_points=25, beta_points=19)
            sampled_bits, sampled_cost, _ = sample_best(
                qubo, qaoa.probabilities, args.shots, seed
            )
            optimal_probability = float(
                qaoa.probabilities[np.isclose(costs, exact_cost, atol=1e-9)].sum()
            )
            records.append(
                {
                    "appliances": n_appliances,
                    "segments": n_segments,
                    "qubits": qubo.n_variables,
                    "seed": seed,
                    "truth_exact_bit_accuracy": float(
                        np.mean(truth.reshape(-1) == exact_bits)
                    ),
                    "qaoa_exact_bit_accuracy": float(
                        np.mean(sampled_bits == exact_bits)
                    ),
                    "sample_hit_optimum": bool(
                        np.isclose(sampled_cost, exact_cost, atol=1e-8)
                    ),
                    "optimal_probability": optimal_probability,
                    "expected_cost": qaoa.expected_cost,
                    "exact_cost": exact_cost,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "instances.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(records[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)

    by_qubits: dict[str, dict[str, float | int]] = {}
    for qubits in sorted({int(record["qubits"]) for record in records}):
        subset = [record for record in records if record["qubits"] == qubits]
        by_qubits[str(qubits)] = {
            "instances": len(subset),
            "sample_success_rate": float(
                np.mean([record["sample_hit_optimum"] for record in subset])
            ),
            "mean_optimal_probability": float(
                np.mean([record["optimal_probability"] for record in subset])
            ),
            "mean_truth_exact_bit_accuracy": float(
                np.mean([record["truth_exact_bit_accuracy"] for record in subset])
            ),
        }
    summary = {
        "algorithm": "Q.NILM",
        "status": "small-instance simulator benchmark; not a quantum-advantage claim",
        "qaoa_depth": 1,
        "shots_per_instance": args.shots,
        "total_instances": len(records),
        "overall_sample_success_rate": float(
            np.mean([record["sample_hit_optimum"] for record in records])
        ),
        "by_qubits": by_qubits,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
