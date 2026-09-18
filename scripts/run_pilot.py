#!/usr/bin/env python3
"""Run the first measured-data Q.NILM circuit-validation experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from quantum_nilm.qaoa import export_openqasm3_p1, optimize_qaoa_p1, sample_best
from quantum_nilm.qubo import build_binary_temporal_qubo
from quantum_nilm.r1hz import (
    DEFAULT_CHANNELS,
    block_mean,
    choose_segments,
    event_compress,
    infer_binary_reference,
    load_window,
)


TARGET_CHANNELS = list(DEFAULT_CHANNELS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-seconds", type=int, default=30)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--event-threshold-ratio", type=float, default=0.05)
    parser.add_argument("--switch-penalty-ratio", type=float, default=0.02)
    parser.add_argument("--shots", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--aggregate-mode", choices=("signed", "legacy-clipped"), default="signed")
    args = parser.parse_args()

    timestamps, second_values = load_window(args.input, TARGET_CHANNELS)
    blocks = block_mean(second_values, args.block_seconds)
    states, incremental, baselines, thresholds = infer_binary_reference(blocks)
    (
        compressed_blocks,
        compressed_aggregate,
        compressed_states,
        compressed_weights,
        boundaries,
        event_threshold,
    ) = event_compress(
        blocks,
        states,
        incremental,
        baselines,
        args.event_threshold_ratio,
        aggregate_mode=args.aggregate_mode,
    )
    if compressed_blocks.shape[0] < args.segments:
        raise RuntimeError("Event detector produced fewer segments than requested")
    start_block = choose_segments(
        compressed_blocks,
        compressed_states,
        incremental,
        baselines,
        args.segments,
    )
    stop_block = start_block + args.segments
    selected = compressed_blocks[start_block:stop_block]
    reference_states = compressed_states[start_block:stop_block]
    aggregate = compressed_aggregate[start_block:stop_block]
    segment_weights = compressed_weights[start_block:stop_block]
    switch_penalties = args.switch_penalty_ratio * incremental**2
    qubo = build_binary_temporal_qubo(
        aggregate, incremental, switch_penalties, segment_weights
    )

    all_bits, all_costs = qubo.energies()
    exact_index = int(np.argmin(all_costs))
    exact_bits = all_bits[exact_index]
    exact_states = qubo.decode(exact_bits)
    exact_cost = float(all_costs[exact_index])

    ising_offset, ising_h, ising_j = qubo.to_ising()
    spins = 1.0 - 2.0 * all_bits
    ising_costs = (
        ising_offset
        + spins @ ising_h
        + np.einsum("bi,ij,bj->b", spins, ising_j, spins, optimize=True)
    )
    equivalence_error = float(np.max(np.abs(all_costs - ising_costs)))

    optimized = optimize_qaoa_p1(qubo)
    sampled_bits, sampled_cost, sampled_count = sample_best(
        qubo, optimized.probabilities, args.shots, args.seed
    )
    sampled_states = qubo.decode(sampled_bits)
    optimal_mask = np.isclose(all_costs, exact_cost, rtol=0.0, atol=1e-8)
    optimal_probability = float(optimized.probabilities[optimal_mask].sum())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    qasm = export_openqasm3_p1(
        qubo,
        optimized.gamma,
        optimized.beta,
        optimized.phase_scale,
    )
    (args.output_dir / "qnilm_p1.qasm").write_text(qasm)

    input_sha256 = hashlib.sha256(args.input.read_bytes()).hexdigest()
    gate_counts = {
        "h": sum(line.startswith("h ") for line in qasm.splitlines()),
        "rz": sum(line.startswith("rz(") for line in qasm.splitlines()),
        "cx": sum(line.startswith("cx ") for line in qasm.splitlines()),
        "rx": sum(line.startswith("rx(") for line in qasm.splitlines()),
    }

    selected_start_block = int(boundaries[start_block])
    start_second = selected_start_block * args.block_seconds
    selected_start_timestamp = int(timestamps[start_second])
    report = {
        "algorithm": "Q.NILM",
        "status": "circuit-validation pilot; not a quantum-advantage claim",
        "aggregate_mode": args.aggregate_mode,
        "aggregate_scope": (
            "selected-channel total minus fitted total baseline; diagnostic, not whole-house mains"
            if args.aggregate_mode == "signed"
            else "legacy sum of positively clipped channel residuals; not whole-house mains"
        ),
        "input": str(args.input),
        "input_sha256": input_sha256,
        "source_dataset_doi": "10.7910/DVN/RCB5VJ",
        "timezone": "America/Vancouver",
        "channels": TARGET_CHANNELS,
        "block_seconds": args.block_seconds,
        "input_blocks": int(blocks.shape[0]),
        "detected_event_segments": int(compressed_blocks.shape[0]),
        "event_threshold_w": event_threshold,
        "event_threshold_ratio": args.event_threshold_ratio,
        "selected_start_unix": selected_start_timestamp,
        "selected_start_local": datetime.fromtimestamp(
            selected_start_timestamp, ZoneInfo("America/Vancouver")
        ).isoformat(),
        "segments": args.segments,
        "selected_segment_durations_blocks": segment_weights.astype(int).tolist(),
        "qubits": qubo.n_variables,
        "nominal_incremental_power_w": dict(zip(TARGET_CHANNELS, incremental.tolist())),
        "baseline_power_w": dict(zip(TARGET_CHANNELS, baselines.tolist())),
        "state_threshold_w": dict(zip(TARGET_CHANNELS, thresholds.tolist())),
        "switch_penalty": dict(zip(TARGET_CHANNELS, switch_penalties.tolist())),
        "switch_penalty_ratio": args.switch_penalty_ratio,
        "aggregate_segment_power_w": aggregate.tolist(),
        "reference_states": reference_states.tolist(),
        "exact_states": exact_states.tolist(),
        "qaoa_best_sample_states": sampled_states.tolist(),
        "reference_vs_exact_accuracy": float(np.mean(reference_states == exact_states)),
        "qaoa_sample_vs_exact_accuracy": float(np.mean(sampled_states == exact_states)),
        "exact_cost": exact_cost,
        "qaoa_expected_cost": optimized.expected_cost,
        "qaoa_best_sample_cost": sampled_cost,
        "qaoa_optimal_probability": optimal_probability,
        "qaoa_best_sample_count": sampled_count,
        "shots": args.shots,
        "qaoa_depth": 1,
        "untranspiled_gate_counts": gate_counts,
        "gamma": optimized.gamma,
        "beta": optimized.beta,
        "phase_scale": optimized.phase_scale,
        "max_qubo_ising_equivalence_error": equivalence_error,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    with (args.output_dir / "selected_segments.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "segment",
                "aggregate_w",
                *[f"{channel}_mean_w" for channel in TARGET_CHANNELS],
                *[f"{channel}_reference_on" for channel in TARGET_CHANNELS],
                *[f"{channel}_exact_on" for channel in TARGET_CHANNELS],
                *[f"{channel}_qaoa_on" for channel in TARGET_CHANNELS],
            ]
        )
        for segment in range(args.segments):
            writer.writerow(
                [
                    segment,
                    aggregate[segment],
                    *selected[segment].tolist(),
                    *reference_states[segment].tolist(),
                    *exact_states[segment].tolist(),
                    *sampled_states[segment].tolist(),
                ]
            )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
