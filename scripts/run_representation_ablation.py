#!/usr/bin/env python3
"""Audit legacy residual clipping against affine baseline centering.

This retrospective diagnostic freezes the calibration and interval boundaries
of the historical pilot. It never submits a quantum job or overwrites results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from quantum_nilm.classical import solve_binary_temporal_exact
from quantum_nilm.qubo import build_binary_temporal_qubo
from quantum_nilm.r1hz import block_mean, load_window


ROOT = Path(__file__).resolve().parents[1]


def legacy_boundaries(aggregate: np.ndarray, threshold: float) -> np.ndarray:
    """Reproduce the frozen pilot's detector, including its two-block grouping."""
    differences = np.abs(np.diff(aggregate))
    candidates = (np.flatnonzero(differences >= threshold) + 1).tolist()
    events: list[int] = []
    group: list[int] = []
    for candidate in candidates:
        if group and candidate - group[-1] >= 2:
            events.append(max(group, key=lambda index: differences[index - 1]))
            group = []
        group.append(candidate)
    if group:
        events.append(max(group, key=lambda index: differences[index - 1]))
    return np.asarray([0, *events, aggregate.size], dtype=int)


def agreement(correct: int, total: int) -> dict[str, int | float]:
    return {"correct": int(correct), "total": int(total), "fraction": correct / total}


def summarize_case(
    name: str,
    boundaries: np.ndarray,
    block_values: np.ndarray,
    block_reference: np.ndarray,
    baselines: np.ndarray,
    powers: np.ndarray,
    penalties: np.ndarray,
    mode_aggregates: dict[str, np.ndarray],
) -> dict:
    starts, stops = boundaries[:-1], boundaries[1:]
    weights = np.diff(boundaries)
    reference = np.asarray(
        [(block_reference[start:stop].mean(axis=0) >= 0.5).astype(np.int8)
         for start, stop in zip(starts, stops)]
    )
    raw_total = block_values[boundaries[0]:boundaries[-1]].sum(axis=1)
    block_truth = block_reference[boundaries[0]:boundaries[-1]]
    result = {
        "name": name,
        "n_segments": len(weights),
        "n_binary_variables": int(reference.size),
        "boundaries_blocks": boundaries.tolist(),
        "segment_weights_blocks": weights.tolist(),
        "reference_states": reference.tolist(),
        "n_covered_blocks": int(weights.sum()),
        "modes": {},
    }
    for mode, block_aggregate in mode_aggregates.items():
        aggregate = np.asarray(
            [block_aggregate[start:stop].mean() for start, stop in zip(starts, stops)]
        )
        exact = solve_binary_temporal_exact(aggregate, powers, penalties, weights)
        states = exact.states
        expanded = np.repeat(states, weights, axis=0)
        physical_prediction = baselines.sum() + expanded @ powers
        matches = states == reference
        reconstruction = weights * (aggregate - states @ powers) ** 2
        switching = np.sum(penalties * np.diff(states, axis=0) ** 2, axis=1)
        ref_reconstruction = weights * (aggregate - reference @ powers) ** 2
        ref_switching = np.sum(penalties * np.diff(reference, axis=0) ** 2, axis=1)
        mode_result = {
            "aggregate_segment_w": aggregate.tolist(),
            "exact_states": states.tolist(),
            "segment_proxy_agreement": agreement(int(matches.sum()), matches.size),
            "duration_weighted_segment_proxy_agreement": agreement(
                int(np.sum(matches * weights[:, None])), int(weights.sum() * len(powers))
            ),
            "block_proxy_agreement": agreement(
                int(np.sum(expanded == block_truth)), block_truth.size
            ),
            "physical_raw_selected_total_mae_w": float(
                np.mean(np.abs(physical_prediction - raw_total))
            ),
            "objective_energy_within_mode": exact.energy,
            "reconstruction_energy_by_segment": reconstruction.tolist(),
            "switching_energy_by_edge": switching.tolist(),
            "reference_objective_energy_within_mode": float(
                ref_reconstruction.sum() + ref_switching.sum()
            ),
            "reference_reconstruction_energy_by_segment": ref_reconstruction.tolist(),
            "reference_switching_energy_by_edge": ref_switching.tolist(),
            "dp_states_per_segment": exact.states_per_segment,
            "dp_wall_time_s_single_run": exact.wall_time_s,
            "dp_traceback_bytes": exact.traceback_bytes,
        }
        if name == "frozen_pilot_3_segments":
            qubo = build_binary_temporal_qubo(aggregate, powers, penalties, weights)
            bits, energies = qubo.energies()
            minimum = float(energies.min())
            minimizers = np.isclose(energies, minimum, rtol=0.0, atol=1e-6)
            if not np.isclose(exact.energy, minimum, rtol=1e-10, atol=1e-6):
                raise AssertionError("Dynamic programming disagrees with exhaustive QUBO")
            if not any(np.array_equal(state, states.reshape(-1)) for state in bits[minimizers]):
                raise AssertionError("Dynamic programming did not return a QUBO minimizer")
            mode_result["exhaustive_qubo_crosscheck"] = {
                "n_enumerated_states": len(bits),
                "n_minimizers": int(minimizers.sum()),
                "minimum_energy": minimum,
                "dp_energy_absolute_difference": abs(exact.energy - minimum),
                "passed": True,
            }
        result["modes"][mode] = mode_result
    return result


def markdown(report: dict) -> str:
    lines = [
        "# Q.NILM representation correction audit",
        "",
        "Retrospective classical ablation using the historical pilot's frozen powers, "
        "baselines, thresholds, switch penalties, and event boundaries. No quantum jobs "
        "were submitted. Both representations are solved exactly by temporal dynamic "
        "programming; both nine-variable pilot solutions also pass exhaustive QUBO checks.",
        "",
        "The legacy transformation sums positive-clipped channel residuals. The corrected "
        "transformation preserves signed baseline residuals: aggregate = sum(channel "
        "measurements) - sum(baselines). In physical units, the prediction is "
        "sum(baselines) + powers @ states.",
        "",
        "| Case | Representation | Segment proxy agreement | Duration-weighted segment proxy agreement | Block proxy agreement | Raw selected-total MAE (W) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        for mode, result in case["modes"].items():
            row = [case["name"], mode]
            for metric in ("segment_proxy_agreement", "duration_weighted_segment_proxy_agreement", "block_proxy_agreement"):
                value = result[metric]
                row.append(f'{value["correct"]}/{value["total"]} ({100 * value["fraction"]:.2f}%)')
            row.append(f'{result["physical_raw_selected_total_mae_w"]:.3f}')
            lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "Segment agreement treats each compressed interval equally. Duration weighting "
        "repeats each interval's majority proxy over its duration. Block agreement compares "
        "expanded predictions with the original 30-second block proxies and exposes "
        "transitions hidden by majority labels. Raw selected-total MAE always uses "
        "unmodified block-mean channel measurements on the same intervals; it is comparable "
        "between representations within each case. Objective energies refer to different "
        "encoded targets and must not be interpreted as a cross-representation improvement.",
        "",
        "## First pilot interval",
        "",
        "| Channel | Mean residual (W) | Positive-clipping bias (W) | Block proxy ON fraction |",
        "|---|---:|---:|---:|",
    ]
    first = report["first_pilot_interval"]
    for channel, values in first["channels"].items():
        lines.append(
            f'| {channel} | {values["mean_signed_residual_w"]:.6f} | '
            f'{values["positive_clipping_bias_w"]:.6f} | {values["reference_on_fraction"]:.3f} |'
        )
    lines += [
        "",
        f'The interval aggregate changes from {first["legacy_aggregate_w"]:.6f} W '
        f'to {first["signed_aggregate_w"]:.6f} W. The excess is '
        f'{first["total_positive_clipping_bias_w"]:.6f} W.',
        "",
        "## Scope and limitations",
        "",
        *["- " + caveat for caveat in report["caveats"]],
        "",
        f'Input SHA-256 verified: `{report["input_sha256"]}`. '
        f'{report["data_checks"]["n_rows"]} contiguous one-second rows; '
        f'{report["data_checks"]["negative_signed_aggregate_blocks"]} of '
        f'{report["data_checks"]["n_blocks"]} centered blocks are negative. '
        "These are residuals relative to fitted baselines, not negative physical consumption.",
        "",
        "Reproduce from the repository root with:",
        "",
        "```sh",
        "PYTHONPATH=src .venv/bin/python scripts/run_representation_ablation.py --output-dir /tmp/qnilm-representation-new",
        "```",
        "",
        "The output directory must not already exist. Full states, boundaries, energies, "
        "timings, and verification details are recorded in `summary.json`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-summary", type=Path, default=ROOT / "results/pilot/summary.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/representation_correction")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    summary_bytes = args.pilot_summary.read_bytes()
    pilot = json.loads(summary_bytes)
    input_path = Path(pilot["input"])
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if input_hash != pilot["input_sha256"]:
        raise ValueError("Input SHA-256 does not match the frozen pilot")
    channels = pilot["channels"]
    timestamps, values = load_window(input_path, channels)
    if timestamps.size < 2 or not np.all(np.diff(timestamps) == 1):
        raise ValueError("This diagnostic requires contiguous one-second input rows")
    seconds = int(pilot["block_seconds"])
    if timestamps.size % seconds:
        raise ValueError("Diagnostic input must contain complete blocks")
    blocks = block_mean(values, seconds)
    if not np.all(np.isfinite(blocks)) or len(blocks) != pilot["input_blocks"]:
        raise ValueError("Invalid data or block count differs from frozen pilot")
    powers, baselines, thresholds, penalties = (
        np.asarray([pilot[field][channel] for channel in channels], dtype=float)
        for field in ("nominal_incremental_power_w", "baseline_power_w", "state_threshold_w", "switch_penalty")
    )
    reference = (blocks > thresholds).astype(np.int8)
    centered = blocks - baselines
    aggregates = {
        "legacy_clipped": np.maximum(centered, 0.0).sum(axis=1),
        "signed_baseline": centered.sum(axis=1),
    }
    boundaries = legacy_boundaries(aggregates["legacy_clipped"], pilot["event_threshold_w"])
    if len(boundaries) - 1 != pilot["detected_event_segments"]:
        raise ValueError("Reconstructed legacy event count differs from pilot")
    selected_start_rows = np.flatnonzero(timestamps == pilot["selected_start_unix"])
    if selected_start_rows.size != 1 or selected_start_rows[0] % seconds:
        raise ValueError("Frozen pilot start is missing or not aligned to a block")
    start = int(selected_start_rows[0] // seconds)
    selected = np.r_[start, start + np.cumsum(pilot["selected_segment_durations_blocks"])]
    selected_index = np.flatnonzero(boundaries == start)
    if selected_index.size != 1 or not np.array_equal(
        selected, boundaries[selected_index[0]:selected_index[0] + len(selected)]
    ):
        raise ValueError("Frozen pilot intervals differ from reconstructed legacy boundaries")
    cases = [
        summarize_case(name, edges, blocks, reference, baselines, powers, penalties, aggregates)
        for name, edges in (
            ("frozen_pilot_3_segments", selected),
            ("full_hour_16_event_segments", boundaries),
            ("full_hour_120_regular_blocks", np.arange(len(blocks) + 1)),
        )
    ]
    frozen_case = cases[0]
    legacy_pilot = frozen_case["modes"]["legacy_clipped"]
    if not np.allclose(legacy_pilot["aggregate_segment_w"], pilot["aggregate_segment_power_w"], rtol=0, atol=1e-9):
        raise ValueError("Legacy aggregate does not reproduce the frozen pilot")
    if frozen_case["reference_states"] != pilot["reference_states"] or legacy_pilot["exact_states"] != pilot["exact_states"]:
        raise ValueError("Legacy labels or exact states do not reproduce the frozen pilot")
    first_slice = slice(selected[0], selected[1])
    first_bias = np.maximum(-centered[first_slice], 0.0).mean(axis=0)
    first_signed = centered[first_slice].mean(axis=0)
    first_fractions = reference[first_slice].mean(axis=0)
    report = {
        "algorithm": "Q.NILM",
        "status": "retrospective classical representation audit; no quantum-advantage claim",
        "frozen_pilot_summary": str(args.pilot_summary.relative_to(ROOT)) if args.pilot_summary.is_relative_to(ROOT) else str(args.pilot_summary),
        "frozen_pilot_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "input": pilot["input"],
        "input_sha256": input_hash,
        "channels": channels,
        "block_seconds": seconds,
        "frozen_calibration": {field: pilot[field] for field in (
            "nominal_incremental_power_w", "baseline_power_w", "state_threshold_w", "switch_penalty", "event_threshold_w"
        )},
        "original_event_boundaries_blocks": boundaries.tolist(),
        "signed_detector_boundaries_match_original": bool(np.array_equal(
            boundaries, legacy_boundaries(aggregates["signed_baseline"], pilot["event_threshold_w"])
        )),
        "data_checks": {
            "input_sha256_verified": True,
            "n_rows": len(timestamps),
            "contiguous_one_second_timestamps": True,
            "n_blocks": len(blocks),
            "negative_signed_aggregate_blocks": int(np.sum(aggregates["signed_baseline"] < 0)),
            "minimum_signed_aggregate_block_w": float(aggregates["signed_baseline"].min()),
            "frozen_pilot_reproduced": True,
        },
        "first_pilot_interval": {
            "start_block": int(selected[0]),
            "stop_block_exclusive": int(selected[1]),
            "legacy_aggregate_w": float(aggregates["legacy_clipped"][first_slice].mean()),
            "signed_aggregate_w": float(aggregates["signed_baseline"][first_slice].mean()),
            "total_positive_clipping_bias_w": float(first_bias.sum()),
            "channels": {channel: {
                "mean_signed_residual_w": float(first_signed[index]),
                "positive_clipping_bias_w": float(first_bias[index]),
                "reference_on_fraction": float(first_fractions[index]),
            } for index, channel in enumerate(channels)},
        },
        "metric_definitions": {
            "segment_proxy_agreement": "Equal weight per appliance and interval; reference is interval majority of frozen block proxies, ties ON.",
            "duration_weighted_segment_proxy_agreement": "Interval-majority agreement weighted by interval duration in 30-second blocks.",
            "block_proxy_agreement": "Interval predictions expanded to original blocks and compared with original block proxies.",
            "physical_raw_selected_total_mae_w": "Mean absolute difference at block grain between raw sum of selected measured channels and sum(baselines) + powers @ expanded_states.",
            "objective_energy_within_mode": "Energy of the encoded target for this representation only; energies across modes are not an improvement metric.",
            "dp_wall_time_s_single_run": "One local solve measured by the DP solver; diagnostic timing, not a controlled speed benchmark.",
        },
        "cases": cases,
        "caveats": [
            "This is a retrospective correction on the same hour used to fit appliance powers, baselines, and proxy thresholds; it is not held-out accuracy evidence.",
            "Reference labels are deterministic circuit-derived proxies, not manual appliance annotations; interval-majority labels can conceal transitions.",
            "The aggregate is the controlled sum of dryer, refrigerator, and vacuum channels, not the measured whole-house main channel.",
            "The original diagnostic pilot was selected using reference state diversity and changes. Its agreement is not representative of unselected data.",
            "A binary low/high dryer model does not capture all motor, heater, and transient operating states. Correcting clipping does not establish that the representation is adequate across the dataset.",
            "Historical IBM measurements executed the original encoded objective. Corrected-objective hardware results require a new circuit and new measurements.",
            "The 48- and 360-variable cases have only three appliances and eight joint states per interval, allowing exact temporal dynamic programming; total binary-variable count alone is not evidence of classical difficulty.",
            "Objective energies differ because encoded targets differ; assess changes using common raw-power error and fixed-reference agreement, and avoid a quantum-advantage claim.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "summary.md").write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
