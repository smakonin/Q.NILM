#!/usr/bin/env python3
"""Benchmark Q.NILM QUBOs with local Ocean samplers or optional Leap solvers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from quantum_nilm.ocean import (
    estimate_zephyr_embedding,
    logical_problem_profile,
    ocean_versions,
    sample_qubo,
    serialize_sampleset,
)
from quantum_nilm.qubo import BinaryTemporalQUBO, build_binary_temporal_qubo
from quantum_nilm.r1hz import (
    DEFAULT_CHANNELS,
    block_mean,
    event_compress,
    infer_binary_reference,
    load_window,
)


DEFAULT_SCENARIOS = ((3, 4), (3, 8), (6, 8), (6, 16), (12, 16), (18, 16))
FIELDNAMES = (
    "family",
    "instance",
    "reference_type",
    "appliances",
    "segments",
    "logical_variables",
    "quadratic_terms",
    "edge_density",
    "maximum_degree",
    "coefficient_dynamic_range",
    "instance_seed",
    "sampler_seed",
    "sampler",
    "num_reads",
    "num_sweeps",
    "wall_time_s",
    "best_energy",
    "reference_energy",
    "best_known_energy",
    "best_known_source",
    "relative_best_known_gap",
    "best_known_hit",
    "exact_energy",
    "certified_relative_gap",
    "certified_optimum_hit",
    "bit_accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "aggregate_mae_w",
    "occurrences",
    "backend_name",
    "physical_qubits",
    "maximum_chain_length",
    "mean_chain_break_fraction",
    "qpu_access_time_us",
)


@dataclass(frozen=True)
class BenchmarkInstance:
    family: str
    name: str
    reference_type: str
    instance_seed: int | None
    aggregate: np.ndarray
    powers: np.ndarray
    reference: np.ndarray
    weights: np.ndarray
    qubo: BinaryTemporalQUBO
    source: dict[str, Any]


def parse_scenarios(value: str) -> tuple[tuple[int, int], ...]:
    scenarios: list[tuple[int, int]] = []
    for item in value.split(","):
        try:
            appliances, segments = (int(part) for part in item.lower().split("x"))
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                "scenarios must look like 3x4,6x8"
            ) from exc
        if appliances <= 0 or segments <= 0:
            raise argparse.ArgumentTypeError("scenario dimensions must be positive")
        scenarios.append((appliances, segments))
    return tuple(scenarios)


def synthetic_instance(
    n_appliances: int, n_segments: int, seed: int
) -> BenchmarkInstance:
    """Construct a deterministic, noisy temporal NILM scaling instance."""
    rng = np.random.default_rng(seed + 10_000 * n_appliances + n_segments)
    powers = np.geomspace(70.0, 5000.0, n_appliances)
    powers *= rng.uniform(0.94, 1.06, n_appliances)
    states = np.empty((n_segments, n_appliances), dtype=np.int8)
    states[0] = rng.integers(0, 2, n_appliances)
    for segment in range(1, n_segments):
        flips = rng.random(n_appliances) < 0.18
        if not np.any(flips):
            flips[rng.integers(0, n_appliances)] = True
        states[segment] = np.logical_xor(states[segment - 1], flips)
    noise = rng.normal(0.0, 0.005 * float(np.max(powers)), n_segments)
    aggregate = np.maximum(states @ powers + noise, 0.0)
    weights = rng.integers(1, 9, n_segments).astype(float)
    qubo = build_binary_temporal_qubo(
        aggregate,
        powers,
        0.02 * powers**2,
        segment_weights=weights,
    )
    return BenchmarkInstance(
        family="synthetic",
        name=f"synthetic-{n_appliances}x{n_segments}-seed-{seed}",
        reference_type="generated appliance states",
        instance_seed=seed,
        aggregate=aggregate,
        powers=powers,
        reference=states,
        weights=weights,
        qubo=qubo,
        source={},
    )


def r1hz_instance(
    path: Path,
    block_seconds: int,
    event_threshold_ratio: float,
    aggregate_mode: str = "signed",
) -> BenchmarkInstance:
    timestamps, second_values = load_window(path, DEFAULT_CHANNELS)
    blocks = block_mean(second_values, block_seconds)
    states, powers, baselines, _ = infer_binary_reference(blocks)
    _, aggregate, reference, weights, _, threshold = event_compress(
        blocks,
        states,
        powers,
        baselines,
        event_threshold_ratio,
        aggregate_mode=aggregate_mode,
    )
    qubo = build_binary_temporal_qubo(
        aggregate,
        powers,
        0.02 * powers**2,
        segment_weights=weights,
    )
    return BenchmarkInstance(
        family="R1Hz",
        name=f"r1hz-{int(timestamps[0])}-{len(reference)}-event-segments",
        reference_type="circuit-derived binary proxy states",
        instance_seed=None,
        aggregate=aggregate,
        powers=powers,
        reference=reference,
        weights=weights,
        qubo=qubo,
        source={
            "dataset_doi": "10.7910/DVN/RCB5VJ",
            "input": str(path),
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "channels": list(DEFAULT_CHANNELS),
            "synthetic_marker_rows": "excluded",
            "block_seconds": block_seconds,
            "event_threshold_ratio": event_threshold_ratio,
            "event_threshold_w": threshold,
            "aggregate_mode": aggregate_mode,
        },
    )


def exact_certificate(
    qubo: BinaryTemporalQUBO, maximum_variables: int
) -> tuple[float | None, np.ndarray | None]:
    if qubo.n_variables > min(maximum_variables, 24):
        return None, None
    bits, energies = qubo.energies()
    index = int(np.argmin(energies))
    return float(energies[index]), bits[index]


def classification_metrics(
    reference: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    truth = np.asarray(reference, dtype=np.int8).reshape(-1)
    estimate = np.asarray(predicted, dtype=np.int8).reshape(-1)
    tp = int(np.sum((truth == 1) & (estimate == 1)))
    tn = int(np.sum((truth == 0) & (estimate == 0)))
    fp = int(np.sum((truth == 0) & (estimate == 1)))
    fn = int(np.sum((truth == 1) & (estimate == 0)))
    precision = tp / (tp + fp) if tp + fp else float(tp + fn == 0)
    recall = tp / (tp + fn) if tp + fn else float(tp + fp == 0)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else float(np.array_equal(truth, estimate))
    return {
        "bit_accuracy": float(np.mean(truth == estimate)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc),
    }


def relative_gap(value: float, reference: float) -> float:
    return float((value - reference) / max(abs(reference), 1.0))


def git_record() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments], capture_output=True, text=True, check=False
        )
        return completed.stdout.strip() if completed.returncode == 0 else "unavailable"

    tracked_diff = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"], check=False
    ).returncode
    return {"commit": run("rev-parse", "HEAD"), "tracked_changes": tracked_diff != 0}


def qpu_access_time(metadata: dict[str, Any]) -> float | None:
    timing = metadata.get("timing", {})
    value = timing.get("qpu_access_time") if isinstance(timing, dict) else None
    return float(value) if value is not None else None


def run_instance(
    instance: BenchmarkInstance,
    sampler_name: str,
    sampler_seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = sample_qubo(
        instance.qubo,
        sampler_name,
        num_reads=args.num_reads,
        seed=sampler_seed,
        num_sweeps=args.num_sweeps,
        tabu_timeout_ms=args.tabu_timeout_ms,
        solver=args.solver,
        annealing_time_us=args.annealing_time_us,
        chain_strength=args.chain_strength,
        hybrid_time_limit_s=args.hybrid_time_limit_s,
        label=f"Q.NILM {instance.name}",
    )
    states = instance.qubo.decode(result.bits)
    metrics = classification_metrics(instance.reference, states)
    profile = logical_problem_profile(instance.qubo)
    row: dict[str, Any] = {
        "family": instance.family,
        "instance": instance.name,
        "reference_type": instance.reference_type,
        "appliances": instance.qubo.n_appliances,
        "segments": instance.qubo.n_segments,
        **profile,
        "instance_seed": instance.instance_seed,
        "sampler_seed": (
            sampler_seed if result.sampler in {"simulated", "tabu"} else None
        ),
        "sampler": result.sampler,
        "num_reads": args.num_reads if result.sampler != "hybrid" else None,
        "num_sweeps": args.num_sweeps if result.sampler == "simulated" else None,
        "wall_time_s": result.wall_time_s,
        "best_energy": result.energy,
        "reference_energy": instance.qubo.energy(instance.reference.reshape(-1)),
        **metrics,
        "aggregate_mae_w": float(
            np.average(
                np.abs(instance.aggregate - states @ instance.powers),
                weights=instance.weights,
            )
        ),
        "occurrences": result.occurrences,
        "backend_name": result.metadata.get("backend_name"),
        "physical_qubits": result.metadata.get("physical_qubits"),
        "maximum_chain_length": result.metadata.get("maximum_chain_length"),
        "mean_chain_break_fraction": result.metadata.get(
            "mean_chain_break_fraction"
        ),
        "qpu_access_time_us": qpu_access_time(result.metadata),
    }
    raw = {
        "instance": instance.name,
        "sampler_seed": sampler_seed,
        "logical_profile": profile,
        "execution_metadata": result.metadata,
        "sampleset": serialize_sampleset(result.sampleset),
    }
    return row, raw


def complete_gaps(
    rows: list[dict[str, Any]], certificates: dict[str, float | None]
) -> None:
    by_instance: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_instance.setdefault(str(row["instance"]), []).append(row)
    for name, subset in by_instance.items():
        exact = certificates[name]
        candidates = [
            (float(row["best_energy"]), str(row["sampler"])) for row in subset
        ]
        candidates.append((float(subset[0]["reference_energy"]), "reference state"))
        if exact is not None:
            candidates.append((exact, "exact enumeration"))
        best_known, source = min(candidates, key=lambda item: item[0])
        for row in subset:
            energy = float(row["best_energy"])
            row["best_known_energy"] = best_known
            row["best_known_source"] = source
            row["relative_best_known_gap"] = relative_gap(energy, best_known)
            row["best_known_hit"] = bool(np.isclose(energy, best_known, atol=1e-7))
            row["exact_energy"] = exact
            row["certified_relative_gap"] = (
                relative_gap(energy, exact) if exact is not None else None
            )
            row["certified_optimum_hit"] = (
                bool(np.isclose(energy, exact, atol=1e-7))
                if exact is not None
                else None
            )


def aggregate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["family"]), int(row["logical_variables"]), str(row["sampler"]))
        groups.setdefault(key, []).append(row)
    summary = []
    for (family, variables, sampler), subset in sorted(groups.items()):
        certified = [
            row for row in subset if row["certified_optimum_hit"] is not None
        ]
        summary.append(
            {
                "family": family,
                "logical_variables": variables,
                "sampler": sampler,
                "runs": len(subset),
                "mean_bit_accuracy": float(np.mean([row["bit_accuracy"] for row in subset])),
                "mean_f1": float(np.mean([row["f1"] for row in subset])),
                "mean_mcc": float(np.mean([row["mcc"] for row in subset])),
                "mean_aggregate_mae_w": float(np.mean([row["aggregate_mae_w"] for row in subset])),
                "mean_wall_time_s": float(np.mean([row["wall_time_s"] for row in subset])),
                "best_known_hit_rate": float(np.mean([row["best_known_hit"] for row in subset])),
                "certified_runs": len(certified),
                "certified_optimum_hit_rate": (
                    float(np.mean([row["certified_optimum_hit"] for row in certified]))
                    if certified
                    else None
                ),
            }
        )
    return summary


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Q.NILM D-Wave Ocean scaling benchmark",
        "",
        f"Status: {summary['status']}",
        "",
        "| Data | Logical variables | Sampler | Runs | Bit accuracy | F1 | MCC | Aggregate MAE (W) | Wall time (s) | Best-known hit rate |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["groups"]:
        lines.append(
            "| {family} | {logical_variables} | {sampler} | {runs} | "
            "{mean_bit_accuracy:.3f} | {mean_f1:.3f} | {mean_mcc:.3f} | "
            "{mean_aggregate_mae_w:.2f} | {mean_wall_time_s:.4f} | "
            "{best_known_hit_rate:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Offline embedding estimates",
            "",
            "| Data | Logical variables | Logical couplers | Ideal physical qubits | Maximum chain | Result |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for estimate in summary["embedding_estimates"]:
        physical = estimate["physical_qubits"]
        maximum_chain = estimate["maximum_chain_length"]
        lines.append(
            f"| {estimate['family']} | {estimate['logical_variables']} | "
            f"{estimate['logical_couplers']} | "
            f"{physical if physical is not None else '---'} | "
            f"{maximum_chain if maximum_chain is not None else '---'} | "
            f"{'embedded' if estimate['success'] else 'not found in timeout'} |"
        )
    lines.extend(
        [
            "",
            "Best-known values combine tested solver samples and the reference state; "
            "they are certified optima only where exact enumeration is reported.",
            "R1Hz labels are deterministic circuit-derived proxies, not manual appliance ground truth.",
            "Embedding figures use an ideal defect-free Zephyr(12) graph and are not live-QPU results.",
            "Direct QPU and Leap hybrid runs must be reported separately from these local classical results.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_checksums(output_dir: Path) -> None:
    """Write a stable SHA-256 manifest for every published benchmark artifact."""
    artifacts = [
        output_dir / "instances.csv",
        output_dir / "embedding_estimates.json",
        output_dir / "summary.json",
        output_dir / "summary.md",
        *sorted((output_dir / "raw").glob("*.json")),
    ]
    lines = []
    for artifact in artifacts:
        if artifact.exists():
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            lines.append(f"{digest}  {artifact.relative_to(output_dir)}")
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenarios", type=parse_scenarios, default=DEFAULT_SCENARIOS)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--samplers",
        nargs="+",
        choices=("simulated", "tabu", "qpu", "hybrid"),
        default=("simulated", "tabu"),
    )
    parser.add_argument("--num-reads", type=int, default=32)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--tabu-timeout-ms", type=int, default=50)
    parser.add_argument("--exact-max-variables", type=int, default=18)
    parser.add_argument("--r1hz-input", type=Path)
    parser.add_argument("--block-seconds", type=int, default=30)
    parser.add_argument("--event-threshold-ratio", type=float, default=0.01)
    parser.add_argument(
        "--aggregate-mode", choices=("signed", "legacy-clipped"), default="signed"
    )
    parser.add_argument("--solver")
    parser.add_argument("--annealing-time-us", type=float)
    parser.add_argument("--chain-strength", type=float)
    parser.add_argument("--hybrid-time-limit-s", type=float)
    parser.add_argument(
        "--zephyr-m",
        type=int,
        default=12,
        help="ideal Zephyr topology size for offline embedding estimates; 0 disables",
    )
    parser.add_argument("--embedding-timeout-s", type=int, default=5)
    parser.add_argument(
        "--save-raw", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive")

    instances = [
        synthetic_instance(appliances, segments, seed)
        for appliances, segments in args.scenarios
        for seed in range(args.seeds)
    ]
    if args.r1hz_input:
        measured = r1hz_instance(
            args.r1hz_input, args.block_seconds, args.event_threshold_ratio,
            args.aggregate_mode,
        )
        instances.extend([measured] * args.seeds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    if args.save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    certificates: dict[str, float | None] = {}
    instance_sources: dict[str, dict[str, Any]] = {}
    for run_index, instance in enumerate(instances):
        if instance.name not in certificates:
            certificates[instance.name], _ = exact_certificate(
                instance.qubo, args.exact_max_variables
            )
            instance_sources[instance.name] = instance.source
        sampler_seed = (
            instance.instance_seed
            if instance.instance_seed is not None
            else run_index % args.seeds
        )
        for sampler_name in args.samplers:
            row, raw = run_instance(instance, sampler_name, sampler_seed, args)
            rows.append(row)
            if args.save_raw:
                filename = f"{instance.name}-{sampler_name}-seed-{sampler_seed}.json"
                (raw_dir / filename).write_text(json.dumps(raw, indent=2) + "\n")
            print(
                f"{instance.name}: {sampler_name}, n={instance.qubo.n_variables}, "
                f"energy={row['best_energy']:.6g}, time={row['wall_time_s']:.3f}s",
                flush=True,
            )
    complete_gaps(rows, certificates)

    embedding_estimates: list[dict[str, Any]] = []
    if args.zephyr_m > 0:
        representatives: dict[tuple[str, int], BenchmarkInstance] = {}
        for instance in instances:
            representatives.setdefault(
                (instance.family, instance.qubo.n_variables), instance
            )
        for instance in representatives.values():
            estimate = estimate_zephyr_embedding(
                instance.qubo,
                zephyr_m=args.zephyr_m,
                seed=7,
                timeout_s=args.embedding_timeout_s,
            )
            estimate["family"] = instance.family
            estimate["instance"] = instance.name
            embedding_estimates.append(estimate)
            print(
                f"{instance.name}: {estimate['topology']}, "
                f"embedding_success={estimate['success']}, "
                f"physical_qubits={estimate['physical_qubits']}",
                flush=True,
            )
        (args.output_dir / "embedding_estimates.json").write_text(
            json.dumps(embedding_estimates, indent=2) + "\n"
        )

    with (args.output_dir / "instances.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in FIELDNAMES} for row in rows)

    remote = any(name in {"qpu", "hybrid"} for name in args.samplers)
    summary: dict[str, Any] = {
        "algorithm": "Q.NILM",
        "status": (
            "includes D-Wave remote execution; inspect sampler labels and timing"
            if remote
            else "classical Ocean readiness benchmark; no QPU execution"
        ),
        "interpretation": (
            "D-Wave quantum annealing samples the shared QUBO/Ising objective; "
            "it does not execute the gate-model QAOA circuit. No quantum-advantage claim."
        ),
        "configuration": {
            "scenarios": [list(item) for item in args.scenarios],
            "seeds": args.seeds,
            "samplers": list(args.samplers),
            "num_reads": args.num_reads,
            "num_sweeps": args.num_sweeps,
            "tabu_timeout_ms": args.tabu_timeout_ms,
            "exact_max_variables": args.exact_max_variables,
            "zephyr_m": args.zephyr_m,
            "embedding_timeout_s": args.embedding_timeout_s,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "ocean_packages": ocean_versions(),
            "git": git_record(),
        },
        "instance_sources": instance_sources,
        "total_solver_runs": len(rows),
        "embedding_estimates": [
            {key: value for key, value in estimate.items() if key != "embedding"}
            for estimate in embedding_estimates
        ],
        "groups": aggregate_summary(rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    write_markdown(args.output_dir / "summary.md", summary)
    write_checksums(args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
