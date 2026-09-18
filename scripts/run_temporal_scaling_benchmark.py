#!/usr/bin/env python3
"""Certify the frozen Ocean synthetic instances with exact temporal DP.

Run from the repository root with PYTHONPATH=src and the Ocean extra installed.
The output directory must not already exist. No remote solver is called.
"""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import dimod
import numpy as np

from quantum_nilm.classical import solve_binary_temporal_exact
from quantum_nilm.qubo import build_binary_temporal_qubo
from run_ocean_scaling_benchmark import DEFAULT_SCENARIOS, synthetic_instance


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def function_ast(source: str, name: str) -> str:
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.dump(node, include_attributes=False)
    raise ValueError(f"Cannot find function {name}")


def verify_archive(archive: Path) -> tuple[dict, list[dict], dict]:
    manifest = archive / "checksums.sha256"
    checked = 0
    for line in manifest.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        artifact = (archive / relative).resolve()
        if not artifact.is_relative_to(archive.resolve()):
            raise ValueError("Archive checksum target escaped archive directory")
        if sha256(artifact) != expected:
            raise ValueError(f"Archive checksum mismatch: {relative}")
        checked += 1
    summary = json.loads((archive / "summary.json").read_text())
    config = summary["configuration"]
    if (
        config["scenarios"] != [list(shape) for shape in DEFAULT_SCENARIOS]
        or config["seeds"] != 3
        or config["samplers"] != ["simulated", "tabu"]
        or config["num_reads"] != 32
        or config["num_sweeps"] != 1000
        or config["tabu_timeout_ms"] != 50
    ):
        raise ValueError("Archive differs from the frozen six-scenario experiment")
    frozen_commit = summary["environment"]["git"]["commit"]
    definitions = {}
    for relative, function in [
        ("scripts/run_ocean_scaling_benchmark.py", synthetic_instance),
        ("src/quantum_nilm/qubo.py", build_binary_temporal_qubo),
    ]:
        original = git_output("show", f"{frozen_commit}:{relative}")
        original_ast = function_ast(original, function.__name__)
        current_ast = function_ast(inspect.getsource(function), function.__name__)
        if original_ast != current_ast:
            raise ValueError(f"{function.__name__} differs from the frozen definition")
        definitions[function.__name__] = hashlib.sha256(current_ast.encode()).hexdigest()
    with (archive / "instances.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["family"] == "synthetic"]
    if len(rows) != 36:
        raise ValueError("Expected 36 archived synthetic solver runs")
    return summary, rows, {
        "archive": str(archive.relative_to(ROOT)) if archive.is_relative_to(ROOT) else str(archive),
        "archive_commit": frozen_commit,
        "archive_manifest_sha256": sha256(manifest),
        "archive_artifacts_verified": checked,
        "frozen_generator_and_objective_ast_verified": True,
        "definition_ast_sha256": definitions,
    }


def full_energies(instance, samples: np.ndarray) -> np.ndarray:
    states = samples.reshape(-1, instance.qubo.n_segments, instance.qubo.n_appliances)
    return np.sum(
        instance.weights * (instance.aggregate - states @ instance.powers) ** 2,
        axis=1,
    ) + np.sum(
        0.02 * instance.powers**2 * np.diff(states, axis=1) ** 2,
        axis=(1, 2),
    )


def inspect_saved_samples(archive: Path, instance, row: dict, optimum: float) -> dict:
    seed = instance.instance_seed
    sampler = row["sampler"]
    relative = f"raw/{instance.name}-{sampler}-seed-{seed}.json"
    raw = json.loads((archive / relative).read_text())
    if (
        raw["instance"] != instance.name
        or raw["sampler_seed"] != seed
        or int(row["sampler_seed"]) != seed
        or int(row["num_reads"]) != 32
        or raw["logical_profile"]["logical_variables"] != instance.qubo.n_variables
        or raw["execution_metadata"]["sampler"] != sampler
        or (sampler == "simulated" and int(row["num_sweeps"]) != 1000)
    ):
        raise ValueError(f"Archived metadata mismatch: {relative}")
    sampleset = dimod.SampleSet.from_serializable(raw["sampleset"])
    if sampleset.vartype != dimod.BINARY or set(sampleset.variables) != set(range(instance.qubo.n_variables)):
        raise ValueError(f"Archived variable mapping mismatch: {relative}")
    samples = sampleset.record.sample[:, [sampleset.variables.index(i) for i in range(instance.qubo.n_variables)]]
    energies = full_energies(instance, samples)
    np.testing.assert_allclose(energies, sampleset.record.energy, rtol=1e-9, atol=1e-5)
    np.testing.assert_allclose(float(row["best_energy"]), energies.min(), rtol=1e-9, atol=1e-5)
    np.testing.assert_allclose(float(row["reference_energy"]), full_energies(instance, instance.reference[None])[0], rtol=1e-9, atol=1e-5)
    if int(sampleset.record.num_occurrences.sum()) != 32:
        raise ValueError(f"Archived read count mismatch: {relative}")
    tolerance = max(1e-5, 1e-9 * abs(optimum))
    if float(energies.min()) < optimum - tolerance:
        raise ValueError(f"Archived sample improves on DP optimum: {relative}")
    archived_energy = float(row["best_energy"])
    if row["exact_energy"]:
        np.testing.assert_allclose(float(row["exact_energy"]), optimum, rtol=1e-9, atol=1e-5)
    return {
        "sampler": sampler,
        "archived_best_energy": archived_energy,
        "best_energy_recomputed_direct": float(energies.min()),
        "relative_gap_to_dp": (archived_energy - optimum) / max(1.0, abs(optimum)),
        "optimal_within_tolerance": abs(archived_energy - optimum) <= tolerance,
        "archived_wall_time_s": float(row["wall_time_s"]),
        "samples_rechecked": len(energies),
        "read_occurrences_rechecked": int(sampleset.record.num_occurrences.sum()),
        "raw_file": relative,
        "raw_sha256": sha256(archive / relative),
    }


def machine_record() -> dict:
    details = {
        "python": sys.version,
        "numpy": np.__version__,
        "dimod": dimod.__version__,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
    }
    if sys.platform == "darwin":
        for name, key in [("cpu_model", "machdep.cpu.brand_string"), ("memory_bytes", "hw.memsize")]:
            result = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=False)
            details[name] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    return details


def markdown(summary: dict) -> str:
    lines = [
        "# Exact temporal optimization of frozen Q.NILM scaling instances",
        "",
        "All 18 synthetic instances have now been optimized exactly by temporal dynamic programming "
        "(subject to floating-point precision). The original Ocean archive remains unchanged.",
        "",
        "| Appliances × segments | Binary variables | Joint states/segment | Median DP time (s) | Optimum range | Worst archived SA gap | Worst archived tabu gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in summary["groups"]:
        lines.append(
            f"| {group['appliances']} × {group['segments']} | {group['logical_variables']} | "
            f"{group['states_per_segment']:,} | {group['median_dp_wall_time_s']:.6f} | "
            f"{group['optimum_energy_min']:.3f}–{group['optimum_energy_max']:.3f} | "
            f"{100 * max(0.0, group['worst_simulated_relative_gap']):.3f}% | "
            f"{100 * max(0.0, group['worst_tabu_relative_gap']):.3f}% |"
        )
    lines.extend([
        "",
        "Each shape uses the original instance seeds 0, 1, and 2. Every instance was solved "
        f"{summary['configuration']['timing_repetitions_per_instance']} times; each table time is the "
        "median across all seeds and repetitions for that shape. Timings include input validation, "
        "DP state setup, optimization, traceback, and direct objective evaluation. They exclude "
        "instance generation, imports, archive verification, and artifact writing. No QPU timing "
        "or speedup comparison is made.",
        "",
        "The objective exactly matches the frozen synthetic generator: duration-weighted squared "
        "aggregate residual plus per-appliance switching penalty 0.02 × power². Relative gap is "
        "(archived best energy − DP optimum) / max(1, |DP optimum|); percentages within floating-point "
        "tolerance of zero are displayed as zero. The archive used 32 reads, 1,000 simulated-annealing "
        "sweeps and a 50 ms tabu timeout per read.",
        "",
        "Verification: all archive checksums and the frozen generator/objective definitions were "
        "checked; all 1,152 saved synthetic samples were reevaluated against the direct objective; "
        "all three 12-variable cases matched both the archived exact energies and fresh complete "
        "enumeration. No saved sample improves on its DP optimum within the stated tolerance.",
        "",
        "For K segments and N appliances, the solver costs O(K N 2^N) time using weighted-Hamming "
        "min-plus passes and O(K 2^N) traceback storage. Consequently, the number N×K of binary "
        "variables alone is not a measure of classical hardness for this objective. These exact "
        "solutions strengthen the required classical baselines; they do not establish quantum advantage.",
        "",
        f"Machine: {summary['environment'].get('cpu_model', summary['environment']['architecture'])}; "
        f"{summary['environment']['platform']}; NumPy {summary['environment']['numpy']}.",
        "",
        "Rerun into a fresh directory:",
        "",
        "```sh",
        "PYTHONPATH=src .venv/bin/python scripts/run_temporal_scaling_benchmark.py --output-dir results/temporal_scaling_repeat",
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "results/dwave_scaling")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/temporal_scaling")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a fresh directory")
    archive = args.archive.resolve()
    frozen, archive_rows, verification = verify_archive(archive)
    records = []
    groups = []
    for n_appliances, n_segments in DEFAULT_SCENARIOS:
        members = []
        for seed in range(3):
            instance = synthetic_instance(n_appliances, n_segments, seed)
            solves = [solve_binary_temporal_exact(
                instance.aggregate, instance.powers, 0.02 * instance.powers**2,
                instance.weights,
            ) for _ in range(args.repetitions)]
            result = solves[0]
            for repeated in solves[1:]:
                np.testing.assert_array_equal(repeated.states, result.states)
                if repeated.energy != result.energy:
                    raise ValueError("Repeated exact solves disagree")
            np.testing.assert_allclose(result.energy, instance.qubo.energy(result.states), rtol=1e-9, atol=1e-5)
            enumerated_energy = None
            if instance.qubo.n_variables <= frozen["configuration"]["exact_max_variables"]:
                _, exhaustive = instance.qubo.energies()
                enumerated_energy = float(exhaustive.min())
                np.testing.assert_allclose(result.energy, enumerated_energy, rtol=1e-9, atol=1e-5)
            rows = [row for row in archive_rows if row["instance"] == instance.name]
            if len(rows) != 2 or {row["sampler"] for row in rows} != {"simulated", "tabu"}:
                raise ValueError(f"Missing or duplicate archived samplers for {instance.name}")
            comparisons = [inspect_saved_samples(archive, instance, row, result.energy) for row in rows]
            record = {
                "instance": instance.name,
                "instance_seed": seed,
                "appliances": n_appliances,
                "segments": n_segments,
                "logical_variables": instance.qubo.n_variables,
                "states_per_segment": result.states_per_segment,
                "optimum_energy": result.energy,
                "optimum_segment_state_ids": (result.states @ (1 << np.arange(n_appliances))).tolist(),
                "traceback_bytes": result.traceback_bytes,
                "dp_wall_times_s": [solve.wall_time_s for solve in solves],
                "fresh_enumerated_energy": enumerated_energy,
                "aggregate": instance.aggregate.tolist(),
                "powers": instance.powers.tolist(),
                "segment_weights": instance.weights.tolist(),
                "archived_solvers": comparisons,
            }
            records.append(record)
            members.append(record)
            print(f"{instance.name}: exact={result.energy:.6f}, median={np.median(record['dp_wall_times_s']):.6f}s", flush=True)
        comparisons = [comparison for member in members for comparison in member["archived_solvers"]]
        groups.append({
            "appliances": n_appliances,
            "segments": n_segments,
            "logical_variables": n_appliances * n_segments,
            "states_per_segment": 1 << n_appliances,
            "instances": len(members),
            "dp_solver_calls": len(members) * args.repetitions,
            "median_dp_wall_time_s": float(np.median([time for member in members for time in member["dp_wall_times_s"]])),
            "optimum_energy_min": min(member["optimum_energy"] for member in members),
            "optimum_energy_max": max(member["optimum_energy"] for member in members),
            "worst_classical_relative_gap": max(comparison["relative_gap_to_dp"] for comparison in comparisons),
            **{f"worst_{sampler}_relative_gap": max(comparison["relative_gap_to_dp"] for comparison in comparisons if comparison["sampler"] == sampler) for sampler in ["simulated", "tabu"]},
        })
    summary = {
        "algorithm": "Q.NILM exact temporal classical baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "18 synthetic instances certified by exhaustive temporal DP, subject to floating-point precision; no QPU execution",
        "configuration": {
            "shapes_appliances_by_segments": [list(shape) for shape in DEFAULT_SCENARIOS],
            "seeds": [0, 1, 2],
            "timing_repetitions_per_instance": args.repetitions,
            "penalty": "0.02 * powers**2",
            "state_encoding": "segment-major; appliance i is bit i of each segment state ID",
            "objective": "sum_k weight[k]*(aggregate[k]-powers@state[k])**2 + sum_ki 0.02*powers[i]**2*(state[k,i]-state[k-1,i])**2",
            "relative_gap": "(archived_best_energy - dp_optimum) / max(1, abs(dp_optimum))",
            "energy_check_rtol": 1e-9,
            "energy_check_atol": 1e-5,
            "timing_scope": "solver-only including validation, DP setup, optimization, traceback and direct objective evaluation; excludes generation/imports/archive verification/output",
        },
        "environment": machine_record(),
        "source": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "tracked_changes": bool(git_output("diff", "--name-only", "HEAD")),
            "source_sha256": {path: sha256(ROOT / path) for path in ["src/quantum_nilm/classical.py", "src/quantum_nilm/qubo.py", "scripts/run_ocean_scaling_benchmark.py", "scripts/run_temporal_scaling_benchmark.py"]},
        },
        "archive_verification": verification,
        "verification_counts": {
            "archived_solver_runs": sum(len(record["archived_solvers"]) for record in records),
            "saved_samples_rechecked": sum(comparison["samples_rechecked"] for record in records for comparison in record["archived_solvers"]),
            "fresh_full_enumerations": sum(record["fresh_enumerated_energy"] is not None for record in records),
        },
        "groups": groups,
        "instances": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "summary.md").write_text(markdown(summary))
    (args.output_dir / "checksums.sha256").write_text("".join(f"{sha256(args.output_dir / name)}  {name}\n" for name in ["summary.json", "summary.md"]))
    print(f"Saved verified benchmark to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
