#!/usr/bin/env python3
"""Post-hoc measurement-distribution distance from the initial feasible W law.

This diagnostic is additive and explicitly post-hoc. It never changes frozen
training, selected angles, probability distributions, primary comparisons, or
the original protocol. It compares measurement probabilities, not quantum
state vectors or phase-sensitive state fidelity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UNIFORMITY_TOLERANCE = 1e-10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tv_to_initial_w(feasible_probabilities, invalid_probability):
    """Full physical TV distance, with all invalid mass compared against zero.

    Initial W measurements assign 1/M to each feasible state and zero to every
    invalid state. Thus exact TV is 0.5*(sum_feasible|p_i-1/M|+invalid_mass).
    Combining invalid outcomes is lossless for this reference distribution.
    """
    probabilities = np.asarray(feasible_probabilities, dtype=float)
    invalid = float(invalid_probability)
    if probabilities.ndim != 1 or not probabilities.size or not np.all(np.isfinite(probabilities)):
        raise ValueError("Expected a nonempty finite feasible probability vector")
    if np.any(probabilities < 0) or not np.isfinite(invalid) or not 0 <= invalid <= 1:
        raise ValueError("Probabilities must be nonnegative and invalid mass in [0,1]")
    if abs(float(probabilities.sum()) + invalid - 1.0) > 1e-9:
        raise ValueError("All-shot probability mass must sum to one")
    return float(0.5 * (np.abs(probabilities - 1 / len(probabilities)).sum() + invalid))


def diagnose(run):
    protocol_path, summary_path = run / "protocol.json", run / "summary.json"
    protocol = json.loads(protocol_path.read_text())
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "complete" or summary["protocol_sha256"] != sha(protocol_path):
        raise ValueError("Completed matching mixer summary required")
    files = sorted((run / "distributions").glob("*.json"))
    expected = {f"{source['id']}_p{depth}_{arm}" for source in protocol["selected_inputs"] for depth in protocol["depths"] for arm in protocol["arms"]}
    records, source_hashes = [], {}
    for path in files:
        value = json.loads(path.read_text())
        if value["protocol_sha256"] != sha(protocol_path):
            raise ValueError("Distribution uses a different frozen protocol")
        distance = tv_to_initial_w(value["probabilities"], value["invalid_probability"])
        relative = str(path.relative_to(ROOT))
        source_hashes[relative] = sha(path)
        records.append({
            "distribution_id": value["distribution_id"], "instance_id": value["instance_id"],
            "state_counts": value["state_counts"], "segments": value["segments"],
            "qubits": value["qubits"], "measurement_noise": value["measurement_noise"],
            "depth": value["depth"], "arm": value["arm"], "seed": value["seed"],
            "total_variation_to_initial_w_measurement_distribution": distance,
            "within_numerical_uniformity_tolerance": bool(distance <= UNIFORMITY_TOLERANCE),
            "feasible_probability": value["feasible_probability"],
            "invalid_probability": value["invalid_probability"],
            "gammas": value["gammas"], "betas": value["betas"],
            "source_path": relative, "source_sha256": source_hashes[relative],
        })
    if len(records) != len(expected) or {value["distribution_id"] for value in records} != expected:
        raise ValueError("Diagnostic must include every frozen arm/depth/input condition")
    grouped = {}
    for value in records:
        key = (tuple(value["state_counts"]), value["segments"], value["measurement_noise"], value["depth"], value["arm"])
        grouped.setdefault(key, []).append(value)
    groups = []
    for key, values in sorted(grouped.items()):
        if sorted(value["seed"] for value in values) != protocol["instance_seeds"]:
            raise ValueError("Every group must retain all eight base seeds")
        counts, segments, sigma, depth, arm = key
        distances = [value["total_variation_to_initial_w_measurement_distribution"] for value in values]
        groups.append({
            "state_counts": counts, "segments": segments, "measurement_noise": sigma,
            "depth": depth, "arm": arm, "n_instances": len(values),
            "mean_total_variation": float(np.mean(distances)),
            "minimum_total_variation": min(distances), "maximum_total_variation": max(distances),
            "instances_within_numerical_uniformity_tolerance": sum(value["within_numerical_uniformity_tolerance"] for value in values),
        })
    return {
        "status": "complete_posthoc_diagnostic", "created_utc": datetime.now(timezone.utc).isoformat(),
        "posthoc": True, "protocol_sha256": sha(protocol_path), "summary_sha256": sha(summary_path),
        "generator_path": str(Path(__file__).relative_to(ROOT)), "generator_sha256": sha(__file__),
        "method": "0.5*(sum_feasible(abs(p_i-1/M))+all_invalid_probability); exact total variation to initial uniform-feasible W measurement law",
        "scope": "All720arm/depth/input conditions, including allthree measurement-noise levels; continuous distances retained, no condition selected or excluded",
        "uniformity_tolerance": UNIFORMITY_TOLERANCE,
        "tolerance_interpretation": "Post-hoc numerical descriptive cutoff only, not a statistical significance threshold; inspect exact continuous distances",
        "interpretation": "Near-zero distance means no observable change in the computational-basis measurement law. It does not imply the quantum state, phases, circuit or training trajectory is unchanged. Matching a uniform feasible classical sampler is not evidence of an optimization or quantum advantage.",
        "conditions": len(records), "groups": groups, "records": records,
        "source_sha256": source_hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/stage_d/mixer/run_001")
    args = parser.parse_args()
    run = args.run.resolve()
    target = run / "initialization_distance_diagnostic.json"
    if target.exists():
        raise FileExistsError("Refusing to overwrite an existing post-hoc diagnostic")
    result = diagnose(run)
    with target.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(target), "status": result["status"], "conditions": result["conditions"],
                      "w_x_groups": [value for value in result["groups"] if value["arm"] == "w_x_penalty"]}, indent=2))


if __name__ == "__main__":
    main()
