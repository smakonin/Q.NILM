#!/usr/bin/env python3
"""Frozen full-coverage categorical QAOA extension; no remote submissions.

The same 30 previously exposed test windows are reused without model retuning.
This script separates prospective configuration, training-only angles, test
inference, and reference scoring. QAOA here is ideal circuit simulation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from quantum_nilm.evaluation import (aggregate_boundaries, contiguous_slices,
    pool_window_metrics, predict_temporal, score_power)
from quantum_nilm.heldout_data import read_block_window

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ["dryr", "frdg", "vacu"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def model_arrays(frozen):
    model = frozen["models"]["multistate"]
    return ([np.asarray(x) for x in model["levels_w"]],
            0.1 * np.asarray(model["ranges_w"]) ** 2,
            model["event_threshold_w"])


def make_chunks(mains, timestamps, threshold):
    """Aggregate-only compression; two intervals per chunk within valid runs."""
    chunks = []
    for run_id, part in enumerate(contiguous_slices(timestamps)):
        signal = np.asarray(mains)[part]
        bounds = aggregate_boundaries(signal, threshold)
        weights = np.diff(bounds)
        means = np.add.reduceat(signal, bounds[:-1]) / weights
        for index in range(0, len(weights), 2):
            stop = min(index + 2, len(weights))
            chunks.append({"chunk": len(chunks), "run": run_id,
                "reset": index == 0, "aggregate": means[index:stop].tolist(),
                "weights": weights[index:stop].tolist(),
                "block_start": int(part.start + bounds[index]),
                "block_stop": int(part.start + bounds[stop]),
                "start_unix": int(timestamps[part.start + bounds[index]])})
    return chunks


def checked_window(protocol, original, window, channels):
    loaded = read_block_window(protocol["source"], window["start_unix"], window["end_unix"],
                               channels=channels, exclude_markers=("s", "+"))
    expected = next(x for x in original[window["split"]] if x["window"]["id"] == window["id"])
    if loaded["source_window_sha256"] != expected["source_window_sha256"]:
        raise RuntimeError(f"Original source hash changed: {window['id']}")
    if len(loaded["timestamps"]) != expected["valid_blocks"]:
        raise RuntimeError(f"Original valid block count changed: {window['id']}")
    return loaded


def freeze(output, original_dir):
    if output.exists():
        raise FileExistsError("Use a new output directory; prior experiments are immutable")
    original = read_json(original_dir / "protocol.json")
    frozen = read_json(original_dir / "frozen_models.json")
    if sha(original_dir / "protocol.json") != frozen["protocol_sha256"]:
        raise RuntimeError("Original protocol/model hash mismatch")
    if frozen["selected_parameters"]["multistate_compressed"]["rho"] != 0.1:
        raise RuntimeError("Unexpected original penalty")
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "post-exposure full-coverage QAOA extension of original held-out mains campaign",
        "source": original["source"], "original_dir": str(original_dir.resolve()),
        "original_hashes": {p: sha(original_dir / p) for p in
                            ("protocol.json", "frozen_models.json", "data_quality.json", "test_windows.json")},
        "windows": original["manifest"]["windows"],
        "model": frozen["models"]["multistate"], "rho": 0.1,
        "state_proxy_thresholds_w": frozen["state_proxy_thresholds_w"],
        "qaoa_depth": 1, "maximum_intervals_per_circuit": 2,
        "initial_state": "uniform one-hot state in each appliance/background register",
        "mixer": "ordered nearest-neighbor XY pair rotations exp(-i beta (XX+YY)/2)",
        "phase_scale": "sum of absolute nonconstant one-hot QUBO coefficients, minimum 1",
        "gamma_grid": np.linspace(0, 16 * np.pi, 33).tolist(),
        "beta_grid": np.linspace(0, np.pi, 17, endpoint=False).tolist(),
        "angle_training": "one two-interval chunk per original training window, start nearest timestamp midpoint; no boundary state; minimize mean expected cost/scale; ties first grid point",
        "simulation_seeds": [1907, 2907, 3907], "shots_per_chunk": 256,
        "decoding": "minimum objective among measured feasible states; ties smaller feasible index",
        "boundary_policy": "each method carries its own last predicted state, reset at every day or data gap",
        "hardware_zero_valid_policy": "repeat own previous state; at reset use each register's lowest-power state; count all such fallback intervals explicitly",
        "hardware_maximum_campaign_usage_s": 540, "hardware_paid_execution": False,
        "baselines": ["same-chunk exact classical", "shot-matched uniform feasible sampling", "full-run exact temporal DP"],
        "primary_metric": "macro appliance MAE on all original valid 30-second blocks",
        "statistics": "descriptive paired 24-hour-window bootstrap, 2000 resamples, seed 7301; average stochastic absolute-error sums across the three fixed seeds before bootstrap",
        "limitations": ["original test periods were already examined; not a fresh blind confirmatory test",
                        "two-interval look-ahead is an adaptation, not monolithic full-day QAOA",
                        "simulated one-hot feasible space is not quantum hardware or speed advantage",
                        "one home, sparse high-state vacuum activity, coarse background and nonphysical high-state proxy thresholds"],
    }
    output.mkdir(parents=True)
    save_json(output / "protocol.json", config)
    save_json(output / "model.json", frozen)
    print("Extension protocol frozen before training or test inference", flush=True)


def prepare_inputs(output):
    config = read_json(output / "protocol.json")
    for name, expected in config["original_hashes"].items():
        if sha(Path(config["original_dir"]) / name) != expected:
            raise RuntimeError(f"Original archive changed: {name}")
    original = read_json(Path(config["original_dir"]) / "data_quality.json")
    frozen = read_json(output / "model.json")
    _, _, threshold = model_arrays(frozen)
    records, training = [], []
    for split in ("train", "test"):
        for window in [w for w in config["windows"] if w["split"] == split]:
            # No appliance reference values are passed to training or inference.
            loaded = checked_window(config, original, window, ["main"])
            chunks = make_chunks(loaded["values"][:, 0], loaded["timestamps"], threshold)
            if not chunks:
                raise RuntimeError(f"No evaluable chunks for fixed window {window['id']}")
            if split == "train":
                midpoint = (window["start_unix"] + window["end_unix"]) / 2
                candidates = [c for c in chunks if len(c["weights"]) == 2] or chunks
                chosen = min(candidates, key=lambda c: (abs(c["start_unix"] - midpoint), c["chunk"]))
                training.append({"window_id": window["id"], **chosen,
                                 "source_window_sha256": loaded["source_window_sha256"]})
            else:
                records.append({"window": window, "blocks": len(loaded["timestamps"]),
                                "source_window_sha256": loaded["source_window_sha256"], "chunks": chunks})
        print(f"Prepared aggregate-only {split} inputs", flush=True)
    save_json(output / "training_chunks.json", training)
    save_json(output / "test_inputs.json", records)
    save_json(output / "input_hashes.json", {p: sha(output / p) for p in
              ("protocol.json", "model.json", "training_chunks.json", "test_inputs.json")})


def check_inputs(output):
    config = read_json(output / "protocol.json")
    for name, expected in config["original_hashes"].items():
        if sha(Path(config["original_dir"]) / name) != expected:
            raise RuntimeError(f"Original archive changed: {name}")
    for name, expected in read_json(output / "input_hashes.json").items():
        if sha(output / name) != expected:
            raise RuntimeError(f"Frozen experiment input changed: {name}")


def train_angles(output):
    from quantum_nilm.categorical_qaoa import prepare_categorical_problem, categorical_qaoa_probabilities
    check_inputs(output)
    config = read_json(output / "protocol.json")
    levels, penalties, _ = model_arrays(read_json(output / "model.json"))
    training = read_json(output / "training_chunks.json")
    started = perf_counter()
    losses = np.zeros((len(config["gamma_grid"]), len(config["beta_grid"])))
    for i, chunk in enumerate(training):
        problem = prepare_categorical_problem(chunk["aggregate"], levels, penalties, chunk["weights"])
        for g, gamma in enumerate(config["gamma_grid"]):
            for b, beta in enumerate(config["beta_grid"]):
                probability = categorical_qaoa_probabilities(problem, [gamma], [beta])
                losses[g, b] += float(probability @ (problem.energies / problem.scale))
        if (i + 1) % 10 == 0:
            print(f"Angle training: {i + 1}/{len(training)} original training windows", flush=True)
    losses /= len(training)
    g, b = np.unravel_index(np.argmin(losses), losses.shape)
    save_json(output / "angles.json", {"frozen_utc": datetime.now(timezone.utc).isoformat(),
        "gammas": [config["gamma_grid"][g]], "betas": [config["beta_grid"][b]],
        "loss_grid": losses.tolist(), "selected_expected_cost_over_scale": float(losses[g, b]),
        "training_instances": len(training), "objective_evaluations": int(losses.size * len(training)),
        "classical_training_wall_time_s": perf_counter() - started,
        "input_hashes": read_json(output / "input_hashes.json"),
        "label_access": False, "test_objectives_used_for_angle_selection": False})


def infer_chunks(chunks, levels, penalties, gammas, betas, method, shots, seed):
    """Inference has no reference/ground-truth argument or classical warm start."""
    from quantum_nilm.categorical_qaoa import prepare_categorical_problem, categorical_qaoa_probabilities
    if method not in ("exact_chunk", "uniform", "qaoa_ideal"):
        raise ValueError("Unknown inference method")
    rng = np.random.default_rng(seed)
    previous, records = None, []
    started = perf_counter()
    for chunk in chunks:
        if chunk["reset"]:
            previous = None
        problem = prepare_categorical_problem(chunk["aggregate"], levels, penalties,
                                             chunk["weights"], previous_states=previous)
        if method == "exact_chunk":
            chosen = int(np.argmin(problem.energies))
            ids, counts = np.array([chosen]), np.array([1])
            expected = None
        else:
            probabilities = (categorical_qaoa_probabilities(problem, gammas, betas)
                             if method == "qaoa_ideal" else np.full(len(problem.energies), 1 / len(problem.energies)))
            observed = rng.choice(len(problem.energies), size=shots, p=probabilities)
            ids, counts = np.unique(observed, return_counts=True)
            chosen = int(ids[np.argmin(problem.energies[ids])])
            expected = float(probabilities @ problem.energies)
        states = problem.states[chosen]
        records.append({"chunk": chunk["chunk"], "previous_states": None if previous is None else previous.tolist(),
                        "states": states.tolist(), "chosen_index": chosen,
                        "best_observed_energy": float(problem.energies[chosen]),
                        "conditional_exact_energy": float(np.min(problem.energies)),
                        "distribution_expected_energy": expected,
                        "raw_feasible_counts": [[int(i), int(n)] for i, n in zip(ids, counts)],
                        "logical_qubits": problem.num_qubits, "feasible_states": len(problem.energies)})
        previous = states[-1].copy()
    return records, perf_counter() - started


def expand_predictions(window, records, levels):
    if len(records) != len(window["chunks"]):
        raise ValueError("Incomplete circuit coverage")
    prediction = np.full((window["blocks"], len(levels)), np.nan)
    covered = np.zeros(window["blocks"], dtype=bool)
    for chunk, record in zip(window["chunks"], records):
        if chunk["chunk"] != record["chunk"]:
            raise ValueError("Chunk order mismatch")
        states = np.asarray(record["states"], dtype=int)
        start, stop = chunk["block_start"], chunk["block_stop"]
        if not 0 <= start < stop <= window["blocks"] or covered[start:stop].any():
            raise ValueError("Invalid or overlapping block coverage")
        if states.shape != (len(chunk["weights"]), len(levels)) or sum(chunk["weights"]) != stop - start:
            raise ValueError("States, durations and coverage disagree")
        for i, level in enumerate(levels):
            if np.any(states[:, i] < 0) or np.any(states[:, i] >= len(level)):
                raise ValueError("Out-of-range categorical state")
        powers = np.column_stack([level[states[:, i]] for i, level in enumerate(levels)])
        prediction[chunk["block_start"]:chunk["block_stop"]] = np.repeat(powers, chunk["weights"], axis=0)
        covered[start:stop] = True
    if not covered.all() or not np.all(np.isfinite(prediction)):
        raise ValueError("Incomplete or invalid block predictions")
    return prediction


def score_prediction(window, loaded, prediction, thresholds, method, seed, elapsed):
    return {"window_id": window["window"]["id"], "model": method, "seed": seed,
            "blocks": len(prediction), "solver_wall_time_s": elapsed,
            "appliances": score_power(loaded["values"][:, 1:], prediction[:, :3], loaded["timestamps"], thresholds, CHANNELS),
            "raw_aggregate_mae_w": float(np.abs(loaded["values"][:, 0] - prediction.sum(axis=1)).mean())}


def simulate(output):
    check_inputs(output)
    config, angles = read_json(output / "protocol.json"), read_json(output / "angles.json")
    for key, expected in angles["input_hashes"].items():
        if sha(output / key) != expected:
            raise RuntimeError("Angle freeze input mismatch")
    directory = output / "simulation"
    if directory.exists():
        raise FileExistsError("Simulation already started; do not silently overwrite/retest")
    directory.mkdir()
    save_json(directory / "freeze.json", {"angles_sha256": sha(output / "angles.json"),
              "protocol_sha256": sha(output / "protocol.json"), "started_utc": datetime.now(timezone.utc).isoformat(),
              "code_sha256": {p: sha(ROOT / p) for p in ("scripts/run_quantum_heldout.py", "src/quantum_nilm/categorical_qaoa.py", "src/quantum_nilm/evaluation.py")}})
    levels, penalties, threshold = model_arrays(read_json(output / "model.json"))
    source_quality = read_json(Path(config["original_dir"]) / "data_quality.json")
    records = []
    started = perf_counter()
    for index, window in enumerate(read_json(output / "test_inputs.json")):
        # Score references are loaded only after every stochastic prediction for this day is fixed.
        pending = []
        for method in ("exact_chunk", "uniform", "qaoa_ideal"):
            seeds = [config["simulation_seeds"][0]] if method == "exact_chunk" else config["simulation_seeds"]
            for seed in seeds:
                raw, elapsed = infer_chunks(window["chunks"], levels, penalties, angles["gammas"], angles["betas"],
                                            method, config["shots_per_chunk"], seed + index * 10000)
                name = f"{window['window']['id']}_{method}_{seed}.json.gz"
                with gzip.open(directory / name, "wt") as handle:
                    json.dump({"window_id": window["window"]["id"], "method": method, "seed": seed,
                               "solver_wall_time_s": elapsed, "records": raw}, handle, allow_nan=False)
                pending.append((method, seed, expand_predictions(window, raw, levels), elapsed))
        loaded = checked_window(config, source_quality, window["window"], ["main", *CHANNELS])
        for method, seed, prediction, elapsed in pending:
            records.append(score_prediction(window, loaded, prediction, config["state_proxy_thresholds_w"], method, seed, elapsed))
        prediction, info = predict_temporal(loaded["values"][:, 0], loaded["timestamps"], levels, penalties, threshold)
        records.append(score_prediction(window, loaded, prediction, config["state_proxy_thresholds_w"], "exact_full", None, info["solver_wall_time_s"]))
        print(f"Complete ideal QAOA and matched classical controls: {index + 1}/30 fixed days", flush=True)
    save_json(directory / "test_windows.json", records)
    pooled = {}
    for method, seed in sorted({(r["model"], r["seed"]) for r in records}, key=str):
        rows = [r for r in records if (r["model"], r["seed"]) == (method, seed)]
        appliances = pool_window_metrics(rows, CHANNELS)
        n = sum(r["blocks"] for r in rows)
        pooled[f"{method}/{seed}"] = {"model": method, "seed": seed, "windows": len(rows), "blocks": n,
            "appliances": appliances, "macro_appliance_mae_w": float(np.mean([a["mae_w"] for a in appliances.values()])),
            "aggregate_mae_w": sum(r["raw_aggregate_mae_w"] * r["blocks"] for r in rows) / n,
            "solver_wall_time_s": sum(r["solver_wall_time_s"] for r in rows)}
    save_json(directory / "summary.json", {"execution": "ideal feasible-subspace QAOA simulation, not QPU",
        "pooled_results": pooled, "total_wall_time_s": perf_counter() - started,
        "classical_angle_training_wall_time_s": angles["classical_training_wall_time_s"],
        "shots_per_circuit": config["shots_per_chunk"], "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "limitations": config["limitations"]})
    check_inputs(output)
    if sha(output / "angles.json") != read_json(directory / "freeze.json")["angles_sha256"]:
        raise RuntimeError("Angles changed during evaluation")
    print(json.dumps({key: value["macro_appliance_mae_w"] for key, value in pooled.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "prepare", "train", "simulate"), required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/quantum_heldout")
    parser.add_argument("--original-dir", type=Path, default=ROOT / "results/heldout_campaign")
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args.output_dir, args.original_dir)
    else:
        {"prepare": prepare_inputs, "train": train_angles, "simulate": simulate}[args.mode](args.output_dir)


if __name__ == "__main__":
    main()
