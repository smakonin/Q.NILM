#!/usr/bin/env python3
"""Frozen offline matched-budget loss study. Never submits hardware jobs."""
import argparse
from datetime import datetime, timezone
import gzip
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from qiskit import qpy
from qiskit_ibm_runtime import RuntimeDecoder
from quantum_nilm import loss_comparison as lc, six_qubit_diagnostic as d
from quantum_nilm.categorical_qaoa import prepare_categorical_problem, categorical_qaoa_probabilities

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "results/quantum_diagnostics/physical_001/preparation_snapshot.json"
PAIRS = [(9101,9102), (9201,9202), (9301,9302), (9401,9402), (9501,9502)]
CONFIGS = {"exact8": (False, 8*np.pi), "sampled8": (True, 8*np.pi), "exact16": (False, 16*np.pi)}
SOURCES = ["src/quantum_nilm/loss_comparison.py", "scripts/run_loss_comparison.py",
           "tests/test_loss_comparison.py", "docs/loss_comparison_protocol.md",
           "src/quantum_nilm/categorical_qaoa.py", "src/quantum_nilm/categorical_ibm.py",
           "src/quantum_nilm/six_qubit_diagnostic.py", "src/quantum_nilm/stage_b.py"]


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text(), cls=RuntimeDecoder)


def save(path, data):
    if path.suffix == ".gz":
        with gzip.open(path, "xt") as handle:
            json.dump(data, handle, allow_nan=False)
    else:
        with path.open("x") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")


def save_qpy(path, circuit):
    with path.open("xb") as handle:
        qpy.dump(circuit, handle)


def problem(a=1, b=2):
    levels = [[0.,400.,900.], [0.,120.,280.]]
    return prepare_categorical_problem([levels[0][a]+levels[1][b]], levels, [1600.,400.])


def freeze(folder):
    lc.require(not folder.exists(), "Choose a fresh archive")
    snapshot = read(SNAPSHOT)
    p = problem()
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "execution": "offline only",
        "source_sha256": {name: sha(ROOT/name) for name in SOURCES},
        "input_sha256": {str(SNAPSHOT.relative_to(ROOT)): sha(SNAPSHOT)},
        "versions": {p: version(p) for p in ("numpy", "scipy", "qiskit", "qiskit-aer", "qiskit-ibm-runtime")},
        "losses": list(lc.LOSSES), "seed_pairs": PAIRS,
        "configs": {key: {"sampled": value[0], "gamma_max": float(value[1])} for key, value in CONFIGS.items()},
        "training_problem": d.problem_receipt(p, np.array([[1,2]])),
        "transfer_cases": [{"id": f"states-{a}-{b}", "truth": [a,b], "aggregate": float(problem(a,b).aggregate[0]),
                            "role": "training" if (a,b)==(1,2) else "synthetic_transfer"} for b in range(3) for a in range(3)],
        "expected_fits": 75, "evaluations_per_fit": 1024, "expected_training_evaluations": 76800,
        "expected_training_shots": 6553600, "shots_per_sampled_call": 256,
        "noise_fits": "exact8 only, all 25 fits plus uniform W preparation",
        "expected_noise_simulations": 78, "calibration_retrieved_utc": snapshot["retrieved_utc"],
        "scope": "One synthetic six-qubit fit task; eight synthetic transfer cases; no held-out real-data or hardware claim"}
    folder.mkdir(parents=True)
    save(folder/"plan.json", plan)
    print(json.dumps({"plan_sha256": sha(folder/"plan.json"), "fits": 75, "noise_simulations": 78}), flush=True)


def verify(folder):
    plan = read(folder/"plan.json")
    for group in ("source_sha256", "input_sha256"):
        for name, expected in plan[group].items():
            lc.require(sha(ROOT/name) == expected, f"Frozen file changed: {name}")
    for package, expected in plan["versions"].items():
        lc.require(version(package) == expected, f"Version changed: {package}")
    return plan


def run(folder):
    plan = verify(folder)
    lc.require(not (folder/"summary.json").exists(), "Completed archive; no rerun")
    binding, fitted, p = sha(folder/"plan.json"), [], problem()
    for config, (sampled, upper) in CONFIGS.items():
        for trial, seeds in enumerate(PAIRS):
            for loss in lc.LOSSES:
                key = f"{config}_trial{trial+1}_{loss}"
                path = folder/f"{key}.json.gz"
                if path.exists():
                    entry = read(path)
                    lc.require(entry["plan_sha256"] == binding and entry["id"] == key, "Checkpoint binding")
                else:
                    fit = lc.train(p, loss, seeds, sampled=sampled, gamma_max=upper)
                    entry = {"id": key, "config": config, "trial": trial+1, "plan_sha256": binding, "fit": fit}
                    save(path, entry)
                    print(f"Trained {key}: {fit['evaluations']} calls", flush=True)
                fit = entry["fit"]
                lc.require(fit["loss"] == loss and fit["seeds"] == list(seeds) and fit["sampled_training"] == sampled
                           and fit["gamma_max"] == upper and fit["evaluations"] == 1024, "Checkpoint identity/budget")
                fitted.append({"id": key, "config": config, "trial": trial+1, "file": path.name,
                               "sha256": sha(path), "angles": fit["angles"], "loss": loss,
                               "evaluations": fit["evaluations"], "training_shots": fit["training_shots"],
                               "runtime_seconds": fit["runtime_seconds"]})
    # Save a manifest before evaluation, freezing selected fits and checkpoint hashes.
    if (folder/"training_manifest.json").exists():
        lc.require(read(folder/"training_manifest.json")["fits"] == fitted, "Training manifest changed")
    else:
        save(folder/"training_manifest.json", {"plan_sha256": binding, "fits": fitted})
    evaluation_path = folder/"ideal_evaluation.json"
    if evaluation_path.exists():
        ideal_rows = read(evaluation_path)["rows"]
        lc.require(read(evaluation_path)["plan_sha256"] == binding, "Ideal binding")
    else:
        ideal_rows = []
        for fit in fitted + [{"id": "uniform", "config": "control", "trial": 0, "angles": [0.,0.], "loss": "uniform"}]:
            for case in plan["transfer_cases"]:
                task = problem(*case["truth"])
                probs = categorical_qaoa_probabilities(task, [fit["angles"][0]], [fit["angles"][1]])
                ideal_rows.append({"fit_id": fit["id"], "case_id": case["id"], "role": case["role"],
                    "config": fit["config"], "trial": fit["trial"], "loss": fit["loss"],
                    "metrics": lc.decode_metrics(task, probs, case["truth"]),
                    "probabilities": probs.tolist(), "exact_reference_cost_w2": float(min(task.energies))})
        save(evaluation_path, {"plan_sha256": binding, "rows": ideal_rows})
    full_target, properties = d.load_calibration(read(SNAPSHOT))
    noise_rows = []
    selected = [f for f in fitted if f["config"] == "exact8"] + [
        {"id": "uniform_w", "config": "control", "trial": 0, "angles": [0.,0.], "loss": "uniform_w"}]
    for fit in selected:
        logical = d.logical_circuit(p, [fit["angles"][0]], [fit["angles"][1]])
        expected = d.reference_probabilities(p, [fit["angles"][0]], [fit["angles"][1]], "full")
        for placement in ("suspect", "comparison"):
            basekey = f"{fit['id']}_{placement}"
            metadata_path = folder/f"{basekey}_compiled.json"
            if metadata_path.exists():
                metadata = read(metadata_path)
                lc.require(metadata["plan_sha256"] == binding, "Circuit binding")
                lc.require(sha(folder/metadata["qpy_file"]) == metadata["qpy_sha256"], "Circuit checksum")
                with (folder/metadata["qpy_file"]).open("rb") as handle:
                    circuit = qpy.load(handle)[0]
                target = d.compact_target(full_target, metadata["physical_qubits"])
            else:
                circuit, target, metadata = d.compile_circuit(logical, placement, full_target)
                error = float(max(abs(d.ideal_probabilities(circuit)-expected)))
                lc.require(error < 1e-8, "Compiled ideal discrepancy")
                qpy_path = folder/f"{basekey}.qpy"
                save_qpy(qpy_path, circuit)
                metadata.update({"plan_sha256": binding, "qpy_file": qpy_path.name, "qpy_sha256": sha(qpy_path),
                                 "ideal_probability_error": error, "fit_id": fit["id"]})
                save(metadata_path, metadata)
            models = [{"name": "calibration_combined", "calibration": "combined", "readout": True}]
            if placement == "suspect":
                models.append({"name": "generic_combined"})
            for model in models:
                key = f"{basekey}_{model['name']}"
                path = folder/f"{key}.json"
                if path.exists():
                    row = read(path)
                    lc.require(row["plan_sha256"] == binding, "Noise binding")
                    for name, checksum in row["files_sha256"].items():
                        lc.require(sha(folder/name) == checksum, "Noise checksum")
                else:
                    noisy, errors, timing = d.noisy_circuit(circuit, target, properties, metadata["physical_qubits"], model)
                    probabilities, mass = d.simulate(noisy, errors, metadata["logical_measurement_to_compact"])
                    qpy_path, npy_path = folder/f"{key}_noise.qpy", folder/f"{key}.npy"
                    save_qpy(qpy_path, noisy)
                    with npy_path.open("xb") as handle:
                        np.save(handle, probabilities, allow_pickle=False)
                    row = {"id": key, "fit_id": fit["id"], "loss": fit["loss"], "trial": fit["trial"],
                        "placement": placement, "model": model, "plan_sha256": binding,
                        "metrics": lc.decode_metrics(p, probabilities[d.feasible_indices(p)], [1,2]),
                        "timing": timing, "readout_errors": errors, "measurement_map": metadata["logical_measurement_to_compact"],
                        "pre_readout_mass": mass, "files_sha256": {qpy_path.name: sha(qpy_path), npy_path.name: sha(npy_path)}}
                    save(path, row)
                    print(f"Simulated {key}", flush=True)
                noise_rows.append(row)
    verify(folder)
    lc.require(len(fitted)==75 and len(ideal_rows)==684 and len(noise_rows)==78, "Coverage incomplete")
    summary = {"completed_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": binding,
        "hardware_jobs": 0, "qpu_seconds": 0, "training_evaluations": sum(f["evaluations"] for f in fitted),
        "classically_sampled_training_shots": sum(f["training_shots"] for f in fitted),
        "training_runtime_seconds": sum(f["runtime_seconds"] for f in fitted),
        "training_manifest_sha256": sha(folder/"training_manifest.json"), "ideal_evaluation_sha256": sha(evaluation_path),
        "fits": fitted, "ideal_rows": ideal_rows, "noise_rows": noise_rows,
        "limitations": [plan["scope"], "Five trials quantify seed variation, not physical or generalization uncertainty",
                        "Finite-shot training uses ideal measurements; only exact8 fits receive device-noise evaluation",
                        "Conditional decoded errors exclude abstention batches, whose probabilities are explicitly reported",
                        "No speed advantage or replacement held-out MAE claimed"]}
    save(folder/"summary.json", summary)
    print(json.dumps({"completed_fits": 75, "noise_simulations": 78, "training_evaluations": summary["training_evaluations"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "run", "verify"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output_dir.resolve()
    lc.require(folder.is_relative_to(ROOT/"results/quantum_diagnostics") and folder != ROOT/"results/quantum_diagnostics", "Named archive required")
    {"freeze": freeze, "run": run, "verify": verify}[args.mode](folder)
