#!/usr/bin/env python3
"""Prepare and run one immutable OFFLINE diagnostic; no hardware APIs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from qiskit import qasm3, qpy
from qiskit_ibm_runtime import RuntimeDecoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quantum_nilm import six_qubit_diagnostic as d
from quantum_nilm.stage_b import optimize_stage_b_qaoa
from quantum_nilm.categorical_ibm import build_categorical_qaoa_circuit

SNAPSHOT = ROOT / "results/quantum_diagnostics/physical_001/preparation_snapshot.json"
FILES = ["scripts/run_six_qubit_diagnostic.py", "src/quantum_nilm/six_qubit_diagnostic.py",
         "src/quantum_nilm/categorical_ibm.py", "src/quantum_nilm/categorical_qaoa.py",
         "src/quantum_nilm/stage_b.py", "docs/six_qubit_diagnostic_protocol.md",
         "tests/test_six_qubit_diagnostic.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), cls=RuntimeDecoder)


def save(path, data):
    with Path(path).open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_qpy(path, circuit):
    with path.open("xb") as handle:
        qpy.dump(circuit, handle)


def load_qpy(path):
    with path.open("rb") as handle:
        return qpy.load(handle)[0]


def save_array(path, values):
    with path.open("xb") as handle:
        np.save(handle, values, allow_pickle=False)


def utc():
    return datetime.now(timezone.utc).isoformat()


def case_problem(case):
    channels, intervals, _, _ = d.CASES[case]
    return d.make_problem(channels, intervals)


def prepare(folder):
    d.require(not folder.exists(), "Archive exists; choose a fresh output folder")
    snapshot = read(SNAPSHOT)
    full_target, _ = d.load_calibration(snapshot)
    problem, truth = d.make_problem()
    # Training uses objective expectations only, not known solution labels.
    trained = optimize_stage_b_qaoa(problem, 1, restart_seeds=(9101, 9102), evaluations_per_restart=512)
    print("Training complete", flush=True)
    folder.mkdir(parents=True)
    save(folder / "training.json", trained)
    gamma, beta = trained["gammas"][0], trained["betas"][0]
    entries, objectives, baselines = [], {}, {}
    for case, (_, _, depth, fold) in d.CASES.items():
        problem, truth = case_problem(case)
        objectives[case] = {**d.problem_receipt(problem, truth), **d.independent_objective_check(problem)}
        gammas, betas = [gamma / depth] * depth, [beta / depth] * depth
        # Equal total angles isolate depth resource growth; no claim p>1 optimal.
        for component in (("W", "W_cost", "W_mixer", "full") if case == "six_p1" else ("full",)):
            logical = d.logical_circuit(problem, gammas, betas, component)
            reference = d.reference_probabilities(problem, gammas, betas, component)
            logical_p = d.ideal_probabilities(logical)
            logical_error = float(abs(reference - logical_p).max())
            d.require(logical_error < 1e-9, "Logical circuit differs from categorical simulator")
            if component == "full":
                original = build_categorical_qaoa_circuit(problem, gammas, betas)
                d.require(float(abs(logical_p - d.ideal_probabilities(original)).max()) < 1e-9,
                          "New diagnostic builder differs from original circuit")
            for placement in ("all_to_all", "suspect", "comparison"):
                circuit, target, metadata = d.compile_circuit(logical, placement, full_target, fold)
                actual = d.ideal_probabilities(circuit)
                error = float(abs(reference - actual).max())
                d.require(error < 1e-8, "Compiled ideal measurement distribution differs")
                key = f"{case}_{component}_{placement}"
                write_qpy(folder / f"{key}.qpy", circuit)
                with (folder / f"{key}.qasm").open("x") as handle:
                    handle.write(qasm3.dumps(circuit))
                save_array(folder / f"{key}_ideal.npy", actual)
                spec = {"id": key, "case": case, "component": component, "gammas": gammas,
                        "betas": betas, **metadata, "logical_max_probability_error": logical_error,
                        "compiled_max_probability_error": error,
                        "ideal_metrics": d.score(problem, actual, reference)}
                if case == "six_p1":
                    baselines[(placement, component)] = actual
                if fold > 1:
                    d.require(float(abs(actual - baselines[(placement, "full")]).max()) < 1e-8,
                              "Odd CZ folding changed the ideal answer")
                entries.append(spec)
                print(f"Prepared {key}: CZ={metadata['native_gate_counts'].get('cz', 0)}", flush=True)
    phase_controls = []
    problem, _ = d.make_problem()
    for component in ("W_cost", "full"):
        expected = d.reference_probabilities(problem, [gamma], [beta], component)
        for phase in (-.3, .3):
            logical = d.logical_circuit(problem, [gamma], [beta], component, phase=phase)
            for placement in ("all_to_all", "suspect", "comparison"):
                circuit, _, metadata = d.compile_circuit(logical, placement, full_target)
                p = d.ideal_probabilities(circuit)
                metrics = d.score(problem, p, expected)
                d.require(metrics["valid_probability"] > 1 - 1e-9, "Diagonal phase broke one-hot feasibility")
                if component == "W_cost":
                    d.require(metrics["full_distribution_tv_to_own_ideal"] < 1e-9,
                              "Pure diagonal phase changed immediate measurement populations")
                key = f"phase_{component}_{'negative' if phase < 0 else 'positive'}_{placement}"
                write_qpy(folder / f"{key}.qpy", circuit)
                save_array(folder / f"{key}.npy", p)
                phase_controls.append({"id": key, "component": component, "placement": placement,
                    "phase_rad": phase, "insertion": "logical RZ on qubit 2 at end of cost, before mixer if present",
                    "metrics": metrics, **metadata})
    sources = {name: sha(ROOT / name) for name in FILES}
    artifacts = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
    plan = {"created_utc": utc(), "execution": "offline only; no hardware submission code",
        "snapshot_relative_path": str(SNAPSHOT.relative_to(ROOT)), "snapshot_sha256": sha(SNAPSHOT),
        "snapshot_retrieved_utc": snapshot["retrieved_utc"],
        "backend": snapshot["backend"], "training_rule": "one six-qubit p1 objective-only fit, 2 × 512 calls; reused across noise, topology and growth",
        "depth_rule": "p2/p3 split p1 gamma and beta equally; not independently optimized",
        "source_sha256": sources, "files_sha256": artifacts,
        "versions": {name: version(name) for name in ("qiskit", "qiskit-aer", "qiskit-ibm-runtime", "numpy", "scipy")},
        "models": d.models(), "problems": objectives, "circuits": entries,
        "logical_phase_controls": phase_controls,
        "limitations": ["Synthetic reduced task, not held-out NILM accuracy or hardware evidence",
            "Restricted physical subgraphs, not the earlier full-width IBM circuit",
            "ASAP timing is a model, not IBM pulse schedule",
            "Independent calibrated noise excludes crosstalk, leakage and drift",
            "Coherent ±0.02-radian injections are sensitivity scenarios, not fitted gate errors",
            "Population distributions are exact within specified models; no experimental uncertainty estimated"]}
    save(folder / "plan.json", plan)
    print(json.dumps({"prepared": len(entries), "phase_controls": len(phase_controls), "plan_sha256": sha(folder / "plan.json")}), flush=True)


def verify(folder):
    plan = read(folder / "plan.json")
    for package, expected in plan["versions"].items():
        d.require(version(package) == expected, f"Frozen package version changed: {package}")
    for name, expected in plan["source_sha256"].items():
        d.require(sha(ROOT / name) == expected, f"Frozen source changed: {name}")
    for name, expected in plan["files_sha256"].items():
        d.require(sha(folder / name) == expected, f"Frozen artifact changed: {name}")
    d.require(sha(SNAPSHOT) == plan["snapshot_sha256"], "Calibration snapshot changed")
    return plan


def task_models(spec, available):
    if spec["placement"] == "all_to_all":
        names = {"generic_combined"}
    elif spec["case"] == "six_p1":
        names = {m["name"] for m in available} - {"ideal"}
    else:
        names = {"generic_combined", "calibration_combined"}
    return [m for m in available if m["name"] in names]


def run(folder):
    plan = verify(folder)
    d.require(not (folder / "summary.json").exists(), "Completed archive; do not overwrite or rerun")
    full_target, properties = d.load_calibration(read(SNAPSHOT))
    rows = []
    for spec in plan["circuits"]:
        problem, _ = case_problem(spec["case"])
        circuit = load_qpy(folder / f"{spec['id']}.qpy")
        physical = spec["physical_qubits"]
        target = d.compact_target(full_target, physical) if physical is not None else None
        ideal = np.load(folder / f"{spec['id']}_ideal.npy", allow_pickle=False)
        for model in task_models(spec, plan["models"]):
            key = f"{spec['id']}__{model['name']}"
            result_path = folder / f"{key}.json"
            if result_path.exists():
                saved = read(result_path)
                d.require(saved["plan_sha256"] == sha(folder / "plan.json"), "Resumed result belongs to another plan")
                d.require(saved["spec_id"] == spec["id"] and saved["model"] == model, "Resumed result identity differs")
                for filename, expected in saved["files_sha256"].items():
                    d.require(sha(folder / filename) == expected, "Saved noisy artifact changed")
                rows.append(saved)
                continue
            noisy, errors, timing = d.noisy_circuit(circuit, target, properties, physical, model)
            started = perf_counter()
            p, mass = d.simulate(noisy, errors, spec["logical_measurement_to_compact"])
            elapsed = perf_counter() - started
            probability_path = folder / f"{key}.npy"
            qpy_path = folder / f"{key}.qpy"
            save_array(probability_path, p)
            write_qpy(qpy_path, noisy)
            row = {"id": key, "spec_id": spec["id"], "plan_sha256": sha(folder / "plan.json"),
                "model": model, "metrics": d.score(problem, p, ideal), "timing_model": timing,
                "readout_errors_physical_order": errors, "simulation_seconds": elapsed,
                "pre_readout_probability_mass": mass,
                "files_sha256": {p.name: sha(p) for p in (probability_path, qpy_path)}}
            save(result_path, row)
            rows.append(row)
            print(json.dumps({"done": key, "valid": row["metrics"]["valid_probability"],
                              "tv": row["metrics"]["full_distribution_tv_to_own_ideal"], "seconds": elapsed}), flush=True)
    verify(folder)
    expected = sum(len(task_models(spec, plan["models"])) for spec in plan["circuits"])
    d.require(len(rows) == expected and len({r["id"] for r in rows}) == expected, "Simulation coverage mismatch")
    save(folder / "summary.json", {"completed_utc": utc(), "execution": "offline exact density-matrix simulation",
        "hardware_jobs": 0, "qpu_seconds": 0, "plan_sha256": sha(folder / "plan.json"),
        "ideal_circuits": len(plan["circuits"]), "phase_controls": len(plan["logical_phase_controls"]),
        "noisy_simulations": len(rows), "total_density_simulation_seconds": sum(r["simulation_seconds"] for r in rows),
        "rows": rows, "limitations": plan["limitations"]})
    print(f"Completed {len(rows)} exact noisy simulations", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "verify"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output_dir.resolve()
    d.require(folder.is_relative_to(ROOT / "results/quantum_diagnostics") and folder != ROOT / "results/quantum_diagnostics",
              "Use a named diagnostic archive under results/quantum_diagnostics")
    {"prepare": prepare, "run": run, "verify": verify}[args.mode](folder)
