#!/usr/bin/env python3
"""Reproduce the frozen R1Hz QAOA pilot locally or on IBM Quantum hardware."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
from qiskit import qasm3, qpy
from qiskit.quantum_info import Statevector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeEncoder, SamplerV2

from quantum_nilm.ibm import build_qaoa_circuit, counts_to_metrics
from quantum_nilm.qaoa import phase_scale_for, qaoa_probabilities
from quantum_nilm.qubo import build_binary_temporal_qubo


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, cls=RuntimeEncoder, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_problem(source: dict):
    channels = source["channels"]
    powers = np.array([source["nominal_incremental_power_w"][key] for key in channels])
    penalties = np.array([source["switch_penalty"][key] for key in channels])
    aggregate = np.array(source["aggregate_segment_power_w"])
    weights = np.array(source["selected_segment_durations_blocks"])
    reference = np.array(source["reference_states"])
    if int(source["qubits"]) > 12:
        raise ValueError("This circuit-validation runner is limited to 12 logical qubits")
    qubo = build_binary_temporal_qubo(aggregate, powers, penalties, weights)
    if qubo.n_variables != source["qubits"] or source["qaoa_depth"] != 1:
        raise ValueError("Expected the frozen depth-one pilot summary")
    if not np.isclose(phase_scale_for(qubo), source["phase_scale"], rtol=1e-12, atol=0):
        raise ValueError("Pilot phase scale does not match its QUBO coefficients")
    return qubo, dict(reference_states=reference, aggregate=aggregate, powers=powers,
                     segment_weights=weights)


def circuit_stats(circuit) -> dict:
    active = {circuit.find_bit(q).index for item in circuit.data
              if item.operation.name != "barrier" for q in item.qubits}
    return {
        "allocated_qubits": circuit.num_qubits,
        "active_qubits": len(active),
        "active_physical_indices": sorted(active),
        "depth": circuit.depth(),
        "two_qubit_gates": sum(len(item.qubits) == 2 and item.operation.name != "barrier"
                               for item in circuit.data),
        "gate_counts": dict(circuit.count_ops()),
        "logical_to_final_physical": (circuit.layout.final_index_layout()
                                      if circuit.layout else None),
    }


def service_for(account: str):
    return QiskitRuntimeService(name=account)


def active_plan(service) -> str | None:
    crn = service.active_instance()
    return next((item.get("plan") for item in service.instances() if item["crn"] == crn), None)


def save_results(directory: Path, plan: dict, result, elapsed: float, job_metrics=None) -> None:
    source = json.loads((directory / "input_summary.json").read_text())
    qubo, metric_inputs = load_problem(source)
    counts = result[0].data.meas.get_counts()
    if sum(counts.values()) != plan["shots"]:
        raise ValueError("Returned shot count does not match the frozen experiment")
    write_json(directory / "counts.json", counts)
    write_json(directory / "runtime_result.json", result)
    score = counts_to_metrics(qubo, counts, **metric_inputs)
    score.update({
        "algorithm": "Q.NILM", "mode": plan["mode"], "backend": plan["backend"],
        "qaoa_depth": 1, "parameter_source": "frozen classical noiseless pilot optimization",
        "reference_type": "circuit-derived proxy states; diagnostic selected interval",
        "aggregate_mode": source.get("aggregate_mode", "not recorded in historical input summary"),
        "aggregate_scope": source.get(
            "aggregate_scope", "legacy selected-channel residual target, not whole-house mains"
        ),
        "aggregate_mae_grain": (
            "segment-mean target; not mean absolute error over original 30-second blocks"
        ),
        "sampler_wait_wall_time_s": elapsed,
        "job_metrics": job_metrics,
        "completed_utc": utc_now(),
        "plan_sha256": checksum(directory / "plan.json"),
        "counts_sha256": checksum(directory / "counts.json"),
    })
    if job_metrics:
        times = job_metrics.get("timestamps", {})
        created, running, finished = (times.get(k) for k in ("created", "running", "finished"))
        parse = lambda x: datetime.fromisoformat(x.replace("Z", "+00:00"))
        if created and running:
            score["queue_and_initialization_s"] = (parse(running) - parse(created)).total_seconds()
        if running and finished:
            score["server_running_wall_time_s"] = (parse(finished) - parse(running)).total_seconds()
    write_json(directory / "summary.json", score)
    print(json.dumps({key: score[key] for key in
                     ("mode", "shots", "exact_optimum_probability", "reference_metrics")}, indent=2))


def prepare_or_simulate(args) -> None:
    started = perf_counter()
    source = json.loads(args.pilot_summary.read_text())
    qubo, _ = load_problem(source)
    logical = build_qaoa_circuit(qubo, [source["gamma"]], [source["beta"]], source["phase_scale"])
    reference_probabilities = qaoa_probabilities(
        qubo, [source["gamma"]], [source["beta"]], source["phase_scale"]
    )
    qiskit_probabilities = Statevector.from_instruction(
        logical.remove_final_measurements(inplace=False)
    ).probabilities()
    error = float(np.max(np.abs(reference_probabilities - qiskit_probabilities)))
    if error > 1e-12:
        raise ValueError("Qiskit circuit disagrees with the independent NumPy simulator")

    service = None
    if args.mode == "prepare":
        service = service_for(args.account)
        backend = (service.backend(args.backend) if args.backend else
                   service.least_busy(operational=True, simulator=False,
                                      min_num_qubits=qubo.n_variables))
        if not backend.status().operational:
            raise RuntimeError("Selected IBM backend is not operational")
    elif args.mode == "noisy":
        from qiskit_ibm_runtime import fake_provider
        fake_name = args.backend or "FakeTorino"
        if not fake_name.startswith("Fake"):
            raise ValueError("Noisy offline mode expects a fake backend class such as FakeTorino")
        backend = getattr(fake_provider, fake_name)()
    else:
        if args.backend:
            raise ValueError("Use --backend with noisy or prepare mode")
        backend = AerSimulator()

    transpile_started = perf_counter()
    manager = generate_preset_pass_manager(
        backend=backend, optimization_level=3, seed_transpiler=args.seed
    )
    prepared = manager.run(logical)
    transpile_time = perf_counter() - transpile_started
    args.output_dir.mkdir(parents=True, exist_ok=False)
    directory = args.output_dir
    write_json(directory / "input_summary.json", source)
    (directory / "logical.qasm").write_text(qasm3.dumps(logical))
    (directory / "prepared.qasm").write_text(qasm3.dumps(prepared))
    with (directory / "prepared.qpy").open("wb") as handle:
        qpy.dump(prepared, handle)
    properties = backend.properties() if hasattr(backend, "properties") else None
    if properties is not None:
        write_json(directory / "backend_properties.json", properties.to_dict())
    if hasattr(backend, "configuration"):
        write_json(directory / "backend_configuration.json", backend.configuration().to_dict())
    plan = {
        "schema_version": 1, "algorithm": "Q.NILM", "created_utc": utc_now(),
        "mode": "qpu" if args.mode == "prepare" else args.mode,
        "backend": backend.name,
        "backend_source": ("live IBM calibration" if service else
                           "packaged IBM calibration snapshot" if args.mode == "noisy" else
                           "noiseless local Aer simulator"),
        "instance_plan": active_plan(service) if service else None,
        "account_name": args.account if service else None,
        "shots": args.shots, "seed": args.seed, "max_execution_time_s": args.max_execution_time,
        "gammas": [source["gamma"]], "betas": [source["beta"]],
        "phase_scale": source["phase_scale"],
        "probability_crosscheck_max_abs_error": error,
        "logical_circuit": circuit_stats(logical), "prepared_circuit": circuit_stats(prepared),
        "transpilation_wall_time_s": transpile_time,
        "preparation_wall_time_s": perf_counter() - started,
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("numpy", "qiskit", "qiskit-aer", "qiskit-ibm-runtime")},
        "files_sha256": {p.name: checksum(p) for p in sorted(directory.iterdir()) if p.is_file()},
    }
    if service:
        plan["sampler_options"] = {
            "max_execution_time": args.max_execution_time,
            "dynamical_decoupling": {"enable": False},
            "twirling": {"enable_gates": False, "enable_measure": False},
            "environment": {"job_tags": ["qnilm", "r1hz-p1-pilot"]},
        }
    write_json(directory / "plan.json", plan)
    if args.mode == "prepare":
        print(json.dumps({"status": "prepared; no QPU job submitted", "backend": backend.name,
                          "shots": args.shots, "instance_plan": plan["instance_plan"],
                          "circuit": plan["prepared_circuit"]}, indent=2))
        return
    simulator = (AerSimulator.from_backend(backend, method="matrix_product_state",
                                          max_parallel_threads=4)
                 if args.mode == "noisy" else backend)
    sampler = SamplerV2(mode=simulator, options={"simulator": {"seed_simulator": args.seed}})
    sample_started = perf_counter()
    result = sampler.run([prepared], shots=args.shots).result()
    save_results(directory, plan, result, perf_counter() - sample_started)


def submit_or_collect(args) -> None:
    directory = args.output_dir
    plan = json.loads((directory / "plan.json").read_text())
    if plan["mode"] != "qpu":
        raise ValueError("Expected a hardware plan made with --mode prepare")
    for name, expected in plan["files_sha256"].items():
        if Path(name).name != name or checksum(directory / name) != expected:
            raise ValueError("Prepared experiment failed its file checksum check")
    service = service_for(args.account)
    job_path = directory / "job.json"
    if args.mode == "submit":
        if job_path.exists():
            raise ValueError("A submission record already exists. Use collect; never resubmit blindly")
        backend = service.backend(plan["backend"])
        if active_plan(service) != "open" and not args.allow_paid:
            raise ValueError("Submission requires an Open Plan instance; paid use requires --allow-paid")
        if not backend.status().operational:
            raise RuntimeError("Prepared backend is no longer operational")
        with (directory / "prepared.qpy").open("rb") as handle:
            circuits = qpy.load(handle)
        if len(circuits) != 1:
            raise ValueError("Expected exactly one prepared circuit")
        sampler = SamplerV2(mode=backend, options=plan["sampler_options"])
        record = {"status": "submission_started", "started_utc": utc_now(),
                  "plan_sha256": checksum(directory / "plan.json")}
        # Exclusive creation also protects against two simultaneous invocations.
        with job_path.open("x") as handle:
            json.dump(record, handle, indent=2)
        # On an ambiguous network failure, retain submission_started and reconcile
        # the IBM dashboard before doing anything that could create another job.
        job = sampler.run(circuits, shots=plan["shots"])
        record.update({"job_id": job.job_id(), "status": "submitted", "submitted_utc": utc_now()})
        write_json(job_path, record)
        print(json.dumps(record, indent=2))
        return
    record = json.loads(job_path.read_text())
    if "job_id" not in record:
        raise RuntimeError("Submission outcome unknown; check IBM Workloads before attempting another run")
    if record["plan_sha256"] != checksum(directory / "plan.json"):
        raise ValueError("The plan changed after submission")
    job = service.job(record["job_id"])
    status = job.status()
    print(f"IBM job {record['job_id']}: {status}")
    if status != "DONE":
        record["status"] = status
        write_json(job_path, record)
        return
    started = perf_counter()
    result = job.result()
    save_results(directory, plan, result, perf_counter() - started, job.metrics())
    record["status"] = "DONE"
    write_json(job_path, record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ideal", "noisy", "prepare", "submit", "collect"), default="ideal")
    parser.add_argument("--pilot-summary", type=Path, default=Path("results/pilot/summary.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", help="IBM backend name, or FakeTorino for offline noise")
    parser.add_argument("--account", default="qnilm")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-execution-time", type=int, default=60, help="QPU job limit, seconds")
    parser.add_argument("--allow-paid", action="store_true", help="Explicitly permit a non-Open-Plan submission")
    args = parser.parse_args()
    if not 1 <= args.shots <= 10000 or not 1 <= args.max_execution_time <= 600:
        parser.error("Pilot requires 1–10000 shots and a 1–600 second execution limit")
    if args.mode in ("submit", "collect"):
        submit_or_collect(args)
    else:
        prepare_or_simulate(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Input errors are ours; provider failures may contain credential details.
        detail = str(exc) if type(exc) in (ValueError, FileExistsError, FileNotFoundError) else type(exc).__name__
        print(f"IBM pilot stopped: {detail}", file=sys.stderr)
        sys.exit(1)
