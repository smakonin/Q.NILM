#!/usr/bin/env python3
"""Offline ideal audit of the eight frozen, physically routed ladder templates.

No account, service, or quantum job APIs are imported or called. QPY circuits
are bound without recompilation; canonical exported MPS tensors are contracted
directly, including every active unmeasured routing-ancilla assignment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit_aer import AerSimulator

from quantum_nilm.categorical_ibm import (build_categorical_parameterized_template,
    categorical_template_parameter_values)
from quantum_nilm.categorical_qaoa import (categorical_qaoa_probabilities,
    prepare_categorical_problem)

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("W", "W_mixer", "W_cost", "full")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_circuit(bound, logical_qubits, saved_map):
    active = sorted({bound.find_bit(q).index for item in bound.data
                     if item.operation.name != "barrier" for q in item.qubits})
    require(len(active) < 64, "Audit bit indices exceed uint64 capacity")
    mapping = {physical: index for index, physical in enumerate(active)}
    circuit, measured = QuantumCircuit(len(active)), {}
    circuit.global_phase = bound.global_phase
    for item in bound.data:
        if item.operation.name == "barrier":
            continue
        if item.operation.name == "measure":
            bit = bound.find_bit(item.clbits[0]).index
            require(bit not in measured, "Repeated classical measurement bit")
            measured[bit] = mapping[bound.find_bit(item.qubits[0]).index]
        else:
            require(not measured and not item.clbits and item.operation.name != "reset",
                    "Expected a unitary circuit followed by terminal measurements")
            circuit.append(item.operation, [mapping[bound.find_bit(q).index] for q in item.qubits])
    require(set(measured) == set(range(logical_qubits))
            and len(set(measured.values())) == logical_qubits, "Logical measurement map is not bijective")
    require([active[measured[b]] for b in range(logical_qubits)] == saved_map,
            "QPY measurement map differs from the frozen resource manifest")
    return circuit, active, measured


def probabilities_from_mps(circuit, problem, measured):
    unmeasured = sorted(set(range(circuit.num_qubits)) - set(measured.values()))
    require(len(unmeasured) <= 12, "Too many ancillary assignments for this bounded audit")
    indices = np.zeros(problem.num_feasible_states, dtype=np.uint64)
    states = problem.states.reshape(problem.num_feasible_states, -1)
    for register, offset in enumerate(problem.register_offsets):
        indices |= np.array([1 << measured[int(offset + category)]
                             for category in states[:, register]], dtype=np.uint64)
    ancillary = np.array([sum(((assignment >> bit) & 1) << q for bit, q in enumerate(unmeasured))
                          for assignment in range(1 << len(unmeasured))], dtype=np.uint64)
    requested = (indices[:, None] | ancillary[None, :]).reshape(-1)
    circuit.save_matrix_product_state(label="mps")
    simulator = AerSimulator(method="matrix_product_state", max_parallel_threads=4,
        matrix_product_state_truncation_threshold=0.0, matrix_product_state_max_bond_dimension=None)
    result = simulator.run(circuit, shots=1).result()
    require(result.success, "Local MPS simulation failed")
    tensors, bonds = result.data(0)["mps"]
    # Exported tensors use canonical qubit order, unlike Aer 0.17.2's
    # save_amplitudes path, which can expose an internal routing permutation.
    amplitudes, environment = np.ones((len(requested), 1), complex), np.ones((1, 1), complex)
    for site, pair in enumerate(tensors):
        factor = np.asarray(bonds[site]) if site < len(bonds) else np.ones(1)
        matrices = [np.asarray(matrix) * factor[None, :] for matrix in pair]
        ones = ((requested >> site) & 1).astype(bool)
        amplitudes = np.where(ones[:, None], amplitudes @ matrices[1], amplitudes @ matrices[0])
        environment = sum(matrix.conj().T @ environment @ matrix for matrix in matrices)
    actual = (np.abs(amplitudes[:, 0]) ** 2).reshape(len(indices), len(ancillary)).sum(axis=1)
    return actual, float(environment[0, 0].real), unmeasured, max(map(len, bonds), default=1)


def audit(folder, output):
    require(not output.exists(), "Audit output exists; choose an unused filename")
    plan = read(folder / "plan.json")
    paths = {folder / "plan.json", Path(__file__).resolve()}
    for group in ("source_sha256", "code_sha256", "files_sha256"):
        base = folder if group == "files_sha256" else ROOT
        for name, expected in plan[group].items():
            path = (base / name).resolve()
            require(path.is_relative_to(base.resolve()) and checksum(path) == expected,
                    f"Frozen artifact hash mismatch: {name}")
            paths.add(path)
    before = {str(path): checksum(path) for path in sorted(paths)}
    expected_keys = {f"K{width}_{condition}" for width in (1, 2) for condition in CONDITIONS}
    require(set(plan["resources"]) == expected_keys, "Expected exactly eight ladder templates")
    used_paths = [folder / name for name in ("pubs.json", "examples.json")]
    used_paths += [folder / f"{key}.qpy" for key in expected_keys]
    used_paths += [ROOT / "results/quantum_heldout" / name for name in ("model.json", "angles.json")]
    require(all(str(path.resolve()) in before for path in used_paths), "Used input is missing a frozen checksum")
    angles = read(ROOT / "results/quantum_heldout/angles.json")
    require(plan["gammas"] == angles["gammas"] and plan["betas"] == angles["betas"], "Frozen angles differ")
    pubs, examples = read(folder / "pubs.json"), read(folder / "examples.json")
    require(len(pubs) == 24 and all(x["window_id"].startswith("train-") for x in examples),
            "Expected 24 PUBs drawn only from training examples")
    example = examples[0]
    frozen = read(ROOT / "results/quantum_heldout/model.json")
    model = frozen["models"]["multistate"]
    levels = [np.asarray(x, float) for x in model["levels_w"]]
    rho = frozen["selected_parameters"]["multistate_compressed"]["rho"]
    require(rho == 0.1, "Frozen ladder runner assumes rho=0.1")
    penalties = rho * np.asarray(model["ranges_w"]) ** 2
    require(len(plan["gammas"]) == len(plan["betas"]) == 1, "Expected depth-one frozen angles")
    rows = []
    for width in (1, 2):
        problem = prepare_categorical_problem(example["aggregate"][:width], levels, penalties,
                                              example["weights"][:width], previous_states=None)
        template = build_categorical_parameterized_template(problem.state_counts, width, plan["betas"][0])
        for condition in CONDITIONS:
            started, key = perf_counter(), f"K{width}_{condition}"
            matches = [pub for pub in pubs if pub["example_index"] == 0 and pub["template"] == key]
            require(len(matches) == 1, f"Missing or duplicate first-example PUB: {key}")
            pub, resource = matches[0], plan["resources"][key]
            require(pub["width"] == resource["width"] == width
                    and pub["condition"] == resource["condition"] == condition, "PUB/resource identity mismatch")
            with (folder / f"{key}.qpy").open("rb") as handle:
                loaded = qpy.load(handle)
            require(len(loaded) == 1, "Expected one circuit per QPY")
            compiled = loaded[0]
            require(list(map(str, compiled.parameters)) == resource["parameter_order"], "Parameter order mismatch")
            values = categorical_template_parameter_values(template, problem, plan["gammas"][0],
                                                           parameter_order=tuple(compiled.parameters))
            require(np.array_equal(values, np.asarray(pub["parameter_values"])), "Saved PUB binding differs")
            compact, active, measured = compact_circuit(compiled.assign_parameters(values), problem.num_qubits,
                resource["logical_measurement_bit_to_physical"])
            actual, norm, unmeasured, max_bond = probabilities_from_mps(compact, problem, measured)
            expected = (np.full(problem.num_feasible_states, 1 / problem.num_feasible_states)
                if condition in ("W", "W_cost") else categorical_qaoa_probabilities(problem,
                    [0.] if condition == "W_mixer" else plan["gammas"], plan["betas"]))
            error, mass = float(np.max(np.abs(actual - expected))), float(actual.sum())
            require(np.isfinite(actual).all() and error < 1e-7 and abs(norm - 1) < 1e-8
                    and abs(norm - mass) < 1e-7, f"Compiled ideal audit failed: {key}, error={error}, norm={norm}, mass={mass}")
            row = {"template": key, "example_index": 0, "window_id": example["window_id"],
                "chunk_index": example["chunk"], "previous_states": None, "active_compact_to_physical": active,
                "logical_measurement_bit_to_compact_qubit": [measured[b] for b in range(problem.num_qubits)],
                "unmeasured_routing_ancillas_compact": unmeasured, "all_ancilla_assignments_marginalized": True,
                "feasible_states": problem.num_feasible_states, "mps_total_norm": norm, "feasible_probability_mass": mass,
                "max_absolute_feasible_probability_error": error, "infeasible_mass_signed_roundoff_residual": norm - mass,
                "total_variation_distance_including_residual": float((abs(actual - expected).sum() + max(0., norm - mass)) / 2),
                "maximum_observed_bond_dimension": max_bond, "local_audit_wall_time_s": perf_counter() - started}
            rows.append(row)
            print(json.dumps(row), flush=True)
    require(before == {str(path): checksum(path) for path in sorted(paths)}, "Inputs changed during audit")
    payload = {"status": "passed; offline compiled ideal audit; no accounts or quantum jobs",
        "created_utc": datetime.now(timezone.utc).isoformat(), "folder": str(folder.resolve()), "rows": rows,
        "gammas": plan["gammas"], "betas": plan["betas"], "source_sha256": before,
        "source_hashes_unchanged_during_audit": True, "mps_truncation_threshold": 0., "mps_max_bond_dimension_limit": None,
        "amplitude_extraction": "direct canonical exported-MPS tensor contraction",
        "versions": {name: version(name) for name in ("numpy", "qiskit", "qiskit-aer")}}
    with output.open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.folder.resolve(), args.output.resolve())
