#!/usr/bin/env python3
"""Independently check saved loss comparisons and selected logical circuits."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import itertools
import json
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector
from quantum_nilm.categorical_ibm import build_categorical_qaoa_circuit
from quantum_nilm.categorical_qaoa import prepare_categorical_problem

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def audit(folder):
    target = folder / "independent_audit.json"
    assert not target.exists()
    result = json.loads((folder / "summary.json").read_text())
    for name, expected in result["source_sha256"].items():
        assert sha(ROOT / name) == expected
    assert sha(folder / "grid.npz") == result["grid_sha256"]
    grid = np.load(folder / "grid.npz", allow_pickle=False)
    p, energies = grid["probabilities"], grid["energies"]
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12)
    direct = np.array([(680-a-b)**2 for b in (0, 120, 280) for a in (0, 400, 900)])
    np.testing.assert_array_equal(direct, energies)
    problem = prepare_categorical_problem([680], [[0, 400, 900], [0, 120, 280]], [1600, 400])
    maes = np.array([(abs(a-400)+abs(b-280))/2 for b in (0, 120, 280) for a in (0, 400, 900)])
    errors, mae_rows = [], []
    for candidate in result["candidates"]:
        i = candidate["grid_index"]
        np.testing.assert_allclose(candidate["probabilities"], p[i], atol=1e-12)
        np.testing.assert_allclose([candidate["gamma"], candidate["beta"]], grid["angles"][i], atol=1e-12)
        circuit = build_categorical_qaoa_circuit(problem, [candidate["gamma"]], [candidate["beta"]], measure=False)
        physical = Statevector.from_instruction(circuit).probabilities()
        feasible = [2**a + 2**(3+b) for b in range(3) for a in range(3)]
        expected = np.zeros(64)
        expected[feasible] = p[i]
        errors.append(float(max(abs(physical-expected))))
        row = {"name": candidate["name"], "expected_selected_macro_mae_w": {}}
        for k in (1, 2, 4, 16, 64, 256):
            # Probability state j wins: all costs >= Cj, minus all > Cj.
            winning = np.array([sum(p[i, energies >= e])**k - sum(p[i, energies > e])**k for e in energies])
            actual = float(winning @ energies)
            np.testing.assert_allclose(actual, candidate["metrics"][f"best_of_{k}_w2"], rtol=1e-10, atol=1e-8)
            row["expected_selected_macro_mae_w"][str(k)] = float(winning @ maes)
        brute = sum(p[i, a]*p[i, b]*min(energies[a], energies[b]) for a, b in itertools.product(range(9), repeat=2))
        np.testing.assert_allclose(brute, candidate["metrics"]["best_of_2_w2"], atol=1e-8)
        for alpha in (.1, .25, .5):
            remaining, total = alpha, 0.
            for j in np.argsort(energies):
                mass = min(remaining, p[i, j])
                total += mass*energies[j]
                remaining -= mass
                if remaining <= 0:
                    break
            np.testing.assert_allclose(total/alpha, candidate["metrics"][f"cvar_{alpha:g}_w2"], atol=1e-8)
        mae_rows.append(row)
    assert max(errors) < 1e-10
    # Recompute all grid losses with per-state tail sets, independent of the
    # production cumulative-difference implementation.
    order = np.argsort(energies)
    losses = {"mean_w2": (p*energies).sum(axis=1)}
    for k in (4, 16):
        losses[f"best_of_{k}_w2"] = sum(e * (p[:, energies >= e].sum(axis=1)**k - p[:, energies > e].sum(axis=1)**k) for e in energies)
    for alpha in (.1, .25, .5):
        remaining, value = np.full(len(p), alpha), np.zeros(len(p))
        for j in order:
            mass = np.minimum(remaining, p[:, j])
            value += mass*energies[j]
            remaining -= mass
        losses[f"cvar_{alpha:g}_w2"] = value/alpha
    for loss, values in losses.items():
        stored = next(c for c in result["candidates"] if c["name"] == f"grid_{loss}")
        assert abs(values[stored["grid_index"]] - min(values)) < 1e-8
    receipt = {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "summary_sha256": sha(folder/"summary.json"), "audit_code_sha256": sha(Path(__file__)),
        "circuit_source_sha256": sha(ROOT/"src/quantum_nilm/categorical_ibm.py"),
        "selected_circuit_checks": len(errors), "max_physical_probability_error": max(errors),
        "grid_loss_checks": len(losses), "candidates_per_loss": len(p),
        "label_use": "Known synthetic appliance truth used only for evaluation below, after loss-based angle selection",
        "evaluation_only": mae_rows,
        "limitations": ["No hardware or held-out validation", "Grid minima are not continuous global optimality proofs"]}
    with target.open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    audit(parser.parse_args().folder.resolve())
