#!/usr/bin/env python3
"""Offline loss-function diagnosis on the frozen six-qubit synthetic example.

No production training changes, dataset-label fitting, accounts or hardware.
The fixed dense grid is exploratory, not a matched-budget solver benchmark.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import itertools
import json
from pathlib import Path

import numpy as np
from quantum_nilm.categorical_qaoa import prepare_categorical_problem, categorical_qaoa_probabilities

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/quantum_diagnostics/six_qubit_001"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def metrics(probabilities, energies):
    """Exact finite-distribution mean, lower CVaR and iid best-of-K cost."""
    p = np.atleast_2d(probabilities)
    order = np.argsort(energies, kind="stable")
    e, q = energies[order], p[:, order]
    assert np.isfinite(p).all() and p.min() >= 0
    assert np.max(abs(p.sum(axis=1) - 1)) < 1e-10
    before = np.cumsum(q, axis=1) - q
    survival = np.clip(1 - np.cumsum(q, axis=1)[:, :-1], 0, 1)
    values = {"mean_w2": p @ energies,
              "optimum_probability": p[:, np.isclose(energies, min(energies), atol=1e-8, rtol=0)].sum(axis=1)}
    for alpha in (.1, .25, .5):
        mass = np.minimum(q, np.maximum(0, alpha - before))
        values[f"cvar_{alpha:g}_w2"] = mass @ e / alpha
    for k in (1, 2, 4, 16, 64, 256):
        values[f"best_of_{k}_w2"] = e[0] + survival**k @ np.diff(e)
        values[f"optimum_hit_{k}"] = -np.expm1(k * np.log1p(-values["optimum_probability"]))
    return values


def checks():
    e, p = np.array([0., 2., 9.]), np.array([.2, .3, .5])
    values = metrics(p, e)
    for k in (1, 2, 4):
        brute = sum(np.prod(p[list(indices)]) * min(e[list(indices)])
                    for indices in itertools.product(range(3), repeat=k))
        np.testing.assert_allclose(values[f"best_of_{k}_w2"], brute, atol=1e-12)
    np.testing.assert_allclose(values["cvar_0.25_w2"], .4, atol=1e-12)
    np.testing.assert_allclose(values["cvar_0.5_w2"], 1.2, atol=1e-12)


def grid_probabilities(problem, angles):
    """Batch evolution, cross-checked against the existing scalar simulator."""
    state = np.exp(-1j * angles[:, :1] * problem.energies / problem.scale) / np.sqrt(len(problem.energies))
    c, s = np.cos(angles[:, 1]), -1j*np.sin(angles[:, 1])
    stride = 1
    for count in problem.register_sizes:
        for category in range(count - 1):
            for k in range(len(problem.energies)):
                if k // stride % count == category:
                    other = k + stride
                    left, right = state[:, k].copy(), state[:, other].copy()
                    state[:, k], state[:, other] = c*left + s*right, s*left + c*right
        stride *= count
    p = abs(state)**2
    assert np.max(abs(p.sum(axis=1)-1)) < 1e-10
    return p / p.sum(axis=1, keepdims=True)


def analyze(output):
    assert output.is_relative_to(ROOT / "results/quantum_diagnostics") and not output.exists()
    inputs = [BASE / "plan.json", BASE / "training.json", Path(__file__),
              ROOT / "src/quantum_nilm/categorical_qaoa.py"]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in inputs}
    plan = json.loads((BASE / "plan.json").read_text())
    training = json.loads((BASE / "training.json").read_text())
    info = plan["problems"]["six_p1"]
    problem = prepare_categorical_problem(info["aggregate_w"], info["levels_w"],
                                          info["switch_penalty_w2"], info["segment_weights"])
    # Every loss ranks exactly the same 129 x 129 grid plus the archived fit.
    angles = np.array(list(itertools.product(np.linspace(0, 8*np.pi, 129), np.linspace(0, np.pi, 129))))
    angles = np.vstack([angles, [training["gammas"][0], training["betas"][0]]])
    probabilities = grid_probabilities(problem, angles)
    values = metrics(probabilities, problem.energies)
    selected = {"uniform": 0, "archived_mean_fit": len(angles)-1}
    plateau_counts = {}
    for loss in ("mean_w2", "cvar_0.1_w2", "cvar_0.25_w2", "cvar_0.5_w2", "best_of_4_w2", "best_of_16_w2"):
        ties = np.flatnonzero(values[loss] <= min(values[loss]) + 1e-8)
        # Declared secondary criterion for flat CVaR plateaus: mean cost.
        selected[f"grid_{loss}"] = int(ties[np.argmin(values["mean_w2"][ties])])
        plateau_counts[loss] = len(ties)
    selected["oracle_max_optimum_probability"] = int(np.argmax(values["optimum_probability"]))
    errors = []
    for i in sorted(set(selected.values()) | set(np.linspace(0, len(angles)-1, 17, dtype=int))):
        reference = categorical_qaoa_probabilities(problem, [angles[i, 0]], [angles[i, 1]])
        errors.append(float(max(abs(reference - probabilities[i]))))
    assert max(errors) < 1e-10
    np.testing.assert_allclose(probabilities[-1], training["probabilities"], atol=1e-12)
    np.testing.assert_allclose(values["mean_w2"], values["best_of_1_w2"], atol=1e-8)
    checks()
    candidates = [{"name": name, "grid_index": i, "gamma": float(angles[i, 0]), "beta": float(angles[i, 1]),
                   "metrics": {key: float(value[i]) for key, value in values.items()},
                   "probabilities": probabilities[i].tolist()} for name, i in selected.items()]
    rows = []
    for i, energy in enumerate(problem.energies):
        watts = [float(problem.levels[j][problem.states[i, 0, j]]) for j in range(2)]
        rows.append({"index": i, "load_w": watts, "aggregate_prediction_w": sum(watts),
                     "cost_w2": float(energy), "uniform_probability": 1/9,
                     "trained_probability": float(probabilities[-1, i]),
                     "mean_cost_improvement_contribution_w2": float((1/9-probabilities[-1, i])*energy)})
    assert abs(sum(row["mean_cost_improvement_contribution_w2"] for row in rows)
               - (values["mean_w2"][0]-values["mean_w2"][-1])) < 1e-7
    for name, expected in hashes.items():
        assert digest(ROOT/name) == expected
    output.mkdir(parents=True)
    with (output / "grid.npz").open("xb") as handle:
        np.savez_compressed(handle, angles=angles, probabilities=probabilities, energies=problem.energies)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "execution": "offline exploratory objective diagnosis",
              "hardware_jobs": 0, "source_sha256": hashes, "grid_candidates": len(angles),
              "grid_rule": "129 x 129 endpoints included, gamma 0..8pi, beta 0..pi, plus archived angles; no local refinement",
              "tie_rule": "Within 1e-8 W² of minimum loss, use lowest mean cost; then first grid index",
              "grid_sha256": digest(output/"grid.npz"), "candidates": candidates, "loss_plateau_counts": plateau_counts,
              "state_breakdown": rows, "checks": {"status": "passed", "scalar_crosschecks": len(errors),
              "max_probability_error": max(errors), "order_statistics": "brute-force K=1,2,4 on a 3-state fixture",
              "cvar": "fractional quantile mass checked on 3-state fixture", "contributions": "reconcile full mean-cost change"},
              "limitations": ["One synthetic instance, noiseless p=1, not held-out or hardware validation",
              "Grid uses more candidates than archived training; not a matched-budget optimizer comparison or a global optimum proof",
              "All grid loss comparisons use identical candidate distributions",
              "Oracle optimum-probability maximization is diagnostic only, not a deployable loss without an optimum oracle",
              "Exact best-of-K projections assume independent samples and no invalid states; hardware needs explicit failure handling",
              "No production parameters or paper results changed"]}
    with (output / "summary.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"checks": result["checks"], "candidates": [{"name": c["name"], "metrics": c["metrics"]} for c in candidates]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    analyze(parser.parse_args().output_dir.resolve())
