"""Bounded ideal one-hot penalty/X controls for the Stage D mixer ablation.

This is additive research code. It does not change the archived W/XY circuit,
objective, optimizer or IBM runner. Full binary simulation is capped at twelve
qubits. Training minimizes the full penalized expected cost, with no labels or
optimal state passed to parameter selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .categorical_ibm import prepare_uniform_onehot
from .categorical_qaoa import encode_onehot


MAX_QUBITS = 12
GAMMA_BOUNDS = (0.0, 8.0 * np.pi)
BETA_BOUNDS = (0.0, np.pi)
ARMS = ("w_xy", "h_x_penalty", "w_x_penalty")


@dataclass(frozen=True)
class PenaltyMixerProblem:
    base: object
    barrier: float
    base_coefficient_l1: float
    linear: np.ndarray
    quadratic: dict
    constant: float
    scale: float
    energies: np.ndarray
    base_binary_energies: np.ndarray
    violations: np.ndarray
    bits: np.ndarray
    feasible_indices: np.ndarray


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _qubo_energies(bits, constant, linear, quadratic):
    values = np.full(len(bits), float(constant))
    for qubit, coefficient in enumerate(linear):
        values += float(coefficient) * bits[:, qubit]
    for (left, right), coefficient in quadratic.items():
        values += float(coefficient) * bits[:, left] * bits[:, right]
    return values


def prepare_penalty_problem(base):
    """Add A sum_r(sum_{q in r} x_q - 1)^2 with coefficient-only A.

    For S=sum|base linear|+sum|base quadratic|, every base binary energy lies
    in [constant-S, constant+S]. A=1+2S makes every invalid penalized energy
    at least constant+S+1, strictly above every feasible energy. This bound
    needs neither a minimum energy nor a reference state. The augmented QUBO
    is used for phase normalization, retaining the scalar constant for costs.
    """
    if base.num_qubits > MAX_QUBITS:
        raise ValueError(f"Stage D full binary simulation is capped at {MAX_QUBITS} qubits")
    width = _integer(base.num_qubits, "num_qubits", 1)
    basis = np.arange(1 << width, dtype=np.uint64)
    bits = ((basis[:, None] >> np.arange(width, dtype=np.uint64)) & 1).astype(np.int8)
    base_l1 = float(np.abs(base.linear).sum() + sum(abs(value) for value in base.quadratic.values()))
    barrier = 1.0 + 2.0 * base_l1
    linear = np.asarray(base.linear, dtype=float).copy()
    quadratic = dict(base.quadratic)
    constant = float(base.constant)
    violations = np.zeros(len(bits), dtype=np.int64)
    for offset, size in zip(base.register_offsets, base.register_sizes):
        offset, size = int(offset), int(size)
        occupancy = bits[:, offset:offset + size].sum(axis=1, dtype=np.int64)
        violations += (occupancy - 1)**2
        constant += barrier
        linear[offset:offset + size] -= barrier
        for left in range(offset, offset + size):
            for right in range(left + 1, offset + size):
                quadratic[(left, right)] = quadratic.get((left, right), 0.0) + 2.0 * barrier
    scale = max(1.0, float(np.abs(linear).sum() + sum(abs(value) for value in quadratic.values())))
    original = _qubo_energies(bits, base.constant, base.linear, base.quadratic)
    energies = original + barrier * violations
    indices = np.asarray([int(encode_onehot(states, base.state_counts), 2) for states in base.states], dtype=np.int64)
    if not all(np.isfinite(value) for value in (barrier, constant, scale)) or not np.all(np.isfinite(energies)):
        raise ValueError("Penalized QUBO exceeds floating-point range")
    return PenaltyMixerProblem(base, barrier, base_l1, linear, quadratic, constant,
                               scale, energies, original, violations, bits, indices)


def audit_penalty_problem(problem):
    """Independent enumeration checks; outputs are never optimization inputs."""
    base = problem.base
    feasible_mask = problem.violations == 0
    if set(np.flatnonzero(feasible_mask)) != set(problem.feasible_indices.tolist()):
        raise ValueError("Penalty feasibility does not match categorical encoding")
    direct_augmented = _qubo_energies(problem.bits, problem.constant, problem.linear, problem.quadratic)
    expansion_error = float(np.max(np.abs(direct_augmented - problem.energies)))
    feasible_error = float(np.max(np.abs(problem.energies[problem.feasible_indices] - base.energies)))
    # Forward-error guard from the sum of magnitudes, not a tolerance chosen
    # after inspecting which state minimizes the objective.
    tolerance = 128 * np.finfo(float).eps * max(1.0, abs(problem.constant) + problem.scale)
    if expansion_error > tolerance or feasible_error > tolerance:
        raise ValueError("Penalized QUBO expansion or feasible objective equivalence failed")
    invalid_minimum = float(problem.energies[~feasible_mask].min())
    feasible_maximum = float(problem.energies[feasible_mask].max())
    feasible_minimum = float(problem.energies[feasible_mask].min())
    if not invalid_minimum > feasible_maximum:
        raise ValueError("Conservative barrier failed to dominate every feasible state")
    return {
        "status": "passed", "num_binary_states": len(problem.energies),
        "num_feasible_states": len(problem.feasible_indices),
        "barrier": problem.barrier, "base_coefficient_l1": problem.base_coefficient_l1,
        "barrier_uses_exact_or_truth": False,
        "invalid_minimum": invalid_minimum, "feasible_maximum": feasible_maximum,
        "feasible_minimum": feasible_minimum,
        "invalid_minus_best_feasible": invalid_minimum - feasible_minimum,
        "invalid_minus_worst_feasible": invalid_minimum - feasible_maximum,
        "feasible_objective_max_absolute_error": feasible_error,
        "augmented_qubo_max_absolute_error": expansion_error,
        "floating_point_tolerance": tolerance,
        "xy_coefficient_scale": float(base.scale), "penalty_coefficient_scale": problem.scale,
        "penalty_to_xy_scale_ratio": problem.scale / base.scale,
        "xy_feasible_phase_range_per_unit_gamma": float(np.ptp(base.energies) / base.scale),
        "penalty_feasible_phase_range_per_unit_gamma": float(np.ptp(base.energies) / problem.scale),
    }


def _angles(gammas, betas):
    gamma = np.asarray(gammas, dtype=float)
    beta = np.asarray(betas, dtype=float)
    if gamma.ndim != 1 or beta.shape != gamma.shape or not gamma.size or not np.all(np.isfinite(gamma)) or not np.all(np.isfinite(beta)):
        raise ValueError("Angles must be nonempty finite matching vectors")
    return gamma, beta


def penalty_mixer_statevector(problem, gammas, betas, initialization):
    """Simulate the full H/X or W/X circuit, without postselection."""
    gamma, beta = _angles(gammas, betas)
    if initialization == "h":
        state = np.full(len(problem.energies), 1 / np.sqrt(len(problem.energies)), dtype=complex)
    elif initialization == "w":
        state = np.zeros(len(problem.energies), dtype=complex)
        state[problem.feasible_indices] = 1 / np.sqrt(len(problem.feasible_indices))
    else:
        raise ValueError("initialization must be h or w")
    normalized = problem.energies / problem.scale
    for cost_angle, mixer_angle in zip(gamma, beta):
        state *= np.exp(-1j * cost_angle * normalized)
        cosine, sine = np.cos(mixer_angle), -1j * np.sin(mixer_angle)
        for qubit in range(problem.base.num_qubits):
            blocks = state.reshape(-1, 2, 1 << qubit)
            zero, one = blocks[:, 0, :].copy(), blocks[:, 1, :].copy()
            blocks[:, 0, :] = cosine * zero + sine * one
            blocks[:, 1, :] = sine * zero + cosine * one
    return state


def penalty_mixer_probabilities(problem, gammas, betas, initialization):
    probabilities = np.abs(penalty_mixer_statevector(problem, gammas, betas, initialization))**2
    total = float(probabilities.sum())
    if abs(total - 1) > 1e-9 or not np.isfinite(total):
        raise ValueError("Full binary circuit lost normalization")
    return probabilities / total


def build_penalty_mixer_circuit(problem, gammas, betas, initialization, *, measure=False):
    from qiskit import QuantumCircuit
    gamma, beta = _angles(gammas, betas)
    circuit = QuantumCircuit(problem.base.num_qubits)
    if initialization == "h":
        for qubit in range(problem.base.num_qubits):
            circuit.h(qubit)
    elif initialization == "w":
        for offset, size in zip(problem.base.register_offsets, problem.base.register_sizes):
            prepare_uniform_onehot(circuit, range(int(offset), int(offset + size)))
    else:
        raise ValueError("initialization must be h or w")
    fields = -problem.linear / 2
    for (left, right), coefficient in problem.quadratic.items():
        fields[left] -= coefficient / 4
        fields[right] -= coefficient / 4
    for cost_angle, mixer_angle in zip(gamma, beta):
        for qubit, field in enumerate(fields):
            if field:
                circuit.rz(float(2 * cost_angle * field / problem.scale), qubit)
        for (left, right), coefficient in sorted(problem.quadratic.items()):
            if coefficient:
                circuit.rzz(float(cost_angle * coefficient / (2 * problem.scale)), left, right)
        for qubit in range(problem.base.num_qubits):
            circuit.rx(float(2 * mixer_angle), qubit)
    if measure:
        circuit.measure_all()
    return circuit


def compile_and_audit(circuit, expected_probabilities):
    """Audit the independently compiled full physical circuit and its cost."""
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    started = perf_counter()
    compiled = transpile(circuit, basis_gates=["rz", "sx", "x", "cx"],
                         optimization_level=1, seed_transpiler=17)
    compilation_seconds = perf_counter() - started
    tick = perf_counter()
    observed = Statevector.from_instruction(compiled).probabilities()
    expected = np.asarray(expected_probabilities)
    if observed.shape != expected.shape:
        raise ValueError("Compiled circuit has unexpected physical width")
    error = float(np.max(np.abs(observed - expected)))
    mass_error = abs(float(observed.sum()) - 1.0)
    if error > 1e-10 or mass_error > 1e-10:
        raise ValueError("NumPy and independent compiled Qiskit circuits disagree")
    return {
        "num_qubits": compiled.num_qubits, "compiled_depth": compiled.depth(),
        "compiled_gate_counts": dict(compiled.count_ops()),
        "basis_gates": ["rz", "sx", "x", "cx"], "optimization_level": 1,
        "transpile_seed": 17, "connectivity": "all-to-all; no routing",
        "compilation_seconds": compilation_seconds, "audit_seconds": perf_counter() - tick,
        "audit_max_probability_error": error, "audit_probability_mass_error": mass_error,
    }


def optimize_penalty_mixer(problem, depth, initialization, restart_seeds=(7001, 7002), evaluations_per_restart=512):
    """Use the frozen Stage B search scheme on a full penalized expectation.

    The fixed-budget algorithm is implemented locally with an explicit
    probability callback, rather than modifying or monkey-patching Stage B.
    Zero angles produce the corresponding H or W initialization candidate.
    """
    from scipy.optimize import minimize
    layers = _integer(depth, "depth", 1)
    budget = _integer(evaluations_per_restart, "evaluations_per_restart", 34)
    seeds = tuple(_integer(value, "restart seed") for value in restart_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("restart seeds must be nonempty and unique")
    if initialization not in ("h", "w"):
        raise ValueError("initialization must be h or w")
    bounds = [GAMMA_BOUNDS] * layers + [BETA_BOUNDS] * layers
    lower, upper = np.asarray(bounds).T
    tolerance = 8 * np.finfo(float).eps * np.maximum(1, np.maximum(np.abs(lower), np.abs(upper)))
    allowances = ((budget - 32) // 2, (budget - 32 + 1) // 2)
    begun = perf_counter()
    records, global_best = [], None
    for seed in seeds:
        started = perf_counter()
        rng = np.random.default_rng(seed)
        history = []
        simulation_seconds = 0.0
        corrections = coordinates = 0
        max_correction = 0.0
        best = None

        def evaluate(parameters):
            nonlocal simulation_seconds, corrections, coordinates, max_correction, best
            if len(history) >= budget:
                raise RuntimeError("Objective call budget exceeded")
            values = np.asarray(parameters, dtype=float)
            if values.shape != lower.shape or not np.all(np.isfinite(values)):
                raise ValueError("Nonfinite or out-of-bounds optimizer angles")
            outside = (values < lower) | (values > upper)
            if np.any(outside):
                excursion = np.maximum(lower - values, values - upper)
                if np.any(excursion > tolerance):
                    raise ValueError("Nonfinite or out-of-bounds optimizer angles")
                corrections += 1
                coordinates += int(np.count_nonzero(outside))
                max_correction = max(max_correction, float(excursion.max()))
                values = np.clip(values, lower, upper)
            tick = perf_counter()
            probabilities = penalty_mixer_probabilities(problem, values[:layers], values[layers:], initialization)
            simulation_seconds += perf_counter() - tick
            expected = float(probabilities @ problem.energies)
            objective = expected / problem.scale
            if not np.isfinite(objective):
                raise ValueError("Nonfinite penalized expectation")
            if best is None or objective < best[0]:
                best = (objective, values.copy(), probabilities.copy(), expected, len(history))
            history.append(objective)
            return objective

        candidates = [np.zeros(2 * layers)]
        candidates.extend(rng.uniform(lower, upper, size=(31, 2 * layers)))
        objectives = [evaluate(value) for value in candidates]
        starts = []
        for index in np.argsort(objectives, kind="stable"):
            candidate = candidates[int(index)]
            if not any(np.array_equal(candidate, other) for other in starts):
                starts.append(candidate)
            if len(starts) == 2:
                break
        if len(starts) != 2:
            raise RuntimeError("Distinct local-search starts unavailable")
        refinements, padding_count = [], 0
        for start, allowance in zip(starts, allowances):
            before = len(history)
            result = minimize(evaluate, start, method="Powell", bounds=bounds,
                              options={"maxfev": allowance, "xtol": 1e-5, "ftol": 1e-8})
            spent = len(history) - before
            if spent > allowance or result.nfev != spent:
                raise RuntimeError("Powell call accounting failed")
            padding = allowance - spent
            for _ in range(padding):
                evaluate(rng.uniform(lower, upper))
            padding_count += padding
            refinements.append({"start_gammas": start[:layers].tolist(), "start_betas": start[layers:].tolist(),
                                "allowance": allowance, "evaluations": spent, "padding_evaluations": padding,
                                "success": bool(result.success), "status": int(result.status), "message": str(result.message)})
        if len(history) != budget:
            raise RuntimeError("Incomplete objective call budget")
        record = {
            "seed": seed, "gammas": best[1][:layers].tolist(), "betas": best[1][layers:].tolist(),
            "expected_cost": best[3], "normalized_expected_cost": best[0], "evaluations": len(history),
            "initial_evaluations": 32, "padding_evaluations": padding_count,
            "best_evaluation_index": best[4], "runtime_seconds": perf_counter() - started,
            "statevector_seconds": simulation_seconds, "objective_history": history,
            "refinements": refinements, "status": "completed_fixed_budget",
            "boundary_roundoff_evaluations": corrections, "boundary_roundoff_coordinates": coordinates,
            "boundary_max_correction": max_correction,
        }
        records.append(record)
        if global_best is None or best[0] < global_best[0][0]:
            global_best = (best, len(records) - 1)
    best, index = global_best
    return {
        "initialization": initialization, "depth": layers,
        "gammas": best[1][:layers].tolist(), "betas": best[1][layers:].tolist(),
        "probabilities": best[2].tolist(), "expected_cost": best[3],
        "normalized_expected_cost": best[0], "chosen_restart_index": index,
        "chosen_seed": seeds[index], "evaluations": sum(value["evaluations"] for value in records),
        "runtime_seconds": perf_counter() - begun,
        "statevector_seconds": sum(value["statevector_seconds"] for value in records),
        "status": "completed_fixed_budget", "restarts": records,
        "optimizer": {"method": "32_candidates_then_two_bounded_powell_with_random_padding_v1",
                      "gamma_bounds": list(GAMMA_BOUNDS), "beta_bounds": list(BETA_BOUNDS),
                      "evaluations_per_restart": budget, "restart_seeds": list(seeds),
                      "initial_candidates": 32, "refinement_budgets": list(allowances),
                      "xtol": 1e-5, "ftol": 1e-8, "warm_start_from_lower_depth": False,
                      "objective": "full_binary_penalized_expected_cost_divided_by_augmented_coefficient_scale",
                      "exact_optimum_used_in_training": False, "planted_truth_used_in_training": False,
                      "boundary_roundoff_tolerance": tolerance.tolist(),
                      "boundary_roundoff_rule": "Snap only excursions <=8*eps*max(1,abs(lower),abs(upper)); reject larger violations"},
    }
