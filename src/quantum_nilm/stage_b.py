"""Reproducible synthetic inputs and bounded ideal-QAOA training for Stage B.

This module is additive: it does not change the circuit or objective used by
the archived IBM evaluation. The optimizer uses objective expectations only;
planted states and an exact optimum are not inputs to angle selection. Circuit
expectation evaluation is a classical simulation cost, not a QPU experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .categorical_qaoa import (
    CategoricalQAOAProblem,
    categorical_qaoa_probabilities,
    prepare_categorical_problem,
)


BASE_MAX_POWERS = (90.0, 420.0, 1250.0, 1800.0)
GAMMA_BOUNDS = (0.0, 8.0 * np.pi)
BETA_BOUNDS = (0.0, np.pi)


@dataclass(frozen=True)
class StageBInstance:
    """A paired input with all generation inputs retained for audit."""

    problem: CategoricalQAOAProblem
    truth: np.ndarray
    metadata: dict


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def make_stage_b_instance(state_counts, n_segments, seed, measurement_noise):
    """Generate paired Gaussian-measurement-noise variants without clipping.

    Each channel's maximum is its fixed base power times U(0.9, 1.1); its
    categories are evenly spaced between zero and that maximum. A starting
    category is uniform. Each subsequent interval independently changes each
    channel with probability 0.35, choosing uniformly among its other states.
    Weights are uniform integers 1..8, and transition penalties are 0.02 times
    each channel maximum squared. Noise sigma is a fraction of the largest
    channel maximum. The RNG draws all base inputs before applying sigma, so
    changing measurement_noise alone preserves levels, truth, weights and the
    underlying standard-normal draws. There is no preceding reference state.
    """
    try:
        counts = tuple(_integer(value, "state count", 2) for value in state_counts)
    except TypeError as error:
        raise ValueError("state_counts must be a non-empty sequence") from error
    if not counts or len(counts) > len(BASE_MAX_POWERS):
        raise ValueError("Stage B supports one to four channels")
    segments = _integer(n_segments, "n_segments", 1)
    if segments not in (1, 2):
        raise ValueError("n_segments must be one or two")
    generation_seed = _integer(seed, "seed")
    try:
        sigma = float(measurement_noise)
    except (TypeError, ValueError) as error:
        raise ValueError("measurement_noise must be finite and nonnegative") from error
    if isinstance(measurement_noise, (bool, np.bool_)) or not np.isfinite(sigma) or sigma < 0:
        raise ValueError("measurement_noise must be finite and nonnegative")

    rng = np.random.default_rng(generation_seed)
    maxima = np.asarray(BASE_MAX_POWERS[:len(counts)]) * rng.uniform(0.9, 1.1, len(counts))
    levels = tuple(np.linspace(0.0, maximum, count) for maximum, count in zip(maxima, counts))
    truth = np.empty((segments, len(counts)), dtype=np.int32)
    truth[0] = [rng.integers(count) for count in counts]
    for interval in range(1, segments):
        truth[interval] = truth[interval - 1]
        changes = rng.random(len(counts)) < 0.35
        for channel, count in enumerate(counts):
            if changes[channel]:
                candidate = int(rng.integers(count - 1))
                truth[interval, channel] = candidate + int(candidate >= truth[interval - 1, channel])
    standard_noise = rng.standard_normal(segments)
    weights = rng.integers(1, 9, size=segments)
    clean_aggregate = np.zeros(segments)
    for channel, values in enumerate(levels):
        clean_aggregate += values[truth[:, channel]]
    noise_watts = sigma * float(maxima.max()) * standard_noise
    aggregate = clean_aggregate + noise_watts
    penalties = 0.02 * maxima**2
    problem = prepare_categorical_problem(aggregate, levels, penalties, weights)
    metadata = {
        "generator": "stage_b_paired_categorical_v1",
        "seed": generation_seed,
        "state_counts": list(counts),
        "n_segments": segments,
        "measurement_noise": sigma,
        "noise_reference_maximum_watts": float(maxima.max()),
        "channel_maxima_watts": maxima.tolist(),
        "levels_watts": [values.tolist() for values in levels],
        "truth": truth.tolist(),
        "standard_normal_noise": standard_noise.tolist(),
        "noise_watts": noise_watts.tolist(),
        "clean_aggregate_watts": clean_aggregate.tolist(),
        "aggregate_watts": aggregate.tolist(),
        "segment_weights": weights.tolist(),
        "switch_penalty": penalties.tolist(),
        "transition_probability": 0.35,
        "previous_states": None,
        "clipped": False,
    }
    return StageBInstance(problem=problem, truth=truth, metadata=metadata)


def optimize_stage_b_qaoa(
    problem,
    depth,
    restart_seeds=(7001, 7002),
    evaluations_per_restart=512,
):
    """Train p-layer angles with an exactly capped, restart-matched budget.

    In each restart, 32 initial candidates comprise the uniform circuit and
    31 seeded independent uniform angle vectors. The two best distinct initial
    vectors each seed a bounded Powell search. The remaining budget is split
    equally between those searches (240 each at the registered 512-call
    setting). A search that converges before its cap is padded with independent
    random candidates to consume its full allotment. Every call counts,
    including repeated points and the initial point in each local search.

    Gamma is bounded to [0, 8*pi], beta to [0, pi]. This is a bounded heuristic
    search, not a proof of globally optimal QAOA parameters. p=2 has no p=1
    warm start and incurs no hidden training cost. Uniform is an evaluated
    candidate, ensuring the chosen ideal expectation is no worse than uniform
    up to floating-point error. The best probability vector is retained during
    counted evaluations, so exporting it needs no extra simulation.

    Powell can form a boundary point a few floating-point units outside its
    bounds. Only excursions <= 8*eps*max(1, abs(lower), abs(upper)) per
    coordinate are snapped back to that boundary. Larger violations raise;
    correction counts and maximum displacement are retained for audit.

    SciPy is imported only when this optional experimental optimizer is used.
    The returned object is JSON-safe, including its chosen probabilities.
    """
    from scipy.optimize import minimize

    layers = _integer(depth, "depth", 1)
    budget = _integer(evaluations_per_restart, "evaluations_per_restart", 34)
    try:
        seeds = tuple(_integer(value, "restart seed") for value in restart_seeds)
    except TypeError as error:
        raise ValueError("restart_seeds must be a non-empty sequence") from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("restart_seeds must be non-empty and unique")
    bounds = [GAMMA_BOUNDS] * layers + [BETA_BOUNDS] * layers
    lower = np.asarray([pair[0] for pair in bounds])
    upper = np.asarray([pair[1] for pair in bounds])
    boundary_tolerance = 8 * np.finfo(float).eps * np.maximum(1.0, np.maximum(np.abs(lower), np.abs(upper)))
    initial_count = 32
    refinement_budgets = ((budget - initial_count) // 2, (budget - initial_count + 1) // 2)
    overall_start = perf_counter()
    records = []
    overall_best = None

    for restart_seed in seeds:
        started = perf_counter()
        rng = np.random.default_rng(restart_seed)
        calls = 0
        simulation_seconds = 0.0
        boundary_roundoff_evaluations = 0
        boundary_roundoff_coordinates = 0
        boundary_max_correction = 0.0
        objective_history = []
        best = None

        def evaluate(parameters):
            nonlocal calls, simulation_seconds, best
            nonlocal boundary_roundoff_evaluations, boundary_roundoff_coordinates, boundary_max_correction
            if calls >= budget:
                raise RuntimeError("Optimizer exceeded the frozen objective-call budget")
            values = np.asarray(parameters, dtype=float)
            if values.shape != (2 * layers,) or not np.all(np.isfinite(values)):
                raise ValueError("Optimizer supplied non-finite or out-of-bounds angles")
            below = values < lower
            above = values > upper
            if np.any(below | above):
                excursion = np.maximum(lower - values, values - upper)
                if np.any(excursion > boundary_tolerance):
                    raise ValueError("Optimizer supplied non-finite or out-of-bounds angles")
                boundary_roundoff_evaluations += 1
                boundary_roundoff_coordinates += int(np.count_nonzero(below | above))
                boundary_max_correction = max(boundary_max_correction, float(excursion.max()))
                values = np.clip(values, lower, upper)
            simulation_start = perf_counter()
            probabilities = categorical_qaoa_probabilities(problem, values[:layers], values[layers:])
            simulation_seconds += perf_counter() - simulation_start
            expected_cost = float(np.dot(probabilities, problem.energies))
            objective = expected_cost / problem.scale
            if not np.isfinite(objective):
                raise ValueError("Non-finite QAOA expectation")
            objective_history.append(objective)
            if best is None or objective < best[0]:
                best = (objective, values.copy(), probabilities.copy(), expected_cost, calls)
            calls += 1
            return objective

        initial_parameters = [np.zeros(2 * layers)]
        initial_parameters.extend(rng.uniform(lower, upper, size=(initial_count - 1, 2 * layers)))
        initial_values = [evaluate(parameters) for parameters in initial_parameters]
        # Stable tie handling selects the first evaluated vector. Random
        # vectors are distinct almost surely; explicitly check to avoid a
        # duplicate initial point ever receiving both refinement allotments.
        chosen_starts = []
        for index in np.argsort(initial_values, kind="stable"):
            candidate = initial_parameters[int(index)]
            if not any(np.array_equal(candidate, value) for value in chosen_starts):
                chosen_starts.append(candidate)
            if len(chosen_starts) == 2:
                break
        if len(chosen_starts) != 2:
            raise RuntimeError("Could not choose two distinct initial angle vectors")

        refinements = []
        padding_count = 0
        for initial_parameters, allowance in zip(chosen_starts, refinement_budgets):
            before = calls
            result = minimize(
                evaluate,
                initial_parameters,
                method="Powell",
                bounds=bounds,
                options={"maxfev": allowance, "xtol": 1e-5, "ftol": 1e-8},
            )
            spent = calls - before
            if spent > allowance or int(result.nfev) != spent:
                raise RuntimeError("Powell objective-call accounting did not match its allotment")
            padding = allowance - spent
            for _ in range(padding):
                evaluate(rng.uniform(lower, upper))
            padding_count += padding
            refinements.append({
                "method": "bounded_powell",
                "start_gammas": initial_parameters[:layers].tolist(),
                "start_betas": initial_parameters[layers:].tolist(),
                "allowance": allowance,
                "evaluations": spent,
                "padding_evaluations": padding,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
            })
        if calls != budget:
            raise RuntimeError("Optimizer did not consume the frozen objective-call budget")
        record = {
            "seed": restart_seed,
            "gammas": best[1][:layers].tolist(),
            "betas": best[1][layers:].tolist(),
            "expected_cost": best[3],
            "normalized_expected_cost": best[0],
            "evaluations": calls,
            "initial_evaluations": initial_count,
            "padding_evaluations": padding_count,
            "best_evaluation_index": best[4],
            "runtime_seconds": perf_counter() - started,
            "statevector_seconds": simulation_seconds,
            "boundary_roundoff_evaluations": boundary_roundoff_evaluations,
            "boundary_roundoff_coordinates": boundary_roundoff_coordinates,
            "boundary_max_correction": boundary_max_correction,
            "objective_history": objective_history,
            "refinements": refinements,
            "status": "completed_fixed_budget",
        }
        records.append(record)
        if overall_best is None or best[0] < overall_best[0][0]:
            overall_best = (best, len(records) - 1)

    best, chosen_index = overall_best
    return {
        "depth": layers,
        "gammas": best[1][:layers].tolist(),
        "betas": best[1][layers:].tolist(),
        "probabilities": best[2].tolist(),
        "expected_cost": best[3],
        "normalized_expected_cost": best[0],
        "chosen_restart_index": chosen_index,
        "chosen_seed": seeds[chosen_index],
        "evaluations": sum(record["evaluations"] for record in records),
        "runtime_seconds": perf_counter() - overall_start,
        "statevector_seconds": sum(record["statevector_seconds"] for record in records),
        "status": "completed_fixed_budget",
        "optimizer": {
            "method": "32_candidates_then_two_bounded_powell_with_random_padding_v1",
            "gamma_bounds": list(GAMMA_BOUNDS),
            "beta_bounds": list(BETA_BOUNDS),
            "evaluations_per_restart": budget,
            "restart_seeds": list(seeds),
            "initial_candidates": initial_count,
            "refinement_budgets": list(refinement_budgets),
            "xtol": 1e-5,
            "ftol": 1e-8,
            "warm_start_from_lower_depth": False,
            "objective": "ideal_expected_cost_divided_by_coefficient_scale",
            "exact_optimum_used_in_training": False,
            "planted_truth_used_in_training": False,
            "boundary_roundoff_rule": "Snap only excursions <= 8*eps*max(1,abs(lower),abs(upper)); reject larger violations",
            "boundary_roundoff_tolerance": boundary_tolerance.tolist(),
        },
        "restarts": records,
    }
