"""Generic matched-objective categorical NILM MILP for Stage E.

This is an independently implemented MILP baseline, not a reproduction of a
published Balletti/Li NILM implementation. It minimizes precisely the weighted
squared residual plus categorical change charges used by ``multistate.py``.
It does not invoke dynamic programming, obtain a warm start from another
solver, or use appliance reference measurements.

SciPy's documented ``milp`` interface wraps HiGHS:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array


MAX_JOINT_STATES = 4096
MAX_VARIABLES = 1_000_000
MAX_NONZEROS = 12_000_000


@dataclass(frozen=True)
class CategoricalMilpResult:
    """Solver outcome, including unsuccessful or time-limited executions.

    ``energy`` is recomputed from a checked discrete incumbent, never copied
    from HiGHS. A failed/absent incumbent gives None, not an invented fallback.
    ``solver_mip_gap`` refers to the shifted/scaled solver objective;
    ``absolute_gap`` and ``relative_gap`` use the restored original objective,
    with relative_gap dividing by max(1, abs(energy)) to handle zero costs.
    Despite the conventional name, absolute_gap is the signed subtraction
    energy - dual_bound: tiny negative roundoff is retained, not clamped.
    ``success`` retains the solver flag and does not assert exact arithmetic.
    """

    states: np.ndarray | None
    predicted_power: np.ndarray | None
    energy: float | None
    status_code: int
    status: str
    success: bool
    message: str
    incumbent_feasible: bool
    incumbent_validation: str
    max_integrality_violation: float | None
    max_bound_violation: float | None
    max_constraint_violation: float | None
    decoded_minus_solver_objective: float | None
    bound_validation: str
    raw_solver_objective: float | None
    raw_solver_dual_bound: float | None
    solver_objective: float | None
    dual_bound: float | None
    absolute_gap: float | None
    relative_gap: float | None
    solver_mip_gap: float | None
    mip_node_count: int | None
    objective_offset: float
    objective_scale: float
    states_per_segment: int
    binary_variables: int
    continuous_variables: int
    constraints: int
    constraint_nonzeros: int
    build_time_s: float
    solve_time_s: float
    decode_time_s: float
    wall_time_s: float
    scipy_version: str
    options: dict


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
    return result


def _optional_finite(value):
    return None if value is None or not np.isfinite(value) else float(value)


def solve_categorical_milp(
    aggregate,
    levels,
    switch_penalty,
    segment_weights=None,
    *,
    previous_state=None,
    time_limit_s=30.0,
    mip_rel_gap=0.0,
    presolve=True,
    node_limit=None,
) -> CategoricalMilpResult:
    """Solve the categorical temporal objective by an exact MILP formulation.

    Binary y[t,j] selects one complete joint categorical state j per interval.
    Emissions are precomputed as w[t]*(aggregate[t]-joint_power[j])**2.
    For each boundary, channel i and category k the linear constraint
    z[t,i] >= sum_j y[t-1,j] 1[j_i=k] - sum_j y[t,j] 1[j_i=k]
    forces z to one for a changed category. Continuous 0<=z<=1 suffices:
    nonnegative penalty[i]*z minimizes to the exact categorical change cost.
    If a penalty is zero the auxiliary z can be arbitrary without changing
    the objective. No change charge precedes the first interval unless an
    explicit ``previous_state`` is supplied by the caller's own history.

    The minimum emission at each interval is subtracted as a constant. The
    remaining coefficients are scaled to a maximum of at most 10,000 (no
    upward scaling). Bounds and reported objectives have both transformations
    reversed. Returned energy is independently recomputed in original units.
    Status zero means optimal within HiGHS's floating-point/tolerance limits;
    ties need not match another solver's chosen path. ``time_limit_s`` limits
    the optimizer, not Python model-building time. No automatic retries occur.
    All structurally valid inputs have at least one feasible path; invalid
    input is rejected rather than passed off as solver infeasibility.
    Candidate integrality, variable-bound and constraint violations are
    checked against 1e-6 in their dimensionless native linear-model units.
    The decoded objective comparison tolerance is 1e-6 * max(1, abs(decoded
    energy), abs(restored solver objective)) in original energy units. All
    discrepancies are retained quantitatively; this validation does not
    replace an independent optimal-objective comparison. No thread count is
    overridden: only the documented SciPy options listed in ``options`` apply.
    """
    started = perf_counter()
    measured = _vector(aggregate, "aggregate")
    if not isinstance(levels, (list, tuple)) or not levels:
        raise ValueError("levels must be a non-empty list of level arrays")
    channel_levels = [_vector(value, f"levels[{i}]") for i, value in enumerate(levels)]
    counts = [len(value) for value in channel_levels]
    n_channels, n_segments = len(counts), len(measured)
    n_states = 1
    for count in counts:
        n_states *= count
        if n_states > MAX_JOINT_STATES:
            raise ValueError(f"MILP is limited to {MAX_JOINT_STATES} joint states")
    penalties = np.asarray(switch_penalty, dtype=float)
    if penalties.ndim == 0:
        penalties = np.full(n_channels, float(penalties))
    if penalties.shape != (n_channels,) or not np.all(np.isfinite(penalties)) or np.any(penalties < 0):
        raise ValueError("switch_penalty must be finite nonnegative with one value per channel")
    weights = np.ones(n_segments) if segment_weights is None else _vector(segment_weights, "segment_weights")
    if weights.shape != (n_segments,) or np.any(weights <= 0):
        raise ValueError("segment_weights must be positive and match aggregate")
    prior = None
    if previous_state is not None:
        raw_prior = np.asarray(previous_state)
        if raw_prior.shape != (n_channels,) or raw_prior.dtype.kind not in "iu":
            raise ValueError("previous_state must contain one integer category per channel")
        if np.any(raw_prior < 0) or np.any(raw_prior >= counts):
            raise ValueError("previous_state category out of range")
        prior = raw_prior.astype(np.int32)
    if isinstance(time_limit_s, (bool, np.bool_)) or not np.isscalar(time_limit_s) or not np.isfinite(time_limit_s) or time_limit_s <= 0:
        raise ValueError("time_limit_s must be finite and positive")
    if isinstance(mip_rel_gap, (bool, np.bool_)) or not np.isscalar(mip_rel_gap) or not np.isfinite(mip_rel_gap) or mip_rel_gap < 0:
        raise ValueError("mip_rel_gap must be finite and nonnegative")
    if not isinstance(presolve, (bool, np.bool_)):
        raise ValueError("presolve must be boolean")
    if node_limit is not None and (isinstance(node_limit, (bool, np.bool_)) or not isinstance(node_limit, (int, np.integer)) or node_limit < 0):
        raise ValueError("node_limit must be a nonnegative integer")
    options = {"time_limit": float(time_limit_s), "mip_rel_gap": float(mip_rel_gap), "presolve": bool(presolve)}
    if node_limit is not None:
        options["node_limit"] = int(node_limit)

    n_binary = n_segments * n_states
    n_continuous = (n_segments - 1) * n_channels
    n_variables = n_binary + n_continuous
    n_constraints = n_segments + (n_segments - 1) * sum(counts)
    n_nonzeros = n_binary + (n_segments - 1) * (2 * n_channels * n_states + sum(counts))
    if n_variables > MAX_VARIABLES or n_nonzeros > MAX_NONZEROS:
        raise ValueError(f"MILP resource cap exceeded: {n_variables} variables, {n_nonzeros} nonzeros")

    state_ids = np.arange(n_states, dtype=np.int32)
    joint = np.empty((n_states, n_channels), dtype=np.int32)
    stride = 1
    for channel, count in enumerate(counts):
        joint[:, channel] = (state_ids // stride) % count
        stride *= count
    try:
        with np.errstate(over="raise", invalid="raise"):
            joint_power = np.zeros(n_states)
            for channel, values in enumerate(channel_levels):
                joint_power += values[joint[:, channel]]
            emissions = weights[:, None] * (measured[:, None] - joint_power[None, :]) ** 2
            if prior is not None:
                emissions[0] += np.sum(penalties * (joint != prior), axis=1)
            minimum = np.min(emissions, axis=1)
            offset = float(np.sum(minimum))
            shifted = emissions - minimum[:, None]
            scale = max(1.0, float(np.max(shifted)) / 10_000, float(np.max(penalties)) / 10_000)
            coefficients = np.concatenate((shifted.ravel(), np.tile(penalties, n_segments - 1))) / scale
            # Reject an objective whose attainable upper bound overflows too,
            # even if one low-emission path could happen to remain finite.
            upper_bound = float(np.sum(np.max(emissions, axis=1)) + (n_segments - 1) * np.sum(penalties))
            if not np.isfinite(offset + scale) or not np.isfinite(upper_bound):
                raise FloatingPointError("Objective overflow")
    except FloatingPointError as error:
        raise ValueError("Objective exceeds finite floating-point range") from error

    # Each joint binary appears once in its selection equation and once in
    # each relevant channel marginal, independently of category count.
    rows = [np.repeat(np.arange(n_segments), n_states)]
    columns = [np.arange(n_binary)]
    values = [np.ones(n_binary)]
    category_offset = 0
    boundaries = np.arange(n_segments - 1)
    for channel, count in enumerate(counts):
        marginal_rows = n_segments + boundaries[:, None] * sum(counts) + category_offset + joint[None, :, channel]
        previous_columns = boundaries[:, None] * n_states + state_ids[None, :]
        rows.extend((marginal_rows.ravel(), marginal_rows.ravel()))
        columns.extend((previous_columns.ravel(), (previous_columns + n_states).ravel()))
        values.extend((np.ones((n_segments - 1) * n_states), -np.ones((n_segments - 1) * n_states)))
        change_rows = n_segments + boundaries[:, None] * sum(counts) + category_offset + np.arange(count)[None, :]
        change_columns = np.broadcast_to((n_binary + boundaries * n_channels + channel)[:, None], change_rows.shape)
        rows.append(change_rows.ravel())
        columns.append(change_columns.ravel())
        values.append(-np.ones((n_segments - 1) * count))
        category_offset += count
    matrix = coo_array((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(n_constraints, n_variables)).tocsc()
    lower = np.concatenate((np.ones(n_segments), np.full(n_constraints - n_segments, -np.inf)))
    upper = np.concatenate((np.ones(n_segments), np.zeros(n_constraints - n_segments)))
    integrality = np.concatenate((np.ones(n_binary, dtype=np.int32), np.zeros(n_continuous, dtype=np.int32)))
    build_time = perf_counter() - started
    solving = perf_counter()
    # SciPy may pop translated entries (e.g., node_limit); preserve the exact
    # requested options in the report by passing a separate dictionary.
    result = milp(coefficients, integrality=integrality, bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)), constraints=LinearConstraint(matrix, lower, upper), options=dict(options))
    solve_time = perf_counter() - solving

    decoding = perf_counter()
    code = int(result.status)
    status = {0: "optimal", 1: "limit_reached", 2: "infeasible", 3: "unbounded", 4: "solver_error"}.get(code, "unknown")
    solver_fun = _optional_finite(getattr(result, "fun", None))
    solver_bound = _optional_finite(getattr(result, "mip_dual_bound", None))
    restored_fun = None if solver_fun is None else offset + scale * solver_fun
    restored_bound = None if solver_bound is None else offset + scale * solver_bound
    states = predicted = energy = None
    valid = False
    validation = "no_incumbent"
    integrality_violation = bound_violation = constraint_violation = discrepancy = None
    candidate = getattr(result, "x", None)
    if candidate is not None:
        candidate = np.asarray(candidate)
        if candidate.shape != (n_variables,) or not np.all(np.isfinite(candidate)):
            validation = "nonfinite_or_wrong_shape"
        else:
            products = matrix @ candidate
            integrality_violation = float(np.max(np.abs(candidate[:n_binary] - np.rint(candidate[:n_binary]))))
            bound_violation = float(max(0.0, -np.min(candidate), np.max(candidate) - 1.0))
            constraint_violation = float(max(0.0, np.max(lower - products), np.max(products - upper)))
            if bound_violation > 1e-6:
                validation = "bound_violation"
            elif integrality_violation > 1e-6:
                validation = "nonintegral_incumbent"
            elif constraint_violation > 1e-6:
                validation = "constraint_violation"
            else:
                selected = np.argmax(candidate[:n_binary].reshape(n_segments, n_states), axis=1)
                states = joint[selected].copy()
                predicted = np.column_stack([channel_levels[i][states[:, i]] for i in range(n_channels)])
                energy = float(np.sum(weights * (measured - predicted.sum(axis=1)) ** 2) + np.sum(penalties * (states[1:] != states[:-1])))
                if prior is not None:
                    energy += float(np.sum(penalties * (states[0] != prior)))
                discrepancy = None if restored_fun is None else energy - restored_fun
                # An incumbent's auxiliary objective may overcharge a change
                # indicator before optimality. The discrete path is feasible
                # even then; expose that discrepancy, never hide it.
                tolerance = 1e-6 * max(1.0, abs(energy), abs(restored_fun or 0.0))
                if restored_fun is None:
                    validation = "feasible_path_missing_solver_objective"
                elif energy > restored_fun + tolerance:
                    validation = "recomputed_energy_exceeds_solver_objective"
                elif code == 0 and abs(energy - restored_fun) > tolerance:
                    validation = "optimal_objective_mismatch"
                else:
                    valid = True
                    validation = "passed" if abs(energy - restored_fun) <= tolerance else "passed_path_auxiliary_overcharge"
    # Keep raw diagnostics even if the solver supplied a numerically invalid
    # candidate, but do not expose it as an accepted NILM prediction.
    if not valid:
        states = predicted = energy = None
    absolute_gap = None if energy is None or restored_bound is None else energy - restored_bound
    relative_gap = None if absolute_gap is None else absolute_gap / max(1.0, abs(energy))
    bound_validation = "unavailable"
    if absolute_gap is not None:
        bound_tolerance = 128 * np.finfo(float).eps * max(1.0, abs(energy), abs(restored_bound))
        bound_validation = "passed" if absolute_gap >= 0 else "negative_within_roundoff" if absolute_gap >= -bound_tolerance else "dual_bound_exceeds_incumbent"
    nodes = getattr(result, "mip_node_count", None)
    return CategoricalMilpResult(
        states=states, predicted_power=predicted, energy=energy,
        status_code=code, status=status, success=bool(result.success), message=str(result.message),
        incumbent_feasible=valid, incumbent_validation=validation,
        max_integrality_violation=integrality_violation, max_bound_violation=bound_violation,
        max_constraint_violation=constraint_violation, decoded_minus_solver_objective=discrepancy,
        bound_validation=bound_validation,
        raw_solver_objective=solver_fun, raw_solver_dual_bound=solver_bound,
        solver_objective=restored_fun, dual_bound=restored_bound,
        absolute_gap=absolute_gap, relative_gap=relative_gap,
        solver_mip_gap=_optional_finite(getattr(result, "mip_gap", None)),
        mip_node_count=None if nodes is None else int(nodes), objective_offset=offset, objective_scale=scale,
        states_per_segment=n_states, binary_variables=n_binary, continuous_variables=n_continuous,
        constraints=n_constraints, constraint_nonzeros=int(matrix.nnz), build_time_s=build_time,
        solve_time_s=solve_time, decode_time_s=perf_counter() - decoding, wall_time_s=perf_counter() - started,
        scipy_version=scipy.__version__, options=options,
    )
