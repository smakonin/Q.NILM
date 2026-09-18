"""Exact classical optimization that exploits the temporal NILM structure."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np


MAX_EXACT_APPLIANCES = 20
MAX_TRACEBACK_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class BinaryTemporalExactResult:
    """An exact solution, with the full objective including its constant term."""

    states: np.ndarray
    energy: float
    states_per_segment: int
    wall_time_s: float
    traceback_bytes: int


def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty, finite, one-dimensional array")
    return array


def _hamming_minimum(
    values: np.ndarray, penalties: np.ndarray, state_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute min_x values[x] + sum_i penalties[i] * (x_i != y_i).

    Each butterfly pass permits changing one additional bit. Tracking the
    original state through these passes provides the predecessor for each y.
    Pair updates are simultaneous, so a pass never reuses its own updates.
    Equal costs prefer the smaller predecessor identifier.
    """
    costs = values.copy()
    predecessors = state_ids.copy()
    for bit, penalty in enumerate(penalties):
        half = 1 << bit
        cost_blocks = costs.reshape(-1, 2 * half)
        predecessor_blocks = predecessors.reshape(-1, 2 * half)
        left_cost = cost_blocks[:, :half].copy()
        right_cost = cost_blocks[:, half:].copy()
        left_predecessor = predecessor_blocks[:, :half].copy()
        right_predecessor = predecessor_blocks[:, half:].copy()
        right_candidate = right_cost + penalty
        left_candidate = left_cost + penalty
        replace_left = (right_candidate < left_cost) | (
            (right_candidate == left_cost) & (right_predecessor < left_predecessor)
        )
        replace_right = (left_candidate < right_cost) | (
            (left_candidate == right_cost) & (left_predecessor < right_predecessor)
        )
        cost_blocks[:, :half] = np.where(replace_left, right_candidate, left_cost)
        cost_blocks[:, half:] = np.where(replace_right, left_candidate, right_cost)
        predecessor_blocks[:, :half] = np.where(
            replace_left, right_predecessor, left_predecessor
        )
        predecessor_blocks[:, half:] = np.where(
            replace_right, left_predecessor, right_predecessor
        )
    return costs, predecessors


def solve_binary_temporal_exact(
    aggregate: np.ndarray,
    powers: np.ndarray,
    switch_penalty: float | np.ndarray,
    segment_weights: np.ndarray | None = None,
) -> BinaryTemporalExactResult:
    """Minimize the binary temporal objective exactly using dynamic programming.

    The objective is ``sum_t w[t] * (aggregate[t] - powers @ x[t])**2``
    plus ``sum_{t>0,i} penalty[i] * (x[t,i] - x[t-1,i])**2``. Aggregate
    readings may be signed (for example after subtracting standby loads).
    Powers and switch penalties must be finite and nonnegative; weights must
    be finite and positive. Inputs are one-dimensional, except a scalar
    switch penalty is broadcast to every appliance.

    For K segments and N appliances, each segment has 2**N joint states. A
    min-plus weighted Hamming transform evaluates all transitions in N
    butterfly passes, giving O(K * N * 2**N) time instead of enumerating
    2**(K*N) complete trajectories or a 2**N by 2**N transition matrix.
    Traceback requires 4 * (K-1) * 2**N bytes, plus O(2**N) working memory
    and O(K*N) output. N is capped at 20 and traceback at 256 MiB; both
    limits are checked before state-space allocation. Floating-point
    arithmetic is used, so 'exact' describes the exhaustive optimization
    algorithm, subject to ordinary floating-point precision.
    """
    started = perf_counter()
    measured = _finite_vector(aggregate, "aggregate")
    nominal = _finite_vector(powers, "powers")
    if np.any(nominal < 0):
        raise ValueError("powers must be nonnegative")
    penalties = np.asarray(switch_penalty, dtype=float)
    if penalties.ndim == 0:
        penalties = np.full(nominal.size, float(penalties))
    if (
        penalties.ndim != 1
        or penalties.size != nominal.size
        or not np.all(np.isfinite(penalties))
        or np.any(penalties < 0)
    ):
        raise ValueError("switch_penalty must be finite and nonnegative, with one value per appliance")
    weights = (
        np.ones(measured.size, dtype=float)
        if segment_weights is None
        else _finite_vector(segment_weights, "segment_weights")
    )
    if weights.size != measured.size or np.any(weights <= 0):
        raise ValueError("segment_weights must be positive and match aggregate")

    n_segments, n_appliances = measured.size, nominal.size
    if n_appliances > MAX_EXACT_APPLIANCES:
        raise ValueError(f"Exact temporal optimization is limited to {MAX_EXACT_APPLIANCES} appliances")
    n_states = 1 << n_appliances
    traceback_bytes = (n_segments - 1) * n_states * np.dtype(np.uint32).itemsize
    if traceback_bytes > MAX_TRACEBACK_BYTES:
        raise ValueError(
            f"Traceback needs {traceback_bytes} bytes; limit is {MAX_TRACEBACK_BYTES} bytes"
        )

    state_ids = np.arange(n_states, dtype=np.uint32)
    predictions = np.zeros(n_states, dtype=float)
    traceback = np.empty((n_segments - 1, n_states), dtype=np.uint32)
    try:
        with np.errstate(over="raise", invalid="raise"):
            for bit, power in enumerate(nominal):
                half = 1 << bit
                predictions[half : 2 * half] = predictions[:half] + power
            values = weights[0] * (measured[0] - predictions) ** 2
            for segment in range(1, n_segments):
                transition_costs, predecessors = _hamming_minimum(
                    values, penalties, state_ids
                )
                values = transition_costs + weights[segment] * (
                    measured[segment] - predictions
                ) ** 2
                traceback[segment - 1] = predecessors

            path = np.empty(n_segments, dtype=np.uint32)
            path[-1] = np.argmin(values)
            for segment in range(n_segments - 1, 0, -1):
                path[segment - 1] = traceback[segment - 1, path[segment]]
            bits = np.arange(n_appliances, dtype=np.uint32)
            states = ((path[:, None] >> bits) & 1).astype(np.int8)
            # Recompute from the recovered trajectory to report the objective
            # with its constant, independent of the dynamic-programming sums.
            energy = float(np.sum(weights * (measured - states @ nominal) ** 2))
            energy += float(np.sum(penalties * np.diff(states, axis=0) ** 2))
            if not np.isfinite(energy):
                raise FloatingPointError("Non-finite recovered objective")
    except FloatingPointError as error:
        raise ValueError("Objective exceeds the finite floating-point range") from error

    return BinaryTemporalExactResult(
        states=states,
        energy=energy,
        states_per_segment=n_states,
        wall_time_s=perf_counter() - started,
        traceback_bytes=traceback_bytes,
    )
