"""Train-only power-level fitting and exact categorical temporal inference."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np


MAX_JOINT_STATES = 4096
MAX_TRACEBACK_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class MultistateTemporalExactResult:
    """Recovered state indices, channel powers, and full objective value."""

    states: np.ndarray
    predicted_power: np.ndarray
    energy: float
    states_per_segment: int
    wall_time_s: float
    traceback_bytes: int


def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty, finite, one-dimensional array")
    return array


def fit_power_levels(values: np.ndarray, n_states: int) -> np.ndarray:
    """Fit deterministic one-dimensional k-means on supplied TRAIN data only.

    The caller must pass only its training partition: this function has no
    validation/test input and never reads external data. Initial centres are
    evenly spaced empirical quantiles including the minimum and maximum.
    Duplicate quantiles are filled using the most distant distinct observed
    value (ties prefer the lower value). Lloyd updates and empty-cluster
    reseeding are deterministic. Returned centroids are sorted and distinct;
    constant or low-diversity data can return fewer than ``n_states`` levels.
    No physical OFF threshold, nonnegativity, or zero state is imposed.

    Scaling to [-1, 1] avoids overflowing distances and centroid sums for
    otherwise finite inputs. A failure to converge in 300 iterations is
    reported rather than silently returning an unconverged fit.
    """
    samples = _finite_vector(values, "values")
    if (
        isinstance(n_states, (bool, np.bool_))
        or not isinstance(n_states, (int, np.integer))
        or n_states < 1
    ):
        raise ValueError("n_states must be a positive integer")
    if samples.size < n_states:
        raise ValueError("values must contain at least n_states training observations")
    scale = max(1.0, float(np.max(np.abs(samples))))
    normalized = samples / scale
    distinct = np.unique(normalized)
    count = min(int(n_states), distinct.size)
    if count == 1:
        return np.array([float(np.mean(normalized)) * scale])

    centres = np.unique(np.quantile(normalized, np.linspace(0.0, 1.0, count)))
    while centres.size < count:
        distances = np.min(np.abs(distinct[:, None] - centres[None, :]), axis=1)
        centres = np.sort(np.append(centres, distinct[np.argmax(distances)]))

    for _ in range(300):
        labels = np.argmin(np.abs(normalized[:, None] - centres[None, :]), axis=1)
        sizes = np.bincount(labels, minlength=count)
        sums = np.bincount(labels, weights=normalized, minlength=count)
        occupied = sizes > 0
        updated = sums[occupied] / sizes[occupied]
        while updated.size < count:
            distances = np.min(np.abs(distinct[:, None] - updated[None, :]), axis=1)
            updated = np.append(updated, distinct[np.argmax(distances)])
        updated.sort()
        if np.array_equal(updated, centres):
            return np.unique(updated * scale)
        centres = updated
    raise ValueError("Power-level k-means did not converge within 300 iterations")


def _categorical_minimum(
    values: np.ndarray,
    penalties: np.ndarray,
    counts: list[int],
    state_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Separable min-plus transform for a per-channel change indicator.

    A channel pass compares each current category with the best category in
    that slice plus its nonnegative change penalty. Permitting the best
    category to be the current one is harmless: that candidate cannot beat
    the unchanged cost. Original predecessor IDs are carried through every
    pass, so equal costs deterministically prefer the smallest predecessor.
    This is O(N*S), where S is the joint-state count, without an S-by-S matrix.
    """
    costs = values.copy()
    predecessors = state_ids.copy()
    stride = 1
    for count, penalty in zip(counts, penalties):
        cost_blocks = costs.reshape(-1, count, stride)
        predecessor_blocks = predecessors.reshape(-1, count, stride)
        best_cost = np.min(cost_blocks, axis=1, keepdims=True)
        best_predecessor = np.min(
            np.where(cost_blocks == best_cost, predecessor_blocks, np.iinfo(np.uint32).max),
            axis=1,
            keepdims=True,
        )
        candidate = best_cost + penalty
        replace = (candidate < cost_blocks) | (
            (candidate == cost_blocks) & (best_predecessor < predecessor_blocks)
        )
        cost_blocks[:] = np.where(replace, candidate, cost_blocks)
        predecessor_blocks[:] = np.where(replace, best_predecessor, predecessor_blocks)
        stride *= count
    return costs, predecessors


def solve_multistate_temporal_exact(
    aggregate: np.ndarray,
    levels: list[np.ndarray],
    switch_penalty: float | np.ndarray,
    segment_weights: np.ndarray | None = None,
) -> MultistateTemporalExactResult:
    """Minimize weighted reconstruction error plus categorical switching.

    The full objective is ``sum_t w[t]*(aggregate[t]-sum_i levels[i][s[t,i]])**2``
    plus ``sum_{t>0,i} penalty[i]*(s[t,i] != s[t-1,i])``. A change from category
    0 to 2 has the same transition charge as a change from 0 to 1; ordinal
    index distance is never used. Levels and observations may be signed.
    Channel arrays are nonempty and finite, with their supplied order kept.
    Duplicate power levels are permitted; fitted levels are normally distinct.
    Penalties must be finite and nonnegative, and weights finite and positive.

    For K segments, N channels and S joint categorical states, separable
    dynamic programming takes O(K*N*S) time and 4*(K-1)*S traceback bytes,
    plus O(N*S) state and O(K*N) output storage. S is capped at 4096 and
    traceback at 256 MiB before joint-state allocation. Appliance 0 is the
    least-significant mixed-radix digit; cost ties select the lowest joint
    state ID, consistent with the binary solver. Exact refers to the
    exhaustive optimization algorithm subject to floating-point precision.
    """
    started = perf_counter()
    measured = _finite_vector(aggregate, "aggregate")
    if not isinstance(levels, (list, tuple)) or len(levels) == 0:
        raise ValueError("levels must be a non-empty list of channel level arrays")
    channel_levels = [_finite_vector(value, f"levels[{i}]") for i, value in enumerate(levels)]
    counts = [value.size for value in channel_levels]
    n_states = 1
    for count in counts:
        n_states *= count
        if n_states > MAX_JOINT_STATES:
            raise ValueError(f"Exact multistate optimization is limited to {MAX_JOINT_STATES} joint states")
    penalties = np.asarray(switch_penalty, dtype=float)
    if penalties.ndim == 0:
        penalties = np.full(len(levels), float(penalties))
    if (
        penalties.ndim != 1
        or penalties.size != len(levels)
        or not np.all(np.isfinite(penalties))
        or np.any(penalties < 0)
    ):
        raise ValueError("switch_penalty must be finite and nonnegative, with one value per channel")
    weights = (
        np.ones(measured.size, dtype=float)
        if segment_weights is None
        else _finite_vector(segment_weights, "segment_weights")
    )
    if weights.size != measured.size or np.any(weights <= 0):
        raise ValueError("segment_weights must be positive and match aggregate")
    traceback_bytes = (measured.size - 1) * n_states * np.dtype(np.uint32).itemsize
    if traceback_bytes > MAX_TRACEBACK_BYTES:
        raise ValueError(f"Traceback needs {traceback_bytes} bytes; limit is {MAX_TRACEBACK_BYTES} bytes")

    state_ids = np.arange(n_states, dtype=np.uint32)
    joint_states = np.empty((n_states, len(levels)), dtype=np.int32)
    stride = 1
    for channel, count in enumerate(counts):
        joint_states[:, channel] = (state_ids // stride) % count
        stride *= count
    traceback = np.empty((measured.size - 1, n_states), dtype=np.uint32)
    try:
        with np.errstate(over="raise", invalid="raise"):
            predictions = np.zeros(n_states, dtype=float)
            for channel, values in enumerate(channel_levels):
                predictions += values[joint_states[:, channel]]
            costs = weights[0] * (measured[0] - predictions) ** 2
            for segment in range(1, measured.size):
                transition_costs, predecessors = _categorical_minimum(
                    costs, penalties, counts, state_ids
                )
                costs = transition_costs + weights[segment] * (measured[segment] - predictions) ** 2
                traceback[segment - 1] = predecessors
            path = np.empty(measured.size, dtype=np.uint32)
            path[-1] = np.argmin(costs)
            for segment in range(measured.size - 1, 0, -1):
                path[segment - 1] = traceback[segment - 1, path[segment]]
            states = joint_states[path]
            predicted_power = np.column_stack(
                [values[states[:, channel]] for channel, values in enumerate(channel_levels)]
            )
            energy = float(np.sum(weights * (measured - predicted_power.sum(axis=1)) ** 2))
            energy += float(np.sum(penalties * (states[1:] != states[:-1])))
            if not np.isfinite(energy):
                raise FloatingPointError("Non-finite recovered objective")
    except FloatingPointError as error:
        raise ValueError("Objective exceeds the finite floating-point range") from error

    return MultistateTemporalExactResult(
        states=states,
        predicted_power=predicted_power,
        energy=energy,
        states_per_segment=n_states,
        wall_time_s=perf_counter() - started,
        traceback_bytes=traceback_bytes,
    )
