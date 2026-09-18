"""A supervised, train-only factorial HMM reference for 30-second NILM."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np


MAX_JOINT_STATES = 4096
MAX_TRACEBACK_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class FHMMModel:
    """Independent appliance chains and a shared Gaussian sum observation.

    Chain order is dryer, refrigerator, vacuum, signed background. Matrices
    index [previous_state, next_state]; stored costs are negative log
    probabilities. Fitting uses one Laplace pseudo-count in every initial
    category and transition cell. ``noise_variance`` is the train residual
    mean square for the zero-mean Gaussian emission, floored at 1 W squared.
    """

    levels: list[np.ndarray]
    transition_costs: list[np.ndarray]
    initial_costs: list[np.ndarray]
    noise_variance: float
    training_blocks: int = 0
    training_runs: int = 0


def _vector(value, name):
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def _valid_runs(timestamps, size):
    times = np.asarray(timestamps, dtype=float)
    if (
        times.ndim != 1
        or times.size != size
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("timestamps must be finite, strictly increasing and match the block count")
    if not size:
        return []
    edges = np.r_[0, np.flatnonzero(np.diff(times) != 30.0) + 1, size]
    return [slice(int(start), int(stop)) for start, stop in zip(edges[:-1], edges[1:])]


def _level_arrays(levels):
    if not isinstance(levels, (tuple, list)) or len(levels) != 4:
        raise ValueError("levels must contain three appliance arrays and one background array")
    arrays = [_vector(values, f"levels[{i}]").copy() for i, values in enumerate(levels)]
    size = 1
    for array in arrays:
        size *= array.size
        if size > MAX_JOINT_STATES:
            raise ValueError(f"FHMM inference is limited to {MAX_JOINT_STATES} joint states")
    return arrays


def fit_fhmm(training_windows: list[dict], levels: list[np.ndarray]) -> FHMMModel:
    """Learn a conventional factorial HMM using TRAIN windows only.

    Every supplied window has ``timestamps`` (one per 30-second block) and
    ``values`` with columns [mains, dryer, refrigerator, vacuum]. The supplied
    four level arrays must already have been fitted exclusively on training
    observations. Train-only supervised nearest-level labels estimate each
    chain. Background observations are mains minus the three metered loads;
    negative net background is retained. Initial counts restart at each
    window and missing-block gap; transitions never cross those boundaries.
    This function accepts no validation or test data and performs no tuning.
    """
    channel_levels = _level_arrays(levels)
    if not isinstance(training_windows, (list, tuple)) or not training_windows:
        raise ValueError("training_windows must be a non-empty list")
    transitions = [np.ones((len(values), len(values)), dtype=float) for values in channel_levels]
    initials = [np.ones(len(values), dtype=float) for values in channel_levels]
    residual_square_sum, blocks, runs = 0.0, 0, 0
    try:
        with np.errstate(over="raise", invalid="raise"):
            for window in training_windows:
                if not isinstance(window, dict) or not {"timestamps", "values"} <= window.keys():
                    raise ValueError("each training window requires timestamps and values")
                observed = np.asarray(window["values"], dtype=float)
                if observed.ndim != 2 or observed.shape[1] != 4 or not len(observed) or not np.all(np.isfinite(observed)):
                    raise ValueError("training values must be a non-empty finite block-by-four array")
                parts = _valid_runs(window["timestamps"], len(observed))
                background = observed[:, 0] - np.sum(observed[:, 1:], axis=1)
                channel_power = np.column_stack((observed[:, 1:], background))
                labels = [np.argmin(np.abs(channel_power[:, i, None] - values[None, :]), axis=1) for i, values in enumerate(channel_levels)]
                reconstructed = sum(values[state] for values, state in zip(channel_levels, labels))
                residual_square_sum += float(np.sum((observed[:, 0] - reconstructed) ** 2))
                for part in parts:
                    for initial, transition, label in zip(initials, transitions, labels):
                        initial[label[part.start]] += 1.0
                        indices = label[part]
                        np.add.at(transition, (indices[:-1], indices[1:]), 1.0)
                blocks += len(observed)
                runs += len(parts)
            variance = max(1.0, residual_square_sum / blocks)
            transition_costs = [-np.log(counts / counts.sum(axis=1, keepdims=True)) for counts in transitions]
            initial_costs = [-np.log(counts / counts.sum()) for counts in initials]
            if not np.isfinite(variance):
                raise FloatingPointError("Non-finite emission variance")
    except FloatingPointError as error:
        raise ValueError("FHMM training exceeds the finite floating-point range") from error
    return FHMMModel(channel_levels, transition_costs, initial_costs, variance, blocks, runs)


def _transition_minimum(values, transition_costs, state_ids):
    """Factored min-plus product with arbitrary per-chain transition costs.

    The original predecessor ID travels through each chain pass. This
    evaluates every joint transition exactly in O(S * sum_i states_i) time,
    without materializing the S-by-S joint transition matrix. Equal costs
    choose the lowest predecessor ID, independent of array traversal order.
    """
    costs, predecessors = values.copy(), state_ids.copy()
    stride = 1
    for transition in transition_costs:
        count = len(transition)
        old_cost = costs.reshape(-1, count, stride)
        old_predecessor = predecessors.reshape(-1, count, stride)
        new_cost = np.empty_like(old_cost)
        new_predecessor = np.empty_like(old_predecessor)
        for destination in range(count):
            candidates = old_cost + transition[None, :, destination, None]
            minima = candidates.min(axis=1, keepdims=True)
            origins = np.min(
                np.where(candidates == minima, old_predecessor, np.iinfo(np.uint32).max),
                axis=1,
            )
            new_cost[:, destination, :] = minima[:, 0, :]
            new_predecessor[:, destination, :] = origins
        costs, predecessors = new_cost.ravel(), new_predecessor.ravel()
        stride *= count
    return costs, predecessors


def predict_fhmm(aggregate, timestamps, model: FHMMModel, include_background=True):
    """Exact MAP inference from aggregate and timestamps, with no test labels.

    Input readings are already 30-second block averages; each valid block is
    one HMM step. A timestamp gap restarts the learned initial distribution.
    With ``include_background=False`` the fourth chain is excluded from both
    inference and output, providing an explicitly misspecified ablation.

    Returns per-block channel power and a timing/objective summary. The
    objective includes initial and transition negative log probabilities
    plus squared sum residual divided by twice the learned variance. The
    state-independent Gaussian normalizing constant is deliberately omitted.
    ``exact`` describes discrete MAP optimization, not a guarantee that the
    fitted generative model describes a real household correctly.
    """
    started = perf_counter()
    measured = np.asarray(aggregate, dtype=float)
    if measured.ndim != 1 or not np.all(np.isfinite(measured)):
        raise ValueError("aggregate must be a finite vector")
    parts = _valid_runs(timestamps, measured.size)
    if not isinstance(model, FHMMModel):
        raise ValueError("model must be an FHMMModel")
    channel_levels = _level_arrays(model.levels)
    if len(model.transition_costs) != 4 or len(model.initial_costs) != 4:
        raise ValueError("model requires one transition matrix and initial cost vector per chain")
    for values, transition, initial in zip(channel_levels, model.transition_costs, model.initial_costs):
        if (
            np.shape(transition) != (len(values), len(values))
            or np.shape(initial) != (len(values),)
            or not np.all(np.isfinite(transition))
            or not np.all(np.isfinite(initial))
            or np.any(np.asarray(transition) < 0)
            or np.any(np.asarray(initial) < 0)
        ):
            raise ValueError("model probability costs must be finite, nonnegative and match its levels")
    if not np.isfinite(model.noise_variance) or model.noise_variance <= 0:
        raise ValueError("model noise_variance must be finite and positive")
    channels = 4 if include_background else 3
    channel_levels = channel_levels[:channels]
    transitions = [np.asarray(matrix, dtype=float) for matrix in model.transition_costs[:channels]]
    initials = [np.asarray(vector, dtype=float) for vector in model.initial_costs[:channels]]
    counts = [len(values) for values in channel_levels]
    n_states = int(np.prod(counts))
    max_run = max((part.stop - part.start for part in parts), default=0)
    traceback_bytes = max(0, max_run - 1) * n_states * np.dtype(np.uint32).itemsize
    if traceback_bytes > MAX_TRACEBACK_BYTES:
        raise ValueError(f"Traceback needs {traceback_bytes} bytes; limit is {MAX_TRACEBACK_BYTES} bytes")
    state_ids = np.arange(n_states, dtype=np.uint32)
    joint_states = np.empty((n_states, channels), dtype=np.int32)
    stride = 1
    for channel, count in enumerate(counts):
        joint_states[:, channel] = (state_ids // stride) % count
        stride *= count
    prediction = np.empty((measured.size, channels), dtype=float)
    objective = 0.0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            sum_power = sum(values[joint_states[:, i]] for i, values in enumerate(channel_levels))
            initial_cost = sum(costs[joint_states[:, i]] for i, costs in enumerate(initials))
            for part in parts:
                signal = measured[part]
                values = initial_cost + (signal[0] - sum_power) ** 2 / model.noise_variance / 2.0
                trace = np.empty((len(signal) - 1, n_states), dtype=np.uint32)
                for block in range(1, len(signal)):
                    minimum, predecessor = _transition_minimum(values, transitions, state_ids)
                    values = minimum + (signal[block] - sum_power) ** 2 / model.noise_variance / 2.0
                    trace[block - 1] = predecessor
                path = np.empty(len(signal), dtype=np.uint32)
                path[-1] = np.argmin(values)
                for block in range(len(signal) - 1, 0, -1):
                    path[block - 1] = trace[block - 1, path[block]]
                recovered = joint_states[path]
                prediction[part] = np.column_stack([levels[recovered[:, i]] for i, levels in enumerate(channel_levels)])
                objective += float(values[path[-1]])
            if not np.isfinite(objective):
                raise FloatingPointError("Non-finite recovered objective")
    except FloatingPointError as error:
        raise ValueError("FHMM inference exceeds the finite floating-point range") from error
    return prediction, {
        "solver_wall_time_s": perf_counter() - started,
        "objective_energy": objective,
        "objective_definition": "negative log initial/transition probabilities plus squared residual/(2*training variance), without Gaussian constant",
        "segments": measured.size,
        "blocks": measured.size,
        "valid_runs": len(parts),
        "joint_states_per_segment": n_states,
        "max_traceback_bytes": traceback_bytes,
        "include_background": bool(include_background),
        "noise_variance_w2": float(model.noise_variance),
    }
