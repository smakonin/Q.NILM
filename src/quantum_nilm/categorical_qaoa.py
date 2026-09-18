"""Feasibility-preserving one-hot QAOA with an explicit XY gate mixer.

The simulator stores only feasible categorical states. This is an exact
representation of the specified ideal quantum circuit, not a classical
optimization oracle: initialization is uniform, angles are supplied, and a
returned sampled solution is selected only from its quantum-distribution
draws. Cost enumeration and simulation are classical resources and must be
included in any end-to-end resource accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np


MAX_FEASIBLE_STATES = 2_000_000
MAX_PHYSICAL_QUBITS = 256


@dataclass(frozen=True)
class CategoricalQAOAProblem:
    """A raw-power objective and its feasible-equivalent one-hot QUBO.

    Register order is time-major, then channel-major. Register 0 is the least
    significant mixed-radix digit of the feasible basis index. Physical qubit
    ``register_offsets[r] + category`` represents that register's category.
    ``quadratic`` contains upper-triangle coefficients, counted exactly once.
    Within-register products are omitted because they vanish in the feasible
    one-hot subspace. ``constant`` is retained for reporting full objectives,
    but never contributes to the coefficient-only gamma ``scale``.
    """

    aggregate: np.ndarray
    levels: tuple[np.ndarray, ...]
    switch_penalty: np.ndarray
    segment_weights: np.ndarray
    previous_states: np.ndarray | None
    state_counts: tuple[int, ...]
    register_sizes: tuple[int, ...]
    register_offsets: tuple[int, ...]
    num_qubits: int
    num_feasible_states: int
    states: np.ndarray
    energies: np.ndarray
    linear: np.ndarray
    quadratic: dict[tuple[int, int], float]
    constant: float
    scale: float


@dataclass(frozen=True)
class CategoricalQAOASample:
    """Measured feasible indices and the best member of those draws only."""

    indices: np.ndarray
    counts: np.ndarray
    best_index: int
    best_states: np.ndarray
    best_energy: float
    mean_sample_energy: float
    shots: int
    feasible_fraction: float


def _finite_vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
    return array


def _counts(value) -> tuple[int, ...]:
    try:
        counts = tuple(value)
    except TypeError as error:
        raise ValueError("state_counts must be positive integers") from error
    if not counts or any(
        isinstance(count, (bool, np.bool_))
        or not isinstance(count, (int, np.integer))
        or count < 1
        for count in counts
    ):
        raise ValueError("state_counts must be positive integers")
    return tuple(int(count) for count in counts)


def _state_array(value, counts, intervals=None):
    states = np.asarray(value)
    if states.ndim != 2 or states.shape[1] != len(counts) or not states.shape[0]:
        raise ValueError("states must be a non-empty interval-by-channel array")
    if intervals is not None and states.shape[0] != intervals:
        raise ValueError("states must match the number of intervals")
    if states.dtype.kind not in "iu" or states.dtype.kind == "b":
        raise ValueError("state categories must be integers")
    for channel, count in enumerate(counts):
        if np.any(states[:, channel] < 0) or np.any(states[:, channel] >= count):
            raise ValueError("state category is outside its register")
    return states.astype(np.int32, copy=True)


def encode_onehot(states, state_counts) -> str:
    """Encode category indices as a Qiskit-order bitstring (highest q first)."""
    counts = _counts(state_counts)
    categories = _state_array(states, counts)
    physical = []
    for row in categories:
        for category, count in zip(row, counts):
            physical.extend(int(index == category) for index in range(count))
    return "".join(str(bit) for bit in reversed(physical))


def decode_onehot(bitstring: str, state_counts, n_intervals: int) -> np.ndarray:
    """Decode a Qiskit-order bitstring, rejecting every infeasible register.

    Internal spaces between classical register groups are accepted. The bit
    count must still match exactly; zero-hot and multi-hot outcomes raise
    ``ValueError`` rather than being repaired or replaced by a classical state.
    """
    counts = _counts(state_counts)
    if isinstance(n_intervals, bool) or not isinstance(n_intervals, (int, np.integer)) or n_intervals < 1:
        raise ValueError("n_intervals must be a positive integer")
    if not isinstance(bitstring, str):
        raise ValueError("bitstring must be text")
    bits = bitstring.replace(" ", "")
    if len(bits) != sum(counts) * n_intervals or any(bit not in "01" for bit in bits):
        raise ValueError("bitstring must contain exactly the expected binary digits")
    physical = bits[::-1]
    states = np.empty((n_intervals, len(counts)), dtype=np.int32)
    offset = 0
    for interval in range(n_intervals):
        for channel, count in enumerate(counts):
            register = physical[offset:offset + count]
            if register.count("1") != 1:
                raise ValueError(f"Infeasible one-hot register at interval {interval}, channel {channel}")
            states[interval, channel] = register.index("1")
            offset += count
    return states


def prepare_categorical_problem(
    aggregate,
    levels,
    switch_penalty,
    segment_weights=None,
    previous_states=None,
) -> CategoricalQAOAProblem:
    """Prepare a one- or two-interval categorical NILM objective.

    ``C(s) = sum_t w[t]*(aggregate[t]-sum_i levels[i][s[t,i]])**2``
    plus ``sum_{t>0,i} penalty[i]*(s[t,i] != s[t-1,i])``. If supplied, the
    preceding inferred state adds ``sum_i penalty[i]*(s[0,i] != previous[i])``.
    The previous state is a known boundary input, not an optimizer seed; the
    caller must not provide a held-out reference label as this input.

    Signed background levels and aggregate are permitted. No power is clipped.
    Weights must be positive; penalties finite and nonnegative. Gamma is
    normalized by ``max(1, sum(abs(linear)) + sum(abs(quadratic)))`` using only
    supplied problem coefficients, never an optimal state or objective value.
    The feasible state cap is checked before enumeration.
    """
    measured = _finite_vector(aggregate, "aggregate").copy()
    if measured.size not in (1, 2):
        raise ValueError("Categorical QAOA supports one or two intervals per problem")
    if not isinstance(levels, (list, tuple)) or not levels:
        raise ValueError("levels must be a non-empty list of channel level arrays")
    channel_levels = tuple(_finite_vector(value, f"levels[{i}]").copy() for i, value in enumerate(levels))
    state_counts = tuple(len(value) for value in channel_levels)
    register_sizes = state_counts * measured.size
    n_qubits = sum(register_sizes)
    n_states = prod(register_sizes)
    if n_qubits > MAX_PHYSICAL_QUBITS:
        raise ValueError(f"Problem exceeds {MAX_PHYSICAL_QUBITS} physical qubits")
    if n_states > MAX_FEASIBLE_STATES:
        raise ValueError(f"Problem exceeds {MAX_FEASIBLE_STATES} feasible states")
    penalties = np.asarray(switch_penalty, dtype=float)
    if penalties.ndim == 0:
        penalties = np.full(len(levels), float(penalties))
    if penalties.ndim != 1 or penalties.size != len(levels) or not np.all(np.isfinite(penalties)) or np.any(penalties < 0):
        raise ValueError("switch_penalty must be finite and nonnegative, one value per channel")
    penalties = penalties.copy()
    weights = np.ones(len(measured)) if segment_weights is None else _finite_vector(segment_weights, "segment_weights").copy()
    if weights.size != measured.size or np.any(weights <= 0):
        raise ValueError("segment_weights must be positive and match aggregate")
    previous = None
    if previous_states is not None:
        value = np.asarray(previous_states)
        if value.shape != (len(levels),):
            raise ValueError("previous_states must have one category per channel")
        previous = _state_array(value[None, :], state_counts)[0]

    offsets, offset = [], 0
    for count in register_sizes:
        offsets.append(offset)
        offset += count
    basis = np.arange(n_states, dtype=np.uint64)
    categories = np.empty((n_states, len(measured), len(levels)), dtype=np.int32)
    stride = 1
    for register, count in enumerate(register_sizes):
        interval, channel = divmod(register, len(levels))
        categories[:, interval, channel] = (basis // stride) % count
        stride *= count

    linear = np.zeros(n_qubits, dtype=float)
    quadratic = {}
    constant = 0.0
    try:
        with np.errstate(over="raise", invalid="raise"):
            prediction = np.zeros((n_states, len(measured)), dtype=float)
            for channel, values in enumerate(channel_levels):
                prediction += values[categories[:, :, channel]]
            energies = np.sum(weights * (measured - prediction) ** 2, axis=1)
            if len(measured) > 1:
                energies += np.sum(penalties * (categories[:, 1:] != categories[:, :-1]), axis=(1, 2))
            if previous is not None:
                energies += np.sum(penalties * (categories[:, 0] != previous), axis=1)

            for interval, (reading, weight) in enumerate(zip(measured, weights)):
                constant += float(weight * reading**2)
                for channel, values in enumerate(channel_levels):
                    start = offsets[interval * len(levels) + channel]
                    linear[start:start + len(values)] += weight * (values**2 - 2 * reading * values)
                    # At most one category can be occupied within a channel.
                    # Thus only products between different channel registers
                    # are needed for the equivalent feasible-subspace cost.
                    for other in range(channel + 1, len(levels)):
                        other_start = offsets[interval * len(levels) + other]
                        products = 2 * weight * values[:, None] * channel_levels[other][None, :]
                        for a, b in np.argwhere(products != 0):
                            quadratic[(start + int(a), other_start + int(b))] = float(products[a, b])
            for interval in range(1, len(measured)):
                for channel, penalty in enumerate(penalties):
                    constant += float(penalty)
                    previous_offset = offsets[(interval - 1) * len(levels) + channel]
                    current_offset = offsets[interval * len(levels) + channel]
                    if penalty:
                        for category in range(state_counts[channel]):
                            quadratic[(previous_offset + category, current_offset + category)] = -float(penalty)
            if previous is not None:
                for channel, (category, penalty) in enumerate(zip(previous, penalties)):
                    constant += float(penalty)
                    linear[offsets[channel] + category] -= penalty
            scale = max(1.0, float(np.sum(np.abs(linear))) + sum(abs(value) for value in quadratic.values()))
            if not np.all(np.isfinite(energies)) or not np.isfinite(constant) or not np.isfinite(scale):
                raise FloatingPointError("Non-finite categorical cost")
    except FloatingPointError as error:
        raise ValueError("Categorical objective exceeds the finite floating-point range") from error
    return CategoricalQAOAProblem(
        measured, channel_levels, penalties, weights, previous, state_counts,
        register_sizes, tuple(offsets), n_qubits, n_states, categories,
        energies, linear, quadratic, constant, scale,
    )


def _angles(gammas, betas):
    gamma = np.atleast_1d(np.asarray(gammas, dtype=float))
    beta = np.atleast_1d(np.asarray(betas, dtype=float))
    if gamma.ndim != 1 or beta.ndim != 1 or not gamma.size or gamma.shape != beta.shape or not np.all(np.isfinite(gamma)) or not np.all(np.isfinite(beta)):
        raise ValueError("gammas and betas must be non-empty matching finite scalar/vector angles")
    return gamma, beta


def categorical_qaoa_statevector(problem: CategoricalQAOAProblem, gammas, betas) -> np.ndarray:
    """Simulate the explicit circuit exactly inside its feasible subspace.

    Initialization is the product of uniform positive-amplitude one-hot W
    states. At every layer, cost first applies ``exp(-i*gamma*C/scale)``.
    Each register's adjacent pairs then receive, in ascending category order,
    ``exp(-i*beta*(XX+YY)/2)`` = ``RXX(beta) RYY(beta)``. In a one-hot pair,
    the two amplitudes transform using diagonal cos(beta), off-diagonal
    -i*sin(beta). Pair rotations on overlapping pairs are ordered, not replaced
    by an exponential of the sum; this distinguishes the circuit precisely.
    This unitary mixer and W initialization preserve one-hot feasibility.
    """
    gamma, beta = _angles(gammas, betas)
    state = np.full(problem.num_feasible_states, 1 / np.sqrt(problem.num_feasible_states), dtype=complex)
    normalized_cost = problem.energies / problem.scale
    for cost_angle, mixer_angle in zip(gamma, beta):
        state *= np.exp(-1j * cost_angle * normalized_cost)
        cosine, sine = np.cos(mixer_angle), -1j * np.sin(mixer_angle)
        stride = 1
        for count in problem.register_sizes:
            blocks = state.reshape(-1, count, stride)
            for category in range(count - 1):
                left = blocks[:, category, :].copy()
                right = blocks[:, category + 1, :].copy()
                blocks[:, category, :] = cosine * left + sine * right
                blocks[:, category + 1, :] = sine * left + cosine * right
            stride *= count
    return state


def categorical_qaoa_probabilities(problem: CategoricalQAOAProblem, gammas, betas) -> np.ndarray:
    """Return normalized probabilities of the specified ideal circuit."""
    state = categorical_qaoa_statevector(problem, gammas, betas)
    probabilities = np.abs(state) ** 2
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0 or abs(total - 1.0) > 1e-9:
        raise ValueError("Categorical QAOA state lost normalization")
    # Remove roundoff in the normalization only; no outcomes are postselected.
    return probabilities / total


def choose_sampled(problem: CategoricalQAOAProblem, indices) -> CategoricalQAOASample:
    """Select only among supplied feasible samples, for matched sampler controls.

    This common selection rule can be applied to QAOA, uniform or other
    feasible-index samples. It never consults an unsampled optimum. Invalid
    indices are rejected, not repaired. Equal sampled costs prefer the lower
    feasible basis index. To handle physical shots, decode and retain their
    infeasibility rate separately rather than silently dropping outcomes here.
    """
    indices = np.asarray(indices)
    if indices.ndim != 1 or not indices.size or indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= problem.num_feasible_states):
        raise ValueError("indices must be a non-empty vector of valid feasible basis integers")
    indices = indices.astype(np.int64, copy=True)
    counts = np.bincount(indices, minlength=problem.num_feasible_states)
    observed = np.flatnonzero(counts)
    best_index = int(observed[np.argmin(problem.energies[observed])])
    return CategoricalQAOASample(
        indices=indices,
        counts=counts,
        best_index=best_index,
        best_states=problem.states[best_index].copy(),
        best_energy=float(problem.energies[best_index]),
        mean_sample_energy=float(np.mean(problem.energies[indices])),
        shots=int(indices.size),
        feasible_fraction=1.0,
    )


def sample_categorical_qaoa(problem: CategoricalQAOAProblem, gammas, betas, shots: int, seed=None) -> CategoricalQAOASample:
    """Draw shots, returning the lowest-cost actually sampled trajectory.

    No exact/DP solution is consulted and no absent optimum is inserted.
    Equal sampled costs prefer the lower feasible basis index. These are
    classical samples of an ideal quantum distribution, not QPU observations.
    """
    if isinstance(shots, (bool, np.bool_)) or not isinstance(shots, (int, np.integer)) or shots < 1:
        raise ValueError("shots must be a positive integer")
    probabilities = categorical_qaoa_probabilities(problem, gammas, betas)
    indices = np.random.default_rng(seed).choice(problem.num_feasible_states, size=int(shots), p=probabilities)
    return choose_sampled(problem, indices)
