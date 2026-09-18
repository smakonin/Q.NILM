"""QUBO and Ising construction for binary temporal Q.NILM instances."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryTemporalQUBO:
    """Upper-triangular QUBO for an appliance-by-segment state matrix.

    Variables are flattened in segment-major order: ``j = t * n_appliances + i``.
    The objective represented here is

    ``sum_t (aggregate[t] - powers @ x[t])**2``
    ``+ switch_penalty * sum_{t>0,i} (x[t,i] - x[t-1,i])**2``.
    """

    constant: float
    linear: np.ndarray
    quadratic: np.ndarray
    n_appliances: int
    n_segments: int

    @property
    def n_variables(self) -> int:
        return int(self.linear.size)

    def energy(self, bits: np.ndarray) -> float:
        x = np.asarray(bits, dtype=float).reshape(-1)
        if x.size != self.n_variables:
            raise ValueError(f"Expected {self.n_variables} bits, received {x.size}")
        return float(
            self.constant
            + self.linear @ x
            + np.sum(np.triu(self.quadratic, 1) * np.outer(x, x))
        )

    def energies(self) -> tuple[np.ndarray, np.ndarray]:
        """Return all basis-state bits and their objective values."""
        if self.n_variables > 24:
            raise ValueError("Full enumeration is limited to 24 variables")
        states = np.arange(1 << self.n_variables, dtype=np.uint64)
        bit_positions = np.arange(self.n_variables, dtype=np.uint64)
        bits = ((states[:, None] >> bit_positions) & 1).astype(np.int8)
        pair_term = np.einsum(
            "bi,ij,bj->b", bits, np.triu(self.quadratic, 1), bits, optimize=True
        )
        values = self.constant + bits @ self.linear + pair_term
        return bits, values.astype(float)

    def to_ising(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Return ``offset, h, J`` for ``offset + h.Z + sum J_ij Z_i Z_j``."""
        offset = float(self.constant)
        h = -0.5 * self.linear.astype(float)
        offset += 0.5 * float(np.sum(self.linear))
        jmat = np.zeros_like(self.quadratic, dtype=float)

        for j in range(self.n_variables):
            for k in range(j + 1, self.n_variables):
                coefficient = float(self.quadratic[j, k])
                if coefficient == 0.0:
                    continue
                offset += coefficient / 4.0
                h[j] -= coefficient / 4.0
                h[k] -= coefficient / 4.0
                jmat[j, k] = coefficient / 4.0
        return offset, h, jmat

    def decode(self, bits: np.ndarray) -> np.ndarray:
        return np.asarray(bits, dtype=np.int8).reshape(
            self.n_segments, self.n_appliances
        )


def build_binary_temporal_qubo(
    aggregate: np.ndarray,
    appliance_powers: np.ndarray,
    switch_penalty: float | np.ndarray,
    segment_weights: np.ndarray | None = None,
) -> BinaryTemporalQUBO:
    """Construct the binary temporal Q.NILM objective."""
    aggregate = np.asarray(aggregate, dtype=float).reshape(-1)
    powers = np.asarray(appliance_powers, dtype=float).reshape(-1)
    if aggregate.size == 0 or powers.size == 0:
        raise ValueError("Aggregate and appliance powers must be non-empty")
    penalties = np.asarray(switch_penalty, dtype=float)
    if penalties.ndim == 0:
        penalties = np.full(powers.size, float(penalties))
    else:
        penalties = penalties.reshape(-1)
    if penalties.size != powers.size:
        raise ValueError("Expected one switch penalty per appliance")
    if np.any(penalties < 0):
        raise ValueError("switch_penalty must be non-negative")

    n_segments = aggregate.size
    n_appliances = powers.size
    if segment_weights is None:
        weights = np.ones(n_segments, dtype=float)
    else:
        weights = np.asarray(segment_weights, dtype=float).reshape(-1)
    if weights.size != n_segments or np.any(weights <= 0):
        raise ValueError("segment_weights must be positive and match aggregate")
    n_variables = n_segments * n_appliances
    linear = np.zeros(n_variables, dtype=float)
    quadratic = np.zeros((n_variables, n_variables), dtype=float)
    constant = float(np.sum(weights * aggregate**2))

    for segment, measured_power in enumerate(aggregate):
        weight = weights[segment]
        base = segment * n_appliances
        for appliance, nominal_power in enumerate(powers):
            variable = base + appliance
            linear[variable] += weight * (
                nominal_power**2 - 2.0 * measured_power * nominal_power
            )
        for left in range(n_appliances):
            for right in range(left + 1, n_appliances):
                quadratic[base + left, base + right] += (
                    2.0 * weight * powers[left] * powers[right]
                )

    for segment in range(1, n_segments):
        current = segment * n_appliances
        previous = (segment - 1) * n_appliances
        for appliance in range(n_appliances):
            penalty = penalties[appliance]
            current_variable = current + appliance
            previous_variable = previous + appliance
            linear[current_variable] += penalty
            linear[previous_variable] += penalty
            quadratic[previous_variable, current_variable] -= 2.0 * penalty

    return BinaryTemporalQUBO(
        constant=constant,
        linear=linear,
        quadratic=quadratic,
        n_appliances=n_appliances,
        n_segments=n_segments,
    )
