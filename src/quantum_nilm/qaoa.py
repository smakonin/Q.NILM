"""Small exact state-vector implementation of a Q.NILM QAOA circuit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qubo import BinaryTemporalQUBO


@dataclass(frozen=True)
class QAOAResult:
    gamma: float
    beta: float
    expected_cost: float
    probabilities: np.ndarray
    phase_scale: float


def _apply_rx_mixer(state: np.ndarray, beta: float, n_qubits: int) -> np.ndarray:
    mixed = state.copy()
    cosine = np.cos(beta)
    sine = -1j * np.sin(beta)
    for qubit in range(n_qubits):
        stride = 1 << qubit
        block = stride << 1
        for start in range(0, mixed.size, block):
            low = mixed[start : start + stride].copy()
            high = mixed[start + stride : start + block].copy()
            mixed[start : start + stride] = cosine * low + sine * high
            mixed[start + stride : start + block] = sine * low + cosine * high
    return mixed


def phase_scale_for(qubo: BinaryTemporalQUBO) -> float:
    """Coefficient-only phase scale that does not use the unknown optimum."""
    coefficient_bound = (
        float(np.sum(np.abs(qubo.linear)))
        + float(np.sum(np.abs(np.triu(qubo.quadratic, 1))))
    )
    return max(coefficient_bound, 1.0)


def qaoa_probabilities(
    qubo: BinaryTemporalQUBO,
    gammas: list[float] | np.ndarray,
    betas: list[float] | np.ndarray,
    phase_scale: float | None = None,
) -> np.ndarray:
    """Evaluate a standard QAOA state using exact state-vector simulation."""
    gammas = np.asarray(gammas, dtype=float).reshape(-1)
    betas = np.asarray(betas, dtype=float).reshape(-1)
    if gammas.size != betas.size or gammas.size == 0:
        raise ValueError("gammas and betas must have the same positive length")

    _, costs = qubo.energies()
    scale = phase_scale_for(qubo) if phase_scale is None else float(phase_scale)
    state = np.ones(costs.size, dtype=complex) / np.sqrt(costs.size)
    for gamma, beta in zip(gammas, betas):
        state *= np.exp(-1j * gamma * costs / scale)
        state = _apply_rx_mixer(state, beta, qubo.n_variables)
    probabilities = np.abs(state) ** 2
    return probabilities / probabilities.sum()


def optimize_qaoa_p1(
    qubo: BinaryTemporalQUBO,
    gamma_points: int = 41,
    beta_points: int = 31,
) -> QAOAResult:
    """Deterministic grid optimization for a depth-one validation circuit."""
    if gamma_points < 2 or beta_points < 2:
        raise ValueError("Each parameter grid needs at least two points")
    _, costs = qubo.energies()
    scale = phase_scale_for(qubo)
    best: QAOAResult | None = None
    for gamma in np.linspace(0.0, 2.0 * np.pi, gamma_points, endpoint=False):
        for beta in np.linspace(0.0, np.pi, beta_points, endpoint=False):
            probabilities = qaoa_probabilities(qubo, [gamma], [beta], scale)
            expected_cost = float(probabilities @ costs)
            if best is None or expected_cost < best.expected_cost:
                best = QAOAResult(
                    gamma=float(gamma),
                    beta=float(beta),
                    expected_cost=expected_cost,
                    probabilities=probabilities,
                    phase_scale=scale,
                )
    assert best is not None
    return best


def sample_best(
    qubo: BinaryTemporalQUBO,
    probabilities: np.ndarray,
    shots: int,
    seed: int,
) -> tuple[np.ndarray, float, int]:
    rng = np.random.default_rng(seed)
    sampled_states = rng.choice(probabilities.size, size=shots, p=probabilities)
    unique_states = np.unique(sampled_states)
    bits, costs = qubo.energies()
    best_state = int(unique_states[np.argmin(costs[unique_states])])
    count = int(np.sum(sampled_states == best_state))
    return bits[best_state], float(costs[best_state]), count


def export_openqasm3_p1(
    qubo: BinaryTemporalQUBO,
    gamma: float,
    beta: float,
    phase_scale: float,
) -> str:
    """Export the optimized p=1 cost and mixer layers as OpenQASM 3."""
    _, h, jmat = qubo.to_ising()
    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"bit[{qubo.n_variables}] c;",
        f"qubit[{qubo.n_variables}] q;",
    ]
    for qubit in range(qubo.n_variables):
        lines.append(f"h q[{qubit}];")
    for qubit, coefficient in enumerate(h):
        angle = 2.0 * gamma * coefficient / phase_scale
        if abs(angle) > 1e-15:
            lines.append(f"rz({angle:.16g}) q[{qubit}];")
    for left in range(qubo.n_variables):
        for right in range(left + 1, qubo.n_variables):
            coefficient = float(jmat[left, right])
            if coefficient == 0.0:
                continue
            angle = 2.0 * gamma * coefficient / phase_scale
            lines.extend(
                [
                    f"cx q[{left}], q[{right}];",
                    f"rz({angle:.16g}) q[{right}];",
                    f"cx q[{left}], q[{right}];",
                ]
            )
    mixer_angle = 2.0 * beta
    for qubit in range(qubo.n_variables):
        lines.append(f"rx({mixer_angle:.16g}) q[{qubit}];")
    for qubit in range(qubo.n_variables):
        lines.append(f"c[{qubit}] = measure q[{qubit}];")
    return "\n".join(lines) + "\n"
