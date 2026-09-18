"""Optional Qiskit circuit construction and auditable IBM sampling metrics.

Qiskit strings display the highest-index classical bit first.  Q.NILM stores
variable zero first, so a measurement string is reversed when it is decoded.
This module does not submit jobs or load IBM credentials.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np

from .qaoa import phase_scale_for
from .qubo import BinaryTemporalQUBO

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


EXACT_ENUMERATION_LIMIT = 12


def _validate_qubo(qubo: BinaryTemporalQUBO) -> None:
    n = qubo.n_variables
    if (
        n < 1
        or qubo.n_appliances < 1
        or qubo.n_segments < 1
        or n != qubo.n_appliances * qubo.n_segments
        or np.shape(qubo.linear) != (n,)
        or np.shape(qubo.quadratic) != (n, n)
    ):
        raise ValueError("QUBO dimensions must match a nonempty appliance-state matrix")
    if not (
        np.isfinite(qubo.constant)
        and np.all(np.isfinite(qubo.linear))
        and np.all(np.isfinite(qubo.quadratic))
    ):
        raise ValueError("QUBO coefficients must be finite")


def build_qaoa_circuit(
    qubo: BinaryTemporalQUBO,
    gammas: Sequence[float] | np.ndarray,
    betas: Sequence[float] | np.ndarray,
    phase_scale: float | None = None,
    measure: bool = True,
) -> QuantumCircuit:
    """Construct depth-p QAOA for the existing Q.NILM Ising convention.

The Ising offset is a global phase and is omitted.  Each cost layer implements
``exp(-i gamma H_cost / phase_scale)`` and each mixer implements
``exp(-i beta sum X)``.  Classical bit j measures logical variable j; the single
classical register is named ``meas`` for SamplerV2's ``data.meas`` interface.
Qiskit is imported only when this function is used.
    """
    _validate_qubo(qubo)
    gamma_values = np.asarray(gammas, dtype=float).reshape(-1)
    beta_values = np.asarray(betas, dtype=float).reshape(-1)
    if gamma_values.size == 0 or gamma_values.size != beta_values.size:
        raise ValueError("gammas and betas must have the same positive length")
    if not (np.all(np.isfinite(gamma_values)) and np.all(np.isfinite(beta_values))):
        raise ValueError("gammas and betas must be finite")
    scale = phase_scale_for(qubo) if phase_scale is None else float(phase_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("phase_scale must be finite and positive")
    if not isinstance(measure, bool):
        raise ValueError("measure must be a boolean")

    try:
        from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    except ImportError as exc:
        raise ImportError("Install Q.NILM's [ibm] extra to construct Qiskit circuits") from exc

    _, fields, couplings = qubo.to_ising()
    register = QuantumRegister(qubo.n_variables, "q")
    circuit = QuantumCircuit(register, name="Q.NILM-QAOA")
    if measure:
        circuit.add_register(ClassicalRegister(qubo.n_variables, "meas"))
    circuit.h(register)
    for gamma, beta in zip(gamma_values, beta_values):
        for index, field in enumerate(fields):
            if field != 0.0:
                circuit.rz(float(2.0 * gamma * field / scale), index)
        for left, right in zip(*np.nonzero(np.triu(couplings, 1))):
            circuit.rzz(
                float(2.0 * gamma * couplings[left, right] / scale),
                int(left),
                int(right),
            )
        for index in range(qubo.n_variables):
            circuit.rx(float(2.0 * beta), index)
    if measure:
        circuit.measure(register, circuit.cregs[0])
    circuit.metadata = {
        "algorithm": "Q.NILM",
        "qaoa_depth": int(gamma_values.size),
        "gammas": gamma_values.tolist(),
        "betas": beta_values.tolist(),
        "phase_scale": scale,
        "variable_order": "segment-major; variable j measured in meas[j]",
    }
    return circuit


def _reference_metrics(truth: np.ndarray, estimate: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum((truth == 1) & (estimate == 1)))
    tn = int(np.sum((truth == 0) & (estimate == 0)))
    fp = int(np.sum((truth == 0) & (estimate == 1)))
    fn = int(np.sum((truth == 1) & (estimate == 0)))
    denominator = float(np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
    return {
        "bit_accuracy": float(np.mean(truth == estimate)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "mcc": float((tp * tn - fp * fn) / denominator) if denominator else None,
        "confusion_counts": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "zero_division_policy": "precision/recall/F1: 0; undefined MCC: null",
        "weighting": "one vote per appliance-segment bit",
    }


def counts_to_metrics(
    qubo: BinaryTemporalQUBO,
    counts: Mapping[str, int],
    reference_states: np.ndarray | None = None,
    aggregate: np.ndarray | None = None,
    powers: np.ndarray | None = None,
    segment_weights: np.ndarray | None = None,
    *,
    exact_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Decode full-width Qiskit counts and score the lowest-cost observed state.

Counts must be positive integer shot counts keyed by strings of exactly n binary
digits, with no register separators.  Ties in sampled objective use the smallest
integer basis state.  Reference metrics describe that selected state, not an
average over shots, and are unweighted appliance-segment metrics.  Aggregate MAE
is weighted by segment durations when supplied.

Exact enumeration is restricted to at most 12 variables.  Every optimum within
the absolute ``exact_tolerance`` contributes to optimum probability (rtol=0).
Agreement with the smallest-index exact minimizer is reported separately from
agreement with supplied reference states; degenerate optima can disagree bitwise.
Above 12 variables, exact fields are null.  The return value is JSON serializable.
    """
    _validate_qubo(qubo)
    if not np.isfinite(exact_tolerance) or exact_tolerance < 0:
        raise ValueError("exact_tolerance must be finite and nonnegative")
    if not isinstance(counts, Mapping) or not counts:
        raise ValueError("counts must be a nonempty mapping of bitstrings to counts")
    n = qubo.n_variables
    records: list[tuple[int, str, int]] = []
    for bitstring, count in counts.items():
        if (
            not isinstance(bitstring, str)
            or len(bitstring) != n
            or any(bit not in "01" for bit in bitstring)
        ):
            raise ValueError(f"Count keys must contain exactly {n} binary digits")
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count <= 0:
            raise ValueError("Each shot count must be a positive integer")
        records.append((int(bitstring, 2), bitstring, int(count)))
    records.sort()
    shots = sum(record[2] for record in records)
    sampled_bits = np.array(
        [[int(bit) for bit in record[1][::-1]] for record in records], dtype=np.int8
    )
    exact_bits = exact_costs = None
    if n <= EXACT_ENUMERATION_LIMIT:
        exact_bits, exact_costs = qubo.energies()
        sampled_costs = exact_costs[[record[0] for record in records]]
    else:
        sampled_costs = np.array([qubo.energy(bits) for bits in sampled_bits])
    if not np.all(np.isfinite(sampled_costs)):
        raise ValueError("QUBO energies must be finite")
    best_index = int(np.argmin(sampled_costs))
    best_bits = sampled_bits[best_index]
    best_states = qubo.decode(best_bits)
    probabilities = np.array([record[2] / shots for record in records])
    summary: dict[str, Any] = {
        "shots": shots,
        "unique_bitstrings": len(records),
        "best_bitstring": records[best_index][1],
        "best_bits": best_bits.tolist(),
        "best_states": best_states.tolist(),
        "best_objective": float(sampled_costs[best_index]),
        "best_sample_count": records[best_index][2],
        "mean_objective": float(probabilities @ sampled_costs),
        "exact_enumeration_limit": EXACT_ENUMERATION_LIMIT,
        "exact_tolerance": float(exact_tolerance),
        "exact_objective": None,
        "exact_objective_gap": None,
        "exact_optimum_probability": None,
        "best_is_exact_optimum": None,
        "exact_representative_bits": None,
        "best_vs_exact_bit_accuracy": None,
        "reference_metrics": None,
        "aggregate_mae_w": None,
        "aggregate_mae_weighting": None,
    }
    if exact_costs is not None:
        optimum_index = int(np.argmin(exact_costs))
        optimum = float(exact_costs[optimum_index])
        sampled_optimal = np.isclose(sampled_costs, optimum, rtol=0, atol=exact_tolerance)
        representative = exact_bits[optimum_index]
        summary.update(
            exact_objective=optimum,
            exact_objective_gap=float(sampled_costs[best_index] - optimum),
            exact_optimum_probability=float(np.sum(probabilities[sampled_optimal])),
            best_is_exact_optimum=bool(sampled_optimal[best_index]),
            exact_representative_bits=representative.tolist(),
            best_vs_exact_bit_accuracy=float(np.mean(best_bits == representative)),
        )
    if reference_states is not None:
        truth = np.asarray(reference_states)
        if truth.shape not in ((n,), (qubo.n_segments, qubo.n_appliances)):
            raise ValueError("reference_states must match the QUBO state matrix or flat bits")
        if not np.all((truth == 0) | (truth == 1)):
            raise ValueError("reference_states must contain binary values")
        summary["reference_metrics"] = _reference_metrics(truth.reshape(-1), best_bits)
    if (aggregate is None) != (powers is None):
        raise ValueError("aggregate and powers must be supplied together")
    if segment_weights is not None and aggregate is None:
        raise ValueError("segment_weights requires aggregate and powers")
    if aggregate is not None:
        aggregate_values = np.asarray(aggregate, dtype=float).reshape(-1)
        power_values = np.asarray(powers, dtype=float).reshape(-1)
        if (
            aggregate_values.size != qubo.n_segments
            or power_values.size != qubo.n_appliances
            or not np.all(np.isfinite(aggregate_values))
            or not np.all(np.isfinite(power_values))
        ):
            raise ValueError("aggregate and powers must be finite and match QUBO dimensions")
        weights = (
            np.ones(qubo.n_segments)
            if segment_weights is None
            else np.asarray(segment_weights, dtype=float).reshape(-1)
        )
        if weights.size != qubo.n_segments or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("segment_weights must be finite, positive, and match segments")
        summary["aggregate_mae_w"] = float(
            np.average(np.abs(aggregate_values - best_states @ power_values), weights=weights)
        )
        summary["aggregate_mae_weighting"] = (
            "uniform segments" if segment_weights is None else "supplied segment durations"
        )
    return summary
