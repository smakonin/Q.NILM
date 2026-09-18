#!/usr/bin/env python3
"""Offline calibration burden and independent-readout feasibility bounds."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register_readout_feasibility(false_zero, false_one):
    """P(exactly one observed1 | each ideal one-hot category), independent errors.

    false_zero[j] is P(measure0|prepare1); false_one[j] is P(measure1|prepare0).
    Simultaneous errors that move an excitation also remain feasible.
    """
    fz, fo = np.asarray(false_zero, dtype=float), np.asarray(false_one, dtype=float)
    if (fz.ndim != 1 or not len(fz) or fz.shape != fo.shape
            or not np.all(np.isfinite(fz)) or not np.all(np.isfinite(fo))
            or np.any((fz < 0) | (fz > 1)) or np.any((fo < 0) | (fo > 1))):
        raise ValueError("Expected finite readout probabilities in [0,1]")
    result = []
    for category in range(len(fz)):
        ones = fo.copy()
        ones[category] = 1 - fz[category]
        result.append(sum(ones[j] * np.prod(np.delete(1 - ones, j)) for j in range(len(ones))))
    return np.asarray(result)


def audit(folder):
    from qiskit import qpy, qasm3

    folder = Path(folder)
    plan = json.loads((folder / "plan.json").read_text())
    properties = json.loads((folder / "backend_properties.json").read_text())
    qubits = [{p["name"]: p["value"] for p in q} for q in properties["qubits"]]
    gates = {(g["gate"], tuple(sorted(g["qubits"]))): {p["name"]: p["value"] for p in g["parameters"]}
             for g in properties["gates"]}
    source_paths = [folder / name for name in ("plan.json", "backend_properties.json", "template_1.qpy", "template_2.qpy",
                                               "template_1.qasm", "template_2.qasm")]
    for path in source_paths[1:]:
        if sha(path) != plan["files_sha256"][path.name]:
            raise ValueError("Archived artifact does not match its frozen plan")
    rows = []
    for width in (1, 2):
        with (folder / f"template_{width}.qpy").open("rb") as handle:
            circuit = qpy.load(handle)[0]
        if qasm3.dumps(circuit) != (folder / f"template_{width}.qasm").read_text():
            raise ValueError("Saved QPY and QASM do not describe the same circuit")
        mapping = {circuit.find_bit(i.clbits[0]).index: circuit.find_bit(i.qubits[0]).index
                   for i in circuit.data if i.operation.name == "measure"}
        sizes = tuple(circuit.metadata["state_counts"]) * width
        if set(mapping) != set(range(sum(sizes))):
            raise ValueError("Measurement mapping is incomplete")
        register_probabilities, offset = [], 0
        for size in sizes:
            register_probabilities.append(register_readout_feasibility(
                [qubits[mapping[offset + j]]["prob_meas0_prep1"] for j in range(size)],
                [qubits[mapping[offset + j]]["prob_meas1_prep0"] for j in range(size)]).tolist())
            offset += size
        pairs = Counter(tuple(sorted(circuit.find_bit(q).index for q in i.qubits))
                        for i in circuit.data if i.operation.name == "cz")
        errors = np.array([gates["cz", pair]["gate_error"] for pair, count in pairs.items() for _ in range(count)])
        rows.append({"intervals": width, "statistics": plan["templates"][str(width)]["statistics"],
                     "duration_s": plan["templates"][str(width)]["duration_s"],
                     "logical_bit_to_physical_measurement": [mapping[j] for j in range(sum(sizes))],
                     "register_category_readout_feasible_probabilities": register_probabilities,
                     "readout_only_feasible_probability_min": float(np.prod([min(x) for x in register_probabilities])),
                     "readout_only_feasible_probability_max": float(np.prod([max(x) for x in register_probabilities])),
                     "uniform_onehot_readout_only_feasible_probability": float(np.prod([np.mean(x) for x in register_probabilities])),
                     "cz_gate_error_weighted_mean": float(errors.mean()),
                     "cz_gate_error_min": float(errors.min()), "cz_gate_error_max": float(errors.max()),
                     "cz_product_one_minus_error_proxy": float(np.prod(1 - errors)),
                     "used_cz_couplers": [{"qubits": pair, "count": count, **gates["cz", pair]}
                                          for pair, count in sorted(pairs.items())]})
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": {str(p): sha(p) for p in source_paths},
            "code_sha256": {str(Path(__file__)): sha(__file__)}, "backend": plan["backend"],
            "calibration_last_update": properties["last_update_date"], "rows": rows,
            "assumptions": ["Ideal pre-readout state has support only on the one-hot feasible subspace",
                            "Independent, stationary asymmetric readout channels from the archived snapshot",
                            "Bounds minimize/maximize over every feasible categorical register assignment"],
            "limitations": ["Model bounds, not physical guarantees under drift, correlated errors or leakage",
                            "CZ product(1-error) is only a burden proxy, not circuit fidelity or predicted feasibility",
                            "No new calibration/account/QPU call; no causal isolation of gate, idle or readout noise"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/quantum_heldout/hardware"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "readout_only_bounds": [[r["readout_only_feasible_probability_min"],
           r["readout_only_feasible_probability_max"]] for r in result["rows"]]}))


if __name__ == "__main__":
    main()
