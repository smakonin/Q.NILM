"""Bounded native-CZ diagnostics; circuit construction only, never account access.

The reference/interleaved randomized benchmarking design follows Magesan
et al., Phys. Rev. Lett. 109, 080505 (2012), DOI
https://doi.org/10.1103/PhysRevLett.109.080505. This is a small independent
implementation, not a reproduction of the qiskit-experiments library.
Random operations are full two-qubit Cliffords, not tensor products of
single-qubit Cliffords. Each arm has its own exact recovery Clifford.

Error-rate extraction requires decay-model and noise assumptions; circuits
alone do not identify a unique error channel, pulse fault, or causality.
The accompanying phase-sensitive controls measure the second qubit's X/Y
expectation without conditioning on a successful first-qubit readout.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np


EDGES = (("suspect", (132, 133)), ("control", (141, 142)))
RB_LENGTHS = (1, 4, 16, 64, 128)
RB_SEEDS = tuple(range(3101, 3109))
RAMSEY_REPETITIONS = (0, 1, 4, 16)
RB_SHOTS = 256
RAMSEY_SHOTS = 512
TRANSPILE_SEED = 4917
NATIVE_NAMES = ("rz", "sx", "x", "cz", "measure")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _new_logical(name):
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    return QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(2, "meas"), name=name)


def _clifford_sequence(seed, maximum_length):
    from qiskit.quantum_info import random_clifford

    rng = np.random.default_rng(seed)
    operators = [random_clifford(2, seed=rng) for _ in range(maximum_length)]
    return operators, [operator.to_circuit() for operator in operators]


def _sequence_hash(operators):
    payload = [operator.tableau.astype(int).tolist() for operator in operators]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _rb_logical(operators, decompositions, *, interleaved, name):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Clifford

    _require(len(operators) == len(decompositions) and len(operators) > 0, "Nonempty paired Clifford sequence required")
    circuit = _new_logical(name)
    accumulated = Clifford(QuantumCircuit(2))
    cz_circuit = QuantumCircuit(2)
    cz_circuit.cz(0, 1)
    cz_operator = Clifford(cz_circuit)
    for index, (operator, gates) in enumerate(zip(operators, decompositions)):
        circuit.barrier(0, 1, label=f"rb_random_{index:03d}")
        circuit.compose(gates, qubits=[0, 1], inplace=True)
        accumulated = accumulated.compose(operator)
        if interleaved:
            # A separate labelled barrier block prevents cancellation with
            # random or recovery Cliffords during basis compilation.
            circuit.barrier(0, 1, label=f"rb_interleaved_{index:03d}")
            circuit.cz(0, 1)
            accumulated = accumulated.compose(cz_operator)
    circuit.barrier(0, 1, label="rb_inverse")
    inverse = accumulated.adjoint()
    _require(accumulated.compose(inverse) == Clifford(QuantumCircuit(2)),
             "Recovery Clifford is not the inverse of the complete sequence")
    circuit.compose(inverse.to_circuit(), qubits=[0, 1], inplace=True)
    circuit.barrier(0, 1, label="rb_measure")
    circuit.measure([0, 1], [0, 1])
    return circuit


def _ramsey_logical(control, basis, repetitions, name):
    _require(control in (0, 1) and basis in ("X", "Y") and repetitions in RAMSEY_REPETITIONS,
             "Ramsey condition is outside the frozen matrix")
    circuit = _new_logical(name)
    circuit.barrier(0, 1, label="ramsey_prepare")
    if control:
        circuit.x(0)
    circuit.h(1)
    for index in range(repetitions):
        circuit.barrier(0, 1, label=f"ramsey_cz_{index:03d}")
        circuit.cz(0, 1)
    circuit.barrier(0, 1, label="ramsey_analysis")
    if basis == "Y":
        # Sdg then H maps positive Y to positive measured Z, not negative Y.
        circuit.sdg(1)
    circuit.h(1)
    circuit.barrier(0, 1, label="ramsey_measure")
    circuit.measure([0, 1], [0, 1])
    expectation = float((-1) ** (control * repetitions)) if basis == "X" else 0.0
    probabilities = {key: 0.0 for key in ("00", "01", "10", "11")}
    probabilities[f"0{control}"] = (1.0 + expectation) / 2.0
    probabilities[f"1{control}"] = (1.0 - expectation) / 2.0
    return circuit, expectation, probabilities


def _edge_target(target, edge):
    """Construct an exact two-qubit subset of the physical native target.

    Local compilation cannot visit any other physical qubit. This subset is
    used only for basis translation; the result is subsequently embedded back
    into the full physical wire indexing and rechecked against the target.
    """
    from qiskit.transpiler import Target

    _require(getattr(target, "num_qubits", None) is not None
             and len(set(edge)) == 2 and min(edge) >= 0 and max(edge) < target.num_qubits,
             "Physical diagnostic edge is outside the supplied target")
    reduced = Target(num_qubits=2, dt=target.dt)
    for name in NATIVE_NAMES:
        _require(name in target.operation_names, f"Native target lacks required instruction {name}")
        operation = target.operation_from_name(name)
        properties = {}
        qargs = [(0, 1), (1, 0)] if name == "cz" else [(0,), (1,)]
        for local in qargs:
            physical = tuple(edge[index] for index in local)
            if target.instruction_supported(operation_name=name, qargs=physical):
                props = target[name].get(physical, target[name].get(None))
                properties[local] = copy.copy(props)
        _require(properties and (name == "cz" or len(properties) == 2),
                 f"Instruction {name} is not available on both required edge qubits: {edge}")
        reduced.add_instruction(operation, properties, name=name)
    return reduced


def _segment_cz_counts(circuit):
    counts, segment = {}, None
    for item in circuit.data:
        if item.operation.name == "barrier":
            segment = item.operation.label
            _require(segment is not None and segment not in counts, "Missing or duplicated diagnostic barrier label")
            counts[segment] = 0
        elif item.operation.num_qubits == 2:
            _require(item.operation.name == "cz" and segment is not None,
                     "Unexpected non-CZ or unlabelled two-qubit instruction")
            counts[segment] += 1
    return counts


def _embed_and_check(local, target, edge, spec):
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    _require(local.num_qubits == 2 and local.num_clbits == 2 and local.num_parameters == 0,
             "Compiled diagnostic must remain a bound two-qubit circuit")
    physical = QuantumCircuit(QuantumRegister(target.num_qubits, "q"),
                              ClassicalRegister(2, "meas"), name=spec["id"])
    physical.compose(local, qubits=list(edge), clbits=[0, 1], inplace=True)
    measurement_map = []
    for item in physical.data:
        qargs = tuple(physical.find_bit(bit).index for bit in item.qubits)
        _require(set(qargs).issubset(edge), "Diagnostic compilation used qubits outside its fixed edge")
        if item.operation.name == "barrier":
            continue
        _require(target.instruction_supported(operation_name=item.operation.name, qargs=qargs,
                                             parameters=list(item.operation.params)),
                 f"Compiled operation is not supported by the physical target: {item.operation.name}{qargs}")
        if item.operation.name == "measure":
            measurement_map.append((qargs[0], physical.find_bit(item.clbits[0]).index))
    _require(measurement_map == [(edge[0], 0), (edge[1], 1)], "Measurement order changed during compilation")
    counts = _segment_cz_counts(physical)
    if spec["kind"] == "rb":
        interleaved = [value for name, value in counts.items() if name.startswith("rb_interleaved_")]
        expected = spec["length"] if spec["arm"] == "interleaved" else 0
        _require(len(interleaved) == expected and all(value == 1 for value in interleaved),
                 "An interleaved native CZ was removed or altered")
        random_counts = [value for name, value in counts.items() if name.startswith("rb_random_")]
        _require(len(random_counts) == spec["length"] and "rb_inverse" in counts and "rb_measure" in counts,
                 "Random Clifford or recovery boundaries were lost")
        spec.update({"interleaved_cz_count": sum(interleaved),
                     "random_clifford_native_cz_count": sum(random_counts),
                     "inverse_native_cz_count": counts["rb_inverse"]})
    else:
        repeated = [value for name, value in counts.items() if name.startswith("ramsey_cz_")]
        _require(len(repeated) == spec["repetitions"] and all(value == 1 for value in repeated),
                 "A Ramsey native CZ was removed or altered")
        _require(sum(counts.values()) == spec["repetitions"], "Unexpected native CZ outside Ramsey repetition blocks")
        spec["retained_cz_count"] = sum(repeated)
    spec.update({"total_native_cz_count": sum(counts.values()), "native_cz_count_by_barrier_block": counts,
                 "native_operation_counts": {str(k): int(v) for k, v in physical.count_ops().items()},
                 "compiled_depth": int(physical.depth()), "physical_qubit_count": target.num_qubits,
                 "logical_qubits": 2, "physical_layout": list(edge),
                 "measurement_physical_qubits": list(edge),
                 "logical_measurement_bit_to_physical": list(edge),
                 "measurement_map": [{"physical_qubit": q, "classical_bit": c} for q, c in measurement_map],
                 "classical_register": "meas", "bitstring_order": "c1c0; c0=edge[0], c1=edge[1]",
                 "optimization_level": 1, "seed_transpiler": TRANSPILE_SEED,
                 "routing_method": "none; compile on exact two-qubit native subset then embed on physical edge",
                 "barriers_preserved": True, "runtime_gate_twirling": False,
                 "runtime_measurement_twirling": False, "runtime_dynamical_decoupling": False})
    physical.metadata = copy.deepcopy(spec)
    return physical, spec


def build_gate_benchmarks(target):
    """Return all 192 fixed native-CZ diagnostic ``(circuit, spec)`` pairs.

    There are 160 RB circuits at 256 shots and 32 Ramsey circuits at 512
    shots (57,344 requested shots total). Eight seeded full two-qubit
    Clifford streams are shared across reference/interleaved arms AND
    physical edges. Each shorter length uses the same stream prefix.
    Thus seed, rather than a length/arm cell, is the resampling unit for
    paired uncertainty across lengths and edges.

    All wires outside the chosen two-qubit edge are idle. Compiling on a
    restricted native target prevents routing to other physical qubits.
    Returned circuits have the full target width for explicit physical wire
    indices, but only two measured classical bits in register ``meas``.
    Metadata records intended Runtime settings; the submitting caller must
    explicitly enforce those settings. This function never submits work.
    """
    from qiskit import transpile

    streams = {seed: _clifford_sequence(seed, max(RB_LENGTHS)) for seed in RB_SEEDS}
    output = []
    for label, edge in EDGES:
        reduced = _edge_target(target, edge)
        logical, specifications = [], []
        for seed in RB_SEEDS:
            operators, decompositions = streams[seed]
            for length in RB_LENGTHS:
                sequence_hash = _sequence_hash(operators[:length])
                for arm in ("reference", "interleaved"):
                    name = f"gate_rb_{label}_m{length}_seed{seed}_{arm}"
                    logical.append(_rb_logical(operators[:length], decompositions[:length],
                                              interleaved=arm == "interleaved", name=name))
                    specifications.append({"id": name, "group": "gate", "kind": "rb",
                        "edge": list(edge), "edge_label": label, "seed": seed, "length": length,
                        "arm": arm, "shots": RB_SHOTS, "pair_id": f"rb_{label}_m{length}_seed{seed}",
                        "cross_edge_pair_id": f"rb_m{length}_seed{seed}_{arm}",
                        "clifford_tableau_sha256": sequence_hash, "prefix_stream_seed": seed,
                        "clifford_sampling": "full two-qubit Qiskit random_clifford; same prefixes across lengths, arms and edges",
                        "recovery": "exact inverse of this arm's entire Clifford/CZ sequence",
                        "ideal_probs": {"00": 1.0, "01": 0.0, "10": 0.0, "11": 0.0},
                        "ideal_return_probability": 1.0,
                        "method_reference_doi": "10.1103/PhysRevLett.109.080505",
                        "implementation": "independent bounded circuit construction, not qiskit-experiments"})
        for control in (0, 1):
            for basis in ("X", "Y"):
                for repetitions in RAMSEY_REPETITIONS:
                    name = f"gate_ramsey_{label}_control{control}_{basis}_n{repetitions}"
                    circuit, expectation, probabilities = _ramsey_logical(control, basis, repetitions, name)
                    logical.append(circuit)
                    specifications.append({"id": name, "group": "gate", "kind": "ramsey",
                        "edge": list(edge), "edge_label": label, "control": control,
                        "basis": basis, "repetitions": repetitions, "shots": RAMSEY_SHOTS,
                        "ideal_expectation": expectation, "ideal_probs": probabilities,
                        "target_classical_bit": 1, "control_classical_bit": 0,
                        "expectation_estimator": "all-shot target-bit Pauli marginal; no conditioning on control readout",
                        "Y_analysis_rotation": "Sdg then H, so measured Z has positive Y sign"})
        compiled = transpile(logical, target=reduced, initial_layout=[0, 1], routing_method="none",
                             optimization_level=1, approximation_degree=1.0,
                             seed_transpiler=TRANSPILE_SEED, num_processes=1)
        for circuit, spec in zip(compiled, specifications):
            output.append(_embed_and_check(circuit, target, edge, spec))
    _require(len(output) == 192 and len({spec["id"] for _, spec in output}) == 192,
             "Diagnostic matrix is incomplete or has duplicate circuit IDs")
    _require(sum(spec["shots"] for _, spec in output) == 57344, "Diagnostic shot budget changed")
    return output
