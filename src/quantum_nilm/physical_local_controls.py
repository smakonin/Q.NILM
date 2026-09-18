"""Offline, fixed-placement local preparation/readout/reset controls.

These are newly compiled diagnostic circuits, not replacements for historical
Q.NILM circuits. No account, service, or submission APIs are imported or used.
"""
from __future__ import annotations

from collections import Counter

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import Target, generate_preset_pass_manager

from quantum_nilm.categorical_ibm import prepare_uniform_onehot


PLACEMENTS = {"suspect": (131, 132, 133), "control": (141, 142, 143)}
SHOTS = 1024
COMPILER = {
    "optimization_level": 1,
    "seed_transpiler": 17,
    "approximation_degree": 1.0,
    "routing_method": "none",
    "fixed_initial_layout": True,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _logical(kind, pattern=None):
    circuit = QuantumCircuit(QuantumRegister(3, "q"), ClassicalRegister(3, "meas"))
    if kind == "basis":
        for bit in range(3):
            if (pattern >> bit) & 1:
                circuit.x(bit)
    elif kind == "reset":
        circuit.x(range(3))
        # Barriers deliberately retain the excited-state preamble before the
        # tested resets; this is not a removable initial reset on |000>.
        circuit.barrier()
        circuit.reset(range(3))
    elif kind == "w":
        prepare_uniform_onehot(circuit, range(3))
    else:
        raise ValueError("Unknown local diagnostic kind")
    circuit.barrier()
    circuit.measure(range(3), range(3))
    return circuit


def _inspect(compiled, placement, kind):
    active, mapping = set(), {}
    reset_counts, prefix_x = Counter(), Counter()
    saw_reset = False
    for item in compiled.data:
        name = item.operation.name
        if name == "barrier":
            continue
        physical = [compiled.find_bit(qubit).index for qubit in item.qubits]
        active.update(physical)
        if name == "reset":
            saw_reset = True
            reset_counts.update(physical)
        elif name == "x" and not saw_reset:
            prefix_x.update(physical)
        elif name == "measure":
            bit = compiled.find_bit(item.clbits[0]).index
            _require(bit not in mapping, "Repeated measurement bit")
            mapping[bit] = physical[0]
    _require(active == set(placement), "Local control uses qubits outside its fixed triple")
    _require(mapping == dict(enumerate(placement)), "Logical measurement mapping changed")
    _require(compiled.num_clbits == 3 and compiled.num_parameters == 0, "Unexpected control width or parameters")
    if kind == "reset":
        _require(reset_counts == Counter({q: 1 for q in placement}), "Explicit reset was removed or duplicated")
        _require(prefix_x == Counter({q: 1 for q in placement}), "Excited-state reset preamble was changed")
        _require(compiled.count_ops().get("barrier", 0) >= 2, "Reset phase barriers were removed")
    else:
        _require(not reset_counts, "Unrequested reset in basis/W control")
    return {"active_physical_qubits": sorted(active),
            "logical_measurement_bit_to_physical": [mapping[i] for i in range(3)],
            "compiled_gate_counts": dict(compiled.count_ops()),
            "compiled_depth": compiled.depth(),
            "allocated_qubits": compiled.num_qubits}


def build_local_controls(target: Target) -> list[tuple[QuantumCircuit, dict]]:
    """Build the frozen 24-circuit local-control group entirely offline.

    Each triple receives all eight computational-basis patterns, one
    |111> -> explicit reset -> |000> check, and three separately identified W3
    repeats. Repeats share the same preparation and are not independent
    dates/layouts. Classical bit i maps to physical_qubits[i]; displayed
    bitstrings are high-bit-first, so their rightmost bit is physical_qubits[0].

    Routing is disabled because only adjacent interactions on each supplied
    chain are needed. Missing native target support or any changed placement
    fails explicitly rather than silently using additional qubits.
    """
    _require(isinstance(target, Target), "A saved Qiskit Target is required")
    _require(target.num_qubits is not None and target.num_qubits > max(max(q) for q in PLACEMENTS.values()),
             "Target does not contain both fixed diagnostic triples")
    for placement in PLACEMENTS.values():
        for qubit in placement:
            for operation in ("x", "measure", "reset"):
                _require(target.instruction_supported(operation_name=operation, qargs=(qubit,)),
                         f"Target lacks {operation} on physical qubit {qubit}")
    output = []
    for label, placement in PLACEMENTS.items():
        manager = generate_preset_pass_manager(target=target, initial_layout=list(placement),
            optimization_level=1, seed_transpiler=17, approximation_degree=1.0, routing_method="none")
        definitions = [("basis", pattern, 0) for pattern in range(8)]
        definitions += [("reset", None, 0)] + [("w", None, repeat) for repeat in range(3)]
        for kind, pattern, repeat in definitions:
            suffix = f"basis_{pattern:03b}" if kind == "basis" else "reset" if kind == "reset" else f"w_{repeat}"
            identifier = f"local_{label}_{suffix}"
            logical = _logical(kind, pattern)
            logical.name = identifier
            compiled = manager.run(logical)
            expected = ({f"{pattern:03b}": 1.0} if kind == "basis" else {"000": 1.0}
                        if kind == "reset" else {"001": 1 / 3, "010": 1 / 3, "100": 1 / 3})
            spec = {"id": identifier, "group": "local", "kind": kind,
                    "placementlabel": label, "physical_qubits": list(placement),
                    "repeat": repeat, "shots": SHOTS,
                    "expected_ideal_probabilities": expected,
                    "bitstring_order": "high classical bit first; rightmost bit maps to physical_qubits[0]",
                    "compiler": dict(COMPILER), **_inspect(compiled, placement, kind)}
            if kind == "basis":
                spec["input_state"] = f"{pattern:03b}"
            if kind == "reset":
                spec["input_state_before_reset"] = "111"
            compiled.metadata = {"algorithm": "Q.NILM local physical diagnostic", **spec}
            output.append((compiled, spec))
    _require(len(output) == 24 and len({spec["id"] for _, spec in output}) == 24,
             "Unexpected or duplicate local-control schedule")
    return output
