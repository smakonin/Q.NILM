"""Offline ideal and provenance tests for local physical controls."""

from collections import Counter
import unittest
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Measure, Parameter, Reset
from qiskit.circuit.library import CZGate, RZGate, SXGate, XGate
from qiskit.transpiler import InstructionProperties, Target
from qiskit_aer import AerSimulator

from quantum_nilm.physical_local_controls import (
    PLACEMENTS, _inspect, build_local_controls,
)


def small_saved_target():
    """Offline target with exactly the two fixed chains, no outside gates."""
    target = Target(num_qubits=156)
    qubits = [(q,) for values in PLACEMENTS.values() for q in values]
    for gate, duration in [(XGate(), 24e-9), (SXGate(), 24e-9),
                           (RZGate(Parameter("theta")), 0), (Measure(), 1e-6),
                           (Reset(), 2e-6)]:
        target.add_instruction(gate, {q: InstructionProperties(duration=duration, error=0) for q in qubits})
    edges = []
    for triple in PLACEMENTS.values():
        for left, right in zip(triple[:-1], triple[1:]):
            edges.extend([(left, right), (right, left)])
    target.add_instruction(CZGate(), {edge: InstructionProperties(duration=68e-9, error=0) for edge in edges})
    return target


def compact(compiled, placement, *, measurements=False):
    circuit = QuantumCircuit(3, 3 if measurements else 0)
    circuit.global_phase = compiled.global_phase
    reverse = {q: i for i, q in enumerate(placement)}
    for item in compiled.data:
        name = item.operation.name
        if name == "barrier" or (name == "measure" and not measurements):
            continue
        qubits = [reverse[compiled.find_bit(q).index] for q in item.qubits]
        bits = [compiled.find_bit(bit).index for bit in item.clbits] if measurements else []
        circuit.append(item.operation, qubits, bits)
    return circuit


class PhysicalLocalControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target = small_saved_target()
        cls.items = build_local_controls(cls.target)

    def test_exact_schedule_and_metadata(self):
        self.assertEqual(len(self.items), 24)
        self.assertEqual(len({spec["id"] for _, spec in self.items}), 24)
        self.assertEqual(Counter(spec["kind"] for _, spec in self.items), {"basis": 16, "reset": 2, "w": 6})
        for circuit, spec in self.items:
            self.assertEqual(spec["group"], "local")
            self.assertEqual(spec["shots"], 1024)
            self.assertEqual(spec["physical_qubits"], list(PLACEMENTS[spec["placementlabel"]]))
            self.assertEqual(spec["logical_measurement_bit_to_physical"], spec["physical_qubits"])
            self.assertEqual(spec["active_physical_qubits"], spec["physical_qubits"])
            self.assertEqual(spec["compiled_gate_counts"], dict(circuit.count_ops()))
            self.assertEqual(spec["compiler"]["routing_method"], "none")
            self.assertEqual(spec["compiler"]["seed_transpiler"], 17)
            self.assertEqual(circuit.num_parameters, 0)
            self.assertAlmostEqual(sum(spec["expected_ideal_probabilities"].values()), 1)

    def test_all_24_compiled_ideal_probabilities_including_resets(self):
        circuits = []
        for compiled, spec in self.items:
            circuit = compact(compiled, spec["physical_qubits"])
            circuit.save_probabilities(label="ideal")
            circuits.append(circuit)
        result = AerSimulator(method="density_matrix", max_parallel_threads=1).run(circuits, shots=1).result()
        self.assertTrue(result.success)
        for index, (_, spec) in enumerate(self.items):
            with self.subTest(control=spec["id"]):
                expected = np.zeros(8)
                for bitstring, probability in spec["expected_ideal_probabilities"].items():
                    expected[int(bitstring, 2)] = probability
                actual = np.asarray(result.data(index)["ideal"])
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_basis_actual_count_bit_order_is_high_bit_first(self):
        for compiled, spec in self.items:
            if spec["kind"] != "basis":
                continue
            circuit = compact(compiled, spec["physical_qubits"], measurements=True)
            counts = AerSimulator(max_parallel_threads=1).run(circuit, shots=8).result().get_counts()
            self.assertEqual(counts, {spec["input_state"]: 8})
            x_qubits = sorted(compiled.find_bit(item.qubits[0]).index for item in compiled.data if item.operation.name == "x")
            pattern = int(spec["input_state"], 2)
            expected = [q for bit, q in enumerate(spec["physical_qubits"]) if (pattern >> bit) & 1]
            self.assertEqual(x_qubits, expected)

    def test_reset_preserves_excited_preamble_and_resets(self):
        for circuit, spec in self.items:
            if spec["kind"] != "reset":
                continue
            names = [item.operation.name for item in circuit.data]
            self.assertEqual(names.count("reset"), 3)
            self.assertEqual(names.count("x"), 3)
            self.assertEqual(names.count("barrier"), 2)
            self.assertLess(max(i for i, name in enumerate(names) if name == "x"),
                            min(i for i, name in enumerate(names) if name == "reset"))
            self.assertEqual(spec["input_state_before_reset"], "111")
            self.assertEqual(spec["expected_ideal_probabilities"], {"000": 1.0})

    def test_w_repeats_separately_identified_with_equal_native_operations(self):
        for label in PLACEMENTS:
            group = [(c, s) for c, s in self.items if s["placementlabel"] == label and s["kind"] == "w"]
            self.assertEqual([s["repeat"] for _, s in group], [0, 1, 2])
            signatures = [[(item.operation.name, tuple(c.find_bit(q).index for q in item.qubits), tuple(map(str, item.operation.params))) for item in c.data] for c, _ in group]
            self.assertEqual(signatures[0], signatures[1])
            self.assertEqual(signatures[1], signatures[2])
            self.assertEqual(group[0][1]["expected_ideal_probabilities"], {"001": 1 / 3, "010": 1 / 3, "100": 1 / 3})

    def test_target_support_failures_are_explicit(self):
        for target in (None, object(), Target(num_qubits=8), Target(num_qubits=156)):
            with self.subTest(target=target), self.assertRaises(ValueError):
                build_local_controls(target)

    def test_inspector_rejects_outside_qubit_and_mapping_changes(self):
        circuit = QuantumCircuit(156, 3)
        circuit.x(130)
        circuit.measure([131, 132, 133], range(3))
        with self.assertRaisesRegex(ValueError, "outside"):
            _inspect(circuit, (131, 132, 133), "basis")
        circuit = QuantumCircuit(156, 3)
        circuit.measure([133, 132, 131], range(3))
        with self.assertRaisesRegex(ValueError, "mapping"):
            _inspect(circuit, (131, 132, 133), "basis")

    def test_inspector_rejects_silently_removed_reset_preamble(self):
        circuit = QuantumCircuit(156, 3)
        circuit.reset([131, 132, 133])
        circuit.measure([131, 132, 133], range(3))
        with self.assertRaisesRegex(ValueError, "preamble"):
            _inspect(circuit, (131, 132, 133), "reset")

    def test_compiler_options_are_fixed(self):
        from quantum_nilm import physical_local_controls as module
        original = module.generate_preset_pass_manager
        with patch.object(module, "generate_preset_pass_manager", wraps=original) as manager:
            build_local_controls(self.target)
        self.assertEqual(manager.call_count, 2)
        for call, placement in zip(manager.call_args_list, PLACEMENTS.values()):
            self.assertEqual(call.kwargs["initial_layout"], list(placement))
            self.assertEqual(call.kwargs["optimization_level"], 1)
            self.assertEqual(call.kwargs["seed_transpiler"], 17)
            self.assertEqual(call.kwargs["approximation_degree"], 1.)
            self.assertEqual(call.kwargs["routing_method"], "none")


if __name__ == "__main__":
    unittest.main()
