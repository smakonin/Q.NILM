"""Offline circuit, recovery, phase-sign and physical-edge provenance checks."""

from collections import Counter
import copy
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Measure, Parameter
from qiskit.circuit.library import CZGate, RZGate, SXGate, XGate
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler import InstructionProperties, Target

from quantum_nilm.physical_gate_benchmarks import (
    EDGES, RB_LENGTHS, RB_SEEDS, _clifford_sequence, _edge_target,
    _embed_and_check, _ramsey_logical, _segment_cz_counts, _sequence_hash,
    build_gate_benchmarks,
)


def native_target():
    """Only the two in-scope physical edges exist on this offline target."""
    target = Target(num_qubits=156)
    qubits = [(q,) for _, edge in EDGES for q in edge]
    for operation in (RZGate(Parameter("theta")), SXGate(), XGate(), Measure()):
        target.add_instruction(operation, {
            q: InstructionProperties(duration=1e-8, error=0) for q in qubits
        })
    edges = [direction for _, edge in EDGES for direction in (edge, edge[::-1])]
    target.add_instruction(CZGate(), {
        edge: InstructionProperties(duration=68e-9, error=0) for edge in edges
    })
    return target


def compact(circuit, edge):
    """Independent explicit physical-to-logical map; never simulate 156 wires."""
    result = QuantumCircuit(2)
    result.global_phase = circuit.global_phase
    reverse = {physical: logical for logical, physical in enumerate(edge)}
    for item in circuit.data:
        if item.operation.name not in ("measure", "barrier"):
            qubits = [reverse[circuit.find_bit(q).index] for q in item.qubits]
            result.append(item.operation, qubits)
    return result


class PhysicalGateBenchmarksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target = native_target()
        cls.items = build_gate_benchmarks(cls.target)

    def test_exact_matrix_shots_and_metadata(self):
        self.assertEqual(len(self.items), 192)
        self.assertEqual(len({spec["id"] for _, spec in self.items}), 192)
        self.assertEqual(Counter(s["kind"] for _, s in self.items), {"rb": 160, "ramsey": 32})
        self.assertEqual(sum(s["shots"] for _, s in self.items), 57344)
        for circuit, spec in self.items:
            self.assertEqual(spec["group"], "gate")
            self.assertEqual(spec["shots"], 256 if spec["kind"] == "rb" else 512)
            self.assertEqual(circuit.num_qubits, 156)
            self.assertEqual(circuit.num_clbits, 2)
            self.assertEqual(circuit.num_parameters, 0)
            self.assertEqual(circuit.cregs[0].name, "meas")
            self.assertEqual(circuit.metadata, spec)
            self.assertIsNot(circuit.metadata, spec)
            self.assertEqual(spec["physical_layout"], spec["edge"])
            self.assertEqual(spec["logical_measurement_bit_to_physical"], spec["edge"])
            self.assertEqual(spec["total_native_cz_count"], circuit.count_ops().get("cz", 0))
            self.assertFalse(spec["runtime_gate_twirling"])
            self.assertFalse(spec["runtime_measurement_twirling"])
            self.assertFalse(spec["runtime_dynamical_decoupling"])

    def test_all_192_compiled_ideal_probabilities(self):
        for circuit, spec in self.items:
            with self.subTest(circuit=spec["id"]):
                actual = Statevector.from_instruction(compact(circuit, spec["edge"])).probabilities()
                expected = np.asarray([spec["ideal_probs"][format(i, "02b")] for i in range(4)])
                np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=0)

    def test_all_160_rb_recover_full_identity_not_only_zero_state(self):
        for circuit, spec in self.items:
            if spec["kind"] != "rb":
                continue
            with self.subTest(circuit=spec["id"]):
                matrix = Operator(compact(circuit, spec["edge"])).data
                phase = matrix[0, 0] / abs(matrix[0, 0])
                np.testing.assert_allclose(matrix, phase * np.eye(4), atol=3e-11, rtol=0)

    def test_paired_random_clifford_streams_across_arms_edges_and_prefixes(self):
        for seed in RB_SEEDS:
            operators, _ = _clifford_sequence(seed, max(RB_LENGTHS))
            for length in RB_LENGTHS:
                group = [s for _, s in self.items if s["kind"] == "rb"
                         and s["seed"] == seed and s["length"] == length]
                self.assertEqual(len(group), 4)
                self.assertEqual({s["clifford_tableau_sha256"] for s in group},
                                 {_sequence_hash(operators[:length])})
                # Target CZ blocks are separate; random Clifford compilation
                # stays identical across the paired arms and edge subsets.
                self.assertEqual(len({s["random_clifford_native_cz_count"] for s in group}), 1)

    def test_full_two_qubit_random_cliffords_include_entangling_operations(self):
        # Tensor-product-only sampling would invalidate the two-qubit RB model.
        operators, decompositions = _clifford_sequence(RB_SEEDS[0], 128)
        self.assertEqual(len({op.tableau.tobytes() for op in operators}), len(operators))
        self.assertTrue(any(gates.num_nonlocal_gates() > 0 for gates in decompositions))
        purities = []
        for gates in decompositions[:16]:
            amplitudes = Statevector.from_instruction(gates).data.reshape(2, 2)
            density = amplitudes @ amplitudes.conj().T
            purities.append(np.trace(density @ density).real)
        self.assertLess(min(purities), 0.75)

    def test_native_cz_blocks_are_retained_even_for_even_repetition_counts(self):
        for circuit, spec in self.items:
            with self.subTest(circuit=spec["id"]):
                counts = _segment_cz_counts(circuit)
                self.assertEqual(counts, spec["native_cz_count_by_barrier_block"])
                if spec["kind"] == "rb":
                    actual = {k: v for k, v in counts.items() if k.startswith("rb_interleaved_")}
                    expected = spec["length"] if spec["arm"] == "interleaved" else 0
                    self.assertEqual(len(actual), expected)
                    self.assertEqual(sum(actual.values()), expected)
                    self.assertTrue(all(v == 1 for v in actual.values()))
                    self.assertEqual(spec["interleaved_cz_count"], expected)
                    self.assertEqual(spec["total_native_cz_count"], expected
                                     + spec["random_clifford_native_cz_count"]
                                     + spec["inverse_native_cz_count"])
                else:
                    self.assertEqual(spec["retained_cz_count"], spec["repetitions"])
                    self.assertEqual(sum(counts.values()), spec["repetitions"])

    def test_native_support_no_off_edge_routing_and_measurement_order(self):
        for circuit, spec in self.items:
            measurements = []
            for item in circuit.data:
                qargs = tuple(circuit.find_bit(q).index for q in item.qubits)
                self.assertTrue(set(qargs).issubset(spec["edge"]))
                if item.operation.name == "barrier":
                    continue
                self.assertTrue(self.target.instruction_supported(
                    operation_name=item.operation.name, qargs=qargs,
                    parameters=list(item.operation.params)))
                if item.operation.name == "measure":
                    measurements.append((qargs[0], circuit.find_bit(item.clbits[0]).index))
            self.assertEqual(measurements, [(spec["edge"][0], 0), (spec["edge"][1], 1)])

    def test_ramsey_analysis_sign_is_positive_y_not_negative_y(self):
        logical, _, _ = _ramsey_logical(0, "Y", 0, "analysis_sign")
        analysis = QuantumCircuit(1)
        active = False
        for item in logical.data:
            if item.operation.name == "barrier":
                active = item.operation.label == "ramsey_analysis"
            elif active:
                self.assertEqual([logical.find_bit(q).index for q in item.qubits], [1])
                analysis.append(item.operation, [0])
        self.assertEqual([item.operation.name for item in analysis.data], ["sdg", "h"])
        unitary = Operator(analysis).data
        z = np.diag([1., -1.])
        y = np.asarray([[0., -1j], [1j, 0.]])
        np.testing.assert_allclose(unitary.conj().T @ z @ unitary, y, atol=1e-12)

    def test_ramsey_bit_order_and_all_shot_pauli_expectations(self):
        for circuit, spec in self.items:
            if spec["kind"] != "ramsey":
                continue
            probabilities = Statevector.from_instruction(compact(circuit, spec["edge"])).probabilities()
            target_expectation = sum(p * (-1) ** ((i >> 1) & 1) for i, p in enumerate(probabilities))
            control_flip = sum(p for i, p in enumerate(probabilities) if i & 1 != spec["control"])
            self.assertAlmostEqual(target_expectation, spec["ideal_expectation"], places=11)
            self.assertAlmostEqual(control_flip, 0., places=11)

    def test_target_failures_are_explicit(self):
        for target in (None, object(), Target(num_qubits=8), Target(num_qubits=156)):
            with self.subTest(target=target), self.assertRaises(ValueError):
                _edge_target(target, (132, 133))
        with self.assertRaises(ValueError):
            _edge_target(self.target, (132, 141))

    def test_missing_or_duplicate_barriers_are_rejected(self):
        circuit = QuantumCircuit(2)
        circuit.cz(0, 1)
        with self.assertRaisesRegex(ValueError, "unlabelled"):
            _segment_cz_counts(circuit)
        circuit = QuantumCircuit(2)
        circuit.barrier(label="same")
        circuit.barrier(label="same")
        with self.assertRaisesRegex(ValueError, "duplicated"):
            _segment_cz_counts(circuit)

    def test_cancelled_target_cz_is_rejected(self):
        # Use one actual already-compiled circuit to isolate the retention gate.
        compiled, original = next((c, s) for c, s in self.items
                                 if s["kind"] == "ramsey" and s["repetitions"] == 4)
        local = QuantumCircuit(2, 2)
        reverse = {q: i for i, q in enumerate(original["edge"])}
        dropped = False
        for item in compiled.data:
            if item.operation.name == "cz" and not dropped:
                dropped = True
                continue
            qargs = [reverse[compiled.find_bit(q).index] for q in item.qubits]
            cargs = [compiled.find_bit(c).index for c in item.clbits]
            local.append(item.operation, qargs, cargs)
        self.assertTrue(dropped)
        with self.assertRaisesRegex(ValueError, "removed or altered"):
            _embed_and_check(local, self.target, tuple(original["edge"]), copy.deepcopy(original))


if __name__ == "__main__":
    unittest.main()
