"""Offline one-hot circuit, endian, and explicit infeasibility tests."""

import importlib.util
import unittest

import numpy as np

from quantum_nilm.categorical_ibm import (
    build_categorical_qaoa_circuit, decode_categorical_counts,
    prepare_uniform_onehot, score_categorical_counts,
    build_categorical_parameterized_template, categorical_template_parameter_values,
    bind_categorical_template,
)
from quantum_nilm.categorical_qaoa import (
    categorical_qaoa_probabilities, encode_onehot, prepare_categorical_problem,
)


class CategoricalDecodingTests(unittest.TestCase):
    def setUp(self):
        self.problem = prepare_categorical_problem(
            [105., 213.], [np.array([-1., 101.]), np.array([4., 53., 112.])],
            [20., 31.], [2., 3.], previous_states=[1, 2])

    def test_mixed_radix_and_qiskit_endian_roundtrip(self):
        index = 17
        states = self.problem.states[index]
        key = encode_onehot(states, self.problem.state_counts)
        decoded = decode_categorical_counts({key: 9}, self.problem.state_counts, 2)
        self.assertEqual(decoded["feasible_shots"], 9)
        self.assertEqual(decoded["feasible_records"][0]["feasible_basis_index"], index)
        self.assertEqual(decoded["feasible_records"][0]["states"], states.tolist())

    def test_invalid_shots_are_retained_not_silently_postselected(self):
        optimum = int(np.argmin(self.problem.energies))
        key = encode_onehot(self.problem.states[optimum], self.problem.state_counts)
        invalid = "0" * self.problem.num_qubits
        result = score_categorical_counts(self.problem, {key: 3, invalid: 7})
        self.assertEqual(result["feasible_fraction"], 0.3)
        self.assertEqual(result["exact_optimum_probability_all_shots"], 0.3)
        self.assertEqual(result["invalid_counts"], {invalid: 7})
        self.assertFalse(result["classical_fallback_used"])
        self.assertFalse(result["raw_counts_are_postselected"])

    def test_all_invalid_shots_produce_missing_prediction_not_classical_fallback(self):
        result = score_categorical_counts(self.problem, {"1" * self.problem.num_qubits: 10})
        self.assertEqual(result["prediction_status"], "no_feasible_samples")
        self.assertIsNone(result["best_states"])
        self.assertIsNone(result["best_feasible_objective"])
        self.assertEqual(result["feasible_shots"], 0)
        self.assertFalse(result["classical_fallback_used"])

    def test_best_state_is_only_selected_from_observed_feasible_records(self):
        index = int(np.argmax(self.problem.energies))
        key = encode_onehot(self.problem.states[index], self.problem.state_counts)
        result = score_categorical_counts(self.problem, {key: 2})
        self.assertEqual(result["best_states"], self.problem.states[index].tolist())
        self.assertEqual(result["best_feasible_objective"], self.problem.energies[index])
        self.assertEqual(result["exact_optimum_shots"], 0)

    def test_malformed_counts_are_rejected(self):
        for counts in ({}, {"010": 1}, {"0" * 10: 0}, {"0" * 10: True},
                       {"0" * 10: 1.2}, {"x" * 10: 1}, {12: 1}):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                decode_categorical_counts(counts, [2, 3], 2)
        for sizes, intervals in (([], 1), ([0, 2], 1), ([True], 1), ([2], 0)):
            with self.assertRaises(ValueError):
                decode_categorical_counts({"10": 1}, sizes, intervals)


@unittest.skipUnless(importlib.util.find_spec("qiskit"), "Optional Qiskit dependency absent")
class CategoricalCircuitTests(unittest.TestCase):
    def test_uniform_onehot_preparation_has_positive_real_amplitudes(self):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        for size in (1, 2, 3, 4):
            circuit = QuantumCircuit(size)
            prepare_uniform_onehot(circuit, range(size))
            actual = Statevector.from_instruction(circuit).data
            expected = np.zeros(1 << size, dtype=complex)
            expected[1 << np.arange(size)] = 1 / np.sqrt(size)
            np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)

    def test_full_physical_circuit_matches_feasible_simulator_p1_p2(self):
        from qiskit.quantum_info import Statevector

        problem = prepare_categorical_problem(
            [105., 213.], [np.array([-1., 101.]), np.array([4., 53., 112.])],
            [20., 31.], [2., 3.], previous_states=[1, 2])
        indices = np.array([int(encode_onehot(states, problem.state_counts), 2)
                            for states in problem.states])
        for gammas, betas in (([0.7], [0.2]), ([1.1, -0.4], [0.3, 0.8])):
            circuit = build_categorical_qaoa_circuit(problem, gammas, betas, measure=False)
            physical = Statevector.from_instruction(circuit).probabilities()
            feasible = categorical_qaoa_probabilities(problem, gammas, betas)
            np.testing.assert_allclose(physical[indices], feasible, rtol=1e-11, atol=1e-13)
            self.assertAlmostEqual(float(physical[indices].sum()), 1.0, places=12)

    def test_measurement_register_preserves_every_logical_bit(self):
        problem = prepare_categorical_problem([5.], [np.array([0., 5., 10.])], 1.)
        circuit = build_categorical_qaoa_circuit(problem, [0.3], [0.2])
        self.assertEqual(circuit.cregs[0].name, "meas")
        pairs = [(circuit.find_bit(item.qubits[0]).index,
                  circuit.find_bit(item.clbits[0]).index)
                 for item in circuit.data if item.operation.name == "measure"]
        self.assertEqual(pairs, [(0, 0), (1, 1), (2, 2)])

    def test_bad_angles_are_rejected(self):
        problem = prepare_categorical_problem([5.], [np.array([0., 5.])], 1.)
        for gammas, betas in (([], []), ([1.], [2., 3.]), ([np.nan], [1.]),
                             ([[1.]], [1.]), ([1.], [np.inf])):
            with self.assertRaises(ValueError):
                build_categorical_qaoa_circuit(problem, gammas, betas)

    def test_parameterized_template_matches_numeric_builder_across_changed_boundaries(self):
        from qiskit import transpile
        from qiskit.quantum_info import Statevector

        levels = [np.array([0., 100.]), np.array([4., 53., 112.])]
        beta, gamma = 0.31, 1.13
        for intervals in (1, 2):
            template = build_categorical_parameterized_template([2, 3], intervals, beta)
            prepared = transpile(template.circuit, basis_gates=["rz", "sx", "x", "cx"],
                                 optimization_level=3, seed_transpiler=7)
            for previous in (None, [1, 2]):
                problem = prepare_categorical_problem([105., 213.][:intervals], levels,
                    [20., 31.], [2., 3.][:intervals], previous_states=previous)
                bound = bind_categorical_template(template, problem, gamma, compiled_circuit=prepared)
                self.assertEqual(bound.num_parameters, 0)
                probabilities = Statevector.from_instruction(bound.remove_final_measurements(inplace=False)).probabilities()
                numeric = build_categorical_qaoa_circuit(problem, [gamma], [beta], measure=False)
                np.testing.assert_allclose(probabilities, Statevector.from_instruction(numeric).probabilities(),
                                           rtol=1e-10, atol=1e-12)
                values = categorical_template_parameter_values(template, problem, gamma,
                                                               parameter_order=tuple(prepared.parameters))
                vector_bound = prepared.assign_parameters(values)
                np.testing.assert_allclose(Statevector.from_instruction(vector_bound.remove_final_measurements(inplace=False)).probabilities(),
                                           probabilities, rtol=1e-12, atol=1e-13)

    def test_parameterized_template_rejects_topology_mismatch(self):
        problem = prepare_categorical_problem([5.], [np.array([0., 5.])], 1.)
        template = build_categorical_parameterized_template([3], 1, 0.2)
        with self.assertRaises(ValueError):
            categorical_template_parameter_values(template, problem, 0.3)
        with self.assertRaises(ValueError):
            build_categorical_parameterized_template([2], 1.5, 0.2)

    def test_qpy_resume_uses_verified_cost_parameter_names(self):
        from io import BytesIO
        from qiskit import qpy, transpile
        from qiskit.quantum_info import Statevector

        original = build_categorical_parameterized_template([2, 3], 1, .3)
        compiled = transpile(original.circuit, basis_gates=["rz", "sx", "x", "cx"], optimization_level=3)
        saved = BytesIO()
        qpy.dump(compiled, saved)
        saved.seek(0)
        restored = qpy.load(saved)[0]
        reconstructed = build_categorical_parameterized_template([2, 3], 1, .3)
        self.assertNotEqual(original.parameters[0], reconstructed.parameters[0])
        problem = prepare_categorical_problem([105.], [np.array([0., 101.]), np.array([4., 53., 112.])],
                                             [20., 31.], [2.], previous_states=[1, 2])
        bound = bind_categorical_template(reconstructed, problem, .7, compiled_circuit=restored)
        expected = build_categorical_qaoa_circuit(problem, [.7], [.3], measure=False)
        np.testing.assert_allclose(Statevector.from_instruction(bound.remove_final_measurements(inplace=False)).probabilities(),
                                   Statevector.from_instruction(expected).probabilities(), atol=1e-12, rtol=1e-10)
        wrong_beta = build_categorical_parameterized_template([2, 3], 1, .4)
        with self.assertRaises(ValueError):
            bind_categorical_template(wrong_beta, problem, .7, compiled_circuit=restored)


if __name__ == "__main__":
    unittest.main()
