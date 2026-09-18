import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from quantum_nilm.categorical_qaoa import (
    categorical_qaoa_probabilities,
    categorical_qaoa_statevector,
    choose_sampled,
    decode_onehot,
    encode_onehot,
    prepare_categorical_problem,
    sample_categorical_qaoa,
)


class CategoricalQAOATests(unittest.TestCase):
    @staticmethod
    def problem():
        return prepare_categorical_problem(
            [2.5, 6.0], [[0.0, 4.0], [-2.0, 1.0, 3.0]],
            [1.0, 2.0], [2.0, 1.0], previous_states=[1, 0],
        )

    def test_direct_cost_and_onehot_qubo_agree_for_every_feasible_state(self):
        problem = self.problem()
        for index, states in enumerate(problem.states):
            power = np.column_stack([values[states[:, i]] for i, values in enumerate(problem.levels)])
            direct = np.sum(problem.segment_weights * (problem.aggregate - power.sum(axis=1)) ** 2)
            direct += np.sum(problem.switch_penalty * (states[1:] != states[:-1]))
            direct += np.sum(problem.switch_penalty * (states[0] != problem.previous_states))
            bits = np.asarray([int(bit) for bit in encode_onehot(states, problem.state_counts)[::-1]])
            qubo = problem.constant + problem.linear @ bits
            qubo += sum(value * bits[a] * bits[b] for (a, b), value in problem.quadratic.items())
            self.assertAlmostEqual(problem.energies[index], direct)
            self.assertAlmostEqual(qubo, direct)
        self.assertEqual(problem.scale, max(1.0, np.abs(problem.linear).sum() + sum(abs(value) for value in problem.quadratic.values())))

    def test_encoding_order_and_round_trip(self):
        self.assertEqual(encode_onehot([[1, 0]], [2, 3]), "00110")
        np.testing.assert_array_equal(decode_onehot("001 10", [2, 3], 1), [[1, 0]])
        problem = self.problem()
        for states in problem.states:
            np.testing.assert_array_equal(decode_onehot(encode_onehot(states, [2, 3]), [2, 3], 2), states)
        # The first register is the least significant categorical digit.
        np.testing.assert_array_equal(problem.states[0], [[0, 0], [0, 0]])
        np.testing.assert_array_equal(problem.states[1], [[1, 0], [0, 0]])
        np.testing.assert_array_equal(problem.states[2], [[0, 1], [0, 0]])

    def test_infeasible_measured_registers_are_rejected_without_repair(self):
        for bitstring in ["00000", "00111", "10110", "010", "0012x"]:
            with self.subTest(bitstring=bitstring), self.assertRaises(ValueError):
                decode_onehot(bitstring, [2, 3], 1)
        for states in ([[2, 0]], [[-1, 0]], [[1.0, 0.0]], [1, 0]):
            with self.subTest(states=states), self.assertRaises(ValueError):
                encode_onehot(states, [2, 3])

    def test_normalization_uniform_initialization_and_nontrivial_mixer(self):
        problem = self.problem()
        for gammas, betas in [([0.0], [0.0]), ([2.3], [0.47]), ([0.7, 1.1], [-0.9, 0.3])]:
            state = categorical_qaoa_statevector(problem, gammas, betas)
            self.assertAlmostEqual(float(np.abs(state @ state.conj())), 1.0, places=12)
        uniform = categorical_qaoa_probabilities(problem, 1.7, 0.0)
        np.testing.assert_allclose(uniform, np.ones(problem.num_feasible_states) / problem.num_feasible_states)
        mixed = categorical_qaoa_probabilities(problem, 2.3, 0.47)
        self.assertGreater(np.max(np.abs(mixed - uniform)), 1e-3)

    def test_categorical_xy_chain_matches_independent_dense_pair_matrices(self):
        problem = prepare_categorical_problem([3.0], [[-1.0, 2.0, 8.0]], 0.0)
        gamma, beta = 1.4, 0.31
        expected = np.exp(-1j * gamma * problem.energies / problem.scale) / np.sqrt(3)
        for a, b in [(0, 1), (1, 2)]:
            unitary = np.eye(3, dtype=complex)
            unitary[a, a] = unitary[b, b] = np.cos(beta)
            unitary[a, b] = unitary[b, a] = -1j * np.sin(beta)
            expected = unitary @ expected
        np.testing.assert_allclose(categorical_qaoa_statevector(problem, gamma, beta), expected, atol=1e-14)

    @unittest.skipUnless(importlib.util.find_spec("qiskit"), "optional Qiskit is not installed")
    def test_matches_independent_full_qubit_qiskit_circuit(self):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        problem = self.problem()
        circuit = QuantumCircuit(problem.num_qubits)
        # The physical initialization is built using controlled rotations,
        # not by inserting the feasible-space simulator's vector.
        for start, count in zip(problem.register_offsets, problem.register_sizes):
            circuit.x(start)
            for category in range(count - 1):
                theta = 2 * np.arccos(1 / np.sqrt(count - category))
                circuit.cry(theta, start + category, start + category + 1)
                circuit.cx(start + category + 1, start + category)
        gammas, betas = [1.7, -0.4], [0.32, 0.7]
        for gamma, beta in zip(gammas, betas):
            # P(phi) gives exp(i phi*x), CP(phi) exp(i phi*x*y).
            # This omits the global phase from the scalar constant only.
            for qubit, coefficient in enumerate(problem.linear):
                circuit.p(-gamma * coefficient / problem.scale, qubit)
            for (left, right), coefficient in problem.quadratic.items():
                circuit.cp(-gamma * coefficient / problem.scale, left, right)
            for start, count in zip(problem.register_offsets, problem.register_sizes):
                for category in range(count - 1):
                    circuit.rxx(beta, start + category, start + category + 1)
                    circuit.ryy(beta, start + category, start + category + 1)
        physical = Statevector.from_instruction(circuit).data
        feasible_ids = np.asarray([int(encode_onehot(state, problem.state_counts), 2) for state in problem.states])
        observed = physical[feasible_ids]
        expected = categorical_qaoa_statevector(problem, gammas, betas)
        expected *= np.exp(1j * np.sum(gammas) * problem.constant / problem.scale)
        np.testing.assert_allclose(observed, expected, atol=1e-12)
        self.assertAlmostEqual(float(np.abs(observed).dot(np.abs(observed))), 1.0, places=12)
        infeasible = np.ones(len(physical), dtype=bool)
        infeasible[feasible_ids] = False
        self.assertLess(float(np.sum(np.abs(physical[infeasible]) ** 2)), 1e-24)

    def test_sampling_selects_only_observed_indices_without_exact_fallback(self):
        problem = self.problem()
        with patch("quantum_nilm.categorical_qaoa.np.random.default_rng") as factory:
            factory.return_value.choice.return_value = np.array([0, 1, 0], dtype=int)
            result = sample_categorical_qaoa(problem, 1.4, 0.3, shots=3, seed=2)
        self.assertIn(result.best_index, [0, 1])
        self.assertEqual(result.best_energy, float(min(problem.energies[:2])))
        self.assertEqual(result.shots, 3)
        self.assertEqual(int(result.counts.sum()), 3)
        self.assertEqual(result.feasible_fraction, 1.0)
        first = sample_categorical_qaoa(problem, 1.4, 0.3, shots=50, seed=93)
        second = sample_categorical_qaoa(problem, 1.4, 0.3, shots=50, seed=93)
        np.testing.assert_array_equal(first.indices, second.indices)

    def test_target_problem_has_24_qubits_and_5184_feasible_states(self):
        problem = prepare_categorical_problem([500.0, 800.0], [[0, 100, 500, 1000], [0, 100, 300], [0, 900], [-50, 20, 300]], [1, 2, 3, 4])
        self.assertEqual(problem.num_qubits, 24)
        self.assertEqual(problem.num_feasible_states, 5184)
        self.assertEqual(problem.states.shape, (5184, 2, 4))
        probabilities = categorical_qaoa_probabilities(problem, 1.3, 0.8)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=12)

    def test_common_sample_selection_rejects_invalid_indices(self):
        problem = self.problem()
        result = choose_sampled(problem, [0, 2, 0, 2])
        self.assertIn(result.best_index, [0, 2])
        self.assertEqual(result.shots, 4)
        for invalid in ([], [-1], [problem.num_feasible_states], [0.0], [[0]], [True]):
            with self.subTest(indices=invalid), self.assertRaises(ValueError):
                choose_sampled(problem, invalid)

    def test_single_category_and_signed_background(self):
        problem = prepare_categorical_problem([-2.0], [[-2.0]], 1.0, previous_states=[0])
        self.assertEqual(problem.num_qubits, 1)
        self.assertEqual(problem.num_feasible_states, 1)
        np.testing.assert_array_equal(problem.energies, [0.0])
        np.testing.assert_array_equal(categorical_qaoa_probabilities(problem, 1.0, 0.3), [1.0])

    def test_validation_and_resource_caps(self):
        invalid = [
            ([], [[0]], 0, None, None), ([1, 2, 3], [[0]], 0, None, None),
            ([np.nan], [[0]], 0, None, None), ([1], [], 0, None, None),
            ([1], [[np.inf]], 0, None, None), ([1], [[0]], -1, None, None),
            ([1], [[0]], [0, 0], None, None), ([1], [[0]], 0, [0], None),
            ([1], [[0]], 0, [1, 1], None), ([1], [[0, 1]], 0, None, [2]),
            ([1], [[0, 1]], 0, None, [0.0]), ([1], [[0, 1]], 0, None, [[0]]),
        ]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                prepare_categorical_problem(*args)
        with self.assertRaisesRegex(ValueError, "feasible states"):
            prepare_categorical_problem([0, 0], [[0, 1]] * 11, 0)
        with self.assertRaisesRegex(ValueError, "physical qubits"):
            prepare_categorical_problem([0], [np.arange(257)], 0)
        with self.assertRaisesRegex(ValueError, "floating-point range"):
            prepare_categorical_problem([1e300], [[0, 1]], 0)
        problem = self.problem()
        for gammas, betas in [([], []), ([1, 2], [1]), ([np.nan], [1]), ([[1]], [1])]:
            with self.subTest(gammas=gammas, betas=betas), self.assertRaises(ValueError):
                categorical_qaoa_probabilities(problem, gammas, betas)
        for shots in [0, -1, True, 1.5]:
            with self.subTest(shots=shots), self.assertRaises(ValueError):
                sample_categorical_qaoa(problem, 1, 1, shots)


if __name__ == "__main__":
    unittest.main()
