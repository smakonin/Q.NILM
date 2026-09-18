import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from quantum_nilm.categorical_qaoa import prepare_categorical_problem
from quantum_nilm.stage_b import make_stage_b_instance
from quantum_nilm.stage_d_mixer import (
    audit_penalty_problem, build_penalty_mixer_circuit, compile_and_audit,
    optimize_penalty_mixer, penalty_mixer_probabilities, prepare_penalty_problem,
)


class StageDPenaltyTests(unittest.TestCase):
    def test_barrier_and_augmented_qubo_over_all_binary_strings(self):
        base = prepare_categorical_problem([2.0, -3.0], [[-2.0, 1.0, 4.0], [0.0, 5.0]], [1.3, 2.1], [2, 3])
        problem = prepare_penalty_problem(base)
        audit = audit_penalty_problem(problem)
        self.assertEqual(problem.barrier, 1 + 2 * (np.abs(base.linear).sum() + sum(abs(value) for value in base.quadratic.values())))
        self.assertGreater(audit["invalid_minus_worst_feasible"], 0)
        np.testing.assert_allclose(problem.energies[problem.feasible_indices], base.energies, atol=1e-9)
        for bits, energy in zip(problem.bits, problem.energies):
            direct = problem.constant + np.dot(problem.linear, bits)
            direct += sum(value * bits[a] * bits[b] for (a, b), value in problem.quadratic.items())
            self.assertAlmostEqual(direct, energy, places=8)
        self.assertEqual(problem.scale, max(1.0, np.abs(problem.linear).sum() + sum(abs(value) for value in problem.quadratic.values())))
        self.assertGreater(problem.scale, base.scale)

    def test_initializations_and_x_leakage_are_unconditional(self):
        base = make_stage_b_instance((3, 2), 1, 99999, 0.01).problem
        problem = prepare_penalty_problem(base)
        h = penalty_mixer_probabilities(problem, [0], [0], "h")
        w = penalty_mixer_probabilities(problem, [0], [0], "w")
        np.testing.assert_allclose(h, np.full(32, 1 / 32))
        self.assertAlmostEqual(h[problem.feasible_indices].sum(), 6 / 32)
        self.assertAlmostEqual(w[problem.feasible_indices].sum(), 1)
        mixed = penalty_mixer_probabilities(problem, [1.3], [0.31], "w")
        self.assertLess(mixed[problem.feasible_indices].sum(), 0.99)
        self.assertAlmostEqual(mixed.sum(), 1.0)

    def test_width_and_angle_guards(self):
        with self.assertRaisesRegex(ValueError, "12 qubits"):
            prepare_penalty_problem(SimpleNamespace(num_qubits=13))
        problem = prepare_penalty_problem(make_stage_b_instance((2, 2), 1, 99999, 0).problem)
        for gamma, beta, initialization in (([], [], "w"), ([1], [1, 2], "h"), ([np.nan], [1], "w"), ([1], [1], "unknown")):
            with self.assertRaises(ValueError):
                penalty_mixer_probabilities(problem, gamma, beta, initialization)

    @unittest.skipUnless(importlib.util.find_spec("qiskit"), "optional Qiskit not installed")
    def test_full_qiskit_circuits_match_numpy_for_both_depths_and_initializations(self):
        for counts in ((2, 2), (3, 2)):
            problem = prepare_penalty_problem(make_stage_b_instance(counts, 1, 99999, 0.01).problem)
            for depth in (1, 2):
                gammas, betas = [1.37, 2.11][:depth], [0.29, 0.61][:depth]
                for initialization in ("h", "w"):
                    with self.subTest(counts=counts, depth=depth, initialization=initialization):
                        expected = penalty_mixer_probabilities(problem, gammas, betas, initialization)
                        circuit = build_penalty_mixer_circuit(problem, gammas, betas, initialization)
                        audit = compile_and_audit(circuit, expected)
                        self.assertLess(audit["audit_max_probability_error"], 1e-12)
                        self.assertEqual(audit["num_qubits"], sum(counts))


@unittest.skipUnless(importlib.util.find_spec("scipy"), "optional SciPy not installed")
class StageDOptimizerTests(unittest.TestCase):
    def test_fixed_budget_and_archived_expectation_at_both_depths(self):
        problem = prepare_penalty_problem(make_stage_b_instance((2, 2), 1, 99999, 0.01).problem)
        for initialization in ("h", "w"):
            for depth in (1, 2):
                with self.subTest(initialization=initialization, depth=depth):
                    with patch("quantum_nilm.stage_d_mixer.penalty_mixer_probabilities", wraps=penalty_mixer_probabilities) as simulate:
                        result = optimize_penalty_mixer(problem, depth, initialization)
                    self.assertEqual(result["evaluations"], 1024)
                    self.assertEqual(simulate.call_count, 1024)
                    self.assertAlmostEqual(np.dot(result["probabilities"], problem.energies), result["expected_cost"])
                    initial = penalty_mixer_probabilities(problem, [0] * depth, [0] * depth, initialization)
                    self.assertLessEqual(result["expected_cost"], initial @ problem.energies + 1e-8)
                    for restart in result["restarts"]:
                        self.assertEqual(restart["evaluations"], 512)
                        self.assertEqual(len(restart["objective_history"]), 512)
                        self.assertEqual(restart["normalized_expected_cost"], min(restart["objective_history"]))
                    self.assertFalse(result["optimizer"]["exact_optimum_used_in_training"])
                    self.assertFalse(result["optimizer"]["planted_truth_used_in_training"])

    def test_reproducibility_and_no_label_or_optimal_state_access(self):
        original = prepare_penalty_problem(make_stage_b_instance((2, 2), 1, 99999, 0.01).problem)
        objective_only = SimpleNamespace(base=SimpleNamespace(num_qubits=4), energies=original.energies,
                                         scale=original.scale, feasible_indices=original.feasible_indices)
        first = optimize_penalty_mixer(objective_only, 2, "w", (99999,), 96)
        second = optimize_penalty_mixer(objective_only, 2, "w", (99999,), 96)
        self.assertEqual(first["probabilities"], second["probabilities"])
        self.assertEqual(first["restarts"][0]["objective_history"], second["restarts"][0]["objective_history"])

    def test_boundary_roundoff_and_early_convergence_padding(self):
        problem = prepare_penalty_problem(make_stage_b_instance((2, 2), 1, 99999, 0).problem)

        def probe(objective, start, **kwargs):
            values = np.asarray(start).copy()
            values[0] = -4.440892098500626e-16
            objective(values)
            return SimpleNamespace(nfev=1, success=True, status=0, message="test boundary")

        with patch("scipy.optimize.minimize", side_effect=probe):
            result = optimize_penalty_mixer(problem, 1, "w", (99999,))
        self.assertEqual(result["evaluations"], 512)
        self.assertEqual(result["restarts"][0]["padding_evaluations"], 478)
        self.assertEqual(result["restarts"][0]["boundary_roundoff_evaluations"], 2)

        def violation(objective, start, **kwargs):
            values = np.asarray(start).copy()
            values[0] = -1e-8
            return objective(values)

        with patch("scipy.optimize.minimize", side_effect=violation):
            with self.assertRaisesRegex(ValueError, "out-of-bounds"):
                optimize_penalty_mixer(problem, 1, "w", (99999,))

    def test_optimizer_configuration_guards(self):
        problem = prepare_penalty_problem(make_stage_b_instance((2, 2), 1, 99999, 0).problem)
        for depth, initialization, seeds, budget in ((0, "w", (1,), 512), (1, "a", (1,), 512), (1, "w", (), 512), (1, "w", (1, 1), 512), (1, "w", (1,), 33)):
            with self.assertRaises(ValueError):
                optimize_penalty_mixer(problem, depth, initialization, seeds, budget)


if __name__ == "__main__":
    unittest.main()
