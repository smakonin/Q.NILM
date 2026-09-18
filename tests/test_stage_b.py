import importlib.util
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np

from quantum_nilm.categorical_qaoa import categorical_qaoa_probabilities
from quantum_nilm.stage_b import make_stage_b_instance, optimize_stage_b_qaoa


class StageBGeneratorTests(unittest.TestCase):
    def test_noise_variants_share_every_base_draw(self):
        clean = make_stage_b_instance((4, 3, 2, 3), 2, 99999, 0.0)
        noisy = make_stage_b_instance((4, 3, 2, 3), 2, 99999, 0.05)
        np.testing.assert_array_equal(clean.truth, noisy.truth)
        for field in ("levels_watts", "channel_maxima_watts", "truth", "standard_normal_noise", "clean_aggregate_watts", "segment_weights", "switch_penalty"):
            self.assertEqual(clean.metadata[field], noisy.metadata[field])
        np.testing.assert_allclose(
            noisy.problem.aggregate - clean.problem.aggregate,
            0.05 * clean.metadata["noise_reference_maximum_watts"] * np.asarray(clean.metadata["standard_normal_noise"]),
        )
        self.assertIsNone(noisy.problem.previous_states)
        self.assertEqual(noisy.problem.num_qubits, 24)
        self.assertEqual(noisy.problem.num_feasible_states, 5184)
        json.dumps(noisy.metadata, allow_nan=False)

    def test_generator_objective_matches_direct_planted_truth_cost(self):
        instance = make_stage_b_instance((3, 2), 2, 99999, 0.05)
        problem = instance.problem
        truth_index = np.flatnonzero(np.all(problem.states == instance.truth, axis=(1, 2)))
        self.assertEqual(len(truth_index), 1)
        direct = np.sum(problem.segment_weights * np.asarray(instance.metadata["noise_watts"])**2)
        direct += np.sum(problem.switch_penalty * (instance.truth[0] != instance.truth[1]))
        self.assertAlmostEqual(problem.energies[truth_index[0]], direct)
        np.testing.assert_allclose(problem.switch_penalty, 0.02 * np.asarray(instance.metadata["channel_maxima_watts"])**2)

    def test_negative_measurements_are_not_clipped(self):
        # Only generator invariants are examined here; no scored matrix seed.
        rng = Mock(wraps=np.random.default_rng(99999))
        rng.standard_normal.return_value = -np.ones(2)
        with patch("quantum_nilm.stage_b.np.random.default_rng", return_value=rng):
            instance = make_stage_b_instance((2, 2), 2, 99999, 100.0)
        expected = np.asarray(instance.metadata["clean_aggregate_watts"]) + np.asarray(instance.metadata["noise_watts"])
        np.testing.assert_array_equal(instance.problem.aggregate, expected)
        self.assertTrue(np.any(instance.problem.aggregate < 0))
        self.assertFalse(instance.metadata["clipped"])

    def test_input_validation(self):
        for counts, segments, seed, sigma in (
            ((), 1, 1, 0), ((1, 2), 1, 1, 0), ((2.0, 2), 1, 1, 0),
            ((True, 2), 1, 1, 0), ((2,) * 5, 1, 1, 0), ((2, 2), 3, 1, 0),
            ((2, 2), 1, -1, 0), ((2, 2), 1, 1, -0.1), ((2, 2), 1, 1, np.nan),
        ):
            with self.subTest(counts=counts, segments=segments, seed=seed, sigma=sigma):
                with self.assertRaises(ValueError):
                    make_stage_b_instance(counts, segments, seed, sigma)


@unittest.skipUnless(importlib.util.find_spec("scipy"), "optional SciPy is not installed")
class StageBOptimizerTests(unittest.TestCase):
    def test_fixed_call_counts_and_candidate_selection_at_both_depths(self):
        problem = make_stage_b_instance((2, 2), 1, 99999, 0.01).problem
        for depth in (1, 2):
            with self.subTest(depth=depth):
                with patch("quantum_nilm.stage_b.categorical_qaoa_probabilities", wraps=categorical_qaoa_probabilities) as simulator:
                    result = optimize_stage_b_qaoa(problem, depth)
                self.assertEqual(result["evaluations"], 1024)
                self.assertEqual(simulator.call_count, 1024)
                self.assertEqual(len(result["gammas"]), depth)
                self.assertEqual(len(result["betas"]), depth)
                self.assertLessEqual(result["expected_cost"], np.mean(problem.energies) + 1e-8)
                self.assertAlmostEqual(sum(result["probabilities"]), 1.0, places=12)
                np.testing.assert_allclose(result["probabilities"], categorical_qaoa_probabilities(problem, result["gammas"], result["betas"]))
                self.assertAlmostEqual(result["expected_cost"], np.dot(result["probabilities"], problem.energies))
                self.assertEqual(result["normalized_expected_cost"], min(record["normalized_expected_cost"] for record in result["restarts"]))
                self.assertFalse(result["optimizer"]["warm_start_from_lower_depth"])
                self.assertFalse(result["optimizer"]["exact_optimum_used_in_training"])
                self.assertFalse(result["optimizer"]["planted_truth_used_in_training"])
                for record in result["restarts"]:
                    self.assertEqual(record["evaluations"], 512)
                    self.assertEqual(len(record["objective_history"]), 512)
                    self.assertEqual(record["normalized_expected_cost"], min(record["objective_history"]))
                    self.assertEqual(record["initial_evaluations"] + sum(item["evaluations"] + item["padding_evaluations"] for item in record["refinements"]), 512)
                    self.assertEqual(record["normalized_expected_cost"], record["objective_history"][record["best_evaluation_index"]])
                json.dumps(result, allow_nan=False)

    def test_reproducible_angles_and_histories(self):
        problem = make_stage_b_instance((3, 2), 2, 99999, 0.01).problem
        first = optimize_stage_b_qaoa(problem, 2, restart_seeds=(99999,), evaluations_per_restart=96)
        second = optimize_stage_b_qaoa(problem, 2, restart_seeds=(99999,), evaluations_per_restart=96)
        for field in ("gammas", "betas", "probabilities", "expected_cost", "normalized_expected_cost"):
            self.assertEqual(first[field], second[field])
        self.assertEqual(first["restarts"][0]["objective_history"], second["restarts"][0]["objective_history"])

    def test_early_convergence_calls_are_padded_to_budget(self):
        problem = make_stage_b_instance((2, 2), 1, 99999, 0.0).problem
        from types import SimpleNamespace

        def immediate_convergence(objective, x0, **kwargs):
            objective(x0)
            return SimpleNamespace(nfev=1, success=True, status=0, message="test convergence")

        with patch("scipy.optimize.minimize", side_effect=immediate_convergence):
            result = optimize_stage_b_qaoa(problem, 1, restart_seeds=(99999,))
        record = result["restarts"][0]
        self.assertEqual(record["evaluations"], 512)
        self.assertEqual(record["padding_evaluations"], 478)

    def test_training_does_not_access_planted_truth_or_optimal_basis(self):
        from types import SimpleNamespace
        original = make_stage_b_instance((2, 2), 1, 99999, 0.01).problem
        # The optimizer receives an objective-compatible object with no states
        # or truth attribute. The circuit simulator likewise needs neither.
        objective_only = SimpleNamespace(
            energies=original.energies,
            scale=original.scale,
            num_feasible_states=original.num_feasible_states,
            register_sizes=original.register_sizes,
        )
        result = optimize_stage_b_qaoa(objective_only, 2, restart_seeds=(99999,), evaluations_per_restart=34)
        self.assertEqual(result["evaluations"], 34)

    def test_invalid_optimizer_configuration(self):
        problem = make_stage_b_instance((2, 2), 1, 99999, 0.0).problem
        for depth, seeds, budget in ((0, (1,), 512), (True, (1,), 512), (1, (), 512), (1, (1, 1), 512), (1, (-1,), 512), (1, (1,), 33)):
            with self.subTest(depth=depth, seeds=seeds, budget=budget):
                with self.assertRaises(ValueError):
                    optimize_stage_b_qaoa(problem, depth, seeds, budget)

    def test_powell_boundary_roundoff_regression_preserves_fixed_budget(self):
        # Run 001 stopped at this input after SciPy 1.18.1 Powell formed
        # gamma=-4.440892098500626e-16. This is an error regression only, not
        # selection of a favourable scored outcome or a changed search budget.
        problem = make_stage_b_instance((2, 2), 1, 1006, 0.01).problem
        result = optimize_stage_b_qaoa(problem, 2)
        self.assertEqual(result["evaluations"], 1024)
        for record in result["restarts"]:
            self.assertEqual(record["evaluations"], 512)
            self.assertGreaterEqual(record["boundary_roundoff_evaluations"], 0)
            self.assertLessEqual(record["boundary_max_correction"], max(result["optimizer"]["boundary_roundoff_tolerance"]))
        self.assertTrue(all(value >= 0 for value in result["gammas"] + result["betas"]))

    def test_only_roundoff_is_snapped_and_corrections_are_archived(self):
        from types import SimpleNamespace
        problem = make_stage_b_instance((2, 2), 1, 99999, 0.0).problem

        def roundoff_probe(objective, x0, **kwargs):
            candidate = np.asarray(x0).copy()
            candidate[0] = -4.440892098500626e-16
            objective(candidate)
            return SimpleNamespace(nfev=1, success=True, status=0, message="boundary regression")

        with patch("scipy.optimize.minimize", side_effect=roundoff_probe):
            with patch("quantum_nilm.stage_b.categorical_qaoa_probabilities", wraps=categorical_qaoa_probabilities) as simulator:
                result = optimize_stage_b_qaoa(problem, 1, restart_seeds=(99999,))
        record = result["restarts"][0]
        self.assertEqual(record["evaluations"], 512)
        self.assertEqual(simulator.call_count, 512)
        self.assertEqual(record["boundary_roundoff_evaluations"], 2)
        self.assertEqual(record["boundary_roundoff_coordinates"], 2)
        self.assertEqual(record["boundary_max_correction"], 4.440892098500626e-16)
        self.assertEqual(simulator.call_args_list[32].args[1][0], 0.0)

        def material_violation(objective, x0, **kwargs):
            candidate = np.asarray(x0).copy()
            candidate[0] = -1e-8
            return objective(candidate)

        with patch("scipy.optimize.minimize", side_effect=material_violation):
            with self.assertRaisesRegex(ValueError, "out-of-bounds"):
                optimize_stage_b_qaoa(problem, 1, restart_seeds=(99999,))


if __name__ == "__main__":
    unittest.main()
