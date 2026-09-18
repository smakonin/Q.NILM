import itertools
import unittest
from unittest.mock import patch

import numpy as np

from quantum_nilm.fhmm import FHMMModel, fit_fhmm, predict_fhmm


class FHMMTests(unittest.TestCase):
    @staticmethod
    def training_window(labels=(0, 1, 1, 0), timestamps=(0, 30, 120, 150)):
        dryer = np.asarray(labels, dtype=float) * 10.0
        values = np.column_stack((dryer - 2.0, dryer, np.zeros(len(dryer)), np.zeros(len(dryer))))
        return {"timestamps": np.asarray(timestamps), "values": values}

    @staticmethod
    def fixed_model():
        return FHMMModel(
            levels=[np.array([0.0, 10.0]), np.array([0.0]), np.array([0.0]), np.array([-2.0])],
            transition_costs=[-np.log(np.array([[0.999, 0.001], [0.001, 0.999]])), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))],
            initial_costs=[-np.log(np.array([0.5, 0.5])), np.zeros(1), np.zeros(1), np.zeros(1)],
            noise_variance=10.0,
        )

    def test_training_counts_reset_at_gaps_and_have_laplace_smoothing(self):
        levels = [[0, 10], [0], [0], [-2]]
        model = fit_fhmm([self.training_window()], levels)
        # Only 0->1 and 1->0 are observed. The gap must not add 1->1.
        np.testing.assert_allclose(np.exp(-model.transition_costs[0]), [[1 / 3, 2 / 3], [2 / 3, 1 / 3]])
        np.testing.assert_allclose(np.exp(-model.initial_costs[0]), [0.5, 0.5])
        self.assertEqual(model.training_blocks, 4)
        self.assertEqual(model.training_runs, 2)
        self.assertEqual(model.noise_variance, 1.0)
        np.testing.assert_array_equal(model.levels[-1], [-2.0])

    def test_window_boundaries_restart_initial_counts(self):
        first = self.training_window((0, 0), (0, 30))
        second = self.training_window((1, 1), (60, 90))
        model = fit_fhmm([first, second], [[0, 10], [0], [0], [-2]])
        np.testing.assert_allclose(np.exp(-model.transition_costs[0]), [[2 / 3, 1 / 3], [1 / 3, 2 / 3]])
        self.assertEqual(model.training_runs, 2)

    def test_prediction_resets_initial_prior_at_gaps(self):
        model = self.fixed_model()
        independent, info = predict_fhmm([-2.0, 8.0], [0, 120], model)
        connected, connected_info = predict_fhmm([-2.0, 8.0], [0, 30], model)
        np.testing.assert_array_equal(independent[:, 0], [0.0, 10.0])
        np.testing.assert_array_equal(connected[:, 0], [0.0, 0.0])
        self.assertEqual(info["valid_runs"], 2)
        self.assertEqual(connected_info["valid_runs"], 1)
        self.assertEqual(info["segments"], 2)

    def test_prediction_matches_exhaustive_joint_map(self):
        rng = np.random.default_rng(401)
        for repetition in range(5):
            levels = [np.array([-1.0, 6.0]), np.array([0.0, 3.0]), np.array([1.0]), np.array([-4.0, 2.0])]
            transition_probabilities = [rng.uniform(0.1, 1.0, (len(values), len(values))) for values in levels]
            transitions = [-np.log(p / p.sum(axis=1, keepdims=True)) for p in transition_probabilities]
            initial_probabilities = [rng.uniform(0.1, 1.0, len(values)) for values in levels]
            initials = [-np.log(p / p.sum()) for p in initial_probabilities]
            model = FHMMModel(levels, transitions, initials, 2.0)
            signal = rng.uniform(-5.0, 12.0, 3)
            possible = list(itertools.product(*(range(len(values)) for values in levels)))
            optimum, expected = float("inf"), None
            for path in itertools.product(possible, repeat=3):
                states = np.asarray(path)
                prediction = np.column_stack([values[states[:, i]] for i, values in enumerate(levels)])
                energy = np.sum((signal - prediction.sum(axis=1)) ** 2) / 4.0
                energy += sum(costs[states[0, i]] for i, costs in enumerate(initials))
                energy += sum(np.sum(costs[states[:-1, i], states[1:, i]]) for i, costs in enumerate(transitions))
                if energy < optimum:
                    optimum, expected = energy, prediction
            with self.subTest(repetition=repetition):
                observed, info = predict_fhmm(signal, [0, 30, 60], model)
                np.testing.assert_array_equal(observed, expected)
                self.assertAlmostEqual(info["objective_energy"], optimum, places=10)

    def test_omit_background_is_an_explicit_three_chain_ablation(self):
        prediction, info = predict_fhmm([-2.0], [0], self.fixed_model(), include_background=False)
        self.assertEqual(prediction.shape, (1, 3))
        self.assertFalse(info["include_background"])
        self.assertEqual(info["joint_states_per_segment"], 2)

    def test_noise_variance_is_train_emission_residual_mean_square(self):
        window = self.training_window((0, 0), (0, 30))
        window["values"][:, 0] = [0, 4]
        model = fit_fhmm([window], [[0], [0], [0], [0]])
        self.assertEqual(model.noise_variance, 8.0)

    def test_fit_is_training_only_and_does_not_alias_supplied_levels(self):
        levels = [np.array([0.0, 10.0]), np.array([0.0]), np.array([0.0]), np.array([-2.0])]
        train = self.training_window()
        model = fit_fhmm([train], levels)
        before, _ = predict_fhmm([-2.0, 8.0], [0, 30], model)
        levels[0][1] = 9999.0
        train["values"][:] = 12345.0
        after, _ = predict_fhmm([-2.0, 8.0], [0, 30], model)
        np.testing.assert_array_equal(before, after)
        with self.assertRaises(TypeError):
            predict_fhmm([-2.0], [0], model, test_labels=[0])

    def test_empty_prediction_has_defined_shape_and_summary(self):
        prediction, info = predict_fhmm([], [], self.fixed_model())
        self.assertEqual(prediction.shape, (0, 4))
        self.assertEqual(info["objective_energy"], 0.0)
        self.assertEqual(info["blocks"], 0)

    def test_input_and_resource_validation(self):
        levels = [[0, 10], [0], [0], [-2]]
        for windows in ([], [{}], [{"timestamps": [0], "values": [[1, 2, 3]]}], [{"timestamps": [0, 0], "values": np.ones((2, 4))}]):
            with self.subTest(windows=windows), self.assertRaises(ValueError):
                fit_fhmm(windows, levels)
        with self.assertRaises(ValueError):
            fit_fhmm([self.training_window()], [[0]])
        model = self.fixed_model()
        for aggregate, times in (([np.nan], [0]), ([1], [np.nan]), ([1, 2], [30, 0]), ([1], [])):
            with self.subTest(aggregate=aggregate, times=times), self.assertRaises(ValueError):
                predict_fhmm(aggregate, times, model)
        oversized = [np.zeros(9)] * 4
        with self.assertRaisesRegex(ValueError, "4096 joint states"):
            fit_fhmm([self.training_window()], oversized)
        with patch("quantum_nilm.fhmm.MAX_TRACEBACK_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "Traceback needs"):
                predict_fhmm([0, 0], [0, 30], model)


if __name__ == "__main__":
    unittest.main()
