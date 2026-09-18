import itertools
import unittest
from unittest.mock import patch

import numpy as np

from quantum_nilm.classical import solve_binary_temporal_exact
from quantum_nilm.multistate import fit_power_levels, solve_multistate_temporal_exact


class PowerLevelFitTests(unittest.TestCase):
    def test_fit_is_deterministic_sorted_and_uses_only_supplied_training_values(self):
        train = np.array([-1.0, 0.0, 1.0, 9.0, 10.0, 11.0, 29.0, 30.0, 31.0])
        unchanged = train.copy()
        first = fit_power_levels(train, 3)
        second = fit_power_levels(train, 3)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, [0.0, 10.0, 30.0], atol=1e-12)
        np.testing.assert_array_equal(train, unchanged)
        # A future outlier is not available to the fit; only explicitly
        # contaminating the supplied training array can change the centres.
        contaminated = fit_power_levels(np.append(train, 1000.0), 3)
        self.assertFalse(np.array_equal(first, contaminated))
        np.testing.assert_array_equal(fit_power_levels(train, 3), first)

    def test_constant_and_low_diversity_training_have_distinct_levels(self):
        np.testing.assert_array_equal(fit_power_levels([4.0] * 10, 3), [4.0])
        np.testing.assert_allclose(fit_power_levels([0.0] * 20 + [10.0], 3), [0.0, 10.0])
        np.testing.assert_allclose(fit_power_levels([-4.0, 4.0], 1), [0.0])

    def test_duplicated_quantiles_are_seeded_deterministically(self):
        values = np.array([0.0] * 90 + [10.0] * 5 + [20.0] * 3 + [30.0] * 2)
        np.testing.assert_allclose(fit_power_levels(values, 4), [0.0, 10.0, 20.0, 30.0])

    def test_large_finite_values_do_not_overflow_fit(self):
        np.testing.assert_allclose(fit_power_levels([-1e300, -1e300, 1e300, 1e300], 2), [-1e300, 1e300])

    def test_fit_rejects_invalid_inputs(self):
        cases = [([], 1), ([[1.0]], 1), ([np.nan], 1), ([np.inf], 1), ([1.0], 2), ([1.0], 0), ([1.0], -1), ([1.0], 1.5), ([1.0], True)]
        for values, count in cases:
            with self.subTest(values=values, count=count), self.assertRaises(ValueError):
                fit_power_levels(values, count)


class MultistateTemporalExactTests(unittest.TestCase):
    @staticmethod
    def objective(aggregate, levels, penalties, weights, states):
        predicted = np.column_stack([values[states[:, i]] for i, values in enumerate(levels)])
        return float(np.sum(weights * (aggregate - predicted.sum(axis=1)) ** 2) + np.sum(penalties * (states[1:] != states[:-1])))

    def test_matches_exhaustive_tiny_random_instances(self):
        rng = np.random.default_rng(73)
        for counts, segments in [([3], 3), ([2, 3], 3), ([1, 2, 3], 2), ([3, 3], 2)]:
            for repeat in range(4):
                with self.subTest(counts=counts, segments=segments, repeat=repeat):
                    levels = [np.sort(rng.uniform(-3.0, 7.0, count)) for count in counts]
                    aggregate = rng.uniform(-5.0, 20.0, segments)
                    penalties = rng.uniform(0.0, 10.0, len(counts))
                    penalties[repeat % len(counts)] = 0.0
                    weights = rng.uniform(0.2, 3.0, segments)
                    result = solve_multistate_temporal_exact(aggregate, levels, penalties, weights)
                    joint = list(itertools.product(*(range(count) for count in counts)))
                    optimum = min(self.objective(aggregate, levels, penalties, weights, np.asarray(path)) for path in itertools.product(joint, repeat=segments))
                    self.assertAlmostEqual(result.energy, optimum, places=8)
                    self.assertAlmostEqual(result.energy, self.objective(aggregate, levels, penalties, weights, result.states), places=10)
                    self.assertEqual(result.states_per_segment, int(np.prod(counts)))
                    self.assertEqual(result.traceback_bytes, 4 * (segments - 1) * int(np.prod(counts)))
                    self.assertEqual(result.predicted_power.shape, (segments, len(counts)))

    def test_binary_levels_match_existing_binary_solver(self):
        rng = np.random.default_rng(22)
        for count in [1, 2, 4]:
            powers = rng.uniform(0.0, 20.0, count)
            aggregate = rng.uniform(-5.0, 50.0, 40)
            penalties = rng.uniform(0.0, 40.0, count)
            weights = rng.uniform(0.1, 3.0, aggregate.size)
            binary = solve_binary_temporal_exact(aggregate, powers, penalties, weights)
            multi = solve_multistate_temporal_exact(aggregate, [np.array([0.0, power]) for power in powers], penalties, weights)
            self.assertAlmostEqual(binary.energy, multi.energy, places=8)
            np.testing.assert_array_equal(binary.states, multi.states)
            np.testing.assert_allclose(multi.predicted_power, multi.states * powers)

    def test_switch_penalty_is_categorical_not_squared_index_distance(self):
        result = solve_multistate_temporal_exact([0.0, 100.0], [[0.0, 50.0, 100.0]], 2000.0)
        np.testing.assert_array_equal(result.states, [[0], [2]])
        self.assertEqual(result.energy, 2000.0)

    def test_signed_levels_single_level_and_scalar_penalty(self):
        result = solve_multistate_temporal_exact([-4.0, 1.0], [[-4.0], [0.0, 5.0]], 2.0)
        np.testing.assert_array_equal(result.states, [[0, 0], [0, 1]])
        np.testing.assert_array_equal(result.predicted_power, [[-4.0, 0.0], [-4.0, 5.0]])
        self.assertEqual(result.energy, 2.0)
        self.assertGreaterEqual(result.wall_time_s, 0.0)

    def test_ties_and_duplicate_levels_are_deterministic(self):
        result = solve_multistate_temporal_exact([3.0, -2.0, 1.0], [[0.0, 0.0, 0.0], [0.0, 0.0]], 0.0)
        np.testing.assert_array_equal(result.states, np.zeros((3, 2), dtype=int))
        self.assertEqual(result.energy, 14.0)
        tied = solve_multistate_temporal_exact([5.0, 5.0], [[0.0, 5.0], [0.0, 5.0]], 0.0)
        np.testing.assert_array_equal(tied.states, [[1, 0], [1, 0]])

    def test_single_segment_has_no_switch_cost(self):
        result = solve_multistate_temporal_exact([11.0], [[-1.0, 3.0], [0.0, 8.0, 20.0]], 1e6)
        self.assertEqual(result.energy, 0.0)
        self.assertEqual(result.traceback_bytes, 0)
        np.testing.assert_array_equal(result.states, [[1, 1]])

    def test_invalid_inputs(self):
        cases = [
            ([], [[1]], 0, None), ([[1]], [[1]], 0, None), ([np.nan], [[1]], 0, None),
            ([1], [], 0, None), ([1], [1], 0, None), ([1], [[]], 0, None),
            ([1], [[[1]]], 0, None), ([1], [[np.nan]], 0, None), ([1], [[np.inf]], 0, None),
            ([1], [[1]], -1, None), ([1], [[1]], np.nan, None), ([1], [[1]], np.inf, None),
            ([1], [[1]], [0, 0], None), ([1], [[1]], [[0]], None),
            ([1], [[1]], 0, [0]), ([1], [[1]], 0, [-1]), ([1], [[1]], 0, [np.inf]),
            ([1], [[1]], 0, [np.nan]), ([1], [[1]], 0, [1, 1]), ([1], [[1]], 0, [[1]]),
        ]
        for aggregate, levels, penalties, weights in cases:
            with self.subTest(aggregate=aggregate, levels=levels, penalties=penalties, weights=weights), self.assertRaises(ValueError):
                solve_multistate_temporal_exact(aggregate, levels, penalties, weights)

    def test_resource_caps_apply_before_joint_state_allocation(self):
        with patch("quantum_nilm.multistate.np.arange", side_effect=AssertionError("allocated")):
            with self.assertRaisesRegex(ValueError, "4096 joint states"):
                solve_multistate_temporal_exact([1.0], [[0.0, 1.0]] * 13, 0.0)
            with self.assertRaisesRegex(ValueError, "Traceback needs"):
                solve_multistate_temporal_exact(np.ones(16386), [[0.0, 1.0]] * 12, 0.0)

    def test_overflow_has_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "floating-point range"):
            solve_multistate_temporal_exact([1e300], [[0.0, 1.0]], 0.0)


if __name__ == "__main__":
    unittest.main()
