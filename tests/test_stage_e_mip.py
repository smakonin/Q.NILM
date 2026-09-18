"""Independent exhaustive and failure-path checks for the additive MILP."""

import itertools
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult

from quantum_nilm.stage_e_mip import solve_categorical_milp


def direct_objective(aggregate, levels, penalties, weights, path, prior=None):
    value = sum(float(weights[t]) * (float(aggregate[t]) - sum(float(levels[i][path[t][i]]) for i in range(len(levels)))) ** 2 for t in range(len(aggregate)))
    value += sum(float(penalties[i]) for t in range(1, len(path)) for i in range(len(levels)) if path[t][i] != path[t - 1][i])
    if prior is not None:
        value += sum(float(penalties[i]) for i in range(len(levels)) if path[0][i] != prior[i])
    return value


class StageEMilpTests(unittest.TestCase):
    def test_random_tiny_matches_independent_enumeration(self):
        rng = np.random.default_rng(5091)
        for counts, segments in [([2], 4), ([3], 3), ([2, 3], 3), ([1, 2, 3], 2)]:
            for repeat in range(3):
                with self.subTest(counts=counts, segments=segments, repeat=repeat):
                    levels = [rng.uniform(-8, 10, count) for count in counts]
                    aggregate = rng.uniform(-10, 15, segments)
                    weights = rng.uniform(.1, 5, segments)
                    penalties = rng.uniform(0, 50, len(counts))
                    penalties[repeat % len(counts)] = 0
                    prior = np.array([rng.integers(count) for count in counts]) if repeat == 2 else None
                    result = solve_categorical_milp(aggregate, levels, penalties, weights, previous_state=prior)
                    joint = list(itertools.product(*(range(count) for count in counts)))
                    optimum = min(direct_objective(aggregate, levels, penalties, weights, path, prior) for path in itertools.product(joint, repeat=segments))
                    self.assertEqual(result.status_code, 0)
                    self.assertTrue(result.success)
                    self.assertTrue(result.incumbent_feasible)
                    self.assertAlmostEqual(result.energy, optimum, places=7)
                    self.assertAlmostEqual(result.energy, direct_objective(aggregate, levels, penalties, weights, result.states, prior), places=7)
                    self.assertAlmostEqual(result.dual_bound, optimum, places=7)
                    self.assertLessEqual(result.max_integrality_violation, 1e-6)
                    self.assertLessEqual(result.max_bound_violation, 1e-6)
                    self.assertLessEqual(result.max_constraint_violation, 1e-6)

    def test_categorical_switch_is_not_ordinal_distance(self):
        result = solve_categorical_milp([0, 100], [[0, 50, 100]], 2000)
        np.testing.assert_array_equal(result.states, [[0], [2]])
        self.assertEqual(result.energy, 2000)

    def test_signed_background_duplicate_levels_zero_weights_penalty(self):
        result = solve_categorical_milp([-4, 1, -4], [[-4], [0, 5, 5]], [0, 0], [1, 3, 2])
        np.testing.assert_array_equal(result.predicted_power, [[-4, 0], [-4, 5], [-4, 0]])
        self.assertEqual(result.energy, 0)
        self.assertEqual(result.dual_bound, 0)
        self.assertEqual(result.relative_gap, 0)

    def test_all_tied_emissions_return_valid_energy_not_promised_dp_tie_break(self):
        result = solve_categorical_milp([3, -2, 1], [[0, 0, 0], [0, 0]], 0)
        self.assertEqual(result.energy, 14)
        self.assertEqual(result.objective_offset, 14)
        self.assertEqual(result.objective_scale, 1)
        self.assertEqual(result.solver_objective, 14)
        self.assertTrue(result.incumbent_feasible)

    def test_single_interval_prior_is_charged(self):
        without = solve_categorical_milp([8], [[0, 10]], 100)
        with_prior = solve_categorical_milp([8], [[0, 10]], 100, previous_state=[0])
        self.assertEqual(without.energy, 4)
        self.assertEqual(with_prior.energy, 64)
        np.testing.assert_array_equal(with_prior.states, [[0]])

    def test_scaling_and_constant_restoration(self):
        aggregate = np.array([1e6, -1e6, 2e6])
        levels = [np.array([-1e5, 3e5]), np.array([-6e5, 1e5])]
        penalties, weights = np.array([1e8, 3e8]), np.array([3., 2., 1.])
        result = solve_categorical_milp(aggregate, levels, penalties, weights)
        paths = itertools.product(list(itertools.product(range(2), range(2))), repeat=3)
        optimum = min(direct_objective(aggregate, levels, penalties, weights, path) for path in paths)
        self.assertAlmostEqual(result.energy / optimum, 1, places=12)
        self.assertGreater(result.objective_scale, 1)
        self.assertGreater(result.objective_offset, 0)
        self.assertEqual(result.solver_objective, result.objective_offset + result.objective_scale * result.raw_solver_objective)
        self.assertEqual(result.dual_bound, result.objective_offset + result.objective_scale * result.raw_solver_dual_bound)

    def test_sizes_timing_and_documented_options(self):
        result = solve_categorical_milp([1, 7], [[0, 2], [0, 4, 8]], [1, 2], time_limit_s=2.5, mip_rel_gap=1e-8, presolve=False, node_limit=50)
        self.assertEqual(result.binary_variables, 12)
        self.assertEqual(result.continuous_variables, 2)
        self.assertEqual(result.constraints, 7)
        self.assertEqual(result.constraint_nonzeros, 41)
        self.assertEqual(result.states_per_segment, 6)
        self.assertEqual(result.options, {"time_limit": 2.5, "mip_rel_gap": 1e-8, "presolve": False, "node_limit": 50})
        for value in [result.build_time_s, result.solve_time_s, result.decode_time_s, result.wall_time_s]:
            self.assertGreaterEqual(value, 0)
        self.assertGreaterEqual(result.wall_time_s, result.build_time_s + result.solve_time_s)

    def test_real_tiny_time_limit_is_reported_without_retry(self):
        result = solve_categorical_milp(np.arange(12), [[0, 1]] * 6, 1, time_limit_s=1e-9, presolve=False)
        self.assertEqual(result.status_code, 1)
        self.assertEqual(result.status, "limit_reached")
        self.assertFalse(result.success)

    def test_time_limit_without_incumbent_preserves_status(self):
        response = OptimizeResult(status=1, success=False, message="Time limit reached", x=None, fun=None, mip_dual_bound=0.0, mip_node_count=2, mip_gap=None)
        with patch("quantum_nilm.stage_e_mip.milp", return_value=response) as solver:
            result = solve_categorical_milp([3], [[0, 5]], 0, time_limit_s=.2)
        self.assertEqual(solver.call_count, 1)
        self.assertFalse(result.incumbent_feasible)
        self.assertIsNone(result.states)
        self.assertIsNone(result.energy)
        self.assertEqual(result.dual_bound, 4)
        self.assertEqual(result.mip_node_count, 2)
        self.assertEqual(result.incumbent_validation, "no_incumbent")

    def test_time_limited_feasible_incumbent_keeps_nonzero_native_gap(self):
        response = OptimizeResult(status=1, success=False, message="Time limit reached", x=np.array([1., 0.]), fun=5., mip_dual_bound=0., mip_node_count=1, mip_gap=1.)
        with patch("quantum_nilm.stage_e_mip.milp", return_value=response):
            result = solve_categorical_milp([3], [[0, 5]], 0)
        self.assertTrue(result.incumbent_feasible)
        self.assertFalse(result.success)
        self.assertEqual(result.energy, 9)
        self.assertEqual(result.dual_bound, 4)
        self.assertEqual(result.absolute_gap, 5)
        self.assertEqual(result.relative_gap, 5 / 9)
        self.assertEqual(result.solver_mip_gap, 1)

    def test_limit_incumbent_auxiliary_overcharge_is_not_hidden(self):
        # Same category both intervals with arbitrary positive z: the path is
        # feasible and its true cost is lower than the solver's incumbent.
        response = OptimizeResult(status=1, success=False, message="Time limit reached", x=np.array([1., 0., 1., 0., 1.]), fun=5., mip_dual_bound=0., mip_node_count=1, mip_gap=1.)
        with patch("quantum_nilm.stage_e_mip.milp", return_value=response):
            result = solve_categorical_milp([0, 0], [[0, 2]], 5)
        self.assertEqual(result.energy, 0)
        self.assertEqual(result.solver_objective, 5)
        self.assertEqual(result.decoded_minus_solver_objective, -5)
        self.assertEqual(result.incumbent_validation, "passed_path_auxiliary_overcharge")

    def test_negative_dual_gap_roundoff_is_preserved(self):
        response = OptimizeResult(status=0, success=True, message="Optimal", x=np.array([1., 0.]), fun=0., mip_dual_bound=1e-15, mip_node_count=0, mip_gap=0.)
        with patch("quantum_nilm.stage_e_mip.milp", return_value=response):
            result = solve_categorical_milp([0], [[0, 1]], 0)
        self.assertEqual(result.absolute_gap, -1e-15)
        self.assertEqual(result.relative_gap, -1e-15)
        self.assertEqual(result.bound_validation, "negative_within_roundoff")

    def test_infeasible_solver_response_is_preserved_not_filled(self):
        # Valid model is mathematically feasible; this diagnostic branch
        # preserves an unexpected solver-infeasible outcome without fallback.
        response = OptimizeResult(status=2, success=False, message="Infeasible", x=None, fun=None)
        with patch("quantum_nilm.stage_e_mip.milp", return_value=response):
            result = solve_categorical_milp([3], [[0, 5]], 0)
        self.assertEqual(result.status, "infeasible")
        self.assertIsNone(result.energy)
        self.assertIsNone(result.dual_bound)

    def test_invalid_incumbents_are_not_decoded_as_predictions(self):
        for values, expected in [([.5, .5], "nonintegral_incumbent"), ([1.1, -.1], "bound_violation"), ([0., 0.], "constraint_violation"), ([float("nan"), 0.], "nonfinite_or_wrong_shape")]:
            with self.subTest(values=values):
                response = OptimizeResult(status=0, success=True, message="Optimal", x=np.array(values), fun=0.)
                with patch("quantum_nilm.stage_e_mip.milp", return_value=response):
                    result = solve_categorical_milp([0], [[0, 1]], 0)
                self.assertFalse(result.incumbent_feasible)
                self.assertEqual(result.incumbent_validation, expected)
                self.assertIsNone(result.energy)

    def test_objective_mismatch_does_not_pass(self):
        response = OptimizeResult(status=0, success=True, message="Optimal", x=np.array([0., 1.]), fun=0.)
        with patch("quantum_nilm.stage_e_mip.milp", return_value=response):
            result = solve_categorical_milp([0], [[0, 1]], 0)
        self.assertFalse(result.incumbent_feasible)
        self.assertEqual(result.incumbent_validation, "recomputed_energy_exceeds_solver_objective")
        self.assertEqual(result.decoded_minus_solver_objective, 1)

    def test_rejects_invalid_input(self):
        cases = [([], [[0]], 0, None), ([np.nan], [[0]], 0, None), ([0], [], 0, None), ([0], [1], 0, None), ([0], [[np.inf]], 0, None), ([0], [[0]], -1, None), ([0], [[0]], [0, 0], None), ([0], [[0]], 0, [0]), ([0], [[0]], 0, [np.nan]), ([0], [[0]], 0, [1, 1])]
        for aggregate, levels, penalties, weights in cases:
            with self.subTest(aggregate=aggregate, levels=levels, penalties=penalties, weights=weights), self.assertRaises(ValueError):
                solve_categorical_milp(aggregate, levels, penalties, weights)
        for options in [{"time_limit_s": 0}, {"time_limit_s": np.inf}, {"time_limit_s": True}, {"mip_rel_gap": -1}, {"mip_rel_gap": np.nan}, {"presolve": 1}, {"node_limit": -1}, {"node_limit": .5}, {"previous_state": [1]}, {"previous_state": [0.]}, {"previous_state": [True]}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                solve_categorical_milp([0], [[0]], 0, **options)

    def test_resource_caps_and_overflow_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "4096 joint states"):
            solve_categorical_milp([0], [[0, 1]] * 13, 0)
        with self.assertRaisesRegex(ValueError, "resource cap"):
            solve_categorical_milp(np.ones(300), [[0, 1]] * 12, 0)
        with self.assertRaisesRegex(ValueError, "floating-point range"):
            solve_categorical_milp([1e300], [[0, 1]], 0)


if __name__ == "__main__":
    unittest.main()
