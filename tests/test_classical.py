import unittest
from unittest.mock import patch

import numpy as np

from quantum_nilm.classical import solve_binary_temporal_exact
from quantum_nilm.qubo import build_binary_temporal_qubo


class BinaryTemporalExactTests(unittest.TestCase):
    @staticmethod
    def direct_objective(aggregate, powers, penalties, weights, states):
        return float(
            np.sum(weights * (aggregate - states @ powers) ** 2)
            + np.sum(penalties * np.diff(states, axis=0) ** 2)
        )

    def test_matches_exhaustive_tiny_random_instances(self):
        generator = np.random.default_rng(915)
        for n_appliances, n_segments in [(1, 1), (2, 3), (3, 3), (4, 2)]:
            for repetition in range(8):
                with self.subTest(n=n_appliances, k=n_segments, repetition=repetition):
                    powers = generator.uniform(0, 20, n_appliances)
                    aggregate = generator.uniform(-10, 30, n_segments)
                    weights = generator.uniform(0.1, 3, n_segments)
                    penalties = generator.uniform(0, 30, n_appliances)
                    penalties[repetition % n_appliances] = 0.0
                    result = solve_binary_temporal_exact(
                        aggregate, powers, penalties, weights
                    )
                    n_variables = n_segments * n_appliances
                    all_ids = np.arange(1 << n_variables, dtype=np.uint32)
                    all_states = (
                        (all_ids[:, None] >> np.arange(n_variables, dtype=np.uint32)) & 1
                    ).astype(np.int8).reshape(-1, n_segments, n_appliances)
                    exhaustive = min(
                        self.direct_objective(aggregate, powers, penalties, weights, states)
                        for states in all_states
                    )
                    self.assertAlmostEqual(result.energy, exhaustive, places=8)
                    self.assertAlmostEqual(
                        result.energy,
                        self.direct_objective(
                            aggregate, powers, penalties, weights, result.states
                        ),
                        places=10,
                    )

    def test_matches_qubo_including_constant_and_scalar_penalty(self):
        aggregate = np.array([-5.0, 24.0, 7.0])
        powers = np.array([7.0, 20.0])
        weights = np.array([4.0, 1.0, 2.0])
        qubo = build_binary_temporal_qubo(aggregate, powers, 13.0, weights)
        result = solve_binary_temporal_exact(aggregate, powers, 13.0, weights)
        _, energies = qubo.energies()
        self.assertAlmostEqual(result.energy, float(energies.min()))
        self.assertAlmostEqual(result.energy, qubo.energy(result.states))

    def test_long_sequence_matches_independent_dense_dynamic_program(self):
        generator = np.random.default_rng(42)
        aggregate = generator.uniform(-2, 18, 1200)
        powers = np.array([2.0, 5.0, 10.0])
        penalties = np.array([0.0, 11.0, 80.0])
        weights = generator.uniform(0.2, 2.0, aggregate.size)
        joint_states = (
            (np.arange(8)[:, None] >> np.arange(3)) & 1
        ).astype(np.int8)
        # The dense matrix is deliberately used only as a tiny-N independent
        # test oracle; the implementation must never allocate this matrix.
        transitions = np.sum(
            penalties * (joint_states[:, None] - joint_states[None, :]) ** 2,
            axis=2,
        )
        predictions = joint_states @ powers
        dense_values = weights[0] * (aggregate[0] - predictions) ** 2
        for segment in range(1, aggregate.size):
            dense_values = np.min(dense_values[:, None] + transitions, axis=0)
            dense_values += weights[segment] * (aggregate[segment] - predictions) ** 2
        result = solve_binary_temporal_exact(aggregate, powers, penalties, weights)
        self.assertAlmostEqual(result.energy, float(dense_values.min()), places=7)
        self.assertEqual(result.states.shape, (1200, 3))
        self.assertEqual(result.states_per_segment, 8)
        self.assertEqual(result.traceback_bytes, 1199 * 8 * 4)
        self.assertGreaterEqual(result.wall_time_s, 0.0)

    def test_degenerate_powers_and_zero_penalties_have_deterministic_traceback(self):
        result = solve_binary_temporal_exact([3.0, -2.0, 1.0], [0.0, 0.0], 0.0)
        np.testing.assert_array_equal(result.states, np.zeros((3, 2), dtype=np.int8))
        self.assertEqual(result.energy, 14.0)
        tied = solve_binary_temporal_exact([5.0, 5.0], [5.0, 5.0], [0.0, 0.0])
        np.testing.assert_array_equal(tied.states, [[1, 0], [1, 0]])

    def test_single_segment_has_no_transition_cost(self):
        result = solve_binary_temporal_exact([11.0], [3.0, 8.0], [1e6, 1e6])
        self.assertEqual(result.energy, 0.0)
        self.assertEqual(result.traceback_bytes, 0)
        np.testing.assert_array_equal(result.states, [[1, 1]])

    def test_invalid_inputs(self):
        cases = [
            ([], [1], 0, None),
            ([[1]], [1], 0, None),
            ([np.nan], [1], 0, None),
            ([np.inf], [1], 0, None),
            ([1], [], 0, None),
            ([1], [[1]], 0, None),
            ([1], [-1], 0, None),
            ([1], [np.nan], 0, None),
            ([1], [np.inf], 0, None),
            ([1], [1], -1, None),
            ([1], [1], np.nan, None),
            ([1], [1], np.inf, None),
            ([1], [1], [0, 0], None),
            ([1], [1], [[0]], None),
            ([1], [1], 0, [0]),
            ([1], [1], 0, [-1]),
            ([1], [1], 0, [np.inf]),
            ([1], [1], 0, [np.nan]),
            ([1], [1], 0, [1, 1]),
            ([1], [1], 0, [[1]]),
        ]
        for aggregate, powers, penalties, weights in cases:
            with self.subTest(aggregate=aggregate, powers=powers, penalties=penalties, weights=weights):
                with self.assertRaises(ValueError):
                    solve_binary_temporal_exact(aggregate, powers, penalties, weights)

    def test_resource_caps_apply_before_state_space_allocation(self):
        with self.assertRaisesRegex(ValueError, "20 appliances"):
            solve_binary_temporal_exact([1.0], np.ones(21), 0.0)
        # 65 transition rows * 2**20 states * 4 bytes exceeds 256 MiB.
        with patch("quantum_nilm.classical.np.arange", side_effect=AssertionError("allocated")):
            with self.assertRaisesRegex(ValueError, "Traceback needs"):
                solve_binary_temporal_exact(np.ones(66), np.ones(20), 0.0)

    def test_overflow_has_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "floating-point range"):
            solve_binary_temporal_exact([1e300], [1.0], 0.0)


if __name__ == "__main__":
    unittest.main()
