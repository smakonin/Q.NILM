"""Regression checks for baseline centering independent of pilot labels."""

import unittest

import numpy as np

from quantum_nilm.qubo import build_binary_temporal_qubo
from quantum_nilm.r1hz import baseline_centered_aggregate, event_compress


class R1HzCenteringTests(unittest.TestCase):
    def test_symmetric_off_state_residuals_do_not_gain_power(self):
        blocks = np.array([[80.0], [120.0]])
        signed = baseline_centered_aggregate(blocks, np.array([100.0]))
        np.testing.assert_array_equal(signed, [-20.0, 20.0])
        self.assertEqual(float(signed.mean()), 0.0)
        legacy = baseline_centered_aggregate(blocks, np.array([100.0]), mode="legacy-clipped")
        self.assertEqual(float(legacy.mean()), 10.0)

    def test_centering_depends_on_total_baseline_not_channel_allocation(self):
        blocks = np.array([[10.0, 100.0], [40.0, 50.0]])
        np.testing.assert_allclose(
            baseline_centered_aggregate(blocks, np.array([20.0, 80.0])),
            baseline_centered_aggregate(blocks, np.array([50.0, 50.0])),
        )

    def test_compression_preserves_signed_power_sum(self):
        blocks = np.array([[80.0], [120.0], [180.0], [220.0]])
        compressed = event_compress(blocks, np.array([[0], [0], [1], [1]]),
                                    np.array([100.0]), np.array([100.0]), 0.5)
        aggregate, weights = compressed[1], compressed[3]
        expected = float((blocks.sum(axis=1) - 100.0).sum())
        self.assertAlmostEqual(float(aggregate @ weights), expected)

    def test_clipping_can_create_a_false_positive_without_optimization_error(self):
        blocks = np.array([[80.0], [120.0]])
        for mode, expected_state in (("signed", 0), ("legacy-clipped", 1)):
            aggregate = baseline_centered_aggregate(blocks, np.array([100.0]), mode=mode)
            q = build_binary_temporal_qubo(np.array([aggregate.mean()]), np.array([15.0]), 0.0)
            bits, costs = q.energies()
            self.assertEqual(int(bits[np.argmin(costs), 0]), expected_state)

    def test_raw_offset_objective_equals_centered_objective(self):
        blocks = np.array([[100.0, 20.0], [120.0, 10.0], [250.0, 40.0]])
        base = np.array([100.0, 20.0])
        powers = np.array([130.0, 25.0])
        q = build_binary_temporal_qubo(baseline_centered_aggregate(blocks, base), powers, 3.0)
        bits, costs = q.energies()
        for sample, cost in zip(bits, costs):
            states = q.decode(sample)
            direct = np.sum((blocks.sum(axis=1) - base.sum() - states @ powers)**2)
            direct += 3 * np.sum(np.diff(states, axis=0)**2)
            self.assertAlmostEqual(float(cost), float(direct))

    def test_invalid_dimensions_values_and_mode(self):
        for blocks, baselines, mode in (
            (np.array([1.0]), np.array([0.0]), "signed"),
            (np.empty((0, 1)), np.array([0.0]), "signed"),
            (np.ones((2, 1)), np.zeros(2), "signed"),
            (np.array([[np.nan]]), np.array([0.0]), "signed"),
            (np.ones((2, 1)), np.array([np.inf]), "signed"),
            (np.ones((2, 1)), np.array([0.0]), "clip"),
        ):
            with self.assertRaises(ValueError):
                baseline_centered_aggregate(blocks, baselines, mode=mode)


if __name__ == "__main__":
    unittest.main()
