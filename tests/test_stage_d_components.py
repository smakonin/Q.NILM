"""Independent, small-case Stage D objective, expansion and statistics checks."""
import itertools
import unittest

import numpy as np

from scripts.run_stage_d_components import (
    ARMS, arm_penalties, assemble_runs, direct_objective, expand_trace,
    infer_window, paired_bootstrap, score_trace,
)


def example(aggregates=(0., 6.), weights=(1, 8), separate=False):
    chunks = []
    cursor = 0
    if separate:
        for index, (value, weight) in enumerate(zip(aggregates, weights)):
            chunks.append({"chunk": index, "run": index, "reset": True,
                           "aggregate": [value], "weights": [weight],
                           "block_start": cursor, "block_stop": cursor + weight})
            cursor += weight
    else:
        chunks.append({"chunk": 0, "run": 0, "reset": True,
                       "aggregate": list(aggregates), "weights": list(weights),
                       "block_start": 0, "block_stop": sum(weights)})
        cursor = sum(weights)
    return {"window": {"id": "small"}, "blocks": cursor, "chunks": chunks}


class StageDComponentTests(unittest.TestCase):
    def test_global_mean_preserves_total_coefficient_not_global_max(self):
        ranges, rho = np.array([10., 2., 5., 3.]), .1
        original = arm_penalties(ranges, rho, ARMS[0])
        mean = arm_penalties(ranges, rho, ARMS[3])
        maximum = arm_penalties(ranges, rho, ARMS[2])
        self.assertAlmostEqual(float(original.sum()), float(mean.sum()))
        np.testing.assert_array_equal(maximum, [10., 10., 10., 10.])
        self.assertGreater(float(maximum.sum()), float(original.sum()))

    def test_duration_ablation_changes_fit_but_not_expansion(self):
        window = example()
        levels, ranges, rho = [np.array([0., 10.])], np.array([10.]), .5
        original = infer_window(window, levels, ranges, rho, ARMS[0])
        unit = infer_window(window, levels, ranges, rho, ARMS[1])
        self.assertEqual(original["runs"][0]["states"], [[0], [1]])
        self.assertEqual(unit["runs"][0]["states"], [[0], [0]])
        prediction, means = expand_trace(window, original, levels)
        np.testing.assert_array_equal(prediction[:, 0], [0.] + [10.] * 8)
        np.testing.assert_array_equal(means, [0.] + [6.] * 8)
        self.assertEqual(len(expand_trace(window, unit, levels)[0]), 9)

    def test_every_arm_matches_independent_exhaustive_small_case(self):
        window = example((5., 11.), (2, 5))
        levels = [np.array([-1., 8.]), np.array([1., 4., 9.])]
        ranges, rho = np.array([9., 8.]), .3
        for arm in ARMS:
            actual = infer_window(window, levels, ranges, rho, arm)
            weights = np.ones(2) if arm == ARMS[1] else np.array([2., 5.])
            penalties = arm_penalties(ranges, rho, arm)
            candidates = []
            for labels in itertools.product(range(2), range(3), range(2), range(3)):
                states = np.array(labels).reshape(2, 2)
                energy = sum(weights[t] * ([5., 11.][t] - sum(levels[i][states[t, i]] for i in range(2)))**2 for t in range(2))
                energy += sum(penalties[i] * int(states[0, i] != states[1, i]) for i in range(2))
                candidates.append(energy)
            self.assertAlmostEqual(actual["runs"][0]["native_segment_objective"], min(candidates), places=10)

    def test_resets_do_not_charge_or_discourage_cross_gap_switches(self):
        window = example((0., 10.), (1, 1), separate=True)
        trace = infer_window(window, [np.array([0., 10.])], np.array([10.]), 100., ARMS[0])
        self.assertEqual([run["states"] for run in trace["runs"]], [[[0]], [[1]]])
        self.assertEqual(sum(run["native_transition"] for run in trace["runs"]), 0)
        self.assertEqual(sum(run["switch_counts"][0] for run in trace["runs"]), 0)

    def test_direct_objective_charges_category_change_not_ordinal_distance(self):
        total, reconstruction, transition, switches = direct_objective(
            [0., 20.], [2., 3.], [[0], [2]], [np.array([0., 10., 20.])], [7.])
        self.assertEqual(total, 7.)
        self.assertEqual(reconstruction, 0.)
        self.assertEqual(transition, 7.)
        np.testing.assert_array_equal(switches, [1])

    def test_raw_block_residual_identity_and_manual_mae(self):
        window = example((1., 8.), (2, 1))
        levels = [np.array([0., 10.]), np.array([0.]), np.array([0.]), np.array([0.])]
        trace = infer_window(window, levels, np.array([10., 1., 1., 1.]), 0., ARMS[0])
        loaded = {"timestamps": np.array([0, 30, 60]),
                  "values": np.array([[0., 0., 0., 0.], [2., 2., 0., 0.], [8., 8., 0., 0.]])}
        score = score_trace(window, trace, loaded, levels, [5., 5., 5.])
        self.assertAlmostEqual(score["compression_residual_constant"], 2.)
        self.assertAlmostEqual(score["common_full_block_objective"], 8.)
        self.assertAlmostEqual(score["appliances"]["dryr"]["mae_w"], 4. / 3.)

    def test_missing_chunk_or_reset_is_rejected(self):
        bad = example(separate=True)
        bad["chunks"][1]["reset"] = False
        with self.assertRaises(ValueError):
            assemble_runs(bad)
        bad = example()
        bad["blocks"] += 1
        with self.assertRaises(ValueError):
            assemble_runs(bad)

    def test_bootstrap_is_ratio_of_paired_sums_not_mean_window_mae(self):
        def rows(values):
            return [{"window_id": str(i), "blocks": blocks,
                     "appliances": {c: {"absolute_error_sum_w": error} for c in ("dryr", "frdg", "vacu")}}
                    for i, (blocks, error) in enumerate(values)]
        left, right = rows([(1, 10.), (9, 0.)]), rows([(1, 0.), (9, 0.)])
        actual = paired_bootstrap(left, right)
        self.assertEqual(actual["difference_w"], 1.)
        self.assertNotEqual(actual["difference_w"], 5.)
        self.assertEqual(actual["seed"], 7031)
        self.assertEqual(actual["replicates"], 2000)
        self.assertEqual(actual, paired_bootstrap(left, right))
        with self.assertRaises(ValueError):
            paired_bootstrap(left + [left[0]], right)


if __name__ == "__main__":
    unittest.main()
