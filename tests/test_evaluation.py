import unittest
import numpy as np
from quantum_nilm.evaluation import contiguous_slices, aggregate_boundaries, predict_temporal, score_power, pool_window_metrics


class EvaluationTests(unittest.TestCase):
    def test_gap_resets_penalty_and_compression(self):
        times = np.array([0, 30, 120, 150])
        prediction, info = predict_temporal(np.array([0., 0., 10., 10.]), times, [np.array([0., 10.])], [1e6], 1.)
        np.testing.assert_array_equal(prediction[:, 0], [0, 0, 10, 10])
        self.assertEqual(info["objective_energy"], 0)
        self.assertEqual(info["segments"], 2)

    def test_raw_and_centered_prediction_equivalence(self):
        y = np.array([8., 17., 11., 21.])
        t = np.arange(4) * 30
        a, _ = predict_temporal(y, t, [np.array([10., 20.])], [3.])
        b, _ = predict_temporal(y - 10, t, [np.array([0., 10.])], [3.])
        np.testing.assert_array_equal(a, b + 10)

    def test_compressed_constant_recovers_full_block_objective(self):
        predicted, info = predict_temporal(np.array([8., 12.]), np.array([0, 30]),
                                           [np.array([10.])], [0.], 100.)
        self.assertEqual(info["objective_energy"], 0)
        self.assertEqual(info["within_segment_residual_constant"], 8)
        self.assertEqual(info["full_block_objective_energy"], 8)

    def test_perfect_state_metrics_and_no_gap_event(self):
        truth = np.array([[0.], [0.], [100.], [100.]])
        metrics = score_power(truth, truth, [0, 30, 120, 150], [50], ["load"])["load"]
        self.assertEqual(metrics["mae_w"], 0)
        self.assertEqual(metrics["state"]["f1"], 1)
        self.assertIsNone(metrics["event_f1"])

    def test_event_one_to_one_and_pooling(self):
        truth = np.array([[0.], [100.], [0.], [100.]])
        predicted = np.array([[0.], [0.], [100.], [100.]])
        metrics = score_power(truth, predicted, np.arange(4) * 30, [50], ["load"])
        self.assertEqual(metrics["load"]["event_counts"], {"tp": 1, "fp": 0, "fn": 2})
        pooled = pool_window_metrics([{"appliances": metrics}] * 2, ["load"])["load"]
        self.assertEqual(pooled["n_blocks"], 8)
        self.assertEqual(pooled["mae_w"], 50)

    def test_boundaries_and_input_validation(self):
        np.testing.assert_array_equal(aggregate_boundaries(np.array([0., 0., 100.]), 50.), [0, 2, 3])
        for invalid in ([0, 0], [0, np.nan], [0, np.inf], [-np.inf, 0]):
            with self.subTest(timestamps=invalid):
                with self.assertRaises(ValueError):
                    contiguous_slices(np.array(invalid))
                with self.assertRaises(ValueError):
                    predict_temporal([1, 2], invalid, [np.array([0., 10.])], [0.])
        with self.assertRaises(ValueError):
            aggregate_boundaries(np.array([0., 1.]), 0)


if __name__ == "__main__":
    unittest.main()
