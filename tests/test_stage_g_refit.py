"""Small, non-REFIT-data tests for external inference and paired statistics."""
import inspect
import unittest

import numpy as np

from scripts.run_stage_g_refit import (CHANNELS, HOMES, METHODS, SEEDS,
                                      errors_by_day, fit_model, paired_days,
                                      predict, summarize_records, validate_coverage)


class StageGTests(unittest.TestCase):
    def test_metadata_cohort_and_channel_order(self):
        self.assertEqual(len(HOMES), 14)
        self.assertNotIn(13, HOMES)
        self.assertEqual(HOMES[11], (3, 2, 4))
        self.assertEqual(HOMES[18], (5, 3, 6))
        self.assertEqual(CHANNELS, ['washing', 'cooling', 'dishwasher'])

    def test_training_model_uses_raw_background_and_retains_constant_channels(self):
        values = np.array([[10., 20., 0., 0.], [30., 40., 0., 0.],
                           [60., 70., 0., 0.], [80., 90., 0., 0.]])
        model = fit_model(values)
        np.testing.assert_allclose(model['levels'][3], [-10.])
        self.assertEqual(model['actual_counts'], [4, 1, 1, 1])
        self.assertEqual(model['training_blocks'], 4)
        self.assertEqual(model['training_mean'][3], -10.)
        np.testing.assert_allclose(model['penalties'], .1 * np.square(model['ranges']))
        self.assertEqual(model['thresholds'][1:], [1., 1.])

    def test_no_reference_argument_and_all_predictions_cover_gaps(self):
        self.assertNotIn('truth', inspect.signature(predict).parameters)
        model = {'levels': [[0., 10.], [0.], [0.], [0.]],
                 'penalties': [1., 0., 0., 0.], 'event_threshold': 1.,
                 'training_mean': [5., 0., 0., 0.]}
        pending, traces = predict(np.array([0., 10., 10.]), np.array([0, 30, 90]),
                                  model, {'gammas': [1.], 'betas': [.5]}, 1, 0, shots=8)
        self.assertEqual(len(pending), 9)
        self.assertEqual(len(traces['chunks']), 2)
        self.assertTrue(all(c['reset'] for c in traces['chunks']))
        for _, _, prediction, _ in pending:
            self.assertEqual(prediction.shape, (3, 4))
            self.assertTrue(np.isfinite(prediction).all())
        exact = next(p for m, s, p, t in pending if m == 'exact_full')
        np.testing.assert_array_equal(exact[:, 0], [0., 10., 10.])

    def test_seed_averaging_precedes_ratio(self):
        rows = []
        for day, blocks in [('a', 1), ('b', 9)]:
            for i, seed in enumerate(SEEDS):
                error = (i + 1) * 10 if day == 'a' else 0
                rows.append({'house': 1, 'window_id': day, 'model': 'qaoa_ideal',
                             'seed': seed, 'blocks': blocks,
                             'appliances': {c: {'absolute_error_sum_w': error} for c in CHANNELS}})
        result = errors_by_day(rows, 1, 'qaoa_ideal')
        self.assertEqual(result['a']['error'], 60.)
        self.assertEqual(result['b']['denominator'], 27)
        zero = {k: {'error': 0., 'denominator': r['denominator']} for k, r in result.items()}
        interval = paired_days(result, zero, 8102)
        self.assertEqual(interval['difference_w'], 2.)
        self.assertEqual(interval, paired_days(result, zero, 8102))
        with self.assertRaises(ValueError):
            errors_by_day(rows[:-1], 1, 'qaoa_ideal')

    def test_paired_days_rejects_unmatched_coverage(self):
        left = {'a': {'error': 5., 'denominator': 3}}
        with self.assertRaises(ValueError):
            paired_days(left, {'b': {'error': 2., 'denominator': 3}}, 1)
        with self.assertRaises(ValueError):
            paired_days(left, {'a': {'error': 2., 'denominator': 6}}, 1)

    def test_equal_home_estimand_differs_from_block_pooling(self):
        from quantum_nilm.evaluation import score_power
        records = []
        for home, blocks in [(1, 1), (2, 9)]:
            for method in METHODS:
                seeds = SEEDS if method in ('qaoa_ideal', 'uniform') else (None,)
                for seed in seeds:
                    power = 10. if home == 1 and method == 'qaoa_ideal' else 0.
                    appliances = score_power(np.zeros((blocks, 3)), np.full((blocks, 3), power),
                                             np.arange(blocks) * 30, [1.] * 3, CHANNELS)
                    records.append({'house': home, 'window_id': 'test-000', 'model': method,
                                    'seed': seed, 'blocks': blocks, 'solver_wall_time_s': 0.,
                                    'raw_aggregate_mae_w': power * 3, 'appliances': appliances})
        result = summarize_records(records)
        self.assertEqual(result['overall']['qaoa_ideal']['equal_home_macro_mae_w'], 5.)
        self.assertEqual(result['overall']['qaoa_ideal']['block_pooled_macro_mae_w'], 1.)
        self.assertEqual(result['paired_home_comparisons'][0]['difference_w'], 5.)

    def test_no_evaluable_home_returns_null_not_successful_accuracy(self):
        result = summarize_records([])
        self.assertEqual(result['status'], 'no_evaluable_homes')
        self.assertIsNone(result['overall']['qaoa_ideal']['equal_home_macro_mae_w'])
        self.assertEqual(result['nonevaluable_homes'], sorted(HOMES))

    def test_missing_selected_window_and_missing_method_fail_coverage_gate(self):
        window = {'id': 'test-000', 'start_unix': 0, 'end_unix': 86400}
        config = {'homes': [{'house': 1, 'manifest': {'splits': {'test': {'windows': [window]}}}}]}
        with self.assertRaises(ValueError):
            validate_coverage(config, [], [])
        quality = [{'house': 1, 'window': window, 'valid_blocks': 1, 'quality': {}}]
        with self.assertRaises(ValueError):
            validate_coverage(config, quality, [])
        with self.assertRaises(ValueError):
            validate_coverage(config, quality * 2, [])


if __name__ == '__main__':
    unittest.main()
