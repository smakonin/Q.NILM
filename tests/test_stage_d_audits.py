"""Small analytical controls for independent Stage D audit implementations."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_stage_d_components import class_metrics, independent_events, independently_score
from audit_stage_d_mixer import augmented_objective, independent_x_probabilities


class StageDAuditTests(unittest.TestCase):
    def problem(self):
        return dict(state_counts=[2], segments=1, levels=[[0., 10.]],
                    aggregate=[4.], segment_weights=[1.], switch_penalty=[0.],
                    qubits=2, feasible_states=2, truth=[[0]], scale=20.)

    def test_independent_penalty_expansion_known_two_qubit_example(self):
        _, energies, ids, base_scale, full, scale, certificate = augmented_objective(self.problem())
        np.testing.assert_array_equal(energies, [16., 36.])
        np.testing.assert_array_equal(ids, [1, 2])
        np.testing.assert_array_equal(full, [57., 16., 36., 77.])
        self.assertEqual((base_scale, scale, certificate["barrier"]), (20., 144., 41.))
        self.assertEqual(certificate["invalid_minus_worst_feasible"], 21.)

    def test_independent_full_x_has_no_hidden_postselection(self):
        _, _, ids, _, full, scale, _ = augmented_objective(self.problem())
        h = independent_x_probabilities(full, scale, ids, [0.], [0.], "h")
        w = independent_x_probabilities(full, scale, ids, [0.], [0.], "w")
        leaked = independent_x_probabilities(full, scale, ids, [0.], [np.pi/4], "w")
        np.testing.assert_allclose(h, [.25]*4)
        np.testing.assert_allclose(w, [0., .5, .5, 0.])
        self.assertAlmostEqual(float(leaked[ids].sum()), 0., places=14)
        self.assertAlmostEqual(float(leaked.sum()), 1., places=14)

    def test_event_matching_never_crosses_timestamp_gap(self):
        truth = np.array([False, True, True, False])
        prediction = np.array([False, True, False, False])
        timestamps = np.array([0, 30, 120, 150])
        self.assertEqual(independent_events(truth, prediction, timestamps), dict(tp=1, fp=0, fn=1))

    def test_classification_empty_denominators_remain_missing(self):
        values = class_metrics(0, 4, 0, 0)
        self.assertEqual(values["bit_accuracy"], 1.)
        self.assertIsNone(values["f1"])
        self.assertIsNone(values["mcc"])

    def test_raw_power_mae_energy_and_proxy_counts(self):
        truth = np.array([[0., 0., 0.], [10., 0., 0.]])
        prediction = np.array([[0., 0., 0.], [8., 0., 0.]])
        score = independently_score(truth, prediction, np.array([0, 30]), [5., 5., 5.])
        self.assertEqual(score["dryr"]["mae_w"], 1.)
        self.assertAlmostEqual(score["dryr"]["energy_error_wh"], -2/120)
        self.assertEqual(score["dryr"]["state"]["tp"], 1)
        self.assertEqual(score["dryr"]["event_counts"], dict(tp=1, fp=0, fn=0))


if __name__ == "__main__":
    unittest.main()
