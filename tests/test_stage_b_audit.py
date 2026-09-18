"""Boundary tests for the additive independent Stage B archive auditor."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    "independent_stage_b_audit", Path(__file__).resolve().parents[1]/"scripts/audit_stage_b.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class StageBAuditTests(unittest.TestCase):
    def instance(self):
        return dict(state_counts=[2], segments=1, levels=[[0., 10.]],
                    aggregate=[4.], segment_weights=[1.], switch_penalty=[0.],
                    qubits=2, feasible_states=2, truth=[[0]], scale=20.)

    def test_direct_enumeration_cost_and_physical_order(self):
        states, energies, physical, scale = audit.enumerate_direct(self.instance())
        np.testing.assert_array_equal(states, [[[0]], [[1]]])
        np.testing.assert_array_equal(energies, [16., 36.])
        np.testing.assert_array_equal(physical, [1, 2])
        self.assertEqual(scale, 20.)

    def test_corrupted_scale_is_rejected(self):
        bad = self.instance()
        bad["scale"] = 21.
        with self.assertRaisesRegex(ValueError, "coefficient normalization"):
            audit.enumerate_direct(bad)

    def test_independent_mixer_matches_dense_two_by_two(self):
        states, energies, _, scale = audit.enumerate_direct(self.instance())
        gamma, beta = .7, .3
        matrix = np.array([[np.cos(beta), -1j*np.sin(beta)],
                           [-1j*np.sin(beta), np.cos(beta)]])
        expected = matrix@(np.exp(-1j*gamma*energies/scale)/np.sqrt(2))
        actual = audit.direct_ideal_probabilities(states, energies, scale, [2], [gamma], [beta])
        np.testing.assert_allclose(actual, np.abs(expected)**2, atol=1e-14)

    def test_bootstrap_uses_eight_instance_clusters(self):
        result = audit.independently_bootstrap([.5]*8, 13, 100)
        self.assertEqual(result, dict(mean=.5, ci95=[.5, .5], n_instances=8))
        with self.assertRaisesRegex(ValueError, "eight finite"):
            audit.independently_bootstrap([.5]*32, 13, 100)

    def test_missing_and_zero_are_distinct(self):
        audit.close(None, None, "both missing")
        with self.assertRaisesRegex(ValueError, "expected missing"):
            audit.close(0., None, "zero is not missing")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            audit.close(float("nan"), 0., "nonfinite")


if __name__ == "__main__":
    unittest.main()
