import importlib.util
from pathlib import Path
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "scripts/diagnose_stage_d_initialization.py"
SPEC = importlib.util.spec_from_file_location("stage_d_initialization_diagnostic", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StageDInitializationDistanceTests(unittest.TestCase):
    def test_uniform_and_concentrated_feasible_distributions(self):
        self.assertEqual(MODULE.tv_to_initial_w([0.25] * 4, 0), 0)
        self.assertEqual(MODULE.tv_to_initial_w([1, 0, 0, 0], 0), 0.75)

    def test_invalid_mass_is_not_postselected_away(self):
        self.assertAlmostEqual(MODULE.tv_to_initial_w([0.2] * 4, 0.2), 0.2)
        self.assertAlmostEqual(MODULE.tv_to_initial_w([0] * 4, 1), 1)

    def test_matches_independent_full_physical_vector_distance(self):
        rng = np.random.default_rng(99999)
        physical = rng.random(32)
        physical /= physical.sum()
        feasible = np.asarray([5, 6, 9, 10, 17, 18])
        reference = np.zeros(32)
        reference[feasible] = 1 / 6
        invalid = np.ones(32, dtype=bool)
        invalid[feasible] = False
        direct = 0.5 * np.abs(physical - reference).sum()
        aggregated = MODULE.tv_to_initial_w(physical[feasible], physical[invalid].sum())
        self.assertAlmostEqual(aggregated, direct, places=14)

    def test_invalid_probability_inputs_are_rejected(self):
        for probabilities, invalid in (([], 1), ([-0.1, 0.6], 0.5), ([0.25] * 4, 0.2), ([np.nan], 0), ([1], -0.1)):
            with self.assertRaises(ValueError):
                MODULE.tv_to_initial_w(probabilities, invalid)


if __name__ == "__main__":
    unittest.main()
