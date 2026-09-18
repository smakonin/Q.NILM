"""Finite-shot endpoint boundary cases for the frozen Stage B runner."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from quantum_nilm.categorical_qaoa import prepare_categorical_problem

spec = importlib.util.spec_from_file_location("stage_b_runner", Path(__file__).resolve().parents[1]/"scripts/run_stage_b.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class StageBRunnerTests(unittest.TestCase):
    def setUp(self):
        self.problem = prepare_categorical_problem([10], [[0, 10]], [0])
        self.truth = np.array([[1]])

    def test_zero_feasibility_is_failure_not_exact_fallback(self):
        rows = list(runner.sample_trials(self.problem,self.truth,[0,0],"x_p1_high",[4],3))
        for row in rows:
            self.assertEqual(row["no_feasible"],1)
            self.assertEqual(row["hit_optimum"],0)
            self.assertEqual(row["failure_penalized_normalized_gap"],1)
            self.assertEqual(row["truth_category_accuracy"],0)
            self.assertIsNone(row["certified_gap"])
            self.assertIsNone(row["best_index"])

    def test_sampled_best_cannot_insert_unobserved_optimum(self):
        rows=list(runner.sample_trials(self.problem,self.truth,[1,0],"x_p1_ideal",[4],3))
        self.assertTrue(all(row["best_index"]==0 and row["hit_optimum"]==0 for row in rows))
        self.assertTrue(all(row["certified_gap"]==100 for row in rows))

    def test_optimum_probability_is_unconditional(self):
        meta=runner.endpoints(self.problem,self.truth,[.1,.2])
        self.assertAlmostEqual(meta["feasible_probability"],.3)
        self.assertAlmostEqual(meta["optimum_probability"],.2)
        self.assertEqual(meta["shots_to_optimum_99"],21)
        self.assertAlmostEqual(meta["conditional_expected_cost"],100/3)

    def test_shots99_endpoints(self):
        self.assertIsNone(runner.shots99(0))
        self.assertEqual(runner.shots99(1),1)
        self.assertEqual(runner.shots99(.5),7)

    def test_common_random_streams_but_independent_repeats(self):
        left=list(runner.sample_trials(self.problem,self.truth,[.5,.5],"x_p1_ideal",[4],3))
        right=list(runner.sample_trials(self.problem,self.truth,[.5,.5],"x_p0_uniform",[4],3))
        self.assertEqual([r["sample_seed"] for r in left],[r["sample_seed"] for r in right])
        self.assertEqual(len(set(r["sample_seed"] for r in left)),3)

    def test_tied_optima_are_summed(self):
        problem=prepare_categorical_problem([5],[[0,10]],[0])
        meta=runner.endpoints(problem,self.truth,[.2,.3])
        self.assertEqual(meta["optimum_count"],2)
        self.assertAlmostEqual(meta["optimum_probability"],.5)


if __name__ == "__main__":
    unittest.main()
