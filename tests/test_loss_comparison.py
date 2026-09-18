import itertools
import unittest
import numpy as np
from quantum_nilm import loss_comparison as lc
from quantum_nilm.categorical_qaoa import prepare_categorical_problem
from quantum_nilm.stage_b import optimize_stage_b_qaoa


class LossTests(unittest.TestCase):
    def test_mean(self):
        self.assertAlmostEqual(lc.exact_loss([.2,.3,.5], [0,2,9], "mean"), 5.1)

    def test_fractional_cvar(self):
        self.assertAlmostEqual(lc.exact_loss([.2,.3,.5], [0,2,9], "cvar25"), .4)
        self.assertAlmostEqual(lc.exact_loss([.2,.3,.5], [0,2,9], "cvar50"), 1.2)

    def test_cvar_tied_costs(self):
        self.assertAlmostEqual(lc.exact_loss([.2,.2,.6], [0,0,10], "cvar25"), 0)
        self.assertAlmostEqual(lc.exact_loss([.2,.2,.6], [0,0,10], "cvar50"), 2)

    def test_best4_enumeration(self):
        p, e = np.array([.2,.3,.5]), np.array([0.,2.,9.])
        brute = sum(np.prod(p[list(x)])*min(e[list(x)]) for x in itertools.product(range(3), repeat=4))
        self.assertAlmostEqual(lc.exact_loss(p, e, "best4"), brute)

    def test_best16_two_states(self):
        self.assertAlmostEqual(lc.exact_loss([.2,.8], [3,8], "best16"), 3+5*.8**16)

    def test_sampled_same_budget(self):
        x = np.arange(256.)
        self.assertEqual(lc.sampled_loss(x, "mean"), 127.5)
        self.assertEqual(lc.sampled_loss(x, "cvar25"), 31.5)
        self.assertEqual(lc.sampled_loss(x, "cvar50"), 63.5)
        self.assertEqual(lc.sampled_loss(x, "best4"), 126.)
        self.assertEqual(lc.sampled_loss(x, "best16"), 120.)

    def test_invalid_inputs(self):
        for p, e in (([.2,.2],[0,1]), ([1.1,-.1],[0,1]), ([np.nan,0],[0,1])):
            with self.assertRaises(ValueError): lc.exact_loss(p, e, "mean")
        with self.assertRaises(ValueError): lc.sampled_loss([0], "mean")
        with self.assertRaises(ValueError): lc.exact_loss([1],[0], "unknown")

    def test_training_budget_and_pools(self):
        p = prepare_categorical_problem([1], [[0,1],[0,2]], [0,0])
        a = lc.train(p, "mean", [41,42], budget=40)
        b = lc.train(p, "best4", [41,42], sampled=True, budget=40)
        self.assertEqual(a["evaluations"], 80)
        self.assertEqual(b["training_shots"], 80*256)
        for left, right in zip(a["restarts"], b["restarts"]):
            self.assertEqual(left["initial_candidates"], right["initial_candidates"])
            self.assertEqual(len(right["trace"]), 40)
            self.assertTrue(all(len(x["sample_indices"]) == 256 for x in right["trace"]))

    def test_mean_stage_b_compatibility(self):
        p = prepare_categorical_problem([680], [[0,400,900],[0,120,280]], [1600,400])
        old = optimize_stage_b_qaoa(p, 1, restart_seeds=(9101,9102), evaluations_per_restart=512)
        new = lc.train(p, "mean", [9101,9102])
        np.testing.assert_allclose(new["angles"], old["gammas"]+old["betas"], atol=1e-12)
        np.testing.assert_allclose(new["probabilities"], old["probabilities"], atol=1e-12)
        self.assertEqual(new["evaluations"], old["evaluations"])

    def test_abstention_not_repaired(self):
        p = prepare_categorical_problem([1], [[0,1],[0,2]], [0,0])
        result = lc.decode_metrics(p, np.zeros(4), [1,0])
        self.assertEqual(result["raw_valid_probability"], 0)
        self.assertEqual(result["budgets"]["16"]["no_valid_probability"], 1)
        self.assertIsNone(result["budgets"]["16"]["conditional_best_cost_w2"])

    def test_decoder_tie_rule_with_invalid_mass(self):
        p = prepare_categorical_problem([1], [[0,1],[0,1]], [0,0])
        probabilities = np.array([.1,.2,.3,.1])
        result = lc.decode_metrics(p, probabilities, [1,0], budgets=(2,))
        # Exhaustive enumeration, invalid symbol 4. Equal energies prefer index.
        cost = error = mass = 0.
        raw = list(probabilities)+[.3]
        for a,b in itertools.product(range(5), repeat=2):
            valid = [x for x in (a,b) if x < 4]
            if not valid: continue
            chosen = min(valid, key=lambda x:(p.energies[x],x))
            weight = raw[a]*raw[b]
            mass += weight; cost += weight*p.energies[chosen]
            error += weight*np.mean(abs(p.states[chosen,0]-np.array([1,0])))
        self.assertAlmostEqual(result["budgets"]["2"]["no_valid_probability"], .09)
        self.assertAlmostEqual(result["budgets"]["2"]["conditional_best_cost_w2"], cost/mass)
        self.assertAlmostEqual(result["budgets"]["2"]["conditional_selected_macro_mae_w"], error/mass)


if __name__ == "__main__":
    unittest.main()
