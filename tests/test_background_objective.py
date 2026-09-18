import itertools
import unittest
import numpy as np
from scripts import background_objective_core as c
from scripts.investigate_background_objective import candidates, control_predictions


class BackgroundObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.levels = [[0., 10.], [0., 3.], [0., 2.], [0., 7.]]

    def test_mixed_radix_order(self):
        state, power = c.basis(self.levels)
        np.testing.assert_array_equal(state[3], [1, 1, 0, 0])
        np.testing.assert_array_equal(power[3], [10, 3, 0, 0])

    def test_static_matches_independent_loop(self):
        y = np.arange(-2, 30, .3)
        pred, ids, cost = c.infer_static(y, self.levels)
        powers = np.array([x[::-1] for x in itertools.product(*self.levels[::-1])])
        expected = np.array([min(range(len(powers)), key=lambda j: (v-powers[j].sum())**2) for v in y])
        np.testing.assert_array_equal(ids, expected)
        np.testing.assert_allclose(pred, powers[expected])
        np.testing.assert_allclose(cost, (y-pred.sum(axis=1))**2)

    def test_squared_absolute_same_argmin(self):
        _, p = c.basis(self.levels)
        y = np.arange(-4, 34, .37)
        np.testing.assert_array_equal(abs(y[:, None]-p.sum(axis=1)).argmin(axis=1), c.infer_static(y, self.levels)[1])

    def test_prior_matches_enumeration(self):
        priors = [[.9, .1], [.3, .7], [.8, .2], [.4, .6]]
        state, power = c.basis(self.levels); y = np.arange(24.)
        for scope, which in [("all", [0, 1, 2, 3]), ("appliances", [0, 1, 2]), ("background", [3])]:
            unary = sum(-np.log(np.array(priors[i])[state[:, i]]) for i in which)
            scores = (y[:, None]-power.sum(axis=1))**2+12*unary
            pred, idx, cost = c.infer_static(y, self.levels, priors, weight_w2=12, prior_scope=scope)
            np.testing.assert_array_equal(idx, scores.argmin(axis=1))
            np.testing.assert_allclose(cost, scores.min(axis=1))

    def test_zero_prior_preserves_baseline(self):
        y = np.arange(20.)
        np.testing.assert_array_equal(c.infer_static(y, self.levels)[1], c.infer_static(y, self.levels, weight_w2=0)[1])

    def test_smoothed_frequencies(self):
        values = np.zeros((10, 4))
        priors = c.fit_frequencies(values, self.levels)
        for p in priors: np.testing.assert_allclose(p, [11/12, 1/12])

    def test_mixture_one_component(self):
        y = np.arange(30.)
        mixture = {"weights": [1.], "sd_w": [2.], "centres_w": [7.]}
        pred, ids, cost = c.infer_marginal(y, self.levels[:3], mixture, [[.5, .5]]*3, 0)
        expected = c.infer_static(y-7, self.levels[:3])[0]
        np.testing.assert_allclose(pred[:, :3], expected)
        np.testing.assert_allclose(pred[:, 3], 7.)
        np.testing.assert_allclose(cost, .5*((y-pred.sum(axis=1))/2)**2+np.log(2*np.sqrt(2*np.pi)))

    def test_background_variance_floor(self):
        mixture = c.fit_background_mixture(np.array([0., 0., 7., 7.]), [0., 7.])
        np.testing.assert_array_equal(mixture["sd_w"], [1., 1.])

    def test_metric_pool_uses_block_weights(self):
        levels = [[0, 100], [0, 100], [0, 100], [0]]
        v1 = np.array([[30, 10, 10, 10.]])
        v2 = np.tile([60, 20, 20, 20.], (3, 1))
        rows = [c.metrics(v, np.zeros_like(v), levels, [50]*3) for v in [v1, v2]]
        self.assertEqual(c.pool(rows)["macro_mae_w"], 17.5)
        self.assertEqual(c.pool(rows)["blocks"], 4)

    def test_controls_exact_in_model_recovers_categories(self):
        levels = [[0., 1000], [0., 100.], [0., 10.], [0., 1.]]
        _, p = c.basis(levels)
        values = np.column_stack((p.sum(axis=1), p[:, :3]))
        result = control_predictions(values, levels)
        np.testing.assert_allclose(result["quantized_targets_quantized_background"], p)
        self.assertEqual(c.ambiguity(values, levels)["oracle_target_is_cost_minimizer"], len(p))

    def test_known_background_control(self):
        values = np.array([[53., 10., 3., 0.]])
        pred = control_predictions(values, self.levels)["known_background"]
        self.assertEqual(pred[0, 3], 40)
        self.assertEqual(pred[0, :3].sum(), 13)

    def test_candidate_grid_is_bounded_unique(self):
        grid = candidates()
        self.assertEqual(len(grid), 30)
        self.assertEqual(len({r["id"] for r in grid}), 30)
        self.assertEqual(sum(r["family"] == "temporal" for r in grid), 4)

    def test_invalid_prior_rejected(self):
        with self.assertRaises(ValueError): c.infer_static([1], self.levels, [[0, 1]]*4, weight_w2=1)

    def test_free_background_is_degenerate(self):
        values = np.array([[100., 10., 3., 2.]])
        self.assertEqual(c.ambiguity(values, self.levels)["free_nonnegative_background_feasible_targets"]["median"], 8)


if __name__ == "__main__": unittest.main()
