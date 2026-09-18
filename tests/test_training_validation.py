import itertools
import unittest
from unittest.mock import patch
import numpy as np
from quantum_nilm import training_validation as tv
from quantum_nilm.categorical_qaoa import prepare_categorical_problem, categorical_qaoa_probabilities


class TrainingValidationTests(unittest.TestCase):
    def test_batch_circuit_scalar_parity(self):
        levels = [[.1, .8, 400., 5000.], [4., 101., 440.], [.8, 1216.], [145., 426., 1296.]]
        y = [-2., 0., 680., 1290., 5200.]
        batch = tv.static_batch(y, levels)
        for angles in ([0,0], [33.05,2.01], [16*np.pi,np.pi], [1.,.5]):
            p = tv.probabilities(batch, angles)
            for i, power in enumerate(y):
                scalar = prepare_categorical_problem([power], levels, [0]*len(levels))
                np.testing.assert_allclose(batch["energies"][i], scalar.energies, atol=1e-8)
                self.assertAlmostEqual(batch["scales"][i], scalar.scale, places=5)
                np.testing.assert_allclose(p[i], categorical_qaoa_probabilities(scalar, [angles[0]], [angles[1]]), atol=1e-13)

    def test_mean_and_cvar_case_weighting(self):
        b = tv.static_batch([1., 100.], [[0,2], [0,3]])
        raw = np.array([[0,1,2,3], [3,3,0,1]], dtype=np.uint8)
        values = b["normalized"][np.arange(2)[:,None],raw]
        self.assertAlmostEqual(tv.sample_loss(b,raw,"mean"), values.mean())
        self.assertAlmostEqual(tv.sample_loss(b,raw,"cvar50"), np.sort(values,axis=1)[:,:2].mean())

    def test_rng_and_samples(self):
        p = np.array([[.1,.2,.3,.4], [1.,0.,0.,0.]])
        r = tv.sample_indices(p,np.random.default_rng(15),256)
        u = np.random.default_rng(15).random((2,256))
        np.testing.assert_array_equal(r[0], np.searchsorted(np.cumsum(p[0]), u[0], side="right"))
        self.assertTrue(np.all(r[1] == 0))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): tv.static_batch([np.nan], [[0,1]])
        with self.assertRaises(ValueError): tv.static_batch([1], [list(range(257))])
        with self.assertRaises(ValueError): tv.sample_indices(np.array([[.2,.2]]),np.random.default_rng(1),2)
        b = tv.static_batch([1],[[0,1]])
        with self.assertRaises(ValueError): tv.sample_loss(b,np.array([[2,0]]),"mean")
        with self.assertRaises(ValueError): tv.sample_loss(b,np.array([[0,0,1]]),"cvar50")

    def test_training_and_recheck_budget(self):
        b = tv.static_batch([1., 2.], [[0,1],[0,2]])
        a, raw = tv.train(b,"mean",[41,42],8*np.pi,budget=40,recheck_shots=32)
        c, other = tv.train(b,"cvar50",[41,42],8*np.pi,budget=40,recheck_shots=32)
        self.assertEqual(a["evaluations"],80)
        self.assertEqual(a["total_measurements"],2*(80*256+5*32))
        self.assertEqual(raw["search"].shape,(80,2,256))
        self.assertEqual(raw["recheck"].shape,(5,2,32))
        for x,y in zip(a["restarts"],c["restarts"]): self.assertEqual(x["initial_candidates"],y["initial_candidates"])
        self.assertEqual(len({tuple(a["trace"][i]["angles"]) for i in a["shortlist_indices"]}),5)
        chosen = int(np.argmin([r["loss"] for r in a["rechecks"]]))
        self.assertEqual(a["angles"],a["rechecks"][chosen]["angles"])
        self.assertEqual(a["precheck_training_index"],int(np.argmin([r["loss"] for r in a["trace"]])))

    def test_truth_not_optimizer_argument(self):
        import inspect
        self.assertNotIn("truth", inspect.signature(tv.train).parameters)
        self.assertNotIn("true_power", inspect.signature(tv.train).parameters)

    def test_decoder_exhaustive_ties(self):
        b = tv.static_batch([1.], [[0,1],[0,1]])
        truth = np.array([[1.,0.]])
        m = tv.expected_metrics(b,truth,mode="uniform",budgets=(2,))["2"]
        error = np.zeros(2); cost = 0.
        for a,c in itertools.product(range(4),repeat=2):
            i = min([a,c],key=lambda j:(b["energies"][0,j],j))
            error += abs(b["powers"][i]-truth[0])/16
            cost += b["energies"][0,i]/16
        np.testing.assert_allclose(m["absolute_error"][0],error)
        self.assertAlmostEqual(m["best_cost"][0],cost)

    def test_oracle_lower_bound_and_raw_truth(self):
        b = tv.static_batch([1.,9.], [[0,4],[0,5]])
        truth = np.array([[2.,.3],[3.,4.]])
        oracle = tv.expected_metrics(b,truth,mode="oracle")["oracle"]["absolute_error"]
        exact = tv.expected_metrics(b,truth,mode="exact")["exact"]["absolute_error"]
        self.assertTrue(np.all(exact>=oracle-1e-12))
        self.assertGreater(oracle.mean(),0)

    def test_background_excluded_from_mae(self):
        b = tv.static_batch([1.],[[0,1],[0,100]])
        result = tv.expected_metrics(b,[[1.]],mode="exact")["exact"]
        self.assertEqual(result["absolute_error"].shape,(1,1))
        self.assertEqual(result["absolute_error"][0,0],0)

    def test_unused_windows_guard_alignment(self):
        old = [(0,10),(50,60),(100,110),(150,160)]
        chosen = tv.unused_windows(0,200,old,3,duration=12,guard=8,block=2)
        for a,b in chosen:
            self.assertEqual(a%2,0); self.assertEqual(b-a,12)
            self.assertTrue(all(b<=x-8 or a>=y+8 for x,y in old))
        self.assertTrue(all(chosen[i][1]<=chosen[i+1][0] for i in range(len(chosen)-1)))

    def test_unused_windows_refuses_replacement_shortage(self):
        with self.assertRaises(ValueError): tv.unused_windows(0,100,[(0,100)],1,duration=10,guard=1)

    def test_timestamp_exposure_parser_ignores_partitions(self):
        from scripts.run_training_validation import recorded_ranges
        start = 1500000000
        d = {"partition":{"start":start,"end":start+1000000},"windows":[{"start_unix":start,"end_unix":start+86400}]}
        self.assertEqual(list(recorded_ranges(d,start,start+2000000)),[(start,start+86400)])


if __name__ == "__main__": unittest.main()
