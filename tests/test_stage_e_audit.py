"""Independent-auditor helper regression tests, using only synthetic cases."""
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.audit_stage_e_mip import (audit_solver, audit_trace, basic_metrics, candidate_key, check_compression,
                                      dense_oracle, direct_energy, paired_interval,
                                      raw_appliance_metrics, reconstruct_runs)


class StageEAuditTests(unittest.TestCase):
    def test_dense_oracle_matches_enumeration_signed_and_nonordinal(self):
        levels = [np.array([-3., 0., 9.]), np.array([-2., 5.])]
        measured, weights, penalties = [2., 8., -1.], [2, 1, 4], [3., 1.]
        candidates = itertools.product(list(itertools.product(range(3), range(2))), repeat=3)
        energies = [direct_energy(measured, weights, path, levels, penalties)["energy"] for path in candidates]
        self.assertAlmostEqual(dense_oracle(measured, weights, levels, penalties)["optimum"], min(energies))

    def test_direct_energy_rejects_noninteger_and_out_of_range(self):
        for states in ([[0.0]], [[2]], [[-1]]):
            with self.assertRaises(ValueError):
                direct_energy([0.], [1], states, [[0., 1.]], [0.])

    def test_reconstruction_preserves_resets(self):
        window = {"blocks": 4, "chunks": [
            {"chunk": 0, "block_start": 0, "block_stop": 3, "run": 0, "reset": True, "weights": [1, 2], "aggregate": [0., 1.]},
            {"chunk": 1, "block_start": 3, "block_stop": 4, "run": 1, "reset": True, "weights": [1], "aggregate": [3.]},
        ]}
        self.assertEqual(len(reconstruct_runs(window)), 2)
        window["chunks"][1]["reset"] = False
        with self.assertRaises(ValueError):
            reconstruct_runs(window)

    def test_independent_compression_and_gaps(self):
        window = {"blocks": 6, "chunks": [
            {"chunk": 0, "block_start": 0, "block_stop": 4, "run": 0, "reset": True,
             "weights": [2, 2], "aggregate": [1., 7.]},
            {"chunk": 1, "block_start": 4, "block_stop": 6, "run": 1, "reset": True,
             "weights": [2], "aggregate": [0.]},
        ]}
        self.assertEqual(len(check_compression(window, [0, 30, 60, 90, 150, 180], [0., 2., 7., 7., 0., 0.], 4.)), 2)
        with self.assertRaises(ValueError):
            check_compression(window, [0, 30, 60, 90, 120, 150], [0., 2., 7., 7., 0., 0.], 4.)

    def test_raw_metrics_and_ratio_bootstrap(self):
        a = basic_metrics(np.zeros((2, 3)), np.ones((2, 3)))
        b = basic_metrics(np.zeros((1, 3)), np.full((1, 3), 4.))
        zeroa = basic_metrics(np.zeros((2, 3)), np.zeros((2, 3)))
        zerob = basic_metrics(np.zeros((1, 3)), np.zeros((1, 3)))
        result = paired_interval({"a": a, "b": b}, {"a": zeroa, "b": zerob})
        self.assertEqual(result["difference_w"], 2.)
        self.assertEqual(result["paired_window_bootstrap_95_interval_w"], [1., 4.])

    def test_proxy_events_reset_and_undefined_zero_energy(self):
        truth = np.array([[0., 0., 0.], [2., 0., 0.], [0., 0., 0.]])
        result = raw_appliance_metrics(truth, truth, [0, 30, 90], [1., 1., 1.])
        self.assertEqual(result["dryr"]["event_counts"], {"tp": 1, "fp": 0, "fn": 0})
        self.assertIsNone(result["frdg"]["signal_aggregate_error"])
        self.assertIsNone(result["frdg"]["state"]["f1"])

    def test_tuning_rank_prefers_coverage_then_optimality(self):
        complete = [{"incumbent": True, "solver_optimal": False, "certified_relative_gap": .2, "formulation_plus_solve_time_s": 3.}]
        absent = [{"incumbent": False, "solver_optimal": False, "certified_relative_gap": None, "formulation_plus_solve_time_s": 0.}]
        self.assertLess(candidate_key(complete, False), candidate_key(absent, True))
        self.assertLess(candidate_key(complete, True), candidate_key(complete, False))

    def solver_fixture(self):
        direct = {"energy": 10.}
        oracle = {"optimum": 10., "objective_offset": 2., "objective_scale": 4., "joint_states": 2}
        solver = {"objective_offset": 2., "objective_scale": 4., "states_per_segment": 2,
                  "raw_solver_objective": 2., "raw_solver_dual_bound": 2., "solver_objective": 10., "dual_bound": 10.,
                  "energy": 10., "incumbent_feasible": True, "status_code": 0, "success": True,
                  "absolute_gap": 0., "relative_gap": 0., "bound_validation": "passed",
                  "max_integrality_violation": 0., "max_bound_violation": 0., "max_constraint_violation": 0.}
        return solver, direct, oracle

    def test_restored_bounds_and_equal_cost_not_path_equality(self):
        solver, direct, oracle = self.solver_fixture()
        self.assertTrue(audit_solver(solver, direct, oracle)["incumbent"])
        solver["dual_bound"] = 9.
        with self.assertRaises(ValueError):
            audit_solver(solver, direct, oracle)

    def test_invalid_lower_bound_and_negative_optimality_gap_rejected(self):
        solver, direct, oracle = self.solver_fixture()
        solver["dual_bound"], solver["raw_solver_dual_bound"] = 12., 2.5
        with self.assertRaises(ValueError):
            audit_solver(solver, direct, oracle)
        solver, direct, oracle = self.solver_fixture()
        direct["energy"] = 9.
        with self.assertRaises(ValueError):
            audit_solver(solver, direct, oracle)

    def test_failure_is_not_optimal_and_no_fake_incumbent(self):
        solver, _, oracle = self.solver_fixture()
        solver.update(status_code=1, success=False, energy=None, states=None, incumbent_feasible=False)
        self.assertFalse(audit_solver(solver, None, oracle)["incumbent"])
        solver["status_code"] = 0
        with self.assertRaises(ValueError):
            audit_solver(solver, None, oracle)

    def test_missing_bound_retains_incumbent_without_gap_certificate(self):
        solver, direct, oracle = self.solver_fixture()
        solver.update(dual_bound=None, raw_solver_dual_bound=None, absolute_gap=None, relative_gap=None,
                      bound_validation="unavailable")
        self.assertTrue(audit_solver(solver, direct, oracle)["incumbent"])

    def test_small_real_solver_trace_and_checkpoint_audited(self):
        # Production code creates only a synthetic fixture. The audit checks
        # it via its independent oracle, direct objective and archive reader.
        from scripts.run_stage_e_mip import infer_window, rederive_window
        levels, penalties = [np.array([0., 10.]), np.array([0., 4.])], [1., 2.]
        loaded = {"timestamps": np.array([0, 30, 60]), "values": np.array([[0.], [6.], [8.]]),
                  "source_window_sha256": "synthetic"}
        window = rederive_window({"id": "validation-synthetic"}, loaded, 4.)
        binding = {"protocol_sha256": "synthetic", "phase": "validation", "candidate": "presolve_on"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "tuning/run_records/presolve_on/validation-synthetic"
            trace, expected, _, _ = infer_window(window, loaded, levels, penalties, 4., "milp",
                presolve=True, time_limit_s=3., checkpoint=folder, binding=binding)
            runs = reconstruct_runs(window)
            oracles = [dense_oracle(r["aggregate"], r["weights"], levels, penalties) for r in runs]
            result = audit_trace(root, window, trace, levels, penalties, oracles, binding,
                                 label="presolve_on", validation=True)
            self.assertTrue(np.array_equal(result["prediction"], expected))
            trace["covered_blocks"] -= 1
            with self.assertRaises(ValueError):
                audit_trace(root, window, trace, levels, penalties, oracles, binding,
                            label="presolve_on", validation=True)


if __name__ == "__main__":
    unittest.main()
