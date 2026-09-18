"""Bounded independent-auditor fixtures; no account or submission calls."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_quantum_heldout_hardware.py"
SPEC = importlib.util.spec_from_file_location("hardware_audit", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class HardwareAuditTests(unittest.TestCase):
    def fixture(self):
        levels = [np.array([0., 10.]), np.array([0., 20.])]
        chunk = {"aggregate": [10.], "weights": [2], "block_start": 0, "block_stop": 2}
        row = {"raw_counts": {"0110": 3, "0101": 1, "0000": 2}, "states": [[1, 0]],
               "fallback_used": False,
               "sample_score": {"total_shots": 6, "feasible_shots": 4, "infeasible_shots": 2,
                    "feasible_fraction": 4 / 6, "classical_fallback_used": False,
                    "raw_counts_are_postselected": False, "best_states": [[1, 0]],
                    "best_feasible_bitstring": "0110", "best_feasible_objective": 0.,
                    "prediction_status": "feasible_sample_selected",
                    "mean_objective_conditional_on_feasibility": 50., "exact_objective": 0.,
                    "exact_optimum_shots": 3, "exact_optimum_probability_all_shots": .5}}
        return row, chunk, levels

    def test_independent_endian_onehot_and_observed_best(self):
        row, chunk, levels = self.fixture()
        np.testing.assert_array_equal(AUDIT.decode_raw_bitstring("0110", (2, 2), 1), [[1, 0]])
        self.assertIsNone(AUDIT.decode_raw_bitstring("1110", (2, 2), 1))
        checked = AUDIT.audit_prediction(row, chunk, levels, [3., 5.], None, 6)
        self.assertEqual(checked["feasible_shots"], 4)
        self.assertEqual(checked["invalid_shots"], 2)

    def test_unobserved_or_inferior_state_is_rejected(self):
        row, chunk, levels = self.fixture()
        row["states"] = [[0, 1]]
        with self.assertRaises(AUDIT.AuditFailure):
            AUDIT.audit_prediction(row, chunk, levels, [3., 5.], None, 6)

    def test_hidden_postselection_is_rejected(self):
        row, chunk, levels = self.fixture()
        row["sample_score"]["feasible_shots"] = 6
        with self.assertRaises(AUDIT.AuditFailure):
            AUDIT.audit_prediction(row, chunk, levels, [3., 5.], None, 6)

    def test_declared_all_invalid_fallback_is_audited_not_counted_as_quantum(self):
        row, chunk, levels = self.fixture()
        row["raw_counts"], row["states"], row["fallback_used"] = {"0000": 6}, [[0, 0]], True
        row["sample_score"].update(feasible_shots=0, infeasible_shots=6, feasible_fraction=0.,
            best_states=None, best_feasible_objective=None, prediction_status="no_feasible_samples",
            exact_optimum_shots=0, exact_optimum_probability_all_shots=0.)
        checked = AUDIT.audit_prediction(row, chunk, levels, [3., 5.], None, 6)
        self.assertTrue(checked["fallback"])
        row["states"] = [[1, 0]]
        with self.assertRaises(AUDIT.AuditFailure):
            AUDIT.audit_prediction(row, chunk, levels, [3., 5.], None, 6)

    def test_categorical_change_is_not_ordinal_distance(self):
        cost = AUDIT.direct_objective([[0], [2]], [0., 20.], [2., 3.], [np.array([0., 10., 20.])], [5.], [1])
        self.assertEqual(cost, 10.)

    def test_parameter_expansion_and_compiled_parameter_order(self):
        chunk = {"aggregate": [10.], "weights": [2.]}
        values = AUDIT.independent_cost_bindings(chunk, [np.array([0., 10.])], [3.], [0], 2.,
                                                 ["cost_angle[1]", "cost_angle[0]"])
        np.testing.assert_allclose(values, [400 / 203, 6 / 203], atol=1e-14)

    def test_accounted_usage_accepts_aliases_not_execution_time(self):
        self.assertEqual(AUDIT.accounted_usage({"usage": {"quantum_seconds": 3}}), 3.)
        self.assertEqual(AUDIT.accounted_usage({"usage": {"qpu_charge_time_seconds": 3}}), 3.)
        for value in ({"usage": {"quantum_seconds": 2, "qpu_charge_time_seconds": 3}},
                      {"circuits_execution_time_ns": 1e9}, {"usage": {"quantum_seconds": -1}}):
            with self.assertRaises(AUDIT.AuditFailure):
                AUDIT.accounted_usage(value)

    def test_window_coverage_and_gap_reset(self):
        window = {"window": {"id": "test-000", "start_unix": 100, "end_unix": 250}, "blocks": 4,
                  "chunks": [{"chunk": 0, "run": 0, "reset": True, "aggregate": [10.],
                              "weights": [2], "block_start": 0, "block_stop": 2, "start_unix": 100},
                             {"chunk": 1, "run": 1, "reset": True, "aggregate": [20.],
                              "weights": [2], "block_start": 2, "block_stop": 4, "start_unix": 190}]}
        self.assertEqual(AUDIT.check_schedule([window])["blocks"], 4)
        wrong = copy.deepcopy(window)
        wrong["chunks"][1]["reset"] = False
        with self.assertRaises(AUDIT.AuditFailure):
            AUDIT.check_schedule([wrong])
        wrong = copy.deepcopy(window)
        wrong["chunks"][1]["block_start"] = 1
        with self.assertRaises(AUDIT.AuditFailure):
            AUDIT.check_schedule([wrong])

    def test_code_revision_chain_is_anchored_and_tamper_evident(self):
        with TemporaryDirectory() as directory:
            repo = Path(directory)
            folder = repo / "hardware"
            folder.mkdir()
            (repo / "scripts").mkdir()
            old, new = "# frozen initial runner\n", "# accounting compatibility revision\n"
            target = repo / "scripts/run_quantum_heldout_ibm.py"
            target.write_text(new)
            old_hash = hashlib.sha256(old.encode()).hexdigest()
            new_hash = hashlib.sha256(new.encode()).hexdigest()
            name = "scripts/run_quantum_heldout_ibm.py"
            plan = {"code_sha256": {name: old_hash}}
            (folder / "execution_code_initial.py").write_text(old)
            revision = {"revision": 1, "plan_sha256": "planhash",
                        "previous_code_sha256": {name: old_hash}, "code_sha256": {name: new_hash},
                        "initial_code_copy": "execution_code_initial.py", "initial_code_copy_sha256": old_hash}
            (folder / "code_revision_001.json").write_text(json.dumps(revision))
            checked = AUDIT.verify_code_hashes(repo, folder, plan, "planhash")
            self.assertTrue(checked["revision_chain_present"])
            target.write_text("# unrecorded change\n")
            with self.assertRaises(AUDIT.AuditFailure):
                AUDIT.verify_code_hashes(repo, folder, plan, "planhash")
if __name__ == "__main__":
    unittest.main()
