"""Synthetic-only tests for observed-effective-shot conditional diagnostics."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("effective_shots_tests", SCRIPTS / "run_effective_shot_diagnostic.py")
diagnostic = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC.loader.exec_module(diagnostic)
finally:
    sys.path.pop(0)


class EffectiveShotTests(unittest.TestCase):
    @staticmethod
    def chunks():
        return [{"chunk": i, "reset": reset, "aggregate": [reading], "weights": [1],
                 "block_start": i, "block_stop": i + 1}
                for i, (reset, reading) in enumerate(((True, 10.), (False, 0.), (True, 0.)))]

    @staticmethod
    def budgets(values):
        return [{"chunk": i, "effective_shots": n, "hardware_raw_shots": 256} for i, n in enumerate(values)]

    def test_raw_onehot_validity_counts_all_zero_and_multihot_shots(self):
        valid, total, histograms = diagnostic.count_raw_feasibility(
            {"00101": 3, "00000": 2, "11111": 5}, [2, 3])
        self.assertEqual((valid, total), (3, 10))
        self.assertEqual(dict(histograms[0]), {1: 3, 0: 2, 2: 5})
        self.assertEqual(dict(histograms[1]), {1: 3, 0: 2, 3: 5})

    def test_raw_invalid_count_or_encoding_rejected(self):
        for raw in ({}, {"01": 0}, {"01": True}, {"01": 1.5}, {"0x": 1}, {"1": 2}):
            with self.assertRaises(ValueError):
                diagnostic.count_raw_feasibility(raw, [2])

    def test_zero_budget_at_reset_is_lowest_power_not_exact_or_sampled(self):
        with patch.object(diagnostic, "categorical_qaoa_probabilities", side_effect=AssertionError("No samples at zero budget")):
            records, _ = diagnostic.replay_chunks(self.chunks()[:1], self.budgets([0]),
                [np.array([10., 0.])], np.array([5.]), [1.], [.3], "qaoa_effective", 7)
        row = records[0]
        self.assertEqual(row["states"], [[1]])
        self.assertIsNone(row["chosen_index"])
        self.assertTrue(row["fallback_used"])
        self.assertEqual(row["raw_feasible_counts"], [])
        self.assertEqual(row["output_objective"], 100.)
        self.assertEqual(row["conditional_exact_energy"], 0.)

    def test_one_sample_then_zero_carries_own_state_and_reset_breaks_chain(self):
        with patch.object(diagnostic, "categorical_qaoa_probabilities", return_value=np.array([0., 1.])):
            records, _ = diagnostic.replay_chunks(self.chunks(), self.budgets([1, 0, 0]),
                [np.array([0., 10.])], np.array([120.]), [1.], [.3], "qaoa_effective", 7)
        self.assertEqual([r["states"] for r in records], [[[1]], [[1]], [[0]]])
        self.assertEqual([r["previous_states"] for r in records], [None, [1], None])
        self.assertEqual([r["fallback_used"] for r in records], [False, True, True])
        self.assertEqual(records[0]["raw_feasible_counts"], [[1, 1]])
        self.assertEqual(records[1]["output_objective"], 100.)

    def test_one_sample_never_injects_unsampled_optimum(self):
        with patch.object(diagnostic, "categorical_qaoa_probabilities", return_value=np.array([1., 0.])):
            records, _ = diagnostic.replay_chunks(self.chunks()[:1], self.budgets([1]),
                [np.array([0., 10.])], np.array([0.]), [1.], [.3], "qaoa_effective", 7)
        row = records[0]
        self.assertEqual(row["states"], [[0]])
        self.assertEqual(row["chosen_index"], 0)
        self.assertEqual(row["output_objective"], 100.)
        self.assertEqual(row["conditional_exact_energy"], 0.)

    def test_variable_budgets_and_reproducibility(self):
        args = (self.chunks(), self.budgets([0, 1, 7]), [np.array([0., 10.])],
                np.array([5.]), [1.], [.3], "uniform_effective", 11)
        first, _ = diagnostic.replay_chunks(*args)
        second, _ = diagnostic.replay_chunks(*args)
        self.assertEqual(first, second)
        self.assertEqual([sum(n for _, n in row["raw_feasible_counts"]) for row in first], [0, 1, 7])
        prediction = diagnostic.expand_predictions({"blocks": 3, "chunks": self.chunks()}, first, [np.array([0., 10.])])
        self.assertEqual(prediction.shape, (3, 1))
        self.assertTrue(np.all(np.isfinite(prediction)))

    def test_bad_budget_coverage_order_and_metadata_rejected(self):
        cases = [self.budgets([0]), self.budgets([-1, 1, 1]), self.budgets([True, 1, 1]),
                 self.budgets([1.5, 1, 1]), self.budgets([257, 1, 1])]
        out_of_order = self.budgets([1, 1, 1]); out_of_order[0]["chunk"] = 1; cases.append(out_of_order)
        wrong_reset = self.budgets([1, 1, 1]); wrong_reset[0]["reset"] = False; cases.append(wrong_reset)
        wrong_shape = self.budgets([1, 1, 1]); wrong_shape[0]["intervals"] = 2; cases.append(wrong_shape)
        for budgets in cases:
            with self.assertRaises(ValueError):
                diagnostic.replay_chunks(self.chunks(), budgets, [np.array([0., 10.])],
                    np.array([5.]), [1.], [.3], "uniform_effective", 7)

    def test_invalid_method_rejected(self):
        with self.assertRaises(ValueError):
            diagnostic.replay_chunks(self.chunks(), self.budgets([1, 1, 1]), [np.array([0., 10.])],
                np.array([5.]), [1.], [.3], "not-a-method", 7)

    def test_provenance_detects_source_and_frozen_artifact_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / "source.json"; source.write_text('{"value":1}')
            hashes = {str(source): diagnostic.sha(source)}
            diagnostic.check_hashes(hashes)
            source.write_text('{"value":2}')
            with self.assertRaisesRegex(RuntimeError, "hash changed"):
                diagnostic.check_hashes(hashes)
            protocol = {"source_sha256": {}, "code_sha256": {}}
            diagnostic.save_json(root / "protocol.json", protocol)
            diagnostic.save_json(root / "frozen_artifacts.json", {"protocol.json": diagnostic.sha(root / "protocol.json")})
            self.assertEqual(diagnostic.verify_frozen(root), protocol)
            (root / "protocol.json").write_text(json.dumps({**protocol, "changed": True}))
            with self.assertRaisesRegex(RuntimeError, "Frozen diagnostic artifact"):
                diagnostic.verify_frozen(root)

    def test_original_analysis_wording_handles_one_hardware_campaign(self):
        # Reuse existing JSON solely to render in memory, without rewriting it.
        path = SCRIPTS.parent / "results/quantum_heldout/analysis_complete/analysis.json"
        if not path.exists():
            self.skipTest("Optional archived full analysis is unavailable")
        from summarize_quantum_heldout import make_markdown
        before = diagnostic.sha(path)
        markdown = make_markdown(json.loads(path.read_text()))
        self.assertIn("1 hardware campaign", markdown)
        self.assertIn("Ideal simulated QAOA improves", markdown)
        self.assertNotIn("after hardware scoring completes", markdown)
        self.assertEqual(before, diagnostic.sha(path))


if __name__ == "__main__":
    unittest.main()
