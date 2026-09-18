"""Small, deterministic tests of Stage E selection, provenance and coverage."""
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_stage_e_mip as runner
from quantum_nilm.stage_e_mip import solve_categorical_milp


def fixture(mains=(0., 2., 8.), timestamps=(0, 30, 60), threshold=5.):
    metadata = {"id": "test_000", "split": "test", "start_unix": 0, "end_unix": 90}
    loaded = {"timestamps": np.asarray(timestamps), "values": np.asarray(mains)[:, None],
              "source_window_sha256": "source-test"}
    return runner.rederive_window(metadata, loaded, threshold), loaded


def candidate(presolve=True, missing=0, nonoptimal=0, gap=0., elapsed=1.):
    return {"presolve": presolve, "no_incumbent_runs": missing,
            "not_solver_optimal_runs": nonoptimal, "sum_certified_relative_gap": gap,
            "formulation_plus_solve_time_s": elapsed}


def tiny_archive(output):
    model = {"models": {"multistate": {"levels_w": [[0., 10.], [0., 1.], [0., 2.], [0., 3.]],
              "ranges_w": [10., 1., 2., 3.]}},
              "selected_parameters": {"multistate_compressed": {"family": "multistate", "compressed": True, "rho": .1}}}
    protocol = {"event_threshold_w": 5., "validation_time_limit_s": 15.,
                "test_time_limit_s": 30., "mip_rel_gap": 1e-8, "blocks": 6,
                "limitations": ["unit-test cohort only"]}
    windows, loaded_by_id = [], {}
    for i in range(2):
        window, loaded = fixture()
        window["window"]["id"] = f"test_{i:03d}"
        windows.append(window)
        loaded_by_id[window["window"]["id"]] = loaded
    validation = []
    for i in range(3):
        window, loaded = fixture()
        window["window"].update({"id": f"validation_{i:03d}", "split": "validation"})
        validation.append(window)
        runner.save_npz(output / "validation_data" / f"validation_{i:03d}.npz",
                        timestamps=loaded["timestamps"], mains_w=loaded["values"][:, 0])
    runner.save(output / "validation_inputs.json", validation)
    frozen = {"model": model, "windows": windows, "source_quality": {},
              "config": {"state_proxy_thresholds_w": [5., .5, 1.]}}
    def source_reader(config, quality, metadata, channels):
        loaded = deepcopy(loaded_by_id[metadata["id"]])
        if len(channels) > 1:
            for arm in runner.ARMS:
                if not (output / "traces" / arm / f"{metadata['id']}.json").exists() \
                        or not (output / "predictions" / arm / f"{metadata['id']}.npz").exists():
                    raise AssertionError("References loaded before both arms' predictions were saved")
            loaded["values"] = np.c_[loaded["values"], np.zeros((len(loaded["timestamps"]), 3))]
        return loaded
    return protocol, frozen, source_reader


class StageERunnerTests(unittest.TestCase):
    def test_environment_versions_are_enforced_not_merely_reported(self):
        versions = {"python": runner.platform.python_version(), "numpy": np.__version__,
                    "scipy": runner.scipy.__version__}
        runner.check_environment(versions)
        for name in versions:
            with self.assertRaisesRegex(ValueError, "versions changed"):
                runner.check_environment({**versions, name: "different"})

    def test_validation_selection_is_chronological_not_list_or_quality_order(self):
        windows = [{"id": str(i), "split": split, "start_unix": time}
                   for i, split, time in [(0, "test", 0), (1, "val", 60),
                                          (2, "validation", 10), (3, "val", 40),
                                          (4, "val", 90), (5, "train", 1)]]
        self.assertEqual([w["id"] for w in runner.validation_selection(windows)], ["2", "3", "1"])
        with self.assertRaisesRegex(ValueError, "Three"):
            runner.validation_selection(windows[:2])

    def test_selection_prioritizes_coverage_then_optimality_then_gap_then_time(self):
        true = candidate(True, missing=1, nonoptimal=1, gap=None, elapsed=.1)
        false = candidate(False, nonoptimal=10, gap=1., elapsed=10.)
        self.assertIs(runner.select_candidate([true, false]), false)
        true = candidate(True, nonoptimal=1, gap=0., elapsed=.1)
        false = candidate(False, gap=.9, elapsed=10.)
        self.assertIs(runner.select_candidate([true, false]), false)
        true, false = candidate(True, gap=.1, elapsed=.1), candidate(False, elapsed=10.)
        self.assertIs(runner.select_candidate([true, false]), false)
        true, false = candidate(True, elapsed=2.), candidate(False, elapsed=1.)
        self.assertIs(runner.select_candidate([true, false]), false)

    def test_selection_final_tie_prefers_on_and_rejects_duplicate_candidates(self):
        self.assertTrue(runner.select_candidate([candidate(False), candidate(True)])["presolve"])
        with self.assertRaises(ValueError):
            runner.select_candidate([candidate(), candidate()])

    def test_archive_writes_are_exclusive_and_nonfinite_values_leave_no_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.json"
            with self.assertRaises(ValueError):
                runner.save(path, {"gap": float("inf")})
            self.assertFalse(path.exists())
            runner.save(path, {"gap": None, "array": np.array([1, 2])})
            self.assertEqual(runner.read(path), {"gap": None, "array": [1, 2]})
            with self.assertRaises(FileExistsError):
                runner.save(path, {"gap": 2})
            array_path = Path(directory) / "values.npz"
            runner.save_npz(array_path, values=np.array([1.]))
            with self.assertRaises(FileExistsError):
                runner.save_npz(array_path, values=np.array([2.]))

    def test_compression_rejects_reference_columns(self):
        window, loaded = fixture()
        loaded["values"] = np.c_[loaded["values"], np.zeros(3)]
        with self.assertRaisesRegex(ValueError, "mains-only"):
            runner.rederive_window(window["window"], loaded, 5.)

    def test_fresh_compression_must_equal_frozen_inputs(self):
        window, loaded = fixture()
        window["chunks"][0]["aggregate"][0] += 1.
        with self.assertRaisesRegex(ValueError, "compression differs"):
            runner.infer_window(window, loaded, [np.array([0., 10.])], np.array([1.]), 5., "exact_dp")

    def test_exact_and_milp_expand_same_complete_small_objective(self):
        window, loaded = fixture()
        for arm in runner.ARMS:
            trace, prediction, means, mask = runner.infer_window(
                window, loaded, [np.array([0., 10.])], np.array([1.]), 5., arm)
            self.assertTrue(trace["complete"])
            self.assertEqual(trace["covered_blocks"], 3)
            np.testing.assert_array_equal(means, [1., 1., 8.])
            np.testing.assert_array_equal(prediction[:, 0], [0., 0., 10.])
            self.assertTrue(mask.all())
            self.assertAlmostEqual(trace["runs"][0]["common_segment_objective"], 7.)
            self.assertAlmostEqual(trace["inference_wall_time_s"], trace["compression_wall_time_s"]
                                   + trace["expansion_wall_time_s"]
                                   + sum(r["solver_call_wall_time_s"] for r in trace["runs"]))

    def test_no_incumbent_retains_failure_and_nan_predictions_without_fallback(self):
        window, loaded = fixture()
        base = solve_categorical_milp([1., 8.], [np.array([0., 10.])], [1.], [2, 1])
        absent = replace(base, states=None, predicted_power=None, energy=None,
                         incumbent_feasible=False, status_code=1, success=False,
                         status="limit_reached", message="no incumbent")
        with patch.object(runner, "solve_categorical_milp", return_value=absent), \
             patch.object(runner, "solve_multistate_temporal_exact", side_effect=AssertionError("No DP fallback")):
            trace, prediction, _, mask = runner.infer_window(
                window, loaded, [np.array([0., 10.])], np.array([1.]), 5., "milp")
        self.assertFalse(trace["complete"])
        self.assertEqual(trace["covered_blocks"], 0)
        self.assertFalse(mask.any())
        self.assertTrue(np.isnan(prediction).all())
        self.assertIsNone(trace["runs"][0]["states"])
        self.assertNotIn("common_segment_objective", trace["runs"][0])

    def test_valid_timeout_is_retained_but_not_solver_optimal(self):
        window, loaded = fixture()
        base = solve_categorical_milp([1., 8.], [np.array([0., 10.])], [1.], [2, 1])
        limited = replace(base, status_code=1, success=False, status="limit_reached")
        with patch.object(runner, "solve_categorical_milp", return_value=limited):
            trace, _, _, _ = runner.infer_window(
                window, loaded, [np.array([0., 10.])], np.array([1.]), 5., "milp")
        self.assertTrue(trace["complete"])
        self.assertFalse(trace["runs"][0]["solver_certified_optimal"])

    def test_gap_reset_is_solved_independently(self):
        window, loaded = fixture((0., 10.), (0, 90), threshold=5.)
        trace, prediction, _, _ = runner.infer_window(
            window, loaded, [np.array([0., 10.])], np.array([10000.]), 5., "milp")
        self.assertEqual(len(trace["runs"]), 2)
        np.testing.assert_array_equal(prediction[:, 0], [0., 10.])
        self.assertEqual(sum(r["common_transition"] for r in trace["runs"]), 0.)

    def test_bad_or_missing_bound_is_not_counted_solver_certified(self):
        window, loaded = fixture()
        base = solve_categorical_milp([1., 8.], [np.array([0., 10.])], [1.], [2, 1])
        for bad in (replace(base, bound_validation="dual_bound_exceeds_incumbent"),
                    replace(base, dual_bound=None, bound_validation="unavailable")):
            with patch.object(runner, "solve_categorical_milp", return_value=bad):
                trace, _, _, _ = runner.infer_window(window, loaded,
                    [np.array([0., 10.])], np.array([1.]), 5., "milp")
            self.assertTrue(trace["complete"])
            self.assertFalse(trace["runs"][0]["solver_certified_optimal"])

    def test_checkpoint_resume_never_repeats_a_completed_solver_call(self):
        window, loaded = fixture()
        with tempfile.TemporaryDirectory() as directory:
            binding = {"protocol_sha256": "test", "phase": "test"}
            args = (window, loaded, [np.array([0., 10.])], np.array([1.]), 5., "exact_dp")
            first = runner.infer_window(*args, checkpoint=directory, binding=binding)
            with patch.object(runner, "solve_multistate_temporal_exact", side_effect=AssertionError("Repeated solve")):
                second = runner.infer_window(*args, checkpoint=directory, binding=binding)
            self.assertEqual(first[0]["runs"], second[0]["runs"])
            np.testing.assert_array_equal(first[1], second[1])
            with self.assertRaisesRegex(ValueError, "provenance"):
                runner.infer_window(*args, checkpoint=directory, binding={"protocol_sha256": "changed"})

    def test_unfinished_solve_intent_requires_reconciliation(self):
        window, loaded = fixture()
        with tempfile.TemporaryDirectory() as directory:
            runner.save(Path(directory) / "run_000.intent.json", {"unfinished": True})
            with self.assertRaisesRegex(ValueError, "reconciliation"):
                runner.infer_window(window, loaded, [np.array([0., 10.])], np.array([1.]), 5.,
                                    "exact_dp", checkpoint=directory)

    def test_trace_validation_detects_run_checkpoint_mutation(self):
        window, loaded = fixture()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoints"
            binding = {"protocol_sha256": "test"}
            trace, _, _, _ = runner.infer_window(window, loaded, [np.array([0., 10.])],
                     np.array([1.]), 5., "exact_dp", checkpoint=checkpoint, binding=binding)
            path = Path(directory) / "trace.json"
            runner.save(path, trace)
            self.assertEqual(runner.checked_trace(path, checkpoint, binding), trace)
            # An additional field changes the file hash without affecting JSON validity.
            changed = runner.read(checkpoint / "run_000.json")
            changed["tampered"] = True
            (checkpoint / "run_000.json").write_text(__import__("json").dumps(changed))
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                runner.checked_trace(path, checkpoint, binding)

    def test_tiny_tune_run_summary_and_resume_preserve_predictions_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            protocol, frozen, source_reader = tiny_archive(output)
            with patch.object(runner, "check_frozen", return_value=(protocol, frozen, "unit-protocol")), \
                 patch.object(runner, "checked_window", side_effect=source_reader), \
                 patch("builtins.print"):
                selection = runner.tune(output)
                self.assertIn(selection["presolve"], (True, False))
                result = runner.run(output)
                self.assertEqual(result["status"], "complete")
                self.assertTrue(result["full_coverage"])
                self.assertEqual(result["arms"]["milp"]["covered_blocks"], 6)
                self.assertIsNotNone(result["paired_milp_minus_dp_macro_mae"])
                with patch.object(runner, "solve_categorical_milp", side_effect=AssertionError("Repeated MILP")), \
                     patch.object(runner, "solve_multistate_temporal_exact", side_effect=AssertionError("Repeated DP")):
                    self.assertEqual(runner.run(output), result)

    def test_failure_pipeline_retains_all_windows_without_full_score_or_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            protocol, frozen, source_reader = tiny_archive(output)
            with patch.object(runner, "check_frozen", return_value=(protocol, frozen, "unit-protocol")), \
                 patch.object(runner, "checked_window", side_effect=source_reader), patch("builtins.print"):
                runner.tune(output)
                base = solve_categorical_milp([1., 8.], [np.array(x) for x in frozen["model"]["models"]["multistate"]["levels_w"]],
                                             [.1, .1, .1, .1], [2, 1])
                absent = replace(base, states=None, predicted_power=None, energy=None,
                                 incumbent_feasible=False, status_code=1, success=False,
                                 status="limit_reached", message="no incumbent")
                with patch.object(runner, "solve_categorical_milp", return_value=absent):
                    result = runner.run(output)
                self.assertEqual(result["status"], "all_attempted_incomplete_prediction_coverage")
                self.assertFalse(result["full_coverage"])
                self.assertEqual(result["arms"]["milp"]["scheduled_windows"], 2)
                self.assertEqual(result["arms"]["milp"]["covered_blocks"], 0)
                self.assertIsNone(result["arms"]["milp"]["full_coverage_metrics"])
                self.assertIsNone(result["paired_milp_minus_dp_macro_mae"])
                self.assertTrue(result["arms"]["exact_dp"]["full_coverage"])


if __name__ == "__main__":
    unittest.main()
