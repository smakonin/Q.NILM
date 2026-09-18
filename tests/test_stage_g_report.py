"""Synthetic metadata fixtures only; never read REFIT measurement sources."""
import copy
import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / "scripts/report_stage_g_refit.py"
SPEC = importlib.util.spec_from_file_location("stage_g_report", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture():
    train = {"id": "train-0", "start_unix": 0, "end_unix": 60}
    test = [{"id": "test-0", "start_unix": 120, "end_unix": 240},
            {"id": "test-1", "start_unix": 300, "end_unix": 360}]
    channels = ["Appliance2", "Appliance1", "Appliance3"]
    protocol = {"homes": [{"house": 7, "channels": channels, "manifest": {"splits": {
        "train": {"windows": [train]}, "test": {"windows": test}}}}],
        "methods": list(MODULE.METHODS), "requested_counts": [4, 3, 2, 3],
        "target_order": ["washing", "cooling", "dishwasher"], "shots": 256,
        "seeds": [1907, 2907, 3907], "limitations": ["synthetic fixture metadata"]}
    models = {"protocol_sha256": "protocol", "homes": {"7": {
        "quality": [{"window": train, "valid_blocks": 2}], "training_blocks": 2,
        "actual_counts": [4, 3, 2, 3], "levels": [[0, 1, 2, 3], [0, 1, 2], [0, 1], [0, 1, 2]]}}}
    quality = []
    for index, window in enumerate(test):
        empty = index == 1
        q = {"valid_blocks": 0 if empty else 1, "expected_complete_blocks": 2 if empty else 4,
             "partial_final_seconds": 0, "valid_covered_seconds": 0 if empty else 82,
             "supported_invalid_seconds": 0 if empty else 8, "unsupported_large_gap_seconds": 60 if empty else 30,
             "uncovered_boundary_seconds": 0, "rows_issues": 0 if empty else 1,
             "rows_nonfinite": 0, "rows_negative": 0, "rows_invalid_schema": 0,
             "valid_source_rows": 0 if empty else 12, "all_selected_appliances_zero_source_rows": 0 if empty else 8,
             "all_selected_appliances_zero_retained_blocks": 0,
             "zero_retained_blocks_by_channel": {"Aggregate": 0, "Appliance2": 0 if empty else 1,
                                                  "Appliance1": 0, "Appliance3": 0 if empty else 1}}
        quality.append({"house": 7, "window": window, "quality": q, "valid_blocks": q["valid_blocks"], "source_window_sha256": f"window-{index}"})
    metrics = {method: {"macro_mae_w": value, "absolute_error_sum_w": 3 * value, "blocks": 1, "windows": 1}
               for method, value in zip(MODULE.METHODS, (10., 11., 9., 8., 12.))}
    days, cross = [], []
    for method in MODULE.METHODS[1:]:
        difference = 10 - metrics[method]["macro_mae_w"]
        days.append({"house": 7, "left": "qaoa_ideal", "right": method, "difference_w": difference,
                     "descriptive_95_interval_w": [difference, difference], "days": 1, "replicates": 2000})
        cross.append({"left": "qaoa_ideal", "right": method, "difference_w": difference,
                      "descriptive_95_interval_w": [difference, difference], "homes": 1, "replicates": 10000})
    summary = {"status": "complete", "protocol_sha256": "protocol", "nonevaluable_homes": [],
               "per_home": {"7": metrics}, "paired_day_comparisons": days, "paired_home_comparisons": cross,
               "overall": {method: {"equal_home_macro_mae_w": value["macro_mae_w"], "block_pooled_macro_mae_w": value["macro_mae_w"],
                                    "evaluable_homes": 1, "blocks": 1, "windows": 1} for method, value in metrics.items()},
               "coverage": {"7": {"selected_windows": 2, "nonempty_windows": 1, "valid_blocks": 1, "expected_blocks": 6,
                                  "issues_rows": 1, "all_appliances_zero_retained_blocks": 0, "empty_windows": ["test-1"],
                                  "large_gap_seconds": 90, "invalid_supported_seconds": 8}}}
    provenance = {"sha256": {"protocol.json": "protocol", "summary.json": "summary", "frozen_models.json": "models", "test_quality.json": "quality"}}
    return protocol, models, quality, summary, provenance


class StageGReportTests(unittest.TestCase):
    def test_support_loss_decomposition_and_all_window_retention(self):
        protocol, models, quality, summary, provenance = fixture()
        data = MODULE.build_evidence(protocol, models, quality, summary, provenance)
        home = data["per_home"][0]
        self.assertEqual(home["excluded_seconds"], 150)
        self.assertEqual(home["large_gap_seconds"], 90)
        self.assertEqual(home["invalid_supported_seconds"], 8)
        self.assertEqual(home["valid_seconds_discarded_with_partial_blocks"], 52)
        self.assertEqual(home["empty_test_windows"], ["test-1"])
        self.assertEqual(data["coverage_totals"]["expected_method_seed_score_records"], 9)
        self.assertEqual(len(data["all_selected_window_accounting"]), 2)
        self.assertFalse(data["stage_closure_eligible"])

    def test_audit_must_pass_and_bind_current_artifacts(self):
        values = fixture()
        audit = {"status": "passed", "protocol_sha256": "protocol", "summary_sha256": "summary", "frozen_models_sha256": "models"}
        data = MODULE.build_evidence(*values, audit)
        self.assertTrue(data["stage_closure_eligible"])
        audit["summary_sha256"] = "old-summary"
        self.assertFalse(MODULE.build_evidence(*values, audit)["stage_closure_eligible"])
        audit["summary_sha256"] = "summary"
        audit["status"] = "running"
        self.assertFalse(MODULE.build_evidence(*values, audit)["stage_closure_eligible"])

    def test_current_summary_and_per_home_arithmetic_are_checked(self):
        values = list(fixture())
        values[3]["overall"]["qaoa_ideal"]["equal_home_macro_mae_w"] = 999
        with self.assertRaisesRegex(ValueError, "arithmetic"):
            MODULE.build_evidence(*values)
        values = list(fixture())
        values[3]["per_home"]["7"]["uniform"]["blocks"] = 2
        with self.assertRaisesRegex(ValueError, "coverage"):
            MODULE.build_evidence(*values)

    def test_missing_window_or_model_is_not_silently_dropped(self):
        values = list(fixture())
        values[2] = values[2][:-1]
        with self.assertRaisesRegex(ValueError, "every|Every"):
            MODULE.build_evidence(*values)
        values = list(fixture())
        values[1]["homes"]["7"]["actual_counts"] = [4, 3, 1, 3]
        with self.assertRaisesRegex(ValueError, "counts"):
            MODULE.build_evidence(*values)

    def test_support_seconds_must_reconcile_without_double_counting(self):
        values = list(fixture())
        values[2][0]["quality"]["unsupported_large_gap_seconds"] = 31
        with self.assertRaisesRegex(ValueError, "reconcile"):
            MODULE.build_evidence(*values)

    def test_missing_zero_count_is_not_silently_reported_as_zero(self):
        values = list(fixture())
        del values[2][0]["quality"]["zero_retained_blocks_by_channel"]["Appliance2"]
        with self.assertRaisesRegex(ValueError, "zero-block"):
            MODULE.build_evidence(*values)

    def test_rendered_outputs_keep_pending_and_scope_cautions(self):
        data = MODULE.build_evidence(*fixture())
        report, table = MODULE.render_report(data), MODULE.render_latex(data)
        self.assertIn("Do not mark Stage G closed", report)
        self.assertIn("within-home", report)
        self.assertIn("unflagged", report)
        self.assertIn("not QPU", report)
        self.assertIn("concurrent local audit", report)
        self.assertIn("not a controlled speed benchmark", report)
        self.assertIn("QPU-time forecast", report)
        self.assertIn("Independent audit pending", table)
        self.assertIn("Exact full DP", table)
        self.assertIn("-1.000 [-1.000, -1.000]", report)
        self.assertIn("test-1", report)

    def test_all_unevaluable_cohort_has_null_outcomes_and_cannot_close(self):
        protocol, models, quality, summary, provenance = fixture()
        quality[0]["valid_blocks"] = 0
        quality[0]["quality"]["valid_blocks"] = 0
        quality[0]["quality"]["zero_retained_blocks_by_channel"] = {key: 0 for key in quality[0]["quality"]["zero_retained_blocks_by_channel"]}
        summary.update(status="no_evaluable_homes", nonevaluable_homes=[7], per_home={}, paired_day_comparisons=[], paired_home_comparisons=[])
        summary["coverage"]["7"].update(nonempty_windows=0, valid_blocks=0, empty_windows=["test-0", "test-1"])
        for value in summary["overall"].values():
            value.update(equal_home_macro_mae_w=None, block_pooled_macro_mae_w=None, evaluable_homes=0, blocks=0, windows=0)
        audit = {"status": "passed", "protocol_sha256": "protocol", "summary_sha256": "summary"}
        data = MODULE.build_evidence(protocol, models, quality, summary, provenance, audit)
        self.assertFalse(data["stage_closure_eligible"])
        self.assertEqual(data["status"], "no_evaluable_homes")
        self.assertTrue(all(value is None for value in data["per_home"][0]["macro_mae_w"].values()))
        self.assertIn("unevaluable", MODULE.render_report(data))


if __name__ == "__main__":
    unittest.main()
