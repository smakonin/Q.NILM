"""One-time controller tests; no child process, IBM account, or job execution."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/continue_quantum_heldout.py"
SPEC = importlib.util.spec_from_file_location("quantum_continuation", SCRIPT)
CONTINUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTINUE)


class QuantumContinuationTests(unittest.TestCase):
    def test_waits_for_exact_process_exit_without_signalling_it(self):
        with TemporaryDirectory() as directory:
            campaign = Path(directory)
            identity = Mock(side_effect=["fixed start python", "fixed start python", None])
            sleep, checkpoint = Mock(), Mock()
            CONTINUE.wait_for_initial_exit(43269, "fixed start python", campaign, checkpoint,
                identity_reader=identity, sleeper=sleep, poll_seconds=1)
            self.assertEqual(identity.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual([call.args[0] for call in checkpoint.call_args_list],
                             ["waiting_initial_process", "initial_process_exited"])

    def test_missing_or_reused_process_fails_closed(self):
        with TemporaryDirectory() as directory:
            for readings in ([None], ["wrong process"], ["expected", "different start"]):
                with self.subTest(readings=readings), self.assertRaises(CONTINUE.ContinuationStopped):
                    CONTINUE.wait_for_initial_exit(43269, "expected", Path(directory), Mock(),
                        identity_reader=Mock(side_effect=readings), sleeper=Mock())

    def test_ps_errors_are_not_treated_as_process_exit(self):
        for response in (SimpleNamespace(returncode=1, stdout="", stderr="permission denied"),
                         SimpleNamespace(returncode=0, stdout="", stderr="")):
            with patch.object(CONTINUE.subprocess, "run", return_value=response), self.assertRaises(CONTINUE.ContinuationStopped):
                CONTINUE.process_identity(43269)
        with patch.object(CONTINUE.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="", stderr="")):
            self.assertIsNone(CONTINUE.process_identity(43269))

    def pipeline(self, campaign, work, *, fail=None, omit_completion=False, stop_after_run=False):
        calls = []
        def launch(label, command):
            calls.append(label)
            if label == fail:
                return 1
            if label == "patched_run_once":
                if not omit_completion:
                    (campaign / "hardware/completed.json").write_text("{}")
                if stop_after_run:
                    (campaign / "hardware/budget_stop.json").write_text("{}")
            if label == "audit_complete":
                (work / "final_hardware_audit.json").write_text(json.dumps({"complete": True, "status": "passed_complete"}))
            if label == "summarize":
                (campaign / "analysis_complete").mkdir()
                (campaign / "analysis_complete/analysis.json").write_text(json.dumps({"hardware": {"status": "complete_scored"}}))
            return 0
        return calls, launch

    def test_successful_pipeline_calls_runner_once_then_score_audit_summary(self):
        with TemporaryDirectory() as directory:
            campaign = Path(directory)
            work = campaign / "hardware/continuation"
            work.mkdir(parents=True)
            calls, launch = self.pipeline(campaign, work)
            checkpoint = Mock()
            CONTINUE.execute_pipeline(campaign, work, checkpoint, launch)
            self.assertEqual(calls, ["patched_run_once", "score", "audit_complete", "summarize"])
            self.assertEqual(checkpoint.call_args.args[0], "complete")
            self.assertFalse(checkpoint.call_args.kwargs["paper_edited"])

    def test_nonzero_exit_never_retries_or_runs_later_phases(self):
        for failed, expected in (("patched_run_once", ["patched_run_once"]),
                                 ("score", ["patched_run_once", "score"]),
                                 ("audit_complete", ["patched_run_once", "score", "audit_complete"])):
            with self.subTest(failed=failed), TemporaryDirectory() as directory:
                campaign = Path(directory)
                work = campaign / "hardware/continuation"
                work.mkdir(parents=True)
                calls, launch = self.pipeline(campaign, work, fail=failed)
                with self.assertRaises(CONTINUE.ContinuationStopped):
                    CONTINUE.execute_pipeline(campaign, work, Mock(), launch)
                self.assertEqual(calls, expected)

    def test_zero_run_exit_without_completion_does_not_score(self):
        with TemporaryDirectory() as directory:
            campaign = Path(directory)
            work = campaign / "hardware/continuation"
            work.mkdir(parents=True)
            calls, launch = self.pipeline(campaign, work, omit_completion=True)
            with self.assertRaises(CONTINUE.ContinuationStopped):
                CONTINUE.execute_pipeline(campaign, work, Mock(), launch)
            self.assertEqual(calls, ["patched_run_once"])

    def test_stop_and_budget_markers_prevent_further_work(self):
        with TemporaryDirectory() as directory:
            campaign = Path(directory)
            work = campaign / "hardware/continuation"
            work.mkdir(parents=True)
            calls, launch = self.pipeline(campaign, work, stop_after_run=True)
            with self.assertRaises(CONTINUE.ContinuationStopped):
                CONTINUE.execute_pipeline(campaign, work, Mock(), launch)
            self.assertEqual(calls, ["patched_run_once"])
            with self.assertRaises(CONTINUE.ContinuationStopped):
                CONTINUE.wait_for_initial_exit(43269, "expected", campaign, Mock(),
                    identity_reader=Mock(), sleeper=Mock())

    def test_existing_analysis_is_not_overwritten(self):
        with TemporaryDirectory() as directory:
            campaign = Path(directory)
            work = campaign / "hardware/continuation"
            work.mkdir(parents=True)
            (campaign / "analysis_complete").mkdir()
            launch = Mock()
            with self.assertRaises(CONTINUE.ContinuationStopped):
                CONTINUE.execute_pipeline(campaign, work, Mock(), launch)
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
