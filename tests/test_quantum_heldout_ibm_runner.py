"""Network-free safety and recovery checks for adaptive held-out IBM jobs."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from quantum_nilm.categorical_qaoa import encode_onehot, prepare_categorical_problem


IBM_AVAILABLE = all(importlib.util.find_spec(name) for name in ("qiskit", "qiskit_ibm_runtime"))


class _Pub(dict):
    """Serializable fixture with the real Sampler result access shape."""

    def __init__(self, counts):
        super().__init__(counts=counts)
        self.data = SimpleNamespace(meas=SimpleNamespace(get_counts=lambda: counts))


@unittest.skipUnless(IBM_AVAILABLE, "IBM optional dependencies absent")
class QuantumHeldoutIBMSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = Path(__file__).resolve().parents[1] / "scripts/run_quantum_heldout_ibm.py"
        spec = importlib.util.spec_from_file_location("qnilm_test_quantum_heldout_ibm", script)
        cls.runner = importlib.util.module_from_spec(spec)
        with patch.object(sys, "path", [str(script.parent), *sys.path]):
            spec.loader.exec_module(cls.runner)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="qnilm-hardware-lifecycle-")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.hardware = self.output / "hardware"
        self.hardware.mkdir()
        self.levels = [[0.0, 10.0], [0.0], [0.0], [-2.0]]
        self.plan = {
            "backend": "mock-open-backend", "shots": 4,
            "adaptive_stages": 2, "circuits": 2,
            "maximum_campaign_usage_s": 540.0, "rep_delay_s": 0.001,
            "templates": {str(k): {"parameter_order": [], "duration_s": 0.001} for k in (1, 2)},
        }
        self.windows = [{"window": {"id": "test-000"}, "chunks": [
            {"chunk": 0, "reset": True, "aggregate": [8.0], "weights": [1.0]},
            {"chunk": 1, "reset": False, "aggregate": [8.0], "weights": [1.0]},
        ]}]
        self.write(self.hardware / "plan.json", self.plan)
        self.write(self.output / "protocol.json", {})
        self.write(self.output / "angles.json", {"gammas": [1.0], "betas": [0.3]})
        self.write(self.output / "test_inputs.json", self.windows)
        self.write(self.output / "model.json", {"models": {"multistate": {
            "levels_w": self.levels, "ranges_w": [10.0, 1.0, 1.0, 1.0], "event_threshold_w": 1.0,
        }}})
        for length in (1, 2):
            (self.hardware / f"template_{length}.qpy").touch()
        self.service = Mock()
        self.service.active_instance.return_value = "mock-open-instance"
        self.service.instances.return_value = [{"crn": "mock-open-instance", "plan": "open"}]
        self.service.usage.return_value = {"usage_remaining_seconds": 600.0, "usage_limit_reached": False}
        self.service.backend.return_value.status.return_value.operational = True
        # All access to an IBM client is replaced before any test calls run().
        self.patch("QiskitRuntimeService", return_value=self.service)
        self.patch("check_plan", side_effect=lambda output: self.plan)
        self.patch("build_categorical_parameterized_template", return_value=Mock())
        self.patch("categorical_template_parameter_values", return_value=np.array([]))
        self.qpy_patch = patch.object(self.runner.qpy, "load", return_value=[SimpleNamespace(parameters=())])
        self.qpy_patch.start()
        self.addCleanup(self.qpy_patch.stop)
        self.sampler = self.patch("SamplerV2")
        self.job = Mock()
        self.job.job_id.return_value = "mock-job-001"
        self.valid_key = encode_onehot([[1, 0, 0, 0]], [2, 1, 1, 1])
        self.job.result.return_value = [_Pub({self.valid_key: 4})]
        self.job.metrics.return_value = {"usage": {"quantum_seconds": 1.0}}
        self.sampler.return_value.run.return_value = self.job
        self.service.job.return_value = self.job

    def patch(self, name, **kwargs):
        handle = patch.object(self.runner, name, **kwargs)
        result = handle.start()
        self.addCleanup(handle.stop)
        return result

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, indent=2) + "\n")

    @staticmethod
    def read(path):
        return json.loads(path.read_text())

    def run_campaign(self):
        with redirect_stdout(io.StringIO()):
            self.runner.run(self.output)

    def current(self, previous=None, length=1):
        problem = prepare_categorical_problem([8.0] * length, self.levels, [1.0, 0.0, 0.0, 0.0], previous_states=previous)
        return [{"problem": problem, "window_id": "test-000", "chunk": {"chunk": 0}, "previous": previous}]

    def intent(self, stage=0):
        return {
            "stage": stage, "job_id": "mock-existing-job", "status": "submitted",
            "plan_sha256": self.runner.sha(self.hardware / "plan.json"),
            "days": ["test-000"], "shots_per_circuit": 4,
            "max_execution_time_s": 6, "estimated_usage_s": 2.008,
            "previous_states": {"test-000": None}, "parameter_values": [[]],
        }

    def saved_first_stage(self, usage=1.0):
        saved = self.runner.decode_stage(self.plan, self.current(), [_Pub({self.valid_key: 4})], {"usage": {"quantum_seconds": usage}}, 0.0)
        saved.update({"stage": 0, "job_id": "mock-existing-job", "plan_sha256": self.runner.sha(self.hardware / "plan.json")})
        return saved

    def historical_fez_usage(self):
        """Use the immutable real Fez response schema, not an invented alias."""
        archive = Path(__file__).resolve().parents[1] / "results/ibm_corrected_qpu_check/summary.json"
        usage = self.read(archive)["job_metrics"]["usage"]
        self.assertEqual(usage["qpu_charge_time_seconds"], 3)
        self.assertEqual(usage["status"], "complete")
        self.assertNotIn("quantum_seconds", usage)
        return usage

    def write_completed_first_stage_with_usage(self, usage):
        saved = self.saved_first_stage()
        saved["job_metrics"] = {"usage": usage}
        self.write(self.hardware / "stage_000_result.json", saved)
        self.write(self.hardware / "stage_000_job.json", self.intent())
        return saved

    def test_account_requires_known_open_plan(self):
        for plan in ("pay-as-you-go", "flex", None):
            self.service.instances.return_value = [{"crn": "mock-open-instance", "plan": plan}]
            with self.subTest(plan=plan), self.assertRaisesRegex(RuntimeError, "Open Plan"):
                self.runner.account()
        self.service.instances.return_value = []
        with self.assertRaisesRegex(RuntimeError, "Open Plan"):
            self.runner.account()
        self.sampler.assert_not_called()

    def test_invalid_or_exhausted_allowance_is_rejected(self):
        for value in (None, -1.0, float("nan"), float("inf")):
            self.service.usage.return_value = {"usage_remaining_seconds": value}
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "allowance"):
                self.runner.remaining(self.service)
        self.service.usage.return_value = {"usage_remaining_seconds": 10, "usage_limit_reached": True}
        with self.assertRaisesRegex(RuntimeError, "allowance"):
            self.runner.remaining(self.service)

    def test_decode_keeps_all_shots_and_chooses_feasible_observed_state(self):
        current = self.current()
        result = [_Pub({self.valid_key: 2, "00000": 2})]
        saved = self.runner.decode_stage(self.plan, current, result, {}, 0.0)
        row = saved["rows"][0]
        self.assertFalse(row["fallback_used"])
        self.assertEqual(row["states"], [[1, 0, 0, 0]])
        self.assertEqual(row["sample_score"]["total_shots"], 4)
        self.assertEqual(row["sample_score"]["feasible_shots"], 2)
        self.assertEqual(row["raw_counts"]["00000"], 2)

    def test_all_invalid_falls_back_to_own_prior_or_lowest_state_at_reset(self):
        for prior in (None, np.array([1, 0, 0, 0])):
            with self.subTest(prior=prior):
                saved = self.runner.decode_stage(self.plan, self.current(prior, length=2), [_Pub({"0000000000": 4})], {}, 0.0)
                row = saved["rows"][0]
                self.assertTrue(row["fallback_used"])
                expected = [0, 0, 0, 0] if prior is None else prior.tolist()
                self.assertEqual(row["states"], [expected, expected])
                self.assertEqual(row["sample_score"]["feasible_shots"], 0)

    def test_decode_rejects_missing_pubs_and_wrong_shot_totals(self):
        with self.assertRaisesRegex(RuntimeError, "PUB count"):
            self.runner.decode_stage(self.plan, self.current(), [], {}, 0.0)
        with self.assertRaisesRegex(RuntimeError, "shot count"):
            self.runner.decode_stage(self.plan, self.current(), [_Pub({self.valid_key: 3})], {}, 0.0)

    def test_budget_stop_occurs_before_intent_or_submission(self):
        self.plan["maximum_campaign_usage_s"] = 1.0
        self.run_campaign()
        self.sampler.assert_not_called()
        self.assertTrue((self.hardware / "budget_stop.json").exists())
        self.assertFalse((self.hardware / "stage_000_job.json").exists())
        self.assertFalse((self.hardware / "completed.json").exists())

    def test_stop_marker_prevents_submission(self):
        (self.hardware / "STOP").touch()
        self.run_campaign()
        self.sampler.assert_not_called()

    def test_ambiguous_submission_leaves_intent_and_cannot_be_repeated(self):
        self.sampler.return_value.run.side_effect = ConnectionError("ambiguous test transport outcome")
        with self.assertRaises(ConnectionError):
            self.run_campaign()
        intent = self.read(self.hardware / "stage_000_job.json")
        self.assertEqual(intent["status"], "submission_started")
        self.assertNotIn("job_id", intent)
        with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
            self.run_campaign()
        self.sampler.return_value.run.assert_called_once()
        self.service.job.assert_not_called()

    def test_known_job_is_recovered_without_resubmission(self):
        self.plan["adaptive_stages"] = 1
        self.write(self.hardware / "stage_000_job.json", self.intent())
        self.run_campaign()
        self.service.job.assert_called_once_with("mock-existing-job")
        self.sampler.assert_not_called()
        self.assertTrue((self.hardware / "stage_000_result.json").exists())

    def test_restored_stage_carries_its_own_state_into_next_submission(self):
        saved = self.saved_first_stage()
        self.write(self.hardware / "stage_000_result.json", saved)
        self.write(self.hardware / "stage_000_job.json", self.intent())
        self.run_campaign()
        self.sampler.return_value.run.assert_called_once()
        request = self.read(self.hardware / "stage_001_job.json")
        self.assertEqual(request["previous_states"], {"test-000": [1, 0, 0, 0]})
        completed = self.read(self.hardware / "completed.json")
        self.assertEqual(completed["accounted_quantum_seconds"], 2.0)

    def test_restored_state_is_discarded_at_declared_gap_reset(self):
        saved = self.saved_first_stage()
        self.write(self.hardware / "stage_000_result.json", saved)
        self.write(self.hardware / "stage_000_job.json", self.intent())
        self.windows[0]["chunks"][1]["reset"] = True
        self.write(self.output / "test_inputs.json", self.windows)
        self.run_campaign()
        request = self.read(self.hardware / "stage_001_job.json")
        self.assertEqual(request["previous_states"], {"test-000": None})

    def test_recovery_rejects_intent_from_another_plan(self):
        intent = self.intent()
        intent["plan_sha256"] = "wrong-plan"
        self.write(self.hardware / "stage_000_job.json", intent)
        with self.assertRaises(RuntimeError):
            self.run_campaign()
        self.sampler.assert_not_called()
        self.service.job.assert_not_called()

    def test_recovery_rejects_changed_parameter_bindings(self):
        intent = self.intent()
        intent["parameter_values"] = [[0.123]]
        self.write(self.hardware / "stage_000_job.json", intent)
        with self.assertRaises(RuntimeError):
            self.run_campaign()
        self.sampler.assert_not_called()
        self.service.job.assert_not_called()

    def test_saved_usage_is_charged_before_next_submission_budget(self):
        self.write(self.hardware / "stage_000_result.json", self.saved_first_stage(usage=539.0))
        self.write(self.hardware / "stage_000_job.json", self.intent())
        self.run_campaign()
        self.sampler.assert_not_called()
        stop = self.read(self.hardware / "budget_stop.json")
        self.assertEqual(stop["accounted_s"], 539.0)
        self.assertEqual(stop["next_stage"], 1)

    def test_missing_usage_is_refreshed_without_resubmission_or_result_overwrite(self):
        self.plan["adaptive_stages"] = 1
        saved = self.saved_first_stage(usage=None)
        self.write(self.hardware / "stage_000_result.json", saved)
        self.write(self.hardware / "stage_000_job.json", self.intent())
        before = (self.hardware / "stage_000_result.json").read_bytes()
        self.run_campaign()
        self.service.job.assert_called_once_with("mock-existing-job")
        self.sampler.assert_not_called()
        self.assertEqual((self.hardware / "stage_000_result.json").read_bytes(), before)
        self.assertTrue((self.hardware / "stage_000_finalized_metrics.json").exists())
        self.assertEqual(self.read(self.hardware / "completed.json")["accounted_quantum_seconds"], 1.0)

    def test_submitted_options_fix_shots_cap_and_disable_undeclared_processing(self):
        self.plan["adaptive_stages"] = 1
        self.run_campaign()
        self.sampler.return_value.run.assert_called_once()
        self.assertEqual(self.sampler.return_value.run.call_args.kwargs["shots"], 4)
        options = self.sampler.call_args.kwargs["options"]
        self.assertLessEqual(options["max_execution_time"], self.plan["maximum_campaign_usage_s"])
        self.assertEqual(options["dynamical_decoupling"], {"enable": False})
        self.assertEqual(options["twirling"], {"enable_gates": False, "enable_measure": False})
        self.assertTrue(options["execution"]["init_qubits"])

    def test_real_fez_charge_schema_restores_stage_zero_without_resubmitting_it(self):
        saved = self.write_completed_first_stage_with_usage(self.historical_fez_usage())
        self.run_campaign()
        self.sampler.return_value.run.assert_called_once()
        self.service.job.assert_not_called()
        self.assertEqual(self.read(self.hardware / "stage_000_result.json"), saved)
        request = self.read(self.hardware / "stage_001_job.json")
        self.assertEqual(request["stage"], 1)
        self.assertEqual(request["previous_states"], {"test-000": [1, 0, 0, 0]})
        # Three seconds from the real schema, plus the legacy one-second mock
        # for the newly submitted second stage. Both field forms are supported.
        completed = self.read(self.hardware / "completed.json")
        self.assertEqual(completed["accounted_quantum_seconds"], 4.0)

    def test_primary_charge_is_counted_before_budget_guard(self):
        usage = self.historical_fez_usage()
        usage["qpu_charge_time_seconds"] = 539.0
        self.write_completed_first_stage_with_usage(usage)
        self.run_campaign()
        self.sampler.assert_not_called()
        self.service.job.assert_not_called()
        self.assertEqual(self.read(self.hardware / "budget_stop.json")["accounted_s"], 539.0)

    def test_consistent_primary_and_legacy_charge_aliases_are_accepted(self):
        self.plan["adaptive_stages"] = 1
        usage = self.historical_fez_usage()
        usage["quantum_seconds"] = usage["qpu_charge_time_seconds"]
        self.write_completed_first_stage_with_usage(usage)
        self.run_campaign()
        self.sampler.assert_not_called()
        self.service.job.assert_not_called()
        self.assertEqual(self.read(self.hardware / "completed.json")["accounted_quantum_seconds"], 3.0)

    def test_disagreeing_primary_and_legacy_usage_fail_before_another_submission(self):
        usage = self.historical_fez_usage()
        usage["quantum_seconds"] = 1.0
        self.write_completed_first_stage_with_usage(usage)
        with self.assertRaises(RuntimeError):
            self.run_campaign()
        self.sampler.assert_not_called()
        self.service.job.assert_not_called()
        self.assertFalse((self.hardware / "stage_001_job.json").exists())

    def test_invalid_primary_charge_fails_closed_even_if_legacy_is_valid(self):
        for invalid in (-1.0, float("nan"), float("inf"), float("-inf"), "3", True):
            usage = {"qpu_charge_time_seconds": invalid, "quantum_seconds": 1.0}
            self.write_completed_first_stage_with_usage(usage)
            with self.subTest(value=invalid), self.assertRaises(RuntimeError):
                self.run_campaign()
        self.sampler.assert_not_called()
        self.service.job.assert_not_called()
        self.assertFalse((self.hardware / "stage_001_job.json").exists())

    def test_invalid_legacy_charge_fails_closed_even_if_primary_is_valid(self):
        for invalid in (-1.0, float("nan"), float("inf"), float("-inf"), "3", True):
            usage = {"qpu_charge_time_seconds": 3.0, "quantum_seconds": invalid}
            self.write_completed_first_stage_with_usage(usage)
            with self.subTest(value=invalid), self.assertRaises(RuntimeError):
                self.run_campaign()
        self.sampler.assert_not_called()
        self.service.job.assert_not_called()
        self.assertFalse((self.hardware / "stage_001_job.json").exists())

    def test_delayed_primary_charge_refresh_uses_actual_schema_and_preserves_counts(self):
        self.plan["adaptive_stages"] = 1
        saved = self.write_completed_first_stage_with_usage({"qpu_charge_time_seconds": None, "status": "pending"})
        before = (self.hardware / "stage_000_result.json").read_bytes()
        self.job.metrics.return_value = {"usage": self.historical_fez_usage()}
        self.run_campaign()
        self.service.job.assert_called_once_with("mock-existing-job")
        self.sampler.assert_not_called()
        self.assertEqual((self.hardware / "stage_000_result.json").read_bytes(), before)
        self.assertEqual(self.read(self.hardware / "stage_000_result.json"), saved)
        finalized = self.read(self.hardware / "stage_000_finalized_metrics.json")
        self.assertEqual(finalized["job_metrics"]["usage"]["qpu_charge_time_seconds"], 3)
        self.assertEqual(self.read(self.hardware / "completed.json")["accounted_quantum_seconds"], 3.0)


if __name__ == "__main__":
    unittest.main()
