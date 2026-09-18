"""Exercise the hardware runner lifecycle with no IBM credentials or network."""

import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from quantum_nilm.qaoa import phase_scale_for
from quantum_nilm.qubo import build_binary_temporal_qubo


IBM_EXTRAS_AVAILABLE = all(
    importlib.util.find_spec(name) for name in ("qiskit", "qiskit_aer", "qiskit_ibm_runtime")
)


@unittest.skipUnless(IBM_EXTRAS_AVAILABLE, "IBM optional dependencies absent")
class IBMRunnerLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_ibm_pilot.py"
        spec = importlib.util.spec_from_file_location("qnilm_test_ibm_runner", script)
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def setUp(self) -> None:
        from qiskit_aer import AerSimulator

        self.temporary = tempfile.TemporaryDirectory(prefix="qnilm-ibm-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "prepared"
        qubo = build_binary_temporal_qubo(
            np.array([100.0, 300.0]), np.array([100.0, 300.0]), 0.0,
            segment_weights=np.array([4.0, 1.0]),
        )
        source = {
            "channels": ["first", "second"],
            "nominal_incremental_power_w": {"first": 100.0, "second": 300.0},
            "switch_penalty": {"first": 0.0, "second": 0.0},
            "aggregate_segment_power_w": [100.0, 300.0],
            "selected_segment_durations_blocks": [4.0, 1.0],
            "reference_states": [[1, 0], [0, 1]],
            "qubits": 4, "qaoa_depth": 1,
            "gamma": 0.7, "beta": 0.3, "phase_scale": phase_scale_for(qubo),
        }
        self.source_path = self.root / "source.json"
        self.runner.write_json(self.source_path, source)
        self.args = argparse.Namespace(
            mode="prepare", pilot_summary=self.source_path, output_dir=self.directory,
            account="mock-account", backend="aer_simulator", shots=8, seed=7,
            max_execution_time=60, allow_paid=False,
        )
        self.service = Mock()
        self.service.backend.return_value = AerSimulator()
        self.service.backend.return_value.status = Mock(return_value=Mock(operational=True))
        self.service.active_instance.return_value = "mock-open-instance"
        self.service.instances.return_value = [{"crn": "mock-open-instance", "plan": "open"}]
        self.service_patch = patch.object(self.runner, "service_for", return_value=self.service)
        self.service_patch.start()
        self.addCleanup(self.service_patch.stop)
        self.network_guard = patch.object(
            self.runner, "QiskitRuntimeService", side_effect=AssertionError("Real IBM service forbidden")
        )
        self.network_guard.start()
        self.addCleanup(self.network_guard.stop)
        self.sampler_patch = patch.object(self.runner, "SamplerV2")
        self.sampler_class = self.sampler_patch.start()
        self.addCleanup(self.sampler_patch.stop)
        with redirect_stdout(io.StringIO()):
            self.runner.prepare_or_simulate(self.args)

    def submit(self) -> None:
        self.args.mode = "submit"
        with redirect_stdout(io.StringIO()):
            self.runner.submit_or_collect(self.args)

    def read_json(self, name: str) -> dict:
        return json.loads((self.directory / name).read_text())

    def test_prepare_freezes_circuit_and_manifest_without_submitting(self) -> None:
        self.sampler_class.assert_not_called()
        self.service.job.assert_not_called()
        self.assertFalse((self.directory / "job.json").exists())
        plan = self.read_json("plan.json")
        self.assertEqual(plan["mode"], "qpu")
        self.assertEqual(plan["shots"], 8)
        self.assertEqual(plan["instance_plan"], "open")
        self.assertLess(plan["probability_crosscheck_max_abs_error"], 1e-12)
        self.assertIn("prepared.qpy", plan["files_sha256"])
        self.assertIn("input_summary.json", plan["files_sha256"])
        for name, digest in plan["files_sha256"].items():
            self.assertEqual(self.runner.checksum(self.directory / name), digest)

    def test_existing_submission_record_prevents_duplicate(self) -> None:
        self.runner.write_json(self.directory / "job.json", {"status": "submission_started"})
        with self.assertRaisesRegex(ValueError, "submission record already exists"):
            self.submit()
        self.sampler_class.assert_not_called()

    def test_modified_prepared_input_blocks_submission(self) -> None:
        source = self.read_json("input_summary.json")
        source["gamma"] += 0.1
        self.runner.write_json(self.directory / "input_summary.json", source)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.submit()
        self.sampler_class.assert_not_called()
        self.assertFalse((self.directory / "job.json").exists())

    def test_paid_or_unidentified_plan_is_blocked_without_explicit_permission(self) -> None:
        for plan_name in ("pay-as-you-go", None):
            with self.subTest(plan=plan_name):
                self.service.instances.return_value = [{"crn": "mock-open-instance", "plan": plan_name}]
                with self.assertRaisesRegex(ValueError, "Open Plan"):
                    self.submit()
                self.sampler_class.assert_not_called()
                self.assertFalse((self.directory / "job.json").exists())

    def test_submit_persists_job_id_and_uses_frozen_shots(self) -> None:
        self.sampler_class.return_value.run.return_value.job_id.return_value = "mock-job-123"
        # CLI defaults must not override the prepared experiment on resumption.
        self.args.shots = 4096
        self.submit()
        record = self.read_json("job.json")
        self.assertEqual(record["job_id"], "mock-job-123")
        self.assertEqual(record["status"], "submitted")
        self.assertEqual(record["plan_sha256"], self.runner.checksum(self.directory / "plan.json"))
        run = self.sampler_class.return_value.run
        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs["shots"], 8)
        circuits = run.call_args.args[0]
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0].num_clbits, 4)
        with self.assertRaisesRegex(ValueError, "submission record already exists"):
            self.submit()
        run.assert_called_once()

    def test_ambiguous_submission_retains_guard_and_cannot_repeat(self) -> None:
        self.sampler_class.return_value.run.side_effect = ConnectionError("Ambiguous provider response")
        with self.assertRaises(ConnectionError):
            self.submit()
        record = self.read_json("job.json")
        self.assertEqual(record["status"], "submission_started")
        self.assertNotIn("job_id", record)
        with self.assertRaisesRegex(ValueError, "submission record already exists"):
            self.submit()
        self.sampler_class.return_value.run.assert_called_once()
        self.args.mode = "collect"
        with self.assertRaisesRegex(RuntimeError, "outcome unknown"):
            self.runner.submit_or_collect(self.args)
        self.service.job.assert_not_called()

    def test_collect_archives_raw_counts_result_and_decoded_metrics(self) -> None:
        from qiskit.primitives.containers import BitArray, DataBin, PrimitiveResult, SamplerPubResult

        self.runner.write_json(self.directory / "job.json", {
            "status": "submitted", "job_id": "mock-job-123",
            "plan_sha256": self.runner.checksum(self.directory / "plan.json"),
        })
        raw_counts = {"1001": 6, "0001": 2}
        result = PrimitiveResult([
            SamplerPubResult(DataBin(meas=BitArray.from_counts(raw_counts, num_bits=4)))
        ])
        job = self.service.job.return_value
        job.status.return_value = "DONE"
        job.result.return_value = result
        job.metrics.return_value = {"timestamps": {
            "created": "2026-09-13T12:00:00Z", "running": "2026-09-13T12:00:10Z",
            "finished": "2026-09-13T12:00:12Z",
        }}
        self.args.mode = "collect"
        with redirect_stdout(io.StringIO()):
            self.runner.submit_or_collect(self.args)
        self.sampler_class.assert_not_called()
        self.service.job.assert_called_once_with("mock-job-123")
        self.assertEqual(self.read_json("counts.json"), raw_counts)
        self.assertTrue((self.directory / "runtime_result.json").is_file())
        self.assertEqual(self.read_json("job.json")["status"], "DONE")
        summary = self.read_json("summary.json")
        self.assertEqual(summary["shots"], 8)
        self.assertEqual(summary["best_bits"], [1, 0, 0, 1])
        self.assertEqual(summary["exact_optimum_probability"], 0.75)
        self.assertEqual(summary["mean_objective"], 22500.0)
        self.assertEqual(summary["reference_metrics"]["bit_accuracy"], 1.0)
        self.assertEqual(summary["aggregate_mae_w"], 0.0)
        self.assertEqual(summary["queue_and_initialization_s"], 10.0)
        self.assertEqual(summary["server_running_wall_time_s"], 2.0)
        self.assertIn("proxy", summary["reference_type"])

    def test_collect_rejects_plan_change_after_submission(self) -> None:
        self.runner.write_json(self.directory / "job.json", {
            "status": "submitted", "job_id": "mock-job-123",
            "plan_sha256": self.runner.checksum(self.directory / "plan.json"),
        })
        plan = self.read_json("plan.json")
        plan["shots"] = 16
        self.runner.write_json(self.directory / "plan.json", plan)
        self.args.mode = "collect"
        with self.assertRaisesRegex(ValueError, "plan changed"):
            self.runner.submit_or_collect(self.args)
        self.service.job.assert_not_called()

    def test_collect_pending_job_saves_status_without_waiting_or_resubmitting(self) -> None:
        self.runner.write_json(self.directory / "job.json", {
            "status": "submitted", "job_id": "mock-job-123",
            "plan_sha256": self.runner.checksum(self.directory / "plan.json"),
        })
        self.service.job.return_value.status.return_value = "QUEUED"
        self.args.mode = "collect"
        with redirect_stdout(io.StringIO()):
            self.runner.submit_or_collect(self.args)
        self.assertEqual(self.read_json("job.json")["status"], "QUEUED")
        self.service.job.return_value.result.assert_not_called()
        self.sampler_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
