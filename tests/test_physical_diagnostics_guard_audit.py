"""Independent offline failure-boundary tests; every IBM API is mocked."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.primitives.containers import BitArray, DataBin, PrimitiveResult, SamplerPubResult

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_physical_diagnostics as runner


def submission_fixture(folder):
    plan = {"options": {"max_execution_time": 45}}
    runner.save(folder / "plan.json", plan)
    runner.save(folder / "pubs.json", [])
    runner.save(folder / "ideal_audit.json", {
        "status": "passed", "circuits": runner.EXPECTED_CIRCUITS,
        "plan_sha256": runner.sha(folder / "plan.json"),
    })
    circuit = QuantumCircuit(1)
    circuit.measure_all()
    with (folder / "fixture.qpy").open("xb") as handle:
        qpy.dump(circuit, handle)
    specs = [{"qpy_file": "fixture.qpy", "shots": 8}]
    backend = Mock()
    backend.target.instruction_supported.return_value = True
    snapshot = {"available_free_s": 148, "required_free_s": 143,
                "pending_accounting_reserve_s": 68}
    return plan, specs, backend, snapshot


class PhysicalGuardAuditTests(unittest.TestCase):
    def test_existing_submission_snapshot_blocks_before_live_access(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            plan, specs, _, _ = submission_fixture(folder)
            runner.save(folder / "submission_snapshot.json", {"preflight_already_started": True})
            with patch.object(runner, "check", return_value=(plan, specs)), \
                 patch.object(runner, "live_snapshot") as live, \
                 patch.object(runner, "SamplerV2") as sampler:
                with self.assertRaises(ValueError):
                    runner.submit(folder, True)
                live.assert_not_called()
                sampler.assert_not_called()

    def test_ambiguous_submission_persists_intent_and_cannot_resubmit(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            plan, specs, backend, snapshot = submission_fixture(folder)
            sampler = Mock()
            sampler.run.side_effect = RuntimeError("simulated uncertain network outcome")
            with patch.object(runner, "check", return_value=(plan, specs)), \
                 patch.object(runner, "live_snapshot", return_value=(Mock(), backend, snapshot)) as live, \
                 patch.object(runner, "SamplerV2", return_value=sampler):
                with self.assertRaises(RuntimeError):
                    runner.submit(folder, True)
                self.assertTrue((folder / "intent.json").exists())
                self.assertTrue((folder / "submission_snapshot.json").exists())
                self.assertFalse((folder / "job.json").exists())
                intent_hash = runner.sha(folder / "intent.json")
                with self.assertRaises(ValueError):
                    runner.submit(folder, True)
                self.assertEqual(sampler.run.call_count, 1)
                self.assertEqual(live.call_count, 1)
                self.assertEqual(runner.sha(folder / "intent.json"), intent_hash)

    def test_stop_arriving_during_preflight_prevents_submission(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            plan, specs, backend, snapshot = submission_fixture(folder)
            def preflight():
                (folder / "STOP").touch()
                return Mock(), backend, snapshot
            with patch.object(runner, "check", return_value=(plan, specs)), \
                 patch.object(runner, "live_snapshot", side_effect=preflight), \
                 patch.object(runner, "SamplerV2") as sampler:
                with self.assertRaises(ValueError):
                    runner.submit(folder, True)
                sampler.assert_not_called()
                self.assertFalse((folder / "intent.json").exists())

    def test_readonly_status_never_constructs_sampler(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            runner.save(folder / "job.json", {"job_id": "known-job"})
            service, job = Mock(), Mock()
            service.job.return_value = job
            job.status.return_value = "DONE"
            job.metrics.return_value = {"usage": {"status": "complete", "qpu_charge_time_seconds": 12}}
            with patch.object(runner, "account", return_value=service), \
                 patch.object(runner, "SamplerV2") as sampler:
                runner.status(folder)
                service.job.assert_called_once_with("known-job")
                sampler.assert_not_called()
                job.result.assert_not_called()

    def test_pending_accounting_keeps_raw_data_without_a_final_result(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            runner.save(folder / "plan.json", {})
            plan_hash = runner.sha(folder / "plan.json")
            runner.save(folder / "intent.json", {"plan_sha256": plan_hash})
            runner.save(folder / "job.json", {"job_id": "known-job", "plan_sha256": plan_hash,
                                               "intent_sha256": runner.sha(folder / "intent.json")})
            bitarray = BitArray(np.zeros((8, 1), dtype=np.uint8), num_bits=1)
            returned = PrimitiveResult([SamplerPubResult(DataBin(meas=bitarray))])
            job, service = Mock(), Mock()
            service.job.return_value = job
            job.status.return_value = "DONE"
            job.result.return_value = returned
            job.metrics.return_value = {"usage": {"status": "pending", "qpu_charge_time_seconds": 0}}
            with patch.object(runner, "check", return_value=({}, [])), \
                 patch.object(runner, "account", return_value=service), \
                 patch.object(runner, "SamplerV2") as sampler:
                with self.assertRaises(ValueError):
                    runner.collect(folder)
                self.assertTrue((folder / "runtime.json.gz").exists())
                self.assertFalse((folder / "runtime.json.gz.pending").exists())
                self.assertFalse((folder / "metrics.json").exists())
                self.assertFalse((folder / "result.json").exists())
                with self.assertRaises(ValueError):
                    runner.collect(folder)
                job.result.assert_called_once()
                sampler.assert_not_called()

    def test_exact_declared_budget_is_arithmetically_consistent(self):
        self.assertEqual(24 + 12 + 2 * 5 * 8 * 2 + 2 * 2 * 2 * 4, runner.EXPECTED_CIRCUITS)
        self.assertEqual(24 * 1024 + 12 * 1024 + 160 * 256 + 32 * 512, runner.EXPECTED_SHOTS)
        self.assertEqual(runner.CAP + runner.RESERVE + 68, 143)
        self.assertLess(runner.MAX_EXECUTION, runner.CAP)

    def test_decode_pub_reads_exact_packed_byte_counts_with_correct_endianness(self):
        spec = {"n_clbits": 12, "shots": 4}
        # Big-endian bytes encode classical bitstrings with c0 on the right.
        packed = np.array([[0, 1], [8, 0], [0, 1], [0, 16]], dtype=np.uint8)
        pub = SamplerPubResult(DataBin(meas=BitArray(packed, num_bits=12)))
        counts = runner.decode_pub(pub, spec)
        self.assertEqual(counts, {"000000000001": 2, "100000000000": 1, "000000010000": 1})

    def test_decode_pub_rejects_missing_or_extra_measurement_registers(self):
        bitarray = BitArray(np.zeros((4, 1), dtype=np.uint8), num_bits=3)
        for data in (DataBin(other=bitarray), DataBin(meas=bitarray, other=bitarray)):
            with self.subTest(fields=list(data.keys())), self.assertRaises(ValueError):
                runner.decode_pub(SamplerPubResult(data), {"n_clbits": 3, "shots": 4})

    def test_decode_pub_rejects_wrong_bits_shots_and_parameter_broadcast_shape(self):
        cases = [
            BitArray(np.zeros((4, 1), dtype=np.uint8), num_bits=4),
            BitArray(np.zeros((5, 1), dtype=np.uint8), num_bits=3),
            BitArray(np.zeros((1, 4, 1), dtype=np.uint8), num_bits=3),
        ]
        for array in cases:
            with self.subTest(bits=array.num_bits, shape=array.array.shape), self.assertRaises(ValueError):
                runner.decode_pub(SamplerPubResult(DataBin(meas=array)), {"n_clbits": 3, "shots": 4})

    def test_decode_pub_rejects_nonzero_padding_even_if_histogram_masks_it(self):
        array = BitArray(np.array([[8], [0], [0], [0]], dtype=np.uint8), num_bits=3)
        with self.assertRaises(ValueError):
            runner.decode_pub(SamplerPubResult(DataBin(meas=array)), {"n_clbits": 3, "shots": 4})

    def test_decode_pub_requires_byte_histogram_agreement(self):
        array = BitArray(np.array([[1], [0], [0], [0]], dtype=np.uint8), num_bits=3)
        with patch.object(BitArray, "get_counts", return_value={"000": 4}), self.assertRaises(ValueError):
            runner.decode_pub(SamplerPubResult(DataBin(meas=array)), {"n_clbits": 3, "shots": 4})


if __name__ == "__main__":
    unittest.main()
