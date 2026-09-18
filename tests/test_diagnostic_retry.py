import unittest
from scripts.retry_quantum_diagnostic_ladder import planning_reserve


class RetryReserveTests(unittest.TestCase):
    def test_pending_elapsed_is_not_assumed_free(self):
        metrics = {"usage": {"qpu_charge_time_seconds": 0, "status": "pending"},
                   "timestamps": {"running": "2026-09-14T14:33:37.738409Z",
                                  "finished": "2026-09-14T14:34:45.081028Z"}}
        self.assertEqual(planning_reserve(metrics), 68)

    def test_finalized_zero_needs_no_extra_reserve(self):
        self.assertEqual(planning_reserve({"usage": {"status": "completed", "qpu_charge_time_seconds": 0}}), 0)

    def test_pending_without_timestamps_fails_closed(self):
        with self.assertRaises(RuntimeError):
            planning_reserve({"usage": {"status": "pending"}})

    def test_unknown_usage_reserves_full_cap(self):
        self.assertEqual(planning_reserve({"timestamps": {"running": "2026-09-14T00:00:00Z",
                                                        "finished": "2026-09-14T00:00:01Z"}}), 60)
