"""Synthetic-only tests; never read actual REFIT appliance measurements."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from quantum_nilm.refit import HEADER, read_block_window, source_bounds


def line(timestamp, aggregate=100, appliance=10, issues=0, overrides=None):
    values = ["2014-01-01 00:00:00", str(timestamp), str(aggregate), str(appliance), *(["0"] * 8), str(issues)]
    for column, value in (overrides or {}).items():
        values[HEADER.index(column)] = str(value)
    return (",".join(values) + "\n").encode()


class RefitReaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "synthetic.csv"

    def fixture(self, rows, header=None):
        raw_header = ((",".join(HEADER) if header is None else header) + "\n").encode()
        self.path.write_bytes(raw_header + b"".join(rows))
        return raw_header

    def test_irregular_six_eight_fourteen_second_integral(self):
        self.fixture([line(0, 30, 3), line(6, 60, 6), line(14, 90, 9), line(28, 120, 12), line(36)])
        actual = read_block_window(self.path, 0, 30)
        expected = (6 * 30 + 8 * 60 + 14 * 90 + 2 * 120) / 30
        np.testing.assert_allclose(actual["values"], [[expected, expected / 10]])
        np.testing.assert_array_equal(actual["timestamps"], [0])
        self.assertEqual(actual["quality"]["support_successor_rows"], 1)
        self.assertEqual(actual["coverage"]["valid_seconds"], [30.])

    def test_fractional_boundaries_are_integrated_without_grid_rounding(self):
        self.fixture([line("-0.5", 20, 2), line("7.25", 40, 4), line("15.75", 60, 6), line("29.5", 80, 8), line("30.25")])
        actual = read_block_window(self.path, 0, 30)
        expected = (7.25 * 20 + 8.5 * 40 + 13.75 * 60 + .5 * 80) / 30
        self.assertAlmostEqual(actual["values"][0, 0], expected)
        self.assertEqual(actual["coverage"]["valid_seconds"], [30.])

    def test_issues_left_hold_propagates_across_block_boundary(self):
        self.fixture([line(t, issues=int(t == 24)) for t in (0, 8, 16, 24, 32, 40, 48, 56, 64)])
        actual = read_block_window(self.path, 0, 60)
        self.assertEqual(actual["quality"]["valid_blocks"], 0)
        self.assertEqual(actual["coverage"]["valid_seconds"], [24., 28.])
        self.assertEqual(actual["quality"]["supported_invalid_seconds"], 8.)

    def test_invalid_right_row_does_not_retroactively_invalidate_left(self):
        self.fixture([line(t, issues=int(t == 30)) for t in (0, 8, 16, 24, 30)])
        actual = read_block_window(self.path, 0, 30)
        self.assertEqual(actual["quality"]["valid_blocks"], 1)
        self.assertEqual(actual["quality"]["rows_issues"], 1)

    def test_large_gap_is_not_held_even_partially(self):
        self.fixture([line(t) for t in (0, 8, 32, 40, 48, 56, 64)])
        actual = read_block_window(self.path, 0, 60)
        self.assertEqual(actual["coverage"]["valid_seconds"], [8., 28.])
        self.assertEqual(actual["coverage"]["large_gap_seconds"], [22., 2.])
        self.assertEqual(actual["quality"]["valid_blocks"], 0)

    def test_missing_predecessor_rejects_only_incompletely_covered_block(self):
        self.fixture([line(t) for t in (2, 10, 18, 26, 34, 42, 50, 58, 66)])
        actual = read_block_window(self.path, 0, 60)
        np.testing.assert_array_equal(actual["timestamps"], [30])
        self.assertEqual(actual["quality"]["uncovered_boundary_seconds"], 2.)

    def test_no_extrapolation_after_last_observation(self):
        self.fixture([line(t) for t in (0, 8, 16, 24, 29)])
        actual = read_block_window(self.path, 0, 30)
        self.assertEqual(actual["quality"]["valid_blocks"], 0)
        self.assertEqual(actual["coverage"]["valid_seconds"], [29.])
        self.assertEqual(source_bounds(self.path), (0, 29))

    def test_nonfinite_negative_and_nonnumeric_selected_values_invalidate_support(self):
        for value in ("NaN", "inf", "-1", "bad"):
            with self.subTest(value=value):
                self.fixture([line(t, appliance=value if t == 8 else 10) for t in (0, 8, 16, 24, 32)])
                actual = read_block_window(self.path, 0, 30)
                self.assertEqual(actual["coverage"]["valid_seconds"], [22.])
                self.assertEqual(actual["quality"]["valid_blocks"], 0)

    def test_unselected_channel_is_not_used_for_validity(self):
        self.fixture([line(t, overrides={"Appliance9": "NaN"}) for t in (0, 8, 16, 24, 32)])
        self.assertEqual(read_block_window(self.path, 0, 30)["quality"]["valid_blocks"], 1)
        self.assertEqual(read_block_window(self.path, 0, 30, channels=("Appliance9",))["quality"]["valid_blocks"], 0)

    def test_duplicate_and_unsorted_selected_rows_raise(self):
        for timestamps in ((0, 8, 8, 24, 32), (0, 16, 8, 24, 32)):
            self.fixture([line(t) for t in timestamps])
            with self.assertRaises(ValueError):
                read_block_window(self.path, 0, 30)

    def test_schema_timestamp_and_channel_errors_raise(self):
        self.fixture([line(0), line(8)], header="Unix,Time,Aggregate")
        with self.assertRaises(ValueError):
            source_bounds(self.path)
        self.fixture([line(0), line("bad"), line(32)])
        with self.assertRaises(ValueError):
            read_block_window(self.path, 0, 30)
        self.fixture([line(t) for t in (0, 8, 16, 24, 32)])
        for channels in ((), ("Aggregate",), ("Appliance1", "Appliance1"), ("Appliance10",)):
            with self.assertRaises(ValueError):
                read_block_window(self.path, 0, 30, channels=channels)
        with self.assertRaises(ValueError):
            read_block_window(self.path, 0, 30, max_gap_seconds=17)

    def test_supporting_rows_are_in_the_reproducible_hash(self):
        rows = [line(t) for t in (-20, -12, -4, 4, 12, 20, 28, 36, 44)]
        header = self.fixture(rows)
        actual = read_block_window(self.path, 0, 30)
        expected = hashlib.sha256(header + b"".join(rows[1:8])).hexdigest()
        self.assertEqual(actual["source_window_sha256"], expected)
        self.assertEqual(actual["source_window_sha256"], read_block_window(self.path, 0, 30)["source_window_sha256"])
        self.assertEqual(actual["quality"]["support_predecessor_rows"], 2)

    def test_byte_seek_over_64k_matches_hand_selected_support(self):
        rows = [line(t, aggregate=100 + t // 8) for t in range(0, 40008, 8)]
        header = self.fixture(rows)
        self.assertGreater(self.path.stat().st_size, 65536)
        start, end = 32001, 32061
        actual = read_block_window(self.path, start, end)
        first = (start - 16 + 7) // 8
        last = (end + 7) // 8
        selected = rows[first:last + 1]
        self.assertEqual(actual["source_window_sha256"], hashlib.sha256(header + b"".join(selected)).hexdigest())
        self.assertEqual(actual["quality"]["valid_blocks"], 2)
        for block, block_start in enumerate((start, start + 30)):
            expected = sum(100 + (second // 8) for second in range(block_start, block_start + 30)) / 30
            self.assertAlmostEqual(actual["values"][block, 0], expected)

    def test_partial_final_block_is_not_returned(self):
        self.fixture([line(t) for t in range(0, 48, 8)])
        actual = read_block_window(self.path, 0, 35)
        self.assertEqual(actual["quality"]["partial_final_seconds"], 5)
        self.assertEqual(actual["quality"]["valid_blocks"], 1)

    def test_zero_prevalence_counts_only_valid_rows_and_retained_blocks(self):
        self.fixture([line(t, aggregate=0 if t == 0 else 100, appliance=0,
                           issues=int(t == 40), overrides={"Appliance2": 0 if t < 32 else 5})
                      for t in (0, 8, 16, 24, 30, 32, 40, 48, 56, 64)])
        actual = read_block_window(self.path, 0, 60, channels=("Appliance1", "Appliance2"))
        quality = actual["quality"]
        self.assertEqual(quality["valid_source_rows"], 9)
        self.assertEqual(quality["zero_source_rows_by_channel"],
                         {"Aggregate": 1, "Appliance1": 9, "Appliance2": 5})
        self.assertEqual(quality["all_selected_appliances_zero_source_rows"], 5)
        self.assertEqual(quality["zero_retained_blocks_by_channel"],
                         {"Aggregate": 0, "Appliance1": 1, "Appliance2": 1})
        self.assertEqual(quality["all_selected_appliances_zero_retained_blocks"], 1)
        self.assertIn("does not identify imputation", actual["metadata"]["zero_counter_interpretation"])


if __name__ == "__main__":
    unittest.main()
