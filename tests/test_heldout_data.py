"""Synthetic-only checks: sampling does not use test measurements."""

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from quantum_nilm.heldout_data import (
    _header, _seek_timestamp, make_chronological_manifest, read_block_window, source_bounds,
)


class HeldoutDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "power.csv"

    def write_rows(self, rows, newline=True):
        raw = b"unix_ts,marker,main,dryr,frdg,vacu,note\n"
        raw += b"\n".join(",".join(map(str, row)).encode() for row in rows)
        if newline:
            raw += b"\n"
        self.path.write_bytes(raw)
        return raw

    def rows(self, start, stop):
        return [[t, "", t + 10, t + 1, t + 2, t + 3, "x" * (t % 13)]
                for t in range(start, stop)]

    def test_source_bounds_first_last_and_no_final_newline(self):
        self.write_rows(self.rows(101, 108), newline=False)
        self.assertEqual(source_bounds(self.path), (101, 108))
        self.write_rows(self.rows(101, 102))
        self.assertEqual(source_bounds(self.path), (101, 102))

    def test_manifest_is_disjoint_aligned_and_timestamp_only(self):
        first, end = 123456789, 123456789 + 757 * 86400 + 17
        plan = make_chronological_manifest(first, end)
        self.assertEqual([plan["splits"][s]["count"] for s in ("train", "validation", "test")],
                         [90, 30, 30])
        self.assertEqual(len(plan["windows"]), 150)
        previous_end = first
        for window in plan["windows"]:
            split = plan["splits"][window["split"]]
            self.assertGreaterEqual(window["start"], split["start"])
            self.assertLessEqual(window["end"], split["end"])
            self.assertGreaterEqual(window["start"], previous_end)
            self.assertEqual((window["start"] - first) % 30, 0)
            self.assertEqual(window["end"] - window["start"], 86400)
            previous_end = window["end"]
        self.assertEqual(plan, make_chronological_manifest(first, end))

    def test_manifest_small_single_and_zero_counts(self):
        plan = make_chronological_manifest(100, 1100, (1, 0, 1), 100, 10)
        self.assertEqual(plan["windows"][0]["start"], 350)
        self.assertEqual(plan["windows"][1]["start"], 950)
        for args in ((0, 100, (1, 1, 1), 100, 10),
                     (0, 1000, (1, 1), 100, 10),
                     (0, 1000, (1, -1, 1), 100, 10),
                     (0, 1000, (1, 1, 1), 101, 10)):
            with self.assertRaises(ValueError):
                make_chronological_manifest(*args)

    def test_byte_seek_variable_lines_and_timestamp_gaps(self):
        rows = self.rows(1000, 9000)
        rows = [row for row in rows if row[0] % 7 != 0]
        rows[500][-1] = "z" * 100000  # one line exceeds bisection tail size
        self.write_rows(rows, newline=False)
        available = [row[0] for row in rows]
        with self.path.open("rb") as handle:
            _, first_byte = _header(handle)
            for target in (0, 999, 1000, 1001, 1008, 1500, 1583, 4499, 8999, 9000, 10000):
                position = _seek_timestamp(handle, first_byte, target)
                handle.seek(position)
                found = handle.readline()
                expected = next((t for t in available if t >= target), None)
                self.assertEqual(int(found.split(b",", 1)[0]) if found else None, expected)

    def test_clean_means_digest_and_anchoring(self):
        raw = self.write_rows(self.rows(100, 140))
        data = read_block_window(self.path, 105, 135, block_seconds=10)
        np.testing.assert_array_equal(data["timestamps"], [105, 115, 125])
        np.testing.assert_allclose(data["values"][:, 0], [119.5, 129.5, 139.5])
        np.testing.assert_array_equal(data["run_ids"], [0, 0, 0])
        expected_raw = b"".join(line for line in raw.splitlines(keepends=True)[1:]
                                if 105 <= int(line.split(b",", 1)[0]) < 135)
        self.assertEqual(data["source_window_sha256"], hashlib.sha256(expected_raw).hexdigest())
        self.assertEqual(data["quality"]["total_rows"], 30)
        self.assertEqual(data["quality"]["dropped_blocks"], 0)

    def test_invalid_markers_missing_nonnumeric_preserve_time_gaps(self):
        rows = self.rows(0, 70)
        rows[12][1] = "s"
        rows[21][1] = "+"
        rows[33][2] = "nan"
        rows = [row for row in rows if row[0] != 43]
        self.write_rows(rows)
        data = read_block_window(self.path, 0, 70, block_seconds=10)
        np.testing.assert_array_equal(data["timestamps"], [0, 50, 60])
        np.testing.assert_array_equal(data["run_ids"], [0, 1, 1])
        q = data["quality"]
        self.assertEqual(q["total_rows"], 69)
        self.assertEqual(q["excluded_marker"], 2)
        self.assertEqual(q["nonnumeric"], 1)
        self.assertEqual(q["missing_seconds"], 1)
        self.assertEqual(q["gaps"], 1)
        self.assertEqual(q["dropped_blocks"], 4)
        self.assertEqual(q["blocks_with_excluded_markers"], 2)
        self.assertEqual(q["blocks_with_nonnumeric_values"], 1)
        self.assertEqual(q["blocks_with_missing_seconds"], 1)

    def test_window_boundary_gaps_empty_window_and_partial_block(self):
        self.write_rows(self.rows(3, 29))
        data = read_block_window(self.path, 0, 35, block_seconds=10)
        np.testing.assert_array_equal(data["timestamps"], [10])
        self.assertEqual(data["quality"]["missing_seconds"], 9)
        self.assertEqual(data["quality"]["gaps"], 2)
        self.assertEqual(data["quality"]["dropped_blocks"], 3)
        self.assertEqual(data["quality"]["partial_final_seconds"], 5)
        empty = read_block_window(self.path, 100, 120, block_seconds=10)
        self.assertEqual(empty["values"].shape, (0, 4))
        self.assertEqual(empty["quality"]["missing_seconds"], 20)
        self.assertEqual(empty["quality"]["gaps"], 1)

    def test_duplicate_or_unsorted_timestamps_fail_window(self):
        for rows in (self.rows(0, 10) + self.rows(9, 20),
                     self.rows(0, 10) + self.rows(8, 20)):
            self.write_rows(rows)
            with self.assertRaisesRegex(ValueError, "duplicate|unsorted"):
                read_block_window(self.path, 0, 20, block_seconds=10)

    def test_missing_schema_invalid_bounds_and_nonfinite_values(self):
        self.path.write_text("unix_ts,marker,dryr\n0,,1\n")
        with self.assertRaisesRegex(ValueError, "missing columns"):
            read_block_window(self.path, 0, 10)
        self.write_rows(self.rows(0, 20))
        for start, end, block in ((0, 0, 10), (0, 10, 0), (0.1, 10, 10)):
            with self.assertRaises(ValueError):
                read_block_window(self.path, start, end, block_seconds=block)
        rows = self.rows(0, 20)
        rows[12][3] = "inf"
        self.write_rows(rows)
        data = read_block_window(self.path, 0, 20, block_seconds=10)
        self.assertEqual(data["quality"]["nonnumeric"], 1)
        np.testing.assert_array_equal(data["timestamps"], [0])


if __name__ == "__main__":
    unittest.main()
