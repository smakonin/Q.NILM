"""Bounded, gap-aware reads of the official cleaned REFIT CSV files.

The published cleaning already forward-filled NaNs and replaced IAM spikes
above 4,000 W with zero.  Those unflagged imputations cannot be undone here.
This reader adds no filling: it integrates left-held observations only between
successive source timestamps separated by at most 16 seconds, never beyond the
last row.  Only completely valid 30-second blocks are returned.

Byte seeking assumes globally sorted timestamps.  Strict increasing order is
checked over every selected/supporting row, not certified over the whole file.
"""

from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
import hashlib
from pathlib import Path

import numpy as np


HEADER = ("Time", "Unix", "Aggregate", *(f"Appliance{i}" for i in range(1, 10)), "Issues")
CLEANING_CAVEAT = (
    "Official cleaned REFIT already forward-filled NaNs and replaced IAM spikes above 4000 W with zeros; "
    "unflagged imputation cannot be identified or undone. Sensor readings are asynchronously polled."
)


def _timestamp(raw: bytes) -> Decimal:
    try:
        value = Decimal(raw.split(b",", 2)[1].decode("ascii"))
    except (IndexError, UnicodeDecodeError, InvalidOperation) as error:
        raise ValueError("Every REFIT row must have a numeric Unix timestamp in column two") from error
    if not value.is_finite():
        raise ValueError("REFIT Unix timestamps must be finite")
    return value


def _header(handle):
    handle.seek(0)
    raw = handle.readline()
    try:
        names = tuple(next(csv.reader([raw.decode("utf-8-sig").strip()])))
    except (UnicodeDecodeError, csv.Error, StopIteration) as error:
        raise ValueError("Invalid REFIT CSV header") from error
    if names != HEADER:
        raise ValueError(f"Expected the official cleaned REFIT schema: {HEADER}")
    return raw, handle.tell()


def _last_line(handle, data_start):
    handle.seek(0, 2)
    cursor, suffix = handle.tell(), b""
    while cursor > data_start:
        left = max(data_start, cursor - 65536)
        handle.seek(left)
        suffix = handle.read(cursor - left) + suffix
        stripped = suffix.rstrip(b"\r\n")
        if b"\n" in stripped:
            return stripped.rsplit(b"\n", 1)[-1]
        cursor = left
    if suffix.rstrip(b"\r\n"):
        return suffix.rstrip(b"\r\n")
    raise ValueError("REFIT file has no source rows")


def _number(value: Decimal):
    return int(value) if value == value.to_integral_value() else float(value)


def source_bounds(path: str | Path) -> tuple[int | float, int | float]:
    """Read boundary timestamps only: [first timestamp, last timestamp).

    The last source row has no supported extrapolation interval, so its
    timestamp itself is the exclusive bound, not timestamp plus one second.
    This does not establish uninterrupted coverage or global sortedness.
    Power values are neither parsed nor used.
    """
    with Path(path).open("rb") as handle:
        _, data_start = _header(handle)
        first = _timestamp(handle.readline())
        last = _timestamp(_last_line(handle, data_start))
    if last <= first:
        raise ValueError("At least two increasing boundary timestamps are required")
    return _number(first), _number(last)


def _seek_timestamp(handle, data_start, target):
    """Seek the first timestamp >= target, retaining whole-line boundaries."""
    handle.seek(0, 2)
    low, high = data_start, handle.tell()
    while high - low > 65536:
        middle = (low + high) // 2
        handle.seek(middle - 1)
        if handle.read(1) != b"\n":
            handle.readline()
        position = handle.tell()
        if position >= high:
            break
        raw = handle.readline()
        if _timestamp(raw) < target:
            low = handle.tell()
        else:
            high = position
    handle.seek(low)
    while True:
        position = handle.tell()
        raw = handle.readline()
        if not raw or _timestamp(raw) >= target:
            handle.seek(position)
            return position


def _values(raw, positions):
    """Classify a left row; malformed measurements invalidate its support."""
    try:
        row = next(csv.reader([raw.decode("utf-8")]))
    except (UnicodeDecodeError, csv.Error, StopIteration):
        row = []
    if len(row) != len(HEADER):
        return None, {"invalid_schema": True, "issues": False, "nonfinite": False, "negative": False}
    try:
        issue = float(row[-1])
        issue_bad = not np.isfinite(issue) or issue != 0
    except ValueError:
        issue_bad = True
    try:
        values = np.asarray([float(row[position]) for position in positions], dtype=float)
        nonfinite = not np.all(np.isfinite(values))
        negative = bool(np.any(values < 0))
    except ValueError:
        values, nonfinite, negative = None, True, False
    flags = {"invalid_schema": False, "issues": bool(issue_bad),
             "nonfinite": bool(nonfinite), "negative": negative}
    return values, flags


def read_block_window(path: str | Path, start: int, end: int,
                      channels: tuple[str, ...] = ("Appliance1",),
                      block_seconds: int = 30, max_gap_seconds: int = 16) -> dict:
    """Integrate supported left-held power into completely valid fixed blocks.

    ``channels`` contains distinct selected Appliance1..Appliance9 names;
    Aggregate is always returned first.  A left row is usable only when
    Issues=0 and all requested powers are finite and nonnegative.  The right
    row supplies the interval endpoint; its Issues flag does not retroactively
    invalidate the preceding left-held observation.

    Supporting reads start at the first timestamp >= start-16 and include
    the first source row at or after end, if present.  SHA256 covers the raw
    header followed by all scanned raw source rows, including boundary rows
    and unusable rows.  Decimal timestamp arithmetic gives exact fractional-
    second overlap/coverage decisions, without rounding to a sampling grid.
    Partial final blocks are excluded, not extrapolated or repaired.
    """
    for name, value in (("start", start), ("end", end), ("block_seconds", block_seconds),
                        ("max_gap_seconds", max_gap_seconds)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if end <= start or block_seconds != 30 or max_gap_seconds != 16:
        raise ValueError("Require increasing window bounds and the fixed 30-second / 16-second support contract")
    channels = tuple(channels)
    if (not channels or len(set(channels)) != len(channels)
            or any(channel not in HEADER[3:12] for channel in channels)):
        raise ValueError("Select distinct Appliance1..Appliance9 channels; Aggregate is added automatically")
    n_blocks = (end - start) // block_seconds
    if n_blocks > 1_000_000:
        raise ValueError("Selected window exceeds the bounded one-million-block allocation")
    names = ("Aggregate", *channels)
    positions = [HEADER.index(name) for name in names]
    sums = np.zeros((n_blocks, len(names)), dtype=float)
    valid_coverage = [Decimal(0) for _ in range(n_blocks)]
    supported_coverage = [Decimal(0) for _ in range(n_blocks)]
    gap_coverage = [Decimal(0) for _ in range(n_blocks)]
    interval_width = Decimal(block_seconds)
    window_start, window_end = Decimal(int(start)), Decimal(int(end))
    complete_end = window_start + n_blocks * interval_width
    quality = {"source_rows_scanned": 0, "source_rows_within_window": 0,
               "support_predecessor_rows": 0, "support_successor_rows": 0,
               "rows_invalid_schema": 0, "rows_issues": 0, "rows_nonfinite": 0,
               "rows_negative": 0, "intervals_examined": 0,
               "valid_source_rows": 0,
               "zero_source_rows_by_channel": {name: 0 for name in names},
               "all_selected_appliances_zero_source_rows": 0,
               "overlapping_intervals_over_16_seconds": 0,
               "overlapping_invalid_left_intervals": 0,
               "expected_complete_blocks": n_blocks,
               "partial_final_seconds": (end - start) % block_seconds}
    digest = hashlib.sha256()
    previous = None
    first_scanned = last_scanned = None
    with Path(path).open("rb") as handle:
        raw_header, data_start = _header(handle)
        digest.update(raw_header)
        _seek_timestamp(handle, data_start, window_start - max_gap_seconds)
        for raw in handle:
            timestamp = _timestamp(raw)
            if previous is not None and timestamp <= previous[0]:
                kind = "duplicate" if timestamp == previous[0] else "unsorted"
                raise ValueError(f"{kind} REFIT timestamp in selected/supporting rows: {timestamp}")
            digest.update(raw)
            values, flags = _values(raw, positions)
            quality["source_rows_scanned"] += 1
            quality["source_rows_within_window"] += int(window_start <= timestamp < window_end)
            quality["support_predecessor_rows"] += int(timestamp < window_start)
            quality["support_successor_rows"] += int(timestamp >= window_end)
            for key, bad in flags.items():
                quality[f"rows_{key}"] += int(bad)
            if not any(flags.values()):
                quality["valid_source_rows"] += 1
                for name, value in zip(names, values):
                    quality["zero_source_rows_by_channel"][name] += int(value == 0)
                quality["all_selected_appliances_zero_source_rows"] += int(np.all(values[1:] == 0))
            first_scanned = timestamp if first_scanned is None else first_scanned
            last_scanned = timestamp
            if previous is not None:
                left_time, left_values, left_flags = previous
                delta = timestamp - left_time
                quality["intervals_examined"] += 1
                left, right = max(left_time, window_start), min(timestamp, complete_end)
                if right > left:
                    oversized = delta > max_gap_seconds
                    invalid_left = any(left_flags.values())
                    quality["overlapping_intervals_over_16_seconds"] += int(oversized)
                    quality["overlapping_invalid_left_intervals"] += int(not oversized and invalid_left)
                    while left < right:
                        block = int((left - window_start) // interval_width)
                        stop = min(right, window_start + (block + 1) * interval_width)
                        seconds = stop - left
                        if oversized:
                            gap_coverage[block] += seconds
                        else:
                            supported_coverage[block] += seconds
                            if not invalid_left:
                                valid_coverage[block] += seconds
                                sums[block] += float(seconds) * left_values
                        left = stop
            previous = (timestamp, values, flags)
            if timestamp >= window_end:
                break
    complete = np.asarray([seconds == interval_width for seconds in valid_coverage], dtype=bool)
    valid_seconds = sum(valid_coverage, Decimal(0))
    supported_seconds = sum(supported_coverage, Decimal(0))
    gap_seconds = sum(gap_coverage, Decimal(0))
    requested_complete_seconds = Decimal(n_blocks * block_seconds)
    quality.update({"valid_blocks": int(np.sum(complete)), "rejected_blocks": int(n_blocks - np.sum(complete)),
                    "zero_retained_blocks_by_channel": {name: int(np.sum(sums[complete, i] == 0))
                                                        for i, name in enumerate(names)},
                    "all_selected_appliances_zero_retained_blocks": int(np.sum(np.all(sums[complete, 1:] == 0, axis=1))),
                    "valid_covered_seconds": float(valid_seconds),
                    "supported_covered_seconds": float(supported_seconds),
                    "supported_invalid_seconds": float(supported_seconds - valid_seconds),
                    "unsupported_large_gap_seconds": float(gap_seconds),
                    "uncovered_boundary_seconds": float(requested_complete_seconds - supported_seconds - gap_seconds),
                    "first_scanned_unix": None if first_scanned is None else _number(first_scanned),
                    "last_scanned_unix": None if last_scanned is None else _number(last_scanned)})
    grid = np.arange(n_blocks, dtype=np.int64) * block_seconds + start
    return {"timestamps": grid[complete], "values": sums[complete] / block_seconds,
            "channels": list(names), "quality": quality,
            "coverage": {"block_start_unix": grid.tolist(),
                         "valid_seconds": [float(x) for x in valid_coverage],
                         "supported_seconds": [float(x) for x in supported_coverage],
                         "large_gap_seconds": [float(x) for x in gap_coverage]},
            "source_window_sha256": digest.hexdigest(),
            "source_hash_scope": "raw header plus every scanned row from first Unix >= start-16 through first Unix >= end, when present",
            "metadata": {"cleaned_data_caveat": CLEANING_CAVEAT,
                         "integration": "piecewise-constant left hold; exact Decimal overlap boundaries; no extrapolation",
                         "maximum_support_seconds": max_gap_seconds, "block_seconds": block_seconds,
                         "row_validity": "left row Issues=0; Aggregate and selected appliances finite and nonnegative",
                         "zero_counter_scope": "source zeros count all scanned rows including supporting boundaries, among rows valid for every requested channel and Issues=0; retained-block zeros count exact zero integrated means",
                         "zero_counter_interpretation": "zero prevalence does not identify imputation; genuine zero readings are retained",
                         "ordering_assumption": "globally sorted for byte seek; strictly increasing checked only in selected/supporting rows"}}
