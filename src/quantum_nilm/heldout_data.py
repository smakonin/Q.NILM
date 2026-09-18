"""Timestamp-only chronological sampling and gap-safe R1Hz extraction.

The source contract is a UTF-8 CSV with one physical line per row, UNIX seconds
in the first ``unix_ts`` column, and globally sorted timestamps. Seeking relies
on that global ordering; it is checked within extracted windows, not by scanning
the entire source. Selection uses elapsed calendar time, never load activity or
measurement quality. Rejected windows/blocks must not be replaced adaptively.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import BinaryIO

import numpy as np


def _timestamp(raw: bytes) -> int:
    try:
        return int(raw.split(b",", 1)[0])
    except (ValueError, IndexError) as exc:
        raise ValueError("Every source row must begin with an integer UNIX timestamp") from exc


def _header(handle: BinaryIO) -> tuple[list[str], int]:
    handle.seek(0)
    raw = handle.readline()
    try:
        names = next(csv.reader([raw.decode("utf-8-sig").strip()]))
    except (UnicodeDecodeError, StopIteration, csv.Error) as exc:
        raise ValueError("Invalid CSV header") from exc
    if not names or names[0] != "unix_ts" or len(names) != len(set(names)):
        raise ValueError("Expected a unique-column CSV header beginning with unix_ts")
    return names, handle.tell()


def _last_nonempty_line(handle: BinaryIO, data_start: int) -> bytes:
    handle.seek(0, 2)
    cursor = handle.tell()
    suffix = b""
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
    raise ValueError("CSV contains no data rows")


def source_bounds(path: str | Path) -> tuple[int, int]:
    """Return first timestamp and last timestamp + 1 using boundary lines only.

    This does not certify global ordering, completeness, or load quality.
    """
    with Path(path).open("rb") as handle:
        _, data_start = _header(handle)
        raw = handle.readline()
        if not raw.strip():
            raise ValueError("CSV contains no first data row")
        first = _timestamp(raw)
        last = _timestamp(_last_nonempty_line(handle, data_start))
    if last < first:
        raise ValueError("Source boundaries are not chronologically ordered")
    return first, last + 1


def make_chronological_manifest(
    first: int,
    end: int,
    counts: tuple[int, int, int] = (90, 30, 30),
    window_seconds: int = 86400,
    block_seconds: int = 30,
) -> dict:
    """Freeze 60/20/20 calendar partitions and evenly spaced fixed windows.

    All starts align to the block grid relative to ``first``. Partition cutoffs
    are also aligned down to this grid. Within each partition, starts span the
    earliest and latest possible aligned positions (the sole window is centered
    if count = 1). No measurements are read. A window must fit wholly in its
    partition, and requested windows must not overlap. A zero count is allowed.
    """
    for name, value in (("first", first), ("end", end), ("window_seconds", window_seconds),
                        ("block_seconds", block_seconds)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if end <= first or block_seconds <= 0 or window_seconds <= 0:
        raise ValueError("Expected increasing bounds and positive durations")
    if window_seconds % block_seconds:
        raise ValueError("window_seconds must be a multiple of block_seconds")
    if len(counts) != 3 or any(isinstance(n, bool) or not isinstance(n, (int, np.integer))
                               or n < 0 for n in counts):
        raise ValueError("counts must contain three nonnegative integers")
    total = end - first
    cutoffs = [first, first + (3 * total // 5 // block_seconds) * block_seconds,
               first + (4 * total // 5 // block_seconds) * block_seconds, end]
    splits = {}
    windows = []
    for index, (name, count) in enumerate(zip(("train", "validation", "test"), counts)):
        left, right = cutoffs[index:index + 2]
        usable_slots = (right - left - window_seconds) // block_seconds
        if count and (usable_slots < 0 or right - left < count * window_seconds):
            raise ValueError(f"Requested non-overlapping windows do not fit {name} partition")
        if count == 1:
            offsets = [usable_slots // 2]
        elif count:
            offsets = [i * usable_slots // (count - 1) for i in range(count)]
        else:
            offsets = []
        selected = []
        for number, offset in enumerate(offsets):
            start = int(left + offset * block_seconds)
            item = {"id": f"{name}-{number:03d}", "split": name,
                    "start": start, "end": start + int(window_seconds),
                    "start_unix": start, "end_unix": start + int(window_seconds)}
            if selected and item["start"] < selected[-1]["end"]:
                raise ValueError(f"Aligned windows overlap in {name} partition")
            selected.append(item)
            windows.append(item)
        splits[name] = {"start": int(left), "end": int(right), "count": int(count),
                        "windows": selected}
    return {"schema_version": 1, "selection": "timestamp-only evenly spaced fixed calendar windows",
            "partition_fractions": [0.6, 0.2, 0.2],
            "source_bounds": {"first": int(first), "end_exclusive": int(end)},
            "window_seconds": int(window_seconds), "block_seconds": int(block_seconds),
            "splits": splits, "windows": windows,
            "replacement_policy": "No replacement based on measurements or data quality",
            "ordering_assumption": "Source globally sorted; checked only within selected windows"}


def _seek_timestamp(handle: BinaryIO, data_start: int, target: int) -> int:
    """Seek the first row with timestamp >= target in a sorted line-oriented CSV.

    Byte bisection narrows to at most 64 KiB, then a forward scan finds the exact
    boundary. Bisection always retains whole-line boundaries; a line larger than
    the remaining interval falls back safely to the forward scan.
    """
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
        if not raw.strip():
            high = position
        elif _timestamp(raw) < target:
            low = handle.tell()
        else:
            high = position
    handle.seek(low)
    while True:
        position = handle.tell()
        raw = handle.readline()
        if not raw or (raw.strip() and _timestamp(raw) >= target):
            handle.seek(position)
            return position


def read_block_window(
    path: str | Path,
    start: int,
    end: int,
    channels: tuple[str, ...] = ("main", "dryr", "frdg", "vacu"),
    block_seconds: int = 30,
    exclude_markers: tuple[str, ...] = ("s", "+"),
) -> dict:
    """Extract complete valid 1 Hz blocks without squeezing gaps together.

    Blocks are anchored to ``start`` and accepted only when every expected second
    has exactly one non-excluded row with finite requested values. Non-increasing
    timestamps fail the window rather than silently choosing a duplicate. Invalid
    timestamps fail because their time bucket cannot be identified. Physical CSV
    lines are hashed exactly as read for timestamps in [start, end), including
    excluded and nonnumeric rows; the header is not included in that digest.
    """
    for name, value in (("start", start), ("end", end), ("block_seconds", block_seconds)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if end <= start or block_seconds <= 0 or not channels or len(set(channels)) != len(channels):
        raise ValueError("Expected increasing bounds, positive block size, and unique channels")
    full_blocks = (end - start) // block_seconds
    expected_blocks = math.ceil((end - start) / block_seconds)
    # Bounded selected windows, not an allocation proportional to the 5 GB source.
    sums = np.zeros((full_blocks, len(channels)), dtype=np.float64)
    valid_counts = np.zeros(full_blocks, dtype=np.int64)
    raw_counts = np.zeros(full_blocks, dtype=np.int64)
    excluded_flags = np.zeros(full_blocks, dtype=bool)
    nonnumeric_flags = np.zeros(full_blocks, dtype=bool)
    quality = {"total_rows": 0, "valid_rows": 0, "excluded_marker": 0,
               "nonnumeric": 0, "duplicates": 0, "gaps": 0,
               "missing_seconds": 0, "expected_complete_blocks": full_blocks,
               "partial_final_seconds": (end - start) % block_seconds}
    digest = hashlib.sha256()
    previous = start - 1
    with Path(path).open("rb") as handle:
        names, data_start = _header(handle)
        missing = set(("marker", *channels)) - set(names)
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        positions = [names.index(channel) for channel in channels]
        marker_index = names.index("marker")
        _seek_timestamp(handle, data_start, int(start))
        while True:
            raw = handle.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            timestamp = _timestamp(raw)
            if timestamp >= end:
                break
            if timestamp < start or timestamp <= previous:
                kind = "duplicate" if timestamp == previous else "unsorted"
                raise ValueError(f"{kind} timestamp {timestamp} in selected window")
            if timestamp > previous + 1:
                quality["gaps"] += 1
                quality["missing_seconds"] += timestamp - previous - 1
            previous = timestamp
            digest.update(raw)
            quality["total_rows"] += 1
            block = (timestamp - start) // block_seconds
            if block < full_blocks:
                raw_counts[block] += 1
            try:
                row = next(csv.reader([raw.decode("utf-8")]))
                if len(row) != len(names):
                    raise ValueError("Wrong column count")
                excluded = row[marker_index] in exclude_markers
            except (UnicodeDecodeError, csv.Error, ValueError, StopIteration):
                excluded = False
                row = []
            if excluded:
                quality["excluded_marker"] += 1
                if block < full_blocks:
                    excluded_flags[block] = True
                continue
            try:
                values = [float(row[i]) for i in positions]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("Non-finite value")
            except (ValueError, IndexError):
                quality["nonnumeric"] += 1
                if block < full_blocks:
                    nonnumeric_flags[block] = True
                continue
            quality["valid_rows"] += 1
            if block < full_blocks:
                sums[block] += values
                valid_counts[block] += 1
    if previous < end - 1:
        quality["gaps"] += 1
        quality["missing_seconds"] += end - previous - 1
    accepted = valid_counts == block_seconds
    timestamps = start + np.flatnonzero(accepted) * block_seconds
    values = sums[accepted] / block_seconds
    run_ids = np.zeros(len(timestamps), dtype=np.int64)
    if len(timestamps) > 1:
        run_ids[1:] = np.cumsum(np.diff(timestamps) != block_seconds)
    quality.update({"accepted_blocks": int(np.count_nonzero(accepted)),
                    "dropped_blocks": int(expected_blocks - np.count_nonzero(accepted)),
                    "invalid_complete_blocks": int(full_blocks - np.count_nonzero(accepted)),
                    "blocks_with_missing_seconds": int(np.count_nonzero(raw_counts < block_seconds)),
                    "blocks_with_excluded_markers": int(np.count_nonzero(excluded_flags)),
                    "blocks_with_nonnumeric_values": int(np.count_nonzero(nonnumeric_flags)),
                    "contiguous_runs": int(run_ids[-1] + 1) if len(run_ids) else 0})
    return {"timestamps": timestamps.astype(np.int64), "values": values,
            "run_ids": run_ids, "quality": quality, "channels": list(channels),
            "start": int(start), "end": int(end), "block_seconds": int(block_seconds),
            "source_window_sha256": digest.hexdigest()}
