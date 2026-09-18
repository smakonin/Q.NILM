#!/usr/bin/env python3
"""Extract a bounded, publication-safe R1Hz real-power window."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


DEFAULT_SOURCE = Path(
    "/Users/stephen/Documents/Research/Datasets/R1Hz/recovered/power.csv"
)
DEFAULT_CHANNELS = ["main", "boil", "cwsh", "dryr", "dwsh", "frdg", "vacu"]


def local_timestamp(value: str) -> int:
    local_time = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=ZoneInfo("America/Vancouver")
    )
    return int(local_time.timestamp())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--start", required=True)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    start = local_timestamp(args.start)
    end = start + args.seconds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    retained = 0
    skipped_synthetic = 0

    with args.source.open(newline="") as source_handle, args.output.open(
        "w", newline=""
    ) as output_handle:
        reader = csv.DictReader(source_handle)
        missing = [channel for channel in args.channels if channel not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing channels: {missing}")
        fieldnames = ["unix_ts", "marker", "date", "time", *args.channels]
        writer = csv.DictWriter(
            output_handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for row in reader:
            timestamp = int(row["unix_ts"])
            if timestamp < start:
                continue
            if timestamp >= end:
                break
            if row["marker"] == "s":
                skipped_synthetic += 1
                continue
            writer.writerow({field: row[field] for field in fieldnames})
            retained += 1

    if retained == 0:
        raise RuntimeError("No measured rows were extracted")
    print(
        f"Extracted {retained} measured rows to {args.output}; "
        f"excluded {skipped_synthetic} synthetic rows."
    )


if __name__ == "__main__":
    main()
