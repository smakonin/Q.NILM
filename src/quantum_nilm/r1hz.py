"""Deterministic preprocessing for R1Hz Q.NILM experiments."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


DEFAULT_CHANNELS = ("dryr", "frdg", "vacu")


def baseline_centered_aggregate(
    blocks: np.ndarray,
    baselines: np.ndarray,
    *,
    mode: str = "signed",
) -> np.ndarray:
    """Center measured total power without rectifying model residuals.

    The physical model is ``raw_total = sum(baselines) + powers @ states
    + residual``. Negative centered observations are valid residuals around
    the fitted OFF centroids, not negative physical consumption. Clipping
    them before averaging creates an upward bias. ``legacy-clipped`` exists
    solely to reproduce the original discovery pilot and Ocean benchmark.
    """
    values = np.asarray(blocks, dtype=float)
    offsets = np.asarray(baselines, dtype=float)
    if values.ndim != 2 or not all(values.shape):
        raise ValueError("blocks must be a nonempty time-by-channel matrix")
    if offsets.shape != (values.shape[1],):
        raise ValueError("Expected one baseline per channel")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(offsets)):
        raise ValueError("blocks and baselines must be finite")
    if mode == "signed":
        return values.sum(axis=1) - offsets.sum()
    if mode == "legacy-clipped":
        return np.maximum(values - offsets[None, :], 0.0).sum(axis=1)
    raise ValueError("mode must be 'signed' or 'legacy-clipped'")


def load_window(
    path: Path, channels: list[str] | tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Load a bounded R1Hz CSV while excluding synthetically recovered rows."""
    timestamps: list[int] = []
    values: list[list[float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["marker"] == "s":
                continue
            timestamps.append(int(row["unix_ts"]))
            values.append([float(row[channel]) for channel in channels])
    return np.asarray(timestamps), np.asarray(values, dtype=float)


def block_mean(values: np.ndarray, block_seconds: int) -> np.ndarray:
    """Return non-overlapping block means after dropping a partial final block."""
    usable = values.shape[0] - values.shape[0] % block_seconds
    if usable == 0:
        raise ValueError("Window is shorter than one block")
    return values[:usable].reshape(-1, block_seconds, values.shape[1]).mean(axis=1)


def infer_binary_reference(
    blocks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive auditable two-state circuit proxies with deterministic 1-D k-means.

    Min/max initialization is deliberate: brief loads such as a vacuum can occupy
    less than ten percent of a window and disappear under percentile thresholds.
    These labels are circuit-channel proxies, not manual appliance annotations.
    """
    states = np.zeros_like(blocks, dtype=np.int8)
    baselines = np.empty(blocks.shape[1], dtype=float)
    nominal = np.empty(blocks.shape[1], dtype=float)
    thresholds = np.empty(blocks.shape[1], dtype=float)
    for channel in range(blocks.shape[1]):
        values = blocks[:, channel]
        centers = np.array([float(np.min(values)), float(np.max(values))])
        if np.isclose(centers[0], centers[1]):
            baselines[channel] = centers[0]
            nominal[channel] = centers[1]
            thresholds[channel] = centers[0]
            continue
        for _ in range(100):
            labels = (
                np.abs(values - centers[1]) < np.abs(values - centers[0])
            ).astype(np.int8)
            if labels.min() == labels.max():
                break
            updated = np.array(
                [values[labels == label].mean() for label in (0, 1)], dtype=float
            )
            if np.allclose(updated, centers, rtol=0.0, atol=1e-9):
                centers = updated
                break
            centers = updated
        if centers[0] > centers[1]:
            centers = centers[::-1]
        thresholds[channel] = float(np.mean(centers))
        states[:, channel] = values > thresholds[channel]
        baselines[channel] = centers[0]
        nominal[channel] = centers[1]
    incremental = np.maximum(nominal - baselines, 1.0)
    return states, incremental, baselines, thresholds


def choose_segments(
    blocks: np.ndarray,
    states: np.ndarray,
    incremental: np.ndarray,
    baselines: np.ndarray,
    n_segments: int,
) -> int:
    """Choose a compact diagnostic interval with state changes and diversity."""
    best_start = 0
    best_score = -np.inf
    modeled = states * incremental[None, :] + baselines[None, :]
    residual = np.abs(blocks - modeled).sum(axis=1)
    for start in range(blocks.shape[0] - n_segments + 1):
        stop = start + n_segments
        changes = np.abs(np.diff(states[start:stop], axis=0)).sum()
        diversity = np.unique(states[start:stop], axis=0).shape[0]
        score = 1000.0 * changes + 100.0 * diversity - residual[start:stop].mean()
        if score > best_score:
            best_score = score
            best_start = start
    return best_start


def event_compress(
    blocks: np.ndarray,
    states: np.ndarray,
    incremental: np.ndarray,
    baselines: np.ndarray,
    threshold_ratio: float,
    minimum_run_blocks: int = 2,
    *,
    aggregate_mode: str = "signed",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Compress regular blocks into aggregate-detected, piecewise-stable segments."""
    block_aggregate = baseline_centered_aggregate(blocks, baselines, mode=aggregate_mode)
    event_threshold = threshold_ratio * float(np.max(incremental))
    differences = np.abs(np.diff(block_aggregate))
    candidates = (np.flatnonzero(differences >= event_threshold) + 1).tolist()

    events: list[int] = []
    group: list[int] = []
    for candidate in candidates:
        if group and candidate - group[-1] >= minimum_run_blocks:
            events.append(max(group, key=lambda index: differences[index - 1]))
            group = []
        group.append(candidate)
    if group:
        events.append(max(group, key=lambda index: differences[index - 1]))

    boundaries = np.asarray([0, *events, blocks.shape[0]], dtype=int)
    compressed = []
    compressed_aggregate = []
    compressed_states = []
    weights = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        compressed.append(blocks[start:stop].mean(axis=0))
        compressed_aggregate.append(float(block_aggregate[start:stop].mean()))
        compressed_states.append(
            (states[start:stop].mean(axis=0) >= 0.5).astype(np.int8)
        )
        weights.append(stop - start)
    return (
        np.asarray(compressed),
        np.asarray(compressed_aggregate),
        np.asarray(compressed_states),
        np.asarray(weights, dtype=float),
        boundaries,
        event_threshold,
    )
