"""Gap-aware prediction and fixed-definition metrics for held-out NILM."""

from __future__ import annotations

from time import perf_counter
import numpy as np

from quantum_nilm.multistate import solve_multistate_temporal_exact


def contiguous_slices(timestamps: np.ndarray, step: int = 30):
    timestamps = np.asarray(timestamps)
    if (timestamps.ndim != 1 or not np.all(np.isfinite(timestamps))
            or np.any(np.diff(timestamps) <= 0)):
        raise ValueError("timestamps must be a finite strictly increasing vector")
    if not len(timestamps):
        return []
    edges = np.r_[0, np.flatnonzero(np.diff(timestamps) != step) + 1, len(timestamps)]
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def aggregate_boundaries(aggregate: np.ndarray, threshold: float | None):
    """Detect events using aggregate only; no reference states are accepted."""
    if threshold is None:
        return np.arange(len(aggregate) + 1)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("event threshold must be positive")
    differences = np.abs(np.diff(aggregate))
    candidates = np.flatnonzero(differences >= threshold) + 1
    events, group = [], []
    for index in candidates:
        if group and index - group[-1] >= 2:
            events.append(max(group, key=lambda k: differences[k - 1]))
            group = []
        group.append(int(index))
    if group:
        events.append(max(group, key=lambda k: differences[k - 1]))
    return np.asarray([0, *events, len(aggregate)])


def predict_temporal(aggregate, timestamps, levels, penalties, event_threshold=None):
    """Fit the raw-offset-equivalent objective, never joining missing blocks."""
    started = perf_counter()
    measured = np.asarray(aggregate, dtype=float)
    timestamps = np.asarray(timestamps)
    if measured.shape != timestamps.shape or not np.all(np.isfinite(measured)):
        raise ValueError("one finite aggregate per timestamp is required")
    prediction = np.empty((len(measured), len(levels)))
    energy, omitted_constant, segments, max_states = 0.0, 0.0, 0, 0
    for part in contiguous_slices(timestamps):
        signal = measured[part]
        bounds = aggregate_boundaries(signal, event_threshold)
        weights = np.diff(bounds)
        means = np.add.reduceat(signal, bounds[:-1]) / weights
        omitted_constant += float(np.sum((signal - np.repeat(means, weights)) ** 2))
        result = solve_multistate_temporal_exact(means, levels, penalties, weights)
        prediction[part] = np.repeat(result.predicted_power, weights, axis=0)
        energy += result.energy
        segments += len(weights)
        max_states = max(max_states, result.states_per_segment)
    return prediction, {
        "solver_wall_time_s": perf_counter() - started,
        "objective_energy": energy,
        "objective_energy_basis": "duration-weighted segment-mean reconstruction plus categorical transition costs",
        "within_segment_residual_constant": omitted_constant,
        "full_block_objective_energy": energy + omitted_constant,
        "segments": segments,
        "joint_states_per_segment": max_states,
        "blocks": len(measured),
    }


def _transition_counts(truth, prediction, timestamps, tolerance_blocks=1):
    """One-to-one, sign-matched transition events within each valid run."""
    tp = fp = fn = 0
    for part in contiguous_slices(timestamps):
        t = np.diff(truth[part].astype(int))
        p = np.diff(prediction[part].astype(int))
        for sign in (-1, 1):
            actual = np.flatnonzero(t == sign)
            inferred = np.flatnonzero(p == sign)
            left = right = 0
            while left < len(actual) and right < len(inferred):
                delta = inferred[right] - actual[left]
                if abs(delta) <= tolerance_blocks:
                    tp += 1
                    left += 1
                    right += 1
                elif delta < -tolerance_blocks:
                    fp += 1
                    right += 1
                else:
                    fn += 1
                    left += 1
            fn += len(actual) - left
            fp += len(inferred) - right
    return {"tp": tp, "fp": fp, "fn": fn}


def classification_metrics(tp, tn, fp, fn):
    n = tp + tn + fp + fn
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "bit_accuracy": (tp + tn) / n if n else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "mcc": (tp * tn - fp * fn) / np.sqrt(float(denom)) if denom else None,
    }


def score_power(truth, prediction, timestamps, thresholds, channel_names):
    """Score continuous measured appliance power and common train-fixed proxies."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim != 2 or not len(truth):
        raise ValueError("nonempty matching time-by-channel arrays required")
    if len(timestamps) != len(truth) or len(thresholds) != truth.shape[1]:
        raise ValueError("timestamps and thresholds must match data")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError("power values must be finite")
    result = {}
    for i, name in enumerate(channel_names):
        error = prediction[:, i] - truth[:, i]
        actual = truth[:, i] > thresholds[i]
        inferred = prediction[:, i] > thresholds[i]
        tp, tn = int(np.sum(actual & inferred)), int(np.sum(~actual & ~inferred))
        fp, fn = int(np.sum(~actual & inferred)), int(np.sum(actual & ~inferred))
        event = _transition_counts(actual, inferred, timestamps)
        event_denom = 2 * event["tp"] + event["fp"] + event["fn"]
        energy = float(truth[:, i].sum())
        squared = float(np.square(truth[:, i]).sum())
        result[name] = {
            "n_blocks": len(truth),
            "mae_w": float(np.abs(error).mean()),
            "absolute_error_sum_w": float(np.abs(error).sum()),
            "squared_error_sum_w2": float(np.square(error).sum()),
            "true_power_sum_w": energy,
            "true_squared_power_sum_w2": squared,
            "signed_power_error_sum_w": float(error.sum()),
            "energy_error_wh": float(error.sum()) / 120.0,
            "normalized_disaggregation_error": float(np.square(error).sum()) / squared if squared else None,
            "signal_aggregate_error": abs(float(error.sum())) / abs(energy) if energy else None,
            "active_blocks": int(actual.sum()),
            "state": classification_metrics(tp, tn, fp, fn),
            "event_counts": event,
            "event_f1": 2 * event["tp"] / event_denom if event_denom else None,
        }
    return result


def pool_window_metrics(records, channels):
    """Pool underlying counts/errors, not averages of ratios."""
    result = {}
    for channel in channels:
        rows = [record["appliances"][channel] for record in records]
        n = sum(r["n_blocks"] for r in rows)
        absolute = sum(r["absolute_error_sum_w"] for r in rows)
        signed = sum(r["signed_power_error_sum_w"] for r in rows)
        squared = sum(r["squared_error_sum_w2"] for r in rows)
        true_power = sum(r["true_power_sum_w"] for r in rows)
        true_squared = sum(r["true_squared_power_sum_w2"] for r in rows)
        counts = {k: sum(r["state"][k] for r in rows) for k in ("tp", "tn", "fp", "fn")}
        event = {k: sum(r["event_counts"][k] for r in rows) for k in ("tp", "fp", "fn")}
        event_denom = 2 * event["tp"] + event["fp"] + event["fn"]
        result[channel] = {
            "n_blocks": n, "mae_w": absolute / n if n else None,
            "rmse_w": np.sqrt(squared / n) if n else None,
            "energy_error_wh": signed / 120.0,
            "signal_aggregate_error": abs(signed) / abs(true_power) if true_power else None,
            "normalized_disaggregation_error": squared / true_squared if true_squared else None,
            "active_blocks": sum(r["active_blocks"] for r in rows),
            "active_windows": sum(r["active_blocks"] > 0 for r in rows),
            "state": classification_metrics(**counts),
            "event_counts": event,
            "event_f1": 2 * event["tp"] / event_denom if event_denom else None,
        }
    return result
