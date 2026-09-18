#!/usr/bin/env python3
"""Independent read-only audit of saved categorical IBM held-out results.

No account/network/submission code is present. The main algorithm's builder,
decoder, objective evaluator, and scoring helpers are deliberately not used.
Optional output is a new JSON file; source experiment files are never changed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import json
from numbers import Integral
from pathlib import Path
import re

import numpy as np


EXPECTED_STUDY = {"windows": 30, "blocks": 86396, "circuits": 2451,
                  "stages": 129, "shots": 627456, "intervals": 4881}


class AuditFailure(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditFailure(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def numeric_equal(left, right):
    return bool(np.allclose(left, right, rtol=1e-12, atol=1e-6))


def accounted_usage(metrics):
    """Accept both IBM accounted-usage schemas, never circuit duration."""
    usage = metrics.get("usage", {})
    values = {name: usage[name] for name in ("quantum_seconds", "qpu_charge_time_seconds")
              if usage.get(name) is not None}
    require(bool(values), "Accounted QPU usage is missing")
    require(all(not isinstance(value, bool) and isinstance(value, (int, float))
                and np.isfinite(value) and value >= 0 for value in values.values()),
            "Invalid accounted QPU usage")
    require(max(values.values()) - min(values.values()) < 1e-6,
            "IBM accounted-usage aliases disagree")
    return float(next(iter(values.values())))


def decode_raw_bitstring(bitstring, sizes, intervals):
    """Independent explicit endian/one-hot decoding; None denotes infeasible."""
    width = sum(sizes) * intervals
    require(isinstance(bitstring, str) and len(bitstring) == width
            and all(character in "01" for character in bitstring), "Malformed raw count bitstring")
    low_first = list(reversed(bitstring))
    result = np.empty((intervals, len(sizes)), dtype=int)
    offset = 0
    for segment in range(intervals):
        for channel, size in enumerate(sizes):
            occupied = [index for index in range(size) if low_first[offset + index] == "1"]
            if len(occupied) != 1:
                return None
            result[segment, channel] = occupied[0]
            offset += size
    return result


def categorical_id(states, sizes):
    value, stride = 0, 1
    for segment in states:
        for category, size in zip(segment, sizes):
            value += int(category) * stride
            stride *= size
    return value


def direct_objective(states, aggregate, weights, levels, penalties, previous):
    """Direct physical reconstruction error and categorical switching cost."""
    values = np.asarray(states)
    single = values.ndim == 2
    if single:
        values = values[None, :, :]
    prediction = np.zeros(values.shape[:2], dtype=float)
    for channel, power_levels in enumerate(levels):
        prediction += power_levels[values[:, :, channel]]
    result = np.sum(np.asarray(weights) * (np.asarray(aggregate) - prediction) ** 2, axis=1)
    result += np.sum(np.asarray(penalties) * (values[:, 1:] != values[:, :-1]), axis=(1, 2))
    if previous is not None:
        result += np.sum(np.asarray(penalties) * (values[:, 0] != np.asarray(previous)), axis=1)
    return float(result[0]) if single else result


@lru_cache(maxsize=8)
def all_categories(sizes, intervals):
    widths = sizes * intervals
    basis = np.arange(int(np.prod(widths)), dtype=np.int64)
    digits, stride = [], 1
    for width in widths:
        digits.append((basis // stride) % width)
        stride *= width
    return np.column_stack(digits).reshape(len(basis), intervals, len(sizes))


def audit_prediction(row, chunk, levels, penalties, previous, shots):
    sizes = tuple(len(values) for values in levels)
    intervals = len(chunk["weights"])
    counts = row["raw_counts"]
    require(isinstance(counts, dict) and counts, "Raw counts missing")
    require(all(isinstance(count, Integral) and not isinstance(count, bool) and count > 0
                for count in counts.values()), "Raw shot count is not a positive integer")
    require(sum(counts.values()) == shots, "Raw count total differs from declared shot budget")
    candidates = []
    invalid_shots = 0
    for bitstring, count in counts.items():
        decoded = decode_raw_bitstring(bitstring, sizes, intervals)
        if decoded is None:
            invalid_shots += count
        else:
            energy = direct_objective(decoded, chunk["aggregate"], chunk["weights"], levels, penalties, previous)
            candidates.append({"states": decoded, "count": count, "energy": energy,
                               "id": categorical_id(decoded, sizes), "bitstring": bitstring})
    feasible = shots - invalid_shots
    saved = row["sample_score"]
    require(saved["total_shots"] == shots and saved["feasible_shots"] == feasible
            and saved["infeasible_shots"] == invalid_shots, "Saved feasibility totals disagree with raw bits")
    require(numeric_equal(saved["feasible_fraction"], feasible / shots), "Saved feasible fraction differs")
    require(saved.get("classical_fallback_used") is False and saved.get("raw_counts_are_postselected") is False,
            "Unexpected hidden fallback or raw-count postselection flag")
    if candidates:
        selected = min(candidates, key=lambda item: (item["energy"], item["id"]))
        require(row["fallback_used"] is False, "Fallback declared despite valid observations")
        require(np.array_equal(row["states"], selected["states"]), "Recovered state is not the best observed feasible state")
        require(np.array_equal(saved["best_states"], selected["states"]), "Saved decoder state differs")
        require(saved["best_feasible_bitstring"] == selected["bitstring"], "Saved selected bitstring differs")
        require(numeric_equal(saved["best_feasible_objective"], selected["energy"]), "Saved selected objective differs")
        require(saved["prediction_status"] == "feasible_sample_selected", "Unexpected feasible prediction status")
        mean = sum(item["count"] * item["energy"] for item in candidates) / feasible
        require(numeric_equal(saved["mean_objective_conditional_on_feasibility"], mean), "Saved conditional mean objective differs")
    else:
        initial = np.array([int(np.argmin(values)) for values in levels]) if previous is None else np.asarray(previous)
        expected = np.tile(initial, (intervals, 1))
        require(row["fallback_used"] is True and np.array_equal(row["states"], expected),
                "All-invalid fallback does not follow declared own-carry/lowest-power policy")
        require(saved["best_states"] is None and saved["best_feasible_objective"] is None
                and saved["prediction_status"] == "no_feasible_samples", "All-invalid raw result disguised as a quantum sample")
    energies = direct_objective(all_categories(sizes, intervals), chunk["aggregate"], chunk["weights"],
                                levels, penalties, previous)
    minimum = float(np.min(energies))
    require(numeric_equal(saved["exact_objective"], minimum), "Saved exact objective differs from independent enumeration")
    optimum_shots = sum(item["count"] for item in candidates
                        if np.isclose(item["energy"], minimum, rtol=0, atol=1e-8))
    require(saved["exact_optimum_shots"] == optimum_shots, "Saved optimum-hit count differs")
    require(numeric_equal(saved["exact_optimum_probability_all_shots"], optimum_shots / shots),
            "Saved unconditional optimum probability differs")
    return {"shots": shots, "feasible_shots": feasible, "invalid_shots": invalid_shots,
            "fallback": not bool(candidates), "states": np.asarray(row["states"], dtype=int)}


def independent_cost_bindings(chunk, levels, penalties, previous, gamma, parameter_order):
    """Independently expand physical objective into the normalized RZ/RZZ angles."""
    sizes = tuple(len(values) for values in levels)
    widths = sizes * len(chunk["weights"])
    offsets = np.cumsum((0,) + widths[:-1]).tolist()
    linear = np.zeros(sum(widths))
    quadratic = {}
    for t, (aggregate, weight) in enumerate(zip(chunk["aggregate"], chunk["weights"])):
        for i, powers in enumerate(levels):
            oi = offsets[t * len(sizes) + i]
            linear[oi:oi + len(powers)] += weight * (powers ** 2 - 2 * aggregate * powers)
            for j in range(i + 1, len(sizes)):
                oj = offsets[t * len(sizes) + j]
                for left, power_left in enumerate(powers):
                    for right, power_right in enumerate(levels[j]):
                        quadratic[(oi + left, oj + right)] = float(2 * weight * power_left * power_right)
    for t in range(1, len(chunk["weights"])):
        for channel, size in enumerate(sizes):
            for category in range(size):
                quadratic[(offsets[(t - 1) * len(sizes) + channel] + category,
                           offsets[t * len(sizes) + channel] + category)] = -float(penalties[channel])
    if previous is not None:
        for channel, category in enumerate(previous):
            linear[offsets[channel] + category] -= penalties[channel]
    scale = max(1., float(np.sum(np.abs(linear))) + sum(abs(value) for value in quadratic.values()))
    fields = -linear / 2
    for (left, right), value in quadratic.items():
        fields[left] -= value / 4
        fields[right] -= value / 4
    values = list(2 * gamma * fields / scale)
    values.extend(gamma * quadratic[edge] / (2 * scale) for edge in sorted(quadratic))
    output = []
    for name in parameter_order:
        match = re.fullmatch(r"cost_angle\[(\d+)\]", name)
        require(match is not None and int(match[1]) < len(values), "Unknown compiled cost parameter name")
        output.append(values[int(match[1])])
    return np.array(output)


def check_schedule(windows, block_seconds=30):
    totals = {"windows": len(windows), "blocks": 0, "circuits": 0, "intervals": 0,
              "stages": max(len(window["chunks"]) for window in windows)}
    require(len({window["window"]["id"] for window in windows}) == len(windows), "Repeated test window")
    for window in windows:
        stop, last_time, last_run = 0, None, None
        for index, chunk in enumerate(window["chunks"]):
            weights = chunk["weights"]
            require(len(weights) in (1, 2) and len(chunk["aggregate"]) == len(weights), "Invalid chunk interval count")
            require(all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in weights),
                    "Invalid interval duration")
            require(chunk["chunk"] == index and chunk["block_start"] == stop
                    and chunk["block_stop"] - chunk["block_start"] == sum(weights), "Chunk coverage gap/overlap")
            reset = index == 0 or chunk["run"] != last_run
            require(chunk["reset"] is reset, "Gap/day reset declaration differs from run boundary")
            if index:
                require(chunk["run"] == last_run + int(reset), "Unexpected contiguous-run numbering")
                require(chunk["start_unix"] > last_time if reset else chunk["start_unix"] == last_time,
                        "Timestamp continuity does not match gap reset")
            else:
                require(chunk["run"] == 0, "First run must be zero")
            last_time = chunk["start_unix"] + sum(weights) * block_seconds
            require(chunk["start_unix"] >= window["window"]["start_unix"]
                    and last_time <= window["window"]["end_unix"], "Chunk extends outside frozen day")
            stop, last_run = chunk["block_stop"], chunk["run"]
            totals["intervals"] += len(weights)
        require(stop == window["blocks"], "Chunks do not cover every valid block")
        totals["blocks"] += stop
        totals["circuits"] += len(window["chunks"])
    return totals


def verify_code_hashes(repo, folder, plan, plan_hash):
    """Check explicit execution-code revisions anchored to the original plan."""
    expected = dict(plan["code_sha256"])
    revisions = sorted(folder.glob("code_revision_*.json"))
    for number, path in enumerate(revisions, 1):
        revision = read(path)
        require(path.name == f"code_revision_{number:03d}.json" and revision.get("revision") == number,
                "Code revision numbering is not consecutive")
        require(revision.get("plan_sha256") == plan_hash, "Code revision chain has wrong plan anchor")
        require(revision.get("previous_code_sha256") == expected
                and set(revision["code_sha256"]) == set(expected), "Broken code revision hash/key chain")
        if revision.get("initial_code_copy"):
            name = revision["initial_code_copy"]
            require(Path(name).name == name and sha(folder / name) == revision["initial_code_copy_sha256"]
                    and revision["initial_code_copy_sha256"] == plan["code_sha256"]["scripts/run_quantum_heldout_ibm.py"],
                    "Preserved initial execution code does not match the original plan")
        expected = revision["code_sha256"]
    for name, checksum in expected.items():
        path = (repo / name).resolve()
        require(path.is_relative_to(repo.resolve()) and sha(path) == checksum, f"Frozen/revised code hash differs: {name}")
    return {"revision_chain_present": bool(revisions), "revisions": [path.name for path in revisions],
            "effective_code_sha256": expected}


def audit(root, *, expected_study=EXPECTED_STUDY, verify_runtime=True):
    root = Path(root).resolve()
    folder, repo = root / "hardware", Path(__file__).resolve().parents[1]
    plan, protocol, model, angles = (read(path) for path in
                                   (folder / "plan.json", root / "protocol.json", root / "model.json", root / "angles.json"))
    plan_hash = sha(folder / "plan.json")
    for name, checksum in read(root / "input_hashes.json").items():
        require(Path(name).name == name and sha(root / name) == checksum, f"Frozen input hash differs: {name}")
    require(sha(root / "angles.json") == plan["angles_sha256"] and sha(root / "protocol.json") == plan["protocol_sha256"],
            "Angle/protocol hash differs from hardware plan")
    for name, checksum in plan["files_sha256"].items():
        require(Path(name).name == name and sha(folder / name) == checksum, f"Prepared hardware hash differs: {name}")
    code = verify_code_hashes(repo, folder, plan, plan_hash)
    source_manifest = read(Path(protocol["original_dir"]) / "data_quality.json")
    for name, checksum in protocol["original_hashes"].items():
        require(sha(Path(protocol["original_dir"]) / name) == checksum, f"Original campaign source hash differs: {name}")
    windows = read(root / "test_inputs.json")
    for window in windows:
        original = next(item for item in source_manifest["test"] if item["window"]["id"] == window["window"]["id"])
        require(window["source_window_sha256"] == original["source_window_sha256"]
                and window["blocks"] == original["valid_blocks"], "Frozen source-window hash/coverage differs")
    schedule = check_schedule(windows)
    schedule["shots"] = schedule["circuits"] * plan["shots"]
    if expected_study is not None:
        require(schedule == expected_study, f"Campaign schedule differs from expected study: {schedule}")
    require(plan["days"] == schedule["windows"] and plan["circuits"] == schedule["circuits"]
            and plan["adaptive_stages"] == schedule["stages"], "Plan coverage differs from frozen inputs")
    require(plan["instance_plan"] == "open" and protocol["hardware_paid_execution"] is False,
            "Experiment is not declared Open Plan/no-paid")
    require(plan["shots"] == protocol["shots_per_chunk"]
            and plan["fallback_policy"] == protocol["hardware_zero_valid_policy"], "Protocol shot/fallback policy differs")
    require(read(folder / "start.json")["plan_sha256"] == plan_hash, "Start record plan anchor differs")
    trained = model["models"]["multistate"]
    levels = [np.array(values) for values in trained["levels_w"]]
    penalties = model["selected_parameters"]["multistate_compressed"]["rho"] * np.array(trained["ranges_w"]) ** 2
    prior = {window["window"]["id"]: None for window in windows}
    totals = {"stages": 0, "circuits": 0, "shots": 0, "feasible_shots": 0, "invalid_shots": 0,
              "fallback_chunks": 0, "fallback_blocks": 0, "covered_blocks": 0, "accounted_qpu_seconds": 0.}
    jobs, reports = set(), []
    for stage in range(schedule["stages"]):
        result_path = folder / f"stage_{stage:03d}_result.json"
        if not result_path.exists():
            later = [path.name for path in folder.glob("stage_*_result.json")
                     if int(path.name.split("_")[1]) > stage]
            require(not later, "Completed stages have a missing earlier result")
            break
        intent = read(folder / f"stage_{stage:03d}_job.json")
        saved = read(result_path)
        current = [window for window in windows if stage < len(window["chunks"])]
        expected_days = [window["window"]["id"] for window in current]
        for record in (intent, saved):
            require(record["plan_sha256"] == plan_hash and record["stage"] == stage, "Stage plan/index anchor differs")
        require(intent["days"] == expected_days and [row["window_id"] for row in saved["rows"]] == expected_days,
                "Stage window/PUB ordering differs")
        require(intent["shots_per_circuit"] == plan["shots"], "Intent shot count differs")
        require(intent["job_id"] == saved["job_id"] and intent["job_id"] not in jobs, "Job ID reused or mismatched")
        jobs.add(intent["job_id"])
        raw = None
        if verify_runtime:
            from qiskit_ibm_runtime import RuntimeDecoder
            with gzip.open(folder / f"stage_{stage:03d}_runtime.json.gz", "rt") as handle:
                raw = json.load(handle, cls=RuntimeDecoder)
            require(len(raw) == len(current), "Serialized Runtime PUB count differs")
        stage_feasible = 0
        for position, (window, row) in enumerate(zip(current, saved["rows"])):
            wid, chunk = window["window"]["id"], window["chunks"][stage]
            previous = None if chunk["reset"] else prior[wid]
            expected_previous = None if previous is None else previous.tolist()
            require(row["chunk"] == chunk["chunk"] and row["previous_states"] == expected_previous
                    and intent["previous_states"][wid] == expected_previous, "Prediction/intent violates own-state carry or gap reset")
            order = plan["templates"][str(len(chunk["weights"]))]["parameter_order"]
            parameters = independent_cost_bindings(chunk, levels, penalties, previous, angles["gammas"][0], order)
            require(np.allclose(parameters, intent["parameter_values"][position], rtol=1e-11, atol=1e-12),
                    "Submitted parameters differ from independently expanded objective")
            if raw is not None:
                require(raw[position].data.meas.get_counts() == row["raw_counts"], "Runtime measurements differ from saved raw counts")
            try:
                checked = audit_prediction(row, chunk, levels, penalties, previous, plan["shots"])
            except AuditFailure as error:
                raise AuditFailure(f"Stage {stage}, {wid}: {error}") from error
            prior[wid] = checked["states"][-1]
            totals["circuits"] += 1
            for field in ("shots", "feasible_shots", "invalid_shots"):
                totals[field] += checked[field]
            blocks = chunk["block_stop"] - chunk["block_start"]
            totals["covered_blocks"] += blocks
            totals["fallback_chunks"] += int(checked["fallback"])
            totals["fallback_blocks"] += blocks * int(checked["fallback"])
            stage_feasible += checked["feasible_shots"]
        finalized = folder / f"stage_{stage:03d}_finalized_metrics.json"
        metrics = read(finalized)["job_metrics"] if finalized.exists() else saved["job_metrics"]
        usage = accounted_usage(metrics)
        require(usage <= intent["max_execution_time_s"] + 1e-6, "Reported job usage exceeds authorized job cap")
        totals["accounted_qpu_seconds"] += usage
        totals["stages"] += 1
        reports.append({"stage": stage, "job_id": intent["job_id"], "circuits": len(current),
                        "feasible_shots": stage_feasible, "accounted_qpu_seconds": usage,
                        "result_sha256": sha(result_path)})
    require(totals["accounted_qpu_seconds"] <= plan["maximum_campaign_usage_s"] + 1e-6, "Campaign QPU cap exceeded")
    complete = (folder / "completed.json").exists()
    if complete:
        completion = read(folder / "completed.json")
        require(completion["plan_sha256"] == plan_hash and completion["stages"] == schedule["stages"]
                and completion["circuits"] == schedule["circuits"], "Completion record differs from frozen schedule")
        require(totals["stages"] == schedule["stages"] and totals["circuits"] == schedule["circuits"]
                and totals["shots"] == schedule["shots"] and totals["covered_blocks"] == schedule["blocks"],
                "Completion declared without full hardware coverage")
        require(numeric_equal(completion["accounted_quantum_seconds"], totals["accounted_qpu_seconds"]), "Completion usage total differs")
    summary_path = folder / "summary.json"
    if summary_path.exists():
        require(complete, "A full aggregate hardware summary exists without completion")
        summary = read(summary_path)
        for field, expected in (("windows", schedule["windows"]), ("blocks", schedule["blocks"]),
                                ("circuits", totals["circuits"]), ("raw_shots", totals["shots"]),
                                ("feasible_shots", totals["feasible_shots"]), ("fallback_chunks", totals["fallback_chunks"]),
                                ("fallback_blocks", totals["fallback_blocks"])):
            require(summary[field] == expected, f"Scored hardware summary differs: {field}")
    return {"status": "passed_complete" if complete else "incomplete_observed_prefix_passed",
            "created_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": plan_hash,
            "expected_study": schedule, "observed": totals, "complete": complete,
            "runtime_check_enabled": verify_runtime,
            "runtime_stages_checked": totals["stages"] if verify_runtime else 0,
            "runtime_raw_counts_checked": verify_runtime and totals["stages"] > 0,
            "code_integrity": code, "stages": reports,
            "independence": "No main algorithm builder/decoder/objective/scoring imports; raw bitstrings, direct costs and parameter expansion independently recomputed",
            "limitations": ["No new account/network/QPU calls", "No re-read of the large meter source CSV; frozen per-window source hashes match the original source manifest",
                            "This audits inference/coverage/accounting, not independently recomputed appliance MAE from source labels",
                            "An incomplete prefix is not a completed held-out result or an advantage claim"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/quantum_heldout"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--skip-runtime", action="store_true", help="Metadata-only raw-count audit; full Runtime comparison is default")
    args = parser.parse_args()
    try:
        result = audit(args.input_dir, verify_runtime=not args.skip_runtime)
        exit_code = 0 if result["complete"] or not args.require_complete else 2
    except (AuditFailure, FileNotFoundError, KeyError, ValueError) as error:
        result = {"status": "failed", "error_type": type(error).__name__, "error": str(error), "complete": False}
        exit_code = 1
    if args.output:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
