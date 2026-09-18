#!/usr/bin/env python3
"""Offline independent byte/count and categorical-objective ladder audit.

No quantum account or job APIs are used. This does not call the production
count decoder, scorer, or categorical objective builder.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from qiskit_ibm_runtime import RuntimeDecoder


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_integer(value, widths):
    states, offset = [], 0
    for width in widths:
        register = (value >> offset) & ((1 << width) - 1)
        if register.bit_count() != 1:
            return None
        states.append(register.bit_length() - 1)
        offset += width
    return states


def direct_energy(categories, example, width, levels, penalties):
    path = np.asarray(categories, dtype=int).reshape(width, len(levels))
    value = 0.0
    for t in range(width):
        power = sum(float(levels[i][path[t, i]]) for i in range(len(levels)))
        value += float(example["weights"][t]) * (float(example["aggregate"][t]) - power) ** 2
    for t in range(1, width):
        value += sum(float(penalties[i]) for i in range(len(levels)) if path[t, i] != path[t - 1, i])
    return value


def audit(folder, output):
    require(not output.exists(), "Use a new audit receipt; never overwrite")
    plan, result = read(folder / "plan.json"), read(folder / "result.json")
    pubs, examples, metrics = read(folder / "pubs.json"), read(folder / "examples.json"), read(folder / "metrics.json")
    receipt, intent = read(folder / "job.json"), read(folder / "intent.json")
    require(result["job_id"] == receipt["job_id"], "Result job differs")
    require(result["plan_sha256"] == receipt["plan_sha256"] == intent["plan_sha256"] == sha(folder / "plan.json"), "Plan binding differs")
    require(receipt["intent_sha256"] == sha(folder / "intent.json") and intent["pubs_sha256"] == sha(folder / "pubs.json"), "Submission binding differs")
    require(result["runtime_sha256"] == sha(folder / "runtime.json.gz") and result["metrics_sha256"] == sha(folder / "metrics.json"), "Returned-data binding differs")
    require(metrics["usage"]["status"] == "complete", "IBM accounting is not finalized")
    require(metrics["usage"]["qpu_charge_time_seconds"] == result["accounted_s"] <= plan["campaign_cap_s"], "Accounted charge differs or exceeds cap")
    paths = {Path(__file__).resolve(), *(folder / name for name in ("plan.json", "pubs.json", "examples.json", "job.json", "intent.json", "runtime.json.gz", "metrics.json", "result.json"))}
    for group in ("source_sha256", "code_sha256", "files_sha256"):
        base = folder if group == "files_sha256" else ROOT
        for name, expected in plan[group].items():
            path = (base / name).resolve()
            require(path.is_relative_to(base.resolve()) and sha(path) == expected, f"Frozen source changed: {name}")
            paths.add(path)
    if (folder / "retry_origin.json").exists():
        origin = read(folder / "retry_origin.json")
        previous = Path(origin["source_directory"])
        require(sha(previous / "plan.json") == origin["source_plan_sha256"] == sha(folder / "plan.json"), "Retry plan changed")
        require(sha(previous / "failure_status.json") == origin["source_failure_sha256"], "Original failure archive changed")
        paths.update((folder / "retry_origin.json", previous / "plan.json", previous / "failure_status.json"))
    before = {str(path.resolve()): sha(path) for path in paths}
    with gzip.open(folder / "runtime.json.gz", "rt") as handle:
        runtime = json.load(handle, cls=RuntimeDecoder)
    frozen = read(ROOT / "results/quantum_heldout/model.json")
    model = frozen["models"]["multistate"]
    levels = model["levels_w"]
    counts = tuple(map(len, levels))
    penalties = float(frozen["selected_parameters"]["multistate_compressed"]["rho"]) * np.asarray(model["ranges_w"]) ** 2
    require(counts == (4, 3, 2, 3), "Frozen register counts changed")
    require(len(runtime) == len(pubs) == len(result["rows"]) == 24, "Incorrect PUB count")
    require(result["no_fallback"] and not result["postselected_raw_counts"], "Fallback or prefiltered counts")
    rows, grouped, maximum_energy_error = [], {}, 0.0
    minima = {}
    for index, (returned, pub, saved) in enumerate(zip(runtime, pubs, result["rows"])):
        require(all(saved[key] == value for key, value in pub.items()), "PUB ordering/specification changed")
        width, example_index = pub["width"], pub["example_index"]
        widths = counts * width
        nbits = sum(widths)
        measured = returned.data.meas
        require(measured.num_bits == nbits and measured.num_shots == 1024 and measured.array.shape == (1024, (nbits + 7) // 8), "Wrong raw byte array shape")
        byte_values = [int.from_bytes(bytes(value), "big") for value in measured.array]
        require(all(value < (1 << nbits) for value in byte_values), "Nonzero padding bits")
        raw_counts = dict(Counter(format(value, f"0{nbits}b") for value in byte_values))
        require(raw_counts == saved["raw_counts"], "Saved histogram differs from raw packed bytes")
        score = saved["sample_score"]
        require(not score["classical_fallback_used"] and not score["raw_counts_are_postselected"], "Scorer applied fallback or filtering")
        example = examples[example_index]
        require(example["window_id"].startswith("train-"), "Diagnostic is not training-only")
        pair = (example_index, width)
        if pair not in minima:
            minima[pair] = min(direct_energy(categories, example, width, levels, penalties)
                               for categories in itertools.product(*(range(count) for count in widths)))
        optimum = minima[pair]
        require(np.isclose(optimum, score["exact_objective"], rtol=1e-12, atol=1e-8), "Independent exact objective differs")
        feasible, invalid = {}, {}
        for key, count in raw_counts.items():
            categories = decode_integer(int(key, 2), widths)
            if categories is None:
                invalid[key] = count
            else:
                feasible[key] = (count, categories, direct_energy(categories, example, width, levels, penalties))
        valid_shots = sum(value[0] for value in feasible.values())
        require(valid_shots == score["feasible_shots"] and 1024 - valid_shots == score["infeasible_shots"] and score["total_shots"] == 1024, "Independent one-hot classification differs")
        require(invalid == score["invalid_counts"] and valid_shots / 1024 == score["feasible_fraction"], "Invalid pattern archive differs")
        saved_feasible = {row["bitstring"]: row for row in score["feasible_records"]}
        require(set(saved_feasible) == set(feasible), "Feasible record keys differ")
        for key, (count, categories, energy) in feasible.items():
            row = saved_feasible[key]
            require(row["count"] == count and np.array_equal(row["states"], np.asarray(categories).reshape(width, len(counts))), "Decoded category or frequency differs")
            error = abs(energy - row["objective"])
            maximum_energy_error = max(maximum_energy_error, error)
            require(np.isclose(energy, row["objective"], rtol=1e-12, atol=1e-8), "Decoded energy differs")
        optimum_shots = sum(count for count, _, energy in feasible.values() if abs(energy - optimum) <= 1e-8)
        require(optimum_shots == score["exact_optimum_shots"], "Exact optimum shot count differs")
        if feasible:
            best = min(value[2] for value in feasible.values())
            require(np.isclose(best, score["best_feasible_objective"], rtol=1e-12, atol=1e-8), "Selected objective differs")
            require(score["best_feasible_bitstring"] in feasible and np.isclose(feasible[score["best_feasible_bitstring"]][2], best, rtol=1e-12, atol=1e-8), "Selected bitstring is not minimum-cost feasible")
        else:
            require(score["best_states"] is None and score["best_feasible_objective"] is None and score["prediction_status"] == "no_feasible_samples", "Missing feasible samples were filled")
        row = {"pub_index": index, "template": pub["template"], "example_index": example_index, "training_window_id": example["window_id"], "logical_qubits": nbits, "shots": 1024, "feasible_shots": valid_shots, "feasible_fraction": valid_shots / 1024, "exact_optimum_shots": optimum_shots}
        rows.append(row)
        group = grouped.setdefault(pub["template"], {"circuits": 0, "shots": 0, "feasible_shots": 0})
        group["circuits"] += 1
        group["shots"] += 1024
        group["feasible_shots"] += valid_shots
    for group in grouped.values():
        group["feasible_fraction"] = group["feasible_shots"] / group["shots"]
    require(grouped == result["groups"] and sum(row["shots"] for row in rows) == result["total_shots"] == 24576, "Pooled counts differ")
    require(before == {name: sha(Path(name)) for name in before}, "Inputs changed during audit")
    payload = {"status": "passed", "audit": "independent packed-byte histogram, integer-register feasibility and directly enumerated objective audit; offline only", "created_utc": datetime.now(timezone.utc).isoformat(), "job_id": result["job_id"], "accounting_status": "complete", "accounted_s": result["accounted_s"], "pubs": 24, "raw_shots": 24576, "no_feasible_sample_pubs": sum(row["feasible_shots"] == 0 for row in rows), "rows": rows, "groups": grouped, "maximum_observed_energy_discrepancy": maximum_energy_error, "source_sha256": before, "source_hashes_unchanged_during_audit": True, "scope": "three training examples per width, one barrier-separated diagnostic job; not a full NILM evaluation or independent date/layout replication"}
    with output.open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: payload[key] for key in ("status", "job_id", "accounted_s", "raw_shots", "no_feasible_sample_pubs", "maximum_observed_energy_discrepancy", "groups")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.folder.resolve(), args.output.resolve())
