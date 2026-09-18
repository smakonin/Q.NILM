#!/usr/bin/env python3
"""Frozen offline Steps 1 and 2: finite-shot robustness and new R1Hz windows."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from quantum_nilm import training_validation as tv
from quantum_nilm.heldout_data import read_block_window, source_bounds

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/"results/heldout_campaign"
CHANNELS = ("main", "dryr", "frdg", "vacu")
SOURCES = ["scripts/run_training_validation.py", "src/quantum_nilm/training_validation.py",
           "tests/test_training_validation.py", "docs/training_validation_protocol.md",
           "src/quantum_nilm/categorical_qaoa.py", "src/quantum_nilm/heldout_data.py"]


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write("\n")


def npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f: np.savez_compressed(f, **arrays)


def recorded_ranges(value, first, end):
    if isinstance(value, dict):
        a, b = value.get("start_unix", value.get("start")), value.get("end_unix", value.get("end"))
        if isinstance(a, int) and isinstance(b, int) and first <= a < b <= end and b-a <= 86400:
            yield a, b
        for v in value.values(): yield from recorded_ranges(v, first, end)
    elif isinstance(value, list):
        for v in value: yield from recorded_ranges(v, first, end)


def freeze(folder):
    tv.require(not folder.exists(), "Choose a fresh archive")
    old = read(OLD/"protocol.json"); model = read(OLD/"frozen_models.json")
    tv.require(model["protocol_sha256"] == sha(OLD/"protocol.json"), "Frozen model lineage")
    source = Path(old["source"]); first, end = source_bounds(source)
    tv.require((first, end) == tuple(old["manifest"]["source_bounds"].values()), "Source endpoints changed")
    tv.require(source.stat().st_size == old["source_size_bytes"] and source.stat().st_mtime_ns == old["source_mtime_ns"], "Source changed")
    # Read design metadata, never new validation/test measurements here.
    prior = sorted(set((ROOT/"results").rglob("protocol.json")) | set((ROOT/"results").rglob("plan.json")))
    exposed = set(); evidence = {}
    for path in prior:
        if path.stat().st_size > 8_000_000: raise ValueError("Unexpected large design file; review manually")
        content = read(path)
        ranges = list(recorded_ranges(content, first, end))
        if ranges:
            exposed.update(ranges); evidence[str(path.relative_to(ROOT))] = sha(path)
    # The original pilot and every later pilot adapter use this extracted hour.
    pilot = ROOT/"data/derived/r1hz_2018-01-28_1200.csv"
    exposed.add(source_bounds(pilot)); evidence[str(pilot.relative_to(ROOT))] = sha(pilot)
    windows = {}
    for split, count in (("validation", 10), ("test", 20)):
        part = old["manifest"]["splits"][split]
        selected = tv.unused_windows(part["start"], part["end"], exposed, count, origin=first)
        windows[split] = [{"id": f"new-{split}-{i:03}", "split": split, "start": a, "end": b}
                          for i, (a, b) in enumerate(selected)]
    original_train = old["manifest"]["splits"]["train"]["windows"]
    train = []
    for j in range(32):
        w = original_train[j*(len(original_train)-1)//31]
        a = w["start"]+43200
        train.append({"id": f"angle-train-{j:03}", "split": "train", "start": a, "end": a+30, "original_window": w["id"]})
    evidence.update({str((OLD/name).relative_to(ROOT)): sha(OLD/name) for name in ("protocol.json", "frozen_models.json")})
    plan = {"created_utc": now(), "scope": "Offline synthetic and static single-interval R1Hz diagnostics; no hardware or production/paper changes",
            "source": str(source), "source_size_bytes": source.stat().st_size, "source_mtime_ns": source.stat().st_mtime_ns,
            "source_bounds": [first, end], "source_global_ordering": "assumed sorted; checked within extracted windows, not globally certified",
            "source_sha256": {x: sha(ROOT/x) for x in SOURCES}, "input_sha256": evidence,
            "versions": {x: version(x) for x in ("numpy", "scipy", "qiskit")},
            "exposed_ranges": [list(x) for x in sorted(exposed)], "fresh_window_guard_seconds": 86400,
            "freshness_scope": "Disjoint from recorded Q.NILM design windows, not a guarantee of no earlier human inspection of the dataset",
            "windows": {"train": train, **windows}, "missing_policy": "Discard invalid complete 30-second blocks; never replace windows or training timestamps",
            "minimum_training_cases": 16, "channels": list(CHANNELS), "excluded_markers": ["s", "+"],
            "model": model["models"]["multistate"], "model_training_scope": "Reuse centroids fitted on original first-partition training days only",
            "losses": ["mean", "cvar50"], "synthetic_seed_pairs": [[40101+100*i, 40102+100*i] for i in range(20)],
            "real_seed_pairs": [[60101+100*i, 60102+100*i] for i in range(10)],
            "synthetic_gamma_bounds": {"original": float(8*np.pi), "wider": float(16*np.pi)}, "real_gamma_max": float(16*np.pi),
            "search": {"evaluations_per_restart": 512, "restarts": 2, "shots_per_case_evaluation": 256,
                       "shortlist": 5, "independent_recheck_shots_per_case": 4096, "policy": "Same initial pool, fixed bounded Powell plus random padding; ties first"},
            "real_training_loss": "Equal mean of per-case normalized mean or per-case lower CVaR50, not pooled global CVaR",
            "real_validation": "Select one rechecked fit per loss by pooled measured appliance MAE at 16 evaluation shots; ties first trial. Also freeze overall validation winner, ties mean.",
            "real_primary_metric": "Pooled expected appliance MAE at 16 raw evaluation shots; each accepted 30-second block and each of three appliances equally weighted",
            "secondary_shots": 256, "controls": ["exact 72-state mains optimization", "uniform feasible 16/256", "lowest-power constant", "label-only nearest-centroid oracle", "exact 24-state selected-circuit aggregate control"],
            "real_representation": "12 qubits, 72 feasible assignments, raw mains plus learned background; independent intervals, no temporal penalty/compression or state carry",
            "inference": "No labels in angle training or test inference; measurements of appliance channels used for validation scoring/test scoring only, except explicitly label-dependent oracle and selected-circuit controls",
            "bootstrap": {"replicates": 2000, "seed": 889901, "unit": "24-hour window", "scope": "descriptive paired day-cluster intervals; not independent-home uncertainty or a significance claim"},
            "test_gate": "All training complete, all validation comparisons saved, selection hash frozen before test source read",
            "expected_synthetic_fits": 80, "expected_real_fits": 20, "hardware_jobs": 0}
    save(folder/"plan.json", plan)
    print(json.dumps({"plan_sha256": sha(folder/"plan.json"), "exposed_ranges": len(exposed), "validation_windows": 10, "test_windows": 20}), flush=True)


def verify(folder):
    plan = read(folder/"plan.json")
    for group in ("source_sha256", "input_sha256"):
        for path, value in plan[group].items(): tv.require(sha(ROOT/path) == value, f"Frozen input changed: {path}")
    for name, value in plan["versions"].items(): tv.require(version(name) == value, "Dependency version changed")
    source = Path(plan["source"])
    tv.require(source.stat().st_size == plan["source_size_bytes"] and source.stat().st_mtime_ns == plan["source_mtime_ns"], "Source changed")
    return plan


def fit_one(folder, identifier, batch, loss, seeds, bound, binding):
    path = folder/"fits"/(identifier+".json"); binary = path.with_suffix(".npz")
    if path.exists():
        record = read(path)
        tv.require(record["plan_sha256"] == binding and record["samples_sha256"] == sha(binary), "Fit checksum")
        return record
    fit, arrays = tv.train(batch, loss, seeds, bound)
    npz(binary, arrays)
    record = {"id": identifier, "plan_sha256": binding, "samples_file": binary.name,
              "samples_sha256": sha(binary), "fit": fit}
    save(path, record); print(f"Completed {identifier}", flush=True)
    return record


def synthetic(folder):
    plan = verify(folder); binding = sha(folder/"plan.json")
    tv.require(not (folder/"synthetic.json").exists(), "Synthetic phase complete")
    train_batch = tv.static_batch([680.], [[0,400,900], [0,120,280]])
    powers = train_batch["powers"]
    transfer_batch = tv.static_batch(powers.sum(axis=1), train_batch["levels"])
    results = []; fits = []
    for bound_name, bound in plan["synthetic_gamma_bounds"].items():
        for trial, seeds in enumerate(plan["synthetic_seed_pairs"]):
            for loss in plan["losses"]:
                identifier = f"synthetic_{bound_name}_{loss}_{trial:02}"
                record = fit_one(folder, identifier, train_batch, loss, seeds, bound, binding)
                fits.append({"id": identifier, "file": "fits/"+identifier+".json", "sha256": sha(folder/"fits"/(identifier+".json"))})
                for phase, key in (("before_recheck", "precheck_angles"), ("after_recheck", "angles")):
                    angles = record["fit"][key]
                    metrics = tv.expected_metrics(transfer_batch, powers, angles=angles)
                    results.append({"id": identifier, "bound": bound_name, "loss": loss, "trial": trial, "phase": phase,
                                    "angles": angles, "metrics": {k: {x: v.tolist() for x, v in m.items()} for k, m in metrics.items()}})
    metrics = tv.expected_metrics(transfer_batch, powers, mode="uniform")
    save(folder/"synthetic.json", {"completed_utc": now(), "plan_sha256": binding, "fits": fits,
        "cases": powers.tolist(), "training_case_index": 7, "rows": results,
        "uniform": {k: {x: v.tolist() for x, v in m.items()} for k, m in metrics.items()}})


def extract(folder, plan, split):
    receipt_path = folder/(split+"_data.json")
    if receipt_path.exists():
        receipt = read(receipt_path)
        tv.require(receipt["plan_sha256"] == sha(folder/"plan.json"), "Data binding")
        for item in receipt["windows"]: tv.require(sha(folder/item["file"]) == item["sha256"], "Data checksum")
        return receipt
    if split == "test":
        selection = read(folder/"selection.json")
        tv.require(selection["plan_sha256"] == sha(folder/"plan.json"), "Selection plan mismatch")
        tv.require(selection["validation_sha256"] == sha(folder/"validation.json"), "Validation changed")
    items = []; started = perf_counter()
    for w in plan["windows"][split]:
        loaded = read_block_window(plan["source"], w["start"], w["end"], channels=CHANNELS, exclude_markers=("s", "+"))
        path = folder/"data"/(w["id"]+".npz")
        npz(path, {"timestamps": loaded["timestamps"], "values": loaded["values"]})
        items.append({"window": w, "file": str(path.relative_to(folder)), "sha256": sha(path),
                      "source_window_sha256": loaded["source_window_sha256"], "quality": loaded["quality"],
                      "blocks": len(loaded["timestamps"])})
    result = {"extracted_utc": now(), "split": split, "plan_sha256": sha(folder/"plan.json"), "windows": items,
              "total_blocks": sum(x["blocks"] for x in items), "extraction_seconds": perf_counter()-started}
    if split == "test": result["selection_sha256_before_read"] = sha(folder/"selection.json")
    save(receipt_path, result); print(f"Extracted {split}: {result['total_blocks']} valid blocks", flush=True)
    return result


def training_batch(folder, plan):
    receipt = extract(folder, plan, "train")
    items = [x for x in receipt["windows"] if x["blocks"]]
    values = np.concatenate([np.load(folder/x["file"], allow_pickle=False)["values"] for x in items])
    tv.require(len(values) >= plan["minimum_training_cases"], "Insufficient prespecified training cases")
    return tv.static_batch(values[:, 0], plan["model"]["levels_w"])


def train_real(folder):
    plan = verify(folder); binding = sha(folder/"plan.json")
    tv.require((folder/"synthetic.json").exists(), "Complete Step 1 first")
    tv.require(not (folder/"real_training.json").exists(), "Real training complete")
    batch = training_batch(folder, plan); fits = []
    for trial, seeds in enumerate(plan["real_seed_pairs"]):
        for loss in plan["losses"]:
            identifier = f"r1hz_{loss}_{trial:02}"
            record = fit_one(folder, identifier, batch, loss, seeds, plan["real_gamma_max"], binding)
            fits.append({"id": identifier, "loss": loss, "trial": trial, "angles": record["fit"]["angles"],
                         "file": "fits/"+identifier+".json", "sha256": sha(folder/"fits"/(identifier+".json"))})
    save(folder/"real_training.json", {"completed_utc": now(), "plan_sha256": binding,
         "train_data_sha256": sha(folder/"train_data.json"), "training_cases": len(batch["energies"]), "fits": fits})


def score_window(values, levels, *, angles=None, mode="qaoa", selected_control=False):
    """Bounded-memory exact expectation over every accepted data block."""
    total = {}; started = perf_counter()
    for start in range(0, len(values), 256):
        block = values[start:start+256]
        aggregate = block[:, 1:].sum(axis=1) if selected_control else block[:, 0]
        batch = tv.static_batch(aggregate, levels[:3] if selected_control else levels)
        for k, metrics in tv.expected_metrics(batch, block[:, 1:], angles=angles, mode=mode).items():
            row = total.setdefault(k, {"blocks": 0, "absolute_error_sum": [0., 0., 0.], "category_correct_sum": [0., 0., 0.],
                                      "best_cost_sum": 0. if mode != "oracle" else None, "hit_optimum_sum": 0. if mode != "oracle" else None})
            row["blocks"] += len(block)
            for key, dest in (("absolute_error", "absolute_error_sum"), ("category_correct", "category_correct_sum")):
                row[dest] = (np.array(row[dest])+metrics[key].sum(axis=0)).tolist()
            if mode != "oracle":
                row["best_cost_sum"] += float(metrics["best_cost"].sum())
                row["hit_optimum_sum"] += float(metrics["hit_optimum"].sum())
    return {"budgets": total, "blocks": len(values), "scoring_seconds": perf_counter()-started}


def pooled(rows, budget):
    active = [r["score"]["budgets"][budget] for r in rows if budget in r["score"]["budgets"]]
    count = sum(x["blocks"] for x in active)
    tv.require(count > 0, "No evaluation blocks")
    errors = np.sum([x["absolute_error_sum"] for x in active], axis=0)/count
    return {"blocks": count, "per_appliance_mae_w": errors.tolist(), "macro_mae_w": float(errors.mean())}


def validate(folder):
    plan = verify(folder); training = read(folder/"real_training.json")
    tv.require(training["plan_sha256"] == sha(folder/"plan.json"), "Training plan mismatch")
    tv.require(not (folder/"selection.json").exists(), "Parameters already selected")
    for f in training["fits"]: tv.require(sha(folder/f["file"]) == f["sha256"], "Training changed")
    data = extract(folder, plan, "validation"); rows = []
    for item in data["windows"]:
        path = folder/"validation_windows"/(item["window"]["id"]+".json")
        if path.exists(): records = read(path)
        else:
            values = np.load(folder/item["file"], allow_pickle=False)["values"]; records = []
            for f in training["fits"]:
                records.append({"window_id": item["window"]["id"], "fit_id": f["id"], "loss": f["loss"],
                                "score": score_window(values, plan["model"]["levels_w"], angles=f["angles"])})
            save(path, records)
        rows.extend(records); print(f"Validated {item['window']['id']}", flush=True)
    summaries = [{**f, **pooled([r for r in rows if r["fit_id"] == f["id"]], "16")} for f in training["fits"]]
    selection = {loss: min([s for s in summaries if s["loss"] == loss], key=lambda s: (s["macro_mae_w"], s["trial"])) for loss in plan["losses"]}
    save(folder/"validation.json", {"completed_utc": now(), "plan_sha256": sha(folder/"plan.json"),
        "training_sha256": sha(folder/"real_training.json"), "data_sha256": sha(folder/"validation_data.json"), "rows": rows, "summaries": summaries})
    save(folder/"selection.json", {"frozen_utc_before_test_read": now(), "plan_sha256": sha(folder/"plan.json"),
        "validation_sha256": sha(folder/"validation.json"), "selected_per_loss": selection,
        "validation_winner": min(plan["losses"], key=lambda x: selection[x]["macro_mae_w"]),
        "test_use": "Both selected losses reported; no test-based model choice or retuning"})
    print("Parameters frozen before test extraction", flush=True)


def test(folder):
    plan = verify(folder); selection = read(folder/"selection.json")
    tv.require(not (folder/"test.json").exists(), "Test phase complete")
    data = extract(folder, plan, "test"); rows = []
    for item in data["windows"]:
        path = folder/"test_windows"/(item["window"]["id"]+".json")
        if path.exists(): records = read(path)
        else:
            values = np.load(folder/item["file"], allow_pickle=False)["values"]; records = []
            arms = [(x, "qaoa", f["angles"], False) for x, f in selection["selected_per_loss"].items()]
            arms += [(x, x, None, False) for x in ("exact", "uniform", "oracle", "low_power")]
            arms += [("exact_selected_circuits", "exact", None, True)]
            for name, mode, angles, selected in arms:
                records.append({"window_id": item["window"]["id"], "method": name,
                    "score": score_window(values, plan["model"]["levels_w"], angles=angles, mode=mode, selected_control=selected)})
            save(path, records)
        rows.extend(records); print(f"Evaluated {item['window']['id']}", flush=True)
    save(folder/"test.json", {"completed_utc": now(), "plan_sha256": sha(folder/"plan.json"),
        "selection_sha256": sha(folder/"selection.json"), "data_sha256": sha(folder/"test_data.json"), "rows": rows})
    verify(folder)


def summarize(folder):
    plan = verify(folder); synth = read(folder/"synthetic.json"); test_data = read(folder/"test.json")
    selection = read(folder/"selection.json"); groups = []
    for bound in plan["synthetic_gamma_bounds"]:
        for loss in plan["losses"]:
            for phase in ("before_recheck", "after_recheck"):
                rows = [r for r in synth["rows"] if (r["bound"], r["loss"], r["phase"]) == (bound, loss, phase)]
                tv.require(len(rows) == 20, "Synthetic trial coverage")
                for k in ("16", "256"):
                    train_errors = [float(np.mean(r["metrics"][k]["absolute_error"][7])) for r in rows]
                    transfer_errors = [float(np.mean(np.delete(r["metrics"][k]["absolute_error"], 7, axis=0))) for r in rows]
                    costs = [r["metrics"][k]["best_cost"][7] for r in rows]
                    groups.append({"bound": bound, "loss": loss, "phase": phase, "shots": int(k),
                        "training_mae_w": {"mean": float(np.mean(train_errors)), "min": min(train_errors), "max": max(train_errors), "values": train_errors},
                        "transfer_mae_w": {"mean": float(np.mean(transfer_errors)), "min": min(transfer_errors), "max": max(transfer_errors), "values": transfer_errors},
                        "training_best_cost_w2": {"mean": float(np.mean(costs)), "values": costs}})
    table = []
    for method in ("mean", "cvar50", "exact", "uniform", "oracle", "low_power", "exact_selected_circuits"):
        rows = [r for r in test_data["rows"] if r["method"] == method]
        budgets = ("16", "256") if method in ("mean", "cvar50", "uniform") else ("exact" if method == "exact_selected_circuits" else method,)
        for k in budgets: table.append({"method": method, "budget": k, **pooled(rows, k)})
    contrasts = []
    for method, control, kb in (("cvar50", "mean", "16"), ("cvar50", "mean", "256"), ("mean", "uniform", "16"),
                               ("cvar50", "uniform", "16"), ("exact", "oracle", "exact")):
        days = [w["id"] for w in plan["windows"]["test"]]
        left = []; right = []; counts = []
        for day in days:
            a = next(r for r in test_data["rows"] if r["method"] == method and r["window_id"] == day)
            b = next(r for r in test_data["rows"] if r["method"] == control and r["window_id"] == day)
            if not a["score"]["blocks"]: continue
            x = a["score"]["budgets"][kb]; y = b["score"]["budgets"]["oracle" if control == "oracle" else kb]
            tv.require(x["blocks"] == y["blocks"], "Paired block coverage")
            left.append(float(np.mean(x["absolute_error_sum"]))); right.append(float(np.mean(y["absolute_error_sum"]))); counts.append(x["blocks"])
        delta = np.array(left)-right; counts = np.array(counts)
        draws = np.random.default_rng(plan["bootstrap"]["seed"]).integers(0, len(delta), (2000, len(delta)))
        values = delta[draws].sum(axis=1)/counts[draws].sum(axis=1)
        contrasts.append({"method": method, "control": control, "budget": kb, "difference_w": float(delta.sum()/counts.sum()),
                          "descriptive_day_bootstrap_ci95": np.quantile(values, [.025, .975]).tolist(), "days": len(delta),
                          "per_day_difference_w": (delta/counts).tolist()})
    fits = [read(p)["fit"] for p in sorted((folder/"fits").glob("*.json"))]
    tv.require(len(fits) == 100, "All fits required")
    save(folder/"summary.json", {"completed_utc": now(), "plan_sha256": sha(folder/"plan.json"),
        "source_results_sha256": {x: sha(folder/x) for x in ("synthetic.json", "real_training.json", "validation.json", "selection.json", "test.json")},
        "synthetic": groups, "real_test": table, "paired_contrasts": contrasts, "validation_selection": selection,
        "training_evaluations": sum(x["evaluations"] for x in fits), "training_measurements": sum(x["training_shots"] for x in fits),
        "independent_recheck_measurements": sum(x["recheck_shots"] for x in fits),
        "total_classically_sampled_measurements": sum(x["total_measurements"] for x in fits),
        "summed_training_wall_seconds": sum(x["runtime_seconds"] for x in fits), "hardware_jobs": 0,
        "limits": [plan["scope"], plan["freshness_scope"], plan["real_representation"], plan["bootstrap"]["scope"],
                   "Synthetic cases are not independent homes; real training uses a small fixed timestamp sample", "No speed advantage or replacement full IBM/temporal MAE"]})
    print("All planned phases complete; independent audit required", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--mode", choices=("freeze", "synthetic", "train-real", "validate", "test", "summarize", "verify"), required=True)
    args = parser.parse_args(); folder = args.folder.resolve()
    tv.require(folder.is_relative_to(ROOT/"results/quantum_diagnostics") and folder != ROOT/"results/quantum_diagnostics", "Named archive required")
    {"freeze": freeze, "synthetic": synthetic, "train-real": train_real, "validate": validate,
     "test": test, "summarize": summarize, "verify": verify}[args.mode](folder)
