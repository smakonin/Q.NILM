#!/usr/bin/env python3
"""Report completed REFIT archives without reading source measurements.

Reads only protocol, frozen_models, test_quality, summary and an optional
independent audit receipt. The report is not marked closed unless a passing
audit is bound to the current protocol and summary. Derived outputs are new,
exclusive-create files; no frozen experiment artifact is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("qaoa_ideal", "uniform", "exact_chunk", "exact_full", "training_mean")
LABELS = {"qaoa_ideal": "Q.NILM, ideal QAOA", "uniform": "Uniform feasible",
          "exact_chunk": "Exact two-interval", "exact_full": "Exact full-run DP",
          "training_mean": "Training-mean constant"}
PASS_STATUSES = {"passed", "passed_complete", "passed_independent_audit", "complete_passed"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def number(value, precision=3):
    if value is None:
        return "—"
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite number in report evidence")
    return f"{value:.{precision}f}"


def percentage(numerator, denominator, precision=2):
    return "—" if not denominator else number(100 * numerator / denominator, precision) + "%"


def interval(value):
    if value is None:
        return "—"
    return f"{number(value['difference_w'])} [{number(value['descriptive_95_interval_w'][0])}, {number(value['descriptive_95_interval_w'][1])}]"


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def audit_status(audit, digests):
    """Fail closed if status or current-archive binding is unavailable."""
    if audit is None:
        return {"passed_current_archive": False, "status": "pending", "reason": "No independent audit receipt supplied"}
    status = str(audit.get("status", "")).lower()
    if status not in PASS_STATUSES:
        return {"passed_current_archive": False, "status": status or "unrecognized", "reason": "Independent audit does not explicitly report a recognized passing status"}
    mappings = [audit.get(name, {}) for name in ("source_sha256", "artifact_sha256", "input_sha256", "hashes")]

    def anchored(name):
        key = name.removesuffix(".json") + "_sha256"
        if audit.get(key) is not None:
            return audit[key]
        for mapping in mappings:
            if isinstance(mapping, dict):
                for path, value in mapping.items():
                    if Path(path).name == name:
                        return value
        return None

    for name in ("protocol.json", "summary.json"):
        if anchored(name) != digests[name]:
            return {"passed_current_archive": False, "status": "unbound_or_stale", "reason": f"Audit is not hash-bound to current {name}"}
    for name in ("frozen_models.json", "test_quality.json"):
        if anchored(name) is not None and anchored(name) != digests[name]:
            return {"passed_current_archive": False, "status": "stale", "reason": f"Audit hash differs for {name}"}
    return {"passed_current_archive": True, "status": status, "reason": "Passing independent receipt matches current protocol and summary"}


def summarize_quality(protocol, models, quality):
    """Retain every selected window and reconcile rejected-block seconds."""
    expected = {(home["house"], window["id"]): window for home in protocol["homes"]
                for window in home["manifest"]["splits"]["test"]["windows"]}
    keyed = {(value["house"], value["window"]["id"]): value for value in quality}
    if len(keyed) != len(quality) or keyed.keys() != expected.keys():
        raise ValueError("Every selected test window must appear exactly once")
    homes, windows = [], []
    for home in protocol["homes"]:
        house = home["house"]
        model = models["homes"][str(house)]
        train = model["quality"]
        train_expected = {window["id"]: window for window in home["manifest"]["splits"]["train"]["windows"]}
        if len(train) != len(train_expected) or {value["window"]["id"] for value in train} != set(train_expected):
            raise ValueError("Training quality does not cover every selected training window")
        if sum(value["valid_blocks"] for value in train) != model["training_blocks"]:
            raise ValueError("Training block count differs from frozen model")
        actual_counts = list(model["actual_counts"])
        if len(actual_counts) != 4 or any(not isinstance(value, int) or value < 1 for value in actual_counts):
            raise ValueError("Invalid actual fitted categorical state counts")
        if actual_counts != [len(levels) for levels in model["levels"]]:
            raise ValueError("Actual model counts differ from frozen level arrays")
        selected = [keyed[(house, window["id"])] for window in home["manifest"]["splits"]["test"]["windows"]]
        result = {"house": house, "channels": home["channels"], "actual_model_counts": actual_counts,
                  "training_blocks": model["training_blocks"], "selected_training_windows": len(train),
                  "empty_training_windows": [value["window"]["id"] for value in train if not value["valid_blocks"]],
                  "selected_test_windows": len(selected), "nonempty_test_windows": 0, "empty_test_windows": [],
                  "test_blocks": 0, "expected_complete_blocks": 0, "selected_seconds": 0,
                  "large_gap_seconds": 0.0, "invalid_supported_seconds": 0.0, "boundary_uncovered_seconds": 0.0,
                  "valid_covered_seconds": 0.0, "partial_final_seconds": 0.0,
                  "issues_source_rows_including_support": 0, "nonfinite_source_rows_including_support": 0,
                  "negative_source_rows_including_support": 0, "invalid_schema_source_rows_including_support": 0,
                  "valid_source_rows_including_support": 0, "all_selected_zero_source_rows_including_support": 0,
                  "all_selected_zero_retained_blocks": 0, "zero_retained_blocks_by_channel": {}}
        for value in selected:
            window, q = value["window"], value["quality"]
            if window != expected[(house, window["id"])]:
                raise ValueError("Test-quality window differs from frozen timestamp manifest")
            blocks = int(value["valid_blocks"])
            if blocks != q["valid_blocks"] or not 0 <= blocks <= q["expected_complete_blocks"]:
                raise ValueError("Invalid retained-block accounting")
            zeros = q["zero_retained_blocks_by_channel"]
            if set(zeros) != {"Aggregate", *home["channels"]} or any(not 0 <= count <= blocks for count in zeros.values()):
                raise ValueError("Missing or invalid per-channel zero-block counts")
            if not 0 <= q["all_selected_appliances_zero_retained_blocks"] <= min(zeros[channel] for channel in home["channels"]):
                raise ValueError("Joint-zero count exceeds a selected channel's zero count")
            seconds = window["end_unix"] - window["start_unix"]
            if seconds != 30 * q["expected_complete_blocks"] + q["partial_final_seconds"]:
                raise ValueError("Complete/partial grid duration differs from selected window")
            accounted = q["valid_covered_seconds"] + q["supported_invalid_seconds"] + q["unsupported_large_gap_seconds"] + q["uncovered_boundary_seconds"]
            if not math.isclose(accounted, 30 * q["expected_complete_blocks"], abs_tol=1e-6, rel_tol=1e-12):
                raise ValueError("Source support categories do not reconcile requested complete seconds")
            result["test_blocks"] += blocks
            result["expected_complete_blocks"] += q["expected_complete_blocks"]
            result["selected_seconds"] += seconds
            result["nonempty_test_windows"] += int(blocks > 0)
            if not blocks:
                result["empty_test_windows"].append(window["id"])
            for destination, source in (("large_gap_seconds", "unsupported_large_gap_seconds"),
                                        ("invalid_supported_seconds", "supported_invalid_seconds"),
                                        ("boundary_uncovered_seconds", "uncovered_boundary_seconds"),
                                        ("valid_covered_seconds", "valid_covered_seconds"),
                                        ("partial_final_seconds", "partial_final_seconds"),
                                        ("issues_source_rows_including_support", "rows_issues"),
                                        ("nonfinite_source_rows_including_support", "rows_nonfinite"),
                                        ("negative_source_rows_including_support", "rows_negative"),
                                        ("invalid_schema_source_rows_including_support", "rows_invalid_schema"),
                                        ("valid_source_rows_including_support", "valid_source_rows"),
                                        ("all_selected_zero_source_rows_including_support", "all_selected_appliances_zero_source_rows"),
                                        ("all_selected_zero_retained_blocks", "all_selected_appliances_zero_retained_blocks")):
                result[destination] += q[source]
            for channel, count in q["zero_retained_blocks_by_channel"].items():
                result["zero_retained_blocks_by_channel"][channel] = result["zero_retained_blocks_by_channel"].get(channel, 0) + count
            windows.append({"house": house, "window_id": window["id"], "start_unix": window["start_unix"], "end_unix": window["end_unix"],
                            "valid_blocks": blocks, "expected_complete_blocks": q["expected_complete_blocks"],
                            "empty": blocks == 0, "outcome": "unevaluable: no fully supported block" if not blocks else "included in each of five method summaries",
                            "expected_method_seed_score_records": 9 if blocks else 0,
                            "source_window_sha256": value["source_window_sha256"]})
        result["retained_seconds"] = 30 * result["test_blocks"]
        result["excluded_seconds"] = result["selected_seconds"] - result["retained_seconds"]
        result["valid_seconds_discarded_with_partial_blocks"] = result["valid_covered_seconds"] - result["retained_seconds"]
        if result["valid_seconds_discarded_with_partial_blocks"] < -1e-6:
            raise ValueError("Retained blocks exceed valid supported seconds")
        decomposed = sum(result[key] for key in ("large_gap_seconds", "invalid_supported_seconds", "boundary_uncovered_seconds", "partial_final_seconds", "valid_seconds_discarded_with_partial_blocks"))
        if not math.isclose(decomposed, result["excluded_seconds"], abs_tol=1e-5, rel_tol=1e-12):
            raise ValueError("Excluded-second decomposition does not reconcile")
        result["coverage_fraction"] = result["test_blocks"] / result["expected_complete_blocks"] if result["expected_complete_blocks"] else None
        homes.append(result)
    return homes, windows


def build_evidence(protocol, models, quality, summary, provenance, audit=None):
    if summary.get("status") not in ("complete", "no_evaluable_homes"):
        raise ValueError("A finalized summary is required; do not report partial evaluation")
    if summary["protocol_sha256"] != provenance["sha256"]["protocol.json"] or models["protocol_sha256"] != summary["protocol_sha256"]:
        raise ValueError("Model, summary and current protocol must match")
    houses = [home["house"] for home in protocol["homes"]]
    if len(set(houses)) != len(houses) or set(models["homes"]) != {str(house) for house in houses}:
        raise ValueError("Frozen model/home cohort mismatch")
    if tuple(protocol["methods"]) != METHODS:
        raise ValueError("Unexpected arm set or ordering")
    home_quality, windows = summarize_quality(protocol, models, quality)
    day_pairs = {(value["house"], value["right"]): value for value in summary["paired_day_comparisons"]}
    cross_pairs = {value["right"]: value for value in summary["paired_home_comparisons"]}
    evaluable = [value["house"] for value in home_quality if value["test_blocks"]]
    if set(summary["per_home"]) != {str(value) for value in evaluable}:
        raise ValueError("Summary silently omits or adds a home")
    if set(summary["nonevaluable_homes"]) != set(houses) - set(evaluable):
        raise ValueError("Unevaluable homes not explicitly retained")
    home_rows = []
    for q in home_quality:
        house = q["house"]
        metrics = summary["per_home"].get(str(house))
        if metrics is not None:
            if set(metrics) != set(METHODS):
                raise ValueError("Missing per-home model")
            for method in METHODS:
                value = metrics[method]
                if value["blocks"] != q["test_blocks"] or value["windows"] != q["nonempty_test_windows"]:
                    raise ValueError("Method coverage differs from all selected nonempty windows")
            for right in METHODS[1:]:
                value = day_pairs.get((house, right))
                if value is None or value["days"] != q["nonempty_test_windows"]:
                    raise ValueError("Missing or mismatched paired-day comparison")
        source_coverage = summary["coverage"][str(house)]
        for key, actual in (("selected_windows", q["selected_test_windows"]), ("nonempty_windows", q["nonempty_test_windows"]),
                            ("valid_blocks", q["test_blocks"]), ("expected_blocks", q["expected_complete_blocks"]),
                            ("issues_rows", q["issues_source_rows_including_support"]),
                            ("all_appliances_zero_retained_blocks", q["all_selected_zero_retained_blocks"])):
            if source_coverage[key] != actual:
                raise ValueError(f"Quality-derived {key} differs from independently summarized coverage")
        if source_coverage["empty_windows"] != q["empty_test_windows"]:
            raise ValueError("Empty-window identities differ between quality and summary")
        for key in ("large_gap_seconds", "invalid_supported_seconds"):
            if not math.isclose(source_coverage[key], q[key], abs_tol=1e-6, rel_tol=1e-12):
                raise ValueError("Excluded-source-second accounting differs from summary")
        home_rows.append({**q, "macro_mae_w": {method: metrics[method]["macro_mae_w"] if metrics else None for method in METHODS},
                          "qaoa_minus_uniform_day_interval": day_pairs.get((house, "uniform"))})
    overall = []
    for method in METHODS:
        value = summary["overall"][method]
        if value["evaluable_homes"] != len(evaluable) or value["blocks"] != sum(q["test_blocks"] for q in home_quality) or value["windows"] != sum(q["nonempty_test_windows"] for q in home_quality):
            raise ValueError("Overall denominator differs from selected cohort coverage")
        if evaluable:
            mean = sum(summary["per_home"][str(house)][method]["macro_mae_w"] for house in evaluable) / len(evaluable)
            pooled = sum(summary["per_home"][str(house)][method]["absolute_error_sum_w"] for house in evaluable) / (3 * value["blocks"])
            if not math.isclose(mean, value["equal_home_macro_mae_w"], abs_tol=1e-9, rel_tol=1e-12) or not math.isclose(pooled, value["block_pooled_macro_mae_w"], abs_tol=1e-9, rel_tol=1e-12):
                raise ValueError("Overall macro-MAE arithmetic does not reconcile")
        overall.append({"method": method, "label": LABELS[method], **value})
    if evaluable and set(cross_pairs) != set(METHODS[1:]):
        raise ValueError("All four prespecified cross-home comparisons are required")
    audit_check = audit_status(audit, provenance["sha256"])
    eligible = summary["status"] == "complete" and bool(evaluable) and audit_check["passed_current_archive"]
    return {"status": "complete_audited" if eligible else "audit_pending" if summary["status"] == "complete" else "no_evaluable_homes",
            "stage_closure_eligible": eligible, "audit": audit_check, "provenance": provenance,
            "scope": "External REFIT within-home supervised calibration, ideal simulation; not zero-shot unseen-home inference or QPU execution",
            "selected_homes": houses, "evaluable_homes": evaluable, "nonevaluable_homes": summary["nonevaluable_homes"],
            "requested_model_counts": protocol["requested_counts"], "target_order": protocol["target_order"],
            "shots_per_chunk": protocol["shots"], "stochastic_seeds": protocol["seeds"],
            "overall": overall, "per_home": home_rows,
            "cross_home_comparisons": [cross_pairs[method] for method in METHODS[1:] if method in cross_pairs],
            "all_paired_day_comparisons": summary["paired_day_comparisons"],
            "all_selected_window_accounting": windows,
            "coverage_totals": {"selected_test_windows": len(windows), "nonempty_test_windows": sum(not row["empty"] for row in windows),
                                "empty_test_windows": sum(row["empty"] for row in windows), "test_blocks": sum(row["test_blocks"] for row in home_quality),
                                "training_blocks": sum(row["training_blocks"] for row in home_quality),
                                "expected_complete_blocks": sum(row["expected_complete_blocks"] for row in home_quality),
                                "expected_method_seed_score_records": sum(row["expected_method_seed_score_records"] for row in windows),
                                "excluded_seconds": sum(row["excluded_seconds"] for row in home_quality)},
            "limitations": summary.get("limitations", protocol["limitations"]), "timing": summary.get("timing"),
            "timing_interpretation": "Recorded REFIT wall times were collected during concurrent local audit and other work. They are descriptive run telemetry, not a controlled speed benchmark, quantum speedup comparison or QPU-time forecast."}


def render_report(data):
    homes, totals = data["per_home"], data["coverage_totals"]
    provenance = data["provenance"]
    def source_link(name, label):
        source = provenance.get("paths", {}).get(name)
        target = os.path.relpath(source, provenance["output_path"]) if source and provenance.get("output_path") else name
        return f"[{label}](<{target}>)"
    source_links = ", ".join(source_link(name, label) for name, label in (
        ("protocol.json", "Frozen protocol"), ("frozen_models.json", "train-only frozen models"),
        ("test_quality.json", "all selected test-window quality records"), ("summary.json", "complete metric summary")))
    status = ("The completed external run has a passing, hash-matched independent audit; the bounded calibrated-home Stage G evaluation can be marked complete."
              if data["stage_closure_eligible"] else "Independent acceptance is pending or the cohort is unevaluable. Do not mark Stage G closed from this report. " + data["audit"]["reason"])
    overall_rows = [[row["label"], number(row["equal_home_macro_mae_w"]), number(row["block_pooled_macro_mae_w"]), row["evaluable_homes"], row["blocks"]] for row in data["overall"]]
    home_rows = [[row["house"], *[number(row["macro_mae_w"][method]) for method in METHODS], interval(row["qaoa_minus_uniform_day_interval"])] for row in homes]
    comparisons = [[LABELS[row["right"]], number(row["difference_w"]), f"[{number(row['descriptive_95_interval_w'][0])}, {number(row['descriptive_95_interval_w'][1])}]", row["homes"]] for row in data["cross_home_comparisons"]]
    coverage = [[row["house"], "/".join(map(str, row["actual_model_counts"])), row["training_blocks"], row["test_blocks"],
                 f"{row['nonempty_test_windows']}/{row['selected_test_windows']}", len(row["empty_test_windows"]), percentage(row["test_blocks"], row["expected_complete_blocks"])] for row in homes]
    zero = [[row["house"], *[percentage(row["zero_retained_blocks_by_channel"].get(channel, 0), row["test_blocks"]) for channel in row["channels"]],
             percentage(row["all_selected_zero_retained_blocks"], row["test_blocks"]), row["issues_source_rows_including_support"]] for row in homes]
    excluded = [[row["house"], number(row["excluded_seconds"], 1), number(row["large_gap_seconds"], 1), number(row["invalid_supported_seconds"], 1),
                 number(row["boundary_uncovered_seconds"], 1), number(row["valid_seconds_discarded_with_partial_blocks"], 1), number(row["partial_final_seconds"], 1)] for row in homes]
    empty = [f"Home {row['house']}: " + ", ".join(row["empty_test_windows"]) for row in homes if row["empty_test_windows"]]
    lines = ["# REFIT external validation: within-home calibrated Q.NILM", "", status, "",
             f"All {len(data['selected_homes'])} selected homes and {totals['selected_test_windows']} fixed test windows remain accounted for: {totals['nonempty_test_windows']} windows contribute {totals['test_blocks']:,} retained 30-second blocks; {totals['empty_test_windows']} have no usable block and are explicitly unevaluable. {len(data['evaluable_homes'])} homes contribute accuracy outcomes. No empty or low-quality window is replaced.", "",
             "## Overall appliance-power error", "", "The primary outcome is the equal-home mean of three-appliance macro MAE. Block-pooled MAE is a separately reported, coverage-weighted quantity; it is not the same estimand. Values are watts. For the stochastic methods, error sums are averaged over three fixed seeds before ratios and bootstrap calculations.", "",
             markdown_table(["Method", "Equal-home macro MAE (W)", "Block-pooled macro MAE (W)", "Evaluable homes", "Retained blocks"], overall_rows), "",
             "## Every selected home", "", "All errors are watts; the final column is ideal QAOA minus uniform feasible sampling, with a descriptive paired-day 95% interval (2,000 resamples). Negative differences favour QAOA. Null entries mean unevaluable, not zero error.", "",
             markdown_table(["Home", "Ideal QAOA", "Uniform", "Exact chunk", "Exact full DP", "Training mean", "QAOA − uniform [95%]"], home_rows), "",
             "## Four prespecified cross-home comparisons", "", "The differences below average home-specific macro-MAE differences with equal home weights; intervals use 10,000 paired home resamples. Homes are not a random population sample, so these are descriptive, not multiplicity-adjusted superiority or causal claims.", "",
             markdown_table(["QAOA minus comparator", "Difference (W)", "95% home interval (W)", "Homes"], comparisons), "",
             "## Frozen model sizes and coverage", "", "Actual state counts are in washing/cooling/dishwasher/residual-background order; requested counts were 4/3/2/3. Smaller fitted counts are retained when training diversity is insufficient. Every model was calibrated using only its own home's earlier training partition, then frozen before any REFIT test evaluation. This is not zero-shot transfer to uncalibrated homes.", "",
             markdown_table(["Home", "Actual counts", "Training blocks", "Test blocks", "Nonempty/selected days", "Empty days", "Test coverage"], coverage), "",
             "## Zero observations and Issues flags", "", "Zero rates below concern retained blocks of the published cleaned channels, not verified inactivity. The official cleaned release contains unflagged forward-fills and spike-to-zero replacements; Issues flags do not identify all imputations. Source streams were acquired asynchronously. Flagged-row counts include scanned boundary-support rows and are not counts of rejected blocks.", "",
             markdown_table(["Home", "Washing zero", "Cooling zero", "Dishwasher zero", "All three zero", "Issues-flagged source rows"], zero), "",
             "## Excluded-second accounting", "", "Only fully supported 30-second blocks enter scoring. The total excluded duration decomposes into large-gap time, invalid supported time, uncovered boundary time, otherwise-valid time lost when an incomplete block is rejected, and any incomplete final grid duration. These categories are not added to rejected-block duration again. Invalid supported time is a union of quality failures; overlapping row flags are not treated as additive durations.", "",
             markdown_table(["Home", "Total excluded s", "Gap >16 s", "Invalid support s", "Boundary s", "Valid s lost with rejected blocks", "Partial final s"], excluded), "",
             "## Every selected window and failure accounting", "",
             f"The archived quality log accounts for every selected window. The finalized per-method summaries reconcile the same nonempty windows and block counts for all five arms. There are {totals['expected_method_seed_score_records']:,} expected method/seed score records (nine per nonempty window: three QAOA seeds, three uniform seeds, and one record for each deterministic control). Empty windows produce no error estimate. A model execution failure would prevent a finalized accepted summary; it must not be interpreted as a zero error or silently dropped window.", "",
             "Empty selected test windows: " + ("; ".join(empty) if empty else "none") + ".", "",
             "All selected window IDs, timestamps, retained blocks and empty-window outcomes are included in report-data.json. This report rechecks summary/quality consistency; independent acceptance still depends on the audit status stated above.", "",
             "## Interpretation limits", "",
             f"The ideal QAOA and uniform arms use {data['shots_per_chunk']} candidate shots per chunk and three fixed sampling seeds, with their own inferred preceding states. R1Hz-trained QAOA angles are transferred without REFIT angle optimization; centroids, residual-background levels and thresholds are calibrated within each REFIT home. The two-level dishwasher remains a coarse proxy. Exact chunk/full-run controls and the training-mean constant use this study's implementation, not reproduced published scores.", "",
             "These are ideal circuit simulations and classical controls, not QPU measurements. No quantum speed advantage, causal benefit, universally best method, clean-observation ground-truth guarantee, or zero-shot unseen-home generalization is established. Errors are against resampled published cleaned channels. Recorded local timings do not establish a quantum speedup.", "",
             data["timing_interpretation"], "",
             "## Provenance", "",
             "- " + source_links + ".",
             "- [Exact report values and all-window accounting](report-data.json), [LaTeX home table](table.tex).",
             "- Independent audit status: " + data["audit"]["status"] + "; " + data["audit"]["reason"] + ".",
             "- Source DOI: [official cleaned REFIT dataset](https://doi.org/10.15129/9ab14b0e-19ac-4279-938f-27f643078cec).",
             "- All input-artifact and report-generator SHA-256 hashes are recorded in report-data.json. The reporter never reads measurement CSV files or alters frozen experimental inputs or results; it adds only these derived report files.", ""]
    return "\n".join(lines)


def render_latex(data):
    lines = [r"\begin{table*}[t]", r"\caption{REFIT Within-Home Calibrated Evaluation: Appliance Macro MAE (W)}",
             r"\label{tab:stage-g-refit}", r"\centering", r"\footnotesize", r"\begin{tabular}{rrrrrrl}", r"\toprule",
             r"Home & Ideal QAOA & Uniform & Exact chunk & Exact full DP & Training mean & QAOA--uniform, 95\% interval (W)\\", r"\midrule"]
    for row in data["per_home"]:
        values = [number(row["macro_mae_w"][method]).replace("—", "---") for method in METHODS]
        difference = interval(row["qaoa_minus_uniform_day_interval"]).replace("—", "---")
        lines.append(" & ".join([str(row["house"]), *values, difference]) + r"\\")
    lines += [r"\midrule"]
    for key, label in (("equal_home_macro_mae_w", "Equal home"), ("block_pooled_macro_mae_w", "Block pooled")):
        by_method = {row["method"]: row for row in data["overall"]}
        lines.append(" & ".join([label, *[number(by_method[method][key]).replace("—", "---") for method in METHODS], "---"]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\vspace{1mm}", r"\begin{minipage}{0.98\textwidth}\footnotesize",
              "Every selected home is retained; dashes denote unevaluable outcomes. Each home is calibrated from its own earlier training partition, not zero-shot transfer. Ideal QAOA is a classical circuit simulation, not QPU execution. The stochastic arms average three fixed seeds with 256 shots per chunk. Home intervals are paired-day descriptive bootstrap intervals; negative differences favour QAOA. Errors use the official cleaned channels, which contain unflagged imputations."]
    if not data["stage_closure_eligible"]:
        lines.append(r"\textbf{Independent audit pending: this derived table does not close Stage G.}")
    lines += [r"\end{minipage}", r"\end{table*}", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/stage_g/refit/run_001")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output_dir.resolve() if args.output_dir else run
    targets = [output / name for name in ("report.md", "report-data.json", "table.tex")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Derived outputs already exist; select a new output directory to preserve revisions")
    names = ("protocol.json", "frozen_models.json", "test_quality.json", "summary.json")
    values = {name: read(run / name) for name in names}
    provenance = {"run_path": str(run), "output_path": str(output), "paths": {name: str(run / name) for name in names},
                  "sha256": {name: sha(run / name) for name in names},
                  "report_generator_path": str(Path(__file__).resolve()), "report_generator_sha256": sha(__file__)}
    audit_path = args.audit.resolve() if args.audit else run / "independent_audit.json"
    audit = read(audit_path) if audit_path.exists() else None
    if audit is not None:
        provenance.update(audit_path=str(audit_path), audit_sha256=sha(audit_path))
    data = build_evidence(values["protocol.json"], values["frozen_models.json"], values["test_quality.json"], values["summary.json"], provenance, audit)
    output.mkdir(parents=True, exist_ok=True)
    for path, content in zip(targets, (render_report(data), json.dumps(data, indent=2, allow_nan=False) + "\n", render_latex(data))):
        with path.open("x") as stream:
            stream.write(content)
    print(json.dumps({"status": data["status"], "stage_closure_eligible": data["stage_closure_eligible"],
                      "outputs": [str(path) for path in targets], "coverage": data["coverage_totals"]}, indent=2))


if __name__ == "__main__":
    main()
