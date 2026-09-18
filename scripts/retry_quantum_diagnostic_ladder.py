#!/usr/bin/env python3
"""One explicitly authorized, unchanged retry of a failed IBM diagnostic.

Copy only frozen scientific inputs, never submission intents/results. The
existing submitter still owns the single submission call and free-plan guard.
Reserve a conservative running-to-finished duration while old accounting is
pending; it is a planning reserve, not a reported QPU charge.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from run_quantum_diagnostic_ladder import CAP, RESERVE, account, check, read, remaining, require, save, sha, submit, utc

SOURCE = ROOT / "results/quantum_diagnostics/ladder"


def prepare(folder):
    require(not folder.exists(), "Use a new retry directory")
    plan = check(SOURCE)
    old = read(SOURCE / "job.json")
    failure = read(SOURCE / "failure_status.json")
    require(old["job_id"] == failure["job_id"] and failure["status"] == "ERROR",
            "Original job is not a documented failure")
    folder.mkdir(parents=True)
    for name in [*plan["files_sha256"], "plan.json"]:
        require(Path(name).name == name and name not in ("intent.json", "job.json", "result.json"),
                "Unsafe retry source file")
        shutil.copyfile(SOURCE / name, folder / name)
    check(folder)
    save(folder / "retry_origin.json", {"created_utc": utc(), "original_job_id": old["job_id"],
         "source_directory": str(SOURCE), "source_plan_sha256": sha(SOURCE / "plan.json"),
         "copied_plan_sha256": sha(folder / "plan.json"),
         "source_failure_sha256": sha(SOURCE / "failure_status.json"),
         "retry_wrapper_sha256": sha(Path(__file__)),
         "scope": "one user-authorized unchanged diagnostic retry; not a new full NILM run",
         "automatic_retry": False})
    print("Frozen diagnostic copied and hash-verified; no job submitted", flush=True)


def planning_reserve(metrics):
    usage = metrics.get("usage") or {}
    charge = usage.get("qpu_charge_time_seconds")
    if usage.get("status") in ("completed", "complete", "finalized", "final"):
        require(charge is not None and math.isfinite(float(charge)) and float(charge) >= 0,
                "Finalized accounting lacks a valid charge")
        return 0
    timestamps = metrics.get("timestamps") or {}
    require(timestamps.get("running") and timestamps.get("finished"),
            "Pending accounting lacks bounded execution timestamps")
    start = datetime.fromisoformat(timestamps["running"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(timestamps["finished"].replace("Z", "+00:00"))
    seconds = (end - start).total_seconds()
    require(math.isfinite(seconds) and seconds >= 0, "Invalid execution timestamp interval")
    return max(CAP, math.ceil(seconds))


def run(folder, allowed):
    require(allowed, "Explicit free-plan submission authorization required")
    check(folder)
    origin = read(folder / "retry_origin.json")
    require(sha(Path(__file__)) == origin["retry_wrapper_sha256"], "Retry wrapper changed")
    require(sha(folder / "plan.json") == origin["source_plan_sha256"], "Scientific protocol changed")
    require(not (folder / "intent.json").exists() and not (folder / "retry_preflight.json").exists(),
            "Prior attempt exists; reconcile/collect it instead of retrying")
    service = account()
    job = service.job(origin["original_job_id"])
    require(str(job.status()) == "ERROR", "Original job is not terminal ERROR")
    metrics = job.metrics()
    reserve = planning_reserve(metrics)
    allowance = remaining(service)
    required = reserve + CAP + RESERVE
    require(allowance >= required, "Insufficient free allowance including unresolved original-job reserve")
    save(folder / "retry_preflight.json", {"checked_utc": utc(), "original_job_id": job.job_id(),
         "original_status": str(job.status()), "original_usage_as_observed": metrics.get("usage"),
         "pending_accounting_planning_reserve_s": reserve, "available_free_s": allowance,
         "required_free_s": required, "retry_campaign_cap_s": CAP, "free_reserve_s": RESERVE,
         "note": "Pending reserve is conservative elapsed-time planning, not finalized QPU charge; no paid usage"})
    # submit writes an immutable intent before the only remote mutation.
    submit(folder, allowed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "submit"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-free-submission", action="store_true")
    args = parser.parse_args()
    try:
        folder = args.output_dir.resolve()
        if args.mode == "prepare":
            prepare(folder)
        else:
            run(folder, args.allow_free_submission)
    except Exception as error:
        # Do not expose credential-bearing request details.
        print({"status": "stopped safely", "error_type": type(error).__name__}, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
