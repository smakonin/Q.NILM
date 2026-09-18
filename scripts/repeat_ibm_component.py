#!/usr/bin/env python3
"""One free-only, different-UTC-date repeat of the frozen component ladder.

No retraining, recompilation, changed ordering, or automatic retry. The
original ladder submitter owns the single remote submission. Credentials
and provider exception text are never printed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_quantum_diagnostic_ladder import (
    CAP, RESERVE, account, check, read, remaining, require, save, sha, submit, utc,
)
from retry_quantum_diagnostic_ladder import planning_reserve

SOURCE = ROOT / "results/quantum_diagnostics/ladder_retry_001"
FAILED_JOB = "dajvplj9k43c73ahpfo0"


def independent_date(original_finished: str, now: datetime) -> bool:
    original = datetime.fromisoformat(original_finished.replace("Z", "+00:00"))
    return now.astimezone(timezone.utc).date() > original.astimezone(timezone.utc).date()


def prepare(folder: Path) -> None:
    require(not folder.exists(), "Choose an unused repeat directory")
    plan = check(SOURCE)
    require((SOURCE / "independent_counts_audit.json").exists(), "Original count audit absent")
    require((SOURCE / "independent_ideal_audit.json").exists(), "Original ideal audit absent")
    result, metrics = read(SOURCE / "result.json"), read(SOURCE / "metrics.json")
    require(independent_date(metrics["timestamps"]["finished"], datetime.now(timezone.utc)),
            "Must execute on a later UTC date")
    folder.mkdir(parents=True)
    for name in [*plan["files_sha256"], "plan.json"]:
        require(Path(name).name == name and name not in ("job.json", "intent.json", "result.json"),
                "Unsafe repeat source")
        shutil.copyfile(SOURCE / name, folder / name)
    check(folder)
    save(folder / "repeat_origin.json", {
        "created_utc": utc(), "source_directory": str(SOURCE.relative_to(ROOT)),
        "source_job_id": result["job_id"],
        "source_finished_utc": metrics["timestamps"]["finished"],
        "source_plan_sha256": sha(SOURCE / "plan.json"),
        "source_result_sha256": sha(SOURCE / "result.json"),
        "original_ideal_audit_sha256": sha(SOURCE / "independent_ideal_audit.json"),
        "original_counts_audit_sha256": sha(SOURCE / "independent_counts_audit.json"),
        "wrapper_sha256": sha(Path(__file__)), "automatic_retry": False,
        "scope": "one different-date component repeat; no full-campaign NILM MAE",
        "primary_endpoint": "raw one-hot feasibility by width and component; all 1024 shots retained",
        "analysis": "retain both dates separately; descriptive percentage-point differences; two dates do not estimate general calibration variability",
        "unchanged": "all QPY/QASM, angles, example bindings, PUB order, placements, shots and Runtime options",
    })
    print("Repeat prepared and frozen; no job submitted", flush=True)


def run(folder: Path, allowed: bool) -> None:
    require(allowed, "Explicit free-only authorization required")
    check(folder)
    origin = read(folder / "repeat_origin.json")
    require(sha(Path(__file__)) == origin["wrapper_sha256"], "Wrapper changed")
    require(sha(folder / "plan.json") == origin["source_plan_sha256"], "Plan changed")
    require(not (folder / "repeat_preflight.json").exists() and not (folder / "intent.json").exists(),
            "Prior attempt present; reconcile instead of resubmitting")
    require(independent_date(origin["source_finished_utc"], datetime.now(timezone.utc)),
            "Must execute on a later UTC date")
    service = account()
    previous = service.job(origin["source_job_id"])
    require(str(previous.status()) == "DONE", "Original job not complete")
    failed = service.job(FAILED_JOB)
    require(str(failed.status()) == "ERROR", "Unexpected old failed-job state")
    old_metrics = failed.metrics()
    reserve = planning_reserve(old_metrics)
    available = remaining(service)
    require(available >= CAP + RESERVE + reserve, "Insufficient free allowance")
    save(folder / "repeat_preflight.json", {
        "checked_utc": utc(), "available_free_s": available,
        "required_free_s": CAP + RESERVE + reserve,
        "failed_job_usage": old_metrics.get("usage"),
        "pending_accounting_reserve_s": reserve,
        "campaign_cap_s": CAP, "free_reserve_s": RESERVE,
        "paid_allowed": False, "automatic_retry": False,
    })
    submit(folder, allowed=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "submit"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-free-submission", action="store_true")
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            prepare(args.output_dir.resolve())
        else:
            run(args.output_dir.resolve(), args.allow_free_submission)
    except Exception as error:
        print({"status": "stopped safely", "error_type": type(error).__name__}, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
