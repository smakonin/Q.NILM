#!/usr/bin/env python3
"""One-time serialized continuation of the already-authorized IBM campaign.

Waits for the pinned initial process to exit naturally, then invokes the
existing Open-Plan-only runner once. There are no retries, cancellation calls,
paid overrides, scheduling services, or manuscript edits in this helper.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]


class ContinuationStopped(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ContinuationStopped(message)


def read(path):
    return json.loads(Path(path).read_text())


def process_identity(pid):
    """Read only the pinned PID's start time and executable identity."""
    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart=,comm="],
                            capture_output=True, text=True, check=False)
    if result.returncode == 1 and not result.stdout.strip() and not result.stderr.strip():
        return None
    require(result.returncode == 0 and not result.stderr.strip() and bool(result.stdout.strip()),
            "Cannot reliably read the initial process identity")
    require(len(result.stdout.strip().splitlines()) == 1, "Process identity returned multiple entries")
    return result.stdout.strip()


def check_stop_files(campaign):
    folder = Path(campaign) / "hardware"
    for path in (folder / "STOP", folder / "budget_stop.json", folder / "continuation/STOP"):
        require(not path.exists(), f"Stop/budget marker present: {path.name}")


def wait_for_initial_exit(pid, expected_identity, campaign, checkpoint, *, poll_seconds=10,
                          identity_reader=process_identity, sleeper=time.sleep):
    """Never signal the initial process; reject PID reuse or an absent initial PID."""
    check_stop_files(campaign)
    initial = identity_reader(pid)
    require(initial is not None and initial == expected_identity,
            "Initial process is absent or its exact identity does not match")
    checkpoint("waiting_initial_process", initial_pid=pid, expected_process_identity=expected_identity)
    while True:
        check_stop_files(campaign)
        sleeper(poll_seconds)
        current = identity_reader(pid)
        if current is None:
            checkpoint("initial_process_exited", initial_pid=pid)
            return
        require(current == expected_identity, "Initial PID now identifies a different process; refusing continuation")


def load_auditor():
    path = REPO / "scripts/audit_quantum_heldout_hardware.py"
    spec = importlib.util.spec_from_file_location("qnilm_continuation_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_anchor(campaign, expected_plan, expected_job, *, require_stage_zero):
    """Verify immutable study/initial-job anchors and explicit code revision chain."""
    check_stop_files(campaign)
    auditor = load_auditor()
    folder = campaign / "hardware"
    require(auditor.sha(folder / "plan.json") == expected_plan, "Hardware plan hash differs from pinned plan")
    plan = read(folder / "plan.json")
    protocol = read(campaign / "protocol.json")
    require(plan["instance_plan"] == "open" and protocol["hardware_paid_execution"] is False,
            "Only the original Open Plan/no-paid experiment may continue")
    require(plan["maximum_campaign_usage_s"] == 540 and protocol["hardware_maximum_campaign_usage_s"] == 540,
            "The original 540-second campaign cap changed")
    auditor.verify_code_hashes(REPO, folder, plan, expected_plan)
    revision = read(folder / "code_revision_001.json")
    require(revision.get("job_running_initial_code") == expected_job,
            "Code revision does not identify the pinned initial job")
    intent = read(folder / "stage_000_job.json")
    require(intent.get("job_id") == expected_job and intent.get("stage") == 0
            and intent.get("plan_sha256") == expected_plan, "Initial submission intent differs")
    require(read(folder / "start.json")["plan_sha256"] == expected_plan, "Started campaign plan anchor differs")
    if require_stage_zero:
        result = read(folder / "stage_000_result.json")
        require(result.get("job_id") == expected_job and result.get("stage") == 0
                and result.get("plan_sha256") == expected_plan, "Initial result differs from pinned job/plan")
        require([row["window_id"] for row in result["rows"]] == intent["days"], "Initial result PUB ordering differs")
        require((folder / "stage_000_runtime.json.gz").exists(), "Initial raw Runtime result is missing")
        report = auditor.audit(campaign)
        require(report["observed"]["stages"] >= 1, "Initial stage has not been independently audited")
        require(not report["complete"], "Campaign already completed; do not launch another run")
        return report
    return None


def execute_pipeline(campaign, work, checkpoint, launch):
    """Invoke run once, then gate each artifact phase on successful completion."""
    check_stop_files(campaign)
    runner = REPO / "scripts/run_quantum_heldout_ibm.py"
    auditor = REPO / "scripts/audit_quantum_heldout_hardware.py"
    summarizer = REPO / "scripts/summarize_quantum_heldout.py"
    require(not (campaign / "analysis_complete").exists(), "analysis_complete already exists; refusing overwrite")
    require(not (campaign / "hardware/summary.json").exists()
            and not (campaign / "hardware/test_windows.json").exists(), "Hardware score artifacts already exist")
    command = [sys.executable, str(runner), "--mode", "run", "--output-dir", str(campaign)]
    require(launch("patched_run_once", command) == 0, "Patched runner stopped or failed; no automatic retry")
    check_stop_files(campaign)
    require((campaign / "hardware/completed.json").exists(), "Runner returned without completed.json; no scoring or summary")
    require(launch("score", [sys.executable, str(runner), "--mode", "score", "--output-dir", str(campaign)]) == 0,
            "Hardware scoring failed; no automatic retry")
    check_stop_files(campaign)
    audit_output = work / "final_hardware_audit.json"
    require(launch("audit_complete", [sys.executable, str(auditor), "--input-dir", str(campaign),
                                     "--require-complete", "--output", str(audit_output)]) == 0,
            "Complete hardware audit failed; no summary")
    audited = read(audit_output)
    require(audited.get("complete") is True and audited.get("status") == "passed_complete",
            "Auditor did not confirm complete hardware coverage")
    check_stop_files(campaign)
    require(launch("summarize", [sys.executable, str(summarizer), "--campaign-dir", str(campaign),
                                 "--output-dir", str(campaign / "analysis_complete")]) == 0,
            "Final analysis failed; no automatic retry")
    analysis = read(campaign / "analysis_complete/analysis.json")
    require(analysis.get("hardware", {}).get("status") == "complete_scored",
            "Final analysis did not include complete hardware scoring")
    checkpoint("complete", analysis_directory=str(campaign / "analysis_complete"),
               audit_file=str(audit_output), paper_edited=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=REPO / "results/quantum_heldout")
    parser.add_argument("--initial-pid", type=int, required=True)
    parser.add_argument("--expected-process-identity", required=True,
                        help="Exact stripped output from ps -p INITIAL_PID -o lstart=,comm=")
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-initial-job-id", required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.)
    args = parser.parse_args()
    if args.initial_pid <= 1 or not 1 <= args.poll_seconds <= 30:
        parser.error("Require a specific PID >1 and a polling interval between1 and30seconds")
    campaign = args.campaign_dir.resolve()
    work = campaign / "hardware/continuation"
    # Exclusive directory creation is the one-time controller lock. Never
    # remove it or create a second controller automatically after a failure.
    try:
        work.mkdir(exist_ok=False)
    except OSError as error:
        print(json.dumps({"phase": "stopped", "reason": f"Cannot claim one-time continuation directory: {type(error).__name__}"}), flush=True)
        return 1
    last_phase = None

    def checkpoint(phase, **details):
        nonlocal last_phase
        value = {"phase": phase, "updated_utc": datetime.now(timezone.utc).isoformat(),
                 "controller_pid": os.getpid(), "plan_sha256": args.expected_plan_sha256, **details}
        temporary = work / "status.json.tmp"
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(work / "status.json")
        with (work / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(value, allow_nan=False) + "\n")
        if phase != last_phase:
            print(json.dumps(value, allow_nan=False), flush=True)
            last_phase = phase

    def launch(label, command):
        check_stop_files(campaign)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPO / "src")
        with (work / f"{label}.log").open("x") as log:
            child = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
            checkpoint(label, child_pid=child.pid, log_file=str(work / f"{label}.log"))
            code = child.wait()
        checkpoint(f"{label}_returned", child_exit_code=code)
        return code

    try:
        with (work / "intent.json").open("x") as handle:
            json.dump({"created_utc": datetime.now(timezone.utc).isoformat(),
                       "initial_pid": args.initial_pid, "expected_process_identity": args.expected_process_identity,
                       "expected_initial_job_id": args.expected_initial_job_id, "plan_sha256": args.expected_plan_sha256,
                       "one_time": True, "retry_policy": "none", "paid_override": False}, handle, indent=2)
        validate_anchor(campaign, args.expected_plan_sha256, args.expected_initial_job_id, require_stage_zero=False)
        wait_for_initial_exit(args.initial_pid, args.expected_process_identity, campaign, checkpoint,
                              poll_seconds=args.poll_seconds)
        checkpoint("validating_initial_result")
        preflight = validate_anchor(campaign, args.expected_plan_sha256, args.expected_initial_job_id, require_stage_zero=True)
        with (work / "initial_stage_audit.json").open("x") as handle:
            json.dump(preflight, handle, indent=2, allow_nan=False)
        # Recheck absence immediately before the one permitted new runner.
        require(process_identity(args.initial_pid) is None, "Initial PID reappeared; refusing concurrent continuation")
        execute_pipeline(campaign, work, checkpoint, launch)
    except Exception as error:
        checkpoint("stopped", reason=str(error), error_type=type(error).__name__,
                   retry_allowed_automatically=False)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
