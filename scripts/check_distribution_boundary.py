#!/usr/bin/env python3
"""Reject author-only publication material anywhere in reachable Git history."""
import argparse
from pathlib import PurePosixPath
import subprocess


def forbidden(path):
    path = path.lower()
    parts = PurePosixPath(path).parts
    return (
        bool(parts) and parts[0] in {"paper", "output", "private-publication"}
        or "proposal" in path
        or "cover-letter" in path
        or "submission_guide" in path
        or "submission_checklist" in path
        or PurePosixPath(path).suffix in {".pdf", ".tex", ".bib", ".cls", ".sty", ".docx", ".zip"}
        or any(p in {".qiskit", ".secrets", ".venv"} for p in parts)
        or PurePosixPath(path).name in {".env", "qiskit-ibm.json"}
    )


def check(history=True):
    commits = subprocess.check_output(["git", "rev-list", "--all"], text=True).splitlines() if history else ["HEAD"]
    violations = set()
    for commit in commits:
        paths = subprocess.check_output(["git", "ls-tree", "-rz", "--name-only", commit]).decode().split("\0")
        violations.update(p for p in paths if p and forbidden(p))
    if violations:
        raise SystemExit("Distribution boundary failed; author-only paths in history: " + ", ".join(sorted(violations)))
    print(f"Distribution boundary passed across {len(commits)} reachable commits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-only", action="store_true")
    check(history=not parser.parse_args().head_only)
