#!/usr/bin/env python3
"""Save IBM credentials with hidden input, outside the research repository."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import subprocess
import sys
import warnings


def hidden_input(prompt: str, dialog: bool) -> str:
    if dialog:
        if sys.platform != "darwin":
            raise RuntimeError("Native dialogs require macOS; omit --dialog")
        # The value only travels through this child's captured stdout to Python.
        # It is never included in a command line, log, or printed tool result.
        script = (
            'tell application "System Events"\n'
            'activate\n'
            f'display dialog "{prompt}" default answer "" with hidden answer '
            'with title "Q.NILM — IBM Quantum setup" '
            'buttons {"Cancel", "Save"} default button "Save"\n'
            'return text returned of result\nend tell'
        )
        result = subprocess.run(
            ["osascript", "-e", script], capture_output=True, text=True, check=False
        )
        if result.returncode:
            raise KeyboardInterrupt
        return result.stdout.strip()
    if not sys.stdin.isatty():
        raise RuntimeError("Run in an interactive terminal or use --dialog on macOS")
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        return getpass.getpass(prompt + ": ").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialog", action="store_true", help="Use private macOS dialogs")
    parser.add_argument("--status", action="store_true", help="Check presence; never print credentials")
    parser.add_argument("--replace", action="store_true", help="Replace the named qnilm account")
    args = parser.parse_args()
    from qiskit_ibm_runtime import QiskitRuntimeService

    account_file = Path.home() / ".qiskit" / "qiskit-ibm.json"
    configured = account_file.is_file() and "qnilm" in QiskitRuntimeService.saved_accounts()
    if args.status:
        print("Q.NILM IBM credentials: " + ("configured" if configured else "not configured"))
        return
    if configured and not args.replace:
        print("Q.NILM IBM credentials already configured. Use --replace to update.")
        return

    token = hidden_input("Paste your IBM Quantum API key", args.dialog)
    if not token or any(character.isspace() for character in token):
        raise ValueError("An API key without whitespace is required")
    instance = hidden_input(
        "Paste your instance CRN (or leave empty for Open Plan selection)", args.dialog
    )
    if instance and not instance.startswith("crn:"):
        raise ValueError("Expected an instance CRN beginning with crn:")

    # Use IBM's standard account store, outside the repository. The named account
    # does not replace any other account or change an existing default account.
    account_dir = Path.home() / ".qiskit"
    account_file = account_dir / "qiskit-ibm.json"
    if account_dir.is_symlink() or account_file.is_symlink():
        raise RuntimeError("Refusing to save credentials through a symbolic link")
    previous_umask = os.umask(0o077)
    try:
        account_dir.mkdir(mode=0o700, exist_ok=True)
        QiskitRuntimeService.save_account(
            token=token,
            channel="ibm_quantum_platform",
            instance=instance or "auto",
            plans_preference=["open"],
            name="qnilm",
            overwrite=args.replace,
        )
        account_file.chmod(0o600)
    finally:
        os.umask(previous_umask)
        token = ""
    print("IBM credentials saved privately in the standard Qiskit account store.")
    print("Account name: qnilm. No quantum job was submitted.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Setup cancelled; no new credentials saved.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        # Provider exception messages can contain request details. Never echo them.
        print(f"Setup failed ({type(exc).__name__}). Check the inputs and retry.", file=sys.stderr)
        sys.exit(1)
