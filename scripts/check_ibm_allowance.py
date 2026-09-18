#!/usr/bin/env python3
"""Read only non-secret plan and QPU allowance fields; never submit jobs."""
import json
import sys
from qiskit_ibm_runtime import QiskitRuntimeService


def main():
    try:
        service = QiskitRuntimeService(name="qnilm")
        current = service.active_instance()
        plan = next((x.get("plan") for x in service.instances() if x["crn"] == current), None)
        usage = service.usage()
        fields = ("usage_limit_seconds", "usage_allocation_seconds", "usage_consumed_seconds",
                  "usage_remaining_seconds", "usage_limit_reached")
        print(json.dumps({"plan": plan, "usage": {k: usage.get(k) for k in fields}}, indent=2))
    except Exception as error:
        # Avoid printing credential-bearing request details or account identifiers.
        print(json.dumps({"status": "allowance check failed", "error_type": type(error).__name__}))
        sys.exit(1)


if __name__ == "__main__":
    main()
