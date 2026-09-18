# Q.NILM IBM Quantum pilot

This workflow executes the paper's gate-model QAOA circuit through Qiskit and
IBM Quantum Compute. The first experiment freezes the original nine-variable,
depth-one R1Hz pilot. The circuit builder also supports greater QAOA depths.

## Experimental scope

The source `results/pilot/summary.json` contains the selected aggregate values,
three appliance powers, switching penalties, segment durations, proxy labels,
and classically optimized QAOA angles. The IBM runner reconstructs that exact
QUBO and verifies its phase scale. Before any execution it compares Qiskit's
statevector probabilities against the independent NumPy implementation, with
a maximum absolute error of `1e-12`.

This is a diagnostic selected interval with labels derived from circuit
channels. Its aggregate is the sum of selected circuit channels, not the
whole-house mains. A strong result here validates circuit implementation; it
does not establish held-out NILM accuracy or a quantum advantage. The QPU
samples frozen, classically optimized parameters. It does not perform a
hardware-in-the-loop variational optimization in this pilot.

## Setup

Use a Python 3.12 environment and install the optional dependencies:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[ibm]'
.venv/bin/python scripts/configure_ibm_quantum.py
```

The setup asks for an API key and optional instance CRN with hidden input.
On macOS, `--dialog` uses private native dialogs. The key never appears in shell
history, command-line arguments, printed results, or Git. IBM's `save_account`
stores the named `qnilm` account in `~/.qiskit/qiskit-ibm.json`, with file mode
`0600`. This is a local plaintext credential store protected by filesystem
permissions; it is not a keychain. Other named accounts are preserved.
`--status` prints only whether the named account exists; `--replace` updates it.

Leaving the CRN empty restricts automatic instance selection to Open Plan.
An explicit CRN overrides IBM's plan filter; submission therefore checks the
actual active instance plan again. A non-Open-Plan job requires `--allow-paid`.

IBM reference: [save credentials](https://quantum.cloud.ibm.com/docs/en/guides/save-credentials).

## Local tests

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/run_ibm_pilot.py --mode ideal --output-dir results/ibm_ideal_new
.venv/bin/python scripts/run_ibm_pilot.py --mode noisy --backend FakeTorino --output-dir results/ibm_noisy_new
```

Both modes use 4,096 shots and seed 7 by default. Noise mode uses the installed
IBM fake backend's archived calibration snapshot and Aer simulation. Its
results are simulations, not measurements of the current QPU. The snapshot
properties, software versions, transpilation seed, physical layout, gate
counts, and compilation depth are recorded. Local testing does not require an
IBM account. Simulated noise omits some physical error mechanisms.

IBM reference: [local testing mode](https://quantum.cloud.ibm.com/docs/en/guides/local-testing-mode).

## Hardware preparation and execution

```sh
.venv/bin/python scripts/run_ibm_pilot.py --mode prepare --output-dir results/ibm_qpu_new
```

Preparation authenticates and selects an operational backend with sufficient
qubits using IBM's least-busy selection, or accepts `--backend NAME`. It
compiles the circuit and saves a frozen `plan.json`, OpenQASM and QPY circuit
files, the source pilot summary, live backend configuration and calibration
properties, and SHA-256 hashes. It submits no quantum job.

Review the plan, then submit that exact prepared circuit:

```sh
.venv/bin/python scripts/run_ibm_pilot.py --mode submit --output-dir results/ibm_qpu_new
```

Submission sends one SamplerV2 job in job mode, without a session. The default
request is 4,096 shots with `max_execution_time=60` seconds. This limit is a
QPU execution cap, not a queue or end-to-end timeout. Dynamical decoupling and
gate/measurement twirling are explicitly disabled for the first baseline.
No monetary charge is inferred from QPU seconds; paid-plan costs depend on
the account's terms and must be obtained from billing.

The job identifier is saved immediately. A guard file is created before the
submission call, so an interrupted or ambiguous submission cannot silently
create a second job on rerun. If the guard lacks a job identifier, reconcile
the `qnilm` tagged workload on IBM's dashboard before attempting another job.

Retrieve results without resubmitting:

```sh
.venv/bin/python scripts/run_ibm_pilot.py --mode collect --output-dir results/ibm_qpu_new
```

If the job is queued or running, this reports status and exits. Repeating
collection retrieves the completed job's counts, serialized primitive result,
execution metrics, and scoring summary. Each independent repeat should use a
new output directory and hardware preparation, so its calibration and circuit
provenance remain inspectable.

IBM references: [Sampler inputs and outputs](https://quantum.cloud.ibm.com/docs/en/guides/sampler-input-output),
[maximum execution time](https://quantum.cloud.ibm.com/docs/en/guides/max-execution-time).

## Metrics and interpretation

`summary.json` reports best sampled and mean objective, exact optimum hit
probability, gap of the best sampled objective from the exact minimum,
reference-state accuracy, precision, recall, F1, MCC, and duration-weighted
aggregate reconstruction MAE. Reference metrics describe the lowest-cost
observed state and use one vote per appliance-segment bit. They are not mean
per-shot accuracy. Both mean objective and optimum hit rate should accompany
the best-sample result, which depends on the shot budget.

Qiskit count strings put variable zero on the right; decoding reverses them
into Q.NILM's segment-major order. All degenerate exact minimizers contribute
to optimum probability using an explicit absolute tolerance and zero relative
tolerance. Bitwise agreement against one representative exact minimizer is
reported separately. The pilot runner caps exact validation at 12 qubits.

`sampler_wait_wall_time_s` measures the local execution wait for simulations
and result retrieval for hardware. Hardware queue/initialization and server
running durations, when available, come from IBM timestamps. Raw IBM job
metrics preserve QPU usage and execution details for later analysis.

The manuscript's statistical comparison can be regenerated locally with:

```sh
PYTHONPATH=src .venv/bin/python scripts/analyze_ibm_pilot.py
```

This verifies the saved metrics and writes `results/ibm_pilot_statistics.json`,
including descriptive within-run Wilson intervals, ideal-QAOA expectations,
and the analytical uniform-sampling control. It makes no IBM calls. The
planned confirmatory campaign is specified in `docs/experiment_protocol.md`.
