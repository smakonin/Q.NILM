# Q.NILM

[![tests](https://github.com/smakonin/Q.NILM/actions/workflows/tests.yml/badge.svg)](https://github.com/smakonin/Q.NILM/actions/workflows/tests.yml)

Q.NILM means Quantum Non-Intrusive Load Monitoring. This repository is the
canonical reproducible research implementation for the working paper:

> **Q.NILM: Benchmarking Circuit Feasibility and Objective Alignment in
> Quantum Non-Intrusive Load Monitoring**

Repository: <https://github.com/smakonin/Q.NILM>

Target journal: **IEEE Transactions on Quantum Engineering (TQE)**.
This repository distributes research code and experimental evidence only.
Manuscript files, journal-submission materials and the D-Wave proposal are
not included in the repository or research release.

## TQE submission release — 18 September 2026

This is a quantum-engineering benchmark and diagnostic study, **not a
demonstration of quantum accuracy or speed advantage**. The research
release contains implementation code, tests, experimental protocols and
checksum-verified research archives. See the
[versioned release](https://github.com/smakonin/Q.NILM/releases/tag/tqe-submission-2026-09-18).
No journal submission or acceptance is implied by this release.
Public availability is planned immediately before journal submission.
Until that visibility change is completed and anonymous access is verified,
this remains a private preparation repository, not a publicly accessible release.

The September 18 independent-date repeat returned all 24 component circuits
and 24,576 shots (9 seconds of finalized free QPU usage). At 24 logical
qubits, preparation-only feasibility increased from 19.336% to 44.368%,
but full-circuit feasibility remained below 1% (0.456% to 0.749%). This is
a component repeat, **not IBM full-accuracy Run 2**. Table VI retains Run 1.

The revision incorporates the physical controls, six-qubit simulations,
separately reserved static mean/CVaR tests and development-only background
study. Their populations and information access are not interchangeable.
Complete Balletti and high-frequency Li pipelines are outside the declared
matched-objective scope; no superiority over them is claimed. Large
experimental records are distributed in the release assets, not duplicated
in Git history; extract them at the repository root to restore `results/`.
See [release reproduction instructions](docs/reproducing_release.md) for
installation, checksum verification and offline audits.
Earlier sections below preserve historical experiment and reproduction notes.

## Current scope

The present code combines quantum circuit-validation experiments, a first
held-out classical NILM evaluation, and a full-coverage ideal-QAOA extension
on the same, now-examined test periods. It:

- builds a duration-weighted temporal NILM QUBO;
- converts it exactly to an Ising Hamiltonian;
- simulates a depth-one QAOA circuit without a quantum SDK;
- exports the optimized circuit as OpenQASM 3;
- checks the quantum sample against exact enumeration;
- event-compresses regular samples before qubit allocation; and
- uses appliance-normalized switching penalties;
- fits binary/multistate appliance and background models using training data only; and
- compares exact temporal optimization and exact MAP factorial HMM inference
  on initially untouched, timestamp-selected mains windows; and
- evaluates explicit one-hot/XY-mixer QAOA on all 30 original test windows,
  with matched uniform sampling and exact classical controls.

The same QUBO can also be submitted through the D-Wave Ocean adapter to local
simulated annealing and tabu samplers, a direct Leap QPU, or a Leap hybrid
solver. D-Wave quantum annealing samples the shared QUBO/Ising objective; it
does **not** execute the gate-model QAOA circuit. Direct QPU and hybrid results
must therefore be identified separately.

The optional IBM integration builds the same gate-model QAOA circuit in Qiskit,
checks it against the independent NumPy simulator, and supports local ideal and
IBM snapshot noise simulations plus IBM Quantum hardware execution. Both the
original and corrected nine-qubit diagnostic checks have completed on `ibm_fez`.
The separate 12/24-encoded-qubit full-coverage hardware campaign is complete:
all 30 windows and 86,396 valid blocks, 315.686 W macro appliance MAE including
the predeclared fallback. Only 1.398% of raw shots were feasible. This is a
negative hardware result, not a quantum-advantage claim.

It does not claim quantum advantage. A sampled optimum on a small instance is
a circuit-validation result only.

## Install

~~~bash
python3 -m pip install -e .
~~~

Install the optional D-Wave Ocean integration with:

~~~bash
python3 -m pip install -e '.[dwave]'
~~~

For IBM Quantum and local Qiskit simulation, use Python 3.12 or later in a
virtual environment:

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[ibm]'
~~~

## Reproduce the first R1Hz pilot

The local source is read-only at
/Users/stephen/Documents/Research/Datasets/R1Hz.

~~~bash
python3 scripts/extract_r1hz_window.py \
  --start "2018-01-28 12:00:00" \
  --seconds 3600 \
  --output data/derived/r1hz_2018-01-28_1200.csv

PYTHONPATH=src python3 scripts/run_pilot.py \
  --input data/derived/r1hz_2018-01-28_1200.csv \
  --output-dir results/pilot_reproduced \
  --event-threshold-ratio 0.01 --aggregate-mode legacy-clipped

PYTHONPATH=src python3 scripts/run_synthetic_benchmark.py \
  --output-dir results/synthetic

PYTHONPATH=src python3 scripts/run_ocean_scaling_benchmark.py \
  --output-dir results/dwave_scaling_reproduced \
  --r1hz-input data/derived/r1hz_2018-01-28_1200.csv \
  --aggregate-mode legacy-clipped

PYTHONPATH=src python3 -m unittest discover -s tests -v
~~~

Rows marked s (synthetic recovery) are excluded from this event-timing pilot.
The bounded derived CSV is not committed; regenerate it from the canonical
dataset DOI [10.7910/DVN/RCB5VJ](https://doi.org/10.7910/DVN/RCB5VJ).

## Controlled categorical simulation (Stage B)

The [frozen Stage B protocol](docs/stage_b_protocol.md) expands the historical
binary sweep with categorical circuits, separately optimized depths one and
two, matched uniform sampling, measurement-noise variants, controlled gate/
readout noise, four shot budgets and repeated sampling. It uses 4--24 encoded
qubits under ideal simulation and exact noisy simulation through 10 qubits.
All tests run locally; no IBM credentials or QPU allowance are used.

```sh
python -m pip install -e '.[stage-b]'
python scripts/run_stage_b.py --mode prepare --output results/stage_b/new_run
python scripts/run_stage_b.py --mode run --output results/stage_b/new_run
python scripts/run_stage_b.py --mode summarize --output results/stage_b/new_run
```

The versioned archive is under `results/stage_b/run_002/`; its protocol
and source hashes were frozen before scored execution. The incomplete
`run_001/` is retained with its failure record and original source: a strict
optimizer check rejected a machine-roundoff boundary excursion. Version 002
repeats the same scientific design after a narrow numerical correction.
No earlier NILM, binary-QAOA or IBM results are replaced. These simulated
comparisons do not establish quantum advantage or calibrated hardware accuracy.
The completed [results report](results/stage_b/run_002/report.md) covers all
144 inputs, 288 optimizations, 720 distributions and 92,160 trials. The
[independent audit](results/stage_b/run_002/independent_audit.json) reconstructs
all 62,668,800 simulated draws and all 2,286 bootstrap intervals.

```sh
python scripts/audit_stage_b.py --help
python scripts/summarize_stage_b_report.py --help
```

## Representation correction and exact temporal baseline

New R1Hz runs default to **signed** baseline centering:
`aggregate = sum(channel powers) - sum(fitted baselines)`. The original
pilot clipped each centered channel to zero before averaging, introducing
an upward bias. `--aggregate-mode legacy-clipped` is provided only to
reproduce the frozen pilot and Ocean results. Do not overwrite those records.

With the same three pilot intervals, powers and penalties, signed centering
changes exact proxy agreement from 8/9 to 9/9. This is a retrospective,
same-hour diagnostic using interval-majority proxy labels, not held-out
NILM accuracy or a new quantum-hardware result.
Across all 120 blocks, raw selected-total MAE improves from 262.62 to 243.88 W,
but proxy agreement declines from 316/360 to 293/360. This mixed result exposes
remaining binary-model/proxy mismatch; the selected 9/9 result must not be
presented alone as a general accuracy improvement.

~~~bash
PYTHONPATH=src python3 scripts/run_pilot.py \
  --input data/derived/r1hz_2018-01-28_1200.csv \
  --output-dir results/pilot_corrected_new --event-threshold-ratio 0.01

PYTHONPATH=src python3 scripts/run_representation_ablation.py \
  --output-dir results/representation_correction_new

PYTHONPATH=src python3 scripts/run_temporal_scaling_benchmark.py \
  --output-dir results/temporal_scaling_new

PYTHONPATH=src python3 scripts/run_ibm_pilot.py --mode ideal \
  --pilot-summary results/pilot_corrected_new/summary.json \
  --output-dir results/ibm_corrected_ideal_new
~~~

`quantum_nilm.classical.solve_binary_temporal_exact` solves the binary temporal
objective using dynamic programming in O(K N 2^N) time. For three appliances,
there are only eight joint states per interval; increasing the interval count
alone does not create exponential classical difficulty. It is an essential
baseline for QAOA and annealing comparisons. The code currently caps exact
optimization at 20 appliances and 256 MiB of traceback memory.

See the separate correction and temporal-scaling artifacts under `results/`.
The original IBM counts remain tied to their frozen, legacy objective.

## First held-out classical campaign: completed

The internally frozen campaign selects 150 disjoint 24-hour windows using
timestamps only: 90 training, 30 validation, and 30 test windows inside the
chronological 60/20/20 R1Hz partitions. These are not necessarily calendar days
or consecutive windows. There are 259,196 / 86,396 / 86,396 valid 30-second
blocks respectively; each partition loses four blocks after excluding `s`/`+`
rows. Power levels and FHMM probabilities use training only; temporal penalties
are selected on validation and frozen before test reads. No test window is
replaced because of activity, quality, or results.

The fourteen model/input comparisons cover seven methods on actual mains
and on a separately labelled selected-circuit-total control. Mains inference
includes a three-state background model learned from training-only mains minus
the selected circuits. Test inference does not see test appliance labels.

| Method | Mains macro appliance MAE (W) | Selected-circuit control MAE (W) |
|---|---:|---:|
| Binary, regular blocks | 35.208 | 29.211 |
| Binary, event-compressed | 35.860 | 29.726 |
| Binary, exact MAP FHMM | 32.710 | 29.234 |
| Multistate, regular blocks | 24.729 | 9.961 |
| Multistate, event-compressed | 26.041 | 12.339 |
| Multistate, exact MAP FHMM | 26.950 | 10.743 |
| Training low-power constant | 34.093 | 34.093 |

On actual mains, regular multistate inference reduces macro appliance MAE
from 35.208 to 24.729 W, a **29.76% classical modeling improvement**. The paired
multistate-minus-binary difference is -10.478 W, with a descriptive 95%
24-hour-window bootstrap interval of [-11.725, -9.214] W. This is not quantum
advantage, independent preregistration, or cross-home validation.

The improvement is not universal: vacuum mains MAE is 13.393 W for the regular
multistate model versus 2.198 W for the low-power constant. Only three test
windows contain vacuum readings above the fixed high-state proxy threshold.
These thresholds are not physical ON labels: the fridge threshold is 244.903 W
and excludes its fitted 101.756 W middle state. Continuous appliance-power MAE
on the original valid blocks is primary; compressed predictions are expanded
before scoring.

See the immutable [protocol](results/heldout_campaign/protocol.json),
[frozen models](results/heldout_campaign/frozen_models.json),
[full results](results/heldout_campaign/summary.md), and
[post-run interpretation notes](results/heldout_campaign/interpretation_notes.md).
The implementation includes automated algebra, inference, data-integrity, and hardware-safety tests. To reproduce into a new directory:

~~~bash
PYTHONPATH=src python3 scripts/run_heldout_campaign.py \
  --source /path/to/R1Hz/recovered/power.csv \
  --output-dir results/heldout_campaign_reproduced
~~~

The published test set has now been examined. Further model tuning requires a
new design and untouched evidence, not relabelling these windows as unseen.

## Full-coverage categorical QAOA extension: ideal simulation completed

This frozen **post-exposure extension** reuses all 86,396 valid 30-second
blocks in the original 30 test windows; it is not a newly blind test. The
appliance state counts remain [4, 3, 2], with three signed background states.
Aggregate-only compression yields 4,881 intervals across 34 gap-separated
runs. Each method processes at most two intervals per circuit and carries its
own previous inferred state, resetting at gaps and window boundaries. No
exact-classical state or test appliance label initializes QAOA.

The depth-one circuit initializes uniform one-hot registers, applies the
coefficient-scaled cost, and mixes with ordered adjacent-pair XY rotations.
One/two intervals use 12/24 encoded qubits and 72/5,184 feasible states;
hardware routing can require additional active physical qubits. A single
angle pair, gamma=1.5707963268 and beta=1.1087974071, was selected using
normalized expected objective on 90 training-only chunks, not test appliance
MAE. The exact simulator operates in the feasible subspace and is independently
checked against the full-qubit circuit.

| Method | Macro appliance MAE (W) |
|---|---:|
| Full-run exact DP, compressed | 26.041 |
| Matched two-interval exact optimization | 26.012 |
| Uniform feasible sampling | 41.537 |
| Ideal QAOA simulation | 32.091 |

Each stochastic seed covers 2,451 circuits: 2,430 two-interval circuits and
21 single-interval tails, with 256 shots each. Three fixed seeds give
1,882,368 shots per stochastic method. Errors are averaged across seeds
within each window before pooling; seeds do not triple the test sample size.
QAOA improves on uniform sampling but remains worse than both exact classical
controls. This is **not quantum advantage**. The paired QAOA-minus-exact-chunk
MAE difference is 6.079 W, descriptive 95% window-bootstrap interval
[4.322, 7.945] W. The full
[simulation analysis](results/quantum_heldout/simulation/summary.md) retains
per-appliance errors, all comparisons, and audit limitations.

Reproduce the extension using a new directory and a verified original campaign:

~~~bash
PYTHONPATH=src .venv/bin/python scripts/run_quantum_heldout.py --mode freeze \
  --original-dir results/heldout_campaign --output-dir results/quantum_heldout_reproduced
PYTHONPATH=src .venv/bin/python scripts/run_quantum_heldout.py --mode prepare \
  --output-dir results/quantum_heldout_reproduced
PYTHONPATH=src .venv/bin/python scripts/run_quantum_heldout.py --mode train \
  --output-dir results/quantum_heldout_reproduced
PYTHONPATH=src .venv/bin/python scripts/run_quantum_heldout.py --mode simulate \
  --output-dir results/quantum_heldout_reproduced
PYTHONPATH=src .venv/bin/python scripts/summarize_quantum_heldout.py \
  --campaign-dir results/quantum_heldout_reproduced \
  --output-dir results/quantum_heldout_reproduced/simulation_analysis
~~~

These modes do not submit hardware jobs. Preserve the original protocol,
angles, prediction traces, counts, summaries, and first-pilot records; output
guards reject overwriting existing experiment artifacts.

### Separate full-coverage IBM campaign: complete and audited

The [prepared hardware plan](results/quantum_heldout/hardware/plan.json)
was executed in full: 2,451 circuits, 129 adaptive jobs, 256 shots per
circuit and 627,456 shots. The first job, `dajnp09hvn6c73cul07g`, was submitted
at 2026-09-14 04:48:01 UTC; the campaign completed at 06:43:04 UTC.
Accounted QPU usage was 437 seconds, within the 540-second cap; 157 seconds
of free allowance remained at completion (a historical snapshot, not a live
allowance query). Queue and end-to-end time are not QPU usage.

The [complete analysis](results/quantum_heldout/analysis_complete/summary.md)
reports hardware MAE of 324.017 W (dryer), 140.646 W (fridge), 482.396 W
(vacuum), and 315.686 W macro, versus 32.091 W ideal QAOA and 26.012 W
matched exact classical inference. The hardware row includes fallback in
131 chunks / 4,310 blocks (4.989%). Only 8,770 shots were feasible.
The original windows were already examined; this is a post-exposure
extension, not a fresh blind confirmation or independent hardware replication.

The saved one-time continuation controller is recorded under
`results/quantum_heldout/hardware/continuation/`. Consult its `status.json`
before any manual recovery: do not launch a concurrent runner. Its final
status is `complete`; scoring, independent auditing and final analysis
all exited successfully. The controller did not modify the manuscript or
silently retry a failed job. This archive must not be reused for new diagnostics.

For a separately approved hardware run on a freshly reproduced campaign:

~~~bash
PYTHONPATH=src .venv/bin/python scripts/run_quantum_heldout_ibm.py --mode prepare \
  --output-dir results/quantum_heldout_reproduced
# Only the next mode submits QPU jobs; it also recovers existing job IDs.
PYTHONPATH=src .venv/bin/python scripts/run_quantum_heldout_ibm.py --mode run \
  --output-dir results/quantum_heldout_reproduced
# Requires every frozen hardware stage to have completed.
PYTHONPATH=src .venv/bin/python scripts/run_quantum_heldout_ibm.py --mode score \
  --output-dir results/quantum_heldout_reproduced
~~~

The full-campaign runner permits **Open Plan only**, with no paid override.
Frozen submission intents prevent blind duplicate jobs. Every infeasible shot
is charged and reported; an all-invalid chunk uses the declared own-prior
carry, or lowest-power states at a reset, and counts this fallback explicitly.
No complete hardware MAE may be inferred from partial coverage. Multi-date
replication and the penalty-X versus XY mixer comparison remain separate work.

Independent offline audit tools verify the complete saved simulation traces
(`scripts/audit_quantum_heldout_traces.py`) and completed hardware records
(`scripts/audit_quantum_heldout_hardware.py --require-complete`). The first
audit passed all 210 traces, 17,157 chunk decisions, and 3,764,736 stochastic
draws. The complete hardware audit passed all 129 Runtime records, raw counts,
adaptive priors, direct objectives, parameter bindings, coverage and usage.
It deliberately rejects incomplete coverage. Original records are never overwritten.

## D-Wave execution

The committed scaling study is a classical Ocean readiness benchmark. It
records raw sample sets, logical graph sizes, objective values, reference-state
metrics, exact certificates where tractable, wall-clock time, and deterministic
minor-embedding estimates on an ideal defect-free Zephyr graph. The embedding
estimate is not a live-hardware result. The study is not a QPU result or a
quantum-advantage claim. A SHA-256 manifest covers every published result file.

After configuring Leap credentials locally, run the instances on a direct
quantum annealer or a hybrid solver. These commands use corrected signed
preprocessing; add `--aggregate-mode legacy-clipped` only for a deliberately
matched historical comparison:

~~~bash
PYTHONPATH=src python3 scripts/run_ocean_scaling_benchmark.py \
  --output-dir results/dwave_qpu \
  --r1hz-input data/derived/r1hz_2018-01-28_1200.csv \
  --samplers qpu --num-reads 1000

PYTHONPATH=src python3 scripts/run_ocean_scaling_benchmark.py \
  --output-dir results/dwave_hybrid \
  --r1hz-input data/derived/r1hz_2018-01-28_1200.csv \
  --samplers hybrid
~~~

Credentials are never stored in this repository. QPU outputs include embedding,
chain-break, backend, and timing metadata when Ocean supplies them. Use
`--solver`, `--annealing-time-us`, `--chain-strength`, and
`--hybrid-time-limit-s` to freeze an experimental configuration.

## IBM Quantum execution

The [IBM pilot guide](docs/ibm_quantum_pilot.md) documents secure setup, local
validation, hardware preparation, submission, and result recovery. The first
IBM experiment reuses the frozen nine-qubit R1Hz pilot parameters and objective
already in `results/pilot/summary.json`; it requires no additional dataset copy.

The separate corrected check, job `dajn53r9k43c73ahdmag`, also completed on
`ibm_fez`: the optimum appeared in 75/4,096 shots and the best state matched
9/9 diagnostic proxy bits. IBM reports 3 seconds of accounted QPU usage, not
the whole workflow time. It remains a same-hour, selected-circuit diagnostic,
not held-out QPU accuracy or a speed advantage. Its
[hardware record](results/ibm_corrected_qpu_check/summary.json) is separate from
the original legacy-objective run.

Twelve corrected ideal/archived-Fez-noise control runs, each with 4,096 shots,
are complete in [the local control archive](results/ibm_followup_controls/run_001/summary.json).
These controls are circuit-matched but not calibration-matched to the new
QPU job: 27 active calibration parameters changed between the snapshots.
Depth-two controls use an unoptimized equal-angle split. The
[IBM follow-up plan](docs/ibm_followup_plan.md) distinguishes these completed
checks from multi-date replication, broader size/depth tests, and the
penalty-X versus XY mixer comparison, which remain planned and unapproved.
The separately completed full-coverage categorical campaign is described above.
Component-level and effective-shot diagnostics are stored separately under
`results/quantum_diagnostics/` and must not be mistaken for new blind NILM tests.
The completed [conditional usable-shot replay](results/quantum_diagnostics/effective_shots/summary.md)
assigns each chunk its observed hardware feasible count. Ideal QAOA macro
MAE rises from 32.091 W at 256 usable shots to 199.052 W; matched-budget
uniform sampling gives 387.977 W, versus hardware's 315.686 W. All six
replays retain complete coverage and declared zero-budget fallback. This
supports sensitivity to candidate scarcity, not a causal allocation of
hardware errors or a quantum-advantage claim.

The [IBM experiment register](docs/ibm_run_registry.md) preserves **Run 1**
as the complete NILM evaluation and keeps selected-circuit pilots and component
diagnostics separate. Future full campaigns receive new numbered rows rather
than replacing Run 1.

The completed component diagnostic uses 24 circuits × 1,024 shots:
preparation alone, preparation plus mixer, preparation plus cost, and full
QAOA, at 12/24 encoded qubits on three frozen training examples. All eight
compiled templates pass the saved noiseless probability audit. The unchanged
retry's `results/quantum_diagnostics/ladder_retry_001/result.json` establishes
collection of all 24,576 measurements and 9 seconds of finalized QPU charge.
Independent compiled-ideal and raw-count/objective audits both passed.
No appliance MAE is assigned
to this component study. The runner permits only the free Open Plan,
requires at least 80 seconds remaining, limits execution to 50 seconds,
budgets another 10 seconds for overhead, and refuses duplicate submission.
IBM accounting overhead means 60 seconds is a checked campaign budget,
not a guaranteed billing ceiling. No automatic second job or paid override is allowed.
The unsubmitted earlier preparation is retained in `ladder_preflight_001/`.
The submitted ladder job subsequently failed with IBM Runtime error 9701
(temporary internal error). No measurements were returned; the independent
status receipt is `results/quantum_diagnostics/ladder/failure_status.json`.
After explicit user approval, one unchanged retry completed in the separate
`ladder_retry_001/` archive. The original failure does not change the completed
full IBM Run 1. Its observed zero charge was provisional while accounting was
pending; the retry's finalized 9 seconds does not settle that earlier record.
The 12-qubit preparation/mixer/cost/full-circuit feasible fractions are
74.056%/69.759%/25.911%/24.154%; the 24-qubit fractions are
19.336%/13.509%/1.074%/0.456%. Every denominator includes all raw shots.
This selected-training-example diagnostic is not a full NILM campaign or an
independent-date repetition.

Read the saved retry result and its two audits for completed evidence.
Collection refuses to overwrite `result.json`; do not rerun submission to
recover a queued, interrupted or already completed job.

### Physical follow-up: all three diagnostic groups complete

The [physical diagnostic protocol](docs/physical_diagnostics_protocol.md)
was frozen before one free-plan job on `ibm_fez`,
`dak582ni3e6s738r26a0`. All **228 circuits / 94,208 shots** were collected;
finalized QPU usage was **28 seconds**. The separate
[`physical_001` archive](results/quantum_diagnostics/physical_001/)
contains the compiled circuits, ideal audit, raw results, analysis and
independent result audit. All 379 software tests passed before submission.

- Local W-state preparation yielded 91.504% valid samples on qubits 131–133
  versus 95.475% on 141–143. The earlier strong 132/133 marginal bias did not
  recur in these smaller local circuits; this does not rule out context or
  calibration dependence.
- With the compiled native gate sequence preserved, zero versus nominal cost
  angles gave 29.167% versus 28.711% validity at 12 logical qubits and 1.270%
  versus 1.270% at 24. Zeroing the angles did not restore feasibility.
- The phase-sensitive probe found a substantial deviation after repeated CZ
  gates on 132–133, while the short interleaved randomized benchmark could not
  resolve an average gate-error difference between edges. A phase-error
  signature is not yet a uniquely identified faulty gate: duration-matched
  idle controls and independent-calibration repetitions remain unperformed.

See the [completed results and uncertainty](docs/ibm_run_registry.md#physical-diagnostic-1-completed-results).
No shots were postselected, no automatic retry or paid override was used,
and no new NILM MAE or quantum advantage is claimed. The completed Table VI
IBM Run 1 is preserved; these diagnostics are not Run 2.

### Six-qubit offline investigation: all three steps complete

The [six-qubit diagnostic results](docs/six_qubit_diagnostic_results.md) cover
exact circuit verification, isolated noise mechanisms, and controlled increases
in gate count/depth/width. The separate
[`six_qubit_001` archive](results/quantum_diagnostics/six_qubit_001/) contains
30 ideal circuits, 12 phase controls, 106 exact noisy simulations and an
independent audit. All 397 software tests passed. No hardware job was submitted.

Ideal feasibility is 100%; the archived-calibration model predicts 72.264%
and 79.590% validity for the six-qubit full circuit on two fixed Fez layouts.
Same-ideal CZ folding reduces these to 43.868% and 49.992% at five times the
CZ count. Logical phase distortion changes solution probabilities without
breaking ideal one-hot feasibility; interleaved native errors can break it.
With that experiment's original angle bounds, objective-trained p=1 lowers mean cost but does not improve the
exact-solution probability over uniform sampling on this nine-assignment task.
These are synthetic model results, not recovered held-out MAE, a uniquely
identified faulty IBM gate, or quantum advantage. The paper is unchanged.

See the results document for the frozen protocol, full noise/size tables,
limitations and safe reproduction in a fresh directory.

### Matched-budget training-loss comparison

The [completed loss comparison](docs/loss_comparison_results.md) adds 75 paired
six-qubit synthetic fits: mean, CVaR25/50 and expected best-of-4/16, with exact
training, 256-shot ideal training, and a wider-angle control. Every fit receives
1,024 evaluations. The independent audit reproduced 76,800 losses, 6,553,600
simulated training measurements and 78 noisy distributions; all 408 tests pass.

Tail losses improve the original-bound decoder, but wider bounds also greatly
improve mean training. Finite-shot best-of-16 selection can favor lucky zero-loss
estimates, and training-case gains do not consistently transfer. Noise-model
validity remains essentially unchanged between mean and best-of-16. The results
document also qualifies the routed zero-angle uniform control, which is not a
minimal physical W-only circuit. Production parameters, the paper, prior IBM
results and hardware accounts remain unchanged. No quantum advantage is claimed.

The subsequent [Steps 1 and 2 validation](docs/training_validation_results.md)
is complete: 80 synthetic fits, 20 R1Hz fits, ten new validation days and twenty
new test days outside the earlier recorded windows. CVaR50 improves the wider-
angle finite-shot synthetic case but does not improve the new R1Hz test MAE
over mean loss (38.445 versus 38.055 W at 16 shots). Exact mains optimization
gives 25.390 W versus a 5.919 W label-only representation floor, motivating
objective/background investigation. The fixed test cohort has no active vacuum
examples. All 420 tests and the independent source/measurement/circuit audit
passed. These are static single-interval diagnostics, not replacement temporal
or IBM results; no hardware job or paper update was performed.

The [background/objective investigation](docs/background_objective_results.md)
uses the original ninety training days and only the ten existing development
validation days. Thirty exact classical candidates show that finer background
states alone improve mains reconstruction while worsening appliance MAE.
Training-derived priors reduce MAE, but the lowest-MAE candidates miss more
active dryer use. The report retains labelled controls, all comparisons and
activity-specific trade-offs. All 434 tests and the independent audit pass;
no final-test evaluation, production replacement or hardware run was performed.

### Earlier IBM diagnostic reproduction

To reproduce without overwriting the archive, choose a fresh output directory:

~~~bash
PYTHONPATH=src .venv/bin/python scripts/run_effective_shot_diagnostic.py --mode freeze --output-dir results/quantum_diagnostics/effective_shots_new
PYTHONPATH=src .venv/bin/python scripts/run_effective_shot_diagnostic.py --mode replay --output-dir results/quantum_diagnostics/effective_shots_new
~~~

~~~bash
.venv/bin/python scripts/run_ibm_pilot.py --mode ideal --output-dir results/ibm_ideal_new
.venv/bin/python scripts/run_ibm_pilot.py --mode noisy --output-dir results/ibm_noisy_new
.venv/bin/python scripts/configure_ibm_quantum.py
.venv/bin/python scripts/run_ibm_pilot.py --mode prepare --output-dir results/ibm_qpu_new
.venv/bin/python scripts/run_ibm_pilot.py --mode submit --output-dir results/ibm_qpu_new
.venv/bin/python scripts/run_ibm_pilot.py --mode collect --output-dir results/ibm_qpu_new
~~~

Only `submit` sends a QPU job. Preparation records the selected backend,
calibration snapshot, compiled circuit, shot count, and execution-time limit.
The default pilot is 4,096 shots with a 60-second maximum QPU execution time.
Submission requires Open Plan access unless paid use is explicitly enabled.
Credentials use IBM's standard account file outside this repository; secret
values are never accepted as command-line arguments. Output directories are
created afresh, and a submission record prevents accidental duplicate jobs.

## Core Stage D comparisons and external REFIT

The duration/penalty archive in `results/stage_d/components/run_001/` contains
all four fixed-setting arms on the same 30 R1Hz windows. Macro MAEs are
26.041 W (original), 34.419 W (unit weights), 32.520 W (global maximum penalty),
and 32.927 W (global mean penalty). Independent dense-DP certificates and
paired-window intervals are retained. These are not independently retuned
models or a quantum-advantage comparison.

The bounded mixer archive in `results/stage_d/mixer/run_001/` contains
W/XY, H/X plus one-hot penalty, and W/X plus penalty: 720 distributions and
92,160 shot trials over all five Stage B shapes up to 12 qubits. It is an
architecture/initialization comparison, not an isolated mixer effect. A
sufficient energy penalty does not guarantee valid finite-depth samples;
some W/X controls retain a uniform feasible measurement distribution.

Official cleaned REFIT was downloaded from the University of Strathclyde
with source DOI, license and checksums in `data/external/refit/source.json`.
Bulk CSV files and the archive are excluded from version control. The
[external protocol](docs/stage_g_refit_protocol.md) preselects 14 eligible
homes and 30 train/30 test days per home. It is a calibrated-home ideal
simulation with transferred R1Hz settings, not zero-shot or hardware testing.
Unflagged imputation in cleaned REFIT remains an explicit limitation.
The [complete audited REFIT report](results/stage_g/refit/run_001/report.md)
accounts for all 420 selected test days: 357 nonempty days, 63 empty days,
956,904 retained blocks and 3,213 method/seed/day score records. All 14 homes
are evaluable. Equal-home macro MAEs are 89.636 W (ideal QAOA), 101.527 W
(uniform), 72.547 W (matched exact), 72.622 W (full temporal DP), and
41.345 W (training mean). QAOA improves over uniform in all homes but is worse
than each of those exact/constant controls. Exact DP greatly reduces aggregate
error relative to the constant while worsening appliance MAE: optimizing the
surrogate objective is not sufficient for appliance identification. Stage G is
complete in this bounded scope, not a quantum-advantage or zero-shot claim.

Reproduction uses a new archive directory; do not overwrite a scored run:

~~~bash
PYTHONPATH=src:. .venv/bin/python -m scripts.run_stage_g_refit --mode freeze --output-dir results/stage_g/refit/new_run
PYTHONPATH=src:. .venv/bin/python -m scripts.run_stage_g_refit --mode fit --output-dir results/stage_g/refit/new_run
PYTHONPATH=src:. .venv/bin/python -m scripts.run_stage_g_refit --mode evaluate --output-dir results/stage_g/refit/new_run
PYTHONPATH=src:. .venv/bin/python -m scripts.run_stage_g_refit --mode summarize --output-dir results/stage_g/refit/new_run
~~~

## Matched Stage E mixed-integer benchmark

The [completed benchmark](results/stage_e/mip/run_001/summary.json) and
[independent audit](results/stage_e/mip/run_001/independent_audit.json) compare
generic HiGHS MILP with fresh exact temporal DP on all 30 original windows,
34 valid runs and 86,396 blocks. The validation-selected presolve-on MILP
certifies every run and exactly matches the decoded DP objectives. Both give
26.041 W macro appliance MAE and 51.982 W aggregate MAE; the paired error
difference and 95% interval are zero.

Recorded inference totals are 58.773 s for MILP and 0.444 s for DP, including
compression and solver setup/decoding/expansion but excluding source reads,
model learning, scoring and archive I/O. This is one local execution per
arm/window, not a stable speed benchmark or a QPU comparison. The 72-joint-state
MILP is a matched-objective control, not a reproduction of a published NILM
pipeline. The full Balletti comparison remains pending; Li's full 60 Hz
transient-feature method is explicitly deferred beyond the paper's
low-frequency scope. Stage E therefore remains partially complete overall.
The [frozen protocol](docs/stage_e_mip_protocol.md) specifies validation-only
setting selection, immutable records and no fallback for absent incumbents.

## Open-science status

Code, tests, OpenQASM, raw Ocean samples and result summaries are versioned.
Manuscript sources/PDFs, cover letters, figures prepared for the manuscript,
submission bundles and the D-Wave proposal are author-only local materials.
Large source data remain in Harvard Dataverse. The repository is currently
private and should be made public no later than manuscript submission.

## License

The software is released under the Apache License 2.0. Dataset and third-party
materials retain their own terms.
