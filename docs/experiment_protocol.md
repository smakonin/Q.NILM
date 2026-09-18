# Q.NILM experiment protocol

Version: 0.5 (working protocol, updated after the first held-out classical campaign,
corrected IBM diagnostic check, and full-coverage ideal categorical-QAOA extension;
not a claim of external preregistration)

## Research question

Can event compression and feasibility-preserving quantum optimization solve
physically constrained NILM instances with fewer qubits while remaining
competitive with strong classical optimization methods under matched inputs
and reporting budgets?

The study will not infer quantum advantage from simulator accuracy or from a
quantum method finding an optimum that exact enumeration also finds.

## Primary hypotheses

1. Event compression materially reduces the number of decision variables
   relative to uniform-time encoding while retaining appliance-state and energy
   accuracy.
2. Duration-weighted reconstruction and appliance-normalized transition costs
   avoid the low-power state bias caused by a single global switching penalty.
3. A feasible-subspace mixer improves valid-sample rate for multi-state
   appliances compared with a penalty-only standard mixer at matched circuit
   depth.
4. On instances small enough for certification, Q.NILM samples solutions with a
   measurable, reproducible optimality gap. No speed or advantage claim will be
   made unless it survives comparison with the strongest tested classical
   solver and includes end-to-end overhead.

## Data

### R1Hz (primary high-resolution testbed)

- Canonical DOI: <https://doi.org/10.7910/DVN/RCB5VJ>
- Coverage: 757 days at 1 Hz from one Burnaby, Canada residence.
- Use the main aggregate as the NILM input.
- Use only non-mixed circuits as appliance proxies.
- Never use mixed circuits as single-appliance ground truth.
- Exclude `s`- and `+`-marked recovered/regularized rows from the new campaign;
  retain the original pilot's documented `s`-only policy when reproducing it.
- Run a separately labelled sensitivity analysis that includes recovered rows.
- Split chronologically: first 60% training, next 20% validation, final 20% test.
- Estimate thresholds, centroids, nominal powers, and event parameters from
  training data only.

R1Hz provides deep within-home evidence, not cross-home generalization. Testing
an established multi-home dataset (UK-DALE, REFIT, or REDD) is a study goal
needed before claiming cross-home generalization, not a journal-wide
submission requirement. The manuscript target is IEEE Transactions on
Quantum Engineering (TQE).

### First within-home campaign: completed

The timestamp-only manifest freezes 150 non-overlapping 24-hour windows:
90 training, 30 validation, and 30 test windows within the chronological
60/20/20 partitions. Windows are evenly spaced within each partition and
need not begin at midnight. The 30 test windows span May--October 2019 and
are not consecutive. No window was replaced after observing measurements.
The partitions contain 259,196 / 86,396 / 86,396 valid 30-second blocks;
four excluded rows in each partition invalidate four complete blocks.
Missing or excluded blocks reset temporal inference rather than closing gaps.

The [campaign protocol](../results/heldout_campaign/protocol.json) was written
before measurement reads. Appliance/background levels and FHMM probabilities
use training only. The [frozen models](../results/heldout_campaign/frozen_models.json)
record validation-selected penalties before any test window is read.
This is an internal freeze, not independent preregistration. All four
selected penalties equal the largest prespecified candidate, rho=0.1;
broader tuning requires a new design and untouched test evidence.

The fourteen comparisons comprise binary/multistate regular and compressed
temporal optimization, binary/multistate exact MAP FHMM, and a train-low-power
constant, each on actual mains and a selected-circuit-total control. The
mains models include a training-only three-state background component;
the selected-circuit control is not an input available to deployed NILM.
Test inference sees its stated aggregate, never test circuit labels.

Regular-block mains macro appliance MAE changes from 35.208 W (binary) to
24.729 W (multistate): a 29.76% classical modeling improvement. The paired
multistate-minus-binary difference is -10.478 W, with a descriptive 95%
window-bootstrap interval [-11.725, -9.214] W. The compressed comparison is
-9.819 W, interval [-11.485, -8.388] W. All errors use original valid blocks
after expanding compressed predictions; none are quantum-hardware results.

Vacuum mains MAE is worse than the low-power constant: 13.393 versus 2.198 W
for regular multistate inference, with above-threshold readings in only
three test windows. State/event metrics use common training-derived
**high-state proxies**, not verified physical ON states. The fridge threshold
244.903 W excludes its fitted 101.756 W middle state. Do not interpret
above-threshold coverage as ordinary compressor activity.

The [results table](../results/heldout_campaign/summary.md) and
[post-run interpretation notes](../results/heldout_campaign/interpretation_notes.md)
preserve every result and clarify the original proxy headings and compressed
objective offset. Raw campaign files are unchanged. The current implementation
passes 137 automated tests; these are implementation checks, not research trials.

### Full-coverage categorical extension: simulation completed

The [separate extension protocol](../results/quantum_heldout/protocol.json)
was frozen after the original test results were inspected. It is a
post-exposure extension, not a newly blind confirmatory test. It retains the
original multistate/background centroids, rho=0.1 penalties, event threshold,
all 30 test windows, and all 86,396 valid blocks. It does not retune these
choices using test appliance errors.

The model uses state counts [4, 3, 2, 3] for dryer, fridge, vacuum and signed
background. Mains-only event detection creates 4,881 compressed intervals in
34 valid runs. A sequential two-interval adaptation produces 2,451 circuits
per complete trajectory (2,430 pairs and 21 single-interval tails). Each
method carries its own last inferred state and resets at every data gap or
window boundary. Exact DP states and held-out labels are never QAOA warm starts.

The p=1 circuit uses product uniform one-hot states, a normalized diagonal
objective phase, and ordered adjacent-pair XY rotations implemented as
RXX(beta) followed by RYY(beta). The one/two-interval circuits encode 12/24
qubits, with 72/5,184 feasible states. Hardware routing can add active physical
qubits; encoded and physical counts must be reported separately. Feasible-space
simulation is exact for the specified ideal circuit, not physical execution.

One chunk per original training window, selected nearest the timestamp
midpoint, supplies 90 boundary-free training problems. A fixed 33-by-17 grid
minimizes mean expected objective divided by the coefficient-only scale.
The [frozen angles](../results/quantum_heldout/angles.json) are
gamma=1.5707963268 and beta=1.1087974071. No appliance MAE is used in this
angle selection. Classical cost enumeration and angle training remain part
of end-to-end resource accounting.

Ideal QAOA and uniform feasible sampling each use 256 shots per circuit and
three fixed seeds, totaling 1,882,368 shots per method. Every ideal-QAOA shot
is feasible by construction; no ideal fallback is used. Exact two-interval
inference and full-run compressed DP use the same model and mains input.

| Completed simulation/control | Macro appliance MAE (W) |
|---|---:|
| Ideal categorical QAOA | 32.091 |
| Uniform feasible sampling | 41.537 |
| Matched two-interval exact | 26.012 |
| Full-run compressed exact DP | 26.041 |

Stochastic error sums are averaged over seeds within each window before
pooling original blocks. A paired 2,000-resample window bootstrap yields
QAOA-minus-uniform -9.446 W [-11.264, -7.426], QAOA-minus-exact-chunk
6.079 W [4.322, 7.945], and QAOA-minus-full-DP 6.049 W [3.865, 8.420].
These are descriptive post-exposure, within-home intervals, not independent
hardware-date uncertainty or multiplicity-adjusted superiority claims.
QAOA is worse than the exact controls; no quantum advantage is established.
See the [full analysis](../results/quantum_heldout/simulation/summary.md).

### Full-coverage categorical hardware: complete and audited

The [hardware plan](../results/quantum_heldout/hardware/plan.json) schedules
2,451 one/two-interval circuits at 256 shots each in at most 129 adaptive jobs.
The preparation snapshot records 594 seconds of free allowance, a 540-second
campaign cap, and approximately 430 seconds estimated accounted usage.
The estimate includes approximate per-job overhead but is not a guarantee;
live remaining allowance and next-job caps are checked before each submission.
The first adaptive job, `dajnp09hvn6c73cul07g`, was submitted at
2026-09-14 04:48:01 UTC. All 129 jobs completed by 06:43:04 UTC, with
627,456 shots covering all 86,396 valid blocks. Actual accounted usage was
437 seconds, below the 540-second cap; remaining free allowance was 157
seconds at completion, not necessarily at any later check.

The [complete results](../results/quantum_heldout/analysis_complete/summary.md)
give hardware appliance MAE 324.017/140.646/482.396 W for dryer/fridge/vacuum,
macro MAE 315.686 W, and aggregate MAE 1,159.908 W. These include the
predeclared fallback in 131 chunks / 4,310 blocks (4.989%). Raw feasibility
is 8,770/627,456 (1.398%); two-interval circuits average only 2.89 usable
candidates from 256 shots. A complete independent audit passed all saved
Runtime counts, inference/coverage checks and usage records. This remains a
single post-exposure hardware campaign, not a new blind confirmatory test.

`run_quantum_heldout_ibm.py --mode prepare` freezes templates and metadata;
only `--mode run` submits or recovers jobs. Open Plan is mandatory, with no
paid override. Immutable intents bind each job to its plan, parameter values,
window list and own-prior states. Ambiguous submissions stop for reconciliation
rather than being blindly repeated. `--mode score` requires complete coverage.
Infeasible shots remain in the charged denominator; when a chunk has no valid
shot, use its own prior state, or the lowest-power state at a reset, and count
all fallback chunks and affected blocks. Partial hardware coverage is not a
full-window accuracy result. Existing pilot and simulation archives are retained.

## Experimental stages

| Stage | Purpose | Required evidence |
|---|---|---|
| A. Algebra | Verify objective construction | Exhaustive QUBO/direct and QUBO/Ising equality |
| B. Synthetic | Certify small controlled cases | Complete within the [frozen categorical programme](stage_b_protocol.md): ideal 4--24 qubits, generic exact noise 4--10 qubits, optimized p=1/p=2, three measurement-noise levels, four shot budgets and eight base-instance seeds; full independent audit passed |
| C. R1Hz pilot | Validate end-to-end data path | Fixed time ranges, channels, exclusions, QASM, exact certificate |
| D. Ablation | Identify which components matter | Core local comparisons complete and independently audited: binary/multistate, regular/compressed, duration weights, normalized/global penalties, and bounded W/XY versus H/X and W/X architectures; fixed-setting and post-exposure limitations retained |
| E. Baselines | Establish practical relevance | Exact temporal DP, small enumeration, exact MAP FHMM and the independently audited matched MILP benchmark complete; source-faithful Balletti pipeline pending. Per user scope decision, Li's full 60 Hz transient-feature method is explicitly deferred beyond this low-frequency study; see [source audit](stage_e_sources.md) |
| F. Quantum noise | Test implementation robustness | Legacy and corrected nine-qubit Fez checks, twelve ideal/Fez-noise local controls, full categorical campaign and unchanged component-ladder retry complete. Independent ideal/raw-count audits passed for all 24 diagnostic circuits and 24,576 shots; independent multi-date repeats pending |
| G. External data | Test generalization | Complete within [the frozen external protocol](stage_g_refit_protocol.md): 14 calibrated REFIT homes, all 420 days accounted for, 357 nonempty days and 956,904 retained blocks; independent full audit passed. Not zero-shot or QPU validation |
| H. Annealing readiness | Scale the shared QUBO with Ocean | SA/tabu baselines, graph profiles, exact certificates where tractable |
| I. D-Wave hardware | Evaluate direct annealing separately from QAOA | Embedding, chain breaks, QPU timing, gauges, anneal schedules, raw samples |
| J. Full-coverage categorical QAOA | Test the sequential one-hot/XY algorithm on every original test block | Ideal and matched uniform/exact controls plus the full IBM campaign are complete and audited; hardware performs substantially worse |

Stage I must distinguish direct QPU sampling from Leap hybrid execution. The
D-Wave experiment evaluates an annealing implementation of the same QUBO/Ising
objective, not the gate-model QAOA circuit. The local Ocean scaling benchmark is
readiness evidence only and cannot support a hardware or quantum-advantage claim.
The separate `results/temporal_scaling/` audit now supplies exact temporal-DP
certificates for all 18 synthetic instances, including 288 variables. Median
solver time for the largest shape is 0.288834 s on the recorded Apple M5,
using three seeds and three solves per instance. The original Ocean archive
is unchanged. These local timings are not a matched QPU speed comparison.

The completed [matched MILP benchmark](../results/stage_e/mip/run_001/summary.json)
and [independent audit](../results/stage_e/mip/run_001/independent_audit.json)
cover all 30 original test windows, 34 valid runs, 4,881 compressed intervals
and 86,396 blocks. Validation selected presolve on; all test runs were
solver-certified optimal with valid incumbents and bounds. MILP and fresh
exact DP have identical decoded objectives and appliance/aggregate metrics:
26.0414278901 W macro appliance MAE and 51.9818028765 W aggregate MAE.
The paired-window MAE difference and 95% interval are zero. Recorded inference
totals are 58.7729480409 s (MILP) and 0.4441493324 s (DP), excluding common
source reads, model learning, scoring and archive I/O. This single local
execution is not a stable speed benchmark or a quantum-speed comparison.
It completes the generic matched-objective solver comparison, not the full
Balletti algorithm or all of Stage E. See the
[frozen MILP design](stage_e_mip_protocol.md) for the timing and selection rules.

## Metrics

Primary NILM metric for the completed campaign: appliance-power MAE on original
valid 30-second blocks, with equal appliance weights. Secondary metrics:
high-state-proxy F1 and Matthews correlation coefficient, event F1
with an explicitly fixed tolerance, mean absolute error, signal aggregate error,
normalized disaggregation error, and per-appliance energy error.

Optimization metrics: objective value, certified optimality gap, probability of
sampling an optimum, valid-sample rate, shots-to-solution at 99% confidence, and
wall-clock time separated into preprocessing, parameter optimization, queue,
execution, and postprocessing.

Circuit metrics: logical and physical qubits, one- and two-qubit gate counts,
transpiled depth, connectivity overhead, shots, optimizer evaluations, and
random seed.

Annealing metrics: logical variables and couplers, coefficient range, physical
qubits, maximum and mean chain length, chain strength, chain-break fraction,
annealing time, QPU access time, programming time, readout time, gauges, reads,
embedding failures, solver identifier, calibration timestamp, and wall-clock
time. Hybrid timing and solution quality are reported in a separate stratum.

## Statistical design

- Fix preprocessing and hyperparameters on training/validation data.
- Report test data once after freezing the protocol.
- Use at least 30 non-overlapping event windows per appliance and home where
  data permit.
- Report mean, median, standard deviation, and 95% bootstrap confidence
  intervals across windows and optimizer seeds.
- Use paired comparisons on identical windows.
- Correct families of secondary comparisons for multiple testing.
- Publish failures and zero-result runs.

For the first campaign, the prespecified paired bootstrap resamples the
30 fixed 24-hour test windows, sums per-window absolute errors and valid-block
counts, then divides their resampled totals. It uses 2,000 replicates and seed
7301. Intervals are descriptive within this one-home sample, not simultaneous
multiple-testing-adjusted superiority claims or cross-home uncertainty.
The campaign does not satisfy a requirement for 30 active windows per appliance;
rare-load coverage is explicitly reported rather than filled by reselection.

## IBM discovery, corrected diagnostic, and planned confirmation

The discovery run on `ibm_fez` used one nine-qubit, depth-one circuit and
4,096 shots. It found the unique objective optimum 92 times. All reported
best-state metrics describe that minimum-cost observed state: 8/9 proxy
accuracy, F1 0.8571, MCC 0.7906, and duration-weighted aggregate MAE 234.04 W.
This selected interval uses a sum of three circuit channels and parameters
estimated within the same hour. It is not the held-out mains evaluation.
The discovery record is frozen in `results/ibm_qpu_pilot/`.

The original run used positive-clipped baseline residuals. Its legacy-target
MAE must not be labelled physical raw-power MAE. The correction, new local
controls, and corrected hardware check are separate from this immutable record.

### Representation correction: completed local audit

Signed centering is now the default: `y = raw selected total - sum(baselines)`.
Negative centered values are residuals, not negative physical consumption.
Use `--aggregate-mode legacy-clipped` only to reproduce historical experiments.
`results/representation_correction/summary.json` freezes source hashes,
calibration, original boundaries and labels, with exact DP solutions before
and after correction. No powers, penalties or labels were retuned.

| Scope | Legacy vs signed proxy agreement | Common raw selected-total MAE (W) |
|---|---:|---:|
| Selected 3 intervals | 8/9 vs 9/9 | 447.89 vs 430.89 |
| Full-hour 16 intervals | 41/48 vs 41/48 | 565.08 vs 552.05 |
| Full-hour 120 blocks | 316/360 vs 293/360 | 262.62 vs 243.88 |

MAE is evaluated on identical raw 30-second measurements within each case,
with interval predictions expanded and the fitted total baseline added.
Per-interval majority agreement and original block agreement are distinct:
the corrected selected pilot matches 68/69 block proxy bits. The audit
fixes a demonstrated clipping error but does not validate the binary state
model, same-hour fitting, or proxy definitions. The drop in full-hour block
agreement is a required result, not an excluded failure. Do not retune on
this diagnostic hour and then describe it as test data.

Local corrected QAOA artifacts are in `results/pilot_corrected/` (NumPy,
87 optimum samples / 5,000) and `results/ibm_corrected_ideal/` (noiseless Aer,
78 / 4,096). These used a new classical angle search; they are not a paired
QPU improvement experiment.

### Corrected IBM check and local controls: completed

The corrected nine-qubit, depth-one circuit ran on `ibm_fez` in job
`dajn53r9k43c73ahdmag`. Its exact optimum appeared 75 times in 4,096 shots;
the best sample matched 9/9 diagnostic proxy states. IBM reported 3 seconds
of accounted QPU usage. This excludes the full classical optimization,
preparation, queue, and postprocessing workflow and is not a speed comparison.
The [corrected hardware archive](../results/ibm_corrected_qpu_check/summary.json)
uses the same retrospective selected-circuit hour, not the held-out mains
windows. It does not establish general QPU NILM accuracy or quantum advantage.

Twelve local controls (4,096 shots each) are complete: depths one/two,
three transpiler seeds, and ideal/archived-Fez-noise sampling. The depth-two
equal-angle split is an unoptimized sensitivity control. The noise model uses
the archived calibration and is not a hardware repeat. The controls are
circuit-matched but not calibration-matched to the new QPU job: an audit found
27 changed active calibration parameters between snapshots. The completed record is
[`run_001`](../results/ibm_followup_controls/run_001/summary.json); see the
[IBM follow-up plan](ibm_followup_plan.md) for interpretation and resource plans.

### Next tests and quantum-advantage gate

The first internally frozen split now compares binary versus multistate and
regular versus event-compressed inference on actual mains with background
modeling. Its raw-offset formulation is algebraically equivalent to signed
centering. Extend this design to untouched periods and additional homes;
preserve gaps and evaluate appliance power and appropriately qualified events.
State estimation must never use held-out proxy labels. Do not reuse the exposed
first-campaign test windows for new tuning and call the outcome held-out.

The binary chain has an exact structure-aware baseline implemented in
`quantum_nilm.classical`: O(K N 2^N) time, not O(2^(K N)). At three appliances
there are only eight states per interval. Scale appliance count and state
complexity, not only the number of intervals. Include tuned DP, MIP,
SA/tabu and conventional NILM. For richer constraints where this DP no longer
applies, supply an appropriate strong solver and bounds rather than calling
an uncertified heuristic result exact.

Predefine the claimed benefit (e.g. lower total time to a specified objective
gap and application accuracy at 99% success) and benchmark paired instance
families under common budgets. Include classical parameter-search costs;
the current exact-statevector grid optimization is not scalable training.
Report queue time separately as well as end-to-end time, failures, uncertainty
across independent instances/jobs, and all tuning budgets. A statistically
significant result versus uniform sampling alone is not quantum advantage.
The current evidence supports circuit validation, within-home classical
model comparisons, and fully covered ideal and physical categorical-QAOA
evaluations. Both underperform matched exact classical controls; the physical
campaign exposes severe feasible-sample loss. Neither establishes advantage.

A separate post-exposure usable-shot diagnostic is complete in
`results/quantum_diagnostics/effective_shots/`. Each of the original 2,451
chunks receives its observed hardware feasible count, including zero, in
three fixed-seed ideal-QAOA and three uniform-feasible replays. Every replay
carries its own previous inferred state with the original resets and
declared zero-budget fallback. Frozen training parameters are unchanged;
test labels are used only for scoring after predictions are fixed.
All replays cover 86,396 blocks in 30 windows, with 8,770 draws and 4,310
fallback blocks per seed. Macro MAE is 199.052 W for ideal QAOA and 387.977 W
for uniform sampling, compared with 32.091 W and 41.537 W at 256 usable
shots per chunk. Observed counts are endogenous to the hardware trajectory;
this is a conditional sensitivity diagnostic, not a causal error
decomposition, fresh blind evaluation, or quantum-advantage comparison.

The unchanged, user-authorized component-ladder retry is complete in
`results/quantum_diagnostics/ladder_retry_001/`: 24 circuits, 24,576 raw
measurements and 9 seconds of finalized QPU charge. Independent compiled-ideal
and raw-count/objective audits passed. Its selected training circuits and
barrier-separated component families are not a full NILM evaluation or an
independent-date repetition. The first attempt's error 9701, absent measurements
and provisional accounting remain archived separately; Run 1 is unchanged.

The following additional hardware studies remain planned and unapproved;
they are distinct from the completed full-coverage categorical campaign and
the separately authorized component-level diagnostic ladder. The single
corrected check does not authorize their submission:

1. Repeat the corrected nine-qubit circuit on the same hardware backend using
   three transpiler seeds, three independently submitted jobs per seed,
   and three calibration dates: 27 executions and 110,592 shots. Randomize
   order within each date. Preserve the same compiled circuit for repeats
   within a seed/date cell; archive the calibration and layout for every cell.
2. Evaluate six, nine, and twelve logical qubits at QAOA depths one and two
   across at least 30 preselected non-overlapping held-out windows in total
   across sizes. Apply the main statistical design above for the full NILM
   study. Window selection must not use reference states or favorable results.
3. Compare against exact temporal DP, enumeration/MIP, uniform sampling, SA/tabu, and
   conventional NILM, using identical inputs for NILM and the same QUBO for
   solver comparisons. State matching shot, evaluation, or wall-time budgets
   explicitly. Include preprocessing, classical angle optimization,
   compilation, queueing, execution and postprocessing time.
4. Validate the multi-state XY mixer against a penalty-based X mixer on hardware. State
   initial states, penalty tuning and resource budgets; report valid-sample
   fraction before any postselection, its discarded-shot cost, and resulting
   disaggregation metrics. The binary nine-qubit pilot does not test this claim.
   The bounded local Stage D architecture comparison is now complete; it
   does not establish a universal XY winner or replace hardware repetitions.

The initial 27-run budget is not a power calculation or a guarantee of QPU
usage. Estimate time from each prepared workload before launching it. Use
windows, dates and layouts as the relevant replication units; do not treat
thousands of shots from one job as thousands of independent NILM instances.
Freeze endpoints and multiple-comparison families before confirmatory runs.

For the discovery pilot only, Wilson intervals describe per-shot optimum
frequency under an iid within-job assumption. They exclude hardware drift,
layout and between-window variation. The higher observed hardware optimum
frequency is not evidence of quantum advantage: its mean objective is
39.66% above the exact ideal-QAOA expectation. Also, 4,096 uniform draws from
512 states find the unique optimum at least once with probability 99.9667%.
Report the full objective distribution and hit probability alongside the best
sample, and compare success across shot budgets.

## Reproducibility record

Every result must record the Git commit, dataset DOI/version, exact local time
range and time zone, channel set, treatment of markers and daylight-saving
transitions, software environment, seed, circuit, backend, calibration date,
transpiler settings, and raw result file checksum.
