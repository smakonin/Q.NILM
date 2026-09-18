# Steps 1 and 2 — completed results

Completed 14 September 2026. These are local classical simulations and new
within-home R1Hz evaluations. No hardware job, provider-account operation,
production-parameter replacement, paper edit or GitHub push was performed.

## Main conclusion

CVaR50 is markedly more robust than mean loss on the six-qubit synthetic
training example under the wider-angle finite-shot setting. That advantage
does **not** translate into lower appliance MAE on the newly reserved R1Hz
test windows. Mean and CVaR50 both beat uniform sampling at 16 inference shots,
but neither beats the lowest-power constant baseline on the predeclared
three-appliance metric at that budget. At 256 shots, their errors are close
to uniform sampling and exact optimization.

The more consequential diagnostic is the gap between exact mains optimization
and the label-only representation reference: 25.390 versus 5.919 W. Even a
perfect solver of this static objective does not reliably identify appliance
powers. Improving the objective/background/temporal model is a higher-priority
next investigation than replacing mean loss or requesting another full IBM run.
These results do not establish the cause of the earlier hardware failure.

## Frozen design and data separation

The [protocol](training_validation_protocol.md) and executable sources were
frozen before execution. Diagnostic and data-quality checks shaped the study:
matched complete measurement budgets, all paired trials retained, explicit
measurement rechecking, timestamp-only data selection, and selection before
test access.

The exposure inventory identified the original 150 training/validation/test
days plus the pilot hour. New validation/test windows lie outside these
recorded exposures with a one-day guard on each side. Ten new validation days
were chosen from the original middle chronological partition; twenty test
days from the final partition. Selection used timestamps, not appliance activity
or results. The window starts span December 2018–May 2019 for validation and
May–October 2019 for testing; full timestamp manifests are archived.

“New” means new to the recorded Q.NILM evaluation windows, not proof that no
person previously inspected these portions of the dataset. This remains one
home. Global source ordering is assumed; ordering and duplicates were checked
inside each extracted window.

| Data use | Prespecified samples/windows | Accepted 30-second blocks |
|---|---:|---:|
| Angle training | 32 midpoint blocks from existing training days | 32 |
| Parameter validation | 10 new 24-hour windows | 28,800 |
| Final evaluation | 20 new 24-hour windows | 57,596 |

Four of 57,600 possible test blocks were excluded because each contained one
marked source row. No nonfinite values, duplicate timestamps or missing
seconds occurred in these selected test windows. No window was replaced.
The exclusion and block counts were independently reproduced from the CSV.

Appliance/background centroids were reused from the original train-only
multistate model; they were not refitted on validation or test data. Angle
training uses a small fixed sample of 32 training aggregates, not all the
training blocks used to learn the centroids.

## Step 1: wider angles and finite-shot training together

Eighty fits compare two losses, two angle ranges and twenty paired trials.
Each fit uses the same two-restart 1,024-evaluation search policy, 256 simulated
ideal measurements per evaluation, then 4,096 independent measurements for
each of its five best distinct candidate vectors. The extra 20,480 measurements
are included in the budget. All losses and candidate selections are recorded;
exact probabilities and appliance truth do not select angles.

Results below average all twenty trials for the 680 W six-qubit example.
Errors are analytical expected appliance MAE after selecting the lowest-cost
answer among 16 samples, not the raw training-loss estimates.

| Loss | Gamma range | Before rechecking (W) | After rechecking (W) | After-recheck trial range (W) |
|---|---|---:|---:|---:|
| Mean | [0,8π] | 18.442 | 18.207 | 14.958–24.108 |
| CVaR50 | [0,8π] | 4.117 | 3.288 | 2.415–6.200 |
| Mean | [0,16π] | 5.005 | 4.258 | 0.411–19.565 |
| CVaR50 | [0,16π] | 0.473 | 0.449 | 0.408–0.546 |

Wider-range post-recheck expected best-16 objective is 967.101 W² for mean
versus 67.081 W² for CVaR50. This supports CVaR50 as a candidate for robust
finite-shot training on this example; it is not proof of a universally better
loss. Twenty optimizer seeds are not twenty independent NILM problems.

The eight other synthetic aggregates are secondary transfer checks, not
independent load models. Their wider-range average MAE is 18.029 W for mean
and 15.453 W for CVaR50 after rechecking. Rechecking actually worsens CVaR50's
eight-case transfer average from 13.614 to 15.453 W while improving its training
case. Rechecking therefore does not guarantee generalization or eliminate
all selection effects. Both predeclared real-data arms proceeded regardless.

## Step 2: newly reserved R1Hz data

### Task and validation selection

The real task uses one independent 30-second interval per problem: dryer four
states, fridge three, vacuum two, and latent background three. That is 12
one-hot qubits and 72 feasible assignments. Input is actual mains, with raw
absolute power centroids. There is no event compression, temporal penalty,
state carry or test-label-supplied background. This is a static diagnostic,
not the paper's two-interval temporal IBM task.

Twenty real-data fits compare mean and CVaR50 across ten paired trials. Each
evaluation uses 256 measurements for each of the 32 training cases; the
per-case normalized losses are averaged equally. Each fit also gets the same
five-candidate independent recheck. Rechecking consumes 655,360 measurements
per fit on top of 8,388,608 search measurements.

Each loss's fit was selected by measured appliance MAE on all validation blocks
at 16 inference shots, before test data were read:

| Loss | Selected trial ID | Gamma | Beta | Validation MAE (W) |
|---|---|---:|---:|---:|
| Mean | r1hz_mean_01 | 1.042876 | 1.042263 | 36.044 |
| CVaR50 | r1hz_cvar50_00 | 0.473709 | 0.865104 | 35.934 |

The overall validation winner was CVaR50. Selection was frozen at
2026-09-14 22:12:45 UTC before test extraction. Both selected arms were evaluated
and retained; the test result did not trigger another fit, a new candidate
selection, or any parameter change. Their selected angles lie inside the
original bound too, so the wider domain itself is not the explanation for
their real-data result.

### Primary held-out result

Macro MAE is the mean of the three appliance MAEs over all accepted test
blocks. Measurements of the submeter powers supply scoring truth; they do not
enter the mains inference arms. Values for sampled methods are exact expected
errors over their ideal sampling distributions at the specified raw budget.

| Method | 16-shot MAE (W) | 256-shot MAE (W) |
|---|---:|---:|
| Q.NILM, mean loss | 38.055 | 25.364 |
| Q.NILM, CVaR50 | 38.445 | 25.473 |
| Uniform feasible sampling | 52.473 | 25.367 |

| Deterministic reference | MAE (W) | Interpretation |
|---|---:|---|
| Exact 72-state mains optimization | 25.390 | Perfect minimization of the same static objective; no quantum shots |
| Lowest-power training-centroid constant | 32.830 | Simple deployable reference, no mains disaggregation |
| Nearest-centroid appliance oracle | 5.919 | Label-only representation floor; not a deployable algorithm |
| Exact selected-circuit aggregate control | 9.849 | Privileged target-submeter sum as input; not actual-mains NILM |

CVaR50 minus mean at 16 shots is +0.390 W, with a descriptive paired-day
bootstrap interval of [-0.103,+1.022] W. It does not show a clear improvement
for CVaR50. At 256 shots the difference is +0.110 W, interval [+0.025,+0.213] W,
a small difference favoring mean in this cohort. These intervals resample
twenty same-home 24-hour windows; they are not multiplicity-adjusted significance
claims or measures of cross-home uncertainty.

Mean minus uniform at 16 shots is -14.418 W (interval [-19.640,-8.156] W);
CVaR50 minus uniform is -14.028 W ([-19.748,-7.136] W). However, both 16-shot
trained methods are worse than the constant predictor on the predeclared
three-appliance metric. At 256 shots, uniform, mean and exact results are
numerically very close; the experiment does not establish a practical quantum
benefit or a time advantage.

### Appliance-level breakdown and objective alignment

| Method | Dryer MAE (W) | Fridge MAE (W) | Vacuum MAE (W) | Macro MAE (W) |
|---|---:|---:|---:|---:|
| Mean, 16 shots | 44.378 | 48.228 | 21.558 | 38.055 |
| CVaR50, 16 shots | 45.686 | 47.862 | 21.787 | 38.445 |
| Exact mains objective | 29.490 | 34.562 | 12.119 | 25.390 |
| Nearest-centroid oracle | 13.210 | 3.725 | 0.823 | 5.919 |
| Exact selected-circuit control | 14.983 | 8.018 | 6.545 | 9.849 |

The exact-minus-oracle gap is 19.471 W (descriptive interval 14.349–26.640 W).
The per-appliance excesses are 16.280 W for the dryer, 30.837 W for the fridge
and 11.296 W for the vacuum. These are gaps above the discrete representation's
label-only error floor; they are not a uniquely identified physical or causal
mechanism.

Using the sum of the target circuits instead of mains reduces exact-solver
MAE to 9.849 W. This is consistent with an important background/identifiability
limitation, but the control changes both the observation and the hypothesis
space. Its 15.541 W improvement over mains cannot be presented as a pure causal
attribution to background quantization. Even that privileged control remains
above the representation floor, so background is not the only possible issue.

Mean at 256 shots has slightly lower appliance MAE than exact optimization,
even though its expected objective is worse: 2,629.904 versus 1,655.944 W².
This is not an optimizer beating the exact optimum. It demonstrates that
minimizing aggregate residual and minimizing appliance MAE are different tasks.

### Important cohort limitation: no active vacuum examples

Post-hoc examination of the fixed cohort found measured vacuum power between
0 and 2 W, with every one of the 57,596 blocks closest to its low-power state.
There are no high-state vacuum examples here. The test therefore measures
vacuum false activation/off-state performance, not its active-use recovery.
This also helps the lowest-power constant baseline. The cohort was not changed
after this was discovered.

For context only, the post-hoc dryer/fridge-only 16-shot average is 46.303 W
for mean versus 48.834 W for the constant predictor. This is **not** a replacement
for the prespecified three-appliance primary metric. Activity-conditioned or
appliance-subset metrics must not be used to select a preferred headline after
seeing the test.

Nearest-centroid state counts were [33,361,22,834,945,456] for the dryer,
[27,539,29,813,244] for the fridge, and [57,596,0] for the vacuum. The actual
mains-minus-target residual ranges from -56.4 to 2,773.567 W; this includes
other household demand and measurement mismatch, not negative physical
consumption being clipped or repaired.

## Verification, resources and reproduction

All 420 software tests passed, including 12 new tests. The independent audit
passed with these checks:

- Reaggregated all 62 source selections independently: 32 training blocks,
  ten validation windows and twenty test windows; 86,428 accepted blocks total.
- Replayed all 102,400 search evaluations and 500 shortlist rechecks,
  reproducing 203,489,280 classically generated measurements.
- Recomputed synthetic metrics, all 200 validation method/day results,
  all 140 test method/day results, validation selection, pooled MAEs and the
  paired bootstrap contrasts.
- Checked 140 explicit ideal Qiskit statevectors: all 80 final synthetic fits,
  and all 20 real fits on three fixed training cases each.

Maximum loss discrepancy was 1.12×10^-16; maximum physical-statevector
probability discrepancy was 5.56×10^-16. The audit checks optimizer trajectories
and selection, but does not independently rerun the optimizer. Ideal circuit
verification is not a hardware test.

Search consumed 188,743,680 simulated measurements and rechecking consumed
14,745,600. All are classical RNG draws, not QPU shots. Summed fitting time was
48.96 seconds; that excludes source extraction, validation/test scoring,
serialization and independent auditing, so it is not an end-to-end speed
comparison. No cost or runtime advantage is claimed.

Archive: [training_validation_001](../results/quantum_diagnostics/training_validation_001/).
Plan SHA-256: `47cc6aead11ff7bc5f33f98febfcecd1cec64041014c5d94e6f5de5d2e6816ac`.

- [Plan and data-window manifest](../results/quantum_diagnostics/training_validation_001/plan.json)
- [Complete summary and contrasts](../results/quantum_diagnostics/training_validation_001/summary.json)
- [Frozen validation selection](../results/quantum_diagnostics/training_validation_001/selection.json)
- [Independent audit](../results/quantum_diagnostics/training_validation_001/independent_audit.json)
- [Post-hoc activity and error breakdown](../results/quantum_diagnostics/training_validation_001/diagnostic_breakdown.json)

To replay this exact experiment in the same checked environment, copy only
the frozen plan into a new named folder, then run the modes below in order.
The source dataset, dependency versions and frozen input files must still
match the plan. The completed archive refuses phase overwrites. This is a
same-cohort reproduction, not another untouched evaluation; execution timestamps
and measured runtimes will differ.

```sh
mkdir results/quantum_diagnostics/training_validation_new
cp -n results/quantum_diagnostics/training_validation_001/plan.json results/quantum_diagnostics/training_validation_new/plan.json
PYTHONPATH=src:. .venv/bin/python scripts/run_training_validation.py --mode verify --folder results/quantum_diagnostics/training_validation_new
PYTHONPATH=src:. .venv/bin/python scripts/run_training_validation.py --mode synthetic --folder results/quantum_diagnostics/training_validation_new
PYTHONPATH=src:. .venv/bin/python scripts/run_training_validation.py --mode train-real --folder results/quantum_diagnostics/training_validation_new
PYTHONPATH=src:. .venv/bin/python scripts/run_training_validation.py --mode validate --folder results/quantum_diagnostics/training_validation_new
PYTHONPATH=src:. .venv/bin/python scripts/run_training_validation.py --mode test --folder results/quantum_diagnostics/training_validation_new
PYTHONPATH=src:. .venv/bin/python scripts/run_training_validation.py --mode summarize --folder results/quantum_diagnostics/training_validation_new
PYTHONPATH=src:. .venv/bin/python scripts/audit_training_validation.py --folder results/quantum_diagnostics/training_validation_new
PYTHONPATH=src:. .venv/bin/python scripts/summarize_training_validation_diagnostics.py --folder results/quantum_diagnostics/training_validation_new
```

The archived test windows are now exposed and cannot be reused as a fresh
confirmatory test for a revised objective. Reproduction of these same windows
is still valid reproducibility work, not new independent confirmation.
The `freeze` mode instead designs a new campaign using the expanded exposure
inventory; it will not recreate this cohort and may stop if insufficient
unexposed windows remain. Do not weaken the exclusion rule to force a new run.

## Recommended next work, not performed here

Retain mean loss as the baseline. On training/validation data, investigate
background representation and temporal/state priors, first with exact classical
optimization. If that improves actual appliance MAE, test the revised objective
with QAOA and reserve another untouched evaluation cohort before scoring it.
Keep solver quality, appliance identification and physical circuit validity
as distinct questions. Do not proceed directly from these simulations to a
claim of repaired IBM accuracy or quantum advantage.
