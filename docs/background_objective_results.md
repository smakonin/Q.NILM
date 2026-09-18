# Background representation and objective: diagnostic findings

Completed 14 September 2026. This is an exploratory **training/validation-only**
study. No final-test data were loaded or scored; no production model, QAOA
parameters, IBM archive, paper or GitHub state was changed.

## Main finding

The static Q.NILM cost rewards a close fit to total mains power even when that
fit assigns background demand to the wrong appliances. Coarse background
quantization is a major limitation in the controlled validation comparisons.
However, making the background more flexible **without appliance plausibility
information makes disaggregation worse**, despite improving the mains fit.

Training-derived state probabilities reduce average appliance error, but the
lowest-MAE candidates sacrifice active-dryer recall. None of the thirty
candidates improves macro MAE while simultaneously preserving or improving
both dryer/fridge missed-activity counts and all three false-activation counts
relative to the baseline. This last simultaneous comparison is post-hoc and
descriptive, not a predeclared acceptance test.

The implication is to investigate an activity-aware probabilistic/temporal
objective, not simply increase the number of background qubits or rerun IBM.
This study diagnoses a static model limitation, not the cause of the previous
physical-circuit degradation.

## Scope, data and comparison policy

The [protocol](background_objective_protocol.md), candidate grid and executable
sources were frozen before the new validation comparisons. Diagnostic and
data-quality guidance shaped the design: train-only model fitting, no reuse of
the final test, exact reference solvers, retained failed/poor candidates, and
activity-specific checks before interpreting a lower aggregate MAE.

| Use | Windows | Valid 30-second blocks | Scope |
|---|---:|---:|---|
| Parameter/statistics fitting | Original 90 training days | 259,196 | Training centroids/frequencies only |
| Diagnostic comparison | Existing 10 validation days | 28,800 | Reused development data, not held-out confirmation |

The original appliance levels were preserved. The three original background
centres were reproduced from the training data: 145.435, 426.474 and 1,296.050 W.
All four background fits converged. There were thirty inference candidates and
seven labelled controls, evaluated on every validation day. No day was replaced.

Input to ordinary candidates is actual mains. Appliance labels enter training
statistics and validation scoring, but not ordinary validation inference.
The known-background and quantized-input controls explicitly use labels and
are not deployable methods. Source order is assumed globally and verified
within extracted windows; no full-source ordering certification is claimed.

## 1. What the current objective does

For one interval, the exact baseline minimizes

\[
E(s,b)=\left[y-\sum_{i=1}^{3}p_i(s_i)-b\right]^2.
\]

Here `y` is mains, each `p_i` is a fixed training-derived appliance centroid,
and `b` must equal one of the three background centroids. There is no probability
penalty for a rare appliance state in this static objective. A background
discrepancy can therefore be explained by turning on an appliance.

Changing squared residual to absolute residual alone cannot repair this
single-interval objective: both rank assignments identically. This statement
does not extend to sums across time or costs with additional penalties.

At the opposite extreme, a free continuous background is underdetermined:
for any target combination below mains, `b = y - sum(p_i)` gives zero residual.
There are a median of four such nonnegative-background target combinations per
validation interval. More flexibility is not itself more identifying information.

## 2. Controlled inputs isolate the representation mismatch

All rows use the same measured appliance watts for MAE scoring. The first four
retain the same 72-assignment, three-background-state decoder; only its input
construction changes.

| Input/control | Appliance MAE (W) | Meaning |
|---|---:|---|
| Actual mains, current objective | 36.480 | Ordinary baseline |
| Quantized appliances + actual residual background | 36.208 | Removes target-centroid mismatch, retains background mismatch |
| Measured appliances + quantized background | 3.685 | Removes background mismatch, retains measured target powers |
| Quantized appliances + quantized background | 2.034 | Exactly in the assumed model space |
| Measured target sum, true background removed | 2.989 | Privileged known-background control |
| Nearest appliance-centroid oracle | 2.034 | Label-only representation floor |

The fully quantized input recovered all target categories on all 28,800
intervals. This argues against an exact-decoder implementation error for the
assumed model space. Quantizing the background alone greatly reduces MAE;
quantizing only the targets does not. These are strong controlled diagnostics
of the static representation, but not measurements of an achievable real-world
background estimator. The differences are not additive causal contributions.

Background quantization error is also associated with excess appliance error.
Intervals with residual-background quantization error below -100 W or at least
+100 W comprise 7,617/28,800 blocks (26.45%) but account for 68.99% of the baseline
MAE excess above the representation floor. The bins reconcile exactly to that
gap. This association is not a separate causal estimate.

## 3. More background states alone worsen appliance identification

| Background states | One-hot qubits | Oracle background quantization MAE (W) | Fitted mains RMSE (W) | Appliance MAE (W) |
|---|---:|---:|---:|---:|
| 1 | 10 | 142.430 | 77.771 | 44.112 |
| 3, current | 12 | 69.348 | 40.199 | 36.480 |
| 6 | 15 | 46.646 | 25.944 | 40.705 |
| 12 | 21 | 19.770 | 11.902 | 46.978 |

All rows minimize squared aggregate residual exactly. The background error
column uses true residual background only for scoring the nearest available
centroid, not as inference input. From three to twelve background states,
both representation fidelity and total-power fit improve, while appliance MAE
rises. This is direct evidence that optimizing aggregate reconstruction is not
equivalent to optimizing appliance identification.

The one-hot counts apply to one interval: nine appliance qubits plus K background
qubits. They are not resource estimates for the paper's two-interval circuit.

## 4. Plausibility information helps, with important trade-offs

For state frequencies learned only from training data, we evaluated

\[
E_{\lambda}(s,b)=\left[y-\sum_i p_i(s_i)-b\right]^2
 -\lambda\sum_j\log\pi_j(s_j).
\]

The prior probabilities use one pseudocount per state. Coefficients 100, 1,000
and 10,000 W² were fixed in advance. The prior can cover only appliances, only
background, or all four channels. It adds unary terms in a one-hot encoding,
so the squared-residual-plus-prior objective remains quadratic. A changed cost
still requires QAOA retraining and a separate hardware evaluation.

| Diagnostic candidate | Validation MAE (W) | One-interval encoding / qualification |
|---|---:|---|
| Current exact squared-residual objective | 36.480 | 12 one-hot qubits |
| Appliance-only prior, K=3, lambda=10,000 | 18.685 | Same 12 qubits; quadratic cost |
| All-channel prior, K=6, lambda=10,000 | 16.521 | 15 qubits; quadratic cost |
| All-channel prior, K=12, lambda=10,000 | 16.140 | 21 qubits; quadratic cost |
| Integrated 12-component background density + appliance prior | 13.707 | Classical likelihood diagnostic; not a demonstrated QUBO circuit |
| Lowest-power training-centroid constant | 18.936 | Simple non-disaggregating baseline |

All four listed prior-based candidates improve MAE on each of the ten validation
days relative to the current exact objective. This remains reused validation,
with hyperparameter selection on the same days; it is not independent evidence
of generalization or statistical superiority.

The continuous-background diagnostic integrates a Gaussian mixture fitted from
training residual clusters rather than choosing a single background state.
Means, frequencies and within-cluster standard deviations are training-only;
standard deviation is floored at 1 W. Its lowest-MAE configuration uses twelve
components and unit appliance-log-prior weight. Without that appliance prior,
the twelve-component model gives **44.049 W**, worse than baseline. The
log-mixture likelihood is generally not a simple quadratic cost; no circuit
implementation or hardware improvement is claimed for it.

Background priors alone are counterproductive here: at K=3, lambda=10,000,
MAE rises to **51.023 W**. The training residual median is 189.933 W versus
284.450 W in validation. The nearest-background category occupancy changes
from approximately 67.74%/30.41%/1.85% in training to 50.41%/46.49%/3.10%
in validation. This observed distribution shift is a plausible reason that
a stationary background-frequency penalty misallocates demand; it does not
identify a unique causal explanation for the shift.

## 5. Lower MAE is not sufficient: rare active loads get missed

Activity proxies use train-centroid midpoint thresholds: dryer 211.016 W,
fridge 53.177 W, vacuum 608.872 W. They are not manual event annotations.
Validation contains only **112 active dryer blocks across two days**, 14,030
active fridge blocks across all ten days, and **zero active vacuum blocks**.

| Candidate | Dryer active recall | Dryer false-positive blocks | Active-dryer MAE (W) | Overall MAE (W) |
|---|---:|---:|---:|---:|
| Current objective | 72.32% (81/112) | 2,661 | 563.808 | 36.480 |
| Appliance-only prior, K=3, lambda=10,000 | 37.50% (42/112) | 815 | 592.934 | 18.685 |
| All-channel prior, K=12, lambda=10,000 | 23.21% (26/112) | 13 | 644.600 | 16.140 |
| Integrated background density + prior | 21.43% (24/112) | 6 | 759.775 | 13.707 |

The best overall MAE partly comes from removing large false activations during
the many inactive periods. Dryer F1 can improve because false positives fall
so much, even while recall and active-period MAE worsen. Neither MAE nor F1
alone conveys the full trade-off. There is no estimate of active-vacuum recovery;
its active-period MAE and recall are undefined, not zero.

The same twelve-background-state quadratic prior improves fridge active recall from 64.98%
to 86.75%, but increases its false-positive blocks from 5,778 to 9,921. A single
global prior coefficient does not deliver uniformly better operating behaviour.

## 6. Temporal continuity helps some errors, but is not sufficient

| Exact temporal penalty | Background switching penalized? | MAE (W) |
|---|---|---:|
| rho=0.02 | No | 28.422 |
| rho=0.02 | Yes | 27.530 |
| rho=0.1 | No | 27.983 |
| rho=0.1 | Yes | 25.239 |

These use the current three background states and no state-frequency priors.
The best temporal candidate reduces dryer false activations to zero, but
fridge MAE rises from 57.251 to 62.700 W and active-dryer MAE rises to 803.160 W.
This is full-day offline exact decoding, with future mains information inside
each day and resets at gaps/day boundaries. It is neither a causal online
method nor the paper's two-interval QAOA circuit. Prior and temporal penalties
were tested separately; their combination was not evaluated here.

## 7. Ambiguity requires a physically meaningful definition

The raw target-category optimum agrees with the nearest-centroid labels on
only 9,938/28,800 blocks. However, the dryer has two nearly identical low-power
centres (0.003 and 0.980 W). Treating them as distinct physical operating modes
overstates meaningful category error and near-optimal ambiguity.

A post-hoc tolerance check finds that the nearest-centroid target configuration,
allowing alternatives within 2 W in every appliance, is among the aggregate-cost
minimizers on 15,955 blocks (55.40%). On the remaining 44.60%, a meaningfully
different target assignment is preferred by the objective. These are
cost-ranking diagnostics, not probabilistic confidence scores.

Raw near-optimal alternatives occur on 92.62% of blocks, but an alternative
within 10 W of the best absolute mains residual that differs by at least 25 W
in an appliance occurs on only **2,216 blocks (7.69%)**. The distinction matters:
the main problem is not simply many nearly tied answers. Frequently the cost
ranks a wrong appliance assignment as better.

## Recommended next experiment — not performed here

1. Keep the existing model as the baseline. First test an appliance-prior cost
   within the current 12-qubit representation; do not increase circuit size
   merely because a larger classical model improves one validation average.
2. Add explicit active-load safeguards to model selection: predeclare acceptable
   missed-activity/false-activation trade-offs and report active/inactive MAE,
   not only average MAE. The present lowest-MAE candidate should not be adopted
   unchanged as an all-purpose improvement.
3. Investigate learned state-transition likelihoods and background dynamics with
   exact classical inference, addressing the observed rare-load suppression and
   stationary-background-prior failure. Test interactions rather than assuming
   the separately observed gains add together.
4. Freeze a chosen design and reserve a genuinely unused cohort with a
   prespecified activity-coverage policy before making generalization claims.
   Current validation is development data; the previous twenty test days are
   already exposed. Do not select replacement days after observing performance.
5. Only after the objective passes those checks, retrain the small QAOA circuit
   and compare ideal, noisy and hardware results. These classical findings do
   not demonstrate quantum advantage or a repaired IBM evaluation.

## Reproducibility and verification

Archive: [background_objective_001](../results/quantum_diagnostics/background_objective_001/).
Plan SHA-256: `e337a1d63959d3fcf17484eaada39598d57f7de522229ca40698b5c06aad96f1`.

All 434 repository tests passed, including fourteen new diagnostic tests. The
independent audit passed: 260 static/mixture method-day results, forty temporal
global optima using a separate full transition-matrix solver, seventy control
method-day results, and fourteen independently reaggregated source windows
(four training days plus all ten validation days; 40,320 blocks). All ninety
training-window checksums matched the original archive. Background cluster
stationarity, frequencies and variances were checked independently; k-means
initialization was not independently rerun. Maximum prediction discrepancy
was 2.05e-12 W and maximum total temporal-objective discrepancy 1.94e-7 W².

- [Frozen candidate grid and inputs](../results/quantum_diagnostics/background_objective_001/plan.json)
- [Training-only fitted background models and priors](../results/quantum_diagnostics/background_objective_001/fitted.json)
- [Every candidate, control and per-day metric](../results/quantum_diagnostics/background_objective_001/results.json)
- [Independent audit](../results/quantum_diagnostics/background_objective_001/independent_audit.json)
- [Post-hoc activity/ambiguity interpretation](../results/quantum_diagnostics/background_objective_001/interpretation.json)

Notebook packages are unavailable in this environment. The inspectable
calculation trail is therefore provided in the existing executable experiment
format, with archived inputs/predictions and a separate independent audit;
no executed notebook is claimed.

To repeat the fixed development comparisons in the same verified environment,
create a new named folder and copy only this archive's `plan.json` into it.
Run `scripts/investigate_background_objective.py` with `--folder` set to that
folder and `--mode verify`, `extract`, `fit`, then `evaluate`, in that order.
Use `PYTHONPATH=src:. .venv/bin/python`. Then run
`scripts/audit_background_objective.py` and
`scripts/summarize_background_objective.py` with the same `--folder`.
Outputs refuse overwrites. Use `freeze` only to record a new diagnostic plan;
repeating this validation cohort is reproducibility work, not a fresh test.
