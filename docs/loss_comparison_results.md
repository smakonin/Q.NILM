# Matched-budget training-loss comparison

Completed 14 September 2026. Offline six-qubit synthetic experiment; no QPU
jobs, production-parameter changes, paper edits, or replacement held-out results.

## Conclusion

The original mean loss is not the only limitation. Tail-focused training helps
the minimum-sampled-cost decoder within the original angle bounds, but widening
the gamma range greatly improves mean-loss training too. Finite-shot selection
introduces another failure mode: a zero estimated tail loss need not indicate
a good underlying distribution. No single loss dominates the training case,
synthetic transfer checks, and all training configurations.

The diagnostic workflow kept the comparison and budgets fixed before running,
retained every paired trial, and separated training performance from transfer
and physical-noise-model performance. No settings were selected using transfer
labels. These findings refine, rather than confirm wholesale, the earlier
[exploratory objective review](training_objective_review.md).

## What was compared

The unchanged p=1 circuit has six qubits, two three-state loads with levels
[0,400,900] and [0,120,280] W, and nine feasible assignments. The training
aggregate is 680 W; the unique correct assignment is 400+280 W. This single
interval has no active temporal penalty. Appliance truth is evaluation-only.

Five losses each received five paired trials and 1,024 evaluations per trial,
using the same initial candidate pools and two-restart bounded-Powell policy.
Adaptive search points may differ between losses. Mean averages all costs;
CVaR25/50 averages the lowest 25%/50% probability mass; best-of-4/16 minimizes
the expected minimum cost in a batch of four/sixteen samples. All use the same
positive coefficient normalization. Raw numerical values of these different
losses should not be compared as if they were the same metric.

| Configuration | Training observations per evaluation | Gamma range | Fits |
|---|---|---|---:|
| Exact/original | Exact ideal distribution | [0,8π] | 25 |
| Sampled/original | 256 classically simulated ideal measurements | [0,8π] | 25 |
| Exact/wider | Exact ideal distribution | [0,16π] | 25 |

Beta remains in [0,π]. Both the wider range and sampled-training arms were
specified before execution. Total: 75 fits, 76,800 training evaluations,
6,553,600 simulated training measurements, and zero QPU usage. Equal evaluation
counts are a search-budget control, not a claim of equal computation time.

## Primary result: actual mean cost versus returned answer

The following values are equal-weight means over all five trials. Expected
best-16 cost is the squared aggregate residual of the lowest-cost answer among
16 independent samples. Expected decoded MAE averages absolute errors over the
two loads for that selected answer. All ideal samples are feasible. Neither
metric is a new real-data held-out MAE.

| Training loss, exact/original | Mean single-sample cost (W²) | Correct assignment per shot (%) | Expected best-16 cost (W²) | Expected decoded MAE, 16 shots (W) |
|---|---:|---:|---:|---:|
| Mean | 113,547.9 | 10.601 | 4,929.2 | 20.157 |
| CVaR25 | 160,037.9 | 24.628 | 409.2 | 2.647 |
| CVaR50 | 158,721.5 | 20.440 | 2,775.6 | 9.729 |
| Best-of-4 | 165,785.6 | 23.307 | 504.2 | 2.948 |
| Best-of-16 | 159,661.4 | 24.745 | 384.1 | 2.383 |
| Uniform feasible control, no training | 161,555.6 | 11.111 | 4,346.4 | 17.346 |

Mean-loss training succeeds at its own criterion: its mean cost is much lower.
However, this does not necessarily improve the answer returned by a best-of-batch
decoder. Best-of-16, best-of-4 and CVaR25 beat mean on best-16 cost in all five
original-bound exact trials. CVaR50 does so in four of five: its fifth fit has
12,212.4 W² best-16 cost and is retained in the average, not discarded.

### Bounds and finite-shot sensitivity

| Training loss | Exact/original best-16 cost (W²) | Sampled/original best-16 cost (W²) | Exact/wider best-16 cost (W²) |
|---|---:|---:|---:|
| Mean | 4,929.2 | 4,588.7 | 75.7 |
| CVaR25 | 409.2 | 662.3 | 220.7 |
| CVaR50 | 2,775.6 | 640.7 | 60.0 |
| Best-of-4 | 504.2 | 927.0 | 60.0 |
| Best-of-16 | 384.1 | 3,662.9 | 59.9 |

| Training loss | Exact/original decoded MAE (W) | Sampled/original decoded MAE (W) | Exact/wider decoded MAE (W) |
|---|---:|---:|---:|
| Mean | 20.157 | 19.167 | 0.559 |
| CVaR25 | 2.647 | 4.124 | 1.437 |
| CVaR50 | 9.729 | 3.904 | 0.404 |
| Best-of-4 | 2.948 | 5.207 | 0.410 |
| Best-of-16 | 2.383 | 23.080 | 0.407 |

All decoded MAEs above use 16 evaluation shots, regardless of whether training
uses exact probabilities or 256 measurements per call.

**Angle-range limitation.** Wider-bound mean training raises correct-assignment
probability from 10.601% to 32.744% and reduces best-16 cost by 98.464%, to
75.7 W². Its actual mean cost falls from 113,547.9 to 108,713.0 W². The new gamma
is approximately 33.057, outside the former upper bound 8π≈25.133. All five
exact/original best-of-16 fits hit that former upper bound. This supports an
angle-domain limitation on this case; it does not certify the global optimum
of either search domain or establish that still wider bounds are unnecessary.

**Finite-shot selection limitation.** All five sampled best-of-16 fits select
an estimated training loss of zero. Each estimate averages only sixteen
independent groups of sixteen measurements. A batch can contain an optimum in
every group even though another batch would not. Selecting the minimum estimate
over 1,024 evaluations rewards such lucky batches. The independently evaluated
best-16 costs are 1,437.4–5,528.4 W², not zero. This is consistent with selection
optimism and flat/noisy search feedback; the experiment does not isolate their
individual contributions or show that all finite-shot best-of-K methods fail.
Four sampled CVaR25 fits also select zero estimated loss. In exact/wider CVaR25,
all five losses are genuinely zero because at least 25% of mass reaches the
zero-cost optimum, leaving the criterion unable to distinguish those candidates.

Sampled CVaR50 gives 640.7 W² mean best-16 cost (trial range 408.3–1,061.3), versus
4,588.7 (3,250.9–6,613.9) for sampled mean. CVaR25/50 and best-of-4 improve that
cost over mean in all five paired sampled trials; best-of-16 does so in three.
These ranges are seed variation, not confidence intervals or significance tests.
The fixed Powell policy is not claimed to be an optimal finite-shot optimizer.

**Objective versus appliance error.** Sampled best-of-16 has lower average
best-16 aggregate cost than sampled mean, but higher decoded appliance MAE
(23.080 versus 19.167 W). This illustrates why cost improvement alone is
insufficient: different incorrect load assignments can have different appliance
errors despite similar aggregate residuals.

At 256 evaluation shots the primary ideal problem largely saturates: even the
uniform control hits its optimum with probability approximately
0.99999999999992. The 16-shot results expose sampling efficiency on a tiny task,
not a practically established quantum advantage. Exact classical enumeration
checks just nine feasible assignments and returns the zero-cost correct answer.

## Synthetic transfer checks

Without retraining, the same angles were evaluated on the other eight aggregate
values from these two load models. Average the eight case expectations within
each trial, then average the five trials. This is not independent household or
real-data validation, and these results were not used to retune the fits.

| Training loss | Exact/original transfer MAE (W) | Sampled/original transfer MAE (W) | Exact/wider transfer MAE (W) |
|---|---:|---:|---:|
| Mean | 19.476 | 19.473 | 17.988 |
| CVaR25 | 17.312 | 21.225 | 17.348 |
| CVaR50 | 28.273 | 19.592 | 14.620 |
| Best-of-4 | 16.874 | 16.807 | 15.030 |
| Best-of-16 | 16.045 | 21.181 | 14.838 |
| Uniform feasible control | 16.170 | 16.170 | 16.170 |

All entries use 16 evaluation shots. Training-case gains do not consistently
transfer. For example, sampled CVaR50 improves the 680 W case substantially but
does not improve the eight-case mean MAE over sampled mean; neither beats the
uniform control here. It would be premature to choose a production loss from
these results alone.

## Archived IBM calibration model, not hardware

Only exact/original fits were evaluated with the saved Fez noise model: all
25 fits plus a logical-uniform circuit, on layouts A (128–133) and B (140–145).
The snapshot was retrieved at 16:30:07 UTC on 14 September 2026. The model
includes the established approximate gate, idle and readout errors, but not
drift, leakage, crosstalk, or an actual pulse schedule. Another 26 simulations
used the fixed common generic model on layout A: 78 noisy simulations in total.
Sampled/original and exact/wider fits have no device-noise evaluation here.

| Loss | A: raw valid (%) | A: optimum hit in 16 (%) | A: conditional MAE (W) | B: raw valid (%) | B: optimum hit in 16 (%) | B: conditional MAE (W) |
|---|---:|---:|---:|---:|---:|---:|
| Mean | 72.264 | 72.588 | 39.925 | 79.590 | 76.038 | 33.237 |
| CVaR25 | 72.183 | 94.931 | 13.410 | 79.580 | 96.423 | 9.096 |
| CVaR50 | 72.511 | 85.252 | 19.241 | 79.885 | 87.059 | 15.275 |
| Best-of-4 | 72.175 | 93.986 | 14.162 | 79.573 | 95.653 | 9.715 |
| Best-of-16 | 72.184 | 95.015 | 12.570 | 79.580 | 96.492 | 8.450 |
| Routed zero-angle uniform control | 78.796 | 77.658 | 30.033 | 84.955 | 79.833 | 25.789 |

Conditional MAE excludes batches with no feasible sample; they are abstentions,
not repaired or assigned a fabricated error. Their probability is explicitly
saved for every row and is at most 1.296×10^-9 for 16 shots in these simulations.
Optimum-hit probabilities use the full raw budget and do not condition away
invalid shots. All five trials contribute equally; the uniform control is one
untrained circuit per layout, not five independent trials.

Tail-focused training improves the modeled chance of obtaining the correct
answer, but does not restore one-hot feasibility. Mean versus best-of-16 raw
validity is essentially unchanged on both layouts. This separates improving
the distribution within the useful state space from repairing physical errors.

### Physical-control qualification found during review

The frozen protocol calls its logical-uniform circuit “W-only,” but the runner
constructs a full circuit with gamma=beta=0 before compilation. Although its
ideal distribution is verified to be uniform, the compiled control retains
39 CZ gates and depth 62. The earlier component-only W preparation has 12 CZ
gates and depth 27. Thus this run is **not a minimal physical W-only baseline**;
its additional compilation/routing exposure must not be attributed to state
preparation alone. Earlier W-only modeled validity was 85.781%/92.369%, rather
than this control's 78.796%/84.955%. Neither archive was altered to hide this
implementation difference. The ideal classical-uniform comparison is unaffected.

All trained circuits have 63 CZ gates and depth 134 except the fifth CVaR50
trial (gamma=0), which compiles to 55 CZ gates and depth 117. Compilation
policy was fixed, but realized physical resources therefore are not identical
across every selected parameter set. Mean/best-of-16 resource comparisons are
matched. The independent audit verifies the circuits actually saved, not the
claim that this control is a minimal W-only implementation.

## Verification and reproduction

- [Frozen protocol](loss_comparison_protocol.md)
- [Plan and source/input hashes](../results/quantum_diagnostics/loss_comparison_001/plan.json)
- [Complete results](../results/quantum_diagnostics/loss_comparison_001/summary.json)
- [All aggregated values and five-trial ranges](../results/quantum_diagnostics/loss_comparison_001/comparison_tables.json)
- [Independent audit](../results/quantum_diagnostics/loss_comparison_001/independent_audit.json)

Plan SHA-256: `d1f538c22dddda552868885d9a8056d97b4c650ced4c27259a58e229e2f37c79`.
The full software suite passed 408 tests, including 11 added tests. An
independent implementation recomputed all 76,800 training losses, reproduced
all 6,553,600 sampled training measurements, checked 684 ideal metric rows,
verified 52 compiled ideal circuits, and reproduced 78 noisy distributions.
Maximum training-loss discrepancy was 8.68×10^-17; maximum noisy-probability
discrepancy was 3.34×10^-14. Optimizer trajectories were audited, not rerun by
a second optimizer. The audit establishes numerical consistency, not hardware
accuracy, statistical significance, or generalization.

To reproduce, choose a fresh archive name under results/quantum_diagnostics;
the completed archive refuses overwriting. From the repository root:

```sh
PYTHONPATH=src:. .venv/bin/python scripts/run_loss_comparison.py --mode freeze --output-dir results/quantum_diagnostics/loss_comparison_new
PYTHONPATH=src:. .venv/bin/python scripts/run_loss_comparison.py --mode run --output-dir results/quantum_diagnostics/loss_comparison_new
PYTHONPATH=src:. .venv/bin/python scripts/audit_loss_comparison.py --folder results/quantum_diagnostics/loss_comparison_new
PYTHONPATH=src:. .venv/bin/python scripts/summarize_loss_comparison.py --folder results/quantum_diagnostics/loss_comparison_new
```

## Next decision

Keep mean loss as the production baseline for now. A follow-up should compare
mean and CVaR50 with the wider angle domain, finite-shot stability checks and
independent validation, including an independent resampling step for selecting
among promising candidates. Select settings on training/validation windows,
then evaluate on untouched real NILM windows with identical decoding budgets.
Device-noise evaluation of the wider/sampled fits remains necessary before
proposing a hardware comparison. Any future physical-uniform control should
compile the component-only W circuit explicitly. None of these follow-ups is
claimed complete here. The paper and previous IBM results remain unchanged.
