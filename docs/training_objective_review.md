# Training-objective review

14 September 2026. Offline diagnosis only. Production objectives, angles,
circuits, the paper and previous experiment archives are unchanged.

## Conclusion

The current loss minimizes the average objective of a sampled state, whereas
the decoder returns the lowest-objective feasible state observed in a batch.
On the six-qubit example these criteria prefer different angles. A shared
16,642-candidate screen confirms that the existing fit is best for the mean
within that candidate set, while another candidate on the same p=1 circuit
is substantially better for best-of-16 selection. This is evidence of a
training/decoding mismatch, not proof that loss choice explains IBM Run 1.

## Three distinct objectives

1. **Problem cost.** The categorical NILM objective is

   `C(s) = sum_t w_t (y_t - sum_i P_i(s_ti))² + sum_ti lambda_i 1[s_ti != s_(t-1)i]`,

   with a preceding inferred-state boundary penalty when provided. It rewards
   aggregate reconstruction and discourages state changes; it is not appliance
   MAE. The six-qubit single-interval instance has no active temporal term.
2. **Angle-training loss.** Stage B and the six-qubit diagnostic minimize
   `J_mean(theta) = sum_s p_theta(s) C(s) / scale`. The full held-out angle
   trainer averages this normalized expectation over training instances.
   Scale is positive and coefficient-derived, not truth-derived. For a fixed
   problem it does not change the ranking of mean-cost candidates; across
   instances it sets their relative weighting.
3. **Returned-answer rule.** Inference selects the minimum objective among
   actually observed feasible states, with a fixed tie rule. The hardware
   campaign has a declared fallback when none are feasible. Therefore the
   decoder is not returning an average sample or the most probable state.

Relevant code: [mean-loss search](../src/quantum_nilm/stage_b.py),
[held-out training and decoding](../scripts/run_quantum_heldout.py),
[cost and feasible sampling](../src/quantum_nilm/categorical_qaoa.py), and
[hardware sample scoring](../src/quantum_nilm/categorical_ibm.py).

## Why the average improves without improving exact hits

The primary example has aggregate 680 W and true powers 400 + 280 W.
Uniform sampling assigns probability 11.111% to each of nine combinations.

| Predicted powers (W) | Cost (W²) | Trained probability | Contribution to mean-cost reduction (W²) |
|---|---:|---:|---:|
| 0 + 0 | 462,400 | 3.860% | +33,527.383 |
| 400 + 0 | 78,400 | 12.752% | −1,286.697 |
| 900 + 0 | 48,400 | 9.325% | +864.335 |
| 0 + 120 | 313,600 | 0.679% | +32,715.574 |
| 400 + 120 | 25,600 | 10.055% | +270.419 |
| 900 + 120 | 115,600 | 28.343% | −19,920.037 |
| 0 + 280 | 160,000 | 19.158% | −12,874.934 |
| 400 + 280, exact solution | 0 | 10.601% | 0 |
| 900 + 280 | 250,000 | 5.226% | +14,711.649 |

Each contribution is `(1/9 - trained_probability) * C(s)`; the unrounded
contributions sum to 48,007.692 W². Suppressing very costly outcomes produces
large gains in the mean despite probability shifting to some mediocre answers.
The zero contribution of the zero-cost state is an accounting identity, not
proof that its probability cannot affect the loss: probabilities must sum to
one, so probability moved into that state must leave other states.

## Exploratory same-candidate loss comparison

All losses rank the same 129 × 129 grid (gamma 0 to 8π, beta 0 to π, endpoints
included), plus the archived fitted angles. No local refinement or noise
optimization was done. Within 1e-8 W² of a minimum, ties use lowest mean cost,
then first index. This larger screen is not a matched-budget comparison with
the earlier 1,024-call training run or a proof of continuous global optimality.

| Selection criterion | Mean cost (W²) | Exact solution per shot | Expected best cost, 16 shots (W²) | At least one exact solution, 16 shots |
|---|---:|---:|---:|---:|
| Uniform feasible sampling | 161,555.556 | 11.111% | 4,346.404 | 84.810% |
| Current mean-cost fit / best mean in screen | 113,547.863 | 10.601% | 4,929.172 | 83.354% |
| Lowest CVaR, best 25% of probability mass | 158,798.570 | 24.825% | 396.678 | 98.960% |
| Lowest expected best-of-16 cost | 159,342.889 | 24.803% | 385.417 | 98.955% |

These are exact noiseless model calculations, not observations from 16-shot
hardware jobs. The two alternative criteria accept a worse average in exchange
for better chances of a useful best sample. Their gamma values lie at the
upper search bound, so bound sensitivity remains untested.

The current loss does help at K=2: expected best cost is 66,646.782 W² versus
82,582.716 W² for uniform. It loses that advantage by K=4 on this instance.
At K=256, current and uniform sampling both almost surely include the optimum;
their tiny nonzero failure probabilities do not demonstrate a useful practical
advantage. Small K here exposes the distribution difference, not a decision to
replace the hardware campaign's declared raw-shot budget.

## Candidate replacement losses

**Expected best-of-K** matches the ideal decoder directly:

`J_K(theta) = E[min(C(S_1), ..., C(S_K))]`, with iid `S_k ~ p_theta`.

For increasing distinct costs `e_1,...,e_m` and `F_j = Pr(C <= e_j)`, the exact
small-instance calculation is

`J_K = e_1 + sum_(j=1)^(m-1) (e_(j+1)-e_j) (1-F_j)^K`.

It does not require appliance truth labels to rank sampled costs. On larger
systems, estimating it requires repeated sample batches and accounting for all
training shots. Large K can make the landscape nearly flat on easy instances.

**Lower-tail CVaR** averages the lowest-cost alpha fraction of probability
mass (including fractional mass at a discrete quantile). It is a published
variational-optimization approach, not a new Q.NILM invention:
[Barkoutsos et al., Quantum 4, 256 (2020)](https://quantum-journal.org/papers/q-2020-04-20-256/).
The [official Qiskit tutorial](https://qiskit-community.github.io/qiskit-optimization/tutorials/08_cvar_optimization.html)
describes the corresponding best-shot aggregation.

Do not simply set alpha=0.1: uniform and the archived fit already have more
than 10% probability on the zero-cost optimum, so both have CVaR_0.1 = 0.
In this screen 5,361 candidates reach the minimum within tolerance. This loss
cannot distinguish their exact-hit probabilities once the threshold is met.
Alpha=0.25 and 0.5 remain informative here, but need validation across instances.
Maximizing known-optimum probability was also calculated as an oracle diagnostic
only; it is not a deployable loss that silently knows the answer beforehand.

## Appliance accuracy and noisy feasibility remain separate

Matching aggregate power is not equivalent to recovering each appliance.
Even in this toy instance, 900 + 0 W has lower residual cost (48,400 W²) than
0 + 280 W (160,000 W²), but worse appliance MAE (390 W versus 200 W relative
to 400 + 280 W). Positive and negative appliance errors can offset in the sum.
Thus a better best-sampled aggregate objective still needs appliance-level
validation; changing the aggregation loss alone does not resolve identifiability.

The independent audit additionally evaluates expected decoded appliance MAE
using the known synthetic truth only after angle selection. At 16 shots it is
17.346 W for uniform, 20.157 W for the current fit, 2.602 W for CVaR_0.25 and
2.436 W for best-of-16. These are synthetic expected errors, not replacements
for the paper's held-out MAEs. CVaR_0.5 has a slightly different cost/MAE tradeoff,
further illustrating that these metrics are not interchangeable.

On hardware, invalid outcomes and no-feasible-sample batches must count. A
loss conditioned only on surviving valid shots can look good while feasibility
collapses. Use the declared raw-shot budget and explicit failure/fallback
handling, and report raw validity separately. The noiseless best-of-K formula
above does not by itself model the earlier IBM invalid-shot problem.

## Recommended next controlled comparison — not yet performed

Keep the current mean-loss arm as a baseline. Compare it with lower CVaR
(validated alpha choices) and expected best-of-K using the same circuit family,
training instances, optimizer/restart budgets, raw shots, decoder and noise
conditions. Include uniform sampling and exact classical answers for evaluation.
Fit/select settings only with training/validation data and assess any claimed
generalization on fresh held-out windows; do not tune on their appliance labels.
Check gamma bounds and finite-shot stability, retain the all-invalid cases,
and count classical optimization plus quantum sampling costs. First establish
a reproducible benefit locally, then propose a bounded hardware comparison.

No hardware job, final-loss change, paper update or publication claim is
authorized or implied by this diagnostic review.

## Reproduction and checks

- [Full screen and state decomposition](../results/quantum_diagnostics/training_objective_001/summary.json)
- [Independent audit](../results/quantum_diagnostics/training_objective_001/independent_audit.json)
- [Saved candidate distributions](../results/quantum_diagnostics/training_objective_001/grid.npz)

The audit recomputed six losses over all 16,642 candidates, re-evaluated
best-of-K with an independent winning-state formula, checked fractional CVaR,
and verified all nine selected/reference entries using the full Qiskit circuit.
Maximum physical probability discrepancy was 5.28e-16. The batch simulator also
matched 20 scalar reference evaluations to 2.23e-16. Source and input hashes
are recorded; no completed archive is overwritten.

```sh
PYTHONPATH=src:. .venv/bin/python scripts/analyze_training_objective.py --output-dir results/quantum_diagnostics/training_objective_new
PYTHONPATH=src:. .venv/bin/python scripts/audit_training_objective.py --folder results/quantum_diagnostics/training_objective_new
```
