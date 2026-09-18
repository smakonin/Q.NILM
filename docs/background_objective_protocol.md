# Background and objective diagnostic — development data only

Scope: explain the static aggregate-objective/appliance-MAE gap. All work is
classical; no quantum provider access, production changes, paper changes,
or final-test evaluation. These are exploratory development comparisons.
The earlier twenty test days are already exposed and are not loaded here.

## Data and frozen comparison policy

- Re-extract the original ninety timestamp-selected training days. Verify their
  source-window hashes against the original campaign. Keep every accepted
  complete thirty-second block, excluding marked/nonfinite/missing blocks by
  the original policy; do not replace windows.
- Reuse only the ten validation days from `training_validation_001`, verifying
  their recorded hashes. These days are reused development data, not a new test.
- Keep the original train-only dryer/fridge/vacuum centroids fixed. Refit three
  background centres on the same training data to check exact lineage.
- Freeze this protocol, code and candidate grid before computing new validation
  comparisons. The selection statistic is three-appliance measured-power MAE.
  Retain all candidates, per-day metrics, predictions and activity-conditioned
  diagnostics. A validation winner is only a candidate for later testing.

## Comparisons

1. Background quantization: deterministic training-only k-means with requested
   counts 1, 3, 6 and 12, preserving signed residuals. Report nonconvergence as
   a failed candidate, not a silent substitution. Each fixed-state objective
   is solved by enumeration of all 24 times K assignments.
2. State plausibility: add Laplace-smoothed independent training marginal state
   priors to squared residual, with coefficients 100, 1,000 and 10,000 W².
   Compare all-channel priors at each K. At K=3 separately compare appliance-only
   and background-only priors at all three coefficients. This remains a
   quadratic one-hot-compatible objective, with extra unary terms only.
3. Continuous background likelihood: at K=3 and K=12 integrate a Gaussian
   mixture whose weights, cluster means and standard deviations come only from
   training residuals. Minimum cluster standard deviation is 1 W. Compare no
   appliance prior and unit-weight appliance log-prior. Exhaustively score all
   24 appliance assignments. This diagnostic likelihood is not claimed to be
   a quadratic one-hot cost or directly hardware-ready.
4. Temporal continuity at K=3: exact gap-safe dynamic programming with change
   penalties rho times each channel's training power range squared, rho=0.02
   and 0.1. Compare background change penalty zero versus the same scaling.
   No compression, no continuity across missing blocks/day boundaries, no state
   carry, and no state priors in these four arms. Full-day offline decoding
   uses future mains within each day and is not a causal online algorithm or
   the original two-interval QAOA task.

The squared- and absolute-residual static objectives have identical minimizers
without additional terms; changing squared to absolute loss alone cannot resolve
the ambiguity. More flexible unpenalized background does not provide additional
observations: with continuous nonnegative background, every target combination
below mains can achieve zero aggregate residual.

## Label-dependent controls (not deployable methods)

Using validation labels only for diagnosis, compare existing K=3 inference on:
raw mains; quantized target powers plus true residual background; measured target
powers plus nearest-centroid background; and quantized targets plus quantized
background. The last control lies exactly in the assumed model space and must
recover its categories absent exact aliases. Also compare true-background removal
(measured target sum), and the nearest-target-centroid representation oracle.
These paired interventions isolate model-space discrepancies on this cohort;
their MAE differences are not assumed additive causal contributions.

## Metrics and validation

Primary: equal-weight mean dryer/fridge/vacuum absolute error against measured
watts over all valid blocks. Also report per-appliance MAE, aggregate residual,
category confusion, ground-truth target cost rank, near-optimal alternatives,
background residual quantization, and active/inactive errors. Activity proxies
are midpoint thresholds between dryer states 1/2, fridge states 0/1 and vacuum
states 0/1 (zero-based). They are power-state proxies, not event ground truth.

Record the training/validation residual distributions and active-state coverage.
Do not claim causality from a correlation, generalization from reused validation,
physical-circuit repair from exact classical inference, or quantum advantage.
Unit tests plus a separately implemented exact-score/data-check pass verify key
results. Save an inspectable analysis companion without altering old archives.
