# Stage D: duration and transition-penalty components

This additive, retrospective exact-DP programme uses the original frozen R1Hz
cohort: 30 test windows, 86,396 valid 30-second blocks, and the same training-only
four-channel multistate/background model. No existing archive or model changes.
The machine-readable protocol and source snapshots are frozen before inference
in `results/stage_d/components/run_001/`.

## Four fixed arms

| Arm | Reconstruction weights | Per-channel transition penalty |
|---|---|---|
| Original | Actual interval durations | `rho * range_i^2` |
| Remove duration weights | One per compressed interval | `rho * range_i^2` |
| Global maximum | Actual interval durations | `rho * max_i(range_i^2)` |
| Global mean | Actual interval durations | `rho * mean_i(range_i^2)` |

The model's selected compressed-multistate `rho` must equal 0.1 or preparation
stops. All four arms retain the same centroids, aggregate-only event boundaries,
source exclusions, state-proxy thresholds and chronological windows. No new
training or validation selection occurs. The global-mean control preserves the
sum of per-channel penalty coefficients; the global-maximum control implements
the single-largest-power scaling discussed in the manuscript. This guards
against attributing an overall increase in penalty magnitude solely to removal
of channel normalization.

Two-interval archived chunks are reassembled into their complete valid runs.
Each run is solved exactly with categorical temporal DP. No transition crosses
a window boundary or missing-data gap. **Actual durations always determine
prediction expansion and scoring**, including the unit-weight arm.

## Evaluation and checks

Inference receives no appliance-reference values; each trace is saved before
scoring. Source references are cached after the baseline pass and never passed
to subsequent arms' inference. The baseline
must reproduce the archived full-run DP macro MAE, 26.0414278901 W within
1e-8 W, plus per-window appliance absolute-error sums and aggregate errors.
All 30 baseline windows must pass before an ablated arm is executed.

Primary endpoint: pooled macro appliance MAE for dryer, fridge and vacuum,
on every original valid block. Secondary endpoints retain the existing
per-appliance MAE/RMSE, energy errors, high-state-proxy counts and event counts,
aggregate MAE, switching counts, and separately measured local solver times.
High-state proxies remain distinct from verified physical ON states.

Every trajectory is also cross-scored under the **original duration-weighted,
channel-normalized objective**, adding the within-interval raw-block residual
constant. Direct raw-block squared error must equal weighted interval-mean
squared error plus that constant. This shared objective gap is measured from
the reproduced exact baseline. Native segment-objective values are archived
separately and are not compared as though their definitions were identical.

Each ablated arm versus baseline, and global maximum versus global mean,
receives a paired percentile bootstrap interval: 2,000 draws of the same 30
window IDs, seed 7031, ratio of resampled error sums to three times resampled
block counts. These are descriptive within-home intervals, not multiplicity-
adjusted significance tests or cross-home confidence. All four arms and all
prespecified contrasts are retained, irrespective of direction.

## Reproduction and limits

```sh
python scripts/run_stage_d_components.py --mode prepare
python scripts/run_stage_d_components.py --mode run
python scripts/run_stage_d_components.py --mode summarize
```

Preparation refuses an existing directory. Execution verifies all frozen input
and code hashes and only resumes missing checkpoints; existing records are not
silently overwritten. Source window hashes and valid-block coverage are checked
against the original campaign. Unit tests cover direct objectives, exhaustive
small-case optima, expansion, resets, penalty definitions and paired estimators.

This tests fixed-model components on already examined data, not a newly blind
evaluation. Removing duration weights changes the reconstruction-to-transition
balance; no alternative receives fresh tuning. Results therefore do not show
which formulation would win after independent optimal tuning. These are
classical objective/modeling ablations, separate from the categorical mixer
comparison, external validation, quantum hardware evidence and quantum advantage.
