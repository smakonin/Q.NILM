# Stage G: frozen-setting REFIT external validation

This is an external, multi-home, **within-home calibrated** evaluation on the
official cleaned REFIT release (DOI 10.15129/9ab14b0e-19ac-4279-938f-27f643078cec).
It is not zero-shot unseen-home inference, quantum hardware execution, or a
quantum-advantage test. No REFIT test outcome is used to choose these settings.

## Cohort and source

The source README identifies washing-machine, cooling, and dishwasher channels
in 14 eligible homes: 1, 2, 3, 5, 6, 7, 9, 10, 11, 15, 16, 18, 20, and 21.
House 14 is absent from the release. Homes 4, 8, 17, and 19 have no named
dishwasher; house 12 has unidentified appliance channels; house 13's only
identified cooling channel stops in August 2014. These are metadata exclusions,
not exclusions for accuracy or activity. The fixed channel map is in the runner
and frozen JSON. Known appliance replacements in eligible homes remain included.
Cooling denotes one identified fridge, fridge-freezer, or freezer, not the sum
of every cooling appliance. House 18 uses its indoor fridge-freezer; house 11
uses its fridge-freezer rather than a fridge with a known late installation.

The official archive, README, source DOI, license, and SHA-256 checksums are
recorded in `data/external/refit/source.json`. Source CSV hashes and timestamp
boundary-only manifests are frozen before training values are read. Selection
uses elapsed calendar time, not activity. Within each home's 60/20/20 calendar
partitions, select 30 evenly spaced, non-overlapping 24-hour training windows and
30 test windows; leave the middle 20% unused because no REFIT hyperparameter
selection is performed. No failed, inactive, or low-quality window is replaced.
If a source endpoint is fractional, ceil the first and floor the last timestamp
to use an interior integer-second support range; preserve both raw endpoints.
Before summary, require all 420 quality-window records and exactly nine
method/seed score rows per nonempty window, none for empty windows. An entirely
unevaluable cohort produces explicit null outcomes, not a reduced silent cohort.

## Data contract and limitations

Interpret the cleaned release's Unix column as seconds. Integrate left-held
reported power only between consecutive source rows separated by at most 16 s.
Do not extrapolate past the last row or bridge larger gaps. A segment with a
flagged Issues value or nonfinite/negative requested power is invalid. Retain
only 30-second blocks with complete valid support, and restart all temporal
inference at missing blocks and day boundaries. Report raw-row and block
exclusions and coverage, including empty windows. This is bounded resampling of
the published cleaned data, not recovery of unobserved original measurements.

Crucially, the official README states that REFIT cleaning forward-fills NaNs,
replaces IAM spikes above 4,000 W with zero, and that an earlier database
iteration zero-coded unavailable sensors. `Issues` marks a submeter/aggregate
inconsistency, **not every imputation**. These latent cleaning effects cannot be
removed with the cleaned release alone. Its appliance streams were acquired
asynchronously. Reference MAE is therefore against these resampled published
cleaned channels, not certified fully observed or manually labelled truth.

## Model and inference freeze

Keep the R1Hz bounded architecture: requested state counts (4, 3, 2, 3) for
washing machine, cooling, dishwasher, and residual background; depth one;
two intervals per chunk; duration weights; transition coefficient
0.1 times the squared train-fitted channel range (minimum range 1 W).
The dishwasher is deliberately a coarse two-level proxy, not a claim that its
physical operating modes are binary. Distinct fitted levels may be fewer for
low-diversity training channels; retain and report this without substitutions.
Fit centroids on the first partition only; background samples are mains minus
the three selected channels, without clipping. The common aggregate event
threshold is 0.01 times the maximum range of training-only binary appliance
centroids. Fit the high-state proxy thresholds from those same binary centroids.
Freeze all home models before reading **any** REFIT test window.

Transfer the already frozen R1Hz QAOA angles without REFIT optimization. Archive
the original angles file hash. Compare ideal feasible-subspace QAOA with
shot-matched uniform feasible sampling, matched two-interval exact enumeration,
full-run exact temporal DP with the same compression, and a training-mean
constant predictor. The stochastic arms each use 256 shots per chunk and three
fixed seeds, 1907/2907/3907, with deterministic home/day offsets. Each method
carries its own predicted preceding state and resets at gaps. No appliance
references enter the inference API; all predictions are saved before scoring.
Training means are not an oracle and are fixed before test access. No QPU is
submitted or billed by this programme.

## Outcomes and uncertainty

Primary outcome: average of the 14 home-specific macro appliance MAEs (equal
home weights). Also report block-pooled MAE, per-appliance errors, aggregate
reconstruction MAE, high-state-proxy F1/MCC/event F1, activity, valid coverage,
and separate extraction/fitting/inference times. Average stochastic error sums
over the three seeds before ratios or resampling; seeds are not independent
homes. A home with no valid test block has null outcomes and is explicitly
excluded from the evaluable-home denominator, never silently replaced.

Prespecified contrasts: ideal QAOA minus uniform, matched exact, full DP, and
training mean. Negative differences favor QAOA. Within each home, use a paired
day bootstrap (2,000 replicates, seed 8101 plus house ID), recomputing the ratio
of error sums to block counts. Across homes, bootstrap paired home-level MAE
differences (10,000 replicates, seed 9101), retaining equal-home weighting.
These intervals are descriptive: homes are not a random population sample;
temporal dependence and source cleaning remain limitations; no multiplicity-
adjusted superiority claim is made. No test-time retuning or best-seed selection.

The code, source hashes, protocol, transferred angles, train-only models, every
selected window, predictions, quality reports and calculation receipts are
retained. Report the frozen-setting scope even if the external results are poor.
