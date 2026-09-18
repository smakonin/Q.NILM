# Stage E: matched-objective mixed-integer benchmark

This additive experiment compares a generic HiGHS mixed-integer linear
formulation with the existing exact temporal dynamic program. It does **not**
reproduce Balletti et al.'s appliance-prior/learning pipeline or Li et al.'s
transient-feature binary integer search. Completing this benchmark closes
only the matched general-purpose solver part of Stage E.
The user has explicitly kept the paper's low-frequency scope and deferred
Li's full 60 Hz transient-feature comparison; the source-faithful Balletti
comparison remains a separate low-frequency gate.

## Fixed population and objective

Reuse the original 30 R1Hz test windows, all 86,396 valid 30-second blocks,
34 gap-separated runs and 4,881 compressed intervals. These previously
examined windows are a retrospective solver comparison, not a new blind test.
No source, original model, prior result or quantum-circuit file is edited.
All four trained registers (dryer, fridge, vacuum, signed background), their
levels, event threshold and rho=0.1 range-squared penalties remain fixed.
Both solvers minimize actual-duration squared reconstruction error plus a
per-channel category-change cost, with no transition across a gap or day.

For each interval, the MILP selects one of the 72 joint categories. Its
emission cost is precomputed exactly from that category's aggregate power.
Continuous change auxiliaries and category-marginal constraints impose the
same categorical switching cost. Per-interval emission minima are subtracted
and the remaining objective is scaled; reported objectives and dual bounds
restore those transformations. This expanded formulation exploits the small
joint-state space and is not a scalable claim for arbitrary appliance counts.
The direct decoded objective is independently checked against exact DP.

## Solver-setting selection, fixed before test inference

Only HiGHS presolve on versus off is selected. Use all valid runs in the first
three chronological windows labelled `validation` in the original manifest
(accept its actual `validation` or `val` label). Each candidate gets the same
15-second solve limit per validation run and relative MIP gap 1e-8. No
appliance reference values or MAE enter selection. Rank candidates by:

1. number of runs with no validated incumbent (smaller is better);
2. number of runs not solver-certified optimal (smaller is better);
3. sum of independently certified relative gaps to DP (smaller is better);
4. summed formulation plus solve time (smaller is better);
5. presolve on (fixed final tie preference).

Archive every candidate, input and exact certificate before freezing the
selected setting. This is a two-choice engineering calibration, not broad
hyperparameter tuning or retraining of the NILM model.

## Test execution and scoring

The selected setting receives a 30-second backend solve limit per full valid
run. This is not a hard end-to-end deadline; preparation, formulation,
decoding and termination overhead are separately recorded. Do not retry a
timed-out run based on its test result and do not warm-start from DP or truth.
Both arms recompress aggregate-only measurements and verify equality with the
frozen inputs. Alternate the arm order by fixed window index. Archive each
arm's predictions before loading appliance references for scoring.

Keep solver status, message, incumbent, dual bound, gap, node count, model
size, and all timings. A timeout with a validated incumbent may be scored but
is not a solver-certified optimum. If any run has no incumbent, retain its
failure: never silently fill with DP, omit it from a full-window metric, or
claim full coverage. Report partial coverage separately if needed.

Primary NILM metric is the original pooled equal-appliance MAE on dryer,
fridge and vacuum. Secondary evidence includes per-appliance metrics,
aggregate MAE, exact-objective gaps and descriptive paired-window bootstrap
intervals (2,000 resamples, seed 9301). Equal objective values need not imply
identical paths or MAE because solvers may select different tied optima.

## Timing and integrity

Separate common source-read time, aggregate compression, formulation,
backend solve and decoding/expansion. The fresh DP comparison includes its
own setup and traceback. Inference totals start at already-loaded aggregate
data, exclude model learning, imports, reference scoring and archive I/O, and
must not be called end-to-end acquisition-to-answer timings. One execution
per arm/window is descriptive local timing, not a stable speed benchmark or
a matched quantum-speed comparison. Preserve environment and actual solver
thread/seed settings where exposed.

Before scored inference, freeze source/model/input and implementation hashes;
use a new archive directory. Recheck them at completion. Recompute raw-block
metrics from saved predictions and independently certify every recovered
trajectory against the same objective. Report the within-interval residual
constant separately when translating compressed to full-block objectives.

Reference interface: [SciPy milp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html).
