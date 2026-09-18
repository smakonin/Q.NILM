# Steps 1 and 2: wider-angle finite-shot training and R1Hz validation

Freeze this protocol, executable sources, dependencies and prior input hashes
before scored execution. This campaign is offline only. Do not modify prior
archives, production parameters, the paper, provider accounts or GitHub.

## Step 1: controlled synthetic robustness

Use the existing six-qubit p=1 task: levels [0,400,900] and [0,120,280] W,
aggregate 680 W, W initialization and the ordered chain-XY mixer. Compare mean
and lower CVaR50 with gamma in [0,8π] and [0,16π], beta in [0,π]. The original
range is a contemporaneous control for the new selection procedure.

Twenty paired seed pairs (40101+100i,40102+100i), i=0..19, yield 80 fits. Each
fit receives two restarts of 512 evaluations, each with 32 initial candidates
(zero angles plus 31 seeded uniform random), two bounded-Powell refinements,
and random padding for unspent evaluations. Use 256 simulated ideal iid shots
per case/evaluation. Measurement RNG seed is restart seed+1,000,000. Candidate
and measurement streams are separate. The initial pool is identical between
losses within each trial/range; adaptive search points may differ. All calls
count, including repeats and padding. Normalize costs by the same coefficient
scale. Mean and CVaR50 are compared on common outcome metrics, not raw loss
values against each other.

After searching, select the five distinct angle vectors with lowest observed
loss, ties chronological. Remeasure each with 4,096 new simulated shots using
seed first_restart+2,000,000; select the lowest rechecked loss, ties first.
Never use exact probabilities, optimum labels or known appliance states for
selection. Count all 20,480 recheck shots per case as part of the training
budget. This reduces dependence on an extreme search estimate; it does not
guarantee an unbiased final selection. Preserve before/after angles and results.

Evaluate all fits analytically at 16 and 256 inference shots, on 680 W and
the other eight aggregates of the same synthetic load models. Keep all twenty
trials. Describe seed variation, not independent-problem confidence. Primary
comparison: post-recheck wider-range CVaR50 versus mean on expected best-16 cost
and decoded appliance MAE. No selection of real-data settings based on these
synthetic outcomes: both predeclared real arms proceed.

## Step 2: static R1Hz check

Reuse the original train-only multistate centroids: dryer four states, fridge
three, vacuum two, background three. This is a 12-qubit, 72-assignment p=1
single-interval diagnostic of actual mains, not the paper's 24-qubit,
two-interval temporal programme. No temporal term, boundary prior, event
compression, clipping, or online state carry. The background is latent and
inferred from mains; no test appliance reading is used to supply it.

Select 32 fixed training timestamps: the midpoint 30-second block of 32 evenly
indexed original training days. Reuse is allowed for training; no activity-based
selection. Reject incomplete/marked/nonfinite blocks, never replace them.
Require at least 16 usable training cases or stop the real-data phase.

Fit shared angles across all usable training cases. Each evaluation draws
256 shots per case. The loss is the equal-case average of mean or lower
CVaR50 of each case's coefficient-normalized cost; it is not CVaR of a pooled
mixture across cases. Use ten paired seed pairs (60101+100i,60102+100i), i=0..9,
the same two-restart 1,024-call policy and five-candidate 4,096-shot recheck.
Gamma [0,16π], beta [0,π]. Twenty fits. Appliance readings do not enter this
training loss. Training samples, losses, all angles, seeds and resource counts
are saved. If all 32 timestamps are usable, each real fit consumes 9,043,968
classically generated measurements including rechecking; report actual counts.

### Fresh-window selection and leakage guard

Before any new validation or test value is read, enumerate recorded timestamp
ranges from existing experiment protocol/plan metadata, plus the pilot input
hour. The original 150 train/validation/test days are excluded. Preserve this
exposure map and its input hashes. Bound the new sets to the original middle
20% and final 20% chronological partitions, respectively. Extend old exposures
by one day at each side. In each remaining gap, propose one centered aligned
24-hour window, then evenly choose ten validation and twenty test windows.
Use timestamps only. Check disjointness and the guard independently. Do not
fill gaps, squeeze missing seconds, or replace low-activity/incomplete windows.

“Fresh” means outside recorded Q.NILM analysis windows. It is not a guarantee
of no previous human inspection of the broader source dataset, an independent
household, or independence between same-home days. Source globally sorted is
an assumption; monotonicity, duplicates and completeness are checked inside
selected windows. Exclude markers s and +; only complete finite 30-second
blocks enter any arm. Preserve raw-window hashes, selected block arrays and
missingness counts. No full-source ordering or checksum certification is implied
by endpoint/stat checks.

### Validation, test and controls

After all training fits finish, evaluate each rechecked fit over all accepted
validation blocks. Select one fit per loss by pooled expected appliance MAE
at 16 inference shots (lower trial index breaks ties). Also record the overall
validation winner, with mean preferred on an exact tie. Freeze selection and
validation hashes before reading any new test data. Evaluate both selected
fits on all accepted test blocks and do not change their parameters afterward.
Previously exposed days do not enter this new evaluation cohort.

Report expected MAE, per-appliance MAE, nearest-centroid category agreement,
expected aggregate objective and optimum-hit probability at 16 and 256 shots.
Analytical shot expectations are not realized physical sample counts. Ideal
feasibility is one and no-valid probability zero by construction; these do not
demonstrate robustness to hardware errors. Truth for scoring is measured
submeter watts, not a quantized state substituted for measured power.

Controls on exactly the same accepted blocks:

- Exact enumeration of all 72 assignments using the same mains objective,
  ties lower feasible index. No ground-truth tie-breaking.
- Uniform feasible sampling at both matched inference budgets.
- Lowest-power training-centroid constant predictor.
- Per-appliance nearest-centroid label-only oracle, a representation-error
  lower bound for this discrete model, not a deployable NILM solver.
- Exact enumeration of 24 assignments using the sum of the three target
  submeter readings as the observation. This is an explicitly privileged
  selected-circuit control, not mains NILM. Its role is to distinguish some
  aggregate identifiability/background limitations, not establish causality.

Primary R1Hz contrast: validation-selected CVaR50 minus validation-selected
mean, pooled measured appliance MAE at 16 shots. Secondary contrasts include
256-shot MAE, uniform controls, and exact-minus-oracle error. Weight accepted
30-second blocks equally and the three target appliances equally; background
is not an appliance metric. Report all days, including inactive ones, and all
training/validation results. Report descriptive 95% paired 24-hour-window
bootstrap intervals (2,000 replicates, seed889901) with actual day counts;
these do not measure cross-home generalization or account for all same-home
dependence. No significance, speedup or quantum-advantage claim is prescribed.

## Verification and scope boundary

Unit-test the batch circuit against the established scalar circuit, normalized
costs against direct enumeration, finite-shot means/CVaR and budget/seed rules,
decoder tie handling, oracle bounds, selection rules and timestamp guard.
Independently replay all training loss/measurement traces and rechecking;
verify selected parameters, all day metrics and paired aggregates; verify
source block extraction with a separate CSV aggregation path and audit a
bounded set of explicit Qiskit statevectors. Keep checks and limitations
inspectable. Numerical consistency does not validate a noise model or establish
that the NILM objective is correct. The new archived data are local results;
no publishing or hardware step is part of this request.
