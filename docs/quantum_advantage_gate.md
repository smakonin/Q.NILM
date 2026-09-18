# What would justify a Q.NILM quantum-advantage claim?

The present experiments do not establish quantum advantage. Fixing a
representation, improving a classical appliance model, executing a circuit,
or outperforming uniform random sampling is insufficient.

## Define the claim before the experiment

A practical optimization claim could be: on a prespecified, representative
family of unseen NILM problems, the complete quantum-assisted method reaches
a fixed certified objective gap and application-accuracy target with 99%
success in less total time than the strongest relevant classical methods
under disclosed resources. A quality-at-fixed-budget claim is a different
claim and must use an explicit common budget and identical test inputs.

There is no qubit-count threshold, number of shots, or p-value that by itself
establishes advantage. A statistically significant improvement over a weak
comparator is still a weak benchmark. A result against the implemented
baselines should be labelled a measured benefit against those baselines,
not a proof that all classical algorithms are inferior.

## Required evidence

1. **A sound application target.** Fit powers, state models and background on
   training data; freeze choices on validation. Score untouched mains and
   separately identify selected-circuit controls. Report per-appliance power,
   event and energy errors, gaps, missing data and inactive-appliance coverage.
   Test additional homes/datasets, not only one long residence record.
2. **Strong, appropriate classical comparison.** Include exact temporal DP,
   exact MAP FHMM, tuned MIP/branch-and-bound and effective approximate
   optimizers. For the separable binary chain, DP is O(K N 2^N); for the
   categorical extension it is O(K N product(state counts)). Our software
   memory caps are safety limits, not proof of classical intractability.
   Evaluate appliance/state complexity and realistic constraints, not merely
   the number of temporal variables. Use valid bounds if exact solving is
   unavailable. Relevant classical simulation of the quantum circuits is
   another control where tractable.
3. **A genuine quantum-assisted execution.** Execute the frozen method on a
   physical device; separate direct QPU, hybrid service and simulation. Include
   parameter optimization and its classical work. Exact-statevector angle
   search cannot be silently excluded from a supposedly scalable pipeline.
   An ablation removing/replacing the quantum component should test whether
   any benefit actually depends on it rather than new preprocessing.
4. **Honest resource accounting.** Record preprocessing, optimization calls,
   compilation/embedding, measurement/readout, mitigation/postselection and
   decoding. Report queue time separately and also end-to-end time. Charge
   unsuccessful runs and discarded shots. Match quality/success targets and
   allow serious classical tuning; do not compare only QPU kernel time to
   an entire classical workflow.
5. **Replicated benefit and scaling.** Use paired unseen instances, independent
   optimization seeds, hardware jobs, layouts and calibration dates. Declare
   primary comparisons and multiplicity handling before test. Confidence
   intervals must reflect those experimental units, not pretend shots from
   one job are independent household trials. Show a robust advantage region
   or scaling trend and its limits; more observations of the same tiny
   optimum are not enough.

The 27-job / three-date IBM programme is a robustness study, not a guarantee
of quantum advantage. Multi-date work cannot be completed in one sitting.
The current 288-variable synthetic family remains exactly solvable locally.

## Completed first campaign and next decision

`scripts/run_heldout_campaign.py` completed 150 timestamp-selected, disjoint
24-hour windows: 90 training, 30 validation, and 30 test windows inside the
chronological 60/20/20 source partitions. These are not necessarily calendar
days or consecutive windows. Valid 30-second block counts are 259,196 / 86,396 /
86,396; four excluded `s`/`+` rows invalidate four blocks in each partition.
No window was replaced after measurement inspection.

The [protocol](../results/heldout_campaign/protocol.json) fixes the window list
before reads. Power levels, background, and FHMM probabilities use training
only; penalties are selected on validation and the
[models are frozen](../results/heldout_campaign/frozen_models.json) before test.
This is an internal freeze, not independent preregistration. The current
implementation passes 137 automated tests; those are implementation checks,
not proof of advantage.

Fourteen model/input rows cover binary/multistate regular and compressed exact
temporal optimization, binary/multistate exact MAP FHMM, and a low-power
constant, each on actual mains and a selected-circuit-total control. Mains
models use a three-state training-only background component. Selected-circuit
totals are diagnostic controls unavailable to deployed mains-only NILM.

Actual-mains regular macro appliance MAE improves from 35.208 W for binary
inference to 24.729 W for multistate inference: **29.76% classical modeling
improvement**, not a quantum result. The paired difference is -10.478 W,
with a descriptive 95% 24-hour-window bootstrap interval [-11.725, -9.214] W.
All power errors are evaluated against original valid blocks, after expanding
compressed predictions. This is a first within-home result, not the full
external-validation gate.

The vacuum remains a failure case: regular multistate mains MAE is 13.393 W
versus 2.198 W for the low-power constant, and only three test windows contain
above-threshold vacuum readings. The common training-derived thresholds define
high/low proxies, not physical ON states. In particular, the 244.903 W fridge
threshold excludes its fitted 101.756 W middle state. Consult the
[complete results](../results/heldout_campaign/summary.md) and
[post-run interpretation notes](../results/heldout_campaign/interpretation_notes.md),
including the limited penalty grid and the compressed objective's omitted
state-independent constant. Neither caveat changes reported appliance MAEs.

`scripts/run_ibm_followup_controls.py` completed twelve ideal/archived-Fez-noise
control runs with 4,096 shots each. Its depth-two equal-angle split is an
unoptimized sensitivity control, not evidence that optimized depth two is
inferior or superior. These controls are circuit-matched but not
calibration-matched to the new QPU job; 27 active calibration parameters changed
between the archived and live snapshots. The separate corrected depth-one Fez job
`dajn53r9k43c73ahdmag` found the optimum in 75/4,096 shots and a best state with
9/9 diagnostic proxy agreement, with 3 seconds of IBM-accounted QPU usage.
It is a same-hour selected-circuit check, not held-out QPU accuracy, a matched
legacy-versus-corrected experiment, or an end-to-end speed advantage.
The [IBM follow-up plan](ibm_followup_plan.md) separates completed checks from
multi-date, broader size/depth, and penalty-X versus XY mixer hardware studies,
which remain planned and unapproved. The new full-coverage categorical
campaign has started under a separate frozen plan; it is not yet completed physical
evidence. Multi-home testing is also still outstanding.

## Full-coverage ideal-QAOA result: no advantage

The [new simulation analysis](../results/quantum_heldout/simulation/summary.md)
covers every one of the original 86,396 valid test blocks across 30 windows,
using the fixed multistate appliance/background model. Because those periods
were already examined, this is a frozen post-exposure extension, not fresh
blind confirmation. Training-only objective calibration selects one p=1
angle pair: gamma=1.5707963268 and beta=1.1087974071. Test appliance MAE
does not enter angle selection.

The circuit is an explicit one-hot algorithm: uniform feasible initialization,
diagonal objective phase, and ordered XY pair rotations. Two intervals encode
24 qubits but only 5,184 feasible states, permitting efficient exact classical
simulation. There is no qubit-count-based hardness claim. The sequential
adaptation needs 2,451 one/two-interval circuits; every method uses its own
prior-state boundary and resets at gaps. No classical solution warms up QAOA.

At 256 shots per circuit and three fixed seeds, ideal QAOA achieves macro
appliance MAE 32.091 W versus 41.537 W for shot-matched uniform sampling,
26.012 W for matched exact two-interval inference, and 26.041 W for full-run
compressed DP. The descriptive paired-window interval for QAOA-minus-exact-
chunk MAE is [4.322, 7.945] W, with a 6.079 W point difference unfavorable
to QAOA. Its improvement over uniform sampling is not an advantage over
strong classical computation. Independent seeds are averaged within windows,
not counted as extra household observations.

The full-coverage IBM campaign has **started; full results are pending**: up to 129
adaptive jobs, 2,451 circuits, and 627,456 shots. Its preparation snapshot has
594 free seconds available, a 540-second campaign cap, and approximately
430 seconds estimated accounted use. Its first adaptive job,
`dajnp09hvn6c73cul07g`, was submitted at 2026-09-14 04:48:01 UTC.
Only the hardware runner's `--mode run`
submits jobs; it enforces Open Plan access and budget guards. Estimates are
not guarantees of completion. The hardware MAE must remain unreported until
all frozen windows are covered and scored, with infeasible shots and the
declared no-valid-shot fallback explicitly counted.

Even successful hardware completion would establish execution and application
accuracy for this one-home sequential adaptation, not speed advantage. Further
evidence must show a replicated benefit on new, representative instances
against the exact controls and other appropriate strong solvers, account for
all angle-training/simulation work, and test multiple homes, hardware dates,
layouts, and realistic complexity. The current exact controls already outperform
the ideal circuit on appliance MAE, so hardware execution alone cannot remove
that limitation.

The next decision must reflect the mixed appliance results: address rare-load
and background attribution using a new training/validation design and untouched
test evidence before investing in a large hardware comparison. Do not tune on
the now-exposed test windows and relabel them as unseen. A useful, reproducible limitations
study is scientifically valid without claiming quantum advantage; journal
acceptance or novelty cannot be guaranteed by completing a checklist.

## Primary methodological references

- Rønnow et al., [Defining and detecting quantum speedup](https://arxiv.org/abs/1401.2910), Science (2014): careful definitions and comparison pitfalls.
- Mills et al., [Application-Motivated, Holistic Benchmarking of a Full Quantum Computing Stack](https://quantum-journal.org/papers/q-2021-03-22-415/), Quantum (2021): application-relevant device/compiler benchmarking.
