# Stage D: bounded ideal mixer/architecture and initialization comparison

This additive comparison follows the completed Stage B simulation programme.
It does not alter the frozen Stage B or IBM circuits, training, or archives.
The design is frozen before new penalty/X training, but after the Stage B
results were available; it is an exploratory controlled follow-up, not an
untouched confirmatory trial.

## Complete matched matrix

Reuse every Stage B Run 002 input with at most 12 physical qubits: state-count
and interval shapes `(2,2),K=1`; `(2,2),K=2`; `(3,2),K=2`; `(3,3),K=2`;
`(4,3,2,3),K=1`. Use all eight base seeds 1001–1008 and all measurement-noise
standard deviations 0%, 1%, and 5% of the largest channel maximum: 120 inputs.
Source input and XY training/distribution files are individually hash-anchored.
No input, seed, level or truth label is reselected based on the new results.

At independently optimized depths p=1 and p=2, compare three arms:

| Arm | Initialization | Mixer | Cost outside feasibility |
|---|---|---|---|
| W/XY | Original positive one-hot W states | Original ordered adjacent RXX(beta), RYY(beta) pairs | Unchanged feasible-equivalent QUBO; circuit remains feasible |
| H/X + penalty | Hadamard on every physical qubit | RX(2 beta) on every qubit | Full QUBO plus conservative one-hot penalty |
| W/X + penalty | Same W state as W/XY | Same X mixer as H/X | Same penalized objective and normalization as H/X |

There are 240 reused XY optimizations and 480 new penalty/X optimizations,
giving 720 distributions and 92,160 finite-shot trials. Full binary simulation
is capped at 12 qubits (4,096 amplitudes). No gate-noise experiment or QPU job
is part of this core mixer comparison; Stage B supplies the noise study.

## Coefficient-only feasibility barrier

Write the original feasible-equivalent binary cost as

`f(x)=c+sum_i l_i x_i+sum_(i<j) q_ij x_i x_j`.

Define `S=sum_i|l_i|+sum_(i<j)|q_ij|` and `A=1+2S`. Add
`A*sum_register(sum_(i in register)x_i-1)^2`.
Every binary base energy lies between `c-S` and `c+S`. Every invalid string
has integer penalty units at least one, so its penalized energy is at least
`c-S+A=c+S+1`, strictly above **every** feasible base energy. The bound uses
coefficients only, not an optimum, minimum-energy label, or planted truth.
An independent read-only derivation confirmed this argument before the new
scored matrix was executed. Reproducible signed-objective enumeration checks
are included in the accompanying unit tests and every scored input is audited.

For R disjoint registers, the augmented constant is `c+A*R`, each linear
coefficient becomes `l_i-A`, and each pair within a register gains `2A`.
Other pair coefficients remain unchanged. Constants are retained for reported
costs; phase normalization excludes constants and uses
`max(1,sum|augmented linear|+sum|aggregated augmented quadratic|)`.
XY retains its original coefficient-only scale. The different phase-scale
ratio and feasible phase range are recorded for each input.

For every scored input, exhaustive full-binary enumeration independently
checks the augmented expansion, original feasible objective equality, exact
encoding feasibility, and invalid-minimum above feasible-maximum. Floating-
point tolerances are fixed from a coefficient magnitude forward-error bound;
the strict energy separation must also pass directly.

## Equal optimizer budgets, explicit cost differences

Reuse all selected XY angles and both-restart histories from Stage B, including
their full objective-call count and training time. Train each new X arm
independently, without lower-depth or XY warm starts, using restart seeds 7001
and 7002 and exactly 512 objective calls each. Every restart evaluates zero
angles and 31 seeded random candidates; the best two distinct candidates seed
bounded Powell refinements with 240 calls each. If a refinement terminates
early, remaining calls are filled by seeded random candidates. Repeated
evaluations count. Bounds remain gamma in `[0,8pi]`, beta in `[0,pi]`.

Only boundary excursions within `8*eps*max(1,abs(bounds))` per coordinate are
snapped to their exact boundary; larger violations fail. Counts and maximum
displacements are archived. For X arms, training minimizes the **full binary
penalized** ideal expected cost, divided by the augmented coefficient scale.
It never selects angles using planted-state accuracy or a minimizing state.
Zero angles retain each arm's own H or W initialization; they do not force a
common distribution across different initialization arms.

This is an architecture comparison, not a pure mixer isolation. W/X controls
initialization relative to W/XY but also changes penalties and phase scaling.
H/X versus W/X isolates initialization within the penalized-X architecture.
The sufficient penalty is conservative and is not claimed to be the best
tuned penalty baseline. Its stiffness can suppress feasible objective phase
differences. Equal depth and optimizer calls do not mean equal gate counts,
expectation-computation cost, or optimal angle quality.

## Independent circuit and resource checks

NumPy explicitly evolves all binary amplitudes: cost phase followed by
`cos(beta) I - i sin(beta) X` on each qubit. Initial H states are uniform over
all bitstrings; initial W states are uniform only on exact one-hot indices.
A separate Qiskit construction prepares the physical H or W circuit, uses
RZ/RZZ angles from the augmented Ising coefficients, and applies RX(2 beta).
Before freezing, fixed-angle controls at p=1 and p=2 on 4- and 5-qubit shapes
with separate seed 99999 are compared with independent compiled Qiskit
statevectors. All 720 selected scored circuits, including reused XY circuits,
are then independently compiled and probability-audited again.

Resource compilation uses full connectivity, `rz/sx/x/cx`, optimization level
1 and transpiler seed 17. Record physical width, compiled depth, gate counts,
compilation/audit time, and maximum probability discrepancy. No topology,
routing, scheduling, calibration or device-noise claim follows.

## Sampling, endpoints and uncertainty

Use all Stage B shot budgets 32, 128, 512 and 2,048, with 32 repeats each.
The same deterministic Stage B sampling-seed function gives common random
numbers across the matched arms/depths within each instance/budget/repeat.
Invalid mass remains an explicit outcome; it is not postselected away before
sampling. Select the lowest-energy **observed feasible** state only, using
the original feasible objective and the same exact-optimum tolerance as B.
No-feasible trials abstain: hit=0, accuracy=0, raw objective gap missing, and
the separately named failure-penalized normalized gap=1. There is no classical
fallback or insertion of an unobserved optimum.

Record all-shot feasible and optimum probabilities, conditional expected
feasible gap, planted category/bit accuracy, hit rates and abstentions.
Compare penalty-H/X minus W/XY and penalty-W/X minus W/XY as the primary
architecture contrasts in failure-penalized normalized best-sample gap at
2,048 shots, separately for each shape, depth and measurement-noise level.
The H/X minus W/X contrast is the initialization control. Average shot repeats
inside each base instance, then bootstrap paired differences over the eight
base seeds, with 10,000 percentile resamples (seed base 88301).
Intervals are exploratory and descriptive, not multiplicity-adjusted
significance tests or population-wide inference. Paired noise variants and
shot repeats never increase the independent base-instance sample count.

## Reproduce and accept

```sh
python scripts/run_stage_d_mixer.py --mode prepare
python scripts/run_stage_d_mixer.py --mode run
python scripts/run_stage_d_mixer.py --mode summarize
```

The default archive is `results/stage_d/mixer/run_001/`. Preparation freezes
code and source-input hashes; execution resumes only absent records; outputs
are never overwritten. A numerical or scientific implementation change after
freezing requires preserving the incomplete run and freezing a new version.
Completion requires every unique planned condition, all training budgets,
all input barrier certificates, independent selected-circuit audits, complete
sampling trials and independent archive verification. This closes only this
bounded ideal mixer/initialization comparison, not the other Stage D controls,
Stage G external validation, or a quantum-advantage claim.
