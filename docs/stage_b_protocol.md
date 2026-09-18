# Stage B: controlled categorical simulation

This additive programme completes the **bounded controlled-simulation gate**
in Table I. It does not establish quantum advantage, cross-home NILM accuracy,
optimal variational parameters, or calibrated hardware performance. The older
binary 6/9/12-qubit results and all IBM results remain separate and unchanged.

The machine-readable protocol is created before scored execution. It freezes
the source hashes, seeds, complete matrix, endpoints and comparison rules in
`results/stage_b/run_002/protocol.json`. Existing archives are not overwritten.
Run 001 stopped when SciPy supplied a gamma of `-4.440892098500626e-16` at a
zero lower bound. Its partial results and original source copies are retained;
they are not merged into the completed comparison. Run 002 repeats the entire
same matrix after a regression-tested, roundoff-only bound correction. No
seeds, noise levels, optimizer budgets or endpoints were changed.
Only optimizer excursions within `8 * machine_epsilon * max(1, abs(bounds))`
per coordinate are snapped to an exact bound; larger violations still stop.
Every corrected evaluation and its maximum displacement is recorded.

## Frozen matrix

| Channel state counts | Intervals | Encoded qubits | Feasible states | Ideal | Exact gate/readout noise |
|---|---:|---:|---:|---|---|
| 2, 2 | 1 | 4 | 4 | Yes | Yes |
| 2, 2 | 2 | 8 | 16 | Yes | Yes |
| 3, 2 | 2 | 10 | 36 | Yes | Yes |
| 3, 3 | 2 | 12 | 81 | Yes | No |
| 4, 3, 2, 3 | 1 | 12 | 72 | Yes | No |
| 4, 3, 2, 3 | 2 | 24 | 5,184 | Yes | No |

- Eight base-instance seeds: 1001 through 1008.
- Measurement-noise standard deviation: 0%, 1%, or 5% of the largest channel
  maximum. Variants share levels, planted states, weights, penalties and the
  same standard-normal noise vector. Measurements are not clipped.
- Each channel has evenly spaced levels between zero and its maximum.
  Maxima are `[90, 420, 1250, 1800] W` for the channels present, independently
  multiplied by a uniform factor in `[0.9, 1.1]`. Initial categories are
  uniform; each following interval changes each category with probability
  0.35, uniformly among its other categories. Integer duration weights are
  1 through 8. Transition penalties are 0.02 times channel maximum squared.
  No preceding reference state is supplied.
- Depths `p=1` and `p=2`, separately optimized for every noisy-measurement
  instance. Each has two optimizer restarts (7001, 7002), exactly 512 objective
  calls per restart. Each restart evaluates the uniform circuit and 31 random
  candidates, refines the two best distinct starts with bounded Powell searches
  of at most 240 calls each, and fills early-convergence unused calls with
  random candidates. All calls, including repeated points, count.
- Gamma bounds are `[0, 8 pi]`, beta bounds `[0, pi]`; these are finite search
  bounds, not a claim of objective periodicity or globally optimal angles.
  Minimize the ideal expected objective using supplied measurements, not
  planted truth or a known minimizing state. There is no lower-depth warm start.
  Archive both restart histories and select the lower expected objective.
- Use the selected ideal-trained angles unchanged for both gate-noise levels.
  This tests noise sensitivity, not noise-aware variational training.
- Full-connectivity compilation uses `rz/sx/x/cx`, optimization level 1 and
  transpiler seed 17. Depolarization follows every compiled gate, **including
  rz**. The low model uses parameters `0.0001/0.001/0.005` for one-qubit
  depolarization/two-qubit depolarization/independent readout bit flips; the
  high model uses `0.001/0.01/0.02`. Parameters are channel strengths, not IBM
  error calibrations. Exact density-matrix evolution is followed by an exact
  readout channel, including transfers into and out of the feasible subspace.
- Shot budgets: 32, 128, 512, 2,048, with 32 independent repeats at each budget.
  Within a matched instance/budget/repeat, uniform, depths and noise levels
  use common random numbers; distinct budgets and repeats have separate
  deterministic streams. Noise-amplitude variants remain clustered by base
  seed for interpretation.
- Uniform sampling over feasible categories is a zero-training classical
  reference. It uses identical measured objectives and shot budgets; it is
  not presented as a noisy physical-circuit baseline.

This gives 144 measured instances, 288 optimizations / 576 optimizer restarts,
294,912 objective evaluations, 720 distributions (including 144 uniform
references), and 92,160 finite-shot trials. The exact-noise ceiling was chosen
using a separate seed-99999 performance check, not scored-instance quality:
the p=2 high-noise density simulations took about 0.035/0.574/17.0 seconds at
8/10/12 qubits with four threads. Categorical horizons remain one or two
intervals; the archived binary sweep separately covers three and four.

## Endpoints and uncertainty

Exhaustive feasible-state enumeration provides a floating-point optimum
certificate. All energies within `1e-10 * max(1, maximum energy - minimum
energy)` of the minimum count as optima. Sampling selects the lowest-energy
**observed feasible** state only; tied observed energies use the lowest index.
The canonical exact-state accuracy reference uses the lowest minimizing index.
There is no exact-solver fallback or injection of an unobserved optimum.

Primary recovery accuracy is category accuracy against planted categories.
One-hot bit accuracy is secondary because it is inflated as register sizes
grow. The exact objective can prefer a different trajectory from planted truth
when noise, ambiguity or transition penalties matter; report that discrepancy.

Archive the optimum probability in the **all-shot** denominator, total feasible
probability, conditional expected feasible cost, raw sampled certified gap,
gap divided by the feasible energy span, category/bit accuracy, and failures.
Invalid outcomes remain an explicit sampling outcome. A trial with no feasible
sample abstains: hit probability and accuracy are zero, raw gap is missing,
and the separately named **failure-penalized normalized gap** is one (a declared
worst-feasible-loss convention, not the cost of an invalid string). Conditional
gap summaries must always be accompanied by coverage. No invalid QUBO extension
is interpreted as an appliance-power objective.

For each instance, compute `ceil(log(0.01)/log(1-p_opt))` shots for a 99%
probability of observing at least one optimum; `p_opt=0` means infinity and
`p_opt=1` requires one shot. Calculate these before summarizing across instances.
Likewise, analytic finite-shot success is `1-(1-p_opt)^shots` per instance.

The primary depth contrast is p=2 minus p=1 failure-penalized normalized
best-sample gap at 2,048 shots, separately by shape and measurement-noise
level under ideal simulation. Noisy contrasts are secondary. Average the
32 repeat outcomes within each base instance, then use a paired percentile
bootstrap of eight base-instance seeds (10,000 resamples) for 95% intervals.
These are exploratory, small-sample descriptive intervals; no multiplicity-
adjusted significance, population generalization or advantage claim is made.

Classical preparation/enumeration, optimizer evaluation count and runtime,
statevector time, compilation, independent ideal-circuit audit, density
simulation and finite-shot sampling times are archived separately. These
local classical-simulation times are not QPU or comparative-speed estimates.

## Reproduction and acceptance

Install the optional `stage-b` dependencies into an isolated environment.
The frozen archive records exact Python, NumPy, SciPy, Qiskit and Aer versions.

```sh
python scripts/run_stage_b.py --mode prepare --output results/stage_b/new_run --noise-max-qubits 10
python scripts/run_stage_b.py --mode run --output results/stage_b/new_run
python scripts/run_stage_b.py --mode summarize --output results/stage_b/new_run
```

Preparation refuses an existing protocol. Execution checks source hashes and
resumes only missing records. Do not change frozen source files to repair a
scored archive: retain it, document the issue and create a new versioned run.
The complete probability vectors, optimizer histories and deterministic
sampling seeds permit independent regeneration of every trial.

Completion requires the exact unique matrix, all restart/call counts, direct-
objective agreement, physical/reduced ideal circuit agreement at both depths
for every noise-supported instance, probability-mass checks, independent
endpoint and trial reconstruction, paired-interval checks and paper/source
agreement. Scope completion does not close Stage D, external Stage G, or
independent-date hardware robustness work.
