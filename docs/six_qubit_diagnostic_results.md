# Six-qubit Q.NILM diagnostic: completed offline results

Completed 14 September 2026. All three planned investigation steps are complete
locally: 30 ideal component/scaling circuits, 12 ideal phase controls, and 106
exact noisy simulations. No new hardware job was submitted (0 QPU seconds).
These are synthetic diagnostic results, not held-out NILM accuracy, an IBM
hardware repetition, or evidence of quantum advantage.

## Evidence and scope

- [Frozen protocol](six_qubit_diagnostic_protocol.md)
- [Plan, inputs, training rules and hashes](../results/quantum_diagnostics/six_qubit_001/plan.json)
- [All 106 results](../results/quantum_diagnostics/six_qubit_001/summary.json)
- [Independent audit](../results/quantum_diagnostics/six_qubit_001/independent_audit.json)
- [Circuit/probability archive](../results/quantum_diagnostics/six_qubit_001/)

The primary problem is a 680 W aggregate from two three-state loads:
`[0,400,900] W` and `[0,120,280] W`. The unique exact solution is 400 + 280 W.
There are nine feasible assignments encoded in six one-hot qubits (64 basis
states). A valid answer selects exactly one state for each load; validity is
not the probability of selecting the correct assignment.

The physical layouts are A = qubits 128–133 (archive name `suspect`) and
B = 140–145 (`comparison`). These names identify fixed test groups, not
confirmed faulty/healthy hardware. Both use the archived Fez calibration
retrieved on 14 September at 16:30:07 UTC. The all-to-all control is hypothetical.

## Step 1 — verify the objective and circuit

All direct-objective, logical-circuit, compilation and bit-order checks passed.
An independent reconstruction checked all 42 ideal circuits/controls, with
maximum probability discrepancy **1.67e-15**. All ideal circuits preserve
one-hot feasibility to numerical precision.

The six-qubit full circuit uses 36 native CZ gates and depth 82 without routing,
versus 63 CZ gates and depth 134 on either restricted IBM layout. Thus a small
logical width is not automatically a short physical circuit.

The p=1 objective-only search used 1,024 evaluations (two fixed restarts).
Its frozen angles are gamma = 20.900634810095056 and beta = 2.0095088818515365.
Mean objective falls from the uniform baseline's 161,555.556 W² to
113,547.863 W², a **29.716% reduction**. However, the optimum's probability
is **10.601%**, slightly below uniform sampling's **11.111%**. Minimizing
mean objective did not maximize exact-solution probability. This is an
algorithm/selection-objective limitation on this instance, separate from
hardware noise; correctness of a circuit does not establish its usefulness.

There is no search-speed benefit demonstrated here: the classical exact answer
requires examining only nine assignments. A high projected chance of seeing
the optimum after 256 shots would not establish an advantage, since uniform
sampling already makes that probability almost one on this tiny problem.

## Step 2 — separate error mechanisms

Percentages below are exact model probabilities for the full six-qubit p=1
circuit. They are not measured QPU rates or confidence intervals.

| Noise model | Valid, layout A | Valid, layout B | Optimum per raw shot, A | Optimum per raw shot, B |
|---|---:|---:|---:|---:|
| Ideal | 100.000% | 100.000% | 10.601% | 10.601% |
| Readout only | 88.321% | 94.942% | 9.459% | 10.095% |
| Depolarization only | 86.279% | 86.857% | 9.177% | 9.323% |
| Gate and idle relaxation only | 88.732% | 90.841% | 9.356% | 9.630% |
| Combined calibration approximation | 72.264% | 79.590% | 7.770% | 8.542% |
| Native local Z, −0.02 rad per CZ | 99.070% | 99.070% | 11.360% | 11.360% |
| Native local Z, +0.02 rad per CZ | 99.080% | 99.080% | 9.680% | 9.680% |
| Native ZZ, −0.02 rad per CZ | 99.669% | 99.669% | 9.104% | 9.104% |
| Native ZZ, +0.02 rad per CZ | 99.682% | 99.682% | 12.148% | 12.148% |
| Common generic noise | 78.098% | 78.098% | 8.392% | 8.392% |

Readout, depolarization and relaxation each reduce validity. Their losses are
not an additive causal decomposition: the standalone depolarization and
combined models are alternative uses of calibrated average gate infidelity.
The generic model gives identical results on A/B at six qubits because their
compiled structure and generic noise parameters are the same.

The separately compiled component circuits give this calibration-model ladder:

| Component | Native CZ gates on either layout | Valid, A | Valid, B |
|---|---:|---:|---:|
| W preparation | 12 | 85.781% | 92.369% |
| W + cost | 38 | 79.054% | 85.392% |
| W + mixer | 28 | 81.757% | 88.613% |
| W + cost + mixer | 63 | 72.264% | 79.590% |

This ladder is descriptive, not gate-count-matched attribution: removing a
component allows recompilation to change the physical sequence.

### Phase distortion versus invalid answers

An RZ(±0.3) inserted after the logical cost circuit leaves immediate measurement
probabilities unchanged without the mixer. With the ideal XY mixer, the optimum
probability changes from 10.601% to 8.841% or 12.150%, while validity remains
100%. Diagonal phase plus an ideal excitation-preserving mixer cannot leave the
one-hot subspace. In contrast, the ±0.02-radian errors inserted between native
gates produce some invalid answers because intermediate basis-changing gates
do not individually preserve that subspace.

This distinction limits interpretation of the earlier Ramsey anomaly. It does
not show that all coherent errors are harmless, nor does the small native-phase
sweep rule out larger or context-dependent errors. These injected angles were
fixed sensitivity scenarios, not measured or fitted IBM gate errors.

## Step 3 — increase gate count, depth and width

### Same ideal circuit, more physical gates

Replacing every compiled CZ with three or five CZs leaves its ideal operation
unchanged (CZ³ = CZ⁵ = CZ). No recompilation removed the extra pairs. This is
the cleanest accumulation comparison in this experiment.

| CZ multiplier | CZ count, routed | Valid, calibration A | Valid, calibration B | Valid, common noise routed | Valid, common noise all-to-all |
|---|---:|---:|---:|---:|---:|
| 1 | 63 | 72.264% | 79.590% | 78.098% | 86.042% |
| 3 | 189 | 55.222% | 62.170% | 62.010% | 76.673% |
| 5 | 315 | 43.868% | 49.992% | 50.597% | 68.717% |

On layout A, tripling and quintupling CZ count reduce validity by 17.042 and
28.395 percentage points, respectively, with the same ideal answer. This
establishes error accumulation within the specified simulation model, not a
causal attribution of historical hardware errors solely to CZ infidelity:
extra gates also add duration and change modeled idle exposure.

### More layers or qubits

| Case | CZ count A / B | Valid, calibration A | Valid, calibration B | Valid, common noise A / B |
|---|---:|---:|---:|---:|
| Six qubits, p=1 | 63 / 63 | 72.264% | 79.590% | 78.098% / 78.098% |
| Six qubits, p=2 | 105 / 105 | 61.878% | 70.564% | 67.934% / 67.934% |
| Six qubits, p=3 | 147 / 147 | 54.939% | 63.819% | 61.100% / 61.100% |
| Nine qubits, p=1 | 141 / 147 | 42.011% | 57.638% | 57.710% / 56.849% |
| Twelve qubits, p=1 | 210 / 207 | 29.894% | 15.961% | 43.761% / 44.774% |

Depth uses split p=1 angles, not independently optimized p=2/p=3 parameters.
Their ideal mean objectives already worsen to 264,503.833 and 305,082.722 W²;
their quality decline cannot all be attributed to noise. Nine qubits add a
load and twelve add a time interval, so these are different tasks, each checked
against its own exact ideal reference. They are not a fitted scaling law.

The expanded B layout is worse at twelve qubits despite being better at six.
Inspection of the archived calibration finds short T2 values on newly included
qubits 149 and 150 (3.853 and 5.636 microseconds), and reported CZ errors of
5.350% on edge 148–149 and 2.122% on 149–150. These are plausible contributors
within the model, not separately measured contributions. The reversal is absent
under common generic noise, warning against labeling an entire region “good”
from its six-qubit subset. No post-result layout substitution was made.

## Verification, limitations and next hardware test

All 397 software tests passed, including 18 new diagnostic tests. The independent
audit recomputed every noisy metric and checked artifact hashes, measurement
order and timing accounting. It also re-evolved all 96 six-qubit noisy circuits
with Qiskit's DensityMatrix implementation rather than Aer; maximum probability
discrepancy was 6.89e-14. Nine/twelve-qubit cases received independent ideal,
metric and provenance checks, not a second noisy evolution.

The model excludes leakage, crosstalk, correlated readout, reset error and drift.
Its ASAP timing is not IBM's actual pulse schedule. Independent arithmetic
checks validate the implementation, not these modeling assumptions. Exact
model probabilities have no finite-shot sampling error; uncertainty in the
physical noise parameters was not estimated.

The next hardware comparison should use this frozen six-qubit circuit, W-only
and cost/mixer controls, both fixed layouts, and the same-ideal CZ-fold variants.
Measure readout calibration in the same session; compare complete raw
distributions and validity against these predictions with finite-shot
uncertainty. A duration-matched idle control and a second calibration date would
help distinguish coherent/duration effects from a persistent edge defect.
That is a proposed next experiment, not a submitted job or completed test.

The present evidence supports reducing physical gate count and testing the
small circuit on hardware before another large campaign. It does not uniquely
identify the cause of IBM Run 1, establish recovered NILM accuracy, or justify
a quantum-advantage claim. The paper and all earlier IBM results are unchanged.

## Reproduction

Use a fresh named directory; preparation and completed-result overwrites are
refused. The snapshot, source versions and scientific protocol are frozen in
the existing plan. The audit requires the recorded dependencies and source.

```sh
PYTHONPATH=src:. .venv/bin/python scripts/run_six_qubit_diagnostic.py --mode prepare --output-dir results/quantum_diagnostics/six_qubit_new
PYTHONPATH=src:. .venv/bin/python scripts/run_six_qubit_diagnostic.py --mode run --output-dir results/quantum_diagnostics/six_qubit_new
PYTHONPATH=src:. .venv/bin/python scripts/audit_six_qubit_diagnostic.py --folder results/quantum_diagnostics/six_qubit_new --output results/quantum_diagnostics/six_qubit_new/independent_audit.json
```
