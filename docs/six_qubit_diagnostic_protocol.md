# Six-qubit Q.NILM diagnostic: offline protocol

Scope: the user requested the six-qubit experiment and all three investigation
steps. This authorizes local implementation and simulation only. There are no
account, hardware-submission or paid-service operations in this experiment.
The original IBM archives, frozen source files and paper results are preserved.

## Step 1: exact small-problem and circuit verification

The primary synthetic problem contains two three-state loads:
`[0, 400, 900] W` and `[0, 120, 280] W`. Its aggregate is 680 W, generated
by states `[1, 2]`. This has nine feasible assignments in a 64-dimensional
six-qubit Hilbert space and a unique zero-residual solution. Ground truth is
used for evaluation, never as an optimizer input. This is a controlled sum of
the included loads; no unmodeled whole-house appliances are silently dropped.

Use the existing coefficient-normalized categorical objective, positive W
initialization and ordered chain-XY mixer. Train p=1 on ideal objective
expectation with two restarts (9101, 9102), exactly 512 calls each, using the
existing Stage B optimizer and bounds. Reuse the resulting angles unchanged
across all p=1 noise and placement comparisons. No noisy-result tuning.

Enumerate the categorical objective directly and compare it with the QUBO
including its constant. Compare full physical probabilities from logical
Qiskit, the independent feasible-space simulator, and compiled Qiskit.
Check all output bit mappings and the whole probability distribution, not
only the most probable answer. All ideal output and feasibility discrepancies
must be less than 1e-8 (direct objective tolerance 1e-7 W²).

Compilation uses native rz/sx/x/cz, optimization level 1 and seed 17. Compare:

- All-to-all connectivity: a hypothetical routing-free control, not an IBM QPU.
- Restricted Fez subgraph starting at `[128,129,130,131,132,133]`.
- Restricted comparison subgraph starting at `[140,141,142,143,144,145]`.

Use only the pre-existing public calibration snapshot retrieved at
2026-09-14 16:30:07 UTC in `physical_001/preparation_snapshot.json`. This is
not a fresh calibration or the actual pulse schedule of the full IBM run.
Fixed initial layouts and SABRE routing are confined to each selected
induced subgraph, so all simulated physical qubits fit the declared width.
Archive QPY/QASM and measurement permutations.

## Step 2: isolate error mechanisms

At six qubits compare preparation only (W), W plus cost, W plus mixer, and
the full W-cost-mixer circuit. Each component is compiled separately; these
are not gate-count-matched causal ablations of the full circuit. For each
compiled circuit, all noise arms reuse precisely the same compiled template.

On both actual Fez subgraphs, evaluate:

| Model | Definition |
|---|---|
| Ideal | Exact physical statevector reference |
| Readout only | Independent asymmetric assignment channels from archived per-qubit probabilities |
| Depolarization only | Public Aer device builder, gate error enabled and thermal relaxation disabled |
| Relaxation only | Archived T1/T2; gate error disabled; gate-time and explicitly modeled idle-time relaxation |
| Combined calibration approximation | Public Aer combined gate-depolarization/thermal channels, plus idle relaxation and asymmetric readout once |
| Coherent local Z | RZ(−0.02) or RZ(+0.02) after every native CZ on its second listed operand |
| Coherent ZZ | RZZ(−0.02) or RZZ(+0.02) after every native CZ |
| Common generic noise | Same controlled noise/timing parameters on every topology, defined below |

The coherent angles are sensitivity scenarios, not estimates fitted from the
earlier Ramsey measurement and not claims about a particular IBM gate.
The calibration depolarization-only arm and combined arm are alternative
models of reported average infidelity; their individual effects must not be
added arithmetically as independent causal contributions.

Gate timing is an explicit ASAP schedule of the compiled gate DAG. A qubit
waits for the other operand before a two-qubit gate; all output qubits are
aligned to terminal readout. Add relaxation to these idle intervals. Gate
relaxation is already inside the calibrated gate channel and is not added
twice. RZ is virtual and has no extra generic depolarizing channel. T2 is
bounded by 2*T1 as required by the thermal model. Initialization is ideal;
no reset, leakage, correlated readout, crosstalk or time drift is modeled.
Measurement-time relaxation is not separately added to the assignment model.

Common generic noise: depolarizing channel strength 0.0001 after physical
single-qubit gates and 0.001 after CZ; T1=100 microseconds, T2=80 microseconds;
single-qubit duration 40 ns, CZ 80 ns, RZ zero; independent symmetric readout
flip probability 0.01. These are controlled scenario parameters, not a
measured IBM or competitor specification.

Additional exact logical-phase controls insert RZ(±0.3) on logical qubit 2
after cost, with or without the mixer, on all three topologies. A purely
diagonal phase immediately before computational-basis measurement must leave
the probabilities unchanged. With the ideal XY mixer, it may change the
distribution among feasible assignments but cannot break one-hot feasibility.
This distinguishes logical phase distortion from errors between intermediate
physical gates that can produce population errors.

## Step 3: increase one aspect of complexity

| Case | Change from the six-qubit p=1 baseline |
|---|---|
| six_p2 | Two layers, splitting total p=1 gamma and beta equally |
| six_p3 | Three layers, splitting total p=1 gamma and beta equally |
| six_cz3 | Replace each compiled CZ with three CZs, without recompilation |
| six_cz5 | Replace each compiled CZ with five CZs, without recompilation |
| nine_p1 | Add one three-state load `[0,55,175] W`, in state 1 |
| twelve_p1 | Add a second interval with first two loads in states `[2,1]`, aggregate 1,020 W |

The two-interval penalties are `[1600,400] W²` for changing appliance state;
the third load has penalty 100 W², inactive in its single-interval case.
All segment weights are one and there is no previous-state boundary input.
Widths use their own coefficient normalization with the same p=1 angles.
The higher-depth split-angle circuits are not independently optimized and
do not test the best achievable p=2/p=3 performance.

Odd CZ folding is the clean gate-count control: CZ³=CZ⁵=CZ ideally, so its
full ideal measurement distribution must match the unfolded circuit. No
compiler is permitted to cancel the inserted pairs. All-to-all and the two
fixed subgraphs are tested at every size/depth. The suspect subgraph grows
by appending `[127,126,125,124,123,122]`; the comparison by appending
`[146,147,148,149,150,151]`, preserving the initial six-qubit mappings.
Apply common generic noise across all topologies, and the combined calibration
approximation additionally on the two actual subgraphs. Width/depth changes
are compared with their own ideal references; they are not identical tasks.

## Outputs and verification

Save all 30 compiled component/growth circuits and 12 logical-phase controls,
plus 106 exact noisy simulations. Density matrices include all 2^n states,
with a hard limit of 12 simulated physical qubits and four worker threads.
No finite-shot trajectories or noise-conditioned angle selection are used.

Report full-space probability mass, valid/invalid probability, optimal-state
probability per raw shot, conditional feasible objective in W², and total
variation distance to the same circuit's ideal distribution. A separate
256-shot iid projection reports the chance of at least one optimum and the
chance of no feasible sample; these are model calculations, not measurements.
Conditional metrics are explicitly labeled and never substituted for raw
feasibility. No invalid shot is repaired or assigned a hidden fallback.

Freeze source hashes, versions, training, circuits and calibration identity
in a plan before the production noise sweep. A technical smoke check with
untrained gamma=3, beta=0.7 verified execution and serialization before this
freeze; it did not select the trained angles, noise values or layouts.
Completed per-row checkpoints can be resumed without overwriting; partial
artifacts without a checkpoint cause a stop. A completed archive cannot rerun.

Independent verification will reconstruct ideal amplitudes and measurement
order, recompute objective/feasibility metrics and evolve all six-qubit noisy
circuits with Qiskit's separate DensityMatrix implementation. Larger cases
retain exact Aer evolution, ideal checks and independently recomputed metrics;
they are not falsely labeled independently noise-simulated if only the small
cases received that check. Results describe the specified models, not a
confirmed explanation of the historical IBM hardware failure or quantum advantage.

Method references:

- Qiskit Aer device noise models: https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.NoiseModel.html
- Thermal relaxation: https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.thermal_relaxation_error.html
