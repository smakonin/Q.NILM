# Q.NILM physical diagnostic campaign 001

User authorization: “do your 3 suggested next tests.” This authorizes the
bounded diagnostic campaign below, not paid execution, a full NILM rerun,
automatic retries, or changes to IBM calibrations. The three groups are
implemented together and analyzed separately. This is exploratory hardware
characterization motivated by the already observed component-ladder results.

## Questions and frozen design

1. Do local preparation, measurement and reset errors reproduce the excess-one
   bias near physical qubits 132 and 133 without the large cost circuit?
2. Do zero and nominal cost angles behave differently in precisely the same
   compiled physical cost template?
3. Do targeted two-qubit benchmarking and phase-sensitive circuits identify
   a reproducible local gate-level error signature?

| Group | Design | Circuits | Shots per circuit | Raw shots |
|---|---|---:|---:|---:|
| Local controls | Two triples; eight basis states, one excited-state reset, three W preparations each | 24 | 1,024 | 24,576 |
| Matched cost | Two widths, three frozen training examples, zero/nominal cost angles | 12 | 1,024 | 12,288 |
| Gate benchmarking | Two edges, five lengths, eight paired random sequences, reference/interleaved arms | 160 | 256 | 40,960 |
| Phase-sensitive checks | Two edges, two control states, two quadratures, four CZ repetition counts | 32 | 512 | 16,384 |
| Total | One interleaved diagnostic job | 228 | Varies | 94,208 |

No sample-size-based power claim is made. Shots characterize one job, not
independent hardware dates. All raw outcomes, including infeasible samples,
remain in the archive. No NILM appliance MAE is inferred from these circuits.

### Local controls

The preselected suspect triple is (131,132,133); the comparison triple is
(141,142,143), in logical-bit order. The reference triple is a comparison
placement, not a declared error-free reference. Both are on ibm_fez.
All eight three-bit basis patterns are prepared, providing contextual
assignment probabilities. These combine state-preparation and readout errors
and cannot uniquely separate them. An additional circuit prepares |111>,
explicitly resets all three qubits, and measures. W3 preparations are repeated
three times with identical circuits, separately labelled in the randomized
job order. Implicit shot initialization remains enabled in every group.

The local circuits are new fixed-placement compilations, with no routing
outside each triple; they are not replacements for the historical W template.
Ideal density-matrix checks include the explicit reset operation.

### Matched cost controls

Reuse the exact symbolic K1_W_cost and K2_W_cost QPY circuits from the completed
ladder retry. Do not recompile them. For each of its three original training
examples, bind either all symbolic cost_angle parameters to zero or to the
original frozen values. Fixed native RZ offsets must not be zeroed.
The two arms retain the same native instruction sequence, gate count,
measurement mapping and routing ancilla. Register sizes are (4,3,2,3), repeated
in interval-major order for K=2. Both arms ideally yield uniform feasible
measurement distributions, which are checked explicitly including routing
ancillas. The earlier separately compiled W-only circuit is not this matched
zero-angle control.

Changing angles can probe angle-dependent execution effects, but zero angles
still retain physical entangling and routing operations. A diagonal operator
followed by computational-basis measurement is blind to pure diagonal phase
errors. No logical-angle or quantum-advantage claim follows from this control.

### Gate and phase checks

The preselected edges are suspect (132,133) and comparison (141,142). Each edge
gets reference randomized Clifford sequences and the paired sequences with a
native CZ interleaved between Cliffords. Lengths are 1,4,16,64,128, with eight
fixed seeded sequences. Each complete sequence is inverted to return ideally
to |00>. The inverse includes the interleaved CZ operations. Barriers preserve
the intended benchmark cycles and interleaved CZs; no off-edge routing is used.
This is an implementation of the two-qubit Clifford interleaved-benchmarking
protocol using Qiskit's Clifford operations, not an invocation or claimed
reproduction of the Qiskit Experiments analysis package.

Per-arm survival is fitted to A alpha^m+B, with separate A and B for the two
arms. The diagnostic gate-error ratio is (3/4)(1-alpha_interleaved/alpha_reference).
Negative estimates are retained and flagged, not silently clipped into a
physical claim. Paired sequence-seed resampling (500 replicates) characterizes
sequence variation; failed, boundary, flat and ill-conditioned fits are exposed.
These intervals are not independent-date uncertainty or bounds on model bias.
The approximately exponential, stationary, weakly gate-dependent noise
assumptions can fail. A short-circuit estimate is not a causal allocation of
full-width NILM error and is not a complete gate-set characterization.

For phase checks the first qubit is prepared in 0 or 1 and the second in |+>.
Apply n=0,1,4,16 CZ gates with barriers, then measure the target in X or Y and
the control in Z. Ideal target expectations are X=(-1)^(control*n), Y=0.
Both target quadratures and control-state flips are reported. This tests
population and phase-sensitive behavior without claiming a unique mechanism.
Binary classified outputs do not directly identify leakage outside the qubit
subspace. These two-qubit circuits do not reproduce full-width crosstalk.

## Execution safeguards and provenance

- Use only the existing Open Plan account and ibm_fez; no paid override.
- One job, maximum QPU execution time 45 seconds. Planning/accounting cap is
  55 seconds plus a separate 20-second free reserve. Execution limits do not
  cap queue wait or every accounting overhead.
- Immediately before submission, recheck allowance and add a conservative
  reserve for the original failed ladder job while its accounting is pending.
  A pending reserve is not reported as a finalized charge.
- Estimate durations against the saved preparation snapshot and retain the
  original repetition delay. Preparation estimates include fixed and per-circuit
  planning overhead; they are not IBM guarantees.
- Disable dynamical decoupling and gate/measurement twirling. No outcome-based
  changes to layouts, angles, sequences, shots or selection after freezing.
- Randomize all 228 PUBs with fixed order seed 20260915.
- Archive native QPY/QASM, bindings, physical mappings, public calibration,
  versions, source hashes, protocol, ideal checks, immutable submission intent,
  job ID, all packed outcomes, accounting and analysis.
- An ambiguous submission or failed job stops the campaign. Never retry
  automatically. Existing experiments and their source files remain unchanged.

The circuit, parameter and probability checks are required before submission.
Post-run byte-to-count reconstruction checks all returned samples. Published
hardware conclusions remain descriptive unless stronger independent evidence
is collected. No quantum advantage or speed advantage is tested here.

## Method references

- Magesan et al., “Efficient measurement of quantum gate error by interleaved
  randomized benchmarking,” PRL 109,080505 (2012): https://arxiv.org/abs/1203.4550
- Qiskit Experiments protocol documentation:
  https://qiskit-community.github.io/qiskit-experiments/manuals/verification/randomized_benchmarking.html
- IBM shot initialization:
  https://quantum.cloud.ibm.com/docs/en/guides/repetition-rate-execution
- IBM Sampler execution controls:
  https://quantum.cloud.ibm.com/docs/en/guides/sampler-options
