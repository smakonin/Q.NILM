# Q.NILM IBM experiment register

Keep completed results immutable. Number full-coverage NILM campaigns
separately from selected-circuit pilots and component diagnostics; a later
experiment does not replace an earlier result.

| Publication label | Scope | Hardware identifier | Result archive |
|---|---|---|---|
| Q.NILM, IBM — Run 1 (Table VI) | Full-coverage categorical campaign: 30 windows, 86,396 blocks, 2,451 circuits, 627,456 shots; 315.686 W macro appliance MAE including declared fallback | 129 stage jobs, beginning `dajnp09hvn6c73cul07g`; complete IDs in archive | `results/quantum_heldout/hardware/` |
| Original IBM pilot (Table IV) | Selected nine-qubit legacy-objective circuit; not full NILM accuracy | `dajl817i3e6s738qdka0` | `results/ibm_qpu_pilot/` |
| Corrected IBM pilot | Selected nine-qubit corrected-objective circuit; not full NILM accuracy | `dajn53r9k43c73ahdmag` | `results/ibm_corrected_qpu_check/` |
| Component diagnostic 1 — execution failed | Planned four stage conditions, two widths, three training examples, 24 × 1,024 shots; IBM error 9701, no returned measurements or accuracy result | `dajvplj9k43c73ahpfo0` | `results/quantum_diagnostics/ladder/failure_status.json` |
| Component diagnostic 1 — retry 1, complete and audited | One user-authorized unchanged retry; all 24 circuits and 24,576 raw shots collected; 9 s finalized QPU charge; independent compiled-ideal and raw-count audits passed | `dak1kar9k43c73ahrfd0` | `results/quantum_diagnostics/ladder_retry_001/` |
| Physical diagnostic 1 — complete and audited | Three user-authorized groups: local preparation/reset/readout, matched zero/nominal cost angles, and CZ benchmarking/phase checks; all 228 circuits and 94,208 raw shots collected; 28 s finalized QPU charge; no new NILM MAE | `dak582ni3e6s738r26a0` | `results/quantum_diagnostics/physical_001/` |
| Component diagnostic 1 — independent-date repeat, complete and audited | September 18 unchanged repeat of the September 14 compiled circuits: 24 circuits, 24,576 shots, 9 s finalized QPU charge; not a second full-accuracy campaign | `damrmo02fm4c73f3ejt0` | `results/quantum_diagnostics/ladder_date_repeat_001/` |

The component jobs were submitted on 14 September 2026. A `job.json`
receipt proves submission, not collection. The retry's `result.json`,
`metrics.json`, `independent_ideal_audit.json` and
`independent_counts_audit.json` establish its completed and checked result.
The first attempt finished with ERROR at 2026-09-14 14:34:45 UTC. IBM reported
error 9701, “Temporary Internal Error.” Its failed-result exception and
read-only job status were checked. The initial usage record reports zero
QPU charge with accounting still pending; this is not finalized billing.
The completed full Run 1 and Table VI are unchanged. After explicit user
approval, one unchanged retry was submitted at 2026-09-14 16:00 UTC. Its
preflight observed 157 free seconds and reserved 68 seconds for unresolved
old-job accounting, the new 60-second planning cap and a 20-second free
reserve (148 seconds total). The 68-second amount is a conservative
running-to-finished planning reserve, not an asserted QPU charge. The copied
scientific plan and all compiled artifacts retain identical hashes. The new
job has a 50-second execution limit, no paid override and no automatic retry.
The retry finished at 2026-09-14 16:01:16 UTC and was collected at
16:03:08 UTC. Its accounting is finalized at 9 seconds, independently checked
from the saved metrics; this does not settle the failed first job's provisional
accounting. Both widths retain all 1,024 raw shots per circuit and three
training examples per condition. Preparation/mixer/cost/full-circuit valid
fractions are 74.056%/69.759%/25.911%/24.154% at 12 logical qubits and
19.336%/13.509%/1.074%/0.456% at 24 logical qubits. There is no fallback,
postselected denominator, or derived full-NILM accuracy result.

Future complete NILM campaigns receive Run 2, Run 3, etc., with separate
Table VI rows and immutable archives. Record changed models, circuits,
budgets and exposure status explicitly. The 129 adaptive stages of Run 1
are not 129 independent full evaluations. Component diagnostic 1 is not
Run 2 and must not be assigned an invented MAE.

Physical diagnostic 1 was submitted on 14 September 2026 at 20:07:38 UTC.
All 228 circuits passed ideal-output checks before submission, with maximum
probability discrepancy 1.403e-9, and all 379 software tests passed. Its
[frozen protocol](physical_diagnostics_protocol.md) sets one job, a 45-second
execution limit, a 55-second accounting/planning cap and no automatic retry.
Immediately before submission the Open Plan had 148 seconds available; the
guard reserved 68 seconds for the failed component job's still-pending
accounting plus a separate 20 seconds (143 seconds required in total).
The 36.096-second usage estimate is planning telemetry, not a finalized
charge. The scientific protocol and analysis were frozen before this job's
results were observed. This diagnostic is not a second full NILM run.

### Physical diagnostic 1: completed results

The job finished at 2026-09-14 20:11:00 UTC. All 228 circuits and 94,208
measurements were returned, with **28 seconds of finalized QPU charge**
(24.904 seconds of recorded circuit execution). Creation-to-finish elapsed
time was 201.567 seconds; QPU usage is not end-to-end turnaround time. The
job remained within its free-plan planning budget, with no retry or paid
override. This does not settle the failed component job's earlier accounting.

Evidence is retained in
[`result.json`](../results/quantum_diagnostics/physical_001/result.json),
[`metrics.json`](../results/quantum_diagnostics/physical_001/metrics.json),
[`analysis.json`](../results/quantum_diagnostics/physical_001/analysis.json), and
[`independent_result_audit.json`](../results/quantum_diagnostics/physical_001/independent_result_audit.json).
The independent audit decoded every packed measurement through a separate
implementation, reconciled primary metrics and frozen-file hashes, and
cross-checked the four randomized-benchmarking fits with a different optimizer.
Bootstrap quantiles were checked, but bootstrap resamples were not
independently refitted. All denominators below retain the raw shots.

| Diagnostic group | Observed result | Interpretation and limit |
|---|---|---|
| Local preparation/readout/reset: 24 circuits, 24,576 shots | Three local W-state repeats yielded 2,811/3,072 valid samples (91.504%) on qubits 131–133 and 2,933/3,072 (95.475%) on 141–143. Qubit 131 showed 4.883–8.594% basis-pattern assignment error and 9.668% error after prepare-111/reset/readout. | The earlier large marginal bias on 132/133 did not recur in these smaller local W circuits. These are different circuits at a later time, not evidence that the earlier anomaly was absent or that the qubits are fault-free. Preparation, reset and readout contributions are not separately identified. |
| Matched physical cost control: 12 circuits, 12,288 shots | At 12 logical qubits, zero/nominal cost angles yielded 896/3,072 (29.167%) and 882/3,072 (28.711%) valid samples. At 24 logical qubits both yielded 39/3,072 (1.270%). | Setting cost angles to zero did not restore feasibility when the same native gate sequence, routing and mapping were retained. This supports a physical-execution limitation; it is not a formal equivalence result or an attribution to CZ alone. |
| Gate-focused checks: 160 RB circuits plus 32 Ramsey circuits, 57,344 shots | Repeated-CZ Ramsey checks show a substantial negative Y component on edge 132–133 after 16 CZ gates, for both prepared control states (table below). The short interleaved-RB study does not resolve a difference in average gate error between the two edges. | A phase-sensitive sequence-level anomaly is localized, but the exact gate mechanism remains unidentified. No duration-matched idle control was run; detuning, idle evolution, crosstalk and state-preparation/readout errors remain alternatives. |

The matched cost comparison reused the existing compiled W-plus-cost circuits
without recompilation. Only symbolic cost angles were changed; native fixed
phase offsets and the entangling/single-qubit gate sequence were preserved.
Each width/arm pools three frozen training examples at 1,024 shots each.
The equal 24-qubit counts do not imply identical output distributions or prove
that cost angles have no effect.

The phase probe measures the target's Y expectation, whose ideal value is
zero for every tested repetition count and both control preparations. The
following is the prespecified 16-CZ endpoint; the archive also retains all
0-, 1- and 4-CZ conditions and the companion X measurements.

| Physical edge | Prepared control | Target Y after 16 CZ | 95% within-circuit Wilson interval |
|---|---|---:|---:|
| 132–133 | 0 | −0.4922 | [−0.5637, −0.4133] |
| 141–142 | 0 | +0.0117 | [−0.0747, +0.0979] |
| 132–133 | 1 | −0.3789 | [−0.4560, −0.2962] |
| 141–142 | 1 | +0.1016 | [+0.0150, +0.1867] |

Every expectation uses all 512 shots, without conditioning on the measured
control. These are individual binomial intervals, not simultaneous intervals
or uncertainty over calibration dates. With control prepared as zero, the
132–133 edge's measured X/Y vector has a wrapped phase of −0.593 radians
(approximately −34 degrees) at 16 CZ; this is not an isolated per-gate
overrotation estimate. The similar negative Y shift with both control states
is compatible with a common local phase contribution, but does not establish
that explanation.

Interleaved randomized benchmarking used five sequence lengths, eight paired
random seeds and 256 shots per circuit. The inferred CZ error estimates are
0.0348% on 132–133 (95% paired-seed bootstrap interval −0.2885% to +0.4214%)
and 0.2489% on 141–142 (−0.0164% to +0.4559%). The suspect-minus-comparison
contrast is −0.2141 percentage points, with interval [−0.6272, +0.2288].
Both individual fits are flagged `gate_error_not_resolved_from_zero`.
Negative interval endpoints indicate limited precision in this estimator,
not negative physical error. These results do not establish that either edge
is better, healthy, or the sole cause of full-circuit failure.

The next discriminating experiment would compare repeated CZ with
duration-matched idle and single-qubit phase controls on the same qubits,
then repeat under another calibration snapshot. That experiment has **not**
been submitted. No leakage measurement, new full-held-out MAE or
quantum-advantage claim follows from these diagnostics. Table VI's completed
full-coverage IBM Run 1 remains unchanged.

### Component diagnostic 1: independent-date repeat

On September 18 the unchanged compiled component ladder was run once on
`ibm_fez`, with three frozen training examples per condition and 1,024 shots
per circuit. The new job finished at 22:30:07.631713 UTC; finalized usage
was 9 seconds. The preflight checked the Open Plan, a 60-second planning
cap plus 20-second reserve, and the previous job's final accounting. The
previous failed job is now confirmed finalized at zero charge; its original
provisional receipt is preserved rather than rewritten. No paid override,
automatic retry, or new full-coverage NILM inference was performed.

| Width | Condition | September 14 feasible / 3,072 | September 18 feasible / 3,072 |
|---|---|---:|---:|
| 12 logical qubits | W preparation | 2,275 (74.056%) | 2,503 (81.478%) |
| 12 logical qubits | W + mixer | 2,143 (69.759%) | 2,372 (77.214%) |
| 12 logical qubits | W + cost | 796 (25.911%) | 891 (29.004%) |
| 12 logical qubits | Full | 742 (24.154%) | 825 (26.855%) |
| 24 logical qubits | W preparation | 594 (19.336%) | 1,363 (44.368%) |
| 24 logical qubits | W + mixer | 415 (13.509%) | 977 (31.803%) |
| 24 logical qubits | W + cost | 33 (1.074%) | 27 (0.879%) |
| 24 logical qubits | Full | 14 (0.456%) | 23 (0.749%) |

The repeat retained all raw shots. Independent packed-count and compiled
ideal audits passed; the plan, QPY/OpenQASM, parameter bindings, placements,
PUB order and options match the first completed component date. A fresh
backend snapshot is archived separately. Improved preparation did not
restore feasibility for the 24-qubit cost/full circuits. Two dates show
recurrence under these conditions, not a general calibration-drift estimate,
an isolated gate mechanism, or an appliance-accuracy improvement. Table VI
still reports only the original full IBM Run 1.
