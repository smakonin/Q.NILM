# Q.NILM corrected IBM follow-up: controls and staged hardware plan

Status: **12 local controls and one corrected p=1 IBM hardware confirmation completed**. The local-control runner itself never accesses an account or submits jobs; the parent task separately submitted the single bounded Open Plan job described below. Expanded p=2/multiwindow/multidate hardware campaigns have not been executed or authorized by this plan. This remains a circuit-validation study, not a quantum-advantage experiment. Complete local results are in `results/ibm_followup_controls/run_001/summary.json`; the corrected hardware record is in `results/ibm_corrected_qpu_check/`.

## What ran locally

The corrected, nine-qubit selected-channel pilot was evaluated at QAOA depths 1 and 2, with three compilation seeds (7, 17, 27). Each of the six compiled circuits was run for 4096 shots both ideally and with noise derived from the **same archived ibm_fez calibration used for the original hardware pilot**, updated September 13, 2026, at 18:29:38 PDT. Thus 12 local experiments and 49,152 simulated shots were completed. These are not twelve hardware runs or independent datasets.

For p=1 we froze the corrected pilot's classical noiseless grid optimum. For p=2, each p=1 angle was divided equally across two layers: gamma = (2.7584715983, 2.7584715983), beta = (1.2161003820, 1.2161003820). This deliberately simple depth ablation is **not optimized p=2**; splitting angles does not preserve a noncommuting QAOA evolution. Infer neither a general p=2 failure nor a depth advantage from this comparison.

| Local control | Exact optimum frequency across three seeds | Mean objective across three seeds | Compiled CZ gates / circuit depth |
|---|---:|---:|---:|
| p=1 ideal | 1.880–2.002% | 45.13–48.65 million | 38–46 / 100–114 |
| p=1 matched-noise | 1.733–1.855% | 58.08–58.58 million | Same compiled circuits |
| p=2 ideal, split angles | 0.244% | 218.65–223.53 million | 79–84 / 152–162 |
| p=2 matched-noise, split angles | 0.269–0.391% | 222.63–228.05 million | Same compiled circuits |

Every experiment's lowest-cost sample matches all nine proxy states. However, uniform random sampling has a **99.9667% probability of finding this tiny problem's unique optimum at least once in 4096 shots**. Best-sample agreement alone therefore provides almost no evidence of computational advantage. The analytic ideal optimum probabilities are 1.8334% for p=1 and 0.2618% for split-angle p=2, versus 0.1953% per uniform shot. No statistical hardware claim is made from three seeded simulator draws.

## Audit and limitations

All circuits agree with the independent NumPy implementation; the maximum compiled measurement-probability discrepancy is 1.80e-8. Only nine active physical qubits are retained in the local simulation, with explicit remapping of both measurement bits and every included noise channel. Seven regression tests cover this remapping, missing frequency data, and corrected-input requirements.

The model includes calibrated local readout errors and gate depolarization/thermal relaxation at zero temperature. Fez's saved calibration lacks qubit frequencies; Aer's public gate/readout builders handle this without inventing frequencies. Unscheduled idle effects, correlated readout, crosstalk, calibration drift, and other unmodelled effects are excluded. This is an approximate snapshot simulation, not a digital duplicate of the QPU. [Aer noise documentation](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.NoiseModel.html) and [IBM local-testing guidance](https://quantum.cloud.ibm.com/docs/en/guides/local-testing-mode).

The input was selected retrospectively, uses selected-channel aggregate residuals rather than whole-house mains, and shares model-estimation data with the diagnostic reference. These controls do not replace held-out NILM tests. The p=1 parameter search itself used 1271 exact classical statevector objective evaluations; it cannot be excluded from an end-to-end speed comparison. Before a confirmatory depth comparison, optimize both depths under a declared and comparable parameter-evaluation budget, with all costs recorded.

## Completed corrected hardware confirmation

The parent task executed **one** p=1 job on `ibm_fez`: `dajn53r9k43c73ahdmag`, September 13, 2026, at approximately 21:05 PDT (September 14, 04:05 UTC). It used 4096 shots, seed 7, Open Plan, a 60-second maximum QPU execution limit, and the frozen corrected pilot parameters. Freshly compiled resources were nine active qubits, 46 CZ gates, and depth 112. No p=2, held-out-window, or multidate hardware campaign is implied.

Independent read-only audit confirmed every frozen file checksum, the input's equality with `results/pilot_corrected/summary.json`, and agreement between the serialized Runtime result and saved counts. Direct evaluation of all 512 bitstrings independently reproduced the unique optimum and the following metrics:

| Corrected hardware measure | Result |
|---|---:|
| Exact-optimum observations | 75/4096 = 1.8311% |
| Conditional Wilson 95% interval for per-shot optimum frequency | 1.4633–2.2891% |
| Lowest observed objective / exact objective | 1,165,486.4338 / 1,165,486.4338 |
| Mean sampled objective | 71,342,241.0971 |
| Lowest-cost sample's diagnostic proxy-state agreement | 9/9 |
| Duration-weighted aggregate MAE at segment-mean grain | 100.3526 W |
| Accounted QPU usage / circuit execution | 3 s / 1.073021184 s |
| Server creation-to-finish elapsed time | 7.607168 s |

The shot interval is conditional on an independent-binomial approximation within this job, not evidence of repeatability over jobs, dates, or households. The mean sampled objective is **52.76% higher (worse)** than the corrected p=1 ideal expectation of 46,703,500.3156 despite the nearly identical optimum frequency (ideal: 1.8334%). This illustrates why the best sample and optimum frequency cannot substitute for the complete distribution or end-to-end performance.

The compiled QASM is byte-identical to the earlier local p=1 seed-7 control, but the **calibration is not**: this hardware preparation archived an update at 20:43:06 PDT, whereas local controls used the original 18:29:38 PDT snapshot. Twenty-seven of 343 saved parameter values on the active qubits and their included calibrated gates changed. Accordingly, the earlier noisy control is **not calibration-matched to this new hardware job**. Its mean objective (58,489,532.1628) is a contextual comparison only, not a controlled attribution of hardware noise. A fresh-calibration noisy rerun would be needed for that comparison; it still would not model all real-device noise.

The observed 9/9 agreement is retrospective proxy agreement on the selected diagnostic interval, not held-out NILM accuracy. The former 8/9 result used a different, erroneous preprocessing objective; the change does not show improved quantum hardware or quantum advantage. Uniform sampling still has a 99.9667% probability of hitting the corrected unique optimum at least once at the same shot budget.

## Staged hardware experiment

**Stage 0 is now completed**, as audited above. The earlier `single_p1_recommendation.json` remains an immutable record of its proposed limits, not a request to submit it again. Do not submit p=2 or expanded stages automatically. Stages A and B below are possible future designs, not completed or approved campaigns; actual future designs must be frozen prospectively and must not retroactively call this diagnostic job a preregistered confirmatory observation.

| Stage | Predeclared design | Circuits / shots | Preliminary QPU-use planning range |
|---|---|---:|---:|
| 0: completed corrected check | One fixed p=1 pilot, seed 7 | 1 / 4096 | Observed accounted usage: 3 seconds; configured limit 60 seconds |
| A: corrected-circuit check | Fixed pilot, p=1 and p=2, three layouts | 6 / 24,576 | 18–36 seconds |
| B: held-out replication | 30 untouched windows, two depths, three layouts, three calibration dates | 540 / 2,211,840 | 27–54 minutes |

The ranges are **not IBM estimates or guarantees**: they extrapolate the historic 3-second pilot usage with up to a twofold allowance for depth, and omit uncertain job overhead and queue delays. Stage A should start as one explicitly limited job with `max_execution_time=60` seconds; this limits QPU usage, not waiting time. Stage B needs a fresh allocation/budget decision and cannot be assumed to fit the Open Plan, whose currently documented allowance is 10 minutes per rolling 28 days. [IBM execution limits](https://quantum.cloud.ibm.com/docs/en/guides/max-execution-time).

Before any hardware submission:

1. Check the active instance and remaining allowance read-only; require Open Plan and reject paid execution without an override in this campaign.
2. Select an operational live backend; archive fresh calibration, recompile, and rerun the circuit audit. The archived resources above are planning data, not current hardware availability.
3. Freeze model, unseen windows, angle-selection procedure, metrics, layouts, repetitions, and stopping rule before examining hardware outputs. Include exact temporal optimization, tuned classical competitors, ideal/noisy QAOA, and uniform sampling.
4. Implement and test the depth-two submission path: the existing hardware pilot runner accepts only frozen p=1 inputs. The new local-control script deliberately has **no submission capability**, so its safety guard is structural, not a tested hardware campaign runner.
5. Enforce total job/shot budgets, an exclusive pre-submission record, and no blind retries after ambiguous failures. Do not start Stage B until held-out model quality and classical baselines justify it.

A quantum-advantage claim would require reproducible improvement over the strongest applicable classical methods on a predeclared useful task, with uncertainty and the full parameter-learning/compilation/execution/decoding cost accounted for. Merely using more qubits, beating uniform sampling, obtaining a good minimum from thousands of shots, or observing differences between layouts is insufficient. The current binary temporal objective also has an efficient exact dynamic-programming baseline when the number of appliances is small; lengthening the timeline alone does not defeat it.

## Reproduction

Run `env PYTHONPATH=src .venv/bin/python scripts/run_ibm_followup_controls.py --output-dir results/ibm_followup_controls/run_002` from the repository root. Use a new output directory: completed runs are never overwritten. Python/Qiskit/Aer versions, source hashes, counts, compiled circuits, compact noise descriptions, and measurement mappings are archived for every completed run. No account setup or network connection is needed.
