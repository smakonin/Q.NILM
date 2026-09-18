# Q.NILM D-Wave Ocean scaling benchmark

Status: classical Ocean readiness benchmark; no QPU execution

| Data | Logical variables | Sampler | Runs | Bit accuracy | F1 | MCC | Aggregate MAE (W) | Wall time (s) | Best-known hit rate |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| R1Hz | 48 | simulated | 3 | 0.854 | 0.829 | 0.703 | 335.13 | 0.0131 | 1.000 |
| R1Hz | 48 | tabu | 3 | 0.854 | 0.829 | 0.703 | 335.13 | 1.6323 | 1.000 |
| synthetic | 12 | simulated | 3 | 0.917 | 0.916 | 0.827 | 14.25 | 0.0082 | 1.000 |
| synthetic | 12 | tabu | 3 | 0.917 | 0.916 | 0.827 | 14.25 | 1.6324 | 1.000 |
| synthetic | 24 | simulated | 3 | 0.972 | 0.967 | 0.943 | 17.62 | 0.0077 | 1.000 |
| synthetic | 24 | tabu | 3 | 0.972 | 0.967 | 0.943 | 17.62 | 1.6323 | 1.000 |
| synthetic | 48 | simulated | 3 | 0.993 | 0.994 | 0.986 | 15.58 | 0.0163 | 1.000 |
| synthetic | 48 | tabu | 3 | 0.889 | 0.892 | 0.779 | 87.62 | 1.6322 | 0.333 |
| synthetic | 96 | simulated | 3 | 0.972 | 0.975 | 0.942 | 19.36 | 0.0324 | 1.000 |
| synthetic | 96 | tabu | 3 | 0.906 | 0.912 | 0.807 | 50.54 | 1.6325 | 0.000 |
| synthetic | 192 | simulated | 3 | 0.557 | 0.588 | 0.105 | 28.44 | 0.0749 | 0.333 |
| synthetic | 192 | tabu | 3 | 0.547 | 0.522 | 0.116 | 80.44 | 1.6328 | 0.000 |
| synthetic | 288 | simulated | 3 | 0.539 | 0.538 | 0.061 | 17.01 | 0.1268 | 0.000 |
| synthetic | 288 | tabu | 3 | 0.544 | 0.432 | 0.080 | 42.25 | 1.6385 | 0.667 |

## Offline embedding estimates

| Data | Logical variables | Logical couplers | Ideal physical qubits | Maximum chain | Result |
|---|---:|---:|---:|---:|---|
| synthetic | 12 | 21 | 12 | 1 | embedded |
| synthetic | 24 | 45 | 26 | 2 | embedded |
| synthetic | 48 | 162 | 72 | 2 | embedded |
| synthetic | 96 | 330 | 151 | 2 | embedded |
| synthetic | 192 | 1236 | 632 | 8 | embedded |
| synthetic | 288 | 2718 | --- | --- | not found in timeout |
| R1Hz | 48 | 93 | 52 | 2 | embedded |

Best-known values combine tested solver samples and the reference state; they are certified optima only where exact enumeration is reported.
R1Hz labels are deterministic circuit-derived proxies, not manual appliance ground truth.
Embedding figures use an ideal defect-free Zephyr(12) graph and are not live-QPU results.
Direct QPU and Leap hybrid runs must be reported separately from these local classical results.
