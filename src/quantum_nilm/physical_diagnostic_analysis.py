"""Raw-count analysis for the separately authorized physical diagnostic tests.

No hardware calls, readout mitigation, NILM reference labels or file writes.
Bitstrings use Qiskit order: the rightmost bit is classical bit zero.
IRB uses independent SPAM nuisance parameters for each arm and paired seed-
cluster resampling. Its intervals exclude between-job drift and systematic
IRB model error. Binary counts cannot establish leakage.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
from scipy.optimize import minimize_scalar

BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_SEED = 84107
RB_LENGTHS = (1, 4, 16, 64, 128)
SOURCES = ["https://arxiv.org/abs/1203.4550",
           "https://qiskit-community.github.io/qiskit-experiments/stubs/qiskit_experiments.library.randomized_benchmarking.InterleavedRBAnalysis.html"]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=0):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)) and value >= minimum,
             f"{name} must be an integer >= {minimum}")
    return int(value)


def _counts(row):
    bits = _integer(row["n_clbits"], "n_clbits", 1)
    shots = _integer(row["shots"], "shots", 1)
    _require(isinstance(row["raw_counts"], dict) and row["raw_counts"], "raw_counts must be a nonempty dictionary")
    counts = {}
    for key, count in row["raw_counts"].items():
        _require(isinstance(key, str), "Count keys must be binary strings")
        normalized = key.replace(" ", "")
        _require(len(normalized) == bits and set(normalized) <= {"0", "1"}, "Count key width or alphabet mismatch")
        _require(normalized not in counts, "Duplicate normalized count key")
        counts[normalized] = _integer(count, "count")
    _require(sum(counts.values()) == shots, "Raw-count sum differs from declared shots")
    if "logical_measurement_bit_to_physical" in row:
        mapping = row["logical_measurement_bit_to_physical"]
        _require(isinstance(mapping, (list, tuple)) and len(mapping) == bits, "Measurement mapping has the wrong width")
        for physical in mapping:
            _integer(physical, "physical qubit")
        _require(len(set(mapping)) == bits, "Measurement mapping repeats a physical qubit")
        if row.get("group") == "local":
            _require(list(mapping) == row["physical_qubits"], "Local bit order disagrees with physical mapping")
        elif row.get("group") == "gate":
            _require(list(mapping) == row["edge"], "Gate bit order disagrees with physical mapping")
    return counts, bits, shots


def proportion(count, shots):
    """Point frequency and Wilson95% interval, with a within-row iid caveat."""
    if not shots:
        return {"count": int(count), "shots": 0, "fraction": None, "wilson_95": None}
    p, z = count / shots, 1.959963984540054
    denominator = 1 + z * z / shots
    center = (p + z * z / (2 * shots)) / denominator
    radius = z * math.sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots)) / denominator
    return {"count": int(count), "shots": int(shots), "fraction": float(p),
            "wilson_95": [max(0., center - radius), min(1., center + radius)]}


def _one_counts(counts, bits):
    return [sum(n for text, n in counts.items() if (int(text, 2) >> bit) & 1) for bit in range(bits)]


def _local(rows):
    placements = {}
    for label in sorted({r["placementlabel"] for r in rows}):
        selected = [r for r in rows if r["placementlabel"] == label]
        physical = selected[0]["physical_qubits"]
        basis, resets, wrows = [], [], []
        for row in selected:
            counts, bits, shots = _counts(row)
            _require(bits == 3 and row["physical_qubits"] == physical and len(set(physical)) == 3,
                     "Local-control width or placement mismatch")
            ones = _one_counts(counts, bits)
            common = {"id": row["id"], "shots": shots,
                      "bit_one": [proportion(n, shots) for n in ones]}
            if row["kind"] in ("basis", "reset"):
                expected = row["input_state"] if row["kind"] == "basis" else "000"
                _require(isinstance(expected, str) and len(expected) == 3 and set(expected) <= {"0", "1"}, "Invalid basis state")
                errors = [shots - ones[i] if expected[-1 - i] == "1" else ones[i] for i in range(bits)]
                result = {**common, "expected_bitstring": expected,
                          "exact_pattern_error": proportion(shots - counts.get(expected, 0), shots),
                          "bit_error": [proportion(n, shots) for n in errors],
                          "measured_distribution": {f"{i:03b}": counts.get(f"{i:03b}", 0) / shots for i in range(8)}}
                (basis if row["kind"] == "basis" else resets).append(result)
            elif row["kind"] == "w":
                feasible = sum(n for text, n in counts.items() if text.count("1") == 1)
                tv = .5 * math.fsum(abs(counts.get(f"{i:03b}", 0) / shots - (1 / 3 if i in (1, 2, 4) else 0)) for i in range(8))
                wrows.append({**common, "repeat": row["repeat"], "onehot_feasibility": proportion(feasible, shots),
                              "measurement_tv_to_uniform_onehot": tv})
            else:
                raise ValueError("Unknown local-control kind")
        _require(len({r["expected_bitstring"] for r in basis}) == len(basis), "Duplicate local basis condition")
        _require(len(resets) <= 1 and len({r["repeat"] for r in wrows}) == len(wrows), "Duplicate local reset/W condition")
        placements[label] = {"physical_qubits_in_classical_bit_order": physical,
            "basis": sorted(basis, key=lambda r: r["expected_bitstring"]), "reset": resets, "w": wrows,
            "complete_basis_grid": len(basis) == 8,
            "w_pooled_feasibility": (sum(r["onehot_feasibility"]["count"] for r in wrows) / sum(r["shots"] for r in wrows)) if wrows else None,
            "interpretation": "Basis errors include preparation and readout; reset is111 preparation followed by reset. W feasibility/measurement TV are not quantum-state fidelity."}
    return placements


def _cost(rows):
    results, keys = [], set()
    for row in rows:
        counts, bits, shots = _counts(row)
        width = _integer(row["width"], "width", 1)
        _require(width in (1, 2) and row["state_counts"] == [4, 3, 2, 3] and bits == 12 * width,
                 "Cost register shape mismatch")
        _require(row["arm"] in ("zero", "nominal"), "Unknown cost arm")
        key = (width, row["example_id"], row["arm"])
        _require(key not in keys, "Duplicate cost condition")
        keys.add(key)
        registers, all_feasible, offset = [], 0, 0
        spans = []
        for interval in range(width):
            for channel, size in enumerate(row["state_counts"]):
                spans.append((interval, channel, offset, size))
                offset += size
        for text, n in counts.items():
            value = int(text, 2)
            if all(((value >> offset) & ((1 << size) - 1)).bit_count() == 1 for _, _, offset, size in spans):
                all_feasible += n
        for interval, channel, offset, size in spans:
            weights = [sum(n for text, n in counts.items() if ((int(text, 2) >> offset) & ((1 << size) - 1)).bit_count() == k)
                       for k in range(size + 1)]
            registers.append({"interval": interval, "channel": channel, "classical_bits": list(range(offset, offset + size)),
                              "onehot_feasibility": proportion(weights[1], shots), "hamming_weight_counts": weights})
        results.append({"id": row["id"], "width": width, "example_id": row["example_id"], "arm": row["arm"],
                        "shots": shots, "all_register_onehot": proportion(all_feasible, shots), "registers": registers,
                        "bit_one": [proportion(n, shots) for n in _one_counts(counts, bits)]})
    paired = []
    by_key = {(r["width"], r["example_id"], r["arm"]): r for r in results}
    for width, example in sorted({(r["width"], r["example_id"]) for r in results}):
        a, b = by_key.get((width, example, "nominal")), by_key.get((width, example, "zero"))
        paired.append({"width": width, "example_id": example, "complete_pair": a is not None and b is not None,
                       "nominal_minus_zero_feasibility": None if a is None or b is None else a["all_register_onehot"]["fraction"] - b["all_register_onehot"]["fraction"]})
    return {"conditions": results, "paired_conditions": paired,
            "interpretation": "Unconditional raw one-hot mass and bit marginals; no appliance reference MAE or causal attribution. Zero/nominal must share the frozen compiled schedule."}


def _profile_ab(alpha, lengths, y, weights):
    x = np.power(alpha, lengths)
    sx, sy, sxx, sxy, total = np.dot(weights, x), np.dot(weights, y), np.dot(weights, x * x), np.dot(weights, x * y), weights.sum()
    determinant = total * sxx - sx * sx
    candidates = []
    if determinant > 1e-18:
        a = (total * sxy - sx * sy) / determinant
        b = (sxx * sy - sx * sxy) / determinant
        if a >= 0 and b >= 0 and a + b <= 1:
            candidates.append((float(a), float(b)))
    candidates.append((0., float(np.clip(sy / total, 0, 1))))
    candidates.append((float(np.clip(sxy / sxx, 0, 1)) if sxx > 0 else 0., 0.))
    v = 1 - x
    denominator = np.dot(weights, v * v)
    a = float(np.clip(np.dot(weights, v * (1 - y)) / denominator, 0, 1)) if denominator > 0 else 0.
    candidates.append((a, 1 - a))
    best = min((float(np.dot(weights, (a * x + b - y) ** 2)), a, b) for a, b in candidates)
    return best


def fit_rb_decay(lengths, probabilities, weights=None):
    """Constrained A*alpha**m+B fit; report identifiability, never force r>=0.

    Profile linear A,B over the triangle A>=0,B>=0,A+B<=1; search alpha in
    [0,1] using a fixed grid and bounded scalar refinements. Least squares
    weights are shot counts, not data-dependent inverse binomial variances.
    """
    x, y = np.asarray(lengths, dtype=float), np.asarray(probabilities, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    _require(x.ndim == 1 and len(x) >= 4 and y.shape == w.shape == x.shape and np.all(np.isfinite(x))
             and np.all(np.isfinite(y)) and np.all(np.isfinite(w)) and np.all(x > 0)
             and len(set(x)) == len(x) and np.all((0 <= y) & (y <= 1)) and np.all(w > 0), "Invalid RB fit arrays")
    w = w / w.sum()
    grid = np.unique(np.r_[np.linspace(0., 1., 25), .9, .95, .98, .99, .995, .999, .9999])
    scores = [_profile_ab(alpha, x, y, w) for alpha in grid]
    candidates = [(s[0], float(alpha), s[1], s[2]) for alpha, s in zip(grid, scores)]
    for index in sorted(range(len(grid)), key=lambda i: scores[i][0])[:2]:
        lo, hi = grid[max(0, index - 1)], grid[min(len(grid) - 1, index + 1)]
        if hi > lo:
            result = minimize_scalar(lambda alpha: _profile_ab(alpha, x, y, w)[0], bounds=(lo, hi),
                                     method="bounded", options={"xatol": 1e-10, "maxiter": 80})
            if result.success:
                score, a, b = _profile_ab(result.x, x, y, w)
                candidates.append((score, float(result.x), a, b))
    score, alpha, a, b = min(candidates)
    predicted = a * alpha ** x + b
    jac = np.column_stack((alpha ** x, a * x * np.power(alpha, x - 1), np.ones_like(x)))
    norms = np.linalg.norm(jac, axis=0)
    condition = float(np.linalg.cond(jac / np.where(norms > 0, norms, 1)))
    flags = []
    if a < 1e-6 or np.ptp(predicted) < 1e-5 or not math.isfinite(condition) or condition > 1e8:
        flags.append("unidentified_or_nearly_flat_decay")
    if alpha <= 1e-6 or alpha >= 1 - 1e-6:
        flags.append("alpha_at_fit_boundary")
    if max(abs(predicted - y)) > .1:
        flags.append("large_exponential_model_residual")
    return {"A": a, "alpha": alpha, "B": b, "weighted_mean_squared_residual": score,
            "predicted_survival": predicted.tolist(), "residuals": (y - predicted).tolist(),
            "jacobian_condition": condition if math.isfinite(condition) else None,
            "identifiable": "unidentified_or_nearly_flat_decay" not in flags,
            "quality_flags": flags}


def _rb(rows, bootstrap_replicates, bootstrap_seed):
    by_edge = defaultdict(list)
    for row in rows:
        _require(row["n_clbits"] == 2 and row["arm"] in ("reference", "interleaved"), "Invalid RB width or arm")
        _integer(row["seed"], "RB seed")
        _integer(row["length"], "RB length", 1)
        by_edge[row["edge_label"]].append(row)
    edges, distributions = {}, {}
    for label in sorted(by_edge):
        selected = by_edge[label]
        edge = selected[0]["edge"]
        _require(all(r["edge"] == edge for r in selected), "RB edge label has multiple physical edges")
        seeds, lengths = sorted({r["seed"] for r in selected}), sorted({r["length"] for r in selected})
        _require(len(seeds) >= 2 and len(lengths) >= 4, "RB requires paired seed clusters and at least four lengths")
        indexed = {(r["arm"], r["seed"], r["length"]): r for r in selected}
        _require(len(indexed) == len(selected) == 2 * len(seeds) * len(lengths), "Missing or duplicate paired RB cell")
        matrices, denominators, points = {}, {}, []
        for arm in ("reference", "interleaved"):
            successes, shots = [], []
            for seed in seeds:
                counts_row, shots_row = [], []
                for length in lengths:
                    _require((arm, seed, length) in indexed, "Incomplete rectangular RB grid")
                    row = indexed[(arm, seed, length)]
                    counts, _, n = _counts(row)
                    counts_row.append(counts.get("00", 0))
                    shots_row.append(n)
                    points.append({"id": row["id"], "arm": arm, "seed": seed, "length": length,
                                   "survival": proportion(counts.get("00", 0), n)})
                successes.append(counts_row)
                shots.append(shots_row)
            matrices[arm], denominators[arm] = np.array(successes), np.array(shots)

        def fit(indices):
            fitted = {}
            for arm in matrices:
                n = denominators[arm][indices].sum(axis=0)
                p = matrices[arm][indices].sum(axis=0) / n
                fitted[arm] = fit_rb_decay(lengths, p, n)
            reference, interleaved = fitted["reference"], fitted["interleaved"]
            error = (.75 * (1 - interleaved["alpha"] / reference["alpha"])) if (
                reference["identifiable"] and interleaved["identifiable"] and reference["alpha"] > 1e-8) else None
            return fitted, error

        fitted, error = fit(np.arange(len(seeds)))
        rng = np.random.default_rng(bootstrap_seed)
        draws = rng.integers(0, len(seeds), size=(bootstrap_replicates, len(seeds)))
        bootstrap = []
        for draw in draws:
            _, estimate = fit(draw)
            bootstrap.append(estimate)
        valid = [x for x in bootstrap if x is not None and math.isfinite(x)]
        interval = np.quantile(valid, [.025, .975]).tolist() if valid else None
        flags = sorted(set(flag for arm in fitted.values() for flag in arm["quality_flags"]))
        if len(valid) < bootstrap_replicates:
            flags.append("bootstrap_contains_unidentified_fits")
        if interval is not None and interval[1] - interval[0] > .05:
            flags.append("wide_gate_error_interval_over_0.05")
        if error is not None and error < 0:
            flags.append("negative_gate_error_estimate_retained")
        if interval is not None and interval[0] <= 0 <= interval[1]:
            flags.append("gate_error_not_resolved_from_zero")
        status = "unresolved" if error is None or len(valid) < .8 * bootstrap_replicates else "fit_with_caveats" if flags else "fit"
        edges[label] = {"edge": edge, "seeds": seeds, "lengths": lengths, "points": points,
            "complete_prespecified_length_grid": lengths == list(RB_LENGTHS), "fits": fitted,
            "gate_error_estimate_unclipped": error, "seed_cluster_bootstrap_95": interval,
            "bootstrap_valid": len(valid), "bootstrap_unidentified": bootstrap_replicates - len(valid),
            "bootstrap_error_estimates": bootstrap, "status": status, "quality_flags": flags}
        distributions[label] = bootstrap
    contrast = None
    if set(edges) == {"suspect", "control"} and edges["suspect"]["seeds"] == edges["control"]["seeds"]:
        values = [a - b for a, b in zip(distributions["suspect"], distributions["control"]) if a is not None and b is not None]
        a, b = edges["suspect"]["gate_error_estimate_unclipped"], edges["control"]["gate_error_estimate_unclipped"]
        contrast = {"definition": "suspect minus control error estimate, using the same seed-cluster bootstrap draws",
                    "difference": None if a is None or b is None else a - b,
                    "paired_seed_cluster_95": np.quantile(values, [.025, .975]).tolist() if values else None,
                    "bootstrap_valid_pairs": len(values)}
    return {"edges": edges, "paired_edge_contrast": contrast,
            "model": "Separate per-arm A*alpha**m+B; A>=0, B>=0, A+B<=1 and0<=alpha<=1; independent SPAM parameters, shot-weighted least squares",
            "error_formula": "(3/4)*(1-alpha_interleaved/alpha_reference); never clipped to zero",
            "bootstrap": {"replicates": bootstrap_replicates, "seed": bootstrap_seed,
                          "unit": "one common random-sequence seed across all lengths, both arms and matched edges",
                          "interval": "percentile95%, conditional on identified bootstrap fits; all failed-fit counts retained"},
            "limitations": ["Eight seed clusters and five lengths give limited precision; shared prefixes are not independent trials",
                            "Intervals quantify seed-sampling uncertainty, not between-job/date drift or systematic IRB model error",
                            "Gate-error interpretation assumes approximately exponential/stationary and weakly gate-dependent noise; not a mechanism-specific causal proof"]}


def _ramsey(rows):
    results, keys = [], set()
    for row in rows:
        counts, bits, shots = _counts(row)
        _require(bits == 2 and row["control"] in (0, 1) and not isinstance(row["control"], bool)
                 and row["basis"] in ("X", "Y"), "Invalid Ramsey condition")
        repetitions = _integer(row["repetitions"], "Ramsey repetitions")
        ideal = float((-1) ** (row["control"] * repetitions)) if row["basis"] == "X" else 0.
        if "ideal_expectation" in row:
            _require(row["ideal_expectation"] == ideal, "Ramsey sign convention disagrees with frozen circuit ideal")
        key = (row["edge_label"], row["control"], row["basis"], repetitions)
        _require(key not in keys, "Duplicate Ramsey condition")
        keys.add(key)
        target_one = sum(n for text, n in counts.items() if (int(text, 2) >> 1) & 1)
        correct_control = {text: n for text, n in counts.items() if (int(text, 2) & 1) == row["control"]}
        retained = sum(correct_control.values())
        retained_one = sum(n for text, n in correct_control.items() if (int(text, 2) >> 1) & 1)
        target = proportion(target_one, shots)
        results.append({"id": row["id"], "edge_label": row["edge_label"], "edge": row["edge"],
            "control": row["control"], "basis": row["basis"], "repetitions": repetitions, "shots": shots,
            "target_pauli_expectation_all_shots": 1 - 2 * target["fraction"],
            "target_pauli_wilson_95": [1 - 2 * target["wilson_95"][1], 1 - 2 * target["wilson_95"][0]],
            "ideal_target_expectation": ideal,
            "control_flip": proportion(shots - retained, shots),
            "conditional_target_expectation_secondary": None if not retained else 1 - 2 * retained_one / retained,
            "conditional_retained_shots": retained})
    indexed = {(r["edge_label"], r["control"], r["repetitions"], r["basis"]): r for r in results}
    xy = []
    for label, control, n in sorted({(r["edge_label"], r["control"], r["repetitions"]) for r in results}):
        a, b = indexed.get((label, control, n, "X")), indexed.get((label, control, n, "Y"))
        if a is None or b is None:
            continue
        x, y = a["target_pauli_expectation_all_shots"], b["target_pauli_expectation_all_shots"]
        xy.append({"edge_label": label, "control": control, "repetitions": n, "X": x, "Y": y,
                   "raw_xy_magnitude_unclipped": math.hypot(x, y),
                   "wrapped_phase_rad": math.atan2(y, x) if math.hypot(x, y) > 1e-12 else None})
    return {"conditions": results, "xy_pairs": xy,
            "interpretation": "Target classical bit1 marginal is primary and never postselected; bit0 control flips and conditional target means are secondary. Raw X/Y visibility and wrapped phase include SPAM and do not uniquely identify coherent gate error."}


def analyze_rows(rows, *, bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED):
    """Analyze flattened circuit-spec rows with raw_counts, without mutation."""
    _require(isinstance(rows, (list, tuple)) and rows, "rows must be a nonempty sequence")
    _integer(bootstrap_replicates, "bootstrap_replicates", 1)
    _integer(bootstrap_seed, "bootstrap_seed")
    identifiers = set()
    groups = {"local": [], "cost": [], "rb": [], "ramsey": []}
    for row in rows:
        _require(isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"] not in identifiers,
                 "Rows need unique string circuit IDs")
        identifiers.add(row["id"])
        _counts(row)
        group = row.get("group")
        if group in ("local", "cost"):
            groups[group].append(row)
        elif group == "gate" and row.get("kind") in ("rb", "ramsey"):
            groups[row["kind"]].append(row)
        else:
            raise ValueError("Unknown diagnostic group/kind")
    return {"status": "analyzed_provided_rows", "circuits": len(rows), "shots": sum(r["shots"] for r in rows),
            "group_counts": {k: len(v) for k, v in groups.items()},
            "local": _local(groups["local"]) if groups["local"] else {},
            "cost": _cost(groups["cost"]) if groups["cost"] else {},
            "rb": _rb(groups["rb"], bootstrap_replicates, bootstrap_seed) if groups["rb"] else {},
            "ramsey": _ramsey(groups["ramsey"]) if groups["ramsey"] else {},
            "sources": SOURCES,
            "limitations": ["Raw binary counts; no mitigation, leakage measurement, appliance MAE or quantum-advantage claim",
                            "Binomial Wilson intervals are within-row iid summaries and exclude calibration/time dependence",
                            "Analysis of provided rows is not a campaign-completeness certificate; compare IDs and shots against the frozen plan",
                            "Calibration associations, control contrasts and fitted decay parameters do not prove a unique physical cause"]}
