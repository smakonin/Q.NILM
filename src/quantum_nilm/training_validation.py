"""Additive offline training and static real-data diagnostics.

No provider APIs, production parameter changes, or access to appliance labels
from the optimizer. All calculations are classical, including exact simulation.
"""
from time import perf_counter
from math import prod
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def static_batch(aggregate, levels):
    """One independent interval per row; no temporal term or boundary prior."""
    y = np.asarray(aggregate, float)
    lev = tuple(np.asarray(x, float) for x in levels)
    require(y.ndim == 1 and y.size and np.isfinite(y).all(), "Finite nonempty aggregate")
    require(lev and all(x.ndim == 1 and x.size and np.isfinite(x).all() for x in lev), "Levels")
    sizes = tuple(len(x) for x in lev)
    require(prod(sizes) <= 256, "Small static diagnostic only")
    basis = np.arange(prod(sizes)); stride = 1; states = []
    for n in sizes:
        states.append(basis // stride % n); stride *= n
    states = np.array(states).T
    powers = np.column_stack([x[states[:, i]] for i, x in enumerate(lev)])
    e = (y[:, None] - powers.sum(axis=1))**2
    all_levels = np.concatenate(lev)
    linear_norm = np.abs(all_levels[None, :]**2 - 2*y[:, None]*all_levels).sum(axis=1)
    quadratic_norm = sum(float(np.abs(2*a[:, None]*b).sum()) for i, a in enumerate(lev) for b in lev[i+1:])
    scales = np.maximum(1., linear_norm+quadratic_norm)
    require(np.isfinite(e).all() and np.isfinite(scales).all(), "Finite costs/scales")
    return dict(aggregate=y, levels=lev, sizes=sizes, states=states, powers=powers,
                energies=e, scales=scales, normalized=e/scales[:, None])


def probabilities(batch, angles):
    """Batch of identical-register p=1 W / cost / ordered chain-XY circuits."""
    a = np.asarray(angles, float)
    require(a.shape == (2,) and np.isfinite(a).all(), "Angles")
    rows, width = batch["normalized"].shape
    v = np.exp(-1j*a[0]*batch["normalized"])/np.sqrt(width)
    c, s = np.cos(a[1]), -1j*np.sin(a[1]); stride = 1
    for size in batch["sizes"]:
        blocks = v.reshape(rows, -1, size, stride)
        for j in range(size-1):
            left, right = blocks[:, :, j, :].copy(), blocks[:, :, j+1, :].copy()
            blocks[:, :, j, :] = c*left+s*right
            blocks[:, :, j+1, :] = s*left+c*right
        stride *= size
    p = np.abs(v)**2; total = p.sum(axis=1)
    require(np.max(np.abs(total-1)) < 1e-9, "Normalization")
    return p/total[:, None]


def sample_indices(p, rng, shots):
    require(isinstance(shots, int) and shots > 0, "Positive shots")
    require(p.ndim == 2 and np.isfinite(p).all() and p.min() >= 0 and np.max(abs(p.sum(axis=1)-1)) < 1e-9, "Sampling distribution")
    u = rng.random((len(p), shots)); cdf = np.cumsum(p, axis=1); cdf[:, -1] = 1.
    return np.array([np.searchsorted(c, z, side="right") for c, z in zip(cdf, u)], dtype=np.uint8)


def sample_loss(batch, raw, loss):
    """Equal weight per training case; normalize each case before averaging."""
    raw = np.asarray(raw)
    require(loss in ("mean", "cvar50"), "Declared losses only")
    require(raw.ndim == 2 and len(raw) == len(batch["energies"]) and raw.shape[1] > 0
            and raw.dtype.kind in "iu" and raw.min() >= 0 and raw.max() < batch["energies"].shape[1], "Sample indices")
    values = np.take_along_axis(batch["normalized"], raw, axis=1)
    if loss == "cvar50":
        require(raw.shape[1] % 2 == 0, "Even CVaR shots")
        values = np.sort(values, axis=1)[:, :raw.shape[1]//2]
    return float(values.mean())


def train(batch, loss, seeds, gamma_max, *, budget=512, shots=256, shortlist_size=5, recheck_shots=4096):
    """Fixed search budget plus fresh-measurement shortlist selection.

    Returns every sampled index in binary arrays; exact final probabilities are
    never used for fitting or parameter selection. Shortlist uses first distinct
    parameter vectors ranked by observed training loss, ties chronological.
    """
    from scipy.optimize import minimize
    require(loss in ("mean", "cvar50") and len(seeds) == 2 and len(set(seeds)) == 2, "Loss/seeds")
    require(budget >= 34 and gamma_max > 0 and shortlist_size >= 1, "Budget/range")
    lower, upper = np.array([0., 0.]), np.array([gamma_max, np.pi])
    tolerance = 8*np.finfo(float).eps*np.maximum(1., upper)
    allowances = [(budget-32)//2, (budget-32+1)//2]
    started = perf_counter(); records = []; sample_sets = []; trace = []
    for seed in seeds:
        rng = np.random.default_rng(seed); measurement = np.random.default_rng(seed+1000000)
        initial = np.vstack([np.zeros(2), rng.uniform(lower, upper, size=(31, 2))])
        offset = len(trace); raw_rows = []; corrections = 0

        def evaluate(angles):
            nonlocal corrections
            require(len(raw_rows) < budget, "Search budget exceeded")
            a = np.asarray(angles, float)
            require(a.shape == (2,) and np.isfinite(a).all() and np.all(np.maximum(lower-a, a-upper) <= tolerance), "Search bounds")
            if np.any(a < lower) or np.any(a > upper):
                corrections += 1; a = np.clip(a, lower, upper)
            raw = sample_indices(probabilities(batch, a), measurement, shots)
            score = sample_loss(batch, raw, loss)
            raw_rows.append(raw); trace.append({"angles": a.tolist(), "loss": score})
            return score

        scores = [evaluate(a) for a in initial]; starts = []
        for i in np.argsort(scores, kind="stable"):
            if not any(np.array_equal(initial[i], a) for a in starts): starts.append(initial[i])
            if len(starts) == 2: break
        refinements = []
        for a, allowance in zip(starts, allowances):
            before = len(raw_rows)
            result = minimize(evaluate, a, method="Powell", bounds=list(zip(lower, upper)),
                              options={"maxfev": allowance, "xtol": 1e-5, "ftol": 1e-8})
            spent = len(raw_rows)-before
            require(spent == result.nfev and spent <= allowance, "Optimizer accounting")
            for _ in range(allowance-spent): evaluate(rng.uniform(lower, upper))
            refinements.append({"start": a.tolist(), "calls": spent, "padding_calls": allowance-spent,
                                "success": bool(result.success), "message": str(result.message)})
        require(len(raw_rows) == budget, "Incomplete restart")
        records.append({"seed": int(seed), "offset": offset, "initial_candidates": initial.tolist(),
                        "evaluations": budget, "refinements": refinements, "roundoff_corrections": corrections})
        sample_sets.extend(raw_rows)
    order = np.argsort([x["loss"] for x in trace], kind="stable")
    shortlist = []; seen = set()
    for i in order:
        key = tuple(trace[i]["angles"])
        if key not in seen: shortlist.append(int(i)); seen.add(key)
        if len(shortlist) == shortlist_size: break
    require(len(shortlist) == shortlist_size, "Not enough distinct candidates")
    measurement = np.random.default_rng(seeds[0]+2000000)
    rechecks, recheck_raw = [], []
    for i in shortlist:
        raw = sample_indices(probabilities(batch, trace[i]["angles"]), measurement, recheck_shots)
        recheck_raw.append(raw)
        rechecks.append({"training_index": i, "angles": trace[i]["angles"], "loss": sample_loss(batch, raw, loss)})
    selected = int(np.argmin([x["loss"] for x in rechecks]))
    n_cases = len(batch["energies"])
    fit = {"loss": loss, "seeds": list(seeds), "gamma_max": float(gamma_max), "training_cases": n_cases,
           "evaluations": len(trace), "shots_per_case_call": shots, "recheck_shots_per_case": recheck_shots,
           "shortlist_size": shortlist_size, "shortlist_indices": shortlist, "trace": trace, "restarts": records,
           "precheck_training_index": int(order[0]), "precheck_angles": trace[order[0]]["angles"],
           "selected_recheck_index": selected, "rechecks": rechecks, "angles": rechecks[selected]["angles"],
           "training_shots": n_cases*shots*len(trace), "recheck_shots": n_cases*recheck_shots*shortlist_size,
           "total_measurements": n_cases*(shots*len(trace)+recheck_shots*shortlist_size),
           "runtime_seconds": perf_counter()-started}
    return fit, {"search": np.array(sample_sets, dtype=np.uint8), "recheck": np.array(recheck_raw, dtype=np.uint8)}


def expected_metrics(batch, true_power, *, angles=None, mode="qaoa", budgets=(16, 256)):
    """Expected best-sample metrics against measured appliance watts.

    Exact and oracle controls have no shot budget. Oracle independently chooses
    nearest appliance centroids using labels; it is NOT an inference algorithm.
    All sampled distributions here are ideal and feasible, with no postselection.
    """
    truth = np.asarray(true_power, float)
    require(truth.ndim == 2 and len(truth) == len(batch["energies"]) and np.isfinite(truth).all()
            and 1 <= truth.shape[1] <= len(batch["sizes"]), "Truth shape")
    require(mode in ("qaoa", "uniform", "exact", "oracle", "low_power"), "Metric mode")
    e = batch["energies"]; powers = batch["powers"][:, :truth.shape[1]]
    absolute = abs(powers[None, :, :] - truth[:, None, :])
    closest = np.column_stack([np.argmin(abs(truth[:, i, None]-lev[None, :]), axis=1)
                              for i, lev in enumerate(batch["levels"][:truth.shape[1]])])
    category = batch["states"][None, :, :truth.shape[1]] == closest[:, None, :]
    floor = np.column_stack([np.min(abs(truth[:, i, None]-lev[None, :]), axis=1)
                            for i, lev in enumerate(batch["levels"][:truth.shape[1]])])
    if mode == "oracle":
        return {"oracle": {"absolute_error": floor, "category_correct": np.ones_like(floor),
                           "best_cost": np.full(len(e), np.nan), "hit_optimum": np.full(len(e), np.nan)}}
    order = np.argsort(e, axis=1, kind="stable"); ordered_e = np.take_along_axis(e, order, axis=1)
    rows = np.arange(len(e))
    if mode in ("exact", "low_power"):
        index = order[:, 0] if mode == "exact" else np.zeros(len(e), dtype=int)
        return {mode: {"absolute_error": absolute[rows, index], "category_correct": category[rows, index].astype(float),
                       "best_cost": e[rows, index], "hit_optimum": (e[rows, index] == ordered_e[:, 0]).astype(float)}}
    p = probabilities(batch, angles) if mode == "qaoa" else np.full_like(e, 1/e.shape[1])
    op = np.take_along_axis(p, order, axis=1)
    after = np.cumsum(op, axis=1); before = after-op
    ordered_absolute = np.take_along_axis(absolute, order[:, :, None], axis=1)
    ordered_category = np.take_along_axis(category, order[:, :, None], axis=1)
    popt = np.sum(p*(e == ordered_e[:, :1]), axis=1)
    results = {}
    for k in budgets:
        win = np.clip(1-before, 0, 1)**k - np.clip(1-after, 0, 1)**k
        require(np.max(abs(win.sum(axis=1)-1)) < 1e-8 and win.min() >= -1e-12, "Decoder accounting")
        results[str(k)] = {"absolute_error": np.einsum("ij,ijk->ik", win, ordered_absolute),
                           "category_correct": np.einsum("ij,ijk->ik", win, ordered_category),
                           "best_cost": np.sum(win*ordered_e, axis=1),
                           "hit_optimum": 1-(1-popt)**k}
    return results


def unused_windows(left, right, exposed, count, *, duration=86400, guard=86400, block=30, origin=0):
    """One centered window per free interval, evenly subsample using time only."""
    require(left < right and count > 0 and duration > 0 and guard >= 0, "Window bounds")
    cuts = sorted((max(left, int(a)-guard), min(right, int(b)+guard)) for a, b in exposed
                  if a-guard < right and b+guard > left)
    merged = []
    for a, b in cuts:
        if merged and a <= merged[-1][1]: merged[-1][1] = max(b, merged[-1][1])
        else: merged.append([a, b])
    gaps = []; cursor = left
    for a, b in merged + [[right, right]]:
        if a-cursor >= duration:
            start = origin+((cursor+(a-cursor-duration)//2-origin)//block)*block
            if start < cursor: start += block
            if start+duration <= a: gaps.append((start, start+duration))
        cursor = max(cursor, b)
    require(len(gaps) >= count, "Insufficient untouched separated windows")
    indices = [len(gaps)//2] if count == 1 else [i*(len(gaps)-1)//(count-1) for i in range(count)]
    return [gaps[i] for i in indices]
