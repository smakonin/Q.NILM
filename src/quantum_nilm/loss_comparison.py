"""Additive, offline loss comparison; archived training code is not modified."""
from time import perf_counter
import numpy as np
from .categorical_qaoa import categorical_qaoa_probabilities

LOSSES = ("mean", "cvar25", "cvar50", "best4", "best16")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def probability_check(p, energies):
    p, e = np.asarray(p, float), np.asarray(energies, float)
    require(p.ndim == e.ndim == 1 and p.shape == e.shape and len(p) > 0, "Probability/cost dimensions")
    require(np.isfinite(p).all() and np.isfinite(e).all() and p.min() >= 0 and abs(p.sum()-1) < 1e-9, "Invalid distribution")
    return p, e


def exact_loss(p, energies, loss):
    p, e = probability_check(p, energies)
    require(loss in LOSSES, "Unknown loss")
    if loss == "mean":
        return float(np.dot(p, e))
    order = np.argsort(e, kind="stable")
    p, e = p[order], e[order]
    if loss.startswith("cvar"):
        alpha = {"cvar25": .25, "cvar50": .5}[loss]
        before = np.cumsum(p) - p
        mass = np.minimum(p, np.maximum(0., alpha-before))
        return float(mass @ e / alpha)
    k = {"best4": 4, "best16": 16}[loss]
    return float(e[0] + np.clip(1-np.cumsum(p)[:-1], 0, 1)**k @ np.diff(e))


def sampled_loss(sample_costs, loss):
    values = np.asarray(sample_costs, float)
    require(values.ndim == 1 and len(values) == 256 and np.isfinite(values).all(), "Exactly 256 finite costs required")
    require(loss in LOSSES, "Unknown loss")
    if loss == "mean":
        return float(values.mean())
    if loss.startswith("cvar"):
        count = {"cvar25": 64, "cvar50": 128}[loss]
        return float(np.sort(values)[:count].mean())
    k = {"best4": 4, "best16": 16}[loss]
    return float(values.reshape(-1, k).min(axis=1).mean())


def train(problem, loss, seeds, *, sampled=False, gamma_max=8*np.pi, budget=512):
    """Same 32-initial/two-Powell fixed-budget policy as Stage B, arbitrary loss.

    Measurement randomness has a separate stream so it cannot consume the
    initial/padding-candidate RNG. Equal losses prefer first evaluated vector.
    Exact final evaluation is reporting only, never reranking noisy candidates.
    """
    from scipy.optimize import minimize
    require(loss in LOSSES and len(seeds) == len(set(seeds)) > 0, "Loss/seeds")
    require(isinstance(budget, int) and budget >= 34 and gamma_max > 0, "Budget/bounds")
    lower, upper = np.array([0., 0.]), np.array([gamma_max, np.pi])
    bounds = list(zip(lower, upper))
    tolerance = 8*np.finfo(float).eps*np.maximum(1, np.maximum(abs(lower), abs(upper)))
    allowances = [(budget-32)//2, (budget-32+1)//2]
    started, records, overall = perf_counter(), [], None
    for seed in seeds:
        rng = np.random.default_rng(seed)
        measurement_rng = np.random.default_rng(seed+1000000)
        initial = [np.zeros(2), *rng.uniform(lower, upper, size=(31, 2))]
        calls, trace, best, boundary_corrections = 0, [], None, 0

        def evaluate(angles):
            nonlocal calls, best, boundary_corrections
            require(calls < budget, "Budget exceeded")
            a = np.asarray(angles, float)
            require(a.shape == (2,) and np.isfinite(a).all(), "Angles")
            excursion = np.maximum(lower-a, a-upper)
            require(np.all(excursion <= tolerance), "Angles beyond roundoff tolerance")
            if np.any(excursion > 0):
                boundary_corrections += 1
                a = np.clip(a, lower, upper)
            p = categorical_qaoa_probabilities(problem, [a[0]], [a[1]])
            raw = None
            if sampled:
                u = measurement_rng.random(256)
                cdf = np.cumsum(p); cdf[-1] = 1.
                raw = np.searchsorted(cdf, u, side="right")
                score = sampled_loss(problem.energies[raw], loss) / problem.scale
            else:
                score = exact_loss(p, problem.energies, loss) / problem.scale
            trace.append({"angles": a.tolist(), "loss": score,
                          "sample_indices": None if raw is None else raw.tolist()})
            if best is None or score < best[0]:
                best = (score, a.copy(), p.copy(), calls)
            calls += 1
            return score

        initial_scores = [evaluate(a) for a in initial]
        starts = []
        for i in np.argsort(initial_scores, kind="stable"):
            if not any(np.array_equal(initial[int(i)], a) for a in starts):
                starts.append(initial[int(i)])
            if len(starts) == 2:
                break
        require(len(starts) == 2, "Not enough distinct starts")
        refinements = []
        for a, allowance in zip(starts, allowances):
            before = calls
            result = minimize(evaluate, a, method="Powell", bounds=bounds,
                              options={"maxfev": allowance, "xtol": 1e-5, "ftol": 1e-8})
            spent = calls-before
            require(spent <= allowance and spent == result.nfev, "Powell accounting")
            for _ in range(allowance-spent):
                evaluate(rng.uniform(lower, upper))
            refinements.append({"start": a.tolist(), "calls": spent, "padding_calls": allowance-spent,
                                "success": bool(result.success), "message": str(result.message)})
        require(calls == budget, "Incomplete budget")
        record = {"seed": int(seed), "initial_candidates": np.array(initial).tolist(),
                  "evaluations": calls, "training_shots": calls*256 if sampled else 0,
                  "best_loss": best[0], "angles": best[1].tolist(), "probabilities": best[2].tolist(),
                  "best_evaluation_index": best[3], "refinements": refinements,
                  "boundary_roundoff_corrections": boundary_corrections, "trace": trace}
        records.append(record)
        if overall is None or best[0] < overall[0][0]:
            overall = (best, len(records)-1)
    best, index = overall
    return {"loss": loss, "sampled_training": sampled, "gamma_max": float(gamma_max),
            "seeds": list(seeds), "angles": best[1].tolist(), "probabilities": best[2].tolist(),
            "selected_restart_index": index, "selected_training_loss": best[0],
            "evaluations": sum(r["evaluations"] for r in records),
            "training_shots": sum(r["training_shots"] for r in records),
            "runtime_seconds": perf_counter()-started, "restarts": records}


def decode_metrics(problem, p, true_states, budgets=(4, 16, 256)):
    """Exact raw-probability projections; invalid shots are abstentions, not repair.

    p contains the feasible state probabilities, whose sum may be below one.
    Ties follow the decoder's lower feasible-index rule. Conditional errors are
    accompanied by all-invalid probability; there is no implicit fallback.
    """
    p = np.asarray(p, float)
    e = problem.energies
    require(p.shape == e.shape and np.isfinite(p).all() and p.min() >= 0 and p.sum() <= 1+1e-9, "Feasible probabilities")
    valid = float(np.clip(p.sum(), 0, 1))
    order = np.lexsort((np.arange(len(e)), e))
    mae = np.mean([np.abs(levels[problem.states[:, 0, i]]-levels[true_states[i]]) for i, levels in enumerate(problem.levels)], axis=0)
    optimum = np.isclose(e, min(e), atol=1e-8, rtol=0)
    popt = float(p[optimum].sum())
    data = {"raw_valid_probability": valid, "raw_optimum_probability": popt,
            "conditional_mean_cost_w2": float(p @ e/valid) if valid else None,
            "budgets": {}}
    before = np.cumsum(p[order])-p[order]
    for k in budgets:
        no_valid = max(0., 1-valid)**k
        # Includes invalid mass in both tails, so it cancels except all-invalid.
        win = np.clip(1-before, 0, 1)**k - np.clip(1-before-p[order], 0, 1)**k
        batch_valid = float(win.sum())
        require(abs(batch_valid-(1-no_valid)) < 1e-8, "Best-sample probability accounting")
        data["budgets"][str(k)] = {"no_valid_probability": no_valid,
            "optimum_hit_probability": float(1-max(0., 1-popt)**k),
            "conditional_best_cost_w2": float(win @ e[order]/batch_valid) if batch_valid > 1e-15 else None,
            "conditional_selected_macro_mae_w": float(win @ mae[order]/batch_valid) if batch_valid > 1e-15 else None}
    return data
