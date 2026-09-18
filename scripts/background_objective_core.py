"""Offline diagnostic models only: no production changes or provider access."""
from math import prod
import numpy as np
from scipy.special import logsumexp


def require(ok, message):
    if not ok:
        raise ValueError(message)


def basis(levels):
    lev = [np.asarray(x, float) for x in levels]
    require(all(x.ndim == 1 and len(x) and np.isfinite(x).all() for x in lev), "Finite levels")
    require(0 < prod(map(len, lev)) <= 4096, "Bounded state space")
    ids = np.arange(prod(map(len, lev))); stride = 1; states = []
    for x in lev:
        states.append(ids // stride % len(x)); stride *= len(x)
    states = np.array(states).T
    return states, np.column_stack([x[states[:, i]] for i, x in enumerate(lev)])


def nearest(values, levels):
    return np.argmin(abs(np.asarray(values)[:, None] - np.asarray(levels)), axis=1)


def fit_frequencies(values, levels):
    """Laplace-smoothed empirical marginal state frequencies, TRAIN input only."""
    return [(np.bincount(nearest(values[:, i], x), minlength=len(x)) + 1) /
            (len(values) + len(x)) for i, x in enumerate(levels)]


def fit_background_mixture(background, centres):
    """Train-assigned heteroscedastic Gaussian mixture; no EM or validation fit."""
    ids = nearest(background, centres)
    counts = np.bincount(ids, minlength=len(centres))
    require(np.all(counts > 0), "Occupied training background clusters")
    sd = np.array([max(1., float(np.std(background[ids == j]))) for j in range(len(centres))])
    return {"weights": ((counts + 1) / (len(background) + len(centres))).tolist(),
            "sd_w": sd.tolist(), "centres_w": list(map(float, centres))}


def infer_static(mains, levels, priors=None, *, weight_w2=0., prior_scope="all"):
    """Exact squared residual plus optional unaries; ties lowest mixed-radix ID."""
    y = np.asarray(mains, float)
    require(y.ndim == 1 and np.isfinite(y).all(), "Finite mains")
    require(weight_w2 >= 0 and np.isfinite(weight_w2), "Nonnegative prior weight")
    require(prior_scope in ("all", "appliances", "background"), "Prior scope")
    states, powers = basis(levels)
    unary = np.zeros(len(states))
    if weight_w2:
        require(priors is not None and len(priors) == len(levels), "Matching priors")
        for i, p in enumerate(priors):
            p = np.asarray(p, float)
            require(len(p) == len(levels[i]) and np.all(p > 0) and np.isclose(p.sum(), 1), "Probability prior")
            if prior_scope == "all" or (prior_scope == "appliances" and i < 3) or (prior_scope == "background" and i == 3):
                unary -= np.log(p[states[:, i]])
    indices = []; costs = []
    for a in range(0, len(y), 1024):
        energy = (y[a:a+1024, None] - powers.sum(axis=1))**2 + weight_w2*unary
        chosen = energy.argmin(axis=1); indices.append(chosen)
        costs.append(energy[np.arange(len(chosen)), chosen])
    index = np.concatenate(indices)
    return powers[index], index, np.concatenate(costs)


def infer_marginal(mains, appliance_levels, mixture, appliance_priors, prior_weight):
    """Enumerate 24 target states after integrating a TRAIN background density.

    This diagnostic likelihood is not asserted to be a quadratic binary cost.
    The fourth output is conditional expected background, not an extra input.
    """
    states, powers = basis(appliance_levels)
    centres, sd, weights = (np.asarray(mixture[k]) for k in ("centres_w", "sd_w", "weights"))
    unary = sum(-np.log(np.asarray(p)[states[:, i]]) for i, p in enumerate(appliance_priors))
    output = []; ids = []; scores = []
    for a in range(0, len(mains), 1024):
        residue = np.asarray(mains)[a:a+1024, None] - powers.sum(axis=1)
        terms = np.log(weights) - np.log(sd) - .5*np.log(2*np.pi) - .5*((residue[:, :, None]-centres)/sd)**2
        nll = -logsumexp(terms, axis=2) + prior_weight*unary
        index = nll.argmin(axis=1); row = np.arange(len(index)); selected = terms[row, index]
        conditional = np.exp(selected - logsumexp(selected, axis=1)[:, None]) @ centres
        output.append(np.column_stack((powers[index], conditional))); ids.append(index); scores.append(nll[row, index])
    return np.concatenate(output), np.concatenate(ids), np.concatenate(scores)


def metrics(values, prediction, levels, active_thresholds):
    truth = values[:, 1:4]; predicted = prediction[:, :3]
    require(truth.shape == predicted.shape and np.isfinite(prediction).all(), "Matching finite predictions")
    error = abs(predicted-truth)
    tstate = np.column_stack([nearest(truth[:, i], x) for i, x in enumerate(levels[:3])])
    pstate = np.column_stack([nearest(predicted[:, i], x) for i, x in enumerate(levels[:3])])
    true_active = truth > active_thresholds; pred_active = predicted > active_thresholds
    n = len(values)
    result = {"blocks": n, "absolute_error_sum_w": error.sum(axis=0).tolist(),
              "macro_mae_w": float(error.mean()), "per_appliance_mae_w": error.mean(axis=0).tolist(),
              "category_correct": (tstate == pstate).sum(axis=0).tolist(),
              "all_categories_correct": int(np.all(tstate == pstate, axis=1).sum()),
              "active_blocks": true_active.sum(axis=0).tolist(),
              "predicted_active_blocks": pred_active.sum(axis=0).tolist(),
              "active_absolute_error_sum_w": (error*true_active).sum(axis=0).tolist(),
              "inactive_absolute_error_sum_w": (error*~true_active).sum(axis=0).tolist(),
              "tp": (true_active & pred_active).sum(axis=0).tolist(),
              "fp": (~true_active & pred_active).sum(axis=0).tolist(),
              "fn": (true_active & ~pred_active).sum(axis=0).tolist(),
              "confusion": [np.bincount(tstate[:, i]*len(x)+pstate[:, i], minlength=len(x)**2).reshape(len(x), len(x)).tolist()
                            for i, x in enumerate(levels[:3])]}
    if prediction.shape[1] == 4:
        residual = values[:, 0]-prediction.sum(axis=1)
        result.update(aggregate_squared_error_sum_w2=float(residual@residual),
                      background_absolute_error_sum_w=float(abs(prediction[:, 3]-(values[:, 0]-truth.sum(axis=1))).sum()))
    return result


def pool(rows):
    count = sum(r["blocks"] for r in rows)
    result = {"blocks": count}
    keys = ("absolute_error_sum_w", "category_correct", "all_categories_correct", "active_blocks",
            "predicted_active_blocks", "active_absolute_error_sum_w", "inactive_absolute_error_sum_w", "tp", "fp", "fn")
    for key in keys:
        result[key] = np.sum([r[key] for r in rows], axis=0).tolist()
    result["per_appliance_mae_w"] = (np.array(result["absolute_error_sum_w"])/count).tolist()
    result["macro_mae_w"] = float(np.mean(result["per_appliance_mae_w"]))
    for key in ("aggregate_squared_error_sum_w2", "background_absolute_error_sum_w"):
        if key in rows[0]: result[key] = float(sum(r[key] for r in rows))
    if "aggregate_squared_error_sum_w2" in result:
        result["aggregate_rmse_w"] = float(np.sqrt(result["aggregate_squared_error_sum_w2"]/count))
        result["background_mae_w"] = result["background_absolute_error_sum_w"]/count
    return result


def quantiles(x):
    return dict(zip(("min", "p05", "median", "p95", "p99", "max"), map(float, np.quantile(x, [0, .05, .5, .95, .99, 1]))))


def ambiguity(values, levels):
    """Ground-truth rank and near-optimal target alternatives, for diagnosis only."""
    state, power = basis(levels[:3]); background = np.asarray(levels[3]); y = values[:, 0]
    labels = np.column_stack([nearest(values[:, i+1], x) for i, x in enumerate(levels[:3])])
    stride = np.cumprod([1]+list(map(len, levels[:2])))
    target_id = labels @ stride
    all_cost = np.min((y[:, None, None]-power.sum(axis=1)[None, :, None]-background)**2, axis=2)
    ground = all_cost[np.arange(len(y)), target_id]
    best = all_cost.min(axis=1)
    best_residual = np.sqrt(best)
    near = (np.sqrt(all_cost) <= best_residual[:, None]+10).sum(axis=1)
    rank = 1+(all_cost < ground[:, None]-1e-8).sum(axis=1)
    return {"blocks": len(y), "oracle_target_is_cost_minimizer": int(np.isclose(ground, best, rtol=0, atol=1e-8).sum()),
            "oracle_target_rank_quantiles": quantiles(rank), "targets_within_10w_of_best_residual": quantiles(near),
            "multiple_targets_within_10w_blocks": int((near > 1).sum()),
            "free_nonnegative_background_feasible_targets": quantiles((y[:, None] >= power.sum(axis=1)).sum(axis=1)),
            "ground_target_mean_best_background_cost_w2": float(ground.mean()), "optimum_mean_cost_w2": float(best.mean())}
