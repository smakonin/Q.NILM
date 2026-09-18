#!/usr/bin/env python3
"""Independent offline raw-byte/metric audit for physical diagnostic campaign.

Does not import the production runner, count decoder, or analysis functions.
No account APIs or hardware submission capabilities.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from qiskit_ibm_runtime import RuntimeDecoder
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, tolerance=1e-10):
    require(abs(a - b) <= tolerance, f'Numerical mismatch: {a} != {b}')


def audit(folder, output):
    require(not output.exists(), 'Never overwrite an audit receipt')
    plan, specs = read(folder / 'plan.json'), read(folder / 'pubs.json')
    result, analysis = read(folder / 'result.json'), read(folder / 'analysis.json')
    metrics, job, intent = [read(folder / n) for n in ('metrics.json', 'job.json', 'intent.json')]
    require(plan['circuits'] == 228 and plan['total_shots'] == 94208, 'Unexpected campaign size')
    require(result['job_id'] == analysis['job_id'] == job['job_id'], 'Job identities differ')
    require(job['intent_sha256'] == sha(folder / 'intent.json'), 'Intent binding differs')
    require(job['plan_sha256'] == result['plan_sha256'] == intent['plan_sha256'] == sha(folder / 'plan.json'), 'Plan binding differs')
    require(result['runtime_sha256'] == sha(folder / 'runtime.json.gz') and result['metrics_sha256'] == sha(folder / 'metrics.json'), 'Result provenance differs')
    require(analysis['result_sha256'] == sha(folder / 'result.json'), 'Analysis provenance differs')
    require(intent['pubs_sha256'] == sha(folder / 'pubs.json') and intent['ideal_audit_sha256'] == sha(folder / 'ideal_audit.json'), 'Submission audit or PUB binding differs')
    require(intent['snapshot_sha256'] == sha(folder / 'submission_snapshot.json'), 'Submission snapshot changed')
    require(metrics['usage']['status'] in ('complete', 'completed'), 'Accounting not finalized')
    close(metrics['usage']['qpu_charge_time_seconds'], result['accounted_s'])
    require(result['accounted_s'] <= plan['campaign_cap_s'] == 55, 'Campaign cap exceeded')
    require(not result['raw_counts_postselected'], 'Counts were postselected')
    observed_hashes = {}
    for group, base in (('source_sha256', ROOT), ('files_sha256', folder)):
        for name, expected in plan[group].items():
            path = (base / name).resolve()
            require(path.is_relative_to(base.resolve()) and sha(path) == expected, 'Frozen file changed')
            observed_hashes[str(path.relative_to(ROOT))] = expected
    with gzip.open(folder / 'runtime.json.gz', 'rt') as handle:
        runtime = json.load(handle, cls=RuntimeDecoder)
    require(len(runtime) == len(specs) == len(result['rows']) == 228, 'Missing PUBs')
    summary = analysis['summary']
    local = {r['id']: r for placement in summary['local'].values()
             for kind in ('basis', 'reset', 'w') for r in placement[kind]}
    costs = {r['id']: r for r in summary['cost']['conditions']}
    phase = {r['id']: r for r in summary['ramsey']['conditions']}
    rb = {r['id']: r for edge in summary['rb']['edges'].values() for r in edge['points']}
    require(len(local) + len(costs) + len(phase) + len(rb) == 228, 'Missing analyzed circuits')
    group_counts, total_shots = Counter(), 0
    for spec, row, returned in zip(specs, result['rows'], runtime):
        require(all(row[k] == v for k, v in spec.items()), 'Saved result changed a planned specification')
        bits, shots = spec['n_clbits'], spec['shots']
        data = returned.data.meas
        require(set(returned.data.keys()) == {'meas'} and data.num_bits == bits and data.num_shots == shots, 'Measurement register mismatch')
        raw = np.asarray(data.array)
        require(raw.dtype == np.uint8 and raw.shape == (shots, (bits + 7) // 8), 'Packed byte shape or type mismatch')
        # Independent unpackbits route; production uses int.from_bytes.
        binary = np.unpackbits(raw, axis=1, bitorder='big')
        require(not binary[:, :binary.shape[1] - bits].any(), 'Nonzero leading padding')
        binary = binary[:, -bits:]
        counts = dict(Counter(''.join(map(str, sample)) for sample in binary.tolist()))
        require(counts == row['raw_counts'], 'Packed bits and stored raw counts differ')
        ones = binary[:, ::-1].sum(axis=0)
        total_shots += shots
        kind = row['group'] if row['group'] != 'gate' else row['kind']
        group_counts[kind] += 1
        if kind == 'local':
            saved = local[row['id']]
            for actual, entry in zip(ones, saved['bit_one']):
                require(int(actual) == entry['count'] and entry['shots'] == shots, 'Local marginal count mismatch')
            if row['kind'] == 'w':
                require(int((binary.sum(axis=1) == 1).sum()) == saved['onehot_feasibility']['count'], 'W feasibility differs')
            else:
                expected = row['input_state'] if row['kind'] == 'basis' else '000'
                require(shots - counts.get(expected, 0) == saved['exact_pattern_error']['count'], 'Basis/reset error differs')
        elif kind == 'cost':
            saved, offset = costs[row['id']], 0
            valid = np.ones(shots, dtype=bool)
            low_first = binary[:, ::-1]
            for register in saved['registers']:
                size = len(register['classical_bits'])
                weight = low_first[:, offset:offset + size].sum(axis=1)
                histogram = np.bincount(weight.astype(int), minlength=size + 1).tolist()
                require(histogram == register['hamming_weight_counts'], 'Register weight histogram differs')
                valid &= weight == 1
                offset += size
            require(int(valid.sum()) == saved['all_register_onehot']['count'], 'Cost feasibility differs')
        elif kind == 'ramsey':
            saved = phase[row['id']]
            close(1 - 2 * ones[1] / shots, saved['target_pauli_expectation_all_shots'])
            expected_control = row['control']
            errors = ones[0] if expected_control == 0 else shots - ones[0]
            require(int(errors) == saved['control_flip']['count'], 'Ramsey control error differs')
        else:
            require(counts.get('00', 0) == rb[row['id']]['survival']['count'], 'RB survival differs')
    require(total_shots == summary['shots'] == result['total_shots'] == 94208, 'Total shot count differs')
    require(dict(group_counts) == summary['group_counts'] == {'local': 24, 'cost': 12, 'rb': 160, 'ramsey': 32}, 'Group coverage differs')
    fit_checks = []
    for label, edge in summary['rb']['edges'].items():
        lengths = np.asarray(edge['lengths'], dtype=float)
        for arm, fitted in edge['fits'].items():
            points = [p for p in edge['points'] if p['arm'] == arm]
            observed = np.array([sum(p['survival']['count'] for p in points if p['length'] == length)
                                 / sum(p['survival']['shots'] for p in points if p['length'] == length)
                                 for length in lengths])
            a, alpha, b = [fitted[k] for k in ('A', 'alpha', 'B')]
            predicted = a * alpha ** lengths + b
            require(np.allclose(predicted, fitted['predicted_survival'], atol=1e-10, rtol=0), 'RB prediction differs')
            score = float(np.mean((predicted - observed) ** 2))
            close(score, fitted['weighted_mean_squared_residual'])
            # Direct constrained three-parameter optimization, independent of
            # production's profiled closed-form A/B scalar-alpha search.
            candidates = []
            for start_alpha in (.5, .9, .98, .995, .9999, alpha):
                optimum = minimize(lambda p: np.mean((p[0] * p[1] ** lengths + p[2] - observed) ** 2),
                    [a, start_alpha, b], method='SLSQP', bounds=[(0, 1)] * 3,
                    constraints=[{'type': 'ineq', 'fun': lambda p: 1 - p[0] - p[2]}],
                    options={'maxiter': 1000, 'ftol': 1e-13})
                if optimum.success:
                    candidates.append(float(optimum.fun))
            require(candidates and score <= min(candidates) + 1e-7, 'Independent optimization found a materially better RB fit')
            fit_checks.append({'edge': label, 'arm': arm, 'production_mse': score,
                               'best_independent_mse': min(candidates)})
        error = edge['gate_error_estimate_unclipped']
        if error is not None:
            close(error, .75 * (1 - edge['fits']['interleaved']['alpha'] / edge['fits']['reference']['alpha']))
        draws = [v for v in edge['bootstrap_error_estimates'] if v is not None]
        require(len(draws) == edge['bootstrap_valid'], 'Bootstrap valid count differs')
        if draws:
            require(np.allclose(np.quantile(draws, [.025, .975]), edge['seed_cluster_bootstrap_95']), 'Bootstrap quantiles differ')
    for name, expected in observed_hashes.items():
        require(sha(ROOT / name) == expected, 'Frozen source changed during audit')
    payload = {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'job_id': result['job_id'], 'circuits': 228, 'raw_shots': total_shots,
        'group_counts': dict(group_counts), 'accounted_s': result['accounted_s'],
        'fit_checks': fit_checks, 'audit_code_sha256': sha(Path(__file__)),
        'plan_sha256': sha(folder / 'plan.json'), 'analysis_sha256': sha(folder / 'analysis.json'),
        'scope': 'All packed measurements, primary local/cost/Ramsey/RB metrics, fit arithmetic and independent fit check. Bootstrap quantiles checked; resamples not independently refitted.'}
    with output.open('x') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(payload, allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.folder.resolve(), args.output.resolve())
