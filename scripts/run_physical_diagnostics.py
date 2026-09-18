#!/usr/bin/env python3
"""One bounded, free-only physical diagnostic campaign; no automatic retries.

snapshot/status/collect are read-only IBM operations. prepare/audit/analyze
are offline. Only submit with --allow-free-submission can create a job.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys

import numpy as np
from qiskit import QuantumCircuit, qasm3, qpy
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import RuntimeDecoder, RuntimeEncoder, SamplerV2
from qiskit_ibm_runtime.models import BackendConfiguration, BackendProperties
from qiskit_ibm_runtime.utils.backend_converter import convert_to_target

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from run_quantum_heldout_ibm import account, remaining, circuit_info, accounted_usage
from retry_quantum_diagnostic_ladder import planning_reserve

SOURCE = ROOT / 'results/quantum_diagnostics/ladder_retry_001'
FAILED = ROOT / 'results/quantum_diagnostics/ladder'
CAP, MAX_EXECUTION, RESERVE = 55, 45, 20
EXPECTED_CIRCUITS, EXPECTED_SHOTS = 228, 94208
SOURCE_FILES = [
    'scripts/run_physical_diagnostics.py',
    'src/quantum_nilm/physical_local_controls.py',
    'src/quantum_nilm/physical_gate_benchmarks.py',
    'src/quantum_nilm/physical_diagnostic_analysis.py',
    'scripts/run_quantum_heldout_ibm.py',
    'scripts/retry_quantum_diagnostic_ladder.py',
    'scripts/run_quantum_diagnostic_ladder.py',
    'scripts/audit_quantum_diagnostic_ladder.py',
    'src/quantum_nilm/categorical_qaoa.py',
    'src/quantum_nilm/categorical_ibm.py',
    'tests/test_physical_local_controls.py',
    'tests/test_physical_gate_benchmarks.py',
    'tests/test_physical_diagnostic_analysis.py',
    'tests/test_physical_diagnostics_runner.py',
    'tests/test_physical_diagnostics_guard_audit.py',
    'docs/physical_diagnostics_protocol.md',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), cls=RuntimeDecoder)


def save(path, payload):
    with Path(path).open('x') as handle:
        json.dump(payload, handle, cls=RuntimeEncoder, indent=2, allow_nan=False)
        handle.write('\n')


def target_from(snapshot):
    return convert_to_target(
        configuration=BackendConfiguration.from_dict(snapshot['backend_configuration']),
        properties=BackendProperties.from_dict(snapshot['backend_properties']),
        include_control_flow=False, include_fractional_gates=False)


def live_snapshot():
    service = account()
    allowance = remaining(service)
    old = read(FAILED / 'job.json')
    old_job = service.job(old['job_id'])
    require(str(old_job.status()) == 'ERROR', 'Original failed job changed terminal state')
    metrics = old_job.metrics()
    pending = planning_reserve(metrics)
    backend = service.backend('ibm_fez')
    require(backend.status().operational, 'Selected backend is not operational')
    return service, backend, {
        'retrieved_utc': utc(), 'backend': 'ibm_fez', 'instance_plan': 'open',
        'available_free_s': allowance, 'old_failed_job_id': old['job_id'],
        'old_failed_job_usage': metrics.get('usage'),
        'pending_accounting_reserve_s': pending,
        'required_free_s': CAP + RESERVE + pending,
        'backend_properties': backend.properties().to_dict(),
        'backend_configuration': backend.configuration().to_dict(),
        'note': 'Public calibration snapshot, not instantaneous characterization of each circuit.'}


def snapshot(folder):
    require(not folder.exists(), 'Use a new diagnostic directory')
    _, _, payload = live_snapshot()
    folder.mkdir(parents=True)
    save(folder / 'preparation_snapshot.json', payload)
    print(json.dumps({k: payload[k] for k in ('backend', 'available_free_s',
        'pending_accounting_reserve_s', 'required_free_s')}), flush=True)


def measurement_map(circuit):
    mapping = {}
    for item in circuit.data:
        if item.operation.name == 'measure':
            bit = circuit.find_bit(item.clbits[0]).index
            require(bit not in mapping, 'Repeated output-bit measurement')
            mapping[bit] = circuit.find_bit(item.qubits[0]).index
    require(set(mapping) == set(range(circuit.num_clbits)), 'Incomplete output measurement')
    require(len(set(mapping.values())) == circuit.num_clbits, 'Repeated physical measurement')
    return [mapping[i] for i in range(circuit.num_clbits)]


def build_cost_controls():
    old_plan, pubs = read(SOURCE / 'plan.json'), read(SOURCE / 'pubs.json')
    for filename, expected in old_plan['files_sha256'].items():
        require(Path(filename).name == filename and sha(SOURCE / filename) == expected,
                'Frozen ladder file changed')
    entries = []
    for width in (1, 2):
        key = f'K{width}_W_cost'
        with (SOURCE / f'{key}.qpy').open('rb') as handle:
            circuit = qpy.load(handle)[0]
        require(all(str(p).startswith('cost_angle[') for p in circuit.parameters),
                'Unexpected non-cost symbolic parameter')
        resource = old_plan['resources'][key]
        require(measurement_map(circuit) == resource['logical_measurement_bit_to_physical'],
                'Frozen cost measurement map changed')
        for example in range(3):
            matched = [p for p in pubs if p['template'] == key and p['example_index'] == example]
            require(len(matched) == 1, 'Missing or duplicate frozen example')
            for arm in ('zero', 'nominal'):
                values = (np.zeros(circuit.num_parameters).tolist() if arm == 'zero'
                          else matched[0]['parameter_values'])
                spec = {'id': f'cost_K{width}_example{example}_{arm}',
                    'group': 'cost', 'kind': 'matched_cost', 'width': width,
                    'example_id': example, 'arm': arm, 'state_counts': [4, 3, 2, 3],
                    'shots': 1024, 'n_clbits': 12 * width, 'source_template': key,
                    'parameter_values': values, 'parameter_order': list(map(str, circuit.parameters)),
                    'logical_measurement_bit_to_physical': measurement_map(circuit),
                    'ideal_feasible_fraction': 1., 'ideal_distribution': 'uniform feasible',
                    'compiler': 'Frozen W+cost QPY, no recompilation; only symbolic cost angles bound'}
                entries.append((circuit.copy(), spec))
    return entries


def bound_circuit(circuit, spec):
    values = spec.get('parameter_values', [])
    require(len(values) == circuit.num_parameters, 'Parameter count differs')
    require(np.all(np.isfinite(values)), 'Non-finite parameter value')
    return circuit.assign_parameters(values) if values else circuit.copy()


def prepare(folder):
    from quantum_nilm.physical_local_controls import build_local_controls
    from quantum_nilm.physical_gate_benchmarks import build_gate_benchmarks
    require((folder / 'preparation_snapshot.json').exists() and not (folder / 'plan.json').exists(),
            'Snapshot required; prepared plan cannot be overwritten')
    require(set(p.name for p in folder.iterdir()) == {'preparation_snapshot.json'},
            'Directory contains partial preparation; use a fresh archive')
    snap = read(folder / 'preparation_snapshot.json')
    target = target_from(snap)
    entries = build_local_controls(target) + build_cost_controls() + build_gate_benchmarks(target)
    require(len(entries) == EXPECTED_CIRCUITS, 'Incorrect diagnostic circuit count')
    require(sum(spec['shots'] for _, spec in entries) == EXPECTED_SHOTS, 'Incorrect total shot count')
    require(len({spec['id'] for _, spec in entries}) == len(entries), 'Duplicate diagnostic IDs')
    old_plan = read(SOURCE / 'plan.json')
    delay = old_plan['rep_delay_s']
    # This is a planning model, never reported as measured or guaranteed QPU time.
    estimate = 5. + .03 * len(entries)
    specs = []
    for circuit, original_spec in entries:
        spec = dict(original_spec)
        spec['n_clbits'] = circuit.num_clbits
        require(measurement_map(circuit) == spec['logical_measurement_bit_to_physical'],
                'Generator output measurement mapping mismatch')
        spec['qpy_file'] = spec['id'] + '.qpy'
        spec['qasm_file'] = spec['id'] + '.qasm'
        require(Path(spec['qpy_file']).name == spec['qpy_file'], 'Unsafe circuit ID')
        with (folder / spec['qpy_file']).open('xb') as handle:
            qpy.dump(circuit, handle)
        with (folder / spec['qasm_file']).open('x') as handle:
            handle.write(qasm3.dumps(circuit))
        spec['statistics'] = circuit_info(circuit)
        numeric = bound_circuit(circuit, spec)
        duration = float(numeric.estimate_duration(target, unit='s'))
        require(math.isfinite(duration) and duration >= 0, 'Invalid estimated circuit duration')
        spec['estimated_duration_s'] = duration
        estimate += spec['shots'] * (duration + delay)
        specs.append(spec)
    require(estimate < MAX_EXECUTION, 'Estimated diagnostic use exceeds bounded execution budget')
    order = np.random.default_rng(20260915).permutation(len(specs)).tolist()
    specs = [dict(specs[j], pub_index=i) for i, j in enumerate(order)]
    save(folder / 'pubs.json', specs)
    source_files = [ROOT / p for p in SOURCE_FILES]
    source_files += [SOURCE / name for name in ('plan.json', 'pubs.json', 'examples.json',
        'K1_W_cost.qpy', 'K2_W_cost.qpy', 'independent_ideal_audit.json', 'independent_counts_audit.json')]
    source_files += [ROOT / 'results/quantum_heldout/model.json']
    plan = {'created_utc': utc(), 'backend': 'ibm_fez', 'instance_plan': 'open',
        'paid_allowed': False, 'circuits': len(specs), 'total_shots': EXPECTED_SHOTS,
        'groups': dict(Counter(s['group'] for s in specs)),
        'campaign_cap_s': CAP, 'max_execution_time_s': MAX_EXECUTION,
        'free_reserve_s': RESERVE, 'rep_delay_s': delay, 'estimated_accounted_s': estimate,
        'estimate_method': 'sum(shots*(snapshot-duration+rep-delay))+5s+0.03s/circuit planning overhead; not a guarantee',
        'order': 'seed20260915 fixed random permutation; one job, no independent-date replication',
        'scope': 'three user-authorized physical diagnostics, not a new NILM evaluation',
        'versions': {p: version(p) for p in ('qiskit', 'qiskit-aer', 'qiskit-ibm-runtime', 'numpy', 'scipy')},
        'options': {'max_execution_time': MAX_EXECUTION,
            'dynamical_decoupling': {'enable': False},
            'twirling': {'enable_gates': False, 'enable_measure': False},
            'execution': {'rep_delay': delay, 'init_qubits': True},
            'environment': {'job_tags': ['qnilm', 'physical-diagnostics-001']}},
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in source_files},
        'files_sha256': {p.name: sha(p) for p in folder.iterdir() if p.is_file()},
        'automatic_retry': False, 'raw_counts_postselected': False,
        'limitations': ['SPAM and reset checks do not independently separate every error mechanism',
            'Zero-angle cost controls retain entangling/routing overhead and are blind to pure diagonal phase errors',
            'Gate characterization uses short two-qubit circuits; full-width crosstalk is not replicated',
            'IRB assumes approximately exponential stationary noise; eight seed clusters, diagnostic precision only',
            'Archived calibration and compiled duration estimates are not instantaneous hardware error measurements']}
    save(folder / 'plan.json', plan)
    print(json.dumps({k: plan[k] for k in ('circuits', 'total_shots', 'groups', 'estimated_accounted_s')}), flush=True)


def check(folder):
    plan = read(folder / 'plan.json')
    require(plan['backend'] == 'ibm_fez' and plan['instance_plan'] == 'open'
            and plan['paid_allowed'] is False and plan['campaign_cap_s'] == CAP
            and plan['max_execution_time_s'] == MAX_EXECUTION and plan['free_reserve_s'] == RESERVE
            and plan['circuits'] == EXPECTED_CIRCUITS and plan['total_shots'] == EXPECTED_SHOTS,
            'Campaign scope or budget differs')
    require(plan['options'] == {'max_execution_time': MAX_EXECUTION,
        'dynamical_decoupling': {'enable': False},
        'twirling': {'enable_gates': False, 'enable_measure': False},
        'execution': {'rep_delay': plan['rep_delay_s'], 'init_qubits': True},
        'environment': {'job_tags': ['qnilm', 'physical-diagnostics-001']}}, 'Execution options differ')
    for group, base in (('source_sha256', ROOT), ('files_sha256', folder)):
        for name, expected in plan[group].items():
            path = (base / name).resolve()
            require(path.is_relative_to(base.resolve()) and sha(path) == expected,
                    f'Frozen hash mismatch: {name}')
    require(plan['versions'] == {p: version(p) for p in plan['versions']}, 'Runtime versions changed')
    specs = read(folder / 'pubs.json')
    require(len(specs) == EXPECTED_CIRCUITS and sum(s['shots'] for s in specs) == EXPECTED_SHOTS,
            'Actual PUB or shot count differs')
    require([s['pub_index'] for s in specs] == list(range(len(specs)))
            and len({s['id'] for s in specs}) == len(specs), 'PUB order or IDs differ')
    return plan, specs


def compact_measured(circuit):
    active = sorted({circuit.find_bit(q).index for item in circuit.data
                     if item.operation.name != 'barrier' for q in item.qubits})
    remap = {q: i for i, q in enumerate(active)}
    small = QuantumCircuit(len(active), circuit.num_clbits)
    small.global_phase = circuit.global_phase
    for item in circuit.data:
        if item.operation.name != 'barrier':
            small.append(item.operation, [remap[circuit.find_bit(q).index] for q in item.qubits],
                         [circuit.find_bit(c).index for c in item.clbits])
    return small


def audit(folder):
    from audit_quantum_diagnostic_ladder import compact_circuit, probabilities_from_mps
    from quantum_nilm.categorical_qaoa import prepare_categorical_problem
    require(not (folder / 'ideal_audit.json').exists(), 'Ideal audit receipt already exists')
    plan, specs = check(folder)
    model = read(ROOT / 'results/quantum_heldout/model.json')
    frozen = model['models']['multistate']
    examples = read(SOURCE / 'examples.json')
    levels = [np.asarray(x, float) for x in frozen['levels_w']]
    penalties = .1 * np.asarray(frozen['ranges_w']) ** 2
    rows, signatures = [], {}
    for spec in specs:
        with (folder / spec['qpy_file']).open('rb') as handle:
            circuit = qpy.load(handle)[0]
        require(measurement_map(circuit) == spec['logical_measurement_bit_to_physical'], 'QPY mapping differs')
        bound = bound_circuit(circuit, spec)
        if spec['group'] == 'cost':
            sig = [(i.operation.name, tuple(circuit.find_bit(q).index for q in i.qubits),
                    tuple(circuit.find_bit(c).index for c in i.clbits)) for i in circuit.data]
            pair = (spec['width'], spec['example_id'])
            if pair in signatures:
                require(sig == signatures[pair], 'Matched cost skeletons differ')
            signatures[pair] = sig
            ex, width = examples[spec['example_id']], spec['width']
            problem = prepare_categorical_problem(ex['aggregate'][:width], levels, penalties, ex['weights'][:width])
            compact, _, measured = compact_circuit(bound, 12 * width, spec['logical_measurement_bit_to_physical'])
            probs, norm, _, bond = probabilities_from_mps(compact, problem, measured)
            error = float(np.max(np.abs(probs - 1 / len(probs))))
            require(error < 1e-7 and abs(norm - 1) < 1e-8 and abs(probs.sum() - 1) < 1e-7,
                    'Matched cost ideal probability audit failed')
            row = {'id': spec['id'], 'maximum_probability_error': error,
                   'feasible_mass': float(probs.sum()), 'norm': norm, 'max_mps_bond': bond}
        else:
            small = compact_measured(bound)
            expected = spec.get('expected_ideal_probabilities', spec.get('ideal_probs'))
            require(expected is not None, 'Missing ideal reference distribution')
            state_type = DensityMatrix if spec['kind'] == 'reset' else Statevector
            actual = state_type.from_instruction(small.remove_final_measurements(inplace=False)).probabilities()
            mapped = {}
            output_mapping = measurement_map(small)
            for integer, probability in enumerate(actual):
                out = sum(((integer >> q) & 1) << bit for bit, q in enumerate(output_mapping))
                key = format(out, f"0{spec['n_clbits']}b")
                mapped[key] = mapped.get(key, 0.) + float(probability)
            error = max(abs(mapped.get(k, 0.) - expected.get(k, 0.)) for k in set(mapped) | set(expected))
            require(error < 1e-8, 'Small-circuit ideal probability check failed')
            row = {'id': spec['id'], 'maximum_probability_error': error}
        rows.append(row)
    check(folder)
    payload = {'created_utc': utc(), 'status': 'passed', 'plan_sha256': sha(folder / 'plan.json'),
        'circuits': len(rows), 'rows': rows, 'matched_cost_pairs': len(signatures),
        'maximum_probability_error': max(r['maximum_probability_error'] for r in rows),
        'method': 'Complete ideal output probabilities; MPS contraction includes routing ancillas; density matrix handles explicit resets'}
    save(folder / 'ideal_audit.json', payload)
    print(json.dumps({k: payload[k] for k in ('status', 'circuits', 'maximum_probability_error')}), flush=True)


def submit(folder, allowed):
    require(allowed, 'Explicit free-submission authorization required')
    plan, specs = check(folder)
    require(not (folder / 'STOP').exists() and not (folder / 'intent.json').exists()
            and not (folder / 'submission_snapshot.json').exists(), 'Prior submission state exists; never resubmit')
    ideal = read(folder / 'ideal_audit.json')
    require(ideal['status'] == 'passed' and ideal['circuits'] == EXPECTED_CIRCUITS
            and ideal['plan_sha256'] == sha(folder / 'plan.json'), 'Missing or unrelated ideal audit')
    service, backend, snap = live_snapshot()
    require(snap['available_free_s'] >= snap['required_free_s'], 'Free allowance insufficient including pending accounting')
    pubs = []
    for spec in specs:
        with (folder / spec['qpy_file']).open('rb') as handle:
            circuit = qpy.load(handle)[0]
        for item in circuit.data:
            if item.operation.name == 'barrier':
                continue
            qubits = tuple(circuit.find_bit(q).index for q in item.qubits)
            require(backend.target.instruction_supported(operation_name=item.operation.name, qargs=qubits),
                    'Frozen native instruction not supported by live backend')
        pubs.append((circuit, spec.get('parameter_values', []), spec['shots']))
    save(folder / 'submission_snapshot.json', snap)
    require(not (folder / 'STOP').exists(), 'STOP marker present before submission')
    save(folder / 'intent.json', {'created_utc': utc(), 'plan_sha256': sha(folder / 'plan.json'),
        'pubs_sha256': sha(folder / 'pubs.json'), 'ideal_audit_sha256': sha(folder / 'ideal_audit.json'),
        'snapshot_sha256': sha(folder / 'submission_snapshot.json'), 'available_free_s': snap['available_free_s'],
        'required_free_s': snap['required_free_s'], 'options': plan['options'],
        'state': 'submission started; reconcile remote receipt before any further action', 'automatic_retry': False})
    job = SamplerV2(mode=backend, options=plan['options']).run(pubs)
    save(folder / 'job.json', {'job_id': job.job_id(), 'submitted_utc': utc(),
        'plan_sha256': sha(folder / 'plan.json'), 'intent_sha256': sha(folder / 'intent.json')})
    print(json.dumps({'job_id': job.job_id(), 'circuits': len(pubs), 'shots': EXPECTED_SHOTS,
                      'max_execution_time_s': MAX_EXECUTION, 'campaign_planning_cap_s': CAP}), flush=True)


def status(folder):
    receipt = read(folder / 'job.json')
    service = account()
    job = service.job(receipt['job_id'])
    state = str(job.status())
    payload = {'job_id': receipt['job_id'], 'status': state}
    if state in ('DONE', 'ERROR', 'CANCELLED'):
        metrics = job.metrics()
        payload['usage'] = metrics.get('usage')
        if state != 'DONE' and not (folder / 'failure_status.json').exists():
            save(folder / 'failure_status.json', {'checked_utc': utc(), **payload,
                'timestamps': metrics.get('timestamps'), 'note': 'No automatic retry; no inferred scientific result'})
    print(json.dumps(payload, cls=RuntimeEncoder), flush=True)


def collect(folder):
    plan, specs = check(folder)
    receipt, intent = read(folder / 'job.json'), read(folder / 'intent.json')
    require(receipt['plan_sha256'] == intent['plan_sha256'] == sha(folder / 'plan.json')
            and receipt['intent_sha256'] == sha(folder / 'intent.json'), 'Submission binding differs')
    service = account()
    job = service.job(receipt['job_id'])
    require(str(job.status()) == 'DONE', 'Recorded job is not completed; status only, never retry')
    path = folder / 'runtime.json.gz'
    if not path.exists():
        result = job.result()
        temporary = folder / 'runtime.json.gz.pending'
        with temporary.open('xb') as raw, gzip.open(raw, 'wt') as handle:
            json.dump(result, handle, cls=RuntimeEncoder, allow_nan=False)
        temporary.rename(path)
    metrics = job.metrics()
    charge = accounted_usage(metrics)
    require((metrics.get('usage') or {}).get('status') in ('complete', 'completed') and charge is not None,
            'Usage is not finalized; raw result saved, collect accounting later')
    if not (folder / 'metrics.json').exists():
        save(folder / 'metrics.json', metrics)
    else:
        saved_metrics = read(folder / 'metrics.json')
        require((saved_metrics.get('usage') or {}).get('status') in ('complete', 'completed')
                and accounted_usage(saved_metrics) == charge, 'Stored and current finalized accounting conflict')
        metrics = saved_metrics
    require(charge <= CAP, 'Accounted use exceeds planning cap; stop and report')
    with gzip.open(path, 'rt') as handle:
        returned = json.load(handle, cls=RuntimeDecoder)
    require(len(returned) == len(specs), 'Returned PUB count differs')
    rows = []
    for spec, pub in zip(specs, returned):
        counts = decode_pub(pub, spec)
        rows.append({**spec, 'raw_counts': counts})
    require(sum(sum(r['raw_counts'].values()) for r in rows) == EXPECTED_SHOTS, 'Incomplete shot coverage')
    payload = {'job_id': receipt['job_id'], 'plan_sha256': sha(folder / 'plan.json'),
        'runtime_sha256': sha(path), 'metrics_sha256': sha(folder / 'metrics.json'),
        'accounted_s': charge, 'total_shots': EXPECTED_SHOTS, 'rows': rows, 'raw_counts_postselected': False}
    if not (folder / 'result.json').exists():
        save(folder / 'result.json', {'completed_utc': utc(), **payload})
    else:
        saved_result = read(folder / 'result.json')
        require(all(saved_result[k] == value for k, value in payload.items()),
                'Existing result differs from reconstructed raw evidence')
    print(json.dumps({'job_id': receipt['job_id'], 'circuits': len(rows),
                      'raw_shots': EXPECTED_SHOTS, 'accounted_s': charge}), flush=True)


def decode_pub(pub, spec):
    require(set(pub.data.keys()) == {'meas'}, 'Expected exactly one meas register')
    measured = pub.data.meas
    require(measured.num_bits == spec['n_clbits'] and measured.num_shots == spec['shots']
            and measured.array.shape == (spec['shots'], (spec['n_clbits'] + 7) // 8),
            'Returned measurement size differs')
    values = [int.from_bytes(bytes(v), 'big') for v in measured.array]
    require(all(v < (1 << spec['n_clbits']) for v in values), 'Nonzero measurement padding')
    counts = dict(Counter(format(v, f"0{spec['n_clbits']}b") for v in values))
    require(counts == measured.get_counts() and sum(counts.values()) == spec['shots'],
            'Packed bytes and returned histogram disagree')
    return counts


def analyze(folder):
    from quantum_nilm.physical_diagnostic_analysis import analyze_rows
    check(folder)
    result = read(folder / 'result.json')
    require(result['plan_sha256'] == sha(folder / 'plan.json')
            and result['runtime_sha256'] == sha(folder / 'runtime.json.gz')
            and result['metrics_sha256'] == sha(folder / 'metrics.json'), 'Result provenance differs')
    summary = analyze_rows(result['rows'])
    save(folder / 'analysis.json', {'created_utc': utc(), 'job_id': result['job_id'],
        'result_sha256': sha(folder / 'result.json'), 'accounted_s': result['accounted_s'],
        'scope': 'one diagnostic job; no NILM MAE or quantum-advantage inference', 'summary': summary})
    print(json.dumps(summary, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('snapshot', 'prepare', 'audit', 'submit', 'status', 'collect', 'analyze'))
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/quantum_diagnostics/physical_001')
    parser.add_argument('--allow-free-submission', action='store_true')
    args = parser.parse_args()
    try:
        folder = args.output_dir.resolve()
        require(folder.is_relative_to(ROOT / 'results/quantum_diagnostics'), 'Archive must stay in the diagnostic results directory')
        if args.mode == 'submit':
            submit(folder, args.allow_free_submission)
        else:
            globals()[args.mode](folder)
    except Exception as error:
        # Network/authentication exceptions can contain credentials or account identifiers.
        print(json.dumps({'status': 'stopped safely', 'mode': args.mode, 'error_type': type(error).__name__}), flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
