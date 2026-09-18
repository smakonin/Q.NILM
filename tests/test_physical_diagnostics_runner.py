"""Offline guard and frozen-template tests. No IBM account calls."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives.containers import SamplerPub

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_physical_diagnostics as runner


class PhysicalRunnerTests(unittest.TestCase):
    def test_authorization_required_before_account_access(self):
        with patch.object(runner, 'account') as account, self.assertRaises(ValueError):
            runner.submit(Path('/unused'), False)
        account.assert_not_called()

    def test_existing_intent_stops_before_account_access(self):
        with TemporaryDirectory() as temp, patch.object(runner, 'check', return_value=({}, [])), \
                patch.object(runner, 'live_snapshot') as live:
            folder = Path(temp)
            runner.save(folder / 'intent.json', {'state': 'unknown'})
            with self.assertRaises(ValueError):
                runner.submit(folder, True)
            live.assert_not_called()

    def test_stop_file_prevents_submission(self):
        with TemporaryDirectory() as temp, patch.object(runner, 'check', return_value=({}, [])), \
                patch.object(runner, 'live_snapshot') as live:
            folder = Path(temp)
            (folder / 'STOP').touch()
            with self.assertRaises(ValueError):
                runner.submit(folder, True)
            live.assert_not_called()

    def test_insufficient_allowance_never_constructs_sampler(self):
        with TemporaryDirectory() as temp, patch.object(runner, 'check', return_value=({}, [])), \
                patch.object(runner, 'live_snapshot', return_value=(Mock(), Mock(),
                    {'available_free_s': 142, 'required_free_s': 143})), \
                patch.object(runner, 'SamplerV2') as sampler:
            folder = Path(temp)
            runner.save(folder / 'plan.json', {})
            runner.save(folder / 'ideal_audit.json', {'status': 'passed', 'circuits': runner.EXPECTED_CIRCUITS,
                                                     'plan_sha256': runner.sha(folder / 'plan.json')})
            with self.assertRaises(ValueError):
                runner.submit(folder, True)
            sampler.assert_not_called()
            self.assertFalse((folder / 'intent.json').exists())

    def test_save_never_overwrites(self):
        with TemporaryDirectory() as temp:
            file = Path(temp) / 'receipt.json'
            runner.save(file, {'original': True})
            with self.assertRaises(FileExistsError):
                runner.save(file, {'original': False})
            self.assertEqual(runner.read(file), {'original': True})

    def test_measurement_map_is_explicit_and_rejects_repeated_output(self):
        circuit = QuantumCircuit(4, 2)
        circuit.measure(3, 0)
        circuit.measure(1, 1)
        self.assertEqual(runner.measurement_map(circuit), [3, 1])
        circuit.measure(0, 0)
        with self.assertRaises(ValueError):
            runner.measurement_map(circuit)

    def test_cost_binding_does_not_zero_fixed_rotations(self):
        circuit = QuantumCircuit(2, 2)
        circuit.rz(.125, 0)
        circuit.rzz(Parameter('cost_angle[0]'), 0, 1)
        circuit.measure(range(2), range(2))
        zero = runner.bound_circuit(circuit, {'parameter_values': [0.]})
        self.assertEqual(float(zero.data[0].operation.params[0]), .125)
        self.assertEqual(float(zero.data[1].operation.params[0]), 0.)
        self.assertEqual(dict(zero.count_ops()), dict(circuit.count_ops()))
        with self.assertRaises(ValueError):
            runner.bound_circuit(circuit, {'parameter_values': [np.nan]})

    def test_frozen_cost_pairs_retain_identical_templates_and_maps(self):
        entries = runner.build_cost_controls()
        self.assertEqual(len(entries), 12)
        pairs = {}
        for circuit, spec in entries:
            self.assertEqual(spec['shots'], 1024)
            SamplerPub.coerce((circuit, spec['parameter_values'], spec['shots']))
            pairs.setdefault((spec['width'], spec['example_id']), []).append((circuit, spec))
        self.assertEqual(len(pairs), 6)
        for zero, nominal in pairs.values():
            self.assertEqual(zero[0], nominal[0])
            self.assertTrue(np.all(np.asarray(zero[1]['parameter_values']) == 0))
            self.assertGreater(np.max(np.abs(nominal[1]['parameter_values'])), 0)
            self.assertEqual(runner.measurement_map(zero[0]), runner.measurement_map(nominal[0]))

    def test_per_pub_shots_supported_for_unparameterized_circuits(self):
        circuit = QuantumCircuit(2)
        circuit.measure_all()
        pub = SamplerPub.coerce((circuit, [], 256))
        self.assertEqual(pub.shots, 256)

    def test_compaction_keeps_classical_bit_order(self):
        circuit = QuantumCircuit(10, 2)
        circuit.x(9)
        circuit.measure(9, 0)
        circuit.measure(3, 1)
        small = runner.compact_measured(circuit)
        self.assertEqual(small.num_qubits, 2)
        self.assertEqual(runner.measurement_map(small), [1, 0])


if __name__ == '__main__':
    unittest.main()
