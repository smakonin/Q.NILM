"""Bounded offline diagnostics tests; no IBM account or quantum job calls."""
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit.primitives.containers import SamplerPub
from quantum_nilm.categorical_ibm import build_categorical_parameterized_template, categorical_template_parameter_values
from quantum_nilm.categorical_qaoa import prepare_categorical_problem, categorical_qaoa_probabilities, encode_onehot

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("diagnostic_ladder", SCRIPTS / "run_quantum_diagnostic_ladder.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DiagnosticLadderTests(unittest.TestCase):
    def test_components_match_ideal_probabilities_and_keep_barriers(self):
        problem = prepare_categorical_problem([73., 91.], [[0., 40.], [5., 80., 130.]], [10., 20.], [2., 3.])
        template = build_categorical_parameterized_template((2, 3), 2, .37)
        indices = [int(encode_onehot(s, problem.state_counts), 2) for s in problem.states]
        for condition in MODULE.CONDITIONS:
            with self.subTest(condition=condition):
                circuit = MODULE.component_circuit(template, condition)
                compiled = transpile(circuit, basis_gates=['rz', 'sx', 'x', 'cz'], optimization_level=3,
                                     approximation_degree=1.0, seed_transpiler=7)
                self.assertEqual(set(map(str, circuit.parameters)), set(map(str, compiled.parameters)))
                self.assertGreaterEqual(compiled.count_ops().get('barrier', 0), 1)
                values = categorical_template_parameter_values(template, problem, .7,
                    parameter_order=tuple(compiled.parameters)) if compiled.num_parameters else []
                bound = compiled.assign_parameters(values)
                SamplerPub.coerce((compiled, values), shots=1024)
                actual = Statevector.from_instruction(bound.remove_final_measurements(inplace=False)).probabilities()
                gamma = .7 if condition == 'full' else 0.
                beta = .37 if condition in ('full', 'W_mixer') else 0.
                expected = categorical_qaoa_probabilities(problem, [gamma], [beta])
                np.testing.assert_allclose(actual[indices], expected, atol=1e-11, rtol=1e-10)
                self.assertAlmostEqual(float(actual[indices].sum()), 1., places=10)

    def test_explicit_submission_flag_required_without_account_access(self):
        with patch.object(MODULE, 'account') as account, self.assertRaises(RuntimeError):
            MODULE.submit(Path('/unused'), False)
        account.assert_not_called()

    def test_prior_unknown_intent_never_resubmits(self):
        with TemporaryDirectory() as d, patch.object(MODULE, 'check', return_value={}), patch.object(MODULE, 'account') as account:
            folder = Path(d)
            (folder / 'intent.json').write_text('{}')
            with self.assertRaises(RuntimeError):
                MODULE.submit(folder, True)
            account.assert_not_called()

    def test_stop_prevents_account_access(self):
        with TemporaryDirectory() as d, patch.object(MODULE, 'check', return_value={}), patch.object(MODULE, 'account') as account:
            folder = Path(d)
            (folder / 'STOP').touch()
            with self.assertRaises(RuntimeError):
                MODULE.submit(folder, True)
            account.assert_not_called()

    def test_free_budget_reserve_checked_before_backend_or_job(self):
        service = Mock()
        with TemporaryDirectory() as d, patch.object(MODULE, 'check', return_value={}), \
             patch.object(MODULE, 'account', return_value=service), patch.object(MODULE, 'remaining', return_value=79.99):
            with self.assertRaises(RuntimeError):
                MODULE.submit(Path(d), True)
            service.backend.assert_not_called()

    def test_save_is_exclusive(self):
        with TemporaryDirectory() as d:
            path = Path(d) / 'test.json'
            MODULE.save(path, {'original': 1})
            with self.assertRaises(FileExistsError):
                MODULE.save(path, {'changed': 1})
            self.assertEqual(json.loads(path.read_text()), {'original': 1})

    def test_prepare_refuses_existing_output_before_work(self):
        with TemporaryDirectory() as d, self.assertRaises(RuntimeError):
            MODULE.prepare(Path(d))


if __name__ == '__main__':
    unittest.main()
