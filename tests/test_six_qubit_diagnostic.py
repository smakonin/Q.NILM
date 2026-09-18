"""Small circuit experiment invariants; no network or account use."""
from datetime import datetime, timezone
from io import BytesIO
import itertools
import json
from pathlib import Path
import unittest

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import Measure, Parameter
from qiskit.circuit.library import CZGate, RZGate, SXGate, XGate
from qiskit.quantum_info import DensityMatrix
from qiskit.transpiler import InstructionProperties, QubitProperties, Target
from qiskit_ibm_runtime.models import BackendProperties

from quantum_nilm import six_qubit_diagnostic as d

ROOT = Path(__file__).resolve().parents[1]


def synthetic_calibration():
    """Portable fixture, not a dependency on private/local experiment archives."""
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    def parameter(name, value, unit=""):
        return {"date": now, "name": name, "value": value, "unit": unit}
    qubits = [[parameter("T1", 100., "us"), parameter("T2", 80., "us"),
               parameter("frequency", 5., "GHz"), parameter("readout_error", .015),
               parameter("prob_meas1_prep0", .01), parameter("prob_meas0_prep1", .02),
               parameter("readout_length", 1000., "ns")] for _ in range(156)]
    target = Target(num_qubits=156, dt=1e-9,
        qubit_properties=[QubitProperties(t1=100e-6, t2=80e-6, frequency=5e9) for _ in range(156)])
    gates = []
    active = sorted(set(sum(d.PLACEMENTS.values(), [])))
    for op, duration, error in ((RZGate(Parameter("a")), 0., 0.), (SXGate(), 40e-9, .0001),
                                (XGate(), 40e-9, .0001), (Measure(), 1e-6, .015)):
        target.add_instruction(op, {(q,): InstructionProperties(duration=duration, error=error) for q in active})
        if op.name != "measure":
            for q in active:
                gates.append({"name": f"{op.name}{q}", "gate": op.name, "qubits": [q],
                              "parameters": [parameter("gate_error", error), parameter("gate_length", duration * 1e9, "ns")]})
    edges = [(q, r) for q in active for r in active if abs(q-r) == 1]
    target.add_instruction(CZGate(), {pair: InstructionProperties(duration=80e-9, error=.003) for pair in edges})
    gates.extend({"name": f"cz{a}_{b}", "gate": "cz", "qubits": [a, b],
                  "parameters": [parameter("gate_error", .003), parameter("gate_length", 80., "ns")]} for a, b in edges)
    props = BackendProperties.from_dict({"backend_name": "fixture", "backend_version": "0",
        "last_update_date": now, "qubits": qubits, "gates": gates, "general": []})
    return target, props


class ArithmeticTests(unittest.TestCase):
    def test_known_six_qubit_problem(self):
        p, truth = d.make_problem()
        self.assertEqual(p.num_qubits, 6)
        self.assertEqual(p.num_feasible_states, 9)
        self.assertEqual(p.aggregate.tolist(), [680.])
        np.testing.assert_equal(p.states[np.argmin(p.energies)], truth)
        self.assertEqual(d.independent_objective_check(p)["max_absolute_error_w2"], 0)

    def test_all_shapes_direct_objective(self):
        for shape in ((2, 1), (3, 1), (2, 2)):
            p, truth = d.make_problem(*shape)
            d.independent_objective_check(p)
            np.testing.assert_equal(p.states[np.argmin(p.energies)], truth)

    def test_probability_mass_guard(self):
        for p in ([.2, .2], [1, 0, 0], [1.1, -.1], [np.nan, 0.]):
            with self.assertRaises(ValueError):
                d.validate_probabilities(p)

    def test_asymmetric_assignment_against_explicit_enumeration(self):
        p = np.array([.1, .2, .3, .4])
        errors = [(.07, .2), (.15, .03)]
        expected = np.zeros(4)
        for source, dest in itertools.product(range(4), repeat=2):
            chance = p[source]
            for q, (p10, p01) in enumerate(errors):
                x, y = (source >> q) & 1, (dest >> q) & 1
                chance *= (p10 if not x else p01) if x != y else (1 - p10 if not x else 1 - p01)
            expected[dest] += chance
        np.testing.assert_allclose(d.apply_readout(p, errors), expected, atol=1e-15)

    def test_onehot_readout_includes_category_exchange(self):
        p, _ = d.make_problem()
        ideal = d.reference_probabilities(p, [0.], [0.], "W")
        rate = .1
        measured = d.apply_readout(ideal, [(rate, rate)] * 6)
        expected = ((1 - rate) ** 3 + 2 * rate ** 2 * (1 - rate)) ** 2
        self.assertAlmostEqual(d.score(p, measured, ideal)["valid_probability"], expected)

    def test_bit_permutation(self):
        p = np.zeros(8); p[1] = 1
        self.assertEqual(np.argmax(d.reorder_probabilities(p, [2, 0, 1])), 2)
        with self.assertRaises(ValueError):
            d.reorder_probabilities(p, [0, 0, 2])

    def test_invalid_shots_not_silently_conditioned(self):
        p, _ = d.make_problem()
        ideal = d.reference_probabilities(p, [0.], [0.], "W")
        all_zero = np.zeros(64); all_zero[0] = 1
        metrics = d.score(p, all_zero, ideal)
        self.assertEqual(metrics["valid_probability"], 0)
        self.assertEqual(metrics["no_valid_sample_probability"], 1)
        self.assertEqual(metrics["optimal_hit_probability"], 0)
        self.assertIsNone(metrics["conditional_mean_objective_w2"])

    def test_projection_one_shot(self):
        p, _ = d.make_problem()
        ideal = d.reference_probabilities(p, [0.], [0.], "W")
        metrics = d.score(p, ideal, ideal, shots=1)
        self.assertAlmostEqual(metrics["optimal_hit_probability"], 1/9)


class CircuitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target, cls.properties = synthetic_calibration()
        cls.problem, _ = d.make_problem()

    def test_components_match_feasible_reference(self):
        for component in ("W", "W_cost", "W_mixer", "full"):
            qc = d.logical_circuit(self.problem, [3.], [.7], component)
            np.testing.assert_allclose(d.ideal_probabilities(qc), d.reference_probabilities(self.problem, [3.], [.7], component), atol=1e-12)

    def test_compilation_and_fold_preserve_probabilities(self):
        logical = d.logical_circuit(self.problem, [3.], [.7])
        ideal = d.ideal_probabilities(logical)
        for placement in ("all_to_all", "suspect", "comparison"):
            counts = []
            for fold in (1, 3, 5):
                qc, _, meta = d.compile_circuit(logical, placement, self.target, fold)
                np.testing.assert_allclose(d.ideal_probabilities(qc), ideal, atol=1e-10)
                counts.append(meta["native_gate_counts"]["cz"])
            self.assertEqual(counts, [counts[0], counts[0]*3, counts[0]*5])

    def test_diagonal_phase_changes_quality_not_feasibility(self):
        for component in ("W_cost", "full"):
            original = d.reference_probabilities(self.problem, [3.], [.7], component)
            phased = d.ideal_probabilities(d.logical_circuit(self.problem, [3.], [.7], component, phase=.3))
            metrics = d.score(self.problem, phased, original)
            self.assertAlmostEqual(metrics["valid_probability"], 1)
            if component == "W_cost":
                self.assertLess(metrics["full_distribution_tv_to_own_ideal"], 1e-12)
            else:
                self.assertGreater(metrics["full_distribution_tv_to_own_ideal"], 1e-5)

    def test_zero_noise_reproduces_exact_distribution(self):
        logical = d.logical_circuit(self.problem, [3.], [.7])
        qc, target, meta = d.compile_circuit(logical, "suspect", self.target)
        noisy, errors, timing = d.noisy_circuit(qc, target, self.properties, meta["physical_qubits"], {"name": "ideal"})
        p, _ = d.simulate(noisy, errors, meta["logical_measurement_to_compact"])
        np.testing.assert_allclose(p, d.ideal_probabilities(qc), atol=1e-11)
        self.assertAlmostEqual(timing["summed_idle_qubit_s"] + timing["summed_active_qubit_s"], 6 * timing["gate_makespan_s"], places=14)

    def test_separate_density_engine_and_qpy_roundtrip(self):
        logical = d.logical_circuit(self.problem, [3.], [.7], "W")
        qc, target, meta = d.compile_circuit(logical, "comparison", self.target)
        model = {"name": "combined", "calibration": "combined", "readout": True}
        noisy, errors, _ = d.noisy_circuit(qc, target, self.properties, meta["physical_qubits"], model)
        buffer = BytesIO(); qpy.dump(noisy, buffer); buffer.seek(0)
        restored = qpy.load(buffer)[0]
        actual, _ = d.simulate(restored, errors, meta["logical_measurement_to_compact"])
        independent = DensityMatrix.from_instruction(restored).probabilities()
        independent = d.reorder_probabilities(d.apply_readout(independent, errors), meta["logical_measurement_to_compact"])
        np.testing.assert_allclose(actual, independent, atol=1e-10)

    def test_noise_channels_lower_feasibility(self):
        logical = d.logical_circuit(self.problem, [3.], [.7], "W")
        qc, target, meta = d.compile_circuit(logical, "suspect", self.target)
        for model in (d.models()[1], d.models()[2], d.models()[3], d.models()[-1]):
            noisy, errors, _ = d.noisy_circuit(qc, target, self.properties, meta["physical_qubits"], model)
            measured, _ = d.simulate(noisy, errors, meta["logical_measurement_to_compact"])
            self.assertLess(d.score(self.problem, measured, d.ideal_probabilities(qc))["valid_probability"], .999)

    def test_width_and_fold_guards(self):
        with self.assertRaises(ValueError):
            d.compile_circuit(QuantumCircuit(13, 13), "all_to_all", self.target)
        with self.assertRaises(ValueError):
            d.compile_circuit(QuantumCircuit(6, 6), "all_to_all", self.target, 2)

    def test_native_coherent_injection_count(self):
        qc, target, meta = d.compile_circuit(d.logical_circuit(self.problem, [3.], [.7]), "suspect", self.target)
        noisy, _, timing = d.noisy_circuit(qc, target, self.properties, meta["physical_qubits"], d.models()[5])
        self.assertEqual(timing["coherent_injections"], qc.count_ops()["cz"])
        self.assertEqual(noisy.count_ops()["rz"], qc.count_ops()["rz"] + qc.count_ops()["cz"])

    def test_only_offline_modes(self):
        source = (ROOT / "scripts/run_six_qubit_diagnostic.py").read_text()
        for forbidden in ("QiskitRuntimeService", "SamplerV2", ".save_account(", "service.job("):
            self.assertNotIn(forbidden, source)

    def test_frozen_task_coverage(self):
        from scripts.run_six_qubit_diagnostic import task_models
        specs = [{"case": case, "placement": placement} for case in d.CASES
                 for _ in (range(4) if case == "six_p1" else range(1))
                 for placement in ("all_to_all", "suspect", "comparison")]
        self.assertEqual(len(specs), 30)
        self.assertEqual(sum(len(task_models(s, d.models())) for s in specs), 106)


if __name__ == "__main__":
    unittest.main()
