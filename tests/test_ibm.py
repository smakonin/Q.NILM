"""Local circuit and decoding tests; never authenticate or submit IBM jobs."""

import importlib.util
import json
import unittest

import numpy as np

from quantum_nilm.ibm import build_qaoa_circuit, counts_to_metrics
from quantum_nilm.qaoa import qaoa_probabilities
from quantum_nilm.qubo import BinaryTemporalQUBO, build_binary_temporal_qubo


@unittest.skipUnless(importlib.util.find_spec("qiskit"), "Qiskit optional dependency absent")
class IBMCircuitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.qubo = build_binary_temporal_qubo(
            np.array([105.0, 390.0, 300.0]),
            np.array([100.0, 300.0]),
            np.array([73.0, 211.0]),
            segment_weights=np.array([2.0, 1.0, 4.0]),
        )

    def test_qiskit_matches_independent_numpy_statevector_p1_and_p2(self) -> None:
        from qiskit.quantum_info import Statevector

        for gammas, betas, scale in (
            ([0.7], [0.3], None),
            ([0.7, -1.13], [0.3, 0.89], 817263.0),
        ):
            with self.subTest(depth=len(gammas)):
                circuit = build_qaoa_circuit(self.qubo, gammas, betas, scale, measure=False)
                actual = Statevector.from_instruction(circuit).probabilities()
                expected = qaoa_probabilities(self.qubo, gammas, betas, scale)
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)

    def test_measurements_preserve_variable_indices_in_meas_register(self) -> None:
        circuit = build_qaoa_circuit(self.qubo, [0.7], [0.3])
        self.assertEqual([register.name for register in circuit.cregs], ["meas"])
        measurements = [instruction for instruction in circuit.data if instruction.operation.name == "measure"]
        self.assertEqual(len(measurements), self.qubo.n_variables)
        for variable, instruction in enumerate(measurements):
            self.assertEqual(circuit.find_bit(instruction.qubits[0]).index, variable)
            self.assertEqual(circuit.find_bit(instruction.clbits[0]).index, variable)

    def test_invalid_angles_and_scale_are_rejected(self) -> None:
        for gammas, betas, scale in (
            ([], [], None),
            ([0.1], [0.2, 0.3], None),
            ([float("nan")], [0.1], None),
            ([0.1], [float("inf")], None),
            ([0.1], [0.2], 0.0),
            ([0.1], [0.2], -1.0),
            ([0.1], [0.2], float("nan")),
            ([0.1], [0.2], float("inf")),
        ):
            with self.subTest(gammas=gammas, betas=betas, scale=scale):
                with self.assertRaises(ValueError):
                    build_qaoa_circuit(self.qubo, gammas, betas, scale)


class IBMCountsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.qubo = build_binary_temporal_qubo(
            np.array([100.0, 300.0]), np.array([100.0, 300.0]), 0.0
        )

    def test_little_endian_decoding_objective_reference_and_weighted_mae(self) -> None:
        # Display '0001' = x[t=0, appliance=0] on, all other states off.
        # Its direct objective is 90000.  '0000' has direct objective 100000.
        summary = counts_to_metrics(
            self.qubo,
            {"0000": 3, "0001": 1},
            reference_states=np.array([[1, 0], [0, 1]]),
            aggregate=np.array([100.0, 300.0]),
            powers=np.array([100.0, 300.0]),
            segment_weights=np.array([4.0, 1.0]),
        )
        self.assertEqual(summary["shots"], 4)
        self.assertEqual(summary["best_bitstring"], "0001")
        self.assertEqual(summary["best_bits"], [1, 0, 0, 0])
        self.assertEqual(summary["best_states"], [[1, 0], [0, 0]])
        self.assertEqual(summary["best_objective"], 90000.0)
        self.assertEqual(summary["mean_objective"], 97500.0)
        self.assertEqual(summary["aggregate_mae_w"], 60.0)
        self.assertEqual(summary["reference_metrics"]["bit_accuracy"], 0.75)
        self.assertAlmostEqual(summary["reference_metrics"]["f1"], 2 / 3)
        self.assertAlmostEqual(summary["reference_metrics"]["mcc"], 1 / np.sqrt(3))
        self.assertEqual(summary["exact_objective"], 0.0)
        self.assertEqual(summary["exact_optimum_probability"], 0.0)
        self.assertFalse(summary["best_is_exact_optimum"])
        json.dumps(summary, allow_nan=False)

    def test_degenerate_optima_all_contribute_to_probability(self) -> None:
        qubo = build_binary_temporal_qubo(np.array([100.0]), np.array([100.0, 100.0]), 0.0)
        summary = counts_to_metrics(qubo, {"10": 4, "01": 3, "00": 1})
        self.assertEqual(summary["exact_optimum_probability"], 7 / 8)
        self.assertEqual(summary["best_bitstring"], "01")
        self.assertEqual(summary["best_sample_count"], 3)

    def test_exact_agreement_and_proxy_accuracy_are_distinct(self) -> None:
        summary = counts_to_metrics(
            self.qubo, {"1001": 2}, reference_states=np.zeros((2, 2), dtype=np.int8)
        )
        self.assertEqual(summary["best_vs_exact_bit_accuracy"], 1.0)
        self.assertEqual(summary["reference_metrics"]["bit_accuracy"], 0.5)
        self.assertEqual(summary["exact_optimum_probability"], 1.0)
        self.assertIsNone(summary["reference_metrics"]["mcc"])

    def test_large_energy_offset_does_not_make_nonoptimal_state_an_optimum(self) -> None:
        # np.isclose's default relative tolerance would incorrectly accept both.
        qubo = BinaryTemporalQUBO(1e10, np.array([1.0]), np.zeros((1, 1)), 1, 1)
        summary = counts_to_metrics(qubo, {"0": 1, "1": 3})
        self.assertEqual(summary["exact_optimum_probability"], 0.25)

    def test_larger_instance_does_not_enumerate(self) -> None:
        from unittest.mock import patch

        qubo = build_binary_temporal_qubo(np.zeros(13), np.array([1.0]), 0.0)
        with patch.object(BinaryTemporalQUBO, "energies", side_effect=AssertionError("enumerated")):
            summary = counts_to_metrics(qubo, {"0" * 13: 2})
        self.assertIsNone(summary["exact_objective"])
        self.assertIsNone(summary["exact_optimum_probability"])
        self.assertEqual(summary["best_objective"], 0.0)

    def test_invalid_counts_are_rejected(self) -> None:
        for counts in (
            {}, {"001": 1}, {"00000": 1}, {"00 1": 1}, {"0x01": 1},
            {"0002": 1}, {1: 1}, {"0000": 0}, {"0000": -1},
            {"0000": 1.0}, {"0000": True}, {"0000": float("nan")},
        ):
            with self.subTest(counts=counts):
                with self.assertRaises(ValueError):
                    counts_to_metrics(self.qubo, counts)

    def test_invalid_reference_and_reconstruction_inputs_are_rejected(self) -> None:
        for kwargs in (
            {"reference_states": [1, 0]},
            {"reference_states": [1, 0, 0, 2]},
            {"reference_states": [1, 0, 0, float("nan")]},
            {"aggregate": [100.0, 300.0]},
            {"powers": [100.0, 300.0]},
            {"segment_weights": [1, 2]},
            {"aggregate": [100.0], "powers": [100.0, 300.0]},
            {"aggregate": [100.0, 300.0], "powers": [100.0, float("inf")]},
            {"aggregate": [100.0, 300.0], "powers": [100.0, 300.0], "segment_weights": [1, 0]},
            {"aggregate": [100.0, 300.0], "powers": [100.0, 300.0], "segment_weights": [1, float("nan")]},
            {"exact_tolerance": -1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    counts_to_metrics(self.qubo, {"0000": 1}, **kwargs)


if __name__ == "__main__":
    unittest.main()
