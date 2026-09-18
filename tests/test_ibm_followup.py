"""Offline control audits, including physical noise and classical-bit remapping."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("qiskit_aer"), "Optional Aer dependency absent")
class IBMFollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts" / "run_ibm_followup_controls.py"
        spec = importlib.util.spec_from_file_location("ibm_followup", path)
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_equal_angle_split_is_not_optimized_depth(self) -> None:
        self.assertEqual(self.runner.depth_angles(4.0, 2.0, 1), ([4.0], [2.0]))
        self.assertEqual(self.runner.depth_angles(4.0, 2.0, 2), ([2.0, 2.0], [1.0, 1.0]))
        for gamma, beta, depth in ((1, 1, 0), (1, 1, 3), (np.nan, 1, 2)):
            with self.assertRaises(ValueError):
                self.runner.depth_angles(gamma, beta, depth)

    def test_compaction_preserves_classical_measurement_order_and_probabilities(self) -> None:
        from qiskit import QuantumCircuit

        circuit = QuantumCircuit(10, 2)
        circuit.h(3)
        circuit.x(7)
        circuit.measure(7, 0)
        circuit.measure(3, 1)
        compact, physical = self.runner.compact_circuit(circuit)
        self.assertEqual(physical, [3, 7])
        self.assertEqual(compact.num_qubits, 2)
        actual = self.runner.classical_order_probabilities(compact)
        np.testing.assert_allclose(actual, [0, 0.5, 0, 0.5], atol=1e-14)
        np.testing.assert_allclose(actual, self.runner.classical_order_probabilities(circuit), atol=1e-14)

    def test_remapped_physical_readout_error_still_acts_on_correct_bit(self) -> None:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, ReadoutError

        noise = NoiseModel()
        noise.add_readout_error(ReadoutError([[0, 1], [1, 0]]), [7])
        circuit = QuantumCircuit(10, 2)
        circuit.measure(3, 0)
        circuit.measure(7, 1)
        compact, active = self.runner.compact_circuit(circuit)
        remapped = self.runner.compact_noise_dictionary(noise.to_dict(serializable=True), active)
        simulator = AerSimulator(noise_model=NoiseModel.from_dict(remapped))
        counts = simulator.run(compact, shots=16, seed_simulator=0).result().get_counts()
        self.assertEqual(counts, {"10": 16})

    def test_remapped_noise_retains_gate_order_and_drops_unused_edges(self) -> None:
        original = {"errors": [{"type": "qerror", "gate_qubits": [[7, 3], [7, 8]]},
                               {"type": "roerror", "gate_qubits": [[8]]},
                               {"type": "qerror", "all_qubits": True}]}
        remapped = self.runner.compact_noise_dictionary(original, [3, 7])
        self.assertEqual(remapped["errors"][0]["gate_qubits"], [[1, 0]])
        self.assertEqual(len(remapped["errors"]), 2)
        self.assertEqual(original["errors"][0]["gate_qubits"], [[7, 3], [7, 8]])
        with self.assertRaises(ValueError):
            self.runner.compact_noise_dictionary(original, [3, 3])

    def test_gate_noise_handles_missing_frequency_without_inventing_calibration(self) -> None:
        from datetime import datetime, timezone
        from qiskit_ibm_runtime.models import BackendProperties
        from qiskit_aer.noise import NoiseModel

        date = datetime(2026, 1, 1, tzinfo=timezone.utc)
        def parameter(name, value, unit=""):
            return {"name": name, "value": value, "unit": unit, "date": date}

        raw = {"backend_name": "test", "backend_version": "1.0", "last_update_date": date,
               "qubits": [[parameter("T1", 100, "us"), parameter("T2", 120, "us"),
                           parameter("readout_error", 0.01)]],
               "gates": [{"gate": "sx", "name": "sx0", "qubits": [0],
                          "parameters": [parameter("gate_length", 100, "ns"),
                                         parameter("gate_error", 0.002)]}], "general": []}
        properties = BackendProperties.from_dict(raw)
        noise = self.runner.archived_noise_model(properties)
        remapped = self.runner.compact_noise_dictionary(noise.to_dict(), [0])
        rebuilt = NoiseModel.from_dict(remapped)
        self.assertFalse(rebuilt.is_ideal())
        self.assertNotIn("frequency", [item["name"] for item in raw["qubits"][0]])

    def test_legacy_problem_is_rejected_before_other_fields_are_accessed(self) -> None:
        with self.assertRaises(ValueError):
            self.runner.load_problem({"aggregate_mode": "legacy-clipped"})

    def test_partial_measurement_is_rejected(self) -> None:
        from qiskit import QuantumCircuit

        circuit = QuantumCircuit(2, 2)
        circuit.measure(0, 0)
        with self.assertRaises(ValueError):
            self.runner.classical_order_probabilities(circuit)


if __name__ == "__main__":
    unittest.main()
