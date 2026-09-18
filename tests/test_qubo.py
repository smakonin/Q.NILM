import unittest

import numpy as np

from quantum_nilm.qaoa import export_openqasm3_p1, qaoa_probabilities
from quantum_nilm.qubo import build_binary_temporal_qubo


class QNILMQUBOTests(unittest.TestCase):
    def test_qubo_matches_direct_objective(self) -> None:
        aggregate = np.array([110.0, 410.0, 300.0])
        powers = np.array([100.0, 300.0])
        penalty = 250.0
        qubo = build_binary_temporal_qubo(aggregate, powers, penalty)
        bits, costs = qubo.energies()
        for bitstring, encoded_cost in zip(bits, costs):
            states = qubo.decode(bitstring)
            direct = np.sum((aggregate - states @ powers) ** 2)
            direct += penalty * np.sum(np.diff(states, axis=0) ** 2)
            self.assertAlmostEqual(float(encoded_cost), float(direct), places=7)

    def test_ising_matches_qubo_for_every_basis_state(self) -> None:
        qubo = build_binary_temporal_qubo(
            np.array([125.0, 500.0]), np.array([100.0, 400.0]), 300.0
        )
        bits, qubo_costs = qubo.energies()
        offset, h, jmat = qubo.to_ising()
        spins = 1.0 - 2.0 * bits
        ising_costs = (
            offset
            + spins @ h
            + np.einsum("bi,ij,bj->b", spins, jmat, spins, optimize=True)
        )
        np.testing.assert_allclose(qubo_costs, ising_costs, atol=1e-9)

    def test_qaoa_probabilities_are_normalized(self) -> None:
        qubo = build_binary_temporal_qubo(
            np.array([100.0, 400.0]), np.array([100.0, 300.0]), 100.0
        )
        probabilities = qaoa_probabilities(qubo, [0.7], [0.3])
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=12)
        self.assertTrue(np.all(probabilities >= 0.0))

    def test_appliance_specific_transition_penalties(self) -> None:
        aggregate = np.array([0.0, 110.0])
        powers = np.array([100.0, 10.0])
        penalties = np.array([200.0, 3.0])
        qubo = build_binary_temporal_qubo(aggregate, powers, penalties)
        states = np.array([[0, 0], [1, 1]], dtype=np.int8)
        direct = np.sum((aggregate - states @ powers) ** 2)
        direct += np.sum(penalties * np.sum(np.diff(states, axis=0) ** 2, axis=0))
        self.assertAlmostEqual(qubo.energy(states.reshape(-1)), float(direct), places=7)

    def test_duration_weighted_segments(self) -> None:
        aggregate = np.array([100.0, 310.0])
        powers = np.array([100.0, 300.0])
        weights = np.array([4.0, 1.0])
        penalty = 25.0
        states = np.array([[1, 0], [0, 1]], dtype=np.int8)
        qubo = build_binary_temporal_qubo(
            aggregate, powers, penalty, segment_weights=weights
        )
        direct = np.sum(weights * (aggregate - states @ powers) ** 2)
        direct += penalty * np.sum(np.diff(states, axis=0) ** 2)
        self.assertAlmostEqual(qubo.energy(states.reshape(-1)), float(direct), places=7)

    def test_openqasm_export_has_measurement_for_every_qubit(self) -> None:
        qubo = build_binary_temporal_qubo(
            np.array([100.0, 400.0]), np.array([100.0, 300.0]), 100.0
        )
        qasm = export_openqasm3_p1(qubo, gamma=0.4, beta=0.2, phase_scale=1e5)
        self.assertTrue(qasm.startswith("OPENQASM 3.0;"))
        self.assertEqual(qasm.count("= measure"), qubo.n_variables)


if __name__ == "__main__":
    unittest.main()
