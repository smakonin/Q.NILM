import unittest

import numpy as np

from quantum_nilm.ocean import (
    estimate_zephyr_embedding,
    logical_problem_profile,
    qubo_dictionary,
    sample_qubo,
    to_dimod_bqm,
)
from quantum_nilm.qubo import build_binary_temporal_qubo


class OceanAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.qubo = build_binary_temporal_qubo(
            aggregate=np.array([110.0, 410.0, 305.0]),
            appliance_powers=np.array([100.0, 300.0]),
            switch_penalty=np.array([200.0, 800.0]),
            segment_weights=np.array([2.0, 1.0, 3.0]),
        )

    def test_qubo_dictionary_matches_internal_energy(self) -> None:
        coefficients = qubo_dictionary(self.qubo)
        bits, internal = self.qubo.energies()
        external = np.full(bits.shape[0], self.qubo.constant, dtype=float)
        for (left, right), coefficient in coefficients.items():
            external += coefficient * bits[:, left] * bits[:, right]
        np.testing.assert_allclose(external, internal, atol=1e-8)

    def test_logical_profile_has_expected_temporal_graph(self) -> None:
        profile = logical_problem_profile(self.qubo)
        self.assertEqual(profile["logical_variables"], 6)
        self.assertEqual(profile["quadratic_terms"], 7)
        self.assertEqual(profile["maximum_degree"], 3)
        self.assertAlmostEqual(profile["edge_density"], 7 / 15)

    def test_dimod_bqm_is_energy_equivalent(self) -> None:
        bqm = to_dimod_bqm(self.qubo)
        bits, internal = self.qubo.energies()
        external = [
            bqm.energy({index: int(value) for index, value in enumerate(bitstring)})
            for bitstring in bits
        ]
        np.testing.assert_allclose(external, internal, atol=1e-8)

    def test_local_ocean_sampler_returns_verified_sample(self) -> None:
        result = sample_qubo(
            self.qubo,
            "simulated",
            num_reads=8,
            num_sweeps=20,
            seed=13,
        )
        self.assertEqual(result.bits.shape, (6,))
        self.assertAlmostEqual(result.energy, self.qubo.energy(result.bits), places=7)
        self.assertEqual(result.metadata["backend_name"], "simulated")

    def test_ideal_zephyr_embedding_estimate(self) -> None:
        estimate = estimate_zephyr_embedding(
            self.qubo, zephyr_m=2, seed=3, timeout_s=2
        )
        self.assertTrue(estimate["success"])
        self.assertEqual(estimate["logical_variables"], 6)
        self.assertGreaterEqual(estimate["physical_qubits"], 6)
        self.assertIn("not an embedding on a live", estimate["topology_note"])


if __name__ == "__main__":
    unittest.main()
