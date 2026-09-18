"""Independent checks for controlled local Stage B noise channels."""

import importlib.util
import itertools
import unittest

import numpy as np

from quantum_nilm.categorical_qaoa import categorical_qaoa_probabilities, prepare_categorical_problem
from quantum_nilm.stage_b_noise import (
    NoiseParameters, apply_readout_bitflips, prepare_noise_circuit, simulate_prepared_noise,
)


class ReadoutChannelTests(unittest.TestCase):
    def test_channel_matches_direct_hamming_probability_sum(self):
        initial = np.array([0.1, 0.2, 0.3, 0.4])
        rate = 0.17
        expected = np.zeros(4)
        for source, destination in itertools.product(range(4), repeat=2):
            distance = (source ^ destination).bit_count()
            expected[destination] += initial[source] * rate**distance * (1 - rate)**(2 - distance)
        np.testing.assert_allclose(apply_readout_bitflips(initial, rate), expected, atol=1e-15)

    def test_zero_and_full_flip_have_exact_basis_semantics(self):
        initial = np.array([0.0, 0.1, 0.3, 0.6])
        np.testing.assert_allclose(apply_readout_bitflips(initial, 0), initial)
        np.testing.assert_allclose(apply_readout_bitflips(initial, 1), initial[::-1])
        np.testing.assert_allclose(apply_readout_bitflips(initial, .5), np.full(4, .25))

    def test_readout_rejects_missing_probability_mass(self):
        for initial in ([.2, .2], [.5, -.5, .5, .5], [1., 0., 0.]):
            with self.assertRaises(ValueError):
                apply_readout_bitflips(initial, .1)
        for rate in (-.1, 1.1, np.nan, True):
            with self.assertRaises(ValueError):
                apply_readout_bitflips([.5, .5], rate)

    def test_noise_parameters_are_bounded(self):
        for rate in (-.1, 1.1, np.inf, True):
            with self.assertRaises(ValueError):
                NoiseParameters("bad", rate, 0., 0.)


@unittest.skipUnless(importlib.util.find_spec("qiskit_aer"), "Optional Aer dependency absent")
class CircuitNoiseTests(unittest.TestCase):
    def test_ideal_audit_matches_mixed_register_p1_and_p2(self):
        problem = prepare_categorical_problem([27.], [[0., 20.], [0., 7., 30.]], [1., 2.])
        for gammas, betas in (([1.2], [.4]), ([.7, 1.2], [.3, .5])):
            prepared = prepare_noise_circuit(problem, gammas, betas)
            result = simulate_prepared_noise(prepared)
            np.testing.assert_allclose(result["feasible_probabilities"],
                categorical_qaoa_probabilities(problem, gammas, betas), atol=1e-12, rtol=1e-10)
            self.assertLess(result["invalid_probability"], 1e-12)
            self.assertFalse(result["probabilities_conditioned_on_feasibility"])

    def test_readout_only_onehot_retains_transfers_between_categories(self):
        problem = prepare_categorical_problem([10.], [[0., 10., 25.]], [0.])
        prepared = prepare_noise_circuit(problem, [0.], [0.])
        rate = .13
        result = simulate_prepared_noise(prepared, NoiseParameters("readout", 0., 0., rate))
        # A valid three-bit one-hot word can remain unchanged, or move its
        # excitation via exactly two flips; both mechanisms remain feasible.
        expected_feasible = (1 - rate)**3 + 2 * rate**2 * (1 - rate)
        self.assertAlmostEqual(result["feasible_probability"], expected_feasible, places=12)
        self.assertAlmostEqual(result["invalid_probability"], 1 - expected_feasible, places=12)
        np.testing.assert_allclose(result["feasible_probabilities"], expected_feasible / 3, atol=1e-12)

    def test_gate_noise_leaks_and_never_conditions_on_valid_outcomes(self):
        problem = prepare_categorical_problem([10.], [[0., 10.], [0., 6.]], [0., 0.])
        prepared = prepare_noise_circuit(problem, [.7], [.4])
        result = simulate_prepared_noise(prepared, "high", max_parallel_threads=1)
        self.assertGreater(result["invalid_probability"], 0)
        self.assertAlmostEqual(float(result["physical_probabilities"].sum()), 1., places=12)
        self.assertAlmostEqual(float(result["feasible_probabilities"].sum()) + result["invalid_probability"], 1., places=12)
        self.assertEqual(result["simulation_method"], "exact_density_matrix")
        self.assertFalse(result["classical_fallback_used"])
        self.assertFalse(result["noise_model_is_hardware_calibration"])

    def test_full_depolarization_is_uniform_over_physical_space(self):
        problem = prepare_categorical_problem([10.], [[0., 10.]], [0.])
        prepared = prepare_noise_circuit(problem, [.7], [.4])
        result = simulate_prepared_noise(prepared, NoiseParameters("maximal", 1., 1., 0.))
        np.testing.assert_allclose(result["physical_probabilities"], .25, atol=1e-12)
        self.assertAlmostEqual(result["feasible_probability"], .5, places=12)

    def test_width_cap_precedes_density_allocation(self):
        problem = prepare_categorical_problem([10., 5.], [[0., 10., 20.]], [0.])
        with self.assertRaises(ValueError):
            prepare_noise_circuit(problem, [.1], [.2], max_qubits=5)
        with self.assertRaises(ValueError):
            prepare_noise_circuit(problem, [.1], [.2], max_qubits=13)


if __name__ == "__main__":
    unittest.main()
