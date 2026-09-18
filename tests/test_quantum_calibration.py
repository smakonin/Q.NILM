import importlib.util
import itertools
from pathlib import Path
import unittest
import numpy as np

SPEC = importlib.util.spec_from_file_location("calibration_audit", Path(__file__).resolve().parents[1] / "scripts/audit_quantum_calibration.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReadoutTests(unittest.TestCase):
    def test_exact_enumeration_of_asymmetric_bit_errors(self):
        fz, fo = [.03, .08, .12], [.01, .09, .15]
        expected = []
        for category in range(3):
            total = 0.
            for bits in itertools.product((0, 1), repeat=3):
                if sum(bits) == 1:
                    total += np.prod([(1 - fz[j] if bits[j] else fz[j]) if j == category
                                      else (fo[j] if bits[j] else 1 - fo[j]) for j in range(3)])
            expected.append(total)
        np.testing.assert_allclose(MODULE.register_readout_feasibility(fz, fo), expected, atol=1e-15)

    def test_perfect_and_certain_bitflip(self):
        np.testing.assert_array_equal(MODULE.register_readout_feasibility([0, 0], [0, 0]), [1, 1])
        np.testing.assert_array_equal(MODULE.register_readout_feasibility([1, 1], [1, 1]), [1, 1])
        np.testing.assert_array_equal(MODULE.register_readout_feasibility([1, 1, 1], [1, 1, 1]), [0, 0, 0])

    def test_invalid_probabilities_rejected(self):
        for fz, fo in (([], []), ([.1], [.2, .3]), ([float('nan')], [0]), ([0], [-.1]), ([1.1], [0])):
            with self.assertRaises(ValueError):
                MODULE.register_readout_feasibility(fz, fo)
