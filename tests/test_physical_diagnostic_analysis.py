"""Synthetic controls for raw-bit ordering, decay fits and failure reporting."""
import json
import unittest

import numpy as np

from quantum_nilm.physical_diagnostic_analysis import analyze_rows, fit_rb_decay


def row(identifier, group, counts, **metadata):
    return {"id": identifier, "group": group, "n_clbits": len(next(iter(counts))),
            "shots": sum(counts.values()), "raw_counts": counts, **metadata}


def rb_rows(reference=.98, interleaved=.96, shots=1000000, edges=("suspect",)):
    result = []
    for edge in edges:
        for seed in range(8):
            for length in (1, 4, 16, 64, 128):
                for arm, alpha in (("reference", reference), ("interleaved", interleaved)):
                    p = .72 * alpha ** length + .25
                    n = round(shots * p)
                    result.append(row(f"{edge}_{seed}_{length}_{arm}", "gate", {"00": n, "11": shots - n},
                        kind="rb", edge=[1, 2], edge_label=edge, seed=seed, length=length, arm=arm))
    return result


class PhysicalDiagnosticAnalysisTests(unittest.TestCase):
    def test_local_bit_order_and_reset(self):
        rows = [row("basis", "local", {"100": 9, "101": 1}, kind="basis", placementlabel="suspect",
                    physical_qubits=[131, 132, 133], input_state="100"),
                row("reset", "local", {"000": 7, "001": 3}, kind="reset", placementlabel="suspect",
                    physical_qubits=[131, 132, 133])]
        result = analyze_rows(rows)["local"]["suspect"]
        self.assertEqual([x["fraction"] for x in result["basis"][0]["bit_error"]], [.1, 0., 0.])
        self.assertEqual(result["reset"][0]["bit_error"][0]["fraction"], .3)

    def test_w_counts_are_not_fidelity(self):
        rows = [row("w", "local", {"001": 30, "010": 30, "100": 30, "111": 10}, kind="w",
                    placementlabel="suspect", physical_qubits=[1, 2, 3], repeat=0)]
        result = analyze_rows(rows)["local"]["suspect"]["w"][0]
        self.assertAlmostEqual(result["onehot_feasibility"]["fraction"], .9)
        self.assertAlmostEqual(result["measurement_tv_to_uniform_onehot"], .1)

    def test_cost_onehot_register_order_and_unconditional_mass(self):
        valid = sum(1 << offset for offset in (0, 4, 7, 9))
        counts = {f"{valid:012b}": 8, "000000000000": 2}
        rows = [row("cost", "cost", counts, width=1, state_counts=[4, 3, 2, 3], example_id=0, arm="nominal")]
        result = analyze_rows(rows)["cost"]["conditions"][0]
        self.assertEqual(result["all_register_onehot"]["fraction"], .8)
        self.assertEqual(result["registers"][1]["classical_bits"], [4, 5, 6])
        self.assertEqual(result["bit_one"][0]["fraction"], .8)

    def test_two_interval_cost_checks_all_eight_registers(self):
        first = sum(1 << offset for offset in (0, 4, 7, 9))
        both = first | (first << 12)
        only_first = first
        rows = [row("cost24", "cost", {f"{both:024b}": 7, f"{only_first:024b}": 3},
                    width=2, state_counts=[4, 3, 2, 3], example_id=1, arm="zero")]
        result = analyze_rows(rows)["cost"]["conditions"][0]
        self.assertEqual(result["all_register_onehot"]["fraction"], .7)
        self.assertEqual(len(result["registers"]), 8)
        self.assertEqual(result["registers"][0]["onehot_feasibility"]["fraction"], 1.)
        self.assertEqual(result["registers"][4]["onehot_feasibility"]["fraction"], .7)

    def test_ramsey_primary_target_bit1_not_postselected(self):
        rows = [row("ramsey", "gate", {"00": 60, "10": 20, "11": 20}, kind="ramsey",
                    edge=[132, 133], edge_label="suspect", control=0, basis="X", repetitions=1)]
        result = analyze_rows(rows)["ramsey"]["conditions"][0]
        self.assertAlmostEqual(result["target_pauli_expectation_all_shots"], .2)
        self.assertAlmostEqual(result["control_flip"]["fraction"], .2)
        self.assertAlmostEqual(result["conditional_target_expectation_secondary"], .5)

    def test_ramsey_ideal_x_and_y(self):
        rows = [row("x", "gate", {"11": 100}, kind="ramsey", edge=[1, 2], edge_label="suspect", control=1, basis="X", repetitions=1),
                row("y", "gate", {"01": 50, "11": 50}, kind="ramsey", edge=[1, 2], edge_label="suspect", control=1, basis="Y", repetitions=1)]
        result = analyze_rows(rows)["ramsey"]
        self.assertEqual(result["conditions"][0]["target_pauli_expectation_all_shots"], -1.)
        self.assertEqual(result["conditions"][1]["target_pauli_expectation_all_shots"], 0.)
        self.assertAlmostEqual(result["xy_pairs"][0]["wrapped_phase_rad"], np.pi)

    def test_exact_depolarizing_decay_fit(self):
        lengths = np.array([1, 4, 16, 64, 128])
        fit = fit_rb_decay(lengths, .72 * .97 ** lengths + .25)
        self.assertAlmostEqual(fit["alpha"], .97, places=6)
        self.assertAlmostEqual(fit["A"], .72, places=5)
        self.assertAlmostEqual(fit["B"], .25, places=5)
        self.assertTrue(fit["identifiable"])

    def test_paired_rb_and_edge_contrast(self):
        result = analyze_rows(rb_rows(edges=("suspect", "control")), bootstrap_replicates=5)["rb"]
        expected = .75 * (1 - .96 / .98)
        self.assertAlmostEqual(result["edges"]["suspect"]["gate_error_estimate_unclipped"], expected, places=5)
        self.assertEqual(result["paired_edge_contrast"]["difference"], 0.)
        self.assertEqual(result["paired_edge_contrast"]["paired_seed_cluster_95"], [0., 0.])
        json.dumps(result, allow_nan=False)

    def test_negative_irb_error_is_preserved(self):
        result = analyze_rows(rb_rows(reference=.96, interleaved=.98), bootstrap_replicates=2)["rb"]["edges"]["suspect"]
        self.assertLess(result["gate_error_estimate_unclipped"], 0.)
        self.assertIn("negative_gate_error_estimate_retained", result["quality_flags"])

    def test_ideal_flat_rb_is_unresolved_not_falsely_precise_zero(self):
        result = analyze_rows(rb_rows(reference=1., interleaved=1.), bootstrap_replicates=2)["rb"]["edges"]["suspect"]
        self.assertEqual(result["status"], "unresolved")
        self.assertIsNone(result["gate_error_estimate_unclipped"])

    def test_invalid_inputs_and_unpaired_conditions_rejected(self):
        rows = rb_rows()
        with self.assertRaises(ValueError):
            analyze_rows(rows[:-1], bootstrap_replicates=1)
        with self.assertRaises(ValueError):
            analyze_rows([rows[0], rows[0]])
        broken = dict(rows[0], shots=0)
        with self.assertRaises(ValueError):
            analyze_rows([broken])
        broken = dict(rows[0], raw_counts={"0": rows[0]["shots"]})
        with self.assertRaises(ValueError):
            analyze_rows([broken])
        broken = dict(rows[0], logical_measurement_bit_to_physical=[2, 1])
        with self.assertRaises(ValueError):
            analyze_rows([broken])


if __name__ == "__main__":
    unittest.main()
