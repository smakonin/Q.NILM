"""Synthetic-only workflow checks for the full-coverage QAOA extension."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from quantum_nilm.categorical_qaoa import prepare_categorical_problem


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_quantum_heldout.py"
SPEC = importlib.util.spec_from_file_location("run_quantum_heldout_for_tests", SCRIPT)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


class QuantumHeldoutTests(unittest.TestCase):
    def test_gap_chunks_cover_every_block_and_never_bridge_runs(self):
        timestamps = np.r_[np.arange(8) * 30, 270 + np.arange(6) * 30]
        mains = np.array([0, 0, 10, 10, 0, 0, 10, 10, 20, 20, 0, 0, 20, 20])
        chunks = campaign.make_chunks(mains, timestamps, 5)
        self.assertEqual([c["run"] for c in chunks], [0, 0, 1, 1])
        self.assertEqual([c["reset"] for c in chunks], [True, False, True, False])
        self.assertEqual([c["weights"] for c in chunks], [[2, 2], [2, 2], [2, 2], [2]])
        self.assertEqual([(c["block_start"], c["block_stop"]) for c in chunks],
                         [(0, 4), (4, 8), (8, 12), (12, 14)])
        self.assertEqual(sum(sum(c["weights"]) for c in chunks), len(mains))
        self.assertEqual(chunks[2]["start_unix"], 270)

    @staticmethod
    def boundary_chunks():
        return [
            {"chunk": 0, "reset": True, "aggregate": [10], "weights": [1]},
            {"chunk": 1, "reset": False, "aggregate": [0], "weights": [1]},
            {"chunk": 2, "reset": True, "aggregate": [0], "weights": [1]},
        ]

    def test_exact_chunk_uses_own_previous_state_and_resets_at_gap(self):
        records, _ = campaign.infer_chunks(self.boundary_chunks(), [np.array([0., 10.])],
                                          np.array([120.]), [1.], [.3], "exact_chunk", 16, 5)
        self.assertEqual([r["previous_states"] for r in records], [None, [1], None])
        self.assertEqual([r["states"] for r in records], [[[1]], [[1]], [[0]]])
        self.assertEqual(records[1]["best_observed_energy"], 100.)

    def test_qaoa_never_injects_unsampled_exact_state_or_its_boundary(self):
        # All hypothetical quantum shots select category 0 even when category 1
        # is the exact optimum. The next problem must carry this sampled 0.
        with patch("quantum_nilm.categorical_qaoa.categorical_qaoa_probabilities",
                   return_value=np.array([1., 0.])):
            records, _ = campaign.infer_chunks(self.boundary_chunks(), [np.array([0., 10.])],
                                              np.array([120.]), [1.], [.3], "qaoa_ideal", 16, 5)
        self.assertEqual([r["previous_states"] for r in records], [None, [0], None])
        self.assertEqual([r["states"] for r in records], [[[0]], [[0]], [[0]]])
        self.assertEqual(records[0]["conditional_exact_energy"], 0.)
        self.assertEqual(records[0]["best_observed_energy"], 100.)
        for record in records:
            self.assertEqual(record["raw_feasible_counts"], [[0, 16]])

    def test_uniform_draws_are_reproducible_and_selected_from_observed_states(self):
        arguments = (self.boundary_chunks(), [np.array([0., 10.])], np.array([120.]),
                     [1.], [.3], "uniform", 7, 12)
        first, _ = campaign.infer_chunks(*arguments)
        second, _ = campaign.infer_chunks(*arguments)
        self.assertEqual(first, second)
        for record in first:
            self.assertEqual(sum(count for _, count in record["raw_feasible_counts"]), 7)
            self.assertIn(record["chosen_index"], [index for index, _ in record["raw_feasible_counts"]])

    def test_conditional_cost_equals_direct_formula_and_onehot_polynomial(self):
        aggregate = np.array([8., 3.])
        weights = np.array([2., 4.])
        levels = [np.array([0., 5.]), np.array([1., 3., 7.])]
        penalties = np.array([9., 13.])
        previous = np.array([1, 2])
        problem = prepare_categorical_problem(aggregate, levels, penalties, weights, previous)
        for states, stored in zip(problem.states, problem.energies):
            reconstructed = np.array([sum(levels[i][row[i]] for i in range(2)) for row in states])
            direct = np.sum(weights * (aggregate - reconstructed) ** 2)
            direct += np.sum(penalties * (states[0] != previous))
            direct += np.sum(penalties * (states[1] != states[0]))
            bits = np.zeros(problem.num_qubits)
            for register, category in enumerate(states.ravel()):
                bits[problem.register_offsets[register] + category] = 1
            polynomial = problem.constant + bits @ problem.linear
            polynomial += sum(value * bits[a] * bits[b] for (a, b), value in problem.quadratic.items())
            self.assertAlmostEqual(stored, direct)
            self.assertAlmostEqual(stored, polynomial)

    def test_expansion_scores_original_block_duration_not_interval_average(self):
        timestamps = np.arange(6) * 30
        mains = np.array([0., 0., 10., 10., 10., 10.])
        chunks = campaign.make_chunks(mains, timestamps, 5)
        self.assertEqual(chunks[0]["weights"], [2, 4])
        levels = [np.array([0., 10.]), np.array([0.]), np.array([0.]), np.array([0.])]
        window = {"blocks": 6, "chunks": chunks, "window": {"id": "fake-test"}}
        records = [{"chunk": 0, "states": [[1, 0, 0, 0], [1, 0, 0, 0]]}]
        prediction = campaign.expand_predictions(window, records, levels)
        loaded = {"values": np.column_stack([mains, mains, np.zeros((6, 2))]), "timestamps": timestamps}
        score = campaign.score_prediction(window, loaded, prediction, [5., 1., 1.],
                                           "qaoa_ideal", 5, .1)
        self.assertEqual(score["blocks"], 6)
        self.assertAlmostEqual(score["appliances"]["dryr"]["mae_w"], 20 / 6)
        self.assertAlmostEqual(score["raw_aggregate_mae_w"], 20 / 6)
        self.assertEqual(score["appliances"]["dryr"]["n_blocks"], 6)

    def test_expansion_rejects_missing_records_order_and_uncovered_blocks(self):
        levels = [np.array([0., 10.])]
        chunk = {"chunk": 0, "block_start": 0, "block_stop": 2, "weights": [2]}
        window = {"blocks": 2, "chunks": [chunk]}
        with self.assertRaisesRegex(ValueError, "coverage"):
            campaign.expand_predictions(window, [], levels)
        with self.assertRaisesRegex(ValueError, "order"):
            campaign.expand_predictions(window, [{"chunk": 1, "states": [[0]]}], levels)
        with self.assertRaisesRegex(ValueError, "Incomplete|coverage"):
            campaign.expand_predictions({**window, "blocks": 3}, [{"chunk": 0, "states": [[0]]}], levels)

    def test_angle_training_uses_only_supplied_training_chunks_and_objectives(self):
        # No CSV, appliance labels, test inputs, or datasource is available in
        # this fixture. The controlled objective prefers gamma=1, beta=0.
        config = {"gamma_grid": [0., 1.], "beta_grid": [0., 1.]}
        frozen = {"models": {"multistate": {"levels_w": [[0., 1.]],
                                             "ranges_w": [1.], "event_threshold_w": .5}}}
        chunks = [{"aggregate": [3., 4.], "weights": [2, 1]},
                  {"aggregate": [5., 6.], "weights": [1, 3]}]
        documents = {"protocol.json": config, "model.json": frozen,
                     "training_chunks.json": chunks, "input_hashes.json": {"protocol.json": "fake-hash"}}
        accessed, prepared = [], []

        def reading(path):
            accessed.append(Path(path).name)
            return documents[Path(path).name]

        def preparing(aggregate, levels, penalties, weights):
            prepared.append((list(aggregate), list(weights)))
            return SimpleNamespace(energies=np.array([0., 10.]), scale=2.)

        def probability(problem, gamma, beta):
            return np.array([1., 0.]) if gamma == [1.] and beta == [0.] else np.array([0., 1.])

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch.object(campaign, "read_json", side_effect=reading), \
                    patch.object(campaign, "check_inputs"), \
                    patch("quantum_nilm.categorical_qaoa.prepare_categorical_problem", side_effect=preparing), \
                    patch("quantum_nilm.categorical_qaoa.categorical_qaoa_probabilities", side_effect=probability):
                campaign.train_angles(output)
            result = json.loads((output / "angles.json").read_text())
        self.assertEqual(prepared, [([3., 4.], [2, 1]), ([5., 6.], [1, 3])])
        self.assertNotIn("test_inputs.json", accessed)
        self.assertNotIn("test_windows.json", accessed)
        self.assertEqual(result["gammas"], [1.])
        self.assertEqual(result["betas"], [0.])
        self.assertEqual(result["objective_evaluations"], 8)
        self.assertFalse(result["label_access"])
        self.assertFalse(result["test_objectives_used_for_angle_selection"])


if __name__ == "__main__":
    unittest.main()
