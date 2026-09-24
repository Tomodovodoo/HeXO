"""Trajectory provenance and teacher semantics for native NNUE reanalysis."""
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hexo import Game
from nnue_model import NNUE, collate, objective
from reanalysis import analyze, reconstruct, run, validate_history


def record():
    game = Game([(0, 0), (0, 3), (1, 3), (1, 0), (2, 0), (3, 3), (4, 3),
                 (3, 0), (4, 0), (5, 3), (6, 3), (5, 0)])
    return {"cells": game.cells, "winner": game.winner, "reason": "six-in-a-row",
            "family": 101, "opening": [[0, 0], [0, 3], [1, 3], [1, 0], [2, 0]],
            "prefix_length": 5, "curriculum": "mixed-v1"}


class ReanalysisTest(unittest.TestCase):
    def test_absolute_ply_variable_prefix_and_conditional_phase(self):
        history = record()
        self.assertEqual(validate_history(history), 5)
        first, second = reconstruct(history, 7), reconstruct(history, 8)
        self.assertEqual((first.player, first.remaining), (0, 2))
        self.assertEqual((second.player, second.remaining), (0, 1))
        np.testing.assert_array_equal(second.cells, history["cells"][:8])
        bad = {**history, "opening": [[0, 0]]}
        with self.assertRaises(ValueError):
            validate_history(bad)

    def teacher(self, moves, depth=2):
        return {"moves": moves, "score": 1200, "depth": depth, "nodes": 100, "elapsed_ms": 20}

    def test_counterfactual_continuation_does_not_inherit_outcome(self):
        with patch.object(Game, "load_model"), patch.object(Game, "search", return_value=self.teacher([(2, 1), (3, 1)])):
            data, meta = analyze(record(), 7, "unused", 20, 8, 2, 8)
        np.testing.assert_array_equal(data["ply"], [7, 8])
        self.assertEqual(data["outcome"][0], 1)
        self.assertTrue(np.isnan(data["outcome"][1]))
        self.assertEqual([r["on_source_trajectory"] for r in meta["rows"]], [True, False])
        expected = reconstruct(record(), 7)
        expected.play(2, 1)
        lo, hi = data["center_offsets"][1:3]
        np.testing.assert_array_equal(data["centers"][lo:hi], expected.nnue_centers())
        self.assertEqual(meta["score_semantics"], "selective-root-estimate")

    def test_matching_prefix_retains_outcome_and_fallback_masks_policy(self):
        with patch.object(Game, "load_model"), patch.object(Game, "search", return_value=self.teacher([(3, 0), (4, 0)], 0)):
            data, meta = analyze(record(), 7, "unused", 20, 8, 2, 8)
        np.testing.assert_array_equal(data["outcome"], [1, 1])
        self.assertFalse(data["policy_valid"].any())
        self.assertFalse(data["search_valid"].any())
        self.assertEqual(meta["score_semantics"], "incomplete-search-fallback")
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            model = NNUE()
            loss = objective(model, collate(data, [0, 1], "cpu"), .25)[0]
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(any(p.grad is not None for p in model.parameters()))
        finally:
            torch.set_num_threads(old_threads)

    def test_missing_mixed_prefix_and_bad_owner_rejected(self):
        history = record()
        del history["opening"]
        del history["prefix_length"]
        with self.assertRaises(ValueError):
            validate_history(history)
        history = record()
        history["cells"][6][2] = 0
        with self.assertRaises(ValueError):
            validate_history(history)

    def test_durable_resume_and_provenance_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"source"
            (source/"data").mkdir(parents=True)
            (source/"data/0001-games.json").write_text(json.dumps([record()]))
            # Search/model loading are replaced here; native geometry remains real.
            (source/"teacher.nnue").write_bytes(b"frozen evaluator")
            (source/"summary.json").write_text(json.dumps({"config": {"curriculum": "mixed-v1"},
                "checkpoints": [{"id": 0, "kind": "nnue", "nnue": "teacher.nnue"}]}))
            args = SimpleNamespace(run=source, output=Path(directory)/"reanalysis", iteration=1,
                checkpoint=0, seed=2, max_positions=1, ms=20, width=8, depth=2, policy_candidates=8)
            def teacher(game, *unused, **kwargs):
                ply = len(game.cells)
                return self.teacher([tuple(c[:2]) for c in record()["cells"][ply:ply+game.remaining]])
            with patch.object(Game, "load_model"), patch.object(Game, "search", autospec=True, side_effect=teacher) as search:
                first = run(args)
                second = run(args)
                self.assertEqual(search.call_count, 1)
                self.assertEqual(first["replay_sha256"], second["replay_sha256"])
                args.ms += 1
                with self.assertRaisesRegex(ValueError, "provenance"):
                    run(args)
                args.ms -= 1
                (args.output/first["completed"][0]["file"]).write_bytes(b"corruption")
                with self.assertRaisesRegex(ValueError, "fragment changed"):
                    run(args)


if __name__ == "__main__":
    unittest.main()
