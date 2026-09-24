"""Checkpoint import must preserve learned state without importing old ratings."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from hexo import Game
from nnue_model import NNUE
from train import _run_training, initial_artifacts, initialize_nnue


class WarmStartTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.threads)
        torch.manual_seed(43)
        model = NNUE()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, .03)
        optimizer.step()
        torch.save(model.state_dict(), self.root/"source.pt")
        torch.save(optimizer.state_dict(), self.root/"optimizer.pt")
        model.export(self.root/"source.nnue")
        self.args = SimpleNamespace(model="nnue", initial_model=str(self.root/"source.pt"),
            initial_optimizer=str(self.root/"optimizer.pt"), lr=.002, run=str(self.root/"run"),
            seed=1729, curriculum="mixed-v1", selfplay_backend="native", workers=1, iterations=0)

    def test_exact_weights_native_export_optimizer_and_zero_ratings(self):
        _run_training(self.args)
        run = Path(self.args.run)
        initial = run/"checkpoints/0000"
        self.assertEqual((initial/"model.pt").read_bytes(), (self.root/"source.pt").read_bytes())
        self.assertEqual((initial/"model.nnue").read_bytes(), (self.root/"source.nnue").read_bytes())
        original = torch.load(self.root/"optimizer.pt", weights_only=True)
        imported = torch.load(initial/"optimizer.pt", weights_only=True)
        self.assertEqual(imported["param_groups"][0]["lr"], .002)
        for index, state in original["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(imported["state"][index][key], value, rtol=0, atol=0)
        game = Game([(0, 0), (1, 1), (0, 1), (1, 0)])
        self.addCleanup(game.close)
        game.load_model(self.root/"source.nnue")
        expected = game.evaluation, game.nnue_rank((2, 0))
        game.load_model(initial/"model.nnue")
        self.assertEqual((game.evaluation, game.nnue_rank((2, 0))), expected)
        checkpoint = json.loads((run/"summary.json").read_text())["checkpoints"][0]
        self.assertEqual(checkpoint["anchor_elo"], 0)
        self.assertEqual(checkpoint["evaluations"], [])
        self.assertEqual(checkpoint["optimizer"], "checkpoints/0000/optimizer.pt")
        self.assertEqual(checkpoint["initial_artifacts"]["model"]["sha256"], hashlib.sha256((self.root/"source.pt").read_bytes()).hexdigest())

    def test_resume_rejects_changed_source_without_overwriting(self):
        _run_training(self.args)
        initial = Path(self.args.run)/"checkpoints/0000/model.pt"
        original = initial.read_bytes()
        _run_training(self.args)
        with (self.root/"source.pt").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "same configuration"):
            _run_training(self.args)
        self.assertEqual(initial.read_bytes(), original)

    def test_incompatible_model_and_existing_checkpoint_never_overwritten(self):
        parent = self.root/"checkpoints"
        parent.mkdir()
        initial = parent/"0000"
        torch.save({"bad": torch.ones(2)}, self.root/"source.pt")
        with self.assertRaises(RuntimeError):
            initialize_nnue(initial, self.args, initial_artifacts(self.args))
        self.assertFalse(initial.exists())
        initial.mkdir()
        (initial/"model.pt").write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            initialize_nnue(initial, self.args, initial_artifacts(self.args))
        self.assertEqual((initial/"model.pt").read_bytes(), b"keep")

    def test_incompatible_optimizer_rejected_before_checkpoint_publication(self):
        state = torch.load(self.root/"optimizer.pt", weights_only=True)
        state["state"][0]["exp_avg"] = torch.ones(1)
        torch.save(state, self.root/"optimizer.pt")
        parent = self.root/"checkpoints"
        parent.mkdir()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            initialize_nnue(parent/"0000", self.args, initial_artifacts(self.args))
        self.assertFalse((parent/"0000").exists())


if __name__ == "__main__":
    unittest.main()
