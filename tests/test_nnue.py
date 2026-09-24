"""Run after building the native library: python -m unittest discover -s tests."""
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from hexo import Game
from nnue_model import NNUE, collate, objective
from train import nnue_example, pack_nnue, play_game


class NNUETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(137)
        cls.folder = tempfile.TemporaryDirectory(prefix="hexo-nnue-test-")
        cls.path = Path(cls.folder.name) / "model.nnue"
        cls.model = NNUE().eval()
        with torch.no_grad():
            for head in (cls.model.value, cls.model.policy):
                head[-1].weight.uniform_(-.08, .08)
                head[-1].bias.uniform_(-.08, .08)
        cls.model.export(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()
        torch.set_num_threads(cls.old_threads)

    def example(self, game, move):
        result = {"score": game.evaluation, "depth": 1, "elapsed_ms": 1.}
        row = nnue_example(game, move, result, 8)
        return row, collate(pack_nnue([row], float("nan"), 123), [0], "cpu")

    def test_native_value_policy_and_undo(self):
        game = Game([(0, 0), (8, 0), (16, 0)])
        game.load_model(self.path)
        rng = random.Random(71)
        try:
            for _ in range(20):
                move = rng.choice(game.legal_moves())
                row, batch = self.example(game, move)
                with torch.no_grad():
                    value, logits = self.model(batch)
                self.assertLessEqual(abs(float(value[0])-np.tanh(game.evaluation/6000)), 1/6000+1e-6)
                for index, candidate in enumerate(row["candidate_coords"]):
                    score = game.nnue_rank(tuple(map(int, candidate)))
                    self.assertAlmostEqual(score, float(logits[0, index]), delta=2e-5)
                before = game.nnue_inputs()
                state = game.state()
                game.play(*move)
                self.assertTrue(game.undo())
                self.assertEqual(game.state(), state)
                self.assertEqual(game.nnue_inputs(), before)
                game.play(*move)
                if game.winner >= 0:
                    break
        finally:
            game.close()

    def test_conditional_second_placement_and_phase(self):
        game = Game([(0, 0), (8, 0), (16, 0)])
        game.load_model(self.path)
        try:
            first = (1, 0)
            first_row, _ = self.example(game, first)
            self.assertEqual(first_row["phase"][:2], [0., 1.])
            self.assertTrue(np.all(first_row["pairs"] == 0))
            game.play(*first)
            second_row, batch = self.example(game, (2, 0))
            self.assertEqual(second_row["phase"][:2], [1., 0.])
            chosen = second_row["chosen"]
            np.testing.assert_allclose(second_row["pairs"][chosen], [1., 1., .125, 1.])
            self.assertEqual(second_row["ply"], 4)
            self.assertEqual(second_row["player"], first_row["player"])
            with torch.no_grad():
                _, logits = self.model(batch)
            self.assertAlmostEqual(float(logits[0, chosen]), game.nnue_rank((2, 0)), delta=2e-5)
        finally:
            game.close()

    def test_color_and_reversal_quantized_identity(self):
        codes = torch.arange(0, 3**11, 137)
        powers = 3**torch.arange(11)
        digits = codes[:, None]//powers % 3
        swap = (torch.where(digits == 0, 0, 3-digits)*powers).sum(1)
        reverse = (digits.flip(1)*powers).sum(1)
        with torch.no_grad():
            table = self.model.embeddings(codes)
            flipped = self.model.embeddings(swap)
            torch.testing.assert_close(table, self.model.embeddings(reverse), rtol=0, atol=0)
            torch.testing.assert_close(table[:, :16], -flipped[:, :16], rtol=0, atol=0)
            torch.testing.assert_close(table[:, 16:], flipped[:, 16:], rtol=0, atol=0)
            torch.testing.assert_close(table*256, torch.round(table*256), rtol=0, atol=0)

    def test_truncated_fallback_has_lower_gradient_weight(self):
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.zeros(1))

            def forward(self, batch):
                return self.value, torch.zeros((1, 2))

        batch = {"outcome": torch.tensor([float("nan")]), "search": torch.ones(1),
                 "search_valid": torch.tensor([True]), "policy_valid": torch.tensor([False]),
                 "chosen": torch.tensor([0])}
        model = Constant()
        full = objective(model, batch, .25)[0]
        full_grad = torch.autograd.grad(full, model.value)[0]
        batch["search_valid"][:] = False
        fallback = objective(model, batch, .25)[0]
        fallback_grad = torch.autograd.grad(fallback, model.value)[0]
        torch.testing.assert_close(fallback, full*.1)
        torch.testing.assert_close(fallback_grad, full_grad*.1)

    def test_malformed_native_file_is_rejected(self):
        invalid = Path(self.folder.name) / "invalid.nnue"
        invalid.write_bytes(self.path.read_bytes()[:-2])
        game = Game()
        try:
            with self.assertRaises(ValueError):
                game.load_model(invalid)
        finally:
            game.close()

    def test_quiet_exploration_does_not_create_teacher_labels(self):
        task = {"seed": 29, "tables": [str(self.path)]*2, "model_kind": "nnue",
                "opening": [(0, 0), (8, 0), (16, 0)], "family": 123,
                "evaluation": False, "ms": 5, "width": 4, "max_stones": 31,
                "policy_candidates": 8, "native_exploration": 1.}
        record, replay = play_game(task)
        self.assertGreater(record["exploratory_turns"], 0)
        self.assertLess(len(replay["family"]), len(record["cells"])-record["prefix_length"])
        game = Game(record["opening"])
        current = record["prefix_length"]
        try:
            for row, ply in enumerate(replay["ply"]):
                while current < ply:
                    game.play(*record["cells"][current][:2])
                    current += 1
                lo, hi = replay["center_offsets"][row:row+2]
                self.assertEqual(sorted(map(tuple, game.nnue_centers())),
                                 sorted(map(tuple, replay["centers"][lo:hi])))
                chosen = replay["candidate_offsets"][row]+replay["chosen"][row]
                self.assertEqual(tuple(replay["candidate_coords"][chosen]), tuple(record["cells"][ply][:2]))
        finally:
            game.close()


if __name__ == "__main__":
    unittest.main()
