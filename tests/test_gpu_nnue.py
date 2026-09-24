"""GPU NNUE behavior checks; tensor semantics also run without CUDA."""
import gc
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gpu_nnue import NNUEBatch, generate_games
from hexo import Game
from nnue_model import NNUE


class GPUNNUETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.folder = tempfile.TemporaryDirectory(prefix="hexo-gpu-nnue-test-")
        cls.paths = []
        for seed in (843, 844):
            torch.manual_seed(seed)
            model = NNUE().eval()
            with torch.no_grad():
                for head in (model.value, model.policy):
                    head[-1].weight.uniform_(-.12, .12)
                    head[-1].bias.uniform_(-.12, .12)
            path = str(Path(cls.folder.name)/f"{seed}.nnue")
            model.export(path)
            cls.paths.append(path)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()
        torch.set_num_threads(cls.old_threads)

    def tensor(self, value):
        return torch.tensor(value, device=self.device)

    def test_nonzero_heads_incremental_cache_and_colors(self):
        state = NNUEBatch([self.paths], self.device, capacity=1)
        native = Game()
        try:
            for move in [(0, 0), (1, 0), (0, 1), (-1, 1), (2, 0), (2, -1), (1, -1), (3, 0)]:
                actor = state.env.player.clone()
                native.load_model(self.paths[native.player])
                np.testing.assert_allclose(state.state_inputs().cpu()[0], native.nnue_inputs(), atol=2e-5)
                actions = self.tensor([[move]])
                rank, codes, pairs = state.policy(actions)
                expected_codes, expected_pairs = native.nnue_policy_features(move)
                np.testing.assert_array_equal(codes.cpu()[0, 0], expected_codes)
                np.testing.assert_allclose(pairs.cpu()[0, 0], expected_pairs, atol=1e-6)
                self.assertAlmostEqual(rank.item(), native.nnue_rank(move), delta=3e-5)
                overlay = state.virtual(actions)
                state.step(actions[:, 0])
                native.play(*move)
                torch.testing.assert_close(overlay.pool[:, 0], state.pool, rtol=0, atol=0)
                np.testing.assert_allclose(state.state_inputs(actor).cpu()[0], native.nnue_inputs(), atol=2e-5)
                count = int(state.center_counts.item())
                self.assertEqual(sorted(map(tuple, state.codes[0, :count].cpu().tolist())),
                                 sorted(map(tuple, native.nnue_centers().tolist())))
        finally:
            native.close()

    def test_actual_conditional_turn_and_immutable_overlay(self):
        state = NNUEBatch([self.paths], self.device, capacity=1)
        state.step(self.tensor([[0, 0]]))
        native = Game([(0, 0)])
        native.load_model(self.paths[1])
        generator = torch.Generator(device=self.device).manual_seed(14)
        root_pool, root_codes = state.pool.clone(), state.codes.clone()
        parent = state.virtual(self.tensor([[[8, 0]]])).column(0)
        self.assertFalse(state.legal(self.tensor([[[16, 0]]])).item())
        self.assertTrue(state.legal(self.tensor([[[16, 0]]]), parent).item())
        native.play(8, 0)
        np.testing.assert_allclose(state.state_inputs(parent=parent).cpu()[0], native.nnue_inputs(), atol=2e-5)
        leaf = state.virtual(self.tensor([[[16, 0]]]), parent)
        native.play(16, 0)
        expected = np.tanh(-native.evaluation/6000)
        self.assertAlmostEqual(state.value(leaf, state.env.player).item(), expected, delta=3e-4)
        torch.testing.assert_close(root_pool, state.pool, rtol=0, atol=0)
        torch.testing.assert_close(root_codes, state.codes, rtol=0, atol=0)
        native.undo()
        native.undo()
        np.testing.assert_allclose(state.state_inputs().cpu()[0], native.nnue_inputs(), atol=2e-5)

        def controlled_candidates(count, rng, parent=None):
            action = self.tensor([[[8, 0] if parent is None else [16, 0]]])
            return action, state.legal(action, parent)

        with patch.object(state, "sample", controlled_candidates):
            first = state.choose(1, generator, epsilon=0, beam=1)
            self.assertEqual(first["move"].tolist(), [[8, 0]])
            self.assertEqual(first["planned"].tolist(), [[16, 0]])
            self.assertAlmostEqual(first["search"].item(), expected, delta=3e-4)
            state.step(first["move"])
        second = state.choose(1, generator, epsilon=0, beam=1, planned=first["planned"])
        self.assertEqual(second["move"].tolist(), [[16, 0]])
        state.step(second["move"])
        self.assertEqual(state.env.player.item(), 0)
        self.assertEqual(state.env.remaining.item(), 2)
        native.close()

    def test_planned_second_cannot_be_discarded_by_policy_rank(self):
        state = NNUEBatch([self.paths], self.device, capacity=8)
        for move in [(0, 0), (1, 0), (0, 1), (3, 0)]:
            state.step(self.tensor([move]))
        planned = self.tensor([[8, 0]])
        original_policy = state.policy

        def low_rank_for_plan(actions, actor=None, parent=None):
            rank, codes, pairs = original_policy(actions, actor, parent)
            rank = torch.where((actions == planned[:, None]).all(-1), -100., 0.)
            return rank, codes, pairs

        def value_with_better_plan(overlay, actor):
            return torch.where((overlay.action == planned[:, None]).all(-1), .8, -.2)

        def alternatives(count, generator, parent=None):
            actions = self.tensor([[[1, 1], [2, 0]]])
            return actions, state.legal(actions, parent)

        generator = torch.Generator(device=self.device).manual_seed(843)
        with patch.object(state, "policy", low_rank_for_plan), patch.object(state, "value", value_with_better_plan), patch.object(state, "sample", alternatives):
            decision = state.choose(2, generator, epsilon=0, beam=2, planned=planned)
        self.assertEqual(decision["move"].tolist(), planned.tolist())
        self.assertAlmostEqual(decision["search"].item(), .8, delta=1e-6)
        self.assertTrue(decision["search_valid"].item())

    def test_variable_prefix_replay_is_on_host_and_native_exact(self):
        prefixes = [[(0, 0)], [(0, 0), (8, 0), (16, 0)],
                    [(0, 0), (1, 0), (0, 1), (3, 0), (3, -1)]]
        tasks = [dict(seed=50+i, model_kind="nnue", evaluation=False, tables=self.paths,
                      opening=opening, family=100+i, curriculum="test-prefixes", max_stones=13)
                 for i, opening in enumerate(prefixes)]
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        results = generate_games(tasks, device=self.device, candidates=8, beam=2, epsilon=.25)
        if self.device == "cuda":
            self.assertLess(torch.cuda.max_memory_allocated()-initial_memory, 96*1024**2)
        for task, (record, replay) in zip(tasks, results):
            self.assertEqual(record["opening"], task["opening"])
            self.assertEqual(record["prefix_length"], len(task["opening"]))
            self.assertEqual(record["family"], task["family"])
            self.assertTrue(all(isinstance(value, np.ndarray) for value in replay.values()))
            native = Game()
            try:
                for ply, (q, r, owner) in enumerate(record["cells"]):
                    self.assertEqual(native.player, owner)
                    if ply >= record["prefix_length"]:
                        index = ply-record["prefix_length"]
                        native.load_model(self.paths[owner])
                        self.assertEqual(replay["ply"][index], ply)
                        lo, hi = replay["center_offsets"][index:index+2]
                        self.assertEqual(sorted(map(tuple, replay["centers"][lo:hi].tolist())),
                                         sorted(map(tuple, native.nnue_centers().tolist())))
                        np.testing.assert_allclose(replay["phase"][index], native.nnue_context())
                        lo, hi = replay["candidate_offsets"][index:index+2]
                        for candidate in range(lo, hi):
                            coords = tuple(map(int, replay["candidate_coords"][candidate]))
                            codes, pairs = native.nnue_policy_features(coords)
                            np.testing.assert_array_equal(replay["candidate_codes"][candidate], codes)
                            np.testing.assert_allclose(replay["pairs"][candidate], pairs)
                        self.assertEqual(replay["candidate_coords"][lo+replay["chosen"][index]].tolist(), [q, r])
                    native.play(q, r)
                self.assertEqual(native.winner, record["winner"])
                if native.winner < 0:
                    self.assertTrue(np.isnan(replay["outcome"]).all())
            finally:
                native.close()


if __name__ == "__main__":
    unittest.main()
