"""KLENT equations, mover frames, uncropped actions, and durable corpus semantics."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import copy
import hashlib

import numpy as np
import torch

from hexo import Game
from klent import (Model, act, collect, improved_policy, load_corpus, loss, main,
                   observe, pack, rebuild, save_corpus, signed_returns, fit)
from dashboard import klent_run


class KlentTest(unittest.TestCase):
    def test_fitting_progress_counts_actor_and_value_passes(self):
        model = Model()
        with torch.no_grad():
            model.q[2].bias.fill_(.2)
        args = self.args()
        episodes, rows = collect(model, args, 1)
        optimizer = torch.optim.Adam([p for n, p in model.named_parameters() if not n.startswith("nnue.value.")])
        value_optimizer = torch.optim.Adam(model.nnue.value.parameters())
        updates = []
        metrics = fit(model, optimizer, value_optimizer, episodes, rows, args, 1, progress=updates.append)
        self.assertEqual({u["fit_phase"] for u in updates}, {"actor and Q", "deployment value"})
        self.assertEqual(metrics["examples_processed"], len(rows))
        self.assertEqual(metrics["value_examples_processed"], len(rows))
        self.assertEqual(updates[-1]["fit_completed"], len(rows))
        self.assertEqual(updates[-1]["optimizer_steps"], metrics["optimizer_steps"])
        self.assertEqual(updates[-1]["value_optimizer_steps"], metrics["value_optimizer_steps"])

    def test_dashboard_old_status_and_published_checkpoint_not_double_counted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"checkpoints/0000").mkdir(parents=True)
            initial = {"schema": "hexo-klent-scalar-v1", "identity": {"config": {"games": 2}},
                       "metrics": None, "files": {"klent.pt": "initial-actor"}}
            (root/"checkpoints/0000/manifest.json").write_text(json.dumps(initial))
            (root/"status.json").write_text(json.dumps({"iteration": 1, "stage": "fitting", "positions": 6}))
            (root/"corpus/0001").mkdir(parents=True)
            (root/"corpus/0001/manifest.json").write_text(json.dumps({"identity": {"actor_sha256": "initial-actor"}}))
            (root/"corpus/0001/episodes.json").write_text(json.dumps([{"winner": 0}, {"winner": -1}]))
            active = klent_run(root)
            self.assertEqual(active["rating"], "UNRATED")
            self.assertEqual(active["totals"]["games"], 2)
            self.assertEqual(active["totals"]["terminal_games"], 1)
            self.assertNotIn("examples_processed", active["active"])
            metrics = {"games": 2, "positions": 6, "terminal_games": 1, "bootstrapped_games": 1, "optimizer_steps": 2}
            (root/"checkpoints/0001").mkdir()
            (root/"checkpoints/0001/manifest.json").write_text(json.dumps({**initial, "metrics": metrics}))
            (root/"status.json").write_text(json.dumps({"iteration": 1, "stage": "finished", **metrics}))
            self.assertEqual(klent_run(root)["totals"], metrics)
            (root/"evaluation").mkdir()
            evaluation = {"stage": "native", "completed": 4, "total": 160, "wins": 1, "losses": 3}
            (root/"evaluation/status.json").write_text(json.dumps(evaluation))
            observed = klent_run(root)
            self.assertEqual(observed["evaluation"], evaluation)
            self.assertEqual(observed["rating"], "UNRATED")

    def setUp(self):
        old = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, old)
        torch.manual_seed(54)

    def test_bulk_features_match_scalar_for_full_legal_actions(self):
        import random
        from tests.reference import interleave
        rng = random.Random(20261008)
        game = Game()
        self.addCleanup(game.close)
        phases = set()
        for ply in range(32):
            legal = game.legal_moves()
            phases.add((game.player, game.remaining))
            before = (game.key, game.state(), game.features(), game.nnue_centers().tobytes())
            # Reversed views exercise noncontiguous input while preserving order.
            coords = np.asarray(legal, dtype=np.int64)[::-1]
            codes, pairs = game.nnue_policy_batch(coords)
            scalar = [game.nnue_policy_features(tuple(c)) for c in coords]
            expected_codes = np.asarray([x[0] for x in scalar], np.int32)
            expected_pairs = np.asarray([x[1] for x in scalar], np.float32)
            self.assertEqual(codes.tobytes(), expected_codes.tobytes())
            self.assertEqual(pairs.tobytes(), expected_pairs.tobytes())
            observation = observe(game)
            self.assertEqual(observation['legal'], legal)
            self.assertEqual(observation['candidate_codes'].tobytes(), codes[::-1].tobytes())
            self.assertEqual(observation['pairs'].tobytes(), pairs[::-1].tobytes())
            self.assertEqual((game.key, game.state(), game.features(), game.nnue_centers().tobytes()), before)
            # The second placement at distance sixteen relies on the first one.
            point = ((0,0), (8,0), (16,0))[ply] if ply < 3 else rng.choice(legal)
            game.play(*point)
        self.assertEqual(phases, {(0,1), (0,2), (1,1), (1,2)})
        codes, pairs = game.nnue_policy_batch([])
        self.assertEqual((codes.shape, pairs.shape), ((0,3), (0,4)))
        for bad in ([[0,0]], [[10**12+1,0]], [[1.5,0]], [[True,False]], [1,2], [[2**64-1,0]]):
            with self.assertRaises(ValueError):
                game.nnue_policy_batch(bad)
        history = interleave([[(q,0) for q in range(6)], [(2*q,6) for q in range(6)]])
        terminal = Game(history)
        self.addCleanup(terminal.close)
        with self.assertRaises(ValueError):
            terminal.nnue_policy_batch([[0,8]])

    def args(self, **updates):
        values = dict(seed=1729, device="cpu", alpha=.03, beta=.1, gamma=1., lambda_return=.939,
                      games=2, envs=2, max_plies=3, batch=4, cells=65536, centers=65536,
                      lr=.001, initial_model=None, iterations=1)
        return SimpleNamespace(**(values | updates))

    def test_exact_improvement_and_ragged_segment_normalization(self):
        logits = torch.tensor([2., -1., 3., 0., -2.], dtype=torch.float64)
        q = torch.tensor([.2, -.4, .1, .8, -.5], dtype=torch.float64)
        owner = torch.tensor([0, 0, 1, 1, 1])
        mu, value, kl, entropy = improved_policy(logits, q, owner, 2, .03, .1)
        for i, span in enumerate((slice(0, 2), slice(2, 5))):
            expected = torch.softmax((q[span]+.1*torch.log_softmax(logits[span], 0))/.13, 0)
            torch.testing.assert_close(mu[span].double(), expected, atol=1e-7, rtol=1e-6)
            self.assertAlmostEqual(value[i].item(), (mu[span]*q[span]).sum().item(), places=6)
            self.assertGreaterEqual(kl[i].item(), -1e-6)
            self.assertGreaterEqual(entropy[i].item(), 0)

    def test_signed_returns_follow_mover_not_ply(self):
        players = [0, 1, 1, 0, 0]
        np.testing.assert_array_equal(signed_returns(players, [.2]*5, True, lam=1), [1, -1, -1, 1, 1])
        np.testing.assert_allclose(signed_returns(players, [.2]*5, True, lam=0), [-.2, .2, -.2, .2, 1])
        self.assertAlmostEqual(signed_returns([0], [.1], False, 0, .7)[0], .7, places=6)
        self.assertAlmostEqual(signed_returns([0], [.1], False, 1, .7)[0], -.7, places=6)
        # Terminal winning placement never bootstraps a flipped post-win mover.
        self.assertEqual(signed_returns([0], [.1], True, 1, -.9)[0], 1)
        with self.assertRaisesRegex(ValueError, "tail bootstrap"):
            signed_returns([0], [.1], False)

    def test_full_legal_support_includes_distant_and_conditional_cells(self):
        game = Game([(0, 0)])
        self.addCleanup(game.close)
        first = observe(game)
        self.assertEqual(len(first["legal"]), 216)
        self.assertIn((8, 0), first["legal"])
        game.play(8, 0)
        second = observe(game)
        self.assertIn((16, 0), second["legal"])
        self.assertNotIn((16, 0), first["legal"])
        batch = pack([first, second], "cpu")
        model = Model()
        logits, q, inputs = model(batch)
        self.assertEqual(len(logits), len(first["legal"])+len(second["legal"]))
        self.assertEqual(inputs.shape, (2, 68))
        mu, _, _, _ = improved_policy(logits, q, batch["candidate_owner"], 2, .03, .1)
        target = mu.detach()
        taken = batch["offsets"][:-1]
        objective, ce, mse = loss(model, batch, target, taken, torch.tensor([1., -1.]))
        self.assertAlmostEqual(mse.item(), 1.)
        objective.backward()
        self.assertIsNotNone(model.q[2].weight.grad)
        self.assertTrue(all(p.grad is None for p in model.nnue.value.parameters()))

    def test_terminal_collection_and_counterfactual_free_rebuild(self):
        winning = [(0, 0), (0, 3), (1, 3), (1, 0), (2, 0), (3, 3),
                   (4, 3), (3, 0), (4, 0), (5, 3), (6, 3), (5, 0)]
        cursor = [0]
        def scripted(model, observations, args):
            obs = observations[0]
            mu = np.zeros(len(obs["legal"]), np.float32)
            mu[obs["legal"].index(winning[cursor[0]])] = 1
            cursor[0] += 1
            return [(mu, .2, 0., 0.)]
        with patch("klent.act", side_effect=scripted):
            episodes, rows = collect(Model(), self.args(games=1, envs=1, max_plies=20, lambda_return=1), 1)
        self.assertEqual(episodes[0]["winner"], 0)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(not row["bootstrapped"] for row in rows))
        self.assertEqual(rows[-1]["target"], 1)
        for row in rows:
            self.assertEqual(row["target"], 1 if row["player"] == 0 else -1)
            rebuild(row, {0: episodes[0]})

    def test_cap_corpus_roundtrip_and_modified_resume_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.args(run=str(Path(folder)/"run"))
            main(args)
            root = Path(args.run)
            manifest = json.loads((root/"corpus/0001/manifest.json").read_text())
            episodes, rows, _ = load_corpus(root/"corpus/0001", manifest["identity"])
            self.assertTrue(all(e["winner"] == -1 and e["reason"] == "cap" for e in episodes))
            self.assertTrue(all(r["bootstrapped"] for r in rows))
            self.assertEqual(len(rows), 6)
            args.iterations = 0
            main(args)
            with (root/"corpus/0001/policies.npz").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                main(args)

    def test_finished_corpus_reused_after_interrupted_fit(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.args(run=str(Path(folder)/"run"))
            with patch("klent.fit", side_effect=RuntimeError("interrupted fit")):
                with self.assertRaisesRegex(RuntimeError, "interrupted fit"):
                    main(args)
            corpus = Path(args.run)/"corpus/0001/policies.npz"
            before = corpus.read_bytes()
            with patch("klent.collect", side_effect=AssertionError("must reuse corpus")):
                main(args)
            self.assertEqual(corpus.read_bytes(), before)
            self.assertTrue((Path(args.run)/"checkpoints/0001/klent.pt").exists())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_nonzero_q_matches_cpu_over_full_actions(self):
        game = Game([(0, 0), (8, 0)])
        self.addCleanup(game.close)
        model = Model()
        with torch.no_grad():
            model.q[2].weight.uniform_(-.05, .05)
            model.q[2].bias.fill_(.3)
        observation = observe(game)
        cpu = act(model, [observation], self.args())[0]
        gpu = act(copy.deepcopy(model).cuda(), [observation], self.args(device="cuda"))[0]
        np.testing.assert_allclose(gpu[0], cpu[0], rtol=1e-4, atol=1e-7)
        self.assertAlmostEqual(gpu[1], cpu[1], places=5)
        self.assertGreater(abs(gpu[1]), .01)

    def test_q_initialization_requires_matching_representation(self):
        from klent import SCHEMA
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = Model()
            with torch.no_grad():
                model.q[2].bias.fill_(.5)
            torch.save(model.nnue.state_dict(), root/"model.pt")
            payload = {"schema": SCHEMA, "model_sha256": hashlib.sha256((root/"model.pt").read_bytes()).hexdigest(),
                       "state": model.q.state_dict()}
            torch.save(payload, root/"q.pt")
            args = self.args(run=str(root/"good"), iterations=0, initial_model=str(root/"model.pt"), initial_q=str(root/"q.pt"))
            main(args)
            loaded = torch.load(root/"good/checkpoints/0000/klent.pt", weights_only=True)
            torch.testing.assert_close(loaded["model"]["q.2.bias"], model.q[2].bias)
            payload["model_sha256"] = "0"*64
            torch.save(payload, root/"q.pt")
            args.run = str(root/"bad")
            with self.assertRaisesRegex(ValueError, "matching model.pt"):
                main(args)


if __name__ == "__main__":
    unittest.main()
