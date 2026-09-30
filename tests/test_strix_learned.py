import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

from hexo import Game
from tools.strix_learned_adapter import MODEL_SHA256, StrixLearned, mirror, validate_turn


class TurnValidation(unittest.TestCase):
    def game(self, moves):
        game = Game(moves)
        self.addCleanup(game.close)
        return game

    def test_sequential_radius_and_conditional_second(self):
        game = self.game([(0,0)])
        before = (game.key, game.state(), game.features())
        self.assertEqual(validate_turn(game, [[8,0],[16,0]]), [(8,0),(16,0)])
        self.assertEqual((game.key, game.state(), game.features()), before)
        game.play(8,0)
        self.assertEqual(game.remaining, 1)
        self.assertEqual(validate_turn(game, [[16,0]]), [(16,0)])

    def test_first_stone_win_and_invalid_turn_restore(self):
        game = self.game([(0,0),(0,3),(1,3),(1,0),(2,0),(3,3),(4,3),(3,0),(4,0),(5,3),(6,3)])
        before = (game.key, game.state(), game.features())
        self.assertEqual(validate_turn(game, [[5,0]]), [(5,0)])
        for moves in ([[5,0],[6,0]], [[7,1]], [[0,0]], [[True,3],[0,2]], [[1000,1000]]):
            with self.assertRaises(ValueError):
                validate_turn(game, moves)
            self.assertEqual((game.key, game.state(), game.features()), before)

    def test_wrong_checkpoint_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"wrong.safetensors"
            path.write_bytes(b"not the pinned model")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                StrixLearned(path)


class Frame(unittest.TestCase):
    def test_stones_and_moves_cross_into_strix_frame(self):
        opponent = object.__new__(StrixLearned)
        opponent.__dict__.update(timeout_ms=5000, lock=threading.Lock(), process=object(), simulations=2, actions=2,
                                 seed=0, calls=0, executable_sha256='x')
        sent = []
        def exchange(request, deadline):
            sent.append(request)
            return dict(status='OK', moves=[list(mirror(1, 1)), list(mirror(-2, 3))])
        opponent._exchange = exchange
        game = Game([(0, 0), (1, 0), (2, -1)])
        self.addCleanup(game.close)
        self.assertEqual(opponent(game, 100), [(1, 1), (-2, 3)])
        self.assertEqual(sent[0]['stones'], [[0, 0, 0], [1, 0, 1], [1, 1, 1]])
        self.assertEqual(mirror(*mirror(4, -7)), (4, -7))


class LearnedProcess(unittest.TestCase):
    def setUp(self):
        model = os.environ.get("HEXO_STRIX_PUBLIC_MODEL")
        if not model:
            self.skipTest("set HEXO_STRIX_PUBLIC_MODEL to the pinned external checkpoint")
        self.opponent = StrixLearned(model, simulations=2, actions=2)
        if not Path(self.opponent.executable).exists():
            self.skipTest("build tools/strix_learned first")
        self.addCleanup(self.opponent.close)
        self.opponent.warm_up()

    def test_loaded_relational_model_persistent_turns(self):
        metadata = self.opponent.metadata
        config = json.loads(metadata["loaded_metadata"]["model_config"])
        self.assertTrue(config["axis_relational"])
        self.assertEqual(config["axis_window"], 8)
        self.assertEqual(metadata["model_sha256"], MODEL_SHA256)
        self.assertEqual(metadata["loaded_metadata"]["train_steps"], "10")
        self.assertFalse(metadata["equal_wall_budget"])
        self.assertEqual(metadata["root_forcing"], dict(enabled=True, phases=[1,2],generator="wide",depth=6,nodes=2000))
        self.assertEqual(metadata["build_provenance"]["executable_sha256"], metadata["executable_sha256"])
        self.assertIn("Cargo.lock", metadata["build_provenance"]["wrapper_sha256"])
        pid = self.opponent.process.pid
        game = Game([(0,0),(1,0),(0,1)])
        self.addCleanup(game.close)
        before = (game.key, game.state(), game.features())
        moves = self.opponent(game, 100)
        self.assertEqual(len(moves), 2)
        self.assertGreater(self.opponent.last_result["eval_states"], 0)
        self.assertGreaterEqual(self.opponent.last_result["wall_ms"], self.opponent.last_result["search_ms"])
        self.assertEqual((game.key, game.state(), game.features()), before)
        game.play(*moves[0])
        moves = self.opponent(game, 100)
        self.assertEqual(len(moves), 1)
        self.assertEqual(self.opponent.process.pid, pid)

    def test_deadline_unknown_closes_worker_and_recovers(self):
        game = Game([(0,0),(1,0),(0,1)])
        self.addCleanup(game.close)
        key = game.key
        self.opponent.timeout_ms = .000001
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            self.opponent(game, 100)
        self.assertEqual(self.opponent.last_result["status"], "UNKNOWN")
        self.assertIsNone(self.opponent.process)
        self.assertIsNone(self.opponent.snapshot_directory)
        self.assertEqual(game.key, key)
        self.opponent.timeout_ms = 5000
        self.assertEqual(len(self.opponent(game, 100)), 2)

    def test_mismatched_build_report_stops_before_inference(self):
        self.opponent.close()
        self.opponent.build_provenance["executable_sha256"] = "0"*64
        with self.assertRaisesRegex(RuntimeError, "build provenance"):
            self.opponent.warm_up()
        self.assertIsNone(self.opponent.process)
        self.assertIsNone(self.opponent.snapshot_directory)


if __name__ == "__main__":
    unittest.main()
