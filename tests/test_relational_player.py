import unittest
from unittest.mock import Mock, patch
import numpy as np

from hexo import Game
from relational_player import RelationalPlayer


class DirectPlayer(unittest.TestCase):
    def actor(self, mode='pi'):
        actor = RelationalPlayer.__new__(RelationalPlayer)
        actor.mode, actor.model_sha256 = mode, 'test-model'
        actor.history, actor.tree, actor.prover = [], None, None
        actor.milliseconds, actor.alpha, actor.beta = 1000, .03, .1
        actor.simulations, actor.root_samples, actor.batch_size = 16, 8, 4
        actor.evaluator = Mock()
        return actor

    def test_sequential_full_legal_turn_does_not_mutate_input(self):
        actor = self.actor()
        def predict(histories):
            with_game = Game(histories[0])
            try:
                actions = np.asarray(with_game.legal_moves(), np.int64)
                wanted = (8,0) if len(histories[0]) == 1 else (16,0)
                scores = np.asarray([2 if tuple(p) == wanted else -2 for p in actions], np.float32)
                return [{'actions':actions, 'logits':scores, 'q':np.zeros(len(actions), np.float32)}]
            finally:
                with_game.close()
        actor.evaluator.evaluate.side_effect = predict
        game = Game([(0,0)])
        try:
            before = game.cells
            result = actor.turn(game)
            self.assertEqual(result['moves'], [[8,0],[16,0]])
            self.assertEqual(game.cells, before)
            self.assertEqual(actor.history, [(0,0),(8,0),(16,0)])
            actor.set_history([(0,0)])
            self.assertEqual(actor.history, [(0,0)])
        finally:
            game.close()

    def test_mu_uses_q_and_pi_without_native_search(self):
        game = Game([(0,0)])
        try:
            actions = np.asarray(game.legal_moves(), np.int64)
            logits, q = np.zeros(len(actions), np.float32), np.zeros(len(actions), np.float32)
            logits[0], q[1] = 2, 1
            actor = self.actor()
            actor.evaluator.evaluate.return_value = [{'actions':actions,'logits':logits,'q':q}]
            self.assertEqual(actor._reactive(game)[0], actions[0].tolist())
            actor.mode = 'mu'
            self.assertEqual(actor._reactive(game)[0], actions[1].tolist())
        finally:
            game.close()

    def test_gumbel_no_action_fails_without_policy_fallback(self):
        actor = self.actor('gumbel')
        actor.history = [(0,0)]
        actor.tree = Mock()
        actor.tree.search.return_value = {'action':None}
        game = Game([(0,0)])
        try:
            with patch.object(actor, 'set_history') as reset:
                with self.assertRaisesRegex(TimeoutError, 'no fallback'):
                    actor.turn(game)
                reset.assert_called_once_with([[0,0]])
            actor.evaluator.evaluate.assert_not_called()
            self.assertEqual(len(game.cells), 1)
        finally:
            game.close()

    def test_history_reset_discards_tree_and_cached_predictions(self):
        actor = self.actor('gumbel')
        actor.seed = 0
        actor.set_history([(0,0)])
        old_tree, old_cache = actor.tree, actor.cache
        old_cache.put('stale', 'prediction')
        try:
            actor.set_history([(0,0),(1,0)])
            self.assertFalse(old_tree.ptr)
            self.assertIsNot(actor.cache, old_cache)
            self.assertFalse(actor.cache.entries)
            self.assertEqual(actor.tree.history, [(0,0),(1,0)])
        finally:
            actor.close()

    def test_proof_mode_completes_mandatory_defense_when_solver_unknown(self):
        from tests.test_neural_search import Uniform
        from proof import _completions
        actor = self.actor('gumbel-proof')
        actor.seed, actor.evaluator = 0, Uniform()
        actor.prover = Mock()
        actor.prover.solve.return_value = actor.prover.history.return_value = {'status':'UNKNOWN'}
        history = [[0,0],[1,5],[3,3],[-2,2],[-1,1],[2,4],[0,6]]
        actor.set_history(history)
        game = Game(history)
        try:
            result = actor.turn(game)
            self.assertEqual(len(game.cells), len(history))
            for action in result['moves']:
                game.play(*action)
            self.assertEqual(len(result['moves']), 2)
            self.assertFalse(_completions({(q,r):p for q,r,p in game.cells}, 1, 2, lambda: None))
            self.assertEqual(result['proof_scope'], 'verified-root-and-tree-tactics')
            actor.prover.history.assert_called()
        finally:
            actor.close()
            game.close()

    def test_tree_immediate_win_reaches_turn_proof_status(self):
        from tests.test_neural_search import Uniform
        actor = self.actor('gumbel-proof')
        actor.seed, actor.evaluator = 0, Uniform()
        actor.prover = Mock()
        actor.prover.solve.return_value = actor.prover.history.return_value = {'status':'UNKNOWN'}
        history = [(0,0),(0,2),(1,2),(1,0),(2,0),(2,2),(3,2),(3,0),(4,0),(-2,2),(-3,2)]
        actor.set_history(history)
        game = Game(history)
        try:
            result = actor.turn(game)
            self.assertEqual(result['proof_status'], 'PROVEN_WIN')
            self.assertEqual(result['proof']['status'], 'UNKNOWN')
            self.assertEqual(len(game.cells), len(history))
            for action in result['moves']:
                game.play(*action)
            self.assertEqual(game.winner, 0)
        finally:
            actor.close()
            game.close()

    def test_search_that_uses_its_budget_still_completes_two_placements(self):
        actor = self.actor('gumbel')
        actor.history, actor.tree = [(0,0)], Mock()
        clock, actions = [0.], iter(([8,0], [16,0]))
        def search(**kwargs):
            clock[0] += kwargs['milliseconds']/1000
            return {'action':next(actions)}
        actor.tree.search.side_effect = search
        game = Game([(0,0)])
        try:
            with patch('relational_player.time.perf_counter', side_effect=lambda:clock[0]):
                result = actor.turn(game)
            self.assertEqual(result['moves'], [[8,0],[16,0]])
            self.assertEqual(result['elapsed_ms'], 1000)
            self.assertEqual(len(game.cells), 1)
        finally:
            game.close()

    def test_snapshot_identity_uses_copied_checkpoint(self):
        import hashlib
        import json
        from pathlib import Path
        import shutil
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from relational_evaluate import freeze
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, seal = root/'model.pt', root/'seal.dll'
            checkpoint.write_bytes(b'evaluated')
            seal.write_bytes(b'seal')
            (root/'seal_revision.txt').write_text('pinned')
            output = root/'evaluation'
            args = SimpleNamespace(output=output, checkpoint=checkpoint, seal_library=seal, mode='pi',
                games=2, seal_ms=100, neural_ms=1000, simulations=8, root_samples=4, batch_size=4, max_stones=80)
            original = shutil.copyfile
            def copy(source, target):
                original(source, target)
                if Path(target) == output/'models/candidate.pt':
                    checkpoint.write_bytes(b'replacement')
            def git_snapshot(command, **kwargs):
                self.assertFalse(output.exists())
                return 'source-revision' if command[1] == 'rev-parse' else ''
            with patch('relational_evaluate.shutil.copyfile', side_effect=copy), patch(
                    'relational_evaluate.subprocess.check_output', side_effect=git_snapshot):
                freeze(args)
            provenance = json.loads((output/'provenance.json').read_text())
            self.assertFalse(provenance['dirty'])
            identity = provenance['model_input_sha256']['candidate']
            self.assertEqual(identity, hashlib.sha256(b'evaluated').hexdigest())
            self.assertNotEqual(identity, hashlib.sha256(checkpoint.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
