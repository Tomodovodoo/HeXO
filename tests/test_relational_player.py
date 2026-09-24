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


if __name__ == '__main__':
    unittest.main()
