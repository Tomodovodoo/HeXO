"""Native neural tree checks independent of trained model quality."""
import unittest
import numpy as np
from hexo import Game
from neural_search import NeuralSearch, EvaluationCache, SearchCoordinator, native
from tests.reference import Reference

class Uniform:
    def evaluate(self, histories):
        out = []
        for history in histories:
            game = Game(history)
            actions = np.asarray(game.legal_moves(), dtype=np.int64)
            out.append(dict(actions=actions, logits=np.zeros(len(actions)), q=np.zeros(len(actions))))
            game.close()
        return out

class NeuralTree(unittest.TestCase):
    def test_unknown_proofs_reserve_inference_across_large_batches(self):
        from unittest.mock import patch
        clock, spent = [0.], [0.]
        class Unknown:
            def history(self, history, ms, **kwargs):
                spent[0] += ms
                clock[0] += ms/1000
                return {'status': 'UNKNOWN'}
        class Timed(Uniform):
            def evaluate(self, histories):
                clock[0] += .005
                return super().evaluate(histories)
        search = NeuralSearch(Timed(), 'proof-reserve', [(0,0)],
                              proof_solver=Unknown(), proof_ms=1000)
        self.addCleanup(search.close)
        with patch('neural_search.time.perf_counter', side_effect=lambda: clock[0]):
            result = search.search(32, root_samples=16, batch_size=16, milliseconds=100)
        self.assertLessEqual(spent[0], 25)
        self.assertGreater(result['evaluated'], 0)
        ref = Reference()
        ref.play(0,0)
        ref.play(*result['action'])

    def test_verified_pending_turn_and_rejected_certificate(self):
        from tactical_proof import NativeTactics
        try:
            solver = NativeTactics()
        except FileNotFoundError:
            self.skipTest('Build tools/tactical with tools/build_tactical.py first')
        history = [[0,0],[0,3],[1,3],[1,0],[2,0],[2,3],[3,3],[3,0],[7,4],[4,3],[5,4]]
        cert = dict(version=1, width='wide', root=0,
                    nodes=[dict(kind='immediate_win', action=[[-1,0],[4,0]])])
        search = NeuralSearch(Uniform(), 'verified-turn', history, proof_solver=solver, proof_ms=1000)
        self.addCleanup(search.close)
        self.assertTrue(native.hxg_begin(search.ptr, 8, 4))
        request, pending_history = search.request()
        bad = dict(version=1, width='wide', root=0,
                   nodes=[dict(kind='immediate_win', action=[[8,0],[9,0]])])
        self.assertFalse(search.fulfill_proof(request, pending_history, bad))
        self.assertEqual(native.hxg_exact(search.ptr), -1)
        changed = [list(p) for p in pending_history]
        changed[8] = [8,4]
        with self.assertRaisesRegex(ValueError, 'Proof history mismatch'):
            search.fulfill_proof(request, changed, cert)
        self.assertTrue(search.fulfill_proof(request, pending_history, cert))
        # The second certified placement survives tree advancement even if
        # later proof work is unavailable.
        search.proof_solver = None
        result = search.search(8)
        self.assertEqual(result['action'], [-1,0])
        self.assertEqual(result['exact_winner'], 0)
        search.advance(result['action'])
        result = search.search(8)
        self.assertEqual(result['action'], [4,0])
        self.assertEqual(result['exact_winner'], 0)
        search.advance(result['action'])
        game = Game(search.history)
        self.assertEqual(game.winner, 0)
        game.close()
        result = search.search(8)
        self.assertEqual(result['exact_winner'], 0)
        self.assertEqual(result['proof_status'], 'PROVEN_LOSS')
        self.assertIsNone(result['action'])

    def test_exact_defense_preserves_a_complete_turn(self):
        history = [[0,0],[1,5],[3,3],[-2,2],[-1,1],[2,4],[0,6]]
        good = [[-2,8],[-1,7],[4,2],[5,1]]
        from proof import _completions
        for seed in range(4):
            search = NeuralSearch(Uniform(), 'exact-defense', history, seed, tactics=True)
            self.addCleanup(search.close)
            result = search.search(8, root_samples=8)
            self.assertIn(result['action'], good)
            self.assertEqual(result['exact_winner'], -1)
            self.assertEqual(result['proof_status'], 'UNKNOWN')
            game = Game(history)
            self.assertEqual(result['actions'].tolist(), [list(p) for p in game.legal_moves()])
            game.close()
            self.assertEqual([a for a,p in zip(result['actions'].tolist(),result['policy']) if p>0], good)
            search.advance(result['action'])
            result = search.search(8, root_samples=8)
            search.advance(result['action'])
            game = Game(search.history)
            cells = {(q,r):p for q,r,p in game.cells}
            self.assertFalse(_completions(cells, 1, 2, lambda: None))
            game.close()

    def test_exact_loss_and_counterwin_keep_absolute_winner(self):
        history = [[0,0],[1,5],[3,3],[-2,2],[-1,1],[2,4],[0,6],[1,-1]]
        search = NeuralSearch(Uniform(), 'exact-loss', history, tactics=True)
        self.addCleanup(search.close)
        result = search.search(8)
        self.assertEqual(result['exact_winner'], 1)
        self.assertEqual(result['proof_status'], 'PROVEN_LOSS')
        self.assertTrue(np.all(result['values'] == -1))
        self.assertEqual(len(result['actions']), 372)
        # A current-player completion takes precedence over mandatory defense.
        history = [[0,0],[0,3],[1,3],[1,0],[2,0],[2,3],[3,3],[3,0],[4,0],[4,3],[5,4]]
        search = NeuralSearch(Uniform(), 'exact-win', history, tactics=True)
        self.addCleanup(search.close)
        result = search.search(8)
        self.assertEqual(result['exact_winner'], 0)
        self.assertEqual(result['proof_status'], 'PROVEN_WIN')
        search.advance(result['action'])
        game = Game(search.history)
        if game.winner < 0:
            game.close()
            search.advance(search.search(8)['action'])
            game = Game(search.history)
        self.assertEqual(game.winner, 0)
        game.close()

    def searcher(self, history=(), seed=7, evaluator=None):
        search = NeuralSearch(evaluator or Uniform(), 'test-v1', history, seed)
        self.addCleanup(search.close)
        return search

    def test_without_replacement_and_full_legal_support(self):
        search = self.searcher([(0, 0)])
        result = search.search(8, root_samples=8, batch_size=8)
        self.assertEqual(len(result['actions']), 216)
        self.assertEqual(np.count_nonzero(result['visits']), 8)
        self.assertEqual(result['visits'].max(), 1)
        self.assertEqual(result['completed'], 8)
        self.assertEqual(result['proof_status'], 'UNKNOWN')

    def test_reuse_across_both_phases_and_opponent(self):
        search = self.searcher([(0, 0)])
        for remaining in (2, 1, 2):
            game = Game(search.history)
            self.assertEqual(game.remaining, remaining)
            game.close()
            result = search.search(32, root_samples=4)
            self.assertGreaterEqual(result['visits'].sum(), 32)
            search.advance(result['action'])
            retained = native.hxg_stats(search.ptr, None, None, None, None)
            self.assertGreater(retained, 0)

    def test_new_budget_after_reuse_and_batch_equivalence(self):
        one, many = self.searcher([(0, 0)]), self.searcher([(0, 0)])
        a = one.search(32, root_samples=4, batch_size=1)
        b = many.search(32, root_samples=4, batch_size=8)
        np.testing.assert_array_equal(a['visits'], b['visits'])
        self.assertEqual(a['action'], b['action'])
        one.advance(a['action'])
        before = native.hxg_stats(one.ptr, None, None, None, None)
        self.assertGreater(before, 0)
        c = one.search(17, root_samples=4)
        self.assertEqual(c['completed'], 17)
        retained = int(c['visits'].sum())
        d = one.search(13, root_samples=4)
        self.assertEqual(d['completed'], 13)
        self.assertEqual(int(d['visits'].sum()), retained + 13)

    def test_reserved_leaf_and_cancel(self):
        search = self.searcher([(0, 0)])
        self.assertTrue(native.hxg_begin(search.ptr, 8, 4))
        request, history = search.request()
        self.assertGreater(request, 0)
        self.assertEqual(search.request()[0], 0)
        native.hxg_cancel(search.ptr)
        new, _ = search.request()
        self.assertGreater(new, request)
        search.fulfill(new, Uniform().evaluate([history])[0])
        pending = []
        for _ in range(4):
            request, history = search.request()
            self.assertGreater(request, 0)
            pending.append(tuple(map(tuple, history)))
        self.assertEqual(len(set(pending)), 4)
        native.hxg_cancel(search.ptr)

    def test_advance_rejects_fractional_coordinates(self):
        search = self.searcher()
        with self.assertRaises(ValueError):
            search.advance((0.5, 0))
        self.assertEqual(search.history, [])

    def test_exact_cache_phase_color_and_version(self):
        cache = EvaluationCache()
        a = cache.key([(0, 0), (1, 0), (2, 0)], 'a')
        self.assertEqual(a, cache.key([(0, 0), (2, 0), (1, 0)], 'a'))
        self.assertNotEqual(a, cache.key([(0, 0), (2, 0), (1, 0)], 'b'))
        self.assertNotEqual(a, cache.key([(0, 0), (1, 0)], 'a'))

    def test_first_placement_win_stops_and_terminal_is_exact(self):
        history = [(0,0),(0,2),(1,2),(1,0),(2,0),(2,2),(3,2),(3,0),(4,0),(-2,2),(-3,2)]
        game = Game(history)
        winning = []
        for action in game.legal_moves():
            game.play(*action)
            if game.winner >= 0:
                winning.append(action)
            game.undo()
        game.close()
        self.assertTrue(winning)
        class Tactical(Uniform):
            def evaluate(self, histories):
                result = super().evaluate(histories)
                for h, prediction in zip(histories, result):
                    g = Game(h)
                    for i, action in enumerate(prediction['actions']):
                        g.play(*map(int, action))
                        prediction['logits'][i] = 100 if g.winner >= 0 else 0
                        g.undo()
                    g.close()
                return result
        search = self.searcher(history, evaluator=Tactical())
        result = search.search(16, root_samples=2)
        self.assertIn(tuple(result['action']), map(tuple, winning))
        index = np.where(np.all(result['actions'] == result['action'], axis=1))[0][0]
        self.assertEqual(result['values'][index], 1)
        search.advance(result['action'])
        self.assertIsNone(search.search(4)['action'])

    def test_exhaustive_small_payoff_tree_with_opponent_branches(self):
        # Independent minimax table: A loses against R; B guarantees +0.3.
        # The evaluator deliberately overrates A before the opponent replies.
        A, B, C, R, S = (1,0), (0,1), (2,0), (0,2), (-1,2)
        class Payoff:
            def __init__(self):
                self.seen = []
            def evaluate(self, histories):
                predictions = []
                for history in histories:
                    h = list(map(tuple, history))
                    self.seen.append(h)
                    game = Game(h)
                    actions = np.asarray(game.legal_moves(), dtype=np.int64)
                    chosen = [A,B] if len(h)==1 else [C] if len(h)==2 else [R,S] if len(h)==3 else [tuple(actions[0])]
                    logits = np.array([0. if tuple(a) in chosen else -100. for a in actions])
                    if len(h)<=2:
                        value = .8 if len(h)==2 and h[1]==A else .3
                    elif len(h)==3:
                        value = -.8 if h[1]==A else -.3
                    else:
                        payoff = (-1 if h[3]==R else 1) if h[1]==A else .3
                        value = payoff if game.player==1 else -payoff
                    predictions.append(dict(actions=actions, logits=logits, q=np.full(len(actions), value)))
                    game.close()
                return predictions
        evaluator = Payoff()
        search = self.searcher([(0,0)], seed=8, evaluator=evaluator)
        result = search.search(128, root_samples=2, batch_size=1)
        self.assertEqual(tuple(result['action']), B)
        by_action = dict(zip(map(tuple, result['actions']), result['values']))
        self.assertLess(by_action[A], 0)
        self.assertAlmostEqual(by_action[B], .3)
        for reply in (R,S):
            self.assertTrue(any(len(h)>=4 and h[1:4]==[A,C,reply] for h in evaluator.seen))

    def test_corrupt_evaluator_coordinates_rejected_before_cast(self):
        search = self.searcher([(0, 0)])
        native.hxg_begin(search.ptr, 4, 2)
        request, history = search.request()
        original = Uniform().evaluate([history])[0]
        for dtype, value in ((np.float64, -7.5), (np.uint64, 2**64-1)):
            prediction = {k: v.copy() for k,v in original.items()}
            prediction['actions'] = prediction['actions'].astype(dtype)
            prediction['actions'][0, 0] = value
            with self.assertRaises(ValueError):
                search.fulfill(request, prediction)
        native.hxg_cancel(search.ptr)

    def test_cached_work_respects_deadline_during_gather(self):
        from unittest.mock import patch
        source = self.searcher([(0, 0)], seed=7)
        source.search(16, root_samples=4, batch_size=1)
        search = self.searcher([(0, 0)], seed=7)
        search.cache = source.cache
        ticks = iter(np.arange(0, 10, .0005))
        with patch('neural_search.time.perf_counter', side_effect=lambda: float(next(ticks))):
            result = search.search(16, root_samples=4, batch_size=16, milliseconds=3)
        self.assertLess(result['completed'], 16)
        self.assertGreater(result['cache_hits'], 0)
        self.assertEqual(result['evaluated'], 0)
        # Cancelled reservations must permit another full search budget.
        self.assertEqual(search.search(4, root_samples=2)['completed'], 4)

    def test_multiple_trees_share_batches_and_keep_visits_local(self):
        class Recording(Uniform):
            def __init__(self):
                self.batches = []
            def evaluate(self, histories):
                self.batches.append([list(map(tuple, h)) for h in histories])
                return super().evaluate(histories)
        evaluator = Recording()
        first = self.searcher([(0,0)], seed=1, evaluator=evaluator)
        second = self.searcher([(0,0),(1,0)], seed=2, evaluator=evaluator)
        coordinator = SearchCoordinator(evaluator, 'test-v1')
        result = coordinator.search_many([first, second], simulations=[16,24], root_samples=4, batch_size=8)
        self.assertEqual([r['completed'] for r in result], [16,24])
        self.assertEqual([int(r['visits'].sum()) for r in result], [16,24])
        self.assertGreater(coordinator.last_stats['largest_batch'], 1)
        self.assertTrue(any(any(len(h)==1 for h in batch) and any(len(h)==2 for h in batch) for batch in evaluator.batches))
        for tree, report in zip((first, second), result):
            tree.advance(report['action'])
        again = coordinator.search_many([first, second], simulations=7, root_samples=2)
        self.assertEqual([r['completed'] for r in again], [7,7])

    def test_coordinator_deduplicates_evaluation_not_visits(self):
        evaluator = Uniform()
        trees = [self.searcher([(0,0)], seed=5, evaluator=evaluator) for _ in range(2)]
        coordinator = SearchCoordinator(evaluator, 'test-v1')
        results = coordinator.search_many(trees, simulations=4, root_samples=4, batch_size=8)
        self.assertEqual([int(r['visits'].sum()) for r in results], [4,4])
        self.assertLess(coordinator.last_stats['unique_positions'], sum(r['evaluated'] for r in results))
        with self.assertRaises(ValueError):
            coordinator.search_many([trees[0],trees[0]])
        trees[1].model_version = 'different'
        with self.assertRaises(ValueError):
            coordinator.search_many(trees)

    def test_one_tree_deadline_does_not_cancel_other_tree(self):
        from unittest.mock import patch
        evaluator = Uniform()
        trees = [self.searcher([(0,0)], evaluator=evaluator) for _ in range(2)]
        coordinator = SearchCoordinator(evaluator, 'test-v1')
        ticks = iter(np.arange(0, 10, .0001))
        with patch('neural_search.time.perf_counter', side_effect=lambda: float(next(ticks))):
            results = coordinator.search_many(trees, simulations=4, milliseconds=[.01,None])
        self.assertEqual(results[0]['completed'], 0)
        self.assertIsNone(results[0]['action'])
        self.assertEqual(results[1]['completed'], 4)
        self.assertEqual(coordinator.search_many([trees[0]], simulations=4)[0]['completed'], 4)

    def test_failed_batch_cancels_every_tree(self):
        class Broken(Uniform):
            def evaluate(self, histories):
                raise RuntimeError('inference failed')
        evaluator = Broken()
        trees = [self.searcher([(0,0)], evaluator=evaluator) for _ in range(2)]
        coordinator = SearchCoordinator(evaluator, 'test-v1')
        with self.assertRaises(RuntimeError):
            coordinator.search_many(trees, simulations=4)
        evaluator.evaluate = Uniform().evaluate
        self.assertEqual([r['completed'] for r in coordinator.search_many(trees, simulations=4)], [4,4])

    def test_wrong_legal_order_rejected(self):
        search = self.searcher([(0, 0)])
        native.hxg_begin(search.ptr, 4, 2)
        request, history = search.request()
        prediction = Uniform().evaluate([history])[0]
        prediction['actions'] = prediction['actions'][::-1]
        with self.assertRaises(ValueError):
            search.fulfill(request, prediction)
        native.hxg_cancel(search.ptr)

if __name__ == '__main__':
    unittest.main()
