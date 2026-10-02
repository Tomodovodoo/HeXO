"""Native neural tree checks independent of trained model quality."""
import importlib.util
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
    def test_puct_initial_value_uses_policy_weighted_action_values(self):
        from puct_search import PUCTSearch
        search = PUCTSearch(Uniform(), 'puct-weighted', [(0, 0)], cache=EvaluationCache(64))
        self.addCleanup(search.close)
        search.begin(1)
        request = search.request()
        prediction = search.evaluator.evaluate([request[0].history])[0]
        count = len(prediction['actions'])
        prediction['logits'][0] = np.log(9*(count-1))
        prediction['q'][:] = -1.
        prediction['q'][0] = 1.
        search.fulfill(request, prediction)
        self.assertAlmostEqual(search.root.value, .8)

    def test_puct_backup_keeps_sign_within_turn_and_flips_between_turns(self):
        from puct_search import PUCTSearch, search_many
        class Positive(Uniform):
            def evaluate(self, histories):
                return [dict(p, q=np.full(len(p['actions']), .8)) for p in super().evaluate(histories)]
        search = PUCTSearch(Positive(), 'puct-sign', [(0, 0)], cache=EvaluationCache(64))
        self.addCleanup(search.close)
        result = search_many([search], 1)[0]
        chosen = int(np.argmax(result['visits']))
        self.assertAlmostEqual(result['values'][chosen], .8)
        search.advance(result['action'])
        self.assertEqual(search.game.remaining, 1)
        result = search_many([search], 1)[0]
        chosen = int(np.argmax(result['visits']))
        self.assertAlmostEqual(result['values'][chosen], -.8)
        self.assertFalse(search.root.parents)

    def test_puct_unresolved_reply_prevents_loss_and_winning_reply_proves_node(self):
        from puct_search import Node
        root, loss, win = Node((), 1), Node((), 1, 0), Node((), 1, 1)
        root.expanded, root.eligible = True, np.ones(2, bool)
        root.children[0] = loss
        root.settle()
        self.assertEqual(root.winner, -1)
        self.assertFalse(root.eligible[0])
        root.children[1] = win
        root.settle()
        self.assertEqual(root.winner, 1)
        np.testing.assert_array_equal(root.eligible, [False, True])

    def test_puct_native_tactics_and_retained_complement(self):
        from puct_search import PUCTSearch, search_many
        history = [[0,0],[0,3],[1,3],[1,0],[2,0],[2,3],[3,3],[3,0],[7,4],[4,3],[5,4]]
        search = PUCTSearch(Uniform(), 'puct-tactics', history, cache=EvaluationCache(64), tactics=True, graph=True)
        self.addCleanup(search.close)
        result = search_many([search], 8)[0]
        self.assertEqual(result['proven'], 1)
        search.advance(result['action'])
        if search.game.winner < 0:
            result = search_many([search], 8)[0]
            self.assertEqual(result['proven'], 1)
            search.advance(result['action'])
        self.assertEqual(search.game.winner, 0)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Timed Bubble turns require torch')
    def test_timed_turn_reserves_simulations_for_both_stones(self):
        import threading
        from types import SimpleNamespace
        from timed_engine import dense_turn, legal_turn
        player = SimpleNamespace(evaluator=Uniform(), model_sha256='turn-cap', checkpoint='test',
                                 cache=EvaluationCache(64), prover=None, options=dict(search=True, solver=False))
        progress = []
        result = dense_turn(player, [(0, 0)], dict(normal_ms=1000, hard_ms=1500, reserve_ms=10, simulations=8),
                            threading.Event(), publish=lambda r: progress.append(dict(r)))
        self.addCleanup(player._timed_tree.close)
        counts = [r['completed'] for r in progress if r['completed']]
        self.assertGreaterEqual(len(counts), 2)
        self.assertLess(counts[0], counts[-1])
        self.assertLessEqual(result['completed'], 8)
        self.assertEqual(legal_turn([(0, 0)], result['moves']), result['moves'])

    def test_timed_interruption_does_not_select_from_a_partial_comparison(self):
        class Interrupted(Uniform):
            calls = 0
            def evaluate(self, histories):
                self.calls += 1
                return super().evaluate(histories)
        evaluator = Interrupted()
        search = NeuralSearch(evaluator, 'timed-comparison', [(0, 0)])
        self.addCleanup(search.close)
        result = search.search(128, root_samples=16, batch_size=1, anytime=True,
                               stop=lambda: evaluator.calls >= 2)
        self.assertIsNone(result['action'])
        self.assertGreater(result['completed'], 0)
        result = search.search(8, root_samples=4, batch_size=1, anytime=True)
        self.assertIn(tuple(result['action']), map(tuple, result['actions']))

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
        self.assertEqual(result['completed'], 0)
        self.assertEqual(result['proven'], -1)
        self.assertIsNotNone(result['action'])
        self.assertTrue(np.all(result['values'] == -1))
        self.assertEqual(len(result['actions']), 372)
        # A current-player completion takes precedence over mandatory defense.
        history = [[0,0],[0,3],[1,3],[1,0],[2,0],[2,3],[3,3],[3,0],[4,0],[4,3],[5,4]]
        search = NeuralSearch(Uniform(), 'exact-win', history, tactics=True)
        self.addCleanup(search.close)
        result = search.search(8)
        self.assertEqual(result['exact_winner'], 0)
        self.assertEqual(result['proof_status'], 'PROVEN_WIN')
        self.assertEqual(result['completed'], 0)
        self.assertEqual(result['proven'], 1)
        self.assertEqual(search.search(65536)['evaluated'], 0)
        search.advance(result['action'])
        game = Game(search.history)
        if game.winner < 0:
            game.close()
            search.advance(search.search(8)['action'])
            game = Game(search.history)
        self.assertEqual(game.winner, 0)
        game.close()

    def test_terminal_child_propagates_and_stops_without_tactics(self):
        history = [(0,0),(0,2),(1,2),(1,0),(2,0),(2,2),(3,2),(3,0),(4,0),(-2,2),(-3,2)]
        class Winning(Uniform):
            def evaluate(self, histories):
                predictions = super().evaluate(histories)
                for h, prediction in zip(histories, predictions):
                    g = Game(h)
                    for i, action in enumerate(prediction['actions']):
                        g.play(*map(int, action))
                        prediction['logits'][i] = 100 if g.winner >= 0 else -100
                        g.undo()
                    g.close()
                return predictions
        search = self.searcher(history, evaluator=Winning())
        result = search.search(65536, root_samples=2, batch_size=8)
        self.assertEqual((result['exact_winner'], result['completed'], result['evaluated']), (0, 1, 1))
        self.assertEqual(result['proven'], 1)
        self.assertEqual(np.count_nonzero(result['policy']), 1)
        search.advance(result['action'])
        self.assertIsNone(search.search(65536)['action'])

    def test_proven_root_waits_for_reserved_leaves_and_reuses_witness(self):
        search = self.searcher([(0,0)])
        self.assertTrue(native.hxg_begin(search.ptr, 64, 4))
        request, history = search.request()
        search.fulfill(request, Uniform().evaluate([history])[0])
        pending = [search.request() for _ in range(4)]
        actions = search.result(0, 0, 0, 0)['actions']
        # This entry represents a caller-verified proof; the synthetic outcome is not a game certificate.
        self.assertTrue(native.hxg_mark_exact(search.ptr, *map(int, actions[0]), 1, 1))
        self.assertFalse(native.hxg_done(search.ptr))
        self.assertEqual(search.request()[0], 0)
        for request, history in pending:
            search.fulfill(request, Uniform().evaluate([history])[0])
        self.assertTrue(native.hxg_done(search.ptr))
        result = search.search(65536)
        self.assertEqual((result['completed'], result['evaluated'], result['proven']), (0, 0, 1))
        self.assertEqual(result['action'], actions[0].tolist())

    def test_two_placement_terminal_proof_propagates_and_keeps_second_stone(self):
        history = [[0,0],[0,3],[1,3],[1,0],[2,0],[2,3],[3,3],[3,0],[7,4],[4,3],[5,4]]
        class Line(Uniform):
            def evaluate(self, histories):
                predictions = super().evaluate(histories)
                for h, prediction in zip(histories, predictions):
                    move = [-1,0] if len(h) == len(history) else [4,0]
                    prediction['logits'][:] = -100
                    prediction['logits'][(prediction['actions'] == move).all(axis=1)] = 100
                return predictions
        search = self.searcher(history, evaluator=Line())
        result = search.search(65536, root_samples=1, batch_size=8)
        self.assertEqual((result['action'], result['exact_winner'], result['completed'], result['evaluated']),
                         ([-1,0], 0, 2, 2))
        search.advance(result['action'])
        result = search.search(65536)
        self.assertEqual((result['action'], result['exact_winner'], result['completed'], result['evaluated']),
                         ([4,0], 0, 0, 0))
        search.advance(result['action'])
        self.assertEqual(search.search(65536)['exact_winner'], 0)

    def test_forced_block_policy_ignores_proven_losing_moves(self):
        # Player 1's four on r=3 is blocked at (0,3); player 0 has one placement left, so every move except
        # (5,3) and (6,3) is a proven loss. The better block must keep its full completed-Q margin.
        history = [(0,0),(1,3),(2,3),(0,3),(5,-3),(3,3),(4,3),(-3,-2)]
        better, worse = (5,3), (6,3)
        class Blocks(Uniform):
            def evaluate(self, histories):
                predictions = super().evaluate(histories)
                for h, prediction in zip(histories, predictions):
                    first = tuple(h[len(history)]) if len(h) > len(history) else None
                    prediction['q'][:] = -.1 if first == better else .1 if first == worse else 0.
                return predictions
        search = NeuralSearch(Blocks(), 'forced-block', history, 7, tactics=True)
        self.addCleanup(search.close)
        result = search.search(16, root_samples=16, batch_size=1)
        actions = result['actions'].tolist()
        good, bad = actions.index(list(better)), actions.index(list(worse))
        self.assertEqual(np.count_nonzero(result['policy']), 2)
        np.testing.assert_allclose([result['values'][good], result['values'][bad]], [.1, -.1])
        self.assertGreater(result['policy'][good], .99)
        self.assertEqual(result['action'], list(better))

    def test_refutation_two_opponent_placements_deep_proves_the_root_move_lost(self):
        # Player 1 holds two open threes. After a, the prior leads to c then d: two open fours, so every move of
        # player 0 is lost there. The other root moves are only estimated at -0.95; the refuted a must not be played.
        history = [(0,0),(0,4),(1,4),(5,-4),(-2,-3),(2,4),(-4,0),(4,-1),(6,-2),(-4,1),(-4,2),(7,-6)]
        a, c, d = (8,-8), (3,4), (-4,3)
        class Refutation(Uniform):
            def evaluate(self, histories):
                predictions = super().evaluate(histories)
                for h, prediction in zip(histories, predictions):
                    tail = [tuple(m) for m in h[len(history):]]
                    on_line = not tail or tail[0] == a
                    favourite = {0: a, 1: c, 2: d}.get(len(tail)) if on_line else None
                    if favourite is not None:
                        prediction['logits'][(prediction['actions'] == favourite).all(axis=1)] = 8.
                    prediction['q'][:] = 0. if on_line else .95
                return predictions
        search = NeuralSearch(Refutation(), 'refutation', history, 3, tactics=True)
        self.addCleanup(search.close)
        result = search.search(64, root_samples=16, batch_size=1)
        i = result['actions'].tolist().index(list(a))
        self.assertEqual((result['values'][i], result['policy'][i]), (-1., 0.))
        self.assertLessEqual(result['visits'][i], 3)
        self.assertNotEqual(result['action'], list(a))
        self.assertEqual(result['exact_winner'], -1)

    def test_exact_roots_report_the_shortest_win_and_the_longest_resistance(self):
        # Player 0 has five in a row with two placements left: either end wins with the first stone.
        history = [[0,0],[0,3],[1,3],[1,0],[2,0],[2,3],[3,3],[3,0],[4,0],[4,3],[5,4]]
        search = NeuralSearch(Uniform(), 'shortest', history, tactics=True)
        self.addCleanup(search.close)
        result = search.search(8)
        self.assertEqual((result['proven'], result['proof_plies']), (1, 1))
        self.assertEqual(sorted(result['proof_action']), [[-1,0],[5,0]])
        self.assertIn(result['action'], [[-1,0],[5,0]])
        self.assertEqual(np.count_nonzero(result['policy']), 2)
        # Player 1 has one placement left against player 0's five (open at (5,0)) and open four on r=3: every move
        # loses, and blocking the five is the only one that makes player 0 need two placements instead of one.
        history = [[0,0],[-1,0],[-1,1],[1,0],[2,0],[-2,5],[6,-3],[3,0],[4,0],[0,6],[7,-5],[0,3],[1,3],[-5,6],[8,-6],
                   [2,3],[3,3],[-6,-2]]
        search = NeuralSearch(Uniform(), 'longest', history, tactics=True)
        self.addCleanup(search.close)
        result = search.search(8)
        self.assertEqual((result['proven'], result['proof_plies']), (-1, 3))
        self.assertEqual(result['actions'][np.flatnonzero(result['policy'])].tolist(), [[5,0]])
        self.assertEqual(result['action'], [5,0])
    def test_graph_search_shares_both_orders_of_a_turn(self):
        # The prior favours the same two cells for either stone, so the tree expands the turn twice.
        history = [(0,0),(1,2),(2,1)]
        favoured = [(-1,-1),(-2,1)]
        class Favour(Uniform):
            def evaluate(self, histories):
                predictions = super().evaluate(histories)
                for prediction in predictions:
                    for cell in favoured:
                        prediction['logits'][(prediction['actions'] == cell).all(axis=1)] = 10.
                return predictions
        found = {}
        for graph in (False, True):
            search = NeuralSearch(Favour(), 'orders', history, 3, tactics=True, graph=graph)
            self.addCleanup(search.close)
            found[graph] = search.search(24, root_samples=2, batch_size=1), search.census()
        (tree, tree_census), (shared, graph_census) = found[False], found[True]
        self.assertGreater(tree_census['duplicates'], 0)
        self.assertEqual(graph_census['duplicates'], 0)
        expansions = lambda r: r['evaluated']+r['cache_hits']   # the tree's repeated turn is a cache hit
        self.assertLess(expansions(shared), expansions(tree))
        self.assertEqual(shared['completed'], 24)
        self.assertAlmostEqual(float(shared['policy'].sum()), 1.)

    def test_graph_search_proves_what_the_tree_proves(self):
        # Exact results do not depend on sharing: the proven-loss and proven-win roots of the tactics test.
        for history, winner in (([[0,0],[1,5],[3,3],[-2,2],[-1,1],[2,4],[0,6],[1,-1]], 1),
                                ([[0,0],[0,3],[1,3],[1,0],[2,0],[2,3],[3,3],[3,0],[4,0],[4,3],[5,4]], 0)):
            search = NeuralSearch(Uniform(), 'graph-exact', history, tactics=True, graph=True)
            self.addCleanup(search.close)
            result = search.search(8)
            self.assertEqual((result['exact_winner'], result['completed']), (winner, 0))

    def test_a_longer_certificate_keeps_the_shorter_tactical_win(self):
        # Player 0 completes six with one stone; a (stub-verified) certificate names a slower turn elsewhere.
        history = [[0,0],[0,3],[1,3],[1,0],[2,0],[2,3],[3,3],[3,0],[4,0],[4,3],[5,4]]
        class Slow:
            def history(self, history, ms, certificate=None, **kwargs):
                return dict(status='PROVEN_WIN', native_verified=True, moves=[[-3,-3],[-3,-2]], proof_turns=3)
        search = NeuralSearch(Uniform(), 'longer-certificate', history, tactics=True, proof_solver=Slow())
        self.addCleanup(search.close)
        self.assertTrue(native.hxg_begin(search.ptr, 8, 4))
        request, pending = search.request()
        self.assertTrue(search.fulfill_proof(request, pending, dict()))
        result = search.search(8)
        self.assertEqual((result['proven'], result['proof_plies']), (1, 1))
        self.assertIn(result['action'], [[-1,0],[5,0]])

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

    def test_cache_separates_dense_turn_context(self):
        import hexcrop
        histories = (
            ([(0,0),(1,0),(2,0),(0,1),(0,2),(3,0),(4,0)],
             [(0,0),(3,0),(4,0),(0,1),(0,2),(1,0),(2,0)], 7),
            ([(0,0),(1,0),(2,0),(0,1),(0,2),(3,0)],
             [(0,0),(3,0),(2,0),(0,1),(0,2),(1,0)], 6),
        )
        for first, second, plane in histories:
            with self.subTest(plane=plane):
                a, b = hexcrop.encode(first), hexcrop.encode(second)
                np.testing.assert_array_equal(a.actions, b.actions)
                np.testing.assert_array_equal(a.planes[:6], b.planes[:6])
                self.assertFalse(np.array_equal(a.planes[plane], b.planes[plane]))
                cache = EvaluationCache()
                cache.put(cache.key(first, 'dense'), 'prediction for first')
                self.assertIsNone(cache.get(cache.key(second, 'dense')))

    def test_cache_reuses_reordered_stones_with_same_dense_inputs(self):
        import hexcrop
        first = [(0,0),(1,0),(2,0),(0,1),(0,2),(3,0),(4,0)]
        second = [(0,0),(2,0),(1,0),(0,2),(0,1),(4,0),(3,0)]
        a, b = hexcrop.encode(first), hexcrop.encode(second)
        np.testing.assert_array_equal(a.planes, b.planes)
        cache = EvaluationCache()
        cache.put(cache.key(first, 'dense'), 'same prediction')
        self.assertEqual(cache.get(cache.key(second, 'dense')), 'same prediction')

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
