"""Native neural tree checks independent of trained model quality."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from hexo import Game
from neural_search import NeuralSearch, EvaluationCache, GameGraph, Recheck, SearchCoordinator, native
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

class Spread(Uniform):
    """Uniform logits. Below the root [(0, 0)] every position values the root mover's first stone (q, r) at
    scale * sin(7.3q + 3.1r) for that mover, so the root's Q spread is about 2 * scale."""
    def __init__(self, scale):
        self.scale = scale

    def evaluate(self, histories):
        out = []
        for h, p in zip(histories, super().evaluate(histories)):
            value = self.scale*np.sin(7.3*h[1][0]+3.1*h[1][1]) if len(h) > 1 else 0.
            out.append(dict(p, q=np.full(len(p['actions']), value if (len(h)+1)//2 % 2 == 1 else -value)))
        return out

class Ranked(Uniform):
    """Logit -2k for the k-th legal move in native order, so a move's prior rank is its index."""
    def evaluate(self, histories):
        return [dict(p, logits=-2.*np.arange(len(p['actions']))) for p in super().evaluate(histories)]

def recorded_position(ply=12):
    """The first `ply` placements of the first game in tests/fixtures/actor_selfplay.npz."""
    moves = np.load(Path(__file__).parent/'fixtures'/'actor_selfplay.npz')['moves']
    return [tuple(map(int, m)) for m in moves[:ply]]

class NeuralTree(unittest.TestCase):
    def test_root_noise_samples_beyond_the_prior_and_leaves_the_target(self):
        history = recorded_position()
        def tree(**options):
            search = NeuralSearch(Ranked(), 'root-noise', history, seed=11, **options)
            self.addCleanup(search.close)
            return search
        plain, zero, noisy = tree().search(64, root_samples=8, batch_size=8), \
            tree(root_noise=0.).search(64, root_samples=8, batch_size=8), tree(root_noise=.25)
        for key in ('action', 'completed'):
            self.assertEqual(zero[key], plain[key])
        for key in ('visits', 'policy', 'scores', 'values'):
            np.testing.assert_array_equal(zero[key], plain[key])
        self.assertTrue((np.flatnonzero(plain['visits']) < 8).all())
        result = noisy.search(64, root_samples=8, batch_size=8)
        sampled = np.flatnonzero(result['visits'])
        self.assertEqual(len(sampled), 8)
        self.assertTrue((sampled >= 8).any())
        native.hxg_root_noise(noisy.ptr, 0.)
        np.testing.assert_array_equal(noisy.result(0, 0, 0, 0)['policy'], result['policy'])
        unsampled = np.flatnonzero(result['visits'][:100] == 0)
        np.testing.assert_allclose(result['policy'][unsampled]/result['policy'][unsampled[0]],
                                   np.exp(-2.*(unsampled-unsampled[0])), rtol=1e-9)
        given = tree().search(64, root_samples=8, batch_size=8, root_noise=.25)
        np.testing.assert_array_equal(given['visits'], result['visits'])
        for bad in (-.1, 1., float('nan')):
            with self.assertRaisesRegex(ValueError, 'Invalid root noise'):
                NeuralSearch(Uniform(), 'root-noise', root_noise=bad)

    def test_q_range_floor_flattens_only_small_spreads(self):
        def entropy(p):
            return float(-(p[p > 0]*np.log(p[p > 0])).sum())
        for scale in (.9, .05):
            search = NeuralSearch(Spread(scale), 'q-range-floor', [(0, 0)], seed=3)
            self.addCleanup(search.close)
            plain = search.search(64, root_samples=8, batch_size=8)
            values = plain['values'][plain['visits'] > 0]
            native.hxg_q_range_floor(search.ptr, .5)
            floored = search.result(0, 0, 0, 0)['policy']
            if scale > .5:
                self.assertGreater(values.max()-values.min(), .5)
                np.testing.assert_array_equal(floored, plain['policy'])
            else:
                self.assertGreater(entropy(floored), entropy(plain['policy'])+.1)
        with self.assertRaisesRegex(ValueError, 'Invalid Q range floor'):
            NeuralSearch(Uniform(), 'q-range-floor', q_range_floor=-1.)
        built, given = (NeuralSearch(Spread(.05), 'q-range-floor', [(0, 0)], seed=3, q_range_floor=floor) for floor in (.5, 0.))
        self.addCleanup(built.close)
        self.addCleanup(given.close)
        np.testing.assert_array_equal(built.search(64, root_samples=8, batch_size=8)['policy'],
                                      given.search(64, root_samples=8, batch_size=8, q_range_floor=.5)['policy'])

    def test_play_defaults_to_policy_and_actor_result_retains_gumbel(self):
        search = NeuralSearch(Uniform(), 'choice-default', [(0, 0)], seed=0)
        self.addCleanup(search.close)
        result = search.search(32, root_samples=4, batch_size=4)
        policy_action = result['actions'][result['policy'].argmax()].tolist()
        self.assertEqual(result['action'], policy_action)
        self.assertFalse(np.isfinite(result['scores'][result['policy'].argmax()]))
        actor = search.result(0, 0, 0, 0)
        self.assertEqual(actor['action'], actor['actions'][actor['scores'].argmax()].tolist())
        self.assertNotEqual(actor['action'], result['action'])
        with self.assertRaisesRegex(ValueError, 'choice must be'):
            search.search(32, choice='invalid')

    def test_explicit_gumbel_and_timed_fallback_keep_the_selected_mode(self):
        from unittest.mock import patch
        for choice in ('policy', 'gumbel'):
            search = NeuralSearch(Uniform(), 'timed-choice', [(0, 0)], seed=0)
            self.addCleanup(search.close)
            result = search.search(32, root_samples=4, batch_size=4, choice=choice)
            ranking = result['policy'] if choice == 'policy' else result['scores']
            self.assertEqual(result['action'], result['actions'][ranking.argmax()].tolist())
            with patch.object(search, 'result', wraps=search.result) as snapshot:
                result = search.search(32, root_samples=4, batch_size=1, anytime=True,
                                       stop=lambda: snapshot.call_count > 0, choice=choice)
            ranking = result['policy'] if choice == 'policy' else result['scores']
            self.assertEqual(result['action'], result['actions'][ranking.argmax()].tolist())
            self.assertLess(result['completed'], 32)

    def test_puct_initial_value_uses_policy_weighted_action_values(self):
        from puct_search import PUCTSearch
        search = PUCTSearch(Uniform(), 'puct-weighted', [(0, 0)])
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
        verified = solver.history(pending_history, ms=1000, certificate=cert)
        self.assertEqual(verified['status'], 'PROVEN_WIN')
        for attacker in ('opponent', None):
            self.assertFalse(search._install_verified_proof(request, pending_history, dict(verified, attacker=attacker)))
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
                return dict(status='PROVEN_WIN', native_verified=True, attacker='mover',
                            moves=[[-3,-3],[-3,-2]], proof_turns=3)
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
            result = search.search(32, root_samples=4, choice='gumbel')
            self.assertGreaterEqual(result['visits'].sum(), 32)
            search.advance(result['action'])
            retained = native.hxg_stats(search.ptr, None, None, None, None)
            self.assertGreater(retained, 0)

    def test_new_budget_after_reuse_and_batch_equivalence(self):
        one, many = self.searcher([(0, 0)]), self.searcher([(0, 0)])
        a = one.search(32, root_samples=4, batch_size=1, choice='gumbel')
        b = many.search(32, root_samples=4, batch_size=8, choice='gumbel')
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

class Refuted(Ranked):
    """Ranked logits. A position below `line` (a strict extension of it) is lost for `mover`: value -0.9 for that
    side to move, 0.9 for the other; every other position is worth 0. `line` None refutes nothing."""
    def __init__(self, mover, line=None):
        self.mover, self.line = mover, line

    def evaluate(self, histories):
        out = []
        for h, p in zip(histories, super().evaluate(histories)):
            below = self.line is not None and len(h) > len(self.line) and \
                [tuple(c) for c in h[:len(self.line)]] == [tuple(c) for c in self.line]
            value = (-.9 if (len(h)+1)//2 % 2 == self.mover else .9) if below else 0.
            out.append(dict(p, q=np.full(len(p['actions']), value)))
        return out

class SharedGraph(unittest.TestCase):
    def test_views_share_evidence_and_keep_comparison_credits_local(self):
        graph = self.graph(Uniform(), recorded_position(11))
        graph.search(16, root_samples=4, batch_size=4)
        own = graph.counters()
        own_credits = graph.credits().copy()
        before = graph.result(0, 0, 0, 0)['visits'].sum()
        view = graph.view(seed=19)
        self.addCleanup(view.close)
        view.search(32, root_samples=8, batch_size=8)
        self.assertEqual(graph.counters(), dict(own, views=2))
        np.testing.assert_array_equal(graph.credits(), own_credits)
        self.assertEqual(int(view.credits().sum()), 32)
        self.assertGreater(graph.result(0, 0, 0, 0)['visits'].sum(), before)
        self.assertEqual(view.counters()['completed'], 32)
        self.assertEqual(view.counters()['pending'], 0)

    def test_polling_cold_view_metrics_does_not_change_its_first_choice(self):
        graph = self.graph(Uniform(), recorded_position(11))
        native.hxg_begin(graph.ptr, 8, 4)
        request, history = graph.request()
        graph.fulfill(request, Uniform().evaluate([history])[0])
        polled, plain = graph.view(seed=11), graph.view(seed=11)
        self.addCleanup(polled.close)
        self.addCleanup(plain.close)
        for _ in range(4):
            polled.result(0, 0, 0, 0)
            self.assertEqual(int(polled.credits().sum()), 0)
        native.hxg_begin(polled.ptr, 8, 4)
        native.hxg_begin(plain.ptr, 8, 4)
        request, chosen = polled.request()
        polled.fulfill(request, Uniform().evaluate([chosen])[0])
        request, unpolled = plain.request()
        self.assertGreater(request, 0)
        self.assertEqual(chosen[len(graph.history)], unpolled[len(graph.history)])
        native.hxg_cancel(plain.ptr)

    def test_view_survives_source_close_and_other_view_can_move_with_pending_work(self):
        graph = self.graph(Uniform(), recorded_position(11))
        graph.search(16, root_samples=4, batch_size=4)
        view = graph.view(seed=12)
        self.addCleanup(view.close)
        native.hxg_begin(graph.ptr, 8, 4)
        request, history = graph.request()
        self.assertGreater(request, 0)
        view.at(history)
        with self.assertRaisesRegex(ValueError, 'pending requests'):
            graph.at(history)
        graph.close()
        result = view.search(16, root_samples=4, batch_size=4)
        self.assertEqual(result['completed'], 16)
        self.assertEqual(view.counters()['views'], 1)
        self.assertEqual(view.counters()['pending'], 0)

    def test_view_proof_retires_late_prediction_without_changing_exact_value(self):
        graph = self.graph(Uniform(), recorded_position(11))
        graph.search(16, root_samples=4, batch_size=4)
        native.hxg_begin(graph.ptr, 8, 4)
        request, history = graph.request()
        self.assertGreater(request, 0)
        proof = graph.view(history, seed=9)
        self.addCleanup(proof.close)
        game = Game(history)
        winner = 1-game.player
        game.close()
        # A caller-verified loss arrives from another producer while this leaf is on the GPU.
        self.assertTrue(native.hxg_prove_loss(proof.ptr, winner, 7))
        graph.fulfill(request, Uniform().evaluate([history])[0])
        self.assertEqual(native.hxg_exact(proof.ptr), winner)
        self.assertEqual(native.hxg_value(proof.ptr), -1.)
        self.assertEqual(graph.counters()['retired'], 1)
        self.assertEqual(graph.counters()['pending'], 0)
        self.assertEqual(graph.counters()['completed'], 1)
        self.assertEqual(int(graph.credits().sum()), 1)
        self.assertNotEqual(graph.request()[0], -2)
        native.hxg_cancel(graph.ptr)

    def test_view_sessions_resume_without_inheriting_another_roots_credits(self):
        history = recorded_position(11)
        graph = self.graph(Uniform(), history)
        graph.search(16, root_samples=4, batch_size=4)
        credits = graph.credits().copy()
        counters = graph.counters()
        graph.at([*history, tuple(graph.result(0, 0, 0, 0)['action'])])
        self.assertEqual(int(graph.credits().sum()), 0)
        graph.search(8, root_samples=4, batch_size=4)
        graph.at(history)
        np.testing.assert_array_equal(graph.credits(), credits)
        self.assertEqual(graph.counters(), counters)

    def test_eviction_pins_other_views_and_exact_roots_need_no_prediction(self):
        graph = self.graph(Uniform(), recorded_position(11), limit=1)
        graph.search(16, root_samples=4, batch_size=4)
        child = [*graph.history, tuple(graph.result(0, 0, 0, 0)['action'])]
        view = graph.view(child, seed=4)
        self.addCleanup(view.close)
        view.search(8, root_samples=4, batch_size=4)
        graph.at(graph.history)
        self.assertGreaterEqual(graph.store()['expanded'], 2)
        self.assertEqual(view.search(8, root_samples=4, batch_size=4)['completed'], 8)
        fresh = graph.view([*child, tuple(view.result(0, 0, 0, 0)['action'])], seed=2)
        self.addCleanup(fresh.close)
        game = Game(fresh.history)
        winner = 1-game.player
        game.close()
        self.assertTrue(native.hxg_prove_loss(fresh.ptr, winner, 9))
        native.hxg_begin(fresh.ptr, 32, 8)
        self.assertEqual(fresh.request()[0], 0)
        self.assertTrue(native.hxg_done(fresh.ptr))
        self.assertEqual(fresh.counters()['pending'], 0)
        self.assertEqual(fresh.counters()['issued'], 0)
        result = fresh.result(0, 0, 0, 0)
        self.assertTrue(np.isfinite(result['policy']).all())
        self.assertAlmostEqual(float(result['policy'].sum()), 1.)

    def test_live_views_prevent_resetting_shared_tables(self):
        graph = self.graph(Uniform(), recorded_position(11))
        view = graph.view(seed=2)
        self.addCleanup(view.close)
        self.assertEqual(native.hxg_graph(graph.ptr, 0), 0)
        self.assertEqual(graph.counters()['views'], 2)

    def test_late_certificate_tightens_materialized_leaf_without_duplicate_edges(self):
        history = [(0,0),(0,3),(1,3),(1,0),(2,0),(2,3),(3,3),(3,0),(7,4),(4,3),(5,4)]
        parent = GameGraph(Uniform(), 'late-certificate', history, seed=3)
        self.addCleanup(parent.close)
        native.hxg_begin(parent.ptr, 8, 4)
        parent_request, _ = parent.request()
        child_history = [*history, (-1,0)]
        child = parent.view(child_history, seed=4)
        self.addCleanup(child.close)
        native.hxg_begin(child.ptr, 8, 4)
        child_request, _ = child.request()
        # These turns really complete six; the first bound is deliberately loose.
        game = Game(history)
        game.play(-1,0)
        game.play(4,0)
        self.assertEqual(game.winner, 0)
        game.close()
        turn = np.array([[-1,0],[4,0]], np.int64)
        self.assertTrue(native.hxg_prove(parent.ptr, parent_request, np.array(history, np.int64), len(history),
                                        0, 2, turn, 2, 2))
        before = child.result(0, 0, 0, 0)
        self.assertEqual(native.hxg_distance(child.ptr), 5)
        self.assertTrue(native.hxg_prove(child.ptr, child_request, np.array(child_history, np.int64),
                                        len(child_history), 0, 1, np.array([[4,0]], np.int64), 1, 1))
        after = child.result(0, 0, 0, 0)
        self.assertEqual(len(after['actions']), len(before['actions']))
        self.assertEqual(len({tuple(p) for p in after['actions']}), len(after['actions']))
        self.assertEqual(native.hxg_distance(child.ptr), 1)
        self.assertEqual(native.hxg_distance(parent.ptr), 2)
        self.assertEqual(child.counters()['pending'], 0)
        self.assertEqual(child.counters()['retired'], 1)
        self.assertEqual(child.request()[0], 0)
        self.assertAlmostEqual(float(after['policy'].sum()), 1.)

    def graph(self, evaluator, history, **options):
        graph = GameGraph(evaluator, 'game-graph', history, seed=5, tactics=True, **options)
        self.addCleanup(graph.close)
        return graph

    def edge(self, result, action):
        return result['actions'].tolist().index(list(action))

    def test_a_later_search_reaches_the_earlier_root(self):
        # A is a turn start (two stones to place). Its first search picks the turn A -> B -> C; a deeper search at C
        # finds C lost for A's mover, and A's edge to B carries C's visits and value without searching A again.
        a = recorded_position(11)
        evaluator = Refuted(Game(a).player)
        graph = self.graph(evaluator, a)
        first = graph.search(64, root_samples=8, batch_size=8)
        c = graph.after_turn(first)
        self.assertEqual(len(c), len(a)+2)
        before = first['values'][self.edge(first, c[len(a)])]
        evaluator.line = c
        graph.at(c)
        deep = graph.search(2048, root_samples=16, batch_size=64)
        self.assertEqual(deep['completed'], 2048)
        self.assertGreater(float(deep['policy'] @ deep['values']), .8)
        graph.at(c[:-1])
        b = graph.result(0, 0, 0, 0)
        self.assertGreaterEqual(b['visits'][self.edge(b, c[-1])], 2048)
        graph.at(a)
        after = graph.result(0, 0, 0, 0)
        i = self.edge(after, c[len(a)])
        self.assertGreaterEqual(after['visits'][i], 2048)
        self.assertLess(after['values'][i], before-.5)
        self.assertLess(after['values'][i], -.8)
        again = graph.search(32, root_samples=8, batch_size=8)
        self.assertNotEqual(again['action'], list(c[len(a)]))

    def test_returning_to_a_position_reads_and_resumes_the_deeper_branch(self):
        # A -> B -> A: B, A's favourite, turns out lost for A's mover once B is searched as a root. Back at A the
        # statistics under B already hold that, A stops preferring B, and A's next search continues its counts.
        a = recorded_position(11)
        evaluator = Refuted(Game(a).player)
        graph = self.graph(evaluator, a)
        first = graph.search(64, root_samples=8, batch_size=8)
        i = self.edge(first, first['action'])
        self.assertGreater(first['policy'][i], .5)
        b = [*a, tuple(first['action'])]
        evaluator.line = b
        graph.at(b)
        graph.search(1024, root_samples=16, batch_size=32)
        graph.at(a)
        back = graph.result(0, 0, 0, 0)
        self.assertLess(back['completed_q'][i], first['completed_q'][i]-.5)
        self.assertLess(back['policy'][i], .5)
        self.assertGreaterEqual(back['visits'][i], 1024)
        again = graph.search(32, root_samples=8, batch_size=8)
        self.assertNotEqual(again['action'], first['action'])
        self.assertEqual(int(again['visits'].sum()), int(back['visits'].sum())+32)

    def test_a_proof_marked_at_a_later_root_reaches_the_earlier_one(self):
        a = recorded_position(11)
        mover = Game(a).player
        graph = self.graph(Ranked(), a)
        first = graph.search(32, root_samples=4, batch_size=4)
        b = [*a, tuple(first['action'])]
        graph.at(b)
        second = graph.search(16, root_samples=4, batch_size=4)
        graph.mark(tuple(second['actions'][0]), mover, 5)
        graph.at(a)
        after = graph.result(0, 0, 0, 0)
        self.assertEqual((after['exact_winner'], after['proven']), (mover, 1))
        self.assertEqual(after['values'][self.edge(after, b[-1])], 1.)

    def test_losing_first_stone_is_refuted_in_either_order_without_pair_expansion(self):
        # P1's open four is unstoppable after P0 spends A away from it.
        root = [(0,0),(1,2),(2,2),(0,-2),(-2,0),(3,2),(4,2)]
        a, b = (0,-3), (-3,0)
        for materialized in (False, True):
            with self.subTest(materialized=materialized):
                graph = GameGraph(Uniform(), 'half-turn-permutation', root, tactics=False)
                self.addCleanup(graph.close)
                graph.search(1, root_samples=1)
                graph.at(root + [b])
                graph.search(1, root_samples=1)
                graph.at(root)
                if materialized:
                    graph.at(root + [a])
                    graph.search(1, root_samples=1)
                    graph.prove_loss(1, 3)
                else:
                    graph.mark(a, 1, 4)
                graph.at(root + [b])
                result = graph.result(0,0,0,0)
                edge = self.edge(result, a)
                self.assertEqual((result['values'][edge], result['policy'][edge]), (-1., 0.))
                self.assertEqual(result['proof_status'], 'UNKNOWN')
                self.assertIn(dict(history=[list(p) for p in root + [b,a]], winner=1, plies=2), graph.facts())
                graph.search(16, root_samples=4)
                self.assertEqual(graph.result(0,0,0,0)['visits'][edge], 0)

    def test_the_same_seed_and_budgets_choose_the_same_moves(self):
        a = recorded_position(11)
        def run():
            graph = self.graph(Ranked(), a)
            first = graph.search(32, root_samples=8, batch_size=8)
            graph.at(graph.after_turn(first))
            deep = graph.search(64, root_samples=8, batch_size=8)
            graph.at(a)
            return [first['action'], deep['action'], graph.search(16, root_samples=8, batch_size=8)['action']]
        self.assertEqual(run(), run())

    def test_eviction_keeps_proofs_and_a_proven_root_is_not_searched_again(self):
        a = recorded_position(11)
        mover = Game(a).player
        graph = self.graph(Ranked(), a, limit=1)
        first = graph.search(32, root_samples=4, batch_size=4)
        b = [*a, tuple(first['action'])]
        graph.at(b)
        second = graph.search(16, root_samples=4, batch_size=4)
        graph.mark(tuple(second['actions'][0]), mover, 5)
        graph.at(a)   # evicts every expanded node but the root
        self.assertEqual(graph.store()['expanded'], 1)
        graph.at(b)
        again = graph.search(16, root_samples=4, batch_size=4)
        self.assertEqual((again['exact_winner'], again['completed']), (mover, 0))
        graph.at(a)
        self.assertEqual(graph.search(16, root_samples=4, batch_size=4)['completed'], 0)

    def test_the_store_survives_advances_and_keeps_its_bound(self):
        a = recorded_position(11)
        graph = self.graph(Uniform(), a, limit=48)
        first = graph.search(128, root_samples=8, batch_size=8)
        self.assertGreater(graph.store()['expanded'], 48)
        graph.advance(first['action'])
        graph.at(a)
        np.testing.assert_array_equal(graph.result(0, 0, 0, 0)['visits'], first['visits'])
        store = graph.store()
        self.assertLessEqual(store['expanded'], 48)
        self.assertGreater(store['evicted'], 0)
        graph.search(8, root_samples=4, batch_size=4)
        self.assertLessEqual(graph.store()['expanded'], 48+8)
        self.assertEqual(int(graph.result(0, 0, 0, 0)['visits'].sum()), int(first['visits'].sum())+8)
        unbounded = self.graph(Uniform(), a, limit=0)
        unbounded.search(128, root_samples=8, batch_size=8)
        unbounded.search(8, root_samples=4, batch_size=4)
        self.assertEqual(unbounded.store()['evicted'], 0)

    def test_a_search_counts_toward_the_order_of_its_own_history(self):
        # Both orders of A's turn reach C; a search at C reached by one order counts at that order's first stone only.
        a = recorded_position(11)
        graph = self.graph(Ranked(), a)
        first = graph.search(16, root_samples=4, batch_size=4)
        x, y = (tuple(first['actions'][i]) for i in range(2))
        for stone in (x, y):
            graph.at([*a, stone])
            graph.search(8, root_samples=4, batch_size=4)
        graph.at(a)
        before = graph.result(0, 0, 0, 0)['visits']
        graph.at([*a, y, x])
        graph.search(64, root_samples=8, batch_size=8)
        graph.at(a)
        after = graph.result(0, 0, 0, 0)['visits']
        self.assertIn((after - before)[self.edge(first, y)], (64, 65))   # 65 when the search expanded C itself
        self.assertEqual((after - before)[self.edge(first, x)], 0)
        # A graph built at C stores its prefixes unexpanded, without edges to credit, so A's edges hold only A's own
        # playouts; C reaches A through its value alone.
        late = self.graph(Ranked(), [*a, y, x])
        late.search(64, root_samples=8, batch_size=8)
        late.at(a)
        self.assertEqual(int(late.search(8, root_samples=4, batch_size=4)['visits'].sum()), 8)
        self.assertEqual(late.root_version, len(a)+2+1)

    def test_an_evicted_child_hands_its_statistics_to_the_next_one(self):
        a = recorded_position(11)
        class Varied(Ranked):
            def evaluate(self, histories):
                return [dict(p, q=np.full(len(p['actions']), .8*np.sin(1.3*h[-1][0]+.7*h[-1][1])))
                        for h, p in zip(histories, super().evaluate(histories))]
        graph = self.graph(Varied(), a, limit=1)
        first = graph.search(256, root_samples=8, batch_size=8)
        graph.at(a)
        self.assertEqual(graph.store()['expanded'], 1)
        kept = graph.result(0, 0, 0, 0)
        i = int(np.argmax(kept['visits']))
        stone, visits, value = tuple(kept['actions'][i]), int(kept['visits'][i]), float(kept['values'][i])
        np.testing.assert_array_equal(kept['visits'], first['visits'])
        graph.at([*a, stone])
        graph.search(1, root_samples=1, batch_size=1)
        graph.at(a)
        after = graph.result(0, 0, 0, 0)
        self.assertGreater(int(after['visits'][i]), visits)
        self.assertLess(abs(float(after['values'][i])-value), 2/(visits+1))
        # An exact edge keeps its visits as well when a node is created for it again.
        j = int(np.argsort(kept['visits'])[-2])
        stone, visits = tuple(kept['actions'][j]), int(after['visits'][j])
        graph.mark(stone, Game(a).player, 3)
        graph.at([*a, stone])
        graph.at(a)
        self.assertEqual(int(graph.result(0, 0, 0, 0)['visits'][j]), visits)
        graph.search(64, root_samples=8, batch_size=8)
        graph.at(a)
        store = graph.store()
        self.assertGreater(store['evicted'], 4)
        self.assertLessEqual(store['summaries'], 4)   # four times the limit of one
        self.assertLessEqual(store['outcomes'], max(16, store['nodes']))

    def test_the_pv_check_searches_again_only_after_a_drop(self):
        a = recorded_position(11)
        mover = Game(a).player
        quiet = self.graph(Refuted(mover), a).search(64, root_samples=8, batch_size=8, pv_check=.25)
        self.assertFalse(quiet['pv_check']['searched'])
        self.assertEqual(quiet['completed'], 48)
        evaluator = Refuted(mover)
        graph = self.graph(evaluator, a)
        check = Recheck(graph, 64, .25)
        self.assertEqual((check.budget, check.reserve), (32, 16))
        first = graph.search(check.budget, root_samples=8, batch_size=8)
        evaluator.line = graph.after_turn(first)
        spent, budget = [first['completed']], check.step(first)
        self.assertEqual(graph.history, evaluator.line)
        while budget:
            result = graph.search(budget, root_samples=8, batch_size=8)
            spent.append(result['completed'])
            budget = check.step(result)
        summary = check.summary()
        self.assertEqual(graph.history, a)
        self.assertTrue(summary['searched'])
        self.assertLess(summary['after'], summary['before']-.1)
        self.assertEqual(spent, [32, 16, 16])
        with self.assertRaisesRegex(ValueError, 'pv_check must lie'):
            Recheck(graph, 64, .5)
        # A chosen first stone never searched has no known second stone: no check.
        unvisited = graph.result(0, 0, 0, 0)
        unvisited.update(action=unvisited['actions'][int(np.argmin(unvisited['visits']))].tolist(), proven=0)
        self.assertIsNone(graph.after_turn(unvisited))
        late = Recheck(graph, 64, .25)
        late.step(graph.search(late.budget, root_samples=8, batch_size=8))
        self.assertNotEqual(graph.history, a)
        late.abandon()   # out of time before the check's search
        self.assertEqual((graph.history, late.summary()['searched'], late.step(None)), (a, False, 0))

class NativeScheduler(unittest.TestCase):
    def graph(self, history=((0, 0),), version='scheduler'):
        graph = GameGraph(Uniform(), version, history, limit=96)
        self.addCleanup(graph.close)
        return graph

    def pool(self, graphs, **options):
        from native_scheduler import SearchPool
        pool = SearchPool(graphs, **options)
        # These checks launch no GPU, so every taken row is already fenced.
        self.addCleanup(lambda: pool._ptr and (pool.abandon_fenced(), pool.close()))
        return pool

    def answer(self, pool, batch=None):
        batch = batch if batch is not None else pool.feed.take(128)
        if batch is None:
            return 0
        ids, leaves = batch
        predictions = Uniform().evaluate([h.tolist() for _, _, h in leaves])
        pool.feed.install(ids, [(p['actions'], p['logits'], p['q']) for p in predictions])
        return len(ids)

    def finish(self, pool):
        for _ in range(1000):
            pool.step()
            self.answer(pool)
            if pool.done():
                return
        self.fail('A bounded search did not complete')

    def test_adaptive_views_preserve_legal_coverage_and_credit_provenance(self):
        pool = self.pool([self.graph()], quantum=16, views=8, depth=4, work=256)
        self.finish(pool)
        stats = pool.games[0].stats()
        self.assertGreater(stats['created'], 1)
        self.assertGreater(stats['depth'], 0)
        self.assertLessEqual(stats['completed'], 256)
        self.assertEqual(stats['issued'], stats['completed']+stats['cancelled'])
        self.assertEqual((stats['pending'], pool.feed.stats()['pending_rows']), (0, 0))
        evidence = pool.games[0].evidence()
        game = Game([(0, 0)])
        try:
            self.assertEqual(evidence['actions'].tolist(), [list(a) for a in game.legal_moves()])
            self.assertTrue(evidence['eligible'].all())
            self.assertIn(pool.games[0].choice(), evidence['actions'].tolist())
        finally:
            game.close()
        self.assertEqual(int(evidence['lifetime_credits'].sum()), stats['root_completed'])
        records = pool.games[0].records()
        self.assertTrue(any(r['depth'] > 0 for r in records))
        for row in records:
            self.assertEqual(len(row['history']), 1+row['depth'])
            self.assertEqual(row['exact_winner'], -1)
            self.assertEqual(row['root_estimate'], 0.)
            self.assertEqual(row['raw_value'], 0.)

    def test_global_queue_coalesces_games_without_sharing_their_sampling_credits(self):
        pool = self.pool([self.graph(), self.graph()], quantum=16, views=1, work=16)
        pool.step()
        self.assertEqual(pool.feed.stats()['joined'], 1)
        self.assertEqual(self.answer(pool), 1)
        self.finish(pool)
        self.assertEqual(pool.stats()['failed'], 0)
        for game in pool.games:
            stats = game.stats()
            self.assertEqual(stats['completed'], 16)
            self.assertEqual(int(game.evidence()['lifetime_credits'].sum()), 16)

    def test_source_closure_is_safe_and_external_search_cannot_steal_the_graph(self):
        from native_scheduler import SearchPool
        source = self.graph()
        pool = self.pool([source], quantum=16, work=32)
        with self.assertRaisesRegex(ValueError, 'already has a native owner'):
            SearchPool([source], work=32)
        self.assertFalse(native.hxg_begin(source.ptr, 16, 8))
        self.assertIn(b'native scheduler', native.hxg_error())
        with self.assertRaisesRegex(ValueError, 'native scheduler'):
            source.at([(0, 0), (1, 0)])
        source.close()
        self.finish(pool)
        self.assertIsNotNone(pool.games[0].choice())
        pool.close()
        with self.assertRaisesRegex(ValueError, 'closed'):
            pool.games[0].stats()

    def test_retarget_cancels_old_subscribers_but_keeps_bound_record_histories(self):
        source = self.graph()
        pool = self.pool([source], quantum=16, views=2, work=64)
        pool.step()
        old = pool.feed.take(128)
        pool.retarget(0, [(0, 0), (1, 0), (2, 0)], work=32)
        self.answer(pool, old)
        self.assertEqual(pool.games[0].stats()['completed'], 0)
        self.finish(pool)
        self.assertEqual(pool.games[0].history(), [[0, 0], [1, 0], [2, 0]])
        records = pool.games[0].records()
        self.assertTrue(any(len(r['history']) == 1 for r in records))
        self.assertTrue(any(r['history'][:3] == [[0, 0], [1, 0], [2, 0]] for r in records))
        self.assertEqual(pool.feed.stats()['pending_requests'], 0)

    def test_teardown_refuses_submitted_work_until_fenced_abandon(self):
        pool = self.pool([self.graph()], quantum=16, views=2, work=64)
        pool.step()
        batch = pool.feed.take(128)
        self.assertIsNotNone(batch)
        with self.assertRaisesRegex(ValueError, 'Drain or fenced-abandon'):
            pool.close()
        self.assertEqual(pool.feed.stats()['pending_requests'], 0)
        # No GPU was launched in this check, so the batch is already fenced.
        pool.abandon_fenced()
        self.assertEqual(pool.feed.stats()['pending_rows'], 0)
        pool.close()

    def test_failed_prediction_stops_its_game_while_other_games_finish(self):
        pool = self.pool([self.graph(), self.graph([(0, 0), (1, 0), (2, 0)])],
                         quantum=16, views=2, work=32)
        pool.step()
        ids, leaves = pool.feed.take(128)
        self.assertEqual(len(ids), 2)
        prediction = Uniform().evaluate([leaves[1][2].tolist()])[0]
        pool.feed.install(ids, [None, (prediction['actions'], prediction['logits'], prediction['q'])])
        self.finish(pool)
        self.assertEqual(pool.stats()['failed'], 1)
        self.assertEqual(pool.games[1].stats()['completed'], 32)
        self.assertEqual(pool.feed.stats()['pending_rows'], 0)

    def test_common_clock_stops_admission_and_retarget_can_resume(self):
        import time
        pool = self.pool([self.graph(), self.graph()], quantum=16, views=2, work=64)
        pool.clock(.1)
        time.sleep(.002)
        self.assertFalse(pool.admit())
        self.assertTrue(pool.done())
        self.assertTrue(all(g.stats()['deadline'] for g in pool.games))
        self.assertTrue(all(g.records()[0]['root_estimate'] is None for g in pool.games))
        pool.retarget(0, [(0, 0)], work=16)
        self.finish(pool)
        self.assertEqual(pool.games[0].stats()['completed'], 16)

    def test_model_and_illegal_retarget_rejection_do_not_change_existing_search(self):
        from native_scheduler import SearchPool
        a, b = self.graph(version='a'), self.graph(version='b')
        with self.assertRaisesRegex(ValueError, 'one fixed model'):
            SearchPool([a, b])
        pool = self.pool([a], quantum=16, work=16)
        with self.assertRaisesRegex(ValueError, 'Illegal retarget'):
            pool.retarget(0, [(0, 0), (0, 0)], work=16)
        self.assertEqual(pool.games[0].history(), [[0, 0]])
        self.finish(pool)

    def test_terminal_focus_stops_without_starving_an_active_game(self):
        terminal = [(0,0),(0,3),(1,3),(1,0),(2,0),(2,3),(3,3),(3,0),(7,4),(4,3),(5,4),(-1,0),(4,0)]
        pool = self.pool([self.graph(terminal), self.graph()], quantum=16, views=2, work=32)
        self.finish(pool)
        self.assertEqual(pool.games[0].stats()['completed'], 0)
        self.assertIsNone(pool.games[0].choice())
        self.assertEqual(pool.games[1].stats()['completed'], 32)

    def test_cancel_last_game_reports_completion_without_another_step(self):
        pool = self.pool([self.graph(), self.graph()], quantum=16, views=2, work=32)
        pool.step()
        self.assertFalse(pool.done())
        pool.cancel(game=0)
        self.assertFalse(pool.done())
        self.assertEqual(pool.stats()['active'], 1)
        pool.cancel(game=1)
        self.assertTrue(pool.done())
        self.assertEqual((pool.stats()['active'], pool.feed.stats()['pending_requests']), (0, 0))
        pool.retarget(0, [(0, 0)], work=16)
        self.assertFalse(pool.done())
        self.finish(pool)

    def test_six_hundred_games_share_one_queue_and_cancel_without_expansion(self):
        graphs = [self.graph() for _ in range(600)]
        pool = self.pool(graphs, quantum=4, views=1, work=4)
        pool.step()
        self.assertEqual(pool.stats()['games'], 600)
        self.assertEqual(pool.feed.stats()['new_rows'], 1)
        self.assertEqual(pool.feed.stats()['joined'], 599)
        pool.cancel()
        self.assertEqual((pool.feed.stats()['pending_rows'], pool.feed.stats()['pending_requests']), (0, 0))

    def test_ready_watermark_returns_before_walking_every_game_and_resumes_fairly(self):
        game = Game([(0, 0)])
        try:
            histories = [[(0, 0), action] for action in game.legal_moves()[:24]]
        finally:
            game.close()
        pool = self.pool([self.graph(h) for h in histories], quantum=4, views=1, work=4)
        pool.limit_ready(4)
        pool.step()
        self.assertEqual(pool.feed.queued(), 4)
        self.assertEqual(pool.feed.stats()['new_rows'], 4)
        pool.step()
        self.assertEqual(pool.feed.stats()['new_rows'], 4)
        self.finish(pool)
        self.assertTrue(all(g.stats()['completed'] == 4 for g in pool.games))
        self.assertEqual(pool.feed.queued(), 0)

    def test_watermark_does_not_split_a_tree_visit_layer(self):
        pool = self.pool([self.graph()], quantum=16, views=1, work=32)
        pool.limit_ready(1)
        pool.step()
        self.assertEqual(pool.feed.queued(), 1)
        self.answer(pool)
        pool.step()
        # Finish gathering the layer even when it exceeds the queue watermark.
        self.assertGreater(pool.feed.queued(), 1)
        self.assertLessEqual(pool.feed.queued(), 16)
        self.finish(pool)
        self.assertEqual(pool.games[0].stats()['completed'], 32)

    def test_bounded_supply_keeps_deeper_views_and_sampling_credits_separate(self):
        pool = self.pool([self.graph(), self.graph([(0, 0), (1, 0), (2, 0)])],
                         quantum=16, views=4, depth=4, work=128)
        pool.limit_ready(2)
        self.finish(pool)
        for game in pool.games:
            stats = game.stats()
            self.assertGreater(stats['depth'], 0)
            self.assertEqual(stats['issued'], stats['completed']+stats['cancelled'])
            self.assertEqual(int(game.evidence()['lifetime_credits'].sum()), stats['root_completed'])
        self.assertEqual((pool.feed.queued(), pool.feed.stats()['pending_requests']), (0, 0))

    def test_ready_counter_excludes_subscribers_and_submitted_rows_during_cancel(self):
        pool = self.pool([self.graph(), self.graph()], quantum=16, views=1, work=16)
        pool.step()
        self.assertEqual((pool.feed.queued(), pool.feed.stats()['pending_requests']), (1, 2))
        batch = pool.feed.take(1)
        self.assertEqual((pool.feed.queued(), pool.feed.stats()['pending_rows']), (0, 1))
        pool.cancel(game=0)
        self.answer(pool, batch)
        pool.step()
        self.assertGreater(pool.feed.queued(), 0)
        pool.cancel()
        self.assertEqual((pool.feed.queued(), pool.feed.stats()['pending_rows']), (0, 0))

    def test_full_ready_queue_still_services_the_clock_without_admission_calls(self):
        import time
        pool = self.pool([self.graph(), self.graph()], quantum=16, views=1, work=64)
        pool.limit_ready(1)
        pool.clock(100.)
        pool.step()
        self.assertEqual(pool.feed.queued(), 1)
        time.sleep(.12)
        pool.step()
        self.assertTrue(pool.done())
        self.assertTrue(all(g.stats()['deadline'] for g in pool.games))
        self.assertEqual((pool.feed.queued(), pool.feed.stats()['pending_requests']), (0, 0))

    def test_full_ready_queue_retires_a_cold_root_proved_by_another_producer(self):
        graph = self.graph()
        pool = self.pool([graph], quantum=16, views=1, work=64)
        pool.limit_ready(1)
        pool.step()
        self.assertEqual(pool.feed.queued(), 1)
        # Caller-verified defender evidence reaches the shared cold node.
        self.assertTrue(native.hxg_prove_loss(graph.ptr, 0, 7))
        pool.step()
        self.assertTrue(pool.done())
        self.assertEqual((pool.feed.queued(), pool.feed.stats()['pending_requests']), (0, 0))
        self.assertEqual(pool.games[0].records()[-1]['root_estimate'], -1.)
        game = Game([(0, 0)])
        try:
            self.assertEqual(pool.games[0].evidence()['actions'].tolist(),
                             [list(a) for a in game.legal_moves()])
        finally:
            game.close()

    def test_parallel_games_share_predictions_and_keep_each_roots_credits(self):
        pool = self.pool([self.graph() for _ in range(24)], quantum=16, views=4,
                         depth=4, work=96, workers=4, cache=2)
        pool.limit_ready(16)
        pool.step()
        self.assertEqual((pool.feed.stats()['new_rows'], pool.feed.stats()['joined']), (1, 23))
        self.finish(pool)
        for game in pool.games:
            stats = game.stats()
            self.assertEqual(stats['completed'], 96)
            self.assertEqual(stats['issued'], stats['completed']+stats['cancelled'])
            self.assertEqual(int(game.evidence()['lifetime_credits'].sum()), stats['root_completed'])
            self.assertGreater(stats['depth'], 0)
        self.assertEqual((pool.feed.queued(), pool.feed.stats()['pending_requests']), (0, 0))

    def test_parallel_watermark_admits_cursor_first_and_serves_all_games(self):
        game = Game([(0, 0)])
        try:
            histories = [[(0, 0), action] for action in game.legal_moves()[:24]]
        finally:
            game.close()
        for workers in (2, 4, 8):
            with self.subTest(workers=workers):
                pool = self.pool([self.graph(h) for h in histories], quantum=4,
                                 views=1, work=4, workers=workers)
                pool.limit_ready(1)
                pool.step()
                # Earlier cursor admission cannot be stolen by a faster worker.
                self.assertGreater(pool.games[0].stats()['pending'], 0)
                self.finish(pool)
                self.assertTrue(all(g.stats()['completed'] == 4 for g in pool.games))
                self.assertEqual(pool.feed.queued(), 0)
                pool.close()

    def test_parallel_queue_cancellation_and_retarget_ignore_late_subscribers(self):
        pool = self.pool([self.graph() for _ in range(12)], quantum=16, views=4,
                         work=64, workers=4)
        pool.step()
        late = pool.feed.take(128)
        pool.cancel(game=0)
        pool.retarget(1, [(0, 0), (1, 0), (2, 0)], work=32)
        self.answer(pool, late)
        self.assertEqual(pool.games[1].stats()['completed'], 0)
        self.finish(pool)
        self.assertEqual(pool.games[0].stats()['completed'], 0)
        self.assertEqual(pool.games[1].stats()['completed'], 32)
        self.assertEqual((pool.feed.stats()['pending_rows'], pool.feed.stats()['pending_requests']), (0, 0))

    def test_invalid_host_worker_configuration_releases_graph_ownership(self):
        from native_scheduler import SearchPool
        graph = self.graph()
        for workers in (0, 17):
            with self.assertRaisesRegex(ValueError, 'host worker count'):
                SearchPool([graph], work=16, workers=workers)
        pool = self.pool([graph], quantum=16, work=16, workers=2)
        self.finish(pool)
        self.assertEqual(pool.games[0].stats()['completed'], 16)

    def test_parallel_install_error_joins_workers_and_releases_games(self):
        graphs = [self.graph() for _ in range(12)]
        pool = self.pool(graphs, quantum=16, work=32, workers=4)
        pool.step()
        ids, leaves = pool.feed.take(128)
        predictions = Uniform().evaluate([h.tolist() for _, _, h in leaves])
        with self.assertRaisesRegex(ValueError, 'Incomplete legal actions'):
            pool.feed.install(ids, [(p['actions'][:-1], p['logits'][:-1], p['q'][:-1])
                                    for p in predictions])
        pool.abandon_fenced()
        pool.close()
        # The failed phase returned every graph before caller-side teardown.
        recovered = self.pool(graphs, quantum=16, work=16, workers=4)
        self.finish(recovered)
        self.assertTrue(all(g.stats()['completed'] == 16 for g in recovered.games))

    def test_unexpanded_exact_child_records_the_proof_instead_of_an_unset_mean(self):
        from neural_search import checked
        history = [(0,0),(0,3),(1,3),(1,0),(2,0),(2,3),(3,3),(3,0),(7,4),(4,3),(5,4)]
        graph = self.graph(history)
        child = graph.view(history+[(-1,0)])
        self.addCleanup(child.close)
        checked(native.hxg_begin(graph.ptr, 16, 8))
        request, _ = graph.request()
        # This complete same-turn witness makes six, with no intervening defence.
        witness = np.asarray([[-1,0],[4,0]], np.int64)
        checked(native.hxg_prove(graph.ptr, request, np.asarray(history, np.int64), len(history),
                                 0, 2, witness, 2, 1))
        self.assertEqual(native.hxg_exact(child.ptr), 0)
        pool = self.pool([child], quantum=16, views=1, work=16)
        pool.cancel()
        record = pool.games[0].records()[0]
        self.assertEqual((record['exact_winner'], record['root_estimate']), (0, 1.))
        self.assertIsNone(record['raw_value'])


class NativeProofs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tactical_proof import library
        if not library().is_file():
            raise unittest.SkipTest('Build tools/tactical with tools/build_tactical.py first')

    graph = NativeScheduler.graph
    pool = NativeScheduler.pool
    answer = NativeScheduler.answer
    opening = [[0,0],[0,8],[2,8],[1,0],[2,0],[4,8],[6,8]]

    def loop(self, pool, **options):
        return pool.enable_proofs(slice_ms=1000, table_mb=1, **options)

    def wait(self, predicate, seconds=5):
        import time
        end = time.monotonic()+seconds
        while not predicate():
            if time.monotonic() >= end:
                self.fail('Native proof work did not finish')
            time.sleep(.002)

    def test_parallel_host_phases_deliver_proofs_without_cross_game_credits(self):
        from tactical_proof import independent_verify
        pool = self.pool([self.graph(self.opening) for _ in range(12)], quantum=16,
                         views=4, work=4096, workers=4)
        proofs = self.loop(pool, workers=2, queue=8)
        for _ in range(1000):
            pool.step()
            self.answer(pool)
            if pool.done():
                break
        self.assertTrue(pool.done())
        proofs.drain()
        self.assertTrue(all(native.hxg_exact(native.hxgo_root(game.ptr)) == 0 for game in pool.games))
        self.assertEqual((pool.feed.stats()['pending_rows'], pool.feed.stats()['pending_requests']), (0, 0))
        records = proofs.records()
        self.assertEqual({r['game'] for r in records}, set(range(12)))
        for row in records:
            self.assertEqual(independent_verify(row['result']['certificate'], row['request']['history'],
                             attacker=row['result']['attacker'], known=row['request']['known']), row['result']['status'])
        for game in pool.games:
            stats = game.stats()
            self.assertEqual(stats['issued'], stats['completed']+stats['cancelled'])
            self.assertEqual(int(game.evidence()['lifetime_credits'].sum()), stats['root_completed'])

    def test_workers_consume_queued_jobs_without_owner_polling(self):
        from tactical_proof import independent_verify
        graphs = [self.graph(self.opening) for _ in range(4)]
        pool = self.pool(graphs, quantum=16, views=1, work=4096)
        proofs = self.loop(pool, workers=2, queue=8)
        pool.step()
        self.assertEqual(self.answer(pool), 1)  # Identical neural contexts coalesce across games.
        proofs.step()
        submitted = proofs.stats()['submitted']
        self.assertEqual(submitted, 4)
        # stats() neither admits work nor installs results. Workers consume all four jobs themselves.
        self.wait(lambda: proofs.stats()['finished'] == submitted)
        self.assertEqual(proofs.stats()['ready'], 4)
        proofs.step()
        self.assertEqual(proofs.stats()['installed'], 4)
        self.assertEqual({native.hxg_exact(g.ptr) for g in graphs}, {0})
        records = proofs.records()
        self.assertEqual(len({r['id'] for r in records}), 4)
        self.assertEqual({r['game'] for r in records}, set(range(4)))
        for row in records:
            self.assertEqual(row['request']['history'], self.opening)
            self.assertEqual(independent_verify(row['result']['certificate'], self.opening,
                             known=row['request']['known']), 'PROVEN_WIN')

    def test_root_proof_overrides_late_neural_rows_and_releases_reservations(self):
        graph = self.graph(self.opening)
        pool = self.pool([graph], quantum=32, views=1, work=4096)
        proofs = self.loop(pool, workers=1, queue=4)
        pool.step()
        self.answer(pool)
        pool.step()
        late = pool.feed.take(128)
        self.assertIsNotNone(late)
        self.wait(lambda: proofs.stats()['finished'] > 0)
        proofs.step()
        self.assertEqual(native.hxg_exact(graph.ptr), 0)
        self.answer(pool, late)
        pool.step()
        self.assertTrue(pool.done())
        self.assertEqual(native.hxg_exact(graph.ptr), 0)
        self.assertEqual((pool.feed.stats()['pending_rows'], pool.games[0].stats()['pending']), (0, 0))
        self.assertEqual(pool.games[0].records()[-1]['root_estimate'], 1.)

    def test_arbitrary_defender_proof_reaches_other_contexts_and_ancestors(self):
        from tactical_proof import independent_verify
        history = self.opening+[[-1,0],[2,1]]
        graph = self.graph(self.opening)
        graph.search(4, root_samples=4, batch_size=4)
        middle = graph.view(self.opening+[[-1,0]])
        self.addCleanup(middle.close)
        middle.search(4, root_samples=4, batch_size=4)
        peer = graph.view(self.opening+[[2,1],[-1,0]])
        self.addCleanup(peer.close)
        pool = self.pool([graph], quantum=16, views=1, work=4096)
        proofs = self.loop(pool, workers=1, queue=4)
        proofs.offer(0, history, relevance=10.)
        # The two placements have another neural context, but share rule-position proofs.
        def settled():
            proofs.step()
            return native.hxg_exact(peer.ptr) == 0
        self.wait(settled)
        self.assertEqual(native.hxg_exact(graph.ptr), 0)
        loss = next(r for r in proofs.records() if r['request']['history'] == history)
        self.assertEqual(loss['result']['status'], 'PROVEN_LOSS')
        self.assertEqual(independent_verify(loss['result']['certificate'], history, attacker='defender',
                         known=loss['request']['known']), 'PROVEN_LOSS')

    def test_unknown_and_forcing_disproof_never_become_game_losses(self):
        graph = self.graph([[0,0],[1,2],[3,-1]])
        pool = self.pool([graph], quantum=16, views=1, work=4096)
        proofs = self.loop(pool, workers=1, queue=4)
        pool.step()
        self.answer(pool)
        def tried_both():
            proofs.step()
            return proofs.stats()['unknown'] >= 2
        self.wait(tried_both)
        self.assertEqual(native.hxg_exact(graph.ptr), -1)
        self.assertTrue(pool.games[0].evidence()['eligible'].all())
        self.assertEqual(proofs.records(), [])

    def test_drain_stops_admission_and_retarget_discards_old_completions(self):
        graph = self.graph(self.opening)
        pool = self.pool([graph], quantum=16, views=1, work=4096)
        proofs = self.loop(pool, workers=1, queue=4)
        pool.step()
        self.answer(pool)
        proofs.step()
        self.wait(lambda: proofs.stats()['finished'] > 0)
        pool.retarget(0, [[0,0],[1,2],[3,-1]], work=4096)
        proofs.drain()
        stats = proofs.stats()
        self.assertEqual((stats['queued'], stats['active'], stats['ready']), (0, 0, 0))
        self.assertEqual(proofs.records(), [])
        pool.step()
        self.answer(pool)
        proofs.step()
        self.assertEqual(proofs.stats()['submitted'], stats['submitted'])
        self.assertEqual(native.hxg_exact(native.hxgo_root(pool.games[0].ptr)), -1)
        proofs.resume()
        proofs.step()
        self.assertGreater(proofs.stats()['submitted'], stats['submitted'])

    def test_attached_loop_prevents_pool_free_and_duplicate_owners(self):
        graph = self.graph()
        pool = self.pool([graph], quantum=16, work=32)
        proofs = self.loop(pool, workers=1, queue=4)
        with self.assertRaisesRegex(ValueError, 'already has a proof loop'):
            pool.enable_proofs()
        self.assertEqual(native.hxgm_free(pool.ptr), 0)
        self.assertIn(b'Close the proof loop', native.hxg_error())
        proofs.close()
        pool.close()
        with self.assertRaisesRegex(ValueError, 'closed'):
            proofs.stats()

    def test_cancel_rejects_a_ready_completion_before_installation(self):
        graph = self.graph(self.opening)
        pool = self.pool([graph], quantum=16, views=1, work=4096)
        proofs = self.loop(pool, workers=1, queue=4)
        pool.step()
        self.answer(pool)
        proofs.step()
        self.wait(lambda: proofs.stats()['ready'] > 0)
        pool.cancel()
        proofs.drain()
        self.assertEqual(proofs.stats()['installed'], 0)
        self.assertGreater(proofs.stats()['cancelled'], 0)
        self.assertEqual(native.hxg_exact(graph.ptr), -1)
        self.assertEqual(proofs.records(), [])

    def test_token_exhaustion_reports_zero_fresh_work_and_no_game_verdict(self):
        graph = self.graph(self.opening)
        pool = self.pool([graph], quantum=16, views=1, work=4096)
        proofs = self.loop(pool, workers=1, queue=4)
        lib = proofs.library.lib
        tokens = [lib.hexo_tactical_prepare() for _ in range(64)]
        for token in tokens:
            self.addCleanup(lib.hexo_tactical_release, token)
        self.assertTrue(all(tokens))
        self.assertEqual(lib.hexo_tactical_prepare(), 0)
        pool.step()
        self.answer(pool)
        proofs.step()
        self.wait(lambda: proofs.stats()['finished'] > 0)
        with self.assertRaisesRegex(ValueError, 'cancellation token limit'):
            proofs.step()
        self.assertEqual((proofs.stats()['fresh_nodes'], proofs.stats()['missing_fresh']), (0, 0))
        self.assertEqual(native.hxg_exact(graph.ptr), -1)

    def test_short_slice_collects_actual_work_without_publishing_a_late_verdict(self):
        import ctypes as C
        import json
        from tactical_proof import NativeTactics
        with NativeTactics(independent=True) as worker:
            lib = worker.lib
            for name, result, args in (
                ('worker_answer', C.c_void_p, [C.c_void_p, C.c_char_p]),
                ('answer_info', C.c_bool, [C.c_void_p, C.POINTER(C.c_uint64)]),
                ('answer_json', C.c_void_p, [C.c_void_p]),
                ('answer_free', None, [C.c_void_p]),
            ):
                function = getattr(lib, 'hexo_tactical_'+name)
                function.argtypes, function.restype = args, result
            token = lib.hexo_tactical_prepare()
            self.assertTrue(token)
            try:
                request = dict(history=self.opening, ms=8, nodes=10_000_000, idtt_nodes=0, depth=8,
                               table_mb=1, bounds=True, resume=True, request_id=token)
                answer = lib.hexo_tactical_worker_answer(worker.worker, json.dumps(request).encode())
                self.assertTrue(answer)
                try:
                    info = (C.c_uint64*13)()
                    self.assertTrue(lib.hexo_tactical_answer_info(answer, info))
                    raw = lib.hexo_tactical_answer_json(answer)
                    try:
                        row = json.loads(C.string_at(raw))
                    finally:
                        lib.hexo_tactical_free(raw)
                    self.assertEqual((info[0], row['status'], row['native_verified']),
                                     (0, 'UNKNOWN', False))
                    self.assertEqual(info[4], 1)
                    self.assertEqual(info[3], row['nodes_fresh'])
                    self.assertIsNone(row['certificate'])
                    self.assertFalse(worker.busy)
                    self.assertEqual(row['last_worker_completion']['completed_queries'], 1)
                finally:
                    lib.hexo_tactical_answer_free(answer)
            finally:
                lib.hexo_tactical_release(token)


if __name__ == '__main__':
    unittest.main()
