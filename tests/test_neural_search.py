"""Native neural tree checks independent of trained model quality."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from hexo import Game
from neural_search import NeuralSearch, EvaluationCache, GameGraph, Recheck, SearchCoordinator, native
from tests import PATIENCE
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

def take_owed(service, limit, *out):
    """hxb_take for a batch the producers owe: retries its 1 s wait until a batch or an error, or PATIENCE runs out."""
    import time
    end = time.monotonic()+PATIENCE
    while not (count := native.hxb_take(service, limit, 1000., *out)) and time.monotonic() < end:
        pass
    return count


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
        # A completed native oracle still has to validate the supplied
        # prediction before PUCT consumes its arrays in legal-action order.
        for changed in ('actions', 'logit_shape', 'q_shape', 'logits', 'q', 'range'):
            prediction = Uniform().evaluate([history])[0]
            if changed == 'actions':
                prediction['actions'] = prediction['actions'][::-1]
            elif changed == 'logit_shape':
                prediction['logits'] = prediction['logits'][:-1]
            elif changed == 'q_shape':
                prediction['q'] = prediction['q'][:,None]
            elif changed == 'range':
                prediction['q'][0] = 2
            else:
                prediction[changed][0] = np.nan
            invalid = PUCTSearch(Uniform(), 'puct-invalid', history, tactics=True)
            self.addCleanup(invalid.close)
            invalid.begin(8)
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'Invalid evaluation'):
                invalid.fulfill(invalid.request(), prediction)

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
        self.assertEqual(result['evaluated'], 0)
        game = Game(history)
        try:
            self.assertEqual(result['actions'].tolist(), [list(p) for p in game.legal_moves()])
        finally:
            game.close()
        self.assertTrue(np.isfinite(result['policy']).all())
        self.assertAlmostEqual(float(result['policy'].sum()), 1.)
        self.assertEqual(search.search(65536)['evaluated'], 0)
        search.advance(result['action'])
        game = Game(search.history)
        if game.winner < 0:
            game.close()
            second = search.search(8)
            self.assertEqual(second['evaluated'], 0)
            search.advance(second['action'])
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

    def test_immediate_win_does_not_query_a_slower_certificate(self):
        # Player 0 completes six with one stone. Existing exact evidence must
        # settle this root before a solver or the network receives work.
        history = [[0,0],[0,3],[1,3],[1,0],[2,0],[2,3],[3,3],[3,0],[4,0],[4,3],[5,4]]
        class Slow:
            def history(self, history, ms, certificate=None, **kwargs):
                raise AssertionError('An immediate win does not need a solver query')
        search = NeuralSearch(Uniform(), 'longer-certificate', history, tactics=True, proof_solver=Slow())
        self.addCleanup(search.close)
        result = search.search(8)
        self.assertEqual((result['proven'], result['proof_plies']), (1, 1))
        self.assertEqual(result['evaluated'], 0)
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
    def test_forward_archive_discards_both_colour_conflicts_but_keeps_missing_stones(self):
        for forward in (False, True):
            for opposite in (False, True):
                graph = GameGraph(Uniform(), 'forward', [(0,0)], seed=7, limit=1,
                                  archive_bytes=65536, archive_forward=forward)
                try:
                    for history in ([(0,0)], [(0,0),(1,0),(2,0)], [(0,0),(1,0),(2,0),(3,0)]):
                        graph.at(history)
                        graph.expand()
                    graph.at([(0,0)])
                    before = graph.archive()
                    self.assertEqual(before['nodes'], 2)
                    self.assertLess(before['bytes'], before['limit'])
                    focus = [(0,0),(4,0),(5,0),(1,0)] if opposite else [(0,0),(3,0)]
                    graph.at(focus)
                    after = graph.archive()
                    self.assertEqual(after['nodes'], 0 if forward and opposite else 1 if forward else 2)
                    self.assertEqual(after['discarded']-before['discarded'],
                                     2 if forward and opposite else 1 if forward else 0)
                    self.assertEqual(graph.counters()['pending'], 0)
                finally:
                    graph.close()

    def test_dormant_reconvergence_restores_edge_evidence_and_new_credits_are_separate(self):
        original = [(0,0),(1,1),(2,1),(2,0),(0,3),(0,2),(1,2),(-1,2),(3,1),(-1,0),(0,-1)]
        current = original[:3]+original[7:9]+original[5:7]
        returned = current+original[3:5]+original[9:11]
        graph = GameGraph(Uniform(), 'archive', original, seed=51, limit=4, archive_bytes=262144, archive_forward=True,
                          cache=EvaluationCache(capacity=0))
        self.addCleanup(graph.close)
        first = graph.search(128, root_samples=16, batch_size=16)
        graph.at(current)
        graph.search(128, root_samples=16, batch_size=16)
        graph.search(4, root_samples=4, batch_size=16)
        self.assertGreater(graph.archive()['discarded'], 0)
        self.assertLessEqual(graph.archive()['bytes'], graph.archive()['limit'])
        graph.at(returned)
        reused = graph.result(0, 0, 0, 0)
        np.testing.assert_array_equal(reused['visits'], first['visits'])
        self.assertGreater(graph.archive()['reused'], 0)
        self.assertEqual(len(reused['actions']), len(first['actions']))
        result = graph.search(8, root_samples=8, batch_size=16)
        self.assertEqual(result['completed'], 8)
        self.assertEqual(int(graph.credits().sum()), 8)
        self.assertEqual(int(result['visits'].sum()), 136)
        self.assertEqual(graph.counters()['pending'], 0)
        different_context = original[:5]+original[9:11]+original[7:9]+original[5:7]
        graph.at(different_context)
        self.assertEqual(len(graph.result(0, 0, 0, 0)['actions']), 0)
        different_colours = list(original)
        different_colours[1], different_colours[3] = different_colours[3], different_colours[1]
        graph.at(different_colours)
        self.assertEqual(len(graph.result(0, 0, 0, 0)['actions']), 0)

    def test_auxiliary_archive_focus_and_rejected_configuration_leave_game_usable(self):
        graph = GameGraph(Uniform(), 'archive-focus', [(0,0)], limit=4, archive_bytes=65536)
        self.addCleanup(graph.close)
        graph.search(64, root_samples=8, batch_size=8)
        graph.at(graph.history)
        before = graph.archive()['focus_stones']
        view = graph.view([(0,0),(1,1)])
        self.addCleanup(view.close)
        self.assertEqual(graph.archive()['focus_stones'], before)
        with self.assertRaisesRegex(ValueError, 'Archive needs'):
            from neural_search import checked
            checked(native.hxg_archive(graph.ptr, 65536))
        self.assertEqual(graph.archive()['limit'], 65536)
        graph.search(8, root_samples=4, batch_size=4)
        self.assertLessEqual(graph.archive()['bytes'], 65536)

    def test_later_marks_tighten_a_proven_root_and_its_stored_parent(self):
        history = [(0,0),(0,3),(1,3),(1,0),(2,0),(2,3),(3,3),(3,0),(7,4),(4,3),(5,4)]
        graph = GameGraph(Uniform(), 'tighter-proof', history, seed=3, tactics=False)
        self.addCleanup(graph.close)
        graph.expand()
        graph.mark((-1,0),0,42)
        self.assertEqual(native.hxg_distance(graph.ptr),42)
        graph.mark((4,0),0,34)
        self.assertEqual(native.hxg_distance(graph.ptr),34)
        self.assertEqual(graph.result(0,0,0,0)['action'],[4,0])
        child = graph.view(history+[(-1,0)],seed=4)
        self.addCleanup(child.close)
        child.expand()
        child.mark((4,0),0,29)
        self.assertEqual(native.hxg_distance(graph.ptr),30)
        self.assertEqual(graph.result(0,0,0,0)['action'],[-1,0])
        child.mark((4,0),0,41)
        graph.mark((4,0),0,42)
        self.assertEqual(native.hxg_distance(graph.ptr),30)

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

    def test_root_sampling_considers_a_stored_high_value_child(self):
        # A neural estimate learned below x competes on its stored Q when sampling the earlier root.
        a = recorded_position(11)
        probe = self.graph(Uniform(), a)
        actions = [tuple(map(int, action)) for action in probe.search(2, root_samples=2, batch_size=2)['actions']]
        x = actions[-1]
        mover = lambda n: 0 if n == 0 else ((n - 1) // 2 + 1) % 2
        class Likes(Uniform):
            def evaluate(self, histories):
                predictions = super().evaluate(histories)
                for history, prediction in zip(histories, predictions):
                    if len(history) > len(a) and tuple(map(int, history[len(a)])) == x:
                        prediction['q'][:] = 1. if mover(len(history)) == mover(len(a) + 1) else -1.
                return predictions
        graph = self.graph(Likes(), [*a, x])
        graph.search(32, root_samples=4, batch_size=4)
        graph.at(a)
        chosen = graph.search(4, root_samples=2, batch_size=2)
        self.assertEqual(tuple(map(int, chosen['action'])), x)
        self.assertGreater(chosen['visits'][self.edge(chosen, x)], 0)

    def test_opening_candidates_use_stored_scores_independent_of_batch_size(self):
        class Changing(Spread):
            def evaluate(self, histories):
                predictions = super().evaluate(histories)
                for history, prediction in zip(histories, predictions):
                    if len(history) > 2:
                        prediction['q'] *= -1.
                return predictions
        for retained in (False, True):
            candidates = []
            for batch in (1, 2, 8):
                graph = GameGraph(Changing(.9), 'sampling', [(0, 0)], seed=0, tactics=False)
                self.addCleanup(graph.close)
                if retained:
                    graph.search(1, root_samples=4, batch_size=8)
                result = graph.search(8, root_samples=8, batch_size=batch)
                chosen = result['actions'][graph.credits() > 0].tolist()
                self.assertEqual(len(chosen), 8)
                candidates.append(chosen)
            with self.subTest(retained=retained):
                self.assertEqual(candidates[0], candidates[1])
                self.assertEqual(candidates[0], candidates[2])

    def test_opening_scores_survive_leaving_and_resuming_the_root(self):
        candidates = []
        for revisit in (False, True):
            evaluator = Spread(.9)
            graph = self.graph(evaluator, [(0, 0)])
            graph.search(1, root_samples=1)
            native.hxg_begin(graph.ptr, 8, 8)
            for i in range(8):
                request, history = graph.request()
                self.assertGreater(request, 0)
                graph.fulfill(request, evaluator.evaluate([history])[0])
                if revisit and i == 0:
                    graph.at(history)
                    graph.at([(0, 0)])
            result = graph.result(0, 0, 0, 0)
            self.assertEqual(result['completed'], 8)
            candidates.append(result['actions'][graph.credits() > 0].tolist())
        self.assertEqual(candidates[0], candidates[1])

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
    def test_launcher_feeds_arriving_rows_during_gpu_wait_and_before_return(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from native_scheduler import InferenceService
        service = InferenceService.__new__(InferenceService)
        service.pending = []
        service.flight_limit = 2
        service.models = [None]
        launched, collected, delivered = [], [], []
        batches = iter([(1,0,'first'), None, (2,0,'second'), (3,0,'third')])
        service.take = lambda wait: next(batches,None)
        service.complete = lambda token,rows: delivered.append((token,rows))
        service.cancel = lambda: self.fail('Healthy forwards must not cancel the service')
        def submit(model,rows):
            launched.append(rows)
            def collect():
                # The newly ready second batch starts before the first fence.
                self.assertIn('second',launched)
                collected.append(rows)
                return rows
            return SimpleNamespace(event=SimpleNamespace(query=lambda:False),collect=collect)
        backend = SimpleNamespace(submit=submit,_quarantined=[])
        with patch.dict(sys.modules, native_dense=backend):
            service.pump()
        self.assertEqual(launched,['first','second','third'])
        self.assertEqual(collected,['first'])
        self.assertEqual(delivered,[(1,'first')])
        self.assertEqual([token for token,_ in service.pending],[2,3])

    def test_launcher_refills_the_configured_batch_capacity(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from native_scheduler import InferenceService
        for limit in (1,2,4):
            with self.subTest(limit=limit):
                service = InferenceService.__new__(InferenceService)
                service.pending, service.models, service.flight_limit = [], [None], limit
                launched, collected = [], []
                batches = iter((i,0,i) for i in range(1,limit+2))
                service.take = lambda wait:next(batches,None)
                service.complete = lambda token,rows:collected.append(token)
                def submit(model,rows):
                    launched.append(rows)
                    def collect():
                        self.assertEqual(len(service.pending),limit)
                        return rows
                    return SimpleNamespace(event=SimpleNamespace(query=lambda:True),collect=collect)
                with patch.dict(sys.modules,native_dense=SimpleNamespace(submit=submit)):
                    service.pump()
                self.assertEqual(launched,list(range(1,limit+2)))
                self.assertEqual(collected,[1])
                self.assertEqual([token for token,_ in service.pending],list(range(2,limit+2)))

    def test_launcher_keeps_failed_gpu_fence_owned_and_quarantined_submit_live(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from native_scheduler import InferenceService
        service = InferenceService.__new__(InferenceService)
        service.pending = []
        service.flight_limit = 2
        service.models = [None]
        rows = object()
        service.take = lambda wait:(1,0,rows)
        cancelled = []
        service.cancel = lambda:cancelled.append(True)
        service.abandon_fenced = lambda token:self.fail('Unfenced storage must stay owned')
        service.complete = lambda *args:self.fail('Failed forward cannot be delivered')
        handle = SimpleNamespace(rows=rows)
        def submit(*args):raise RuntimeError('submission fence failed')
        backend = SimpleNamespace(submit=submit,_quarantined=[handle])
        with patch.dict(sys.modules,native_dense=backend):
            with self.assertRaisesRegex(RuntimeError,'submission fence failed'):
                service.pump()
        self.assertEqual(cancelled,[True])
        self.assertEqual(service.pending,[(1,handle)])
        def collect():raise RuntimeError('completion fence failed')
        handle.event = SimpleNamespace(query=lambda:True)
        handle.collect = collect
        service.take = lambda wait:None
        with patch.dict(sys.modules,native_dense=backend):
            with self.assertRaisesRegex(RuntimeError,'completion fence failed'):
                service.pump()
        self.assertEqual(service.pending,[(1,handle)])

    def test_round_choice_survives_refuted_finalists_and_new_root_lease(self):
        graph = GameGraph(Uniform(), 'scheduler', [(0,0)], limit=96, round_barrier=True)
        self.addCleanup(graph.close)
        pool = self.pool([graph], quantum=128, views=1, work=128, cache=0)
        self.finish(pool)
        game = pool.games[0]
        root = native.hxgo_root(game.ptr)
        from neural_search import checked
        checked(native.hxg_begin(root,32,8))
        # The completed comparison remains usable while a new one has no credits.
        chosen = []
        for _ in range(5):
            action = game.choice()
            self.assertIsNotNone(action)
            self.assertNotIn(action,chosen)
            chosen.append(action)
            checked(native.hxg_mark_exact(root,*action,0,4))  # P2-to-play continuation refuted.
        self.assertEqual(int(game.evidence()['lifetime_credits'].sum()),128)

    def test_scheduled_primary_owns_archive_focus_after_source_closes(self):
        history = [(0,0),(1,1),(2,1)]
        graph = GameGraph(Uniform(), 'scheduler', history, limit=4, archive_bytes=65536, archive_forward=True)
        self.addCleanup(graph.close)
        pool = self.pool([graph], quantum=8, views=4, work=64)
        graph.close()
        self.finish(pool)
        root = native.hxgo_root(pool.games[0].ptr)
        counts = np.zeros(10, np.int64)
        self.assertTrue(native.hxg_archive_stats(root, counts.ctypes.data))
        self.assertEqual(counts[9], len(history))
        pool.retarget(0, [*history, (2,0), (0,3)], work=32)
        self.finish(pool)
        self.assertTrue(native.hxg_archive_stats(root, counts.ctypes.data))
        self.assertEqual(counts[9], len(history)+2)
    def test_retarget_same_context_keeps_new_subscriber_after_old_empty_completion(self):
        pool = self.pool([self.graph()], views=1, work=16, cache=0)
        pool.step()
        old = pool.feed.take(128)
        self.assertIsNotNone(old)
        pool.retarget(0, [(0,0)], work=16)
        pool.step()
        new = pool.feed.take(128)
        self.assertIsNotNone(new)
        self.assertNotEqual(old[0].tolist(), new[0].tolist())
        # A cancelled immutable batch remains completable. Its empty result
        # must neither fail the new root nor erase its pending identity.
        pool.feed.install(old[0], [None]*len(old[0]))
        self.assertEqual(pool.stats()['failed'], 0)
        self.assertEqual(pool.feed.stats()['pending_rows'], len(new[0]))
        self.answer(pool, new)
        self.finish(pool)
        self.assertEqual(pool.games[0].stats()['completed'], 16)
        self.assertEqual((pool.feed.stats()['pending_rows'], pool.feed.stats()['pending_requests']), (0,0))

    def test_cached_search_progress_is_live_without_new_neural_rows(self):
        history = [(0,0)]
        graph = self.graph(history)
        pool = self.pool([graph], quantum=4, views=1, work=4, cache=1024)
        from neural_search import checked
        root = Uniform().evaluate([history])[0]
        # Prime one complete legal ply from another independent graph. The
        # scheduler must consume these cached predictions without idling.
        for cells in [history]+[[*history,tuple(map(int,a))] for a in root['actions']]:
            child = GameGraph(Uniform(), 'scheduler', cells)
            try:
                prediction = Uniform().evaluate([cells])[0]
                h = np.asarray(cells,np.int64)
                checked(native.hxgf_begin(pool.feed.ptr,child.ptr,h.ctypes.data,len(h)))
                checked(native.hxgf_seed(pool.feed.ptr,child.ptr,prediction['actions'].ctypes.data,
                                        prediction['logits'].ctypes.data,prediction['q'].ctypes.data,
                                        len(prediction['actions'])))
                native.hxgf_detach(pool.feed.ptr,child.ptr)
            finally:
                child.close()
        self.assertGreater(pool.step(), 0)
        self.assertIsNone(pool.feed.take(128))
        self.assertEqual(pool.feed.stats()['new_rows'], 0)
        self.assertGreater(pool.feed.stats()['cache_hits'], 0)
        self.finish(pool)
        stats = pool.games[0].stats()
        self.assertEqual((stats['completed'],stats['cancelled']), (4,0))
        self.assertEqual(int(pool.games[0].evidence()['lifetime_credits'].sum()), 4)

    def test_native_service_delivers_zero_row_root_and_release_without_batch_wait(self):
        import ctypes as C
        import time
        from types import SimpleNamespace
        from neural_search import checked, ptr
        from native_scheduler import InferenceService
        history = [(0,0),(0,3),(1,3),(1,0),(2,0),(2,3),(3,3),(3,0),(7,4),(4,3),(5,4)]
        graph = self.graph(history)
        checked(native.hxg_tactics(graph.ptr, True))
        pool = self.pool([graph], quantum=16, views=1, work=16, cache=0)
        service = InferenceService([pool],[SimpleNamespace(model_version='scheduler')])
        try:
            service.start(continuous=True)
            service.retarget(0,0,history,work=16,views=1)
            token, model, snapshot = C.c_uint64(), C.c_int(), ptr()
            start = time.monotonic()
            self.assertEqual(native.hxb_take(service.ptr,128,1000.,C.byref(token),C.byref(model),C.byref(snapshot)),0)
            self.assertLess(time.monotonic()-start, .5)
            event = service.event()
            self.assertEqual((event['token'],event['exact_winner']), (1,0))
            self.assertIsNone(service.event())
            self.assertEqual(service.stats()['unique_rows'], 0)
            self.assertFalse(service.done())
            # With no remaining event or ready work, take really waits.
            start = time.monotonic()
            self.assertEqual(native.hxb_take(service.ptr,128,20.,C.byref(token),C.byref(model),C.byref(snapshot)),0)
            self.assertGreaterEqual(time.monotonic()-start, .01)
            service.release(0,0,expected=1)
            start = time.monotonic()
            self.assertEqual(native.hxb_take(service.ptr,128,1000.,C.byref(token),C.byref(model),C.byref(snapshot)),0)
            self.assertLess(time.monotonic()-start, .5)
            event = service.event()
            self.assertEqual((event['kind'],event['token']), ('released',2))
            self.assertIsNone(service.event())
        finally:
            service.close()
        self.assertEqual((service.stats()['pending_rows'],service.stats()['active_producers']), (0,0))

    def test_native_service_pending_root_event_does_not_block_ready_inference(self):
        import ctypes as C
        import time
        from types import SimpleNamespace
        from neural_search import checked, ptr
        from native_scheduler import InferenceService
        history = [(0,0),(0,3),(1,3),(1,0),(2,0),(2,3),(3,3),(3,0),(7,4),(4,3),(5,4)]
        graph = self.graph(history)
        checked(native.hxg_tactics(graph.ptr, True))
        pool = self.pool([graph,self.graph()], quantum=16, views=1, work=16, cache=0)
        service = InferenceService([pool],[SimpleNamespace(model_version='scheduler')], latency_ms=20.)
        token, model, snapshot = C.c_uint64(), C.c_int(), ptr()
        try:
            service.start(continuous=True)
            service.retarget(0,0,history,work=16,views=1)
            service.retarget(0,1,[(0,0)],work=16,views=1)
            counts = np.empty(10,np.uint64)
            until = time.monotonic()+PATIENCE
            while time.monotonic()<until:
                native.hxb_stats(service.ptr,counts.ctypes.data)
                if counts[0]:break
                time.sleep(.001)
            self.assertEqual(counts[0], 1)
            # A ready row starts inference even though the requested batch
            # is not full and the exact neighbor's root event is still pending.
            self.assertEqual(native.hxb_take(service.ptr,128,0.,C.byref(token),C.byref(model),C.byref(snapshot)),1)
            event = service.event()
            self.assertEqual((event['game'],event['exact_winner']), (0,0))
        finally:
            if token.value:
                checked(native.hxb_abort(service.ptr,token))
                native.hxgp_free(snapshot)
            service.close()
        self.assertEqual((service.stats()['pending_rows'],service.stats()['inflight_batches']), (0,0))

    def test_native_service_coalesces_producers_and_isolates_model_predictions(self):
        import ctypes as C
        import time
        from neural_search import bind, ptr
        bind('hxgp_groups', C.c_int, ptr)
        bind('hxgp_group', C.c_int, ptr, C.c_int, ptr)
        bind('hxgp_decode', C.c_int, ptr, C.c_int, C.c_int, C.c_int, ptr, C.c_int64)
        bind('hxgp_outputs', C.c_int, ptr, ptr)
        bind('hxgp_free', None, ptr)
        pools = [self.pool([self.graph(version=version)], views=1, work=16)
                 for version in ('scheduler','scheduler','other')]
        service = native.hxb_new(16, 2, 0, 20.)
        leases, models = {}, set()
        try:
            self.assertTrue(native.hxb_attach(service, pools[0].ptr, 0))
            self.assertTrue(native.hxb_attach(service, pools[1].ptr, 0))
            self.assertEqual(native.hxb_attach(service, pools[2].ptr, 0), 0)
            self.assertTrue(native.hxb_attach(service, pools[2].ptr, 1))
            self.assertEqual(native.hxgm_step(pools[0].ptr), -1)
            self.assertEqual(native.hxgm_free(pools[0].ptr), 0)
            self.assertTrue(native.hxb_start(service, 0.))
            # Requests share a task only while one is queued. Both identical producers ask for their root before any
            # answer, so hold the batches until their requests have met, however the threads are scheduled.
            counters = np.zeros(10, np.uint64)
            deadline = time.monotonic()+PATIENCE
            while not counters[1] and time.monotonic() < deadline:
                time.sleep(.001)
                native.hxb_stats(service, counters.ctypes.data)
            self.assertGreater(counters[1], 0, 'identical requests from two producers did not share a task')
            for _ in range(1000):
                done = native.hxb_done(service)
                self.assertGreaterEqual(done, 0, native.hxg_error().decode())
                if done:
                    break
                token, model, snapshot = C.c_uint64(), C.c_int(), ptr()
                count = native.hxb_take(service, 128, 50., C.byref(token), C.byref(model), C.byref(snapshot))
                self.assertGreaterEqual(count, 0, native.hxg_error().decode())
                if not count:
                    continue
                leases[token.value] = snapshot.value
                models.add(model.value)
                value = .25 if model.value==0 else -.5
                for group in range(native.hxgp_groups(snapshot)):
                    size = np.empty(2, np.int64)
                    self.assertTrue(native.hxgp_group(snapshot, group, size.ctypes.data))
                    side, rows = map(int,size)
                    output = np.zeros((rows,side*side+2), np.float32)
                    output[:,-1] = 2*np.arctanh(value)
                    self.assertTrue(native.hxgp_decode(snapshot, group, 0, rows, output.ctypes.data, output.size))
                outputs = (ptr*4)()
                self.assertTrue(native.hxgp_outputs(snapshot, outputs))
                self.assertTrue(native.hxb_complete(service, token, *outputs))
                native.hxgp_free(snapshot)
                del leases[token.value]
            else:
                self.fail('Native producer service did not complete bounded work')
            self.assertEqual(models, {0,1})
            native.hxb_stats(service, counters.ctypes.data)
            self.assertEqual(tuple(counters[7:]), (0,0,0))
        finally:
            native.hxb_cancel(service)
            for token,snapshot in leases.items():
                native.hxb_abort(service, token)
                native.hxgp_free(snapshot)
            self.assertTrue(native.hxb_join(service))
            self.assertTrue(native.hxb_free(service))
        for pool,value in zip(pools,(.25,.25,-.5)):
            stats = pool.games[0].stats()
            self.assertEqual((stats['completed'],stats['pending']), (16,0))
            self.assertEqual(int(pool.games[0].evidence()['lifetime_credits'].sum()), 16)
            self.assertAlmostEqual(pool.games[0].records()[0]['raw_value'], value)
            self.assertEqual((pool.feed.stats()['pending_rows'],pool.feed.stats()['pending_requests']), (0,0))

    def test_native_service_refuses_free_until_its_batch_is_fenced(self):
        import ctypes as C
        from neural_search import bind, ptr
        bind('hxgp_free', None, ptr)
        pool = self.pool([self.graph()], views=1, work=16)
        service = native.hxb_new(16,2,0,0.)
        token, model, snapshot = C.c_uint64(), C.c_int(), ptr()
        try:
            self.assertTrue(native.hxb_attach(service, pool.ptr, 0))
            self.assertTrue(native.hxb_start(service, 1000.))
            count = take_owed(service,128,C.byref(token),C.byref(model),C.byref(snapshot))
            self.assertGreater(count,0)
            self.assertEqual(native.hxb_join(service),0)
            self.assertEqual(native.hxb_free(service),0)
            # This check launched no GPU; the immutable batch is already fenced.
            self.assertTrue(native.hxb_abort(service,token))
            native.hxgp_free(snapshot)
            token.value=0
            self.assertTrue(native.hxb_join(service))
        finally:
            native.hxb_cancel(service)
            if token.value:
                native.hxb_abort(service,token)
                native.hxgp_free(snapshot)
            self.assertTrue(native.hxb_join(service))
            self.assertTrue(native.hxb_free(service))
        self.assertEqual((pool.feed.stats()['pending_rows'],pool.feed.stats()['pending_requests']),(0,0))
        self.assertEqual(native.hxg_exact(native.hxgo_root(pool.games[0].ptr)),-1)

    def test_native_service_installs_a_returned_row_while_its_snapshot_tail_waits(self):
        import ctypes as C
        import time
        from neural_search import bind, ptr
        bind('hxgp_groups', C.c_int, ptr)
        bind('hxgp_shape', C.c_int, ptr, C.c_int, ptr)
        bind('hxgp_decode', C.c_int, ptr, C.c_int, C.c_int, C.c_int, ptr, C.c_int64)
        bind('hxgp_outputs', C.c_int, ptr, ptr)
        bind('hxgp_free', None, ptr)
        for cancel_tail in (False,True):
            with self.subTest(cancel_tail=cancel_tail):
                pool=self.pool([self.graph(),self.graph([(0,0),(4,0),(7,0)])],
                               quantum=16,views=1,work=16,cache=0)
                service=native.hxb_new(16,2,0,20.)
                token,model,snapshot=C.c_uint64(),C.c_int(),ptr()
                counters=np.empty(10,np.uint64)
                try:
                    self.assertTrue(native.hxb_attach(service,pool.ptr,0))
                    self.assertTrue(native.hxb_start(service,0.))
                    until=time.monotonic()+PATIENCE
                    while time.monotonic()<until:
                        native.hxb_stats(service,counters.ctypes.data)
                        if counters[0]>=2:break
                        time.sleep(.001)
                    self.assertEqual(counters[0],2)
                    count=take_owed(service,1,C.byref(token),C.byref(model),C.byref(snapshot))
                    self.assertEqual(count,1)
                    for group in range(native.hxgp_groups(snapshot)):
                        info=np.empty(3,np.int64)
                        self.assertTrue(native.hxgp_shape(snapshot,group,info.ctypes.data))
                        height,width,rows=map(int,info)
                        output=np.zeros((rows,height*width+2),np.float32)
                        self.assertTrue(native.hxgp_decode(snapshot,group,0,rows,
                                                         output.ctypes.data,output.size))
                    outputs=(ptr*4)()
                    self.assertTrue(native.hxgp_outputs(snapshot,outputs))
                    self.assertTrue(native.hxb_complete(service,token,*outputs))
                    native.hxgp_free(snapshot)
                    token.value=0
                    # The other cold root remains unsent. Its neighbor must
                    # install this result and supply useful continuations.
                    until=time.monotonic()+PATIENCE
                    while time.monotonic()<until:
                        native.hxb_stats(service,counters.ctypes.data)
                        if counters[0]>2 and native.hxb_installed(service)==1:break
                        time.sleep(.001)
                    self.assertEqual(counters[3],1)
                    self.assertEqual(native.hxb_installed(service),1)
                    self.assertGreater(counters[0],2)
                    if not cancel_tail:
                        for _ in range(1000):
                            if native.hxb_done(service):break
                            count=native.hxb_take(service,1,50.,C.byref(token),C.byref(model),C.byref(snapshot))
                            self.assertGreaterEqual(count,0)
                            if not count:continue
                            for group in range(native.hxgp_groups(snapshot)):
                                self.assertTrue(native.hxgp_shape(snapshot,group,info.ctypes.data))
                                height,width,rows=map(int,info)
                                output=np.zeros((rows,height*width+2),np.float32)
                                self.assertTrue(native.hxgp_decode(snapshot,group,0,rows,
                                                                 output.ctypes.data,output.size))
                            self.assertTrue(native.hxgp_outputs(snapshot,outputs))
                            self.assertTrue(native.hxb_complete(service,token,*outputs))
                            native.hxgp_free(snapshot)
                            token.value=0
                        else:self.fail('Partial native delivery did not finish both searches')
                finally:
                    native.hxb_cancel(service)
                    if token.value:
                        native.hxb_abort(service,token)
                        native.hxgp_free(snapshot)
                    self.assertTrue(native.hxb_join(service))
                    native.hxb_stats(service,counters.ctypes.data)
                    self.assertTrue(native.hxb_free(service))
                self.assertEqual(tuple(counters[7:]),(0,0,0))
                self.assertEqual((pool.feed.stats()['pending_rows'],pool.feed.stats()['pending_requests']),(0,0))
                if not cancel_tail:
                    for game in pool.games:
                        self.assertEqual(game.stats()['completed'],16)
                        self.assertEqual(int(game.evidence()['lifetime_credits'].sum()),16)
                pool.close()

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

    def test_new_continuations_skip_positions_proven_after_discovery(self):
        from neural_search import checked
        graph = self.graph()
        actions = Uniform().evaluate([[(0,0)]])[0]['actions'].tolist()
        children = [graph.view([(0,0), tuple(action)]) for action in actions]
        for child in children:
            self.addCleanup(child.close)
        pool = self.pool([graph], quantum=16, views=4, depth=3, work=256)
        pool.step()
        self.answer(pool)
        pool.step()  # The expanded focus has now offered its continuations.
        before = pool.games[0].stats()['created']
        # Caller-verified synthetic losses exercise delivery and admission, not
        # the game solver. Keep two unresolved first stones for a complete turn.
        for child in children[:-2]:
            checked(native.hxg_prove_loss(child.ptr, 0, 4))
        self.finish(pool)
        new = [r for r in pool.games[0].records() if r['view'] > before and r['depth']]
        self.assertTrue(new)
        self.assertTrue(all(r['history'][1] in actions[-2:] for r in new))
        stats = pool.games[0].stats()
        self.assertEqual(stats['issued'], stats['completed']+stats['cancelled'])
        self.assertEqual((pool.feed.stats()['pending_rows'], pool.feed.stats()['pending_requests']), (0,0))

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
        for views in (1, 4, 16):
            with self.subTest(views=views):
                pool = self.pool([self.graph()], quantum=16, views=views, work=128)
                pool.limit_ready(1)
                pool.step()
                self.assertEqual(pool.feed.queued(), 1)
                self.answer(pool)
                pool.step()
                # Finish gathering the layer even when it exceeds the watermark.
                self.assertGreater(pool.feed.queued(), 1)
                self.assertLessEqual(pool.feed.queued(), 16)
                # A supplied queue needs at most one continuation held ready;
                # raising the ceiling must not eagerly open every root view.
                self.assertLessEqual(pool.games[0].stats()['slots'], 2)
                self.finish(pool)
                stats = pool.games[0].stats()
                self.assertEqual(stats['completed'], 128)
                self.assertEqual(stats['issued'], stats['completed']+stats['cancelled'])
                self.assertEqual(int(pool.games[0].evidence()['lifetime_credits'].sum()), stats['root_completed'])
                if views > 1:
                    self.assertGreater(stats['depth'], 0)

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

    def test_prelaunch_proofs_retire_only_the_settled_games_subscribers(self):
        from neural_search import checked
        history = [(0,0),(0,3),(1,3),(1,0),(2,0),(2,3),(3,3),(3,0),(7,4),(4,3),(5,4)]
        graphs = [self.graph(history) for _ in range(2)]
        for graph in graphs:
            checked(native.hxg_tactics(graph.ptr, False))
            graph.expand()
        pool = self.pool(graphs, quantum=16, views=1, work=64)
        pool.step()
        before = pool.feed.stats()
        # Only these first stones can finish P1's four or block both ends of P2's five.
        safe = {(-2,0),(-1,0),(4,0),(5,0),(-1,3),(5,3)}
        for action in Uniform().evaluate([history])[0]['actions']:
            if tuple(action) not in safe:
                graphs[0].mark(action, 1, 3)
        self.assertEqual(native.hxg_exact(native.hxgo_root(pool.games[0].ptr)), -1)
        retired = pool.feed.prune()
        self.assertGreater(retired, 0)
        self.assertEqual(before['pending_requests']-pool.feed.stats()['pending_requests'], retired)
        self.assertEqual(pool.games[0].stats()['completed'], retired)
        self.assertGreater(pool.games[1].stats()['pending'], 0)
        self.assertEqual(pool.feed.pruning()['retired_requests'], retired)
        self.assertEqual(pool.feed.prune(), 0)
        for action in Uniform().evaluate([history])[0]['actions']:
            if tuple(action) not in safe:
                graphs[1].mark(action, 1, 3)
        retired += pool.feed.prune()
        self.assertGreater(pool.feed.pruning()['avoided_rows'], 0)
        self.assertEqual(pool.feed.prune(), 0)
        self.finish(pool)
        self.assertGreaterEqual(pool.feed.pruning()['retired_requests'], retired)
        for game in pool.games:
            stats = game.stats()
            self.assertEqual(stats['issued'], stats['completed']+stats['cancelled'])
            self.assertEqual(int(game.evidence()['lifetime_credits'].sum()), stats['root_completed'])

    def test_submitted_rows_keep_their_reservations_until_completion(self):
        from neural_search import checked
        history = [(0,0),(0,3),(1,3),(1,0),(2,0),(2,3),(3,3),(3,0),(7,4),(4,3),(5,4)]
        graph = self.graph(history)
        checked(native.hxg_tactics(graph.ptr, False))
        graph.expand()
        pool = self.pool([graph], quantum=16, views=1, work=32)
        pool.step()
        batch = pool.feed.take(128)
        before = pool.feed.stats()['pending_requests']
        safe = {(-2,0),(-1,0),(4,0),(5,0),(-1,3),(5,3)}
        for action in Uniform().evaluate([history])[0]['actions']:
            if tuple(action) not in safe:
                graph.mark(action, 1, 3)
        self.assertEqual(pool.feed.prune(), 0)
        self.assertEqual(pool.feed.stats()['pending_requests'], before)
        self.answer(pool, batch)
        counters = np.empty(6, np.uint64)
        native.hxg_view_counters(native.hxgo_root(pool.games[0].ptr), counters.ctypes.data)
        self.assertGreater(counters[4], 0)
        self.finish(pool)
        stats = pool.games[0].stats()
        self.assertEqual(stats['issued'], stats['completed']+stats['cancelled'])
        self.assertEqual((pool.feed.stats()['pending_rows'], pool.feed.stats()['pending_requests']), (0,0))

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

    def test_immediate_turn_never_enters_the_inference_queue_or_invents_a_raw_value(self):
        from neural_search import checked
        history = [(0,0),(0,3),(1,3),(1,0),(2,0),(2,3),(3,3),(3,0),(7,4),(4,3),(5,4)]
        graph = self.graph(history)
        checked(native.hxg_tactics(graph.ptr, True))
        pool = self.pool([graph], quantum=16, views=8, work=128)
        self.assertGreater(pool.step(), 0)
        self.assertTrue(pool.done())
        self.assertIsNone(pool.feed.take(128))
        self.assertEqual(pool.feed.stats()['new_rows'], 0)
        self.assertEqual((pool.feed.stats()['pending_rows'],pool.feed.stats()['pending_requests']), (0,0))
        self.assertEqual(pool.games[0].stats()['root_completed'], 0)
        record = pool.games[0].records()[0]
        self.assertEqual((record['exact_winner'],record['root_estimate']), (0,1.))
        self.assertIsNone(record['raw_value'])


    def test_replaced_slots_finish_cleanup_and_keep_retirement_bounded(self):
        import time
        from types import SimpleNamespace
        from native_scheduler import InferenceService
        graphs = [self.graph() for _ in range(8)]
        pool = self.pool(graphs, quantum=8, views=1, work=8, cache=0)
        service = InferenceService([pool], [SimpleNamespace(model_version='scheduler')])
        try:
            service.start(continuous=True)
            for game in range(len(graphs)):
                service.release(0, game, expected=0)
            released, end = set(), time.monotonic()+PATIENCE
            while len(released)<len(graphs) and time.monotonic()<end:
                event = service.event()
                if event is None:
                    time.sleep(.001)
                    continue
                self.assertEqual((event['kind'], event['token']), ('released', 1))
                released.add(event['game'])
            self.assertEqual(len(released), len(graphs))
            for graph in graphs:
                graph.close()
            for game in range(len(graphs)):
                service.replace(0, game, self.graph(), expected=1, views=1)
            replaced, end = set(), time.monotonic()+PATIENCE
            while len(replaced)<len(graphs) and time.monotonic()<end:
                stats = service.stats()
                self.assertLessEqual(stats['reclaim_queued']+stats['reclaim_active']+stats['reclaim_reserved'], 2)
                event = service.event()
                if event is None:
                    time.sleep(.001)
                    continue
                self.assertEqual((event['kind'], event['token']), ('replaced', 2))
                replaced.add(event['game'])
            self.assertEqual(len(replaced), len(graphs))
        finally:
            service.close()
        stats = service.stats()
        self.assertEqual(stats['reclaimed_games'], len(graphs))
        self.assertEqual((stats['reclaim_queued'], stats['reclaim_active'], stats['reclaim_reserved'],
                          stats['pending_rows'], stats['active_producers']), (0, 0, 0, 0, 0))


    def test_cancelling_queued_replacements_drains_cleanup_reservations(self):
        import time
        from types import SimpleNamespace
        from native_scheduler import InferenceService
        graphs = [self.graph() for _ in range(8)]
        pool = self.pool(graphs, quantum=8, views=1, work=8, cache=0)
        service = InferenceService([pool], [SimpleNamespace(model_version='scheduler')])
        try:
            service.start(continuous=True)
            for game in range(len(graphs)):
                service.release(0, game, expected=0)
            released, end = set(), time.monotonic()+PATIENCE
            while len(released)<len(graphs) and time.monotonic()<end:
                event = service.event()
                if event is None:
                    time.sleep(.001)
                    continue
                released.add(event['game'])
            self.assertEqual(len(released), len(graphs))
            for graph in graphs:
                graph.close()
            for game in range(len(graphs)):
                service.replace(0, game, self.graph(), expected=1, views=1)
            service.cancel()
        finally:
            service.close()
        stats = service.stats()
        self.assertEqual((stats['reclaim_queued'], stats['reclaim_active'], stats['reclaim_reserved'],
                          stats['pending_rows'], stats['inflight_batches'], stats['active_producers']),
                         (0, 0, 0, 0, 0, 0))


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

    def wait(self, predicate, seconds=PATIENCE):
        import time
        end = time.monotonic()+seconds
        while not predicate():
            if time.monotonic() >= end:
                self.fail('Native proof work did not finish')
            time.sleep(.002)

    def test_quiet_solver_endpoint_enters_neural_queue_without_a_game_verdict(self):
        history=[[0,0],[4,0],[7,0],[-1,0],[-2,0]]
        endpoint=history+[[5,0],[6,0],[2,0],[8,0]]
        graph=self.graph(history)
        winner=self.graph(self.opening)
        pool=self.pool([graph,winner],quantum=16,views=8,depth=1,work=1024)
        proofs=pool.enable_proofs(slice_ms=8,table_mb=1,workers=1,queue=2,endpoints=8)
        pool.step();self.answer(pool);proofs.step()
        submitted=proofs.stats()['submitted']
        self.assertGreaterEqual(submitted,2)
        self.wait(lambda:proofs.stats()['finished']==submitted)
        credits=pool.games[0].stats()['root_completed']
        pool.step(neural=False)
        frontier=proofs.stats()['neural_frontier']
        self.assertGreater(frontier['queue']['pending'],0)
        self.assertEqual(frontier['candidates'],0)
        self.assertEqual(pool.games[0].stats()['root_completed'],credits)
        # Pausing neural admission defers estimates, never a verified proof.
        self.assertEqual(native.hxg_exact(winner.ptr),0)
        pool.step()
        self.assertGreater(proofs.stats()['neural_frontier']['candidates'],0)
        self.assertEqual(proofs.stats()['neural_frontier']['queue']['admitted'],1)
        self.assertEqual(native.hxg_exact(graph.ptr),-1)
        seen=set()
        for _ in range(300):
            pool.step();batch=pool.feed.take(128)
            if batch is not None:
                seen.update(tuple(map(tuple,h.tolist())) for _,_,h in batch[1])
                self.answer(pool,batch)
            if tuple(map(tuple,endpoint)) in seen:break
        self.assertIn(tuple(map(tuple,endpoint)),seen)
        self.assertTrue(all(r['result']['status']=='UNKNOWN' for r in proofs.frontier_records()))
        self.assertEqual(native.hxg_exact(graph.ptr),-1)
        pool.cancel();proofs.drain();pool.abandon_fenced()
        self.assertEqual(proofs.stats()['neural_frontier']['queue']['pending'],0)
        self.assertEqual(pool.games[0].stats()['pending'],0)
        self.assertEqual(int(pool.games[0].evidence()['lifetime_credits'].sum()),pool.games[0].stats()['root_completed'])

    def test_every_queued_job_reaches_a_worker_without_another_owner_step(self):
        pool=self.pool([self.graph(self.opening) for _ in range(16)],quantum=16,views=1,work=4096)
        proofs=pool.enable_proofs(slice_ms=2,table_mb=1,workers=4,queue=16)
        pool.step();self.answer(pool);proofs.step()
        submitted=proofs.stats()['submitted']
        self.assertGreater(submitted,4)
        # Only workers run from here on; a lost wakeup would leave jobs queued.
        self.wait(lambda:proofs.stats()['finished']==submitted)
        stats=proofs.stats()
        self.assertEqual((stats['queued'],stats['active'],stats['submitted']),(0,0,submitted))
        pool.cancel();proofs.drain();pool.abandon_fenced()
        stats=proofs.stats()
        self.assertEqual((stats['queued'],stats['active'],stats['ready']),(0,0,0))

    def test_default_queue_holds_eight_jobs_per_worker(self):
        graph = self.graph([[0,0]])
        pool = self.pool([graph], quantum=4, views=1, work=4096)
        proofs = pool.enable_proofs(slice_ms=50, table_mb=1, workers=2, tasks=32)
        for k in range(1, 21):
            proofs.offer(0, [[0,0],[k%8+1,-1-k//8],[k%8+1,1+k//8]])
        proofs.step()
        # Answers stay held until the owner's next step, so one refill shows the bound.
        self.assertEqual(proofs.stats()['submitted'], 16)
        proofs.drain()

    def test_rejected_endpoint_limit_leaves_proof_owner_reusable(self):
        pool=self.pool([self.graph([[0,0]])],work=32)
        for endpoints in [-1,9,1.5]:
            with self.assertRaises(ValueError):pool.enable_proofs(endpoints=endpoints)
            self.assertIsNone(pool.proofs)
        proofs=pool.enable_proofs(endpoints=0,workers=1,queue=1,table_mb=1)
        proofs.close()
        self.assertIsNone(pool.proofs)

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
        graphs = [self.graph(self.opening) for _ in range(19)]
        pool = self.pool(graphs, quantum=16, views=1, work=4096)
        proofs = self.loop(pool, workers=2, queue=8)
        pool.step()
        self.assertEqual(self.answer(pool), 1)  # Identical neural contexts coalesce across games.
        proofs.step()
        submitted = proofs.stats()['submitted']
        self.assertEqual(submitted, 8)
        # Workers consume each refill without polling. More games than queue slots
        # exercise cursor rotation and selection of the remaining eligible games.
        self.wait(lambda: proofs.stats()['finished'] == submitted)
        self.assertEqual(proofs.stats()['ready'], 8)
        proofs.step()
        self.assertEqual(proofs.stats()['submitted'], 16)
        self.wait(lambda: proofs.stats()['finished'] == 16)
        proofs.step()
        self.assertEqual(proofs.stats()['submitted'], 19)
        self.wait(lambda: proofs.stats()['finished'] == 19)
        proofs.step()
        self.assertEqual(proofs.stats()['installed'], 19)
        self.assertEqual({native.hxg_exact(g.ptr) for g in graphs}, {0})
        records = proofs.records()
        self.assertEqual(len({r['id'] for r in records}), 19)
        self.assertEqual({r['game'] for r in records}, set(range(19)))
        for row in records:
            self.assertEqual(row['request']['history'], self.opening)
            self.assertEqual(independent_verify(row['result']['certificate'], self.opening,
                             known=row['request']['known']), 'PROVEN_WIN')

    def test_refill_explores_an_older_low_priority_position_without_duplicates(self):
        import ctypes as C
        import json
        from neural_search import bind, checked, ptr
        bind('hxpe_new', ptr, ptr, *([C.c_int]*7))
        bind('hxpe_take', C.c_uint64, ptr, C.c_int)
        bind('hxpe_request', C.c_char_p, ptr, C.c_int)
        bind('hxpe_complete', C.c_int, ptr, C.c_int, C.c_uint64, ptr, ptr, C.c_int, C.c_char_p, C.c_char_p)
        graph=self.graph([[0,0]])
        starts=[[[0,0],[q,r],[q+1,r]] for q,r in
                ((4,0),(4,1),(4,-1),(3,1),(3,-1),(2,2),(2,-2),(1,3),(1,-3))]
        for history in starts:
            view=graph.view(history);self.addCleanup(view.close)
            view.search(1,root_samples=1,batch_size=1)
        pool=self.pool([graph],quantum=4,views=1,work=4096)
        loop=native.hxpe_new(pool.ptr,1,8,8,1,64,0,64)
        self.assertTrue(loop)
        try:
            for i,history in enumerate(starts):
                cells=np.asarray(history,np.int64)
                checked(native.hxp_offer(loop,0,cells.ctypes.data,len(cells),.01 if i<2 else i+1.))
            checked(native.hxp_step(loop))
            seen=[];ids=[]
            while job:=native.hxpe_take(loop,0):
                self.assertNotEqual(job,2**64-1)
                row=json.loads(native.hxpe_request(loop,0));seen.append(row['history']);ids.append(job)
                info=np.zeros(13,np.uint64);info[4]=1
                checked(native.hxpe_complete(loop,0,job,info.ctypes.data,None,0,None,None))
            self.assertEqual(seen[:4],list(reversed(starts[5:])))
            self.assertEqual(seen[4],starts[0])
            self.assertEqual(len(ids),8)
            self.assertEqual(len(set(ids)),8)
            self.assertEqual(len({tuple(map(tuple,h)) for h in seen}),8)
        finally:
            native.hxp_cancel(loop);checked(native.hxp_drain(loop));checked(native.hxp_free(loop))

    def quiet_closed_frontier(self, count=8, prepare=None):
        """A full frontier of quiet positions whose mover and defender searches are both closed.

        `prepare(graph)` runs before the pool takes the graph over."""
        graph = self.graph([[0,0]])
        if prepare:
            prepare(graph)
        pool = self.pool([graph], quantum=4, views=1, work=4096)
        proofs = pool.enable_proofs(slice_ms=50, table_mb=1, workers=2, queue=4, tasks=count)
        for k in range(1, count+1):
            proofs.offer(0, [[0,0],[k,-1],[k,1]])
        def closed():
            proofs.step()
            return proofs.stats()['scope']['closed_scopes'] == 2*count
        self.wait(closed)
        return graph, pool, proofs

    def test_only_unstamped_loops_skip_the_quiet_defender(self):
        for stamps in (False, True):
            graph = self.graph([[0,0]])
            pool = self.pool([graph], quantum=4, views=1, work=4096)
            proofs = pool.enable_proofs(slice_ms=50, table_mb=1, workers=1, queue=1, tasks=8, stamps=stamps)
            proofs.offer(0, [[0,0],[1,-1],[1,1]])
            proofs.step()
            # Stamped solvers search quiet defender zones; unstamped ones cannot start there.
            self.assertEqual(proofs.stats()['scope']['closed_scopes'], 0 if stamps else 1)
            proofs.drain();proofs.close();pool.close();graph.close()

    def test_closed_positions_give_way_to_new_offers(self):
        graph, pool, proofs = self.quiet_closed_frontier()
        before = proofs.stats()
        self.assertEqual(before['tasks'], 8)
        # Far weaker than any retained entry, yet only closed entries stand in its way.
        proofs.offer(0, [[0,0],[-3,1],[-3,2]], relevance=1e-6)
        proofs.step()
        after = proofs.stats()
        self.assertEqual(after['tasks'], 8)
        self.assertEqual(after['submitted'], before['submitted']+1)
        self.assertEqual(after['supply_first_queries'], before['supply_first_queries']+1)
        proofs.drain()

    def test_a_new_fact_keeps_closed_positions_until_their_scope_is_refreshed(self):
        won = [[0,0],[0,8],[2,8],[1,0],[2,0],[4,8],[6,8],[3,0],[4,0],[8,8],[10,8]]
        def prepare(graph):
            view = graph.view(won)
            self.addCleanup(view.close)
            self.assertTrue(native.hxg_tactics(view.ptr, 1))
            view.search(1, root_samples=1, batch_size=1)
            self.assertEqual(native.hxg_exact(view.ptr), 0)
        graph, pool, proofs = self.quiet_closed_frontier(prepare=prepare)
        proofs.offer(0, won)
        before = proofs.stats()
        self.assertEqual(before['facts'], 1)
        # Offers are counted; this repeat keeps the weak offer below off the periodic admission.
        proofs.offer(0, [[0,0],[1,-1],[1,1]])
        # The fact may reopen any closed entry, so none of them is disposable yet.
        proofs.offer(0, [[0,0],[-3,1],[-3,2]], relevance=1e-6)
        proofs.step()
        after = proofs.stats()
        self.assertEqual(after['tasks'], 8)
        self.assertEqual(after['supply_first_queries'], before['supply_first_queries'])
        proofs.drain()

    def test_idle_workers_are_charged_to_the_refill_that_left_them_idle(self):
        import time
        graph = self.graph([[0,0]])
        pool = self.pool([graph], quantum=4, views=1, work=4096)
        empty = pool.enable_proofs(slice_ms=50, table_mb=1, workers=2, queue=4, tasks=8)
        empty.step();time.sleep(.05)
        stats = empty.stats()
        self.assertGreaterEqual(stats['idle_empty_ms'], 2*45)
        self.assertEqual([stats[k] for k in ('idle_capacity_ms','idle_held_ms','idle_pending_ms','idle_closed_ms','idle_dormant_ms')], [0]*5)
        empty.close()
        graph, pool, proofs = self.quiet_closed_frontier()
        stats = proofs.stats()
        self.assertEqual(stats['supply_seen'], stats['supply_eligible']+stats['supply_pending']
                         +stats['supply_closed']+stats['supply_dormant'])
        self.assertEqual(stats['supply_first_queries'], 8)
        self.assertEqual(stats['submitted'], 8)
        before = stats['idle_closed_ms']
        time.sleep(.05);proofs.step();time.sleep(.05)
        stats = proofs.stats()
        self.assertGreaterEqual(stats['idle_closed_ms']-before, 2*90)
        self.assertEqual(stats['submitted'], 8)
        proofs.drain()

    def test_retries_take_only_idle_workers_and_never_queue_ahead_of_fresh_work(self):
        import ctypes as C
        import json
        from neural_search import bind, checked, ptr
        bind('hxpe_new', ptr, ptr, *([C.c_int]*7))
        bind('hxpe_take', C.c_uint64, ptr, C.c_int)
        bind('hxpe_request', C.c_char_p, ptr, C.c_int)
        bind('hxpe_complete', C.c_int, ptr, C.c_int, C.c_uint64, ptr, ptr, C.c_int, C.c_char_p, C.c_char_p)
        first,second,fresh=[[0,0],[4,0],[5,0]],[[0,0],[4,0],[5,1]],[[0,0],[4,0],[5,0],[6,1]]
        graph=self.graph(first)
        graph.search(1,root_samples=1,batch_size=1)
        pool=self.pool([graph],quantum=4,views=1,work=4096)
        loop=native.hxpe_new(pool.ptr,1,4,10,1,64,0,1000)
        self.assertTrue(loop)
        def offer(moves):
            cells=np.asarray(moves,np.int64)
            checked(native.hxp_offer(loop,0,cells.ctypes.data,len(cells),1.))
        def take():
            job=native.hxpe_take(loop,0)
            self.assertNotIn(job,(0,2**64-1))
            return job,json.loads(native.hxpe_request(loop,0))['history']
        def unknown(job,history):
            info=np.zeros(13,np.uint64);info[4]=1;info[5]=7;info[10]=2 if len(history)%2 else 1;info[12]=1
            checked(native.hxpe_complete(loop,0,job,info.ctypes.data,None,0,None,None))
        try:
            offer(first);offer(second)
            checked(native.hxp_step(loop))
            job,history=take()
            unknown(job,history)
            # The other fresh task is still queued, so the free worker has work and
            # the cooling retry stays out of the queue.
            checked(native.hxp_step(loop))
            job,other=take()
            self.assertNotEqual(other,history)
            offer(fresh)
            unknown(job,other)
            checked(native.hxp_step(loop))
            job,history=take()
            self.assertEqual(history,fresh)
            unknown(job,history)
            self.assertEqual(native.hxpe_take(loop,0),0)
            # Three retries are due later; the one idle worker continues one of them.
            checked(native.hxp_step(loop))
            job,history=take()
            unknown(job,history)
            self.assertEqual(native.hxpe_take(loop,0),0)
        finally:
            native.hxp_cancel(loop);checked(native.hxp_drain(loop));checked(native.hxp_free(loop))

    def test_unknown_task_waits_behind_fresh_work_then_continues_with_a_larger_slice(self):
        import ctypes as C
        import json
        from neural_search import bind, checked, ptr
        bind('hxpe_new', ptr, ptr, *([C.c_int]*7))
        bind('hxpe_take', C.c_uint64, ptr, C.c_int)
        bind('hxpe_request', C.c_char_p, ptr, C.c_int)
        bind('hxpe_complete', C.c_int, ptr, C.c_int, C.c_uint64, ptr, ptr, C.c_int, C.c_char_p, C.c_char_p)
        history=[[0,0],[4,0],[5,0]]
        fresh=history+[[6,1]]
        graph=self.graph(history)
        graph.search(1,root_samples=1,batch_size=1)
        pool=self.pool([graph],quantum=4,views=1,work=4096)
        loop=native.hxpe_new(pool.ptr,2,4,10,1,64,0,1000)
        self.assertTrue(loop)
        def offer(moves):
            cells=np.asarray(moves,np.int64)
            checked(native.hxp_offer(loop,0,cells.ctypes.data,len(cells),1.))
        def take(worker):
            job=native.hxpe_take(loop,worker)
            self.assertNotIn(job,(0,2**64-1))
            return job,json.loads(native.hxpe_request(loop,worker))
        def unknown(worker,job,side,stones):
            # A searched UNKNOWN: nodes were used, no disproof.
            info=np.zeros(13,np.uint64);info[4]=1;info[5]=7;info[9]=0;info[10]=2 if stones%2 else 1;info[11]=side;info[12]=1
            checked(native.hxpe_complete(loop,worker,job,info.ctypes.data,None,0,None,None))
        try:
            offer(history)
            checked(native.hxp_step(loop))
            first,request=take(0)
            self.assertEqual((request['history'],request['ms']),(history,10))
            unknown(0,first,0,3)
            offer(fresh)
            checked(native.hxp_step(loop))
            # The retry is due later, so the fresh position goes first; the idle
            # second worker still continues the retry now, with a doubled slice.
            # The quiet defender has nothing to search, so the retry stays with the mover.
            job,request=take(0)
            self.assertEqual(request['history'],fresh)
            retry,again=take(1)
            self.assertEqual((again['history'],again['attacker'],again['ms']),(history,'mover',20))
            unknown(0,job,0,4);unknown(1,retry,0,3)
            self.assertEqual(native.hxg_exact(graph.ptr),-1)
        finally:
            native.hxp_cancel(loop);checked(native.hxp_drain(loop));checked(native.hxp_free(loop))

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
            stats = proofs.stats()
            return stats['unknown'] >= 1 and stats['scope']['closed_scopes'] >= 2
        self.wait(tried_both)
        self.assertEqual(native.hxg_exact(graph.ptr), -1)
        self.assertTrue(pool.games[0].evidence()['eligible'].all())
        self.assertEqual(proofs.records(), [])

    def test_unrelated_facts_preserve_closed_search_and_relevant_facts_reopen_it(self):
        history = [[0,0],[1,2],[3,-1]]
        graph = self.graph(history)
        graph.search(4, root_samples=4, batch_size=4)
        unrelated = [[0,0],[0,8],[2,8],[1,0],[2,0],[4,8],[6,8],
                     [3,0],[4,0],[8,8],[10,8]]
        wrong_color = [[0,0],[3,-1],[0,8],[1,0],[2,0],[2,8],[4,8],
                       [3,0],[4,0],[6,8],[8,8],[1,2],[0,1],[10,8],[12,8]]
        relevant = history+[[1,0],[2,0],[0,8],[2,8],[3,0],[4,0],[4,8],[6,8]]
        for moves in (unrelated, wrong_color, relevant):
            view = graph.view(moves)
            self.addCleanup(view.close)
            self.assertTrue(native.hxg_tactics(view.ptr, 1))
            view.search(1, root_samples=1, batch_size=1)
            self.assertEqual(native.hxg_exact(view.ptr), 0)
        pool = self.pool([graph], quantum=4, views=1, work=4096)
        proofs = pool.enable_proofs(slice_ms=250, table_mb=1, workers=1, queue=1, tasks=8)
        def closed():
            proofs.step()
            stats = proofs.stats()
            return stats['unknown'] >= 1 and stats['scope']['closed_scopes'] == 2
        # The mover's forcing search is disproved; the quiet defender has no threat to search,
        # so it is never queried.
        self.wait(closed)
        before = proofs.stats()
        self.assertEqual(before['unknown'], 1)
        for moves in (unrelated, wrong_color):
            proofs.offer(0, moves)
            proofs.step()
            stats = proofs.stats()
            self.assertEqual(stats['submitted'], before['submitted'])
            self.assertEqual(stats['scope']['closed_scopes'], 2)
        self.assertEqual(proofs.stats()['scope']['unchanged']-before['scope']['unchanged'], 2)
        proofs.offer(0, relevant)
        proofs.step()
        after = proofs.stats()
        # Relevant facts reopen the mover; the quiet defender stays closed under any premises.
        self.assertEqual(after['scope']['closed_scopes'], 1)
        self.assertEqual(after['submitted'], before['submitted']+1)
        self.assertEqual(after['scope']['sent_facts']-before['scope']['sent_facts'], 1)
        # Retrying a changed premise set keeps the effort already invested in this task.
        self.assertEqual(after['scope']['quantum_ms']-before['scope']['quantum_ms'], 500)
        proofs.drain()
        self.assertEqual(native.hxg_exact(graph.ptr), -1)
        self.assertTrue(pool.games[0].evidence()['eligible'].all())

    def test_evicted_fact_membership_does_not_leak_into_later_jobs(self):
        from itertools import combinations
        history = [[0,0],[1,2],[3,-1]]
        graph = self.graph(history)
        graph.search(4, root_samples=4, batch_size=4)
        marker = history+[[1,0],[2,0],[0,8]]
        view = graph.view(marker)
        self.addCleanup(view.close)
        view.search(4, root_samples=4, batch_size=4)
        pairs = [([0,8],[2,8])]+list(combinations(
            [[q,r] for q in range(-4,5) for r in range(4,8) if [q,r] not in ([2,7],[3,7])], 2))[:256]
        facts = []
        for a,b in pairs:
            moves = history+[[1,0],[2,0],a,b,[3,0],[4,0],[2,7],[3,7]]
            child = graph.view(moves)
            self.addCleanup(child.close)
            self.assertTrue(native.hxg_tactics(child.ptr, 1))
            child.search(1, root_samples=1, batch_size=1)
            self.assertEqual(native.hxg_exact(child.ptr), 0)
            facts.append(moves)
        pool = self.pool([graph], quantum=4, views=1, work=4096)
        proofs = pool.enable_proofs(slice_ms=40, table_mb=1, workers=1, queue=1, tasks=8)
        for moves in facts:
            proofs.offer(0, moves)
        self.assertEqual(proofs.stats()['facts'], 256)
        proofs.offer(0, marker, relevance=10.)
        proofs.step()
        stats = proofs.stats()
        self.assertEqual(stats['submitted'], 1)
        self.assertEqual(stats['scope']['available_facts'], 256)
        self.assertEqual(stats['scope']['sent_facts'], 0)
        self.assertEqual(stats['scope']['empty_jobs'], 1)
        proofs.drain()

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
        proofs.drain()
        effort = proofs.effort(0)
        self.assertEqual(set(effort), {1,2})
        self.assertEqual(sum(r['fresh_nodes'] for r in effort.values()), proofs.stats()['fresh_nodes'])
        self.assertEqual(sum(r['missing_fresh'] for r in effort.values()), proofs.stats()['missing_fresh'])
        self.assertEqual(sum(r['queries'] for r in effort.values()), proofs.stats()['finished'])
        self.assertEqual(effort[1]['queries'], stats['finished'])

        # A completed UNKNOWN continuation can wait without a graph pin. A new
        # root discards that old-generation estimate before it reaches the GPU.
        history=[[0,0],[4,0],[7,0],[-1,0],[-2,0]]
        pending_pool=self.pool([self.graph(history)],quantum=16,views=8,depth=1,work=1024)
        pending=pending_pool.enable_proofs(slice_ms=8,table_mb=1,workers=1,queue=1,endpoints=8)
        pending_pool.step();self.answer(pending_pool);pending.step()
        self.wait(lambda:pending.stats()['finished']>0)
        pending_pool.step(neural=False)
        self.assertGreater(pending.stats()['neural_frontier']['queue']['pending'],0)
        self.assertEqual(pending.stats()['neural_frontier']['candidates'],0)
        pending_pool.retarget(0,[[0,0],[1,2],[3,-1]],work=1024)
        self.assertEqual(pending.stats()['neural_frontier']['queue']['pending'],0)
        self.assertGreater(pending.stats()['neural_frontier']['queue']['obsolete'],0)
        self.assertEqual(native.hxg_exact(native.hxgo_root(pending_pool.games[0].ptr)),-1)
        pending_pool.cancel();pending.drain();pending_pool.abandon_fenced()
        self.assertEqual(pending_pool.games[0].stats()['pending'],0)

    def shared(self, **options):
        from native_scheduler import ProofWorkers
        workers = ProofWorkers(**options)
        self.addCleanup(workers.close)  # Registered before any pool, so it closes last.
        return workers

    def offers(self, loop, count, game=0):
        for k in range(1, count+1):
            loop.offer(game, [[0,0],[k%8+1,-1-k//8],[k%8+1,1+k//8]])

    def test_one_shared_worker_serves_every_producer_and_answers_wait_for_their_owner(self):
        from tactical_proof import independent_verify
        workers = self.shared(workers=1, queue=8)
        pools = [self.pool([self.graph(self.opening) for _ in range(3)], quantum=16, views=1, work=4096)
                 for _ in range(2)]
        loops = [self.loop(pool, shared=workers) for pool in pools]
        for pool, loop in zip(pools, loops):
            pool.step();self.answer(pool);loop.step()
        submitted = [loop.stats()['submitted'] for loop in loops]
        self.assertEqual(submitted, [3, 3])
        self.wait(lambda: [loop.stats()['finished'] for loop in loops] == submitted)
        # Every answer is back, yet a producer's graphs change only on its own owner's step.
        loops[1].step()
        self.assertEqual([native.hxg_exact(native.hxgo_root(g.ptr)) for g in pools[0].games], [-1]*3)
        self.assertEqual([native.hxg_exact(native.hxgo_root(g.ptr)) for g in pools[1].games], [0]*3)
        loops[0].step()
        self.assertEqual([native.hxg_exact(native.hxgo_root(g.ptr)) for g in pools[0].games], [0]*3)
        for loop in loops:
            stats, records = loop.stats(), loop.records()
            self.assertEqual((stats['started'], stats['finished'], stats['installed']), (3, 3, 3))
            self.assertEqual(sorted(r['game'] for r in records), [0, 1, 2])
            self.assertEqual(len({r['id'] for r in records}), 3)
            for row in records:
                self.assertEqual(independent_verify(row['result']['certificate'], row['request']['history'],
                                 known=row['request']['known']), 'PROVEN_WIN')
        stats = workers.stats()
        self.assertEqual((stats['queued'], stats['live'], stats['active'], stats['loops']), (0, 0, 0, 2))
        self.assertAlmostEqual(stats['worker_service_ms'], sum(l.stats()['worker_service_ms'] for l in loops))

    def test_shared_queue_splits_between_producers_with_work(self):
        workers = self.shared(workers=1, queue=4)
        pools = [self.pool([self.graph([[0,0]])], quantum=4, views=1, work=4096) for _ in range(2)]
        first, second = (pool.enable_proofs(slice_ms=50, table_mb=1, tasks=32, shared=workers) for pool in pools)
        self.offers(first, 20);first.step()
        # Alone with work, one producer may hold the whole queue.
        self.assertEqual(first.stats()['submitted'], 4)
        self.offers(second, 20);second.step()
        self.assertEqual((second.stats()['submitted'], second.stats()['supply_full_exits']), (0, 1))
        self.wait(lambda: first.stats()['finished'] == 4)
        first.step();second.step()
        self.assertEqual((first.stats()['submitted'], second.stats()['submitted']), (6, 2))
        for loop in (first, second):
            loop.drain()
        self.assertEqual(workers.stats()['live'], 0)

    def held_workers(self, capacity, workers=1, gate=None):
        """A raw shared service whose answers wait for `release`, or for `gate(history)`'s event.

        Create the pools first: cleanups free every joined loop, then the service."""
        import ctypes as C
        import json
        import threading
        from neural_search import checked, ptr
        from tactical_proof import NativeTactics
        library = NativeTactics(independent=True)
        self.addCleanup(library.close)
        entered, release, order = threading.Event(), threading.Event(), []
        actual = library.lib.hexo_tactical_worker_answer
        actual.argtypes, actual.restype = [ptr, C.c_char_p], ptr
        @C.CFUNCTYPE(ptr, ptr, C.c_char_p)
        def query(worker, request):
            history = json.loads(request)['history']
            order.append(history)
            entered.set();(gate(history) if gate else release).wait(5)
            return actual(worker, request)
        names = ('worker_new', 'worker_free', 'worker_answer', 'answer_info', 'answer_moves',
                 'answer_json', 'answer_free', 'free', 'prepare', 'cancel', 'release', 'worker_busy')
        functions = np.asarray([C.cast(query if name == 'worker_answer' else
                                getattr(library.lib, 'hexo_tactical_'+name), ptr).value for name in names], np.uint64)
        service = native.hxps_new(functions.ctypes.data, workers, capacity)
        self.assertTrue(service)
        self.addCleanup(lambda: (query, checked(native.hxps_free(service))))
        def join(pool):
            loop = native.hxp_join(pool.ptr, service, 50, 1, 32, 0)
            self.assertTrue(loop)
            self.addCleanup(lambda: (release.set(), native.hxp_cancel(loop), checked(native.hxp_drain(loop)),
                                     checked(native.hxp_free(loop))))
            return loop
        return service, join, entered, release, order

    def test_shared_workers_take_turns_between_producers_with_queued_work(self):
        from neural_search import checked
        pools = [self.pool([self.graph([[0,0]])], quantum=4, views=1, work=4096) for _ in range(2)]
        service, join, entered, release, order = self.held_workers(16)
        loops = [join(pool) for pool in pools]
        sides = ([[0,0],[k,-1],[k,1]] for k in range(1, 9)), ([[0,0],[-k,1],[-k,2]] for k in range(1, 9))
        # The first producer's positions rank far above the second's and it refills first.
        for loop, histories, relevance in zip(loops, sides, (100., .01)):
            for history in histories:
                cells = np.asarray(history, np.int64)
                checked(native.hxp_offer(loop, 0, cells.ctypes.data, len(cells), relevance))
            checked(native.hxp_step(loop))
            self.assertTrue(entered.wait(2))
        release.set()
        self.wait(lambda: len(order) == 16)
        self.assertEqual([0 if h[1][1] == -1 else 1 for h in order], [0]+[1, 0]*7+[1])

    def test_a_producer_with_long_jobs_leaves_other_workers_to_other_producers(self):
        from neural_search import checked
        pools = [self.pool([self.graph([[0,0]])], quantum=4, views=1, work=4096) for _ in range(2)]
        import threading
        start = threading.Event()
        # The first producer's answers hold their workers; the second's wait only for `start`.
        service, join, entered, release, order = self.held_workers(
            16, workers=2, gate=lambda h: start if h[1][1] == 1 else release)
        loops = [join(pool) for pool in pools]
        sides = ([[0,0],[k,-1],[k,1]] for k in range(1, 9)), ([[0,0],[-k,1],[-k,2]] for k in range(1, 9))
        for loop, histories in zip(reversed(loops), reversed(list(sides))):
            for history in histories:
                cells = np.asarray(history, np.int64)
                checked(native.hxp_offer(loop, 0, cells.ctypes.data, len(cells), 1.))
            checked(native.hxp_step(loop))
        self.wait(lambda: len(order) == 2)
        start.set()
        self.wait(lambda: sum(h[1][1] == 1 for h in order) == 8)
        # One held job of the first producer at a time; the other worker runs the second's queue dry.
        last = max(i for i, h in enumerate(order) if h[1][1] == 1)
        self.assertEqual(sum(h[1][1] == -1 for h in order[:last]), 1)

    def test_a_producer_that_stops_refilling_gives_back_its_queue_share(self):
        import time
        from neural_search import checked
        pools = [self.pool([self.graph([[0,0]])], quantum=4, views=1, work=4096) for _ in range(2)]
        service, join, entered, release, order = self.held_workers(4)
        loops = [join(pool) for pool in pools]
        stats, times = np.empty(16, np.uint64), np.empty(5, np.float64)
        def submitted(loop):
            native.hxp_stats(loop, stats.ctypes.data, times.ctypes.data)
            return int(stats[1])
        for i, loop in enumerate(loops):
            for k in range(1, 9):
                cells = np.asarray([[0,0],[k,-1-i],[k,1+i]], np.int64)
                checked(native.hxp_offer(loop, 0, cells.ctypes.data, len(cells), 1.))
            checked(native.hxp_step(loop))
        # The first producer holds the whole queue; the second found it full and then stops stepping.
        self.assertEqual([submitted(loop) for loop in loops], [4, 0])
        release.set()
        def answered():
            native.hxp_stats(loops[0], stats.ctypes.data, times.ctypes.data)
            return int(stats[3]) == 4
        self.wait(answered)
        time.sleep(.1)
        checked(native.hxp_step(loops[0]))
        self.assertEqual(submitted(loops[0]), 8)

    def test_external_workers_report_their_idle_time(self):
        import ctypes as C
        import time
        from neural_search import bind, checked, ptr
        bind('hxpe_new', ptr, ptr, *([C.c_int]*7))
        pool = self.pool([self.graph([[0,0]])], quantum=4, views=1, work=4096)
        loop = native.hxpe_new(pool.ptr, 2, 4, 10, 1, 64, 0, 64)
        self.assertTrue(loop)
        try:
            checked(native.hxp_step(loop))
            time.sleep(.05)
            stats, times = np.empty(16, np.uint64), np.empty(5, np.float64)
            native.hxp_stats(loop, stats.ctypes.data, times.ctypes.data)
            self.assertGreaterEqual(times[1], 2*45)
        finally:
            native.hxp_cancel(loop);checked(native.hxp_drain(loop));checked(native.hxp_free(loop))

    def test_concurrent_producer_refills_never_exceed_the_shared_bound(self):
        import threading
        from neural_search import checked
        pools = [self.pool([self.graph([[0,0]])], quantum=4, views=1, work=4096) for _ in range(3)]
        service, join, entered, release, order = self.held_workers(4)
        loops = [join(pool) for pool in pools]
        for i, loop in enumerate(loops):
            for k in range(1, 9):
                cells = np.asarray([[0,0],[k,-1-i],[k,1+i]], np.int64)
                checked(native.hxp_offer(loop, 0, cells.ctypes.data, len(cells), 1.))
        # Owners refill at once while the worker holds its first answer, so no slot frees up.
        start = threading.Barrier(len(loops))
        def refill(loop):
            start.wait()
            for _ in range(50):
                checked(native.hxp_step(loop))
        threads = [threading.Thread(target=refill, args=(loop,)) for loop in loops]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        out, times = np.empty(5, np.uint64), np.empty(5, np.float64)
        native.hxps_stats(service, out.ctypes.data, times.ctypes.data)
        self.assertEqual(int(out[3]), 4)
        submitted, stats = 0, np.empty(16, np.uint64)
        for loop in loops:
            native.hxp_stats(loop, stats.ctypes.data, times.ctypes.data)
            submitted += int(stats[1])
        self.assertEqual(submitted, 4)

    def test_retiring_a_game_or_producer_cancels_only_its_jobs_on_shared_workers(self):
        workers = self.shared(workers=1, queue=16)
        pools = [self.pool([self.graph([[0,0]])], quantum=4, views=1, work=4096) for _ in range(3)]
        loops = [pool.enable_proofs(slice_ms=50, table_mb=1, tasks=32, shared=workers) for pool in pools]
        for loop in loops:
            self.offers(loop, 5);loop.step()
        self.assertEqual([loop.stats()['submitted'] for loop in loops], [5, 5, 5])
        pools[0].retarget(0, [[0,0],[1,2],[3,-1]], work=4096)
        retired = loops[0].stats()
        self.assertEqual((retired['queued'], retired['cancelled']), (0, 5))
        loops[1].close()
        self.assertEqual(workers.stats()['loops'], 2)
        survivor = loops[2]
        self.wait(lambda: survivor.stats()['finished'] == 5)
        self.assertEqual(survivor.stats()['cancelled'], 0)
        loops[0].drain();survivor.step();survivor.drain()
        self.assertEqual(survivor.stats()['unknown']+survivor.stats()['installed'], 5)
        stats = workers.stats()
        self.assertEqual((stats['queued'], stats['live'], stats['active']), (0, 0, 0))

    def test_owner_over_its_proof_budget_installs_answers_but_admits_nothing(self):
        import time
        pool = self.pool([self.graph([[0,0]])], quantum=4, views=1, work=4096)
        for budget in (0., 1.5):
            with self.assertRaisesRegex(ValueError, 'budget'):
                pool.enable_proofs(slice_ms=50, table_mb=1, workers=1, owner_budget=budget)
            self.assertIsNone(pool.proofs)
        proofs = pool.enable_proofs(slice_ms=50, table_mb=1, workers=1, queue=4, tasks=32, owner_budget=1e-6)
        self.offers(proofs, 8)
        proofs.step()
        self.assertEqual(proofs.stats()['submitted'], 4)
        self.wait(lambda: proofs.stats()['finished'] == 4)
        # That step's own work puts the owner over a tiny budget: answers install, nothing new goes out.
        proofs.step()
        stats = proofs.stats()
        self.assertEqual((stats['submitted'], stats['ready'], stats['unknown']+stats['installed']), (4, 0, 4))
        self.assertEqual(stats['supply_owner_exits'], 1)
        time.sleep(.05)
        self.assertGreaterEqual(proofs.stats()['idle_owner_ms'], 45)
        # The load decays over about a second while the owner does other work, and admission resumes.
        time.sleep(2)
        proofs.step()
        self.assertGreater(proofs.stats()['submitted'], 4)
        proofs.drain()

    def test_shared_idle_time_goes_to_the_producer_held_back(self):
        import time
        workers = self.shared(workers=2, queue=8)
        pools = [self.pool([self.graph([[0,0]])], quantum=4, views=1, work=4096) for _ in range(2)]
        empty = pools[0].enable_proofs(slice_ms=50, table_mb=1, tasks=8, shared=workers)
        held = pools[1].enable_proofs(slice_ms=50, table_mb=1, tasks=32, shared=workers, owner_budget=1e-6)
        empty.step()
        self.offers(held, 8)
        held.step()
        self.wait(lambda: held.stats()['finished'] == held.stats()['submitted'] > 0)
        held.step()
        before = empty.stats()['idle_empty_ms'], held.stats()['idle_owner_ms']
        time.sleep(.05)
        after = empty.stats()['idle_empty_ms'], held.stats()['idle_owner_ms']
        # Workers wait on the producer over its owner budget, not on the one without candidates.
        self.assertGreaterEqual(after[1]-before[1], 2*45)
        self.assertEqual(after[0], before[0])
        for loop in (held, empty):
            loop.drain()

    def test_shared_workers_close_after_their_loops_and_split_idle_time(self):
        import time
        from native_scheduler import ProofWorkers
        workers = ProofWorkers(workers=2, queue=4)
        pools = [self.pool([self.graph([[0,0]])], quantum=4, views=1, work=4096) for _ in range(2)]
        with self.assertRaisesRegex(ValueError, 'fix the package'):
            pools[0].enable_proofs(workers=2, shared=workers)
        self.assertIsNone(pools[0].proofs)
        loops = [pool.enable_proofs(slice_ms=50, table_mb=1, tasks=8, shared=workers) for pool in pools]
        for loop in loops:
            loop.step()
        time.sleep(.05)
        idle = [loop.stats()['idle_empty_ms'] for loop in loops]
        # Two idle workers, charged half to each loop whose refill found nothing.
        self.assertGreaterEqual(min(idle), 45)
        self.assertLessEqual(sum(idle), workers.stats()['worker_idle_ms']+1e-6)
        with self.assertRaisesRegex(ValueError, 'Free every proof loop'):
            workers.close()
        for loop in loops:
            loop.close()
        workers.close()
        with self.assertRaisesRegex(ValueError, 'closed'):
            workers.stats()

    def test_attached_loop_prevents_pool_free_and_duplicate_owners(self):
        graph = self.graph()
        pool = self.pool([graph], quantum=16, work=32)
        proofs = self.loop(pool, workers=1, queue=4)
        with self.assertRaisesRegex(ValueError, 'already has a proof loop'):
            pool.enable_proofs()
        self.assertEqual(native.hxgm_free(pool.ptr), 0)
        self.assertIn(b'Close the proof loop', native.hxg_error())
        raw=proofs.ptr
        service=native.hxb_new(16,2,0,0.)
        self.assertTrue(service)
        try:
            self.assertTrue(native.hxb_attach(service,pool.ptr,0))
            self.assertEqual(native.hxp_free(raw),0)
            self.assertIn(b'Detach the native inference service',native.hxg_error())
        finally:
            self.assertTrue(native.hxb_free(service))
        proofs.close()
        pool.close()
        with self.assertRaisesRegex(ValueError, 'closed'):
            proofs.stats()

    def test_native_service_retires_a_neural_lease_when_real_solver_finishes(self):
        for feedback in (False,True):
            with self.subTest(feedback=feedback):
                import ctypes as C
                import json
                import threading
                from neural_search import checked,ptr,bind
                from tactical_proof import NativeTactics,independent_verify
                bind('hxgp_free',None,ptr)
                graph=self.graph(self.opening)
                graph.search(1,root_samples=1,batch_size=1)
                pool=self.pool([graph],quantum=16,views=4,work=4096)
                entered,release=threading.Event(),threading.Event()
                loop=service=None
                token,model,snapshot=C.c_uint64(),C.c_int(),ptr()
                with NativeTactics(independent=True) as library:
                    actual=library.lib.hexo_tactical_worker_answer
                    actual.argtypes,actual.restype=[ptr,C.c_char_p],ptr
                    # Hold the actual verified answer until a neural lease is in flight.
                    # This controls result ordering without inventing a game verdict.
                    @C.CFUNCTYPE(ptr,ptr,C.c_char_p)
                    def query(worker,request):
                        answer=actual(worker,request)
                        entered.set();release.wait(PATIENCE)
                        return answer
                    names=('worker_new','worker_free','worker_answer','answer_info','answer_moves',
                           'answer_json','answer_free','free','prepare','cancel','release','worker_busy')
                    functions=np.asarray([C.cast(query if name=='worker_answer' else
                                         getattr(library.lib,'hexo_tactical_'+name),ptr).value
                                          for name in names],np.uint64)
                    try:
                        loop=native.hxp_new(pool.ptr,functions.ctypes.data,1,4,1000,1,64,0)
                        self.assertTrue(loop)
                        service=native.hxb_new(16,2,0,0.)
                        self.assertTrue(service)
                        checked(native.hxb_feedback(service,feedback))
                        checked(native.hxb_attach(service,pool.ptr,0))
                        checked(native.hxb_start(service,0.))
                        self.assertTrue(entered.wait(PATIENCE))
                        self.assertGreater(take_owed(service,128,C.byref(token),
                                                           C.byref(model),C.byref(snapshot)),0)
                        release.set()
                        counters=np.empty(10,np.uint64)
                        def retired():
                            native.hxb_stats(service,counters.ctypes.data)
                            return counters[9]==0
                        self.wait(retired)
                        # No neural answer was installed: CPU evidence retired its lease.
                        self.assertEqual(native.hxb_installed(service),0)
                        checked(native.hxb_abort(service,token));native.hxgp_free(snapshot);token.value=0
                        checked(native.hxb_join(service));checked(native.hxb_free(service));service=None
                        self.assertEqual(native.hxg_exact(graph.ptr),0)
                        row=json.loads(native.hxp_record(loop,0))
                        self.assertEqual(independent_verify(row['result']['certificate'],self.opening,
                                                           known=row['request']['known']),'PROVEN_WIN')
                    finally:
                        release.set()
                        if service:
                            native.hxb_cancel(service)
                            if token.value:
                                native.hxb_abort(service,token);native.hxgp_free(snapshot)
                            checked(native.hxb_join(service));checked(native.hxb_free(service))
                        if loop:
                            checked(native.hxp_drain(loop));checked(native.hxp_free(loop))

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
