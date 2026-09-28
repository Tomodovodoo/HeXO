"""Solver points inside the dense searches (dense_solver): the native hold, mark and priority entries, each
injection point on positions with known forced wins, learner targets of proven rows and reproducibility of
seeded self-play with the solver on, across runs and backends. CPU only, tiny models."""
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import dense_config
import dense_data
import dense_eval
import dense_selfplay
from dense_solver import Budgets
import hexnet
from hexo import Game
from neural_search import HOLD, NeuralSearch, checked, native
from tactical_proof import NativeTactics
from tests.test_dense import episode_rows, winning_game
from tests.test_tactical_proof import FIXTURE, TWO_TURN

TINY = hexnet.HexNetConfig(blocks=2, channels=16, pool_every=2, line_length=5, value_hidden=16, head_channels=8)
PROOF = '1790600149713752:2:253'  # the side to move wins in 4 turns; proven within 135 nodes
NODES = 135


def tiny_model():
    torch.manual_seed(11)
    return dense_selfplay.Model(hexnet.HexNet(TINY).eval(), 'tiny', 'test', 'cpu', 64, 256)


def settings(**changes):
    base = dict(full_sims=8, cheap_sims=4, root_samples=4, max_plies=40, full_fraction=.5, opening_random_plies=3.,
                leaf_batch=64)
    return replace(dense_config.ActorSettings(), **{**base, **changes})


def run(slots, asynchronous=True):
    """Play `slots` to the end on one Engine."""
    engine = dense_selfplay.Engine(64, asynchronous)
    try:
        for slot in slots:
            engine.add(slot)
        while engine.slots:
            engine.step()
    finally:
        engine.close()


def from_position(game, opening):
    """`game` (a SelfPlayGame not yet on an Engine) moved to `opening` without rows, sampling no opening plies."""
    for q, r in opening:
        game.game.play(q, r)
        for tree in game.trees.values():
            tree.advance((q, r))
        game.moves.append([q, r])
    game.random_plies = 0
    return game


class Recorded(dense_eval.MatchGame):
    """A MatchGame keeping every search result it was given."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = []

    def searched(self, result):
        self.results.append(dict(result))
        return super().searched(result)


def match(model, opening, budgets, sims=16, samples=4, plies=1):
    return Recorded([model, model], opening, 3, sims, samples, True, len(opening)+plies, {}, solvers=(budgets, budgets))


def drive(search, budget, on_hold=lambda: None):
    """Finish a begun search of a bare tree with uniform zero evaluations, calling on_hold() and clearing the hold
    at each HOLD; returns hxg_completed at each hold."""
    holds = []
    while True:
        request = native.hxg_next(search.ptr)
        if request == HOLD:
            holds.append(native.hxg_completed(search.ptr))
            on_hold()
            checked(native.hxg_hold(search.ptr, 0))
        elif request > 0:
            legal = np.empty((native.hxg_legal(search.ptr, request, None), 2), np.int64)
            native.hxg_legal(search.ptr, request, legal.ctypes.data)
            zeros = np.zeros(len(legal))
            checked(native.hxg_fulfill(search.ptr, request, legal, zeros, zeros, len(legal)))
        elif native.hxg_completed(search.ptr) >= budget:
            return holds
        else:
            raise AssertionError('search stalled')


def stats(search):
    n = native.hxg_stats(search.ptr, None, None, None, None)
    actions, visits, values, scores = np.empty((n, 2), np.int64), np.empty(n, np.int32), np.empty(n), np.empty(n)
    native.hxg_stats(search.ptr, actions.ctypes.data, visits.ctypes.data, values.ctypes.data, scores.ctypes.data)
    policy = np.empty(n)
    native.hxg_policy(search.ptr, policy.ctypes.data)
    return actions.tolist(), visits, values, scores, policy


class NativeEntries(unittest.TestCase):
    def tree(self):
        search = NeuralSearch(object(), 'test', [(0, 0)], seed=5)
        self.addCleanup(search.close)
        return search

    def test_hold_stops_at_the_last_halving_boundary_or_the_end(self):
        for budget, samples, boundary in ((64, 16, 48), (16, 4, 8), (12, 12, 12)):
            search = self.tree()
            checked(native.hxg_begin(search.ptr, budget, samples))
            checked(native.hxg_hold(search.ptr, 1))
            self.assertEqual(drive(search, budget), [boundary], (budget, samples))

    def test_mark_exact_gives_q_minus_one_and_settles_the_root(self):
        search = self.tree()
        checked(native.hxg_begin(search.ptr, 8, 4))
        drive(search, 8)
        actions, _, _, _, _ = stats(search)
        mover = 1
        checked(native.hxg_mark_exact(search.ptr, *actions[0], 1-mover))
        actions, _, values, scores, policy = stats(search)
        self.assertEqual((values[0], policy[0], scores[0]), (-1., 0., -np.inf))
        self.assertEqual(native.hxg_exact(search.ptr), -1)
        self.assertFalse(native.hxg_mark_exact(search.ptr, 99, 99, 0))
        for action in actions:
            checked(native.hxg_mark_exact(search.ptr, *action, 1-mover))
        self.assertEqual(native.hxg_exact(search.ptr), 1-mover)
        self.assertTrue(np.all(stats(search)[4] > 0))   # a lost root keeps every edge eligible

    def test_marked_candidates_are_replaced_without_stalling(self):
        search = self.tree()
        marked = []

        def prune():
            actions, _, _, scores, _ = stats(search)
            for i in np.flatnonzero(np.isfinite(scores))[:3]:
                marked.append(actions[i])
                checked(native.hxg_mark_exact(search.ptr, *actions[i], 0))
        checked(native.hxg_begin(search.ptr, 16, 4))
        checked(native.hxg_hold(search.ptr, 1))
        self.assertEqual(drive(search, 16, prune), [8])
        self.assertEqual(len(marked), 3)
        actions, _, values, scores, _ = stats(search)
        chosen = actions[int(np.argmax(scores))]
        self.assertNotIn(chosen, marked)
        self.assertTrue(all(values[actions.index(a)] == -1 for a in marked))

    def test_priority_cells_are_sampled_first(self):
        plain, first = self.tree(), self.tree()
        checked(native.hxg_begin(plain.ptr, 8, 4))
        drive(plain, 8)
        actions, visits, _, _, _ = stats(plain)
        unsampled = [a for a, v in zip(actions, visits) if not v][-2:]
        checked(native.hxg_begin(first.ptr, 8, 4))
        checked(native.hxg_priority(first.ptr, np.asarray(unsampled, np.int64), 2))
        drive(first, 8)
        actions, visits, _, _, policy = stats(first)
        self.assertTrue(all(visits[actions.index(a)] > 0 for a in unsampled))
        self.assertTrue(np.all(policy > 0))


class InjectionPoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.engine = NativeTactics()
        except FileNotFoundError:
            raise unittest.SkipTest('Build tools/tactical with tools/build_tactical.py first')
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_root_proof_plays_the_certificate_turn_and_keeps_the_search_policy(self):
        opening = FIXTURE['positions'][PROOF]
        proof = self.engine.history(opening, nodes=NODES)
        self.assertEqual(proof['status'], 'PROVEN_WIN')
        model = tiny_model()
        on = from_position(dense_selfplay.SelfPlayGame([model, model], settings(
            full_fraction=1., max_plies=len(opening)+2, solver_root_nodes=NODES), 7), opening)
        off = from_position(dense_selfplay.SelfPlayGame([model, model], settings(
            full_fraction=1., max_plies=len(opening)+2), 7), opening)
        run([on, off])
        episode, rows = on.episode()
        self.assertEqual(episode['moves'][len(opening):], proof['moves'])
        self.assertEqual([(r['proven'], r['proof_turns']) for r in rows], [(1, proof['proof_turns'])]*2)
        self.assertEqual(rows[0]['solver_nodes'], proof['nodes_used'])
        self.assertEqual(episode['root_values'], [1., 1.])
        self.assertEqual(episode['solver'], dict(root_nodes=NODES, finalists=0, finalist_nodes=0, threat_nodes=0,
                                                 build_hash=self.engine.metadata['binary_sha256']))
        policy = rows[0]['policy']
        self.assertGreater(int((policy > 0).sum()), 1)
        self.assertLess(float(policy.max()), 1-1e-3)
        # The proof decides only the move: the search and its recorded policy are those of the solver-off game.
        _, plain = off.episode()
        np.testing.assert_array_equal(policy, plain[0]['policy'])
        self.assertNotIn('proven', plain[0])

    def test_finalist_proof_prunes_the_candidate(self):
        game = Game([tuple(m) for m in TWO_TURN])
        first = next(m for m in game.legal_moves() if abs(m[0])+abs(m[1]) > 9)
        game.close()
        opening = TWO_TURN+[list(first)]
        model = tiny_model()
        on, off = match(model, opening, Budgets(finalists=2, finalist_nodes=NODES)), match(model, opening, None)
        run([on, off])
        result, plain = on.results[0], off.results[0]
        self.assertEqual(len(result['pruned']), 2)
        actions = result['actions'].tolist()
        for action in result['pruned']:
            i = actions.index(action)
            self.assertEqual((result['values'][i], result['policy'][i]), (-1., 0.))
            self.assertGreater(plain['policy'][i], 0.)
            self.assertNotEqual(result['action'], action)
            after = self.engine.history(opening+[action], nodes=NODES)
            self.assertEqual(after['status'], 'PROVEN_WIN')
        self.assertGreater(result['solver_nodes'], 0)
        self.assertEqual(result['proven'], 0)   # other second stones stay unproven

    def test_threat_certificate_orders_the_root_samples(self):
        cells = self.engine.history(TWO_TURN, nodes=NODES, attacker='opponent')['moves']
        self.assertEqual(len(cells), 2)
        model = tiny_model()
        on, off = match(model, TWO_TURN, Budgets(threat_nodes=NODES), sims=8), match(model, TWO_TURN, None, sims=8)
        run([on, off])
        result, plain = on.results[0], off.results[0]
        actions = result['actions'].tolist()
        self.assertTrue(all(result['visits'][actions.index(c)] > 0 for c in cells))
        self.assertFalse(all(plain['visits'][actions.index(c)] > 0 for c in cells))
        np.testing.assert_array_equal(result['policy'] > 0, plain['policy'] > 0)   # ordering, never pruning

    def test_paired_games_give_each_side_its_budgets(self):
        config = dense_config.RunConfig()
        model = tiny_model()
        mine, theirs = Budgets(root_nodes=NODES), Budgets()
        games = dense_eval.paired_games(model, model, 2, 'test', config, config.evaluation, None, budgets=(mine, theirs))
        for game in games:
            colour = game.record['challenger_color']
            self.assertEqual((game.solvers[colour], game.solvers[1-colour]), (mine, theirs))
            game.finish()


def shards(asynchronous, root):
    """Seeded self-play with every solver point on, including two games from positions with forced wins, published
    as a shard under `root`; returns the digests of its data files."""
    model = tiny_model()
    s = settings(solver_root_nodes=NODES, solver_finalists=2, solver_finalist_nodes=NODES, solver_threat_nodes=NODES,
                 solver_async=asynchronous)
    games = [dense_selfplay.SelfPlayGame([model, model], s, seed) for seed in (1, 2)]
    for seed, key in ((3, PROOF), (4, '1790600287230040:25:213')):
        opening = FIXTURE['positions'][key]
        games.append(from_position(dense_selfplay.SelfPlayGame([model, model], replace(s, max_plies=len(opening)+12),
                                                               seed), opening))
    run(games, asynchronous)
    episodes, rows = [], []
    for game in games:
        episode, items = game.episode()
        episodes.append(episode)
        rows += [dict(r, game=len(episodes)-1) for r in items]
    manifest = dense_data.write_shard(Path(root)/'shards'/'000001', dict(actor_sha256='a'*64), episodes, rows)
    return manifest['files'], manifest['counts']


class Determinism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            NativeTactics()
        except FileNotFoundError:
            raise unittest.SkipTest('Build tools/tactical with tools/build_tactical.py first')

    def test_seeded_selfplay_shards_repeat_across_runs_and_backends(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, tempfile.TemporaryDirectory() as c:
                first, counts = shards(True, a)
                self.assertEqual(shards(True, b), (first, counts))
                self.assertEqual(shards(False, c), (first, counts))
        finally:
            torch.set_num_threads(threads)
        self.assertGreaterEqual(counts['proven_rows'], 4)


class ProvenTargets(unittest.TestCase):
    def test_proven_rows_get_the_exact_value_with_their_weight(self):
        moves = winning_game()
        roots = [float(v) for v in np.random.default_rng(3).uniform(-1, 1, len(moves))]
        episode, rows = episode_rows(moves, 0, roots)
        proven = {2: 1, 3: 1, 6: -1}
        for ply, value in proven.items():
            rows[ply]['proven'] = value
        with tempfile.TemporaryDirectory() as tmp:
            manifest = dense_data.write_shard(Path(tmp)/'shards'/'000001', dict(actor_sha256='a'*64), [episode], rows)
            self.assertEqual(manifest['counts']['proven_rows'], 3)
            window = dense_data.ReplayWindow(tmp, capacity_rows=1000)
            refs = [window.ref('000001', i) for i in range(len(window.index))]
            options = dense_data.target_options(replace(dense_config.LearnerSettings(), value_target='td',
                                                        proven_value_weight=2.5))
            targets = dense_data.examples(window, refs, np.random.default_rng(0), **options)[1]
            plain = dense_data.examples(window, refs, np.random.default_rng(0), **dict(options, proven_weight=1.))[1]
        for ref, target, base in zip(refs, targets, plain):
            ply = ref.row['ply']
            if ply in proven:
                self.assertEqual((target['value'], target['value_weight']), (float(proven[ply] > 0), 2.5))
                self.assertEqual(target['outcome_weight'], 1.)
            else:
                self.assertEqual((target['value'], target['value_weight']), (base['value'], base['value_weight']))
                if ply < len(moves)-1:
                    self.assertNotIn(target['value'], (0., 1.))
            np.testing.assert_array_equal(target['policy'], base['policy'])


class Protocol(unittest.TestCase):
    def test_reports_before_the_solver_settings_count_as_solver_off(self):
        settings = dense_config.EvaluationSettings()
        old = dict(settings={k: v for k, v in asdict(settings).items() if not k.startswith('solver_')})
        self.assertTrue(dense_eval.same_protocol(old, settings))
        self.assertFalse(dense_eval.same_protocol(old, replace(settings, solver_root_nodes=NODES)))
        on = dict(settings=asdict(replace(settings, solver_root_nodes=NODES)))
        self.assertTrue(dense_eval.same_protocol(on, replace(settings, solver_root_nodes=NODES)))
        self.assertFalse(dense_eval.same_protocol(on, settings))

    def test_budgets_validate(self):
        with self.assertRaises(ValueError):
            Budgets(finalists=2)
        with self.assertRaises(ValueError):
            Budgets(root_nodes=-1)
        self.assertFalse(Budgets.of(dense_config.ActorSettings()).active)
        self.assertTrue(Budgets.of(dense_config.EvaluationSettings(solver_threat_nodes=NODES)).active)


if __name__ == '__main__':
    unittest.main()
