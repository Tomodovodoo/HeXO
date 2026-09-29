"""Solver points inside the dense searches (dense_solver): the native hold, mark and priority entries, each
injection point on positions with known forced wins, proofs followed and labelled, the adaptive scheduler, learner
targets of proven rows and reproducibility of seeded self-play with fixed budgets, across runs and backends. CPU
only, tiny models."""
from concurrent.futures import Future
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import threading
import time
import unittest

import numpy as np
import torch

import dense_config
import dense_data
import dense_eval
import dense_openings
import dense_selfplay
import dense_solver
from dense_solver import Budgets, Proof, Schedule
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


def run(slots, asynchronous=True, schedule=None):
    """Play `slots` to the end on one Engine; returns the Solver's summary (None when no slot used it)."""
    engine = dense_selfplay.Engine(64, asynchronous, schedule)
    try:
        for slot in slots:
            engine.add(slot)
        while engine.slots or engine.closing:
            engine.step()
        return engine.solver.summary(1.) if engine.solver else None
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
                                                 schedule=asdict(Schedule()),
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
        sides = (replace(config.evaluation, solver_root_nodes=NODES), config.evaluation)
        with tempfile.TemporaryDirectory() as run:
            book = dense_openings.Book(run, config.evaluation)
        games = dense_eval.paired_games(model, model, 2, 'test', config, config.evaluation, None, book, sides=sides)
        for game in games:
            colour = game.record['challenger_color']
            self.assertEqual((game.solvers[colour], game.solvers[1-colour]), (mine, theirs))
            game.finish()


def line(certificate):
    """The certificate's first attacker turn, the first defender reply it covers and the attacker's next turn."""
    nodes = certificate['nodes']
    first = nodes[certificate['root']]
    reply = nodes[first['child']]['responses'][0]
    return first['action'], reply['action'], nodes[reply['child']]['action']


class Proofs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            engine = NativeTactics()
        except FileNotFoundError:
            raise unittest.SkipTest('Build tools/tactical with tools/build_tactical.py first')
        cls.opening = [tuple(m) for m in FIXTURE['positions'][PROOF]]
        cls.result = engine.history(cls.opening, nodes=NODES)

    def test_path_walks_the_game_labels_both_sides_and_names_the_next_stones(self):
        certificate, base = self.result['certificate'], self.opening
        first, reply, second = ([tuple(c) for c in turn] for turn in line(certificate))
        proof = Proof(base, certificate)
        turns = self.result['proof_turns']
        self.assertEqual(proof.path(base), ([], (first, turns)))
        self.assertEqual(proof.path(base+first[:1]), ([(len(base), 1, turns)], (first[1:], turns)))
        self.assertEqual(proof.path(base+first)[1], ([], turns-1))
        labels, move = proof.path(base+first+reply)
        self.assertEqual(labels, [(len(base), 1, turns), (len(base)+1, 1, turns), (len(base)+2, -1, turns-1),
                                  (len(base)+3, -1, turns-1)])
        self.assertEqual(move, (second, turns-1))
        self.assertIsNone(Proof(base, certificate, first_turn_only=True).path(base+first+reply)[1])
        self.assertEqual(proof.path(base[:-1]), ([], None))
        other = next(m for m in Game(list(base)).legal_moves() if tuple(m) not in first)
        self.assertEqual(proof.path(base+[tuple(other)]), ([(len(base), 1, turns)], None))  # a stone off the turn

    def test_followed_proof_plays_the_certificate_to_the_win_and_labels_the_loser(self):
        model = tiny_model()
        s = settings(full_fraction=1., solver_root_nodes=NODES, solver_follow=True)
        game = from_position(dense_selfplay.SelfPlayGame([model, model], replace(s, max_plies=len(self.opening)+16), 5),
                             self.opening)
        summary = run([game], schedule=Schedule.of(s))
        episode, rows = game.episode()
        winner = dense_solver.mover(self.opening)
        self.assertEqual(episode['winner'], winner)
        labels, move = Proof(self.opening, self.result['certificate']).path([tuple(m) for m in episode['moves']])
        self.assertEqual([p for p, _, _ in labels], [r['ply'] for r in rows])   # the game never left the certificate
        self.assertEqual([r['proven'] for r in rows], [1 if r['player'] == winner else -1 for r in rows])
        # One root query for the winner; the loser asks at each of its turn starts.
        self.assertEqual(summary['root_queries'], 1+len([r for r in rows if r['player'] != winner and r['remaining'] == 2]))
        self.assertEqual(summary['followed'], len([r for r in rows if r['player'] == winner])-1)  # all but the root's

    def test_deep_proof_of_a_committed_turn_is_followed(self):
        first = [tuple(c) for c in self.result['moves']]
        model = tiny_model()
        s = settings(full_fraction=1., solver_deep_nodes=2000, solver_follow=True)   # deep proofs alone
        opening = self.opening+first
        game = from_position(dense_selfplay.SelfPlayGame([model, model], replace(s, max_plies=len(opening)+24), 5),
                             opening)
        summary = run([game], schedule=Schedule.of(s))
        episode, rows = game.episode()
        winner = dense_solver.mover(self.opening)
        self.assertEqual(episode['winner'], winner)
        self.assertEqual(summary['root_queries'], 0)
        self.assertGreater(summary['deep_hit_rate'], 0)
        # The defender's turn after the committed one is labelled lost; the winner follows the proof from then on.
        self.assertEqual([r['proven'] for r in rows], [1 if r['player'] == winner else -1 for r in rows])
        self.assertEqual(summary['followed'], len([r for r in rows if r['player'] == winner])-1)  # adopted at its first search end


class Adjudication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            engine = NativeTactics()
        except FileNotFoundError:
            raise unittest.SkipTest('Build tools/tactical with tools/build_tactical.py first')
        cls.opening = [tuple(m) for m in FIXTURE['positions'][PROOF]]
        cls.proof = engine.history(cls.opening, nodes=NODES)

    def play(self, plies=40, **changes):
        model = tiny_model()
        s = settings(full_fraction=1., solver_root_nodes=NODES, adjudicate_proven=True, **changes)
        game = from_position(dense_selfplay.SelfPlayGame([model, model], replace(s, max_plies=len(self.opening)+plies), 5),
                             self.opening)
        run([game], schedule=Schedule.of(s))
        return game.episode()

    def test_a_root_proof_ends_the_game(self):
        episode, rows = self.play()
        winner = dense_solver.mover(self.opening)
        self.assertEqual((episode['reason'], episode['winner']), ('proven', winner))
        self.assertEqual(episode['moves'][len(self.opening):], [self.proof['moves'][0]])
        self.assertEqual([r['proven'] for r in rows], [1])
        self.assertEqual(episode['adjudicated']['ply'], len(self.opening)+1)
        self.assertGreater(episode['adjudicated']['line_plies'], 1)
        with tempfile.TemporaryDirectory() as root:
            manifest = dense_data.write_shard(Path(root)/'shards'/'000001', dict(actor_sha256='a'*64), [episode],
                                              [dict(r, game=0) for r in rows])
        self.assertEqual((manifest['counts']['proven_games'], manifest['counts']['line_rows']), (1, 0))
        self.assertEqual(manifest['counts']['adjudicated_plies'], episode['adjudicated']['line_plies'])

    def test_a_proof_on_the_capped_ply_still_adjudicates(self):
        episode, rows = self.play(plies=1)
        self.assertEqual((episode['reason'], episode['winner'], len(rows)), ('proven', dense_solver.mover(self.opening), 1))

    def test_line_rows_play_the_certificate_to_six_without_search(self):
        episode, rows = self.play(proven_line_rows=True)
        winner = dense_solver.mover(self.opening)
        game = Game([tuple(m) for m in episode['moves']])
        self.assertEqual((game.winner, episode['winner'], episode['reason']), (winner, winner, 'proven'))
        game.close()
        line = rows[1:]
        self.assertEqual(len(line), episode['adjudicated']['line_plies'])
        self.assertTrue(all(r['line'] and r['policy'] is None for r in line))
        self.assertEqual([r['proven'] for r in rows], [1 if r['player'] == winner else -1 for r in rows])
        self.assertEqual(episode['root_values'][1:], [1. if r['player'] == winner else -1. for r in line])
        self.assertFalse(any(episode['full_search'][1:]))
        with tempfile.TemporaryDirectory() as root:
            manifest = dense_data.write_shard(Path(root)/'shards'/'000001', dict(actor_sha256='a'*64), [episode],
                                              [dict(r, game=0) for r in rows])
            window = dense_data.ReplayWindow(root, capacity_rows=1000)
            refs = [window.ref('000001', i) for i in range(len(window.index))]
            options = dense_data.target_options(dense_config.LearnerSettings())
            targets = dense_data.examples(window, refs, np.random.default_rng(0), **options)[1]   # rows replay
        self.assertEqual(manifest['counts']['line_rows'], len(line))
        self.assertEqual([t['value'] for t in targets], [float(r['proven'] > 0) for r in rows])


class Scheduler(unittest.TestCase):
    def test_schedule_defaults_and_validation(self):
        self.assertEqual(Schedule.of(dense_config.ActorSettings()), Schedule())
        self.assertEqual(Schedule.of(dense_config.EvaluationSettings()), Schedule())
        self.assertTrue(Schedule().fixed_budgets)
        for bad in (dict(deep_nodes=100), dict(workers=0), dict(min_nodes=600), dict(cap_nodes=9000),
                    dict(overrun_fraction=-.1), dict(deep_nodes=70000, follow=True), dict(gate_weight=101)):
            with self.assertRaises(ValueError):
                Schedule(**bad)

    def test_allocation_follows_the_measured_lead(self):
        try:
            solver = dense_solver.Solver(Schedule(fixed_budgets=False, gate_weight=3.))
        except FileNotFoundError:
            raise unittest.SkipTest('Build tools/tactical with tools/build_tactical.py first')
        self.addCleanup(solver.close)
        with self.assertRaises(ValueError):
            dense_solver.Solver(Schedule(fixed_budgets=False), asynchronous=False)
        self.assertEqual(solver.allocate('root', NODES)[0], 32)          # nothing measured yet: the floor
        solver.leads['root'].extend([70.]*8)
        solver.tick(10., 5.)
        budget, gate, _ = solver.allocate('root', NODES)
        overrun = .05*10.   # root verdicts may be waited for: the step's overrun allowance counts as slack
        self.assertEqual(budget, int(dense_solver.RATE*(.95*(70-dense_solver.GUARD_MS+overrun)-dense_solver.OVERHEAD_MS)))
        self.assertEqual(gate, dict(weight=3., floor=32, cap_low=512, cap_high=8192))
        solver.leads['finalist'].extend([5000.]*8)
        solver.tick(10., 5.)
        self.assertEqual(solver.allocate('finalist', NODES)[0], 512)
        self.assertEqual(solver.allocate('threat', NODES)[:2], (NODES, None))
        solver.leads['deep'].extend([5000.]*8)
        solver.tick(10., 5.)
        self.assertEqual(solver.allocate('deep', 2000)[1]['floor'], 2000)   # below the gate a deep query keeps its minimum
        fixed = dense_solver.Solver(Schedule(gate_weight=3.), asynchronous=False)
        fixed.leads['root'].extend([5000.]*8)
        self.assertEqual(fixed.allocate('root', NODES)[:2],
                         (NODES, dict(weight=3., floor=NODES, cap_low=dense_solver.MAX_NODES, cap_high=dense_solver.MAX_NODES)))

    def test_late_proofs_are_played_only_as_their_own_root_turn_without_follow(self):
        try:
            engine = NativeTactics()
            solver = dense_solver.Solver(Schedule(fixed_budgets=False))
        except FileNotFoundError:
            raise unittest.SkipTest('Build tools/tactical with tools/build_tactical.py first')
        self.addCleanup(solver.close)
        opening = [tuple(m) for m in FIXTURE['positions'][PROOF]]
        result = dict(engine.history(opening, nodes=NODES), budget=NODES)
        first = [tuple(c) for c in result['moves']]
        for point, base, history, played in (('finalist', opening, opening, False),
                                             ('root', opening, opening+first[:1], True)):
            plan, future = dense_solver.Plan(solver), Future()
            future.set_result(result)
            plan.late.append(dense_solver.Query(solver, point, tuple(base), NODES, future))
            plan.poll(history)
            self.assertEqual(plan.move(dense_solver.mover(opening), history) is not None, played, point)
            self.assertEqual(len(plan.found), 1)

    def test_a_finished_game_waits_for_a_proof_that_labels_its_rows(self):
        try:
            engine = NativeTactics()
        except FileNotFoundError:
            raise unittest.SkipTest('Build tools/tactical with tools/build_tactical.py first')
        opening = [tuple(m) for m in FIXTURE['positions'][PROOF]]
        proof = dict(engine.history(opening, nodes=NODES), budget=NODES)
        first = [tuple(c) for c in proof['moves']]
        engine = dense_selfplay.Engine(64, schedule=Schedule(fixed_budgets=False, follow=True))
        engine.solver = dense_solver.Solver(engine.schedule)
        self.addCleanup(engine.close)
        plan, future = dense_solver.Plan(engine.solver), Future()
        plan.late.append(dense_solver.Query(engine.solver, 'root', tuple(opening), NODES, future))
        rows = []
        slot = type('Slot', (), dict(tree=type('Tree', (), dict(history=opening+first))(),
                                     label=lambda self, ply, proven, turns: rows.append((ply, proven)) or 1))()
        engine.closing.append((slot, plan, time.perf_counter()+10))
        self.assertEqual(engine.step(), [])
        future.set_result(proof)
        self.assertEqual(engine.step(), [slot])
        self.assertEqual(rows, [(len(opening), 1), (len(opening)+1, 1)])

    def test_a_failing_query_fails_its_future_and_releases_its_reservation(self):
        try:
            pool = dense_solver.Pool(1, None)
        except FileNotFoundError:
            raise unittest.SkipTest('Build tools/tactical with tools/build_tactical.py first')
        try:
            future = pool.submit(0., 5., [[0, 0]], dict(nodes=0, ms=100))
            with self.assertRaises(ValueError):
                future.result(10)
            self.assertEqual(pool.reserved, 0.)
        finally:
            pool.close()

    def test_a_late_verdict_defers_its_game_once_then_finishes_in_the_background(self):
        try:
            solver = dense_solver.Solver(Schedule(fixed_budgets=False))
        except FileNotFoundError:
            raise unittest.SkipTest('Build tools/tactical with tools/build_tactical.py first')
        self.addCleanup(solver.close)
        plan, future = dense_solver.Plan(solver), Future()
        query = dense_solver.Query(solver, 'root', ((0, 0),), 100, future)
        solver.tick(100., 50.)                                          # allowance: 5 ms
        start = time.perf_counter()
        self.assertFalse(plan.defer([query]))
        self.assertGreaterEqual(time.perf_counter()-start, .003)       # waited out the overrun allowance (1 ms timer)
        self.assertLess(solver.allowance, 1.5)
        self.assertTrue(plan.defer([query]))
        self.assertEqual((plan.late, solver.stats['deferred'], solver.stats['late']), ([query], 1, 1))
        future.set_result(dict(status='UNKNOWN', reason='no verified strategy', nodes_used=100, budget=100))
        plan.poll(((0, 0),))
        self.assertEqual((plan.late, solver.stats['points']['root']['queries']), ([], 1))
        solver.tick(100., 50.)
        self.assertEqual(solver.summary(1.)['wait_step_fraction'], .5)
        # A threat verdict missing its visit is dropped but still accounted when it completes.
        slot = type('Slot', (), dict(tree=type('Tree', (), dict(ptr=None))))()
        plan.threat = threat = dense_solver.Query(solver, 'threat', None, 135, Future())
        self.assertTrue(plan.ready(slot))
        self.assertEqual((plan.threat, plan.late, solver.stats['dropped']), (None, [threat], 1))
        threat.future.set_result(dict(status='UNKNOWN', reason='no verified strategy', nodes_used=135, budget=135))
        plan.poll(((0, 0),))
        self.assertEqual((plan.late, solver.stats['points']['threat']['queries']), ([], 1))
        # Without follow no running query holds a finished game back.
        plan.late.append(dense_solver.Query(solver, 'root', ((0, 0),), 100, Future()))
        self.assertFalse(plan.pending())
        plan.late.clear()
        # A game ending with a query still running hands it to the Solver, which accounts it once it completes.
        plan.late.append(orphan := dense_solver.Query(solver, 'root', ((0, 0),), 100, Future()))
        plan.close(slot, ((0, 0),))
        self.assertEqual(solver.orphans, [orphan])
        orphan.future.set_result(dict(status='UNKNOWN', reason='no verified strategy', nodes_used=100, budget=100))
        solver.tick(100., 50.)
        self.assertEqual((solver.orphans, solver.stats['points']['root']['queries']), ([], 2))
        # At the end of a run the orphans are drained.
        solver.orphans.append(last := dense_solver.Query(solver, 'deep', ((0, 0),), 100, Future()))
        threading.Timer(.05, last.future.set_result, [dict(status='UNKNOWN', reason='no verified strategy',
                                                            nodes_used=100, budget=100)]).start()
        solver.drain(5.)
        self.assertEqual((solver.orphans, solver.stats['points']['deep']['queries']), ([], 1))

    def test_adaptive_selfplay_plays_proofs_within_the_caps(self):
        try:
            NativeTactics()
        except FileNotFoundError:
            raise unittest.SkipTest('Build tools/tactical with tools/build_tactical.py first')
        model = tiny_model()
        s = settings(solver_root_nodes=NODES, solver_finalists=2, solver_finalist_nodes=NODES, solver_threat_nodes=NODES,
                     solver_fixed_budgets=False, solver_workers=2, solver_min_nodes=NODES, solver_gate_weight=3.,
                     solver_deep_nodes=NODES, solver_follow=True, solver_overrun_fraction=.5)
        games = [dense_selfplay.SelfPlayGame([model, model], s, seed) for seed in (1, 2)]
        for seed, key in ((3, PROOF), (4, '1790600287230040:25:213')):
            opening = FIXTURE['positions'][key]
            games.append(from_position(dense_selfplay.SelfPlayGame([model, model], replace(s, max_plies=len(opening)+12),
                                                                   seed), opening))
        summary = run(games, schedule=Schedule.of(s))
        rows = [r for g in games for r in g.episode()[1]]
        self.assertGreater(sum(r['proven'] == 1 for r in rows), 0)
        self.assertTrue(all(r['solver_budget'] >= 0 for r in rows))
        self.assertLessEqual(summary['budget_p95'], 8192)
        self.assertGreaterEqual(min(solver_budgets(summary)), NODES)
        self.assertEqual(summary['failures'], 0)
        self.assertEqual(set(summary['idle_fraction']), {'foreground', 'background'})
        self.assertTrue(0 <= summary['slack_utilisation'])
        for key in ('wait_step_fraction', 'overrun_fraction', 'band_hit_rate', 'utilisation', 'lead_ms', 'nodes_per_ms'):
            self.assertIn(key, summary)


def solver_budgets(summary):
    return [summary[f'{p}_budget'] for p in dense_solver.POINTS if summary[f'{p}_queries']]


def shards(asynchronous, root):
    """Seeded self-play with every solver point, gate, deep proofs and following on under fixed budgets, including two
    games from positions with forced wins, published as a shard under `root`; returns the digests of its data
    files."""
    model = tiny_model()
    s = settings(solver_root_nodes=NODES, solver_finalists=2, solver_finalist_nodes=NODES, solver_threat_nodes=NODES,
                 solver_async=asynchronous, solver_gate_weight=3., solver_deep_nodes=NODES, solver_follow=True)
    games = [dense_selfplay.SelfPlayGame([model, model], s, seed) for seed in (1, 2)]
    for seed, key in ((3, PROOF), (4, '1790600287230040:25:213')):
        opening = FIXTURE['positions'][key]
        games.append(from_position(dense_selfplay.SelfPlayGame([model, model], replace(s, max_plies=len(opening)+12),
                                                               seed), opening))
    run(games, asynchronous, Schedule.of(s))
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
