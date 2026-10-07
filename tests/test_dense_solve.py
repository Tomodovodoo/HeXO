"""Offline proof pass (dense_solve): gate and solve on a position with a known win, windows, lookback, sidecars,
the restart buffer, restart games in the actor, and the learner's use of sidecar labels. CPU only, tiny models."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import unittest.mock

import numpy as np
import torch

import dashboard
import dense_bootstrap
import dense_config
import dense_data
import dense_learn
import dense_selfplay
import dense_solve
import hexcrop
from dense_data import player_at
from forcing_material import worth_solving
from hexo import Game
from proof import VerificationTimeout
from tactical_proof import NativeTactics
from tests import slow
from tests.test_dense import source_shard, winning_game, write_games
from tests.test_dense_solver import TINY, tiny_model
from tests.test_tactical_proof import FIXTURE
from legacy.train import write_json

PROOF = '1790600149713752:2:253'  # the side to move wins in 4 turns; proven within 135 nodes
SMALL = dense_solve.PassSettings(solve_nodes=135, scan_nodes=1500, saving_nodes=300, verify_fraction=1.)
PREFIX = [[0, 0], [0, 5], [1, 5], [1, 0], [2, 0], [3, 5]]


class Recorder:
    """NativeTactics that records every query as (history length, attacker, nodes)."""

    def __init__(self, engine):
        self.engine, self.calls = engine, []

    def history(self, history, **query):
        self.calls.append((len(history), query['attacker'], query['nodes']))
        return self.engine.history(history, **query)


def new_run(path, **learner):
    dense_config.save(path, dense_config.RunConfig(device='cpu', model=dense_config.ModelSettings(**{
        k: getattr(TINY, k) for k in ('blocks', 'channels', 'pool_every', 'line_length', 'value_hidden', 'head_channels')}),
        learner=dense_config.LearnerSettings(**learner)))
    return path


def proof_game(engine):
    """(moves, mover): the PROOF position followed by its proof's first turn."""
    history = FIXTURE['positions'][PROOF]
    moves = engine.history(history, nodes=135, ms=60000)['moves']
    return [list(m) for m in history]+[list(m) for m in moves], player_at(len(history))


def shard_of(run, name, episodes, identity=None):
    """Write an actor shard of `episodes` (full episode dicts) with rows at every ply from its restart ply on."""
    rows = []
    for g, e in enumerate(episodes):
        start = e['restart']['ply'] if e.get('origin') == 'restart' else \
            e['book']['ply'] if e.get('origin') == 'book' else 0
        game = Game(e['moves'][:start])
        for t in range(start, len(e['moves'])):
            rows.append(dict(game=g, ply=t, player=game.player, remaining=game.remaining, policy=None,
                             legal_sha256=dense_data.legal_digest(np.asarray(game.legal_moves(), np.int64).reshape(-1, 2))))
            game.play(*e['moves'][t])
        game.close()
    return dense_data.write_shard(Path(run)/'shards'/name, identity or dict(actor_sha256='a'*64, checkpoint='main/000010'),
                                  episodes, rows)


def episode(moves, winner, **extra):
    return dict(moves=moves, winner=winner, reason='test', opening_plies=0, actor='a'*64,
                root_values=[0.]*len(moves), full_search=[True]*len(moves), **extra)


class PassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.engine = NativeTactics()
        except FileNotFoundError:
            raise unittest.SkipTest('Build tools/tactical with tools/build_tactical.py first')
        cls.moves, cls.mover = proof_game(cls.engine)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run = new_run(Path(tmp.name)/'run')
        shard_of(self.run, '1000000000001', [episode(self.moves, self.mover)])

    def solve(self, out=None, settings=SMALL, name='1000000000001', tactics=None, prepare=lambda coordinator: None):
        """Solve shard `name` in process and record it through JSON as the coordinator does a worker's result."""
        tactics = tactics or Recorder(self.engine)
        result = dense_solve.Solver(self.run, out or self.run, settings, tactics, clock=lambda: 1000.).solve(name)
        coordinator = dense_solve.Pass(self.run, out or self.run, settings, clock=lambda: 1000.)
        prepare(coordinator)
        coordinator.record(name, json.loads(json.dumps(result)))
        return coordinator, tactics, result['windows']

    def test_known_win_opens_a_window_with_lookback_and_buffer_entries(self):
        solver, tactics, windows = self.solve()
        self.assertEqual(dense_data.proof_windows(self.run/'shards'/'1000000000001'), windows)
        hit = len(FIXTURE['positions'][PROOF])
        window = next(w for w in windows if hit in w['plies'])
        m, first = window['mover'], window['first_ply']
        self.assertEqual((m, window['last_ply'], window['persistent']), (self.mover, window['plies'][-1], True))
        self.assertTrue(all(t % 2 or t-1 in window['plies'] for t in window['plies']))
        self.assertTrue(all(player_at(t) == m for t in window['plies']))
        self.assertEqual(window['plies'][0], first)
        for t in window['plies']:
            result = self.engine.history(self.moves[:t], nodes=SMALL.scan_nodes, ms=60000)
            self.assertEqual(result['status'], 'PROVEN_WIN', t)
            self.assertTrue(window['proof_action'][str(t)])
            legal = hexcrop.legal_array(hexcrop.Position(self.moves[:t]), np.asarray(self.moves[:t]))
            self.assertTrue(all(tuple(a) in set(map(tuple, legal)) for a in window['proof_action'][str(t)]))
        if first-4 >= 1:
            result = self.engine.history(self.moves[:first-4], nodes=SMALL.scan_nodes, ms=60000)
            self.assertEqual(result['status'], 'UNKNOWN')
        self.assertEqual([d['ply'] for d in window['defence']], [first-2, first-6])
        for d in window['defence']:
            self.assertEqual(d['saving_turns'] is None, not d['threat'])
            for a, b in d['saving_turns'] or ():
                after = self.engine.history(self.moves[:d['ply']]+[a, b], nodes=SMALL.saving_nodes, ms=60000)
                self.assertEqual(after['status'], 'UNKNOWN')
        stats = solver.stats
        proofs = sum(q['proofs'] for q in stats['queries'].values())
        self.assertEqual((stats['verified'], stats['failures']), (proofs, 0))
        self.assertGreater(stats['positions'], stats['gated'])
        entries = {(e['ply'], e['kind']): e for e in solver.buffer.entries.values()}
        attack = entries[first, 'attack']
        self.assertEqual((attack['side_to_move'], attack['regret'], attack['plies_to_proof'], attack['checkpoint']),
                         (m, .5, 0, 'main/000010'))
        for d in window['defence']:
            defence = entries[d['ply'], 'defence']
            self.assertEqual((defence['side_to_move'], defence['plies_to_proof'], defence['saving_turns']),
                             (1-m, first-d['ply'], d['saving_turns']))

    def test_the_gate_decides_which_turn_starts_are_solved(self):
        solver, _, _ = self.solve(settings=replace(SMALL, verify_fraction=0.))
        game, gated = Game(), 0
        for t, move in enumerate(self.moves):
            gated += bool(t % 2 and worth_solving(game))
            game.play(*move)
        game.close()
        self.assertEqual((solver.stats['positions'], solver.stats['gated']), (len(self.moves)//2, gated))
        self.assertLess(gated, len(self.moves)//2)
        with unittest.mock.patch.object(dense_solve, 'worth_solving', return_value=False):
            solver, tactics, windows = self.solve(self.run.parent/'closed')
        self.assertEqual((tactics.calls, windows, solver.stats['gated']), ([], [], 0))

    def test_network_error_survives_solver_corrected_search_values(self):
        e = dict(episode(self.moves, self.mover), root_values=[1.]*len(self.moves),
                 network_values=[-.6]*len(self.moves))
        shard_of(self.run, '1000000000002', [e])
        solver, _, windows = self.solve(name='1000000000002')
        attacks = [entry for entry in solver.buffer.entries.values() if entry['kind'] == 'attack']
        self.assertTrue(attacks)
        self.assertTrue(all(entry['regret'] == .8 and entry['value_source'] == 'network' for entry in attacks))
        self.assertTrue(all(entry['regret'] == 1. and 'value_source' not in entry
                            for entry in solver.buffer.entries.values() if entry['kind'] == 'defence'))
        self.assertTrue(all(w['search_value_at_first_ply'] == 1. for w in windows))
        del e['network_values']
        shard_of(self.run, '1000000000003', [e])
        legacy, _, _ = self.solve(self.run.parent/'legacy', name='1000000000003')
        self.assertFalse(any(entry['kind'] == 'attack' for entry in legacy.buffer.entries.values()))

    def test_same_shard_same_sidecar_and_buffer(self):
        outputs = []
        for k in range(2):
            out = self.run.parent/f'out{k}'
            self.solve(out)
            outputs.append(((out/'shards'/'1000000000001'/'proofs.jsonl').read_bytes(),
                            json.loads((out/'restarts.json').read_text())['entries']))
        self.assertEqual(outputs[0], outputs[1])
        self.assertFalse((self.run/'shards'/'1000000000001'/'proofs.jsonl').exists())

    @slow
    def test_worker_processes_match_the_in_process_solver_and_resume(self):
        out = self.run.parent/'out'
        coordinator = dense_solve.Pass(self.run, out, SMALL)
        coordinator.loop(once=True)
        self.solve(self.run.parent/'reference')
        for name in ('shards/1000000000001/proofs.jsonl', 'restarts.json'):
            read = lambda root: json.loads((root/name).read_text()) if name.endswith('.json') else (root/name).read_text()
            expected, got = read(self.run.parent/'reference'), read(out)
            if name == 'restarts.json':
                expected, got = ([dict(e, added_at=None) for e in x['entries']] for x in (expected, got))
            self.assertEqual(got, expected)
        resumed = dense_solve.Pass(self.run, out, SMALL)
        self.assertFalse(resumed.step(spawn=lambda name: self.fail(f'{name} solved twice')))
        status = json.loads((out/'solver-status.json').read_text())
        self.assertEqual((status['stage'], status['shards_pending'], status['buffer_size'], status['workers']),
                         ('idle', 0, len(coordinator.buffer.entries), 1))
        for key in ('positions_per_hour', 'hits_per_hour', 'gate_pass_rate', 'transient', 'persistent', 'window_plies',
                    'buffer_mean_regret', 'busy_cores', 'worker_switches'):
            self.assertIn(key, status)
        self.assertEqual(list((out/'.solve').glob('*.json')), [])

    def test_failed_queries_are_no_saving_turns(self):
        engine = self.engine

        class Failing(Recorder):
            def history(self, history, **query):
                if query['nodes'] == SMALL.saving_nodes:
                    return dict(status='UNKNOWN', reason='deadline', elapsed_ms=1.)
                return engine.history(history, **query)
        solver, _, windows = self.solve(self.run.parent/'failing', tactics=Failing(engine))
        threats = [d for w in windows for d in w['defence'] if d['threat']]
        self.assertTrue(threats)
        self.assertTrue(all(d['saving_turns'] == [] for d in threats))
        self.assertEqual((solver.stats['failures'], solver.stats['last_failure']),
                         (solver.stats['queries']['saving']['queries'], 'deadline'))

    def test_rejected_independent_check_aborts_with_an_event(self):
        with unittest.mock.patch.object(dense_solve, 'independent_verify', side_effect=ValueError('bad')):
            with self.assertRaises(dense_solve.Rejected):
                self.solve()
        events = [json.loads(line) for line in (self.run/'events.jsonl').read_text().splitlines()]
        self.assertEqual([e['kind'] for e in events], ['error'])
        self.assertIsNone(dense_data.proof_windows(self.run/'shards'/'1000000000001'))

    def test_timed_out_independent_check_keeps_the_proof_unverified(self):
        settings = replace(SMALL, verify_seconds=7.)
        _, _, reference = self.solve(self.run.parent/'reference')
        with unittest.mock.patch.object(dense_solve, 'independent_verify', side_effect=VerificationTimeout('late')) as check:
            coordinator, _, windows = self.solve(self.run.parent/'late', settings=settings)
        self.assertEqual(windows, reference)
        proofs = sum(q['proofs'] for q in coordinator.stats['queries'].values())
        self.assertEqual((coordinator.stats['verify_timeouts'], coordinator.stats['verified']), (proofs, 0))
        self.assertTrue(all(call.args[3] == 7. for call in check.call_args_list))
        self.assertFalse((self.run.parent/'late'/'events.jsonl').exists())
        coordinator.status('idle', 0)
        status = json.loads((self.run.parent/'late'/'solver-status.json').read_text())
        self.assertEqual((status['verify_timeouts'], status['verified'], status['settings']['verify_seconds']),
                         (proofs, 0, 7.))

    def test_restart_games_record_observations_instead_of_early_entries(self):
        source = dict(shard='1000000000001', game=0, ply=5, kind='attack', regret=.5, plies_to_proof=0)
        moves = [list(m) for m in winning_game()]
        restart = dict(episode(moves, 0, origin='restart', restart=source), root_values=[None]*5+[.4]*(len(moves)-5))
        shard_of(self.run, '1000000000002', [restart])
        known = lambda c: c.buffer.entries.update({('1000000000001', 0, 5, 'attack'): dict(
            source, side_to_move=0, added_at=0., checkpoint=None)})
        solver, _, windows = self.solve(name='1000000000002', prepare=known)
        self.assertTrue(windows and all(w['first_ply'] >= 5 for w in windows))
        self.assertTrue(all(e['ply'] > 5 for e in solver.buffer.entries.values() if e['shard'] == '1000000000002'))
        self.assertEqual(solver.buffer.entries[('1000000000001', 0, 5, 'attack')]['observed'],
                         {'main/000010': dict(value=.4, shard='1000000000002')})

    def test_network_restart_observations_keep_the_network_source(self):
        source = dict(shard='1000000000001', game=0, ply=5, kind='attack', regret=.8, plies_to_proof=0,
                      value_source='network')
        moves = [list(m) for m in winning_game()]
        e = dict(episode(moves, 0, origin='restart', restart=source),
                 root_values=[None]*5+[1.]*(len(moves)-5),
                 network_values=[None]*5+[-.6]*(len(moves)-5))
        shard_of(self.run, '1000000000002', [e])
        result = dense_solve.Solver(self.run, self.run, SMALL, self.engine).solve('1000000000002')
        key = [source['shard'], source['game'], source['ply'], source['kind']]
        self.assertEqual(result['observations'], [(key, 'main/000010', -.6, 'network')])
        del e['network_values']
        shard_of(self.run, '1000000000003', [e])
        result = dense_solve.Solver(self.run, self.run, SMALL, self.engine).solve('1000000000003')
        self.assertEqual(result['observations'], [])


class Process:
    """A shard worker stand-in that never finishes until killed."""

    def __init__(self):
        self.killed = False

    def poll(self):
        return None

    def kill(self):
        self.killed = True

    def wait(self):
        return -9


class WorkerCountTests(unittest.TestCase):
    def test_book_prefix_is_excluded_from_proof_queries_and_first_searched_ply_can_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = new_run(Path(tmp)/'run')
            moves = [list(m) for m in winning_game()]
            e = episode(moves, 0, origin='book', book=dict(ply=5))
            e['root_values'] = [None]*5+[.4]*(len(moves)-5)
            solver = dense_solve.Solver(run, run, SMALL, None)
            queried = []

            def query(where, history, attacker, nodes, kind):
                queried.append(len(history))
                return ([tuple(moves[len(history)])], 1, 'test') if attacker == 'mover' else None

            solver.stats = dense_solve.new_stats()
            with unittest.mock.patch.object(solver, 'query', side_effect=query), \
                 unittest.mock.patch.object(dense_solve, 'worth_solving', return_value=True):
                windows, entries = solver.game('book', 0, e)
            self.assertTrue(queried and min(queried) >= 5)
            self.assertTrue(all(w['first_ply'] >= 5 for w in windows))
            self.assertTrue(any(entry['ply'] == 5 for entry in entries))

    def test_workers_follow_the_phased_learner(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = new_run(Path(tmp)/'run')
            for k in range(5):
                shard_of(run, f'100000000000{k}', [episode(PREFIX, -1)])
            now = [1000.]
            coordinator = dense_solve.Pass(run, run, replace(SMALL, solve_workers_max=3), clock=lambda: now[0])
            heartbeat = lambda **status: write_json(run/'learner-status.json', dict(updated_at=now[0], **status))
            started = []

            def spawn(name):
                started.append((name, Process()))
                return started[-1][1]
            self.assertEqual(coordinator.workers()[0], 1)
            heartbeat(stage='training', phase_rows=0)
            self.assertTrue(coordinator.step(spawn=spawn))
            self.assertEqual([n for n, _ in started], ['1000000000004'])
            heartbeat(stage='training', phase_rows=1000)
            coordinator.step(spawn=spawn)
            self.assertEqual([n for n, _ in started], [f'100000000000{k}' for k in (4, 3, 2)])
            heartbeat(stage='phase-idle', phase_rows=1000)
            coordinator.step(spawn=spawn)
            self.assertEqual(([p.killed for _, p in started], sorted(coordinator.running)), ([False, True, True], ['1000000000004']))
            now[0] += dense_solve.STALE_SECONDS+1
            heartbeat(stage='exporting', phase_rows=1000)
            now[0] += dense_solve.STALE_SECONDS+1
            self.assertEqual(coordinator.workers()[0], 1)
            status = json.loads((run/'solver-status.json').read_text())
            self.assertEqual([s['workers'] for s in status['worker_switches']], [1, 3, 1])
            self.assertEqual(status['workers_reason'], 'learner phase-idle')
            events = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
            self.assertEqual([e['workers'] for e in events], [1, 3, 1])
            self.assertEqual(dense_solve.Pass(run, run, SMALL).workers()[0], 1)
            heartbeat(stage='training', phase_rows=1000)
            self.assertEqual(dense_solve.Pass(run, run, SMALL, clock=lambda: now[0]).workers()[0],
                             max(1, dense_solve.physical_cores()-2))


    def test_workers_follow_the_configured_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = new_run(Path(tmp)/'run', variant='reset')
            now = [1000.]
            coordinator = dense_solve.Pass(run, run, replace(SMALL, solve_workers_max=3), clock=lambda: now[0])
            heartbeat = lambda variant, **status: write_json(run/f'learner-status{variant}.json',
                                                           dict(updated_at=now[0], **status))
            heartbeat('', stage='training', phase_rows=1000)
            self.assertEqual(coordinator.workers()[0], 1)
            heartbeat('-reset', stage='phase-idle', phase_rows=1000)
            self.assertEqual(coordinator.workers()[0], 1)
            heartbeat('-reset', stage='training', phase_rows=1000)
            heartbeat('', stage='phase-idle', phase_rows=1000)
            self.assertEqual(coordinator.workers()[0], 3)
            heartbeat('-reset', stage='exporting', phase_rows=1000)
            self.assertEqual(coordinator.workers()[0], 3)
            now[0] += dense_solve.STALE_SECONDS+1
            self.assertEqual(coordinator.workers()[0], 1)


class RestartBufferTests(unittest.TestCase):
    def entry(self, ply, value, kind='attack', added_at=10.):
        return dict(shard='s', game=0, ply=ply, side_to_move=0, kind=kind, regret=dense_solve.regret(kind, value),
                    added_at=added_at, checkpoint='main/000010', plies_to_proof=0)

    def test_priority_capacity_and_minimum(self):
        with tempfile.TemporaryDirectory() as tmp:
            buffer = dense_solve.RestartBuffer(Path(tmp)/'restarts.json', 3, 2, .1)
            for ply, value in enumerate((.9, .0, -.5, -1., .5)):
                buffer.add(self.entry(ply, value))
            buffer.add(self.entry(9, .7, 'defence'))
            self.assertEqual(sorted((e['ply'], e['regret']) for e in buffer.entries.values()),
                             [(2, .75), (3, 1.), (9, .85)])
            buffer.save()
            again = dense_solve.RestartBuffer(Path(tmp)/'restarts.json', 3, 2, .1)
            self.assertEqual(again.entries, buffer.entries)
            again.observe(('s', 0, 3, 'attack'), 'main/000010', .2, '0001')
            again.add(self.entry(3, -1., added_at=99.))
            self.assertEqual((again.entries['s', 0, 3, 'attack']['added_at'], again.entries['s', 0, 3, 'attack']['observed']),
                             (10., {'main/000010': dict(value=.2, shard='0001')}))
            self.assertEqual([e['ply'] for e in json.loads((Path(tmp)/'restarts.json').read_text())['entries']], [3, 9, 2])
            smaller = dense_solve.RestartBuffer(Path(tmp)/'restarts.json', 2, 2, .1)
            self.assertEqual(sorted(e['ply'] for e in smaller.entries.values()), [3, 9])

    def test_refresh_reads_newest_observations_and_ages_by_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            buffer = dense_solve.RestartBuffer(Path(tmp)/'restarts.json', 10, 2, .1)
            for ply in range(4):
                buffer.add(self.entry(ply, -.5, added_at=10.*ply))
            buffer.observe(('s', 0, 1, 'attack'), 'main/000030', .6, '0003')
            buffer.observe(('s', 0, 1, 'attack'), 'main/000030', .9, '0001')
            buffer.observe(('s', 0, 1, 'attack'), 'main/000020', .9, '0004')
            buffer.observe(('s', 0, 2, 'attack'), 'main/000020', .9, '0002')
            buffer.observe(('s', 0, 3, 'attack'), 'main/000030', -.9, '0003')
            buffer.refresh([5., 15., 25.], 'main/000030')
            self.assertEqual({e['ply']: (round(e['regret'], 9), e['checkpoint']) for e in buffer.entries.values()},
                             {1: (.2, 'main/000030'), 2: (.75, 'main/000010'), 3: (.95, 'main/000030')})
            self.assertEqual({e['ply']: e.get('observed') for e in buffer.entries.values()},
                             {1: {'main/000030': dict(value=.6, shard='0003')}, 2: None,
                              3: {'main/000030': dict(value=-.9, shard='0003')}})
            buffer.refresh([5., 15., 25., 35.], 'main/000030')
            self.assertEqual(sorted(e['ply'] for e in buffer.entries.values()), [2, 3])
            buffer.refresh([5., 15., 25., 35., 45.], 'main/000030')
            self.assertEqual(sorted(e['ply'] for e in buffer.entries.values()), [3])
            buffer.observe(('s', 0, 3, 'attack'), 'main/000040', .9, '0004')
            buffer.refresh([5., 15., 25., 35., 45.], 'main/000040')
            self.assertEqual(buffer.entries, {})

    def test_observations_wait_for_their_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            buffer = dense_solve.RestartBuffer(Path(tmp)/'restarts.json', 10, 2, .1)
            buffer.observe(('s', 0, 1, 'attack'), 'main/000030', .6, '0003')
            buffer.observe(('t', 0, 1, 'attack'), 'main/000030', .6, '0003')
            buffer.save()
            buffer = dense_solve.RestartBuffer(Path(tmp)/'restarts.json', 10, 2, .1)
            buffer.add(self.entry(1, -.5))
            self.assertEqual(buffer.entries['s', 0, 1, 'attack']['observed'], {'main/000030': dict(value=.6, shard='0003')})
            self.assertEqual(list(buffer.waiting), [('t', 0, 1, 'attack')])
            buffer.refresh([], 'main/000030', solved=lambda shard: shard == 't')
            self.assertEqual((buffer.waiting, round(buffer.entries['s', 0, 1, 'attack']['regret'], 9)), ({}, .2))

    def test_network_entries_ignore_search_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            buffer = dense_solve.RestartBuffer(Path(tmp)/'restarts.json', 10, 2, .1)
            key = ('s', 0, 1, 'attack')
            buffer.observe(key, 'main/000030', -.6, '0003', 'network')
            buffer.add(dict(self.entry(1, -.6), value_source='network'))
            buffer.observe(key, 'main/000030', 1., '0004')
            buffer.refresh([], 'main/000030')
            self.assertEqual(buffer.entries[key]['regret'], .8)
            self.assertEqual(buffer.entries[key]['observed'],
                             {'main/000030': dict(value=-.6, shard='0003', value_source='network')})
            buffer.observe(key, 'main/000040', .9, '0005', 'network')
            buffer.refresh([], 'main/000040')
            self.assertEqual(buffer.entries, {})

    def test_one_shot_pass_refreshes_after_its_last_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = new_run(Path(tmp)/'run')
            coordinator = dense_solve.Pass(run, run, SMALL)
            with unittest.mock.patch.object(coordinator, 'refresh') as refresh:
                coordinator.loop(once=True)
            self.assertEqual(refresh.call_count, 2)

    def test_exports_count_the_learner_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            for cid, created in (('main/000010', 1.), ('main/000020', 2.), ('wide/000030', 3.)):
                path = Path(tmp)/'checkpoints'/cid
                path.mkdir(parents=True)
                (path/'manifest.json').write_text(json.dumps(dict(created_at=created)))
            self.assertEqual(dense_solve.exports(tmp, 'main'), ([1., 2.], 'main/000020'))
            self.assertEqual(dense_solve.exports(tmp, 'none'), ([], None))


class RestartActorTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run = new_run(Path(tmp.name)/'run', window_min_rows=1)
        shard_of(self.run, '1000000000001', [episode(PREFIX+[[5, 5], [6, 6]], -1), episode(PREFIX, -1)])
        self.buffer = [dict(shard='1000000000001', game=0, ply=6, kind='attack', regret=.8, plies_to_proof=0),
                       dict(shard='1000000000001', game=1, ply=4, kind='defence', regret=.2, plies_to_proof=2)]
        (self.run/'restarts.json').write_text(json.dumps(dict(entries=self.buffer)))

    def test_draws_follow_regret_and_temperature(self):
        for temperature, share in ((1., .8), (.5, .64/.68), (1e-3, 1.)):
            restarts = dense_selfplay.Restarts(self.run, temperature, 10)
            rng = np.random.default_rng(0)
            draws = [restarts.draw(rng) for _ in range(4000)]
            first = sum(d[0]['ply'] == 6 for d in draws)/len(draws)
            self.assertAlmostEqual(first, share, delta=.02)
            self.assertLessEqual({len(d[1]) for d in draws}, {4, 6})
            self.assertTrue(all(d[1] == PREFIX[:d[0]['ply']] for d in draws))
        capped = dense_selfplay.Restarts(self.run, 1., 5)
        self.assertEqual({capped.draw(np.random.default_rng(k))[0]['ply'] for k in range(20)}, {4})
        self.assertIsNone(dense_selfplay.Restarts(self.run, 1., 4).draw(np.random.default_rng(0)))
        (self.run/'restarts.json').write_text(json.dumps(dict(entries=[dict(self.buffer[0], shard='gone')])))
        self.assertIsNone(dense_selfplay.Restarts(self.run, 1., 10).draw(np.random.default_rng(0)))
        (self.run/'restarts.json').write_text('{"entries": [')
        self.assertIsNone(dense_selfplay.Restarts(self.run, 1., 10).draw(np.random.default_rng(0)))
        (self.run/'restarts.json').unlink()
        self.assertIsNone(dense_selfplay.Restarts(self.run, 1., 10).draw(np.random.default_rng(0)))

    def test_restart_game_replays_the_prefix_without_rows(self):
        settings = replace(dense_config.ActorSettings(), full_sims=4, cheap_sims=2, root_samples=2, max_plies=12,
                           full_fraction=.5, opening_random_plies=3., leaf_batch=64)
        model = tiny_model()
        games = [dense_selfplay.SelfPlayGame([model, model], settings, 7, restart=(self.buffer[0], PREFIX)),
                 dense_selfplay.SelfPlayGame([model, model], settings, 8)]
        engine = dense_selfplay.Engine(64)
        for game in games:
            engine.add(game)
        while engine.slots:
            engine.step()
        (restart, rows), (plain, _) = [game.episode() for game in games]
        self.assertEqual((restart['origin'], plain['origin']), ('restart', 'selfplay'))
        self.assertEqual(restart['restart'], {k: self.buffer[0][k] for k in dense_selfplay.RESTART_SOURCE})
        self.assertNotIn('restart', plain)
        self.assertEqual((restart['moves'][:6], restart['opening_plies']), (PREFIX, 6))
        self.assertEqual(restart['root_values'][:6], [None]*6)
        self.assertEqual(restart['network_values'][:6], [None]*6)
        self.assertEqual(len(restart['network_values']), len(restart['moves']))
        self.assertEqual(restart['full_search'][:6], [False]*6)
        self.assertTrue(all(v is not None for v in restart['root_values'][6:]))
        self.assertEqual([r['ply'] for r in rows], list(range(6, len(restart['moves']))))
        manifest = dense_data.write_shard(self.run/'shards'/'1000000000002', dict(actor_sha256='a'*64),
                                          [restart, plain], [dict(r, game=0) for r in rows])
        self.assertEqual((manifest['counts']['restart_games'], manifest['counts']['forced_plies']), (1, 6))
        (self.run/'shards'/'1000000000002'/'manifest.json').write_text(json.dumps(dict(
            manifest, identity=dict(manifest['identity'], process=0))))
        totals = dense_selfplay.published(self.run, 0)
        self.assertEqual((totals['positions'], totals['plies']), (len(rows), len(restart['moves'])))
        dense_bootstrap.check(self.run/'shards'/'1000000000002')
        window = dense_data.ReplayWindow(self.run, 10**6, 1)
        trained = [window.ref(*k) for k in window.index] + [window.ref(*k) for k in window.validation]
        plies = [r.row['ply'] for r in trained if r.shard == '1000000000002']
        self.assertTrue(plies and min(plies) >= 6)
        samples, targets = dense_data.examples(window, [r for r in trained if r.shard == '1000000000002'],
                                               np.random.default_rng(0))
        self.assertEqual(len(samples), len(plies))

    def test_proven_roots_share_an_extra_network_prediction(self):
        model = tiny_model()
        model.cache.capacity = 0
        history = [list(m) for m in winning_game()[:9]]
        settings = replace(dense_config.ActorSettings(hybrid_scheduler=False), full_sims=4, cheap_sims=2, root_samples=2,
                           max_plies=len(history)+30, full_fraction=1., opening_random_plies=0.,
                           solver_root_nodes=135, solver_fixed_budgets=True, adjudicate_proven=True,
                           proven_line_rows=True)
        games = [dense_selfplay.SelfPlayGame([model, model], settings, seed,
                 book=(dict(ply=len(history)), history)) for seed in (3, 4)]
        engine = dense_selfplay.Engine(64, solver_async=False, leaf_nodes=135)
        self.addCleanup(engine.close)
        for game in games:
            self.addCleanup(game.game.close)
            for tree in game.trees.values():
                self.addCleanup(tree.close)
            engine.add(game)
        while engine.slots or engine.closing:
            engine.step()
        ply = len(history)
        self.assertEqual([game.values[ply] for game in games], [1., 1.])
        self.assertEqual([game.network_values[ply] for game in games], [None, None])
        self.assertTrue(all(any(row.get('line') for row in game.rows) for game in games))
        before = [game.moves.copy() for game in games]
        histories = {dense_selfplay.position_key(h): h for game in games for row in game.rows
                     if game.values[row['ply']] is not None
                     for h in [np.asarray(game.moves[:row['ply']], np.int64).reshape(-1, 2)]}
        expected = dict(zip(histories, (p[2][0] for p in model.evaluator.evaluate(list(histories.values())))))
        model.cache.capacity = 256
        with unittest.mock.patch.object(model.evaluator, 'evaluate', wraps=model.evaluator.evaluate) as evaluate:
            self.assertEqual(dense_selfplay.record_network_values(games), [len(histories)])
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual([game.moves for game in games], before)
            self.assertEqual([game.values[ply] for game in games], [1., 1.])
            for game in games:
                self.assertTrue(all(v is None for v in game.network_values[:ply]))
                for row in game.rows:
                    h = np.asarray(game.moves[:row['ply']], np.int64).reshape(-1, 2)
                    self.assertEqual(game.network_values[row['ply']], expected[dense_selfplay.position_key(h)])
                    game.network_values[row['ply']] = None
            self.assertEqual(dense_selfplay.record_network_values(games), [])
            self.assertEqual(evaluate.call_count, 1)
        episodes, rows = [], []
        for game in games:
            episode, items = game.episode()
            rows.extend(dict(row, game=len(episodes)) for row in items)
            episodes.append(episode)
        dense_data.write_shard(self.run/'shards'/'1000000000003', dict(actor_sha256='a'*64), episodes, rows)
        window = dense_data.ReplayWindow(self.run, capacity_rows=1000)
        priorities = {(ref.row['game'], ref.row['ply']): w
                      for k, w in zip(window.regret_positions, window.regret_weights)
                      for ref in [window.ref(*window.index[k])] if ref.shard == '1000000000003'}
        lines = [row for row in rows if row.get('line')]
        self.assertTrue(lines)
        for row in lines:
            value = episodes[row['game']]['network_values'][row['ply']]
            self.assertAlmostEqual(priorities[row['game'], row['ply']], (1-row['proven']*value)/2)

    def test_network_restart_gets_a_prediction_without_a_new_proof(self):
        model = tiny_model()
        model.cache.capacity = 0
        source = dict(self.buffer[0], value_source='network')
        settings = replace(dense_config.ActorSettings(), full_sims=2, cheap_sims=2, root_samples=2,
                           max_plies=8, tactics=False, opening_random_plies=0.)
        game = dense_selfplay.SelfPlayGame([model, model], settings, 7, restart=(source, PREFIX))
        self.addCleanup(game.game.close)
        for tree in game.trees.values():
            self.addCleanup(tree.close)
        engine = dense_selfplay.Engine(64)
        self.addCleanup(engine.close)
        engine.add(game)
        while engine.slots:
            engine.step()
        self.assertTrue(all(not row.get('proven') for row in game.rows))
        self.assertIsNotNone(game.network_values[6])
        dense_selfplay.record_network_values([game])
        self.assertTrue(all(value is not None for value in game.network_values[6:]))
        e, rows = game.episode()
        self.assertEqual(e['restart']['value_source'], 'network')
        self.assertEqual(e['network_values'][:6], [None]*6)

    def test_unproven_rows_keep_values_without_a_cache(self):
        model = tiny_model()
        model.cache.capacity = 0
        source = dict(self.buffer[0], ply=5)
        settings = replace(dense_config.ActorSettings(), full_sims=2, cheap_sims=2, root_samples=2,
                           max_plies=7, tactics=False, opening_random_plies=0.)
        game = dense_selfplay.SelfPlayGame([model, model], settings, 7, restart=(source, PREFIX[:5]))
        self.addCleanup(game.game.close)
        for tree in game.trees.values():
            self.addCleanup(tree.close)
        engine = dense_selfplay.Engine(64)
        self.addCleanup(engine.close)
        engine.add(game)
        while engine.slots:
            engine.step()
        self.assertTrue(all(not row.get('proven') for row in game.rows))
        self.assertIsNotNone(game.network_values[5])
        self.assertIsNone(game.network_values[6])
        dense_selfplay.record_network_values([game])
        expected = model.evaluator.evaluate([np.asarray(game.moves[:ply], np.int64) for ply in (5, 6)])
        np.testing.assert_allclose(game.network_values[5:], [p[2][0] for p in expected], atol=1e-7)
        self.assertEqual(game.network_values[:5], [None]*5)
        self.assertEqual(engine.root_predictions, {})

    def test_worker_starts_games_from_the_buffer(self):
        config = dense_config.load(self.run)
        for fraction, origin in ((0., 'restart'), (1., 'selfplay')):
            with self.subTest(validation_fraction=fraction):
                (self.run/'config.json').unlink()
                dense_config.save(self.run, replace(config, learner=replace(config.learner, validation_fraction=fraction),
                    actor=dense_config.ActorSettings(games_in_flight=2, leaf_batch=64, full_sims=2, cheap_sims=2,
                        root_samples=2, max_plies=10, cache_positions=256, shard_games=2, restart_fraction=1.)))
                dense_selfplay.worker(SimpleNamespace(run=str(self.run), worker=0, games=2, initial_model=None))
                name = [p.name for p in dense_data.shard_dirs(self.run)][-1]
                episodes, _ = dense_data.read_shard(self.run/'shards'/name, policies=False)
                self.assertEqual([e['origin'] for e in episodes], [origin, origin])
                self.assertEqual(dense_data.manifest(self.run/'shards'/name)['counts']['restart_games'],
                                 2 if origin == 'restart' else 0)
                if origin == 'restart':
                    for e in episodes:
                        self.assertEqual(e['moves'][:e['restart']['ply']], PREFIX[:e['restart']['ply']])

    def test_validation_ancestor_is_not_restarted(self):
        config = dense_config.load(self.run)
        (self.run/'config.json').unlink()
        dense_config.save(self.run, replace(config, learner=replace(config.learner, validation_fraction=.5)))
        ancestor = episode(PREFIX+[[5, 5], [6, 6]], -1)
        child = episode(PREFIX+[[5, 5], [7, 6]], -1, origin='restart', restart=self.buffer[0])
        self.assertTrue(dense_data.holdout(ancestor, .5))
        self.assertFalse(dense_data.holdout(child, .5))
        shard_of(self.run, '1000000000002', [child])
        (self.run/'restarts.json').write_text(json.dumps(dict(entries=[dict(self.buffer[0],
            shard='1000000000002', game=0, ply=7)])))
        self.assertIsNone(dense_selfplay.Restarts(self.run, 1., 10).draw(np.random.default_rng(0)))

    def test_learner_cannot_override_the_shared_validation_split(self):
        with unittest.mock.patch('sys.argv', ['dense_learn.py', '--run', str(self.run),
                '--validation-fraction', '.5', '--steps', '0', '--workers', '0']):
            with self.assertRaisesRegex(ValueError, 'actors and learners share the validation split'):
                dense_learn.main()

    def test_failed_workers_are_retried_once_and_rejections_stop_the_pass(self):
        for name in ('1000000000002', '1000000000003'):
            shard_of(self.run, name, [episode(PREFIX, -1)])
        codes = {'1000000000003': [1, 1], '1000000000002': [0]}
        started = []

        def spawn(name):
            started.append(name)
            code = codes[name].pop(0)
            if code == 0:
                write_json(out/'.solve'/f'{name}.json', dict(windows=[], entries=[], observations=[],
                                                             stats=dense_solve.new_stats()))
            return SimpleNamespace(poll=lambda: code)
        out = self.run.parent/'out'
        (out/'.solve').mkdir(parents=True)
        coordinator = dense_solve.Pass(self.run, out, replace(SMALL, solve_workers_min=1))
        while coordinator.step(limit=2, spawn=spawn):
            pass
        self.assertEqual(started, ['1000000000003', '1000000000002', '1000000000003'])
        self.assertTrue((out/'shards'/'1000000000002'/dense_data.SIDECAR).exists())
        status = json.loads((out/'solver-status.json').read_text())
        self.assertEqual((status['shards_failed'], status['shards_done'], status['shards_pending']),
                         ({'1000000000003': 2}, 1, 0))
        events = [json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()]
        self.assertEqual([e['attempts'] for e in events if e['kind'] == 'error'], [1, 2])
        coordinator.running['1000000000001'] = (SimpleNamespace(poll=lambda: dense_solve.REJECTED), 0.)
        with self.assertRaises(RuntimeError):
            coordinator.step(spawn=spawn)

    def test_a_failed_shard_waits_for_first_attempts_still_running(self):
        for name in ('1000000000002', '1000000000003'):
            shard_of(self.run, name, [episode(PREFIX, -1)])
        out = self.run.parent/'out'
        (out/'.solve').mkdir(parents=True)
        codes = {'1000000000003': 1, '1000000000002': None}
        started = []

        def spawn(name):
            started.append(name)
            return SimpleNamespace(poll=lambda: codes[name])
        coordinator = dense_solve.Pass(self.run, out, replace(SMALL, solve_workers_min=2))
        coordinator.step(limit=2, spawn=spawn)
        self.assertTrue(coordinator.step(limit=2, spawn=spawn))
        codes['1000000000003'] = None
        self.assertEqual((started, list(coordinator.running)), (['1000000000003', '1000000000002'], ['1000000000002']))
        codes['1000000000002'] = 1
        coordinator.step(limit=2, spawn=spawn)
        self.assertEqual(sorted(coordinator.running), ['1000000000002', '1000000000003'])

    def test_settings_bounds(self):
        with self.assertRaises(ValueError):
            dense_config.ActorSettings(restart_fraction=1.5)
        with self.assertRaises(ValueError):
            dense_config.ActorSettings(restart_temperature=0.)


class ProvenLabelTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run = Path(tmp.name)
        write_games(self.run/'shards'/'000001', [(winning_game(), 0, [0.]*12)])

    def label(self, plies):
        dense_solve.write_sidecar(self.run/'shards'/'000001', [dict(game=0, first_ply=plies[0], last_ply=plies[-1],
                                                                    mover=0, plies=plies)])

    def test_proof_pass_emits_losing_owner_deblunder_record(self):
        new_run(self.run)
        solver = dense_solve.Solver(self.run, self.run, SMALL, None)
        def query(where, history, *args):
            return ([(20, 20), (21, 20)], 1, 'verified') if len(history) in (5, 6) else None
        with unittest.mock.patch.object(solver, 'query', side_effect=query), \
                unittest.mock.patch.object(dense_solve, 'worth_solving', return_value=True):
            result = solver.solve('000001')
        self.assertEqual(result['deblunders'], [dict(kind='deblunder', game=0, first_ply=5, owner=1)])
        dense_solve.Pass(self.run, self.run, SMALL).record('000001', result)
        path = self.run/'shards'/'000001'
        self.assertEqual(dense_data.proof_windows(path), result['windows'])
        self.assertEqual(dense_data.proof_records(path)[-1], result['deblunders'][0])
        self.assertEqual(dense_data.proof_labels(path), {(0, 5), (0, 6)})

    def test_deblunder_blend_signs_default_bytes_and_exact_precedence(self):
        write_games(self.run/'shards'/'000002', [(winning_game(), 1, [.2]*12)])
        window = dense_data.ReplayWindow(self.run, 1000, 100)
        refs = [window.ref(name, i) for name in ('000001', '000002') for i in range(12)]
        before = dense_data.collate_arrays(*dense_data.examples(window, refs, np.random.default_rng(4)))
        for name, owner, first in (('000001', 1, 5), ('000002', 0, 7)):
            dense_solve.write_sidecar(self.run/'shards'/name, [dict(kind='deblunder', game=0, first_ply=first, owner=owner)])
        window.refresh()
        refs = [window.ref(name, i) for name in ('000001', '000002') for i in range(12)]
        after = dense_data.collate_arrays(*dense_data.examples(window, refs, np.random.default_rng(4), deblunder_weight=0.))
        self.assertEqual(before.keys(), after.keys())
        for size in before:
            self.assertEqual(before[size].keys(), after[size].keys())
            for key in before[size]:
                self.assertEqual(before[size][key].tobytes(), after[size][key].tobytes(), key)
        refs[1].row['proven'] = -1
        refs[12].row['proven'] = 1
        for options in ({}, dict(outcome_lam=.8), dict(calibration=dense_data.Calibration((0.,)*dense_data.CALIBRATION_FEATURES, .3))):
            # The outcome blend is the same under each learner target mode.
            _, targets = dense_data.examples(window, refs, np.random.default_rng(4), deblunder_weight=.25, **options)
            for ref, target in zip(refs, targets):
                owner, first = (1, 5) if ref.shard == '000001' else (0, 7)
                changed = ref.row['player'] == owner and ref.row['ply'] < first and not ref.row['proven']
                self.assertEqual(target['deblundered'], float(changed))
                self.assertEqual(target['outcome'], float(ref.row['player'] == ref.episode['winner']))
                if changed:
                    self.assertEqual((target['value'], target['outcome_target'], target['exact']), (.25, .25, 0.))
                    self.assertEqual(ref.row['proven'], 0)
                elif ref.row['proven']:
                    self.assertEqual((target['value'], target['exact'], target['outcome_weight']),
                                     (float(ref.row['proven'] > 0), 1., 0.))
                else:
                    self.assertEqual(target['outcome_target'], target['outcome'])

    def test_deblunder_stops_after_previous_window_and_refreshes_validation(self):
        window = dense_data.ReplayWindow(self.run, 1000, 100)
        sets = dense_data.ValidationSets(self.run, 1., 0, 100, 100)
        sets.refresh()
        dense_solve.write_sidecar(self.run/'shards'/'000001', [
            dict(game=0, first_ply=3, last_ply=4, mover=0, plies=[3, 4],
                 proof_action={'3': [list(m) for m in winning_game()[3:5]], '4': [list(winning_game()[4])]}),
            dict(game=0, first_ply=9, last_ply=10, mover=1, plies=[9, 10]),
            dict(kind='deblunder', game=0, first_ply=9, owner=1)])
        window.refresh(); sets.refresh()
        for source, refs in ((window, [window.ref('000001', i) for i in range(12)]),
                             (sets, sets.subsets['fresh', 'held'])):
            _, targets = dense_data.examples(source, refs, np.random.default_rng(0), deblunder_weight=.5, proof_policy_weight=.5)
            self.assertEqual(sorted(r.row['ply'] for r, t in zip(refs, targets) if t['deblundered']), [5, 6])
            for ref, target in zip(refs, targets):
                if ref.row['ply'] in (3, 4):
                    self.assertTrue(ref.row['proof_action'])
                    self.assertFalse(np.array_equal(target['policy'], source.policy(ref)))
                    self.assertEqual((target['value'], target['deblundered']), (1., 0.))

    def test_deblunder_loss_and_validation(self):
        torch.set_num_threads(2)
        dense_solve.write_sidecar(self.run/'shards'/'000001', [dict(kind='deblunder', game=0, first_ply=9, owner=1)])
        config = dense_config.RunConfig(device='cpu', model=dense_config.ModelSettings(**asdict(TINY)),
                                        learner=dense_config.LearnerSettings(batch=8, deblunder_weight=.25, validation_fraction=1.))
        learner = dense_learn.Learner(self.run, config.learner, config)
        sets = dense_data.ValidationSets(self.run, 1., 0, 100, 100)
        metrics = learner.validate_sources(sets)
        rows = learner.row_losses(learner.render(sets, sets.subsets['fresh', 'held']))
        changed = rows['deblundered'] > 0
        self.assertEqual(metrics['fresh_deblundered_rows'], 4)
        self.assertAlmostEqual(metrics['fresh_value_bce_deblundered'], float(rows['value_bce'][changed].mean()))
        self.assertTrue(np.all(rows['outcome'][changed] == 0))
        batch = dense_data.collate(*dense_data.examples(sets, sets.subsets['fresh', 'held'], np.random.default_rng(0),
                                                        **learner.targets()))
        for b in batch.values():
            with torch.no_grad():
                losses, _ = dense_learn.head_losses(learner.ema, b, learner.device, learner.memory_format)
                logit = dense_learn.forward(learner.ema, b['planes'], learner.device, learner.memory_format)[0]['value_logit']
                expected = dense_learn.hexnet.value_loss(logit, b['outcome_target'], b['outcome_weight'])
                torch.testing.assert_close(losses[-1], expected)
        window = dense_data.ReplayWindow(self.run, 1000, 100, validation_fraction=1.)
        with unittest.mock.patch.object(dense_learn, 'VALIDATION_ROWS', 16):
            aggregate = learner.validate(window)
        self.assertGreater(aggregate['deblundered_rows'], 0)
        self.assertGreater(aggregate['value_bce_deblundered'], 0)

    def test_deblunder_flag(self):
        import argparse
        parser = argparse.ArgumentParser()
        dense_config.add_arguments(parser, dense_config.LearnerSettings)
        settings = dense_config.LearnerSettings()
        self.assertEqual(settings.deblunder_weight, 0.)
        self.assertEqual(dense_config.override(settings, parser.parse_args(['--deblunder-weight', '.25'])).deblunder_weight, .25)
        self.assertIn('deblunder_weight', dense_learn.KEEP)
        for weight in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                replace(settings, deblunder_weight=weight)

    def test_window_applies_labels_when_the_sidecar_appears(self):
        window = dense_data.ReplayWindow(self.run, 1000, 100)
        self.assertEqual(window.proven_rows, 0)
        self.label([7, 8, 11])
        window.refresh()
        self.assertEqual(window.proven_rows, 3)
        refs = [window.ref('000001', i) for i in range(12)]
        self.assertEqual([r.row['ply'] for r in refs if r.row['proven']], [7, 8, 11])
        _, targets = dense_data.examples(window, refs, np.random.default_rng(0), proven_weight=3.)
        self.assertEqual([(t['value'], t['value_weight']) for t, r in zip(targets, refs) if r.row['proven']], [(1., 3.)]*3)
        self.assertEqual(dense_data.ReplayWindow(self.run, 1000, 100).proven_rows, 3)

    def test_validation_sets_apply_labels(self):
        sets = dense_data.ValidationSets(self.run, 1., 0, 100, 100)
        sets.refresh()
        self.assertFalse(any(r.row.get('proven') for r in sets.subsets['fresh', 'held']))
        self.label([7, 8])
        sets.refresh()
        self.assertEqual(sorted(r.row['ply'] for r in sets.subsets['fresh', 'held'] if r.row.get('proven')), [7, 8])

    def test_sidecar_actions_reach_replay_and_fixed_panels(self):
        window = dense_data.ReplayWindow(self.run, 1000, 100)
        sets = dense_data.ValidationSets(self.run, 1., 0, 100, 100)
        sets.refresh()
        actions = {'7': [list(m) for m in winning_game()[7:9]], '8': [list(winning_game()[8])]}
        dense_solve.write_sidecar(self.run/'shards'/'000001', [dict(game=0, plies=[7, 8], proof_action=actions)])
        window.refresh()
        sets.refresh()
        for reader, refs in ((window, [window.ref('000001', i) for i in (7, 8)]),
                             (sets, [r for r in sets.subsets['fresh', 'held'] if r.row['ply'] in (7, 8)])):
            for ref in refs:
                self.assertEqual(ref.row['proof_action'], actions[str(ref.row['ply'])])
                self.assertEqual(ref.row['proven'], 1)
            self.assertTrue(all(t['policy_weight'] > 0 for t in dense_data.examples(
                reader, refs, np.random.default_rng(0), proof_policy_weight=1.)[1]))

    def test_certificate_policy_fills_only_missing_targets(self):
        window = dense_data.ReplayWindow(self.run, 1000, 100)
        actions = {'7': [list(m) for m in winning_game()[7:9]], '8': [list(winning_game()[8])]}
        dense_solve.write_sidecar(self.run/'shards'/'000001', [dict(game=0, plies=[7, 8], proof_action=actions)])
        window.refresh()
        refs = [window.ref('000001', i) for i in (7, 8, 9)]
        refs[2].row.update(proven=-1, proof_action=actions['8'])
        ordinary = window.policy
        def policy(ref):
            return np.empty(0, np.float32) if ref.row['ply'] == 8 else ordinary(ref)
        with unittest.mock.patch.object(window, 'policy', side_effect=policy):
            samples, base = dense_data.examples(window, refs, np.random.default_rng(0))
            _, targets = dense_data.examples(window, refs, np.random.default_rng(0),
                    proof_policy_weight=.25, proof_policy_missing_only=True)
            _, disabled = dense_data.examples(window, refs, np.random.default_rng(0), proof_policy_missing_only=True)
        for i, (before, after) in enumerate(zip(base, targets)):
            for key in before:
                np.testing.assert_array_equal(before[key], disabled[i][key])
                if i != 1 or key not in ('policy', 'policy_weight'):
                    np.testing.assert_array_equal(before[key], after[key])
        self.assertEqual(targets[1]['policy_weight'], .25)
        selected = samples[1].actions[targets[1]['policy'] > 0].tolist()
        self.assertEqual(selected, actions['8'])
        self.assertAlmostEqual(float(targets[1]['policy'].sum()), 1.)

    def test_validation_reports_value_regret_on_proven_rows(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        run = self.run/'sources'
        source_shard(run/'shards'/'1000000000001', 2, 'x', checkpoint='main/000010', winner=0)
        config = dense_config.RunConfig(device='cpu', model=dense_config.ModelSettings(**asdict(TINY)),
                                        learner=dense_config.LearnerSettings(batch=8, validation_fraction=.5, proof_policy_weight=.5))
        learner = dense_learn.Learner(run, config.learner, config)
        sets = dense_data.ValidationSets(run, .5, config.seed, limit=12, quota=12)
        sets.refresh()
        held = sets.subsets['newest', 'held']
        self.assertTrue(held)
        chosen = held[:3]
        windows = [dict(game=r.row['game'], plies=[r.row['ply']],
                        proof_action={str(r.row['ply']): [r.episode['moves'][r.row['ply']]]}) for r in chosen]
        dense_solve.write_sidecar(run/'shards'/'1000000000001', windows)
        out = learner.validate_sources(sets)
        rows = learner.row_losses(learner.render(sets, sets.subsets['newest', 'held']))
        proven = rows['proven'] != 0
        self.assertEqual(out['newest_proven_rows'], len({(r.row['game'], r.row['ply']) for r in chosen}))
        self.assertAlmostEqual(out['newest_value_regret_proven'], float(np.mean(1-np.exp(-rows['value_bce'][proven]))))
        self.assertTrue(0 < out['newest_value_regret_proven'] < 1)
        self.assertEqual(out['newest_policy_ce_proof_rows'], int(proven.sum()))
        self.assertAlmostEqual(out['newest_policy_ce_proof'], float(rows['policy_ce'][proven].mean()))
        self.assertIsNone(out['converted_policy_ce_proof'])
        fields = dense_learn.validation_fields(dict(validation=None, validation_sources=out))
        self.assertEqual(fields['newest_policy_ce_proof'], out['newest_policy_ce_proof'])

    def test_validation_splits_the_outcome_loss_by_exact_label(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        run = self.run/'split'
        source_shard(run/'shards'/'1000000000001', 2, 'x', checkpoint='main/000010', winner=0)
        dense_solve.write_sidecar(run/'shards'/'1000000000001', [dict(game=g, plies=[3, 4, 5]) for g in range(6)])
        config = dense_config.RunConfig(device='cpu', model=dense_config.ModelSettings(**asdict(TINY)),
                                        learner=dense_config.LearnerSettings(batch=8, validation_fraction=.5))
        learner = dense_learn.Learner(run, config.learner, config)
        sets = dense_data.ValidationSets(run, .5, config.seed, limit=12, quota=12)
        out = learner.validate_sources(sets)
        for source in ('fresh', 'newest'):
            r = learner.row_losses(learner.render(sets, sets.subsets[source, 'held']))
            exact = r['proven'] != 0
            self.assertTrue(exact.any() and (~exact).any())
            self.assertEqual((out[f'{source}_outcome_bce_exact_rows'], out[f'{source}_outcome_bce_unproven_rows']),
                             (int(exact.sum()), int((~exact).sum())))
            self.assertAlmostEqual(out[f'{source}_outcome_bce_exact'], float(r['outcome_bce'][exact].mean()))
            self.assertAlmostEqual(out[f'{source}_outcome_bce_unproven'], float(r['outcome_bce'][~exact].mean()))
        self.assertEqual((out['converted_outcome_bce_exact'], out['converted_outcome_bce_exact_rows']), (None, 0))
        window = dense_data.ReplayWindow(run, 1000, 10, validation_fraction=.5)
        self.assertTrue(window.validation)
        v = learner.validate(window)
        self.assertEqual(v['outcome_bce_exact_rows']+v['outcome_bce_unproven_rows'], dense_learn.VALIDATION_ROWS)
        self.assertTrue(v['outcome_bce_exact_rows'] > 0 and v['outcome_bce_unproven_rows'] > 0)
        self.assertTrue(0 < v['outcome_bce_exact'] and 0 < v['outcome_bce_unproven'])
        fields = dense_learn.validation_fields(dict(validation=v, validation_sources=out))
        self.assertEqual((fields['outcome_bce_exact'], fields['newest_outcome_bce_unproven_rows'], fields['next_ce']),
                         (v['outcome_bce_exact'], out['newest_outcome_bce_unproven_rows'], v['opponent_ce']))


class DashboardTests(unittest.TestCase):
    def test_actor_tile_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            episodes = [episode(PREFIX, -1), episode(PREFIX, -1, origin='restart', restart=dict(ply=2)),
                        episode(PREFIX, -1, origin='selfplay'), episode(PREFIX, -1)]
            shard_of(run, '1000000000001', episodes)
            (run/'solver-status.json').write_text(json.dumps(dict(buffer_size=17, verified=5, verify_timeouts=2)))
            state = dashboard.dense_run(run, dict(created_at=0.))
            self.assertEqual((state['actor']['restart_buffer'], state['data']['restart_share_6h']), (17, .25))
            self.assertEqual((state['actor']['proofs_verified'], state['actor']['verify_timeouts']), (5, 2))
            self.assertIn('verify_timeouts', (Path(dashboard.__file__).resolve().parents[1]/'web'/'training.html').read_text(encoding='utf-8'))
            self.assertIn('restart_share_6h', (Path(dashboard.__file__).resolve().parents[1]/'web'/'training.html').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
