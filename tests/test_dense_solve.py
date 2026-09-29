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
from dense_data import player_at
from forcing_material import worth_solving
import hexnet
from hexo import Game
from tactical_proof import NativeTactics
from tests.test_dense import source_shard, winning_game, write_games
from tests.test_dense_solver import TINY, tiny_model
from tests.test_tactical_proof import FIXTURE

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
        start = e['restart']['ply'] if e.get('origin') == 'restart' else 0
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

    def solve(self, out=None, settings=SMALL):
        tactics = Recorder(self.engine)
        solver = dense_solve.Pass(self.run, out or self.run, settings, tactics, clock=lambda: 1000.)
        return solver, tactics, solver.shard('1000000000001')

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

    def test_same_shard_same_sidecar_and_buffer(self):
        outputs = []
        for k in range(2):
            out = self.run.parent/f'out{k}'
            self.solve(out)
            outputs.append(((out/'shards'/'1000000000001'/'proofs.jsonl').read_bytes(),
                            json.loads((out/'restarts.json').read_text())['entries']))
        self.assertEqual(outputs[0], outputs[1])
        self.assertFalse((self.run/'shards'/'1000000000001'/'proofs.jsonl').exists())

    def test_loop_is_restart_safe_and_writes_status(self):
        out = self.run.parent/'out'
        tactics = Recorder(self.engine)
        solver = dense_solve.Pass(self.run, out, SMALL, tactics)
        solver.loop(once=True)
        calls = len(tactics.calls)
        dense_solve.Pass(self.run, out, SMALL, tactics).loop(once=True)
        self.assertEqual(len(tactics.calls), calls)
        status = json.loads((out/'solver-status.json').read_text())
        self.assertEqual((status['stage'], status['shards_pending'], status['buffer_size']),
                         ('idle', 0, len(solver.buffer.entries)))
        for key in ('positions_per_hour', 'hits_per_hour', 'gate_pass_rate', 'transient', 'persistent', 'window_plies',
                    'buffer_mean_regret', 'busy_fraction'):
            self.assertIn(key, status)

    def test_rejected_independent_check_aborts_with_an_event(self):
        with unittest.mock.patch.object(dense_solve, 'independent_verify', side_effect=ValueError('bad')):
            with self.assertRaises(RuntimeError):
                self.solve()
        events = [json.loads(line) for line in (self.run/'events.jsonl').read_text().splitlines()]
        self.assertEqual([e['kind'] for e in events], ['error'])
        self.assertIsNone(dense_data.proof_windows(self.run/'shards'/'1000000000001'))

    def test_restart_games_record_observations_instead_of_early_entries(self):
        source = dict(shard='1000000000001', game=0, ply=5, kind='attack', regret=.5, plies_to_proof=0)
        moves = [list(m) for m in winning_game()]
        restart = dict(episode(moves, 0, origin='restart', restart=source), root_values=[None]*5+[.4]*(len(moves)-5))
        shard_of(self.run, '1000000000002', [restart])
        solver = dense_solve.Pass(self.run, self.run, SMALL, Recorder(self.engine), clock=lambda: 1000.)
        solver.buffer.entries[('1000000000001', 0, 5, 'attack')] = dict(source, side_to_move=0, added_at=0., checkpoint=None)
        windows = solver.shard('1000000000002')
        self.assertTrue(windows and all(w['first_ply'] >= 5 for w in windows))
        self.assertTrue(all(e['ply'] > 5 for e in solver.buffer.entries.values() if e['shard'] == '1000000000002'))
        self.assertEqual(solver.buffer.entries[('1000000000001', 0, 5, 'attack')]['observed'],
                         {'main/000010': dict(value=.4, shard='1000000000002')})


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
            self.assertEqual([e['ply'] for e in json.loads((Path(tmp)/'restarts.json').read_text())['entries']], [3, 9, 2])

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
        for temperature, share in ((1., .8), (.5, .64/.68)):
            restarts = dense_selfplay.Restarts(self.run, temperature, 10)
            rng = np.random.default_rng(0)
            draws = [restarts.draw(rng) for _ in range(4000)]
            first = sum(d[0]['ply'] == 6 for d in draws)/len(draws)
            self.assertAlmostEqual(first, share, delta=.02)
            self.assertEqual({len(d[1]) for d in draws}, {4, 6})
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
        self.assertEqual(restart['full_search'][:6], [False]*6)
        self.assertTrue(all(v is not None for v in restart['root_values'][6:]))
        self.assertEqual([r['ply'] for r in rows], list(range(6, len(restart['moves']))))
        manifest = dense_data.write_shard(self.run/'shards'/'1000000000002', dict(actor_sha256='a'*64),
                                          [restart, plain], [dict(r, game=0) for r in rows])
        self.assertEqual(manifest['counts']['restart_games'], 1)
        dense_bootstrap.check(self.run/'shards'/'1000000000002')
        window = dense_data.ReplayWindow(self.run, 10**6, 1)
        trained = [window.ref(*k) for k in window.index] + [window.ref(*k) for k in window.validation]
        plies = [r.row['ply'] for r in trained if r.shard == '1000000000002']
        self.assertTrue(plies and min(plies) >= 6)
        samples, targets = dense_data.examples(window, [r for r in trained if r.shard == '1000000000002'],
                                               np.random.default_rng(0))
        self.assertEqual(len(samples), len(plies))

    def test_worker_starts_games_from_the_buffer(self):
        config = dense_config.load(self.run)
        (self.run/'config.json').unlink()
        dense_config.save(self.run, replace(config, actor=dense_config.ActorSettings(
            games_in_flight=2, leaf_batch=64, full_sims=2, cheap_sims=2, root_samples=2, max_plies=10,
            cache_positions=256, shard_games=2, restart_fraction=1.)))
        dense_selfplay.worker(SimpleNamespace(run=str(self.run), worker=0, games=2, initial_model=None))
        name = [p.name for p in dense_data.shard_dirs(self.run)][-1]
        episodes, _ = dense_data.read_shard(self.run/'shards'/name, policies=False)
        self.assertEqual([e['origin'] for e in episodes], ['restart', 'restart'])
        self.assertEqual(dense_data.manifest(self.run/'shards'/name)['counts']['restart_games'], 2)
        for e in episodes:
            self.assertEqual(e['moves'][:e['restart']['ply']], PREFIX[:e['restart']['ply']])

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

    def test_validation_reports_value_regret_on_proven_rows(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        run = self.run/'sources'
        source_shard(run/'shards'/'1000000000001', 2, 'x', checkpoint='main/000010', winner=0)
        config = dense_config.RunConfig(device='cpu', model=dense_config.ModelSettings(**asdict(TINY)),
                                        learner=dense_config.LearnerSettings(batch=8, validation_fraction=.5))
        learner = dense_learn.Learner(run, config.learner, config)
        sets = dense_data.ValidationSets(run, .5, config.seed, limit=12, quota=12)
        sets.refresh()
        held = sets.subsets['newest', 'held']
        self.assertTrue(held)
        chosen = held[:3]
        windows = [dict(game=r.row['game'], plies=[r.row['ply']]) for r in chosen]
        dense_solve.write_sidecar(run/'shards'/'1000000000001', windows)
        out = learner.validate_sources(sets)
        rows = learner.row_losses(sets, sets.subsets['newest', 'held'])
        proven = rows['proven'] != 0
        self.assertEqual(out['newest_proven_rows'], len({(r.row['game'], r.row['ply']) for r in chosen}))
        self.assertAlmostEqual(out['newest_value_regret_proven'], float(np.mean(1-np.exp(-rows['value_bce'][proven]))))
        self.assertTrue(0 < out['newest_value_regret_proven'] < 1)


class DashboardTests(unittest.TestCase):
    def test_actor_tile_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            episodes = [episode(PREFIX, -1), episode(PREFIX, -1, origin='restart', restart=dict(ply=2)),
                        episode(PREFIX, -1, origin='selfplay'), episode(PREFIX, -1)]
            shard_of(run, '1000000000001', episodes)
            (run/'solver-status.json').write_text(json.dumps(dict(buffer_size=17)))
            state = dashboard.dense_run(run, dict(created_at=0.))
            self.assertEqual((state['actor']['restart_buffer'], state['data']['restart_share_6h']), (17, .25))
            self.assertIn('restart_share_6h', (Path(dashboard.__file__).parent/'web'/'training.html').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
