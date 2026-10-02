"""Play server: evaluation store, review labels, background jobs and the HTTP surface, with fake engines."""
import json
import os
import subprocess
import sys
import random
import tempfile
import threading
import time
import unittest
import unittest.mock
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import formats
from hexo import Game
from play import (Cancelled, Engines, Evaluations, Handler, PRESETS, SIX_LIBRARIES, SearchChild, Session, book_openings,
                  budget_of, command_of, export, export_path, file_digest, file_identity, import_history, linked_history,
                  model_key, move_row, pair_elo, pick_opening, position_text, presets_of, proof_turns, read_game, review, scan, six_backend)
from process_tree import TreeProcess

STANDARD = PRESETS['bubble']['standard']
SITE = Path(__file__).parent / 'fixtures' / 'hexo-site'


def legal_turn(history):
    game = Game(history)
    try:
        side, moves = game.player, []
        while game.player == side and game.winner < 0:
            move = sorted(game.legal_moves())[0]
            moves.append(list(move))
            game.play(*move)
        return moves
    finally:
        game.close()


SLOW_ENGINE = """import sys
stones = False
for raw in sys.stdin:
    words = raw.split()
    if not words: continue
    if words[0] == 'six': print('sixok', flush=True)
    elif words[0] == 'isready': print('readyok', flush=True)
    elif words[0] == 'position': stones = len(words) > 3
    elif words[0] == 'go' and not stones: print('bestmove 0 0', flush=True)
    elif words[0] == 'quit': break
"""


def wait(condition, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return
        time.sleep(.01)
    raise AssertionError('condition not reached')


class FakeEngines:
    """Plays the first legal cells; `hold` makes evaluations wait until cancelled or released."""

    def __init__(self):
        self.calls, self.turns, self.hold, self.release = [], [], False, threading.Event()

    def evaluate(self, entry, checkpoint, budget, history, watch, live=None, keep=False):
        self.calls.append((checkpoint, dict(budget), [tuple(p) for p in history]))
        while self.hold and not self.release.is_set():
            watch(1)
            time.sleep(.01)
        watch(budget['simulations'])
        moves = legal_turn(history)
        found = dict(moves=moves, value=.5, top=[[*moves[0], .9, .5]], proof=None, line=[], threat=[], ms=1)
        return found, budget, f'{model_key(export_path(entry, checkpoint))}:none' + (':kept' if keep else '')

    def evaluate_many(self, entry, checkpoint, budget, histories, watch):
        return [self.evaluate(entry, checkpoint, budget, history, watch) for history in histories]

    def solver_build(self):
        return 'none'

    def effective(self, budget):
        return budget

    def turn(self, entry, budget, history, stop, checkpoint=None):
        self.turns.append((entry['id'], checkpoint, dict(budget)))
        return legal_turn(history)

    def close(self):
        self.release.set()


RUN = tempfile.TemporaryDirectory()
for number in (1, 2):
    (Path(RUN.name) / f'checkpoints/main/00000{number}').mkdir(parents=True)
    (Path(RUN.name) / f'checkpoints/main/00000{number}/ema.pt').write_bytes(bytes([number]))


def tearDownModule():
    RUN.cleanup()


def entries():
    bubble = dict(kind='bubble', presets=PRESETS['bubble'], checkpoints=['main/000002', 'main/000001'], path=Path(RUN.name))
    return {'bubble:fake': dict(bubble, id='bubble:fake', name='fake'),
            'bubble:fake~2': dict(bubble, id='bubble:fake~2', name='fake', checkpoints=['main/000001']),
            'native:Native': dict(id='native:Native', name='Native', kind='native', presets=PRESETS['native'])}


class Store(unittest.TestCase):
    def test_appends_reloads_prefers_deepest_and_keeps_backups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evaluations.jsonl'
            store = Evaluations(path)
            store.add([(0, 0)], 'run/a', dict(simulations=32, solver_nodes=2048), dict(value=.4, moves=[], top=[]))
            store.add([(0, 0)], 'run/a', dict(simulations=512, solver_nodes=2048), dict(value=.6, moves=[], top=[]))
            store.add([(0, 0)], 'run/b', dict(simulations=32, solver_nodes=2048), dict(value=.9, moves=[], top=[]))
            store.add([(0, 0)], 'run/a', dict(simulations=32, solver_nodes=2048), dict(value=.3, moves=[], top=[]))
            self.assertEqual(len(path.read_text().splitlines()), 4)
            self.assertEqual(store.best([(0, 0)], 'run/a')['value'], .6)
            self.assertEqual(store.get([(0, 0)], 'run/a', dict(simulations=32, solver_nodes=2048))['value'], .3)
            self.assertIsNone(store.best([(0, 0), (1, 0)], 'run/a'))
            for _ in range(5):
                reloaded = Evaluations(path)
            self.assertEqual(reloaded.best([(0, 0)], 'run/a')['value'], .6)
            self.assertEqual(reloaded.get([(0, 0)], 'run/a', dict(simulations=32, solver_nodes=2048))['value'], .3)
            self.assertEqual(len(list(Path(directory).glob('evaluations.jsonl.*.bak'))), 3)
            self.assertEqual(len(path.read_text().splitlines()), 4)

    def test_a_torn_last_line_does_not_swallow_the_next_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evaluations.jsonl'
            Evaluations(path).add([], 'e', STANDARD, dict(value=.1, moves=[], top=[]))
            with open(path, 'a', encoding='utf-8') as out:
                out.write('[1]\n{"position": 5, "engine": "e", "simulations": 1, "solver_nodes": 1}\n')
                out.write(json.dumps(dict(position='0,0', engine='e', simulations=1, solver_nodes=1, value=.5,
                                          moves=[[0]], top=[])) + '\n')
                out.write('{"position": "0,0", "engine": "e", "simulations": 3, "solver_nodes": 1, "value": NaN, '
                          '"moves": [], "top": []}\n')
                out.write('{"position": "0,0", "eng')
            Evaluations(path).add([(0, 0)], 'e', STANDARD, dict(value=.2, moves=[], top=[]))
            reloaded = Evaluations(path)
            self.assertEqual((reloaded.best([], 'e')['value'], reloaded.best([(0, 0)], 'e')['value']), (.1, .2))

    def test_reuse_needs_every_budget_dimension(self):
        store = Evaluations()
        store.add([], 'e', dict(simulations=512, solver_nodes=0), dict(value=.1, moves=[], top=[]))
        self.assertIsNone(store.covering([], 'e', STANDARD))
        store.add([], 'e', dict(simulations=128, solver_nodes=131072), dict(value=.2, moves=[], top=[]))
        self.assertEqual(store.covering([], 'e', STANDARD)['value'], .2)
        self.assertEqual(store.covering([], 'e', dict(simulations=64, solver_nodes=0))['value'], .1)
        self.assertEqual(store.best([], 'e')['value'], .1)
        store.add([], 'e', dict(simulations=32, solver_nodes=131072), dict(value=1., moves=[], top=[], proof=dict(winner=0, turns=2)))
        self.assertEqual(store.best([], 'e')['value'], 1.)

    def test_evaluations_without_move_values_are_shown_last_and_never_reused(self):
        store = Evaluations()
        store.add([], 'e', dict(simulations=512, solver_nodes=0), dict(value=.1, moves=[], top=[[0, 0, 1.]]))
        self.assertIsNone(store.covering([], 'e', dict(simulations=32, solver_nodes=0)))
        store.add([], 'e', dict(simulations=0, solver_nodes=0), dict(value=.4, moves=[], top=[[0, 0, 1.]]))
        self.assertEqual(store.covering([], 'e', dict(simulations=0, solver_nodes=0))['value'], .4)
        store.add([], 'e', dict(simulations=32, solver_nodes=0), dict(value=.2, moves=[], top=[[0, 0, 1., .2]]))
        self.assertEqual(store.covering([], 'e', dict(simulations=32, solver_nodes=0))['value'], .2)
        self.assertEqual(store.best([], 'e')['value'], .2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evaluations.jsonl'
            path.write_text(json.dumps(dict(position='', engine='e', simulations=1, solver_nodes=0, value=.3, moves=[],
                                            top=[[0, 0, 1., 'x']])), encoding='utf-8')
            self.assertIsNone(Evaluations(path).best([], 'e'))

    def test_rows_mark_stones_the_search_proved(self):
        self.assertEqual([move_row([1, 2], .5, v)[3:] for v in (1., -1., .2)], [[1., 1], [0., -1], [.6, 0]])
        self.assertEqual(move_row([1, 2], .5), [1, 2, .5])

    def test_turn_evaluations_kept_trees_and_review_stay_apart(self):
        session = Session(entries(), FakeEngines(), Evaluations())
        self.addCleanup(session.close)
        session.configure_analysis('bubble:fake', preset='standard', auto=False)
        key = session.engine_key(session.analysis)
        later = dict(history=[(0, 0), (1, 0)], moves=[[1, 1]], value=.3, top=[[1, 1, 1., .3, 0]], proof=None, line=[],
                     threat=[])
        found = dict(moves=[[1, 0], [1, 1]], value=.6, top=[[1, 0, 1., .6, 0]], proof=None, line=[], threat=[], later=[later])
        session.save([(0, 0)], key, STANDARD, found, 'fake')
        self.assertEqual(session.lookup([(0, 0), (1, 0)])['value'], .3)
        self.assertIsNone(session.review_lookup([(0, 0), (1, 0)]))
        session.store.add([(0, 0), (1, 0), (1, 1)], key + ':kept', STANDARD, dict(value=.9, moves=[], top=[]))
        self.assertEqual(session.lookup([(0, 0), (1, 0), (1, 1)])['value'], .9)
        self.assertIsNone(session.review_lookup([(0, 0), (1, 0), (1, 1)]))
        self.assertEqual(session.review_lookup([(0, 0)])['value'], .6)

    def test_index_keeps_the_newest_entries(self):
        store = Evaluations(limit=2)
        for n in range(3):
            store.add([(0, 0)] if n == 0 else [(0, 0), (n, 0)], 'e', STANDARD, dict(value=n / 10, moves=[], top=[]))
        self.assertIsNone(store.best([(0, 0)], 'e'))
        self.assertEqual(store.best([(0, 0), (2, 0)], 'e')['value'], .2)


def evaluation(value, moves=(), proof=None, pv=()):
    return dict(value=value, moves=[list(m) for m in moves], proof=proof, pv=[list(p) for p in pv])


class Review(unittest.TestCase):
    history = [(0, 0), (1, 0), (1, 1), (-1, 0), (-2, 0)]

    def labels(self, evaluations, winner=-1, history=None):
        history = history or self.history
        table = {len(h): e for h, e in evaluations.items()}
        lookup = lambda prefix: table.get(len(prefix)) if tuple(map(tuple, prefix)) == tuple(history[:len(prefix)]) else None
        return review(history, lookup, winner)

    def test_bands_best_and_better_line(self):
        cases = [(.5, .52, 'good'), (.5, .43, 'inaccuracy'), (.5, .35, 'mistake'), (.5, .25, 'blunder')]
        for before, after, label in cases:
            turns = self.labels({(): evaluation(.5, [(0, 0)]), (0,): evaluation(before, [(2, 2), (3, 3)]),
                                 (0, 1, 2): evaluation(1 - after)})
            self.assertEqual(turns[1]['label'], label)
            self.assertAlmostEqual(turns[1]['before'] - turns[1]['after'], before - after)
        self.assertEqual(turns[0]['label'], 'best')
        self.assertEqual(turns[1]['better'], [[2, 2], [3, 3]])
        self.assertEqual(turns[1]['line'], [[2, 2, 1], [3, 3, 1]])
        turns = self.labels({(0,): evaluation(.5, [(1, 1), (1, 0)]), (0, 1, 2): evaluation(.9)})
        self.assertEqual(turns[1]['label'], 'best')
        self.assertIsNone(turns[0]['label'])
        self.assertIsNone(turns[2]['label'])

    def test_proof_labels(self):
        win = lambda side: dict(winner=side, turns=2)
        base = {(): evaluation(.5, [(0, 0)])}
        kept = self.labels(base | {(0,): evaluation(.5, [(1, 0), (1, 1)], win(1), [(1, 0, 1)]),
                                   (0, 1, 2): evaluation(.5, [], None)})
        self.assertEqual(kept[1]['label'], 'kept')
        cases = [(win(0), None, 'lost'), (win(1), win(0), 'missed'), (win(1), win(1), 'kept'),
                 (None, win(0), 'allowed'), (None, win(1), 'found')]
        for before, after, label in cases:
            turns = self.labels(base | {(0,): evaluation(.5, [(5, 5), (6, 6)], before, [(5, 5, 1)]),
                                        (0, 1, 2): evaluation(.5, [], after)})
            self.assertEqual(turns[1]['label'], label)
        self.assertEqual(turns[1]['line'], None)
        allowed = self.labels(base | {(0,): evaluation(.5, [(5, 5), (6, 6)], None, [(5, 5, 1)]),
                                      (0, 1, 2): evaluation(.5, [], win(0))})[1]
        self.assertEqual((allowed['better'], allowed['line']), ([[5, 5], [6, 6]], [[5, 5, 1]]))
        history = [(0, 0), (0, 5), (1, 5), (5, 0), (-5, 0), (2, 5), (3, 5), (0, -5), (0, -6), (4, 5), (5, 5)]
        game = Game(history)
        self.assertEqual(game.winner, 1)
        game.close()
        self.assertEqual(self.labels({}, 1, history)[-1]['label'], 'win')


class Jobs(unittest.TestCase):
    def setUp(self):
        self.engines = FakeEngines()
        self.session = Session(entries(), self.engines, Evaluations())
        self.session.configure_analysis('bubble:fake', auto=False)

    def history(self):
        return list(self.session.history)

    def test_engine_replies_in_the_background_and_saves_its_evaluation(self):
        self.session.play(0, 0)
        wait(lambda: len(self.history()) == 3)
        state = self.session.state()
        self.assertEqual(state['player'], 0)
        self.assertEqual(state['evaluations'][1]['value'], .5)
        self.assertEqual(self.engines.calls[0][:2], ('main/000002', STANDARD))
        with self.assertRaises(ValueError):
            self.session.configure_seat(1, 'bubble:fake', 'main/000009')

    def test_six_protocol_engines_play_and_end_on_cancel(self):
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / 'engine.py'
            script.write_text(SLOW_ENGINE)
            entry = dict(id='six:slow', name='slow', kind='six', presets=presets_of('six', None),
                         command=[sys.executable, str(script)], cwd=Path(folder), mirrored=True, libraries=[])
            engines = Engines('cpu')
            session = Session(entries() | {'six:slow': entry}, engines, Evaluations())
            try:
                session.configure_analysis('bubble:fake', auto=False)
                session.configure_seat(1, 'native:Native', preset='quick')
                session.configure_seat(0, 'six:slow')
                self.assertEqual(json.loads(json.dumps(session.state()))['engines'][-1], dict(
                    id='six:slow', name='slow', kind='six', presets=presets_of('six', None)))
                wait(lambda: len(session.history) == 1)
                session.pause(True)
                session.load([(0, 0), (1, 0), (2, 0)], False)
                thinking = lambda: [j['id'] for j in session.state()['jobs'] if j['status'] == 'running' and j['ply'] == 3]
                wait(thinking)
                started = time.time()
                session.cancel(thinking()[0])
                wait(lambda: not session.state()['jobs'])
                self.assertLess(time.time() - started, 5)
                self.assertEqual(session.history, [(0, 0), (1, 0), (2, 0)])
                session.pause(False)
                wait(lambda: any(j['status'] == 'running' and j['ply'] == 3 for j in session.state()['jobs']))
            finally:
                session.close()

    def test_a_silent_engine_can_be_cancelled_during_its_handshake(self):
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / 'engine.py'
            script.write_text('import time\ntime.sleep(60)\n')
            entry = dict(id='six:mute', name='mute', kind='six', presets=presets_of('six', None),
                         command=[sys.executable, str(script)], cwd=Path(folder), mirrored=False, libraries=[])
            engines = Engines('cpu')
            started = time.time()
            try:
                with self.assertRaises(Cancelled):
                    engines.turn(entry, dict(nodes=10), [], lambda: time.time() - started > .5)
                self.assertLess(time.time() - started, 5)
            finally:
                engines.close()

    def test_cancel_stops_a_thinking_engine_and_pauses(self):
        self.engines.hold = True
        self.session.play(0, 0)
        wait(lambda: any(j['status'] == 'running' for j in self.session.state()['jobs']))
        started = time.time()
        state = self.session.state()
        self.assertLess(time.time() - started, 1)
        self.session.cancel(state['jobs'][0]['id'])
        wait(lambda: not self.session.state()['jobs'])
        self.assertTrue(self.session.paused)
        self.assertEqual(self.history(), [(0, 0)])
        self.engines.hold = False
        self.session.pause(False)
        wait(lambda: len(self.history()) == 3)

    def test_bots_play_each_other_until_paused_and_moves_never_apply_to_a_changed_game(self):
        self.session.configure_seat(0, 'native:Native', preset='quick')
        wait(lambda: len(self.history()) >= 9)
        self.session.pause(True)
        length = len(self.history())
        time.sleep(.1)
        self.assertLessEqual(len(self.history()), length + 2)
        self.engines.hold = True
        self.session.load([(0, 0)], False)
        wait(lambda: any(j['status'] == 'running' for j in self.session.state()['jobs']))
        self.session.load([(0, 0), (1, 0), (2, 0)], True)
        self.engines.release.set()
        wait(lambda: not self.session.state()['jobs'])
        self.assertEqual(self.history(), [(0, 0), (1, 0), (2, 0)])

    def test_analysis_reuses_deeper_evaluations_and_review_fills_every_turn(self):
        self.session.configure_seat(1, 'human')
        for move in [(0, 0), (1, 0), (2, 0)]:
            self.session.play(*move)
        self.session.analyse(1, force=True)
        wait(lambda: not self.session.state()['jobs'])
        calls = len(self.engines.calls)
        self.assertIsNone(self.session.analyse(1))
        job = self.session.jobs[self.session.review_game()]
        self.assertEqual(self.session.review_game(), job.id)
        wait(lambda: not self.session.state()['jobs'])
        self.assertEqual((job.done, job.total), (3, 3))
        self.assertEqual(len(self.engines.calls), calls + 2)
        self.assertEqual([t['label'] for t in self.session.state()['review']], ['best', 'good'])
        self.session.configure_analysis('bubble:fake', preset='deep', auto=False)
        self.session.analyse(1)
        wait(lambda: not self.session.state()['jobs'])
        self.session.configure_analysis('bubble:fake', preset='quick', auto=False)
        self.assertIsNone(self.session.analyse(1))
        self.assertEqual(self.session.state()['evaluations'][1]['simulations'], PRESETS['bubble']['deep']['simulations'])

    def test_evaluations_follow_the_weights_not_the_name(self):
        self.session.configure_seat(1, 'human')
        self.session.analyse(0)
        wait(lambda: not self.session.state()['jobs'])
        self.assertIsNotNone(self.session.state()['evaluations'].get(0))
        self.session.configure_analysis('bubble:fake', 'main/000001', auto=False)
        self.assertIsNone(self.session.state()['evaluations'].get(0))
        self.session.analyse(0)
        wait(lambda: not self.session.state()['jobs'])
        self.session.configure_analysis('bubble:fake~2', auto=False)
        self.assertIsNotNone(self.session.state()['evaluations'].get(0))
        self.assertEqual(len(self.engines.calls), 2)

    def test_review_steps_aside_for_the_viewed_position(self):
        self.session.configure_seat(1, 'human')
        for move in [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]:
            self.session.play(*move)
        self.engines.hold = True
        patcher = unittest.mock.patch('play.REVIEW_CHUNK', 1)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.session.review_game()
        wait(lambda: len(self.engines.calls) == 1)
        self.session.analyse(3)
        self.engines.release.set()
        wait(lambda: not self.session.state()['jobs'])
        self.assertEqual([len(call[2]) for call in self.engines.calls], [0, 3, 1, 5])

    def test_changing_the_analysis_engine_cancels_its_old_work(self):
        self.session.configure_seat(1, 'human')
        self.session.play(0, 0)
        self.engines.hold = True
        self.session.analyse(1, force=True)
        wait(lambda: any(j['status'] == 'running' for j in self.session.state()['jobs']))
        self.session.configure_analysis('bubble:fake', auto=True)
        self.assertEqual(self.session.state()['jobs'][0]['status'], 'running')
        self.session.configure_analysis('bubble:fake', preset='deep', auto=False)
        wait(lambda: not self.session.state()['jobs'])

    def test_a_move_queued_behind_analysis_is_replaced_after_a_change(self):
        self.session.configure_seat(1, 'human')
        self.session.play(0, 0)
        self.engines.hold = True
        self.session.analyse(1, force=True)
        wait(lambda: any(j['status'] == 'running' for j in self.session.state()['jobs']))
        self.session.configure_seat(1, 'bubble:fake')
        self.session.configure_seat(1, 'bubble:fake', 'main/000001')
        self.engines.release.set()
        wait(lambda: len(self.history()) == 3)

    def test_rescans_cancel_work_for_a_vanished_analysis_model(self):
        self.session.configure_seat(1, 'human')
        self.engines.hold = True
        old = self.session.analyse(0, force=True)
        wait(lambda: any(j['status'] == 'running' for j in self.session.state()['jobs']))
        self.session.rescan_entries = lambda: {k: v for k, v in entries().items() if k != 'bubble:fake'}
        self.session.rescan()
        wait(lambda: self.session.jobs[old].status == 'cancelled')
        self.engines.hold = False

    def test_viewing_another_position_cancels_the_running_view_analysis(self):
        self.session.configure_seat(1, 'human')
        for move in [(0, 0), (1, 0)]:
            self.session.play(*move)
        self.engines.hold = True
        first = self.session.analyse(1, force=True)
        wait(lambda: self.session.jobs[first].status == 'running')
        self.session.analyse(2, force=True)
        wait(lambda: self.session.jobs[first].status == 'cancelled')
        self.engines.release.set()
        wait(lambda: not self.session.state()['jobs'])

    def test_analysis_of_positions_left_behind_by_a_retry_is_cancelled(self):
        self.session.configure_seat(1, 'human')
        for move in [(0, 0), (1, 0), (2, 0)]:
            self.session.play(*move)
        self.engines.hold = True
        old = self.session.analyse(3, force=True)
        review = self.session.review_game()
        wait(lambda: self.session.jobs[old].status == 'running')
        self.session.load([(0, 0), (1, 1), (2, 2)], True)
        wait(lambda: self.session.jobs[old].status == self.session.jobs[review].status == 'cancelled')
        self.engines.hold = False

    def test_solver_retries_skip_positions_from_another_game(self):
        self.session.configure_seat(1, 'human')
        self.session.configure_analysis('bubble:fake', auto=True)
        self.session.play(0, 0)
        wait(lambda: not self.session.state()['jobs'])
        calls = len(self.engines.calls)
        self.session.retry([(0, 0), (5, 5)])
        self.session.retry([(0, 0)])
        wait(lambda: not self.session.state()['jobs'])
        self.assertEqual(len(self.engines.calls), calls)

    def test_the_worker_survives_a_model_file_that_vanished(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'gone.pt').write_bytes(b'')
            gone = dict(id='bubble:gone', name='gone', kind='bubble', presets=PRESETS['bubble'], checkpoints=[''],
                        path=Path(directory) / 'gone.pt')
            session = Session(entries() | {'bubble:gone': gone}, self.engines, Evaluations())
            session.configure_analysis('bubble:gone', auto=True)
            session.configure_seat(1, 'bubble:gone')
            (Path(directory) / 'gone.pt').unlink()
            session.play(0, 0)
            wait(lambda: session.paused)
            self.assertIsNone(session.state()['evaluations'].get(1))
            session.configure_seat(1, 'native:Native', preset='quick')
            session.pause(False)
            wait(lambda: len(session.history) == 3)

    def test_changing_one_seat_leaves_the_other_engine_thinking(self):
        self.engines.hold = True
        self.session.play(0, 0)
        wait(lambda: any(j['status'] == 'running' for j in self.session.state()['jobs']))
        self.session.configure_seat(0, 'native:Native', preset='quick')
        self.engines.release.set()
        wait(lambda: len(self.history()) >= 3)

    def test_closing_cancels_work_and_stops_the_worker(self):
        closed = []
        self.engines.close = lambda: closed.append(True)
        self.engines.hold = True
        self.session.analyse(0, force=True)
        wait(lambda: any(j['status'] == 'running' for j in self.session.state()['jobs']))
        self.session.close()
        self.assertFalse(any(worker.is_alive() for worker in self.session.workers))
        self.assertEqual(closed, [True])

    def test_a_stone_placed_on_a_paused_game_gets_its_answer(self):
        self.session.pause(True)
        self.session.play(0, 0)
        wait(lambda: len(self.history()) == 3)
        self.assertFalse(self.session.paused)

    def test_six_seats_play_the_chosen_network(self):
        six = dict(id='six:Six · CPU', name='Six · CPU', label='CPU', kind='six', presets=PRESETS['six'],
                   checkpoints=['gen-2', 'gen-1'], networks={'gen-2': Path('gen-2.onnx'), 'gen-1': Path('gen-1.onnx')},
                   command=['sixengine', '--cpu'], cwd=Path('.'), mirrored=True, libraries=[])
        self.session.entries[six['id']] = six
        self.session.configure_seat(1, six['id'], preset='quick')
        self.assertEqual(self.session.seats[1]['checkpoint'], 'gen-2')
        self.session.configure_seat(1, six['id'], 'gen-1', 'lightning')
        self.session.play(0, 0)
        wait(lambda: len(self.history()) == 3)
        self.assertEqual(self.engines.turns[-1], (six['id'], 'gen-1', dict(nodes=1500)))
        with self.assertRaises(ValueError):
            self.session.configure_seat(1, six['id'], 'gen-3')

    def test_a_six_process_lives_across_presets_and_two_networks(self):
        started, closed = [], []

        class Fake:
            def __init__(self, command, **options):
                self.command, self.cancel, self.info = command, options['cancel'], {}
                started.append(command)

            def __call__(self, game, ms, nodes=None):
                return legal_turn([tuple(cell[:2]) for cell in game.cells])

            def close(self):
                closed.append(self.command)

        six = dict(kind='six', name='Six', command=['sixengine', '--cpu'], cwd=Path('.'), mirrored=True, libraries=[],
                   networks={f'gen-{n}': Path(f'gen-{n}.onnx') for n in range(3)})
        engines = Engines('cpu')
        with unittest.mock.patch('six_engine.SixEngine', Fake):
            for network, nodes in (('gen-2', 30000), ('gen-2', 6000), ('gen-1', 6000), ('gen-2', 6000), ('gen-0', 6000)):
                engines.turn(six, dict(nodes=nodes), [], checkpoint=network)
        self.assertEqual([c[2] for c in started], ['gen-2.onnx', 'gen-1.onnx', 'gen-0.onnx'])
        self.assertEqual([c[2] for c in closed], ['gen-1.onnx'])

    def test_analysis_deepens_through_every_preset_while_an_engine_plays(self):
        self.session.configure_analysis('bubble:fake', preset='quick', auto=True)
        budgets = [PRESETS['bubble'][name] for name in PRESETS['bubble']]
        wait(lambda: all(self.session.store.covering([], key, budget) for budget in budgets
                         for key in [self.session.engine_key(self.session.analysis) + ':kept']))
        deep = [budget for checkpoint, budget, history in self.engines.calls if not history]
        self.assertEqual(deep, budgets)
        self.assertIsNone(self.session.analyse(0))
        self.engines.hold = True
        self.session.load([(0, 0), (1, 0), (1, 1)], False)
        wait(lambda: any(j['status'] == 'running' and j['ply'] == 3 and j['kind'] == 'analyse' for j in self.session.state()['jobs']))
        self.session.configure_analysis('bubble:fake', preset='quick', auto=False)
        wait(lambda: not [j for j in self.session.state()['jobs'] if j['kind'] == 'analyse'])
        self.engines.release.set()
        self.session.load([(0, 0)], True)
        time.sleep(.2)
        self.assertFalse([j for j in self.session.state()['jobs'] if j['ply'] == 1 and j['kind'] == 'analyse'
                          and j['status'] == 'queued'])

    def test_a_preset_without_a_solver_verdict_is_not_deepened_again(self):
        def unsolved(entry, checkpoint, budget, history, watch, live=None, keep=False):
            found, spent, key = FakeEngines.evaluate(self.engines, entry, checkpoint, budget, history, watch, keep=keep)
            return found, spent | dict(solver_nodes=0), key
        self.engines.evaluate = unsolved
        self.session.configure_analysis('bubble:fake', preset='quick', auto=True)
        wait(lambda: not self.session.state()['jobs'])
        self.assertEqual(len([c for c in self.engines.calls if not c[2]]), len(PRESETS['bubble']))

    def test_finished_positions_are_not_analysed(self):
        final = [(0, 0), (0, 5), (1, 5), (5, 0), (-5, 0), (2, 5), (3, 5), (0, -5), (0, -6), (4, 5), (5, 5)]
        self.session.load(final, True)
        self.assertIsNone(self.session.analyse(len(final)))
        self.session.configure_analysis('bubble:fake', preset='deep', auto=True)
        time.sleep(.2)
        self.assertFalse([j for j in self.session.state()['jobs'] if j['ply'] == len(final)])

    def test_reviewing_an_empty_board_evaluates_it_once(self):
        self.session.configure_seat(1, 'human')
        self.session.review_game()
        wait(lambda: not self.session.state()['jobs'])
        self.assertEqual([len(call[2]) for call in self.engines.calls], [0])

    def test_review_uses_the_review_budget_whatever_the_slider_says(self):
        self.session.configure_seat(1, 'human')
        self.session.load([(0, 0), (1, 0), (2, 0)], True)
        self.session.configure_analysis('bubble:fake', preset='deep', auto=False)
        self.session.review_game()
        wait(lambda: not self.session.state()['jobs'])
        self.assertEqual({tuple(b.items()) for _, b, _ in self.engines.calls}, {tuple(STANDARD.items())})
        state = self.session.state()
        self.assertEqual((state['review_preset'], [t['label'] is not None for t in state['review']]), ('standard', [True, True]))

    def test_undo_returns_to_the_players_last_turn(self):
        self.session.play(0, 0)
        wait(lambda: len(self.history()) == 3)
        self.session.pause(True)
        self.session.play(0, 1)
        self.session.undo()
        self.assertEqual(self.history(), [(0, 0), *map(tuple, legal_turn([(0, 0)]))])
        self.session.undo()
        self.assertEqual(self.history(), [])

    def test_undo_skips_a_seat_the_page_plays(self):
        self.session.configure_seat(1, 'human')
        self.session.play(0, 0)
        self.session.play(1, 0)
        self.session.play(2, 0)
        self.session.undo([0])
        self.assertEqual(self.history(), [])
        with self.assertRaises(ValueError):
            self.session.undo([2])

    def test_failures_reach_the_page_and_rescans_drop_vanished_engines(self):
        def broken(*args, **options):
            raise RuntimeError('weights unreadable')
        self.engines.evaluate = broken
        self.session.analyse(0)
        wait(lambda: any(j['status'] == 'failed' for j in self.session.state()['jobs']))
        self.assertEqual(self.session.state()['jobs'][0]['error'], 'weights unreadable')
        self.session.rescan_entries = lambda: {k: v for k, v in entries().items() if k != 'bubble:fake'}
        self.session.rescan()
        state = self.session.state()
        self.assertEqual((state['seats'][1], state['analysis']['engine']), (dict(engine='human'), 'bubble:fake~2'))
        renamed = {k: v for k, v in entries().items() if k != 'bubble:fake'}
        renamed['bubble:fake'] = dict(renamed.pop('bubble:fake~2'), id='bubble:fake')
        self.session.rescan_entries = lambda: renamed
        self.session.rescan()
        self.assertEqual(self.session.state()['analysis']['engine'], 'bubble:fake')
        with tempfile.TemporaryDirectory() as directory:
            elsewhere = {k: v for k, v in renamed.items()}
            elsewhere['bubble:fake'] = dict(renamed['bubble:fake'], path=Path(directory))
            self.session.configure_seat(1, 'bubble:fake')
            self.session.rescan_entries = lambda: elsewhere
            self.session.rescan()
            self.assertEqual(self.session.state()['seats'][1], dict(engine='human'))

    def test_cancelled_native_searches_end_and_the_next_starts_at_once(self):
        engines = Engines('cpu')
        try:
            self.assertEqual(len(engines.turn(entries()['native:Native'], dict(ms=50), [(0, 0)])), 2)
            started = time.time()
            with self.assertRaises(Cancelled):
                engines.turn(entries()['native:Native'], dict(ms=20000), [(0, 0)], lambda: time.time() - started > .2)
            self.assertEqual(len(engines.turn(entries()['native:Native'], dict(ms=50), [(0, 0)])), 2)
            self.assertLess(time.time() - started, 10)
        finally:
            engines.close()

    def test_ending_a_search_child_ends_what_it_started_and_removes_its_temporary_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            beat = Path(folder) / 'beat'
            (Path(folder) / 'grandchild.py').write_text(
                'import pathlib, sys, time\n'
                'while True:\n'
                '    pathlib.Path(sys.argv[1]).write_text(str(time.time()))\n'
                '    time.sleep(.02)\n')
            (Path(folder) / 'child.py').write_text(
                'import subprocess, sys, tempfile, time\n'
                'subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])\n'
                'print(tempfile.gettempdir(), flush=True)\n'
                'time.sleep(60)\n')
            child = SearchChild([sys.executable, str(Path(folder) / 'child.py'), str(Path(folder) / 'grandchild.py'),
                                 str(beat)])
            temporary = Path(child.lines.get(timeout=10).strip())
            self.assertTrue(temporary.is_dir())
            wait(beat.exists)
            child.end()
            time.sleep(.1)
            last = beat.read_text()
            time.sleep(.3)
            self.assertEqual(beat.read_text(), last)
            self.assertFalse(temporary.exists())

    def test_strix_opens_at_the_origin_without_a_search(self):
        entry = dict(id='strix:Strix', name='Strix', kind='strix', presets=presets_of('strix', None),
                     model=Path('missing.safetensors'))
        engines = Engines('cpu')
        try:
            self.assertEqual(engines.turn(entry, dict(simulations=8), []), [[0, 0]])
            self.assertEqual(engines.children, {})
        finally:
            engines.close()

    def test_reaping_a_tree_process_ends_what_it_left_running(self):
        with tempfile.TemporaryDirectory() as folder:
            beat = Path(folder) / 'beat'
            (Path(folder) / 'grandchild.py').write_text(
                'import pathlib, sys, time\n'
                'while True:\n'
                '    pathlib.Path(sys.argv[1]).write_text(str(time.time()))\n'
                '    time.sleep(.02)\n')
            leader = TreeProcess([sys.executable, '-c', 'import subprocess, sys; subprocess.Popen(sys.argv[1:])',
                                  sys.executable, str(Path(folder) / 'grandchild.py'), str(beat)],
                                 stdout=subprocess.DEVNULL)
            wait(beat.exists)
            wait(lambda: leader.poll() is not None)
            leader.wait()
            time.sleep(.1)
            last = beat.read_text()
            time.sleep(.3)
            self.assertEqual(beat.read_text(), last)

    def test_solver_identity_follows_the_built_library(self):
        import tactical_proof
        with tempfile.TemporaryDirectory() as directory:
            engines = Engines('cpu', tactical_package=Path(directory))
            binary = tactical_proof.library(Path(directory))
            binary.parent.mkdir(parents=True)
            record = binary.with_name(binary.name + '.json')
            record.write_text(json.dumps(dict(binary_sha256='a' * 64)))
            self.assertEqual((engines.solver_build(), engines.solver()), ('none', (None, 'none')))
            binary.write_bytes(b'')
            record.write_text('{"binary_')
            self.assertEqual(engines.solver_build(), 'none')
            record.write_text(json.dumps(dict(binary_sha256='a' * 64)))
            os.utime(record, ns=(10 ** 18, 10 ** 18))
            with unittest.mock.patch.object(tactical_proof, 'IsolatedTactics') as isolated:
                first, build = engines.solver()
                self.assertEqual((build, engines.solver()[0]), ('aaaaaaaa', first))
                record.write_text(json.dumps(dict(binary_sha256='b' * 64)))
                os.utime(record, ns=(2 * 10 ** 18, 2 * 10 ** 18))
                second, build = engines.solver()
                self.assertEqual(build, 'bbbbbbbb')
                first.abort.assert_called_once()
                self.assertEqual(isolated.call_count, 2)

    def test_budgets(self):
        bubble = PRESETS['bubble']
        self.assertEqual(budget_of(bubble, 'custom', dict(simulations=0)), dict(simulations=0, solver_nodes=32768))
        for custom in (dict(simulations=10 ** 6), dict(ms=5), dict(simulations='8')):
            with self.assertRaises(ValueError):
                budget_of(bubble, 'custom', custom)
        with self.assertRaises(ValueError):
            budget_of(PRESETS['native'], 'heavy')
        self.assertEqual(presets_of('six', dict(quick=dict(args=['--visits', '8'])))['quick'],
                         dict(nodes=6000, args=['--visits', '8']))
        shrimp = presets_of('six', dict(quick=dict(nodes=1, args=['--visits', '32'])))
        self.assertEqual(budget_of(shrimp, 'custom', dict(nodes=9, args=['--visits', '1'])), dict(nodes=9))
        self.assertEqual(budget_of(shrimp, 'quick'), dict(nodes=1, args=['--visits', '32']))
        lightning = {kind: presets_of(kind, None)['lightning'] for kind in PRESETS}
        self.assertEqual(lightning, dict(bubble=dict(simulations=8, solver_nodes=2048), native=dict(ms=100),
                                         seal=dict(ms=50), six=dict(nodes=1500), strix=dict(simulations=2)))
        self.assertEqual([list(PRESETS[kind]) for kind in PRESETS], [['lightning', 'quick', 'standard', 'strong', 'deep', 'dangerous']] * 5)
        self.assertEqual(presets_of('six', dict(quick=dict(nodes=1)))['lightning'], dict(nodes=1500))
        self.assertEqual({kind: presets_of(kind, None)['dangerous'] for kind in PRESETS},
                         dict(bubble=dict(simulations=65536, solver_nodes=4_000_000), native=dict(ms=60000),
                              seal=dict(ms=30000), six=dict(nodes=2_000_000), strix=dict(simulations=4096)))
        self.assertEqual(budget_of(PRESETS['bubble'], 'custom', dict(simulations=65536, solver_nodes=4_000_000)),
                         dict(simulations=65536, solver_nodes=4_000_000))
        for spec in (dict(heavy=dict(nodes=1)), dict(quick=dict(nodes=0)), dict(quick=dict(args='--x')), [1],
                     dict(quick=dict(ms=1000))):
            with self.assertRaises(ValueError):
                presets_of('six', spec)
        with self.assertRaises(ValueError):
            budget_of(presets_of('strix', None), 'custom', dict(simulations=0), 'strix')
        with self.assertRaises(ValueError):
            presets_of('strix', dict(quick=dict(simulations=0)))
        with self.assertRaises(ValueError):
            presets_of('strix', dict(quick=dict(nodes=100)))


class Http(unittest.TestCase):
    def setUp(self):
        self.engines = FakeEngines()
        self.session = Session(entries(), self.engines, Evaluations())
        handler = type('H', (Handler,), dict(session=self.session))
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.root = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def post(self, path, body=None, headers=None):
        request = Request(self.root + path, json.dumps(body or {}).encode(),
                          {'Content-Type': 'application/json', **(headers or {})})
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    def get(self, path):
        with urlopen(self.root + path, timeout=5) as response:
            return response.read().decode()

    def test_engine_bundle_is_served_cross_origin_isolated(self):
        for path, kind in (('/', 'text/html'), ('/engine/search.mjs', 'text/javascript'), ('/coi-sw.js', 'text/javascript')):
            with urlopen(self.root + path, timeout=5) as response:
                self.assertTrue(response.headers['Content-Type'].startswith(kind))
                self.assertEqual(response.headers['Cross-Origin-Opener-Policy'], 'same-origin')
                self.assertEqual(response.headers['Cross-Origin-Embedder-Policy'], 'credentialless')
        for path in ('/engine/../../python/play.py', '/engine/%2e%2e/index.html', '/engine/missing.mjs', '/index.html'):
            with self.assertRaises(HTTPError) as caught:
                urlopen(self.root + path, timeout=5)
            self.assertEqual(caught.exception.code, 404)
            caught.exception.close()

    def test_page_stays_responsive_while_an_engine_thinks(self):
        self.engines.hold = True
        self.post('/play', dict(q=0, r=0))
        wait(lambda: any(j['status'] == 'running' for j in self.session.state()['jobs']))
        started = time.time()
        state = json.loads(self.get('/state'))
        quiet = json.loads(self.get(f"/state?since={state['revision']}"))
        self.assertLess(time.time() - started, 1)
        self.assertEqual(set(quiet), {'instance', 'revision', 'jobs'})
        self.engines.hold = False
        self.post('/cancel', dict(id=state['jobs'][0]['id']))
        wait(lambda: not self.session.state()['jobs'])

    def test_match_api_starts_reports_results_and_preserves_live_state(self):
        with tempfile.TemporaryDirectory() as directory:
            try:
                state = self.post('/match', dict(players=['Native', 'Native'], games=2,
                                                output=str(Path(directory) / 'match'), max_placements=3))
                self.assertEqual(state['match']['games'], 2)
                wait(lambda: not self.session.match_worker.is_alive())
                live = json.loads(self.get('/state'))
                result = json.loads(self.get('/match/results'))
                self.assertEqual((live['match']['completed'], result['capped']), (2, 2))
                self.assertEqual(len(live['history']), 3)
                catalogue = json.loads(self.get('/matches'))['matches']
                ident = catalogue[0]['id']
                self.assertEqual(catalogue[0]['completed'], 2)
                saved = json.loads(self.get(f'/matches/game?batch={ident}&game=1'))
                self.assertEqual(import_history(self.get(f'/matches/game?batch={ident}&game=1&format=htttx')), saved['history'])
                with unittest.mock.patch('play.Engines', return_value=FakeEngines()):
                    self.post('/matches/open', dict(batch=ident, game=1))
                self.assertEqual(json.loads(self.get('/study/state'))['history'], saved['history'])
                self.post('/study/retry', dict(ply=1))
                self.assertEqual(json.loads(self.get('/study/state'))['history'], [[0, 0]])
                self.assertEqual(json.loads(self.get('/state'))['history'], live['history'])
                self.assertEqual(self.post('/match/stop')['match']['active'], False)
                self.post('/new')
                self.assertIsNone(json.loads(self.get('/state'))['match'])
            finally:
                self.session.close()

    def test_import_loads_a_hexo_site_link(self):
        with unittest.mock.patch('play.fetch_json', return_value=SiteImport.sandbox) as fetch:
            state = self.post('/import', dict(text='https://hexo.did.science/sandbox/2mdyn02'))
        fetch.assert_called_once_with('https://hexo.did.science/api/sandbox-positions/2mdyn02')
        self.assertEqual((len(state['history']), state['paused']), (7, True))
        state = self.post('/import', dict(text=Formats.GAME))
        self.assertEqual((len(state['history']), state['winner']), (91, 1))
        shown = json.loads(self.get('/export?format=rectilinear&ply=3'))
        self.assertEqual(formats.rectilinear_loads(shown['text']), state['history'][:3])
        self.assertEqual(json.loads(self.get('/export?format=tyto'))['text'], formats.tyto_dumps(state['history']))

    def test_import_export_retry_and_origin(self):
        self.post('/seat', dict(side=1, engine='human'))
        state = self.post('/import', dict(text='version[1];\n1. [1,0][2,0];'))
        self.assertEqual(state['history'], [[0, 0], [1, 0], [2, 0]])
        self.assertTrue(state['paused'])
        self.assertEqual(self.get('/htttx'), 'version[1];\n1. [1,0][2,0];')
        replay = json.loads(self.get('/replay'))
        self.assertEqual(import_history(json.dumps(replay)), state['history'])
        self.assertEqual(self.post('/retry', dict(ply=2))['history'], [[0, 0], [1, 0]])
        self.assertEqual(self.get('/htttx'), 'version[1];\n1. [1,0];')
        self.post('/new')
        with self.assertRaises(HTTPError) as caught:
            self.get('/htttx')
        self.assertEqual(caught.exception.code, 409)
        caught.exception.close()
        for path, body in (('/play', dict(q=40, r=0)), ('/import', dict(text='1. [9,9]')), ('/retry', dict(ply=9)),
                           ('/seat', [1]), ('/seat', dict(side=1, engine='native:Native', preset='custom', custom=[]))):
            with self.assertRaises(HTTPError) as caught:
                self.post(path, body)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()
        final = [(0, 0), (0, 5), (1, 5), (5, 0), (-5, 0), (2, 5), (3, 5), (0, -5), (0, -6), (4, 5), (-2, 5),
                 (0, -7), (0, -8), (5, 5)]
        self.post('/new')
        self.session.load(final, True)
        text = self.get('/htttx')
        self.assertTrue(text.endswith('7. [5,5];'))
        self.post('/new')
        state = self.post('/import', dict(text=text))
        self.assertEqual((state['history'], state['winner']), ([list(p) for p in final], 1))
        port = self.server.server_port
        rebound = {'Host': f'evil.example:{port}', 'Origin': f'http://evil.example:{port}'}
        for headers in ({'Origin': 'http://example.com'}, rebound):
            with self.assertRaises(HTTPError) as caught:
                self.post('/new', headers=headers)
            self.assertEqual(caught.exception.code, 403)
            caught.exception.close()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.root + '/state', headers={'Host': f'evil.example:{port}'}), timeout=5)
        self.assertEqual(caught.exception.code, 403)
        caught.exception.close()
        local = {'Origin': f'http://localhost:{port}', 'Host': f'localhost:{port}'}
        self.assertEqual(self.post('/new', headers=local)['history'], [])


class SiteImport(unittest.TestCase):
    """Recorded API answers: hexo.did.science (hexo.mineking.dev serves the same API under /proxy/api) and
    hexo.tyto.cc."""
    game = json.loads((SITE / 'finished-game.json').read_text(encoding='utf-8'))
    sandbox = json.loads((SITE / 'sandbox-position.json').read_text(encoding='utf-8'))
    shared = json.loads((SITE / 'sandbox-z108lz7.json').read_text(encoding='utf-8'))
    tyto = json.loads((SITE / 'tyto-game.json').read_text(encoding='utf-8'))

    def fetched(self, url, answer):
        asked = []
        history = linked_history(url, lambda api, body=None: asked.append((api, body)) or answer)
        return history, [api for api, _ in asked]

    def test_games_and_sandbox_positions_become_htttx_histories(self):
        history, asked = self.fetched('https://hexo.did.science/games/8211f449-5020-4a5a-9a93-581c5f720aac', self.game)
        self.assertEqual(asked, ['https://hexo.did.science/api/finished-games/8211f449-5020-4a5a-9a93-581c5f720aac'])
        self.assertEqual((len(history), history[:3]), (39, [[0, 0], [1, 2], [2, 1]]))
        game = Game(history)
        self.assertEqual(game.winner, 1)
        game.close()
        _, asked = self.fetched('https://hexo.mineking.dev/games/8211f449-5020-4a5a-9a93-581c5f720aac/', self.game)
        self.assertEqual(asked, ['https://hexo.mineking.dev/proxy/api/finished-games/8211f449-5020-4a5a-9a93-581c5f720aac'])
        history, asked = self.fetched('https://hexo.mineking.dev/sandbox/2MDYN02', self.sandbox)
        self.assertEqual(asked, ['https://hexo.mineking.dev/proxy/api/sandbox-positions/2mdyn02'])
        self.assertEqual(history, [[0, 0], [1, -1], [0, 1], [1, 0], [-1, 0], [2, 0], [-4, 0]])
        history, asked = self.fetched('https://hexo.did.science/sandbox/z108lz7', self.shared)
        self.assertEqual((len(history), history[:3]), (18, [[0, 0], [3, 0], [1, 2]]))

    def test_tyto_analysis_and_game_links(self):
        self.assertEqual(linked_history('https://hexo.tyto.cc/analysis#c=BAEIAw', self.offline), [[0, 0], [1, 1], [2, 2]])
        asked = []
        history = linked_history('https://hexo.tyto.cc/#g=ebb77124-db4c-4c42-979c-3e4d49244cec',
                                 lambda api, body=None: asked.append((api, body)) or self.tyto)
        self.assertEqual(asked, [('https://hexo.tyto.cc/game_htttx', dict(game_id='ebb77124-db4c-4c42-979c-3e4d49244cec'))])
        self.assertEqual(history, import_history(self.tyto['htttx']))
        with self.assertRaises(ValueError):
            linked_history('https://hexo.tyto.cc/analysis', self.offline)

    def test_other_text_is_left_alone_and_bad_links_are_refused(self):
        for text in ('version[1];\n1. [1,0][2,0];', 'https://example.com/games/1', '[[0, 0]]'):
            self.assertIsNone(linked_history(text, self.offline))
        with self.assertRaises(ValueError):
            linked_history('https://hexo.did.science/leaderboard', self.offline)
        moved = json.loads(json.dumps(self.sandbox))
        for cell in moved['gamePosition']['cells']:
            cell['x'] += 3
        self.assertEqual(self.fetched('https://hexo.did.science/sandbox/2mdyn02', moved)[0][:2], [[0, 0], [1, -1]])
        swapped = json.loads(json.dumps(self.sandbox))
        swapped['gamePosition']['cells'][1]['player'] = 'player-1'
        with self.assertRaises(ValueError):
            self.fetched('https://hexo.did.science/sandbox/2mdyn02', swapped)

    def offline(self, url, body=None):
        raise AssertionError(f'fetched {url}')


class Formats(unittest.TestCase):
    """Rectilinear notation (MineKing9534/HeXO) and Tyto analysis links (SootyOwl/hexo-strix)."""
    GAME = ('o/xo, q @(1, 1) x A0 C2.1 o A1 B2.0 x C1.2 B4.0 o B1.0 C1.0 x D1.0 C4.1 o E4.3 B3.0 x C4.0 D3.3 o C3.2 '
            'E3.3 x C3.1 F4.1 o D4.1 E3.2 x E3.0 F4.2 o D4.0 D4.2 x D4.3 F4.3 o F4.4 D5.1 x D5.0 F3.1 o G3.2 F3.2 x '
            'H3.2 G3.1 o E3.1 G3.4 x F3.4 F3.0 o G3.3 H3.4 x D2.3 I3.5 o F2.5 E5.1 x F5.1 G2.5 o H3.0 K3.4 x L3.4 '
            'G2.4 o G2.6 K3.0 x L3.0 M2.12 o L3.3 E5.0 x F4.5 N2.12 o O2.12 K2.10 x I2.8 M3.2 o M2.10 N2.10 x I3.3 '
            'O2.10 o J3.4 K2.7 x I3.2 I3.4 o I3.0 I3.6 x K2.8 M3.3 o M3.1 M2.9 x L3.2 J3.1 o K3.2 K3.1 x K3.3 N2.11 '
            'o L2.8 D3.1 x O2.11 I2.5 o F3.3 H3.5')

    def test_a_rectilinear_game_reads_and_writes_back(self):
        history = formats.rectilinear_loads(self.GAME)
        game = Game(history)
        self.assertEqual((len(history), game.winner, history[:3]), (91, 1, [[0, 0], [-1, 1], [1, 0]]))
        game.close()
        text, spans = formats.rectilinear_dumps(history)
        self.assertTrue(text.startswith('x, b @(0, 0) o A0 A2 x B2.1 B5.1'))
        self.assertEqual(formats.rectilinear_loads(text), history)
        self.assertEqual([text[a:b] for a, b in spans[:3]], ['x', 'A0', 'A2'])
        self.assertEqual(formats.rectilinear_dumps([[0, 0]]), ('x', [(0, 0 + 1)]))

    def test_rings_follow_the_reference_parser(self):
        _, turns = formats.bke_turns('b@(1,0): o A0 A1 x B3.1 B3.2', implicit=False)
        self.assertEqual([(player, set(cells)) for player, cells in turns],
                         [('o', {(1, -1), (2, -1)}), ('x', {(-1, 2), (0, 2)})])
        for baseline in range(6):
            for ring in range(1, 30):
                for offset in range(6 * ring):
                    cell = formats.ring_cell((3, -2), baseline, True, ring, offset)
                    self.assertEqual(formats.ring_offset(cell, (3, -2), baseline), (ring, *divmod(offset, ring)))
        history = formats.rectilinear_loads('x A0 H2.2 o A1 G2.2')
        self.assertEqual((len(history), history[0]), (5, [0, 0]))
        self.assertEqual(formats.rectilinear_loads('x A0 H2.2 o A1 G2.2'),
                         formats.rectilinear_loads('o, d @(0, 0) x A0 H2.2 o A1 G2.2'))
        self.assertEqual(formats.rectilinear_loads('c-x'), [[0, 0]])
        self.assertEqual(formats.rectilinear_loads('oxo\r\nxx'), formats.rectilinear_loads('oxo/xx'))
        for text in ('x7x7o7o7x', 'oxo/xx'):
            self.assertEqual(len(formats.rectilinear_loads(text)), len(formats.drawing(text)))
        with self.assertRaisesRegex(ValueError, 'cannot be played'):
            formats.rectilinear_loads('x9oo')
        self.assertEqual(len(formats.rectilinear_loads('x, o A0 A1 x B1')), 4)
        for bad in ('xx', 'o/xo, q @(1, 1) o A0 B1', 'x(!', 'xz', 'x, o A0 x A1 A2', 'x, o A0 A1 A2'):
            with self.assertRaises(ValueError):
                formats.rectilinear_loads(bad)

    def test_tyto_links_both_ways(self):
        for history, code in (([(0, 0), (1, 1)], 'BAE'), ([(0, 0), (1, 1), (2, 2)], 'BAEIAw'),
                              ([(0, 0), (1, 0), (-1, 2), (3, -1), (0, -2)], 'AgACAwQCAwQ')):
            self.assertEqual(formats.tyto_dumps(history), formats.TYTO + code)
            self.assertEqual(formats.tyto_loads(code), [list(p) for p in history])
        far = formats.tyto_dumps([(0, 0), (1, 0), (2, 0), (3, -1), (70, -1)])
        with self.assertRaises(ValueError):
            formats.tyto_loads(far[len(formats.TYTO):])
        for broken in ('BA', '!!!', 'BAE=', 'BAEIA'):
            with self.assertRaises(ValueError):
                formats.tyto_loads(broken)
        with self.assertRaises(ValueError):
            formats.tyto_dumps([])

    def test_pasted_text_of_any_kind_is_read(self):
        self.assertEqual(read_game('version[1];\n1. [1,0][2,0];', self.offline), [[0, 0], [1, 0], [2, 0]])
        self.assertEqual(len(read_game(self.GAME, self.offline)), 91)
        self.assertEqual(read_game('https://hexo.tyto.cc/analysis#c=BAE', self.offline), [[0, 0], [1, 1]])
        with self.assertRaisesRegex(ValueError, 'Illegal'):
            read_game('version[1];\n1. [1,0][1,0];', self.offline)

    def test_exports_mark_each_stone(self):
        history = [(0, 0), (1, 0), (-1, 2), (3, -1), (0, -2)]
        htttx = export(history, 'htttx')
        self.assertEqual([[htttx['text'][a:b], q, r] for a, b, q, r in htttx['spans']],
                         [['[1,0]', 1, 0], ['[-1,2]', -1, 2], ['[3,-1]', 3, -1], ['[0,-2]', 0, -2]])
        rectilinear = export(history, 'rectilinear')
        self.assertEqual(formats.rectilinear_loads(rectilinear['text']), [list(p) for p in history])
        self.assertEqual([s[2:] for s in rectilinear['spans']], [list(p) for p in history])
        self.assertEqual(export(history, 'tyto'), dict(text=formats.TYTO + 'AgACAwQCAwQ', spans=[]))

    def offline(self, url, body=None):
        raise AssertionError(f'fetched {url}')


class Matches(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.engines = FakeEngines()
        registry = entries()
        registry['native:Other'] = registry['native:Native'] | dict(id='native:Other', name='Other')
        self.session = Session(registry, self.engines, Evaluations())
        self.session.configure_analysis('bubble:fake', auto=False)
        self.addCleanup(self.session.close)
        self.output = Path(self.directory.name) / 'match'

    def test_real_winners_are_counted_for_the_right_bot_after_swapping(self):
        opening = [(0, 0), (0, 5), (1, 5), (5, 0), (-5, 0), (2, 5), (3, 5), (0, -5), (0, -6),
                   (4, 5), (-2, 5), (0, -7), (0, -8)]
        self.engines.turn = lambda *args: [[5, 5]]
        self.session.start_match(['Native', 'Other'], output=self.output, openings=[opening])
        wait(lambda: not self.session.match_worker.is_alive())
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual((summary['completed'], summary['wins'], summary['capped']), (2, [1, 1], 0))
        first = json.loads((self.output / 'game-0001.json').read_text())
        second = json.loads((self.output / 'game-0002.json').read_text())
        self.assertEqual(first['history'], second['history'])
        self.assertEqual(first['players'], second['players'][::-1])
        self.assertEqual(import_history((self.output / 'game-0002.htttx').read_text()), second['history'])
        self.assertTrue(self.session.paused)
        stale = summary | dict(completed=1, wins=[0, 1], results=summary['results'][:1])
        (self.output / 'summary.json').write_text(json.dumps(stale))
        recovered = self.session.match_catalogue()[0]
        self.assertEqual((recovered['completed'], recovered['wins']), (2, [1, 1]))
        self.assertEqual([r['game'] for r in recovered['results']], [1, 2])

    def test_pause_holds_the_next_game_and_settings_cannot_change_mid_match(self):
        self.session.start_match(['Native', 'Other'], output=self.output, max_placements=3)
        wait(lambda: self.session.match['completed'] == 1)
        self.session.pause(True)
        for change in (lambda: self.session.load([], True), lambda: self.session.undo(),
                       lambda: self.session.configure_seat(0, 'human'), lambda: self.session.rescan()):
            with self.assertRaises(ValueError):
                change()
        time.sleep(2.1)
        self.assertEqual(self.session.match['completed'], 1)
        self.session.pause(False)
        wait(lambda: not self.session.match_worker.is_alive())
        self.assertEqual(self.session.match['capped'], 2)
        self.assertEqual(self.session.match['wins'], [0, 0])

    def test_engine_failure_pauses_without_inventing_a_result_and_stop_saves_position(self):
        self.engines.turn = unittest.mock.Mock(side_effect=ValueError('engine failed'))
        self.session.start_match(['Native', 'Other'], output=self.output)
        wait(lambda: self.session.match['error'] is not None)
        self.assertEqual((self.session.paused, self.session.match['completed']), (True, 0))
        self.session.stop_match()
        wait(lambda: not self.session.match_worker.is_alive())
        self.assertEqual(json.loads((self.output / 'current.json').read_text())['history'], [[0, 0]])
        self.assertEqual(json.loads((self.output / 'summary.json').read_text())['error'], 'engine failed')

    def test_book_counts_cutoff_repeatability_and_read_only_selection(self):
        path = Path(self.directory.name) / 'book.json'
        nodes = [dict(key=str(i), status='opening', moves=[[0, 0], [i, 0], [i, 1]], off_policy=i == 4,
                      champion_probability=.1*i, scored_by='main/123') for i in range(1, 5)]
        nodes += [nodes[0] | dict(key='retired', status='retired'),
                  nodes[0] | dict(key='tactical', status='tactical')]
        raw = json.dumps(dict(schema='hexo-opening-book-v2', nodes=nodes, refreshed_by='main/123'))
        path.write_text(raw)
        narrow = book_openings(path, 'narrow', 2, 17)
        self.assertEqual({n['key'] for n in narrow['nodes']}, {'2', '3'})
        self.assertEqual((narrow['unique_openings'], narrow['policy_cutoff']), (2, .2))
        self.assertEqual(narrow, book_openings(path, 'narrow', 2, 17))
        self.assertEqual(len(book_openings(path, 'wide')['nodes']), 3)
        self.assertEqual(len(book_openings(path, 'all')['nodes']), 4)
        self.assertEqual(path.read_text(), raw)
        with self.assertRaisesRegex(ValueError, '3 unique openings'):
            book_openings(path, 'wide', 4)
        self.session.start_match(['Native', 'Other'], output=self.output, book=path,
                                 opening_range='narrow', unique_openings=2, max_placements=5)
        wait(lambda: not self.session.match_worker.is_alive())
        self.assertEqual(self.session.match['games'], 4)
        games = [json.loads(p.read_text()) for p in sorted(self.output.glob('game-*.json'))]
        keys = [g['opening']['key'] for g in games]
        self.assertEqual(keys, [keys[0], keys[0], keys[2], keys[2]])
        self.assertNotEqual(keys[0], keys[2])

    def test_bad_match_specifications_do_not_change_the_board_or_start_jobs(self):
        for players, kwargs in [(['fake', 'Native'], {}), (['Native', 'Other'], dict(games=0)),
                                (['Native', 'Other'], dict(unique_openings=2, games=2)),
                                (['Native', 'Other'], dict(max_placements=0))]:
            with self.assertRaises(ValueError):
                self.session.start_match(players, output=self.output, **kwargs)
        self.assertIsNone(self.session.match)
        self.assertEqual(self.session.history, [])
        self.assertFalse(self.output.exists())

    def test_every_explicit_opening_gets_a_pair_by_default(self):
        openings = [[[0, 0]], [[0, 0], [1, 0], [2, 0]]]
        self.session.start_match(['Native', 'Other'], output=self.output, openings=openings, max_placements=5)
        wait(lambda: not self.session.match_worker.is_alive())
        self.assertEqual(self.session.match['completed'], 4)
        games = [json.loads(p.read_text()) for p in sorted(self.output.glob('game-*.json'))]
        for game, opening in zip(games, [openings[0], openings[0], openings[1], openings[1]]):
            self.assertEqual(game['history'][:len(opening)], opening)

    def test_six_identity_hashes_files_in_its_working_directory(self):
        folder = Path(self.directory.name)
        driver, model = folder / 'driver.py', folder / 'model.onnx'
        driver.write_text('print("test protocol driver")')
        model.write_bytes(b'network weights')
        self.session.entries['six:Test'] = dict(id='six:Test', name='Test', kind='six', cwd=str(folder),
            command=[sys.executable, driver.name, model.name], presets=PRESETS['six'])
        seat = self.session.match_seat('six:Test', 'standard')
        for path in (driver, model):
            self.assertEqual(seat['source']['files'][str(path.resolve())],
                             file_digest(file_identity(path)))

    def test_book_openings_start_games_and_cover_lines_against_a_person(self):
        path = Path(self.directory.name) / 'book.json'
        nodes = [dict(key=f'0,0|{i},0 {i},1', status='opening', moves=[[0, 0], [i, 0], [i, 1]], off_policy=i == 3)
                 for i in (1, 2, 3)]
        path.write_text(json.dumps(dict(schema='hexo-opening-book-v2', nodes=nodes)))
        self.session.book = None
        with self.assertRaises(ValueError):
            self.session.use_book(True)
        self.session.book = path
        self.session.pause(True)
        self.session.configure_seat(0, 'native:Native')
        self.session.configure_seat(1, 'native:Other')
        self.session.use_book(True, 'wide')
        opening = self.session.state()['book']['opening']
        i = int(opening['key'][4])
        self.assertEqual((opening['mode'], sorted(abs(q) + abs(r) + abs(q + r) for q, r in self.session.history[:3])),
                         ('wide', [0, 2 * i, 2 * i + 2]))
        self.session.configure_seat(0, 'human')
        keys = []
        for _ in range(3):
            self.session.new_game()
            keys.append(self.session.state()['book']['opening']['key'])
        self.assertEqual(set(keys[:2]), {'0,0|1,0 1,1', '0,0|2,0 2,1'})
        self.assertEqual(sum(self.session.coverage.counts.values()), 3)
        self.session.use_book(False)
        self.session.new_game()
        self.assertEqual((self.session.history, self.session.state()['book']['opening']), ([], None))

    def test_openings_are_picked_by_unplayed_branch(self):
        nodes = [dict(key=k) for k in ('0,0|a', '0,0|b x', '0,0|b y')]
        counts = {'0,0|a': 0, '0,0|b x': 1, '0,0|b y': 0}
        picks = {pick_opening(nodes, counts.get, random.Random(seed))['key'] for seed in range(40)}
        self.assertEqual(picks, {'0,0|a', '0,0|b y'})
        played = dict.fromkeys(counts, 0)
        for seed in range(3):
            node = pick_opening(nodes, played.get, random.Random(seed))
            played[node['key']] += 1
        self.assertEqual(set(played.values()), {1})
        played['0,0|a'] = 2
        self.assertNotEqual(pick_opening(nodes, played.get, random.Random(5))['key'], '0,0|a')

    def test_saved_tournaments_report_an_elo_from_complete_pairs(self):
        self.assertIsNone(pair_elo([dict(game=1, winner=0)]))
        elo = pair_elo([dict(game=1, winner=0), dict(game=2, winner=0), dict(game=3, winner=None), dict(game=4, winner=1)])
        self.assertEqual(elo['pairs'], 2)
        self.assertGreater(elo['a_minus_b'], 0)
        self.assertLess(elo['interval'][0], elo['a_minus_b'])

    def test_six_networks_are_checkpoints_of_one_entry(self):
        folder = Path(self.directory.name)
        for name in ('sixengine.exe', 'gen-0001.onnx', 'gen-0002.onnx'):
            (folder / name).write_bytes(name.encode())
        self.session.entries['six:Six · CPU'] = dict(
            id='six:Six · CPU', name='Six · CPU', label='CPU', kind='six', presets=PRESETS['six'],
            checkpoints=['gen-0002', 'gen-0001'], backend='CPU', cwd=folder, mirrored=True, libraries=[],
            networks={n: folder / f'{n}.onnx' for n in ('gen-0002', 'gen-0001')},
            command=[str(folder / 'sixengine.exe'), '--cpu'])
        seat = self.session.match_seat('six@gen-0001', 'quick')
        self.assertEqual((seat['checkpoint'], seat['name'], seat['source']['device']), ('gen-0001', 'Six · CPU/gen-0001', 'CPU'))
        self.assertEqual(seat['source']['command'], [str(folder / 'sixengine.exe'), '--net', str(folder / 'gen-0001.onnx'), '--cpu'])
        self.assertIn(str((folder / 'gen-0001.onnx').resolve()), seat['source']['files'])
        self.assertNotIn(str((folder / 'gen-0002.onnx').resolve()), seat['source']['files'])
        self.assertEqual(self.session.match_seat('Six', 'quick')['checkpoint'], 'gen-0002')
        with self.assertRaises(ValueError):
            self.session.match_seat('Six@gen-0009', 'quick')

    def test_resume_keeps_completed_games_and_pair_accounting(self):
        self.session.start_match(['Native', 'Other'], output=self.output, max_placements=3)
        wait(lambda: self.session.match['completed'] == 1)
        self.session.stop_match()
        wait(lambda: not self.session.match_worker.is_alive())
        first = (self.output / 'game-0001.json').read_bytes()
        (self.output / 'game-0001.htttx').write_text('interrupted notation write')
        self.session.resume_match(self.output)
        wait(lambda: not self.session.match_worker.is_alive())
        self.assertEqual((self.output / 'game-0001.json').read_bytes(), first)
        self.assertEqual(import_history((self.output / 'game-0001.htttx').read_text()), json.loads(first)['history'])
        self.assertEqual(self.session.match['completed'], 2)
        self.assertEqual(self.session.match['pentanomial'], [0, 0, 1, 0, 0])
        self.assertEqual(len(list(self.output.glob('game-*.json'))), 2)

    def test_resume_rejects_a_changed_tactical_solver(self):
        self.session.start_match(['bubble:2@quick', 'Other'], output=self.output, max_placements=3)
        wait(lambda: self.session.match['completed'] == 1)
        self.session.stop_match()
        wait(lambda: not self.session.match_worker.is_alive())
        self.engines.solver_build = lambda: 'new-build'
        with self.assertRaisesRegex(ValueError, 'Tactical solver build changed'):
            self.session.resume_match(self.output)
        self.assertFalse(self.session.match['active'])

    def test_saved_games_and_analysis_survive_restart_without_changing_live_play(self):
        self.session.archive = Path(self.directory.name) / 'archive'
        self.session.study_store = Path(self.directory.name) / 'analysis.jsonl'
        self.session.start_match(['Native', 'Other'], output=self.output, max_placements=3)
        wait(lambda: self.session.match['completed'] == 1)
        self.session.pause(True)
        ident = self.session.match_catalogue()[0]['id']
        saved = (self.output / 'game-0001.json').read_bytes()
        with unittest.mock.patch('play.Engines', return_value=FakeEngines()):
            study = self.session.open_saved_game(ident, 1)
        study.configure_analysis('bubble:fake', preset='custom', custom=dict(simulations=1, solver_nodes=0), auto=False)
        study.analyse(1)
        wait(lambda: study.lookup([(0, 0)]) is not None)
        self.assertTrue(self.session.match['active'])
        self.session.pause(False)
        wait(lambda: not self.session.match_worker.is_alive())
        self.assertEqual(self.session.match['completed'], 2)
        self.session.close()
        reopened = Session(entries(), FakeEngines(), Evaluations(), archive=self.session.archive,
                           study_store=self.session.study_store)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.match_catalogue()[0]['completed'], 2)
        with unittest.mock.patch('play.Engines', return_value=FakeEngines()):
            study = reopened.open_saved_game(ident, 1)
        self.assertIsNotNone(study.lookup([(0, 0)]))
        self.assertEqual((self.output / 'game-0001.json').read_bytes(), saved)

    def test_timed_bubble_uses_the_selected_tactical_package(self):
        package = Path(self.directory.name) / 'solver'
        self.engines.tactical_package = package
        seat = self.session.match_seat('bubble:2@quick', 'standard')
        config = self.session.timed_config(seat)
        self.assertEqual(config['tactical_package'], str(package))
        from timed_engine import _worker
        with unittest.mock.patch('dense_player.DensePlayer') as player:
            player.return_value.prover = None
            connection = unittest.mock.Mock()
            connection.recv.return_value = None
            _worker(connection, threading.Event(), config)
            self.assertEqual(player.call_args.kwargs['tactical_package'], package)

    def test_seat_specs_resolve_presets_checkpoint_steps_and_custom_budgets(self):
        seat = self.session.match_seat('bubble:2@quick', 'standard')
        self.assertEqual((seat['engine'], seat['checkpoint'], seat['device']), ('bubble:fake', 'main/000002', 'cpu'))
        self.assertEqual(seat['budget'], PRESETS['bubble']['quick'])
        self.assertEqual(self.session.match_seat('Native@lightning', 'standard')['budget'], dict(ms=100))
        custom = self.session.match_seat('bubble:2{simulations=512,solver_nodes=0}', 'standard')
        self.assertEqual(custom['budget'], dict(simulations=512, solver_nodes=0))
        with self.assertRaisesRegex(ValueError, 'not a budget'):
            self.session.match_seat('Native{simulations=128}', 'standard')

    def test_clock_deadline_discards_late_moves_and_records_time_result(self):
        late_reply = lambda game, *a, **kw: (
            time.sleep(.06) or dict(moves=legal_turn([c[:2] for c in game.cells]), elapsed_ms=60))
        for name, reply in [('late', late_reply), ('timeout', TimeoutError('Opponent exceeded its allowance'))]:
            with self.subTest(name=name), unittest.mock.patch('timed_engine.TimedEngine') as engine:
                engine.return_value.identity = dict(kind='fake')
                engine.return_value.turn.side_effect = reply
                output = self.output / name
                self.session.start_match(['Native', 'Other'], games=1, output=output,
                                         clock=dict(mode='move', ms=20))
                wait(lambda: not self.session.match_worker.is_alive())
                result = json.loads((output / 'game-0001.json').read_text())
                self.assertEqual((result['winner'], result['reason'], result['history']), (0, 'time', [[0, 0]]))
                self.assertEqual(self.session.match['wins'], [1, 0])
                self.assertIsNone(self.session.match['error'])

    def test_a_batch_cannot_take_over_an_unfinished_human_game(self):
        self.session.configure_seat(1, 'human')
        self.session.play(0, 0)
        with self.assertRaisesRegex(ValueError, 'human game'):
            self.session.start_match(['Native', 'Other'], output=self.output)
        self.assertEqual(self.session.history, [(0, 0)])
        self.session.start_match(['Native', 'Other'], games=2, output=self.output, max_placements=3, replace=True)
        wait(lambda: not self.session.match_worker.is_alive())
        self.assertEqual(self.session.match['completed'], 2)

    def test_simulations_only_adapter_refuses_a_clock_before_start(self):
        self.session.entries['strix:Strix'] = dict(id='strix:Strix', name='Strix', kind='strix',
            presets=PRESETS['strix'], model=Path(RUN.name) / 'checkpoints/main/000001/ema.pt')
        with self.assertRaisesRegex(ValueError, 'cannot enforce a clock'):
            self.session.start_match(['Native', 'Strix'], output=self.output, clock=dict(mode='game', tc='180+2'))
        self.assertFalse(self.output.exists())


class Proofs(unittest.TestCase):
    def test_placements_become_the_winners_turns(self):
        self.assertEqual([proof_turns(p, 2, True) for p in (1, 2, 5, 6, 9, 10)], [1, 1, 2, 2, 3, 3])
        self.assertEqual([proof_turns(p, 1, True) for p in (1, 4, 5)], [1, 2, 2])
        self.assertEqual([proof_turns(p, 2, False) for p in (3, 4, 7, 8)], [1, 1, 2, 2])


class TurnTrees(unittest.TestCase):
    """A fixed-budget play turn searches its second stone in the tree its first stone grew."""

    def setUp(self):
        import hexnet
        import neural_search
        import torch
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)/'ema.pt'
        torch.manual_seed(0)
        hexnet.save_model(self.path, hexnet.HexNet(hexnet.HexNetConfig(
            blocks=1, channels=8, pool_every=1, line_length=5, value_hidden=8, head_channels=4)))
        self.trees = []
        trees = self.trees

        class Spy(neural_search.NeuralSearch):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.searched = []
                trees.append(self)

            def search(self, *args, **kwargs):
                result = super().search(*args, **kwargs)
                self.searched.append((list(self.history), result['completed'], int(result['visits'].sum())))
                return result
        patcher = unittest.mock.patch.object(neural_search, 'NeuralSearch', Spy)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_play_evaluation_keeps_one_tree_for_the_turn(self):
        import hexnet
        import neural_search
        from play import evaluate
        from types import SimpleNamespace
        model = hexnet.load_model(self.path)
        bubble = SimpleNamespace(evaluator=hexnet.DenseEvaluator(model, 'cpu', 'tiny', max_batch=16), sha256='tiny',
                                 cache=neural_search.EvaluationCache())
        seen = []
        # A stopped clock: the first glimpse with statistics is shown at once and the throttle holds back the rest.
        with unittest.mock.patch('play.time', SimpleNamespace(monotonic=lambda: 0., perf_counter=time.perf_counter)):
            found = evaluate(bubble, None, [(0, 0)], 32, 0, live=seen.append)
        moves = found['moves']
        self.assertEqual(len(seen), 1)
        self.assertTrue(all(len(g['top']) <= 5 and 0 <= g['value'] <= 1 and g['top'][0][2] >= g['top'][-1][2]
                            for g in seen))
        self.assertTrue(found['top'] and all(len(t) == 5 and 0 <= t[3] <= 1 and t[4] in (-1, 0, 1) for t in found['top']))
        self.assertTrue(all(len(t) == 3 for t in evaluate(bubble, None, [(0, 0)], 0, 0)['top']))
        self.assertEqual(len(self.trees), 1)
        tree = self.trees[0]
        self.assertEqual([h for h, _, _ in tree.searched], [[(0, 0)], [(0, 0), tuple(moves[0])]])
        self.assertGreater(tree.searched[1][2], tree.searched[1][1])
        self.assertIsNone(tree.ptr)
        self.assertEqual([len(step['history']) for step in found['later']], [2])

    def bubble(self):
        import hexnet
        import neural_search
        from types import SimpleNamespace
        model = hexnet.load_model(self.path)
        return SimpleNamespace(evaluator=hexnet.DenseEvaluator(model, 'cpu', 'tiny', max_batch=64), sha256='tiny',
                               cache=neural_search.EvaluationCache())

    def test_a_turn_the_solver_gave_still_ranks_each_of_its_positions(self):
        from play import evaluate
        pv = [[1, 0, 1, 1], [2, 0, 1, 2], [2, 3, 0, 3], [3, 3, 0, 4], [3, 0, 1, 5], [4, 0, 1, 6]]
        given = dict(moves=[[1, 0], [2, 0]], pv=pv, proof=dict(winner=1, turns=2, plies=6), threat=[], solved=True,
                     used=0)
        found = evaluate(self.bubble(), None, [(0, 0)], 16, 0, solved=given)
        self.assertEqual((found['moves'], found['value'], found['pv']), ([[1, 0], [2, 0]], 1., pv))
        self.assertEqual((found['top'][0][:2], found['top'][0][3:]), ([1, 0], [1., 1]))
        [step] = found['later']
        self.assertEqual((step['history'], step['moves'], step['top'][0][:2], step['top'][0][3:]),
                         ([(0, 0), (1, 0)], [[2, 0]], [2, 0], [1., 1]))
        self.assertEqual((step['pv'], step['proof'], step['value']),
                         ([[*p[:3], p[3] - 1] for p in pv[1:]], dict(winner=1, turns=2, plies=5), 1.))

    def test_the_first_row_is_the_stone_played(self):
        from play import evaluate
        for simulations in (0, 16):
            found = evaluate(self.bubble(), None, [(0, 0)], simulations, 0)
            self.assertEqual(found['top'][0][:2], found['moves'][0])
            for step in found['later']:
                self.assertEqual(step['top'][0][:2], step['moves'][0])
            self.assertEqual(found['pv'], [])

    def test_a_search_proof_shows_its_own_turn(self):
        from play import evaluate
        history = [(0, 0), (1, 2), (2, 2), (-2, 0), (-3, 0), (3, 2), (4, 2), (0, -3), (0, -4)]
        found = evaluate(self.bubble(), None, history, 64, 0)
        self.assertEqual(found['proof']['winner'], 1)
        self.assertEqual(found['pv'], [[*m, 1, i + 1] for i, m in enumerate(found['moves'])])
        self.assertGreater(found['proof']['plies'], 0)

    def test_a_proof_found_at_the_second_stone_counts_the_first(self):
        import numpy as np
        from play import TurnSearch, solve
        actions = np.array([[1, 0], [2, 0]])
        results = iter([dict(action=[1, 0], policy=np.array([.6, .4]), actions=actions, values=np.array([.1, .2]),
                             proven=0, exact_winner=-1, proof_plies=0),
                        dict(action=[2, 0], policy=np.array([0., 1.]), actions=actions, values=np.array([0., 1.]),
                             proven=1, exact_winner=1, proof_plies=5)])
        turn = TurnSearch(None, None, [(0, 0)], 1, solve(None, [(0, 0)], 0), trees=lambda *a: (None, 1))
        try:
            turn.take(next(results))
            turn.take(next(results))
            found = turn.record()
        finally:
            turn.close()
        self.assertEqual((found['proof']['winner'], found['proof']['plies']), (1, 6))
        self.assertEqual(found['later'][0]['proof']['plies'], 5)
        self.assertEqual(found['pv'], [[1, 0, 1, 1], [2, 0, 1, 2]])

    def test_a_proven_second_stone_keeps_its_proof(self):
        from play import evaluate
        history = [(0, 0), (1, 2), (2, 2), (-2, 0), (-3, 0), (3, 2), (4, 2), (0, -3), (0, -4)]
        found = evaluate(self.bubble(), None, history, 64, 0)
        later = found['later'][0]
        self.assertEqual((later['proof']['winner'], later['value']), (1, 1.))

    def test_pooled_evaluations_match_single_ones(self):
        from play import evaluate, evaluate_many
        histories = [[(0, 0)], [(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 0)]]
        pooled = evaluate_many(self.bubble(), [], histories, 16, 0, batch_size=64)
        for history, found in zip(histories, pooled):
            alone = evaluate(self.bubble(), None, history, 16, 0)
            self.assertEqual((found['moves'], [t[:2] for t in found['top']]), (alone['moves'], [t[:2] for t in alone['top']]))
            self.assertAlmostEqual(found['value'], alone['value'], places=3)

    def test_a_pooled_review_asks_the_solver_once_per_position(self):
        from dense_solver import VERDICTS
        from play import evaluate_many
        asked = []

        class Prover:
            def history(self, history, **options):
                asked.append((position_text(history), options['attacker']))
                return dict(status='UNKNOWN', reason=next(iter(VERDICTS)), nodes_used=1)

            def abort(self):
                pass
        histories = [[(0, 0)], [(0, 0), (1, 0), (1, 1)], [(0, 0)]]
        found = evaluate_many(self.bubble(), [Prover(), Prover()], histories, 0, 64)
        self.assertEqual(sorted(asked), sorted((position_text(h), side) for h in histories[:2] for side in ('mover', 'opponent')))
        self.assertTrue(all(f['solved'] for f in found))

    def test_kept_trees_run_only_the_simulations_they_lack(self):
        from play import evaluate
        engines, bubble, history = Engines('cpu'), self.bubble(), [(0, 0)]
        self.addCleanup(engines.close)
        for tier in (8, 32, 128):
            evaluate(bubble, None, history, tier, 0, trees=engines.kept_trees(bubble, history))
        first = self.trees[0]
        self.assertEqual([h for h, _, _ in first.searched], [[(0, 0)]] * 3)
        self.assertEqual(sum(1 for t in self.trees if t.history == [(0, 0)]), 1)
        self.assertGreaterEqual(first.searched[-1][2], 128)
        self.assertLess(first.searched[-1][2], 128 + 16)
        engines.kept_trees(bubble, [(0, 0), (1, 0)])
        self.assertIsNone(first.ptr)
        trees = engines.kept_trees(bubble, [(0, 0)], 'abc')
        tree, missing = trees([(0, 0)], 32, bubble.evaluator)
        tree.search(10, root_samples=16, batch_size=16)
        self.assertEqual(trees([(0, 0)], 32, bubble.evaluator), (tree, 32 - int(tree.result(0, 0, 0, 0)['visits'].sum())))
        engines.kept_trees(bubble, [(0, 0)], 'def')
        self.assertIsNone(tree.ptr)

    def test_rows_put_proven_wins_first_and_proven_losses_last(self):
        import numpy as np
        from play import top_rows
        rows = top_rows(np.array([[1, 0], [2, 0], [3, 0], [4, 0]]), np.array([.4, .3, .2, .1]), np.array([-1., .2, 1., .1]))
        self.assertEqual([r[:2] for r in rows], [[3, 0], [2, 0], [4, 0], [1, 0]])

    def test_rows_lead_with_the_stone_played_and_skip_untried_stones(self):
        import numpy as np
        from play import top_rows
        actions = np.array([[-20, 5], [-20, 6], [1, 0], [2, 0], [3, 0]])
        policy, values = np.array([0., 0., .7, .3, 0.]), np.array([.99, .99, .6, .5, 1.])
        self.assertEqual([r[:2] for r in top_rows(actions, policy, values)], [[3, 0], [1, 0], [2, 0]])
        self.assertEqual([r[:2] for r in top_rows(actions, policy, values, lead=[2, 0])], [[2, 0], [3, 0], [1, 0]])
        self.assertEqual([r[:2] for r in top_rows(actions, policy)], [[1, 0], [2, 0]])
        self.assertEqual(top_rows(actions, policy, values, lead=[2, 0], won=True)[0], [2, 0, .3, 1., 1])
        self.assertEqual(top_rows(actions, policy, None, lead=[9, 9], won=True)[0], [9, 9, 0., 1., 1])
        self.assertEqual(len(top_rows(actions, policy, values, count=2, lead=[9, 9], won=True)), 2)
        self.assertEqual(len(top_rows(actions, policy, values, count=3, lead=[2, 0], won=True)), 3)


class PrincipalVariation(unittest.TestCase):
    """`principal_variation` and the solver step of an evaluation."""
    # Side 1 to move at [(0, 0)]: its two stones, then defender covers lasting one or two more attacker turns (of the
    # two longest, the far one listed first), then an unstoppable fork whose defender stones do not matter.
    CERTIFICATE = dict(root=0, nodes=[
        dict(kind='attacker_move', action=[[1, 0], [2, 0]], child=1,
             alternatives=[dict(action=[[0, 1], [0, 2]], child=2)]),
        dict(kind='defender_replies', responses=[dict(action=[[1, 2], [2, 2]], child=2),
                                                 dict(action=[[-5, 5], [-6, 6]], child=3),
                                                 dict(action=[[2, 3], [3, 3]], child=3)]),
        dict(kind='immediate_win', action=[[3, 0], [4, 0]]),
        dict(kind='attacker_move', action=[[3, 0], [4, 0]], child=4),
        dict(kind='unstoppable', threats=[[[5, 0], [6, 0]], [[-1, 0]]])])

    def test_shortest_attack_nearest_longest_defence_and_no_filler(self):
        from play import principal_variation
        pv, plies = principal_variation([(0, 0)], self.CERTIFICATE)
        self.assertEqual(pv, [[1, 0, 1, 1], [2, 0, 1, 2], [2, 3, 0, 3], [3, 3, 0, 4], [3, 0, 1, 5], [4, 0, 1, 6],
                              [-1, 0, 1, 9]])
        self.assertEqual(plies, 9)

    def test_the_winning_turn_keeps_its_ply_numbers_after_left_out_stones(self):
        from play import principal_variation
        certificate = dict(root=0, nodes=[dict(kind='attacker_move', action=[[1, 0], [2, 0]], child=1),
                                          dict(kind='unstoppable', threats=[[[3, 0], [4, 0]]])])
        self.assertEqual(principal_variation([(0, 0)], certificate),
                         ([[1, 0, 1, 1], [2, 0, 1, 2], [3, 0, 1, 5], [4, 0, 1, 6]], 6))

    def test_an_immediate_win_ends_at_the_winning_stone(self):
        from play import principal_variation
        history = [(0, 0), (0, 5), (1, 5), (1, 0), (2, 0), (2, 5), (3, 5), (3, 0), (-1, 3), (5, 5), (6, 5)]
        certificate = dict(root=0, nodes=[dict(kind='immediate_win', action=[[4, 0], [5, 0]])])
        self.assertEqual(principal_variation(history, certificate), ([[4, 0, 0, 1], [5, 0, 0, 2]], 2))

    def test_solve_asks_for_the_shortest_win_and_keeps_its_line(self):
        from play import solve
        result = dict(status='PROVEN_WIN', native_verified=True, moves=[[1, 0], [2, 0]], proof_turns=3,
                      certificate=self.CERTIFICATE, nodes_used=7)
        prover = unittest.mock.Mock(history=unittest.mock.Mock(return_value=result))
        found = solve(prover, [(0, 0)], 4096)
        self.assertTrue(prover.history.call_args.kwargs['shortest'])
        self.assertEqual((found['moves'], found['proof']), ([[1, 0], [2, 0]], dict(winner=1, turns=3, plies=9)))
        self.assertEqual(found['pv'][:2], [[1, 0, 1, 1], [2, 0, 1, 2]])

    def test_a_late_win_shows_the_shortest_line(self):
        import tactical_proof
        from play import solve
        from tests.test_tactical_proof import LATE_WIN
        if not tactical_proof.library().exists():
            self.skipTest('needs the native tactical library')
        prover = tactical_proof.NativeTactics()
        prover.abort = lambda: None
        found = solve(prover, LATE_WIN, 32768)
        self.assertEqual((found['moves'], found['proof']), ([[-1, -11], [-1, -10]], dict(winner=0, turns=4, plies=14)))
        self.assertEqual((found['pv'][:2], len(found['pv'])), ([[-1, -11, 0, 1], [-1, -10, 0, 2]], 12))
        self.assertEqual([p[3] for p in found['pv'][-2:]], [13, 14])
        game = Game([tuple(p) for p in LATE_WIN] + [tuple(p[:2]) for p in found['pv'][:-2]])
        try:
            self.assertEqual((game.winner, game.player), (-1, 1))
            for q, r in [m for m in game.legal_moves() if list(m) not in [p[:2] for p in found['pv']]][-2:]:
                game.play(q, r)
            for q, r, _, _ in found['pv'][-2:]:
                game.play(q, r)
            self.assertEqual(game.winner, 0)
        finally:
            game.close()


class Registry(unittest.TestCase):
    def test_six_takes_the_fastest_backend_it_can_load(self):
        library = lambda kind: SIX_LIBRARIES[kind][os.name != 'nt']
        with tempfile.TemporaryDirectory() as directory:
            six, libraries = Path(directory) / 'six', Path(directory) / 'gpu'
            six.mkdir()
            libraries.mkdir()
            with unittest.mock.patch('importlib.util.find_spec', return_value=None), \
                    unittest.mock.patch.dict(os.environ, dict(PATH=str(libraries))):
                self.assertEqual(six_backend(six), ('CPU', ['--cpu'], []))
                for kind in ('cuda', 'cudnn', 'tensorrt'):
                    (libraries / library(kind)).write_bytes(b'')
                self.assertEqual(six_backend(six), ('CPU', ['--cpu'], []))
                (six / 'DirectML.dll').write_bytes(b'')
                self.assertEqual(six_backend(six)[:2], ('DirectML', []) if os.name == 'nt' else ('CPU', ['--cpu']))
                (six / 'DirectML.dll').unlink()
                (six / library('cuda_build')).write_bytes(b'')
                self.assertEqual(six_backend(six)[:2], ('CUDA', []))
                (six / library('tensorrt_build')).write_bytes(b'')
                self.assertEqual(six_backend(six)[:2], ('TensorRT', ['--trt']))
                (libraries / library('tensorrt')).unlink()
                self.assertEqual(six_backend(six)[:2], ('CUDA', []))

    def test_six_folders_and_engine_entries_are_found(self):
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory)
            (models / 'six').mkdir()
            for name in ('sixengine.exe' if os.name == 'nt' else 'sixengine', 'gen-0100.onnx', 'gen-0120.onnx'):
                (models / 'six' / name).write_bytes(b'')
            (models / 'shrimp.json').write_text(json.dumps(dict(
                kind='six', badge='shrimp', command=['python', 'driver.py'], presets=dict(quick=dict(nodes=1, args=['--visits', '32'])))))
            (models / 'loud.json').write_text(json.dumps(dict(kind='six', badge='Shrimp!', command=['python', 'driver.py'])))
            (models / 'strix.json').write_text(json.dumps(dict(name='Strix', kind='strix', model='strix.safetensors')))
            (models / 'broken.json').write_text(json.dumps(dict(kind='six', command=[], presets=dict(odd={}))))
            (models / 'spaced.json').write_text(json.dumps(dict(kind='six', command='six --cpu')))
            with unittest.mock.patch('play.six_backend', return_value=('CPU', ['--cpu'], [])):
                found = scan(models, None, [], None)
            self.assertEqual(list(found), ['six:Six · CPU', 'six:shrimp', 'strix:Strix', 'native:Native'])
            six = found['six:Six · CPU']
            self.assertEqual((six['label'], six['checkpoints'], found['six:shrimp']['label']),
                             ('CPU', ['gen-0120', 'gen-0100'], 'shrimp'))
            self.assertEqual((command_of(six, 'gen-0100')[1:], six['mirrored']),
                             (['--net', str(models / 'six/gen-0100.onnx'), '--cpu'], True))
            self.assertEqual(command_of(found['six:shrimp'], None), [sys.executable, 'driver.py'])
            shrimp = found['six:shrimp']
            self.assertEqual((shrimp['command'], shrimp['mirrored']), ([sys.executable, 'driver.py'], False))
            self.assertEqual(shrimp['presets']['quick'], dict(nodes=1, args=['--visits', '32']))
            self.assertEqual(found['strix:Strix']['presets']['deep'], dict(simulations=512))
            self.assertEqual({key: e['badge'] for key, e in found.items()},
                             {'six:Six · CPU': 'six', 'six:shrimp': 'shrimp', 'strix:Strix': 'strix', 'native:Native': 'native'})

    def test_runs_models_and_entries_are_found(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for checkpoint in ('main/000100', 'main/000200', 'play/000150'):
                (root / 'runs/alpha/checkpoints' / checkpoint).mkdir(parents=True)
                (root / 'runs/alpha/checkpoints' / checkpoint / 'ema.pt').write_bytes(b'')
            (root / 'runs/alpha/champion.json').write_text(json.dumps(dict(checkpoint='play/000150')))
            (root / 'runs/broken/checkpoints/main/000001').mkdir(parents=True)
            (root / 'runs/broken/checkpoints/main/000001/ema.pt').write_bytes(b'')
            (root / 'runs/broken/champion.json').write_text('{"checkpoint": ')
            (root / 'models/beta').mkdir(parents=True)
            (root / 'models/beta/ema.pt').write_bytes(b'')
            (root / 'elsewhere').mkdir()
            (root / 'elsewhere/net.pt').write_bytes(b'')
            (root / 'models/gamma.json').write_text(json.dumps(dict(name='gamma', kind='bubble', path='../elsewhere/net.pt')))
            (root / 'models/list.json').write_text('[1, 2]')
            found = scan(root / 'models', root / 'runs', [root / 'runs/alpha'], root / 'missing.dll')
            self.assertEqual(list(found), ['bubble:alpha', 'bubble:broken', 'bubble:beta', 'bubble:gamma', 'native:Native'])
            self.assertEqual(found['bubble:alpha']['checkpoints'], ['play/000150', 'main/000200', 'main/000100'])
            self.assertEqual(found['bubble:beta']['checkpoints'], [''])
            self.assertEqual([found[k]['label'] for k in ('bubble:alpha', 'bubble:broken', 'bubble:gamma')],
                             ['Bubble', 'broken', 'gamma'])
            both = scan(None, None, [root / 'elsewhere/net.pt', root / 'runs/alpha'], None)
            self.assertEqual([e['label'] for e in both.values() if e['kind'] == 'bubble'], ['Bubble', 'alpha'])
            (root / 'models/gamma').mkdir()
            (root / 'models/gamma/ema.pt').write_bytes(b'')
            twins = [k for k in scan(root / 'models', root / 'runs', [], None) if k.startswith('bubble:gamma')]
            self.assertEqual(len(twins), 2)
            (root / 'models/a').mkdir()
            (root / 'models/a/gamma.pt').write_bytes(b'')
            triple = scan(root / 'models', root / 'runs', [], None)
            self.assertTrue(set(twins) < set(triple))
            self.assertEqual({triple[k]['path'] for k in twins}, {found['bubble:gamma']['path'], root / 'models/gamma/ema.pt'})


if __name__ == '__main__':
    unittest.main()
