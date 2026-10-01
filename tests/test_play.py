"""Play server: evaluation store, review labels, background jobs and the HTTP surface, with fake engines."""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from hexo import Game
from play import (Cancelled, Engines, Evaluations, Handler, PRESETS, SIX_LIBRARIES, SearchChild, Session, book_openings, budget_of,
                  export_path, file_digest, file_identity, import_history, model_key, presets_of, proof_turns, review, scan, six_backend)
from process_tree import TreeProcess

STANDARD = PRESETS['bubble']['standard']


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
        self.calls, self.hold, self.release = [], False, threading.Event()

    def evaluate(self, entry, checkpoint, budget, history, watch):
        self.calls.append((checkpoint, dict(budget), [tuple(p) for p in history]))
        while self.hold and not self.release.is_set():
            watch(1)
            time.sleep(.01)
        watch(budget['simulations'])
        moves = legal_turn(history)
        found = dict(moves=moves, value=.5, top=[[*moves[0], .9]], proof=None, line=[], threat=[], ms=1)
        return found, budget, f'{model_key(export_path(entry, checkpoint))}:none'

    def solver_build(self):
        return 'none'

    def effective(self, budget):
        return budget

    def turn(self, entry, budget, history, stop):
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

    def test_index_keeps_the_newest_entries(self):
        store = Evaluations(limit=2)
        for n in range(3):
            store.add([(0, 0)] if n == 0 else [(0, 0), (n, 0)], 'e', STANDARD, dict(value=n / 10, moves=[], top=[]))
        self.assertIsNone(store.best([(0, 0)], 'e'))
        self.assertEqual(store.best([(0, 0), (2, 0)], 'e')['value'], .2)


def evaluation(value, moves=(), proof=None, line=()):
    return dict(value=value, moves=[list(m) for m in moves], proof=proof, line=[list(p) for p in line])


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
        self.assertFalse(self.session.worker.is_alive())
        self.assertEqual(closed, [True])

    def test_undo_returns_to_the_players_last_turn(self):
        self.session.play(0, 0)
        wait(lambda: len(self.history()) == 3)
        self.session.pause(True)
        self.session.play(0, 1)
        self.session.undo()
        self.assertEqual(self.history(), [(0, 0), *map(tuple, legal_turn([(0, 0)]))])
        self.session.undo()
        self.assertEqual(self.history(), [])

    def test_failures_reach_the_page_and_rescans_drop_vanished_engines(self):
        def broken(*args):
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
                self.assertEqual(self.post('/match/stop')['match']['active'], False)
                self.post('/new')
                self.assertIsNone(json.loads(self.get('/state'))['match'])
            finally:
                self.session.close()

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

    def test_seat_specs_resolve_presets_checkpoint_steps_and_custom_budgets(self):
        seat = self.session.match_seat('bubble:2@quick', 'standard')
        self.assertEqual((seat['engine'], seat['checkpoint'], seat['device']), ('bubble:fake', 'main/000002', 'cpu'))
        self.assertEqual(seat['budget'], PRESETS['bubble']['quick'])
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
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)/'ema.pt'
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
        moves = evaluate(bubble, None, [(0, 0)], 32, 0)['moves']
        self.assertEqual(len(self.trees), 1)
        tree = self.trees[0]
        self.assertEqual([h for h, _, _ in tree.searched], [[(0, 0)], [(0, 0), tuple(moves[0])]])
        self.assertGreater(tree.searched[1][2], tree.searched[1][1])
        self.assertIsNone(tree.ptr)


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
                kind='six', command=['python', 'driver.py'], presets=dict(quick=dict(nodes=1, args=['--visits', '32'])))))
            (models / 'strix.json').write_text(json.dumps(dict(name='Strix', kind='strix', model='strix.safetensors')))
            (models / 'broken.json').write_text(json.dumps(dict(kind='six', command=[], presets=dict(odd={}))))
            (models / 'spaced.json').write_text(json.dumps(dict(kind='six', command='six --cpu')))
            with unittest.mock.patch('play.six_backend', return_value=('CPU', ['--cpu'], [])):
                found = scan(models, None, [], None)
            self.assertEqual(list(found), ['six:Six gen-0120 · CPU', 'six:Six gen-0100 · CPU', 'six:shrimp',
                                           'strix:Strix', 'native:Native'])
            six = found['six:Six gen-0120 · CPU']
            self.assertEqual((six['command'][1:], six['mirrored']),
                             (['--net', str(models / 'six/gen-0120.onnx'), '--cpu'], True))
            shrimp = found['six:shrimp']
            self.assertEqual((shrimp['command'], shrimp['mirrored']), (['python', 'driver.py'], False))
            self.assertEqual(shrimp['presets']['quick'], dict(nodes=1, args=['--visits', '32']))
            self.assertEqual(found['strix:Strix']['presets']['deep'], dict(simulations=512))

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
