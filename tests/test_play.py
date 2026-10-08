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
from play import (CPU_PRESETS, Cancelled, Engines, Evaluations, Handler, PRESET_NAMES, PRESETS, SIX_LIBRARIES, SearchChild, Session,
                  book_openings, budget_of, command_of, custom_form, export, export_path, file_digest, file_identity,
                  import_history, linked_history, model_key, move_row, pair_elo, pick_opening, position_text, presets_of,
                  proof_turns, read_game, review, review_plies, scan, search_key, six_backend)
from process_tree import TreeProcess
from tests import PATIENCE, slow

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


def wait(condition, timeout=PATIENCE):
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return
        time.sleep(.01)
    raise AssertionError('condition not reached')


class FakeEngines:
    """Plays the first legal cells; `hold` makes evaluations wait until cancelled or released."""

    def __init__(self):
        self.calls, self.turns, self.lines, self.hold, self.release = [], [], [], False, threading.Event()
        self.games, self.refreshes, self.graph = [], [], (None, 0)

    def evaluate(self, entry, checkpoint, budget, history, watch, live=None, keep=False, line=None, known=None,
                 game=None, refresh=None, used=None):
        self.calls.append((checkpoint, dict(budget), [tuple(p) for p in history]))
        self.lines.append(line)
        self.games.append(game)
        if refresh is not None:
            self.refreshes.append(len(history))
        if game is not None:   # one graph per game, a new one when the weights change, as Engines.game_graph
            if self.graph[0] != (game, checkpoint):
                self.graph = (game, checkpoint), self.graph[1] + 1
            used.append(self.graph[1])
        while self.hold and not self.release.is_set():
            watch(1)
            time.sleep(.01)
        watch(budget['simulations'])
        moves = legal_turn(history)
        value = .5 + getattr(self, 'drift', 0) * len(self.refreshes) if refresh is not None else .5
        found = dict(moves=moves, value=value, top=[[*moves[0], .9, .5]], proof=None, line=[], threat=[], ms=1)
        return found, budget, f'{model_key(export_path(entry, checkpoint))}:none' + (':kept' if keep or line is not None else '')

    def evaluate_many(self, entry, checkpoint, budget, histories, watch, known=None):
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
            'drip:Drip': dict(id='drip:Drip', name='Drip', kind='drip', presets=PRESETS['drip'])}


class Store(unittest.TestCase):
    def test_leaf_proofs_survive_a_later_unproven_record_and_disk_reload(self):
        from play import Proofs
        history, budget = [(0, 0)], dict(simulations=8, solver_nodes=0)
        fact = dict(history=[[0, 0], [1, 0]], winner=1, plies=5,
                    pv=[[2, 0, 1, 1], [3, 0, 1, 4], [4, 0, 1, 5]])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'evaluations.jsonl'
            store = Evaluations(path)
            store.add(history, 'test', budget, evaluation(.5) | dict(top=[], proofs=[fact]))
            store.add(history, 'test', budget, evaluation(.5) | dict(top=[]))
            restored = Evaluations(path)
            self.assertEqual(restored.get(history, 'test', budget)['proofs'], [fact])
            table = Proofs()
            table.extend(restored, history)
            self.assertEqual(table.known(history)['pv'], [[1, 0, 1, 1]] + [[*p[:3], p[3]+1] for p in fact['pv']])

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
        store, wanted = Evaluations(), dict(simulations=128, solver_nodes=32768)
        store.add([], 'e', dict(simulations=512, solver_nodes=0), dict(value=.1, moves=[], top=[]))
        self.assertIsNone(store.covering([], 'e', wanted))
        store.add([], 'e', dict(simulations=128, solver_nodes=131072), dict(value=.2, moves=[], top=[]))
        self.assertEqual(store.covering([], 'e', wanted)['value'], .2)
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
        self.assertEqual([g['label'] for g in self.labels({}, 1, history)[-1]['grades']], [None, 'win'])

    def test_each_stone_is_graded_from_the_position_before_it(self):
        base = {(): evaluation(.5, [(0, 0)]), (0,): evaluation(.6, [(2, 2), (3, 3)])}
        first_blunder = self.labels(base | {(0, 1): evaluation(.3, [(4, 4)]), (0, 1, 2): evaluation(.72)})[1]
        self.assertEqual(first_blunder['label'], 'blunder')
        first, second = first_blunder['grades']
        self.assertEqual((first['label'], first['better'], first['line']), ('blunder', [[2, 2]], [[2, 2, 1], [3, 3, 1]]))
        self.assertAlmostEqual(first['before'] - first['after'], .3)
        self.assertEqual((second['label'], second['better']), ('good', None))
        self.assertAlmostEqual(second['before'] - second['after'], .02)
        second_blunder = self.labels(base | {(0, 1): evaluation(.58, [(4, 4)]), (0, 1, 2): evaluation(.7)})[1]
        first, second = second_blunder['grades']
        self.assertEqual((first['label'], second['label'], second['better'], second['line']),
                         ('good', 'blunder', [[4, 4]], [[4, 4, 1]]))
        self.assertEqual(second_blunder['label'], 'blunder')

    def test_a_stone_of_the_engine_turn_is_best_in_either_order(self):
        turn = self.labels({(0,): evaluation(.5, [(3, 3), (1, 0)]), (0, 1): evaluation(.5, [(1, 1)]),
                            (0, 1, 2): evaluation(.5)})[1]
        self.assertEqual([g['label'] for g in turn['grades']], ['best', 'best'])
        self.assertEqual(turn['label'], 'good')
        turn = self.labels({(0,): evaluation(.5, [(3, 3), (1, 0)]), (0, 1): evaluation(.5, [(1, 1)]),
                            (0, 1, 2): evaluation(.5, [], dict(winner=0, turns=1))})[1]
        self.assertEqual([(g['label'], g['better']) for g in turn['grades']], [('best', None), ('allowed', None)])
        self.assertEqual((turn['label'], turn['better']), ('allowed', [[3, 3], [1, 0]]))

    def test_a_turn_waiting_for_its_second_stone_grades_its_first(self):
        history = self.history[:4]
        turns = self.labels({(0, 1, 2): evaluation(.5, [(5, 5), (6, 6)]), (0, 1, 2, 3): evaluation(.25)}, history=history)
        self.assertEqual((len(turns), turns[-1]['stones'], turns[-1]['label']), (3, [[-1, 0]], None))
        self.assertEqual(review([], lambda prefix: evaluation(.5)), [])
        self.assertEqual(turns[-1]['grades'][0]['label'], 'blunder')

    def test_review_plies_count_every_stone(self):
        self.assertEqual(review_plies(self.history), [0, 1, 2, 3, 4, 5])
        self.assertEqual(review_plies(self.history[:2]), [0, 1, 2])


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
                session.configure_seat(1, 'drip:Drip', preset='quick')
                session.configure_seat(0, 'six:slow')
                self.assertEqual(json.loads(json.dumps(session.state()))['engines'][-1], dict(
                    id='six:slow', name='slow', kind='six', presets=presets_of('six', None), clocks=True, device='Server CPU'))
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
        self.session.configure_seat(0, 'drip:Drip', preset='quick')
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

    def test_analysis_reuses_deeper_evaluations_and_review_fills_every_position(self):
        self.session.configure_seat(1, 'human')
        for move in [(0, 0), (1, 0), (2, 0)]:
            self.session.play(*move)
        self.session.analyse(1, force=True)
        wait(lambda: not self.session.state()['jobs'])
        calls = len(self.engines.calls)
        self.assertIsNone(self.session.analyse(1))
        self.engines.hold = True   # a review still running is not queued twice; a finished one may be
        job = self.session.jobs[self.session.review_game()]
        self.assertEqual(self.session.review_game(), job.id)
        self.engines.release.set()
        wait(lambda: not self.session.state()['jobs'])
        self.assertEqual((job.done, job.total), (4, 4))
        self.assertEqual([len(c[2]) for c in self.engines.calls[calls:]], [3, 2, 0])
        turns = self.session.state()['review']
        self.assertEqual([t['label'] for t in turns], ['best', 'good'])
        self.assertTrue(all(g['label'] for t in turns for g in t['grades']))
        self.session.configure_analysis('bubble:fake', preset='deep', auto=False)
        self.session.analyse(1)
        wait(lambda: not self.session.state()['jobs'])
        self.session.configure_analysis('bubble:fake', preset='quick', auto=False)
        self.assertIsNone(self.session.analyse(1))
        self.assertEqual(self.session.state()['evaluations'][1]['simulations'], PRESETS['bubble']['deep']['simulations'])

    def test_the_solver_preset_round_trips_through_the_analysis_state_and_stays_off_the_seats(self):
        from play import SOLVER
        self.session.configure_analysis('bubble:fake', preset='solver', auto=False)
        analysis = self.session.state()['analysis']
        self.assertEqual((analysis['preset'], analysis['budget']), ('solver', SOLVER))
        # The page sends back what it was given.
        self.session.configure_analysis(analysis['engine'], analysis['checkpoint'], analysis['preset'], analysis['budget'],
                                        analysis['auto'])
        self.assertEqual(self.session.state()['analysis'], analysis)
        self.assertEqual(self.session.review_seat()['budget'], STANDARD)
        self.assertEqual(self.session.state()['review_preset'], 'standard')
        with self.assertRaisesRegex(ValueError, 'analysis'):
            self.session.configure_seat(1, 'bubble:fake', preset='solver')
        self.session.configure_analysis('bubble:fake', preset='standard', auto=False)
        self.assertEqual(self.session.state()['analysis']['budget'], STANDARD)

    def test_an_analysis_refreshes_the_saved_positions_before_it(self):
        self.session.configure_seat(1, 'human')
        for move in [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 1)]:
            self.session.play(*move)
        for ply in (1, 3):
            self.session.analyse(ply, force=True)
            wait(lambda: not self.session.state()['jobs'])
        self.engines.refreshes.clear()
        self.session.analyse(6, force=True)
        wait(lambda: not self.session.state()['jobs'])
        self.assertEqual(self.engines.refreshes, [3])
        self.assertEqual(set(self.engines.games), {self.session.analysis_line})
        # Ply 1 is too far back to be refreshed at once; the refresh at 3 re-read the graph and stales nothing, so 6
        # stays fresh. Viewing 1 again searches the graph there again, and that refresh stales nothing either.
        self.assertEqual(self.session.state()['stale'], [1])
        self.session.analyse(1)
        wait(lambda: not self.session.state()['jobs'])
        self.assertEqual(self.engines.refreshes, [3, 1])
        self.assertEqual(self.session.state()['stale'], [])
        # Another network's graph holds nothing for this network's records. A rebuilt graph for this network starts
        # empty, so its first deep analysis leaves every earlier record behind it: 3 is refreshed, 1 is too far back.
        self.session.configure_analysis('bubble:fake', 'main/000001', auto=False)
        self.session.analyse(6, force=True)
        wait(lambda: not self.session.state()['jobs'])
        self.assertEqual(self.engines.refreshes, [3, 1])
        self.session.configure_analysis('bubble:fake', auto=False)
        self.session.analyse(6, force=True)
        wait(lambda: not self.session.state()['jobs'])
        self.assertEqual(self.engines.refreshes, [3, 1, 3])
        self.assertEqual(self.session.state()['stale'], [1])
        self.session.analyse(1)
        wait(lambda: not self.session.state()['jobs'])
        self.assertEqual(self.session.state()['stale'], [])
        # A deepening tier's evaluation is refreshed under its own key and budget.
        session, deep = self.session, PRESETS['bubble']['deep']
        key = session.engine_key(session.analysis) + ':kept'
        session.store.add(session.history[:4], key, deep, dict(value=.5, moves=[[9, 9]], top=[], pv=[], threat=[],
                                                             proof=None, graph=[session.instance, session.analysis_graph, 0]))
        self.assertEqual(session.state()['stale'], [4])
        session.analyse(4)
        wait(lambda: not session.state()['jobs'])
        refreshed = session.store.get(session.history[:4], key, deep)['graph']
        self.assertEqual(refreshed[2], len(session.graph_plies[session.analysis_graph]))
        self.assertEqual(session.state()['stale'], [])
        session.new_lines()   # undo, a new or loaded game: the next analysis searches a new graph
        self.assertEqual(session.state()['stale'], [])
        line = self.session.analysis_line
        self.session.undo()
        self.assertNotEqual(self.session.analysis_line, line)

    def test_a_refresh_that_moves_its_result_continues_and_then_settles(self):
        self.session.configure_seat(1, 'human')
        for move in [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]:
            self.session.play(*move)
        self.engines.drift = .1
        for ply in (2, 5):
            self.session.analyse(ply, force=True)
            wait(lambda: not self.session.state()['jobs'])
        # The analysis at 5 refreshes 2; each refresh moves the value there, so it runs REFRESH_ROUNDS times and stops.
        self.assertEqual(self.engines.refreshes, [2, 2, 2])
        self.assertEqual(self.session.state()['stale'], [])

    def test_a_cancelled_analysis_still_counts_its_graph_search(self):
        self.session.configure_seat(1, 'human')
        for move in [(0, 0), (1, 0), (2, 0)]:
            self.session.play(*move)
        self.session.analyse(1, force=True)
        wait(lambda: not self.session.state()['jobs'])
        self.engines.hold = True
        job = self.session.analyse(3, force=True)
        wait(lambda: any(j['status'] == 'running' for j in self.session.state()['jobs']))
        self.session.cancel(job)
        wait(lambda: not self.session.state()['jobs'])
        self.assertEqual(self.session.state()['stale'], [1])

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
        # The review's later evaluations come from fresh trees after the analysis at 3; it refreshes none of them
        # at once (they do not exist yet), and viewing them later re-reads the graph there.
        self.assertEqual(([len(call[2]) for call in self.engines.calls], self.engines.refreshes), ([5, 3, 4, 2, 1, 0], []))

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
            session.configure_seat(1, 'drip:Drip', preset='quick')
            session.pause(False)
            wait(lambda: len(session.history) == 3)

    def test_changing_one_seat_leaves_the_other_engine_thinking(self):
        self.engines.hold = True
        self.session.play(0, 0)
        wait(lambda: any(j['status'] == 'running' for j in self.session.state()['jobs']))
        self.session.configure_seat(0, 'drip:Drip', preset='quick')
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
        self.assertEqual(self.engines.turns[-1], (six['id'], 'gen-1', PRESETS['six']['lightning']))
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
            for network, nodes in (('gen-2', 3840), ('gen-2', 960), ('gen-1', 960), ('gen-2', 960), ('gen-0', 960)):
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

    def test_a_cancelled_analysis_keeps_auto_and_waits_for_a_change_of_position_or_request(self):
        def running(ply):
            return [j for j in self.session.state()['jobs'] if j['kind'] == 'analyse' and j['ply'] == ply
                    and j['status'] in ('queued', 'running')]
        self.session.configure_analysis('bubble:fake', preset='quick', auto=True)
        self.engines.hold = True
        self.session.load([(0, 0), (1, 0), (1, 1)], False)
        wait(lambda: running(3))
        self.session.cancel(running(3)[0]['id'])
        wait(lambda: not running(3))
        time.sleep(.2)
        self.assertFalse(running(3))
        self.assertTrue(self.session.state()['analysis']['auto'])
        self.session.analyse(3)   # asking again clears the dismissal
        wait(lambda: running(3))
        self.session.cancel(running(3)[0]['id'])
        wait(lambda: not running(3))
        self.session.play(2, 2)   # so does a move
        wait(lambda: running(4))
        self.engines.release.set()

    def test_a_preset_without_a_solver_verdict_is_not_deepened_again(self):
        def unsolved(entry, checkpoint, budget, history, watch, live=None, keep=False, line=None, known=None, game=None,
                     used=None):
            found, spent, key = FakeEngines.evaluate(self.engines, entry, checkpoint, budget, history, watch, keep=keep,
                                                     line=line, game=game, used=used)
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

    def test_review_uses_the_analysis_strength(self):
        self.session.configure_seat(1, 'human')
        self.session.load([(0, 0), (1, 0), (2, 0)], True)
        self.session.configure_analysis('bubble:fake', preset='quick', auto=False)
        self.session.review_game()
        wait(lambda: not self.session.state()['jobs'])
        self.assertEqual({tuple(b.items()) for _, b, _ in self.engines.calls}, {tuple(PRESETS['bubble']['quick'].items())})
        state = self.session.state()
        self.assertEqual((state['review_preset'], [t['label'] is not None for t in state['review']]), ('quick', [True, True]))

    def test_undo_returns_to_the_players_last_turn(self):
        self.session.play(0, 0)
        wait(lambda: len(self.history()) == 3)
        self.session.pause(True)
        self.session.play(0, 1)
        self.session.undo()
        self.assertEqual(self.history(), [(0, 0), *map(tuple, legal_turn([(0, 0)]))])
        self.session.undo()
        self.assertEqual(self.history(), [])

    def test_a_seat_keeps_its_game_tree_line_until_undo_a_new_game_or_a_seat_change(self):
        def reply(count):
            for move in legal_turn(self.history()):
                self.session.play(*move)
            wait(lambda: len(self.history()) == count)

        self.session.play(0, 0)
        wait(lambda: len(self.history()) == 3)
        reply(7)
        self.session.undo()
        reply(7)
        self.session.new_game()
        self.session.play(0, 0)
        wait(lambda: len(self.history()) == 3)
        self.session.configure_seat(1, 'bubble:fake', 'main/000001')
        reply(7)
        first, second, undone, fresh, changed = self.engines.lines
        self.assertEqual(first, second)
        self.assertEqual(len({first, undone, fresh, changed}), 4)

    def test_undo_skips_a_seat_the_page_plays(self):
        self.session.configure_seat(1, 'human')
        self.session.play(0, 0)
        self.session.play(1, 0)
        self.session.play(2, 0)
        self.session.undo([0])
        self.assertEqual(self.history(), [])
        with self.assertRaises(ValueError):
            self.session.undo([2])

    def test_an_unseen_analysis_failure_belongs_to_its_settings_and_position(self):
        self.engines.evaluate = unittest.mock.Mock(side_effect=RuntimeError('weights unreadable'))
        old = self.session.analyse(0)
        wait(lambda: self.session.jobs[old].status == 'failed')
        # The page has not polled the failure before changing its selected preset.
        self.session.configure_analysis('bubble:fake', preset='quick', auto=False)
        self.assertFalse(self.session.state()['jobs'])
        current = self.session.analyse(0, force=True)
        wait(lambda: self.session.jobs[current].status == 'failed')
        self.assertEqual([j['id'] for j in self.session.state()['jobs']], [current])
        self.assertEqual(self.session.state()['jobs'][0]['error'], 'weights unreadable')
        self.session.load([(0, 0), (1, 0)], True)
        branch = self.session.analyse(2, force=True)
        wait(lambda: self.session.jobs[branch].status == 'failed')
        self.session.load([(0, 0), (0, 1)], True)
        self.assertNotIn(branch, [j['id'] for j in self.session.state()['jobs']])

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

    def test_cancelled_drip_searches_end_and_the_next_starts_at_once(self):
        engines = Engines('cpu')
        try:
            self.assertEqual(len(engines.turn(entries()['drip:Drip'], dict(ms=50), [(0, 0)])), 2)
            started = time.time()
            with self.assertRaises(Cancelled):
                engines.turn(entries()['drip:Drip'], dict(ms=20000), [(0, 0)], lambda: time.time() - started > .2)
            self.assertEqual(len(engines.turn(entries()['drip:Drip'], dict(ms=50), [(0, 0)])), 2)
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
                self.assertEqual((build, engines.solver()[0]), ('aaaaaaaa:stamps', first))
                self.assertTrue(isolated.call_args.kwargs['stamps'])
                record.write_text(json.dumps(dict(binary_sha256='b' * 64)))
                os.utime(record, ns=(2 * 10 ** 18, 2 * 10 ** 18))
                second, build = engines.solver()
                self.assertEqual(build, 'bbbbbbbb:stamps')
                reference = Engines('cpu', tactical_package=Path(directory), proof_stamps=False)
                self.assertEqual(reference.solver_build(), 'bbbbbbbb')
                first.abort.assert_called_once()
                self.assertEqual(isolated.call_count, 2)

    def test_exclusive_custom_budgets_apply_the_amount_last_edited(self):
        bubble, six = PRESETS['bubble'], PRESETS['six']
        # An older custom budget moves its simulations onto Nodes and drops its solver nodes.
        form = custom_form('bubble', bubble['standard'], dict(simulations=300, solver_nodes=9))
        self.assertEqual(form, dict(simulations=300, ms=1000, active='simulations', views=8))
        self.assertEqual(budget_of(bubble, 'custom', form, 'bubble'), dict(simulations=300, views=8, solver_nodes=4800))
        timed = form | dict(ms=2500, active='ms')   # editing Time applies it; Nodes keeps its value
        self.assertEqual(budget_of(bubble, 'custom', timed, 'bubble'),
                         dict(simulations=65536, ms=2500, views=8, solver_nodes=10000))
        self.assertEqual(custom_form('bubble', bubble['standard'], timed)['simulations'], 300)
        back = timed | dict(simulations=600, active='simulations', views=4)
        self.assertEqual(budget_of(bubble, 'custom', back, 'bubble'), dict(simulations=600, views=4, solver_nodes=9600))
        self.assertEqual(budget_of(six, 'custom', dict(nodes=700, ms=900, active='ms'), 'six'), dict(ms=900, nodes=2 ** 31 - 1))
        self.assertEqual(budget_of(six, 'custom', dict(nodes=700, ms=900, active='nodes'), 'six'), dict(nodes=700))
        for bad in (dict(active='nodes'), dict(views=0), dict(views=17), dict(ms=5)):
            with self.assertRaises(ValueError):
                custom_form('bubble', bubble['standard'], bad)
        session = Session(entries(), FakeEngines(), Evaluations())
        seat = session.seat('bubble:fake', None, 'custom', timed)
        self.assertEqual((seat['custom'], seat['budget']['ms']), (timed, 2500))
        self.assertNotEqual(search_key('w', dict(kind='bubble'), seat['budget']), search_key('w', dict(kind='bubble'), bubble['deep']))
        session.close()

    def test_budgets(self):
        bubble = PRESETS['bubble']
        for extra in (dict(leaf_nodes=2048), dict(leaf_ms=10)):
            with self.assertRaisesRegex(ValueError, 'not a budget'):
                budget_of(bubble, 'custom', extra, 'bubble')
        self.assertEqual(budget_of(bubble, 'custom', dict(simulations=0)), dict(simulations=0, solver_nodes=4096))
        self.assertEqual(budget_of(bubble, 'custom', dict(simulations=10 ** 6))['simulations'], 10 ** 6)
        for custom in (dict(simulations=-1), dict(simulations=2 ** 31), dict(ms=5), dict(simulations='8')):
            with self.assertRaises(ValueError):
                budget_of(bubble, 'custom', custom)
        with self.assertRaises(ValueError):
            budget_of(PRESETS['drip'], 'heavy')
        self.assertEqual(presets_of('six', dict(quick=dict(args=['--visits', '8'])))['quick'],
                         dict(nodes=240, args=['--visits', '8']))
        shrimp = presets_of('six', dict(quick=dict(nodes=1, args=['--visits', '32'])))
        self.assertEqual(budget_of(shrimp, 'custom', dict(nodes=9, args=['--visits', '1'])), dict(nodes=9))
        self.assertEqual(budget_of(shrimp, 'quick'), dict(nodes=1, args=['--visits', '32']))
        lightning = {kind: presets_of(kind, None)['lightning'] for kind in PRESETS}
        self.assertEqual(lightning, dict(bubble=dict(simulations=16, solver_nodes=1024), drip=dict(ms=100),
                                         seal=dict(ms=100), six=dict(nodes=120), strix=dict(simulations=2)))
        self.assertEqual(presets_of('bubble', None, 'cpu')['lightning'], dict(simulations=4, solver_nodes=512))
        for device in ('cuda', 'cpu'):
            ladder = [presets_of('bubble', None, device)[name] for name in PRESET_NAMES]
            for low, high in zip(ladder, ladder[1:]):
                self.assertLess(low['simulations'], high['simulations'])
                self.assertLess(low['solver_nodes'], high['solver_nodes'])
        self.assertEqual([list(PRESETS[kind]) for kind in PRESETS], [['lightning', 'quick', 'standard', 'strong', 'deep', 'dangerous']] * 5)
        self.assertEqual(presets_of('six', dict(quick=dict(nodes=1)))['lightning'], dict(nodes=120))
        self.assertEqual({kind: presets_of(kind, None)['dangerous'] for kind in PRESETS},
                         dict(bubble=dict(simulations=65536, solver_nodes=4_000_000), drip=dict(ms=60000),
                              seal=dict(ms=60000), six=dict(nodes=2_000_000), strix=dict(simulations=4096)))
        self.assertEqual(budget_of(PRESETS['bubble'], 'custom', dict(simulations=100_000, solver_nodes=10_000_000)),
                         dict(simulations=100_000, solver_nodes=10_000_000))
        with self.assertRaisesRegex(ValueError, 'from 0'):
            budget_of(PRESETS['bubble'], 'custom', dict(solver_nodes=-1))
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


class GameGraphs(unittest.TestCase):
    """A seat's game graph (Engines.game_graph) on the hybrid scheduler with a tiny network."""

    def setUp(self):
        import hexnet
        import play
        import torch
        from play import Bubble
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / 'ema.pt'
        torch.manual_seed(0)
        hexnet.save_model(path, hexnet.HexNet(hexnet.HexNetConfig(
            blocks=1, channels=8, pool_every=1, line_length=5, value_hidden=8, head_channels=4)))
        self.engines = Engines('cpu', tactical_package=Path(RUN.name) / 'missing')
        self.addCleanup(self.engines.close)
        tiny = Bubble(path, 'cpu')
        self.engines.bubble = lambda path, device=None: tiny
        self.seen, search = [], play.search

        def counted(bubble, roots, *args, **options):
            self.seen.extend(int(graph.result(0, 0, 0, 0)['visits'].sum()) for graph, _ in roots)
            return search(bubble, roots, *args, **options)
        patch = unittest.mock.patch.object(play, 'search', counted)
        patch.start()
        self.addCleanup(patch.stop)

    def turn(self, history, line, simulations=128):
        """Bubble's turn at `history` on `line`; the visits each stone's search started from are in `seen`."""
        self.seen.clear()
        entry = dict(kind='bubble', path=Path(RUN.name))
        found, _, weights = self.engines.evaluate(entry, '', dict(simulations=simulations, solver_nodes=0), history,
                                                  lambda n: None, line=line)
        self.assertTrue(weights.endswith(':kept'))
        return [*history, *map(tuple, found['moves'])]

    def graph(self, line):
        return self.engines.graphs[('seat', line)][1]

    def test_defender_proof_uses_refuted_graph_edges_propagates_and_reloads(self):
        from play import evaluate, Proofs
        from neural_search import checked, native
        from tactical_proof import NativeTactics
        from tests.test_tactical_proof import OPEN_THREE
        prover = NativeTactics()
        history = OPEN_THREE + [[-1, 0], [2, 1]]
        cold = prover.history(history, attacker='defender', nodes=10000, ms=5000)
        self.assertEqual(cold['status'], 'PROVEN_LOSS')
        replies = cold['certificate']['nodes'][0]['responses']
        bubble = self.engines.bubble(None)
        trees = self.engines.game_graph(bubble, 'defender')
        graph, _ = trees(history[:-1], 32, bubble.evaluator)
        # Start from a graph with exactly the imported facts. No local tactical
        # classification should fill in its hundreds of still unknown edges.
        checked(native.hxg_tactics(graph.ptr, 0))
        graph.expand()
        graph.at(history)
        graph.expand()
        # Exactly the four cover first stones are refuted. Hundreds of other
        # root edges remain unresolved, so the graph alone cannot settle it.
        for first in {tuple(p) for r in replies for p in r['action']}:
            graph.at(history+[list(first)])
            graph.prove_loss(0, 23)
        graph.at(history)
        self.assertEqual(graph.result(0, 0, 0, 0)['proven'], 0)
        self.assertEqual(len(graph.facts()), 4)
        from types import SimpleNamespace
        found = evaluate(bubble, SimpleNamespace(history=prover.history, abort=prover.cancel), history, 32, 1, trees=trees)
        self.assertEqual((found['proof']['winner'], found['value']), (0, 0.))
        self.assertTrue(found['proof']['dependencies'])
        checked_proof = prover.history(history, attacker='defender', certificate=found['proof']['certificate'],
                                      known=[d['outcome'] for d in found['proof']['dependencies']], nodes=1, ms=1000)
        self.assertEqual(checked_proof['status'], 'PROVEN_LOSS')
        graph.at(history[:-1])
        self.assertEqual(graph.result(0, 0, 0, 0)['proven'], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'proofs.jsonl'
            store = Evaluations(path)
            store.add(history, 'test', dict(simulations=32, solver_nodes=1), found)
            restored = Evaluations(path)
            known = Proofs()
            known.extend(restored, history)
            self.assertEqual(known.known(history)['winner'], 0)
            self.assertEqual(known.known(history[:-1])['winner'], 0)

    def test_fresh_losing_roots_still_play_in_single_and_pooled_evaluations(self):
        from play import evaluate, evaluate_many, Proofs, replay
        from tactical_proof import NativeTactics
        from tests.test_tactical_proof import OPEN_THREE
        from types import SimpleNamespace
        engine = NativeTactics()
        prover = SimpleNamespace(history=engine.history, abort=engine.cancel)
        history = OPEN_THREE + [[-1, 0], [2, 1]]
        bubble = self.engines.bubble(None)
        single = evaluate(bubble, prover, history, 16, 10000)
        pooled = evaluate_many(bubble, [prover], [history], 16, 10000)[0]
        known = Proofs()
        known.add(history, single)
        restored = evaluate(bubble, None, history, 16, 0, known=known)
        for found in (single, pooled, restored):
            self.assertEqual((found['proof']['winner'], found['value']), (0, 0.))
            self.assertEqual(len(found['moves']), 2)
            game = replay(history)
            try:
                for move in found['moves']:
                    game.play(*move)
            finally:
                game.close()

    def test_the_turn_leaves_visits_below_it_for_the_next_turn(self):
        history = self.turn([(0, 0)], 1, 1024)
        self.assertEqual(self.seen[0], 0)
        graph = self.graph(1)
        graph.at(history)
        # The searches of the turn's stones reached the position after it: the reply starts from those visits.
        self.assertGreater(int(graph.result(0, 0, 0, 0)['visits'].sum()), 0)
        history += map(tuple, legal_turn(history))
        self.turn(history, 1)
        self.assertIs(self.graph(1), graph)

    def test_a_line_returns_to_earlier_positions_and_a_new_line_starts_afresh(self):
        history = self.turn([(0, 0)], 1)
        history += map(tuple, legal_turn(history))
        graph = self.graph(1)
        self.turn(history[:1], 1)
        self.assertGreater(self.seen[0], 0)
        self.assertIs(self.graph(1), graph)
        self.turn(history, 2)
        self.assertEqual(self.seen[0], 0)
        self.turn([(0, 0)], 3)
        self.turn([(0, 0)], 4)
        self.assertEqual(list(self.engines.graphs), [('seat', 2), ('seat', 3), ('seat', 4)])
        self.assertIsNone(graph.ptr)

    def test_live_values_belong_to_the_requested_root(self):
        from play import replay
        history = [(0, 0), (1, 0), (1, 1), (-1, 0)]
        for length in (1, 2, 3, 4):
            root = history[:length]
            with self.subTest(length=length):
                live = []
                found, _, _ = self.engines.evaluate(dict(kind='bubble', path=Path(RUN.name)), '',
                    dict(simulations=2048, solver_nodes=0), root, lambda n: None, live=live.append, line=length)
                self.assertTrue(live)
                game = replay(root)
                try:
                    for glimpse in live:
                        self.assertTrue(0 <= glimpse['value'] <= 1)
                        self.assertTrue(all(game.legal(*row[:2]) for row in glimpse['top']))
                finally:
                    game.close()
                self.assertTrue(found['moves'])


CHAMPION = Path(os.environ.get('HEXO_RUN', Path(__file__).resolve().parents[1] / 'runs' / 'dense-v1')) / \
    'checkpoints' / 'main' / '185000' / 'ema.pt'


@unittest.skipUnless(CHAMPION.exists(), 'needs the main/185000 checkpoint of runs/dense-v1 (or of the run HEXO_RUN names)')
@slow
class GraphAnalysis(unittest.TestCase):
    """Analysis on one game graph with the real network on CPU. At A, yellow to move, a deep fresh search plays
    [-2, 0] then [1, 0] at 87 percent; the position C after that turn is about even once searched as a root."""

    def setUp(self):
        self.a = import_history('version[1];\n1. [4,0][7,0];\n2. [0,-1][0,-2];\n3. [0,-3][6,0];\n4. [-1,0][5,0];\n'
                                '5. [-2,1][-1,-1];')
        self.c = [*self.a, (-2, 0), (1, 0)]
        self.engines = Engines('cpu', tactical_package=Path(RUN.name) / 'missing')
        self.addCleanup(self.engines.close)
        self.entry = dict(kind='bubble', path=CHAMPION.parents[3])

    def evaluate(self, history, simulations, refresh=None):
        return self.engines.evaluate(self.entry, 'main/185000', dict(simulations=simulations, solver_nodes=0), history,
                                     lambda n: None, game=1, **({'refresh': refresh} if refresh else {}))[0]

    def turn(self):
        """{stone: (completed Q as yellow's win chance, visits)} of the turn's two stones at A, and A's visits."""
        graph = self.engines.graphs[1][1]
        graph.at(self.a)
        stats = graph.result(0, 0, 0, 0)
        found = {stone: ((stats['completed_q'][i] + 1) / 2, int(stats['visits'][i]))
                 for stone in ((-2, 0), (1, 0)) for i in [stats['actions'].tolist().index(list(stone))]}
        return found, int(stats['visits'].sum())

    def test_a_search_after_the_turn_lowers_the_turn_start(self):
        deep = self.evaluate(self.c, 512)
        self.assertLess(deep['value'], .55)
        self.assertLess(self.evaluate(self.a, 256)['value'], .8)
        for value, _ in self.turn()[0].values():
            self.assertLess(value, .7)   # 87 percent before; C's value reaches A through the visits A gives it

    def test_stepping_back_after_the_turn_redoes_the_turn_start(self):
        # A, then C two stones on, then back at A: A's turn stones now carry C's search, and A's own search goes on
        # from its counts instead of starting again.
        first = self.evaluate(self.a, 256)
        before, visits = self.turn()
        self.evaluate(self.c, 512)
        after, _ = self.turn()
        again = self.evaluate(self.a, 512, refresh=dict(first, simulations=512, solver_nodes=0))   # as saved
        resumed, total = self.turn()
        self.assertGreater(total, visits + 128)
        for stone in before:
            self.assertLess(after[stone][0], .65)
            self.assertGreaterEqual(resumed[stone][1], after[stone][1])
        self.assertNotIn(tuple(again['moves'][0]), before)
        shares = {tuple(row[:2]): row[2] for row in again['top']}
        self.assertTrue(all(shares.get(stone, 0.) < .5 for stone in before))


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
                state = self.post('/match', dict(players=['Drip', 'Drip'], games=2,
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
                           ('/seat', [1]), ('/seat', dict(side=1, engine='drip:Drip', preset='custom', custom=[]))):
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
        registry['drip:Other'] = registry['drip:Drip'] | dict(id='drip:Other', name='Other')
        self.session = Session(registry, self.engines, Evaluations())
        self.session.configure_analysis('bubble:fake', auto=False)
        self.addCleanup(self.session.close)
        self.output = Path(self.directory.name) / 'match'

    def test_real_winners_are_counted_for_the_right_bot_after_swapping(self):
        self.session.archive = Path(self.directory.name) / 'archive'
        opening = [(0, 0), (0, 5), (1, 5), (5, 0), (-5, 0), (2, 5), (3, 5), (0, -5), (0, -6),
                   (4, 5), (-2, 5), (0, -7), (0, -8)]
        self.engines.turn = lambda *args: [[5, 5]]
        self.session.start_match(['Drip', 'Other'], output=self.output, openings=[opening], max_placements=0)
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
        self.assertEqual(len(self.session.match_catalogue()), 1)
        self.assertFalse(recovered['single'])

    def test_pause_holds_the_next_game_and_settings_cannot_change_mid_match(self):
        self.session.start_match(['Drip', 'Other'], output=self.output, max_placements=3)
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
        self.session.start_match(['Drip', 'Other'], output=self.output)
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
        self.session.start_match(['Drip', 'Other'], output=self.output, book=path,
                                 opening_range='narrow', unique_openings=2, max_placements=5)
        wait(lambda: not self.session.match_worker.is_alive())
        self.assertEqual(self.session.match['games'], 4)
        games = [json.loads(p.read_text()) for p in sorted(self.output.glob('game-*.json'))]
        keys = [g['opening']['key'] for g in games]
        self.assertEqual(keys, [keys[0], keys[0], keys[2], keys[2]])
        self.assertNotEqual(keys[0], keys[2])

    def test_bad_match_specifications_do_not_change_the_board_or_start_jobs(self):
        for players, kwargs in [(['fake', 'Drip'], {}), (['Drip', 'Other'], dict(games=0)),
                                (['Drip', 'Other'], dict(unique_openings=2, games=2)),
                                (['Drip', 'Other'], dict(max_placements=1))]:
            with self.assertRaises(ValueError):
                self.session.start_match(players, output=self.output, **kwargs)
        self.assertIsNone(self.session.match)
        self.assertEqual(self.session.history, [])
        self.assertFalse(self.output.exists())

    def test_every_explicit_opening_gets_a_pair_by_default(self):
        openings = [[[0, 0]], [[0, 0], [1, 0], [2, 0]]]
        self.session.start_match(['Drip', 'Other'], output=self.output, openings=openings, max_placements=5)
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
        self.session.configure_seat(0, 'drip:Drip')
        self.session.configure_seat(1, 'drip:Other')
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

    def test_a_page_names_the_people_for_book_openings(self):
        path = Path(self.directory.name) / 'book.json'
        nodes = [dict(key=f'0,0|{i},0 {i},1', status='opening', moves=[[0, 0], [i, 0], [i, 1]]) for i in (1, 2)]
        path.write_text(json.dumps(dict(schema='hexo-opening-book-v2', nodes=nodes)))
        self.session.book = path
        self.session.pause(True)
        self.session.configure_seat(0, 'human')
        self.session.configure_seat(1, 'human')
        self.session.use_book(False)
        self.session.use_book(True, 'wide', [1])
        self.assertEqual([side for (_, _, side), n in self.session.coverage.counts.items() if n], [1])
        self.session.new_game(people=[])
        self.assertEqual(sum(self.session.coverage.counts.values()), 1)
        with self.assertRaises(ValueError):
            self.session.new_game(people=[2])

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
        self.session.start_match(['Drip', 'Other'], output=self.output, max_placements=3)
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

    def test_a_batch_saved_under_older_engine_ids_resumes_with_drip(self):
        self.session.start_match(['Drip', 'Other'], output=self.output, max_placements=3)
        wait(lambda: self.session.match['completed'] == 1)
        self.session.stop_match()
        wait(lambda: not self.session.match_worker.is_alive())
        summary = self.output / 'summary.json'
        older = json.loads(summary.read_text(encoding='utf-8').replace('"drip:Drip"', '"native:Native"').replace(
            '"Drip"', '"Native"').replace('"drip"', '"native"'))
        older['players'][0]['source']['badge'] = 'native'
        self.assertNotIn('Drip', json.dumps(older))
        summary.write_text(json.dumps(older), encoding='utf-8')
        self.session.resume_match(self.output)
        wait(lambda: not self.session.match_worker.is_alive())
        self.assertEqual(self.session.match['completed'], 2)
        self.assertEqual([(p['engine'], p['name'], p['source']['kind']) for p in self.session.match['players']],
                         [('drip:Drip', 'Drip', 'drip'), ('drip:Other', 'Other', 'drip')])
        self.assertEqual(self.session.entries['drip:Drip']['badge'], 'drip')

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
        self.session.start_match(['Drip', 'Other'], output=self.output, max_placements=3)
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
        self.assertEqual(next(m for m in reopened.match_catalogue() if m['id'] == ident)['completed'], 2)
        with unittest.mock.patch('play.Engines', return_value=FakeEngines()):
            study = reopened.open_saved_game(ident, 1)
        self.assertIsNotNone(study.lookup([(0, 0)]))
        self.assertEqual((self.output / 'game-0001.json').read_bytes(), saved)

    def test_freeplay_autosaves_moves_new_games_and_analysis_and_preserves_studies(self):
        self.session.configure_seat(1, 'human')
        self.session.archive = Path(self.directory.name) / 'archive'
        with self.session.lock:
            self.session.changed()
        first = self.session.match_catalogue()[0]['id']
        created = self.session.match_catalogue()[0]['created_at']
        self.session.play(0, 0)
        self.session.play(1, 0)
        directory, game = self.session.saved_replay(first, 1)
        self.assertEqual(game['history'], [[0, 0], [1, 0]])
        self.assertEqual(game['reason'], 'saved')
        self.session.configure_analysis('bubble:fake', preset='custom', custom=dict(simulations=1, solver_nodes=0), auto=False)
        self.session.analyse(1)
        wait(lambda: self.session.lookup([(0, 0)]) is not None and
             '"value":0.5' in (directory / 'evaluations.jsonl').read_text())
        original = (directory / 'game-0001.json').read_bytes()
        self.assertEqual(self.session.match_catalogue()[0]['created_at'], created)
        evaluations = (directory / 'evaluations.jsonl').read_bytes()
        self.session.configure_analysis('bubble:fake', checkpoint='main/000001', auto=False)
        self.assertEqual((directory / 'evaluations.jsonl').read_bytes(), evaluations)
        self.session.new_game()
        self.assertEqual(len(self.session.match_catalogue()), 2)
        reopened = Session(entries(), FakeEngines(), Evaluations(), archive=self.session.archive, save_initial=False)
        self.addCleanup(reopened.close)
        with unittest.mock.patch('play.Engines', return_value=FakeEngines()):
            study = reopened.open_saved_game(first, 1)
        self.assertIsNotNone(study.lookup([(0, 0)]))
        self.assertEqual(len(reopened.match_catalogue()), 2)
        study.play(2, 0)
        self.assertEqual(len(reopened.match_catalogue()), 3)
        self.assertEqual((directory / 'game-0001.json').read_bytes(), original)
        self.assertTrue(all(m['kind'] == 'freeplay' for m in reopened.match_catalogue()))

    def test_timed_bubble_uses_the_selected_tactical_package(self):
        package = Path(self.directory.name) / 'solver'
        self.engines.tactical_package = package
        self.engines.proof_stamps = True
        seat = self.session.match_seat('bubble:2@quick', 'standard')
        config = self.session.timed_config(seat)
        self.assertEqual(config['tactical_package'], str(package))
        self.assertTrue(config['solver']['stamps'])
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
        self.engines.device = 'cuda'   # a CPU seat on a GPU server plays the CPU ladder
        self.assertEqual(self.session.match_seat(dict(engine='bubble:2@quick', device='cpu'), 'standard')['budget'],
                         CPU_PRESETS['quick'])
        del self.engines.device
        self.assertEqual(self.session.match_seat('Drip@lightning', 'standard')['budget'], dict(ms=100))
        custom = self.session.match_seat('bubble:2{simulations=512,views=4}', 'standard')
        self.assertEqual(custom['budget'], dict(simulations=512, views=4, solver_nodes=8192))
        for selector in ('bubble:2{ms=2500}', 'bubble:2{simulations=64,ms=2500,active=ms}'):
            self.assertEqual(self.session.match_seat(selector, 'standard')['budget']['ms'], 2500)
        with self.assertRaisesRegex(ValueError, 'not a budget'):
            self.session.match_seat('Drip{simulations=128}', 'standard')

    def test_clock_deadline_discards_late_moves_and_records_time_result(self):
        late_reply = lambda game, *a, **kw: (
            time.sleep(.06) or dict(moves=legal_turn([c[:2] for c in game.cells]), elapsed_ms=60))
        for name, reply in [('late', late_reply), ('timeout', TimeoutError('Opponent exceeded its allowance'))]:
            with self.subTest(name=name), unittest.mock.patch('timed_engine.TimedEngine') as engine:
                engine.return_value.identity = dict(kind='fake')
                engine.return_value.turn.side_effect = reply
                output = self.output / name
                self.session.start_match(['Drip', 'Other'], games=1, output=output,
                                         clock=dict(mode='move', ms=20))
                wait(lambda: not self.session.match_worker.is_alive())
                result = json.loads((output / 'game-0001.json').read_text())
                self.assertEqual((result['winner'], result['reason'], result['history']), (0, 'time', [[0, 0]]))
                self.assertEqual((result['turns'][-1]['side'], result['turns'][-1]['circle_ms'], result['turns'][-1]['cross_ms']), (1, 0, 20))
                ident = next(m['id'] for m in self.session.match_catalogue() if m['name'] == name)
                opened = self.session.open_saved_game(ident, 1).state()['clock']
                self.assertEqual((opened['cross_ms'], opened['circle_ms']), (20, 0))
                self.assertEqual(self.session.match['wins'], [1, 0])
                self.assertIsNone(self.session.match['error'])

    def test_a_batch_cannot_take_over_an_unfinished_human_game(self):
        self.session.configure_seat(1, 'human')
        self.session.play(0, 0)
        with self.assertRaisesRegex(ValueError, 'human game'):
            self.session.start_match(['Drip', 'Other'], output=self.output)
        self.assertEqual(self.session.history, [(0, 0)])
        self.session.start_match(['Drip', 'Other'], games=2, output=self.output, max_placements=3, replace=True)
        wait(lambda: not self.session.match_worker.is_alive())
        self.assertEqual(self.session.match['completed'], 2)

    def test_simulations_only_adapter_refuses_a_clock_before_start(self):
        self.session.entries['strix:Strix'] = dict(id='strix:Strix', name='Strix', kind='strix',
            presets=PRESETS['strix'], model=Path(RUN.name) / 'checkpoints/main/000001/ema.pt')
        with self.assertRaisesRegex(ValueError, 'cannot keep a clock'):
            self.session.start_match(['Drip', 'Strix'], output=self.output, clock=dict(mode='game', tc='180+2'))
        self.assertFalse(self.output.exists())


class FreeplayClock(unittest.TestCase):
    """A single game on a clock: balances per side, the increment after a complete turn, a loss on time and the turn
    log in the saved game."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.session = Session(entries(), FakeEngines(), Evaluations(), archive=Path(self.directory.name))
        self.addCleanup(self.session.close)
        self.session.configure_analysis('bubble:fake', auto=False)
        self.session.configure_seat(1, 'human')

    def test_people_play_on_a_fischer_clock_and_lose_on_time(self):
        self.session.set_clock(dict(mode='game', tc='0.4+1'))
        state = self.session.state()
        self.assertEqual((state['clock_spec'], state['clock']['running']), (dict(mode='game', base_ms=400., increment_ms=1000.), 'x'))
        self.session.play(0, 0)
        state = self.session.state()
        self.assertGreater(state['clock']['cross_ms'], 1000)
        self.assertEqual(state['clock']['running'], 'o')
        self.session.play(1, 0)
        self.assertEqual(self.session.state()['clock']['running'], 'o')
        time.sleep(.5)
        state = self.session.state()
        self.assertEqual((state['winner'], state['outcome']), (0, dict(winner=0, reason='time')))
        with self.assertRaisesRegex(ValueError, 'finished'):
            self.session.play(2, 0)
        saved = json.loads(next(Path(self.directory.name).glob('*-freeplay-*/game-0001.json')).read_text())
        self.assertEqual((saved['winner'], saved['reason'], saved['clock']['increment_ms']), (0, 'time', 1000.))
        self.assertEqual(saved['turns'][-1]['circle_ms'], 0)
        self.assertGreater(saved['turns'][-1]['cross_ms'], 1000)
        self.assertEqual([t['side'] for t in saved['turns']], [0, 1])
        ident = next(m['id'] for m in self.session.match_catalogue() if m['results'][0]['reason'] == 'time')
        study = self.session.open_saved_game(ident, 1)
        opened = study.state()
        self.assertEqual((opened['winner'], opened['outcome']['reason'], opened['clock_spec']['increment_ms']), (0, 'time', 1000.))
        self.assertEqual((opened['clock']['circle_ms'], opened['clock']['running']), (0, None))
        self.assertEqual(len(study.clock_turns), 2)
        self.session.new_game()
        state = self.session.state()
        self.assertEqual((state['winner'], state['outcome'], state['clock']['running']), (-1, None, 'x'))
        self.assertGreater(state['clock']['cross_ms'], 300)

    def test_a_paused_turn_keeps_its_time_and_a_failed_engine_stops_the_clock(self):
        self.session.set_clock(dict(mode='game', tc='60'))
        time.sleep(.2)
        self.session.pause(True)
        self.session.pause(False)
        time.sleep(.1)
        self.session.play(0, 0)
        self.assertGreaterEqual(self.session.clock_turns[0]['spent_ms'], 290)
        class Broken:
            def turn(self, *args, **kwargs):
                raise RuntimeError('engine broke')

            def close(self):
                pass
        self.session.prepare_timed = lambda sides=(0, 1): setattr(self.session, 'seat_engines', [None, Broken()])
        self.session.configure_seat(1, 'drip:Drip')
        wait(lambda: self.session.paused)
        before = self.session.state()['clock']
        time.sleep(.2)
        after = self.session.state()
        self.assertEqual((after['clock']['running'], after['clock']['circle_ms'], after['outcome']), (None, before['circle_ms'], None))

    def test_changing_one_seat_keeps_the_other_seats_timed_engine(self):
        with unittest.mock.patch('timed_engine.TimedEngine') as engine:
            engine.side_effect = lambda config: unittest.mock.MagicMock(name=config['kind'])
            self.session.configure_seat(0, 'drip:Drip')
            self.session.configure_seat(1, 'drip:Drip')
            self.session.pause(True)
            self.session.set_clock(dict(mode='game', tc='60'))
            wait(lambda: self.session.clock_preparing is None and all(self.session.seat_engines))
            kept = self.session.seat_engines[1]
            with self.session.lock:
                self.session.prepare_timed([0])
            wait(lambda: self.session.clock_preparing is None and self.session.seat_engines[0] is not None)
            self.assertIs(self.session.seat_engines[1], kept)
            kept.close.assert_not_called()

    def test_a_timed_bubble_move_is_kept_as_an_evaluation(self):
        class Timed:
            def turn(self, game, ms=None, clock=None, cancel=None, publish=None):
                moves = legal_turn([cell[:2] for cell in game.cells])
                return dict(moves=moves, win_probability=.62, completed=40, solver_nodes=0)

            def close(self):
                pass
        self.session.prepare_timed = lambda sides=(0, 1): setattr(self.session, 'seat_engines', [None, Timed()])
        self.session.configure_seat(1, 'bubble:fake')
        self.session.set_clock(dict(mode='game', tc='60'))
        self.session.play(0, 0)
        wait(lambda: len(self.session.history) == 3)
        found = self.session.state()['evaluations'][1]
        self.assertEqual((found['value'], found['simulations'], found['moves']), (.62, 40, [list(p) for p in self.session.history[1:]]))

    def test_leaving_a_match_rebuilds_the_seats_timed_engines(self):
        rebuilt = []
        self.session.prepare_timed = lambda sides=(0, 1): rebuilt.append(tuple(sides))
        self.session.match = dict(active=False, clock=dict(mode='fixed'))
        self.session.load([(0, 0)], True)
        self.assertEqual((self.session.match, rebuilt), (None, [(0, 1)]))
        self.session.load([], True)
        self.assertEqual(rebuilt, [(0, 1)])

    def test_engines_report_where_the_server_runs_them(self):
        from play import device_of
        self.assertEqual([device_of(dict(kind='bubble'), 'cuda'), device_of(dict(kind='bubble'), 'cpu'),
                          device_of(dict(kind='six', backend='TensorRT'), 'cpu'), device_of(dict(kind='six', backend='CPU'), 'cuda'),
                          device_of(dict(kind='six', badge='shrimp'), 'cuda'), device_of(dict(kind='drip'), 'cuda')],
                         ['GPU', 'CPU', 'GPU', 'CPU', 'CPU', 'CPU'])
        self.assertEqual({e['device'] for e in self.session.state()['engines']}, {'Server CPU'})

    def test_engines_that_cannot_keep_a_clock_are_refused(self):
        self.session.entries['six:shrimp'] = dict(id='six:shrimp', name='Shrimp', kind='six', badge='shrimp',
                                                  presets=PRESETS['six'], command=['shrimp'])
        self.session.configure_seat(1, 'six:shrimp')
        with self.assertRaisesRegex(ValueError, 'cannot keep a clock'):
            self.session.set_clock(dict(mode='move', ms=1000))
        self.assertEqual(self.session.state()['clock_spec'], dict(mode='fixed'))
        self.session.configure_seat(1, 'human')
        self.session.set_clock(dict(mode='move', ms=1000))
        with self.assertRaisesRegex(ValueError, 'cannot keep a clock'):
            self.session.configure_seat(1, 'six:shrimp')
        self.assertEqual([e['clocks'] for e in self.session.models() if e['id'] in ('six:shrimp', 'drip:Drip')], [True, False])


class Proofs(unittest.TestCase):
    def test_padded_proof_bound_does_not_index_a_finished_board(self):
        from play import Proofs
        from tactical_proof import NativeTactics
        from tests.test_tactical_proof import IMMEDIATE
        history = IMMEDIATE
        table = Proofs()
        table.add(history, dict(proof=dict(winner=0, plies=6), pv=[[5,0,0,1]]))
        self.assertEqual(len(table.facts(history)), 1)
        self.assertEqual(table.facts(history)[0]['history'], history)
        self.assertEqual(table.known(history)['plies'], 6)
        result = NativeTactics().history(history, known=[
            {k:f[k] for k in ('history', 'winner', 'plies')} for f in table.facts(history)], nodes=1, ms=1000)
        self.assertEqual(result['status'], 'PROVEN_WIN', result['reason'])

    def test_tighter_scalar_proof_does_not_suggest_half_a_turn(self):
        from play import Proofs
        from types import SimpleNamespace
        table, root = Proofs(), [(0,0)]
        old = evaluation(1.,moves=[[2,0],[3,0]],proof=dict(winner=1,plies=10,turns=3),pv=[])
        table.add(root,old)
        table.add(root+[(1,0)],dict(proof=dict(winner=1,plies=5),pv=[]))
        shown = Session.proven(SimpleNamespace(proofs=table),root,old)
        self.assertEqual((shown['proof']['plies'],shown['moves'],shown['pv']),(6,[],[[1,0,1,1]]))
        table.add(root,dict(proof=dict(winner=1,plies=1),pv=[[1,0,1,1]]))
        self.assertEqual(Session.proven(SimpleNamespace(proofs=table),root,old)['moves'],[[1,0]])

    def test_shorter_child_replaces_an_existing_root_proof_and_saved_line(self):
        from play import Proofs
        from types import SimpleNamespace
        root = [(0,0),(1,-2),(-1,-2),(0,-2),(2,-1),(-2,-3),(3,-3),(4,-2),(6,-3),
                (1,-4),(5,-4),(8,-4),(10,-5),(7,-5),(6,-6),(12,-6),(14,-7),(9,-6),
                (11,-7),(15,-10),(17,-11),(13,-8),(14,-11),(15,-9),(16,-10),(17,-10),(13,-7)]
        old = evaluation(1., moves=[[0,-1],[0,1]], proof=dict(winner=0,plies=42,turns=11),
                         pv=[[0,-1,0,1],[0,1,0,2]])
        child = evaluation(1., proof=dict(winner=0,plies=29,turns=8), pv=[[19,-12,0,1]])
        for records in ([(root,old),(root+[(18,-12)],child)], [(root+[(18,-12)],child),(root,old)]):
            table = Proofs()
            for history, record in records:
                table.add(history, record)
            expected = dict(winner=0,plies=30,pv=[[18,-12,0,1],[19,-12,0,2]])
            self.assertEqual(table.known(root), expected)
            shown = Session.proven(SimpleNamespace(proofs=table), root, old)
            self.assertEqual(shown['proof'], dict(winner=0,plies=30,turns=8))
            self.assertEqual(shown['moves'], [[18,-12],[19,-12]])
            self.assertEqual(shown['pv'], expected['pv'])
            self.assertEqual(old['proof']['plies'], 42)

    def test_defender_line_keeps_longest_covered_reply_without_tightening_from_a_subset(self):
        from play import Proofs
        root = [(0,0),(1,0),(2,0)]
        table = Proofs()
        old = dict(proof=dict(winner=1,plies=12), pv=[[0,1,0,1],[0,2,0,2],[3,0,1,3]])
        table.add(root, old)
        table.add(root+[(0,1)], dict(proof=dict(winner=1,plies=3),pv=[[0,2,0,1],[3,0,1,2]]))
        table.add(root+[(1,1)], dict(proof=dict(winner=1,plies=7),pv=[[1,2,0,1],[3,0,1,2]]))
        outcome = table.known(root)
        self.assertEqual(outcome['plies'], 12)
        self.assertEqual(outcome['pv'][0], [1,1,0,1])
        from types import SimpleNamespace
        saved = dict(old, moves=[[0,1],[0,2]], top=[[0,1,1,.1,0]])
        shown = Session.proven(SimpleNamespace(proofs=table), root, saved)
        self.assertEqual(shown['moves'], [[1,1],[1,2]])
        self.assertEqual(shown['top'][0], [1,1,0.,0.,-1])
        self.assertEqual(saved['moves'], [[0,1],[0,2]])
        # An unrelated slower upper bound cannot weaken the existing root guarantee.
        table.add(root+[(2,1)], dict(proof=dict(winner=1,plies=15),pv=[[2,2,0,1]]))
        self.assertEqual(table.known(root), outcome)

    def test_losing_half_turn_covers_both_orders_without_inventing_a_winning_line(self):
        from play import Proofs, answered
        root = [(0,0),(1,2),(2,2),(0,-2),(-2,0),(3,2),(4,2)]
        a, b = (0,-3), (-3,0)
        table = Proofs()
        table.add(root+[a], dict(proof=dict(winner=1, plies=3), pv=[[2,-3,0,1]]))
        self.assertEqual(table.edges(root+[b])[a], (1,3,dict(winner=1,plies=2,pv=[])))
        self.assertIn(dict(history=[list(p) for p in root+[b,a]],winner=1,plies=2,pv=[]),table.facts(root+[b]))
        self.assertIsNone(table.known(root+[b]))
        for pair in ([a,b],[b,a]):
            self.assertEqual(table.known(root+pair), dict(winner=1,plies=2,pv=[]))
            self.assertIsNone(answered(root+pair, table))
        # A winning half-turn is existential, so it cannot refute every second stone.
        winning = Proofs()
        winning.add(root+[a], dict(proof=dict(winner=0, plies=1)))
        self.assertNotIn(a, winning.edges(root+[b]))
        self.assertIsNone(winning.known(root+[b,a]))

    def test_reordered_earlier_turn_reaches_known_win_with_the_remaining_stones(self):
        from play import Proofs, answered
        root = [(0,0),(8,0),(0,8),(1,0),(2,0),(-8,0),(0,-8),(0,3),(1,3),(8,-8),(-8,8)]
        a,b,c,d,e,f = (3,0),(2,3),(8,1),(8,2),(3,3),(10,0)
        table = Proofs()
        table.add(root+[a,b,c,d,e,f], dict(proof=dict(winner=0, plies=4)))
        result = answered(root+[a,e,c,d], table)
        self.assertEqual({tuple(p) for p in result['moves']}, {b,f})
        self.assertEqual((result['value'], result['proof']['plies'], result['actual_completed'], result['actual_solver_nodes']), (1.,6,0,0))

    def test_placements_become_the_winners_turns(self):
        self.assertEqual([proof_turns(p, 2, True) for p in (1, 2, 5, 6, 9, 10)], [1, 1, 2, 2, 3, 3])
        self.assertEqual([proof_turns(p, 1, True) for p in (1, 4, 5)], [1, 2, 2])
        self.assertEqual([proof_turns(p, 2, False) for p in (3, 4, 7, 8)], [1, 1, 2, 2])

    def test_later_witness_fills_saved_lines_back_through_the_turn(self):
        from play import Proofs
        from types import SimpleNamespace
        history = [[0, 0], [1, 0], [2, 0]]
        pv = [[1, 0, 1, 1], [2, 0, 1, 2], [2, 3, 0, 3], [3, 3, 0, 4], [3, 0, 1, 5], [4, 0, 1, 6]]
        records = [(history[:1], evaluation(1., proof=dict(winner=1, plies=6), pv=pv[:2])),
                   (history[:2], evaluation(1., proof=dict(winner=1, plies=5), pv=[[2, 0, 1, 1]])),
                   (history, evaluation(0., proof=dict(winner=1, plies=4), pv=[[*p[:3], p[3] - 2] for p in pv[2:]]))]
        for order in (records, records[::-1]):
            table = Proofs()
            for prefix, record in order:
                table.add(prefix, record)
            session = SimpleNamespace(proofs=table)
            for ply, (prefix, saved) in enumerate(records, 1):
                expected = [[*p[:3], p[3] - ply + 1] for p in pv[ply - 1:]]
                self.assertEqual(table.known(prefix)['pv'], expected)
                shown = Session.proven(session, prefix, saved)
                self.assertEqual((shown['pv'], shown['proof']), (expected, saved['proof']))
            self.assertEqual(len(records[0][1]['pv']), 2)

    def test_longer_witness_survives_an_older_partial_record(self):
        from play import Proofs
        table = Proofs()
        pv = [[1, 0, 1, 1], [2, 0, 1, 2]]
        for line in (pv[:1], pv, pv[:1]):
            table.add([[0, 0]], evaluation(1., proof=dict(winner=1, plies=6), pv=line))
        self.assertEqual(table.known([[0, 0]])['pv'], pv)

    def test_line_extension_keeps_the_winner_and_proof_bound(self):
        from play import Proofs
        history, pv = [[0, 0]], [[1, 0, 1, 1]]
        table = Proofs()
        table.add(history + [[1, 0]], evaluation(1., proof=dict(winner=1, plies=5), pv=[[2, 0, 1, 1]]))
        for winner, plies in ((1, 5), (0, 6)):
            self.assertEqual(table.line(history, dict(winner=winner, plies=plies, pv=pv)), pv)


class TurnTrees(unittest.TestCase):
    """A fixed-budget play turn searches its second stone in the tree its first stone grew."""

    def test_certificate_line_survives_a_saved_evaluation_and_reaches_the_turn_start(self):
        import tactical_proof
        from play import evaluate, Proofs
        from tests.test_tactical_proof import LATE_WIN
        history = [tuple(p) for p in LATE_WIN] + [(-1, -11)]
        from types import SimpleNamespace
        engine = tactical_proof.NativeTactics()
        found = evaluate(self.bubble(), SimpleNamespace(history=engine.history, abort=engine.cancel), history, 8, 32768)
        self.assertEqual(found['proof']['winner'], 0)
        self.assertEqual(len(found['moves']), 1)
        self.assertGreater(len(found['pv']), 5)
        saved = json.loads(json.dumps(found))
        table = Proofs()
        table.add(history, saved)
        half = table.known(history)
        root = table.known(history[:-1])
        self.assertEqual(half['pv'], saved['pv'])
        self.assertEqual(root['plies'], half['plies'] + 1)
        self.assertEqual(root['pv'], [[-1, -11, 0, 1]] + [[*p[:3], p[3] + 1] for p in saved['pv']])
        # A proven continuation is retained even if the search has not established the whole root.
        table = Proofs()
        fact = dict(history=[list(p) for p in history], winner=0, plies=saved['proof']['plies'], pv=saved['pv'])
        table.add(history[:-1], dict(proof=None, proofs=[fact]))
        self.assertEqual(table.known(history[:-1]), root)

    def setUp(self):
        import hexnet
        import play
        import torch
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)/'ema.pt'
        torch.manual_seed(0)
        hexnet.save_model(self.path, hexnet.HexNet(hexnet.HexNetConfig(
            blocks=1, channels=8, pool_every=1, line_length=5, value_hidden=8, head_channels=4)))
        self.trees = []
        trees, search = self.trees, play.search

        def spy(bubble, roots, *args, **kwargs):
            """Each searched graph in `trees`, with its searches as (root, simulations asked, completed)."""
            results = search(bubble, roots, *args, **kwargs)
            for (graph, simulations), result in zip(roots, results):
                if not any(graph is tree for tree in trees):
                    graph.searched = []
                    trees.append(graph)
                graph.searched.append((list(graph.history), simulations, result['completed']))
            return results
        patcher = unittest.mock.patch.object(play, 'search', spy)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_engine_puzzle_benchmark_checks_serialized_certificates(self):
        from notation import dumps
        from tests.test_tactical_proof import IMMEDIATE, NO_THREAT
        from tools.proof_stamps import engine_benchmark
        source, target = self.path.with_name('puzzles.txt'), self.path.with_name('results.json')
        source.write_text(dumps(IMMEDIATE)+'\n'+dumps(NO_THREAT), encoding='utf-8')
        report = engine_benchmark(source, target, self.path, simulations=4, nodes=1000, ms=5000, stamps=True)
        self.assertEqual([r['status'] for r in report['rows']], ['PROVEN_WIN', 'UNKNOWN'])
        self.assertGreater(report['rows'][0]['verified_certificates'], 0)
        self.assertGreater(report['rows'][0]['verification_ms'], 0)
        self.assertEqual(report['rows'][1]['verified_certificates'], 0)
        self.assertEqual(json.loads(target.read_text())['rows'], report['rows'])

    def test_play_evaluation_keeps_one_tree_for_the_turn(self):
        from play import evaluate
        from types import SimpleNamespace
        bubble = self.bubble()
        seen = []
        # A stopped clock: the first glimpse with statistics is shown at once and the throttle holds back the rest.
        with unittest.mock.patch('play.time', SimpleNamespace(**(vars(time) | dict(monotonic=lambda: 0.)))), \
                unittest.mock.patch.object(bubble.evaluator, 'evaluate', side_effect=AssertionError('searches use packed batches')):
            found = evaluate(bubble, None, [(0, 0)], 1024, 0, live=seen.append)
        moves = found['moves']
        self.assertEqual(len(seen), 1)
        self.assertTrue(all(len(g['top']) <= 5 and 0 <= g['value'] <= 1 and g['top'][0][2] >= g['top'][-1][2]
                            for g in seen))
        self.assertTrue(found['top'] and all(len(t) == 5 and 0 <= t[3] <= 1 and t[4] in (-1, 0, 1) for t in found['top']))
        self.assertTrue(all(len(t) == 3 for t in evaluate(bubble, None, [(0, 0)], 0, 0)['top']))
        self.assertEqual(len(self.trees), 1)
        tree = self.trees[0]
        self.assertEqual([h for h, _, _ in tree.searched], [[(0, 0)], [(0, 0), tuple(moves[0])]])
        self.assertEqual([c for _, _, c in tree.searched], [1024, 1024])
        self.assertIsNone(tree.ptr)
        self.assertEqual([len(step['history']) for step in found['later']], [2])

    def bubble(self):
        from play import Bubble
        return Bubble(self.path, 'cpu')

    def test_solver_preset_proves_puzzles_with_a_known_forced_win_within_its_clock(self):
        import re
        import tactical_proof
        from notation import loads
        from play import Bubble, SOLVER, evaluate, replay
        if not tactical_proof.library().exists():
            self.skipTest('Build tools/tactical with tools/build_tactical.py first')
        prover = tactical_proof.IsolatedTactics(package=tactical_proof.PACKAGE, priority='below_normal')
        self.addCleanup(prover.close)
        bubble = Bubble(self.path, 'cpu')
        text = (Path(__file__).parent/'fixtures'/'forced-wins.htttx').read_text(encoding='utf-8')
        for record in re.split(r'(?=version\[1\];)', text)[1:]:
            history = [tuple(p) for p in loads(record).history]
            game = replay(history)
            mover = game.player
            game.close()
            found = evaluate(bubble, prover, history, 8, SOLVER['solver_nodes'], solver_ms=SOLVER['solver_ms'])
            self.assertEqual(found['proof']['winner'], mover)
            self.assertLess(found['solver']['elapsed_ms'], SOLVER['solver_ms'])
            self.assertGreater(found['solver']['root_nodes'] + found['solver']['native_nodes'], 0)
            game = replay(history + [tuple(m) for m in found['moves']])
            self.assertTrue(game.winner == mover or game.player != mover)
            game.close()

    def test_deep_solve_reports_running_frontier_work_after_the_root_has_no_forcing_win(self):
        import tactical_proof
        from play import Bubble, evaluate
        if not tactical_proof.library().exists():
            self.skipTest('Build tools/tactical with tools/build_tactical.py first')
        prover = tactical_proof.IsolatedTactics(package=tactical_proof.PACKAGE, priority='below_normal')
        self.addCleanup(prover.close)
        frames = []
        found = evaluate(Bubble(self.path, 'cpu'), prover, [(0, 0), (1, 0), (0, 1)], 8, 2048, solver_ms=2500,
                         live=lambda seen: frames.append(seen['solver']) if 'solver' in seen else None)
        self.assertEqual(found['solver']['root'], 'no forcing win')
        ruled = [f for f in frames if f['root'] == 'no forcing win']
        self.assertGreater(len(ruled), 2)
        self.assertEqual(sorted(f['elapsed_ms'] for f in ruled), [f['elapsed_ms'] for f in ruled])
        self.assertGreater(ruled[-1]['checked'], ruled[0]['checked'])
        self.assertGreater(ruled[-1]['frontier_nodes'], ruled[0]['frontier_nodes'])

    def test_a_position_after_the_first_stone_ranks_the_second(self):
        import tactical_proof
        from play import evaluate
        history = [(0, 0), (1, 0), (1, 1), (-1, 0)]
        provers = [None]
        if tactical_proof.library().exists():
            provers.append(tactical_proof.IsolatedTactics(package=tactical_proof.PACKAGE, priority='below_normal'))
            self.addCleanup(provers[-1].close)
        for prover in provers:
            self.trees.clear()
            found = evaluate(self.bubble(), prover, history, 16, 64 if prover else 0)
            [stone] = found['moves']
            self.assertNotIn(tuple(stone), history)
            self.assertEqual(found['top'][0][:2], stone)
            self.assertTrue(all(tuple(row[:2]) not in history and 0 <= row[3] <= 1 for row in found['top']))
            self.assertTrue(0 <= found['value'] <= 1)
            self.assertEqual((found['later'], [h for h, _, _ in self.trees[0].searched]), ([], [history]))

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
                             completed_q=np.array([.1, .2]), proven=0, exact_winner=-1, proof_plies=0),
                        dict(action=[2, 0], policy=np.array([0., 1.]), actions=actions, values=np.array([0., 1.]),
                             completed_q=np.array([0., 1.]), proven=1, exact_winner=1, proof_plies=5)])
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

    def test_a_proven_continuation_settles_its_edge_before_the_search(self):
        from play import Proofs, evaluate
        known = Proofs()
        known.add([(0, 0), (1, 0)], dict(proof=dict(winner=1, turns=2, plies=5), pv=[]))
        found = evaluate(self.bubble(), None, [(0, 0)], 16, 0, known=known)
        self.assertEqual((found['moves'][0], found['proof']), ([1, 0], dict(winner=1, plies=6, turns=2)))
        self.assertEqual((found['top'][0][:2], found['top'][0][3:], found['pv'][0]), ([1, 0], [1., 1], [1, 0, 1, 1]))
        self.assertEqual(self.trees[0].searched[0][0], [(0, 0)])

    def test_the_shortest_of_several_proven_continuations_settles_the_root(self):
        from play import Proofs, evaluate
        known = Proofs()
        known.add([(0, 0), (1, 0)], dict(proof=dict(winner=1, turns=3, plies=9), pv=[]))
        known.add([(0, 0), (2, 0)], dict(proof=dict(winner=1, turns=2, plies=5), pv=[]))
        found = evaluate(self.bubble(), None, [(0, 0)], 16, 0, known=known)
        self.assertEqual((found['moves'][0], found['proof']), ([2, 0], dict(winner=1, plies=6, turns=2)))

    def test_a_lost_continuation_leaves_the_search(self):
        import play
        from play import Proofs, TurnSearch, solve
        known, bubble = Proofs(), self.bubble()
        known.add([(0, 0), (1, 0)], dict(proof=dict(winner=0, turns=1, plies=3), pv=[]))
        turn = TurnSearch(bubble, bubble.evaluator, [(0, 0)], 16, solve(None, [(0, 0)], 0), known=known)
        try:
            tree, simulations = turn.request()
            [result] = play.search(bubble, [(tree, simulations)])
        finally:
            turn.close()
        lost = result['actions'].tolist().index([1, 0])
        self.assertEqual((result['values'][lost], result['policy'][lost]), (-1., 0.))
        self.assertNotEqual(result['action'], [1, 0])

    def test_a_known_win_without_its_turn_is_not_claimed_for_another_turn(self):
        from play import Proofs, evaluate
        won, known = Proofs(), Proofs()
        won.add([(0, 0)], dict(proof=dict(winner=1, turns=3), pv=[]))
        self.assertEqual(evaluate(self.bubble(), None, [(0, 0)], 8, 0, known=won)['proof'], None)
        known.add([(0, 0), (1, 0), (2, 0)], dict(proof=dict(winner=1, turns=2, plies=7), pv=[]))
        lost = evaluate(self.bubble(), None, [(0, 0), (1, 0), (2, 0)], 8, 0, known=known)
        self.assertEqual((lost['proof'], lost['value']), (dict(winner=1, turns=2, plies=7), 0.))

    def test_two_stones_to_a_won_position_answer_without_a_search(self):
        from play import Proofs, evaluate
        known = Proofs()
        known.add([(0, 0), (1, 0), (2, 0)], dict(proof=dict(winner=1, turns=1, plies=4), pv=[[3, 0, 0, 1], [-1, 0, 0, 2]]))
        self.assertEqual(known.known([(0, 0)]), dict(winner=1, plies=6, pv=[[1, 0, 1, 1], [2, 0, 1, 2], [3, 0, 0, 3],
                                                                             [-1, 0, 0, 4]]))
        self.assertEqual(known.known([(0, 0), (1, 0), (2, 0), (3, 0)])['plies'], 3)
        found = evaluate(self.bubble(), None, [(0, 0)], 16, 0, known=known)
        self.assertEqual((found['moves'], found['value'], found['proof']), ([[1, 0], [2, 0]], 1., dict(winner=1, turns=2, plies=6)))
        self.assertEqual(self.trees, [])

    @slow
    def test_the_supplied_losing_half_turn_is_proven_and_the_frontier_adds_proofs(self):
        import tactical_proof
        from types import SimpleNamespace
        from play import evaluate, solve
        try:
            prover = tactical_proof.NativeTactics(package=tactical_proof.PACKAGE)
        except FileNotFoundError:
            self.skipTest('needs the built tactical library')
        history = [(0, 0), (4, 0), (7, 0), (-2, 0), (-1, 0), (1, 0), (6, 0), (5, 0), (-1, -1),
                   (-3, 1), (-1, 1), (-2, -1), (-4, 0), (-3, 0), (0, -1), (-2, -3), (-2, -2),
                   (-2, 1), (-2, -5), (-3, -1), (-1, -3), (-5, 1), (0, -4), (-4, 1), (-4, -1), (-5, -1)]
        frontier = evaluate(self.bubble(), prover, history, 2048, 1, solved=solve(None, history, 0))
        self.assertTrue(frontier['proofs'])
        self.assertTrue(all(f['history'][:len(history)] == [list(p) for p in history] for f in frontier['proofs']))
        self.assertGreater(frontier['actual_solver_nodes'], 0)
        found = evaluate(self.bubble(), SimpleNamespace(history=prover.history, abort=prover.cancel), history, 8192, 32768)
        self.assertEqual((found['proof']['winner'], found['value']), (0, 0.))
        self.assertGreater(found['proof']['plies'], 0)
        self.assertEqual(found['top'][0][:2], found['moves'][0])
        self.assertTrue(all(row[3:] == [0., -1] for row in found['top']))
        self.assertGreater(found['actual_solver_nodes'], 0)

    def test_an_opponent_threat_does_not_prove_a_defensible_root_lost(self):
        import tactical_proof
        from play import evaluate, solve
        from tests.test_tactical_proof import ONE_TURN
        try:
            prover = tactical_proof.NativeTactics(package=tactical_proof.PACKAGE)
        except FileNotFoundError:
            self.skipTest('needs the built tactical library')
        solved = solve(None, ONE_TURN, 0)
        solved['threat'] = [[0, 2], [5, 2]]
        found = evaluate(self.bubble(), prover, ONE_TURN, 64, 2048, solved=solved)
        self.assertTrue(found['proof'] is None or found['proof']['winner'] == 0)
        after = prover.history([list(p) for p in ONE_TURN] + found['moves'], nodes=2048, ms=1000)
        self.assertFalse(after['status'] == 'PROVEN_WIN' and after['proof_turns'] == 1)

    def test_a_held_search_launches_no_network_rows_and_then_finishes(self):
        from play import evaluate
        rows = []

        def watch(count):
            rows.append(count)
            return len(rows) <= 6

        found = evaluate(self.bubble(), None, [(0, 0)], 256, 0, watch=watch)
        self.assertEqual(sum(rows[2:6]), 0)   # rows launched before the hold took effect arrive at the second call
        self.assertGreater(sum(rows[6:]), 0)
        self.assertEqual(found['actual_completed'], 512)

    def test_a_cancelled_search_stops_and_closes_its_graph(self):
        from neural_search import GameGraph
        from play import evaluate
        graphs, made = [], GameGraph.__init__

        def track(graph, *args, **kwargs):
            made(graph, *args, **kwargs)
            graphs.append(graph)

        def watch(count):
            raise Cancelled()

        with unittest.mock.patch.object(GameGraph, '__init__', track), self.assertRaises(Cancelled):
            evaluate(self.bubble(), None, [(0, 0)], 1_000_000, 0, watch=watch)
        self.assertEqual(len(graphs), 1)
        self.assertIsNone(graphs[0].ptr)

    def test_cancelled_root_query_gets_its_stop_event_set(self):
        from play import solve
        from types import SimpleNamespace
        events, aborted = [], []

        def watch(count):
            raise Cancelled()

        def history(*args, cancel_event, **options):
            events.append(cancel_event)
            cancel_event.wait(PATIENCE)
            return dict(status='UNKNOWN', reason='cancelled', nodes_used=0)

        with self.assertRaises(Cancelled):
            solve(SimpleNamespace(history=history, abort=lambda: aborted.append(True)), [(0, 0)], 100, watch)
        self.assertEqual(aborted, [True])
        self.assertTrue(events and events[0].is_set())

    def test_pooled_evaluations_complete_each_turn_like_single_ones(self):
        from play import evaluate, evaluate_many
        histories = [[(0, 0)], [(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 0)]]
        pooled = evaluate_many(self.bubble(), [], histories, 16, 0)
        for history, found in zip(histories, pooled):
            alone = evaluate(self.bubble(), None, history, 16, 0)
            self.assertEqual(len(found['moves']), len(alone['moves']))
            self.assertEqual(found['actual_completed'], alone['actual_completed'])
            self.assertEqual(found['top'][0][:2], found['moves'][0])
            game = Game(history)
            try:
                mover = game.player
                for move in found['moves']:
                    game.play(*move)
                self.assertNotEqual(game.player, mover)
            finally:
                game.close()
            self.assertAlmostEqual(found['value'], alone['value'], places=1)

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

    def test_a_tier_runs_only_the_simulations_its_root_lacks(self):
        from play import evaluate
        engines, bubble, history = Engines('cpu'), self.bubble(), [(0, 0)]
        self.addCleanup(engines.close)
        for tier in (8, 32, 128):
            evaluate(bubble, None, history, tier, 0, trees=engines.game_graph(bubble, 1, keep=True))
        graph = engines.graphs[1][1]
        visits = int(graph.result(0, 0, 0, 0)['visits'].sum())
        self.assertGreaterEqual(visits, 128)
        # Both roots of the turn already hold 128 visits, so one more tier runs one simulation at each; the second
        # stone's also counts at the first stone's edge.
        evaluate(bubble, None, history, 128, 0, trees=engines.game_graph(bubble, 1, keep=True))
        graph.at(history)
        self.assertEqual(int(graph.result(0, 0, 0, 0)['visits'].sum()), visits + 2)
        trees = engines.game_graph(bubble, 1, 'abc', keep=True)
        tree, missing = trees([(0, 0)], 32, bubble.evaluator)
        self.assertIsNot(tree, graph)
        self.assertIsNone(graph.ptr)
        self.assertEqual(missing, 32)
        tree.search(10, root_samples=16, batch_size=16)
        self.assertEqual(trees([(0, 0)], 32, bubble.evaluator), (tree, 32 - int(tree.result(0, 0, 0, 0)['visits'].sum())))

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


class GameProofs(unittest.TestCase):
    """A proof found at one position of the game is shown and used at the positions before it, with the solver."""

    def setUp(self):
        import hexnet
        import tactical_proof
        import torch
        from tests.test_tactical_proof import LATE_WIN
        if not tactical_proof.library().exists():
            self.skipTest('needs the native tactical library')
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / 'ema.pt'
        torch.manual_seed(0)
        hexnet.save_model(path, hexnet.HexNet(hexnet.HexNetConfig(
            blocks=1, channels=8, pool_every=1, line_length=5, value_hidden=8, head_channels=4)))
        entry = dict(id='bubble:tiny', name='tiny', kind='bubble', presets=PRESETS['bubble'], checkpoints=[], path=path)
        self.session = Session({'bubble:tiny': entry}, Engines('cpu'), Evaluations(), save_initial=False)
        self.addCleanup(self.session.close)
        self.session.configure_seat(1, 'human')
        # Side 0 has a four-turn win whose winning turn starts at (-1, -11). Which checked route the solver returns
        # depends on its search, so the tests hold any route of four or five turns and relate the other positions to it.
        self.start = [tuple(p) for p in LATE_WIN]
        self.session.load(self.start + [(-1, -11)], True)
        # Every solver answer of the test, for the failure message when a proof is missing.
        self.answers, history = [], tactical_proof.IsolatedTactics.history
        def answer(prover, *args, **options):
            result = history(prover, *args, **options)
            self.answers.append({k: result.get(k) for k in ('status', 'reason', 'nodes_used', 'elapsed_ms')})
            return result
        patcher = unittest.mock.patch.object(tactical_proof.IsolatedTactics, 'history', answer)
        patcher.start()
        self.addCleanup(patcher.stop)

    def analyse(self, ply, solver_nodes, preset='custom', force=False):
        self.session.configure_analysis('bubble:tiny', preset=preset, auto=False,
                                        custom=dict(simulations=8, solver_nodes=solver_nodes))
        self.session.analyse(ply, force)
        wait(lambda: not self.session.state()['jobs'], 60)
        return self.session.state()['evaluations'][ply]

    def label(self, history, plies):
        """The proof label of a side-0 win `plies` placements long at `history`: its turns follow from the stones
        the side to move has left."""
        game = Game(history)
        try:
            return dict(winner=0, turns=proof_turns(plies, game.remaining, game.player == 0), plies=plies)
        finally:
            game.close()

    def found_win(self):
        """The solver's win for side 0 at ply 80, checked against the position's known length and its own line."""
        found = self.analyse(80, 32768)
        self.assertIsNotNone(found['proof'], (found, self.answers))
        plies = found['proof']['plies']
        self.assertEqual(found['proof'], self.label(self.start + [(-1, -11)], plies))
        self.assertIn(found['proof']['turns'], (4, 5))
        self.assertEqual(found['pv'][-1][2:], [0, plies])
        return found

    def test_a_proof_carries_back_to_the_played_move_and_stays(self):
        seven = self.found_win()
        before = seven['proof']['plies'] + 1
        six = self.analyse(79, 0)
        line = [[-1, -11, 0, 1]] + [[*p[:3], p[3] + 1] for p in seven['pv']]
        self.assertEqual((six['proof'], six['value'], six['pv']), (self.label(self.start, before), 1., line))
        self.assertEqual((six['top'][0][:2], six['top'][0][3:]), ([-1, -11], [1., 1]))
        saved = self.session.lookup(self.start)
        self.assertEqual(saved['proof']['plies'], before)
        budget = dict(simulations=saved['simulations'], solver_nodes=saved['solver_nodes'])
        self.session.save(self.start, self.session.engine_key(self.session.analysis), budget,
                          dict(moves=[], value=.5, top=[], proof=None, pv=[], threat=[]), 'tiny')
        self.assertEqual(self.session.lookup(self.start)['proof']['plies'], before)
        self.session.undo()
        self.assertEqual(len(self.session.history), 79)
        self.assertEqual(self.session.state()['evaluations'][79]['proof']['plies'], before)
        self.assertEqual(self.analyse(79, 0, force=True)['proof']['plies'], before)
        self.assertEqual(self.analyse(79, 0, preset='lightning')['proof']['plies'], before)

    def test_a_reopened_game_keeps_the_proofs_of_positions_it_undid(self):
        archive = tempfile.TemporaryDirectory()
        self.addCleanup(archive.cleanup)
        self.session.archive = Path(archive.name)
        self.session.load(self.start + [(-1, -11)], True)
        found = self.found_win()
        plies = found['proof']['plies']
        # A record whose root is unproven can still contain a verified leaf continuation, even on an undone branch.
        fact = dict(history=[list(p) for p in self.session.history], winner=0, plies=plies, pv=found['pv'])
        self.session.store.add(self.session.history, self.session.engine_key(self.session.analysis), self.session.analysis['budget'],
                               dict(moves=[], value=.5, top=[], proof=None, pv=[], threat=[], proofs=[fact]))
        self.session.save_freeplay()
        self.session.undo()
        self.session.save_freeplay()
        ident = self.session.remember_match(self.session.freeplay_directory)
        reopened = Session(dict(self.session.entries), FakeEngines(), Evaluations(), archive=archive.name, save_initial=False)
        self.addCleanup(reopened.close)
        study = reopened.open_saved_game(ident, 1)
        self.addCleanup(study.close)
        self.assertEqual(len(study.history), 79)
        self.assertEqual(study.state()['evaluations'][79]['proof'], self.label(self.start, plies + 1))

    def test_positions_inside_a_line_are_proven_after_a_reload(self):
        seven = self.found_win()
        plies = seven['proof']['plies']
        line = [tuple(p[:2]) for p in seven['pv'][:3]]
        history = self.start + [(-1, -11), *line]
        self.session.load(history, True)
        inside = self.analyse(81, 0)
        # The root query may prove it again, with its certificate as evidence; the verdict is the line's.
        verdict = {k: inside['proof'][k] for k in ('winner', 'turns', 'plies')}
        self.assertEqual((verdict, inside['value']), (self.label(history[:81], plies - 1), 0.))
        self.assertEqual(self.session.state()['evaluations'][83]['proof'], self.label(history, plies - 3))
        self.assertEqual(self.session.state()['evaluations'][79]['proof']['plies'], plies + 1)


class PrincipalVariation(unittest.TestCase):
    """`principal_variation` and the solver step of an evaluation."""

    def test_a_known_win_without_a_move_still_finds_and_saves_its_witness(self):
        from play import solve
        from tactical_proof import NativeTactics, independent_verify
        from tests.test_tactical_proof import OPEN_THREE
        from types import SimpleNamespace
        engine = NativeTactics()
        known = [dict(history=OPEN_THREE, winner=0, plies=24, pv=[]),
                 dict(history=OPEN_THREE+[[-1, 0], [2, 1]], winner=0, plies=20, pv=[])]
        found = solve(SimpleNamespace(history=engine.history, abort=engine.cancel), OPEN_THREE, 1000, known=known)
        self.assertTrue(found['moves'])
        proof = found['proof']
        self.assertEqual(proof['winner'], 0)
        self.assertTrue(proof['dependencies'])
        self.assertEqual(independent_verify(proof['certificate'], OPEN_THREE,
                                           known=[d['outcome'] for d in proof['dependencies']]), 'PROVEN_WIN')

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
        # LATE_WIN has a four-turn win and none shorter; the line starts with the turn and ends on the winning ply.
        self.assertEqual((found['proof']['winner'], found['proof']['turns']), (0, 4))
        self.assertEqual([p[:2] for p in found['pv'][:2]], found['moves'])
        self.assertEqual([p[3] for p in found['pv'][-2:]], [found['proof']['plies']-1, found['proof']['plies']])
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
            self.assertEqual(list(found), ['six:Six · CPU', 'six:shrimp', 'strix:Strix', 'drip:Drip'])
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
                             {'six:Six · CPU': 'six', 'six:shrimp': 'shrimp', 'strix:Strix': 'strix', 'drip:Drip': 'drip'})

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
            self.assertEqual(list(found), ['bubble:alpha', 'bubble:broken', 'bubble:beta', 'bubble:gamma', 'drip:Drip'])
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

    def test_a_bubble_entry_carries_its_q_range_floor_to_its_searches(self):
        import play
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'runs/alpha/checkpoints/main/000100').mkdir(parents=True)
            (root / 'runs/alpha/checkpoints/main/000100/ema.pt').write_bytes(b'')
            (root / 'models').mkdir()
            for name, floor in (('flat', .5), ('wrong', 3), ('alpha', 0)):
                spec = dict(name=name, kind='bubble', path='../runs/alpha', q_range_floor=floor)
                (root / f'models/{name}.json').write_text(json.dumps(spec))
            found = scan(root / 'models', root / 'runs', [], None)
        self.assertEqual(list(found), ['bubble:alpha', 'bubble:flat', 'drip:Drip'])
        self.assertNotIn('q_range_floor', found['bubble:alpha'])
        self.assertEqual(found['bubble:flat']['q_range_floor'], .5)
        self.assertEqual([play.search_key('ab', e) for e in (found['bubble:alpha'], found['bubble:flat'])], ['ab', 'ab~q0.5'])
        self.assertNotEqual(*(play.search_key('ab', dict(q_range_floor=x)) for x in (.5000001, .5000002)))
        bubble = unittest.mock.Mock(sha256='ab'*32)
        with unittest.mock.patch('neural_search.GameGraph') as tree:
            turn = play.TurnSearch(bubble, None, [(0, 0)], 8, play.solve(None, [(0, 0)], 0), q_range_floor=.5)
            turn.advanced([(0, 0)], 8, None)
            turn.close()
        self.assertEqual(tree.call_args.kwargs['q_range_floor'], .5)


if __name__ == '__main__':
    unittest.main()
