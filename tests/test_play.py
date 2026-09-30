"""Play server: evaluation store, review labels, background jobs and the HTTP surface, with fake engines."""
import json
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from hexo import Game
from play import Cancelled, Evaluations, Handler, PRESETS, Session, budget_of, import_history, review, scan

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
        return dict(moves=moves, value=.5, top=[[*moves[0], .9]], proof=None, line=[], threat=[], ms=1)

    def turn(self, entry, budget, history):
        return legal_turn(history)


RUN = tempfile.TemporaryDirectory()
for number in (1, 2):
    (Path(RUN.name) / f'checkpoints/main/00000{number}').mkdir(parents=True)
    (Path(RUN.name) / f'checkpoints/main/00000{number}/ema.pt').write_bytes(bytes([number]))


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
            store.add([(0, 0)], 'run/a', dict(simulations=32, solver_nodes=2048), dict(value=.4, moves=[]))
            store.add([(0, 0)], 'run/a', dict(simulations=512, solver_nodes=2048), dict(value=.6, moves=[]))
            store.add([(0, 0)], 'run/b', dict(simulations=32, solver_nodes=2048), dict(value=.9, moves=[]))
            store.add([(0, 0)], 'run/a', dict(simulations=32, solver_nodes=2048), dict(value=.3, moves=[]))
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

    def test_reuse_needs_every_budget_dimension(self):
        store = Evaluations()
        store.add([], 'e', dict(simulations=512, solver_nodes=0), dict(value=.1, moves=[]))
        self.assertIsNone(store.covering([], 'e', STANDARD))
        store.add([], 'e', dict(simulations=128, solver_nodes=131072), dict(value=.2, moves=[]))
        self.assertEqual(store.covering([], 'e', STANDARD)['value'], .2)
        self.assertEqual(store.covering([], 'e', dict(simulations=64, solver_nodes=0))['value'], .1)

    def test_index_keeps_the_newest_entries(self):
        store = Evaluations(limit=2)
        for n in range(3):
            store.add([(0, 0)] if n == 0 else [(0, 0), (n, 0)], 'e', STANDARD, dict(value=n / 10, moves=[]))
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
        self.session.review_game()
        wait(lambda: not self.session.state()['jobs'])
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

    def test_undo_returns_to_the_players_last_turn(self):
        self.session.play(0, 0)
        wait(lambda: len(self.history()) == 3)
        self.session.pause(True)
        self.session.play(0, 1)
        self.session.undo()
        self.assertEqual(self.history(), [(0, 0), *map(tuple, legal_turn([(0, 0)]))])
        self.session.undo()
        self.assertEqual(self.history(), [])

    def test_budgets(self):
        self.assertEqual(budget_of('bubble', 'custom', dict(simulations=0)), dict(simulations=0, solver_nodes=32768))
        for custom in (dict(simulations=10 ** 6), dict(ms=5), dict(simulations='8')):
            with self.assertRaises(ValueError):
                budget_of('bubble', 'custom', custom)
        with self.assertRaises(ValueError):
            budget_of('native', 'heavy')


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
        self.assertEqual(set(quiet), {'revision', 'jobs'})
        self.engines.hold = False
        self.post('/cancel', dict(id=state['jobs'][0]['id']))
        wait(lambda: not self.session.state()['jobs'])

    def test_import_export_retry_and_origin(self):
        self.post('/seat', dict(side=1, engine='human'))
        state = self.post('/import', dict(text='version[1];\n1. [1,0][2,0];'))
        self.assertEqual(state['history'], [[0, 0], [1, 0], [2, 0]])
        self.assertTrue(state['paused'])
        self.assertEqual(self.get('/htttx'), 'version[1];\n1. [1,0][2,0];')
        replay = json.loads(self.get('/replay'))
        self.assertEqual(import_history(json.dumps(replay)), state['history'])
        self.assertEqual(self.post('/retry', dict(ply=2))['history'], [[0, 0], [1, 0]])
        with self.assertRaises(HTTPError) as caught:
            self.get('/htttx')
        self.assertEqual(caught.exception.code, 409)
        caught.exception.close()
        for path, body in (('/play', dict(q=40, r=0)), ('/import', dict(text='1. [9,9]')), ('/retry', dict(ply=9))):
            with self.assertRaises(HTTPError) as caught:
                self.post(path, body)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()
        with self.assertRaises(HTTPError) as caught:
            self.post('/new', headers={'Origin': 'http://example.com'})
        self.assertEqual(caught.exception.code, 403)
        caught.exception.close()


class Registry(unittest.TestCase):
    def test_runs_models_and_entries_are_found(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for checkpoint in ('main/000100', 'main/000200', 'play/000150'):
                (root / 'runs/alpha/checkpoints' / checkpoint).mkdir(parents=True)
                (root / 'runs/alpha/checkpoints' / checkpoint / 'ema.pt').write_bytes(b'')
            (root / 'runs/alpha/champion.json').write_text(json.dumps(dict(checkpoint='play/000150')))
            (root / 'models/beta').mkdir(parents=True)
            (root / 'models/beta/ema.pt').write_bytes(b'')
            (root / 'elsewhere').mkdir()
            (root / 'elsewhere/net.pt').write_bytes(b'')
            (root / 'models/gamma.json').write_text(json.dumps(dict(name='gamma', kind='bubble', path='../elsewhere/net.pt')))
            found = scan(root / 'models', root / 'runs', [root / 'runs/alpha'], root / 'missing.dll')
            self.assertEqual(list(found), ['bubble:alpha', 'bubble:beta', 'bubble:gamma', 'native:Native'])
            self.assertEqual(found['bubble:alpha']['checkpoints'], ['play/000150', 'main/000200', 'main/000100'])
            self.assertEqual(found['bubble:beta']['checkpoints'], [''])


if __name__ == '__main__':
    unittest.main()
