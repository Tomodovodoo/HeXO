import json
import asyncio
import importlib.util
import random
import socket
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from bot_api import APIError, Adapter, MAX_CELLS, board_game, server
from hexo import Game
from notation import MAX_STONES, NotationConflict, Record, dumps, loads
from tests import PATIENCE
from tests.reference import interleave


def board(history):
    game = Game(history)
    try:
        return {'to_move': ('x', 'o')[game.player],
                'cells': [{'q': q, 'r': r, 'p': ('x', 'o')[side]} for q, r, side in game.cells]}
    finally:
        game.close()


class OfficialNotation(unittest.TestCase):
    def test_official_example_and_metadata_roundtrip(self):
        source = ('name[GameName 0]platform[WebsiteXY 0]playercross[BlueWhale 0]'
                  'playercircle[GreenSnake 0]timecontrol[Fischer 60+5]endreason[win]'
                  'winner[cross]datetime[2026-03-18 23:20:14];\n'
                  '1. [-1,0][0,1]; 2. [-1,1][-2,2]; 3. [1,-1][-5,5];'
                  '4. [-1,2][1,0]; 5. [5,0][-3,2];')
        record = loads(source)
        self.assertEqual(record.history[0], (0, 0))
        self.assertEqual(len(record.history), 11)
        self.assertEqual(loads(dumps(record)), record)
        # Metadata is opaque: the upstream example claims a winner even though
        # these listed moves do not produce one, and uses the legacy datetime key.
        self.assertEqual(record.metadata['winner'], 'cross')
        record = loads(' version[1] name[  a ; b  ]; 1. [ 8 , 0 ][ 1 6 , 0 ] ! ! ;')
        self.assertEqual(record.metadata['name'], '  a ; b  ')
        self.assertEqual(record.history, [(0, 0), (8, 0), (16, 0)])
        self.assertEqual(record.threats, [2])
        self.assertEqual(loads(dumps(record)), record)
        self.assertEqual(loads('').history, [(0, 0)])

    def test_strict_turns_native_legality_and_bounds(self):
        bad = ['1.[1,0];2.[2,0][3,0];', '2.[1,0][2,0];', '1.[0,0][1,0];',
               '1.[9,0][1,0];', '1.[1,0][1,0];', 'version[2];',
               'version[1]version[1];', '1.[1,0][2,0];garbage',
               '1.[1000000000001,0][1,0];']
        for source in bad:
            with self.assertRaises(ValueError, msg=source):
                loads(source)
        for history in ([], [(0, 0)]*(MAX_STONES+1)):
            with self.assertRaises(ValueError):
                dumps(history)
        self.assertEqual(dumps([(0, 0), (1, 0)]), 'version[1];\n1. [1,0];')
        self.assertEqual(loads('1.[1,0];').history, [(0, 0), (1, 0)])
        with self.assertRaises(ValueError):
            dumps(Record([(0, 0)], {'name': 'bad]value'}, []))

    def test_terminal_first_and_second_stone_roundtrip(self):
        history = interleave([[(q, 0) for q in range(6)], [(2*q, 6) for q in range(6)]])
        text = dumps(history)
        self.assertEqual(loads(text).history, history)
        self.assertTrue(text.endswith('6. [5,0];'))
        annotated = Record(history, {'version': '1', 'name': 'first-stone win'}, [0]*5+[2])
        self.assertEqual(loads(dumps(annotated)), annotated)
        with self.assertRaises(ValueError):
            loads(text+'\n7.[15,7][16,7];')
        with self.assertRaises(ValueError):
            loads(dumps(history[:-1])+'\n6.[5,0][15,7];')
        history = interleave([[(q, 0) for q in range(5)], [(0, 2), (2, 2), (3, 2), (5, 2)]])
        history += [(1, 2), (4, 2)]
        self.assertEqual(loads(dumps(history)).history, history)


class OfficialAPI(unittest.TestCase):
    def reconstructed(self, data, **kwargs):
        game = board_game(data, deadline=time.perf_counter()+1, **kwargs)
        self.addCleanup(game.close)
        return game

    def test_unordered_board_and_backtracking_without_invented_cells(self):
        history = [(0,0),(8,0),(16,0),(24,0),(32,0),(-8,0),(40,0)]
        data = board(history)
        random.Random(47).shuffle(data['cells'])
        game = self.reconstructed(data)
        self.assertEqual({(q,r,p) for q,r,p in game.cells},
                         {(c['q'],c['r'],0 if c['p']=='x' else 1) for c in data['cells']})
        self.assertEqual((game.player, game.remaining), (0, 2))
        # Sorted greedy would begin (-8,0),(8,0), leaving cross stranded.
        self.assertEqual([c[:2] for c in game.cells][1:3], [[8,0],[16,0]])
        data = board([(0,0),(8,0),(16,0)])
        data['cells'].reverse()
        self.assertEqual(len(self.reconstructed(data).cells), 3)

    def test_phase_terminal_type_and_resource_rejections(self):
        self.assertEqual(self.reconstructed({'to_move':'x','cells':[]}).remaining, 1)
        self.assertEqual(self.reconstructed(board([(0,0),(1,0)])).remaining, 1)
        cases = [({'to_move':'o','cells':[]}, 400),
                 ({'to_move':'x','cells':[{'q':0,'r':0,'p':'x'}]}, 400),
                 ({'to_move':'o','cells':[{'q':0,'r':0,'p':'x'}]*2}, 400),
                 ({'to_move':'o','cells':[{'q':False,'r':0,'p':'x'}]}, 400),
                 ({'to_move':'o','cells':[{'q':2**63,'r':0,'p':'x'}]}, 400),
                 ({'to_move':'o','cells':[{}]*(MAX_CELLS+1)}, 400)]
        for data, status in cases:
            with self.assertRaises(APIError) as caught:
                self.reconstructed(data)
            self.assertEqual(caught.exception.status, status)
        with self.assertRaises(APIError) as caught:
            self.reconstructed(board([(0,0)]), node_limit=0)
        self.assertEqual(caught.exception.status, 503)
        unreachable = {'to_move':'o','cells':[{'q':q,'r':0,'p':p}
                       for q,p in ((0,'x'),(8,'x'),(-8,'x'),(24,'o'),(32,'o'))]}
        with self.assertRaises(APIError) as caught:
            self.reconstructed(unreachable)
        self.assertEqual(caught.exception.status, 400)
        history = interleave([[(q, 0) for q in range(5)], [(0,2),(2,2),(3,2),(5,2)]])
        history += [(1,2),(4,2)]
        with self.assertRaises(APIError) as caught:
            self.reconstructed(board(history))
        self.assertEqual(caught.exception.status, 409)

    def test_schema_moves_echo_limits_and_winning_first_stone(self):
        adapter = Adapter(ms=1, width=4)
        response = adapter.turn({'board':board([(0,0)]), 'request_id':73, 'time_limit':.1})
        self.assertEqual(response['request_id'], 73)
        self.assertEqual(len(response['move']['pieces']), 2)
        game = Game([(0,0)])
        try:
            for c in response['move']['pieces']:
                game.play(c['q'], c['r'])
            self.assertEqual(game.player, 0)
        finally:
            game.close()
        for field, value in [('request_id',True),('request_id',-1),('time_limit',float('nan')),
                             ('time_limit',float('inf')),('time_limit',-1),('time_limit',True)]:
            with self.assertRaises(APIError):
                adapter.turn({'board':board([(0,0)]), field:value})
        with self.assertRaises(APIError) as caught:
            adapter.turn({'board':board([(0,0)]), 'time_limit':0})
        self.assertEqual(caught.exception.status, 408)
        history = interleave([[(q,0) for q in range(6)], [(2*q,6) for q in range(6)]])[:-1]
        response = adapter.turn({'board':board(history)})
        self.assertEqual(len(response['move']['pieces']), 1)
        game = Game(history)
        try:
            game.play(**response['move']['pieces'][0])
            self.assertGreaterEqual(game.winner, 0)
        finally:
            game.close()


    def test_incomplete_body_times_out_and_server_recovers(self):
        httpd = server(Adapter(ms=1, width=4), port=0, read_timeout=.1)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        client = socket.create_connection(httpd.server_address, timeout=2)
        try:
            client.sendall(b'POST /stateless/v1-alpha/turn HTTP/1.0\r\n'
                           b'Content-Type: application/json\r\nContent-Length: 100\r\n\r\n{')
            # Keep the incomplete sender connected while requesting another route.
            root = f'http://127.0.0.1:{httpd.server_port}'
            with urlopen(root+'/capabilities.json', timeout=2) as response:
                self.assertEqual(response.status, 200)
            self.assertIn(b'408', client.recv(4096).split(b'\r\n', 1)[0])
        finally:
            client.close()
            httpd.shutdown()
            httpd.server_close()
            thread.join()

    def test_dribbled_request_line_and_headers_have_absolute_deadline(self):
        httpd = server(Adapter(ms=1, width=4), port=0, read_timeout=.12)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        root = f'http://127.0.0.1:{httpd.server_port}'
        try:
            for prefix in (b'GET /capab', b'GET /capabilities.json HTTP/1.0\r\nX-Drip: '):
                client = socket.create_connection(httpd.server_address, timeout=2)
                client.sendall(prefix)
                stop = threading.Event()
                def dribble():
                    while not stop.wait(.02):
                        try:
                            client.sendall(b'a')
                        except OSError:
                            break
                sender = threading.Thread(target=dribble, daemon=True)
                sender.start()
                try:
                    # Sender remains active; idle timeout alone never expires.
                    start = time.monotonic()
                    with urlopen(root+'/capabilities.json', timeout=1) as response:
                        self.assertEqual(response.status, 200)
                    self.assertLess(time.monotonic()-start, .8)
                finally:
                    stop.set()
                    sender.join()
                    client.close()
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join()

    def test_missing_unreadable_or_unloadable_model_returns_json_503(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'model.bin'
            path.write_bytes(b'test-only model')
            with patch.object(Game, 'load_model'):
                adapter = Adapter(ms=1, width=4, model=path)
            httpd = server(adapter, port=0)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            request = Request(f'http://127.0.0.1:{httpd.server_port}/stateless/v1-alpha/turn',
                              json.dumps({'board':board([(0,0)])}).encode(),
                              {'Content-Type':'application/json'})
            def assert_unavailable():
                with self.assertRaises(HTTPError) as caught:
                    urlopen(request, timeout=2)
                with caught.exception as response:
                    self.assertEqual(response.code, 503)
                    self.assertIn('Configured model', json.load(response)['error'])
            try:
                path.unlink()
                assert_unavailable()
                path.write_bytes(b'test-only model')
                with patch.object(Path, 'read_bytes', side_effect=PermissionError('unreadable')):
                    assert_unavailable()
                with patch.object(Game, 'load_model', side_effect=ValueError('disappeared before load')):
                    assert_unavailable()
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join()

    def test_local_http_routes_and_errors(self):
        httpd = server(Adapter(ms=1, width=4), port=0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        root = f'http://127.0.0.1:{httpd.server_port}'
        try:
            with urlopen(root+'/capabilities.json') as response:
                capabilities = json.load(response)
            self.assertFalse(capabilities['stateless']['versions']['v1-alpha']['move_time_limit'])
            self.assertNotIn('basic_websocket', capabilities)
            self.assertNotIn('matchmaking', capabilities)
            request = Request(root+'/stateless/v1-alpha/turn',
                              json.dumps({'board':board([(0,0)]),'request_id':3}).encode(),
                              {'Content-Type':'application/json'})
            with urlopen(request) as response:
                self.assertEqual(json.load(response)['request_id'], 3)
            request = Request(root+'/stateless/v1-alpha/turn', b'{}', {'Content-Type':'application/json'})
            with self.assertRaises(HTTPError) as caught:
                urlopen(request)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join()


class TimedClocks(unittest.TestCase):
    def test_clocked_comparison_book_settings_and_paired_scores(self):
        from timed_match import side_settings, paired_openings, comparison_summary
        from dense_openings import canonical
        with TemporaryDirectory() as directory:
            path = Path(directory)/'settings.json'
            path.write_text(json.dumps(dict(search=dict(root_samples=32, max_simulations=None, native_scheduler=True),
                                            solver=dict(enabled=False))), encoding='utf-8')
            self.assertIsNone(side_settings(path)['search']['max_simulations'])
            self.assertTrue(side_settings(path)['search']['native_scheduler'])
            path.write_text(json.dumps(dict(search=dict(native_scheduler=True), solver=dict(nodes=512))), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'time slices'):
                side_settings(path)
            from timed_engine import TimedEngine
            with self.assertRaisesRegex(ValueError, 'time slices'):
                TimedEngine(dict(kind='bubble', search=dict(native_scheduler=True), solver=dict(nodes=512)))
            with self.assertRaisesRegex(ValueError, 'proof frontier'):
                TimedEngine(dict(kind='bubble', search=dict(native_scheduler=True), solver=dict(leaf=True)))
            path.write_text(json.dumps(dict(search=dict(native_scheduler=False), solver=dict(nodes=512))), encoding='utf-8')
            self.assertEqual(side_settings(path)['solver']['nodes'], 512)
            path.write_text(json.dumps(dict(search=dict(native_scheduler=True, enabled=False), solver=dict(nodes=512))), encoding='utf-8')
            self.assertEqual(side_settings(path)['solver']['nodes'], 512)
            path.write_text(json.dumps(dict(search=dict(native_scheduler=True), solver=dict(workers=12, budget=.1))), encoding='utf-8')
            self.assertEqual(side_settings(path)['solver'], dict(workers=12, budget=.1))
            for solver in (dict(budget=0), dict(budget=1.5), dict(workers=0)):
                path.write_text(json.dumps(dict(search=dict(native_scheduler=True), solver=solver)), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'solver'):
                    side_settings(path)
            path.write_text(json.dumps(dict(solver=dict(budget=.1))), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'native timed solving only'):
                side_settings(path)
            with self.assertRaisesRegex(ValueError, 'native timed solving only'):
                TimedEngine(dict(kind='bubble', solver=dict(workers=4)))
            path.write_text(json.dumps(dict(search=dict(native_feed=True))), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Unsupported search'):
                side_settings(path)
            starts = paired_openings(Path(directory), 4, 73, suite='standard-v1')
            self.assertEqual(starts, paired_openings(Path(directory), 4, 73, suite='standard-v1'))
            self.assertEqual(len({canonical(s['history'])[0] for s in starts}), 4)
            self.assertTrue(all(s['key'] == canonical(s['history'])[0] for s in starts))
            with self.assertRaisesRegex(ValueError, 'frozen'):
                paired_openings(Path(directory), 1, 0, suite='book')
        rows = [dict(seed=0, swapped=False, winner='x', reason='win'),
                dict(seed=0, swapped=True, winner='o', reason='win'),
                dict(seed=1, swapped=False, winner=None, reason='capped'),
                dict(seed=1, swapped=True, winner='x', reason='win')]
        self.assertIsNone(comparison_summary(rows[:1])['elo_delta'])
        stats = comparison_summary(rows, True)
        self.assertEqual((stats['wins'], stats['losses'], stats['capped'], stats['pairs']), (2, 1, 1, 2))
        self.assertEqual(stats['pair_score'], .625)
        self.assertEqual(stats['elo_interval'], [None, None])
        rows[-1]['reason'] = 'crash'
        stats = comparison_summary(rows, True)
        self.assertFalse(stats['valid'])
        self.assertIsNone(stats['elo_delta'])
        self.assertIsNone(stats['decision'])

    def test_fixed_turn_resource_drain_is_outside_both_players_clocks(self):
        from timed_match import Match, play_turn
        clock = [0]
        class Engine:
            def turn(self, game, **options):
                self.options = options
                clock[0] = 80_000_000
                return dict(moves=[[1, 0], [2, 0]], evaluated=7, completed=8)
            def wait_idle(self):
                clock[0] = 500_000_000
                return 420
        engine = Engine()
        with TemporaryDirectory() as directory:
            match = Match(dict(players=dict(cross=dict(kind='native'), circle=dict(kind='native')),
                               time_control='1', turn_cap_ms=100, fixed_turn_time=True),
                          now=lambda: clock[0], directory=directory)
            try:
                match.start()
                clock[0] = 20_000_000
                state = play_turn(match, engine, wait_worker=True, record_engine=True)
                self.assertEqual(state['history'], [[0, 0], [1, 0], [2, 0]])
                self.assertEqual((state['cross_ms'], state['circle_ms']), (1000, 920))
                self.assertIsNone(engine.options['clock'])
                self.assertEqual(engine.options['milliseconds'], 80)
                events = [json.loads(line) for line in (match.directory/'events.jsonl').read_text().splitlines()]
                reply = next(e for e in events if e['type'] == 'engine_reply')
                self.assertEqual((reply['controller_ns'], reply['worker_wait_ms']), (60_000_000, 420))
                self.assertEqual(next(e for e in events if e['type'] == 'turn')['elapsed_ns'], 80_000_000)
                clock[0] += 10_000_000
                self.assertEqual(match.snapshot()['cross_ms'], 990)
            finally:
                match.close()

    def test_worker_drain_failure_invalidates_a_comparison(self):
        from timed_match import Match, play_turn, comparison_summary
        clock = [0]
        class Engine:
            def turn(self, *_args, **_kwargs):
                clock[0] = 80_000_000
                return dict(moves=[[1, 0], [2, 0]])
            def wait_idle(self):
                clock[0] = 500_000_000
                raise TimeoutError('still running')
        match = Match(dict(players=dict(cross=dict(kind='native'), circle=dict(kind='native')),
                           time_control='1', turn_cap_ms=100), now=lambda: clock[0])
        self.addCleanup(match.close)
        match.start()
        state = play_turn(match, Engine(), wait_worker=True)
        self.assertEqual(state['result']['reason'], 'crash')
        self.assertEqual(state['history'], [[0, 0]])
        stats = comparison_summary([dict(seed=0, swapped=False, **state['result'])])
        self.assertFalse(stats['valid'])
        self.assertIsNone(stats['elo_delta'])

    def test_clocked_runner_saves_every_game_and_color_pair(self):
        from timed_match import main
        from timed_engine import legal_turn
        from unittest.mock import MagicMock
        engines = [MagicMock(), MagicMock()]
        for index, engine in enumerate(engines):
            engine.__enter__.return_value = engine
            engine.identity = dict(checkpoint=f'arm-{index}')
            engine.wait_idle.return_value = 0
            engine.turn.side_effect = lambda game, **_kwargs: dict(
                moves=legal_turn([cell[:2] for cell in game.cells]), evaluated=3, completed=4)
        with TemporaryDirectory() as directory, patch('timed_match.TimedEngine', side_effect=engines):
            output = Path(directory)/'match'
            args = ['--a', 'native', '--b', 'native', '--pairs', '2', '--turn-ms', '100',
                    '--max-placements', '5', '--out', str(output)]
            main(args)
            saved = json.loads((output/'summary.json').read_text())
            self.assertEqual((saved['comparison']['games'], saved['comparison']['pairs']), (4, 2))
            self.assertEqual(saved['comparison']['pair_score'], .5)
            self.assertTrue(saved['comparison']['valid'])
            self.assertEqual(len(list(output.glob('*/game.htttx'))), 4)
            self.assertEqual(saved['timing']['a']['reported_evaluations'], 12)
            self.assertEqual(saved['timing']['b']['reported_completed_visits'], 16)
            self.assertTrue(all(e.reset.call_count == 4 for e in engines))
            with self.assertRaises(SystemExit):
                main(args)

    def test_half_turn_pause_increment_and_exact_deadline(self):
        from timed_match import Match
        clock = [0]
        match = Match(dict(players=dict(cross=dict(kind='human'), circle=dict(kind='human')),
                           time_control=dict(base_ms=1000, increment_ms=200), turn_cap_ms=1000), now=lambda: clock[0])
        self.addCleanup(match.close)
        match.start()
        clock[0] = 100_000_000
        state = match.submit([[1, 0]], 0, 0, partial=True)
        self.assertEqual((state['running'], state['remaining'], state['circle_ms']), ('o', 1, 900))
        clock[0] = 150_000_000
        match.pause()
        clock[0] = 900_000_000
        state = match.resume()
        self.assertEqual(state['circle_ms'], 850)
        clock[0] += 100_000_000
        state = match.submit([[2, 0]], state['turn_id'], state['revision'], partial=True)
        self.assertEqual((state['running'], state['circle_ms']), ('x', 950))
        clock[0] += 1_000_000_000
        state = match.submit([[0, 1], [0, 2]], state['turn_id'], state['revision'])
        self.assertEqual(state['result'], dict(winner='o', reason='time'))
        self.assertEqual(len(state['history']), 3)
        self.assertEqual(state['turn_cap_remaining_ms'], 0)
        self.assertEqual(match.turn_spent_ns, 1_000_000_000)

    def test_winning_turn_retains_its_elapsed_time(self):
        from timed_match import Match
        clock = [0]
        history = [[0, 0], [0, 3], [1, 3], [1, 0], [2, 0], [3, 3],
                   [4, 3], [3, 0], [4, 0], [5, 3], [6, 3]]
        match = Match(dict(players=dict(cross=dict(kind='human'), circle=dict(kind='human')),
                           history=history, time_control='1', turn_cap_ms=100), now=lambda: clock[0])
        self.addCleanup(match.close)
        match.start()
        clock[0] = 50_000_000
        result = match.submit([[5, 0]], 0, 0)
        self.assertEqual(result['result'], dict(winner='x', reason='win'))
        self.assertEqual((match.turn_spent_ns, result['turn_cap_remaining_ms']), (50_000_000, 50))

    def test_receipt_before_deadline_is_atomic_with_watchdog(self):
        from timed_match import Match, play_turn
        clock = [0]
        thinking, release, watchdog_started = (threading.Event() for _ in range(3))
        class Engine:
            def turn(self, *_args, **_kwargs):
                thinking.set()
                release.wait(1)
                return dict(moves=[[1, 0], [2, 0]])
        match = Match(dict(players=dict(cross=dict(kind='human'), circle=dict(kind='native')),
                           time_control='1'), now=lambda: clock[0])
        self.addCleanup(match.close)
        match.start()
        clock[0] = 900_000_000
        reply_thread = threading.Thread(target=play_turn, args=(match, Engine()))
        reply_thread.start()
        watchdog = None
        try:
            self.assertTrue(thinking.wait(1))
            with match.lock:
                def tick():
                    watchdog_started.set()
                    match.tick()
                watchdog = threading.Thread(target=tick)
                watchdog.start()
                self.assertTrue(watchdog_started.wait(1))
                release.set()
                until = time.monotonic()+1
                while match.receipt[2][0] is None and time.monotonic() < until:
                    time.sleep(.001)
                self.assertEqual(match.receipt[2][0], 900_000_000)
                clock[0] = 1_100_000_000
        finally:
            release.set()
            reply_thread.join(1)
            if watchdog:
                watchdog.join(1)
        self.assertFalse(reply_thread.is_alive())
        self.assertEqual(match.snapshot()['history'], [[0, 0], [1, 0], [2, 0]])
        self.assertIsNone(match.result)
        self.assertEqual(match.snapshot()['circle_ms'], 100)

    def test_restore_is_paused_and_half_turn_round_trips(self):
        from timed_match import Match
        clock = [0]
        with TemporaryDirectory() as directory:
            match = Match(dict(players=dict(cross=dict(kind='human'), circle=dict(kind='human')),
                               time_control='1+0.2'), now=lambda: clock[0], directory=directory)
            match.start()
            clock[0] = 100_000_000
            match.submit([[1, 0]], 0, 0, partial=True)
            match.pause()
            self.assertEqual(loads(match.notation()).history, [(0, 0), (1, 0)])
            self.assertNotIn('timecontrol', loads(match.notation()).metadata)
            restored = Match.restore(match.directory, now=lambda: 9_000_000_000)
            try:
                self.assertEqual(restored.state, 'paused')
                self.assertIsNone(restored.clock.running)
                self.assertEqual(restored.snapshot()['circle_ms'], 900)
                self.assertEqual(restored.created, match.created)
            finally:
                match.close()
                restored.close()

    def test_future_increment_does_not_extend_current_clock(self):
        from time_control import allowance
        budget = allowance(dict(cross_ms=15, circle_ms=1000, increment_ms=10000), 0)
        self.assertLessEqual(budget['hard_ms'], 15)
        self.assertEqual(budget['hard_ms']-budget['reserve_ms'], 5)
        budget = allowance(dict(cross_ms=15, circle_ms=1000, increment_ms=0), 0)
        self.assertGreater(budget['hard_ms']-budget['reserve_ms'], 0)

    def test_native_controller_returns_a_complete_turn_by_its_allowance(self):
        from timed_engine import TimedEngine, legal_turn
        for cap in (0, -1):
            with self.assertRaisesRegex(ValueError, 'simulation cap'):
                TimedEngine(dict(kind='bubble', search=dict(max_simulations=cap)))
        with TimedEngine(dict(kind='native')) as engine:
            game = Game([[0, 0]])
            try:
                started = time.monotonic()
                result = engine.turn(game, 30)
                self.assertLess(time.monotonic()-started, .5)
                self.assertEqual(legal_turn([[0, 0]], result['moves']), result['moves'])
                engine.lock.acquire()
                try:
                    started = time.monotonic()
                    queued = engine.turn(game, 30)
                    self.assertLess(time.monotonic()-started, .2)
                    self.assertEqual(queued['stop_reason'], 'busy')
                    self.assertEqual(legal_turn([[0, 0]], queued['moves']), queued['moves'])
                finally:
                    engine.lock.release()
                stopped = threading.Event()
                stopped.set()
                result = engine.turn(game, 1000, cancel=stopped)
                self.assertEqual(legal_turn([[0, 0]], result['moves']), result['moves'])
                self.assertGreaterEqual(engine.wait_idle(), 0)
                self.assertFalse(engine.busy)
                engine.reset([[0, 0]])
                reply = engine.turn(game, 30)
                self.assertEqual(legal_turn([[0, 0]], reply['moves']), reply['moves'])
            finally:
                game.close()

    def test_http_opponent_failures_keep_timeout_and_illegal_results(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from timed_engine import TimedEngine
        from timed_match import Match, play_turn
        class Handler(BaseHTTPRequestHandler):
            mode = 'timeout'
            def log_message(self, *_):
                pass
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(dict(stateless=dict(versions={'v1-alpha': {}}))).encode())
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                mode = self.mode
                if mode == 'timeout':
                    time.sleep(1)
                self.send_response(408 if mode == 'http_timeout' else 200)
                self.end_headers()
                piece = dict(q='bad' if mode == 'coordinates' else 0, r=0)
                try:
                    self.wfile.write(json.dumps(dict(move=dict(pieces=[piece]))).encode())
                except ConnectionError:
                    pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for mode in ('timeout', 'http_timeout', 'coordinates', 'occupied'):
                with self.subTest(mode=mode):
                    Handler.mode = mode
                    with TimedEngine(dict(kind='htttx', url=f'http://127.0.0.1:{server.server_port}')) as engine:
                        match = Match(dict(players=dict(cross=dict(kind='human'), circle=dict(kind='htttx')),
                                           time_control='5'))
                        try:
                            match.start()
                            result = play_turn(match, engine)
                            self.assertEqual(result['result'], dict(winner='x', reason='engine_timeout' if mode in ('timeout', 'http_timeout') else 'illegal'))
                            self.assertEqual(result['history'], [[0, 0]])
                        finally:
                            match.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


@unittest.skipUnless(importlib.util.find_spec('aiohttp'), 'Install the api extra')
class TimedAPI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from aiohttp.test_utils import TestClient, TestServer
        from timed_api import create_app
        from timed_engine import legal_turn
        self.calls = []
        self.engines = []
        calls = self.calls
        engines = self.engines
        class Engine:
            identity = dict(checkpoint='fake')
            def __init__(self, config):
                self.closed = False
                engines.append(self)
            def turn(self, game, milliseconds=None, *, clock=None, cancel=None, publish=None):
                history = [list(c[:2]) for c in game.cells]
                calls.append((history, clock))
                for _ in range(10):
                    if cancel and cancel.is_set():
                        break
                    time.sleep(.002)
                return dict(moves=legal_turn(history), win_probability=.6)
            def close(self):
                self.closed = True
        self.client = TestClient(TestServer(create_app(engine_factory=Engine)))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_match_places_live_clock_stream_and_pause(self):
        created = await self.client.post('/matches', json=dict(players=dict(cross=dict(kind='human'),
            circle=dict(kind='human')), time_control=dict(base_ms=10000, increment_ms=2000)))
        state = await created.json()
        root = '/matches/'+state['match_id']
        for _ in range(20):
            state = await (await self.client.get(root)).json()
            if state['state'] == 'ready':
                break
            await asyncio.sleep(.01)
        state = await (await self.client.post(root+'/start')).json()
        state = await (await self.client.post(root+'/place', json=dict(q=1, r=0,
                            turn_id=state['turn_id'], revision=state['revision']))).json()
        self.assertEqual((state['remaining'], state['running']), (1, 'o'))
        feed = await self.client.get(root+'/events')
        event = json.loads((await feed.content.readline()).decode().removeprefix('data: '))
        self.assertEqual(event['running'], 'o')
        feed.close()
        state = await (await self.client.post(root+'/pause')).json()
        self.assertEqual(state['state'], 'paused')
        balance = state['circle_ms']
        await asyncio.sleep(.02)
        state = await (await self.client.get(root)).json()
        self.assertEqual(state['circle_ms'], balance)
        notation = await (await self.client.get(root+'/notation')).text()
        self.assertEqual(loads(notation).history, [(0, 0), (1, 0)])

    async def test_websocket_clock_and_interrupt_roll_back_previous(self):
        from timed_engine import legal_turn
        ws = await self.client.ws_connect('/bws/v1-alpha/game')
        await ws.send_json(dict(type='config', **{'x-bubble-clock': dict(version=1,
            cross_ms=5000, circle_ms=6000, increment_ms=100)}))
        await ws.send_json(dict(type='move_request', side='o', previous=[], request_id=1,
                               move_time_limit=.2))
        first = await ws.receive_json(timeout=1)
        self.assertEqual(first['request_id'], 1)
        self.assertLessEqual(self.calls[-1][1]['circle_ms'], 6000)
        own = first['move']['pieces']
        history = [[0, 0]]+[[p['q'], p['r']] for p in own]
        reply = legal_turn(history)
        previous = [dict(side='o', pieces=own),
                    dict(side='x', pieces=[dict(q=q, r=r) for q, r in reply])]
        await ws.send_json(dict(type='move_request', side='o', previous=previous, request_id=2))
        await asyncio.sleep(.003)
        await ws.send_json(dict(type='interrupt', request_id=2))
        await ws.send_json(dict(type='move_request', side='o', previous=previous, request_id=3))
        response = await ws.receive_json(timeout=1)
        self.assertEqual(response['request_id'], 3)
        self.assertEqual(self.calls[-1][0], history+reply)
        await ws.close()

    async def test_server_shutdown_closes_an_open_bot_session(self):
        ws = await self.client.ws_connect('/bws/v1-alpha/game')
        await ws.send_json(dict(type='move_request', side='o', previous=[], request_id=1))
        await ws.receive_json(timeout=1)
        await asyncio.wait_for(self.client.close(), 1)
        # The session's handler releases its engine as the server closes its socket, possibly after close returns.
        end = time.monotonic()+PATIENCE
        while not all(engine.closed for engine in self.engines) and time.monotonic() < end:
            await asyncio.sleep(.01)
        self.assertTrue(all(engine.closed for engine in self.engines))

    async def test_finished_match_releases_its_engines(self):
        created = await self.client.post('/matches', json=dict(players=dict(cross=dict(kind='native'),
            circle=dict(kind='human')), time_control='10'))
        root = '/matches/'+(await created.json())['match_id']
        for _ in range(50):
            state = await (await self.client.get(root)).json()
            if state['state'] == 'ready':
                break
            await asyncio.sleep(.01)
        await self.client.post(root+'/start')
        state = await (await self.client.post(root+'/resign', json=dict(side='o'))).json()
        self.assertEqual(state['state'], 'finished')
        for _ in range(50):
            if self.engines[-1].closed:
                break
            await asyncio.sleep(.01)
        self.assertTrue(self.engines[-1].closed)
        self.assertFalse(self.engines[0].closed)

    async def test_resumed_match_waits_for_abandoned_worker(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from aiohttp.test_utils import TestClient, TestServer
        from timed_api import create_app
        arrived, release = threading.Event(), threading.Event()
        class Handler(BaseHTTPRequestHandler):
            calls = 0
            def log_message(self, *_):
                pass
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(dict(stateless=dict(versions={'v1-alpha': {}}))).encode())
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                Handler.calls += 1
                if Handler.calls == 1:
                    arrived.set()
                    release.wait(.4)
                self.send_response(200)
                self.end_headers()
                try:
                    self.wfile.write(json.dumps(dict(move=dict(pieces=[dict(q=1, r=0), dict(q=2, r=0)]))).encode())
                except ConnectionError:
                    pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = TestClient(TestServer(create_app()))
        await client.start_server()
        try:
            created = await client.post('/matches', json=dict(players=dict(cross=dict(kind='human'),
                circle=dict(kind='htttx', url=f'http://127.0.0.1:{server.server_port}')), time_control='10'))
            root = '/matches/'+(await created.json())['match_id']
            for _ in range(100):
                state = await (await client.get(root)).json()
                if state['state'] == 'ready':
                    break
                await asyncio.sleep(.01)
            self.assertEqual(state['state'], 'ready')
            await client.post(root+'/start')
            self.assertTrue(await asyncio.to_thread(arrived.wait, 1))
            self.assertEqual((await client.post(root+'/pause')).status, 200)
            self.assertEqual((await client.post(root+'/resume')).status, 200)
            await asyncio.sleep(.03)
            release.set()
            for _ in range(100):
                state = await (await client.get(root)).json()
                if state['result'] or len(state['history']) == 3:
                    break
                await asyncio.sleep(.01)
            self.assertIsNone(state['result'])
            self.assertEqual(state['history'], [[0, 0], [1, 0], [2, 0]])
        finally:
            release.set()
            await client.close()
            server.shutdown()
            server.server_close()
            thread.join()

    async def test_websocket_evaluation_echoes_request_id(self):
        ws = await self.client.ws_connect('/bws/v1-alpha/game')
        await ws.send_json(dict(type='eval_request', side='o', request_id=7))
        response = await ws.receive_json(timeout=1)
        self.assertEqual((response['type'], response['request_id']), ('eval_response', 7))
        await ws.close()

    async def test_interrupt_during_send_cannot_restore_old_history(self):
        from aiohttp import web
        from timed_engine import legal_turn
        blocked, release = asyncio.Event(), asyncio.Event()
        original_send = web.WebSocketResponse.send_json
        async def delayed_send(socket, data, *args, **kwargs):
            if data.get('request_id') == 2 and data.get('type') == 'move_response':
                blocked.set()
                await release.wait()
            return await original_send(socket, data, *args, **kwargs)
        with patch.object(web.WebSocketResponse, 'send_json', delayed_send):
            ws = await self.client.ws_connect('/bws/v1-alpha/game')
            try:
                await ws.send_json(dict(type='move_request', side='o', previous=[], request_id=1))
                first = await ws.receive_json(timeout=1)
                own = first['move']['pieces']
                advanced = [[0, 0]]+[[p['q'], p['r']] for p in own]
                previous = [dict(side='o', pieces=own),
                            dict(side='x', pieces=[dict(q=q, r=r) for q, r in legal_turn(advanced)])]
                await ws.send_json(dict(type='move_request', side='o', previous=previous, request_id=2))
                await asyncio.wait_for(blocked.wait(), 1)
                await ws.send_json(dict(type='interrupt', request_id=2))
                await ws.send_json(dict(type='setup'))
                await ws.send_json(dict(type='config', **{'x-bubble-clock': dict(version=1,
                    cross_ms=2000, circle_ms=2000, increment_ms=100)}))
                await ws.send_json(dict(type='move_request', side='o', previous=[], request_id=3))
                self.assertEqual((await ws.receive_json(timeout=1))['request_id'], 3)
                self.assertEqual(self.calls[-1][1]['increment_ms'], 100)
                self.assertGreater(self.calls[-1][1]['circle_ms'], 1900)
                release.set()
                self.assertEqual((await ws.receive_json(timeout=1))['request_id'], 2)
                await ws.send_json(dict(type='move_request', side='o', previous=[], request_id=4))
                self.assertEqual((await ws.receive_json(timeout=1))['request_id'], 4)
                self.assertEqual(self.calls[-1][0], [[0, 0]])
            finally:
                release.set()
                await ws.close()

    async def test_restored_match_records_the_resumed_engines(self):
        from aiohttp.test_utils import TestClient, TestServer
        from timed_api import create_app
        from timed_match import Match
        with TemporaryDirectory() as folder:
            original = Match(dict(players=dict(cross=dict(kind='native'), circle=dict(kind='human')),
                                  time_control='10', identities=[dict(checkpoint='old'), dict(checkpoint='human')]),
                             directory=folder)
            original.start()
            original.pause()
            root = '/matches/'+original.id
            directory = original.directory
            original.close()
            class Engine:
                identity = dict(checkpoint='new')
                def __init__(self, config):
                    pass
                def close(self):
                    pass
            client = TestClient(TestServer(create_app(directory=folder, engine_factory=Engine)))
            await client.start_server()
            try:
                response = await client.post(root+'/resume')
                self.assertEqual(response.status, 200)
                await client.post(root+'/pause')
                spec = json.loads((directory/'spec.json').read_text())
                self.assertEqual(spec['identities'][0]['checkpoint'], 'new')
                events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
                change = next(event for event in events if event['type'] == 'engines')
                self.assertEqual(change['previous_identities'][0]['checkpoint'], 'old')
                self.assertEqual(change['identities'][0]['checkpoint'], 'new')
            finally:
                await client.close()


@unittest.skipUnless(importlib.util.find_spec('aiohttp'), 'API extra not installed')
class ArenaNative(unittest.IsolatedAsyncioTestCase):
    async def test_gateway_errors_retry_account_and_presence(self):
        from aiohttp import web
        from aiohttp.test_utils import TestServer
        from arena_bot import NativeArena
        calls = dict(account=0, presence=0)
        connected, release = asyncio.Event(), asyncio.Event()

        async def account(request):
            if request.method == 'GET':
                calls['account'] += 1
                if calls['account'] == 1:
                    return web.Response(status=502)
            return web.json_response(dict(name='local'))

        async def presence(request):
            calls['presence'] += 1
            if calls['presence'] == 1:
                return web.Response(status=502)
            response = web.StreamResponse()
            await response.prepare(request)
            await response.write(b'\n')
            connected.set()
            await release.wait()
            return response

        app = web.Application()
        app.router.add_route('*', '/api/bot/account', account)
        app.router.add_get('/api/bot/stream', presence)
        async with TestServer(app) as server:
            bot = NativeArena(str(server.make_url('/')), 'private')
            task = asyncio.create_task(bot.run())
            try:
                await asyncio.wait_for(connected.wait(), 4)
                self.assertFalse(task.done())
                self.assertEqual(calls, dict(account=2, presence=2))
            finally:
                release.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_finished_game_cleanup_does_not_delay_the_next_game(self):
        from arena_bot import NativeArena
        started = {game_id: asyncio.Event() for game_id in ('old', 'next')}
        draining, release = asyncio.Event(), asyncio.Event()

        async def play(event):
            started[event['gameId']].set()
            try:
                await asyncio.Event().wait()
            finally:
                draining.set()
                await release.wait()

        bot = NativeArena('http://localhost', 'private')
        first = dict(type='gameStart', gameId='old', side='o', opponent=dict(name='local'))
        with patch.object(bot, 'play', side_effect=play):
            try:
                await bot.event(first)
                await asyncio.wait_for(started['old'].wait(), 1)
                await asyncio.wait_for(bot.event(dict(type='gameFinish', gameId='old',
                    reason='resign', winner='o')), .2)
                await asyncio.wait_for(draining.wait(), 1)
                await bot.event(dict(first, gameId='next'))
                await asyncio.wait_for(started['next'].wait(), 1)
                self.assertFalse(release.is_set())
            finally:
                release.set()
                for game_id in list(bot.games):
                    await bot.finish(game_id)
                await asyncio.gather(*bot.cleanups, return_exceptions=True)

    async def test_dropped_socket_drains_native_before_requesting_a_new_clock(self):
        import aiohttp
        from aiohttp import web
        from aiohttp.test_utils import TestServer
        from arena_bot import NativeArena, answer
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        dials, replies = [], []

        def delayed_answer(*args):
            if not started.is_set():
                started.set()
                if not release.wait(4):
                    raise TimeoutError('Test did not release Native')
                result = answer(*args)
                finished.set()
                return result
            return answer(*args)

        async def session(request):
            dials.append(finished.is_set())
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.send_json(dict(type='setup', board=dict(cells=[dict(q=0, r=0, p='x')])))
            await ws.send_json(dict(type='move_request', side='o', previous=[], request_id=len(dials),
                                    move_time_limit=5 if len(dials) == 1 else .2))
            if len(dials) == 1:
                self.assertTrue(await asyncio.to_thread(started.wait, 3))
                asyncio.get_running_loop().call_later(1.2, release.set)
                await ws.close(code=1011)
            else:
                replies.append(await ws.receive_json(timeout=3))
                await ws.close()
            return ws

        app = web.Application()
        app.router.add_get('/engine', session)
        async with TestServer(app) as server, aiohttp.ClientSession() as client:
            bot = NativeArena(str(server.make_url('/')), 'private', ms=20, width=8, depth=1)
            bot.http = client
            with patch('arena_bot.answer', side_effect=delayed_answer):
                try:
                    await asyncio.wait_for(bot.play(dict(gameId='redial',
                        engine=dict(socketUrl='/engine', token='game-only'))), 6)
                finally:
                    release.set()
        self.assertEqual(dials, [False, True])
        self.assertEqual([reply['request_id'] for reply in replies], [2])

    async def test_confirmed_turns_replay_without_duplicating_our_move(self):
        from arena_bot import answer
        first = dict(type='move_request', side='o', previous=[], request_id=17, move_time_limit=5)
        history, response, _ = answer([[0, 0]], first, 20, 8, 1)
        self.assertEqual(history, [[0, 0]])
        self.assertEqual(response['request_id'], 17)
        own = response['move']['pieces']
        game = Game([(0, 0), *[(p['q'], p['r']) for p in own]])
        try:
            opponent = game.search(ms=20, depth=1, width=8)['moves']
        finally:
            game.close()
        packet = dict(first, request_id=18, previous=[dict(side='o', pieces=own),
                      dict(side='x', pieces=[dict(q=q, r=r) for q, r in opponent])])
        confirmed, reply, _ = answer(history, packet, 20, 8, 1)
        self.assertEqual(confirmed, [[0, 0], *[[p['q'], p['r']] for p in own], *map(list, opponent)])
        self.assertEqual(reply['request_id'], 18)
        self.assertEqual(answer([[0, 0]], packet, 20, 8, 1)[:2], (confirmed, reply))

    async def test_first_stone_win_and_allowance(self):
        from arena_bot import answer
        history = interleave([[(q, 0) for q in range(6)], [(2*q, 6) for q in range(6)]])[:-1]
        packet = dict(side='x', previous=[], request_id=3, move_time_limit=.2)
        with patch.object(Game, 'search', autospec=True, side_effect=Game.search) as search:
            confirmed, reply, _ = answer(history, packet, 100, 8, 2)
        self.assertEqual(confirmed, list(map(list, history)))
        self.assertEqual(len(reply['move']['pieces']), 1)
        self.assertEqual(search.call_args.kwargs['ms'], 50)

    async def test_socket_keeps_search_through_presence_replay_and_heartbeats(self):
        import aiohttp
        from aiohttp import web
        from aiohttp.test_utils import TestServer
        from arena_bot import NativeArena, answer
        received = []
        threads = []
        started, release = threading.Event(), threading.Event()

        def slow_answer(*args):
            threads.append(threading.get_ident())
            if len(threads) == 1:
                started.set()
                if not release.wait(3):
                    raise TimeoutError('Test did not release Native')
            return answer(*args)

        async def session(request):
            self.assertNotIn('Authorization', request.headers)
            self.assertEqual(request.query['token'], 'game-only')
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.send_json(dict(type='setup', board=dict(cells=[dict(q=0, r=0, p='x')])))
            await ws.send_json(dict(type='move_request', side='o', previous=[], request_id=9,
                                    move_time_limit=5))
            self.assertTrue(await asyncio.to_thread(started.wait, 3))
            await bot.event(event)  # A reconnected presence stream repeats gameStart.
            await ws.send_json(dict(type='heartbeat', waiting=True))
            await asyncio.sleep(.03)
            release.set()
            received.append(await ws.receive_json(timeout=3))
            # The next turn must use the same connection and Native worker.
            own = received[0]['move']['pieces']
            game = Game([(0, 0), *[(p['q'], p['r']) for p in own]])
            try:
                opponent = game.search(ms=20, width=8, depth=1)['moves']
            finally:
                game.close()
            await ws.send_json(dict(type='move_request', side='o', request_id=10,
                move_time_limit=5, previous=[dict(side='o', pieces=own),
                    dict(side='x', pieces=[dict(q=q, r=r) for q, r in opponent])]))
            received.append(await ws.receive_json(timeout=3))
            await ws.close()
            return ws

        app = web.Application()
        app.router.add_get('/engine', session)
        async with TestServer(app) as server, aiohttp.ClientSession() as client:
            bot = NativeArena(str(server.make_url('/')), 'private-bot-token', ms=20, width=8, depth=1)
            bot.http = client
            event = dict(type='gameStart', gameId='local', side='o', opponent=dict(name='local'),
                         engine=dict(socketUrl='/engine', token='game-only'))
            with patch('arena_bot.answer', side_effect=slow_answer):
                try:
                    await bot.event(event)
                    await asyncio.wait_for(bot.games['local'], 5)
                finally:
                    release.set()
                    await bot.finish('local')
        self.assertEqual(received[0]['request_id'], 9)
        self.assertEqual(received[1]['request_id'], 10)
        self.assertEqual(len(threads), 2)
        self.assertEqual(threads[0], threads[1])
        game = Game([(0, 0)])
        try:
            for p in received[0]['move']['pieces']:
                game.play(p['q'], p['r'])
            self.assertEqual(game.player, 0)
        finally:
            game.close()


if __name__ == '__main__':
    unittest.main()
