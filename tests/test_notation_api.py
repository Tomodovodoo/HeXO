import json
import asyncio
import importlib.util
import random
import socket
import threading
import time
import unittest
from http.server import HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from bot_api import APIError, Adapter, MAX_CELLS, board_game, server
from hexo import Game
from notation import MAX_STONES, NotationConflict, Record, dumps, loads
from play import Handler as PlayHandler
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


class PlayNotationEndpoint(unittest.TestCase):
    def test_current_history_and_conflict(self):
        class Handler(PlayHandler):
            game = Game()

        httpd = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        root = f'http://127.0.0.1:{httpd.server_port}'

        def play(q, r):
            request = Request(root+'/play', json.dumps({'q': q, 'r': r}).encode(),
                              {'Content-Type': 'application/json'})
            with urlopen(request, timeout=2) as response:
                self.assertEqual(response.status, 200)

        try:
            play(0, 0)
            with urlopen(root+'/htttx', timeout=2) as response:
                self.assertEqual(response.headers.get_content_type(), 'text/plain')
                self.assertEqual(response.read().decode(), 'version[1];')
            play(1, 0)
            with urlopen(root+'/htttx', timeout=2) as response:
                self.assertEqual(response.read().decode(), 'version[1];\n1. [1,0];')
            play(2, 0)
            with urlopen(root+'/htttx', timeout=2) as response:
                self.assertEqual(response.read().decode(), 'version[1];\n1. [1,0][2,0];')
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join()
            Handler.game.close()


class TimedClocks(unittest.TestCase):
    def test_half_turn_pause_increment_and_exact_deadline(self):
        from timed_match import Match
        clock = [0]
        match = Match(dict(players=dict(cross=dict(kind='human'), circle=dict(kind='human')),
                           time_control=dict(base_ms=1000, increment_ms=200)), now=lambda: clock[0])
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
            restored = Match.restore(match.directory, now=lambda: 9_000_000_000)
            try:
                self.assertEqual(restored.state, 'paused')
                self.assertIsNone(restored.clock.running)
                self.assertEqual(restored.snapshot()['circle_ms'], 900)
            finally:
                match.close()
                restored.close()

    def test_future_increment_does_not_extend_current_clock(self):
        from time_control import allowance
        budget = allowance(dict(cross_ms=15, circle_ms=1000, increment_ms=10000), 0)
        self.assertLessEqual(budget['hard_ms'], 15)

    def test_native_controller_returns_a_complete_turn_by_its_allowance(self):
        from timed_engine import TimedEngine, legal_turn
        with TimedEngine(dict(kind='native')) as engine:
            game = Game([[0, 0]])
            try:
                started = time.monotonic()
                result = engine.turn(game, 30)
                self.assertLess(time.monotonic()-started, .5)
                self.assertEqual(legal_turn([[0, 0]], result['moves']), result['moves'])
                stopped = threading.Event()
                stopped.set()
                result = engine.turn(game, 1000, cancel=stopped)
                self.assertEqual(legal_turn([[0, 0]], result['moves']), result['moves'])
            finally:
                game.close()


@unittest.skipUnless(importlib.util.find_spec('aiohttp'), 'Install the api extra')
class TimedAPI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from aiohttp.test_utils import TestClient, TestServer
        from timed_api import create_app
        from timed_engine import legal_turn
        self.calls = []
        calls = self.calls
        class Engine:
            identity = dict(checkpoint='fake')
            def __init__(self, config):
                pass
            def turn(self, game, milliseconds=None, *, clock=None, cancel=None, publish=None):
                history = [list(c[:2]) for c in game.cells]
                calls.append((history, clock))
                for _ in range(10):
                    if cancel and cancel.is_set():
                        break
                    time.sleep(.002)
                return dict(moves=legal_turn(history), win_probability=.6)
            def close(self):
                pass
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


if __name__ == '__main__':
    unittest.main()
