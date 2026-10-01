import json
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
        cases = [({'to_move':'x','cells':[]}, 409),
                 (board([(0,0),(1,0)]), 409),
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

    def test_schema_moves_echo_limits_and_first_stone_conflict(self):
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
        with self.assertRaises(APIError) as caught:
            adapter.turn({'board':board(history)})
        self.assertEqual(caught.exception.status, 409)

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


if __name__ == '__main__':
    unittest.main()
