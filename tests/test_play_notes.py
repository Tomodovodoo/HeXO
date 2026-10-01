"""Saved browser analysis uses the exact played history and selected search settings."""
import json
import tempfile
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

from hexo import Game
from play import AnalysisStore, DensePlayer, Handler, position_key


class AnalysisNotes(unittest.TestCase):
    def test_dense_worker_plays_a_clocked_complete_turn_on_cpu(self):
        import hexnet
        from timed_engine import TimedEngine, legal_turn
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'ema.pt'
            model = hexnet.HexNet(hexnet.HexNetConfig(blocks=1, channels=8, pool_every=1,
                                line_length=5, value_hidden=8, head_channels=4))
            hexnet.save_model(path, model)
            with TimedEngine(dict(kind='bubble', model=str(path), device='cpu',
                                  solver=dict(enabled=False))) as engine:
                game = Game([[0, 0]])
                try:
                    result = engine.turn(game, 1000)
                    self.assertEqual(legal_turn([[0, 0]], result['moves']), result['moves'])
                    self.assertGreater(result.get('evaluated', 0), 0)
                    self.assertEqual(result['backend'], 'dense')
                    self.assertEqual(result['model_sha256'], engine.model_sha256)
                finally:
                    game.close()

    def test_position_key_tracks_order_checkpoint_and_settings(self):
        cells = [[0, 0, 0], [1, 0, 1], [2, 0, 1]]
        settings = dict(search=True, simulations=128, solver=True, solver_nodes=32768)
        key = position_key(cells, 'main/065000', settings)
        self.assertEqual(key, position_key(cells,
                                           'main/065000', dict(reversed(list(settings.items())))))
        self.assertNotEqual(key, position_key([cells[0], cells[2], cells[1]], 'main/065000', settings))
        self.assertNotEqual(key, position_key([[0, 0, 0], [1, 0, 0], [2, 0, 1]], 'main/065000', settings))
        self.assertNotEqual(key, position_key(cells, 'main/075000', settings))
        self.assertNotEqual(key, position_key(cells, 'main/065000', settings | dict(simulations=64)))

    def test_store_survives_restart_and_overwrites_one_position(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'play-notes.json'
            settings = dict(search=True, simulations=128)
            store = AnalysisStore(path)
            store.save([], 'main/065000', settings, dict(win_probability=.4))
            store.save([[0, 0, 0]], 'main/065000', settings, dict(win_probability=.6))
            self.assertEqual(store.history([[0, 0, 0], [1, 0, 1]], 'main/065000', settings),
                             [dict(ply=0, win_probability=.4), dict(ply=1, win_probability=.6)])
            reloaded = AnalysisStore(path)
            self.assertEqual(reloaded.get([[0, 0, 0]], 'main/065000', settings)['analysis']['win_probability'], .6)
            reloaded.save([[0, 0, 0]], 'main/065000', settings, dict(win_probability=.7))
            self.assertEqual(len(AnalysisStore(path).entries), 2)
            self.assertEqual(AnalysisStore(path).get([[0, 0, 0]], 'main/065000', settings)['analysis']['win_probability'], .7)

    def test_bot_note_returns_on_undo_and_analyze_uses_cache_until_forced(self):
        class Player(DensePlayer):
            def __init__(self):
                self.checkpoint = 'main/065000'
                self.model_sha256 = 'test'
                self.options = dict(search=True, simulations=128, solver=True, solver_nodes=32768)
                self.calls = 0

            def models(self):
                return [dict(id=self.checkpoint, label=self.checkpoint)]

            def set_history(self, history=()):
                pass

            def turn(self, game, milliseconds=None, analyze=False):
                self.calls += 1
                return dict(moves=[[0, 0]], backend='dense', checkpoint=self.checkpoint,
                            elapsed_ms=1, suggestions=[dict(move=[0, 0], probability=.8)],
                            player=game.player, win_probability=.25 + self.calls / 4,
                            proof_status='UNKNOWN', winning_line=[], threat=None,
                            threat_checked=analyze, solver_status='UNKNOWN', settings=dict(self.options))

        with tempfile.TemporaryDirectory() as directory:
            player = Player()
            notes = AnalysisStore(Path(directory) / 'play-notes.json')
            handler = type('NotesHandler', (Handler,), dict(game=Game(), neural=player, notes=notes,
                            run=None, model=None, search_run=None, label='Bubble',
                            log_message=lambda *args: None))
            server = HTTPServer(('127.0.0.1', 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            def post(path, body=None):
                request = Request(f'http://127.0.0.1:{server.server_port}{path}',
                                  data=json.dumps(body or {}).encode(),
                                  headers={'Content-Type': 'application/json'})
                with urlopen(request) as response:
                    return json.load(response)
            try:
                played = post('/bot')
                self.assertEqual(played['game_analysis'], [dict(ply=0, win_probability=.5)])
                self.assertIsNone(played['position_analysis'])
                undone = post('/undo')
                self.assertEqual(undone['position_analysis']['analysis']['win_probability'], .5)
                self.assertEqual(player.calls, 1)
                self.assertEqual(post('/analyze')['analysis']['win_probability'], .75)
                self.assertEqual(post('/analyze')['analysis']['win_probability'], .75)
                self.assertEqual(player.calls, 2)
                refreshed = post('/analyze', dict(force=True))
                self.assertEqual(refreshed['analysis']['win_probability'], 1.)
                self.assertEqual(refreshed['position_analysis']['analysis']['win_probability'], 1.)
                self.assertEqual(player.calls, 3)
                self.assertEqual(post('/new')['position_analysis']['analysis']['win_probability'], 1.)
                self.assertEqual(AnalysisStore(notes.path).get([], player.checkpoint, player.options)['analysis']['win_probability'], 1.)
                self.assertEqual(post('/bot')['analysis']['win_probability'], 1.)
                self.assertEqual(player.calls, 3)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
                handler.game.close()


if __name__ == '__main__':
    unittest.main()
