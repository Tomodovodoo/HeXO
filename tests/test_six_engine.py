"""CPU protocol checks for both directions of the Six adapter."""
import io
import json
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

from hexo import Game
from six_engine import ProtocolError, SixEngine, serve
import dense_config
from dense_eval import MatchGame, make_report, report_path, write_league
import dashboard


FAKE_ENGINE = '''import os, sys, time
marker = sys.argv[1]
log = sys.argv[2] if len(sys.argv) > 2 else None
moves = []
for raw in sys.stdin:
    words = raw.split()
    if not words: continue
    if words[0] == 'six': print('id name Fake\\nid version 123456789abc\\nsixok', flush=True)
    elif words[0] == 'isready': print('readyok', flush=True)
    elif words[0] == 'newgame' and log:
        with open(log, 'a') as out: out.write('newgame\\n')
    elif words[0] == 'position':
        numbers = list(map(int, words[words.index('moves')+1:])) if 'moves' in words else []
        moves = list(zip(numbers[::2], numbers[1::2]))
    elif words[0] == 'go':
        if marker != '-' and not os.path.exists(marker):
            open(marker, 'w').close()
            time.sleep(2)
        used = set(moves)
        choices = ([(0, 0)] if not moves else
                   [(q, r) for q in range(-2, 4) for r in range(-2, 4)
                    if (q, r) not in used and any(max(abs(q-a), abs(r-b), abs(q-a+r-b)) <= 1 for a, b in moves)])
        count = 1 if not moves or len(moves) % 2 == 0 else 2
        print('bestmove ' + ' '.join(f'{q} {r}' for q, r in choices[:count]), flush=True)
    elif words[0] == 'quit': break
'''


class FakePlayer:
    checkpoint = 'main/000001'
    model_sha256 = 'abcdef1234567890'

    def __init__(self, turns):
        self.turns = iter(turns)
        self.histories = []

    def set_history(self):
        pass

    def turn(self, game, milliseconds=None):
        self.histories.append([tuple(c[:2]) for c in game.cells])
        return {'moves': next(self.turns)}


class FakeTree:
    def advance(self, move):
        pass

    def close(self):
        pass


class FakeModel:
    def tree(self, history, seed, tactics):
        return FakeTree()


class SixProtocolTests(unittest.TestCase):
    def test_external_anchor_needs_its_own_league_id(self):
        for name in ('seal', '', 'main/000001'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'external_name'):
                dense_config.EvaluationSettings(external_engine='fake-engine', external_name=name)
        self.assertEqual(dense_config.EvaluationSettings(external_engine='fake-engine',
                                                          external_name='six-v2').external_name, 'six-v2')

    def test_server_handshake_position_errors_and_turn_sizes(self):
        player = FakePlayer([[(0, 0)], [(1, 0), (0, 1)]])
        source = io.StringIO('six\nisready\nnewgame\nposition radius 8\ngo movetime 10\n'
                             'position radius 8 moves 0 0\ngo nodes 5 depth 2\n'
                             'position radius 8 moves 0 0 0 0\nposition radius 8 setup x 0 0\nquit\n')
        out = io.StringIO()
        serve(player, source, out)
        lines = out.getvalue().splitlines()
        self.assertEqual(lines[:4], ['id name Bubble main/000001', 'id version abcdef123456', 'sixok', 'readyok'])
        self.assertEqual([line for line in lines if line.startswith('bestmove')], ['bestmove 0 0', 'bestmove 1 0 0 1'])
        self.assertEqual(player.histories, [[], [(0, 0)]])
        self.assertEqual(sum(line.startswith('error') for line in lines), 2)

    def test_server_stops_after_winning_first_stone(self):
        opening = [(0, 0), (0, 3), (1, 3), (1, 0), (2, 0), (3, 3),
                   (4, 3), (3, 0), (4, 0), (5, 3), (6, 3)]
        player = FakePlayer([[(5, 0), (6, 0)]])
        flat = ' '.join(f'{q} {r}' for q, r in opening)
        out = io.StringIO()
        serve(player, io.StringIO(f'position radius 8 moves {flat}\ngo\nquit\n'), out)
        self.assertEqual(out.getvalue().strip(), 'bestmove 5 0')

    def test_client_restart_after_timeout(self):
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder)/'engine.py'
            marker = Path(folder)/'slow.once'
            script.write_text(FAKE_ENGINE)
            with SixEngine([sys.executable, str(script), str(marker)], timeout=.2) as engine:
                game = Game()
                try:
                    with self.assertRaisesRegex(ProtocolError, 'timed out'):
                        engine(game, 1)
                    self.assertIsNone(engine.game)
                    self.assertEqual(engine(game, 1), [(0, 0)])
                finally:
                    game.close()

    def test_client_starts_each_game(self):
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder)/'engine.py'
            log = Path(folder)/'games.txt'
            script.write_text(FAKE_ENGINE)
            with SixEngine([sys.executable, str(script), '-', str(log)], timeout=1) as engine:
                for _ in range(2):
                    game = Game()
                    try:
                        self.assertEqual(engine(game, 1), [(0, 0)])
                    finally:
                        game.close()
            self.assertEqual(log.read_text().splitlines(), ['newgame', 'newgame'])

    def test_client_allows_the_move_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder)/'engine.py'
            marker = Path(folder)/'slow.once'
            script.write_text(FAKE_ENGINE)
            with SixEngine([sys.executable, str(script), str(marker)], timeout=.2) as engine:
                game = Game()
                try:
                    self.assertEqual(engine(game, 1000), [(0, 0)])
                finally:
                    game.close()

    def test_external_match_side_plays_through(self):
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder)/'engine.py'
            script.write_text(FAKE_ENGINE)
            with SixEngine([sys.executable, str(script), '-'], timeout=1) as engine:
                game = MatchGame([FakeModel(), 'six'], [(0, 0)], 1, 1, 1, False, 8, {}, engine, 1, anchor='six')
                try:
                    while not game.over():
                        action = next((m for m in game.game.legal_moves() if m not in [(0, 0)]), None)
                        game.searched({'action': action})
                    record = game.finish()
                    self.assertNotIn('error', record)
                    self.assertGreater(record['plies'], 2)
                finally:
                    if game.game.ptr:
                        game.finish()

    def test_league_keeps_seal_and_external_anchor(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            checkpoint = 'main/000001'
            league = dict(champion=checkpoint, checkpoints=[dict(id=checkpoint, variant='main', step=1,
                                                                   elo=0., matches=[])], variants=[])
            config = dense_config.RunConfig()
            records = [dict(seed=1, pair=0, challenger_color=c, winner=c, opening=[[0, 0]],
                            moves=[[0, 0]], plies=1, reason='six-in-a-row') for c in (0, 1)]
            for name in ('seal', 'six'):
                path = report_path(run, checkpoint, name)
                path.parent.mkdir(parents=True)
                settings = (replace(config.evaluation, external_engine='fake', external_name='six')
                            if name == 'six' else config.evaluation)
                path.write_text(json.dumps(make_report(checkpoint, name, records, {checkpoint: 'abc'}, settings)))
            write_league(run, league, config)
            self.assertEqual(set(league['anchors']), {'seal', 'six'})
            self.assertEqual(league['anchors']['seal']['games'], 2)
            self.assertEqual(league['anchors']['six']['games'], 2)

    def test_dashboard_reads_named_anchor(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            checkpoint = run/'checkpoints/main/000001/manifest.json'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text(json.dumps({'created_at': 1}))
            league = dict(checkpoints=[dict(id='main/000001', variant='main', step=1, elo=0.)],
                          anchors={'six': dict(elo=-50., elo_interval=[-80., -20.])})
            (run/'league.json').write_text(json.dumps(league))
            self.assertEqual(dashboard.series(run, {'created_at': 0}, 'main', 'anchor_elo:six')['points'],
                             [[1, -50., -80., -20.]])
            status = dict(stage='playing', comparison=dict(candidate='main/000002', opponent='six'),
                          tally=dict(wins=1, losses=0, capped=0, games=1, elo_delta=20., elo_interval=[0., 40.]))
            self.assertEqual(dashboard.provisional(league, status)['elo'], -30.)


if __name__ == '__main__':
    unittest.main()
