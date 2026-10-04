"""Native and browser Seal obey the same turn, tactic, clock and range contracts."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from hexo import Game, library
from tests.test_tactical_proof import FIXTURE, IMMEDIATE, OPEN_THREE

ROOT = Path(__file__).resolve().parents[1]
WASM = ROOT/'web'/'engine'/'seal'/'engine.wasm'
SEAL = library.with_name(library.name.replace('hexo', 'hexo_seal'))
NODE = shutil.which('node')
MS = 1000
# Timed depth and randomized far candidates can give different moves between native and WASM.
# Exercise both implementations on the same early, tactical and midgame positions.
POSITIONS = [OPEN_THREE, IMMEDIATE] + [FIXTURE['positions'][key] for key in (
    '1790600149713752:2:253', '1790599946154496:12:253', '1790600287230040:30:256', '1790600287230040:25:213',
    '1790604657706760:28:77', '1790621505551580:15:155', '1790621505551580:9:95', '1790621505551580:17:37',
    '1790622219655928:27:23')]


def browser(requests):
    lines = ''.join(json.dumps(request)+'\n' for request in requests)
    output = subprocess.run([NODE, str(ROOT/'tests'/'web'/'seal.mjs')], input=lines, capture_output=True, text=True,
                            check=True).stdout
    return [json.loads(line) for line in output.splitlines()]


def server(history, ms):
    from legacy.arena import Seal
    game = Game()
    try:
        for q, r in history:
            game.play(q, r)
        return [list(move) for move in Seal(SEAL)(game, ms)]
    finally:
        game.close()


@unittest.skipUnless(NODE and WASM.exists() and SEAL.exists(),
                     'needs node, web/engine/seal (python tools/build_web.py seal) and the Seal library')
class WebSealParity(unittest.TestCase):
    def test_both_backends_play_legal_turns_and_take_the_immediate_win(self):
        answers = browser([dict(history=history, ms=MS) for history in POSITIONS])
        for history, answer in zip(POSITIONS, answers, strict=True):
            for backend, moves in (('browser', answer['moves']), ('native', server(history, MS))):
                with self.subTest(stones=len(history), backend=backend):
                    game = Game(history)
                    try:
                        player, remaining = game.player, game.remaining
                        self.assertGreaterEqual(len(moves), 1)
                        self.assertLessEqual(len(moves), 2)
                        played = 0
                        for q, r in moves[:remaining]:
                            self.assertEqual(game.player, player)
                            self.assertTrue(game.legal(q, r))
                            game.play(q, r)
                            played += 1
                            if game.winner >= 0:
                                break
                        self.assertTrue(played == remaining or game.winner == player)
                        if history == IMMEDIATE:
                            self.assertEqual(game.winner, player)
                    finally:
                        game.close()

    def test_turn_is_cut_to_the_stones_left(self):
        empty, whole, one_left = browser([dict(history=history, ms=50) for history in ([], [[0, 0]], [[0, 0], [1, 0]])])
        self.assertEqual(empty['moves'], [[0, 0]])
        self.assertEqual(whole['moves'], whole['raw'])
        self.assertEqual(len(whole['moves']), 2)
        self.assertEqual(len(one_left['raw']), 2)
        self.assertEqual(one_left['moves'], one_left['raw'][:1])

    def test_clock_and_board_range(self):
        far = [[8 * k, 0] for k in range(8)]
        timed, outside = browser([dict(history=FIXTURE['positions']['1790600287230040:17:145'], ms=200),
                                  dict(history=far, ms=50)])
        self.assertGreaterEqual(timed['ms'], 190)
        self.assertLess(timed['ms'], 600)
        self.assertEqual(outside, dict(error='Seal board range exceeded'))
        with self.assertRaisesRegex(RuntimeError, 'range'):
            server(far, 50)


if __name__ == '__main__':
    unittest.main()
