"""Seal in the browser (web/engine/seal/engine.wasm via seal.mjs) plays the turns the server's Seal library plays."""
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
# Seal searches to a clock and adds one random far candidate at the root, so its turn is compared on positions where
# the server library gave one answer over repeated runs at 300 and 1500 ms and both builds agreed on the CI runner.
STABLE = [OPEN_THREE, IMMEDIATE] + [FIXTURE['positions'][key] for key in (
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
    def test_turns_match_the_server_library(self):
        answers = browser([dict(history=history, ms=MS) for history in STABLE])
        for history, answer in zip(STABLE, answers, strict=True):
            with self.subTest(stones=len(history)):
                self.assertEqual(sorted(answer['raw']), sorted(server(history, MS)))

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
