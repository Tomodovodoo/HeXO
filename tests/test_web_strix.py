"""Strix in the browser (web/engine/strix/strix.wasm via strix/core.mjs) plays the server adapter's moves."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

import numpy as np

from hexo import Game
from tools.strix_learned_adapter import StrixLearned

ROOT = Path(__file__).resolve().parents[1]
WASM = ROOT/'web'/'engine'/'strix'/'strix.wasm'
RUNNER = ROOT/'tests'/'web'/'strix.mjs'
NODE = shutil.which('node')
MODEL = Path(os.environ.get('HEXO_STRIX_PUBLIC_MODEL', ROOT/'web'/'engine'/'strix'/'pulsatrix-10-best.safetensors'))
EXECUTABLE = (ROOT/'tools'/'strix_learned'/'target'/'release'/'hexo-strix-learned').with_suffix('.exe' if os.name == 'nt' else '')


def recorded(every=9):
    """Positions of the recorded self-play games, every `every`th ply after the first stone."""
    data = np.load(ROOT/'tests'/'fixtures'/'actor_selfplay.npz')
    found, start = [], 0
    for length in data['lengths']:
        game = data['moves'][start:start+length].tolist()
        start += length
        found += [game[:n] for n in range(1, length, every)]
    return found


def symmetries():
    """The twelve hex symmetries of axial (q, r): six rotations, each with and without the reflection (q, r) -> (r, q)."""
    rotate = lambda p: (-p[1], p[0]+p[1])
    maps = []
    for turns in range(6):
        for reflect in (False, True):
            def g(p, turns=turns, reflect=reflect):
                p = (p[1], p[0]) if reflect else tuple(p)
                for _ in range(turns):
                    p = rotate(p)
                return list(p)
            maps.append(g)
    return maps


def equivalent(history, ours, theirs):
    """Whether turn `ours` is turn `theirs` under a symmetry that maps the position after `history` onto itself."""
    def stones(points):
        return {(tuple(p), 0 if n == 0 else ((n-1)//2+1) % 2) for n, p in enumerate(points)}
    return any(stones([g(p) for p in history]) == stones(history) and [g(m) for m in theirs] == ours for g in symmetries())


def browser(requests):
    lines = ''.join(json.dumps(r)+'\n' for r in requests)
    output = subprocess.run([NODE, str(RUNNER), str(MODEL)], input=lines, capture_output=True, text=True, check=True).stdout
    return [json.loads(line) for line in output.splitlines()]


@unittest.skipUnless(NODE and WASM.exists() and MODEL.exists() and EXECUTABLE.exists(),
                     'needs node, strix.wasm, the network (python tools/build_web.py strix-network) and the native wrapper')
class WebStrixParity(unittest.TestCase):
    def setUp(self):
        self.engines = {}

    def server(self, history, simulations):
        if simulations not in self.engines:
            self.engines[simulations] = StrixLearned(MODEL, simulations=simulations, timeout_ms=600000)
            self.addCleanup(self.engines[simulations].close)
        engine = self.engines[simulations]
        game = Game([tuple(p) for p in history])
        self.addCleanup(game.close)
        return [list(m) for m in engine(game, 0)]

    def test_moves_match_the_server_adapter(self):
        cases = [(h, 8) for h in recorded()] + [(h, 64) for h in recorded(31)]
        answers = browser([dict(history=h, simulations=s) for h, s in cases])
        self.assertEqual(len(answers), len(cases))
        for (history, simulations), answer in zip(cases, answers):
            with self.subTest(stones=len(history), simulations=simulations):
                expected = self.server(history, simulations)
                if answer['moves'] != expected:
                    self.assertTrue(equivalent(history, answer['moves'], expected), (answer['moves'], expected))
                self.assertTrue(0 <= answer['value'] <= 1)

    def test_empty_board_plays_the_origin_unsearched(self):
        empty, = browser([dict(history=[], simulations=4096)])
        self.assertEqual((empty['moves'], empty['simulations'], empty['eval_states']), ([[0, 0]], 0, 1))
        self.assertTrue(0 <= empty['value'] <= 1)


if __name__ == '__main__':
    unittest.main()
