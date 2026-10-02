"""Six in the browser (web/engine/six) against the server's Six (python/six_engine.py driving sixengine)."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest
import export_web
from hexo import Game
from six_engine import SixEngine

ROOT = Path(__file__).resolve().parents[1]
NETWORKS = ROOT/'web'/'engine'/'six'/'networks'
SIX = ROOT/'models'/'six'
BINARY = SIX/('sixengine.exe' if os.name == 'nt' else 'sixengine')
NODE = shutil.which('node')
NETWORK = 'gen-0455'
BUILT = NODE is not None and (NETWORKS/f'{NETWORK}.onnx').exists() and (ROOT/'web'/'engine'/'ort'/'ort.wasm.min.mjs').exists()


def node(job):
    done = subprocess.run([NODE, str(ROOT/'tests'/'web'/'six.mjs')], input=json.dumps({**job, 'network': str(NETWORKS/f'{NETWORK}.onnx')}),
                          capture_output=True, text=True, encoding='utf-8')
    if done.returncode:
        raise RuntimeError(done.stderr[-2000:])
    return json.loads(done.stdout)


def positions():
    """Recorded positions with one or two stones left in the turn, from the opening to the middle game."""
    games = export_web.histories(every=9)
    return [games[i] for i in (0, 1, 7, 20, 22, 24)]


@unittest.skipUnless(BUILT, 'needs node, python tools/build_web.py ort six')
class Browser(unittest.TestCase):
    def test_stop_ends_the_turn_early(self):
        answer = node(dict(kind='stop', history=positions()[2], nodes=100000, after_ms=1500))
        self.assertTrue(answer['stopped'])
        self.assertLess(answer['ms'], 30000)
        self.assertEqual(len(answer['moves']), 2)


@unittest.skipUnless(BUILT and BINARY.exists() and (SIX/f'{NETWORK}.onnx').exists(),
                     'needs the server Six in models/six (the play page installs it) and the browser build')
class Parity(unittest.TestCase):
    """Moves of the browser search at a fixed node count against sixengine on the CPU, through the server adapter.
    A turn costs about 0.4 s per position searched in node (one WebAssembly thread), so the budgets stay small."""

    @staticmethod
    def server():
        return SixEngine([str(BINARY), '--net', str(SIX/f'{NETWORK}.onnx'), '--cpu'], timeout=300, mirrored=True, cwd=SIX)

    def test_turns_match_the_server(self):
        cases = [dict(history=h, nodes=48) for h in positions()]+[dict(history=positions()[3], nodes=160)]
        browser = node(dict(kind='turns', cases=cases))
        with self.server() as engine:
            for case, answer in zip(cases, browser):
                game = Game(case['history'])
                try:
                    self.assertEqual(answer['moves'], [list(p) for p in engine(game, nodes=case['nodes'])], case)
                finally:
                    game.close()

    def test_a_game_keeps_its_tree_like_the_server(self):
        history, nodes = positions()[1], 48
        browser = node(dict(kind='game', history=history, nodes=nodes, turns=3))
        game = Game(history)
        try:
            with self.server() as engine:
                for answer in browser:
                    turn = engine(game, nodes=nodes)
                    self.assertEqual(answer['moves'], [list(p) for p in turn])
                    for q, r in turn:
                        game.play(q, r)
        finally:
            game.close()


if __name__ == '__main__':
    unittest.main()
