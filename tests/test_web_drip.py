"""Drip as WebAssembly (web/engine/native) plays the turns of the native library that python/play.py serves."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest
import export_web
import play
from hexo import Game

ROOT = Path(__file__).resolve().parents[1]
WASM = ROOT/'web'/'engine'/'native'/'native.wasm'
NODE = shutil.which('node')
UNREACHED_MS = 600_000


def wasm_turns(cases):
    done = subprocess.run([NODE, str(ROOT/'tests'/'web'/'drip.mjs')], input=json.dumps(cases), capture_output=True,
                          text=True, encoding='utf-8')
    if done.returncode:
        raise RuntimeError(done.stderr)
    return json.loads(done.stdout)


def positions():
    """Recorded self-play positions and the tactical fixtures, without finished games."""
    tactical = json.loads((ROOT/'tests'/'fixtures'/'tactical_positions.json').read_text())['positions'].values()
    found = []
    for history in [*export_web.histories(every=7), *tactical]:
        game = Game(history)
        if game.winner < 0:
            found.append([list(map(int, p)) for p in history])
        game.close()
    return found


@unittest.skipUnless(NODE and WASM.exists(), 'needs node and a built web/engine (python tools/build_web.py wasm)')
class WebDripParity(unittest.TestCase):
    def test_turns_match_the_library_at_a_fixed_depth(self):
        """Depth-bounded searches with a deadline neither side reaches choose the same turn with the same score and
        completed depth. Each side searches the positions in order in one process, as the server's search child does,
        so a proven plan carries over alike. Node counts may differ: std::sort orders equal-scored turns differently
        in libc++ and libstdc++."""
        found = positions()
        cases = [dict(history=h, ms=UNREACHED_MS, depth=depth) for depth in (2, 3) for h in found]
        cases += [dict(history=h, ms=UNREACHED_MS, depth=4) for h in found[::4]]
        expected = []
        for case in cases:
            game = Game(case['history'])
            try:
                found = game.search(case['ms'], case['depth'])
            finally:
                game.close()
            expected.append(dict(moves=[list(m) for m in found['moves']], score=found['score'], depth=found['depth']))
        actual = wasm_turns(cases)
        self.assertEqual(len(actual), len(expected))
        for case, want, got in zip(cases, expected, actual):
            with self.subTest(stones=len(case['history']), depth=case['depth']):
                self.assertEqual({k: got[k] for k in want}, want)
        self.assertGreater(len(cases), 150)
        self.assertTrue(any(abs(e['score']) >= 10_000_000 for e in expected))

    def test_a_timed_turn_keeps_to_its_budget_and_is_legal(self):
        history = positions()[10]
        for ms in (100, 250):
            [turn] = wasm_turns([dict(history=history, ms=ms, depth=12)])
            self.assertLess(turn['elapsed_ms'], ms + 100)
            self.assertEqual(play.checked_turn(history, turn['moves']), turn['moves'])


if __name__ == '__main__':
    unittest.main()
