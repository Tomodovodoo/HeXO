"""The WebAssembly tactical solver (web/engine/tactical.wasm via tactical.mjs) answers as the native library does."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from tactical_proof import NativeTactics, library, independent_verify
from tests.test_tactical_proof import FIXTURE, IMMEDIATE, NO_THREAT, OPEN_THREE, TWO_TURN

ROOT = Path(__file__).resolve().parents[1]
WASM = ROOT/'web'/'engine'/'tactical.wasm'
RUNNER = ROOT/'tests'/'web'/'tactical.mjs'
NODE = shutil.which('node')
CONTROL = FIXTURE['positions'][FIXTURE['control']]
QUERIES = [
    (OPEN_THREE, dict(nodes=100001, idtt_nodes=1000)),
    (IMMEDIATE, dict(nodes=2500)),
    (CONTROL, dict(nodes=FIXTURE['proving_nodes'][FIXTURE['control']])),
    (FIXTURE['positions']['1790600287230040:30:248'], dict(nodes=1300)),
    (FIXTURE['positions']['1790600287230040:17:145'], dict(nodes=300)),
    (TWO_TURN, dict(nodes=2000, attacker='opponent')),
    (NO_THREAT, dict(nodes=3000, attacker='opponent')),
    (CONTROL, dict(nodes=2000, idtt_nodes=500)),
]
FIELDS = ('status', 'native_verified', 'moves', 'proof_turns', 'nodes_used', 'certificate', 'reason', 'attacker',
          'idtt_verdict', 'scope', 'budget', 'gate_score')


def wasm_results(queries):
    lines = ''.join(json.dumps(dict(history=history, **options))+'\n' for history, options in queries)
    output = subprocess.run([NODE, str(RUNNER)], input=lines, capture_output=True, text=True, check=True).stdout
    return [json.loads(line) for line in output.splitlines()]


@unittest.skipUnless(NODE and WASM.exists() and library().exists(), 'needs node, web/engine/tactical.wasm and the native library')
class WebTacticalParity(unittest.TestCase):
    def test_defender_roots_and_exact_graph_premises_match_native(self):
        native = NativeTactics()
        history = OPEN_THREE + [[-1, 0], [2, 1]]
        known = [dict(history=history+[list(p)], winner=0, plies=23)
                 for p in ((-3, 0), (-2, 0), (3, 0), (4, 0))]
        queries = [(history, dict(attacker='defender', known=known, nodes=1, ms=5000)),
                   (history+[[0, 1]], dict(attacker='defender', nodes=1, ms=5000)),
                   (NO_THREAT, dict(attacker='defender', nodes=1, ms=5000))]
        for (history, options), web in zip(queries, wasm_results(queries)):
            local = native.history(history, **options)
            self.assertEqual({k: web[k] for k in FIELDS}, {k: local[k] for k in FIELDS})
            if web['native_verified']:
                self.assertEqual(independent_verify(web['certificate'], history, attacker='defender',
                                                    known=options.get('known', [])), 'PROVEN_LOSS')

    def test_resumed_browser_slices_report_bounds_and_fresh_work(self):
        history = FIXTURE['positions']['1790600287230040:30:248']
        options = dict(nodes=512, ms=20000, bounds=True, resume=True)
        first, continued, cached = wasm_results([
            (history, dict(options, table_mb=4)),
            (history, dict(options, table_mb=8)),
            (history, dict(options, table_mb=8)),
        ])
        self.assertEqual(first['status'], 'UNKNOWN')
        self.assertFalse(first['native_verified'])
        self.assertEqual(first['proof_numbers']['scope'], 'wide-forcing')
        self.assertFalse(first['proof_numbers']['game_exact'])
        self.assertEqual(continued['status'], 'PROVEN_WIN', continued)
        self.assertTrue(continued['resident_reused'])
        self.assertFalse(continued['cache_hit'])
        self.assertLess(continued['nodes_fresh'], first['nodes_fresh'])
        self.assertEqual(independent_verify(continued['certificate'], history), 'PROVEN_WIN')
        self.assertTrue(cached['cache_hit'])
        self.assertEqual(cached['nodes_fresh'], 0)
        self.assertGreater(cached['nodes_used'], 0)

    def test_browser_resume_requires_a_resident_table(self):
        with self.assertRaises(subprocess.CalledProcessError) as error:
            wasm_results([(NO_THREAT, dict(nodes=1, resume=True))])
        self.assertIn('positive table_mb', error.exception.stderr)

    def test_wasm_matches_native(self):
        native = NativeTactics()
        queries = [(history, dict(options, ms=20000)) for history, options in QUERIES]
        expected = [native.history(history, **options) for history, options in queries]
        actual = wasm_results(queries)
        self.assertEqual(len(actual), len(expected))
        for (history, options), want, got in zip(queries, expected, actual):
            with self.subTest(moves=len(history), **options):
                self.assertEqual({f: got[f] for f in FIELDS}, {f: want[f] for f in FIELDS})
                self.assertEqual((got['background_worker_busy'], got['last_worker_completion']), (False, None))
                self.assertEqual(got['build_hash'], hashlib.sha256(WASM.read_bytes()).hexdigest())
        statuses = [(r['status'], r['attacker']) for r in expected]
        self.assertGreaterEqual(statuses.count(('PROVEN_WIN', 'mover')), 3)
        self.assertIn(('PROVEN_WIN', 'opponent'), statuses)
        self.assertIn(('UNKNOWN', 'opponent'), statuses)
        self.assertIn(('UNKNOWN', 'mover'), statuses)


if __name__ == '__main__':
    unittest.main()
