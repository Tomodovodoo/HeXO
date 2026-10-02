"""The WebAssembly tactical solver (web/engine/tactical.wasm via tactical.mjs) answers as the native library does."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from tactical_proof import NativeTactics, library
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
