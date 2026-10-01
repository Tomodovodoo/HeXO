import json
import os
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

from tactical_proof import IsolatedTactics

ENGINE = 'tests.test_isolated_tactics:ScriptedTactics'


class ScriptedTactics:
    """Child-side stand-in for NativeTactics; the first history cell selects a behaviour."""

    def __init__(self, package):
        if Path(package).name == 'slow-start':
            time.sleep(60)
        self.hoard = [bytearray(512*2**20)] if Path(package).name == 'greedy-start' else []

    def history(self, history, *, ms, **budgets):
        mode = tuple(history[0]) if history else (0, 0)
        if mode == (1, 1):
            time.sleep(60)  # ignores its deadline
        if mode == (3, 3):
            self.hoard.append(bytearray(512*2**20))
        certificate = dict(version=1, nodes=[dict(kind='immediate_win', action=[[5, 0]])]) if mode == (5, 5) else None
        return dict(status='UNKNOWN', native_verified=False, moves=[], certificate=certificate, reason='scripted',
                    pid=os.getpid(), ms=ms, background_worker_busy=mode == (2, 2),
                    padding='x'*(17*2**20) if mode == (4, 4) else '')


class Isolation(unittest.TestCase):
    def setUp(self):
        self.tactics = IsolatedTactics(engine=ENGINE, grace_ms=100, memory_mb=256)

    def tearDown(self):
        self.tactics.close()

    def test_answers_and_reuses_child(self):
        first = self.tactics.history([[0, 0]], ms=10000)
        second = self.tactics.history([[0, 0]], ms=500)
        self.assertEqual(first['reason'], 'scripted')
        self.assertEqual(first['pid'], second['pid'])
        self.assertLess(second['elapsed_ms'], 200)
        self.assertLessEqual(second['ms'], 500)
        self.assertEqual(self.tactics.stats['spawns'], 1)

    def test_hard_deadline_kills_and_replaces_child(self):
        pid = self.tactics.history([[0, 0]], ms=10000)['pid']
        start = time.perf_counter()
        hung = self.tactics.history([[1, 1]], ms=200)
        self.assertLess(time.perf_counter()-start, 0.45)
        self.assertEqual(hung['status'], 'UNKNOWN')
        self.assertIn('hard deadline', hung['reason'])
        self.assertEqual((hung['attacker'], hung['build_hash'], hung['nodes_used']), ('mover', None, 0))
        self.assertEqual(self.tactics.stats['kills'], 1)
        self.assertNotEqual(self.tactics.history([[0, 0]], ms=10000)['pid'], pid)

    def test_abort_ends_a_running_query_and_the_next_one_proceeds(self):
        pid = self.tactics.history([[0, 0]], ms=10000)['pid']
        results = []
        query = threading.Thread(target=lambda: results.append(self.tactics.history([[1, 1]], ms=20000)))
        query.start()
        time.sleep(.3)
        start = time.perf_counter()
        self.tactics.abort()
        query.join(5)
        self.assertLess(time.perf_counter()-start, 1)
        self.assertEqual(results[0]['status'], 'UNKNOWN')
        self.assertNotEqual(self.tactics.history([[0, 0]], ms=10000)['pid'], pid)

    def test_abandoned_native_work_replaces_child(self):
        pid = self.tactics.history([[2, 2]], ms=10000)['pid']
        self.assertEqual(self.tactics.stats['kills'], 1)
        self.assertNotEqual(self.tactics.history([[0, 0]], ms=10000)['pid'], pid)

    def test_memory_cap_ends_child(self):
        self.tactics.history([[0, 0]], ms=10000)
        result = self.tactics.history([[3, 3]], ms=10000)
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertIn('exited', result['reason'])
        self.assertEqual(self.tactics.history([[0, 0]], ms=10000)['reason'], 'scripted')

    def test_worker_that_never_starts_is_replaced(self):
        stuck = IsolatedTactics('slow-start', engine=ENGINE, startup_ms=300)
        try:
            self.assertIn('starting', stuck.history([[0, 0]], ms=100)['reason'])
            time.sleep(0.25)
            self.assertIn('replaced', stuck.history([[0, 0]], ms=1000)['reason'])
            self.assertEqual(stuck.stats['kills'], 1)
        finally:
            stuck.close()

    def test_memory_cap_applies_before_the_engine_loads(self):
        greedy = IsolatedTactics('greedy-start', engine=ENGINE, memory_mb=256)
        try:
            self.assertIn('MemoryError', greedy.history([[0, 0]], ms=10000)['reason'])
        finally:
            greedy.close()

    def test_oversized_request_is_not_sent(self):
        with patch('tactical_proof.REQUEST_LIMIT', 8*2**20):
            result = self.tactics.history([[0, 0]], ms=1000, certificate=dict(padding='x'*(9*2**20)))
        self.assertEqual(result['reason'], 'request size limit')
        self.assertEqual(self.tactics.history([[0, 0]], ms=1000)['reason'], 'scripted')

    def test_oversized_response_is_discarded(self):
        self.tactics.close()
        with patch('tactical_proof.RESPONSE_LIMIT', 16*2**20):
            self.tactics = IsolatedTactics(engine=ENGINE, grace_ms=100, memory_mb=256)
            pid = self.tactics.history([[0, 0]], ms=10000)['pid']
            self.assertEqual(self.tactics.history([[4, 4]], ms=10000)['reason'], 'response size limit')
        self.assertNotEqual(self.tactics.history([[0, 0]], ms=10000)['pid'], pid)

    def test_certificate_arrives_undecoded(self):
        result = self.tactics.history([[5, 5]], ms=10000)
        self.assertIsNone(result['certificate'])
        self.assertEqual(json.loads(result['certificate_json'])['nodes'][0]['kind'], 'immediate_win')
        self.assertNotIn('certificate_json', self.tactics.history([[0, 0]], ms=10000))

    def test_invalid_budget_rejected_in_parent(self):
        with self.assertRaises(ValueError):
            self.tactics.history([[0, 0]], ms=0)


if __name__ == '__main__':
    unittest.main()
