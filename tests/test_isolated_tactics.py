import os
import time
import unittest

from tactical_proof import IsolatedTactics

ENGINE = 'tests.test_isolated_tactics:ScriptedTactics'


class ScriptedTactics:
    """Child-side stand-in for NativeTactics; the first history cell selects a behaviour."""

    def __init__(self, package):
        self.hoard = []

    def history(self, history, *, ms, **budgets):
        mode = tuple(history[0]) if history else (0, 0)
        if mode == (1, 1):
            time.sleep(60)  # ignores its deadline
        if mode == (3, 3):
            self.hoard.append(bytearray(512*2**20))
        return dict(status='UNKNOWN', native_verified=False, moves=[], certificate=None, reason='scripted',
                    pid=os.getpid(), ms=ms, background_worker_busy=mode == (2, 2))


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
        self.assertLess(time.perf_counter()-start, 1.0)
        self.assertEqual(hung['status'], 'UNKNOWN')
        self.assertIn('hard deadline', hung['reason'])
        self.assertEqual(self.tactics.stats['kills'], 1)
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

    def test_invalid_budget_rejected_in_parent(self):
        with self.assertRaises(ValueError):
            self.tactics.history([[0, 0]], ms=0)


if __name__ == '__main__':
    unittest.main()
