import unittest
from unittest.mock import patch

from hexo import Game
from strix_root import StrixRoot

MOVES = [(0,0),(0,3),(1,3),(1,0),(2,0),(3,3),(4,3),(3,0),(4,0),(5,3),(6,3)]


class Client:
    executable_sha256 = "test-client"
    def __init__(self, clock, status="UNKNOWN", pv=(), spent=.02):
        self.clock, self.status, self.pv, self.spent = clock, status, pv, spent
        self.calls = []
        self.closed = False

    def solve(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.clock[0] += self.spent
        return dict(status=self.status, pv=self.pv)

    def close(self):
        self.closed = True


class StrixRootTests(unittest.TestCase):
    def position(self, moves=MOVES):
        game = Game(moves)
        self.addCleanup(game.close)
        return game

    def test_remaining_native_budget_and_actual_combined_time(self):
        for status in ("UNKNOWN", "NO_FORCING_WIN_WITHIN_SCOPE"):
            clock = [0.]
            client = Client(clock, status)
            probe = StrixRoot(20, 100, client, lambda: clock[0])
            game = self.position()
            def native(ms, **kwargs):
                self.assertLessEqual(ms, 80)
                self.assertGreaterEqual(ms, 79)
                self.assertEqual(kwargs, {"width": 16})
                clock[0] += ms/1000
                return dict(moves=[(5,0)], score=10000000, nodes=1, depth=1)
            with patch.object(game, "search", native):
                result = probe.search(game, 100, width=16)
            self.assertEqual(result["source"], "native")
            self.assertAlmostEqual(result["elapsed_ms"], clock[0]*1000)
            self.assertEqual(client.calls[0][1]["timeout_s"], .02)
            self.assertEqual(client.calls[0][1]["nodes"], 1000)
            self.assertFalse(result["strix"]["selected"])
            probe.close()
            self.assertTrue(client.closed)

    def test_first_win_two_stones_and_conditional_phase_restore(self):
        for prefix, extra, pv in ((11, [], [(5,0)]), (9, [], [(2,3),(5,3)]),
                                  (11, [(7,1)], [(5,0)])):
            game = self.position(MOVES[:prefix]+extra)
            before = (game.key, game.state(), game.features())
            clock = [0.]
            client = Client(clock, "REFERENCE_WIN_WITHIN_SCOPE", pv)
            probe = StrixRoot(20, 100, client, lambda: clock[0])
            with patch.object(game, "search", side_effect=AssertionError("native should not run")):
                result = probe.search(game, 100)
            self.assertEqual(result["moves"], pv)
            self.assertIsNone(result["score"])
            self.assertFalse(result["independent_proof"])
            self.assertEqual(client.calls[0][0][2], game.remaining)
            self.assertEqual((game.key, game.state(), game.features()), before)

    def test_invalid_pv_falls_back_and_restores(self):
        for pv in ([(100,100)], [(5,0),(6,0)], [(0,0)], [(7,1)]):
            game = self.position()
            before = (game.key, game.state(), game.features())
            clock = [0.]
            probe = StrixRoot(20, 100, Client(clock, "REFERENCE_WIN_WITHIN_SCOPE", pv), lambda: clock[0])
            with patch.object(game, "search", return_value=dict(moves=[(5,0)])) as native:
                result = probe.search(game, 100)
            native.assert_called_once()
            self.assertFalse(result["strix"]["selected"])
            self.assertEqual(probe.counts["invalid_pv"], 1)
            self.assertEqual((game.key, game.state(), game.features()), before)

    def test_overrun_is_reported_and_budgets_rejected(self):
        for budget in (0, -1, 100, float("nan")):
            with self.assertRaises(ValueError):
                StrixRoot(budget, 100)
        for kwargs in ({"nodes": 0}, {"depth": 0}, {"nodes": 1 << 64}):
            with self.assertRaises(ValueError):
                StrixRoot(20, 100, **kwargs)
        clock = [0.]
        probe = StrixRoot(20, 100, Client(clock, spent=.11), lambda: clock[0])
        game = self.position()
        with patch.object(game, "search", return_value=dict(moves=[(5,0)])) as native:
            result = probe.search(game, 100)
            native.assert_called_once_with(1)
        self.assertTrue(result["strix"]["overrun"])
        self.assertEqual(probe.counts["overruns"], 1)
        self.assertEqual(result["strix"]["native_budget_ms"], 1)

    def test_exhausted_budget_skips_probe(self):
        clock = [1.]
        client = Client(clock)
        probe = StrixRoot(20, 100, client, lambda: clock[0])
        game = self.position()
        with patch.object(game, "search", return_value=dict(moves=[(5,0)])):
            result = probe.search(game, 100, start=.8)
        self.assertFalse(client.calls)
        self.assertEqual(probe.counts["skipped"], 1)
        self.assertEqual(probe.counts["calls"], 0)
        self.assertTrue(result["strix"]["overrun"])


if __name__ == "__main__":
    unittest.main()
