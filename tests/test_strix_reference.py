import unittest
from pathlib import Path

from strix_reference import StrixReference, validate_pv


class SequentialReplay(unittest.TestCase):
    def test_first_stone_win_stops_turn(self):
        stones = [[q, 0, "P1"] for q in range(5)]
        self.assertTrue(validate_pv(stones, "P1", 2, [[5, 0]]))
        self.assertFalse(validate_pv(stones, "P1", 2, [[5, 0], [6, 0]]))

    def test_second_stone_and_phase(self):
        stones = [[q, 0, "P1"] for q in range(4)]
        self.assertTrue(validate_pv(stones, "P1", 2, [[4, 0], [5, 0]]))
        self.assertFalse(validate_pv(stones, "P1", 1, [[4, 0], [5, 0]]))

    def test_no_illegal_radius_or_incomplete_pv(self):
        self.assertFalse(validate_pv([[0, 0, "P1"]], "P2", 2, [[9, 0]]))
        self.assertFalse(validate_pv([[0, 0, "P1"]], "P2", 2, [[8, 0], [16, 0]]))


class NativeReference(unittest.TestCase):
    def setUp(self):
        self.reference = StrixReference()
        if not Path(self.reference.executable).exists():
            self.skipTest("build tools/strix first")
        self.addCleanup(self.reference.close)

    def test_persistent_process_phase_and_scoped_negative(self):
        result = self.reference.solve([[q, 0, "P1"] for q in range(4)], "P1", 2)
        self.assertEqual(result["status"], "REFERENCE_WIN_WITHIN_SCOPE")
        self.assertTrue(result["pv_replay_valid"])
        pid = self.reference.process.pid
        result = self.reference.solve([[0, 0, "P1"]], "P2", 2)
        self.assertEqual(self.reference.process.pid, pid)
        self.assertEqual(result["status"], "NO_FORCING_WIN_WITHIN_SCOPE")
        self.assertFalse(result["independently_verified_proof"])
        result = self.reference.solve([[q, 0, "P1"] for q in range(5)], "P1", 1, wide=True)
        self.assertEqual(result["status"], "REFERENCE_WIN_WITHIN_SCOPE")
        self.assertEqual(len(result["pv"]), 1)
        self.assertEqual(result["scope"]["generator"], "wide")

    def test_timeout_unknown_and_process_recovers(self):
        result = self.reference.solve([[q, 0, "P1"] for q in range(3)], "P1", 2,
                                      depth=30, nodes=10**12, timeout_s=1e-12)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["reason"], "wall_timeout")
        self.assertIsNone(self.reference.process)
        result = self.reference.solve([[q, 0, "P1"] for q in range(5)], "P1", 1)
        self.assertEqual(result["status"], "REFERENCE_WIN_WITHIN_SCOPE")

    def test_invalid_snapshots_do_not_reach_solver(self):
        for stones in ([[0, 0, "P1"], [0, 0, "P2"]], [[1 << 31, 0, "P1"]],
                       [[q, 0, "P1"] for q in range(6)]):
            with self.assertRaises(ValueError):
                self.reference.solve(stones, "P1", 2)
        self.assertIsNone(self.reference.process)

    def test_node_and_coordinate_limits_are_unknown(self):
        result = self.reference.solve([[q, 0, "P1"] for q in range(3)], "P1", 2, nodes=0)
        self.assertEqual(result["status"], "UNKNOWN")
        result = self.reference.solve([[0, 0, "P1"], [1000000, 1000000, "P2"]], "P1", 2)
        self.assertEqual(result["status"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
