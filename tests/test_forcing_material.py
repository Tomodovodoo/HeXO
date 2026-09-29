import unittest

from forcing_material import HIGH, LOW, forcing_material, gate_level, live_windows, worth_solving
from hexo import Game


class ForcingMaterial(unittest.TestCase):
    def test_counts_live_windows_per_side(self):
        # First player holds (0,0),(1,0),(2,0); second holds (0,4),(1,4) and is to move.
        game = Game([(0, 0), (0, 4), (1, 4), (1, 0), (2, 0)])
        mine, theirs = live_windows(game), live_windows(game, 0)
        self.assertEqual(theirs[3], 4)  # the four windows along the q-axis holding all three stones
        self.assertEqual(sum(theirs[4:]), 0)
        self.assertEqual(sum(mine[3:]), 0)
        self.assertEqual(mine[2], 5)
        self.assertFalse(worth_solving(game))
        self.assertTrue(worth_solving(game, 0))
        self.assertEqual(forcing_material(game, 0), 2.0*4+0.25*theirs[2])

    def test_opponent_stone_kills_windows(self):
        game = Game([(0, 0), (0, 4), (1, 4), (1, 0), (2, 0), (3, 0), (-1, 0)])
        # The second player's blocks at (3,0) and (-1,0) leave no live window holding the three.
        self.assertEqual(sum(live_windows(game, 0)[3:]), 0)
        self.assertFalse(worth_solving(game, 0))

    def test_gate_level(self):
        self.assertIsNone(gate_level(LOW-.25))
        self.assertEqual((gate_level(LOW), gate_level((LOW+HIGH)/2), gate_level(HIGH), gate_level(HIGH*2)),
                         (0., .5, 1., 1.))

    def test_empty_board(self):
        self.assertEqual(live_windows(Game()), [0]*7)
        self.assertEqual(forcing_material(Game()), 0)


if __name__ == '__main__':
    unittest.main()
