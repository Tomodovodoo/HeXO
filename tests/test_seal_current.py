from contextlib import closing
import unittest
from unittest.mock import Mock
from hexo import Game
from tools.seal_current import SealCurrent


class SealCurrentContract(unittest.TestCase):
    def adapter(self, moves):
        adapter = SealCurrent.__new__(SealCurrent)
        def reply(data, count, player, remaining, ms, out):
            for i, (q, r) in enumerate(moves):
                out[2*i], out[2*i+1] = q, r
            return len(moves)
        adapter.fn = Mock(side_effect=reply)
        return adapter

    def test_sequential_radius_and_restoration(self):
        with closing(Game([(0, 0)])) as game:
            before = game.cells
            self.assertEqual(self.adapter([(8, 0), (16, 0)])(game, 10), [(8, 0), (16, 0)])
            self.assertEqual(game.cells, before)
            self.assertEqual(game.remaining, 2)

    def test_invalid_second_restores_first(self):
        with closing(Game([(0, 0)])) as game:
            for moves in [[(1, 0), (1, 0)], [(1, 0)], [(1, 0), (50, 50)]]:
                with self.assertRaises(ValueError):
                    self.adapter(moves)(game, 10)
                self.assertEqual(game.cells, [[0, 0, 0]])
                self.assertEqual((game.player, game.remaining), (1, 2))

    def test_partial_turn_rejected_before_native(self):
        with closing(Game([(0, 0), (1, 0)])) as game:
            adapter = self.adapter([(0, 1)])
            with self.assertRaisesRegex(ValueError, "complete-turn"):
                adapter(game, 10)
            adapter.fn.assert_not_called()

    def test_large_coordinates_rejected_before_int32_conversion(self):
        game = Mock(cells=[(2**32, 0, 0)])
        adapter = self.adapter([(0, 1), (1, 0)])
        with self.assertRaisesRegex(ValueError, "range"):
            adapter(game, 10)
        adapter.fn.assert_not_called()

    def test_first_placement_win_and_no_extra_move(self):
        opening = [(0,0),(0,3),(1,3),(1,0),(2,0),(2,3),(3,3),(3,0),(4,0),(4,3),(5,4)]
        with closing(Game(opening)) as game:
            before = game.cells
            self.assertEqual(self.adapter([(5,0)])(game, 10), [(5,0)])
            self.assertEqual(self.adapter([(5,0),(6,0)])(game, 10), [(5,0)])
            self.assertEqual(game.cells, before)
            self.assertEqual(game.winner, -1)

    def test_real_adapter_rotated_first_wins(self):
        try:
            adapter = SealCurrent()
        except FileNotFoundError:
            self.skipTest("Optional pinned external adapter has not been built")
        opening = [(0,0),(0,3),(1,3),(1,0),(2,0),(2,3),(3,3),(3,0),(4,0),(4,3),(5,4)]
        for rotation in range(6):
            with closing(Game(opening)) as game:
                before = game.cells
                adapter.reset()
                moves = adapter(game, 100)
                self.assertEqual(game.cells, before)
                for move in moves:
                    self.assertEqual(game.winner, -1)
                    game.play(*move)
                self.assertEqual(game.winner, 0)
            opening = [(-r,q+r) for q,r in opening]


if __name__ == "__main__":
    unittest.main()
