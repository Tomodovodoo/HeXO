"""Exact leaf losses from development games, independently checked by windows."""
import unittest

from hexo import Game
from tests.reference import Reference, has_cover


# Prefixes of games 0/5/7 in the development arena seeded 20260924.
FORKS = [
    [(0,0),(-2,0),(-1,0),(-3,0),(-3,1),(-2,1),(-3,2),(0,-1),(-2,-1),
     (-1,-1),(-1,-2),(0,-2),(0,-3),(0,-4),(0,2),(-3,-4),(-3,-1),(-1,-3),
     (-3,-2),(-1,-4),(-1,2),(-2,4),(-3,4),(-7,4),(-6,3),(-5,2),(-5,4),
     (-6,5),(-6,4),(-3,5),(-1,3),(-4,6),(2,0),(-2,3),(0,4),(-1,4),(-2,5)],
    [(0,0),(1,0),(0,-1),(-2,0),(-1,-1),(-3,1),(0,1),(-2,1),(-1,2),
     (4,-3),(5,-4),(-2,2),(3,-2),(-3,2),(-3,3),(-1,0),(-1,1),(-1,-2),(-1,3)],
    [(0,0),(1,-2),(1,1),(1,0),(2,0),(-1,0),(0,-1),(-2,1),(-2,2),
     (2,-3),(3,0),(3,-4),(-2,3),(-2,4),(-2,-1),(-1,2),(-3,4),(0,1),
     (3,1),(-3,3),(-3,2),(-3,-1),(-1,-1),(-4,-1),(2,-1),(-1,1),
     (-3,5),(-4,2),(2,1),(-5,2),(1,2),(-4,3),(-1,3),(-5,3),(1,3)],
]


class LeafTactics(unittest.TestCase):
    def position(self, moves):
        game, reference = Game(moves), Reference()
        self.addCleanup(game.close)
        for point in moves:
            reference.play(*point)
        return game, reference

    def test_admitted_forks_are_wins_at_depth_one(self):
        for moves in FORKS:
            with self.subTest(stones=len(moves)):
                game, reference = self.position(moves)
                before = (game.key, game.state(), game.features())
                side = game.player
                self.assertFalse(reference.completions(side, game.remaining))
                result = game.search(1000, depth=1, width=16)
                self.assertEqual((result["depth"], result["score"]), (1, 10000000))
                self.assertEqual((game.key, game.state(), game.features()), before)
                for point in result["moves"]:
                    reference.play(*point)
                self.assertEqual(reference.winner, -1)
                self.assertFalse(reference.completions(reference.player, reference.remaining))
                self.assertFalse(has_cover(reference.completions(side), reference.remaining))

    def test_counterwin_precedes_uncoverable_opponent_threats(self):
        moves = FORKS[0] + [(0,3),(-3,3)]
        for index, q in zip((3,4,7,8,11), range(5,10)):
            moves[index] = (q,0)
        game, reference = self.position(moves)
        side = game.player
        self.assertFalse(has_cover(reference.completions(1-side), game.remaining))
        self.assertTrue(reference.completions(side, game.remaining))
        result = game.search(1000, depth=1, width=16)
        self.assertEqual(result["score"], 10000000)
        self.assertEqual(len(result["moves"]), 1)
        reference.play(*result["moves"][0])
        self.assertEqual(reference.winner, side)

    def test_shared_covers_are_not_mistaken_for_unavoidable_loss(self):
        game, reference = self.position([(0,0),(0,5),(3,3),(1,0),(2,0),(-3,5),(5,-3)])
        side = game.player
        result = game.search(1000, depth=1, width=16)
        self.assertLess(abs(result["score"]), 10000000)
        for point in result["moves"]:
            reference.play(*point)
        threats = reference.completions(side)
        self.assertGreaterEqual(len(threats), 3)
        self.assertTrue(has_cover(threats, reference.remaining))

    def test_one_block_leaves_the_free_placement_unresolved(self):
        game, reference = self.position([(0,0),(-1,0),(0,5),(1,0),(2,0),(3,3),(-3,5)])
        side = game.player
        result = game.search(1000, depth=1, width=16)
        self.assertLess(abs(result["score"]), 10000000)
        for point in result["moves"]:
            reference.play(*point)
        threats = reference.completions(side)
        self.assertTrue(threats)
        self.assertTrue(has_cover(threats, 1))
        self.assertEqual(reference.remaining, 2)


if __name__ == "__main__":
    unittest.main()
