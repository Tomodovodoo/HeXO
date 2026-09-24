import unittest

from curriculum import family, opening_for
from hexo import Game


class PositionCurriculum(unittest.TestCase):
    def test_legal_diverse_prefixes_and_disjoint_partitions(self):
        train, evaluation, validation = set(), set(), set()
        lengths, distant = set(), 0
        for held_out, destination in ((False, train), (True, evaluation)):
            for seed in range(256):
                moves = opening_for(seed, held_out)
                self.assertEqual(moves, opening_for(seed, held_out))
                fid = family(moves)
                destination.add(fid)
                self.assertEqual(fid % 10 == 9, held_out)
                if not held_out and fid % 5 == 0:
                    validation.add(fid)
                game = Game()
                try:
                    for move in moves:
                        self.assertTrue(game.legal(*move))
                        game.play(*move)
                        self.assertEqual(game.winner, -1)
                    self.assertEqual(game.remaining, 2)
                finally:
                    game.close()
                lengths.add(len(moves))
                distant += any(max(abs(q), abs(r), abs(q+r)) > 8 for q, r in moves)
        self.assertFalse(train & evaluation)
        self.assertGreater(len(train), 100)
        self.assertGreater(len(evaluation), 100)
        self.assertGreater(len(validation), 10)
        self.assertGreater(len(lengths), 3)
        self.assertGreater(distant, 50)

    def test_all_twelve_symmetries_share_the_colored_family(self):
        for seed in range(20):
            moves = opening_for(seed)
            expected = family(moves)
            for reflected in (False, True):
                variant = [(r, q) for q, r in moves] if reflected else list(moves)
                for _ in range(6):
                    self.assertEqual(family(variant), expected)
                    variant = [(-r, q+r) for q, r in variant]

    def test_owners_are_part_of_the_family(self):
        first = [(0, 0), (1, 0), (0, 2), (3, 0), (0, 4)]
        changed_owners = [(0, 0), (3, 0), (0, 2), (1, 0), (0, 4)]
        self.assertNotEqual(family(first), family(changed_owners))
