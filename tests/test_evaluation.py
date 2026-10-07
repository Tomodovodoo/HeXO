"""Paired match statistics: a rating comes only from complete colour pairs."""
import math
import unittest

from legacy.train import paired_metrics


class PairedEvaluation(unittest.TestCase):
    def test_pair_statistics_and_uncensored_rating(self):
        records = [{'seed': p, 'challenger_color': c, 'winner': c if p < 3 else 1-c}
                   for p in range(4) for c in (0, 1)]
        result = paired_metrics(records)
        self.assertTrue(result['rated'])
        self.assertEqual((result['wins'], result['losses']), (6, 2))
        self.assertEqual(result['opening_pair_p'], 5/16)
        self.assertAlmostEqual(result['elo_delta'], 400*math.log10(6.5/2.5))
        self.assertLess(result['elo_delta_95pct_open'][0], 0)
        self.assertIsNone(result['elo_delta_95pct_open'][1])
        for partial in (records[:-1], records[:-1]+[{**records[-1], 'winner': -1}]):
            metric = paired_metrics(partial, 8)
            self.assertFalse(metric['rated'])
            self.assertIsNone(metric['elo_delta'])
            self.assertIsNone(metric['win_rate'])
        with self.assertRaises(ValueError):
            paired_metrics([records[0], records[0]], 8)


if __name__ == '__main__':
    unittest.main()
