import copy
import math
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from legacy.evaluate import verify_trace, verify_runtime, sha, publish_failure, freeze
from hexo import Game
from legacy.train import paired_metrics


class PairedEvaluation(unittest.TestCase):
    def test_invalid_width_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'evaluation'
            for width in (1, 129):
                args = SimpleNamespace(output=output, games=2, ms=1, workers=1, max_stones=5, width=width)
                with self.assertRaisesRegex(ValueError, 'width in 2..128'):
                    freeze(args)
                self.assertFalse(output.exists())

    def test_runtime_dependency_changes_and_failed_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dependency = root/'runtime.dll'
            dependency.write_bytes(b'original')
            identity = {str(dependency): sha(dependency)}
            verify_runtime(identity, [dependency])
            with self.assertRaisesRegex(ValueError, 'missing, relocated or changed'):
                verify_runtime(identity, [])
            dependency.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify_runtime(identity, [dependency])
            dependency.unlink()
            with self.assertRaises(ValueError):
                verify_runtime(identity, [dependency])
            (root/'provenance.json').write_text(json.dumps({'model_input_sha256': {'candidate': 'abc', 'reference': 'def'}}))
            (root/'status.json').write_text(json.dumps({'completed': 3, 'total': 8}))
            publish_failure(root, ValueError('dependency changed'))
            status = json.loads((root/'status.json').read_text())
            self.assertEqual(status['stage'], 'failed')
            self.assertEqual(status['candidate_sha256'], 'abc')
            self.assertEqual(status['reference_sha256'], 'def')
            self.assertEqual(status['completed'], 3)
            publish_failure(root, KeyboardInterrupt())
            status = json.loads((root/'status.json').read_text())
            self.assertEqual(status['stage'], 'failed')
            self.assertTrue(status['interrupted'])
            self.assertEqual(status['candidate_sha256'], 'abc')
            self.assertEqual(status['completed'], 3)

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

    def test_ordered_turn_trace_replay_and_corruption(self):
        opening = [(0, 0), (8, 0), (16, 0)]
        game = Game(opening)
        try:
            turn = {'ply': 3, 'player': game.player, 'remaining': game.remaining,
                    'result': {'moves': [(24, 0), (32, 0)]}}
            for move in turn['result']['moves']:
                game.play(*move)
            record = {'opening': opening, 'cells': game.cells, 'winner': -1,
                      'reason': 'truncated', 'search_trace': [turn]}
            verify_trace(record)
            for key, value in (('winner', 0), ('reason', 'six-in-a-row')):
                bad = copy.deepcopy(record)
                bad[key] = value
                with self.assertRaises(ValueError):
                    verify_trace(bad)
            bad = copy.deepcopy(record)
            bad['search_trace'][0]['result']['moves'].reverse()
            with self.assertRaises(ValueError):
                verify_trace(bad)
            bad = copy.deepcopy(record)
            bad['search_trace'][0]['remaining'] = 1
            with self.assertRaises(ValueError):
                verify_trace(bad)
        finally:
            game.close()

    def test_first_stone_win_ends_turn_immediately(self):
        opening = [(0,0), (0,3), (1,3), (1,0), (2,0), (3,3),
                   (4,3), (3,0), (4,0), (5,3), (6,3)]
        game = Game(opening)
        try:
            turn = {'ply': len(opening), 'player': game.player, 'remaining': 2,
                    'result': {'moves': [(5,0)]}}
            game.play(5,0)
            record = {'opening': opening, 'cells': game.cells, 'winner': 0,
                      'reason': 'six-in-a-row', 'search_trace': [turn]}
            verify_trace(record)
            turn['result']['moves'].append((6,0))
            with self.assertRaisesRegex(ValueError, 'first-stone win'):
                verify_trace(record)
        finally:
            game.close()


if __name__ == '__main__':
    unittest.main()
