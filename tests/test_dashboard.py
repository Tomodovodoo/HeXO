import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import dashboard
import dense_config
import dense_openings
from dashboard import bound_evaluation


class EvaluationBinding(unittest.TestCase):
    def test_latest_checkpoint_identity_and_legacy_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root/'checkpoints/0001/model.nnue'
            model.parent.mkdir(parents=True)
            model.write_bytes(b'first')
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            folder = root/'evaluation'
            folder.mkdir()
            def write(name, value):
                (folder/name).write_text(json.dumps(value), encoding='utf-8')
            write('status.json', {'wins': 12})
            self.assertIsNone(bound_evaluation(root, 1))
            write('report.json', {'config': {'candidate': str(model)}, 'identity': {str(model): digest}})
            self.assertEqual(bound_evaluation(root, 1)['wins'], 12)
            model.write_bytes(b'changed')
            self.assertIsNone(bound_evaluation(root, 1))
            write('status.json', {'wins': 5, 'candidate_sha256': hashlib.sha256(model.read_bytes()).hexdigest()})
            self.assertEqual(bound_evaluation(root, 1)['wins'], 5)
            self.assertIsNone(bound_evaluation(root, 2))


class OpeningBookPages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)/'synthetic'
        self.run.mkdir()
        self.write(self.run/'config.json', dict(schema=dense_config.SCHEMA, evaluation=dict(opening_suite='book')))
        self.a = [[0, 0], [1, 0], [0, 2]]
        self.b = [[0, 0], [2, 0], [0, 3]]
        specs = [([[0, 0]], None, 12, 3, 5, None, None),
                 (self.a, 'retired', 6, 3, 1, 'skew', 200),
                 (self.b, 'opening', 4, 0, 4, None, None),
                 ([[0, 0], [1, 0], [2, 0], [1, 1]], 'retired', 0, 0, 0, 'probability', 300),
                 ([[0, 0], [3, 0]], 'opening', 2, 0, 0, None, None)]
        self.nodes = []
        for i, (moves, status, games, p1, p2, reason, retired) in enumerate(specs):
            node = dense_openings.new_node(moves, 100+i)
            node.update(status=status, games=games, p1_wins=p1, p2_wins=p2, capped=games-p1-p2,
                        reason=reason, retired_at=retired, champion_probability=None if i==3 else i/10)
            self.nodes.append(node)
        self.write(self.run/'openings.json', dict(schema=dense_openings.SCHEMA, nodes=self.nodes))
        # Reflect and swap the two placements of P2's turn. These still match A.
        image = [[r, q] for q, r in self.a]
        image[1], image[2] = image[2], image[1]
        games = [dict(opening=image, winner=w, plies=p, pair=i//2, reason='cap' if w<0 else 'win')
                 for i, (w, p) in enumerate([(0, 10), (0, 20), (1, 30), (-1, 40)])]
        self.report = self.run/'evaluations/a-vs-b/report.json'
        self.write(self.report, dict(games=games))
        self.write(self.report.with_name('report-old.json'), dict(games=[
            dict(opening=self.a, winner=0, plies=12, pair=0, reason='win'),
            dict(opening=self.a, winner=-1, plies=18, pair=0, reason='cap')]))
        self.write(self.run/'evaluations/b-vs-c/report.json', dict(games=[
            dict(opening=self.b, winner=1, plies=p, pair=i//2, reason='win') for i, p in enumerate([8, 10, 12, 14])]))
        self.write(self.run/'evaluations/c-vs-d/report.json', dict(games=[
            dict(opening=specs[-1][0], winner=-1, plies=p, pair=0, reason='cap') for p in [50, 60]]))
        handler = type('BookHandler', (dashboard.Handler,), dict(runs=Path(self.tmp.name),
                       log_message=lambda *args: None))
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        self.addCleanup(lambda: dashboard._book_cache.pop(self.run, None))

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding='utf-8')

    def get(self, path='/api/book', **query):
        with urlopen(f'http://127.0.0.1:{self.server.server_port}{path}?'+urlencode(dict(run='synthetic', **query))) as response:
            return json.load(response)

    def test_statistics_and_canonical_prefix_matching(self):
        rows = {row['key']: row for row in self.get()['rows']}
        a = rows[self.nodes[1]['key']]
        self.assertEqual(a['report_games'], 6)
        self.assertEqual(a['median_plies'], 19)
        self.assertAlmostEqual(a['mean_plies'], 130/6)
        self.assertEqual(a['p1_win_rate'], .75)
        self.assertEqual(a['skew_z'], 1)
        self.assertAlmostEqual(a['decisive_share'], 4/6)
        root = rows[self.nodes[0]['key']]
        self.assertEqual(root['report_games'], 12)
        self.assertEqual(root['median_plies'], 16)
        self.assertAlmostEqual(root['mean_plies'], 284/12)
        self.assertAlmostEqual(root['skew_z'], -2/math.sqrt(8))
        empty, capped = (rows[self.nodes[i]['key']] for i in [3, 4])
        for key in ['median_plies', 'mean_plies', 'p1_win_rate', 'decisive_share', 'skew_z']:
            self.assertIsNone(empty[key])
        self.assertEqual(capped['decisive_share'], 0)
        self.assertIsNone(capped['p1_win_rate'])
        self.assertEqual(capped['median_plies'], 55)

    def test_paging_global_sort_and_filters(self):
        first = self.get(page_size=2)
        self.assertEqual((first['total'], first['page'], first['page_size']), (5, 1, 2))
        self.assertEqual([r['games'] for r in first['rows']], [12, 6])
        self.assertEqual([r['games'] for r in self.get(page=2, page_size=2)['rows']], [4, 2])
        self.assertEqual(self.get(page=9)['rows'], [])
        self.assertEqual(self.get()['page_size'], 50)
        self.assertEqual([r['games'] for r in self.get(sort='games', direction='asc')['rows']], [0, 2, 4, 6, 12])
        retired = self.get(status='retired', min_games=1, depth=3)['rows']
        self.assertEqual([r['key'] for r in retired], [self.nodes[1]['key']])
        self.assertEqual((retired[0]['reason'], retired[0]['retired_at']), ('skew', 200))
        self.assertEqual(len(self.get(status='retired', reason='probability')['rows']), 1)
        self.assertEqual(len(self.get(status='prefix')['rows']), 1)
        # B always loses as P1. A has both winning colours; caps alone cannot qualify.
        self.assertEqual([r['key'] for r in self.get(colour_decides=1, min_decisive=4)['rows']], [self.nodes[2]['key']])
        self.assertEqual(self.get(colour_decides=1, min_decisive=5)['rows'], [])
        self.assertEqual(len(self.get(key=self.nodes[1]['key'])['rows']), 1)

    def test_every_sort_and_nulls_last(self):
        rows = self.get()['rows']
        for key in dashboard.BOOK_SORTS:
            for direction in ['asc', 'desc']:
                with self.subTest(key=key, direction=direction):
                    base = sorted(rows, key=lambda r: r['key'])
                    expected = sorted([r for r in base if r[key] is not None], key=lambda r: r[key], reverse=direction=='desc')
                    expected += [r for r in base if r[key] is None]
                    actual = []
                    for page in [1, 2, 3]:
                        actual.extend(self.get(sort=key, direction=direction, page_size=2, page=page)['rows'])
                    self.assertEqual([r['key'] for r in actual], [r['key'] for r in expected])

    def test_cache_invalidates_on_report_book_and_directory_changes(self):
        first = dashboard.book_rows(self.run)
        with patch.object(dashboard, 'read_json', wraps=dashboard.read_json) as reader:
            self.assertIs(dashboard.book_rows(self.run), first)
            self.assertFalse(any(call.args[0] == self.run/'openings.json' or call.args[0] in
                                 (self.run/'evaluations').glob('*/report*.json') for call in reader.call_args_list))
        folder = self.run/'evaluations'
        folder_stat = folder.stat()
        data = json.loads(self.report.read_text())
        data['games'][0]['plies'] = 100
        old_stamp = self.report.stat().st_mtime_ns
        self.write(self.report, data)
        os.utime(self.report, ns=(old_stamp+1_000_000, old_stamp+1_000_000))
        os.utime(folder, ns=(folder_stat.st_atime_ns, folder_stat.st_mtime_ns))
        changed = dashboard.book_rows(self.run)
        self.assertIsNot(changed, first)
        self.assertAlmostEqual(changed[1]['mean_plies'], 220/6)
        self.nodes[1]['games'] = 8
        self.write(self.run/'openings.json', dict(nodes=self.nodes))
        self.assertEqual(self.get(key=self.nodes[1]['key'])['rows'][0]['games'], 8)
        self.write(self.run/'evaluations/new/report.json', dict(games=[dict(opening=self.a, plies=22)]))
        self.assertEqual(self.get(key=self.nodes[1]['key'])['rows'][0]['report_games'], 7)
        self.report.unlink()
        self.assertEqual(self.get(key=self.nodes[1]['key'])['rows'][0]['report_games'], 3)

    def test_default_frozen_suite_and_effective_evaluator_override(self):
        self.write(self.run/'config.json', dict(schema=dense_config.SCHEMA))
        self.write(self.run/'openings-standard-v1.json', dict(nodes=[self.nodes[1]]))
        rows = self.get()['rows']
        self.assertEqual([row['key'] for row in rows], [self.nodes[1]['key']])
        self.assertEqual(rows[0]['report_games'], 6)
        self.assertEqual(len(self.get('/api/book/dag')['nodes']), 1)
        self.write(self.run/'evaluator-status.json', dict(settings=dict(opening_suite='book')))
        self.assertEqual(self.get()['total'], 5)
        self.assertEqual(len(self.get('/api/book/dag')['nodes']), 5)
        (self.run/'evaluator-status.json').unlink()
        self.write(self.run/'config.json', dict(schema=dense_config.SCHEMA, evaluation=dict(opening_suite='custom')))
        self.write(self.run/'openings-custom.json', dict(nodes=[self.nodes[2]]))
        self.write(self.run/'evaluations/custom/report.json', dict(settings=dict(opening_suite='custom'),
                   games=[dict(opening=self.b, plies=9, winner=1, pair=0, reason='win')]))
        rows = self.get()['rows']
        self.assertEqual([row['key'] for row in rows], [self.nodes[2]['key']])
        self.assertEqual(rows[0]['report_games'], 1)
        self.assertEqual(rows[0]['median_plies'], 9)

    def test_dag_payload_and_request_validation(self):
        nodes = self.get('/api/book/dag')['nodes']
        self.assertEqual(len(nodes), len(self.nodes))
        for node in nodes:
            self.assertEqual(set(node), {'key', 'parents', 'depth', 'status', 'games', 'moves'})
            self.assertEqual(node['parents'], dense_openings.parents(node['moves']))
        self.assertEqual(nodes[0]['parents'], [])
        for query in [dict(page=0), dict(page_size=201), dict(sort='bad'), dict(direction='bad'),
                      dict(status='bad'), dict(min_games=-1), dict(depth='bad'), dict(min_decisive=0)]:
            with self.subTest(query=query), self.assertRaises(HTTPError) as error:
                self.get(**query)
            self.assertEqual(error.exception.code, 400)
        with self.assertRaises(HTTPError) as error:
            with urlopen(f'http://127.0.0.1:{self.server.server_port}/api/book?run=../outside'):
                pass
        self.assertEqual(error.exception.code, 404)


if __name__ == '__main__':
    unittest.main()
