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


class GameLengths(unittest.TestCase):
    def test_repeated_mature_run_requests_do_not_reread_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            shards = [(run/'shards'/str(i)/'manifest.json', dict(origin='actor')) for i in range(8200)]
            with patch('dashboard.dense_manifests', return_value=shards), \
                 patch.object(Path, 'stat') as stat, \
                 patch.object(Path, 'read_text', return_value='[{"moves": [[0, 0]], "winner": 0}]') as read:
                stat.return_value.st_mtime_ns = 1
                self.assertEqual(dashboard.game_lengths(run, 0)['games'], 8200)
                self.assertEqual(read.call_count, 8200)
                self.assertEqual(dashboard.game_lengths(run, 0)['games'], 8200)
                self.assertEqual(read.call_count, 8200)

    def test_actor_windows_start_types_endings_and_recorded_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            def shard(name, created, episodes, **fields):
                folder = run/'shards'/name
                folder.mkdir(parents=True)
                (folder/'manifest.json').write_text(json.dumps(dict(created_at=created, **fields)), encoding='utf-8')
                (folder/'episodes.json').write_text(json.dumps(episodes), encoding='utf-8')
            def episode(plies, winner=0, **fields):
                return dict(moves=[[0, 0]]*plies, winner=winner, **fields)
            shard('1', 9999, [episode(10), episode(22, 1, origin='book', reason='proven',
                                                  adjudicated=dict(ply=14, line_plies=8)),
                              episode(44, -1, origin='restart', reason='span')], origin='actor')
            shard('2', 1, [episode(400)], origin='actor')
            shard('3', 9999, [episode(999)], identity=dict(source='converted corpus'))
            with patch('dashboard.time.time', return_value=10000):
                result = dashboard.game_lengths(run, 1)
                self.assertEqual(result['games'], 3)
                self.assertEqual(result['endings'], dict(win=1, proven=1, capped=1))
                self.assertEqual(result['median'], 22)
                self.assertAlmostEqual(result['mean'], 76/3)
                self.assertEqual([(b['win'], b['proven'], b['capped']) for b in result['bins']],
                                 [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
                self.assertEqual(dashboard.game_lengths(run, 1, 'book')['mean'], 22)
                self.assertEqual(dashboard.game_lengths(run, 1, 'restart')['mean'], 44)
                self.assertEqual(dashboard.game_lengths(run, 1, 'selfplay')['mean'], 10)
                self.assertEqual(dashboard.game_lengths(run, 0)['median'], 33)
                shard('4', 10000, [episode(30)])  # Legacy actor origin, newly published after the first request.
                self.assertEqual(dashboard.game_lengths(run, 1)['games'], 4)
            with patch('dashboard.time.time', return_value=20000):
                self.assertEqual(dashboard.game_lengths(run, 1)['bins'], [])
                self.assertIsNone(dashboard.game_lengths(run, 1)['mean'])


class TacticalResults(unittest.TestCase):
    def test_dense_run_groups_by_opening_and_both_models(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            shard = run/'shards/000001/manifest.json'
            shard.parent.mkdir(parents=True)
            (run/'league.json').write_text(json.dumps(dict(checkpoints=[dict(id='main/000020',
                                                                      ema_sha256='b'*64)])), encoding='utf-8')
            self.assertEqual(dashboard.dense_run(run, {})['tactical'], [])
            cases = [dict(opening='opening-a', expected_winner=1, winner=winner,
                          actors={'0': 'a'*64, '1': p2}, p2_value=value, source='manual log')
                     for winner, p2, value in [(1, 'b'*64, .8), (0, 'b'*64, .6),
                                               (-1, 'b'*64, None), (1, 'c'*64, .9)]]
            shard.write_text(json.dumps(dict(identity=dict(actor_sha256='a'*64, checkpoint='main/000010'),
                                             tactical=cases)), encoding='utf-8')
            rows = dashboard.dense_run(run, {})['tactical']
            self.assertEqual(len(rows), 2)
            first = next(row for row in rows if row['p2_sha256'] == 'b'*64)
            self.assertEqual((first['p1_model'], first['p2_model']), ('main/000010', 'main/000020'))
            self.assertEqual(next(row for row in rows if row['p2_sha256'] == 'c'*64)['p2_model'], 'c'*12)
            self.assertEqual((first['games'], first['conversions'], first['opposite_wins'], first['capped']),
                             (3, 1, 1, 1))
            self.assertAlmostEqual(first['mean_p2_value'], .7)
            self.assertEqual(first['sources'], ['manual log'])


class ResumedReports(unittest.TestCase):
    """A checkpoint without a league entry still shows the games its report against the champion already holds,
    with a provisional Elo on the champion's rating, until the evaluator reaches it again."""

    def test_unrated_checkpoint_with_games_on_disk_gets_a_provisional_row(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            league = dict(champion='main/185000', checkpoints=[dict(id='main/185000', elo=1700.)])
            (run/'league.json').write_text(json.dumps(league), encoding='utf-8')
            def report(candidate, games, planned=64, **metrics):
                path = run/'evaluations'/f'{candidate.replace("/", "-")}-vs-main-185000'/'report.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(dict(candidate=candidate, opponent='main/185000', games=[{}]*games,
                                                metrics=dict(planned_games=planned, **metrics))), encoding='utf-8')
            report('new-schedule/205000', 40, wins=35, losses=5, incomplete=0, elo_delta=324., elo_delta_95pct=[50., 2400.])
            report('new-schedule/207500', 10, wins=6, losses=4, incomplete=0, elo_delta=60., elo_delta_95pct=[-100., 300.])
            report('new-schedule/192500', 0, wins=0, losses=0, incomplete=0, elo_delta=0.)
            playing = dict(stage='playing', comparison=dict(candidate='new-schedule/207500', opponent='main/185000'))
            rows = dashboard.resumed(run, league, playing)
            self.assertEqual([r['id'] for r in rows], ['new-schedule/205000'])
            row = rows[0]
            self.assertEqual((row['wins'], row['losses'], row['capped'], row['games'], row['games_planned']), (35, 5, 0, 40, 64))
            self.assertAlmostEqual(row['elo'], 2024.)
            self.assertEqual(row['elo_interval'], [1750., 4100.])
            idle = dict(stage='idle')
            self.assertEqual([r['id'] for r in dashboard.resumed(run, league, idle)], ['new-schedule/205000', 'new-schedule/207500'])
            league['checkpoints'].append(dict(id='new-schedule/205000', elo=2000.))
            self.assertEqual([r['id'] for r in dashboard.resumed(run, league, idle)], ['new-schedule/207500'])
            (run/'league.json').write_text(json.dumps(league), encoding='utf-8')
            (run/'evaluator-status.json').write_text(json.dumps(idle), encoding='utf-8')
            self.assertEqual([r['id'] for r in dashboard.dense_run(run, {})['evaluator']['resumed']], ['new-schedule/207500'])


class ExternalRatings(unittest.TestCase):
    def test_legacy_match_tracks_current_reference_rating(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            folder = run/'matches/opponent'
            folder.mkdir(parents=True)
            report = folder/'report.json'
            report.write_text('{"games": []}', encoding='utf-8')
            saved = dict(schema='hexo-external-elo-estimate-v1', calculated_at=1,
                scale=dict(zero_checkpoint='main/000500'),
                match=dict(local_checkpoint='main/132500', checkpoint_sha256='a'*64,
                    report='report.json', report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(), games=64),
                opponent=dict(model_id='pulsatrix', difficulty='standard'),
                estimate={'85k_minus_pulsatrix_elo': -200, 'pair_adjusted_delta_sd': 100, 'assumption': 'paired games'})
            (folder/'opponent-elo-estimate.json').write_text(json.dumps(saved), encoding='utf-8')
            reference = dict(id='main/132500', elo=1600, elo_interval=[1404, 1796], ema_sha256='a'*64)
            league = dict(checkpoints=[dict(id='main/000500', elo=0), reference])
            rating = dashboard.external_ratings(run, league)['pulsatrix:standard']
            self.assertEqual(rating['elo'], 1800)
            self.assertAlmostEqual(rating['elo_interval'][1]-1800, 1.96*math.sqrt(20000))
            reference['elo'] = 1700
            self.assertEqual(dashboard.external_ratings(run, league)['pulsatrix:standard']['elo'], 1900)
            saved['calculated_at'] = '2026-09-30'
            (folder/'opponent-elo-estimate.json').write_text(json.dumps(saved), encoding='utf-8')
            newer = dict(saved, schema='hexo-external-elo-estimate-v2', calculated_at=1790812800,
                sources=[dict(report='report.json', report_sha256=saved['match']['report_sha256'])],
                estimate=dict(reference_minus_opponent_elo=-300, pair_adjusted_delta_sd=100, assumption='joint fit'))
            # The numeric v2 file sorts before the ISO-string legacy file.
            (folder/'a-newer-elo-estimate.json').write_text(json.dumps(newer), encoding='utf-8')
            self.assertEqual(dashboard.external_ratings(run, league)['pulsatrix:standard']['elo'], 2000)
            newer['calculated_at'] = '2026-09-29'
            saved['calculated_at'] = 1790812800
            (folder/'a-newer-elo-estimate.json').write_text(json.dumps(newer), encoding='utf-8')
            (folder/'opponent-elo-estimate.json').write_text(json.dumps(saved), encoding='utf-8')
            self.assertEqual(dashboard.external_ratings(run, league)['pulsatrix:standard']['elo'], 1900)

    def test_calibrated_match_requires_both_unchanged_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            folder = run/'matches/calibration'
            folder.mkdir(parents=True)
            match = run/'matches/six/report.json'
            match.parent.mkdir()
            calibration = folder/'report.json'
            for report in (match, calibration): report.write_text('{"games": []}', encoding='utf-8')
            saved = dict(schema='hexo-external-elo-estimate-v2', calculated_at=1,
                scale=dict(zero_checkpoint='main/000500'),
                match=dict(local_checkpoint='main/132500', checkpoint_sha256='a'*64,
                    report='../six/report.json', report_sha256=hashlib.sha256(match.read_bytes()).hexdigest(), games=12),
                opponent=dict(model_id='six', difficulty='default', label='Six@default'),
                calibration=dict(games=32), sources=[dict(report='report.json',
                    report_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest())],
                estimate=dict(reference_minus_opponent_elo=-600, pair_adjusted_delta_sd=100, assumption='joint fit'))
            path = folder/'six-elo-estimate.json'
            path.write_text(json.dumps(saved), encoding='utf-8')
            league = dict(checkpoints=[dict(id='main/000500', elo=0),
                dict(id='main/132500', elo=1600, elo_interval=[1404, 1796], ema_sha256='a'*64)])
            rating = dashboard.external_ratings(run, league)['six:default']
            self.assertEqual((rating['label'], rating['elo'], rating['calibration_games']), ('Six@default', 2200, 32))
            self.assertAlmostEqual(rating['elo_interval'][0], 2200-1.96*math.sqrt(20000))
            sources = saved.pop('sources')
            for evidence in (None, []):
                if evidence is not None: saved['sources'] = evidence
                path.write_text(json.dumps(saved), encoding='utf-8')
                self.assertEqual(dashboard.external_ratings(run, league), {})
            saved['sources'] = sources
            path.write_text(json.dumps(saved), encoding='utf-8')
            for report in (match, calibration):
                original = report.read_bytes()
                report.write_text('{"games": [1]}', encoding='utf-8')
                self.assertEqual(dashboard.external_ratings(run, league), {})
                report.write_bytes(original)
            calibration.unlink()
            self.assertEqual(dashboard.external_ratings(run, league), {})


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

    def test_scripts_load_while_data_requests_wait_for_shared_cache(self):
        entered, release, second = threading.Event(), threading.Event(), threading.Event()
        results, errors = [], []
        def slow_project(_):
            if entered.is_set(): second.set()
            entered.set()
            if not release.wait(5): raise TimeoutError('test did not release data request')
            return dict(runs=[])
        def request():
            try: results.append(self.get('/api/project'))
            except Exception as error: errors.append(error)
        with patch('dashboard.project', side_effect=slow_project):
            first = threading.Thread(target=request)
            queued = threading.Thread(target=request)
            first.start()
            try:
                self.assertTrue(entered.wait(2))
                queued.start()
                for script in ('openings.js', 'book.js', 'game-lengths.js'):
                    with urlopen(f'http://127.0.0.1:{self.server.server_port}/{script}', timeout=2) as response:
                        self.assertEqual(response.status, 200)
                        self.assertEqual(response.headers.get_content_type(), 'text/javascript')
                        self.assertTrue(response.read())
                self.assertFalse(second.is_set())
            finally:
                release.set()
                first.join(5)
                if queued.ident is not None: queued.join(5)
        self.assertFalse(errors)
        self.assertEqual(results, [dict(runs=[]), dict(runs=[])])

    def test_game_length_api_and_query_validation(self):
        folder = self.run/'shards/001'
        self.write(folder/'manifest.json', dict(origin='actor', created_at=100))
        self.write(folder/'episodes.json', [dict(moves=[[0, 0]]*17, winner=0, reason='six-in-a-row')])
        result = self.get('/api/game-lengths', hours=0, start='selfplay')
        self.assertEqual((result['games'], result['median']), (1, 17))
        for query in (dict(hours=-1), dict(start='converted')):
            with self.assertRaises(HTTPError) as error:
                self.get('/api/game-lengths', **query)
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
        self.server.RequestHandlerClass.runs = None
        self.server.RequestHandlerClass.run = self.run
        with urlopen(f'http://127.0.0.1:{self.server.server_port}/api/game-lengths?hours=0') as response:
            self.assertEqual(json.load(response)['games'], 1)

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
        self.assertEqual(self.get(min_games='', min_decisive='')['total'], 5)
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

    def test_book_selfplay_win_rates_share_counts_lengths_and_refresh(self):
        book_path = self.run/'openings.json'
        original = book_path.read_bytes()
        image = [[r, q] for q, r in self.a]
        image[1], image[2] = image[2], image[1]
        empty = self.nodes[3]
        def episode(moves, plies, winner, origin='book'):
            return dict(moves=moves+[[9, 9]]*(plies-len(moves)), winner=winner, origin=origin,
                        book=dict(suite='book', ply=len(moves), key=dense_openings.canonical(moves)[0]))
        episodes = [episode(image, plies, winner) for plies, winner in [(11, 0), (21, 1), (31, 1), (41, -1)]]
        episodes += [episode(empty['moves'], 9, 0), episode(image, 100, 0, 'selfplay'),
                     episode(image, 200, 0, 'restart'), dict(episode(image, 150, 0), opponent='older-model')]
        folder = self.run/'shards/001'
        self.write(folder/'manifest.json', dict(origin='actor', counts=dict(book_games=6)))
        self.write(folder/'episodes.json', episodes)
        converted = self.run/'shards/002'
        self.write(converted/'manifest.json', dict(identity=dict(source='converted corpus')))
        self.write(converted/'episodes.json', [episode(image, 300, 0)])
        rows = {row['key']: row for row in self.get()['rows']}
        a = rows[self.nodes[1]['key']]
        self.assertEqual((a['games'], a['p1_wins'], a['p2_wins'], a['capped']), (10, 4, 3, 3))
        self.assertAlmostEqual(a['p1_win_rate'], 4/7)
        self.assertEqual(a['decisive_share'], .7)
        self.assertEqual((a['report_games'], a['selfplay_games'], a['median_plies']), (6, 4, 20.5))
        self.assertEqual(a['mean_plies'], 23.4)
        self.assertEqual(rows[empty['key']]['p1_win_rate'], 1)
        self.assertEqual(rows[self.nodes[0]['key']]['games'], 17)
        self.assertEqual(self.get(key=empty['key'], min_games=1)['total'], 1)
        self.assertEqual(next(n['games'] for n in self.get('/api/book/dag')['nodes'] if n['key']==empty['key']), 1)
        # The histogram and book share the compact cache, even when the book revision changes.
        read = Path.read_text
        def cached_read(path, *args, **kwargs):
            self.assertNotEqual(path.name, 'episodes.json')
            return read(path, *args, **kwargs)
        with patch.object(Path, 'read_text', cached_read):
            self.assertEqual(dashboard.game_lengths(self.run, 0)['games'], 8)
            self.assertEqual(self.get(key=empty['key'])['rows'][0]['games'], 1)
        self.write(folder.with_name('003')/'manifest.json', dict(origin='actor', counts=dict(book_games=1)))
        self.write(folder.with_name('003')/'episodes.json', [episode(empty['moves'], 13, 1)])
        row = self.get(key=empty['key'])['rows'][0]
        self.assertEqual((row['games'], row['p1_win_rate'], row['mean_plies']), (2, .5, 11))
        self.assertEqual(book_path.read_bytes(), original)
        # Live training starts must not leak into a frozen evaluation suite's results.
        self.write(self.run/'evaluator-status.json', dict(settings=dict(opening_suite='custom')))
        self.write(self.run/'openings-custom.json', dict(nodes=[empty]))
        self.assertEqual(self.get()['rows'][0]['games'], 0)

    def test_repeated_book_results_keep_weighted_lengths(self):
        folder = self.run/'shards/001'
        self.write(folder/'manifest.json', dict(origin='actor', counts=dict(book_games=10**9+1)))
        self.write(folder/'episodes.json', [])
        moves = tuple(map(tuple, self.a))
        summary = {(10, 'book', 'win', 0, moves): 10**9, (30, 'book', 'win', 1, moves): 1}
        with patch.object(dashboard, 'episode_summary', return_value=summary):
            row = self.get(key=self.nodes[1]['key'])['rows'][0]
        self.assertEqual(row['games'], 10**9+7)
        self.assertEqual(row['median_plies'], 10)
        self.assertAlmostEqual(row['mean_plies'], (10**10+160)/(10**9+7))

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

    def test_unsaved_frozen_book_uses_repository_suite_without_writing(self):
        self.write(self.run/'config.json', dict(schema=dense_config.SCHEMA))
        frozen = Path(self.tmp.name)/'frozen'
        node = dict(self.nodes[3], status='opening')
        self.write(frozen/'standard-v1.json', dict(nodes=[node], counted={}))
        with patch.object(dense_openings, 'FROZEN', frozen):
            rows = self.get()['rows']
            self.assertEqual([row['key'] for row in rows], [node['key']])
            self.assertEqual(rows[0]['report_games'], 0)
            self.assertEqual(len(self.get('/api/book/dag')['nodes']), 1)
        self.assertFalse((self.run/'openings-standard-v1.json').exists())

    def test_lengths_use_recorded_report_pairs_including_imported_archives(self):
        for path, identity in [(self.report, 'current'), (self.report.with_name('report-old.json'), 'imported')]:
            report = json.loads(path.read_text())
            report['id'] = identity
            report['settings'] = dict(opening_suite='book' if identity=='current' else 'standard-v1')
            for game in report['games']:
                game['seed'] = game['pair']
            if identity == 'current':
                report['games'].append(dict(opening=self.a, winner=0, plies=100, pair=2, seed=2))
            self.write(path, report)
        self.write(self.run/'evaluations/later-foreign/report.json', dict(id='uncounted',
                   settings=dict(opening_suite='standard-v1'), games=[
                       dict(opening=self.a, winner=0, plies=200, pair=0, seed=0),
                       dict(opening=self.a, winner=0, plies=300, pair=0, seed=0)]))
        node = dict(self.nodes[1], games=4, p1_wins=3, p2_wins=0, capped=1)
        book = dict(nodes=[node], counted=dict(current=1, imported=1))
        self.write(self.run/'openings.json', book)
        row = self.get()['rows'][0]
        self.assertEqual(row['report_games'], row['games'])
        self.assertEqual(row['median_plies'], 15)
        self.assertEqual(row['mean_plies'], 15)
        book['counted']['current'] = 2
        node.update(games=6, p2_wins=1, capped=2)
        self.write(self.run/'openings.json', book)
        row = self.get()['rows'][0]
        self.assertEqual(row['report_games'], row['games'])
        self.assertEqual(row['median_plies'], 19)
        self.assertAlmostEqual(row['mean_plies'], 130/6)


class ProjectSmoothing(unittest.TestCase):
    """The comparison page's smoothing spans the same share of the x-axis for every series, however densely it was
    logged: a variant logged every 20 steps and one downsampled to every 220 steps smooth to the same curve."""

    SCRIPT = """
const signal = x => 1.5 + 0.3 * Math.sin(x / 7000) + 0.2 * Math.sin(x / 900);
const dense = Array.from({length: 2001}, (_, i) => [i * 20, signal(i * 20)]);
const sparse = dense.filter((q, i) => i % 11 === 0);
const unit = 40000 / 1000;
const d = ema(dense, 0.95, unit), s = ema(sparse, 0.95, unit);
const at = pts => x => pts.reduce((best, q) => Math.abs(q[0] - x) < Math.abs(best[0] - x) ? q : best)[1];
const xs = [12000, 20000, 30000, 39000];
const gap = Math.max(...xs.map(x => Math.abs(at(d)(x) - at(s)(x))));
const dPoint = ema(dense, 0.95, 0), sPoint = ema(sparse, 0.95, 0);
const gapPerPoint = Math.max(...xs.map(x => Math.abs(at(dPoint)(x) - at(sPoint)(x))));
console.log(JSON.stringify({gap, gapPerPoint, first: d[0][1], firstSignal: dense[0][1]}));
"""

    def test_series_of_different_density_smooth_alike(self):
        import re
        import shutil
        import subprocess
        node = shutil.which('node')
        if node is None:
            raise unittest.SkipTest('node is required to run the page script')
        page = (Path(__file__).resolve().parents[1] / 'web' / 'project.html').read_text(encoding='utf-8')
        ema = re.search(r'const ema=\(pts,w,unit\)=>\{.*?\};', page).group(0)
        out = json.loads(subprocess.run([node, '-e', ema + self.SCRIPT], capture_output=True, text=True, check=True).stdout)
        self.assertLess(out['gap'], 0.05)
        self.assertGreater(out['gapPerPoint'], out['gap'] * 2)
        self.assertAlmostEqual(out['first'], out['firstSignal'], places=9)


if __name__ == '__main__':
    unittest.main()
