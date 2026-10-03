"""Parity of the browser engine bundle (web/engine) with the native engine and the PyTorch network."""
import base64
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np
import torch
import export_web
import hexcrop
import hexnet
import play
import tactical_proof
from neural_search import EvaluationCache, NeuralSearch

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT/'web'/'engine'
NODE = shutil.which('node')
BUILT = NODE is not None and (ENGINE/'gumbel.wasm').exists()
spec = importlib.util.spec_from_file_location('build_web', ROOT/'tools'/'build_web.py')
build_web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_web)


def node(job):
    done = subprocess.run([NODE, str(ROOT/'tests'/'web'/'engine.mjs')], input=json.dumps(job), capture_output=True,
                          text=True, encoding='utf-8')
    if done.returncode:
        raise RuntimeError(done.stderr)
    return json.loads(done.stdout)


def random_model(seed=0):
    """A HexNet with random weights, line taps and batch-norm statistics."""
    torch.manual_seed(seed)
    model = hexnet.HexNet(hexnet.HexNetConfig(aux_heads=False))
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, hexnet.MaskedNorm):
                module.running_mean.uniform_(-.5, .5)
                module.running_var.uniform_(.5, 2)
                module.weight.uniform_(.5, 1.5)
                module.bias.uniform_(-.2, .2)
            if isinstance(module, hexnet.LineConv):
                module.weight.normal_(0, .1)
    return model.eval()


class Recorder:
    """A DenseEvaluator that records every batch it answers as [{history, logits, q}]."""
    def __init__(self, model):
        self.inner, self.batches = hexnet.DenseEvaluator(model, 'cpu', 'test'), []

    def evaluate(self, histories):
        results = self.inner.evaluate(histories)
        self.batches.append([dict(history=[list(map(int, p)) for p in h], logits=r['logits'].astype(float).tolist(),
                                  q=r['q'].astype(float).tolist()) for h, r in zip(histories, results)])
        return results


@unittest.skipUnless(importlib.util.find_spec('onnxruntime'), 'needs onnx and onnxruntime (requirements/web.txt)')
class Export(unittest.TestCase):
    def test_graphs_match_the_reference_path(self):
        model = random_model()
        with tempfile.TemporaryDirectory() as folder:
            for name, half, tolerance in (('fp32.onnx', False, 1e-4), ('fp16.onnx', True, .1)):
                export_web.export_onnx(model, Path(folder)/name, half)
                worst = export_web.parity(model, Path(folder)/name, export_web.histories(every=13), half)
                self.assertLess(max(worst.values()), tolerance, (name, worst))


def page(job):
    done = subprocess.run([NODE, str(ROOT/'tests'/'web'/'book.mjs')], input=json.dumps(job), capture_output=True,
                          text=True, encoding='utf-8')
    if done.returncode:
        raise RuntimeError(done.stderr)
    return json.loads(done.stdout)


@unittest.skipUnless(NODE, 'needs node')
class PlayPage(unittest.TestCase):
    """The opening-book default follows the seats as the page shows them until the person touches the switch."""
    def follow(self, steps, enabled=True, available=True, stored=None, together=False):
        return page(dict(kind='follow', steps=steps, enabled=enabled, available=available, stored=stored, together=together))

    def books(self, answer):
        return [body['enabled'] for path, body in answer['requests'] if path == '/book']

    def test_engines_turn_the_book_on(self):
        answer = self.follow([['/seat', dict(side=0, engine='native')]], enabled=False)
        self.assertEqual(answer['requests'], [['/pause', dict(paused=True)], ['/seat', dict(side=0, engine='native')],
                                              ['/book', dict(enabled=True)]])
        self.assertTrue(answer['enabled'])
        self.assertFalse(answer['paused'])

    def test_people_turn_the_book_off(self):
        answer = self.follow([['/seat', dict(side=1, engine='human')]])
        self.assertEqual(self.books(answer), [False])
        self.assertIsNone(answer['stored'])

    def test_mixed_seats_take_the_server_default(self):
        answer = self.follow([['/seat', dict(side=1, engine='human')], ['/seat', dict(side=0, engine='six')]])
        self.assertEqual(self.books(answer), [False, True])
        self.assertEqual(self.books(self.follow([['/seat', dict(side=1, engine='native')]])), [])

    def test_a_browser_seat_counts_as_an_engine(self):
        answer = self.follow([['/seat', dict(side=1, engine='human')], ['/seat', dict(side=1, engine='browser:bubble')],
                              ['/seat', dict(side=0, engine='browser:bubble')]])
        self.assertEqual(self.books(answer), [False, True])
        self.assertTrue(answer['enabled'])

    def test_quick_seat_changes_end_on_the_last_seats(self):
        answer = self.follow([['/seat', dict(side=1, engine='human')], ['/seat', dict(side=0, engine='native')]], together=True)
        self.assertEqual(self.books(answer), [False, True])
        self.assertTrue(answer['enabled'])

    def test_a_touched_switch_stays(self):
        answer = self.follow([['/book', dict(enabled=False)], ['/seat', dict(side=0, engine='native')],
                              ['/book', dict(enabled=True, mode='wide')], ['/seat', dict(side=0, engine='human')],
                              ['/seat', dict(side=1, engine='human')]])
        self.assertEqual(self.books(answer), [False, True])
        self.assertEqual(answer['stored'], '1')
        self.assertFalse(answer['paused'])
        answer = self.follow([['/seat', dict(side=1, engine='human')]], stored='1')
        self.assertEqual(self.books(answer), [])

    def test_without_a_book_nothing_is_sent(self):
        self.assertEqual(self.books(self.follow([['/seat', dict(side=1, engine='human')]], available=False)), [])

    def test_the_serverless_page_never_asks_for_the_book(self):
        requests = page(dict(kind='offline', steps=[['/seat', dict(side=1, engine='human')], ['/seat', dict(side=0, engine='human')],
                                                    ['/play', dict(q=0, r=0)]]))
        self.assertEqual(requests, ['/seat', '/seat', '/play'])

    def test_the_static_page_session_starts_the_opening_for_new_seats(self):
        answer = page(dict(kind='static', steps=[['/seat', dict(side=1, engine='human')], ['/pause', dict(paused=False)],
                                                 ['/seat', dict(side=0, engine='browser:test')]]))
        self.assertEqual(answer['requests'], ['/seat', '/book', '/pause', '/pause', '/seat', '/book'])
        self.assertTrue(answer['enabled'])
        self.assertFalse(answer['paused'])
        self.assertGreater(answer['stones'], 0)

    def test_the_human_seat_is_labelled_human(self):
        sources = {name: (ROOT/'web'/name).read_text(encoding='utf-8') for name in ('index.html', 'engine/seat.mjs')}
        self.assertIn(".kind[data-k=human]", sources['index.html'])
        self.assertIn("return['human',null]", sources['index.html'])
        self.assertIn("kind:'human',label:null", sources['index.html'])
        self.assertIn("kind: 'human', label: null", sources['engine/seat.mjs'])
        self.assertIn("followSeats(globalThis,()=>S,seat=>isHuman(seat),storage)", sources['index.html'])
        self.assertIn("page.isHuman = seat =>", sources['engine/seat.mjs'])
        for text in sources.values():
            self.assertNotIn("'you'", text)
            self.assertNotIn('data-k=you', text)


@unittest.skipUnless(BUILT, 'needs node and a built web/engine (python tools/build_web.py wasm)')
@unittest.skipUnless(NODE, 'needs node')
class Overlay(unittest.TestCase):
    """web/engine/overlay.js: what the play page draws on the board for an evaluation."""
    stones = [[0, 0, 0], [1, 0, 1], [2, 0, 1], [0, 1, 0], [0, 2, 0], [3, 0, 1], [4, 0, 1]]

    def overlay(self, *evaluations):
        return node(dict(kind='overlay', cases=[dict(ev=ev, stones=self.stones) for ev in evaluations]))

    def test_unproven_positions_show_ranked_candidates_by_relative_share(self):
        top = [[5, 5, .5, .6, 0], [6, 6, .25, .55, 0], [7, 7, .05, .4, -1]]
        [found] = self.overlay(dict(top=top, proof=None, pv=[]))
        self.assertEqual(found['plies'], [])
        self.assertEqual(found['six'], [])
        self.assertEqual([(c['q'], c['r'], c['n']) for c in found['candidates']], [(5, 5, 1), (6, 6, 2), (7, 7, 3)])
        for got, want in zip([c['weight'] for c in found['candidates']], [1, .5, .1]):
            self.assertAlmostEqual(got, want)

    def test_proven_positions_show_only_the_numbered_line_and_its_six(self):
        pv = [[-1, 1, 1, 1], [-2, 2, 1, 2], [9, 9, 0, 3], [9, 10, 0, 4], [5, 0, 1, 7], [6, 0, 1, 8]]
        top = [[-1, 1, 1, 1, 1], [-20, 5, 0, .99, 0]]
        live, old, unnumbered = self.overlay(dict(top=top, proof=dict(winner=1, turns=2, plies=8), pv=pv),
                                             dict(top=top, proof=dict(winner=1, turns=2)),
                                             dict(top=top, proof=dict(winner=1, turns=2), pv=[p[:3] for p in pv]))
        self.assertEqual(live['candidates'], [])
        self.assertEqual([(p['q'], p['r'], p['n'], p['player']) for p in live['plies']],
                         [(q, r, n, player) for q, r, player, n in pv])
        self.assertEqual([p['n'] for p in unnumbered['plies']], [1, 2, 3, 4, 5, 6])
        self.assertEqual(len({(p['q'], p['r']) for p in live['plies']}), len(pv))
        self.assertEqual(live['plies'][0]['fade'], 1)
        self.assertAlmostEqual(live['plies'][-1]['fade'], .45)
        self.assertEqual(live['six'], [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0]])
        self.assertEqual(old, dict(candidates=[], plies=[], six=[]))


@unittest.skipUnless(BUILT, 'needs node and a built web/engine (python tools/build_web.py wasm)')
class BrowserProofs(unittest.TestCase):
    """The game's proof table in the browser: proof.mjs Proofs against play.Proofs, and the static page's session."""

    def test_the_table_matches_play(self):
        records = [([[0, 0], [1, 0], [2, 0]], dict(proof=dict(winner=1, turns=1, plies=4), pv=[[3, 0, 0, 1], [-1, 0, 0, 2]])),
                   ([[0, 0], [4, 4]], dict(proof=dict(winner=0, turns=1, plies=3), pv=[])),
                   ([[0, 0], [4, 4]], dict(proof=dict(winner=0, turns=1, plies=3), pv=[[5, 4, 1, 1], [6, 4, 1, 2]])),
                   ([[0, 0], [5, 5], [6, 6]], dict(proof=dict(winner=1, turns=2, plies=7), pv=[[7, 7, 0, 1], [8, 8, 0, 2]])),
                   ([[0, 0], [2, 2]], dict(proof=dict(winner=0, turns=2), pv=[[3, 2, 1, 1], [3, 3, 1, 2]])),
                   ([[0, 0], [-2, 0], [-2, 1]], dict(proof=dict(winner=0, turns=2), pv=[[-3, 1, 0, 1]]))]
        queries = [[[0, 0]], [[0, 0], [1, 0]], [[0, 0], [5, 5]], [[0, 0], [1, 0], [2, 0], [3, 0]], [[0, 0], [9, 9]],
                   [[0, 0], [2, 2], [3, 2]], [[0, 0], [-2, 0], [-2, 1], [-3, 1]]]
        result = dict(actions=[[1, 0], [4, 4], [7, 7]], values=[.1, .2, .3], policy=[.2, .5, .3], action=[4, 4], proven=0)
        lost = dict(actions=[[4, 4]], values=[.4], policy=[1.], action=[4, 4], proven=0)
        exact = dict(result, action=[7, 7], proven=1, exact_winner=1, proof_plies=3)
        found = node(dict(kind='table', records=records, queries=queries, result=result, lost=lost, exact=exact, mover=1))
        table = play.Proofs()
        for history, record in records:
            table.add(history, record)
        for history, answer in zip(queries, found['queries']):
            edges = sorted([*action, winner, distance] for action, (winner, distance, _) in table.edges(history).items())
            self.assertEqual((answer['known'], sorted(answer['edges'])), (table.known(history), edges), history)
        self.assertEqual({k: found['settled'][k] for k in ('action', 'proven', 'proof_plies', 'values')},
                         dict(action=[1, 0], proven=1, proof_plies=6, values=[1, -1, .3]))
        self.assertEqual({k: found['exact'][k] for k in ('action', 'proven', 'proof_plies', 'values')},
                         dict(action=[7, 7], proven=1, proof_plies=3, values=[1, -1, .3]))
        self.assertEqual({k: found['lost'][k] for k in ('action', 'proven', 'exact_winner', 'proof_plies')},
                         dict(action=[4, 4], proven=-1, exact_winner=0, proof_plies=4))
        self.assertEqual(table.known([[0, 0], [4, 4]])['pv'], [[5, 4, 1, 1], [6, 4, 1, 2]])

    def test_a_proof_carries_back_and_stays_in_the_browser_session(self):
        from tests.test_tactical_proof import LATE_WIN
        if not tactical_proof.library().exists():
            self.skipTest('needs the native tactical library')
        history = [tuple(p) for p in LATE_WIN] + [(-1, -11)]
        prover = tactical_proof.NativeTactics()
        prover.abort = lambda: None
        solved = play.solve(prover, history, 32768)
        found = dict(moves=solved['moves'], value=1, top=[[*solved['moves'][0], 1, 1, 1]], proof=solved['proof'], pv=solved['pv'],
                     threat=[], solved=True)
        answer = node(dict(kind='proofs', history=[list(p) for p in history], ply=len(history), found=found))
        line = [[-1, -11, 0, 1]] + [[*p[:3], p[3] + 1] for p in solved['pv']]
        analysed = answer['analysed']
        self.assertEqual((analysed['proof'], analysed['value'], analysed['pv']), (dict(winner=0, turns=4, plies=14), 1, line))
        self.assertEqual((analysed['top'][0][:2], analysed['top'][0][3:]), ([-1, -11], [1, 1]))
        table = play.Proofs()
        table.add(history, found)
        server = json.loads(json.dumps(play.evaluate(None, None, history[:-1], 8, 0, known=table)))
        fields = ('moves', 'proof', 'value', 'pv')
        for shown in (analysed, answer['given']):
            self.assertEqual({k: shown[k] for k in fields}, {k: server[k] for k in fields})
        self.assertEqual(answer['given']['top'], server['top'])
        self.assertEqual((analysed['top'][0][:2], analysed['top'][0][3:]), (server['top'][0][:2], server['top'][0][3:]))
        self.assertEqual(answer['kept'], dict(winner=0, turns=4, plies=14))
        self.assertEqual(answer['sent'][0], 0)
        self.assertGreater(answer['sent'][1], 1)
        self.assertEqual(answer['undone']['length'], len(history) - 1)
        for shown in (answer['undone']['shown'], answer['quick']['shown'], answer['quick']['saved'], answer['reloaded']):
            self.assertEqual((shown['proof'], shown['pv']), (dict(winner=0, turns=4, plies=14), line))


class Bundle(unittest.TestCase):
    def test_artefacts_match_their_sources(self):
        record = json.loads((ENGINE/'build.json').read_text(encoding='utf-8'))
        self.assertEqual(record['sources'], build_web.sources())
        self.assertEqual(record['artefacts'], {name: build_web.digest(ENGINE/name) for name in record['artefacts']})

    def test_encoder_matches_hexcrop(self):
        model = random_model()
        positions = export_web.histories(every=5)
        samples = [hexcrop.encode(h) for h in positions]
        answers = node(dict(kind='encode', positions=[dict(history=h, actions=s.actions.tolist()) for h, s in zip(positions, samples)]))
        self.assertTrue(any(s.far for s in samples))
        for sample, answer in zip(samples, answers):
            self.assertEqual(answer['size'], sample.size)
            self.assertEqual(answer['cells'], sample.cells.tolist())
            self.assertEqual(answer['far'], sample.far)
            self.assertEqual(answer['ones'], np.flatnonzero(sample.planes).tolist())
            with torch.inference_mode():
                expected = export_web.inputs(model, torch.from_numpy(sample.planes[None]).float())[0].numpy()
            np.testing.assert_array_equal(np.frombuffer(base64.b64decode(answer['features']), np.float32), expected.reshape(-1))

    def test_principal_variation_matches_play(self):
        from tests.test_play import PrincipalVariation
        cases = [([[0, 0]], PrincipalVariation.CERTIFICATE)]
        if tactical_proof.library().exists():
            from tests.test_tactical_proof import IMMEDIATE, LATE_WIN, OPEN_THREE
            for history in (OPEN_THREE, IMMEDIATE, LATE_WIN):
                for shortest in (False, True):
                    result = tactical_proof.NativeTactics().history(history, nodes=100000, ms=20000, shortest=shortest)
                    self.assertEqual(result['status'], 'PROVEN_WIN')
                    cases.append((history, result['certificate']))
        for history, certificate in cases:
            pv, plies = play.principal_variation(history, certificate)
            self.assertEqual(node(dict(kind='pv', history=history, certificate=certificate)), dict(pv=pv, plies=plies))

    def test_rows_match_play(self):
        rng = np.random.default_rng(3)
        for case in range(20):
            n = int(rng.integers(1, 12))
            actions = rng.integers(-9, 9, (n, 2))
            actions[:, 0] += np.arange(n) * 20
            policy = rng.dirichlet(np.ones(n)) * (rng.random(n) > .4)
            values = rng.choice([-1., 1., .3, -.2, .9], n) if case % 3 else None
            lead = actions[int(rng.integers(n))].tolist()
            expected = play.top_rows(actions, policy, values, lead=lead)
            found = node(dict(kind='rows', actions=actions.tolist(), policy=policy.tolist(),
                              values=None if values is None else values.tolist(), lead=lead))
            self.assertEqual(json.loads(json.dumps(found)), json.loads(json.dumps(expected)), case)

    def test_thread_count_follows_isolation_and_cores(self):
        contexts = [dict(isolated=True, cores=24), dict(isolated=False, cores=24), dict(isolated=True, cores=2),
                    dict(isolated=True, cores=5)]
        self.assertEqual(node(dict(kind='threads', contexts=contexts)), [8, 1, 1, 4])

    def test_offline_session_plays_and_undoes_like_the_server(self):
        stones = [[0, 0], [1, 0], [2, 0], [3, 0]]
        answers = node(dict(kind='offline', requests=[*[['/play', dict(q=q, r=r)] for q, r in stones], ['/play', dict(q=0, r=0)],
                                                      ['/undo', dict(people=[0])], ['/undo', dict(people=[0])],
                                                      ['/review', {}], ['/state', {}], ['/pause', dict(paused=True)],
                                                      ['/new', {}]]))
        self.assertEqual([a[0] for a in answers], [200, 200, 200, 200, 400, 200, 200, 501, 200, 200, 200])
        self.assertEqual(answers[3][1], stones)
        self.assertEqual(answers[5][1], stones[:3])
        self.assertEqual(answers[6][1], [])
        self.assertEqual([a[2] for a in answers[-2:]], [True, False])

    def test_browser_game_links_read_the_sites_through_their_open_mirror(self):
        site = ROOT/'tests'/'fixtures'/'hexo-site'
        game, sandbox = (json.loads((site/name).read_text(encoding='utf-8')) for name in ('finished-game.json', 'sandbox-position.json'))
        links = [['https://hexo.did.science/games/8211f449-5020-4a5a-9a93-581c5f720aac/', game],
                 ['https://hexo.mineking.dev/sandbox/2MDYN02', sandbox], ['https://hexo.tyto.cc/#g=ebb77124', {}]]
        answers = node(dict(kind='links', links=links))
        for (url, data), answer in zip(links[:2], answers):
            self.assertEqual(answer['history'], play.linked_history(url, lambda api, body=None: data))
        self.assertEqual([a['asked'] for a in answers], [
            [['https://hexo.mineking.dev/proxy/api/finished-games/8211f449-5020-4a5a-9a93-581c5f720aac', 'GET']],
            [['https://hexo.mineking.dev/proxy/api/sandbox-positions/2mdyn02', 'GET']], []])
        self.assertIn('HTTTX', answers[2]['error'])

    def test_browser_freeplay_runs_on_a_clock_with_increments_and_a_loss_on_time(self):
        result = node(dict(kind='clock'))
        self.assertEqual(result['set'], 200)
        self.assertEqual((result['after_turn']['cross_ms'] > 1000, result['after_turn']['running']), (True, 'o'))
        self.assertEqual(result['fixed_seat'][0], 400)
        self.assertIn('cannot keep a clock', result['fixed_seat'][1]['error'])
        turn = result['engine_turn']
        self.assertEqual((turn['history'], turn['clock']['running']), (3, 'x'))
        self.assertTrue(0 < turn['asked'][0] < 300)
        self.assertGreater(turn['clock']['circle_ms'], 1000)
        self.assertEqual(result['timed_out']['state'], dict(winner=1, outcome=dict(winner=1, reason='time')))
        game = result['timed_out']['game']
        self.assertEqual((game['winner'], game['reason'], game['clock']), (1, 'time', dict(mode='game', base_ms=300, increment_ms=1000)))
        self.assertEqual([t['side'] for t in game['turns']], [0, 1, 0])
        self.assertEqual(game['history'], [[0, 0], [1, 3], [1, 4]])
        self.assertEqual((result['timed_out']['play'], result['late'], result['long_clock']), (400, 400, 200))
        self.assertEqual(result['study_winner'], 1)
        self.assertEqual((result['fresh']['outcome'], result['fresh']['clock']['running']), (None, 'x'))
        self.assertGreater(result['fresh']['clock']['cross_ms'], 250)
        self.assertGreaterEqual(result['paused_turn'], 190)
        self.assertIn('different engine version', result['rebuilt_model'])
        self.assertGreaterEqual(result['load_charged']['load'], 290)
        self.assertLess(result['load_charged']['spent'], 150)
        self.assertLess(result['reloaded']['balance'], 59700)
        self.assertGreaterEqual(result['reloaded']['partial'], 390)

    def test_browser_notations_preserve_a_single_stone_final_turn(self):
        history = [[0, 0], [0, 2], [1, 2], [1, 0], [2, 0], [2, 3], [3, 3], [3, 0], [4, 0], [4, 4], [5, 4], [5, 0]]
        histories = [history[:1], history[:6], history]
        for original, formats in zip(histories, node(dict(kind='notation', histories=histories))):
            for name, data in formats.items():
                self.assertEqual(data['history'], original)
                expected = play.export(original, name)
                self.assertEqual(data['text'], expected['text'])

    def test_browser_book_has_unique_legal_starts_and_separate_policy_ranges(self):
        book = json.loads((ENGINE/'openings.json').read_text())
        counts = {m: len(book['nodes']) if m == 'all' else 8 if m == 'narrow' else sum(not n['off_policy'] for n in book['nodes'])
                  for m in ('narrow', 'wide', 'all')}
        for row in node(dict(kind='book')):
            self.assertEqual(row['count'], counts[row['mode']])
            self.assertEqual(row['unique'], row['count'])
            if row['mode'] != 'all':
                self.assertEqual(row['off_policy'], 0)

    def test_browser_review_and_paired_tournament_are_saved(self):
        history = [[0, 0], [0, 2], [1, 2], [1, 0], [2, 0], [2, 3], [3, 3], [3, 0], [4, 0], [4, 4], [5, 4], [5, 0]]
        requests = [['/import', dict(text=json.dumps(dict(history=history)))], ['/review', {}], ['/review', {}], ['/state', {}],
                    ['/storage/save', {}], ['/match', dict(players=[dict(engine='browser:test')]*2, games=2, clock=dict(mode='fixed'))],
                    ['/state', {}]]
        answers = node(dict(kind='play', history=history, requests=requests))
        self.assertTrue(all(a['status'] == 200 for a in answers[:-1]))
        self.assertEqual(answers[3]['data']['review'][-1]['label'], 'win')
        self.assertEqual(answers[6]['data']['match']['wins'], [1, 1])
        self.assertEqual(answers[6]['data']['match']['completed'], 2)
        games = answers[-1]['backup']['games']
        self.assertEqual(len(games), 4)
        self.assertEqual(sum(g['history'] == history for g in games), 3)
        self.assertTrue(any(g['history'] == [] for g in games))
        for game in games:
            self.assertEqual(len(game['records']), len({r['id'] for r in game['records']}))

    def test_browser_stop_import_and_delete_survive_pending_jobs(self):
        history = [[0, 0], [0, 2], [1, 2], [1, 0], [2, 0], [2, 3], [3, 3], [3, 0], [4, 0], [4, 4], [5, 4], [5, 0]]
        result = node(dict(kind='lifecycle', history=history))
        self.assertEqual(result['stopped_save'], dict(paused=True, active=False, completed=1))
        self.assertEqual(result['resumed'], dict(wins=[1, 1], completed=2))
        self.assertEqual(result['paused_save'], [dict(paused=True, completed=1, pending=True, current=2, resumedSeats=['other', 'test'])]*2)
        self.assertIsNone(result['deleted']['current'])
        self.assertTrue(all(m['single'] for m in result['deleted']['catalogue']))
        self.assertEqual(result['imported'], dict(status=200, history=[[0, 0]], saved=[[0, 0]], renewed=True))
        self.assertEqual(result['stopped_timeout'], dict(paused=True, active=False, completed=0))
        self.assertEqual(result['forked_clock'], dict(match=None, clock=None))
        self.assertEqual(result['finished_opening_status'], 400)
        self.assertEqual((result['capped']['completed'], result['capped']['capped']), (2, 2))
        self.assertTrue(all(r['placements'] == 3 and r['reason'] == 'capped' for r in result['capped']['results']))
        self.assertEqual(result['uncapped'], dict(completed=2, capped=0, wins=[1, 1]))
        self.assertTrue(result['failure_clock_frozen'])
        spec = dict(mode='game', base_ms=60000, increment_ms=1000)
        self.assertEqual(result['timed_replay'], dict(clock=spec, turns=1, opened=dict(spec=spec, turns=1, running=None)))
        self.assertEqual(result['stale_tab'], dict(conflicted=True, history=[[0, 0], [1, 0]], archive=[[0, 0], [1, 0]],
                                                 games=1, identity=True, mutation_status=400))
        self.assertEqual(result['stale_match'], dict(conflicted=True, session_completed=0, archive_completed=0, archived_games=0))

    def test_freeplay_updates_until_new_game(self):
        win = [[0, 0], [0, 2], [1, 2], [1, 0], [2, 0], [2, 3], [3, 3], [3, 0], [4, 0], [4, 4], [5, 4], [5, 0]]
        requests = [['/play', dict(q=0, r=0)], ['/play', dict(q=1, r=0)]] + [['/storage/save', {}]]*20 + [['/new', {}]]
        requests += [['/play', dict(q=q, r=r)] for q, r in win] + [['/storage/save', {}]]*20 + [['/state', {}]]
        answers = node(dict(kind='play', history=[], requests=requests))
        self.assertTrue(all(a['status'] == 200 for a in answers[:-1]))
        saved = answers[-1]
        self.assertEqual(len(saved['catalogue']), 2)
        self.assertTrue(all(m['kind'] == 'freeplay' for m in saved['catalogue']))
        self.assertCountEqual([g['history'] for g in saved['backup']['games']], [[[0, 0], [1, 0]], win])
        finished = next(g for g in saved['backup']['games'] if g['history'] == win)
        self.assertEqual((finished['winner'], finished['reason']), (0, 'six'))

    def test_browser_freeplay_deepens_restores_and_keeps_original_study(self):
        result = node(dict(kind='freeplay'))
        self.assertEqual([n for ply, n in result['calls'] if ply == 1], [1, 2, 4])
        self.assertEqual(result['history'], [[0, 0]])
        self.assertEqual(result['simulations'], 4)
        self.assertEqual(len(result['catalogue']), 2)
        self.assertTrue(result['preserved'])
        self.assertEqual(result['variation'], [[0, 0], [1, 0]])
        self.assertEqual(result['imported_label'], 'best')
        self.assertFalse(result['move_reused'])
        self.assertEqual(result['changed_version'], {})
        self.assertTrue(result['restored_identity'])

    def test_browser_resumes_partial_match_after_switching_boards_and_clears_import_opening(self):
        result = node(dict(kind='resume'))
        self.assertEqual(result['before']['history'], [[0, 0], [0, 2], [1, 2]])
        self.assertEqual(result['after']['history'], result['before']['history'])
        self.assertEqual(result['after']['timings'], result['before']['timings'])
        self.assertEqual(result['after']['turns'], result['before']['turns'])
        self.assertGreater(result['before']['turns'], 0)
        self.assertEqual(result['after']['clock']['circle_ms'], result['before']['clock']['circle_ms'])
        self.assertGreater(result['after']['clock']['circle_ms'], 180000)
        self.assertIsNotNone(result['bookStart'])
        self.assertIsNone(result['imported'])
        self.assertFalse(result['auto'])
        self.assertEqual(result['sameBatchSeats'], ['test', 'test'])

    def test_a_seat_line_carries_its_tree_across_turns(self):
        answer = node(dict(kind='game', simulations=128))
        self.assertEqual(answer['first'][0], 0)
        self.assertGreater(answer['second'][0], 0)
        self.assertTrue(answer['same'])
        self.assertEqual((answer['undone'][0], answer['fresh'][0]), (0, 0))
        self.assertEqual(answer['lines'], ['b', 'c'])

    def test_search_matches_native(self):
        """Same seed, position, budget, Q range floor, root noise and evaluations: the same actions, visits and policy
        as the native library."""
        model, games = random_model(1), export_web.histories(every=9)
        tactical = list(json.loads((ROOT/'tests'/'fixtures'/'tactical_positions.json').read_text())['positions'].values())[:4]
        cases = []
        positions = [(h, 64, None, 0., 0.) for h in games[::3]]+[(h, 128, None, 0., 0.) for h in tactical]
        positions += [(games[5], 512, None, 0., 0.), (games[5], 64, 'gumbel', 0., 0.), (tactical[0], 128, 'gumbel', 0., 0.)]
        positions += [(games[5], 128, None, .5, 0.), (games[27], 128, 'gumbel', 0., .25)]
        positions += [(games[5], 128, None, 0., 0., 'lost'), (games[5], 64, None, 0., 0., 'wins')]
        for history, simulations, choice, floor, noise, *marked in positions:
            recorder, steps, results = Recorder(model), [], []
            tree = NeuralSearch(recorder, 'test', history, seed=1740, cache=EvaluationCache(), tactics=True,
                                q_range_floor=floor, root_noise=noise)
            try:
                for _ in range(2 if len(history) % 2 else 1):
                    option = dict(choice=choice) if choice else {}
                    if marked and not steps:
                        # 'lost': the first root's most likely stone is proven lost for the mover three placements on.
                        # 'wins': its two most likely stones win for the mover, the second sooner; the web side gets
                        # them longest first and must still settle the root on the shorter.
                        prior = recorder.inner.evaluate([history])[0]
                        first, second = (prior['actions'][i].tolist() for i in np.argsort(-prior['logits'])[:2])
                        mover = play.player_at(len(history))
                        option['marks'] = ([[*first, 1 - mover, 3]] if marked == ['lost'] else
                                           [[*first, mover, 7], [*second, mover, 5]])
                        tree.expand()
                        for q, r, winner, distance in sorted(option['marks'], key=lambda m: (m[2] == mover, m[3])):
                            tree.mark((q, r), winner, distance)
                    result = tree.search(simulations, root_samples=16, batch_size=16,
                                         **{k: v for k, v in option.items() if k != 'marks'})
                    steps.append(dict(simulations=simulations, root_samples=16, batch_size=16, **option))
                    results.append(result)
                    if result['action'] is None:
                        break
                    tree.advance(tuple(result['action']))
            finally:
                tree.close()
            cases.append((dict(history=history, seed=1740, tactics=True, q_range_floor=floor, root_noise=noise, steps=steps,
                               batches=recorder.batches), results))
        answers = node(dict(kind='search', cases=[case for case, _ in cases]))
        for (case, results), answer in zip(cases, answers):
            self.assertEqual(len(answer), len(results))
            for native, web in zip(results, answer):
                self.assertEqual(web['action'], native['action'])
                self.assertEqual(web['visits'], native['visits'].tolist())
                self.assertEqual(web['completed'], native['completed'])
                self.assertEqual((web['proven'], web['unmarked']), (native['proven'], 0))
                np.testing.assert_allclose(web['policy'], native['policy'], rtol=0, atol=1e-12)
        marked, web = cases[-2][1][0], answers[-2][0]
        lost = marked['actions'].tolist().index(cases[-2][0]['steps'][0]['marks'][0][:2])
        self.assertEqual((marked['values'][lost], web['policy'][lost]), (-1., 0.))
        won, web = cases[-1][1][0], answers[-1][0]
        self.assertEqual((web['proven'], web['action'], won['proof_plies']), (1, cases[-1][0]['steps'][0]['marks'][1][:2], 5))


if __name__ == '__main__':
    unittest.main()
