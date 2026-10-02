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
        pv = [[-1, 1, 1], [-2, 2, 1], [9, 9, 0], [9, 10, 0], [5, 0, 1], [6, 0, 1]]
        top = [[-1, 1, 1, 1, 1], [-20, 5, 0, .99, 0]]
        live, old = self.overlay(dict(top=top, proof=dict(winner=1, turns=2, plies=8), pv=pv),
                                 dict(top=top, proof=dict(winner=1, turns=2)))
        self.assertEqual(live['candidates'], [])
        self.assertEqual([(p['q'], p['r'], p['n'], p['player']) for p in live['plies']],
                         [(*p[:2], i + 1, p[2]) for i, p in enumerate(pv)])
        self.assertEqual(len({(p['q'], p['r']) for p in live['plies']}), len(pv))
        self.assertEqual(live['plies'][0]['fade'], 1)
        self.assertAlmostEqual(live['plies'][-1]['fade'], .45)
        self.assertEqual(live['six'], [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0]])
        self.assertEqual(old, dict(candidates=[], plies=[], six=[]))


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
        requests = [['/import', dict(text=json.dumps(dict(history=history)))], ['/review', {}], ['/state', {}],
                    ['/storage/save', {}], ['/match', dict(players=[dict(engine='browser:test')]*2, games=2, clock=dict(mode='fixed'))],
                    ['/state', {}]]
        answers = node(dict(kind='play', history=history, requests=requests))
        self.assertTrue(all(a['status'] == 200 for a in answers[:-1]))
        self.assertEqual(answers[2]['data']['review'][-1]['label'], 'win')
        self.assertEqual(answers[5]['data']['match']['wins'], [1, 1])
        self.assertEqual(answers[5]['data']['match']['completed'], 2)
        games = answers[-1]['backup']['games']
        self.assertEqual(len(games), 4)
        self.assertEqual(sum(g['history'] == history for g in games), 3)
        self.assertTrue(any(g['history'] == [] for g in games))

    def test_browser_stop_import_and_delete_survive_pending_jobs(self):
        history = [[0, 0], [0, 2], [1, 2], [1, 0], [2, 0], [2, 3], [3, 3], [3, 0], [4, 0], [4, 4], [5, 4], [5, 0]]
        result = node(dict(kind='lifecycle', history=history))
        self.assertEqual(result['stopped_save'], dict(paused=True, active=False, completed=1))
        self.assertEqual(result['resumed'], dict(wins=[1, 1], completed=2))
        self.assertIsNone(result['deleted']['current'])
        self.assertTrue(all(m['single'] for m in result['deleted']['catalogue']))
        self.assertEqual(result['imported'], dict(status=200, history=[[0, 0]], saved=[[0, 0]]))
        self.assertEqual(result['stopped_timeout'], dict(paused=True, active=False, completed=0))
        self.assertEqual(result['forked_clock'], dict(match=None, clock=None))
        self.assertEqual(result['finished_opening_status'], 400)
        self.assertEqual((result['capped']['completed'], result['capped']['capped']), (2, 2))
        self.assertTrue(all(r['placements'] == 3 and r['reason'] == 'capped' for r in result['capped']['results']))
        self.assertEqual(result['uncapped'], dict(completed=2, capped=0, wins=[1, 1]))
        self.assertTrue(result['failure_clock_frozen'])

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
        self.assertEqual(result['changed_version'], {})
        self.assertTrue(result['restored_identity'])

    def test_browser_resumes_partial_match_after_switching_boards_and_clears_import_opening(self):
        result = node(dict(kind='resume'))
        self.assertEqual(result['before']['history'], [[0, 0], [0, 2], [1, 2]])
        self.assertEqual(result['after']['history'], result['before']['history'])
        self.assertEqual(result['after']['timings'], result['before']['timings'])
        self.assertEqual(result['after']['clock']['circle_ms'], result['before']['clock']['circle_ms'])
        self.assertGreater(result['after']['clock']['circle_ms'], 180000)
        self.assertIsNotNone(result['bookStart'])
        self.assertIsNone(result['imported'])
        self.assertFalse(result['auto'])

    def test_search_matches_native(self):
        """Same seed, position, budget and evaluations: the same actions, visits and policy as the native library."""
        model, games = random_model(1), export_web.histories(every=9)
        tactical = list(json.loads((ROOT/'tests'/'fixtures'/'tactical_positions.json').read_text())['positions'].values())[:4]
        cases = []
        positions = [(h, 64, None) for h in games[::3]]+[(h, 128, None) for h in tactical]+[(games[5], 512, None)]
        positions += [(games[5], 64, 'gumbel'), (tactical[0], 128, 'gumbel')]
        for history, simulations, choice in positions:
            recorder, steps, results = Recorder(model), [], []
            tree = NeuralSearch(recorder, 'test', history, seed=1740, cache=EvaluationCache(), tactics=True)
            try:
                for _ in range(2 if len(history) % 2 else 1):
                    option = dict(choice=choice) if choice else {}
                    result = tree.search(simulations, root_samples=16, batch_size=16, **option)
                    steps.append(dict(simulations=simulations, root_samples=16, batch_size=16, **option))
                    results.append(result)
                    if result['action'] is None:
                        break
                    tree.advance(tuple(result['action']))
            finally:
                tree.close()
            cases.append((dict(history=history, seed=1740, tactics=True, steps=steps, batches=recorder.batches), results))
        answers = node(dict(kind='search', cases=[case for case, _ in cases]))
        for (case, results), answer in zip(cases, answers):
            self.assertEqual(len(answer), len(results))
            for native, web in zip(results, answer):
                self.assertEqual(web['action'], native['action'])
                self.assertEqual(web['visits'], native['visits'].tolist())
                self.assertEqual(web['completed'], native['completed'])
                np.testing.assert_allclose(web['policy'], native['policy'], rtol=0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
