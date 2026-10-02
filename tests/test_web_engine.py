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

    def test_search_matches_native(self):
        """Same seed, position, budget, Q range floor and evaluations: the same actions, visits and policy as the native
        library."""
        model, games = random_model(1), export_web.histories(every=9)
        tactical = list(json.loads((ROOT/'tests'/'fixtures'/'tactical_positions.json').read_text())['positions'].values())[:4]
        cases = []
        positions = [(h, 64, None, 0.) for h in games[::3]]+[(h, 128, None, 0.) for h in tactical]+[(games[5], 512, None, 0.)]
        positions += [(games[5], 64, 'gumbel', 0.), (tactical[0], 128, 'gumbel', 0.), (games[5], 128, None, .5)]
        for history, simulations, choice, floor in positions:
            recorder, steps, results = Recorder(model), [], []
            tree = NeuralSearch(recorder, 'test', history, seed=1740, cache=EvaluationCache(), tactics=True,
                                q_range_floor=floor)
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
            cases.append((dict(history=history, seed=1740, tactics=True, q_range_floor=floor, steps=steps,
                               batches=recorder.batches), results))
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
