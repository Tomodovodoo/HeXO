"""Shrimp in the browser (web/engine/shrimp) plays as the server's Shrimp: Six's driver over hexo-bot's search."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT/'web'/'engine'
NODE = shutil.which('node')
INSTALLED = ROOT/'models'/'shrimp'
spec = importlib.util.spec_from_file_location('shrimp_export', ROOT/'tools'/'shrimp_web'/'export.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)
HAS_ORT = importlib.util.find_spec('onnxruntime') is not None
BUILT = NODE is not None and (ENGINE/'shrimp'/'shrimp.wasm').exists()
DRIVER = INSTALLED/'arena'/'drivers'/'shrimp_driver.py'
WEIGHTS = INSTALLED/'rivals'/'shrimp'/'models'/'shrimp_main7_infer.pt'


def node(job):
    done = subprocess.run([NODE, str(ROOT/'tests'/'web'/'shrimp.mjs')], input=json.dumps(job), capture_output=True,
                          text=True, encoding='utf-8')
    if done.returncode:
        raise RuntimeError(done.stderr)
    return json.loads(done.stdout)


def positions():
    """Turns to play, as [history, visits]: openings, the tactical fixtures and middle games of recorded games."""
    games = list(json.loads((ROOT/'tests'/'fixtures'/'tactical_positions.json').read_text(encoding='utf-8'))['positions'].values())
    found = [([], 16), ([[0, 0]], 32), ([[0, 0], [1, 2], [3, -1]], 32), ([[0, 0], [0, 8], [2, 8], [1, 0], [2, 0], [4, 8], [6, 8]], 64),
             ([[0, 0], [0, 3], [1, 3], [1, 0], [2, 0], [2, 3], [3, 3], [3, 0], [4, 0]], 32)]
    found += [(games[0][:13], 64), (games[1][:24], 32), (games[2][:41], 64), (games[3][:60], 16)]
    return found


def random_net():
    """A small ShrimpNet with every parameter random, including the bias tables and layer scales."""
    import torch
    s = export.shrimp()
    torch.manual_seed(0)
    model = s.model.ShrimpNet(channels=24, attention_heads=2, trunk_layout='CACA')
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith('bias_tables') or name.endswith('gamma'):
                parameter.normal_(0, .5)
    return model.eval()


@unittest.skipUnless(HAS_ORT, 'needs onnx and onnxruntime (requirements/web.txt)')
class Export(unittest.TestCase):
    def test_graph_matches_the_evaluator_forward(self):
        model = random_net()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'shrimp.onnx'
            export.export_onnx(model, path)
            worst = export.parity(model, path, export.synthetic_rows(3, 12), batch=5)
        self.assertLess(max(worst.values()), 1e-4, worst)

    def test_profile_is_the_drivers(self):
        profile = export.search_profile()
        self.assertEqual((profile['support_radius'], profile['cache_states'], profile['search_parity_mode']), (4, 65536, False))
        settings = profile['settings']
        self.assertEqual((settings['c_puct'], settings['virtual_batch_size'], settings['gumbel_m']), (1.5, 32, 32))
        self.assertEqual((settings['gumbel_root'], settings['tss_enabled'], settings['moves_left_utility']), (1, 1, 1))


@unittest.skipUnless(NODE, 'needs node')
class Presets(unittest.TestCase):
    def test_presets_are_the_server_entrys(self):
        script = "import {PRESETS} from './web/engine/shrimp.mjs'; console.log(JSON.stringify(PRESETS))"
        done = subprocess.run([NODE, '--input-type=module', '-e', script], cwd=ROOT, capture_output=True, text=True, check=True)
        browser = {name: preset['visits'] for name, preset in json.loads(done.stdout).items()}
        server = json.loads((ROOT/'tools'/'engines.json').read_text(encoding='utf-8'))['shrimp']['presets']
        self.assertEqual(browser, {name: int(preset['args'][preset['args'].index('--visits') + 1]) for name, preset in server.items()})


class Recorder:
    """The driver's evaluator, recording each row it answers by the sha1 of its int32 coordinates and float32
    features: {value, moves_left, logits of its legal cells} and the row itself."""

    def __init__(self, inner):
        self.inner, self.rows, self.inputs = inner, {}, {}

    def __call__(self, payload):
        reply = self.inner(payload)
        offsets, total = payload['node_row_offsets'], payload['shape'][1]
        legal = np.frombuffer(payload['legal_counts'], np.int32)
        feats = np.frombuffer(payload['node_feats'], np.float16).astype(np.float32).reshape(total, 15)
        coords = np.frombuffer(payload['node_qr'], np.int16).astype(np.int32).reshape(total, 2)
        nbr = np.frombuffer(payload['nbr'], np.uint16).astype(np.int32).reshape(total, 6)
        values = np.frombuffer(reply['values_bytes'], np.float32)
        left = np.frombuffer(reply['moves_left_bytes'], np.float32)
        logits = np.frombuffer(reply['priors_logits_bytes'], np.float32)
        at = 0
        for i in range(len(legal)):
            start, end = offsets[i], offsets[i + 1]
            key = hashlib.sha1(coords[start:end].tobytes() + feats[start:end].tobytes()).hexdigest()
            self.rows[key] = dict(value=float(values[i]), moves_left=float(left[i]),
                                  logits=logits[at:at + legal[i]].astype(float).tolist())
            self.inputs[key] = (feats[start:end], coords[start:end], np.where(nbr[start:end] == 0xFFFF, -1, nbr[start:end]))
            at += int(legal[i])
        return reply


def driver_turns(cases):
    """The server's Shrimp (Six's driver in this process, as models/shrimp/launch.py runs it) on each case in order,
    a new game per turn as the play server sends: ([{moves, stones: [{action, value, visits}]}], recorder)."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for folder in (INSTALLED/'site', DRIVER.parent):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    driver = importlib.import_module('shrimp_driver')
    from six_rules import Game
    bot = driver.Shrimp(16, 2, 1, 0, 1.0)
    recorder = bot.evaluator = Recorder(bot.evaluator)
    stones, search = [], bot.search

    def recorded(state, ply):
        result = search(state, ply)
        stones.append(dict(action=driver.unpack(int(result['action_id'])), value=float(result['root_value']),
                           visits=int(result['visits'])))
        return result

    bot.search = recorded
    turns = []
    for history, visits in cases:
        bot.visits = visits
        bot.new_game()
        game = Game(8)
        for q, r in history:
            if game.place((q + r, -r)):
                raise ValueError(f'illegal {q, r}')
        stones.clear()
        cells = bot.turn(game)
        origin = game.moves[0] if game.moves else (0, 0)
        turns.append(dict(moves=[[q + r, -r] for q, r in cells],
                          stones=[dict(s, action=[s['action'][0] + origin[0] + s['action'][1] + origin[1], -(s['action'][1] + origin[1])])
                                  for s in stones]))
    return turns, recorder


@unittest.skipUnless(BUILT and DRIVER.exists() and WEIGHTS.exists(),
                     "needs node, web/engine/shrimp/shrimp.wasm and the server's Shrimp (engine setup into models/)")
class DriverParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = positions()
        cls.turns, cls.recorder = driver_turns(cls.cases)
        cls.profile = export.search_profile()

    def test_search_with_the_drivers_evaluations_plays_its_turns(self):
        """Same positions, budgets, seeds, game keys and network answers: the same stones, root values and visits."""
        answer = node(dict(kind='turns', profile=self.profile, cases=[dict(history=h, visits=v) for h, v in self.cases],
                           rows=self.recorder.rows))
        self.assertEqual(answer, self.turns)

    @unittest.skipUnless((ENGINE/'ort'/'ort.wasm.min.mjs').exists(), 'needs ONNX Runtime Web (python tools/build_web.py ort)')
    def test_turns_with_the_exported_graph_match_the_driver(self):
        """The whole browser engine (packing, ONNX Runtime Web on WebAssembly, search) against the driver, on the first
        cases (the same game keys)."""
        count = 6
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'shrimp.onnx'
            export.export_onnx(export.load(WEIGHTS), path)
            answer = node(dict(kind='network', profile=self.profile, bias=export.bias_layout(), model=str(path),
                               cases=[dict(history=h, visits=v) for h, v in self.cases[:count]]))
        self.assertEqual([turn['moves'] for turn in answer], [turn['moves'] for turn in self.turns[:count]])
        for turn, expected in zip(answer, self.turns):
            for stone, want in zip(turn['stones'], expected['stones']):
                self.assertEqual(stone['visits'], want['visits'])
                self.assertAlmostEqual(stone['value'], want['value'], delta=1e-4)

    @unittest.skipUnless(HAS_ORT, 'needs onnxruntime')
    def test_graph_matches_the_drivers_network_on_its_rows(self):
        model = export.load(WEIGHTS)
        keys = sorted(self.recorder.inputs)
        rows = [self.recorder.inputs[key] for key in keys]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'shrimp.onnx'
            export.export_onnx(model, path)
            import onnxruntime
            session = onnxruntime.InferenceSession(str(path), providers=['CPUExecutionProvider'])
            worst = dict(policy=0., value=0., moves_left=0.)
            for start in range(0, len(rows), 16):
                chunk = rows[start:start + 16]
                policy, value, left = session.run(None, export.pack(chunk))
                for i, key in enumerate(keys[start:start + 16]):
                    want = self.recorder.rows[key]
                    worst['policy'] = max(worst['policy'], float(np.abs(policy[i, :len(want['logits'])] - want['logits']).max()))
                    worst['value'] = max(worst['value'], abs(float(value[i]) - want['value']))
                    worst['moves_left'] = max(worst['moves_left'], abs(float(left[i]) - want['moves_left']))
        self.assertLess(worst['policy'], 1e-4, worst)
        self.assertLess(worst['value'], 1e-5, worst)
        self.assertLess(worst['moves_left'], 1e-3, worst)


if __name__ == '__main__':
    unittest.main()
