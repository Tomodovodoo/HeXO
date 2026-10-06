"""Parity of the browser engine bundle (web/engine) with the native engine and the PyTorch network."""
import base64
import hashlib
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
from neural_search import EvaluationCache, GameGraph, NeuralSearch

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


@unittest.skipUnless(NODE, 'needs node')
class BoardPerspective(unittest.TestCase):
    """web/engine/symmetry.mjs: the 12 views of the play page's board and the steps between them."""
    cells = [[0, 0], [1, 0], [0, 1], [2, -1], [-3, 5], [4, 2], [-1, -1]]

    def test_views_are_the_hex_symmetries_and_their_inverses_undo_them(self):
        done = subprocess.run([NODE, str(ROOT/'tests'/'web'/'symmetry.mjs')], input=json.dumps(dict(cells=self.cells)),
                              capture_output=True, text=True, encoding='utf-8', check=True)
        answer = json.loads(done.stdout)
        cells = np.array(self.cells)
        self.assertEqual({tuple(map(tuple, image)) for image in answer['images']},
                         {tuple(map(tuple, cells @ m)) for m in hexcrop.SYMMETRIES})
        for undone in answer['undone']:
            self.assertEqual(undone, self.cells)
        self.assertEqual(answer['reached'], list(range(12)))


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
    def test_padded_proof_bound_does_not_send_a_finished_board_to_the_solver(self):
        from tests.test_tactical_proof import IMMEDIATE
        source = """import {readFileSync} from 'node:fs';
import {Proofs} from './web/engine/proof.mjs';
import {loadTactical} from './web/engine/tactical.mjs';
const history=JSON.parse(readFileSync(0,'utf8')),table=new Proofs();
table.add(history,{proof:{winner:0,plies:6},pv:[[5,0,0,1]]});
const facts=new Proofs(table.list()).facts(history),solver=await loadTactical(readFileSync('web/engine/tactical.wasm'));
const found=solver.history(history,{known:facts.map(({history,winner,plies})=>({history,winner,plies})),nodes:1,ms:1000});
console.log(JSON.stringify({facts,found:{status:found.status,reason:found.reason},plies:table.known(history).plies}));"""
        done = subprocess.run([NODE, '--input-type=module', '-e', source], cwd=ROOT, input=json.dumps(IMMEDIATE),
                              capture_output=True, text=True, check=True)
        result = json.loads(done.stdout)
        self.assertEqual([f['history'] for f in result['facts']], [IMMEDIATE])
        self.assertEqual(result['plies'], 6)
        self.assertEqual(result['found']['status'], 'PROVEN_WIN', result['found']['reason'])


    def test_large_archive_storage_and_compressed_files_keep_all_evidence(self):
        source = """import {PlayStorage} from './web/engine/storage.mjs';
import {compressFile,readGameFile,readGame} from './web/engine/notation.mjs';
const record={id:'analysis',document:Array.from({length:40000},(_,i)=>({i,description:'saved evidence '.repeat(4)}))};
const storage=new PlayStorage(null);
await storage.put('evaluations',record);
await storage.saveSession({id:'live',_write_token:'one',records:[record]},null,null);
const cold=new PlayStorage(null);cold.memory=storage.memory;
const saved=await cold.get('sessions','live'),game={history:[[0,0]],records:saved.records};
const json=JSON.stringify(game),file=await compressFile(new Blob([json],{type:'application/json'}));
const text=await readGameFile(file),history=await readGame(text,{game(){}});
await cold.put('evaluations',{...record,document:[{replacement:true}]});
const original=await cold.get('sessions','live'),backup=await cold.backup(),restored=new PlayStorage(null);
await restored.restore(backup,{game(){}});
const largeBackup=JSON.stringify({...backup,note:'x'.repeat(33*1024*1024)});
const backupText=await readGameFile(await compressFile(new Blob([largeBackup])));
console.log(JSON.stringify({bytes:json.length,compressed:file.size,history,equal:JSON.stringify(saved.records[0])===JSON.stringify(record),
 fileEqual:text===json,backupEqual:backupText===largeBackup,oldPreserved:original.records[0].document.length,restored:(await restored.get('sessions','live')).records[0].document.length}));"""
        done = subprocess.run([NODE, '--input-type=module', '-e', source], cwd=ROOT,
                              capture_output=True, text=True, check=True)
        result = json.loads(done.stdout)
        self.assertGreater(result['bytes'], 1048576)
        self.assertLess(result['compressed'], result['bytes'] // 4)
        self.assertEqual(result['history'], [[0,0]])
        self.assertTrue(result['equal'] and result['fileEqual'] and result['backupEqual'])
        self.assertEqual((result['oldPreserved'], result['restored']), (40000, 40000))

    def test_matching_replay_precedes_newer_unrelated_records(self):
        source = """import {Proofs} from './web/engine/proof.mjs';
const proofs=new Proofs(),history=[[0,0],[1,0],[0,1]],record={proof:{winner:0,plies:20},pv:[]};
proofs.add(history,record);
for(let i=0;i<128;i++) proofs.add(Array.from({length:401},(_,j)=>[j,i+1]),record);
const replay=proofs.replay(history);
console.log(JSON.stringify({first:replay[0].history,cells:replay.reduce((n,r)=>n+r.history.length+r.pv.length,0)}));"""
        done = subprocess.run([NODE, '--input-type=module', '-e', source], cwd=ROOT,
                              capture_output=True, text=True, check=True)
        result = json.loads(done.stdout)
        self.assertEqual(result['first'], [[0,0],[1,0],[0,1]])
        self.assertLessEqual(result['cells'], 50000)

    def test_failed_replays_leave_the_full_ordinary_solve_available(self):
        from tests.test_tactical_proof import FIXTURE
        history = FIXTURE['positions']['1790600287230040:30:248']
        replay = [dict(history=history, winner=winner, pv=[]) for winner in (0, 1)]
        found = node(dict(kind='worker-turn', history=history, simulations=0, nodes=1300,
                          replay=replay, replayMiss=True))
        self.assertEqual(found['proof']['winner'], 0)
        self.assertGreater(found['actual_solver_nodes'], 2600)
        self.assertEqual(tactical_proof.independent_verify(found['proof']['certificate'], history), 'PROVEN_WIN')
    """The game's proof table in the browser: proof.mjs Proofs against play.Proofs, and the static page's session."""

    def test_saved_strategy_survives_refresh_reload_and_replays_a_changed_board(self):
        from tests.test_tactical_proof import OPEN_THREE
        original = tactical_proof.NativeTactics().history(OPEN_THREE, nodes=20000, ms=5000)
        pv = node(dict(kind='pv', history=OPEN_THREE, certificate=original['certificate']))
        found = dict(proof=dict(winner=0, plies=pv['plies'], certificate=original['certificate']), pv=pv['pv'],
                     moves=original['moves'], value=1, top=[], solved=True)
        source = """import {readFileSync} from 'node:fs';
import {BrowserSession} from './web/engine/play-session.mjs';
import {Native} from './web/engine/search.mjs';
import createModule from './web/engine/gumbel.mjs';
const {history,found}=JSON.parse(readFileSync(0,'utf8')),native=new Native(await createModule());
const session=new BrowserSession(native),entry={id:'test',kind:'bubble',name:'Test',presets:{standard:{simulations:8,solver_nodes:5000}}};
session.registerEngine(entry,{});session.load(history,true);const spec=session.spec({engine:'test'});
await session.record(history,spec,found);
await session.record(history,spec,{...found,proof:{winner:0,plies:found.proof.plies}});
await session.persist();await session.saving;
const restored=new BrowserSession(native);restored.storage=session.storage;await restored.restore({paused:true});restored.extendProofs();
const changed=history.map(p=>[...p]);changed[changed.length-1]=[7,8];
console.log(JSON.stringify({records:restored.records,replay:restored.proofs.replay(changed),changed}));"""
        done = subprocess.run([NODE, '--input-type=module', '-e', source], cwd=ROOT,
                              input=json.dumps(dict(history=OPEN_THREE, found=found)), capture_output=True, text=True, check=True)
        saved = json.loads(done.stdout)
        self.assertEqual(saved['replay'][0]['certificate'], original['certificate'])
        result = node(dict(kind='worker-turn', adapter=True, history=saved['changed'], records=saved['records'], simulations=8, nodes=5000))
        self.assertEqual(result['proof']['winner'], 0)
        self.assertEqual(result['actual_completed'], 0)
        self.assertEqual(tactical_proof.independent_verify(result['proof']['certificate'], saved['changed']), 'PROVEN_WIN')

    def test_longer_defence_propagates_back_under_a_coarse_proof_bound(self):
        from types import SimpleNamespace
        root = [[0,0],[1,-2],[-1,-2],[0,-2],[2,-1],[-2,-3],[3,-3],[4,-2],[6,-3],
                [1,-4],[5,-4],[8,-4],[10,-5],[7,-5],[6,-6],[12,-6],[14,-7],[9,-6],[11,-7],
                [15,-10],[17,-11],[13,-8],[14,-11],[0,-1],[0,1],[0,-3],[0,2],[1,0],[-1,2],
                [3,-2],[-2,3],[2,0],[3,0],[4,0],[-2,0],[4,-1],[5,-2],[7,-4],[2,1],
                [3,-1],[6,-1],[1,-1],[7,-1],[6,-2],[6,0]]
        # The displayed line has 40 placements, but the certificate's turn bound is 52.
        # The child after (6,-4) later gains a 47-placement witness, within that bound.
        records = [(root, dict(proof=dict(winner=0,plies=52),
                              pv=[[6,1,1,1],[6,-4,1,2],[18,-12,0,39],[19,-12,0,40]])),
                   (root+[[6,-4]], dict(proof=dict(winner=0,plies=47),
                                      pv=[[6,2,1,1],[18,-12,0,46],[19,-12,0,47]])),
                   (root+[[6,-4],[6,2]], dict(proof=dict(winner=0,plies=20),
                                             pv=[[18,-12,0,19],[19,-12,0,20]]))]
        source = """import {readFileSync} from 'node:fs';
import {Proofs,proven} from './web/engine/proof.mjs';
const records=JSON.parse(readFileSync(0,'utf8')),root=records[0][0],table=new Proofs(),out=[];
for(const [history,record] of records){table.add(history,record);out.push(proven(table,root,null,2));}
console.log(JSON.stringify(out));"""
        done = subprocess.run([NODE, '--input-type=module', '-e', source], cwd=ROOT, input=json.dumps(records),
                              capture_output=True, text=True, check=True)
        browser = json.loads(done.stdout)
        table, expected = play.Proofs(), []
        for history, record in records:
            table.add(history, record)
            expected.append(play.Session.proven(SimpleNamespace(proofs=table), root, None))
        self.assertEqual(browser, expected)
        self.assertEqual([r['pv'][-1][3] for r in browser], [40,48,40])
        self.assertEqual(browser[1]['moves'], [[6,-4],[6,2]])
        self.assertEqual(browser[1]['top'][0][:2], [6,-4])
        self.assertEqual([r['proof']['plies'] for r in browser], [52]*3)

    def test_leaf_certificate_is_visible_before_the_first_stone_and_after_reload(self):
        from tests.test_tactical_proof import LATE_WIN
        history = [list(p) for p in LATE_WIN] + [[-1, -11]]
        found = node(dict(kind='worker-turn', adapter=True, history=history, simulations=8, nodes=0,
                          leafNodes=2048, leafQueryMs=1000))
        self.assertEqual(found['proof']['winner'], 0)
        self.assertGreater(len(found['pv']), 5)
        self.assertEqual(found['pv'], found['proofs'][0]['pv'])
        line = [[-1, -11, 0, 1]] + [[*p[:3], p[3] + 1] for p in found['pv']]
        for record in (found, dict(found, proof=None, pv=[])):
            answer = node(dict(kind='proofs', history=history, ply=len(history), found=record))
            for shown in (answer['analysed'], answer['undone']['shown'], answer['quick']['shown'], answer['reloaded']):
                self.assertEqual(shown['pv'], line)
                self.assertEqual(shown['proof']['plies'], found['proof']['plies'] + 1)

    def test_tighter_scalar_proof_does_not_suggest_half_a_turn(self):
        source = """import {Proofs,proven} from './web/engine/proof.mjs';
const table=new Proofs(), root=[[0,0]], old={moves:[[2,0],[3,0]],value:1,proof:{winner:1,plies:10,turns:3},pv:[]};
table.add(root,old); table.add([...root,[1,0]],{proof:{winner:1,plies:5},pv:[]});
const partial=proven(table,root,old,2);
table.add(root,{proof:{winner:1,plies:1},pv:[[1,0,1,1]]});
console.log(JSON.stringify([partial,proven(table,root,old,2)]));"""
        done = subprocess.run([NODE,'--input-type=module','-e',source],cwd=ROOT,capture_output=True,text=True,check=True)
        partial, terminal = json.loads(done.stdout)
        self.assertEqual((partial['proof']['plies'],partial['moves'],partial['pv']),(6,[],[[1,0,1,1]]))
        self.assertEqual(terminal['moves'],[[1,0]])

    def test_shorter_proof_replaces_saved_parent_after_undo_and_reload(self):
        history = [[0,0],[1,0],[2,0],[0,1]]
        old = dict(moves=[[3,1],[4,1]], value=1, top=[], proof=dict(winner=0,plies=42,turns=11),
                   pv=[[3,1,0,1],[4,1,0,2]])
        child = dict(moves=[[0,2]], value=1, top=[], proof=dict(winner=0,plies=29,turns=8),
                     pv=[[0,2,0,1]], threat=[], solved=True)
        answer = node(dict(kind='proofs',history=history,ply=4,found=child,records=[(history[:-1],old)]))
        for shown in (answer['analysed'],answer['undone']['shown'],answer['quick']['shown'],answer['reloaded']):
            self.assertEqual(shown['proof'], dict(winner=0,plies=30,turns=8))
            self.assertEqual(shown['moves'], [[0,1],[0,2]])
            self.assertEqual(shown['pv'], [[0,1,0,1],[0,2,0,2]])

    def test_shorter_complete_line_reaches_parent_even_with_the_same_padded_bound(self):
        root = [[0,0],[0,1]]
        old_moves = [[1,1],[8,0],[8,1],[0,2],[0,3],[8,2],[9,2],[0,4],[0,5],[9,0],[10,0],[0,6]]
        new_moves = [[8,0],[8,1],[0,3],[0,4],[8,2],[9,2],[0,5],[0,6]]
        old = dict(moves=[[1,1]], value=1, top=[], proof=dict(winner=1,plies=25,turns=7),
                   pv=[[*p,play.player_at(len(root)+i),i+1] for i,p in enumerate(old_moves)])
        child = dict(moves=new_moves[:2], value=0, top=[], proof=dict(winner=1,plies=24,turns=6),
                     pv=[[*p,play.player_at(len(root)+1+i),i+1] for i,p in enumerate(new_moves)])
        history = root+[[0,2]]
        expected = [[0,2,1,1]] + [[*p[:3],p[3]+1] for p in child['pv']]
        answer = node(dict(kind='proofs',history=history,ply=len(history),found=child,records=[(root,old)]))
        for shown in (answer['analysed'],answer['undone']['shown'],answer['quick']['shown'],answer['reloaded']):
            self.assertEqual(shown['proof']['plies'],25)
            self.assertEqual(shown['pv'],expected)
            self.assertEqual(shown['moves'],[[0,2]])
        for records in ([(root,old),(history,child)], [(history,child),(root,old)]):
            table = play.Proofs()
            for h,r in records:
                table.add(h,r)
            self.assertEqual(table.known(root),dict(winner=1,plies=25,pv=expected))
        # A just-checked line must not be overwritten by an older, longer
        # complete line while it is being shown, before it has been indexed.
        table = play.Proofs()
        table.add(root,old)
        fresh = dict(winner=1,plies=25,pv=expected)
        self.assertEqual(table.line(root,fresh),expected)
        source = """import {Proofs} from './web/engine/proof.mjs';
const {root,old,fresh}=JSON.parse(process.argv[1]), table=new Proofs();table.add(root,old);
console.log(JSON.stringify(table.line(root,fresh)));"""
        done = subprocess.run([NODE,'--input-type=module','-e',source,json.dumps(dict(root=root,old=old,fresh=fresh))],
                              cwd=ROOT,capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(done.stdout),expected)

    def test_reordered_histories_share_half_turn_deductions_regardless_of_query_order(self):
        loss = [[0,0],[1,0],[2,0],[0,1],[0,2],[4,0]]
        ordered = loss[:-1]+[[3,0]]
        reordered = [ordered[i] for i in (0,5,2,3,4,1)]
        queries = [reordered,ordered,reordered+[[4,0]],ordered+[[4,0]]]
        source = """import {Proofs} from './web/engine/proof.mjs';
const {loss,queries}=JSON.parse(process.argv[1]), results=[];
for(const order of [queries,queries.slice().reverse()]){
 const table=new Proofs();table.add(loss,{proof:{winner:0,plies:7},pv:[]});
 const read=h=>({known:table.known(h),edges:[...table.edges(h).values()].map(e=>[...e.action,e.winner,e.distance])});
 for(const h of order)read(h);results.push(queries.map(read));
}console.log(JSON.stringify(results));"""
        done = subprocess.run([NODE,'--input-type=module','-e',source,json.dumps(dict(loss=loss,queries=queries))],
                              cwd=ROOT,capture_output=True,text=True,check=True)
        first, reverse = json.loads(done.stdout)
        self.assertEqual(first,reverse)
        for result in first[:2]:
            self.assertIn([4,0,0,7],result['edges'])
        for result in first[2:]:
            self.assertEqual(result['known'],dict(winner=0,plies=6,pv=[]))
        table = play.Proofs()
        table.add(loss,dict(proof=dict(winner=0,plies=7),pv=[]))
        for history, expected in zip(queries,first):
            self.assertEqual(table.known(history),expected['known'])
            self.assertEqual(sorted([*a,w,d] for a,(w,d,_) in table.edges(history).items()),sorted(expected['edges']))

    def test_defender_reconsiders_its_longest_line_after_the_attack_improves(self):
        # One defensive choice initially lasts 15, the other 12. Improving the
        # attack under the first to 8 must make the defender choose the second.
        root = [[0,0],[1,0],[2,0]]
        a, b = [0,1], [1,1]
        records = [(root,dict(proof=dict(winner=1,plies=24),pv=[[*a,0,1],[0,2,0,2],[3,0,1,3]])),
                   (root+[a],dict(proof=dict(winner=1,plies=15),pv=[[0,2,0,1],[10,-2,1,2],[11,-2,1,3],
                        [9,-2,0,4],[9,-3,0,5],[3,0,1,6],[4,0,1,7],[8,0,0,8],[8,1,0,9],[5,0,1,10],
                        [7,0,1,11],[9,1,0,12],[10,1,0,13],[6,0,1,14]])),
                   (root+[b],dict(proof=dict(winner=1,plies=11),pv=[[1,2,0,1],[3,0,1,2],[7,0,1,3],
                        [8,0,0,4],[8,1,0,5],[4,0,1,6],[8,-1,1,7],[9,0,0,8],[9,1,0,9],[5,0,1,10],[6,0,1,11]])),
                   (root+[a,[0,2]],dict(proof=dict(winner=1,plies=6),pv=[[3,0,1,1],[4,0,1,2],
                        [8,0,0,3],[8,1,0,4],[5,0,1,5],[6,0,1,6]]))]
        source = """import {Proofs} from './web/engine/proof.mjs';
const {root,records}=JSON.parse(process.argv[1]), table=new Proofs(), result=[];
for(const [h,r] of records){table.add(h,r);result.push(table.known(root));}
result.push(new Proofs(table.list()).known(root));console.log(JSON.stringify(result));"""
        done = subprocess.run([NODE,'--input-type=module','-e',source,json.dumps(dict(root=root,records=records))],
                              cwd=ROOT,capture_output=True,text=True,check=True)
        results = json.loads(done.stdout)
        self.assertEqual(results[-3]['pv'][0][:2],a)
        self.assertEqual(results[-2]['pv'][0][:2],b,results[-2])
        self.assertEqual(results[-1],results[-2])
        self.assertEqual(results[-1]['plies'],24)
        table = play.Proofs()
        for (h,r), expected in zip(records,results):
            table.add(h,r)
            self.assertEqual(table.known(root),expected)


    def test_partial_proofs_gain_the_child_line_and_survive_reload(self):
        history = [[0, 0], [1, 0], [2, 0]]
        pv = [[1, 0, 1, 1], [2, 0, 1, 2], [2, 3, 0, 3], [3, 3, 0, 4], [3, 0, 1, 5], [4, 0, 1, 6]]
        root = dict(moves=history[1:], top=[], value=1, proof=dict(winner=1, plies=6, turns=2), pv=pv[:2])
        half = dict(moves=history[2:], top=[], value=1, proof=dict(winner=1, plies=5, turns=2), pv=[[2, 0, 1, 1]])
        found = dict(moves=[[2, 3], [3, 3]], top=[], value=0, proof=dict(winner=1, plies=4, turns=1),
                     pv=[[*p[:3], p[3] - 2] for p in pv[2:]])
        answer = node(dict(kind='proofs', history=history, ply=3, found=found, records=[(history[:1], root), (history[:2], half)]))
        self.assertTrue(answer['idleReuse'])
        line = [[*p[:3], p[3] - 1] for p in pv[1:]]
        for shown in (answer['analysed'], answer['undone']['shown'], answer['quick']['shown'], answer['reloaded']):
            self.assertEqual((shown['proof'], shown['value'], shown['pv']), (half['proof'], 1, line))
        for shown in (answer['parent'], answer['reloadedParent']):
            self.assertEqual((shown['proof'], shown['pv']), (root['proof'], pv))

    def test_the_table_matches_play(self):
        records = [([[0, 0], [1, 0], [2, 0]], dict(proof=dict(winner=1, turns=1, plies=4), pv=[[3, 0, 0, 1], [-1, 0, 0, 2]])),
                   ([[0, 0], [4, 4]], dict(proof=dict(winner=0, turns=1, plies=3), pv=[])),
                   ([[0, 0], [4, 4]], dict(proof=dict(winner=0, turns=1, plies=3), pv=[[5, 4, 1, 1], [6, 4, 1, 2]])),
                   ([[0, 0], [5, 5], [6, 6]], dict(proof=dict(winner=1, turns=2, plies=7), pv=[[7, 7, 0, 1], [8, 8, 0, 2]])),
                   ([[0, 0], [2, 2]], dict(proof=dict(winner=0, turns=2), pv=[[3, 2, 1, 1], [3, 3, 1, 2]])),
                   ([[0, 0], [-2, 0], [-2, 1]], dict(proof=dict(winner=0, turns=2), pv=[[-3, 1, 0, 1]]))]
        queries = [[[0, 0]], [[0, 0], [1, 0]], [[0, 0], [5, 5]], [[0, 0], [1, 0], [2, 0], [3, 0]], [[0, 0], [9, 9]],
                   [[0, 0], [2, 2], [3, 2]], [[0, 0], [-2, 0], [-2, 1], [-3, 1]]]
        # Lost A covers B,A, including after serializing and rebuilding the table.
        queries += [[[0,0],[5,4],[4,4]], [[0,0],[4,4],[5,4]], [[0,0],[5,4]]]
        defender = [[0,0],[2,-2],[3,-2]]
        records += [(defender,dict(proof=dict(winner=1,plies=12),pv=[[0,1,0,1],[0,2,0,2],[3,0,1,3]])),
                    (defender+[[0,1]],dict(proof=dict(winner=1,plies=3),pv=[[0,2,0,1],[3,0,1,2]])),
                    (defender+[[1,1]],dict(proof=dict(winner=1,plies=7),pv=[[1,2,0,1],[3,0,1,2]]))]
        queries += [defender]
        result = dict(actions=[[1, 0], [4, 4], [7, 7]], values=[.1, .2, .3], completed_q=[.1, .2, .3], policy=[.2, .5, .3],
                      action=[4, 4], proven=0)
        lost = dict(actions=[[4, 4]], values=[.4], completed_q=[.4], policy=[1.], action=[4, 4], proven=0)
        exact = dict(result, action=[7, 7], proven=1, exact_winner=1, proof_plies=3)
        found = node(dict(kind='table', records=records, queries=queries, result=result, lost=lost, exact=exact, mover=1))
        table = play.Proofs()
        for history, record in records:
            table.add(history, record)
        for history, answer in zip(queries, found['queries']):
            edges = sorted([*action, winner, distance] for action, (winner, distance, _) in table.edges(history).items())
            self.assertEqual((answer['known'], sorted(answer['edges'])), (table.known(history), edges), history)
            order = lambda f: json.dumps(f, sort_keys=True)
            self.assertEqual(sorted(answer['facts'], key=order), sorted(table.facts(history), key=order))
        self.assertIn([4,4,0,3], found['queries'][-2]['edges'])
        self.assertEqual({k:found['queries'][-4]['known'][k] for k in ('winner','plies')}, dict(winner=0,plies=2))
        self.assertEqual(found['queries'][-1]['known']['pv'][0],[1,1,0,1])
        self.assertEqual(found['queries'][-1]['known']['plies'],12)
        self.assertEqual(found['queries'][-1]['shown']['moves'], [[1,1],[1,2]])
        self.assertEqual(found['queries'][-1]['shown']['top'][0], [1,1,0,0,-1])
        self.assertEqual({k: found['settled'][k] for k in ('action', 'proven', 'proof_plies', 'values', 'completed_q')},
                         dict(action=[1, 0], proven=1, proof_plies=6, values=[1, -1, .3], completed_q=[1, -1, .3]))
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


@unittest.skipUnless(BUILT, 'needs node and a built web/engine (python tools/build_web.py wasm)')
class Loading(unittest.TestCase):
    """A browser engine's load reports its stages, and a stalled or failed stage moves it down the fallback chain."""
    LOAD = ['checking GPU', 'downloading 5 of 10 MB', 'downloading 10 of 10 MB', 'compiling']

    @classmethod
    def setUpClass(cls):
        done = subprocess.run([NODE, str(ROOT/'tests'/'web'/'loading.mjs')], capture_output=True, text=True, encoding='utf-8')
        if done.returncode:
            raise RuntimeError(done.stderr)
        cls.out = json.loads(done.stdout)

    def test_a_load_reports_each_stage_and_tags_the_device_after_the_probe(self):
        case = self.out['clean']
        self.assertEqual(case['stages'], [*self.LOAD, 'starting GPU', 'warming up'])
        self.assertEqual((case['tags'], case['notices'], len(case['starts'])), (['webgpu'], [], 1))
        self.assertNotIn('fallback', case['device'])

    def test_the_page_words_for_each_stage(self):
        self.assertEqual(self.out['words'], ['thinking', 'checking GPU', 'downloading 12 of 27 MB', 'downloading 2.5 of 4.6 MB',
                                             'downloading 17 MB', 'downloading', 'compiling', 'starting GPU', 'starting CPU',
                                             'warming up', 'warming up'])

    def test_each_gpu_stage_that_hangs_or_fails_falls_back_to_webassembly(self):
        for case, words in [('gpu_compile_hangs', 'compiling timed out'), ('gpu_session_hangs', 'starting GPU timed out'),
                            ('gpu_timing_hangs', 'warming up timed out'), ('gpu_warmup_hangs', 'warming up timed out'),
                            ('gpu_session_fails', 'starting GPU failed')]:
            with self.subTest(case=case):
                found = self.out[case]
                self.assertEqual(found['notices'], [dict(text=f'Bubble: {words}, running on CPU', cpu=True)])
                self.assertEqual(found['starts'], [dict(prefer=None, threads=None), dict(prefer='wasm', threads=None)])
                self.assertEqual(found['device'], dict(provider='wasm', threads=7, fallback=words))
                self.assertEqual(found['tags'], ['webgpu', 'wasm'])
                self.assertEqual(found['stages'][-2:], ['starting CPU', 'warming up'])

    def test_a_threaded_start_that_stalls_retries_on_one_thread(self):
        found = self.out['threads_stall']
        self.assertEqual(found['starts'], [dict(prefer='wasm', threads=None), dict(prefer='wasm', threads=1)])
        self.assertEqual(found['notices'], [dict(text='Bubble: compiling timed out, running on one thread', cpu=False)])
        self.assertEqual(found['device']['threads'], 1)

    def test_the_chain_ends_in_an_error_naming_the_stage(self):
        found = self.out['everything_hangs']
        self.assertEqual(found['starts'], [dict(prefer=None, threads=None), dict(prefer='wasm', threads=None), dict(prefer='wasm', threads=1)])
        self.assertEqual([n['cpu'] for n in found['notices']], [True, False])
        self.assertEqual(found['error'], 'Bubble: starting CPU timed out')
        self.assertEqual(self.out['silent']['error'], 'Bubble: checking GPU timed out')
        self.assertEqual(len(self.out['silent']['starts']), 3)

    def test_a_fixed_device_and_thread_count_do_not_fall_back(self):
        for case in ('fixed_device', 'fixed_gpu'):
            found = self.out[case]
            self.assertEqual((found['error'], found['notices'], len(found['starts'])), ('Bubble: starting GPU timed out', [], 1))

    def test_a_failed_download_is_not_retried_on_another_device(self):
        found = self.out['download_fails']
        self.assertEqual((found['error'], found['notices'], len(found['starts'])),
                         ('Bubble: downloading 5 of 10 MB failed: network error', [], 1))

    def test_a_probe_that_fell_back_is_noticed_once_and_keeps_later_loads_off_webgpu(self):
        found = self.out['probe_fell_back']
        self.assertEqual(found['notices'], [dict(text='Bubble: checking GPU timed out, running on CPU', cpu=True)])
        self.assertEqual((found['device']['fallback'], found['tags']), ('checking GPU timed out', ['wasm']))

    def test_a_call_that_stalls_loading_a_network_restarts_the_worker_and_is_sent_again(self):
        found = self.out['call_restarts']
        self.assertEqual(found['result'], 'wasm')
        self.assertEqual(found['starts'], [dict(prefer=None, threads=None), dict(prefer='wasm', threads=None)])
        self.assertEqual(found['call_stages'], ['starting GPU', *self.LOAD, 'starting CPU', 'warming up'])
        self.assertEqual(self.out['call_first'], 'done')
        overlap = self.out['calls_overlap']
        self.assertEqual((overlap['result'], len(overlap['starts'])), (['done', 'wasm'], 2))

    def test_engines_without_a_chain_still_end_a_silent_stage(self):
        self.assertEqual(self.out['single'], dict(native='Native (browser): compiling timed out', seal='Seal (browser): compiling timed out',
                                                  strix='Strix (browser): compiling timed out', reporting='loaded'))

    def test_threads_follow_device_memory(self):
        self.assertEqual(self.out['threads'], [7, 2, 4, 7, 1, 1, 2])

    def test_the_probe_times_out_and_prefers_fp16_on_a_limited_adapter(self):
        probe = self.out['probe']
        self.assertEqual(probe['hangs'], dict(provider='wasm', precisions=['fp32'], fallback='timed out', same=True))
        self.assertEqual(probe['rejects']['fallback'], 'failed (blocked)')
        self.assertIsNone(probe['none']['fallback'])
        self.assertEqual([probe[k]['precisions'] for k in ('desktop', 'phone', 'phone_without_f16', 'phone_asked_fp32')],
                         [['fp32', 'fp16'], ['fp16'], ['fp32'], ['fp32']])

    def test_network_stages_time_both_graphs_only_when_both_load(self):
        self.assertEqual(self.out['network']['both'], dict(stages=['download', 'session', 'timing'], precision='fp16'))
        self.assertEqual(self.out['network']['fp16_only'], dict(stages=['download', 'session'], precision='fp16'))
        self.assertEqual(self.out['network']['shrimp'], ['download', 'session'])
        self.assertEqual(self.out['network']['missing_manifest'], 'download')

    def test_a_job_stuck_loading_gives_way_and_a_cpu_fallback_moves_choices_to_lightning(self):
        session = self.out['session']
        self.assertEqual(session['stuck'], [dict(kind='move', status='running', stage='starting GPU')])
        self.assertEqual((session['loads'][:2], session['history']), (['stuck', 'quick'], [[0, 0]]))
        self.assertEqual(self.out['lighten'], dict(seats=['lightning', 'lightning'], preset='lightning'))


class Bundle(unittest.TestCase):
    def test_worker_cannot_return_more_endpoints_than_requested(self):
        request=dict(history=[[0,0]],attacker='mover',neural_frontier=1)
        endpoint=dict(path=[[1,0]],reason=0)
        answer=node(dict(kind='proof-answer',request=request,result=dict(status='UNKNOWN',neural_frontier=[endpoint,endpoint])))
        self.assertIn('frontier count',answer['error'])

    def test_rejected_browser_endpoint_limit_does_not_leak_owner(self):
        answer=node(dict(kind='native-proofs',history=[[0,0]],ms=80,delay=1,rejectEndpoints=True,endpoints=0))
        self.assertEqual(len(answer['rejectedEndpoints']),3)
        self.assertNotIn('error',answer)
        self.assertEqual((answer['stats']['pending'],answer['proof']['active'],answer['waits']),(0,0,0))

    def test_cpu_quiet_endpoint_gets_neural_search_and_retains_unknown_status(self):
        history=[[0,0],[4,0],[7,0],[-1,0],[-2,0]]
        endpoint=history+[[5,0],[6,0],[2,0],[8,0]]
        answer=node(dict(kind='native-proofs',history=history,ms=1000,slice=64,delay=1,views=8,depth=1))
        self.assertNotIn('error',answer)
        self.assertGreater(answer['proof']['neural_frontier']['candidates'],0)
        self.assertIn(endpoint,answer['viewHistories'])
        self.assertTrue(any(r['result']['status']=='UNKNOWN' for r in answer['neuralRecords']))
        self.assertEqual((answer['stats']['pending'],answer['proof']['active'],answer['waits']),(0,0,0))

    def test_native_owner_work_limit_survives_default_and_explicit_clocks(self):
        for clock in [dict(defaultClock=True),dict(ms=1000)]:
            answer = node(dict(kind='native-owner',history=[[0,0]],work=32,delay=2,**clock))
            self.assertLessEqual(answer['stats']['issued'],32)
            self.assertEqual((answer['stats']['pending'],answer['stats']['tasks'],answer['stats']['subscribers']),(0,0,0))

    def test_solver_cancellation_before_dispatch_does_not_start_a_slice(self):
        answer = node(dict(kind='native-proofs',history=[[0,0],[1,2],[3,-1]],cancelBeforeDispatch=True))
        self.assertEqual(answer['queries'],0)
        self.assertEqual((answer['proof']['active'],answer['proof']['queued'],answer['stats']['pending']),(0,0,0))

    def test_solver_cancellation_during_preparation_keeps_the_cancel_flag(self):
        answer = node(dict(kind='solver-preparation-cancel'))
        self.assertEqual((answer['messages'],answer['query_messages'],answer['flag']),(1,0,1))
        self.assertEqual((answer['info'][0],answer['info'][3],answer['info'][4]),(0,0,1))

    def test_unchanged_search_disagreement_does_not_bypass_solver_cooldown(self):
        answer = node(dict(kind='native-proofs',history=[[0,0]],cooldown=True))
        self.assertGreater(answer['discrepancy'],.1)
        self.assertLessEqual(answer['queries'],2)
        self.assertEqual((answer['proof']['active'],answer['proof']['queued'],answer['stats']['pending']),(0,0,0))

    def test_native_arbitrary_proof_reaches_reordered_context_and_both_ancestors(self):
        opening = [[0,0],[0,8],[2,8],[1,0],[2,0],[4,8],[6,8]]
        history = opening + [[-1,0],[2,1]]
        answer = node(dict(kind='native-proofs',history=opening,slice=64,offer=dict(history=history,peer=opening+[[2,1],[-1,0]])))
        self.assertEqual((answer['peerExact'],answer['middleExact'],answer['rootExact']),(0,0,0))
        proof = next(r for r in answer['records'] if r['request']['history'] == history)
        self.assertEqual(proof['result']['status'],'PROVEN_LOSS')
        self.assertEqual(tactical_proof.independent_verify(proof['result']['certificate'],history,attacker='defender',known=proof['request']['known']),'PROVEN_LOSS')
        self.assertEqual((answer['proof']['active'],answer['proof']['queued'],answer['stats']['pending'],answer['waits']),(0,0,0,0))

    def test_native_proof_shared_cancel_interrupts_a_long_slice(self):
        history = [[0,0],[0,8],[2,8],[1,0],[2,0],[4,8],[6,8]]
        answer = node(dict(kind='native-proofs',history=history,ms=2000,slice=1000,delay=30,cancelAfterMs=8))
        self.assertTrue(answer['cancelled'])
        self.assertGreater(answer['events'][0]['request']['ms'], 250)
        self.assertEqual(answer['events'][0]['info'][0], 0)
        self.assertLess(answer['events'][0]['ms'], 250)
        self.assertEqual((answer['proof']['installed'],answer['proof']['active'],answer['waits']),(0,0,0))

    def test_browser_worker_native_proofs_return_checked_evidence(self):
        history = [[0,0],[0,8],[2,8],[1,0],[2,0],[4,8],[6,8]]
        answer = node(dict(kind='worker-turn',history=history,simulations=4096,nodes=32768,ms=2000,nativeOwner=True,nativeProof=True,solverSlice=128,proofStamps=False))
        self.assertTrue(answer['proofs'])
        self.assertTrue(answer['proof'])
        self.assertTrue(answer['solved'])
        self.assertGreater(answer['actual_solver_nodes'], 0)
        self.assertGreater(answer['native_scheduler'][0]['proof']['installed'], 0)

    def test_native_proof_frontier_settles_during_inference_and_drains_both_producers(self):
        history = [[0,0],[0,8],[2,8],[1,0],[2,0],[4,8],[6,8]]
        answer = node(dict(kind='native-proofs',history=history,ms=2000,slice=16,delay=30))
        self.assertNotIn('error', answer)
        self.assertEqual(answer['result']['exact_winner'], 0)
        self.assertGreater(answer['proof']['installed'], 0)
        self.assertTrue(any(e['duringForward'] for e in answer['events']))
        self.assertEqual((answer['proof']['queued'],answer['proof']['active'],answer['proof']['ready'],answer['waits']),(0,0,0,0))
        self.assertEqual((answer['stats']['pending'],answer['stats']['tasks'],answer['stats']['subscribers']),(0,0,0))
        self.assertEqual(answer['graph']['views'], 1)
        for record in answer['records']:
            found, request = record['result'], record['request']
            self.assertEqual(tactical_proof.independent_verify(found['certificate'],request['history'],attacker=found['attacker'],known=request['known']),found['status'])

    def test_native_proof_cancellation_rejects_late_answers_and_releases_workers(self):
        answer = node(dict(kind='native-proofs',history=[[0,0],[0,8],[2,8],[1,0],[2,0],[4,8],[6,8]],ms=2000,slice=16,delay=30,cancelOnDispatch=True))
        self.assertTrue(answer['cancelled'])
        self.assertEqual(answer['proof']['installed'], 0)
        self.assertGreater(answer['proof']['cancelled'], 0)
        self.assertEqual((answer['proof']['queued'],answer['proof']['active'],answer['proof']['ready'],answer['waits']),(0,0,0,0))
        self.assertEqual((answer['stats']['pending'],answer['stats']['tasks'],answer['stats']['subscribers']),(0,0,0))

    def test_native_proof_unknown_bounds_do_not_create_game_losses(self):
        answer = node(dict(kind='native-proofs',history=[[0,0],[1,2],[3,-1]],ms=160,slice=8,delay=10,workers=2))
        self.assertNotIn('error', answer)
        self.assertEqual(answer['result']['exact_winner'], -1)
        self.assertGreater(answer['proof']['unknown'], 0)
        self.assertEqual(answer['proof']['installed'], 0)
        self.assertTrue(all(e['request']['ms'] <= 1000 for e in answer['events']))
        self.assertEqual((answer['proof']['queued'],answer['proof']['active'],answer['proof']['ready'],answer['waits']),(0,0,0,0))

    def test_native_owner_features_keep_the_browser_model_input_contract(self):
        histories = [[], [[0,0]], [[0,0],[4,0]], [[0,0],[4,0],[7,0]],
                     [[7*i,0] for i in range(36)]]
        answers = node(dict(kind='native-features', histories=histories))
        self.assertTrue(any(answer['far'] for answer in answers))
        for answer in answers:
            actual = np.frombuffer(base64.b64decode(answer['features']), np.float32)
            expected = np.frombuffer(base64.b64decode(answer['reference']), np.float32)
            np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)

    def test_native_browser_owner_preserves_legal_breadth_and_sampling_credits(self):
        history = [[0,0],[4,0],[7,0]]
        answer = node(dict(kind='native-owner', history=history, work=512, delay=2, batchSize=4))
        result, stats = answer['result'], answer['stats']
        game = play.Game(history)
        try: self.assertEqual(sorted(result['actions']), sorted([list(p) for p in game.legal_moves()]))
        finally: game.close()
        self.assertIn(result['action'], result['actions'])
        self.assertAlmostEqual(sum(result['policy']), 1.)
        self.assertGreater(stats['created'], 1)
        self.assertGreater(stats['depth'], 0)
        self.assertGreater(stats['root_completed'], 0)
        self.assertLessEqual(stats['root_completed'], stats['completed'])
        self.assertLessEqual(stats['last_credits'], stats['root_completed'])
        self.assertEqual(result['completed'], stats['root_completed'])
        self.assertEqual(result['all_view_completed'], stats['completed'])
        self.assertEqual(stats['issued'], stats['completed'] + stats['cancelled'])
        self.assertEqual((stats['pending'], stats['tasks'], stats['subscribers']), (0,0,0))
        self.assertEqual(answer['remainingViews'], 1)
        self.assertGreater(answer['readyBeforeResult'], 0)

    def test_native_browser_owner_drains_late_inference_and_keeps_exact_evidence(self):
        for mode in ('cancel', 'prove', 'nonfinite', 'throwWhilePending'):
            answer = node(dict(kind='native-owner', history=[[0,0]], work=512, delay=2, **{mode: True}))
            stats = answer['stats']
            self.assertEqual((stats['pending'], stats['tasks'], stats['subscribers']), (0,0,0))
            self.assertEqual(stats['issued'], stats['completed'] + stats['cancelled'])
            self.assertEqual(answer['remainingViews'], 1)
            if mode in ('nonfinite','throwWhilePending'):
                self.assertTrue(answer['rejected'])
                self.assertGreater(answer['forwardsFinished'], 0)
            elif mode == 'cancel': self.assertTrue(answer['blockedClose'])
            else:
                self.assertTrue(answer['lateProof'])
                self.assertEqual(answer['result']['proven'], -1)
                self.assertEqual(answer['result']['node_value'], -1.)

    def test_native_browser_owner_turn_uses_shared_graph_and_releases_views(self):
        result = node(dict(kind='worker-turn', history=[[0,0]], simulations=256, nodes=0, nativeOwner=True))
        self.assertEqual(len(result['moves']), 2)
        self.assertEqual(len(result['native_scheduler']), 2)
        game = play.Game([[0,0]])
        try:
            for move in result['moves']:
                self.assertTrue(game.legal(*move))
                game.play(*move)
        finally: game.close()
        for stats in result['native_scheduler']:
            self.assertEqual((stats['pending'], stats['tasks'], stats['subscribers']), (0,0,0))

    def test_native_browser_owner_stops_split_forwards_at_cancellation_or_proof(self):
        for control in (dict(cancel=True, cancelAfter=3), dict(stopAfter=3), dict(prove=True, proveAfter=3)):
            answer = node(dict(kind='native-owner', history=[[0,0]], work=512, maxBatch=1, delay=2, **control))
            self.assertEqual(answer['forwardCalls'], 3)
            self.assertEqual(answer['forwardsFinished'], 3)
            stats = answer['stats']
            self.assertEqual(stats['issued'], stats['completed'] + stats['cancelled'])
            self.assertEqual((stats['pending'], stats['tasks'], stats['subscribers']), (0,0,0))
            returned = answer['result']['scheduler']
            self.assertEqual((returned['pending'], returned['tasks'], returned['subscribers']), (0,0,0))
            self.assertEqual(answer['remainingViews'], 1)
            if control.get('prove'):
                self.assertEqual(answer['result']['proven'], -1)

    def test_native_browser_owner_checks_stop_after_packing_and_decoding(self):
        for control, forwards in (('stopOnFeatures',0), ('stopOnDecode',1)):
            answer = node(dict(kind='native-owner', history=[[0,0]], work=512, **{control: True}))
            self.assertEqual(answer['forwardCalls'], forwards)
            self.assertEqual(answer['result']['actions'], [])
            self.assertEqual(answer['result']['completed'], 0)
            self.assertEqual(answer['stats']['installed'], 0)
            self.assertEqual((answer['result']['scheduler']['pending'], answer['result']['scheduler']['tasks']), (0,0))

    def test_native_browser_owner_checks_its_clock_before_installation(self):
        answer = node(dict(kind='native-owner', history=[[0,0]], ms=200, expireBeforeInstall=True))
        self.assertEqual(answer['forwardCalls'], 1)
        self.assertEqual(answer['result']['actions'], [])
        self.assertEqual(answer['stats']['installed'], 0)
        self.assertEqual(answer['stats']['deadline'], 1)
        self.assertEqual((answer['result']['scheduler']['pending'], answer['result']['scheduler']['tasks']), (0,0))

    def test_native_browser_owner_reads_cancel_messages_between_microtask_forwards(self):
        for after in (1,3):
            answer = node(dict(kind='native-owner', history=[[0,0]], work=512, maxBatch=1,
                               cancelByMessage=True, messageCancelAfter=after))
            self.assertEqual(answer['forwardCalls'], after)
            self.assertEqual((answer['result']['scheduler']['pending'], answer['result']['scheduler']['tasks']), (0,0))
            if after == 1:
                self.assertEqual(answer['result']['actions'], [])
                self.assertEqual(answer['stats']['installed'], 0)

    def test_native_browser_owner_honors_policy_move_selection(self):
        answer = node(dict(kind='native-owner', history=[[0,0]], work=128, choice='policy'))
        result = answer['result']
        self.assertEqual(result['action'], result['actions'][int(np.argmax(result['policy']))])

    def test_browser_adapter_keeps_native_owner_opt_in(self):
        ordinary, compiled = node(dict(kind='owner-adapter'))
        self.assertFalse(ordinary['nativeOwner'])
        self.assertTrue(compiled['nativeOwner'])
        self.assertFalse(ordinary['nativeCapture'])
        self.assertTrue(compiled['nativeCapture'])

    def test_worker_does_not_admit_models_while_cached_session_cleanup_fails(self):
        answer = node(dict(kind='worker-model-cache'))
        self.assertEqual(answer['blocked']['created'], ['A','B'])
        self.assertEqual(answer['blocked']['live'], ['A','B'])
        self.assertEqual(answer['blocked']['errors'], ['Persistent session release failure']*4)
        self.assertEqual(answer['active_model'], 'B')
        self.assertEqual(answer['recovered']['created'], ['A','B','F','G','H'])
        self.assertEqual(answer['recovered']['live'], ['G','H'])
        self.assertEqual(answer['recovered']['largest'], 2)
        self.assertEqual(answer['remaining'], 0)

    def test_captured_inputs_change_and_buffers_drain_before_eviction_or_close(self):
        answer = node(dict(kind='capture-runtime'))
        self.assertEqual(answer['values'], [3,6,9,12,9])
        self.assertEqual(answer['shapes'], [3*24*24,3,3])
        self.assertEqual((answer['physical'], answer['reused']), (4,1))
        self.assertLessEqual(answer['bounds']['actual_cells'], 65536)
        self.assertGreater(answer['bounds']['stats']['evictions'], 0)
        self.assertTrue(answer['stopped'])
        self.assertEqual(answer['fence_error'], 'Device fence failed')
        self.assertEqual(answer['dispose_error'], 'Output release failed')
        self.assertEqual(answer['outputs_retained'], 1)
        self.assertTrue(answer['other_outputs_drained'])
        self.assertEqual((answer['recovered'], answer['last']), (5,7))
        self.assertEqual(answer['closed_error'], 'Native captures are closed')
        self.assertEqual(answer['release_error'], 'Session release failed')
        self.assertEqual(answer['release_failure'], dict(entries=1,sessions=1,base_releases=1,buffers=0))
        self.assertEqual(answer['base_releases'], 1)
        self.assertEqual(answer['eviction_error'], 'Session release failed')
        self.assertTrue(answer['poisoned_model'])
        self.assertEqual(answer['eviction_base_releases'], 1)
        self.assertEqual(answer['reloaded_value'], 15)
        self.assertEqual(answer['final']['stats']['unreleased_outputs'], 0)
        self.assertEqual([answer['final'][key] for key in ('active','freedBusy','mapped','buffers','sessions')], [0]*5)
        self.assertEqual(answer['final']['created'], answer['final']['released'])

    def test_stored_turn_reconnects_proofs_in_either_stone_order(self):
        from tests.test_neural_search import Uniform
        histories = [[[0,0],[1,0],[2,0]], [[0,0],[1,0]], [[0,0],[2,0]]]
        bad_indices = [Uniform().evaluate([history])[0]['actions'].tolist().index(forbidden)
                       for history, forbidden in zip(histories[1:], ([2,0], [1,0]))]
        witness = Uniform().evaluate([histories[0]])[0]['actions'][0].tolist()
        steps = [dict(at=histories[0], simulations=1, marks=[[*witness,0,5]]),
                 dict(at=histories[1], simulations=1), dict(at=histories[2], simulations=1)]
        result = node(dict(kind='search', cases=[dict(history=histories[0], seed=19, tactics=False,
                                                     limit=256, uniform=True, batches=[], steps=steps)]))[0]
        self.assertEqual(result[0]['unmarked'], 0)
        for turn, index in zip(result[1:], bad_indices):
            self.assertEqual(turn['policy'][index], 0.)
            self.assertAlmostEqual(sum(turn['policy']), 1.)

    def test_immediate_turn_keeps_a_complete_policy_without_neural_batches(self):
        history = [[0,0],[0,3],[1,3],[1,0],[2,0],[2,3],[3,3],[3,0],[7,4],[4,3],[5,4]]
        cases = [dict(history=history, seed=7, tactics=True, limit=16 if shared else None,
                      steps=[dict(simulations=32,root_samples=8,batch_size=16)]*(1 if shared else 2),batches=[])
                 for shared in (False,True)]
        results = node(dict(kind='search',cases=cases))
        for case,turns in zip(cases,results):
            self.assertEqual(len(turns), len(case['steps']))
            game = play.Game(history)
            try:
                for turn in turns:
                    self.assertEqual(turn['proven'], 1)
                    self.assertEqual(len(turn['policy']),len(game.legal_moves()))
                    self.assertTrue(np.isfinite(turn['policy']).all())
                    self.assertAlmostEqual(sum(turn['policy']), 1.)
                    game.play(*turn['action'])
                if len(turns)==2:
                    self.assertEqual(game.winner, 0)
            finally:
                game.close()

    def test_late_marks_tighten_proven_browser_roots_and_shared_parents(self):
        from tests.test_neural_search import Uniform
        network = Uniform()
        root = [[0,0],[0,3],[1,3],[1,0],[2,0],[2,3],[3,3],[3,0],[7,4],[4,3],[5,4]]
        child = root+[[-1,0]]
        batches = []
        for history in (root,child,[[0,0]]):
            prediction = network.evaluate([history])[0]
            batches.append([dict(history=history,logits=prediction['logits'].tolist(),q=prediction['q'].tolist())])
        steps = [dict(simulations=1,at=root,marks=[[-1,0,0,42]]),
                 dict(simulations=1,at=root,marks=[[4,0,0,34]]),
                 dict(simulations=1,at=child,marks=[[4,0,0,29]]),
                 dict(simulations=1,at=root)]
        lost = [[0,0]]
        actions = network.evaluate([lost])[0]['actions'].tolist()
        cases = [dict(history=root,seed=3,tactics=False,limit=4096,batches=batches[:2],steps=steps),
                 dict(history=lost,seed=3,tactics=False,limit=4096,batches=batches[2:],
                      steps=[dict(simulations=1,marks=[[*a,0,d] for a in actions]) for d in (33,17)])]
        won, losses = node(dict(kind='search',cases=cases))
        self.assertEqual([r['native_distance'] for r in won],[42,34,29,30])
        self.assertEqual([r['proof_plies'] for r in won],[42,34,29,30])
        self.assertEqual([r['native_distance'] for r in losses],[33,17])
        self.assertEqual([r['proven'] for r in losses],[-1,-1])
        self.assertTrue(all(r['unmarked']==0 for r in won+losses))

    def test_local_proofs_reach_browser_search_and_saved_lines(self):
        from tests.test_tactical_proof import OPEN_THREE
        history = OPEN_THREE + [[8,8],[10,8],[8,10],[10,10]]
        query = dict(history=history, options=dict(nodes=1, ms=10000, stamps=True))
        reference, reused = node(dict(kind='tactical', queries=[dict(history=history, options=dict(nodes=1, ms=10000)), query]))
        self.assertEqual(reference['status'], 'UNKNOWN')
        self.assertEqual(reused['status'], 'PROVEN_WIN', reused)
        self.assertEqual(tactical_proof.independent_verify(reused['certificate'], history), 'PROVEN_WIN')
        local = play.principal_variation(history, reused['certificate'])
        self.assertEqual(node(dict(kind='pv', history=history, certificate=reused['certificate'])),
                         dict(pv=local[0], plies=local[1]))
        self.assertTrue(local[0])
        turn = node(dict(kind='worker-turn', history=history, simulations=16, nodes=1))
        self.assertEqual((turn['proof']['winner'], turn['value']), (0, 1.))
        self.assertEqual(tactical_proof.independent_verify(turn['proof']['certificate'], history), 'PROVEN_WIN')

    def test_enabling_stamps_keeps_an_ordinary_one_stone_win_at_one_ply(self):
        from tests.test_tactical_proof import IMMEDIATE
        for enabled in (False, True):
            with self.subTest(stamps=enabled):
                turn = node(dict(kind='worker-turn', history=IMMEDIATE, simulations=16, nodes=1000,
                                 proofStamps=enabled))
                self.assertEqual((turn['proof']['winner'], turn['proof']['plies']), (0, 1))
                self.assertEqual(len(turn['pv']), 1)

    def test_changed_stamp_returns_the_shortened_strategy_it_checked(self):
        history = [[0,0],[1,-2],[-1,-2],[0,-2],[2,-1],[-2,-3],[3,-3],[4,-2],[6,-3],
                   [1,-4],[5,-4],[8,-4],[10,-5],[7,-5],[6,-6],[12,-6],[14,-7],[9,-6],
                   [11,-7],[15,-10],[17,-11],[13,-8],[14,-11],[15,-9],[16,-10],[13,-9],[14,-9]]
        original = history[:-2] + [[14,-8],[19,-10]]
        # The original five-turn strategy crosses both newly occupied cells.
        # The raw checker can shorten it, so returning that OLD strategy for
        # display used to throw "Illegal placement: 13,-9" after a checked win.
        nodes = [dict(kind='unstoppable', threats=[[[12,-8],[17,-13]],[[16,-14],[16,-13]],[[16,-8],[16,-7]]])]
        for action, responses in [([[16,-12],[16,-11]], [[[11,-9],[17,-9]],[[12,-9],[17,-9]],[[12,-9],[18,-9]]]),
                                  ([[13,-9],[14,-9]], [[[12,-10],[18,-10]]]),
                                  ([[13,-10],[14,-10]], [[[12,-5],[18,-11]],[[13,-6],[18,-11]],[[13,-6],[19,-12]]]),
                                  ([[16,-9],[17,-10]], [[[15,-13],[15,-7]],[[15,-12],[15,-7]],[[15,-12],[15,-6]]])]:
            nodes.append(dict(kind='attacker_move', action=action, child=len(nodes)-1))
            nodes.append(dict(kind='defender_replies', responses=[dict(action=r, child=len(nodes)-1) for r in responses]))
        nodes.append(dict(kind='attacker_move', action=[[15,-11],[15,-8]], child=len(nodes)-1))
        source = dict(stones=[[p,play.player_at(i)] for i,p in enumerate(original)], player=0, remaining=2, winner=0,
                      certificate=dict(version=1, width='wide', root=len(nodes)-1, nodes=nodes))
        certificate = dict(version=1, width='wide', root=0, nodes=[dict(kind='stamp', source=source)])
        cases = [history, history[:-2] + history[-2:][::-1]]
        found = node(dict(kind='tactical', queries=[dict(history=h, options=dict(certificate=certificate, stamps=True,
                                     nodes=20000, ms=10000)) for h in cases]))
        for h, result in zip(cases, found):
            self.assertEqual((result['status'], result['proof_turns']), ('PROVEN_WIN', 2))
            self.assertEqual(tactical_proof.independent_verify(result['certificate'], h), 'PROVEN_WIN')
            local = play.principal_variation(h, result['certificate'])
            self.assertEqual(local[1], 6)
            self.assertEqual(node(dict(kind='pv', history=h, certificate=result['certificate'])), dict(pv=local[0], plies=local[1]))

    def test_quiet_defender_certificate_has_the_same_browser_line(self):
        history = [[0,0],[0,8],[8,0],[1,0],[0,1],[-8,0],[0,-8],[1,1],[12,-8],[-8,8]]
        [result] = node(dict(kind='tactical', queries=[dict(history=history,
            options=dict(nodes=20000, ms=20000, stamps=True, attacker='defender'))]))
        self.assertEqual(result['status'], 'PROVEN_LOSS', result)
        self.assertEqual(tactical_proof.independent_verify(result['certificate'], history, attacker='defender'), 'PROVEN_LOSS')
        local = play.principal_variation(history, result['certificate'], attacker=0)
        self.assertEqual(node(dict(kind='pv', history=history, certificate=result['certificate'], options=dict(attacker=0))),
                         dict(pv=local[0], plies=local[1]))

    def test_worker_proves_a_defender_root_from_saved_exact_replies(self):
        from tests.test_tactical_proof import OPEN_THREE
        history = OPEN_THREE + [[-1, 0], [2, 1]]
        known = [dict(history=history+[list(p)], winner=0, plies=23, pv=[])
                 for p in ((-3, 0), (-2, 0), (3, 0), (4, 0))]
        found = node(dict(kind='worker-turn', history=history, simulations=16, nodes=1, known=known))
        self.assertEqual((found['proof']['winner'], found['value']), (0, 0.))
        self.assertTrue(found['proof']['dependencies'])
        self.assertTrue(all(row[4] < 0 for row in found['top']))
        local = play.principal_variation(history, found['proof']['certificate'], attacker=0, known=known)
        web = node(dict(kind='pv', history=history, certificate=found['proof']['certificate'],
                        options=dict(attacker=0, known=known)))
        self.assertEqual(web, dict(pv=local[0], plies=local[1]))
        known = [dict(history=OPEN_THREE, winner=0, plies=24, pv=[]),
                 dict(history=history, winner=0, plies=20, pv=[])]
        found = node(dict(kind='worker-turn', history=OPEN_THREE, simulations=16, nodes=1000, known=known))
        # The complete winning turn now comes directly from the tighter stored child,
        # even though the root already has a looser scalar proof. No new certificate is needed.
        self.assertEqual({tuple(p) for p in found['moves']}, {(-1,0),(2,1)})
        self.assertEqual(found['value'], 1.)
        self.assertEqual(found['proof']['plies'],22)
        self.assertEqual((found['actual_completed'],found['actual_solver_nodes']),(0,0))

    def test_live_values_stay_at_the_requested_root_during_reply_checks(self):
        history = [[0, 0], [1, 0], [1, 1], [-1, 0]]
        for length in (1, 2, 3, 4):
            root = history[:length]
            for repeat, turn in enumerate(node(dict(kind='glimpse', history=root, simulations=32, nodes=0))):
                with self.subTest(length=length, repeat=repeat):
                    self.assertTrue(turn['checked'], 'The principal-variation check must run')
                    self.assertTrue(turn['live'])
                    fractions = [p['fraction'] for p in turn['progress']]
                    self.assertEqual(fractions, sorted(fractions))
                    self.assertAlmostEqual(turn['result']['value'], .93, places=4)
                    for glimpse in turn['live']:
                        self.assertAlmostEqual(glimpse['value'], turn['result']['value'], places=4)
                        self.assertEqual(glimpse['root'], root)

    def test_root_candidates_are_published_before_the_solver_runs(self):
        root = [[0, 0], [1, 0]]
        for simulations in (0, 32):
            for turn in node(dict(kind='glimpse', history=root, simulations=simulations, nodes=16)):
                self.assertTrue(turn['queries'])
                self.assertTrue(all(q['preview'] for q in turn['queries']))
                self.assertEqual(turn['progress'][0]['stage']['name'], 'checking proof')
                fractions = [p['fraction'] for p in turn['progress']]
                self.assertEqual(fractions, sorted(fractions))
                self.assertEqual(turn['evaluations'].count(root), 1)

    def test_cached_search_receives_cancellation_messages(self):
        result = node(dict(kind='cached-cancel'))
        self.assertTrue(result['received'])
        self.assertTrue(result['stopped'])
        self.assertLess(result['completed'], 128)
        self.assertGreater(result['hits'], 0)
        self.assertEqual(result['forwards'], 0)
        self.assertEqual(result['retry'], dict(completed=128, unchanged=True))

    def test_analysis_failure_stays_visible_and_can_be_retried(self):
        result = node(dict(kind='analysis-failure'))
        message = 'Analysis failed: Temporary inference failure'
        self.assertEqual(result['failure'], dict(stage=message, title=message, retry='Retry analysis',
                                               requests=0, notices=['Temporary inference failure'], calls=1))
        self.assertEqual(result['retry'], ['/analyse', dict(ply=1, force=True)])
        self.assertEqual(result['recovered'], dict(calls=2, value=.5, stage='', label='Analyse again'))
        self.assertEqual(result['queued'], dict(stage='Waiting for engine', progress='visible', label='Cancel analysis'))
        self.assertEqual(result['afterRetry'], '')

    def test_analysis_bar_uses_the_mover_at_half_turn_positions(self):
        history = [[0, 0], [1, 0], [1, 1], [-1, 0]]
        cases = [dict(history=history[:length], value=value, live=live)
                 for length in (1, 2, 3, 4) for value in (.07, .93) for live in (True, False)]
        for case, bar in zip(cases, node(dict(kind='analysis-bar', cases=cases))):
            with self.subTest(**case):
                x = case['value'] if play.player_at(len(case['history'])) == 0 else 1 - case['value']
                self.assertEqual((bar['x'], bar['o']), (round(100 * x), round(100 * (1 - x))))
                self.assertAlmostEqual(float(bar['transform'][7:-1]), x)
        cases = [dict(history=history[:2], value=.001 if live else 0, node_value=.001, top=[[2, 2, 1, 0, -1]], live=live)
                 for live in (True, False)]
        for bar in node(dict(kind='analysis-bar', cases=cases)):
            self.assertEqual((bar['x'], bar['o']), ('>99', '<1'))

    def test_solver_leaves_prove_a_losing_half_turn(self):
        history = [[0, 0], [4, 0], [7, 0], [-2, 0], [-1, 0], [1, 0], [6, 0], [5, 0], [-1, -1],
                   [-3, 1], [-1, 1], [-2, -1], [-4, 0], [-3, 0], [0, -1], [-2, -3], [-2, -2],
                   [-2, 1], [-2, -5], [-3, -1], [-1, -3], [-5, 1], [0, -4], [-4, 1], [-4, -1], [-5, -1]]
        found = node(dict(kind='proof-search', history=history, simulations=2048, nodes=524288))
        self.assertEqual((found['exact_winner'], found['proven']), (0, -1))
        self.assertGreater(found['proof_plies'], 0)
        self.assertLess(found['completed'], 2048)
        self.assertTrue(all(v == -1. for v in found['values']))
        self.assertLessEqual(found['nodes_used'], 524288)
        self.assertGreater(found['queries'], 0)
        turn = node(dict(kind='worker-turn', history=history, simulations=2048, nodes=0, leafNodes=524288, leafQueryMs=100))
        self.assertEqual((turn['proof']['winner'], turn['value']), (0, 0.))
        self.assertLess(turn['actual_completed'], 2048)
        self.assertTrue(all(row[3:] == [0., -1] for row in turn['top']))

    def test_artefacts_match_their_sources(self):
        record = json.loads((ENGINE/'build.json').read_text(encoding='utf-8'))
        self.assertEqual(record['sources'], build_web.sources())
        self.assertEqual(record['artefacts'], {name: build_web.digest(ENGINE/name) for name in record['artefacts']})
        for name, digest in record['artefacts'].items():
            if name.endswith('.wasm'):
                self.assertEqual(digest, hashlib.sha256((ENGINE/name).read_bytes()).hexdigest())

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

    def test_review_matches_play(self):
        rng = np.random.default_rng(5)
        history = [[0, 0], [1, 0], [1, 1], [-1, 0], [-2, 0], [2, 2], [3, 3], [4, 4]]
        win = lambda side: dict(winner=side, turns=1, plies=2)
        cases = []
        for case in range(24):
            length = int(rng.integers(1, len(history) + 1))
            evaluations = []
            for ply in range(length + 1):
                if rng.random() < .15:
                    continue
                moves = history[ply:ply + 2] if ply < length and rng.random() < .3 else rng.integers(5, 9, (2, 2)).tolist()
                proof = win(int(rng.integers(2))) if rng.random() < .2 else None
                evaluations.append([history[:ply], dict(value=round(float(rng.random()), 4), moves=moves, proof=proof,
                                                        pv=[[*moves[0], 0, 1]] if proof and rng.random() < .5 else [])])
            cases.append(dict(history=history[:length], evaluations=evaluations, winner=int(rng.integers(-1, 2))))
        cases.append(dict(history=[], evaluations=[[[], dict(value=.5, moves=[[0, 0]], proof=None, pv=[])]], winner=-1))
        found = node(dict(kind='review', cases=cases))
        for case, answer in zip(cases, found):
            table = {json.dumps(prefix): record for prefix, record in case['evaluations']}
            expected = play.review(case['history'], lambda prefix: table.get(json.dumps([list(p) for p in prefix])),
                                   case['winner'])
            self.assertEqual(answer, json.loads(json.dumps(expected)), case)

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
        self.assertEqual(answers[-1]['asked'][:len(history)], list(range(len(history) - 1, -1, -1)))
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
        self.assertEqual(result['imported'], dict(status=200, history=[[0, 0]], paused=True, saved=[[0, 0]], renewed=True))
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

    def test_a_refresh_never_stales_other_positions(self):
        result = node(dict(kind='stale-loop', history=[[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]))
        # The analysis at 4 refreshes 2 at a quarter of its budget; that refresh stales nothing, so nothing is left to redo.
        self.assertEqual(result['asked'], [[2, 4], [4, 4], [2, 1]])
        self.assertEqual(result['stale'], [])
        self.assertEqual(result['asked_after_view'], result['asked'])
        self.assertEqual(result['stale_after_view'], [])
        # A refresh whose value keeps moving runs REFRESH_ROUNDS times at 2 and then settles.
        result = node(dict(kind='stale-loop', history=[[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], drift=.1))
        self.assertEqual(result['asked'], [[2, 4], [4, 4], [2, 1], [2, 1], [2, 1]])
        self.assertEqual(result['stale'], [])

    def test_restored_budget_game_is_not_paused(self):
        result = node(dict(kind='restore-pause', clock=dict(mode='game', tc='60+1')))
        self.assertFalse(result['budget']['before'])
        self.assertFalse(result['budget']['after'])
        self.assertTrue(result['clocked']['clock'])
        self.assertTrue(result['clocked']['after'])

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
        self.assertGreater(answer['undone'][0], 0)
        self.assertEqual(answer['fresh'][0], 0)
        self.assertEqual(answer['lines'], ['b', 'c', 'd'])

    def test_round_admission_proof_retirement_and_normalized_policy(self):
        found = node(dict(kind='rounds'))
        self.assertEqual(found['blocked'],0)
        self.assertTrue(found['extra'])
        self.assertTrue(found['replacement'])
        self.assertEqual(found['retired']['pending'],0)
        self.assertEqual(found['completed'],32)
        self.assertAlmostEqual(found['mass'],1.)
        self.assertEqual(len(found['action']),2)

    def test_independent_views_share_proofs_and_keep_sampling_credits(self):
        from tests.test_neural_search import recorded_position
        found = node(dict(kind='views', history=recorded_position(11)))
        self.assertEqual(found['after'], dict(found['before'], views=2))
        self.assertEqual(found['beforeCredits'], found['afterCredits'])
        self.assertEqual(sum(found['viewCredits']), 32)
        self.assertEqual(found['survived']['completed'], 8)
        self.assertEqual(found['survived']['views'], 1)
        self.assertEqual(found['retired']['pending'], 0)
        self.assertEqual(found['retired']['retired'], 1)
        self.assertEqual(found['proofValue'], -1.)

    def test_dormant_evidence_reconnects_after_different_turn_order(self):
        found = node(dict(kind='archive'))
        self.assertEqual(found['reused'], found['first'])
        self.assertEqual(sum(found['reused']), 128)
        self.assertEqual(sum(found['after']), 136)
        self.assertEqual(sum(found['credits']), 8)
        self.assertEqual(found['counters']['pending'], 0)
        self.assertGreater(found['archive']['reused'], 0)
        self.assertGreater(found['archive']['discarded'], 0)
        self.assertLessEqual(found['archive']['bytes'], found['archive']['limit'])
        for case in found['conflicts']:
            expected = 0 if case['forward'] and case['opposite'] else 1 if case['forward'] else 2
            self.assertEqual(case['before']['nodes'], 2)
            self.assertLess(case['before']['bytes'], case['before']['limit'])
            self.assertEqual(case['after']['nodes'], expected)
            self.assertEqual(case['after']['discarded']-case['before']['discarded'], 2-expected)
            self.assertEqual(case['counters']['pending'], 0)
        for case in found['ownerConflicts']:
            self.assertEqual(case['before']['nodes'], 2)
            self.assertLess(case['before']['bytes'], case['before']['limit'])
            self.assertEqual(case['after']['focus_stones'], 4)
            self.assertEqual(case['after']['nodes'], 0 if case['forward'] else 2)
            self.assertEqual(case['after']['discarded']-case['before']['discarded'], 2 if case['forward'] else 0)
            self.assertEqual(case['counters']['pending'], 0)
            self.assertEqual(case['counters']['views'], 1)
        for growth in (found['proofGrowth'], found['leafProofGrowth']):
            self.assertGreaterEqual(growth['before']['nodes'], 2)
            self.assertGreater(growth['after']['discarded'], growth['before']['discarded'])
            self.assertLessEqual(growth['after']['bytes'], growth['after']['limit'])
            self.assertEqual(growth['returnedWinner'], growth['winner'])

    def test_a_root_reads_and_resumes_its_deeper_branch(self):
        """A -> B -> A in the browser's GameGraph: after B is searched as a root and found lost for A's mover, A's
        statistics for B hold that, A stops preferring B, and A's next search continues its counts."""
        found = node(dict(kind='revisit', history=export_web.histories(every=9)[5]))
        first, back, again = found['first'], found['back'], found['again']
        stone = [j for j, p in enumerate(first['policy']) if p == max(first['policy'])][0]
        self.assertGreater(first['policy'][stone], .5)
        self.assertLess(back['completed_q'][stone], first['completed_q'][stone]-.5)
        self.assertLess(back['policy'][stone], .5)
        self.assertGreaterEqual(back['visits'][stone], 1024)
        self.assertNotEqual(again['action'], first['action'])
        self.assertEqual(sum(again['visits']), sum(back['visits'])+32)

    def test_search_matches_native(self):
        """Same seed, position, budget, Q range floor, root noise and evaluations: the same actions, visits and policy
        as the native library, on trees and on a shared game graph whose root moves to the position after the turn
        it chose and back."""
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
                        for q, r, winner, distance in option['marks']:
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
        recorder, history, steps, results = Recorder(model), games[5], [], []
        graph = GameGraph(recorder, 'test', history, seed=1740, cache=EvaluationCache(), tactics=True, limit=4096)
        try:
            for simulations in (64, 64, 32):
                at = None if not results else graph.after_turn(results[0]) if len(results) == 1 else history
                if at is not None:
                    graph.at(at)
                results.append(graph.search(simulations, root_samples=16, batch_size=16))
                steps.append(dict(simulations=simulations, root_samples=16, batch_size=16, **({'at': at} if at else {})))
        finally:
            graph.close()
        self.assertGreater(len(steps[1]['at']), len(history))
        cases.insert(0, (dict(history=history, seed=1740, tactics=True, limit=4096, steps=steps,
                              batches=recorder.batches), results))
        history = [[0,0],[1,2],[2,2],[0,-2],[-2,0],[3,2],[4,2]]
        a, b = [0,-3], [-3,0]
        recorder = Recorder(model)
        graph = GameGraph(recorder, 'permutation', history, seed=1740, tactics=False, limit=4096)
        try:
            graph.expand()
            graph.mark(a, 1, 4)
            first = graph.search(16, root_samples=4, batch_size=4)
            graph.at(history+[b])
            second = graph.search(16, root_samples=4, batch_size=4)
            self.assertEqual(second['policy'][second['actions'].tolist().index(a)], 0.)
            steps = [dict(simulations=16, root_samples=4, batch_size=4, marks=[[*a,1,4]]),
                     dict(simulations=16, root_samples=4, batch_size=4, at=history+[b])]
            cases.insert(0, (dict(history=history, seed=1740, tactics=False, limit=4096, steps=steps,
                                  batches=recorder.batches), [first, second]))
        finally:
            graph.close()
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
