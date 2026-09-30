import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from legacy.human_corpus import digest as history_digest, owner
from legacy.klent import digest
from legacy.relational_data import human_examples
from legacy.relational_diagnostics import evaluate as diagnose
from legacy.relational_model import ModelConfig, RelationalNet
from legacy.relational_train import collect, fit, load_model, main, rebuild, save_model, graph
from legacy.relational_warmstart import fit as warmstart
from tests.test_human_corpus import record


def config():
    return ModelConfig(width=16,blocks=1,heads=2,ff=32,global_tokens=2,edge_chunk=1024)


def fixture(root):
    root.mkdir()
    a=record()
    b=record()
    b['moves']=[[0,0],[0,3],[1,3],[2,0],[4,0],[2,3],[3,3],[6,0],[8,0],[4,3],[5,3]]
    b['winner']=-1
    records=[];shards=[]
    for item,split,family in ((a,'train',101),(b,'validation',100)):
        item.update(split=split,family=family,content_sha256=history_digest(item['moves']))
        records.append(item)
        (root/split).mkdir()
        path=root/split/'0000.npz'
        np.savez_compressed(path,family=np.asarray([family]),search_valid=np.asarray([False]),
                            policy_valid=np.asarray([True]),outcome=np.asarray([1]),search=np.asarray([1]))
        shards.append(dict(path=f'{split}/0000.npz',sha256=digest(path),positions=1,
                           games=[item['content_sha256']],game_row_counts=[1]))
    # Unreadable held-out shards must not become training inputs.
    for split in ('test','excluded'):
        shards.append(dict(path=f'{split}/absent.npz',sha256='missing'))
    (root/'manifest.json').write_text(json.dumps(dict(minimum_ply=7,shards=shards)))
    (root/'games.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    return records


class RelationalLearningTests(unittest.TestCase):
    def test_stratified_rejects_invalid_clipping_and_unsupported_previous_kind(self):
        from legacy.relational_stratified import fit as stratified
        for clip in (0,-1,float('inf'),float('nan')):
            with self.assertRaisesRegex(ValueError,'Positive fitting'):
                stratified(self.args(grad_clip=clip))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);previous=root/'previous';previous.mkdir()
            (previous/'manifest.json').write_text(json.dumps({'identity':{'kind':'relational-human-stratified-v1'}}))
            args=self.args(output=str(root/'next'),previous_run=str(previous))
            with patch('legacy.relational_stratified.source_identity',return_value={}),patch('legacy.relational_stratified.verify'):
                with self.assertRaisesRegex(ValueError,'initial human warmstart'):
                    stratified(args)

    def test_stratified_selection_excludes_replay_and_enforces_both_quotas(self):
        from collections import Counter
        from legacy.relational_stratified import select
        histories={key:[None]*20 for key in ('a','b','c')}
        rows=[dict(game=key,ply=ply) for key in histories for ply in range(20)]
        previous=rows[:3]+rows[17:20]
        chosen=select(histories,rows,previous,12,4,19)
        self.assertEqual(chosen,select(histories,rows,previous,12,4,19))
        self.assertEqual(sum(20-r['ply']<=8 for r in chosen),6)
        self.assertEqual(len({(r['game'],r['ply']) for r in chosen}),12)
        self.assertFalse({(r['game'],r['ply']) for r in chosen}&{(r['game'],r['ply']) for r in previous})
        self.assertLessEqual(max(Counter(r['game'] for r in chosen).values()),4)
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            select(histories,rows,previous,14,4,19)
        histories={'long':[None]*20,'short':[None]*8}
        rows=[dict(game=key,ply=ply) for key,history in histories.items() for ply in range(len(history))]
        for seed in range(10):
            selected=select(histories,rows,[],2,1,seed)
            self.assertEqual([r['game'] for r in selected],['short','long'])

    def setUp(self):
        threads=torch.get_num_threads();torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads,threads)
        torch.manual_seed(9)

    def args(self,**overrides):
        args=dict(device='cpu',seed=73,games=2,envs=2,max_plies=4,batch=2,max_nodes=12000,
                  max_edges=250000,alpha=.03,beta=.1,gamma=1.,lambda_return=.939413,
                  lr=.0001,grad_clip=1.,iterations=1,allow_cold_start=False)
        return SimpleNamespace(**(args|overrides))

    def test_full_history_labels_preserve_split_phase_and_missing_actions(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'corpus';fixture(root)
            histories,rows,meta=human_examples(root)
            self.assertEqual(meta['available'],dict(train=5,validation=4))
            for split,examples in rows.items():
                for row in examples:
                    self.assertEqual(row['player'],owner(row['ply']))
                    self.assertEqual(row['action'],histories[row['game']][row['ply']])
                    expected=(1 if split=='train' else -1)*(1 if row['player']==0 else -1)
                    self.assertEqual(row['target'],expected)
                    self.assertNotIn('q_all_actions',row)
            text=(root/'games.jsonl').read_text().replace('"family": 101','"family": 105')
            (root/'games.jsonl').write_text(text)
            with self.assertRaisesRegex(ValueError,'split membership'):human_examples(root)

    def test_declared_conversion_limit_skips_only_unsharded_histories(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'corpus';records=fixture(root)
            extra=copy.deepcopy(records[0])
            extra['content_sha256']='unconverted-game'
            history=root/'games.jsonl'
            original=history.read_text()
            history.write_text(original+'\n'+json.dumps(extra))
            with self.assertRaisesRegex(ValueError,'absent from verified split'):
                human_examples(root)
            manifest=root/'manifest.json'
            metadata=json.loads(manifest.read_text())
            metadata['conversion']={'limit_games_per_split':1}
            manifest.write_text(json.dumps(metadata))
            histories,rows,_=human_examples(root)
            self.assertEqual(set(histories),{r['content_sha256'] for r in records})
            self.assertEqual(len(rows['train']),5)
            history.write_text(original+'\n'+json.dumps(records[0]))
            with self.assertRaisesRegex(ValueError,'Duplicate converted'):
                human_examples(root)
            history.write_text(json.dumps(records[0]))
            with self.assertRaisesRegex(ValueError,'Missing allowed'):
                human_examples(root)

    def test_diagnostics_hash_parsed_fixture_and_reject_source_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);model=RelationalNet(config())
            checkpoint=root/'model.pt';save_model(checkpoint,model)
            moves=record()['moves'][:7];item=graph(moves,model)
            fixtures=root/'fixtures.json'
            fixtures.write_text(json.dumps({'positions':[dict(history=moves,player=item.player,
                remaining=item.remaining,all_legal_actions=item.actions.tolist(),
                good_actions=[item.actions[0].tolist()],category='fixture',source='test')]}))
            expected=digest(fixtures)
            args=self.args()
            def changed_file(*unused):
                fixtures.write_text('{}')
                return {'source':'same'}
            with patch('legacy.relational_diagnostics.source_identity',side_effect=changed_file):
                result=diagnose(checkpoint,fixtures,args)
            self.assertEqual(result['fixtures_sha256'],expected)
            fixtures.write_text(json.dumps({'positions':[]}))
            with patch('legacy.relational_diagnostics.source_identity',side_effect=[{'source':'before'},{'source':'after'}]):
                with self.assertRaisesRegex(ValueError,'changed during evaluation'):
                    diagnose(checkpoint,fixtures,args)

    def test_terminal_teacher_uses_actual_reply_and_rejects_wrong_outcome(self):
        from legacy.relational_teacher import examples, verify_terminal
        for limit in (0,-1):
            with self.assertRaisesRegex(ValueError,'Positive family'):
                examples(SimpleNamespace(train_families=32,validation_families=8,negative_actions=limit))
        moves=record()['moves']
        verify_terminal(moves[:-1],moves[-1],None,1)
        verify_terminal(moves[:-2],moves[-2],moves[-1],-1)
        with self.assertRaisesRegex(ValueError,'terminal'):
            verify_terminal(moves[:-2],moves[-2],None,-1)
        with self.assertRaisesRegex(ValueError,'terminal'):
            verify_terminal(moves[:-2],moves[-2],moves[-1],1)
        with self.assertRaisesRegex(ValueError,'continued after terminal'):
            verify_terminal(moves[:-1],moves[-1],[99,99],1)

    def test_klent_zero_output_control_and_terminal_threshold(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);model=RelationalNet(config())
            checkpoint=root/'start.pt';save_model(checkpoint,model)
            args=self.args(run=str(root/'run'),initial_model=str(checkpoint),games=1,max_plies=2,
                           critic_initialization='zero-output',min_terminal_fraction=.75)
            main(args)
            initial=load_model(root/'run/checkpoints/0000/model.pt')
            for name,value in model.state_dict().items():
                if name.startswith('critic.3.'):
                    self.assertTrue(torch.equal(initial.state_dict()[name],torch.zeros_like(value)))
                else:torch.testing.assert_close(initial.state_dict()[name],value,rtol=0,atol=0)
            status=json.loads((root/'run/status.json').read_text())
            self.assertEqual(status['stage'],'collection_insufficient')
            self.assertFalse(status['fitting_started'])
            self.assertEqual(status['terminal_games'],0)
            self.assertTrue((root/'run/corpus/0001/manifest.json').exists())
            self.assertFalse((root/'run/checkpoints/0001').exists())
            main(args)
            self.assertFalse((root/'run/checkpoints/0001').exists())

    def test_zero_row_short_game_membership_remains_untrained(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'corpus';records=fixture(root)
            short=copy.deepcopy(records[1]);short.update(split='train',family=103)
            longer=copy.deepcopy(records[0])
            longer['moves']=[[r,q] for q,r in longer['moves']]
            longer.update(split='validation',family=100,content_sha256=history_digest(longer['moves']))
            manifest=root/'manifest.json';metadata=json.loads(manifest.read_text())
            metadata['minimum_ply']=11
            metadata['shards'][0]['games'].append(short['content_sha256'])
            metadata['shards'][0]['game_row_counts'].append(0)
            metadata['shards'][1]['games']=[longer['content_sha256']]
            manifest.write_text(json.dumps(metadata))
            (root/'games.jsonl').write_text('\n'.join(json.dumps(r) for r in [records[0],short,longer]))
            histories,rows,_=human_examples(root)
            self.assertNotIn(short['content_sha256'],histories)
            self.assertEqual({k:len(v) for k,v in rows.items()},{'train':1,'validation':1})
            metadata['minimum_ply']=7;manifest.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'Zero-row corpus game has eligible'):
                human_examples(root)

    def test_frozen_capped_actor_full_legal_targets_and_one_fit_pass(self):
        model=RelationalNet(config())
        before={k:v.clone() for k,v in model.state_dict().items()}
        args=self.args()
        episodes,rows=collect(model,args,1)
        self.assertEqual(len(rows),8)
        self.assertTrue(all(e['reason']=='cap' and e['winner']==-1 for e in episodes))
        self.assertTrue(all(r['bootstrapped'] for r in rows))
        self.assertGreater(max(len(r['mu']) for r in rows),200)
        for key,value in model.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
        for row in rows:rebuild(row,{e['id']:e for e in episodes},model)
        optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
        metrics=fit(model,optimizer,episodes,rows,args,1)
        self.assertEqual(metrics['optimizer_steps'],4)
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()))
        changed=copy.deepcopy(rows[1]);changed['action']=[999,999]
        with self.assertRaises(ValueError):rebuild(changed,{e['id']:e for e in episodes},model)

    def test_human_both_heads_checkpoint_roundtrip_and_resume_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fixture(root/'corpus')
            args=self.args(corpus=root/'corpus',output=root/'warm',positions=4,epochs=1)
            report=warmstart(args,config())
            self.assertFalse(report['promotion'])
            model=load_model(root/'warm/model.pt')
            self.assertEqual(model.config,config())
            with self.assertRaisesRegex(ValueError,'changed before loading'):
                load_model(root/'warm/model.pt',expected_sha256='wrong')
            args=self.args(run=str(root/'run'),initial_model=str(root/'warm/model.pt'),games=1,max_plies=2)
            main(args)
            checkpoint=root/'run/checkpoints/0001/model.pt'
            original=digest(checkpoint)
            main(args)
            self.assertEqual(original,digest(checkpoint))
            data=root/'run/corpus/0001/rows.json'
            data.write_text(data.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'hash changed'):main(args)
            self.assertFalse((root/'run/training.lock').exists())

    def test_terminal_reward_and_actual_control_change(self):
        from legacy.relational_encoder import pack
        model=RelationalNet(config())
        moves=record()['moves']
        ply=[0]
        def scripted(model,graphs,args):
            batch=pack(graphs,args.device)
            logits=torch.full((len(graphs[0].actions),),-1000.)
            chosen=np.flatnonzero(np.all(graphs[0].actions==moves[ply[0]],axis=1))[0]
            logits[chosen]=1000
            ply[0]+=1
            return batch,logits,torch.full_like(logits,.25)
        args=self.args(games=1,envs=1,max_plies=20)
        with patch('legacy.relational_train.outputs',side_effect=scripted):
            episodes,rows=collect(model,args,1)
        self.assertEqual(episodes[0]['winner'],0)
        self.assertEqual(len(rows),12)
        self.assertEqual(rows[-1]['target'],1.)
        self.assertEqual(rows[-2]['player'],1)
        self.assertAlmostEqual(rows[-2]['target'],-((1-args.lambda_return)*.25+args.lambda_return),places=6)
        self.assertFalse(any(r['bootstrapped'] for r in rows))

    def test_graph_microbatch_preserves_position_weighted_gradients(self):
        left=RelationalNet(config())
        right=copy.deepcopy(left)
        args=self.args(games=1,envs=1,max_plies=3,batch=3,grad_clip=1000000.)
        episodes,rows=collect(left,args,1)
        graphs=[graph(episodes[0]['moves'][:r['ply']],left) for r in rows]
        split=copy.copy(args)
        split.max_nodes=max(g.node_count for g in graphs)
        split.max_edges=max(g.edge_count for g in graphs)
        fit(left,torch.optim.SGD(left.parameters(),lr=0),episodes,rows,args,1)
        fit(right,torch.optim.SGD(right.parameters(),lr=0),episodes,rows,split,1)
        for a,b in zip(left.parameters(),right.parameters(),strict=True):
            if a.grad is not None:
                torch.testing.assert_close(a.grad,b.grad,rtol=3e-4,atol=3e-6)


if __name__=='__main__':unittest.main()
