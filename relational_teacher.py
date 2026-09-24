"""Bounded critic-only initialization from explicitly terminal tactical teachers.

These are Q_teacher labels, never replacements for policy-conditional KLENT
returns. Unknown actions remain unlabeled. The backbone and policy stay frozen.
"""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from hexo import Game
from human_corpus import AXES, owner, snapshot_key
from klent import digest, improved_policy, publish
from relational_data import human_examples
from relational_train import graph, load_model, outputs, save_model, source_identity
from train import write_json


def verify_terminal(history, action, reply, target):
    """Independent Python replay including legality, phase and first terminal."""
    moves=history+[action]+([reply] if reply is not None else [])
    actor=owner(len(history))
    board={}
    winner=None
    for ply,(q,r) in enumerate(moves):
        if winner is not None:raise ValueError('Teacher continued after terminal')
        if (q,r) in board or max(abs(q),abs(r))>10**12:
            raise ValueError('Illegal teacher coordinate')
        if not board:
            if (q,r)!=(0,0):raise ValueError('Noncentral opening')
        elif not any(max(abs(q-a),abs(r-b),abs(q-a+r-b))<=8 for a,b in board):
            raise ValueError('Teacher exceeds placement radius')
        player=owner(ply)
        board[q,r]=player
        for dq,dr in AXES:
            count=1
            for sign in (-1,1):
                k=1
                while board.get((q+sign*k*dq,r+sign*k*dr))==player:
                    count+=1;k+=1
            if count>=6:winner=player
    if winner is None or (1 if winner==actor else -1)!=target:
        raise ValueError('Teacher target disagrees with independently replayed terminal')
    if reply is not None and owner(len(history)+1)==actor:
        raise ValueError('Negative teacher reply must belong to opponent')


def replay(history):
    game=Game()
    for action in history:game.play(*action)
    return game


def examples(args):
    histories,rows,data_identity=human_examples(args.corpus)
    fixture_bytes=Path(args.fixtures).read_bytes()
    fixtures=json.loads(fixture_bytes)['positions']
    excluded={snapshot_key(p['history']) for p in fixtures}
    rng=np.random.default_rng(args.seed)
    result={'train':[],'validation':[]}
    selected_families={}
    for split,count in [('train',args.train_families),('validation',args.validation_families)]:
        metadata={r['game']:r['family'] for r in rows[split]}
        keys=sorted(k for k in metadata if len(histories[k])%2==0)
        keys=[keys[int(i)] for i in rng.permutation(len(keys))]
        used=set()
        for key in keys:
            family=metadata[key]
            history=histories[key]
            if family in used or any(snapshot_key(history[:-n]) in excluded for n in (1,2)):
                continue
            pair=[]
            for kind,prefix in [('win',history[:-1]),('reply_loss',history[:-2])]:
                with closing(replay(prefix)) as game:
                    actions=[list(a) for a in game.legal_moves()]
                    labels=[]
                    order=range(len(actions)) if kind=='win' else rng.permutation(len(actions))
                    for index in order:
                        action=actions[int(index)]
                        game.play(*action)
                        reply=None
                        valid=game.winner>=0 if kind=='win' else False
                        if kind=='reply_loss' and game.winner<0 and game.legal(*history[-1]):
                            reply=history[-1]
                            game.play(*reply)
                            valid=game.winner>=0
                            game.undo()
                        game.undo()
                        if valid:
                            target=1 if kind=='win' else -1
                            verify_terminal(prefix,action,reply,target)
                            labels.append(dict(action=action,target=target,reply=reply))
                        if kind=='reply_loss' and len(labels)==args.negative_actions:break
                if not labels:raise ValueError('Source terminal teacher yielded no verified examples')
                pair.append(dict(history=prefix,kind=kind,game=key,family=family,labels=labels))
            result[split].extend(pair)
            used.add(family)
            if len(used)==count:break
        if len(used)!=count:raise ValueError(f'Only {len(used)} eligible {split} families')
        selected_families[split]=sorted(used)
    if set(selected_families['train'])&set(selected_families['validation']):
        raise ValueError('Teacher family leakage')
    return result,dict(data=data_identity,families=selected_families,
                      fixtures_sha256=hashlib.sha256(fixture_bytes).hexdigest(),
                      teacher='Own terminal action, or recorded verified opponent terminal reply after actor last placement',
                      target='Q_teacher terminal outcome; not Q_mu or substituted KLENT return',
                      excluded='All 38 fixed diagnostic positions under translation and twelve symmetries')


def evaluate(model,rows,args):
    result=[]
    with torch.no_grad():
        for row in rows:
            item=graph(row['history'],model,args)
            batch,logits,q=outputs(model,[item],args)
            mu,*_=improved_policy(logits,q,batch['action_owner'],1,.03,.1)
            actions=item.actions.tolist()
            labels={tuple(v['action']):v['target'] for v in row['labels']}
            indexes=[i for i,a in enumerate(actions) if tuple(a) in labels]
            record=dict(kind=row['kind'],family=row['family'],position_key=item.position_key,
                actions=actions,logits=logits.cpu().tolist(),q=q.cpu().tolist(),mu=mu.cpu().tolist(),
                labels=row['labels'],q_argmax_labeled=int(q.argmax()) in indexes,
                pi_argmax_labeled=int(logits.argmax()) in indexes,mu_argmax_labeled=int(mu.argmax()) in indexes)
            if row['kind']=='reply_loss':
                try:
                    verify_terminal(row['history'],actions[int(mu.argmax())],row['labels'][0]['reply'],-1)
                    record['mu_allows_teacher_terminal_reply']=True
                except ValueError:
                    record['mu_allows_teacher_terminal_reply']=False
            result.append(record)
    return result


def fit(args):
    output=Path(args.output)
    if output.exists():raise ValueError('New teacher output directory required')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    sources=source_identity(('relational_teacher.py','relational_data.py','human_corpus.py'))
    selected,identity=examples(args)
    fixed=[dict(history=p['history'],kind=p['category'],family=None,
                labels=[dict(action=a,target=None) for a in p['good_actions']])
           for p in json.loads(Path(args.fixtures).read_bytes())['positions']]
    model_sha=digest(Path(args.checkpoint))
    model=load_model(args.checkpoint,args.device,expected_sha256=model_sha).eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    for parameter in model.critic.parameters():parameter.requires_grad_(True)
    frozen={name:value.detach().cpu().clone() for name,value in model.state_dict().items() if not name.startswith('critic.')}
    predeclared=dict(train_families=args.train_families,validation_families=args.validation_families,
        passes=1,lr=args.lr,negative_actions=args.negative_actions,seed=args.seed,
        expected='Held-out immediate-win Q argmax at least 6/8; held-out reply-loss selected-label mu errors do not increase; restore at least 8/10 fixed pi-good/mu-bad positions without new regressions; frozen policy outputs unchanged',
        limitations='Selected reply-loss actions are proven teacher losses. Unlabeled alternatives have unknown eventual value. Loss/sign improvements alone do not pass action-ranking criteria.',
        followup='Same-checkpoint paired pi/mu/Gumbel matches against pinned Seal; no promotion from diagnostic-only results')
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output.with_suffix('.plan.json'),dict(**predeclared,model_sha256=model_sha,**identity))
    write_json(output.with_suffix('.examples.json'),selected)
    before=evaluate(model,selected['validation'],args)
    fixed_before=evaluate(model,fixed,args)
    cpu_args=argparse.Namespace(**(vars(args)|{'device':'cpu'}))
    probe=graph(selected['validation'][0]['history'],model,cpu_args)
    model.to('cpu')
    with torch.no_grad():cpu_policy_before=outputs(model,[probe],cpu_args)[1].clone()
    model.to(args.device)
    optimizer=torch.optim.Adam(model.critic.parameters(),lr=args.lr)
    order=np.random.default_rng(args.seed).permutation(len(selected['train']))
    losses=[]
    started=time.perf_counter()
    for step,index in enumerate(order):
        row=selected['train'][int(index)]
        item=graph(row['history'],model,args)
        _,_,q=outputs(model,[item],args)
        labels={tuple(v['action']):v['target'] for v in row['labels']}
        indexes=[i for i,a in enumerate(item.actions) if tuple(a) in labels]
        targets=torch.tensor([labels[tuple(item.actions[i])] for i in indexes],device=args.device)
        loss=(q[indexes]-targets).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.critic.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        write_json(output.with_suffix('.status.json'),dict(stage='critic_fitting',completed=step+1,total=len(order),loss=losses[-1]))
    after=evaluate(model,selected['validation'],args)
    fixed_after=evaluate(model,fixed,args)
    for name,value in model.state_dict().items():
        if name in frozen and not torch.equal(value.cpu(),frozen[name]):
            raise ValueError('Frozen backbone or policy changed')
    repeated_policy_max_error=max(float(np.max(np.abs(np.asarray(a['logits'])-b['logits'])))
                                 for a,b in zip(before+fixed_before,after+fixed_after,strict=True))
    model.to('cpu')
    with torch.no_grad():cpu_policy_after=outputs(model,[probe],cpu_args)[1]
    if not torch.equal(cpu_policy_before,cpu_policy_after):
        raise ValueError('Frozen CPU FP32 policy probe changed')
    if sources!=source_identity(('relational_teacher.py','relational_data.py','human_corpus.py')):
        raise ValueError('Teacher source or native library changed')
    if digest(Path(args.fixtures))!=identity['fixtures_sha256']:
        raise ValueError('Teacher exclusion fixtures changed')
    identity.update(initial_model_sha256=model_sha,config=vars(args),**sources)
    report=dict(seconds=time.perf_counter()-started,mean_training_loss=float(np.mean(losses)),
                frozen_parameters_unchanged=True,cpu_policy_probe_unchanged=True,
                repeated_cuda_policy_max_error=repeated_policy_max_error,promotion=False)
    def writer(stage):
        save_model(stage/'model.pt',model)
        write_json(stage/'before.json',before);write_json(stage/'after.json',after)
        write_json(stage/'fixed_before.json',fixed_before);write_json(stage/'fixed_after.json',fixed_after)
        write_json(stage/'report.json',report);write_json(stage/'examples.json',selected)
        write_json(stage/'plan.json',predeclared)
    publish(output,identity,writer,report)
    write_json(output.with_suffix('.status.json'),dict(stage='finished',**report))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','corpus','fixtures','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--train-families',type=int,default=32)
    parser.add_argument('--validation-families',type=int,default=8)
    parser.add_argument('--negative-actions',type=int,default=16)
    parser.add_argument('--seed',type=int,default=1730)
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--max-nodes',type=int,default=12000)
    parser.add_argument('--max-edges',type=int,default=1000000)
    fit(parser.parse_args())
