"""Search-trained relational policy/value self-play and an internal checkpoint league."""
import argparse
from contextlib import closing
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from hexo import Game, ROOT, library
from klent import digest, segmented_log_softmax
from neural_search import NeuralSearch, SearchCoordinator, EvaluationCache
from relational_model import RelationalNet, NeuralEvaluator
from relational_train import load_model, save_model, precision, graph, work_batches, VALUE_SCHEMA
from train import write_json, task_opening, paired_metrics

SCHEMA = 'hexo-search-selfplay-v1'


def source_identity():
    names = ('search_train.py','relational_model.py','relational_train.py','relational_native.py',
             'relational_encoder.py','neural_search.py','hexo.py','klent.py','train.py','curriculum.py',
             'src/gumbel.cpp','src/hexo.cpp','src/hexo.hpp','src/relational_graph.cpp')
    files = {name:digest(ROOT/name) for name in names}
    for name in ('hexo','hexo_graph','hexo_gumbel'):
        path = library.with_name(library.name.replace('hexo',name))
        files[str(path.relative_to(ROOT))] = digest(path)
    return files


def verify_artifact(path, expected_identity=None):
    manifest = json.loads((path/'manifest.json').read_text())
    if manifest['schema'] != SCHEMA:
        raise ValueError('Expected a search self-play artifact')
    if expected_identity is not None and manifest['identity'] != expected_identity:
        raise ValueError(f'Artifact run, iteration or actor identity changed: {path}')
    for name, sha in manifest['files'].items():
        if Path(name).name != name or digest(path/name) != sha:
            raise ValueError(f'Artifact changed: {path/name}')
    return manifest


def publish(path, identity, writer, metrics=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise ValueError(f'Artifact already exists: {path}')
    with tempfile.TemporaryDirectory(dir=path.parent,prefix='pending-') as temporary:
        stage=Path(temporary)/'artifact';stage.mkdir();writer(stage)
        write_json(stage/'manifest.json',dict(schema=SCHEMA,identity=identity,metrics=metrics,
            files={p.name:digest(p) for p in stage.iterdir()}))
        stage.rename(path)


def warm_start(path, device, seed, expected_sha256=None):
    torch.manual_seed(seed)
    previous=load_model(path,'cpu',expected_sha256=expected_sha256)
    if previous.config.head=='value':
        return previous.to(device)
    model=RelationalNet(replace(previous.config,head='value'))
    missing,unexpected=model.load_state_dict({k:v for k,v in previous.state_dict().items()
                                             if not k.startswith('critic.')},strict=False)
    if unexpected or any(not name.startswith('value.') for name in missing):
        raise ValueError('Unexpected policy/value warm-start mismatch')
    # Preserve the learned policy exactly. A new scalar value starts neutral;
    # action-Q weights are never reinterpreted as state-outcome values.
    with torch.no_grad():
        model.value[-1].weight.zero_();model.value[-1].bias.zero_()
    return model.to(device)


def play_games(checkpoints, assignments, openings, args, seed, progress, *, training=False):
    """Batch neural leaves across games; every placement comes from Gumbel search."""
    evaluators=[];coordinators=[];versions=[];live=[];episodes=[];rows=[];next_id=0
    try:
        for path in checkpoints:
            versions.append(digest(path))
            model=load_model(path,'cpu',expected_sha256=versions[-1])
            if model.config.head!='value':raise ValueError('Search learning requires a scalar value checkpoint')
            evaluator=NeuralEvaluator(model,args.device,max_nodes=args.max_nodes,max_edges=args.max_edges)
            evaluators.append(evaluator)
            coordinators.append(SearchCoordinator(evaluator,versions[-1],EvaluationCache(args.cache_positions)))
        started=time.perf_counter();completed_placements=0
        while live or next_id<len(assignments):
            while len(live)<args.envs and next_id<len(assignments):
                history=openings[next_id]
                trees=[NeuralSearch(ev,versions[j],history,seed+(next_id if training else next_id//2)*1009+j,
                                   coordinators[j].cache,tactics=False) for j,ev in enumerate(evaluators)]
                live.append(dict(id=next_id,game=Game(history),history=list(history),trees=trees,rows=[]))
                next_id+=1
            # Freeze this sweep's side assignments before applying any move.
            groups=[[e for e in live if assignments[e['id']][e['game'].player]==j] for j in range(len(evaluators))]
            for j,group in enumerate(groups):
                if not group:continue
                results=coordinators[j].search_many([e['trees'][j] for e in group],args.simulations,args.root_samples,args.leaf_batch)
                for e,result in zip(group,results,strict=True):
                    g=e['game'];actions=result['actions'];policy=result['policy']
                    if result['completed']!=args.simulations or result['action'] is None:
                        raise ValueError('Incomplete search cannot produce a training target')
                    if len(policy)!=len(actions) or not np.isfinite(policy).all() or np.any(policy<0) or not np.isclose(policy.sum(),1):
                        raise ValueError('Invalid search policy target')
                    if training:
                        e['rows'].append(dict(game=e['id'],ply=len(e['history']),player=g.player,remaining=g.remaining,
                            action=result['action'],legal_sha256=hashlib.sha256(actions.astype(np.int64).tobytes()).hexdigest(),
                            policy=policy.astype(np.float32),simulations=result['completed'],evaluated=result['evaluated']))
                    action=result['action'];g.play(*action);e['history'].append(action)
                    for tree in e['trees']:tree.advance(action)
                    completed_placements+=1
            finished=[e for e in live if e['game'].winner>=0 or len(e['history'])>=args.max_plies]
            for e in finished:
                winner=e['game'].winner
                episode=dict(id=e['id'],moves=e['history'],opening=openings[e['id']],winner=winner,
                    reason='six-in-a-row' if winner>=0 else 'cap',actors=assignments[e['id']])
                episodes.append(episode)
                # A capped game is not a draw and supplies no outcome target.
                if winner>=0:
                    for row in e['rows']:row['target']=1. if row['player']==winner else -1.
                    rows.extend(e['rows'])
                e['game'].close()
                for tree in e['trees']:tree.close()
            live=[e for e in live if e not in finished]
            progress(dict(completed=len(episodes),total=len(assignments),positions=completed_placements,
                terminal_games=sum(e['winner']>=0 for e in episodes),active_games=len(live),
                placements_per_second=completed_placements/max(.001,time.perf_counter()-started)))
        return sorted(episodes,key=lambda e:e['id']),rows
    finally:
        for e in live:
            e['game'].close()
            for tree in e['trees']:tree.close()
        evaluators.clear();coordinators.clear()
        if args.device=='cuda':torch.cuda.empty_cache()


def save_corpus(path,identity,episodes,rows):
    def writer(stage):
        write_json(stage/'episodes.json',episodes)
        write_json(stage/'rows.json',[{k:v for k,v in r.items() if k!='policy'} for r in rows])
        np.savez_compressed(stage/'targets.npz',offsets=np.cumsum([0]+[len(r['policy']) for r in rows]),
                            probabilities=np.concatenate([r['policy'] for r in rows]))
    publish(path,identity,writer)


def read_corpus(path, expected_identity):
    verify_artifact(path, expected_identity)
    episodes=json.loads((path/'episodes.json').read_text());rows=json.loads((path/'rows.json').read_text())
    with np.load(path/'targets.npz',allow_pickle=False) as data:
        offsets=data['offsets'];probabilities=data['probabilities']
        if len(offsets)!=len(rows)+1 or offsets[0]!=0 or offsets[-1]!=len(probabilities) or np.any(np.diff(offsets)<=0):
            raise ValueError('Malformed search-policy offsets')
        for i,row in enumerate(rows):row['policy']=probabilities[offsets[i]:offsets[i+1]].copy()
    return episodes,rows


def fit(model,optimizer,episodes,rows,args,iteration,progress):
    episodes={e['id']:e for e in episodes}
    families=sorted({r['game'] for r in rows})
    if len(families)<2:raise ValueError('Need at least two terminal games for disjoint validation')
    rng=np.random.default_rng(args.seed+iteration)
    validation=set(rng.permutation(families)[:max(1,len(families)//4)].tolist())
    train=[r for r in rows if r['game'] not in validation];valid=[r for r in rows if r['game'] in validation]
    steps=0;last={}
    for epoch in range(args.epochs):
        for phase,selected in [('train',[train[i] for i in rng.permutation(len(train))]),('validation',valid)]:
            model.train(phase=='train');total=np.zeros(2);completed=0
            for start in range(0,len(selected),args.batch):
                batch_rows=selected[start:start+args.batch]
                graphs=[graph(episodes[r['game']]['moves'][:r['ply']],model,args) for r in batch_rows]
                for row,item in zip(batch_rows,graphs,strict=True):
                    if hashlib.sha256(item.actions.astype(np.int64).tobytes()).hexdigest()!=row['legal_sha256'] or (item.player,item.remaining)!=(row['player'],row['remaining']):
                        raise ValueError('Training position or full legal action order changed')
                if phase=='train':optimizer.zero_grad(set_to_none=True)
                offset=0
                for group in work_batches(graphs,args):
                    from relational_encoder import pack
                    chosen=batch_rows[offset:offset+len(group)];batch=pack(group,args.device)
                    with torch.set_grad_enabled(phase=='train'),precision(args.device):
                        output=model(batch)
                        logpi=segmented_log_softmax(output['logits'],batch['action_owner'],len(group))
                        policy=torch.as_tensor(np.concatenate([r['policy'] for r in chosen]),device=args.device)
                        outcome=torch.tensor([r['target'] for r in chosen],device=args.device)
                        ce=-(policy*logpi).sum()/len(group);mse=(output['value']-outcome).square().mean()
                        loss=ce+mse
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite search-learning loss')
                    if phase=='train':(loss*len(group)/len(batch_rows)).backward()
                    total+=np.array([ce.item(),mse.item()])*len(group);offset+=len(group)
                if phase=='train':
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();steps+=1
                completed+=len(batch_rows)
                progress(dict(epoch=epoch+1,epochs=args.epochs,fit_phase=phase,completed=completed,total=len(selected),
                    policy_ce=total[0]/completed,value_mse=total[1]/completed,optimizer_steps=steps))
            last[phase]=dict(policy_ce=total[0]/completed,value_mse=total[1]/completed,positions=completed)
    return dict(**last,optimizer_steps=steps,validation_game_ids=sorted(validation),training_games=len(families)-len(validation))


def evaluate(run,candidate,opponent,args,progress):
    path=run/'evaluation'/f'{candidate:04d}-vs-{opponent:04d}'
    a=run/'checkpoints'/f'{candidate:04d}'/'model.pt';b=run/'checkpoints'/f'{opponent:04d}'/'model.pt'
    run_identity=json.loads((run/'config.json').read_text())
    verify_artifact(a.parent,dict(run_identity,checkpoint=candidate))
    verify_artifact(b.parent,dict(run_identity,checkpoint=opponent))
    identity=dict(run=run_identity,candidate=digest(a),opponent=digest(b),simulations=args.simulations,root_samples=args.root_samples,
                  games=args.eval_games,seed=args.seed+100000+candidate*1000,opening_suite='standard-v1')
    if path.exists():
        verify_artifact(path,identity)
        return json.loads((path/'report.json').read_text())
    openings=[task_opening(identity['seed']+i//2,True,args.max_plies,'standard-v1')['opening'] for i in range(args.eval_games)]
    assignments=[(0,1) if i%2==0 else (1,0) for i in range(args.eval_games)]
    episodes,_=play_games([a,b],assignments,openings,args,identity['seed'],progress)
    games=[dict(index=e['id'],pair=e['id']//2,seed=identity['seed']+e['id']//2,challenger_color=e['id']%2,winner=e['winner'],reason=e['reason'],
                opening=e['opening'],moves=e['moves']) for e in episodes]
    metrics=paired_metrics(games,args.eval_games)
    point=400*math.log10((metrics['wins']+.5)/(metrics['losses']+.5)) if not metrics['incomplete'] else None
    report=dict(candidate=candidate,opponent=opponent,metrics=metrics,elo=point,
        elo_interval=metrics['elo_delta_95pct_open'],rating_scope='Internal checkpoints at identical search budgets; checkpoint0000 is Elo0',
        estimate='Jeffreys-smoothed log-odds point; paired Hoeffding interval',games=games)
    publish(path,identity,lambda stage:write_json(stage/'report.json',report))
    return report


def main(args):
    run=Path(args.run).resolve();run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    config={k:v for k,v in vars(args).items() if k not in ('run','iterations')}
    config['initial_model']=str(Path(args.initial_model).resolve());config['initial_sha256']=digest(Path(args.initial_model))
    identity=dict(run=str(run),backbone=VALUE_SCHEMA,config=config,sources=source_identity(),runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda))
    with (run/'training.lock').open('x') as stream:stream.write(str(os.getpid()))
    def event(stage,iteration,**data):
        record=dict(schema=SCHEMA,stage=stage,iteration=iteration,updated_at=time.time(),**data)
        write_json(run/'status.json',record)
        with (run/'events.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
    try:
        if (run/'config.json').exists():
            if json.loads((run/'config.json').read_text())!=identity:raise ValueError('Run source or configuration changed')
        else:write_json(run/'config.json',identity)
        league=json.loads((run/'league.json').read_text()) if (run/'league.json').exists() else dict(champion=0,checkpoints=[dict(id=0,elo=0.,elo_interval=[0.,0.],promoted=True,reference=True)])
        zero=run/'checkpoints/0000'
        if not zero.exists():
            model=warm_start(args.initial_model,args.device,args.seed,config['initial_sha256']);optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
            def writer(stage):save_model(stage/'model.pt',model);torch.save(optimizer.state_dict(),stage/'optimizer.pt')
            publish(zero,dict(identity,checkpoint=0),writer);del model,optimizer
        verify_artifact(zero,dict(identity,checkpoint=0))
        def corpus_identity(number):
            return dict(identity,iteration=number,
                actor_sha256=digest(run/'checkpoints'/f'{number-1:04d}'/'model.pt'),
                policy_target='Gumbel completed-Q improved policy',
                value_target='Final terminal outcome from position player perspective')
        # Validate completed history before skipping work on resume, including
        # the fixed anchor and every checkpoint referenced by the league.
        for saved in sorted(league['checkpoints'],key=lambda c:c['id']):
            number=saved['id']
            manifest=verify_artifact(run/'checkpoints'/f'{number:04d}',dict(identity,checkpoint=number))
            if number:
                old_corpus=run/'corpus'/f'{number:04d}'
                verify_artifact(old_corpus,corpus_identity(number))
                if manifest['metrics']['corpus_sha256']!=digest(old_corpus/'manifest.json'):
                    raise ValueError('Previously consumed search corpus changed')
        write_json(run/'league.json',league)
        for iteration in range(1,args.iterations+1):
            if any(c['id']==iteration for c in league['checkpoints']):continue
            previous=run/'checkpoints'/f'{iteration-1:04d}';verify_artifact(previous,dict(identity,checkpoint=iteration-1))
            checkpoint=run/'checkpoints'/f'{iteration:04d}';corpus=run/'corpus'/f'{iteration:04d}'
            if not checkpoint.exists():
                if not corpus.exists():
                    event('self-play',iteration,completed=0,total=args.games)
                    episodes,rows=play_games([previous/'model.pt'],[(0,0)]*args.games,[[] for _ in range(args.games)],args,args.seed+iteration*10000,
                        lambda data:event('self-play',iteration,**data),training=True)
                    if sum(e['winner']>=0 for e in episodes)<max(2,(args.games+1)//2):
                        write_json(run/f'insufficient-{iteration:04d}.json',episodes)
                        raise ValueError('Fewer than half of self-play games terminated; no outcome targets fabricated')
                    save_corpus(corpus,corpus_identity(iteration),episodes,rows)
                episodes,rows=read_corpus(corpus,corpus_identity(iteration))
                model=load_model(previous/'model.pt',args.device)
                optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
                optimizer.load_state_dict(torch.load(previous/'optimizer.pt',map_location=args.device,weights_only=True))
                started=time.perf_counter()
                metrics=fit(model,optimizer,episodes,rows,args,iteration,lambda data:event('fitting',iteration,**data))
                metrics.update(games=len(episodes),terminal_games=sum(e['winner']>=0 for e in episodes),positions=len(rows),seconds=time.perf_counter()-started,corpus_sha256=digest(corpus/'manifest.json'))
                if source_identity()!=identity['sources']:raise ValueError('Training source changed')
                def writer(stage):save_model(stage/'model.pt',model);torch.save(optimizer.state_dict(),stage/'optimizer.pt')
                publish(checkpoint,dict(identity,checkpoint=iteration),writer,metrics);del model,optimizer
                if args.device=='cuda':torch.cuda.empty_cache()
            metrics=verify_artifact(checkpoint,dict(identity,checkpoint=iteration))['metrics']
            verify_artifact(corpus,corpus_identity(iteration))
            if metrics['corpus_sha256']!=digest(corpus/'manifest.json'):
                raise ValueError('Pending checkpoint consumed corpus changed')
            comparisons={}
            for opponent in sorted({0,league['champion']}):
                comparisons[opponent]=evaluate(run,iteration,opponent,args,lambda data:event('validation-matches',iteration,opponent=opponent,**data))
            anchor=comparisons[0];versus=comparisons[league['champion']]['metrics']
            promoted=not versus['incomplete'] and versus['wins']>versus['losses'] and versus['opening_pair_p']<.05
            league['checkpoints'].append(dict(id=iteration,elo=anchor['elo'],elo_interval=anchor['elo_interval'],promoted=promoted,
                anchor_score={k:anchor['metrics'][k] for k in ('wins','losses','incomplete')},versus_champion=league['champion'],
                champion_score={k:versus[k] for k in ('wins','losses','incomplete','opening_pair_p')},loss=metrics))
            if promoted:league['champion']=iteration
            write_json(run/'league.json',league);event('checkpoint',iteration,elo=anchor['elo'],promoted=promoted,champion=league['champion'])
        event('finished',args.iterations,champion=league['champion'])
    except BaseException as error:
        event('failed',locals().get('iteration',0),error=repr(error));raise
    finally:(run/'training.lock').unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--initial-model',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    for name,default in [('iterations',3),('games',32),('eval-games',32),('envs',8),('simulations',16),('root-samples',8),
                         ('leaf-batch',16),('batch',16),('epochs',1),('max-plies',128),('max-nodes',10000),('max-edges',600000),('cache-positions',1024),('seed',1740)]:
        parser.add_argument('--'+name,type=int,default=default)
    parser.add_argument('--lr',type=float,default=.0001)
    args=parser.parse_args()
    if min(args.iterations,args.games,args.eval_games,args.envs,args.simulations,args.root_samples,args.leaf_batch,args.batch,args.epochs,args.max_plies,args.max_nodes,args.max_edges,args.cache_positions)<1 or args.games<2 or args.eval_games%2 or not math.isfinite(args.lr) or args.lr<=0:
        parser.error('Positive settings, at least two self-play games, and an even evaluation count required')
    main(args)
