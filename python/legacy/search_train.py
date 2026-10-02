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
from legacy.klent import digest, segmented_log_softmax
from neural_search import NeuralSearch, SearchCoordinator, EvaluationCache
from legacy.relational_model import RelationalNet, NeuralEvaluator
from legacy.relational_train import load_model, save_model, precision, graph, work_batches, VALUE_SCHEMA
from legacy.train import write_json, task_opening, paired_metrics
from legacy.checkpoint_league import evaluation_schedule, promotion_older, rate_league, RATING_METHOD, PROMOTION_RULE

SCHEMA = 'hexo-search-selfplay-v1'
PROMOTION_PROTOCOL = dict(rule=PROMOTION_RULE, older_target_fraction=.8,
    requires_full_matches=True, threshold='conservative wins > losses for incumbent and older')


def source_identity():
    names = ('search_train.py','checkpoint_league.py','relational_model.py','relational_train.py','relational_native.py',
             'relational_encoder.py','neural_search.py','hexo.py','klent.py','train.py','curriculum.py',
             'src/gumbel.cpp','src/hexo.cpp','src/hexo.hpp','src/relational_graph.cpp')
    files = {name:digest(ROOT/('python/'+('legacy/'+name if name not in ('hexo.py','neural_search.py') else name) if name.endswith('.py') else name)) for name in names}
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


def evaluation_protocol(args):
    return dict(max_plies=args.eval_max_plies,tactics=args.eval_tactics,
                simulations=args.simulations,root_samples=args.root_samples,device=args.device,
                opening_suite='standard-v1',source_sha256=identity_fingerprint(source_identity()),
                runtime_sha256=identity_fingerprint(runtime_identity()))


def identity_fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def runtime_identity():
    return dict(torch=str(torch.__version__),cuda=torch.version.cuda)


def pessimistic_promotion(report):
    """Score every capped game as a challenger loss, retaining its opening pair."""
    records=report['games'];planned=report['metrics']['planned_games']
    if len(records)!=planned:return False,None
    worst=[dict(game,winner=game['winner'] if game['winner']>=0 else 1-game['challenger_color']) for game in records]
    metrics=paired_metrics(worst,planned)
    return metrics['wins']>metrics['losses'],metrics


def outcome_rows(rows,winner):
    """Attach a known winner, or keep policy-only rows from a capped game."""
    for row in rows:
        row['target']=None if winner<0 else (1. if row['player']==winner else -1.)
    return rows


def capture_history(run, previous_identity, config, target_sources, target_runtime, prior_history=None):
    """Bind an existing stopped run's immutable artifacts before changing its evaluator."""
    status=json.loads((run/'status.json').read_text())
    recovering=status['stage']=='failed' and status.get('iteration')==0
    if status['stage']!='finished' and not recovering:
        raise ValueError('Upgrade requires a finished checkpoint boundary')
    mutable={'reference_games','games','replay_positions','reuse_ratio','evaluate_every','actor_tactics',
             'eval_max_plies','eval_tactics','promotion_rule'}
    if {k:v for k,v in config.items() if k not in mutable}!={k:v for k,v in previous_identity['config'].items() if k not in mutable}:
        raise ValueError('Upgrade may change collection and replay scheduling, not model or evaluation settings')
    league=json.loads((run/'league.json').read_text())
    numbers=sorted(c['id'] for c in league['checkpoints'])
    if numbers!=list(range(len(numbers))):raise ValueError('Incomplete checkpoint history')
    if recovering:
        events=run/'events.jsonl'
        if not events.exists() or not any((event:=json.loads(line)).get('stage')=='finished' and event.get('iteration')==numbers[-1]
                                            for line in events.read_text().splitlines()):
            raise ValueError('No prior finished event proves the upgrade boundary')
    prior_history=prior_history or {}
    if prior_history and prior_history.get('previous_identity') is None:
        raise ValueError('Prior upgrade history has no previous identity')
    artifacts={}
    def remember(path,expected=None):
        relative=path.relative_to(run).as_posix()
        inherited=prior_history.get('artifacts',{}).get(relative)
        manifest=verify_artifact(path,None if inherited else expected)
        sha=digest(path/'manifest.json')
        if inherited and sha!=inherited:raise ValueError(f'Prior inherited artifact changed: {path}')
        artifacts[relative]=sha
        return manifest
    for number in numbers:
        path=run/'checkpoints'/f'{number:04d}'
        raw=json.loads((path/'manifest.json').read_text())
        expected=dict(previous_identity,checkpoint=number) if 'checkpoint' in raw.get('identity',{}) else previous_identity
        manifest=remember(path,expected)
        if number:
            corpus=run/'corpus'/f'{number:04d}'
            raw=json.loads((corpus/'manifest.json').read_text())
            target=raw.get('identity',{}).get('value_target')
            if target not in ('Final terminal outcome from position player perspective',
                              'Final terminal outcome when known; masked for capped games'):
                raise ValueError('Unknown historical value target')
            expected=dict(previous_identity,actor_sha256=digest(run/'checkpoints'/f'{number-1:04d}'/'model.pt'),
                policy_target='Gumbel completed-Q improved policy',value_target=target)
            if 'iteration' in raw.get('identity',{}):expected['iteration']=number
            remember(corpus,expected)
            if manifest['metrics']['corpus_sha256']!=digest(corpus/'manifest.json'):
                raise ValueError('Historical checkpoint consumed a different corpus')
    for path in sorted((run/'evaluation').glob('*-vs-*')):
        if not path.is_dir():continue
        manifest=remember(path);report=json.loads((path/'report.json').read_text())
        a,b=report['candidate'],report['opponent']
        if a not in numbers or b not in numbers or path.name!=f'{a:04d}-vs-{b:04d}':
            raise ValueError('Historical comparison identity changed')
        evidence=manifest['identity'];relative=path.relative_to(run).as_posix()
        if relative not in prior_history.get('artifacts',{}):
            if report['metrics']['planned_games'] not in (previous_identity['config']['eval_games'],
                                                           previous_identity['config']['reference_games']):
                raise ValueError('Historical comparison game count changed')
            expected=dict(run=previous_identity,candidate=digest(run/'checkpoints'/f'{a:04d}'/'model.pt'),
                opponent=digest(run/'checkpoints'/f'{b:04d}'/'model.pt'),simulations=config['simulations'],
                root_samples=config['root_samples'],games=report['metrics']['planned_games'],
                seed=config['seed']+100000+a*1000+b*1000003,opening_suite='standard-v1')
            if 'run' not in evidence:
                expected.pop('run');expected['seed']=config['seed']+100000+a*1000
            if 'protocol' in evidence:
                expected['protocol']=dict(max_plies=previous_identity['config']['eval_max_plies'],
                    tactics=previous_identity['config']['eval_tactics'],simulations=config['simulations'],
                    root_samples=config['root_samples'],device=previous_identity['config']['device'],
                    opening_suite='standard-v1',source_sha256=identity_fingerprint(previous_identity['sources']),
                    runtime_sha256=identity_fingerprint(previous_identity['runtime']))
            if evidence!=expected:raise ValueError('Historical comparison settings changed')
    history=dict(previous_identity=previous_identity,previous_config_sha256=digest(run/'config.json'),
                 prepared_target=dict(config=config,sources=target_sources,runtime=target_runtime),
                 through=numbers[-1],artifacts=artifacts)
    paired=run/'paired-ratings.json'
    if paired.exists():
        joint=json.loads(paired.read_text())
        if joint.get('config_sha256')==history['previous_config_sha256']:
            old_ratings={}
            for record in joint.get('checkpoints',[]):
                number=record['id']
                if number in numbers and record.get('model_sha256')==digest(run/'checkpoints'/f'{number:04d}'/'model.pt'):
                    old_ratings[str(number)]={k:record.get(k) for k in ('elo','elo_interval','rating_pairs')}
            history['prior_joint_ratings']=old_ratings
            history['prior_joint_ratings_sha256']=digest(paired)
    return history


def update_ratings(run, league, args, history):
    reports=[];ids={c['id'] for c in league['checkpoints']}
    hashes={number:digest(run/'checkpoints'/f'{number:04d}'/'model.pt') for number in ids}
    for path in sorted((run/'evaluation').glob('*-vs-*')):
        if not path.is_dir():continue
        manifest=verify_artifact(path);report=json.loads((path/'report.json').read_text())
        a,b=report['candidate'],report['opponent']
        if a not in ids or b not in ids:continue
        inherited=history.get('artifacts',{}).get(path.relative_to(run).as_posix())
        if inherited and inherited!=digest(path/'manifest.json'):raise ValueError('Historical evaluation changed')
        if manifest['identity'].get('protocol')!=evaluation_protocol(args):continue
        if path.name!=f'{a:04d}-vs-{b:04d}':raise ValueError('Comparison filename disagrees with opponents')
        for key,number in [('candidate',a),('opponent',b)]:
            if manifest['identity'][key]!=hashes[number]:
                raise ValueError('Rating comparison model changed')
        for key in ('simulations','root_samples'):
            if manifest['identity'][key]!=getattr(args,key):raise ValueError('Rating search budgets differ')
        reports.append(report)
    protocol=evaluation_protocol(args)
    if league.get('rating_protocol')!=protocol and history:
        previous=league.get('rating_protocol')
        if previous is None:
            old=history['previous_identity']['config']
            previous=dict(max_plies=old.get('eval_max_plies',old['max_plies']),
                          tactics=old.get('eval_tactics',False),simulations=old['simulations'],
                          root_samples=old['root_samples'],device=old['device'],opening_suite='standard-v1',
                          source_sha256=identity_fingerprint(history['previous_identity']['sources']),
                          runtime_sha256=identity_fingerprint(history['previous_identity']['runtime']))
        for checkpoint in league['checkpoints']:
            older=history.get('prior_joint_ratings',{}).get(str(checkpoint['id']))
            elo=older['elo'] if older and older.get('elo') is not None else checkpoint.get('elo')
            bounds=older.get('elo_interval') if older and older.get('elo') is not None else checkpoint.get('elo_interval')
            if checkpoint['id'] and elo is not None:
                checkpoint.setdefault('historical_ratings',[]).append(dict(protocol=previous,
                    elo=elo,elo_interval=bounds,source='paired joint rating' if older else 'league rating'))
    ratings,intervals=rate_league(ids,reports,seed=args.seed)
    for checkpoint in league['checkpoints']:
        if 'reference_elo' not in checkpoint:
            checkpoint['reference_elo']=checkpoint.get('elo')
            checkpoint['reference_elo_interval']=checkpoint.get('elo_interval')
        checkpoint['elo']=ratings[checkpoint['id']]
        checkpoint['elo_interval']=intervals.get(checkpoint['id'])
    league['rating_method']=RATING_METHOD
    league['rating_protocol']=protocol
    league['rating_interval_note']='Approximate 95% Bayesian credible intervals from 2048 paired-outcome posterior draws. Each comparison uses a Jeffreys Dirichlet prior over 0, 1 or 2 wins per opening pair. Reference-only intervals are retained separately.'


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
                                    coordinators[j].cache,tactics=args.actor_tactics if training else args.eval_tactics) for j,ev in enumerate(evaluators)]
                live.append(dict(id=next_id,game=Game(history),history=list(history),trees=trees,rows=[]))
                next_id+=1
            # Freeze this sweep's side assignments before applying any move.
            groups=[[e for e in live if assignments[e['id']][e['game'].player]==j] for j in range(len(evaluators))]
            for j,group in enumerate(groups):
                if not group:continue
                results=coordinators[j].search_many([e['trees'][j] for e in group],args.simulations,args.root_samples,args.leaf_batch,choice='gumbel')
                for e,result in zip(group,results,strict=True):
                    g=e['game'];actions=result['actions'];policy=result['policy']
                    if result['completed']!=args.simulations or result['action'] is None:
                        raise ValueError('Incomplete search cannot produce a training target')
                    if len(policy)!=len(actions) or not np.isfinite(policy).all() or np.any(policy<0) or not np.isclose(policy.sum(),1):
                        raise ValueError('Invalid search policy target')
                    if training:
                        e['rows'].append(dict(game=e['id'],ply=len(e['history']),player=g.player,remaining=g.remaining,
                            action=result['action'],legal_sha256=hashlib.sha256(actions.astype(np.int64).tobytes()).hexdigest(),
                            policy=policy.astype(np.float32),simulations=result['completed'],evaluated=result['evaluated'],tactics=args.actor_tactics,exact_winner=result['exact_winner']))
                    action=result['action'];g.play(*action);e['history'].append(action)
                    for tree in e['trees']:tree.advance(action)
                    completed_placements+=1
            cap=args.max_plies if training else args.eval_max_plies
            finished=[e for e in live if e['game'].winner>=0 or len(e['history'])>=cap]
            for e in finished:
                winner=e['game'].winner
                episode=dict(id=e['id'],moves=e['history'],opening=openings[e['id']],winner=winner,
                    reason='six-in-a-row' if winner>=0 else 'cap',actors=assignments[e['id']])
                episodes.append(episode)
                # Keep the searched policy from capped games, but never invent a game result.
                rows.extend(outcome_rows(e['rows'],winner))
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


def recent_replay(run,iteration,capacity,verify):
    episodes=[];rows=[];sources=[]
    for number in range(iteration,0,-1):
        path=run/'corpus'/f'{number:04d}'
        manifest=verify(path)
        games,items=read_corpus(path,manifest['identity'])
        # Keep newest rows when the oldest admitted corpus crosses the bound.
        items=items[-(capacity-len(rows)):]
        used={r['game'] for r in items}
        episodes.extend(dict(e,id=(number,e['id'])) for e in games if e['id'] in used)
        rows.extend(dict(r,game=(number,r['game']),corpus=number,
                         actor_sha256=manifest['identity']['actor_sha256'],target_age=iteration-number) for r in items)
        sources.append(dict(iteration=number,manifest_sha256=digest(path/'manifest.json'),positions=len(items),
                            actor_sha256=manifest['identity']['actor_sha256']))
        if len(rows)>=capacity:break
    return episodes,rows,sources


def fit(model,optimizer,episodes,rows,args,iteration,progress,presentations=None):
    episodes={e['id']:e for e in episodes}
    families=sorted({r['game'] for r in rows})
    if not families or (presentations is None and len(families)<2):raise ValueError('No training rows, or too few games for disjoint validation')
    rng=np.random.default_rng(args.seed+iteration)
    validation=set() if presentations is not None else set(rng.permutation(families)[:max(1,len(families)//4)].tolist())
    train=[r for r in rows if r['game'] not in validation];valid=[r for r in rows if r['game'] in validation]
    steps=0;last={'validation':None};age_counts={}
    if presentations is not None:
        order=[]
        while len(order)<presentations:order.extend(rng.permutation(len(train))[:presentations-len(order)].tolist())
        phases=[('train',[train[i] for i in order])]
    else:phases=None
    for epoch in range(1 if presentations is not None else args.epochs):
        for phase,selected in phases or [('train',[train[i] for i in rng.permutation(len(train))]),('validation',valid)]:
            model.train(phase=='train');total=np.zeros(3);completed=0
            for start in range(0,len(selected),args.batch):
                batch_rows=selected[start:start+args.batch]
                known_in_batch=sum(r['target'] is not None for r in batch_rows)
                if phase=='train':
                    for row in batch_rows:
                        age=row.get('target_age',0);age_counts[age]=age_counts.get(age,0)+1
                graphs=[graph(episodes[r['game']]['moves'][:r['ply']],model,args) for r in batch_rows]
                for row,item in zip(batch_rows,graphs,strict=True):
                    if hashlib.sha256(item.actions.astype(np.int64).tobytes()).hexdigest()!=row['legal_sha256'] or (item.player,item.remaining)!=(row['player'],row['remaining']):
                        raise ValueError('Training position or full legal action order changed')
                if phase=='train':optimizer.zero_grad(set_to_none=True)
                offset=0
                for group in work_batches(graphs,args):
                    from legacy.relational_encoder import pack
                    chosen=batch_rows[offset:offset+len(group)];batch=pack(group,args.device)
                    with torch.set_grad_enabled(phase=='train'),precision(args.device):
                        output=model(batch)
                        logpi=segmented_log_softmax(output['logits'],batch['action_owner'],len(group))
                        policy=torch.as_tensor(np.concatenate([r['policy'] for r in chosen]),device=args.device)
                        known=torch.tensor([r['target'] is not None for r in chosen],device=args.device)
                        outcome=torch.tensor([r['target'] if r['target'] is not None else 0. for r in chosen],device=args.device)
                        ce=-(policy*logpi).sum()/len(group)
                        value_sq=((output['value']-outcome).square()*known).sum()
                        loss=ce*len(group)/len(batch_rows)+value_sq/max(1,known_in_batch)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite search-learning loss')
                    if phase=='train':loss.backward()
                    total+=np.array([ce.item()*len(group),value_sq.item(),known.sum().item()]);offset+=len(group)
                if phase=='train':
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();steps+=1
                completed+=len(batch_rows)
                progress(dict(epoch=epoch+1,epochs=args.epochs,fit_phase=phase,completed=completed,total=len(selected),
                    policy_ce=total[0]/completed,value_mse=total[1]/max(1,total[2]),optimizer_steps=steps))
            last[phase]=dict(policy_ce=total[0]/completed,value_mse=total[1]/max(1,total[2]),
                             positions=completed,terminal_value_positions=int(total[2]))
    return dict(**last,optimizer_steps=steps,validation_game_ids=sorted(validation),training_games=len(families)-len(validation),
        examples_presented=sum(age_counts.values()),target_age_presentations=age_counts)


def evaluate(run,candidate,opponent,args,progress,games=None,verify_checkpoint=None):
    games=args.eval_games if games is None else games
    path=run/'evaluation'/f'{candidate:04d}-vs-{opponent:04d}'
    a=run/'checkpoints'/f'{candidate:04d}'/'model.pt';b=run/'checkpoints'/f'{opponent:04d}'/'model.pt'
    run_identity=json.loads((run/'config.json').read_text())
    verify_checkpoint=verify_artifact if verify_checkpoint is None else verify_checkpoint
    verify_checkpoint(a.parent,dict(run_identity,checkpoint=candidate))
    verify_checkpoint(b.parent,dict(run_identity,checkpoint=opponent))
    identity=dict(run=run_identity,candidate=digest(a),opponent=digest(b),simulations=args.simulations,root_samples=args.root_samples,
                  games=games,seed=args.seed+100000+candidate*1000+opponent*1000003,opening_suite='standard-v1',
                  protocol=evaluation_protocol(args))
    if path.exists():
        verify_artifact(path,identity)
        return json.loads((path/'report.json').read_text())
    openings=[task_opening(identity['seed']+i//2,True,args.eval_max_plies,'standard-v1')['opening'] for i in range(games)]
    assignments=[(0,1) if i%2==0 else (1,0) for i in range(games)]
    episodes,_=play_games([a,b],assignments,openings,args,identity['seed'],progress)
    records=[dict(index=e['id'],pair=e['id']//2,seed=identity['seed']+e['id']//2,challenger_color=e['id']%2,winner=e['winner'],reason=e['reason'],
                opening=e['opening'],moves=e['moves']) for e in episodes]
    metrics=paired_metrics(records,games)
    point=400*math.log10((metrics['wins']+.5)/(metrics['losses']+.5)) if not metrics['incomplete'] else None
    report=dict(candidate=candidate,opponent=opponent,metrics=metrics,elo=point,protocol=evaluation_protocol(args),
        elo_interval=metrics['elo_delta_95pct_open'],rating_scope='Internal checkpoints at identical search budgets; checkpoint0000 is Elo0',
        estimate='Jeffreys-smoothed log-odds point; paired Hoeffding interval',games=records)
    publish(path,identity,lambda stage:write_json(stage/'report.json',report))
    return report


def main(args):
    run=Path(args.run).resolve();run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    config={k:v for k,v in vars(args).items() if k not in ('run','iterations','upgrade_run')}
    config['initial_model']=str(Path(args.initial_model).resolve());config['initial_sha256']=digest(Path(args.initial_model))
    config['promotion_rule']=PROMOTION_RULE
    identity=dict(run=str(run),backbone=VALUE_SCHEMA,config=config,sources=source_identity(),runtime=runtime_identity())
    with (run/'training.lock').open('x') as stream:stream.write(str(os.getpid()))
    def event(stage,iteration,**data):
        record=dict(schema=SCHEMA,stage=stage,iteration=iteration,updated_at=time.time(),**data)
        write_json(run/'status.json',record)
        with (run/'events.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
    try:
        history_path=run/'history.json'
        history=json.loads(history_path.read_text()) if history_path.exists() else {}
        if history.get('prior_history_file'):
            name=history['prior_history_file']
            if Path(name).name!=name or digest(run/name)!=history['prior_history_sha256']:
                raise ValueError('Prior upgrade history snapshot changed')
            if history['previous_identity'].get('history_sha256')!=history['prior_history_sha256']:
                raise ValueError('Prior upgrade history chain changed')
        config_path=run/'config.json'
        existing=json.loads(config_path.read_text()) if config_path.exists() else None
        if args.upgrade_run and existing is not None:
            if existing.get('history_sha256') and history_path.exists() and digest(history_path)!=existing['history_sha256']:
                # A prepared upgrade may have written its history before its config.
                if history.get('previous_identity')!=existing:
                    raise ValueError('Prior run history changed before upgrade')
            if history.get('previous_identity')==existing and history.get('through')==len(json.loads((run/'league.json').read_text())['checkpoints'])-1:
                if history.get('prepared_target')!=dict(config=config,sources=identity['sources'],runtime=identity['runtime']):
                    raise ValueError('Prepared upgrade targets a different source or configuration')
            elif existing['sources']!=identity['sources'] or existing['config']!=config or existing['runtime']!=identity['runtime']:
                prior=history
                if existing.get('history_sha256'):
                    if not history_path.exists() or digest(history_path)!=existing['history_sha256']:
                        raise ValueError('Prior upgrade history changed')
                for lock in ('background-evaluation.lock','paired-ratings.lock'):
                    if (run/lock).exists():raise ValueError(f'Upgrade requires stopped worker: {lock}')
                prepared=capture_history(run,existing,config,identity['sources'],identity['runtime'],prior)
                if prior:
                    sha=digest(history_path);name=f'history-{sha}.json';snapshot=run/name
                    if snapshot.exists():
                        if digest(snapshot)!=sha:raise ValueError('Prior history snapshot changed')
                    else:
                        pending=snapshot.with_suffix('.json.tmp')
                        pending.write_bytes(history_path.read_bytes())
                        if digest(pending)!=sha:raise ValueError('Prior history snapshot copy changed')
                        os.replace(pending,snapshot)
                    prepared.update(prior_history_file=name,prior_history_sha256=sha)
                history=prepared;write_json(history_path,history)
        if history:identity['history_sha256']=digest(history_path)
        if existing is not None:
            if args.upgrade_run and history and existing==history['previous_identity']:
                write_json(config_path,identity)
            elif existing!=identity:raise ValueError('Run source or configuration changed')
        else:write_json(config_path,identity)
        def verify_run_artifact(path,expected):
            inherited=history.get('artifacts',{}).get(path.relative_to(run).as_posix())
            if inherited:
                if digest(path/'manifest.json')!=inherited:raise ValueError('Inherited artifact manifest changed')
                return verify_artifact(path)
            return verify_artifact(path,expected)
        league=json.loads((run/'league.json').read_text()) if (run/'league.json').exists() else dict(champion=0,checkpoints=[dict(id=0,elo=0.,elo_interval=[0.,0.],promoted=True,reference=True)])
        league['promotion_protocol']=PROMOTION_PROTOCOL
        zero=run/'checkpoints/0000'
        if not zero.exists():
            model=warm_start(args.initial_model,args.device,args.seed,config['initial_sha256']);optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
            def writer(stage):save_model(stage/'model.pt',model);torch.save(optimizer.state_dict(),stage/'optimizer.pt')
            publish(zero,dict(identity,checkpoint=0),writer);del model,optimizer
        verify_run_artifact(zero,dict(identity,checkpoint=0))
        def corpus_identity(number):
            return dict(identity,iteration=number,
                actor_sha256=digest(run/'checkpoints'/f'{number-1:04d}'/'model.pt'),
                policy_target='Gumbel completed-Q improved policy',
                value_target='Final terminal outcome when known; masked for capped games')
        # Validate completed history before skipping work on resume, including
        # the fixed anchor and every checkpoint referenced by the league.
        for saved in sorted(league['checkpoints'],key=lambda c:c['id']):
            number=saved['id']
            manifest=verify_run_artifact(run/'checkpoints'/f'{number:04d}',dict(identity,checkpoint=number))
            if number:
                old_corpus=run/'corpus'/f'{number:04d}'
                verify_run_artifact(old_corpus,corpus_identity(number))
                if manifest['metrics']['corpus_sha256']!=digest(old_corpus/'manifest.json'):
                    raise ValueError('Previously consumed search corpus changed')
        update_ratings(run,league,args,history)
        write_json(run/'league.json',league)
        for iteration in range(1,args.iterations+1):
            if any(c['id']==iteration for c in league['checkpoints']):continue
            previous=run/'checkpoints'/f'{iteration-1:04d}';verify_run_artifact(previous,dict(identity,checkpoint=iteration-1))
            checkpoint=run/'checkpoints'/f'{iteration:04d}';corpus=run/'corpus'/f'{iteration:04d}'
            if not checkpoint.exists():
                if not corpus.exists():
                    event('self-play',iteration,completed=0,total=args.games)
                    episodes,rows=play_games([previous/'model.pt'],[(0,0)]*args.games,[[] for _ in range(args.games)],args,args.seed+iteration*10000,
                        lambda data:event('self-play',iteration,**data),training=True)
                    if sum(e['winner']>=0 for e in episodes)<2:
                        write_json(run/f'insufficient-{iteration:04d}.json',episodes)
                        raise ValueError('Fewer than two self-play games terminated; no reliable outcome targets')
                    save_corpus(corpus,corpus_identity(iteration),episodes,rows)
                episodes,rows=read_corpus(corpus,corpus_identity(iteration))
                model=load_model(previous/'model.pt',args.device)
                optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
                optimizer.load_state_dict(torch.load(previous/'optimizer.pt',map_location=args.device,weights_only=True))
                started=time.perf_counter()
                fresh_positions=len(rows);fresh_games=len(episodes);terminals=sum(e['winner']>=0 for e in episodes)
                episodes,rows,replay_sources=recent_replay(run,iteration,args.replay_positions,
                    lambda path:verify_run_artifact(path,corpus_identity(int(path.name))))
                presentations=args.reuse_ratio*min(fresh_positions,args.replay_positions)
                metrics=fit(model,optimizer,episodes,rows,args,iteration,lambda data:event('fitting',iteration,**data),presentations)
                metrics.update(games=fresh_games,terminal_games=terminals,positions=fresh_positions,
                    replay_positions=len(rows),replay_sources=replay_sources,reuse_ratio=args.reuse_ratio,
                    seconds=time.perf_counter()-started,corpus_sha256=digest(corpus/'manifest.json'))
                if source_identity()!=identity['sources']:raise ValueError('Training source changed')
                def writer(stage):save_model(stage/'model.pt',model);torch.save(optimizer.state_dict(),stage/'optimizer.pt')
                publish(checkpoint,dict(identity,checkpoint=iteration),writer,metrics);del model,optimizer
                if args.device=='cuda':torch.cuda.empty_cache()
            metrics=verify_artifact(checkpoint,dict(identity,checkpoint=iteration))['metrics']
            verify_artifact(corpus,corpus_identity(iteration))
            if metrics['corpus_sha256']!=digest(corpus/'manifest.json'):
                raise ValueError('Pending checkpoint consumed corpus changed')
            entry=dict(id=iteration,elo=None,elo_interval=None,promoted=False,loss=metrics,evaluation_due=iteration%args.evaluate_every==0)
            if entry['evaluation_due']:
                comparisons={}
                for opponent,games in evaluation_schedule(iteration,league['champion'],args.eval_games,args.reference_games):
                    comparisons[opponent]=evaluate(run,iteration,opponent,args,lambda data:event('validation-matches',iteration,opponent=opponent,**data),games,verify_run_artifact)
                anchor=comparisons[0];versus=comparisons[league['champion']]['metrics']
                champion_won,worst=pessimistic_promotion(comparisons[league['champion']])
                older=promotion_older(iteration,league['champion'])
                older_won,older_worst=pessimistic_promotion(comparisons[older]) if older is not None else (False,None)
                promoted=champion_won and older_won
                entry.update(elo=anchor['elo'],elo_interval=anchor['elo_interval'],promoted=promoted,
                    promotion_rule=PROMOTION_RULE,promotion_older=older,
                    anchor_score={k:anchor['metrics'][k] for k in ('wins','losses','incomplete')},versus_champion=league['champion'],
                    champion_score=dict({k:versus[k] for k in ('wins','losses','incomplete','opening_pair_p')},
                                        pessimistic_wins=worst['wins'] if worst else None,
                                        pessimistic_losses=worst['losses'] if worst else None,
                                        pessimistic_opening_pair_p=worst['opening_pair_p'] if worst else None))
                entry['previous_score']={k:comparisons[iteration-1]['metrics'][k] for k in ('wins','losses','incomplete','opening_pair_p')}
                if older is not None:
                    entry['older_score']=dict(opponent=older,
                        **{k:comparisons[older]['metrics'][k] for k in ('wins','losses','incomplete','opening_pair_p')},
                        pessimistic_wins=older_worst['wins'] if older_worst else None,
                        pessimistic_losses=older_worst['losses'] if older_worst else None,
                        pessimistic_opening_pair_p=older_worst['opening_pair_p'] if older_worst else None)
                if promoted:league['champion']=iteration
            league['checkpoints'].append(entry)
            update_ratings(run,league,args,history)
            write_json(run/'league.json',league);event('checkpoint',iteration,elo=entry['elo'],promoted=entry['promoted'],champion=league['champion'])
        event('finished',args.iterations,champion=league['champion'])
    except BaseException as error:
        event('failed',locals().get('iteration',0),error=repr(error));raise
    finally:(run/'training.lock').unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--initial-model',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--actor-tactics',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--eval-tactics',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--eval-max-plies',type=int,default=128)
    parser.add_argument('--upgrade-run',action='store_true',help='Preserve and bind a finished run before changing its evaluation schedule')
    for name,default in [('replay-positions',200000),('reuse-ratio',4),('evaluate-every',4),('iterations',100),('games',128),('eval-games',32),('reference-games',8),('envs',8),('simulations',16),('root-samples',8),
                         ('leaf-batch',16),('batch',16),('epochs',1),('max-plies',128),('max-nodes',10000),('max-edges',600000),('cache-positions',1024),('seed',1740)]:
        parser.add_argument('--'+name,type=int,default=default)
    parser.add_argument('--lr',type=float,default=.0001)
    args=parser.parse_args()
    if min(args.replay_positions,args.reuse_ratio,args.evaluate_every,args.iterations,args.games,args.eval_games,args.reference_games,args.envs,args.simulations,args.root_samples,args.leaf_batch,args.batch,args.epochs,args.max_plies,args.max_nodes,args.max_edges,args.cache_positions)<1 or args.eval_max_plies<5 or args.games<2 or args.eval_games%2 or args.reference_games%2 or args.reference_games>args.eval_games or not math.isfinite(args.lr) or args.lr<=0:
        parser.error('Positive settings, evaluation cap at least five, at least two self-play games, and an even evaluation count required')
    main(args)
