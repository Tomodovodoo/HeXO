"""One new human-data pass with explicit near-terminal coverage and no replay."""
import argparse
from collections import Counter
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from legacy.klent import digest, publish, verify
from legacy.relational_data import human_examples
from legacy.relational_train import load_model, save_model, source_identity
from legacy.relational_warmstart import epoch
from legacy.train import write_json


def select(histories,rows,previous,positions,cap,seed):
    if positions<2 or positions%2 or cap<1:
        raise ValueError('Positive even position count and positive per-game cap required')
    excluded={(r['game'],r['ply']) for r in previous}
    buckets=[[],[]]
    for row in rows:
        if (row['game'],row['ply']) not in excluded:
            buckets[int(len(histories[row['game']])-row['ply']>8)].append(row)
    rng=np.random.default_rng(seed)
    counts=Counter();chosen=[]
    earlier_counts=Counter(r['game'] for r in buckets[1])
    earlier_capacity=sum(min(count,cap) for count in earlier_counts.values())
    for stratum,bucket in enumerate(buckets):
        selected=[]
        for index in rng.permutation(len(bucket)):
            row=bucket[int(index)]
            if counts[row['game']]>=cap:continue
            if stratum==0:
                remaining=cap-counts[row['game']]
                consumed=min(earlier_counts[row['game']],remaining)-min(earlier_counts[row['game']],remaining-1)
                if earlier_capacity-consumed<positions//2:continue
                earlier_capacity-=consumed
            selected.append(row);counts[row['game']]+=1
            if len(selected)==positions//2:break
        if len(selected)!=positions//2:raise ValueError('Insufficient new positions for declared stratum and game cap')
        chosen.extend(selected)
    return chosen


def fit(args):
    if (args.batch<1 or args.max_nodes<1 or args.max_edges<1
            or not all(math.isfinite(v) and v>0 for v in (args.lr,args.grad_clip))):
        raise ValueError('Positive fitting and graph budgets required')
    output=Path(args.output)
    if output.exists():raise ValueError('New stratified output directory required')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    source=source_identity(('relational_stratified.py','relational_warmstart.py','relational_data.py','human_corpus.py','corpus_warmstart.py'))
    import json
    previous=Path(args.previous_run)
    prior=json.loads((previous/'manifest.json').read_bytes())
    verify(previous,prior['identity'])
    if prior['identity'].get('kind')!='relational-human-policy-q-v1':
        raise ValueError('Expected the initial human warmstart artifact, not another continuation')
    for name in ('relational_data.py','human_corpus.py','corpus_warmstart.py'):
        if prior['identity']['sources'].get(name)!=source['sources'][name]:
            raise ValueError(f'Previous human sampler changed: {name}')
    old_config=prior['identity']['config']
    histories,all_rows,data=human_examples(args.corpus)
    _,old_rows,_=human_examples(args.corpus,old_config['positions'],old_config['seed'])
    for field in ('manifest_sha256','histories_sha256','shards'):
        if data[field]!=prior['identity']['data'][field]:raise ValueError('Previous human corpus identity changed')
    rows={'train':select(histories,all_rows['train'],old_rows['train'],args.positions,args.per_game,args.seed),
          'validation':old_rows['validation']}
    if {r['family'] for r in rows['train']}&{r['family'] for r in rows['validation']}:
        raise ValueError('Stratified family leakage')
    checkpoint=previous/'model.pt'
    model_sha=prior['files']['model.pt']
    model=load_model(checkpoint,args.device,expected_sha256=model_sha)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    plan=dict(kind='relational-human-stratified-v1',initial_model_sha256=model_sha,
        training_positions=len(rows['train']),validation_positions=len(rows['validation']),
        near_terminal=sum(len(histories[r['game']])-r['ply']<=8 for r in rows['train']),
        previous_training_positions_excluded=len(old_rows['train']),maximum_per_game=max(Counter(r['game'] for r in rows['train']).values()),
        distance_to_terminal=dict(sorted(Counter(str(len(histories[r['game']])-r['ply']) for r in rows['train']).items())),
        optimizer='Fresh Adam; all backbone, policy and critic parameters; one shuffled pass',
        target='Human chosen-action imitation plus observed human-continuation terminal Q; unknown actions unlabeled',
        acceptance={'fixed_validation':'Policy CE <= baseline; action accuracy >= baseline; Q MSE <= 1.05 * baseline',
                    'complete_turns':'At least 4/8 existing held-out winning turns completed by actual sequential mu, baseline 2/8; no regression on independently feasible full-turn defenses',
                    'fixed_tactics':'Policy acceptable choices >= 29/38 and mu >= 19/38, with immediate policy wins >= 5/6',
                    'confirmation':'After choosing the single candidate, evaluate 32 fresh validation families, excluded from training and previous development fixtures',
                    'next_stage':'Only after health checks, 32 fresh KLENT games with cap128; fit once only if terminal fraction >=0.75; report caps and exact critic initialization'},
        promotion=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output.with_suffix('.plan.json'),plan)
    write_json(output.with_suffix('.rows.json'),rows)
    identity=dict(kind='relational-human-stratified-v1',config=vars(args),data=data,
        initial_model_sha256=model_sha,previous_manifest_sha256=digest(previous/'manifest.json'),
        selected_rows_sha256=digest(output.with_suffix('.rows.json')),**source)
    started=time.perf_counter()
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    def progress(stage,completed,total):
        write_json(output.with_suffix('.status.json'),dict(stage=stage,epoch=1,epochs=1,completed=completed,total=total,seconds=time.perf_counter()-started))
    baseline=epoch(model,histories,rows['validation'],args,
                   progress=lambda completed,total:progress('initial_validation',completed,total))
    train=epoch(model,histories,rows['train'],args,optimizer,args.seed,
                progress=lambda completed,total:progress('training',completed,total))
    validation=epoch(model,histories,rows['validation'],args,
                     progress=lambda completed,total:progress('validation',completed,total))
    if source!=source_identity(('relational_stratified.py','relational_warmstart.py','relational_data.py','human_corpus.py','corpus_warmstart.py')):
        raise ValueError('Stratified source or native library changed')
    if digest(Path(args.corpus)/'manifest.json')!=data['manifest_sha256'] or digest(Path(args.corpus)/'games.jsonl')!=data['histories_sha256']:
        raise ValueError('Human corpus changed during fitting')
    for shards in data['shards'].values():
        for name,expected in shards.items():
            if digest(Path(name))!=expected:raise ValueError(f'Human shard changed during fitting: {name}')
    report=dict(initial_validation=baseline,epochs=[dict(epoch=1,train=train,validation=validation)],
        selected={'epoch':1},train_positions=len(rows['train']),validation_positions=len(rows['validation']),
        seconds=time.perf_counter()-started,gpu_memory_peak_mb=torch.cuda.max_memory_allocated()/2**20 if args.device=='cuda' else 0,
        validation_criteria_pass=bool(validation['policy_ce']<=baseline['policy_ce'] and validation['policy_accuracy']>=baseline['policy_accuracy'] and validation['q_mse']<=1.05*baseline['q_mse']),
        promotion=False)
    def writer(stage):
        save_model(stage/'model.pt',model);torch.save(optimizer.state_dict(),stage/'optimizer.pt')
        shutil.copy2(checkpoint,stage/'initial_model.pt')
        if digest(stage/'initial_model.pt')!=model_sha:raise ValueError('Initial checkpoint changed during fitting')
        write_json(stage/'report.json',report);write_json(stage/'rows.json',rows);write_json(stage/'plan.json',plan)
    publish(output,identity,writer,report)
    write_json(output.with_suffix('.status.json'),dict(stage='finished',**report))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','previous-run','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--positions',type=int,default=4096)
    parser.add_argument('--per-game',type=int,default=32)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--max-nodes',type=int,default=12000)
    parser.add_argument('--max-edges',type=int,default=1000000)
    parser.add_argument('--lr',type=float,default=.0001)
    parser.add_argument('--grad-clip',type=float,default=1.)
    parser.add_argument('--seed',type=int,default=1731)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    fit(parser.parse_args())
