"""Continue a search run one saved checkpoint at a time.

This controller never starts beside the original trainer. Evaluation workers
must stop for a source/configuration migration; compatible read-only workers
may run after it. Each child restores the last saved model and Adam state,
evaluates the new checkpoint, and exits before the next begins. A published
checkpoint with interrupted evaluation can resume under the same run identity.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from klent import digest
from search_train import source_identity, runtime_identity, verify_artifact


NUMBERS = ('replay_positions','reuse_ratio','games','eval_games','reference_games','envs',
           'simulations','root_samples','leaf_batch','batch','epochs','max_plies',
           'max_nodes','max_edges','cache_positions','seed')
MIGRATION_LOCKS = ('background-evaluation.lock','paired-ratings.lock')


def saved_boundary(run,*,migration=False):
    present=[name for name in ('training.lock',*(MIGRATION_LOCKS if migration else ())) if (run/name).exists()]
    if present:raise ValueError(f'Active run workers hold {present}')
    status=json.loads((run/'status.json').read_text())
    league=json.loads((run/'league.json').read_text())
    numbers=sorted(c['id'] for c in league['checkpoints'])
    if numbers!=list(range(len(numbers))):raise ValueError('Checkpoint league is not contiguous')
    latest=numbers[-1]
    if status['stage']!='finished':
        if migration:raise ValueError('A source/configuration migration requires a finished boundary')
        identity=json.loads((run/'config.json').read_text())
        if status.get('iteration')==latest and latest>0:
            entry=next(c for c in league['checkpoints'] if c['id']==latest)
            if not entry.get('evaluation_due') or entry.get('champion_score',{}).get('pessimistic_opening_pair_p') is None:
                raise ValueError('Latest league checkpoint has no completed evaluation')
            verify_artifact(run/'checkpoints'/f'{latest:04d}',dict(identity,checkpoint=latest))
        else:
            pending=latest+1;checkpoint=run/'checkpoints'/f'{pending:04d}'
            if status.get('iteration')!=pending or not checkpoint.exists():
                raise ValueError('Continuation requires a finished boundary or a published pending checkpoint')
            manifest=verify_artifact(checkpoint,dict(identity,checkpoint=pending))
            corpus=run/'corpus'/f'{pending:04d}'/'manifest.json'
            if manifest['metrics']['corpus_sha256']!=digest(corpus):
                raise ValueError('Pending checkpoint consumed a different corpus')
    if not (run/'checkpoints'/f'{latest:04d}'/'optimizer.pt').exists():
        raise ValueError('Latest saved Adam state is missing')
    return latest


def command(run,through,settings,upgrade):
    args=[sys.executable,str(Path(__file__).with_name('search_train.py')),
          '--run',str(run),'--initial-model',settings['initial_model'],
          '--iterations',str(through),'--device',settings['device'],
          '--evaluate-every','1','--eval-max-plies',str(settings['eval_max_plies']),
          '--lr',str(settings['lr'])]
    for name in NUMBERS:args.extend(('--'+name.replace('_','-'),str(settings[name])))
    for name in ('actor_tactics','eval_tactics'):
        args.append(('--' if settings[name] else '--no-')+name.replace('_','-'))
    if upgrade:args.append('--upgrade-run')
    return args


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--through',required=True,type=int,help='Last checkpoint to save and evaluate')
    parser.add_argument('--eval-max-plies',required=True,type=int)
    parser.add_argument('--eval-tactics',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--plan',action='store_true',help='Check boundary and print the exact continuation plan')
    args=parser.parse_args();run=args.run.resolve()
    if args.eval_max_plies<5:parser.error('Evaluation cap must fit the five-ply standard opening')
    previous=json.loads((run/'config.json').read_text())
    tactics=previous['config'].get('eval_tactics',False) if args.eval_tactics is None else args.eval_tactics
    settings=dict(previous['config'],eval_max_plies=args.eval_max_plies,eval_tactics=tactics,evaluate_every=1)
    first_upgrade=(previous['sources']!=source_identity() or previous['config']!=settings
                   or previous['runtime']!=runtime_identity())
    latest=saved_boundary(run,migration=first_upgrade)
    if args.through<=latest:parser.error('Through must exceed the saved checkpoint')
    missing=[name for name in (*NUMBERS,'initial_model','device','lr') if name not in settings]
    if missing:raise ValueError(f'Run settings are incomplete: {missing}')
    if digest(Path(settings['initial_model']))!=previous['config']['initial_sha256']:
        raise ValueError('Initial checkpoint changed')
    planned={'run':str(run),'from_checkpoint':latest,'through':args.through,'evaluations':'every saved checkpoint',
             'eval_max_plies':args.eval_max_plies,'eval_tactics':settings['eval_tactics'],
             'actor_tactics':settings['actor_tactics'],'source_changed':previous['sources']!=source_identity(),
             'runtime_changed':previous['runtime']!=runtime_identity(),
             'resume_pending_checkpoint':latest+1 if json.loads((run/'status.json').read_text()).get('iteration')==latest+1 else None,
             'saved_model_sha256':digest(run/'checkpoints'/f'{latest:04d}'/'model.pt'),
             'saved_optimizer_sha256':digest(run/'checkpoints'/f'{latest:04d}'/'optimizer.pt')}
    print(json.dumps(planned,indent=2),flush=True)
    if args.plan:return
    log=run/'continuation.jsonl'
    for number in range(latest+1,args.through+1):
        current=json.loads((run/'config.json').read_text())
        upgrade=(current['sources']!=source_identity() or current['config']!=settings
                 or current['runtime']!=runtime_identity())
        if saved_boundary(run,migration=upgrade)!=number-1:raise ValueError('Run changed between controller iterations')
        child=command(run,number,settings,upgrade)
        with log.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'at':time.time(),'checkpoint':number,'event':'start',
                                     'previous_model':digest(run/'checkpoints'/f'{number-1:04d}'/'model.pt'),
                                     'previous_optimizer':digest(run/'checkpoints'/f'{number-1:04d}'/'optimizer.pt'),
                                     'evaluation':{'max_plies':args.eval_max_plies,'tactics':settings['eval_tactics']}})+'\n')
        subprocess.run(child,check=True,cwd=Path(__file__).resolve().parent)
        if saved_boundary(run)!=number:raise ValueError('Child did not publish its expected checkpoint')
        league=json.loads((run/'league.json').read_text())
        entry=league['checkpoints'][number]
        if not entry['evaluation_due'] or entry['champion_score'].get('pessimistic_opening_pair_p') is None:
            raise ValueError('Child checkpoint has no completed conservative evaluation')
        with log.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'at':time.time(),'checkpoint':number,'event':'complete',
                                     'model':digest(run/'checkpoints'/f'{number:04d}'/'model.pt'),
                                     'optimizer':digest(run/'checkpoints'/f'{number:04d}'/'optimizer.pt'),
                                     'champion':league['champion']})+'\n')


if __name__=='__main__':main()
