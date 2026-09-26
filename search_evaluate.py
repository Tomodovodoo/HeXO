"""Background paired checkpoint estimates; never mutates the learner or champion."""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from checkpoint_league import evaluation_schedule, promotion_older
from klent import digest
from paired_rating import fit_ratings, METHOD as JOINT_METHOD, NOTE as JOINT_NOTE
from search_train import play_games, source_identity, read_corpus, evaluation_protocol, pessimistic_promotion
from relational_train import load_model
from relational_model import NeuralEvaluator
from train import paired_metrics, task_opening, write_json


def read(path):
    return json.loads(path.read_text())


def validation_probe(run, first, status):
    """Evaluate the actor on next-collection games it has never trained on."""
    folder=run/'value-diagnostics';folder.mkdir(exist_ok=True)
    for path in sorted((run/'corpus').glob('[0-9][0-9][0-9][0-9]')):
        number=int(path.name);output=folder/f'{number:04d}.json'
        if number<first+1 or output.exists() or not (path/'manifest.json').exists():continue
        manifest=read(path/'manifest.json');model,sha=checkpoint(run,number-1)
        if manifest['identity']['actor_sha256']!=sha:raise ValueError('Validation actor changed')
        episodes,rows=read_corpus(path,manifest['identity'])
        terminal=[e for e in episodes if e['winner']>=0]
        if not terminal:continue
        selected=np.random.default_rng(1740+number).choice(len(terminal),min(32,len(terminal)),replace=False)
        lookup={(r['game'],r['ply']):r for r in rows}
        evaluator=NeuralEvaluator(load_model(model,'cpu',expected_sha256=sha),'cpu',model_version=sha)
        samples=[]
        status('validation-loss',candidate=number-1,corpus=number)
        for index in selected:
            episode=terminal[index]
            for fraction in (.25,.5,.75):
                ply=int(len(episode['moves'])*fraction);row=lookup[episode['id'],ply]
                prediction=evaluator.evaluate([episode['moves'][:ply]])[0]
                if digest_bytes(prediction['actions'].astype(np.int64).tobytes())!=row['legal_sha256']:
                    raise ValueError('Validation legal action order changed')
                logits=prediction['logits'].astype(np.float64);logp=logits-np.logaddexp.reduce(logits)
                samples.append(dict(game=episode['id'],ply=ply,target=row['target'],value=float(prediction['q'][0]),
                    policy_ce=float(-np.dot(row['policy'],logp))))
        values=np.array([s['value'] for s in samples]);targets=np.array([s['target'] for s in samples])
        write_json(output,dict(corpus=number,checkpoint=number-1,model_sha256=sha,
            corpus_sha256=digest(path/'manifest.json'),games=len(selected),positions=len(samples),
            terminal_games=len(terminal),capped_games=len(episodes)-len(terminal),
            policy_ce=float(np.mean([s['policy_ce'] for s in samples])),value_mse=float(np.mean((values-targets)**2)),
            prediction_mean=float(values.mean()),prediction_std=float(values.std()),zero_baseline_mse=1.,
            note='CPU float32, next-collection terminal games only. Actor has not trained on these games. Policy targets come from its search. Not Elo.',samples=samples))
        return


def digest_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def checkpoint(run,number):
    path=run/'checkpoints'/f'{number:04d}'
    manifest=read(path/'manifest.json');identity=read(run/'config.json')
    history=read(run/'history.json') if (run/'history.json').exists() else {}
    inherited=history.get('artifacts',{}).get(f'checkpoints/{number:04d}')
    if inherited:
        if digest(path/'manifest.json')!=inherited:raise ValueError('Inherited checkpoint changed')
    elif manifest['identity']!=dict(identity,checkpoint=number):
        raise ValueError('Checkpoint identity changed')
    if digest(path/'model.pt')!=manifest['files']['model.pt']:raise ValueError('Checkpoint model changed')
    return path/'model.pt',manifest['files']['model.pt']


def reports(run):
    result=[]
    protocol=read(run/'league.json').get('rating_protocol')
    for path in sorted((run/'evaluation').glob('*-vs-*/report.json')):
        manifest=read(path.parent/'manifest.json')
        if protocol and manifest['identity'].get('protocol')!=protocol:continue
        if digest(path)!=manifest['files']['report.json']:raise ValueError('Evaluation report changed')
        report=read(path)
        for role in ('candidate','opponent'):
            if read(run/'checkpoints'/f"{report[role]:04d}"/'manifest.json')['files']['model.pt']!=manifest['identity'][role]:
                raise ValueError('Evaluation opponents changed')
        result.append(report)
    result.extend(report for path in sorted((run/'background-evaluation').glob('*-vs-*.json'))
                  if (report:=read(path)) and (not protocol or report.get('protocol')==protocol))
    return result


def publish_ratings(run,worker_identity):
    league=read(run/'league.json');ids=[c['id'] for c in league['checkpoints']]
    # CPU estimates are their own league. CUDA scheduled matches and older CPU
    # worker generations must not enter this fit.
    data=[read(path) for path in sorted((run/'background-evaluation').glob('*-vs-*.json'))]
    data=[report for report in data if report.get('background_identity')==worker_identity]
    if not data:return
    model_hashes={n:read(run/'checkpoints'/f'{n:04d}'/'manifest.json')['files']['model.pt'] for n in ids}
    for report in data:
        if 'background_identity' in report:
            if report['background_identity']!=worker_identity:raise ValueError('Background evaluation configuration changed')
            if report['model_hashes']!={str(n):model_hashes[n] for n in (report['candidate'],report['opponent'])}:
                raise ValueError('Background model changed')
    ratings,diagnostics=fit_ratings(ids,data,seed=worker_identity['seed'])
    entries={n:dict(record,model_sha256=model_hashes[n]) for n,record in ratings.items()}
    coverage={(item['candidate'],item['opponent']):item for item in diagnostics['comparison_coverage']
              if item['origin']=='background'}
    for report in data:
        if 'background_identity' not in report:continue
        if report['background_identity']!=worker_identity:raise ValueError('Background evaluation configuration changed')
        a,b=report['candidate'],report['opponent']
        if report['model_hashes']!={str(a):model_hashes[a],str(b):model_hashes[b]}:raise ValueError('Background model changed')
        record=entries[a];record['background']=True
        record['provisional']=record.get('provisional',False) or len(report['games'])<report['target_games'] or \
            bool(record.get('rating_censored_pairs')) or bool(record.get('rating_unplayed_pairs'))
        score={k:report['metrics'][k] for k in ('wins','losses','incomplete','opening_pair_p')}
        score.update(played=len(report['games']),planned=report['target_games'])
        score['rating_coverage']=coverage[a,b]
        if b==a-1:record['previous_score']=score
        if b==report['champion']:
            record['champion_score']=score;record['versus_champion']=b
        if b==0:record['anchor_score']=score
        if b==promotion_older(a,report['champion']):record['older_score']=dict(score,opponent=b)
    for number,record in entries.items():
        if not record.get('background'):continue
        candidate_reports=[r for r in data if r['candidate']==number and 'background_identity' in r]
        champion=candidate_reports[0]['champion']
        expected=dict(evaluation_schedule(number,champion,worker_identity['eval_games'],worker_identity['reference_games']))
        completed={r['opponent']:len(r['games']) for r in candidate_reports}
        record['provisional']=record['provisional'] or any(completed.get(opponent,0)<count for opponent,count in expected.items())
        older=promotion_older(number,champion)
        if older is None:record['background_decision']='No distinct older opponent'
        else:
            wins=[]
            for opponent,key in ((champion,'champion_score'),(older,'older_score')):
                report=next((r for r in candidate_reports if r['opponent']==opponent),None)
                if report is None or len(report['games'])<expected[opponent]:
                    record['background_decision']='Pending';break
                won,worst=pessimistic_promotion(report);wins.append(won)
                record[key].update(pessimistic_wins=worst['wins'],pessimistic_losses=worst['losses'],
                                   pessimistic_opening_pair_p=worst['opening_pair_p'])
            else:record['background_decision']='Promotion evidence' if all(wins) else 'No promotion evidence'
    write_json(run/'background-league.json',dict(config_sha256=digest(run/'config.json'),updated_at=time.time(),
        protocol=worker_identity['protocol'],checkpoints=list(entries.values()),rating_method=JOINT_METHOD,
        diagnostics=diagnostics,note='Separate CPU float32 estimates from background paired games. CUDA scheduled ratings use their own protocol. CPU estimates do not promote champions. '+JOINT_NOTE))


def next_comparison(run,identity,first):
    league=read(run/'league.json');jobs=[]
    for entry in league['checkpoints']:
        number=entry['id']
        if number<first or entry.get('evaluation_due') is not False:continue
        champion=max(c['id'] for c in league['checkpoints'] if c['id']<number and c.get('promoted'))
        for opponent,target in evaluation_schedule(number,champion,identity['eval_games'],identity['reference_games']):
            path=run/'background-evaluation'/f'{number:04d}-vs-{opponent:04d}.json'
            report=read(path) if path.exists() else None
            if report and report['background_identity']!=identity:raise ValueError('Background evaluation settings changed')
            completed=len(report['games']) if report else 0
            if completed<target:jobs.append((completed/target,-number,opponent,target,champion,path,report))
    return min(jobs,key=lambda job:job[:3]) if jobs else None


def main(args):
    run=Path(args.run).resolve();folder=run/'background-evaluation';folder.mkdir(exist_ok=True)
    torch.set_num_threads(args.threads)
    run_identity=read(run/'config.json');config=run_identity['config']
    history=read(run/'history.json') if (run/'history.json').exists() else {}
    inherited_through=history.get('through',-1) if history and run_identity.get('history_sha256')==digest(run/'history.json') else -1
    sources=source_identity()
    if sources!=run_identity['sources']:raise ValueError('Evaluator sources or native libraries differ from the training run')
    play=SimpleNamespace(**config);play.device='cpu';play.envs=2
    identity=dict(config_sha256=digest(run/'config.json'),sources=sources,worker_sha256=digest(Path(__file__)),
        device='cpu',precision='float32',threads=args.threads,simulations=config['simulations'],root_samples=config['root_samples'],
        eval_games=config['eval_games'],reference_games=config['reference_games'],seed=config['seed']+700000,
        protocol=evaluation_protocol(play))
    lock=run/'background-evaluation.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    def status(stage,**data):write_json(run/'background-status.json',dict(stage=stage,updated_at=time.time(),**data))
    try:
        while True:
            if digest(run/'config.json')!=identity['config_sha256']:raise ValueError('Run upgraded; restart the background evaluator explicitly')
            validation_probe(run,max(args.from_checkpoint,inherited_through),status)
            job=next_comparison(run,identity,max(args.from_checkpoint,inherited_through+1))
            if job is None:
                publish_ratings(run,identity);status('waiting-for-checkpoints')
                if args.once:return
                time.sleep(30);continue
            _,negative,opponent,target,champion,path,report=job;candidate=-negative
            a,ah=checkpoint(run,candidate);b,bh=checkpoint(run,opponent)
            base=identity['seed']+candidate*1000+opponent*1000003
            report=report or dict(candidate=candidate,opponent=opponent,champion=champion,target_games=target,games=[],
                background_identity=identity,model_hashes={str(candidate):ah,str(opponent):bh},protocol=identity['protocol'])
            if report['model_hashes']!={str(candidate):ah,str(opponent):bh} or len(report['games'])%2:
                raise ValueError('Invalid saved background pair or model identity')
            pair=len(report['games'])//2
            opening=task_opening(base+pair,True,play.eval_max_plies,'standard-v1')['opening']
            def progress(data):status('playing',candidate=candidate,opponent=opponent,paired_games_saved=2*pair,target_games=target,**data)
            episodes,_=play_games([a,b],[(0,1),(1,0)],[opening,opening],play,base+pair*1009,progress)
            for e in episodes:
                report['games'].append(dict(index=2*pair+e['id'],pair=pair,seed=base+pair,challenger_color=e['id'],
                    winner=e['winner'],reason=e['reason'],opening=e['opening'],moves=e['moves']))
            # Only complete color-swapped pairs enter the current estimate.
            report['metrics']=paired_metrics(report['games'],len(report['games']))
            write_json(path,report);publish_ratings(run,identity)
            if args.once:return
    except BaseException as error:
        status('failed',error=repr(error));raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--from-checkpoint',type=int,default=1);parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if args.threads<1 or args.from_checkpoint<1:parser.error('Positive thread count and first checkpoint required')
    main(args)
