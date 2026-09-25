"""One-time correction of an inherited champion decision from sealed match evidence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def bound_history(run, config):
    expected=config.get('history_sha256')
    if not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('Run config has no valid migration history digest')
    path=run/'history.json'
    if digest(path)!=expected:raise ValueError('Migration history disagrees with run config')
    return read_json(path)


def bound_artifact(run, history, relative, verify_artifact):
    expected=history.get('artifacts',{}).get(relative)
    if expected is None:raise ValueError(f'Artifact is not in inherited history: {relative}')
    path=run/relative
    if digest(path/'manifest.json')!=expected:raise ValueError(f'Inherited manifest changed: {relative}')
    return verify_artifact(path)


def reconsider(run, source, candidate, expected_champion, apply=False):
    run=run.resolve();source=source.resolve()
    if (run/'training.lock').exists():raise ValueError('Stop the trainer before champion reconsideration')
    config_path=run/'config.json';league_path=run/'league.json'
    config_sha=digest(config_path);league_sha=digest(league_path)
    config=read_json(config_path);league=read_json(league_path)
    if config.get('run')!=str(run):raise ValueError('Run path disagrees with config identity')
    sys.path.insert(0,str(source))
    from search_train import pessimistic_promotion, source_identity, runtime_identity, verify_artifact
    from train import paired_metrics, write_json
    if config.get('sources')!=source_identity() or config.get('runtime')!=runtime_identity():
        raise ValueError('Engine source or runtime disagrees with active run identity')
    history=bound_history(run,config)
    if candidate>history.get('through',-1):raise ValueError('Candidate is not inherited by migration history')
    if league['champion']!=expected_champion or candidate<=league['champion']:
        raise ValueError('Champion has changed or candidate is not newer')
    entries=league['checkpoints'];ids=[entry['id'] for entry in entries]
    if ids!=list(range(len(ids))) or candidate>=len(entries):raise ValueError('Checkpoint league is incomplete')
    entry=entries[candidate]
    if entry.get('promoted') or entry.get('promotion_reconsideration'):
        raise ValueError('Candidate already promoted or reconsidered')
    if entry.get('versus_champion')!=expected_champion or not entry.get('evaluation_due'):
        raise ValueError('Candidate was not evaluated against the expected champion')
    for folder in (run/'evaluation').glob('[0-9][0-9][0-9][0-9]-vs-*'):
        try:number=int(folder.name[:4])
        except ValueError:continue
        if number>=len(entries):raise ValueError(f'Pending checkpoint already has evaluation evidence: {folder}')
    candidate_manifest=bound_artifact(run,history,f'checkpoints/{candidate:04d}',verify_artifact)
    champion_manifest=bound_artifact(run,history,f'checkpoints/{expected_champion:04d}',verify_artifact)
    relative=f'evaluation/{candidate:04d}-vs-{expected_champion:04d}'
    manifest=bound_artifact(run,history,relative,verify_artifact)
    report_path=run/relative/'report.json';report=read_json(report_path);identity=manifest['identity']
    if identity.get('run')!=history['previous_identity']:
        raise ValueError('Match does not belong to inherited evaluation protocol')
    if identity.get('candidate')!=candidate_manifest['files']['model.pt'] or \
       identity.get('opponent')!=champion_manifest['files']['model.pt']:
        raise ValueError('Match models disagree with sealed checkpoints')
    old=history['previous_identity']['config']
    if identity.get('games')!=old['eval_games'] or identity.get('simulations')!=old['simulations'] or \
       identity.get('root_samples')!=old['root_samples'] or identity.get('opening_suite')!='standard-v1':
        raise ValueError('Match budget or opening protocol changed')
    if report.get('candidate')!=candidate or report.get('opponent')!=expected_champion:
        raise ValueError('Match report names the wrong checkpoints')
    if report.get('metrics')!=paired_metrics(report['games'],report['metrics']['planned_games']):
        raise ValueError('Saved game outcomes disagree with report metrics')
    raw={key:report['metrics'][key] for key in ('wins','losses','incomplete','opening_pair_p')}
    if entry.get('champion_score')!=raw:raise ValueError('League score disagrees with inherited report')
    promoted,worst=pessimistic_promotion(report)
    if not promoted or worst is None:raise ValueError('Conservative cap rule does not promote candidate')
    protocol=dict(max_plies=old.get('eval_max_plies',old['max_plies']),tactics=old.get('eval_tactics',False),
                  simulations=old['simulations'],root_samples=old['root_samples'],device=old['device'],
                  opening_suite=identity['opening_suite'],games=identity['games'],seed=identity['seed'])
    correction=dict(original_champion=league['champion'],original_promoted=entry['promoted'],
                    original_champion_score=copy.deepcopy(entry['champion_score']),
                    evidence_manifest_sha256=history['artifacts'][relative],report_sha256=digest(report_path),
                    history_sha256=config['history_sha256'],prior_league_sha256=league_sha,
                    rule='pessimistic_promotion: capped games count as challenger losses',
                    rule_source_sha256=config['sources']['search_train.py'],protocol=protocol,
                    pessimistic_score={key:worst[key] for key in ('wins','losses','opening_pair_p')},
                    corrected_at=time.time())
    result=dict(candidate=candidate,previous_champion=expected_champion,new_champion=candidate,
                raw_score=raw,pessimistic_score=correction['pessimistic_score'],
                report_sha256=correction['report_sha256'],history_sha256=config['history_sha256'])
    if apply:
        if (run/'training.lock').exists() or digest(config_path)!=config_sha or digest(league_path)!=league_sha:
            raise ValueError('Run changed while champion correction was prepared')
        if digest(run/'history.json')!=config['history_sha256'] or digest(report_path)!=correction['report_sha256']:
            raise ValueError('Bound match evidence changed while champion correction was prepared')
        for relative_path in (f'checkpoints/{candidate:04d}',f'checkpoints/{expected_champion:04d}',relative):
            bound_artifact(run,history,relative_path,verify_artifact)
        entry['promotion_reconsideration']=correction
        entry['promoted']=True
        entry['champion_score'].update(pessimistic_wins=worst['wins'],pessimistic_losses=worst['losses'],
                                       pessimistic_opening_pair_p=worst['opening_pair_p'])
        league['champion']=candidate
        write_json(league_path,league)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--source-dir',type=Path,required=True,help='Frozen source directory recorded by active config')
    parser.add_argument('--candidate',type=int,required=True)
    parser.add_argument('--expected-champion',type=int,required=True)
    parser.add_argument('--apply',action='store_true',help='Atomically update league.json after all checks')
    args=parser.parse_args()
    result=reconsider(args.run,args.source_dir,args.candidate,args.expected_champion,args.apply)
    print(json.dumps(dict(result,applied=args.apply),indent=2))


if __name__=='__main__':main()
