"""Local training dashboard: one run directory (--run), or every dense run under a root (--runs) with a
multi-run comparison page at / and per-run detail at /?run=<name>. Read-only except metrics/gpu.jsonl, which the
server appends to for dense runs with a live process (dense_config layout)."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import hashlib
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from collections import deque
from functools import lru_cache
import urllib.parse

import dense_config
import dense_openings
from rating_compat import audited_prior, scheduled_revision


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # Missing, locked by a Windows rename, or mid-write.
        return default


_report_digests={}


def report_digest(path,modified,size):
    cached=_report_digests.get(path)
    if cached is None or cached[:2]!=(modified,size):
        cached=(modified,size,hashlib.sha256(Path(path).read_bytes()).hexdigest())
        _report_digests[path]=cached
    return cached[2]


def run_histories(run, config):
    """Follow the hash-bound migration chain without changing the run."""
    expected=config.get('history_sha256');path=run/'history.json';histories=[]
    while expected:
        if len(expected)!=64 or any(character not in '0123456789abcdef' for character in expected):
            raise ValueError('Invalid migration history digest')
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            if path==run/'history.json':
                path=run/f'history-{expected}.json'
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Run migration history changed: {path}')
        history=read_json(path)
        histories.append(history)
        name=history.get('prior_history_file')
        if not name:break
        if Path(name).name!=name:raise ValueError('Invalid prior migration history path')
        path=run/name;expected=history['prior_history_sha256']
    return histories


def validate_archived_games(report, path):
    """Check saved scores against intact color-swapped games; archives lack a report seal."""
    games=report['games'];target=report['target_games'];metrics=report['metrics']
    if target<2 or target%2 or len(games)>target or len(games)%2:
        raise ValueError(f'Invalid archived comparison length: {path}')
    wins=losses=incomplete=pair_wins=pair_losses=ties=valid=0
    for pair in range(len(games)//2):
        first,second=games[2*pair:2*pair+2]
        if any(game.get('index')!=2*pair+color or game.get('pair')!=pair or
               game.get('challenger_color')!=color or game.get('seed')!=first.get('seed') or
               game.get('winner') not in (-1,0,1) or game.get('moves',[])[:len(game.get('opening',[]))]!=game.get('opening')
               for color,game in enumerate((first,second))) or first['opening']!=second['opening']:
            raise ValueError(f'Invalid archived opening pair: {path}')
        outcomes=[]
        for game in (first,second):
            if game['winner']<0:incomplete+=1
            else:
                success=game['winner']==game['challenger_color'];outcomes.append(success)
                wins+=success;losses+=not success
        if len(outcomes)==2:
            valid+=1;pair_wins+=sum(outcomes)==2;pair_losses+=sum(outcomes)==0;ties+=sum(outcomes)==1
    decisive=pair_wins+pair_losses
    pair_p=sum(math.comb(decisive,k) for k in range(pair_wins,decisive+1))/2**decisive if decisive else 1
    expected=dict(wins=wins,losses=losses,incomplete=incomplete,pending=0,played_games=len(games),
                  planned_games=len(games),opening_pair_wins=pair_wins,opening_pair_losses=pair_losses,
                  opening_pair_ties=ties,incomplete_pairs=len(games)//2-valid,opening_pair_p=pair_p)
    if any(metrics.get(key)!=value for key,value in expected.items()):
        raise ValueError(f'Archived comparison metrics disagree with games: {path}')


def historical_background(run, entries, hashes, histories):
    from checkpoint_league import evaluation_schedule, promotion_older, PROMOTION_RULE
    generations={history['previous_config_sha256']:history for history in histories
                 if 'previous_config_sha256' in history}
    inherited_through=max((history['through'] for history in histories),default=-1)
    for entry in entries.values():
        if entry['id']<=inherited_through:
            entry['inherited_checkpoint']=True
            if entry.get('evaluation_due') is False:
                entry['historical_background_status']='Not scheduled under the current protocol'
    seen={}
    for path in sorted((run/'background-evaluation').glob('*-vs-*.json')):
        report=read_json(path);identity=report['background_identity']
        history=generations.get(identity['config_sha256'])
        if history is None:continue
        previous=history['previous_identity']
        if identity['sources']!=previous['sources'] or identity['device']!='cpu':
            raise ValueError(f'Historical background worker identity changed: {path}')
        a,b=report['candidate'],report['opponent']
        if a>history['through'] or a not in entries or b not in entries:continue
        if path.name!=f'{a:04d}-vs-{b:04d}.json' or report.get('protocol')!=identity.get('protocol') or \
                report['target_games'] not in (previous['config']['eval_games'],previous['config']['reference_games']):
            raise ValueError(f'Historical background comparison settings changed: {path}')
        if report['model_hashes']!={str(n):hashes[n] for n in (a,b)}:
            raise ValueError(f'Historical background models changed: {path}')
        validate_archived_games(report,path)
        score={k:report['metrics'][k] for k in ('wins','losses','incomplete','opening_pair_p')}
        score.update(played=len(report['games']),planned=report['target_games'])
        record=entries[a].setdefault('historical_background',{})
        if 'champion' in record and record['champion']!=report['champion']:
            raise ValueError(f'Historical background champion changed: {path}')
        record['champion']=report['champion']
        seen.setdefault((a,identity['config_sha256']),set()).add(b)
        record['partial']=record.get('partial',False) or len(report['games'])<report['target_games']
        record['source']='Archived CPU report (not migration-hash verified)'
        if b==0:record['anchor_score']=score
        if b==a-1:record['previous_score']=score
        if b==report['champion']:record.update(champion_score=score,versus_champion=b)
        old_rule=history['previous_identity']['config'].get('promotion_rule')
        if (old_rule==PROMOTION_RULE and b==promotion_older(a,report['champion'])) or \
           (old_rule!=PROMOTION_RULE and b not in (0,a-1,report['champion'])):
            record['older_score']=dict(score,opponent=b)
    for (number,config_sha),opponents in seen.items():
        record=entries[number]['historical_background'];old=generations[config_sha]['previous_identity']['config']
        expected={opponent for opponent,_ in evaluation_schedule(number,record['champion'],old['eval_games'],
            old['reference_games'],legacy=old.get('promotion_rule')!=PROMOTION_RULE)}
        record['partial']=record['partial'] or opponents!=expected
    matched_record={}
    for history in reversed(histories):
        old=history['previous_identity']['config']
        sources=history['previous_identity']['sources'];runtime=history['previous_identity']['runtime']
        fingerprint=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        protocol=dict(max_plies=old.get('eval_max_plies',old.get('max_plies')),tactics=old.get('eval_tactics',False),
                      simulations=old.get('simulations'),root_samples=old.get('root_samples'),device=old.get('device'),
                      opening_suite='standard-v1',source_sha256=fingerprint(sources),runtime_sha256=fingerprint(runtime))
        for key,rating in history.get('prior_joint_ratings',{}).items():
            entry=entries.get(int(key))
            if entry is None:continue
            records=entry.get('historical_ratings',[]);last=matched_record.get(int(key),-1)
            def matches(record):
                return record.get('source')=='paired joint rating' and record.get('protocol')==protocol and \
                       record.get('elo')==rating.get('elo')
            index=next((i for i in range(last+1,len(records)) if matches(records[i])),None)
            if index is None and last>=0 and matches(records[last]):index=last
            if index is not None:
                records[index]['rating_pairs']=rating.get('rating_pairs')
                matched_record[int(key)]=index


def background_results(run, league):
    """Overlay separately published estimates without changing trainer-owned state."""
    background=read_json(run/'background-league.json',{})
    config_hash=hashlib.sha256((run/'config.json').read_bytes()).hexdigest()
    config=read_json(run/'config.json',{})
    histories=run_histories(run,config)
    entries={c['id']:c for c in league.get('checkpoints',[])}
    hashes={n:read_json(run/'checkpoints'/f'{n:04d}'/'manifest.json',{}).get('files',{}).get('model.pt') for n in entries}
    historical_background(run,entries,hashes,histories)
    for number,entry in entries.items():
        if not entry.get('pending'):continue
        for path in sorted((run/'evaluation').glob(f'{number:04d}-vs-*/report.json')):
            report=read_json(path);manifest=read_json(path.parent/'manifest.json');stat=path.stat()
            if league.get('rating_protocol') and manifest['identity'].get('protocol')!=league['rating_protocol']:
                continue
            a,b=report['candidate'],report['opponent']
            if a!=number or manifest['files']['report.json']!=report_digest(str(path),stat.st_mtime_ns,stat.st_size) or any(
                manifest['identity'][role]!=hashes.get(report[role]) for role in ('candidate','opponent')):
                raise ValueError('Pending evaluation report identity changed')
            score={k:report['metrics'][k] for k in ('wins','losses','incomplete','opening_pair_p')}
            if b==0:entry['anchor_score']=score
            if b==number-1:entry['previous_score']=score
            if b==league.get('champion'):entry.update(champion_score=score,versus_champion=b)
            if b not in (0,number-1,league.get('champion')):entry['older_score']=dict(score,opponent=b)
    if background.get('config_sha256')==config_hash:
        for record in background['checkpoints']:
            number=record['id']
            if number in entries and record.get('model_sha256')==hashes[number]:
                entries[number]['cpu_estimate']=record
        league['background_note']=background['note']
        league['background_protocol']=background.get('protocol')
    joint=read_json(run/'paired-ratings.json',{})
    if joint.get('config_sha256')==config_hash:
        prior=audited_prior(run,config,league.get('rating_protocol')) if league.get('rating_protocol') else None
        paths=[]
        for path in list((run/'evaluation').glob('*-vs-*/report.json'))+list((run/'background-evaluation').glob('*-vs-*.json')):
            if league.get('rating_protocol'):
                if path.name=='report.json':
                    revision=scheduled_revision(run,path,read_json(path),read_json(path.parent/'manifest.json'),
                                                hashes,league['rating_protocol'],prior)
                    if revision is None:continue
                elif read_json(path,{}).get('protocol')!=league['rating_protocol']:continue
            paths.append(path)
        current={}
        for path in paths:
            stat=path.stat()
            current[path.relative_to(run).as_posix()]=report_digest(str(path),stat.st_mtime_ns,stat.st_size)
        for path in set(_report_digests)-{str(path) for path in paths}:del _report_digests[path]
        stale=current!=joint.get('reports')
        for record in joint['checkpoints']:
            number=record['id']
            if number in entries and record.get('model_sha256')==hashes[number]:
                entries[number].update(record)
                entries[number]['rating_stale']=stale
        if not stale:
            for coverage in joint.get('diagnostics',{}).get('comparison_coverage',[]):
                number=coverage['candidate'];opponent=coverage['opponent']
                entry=entries.get(number)
                if entry is None:continue
                if coverage['origin']=='background':entry=entry.get('cpu_estimate')
                elif coverage['origin']!='scheduled':continue
                if entry is None:continue
                for key,matched in (('anchor_score',opponent==0),('previous_score',opponent==number-1),
                                    ('champion_score',opponent==entry.get('versus_champion')),
                                    ('older_score',opponent==entry.get('older_score',{}).get('opponent'))):
                    if matched and entry.get(key) and entry[key]['wins']==coverage['known_wins'] and \
                       entry[key]['losses']==coverage['known_losses'] and \
                       entry[key]['incomplete']==coverage['capped_games']:
                        entry[key]['rating_coverage']=coverage
        league['rating_method']=joint['rating_method']
        league['joint_rating_note']=joint['note']
        league['audited_compatible_reports']=joint.get('diagnostics',{}).get('audited_compatible_reports',0)
        league['rating_updated_at']=joint['updated_at']
        league['rating_stale']=stale
    for path in sorted((run/'value-diagnostics').glob('*.json')):
        probe=read_json(path,{})
        number=probe.get('checkpoint')
        if number in entries and probe.get('model_sha256')==hashes[number]:entries[number]['fresh_validation']=probe
    return read_json(run/'background-status.json',{})


@lru_cache(maxsize=32)
def episode_counts(path, modified):
    episodes = read_json(Path(path), [])
    return {"games": len(episodes), "terminal_games": sum(e["winner"] >= 0 for e in episodes),
            "bootstrapped_games": sum(e["winner"] < 0 for e in episodes),
            "positions": sum(len(e["moves"]) for e in episodes)}


@lru_cache(maxsize=64)
def evaluation_openings(path, modified):
    report = read_json(Path(path), {})
    games = report.get('games', [])
    return [dict(id=f"{report['candidate']}-{report['opponent']}-{game['pair']}",
                 candidate=report['candidate'], opponent=report['opponent'],
                 pair=game['pair'], seed=game['seed'], moves=game['opening'],
                 games=[g['index'] for g in games if g['pair'] == game['pair']])
            for game in games if game['index'] % 2 == 0]


def bound_evaluation(run, checkpoint):
    model = run/"checkpoints"/f"{checkpoint:04d}"/"model.nnue"
    if not model.exists():
        return None
    current = hashlib.sha256(model.read_bytes()).hexdigest()
    for folder in (run/"evaluation/confirmation", run/"evaluation"):
        status = read_json(folder/"status.json")
        if not status:
            continue
        candidate = status.get("candidate_sha256")
        if not candidate:
            report = read_json(folder/"report.json", {})
            path = report.get("config", {}).get("candidate")
            candidate = report.get("identity", {}).get(path)
        if candidate == current:
            return {**status, "candidate_sha256": current, "checkpoint": checkpoint}
    return None


def klent_run(run):
    checkpoints = []
    identity, schema = {}, None
    for path in sorted((run/"checkpoints").glob("[0-9][0-9][0-9][0-9]/manifest.json")):
        manifest = read_json(path)
        if not manifest or not manifest.get("schema", "").startswith("hexo-klent-"):
            continue
        identity, schema = manifest["identity"], manifest["schema"]
        checkpoints.append({"id": int(path.parent.name), "metrics": manifest.get("metrics"),
                            "actor_sha256": manifest.get("files", {}).get("klent.pt")})
    if not schema:
        return None
    status = read_json(run/"status.json", {})
    number = status.get("iteration", checkpoints[-1]["id"])
    corpus = run/"corpus"/f"{number:04d}"
    corpus_manifest = read_json(corpus/"manifest.json", {})
    active = dict(status)
    episodes = corpus/"episodes.json"
    if corpus_manifest and episodes.exists():
        active.update(episode_counts(str(episodes), episodes.stat().st_mtime_ns))
    totals = {key: sum((c["metrics"] or {}).get(key, 0) for c in checkpoints)
              for key in ("games", "positions", "terminal_games", "bootstrapped_games", "optimizer_steps")}
    # A published checkpoint already includes the last status: never double count it.
    if number > checkpoints[-1]["id"]:
        for key in totals:
            if key in active:
                totals[key] += active[key]
    return {"name": run.name, "schema": schema, "config": identity.get("config", {}),
            "stage": status.get("stage", "initialized"), "iteration": number,
            "active": active, "totals": totals, "checkpoints": checkpoints,
            "actor": corpus_manifest.get("identity", {}).get("policy", "softmax((Q+beta*logpi)/(alpha+beta))"),
            "actions": corpus_manifest.get("identity", {}).get("actions", "full-legal"),
            "actor_sha256": corpus_manifest.get("identity", {}).get("actor_sha256") or status.get("actor_sha256") or checkpoints[-1]["actor_sha256"],
            "source_sha256": identity.get("sources", {}).get("klent.py"),
            "engine_sha256": identity.get("engine_sha256"), "training_lock_present": (run/"training.lock").exists(),
            "status_modified": (run/"status.json").stat().st_mtime if (run/"status.json").exists() else None,
            "evaluation": bound_evaluation(run, checkpoints[-1]["id"]),
            "rating": "UNRATED", "rating_reason": "External paired match evidence is not attached to this run."}


def relational_run(run, declared_family=None):
    plan = read_json(run.with_suffix('.plan.json'), {})
    launch = read_json(run.with_suffix('.launch.json'), {})
    status_path = run/"status.json"
    if not status_path.exists():
        status_path = run.with_suffix('.status.json')
    status = read_json(status_path, {})
    manifests = sorted((run/'checkpoints').glob('[0-9][0-9][0-9][0-9]/manifest.json'))
    manifest_path = manifests[-1] if manifests else run/'manifest.json'
    manifest = read_json(manifest_path, {})
    identity = manifest.get('identity', {})
    provenance = read_json(run/'provenance.json', {})
    family = identity.get('backbone') or identity.get('kind') or status.get('schema') or provenance.get('model_family') or plan.get('kind') or declared_family or ''
    if 'relational' not in family:
        return None
    evaluation = status if provenance.get('schema') == 'hexo-relational-evaluation-v1' else None
    evaluation_provenance = provenance if evaluation is not None else {}
    evaluation_path = run if evaluation is not None else None
    model_hash = manifest.get('files', {}).get('model.pt') or status.get('candidate_sha256') or provenance.get('model_input_sha256', {}).get('candidate')
    if not evaluation and model_hash:
        for folder in (run/'evaluation/confirmation', run/'evaluation'):
            candidate = read_json(folder/'status.json', {})
            candidate_provenance = read_json(folder/'provenance.json', {})
            candidate_hash = candidate.get('candidate_sha256') or candidate_provenance.get('model_input_sha256', {}).get('candidate')
            if candidate_hash == model_hash:
                evaluation = candidate
                evaluation_provenance = candidate_provenance
                evaluation_path = folder
                break
    warmstart = identity.get('kind') == 'relational-human-policy-q-v1' or 'epoch' in status or 'epochs' in status
    training_backend = 'Not recorded in evaluation artifact' if provenance else (
        'human policy/Q fitting' if warmstart else 'terminal teacher critic fitting' if identity.get('kind') == 'relational-terminal-teacher-v1' else 'KLENT policy/Q' if identity.get('backbone') else 'Not published yet')
    backend = (evaluation or {}).get('backend') or evaluation_provenance.get('backend')
    opponent = (evaluation or {}).get('opponent')
    if not opponent and evaluation_provenance.get('config', {}).get('seal_revision'):
        config = evaluation_provenance['config']
        opponent = dict(backend='seal', ms=config.get('seal_ms'), revision=config['seal_revision'])
    artifact = manifest_path if manifest else run.with_suffix('.plan.json') if plan else None
    live = status
    live_status_path = status_path
    if evaluation_path and (evaluation_path/'report.json').exists():
        artifact = evaluation_path/'report.json'
    heartbeat = live.get('heartbeat') or live.get('updated_at')
    if heartbeat is None and live_status_path.exists():
        heartbeat = live_status_path.stat().st_mtime
    return dict(name=run.name, path=str(run), model_family='relational-policy-q',
        phase=live.get('stage', 'initialized'), training_backend=training_backend,
        evaluation_backend=backend, checkpoint_sha256=model_hash,
        initial_checkpoint_sha256=identity.get('initial_model_sha256') or identity.get('config', {}).get('initial_model_sha256') or plan.get('initial_model_sha256'),
        launch_source_commit=launch.get('source_commit'),
        source_sha256=identity.get('sources') or provenance.get('files_sha256'),
        opponent=opponent, evaluation=evaluation,
        heartbeat=heartbeat,
        workers=live.get('workers', []), last_artifact=str(artifact) if artifact else None,
        status=live, training_status=status, metrics=manifest.get('metrics', {}), config=identity.get('config', {}),
        rating='UNRATED', model_identity_pending=not bool(manifest or provenance))


def tail_events(path, limit, window=128000):
    recent = []
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size-window))
            if size > window: stream.readline()
            for line in stream:
                try: recent.append(json.loads(line))
                except ValueError: pass  # A writer may be appending the last line.
    except OSError:
        pass
    return recent[-limit:]


_dense_manifests = {}


def dense_manifests(folder, pattern='*/manifest.json'):
    """(path, manifest) pairs under numbered folders, re-read only when their mtime changes."""
    found = []
    for path in folder.glob(pattern):
        if not path.parent.name.isdigit(): continue
        try: modified = path.stat().st_mtime_ns
        except OSError: continue
        cached = _dense_manifests.get(path)
        if cached is None or cached[0] != modified:
            manifest = read_json(path)
            if not isinstance(manifest, dict): continue
            cached = _dense_manifests[path] = (modified, manifest)
        found.append((path, cached[1]))
    return sorted(found, key=lambda item: (item[0].parent.parent.name, int(item[0].parent.name)))


def provisional(league, evaluator):
    """The league row of the checkpoint under evaluation, merged from evaluator-status.json (league.json stays the
    settled record): {id, opponent, wins, losses, capped, games, games_planned, elo, elo_interval} while the
    evaluator is playing or throttled with a tally for a candidate that has no checkpoint entry and is no decided
    variant (one with a verdict), else None. elo and elo_interval are the opponent's league Elo (Seal:
    anchors.seal.elo) plus the tally's elo_delta and elo_interval, None while either is unknown."""
    comparison, tally = evaluator.get('comparison'), evaluator.get('tally')
    if evaluator.get('stage') not in ('playing', 'throttled') or not isinstance(comparison, dict) or not isinstance(tally, dict):
        return None
    entries = {c.get('id'): c for c in league.get('checkpoints') or [] if isinstance(c, dict)}
    decided = {v.get('id') for v in league.get('variants') or [] if isinstance(v, dict) and v.get('verdict')}
    if comparison.get('candidate') in entries or comparison.get('candidate') in decided:
        return None
    opponent = comparison.get('opponent')
    base = ((league.get('anchors') or {}).get('seal') or {}).get('elo') if opponent == 'seal' else (entries.get(opponent) or {}).get('elo')
    interval = tally.get('elo_interval')
    known = finite(base) and finite(tally.get('elo_delta'))
    return dict(id=comparison.get('candidate'), opponent=opponent, **{k: tally.get(k) for k in ('wins', 'losses', 'capped', 'games')},
                games_planned=evaluator.get('games_planned'), elo=base+tally['elo_delta'] if known else None,
                elo_interval=[base+v for v in interval] if known and isinstance(interval, list) and all(map(finite, interval)) else None)


def dense_run(run, config, fresh=30):
    """/api/run payload of a dense run (layout: dense_config); processes silent for `fresh` seconds are not live."""
    now = time.time()
    status = lambda path: (lambda value: value if isinstance(value, dict) else {})(read_json(path, {}))
    beat = lambda value: max(0, now-value['updated_at']) if isinstance(value.get('updated_at'), (int, float)) else None
    actors = [dict(status(path), process=path.stem.removeprefix('actor-status').lstrip('-') or '0')
              for path in sorted(run.glob('actor-status*.json'), key=lambda path: (len(path.stem), path.stem))]
    for actor in actors: actor['heartbeat'] = beat(actor)
    solver = status(run/'solver-status.json')
    # Rates only count live processes; cumulative counters count every process.
    live = [a for a in actors if a['heartbeat'] is not None and a['heartbeat'] <= fresh and a.get('stage') != 'failed']
    total = lambda rows, key: sum(a.get(key) or 0 for a in rows)
    def weighted(key, weight):
        rows = [(a[key], a.get(weight) or 0) for a in live if isinstance(a.get(key), (int, float))]
        mass = sum(w for _, w in rows)
        return sum(v*w for v, w in rows)/mass if mass else (rows[0][0] if len(rows) == 1 else None)
    actor = dict(processes=len(actors), live_processes=len(live),
                 stage=actors[0].get('stage') if len(actors) == 1 else ', '.join(sorted({a.get('stage') or '?' for a in actors})) or None,
                 placements_per_second=total(live, 'placements_per_second'), evals_per_second=total(live, 'evals_per_second'),
                 active_games=total(live, 'active_games'), positions=total(actors, 'positions'),
                 games_completed=total(actors, 'games_completed'), shards_written=total(actors, 'shards_written'),
                 games_total=total(actors, 'games_total') if actors and all(a.get('games_total') is not None for a in actors) else None,
                 mean_batch=weighted('mean_batch', 'evals_per_second'),
                 full_batch_fraction=weighted('full_batch_fraction', 'batch_calls'),
                 terminal_fraction=weighted('terminal_fraction', 'games_completed'),
                 mean_plies=weighted('mean_plies', 'games_completed'),
                 vram_reserved_mb=[a['vram'].get('reserved_mb') for a in live if isinstance(a.get('vram'), dict)],
                 checkpoint=actors[0].get('checkpoint') if actors else None, actor_sha256=actors[0].get('actor_sha256') if actors else None,
                 error='; '.join(f"{a['process']}: {a['error']}" for a in actors if a.get('error')) or None,
                 heartbeat=max((a['heartbeat'] for a in actors if a['heartbeat'] is not None), default=None),
                 **{key: solver.get(name) for key, name in (('restart_buffer', 'buffer_size'), ('proofs_verified', 'verified'),
                                                            ('verify_timeouts', 'verify_timeouts'))})
    learners = {}
    for path in sorted(run.glob('learner-status*.json')):
        learner = status(path)
        variant = learner.get('variant') or path.stem.removeprefix('learner-status').lstrip('-') or 'main'
        learners[variant] = dict(learner, variant=variant, heartbeat=beat(learner))
    learners = dict(sorted(learners.items(), key=lambda item: (item[0] != 'main', item[0])))
    keys = ('games', 'rows', 'policy_rows', 'terminal_games', 'capped_games')
    data = dict.fromkeys(keys, 0)
    shards = [manifest for _, manifest in dense_manifests(run/'shards')]
    recent = {3600: 0, 21600: 0}
    restarts = 0
    for shard in shards:
        counts = shard.get('counts') or {}
        for key in keys: data[key] += counts.get(key) or 0
        for window in recent:
            if now-(shard.get('created_at') or 0) <= window: recent[window] += counts.get('games') or 0
        if now-(shard.get('created_at') or 0) <= 21600: restarts += counts.get('restart_games') or 0
    # A young run is averaged over its lifetime, not the full window.
    age = now-config['created_at'] if isinstance(config.get('created_at'), (int, float)) else None
    for window, games in recent.items():
        span = max(60, min(window, age)) if age is not None else window
        data[f'games_per_hour_{window//3600}h'] = games*3600/span
    data.update(shards=len(shards), latest_shard_at=max((s.get('created_at') or 0 for s in shards), default=None),
                restart_share_6h=restarts/recent[21600] if recent[21600] else None)
    checkpoints, per_variant = [], {}
    for path, manifest in dense_manifests(run/'checkpoints', '*/*/manifest.json'):
        variant, step = path.parent.parent.name, int(path.parent.name)
        per_variant.setdefault(variant, []).append(dict(manifest, variant=variant, step=step, id=f'{variant}/{step:06d}'))
    for rows in per_variant.values(): checkpoints += rows[-50:]  # Newest 50 per variant.
    champion = status(run/'champion.json')
    champion['age'] = beat(champion)
    evaluator = status(run/'evaluator-status.json')
    evaluator['heartbeat'] = beat(evaluator)
    league = status(run/'league.json')
    evaluator['provisional'] = provisional(league, evaluator)
    return dict(name=run.name, config=config, actor=actor, actors=actors, learners=learners, evaluator=evaluator,
                league=league, champion=champion, checkpoints=checkpoints, data=data, now=now)


_jsonl = {}


def read_jsonl(path):
    """Dict records of an append-only JSON-lines file. Only newly appended complete lines are parsed when the
    (mtime, size) changes; malformed lines and a partial last line are skipped; a shrunk file is re-read."""
    try: stat = path.stat()
    except OSError: return []
    cached = _jsonl.get(path)
    if cached and cached[:2] == (stat.st_mtime_ns, stat.st_size): return cached[3]
    offset, rows = (cached[2], cached[3]) if cached and stat.st_size >= cached[2] else (0, [])
    try:
        with path.open('rb') as stream:
            stream.seek(offset)
            chunk = stream.read()
    except OSError: return rows
    end = chunk.rfind(b'\n')+1
    for line in chunk[:end].splitlines():
        try: record = json.loads(line)
        except ValueError: continue
        if isinstance(record, dict): rows.append(record)
    _jsonl[path] = (stat.st_mtime_ns, stat.st_size, offset+end, rows)
    return rows


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def dense_config_of(run):
    config = read_json(run/'config.json', {})
    return config if isinstance(config, dict) and config.get('schema') == dense_config.SCHEMA else None


def openings(run):
    """/api/openings: {suite: dense_openings.Book.graph} of every opening book file of a dense run: the DAG's nodes
    with their statistics, its edges and the book's stats."""
    return {suite: book.graph() for suite, book in dense_openings.books(run).items()}


def heartbeats(run, now):
    """({variant: learner status}, [actor status]) with 'age' seconds since each process's last heartbeat."""
    def load(path):
        value = read_json(path, {})
        value = value if isinstance(value, dict) else {}
        return dict(value, age=now-value['updated_at'] if finite(value.get('updated_at')) else None)
    learners = {}
    for path in run.glob('learner-status*.json'):
        value = load(path)
        learners[value.get('variant') or path.stem.removeprefix('learner-status').lstrip('-') or 'main'] = value
    return learners, [load(path) for path in run.glob('actor-status*.json')]


def live(run, fresh=30):
    learners, actors = heartbeats(run, time.time())
    return any(s['age'] is not None and s['age'] <= fresh and s.get('stage') not in ('failed', 'idle', 'finished')
               for s in [*learners.values(), *actors])


def rated(run):
    """League checkpoints with an Elo, each with its manifest created_at when the checkpoint still exists."""
    league = read_json(run/'league.json', {})
    created = {f'{path.parent.parent.name}/{path.parent.name}': manifest.get('created_at')
               for path, manifest in dense_manifests(run/'checkpoints', '*/*/manifest.json')}
    return [dict(c, created_at=created.get(c.get('id'))) for c in (league.get('checkpoints') if isinstance(league, dict) else None) or []
            if finite(c.get('elo')) and isinstance(c.get('step'), int)]


def project(root, fresh=30):
    """/api/project: every dense run directly under `root`, oldest first."""
    now, runs = time.time(), []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        config = dense_config_of(run)
        if config is None: continue
        learners, actors = heartbeats(run, now)
        steps = {path.parent.parent.name: int(path.parent.name) for path, _ in dense_manifests(run/'checkpoints', '*/*/manifest.json')}
        for path in (run/'metrics').glob('learner-*.jsonl'):
            rows = read_jsonl(path)
            variant = path.stem.removeprefix('learner-')
            steps[variant] = max(steps.get(variant, 0), next((r['step'] for r in reversed(rows) if isinstance(r.get('step'), int)), 0))
        for variant, status in learners.items():
            steps[variant] = max(steps.get(variant, 0), status.get('step') or 0)
        elo = {}
        for c in rated(run):
            if c['step'] >= elo.get(c['variant'], {'step': -1})['step']:
                elo[c['variant']] = dict(id=c['id'], step=c['step'], elo=c['elo'], interval=c.get('elo_interval'))
        ages = [a['age'] for a in actors if a['age'] is not None]
        runs.append(dict(name=run.name, created_at=config.get('created_at'), live=live(run, fresh),
                         variants=sorted(steps, key=lambda v: (v != 'main', v)), steps=steps, elo=elo,
                         champion=read_json(run/'champion.json', {}).get('checkpoint'),
                         learner_heartbeat={v: s['age'] for v, s in learners.items()},
                         learner_stage={v: s.get('stage') for v, s in learners.items()},
                         actor_heartbeat=min(ages, default=None), actor_processes=len(actors),
                         logs=sorted(p.stem for p in (run/'metrics').glob('*.jsonl'))))
    return dict(root=str(root), now=now, runs=runs)


HEADS = ('policy_ce', 'value_bce', 'short_value_bce', 'next_ce', 'future_bce', 'outcome_bce')  # dense_learn.LOGGED
SOURCES = ('converted', 'fresh', 'newest')  # dense_data.SOURCES
SOURCE_METRICS = tuple(f'{s}_{k}{h}' for s in SOURCES for k in ('', 'train_', 'gap_') for h in ('policy_ce', 'value_bce'))
POLICY_METRICS = ('policy_kl', 'policy_target_entropy', 'policy_top1')
POLICY_VALIDATION_METRICS = POLICY_METRICS+tuple(f'{s}_{h}' for s in SOURCES for h in POLICY_METRICS)
CURVE_SOURCES = ('fresh', 'newest')  # dense_learn.CURVE_SOURCES
CURVE_SCALARS = tuple(f'{s}_{k}' for s in CURVE_SOURCES
                      for k in ('value_bce_last20', 'value_horizon', 'policy_ce_early', 'policy_ce_late',
                                'value_regret', 'value_regret_early', 'value_regret_late'))
CURVE_AXES = dict(value_curve='remaining', value_excess_curve='remaining', policy_ce_curve='ply', value_bce_by_ply='ply')
CURVE_METRICS = {f'{s}_{k}': x for s in CURVE_SOURCES for k, x in CURVE_AXES.items()}  # metric: its only x (grid <x>_grid)
SURFACE_METRICS = {f'{s}_{k}': field for s in CURVE_SOURCES for k, field in
                   (('value_surface', 'value'), ('value_excess_surface', 'excess'), ('policy_surface', 'policy'))}  # metric: cell field
LEARNER_METRICS = HEADS+('lr', 'samples_per_second', 'window_rows')+tuple('validation_'+h for h in HEADS+SOURCE_METRICS+POLICY_VALIDATION_METRICS+CURVE_SCALARS)
ACTOR_SUMMED = ('placements_per_second', 'evals_per_second', 'games_per_hour')
ACTOR_METRICS = ACTOR_SUMMED+('terminal_fraction', 'mean_plies')
GPU_METRICS = ('utilization', 'used_mib', 'watts', 'temperature')
ACTOR_STALE = 90.  # seconds after which an actor's last metrics line no longer counts


def downsample(points, max_points):
    """At most max(4, max_points) points of x-sorted (x, y, ...) tuples: the first and last point, and the
    lowest and highest y within each of equal-width x buckets, in x order."""
    if len(points) <= max(4, max_points): return points
    buckets = max(1, (max_points-2)//2)
    x0, width = points[0][0], (points[-1][0]-points[0][0])/buckets or 1.
    groups = {}
    for p in points[1:-1]:
        groups.setdefault(min(buckets-1, int((p[0]-x0)/width)), []).append(p)
    kept = [points[0]]
    for key in sorted(groups):
        group = groups[key]
        low, high = min(group, key=lambda p: p[1]), max(group, key=lambda p: p[1])
        kept += sorted({id(low): low, id(high): high}.values(), key=lambda p: p[0])
    return kept+[points[-1]]


def actor_points(run):
    """(time, {metric: value}) after each actor metrics line, combining every worker's latest line younger than
    ACTOR_STALE: rates summed, terminal_fraction and mean_plies weighted by games completed. games_per_hour is
    each worker's games_completed delta over the time between its consecutive lines."""
    lines = sorted(((path.stem, r) for path in (run/'metrics').glob('actor-*.jsonl') for r in read_jsonl(path)
                    if finite(r.get('time'))), key=lambda item: item[1]['time'])
    latest, previous, out = {}, {}, []
    for worker, r in lines:
        last, previous[worker] = previous.get(worker), r
        r = dict(r, games_per_hour=None)
        if last and finite(r.get('games_completed')) and finite(last.get('games_completed')) and \
                r['games_completed'] >= last['games_completed'] and r['time'] > last['time']:
            r['games_per_hour'] = (r['games_completed']-last['games_completed'])*3600/(r['time']-last['time'])
        latest[worker] = r
        current = [v for v in latest.values() if r['time']-v['time'] <= ACTOR_STALE]
        values = {m: sum(v[m] for v in current if finite(v.get(m))) for m in ACTOR_SUMMED}
        if r['games_per_hour'] is None and not any(finite(v.get('games_per_hour')) for v in current):
            values['games_per_hour'] = None
        for m in ACTOR_METRICS[len(ACTOR_SUMMED):]:
            rows = [(v[m], v.get('games_completed') or 0) for v in current if finite(v.get(m))]
            mass = sum(w for _, w in rows)
            values[m] = sum(x*w for x, w in rows)/mass if mass else None
        out.append((r['time'], values))
    return out


def series(run, config, variant, metric, x='step', max_points=1000, from_step=0):
    """/api/series: [[x, y], ...] (elo: [[x, elo, low, high], ...] with the 95% interval; seal_delta: the direct-match
    Elo minus Seal of each anchored checkpoint, league anchors.seal.matches) sorted by x and
    downsampled after dropping steps below from_step when x is 'step'; x is the learner step or hours since
    the run's created_at. Actor and GPU metrics have hours only. CURVE_METRICS have their own x only ('remaining' or 'ply'): [[grid point, y or null where unsupported], ...]
    over every point of the manifest's <x>_grid (not downsampled) of the newest checkpoint manifest of
    `variant` holding that curve (metrics.validation_sources), whose id is added as `checkpoint` (None without one).
    Raises ValueError for an unknown metric or x."""
    created = config.get('created_at') or 0.
    hours = lambda t: (t-created)/3600
    if metric in CURVE_METRICS:
        if x != CURVE_METRICS[metric]: raise ValueError(f'{metric} has x {CURVE_METRICS[metric]} only')
        found = [(path, v) for path, m in dense_manifests(run/'checkpoints'/variant)
                 if isinstance(v := (m.get('metrics') or {}).get('validation_sources'), dict) and isinstance(v.get(metric), list)]
        path, v = found[-1] if found else (None, {})
        points = [[g, y if finite(y) else None] for g, y in zip(v.get(f'{x}_grid') or [], v.get(metric) or [])]
        return dict(run=run.name, variant=variant, metric=metric, x=x, count=len(points), points=points,
                    checkpoint=path and f'{variant}/{path.parent.name}')
    if x not in ('step', 'hours'): raise ValueError(f'unknown x {x!r}')
    if metric in LEARNER_METRICS:
        validation, key = metric.startswith('validation_'), metric.removeprefix('validation_')
        kept = []
        for r in read_jsonl(run/'metrics'/f'learner-{variant}.jsonl'):
            if bool(r.get('validation')) != validation or not isinstance(r.get('step'), int) or not finite(r.get('time')): continue
            while kept and kept[-1][0] >= r['step']: kept.pop()  # A resumed learner repeats steps after its last export.
            kept.append((r['step'], r['time'], r.get(key)))
        points = [(step if x == 'step' else hours(t), y) for step, t, y in kept if finite(y)]
    elif metric == 'elo':
        points = sorted((c['step'] if x == 'step' else hours(c['created_at']), c['elo'],
                         *(c['elo_interval'] if isinstance(c.get('elo_interval'), list) else (c['elo'], c['elo'])))
                        for c in rated(run) if c.get('variant') == variant and (x == 'step' or finite(c['created_at'])))
    elif metric == 'seal_delta':
        made = {c['id']: c['created_at'] for c in rated(run)}
        league = read_json(run/'league.json', {})
        matches = ((league.get('anchors') or {}).get('seal') or {}).get('matches') or [] if isinstance(league, dict) else []
        points = sorted((int(m['checkpoint'].split('/')[1]) if x == 'step' else hours(made[m['checkpoint']]), m['elo_delta'])
                        for m in matches if m['checkpoint'].split('/')[0] == variant and finite(m.get('elo_delta'))
                        and (x == 'step' or finite(made.get(m['checkpoint']))))
    elif x != 'hours':
        raise ValueError(f'{metric} has hours only')
    elif metric in ACTOR_METRICS:
        points = [(hours(t), values[metric]) for t, values in actor_points(run) if finite(values[metric])]
    elif metric in GPU_METRICS:
        points = sorted((hours(r['time']), r[metric]) for r in read_jsonl(run/'metrics'/'gpu.jsonl')
                        if finite(r.get('time')) and finite(r.get(metric)))
    else:
        raise ValueError(f'unknown metric {metric!r}')
    if x == 'step': points = [p for p in points if p[0] >= from_step]
    return dict(run=run.name, variant=variant, metric=metric, x=x, count=len(points),
                points=[list(p) for p in downsample(points, max_points)])


def surface(run, variant, metric):
    """/api/surface: {run, variant, metric, checkpoint, ply_bins, remaining_bins, values, counts} from SURFACE_METRICS
    `metric` of the newest checkpoint manifest of `variant` holding it (metrics.validation_sources; dense_learn.surfaces):
    values[i][j] is the cell's loss (null under the learner's row minimum) for ply bin i and remaining bin j; empty
    grids and checkpoint None without one. Raises ValueError for an unknown metric."""
    if metric not in SURFACE_METRICS: raise ValueError(f'unknown surface {metric!r}')
    found = [(path, v[metric]) for path, m in dense_manifests(run/'checkpoints'/variant)
             if isinstance(v := (m.get('metrics') or {}).get('validation_sources'), dict) and isinstance(v.get(metric), dict)]
    path, grid = found[-1] if found else (None, {})
    values = [[y if finite(y) else None for y in row] for row in grid.get(SURFACE_METRICS[metric]) or []]
    return dict(run=run.name, variant=variant, metric=metric, checkpoint=path and f'{variant}/{path.parent.name}',
                ply_bins=grid.get('ply_bins') or [], remaining_bins=grid.get('remaining_bins') or [], values=values,
                counts=grid.get('counts') or [])


def sample_gpu(watched, period=10.):
    """Refresh Handler.gpu_status every 2 s; about every `period` s append a metrics/gpu.jsonl line to each
    run of watched()."""
    last = 0.
    while True:
        gpu = Handler.gpu_status()['gpu']
        if gpu and time.time()-last >= period:
            last = time.time()
            for run in watched():
                dense_config.append_metrics(run, 'gpu', **{k: gpu[k] for k in GPU_METRICS})
        time.sleep(2)


class Handler(BaseHTTPRequestHandler):
    run = Path("runs/selfplay")
    runs = None  # project root when started with --runs
    model_family = None
    hardware = {"time": 0, "gpu": None}
    hardware_history = deque(maxlen=300)

    @classmethod
    def gpu_status(cls):
        now = time.time()
        if now-cls.hardware["time"] >= 2:
            fields = "utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu"
            try:
                output = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True, timeout=2,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                output.check_returncode()
                values = output.stdout.strip().splitlines()[0].split(",")
                numbers = [float(v) if v.strip() != "[N/A]" else None for v in values]
                gpu = dict(zip(("utilization", "memory_utilization", "used_mib", "total_mib", "watts", "power_limit", "temperature"), numbers))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                gpu = None
            cls.hardware = {"time": now, "gpu": gpu}
            if gpu:
                cls.hardware_history.append({"time": now, **gpu})
        return {**cls.hardware, "history": list(cls.hardware_history)}

    def resolve(self, name):
        """The run a request addresses: Handler.run without a name, else the directory `name` under Handler.runs."""
        if name is None:
            return None if self.runs else self.run
        run = self.runs/name if self.runs and name == Path(name).name and not name.startswith('.') else None
        if run is None or not (run/'config.json').is_file():
            raise LookupError(name)
        return run

    def do_GET(self):
        url = urllib.parse.urlsplit(self.path)
        query = dict(urllib.parse.parse_qsl(url.query))
        try:
            run = self.resolve(query.get('run'))
        except LookupError:
            self.send_error(404, 'Unknown run')
            return
        if url.path == "/":
            page = "project.html" if self.runs and run is None else "training.html"
            payload = (Path(__file__).parent / "web" / page).read_bytes()
            content_type = "text/html; charset=utf-8"
        elif url.path == "/openings.js":
            payload = (Path(__file__).parent / "web/openings.js").read_bytes()
            content_type = "text/javascript; charset=utf-8"
        elif url.path in ("/api/project", "/api/series", "/api/surface") and self.runs:
            try:
                if url.path == "/api/project":
                    data = project(self.runs)
                else:
                    config = dense_config_of(run) if run else None
                    if config is None: raise ValueError('series need a dense run')
                    variant, metric = query.get('variant', 'main'), query.get('metric', '')
                    data = surface(run, variant, metric) if url.path == "/api/surface" else series(
                        run, config, variant, metric, query.get('x', 'step'), max(10, min(20000, int(query.get('max_points', 1000)))),
                        int(query.get('from_step', 0)))
            except ValueError as error:
                self.send_error(400, str(error))
                return
            payload = json.dumps(data, allow_nan=False).encode()
            content_type = "application/json"
        elif url.path == "/api/openings" and run:
            if dense_config_of(run) is None:
                self.send_error(400, 'opening books belong to dense runs')
                return
            payload = json.dumps(openings(run), allow_nan=False).encode()
            content_type = "application/json"
        elif url.path == "/api/run" and run:
            summary = run / "summary.json"
            events = run / "events.jsonl"
            search_config = read_json(run/'config.json', {})
            dense = search_config.get('schema') == dense_config.SCHEMA
            relational = None if dense else relational_run(run, self.model_family)
            if dense:
                data = dict(kind='dense', dense=dense_run(run, search_config), events=tail_events(events, 300))
            elif search_config.get('backbone') == 'hexo-relational-policy-value-v1':
                recent = []
                if events.exists():
                    with events.open('rb') as stream:
                        stream.seek(0, 2)
                        size = stream.tell()
                        stream.seek(max(0, size-128000))
                        if size > 128000: stream.readline()
                        for line in stream:
                            try: recent.append(json.loads(line))
                            except json.JSONDecodeError: pass
                league = read_json(run/'league.json', {})
                checkpoint_ids = {c['id'] for c in league.get('checkpoints', [])}
                for path in sorted((run/'checkpoints').glob('[0-9][0-9][0-9][0-9]/manifest.json')):
                    number = int(path.parent.name)
                    if number not in checkpoint_ids:
                        manifest = read_json(path, {})
                        league.setdefault('checkpoints', []).append(dict(id=number, elo=None,
                            elo_interval=None, promoted=False, pending=True, loss=manifest.get('metrics')))
                background_status=background_results(run,league)
                data = dict(kind='search', search=dict(name=run.name,background_status=background_status,
                    config=search_config, status=read_json(run/'status.json', {}),
                    league=league, openings=[opening
                        for path in sorted((run/'evaluation').glob('*-vs-*/report.json'), reverse=True)[:16]
                        for opening in evaluation_openings(str(path), path.stat().st_mtime_ns)]), events=recent[-300:])
            elif relational:
                data = {"kind": "relational", "relational": relational, "summary": None, "events": []}
            elif not summary.exists():
                klent = klent_run(run)
                data = {"kind": "klent" if klent else "native", "klent": klent, "summary": None, "events": []}
            else:
                recent = []
                if events.exists():
                    # Read only the tail so a long run does not grow every response.
                    with events.open("rb") as stream:
                        stream.seek(0, 2)
                        size = stream.tell()
                        stream.seek(max(0, size-256000))
                        if size > 256000:
                            stream.readline()
                        for line in stream.readlines():
                            try:
                                recent.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass  # The trainer may be writing the last line.
                data = {"kind": "native", "summary": read_json(summary), "events": recent[-500:]}
            data["hardware"] = self.gpu_status()
            payload = json.dumps(data).encode()
            content_type = "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/selfplay")
    parser.add_argument("--runs", help='serve the multi-run comparison of every dense run under this directory')
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--model-family", choices=['relational-policy-q'], help='Identify a run before its first manifest is published')
    args = parser.parse_args()
    Handler.run = Path(args.run).resolve()
    Handler.runs = Path(args.runs).resolve() if args.runs else None
    Handler.model_family = args.model_family
    candidates = (lambda: [p for p in Handler.runs.iterdir() if p.is_dir()]) if Handler.runs else (lambda: [Handler.run])
    threading.Thread(target=sample_gpu, args=(lambda: [r for r in candidates() if dense_config_of(r) and live(r)],),
                     daemon=True).start()
    print(f"Training dashboard: http://127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
