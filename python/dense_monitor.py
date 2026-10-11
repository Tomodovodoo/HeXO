"""Record training progress without model inference or extra evaluation games.

Run hourly: python python/dense_monitor.py --run runs/dense-v1 --out artifacts/live/training-monitor
Use --baseline once before a rollout and --note to record configuration changes.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None


def write(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def tail(path, limit=16*1024*1024):
    """Bound log reads; discard a leading fragment and a writer's incomplete final line."""
    if not path.exists():
        return [], 0
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size-limit))
        if size > limit:
            stream.readline()
        data = stream.read()
    rows, malformed = [], 0
    for line in data.splitlines(keepends=True):
        if not line.endswith(b'\n') or not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            malformed += 1
    return rows, malformed


def mean(rows, field):
    values = [r[field] for r in rows if isinstance(r.get(field), (int, float)) and math.isfinite(r[field])]
    return statistics.mean(values) if values else None


def progress(rows, field):
    """Observed increments within sessions; omit unknown work across restarts."""
    records = [r for r in rows if field in r]
    total, resets, boundaries = 0, 0, 0
    for a, b in zip(records, records[1:]):
        delta = b[field]-a[field]
        resets += delta < 0
        boundary = a.get('session') != b.get('session') or delta < 0
        boundaries += boundary
        if not boundary:
            total += delta
    return dict(increment=total, resets=resets, unobserved_boundaries=boundaries,
                note='Observed work only. First observations and restart boundaries are excluded; legacy logs cannot identify every restart.',
                observed_seconds=records[-1]['time']-records[0]['time'] if len(records) > 1 else 0)


def metrics(run, variant, now, hours):
    result = {}
    actors = sorted((p.stem for p in (run/'metrics').glob('actor-*.jsonl')
                     if p.stem.removeprefix('actor-').isdigit()),
                    key=lambda name: int(name.removeprefix('actor-')))
    actor_times = []
    for name in (f'learner-{variant}', *actors, 'gpu'):
        stored, malformed = tail(run/'metrics'/f'{name}.jsonl')
        rows = [r for r in stored if r['time'] >= now-hours*3600]
        result[name] = dict(records=len(rows), malformed_records_in_log_tail=malformed, last=rows[-1] if rows else None)
        if name.startswith('learner-'):
            samples = progress(rows, 'samples_seen')
            result[name].update(samples=samples, steps=progress(rows, 'step'),
                wall_samples_per_second=samples['increment']/samples['observed_seconds'] if samples['observed_seconds'] else None,
                active_samples_per_second=mean([r for r in rows if r.get('samples_per_second', 0) > 0], 'samples_per_second'),
                grad_norm=mean(rows, 'grad_norm'), grad_clip_scale=mean(rows, 'grad_clip_scale'),
                clipped_observations=sum(r.get('grad_clip_scale', 1) < 1 for r in rows),
                gradient_observations=sum('grad_clip_scale' in r for r in rows))
        elif name.startswith('actor-'):
            result[name].update(games=progress(rows, 'games_completed'), placements=progress(rows, 'positions'),
                                active_placements_per_second=mean([r for r in rows if r.get('placements_per_second', 0) > 0], 'placements_per_second'))
            if rows:
                actor_times.extend((rows[0]['time'], rows[-1]['time']))
        else:
            temperatures = [r['temperature'] for r in rows if isinstance(r.get('temperature'), (int, float)) and math.isfinite(r['temperature'])]
            result[name].update(utilization=mean(rows, 'utilization'), watts=mean(rows, 'watts'),
                                max_temperature=max(temperatures, default=None))
    seconds = max(actor_times)-min(actor_times) if actor_times else 0
    summary = dict(workers=actors, observed_seconds=seconds)
    for field in ('games', 'placements'):
        summary[field] = {key: sum(result[name][field][key] for name in actors)
                          for key in ('increment', 'resets', 'unobserved_boundaries')}
        summary[f'wall_{field}_per_second'] = summary[field]['increment']/seconds if seconds else None
    rates = [result[name]['active_placements_per_second'] for name in actors
             if result[name]['active_placements_per_second'] is not None]
    summary['active_placements_per_second'] = sum(rates) if rates else None
    summary['note'] = 'Observed work summed across workers; restart boundaries excluded. Active throughput sums each worker\'s active mean, not a simultaneous measurement.'
    result['actors'] = summary
    return result


def checkpoint(path):
    manifest = read(path/'manifest.json')
    measured = manifest.get('metrics') or {}
    sources = measured.get('validation_sources') or {}
    return dict(id=f'{path.parent.name}/{path.name}', step=manifest['step'], time=manifest['created_at'],
                learner=manifest['learner'], retained_rows=manifest.get('rows'),
                validation=measured.get('validation'),
                sources={k: v for k, v in sources.items() if not isinstance(v, (list, dict))})


def recent_shards(run, count):
    # Names are timestamp ordered; hidden staging directories are not published shards.
    names = sorted((p for p in (run/'shards').iterdir() if p.is_dir() and not p.name.startswith('.')),
                   key=lambda p: p.name)
    totals, direct, by_phase, checked = Counter(), {}, {}, []
    for path in reversed(names):
        manifest = read(path/'manifest.json')
        if not manifest or (manifest.get('origin') or ('converted' if 'source' in manifest['identity'] else 'actor')) != 'actor':
            continue
        checked.append(path.name)
        totals.update(manifest['counts'])
        for row in read(path/'rows.json'):
            search = row.get('search') or {}
            stone = 'first' if row['remaining'] == 2 else 'second'
            phase = 'early' if row['ply'] < 20 else 'middle' if row['ply'] < 60 else 'late'
            by_phase.setdefault(phase, Counter()).update(rows=1, proven=bool(row.get('proven')), surprise='surprise' in row)
            if 'surprise' in row:
                direct.setdefault(stone, []).append(search.get('comparison_credits', 0))
        if len(checked) == count:
            break
    return dict(shards=checked, counts=dict(totals), phase_counts={k: dict(v) for k, v in by_phase.items()},
                full_root_comparison_credits={k: dict(rows=len(v), mean=statistics.mean(v), median=statistics.median(v))
                                             for k, v in direct.items()},
                note='Recent actor shard sample, not all newly generated data. Proof shards are excluded.')


def league_summary(league, variant):
    if not league:
        return None
    checkpoints = [c for c in league['checkpoints'] if c['variant'] == variant]
    return dict(champion=league.get('champion'),
                recent=[{k: c.get(k) for k in ('id', 'step', 'elo', 'elo_interval', 'matches')} for c in checkpoints[-8:]],
                note='Internal Elo is relative to this league. Match counts and opponent identities matter; protocols and openings can differ.')


def external_estimates(run):
    """Newest saved estimate per report, provided every referenced report is intact."""
    estimates, timestamps, digests = {}, {}, {}
    for metadata in sorted((run/'matches').glob('*/*-elo-estimate.json')):
        saved = read(metadata)
        if saved.get('schema') not in ('hexo-external-elo-estimate-v1', 'hexo-external-elo-estimate-v2'):
            continue
        sources = saved.get('sources', [])
        if saved['schema'] == 'hexo-external-elo-estimate-v2' and not sources:
            continue
        intact = True
        for source in (saved['match'], *sources):
            evidence = (metadata.parent/source['report']).resolve()
            if not evidence.is_relative_to(run.resolve()) or not evidence.is_file():
                intact = False
                break
            if evidence not in digests:
                digests[evidence] = hashlib.sha256(evidence.read_bytes()).hexdigest()
            if digests[evidence] != source['report_sha256']:
                intact = False
                break
        if not intact:
            continue
        calculated = saved['calculated_at']
        if isinstance(calculated, str):
            try:
                date = datetime.fromisoformat(calculated)
                calculated = (date if date.tzinfo else date.replace(tzinfo=timezone.utc)).timestamp()
            except (ValueError, OverflowError, OSError):
                continue
        if not isinstance(calculated, (int, float)) or not math.isfinite(calculated):
            continue
        report = (metadata.parent/saved['match']['report']).resolve()
        if report not in timestamps or calculated > timestamps[report]:
            estimates[report], timestamps[report] = saved, calculated
    return estimates


def external_reports(run):
    reports = {'six': [], 'pulsatrix': []}
    estimates = external_estimates(run)
    for path in sorted((run/'matches').glob('*/report.json'), key=lambda p: p.stat().st_mtime):
        data = read(path)
        estimate = estimates.get(path.resolve())
        identity = json.dumps((estimate or data).get('opponent', {})).lower()
        if not any(word in identity for word in ('six', 'tyto', 'pulsatrix')):
            identity = path.parent.name.lower()
        if not any(word in identity for word in ('six', 'tyto', 'pulsatrix')):
            continue
        family = 'six' if 'six' in identity else 'pulsatrix'
        reports[family].append(dict(path=path.relative_to(run).as_posix(),
                                   kind='calibration' if 'calibration' in path.parent.name.lower() else 'external_match',
                                   estimate_metadata=estimate,
                                   summary={k: data[k] for k in ('status', 'manifest', 'score', 'summary', 'statistics', 'local', 'opponent', 'settings', 'wins', 'losses', 'capped') if k in data},
                                   games=estimate['match']['games'] if estimate else
                                   len(data['games']) if isinstance(data.get('games'), list) else None))
    return {k: v[-6:] for k, v in reports.items()}


def comparison(current, reference):
    a, b = current['checkpoints'][-1], reference['checkpoints'][-1]
    keys = ('policy_kl', 'policy_first_kl', 'policy_second_kl', 'outcome_bce_unproven', 'certified_policy_first_mass')
    delta = {}
    for name in keys:
        x, y = (a.get('validation') or {}).get(name), (b.get('validation') or {}).get(name)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            delta[name] = x-y
    return dict(from_checkpoint=b['id'], to_checkpoint=a['id'], step_increment=a['step']-b['step'],
                moving_validation_delta=delta,
                identical_newest_source=bool(a['sources'].get('newest_checkpoint')) and a['sources'].get('newest_checkpoint') == b['sources'].get('newest_checkpoint'),
                note='Changing replay or source panels prevent causal attribution from these deltas alone.')


def collect(run, variant, out, hours, shards, baseline=False, note=None, commands=None):
    out.mkdir(parents=True, exist_ok=True)
    if baseline and (out/'baseline.json').exists():
        raise FileExistsError('Baseline already exists; preserve it and journal later changes')
    previous = read(out/'latest.json')
    now = time.time()
    paths = sorted((p for p in (run/'checkpoints'/variant).iterdir() if p.name.isdigit() and (p/'manifest.json').exists()),
                   key=lambda p: int(p.name))
    if not paths:
        raise ValueError(f'No complete checkpoints for {variant}')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    result = dict(time=now, revision=revision, run=str(run.resolve()), variant=variant,
                  hours=hours, checkpoints=[checkpoint(p) for p in paths[-3:]],
                  learner_status=read(run/f'learner-status-{variant}.json') if variant != 'main' else read(run/'learner-status.json'),
                  actor_status=read(run/'actor-status.json'), evaluator_status=read(run/'evaluator-status.json'),
                  actor_statuses={p.stem: read(p) for p in sorted(run.glob('actor-status*.json'))},
                  metrics=metrics(run, variant, now, hours), recent_actor_sample=recent_shards(run, shards),
                  league=league_summary(read(run/'league.json'), variant), external_reports=external_reports(run))
    if commands:
        result['commands'] = read(commands)
        result['commands_sha256'] = hashlib.sha256(commands.read_bytes()).hexdigest()
    if previous:
        result['since_last'] = comparison(result, previous)
    original = read(out/'baseline.json')
    if original:
        result['since_baseline'] = comparison(result, original)
    target = out/f'{int(now)}.json'
    write(target, result)
    write(out/'latest.json', result)
    if baseline:
        write(out/'baseline.json', result)
    if note:
        with (out/'changes.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(time=now, revision=revision, checkpoint=result['checkpoints'][-1]['id'],
                                         commands_sha256=result.get('commands_sha256'), note=note))+'\n')
    return result, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variant', help='Learner variant; defaults to the run configuration')
    parser.add_argument('--hours', type=float, default=24)
    parser.add_argument('--shards', type=int, default=8)
    parser.add_argument('--commands', type=Path)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--note')
    args = parser.parse_args()
    if args.hours <= 0 or args.shards < 1:
        parser.error('hours and shards must be positive')
    if args.variant is None:
        args.variant = read(args.run/'config.json')['learner']['variant']
    result, path = collect(**vars(args))
    print(json.dumps(dict(snapshot=str(path), checkpoint=result['checkpoints'][-1]['id'],
                          learner_stage=(result['learner_status'] or {}).get('stage'),
                          actor_stage=(result['actor_status'] or {}).get('stage'),
                          since_baseline=result.get('since_baseline'))))


if __name__ == '__main__':
    main()
