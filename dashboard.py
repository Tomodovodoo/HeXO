"""Read-only local training dashboard for one run directory."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import hashlib
import os
from pathlib import Path
import subprocess
import time
from collections import deque
from functools import lru_cache


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


_report_digests={}


def report_digest(path,modified,size):
    cached=_report_digests.get(path)
    if cached is None or cached[:2]!=(modified,size):
        cached=(modified,size,hashlib.sha256(Path(path).read_bytes()).hexdigest())
        _report_digests[path]=cached
    return cached[2]


def background_results(run, league):
    """Overlay separately published estimates without changing trainer-owned state."""
    background=read_json(run/'background-league.json',{})
    config_hash=hashlib.sha256((run/'config.json').read_bytes()).hexdigest()
    entries={c['id']:c for c in league.get('checkpoints',[])}
    hashes={n:read_json(run/'checkpoints'/f'{n:04d}'/'manifest.json',{}).get('files',{}).get('model.pt') for n in entries}
    if background.get('config_sha256')==config_hash:
        for record in background['checkpoints']:
            number=record['id']
            if number in entries and record.get('model_sha256')==hashes[number]:
                entry=entries[number]
                entry.update(record)
                # An opponent can acquire a rating before any of its own matches.
                if entry.get('evaluation_due') is False:
                    entry['provisional']=record.get('provisional',True)
        league['background_note']=background['note']
    joint=read_json(run/'paired-ratings.json',{})
    if joint.get('config_sha256')==config_hash:
        paths=list((run/'evaluation').glob('*-vs-*/report.json'))+list((run/'background-evaluation').glob('*-vs-*.json'))
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
        league['rating_method']=joint['rating_method']
        league['joint_rating_note']=joint['note']
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


class Handler(BaseHTTPRequestHandler):
    run = Path("runs/selfplay")
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

    def do_GET(self):
        if self.path == "/":
            payload = (Path(__file__).parent / "web/training.html").read_bytes()
            content_type = "text/html; charset=utf-8"
        elif self.path == "/openings.js":
            payload = (Path(__file__).parent / "web/openings.js").read_bytes()
            content_type = "text/javascript; charset=utf-8"
        elif self.path == "/api/run":
            summary = self.run / "summary.json"
            events = self.run / "events.jsonl"
            search_config = read_json(self.run/'config.json', {})
            relational = relational_run(self.run, self.model_family)
            if search_config.get('backbone') == 'hexo-relational-policy-value-v1':
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
                league = read_json(self.run/'league.json', {})
                checkpoint_ids = {c['id'] for c in league.get('checkpoints', [])}
                for path in sorted((self.run/'checkpoints').glob('[0-9][0-9][0-9][0-9]/manifest.json')):
                    number = int(path.parent.name)
                    if number not in checkpoint_ids:
                        manifest = read_json(path, {})
                        league.setdefault('checkpoints', []).append(dict(id=number, elo=None,
                            elo_interval=None, promoted=False, pending=True, loss=manifest.get('metrics')))
                background_status=background_results(self.run,league)
                data = dict(kind='search', search=dict(name=self.run.name,background_status=background_status,
                    config=search_config, status=read_json(self.run/'status.json', {}),
                    league=league, openings=[opening
                        for path in sorted((self.run/'evaluation').glob('*-vs-*/report.json'), reverse=True)[:16]
                        for opening in evaluation_openings(str(path), path.stat().st_mtime_ns)]), events=recent[-300:])
            elif relational:
                data = {"kind": "relational", "relational": relational, "summary": None, "events": []}
            elif not summary.exists():
                klent = klent_run(self.run)
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
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--model-family", choices=['relational-policy-q'], help='Identify a run before its first manifest is published')
    args = parser.parse_args()
    Handler.run = Path(args.run).resolve()
    Handler.model_family = args.model_family
    print(f"Training dashboard: http://127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
