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


@lru_cache(maxsize=32)
def episode_counts(path, modified):
    episodes = read_json(Path(path), [])
    return {"games": len(episodes), "terminal_games": sum(e["winner"] >= 0 for e in episodes),
            "bootstrapped_games": sum(e["winner"] < 0 for e in episodes),
            "positions": sum(len(e["moves"]) for e in episodes)}


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
    status_path = run/"status.json"
    if not status_path.exists():
        status_path = run.with_suffix('.status.json')
    status = read_json(status_path, {})
    manifests = sorted((run/'checkpoints').glob('[0-9][0-9][0-9][0-9]/manifest.json'))
    manifest_path = manifests[-1] if manifests else run/'manifest.json'
    manifest = read_json(manifest_path, {})
    identity = manifest.get('identity', {})
    provenance = read_json(run/'provenance.json', {})
    family = identity.get('backbone') or identity.get('kind') or status.get('schema') or provenance.get('model_family') or declared_family or ''
    if 'relational' not in family:
        return None
    evaluation = status if provenance.get('schema') == 'hexo-relational-evaluation-v1' else None
    model_hash = manifest.get('files', {}).get('model.pt') or status.get('candidate_sha256')
    if not evaluation and model_hash:
        for folder in (run/'evaluation/confirmation', run/'evaluation'):
            candidate = read_json(folder/'status.json', {})
            if candidate.get('candidate_sha256') == model_hash:
                evaluation = candidate
                break
    warmstart = identity.get('kind') == 'relational-human-policy-q-v1' or 'epoch' in status or 'epochs' in status
    backend = (evaluation or {}).get('backend')
    artifact = manifest_path if manifest else None
    if evaluation is status and (run/'report.json').exists():
        artifact = run/'report.json'
    return dict(name=run.name, path=str(run), model_family='relational-policy-q',
        phase=status.get('stage', 'initialized'), training_backend='human policy/Q fitting' if warmstart else 'KLENT policy/Q',
        evaluation_backend=backend, checkpoint_sha256=model_hash,
        source_sha256=identity.get('sources') or provenance.get('files_sha256'),
        opponent=(evaluation or {}).get('opponent'), evaluation=evaluation,
        heartbeat=status_path.stat().st_mtime if status_path.exists() else None,
        workers=status.get('workers', []), last_artifact=str(artifact) if artifact else None,
        status=status, metrics=manifest.get('metrics', {}), config=identity.get('config', {}),
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
        elif self.path == "/api/run":
            summary = self.run / "summary.json"
            events = self.run / "events.jsonl"
            relational = relational_run(self.run, self.model_family)
            if relational:
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
