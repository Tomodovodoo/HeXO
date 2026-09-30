"""Offline proof pass over a dense run: proven windows per shard and the restart buffer (run layout: dense_config).

  python python/dense_solve.py --run runs/dense-v1 [--out DIR] [--limit N] [--once] [--<setting> ...]

A coordinator process at BelowNormal priority runs shard workers: each is this script with --shard, solving one
shard with its own tactical worker (tactical_proof.IsolatedTactics, one query at a time) and handing the result back
in out/.solve/<name>.json; the coordinator alone writes the buffer and the sidecars. Shards are read and never
changed; everything is written under `--out` (default: the run): the sidecar shards/<name>/proofs.jsonl,
restarts.json, solver-status.json, events and .solve/ (worker results and logs). A shard is done once its sidecar
exists, so a restarted pass resumes where it stopped. Pending shards are taken newest first: the pass starts at the
newest shards, keeps up with new ones and back-fills older ones while it has nothing newer. A worker that exits with
REJECTED stops the pass; one that fails in any other way logs an error event and its shard is retried once, when no
shard on its first attempt is pending or running (status shards_failed: {name: failed attempts}); after a second
failure the coordinator skips it.

Workers (Pass.workers). While the main learner's fresh heartbeat (learner-status.json, at most STALE_SECONDS old)
shows a phased learner (phase_rows > 0) in its training phase (stage 'training' or 'exporting'), the actors are
paused and the pass runs solve_workers_max shard workers (0: physical cores minus 2); otherwise it runs
solve_workers_min, so the actors keep the CPU. Workers beyond a lowered target are stopped at once; their shards
stay pending. Every change of the target is recorded in solver-status.json (workers, workers_reason,
worker_switches) and as an event.

Positions and queries. A game's positions are plies start..T-1 (the position before placement t; start is the
restart ply of a restart game, else 0). attack(t, nodes) asks whether the side to move at t has a forced win,
threat(t, nodes) whether its opponent would have one moving now with a fresh turn (attacker 'opponent'). Budgets are
node counts; SAFETY_MS is only a wall-clock cap. Only a native-verified PROVEN_WIN is a proof. A PROVEN_WIN without
native verification, or an UNKNOWN whose reason is not a search verdict (dense_solver.VERDICTS), counts as a
failure (status `failures`) and as no proof. A deterministic `verify_fraction` of proofs (by shard, game, ply,
attacker and budget) is checked again by tactical_proof.independent_verify within `verify_seconds`: a rejection logs
an error event and ends the pass (the worker exits with REJECTED); a check that runs out of time counts in
`verify_timeouts` and the proof, already accepted by the native verifier, is kept.

Per game:
  gate + solve  each turn start (odd t) whose mover passes forcing_material.worth_solving gets attack(t, solve_nodes);
  windows       a hit at turn start t for mover m outside every window opens one: m's earlier turn starts (t-4,
                t-8, ...) get attack(., scan_nodes) until one fails; m's later turn starts (t+4, t+8, ...) get
                attack(., solve_nodes), then attack(., scan_nodes) when that fails, until one fails or the game
                ends. The window is that run of m's proven turn starts, from first_ply to its last turn start; its
                `plies` are those turn starts and each following mid-turn t+1 (before the game's end) proven by
                attack(t+1, solve_nodes); last_ply is the latest of them. A window is persistent when m won the game
                and it reaches m's last turn start, else transient (the forced win was lost).
  lookback      for each window of the game's winner, the loser's `lookback_turns` turn starts before the window's
                first ply (latest first, at or after start) get threat(d, scan_nodes). When it is proven,
                `saving_turns` lists the candidate turns of the loser at d after which attack(., saving_nodes) for
                the winner is a search verdict UNKNOWN (a failed query does not count): the unordered pairs of cells
                empty at d among the threat's first turn and the first turn of the window's opening proof.
Sidecar: one JSON line per window {game, first_ply, last_ply, mover, plies, proof_turns,
budget (nodes of the first ply's proof), certificate_hash (sha256 of its certificate JSON),
proof_action (mapping from each proven ply string to that proof's first-turn placements),
search_value_at_first_ply (recorded root value or null), persistent, defence: [{ply, threat, saving_turns}]}.
Each transient window whose owner lost also adds {kind: 'deblunder', game, first_ply, owner}. The learner
uses these only with a positive --deblunder-weight; existing sidecars stay unchanged on restart.

Restart buffer (RestartBuffer). Entries {shard, game, ply, side_to_move, regret, kind, added_at, checkpoint,
plies_to_proof, saving_turns?, observed?}: an 'attack' entry at each window's first ply with regret (1 - v)/2, and
a 'defence' entry at each lookback ply d with regret (1 + v)/2 and the saving_turns found there, v the recorded
root value of the side to move at that ply (no entry where it is null); plies_to_proof is first_ply minus the
entry's ply. Restart games add no entries at or before their restart ply; instead their recorded root value at the
restart ply is stored in the source entry's `observed` {checkpoint: {value, shard}} when the game was played by the
shard's checkpoint, one observation per checkpoint, the one from the newest shard; an observation of an entry not
kept yet waits in `waiting` (saved with the buffer) until the entry is added, and is dropped at a refresh once the
source shard has a sidecar without it. Entries with regret below
min_regret are never kept; beyond buffer_size the lowest regrets go. Every refresh_minutes the buffer takes regret
from the observation of the newest complete checkpoint of the learner variant where it has one (checkpoint becomes
it), forgets the observations of other checkpoints, and drops entries added more than buffer_max_exports exports of
that variant ago or whose regret fell below min_regret. No net is evaluated.

solver-status.json: pass counters (positions: turn starts scanned; gated: those passing the gate; hits: forward
proofs; windows, persistent, transient, window_plies histogram of last_ply - first_ply + 1; per query kind
queries, proofs and ms; verified, verify_timeouts, failures, last_failure), their per-hour rates over the coordinator's lifetime,
gate_pass_rate, busy_cores (shard-worker seconds over elapsed seconds), shards_done, shards_pending, shards_running,
shards_failed, workers, workers_reason, worker_switches (the last SWITCHES {time, workers, reason}), buffer_size and
buffer_mean_regret.
"""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass, fields
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import dense_config
import dense_data
from dense_data import player_at
from dense_solver import VERDICTS
from forcing_material import worth_solving
from hexo import Game
from proof import VerificationTimeout
from tactical_proof import PROVEN_WIN, IsolatedTactics, independent_verify
from legacy.train import write_json

SAFETY_MS = 60000
KINDS = ('solve', 'scan', 'threat', 'saving')
FAILED = ()  # query result of a failure: no proof, and not a search verdict UNKNOWN (None)
STALE_SECONDS = 120.  # older learner heartbeats are ignored by Pass.workers
SWITCHES = 20
STATUS_SECONDS = 5.
REJECTED = 17  # exit code of a shard worker whose proof the independent check rejected
ATTEMPTS = 2   # worker runs per shard before the coordinator skips it


class Rejected(RuntimeError):
    """The independent check rejected a native-verified proof."""


@dataclass(frozen=True)
class PassSettings:
    solve_nodes: int = 540         # forward attack query per gated turn start (about 20 ms)
    scan_nodes: int = 27000        # window extension and threat queries (about 1 s)
    saving_nodes: int = 2700       # attack query after each candidate saving turn
    lookback_turns: int = 2        # the loser's turn starts before a window that get defence entries
    verify_fraction: float = .1    # proofs checked again by independent_verify
    verify_seconds: float = 120.   # time limit of one independent check
    buffer_size: int = 20000
    buffer_max_exports: int = 2
    min_regret: float = .1
    refresh_minutes: float = 30.
    poll_seconds: float = 30.
    solve_workers_min: int = 1     # shard workers while the actors play
    solve_workers_max: int = 0     # shard workers during a phased learner's training phase; 0: physical cores - 2


def below_normal():
    """Lower this process (and the processes it starts later) to BelowNormal priority."""
    if sys.platform == 'win32':
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x4000)
    else:
        os.nice(5)


def physical_cores():
    """Physical processor cores (Windows: GetLogicalProcessorInformation core records; elsewhere os.cpu_count())."""
    if sys.platform != 'win32':
        return os.cpu_count() or 1
    import ctypes
    size = ctypes.c_uint32(0)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetLogicalProcessorInformation(None, ctypes.byref(size))
    buffer = ctypes.create_string_buffer(size.value)
    if not kernel32.GetLogicalProcessorInformation(buffer, ctypes.byref(size)):
        return os.cpu_count() or 1
    record = 32 if ctypes.sizeof(ctypes.c_void_p) == 8 else 24  # SYSTEM_LOGICAL_PROCESSOR_INFORMATION
    offset = ctypes.sizeof(ctypes.c_void_p)                     # Relationship follows ProcessorMask
    cores = sum(int.from_bytes(buffer.raw[k+offset:k+offset+4], 'little') == 0 for k in range(0, size.value, record))
    return cores or os.cpu_count() or 1


def regret(kind, value):
    """Regret of a recorded side-to-move value at an entry: (1 - v)/2 for 'attack', (1 + v)/2 for 'defence'."""
    return (1-value)/2 if kind == 'attack' else (1+value)/2


def write_sidecar(path, windows):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    temporary = path/(dense_data.SIDECAR+'.tmp')
    temporary.write_text(''.join(json.dumps(w, separators=(',', ':'))+'\n' for w in windows), encoding='utf-8')
    os.replace(temporary, path/dense_data.SIDECAR)


def exports(run, variant):
    """created_at of the complete checkpoints of `variant`, oldest first, with the newest checkpoint id."""
    found = []
    for path in (Path(run)/'checkpoints'/variant).glob('*/manifest.json'):
        found.append((json.loads(path.read_text(encoding='utf-8'))['created_at'], f'{variant}/{path.parent.name}'))
    found.sort()
    return [t for t, _ in found], found[-1][1] if found else None


def new_stats():
    return dict(positions=0, gated=0, hits=0, windows=0, persistent=0, transient=0, verified=0, verify_timeouts=0, failures=0,
                last_failure=None, window_plies=Counter(), queries={k: dict(queries=0, proofs=0, ms=0.) for k in KINDS})


def merge(total, part):
    """Add the counters of `part` (new_stats layout; window_plies keys may be strings after JSON) into `total`."""
    for key, value in part.items():
        if key == 'window_plies':
            total[key].update({int(k): v for k, v in value.items()})
        elif key == 'queries':
            for kind, counts in value.items():
                for name, v in counts.items():
                    total[key][kind][name] += v
        elif key == 'last_failure':
            total[key] = value or total[key]
        else:
            total[key] += value


class RestartBuffer:
    """restarts.json {updated_at, entries, waiting} (module contract), held as {key: entry} with key (shard, game,
    ply, kind) and waiting observations {key: {checkpoint: {value, shard}}}."""

    def __init__(self, path, size, max_exports, min_regret):
        self.path, self.size, self.max_exports, self.min_regret = Path(path), size, max_exports, min_regret
        data = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else dict(entries=[])
        self.entries = {self.key(e): e for e in data['entries']}
        self.waiting = {tuple(key): observed for key, observed in data.get('waiting', [])}
        self.trim()

    @staticmethod
    def key(entry):
        return entry['shard'], entry['game'], entry['ply'], entry['kind']

    def add(self, entry):
        """Keep `entry` unless its regret is below min_regret or its key is already kept (the kept entry, with its
        observations, stays)."""
        key = self.key(entry)
        if entry['regret'] >= self.min_regret and key not in self.entries:
            self.entries[key] = entry
            for checkpoint, seen in self.waiting.pop(key, {}).items():
                self.observe(key, checkpoint, seen['value'], seen['shard'])
            self.trim()

    def trim(self):
        if len(self.entries) > self.size:
            kept = sorted(self.entries.values(), key=lambda e: -e['regret'])[:self.size]
            self.entries = {self.key(e): e for e in kept}

    def observe(self, key, checkpoint, value, shard):
        """Record a restart game's value at entry `key` for `checkpoint`, unless one from a newer shard is kept; for
        an entry not kept yet, in `waiting`."""
        entry = self.entries.get(key)
        observed = entry.setdefault('observed', {}) if entry else self.waiting.setdefault(key, {})
        if checkpoint not in observed or observed[checkpoint]['shard'] <= shard:
            observed[checkpoint] = dict(value=value, shard=shard)

    def refresh(self, created, newest, solved=lambda shard: False):
        """Apply observations of checkpoint `newest`, drop entries by age (`created`: export times) and regret, and
        drop waiting observations whose source shard is `solved`."""
        self.waiting = {key: observed for key, observed in self.waiting.items() if not solved(key[0])}
        for key, e in list(self.entries.items()):
            seen = e.get('observed', {}).get(newest)
            if seen:
                e.update(regret=regret(e['kind'], seen['value']), checkpoint=newest, observed={newest: seen})
            else:
                e.pop('observed', None)
            if sum(t > e['added_at'] for t in created) > self.max_exports or e['regret'] < self.min_regret:
                del self.entries[key]

    def save(self):
        entries = sorted(self.entries.values(), key=lambda e: (-e['regret'], self.key(e)))
        write_json(self.path, dict(updated_at=time.time(), entries=entries,
                                   waiting=[[list(key), observed] for key, observed in sorted(self.waiting.items())]))

    def summary(self):
        values = [e['regret'] for e in self.entries.values()]
        return dict(buffer_size=len(values), buffer_mean_regret=sum(values)/len(values) if values else None)


class Solver:
    """Solves one shard of `run` (module contract, per game): `tactics` answers history(history, nodes=, ms=,
    attacker=) like tactical_proof.NativeTactics or IsolatedTactics; `out` receives independent-check events."""

    def __init__(self, run, out, settings, tactics, clock=time.time):
        self.run, self.out, self.s, self.tactics, self.clock = Path(run), Path(out), settings, tactics, clock
        self.stats = new_stats()

    def query(self, where, history, attacker, nodes, kind):
        """(moves, proof_turns, certificate hash) of a native-verified proof, None for a search verdict UNKNOWN, else
        FAILED (module contract)."""
        result = self.tactics.history([list(p) for p in history], nodes=nodes, ms=SAFETY_MS, attacker=attacker)
        stats = self.stats['queries'][kind]
        stats['queries'] += 1
        stats['ms'] += float(result.get('elapsed_ms') or 0.)
        if result['status'] != PROVEN_WIN or not result.get('native_verified'):
            if result['status'] == PROVEN_WIN or result.get('reason') not in VERDICTS:
                self.stats['failures'] += 1
                self.stats['last_failure'] = result.get('reason')
                return FAILED
            return None
        stats['proofs'] += 1
        text = result.get('certificate_json') or json.dumps(result['certificate'], separators=(',', ':'))
        tag = json.dumps([*where, attacker, nodes]).encode()
        if int.from_bytes(hashlib.sha256(tag).digest()[:8], 'little') < self.s.verify_fraction*2**64:
            try:
                independent_verify(json.loads(text), history, attacker, self.s.verify_seconds)
                self.stats['verified'] += 1
            except VerificationTimeout:
                self.stats['verify_timeouts'] += 1
            except ValueError as error:
                dense_config.log_event(self.out, 'solve', 'error', f'independent check rejected a proof at {where} '
                                       f'({attacker}, {nodes} nodes): {error}', where=list(where), attacker=attacker)
                raise Rejected(f'Independent check rejected a native proof at {where}') from error
        return [tuple(m) for m in result['moves']], result['proof_turns'], hashlib.sha256(text.encode()).hexdigest()

    def game(self, shard, g, e):
        """(window records, buffer entries) of episode `g` of `shard`."""
        s, moves, T = self.s, e['moves'], len(e['moves'])
        start = e['restart']['ply'] if e.get('origin') == 'restart' else 0
        roots = e.get('root_values') or [None]*T
        memo = {}

        def attack(t, nodes):
            if (t, nodes) not in memo:
                memo[t, nodes] = self.query((shard, g, t), moves[:t], 'mover', nodes, 'solve' if nodes == s.solve_nodes else 'scan')
            return memo[t, nodes]

        hits, game = [], Game()
        for t in range(T):
            if t % 2 and t >= start:
                self.stats['positions'] += 1
                if worth_solving(game):
                    self.stats['gated'] += 1
                    if attack(t, s.solve_nodes):
                        hits.append(t)
            game.play(*moves[t])
        game.close()
        self.stats['hits'] += len(hits)
        windows, entries, covered = [], [], set()
        for hit in hits:
            if hit in covered:
                continue
            m = player_at(hit)
            starts = [t for t in range(start, T) if t % 2 and player_at(t) == m]
            first = last = starts.index(hit)
            while first > 0 and attack(starts[first-1], s.scan_nodes):
                first -= 1
            while last+1 < len(starts) and (attack(starts[last+1], s.solve_nodes) or attack(starts[last+1], s.scan_nodes)):
                last += 1
            turns = starts[first:last+1]
            covered.update(turns)
            plies = sorted(turns+[t+1 for t in turns if t+1 < T and attack(t+1, s.solve_nodes)])
            budget = s.solve_nodes if memo.get((plies[0], s.solve_nodes)) else s.scan_nodes
            opening = memo[plies[0], budget]
            actions = {str(t): [list(a) for a in (memo.get((t, s.solve_nodes)) or memo[t, s.scan_nodes])[0]] for t in plies}
            persistent = e['winner'] == m and turns[-1] == starts[-1]
            defence = self.lookback(shard, g, moves, plies[0], m, opening[0], start) if e['winner'] == m else []
            windows.append(dict(game=g, first_ply=plies[0], last_ply=plies[-1], mover=m, plies=plies, proof_turns=opening[1],
                                budget=budget, certificate_hash=opening[2], search_value_at_first_ply=roots[plies[0]],
                                persistent=persistent, defence=defence, proof_action=actions))
            self.stats['windows'] += 1
            self.stats['persistent' if persistent else 'transient'] += 1
            self.stats['window_plies'][plies[-1]-plies[0]+1] += 1
            candidates = [(plies[0], m, 'attack', None)] + [(d['ply'], 1-m, 'defence', d['saving_turns']) for d in defence]
            for ply, side, kind, saving in candidates:
                if roots[ply] is None or (start and ply <= start):
                    continue
                entry = dict(shard=shard, game=g, ply=ply, side_to_move=side, regret=regret(kind, roots[ply]), kind=kind,
                             added_at=self.clock(), checkpoint=None, plies_to_proof=plies[0]-ply)
                entries.append(entry if kind == 'attack' else dict(entry, saving_turns=saving))
        return windows, entries

    def lookback(self, shard, g, moves, first, winner, opening, start):
        """[{ply, threat, saving_turns}] at the loser's lookback turn starts before ply `first`, latest first."""
        latest = max((t for t in range(1, first, 2) if player_at(t) != winner), default=None)
        found = []
        for d in range(latest, max(start, 1)-1, -4) if latest is not None else ():
            if len(found) == self.s.lookback_turns:
                break
            history = moves[:d]
            threat = self.query((shard, g, d), history, 'opponent', self.s.scan_nodes, 'threat')
            saving = None
            if threat:
                occupied = {tuple(m) for m in history}
                cells = [c for c in dict.fromkeys(threat[0]+opening) if c not in occupied]
                saving = [[list(a), list(b)] for a, b in itertools.combinations(cells, 2)
                          if self.query((shard, g, d, a, b), history+[a, b], 'mover', self.s.saving_nodes, 'saving') is None]
            found.append(dict(ply=d, threat=bool(threat), saving_turns=saving))
        return found

    def solve(self, name):
        """The result of shard `name`: {windows, entries (checkpoint: the shard's), observations [(key, checkpoint,
        value)] of its restart games, stats (this shard's counters)}."""
        self.stats = new_stats()
        path = self.run/'shards'/name
        manifest = dense_data.verify(path)
        episodes = json.loads((path/'episodes.json').read_text(encoding='utf-8'))
        identity = manifest['identity']
        windows, entries, observations = [], [], []
        for g, e in enumerate(episodes):
            found, new = self.game(name, g, e)
            windows += found
            entries += [dict(entry, checkpoint=identity.get('checkpoint')) for entry in new]
            source = e.get('restart') if e.get('origin') == 'restart' else None
            if source and e['actor'] == identity.get('actor_sha256') and e['root_values'][source['ply']] is not None:
                key = [source['shard'], source['game'], source['ply'], source['kind']]
                observations.append((key, identity.get('checkpoint'), e['root_values'][source['ply']]))
        deblunders = [dict(kind='deblunder', game=w['game'], first_ply=w['first_ply'], owner=w['mover'])
                      for w in windows if not w['persistent'] and episodes[w['game']]['winner'] == 1-w['mover']]
        return dict(windows=windows, deblunders=deblunders, entries=entries, observations=observations, stats=self.stats)


class Pass:
    """The coordinator of the proof pass over the shards of `run`, writing under `out` (module contract)."""

    def __init__(self, run, out, settings, clock=time.time):
        self.run, self.out, self.s, self.clock = Path(run), Path(out), settings, clock
        self.out.mkdir(parents=True, exist_ok=True)
        self.variant = dense_config.load(self.run).learner.variant
        self.buffer = RestartBuffer(self.out/'restarts.json', settings.buffer_size, settings.buffer_max_exports,
                                    settings.min_regret)
        self.started = self.refreshed = self.reported = clock()
        self.busy, self.running, self.target, self.switches, self.failed = 0., {}, None, [], Counter()
        self.stats = dict(new_stats(), shards_done=0)

    def record(self, name, result):
        """Add a shard's result to the buffer and the counters, then save the buffer and write the sidecar."""
        for entry in result['entries']:
            self.buffer.add(entry)
        for key, checkpoint, value in result['observations']:
            self.buffer.observe(tuple(key), checkpoint, value, name)
        merge(self.stats, result['stats'])
        self.stats['shards_done'] += 1
        self.buffer.save()
        write_sidecar(self.out/'shards'/name, result['windows']+result.get('deblunders', []))

    def refresh(self):
        """RestartBuffer.refresh against the variant's exports, then save the buffer."""
        created, newest = exports(self.run, self.variant)
        self.buffer.refresh(created, newest, lambda shard: (self.out/'shards'/shard/dense_data.SIDECAR).exists())
        self.buffer.save()
        self.refreshed = self.clock()

    def pending(self, limit=None):
        """Shard names without a sidecar under `out` among the newest `limit` shards, newest first, shards whose
        worker failed after the others and without those that failed ATTEMPTS times."""
        names = [p.name for p in dense_data.shard_dirs(self.run)][::-1][:limit]
        names = [n for n in names if self.failed[n] < ATTEMPTS and not (self.out/'shards'/n/dense_data.SIDECAR).exists()]
        return sorted(names, key=lambda n: self.failed[n])

    def workers(self):
        """(shard workers, reason) from the main learner's heartbeat (module contract)."""
        try:
            status = json.loads((self.run/'learner-status.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            status = {}
        fresh = self.clock()-float(status.get('updated_at') or 0) <= STALE_SECONDS
        if fresh and (status.get('phase_rows') or 0) > 0 and status.get('stage') in ('training', 'exporting'):
            return self.s.solve_workers_max or max(1, physical_cores()-2), 'learner training phase: actors paused'
        return self.s.solve_workers_min, f"learner {status.get('stage') if fresh else 'heartbeat absent or stale'}"

    def spawn(self, name):
        """Start a shard worker process for `name`; its result goes to out/.solve/<name>.json."""
        (self.out/'.solve').mkdir(exist_ok=True)
        flags = [x for f in fields(self.s) for x in ('--'+f.name.replace('_', '-'), str(getattr(self.s, f.name)))]
        log = (self.out/'.solve'/f'{name}.log').open('w')
        try:
            return subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--run', str(self.run), '--out',
                                     str(self.out), '--shard', name, *flags], stdout=log, stderr=subprocess.STDOUT)
        finally:
            log.close()

    def status(self, stage, pending):
        seconds = max(1e-9, self.clock()-self.started)
        q = self.stats
        write_json(self.out/'solver-status.json', dict(
            stage=stage, updated_at=self.clock(), started_at=self.started, shards_pending=pending,
            shards_running=sorted(self.running), shards_failed=dict(sorted(self.failed.items())), workers=self.target[0] if self.target else None,
            workers_reason=self.target[1] if self.target else None, worker_switches=self.switches,
            **{k: v for k, v in q.items() if k != 'window_plies'}, window_plies=dict(sorted(q['window_plies'].items())),
            positions_per_hour=q['positions']*3600/seconds, hits_per_hour=q['hits']*3600/seconds,
            gate_pass_rate=q['gated']/q['positions'] if q['positions'] else None,
            busy_cores=self.busy/seconds, **self.buffer.summary(), settings=asdict(self.s)))
        self.reported = self.clock()

    def step(self, limit=None, spawn=None):
        """One coordinator round: collect finished workers, adjust the worker count, start workers on pending shards
        newest first and report; returns True while work is pending or running. Raises RuntimeError when a worker
        exited with REJECTED; any other failure of a worker is logged and counted in `failed`."""
        spawn = spawn or self.spawn
        for name, (process, began) in list(self.running.items()):
            code = process.poll()
            if code is None:
                continue
            del self.running[name]
            self.busy += self.clock()-began
            log = self.out/'.solve'/f'{name}.log'
            if code == REJECTED:
                raise RuntimeError(f'independent check rejected a proof in shard {name}; see {log}')
            path = self.out/'.solve'/f'{name}.json'
            try:
                if code:
                    raise RuntimeError(f'exited with code {code}')
                result = json.loads(path.read_text(encoding='utf-8'))
            except (RuntimeError, OSError, ValueError) as error:
                self.failed[name] += 1
                retry = 'retried later' if self.failed[name] < ATTEMPTS else 'skipped'
                dense_config.log_event(self.out, 'solve', 'error', f'shard worker for {name} failed ({error}); '
                                       f'{retry}; see {log}', shard=name, attempts=self.failed[name])
                continue
            self.record(name, result)
            path.unlink()
        target = self.workers()
        switched = self.target is None or target[0] != self.target[0]
        if switched:
            self.switches = (self.switches+[dict(time=self.clock(), workers=target[0], reason=target[1])])[-SWITCHES:]
            dense_config.log_event(self.out, 'solve', 'info', f'{target[0]} shard workers: {target[1]}', workers=target[0])
        self.target = target
        for name in sorted(self.running, key=lambda n: self.running[n][1])[target[0]:]:
            process, began = self.running.pop(name)
            process.kill()
            process.wait()
            self.busy += self.clock()-began
        pending = [n for n in self.pending(limit) if n not in self.running]
        first = any(not self.failed[n] for n in [*pending, *self.running])
        startable = [n for n in pending if not (first and self.failed[n])]
        while startable and len(self.running) < target[0]:
            name = startable.pop(0)
            pending.remove(name)
            self.running[name] = (spawn(name), self.clock())
        if switched or self.clock()-self.reported >= STATUS_SECONDS or not self.running:
            self.status('solving' if self.running else 'idle', len(pending))
        return bool(self.running or pending)

    def loop(self, limit=None, once=False, sleep=time.sleep):
        """Refresh the buffer, then run step() every second while there is work and every poll_seconds otherwise,
        refreshing every refresh_minutes; with `once`, refresh again and return once nothing is pending or running.
        Workers still running when it ends are stopped."""
        self.refresh()
        try:
            while True:
                if self.clock()-self.refreshed >= self.s.refresh_minutes*60:
                    self.refresh()
                if self.step(limit):
                    sleep(1.)
                elif once:
                    self.refresh()
                    return
                else:
                    sleep(self.s.poll_seconds)
        finally:
            for process, _ in self.running.values():
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', help='directory for sidecars, buffer, status and events (default: the run)')
    parser.add_argument('--limit', type=int, help='consider only the newest N shards')
    parser.add_argument('--once', action='store_true', help='exit when no shard is pending')
    parser.add_argument('--shard', help=argparse.SUPPRESS)
    dense_config.add_arguments(parser, PassSettings)
    args = parser.parse_args()
    settings = dense_config.override(PassSettings(), args)
    below_normal()
    out = Path(args.out or args.run)
    if args.shard:
        tactics = IsolatedTactics()
        try:
            result = Solver(args.run, out, settings, tactics).solve(args.shard)
        except Rejected:
            traceback.print_exc()
            sys.exit(REJECTED)
        finally:
            tactics.close()
        write_json(out/'.solve'/f'{args.shard}.json', result)
        return
    try:
        solver = Pass(args.run, out, settings)
        dense_config.log_event(out, 'solve', 'info', 'proof pass started', settings=asdict(settings))
        solver.loop(args.limit, args.once)
    except Exception as error:
        dense_config.log_event(out, 'solve', 'error', f'proof pass stopped: {error!r}')
        raise


if __name__ == '__main__':
    main()
