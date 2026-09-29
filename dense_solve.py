"""Offline proof pass over a dense run: proven windows per shard and the restart buffer (run layout: dense_config).

  python dense_solve.py --run runs/dense-v1 [--out DIR] [--limit N] [--once] [--<setting> ...]

One process at BelowNormal priority with one tactical worker (tactical_proof.IsolatedTactics), one query at a time.
Shards are read and never changed; everything is written under `--out` (default: the run): the sidecar
shards/<name>/proofs.jsonl, restarts.json, solver-status.json and events. A shard is done once its sidecar exists,
so a restarted pass resumes where it stopped. The pass always takes the newest shard without a sidecar: it starts
at the newest shards, keeps up with new ones and back-fills older ones while it has nothing newer.

Positions and queries. A game's positions are plies start..T-1 (the position before placement t; start is the
restart ply of a restart game, else 0). attack(t, nodes) asks whether the side to move at t has a forced win,
threat(t, nodes) whether its opponent would have one moving now with a fresh turn (attacker 'opponent'). Budgets are
node counts; SAFETY_MS is only a wall-clock cap. Only a native-verified PROVEN_WIN is a proof. A PROVEN_WIN without
native verification, or an UNKNOWN whose reason is not the search's own, counts as a failure (status `failures`)
and as no proof. A deterministic `verify_fraction` of proofs (by shard, game, ply, attacker and budget) is checked
again by tactical_proof.independent_verify; a rejection logs an error event and ends the process.

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
                the winner is UNKNOWN: the unordered pairs of cells empty at d among the threat's first turn and the
                first turn of the window's opening proof.
Sidecar: one JSON line per window {game, first_ply, last_ply, mover, plies, proof_turns,
budget (nodes of the first ply's proof), certificate_hash (sha256 of its certificate JSON),
search_value_at_first_ply (recorded root value or null), persistent, defence: [{ply, threat, saving_turns}]}.

Restart buffer (RestartBuffer). Entries {shard, game, ply, side_to_move, regret, kind, added_at, checkpoint,
plies_to_proof, saving_turns?, observed?}: an 'attack' entry at each window's first ply with regret (1 - v)/2, and
a 'defence' entry at each lookback ply d with regret (1 + v)/2 and the saving_turns found there, v the recorded
root value of the side to move at that ply (no entry where it is null); plies_to_proof is first_ply minus the
entry's ply. Restart games add no entries at or before their restart ply; instead their recorded root value at the
restart ply is stored in the source entry's `observed` {checkpoint: {value, shard}} when the game was played by the
shard's checkpoint, one observation per checkpoint, the one from the newest shard. Entries with regret below
min_regret are never kept; beyond buffer_size the lowest regrets go. Every refresh_minutes the buffer takes regret
from the observation of the newest complete checkpoint of the learner variant where it has one (checkpoint becomes
it), forgets the observations of other checkpoints, and drops entries added more than buffer_max_exports exports of
that variant ago or whose regret fell below min_regret. No net is evaluated.

solver-status.json: pass counters (positions: turn starts scanned; gated: those passing the gate; hits: forward
proofs; windows, persistent, transient, window_plies histogram of last_ply - first_ply + 1; per query kind
queries, proofs and ms; verified, failures, last_failure), their per-hour rates over the process lifetime, gate_pass_rate,
busy_fraction (time spent on shards over elapsed time), shards_done, shards_pending, buffer_size and
buffer_mean_regret.
"""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

import dense_config
import dense_data
from dense_data import player_at
from dense_solver import SEARCHED
from forcing_material import worth_solving
from hexo import Game
from tactical_proof import PROVEN_WIN, IsolatedTactics, independent_verify
from train import write_json

SAFETY_MS = 60000
KINDS = ('solve', 'scan', 'threat', 'saving')


@dataclass(frozen=True)
class PassSettings:
    solve_nodes: int = 540         # forward attack query per gated turn start (about 20 ms)
    scan_nodes: int = 27000        # window extension and threat queries (about 1 s)
    saving_nodes: int = 2700       # attack query after each candidate saving turn
    lookback_turns: int = 2        # the loser's turn starts before a window that get defence entries
    verify_fraction: float = .1    # proofs checked again by independent_verify
    buffer_size: int = 20000
    buffer_max_exports: int = 2
    min_regret: float = .1
    refresh_minutes: float = 30.
    poll_seconds: float = 30.


def below_normal():
    """Lower this process (and the workers it starts later) to BelowNormal priority."""
    if sys.platform == 'win32':
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x4000)
    else:
        os.nice(5)


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


class RestartBuffer:
    """restarts.json (module contract), held as {key: entry} with key (shard, game, ply, kind)."""

    def __init__(self, path, size, max_exports, min_regret):
        self.path, self.size, self.max_exports, self.min_regret = Path(path), size, max_exports, min_regret
        data = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else dict(entries=[])
        self.entries = {self.key(e): e for e in data['entries']}

    @staticmethod
    def key(entry):
        return entry['shard'], entry['game'], entry['ply'], entry['kind']

    def add(self, entry):
        if entry['regret'] >= self.min_regret:
            self.entries[self.key(entry)] = entry
            self.trim()

    def trim(self):
        if len(self.entries) > self.size:
            kept = sorted(self.entries.values(), key=lambda e: -e['regret'])[:self.size]
            self.entries = {self.key(e): e for e in kept}

    def observe(self, key, checkpoint, value, shard):
        """Record a restart game's value at entry `key` for `checkpoint`, unless one from a newer shard is kept."""
        if key in self.entries:
            observed = self.entries[key].setdefault('observed', {})
            if checkpoint not in observed or observed[checkpoint]['shard'] <= shard:
                observed[checkpoint] = dict(value=value, shard=shard)

    def refresh(self, created, newest):
        """Apply observations of checkpoint `newest` and drop entries by age (`created`: export times) and regret."""
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
        write_json(self.path, dict(updated_at=time.time(), entries=entries))

    def summary(self):
        values = [e['regret'] for e in self.entries.values()]
        return dict(buffer_size=len(values), buffer_mean_regret=sum(values)/len(values) if values else None)


class Pass:
    """The proof pass over the shards of `run`, writing under `out` (module contract). `tactics` answers
    history(history, nodes=, ms=, attacker=) like tactical_proof.NativeTactics or IsolatedTactics."""

    def __init__(self, run, out, settings, tactics, clock=time.time):
        self.run, self.out, self.s, self.tactics, self.clock = Path(run), Path(out), settings, tactics, clock
        self.out.mkdir(parents=True, exist_ok=True)
        self.variant = dense_config.load(self.run).learner.variant
        self.buffer = RestartBuffer(self.out/'restarts.json', settings.buffer_size, settings.buffer_max_exports,
                                    settings.min_regret)
        self.started = self.refreshed = clock()
        self.busy = 0.
        self.stats = dict(shards_done=0, positions=0, gated=0, hits=0, windows=0, persistent=0, transient=0, verified=0,
                          failures=0, last_failure=None, window_plies=Counter(),
                          queries={k: dict(queries=0, proofs=0, ms=0.) for k in KINDS})

    def query(self, where, history, attacker, nodes, kind):
        """(moves, proof_turns, certificate hash) of a native-verified proof, else None (module contract)."""
        result = self.tactics.history([list(p) for p in history], nodes=nodes, ms=SAFETY_MS, attacker=attacker)
        stats = self.stats['queries'][kind]
        stats['queries'] += 1
        stats['ms'] += float(result.get('elapsed_ms') or 0.)
        if result['status'] != PROVEN_WIN or not result.get('native_verified'):
            if result['status'] == PROVEN_WIN or result.get('reason') != SEARCHED:
                self.stats['failures'] += 1
                self.stats['last_failure'] = result.get('reason')
            return None
        stats['proofs'] += 1
        text = result.get('certificate_json') or json.dumps(result['certificate'], separators=(',', ':'))
        tag = json.dumps([*where, attacker, nodes]).encode()
        if int.from_bytes(hashlib.sha256(tag).digest()[:8], 'little') < self.s.verify_fraction*2**64:
            try:
                independent_verify(json.loads(text), history, attacker)
            except ValueError as error:
                dense_config.log_event(self.out, 'solve', 'error', f'independent check rejected a proof at {where} '
                                       f'({attacker}, {nodes} nodes): {error}', where=list(where), attacker=attacker)
                raise RuntimeError(f'Independent check rejected a native proof at {where}') from error
            self.stats['verified'] += 1
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
            persistent = e['winner'] == m and turns[-1] == starts[-1]
            defence = self.lookback(shard, g, moves, plies[0], m, opening[0], start) if e['winner'] == m else []
            windows.append(dict(game=g, first_ply=plies[0], last_ply=plies[-1], mover=m, plies=plies, proof_turns=opening[1],
                                budget=budget, certificate_hash=opening[2], search_value_at_first_ply=roots[plies[0]],
                                persistent=persistent, defence=defence))
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
                          if not self.query((shard, g, d, a, b), history+[a, b], 'mover', self.s.saving_nodes, 'saving')]
            found.append(dict(ply=d, threat=bool(threat), saving_turns=saving))
        return found

    def shard(self, name):
        """Solve shard `name`: add its buffer entries and observations, then write its sidecar; returns its windows."""
        path = self.run/'shards'/name
        manifest = dense_data.verify(path)
        episodes = json.loads((path/'episodes.json').read_text(encoding='utf-8'))
        identity = manifest['identity']
        windows = []
        for g, e in enumerate(episodes):
            found, entries = self.game(name, g, e)
            windows += found
            for entry in entries:
                self.buffer.add(dict(entry, checkpoint=identity.get('checkpoint')))
            source = e.get('restart') if e.get('origin') == 'restart' else None
            if source and e['actor'] == identity.get('actor_sha256') and e['root_values'][source['ply']] is not None:
                key = (source['shard'], source['game'], source['ply'], source['kind'])
                self.buffer.observe(key, identity.get('checkpoint'), e['root_values'][source['ply']], name)
        self.buffer.save()
        write_sidecar(self.out/'shards'/name, windows)
        return windows

    def refresh(self):
        """RestartBuffer.refresh against the variant's exports, then save the buffer."""
        created, newest = exports(self.run, self.variant)
        self.buffer.refresh(created, newest)
        self.buffer.save()
        self.refreshed = self.clock()

    def pending(self, limit=None):
        """Shard names without a sidecar under `out`, newest first, among the newest `limit` shards."""
        names = [p.name for p in dense_data.shard_dirs(self.run)][::-1][:limit]
        return [n for n in names if not (self.out/'shards'/n/dense_data.SIDECAR).exists()]

    def status(self, stage, pending):
        hours = max(1e-9, self.clock()-self.started)/3600
        q = self.stats
        write_json(self.out/'solver-status.json', dict(
            stage=stage, updated_at=self.clock(), started_at=self.started, shards_pending=pending,
            **{k: v for k, v in q.items() if k != 'window_plies'}, window_plies=dict(sorted(q['window_plies'].items())),
            positions_per_hour=q['positions']/hours, hits_per_hour=q['hits']/hours,
            gate_pass_rate=q['gated']/q['positions'] if q['positions'] else None,
            busy_fraction=self.busy/(hours*3600), **self.buffer.summary(), settings=asdict(self.s)))

    def loop(self, limit=None, once=False, sleep=time.sleep):
        """Refresh the buffer, then solve pending shards newest first, refreshing every refresh_minutes; poll for new
        shards every poll_seconds, or return once none is pending with `once`."""
        self.refresh()
        while True:
            if self.clock()-self.refreshed >= self.s.refresh_minutes*60:
                self.refresh()
            pending = self.pending(limit)
            if not pending:
                self.status('idle', 0)
                if once:
                    return
                sleep(self.s.poll_seconds)
                continue
            began = time.perf_counter()
            self.shard(pending[0])
            self.busy += time.perf_counter()-began
            self.stats['shards_done'] += 1
            self.status('solving', len(pending)-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', help='directory for sidecars, buffer, status and events (default: the run)')
    parser.add_argument('--limit', type=int, help='consider only the newest N shards')
    parser.add_argument('--once', action='store_true', help='exit when no shard is pending')
    dense_config.add_arguments(parser, PassSettings)
    args = parser.parse_args()
    settings = dense_config.override(PassSettings(), args)
    below_normal()
    out = Path(args.out or args.run)
    tactics = IsolatedTactics()
    try:
        solver = Pass(args.run, out, settings, tactics)
        dense_config.log_event(out, 'solve', 'info', 'proof pass started', settings=asdict(settings))
        solver.loop(args.limit, args.once)
    except Exception as error:
        dense_config.log_event(out, 'solve', 'error', f'proof pass stopped: {error!r}')
        raise
    finally:
        tactics.close()


if __name__ == '__main__':
    main()
