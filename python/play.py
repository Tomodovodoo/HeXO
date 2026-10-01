"""Local browser game: people and engines on either side, analysis and review from saved evaluations.

Run `python python/play.py --dense-run runs/bubble --device cpu`, then open http://127.0.0.1:8765. Engine work runs
on one background worker thread as jobs with ids; HTTP requests only read or change the session, so the page never
waits on an engine. See docs/play.md.
"""
import argparse
import functools
import hashlib
import heapq
import itertools
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from hexo import Game
from notation import NotationConflict, dumps, loads

ROOT = Path(__file__).resolve().parents[1]
PRESETS = dict(
    bubble=dict(quick=dict(simulations=32, solver_nodes=2048), standard=dict(simulations=128, solver_nodes=32768),
                strong=dict(simulations=512, solver_nodes=131072), deep=dict(simulations=2048, solver_nodes=524288)),
    native=dict(quick=dict(ms=250), standard=dict(ms=1000), strong=dict(ms=3000), deep=dict(ms=10000)),
    seal=dict(quick=dict(ms=100), standard=dict(ms=500), strong=dict(ms=2000), deep=dict(ms=8000)))
LIMITS = dict(simulations=(0, 16384), solver_nodes=(0, 4_000_000), ms=(10, 120_000))


def player_at(ply):
    """Side placing stone number `ply` (0-based): X opens with one stone, then two per turn."""
    return 0 if ply == 0 else ((ply - 1) // 2 + 1) % 2


def turn_starts(length):
    """Plies at which a turn starts, for a history of `length` stones."""
    return [0, *range(1, length, 2)]


def review_plies(history):
    """Positions a review evaluates: every turn start, plus the final position unless the game is over."""
    game = replay(history)
    try:
        return turn_starts(len(history)) + ([len(history)] if game.winner < 0 else [])
    finally:
        game.close()


def replay(history):
    """A native game with `history` played; raises ValueError on an illegal stone."""
    game = Game()
    try:
        for q, r in history:
            game.play(int(q), int(r))
    except Exception:
        game.close()
        raise
    return game


def checked_turn(history, moves):
    """`moves` cut to one legal turn from `history`: stops at a win, raises ValueError when incomplete or illegal."""
    game = replay(history)
    try:
        side, played = game.player, []
        for q, r in moves:
            game.play(int(q), int(r))
            played.append([int(q), int(r)])
            if game.winner >= 0 or game.player != side:
                return played
        raise ValueError('Engine returned an incomplete turn')
    finally:
        game.close()


# Engines


def run_checkpoints(run):
    """Checkpoint ids of a run directory, champion first, then newest first."""
    exports = sorted((p for p in Path(run).glob('checkpoints/*/*/ema.pt') if p.parent.name.isdigit()),
                     key=lambda p: (int(p.parent.name), p.parent.parent.name), reverse=True)
    ids = [p.parent.relative_to(Path(run) / 'checkpoints').as_posix() for p in exports]
    try:
        champion = json.loads((Path(run) / 'champion.json').read_text(encoding='utf-8')).get('checkpoint')
    except (OSError, ValueError, AttributeError):
        champion = None
    return ([champion] if champion in ids else []) + [c for c in ids if c != champion]


def scan(models=None, runs=None, extra_runs=(), seal=None):
    """Every engine on offer, by id. Bubble runs come from `extra_runs`, the directories in `runs`, and `models`;
    single `.pt` exports and `<name>.json` entries ({"name", "kind": "bubble", "path"}) come from `models`.
    Entries carry `id`, `name`, `kind`, `presets`, and for Bubble `checkpoints` plus the server-only `path`. An id
    is `kind:name`; entries sharing one get a suffix from their path, so an id never moves to another model."""
    found, seen = [], set()

    def add(kind, name, **fields):
        found.append(dict(name=name, kind=kind, presets=PRESETS[kind], **fields))

    def bubble(path, name=None):
        path = Path(path).resolve()
        if path in seen:
            return
        seen.add(path)
        if path.is_dir():
            if checkpoints := run_checkpoints(path):
                add('bubble', name or path.name, checkpoints=checkpoints, path=path)
        elif path.suffix == '.pt' and path.exists():
            add('bubble', name or (path.parent.name if path.stem == 'ema' else path.stem), checkpoints=[''], path=path)

    for run in extra_runs:
        bubble(run)
    for folder in (runs, models):
        if folder and Path(folder).is_dir():
            for child in sorted(Path(folder).iterdir()):
                if child.is_dir():
                    bubble(child)
    if models and Path(models).is_dir():
        for path in sorted(Path(models).rglob('*.pt')):
            if 'checkpoints' not in path.relative_to(models).parts:
                bubble(path)
        for path in sorted(Path(models).glob('*.json')):
            try:
                spec = json.loads(path.read_text(encoding='utf-8'))
                if spec.get('kind') == 'bubble':
                    bubble(path.parent / spec['path'], spec.get('name'))
            except (OSError, ValueError, KeyError, TypeError):
                continue
    add('native', 'Native')
    if seal is not None and Path(seal).exists():
        add('seal', 'Seal')
    bases = [f"{e['kind']}:{e['name']}" for e in found]
    entries = OrderedDict()
    for base, entry in zip(bases, found):
        suffix = hashlib.blake2b(str(entry.get('path')).encode(), digest_size=3).hexdigest()
        key = base if bases.count(base) == 1 else f'{base}~{suffix}'
        entries[key] = dict(id=key, **entry)
    return entries


def export_path(entry, checkpoint):
    """The weights file of a Bubble entry at `checkpoint`."""
    return entry['path'] / 'checkpoints' / checkpoint / 'ema.pt' if checkpoint else entry['path']


@functools.lru_cache(maxsize=256)
def file_digest(path, modified_ns, size):
    from legacy.train import digest
    return digest(path)


@functools.lru_cache(maxsize=8)
def file_build(package, modified_ns):
    """The solver build of `package`, or 'none' when its record is unreadable."""
    import tactical_proof
    try:
        return tactical_proof.build_hash(package)[:8]
    except (OSError, ValueError, KeyError, TypeError):
        return 'none'


def model_key(path):
    """Evaluations are keyed by the weights they came from: the first 16 hex digits of the file's SHA-256."""
    stat = Path(path).stat()
    return file_digest(str(path), stat.st_mtime_ns, stat.st_size)[:16]


class Cancelled(Exception):
    """The job was cancelled while the engine was working."""


class Yielded(Exception):
    """A review stepped aside for more urgent jobs; it goes back in the queue."""


class Watched:
    """A network evaluator that reports each batch to `watch`, which may raise Cancelled."""

    def __init__(self, inner, watch):
        self.inner, self.watch = inner, watch

    def evaluate(self, histories):
        self.watch(len(histories))
        return self.inner.evaluate(histories)


class Bubble:
    """One HexNet export on `device` with its evaluation cache."""

    def __init__(self, path, device):
        import hexnet
        from legacy.train import digest
        from neural_search import EvaluationCache
        self.sha256 = digest(path)
        self.evaluator = hexnet.DenseEvaluator(hexnet.load_model(path), device, self.sha256, max_batch=16)
        self.cache = EvaluationCache(4096)


def verified(result):
    return result.get('status') == 'PROVEN_WIN' and result.get('native_verified')


def searched(result):
    """True when a solver query ran to a verdict (a proof, or a reason in `dense_solver.VERDICTS`); False when it
    failed to run (worker starting or restarting, deadline, crash)."""
    from dense_solver import VERDICTS
    return verified(result) or result.get('reason') in VERDICTS


def winning_line(history, result):
    """One legal continuation of a verified strategy as [q, r, player] stones, choosing its first covered reply."""
    from dense_solver import Proof
    certificate = result.get('certificate') or json.loads(result['certificate_json'])
    proof = Proof([tuple(p) for p in history], certificate)
    local, line = replay(history), []
    try:
        while local.winner < 0:
            current = [cell[:2] for cell in local.cells]
            move = proof.path(current)[1]
            if move is None:
                break
            actions = move[0] or proof.reply(current) or local.legal_moves()[:local.remaining]
            for action in actions:
                line.append([*action, local.player])
                local.play(*action)
                if local.winner >= 0:
                    break
        return line
    finally:
        local.close()


def interruptible(call, watch):
    """`call()` on its own thread, polling `watch(0)` meanwhile; when `watch` raises, the call is left to finish
    alone and the exception propagates."""
    result = {}
    def run():
        try:
            result['value'] = call()
        except Exception as error:
            result['error'] = error
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(.05)
        watch(0)
    if 'error' in result:
        raise result['error']
    return result['value']


def evaluate(bubble, prover, history, simulations, solver_nodes, watch=lambda n: None):
    """Bubble's turn from `history` and what it thinks of the position.

    Returns `moves` (the turn it plays), `value` (win probability of the side to move), `top` (five best first
    stones as [q, r, probability]), `proof` (None or {winner, turns}: the solver proved a win for the side to move,
    or the search proved the position exact), `line` (a winning line as [q, r, player] when the solver proved it)
    and `threat` (the stones of a forced win the opponent would have if it moved now). `solved` is False when a
    solver query failed to run (worker restarting, deadline), so the result must not count as solver-checked.
    `simulations` 0 plays the raw policy; `solver_nodes` 0 or no `prover` skips the solver. `watch(n)` is called
    before each network batch of n positions and may raise Cancelled."""
    import numpy as np
    from dense_selfplay import root_value
    from neural_search import NeuralSearch
    history = [tuple(map(int, p)) for p in history]
    local = replay(history)
    start, player = time.perf_counter(), local.player
    moves, top, value, proof, line, threat, solved = [], [], None, None, [], [], True
    network = Watched(bubble.evaluator, watch)
    try:
        if local.winner >= 0:
            raise ValueError('The game has finished')
        if prover is not None and solver_nodes:
            mine = interruptible(lambda: prover.history(history, attacker='mover', nodes=solver_nodes, ms=10000), watch)
            solved = searched(mine)
            if verified(mine):
                moves, line = [list(m) for m in mine['moves']], winning_line(history, mine)
                proof = dict(winner=player, turns=mine['proof_turns'])
            else:
                theirs = interruptible(lambda: prover.history(history, attacker='opponent', nodes=solver_nodes, ms=10000),
                                       watch)
                solved = solved and searched(theirs)
                if verified(theirs):
                    threat = [list(m) for m in theirs['moves']]
        given = bool(moves)
        while not given and local.player == player and local.winner < 0:
            current = [tuple(cell[:2]) for cell in local.cells]
            if simulations:
                tree = NeuralSearch(network, bubble.sha256, current, seed=1740, cache=bubble.cache, tactics=True)
                try:
                    result = tree.search(simulations, root_samples=16, batch_size=16)
                finally:
                    tree.close()
                action, policy, actions = result['action'], result['policy'], result['actions']
                stone_value = root_value(result, local.player)
                if result.get('proven') and proof is None and not moves:
                    proof = dict(winner=player if result['proven'] > 0 else 1 - player,
                                 turns=(result['proof_plies'] + 1) // 2)
            else:
                result = network.evaluate([current])[0]
                actions = result['actions']
                policy = np.exp(result['logits'] - result['logits'].max())
                policy /= policy.sum()
                action, stone_value = actions[policy.argmax()].tolist(), float(result['q'][0])
            if not moves:
                top = [[*map(int, actions[i]), round(float(policy[i]), 4)] for i in np.argsort(-policy)[:5]]
                value = (stone_value + 1) / 2
            moves.append([int(action[0]), int(action[1])])
            local.play(*moves[-1])
        if proof:
            value = 1. if proof['winner'] == player else 0.
        return dict(moves=moves, value=round(value, 4), top=top, proof=proof, line=line, threat=threat,
                    solved=solved, ms=round((time.perf_counter() - start) * 1000))
    finally:
        local.close()


class Engines:
    """Loaded engines, used only from the worker thread. Keeps the three most recent Bubble exports."""

    def __init__(self, device, tactical_package=None, seal=None):
        self.device, self.tactical_package, self.seal_path = device, tactical_package, seal
        self.bubbles, self.prover, self.prover_build = OrderedDict(), None, None
        self.children = {}

    def bubble(self, path):
        """The loaded export at `path`, reloaded when the file changes."""
        stat = Path(path).stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key not in self.bubbles:
            self.bubbles[key] = Bubble(path, self.device)
            while len(self.bubbles) > 3:
                self.bubbles.popitem(last=False)
        self.bubbles.move_to_end(key)
        return self.bubbles[key]

    def solver(self):
        """The tactical solver in its own process, so a cancelled query can be ended, and its build; (None, 'none')
        when not built. A rebuilt library replaces the running solver."""
        import tactical_proof
        build = self.solver_build()
        if build == 'none':
            return None, build
        if self.prover is None or self.prover_build != build:
            if self.prover is not None:
                old = self.prover
                old.abort()
                threading.Thread(target=old.close, daemon=True).start()
            package = self.tactical_package or tactical_proof.PACKAGE
            self.prover, self.prover_build = tactical_proof.IsolatedTactics(package, priority='below_normal'), build
        return self.prover, build

    def evaluate(self, entry, checkpoint, budget, history, watch):
        """`evaluate` with the entry's export; returns the evaluation, the budget it really had (no solver nodes
        when the solver is not built) and the key of the weights it used (see `model_key`)."""
        bubble, (solver, build) = self.bubble(export_path(entry, checkpoint)), self.solver()
        spent = budget if solver else budget | dict(solver_nodes=0)
        try:
            found = evaluate(bubble, solver, history, spent['simulations'], spent['solver_nodes'], watch)
        except Cancelled:
            if solver is not None:
                solver.abort()
            raise
        if not found.pop('solved'):
            spent = spent | dict(solver_nodes=0)
        return found, spent, f'{bubble.sha256[:16]}:{build}'

    def effective(self, budget):
        """`budget` as it can run here: no solver nodes when the tactical library is not built."""
        return budget if self.solver_build() != 'none' else budget | dict(solver_nodes=0)

    def solver_build(self):
        """The first 8 hex digits of the tactical library's recorded SHA-256, or 'none' when the library or its
        record is missing."""
        import tactical_proof
        package = self.tactical_package or tactical_proof.PACKAGE
        binary = tactical_proof.library(package)
        record = binary.with_name(binary.name + '.json')
        if not (binary.exists() and record.exists()):
            return 'none'
        stat = record.stat()
        return file_build(str(package), stat.st_mtime_ns)

    def turn(self, entry, budget, history, stop=lambda: False):
        """A native or Seal turn. Their searches cannot be interrupted in process, so each kind searches in a child
        process; when `stop()` turns true the child is killed, a fresh one starts on the next turn, and this raises
        Cancelled."""
        kind = entry['kind']
        if kind not in self.children or self.children[kind][0].poll() is not None:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'search', kind],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8',
                                     bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            lines = queue.Queue()
            def pump():
                for line in child.stdout:
                    lines.put(line)
                lines.put(None)
            threading.Thread(target=pump, daemon=True).start()
            self.children[kind] = child, lines
        child, lines = self.children[kind]
        child.stdin.write(json.dumps(dict(ms=budget['ms'], history=[list(p) for p in history])) + '\n')
        child.stdin.flush()
        while True:
            try:
                line = lines.get(timeout=.05)
                break
            except queue.Empty:
                if stop():
                    child.kill()
                    child.stdin.close()
                    del self.children[kind]
                    raise Cancelled() from None
        if line is None:
            raise RuntimeError(f'{kind} search process exited')
        answer = json.loads(line)
        if 'error' in answer:
            raise RuntimeError(answer['error'])
        return answer['moves']

    def close(self):
        self.bubbles.clear()
        for child, _ in self.children.values():
            child.kill()


# Saved evaluations


def position_text(history):
    return ' '.join(f'{int(q)},{int(r)}' for q, r in history)


class Evaluations:
    """Append-only JSON lines, one evaluation per line, indexed in memory.

    Each line holds `position` (the ordered stones), `engine` (see `model_key`), `simulations`, `solver_nodes`,
    the evaluation fields, `model` (a readable name) and `at`. The index keeps the newest `limit` (position, engine, budget)
    entries; `best` picks the one to show for a position and engine. The file is only appended to; on start a
    dated copy is kept next to it, the newest `backups` of them."""

    def __init__(self, path=None, limit=200_000, backups=3):
        self.path, self.limit = Path(path) if path else None, limit
        self.lock = threading.Lock()
        self.order, self.by_position = OrderedDict(), {}
        if self.path and self.path.exists():
            stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            shutil.copyfile(self.path, self.path.with_name(f'{self.path.name}.{stamp}.bak'))
            for old in sorted(self.path.parent.glob(f'{self.path.name}.*.bak'))[:-backups]:
                old.unlink()
            with open(self.path, encoding='utf-8') as lines:
                for line in lines:
                    try:
                        self.index(json.loads(line), line.strip())
                    except (ValueError, KeyError, TypeError):
                        continue

    @staticmethod
    def key(history):
        return hashlib.blake2b(position_text(history).encode(), digest_size=16).digest()

    def index(self, record, line):
        position = hashlib.blake2b(record['position'].encode(), digest_size=16).digest()
        budget = (record['simulations'], record['solver_nodes'])
        full = (position, record['engine'], budget)
        self.order.pop(full, None)
        self.order[full] = line
        self.by_position.setdefault((position, record['engine']), set()).add(budget)
        while len(self.order) > self.limit:
            (old, engine, spent), _ = self.order.popitem(last=False)
            budgets = self.by_position[(old, engine)]
            budgets.discard(spent)
            if not budgets:
                del self.by_position[(old, engine)]

    def add(self, history, engine, budget, evaluation):
        record = dict(position=position_text(history), engine=engine, simulations=budget['simulations'],
                      solver_nodes=budget['solver_nodes'], **evaluation,
                      at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
        line = json.dumps(record, separators=(',', ':'))
        with self.lock:
            if self.path:
                with open(self.path, 'a', encoding='utf-8') as out:
                    out.write(line + '\n')
            self.index(record, line)
        return record

    def get(self, history, engine, budget):
        """The saved evaluation of `history` by `engine` at exactly `budget`, or None."""
        with self.lock:
            line = self.order.get((self.key(history), engine, (budget['simulations'], budget['solver_nodes'])))
        return json.loads(line) if line else None

    def covering(self, history, engine, budget):
        """The deepest saved evaluation of `history` by `engine` whose simulations and solver nodes both reach
        `budget`'s, or None."""
        position, need = self.key(history), (budget['simulations'], budget['solver_nodes'])
        with self.lock:
            enough = [b for b in self.by_position.get((position, engine), ()) if b[0] >= need[0] and b[1] >= need[1]]
            line = self.order[(position, engine, max(enough))] if enough else None
        return json.loads(line) if line else None

    def best(self, history, engine):
        """The saved evaluation of `history` by `engine` to show, or None: one holding a proof first, since a proof
        is exact, then the most simulations, then the most solver nodes."""
        position = self.key(history)
        with self.lock:
            lines = [self.order[(position, engine, b)] for b in self.by_position.get((position, engine), ())]
        found = [json.loads(line) for line in lines]
        return max(found, key=lambda e: (e.get('proof') is not None, e['simulations'], e['solver_nodes']), default=None)


# Review


def review(history, lookup, winner=-1):
    """Label every complete turn of `history` from saved evaluations.

    `lookup(prefix)` returns the evaluation of a position (fields of `evaluate`) or None. A turn is judged on the
    mover's win probability before and after it; the first label that applies wins: win (made six), lost (the
    opponent already had a proven win), kept or missed (the mover had one and kept or lost it), allowed (handed the
    opponent one), found (proved one), best (the engine's own turn), then the loss bands good (< 0.05),
    inaccuracy (< 0.10), mistake (< 0.20) and blunder. Turns lacking an evaluation get label None. For
    inaccuracy and worse, missed and allowed, `better` is the engine's turn and `line` its continuation."""
    turns = []
    starts = turn_starts(len(history))
    for s, e in zip(starts, [*starts[1:], len(history)]):
        me = player_at(s)
        stones = [list(p) for p in history[s:e]]
        if e - s < (1 if s == 0 else 2) and not (e == len(history) and winner == me):
            break
        turn = dict(ply=s, player=me, stones=stones, label=None, before=None, after=None, better=None, line=None)
        turns.append(turn)
        if e == len(history) and winner == me:
            turn['label'] = 'win'
            continue
        before, after = lookup(history[:s]), lookup(history[:e])
        if before is None or after is None:
            continue
        turn['before'], turn['after'] = before['value'], 1 - after['value']
        had = (before.get('proof') or {}).get('winner')
        has = (after.get('proof') or {}).get('winner')
        loss = turn['before'] - turn['after']
        if had == 1 - me:
            label = 'lost'
        elif had == me:
            label = 'kept' if has == me else 'missed'
        elif has == 1 - me:
            label = 'allowed'
        elif has == me:
            label = 'found'
        elif before['moves'] and sorted(map(tuple, before['moves'])) == sorted(map(tuple, stones)):
            label = 'best'
        else:
            label = 'good' if loss < .05 else 'inaccuracy' if loss < .1 else 'mistake' if loss < .2 else 'blunder'
        turn['label'] = label
        if label in ('inaccuracy', 'mistake', 'blunder', 'missed', 'allowed') and before['moves']:
            turn['better'] = before['moves']
            if before.get('line'):
                turn['line'] = before['line']
            else:
                reply = lookup([*history[:s], *map(tuple, before['moves'])])
                turn['line'] = [[*p, me] for p in before['moves']] + \
                               [[*p, 1 - me] for p in (reply or {}).get('moves', [])]
    return turns


# Session and jobs


class Job:
    """One unit of engine work. `kind` is move, analyse or review; progress is `done` of `total`."""
    ids = itertools.count(1)

    def __init__(self, kind, priority, history, **fields):
        self.id, self.kind, self.priority, self.history = next(Job.ids), kind, priority, tuple(history)
        self.status, self.done, self.total, self.error, self.cancelled, self.ended = 'queued', 0, 1, None, False, None
        self.__dict__.update(fields)

    def summary(self):
        return dict(id=self.id, kind=self.kind, status=self.status, done=self.done, total=self.total,
                    error=self.error, ply=len(self.history), side=getattr(self, 'side', None))


def budget_of(kind, preset, custom=None):
    """The budget of `preset` for an engine kind, or `custom` checked against LIMITS."""
    if preset != 'custom':
        if preset not in PRESETS[kind]:
            raise ValueError('Unknown preset')
        return dict(PRESETS[kind][preset])
    budget = dict(PRESETS[kind]['standard']) | (custom or {})
    for key, value in budget.items():
        low, high = LIMITS[key]
        if key not in PRESETS[kind]['standard'] or type(value) is not int or not low <= value <= high:
            raise ValueError(f'{key} must be {low}..{high}')
    return budget


class Session:
    """The game, the seats, the analysis settings and the job queue. HTTP threads call the public methods; one
    worker thread runs the jobs through `engines`. `revision` grows with every change the page must redraw."""

    def __init__(self, entries, engines, store, rescan=lambda: None):
        self.entries, self.engines, self.store, self.rescan_entries = entries, engines, store, rescan
        self.lock = threading.Condition()
        self.history, self.revision, self.paused = [], 0, False
        self.jobs, self.queue, self.order = OrderedDict(), [], itertools.count()
        bubble = next((e for e in entries.values() if e['kind'] == 'bubble'), None)
        opponent = bubble or entries['native:Native']
        self.seats = [dict(engine='human'), self.seat(opponent['id'], None, 'standard')]
        self.analysis = self.seat(bubble['id'], None, 'standard') | dict(auto=True) if bubble else None
        self.worker = threading.Thread(target=self.work, daemon=True)
        self.worker.start()

    def seat(self, engine, checkpoint, preset, custom=None):
        if engine == 'human':
            return dict(engine='human')
        entry = self.entries.get(engine)
        if entry is None:
            raise ValueError('Unknown engine')
        if entry['kind'] == 'bubble':
            checkpoint = entry['checkpoints'][0] if checkpoint is None else checkpoint
            if checkpoint not in entry['checkpoints']:
                raise ValueError('Unknown checkpoint')
        else:
            checkpoint = None
        return dict(engine=engine, checkpoint=checkpoint, preset=preset, budget=budget_of(entry['kind'], preset, custom))

    def engine_key(self, seat):
        """Evaluations are keyed by the weights and the solver build that produced them."""
        weights = model_key(export_path(self.entries[seat['engine']], seat['checkpoint']))
        return f'{weights}:{self.engines.solver_build()}'

    # Reading

    def lookup(self, history):
        return self.store.best(history, self.engine_key(self.analysis)) if self.analysis else None

    def state(self):
        with self.lock:
            history, game = list(self.history), replay(self.history)
            try:
                board = dict(player=game.player, remaining=game.remaining, winner=game.winner)
            finally:
                game.close()
            evaluations = {}
            for ply in range(len(history) + 1):
                if (found := self.lookup(history[:ply])) is not None:
                    evaluations[ply] = {k: found.get(k) for k in
                                        ('value', 'moves', 'top', 'proof', 'line', 'threat', 'simulations', 'solver_nodes')}
            entries = [{k: v for k, v in e.items() if k != 'path'} for e in self.entries.values()]
            return dict(revision=self.revision, history=[list(p) for p in history], **board, paused=self.paused,
                        seats=self.seats, analysis=self.analysis, engines=entries, evaluations=evaluations,
                        review=review(history, self.lookup, board['winner']), jobs=self.job_list())

    def job_list(self):
        """Queued and running jobs, then jobs that failed in the last ten seconds with their `error`."""
        now = time.time()
        return [job.summary() for job in self.jobs.values()
                if job.status in ('queued', 'running') or job.status == 'failed' and now - job.ended < 10]

    def poll(self, since):
        with self.lock:
            if since == self.revision:
                return dict(revision=self.revision, jobs=self.job_list())
        return self.state()

    # Changing

    def changed(self):
        """Call with the lock held after any change: bumps the revision and queues the jobs the change calls for."""
        self.revision += 1
        game = replay(self.history)
        try:
            player, winner, remaining = game.player, game.winner, game.remaining
        finally:
            game.close()
        seat = self.seats[player]
        busy = any(j.kind == 'move' and j.status in ('queued', 'running') and not j.cancelled
                   and j.history == tuple(self.history) for j in self.jobs.values())
        if winner < 0 and not self.paused and seat['engine'] != 'human' and not busy:
            self.submit(Job('move', 1, self.history, side=player, seat=dict(seat)))
        opening = not self.history or remaining == 2
        if self.analysis and self.analysis['auto'] and winner < 0 and opening:
            self.request_analysis(self.history, 1)
        self.lock.notify_all()

    def submit(self, job):
        self.jobs[job.id] = job
        heapq.heappush(self.queue, (job.priority, next(self.order), job))
        while len(self.jobs) > 200:
            oldest = next(iter(self.jobs))
            if self.jobs[oldest].status in ('queued', 'running'):
                break
            del self.jobs[oldest]
        self.lock.notify_all()
        return job

    def request_analysis(self, history, priority, force=False):
        """Queue an evaluation of `history` by the analysis engine unless it is saved or already queued at the same
        or a more urgent `priority` (lower runs first)."""
        settings = dict(self.analysis)
        for job in self.jobs.values():
            if job.kind == 'analyse' and job.history == tuple(history) and job.seat == settings:
                if job.status == 'queued' and job.priority > priority:
                    job.cancelled, job.status = True, 'cancelled'
                elif job.status in ('queued', 'running'):
                    return job
                if job.status in ('failed', 'done') and not force and time.time() - job.ended < 30:
                    return None
        if not force and self.store.covering(history, self.engine_key(settings), self.engines.effective(settings['budget'])):
            return None
        return self.submit(Job('analyse', priority, history, seat=settings, force=force))

    def play(self, q, r):
        with self.lock:
            game = replay(self.history)
            try:
                if game.winner >= 0 or self.seats[game.player]['engine'] != 'human':
                    raise ValueError('It is not your turn')
                game.play(q, r)
            finally:
                game.close()
            self.history.append((q, r))
            self.changed()

    def undo(self):
        """Take back stones to the start of the latest turn a person played, or one stone without people."""
        with self.lock:
            if not self.history:
                return
            people = [i for i, seat in enumerate(self.seats) if seat['engine'] == 'human']
            self.history.pop()
            while people and self.history and not (player_at(len(self.history)) in people
                                                   and len(self.history) in turn_starts(len(self.history) + 1)):
                self.history.pop()
            self.stop_moves()
            self.changed()

    def load(self, history, paused):
        """Replace the game with `history` (validated)."""
        replay(history).close()
        with self.lock:
            self.history, self.paused = [tuple(map(int, p)) for p in history], paused
            self.stop_moves()
            self.changed()

    def stop_moves(self):
        for job in self.jobs.values():
            if job.kind == 'move' and job.status in ('queued', 'running'):
                job.cancelled = True
                if job.status == 'queued':
                    job.status = 'cancelled'

    def configure_seat(self, side, engine, checkpoint=None, preset='standard', custom=None):
        seat = self.seat(engine, checkpoint, preset, custom)
        with self.lock:
            self.seats[side] = seat
            self.stop_moves()
            self.changed()

    def configure_analysis(self, engine, checkpoint=None, preset='standard', custom=None, auto=True):
        seat = self.seat(engine, checkpoint, preset, custom)
        if self.entries[engine]['kind'] != 'bubble' or type(auto) is not bool:
            raise ValueError('Analysis needs a Bubble model')
        with self.lock:
            self.analysis = seat | dict(auto=auto)
            self.stop_analysis()
            self.changed()

    def stop_analysis(self):
        """Cancel analysis and review jobs made for other analysis settings than the current ones."""
        current = {k: v for k, v in (self.analysis or {}).items() if k != 'auto'}
        for job in self.jobs.values():
            settings = {k: v for k, v in getattr(job, 'seat', {}).items() if k != 'auto'}
            if job.kind in ('analyse', 'review') and job.status in ('queued', 'running') and settings != current:
                job.cancelled = True
                if job.status == 'queued':
                    job.status = 'cancelled'

    def analyse(self, ply, force=False):
        with self.lock:
            if not self.analysis or not 0 <= ply <= len(self.history):
                raise ValueError('Nothing to analyse')
            for job in self.jobs.values():
                if job.kind == 'analyse' and job.priority == 0 and job.status == 'queued':
                    job.cancelled, job.status = True, 'cancelled'
            job = self.request_analysis(self.history[:ply], 0, force)
            self.lock.notify_all()
            return job.id if job else None

    def review_game(self):
        with self.lock:
            if not self.analysis:
                raise ValueError('Review needs a Bubble model')
            job = self.submit(Job('review', 2, self.history, seat=dict(self.analysis)))
            job.total = len(review_plies(self.history))
            return job.id

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job and job.status in ('queued', 'running'):
                job.cancelled = True
                if job.kind == 'move':
                    self.paused = True
                if job.status == 'queued':
                    job.status = 'cancelled'
                self.revision += 1

    def pause(self, paused):
        with self.lock:
            self.paused = bool(paused)
            if self.paused:
                self.stop_moves()
            self.changed()

    def rescan(self):
        """Replace the engine list; a seat whose engine or checkpoint is gone becomes a person, and analysis falls
        back to the first Bubble model."""
        entries = self.rescan_entries()
        def valid(seat):
            entry = entries.get(seat['engine'])
            return entry is not None and (entry['kind'] != 'bubble' or seat['checkpoint'] in entry['checkpoints'])
        with self.lock:
            previous, self.entries = self.entries, entries
            try:
                seats = [seat if seat['engine'] == 'human' or valid(seat) else dict(engine='human') for seat in self.seats]
                analysis = self.analysis
                if analysis is None or not valid(analysis):
                    bubble = next((e for e in entries.values() if e['kind'] == 'bubble'), None)
                    analysis = self.seat(bubble['id'], None, 'standard') | dict(auto=True) if bubble else None
            except Exception:
                self.entries = previous
                raise
            self.seats, self.analysis = seats, analysis
            self.stop_moves()
            self.stop_analysis()
            self.changed()

    # Working

    def work(self):
        while True:
            with self.lock:
                while not self.queue:
                    self.lock.wait()
                job = heapq.heappop(self.queue)[2]
                if job.cancelled:
                    job.status = 'cancelled'
                    continue
                job.status = 'running'
                self.revision += 1
            try:
                result = self.run(job)
                with self.lock:
                    job.status, job.ended = 'cancelled' if job.cancelled else 'done', time.time()
                    if job.kind == 'move' and not job.cancelled and list(job.history) == self.history:
                        self.history.extend(tuple(p) for p in result)
                    self.changed()
            except Cancelled:
                with self.lock:
                    job.status = 'cancelled'
                    self.changed()
            except Yielded:
                with self.lock:
                    job.status = 'queued'
                    heapq.heappush(self.queue, (job.priority, next(self.order), job))
            except Exception as error:
                with self.lock:
                    job.status, job.error, job.ended = 'failed', str(error), time.time()
                    if job.kind == 'move':
                        self.paused = True
                    self.changed()

    def watcher(self, job, count=True):
        """A network-batch callback that stops a cancelled job and, when `count`, adds batch sizes to its progress."""
        def watch(n):
            if job.cancelled:
                raise Cancelled()
            if count:
                job.done += n
        return watch

    def evaluation(self, job, seat, history, force=False, exact=False):
        """The evaluation of `history` for `seat`, saved. Unless forced, a saved one is reused: at exactly the
        seat's budget when `exact`, else at least as deep."""
        key, budget = self.engine_key(seat), self.engines.effective(seat['budget'])
        saved = None if force else self.store.get(history, key, budget) if exact else self.store.covering(history, key, budget)
        if saved:
            return saved
        found, spent, weights = self.engines.evaluate(self.entries[seat['engine']], seat['checkpoint'], budget,
                                                      history, self.watcher(job, job.kind != 'review'))
        if job.cancelled:
            raise Cancelled()
        entry = self.entries[seat['engine']]
        model = f"{entry['name']}/{seat['checkpoint']}" if seat['checkpoint'] else entry['name']
        return self.store.add(history, weights, spent, found | dict(model=model))

    def run(self, job):
        seat, history = job.seat, list(job.history)
        entry = self.entries[seat['engine']]
        if job.kind == 'move':
            if entry['kind'] == 'bubble':
                job.total = max(1, seat['budget']['simulations']) * 2
                moves = self.evaluation(job, seat, history, exact=True)['moves']
            else:
                moves = self.engines.turn(entry, seat['budget'], history, lambda: job.cancelled)
            return checked_turn(history, moves)
        if job.kind == 'analyse':
            job.total = max(1, seat['budget']['simulations']) * 2
            return self.evaluation(job, seat, history, job.force)
        for index, ply in enumerate(review_plies(history)):
            if job.cancelled:
                raise Cancelled()
            with self.lock:
                if self.queue and self.queue[0][0] < job.priority:
                    raise Yielded()
            self.evaluation(job, seat, history[:ply])
            job.done = index + 1
            with self.lock:
                self.revision += 1
        return None


# HTTP


def import_history(text):
    """A history from HTTTX notation, a replay file ({"history": [...]}) or a JSON list of [q, r]."""
    stripped = text.strip()
    if not stripped:
        raise ValueError('Nothing to import')
    if stripped.startswith(('{', '[')):
        data = json.loads(stripped)
        history = data['history'] if isinstance(data, dict) else data
        if not isinstance(history, list) or any(not isinstance(p, list) or len(p) != 2 or
                                                 any(type(v) is not int for v in p) for p in history):
            raise ValueError('A replay holds a list of [q, r] stones')
        return history
    return [list(p) for p in loads(text).history]


class Handler(BaseHTTPRequestHandler):
    session = None
    page = ROOT / 'web' / 'index.html'

    def log_message(self, *args):
        pass

    def respond(self, status, data, content_type='application/json', headers=()):
        payload = data if isinstance(data, bytes) else data.encode() if isinstance(data, str) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        url, session = urlparse(self.path), self.session
        if url.path == '/':
            return self.respond(200, self.page.read_bytes(), 'text/html; charset=utf-8')
        if url.path == '/state':
            since = parse_qs(url.query).get('since', [''])[0]
            return self.respond(200, session.poll(int(since)) if since.isdigit() else session.state())
        if url.path == '/htttx':
            try:
                return self.respond(200, dumps(list(session.history)), 'text/plain; charset=utf-8')
            except NotationConflict as error:
                return self.respond(409, dict(error=str(error)))
        if url.path == '/replay':
            names = [seat['engine'] for seat in session.seats]
            body = dict(format='bubble-replay', version=1, players=names, history=[list(p) for p in session.history])
            return self.respond(200, json.dumps(body), headers=[('Content-Disposition', 'attachment; filename="game.json"')])
        if url.path == '/evaluations':
            path = session.store.path
            data = path.read_bytes() if path and path.exists() else b''
            return self.respond(200, data, 'application/x-ndjson',
                                [('Content-Disposition', 'attachment; filename="evaluations.jsonl"')])
        self.respond(404, dict(error='Not found'))

    def do_POST(self):
        origin = self.headers.get('Origin')
        if origin and origin != f"http://{self.headers.get('Host')}":
            return self.respond(403, dict(error='Origin rejected'))
        session = self.session
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 <= length <= 1 << 20:
                raise ValueError('Request too large')
            args = json.loads(self.rfile.read(length) or '{}')
            reply = {}
            if self.path == '/play':
                session.play(args['q'], args['r'])
            elif self.path == '/undo':
                session.undo()
            elif self.path == '/new':
                session.load([], False)
            elif self.path == '/retry':
                ply = args['ply']
                if type(ply) is not int or not 0 <= ply <= len(session.history):
                    raise ValueError('No such position')
                session.load(session.history[:ply], False)
            elif self.path == '/import':
                session.load(import_history(args['text']), True)
            elif self.path == '/seat':
                if args.get('side') not in (0, 1):
                    raise ValueError('Side must be 0 or 1')
                session.configure_seat(args['side'], args['engine'], args.get('checkpoint'),
                                       args.get('preset', 'standard'), args.get('custom'))
            elif self.path == '/analysis':
                session.configure_analysis(args['engine'], args.get('checkpoint'), args.get('preset', 'standard'),
                                           args.get('custom'), args.get('auto', True))
            elif self.path == '/analyse':
                if type(args.get('ply')) is not int:
                    raise ValueError('Choose a position')
                reply = dict(job=session.analyse(args['ply'], args.get('force') is True))
            elif self.path == '/review':
                reply = dict(job=session.review_game())
            elif self.path == '/cancel':
                session.cancel(args['id'])
            elif self.path == '/pause':
                session.pause(args['paused'])
            elif self.path == '/rescan':
                session.rescan()
            else:
                return self.respond(404, dict(error='Not found'))
            self.respond(200, session.state() | reply)
        except (ValueError, KeyError, TypeError) as error:
            self.respond(400, dict(error=str(error)))


def search_child(kind):
    """The child process of `Engines.turn`: one JSON line {ms, history} in, one line {moves} or {error} out."""
    engine = None
    if kind == 'seal':
        from legacy.arena import Seal
        engine = Seal()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            game = replay(request['history'])
            try:
                moves = game.search(request['ms'])['moves'] if kind == 'native' else engine(game, request['ms'])
            finally:
                game.close()
            answer = dict(moves=[list(map(int, m)) for m in moves])
        except Exception as error:
            answer = dict(error=str(error))
        print(json.dumps(answer), flush=True)


def main():
    if sys.argv[1:2] == ['search']:
        return search_child(sys.argv[2])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--dense-run', type=Path, help='a Bubble run; its champion is the default opponent')
    parser.add_argument('--dense-model', type=Path, help='an ema.pt file; the default opponent, ahead of --dense-run')
    parser.add_argument('--models', type=Path, default=ROOT / 'models', help='folder scanned for models and engines')
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs', help='folder whose runs are offered as models')
    parser.add_argument('--evaluations', type=Path,
                        help='saved evaluations file; default play-evaluations.jsonl in the run or models folder')
    parser.add_argument('--tactical-package', type=Path, help='directory with the built tactical solver')
    parser.add_argument('--device', default='auto', help='cuda, cpu, or auto: cuda when a GPU is available')
    args = parser.parse_args()
    try:
        import torch
        torch.set_num_threads(2)
        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    if args.device == 'auto':
        args.device = 'cuda' if cuda else 'cpu'
    from hexo import library
    seal = library.with_name(library.name.replace('hexo', 'hexo_seal'))
    runs = [path for path in (args.dense_model, args.dense_run) if path]
    find = lambda: scan(args.models, args.runs, runs, seal)
    store_path = args.evaluations or (args.dense_run or args.models) / 'play-evaluations.jsonl'
    store_path.parent.mkdir(parents=True, exist_ok=True)
    Handler.session = Session(find(), Engines(args.device, args.tactical_package), Evaluations(store_path), find)
    with Handler.session.lock:
        Handler.session.changed()
    print(f'Bubble is ready at http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
