"""Dense self-play actor: many native Gumbel trees batched continuously onto one DenseEvaluator.

Run contract (see dense_config for settings):
  <run>/champion.json       {"checkpoint": "<variant>/<step:06d>", "ema_sha256", "updated_at"}; the actor plays
                            that checkpoint's ema.pt, else the newest complete checkpoint, else --initial-model,
                            else a fresh HexNet(config.model) seeded with config.seed.
  <run>/checkpoints/<variant>/<step:06d>/  complete once manifest.json exists next to ema.pt.
  <run>/shards/<ms:013d><pid%1000:03d>/    dense_data shards; rows carry no value targets (target null,
                            weight 0): the learner derives them from episode root_values and winner.
  <run>/actor-status[-k].json, <run>/events.jsonl.

Scheduling: every game owns one persistent NeuralSearch tree. Engine keeps all trees searching at once and
starts a tree's next search as soon as its previous one finishes, so full (`full_sims`) and cheap
(`cheap_sims`) searches share every GPU batch and no lockstep tail waits on the slowest tree. The champion
is re-read after each shard: games in progress finish with the evaluator they started with, new games use
the new one; a shard may therefore mix actors (identity `actors`; `actor_sha256` is the newest).
"""
import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

import dense_config
import dense_data
import hexcrop
import hexnet
from hexo import Game
from klent import digest
from neural_search import EvaluationCache, NeuralSearch, checked, native
from train import write_json

# Crop cells per forward (48 positions at 48x48; larger crops get proportionally fewer; measured: same
# throughput as 96 at about half the VRAM, ~0.67 GB per process for b6c96). A crop-size group
# is padded into the next larger size present when that adds fewer than MERGE_CELLS cells: measured b6c96
# bf16 forwards cost about 8 ms of launch overhead plus 0.23 us per cell on an RTX 3070 Ti.
MAX_CELLS, MERGE_CELLS = 48*48*48, 32768
COLOR = ((np.arange(1 << 16)+1)//2) % 2


def log_event(run, source, kind, message, **fields):
    """Append one line to <run>/events.jsonl."""
    with (Path(run)/'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(time=time.time(), source=source, kind=kind, message=message, **fields),
                                allow_nan=False)+'\n')


def checkpoints(run):
    """Complete checkpoints as [(id, directory, manifest)] ordered by manifest created_at, then id.
    A checkpoint is complete when both ema.pt and manifest.json exist (learners write the manifest last)."""
    root = Path(run)/'checkpoints'
    found = []
    for path in sorted(root.glob('*/*')) if root.exists() else []:
        if (path/'manifest.json').exists() and (path/'ema.pt').exists():
            found.append((f'{path.parent.name}/{path.name}', path, json.loads((path/'manifest.json').read_text())))
    return sorted(found, key=lambda c: (c[2]['created_at'], c[0]))


def resolve(run, initial=None):
    """(checkpoint id, ema path or None) the actor should play; see the module contract."""
    run = Path(run)
    if (run/'champion.json').exists():
        checkpoint = json.loads((run/'champion.json').read_text())['checkpoint']
        return checkpoint, run/'checkpoints'/checkpoint/'ema.pt'
    found = checkpoints(run)
    if found:
        return found[-1][0], found[-1][1]/'ema.pt'
    return ('initial', Path(initial)) if initial else ('fresh', None)


class Position:
    """hexcrop.encode_game's view of a non-terminal native history [n, 2] without replaying it.
    A Game is built only if legal_array needs the engine fallback (`ptr`)."""
    winner = -1

    def __init__(self, history):
        n = len(history)
        self.history, self.player, self.remaining = history, ((n+1)//2) % 2, 2 if n % 2 else 1

    @property
    def ptr(self):
        self.game = Game(self.history.tolist())
        return self.game.ptr


class Evaluator(hexnet.DenseEvaluator):
    """DenseEvaluator over native int64 histories with a split submit/collect so the GPU runs one batch while
    the caller gathers the next. Predictions are fulfil-ready tuples (actions int64 [N, 2], logits float64 [N],
    q float64 [N]). Small crop-size groups are padded into the next larger size present (top-left
    placement; the network is invariant to where the crop sits in the canvas; see MERGE_CELLS) and every
    forward is capped at MAX_CELLS crop cells."""

    @torch.inference_mode()
    def submit(self, histories):
        samples = [hexcrop.encode_game(Position(h), h) for h in histories]
        groups = hexcrop.group_by_size(samples)
        sizes = sorted(groups)
        for small, large in zip(sizes, sizes[1:]):
            if len(groups[small])*(large*large-small*small) < MERGE_CELLS:
                groups[large] = groups.pop(small)+groups[large]
        chunks = []
        for size, indices in groups.items():
            step = max(1, min(self.max_batch, MAX_CELLS//(size*size)))
            for start in range(0, len(indices), step):
                chunk = indices[start:start+step]
                host = torch.empty((len(chunk), len(hexcrop.PLANES), size, size), dtype=torch.uint8, pin_memory=self.cuda)
                view = host.numpy()
                for j, i in enumerate(chunk):
                    planes = samples[i].planes
                    if planes.shape[-1] == size:
                        view[j] = planes
                    else:
                        view[j] = 0
                        view[j, :, :planes.shape[-1], :planes.shape[-1]] = planes
                x = host.to(self.device, non_blocking=True)
                x = x.to(memory_format=self.memory_format, dtype=torch.bfloat16 if self.cuda else torch.float32)
                with torch.autocast(self.device.type, torch.bfloat16, enabled=self.cuda):
                    out = self.model(x, x[:, 3:4], aux=False)
                packed = torch.cat((out['policy'], out['far'][:, None], out['value_logit'][:, None]), 1)
                result = torch.empty(packed.shape, dtype=torch.float32, pin_memory=self.cuda)
                chunks.append((size, chunk, result.copy_(packed, non_blocking=True)))
        event = torch.cuda.Event() if self.cuda else None
        if event:
            event.record()
        return samples, chunks, event

    def collect(self, handle):
        samples, chunks, event = handle
        if event:
            event.synchronize()
        result = [None]*len(samples)
        for size, chunk, packed in chunks:
            packed = packed.numpy()
            if not np.isfinite(packed).all():
                raise FloatingPointError('Nonfinite dense model predictions')
            for row, i in zip(packed, chunk):
                s = samples[i]
                cells = s.cells if s.size == size else np.where(s.cells >= 0, s.cells//s.size*size+s.cells % s.size, -1)
                logits = row[np.maximum(cells, 0)].astype(np.float64)
                if s.far:
                    logits[cells < 0] = row[-2]-np.log(s.far)
                result[i] = (s.actions, logits, np.full(len(logits), np.tanh(row[-1]/2), np.float64))
        return result

    def evaluate(self, histories):
        return self.collect(self.submit(histories))


class Model:
    """A frozen evaluator, its identity and its position cache; trees of one Model share GPU batches."""

    def __init__(self, net, sha, checkpoint, device, max_batch, cache_positions):
        self.sha, self.checkpoint, self.config = sha, checkpoint, net.config
        self.evaluator = Evaluator(net, device, sha, max_batch)
        self.cache = EvaluationCache(cache_positions)

    def tree(self, history, seed, tactics):
        return NeuralSearch(self.evaluator, self.sha, history, seed, self.cache, tactics)


def load(run, config, initial=None, source=None):
    """Model for `resolve(run, initial)` (or an explicit (checkpoint, path) `source`)."""
    checkpoint, path = source or resolve(run, initial)
    if path is None:
        torch.manual_seed(config.seed)
        net = hexnet.HexNet(hexnet.HexNetConfig(**asdict(config.model)))
        sha = hexnet.model_digest(net)
    else:
        net, sha = hexnet.load_model(path), digest(path)
    return Model(net, sha, checkpoint, config.device, config.actor.leaf_batch, config.actor.cache_positions)


def position_key(history):
    """Transposition key covering every input of hexcrop.encode: the stones of each colour (hence the ply
    count, side to move and remaining placements), the stone placed earlier this turn (plane 6) and the
    opponent's previous turn as a set (plane 7). Two placement orders within one turn therefore share a
    key once the turn is complete, but never while it is still in progress."""
    n = len(history)
    order = np.lexsort((history[:, 1], history[:, 0], COLOR[:n]))
    start = n if n % 2 or n == 0 else n-1
    turn = history[start:n]
    previous = history[max(0, start-2):start]
    previous = previous[np.lexsort((previous[:, 1], previous[:, 0]))]
    return n, history[order].tobytes(), turn.tobytes(), previous.tobytes()


def root_value(result, player):
    """Side-to-move value of a finished search: exact +-1, else the visit-weighted mean child value."""
    if result['exact_winner'] >= 0:
        return 1. if result['exact_winner'] == player else -1.
    visits = result['visits']
    return float(visits @ result['values'])/int(visits.sum())


class Engine:
    """Continuous, pipelined batched Gumbel search over slots.

    A slot exposes `tree` (NeuralSearch whose model owns the evaluator), `model`, `budget`, `samples`, and
    `searched(result) -> bool` which plays the move(s) and returns True after preparing its next search
    (new tree/budget allowed) or False when its game is over. `step()` visits slots round-robin, draining
    each tree's ready leaf requests until `leaf_batch` evaluations are pending, launches them (one forward
    set per model, one evaluation per distinct position), then collects and fulfils the batch launched by
    the previous step, so the GPU works on one batch while the trees produce the next. It returns the
    slots whose games finished.
    """

    def __init__(self, leaf_batch):
        self.leaf_batch, self.slots, self.cursor, self.inflight = leaf_batch, [], 0, []
        self.evals = self.calls = self.hits = self.searches = 0

    def add(self, slot):
        checked(native.hxg_begin(slot.tree.ptr, slot.budget, slot.samples))
        self.slots.append(slot)

    def step(self):
        pending, count, done, progress = {}, 0, [], False
        slots = self.slots
        for _ in range(len(slots)):
            if count >= self.leaf_batch:
                break
            slot = slots[self.cursor % len(slots)]
            self.cursor += 1
            ptr = slot.tree.ptr
            while True:
                if native.hxg_completed(ptr) >= slot.budget:
                    progress = True
                    self.searches += 1
                    if not slot.searched(slot.tree.result(0, 0, 0, 0)):
                        done.append(slot)
                        break
                    ptr = slot.tree.ptr
                    checked(native.hxg_begin(ptr, slot.budget, slot.samples))
                    continue
                request = native.hxg_next(ptr)
                if request == 0:
                    break
                if request == -1:
                    progress = True
                    continue
                if request < 0:
                    checked(False)
                size = native.hxg_history(ptr, request, None)
                history = np.empty((size, 2), np.int64)
                native.hxg_history(ptr, request, history.ctypes.data)
                model = slot.model
                key = position_key(history)
                cached = model.cache.get(key)
                if cached is not None:
                    checked(native.hxg_fulfill(ptr, request, *cached, len(cached[0])))
                    self.hits += 1
                    progress = True
                else:
                    pending.setdefault(model, {}).setdefault(key, [history]).append((ptr, request))
                    count += 1
        launched = []
        for model, positions in pending.items():
            keys = list(positions)
            launched.append((model, positions, keys, model.evaluator.submit([positions[k][0] for k in keys])))
            self.calls += 1
            self.evals += len(keys)
        for model, positions, keys, handle in self.inflight:
            for key, prediction in zip(keys, model.evaluator.collect(handle)):
                for ptr, request in positions[key][1:]:
                    checked(native.hxg_fulfill(ptr, request, *prediction, len(prediction[0])))
                model.cache.put(key, prediction)
        if not launched and not self.inflight and not progress and self.slots:
            raise RuntimeError('Native scheduler stalled without pending evaluations')
        self.inflight = launched
        if done:
            finished = set(map(id, done))
            self.slots = [s for s in self.slots if id(s) not in finished]
        return done


class SelfPlayGame:
    """One self-play game and its tree; records a row, root value and search kind for every placement."""

    def __init__(self, model, settings, seed):
        self.model, self.settings, self.seed = model, settings, seed
        self.rng = np.random.default_rng(seed)
        self.random_plies = int(round(self.rng.exponential(settings.opening_random_plies))) if settings.opening_random_plies > 0 else 0
        self.tree = model.tree((), seed, settings.tactics)
        self.game, self.moves, self.rows, self.values, self.full = Game(), [], [], [], []
        self.plan()

    def plan(self):
        s = self.settings
        self.is_full = bool(self.rng.random() < s.full_fraction)
        self.budget = s.full_sims if self.is_full else s.cheap_sims
        self.samples = s.root_samples if self.is_full else min(s.root_samples, s.cheap_sims)

    def searched(self, result):
        game, actions = self.game, result['actions']
        player, ply = game.player, len(self.moves)
        row = dict(ply=ply, player=player, remaining=game.remaining,
                   legal_sha256=hashlib.sha256(actions.tobytes()).hexdigest(), policy=None)
        policy = result['policy']
        if self.is_full:
            if not np.isclose(policy.sum(), 1, atol=1e-6) or np.any(policy < 0):
                raise ValueError('Search policy is not a distribution')
            row['policy'] = policy.astype(np.float32)
        self.rows.append(row)
        self.values.append(root_value(result, player))
        self.full.append(self.is_full)
        if ply < self.random_plies:
            action = actions[self.rng.choice(len(policy), p=policy/policy.sum())].tolist()
        else:
            action = result['action']
        q, r = int(action[0]), int(action[1])
        game.play(q, r)
        self.tree.advance((q, r))
        self.moves.append([q, r])
        if game.winner >= 0 or len(self.moves) >= self.settings.max_plies:
            return False
        self.plan()
        return True

    def episode(self):
        """(episode, rows without `game`) after closing the native objects."""
        winner = self.game.winner
        self.game.close(); self.tree.close()
        return dict(moves=self.moves, winner=winner, reason='six-in-a-row' if winner >= 0 else 'cap',
                    opening_plies=min(self.random_plies, len(self.moves)), actor=self.model.sha,
                    root_values=self.values, full_search=self.full), self.rows


def shard_name():
    return f'{time.time_ns()//1_000_000:013d}{os.getpid() % 1000:03d}'


def worker(args):
    run = Path(args.run)
    config = dense_config.load(run)
    settings = config.actor
    torch.backends.cudnn.benchmark = False
    status_path = run/('actor-status.json' if args.worker == 0 else f'actor-status-{args.worker}.json')
    entropy = np.random.SeedSequence([config.seed, args.worker, time.time_ns() % 2**63]).entropy
    seeds = np.random.SeedSequence(entropy)
    model = load(run, config, args.initial_model)
    log_event(run, 'actor', 'info', f'worker {args.worker} playing {model.checkpoint} ({model.sha[:12]})'
              + (' - FRESH UNTRAINED NETWORK' if model.checkpoint == 'fresh' else ''), process=args.worker)
    print(f'Worker {args.worker}: {model.checkpoint} {model.sha[:12]}', flush=True)
    engine = Engine(settings.leaf_batch)
    state = dict(games_completed=0, positions=0, shards_written=0, terminal=0, plies=0, error=None)
    episodes, rows, started = [], [], 0
    window = deque([(time.perf_counter(), 0, 0)])
    since = dict(time=time.perf_counter(), positions=0, evals=0)

    def status(stage):
        now = time.perf_counter()
        window.append((now, state['positions'], engine.evals))
        while len(window) > 2 and now-window[1][0] > 60:
            window.popleft()
        t, p, e = window[0]
        g = state['games_completed']
        write_json(status_path, dict(
            stage=stage, updated_at=time.time(), checkpoint=model.checkpoint, actor_sha256=model.sha,
            games_completed=g, games_total=args.games, positions=state['positions'], active_games=len(engine.slots),
            placements_per_second=(state['positions']-p)/max(1e-9, now-t), evals_per_second=(engine.evals-e)/max(1e-9, now-t),
            mean_batch=engine.evals/max(1, engine.calls), terminal_fraction=state['terminal']/g if g else None,
            mean_plies=state['plies']/g if g else None, shards_written=state['shards_written'], error=state['error']))

    def publish():
        nonlocal model
        now = time.perf_counter()
        name = shard_name()
        actors = sorted({e['actor'] for e in episodes})
        identity = dict(actor_sha256=model.sha, actors=actors, checkpoint=model.checkpoint, process=args.worker,
                        pid=os.getpid(), seed_entropy=str(entropy), model=asdict(model.config),
                        actor=asdict(settings), value_targets='not stored; derive from episode root_values and winner')
        dense_data.write_shard(run/'shards'/name, identity, episodes, rows)
        state['shards_written'] += 1
        games = len(episodes); terminal = sum(e['winner'] >= 0 for e in episodes)
        elapsed = now-since['time']
        fields = dict(shard=name, games=games, rows=len(rows), terminal_fraction=terminal/games,
                      mean_plies=sum(len(e['moves']) for e in episodes)/games,
                      placements_per_second=(state['positions']-since['positions'])/elapsed,
                      evals_per_second=(engine.evals-since['evals'])/elapsed, process=args.worker)
        log_event(run, 'actor', 'shard', f'shard {name}: {games} games, {len(rows)} rows', **fields)
        print(json.dumps(fields), flush=True)
        since.update(time=now, positions=state['positions'], evals=engine.evals)
        episodes.clear(); rows.clear()
        if resolve(run, args.initial_model)[0] != model.checkpoint:
            model = load(run, config, args.initial_model)
            log_event(run, 'actor', 'info', f'worker {args.worker} switched to {model.checkpoint} ({model.sha[:12]}); '
                      'games in progress finish with the previous model', process=args.worker)

    try:
        last = 0.
        while True:
            while len(engine.slots) < settings.games_in_flight and (args.games is None or started < args.games):
                engine.add(SelfPlayGame(model, settings, seeds.spawn(1)[0].generate_state(1, np.uint64)[0].item()))
                started += 1
            if not engine.slots:
                break
            before = engine.searches
            for slot in engine.step():
                episode, items = slot.episode()
                episodes.append(episode)
                rows.extend(dict(r, game=len(episodes)-1) for r in items)
                state['games_completed'] += 1; state['terminal'] += episode['winner'] >= 0; state['plies'] += len(episode['moves'])
                if len(episodes) >= settings.shard_games:
                    publish()
            state['positions'] += engine.searches-before
            if time.perf_counter()-last >= 2:
                status('playing'); last = time.perf_counter()
        if episodes:
            publish()
        status('finished')
    except BaseException as error:
        state['error'] = f'{type(error).__name__}: {error}'
        status('error')
        log_event(run, 'actor', 'error', state['error'], process=args.worker)
        raise


def supervise(args):
    """Run --processes workers as subprocesses of this script; restart a crashed worker after logging it."""
    run = Path(args.run)
    dense_config.load(run)
    command = [sys.executable, str(Path(__file__).resolve()), '--run', str(run)]
    command += ['--games', str(args.games)] if args.games is not None else []
    command += ['--initial-model', str(args.initial_model)] if args.initial_model else []
    spawn = lambda k: subprocess.Popen(command+['--worker', str(k)])
    workers = {k: spawn(k) for k in range(args.processes)}
    log_event(run, 'actor', 'info', f'supervisor started {args.processes} workers')
    try:
        while workers:
            time.sleep(1)
            for k, process in list(workers.items()):
                code = process.poll()
                if code == 0:
                    del workers[k]
                elif code is not None:
                    log_event(run, 'actor', 'error', f'worker {k} exited with code {code}; restarting in 10 s', process=k)
                    time.sleep(10)
                    workers[k] = spawn(k)
    finally:
        for process in workers.values():
            process.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', required=True)
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--games', type=int, default=None, help='games per process (default: endless)')
    parser.add_argument('--initial-model', help='hexnet checkpoint used while the run has no checkpoint')
    parser.add_argument('--worker', type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is None:
        supervise(args)
    else:
        worker(args)


if __name__ == '__main__':
    main()
