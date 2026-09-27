"""Dense self-play actor: many native Gumbel trees batched continuously onto one DenseEvaluator.

Run layout: dense_config. The actor plays the ema.pt of champion.json's checkpoint, else of the newest complete
checkpoint, else --initial-model, else a fresh HexNet(config.model) seeded with config.seed. Shards are named
<ms:013d><pid%1000:03d>; their rows carry no value targets (target null, weight 0): the learner derives them
from episode root_values and winner. A game with a searched position wider than the largest crop ends capped with
reason 'span' and an error event.

Scheduling: every game owns one persistent NeuralSearch tree. Engine keeps all trees searching at once and
starts a tree's next search as soon as its previous one finishes, so full (`full_sims`) and cheap
(`cheap_sims`) searches share every GPU batch and no lockstep tail waits on the slowest tree. The champion
is re-read after each shard: games in progress finish with the evaluator they started with, new games use
the new one; a shard may therefore mix actors (identity `actors`; `actor_sha256` is the newest).

Historical opponents (ActorSettings.historical_*): up to round(historical_fraction * games_in_flight) games in
flight pit the champion, alternating colours, against a frozen rated checkpoint (`Historical`); only the
champion's plies become training rows (dense_data.trained). Settings flags given to the actor override config.json
for its processes and are recorded in each shard's identity.
"""
import argparse
from collections import deque
from dataclasses import asdict, replace
import json
import math
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
from dense_config import log_event
from hexo import Game
from klent import digest
from neural_search import EvaluationCache, NeuralSearch, checked, native
from train import write_json

# Crop cells per forward: 48 positions at 48x48, proportionally fewer for larger crops (~0.67 GB per b6c96
# process). A crop-size group is padded into the next larger size present when that adds fewer than
# MERGE_CELLS cells: b6c96 bf16 forwards cost about 8 ms of launch overhead plus 0.23 us per cell on an
# RTX 3070 Ti.
MAX_CELLS, MERGE_CELLS = 48*48*48, 32768
COLOR = ((np.arange(1 << 16)+1)//2) % 2
METRICS_SECONDS = 30.
PRIOR_GAMES = 16.  # weight, in games, of the Elo prediction when PFSP blends in a recorded score
BLOCKS = 2          # historical opponents in flight at once, about: blocks of target/BLOCKS games per opponent
METRICS = ('positions', 'games_completed', 'placements_per_second', 'evals_per_second', 'mean_batch',
           'terminal_fraction', 'mean_plies', 'checkpoint')


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


def expected(a, b):
    """Bradley-Terry expected score of Elo `a` against Elo `b`."""
    return 1/(1+10**((b-a)/400))


def opponent_weights(league, champion, weighting):
    """{checkpoint id: sampling weight} over the league's rated checkpoints other than `champion`. 'uniform': 1 each.
    'pfsp' (prioritised fictitious self-play, hard-opponent form): (1-p)^2, p the champion's expected score against
    the checkpoint: the Elo prediction (1/2 while the champion is unrated) blended with the champion's recorded
    score in the league matrix (caps half a point) as (PRIOR_GAMES*p_elo + points)/(PRIOR_GAMES + games)."""
    checkpoints = league.get('checkpoints', [])
    rated = {c['id']: c['elo'] for c in checkpoints if c.get('elo') is not None and c['id'] != champion}
    if weighting == 'uniform':
        return dict.fromkeys(rated, 1.)
    if weighting != 'pfsp':
        raise ValueError(f'Unknown historical weighting {weighting!r}')
    mine = next((c.get('elo') for c in checkpoints if c['id'] == champion), None)
    row, weights = league.get('matrix', {}).get(champion, {}), {}
    for k, elo in rated.items():
        cell = row.get(k, {})
        points = cell.get('wins', 0)+cell.get('capped', 0)/2
        p = (PRIOR_GAMES*(.5 if mine is None else expected(mine, elo))+points)/(PRIOR_GAMES+cell.get('games', 0))
        weights[k] = (1-p)**2
    return weights


def draw_pool(weights, size, rng):
    """Up to `size` distinct ids drawn without replacement with probability proportional to weight."""
    ids = [k for k, w in weights.items() if w > 0]
    if not ids:
        return []
    w = np.array([weights[k] for k in ids])
    return rng.choice(ids, min(size, len(ids)), replace=False, p=w/w.sum()).tolist()


def opponent_block(weights, block, rng):
    """`block` consecutive historical games against one opponent drawn with probability proportional to weight."""
    ids = list(weights)
    w = np.array([weights[k] for k in ids])
    return [ids[rng.choice(len(ids), p=w/w.sum())]]*block


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
    q float64 [N]), or None for a position wider than the largest crop (hexcrop.SpanError). Small crop-size
    groups are padded into the next larger size present (top-left placement; the network is invariant to where
    the crop sits in the canvas; see MERGE_CELLS) and every forward is capped at MAX_CELLS crop cells."""

    @torch.inference_mode()
    def submit(self, histories):
        samples, groups = [], {}
        for i, h in enumerate(histories):
            try:
                samples.append(hexcrop.encode_game(Position(h), h))
            except hexcrop.SpanError:
                samples.append(None)
                continue
            groups.setdefault(samples[i].size, []).append(i)
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

    A slot exposes `tree` (NeuralSearch whose model owns the evaluator), `model`, `budget`, `samples`, `reason`
    (None) and `searched(result) -> bool` which plays the move(s) and returns True after preparing its next
    search (new tree/budget allowed) or False when its game is over. `step()` visits slots round-robin, draining
    each tree's ready leaf requests until `leaf_batch` evaluations are pending, launches them (one forward
    set per model, one evaluation per distinct position), then collects and fulfils the batch launched by
    the previous step, so the GPU works on one batch while the trees produce the next. It returns the
    slots whose games finished, including slots stopped because a searched position could not be encoded
    (`reason` set to 'span'; none of their requests is fulfilled afterwards).
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
                    pending.setdefault(model, {}).setdefault(key, [history]).append((slot, ptr, request))
                    count += 1
        launched = []
        for model, positions in pending.items():
            keys = list(positions)
            launched.append((model, positions, keys, model.evaluator.submit([positions[k][0] for k in keys])))
            self.calls += 1
            self.evals += len(keys)
        stopped = set()
        for model, positions, keys, handle in self.inflight:
            for key, prediction in zip(keys, model.evaluator.collect(handle)):
                if prediction is None:
                    stopped.update(slot for slot, _, _ in positions[key][1:])
                    continue
                for slot, ptr, request in positions[key][1:]:
                    if slot not in stopped:
                        checked(native.hxg_fulfill(ptr, request, *prediction, len(prediction[0])))
                model.cache.put(key, prediction)
        if stopped:
            for _, positions, _, _ in launched:
                for waiting in positions.values():
                    waiting[1:] = [w for w in waiting[1:] if w[0] not in stopped]
            for slot in stopped:
                slot.reason = 'span'
                if slot in self.slots and slot not in done:
                    done.append(slot)
        if not launched and not self.inflight and not progress and self.slots:
            raise RuntimeError('Native scheduler stalled without pending evaluations')
        self.inflight = launched
        if done:
            finished = set(map(id, done))
            self.slots = [s for s in self.slots if id(s) not in finished]
        return done


class SelfPlayGame:
    """One actor game; records a row, root value and search kind for every placement. `sides[colour]` is the Model
    playing that colour: the trained model twice for self-play, else the trained model as sides[learner] and a
    frozen checkpoint `opponent` (its id) as the other, whose plies keep rows without a policy, with a null root
    value and full_search False (dense_data.trained). Each distinct model owns one tree, advanced on every
    placement; both sides use the same playout-cap randomization and opening sampling."""

    def __init__(self, sides, settings, seed, learner=0, opponent=None):
        self.sides, self.settings, self.seed, self.reason = sides, settings, seed, None
        self.learner, self.opponent = learner, opponent
        self.rng = np.random.default_rng(seed)
        self.random_plies = int(round(self.rng.exponential(settings.opening_random_plies))) if settings.opening_random_plies > 0 else 0
        self.trees = {model: model.tree((), seed+k, settings.tactics) for k, model in enumerate(dict.fromkeys(sides))}
        self.game, self.moves, self.rows, self.values, self.full = Game(), [], [], [], []
        self.plan()

    @property
    def model(self):
        return self.sides[self.game.player]

    @property
    def tree(self):
        return self.trees[self.model]

    def plan(self):
        s = self.settings
        self.is_full = bool(self.rng.random() < s.full_fraction)
        self.budget = s.full_sims if self.is_full else s.cheap_sims
        self.samples = s.root_samples if self.is_full else min(s.root_samples, s.cheap_sims)

    def searched(self, result):
        game, actions = self.game, result['actions']
        player, ply = game.player, len(self.moves)
        trained = self.sides[player] is self.sides[self.learner]
        row = dict(ply=ply, player=player, remaining=game.remaining, legal_sha256=dense_data.legal_digest(actions),
                   policy=None)
        policy = result['policy']
        if self.is_full and trained:
            if not np.isclose(policy.sum(), 1, atol=1e-6) or np.any(policy < 0):
                raise ValueError('Search policy is not a distribution')
            row['policy'] = policy.astype(np.float32)
        self.rows.append(row)
        self.values.append(root_value(result, player) if trained else None)
        self.full.append(self.is_full and trained)
        if ply < self.random_plies:
            action = actions[self.rng.choice(len(policy), p=policy/policy.sum())].tolist()
        else:
            action = result['action']
        q, r = int(action[0]), int(action[1])
        game.play(q, r)
        for tree in self.trees.values():
            tree.advance((q, r))
        self.moves.append([q, r])
        if game.winner >= 0 or len(self.moves) >= self.settings.max_plies:
            return False
        self.plan()
        return True

    def episode(self):
        """(episode, rows without `game`) after closing the native objects."""
        winner = self.game.winner
        self.game.close()
        for tree in self.trees.values():
            tree.close()
        return dict(moves=self.moves, winner=winner, reason=self.reason or ('six-in-a-row' if winner >= 0 else 'cap'),
                    opening_plies=min(self.random_plies, len(self.moves)), actor=self.sides[self.learner].sha,
                    actors={str(c): m.sha for c, m in enumerate(self.sides)}, opponent=self.opponent,
                    root_values=self.values, full_search=self.full), self.rows


class Historical:
    """Frozen historical opponents of one actor worker. `redraw(champion)` re-reads league.json, draws
    `historical_pool` opponents (draw_pool over opponent_weights; rated checkpoints with an ema.pt) and loads their
    ema.pt, keeping models drawn again; `next()` returns the opponent Model of the next historical game from blocks
    of about target/BLOCKS consecutive games per opponent, so few opponent models share the Engine at once. `target`
    is the number of historical games to keep in flight."""

    def __init__(self, run, config, rng):
        self.run, self.config, self.rng = Path(run), config, rng
        s = config.actor
        self.target = round(s.games_in_flight*s.historical_fraction)
        self.block = max(1, math.ceil(self.target/BLOCKS))
        self.models, self.weights, self.plan = {}, {}, deque()

    def redraw(self, champion):
        path = self.run/'league.json'
        league = json.loads(path.read_text()) if path.exists() else {}
        weights = {k: w for k, w in opponent_weights(league, champion, self.config.actor.historical_weighting).items()
                   if (self.run/'checkpoints'/k/'ema.pt').exists()}
        pool = draw_pool(weights, self.config.actor.historical_pool, self.rng)
        self.models = {k: self.models.get(k) or load(self.run, self.config, source=(k, self.run/'checkpoints'/k/'ema.pt'))
                       for k in pool}
        self.weights = {k: weights[k] for k in pool}
        self.plan.clear()

    def next(self):
        if not self.plan:
            self.plan.extend(opponent_block(self.weights, self.block, self.rng))
        return self.models[self.plan.popleft()]


def shard_name():
    return f'{time.time_ns()//1_000_000:013d}{os.getpid() % 1000:03d}'


def worker(args):
    run = Path(args.run)
    config = dense_config.load(run)
    config = replace(config, actor=dense_config.override(config.actor, args))
    settings = config.actor
    torch.backends.cudnn.benchmark = False
    status_path = run/('actor-status.json' if args.worker == 0 else f'actor-status-{args.worker}.json')
    entropy = np.random.SeedSequence([config.seed, args.worker, time.time_ns() % 2**63]).entropy
    seeds = np.random.SeedSequence(entropy)
    model = load(run, config, args.initial_model)
    log_event(run, 'actor', 'info', f'worker {args.worker} playing {model.checkpoint} ({model.sha[:12]})'
              + (' - FRESH UNTRAINED NETWORK' if model.checkpoint == 'fresh' else ''), process=args.worker)
    print(f'Worker {args.worker}: {model.checkpoint} {model.sha[:12]}', flush=True)
    historical = Historical(run, config, np.random.default_rng(seeds.spawn(1)[0])) if settings.historical_fraction > 0 else None
    if historical:
        historical.redraw(model.checkpoint)
    engine = Engine(settings.leaf_batch)
    state = dict(published(run, args.worker), error=None)
    target = None if args.games is None else args.games+state['games_completed']
    episodes, rows, started = [], [], 0
    window = deque([(time.perf_counter(), state['positions'], 0)])
    since = dict(time=time.perf_counter(), positions=state['positions'], evals=0)

    logged = time.perf_counter()

    def status(stage):
        """Rewrite the heartbeat; append a metrics line every METRICS_SECONDS and whenever the stage is not 'playing'."""
        nonlocal logged
        now = time.perf_counter()
        window.append((now, state['positions'], engine.evals))
        while len(window) > 2 and now-window[1][0] > 60:
            window.popleft()
        t, p, e = window[0]
        g = state['games_completed']
        fields = dict(
            stage=stage, updated_at=time.time(), checkpoint=model.checkpoint, actor_sha256=model.sha,
            games_completed=g, games_total=target, positions=state['positions'], active_games=len(engine.slots),
            placements_per_second=(state['positions']-p)/max(1e-9, now-t), evals_per_second=(engine.evals-e)/max(1e-9, now-t),
            mean_batch=engine.evals/max(1, engine.calls), terminal_fraction=state['terminal']/g if g else None,
            mean_plies=state['plies']/g if g else None, shards_written=state['shards_written'], error=state['error'])
        write_json(status_path, fields)
        if stage != 'playing' or now-logged >= METRICS_SECONDS:
            logged = now
            dense_config.append_metrics(run, f'actor-{args.worker}', **{k: fields[k] for k in METRICS})

    def publish():
        nonlocal model
        now = time.perf_counter()
        name = shard_name()
        actors = sorted({e['actor'] for e in episodes})
        opponents = {}
        for e in episodes:
            if e['opponent']:
                opponents[e['opponent']] = opponents.get(e['opponent'], 0)+1
        identity = dict(actor_sha256=model.sha, actors=actors, opponents=sorted(opponents), checkpoint=model.checkpoint,
                        process=args.worker,
                        pid=os.getpid(), seed_entropy=str(entropy), model=asdict(model.config),
                        actor=asdict(settings), value_targets='not stored; derive from episode root_values and winner')
        dense_data.write_shard(run/'shards'/name, identity, episodes, rows)
        state['shards_written'] += 1
        games = len(episodes); terminal = sum(e['winner'] >= 0 for e in episodes)
        elapsed = now-since['time']
        fields = dict(shard=name, games=games, rows=len(rows), terminal_fraction=terminal/games,
                      mean_plies=sum(len(e['moves']) for e in episodes)/games,
                      placements_per_second=(state['positions']-since['positions'])/elapsed,
                      evals_per_second=(engine.evals-since['evals'])/elapsed, process=args.worker, opponents=opponents)
        log_event(run, 'actor', 'shard', f'shard {name}: {games} games, {len(rows)} rows', **fields)
        print(json.dumps(fields), flush=True)
        since.update(time=now, positions=state['positions'], evals=engine.evals)
        episodes.clear(); rows.clear()
        if resolve(run, args.initial_model)[0] != model.checkpoint:
            model = load(run, config, args.initial_model)
            log_event(run, 'actor', 'info', f'worker {args.worker} switched to {model.checkpoint} ({model.sha[:12]}); '
                      'games in progress finish with the previous model', process=args.worker)
        if historical:
            historical.redraw(model.checkpoint)

    try:
        last = 0.
        while True:
            while len(engine.slots) < settings.games_in_flight and (args.games is None or started < args.games):
                seed = seeds.spawn(1)[0].generate_state(1, np.uint64)[0].item()
                if historical and historical.models and sum(g.opponent is not None for g in engine.slots) < historical.target:
                    opponent, learner = historical.next(), started % 2
                    sides = [model, opponent] if learner == 0 else [opponent, model]
                    engine.add(SelfPlayGame(sides, settings, seed, learner, opponent.checkpoint))
                else:
                    engine.add(SelfPlayGame([model, model], settings, seed))
                started += 1
            if not engine.slots:
                break
            before = engine.searches
            for slot in engine.step():
                episode, items = slot.episode()
                if episode['reason'] == 'span':
                    log_event(run, 'actor', 'error', f'worker {args.worker}: game ended at ply {len(episode["moves"])}, '
                              f'a searched position spans more than the largest crop', process=args.worker)
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
        status('failed')
        log_event(run, 'actor', 'error', state['error'], process=args.worker)
        raise


def published(run, worker, since=0.):
    """Cumulative counts from shards written by actor worker `worker` at or after `since`; every ply has a row,
    so rows count both positions and plies."""
    totals = dict(games_completed=0, positions=0, shards_written=0, terminal=0, plies=0)
    for m in (dense_data.manifest(path) for path in dense_data.shard_dirs(run)):
        if m['identity'].get('process') == worker and m['created_at'] >= since:
            c = m['counts']
            totals['games_completed'] += c['games']; totals['positions'] += c['rows']; totals['plies'] += c['rows']
            totals['terminal'] += c['terminal_games']; totals['shards_written'] += 1
    return totals


def supervise(args):
    """Run --processes workers as subprocesses of this script; restart a crashed worker with the games it has
    not published yet."""
    run = Path(args.run)
    dense_config.load(run)
    command = [sys.executable, str(Path(__file__).resolve()), '--run', str(run)]
    command += ['--initial-model', str(args.initial_model)] if args.initial_model else []
    command += dense_config.flags(dense_config.ActorSettings, args)
    remaining = dict.fromkeys(range(args.processes), args.games)

    def spawn(k):
        games = [] if remaining[k] is None else ['--games', str(remaining[k])]
        return subprocess.Popen(command+games+['--worker', str(k)]), time.time()

    workers = {k: spawn(k) for k in range(args.processes)}
    log_event(run, 'actor', 'info', f'supervisor started {args.processes} workers')
    try:
        while workers:
            time.sleep(1)
            for k, (process, started) in list(workers.items()):
                code = process.poll()
                if code is None:
                    continue
                del workers[k]
                if code == 0:
                    continue
                if remaining[k] is not None:
                    remaining[k] -= published(run, k, started)['games_completed']
                restart = remaining[k] is None or remaining[k] > 0
                left = '' if remaining[k] is None else f' with {remaining[k]} games left'
                log_event(run, 'actor', 'error', f'worker {k} exited with code {code}'
                          + (f'; restarting in 10 s{left}' if restart else ''), process=k)
                if restart:
                    time.sleep(10)
                    workers[k] = spawn(k)
    finally:
        for process, _ in workers.values():
            process.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', required=True)
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--games', type=int, default=None, help='games per process (default: endless)')
    parser.add_argument('--initial-model', help='hexnet checkpoint used while the run has no checkpoint')
    parser.add_argument('--worker', type=int, default=None, help=argparse.SUPPRESS)
    dense_config.add_arguments(parser.add_argument_group('actor settings (override config.json)'), dense_config.ActorSettings)
    args = parser.parse_args()
    if args.worker is None:
        supervise(args)
    else:
        worker(args)


if __name__ == '__main__':
    main()
