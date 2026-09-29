"""Dense self-play actor: many native Gumbel trees batched continuously onto one DenseEvaluator.

Run layout: dense_config. The actor plays the ema.pt of the checkpoint `resolve` picks for actor.model_source:
'newest_veto' (default) actor.json's checkpoint, the newest export unless the evaluator vetoed it
(dense_eval.Evaluator.point); 'newest' the newest complete checkpoint of learner.variant; 'champion'
champion.json's checkpoint. Each falls back to champion.json, the newest complete checkpoint, --initial-model and
a fresh HexNet(config.model) seeded with config.seed, in that order. Shards are named
<ms:013d><pid%1000:03d>; their rows carry no value targets (target null, weight 0): the learner derives them
from episode root_values and winner. A game with a searched position wider than the largest crop ends capped with
reason 'span' and an error event.

Scheduling: every game owns one persistent NeuralSearch tree. Engine keeps all trees searching at once and
starts a tree's next search as soon as its previous one finishes, so full (`full_sims`) and cheap
(`cheap_sims`) searches share every GPU batch and no lockstep tail waits on the slowest tree. The model is
re-resolved after each shard ('actor_model' event on a switch): games in progress finish with the evaluator they
started with, new games use the new one; a shard may therefore mix actors (identity `actors`; `actor_sha256` is
the newest).

GPU sharing (Yield): CUDA contexts of separate processes time-slice the GPU, so every busy actor worker takes
a share from the learner. Workers therefore pause between engine steps while a learner falls behind its
pacing: they re-read the learner heartbeats every `yield_check_seconds`, pause once the lowest ratio of
samples_per_row to that learner's own samples_per_row_target (config learner.samples_per_row when a heartbeat
lacks it) among the learners that are training drops below `yield_below` and resume once it reaches
`yield_resume` (hysteresis, so the learner does not reach its own waiting point while the
actors sleep). With phase_follow they also pause while a fresh heartbeat of a phased learner (phase_rows > 0)
shows its training phase (stage 'training' or 'exporting') and play while it idles ('phase-idle'), so actors and
learner alternate on the GPU; the pacing rule alone cannot do that, since it pauses on a backlog of
(1 - yield_below) * rows_available rather than phase_rows. A missing, stale or non-training heartbeat never
pauses. Paused workers keep their trees and in-flight batch and heartbeat with stage 'paused'. Actor settings can
be overridden per process with --<setting> flags (the supervisor forwards them; shards record the effective values).

Historical opponents (ActorSettings.historical_*): up to round(historical_fraction * min(games_in_flight, --games))
games in flight pit the played model (called the champion below), alternating colours over historical games,
against a frozen rated checkpoint (`Historical`); only its plies become training rows (dense_data.trained).

Restarts (ActorSettings.restart_*): with restart_fraction > 0 each new self-play game starts with that probability
from a position of the run's restart buffer (`Restarts`, written by dense_solve); its source moves are replayed as
forced opening plies without search or rows, and the episode records origin 'restart' and its source.
"""
import argparse
from collections import OrderedDict, deque
from dataclasses import asdict, fields, replace
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
import dense_solver
import hexcrop
import hexnet
from dense_config import log_event
from hexo import Game
from klent import digest
from neural_search import HOLD, EvaluationCache, NeuralSearch, checked, native
from train import write_json

# Crop cells per forward: 48 positions at 48x48, proportionally fewer for larger crops (a b6c96 forward peaks
# near 0.14 GB; about 0.33 GB per process with the CUDA context). A crop-size group is padded into the next larger size present when that adds fewer than
# MERGE_CELLS cells: b6c96 bf16 forwards cost about 8 ms of launch overhead plus 0.23 us per cell on an
# RTX 3070 Ti.
MAX_CELLS, MERGE_CELLS = 48*48*48, 32768
COLOR = ((np.arange(1 << 16)+1)//2) % 2
METRICS_SECONDS = 30.
PRIOR_GAMES = 16.  # weight, in games, of the Elo prediction when PFSP blends in a recorded score
BLOCKS = 2          # historical opponents in flight at once, about: blocks of target/BLOCKS games per opponent
METRICS = ('positions', 'games_completed', 'placements_per_second', 'evals_per_second', 'mean_batch',
           'terminal_fraction', 'mean_plies', 'checkpoint', 'paused_seconds')
STALE_SECONDS = 120.  # learner heartbeats older than this are ignored by Yield
RESTART_SOURCE = ('shard', 'game', 'ply', 'kind', 'regret', 'plies_to_proof')  # buffer entry fields a restart records
RESTART_SHARDS = 16   # source shards whose moves Restarts keeps
CLOSE_SECONDS = 10.   # longest a finished game waits for proofs that may label its rows


def checkpoints(run):
    """Complete checkpoints as [(id, directory, manifest)] ordered by manifest created_at, then id.
    A checkpoint is complete when both ema.pt and manifest.json exist (learners write the manifest last)."""
    root = Path(run)/'checkpoints'
    found = []
    for path in sorted(root.glob('*/*')) if root.exists() else []:
        if (path/'manifest.json').exists() and (path/'ema.pt').exists():
            found.append((f'{path.parent.name}/{path.name}', path, json.loads((path/'manifest.json').read_text())))
    return sorted(found, key=lambda c: (c[2]['created_at'], c[0]))


def resolve(run, initial=None, source='champion', variant='main'):
    """(checkpoint id, ema path or None) the actor should play under model_source `source` with learner variant
    `variant`; see the module contract."""
    run = Path(run)
    if source not in ('champion', 'newest', 'newest_veto'):
        raise ValueError(f'Unknown model_source {source!r}')
    if source == 'newest_veto' and (run/'actor.json').exists():
        checkpoint = json.loads((run/'actor.json').read_text())['checkpoint']
        return checkpoint, run/'checkpoints'/checkpoint/'ema.pt'
    own = [c for c in checkpoints(run) if c[0].split('/')[0] == variant] if source == 'newest' else []
    if own:
        return own[-1][0], own[-1][1]/'ema.pt'
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


class Evaluator(hexnet.DenseEvaluator):
    """DenseEvaluator over native int64 histories with a split submit/collect so the GPU runs one batch while
    the caller gathers the next. Predictions are fulfil-ready tuples (actions int64 [N, 2], logits float64 [N],
    q float64 [N]), or None for a position wider than the largest crop (hexcrop.SpanError). Small crop-size
    groups are padded into the next larger size present (top-left placement; the network is invariant to where
    the crop sits in the canvas; see MERGE_CELLS) and every forward is capped at MAX_CELLS crop cells.
    Host staging: each handle owns one set of pinned buffers (an input and an output buffer per crop size, grown
    to the largest group seen) that collect returns to `free`, so the pipelined Engine cycles two sets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.free = []

    @torch.inference_mode()
    def submit(self, histories, legal=None):
        """Launch the forwards for `histories`; `legal[i]`, when given, is history i's native legal list."""
        samples, groups = [], {}
        for i, h in enumerate(histories):
            try:
                samples.append(hexcrop.encode_game(hexcrop.Position(h), h, actions=None if legal is None else legal[i]))
            except hexcrop.SpanError:
                samples.append(None)
                continue
            groups.setdefault(samples[i].size, []).append(i)
        sizes = sorted(groups)
        for small, large in zip(sizes, sizes[1:]):
            if len(groups[small])*(large*large-small*small) < MERGE_CELLS:
                groups[large] = groups.pop(small)+groups[large]
        chunks, staging = [], self.free.pop() if self.free else {}
        for size, indices in groups.items():
            host = hexnet.staging_buffer(staging, ('planes', size), len(indices), (len(hexcrop.PLANES), size, size),
                                         torch.uint8, self.cuda)
            result = hexnet.staging_buffer(staging, ('out', size), len(indices), (size*size+2,), torch.float32, self.cuda)
            view = host.numpy()
            for j, i in enumerate(indices):
                planes = samples[i].planes
                if planes.shape[-1] == size:
                    view[j] = planes
                else:
                    view[j] = 0
                    view[j, :, :planes.shape[-1], :planes.shape[-1]] = planes
            step = max(1, min(self.max_batch, MAX_CELLS//(size*size)))
            for start in range(0, len(indices), step):
                chunk = indices[start:start+step]
                x = host[start:start+len(chunk)].to(self.device, non_blocking=True)
                x = x.to(memory_format=self.memory_format, dtype=torch.bfloat16 if self.cuda else torch.float32)
                with torch.autocast(self.device.type, torch.bfloat16, enabled=self.cuda):
                    out = self.model(x, x[:, 3:4], aux=False)
                packed = torch.cat((out['policy'], out['far'][:, None], out['value_logit'][:, None]), 1)
                chunks.append((size, chunk, result[start:start+len(chunk)].copy_(packed, non_blocking=True)))
        # A blocking-sync event parks collect() in the driver instead of spinning a core while the GPU works.
        event = torch.cuda.Event(blocking=True) if self.cuda else None
        if event:
            event.record()
        return samples, chunks, event, staging

    def collect(self, handle):
        samples, chunks, event, staging = handle
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
        self.free.append(staging)
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
    """Model for `resolve` under config.actor.model_source (or an explicit (checkpoint, path) `source`)."""
    checkpoint, path = source or resolve(run, initial, config.actor.model_source, config.learner.variant)
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
    """Side-to-move value of a finished search: the solver's exact value when it proved one (result `proven`), else
    exact +-1 when the tree root is exact, else the visit-weighted mean child value."""
    if result.get('proven'):
        return float(result['proven'])
    if result['exact_winner'] >= 0:
        return 1. if result['exact_winner'] == player else -1.
    visits = result['visits']
    return float(visits @ result['values'])/int(visits.sum())


class Engine:
    """Continuous, pipelined batched Gumbel search over slots.

    A slot exposes `tree` (NeuralSearch whose model owns the evaluator), `model`, `budget`, `samples`, `solver`
    (dense_solver.Budgets of the side to move, or None), `reason` (None) and `searched(result) -> bool` which plays
    the move(s) and returns True after preparing its next search (new tree/budget allowed) or False when its game
    is over. `step()` visits slots round-robin, draining each tree's ready leaf requests until `leaf_batch`
    evaluations are pending, launches them (one forward set per model, one evaluation per distinct position), then
    collects and fulfils the batch launched by the previous step, so the GPU works on one batch while the trees
    produce the next. It returns the slots whose games finished, including slots stopped because a searched
    position could not be encoded (`reason` set to 'span'; none of their requests is fulfilled afterwards).

    Solver (dense_solver): a slot whose budgets are active under `schedule` (dense_solver.active) gets a dense_solver.Plan on the Engine's Solver (created on
    first use with `schedule`, default fixed budgets; `solver_async` picks its backend). A search that submits
    queries leaves its slot until the next visit, which consumes the verdicts before the search goes on; the root
    verdict is consumed before `searched`, whose result then carries `proven`, `proof_turns`, `solver_nodes`,
    `solver_budget` and `pruned` (and the proof's action) (dense_solver.Plan.finish). Under adaptive budgets a
    slot whose verdict is not in is skipped for the step (the other slots build the batch), and a step in which
    every slot waits and no batch is in flight sleeps briefly instead of raising the stall error. Each step
    reports its wall time and its collect time to the Solver; a finished slot's Plan closes with the game's
    history (dense_solver.Plan.close; with solver_follow it calls slot.label). A finished game whose Plan still
    waits for proofs that may label its rows (Plan.pending) stays in `closing` until they are in, at most
    CLOSE_SECONDS, and is returned by the step that closes it; run until both `slots` and `closing` are empty.
    close() stops the Solver.
    """

    def __init__(self, leaf_batch, solver_async=True, schedule=None):
        self.leaf_batch, self.slots, self.cursor, self.inflight = leaf_batch, [], 0, []
        self.evals = self.calls = self.hits = self.searches = 0
        self.solver_async, self.schedule = solver_async, schedule or dense_solver.Schedule()
        self.solver, self.plans, self.closing = None, {}, []

    def begin(self, slot):
        """Start the slot's next search; True when it must wait for solver verdicts until the next visit."""
        checked(native.hxg_begin(slot.tree.ptr, slot.budget, slot.samples))
        plan = self.plans.get(id(slot))
        if plan is None and dense_solver.active(slot.solver, self.schedule):
            self.solver = self.solver or dense_solver.Solver(self.schedule, self.solver_async)
            plan = self.plans[id(slot)] = dense_solver.Plan(self.solver)
        return plan is not None and plan.begin(slot)

    def add(self, slot):
        self.begin(slot)
        self.slots.append(slot)

    def drain(self):
        """Account the solver queries of finished games that are still running (dense_solver.Solver.drain)."""
        if self.solver:
            self.solver.drain()

    def close(self):
        if self.solver:
            self.solver.close()
            self.solver = None

    def step(self):
        pending, count, done, progress, deferred = {}, 0, [], False, False
        started = time.perf_counter()
        slots = self.slots
        for _ in range(len(slots)):
            if count >= self.leaf_batch:
                break
            slot = slots[self.cursor % len(slots)]
            self.cursor += 1
            plan = self.plans.get(id(slot))
            if plan and not plan.ready(slot):
                deferred = True
                continue
            ptr = slot.tree.ptr
            while True:
                request = native.hxg_next(ptr)
                if request == HOLD:
                    progress = True
                    if plan.hold(slot):
                        break
                    continue
                if request == 0:
                    if native.hxg_completed(ptr) < slot.budget:
                        break
                    progress = True
                    result = slot.tree.result(0, 0, 0, 0)
                    if plan and not plan.finish(slot, result):
                        deferred = True
                        break
                    self.searches += 1
                    if not slot.searched(result):
                        done.append(slot)
                        break
                    ptr = slot.tree.ptr
                    waiting = self.begin(slot)
                    plan = self.plans.get(id(slot))
                    if waiting:
                        break
                    continue
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
                    if key not in pending.setdefault(model, {}):
                        legal = np.empty((native.hxg_legal(ptr, request, None), 2), np.int64)
                        native.hxg_legal(ptr, request, legal.ctypes.data)
                        pending[model][key] = [(history, legal)]
                    pending[model][key].append((slot, ptr, request))
                    count += 1
        launched = []
        for model, positions in pending.items():
            keys = list(positions)
            histories, legal = zip(*(positions[k][0] for k in keys))
            launched.append((model, positions, keys, model.evaluator.submit(histories, legal)))
            self.calls += 1
            self.evals += len(keys)
        stopped, collecting = set(), time.perf_counter()
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
        collected = time.perf_counter()
        if not launched and not self.inflight and not progress and self.slots:
            if not deferred:
                raise RuntimeError('Native scheduler stalled without pending evaluations')
            self.solver.idle()
        self.inflight = launched
        if done:
            finished = set(map(id, done))
            self.slots = [s for s in self.slots if id(s) not in finished]
            self.closing += [(slot, self.plans.pop(id(slot), None), time.perf_counter()+CLOSE_SECONDS) for slot in done]
        done = []
        for entry in list(self.closing):
            slot, plan, deadline = entry
            if plan and plan.pending() and time.perf_counter() < deadline:
                continue
            self.closing.remove(entry)
            if plan:
                plan.close(slot, slot.tree.history)
            done.append(slot)
        if self.closing and not self.slots and not done:
            self.solver.idle()
        if self.solver:
            self.solver.tick((time.perf_counter()-started)*1000, (collected-collecting)*1000)
        return done


class SelfPlayGame:
    """One actor game; records a row, root value and search kind for every placement. `sides[colour]` is the Model
    playing that colour: the trained model twice for self-play, else the trained model as sides[learner] and a
    frozen checkpoint `opponent` (its id) as the other, whose plies keep rows without a policy, with a null root
    value and full_search False (dense_data.trained). Each distinct model owns one tree, advanced on every
    placement; both sides use the same playout-cap randomization, opening sampling and solver budgets (`solver`,
    from the settings' solver_* fields). With the solver active every row records `proven`, `proof_turns`,
    `solver_nodes` and `solver_budget` (Engine), a proof's move is played even on an opening ply, the episode
    records `solver` (dense_solver.record) and `label` marks rows a proof decided after they were searched.
    `restart` (entry, moves) starts the game after `moves`, the forced opening of a restart buffer `entry`: those
    plies get no row, a null root value and full_search False, no opening ply is sampled after them, and the
    episode records origin 'restart' and `restart` (the entry's RESTART_SOURCE fields); other games record origin
    'selfplay'.
    With adjudicate_proven a proven search (+1: the side to move wins, -1: every candidate it kept loses) plays its
    move and ends the game there (`adjudicate`)."""

    def __init__(self, sides, settings, seed, learner=0, opponent=None, restart=None):
        self.sides, self.settings, self.seed, self.reason, self.adjudicated = sides, settings, seed, None, None
        self.solver, self.schedule = dense_solver.Budgets.of(settings), dense_solver.Schedule.of(settings)
        self.learner, self.opponent = learner, opponent
        self.rng = np.random.default_rng(seed)
        self.restart, forced = (None, []) if restart is None else (restart[0], [[int(q), int(r)] for q, r in restart[1]])
        self.random_plies = int(round(self.rng.exponential(settings.opening_random_plies))) \
            if settings.opening_random_plies > 0 and restart is None else 0
        self.trees = {model: model.tree([tuple(m) for m in forced], seed+k, settings.tactics)
                      for k, model in enumerate(dict.fromkeys(sides))}
        self.game, self.moves, self.rows = Game(forced), forced, []
        self.values, self.full = [None]*len(forced), [False]*len(forced)
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
        trained = self.opponent is None or player == self.learner
        row = dict(ply=ply, player=player, remaining=game.remaining, legal_sha256=dense_data.legal_digest(actions),
                   policy=None)
        policy = result['policy']
        if self.is_full and trained:
            if not np.isclose(policy.sum(), 1, atol=1e-6) or np.any(policy < 0):
                raise ValueError('Search policy is not a distribution')
            row['policy'] = policy.astype(np.float32)
        if dense_solver.active(self.solver, self.schedule):
            row.update(proven=result['proven'], proof_turns=result['proof_turns'], solver_nodes=result['solver_nodes'],
                       solver_budget=result['solver_budget'])
            if result.get('proof_action'):
                row['proof_action'] = result['proof_action']
        self.rows.append(row)
        self.values.append(root_value(result, player) if trained else None)
        self.full.append(self.is_full and trained)
        if ply < self.random_plies and result.get('proven', 0) <= 0:
            action = actions[self.rng.choice(len(policy), p=policy/policy.sum())].tolist()
        else:
            action = result['action']
        q, r = int(action[0]), int(action[1])
        game.play(q, r)
        for tree in self.trees.values():
            tree.advance((q, r))
        self.moves.append([q, r])
        if game.winner >= 0:
            return False
        if self.settings.adjudicate_proven and result.get('proven'):
            self.adjudicate(player if result['proven'] > 0 else 1-player, result.get('proof'))
            return False
        if len(self.moves) >= self.settings.max_plies:
            return False
        self.plan()
        return True

    def adjudicate(self, winner, proof):
        """End the game as a proven win of `winner` (reason 'proven'). With proven_line_rows and the winner's
        `proof`, first play the certificate's forced line to six in a row (forced_line), each placement with a row
        and no search: no policy, the exact value, `line` True; the line stops at the ply cap or at a position
        wider than the largest crop. The episode records `adjudicated` {ply, winner,
        line_plies: the placements of that line, played or not}."""
        full = proof and dense_solver.Proof(proof.base, dict(nodes=proof.nodes, root=proof.root))  # past its first turn
        line = self.forced_line(full) if full else []
        ply = len(self.moves)
        if self.settings.proven_line_rows:
            trained = lambda p: self.opponent is None or p == self.learner
            for (q, r), turns in line:
                if len(self.moves) >= self.settings.max_plies:
                    break
                game = self.game
                player = game.player
                history = np.asarray(self.moves, np.int64).reshape(-1, 2)
                try:
                    hexcrop.encode_game(game, history)
                except hexcrop.SpanError:
                    break   # wider than the largest crop: the learner could not encode the row
                legal = hexcrop.legal_array(game, history)
                proven = 1 if player == winner else -1
                self.rows.append(dict(ply=len(self.moves), player=player, remaining=game.remaining,
                                      legal_sha256=dense_data.legal_digest(legal), policy=None, proven=proven,
                                      proof_turns=turns, solver_nodes=0, solver_budget=0, line=True))
                if proven > 0:
                    self.rows[-1]['proof_action'] = full.action(self.moves)
                self.values.append(float(proven) if trained(player) else None)
                self.full.append(False)
                game.play(q, r)
                self.moves.append([q, r])
        self.reason, self.adjudicated = 'proven', dict(ply=ply, winner=winner, line_plies=len(line))

    def forced_line(self, proof):
        """[((q, r), proof_turns)] from the current position to the winner's six in a row along `proof`: the
        attacker's certificate stones, the defender's first covered reply, and after an unstoppable node any legal
        stones off the attacker's threats; the line stops early (never expected) where the certificate gives no
        move."""
        history, line = [tuple(m) for m in self.moves], []
        game = Game(history)
        try:
            while game.winner < 0:
                _, move, node, _ = proof.walk(history)
                if move is None:
                    break
                stones = move[0] or proof.reply(history)
                if stones is None:
                    threats = {tuple(c) for t in node.get('threats', ()) for c in t}
                    stones = [next(tuple(m) for m in game.legal_moves() if tuple(m) not in threats)]
                line.append((stones[0], move[1]))
                game.play(*stones[0])
                history.append(stones[0])
        finally:
            game.close()
        return line

    def label(self, ply, proven, turns, proof_action=None):
        """Record a proof's verdict (+1 / -1: the side to move wins / loses) on the row of `ply` unless it has one;
        1 when set."""
        index = ply-(len(self.moves)-len(self.rows))
        row = self.rows[index] if 0 <= index < len(self.rows) else None
        if row is None:
            return 0
        if proof_action and proven > 0 and row.get('proven', 0) >= 0:
            row.setdefault('proof_action', proof_action)
        if row.get('proven'):
            return 0
        row.update(proven=proven, proof_turns=turns)
        return 1

    def episode(self):
        """(episode, rows without `game`) after closing the native objects."""
        winner = self.adjudicated['winner'] if self.adjudicated else self.game.winner
        self.game.close()
        for tree in self.trees.values():
            tree.close()
        forced = self.restart['ply'] if self.restart else 0
        episode = dict(moves=self.moves, winner=winner, reason=self.reason or ('six-in-a-row' if winner >= 0 else 'cap'),
                       opening_plies=min(forced+self.random_plies, len(self.moves)), actor=self.sides[self.learner].sha,
                       actors={str(c): m.sha for c, m in enumerate(self.sides)}, opponent=self.opponent,
                       trained_side=None if self.opponent is None else self.learner,
                       root_values=self.values, full_search=self.full, origin='restart' if self.restart else 'selfplay')
        if self.restart:
            episode['restart'] = {k: self.restart[k] for k in RESTART_SOURCE}
        if dense_solver.active(self.solver, self.schedule):
            episode['solver'] = dense_solver.record(self.solver, self.schedule)
        if self.adjudicated:
            episode['adjudicated'] = self.adjudicated
        return episode, self.rows


class Restarts:
    """The run's restart buffer (restarts.json, dense_solve) as an actor worker uses it. load() re-reads it (missing:
    empty; unreadable: the previous entries are kept) and keeps the entries whose ply is below `max_plies`, so every
    restart game searches at least one ply. draw(rng) returns (entry, moves) for an entry drawn with probability
    proportional to regret^(1/temperature), `moves` the first entry['ply'] moves of its source game, or None when
    no entry is kept or the source shard is gone. The moves of the RESTART_SHARDS most recently used source shards are kept."""

    def __init__(self, run, temperature, max_plies):
        self.run, self.temperature, self.max_plies = Path(run), temperature, max_plies
        self.entries, self.games, self.p = [], OrderedDict(), None
        self.load()

    def load(self):
        path = self.run/'restarts.json'
        try:
            entries = json.loads(path.read_text(encoding='utf-8'))['entries'] if path.exists() else []
        except (OSError, ValueError):
            return
        self.entries = [e for e in entries if e['ply'] < self.max_plies]
        regrets = np.array([e['regret'] for e in self.entries], np.float64)
        positive = regrets > 0
        if not positive.any():
            self.p = None
            return
        logs = np.log(regrets[positive])/self.temperature
        weights = np.zeros(len(regrets))
        weights[positive] = np.exp(logs-logs.max())
        self.p = weights/weights.sum()

    def moves(self, shard):
        if shard not in self.games:
            path = self.run/'shards'/shard/'episodes.json'
            if not path.exists():
                return None
            self.games[shard] = [e['moves'] for e in json.loads(path.read_text(encoding='utf-8'))]
            while len(self.games) > RESTART_SHARDS:
                self.games.popitem(last=False)
        self.games.move_to_end(shard)
        return self.games[shard]

    def draw(self, rng):
        if self.p is None:
            return None
        entry = self.entries[rng.choice(len(self.entries), p=self.p)]
        games = self.moves(entry['shard'])
        return None if games is None else (entry, games[entry['game']][:entry['ply']])


class Historical:
    """Frozen historical opponents of one actor worker. `redraw(champion, sha)` re-reads league.json, draws
    `historical_pool` opponents (draw_pool over opponent_weights; rated checkpoints with an ema.pt whose league
    ema_sha256 differs from the champion's `sha`) and loads their
    ema.pt, keeping models drawn again; `next()` returns (opponent Model, champion colour) for the next historical
    game: opponents come in blocks of about target/BLOCKS consecutive games, so few opponent models share the
    Engine at once, and the champion's colour alternates over historical games only. `target` is the number of
    historical games to keep in flight: round(historical_fraction * min(games_in_flight, games)), `games` being the
    worker's game budget (None: endless), so a short run keeps the fraction too."""

    def __init__(self, run, config, rng, games=None):
        self.run, self.config, self.rng = Path(run), config, rng
        s = config.actor
        self.target = round(s.historical_fraction*min(s.games_in_flight, s.games_in_flight if games is None else games))
        self.block = max(1, math.ceil(self.target/BLOCKS))
        self.models, self.weights, self.plan, self.started = {}, {}, deque(), 0

    def redraw(self, champion, sha):
        path = self.run/'league.json'
        league = json.loads(path.read_text()) if path.exists() else {}
        same = {c['id'] for c in league.get('checkpoints', []) if c.get('ema_sha256') == sha}
        weights = {k: w for k, w in opponent_weights(league, champion, self.config.actor.historical_weighting).items()
                   if k not in same and (self.run/'checkpoints'/k/'ema.pt').exists()}
        pool = draw_pool(weights, self.config.actor.historical_pool, self.rng)
        self.models = {k: self.models.get(k) or load(self.run, self.config, source=(k, self.run/'checkpoints'/k/'ema.pt'))
                       for k in pool}
        self.weights = {k: weights[k] for k in pool}
        self.plan.clear()

    def next(self):
        if not self.plan:
            self.plan.extend(opponent_block(self.weights, self.block, self.rng))
        self.started += 1
        return self.models[self.plan.popleft()], (self.started-1) % 2


def metrics_due(stage, logged_stage, elapsed):
    """Whether a worker appends a metrics line: on every stage change, else every METRICS_SECONDS."""
    return stage != logged_stage or elapsed >= METRICS_SECONDS


class Yield:
    """Cooperative pause of the actor workers (module contract): paused while the pacing rule holds (`below`,
    `resume`; below 0 disables it) or, with `follow`, while a phased learner is in its training phase. paused()
    re-reads <run>/learner-status*.json at most every `check_seconds` and returns the current state; `reason`
    describes the last decision."""

    def __init__(self, run, target, below, resume, check_seconds, follow=False, clock=time.monotonic, now=time.time):
        if below and not 0 < below <= resume:
            raise ValueError('yield_below must be 0 (off) or in (0, yield_resume]')
        self.run, self.target, self.below, self.resume, self.check_seconds = Path(run), target, below, resume, check_seconds
        self.follow, self.clock, self.now, self.checked = follow, clock, now, None
        self.state, self.behind, self.reason = False, False, 'not checked'

    def training(self):
        """The fresh heartbeats of learners that are training or exporting, 'variant' filled in from the file name
        where missing."""
        found = []
        for path in sorted(self.run.glob('learner-status*.json')):
            try:
                status = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            fresh = self.now()-float(status.get('updated_at') or 0) <= STALE_SECONDS
            if fresh and status.get('stage') in ('training', 'exporting'):
                found.append(dict(status, variant=status.get('variant', path.stem)))
        return found

    def lowest(self, statuses):
        """(samples_per_row / its target, samples_per_row, target, variant) of the furthest-behind learner among
        `statuses`, or None. The target is the heartbeat's samples_per_row_target, else `target`."""
        found = []
        for status in statuses:
            if status.get('samples_per_row') is not None:
                rate, target = float(status['samples_per_row']), float(status.get('samples_per_row_target') or self.target)
                found.append((rate/target, rate, target, status['variant']))
        return min(found) if found else None

    def paused(self):
        if not self.below and not self.follow:
            return False
        if self.checked is not None and self.clock()-self.checked < self.check_seconds:
            return self.state
        self.checked = self.clock()
        statuses = self.training()
        lowest = self.lowest(statuses) if self.below else None
        if lowest is None:
            self.behind, self.reason = False, 'no training learner heartbeat'
        else:
            ratio, rate, target, variant = lowest
            if self.behind and ratio >= self.resume:
                self.behind = False
            elif not self.behind and ratio < self.below:
                self.behind = True
            self.reason = (f'learner {variant} at {rate:.2f} samples/row; pause below {self.below*target:.2f}, '
                           f'resume at {self.resume*target:.2f}')
        phased = [s['variant'] for s in statuses if (s.get('phase_rows') or 0) > 0] if self.follow else []
        if phased:
            self.reason = f'learner {", ".join(phased)} in its training phase'
        self.state = self.behind or bool(phased)
        return self.state


def unsearched(rows):
    """Placements among a game's `rows` played without a search (forced-line rows); the actor adds them to the
    Engine's search count so its positions match the rows it publishes."""
    return sum(bool(r.get('line')) for r in rows)


def shard_name():
    return f'{time.time_ns()//1_000_000:013d}{os.getpid() % 1000:03d}'


def worker(args):
    run = Path(args.run)
    config = dense_config.load(run)
    settings = dense_config.override(config.actor, args)
    config = replace(config, actor=settings)
    torch.backends.cudnn.benchmark = False
    status_path = run/('actor-status.json' if args.worker == 0 else f'actor-status-{args.worker}.json')
    entropy = np.random.SeedSequence([config.seed, args.worker, time.time_ns() % 2**63]).entropy
    seeds = np.random.SeedSequence(entropy)
    model = load(run, config, args.initial_model)
    log_event(run, 'actor', 'info', f'worker {args.worker} playing {model.checkpoint} ({model.sha[:12]})'
              + (' - FRESH UNTRAINED NETWORK' if model.checkpoint == 'fresh' else ''), process=args.worker)
    print(f'Worker {args.worker}: {model.checkpoint} {model.sha[:12]}', flush=True)
    historical = Historical(run, config, np.random.default_rng(seeds.spawn(1)[0]), args.games) if settings.historical_fraction > 0 else None
    if historical:
        historical.redraw(model.checkpoint, model.sha)
    restarts = Restarts(run, settings.restart_temperature, settings.max_plies) if settings.restart_fraction > 0 else None
    restart_rng = np.random.default_rng(seeds.spawn(1)[0]) if restarts else None
    engine = Engine(settings.leaf_batch, settings.solver_async, dense_solver.Schedule.of(settings))
    began, solver_failures = time.perf_counter(), 0
    state = dict(published(run, args.worker), error=None)
    target = None if args.games is None else args.games+state['games_completed']
    episodes, rows, started = [], [], 0
    window = deque([(time.perf_counter(), state['positions'], 0)])
    since = dict(time=time.perf_counter(), positions=state['positions'], evals=0)

    logged, stage_logged = time.perf_counter(), 'playing'
    gate = Yield(run, config.learner.samples_per_row, settings.yield_below, settings.yield_resume, settings.yield_check_seconds,
                 settings.phase_follow)
    paused_since, paused_total = None, 0.

    def status(stage):
        """Rewrite the heartbeat; append a metrics line whenever the stage changes and otherwise every
        METRICS_SECONDS."""
        nonlocal logged, stage_logged, solver_failures
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
            mean_plies=state['plies']/g if g else None, shards_written=state['shards_written'],
            paused_seconds=paused_total+(time.perf_counter()-paused_since if paused_since is not None else 0.),
            vram=hexnet.vram(), error=state['error'],
            solver=engine.solver.summary(now-began) if engine.solver else None)
        write_json(status_path, fields)
        if fields['solver'] and fields['solver']['failures'] > solver_failures:
            solver_failures = fields['solver']['failures']
            log_event(run, 'actor', 'error', f'worker {args.worker}: {solver_failures} solver queries failed, last: '
                      f'{fields["solver"]["last_failure"]}', process=args.worker)
        if metrics_due(stage, stage_logged, now-logged):
            logged, stage_logged = now, stage
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
        dense_data.write_shard(run/'shards'/name, identity, episodes, rows, 'actor')
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
        if resolve(run, args.initial_model, settings.model_source, config.learner.variant)[0] != model.checkpoint:
            previous, model = model.checkpoint, load(run, config, args.initial_model)
            log_event(run, 'actor', 'actor_model', f'worker {args.worker} switched from {previous} to {model.checkpoint} '
                      f'({model.sha[:12]}, {settings.model_source}); games in progress finish with the previous model',
                      process=args.worker, checkpoint=model.checkpoint, previous=previous, reason=settings.model_source)
        if historical:
            historical.redraw(model.checkpoint, model.sha)
        if restarts:
            restarts.load()

    try:
        last = 0.
        while True:
            while len(engine.slots) < settings.games_in_flight and (args.games is None or started < args.games):
                seed = seeds.spawn(1)[0].generate_state(1, np.uint64)[0].item()
                if historical and historical.models and sum(g.opponent is not None for g in engine.slots) < historical.target:
                    opponent, learner = historical.next()
                    sides = [model, opponent] if learner == 0 else [opponent, model]
                    engine.add(SelfPlayGame(sides, settings, seed, learner, opponent.checkpoint))
                else:
                    restart = restarts.draw(restart_rng) if restarts and restart_rng.random() < settings.restart_fraction else None
                    engine.add(SelfPlayGame([model, model], settings, seed, restart=restart))
                started += 1
            if not engine.slots and not engine.closing:
                break
            if gate.paused():
                if paused_since is None:
                    paused_since = time.perf_counter()
                    log_event(run, 'actor', 'info', f'worker {args.worker} paused: {gate.reason}', process=args.worker)
                    status('paused'); last = time.perf_counter()
                time.sleep(1.)
                if time.perf_counter()-last >= 2:
                    status('paused'); last = time.perf_counter()
                continue
            if paused_since is not None:
                paused_total += time.perf_counter()-paused_since; paused_since = None
                log_event(run, 'actor', 'info', f'worker {args.worker} resumed: {gate.reason}', process=args.worker)
                status('playing'); last = time.perf_counter()
            before = engine.searches
            for slot in engine.step():
                episode, items = slot.episode()
                if episode['reason'] == 'span':
                    log_event(run, 'actor', 'error', f'worker {args.worker}: game ended at ply {len(episode["moves"])}, '
                              f'a searched position spans more than the largest crop', process=args.worker)
                episodes.append(episode)
                rows.extend(dict(r, game=len(episodes)-1) for r in items)
                state['positions'] += unsearched(items)
                state['games_completed'] += 1; state['terminal'] += episode['winner'] >= 0; state['plies'] += len(episode['moves'])
                if len(episodes) >= settings.shard_games:
                    publish()
            state['positions'] += engine.searches-before
            if time.perf_counter()-last >= 2:
                status('playing'); last = time.perf_counter()
        if episodes:
            publish()
        engine.drain()
        status('finished')
    except BaseException as error:
        state['error'] = f'{type(error).__name__}: {error}'
        status('failed')
        log_event(run, 'actor', 'error', state['error'], process=args.worker)
        raise
    finally:
        engine.close()


def published(run, worker, since=0.):
    """Cumulative counts from shards written by actor worker `worker` at or after `since`: positions are rows
    (searched plies), plies are rows plus the forced plies of restart games."""
    totals = dict(games_completed=0, positions=0, shards_written=0, terminal=0, plies=0)
    for m in (dense_data.manifest(path) for path in dense_data.shard_dirs(run)):
        if m['identity'].get('process') == worker and m['created_at'] >= since:
            c = m['counts']
            totals['games_completed'] += c['games']; totals['positions'] += c['rows']
            totals['plies'] += c['rows']+c.get('forced_plies', 0)
            totals['terminal'] += c['terminal_games']; totals['shards_written'] += 1
    return totals


def actor_flags(args):
    """The --<actor setting> flags given in `args`, for forwarding to worker processes."""
    flags = []
    for item in fields(dense_config.ActorSettings):
        value, flag = getattr(args, item.name, None), '--'+item.name.replace('_', '-')
        if value is None:
            continue
        flags += [flag if value else '--no-'+flag[2:]] if isinstance(value, bool) else [flag, str(value)]
    return flags


def supervise(args):
    """Run --processes workers as subprocesses of this script; restart a crashed worker with the games it has
    not published yet."""
    run = Path(args.run)
    dense_config.load(run)
    command = [sys.executable, str(Path(__file__).resolve()), '--run', str(run)]
    command += ['--initial-model', str(args.initial_model)] if args.initial_model else []
    command += actor_flags(args)
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
    dense_config.add_arguments(parser.add_argument_group('actor settings (override config.json for this process)'),
                               dense_config.ActorSettings)
    args = parser.parse_args()
    if args.worker is None:
        supervise(args)
    else:
        worker(args)


if __name__ == '__main__':
    main()
