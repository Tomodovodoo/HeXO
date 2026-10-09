"""Generate dense self-play shards with continuously batched native Gumbel trees.

Games keep their starting model while new games follow the selected checkpoint.
Actors record search values; the learner derives targets from those values and
outcomes. Historical-opponent plies are excluded from training. Restarts exclude
validation games and their descendants. Pacing and phase heartbeats coordinate
GPU use with the learner. Settings are defined in dense_config.ActorSettings.
"""
import argparse
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np
import torch

import dense_config
import dense_data
import dense_solver
import hexcrop
import hexnet
from dense_config import log_event
from hexo import Game
from legacy.klent import digest
from neural_search import HOLD, EvaluationCache, GameGraph, NeuralSearch, Recheck, checked, native
from legacy.train import write_json
from proof import VerificationTimeout
from tactical_proof import independent_verify

# Crop cells per forward: 48 positions at 48x48, proportionally fewer for larger crops (a b6c96 forward peaks
# near 0.14 GB; about 0.33 GB per process with the CUDA context). A crop-size group is padded into the next larger size present when that adds fewer than
# MERGE_CELLS cells: b6c96 bf16 forwards cost about 8 ms of launch overhead plus 0.23 us per cell on an
# RTX 3070 Ti.
MAX_CELLS, MERGE_CELLS = 48*48*48, 32768
COLOR = ((np.arange(1 << 16)+1)//2) % 2
METRICS_SECONDS = 30.
PRIOR_GAMES = 16.  # weight, in games, of the Elo prediction when PFSP blends in a recorded score
BLOCKS = 2          # historical opponents in flight at once, about: blocks of target/BLOCKS games per opponent
METRICS = ('positions', 'games_completed', 'placements_per_second', 'evals_per_second', 'mean_batch', 'full_batch_fraction',
           'terminal_fraction', 'mean_plies', 'checkpoint', 'paused_seconds')
STALE_SECONDS = 120.  # learner heartbeats older than this are ignored by Yield
RESTART_SOURCE = ('shard', 'game', 'ply', 'kind', 'regret', 'plies_to_proof')  # buffer entry fields a restart records
RESTART_SHARDS = 16   # source shards whose moves Restarts keeps
CLOSE_SECONDS = 10.   # longest a finished game waits for proofs that may label its rows
# Solver counts a checked search keeps from its first pass; proof fields come from the root after the check (a
# first pass the solver or the tree proved is never checked).
CARRIED = ('solver_nodes', 'solver_budget')


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
    def submit_leaves(self, leaves):
        return self.submit(hexcrop.encode_leaves(native, leaves, allow_span=True, rectangular=True))

    @torch.inference_mode()
    def submit(self, histories, legal=None):
        """Launch the forwards for `histories`; `legal[i]`, when given, is history i's native legal list."""
        samples, groups = [], {}
        for i, h in enumerate(histories):
            if h is None:
                samples.append(None)
                continue
            try:
                samples.append(h if isinstance(h, hexcrop.Sample) else
                               hexcrop.encode_game(hexcrop.Position(h), h, actions=None if legal is None else legal[i], rectangular=True))
            except hexcrop.SpanError:
                samples.append(None)
                continue
            groups.setdefault(samples[i].shape, []).append(i)
        sizes = sorted(groups)
        for small, large in zip(sizes, sizes[1:]):
            if all(a <= b for a, b in zip(small, large)) and len(groups[small])*(large[0]*large[1]-small[0]*small[1]) < MERGE_CELLS:
                groups[large] = groups.pop(small)+groups[large]
        try:  # one pop: a launcher thread may take a staging set concurrently
            staging = self.free.pop()
        except IndexError:
            staging = {}
        chunks = []
        for shape, indices in groups.items():
            height, width = shape
            size = height if height == width else shape
            host = hexnet.staging_buffer(staging, ('planes', size), len(indices), (len(hexcrop.PLANES), height, width),
                                         torch.uint8, self.cuda)
            result = hexnet.staging_buffer(staging, ('out', size), len(indices), (height*width+2,), torch.float32, self.cuda)
            view = host.numpy()
            for j, i in enumerate(indices):
                planes = samples[i].planes
                if planes.shape[-2:] == shape:
                    view[j] = planes
                else:
                    view[j] = 0
                    view[j, :, :planes.shape[-2], :planes.shape[-1]] = planes
            graphed = self.graph is not None and self.graph.supports(size)
            step = max(1, min(self.max_batch, MAX_CELLS//(height*width)))
            if graphed:
                step = min(step, self.graph._limit(size, self.graph.max_batch))
            for start in range(0, len(indices), step):
                chunk = indices[start:start+step]
                x = host[start:start+len(chunk)].to(self.device, non_blocking=True)
                if not (self.cuda and callable(getattr(self.graph, 'copy_predictions', None))):
                    x = x.to(memory_format=self.memory_format, dtype=torch.bfloat16 if self.cuda else torch.float32)
                target = result[start:start+len(chunk)]
                self.copy_predictions(x, target)
                chunks.append((size, chunk, target))
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
            height, width = (size, size) if isinstance(size, int) else size
            packed = packed.numpy()
            if not np.isfinite(packed).all():
                raise FloatingPointError('Nonfinite dense model predictions')
            for row, i in zip(packed, chunk):
                s = samples[i]
                cells = s.cells if s.shape[1] == width else np.where(s.cells >= 0, s.cells//s.shape[1]*width+s.cells % s.shape[1], -1)
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

    def __init__(self, net, sha, checkpoint, device, max_batch, cache_positions, cuda_graphs=False):
        self.sha, self.checkpoint, self.config = sha, checkpoint, net.config
        self.evaluator = Evaluator(net, device, sha, max_batch, cuda_graphs=cuda_graphs)
        self.cache = EvaluationCache(cache_positions)

    def tree(self, history, seed, tactics, graph=False, q_range_floor=0., limit=0):
        """A search tree from `history`: a GameGraph keeping at most `limit` expanded nodes when `limit` > 0, else a
        NeuralSearch (a transposition graph with `graph`)."""
        if limit:
            return GameGraph(self.evaluator, self.sha, history, seed, self.cache, tactics, q_range_floor=q_range_floor,
                             limit=limit)
        return NeuralSearch(self.evaluator, self.sha, history, seed, self.cache, tactics, graph=graph,
                            q_range_floor=q_range_floor)


def load(run, config, initial=None, source=None):
    """Model for `resolve` under config.actor.model_source (or an explicit (checkpoint, path) `source`)."""
    checkpoint, path = source or resolve(run, initial, config.actor.model_source, config.learner.variant)
    if path is None:
        torch.manual_seed(config.seed)
        net = hexnet.HexNet(hexnet.HexNetConfig(**asdict(config.model)))
        sha = hexnet.model_digest(net)
    else:
        net, sha = hexnet.load_model(path), digest(path)
    net.set_kernels(config.actor.net_kernels)
    return Model(net, sha, checkpoint, config.device, config.actor.leaf_batch, config.actor.cache_positions,
                 cuda_graphs=config.actor.cuda_graphs and source is None)


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


def policy_target(result, player):
    """Whether a finished search's improved policy is a policy target. A search that backed up no simulation on an
    exact root repeats the network prior: a lost root (every move lost) records none, a won root records its
    policy, which covers only proven winning moves, unless every legal move wins."""
    if result['completed'] or result['exact_winner'] < 0:
        return True
    return result['exact_winner'] == player and not np.all(result['policy'] > 0)


def policy_surprise(policy, logits):
    """KL(policy || softmax(logits)) in nats: how far a search policy moved from the network prior over the same
    legal moves."""
    logits = np.asarray(logits, np.float64)
    prior = logits-logits.max()
    prior -= np.log(np.exp(prior).sum())
    support = policy > 0
    return float(np.sum(policy[support]*(np.log(policy[support])-prior[support])))


def root_value(result, player):
    """Side-to-move value of a finished search: the solver's exact value when it proved one (result `proven`), else
    exact +-1 when the tree root is exact, else child values under the improved policy. Unvisited child values
    are the root network estimate from hxg_stats. Visits include exploration of rejected moves and must not
    weight the value teacher."""
    if result.get('proven'):
        return float(result['proven'])
    if result['exact_winner'] >= 0:
        return 1. if result['exact_winner'] == player else -1.
    return float(result['policy'] @ result['values'])


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

    A slot may also expose `recheck(result)` and `checking` (SelfPlayGame): each finished search goes through
    `recheck` before `searched`; None means it started another search on the slot's tree, a principal-variation check
    whose searches take no solver plan step, leaf proof or root prediction and whose final result keeps the first
    search's solver counts and network value.

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

    With leaf_nodes > 0, new leaves get an in-process forced-win query before cache lookup or inference,
    bounded to leaf_nodes and 10 ms. NativeTactics verifies the returned certificate against that leaf;
    hxg_prove backs up an exact result and keeps its first turn in the tree. UNKNOWN leaves use the network.
    A proof at the current root also reaches the usual proof-following, adjudication and exact-row path.
    Immediate child proofs exclude losing root moves. A winning first stone extends the certificate to
    the root's two-stone turn. After a root's last stone, a proof after the opponent's first stone also
    certifies that opponent's whole turn and excludes the root move. Native backups propagate deeper exact
    outcomes using one winning continuation at the winner's nodes and complete reply coverage at the loser's.
    Exact roots stop before the simulation cap after outstanding leaf evaluations drain.
    This is opt-in because its CPU cost competes with producing GPU batches.
    """

    def __init__(self, leaf_batch, solver_async=True, schedule=None, leaf_nodes=0, native_feed=False, native_packing=False):
        if native_feed and leaf_nodes:
            raise ValueError('Native feeding does not support synchronous per-leaf proof queries')
        if native_packing and not native_feed:
            raise ValueError('Native packing requires native feeding')
        self.leaf_batch, self.slots, self.cursor, self.inflight = leaf_batch, [], 0, []
        self.native_feed, self.feeds = native_feed, {}
        self.native_packing = native_packing
        self.packed_handles = set()
        self.retired_feed_stats = np.zeros(6, np.int64)
        self.evals = self.calls = self.full_calls = self.hits = self.searches = 0
        self.solver_async, self.schedule = solver_async, schedule or dense_solver.Schedule()
        self.solver, self.plans, self.closing = None, {}, []
        self.leaf_nodes = leaf_nodes
        self.leaf_solver = dense_solver.NativeTactics() if leaf_nodes else None
        self.leaf_queries = self.leaf_proofs = 0
        self.leaf_seconds = 0.
        self.leaf_roots = {}
        self.root_predictions = {}

    def begin(self, slot):
        """Start the slot's next search; True when it must wait for solver verdicts until the next visit."""
        key = position_key(np.asarray(slot.tree.history, np.int64).reshape(-1, 2))
        cached = slot.model.cache.get(key)
        self.root_predictions[id(slot)] = (slot.model, key, float(cached[2][0]), cached[1]) if cached is not None \
            else (slot.model, key, None, None)
        if self.native_feed:
            from native_feed import NativeFeed
            if slot.model not in self.feeds:
                self.feeds[slot.model] = NativeFeed(slot.model.cache.capacity)
            self.feeds[slot.model].begin(slot.tree, cached)
        checked(native.hxg_begin(slot.tree.ptr, slot.budget, slot.samples))
        plan = self.plans.get(id(slot))
        active = dense_solver.active(slot.solver, self.schedule)
        if active:
            self.solver = self.solver or dense_solver.Solver(self.schedule, self.solver_async)
        if plan is None and (active or self.leaf_nodes):
            plan = self.plans[id(slot)] = dense_solver.Plan(self.solver, self.schedule, self.leaf_nodes)
        elif plan is not None and active and plan.solver is None:
            plan.solver = self.solver
        return plan is not None and plan.begin(slot)

    def add(self, slot):
        self.begin(slot)
        self.slots.append(slot)

    def drain(self):
        """Account the solver queries of finished games that are still running (dense_solver.Solver.drain)."""
        if self.solver:
            self.solver.drain()

    def close(self):
        errors = []
        if self.native_packing:
            for handle in self.packed_handles:
                try:
                    handle.close()
                except BaseException as error:
                    errors.append(error)
            self.packed_handles.clear()
            self.inflight.clear()
        for feed in self.feeds.values():
            try:
                feed.close()
            except BaseException as error:
                errors.append(error)
        self.feeds.clear()
        if self.solver:
            try:
                self.solver.close()
            except BaseException as error:
                errors.append(error)
            self.solver = None
        if errors:
            raise errors[0]

    def feed_stats(self):
        totals = self.retired_feed_stats.copy()
        names = ('new_rows', 'joined', 'cache_hits', 'installed', 'pending_rows', 'pending_requests')
        for feed in self.feeds.values():
            totals += np.fromiter(feed.stats().values(), np.int64, count=6)
        return dict(zip(names, map(int, totals)))

    def retire_feeds(self):
        needed = {model for model, _, _, _ in self.inflight}
        for slot in [*self.slots, *(entry[0] for entry in self.closing)]:
            needed.add(slot.model)
            needed.update(getattr(slot, 'trees', {}))
        for model in list(self.feeds):
            if model not in needed:
                feed = self.feeds[model]
                stats = np.fromiter(feed.stats().values(), np.int64, count=6)
                if np.any(stats[4:]):
                    raise RuntimeError('Retiring a native feed with pending requests')
                self.retired_feed_stats += stats
                feed.close()
                del self.feeds[model]

    def synchronize_inflight(self):
        """Finish submitted GPU work without consuming predictions or advancing any tree."""
        for _, _, _, handle in self.inflight:
            event = handle.event if self.native_packing else handle[2]
            if event is not None:
                event.synchronize()

    def step(self):
        pending, count, done, progress, deferred = {}, 0, [], False, False
        self.completed = done
        started = time.perf_counter()
        slots = self.slots
        for _ in range(len(slots)):
            if count >= self.leaf_batch:
                break
            slot = slots[self.cursor % len(slots)]
            self.cursor += 1
            plan = self.plans.get(id(slot))
            if plan and not getattr(slot, 'checking', False) and not plan.ready(slot):
                deferred = True
                continue
            ptr = slot.tree.ptr
            while True:
                if self.native_feed:
                    request, stats = self.feeds[slot.model].gather(slot.tree)
                    count += int(stats[0]+stats[2])
                    self.hits += int(stats[1])
                    progress |= bool(stats[3])
                else:
                    request = native.hxg_next(ptr)
                if request == HOLD:
                    progress = True
                    if plan.hold(slot):
                        break
                    continue
                if request == 0:
                    if not native.hxg_done(ptr):
                        break
                    progress = True
                    result = slot.tree.result(0, 0, 0, 0)
                    checking = getattr(slot, 'checking', False)
                    if not checking and plan and not plan.finish(slot, result):
                        deferred = True
                        break
                    leaf = None if checking else self.leaf_roots.pop(id(slot), None)
                    if leaf is not None:
                        proof, verdict = leaf
                        stones, turns = proof.path(slot.tree.history)[1]
                        bound = dense_solver.proof_plies(len(stones), turns)
                        result.update(solver_nodes=result.get('solver_nodes', 0)+verdict['nodes_used'],
                                      solver_budget=result.get('solver_budget', 0)+verdict['budget'])
                        # A shorter win the tree proved meanwhile keeps its own move, witness and distance.
                        if not (result['proven'] > 0 and result.get('proof_plies', bound) < bound):
                            result.update(proven=1, proof=proof, proof_turns=turns, proof_action=[list(s) for s in stones],
                                          proof_plies=bound, action=list(stones[0]))
                    if not checking:
                        result['network_value'], result['prior_logits'] = self.root_predictions[id(slot)][2:]
                        if self.native_feed:
                            value = self.feeds[slot.model].root_value(slot.tree)
                            if value is not None:
                                result['network_value'] = value
                    if hasattr(slot, 'recheck'):
                        result = slot.recheck(result)
                        if result is None:
                            continue
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
                if self.leaf_solver is not None and not getattr(slot, 'checking', False):
                    before = time.perf_counter()
                    proof = self.leaf_solver.history(history.tolist(), nodes=self.leaf_nodes, ms=10)
                    self.leaf_seconds += time.perf_counter()-before
                    self.leaf_queries += 1
                    if proof['status'] == 'PROVEN_WIN' and proof['native_verified']:
                        if size == len(slot.tree.history):
                            try:
                                hexcrop.encode_game(hexcrop.Position(history), history)
                            except hexcrop.SpanError:
                                slot.reason = 'span'
                                done.append(slot)
                                progress = True
                                break
                        moves = np.ascontiguousarray(proof['moves'], dtype=np.int64).reshape(-1, 2)
                        checked(native.hxg_prove(ptr, request, history, size, dense_data.player_at(size),
                                                2 if size % 2 else 1, moves, len(moves), int(proof['proof_turns'])))
                        self.leaf_proofs += 1
                        root_size = len(slot.tree.history)
                        opponent_second = size == root_size+2 and root_size % 2 == 0
                        if size <= root_size+1 or opponent_second:
                            witness = dense_solver.Proof(tuple(map(tuple, history.tolist())), proof['certificate'],
                                                         first_turn_only=not self.schedule.follow)
                            if size > root_size:
                                winner = dense_data.player_at(size)
                                plan.found.append(witness)
                                if opponent_second or winner == dense_data.player_at(root_size):
                                    # Include the first stone in the certificate's winning turn. For an
                                    # opponent reply this proves the root move losing; for our first stone
                                    # it proves the root winning. No opponent choice is crossed.
                                    certificate = proof['certificate']
                                    nodes = list(certificate['nodes'])
                                    i = certificate['root']
                                    nodes[i] = dict(nodes[i], action=[history[-1].tolist(), *nodes[i]['action']])
                                    witness = dense_solver.Proof(tuple(map(tuple, history[:-1].tolist())),
                                        dict(certificate, nodes=nodes), first_turn_only=not self.schedule.follow)
                                if winner != dense_data.player_at(root_size):
                                    if opponent_second:
                                        plan.found.append(witness)
                                    action = history[root_size]
                                    plan.pruned.append(action.tolist())
                                    plan.turns = max(plan.turns, proof['proof_turns'])
                                    plan.spent(proof)
                                    progress = True
                                    continue
                            self.leaf_roots[id(slot)] = witness, proof
                            if plan is not None:
                                plan.found.append(witness)
                                plan.proofs[dense_data.player_at(root_size)] = witness
                        progress = True
                        continue
                model = slot.model
                key = position_key(history)
                cached = model.cache.get(key)
                if cached is not None:
                    checked(native.hxg_fulfill(ptr, request, *cached, len(cached[0])))
                    root = self.root_predictions[id(slot)]
                    if root[0] is model and root[1] == key:
                        self.root_predictions[id(slot)] = model, key, float(cached[2][0]), cached[1]
                    self.hits += 1
                    progress = True
                else:
                    if key not in pending.setdefault(model, {}):
                        legal = None
                        if not hasattr(model.evaluator, 'submit_leaves'):
                            legal = np.empty((native.hxg_legal(ptr, request, None), 2), np.int64)
                            native.hxg_legal(ptr, request, legal.ctypes.data)
                        pending[model][key] = [(history, legal)]
                    pending[model][key].append((slot, ptr, request))
                    count += 1
        launched = []
        if self.native_feed:
            for model, feed in self.feeds.items():
                batch = feed.take_packed(2**31-1) if self.native_packing else feed.take(2**31-1)
                if batch is None:
                    continue
                keys, leaves = batch
                if self.native_packing:
                    from native_dense import submit
                    handle = submit(model.evaluator, leaves, MAX_CELLS)
                    self.packed_handles.add(handle)
                else:
                    handle = model.evaluator.submit_leaves(leaves)
                launched.append((model, feed, keys, handle))
                self.calls += 1
                self.evals += len(keys)
                self.full_calls += len(keys) == self.leaf_batch
        for model, positions in pending.items():
            keys = list(positions)
            histories, legal = zip(*(positions[k][0] for k in keys))
            if hasattr(model.evaluator, 'submit_leaves'):
                leaves = [(positions[k][1][1], positions[k][1][2], positions[k][0][0]) for k in keys]
                handle = model.evaluator.submit_leaves(leaves)
            else:
                handle = model.evaluator.submit(histories, legal)
            launched.append((model, positions, keys, handle))
            self.calls += 1
            self.evals += len(keys)
            self.full_calls += len(keys) == self.leaf_batch
        stopped, collecting = set(), time.perf_counter()
        for model, positions, keys, handle in self.inflight:
            if self.native_feed:
                if self.native_packing:
                    pointers = set(map(int, positions.install_packed(keys, handle.collect())))
                    self.packed_handles.remove(handle)
                else:
                    pointers = set(map(int, positions.install(keys, model.evaluator.collect(handle))))
                stopped.update(slot for slot in slots if slot.tree.ptr in pointers)
                continue
            for key, prediction in zip(keys, model.evaluator.collect(handle)):
                if prediction is None:
                    stopped.update(slot for slot, _, _ in positions[key][1:])
                    continue
                for slot, ptr, request in positions[key][1:]:
                    if slot not in stopped:
                        checked(native.hxg_fulfill(ptr, request, *prediction, len(prediction[0])))
                        root = self.root_predictions.get(id(slot))
                        if root is not None and root[0] is model and root[1] == key:
                            self.root_predictions[id(slot)] = model, key, float(prediction[2][0]), prediction[1]
                model.cache.put(key, prediction)
        if stopped:
            if not self.native_feed:
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
            if self.native_feed:
                for slot in done:
                    for tree in getattr(slot, 'trees', {None: slot.tree}).values():
                        for feed in self.feeds.values():
                            feed.detach(tree)
            for key in finished:
                self.root_predictions.pop(key, None)
            self.slots = [s for s in self.slots if id(s) not in finished]
            self.closing += [(slot, self.plans.pop(id(slot), None), time.perf_counter()+CLOSE_SECONDS) for slot in done]
        done = []
        for entry in list(self.closing):
            slot, plan, deadline = entry
            if plan and plan.pending() and time.perf_counter() < deadline:
                continue
            self.closing.remove(entry)
            if plan:
                plan.close(slot, tuple(map(tuple, slot.moves)))
            self.completed.append(slot)
            done.append(slot)
        if self.closing and not self.slots and not done:
            self.solver.idle()
        if self.solver:
            self.solver.tick((time.perf_counter()-started)*1000, (collected-collecting)*1000)
        if self.native_feed:
            self.retire_feeds()
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
    episode records origin 'restart' and `restart` (the entry's RESTART_SOURCE fields). `book` (metadata, moves)
    uses the same forced-prefix treatment and records origin 'book' plus its source metadata, and `fork`
    (metadata, moves) from Forks records origin 'fork' and `fork`; ordinary games record origin 'selfplay'.
    `network_values` stores uncorrected predictions for trained searched plies; record_network_values
    fills missing predictions before the worker saves a completed episode.
    With adjudicate_proven a proven search (+1: the side to move wins, -1: every legal move loses) plays its
    move and ends the game there (`adjudicate`). With proven_line_rows, an applicable two-stone winning
    certificate may start with either legal stone so its continuation teaches both conditional complements."""

    def __init__(self, sides, settings, seed, learner=0, opponent=None, restart=None, book=None, *, fork=None,
                 hybrid=False):
        self.hybrid = hybrid
        if hybrid and (settings.graph_nodes <= 0 or settings.pv_check or settings.proven_line_rows):
            raise ValueError('Hybrid self-play requires game_graph, uses native depth views, and has no certificate line rows')
        self.sides, self.settings, self.seed, self.reason, self.adjudicated = sides, settings, seed, None, None
        self.solver, self.schedule = dense_solver.Budgets.of(settings), dense_solver.Schedule.of(settings)
        if hybrid and (dense_solver.active(self.solver, self.schedule) or settings.solver_leaf_nodes):
            raise ValueError('Hybrid self-play uses frontier slice settings, not legacy solver query budgets')
        self.learner, self.opponent = learner, opponent
        self.rng = np.random.default_rng(seed)
        if sum(start is not None for start in (restart, book, fork)) > 1:
            raise ValueError('A game starts from at most one of a restart, an opening book and a fork')
        self.restart, forced = (None, []) if restart is None else (restart[0], [[int(q), int(r)] for q, r in restart[1]])
        self.book = None if book is None else book[0]
        self.fork = None if fork is None else fork[0]
        for start in (book, fork):
            if start is not None:
                forced = [[int(q), int(r)] for q, r in start[1]]
        self.forced_plies = len(forced)
        self.random_plies = int(round(self.rng.exponential(settings.opening_random_plies))) \
            if settings.opening_random_plies > 0 and restart is None and book is None and fork is None else 0
        graph = settings.graph_nodes if hybrid else settings.game_graph
        self.trees = {model: model.tree([tuple(m) for m in forced], seed+k, settings.tactics, settings.search_graph,
                                        settings.q_range_floor, graph)
                      for k, model in enumerate(dict.fromkeys(sides))}
        if settings.hybrid_round_barrier:
            for tree in self.trees.values():
                checked(native.hxg_round_barrier(tree.ptr, 1))
        self.check = None
        self.game, self.moves, self.rows = Game(forced), forced, []
        self.values, self.full = [None]*len(forced), [False]*len(forced)
        self.network_values = [None]*len(forced)
        self.plan()

    @property
    def model(self):
        return self.sides[self.game.player]

    @property
    def tree(self):
        return self.trees[self.model]

    def plan(self):
        """Draw the next search's kind (full with probability full_fraction), budget and root samples, and set the
        side to move's root noise: root_noise at root_noise_concentration for a full search, 0 for a cheap one.
        With full_turns, the second stone of a turn whose first stone this game searched keeps that stone's kind.
        With pv_check a full search's budget is its first pass (neural_search.Recheck, run by `recheck`)."""
        s = self.settings
        ply = len(self.moves) if s.full_turns else 0
        if not (ply and ply % 2 == 0 and ply > self.forced_plies):
            self.is_full = bool(self.rng.random() < s.full_fraction)
        self.budget = s.full_sims if self.is_full else s.cheap_sims
        self.check = Recheck(self.tree, self.budget, s.pv_check) if s.pv_check and self.is_full else None
        self.passes = []
        if self.check is not None:
            self.budget = self.check.budget
        self.samples = s.root_samples if self.is_full else min(s.root_samples, s.cheap_root_samples, s.cheap_sims)
        if s.root_noise and not getattr(self, 'hybrid', False):
            checked(native.hxg_root_noise(self.tree.ptr, s.root_noise if self.is_full else 0.))
            checked(native.hxg_root_concentration(self.tree.ptr, s.root_noise_concentration))

    @property
    def checking(self):
        """True while the tree searches a principal-variation check rather than the position to play."""
        return self.check is not None and self.check.phase != 'first'

    def recheck(self, result):
        """Feed a finished search to the principal-variation check (neural_search.Recheck). None after starting
        the check's next search on the tree; otherwise the result to play and record: `result` itself without a
        check, else the root's statistics after it, with the first search's solver counts and network value and
        `completed` summed over every pass."""
        if self.check is None:
            return result
        self.passes.append(result)
        budget = self.check.step(result)
        if budget:
            checked(native.hxg_begin(self.tree.ptr, budget, self.samples))
            return None
        check, self.check = self.check, None
        first = self.passes[0]
        if check.line is None:
            return first
        final = self.tree.result(0, 0, 0, 0)
        final.update({k: first[k] for k in first if k not in final or k in CARRIED},
                     completed=sum(r['completed'] for r in self.passes), pv_check=check.summary())
        return final

    def searched(self, result):
        game, actions = self.game, result['actions']
        player, ply = game.player, len(self.moves)
        trained = self.opponent is None or player == self.learner
        row = dict(ply=ply, player=player, remaining=game.remaining, legal_sha256=dense_data.legal_digest(actions),
                   policy=None)
        if 'search' in result:
            row['search'] = result['search']
        policy = result['policy']
        if self.is_full and trained and policy_target(result, player):
            if not np.isclose(policy.sum(), 1, atol=1e-6) or np.any(policy < 0):
                raise ValueError('Search policy is not a distribution')
            row['policy'] = policy.astype(np.float32)
            if result.get('prior_logits') is not None:
                row['surprise'] = policy_surprise(row['policy'].astype(np.float64), result['prior_logits'])
        if dense_solver.active(self.solver, self.schedule) or result.get('proven'):
            row.update(proven=result['proven'], proof_turns=result['proof_turns'], solver_nodes=result['solver_nodes'],
                       solver_budget=result['solver_budget'])
            if result['proven'] > 0 and result.get('proof_action'):
                row['proof_action'] = result['proof_action']
            if result['proven']:
                row['proof_plies'] = result.get('proof_plies', 0)
        self.rows.append(row)
        self.values.append(root_value(result, player) if trained else None)
        tactical = self.book is not None and self.book.get('tactical') is not None and ply == self.book['ply']
        self.network_values.append(result.get('network_value') if trained or tactical else None)
        self.full.append(self.is_full and trained)
        if ply < self.random_plies and result.get('proven', 0) <= 0:
            action = actions[self.rng.choice(len(policy), p=policy/policy.sum())].tolist()
        else:
            action = result['action']
            if self.settings.search_choice == 'policy' and not result.get('proven'):
                action = actions[np.argmax(policy)]
        if (self.settings.adjudicate_proven and self.settings.proven_line_rows and result.get('proven', 0) > 0
                and result.get('proof') is not None and game.remaining == 2):
            stones = result['proof'].action(self.moves)
            if stones and len(stones) == 2 and (actions[:, None, :] == np.asarray(stones)[None, :, :]).all(2).any(0).all():
                action = stones[int(self.rng.integers(2))]
        q, r = int(action[0]), int(action[1])
        game.play(q, r)
        if not self.hybrid:
            for tree in self.trees.values():
                tree.advance((q, r))
        self.moves.append([q, r])
        if result.get('proven') and not self.hybrid:
            action = (row.get('proof_action') or [[q, r]]) if result['proven'] > 0 else None
            self.label(ply, result['proven'], result['proof_turns'], action)
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
            for (q, r), turns, proof_action in line:
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
                                      proof_turns=turns, proof_plies=dense_solver.proof_plies(game.remaining, turns, proven > 0),
                                      solver_nodes=0, solver_budget=0, line=True))
                if proven > 0:
                    self.rows[-1]['proof_action'] = proof_action
                self.values.append(float(proven) if trained(player) else None)
                self.network_values.append(None)
                self.full.append(False)
                game.play(q, r)
                self.moves.append([q, r])
        self.reason, self.adjudicated = 'proven', dict(ply=ply, winner=winner, line_plies=len(line))

    def forced_line(self, proof):
        """[((q, r), proof_turns, proof_action)] to the winner's six along a checked strategy.

        Training lines sample retained attacker alternatives after an independent board check, as well as covered
        defender replies and either stone order. A failed or expired check keeps the primary strategy. Each row's
        winning action comes from the strategy followed at that prefix. Analysis without line rows uses the primary
        attacker choices. After an unstoppable node, defender stones avoid its threats; stop if no move is supplied.
        """
        history, line = [tuple(m) for m in self.moves], []
        game = Game(history)
        try:
            while game.winner < 0:
                _, move, node, played = proof.walk(history)
                if move is None:
                    break
                if (self.settings.proven_line_rows and node['kind'] == 'attacker_move' and not played
                        and node.get('alternatives')):
                    choice = self.rng.integers(1+len(node['alternatives']))
                    if choice:
                        alternative = node['alternatives'][choice-1]
                        index, nodes = proof.nodes.index(node), list(proof.nodes)
                        nodes[index] = dict(node, action=alternative['action'], child=alternative['child'])
                        certificate = dict(version=1, width='wide', root=index, nodes=nodes)
                        try:
                            verified = independent_verify(certificate, history, deadline_seconds=.1) == 'PROVEN_WIN'
                        except (ValueError, VerificationTimeout):
                            verified = False
                        if verified:
                            previous_turns = move[1]
                            proof = dense_solver.Proof(history, certificate)
                            _, move, node, played = proof.walk(history)
                            extra = max(0, move[1]-previous_turns)
                            if extra:
                                line = [(stone, turns+extra, action) for stone, turns, action in line]
                stones = move[0] or proof.reply(history, self.rng)
                if stones is None:
                    threats = {tuple(c) for t in node.get('threats', ()) for c in t}
                    stones = [next(tuple(m) for m in game.legal_moves() if tuple(m) not in threats)]
                stone = stones[self.rng.integers(len(stones))]
                line.append((stone, move[1], proof.action(history)))
                game.play(*stone)
                history.append(stone)
        finally:
            game.close()
        return line

    def label(self, ply, proven, turns, proof_action=None):
        """Record a proof's verdict (+1 / -1: the side to move wins / loses) on the row of `ply` unless it has one;
        return the number of newly labelled rows. A winning second-stone row also proves its preceding
        same-player first-stone row, with the played first stone and winning continuations as policy targets
        and one additional placement on its win bound. An opponent's exact loss at the next turn proves
        the preceding played second stone winning, then the same first-stone rule applies."""
        index = ply-(len(self.moves)-len(self.rows))
        row = self.rows[index] if 0 <= index < len(self.rows) else None
        if row is None:
            return 0
        if proof_action and proven > 0 and row.get('proven', 0) >= 0:
            row.setdefault('proof_action', proof_action)
        labelled = 0
        if not row.get('proven'):
            row.update(proven=proven, proof_turns=turns)
            if 'remaining' in row:
                row['proof_plies'] = dense_solver.proof_plies(row['remaining'], turns, proven > 0)
            labelled = 1
        if row.get('proven') == -1 and index > 0:
            previous = self.rows[index-1]
            if (previous.get('remaining') == 1 and previous['player'] != row['player']
                    and previous['ply'] == ply-1 and not previous.get('proven')):
                previous.update(proven=1, proof_turns=turns+1 if turns else 0)
                if row.get('proof_plies', 0) > 0:
                    previous['proof_plies'] = row['proof_plies']+1
                else:
                    previous.pop('proof_plies', None)
                labelled += 1+self.label(ply-1, 1, previous['proof_turns'], [self.moves[ply-1]])
        if row.get('proven') == 1 and row.get('remaining') == 1 and index > 0:
            first = self.rows[index-1]
            if first.get('remaining') == 2 and first['player'] == row['player'] and first['ply'] == ply-1:
                action = [self.moves[ply-1], *row['proof_action']] if row.get('proof_action') else None
                added = self.label(ply-1, 1, turns, action)
                if added:
                    if row.get('proof_plies', 0) > 0:
                        first['proof_plies'] = row['proof_plies']+1
                    else:
                        first.pop('proof_plies', None)
                labelled += added
        return labelled

    def episode(self):
        """(episode, rows without `game`) after closing the native objects."""
        winner = self.adjudicated['winner'] if self.adjudicated else self.game.winner
        self.game.close()
        for tree in self.trees.values():
            tree.close()
        forced = self.forced_plies
        episode = dict(moves=self.moves, winner=winner, reason=self.reason or ('six-in-a-row' if winner >= 0 else 'cap'),
                       opening_plies=min(forced+self.random_plies, len(self.moves)), actor=self.sides[self.learner].sha,
                       actors={str(c): m.sha for c, m in enumerate(self.sides)}, opponent=self.opponent,
                       trained_side=None if self.opponent is None else self.learner,
                       root_values=self.values, network_values=self.network_values, full_search=self.full,
                       origin='book' if self.book is not None else 'fork' if self.fork is not None
                       else 'restart' if self.restart else 'selfplay')
        if self.book is not None:
            episode['book'] = self.book
        if self.fork is not None:
            episode['fork'] = self.fork
        if self.restart:
            episode['restart'] = {k: self.restart[k] for k in RESTART_SOURCE}
            if 'value_source' in self.restart:
                episode['restart']['value_source'] = self.restart['value_source']
        if dense_solver.active(self.solver, self.schedule):
            episode['solver'] = dense_solver.record(self.solver, self.schedule)
        if self.adjudicated:
            episode['adjudicated'] = self.adjudicated
        return episode, self.rows


def record_network_values(slots):
    """Fill missing predictions on trained rows and the tactical opening before saving a completed game.

    Root predictions are saved during search independently of cache eviction. Missing rows share an extra
    batch per model here, including generated proof lines and roots proved later by the offline pass. Forced
    prefixes stay null. Predictions that cannot be encoded also stay null. Return submission sizes for
    actor metrics. The tactical opening also records an untrained opponent's head for reporting only.
    Search values and played moves are already fixed.
    """
    pending, sizes = {}, []
    for slot in slots:
        for row in slot.rows:
            ply = row['ply']
            tactical = slot.book is not None and slot.book.get('tactical') is not None and ply == slot.book['ply']
            if (slot.values[ply] is None and not tactical) or slot.network_values[ply] is not None:
                continue
            history = np.asarray(slot.moves[:ply], np.int64).reshape(-1, 2)
            model, key = slot.sides[row['player']], position_key(history)
            cached = model.cache.get(key)
            if cached is not None:
                slot.network_values[ply] = float(cached[2][0])
            else:
                pending.setdefault(model, {}).setdefault(key, [history]).append((slot, ply))
    for model, positions in pending.items():
        predictions = model.evaluator.evaluate([items[0] for items in positions.values()])
        sizes.append(len(positions))
        for (key, items), prediction in zip(positions.items(), predictions):
            if prediction is None:
                continue
            model.cache.put(key, prediction)
            for slot, ply in items[1:]:
                slot.network_values[ply] = float(prediction[2][0])
    return sizes


class Forks:
    """KataGo's game forks for one actor worker. `offer(episode, model)` forks a finished game with probability
    fork_early_fraction at placement floor(Exp(mean fork_early_plies)), else, drawn again, with probability
    fork_anywhere_fraction at a uniformly drawn placement of the game. At that position it draws a uniform number of candidates between
    fork_min_choices and fork_early_choices (early) or fork_anywhere_choices (anywhere), each a uniform legal move with
    replacement, scores each by `model`'s value head for the side that plays it, and queues the game's moves before
    that placement plus the best candidate. `take()` pops the oldest queued fork as (metadata {kind, ply, choices},
    moves), or None. No fork is queued when the placement lies at or beyond the game's end, the game is a validation
    game, its forced placements reach max_plies, a candidate completes six (the best move would end the game) or no
    candidate fits the largest crop."""

    def __init__(self, settings, validation_fraction, rng):
        self.settings, self.validation_fraction, self.rng = settings, validation_fraction, rng
        self.queue = deque()

    def offer(self, episode, model):
        s, moves = self.settings, episode['moves']
        if self.rng.random() < s.fork_early_fraction:
            kind, ply, most = 'early', int(self.rng.exponential(s.fork_early_plies)), s.fork_early_choices
        elif self.rng.random() < s.fork_anywhere_fraction:
            kind, ply, most = 'anywhere', int(self.rng.integers(max(1, len(moves)))), s.fork_anywhere_choices
        else:
            return None
        if ply >= len(moves) or ply+1 >= s.max_plies or dense_data.holdout(episode, self.validation_fraction):
            return None
        prefix = [[int(q), int(r)] for q, r in moves[:ply]]
        game = Game([tuple(m) for m in prefix])
        try:
            player, legal = game.player, np.asarray(game.legal_moves(), np.int64)
            candidates = legal[self.rng.integers(len(legal), size=int(self.rng.integers(s.fork_min_choices, most+1)))]
            histories, movers = [], []
            for q, r in candidates.tolist():
                game.play(q, r)
                if game.winner >= 0:
                    return None
                histories.append(np.asarray(prefix+[[q, r]], np.int64).reshape(-1, 2))
                movers.append(game.player)
                game.undo()
        finally:
            game.close()
        best, chosen = -np.inf, None
        for candidate, mover, prediction in zip(candidates.tolist(), movers, model.evaluator.evaluate(histories)):
            if prediction is not None:
                value = float(prediction[2][0]) if mover == player else -float(prediction[2][0])
                if value > best:
                    best, chosen = value, candidate
        if chosen is None:
            return None
        fork = dict(kind=kind, ply=ply, choices=len(candidates)), prefix+[chosen]
        self.queue.append(fork)
        return fork

    def take(self):
        return self.queue.popleft() if self.queue else None


class BookStarts:
    """Read-only snapshot of off-policy and tactical starts, reloaded at shard/phase boundaries.

    Draw classes uniformly, then orientations uniformly. An empty eligible pool yields an ordinary game.
    Metadata identifies the book state; the forced prefix has no training rows or invented root values.
    """

    def __init__(self, run, max_plies):
        self.run, self.max_plies = run, max_plies
        self.load()

    def load(self):
        from dense_openings import Book, LIVE
        book = Book(self.run, suite=LIVE)
        self.nodes = [n for n in book.training_openings() if n['depth'] < self.max_plies]
        self.digest = hashlib.sha256(json.dumps([(n['key'], n.get('tactical')) for n in self.nodes],
                                              sort_keys=True).encode()).hexdigest()

    def draw(self, rng):
        if not self.nodes:
            return None
        node = self.nodes[int(rng.integers(len(self.nodes)))]
        moves = np.asarray(node['moves'], np.int64) @ hexcrop.SYMMETRIES[rng.integers(len(hexcrop.SYMMETRIES))]
        metadata = dict(suite='book', key=node['key'], digest=self.digest, ply=len(moves), off_policy=True)
        if node['status'] == 'tactical':
            metadata['tactical'] = {k: node['tactical'][k] for k in ('winner', 'source')}
        return metadata, moves.tolist()


class Restarts:
    """The run's restart buffer (restarts.json, dense_solve) as an actor worker uses it. load() re-reads it (missing:
    empty; unreadable: the previous entries are kept) and keeps the entries whose ply is below `max_plies`, so every
    restart game searches at least one ply. draw(rng) returns (entry, moves) for an entry drawn with probability
    proportional to regret^(1/temperature), `moves` the first entry['ply'] moves of its source game, or None when
    no entry is kept or the source shard is gone. Validation source games and their restart descendants are
    rejected, so the worker starts an ordinary game instead. The RESTART_SHARDS most recently used source shards
    are kept."""

    def __init__(self, run, temperature, max_plies):
        self.run, self.temperature, self.max_plies = Path(run), temperature, max_plies
        self.validation_fraction = dense_config.load(self.run).learner.validation_fraction
        self.entries, self.shards, self.p = [], OrderedDict(), None
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

    def episodes(self, shard):
        if shard not in self.shards:
            path = self.run/'shards'/shard/'episodes.json'
            if not path.exists():
                return None
            self.shards[shard] = json.loads(path.read_text(encoding='utf-8'))
            while len(self.shards) > RESTART_SHARDS:
                self.shards.popitem(last=False)
        self.shards.move_to_end(shard)
        return self.shards[shard]

    def draw(self, rng):
        if self.p is None:
            return None
        entry = self.entries[rng.choice(len(self.entries), p=self.p)]
        episodes = self.episodes(entry['shard'])
        if episodes is None:
            return None
        episode = episodes[entry['game']]
        moves = episode['moves'][:entry['ply']]
        while True:
            if dense_data.holdout(episode, self.validation_fraction):
                return None
            source = episode.get('restart')
            if source is None:
                return entry, moves
            episodes = self.episodes(source['shard'])
            if episodes is None:
                return None
            episode = episodes[source['game']]


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
        self.requests = {}

    def training(self):
        """Fresh learner heartbeats that need actors paused, with a variant for each."""
        found = []
        for path in sorted(self.run.glob('learner-status*.json')):
            try:
                status = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            fresh = self.now()-float(status.get('updated_at') or 0) <= STALE_SECONDS
            stage = status.get('stage')
            if fresh and (stage in ('training', 'exporting') or (self.follow and stage == 'waiting-for-actors')):
                variant = status.get('variant') or ('main' if path.name == 'learner-status.json'
                                                    else path.stem.removeprefix('learner-status-'))
                found.append(dict(status, variant=variant))
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
        self.requests = {s['variant']: s['phase_request'] for s in statuses
                         if self.follow and (s.get('phase_rows') or 0) > 0
                         and isinstance(s.get('phase_request'), str) and s['phase_request']}
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


def shard_name(after=''):
    """Millisecond and process name of a new shard, later than `after` (this process's previous shard) so
    shards queued for the writer within one millisecond never share a directory."""
    ms = time.time_ns()//1_000_000
    if after:
        ms = max(ms, int(after[:13])+1)
    return f'{ms:013d}{os.getpid() % 1000:03d}'


def without_missing_proofs(settings):
    """`settings` with no hybrid proof workers when the tactical solver build does not load; search itself needs
    no solver."""
    if not (settings.hybrid_scheduler and settings.hybrid_proof_workers):
        return settings
    try:
        from tactical_proof import NativeTactics
        NativeTactics().close()
    except (OSError, ValueError, KeyError):
        return replace(settings, hybrid_proof_workers=0)
    return settings


def worker(args):
    run = Path(args.run)
    config = dense_config.load(run)
    requested = dense_config.override(config.actor, args)
    settings = without_missing_proofs(requested)
    if settings != requested:
        log_event(run, 'actor', 'warning', 'tactical solver not built: hybrid search runs without proof workers',
                  process=args.worker)
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
    book_starts = BookStarts(run, settings.max_plies) if settings.book_fraction > 0 else None
    start_rng = np.random.default_rng(seeds.spawn(1)[0]) if book_starts else restart_rng
    forks = Forks(settings, config.learner.validation_fraction, np.random.default_rng(seeds.spawn(1)[0])) \
        if settings.fork_early_fraction or settings.fork_anywhere_fraction else None
    if settings.hybrid_scheduler:
        from hybrid_selfplay import ActorEngine
        engine = ActorEngine(settings)
    else:
        engine = Engine(settings.leaf_batch, settings.solver_async, dense_solver.Schedule.of(settings), settings.solver_leaf_nodes,
                        native_feed=settings.native_feed, native_packing=settings.native_packing)
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
    phase_ack, token_pause = {}, False

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
            phase_ack=dict(phase_ack),
            games_completed=g, games_total=target, positions=state['positions'], active_games=len(engine.slots),
            batch_calls=engine.calls,
            placements_per_second=(state['positions']-p)/max(1e-9, now-t), evals_per_second=(engine.evals-e)/max(1e-9, now-t),
            mean_batch=engine.evals/max(1, engine.calls),
            full_batch_fraction=engine.full_calls/max(1, engine.calls),
            terminal_fraction=state['terminal']/g if g else None,
            mean_plies=state['plies']/g if g else None, shards_written=state['shards_written'],
            paused_seconds=paused_total+(time.perf_counter()-paused_since if paused_since is not None else 0.),
            vram=hexnet.vram(), error=state['error'],
            solver=engine.solver.summary(now-began) if engine.solver else None)
        fields.update(book_fraction=settings.book_fraction, restart_fraction=settings.restart_fraction,
                      off_policy_openings=len(book_starts.nodes) if book_starts else 0)
        if settings.hybrid_scheduler:
            fields['hybrid_scheduler'] = engine.summary()
        write_json(status_path, fields)
        if fields['solver'] and fields['solver']['failures'] > solver_failures:
            solver_failures = fields['solver']['failures']
            log_event(run, 'actor', 'error', f'worker {args.worker}: {solver_failures} solver queries failed, last: '
                      f'{fields["solver"]["last_failure"]}', process=args.worker)
        if metrics_due(stage, stage_logged, now-logged):
            logged, stage_logged = now, stage
            dense_config.append_metrics(run, f'actor-{args.worker}', **{k: fields[k] for k in METRICS})

    def refresh_sources():
        nonlocal model
        if resolve(run, args.initial_model, settings.model_source, config.learner.variant)[0] != model.checkpoint:
            graph = model.evaluator.graph
            if graph is not None and not settings.hybrid_scheduler:
                graph.close()
                model.evaluator.graph = None
                del graph
            previous, model = model.checkpoint, load(run, config, args.initial_model)
            log_event(run, 'actor', 'actor_model', f'worker {args.worker} switched from {previous} to {model.checkpoint} '
                      f'({model.sha[:12]}, {settings.model_source}); games in progress finish with the previous model',
                      process=args.worker, checkpoint=model.checkpoint, previous=previous, reason=settings.model_source)
        if historical:
            historical.redraw(model.checkpoint, model.sha)
        if restarts:
            restarts.load()
        if book_starts:
            book_starts.load()

    def publish():
        now = time.perf_counter()
        name = state['last_shard'] = shard_name(state.get('last_shard', ''))
        actors = sorted({e['actor'] for e in episodes})
        opponents = {}
        for e in episodes:
            if e['opponent']:
                opponents[e['opponent']] = opponents.get(e['opponent'], 0)+1
        identity = dict(actor_sha256=model.sha, actors=actors, opponents=sorted(opponents), checkpoint=model.checkpoint,
                        process=args.worker,
                        pid=os.getpid(), seed_entropy=str(entropy), model=asdict(model.config),
                        actor=asdict(settings), value_targets='not stored; derive from episode root_values and winner')
        games = len(episodes); terminal = sum(e['winner'] >= 0 for e in episodes)
        elapsed = now-since['time']
        fields = dict(shard=name, games=games, rows=len(rows), terminal_fraction=terminal/games,
                      mean_plies=sum(len(e['moves']) for e in episodes)/games,
                      placements_per_second=(state['positions']-since['positions'])/elapsed,
                      evals_per_second=(engine.evals-since['evals'])/elapsed, process=args.worker, opponents=opponents)
        fields.update(book_games=sum(e.get('origin') == 'book' for e in episodes),
                      restart_games=sum(e.get('origin') == 'restart' for e in episodes),
                      fork_games=sum(e.get('origin') == 'fork' for e in episodes))
        # The writer thread owns these lists; the launcher keeps the GPU busy meanwhile. At most two
        # shards wait to be written, so a stalled disk holds back the main loop instead of memory.
        while len(writes) >= 2:
            written(wait=1)
        writes.append((writer.submit(dense_data.write_shard, run/'shards'/name, identity, list(episodes), list(rows), 'actor'),
                       f'shard {name}: {games} games, {len(rows)} rows', fields))
        since.update(time=now, positions=state['positions'], evals=engine.evals)
        episodes.clear(); rows.clear()
        refresh_sources()

    def written(wait=0):
        """Count and log finished shard writes in order, first waiting for `wait` of them; a failed write raises."""
        while writes and (wait > 0 or writes[0][0].done()):
            future, message, fields = writes.popleft()
            future.result()
            wait -= 1
            state['shards_written'] += 1
            log_event(run, 'actor', 'shard', message, **fields)
            print(json.dumps(fields), flush=True)

    def fill_slots():
        nonlocal started
        while len(engine.slots) < settings.games_in_flight and (args.games is None or started < args.games):
            seed = seeds.spawn(1)[0].generate_state(1, np.uint64)[0].item()
            book, restart = None, None
            fork = forks.take() if forks else None
            if fork is None and start_rng is not None:
                # Unconditional shares, also when historical opponents are enabled. Failed restart/book draws
                # become ordinary starts, rather than increasing the other source's allocation.
                draw = start_rng.random()
                if draw < settings.book_fraction:
                    book = book_starts.draw(start_rng)
                elif draw < settings.book_fraction+settings.restart_fraction:
                    restart = restarts.draw(restart_rng) if restarts else None
            if historical and historical.models and sum(g.opponent is not None for g in engine.slots) < historical.target:
                opponent, learner = historical.next()
                sides = [model, opponent] if learner == 0 else [opponent, model]
                engine.add(SelfPlayGame(sides, settings, seed, learner, opponent.checkpoint, restart=restart, book=book,
                                        fork=fork, hybrid=settings.hybrid_scheduler))
            else:
                engine.add(SelfPlayGame([model, model], settings, seed, restart=restart, book=book, fork=fork,
                                        hybrid=settings.hybrid_scheduler))
            started += 1

    writer, writes = ThreadPoolExecutor(1, thread_name_prefix='shard-writer'), deque()
    try:
        last = 0.
        while True:
            if not engine.slots and not engine.closing and args.games is not None and started >= args.games:
                break
            if gate.paused():
                if not gate.requests and not token_pause:
                    fill_slots()
                phase_ack = {variant: token for variant, token in gate.requests.items()
                             if phase_ack.get(variant) == token}
                pending_ack = {variant: token for variant, token in gate.requests.items()
                               if phase_ack.get(variant) != token}
                entered = paused_since is None
                if entered:
                    paused_since = time.perf_counter()
                    log_event(run, 'actor', 'info', f'worker {args.worker} paused: {gate.reason}', process=args.worker)
                if pending_ack or entered and settings.hybrid_scheduler:
                    if settings.hybrid_scheduler:
                        engine.synchronize_inflight([model,*(historical.models.values() if historical else [])])
                    else:
                        engine.synchronize_inflight()
                if pending_ack:
                    phase_ack.update(pending_ack)
                    token_pause = True
                if entered or pending_ack:
                    status('paused'); last = time.perf_counter()
                written()
                time.sleep(1.)
                if time.perf_counter()-last >= 2:
                    status('paused'); last = time.perf_counter()
                continue
            if paused_since is not None:
                paused_total += time.perf_counter()-paused_since; paused_since = None
                phase_ack = {}
                if token_pause:
                    refresh_sources()
                    token_pause = False
                if settings.hybrid_scheduler:
                    engine.resume()
                log_event(run, 'actor', 'info', f'worker {args.worker} resumed: {gate.reason}', process=args.worker)
                status('playing'); last = time.perf_counter()
            fill_slots()
            if not engine.slots and not engine.closing:
                break
            before = engine.searches
            finished = engine.step()
            for size in record_network_values(finished):
                engine.evals += size; engine.calls += 1; engine.full_calls += size == engine.leaf_batch
            for slot in finished:
                episode, items = slot.episode()
                if episode['reason'] == 'span':
                    log_event(run, 'actor', 'error', f'worker {args.worker}: game ended at ply {len(episode["moves"])}, '
                              f'a searched position spans more than the largest crop', process=args.worker)
                episodes.append(episode)
                rows.extend(dict(r, game=len(episodes)-1) for r in items)
                if forks:
                    forks.offer(episode, model)
                state['positions'] += unsearched(items)
                state['games_completed'] += 1; state['terminal'] += episode['winner'] >= 0; state['plies'] += len(episode['moves'])
                if len(episodes) >= settings.shard_games:
                    publish()
            state['positions'] += engine.searches-before
            written()
            if time.perf_counter()-last >= 2:
                status('playing'); last = time.perf_counter()
        if episodes:
            publish()
        written(wait=len(writes))
        engine.drain()
        status('finished')
    except BaseException as error:
        state['error'] = f'{type(error).__name__}: {error}'
        status('failed')
        log_event(run, 'actor', 'error', state['error'], process=args.worker)
        raise
    finally:
        writer.shutdown()
        engine.close()


def published(run, worker, since=0.):
    """Cumulative counts from shards written by actor worker `worker` at or after `since`: positions are rows
    (searched plies), plies are rows plus the forced plies of restart and book games."""
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


@torch.inference_mode()
def warm_fused(run, config, initial):
    """Compile the actor's inference kernels for every crop size in one process."""
    model = load(run, config, initial).evaluator.model
    for size in hexcrop.BUCKETS:
        batch = max(1, min(config.actor.leaf_batch, MAX_CELLS//(size*size)))
        x = torch.zeros((batch, len(hexcrop.PLANES), size, size), device=config.device,
                        dtype=torch.bfloat16).to(memory_format=torch.channels_last)
        x[:, 3:4] = 1
        with torch.autocast('cuda', torch.bfloat16):
            model(x, x[:, 3:4], aux=False)
    torch.cuda.synchronize()


def supervise(args):
    """Run --processes workers as subprocesses of this script; restart a crashed worker with the games it has
    not published yet."""
    run = Path(args.run)
    config = dense_config.load(run)
    settings = dense_config.override(config.actor, args)
    command = [sys.executable, str(Path(__file__).resolve()), '--run', str(run)]
    command += ['--initial-model', str(args.initial_model)] if args.initial_model else []
    command += actor_flags(args)
    remaining = dict.fromkeys(range(args.processes), args.games)
    cache = None
    if settings.net_kernels == 'fused' and torch.device(config.device).type == 'cuda':
        cache = run/'cache'/'triton'/uuid.uuid4().hex
        shared = cache/'shared'

    def spawn(k):
        games = [] if remaining[k] is None else ['--games', str(remaining[k])]
        env, worker_cache = None, None
        if cache is not None:
            worker_cache = cache/f'worker-{k}-{uuid.uuid4().hex}'
            shutil.copytree(shared, worker_cache)
            env = dict(os.environ, TRITON_CACHE_DIR=str(worker_cache.resolve()))
        return subprocess.Popen(command+games+['--worker', str(k)], env=env), time.time(), worker_cache

    workers = {}
    try:
        if cache is not None:
            shared.mkdir(parents=True)
            subprocess.run(command+['--warm-cache'], env=dict(os.environ, TRITON_CACHE_DIR=str(shared.resolve())), check=True)
        for k in range(args.processes):
            workers[k] = spawn(k)
        log_event(run, 'actor', 'info', f'supervisor started {args.processes} workers')
        while workers:
            time.sleep(1)
            for k, (process, started, worker_cache) in list(workers.items()):
                code = process.poll()
                if code is None:
                    continue
                del workers[k]
                if worker_cache is not None:
                    shutil.rmtree(worker_cache)
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
        for process, _, _ in workers.values():
            process.terminate()
        for process, _, _ in workers.values():
            process.wait()
        if cache is not None:
            shutil.rmtree(cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', required=True)
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--games', type=int, default=None, help='games per process (default: endless)')
    parser.add_argument('--initial-model', help='hexnet checkpoint used while the run has no checkpoint')
    parser.add_argument('--worker', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--warm-cache', action='store_true', help=argparse.SUPPRESS)
    dense_config.add_arguments(parser.add_argument_group('actor settings (override config.json for this process)'),
                               dense_config.ActorSettings)
    args = parser.parse_args()
    if args.warm_cache:
        config = dense_config.load(args.run)
        warm_fused(args.run, replace(config, actor=dense_config.override(config.actor, args)), args.initial_model)
    elif args.worker is None:
        supervise(args)
    else:
        worker(args)


if __name__ == '__main__':
    main()
