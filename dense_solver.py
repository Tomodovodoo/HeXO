"""Solver queries inside the dense actor and evaluator searches (dense_selfplay.Engine drives them).

Points (Budgets: settings solver_root_nodes, solver_finalists, solver_finalist_nodes, solver_threat_nodes, plus
Schedule.deep_nodes; 0 = off):
  root       at a turn start (a search with two placements left): does the side to move have a forced win? A proof
             decides the move played for the whole turn, the certificate's first turn (both stones); the search, its
             tree and its recorded policy are unchanged and the rows of both placements get the exact value +1
             (`proven`).
  threat     at a turn start: would the opponent have a forced win if it moved now with a fresh turn? Root actions on
             the certificate's threat cells are sampled first in the search's opening phase (hxg_priority). Ordering
             only: nothing is pruned.
  finalists  in a mid-turn search, at its last halving boundary (its end when it never halves): for each of the
             `finalists` best candidates b (hxg_stats scores), does the opponent have a forced win after our turn
             ends with b? A proof marks b exact-lost (hxg_mark_exact: Q -1, ineligible), so the remaining rounds and
             the final selection discard it and the improved policy gives it no mass.
  deep       background proof of a committed turn: at each turn start, does the side that just moved win against
             every defence of the turn it played (a root_moves query on the position before that turn)?
Only native-verified PROVEN_WIN results act; UNKNOWN is never a loss. Proofs of positions on the game (root, deep,
finalist) are Proofs. Without Schedule.follow a root proof decides its own turn only. With follow, the side a proof
favours plays the certificate's turns for as long as the game stays on it (each defender reply covered, each
attacker turn the certificate's), asking no further root, threat, deep or finalist queries, and when the game ends
every Proof labels the rows it covers (Proof.path; the slot's `label`), including proofs that arrived too late to
decide a move.

Schedule.fixed_budgets (evaluation, engine verification and tests): every query spends its point's fixed node
budget (deep: deep_nodes, awaited at the mover's next turn-start search end; a deep query the game never reaches
there is dropped) and every verdict is awaited where it is needed, so a verdict is a function of (position,
attacker, budget, build) and a seeded run repeats exactly on either backend. Otherwise (actors) budgets follow the
measured slack and verdicts are polled:
  budget     a query's allowance is slack_fraction * (lead - GUARD_MS) minus its pool's reserved work per worker,
             in nodes at the pool's measured rate (the RATE_QUANTILE of recent queries, at most RATE), clamped
             to [min_nodes, cap_nodes]; `lead` is the LEAD_QUANTILE of the measured times from submission to the
             first consumption attempt of that point, refreshed every step (before any was measured: the Engine's
             step time, DEEP_LEAD_MS for deep queries). Threat queries keep their fixed budget. Deep queries run
             on their own idle-priority worker with its own ledger, are skipped while their allowance is below
             deep_nodes, and are capped at deep_cap_nodes.
  gate       with gate_weight > 0 the worker scores the attacker's forcing material (tactical_proof gate): below
             forcing_material.LOW the query gets min_nodes (deep queries: deep_nodes), else its allowance times
             (1 + gate_weight * g), capped by cap_nodes rising linearly to gate_cap_nodes at g = 1 (fixed mode:
             floor and caps are the point budget and unbounded, so only the multiplier applies).
  consume    a root or finalist verdict not ready when needed may be waited for up to the step's overrun allowance,
             overrun_fraction of the measured step time; past it the slot is deferred to its next visit while the
             other games build the batch. After DEFER_VISITS deferrals the search goes on without it (finalist hold
             released, move from the search) and the query finishes in the background as a late proof. A threat
             verdict not in at the next visit is dropped (the search starts unordered; the query is still accounted
             when it completes). Deep and late verdicts are polled at every search start of the game.
SAFETY_MS (deep: DEEP_SAFETY_MS) is only the wall-clock cap: a query that reaches it, or any other UNKNOWN whose
reason is not a search verdict (VERDICTS), is a failure (Solver.stats), not a verdict.
"""
from collections import deque
from concurrent.futures import Future
from dataclasses import asdict, dataclass, fields
import heapq
import json
import threading
import time

import numpy as np

from neural_search import checked, native
from tactical_proof import MAX_NODES, PROVEN_WIN, IsolatedTactics, NativeTactics, build_hash

SAFETY_MS, DEEP_SAFETY_MS = 5000, 60000
POINTS = ('root', 'threat', 'finalist', 'deep')
# UNKNOWN reasons that are search verdicts, not failures: the searcher's own, and a root_moves query whose turn leaves
# no forcing continuation within its budget.
VERDICTS = {'no verified strategy', 'quiet defender unsupported', 'defender counterwin',
            'candidate has unproved defender continuation', 'candidate defense expansion budget',
            'candidate certificate size limit', 'free-second coverage work limit'}
GUARD_MS = 50.          # lead time kept free of solver work
LEAD_QUANTILE = .2      # of the measured leads per point
OVERHEAD_MS = .5        # fixed solver cost plus pipe per query
RATE, RATE_QUANTILE = 26., .5  # nodes per ms of one worker (tools/tactical calibration) and the quantile tracked
RATE_NODES = 1000       # queries searching fewer nodes are dominated by fixed costs and do not update the rate
DEEP_LEAD_MS = 2000.    # lead of a deep query before one was measured
DEFER_VISITS = 1        # visits a root or finalist verdict may hold its game back
BANDS = (250, 1000, 4000)  # hit rates by granted budget: up to 250, 1000, 4000 nodes, and above
WINDOW = 2048           # recent samples kept for quantiles


@dataclass(frozen=True)
class Budgets:
    """Node budgets of one side's solver points (module contract); all 0 is off."""
    root_nodes: int = 0
    finalists: int = 0
    finalist_nodes: int = 0
    threat_nodes: int = 0

    def __post_init__(self):
        if min(self.root_nodes, self.finalists, self.finalist_nodes, self.threat_nodes) < 0 \
                or (self.finalists > 0) != (self.finalist_nodes > 0):
            raise ValueError('Solver budgets are non-negative; solver_finalists and solver_finalist_nodes are both 0 or '
                             'both positive')

    @classmethod
    def of(cls, settings):
        """The budgets of an ActorSettings or EvaluationSettings (its solver_* fields)."""
        return cls(**{f.name: getattr(settings, 'solver_'+f.name) for f in fields(cls)})

    @property
    def active(self):
        return any(getattr(self, f.name) for f in fields(self))


@dataclass(frozen=True)
class Schedule:
    """How one process runs its queries (module contract); the defaults are fixed budgets, one worker, no deep
    proofs, no following. `workers` foreground worker processes (the asynchronous backend); adaptive deep proofs
    add one."""
    fixed_budgets: bool = True
    workers: int = 1
    slack_fraction: float = .8
    overrun_fraction: float = .05
    min_nodes: int = 32
    cap_nodes: int = 512
    gate_cap_nodes: int = 8192
    gate_weight: float = 0.
    deep_nodes: int = 0
    deep_cap_nodes: int = 65536
    follow: bool = False

    def __post_init__(self):
        if (self.workers < 1 or self.slack_fraction < 0 or self.overrun_fraction < 0 or not 0 <= self.gate_weight <= 100
                or not 1 <= self.min_nodes <= self.cap_nodes <= self.gate_cap_nodes <= MAX_NODES
                or not 0 <= self.deep_nodes <= self.deep_cap_nodes <= MAX_NODES):
            raise ValueError('Invalid solver schedule')
        if self.deep_nodes and not self.follow:
            raise ValueError('solver_deep_nodes needs solver_follow: deep proofs act only by being followed')

    @classmethod
    def of(cls, settings):
        """The schedule of an ActorSettings (its solver_* fields); settings without them (EvaluationSettings) get
        the defaults."""
        return cls(**{f.name: getattr(settings, 'solver_'+f.name, f.default) for f in fields(cls)})


def active(budgets, schedule):
    """Whether a side with `budgets` (None: no solver) asks anything under `schedule`: a point budget, or deep
    proofs."""
    return budgets is not None and (budgets.active or bool(schedule.deep_nodes))


def record(budgets, schedule=Schedule()):
    """The episode-level `solver` record of a game played with `budgets` under `schedule`: the budgets, the schedule
    and the tactical build hash (the backend is left out: both give the same games)."""
    return dict(**asdict(budgets), schedule=asdict(schedule), build_hash=build_hash())


def quantile(values, q):
    return float(np.quantile(values, q)) if len(values) else None


class Query:
    """One submitted solver query on `base` (the queried history, a tuple of (q, r) tuples; None for threat
    queries, whose position is not on the game). `budget` is the node budget sent (the worker's gate may change it:
    `result`). `ready(block, timeout)` polls or waits; `result()` returns (proven, result dict) once ready."""

    def __init__(self, solver, point, base, budget, future):
        self.solver, self.point, self.base, self.budget, self.future = solver, point, base, budget, future
        self.submitted, self.polled, self.outcome = time.perf_counter(), False, None

    def ready(self, block=False, timeout=None):
        """True once the verdict is in; the first call records the point's lead time. `block` waits (up to
        `timeout` seconds when given), counted as verdict wait."""
        now = time.perf_counter()
        if not self.polled:
            self.polled = True
            self.solver.leads[self.point].append((now-self.submitted)*1000)
        if self.future.done() or not block:
            return self.future.done()
        try:
            self.future.result(timeout)
        except TimeoutError:
            pass
        self.solver.waited(time.perf_counter()-now)
        return self.future.done()

    def result(self):
        if self.outcome is None:
            self.ready(block=True)
            result = self.future.result()
            self.solver.account(self.point, result)
            self.outcome = result['status'] == PROVEN_WIN and bool(result.get('native_verified')), result
        return self.outcome


def decode(result):
    """`result` with a proven certificate decoded (IsolatedTactics returns it as JSON text)."""
    text = result.pop('certificate_json', None)
    if text is not None and result['status'] == PROVEN_WIN:
        result['certificate'] = json.loads(text)
    return result


class Pool:
    """Worker threads, each owning one IsolatedTactics child at `priority`, serving queries earliest deadline first.
    `reserved` is the expected milliseconds of the work queued or running (the capacity ledger); `rate` the
    RATE_QUANTILE of the nodes per ms of recent queries that searched at least RATE_NODES nodes, at most RATE."""

    def __init__(self, workers, priority):
        self.engines = [IsolatedTactics(priority=priority) for _ in range(workers)]
        self.heap, self.sequence, self.reserved, self.stopped = [], 0, 0., False
        self.rates, self.rate = deque(maxlen=256), RATE
        self.condition = threading.Condition()
        self.threads = [threading.Thread(target=self.serve, args=(e,), daemon=True) for e in self.engines]
        for thread in self.threads:
            thread.start()

    def submit(self, deadline, reserved, history, request):
        future = Future()
        with self.condition:
            self.sequence += 1
            self.reserved += reserved
            heapq.heappush(self.heap, (deadline, self.sequence, reserved, history, request, future))
            self.condition.notify()
        return future

    def serve(self, engine):
        while True:
            with self.condition:
                while not self.heap and not self.stopped:
                    self.condition.wait()
                if self.stopped:
                    return
                _, _, reserved, history, request, future = heapq.heappop(self.heap)
            try:
                result = decode(engine.history(history, **request))
            except Exception as error:
                with self.condition:
                    self.reserved -= reserved
                future.set_exception(error)
                continue
            used, elapsed = int(result.get('nodes_used') or 0), float(result.get('elapsed_ms') or 0.)
            with self.condition:
                self.reserved -= reserved
                if used >= RATE_NODES and elapsed > OVERHEAD_MS:
                    self.rates.append(used/(elapsed-OVERHEAD_MS))
                    self.rate = min(RATE, quantile(self.rates, RATE_QUANTILE))
            future.set_result(result)

    def close(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        for thread in self.threads:
            thread.join()
        for engine in self.engines:
            engine.close()


class Solver:
    """The tactical backend of one process under a Schedule (module contract): asynchronous (a Pool of
    schedule.workers below-normal IsolatedTactics children, plus a one-child idle-priority Pool for adaptive deep
    proofs) or synchronous (NativeTactics, each query answered inside submit; fixed budgets only, since adaptive
    verdicts are polled and deep proofs need their own worker).

    stats: per point {queries, hits, nodes, budget, solver_ms}, hit counts by budget band, recent budgets and gate
    scores, verdict waits (count, ms, steps with a wait), deferred slots, late verdicts, dropped threat verdicts,
    followed placements, labelled rows, failures and last_failure. The Engine reports each step's time (`tick`).
    """

    def __init__(self, schedule=Schedule(), asynchronous=True):
        if not asynchronous and not schedule.fixed_budgets:
            raise ValueError('Adaptive solver budgets need the asynchronous backend (solver_async)')
        self.schedule, self.asynchronous, self.build_hash = schedule, asynchronous, build_hash()
        self.step_ms = self.collect_ms = None
        self.allowance, self.lead, self.orphans = 0., {}, []
        self.leads = {p: deque(maxlen=WINDOW) for p in POINTS}
        self.pool = Pool(schedule.workers, 'below_normal') if asynchronous else None
        self.background = Pool(1, 'idle') if asynchronous and schedule.deep_nodes and not schedule.fixed_budgets else None
        self.engine = None if asynchronous else NativeTactics()
        self.stats = dict(points={p: dict(queries=0, hits=0, nodes=0, budget=0, solver_ms=0.) for p in POINTS},
                          bands=[[0, 0] for _ in range(len(BANDS)+1)], budgets=deque(maxlen=WINDOW),
                          gate_scores=deque(maxlen=WINDOW), steps=0, step_ms=0., waits=0, wait_ms=0., wait_steps=0,
                          step_waits=0, deferred=0, late=0, dropped=0, skipped=0, followed=0, labelled=0, failures=0,
                          last_failure=None)

    def waited(self, seconds):
        self.stats['waits'] += 1
        self.stats['step_waits'] += 1
        self.stats['wait_ms'] += seconds*1000
        self.allowance -= seconds*1000

    def tick(self, step_ms, collect_ms):
        """End of an Engine step of `step_ms`, `collect_ms` of it collecting the previous batch (GPU wait plus
        decoding): refresh the estimates (step and collect times, per-point leads) and the next step's overrun
        allowance, and account the orphaned queries (of finished games) that have completed."""
        for query in [q for q in self.orphans if q.future.done()]:
            self.orphans.remove(query)
            query.result()
        s = self.stats
        s['steps'] += 1
        s['step_ms'] += step_ms
        s['wait_steps'] += s['step_waits'] > 0
        s['step_waits'] = 0
        ema = lambda old, new: new if old is None else .95*old+.05*new
        self.step_ms, self.collect_ms = ema(self.step_ms, step_ms), ema(self.collect_ms, collect_ms)
        self.allowance = self.schedule.overrun_fraction*self.step_ms
        self.lead = {p: quantile(v, LEAD_QUANTILE) for p, v in self.leads.items() if len(v) >= 8}

    def allocate(self, point, nodes):
        """(node budget or None to skip, gate, pool) of a new `point` query whose fixed budget is `nodes` (module
        contract)."""
        sc = self.schedule
        if sc.fixed_budgets or point == 'threat':
            gate = None if not sc.gate_weight or point == 'threat' else \
                dict(weight=sc.gate_weight, floor=nodes, cap_low=MAX_NODES, cap_high=MAX_NODES)
            return nodes, gate, self.pool
        deep = point == 'deep'
        pool = self.background if deep else self.pool
        lead = self.lead.get(point) or (DEEP_LEAD_MS if deep else self.step_ms or 0.)
        if pool:
            with pool.condition:
                rate, queued = pool.rate, pool.reserved/len(pool.engines)
        else:
            rate, queued = RATE, 0.
        nodes_free = rate*(sc.slack_fraction*(lead-GUARD_MS)-queued-OVERHEAD_MS)
        cap, high = (sc.deep_cap_nodes, sc.deep_cap_nodes) if deep else (sc.cap_nodes, sc.gate_cap_nodes)
        if deep and nodes_free < nodes:
            return None, None, pool
        budget = int(min(cap, max(sc.min_nodes, nodes_free)))
        floor = nodes if deep else sc.min_nodes
        gate = dict(weight=sc.gate_weight, floor=floor, cap_low=cap, cap_high=high) if sc.gate_weight else None
        return budget, gate, pool

    def submit(self, point, history, attacker, nodes, root_moves=None):
        """Queue a `point` query on `history` for `attacker` ('mover' or 'opponent') with fixed budget `nodes`; None
        when an adaptive deep query does not fit."""
        budget, gate, pool = self.allocate(point, nodes)
        if budget is None:
            self.stats['skipped'] += 1
            return None
        most = budget if gate is None else max(budget, gate['floor'], min(gate['cap_high'], round(budget*(1+gate['weight']))))
        request = dict(nodes=budget, ms=DEEP_SAFETY_MS if point == 'deep' else min(DEEP_SAFETY_MS, SAFETY_MS+most//4),
                       attacker=attacker, gate=gate, root_moves=root_moves)
        base = None if attacker == 'opponent' else tuple(map(tuple, history))
        history = [list(p) for p in history]
        if pool:
            deadline = time.perf_counter()*1000+(self.lead.get(point) or 0.)
            future = pool.submit(deadline, OVERHEAD_MS+most/pool.rate, history, request)
        else:
            future = Future()
            future.set_result(self.engine.history(history, **request))
        return Query(self, point, base, budget, future)

    def consume(self, query, block):
        """Whether `query`'s verdict may be used now. Fixed budgets wait for it; otherwise a verdict not yet in is
        waited for, when `block`, up to what is left of the step's overrun allowance."""
        if self.schedule.fixed_budgets:
            return query.ready(block=True)
        if query.ready() or not block or self.allowance <= 0:
            return query.future.done()
        return query.ready(block=True, timeout=self.allowance/1000)

    def idle(self, seconds=.002):
        """Every slot waits for a verdict and no batch is in flight: sleep briefly (counted as verdict wait)."""
        start = time.perf_counter()
        time.sleep(seconds)
        self.waited(time.perf_counter()-start)

    def account(self, point, result):
        stats = self.stats['points'][point]
        proven = result['status'] == PROVEN_WIN and bool(result.get('native_verified'))
        if result.get('build_hash') not in (None, self.build_hash):
            raise ValueError('Tactical build changed while the solver was running; rebuild and restart')
        budget = int(result.get('budget') or 0)
        stats['queries'] += 1
        stats['hits'] += proven
        stats['nodes'] += int(result.get('nodes_used') or 0)
        stats['budget'] += budget
        stats['solver_ms'] += float(result.get('elapsed_ms') or 0.)
        band = self.stats['bands'][sum(budget > b for b in BANDS)]
        band[0] += 1
        band[1] += proven
        self.stats['budgets'].append(budget)
        if result.get('gate_score') is not None:
            self.stats['gate_scores'].append(result['gate_score'])
        if not proven and result.get('reason') not in VERDICTS:
            self.stats['failures'] += 1
            self.stats['last_failure'] = result.get('reason')

    def summary(self, seconds):
        """Status fields: query rate; per point hit rate, mean backend ms, mean granted budget and query count;
        granted budget mean and p95; hit rate by budget band; verdict waits (wait_step_fraction: share of steps
        with one, wait_ms_per_waiting_step, overrun_fraction: wait time over step time); the measured nodes per ms,
        step, collect and per-point lead times; worker utilisation (backend time over worker wall time); deferred,
        late, dropped, skipped, followed and labelled counts; failures."""
        s, points = self.stats, self.stats['points']
        queries = sum(p['queries'] for p in points.values())
        rate = lambda a, b: a/b if b else None
        budgets, labels = list(s['budgets']), [f'<={b}' for b in BANDS]+[f'>{BANDS[-1]}']
        pools = [p for p in (self.pool, self.background) if p]
        workers = sum(len(p.engines) for p in pools) or 1
        return dict(
            asynchronous=self.asynchronous, build_hash=self.build_hash, schedule=asdict(self.schedule),
            queries=queries, queries_per_second=rate(queries, seconds), failures=s['failures'],
            last_failure=s['last_failure'],
            **{f'{k}_hit_rate': rate(p['hits'], p['queries']) for k, p in points.items()},
            **{f'{k}_ms': rate(p['solver_ms'], p['queries']) for k, p in points.items()},
            **{f'{k}_budget': rate(p['budget'], p['queries']) for k, p in points.items()},
            **{f'{k}_queries': p['queries'] for k, p in points.items()},
            budget_mean=float(np.mean(budgets)) if budgets else None, budget_p95=quantile(budgets, .95),
            band_queries=dict(zip(labels, (b[0] for b in s['bands']))),
            band_hit_rate=dict(zip(labels, (rate(b[1], b[0]) for b in s['bands']))),
            wait_step_fraction=rate(s['wait_steps'], s['steps']),
            wait_ms_per_waiting_step=rate(s['wait_ms'], s['wait_steps']), overrun_fraction=rate(s['wait_ms'], s['step_ms']),
            nodes_per_ms=[p.rate for p in pools], step_ms=self.step_ms, collect_ms=self.collect_ms,
            lead_ms={p: quantile(v, LEAD_QUANTILE) for p, v in self.leads.items()},
            utilisation=rate(sum(p['solver_ms'] for p in points.values()), workers*seconds*1000),
            gate_score_mean=float(np.mean(s['gate_scores'])) if s['gate_scores'] else None,
            **{k: s[k] for k in ('deferred', 'late', 'dropped', 'skipped', 'followed', 'labelled')})

    def drain(self, timeout=DEEP_SAFETY_MS/1000):
        """Wait up to `timeout` seconds for the orphaned queries and account them (the end of a run)."""
        end = time.perf_counter()+timeout
        for query in list(self.orphans):
            try:
                query.future.result(max(0., end-time.perf_counter()))
            except TimeoutError:
                continue
            self.orphans.remove(query)
            query.result()

    def close(self):
        for pool in (self.pool, self.background):
            if pool:
                pool.close()


def mover(history):
    """The side to move after `history` placements (one opening placement, then turns of two)."""
    return ((len(history)+1)//2) % 2


class Proof:
    """A verified certificate of a forced win for the side to move at `base` (a history of (q, r) tuples).
    `path(history)` walks it along a game history extending `base` and returns (labels, move): labels
    [(ply, proven, proof_turns)] for the plies of `history` it decides (+1 at the attacker's turn start and after
    a first stone of the certificate's turn, -1 at both plies of a defender turn it reaches), and move (stones of the
    attacker's current turn still to play, [] while the defender is to move; attacker turns left on the longest
    path) while `history` stays on the certificate, None once it leaves it or passes its last attacker decision.
    After an `unstoppable` node the attacker's next turn completes the first of its threats the defender's reply
    left open. `first_turn_only` ends the walk with the certificate's first turn."""

    def __init__(self, base, certificate, first_turn_only=False):
        self.base, self.nodes, self.root = tuple(base), certificate['nodes'], certificate['root']
        self.first_turn_only, self.depth = first_turn_only, {}
        self.replies = {i: {tuple(sorted(map(tuple, r['action']))): r['child'] for r in n['responses']}
                        for i, n in enumerate(self.nodes) if n['kind'] == 'defender_replies'}

    def turns(self, index):
        if index not in self.depth:
            node = self.nodes[index]
            self.depth[index] = (1+self.turns(node['child']) if node['kind'] == 'attacker_move' else
                                 max(map(self.turns, self.replies[index].values())) if index in self.replies else 1)
        return self.depth[index]

    def path(self, history):
        history, i = tuple(map(tuple, history)), len(self.base)
        if history[:i] != self.base:
            return [], None
        labels, index, first = [], self.root, True
        while True:
            node, turns = self.nodes[index], self.turns(index)
            if node['kind'] in ('defender_replies', 'unstoppable'):
                labels += [(p, -1, turns) for p in range(i, min(i+2, len(history)))]
                reply = history[i:i+2]
                if len(reply) < 2:
                    return labels, ([], turns)
                i += 2
                if node['kind'] == 'unstoppable':
                    threat = next((t for t in node['threats'] if not set(map(tuple, t)) & set(reply)), None)
                    if threat is None:
                        return labels, None
                    node, turns = dict(kind='immediate_win', action=threat), 1
                else:
                    index = self.replies[index].get(tuple(sorted(reply)))
                    if index is None:
                        return labels, None
                    continue
            action = [tuple(a) for a in node['action']]
            played = history[i:i+len(action)]
            if i < len(history):
                labels.append((i, 1, turns))
            if played[:1] and played[0] in action and len(action) > 1 and i+1 < len(history):
                labels.append((i+1, 1, turns))
            if len(played) < len(action):
                return labels, ([a for a in action if a not in played], turns) if set(played) <= set(action) else None
            if set(played) != set(action) or node['kind'] == 'immediate_win' or (first and self.first_turn_only):
                return labels, None
            index, i, first = node['child'], i+len(action), False


class Plan:
    """The solver work of one Engine slot across its searches (module contract). `budgets` is re-read from the
    slot at every search start, so each colour of a match uses its own; the proof a side plays from and the pending
    deep queries are kept per side."""

    def __init__(self, solver):
        self.solver, self.schedule = solver, solver.schedule
        self.proofs, self.deep, self.late, self.found = {}, {}, [], []
        self.threat = self.root = self.finalists = None
        self.nodes, self.budget, self.turns, self.pruned, self.following, self.deferrals = 0, 0, 0, [], False, 0

    def spent(self, result):
        self.nodes += int(result.get('nodes_used') or 0)
        self.budget += int(result.get('budget') or 0)

    def alive(self, player, history):
        """Whether `player` keeps a proof `history` lies on (one the game has left is dropped)."""
        proof = self.proofs.get(player)
        if proof and proof.path(history)[1] is None:
            del self.proofs[player]
        return player in self.proofs

    def move(self, player, history):
        """The (stones, turns) `player`'s kept proof prescribes at `history`, or None."""
        move = self.proofs[player].path(history)[1] if self.alive(player, history) else None
        return move if move and move[0] else None

    def proven(self, query, history):
        """Keep `query`'s proof for the labels at the end (with follow), and as the proof its side plays from when
        `history` lies on it and that side has no live one. Without follow only a root proof is played, and only
        for its own first turn, even when it arrives late."""
        proven, result = query.result()
        if not proven or query.base is None:
            return
        first_turn_only = not self.schedule.follow
        proof = Proof(query.base, result['certificate'], first_turn_only)
        self.found.append(proof)
        player = mover(query.base)
        if first_turn_only and query.point != 'root':
            return
        if not self.alive(player, history) and proof.path(history)[1] is not None:
            self.proofs[player] = proof

    def poll(self, history):
        """Adaptive budgets: take the deep and late verdicts that are in."""
        for player, query in list(self.deep.items()):
            if query.ready():
                del self.deep[player]
                self.proven(query, history)
        for query in [q for q in self.late if q.ready()]:
            self.late.remove(query)
            self.proven(query, history)

    def begin(self, slot):
        """After hxg_begin: submit this search's queries or arm the finalist hold. True when a verdict should be
        consumed on a later visit (`ready`) before the search requests anything."""
        budgets, tree, schedule = slot.solver, slot.tree, self.schedule
        history = tuple(map(tuple, tree.history))
        self.threat = self.root = self.finalists = None
        self.nodes, self.budget, self.turns, self.pruned, self.following, self.deferrals = 0, 0, 0, [], False, 0
        if not active(budgets, schedule) or not history:
            return False
        player, other = mover(history), 1-mover(history)
        if not schedule.fixed_budgets:
            self.poll(history)
        self.following = schedule.follow and self.move(player, history) is not None
        if self.following:
            return False
        if len(history) % 2 == 0:
            if budgets.finalists and self.move(player, history) is None:
                checked(native.hxg_hold(tree.ptr, 1))
            return False
        if schedule.deep_nodes and len(history) >= 3 and other not in self.deep and not self.alive(other, history):
            query = self.solver.submit('deep', history[:-2], 'mover', schedule.deep_nodes,
                                       root_moves=[list(m) for m in history[-2:]])
            if query:
                self.deep[other] = query
        if budgets.threat_nodes:
            self.threat = self.solver.submit('threat', history, 'opponent', budgets.threat_nodes)
        if budgets.root_nodes:
            self.root = self.solver.submit('root', history, 'mover', budgets.root_nodes)
        return self.threat is not None

    def defer(self, queries):
        """Adaptive budgets: False (defer the slot) while the verdicts of `queries` are not all in and the search may
        still wait for them; after DEFER_VISITS deferrals the missing ones become late proofs and True."""
        if all(self.solver.consume(q, block=True) for q in queries):
            return True
        if self.deferrals < DEFER_VISITS:
            self.deferrals += 1
            self.solver.stats['deferred'] += 1
            return False
        missing = [q for q in queries if not q.future.done()]
        self.late += missing
        self.solver.stats['late'] += len(missing)
        return True

    def ready(self, slot):
        """Apply the verdicts consumed before the search continues: threat ordering, finalist marks. False defers
        the slot to its next visit."""
        ptr = slot.tree.ptr
        if self.threat is not None:
            if self.solver.consume(self.threat, block=False):
                proven, result = self.threat.result()
                self.spent(result)
                if proven:
                    cells = np.ascontiguousarray(result['moves'], np.int64).reshape(-1, 2)
                    checked(native.hxg_priority(ptr, cells, len(cells)))
            else:
                self.solver.stats['dropped'] += 1
                self.late.append(self.threat)   # accounted once it completes; its verdict no longer acts
            self.threat = None
        if self.finalists is not None:
            if not self.defer([query for _, query in self.finalists]):
                return False
            winner = 1-mover(slot.tree.history)
            for action, query in self.finalists:
                if query in self.late:
                    continue
                proven, result = query.result()
                self.spent(result)
                if proven:
                    checked(native.hxg_mark_exact(ptr, int(action[0]), int(action[1]), winner))
                    self.turns = max(self.turns, int(result['proof_turns']))
                    self.pruned.append(action)
                    self.proven(query, slot.tree.history)
            self.finalists = None
            checked(native.hxg_hold(ptr, 0))
        return True

    def hold(self, slot):
        """At the armed hold: submit the finalist queries and return True (leave the slot until the next visit), or
        clear the hold and return False when there is nothing to ask."""
        tree = slot.tree
        ptr, k = tree.ptr, slot.solver.finalists
        n = native.hxg_stats(ptr, None, None, None, None)
        if native.hxg_exact(ptr) >= 0 or not n:
            checked(native.hxg_hold(ptr, 0))
            return False
        actions = np.empty((n, 2), np.int64)
        visits, values, scores = np.empty(n, np.int32), np.empty(n), np.empty(n)
        native.hxg_stats(ptr, actions.ctypes.data, visits.ctypes.data, values.ctypes.data, scores.ctypes.data)
        order = [i for i in np.argsort(-scores, kind='stable')[:k] if np.isfinite(scores[i])]
        history = list(tree.history)
        self.finalists = [(actions[i].tolist(), self.solver.submit(
            'finalist', history+[tuple(actions[i].tolist())], 'mover', slot.solver.finalist_nodes)) for i in order]
        return True

    def finish(self, slot, result):
        """Before the move is played: consume the root verdict (and, with fixed budgets, the mover's pending deep
        verdict at a turn start), play the kept proof's stone, and set result proven (+1 proof, -1 a root the
        finalist marks left exact-lost, else 0), proof_turns, solver_nodes (nodes spent on this search's queries),
        solver_budget (their granted budgets) and pruned (the finalists marked lost). False defers the slot to its
        next visit."""
        history = tuple(map(tuple, slot.tree.history))
        player = mover(history)
        if self.root is not None:
            if not self.schedule.fixed_budgets and not self.defer([self.root]):
                return False
            if self.root not in self.late:
                self.spent(self.root.result()[1])
                self.proven(self.root, history)
            self.root = None
        if self.schedule.fixed_budgets and len(history) % 2 and player in self.deep:
            self.spent(self.deep[player].result()[1])
            self.proven(self.deep.pop(player), history)
        result.update(proven=0, proof_turns=0)
        move = self.move(player, history) if active(slot.solver, self.schedule) else None
        if move is not None:
            result.update(action=list(move[0][0]), proven=1, proof_turns=move[1])
            self.solver.stats['followed'] += self.following
        elif self.pruned and native.hxg_exact(slot.tree.ptr) == 1-mover(history):
            result.update(proven=-1, proof_turns=self.turns)
        result.update(solver_nodes=self.nodes, solver_budget=self.budget, pruned=self.pruned)
        return True

    def pending(self):
        """Adaptive budgets with follow: whether a root, finalist or deep query that may still label this game's
        rows is running."""
        if self.schedule.fixed_budgets or not self.schedule.follow:
            return False
        queries = [*self.deep.values(), *(q for q in self.late if q.point != 'threat')]
        return any(not q.future.done() for q in queries)

    def close(self, slot, moves):
        """The game of `slot` ended with `moves`: with follow, label the rows every kept proof decides through
        slot.label(ply, proven, proof_turns) (adaptive budgets first take the deep and late verdicts already in;
        fixed budgets drop the ones never consumed). Queries still running pass to the Solver, which accounts
        them when they complete."""
        if not self.schedule.fixed_budgets:
            self.poll(moves)
        pending = [self.threat, self.root, *(q for _, q in self.finalists or ()), *self.deep.values(), *self.late]
        self.solver.orphans += [q for q in pending if q is not None and q.outcome is None]
        if not self.schedule.follow:
            return
        for proof in self.found:
            for ply, proven, turns in proof.path(moves)[0]:
                self.solver.stats['labelled'] += slot.label(ply, proven, turns)
