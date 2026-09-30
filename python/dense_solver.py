"""Solver queries inside the dense actor and evaluator searches (dense_selfplay.Engine drives them).

Points (Budgets: settings solver_root_nodes, solver_finalists, solver_finalist_nodes, solver_threat_nodes, plus
Schedule.deep_nodes; 0 = off):
  root       at a turn start (a search with two placements left): does the side to move have a forced win? A proof
             decides the move played for the whole turn, the certificate's first turn (both stones). The search runs
             unchanged; at its end the certificate stone is marked exact-won (hxg_mark_exact), so the recorded
             policy covers only proven winning stones, and the rows of both placements get the exact value +1
             (`proven`).
  threat     at a turn start: would the opponent have a forced win if it moved now with a fresh turn? Root actions on
             the certificate's threat cells are sampled first in the search's opening phase (hxg_priority). Ordering
             only by default. With Budgets.defence, verify up to defence_candidates complete turns at the same
             node budget. An UNKNOWN search verdict, including a defender counterwin, breaks the threat within
             this budget. Surviving first stones enter the Gumbel root set with 5/defence_candidates bonus per turn;
             compatible second stones get the same treatment next search. No proof labels or pruning follow.
  finalists  in a mid-turn search, at its last halving boundary (its end when it never halves): for each of the
             `finalists` best candidates b (hxg_stats scores), does the opponent have a forced win after our turn
             ends with b? A proof marks b exact-lost when its verdict is consumed (hxg_mark_exact: Q -1, ineligible), so
             the remaining visits go to the survivors, the final selection discards b and the improved policy gives it
             no mass.
  deep       background proof of a committed turn: at each turn start, does the side that just moved win against
             every defence of the turn it played (a root_moves query on the position before that turn)?
Only native-verified PROVEN_WIN results act as proofs; UNKNOWN is never a loss. Proofs of positions on the game (root, deep,
finalist) are Proofs. Without Schedule.follow a root proof decides its own turn only. With follow, the side a proof
favours plays the certificate's turns for as long as the game stays on it (each defender reply covered, each
attacker turn the certificate's), asking no further root, threat, deep or finalist queries, and when the game ends
every Proof labels the rows it covers (Proof.path; the slot's `label`), including proofs that arrived too late to
decide a move.

Schedule.fixed_budgets (evaluation, engine verification and tests): every query spends its point's fixed node
budget (deep: deep_nodes, awaited at the mover's next turn-start search end; a deep query the game never reaches
there is dropped) and every verdict is awaited where it is needed, so a verdict is a function of (position,
attacker, budget, build). With nonblocking_fixed, a game waits for its required verdicts by yielding its Engine
slot, allowing other games to search while the proof runs. Fixed budgets do not promise the same move sequence
across batching schedules: immediate cache hits versus delayed leaf evaluations can change native visit selection.
Otherwise (actors) budgets follow the measured slack and verdicts are polled:
  budget     a query's allowance is slack_fraction * (lead - GUARD_MS, plus the step's overrun allowance for root
             and finalist queries, which may be waited for) minus its pool's reserved work per worker, in nodes at the pool's measured rate (the RATE_QUANTILE of recent queries, at most RATE), clamped
             to [min_nodes, cap_nodes]; `lead` is the LEAD_QUANTILE of the measured times from submission to the
             first consumption attempt of that point, refreshed every step (before any was measured: the Engine's
             step time, DEEP_LEAD_MS for deep queries). Threat queries keep their fixed budget. Deep queries run
             on their own idle-priority worker with its own ledger, are skipped while their allowance is below
             deep_nodes, and are capped at deep_cap_nodes.
  gate       with gate_weight > 0 the worker scores the attacker's forcing material (tactical_proof gate): below
             forcing_material.LOW the query gets min_nodes (deep queries: deep_nodes), else its allowance times
             (1 + gate_weight * g), capped by cap_nodes rising linearly to gate_cap_nodes at g = 1 (fixed evaluator
             gate: floor and lower cap are the point budget, upper cap is gate_cap_nodes).
  consume    a root or finalist verdict not ready when needed may be waited for up to the step's overrun allowance,
             overrun_fraction of the measured step time; past it the slot is deferred to its next visit while the
             other games build the batch. After DEFER_VISITS deferrals the search goes on without it (finalist hold
             released, move from the search) and the query finishes in the background as a late proof. A threat
             verdict not in at the next visit is dropped (the search starts unordered; the query is still accounted
             when it completes). Deep and late verdicts are polled at every search start of the game.
SAFETY_MS (deep: DEEP_SAFETY_MS) is only the wall-clock cap: a query that reaches it, or any other UNKNOWN whose
reason is not a search verdict (VERDICTS), is a failure (Solver.stats), not a verdict.
"""
from collections import Counter, deque
from concurrent.futures import Future
from dataclasses import asdict, dataclass, fields
import heapq
from itertools import combinations
import json
import threading
import time

import numpy as np

from dense_config import EvaluationSettings
from neural_search import checked, native
from hexo import Game
from tactical_proof import MAX_NODES, MAX_TABLE_MB, PROVEN_WIN, IsolatedTactics, NativeTactics, build_hash

SAFETY_MS, DEEP_SAFETY_MS = 5000, 60000
POINTS = ('root', 'threat', 'finalist', 'deep', 'defence')
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
    defence: bool = False
    defence_candidates: int = 8

    def __post_init__(self):
        if self.defence_candidates < 1 or min(self.root_nodes, self.finalists, self.finalist_nodes, self.threat_nodes) < 0 \
                or (self.finalists > 0) != (self.finalist_nodes > 0):
            raise ValueError('Solver budgets are non-negative; solver_finalists and solver_finalist_nodes are both 0 or '
                             'both positive')

    @classmethod
    def of(cls, settings):
        """The budgets of an ActorSettings or EvaluationSettings (its solver_* fields)."""
        return cls(**{f.name: getattr(settings, 'solver_'+f.name) for f in fields(cls)})

    @property
    def active(self):
        return any((self.root_nodes, self.finalists, self.finalist_nodes, self.threat_nodes))


@dataclass(frozen=True)
class Schedule:
    """How one process runs its queries (module contract); the defaults are fixed budgets, one worker, no deep
    proofs, no following. `workers` foreground worker processes (the asynchronous backend); adaptive deep proofs
    add one. `nonblocking_fixed` preserves fixed verdicts while yielding only their waiting game."""
    fixed_budgets: bool = True
    nonblocking_fixed: bool = False
    workers: int = 1
    slack_fraction: float = .95
    overrun_fraction: float = .05
    min_nodes: int = 32
    cap_nodes: int = 512
    gate_cap_nodes: int = 8192
    gate_weight: float = 0.
    fixed_gate_cap: bool = False
    deep_nodes: int = 0
    deep_cap_nodes: int = 65536
    follow: bool = False
    table_mb: int = 32

    def __post_init__(self):
        if (self.workers < 1 or self.slack_fraction < 0 or self.overrun_fraction < 0 or not 0 <= self.gate_weight <= 100
                or not 0 <= self.table_mb <= MAX_TABLE_MB
                or not 1 <= self.min_nodes <= self.cap_nodes <= self.gate_cap_nodes <= MAX_NODES
                or not 0 <= self.deep_nodes <= self.deep_cap_nodes <= MAX_NODES):
            raise ValueError('Invalid solver schedule')
        if self.deep_nodes and not self.follow:
            raise ValueError('solver_deep_nodes needs solver_follow: deep proofs act only by being followed')
        if self.nonblocking_fixed and not self.fixed_budgets:
            raise ValueError('nonblocking_fixed requires fixed_budgets')

    @classmethod
    def of(cls, settings):
        """The schedule of settings' solver_* fields; absent fields use the defaults."""
        values = {f.name: getattr(settings, 'solver_'+f.name, f.default) for f in fields(cls)}
        if isinstance(settings, EvaluationSettings):
            values['nonblocking_fixed'] = bool(getattr(settings, 'pipeline', False))
            if settings.solver_gate_cap_nodes and settings.solver_gate_cap_nodes < max(
                    settings.solver_root_nodes, settings.solver_finalist_nodes, settings.solver_threat_nodes):
                raise ValueError('solver_gate_cap_nodes must cover every enabled evaluation query budget')
            values['gate_weight'] = 3. if settings.solver_gate_cap_nodes else 0.
            values['fixed_gate_cap'] = bool(settings.solver_gate_cap_nodes)
            values['gate_cap_nodes'] = settings.solver_gate_cap_nodes or cls.gate_cap_nodes
            values['cap_nodes'] = min(values['cap_nodes'], values['gate_cap_nodes'])
            values['min_nodes'] = min(values['min_nodes'], values['cap_nodes'])
        return cls(**values)


def active(budgets, schedule):
    """Whether a side with `budgets` (None: no solver) asks anything under `schedule`: a point budget, or deep
    proofs."""
    return budgets is not None and (budgets.active or bool(schedule.deep_nodes))


def record(budgets, schedule=Schedule()):
    """The episode-level `solver` record of a game played with `budgets` under `schedule`: the budgets, the schedule
    and the tactical build hash (the backend is left out: both give the same games)."""
    values = asdict(budgets)
    if not budgets.defence:
        del values['defence'], values['defence_candidates']
    return dict(**values, schedule=asdict(schedule), build_hash=build_hash())


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
        self.rates, self.rate, self.busy_ms = deque(maxlen=256), RATE, 0.
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
            start = time.perf_counter()
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
                self.busy_ms += (time.perf_counter()-start)*1000
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
                          step_waits=0, slack_ms=0., deferred=0, late=0, dropped=0, skipped=0, followed=0, labelled=0,
                          failures=0, last_failure=None)

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
        s['slack_ms'] += collect_ms+self.schedule.overrun_fraction*step_ms
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
        if sc.fixed_budgets or point in ('threat', 'defence'):
            cap = max(nodes, sc.gate_cap_nodes) if sc.fixed_gate_cap else MAX_NODES
            gate = None if not sc.gate_weight or point == 'defence' or (point == 'threat' and not sc.fixed_gate_cap) else \
                dict(weight=sc.gate_weight, floor=nodes, cap_low=nodes if sc.fixed_gate_cap else MAX_NODES,
                     cap_high=cap)
            return nodes, gate, self.pool
        deep = point == 'deep'
        pool = self.background if deep else self.pool
        lead = self.lead.get(point) or (DEEP_LEAD_MS if deep else self.step_ms or 0.)
        if pool:
            with pool.condition:
                rate, queued = pool.rate, pool.reserved/len(pool.engines)
        else:
            rate, queued = RATE, 0.
        overrun = sc.overrun_fraction*(self.step_ms or 0.) if point in ('root', 'finalist') else 0.
        nodes_free = rate*(sc.slack_fraction*(lead-GUARD_MS+overrun)-queued-OVERHEAD_MS)
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
                       attacker=attacker, gate=gate, root_moves=root_moves,
                       table_mb=0 if self.schedule.fixed_budgets else self.schedule.table_mb)
        base = None if attacker == 'opponent' or point == 'defence' else tuple(map(tuple, history))
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
        stats['hits'] += defence_hit(result) if point == 'defence' else proven
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
        step, collect and per-point lead times; worker utilisation (backend time over worker wall time),
        idle_fraction per pool (share of its workers' wall time without a query) and slack_utilisation (foreground
        busy time over its workers' share of the collect time plus overrun allowance of every step: the capacity
        the scheduler targets); deferred, late, dropped, skipped, followed and labelled counts; failures."""
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
            defence_hits=points['defence']['hits'], defence_nodes=points['defence']['nodes'],
            budget_mean=float(np.mean(budgets)) if budgets else None, budget_p95=quantile(budgets, .95),
            band_queries=dict(zip(labels, (b[0] for b in s['bands']))),
            band_hit_rate=dict(zip(labels, (rate(b[1], b[0]) for b in s['bands']))),
            wait_step_fraction=rate(s['wait_steps'], s['steps']),
            wait_ms_per_waiting_step=rate(s['wait_ms'], s['wait_steps']), overrun_fraction=rate(s['wait_ms'], s['step_ms']),
            nodes_per_ms=[p.rate for p in pools], step_ms=self.step_ms, collect_ms=self.collect_ms,
            lead_ms={p: quantile(v, LEAD_QUANTILE) for p, v in self.leads.items()},
            utilisation=rate(sum(p['solver_ms'] for p in points.values()), workers*seconds*1000),
            idle_fraction={name: 1-pool.busy_ms/(len(pool.engines)*seconds*1000) if seconds else None
                           for name, pool in (('foreground', self.pool), ('background', self.background)) if pool},
            slack_utilisation=rate(self.pool.busy_ms, len(self.pool.engines)*s['slack_ms']) if self.pool else None,
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


def proof_plies(remaining, turns, attacker=True):
    """Placements within which a certificate's attacker completes six from a position with `remaining` placements
    left for its mover and `turns` attacker turns left on the certificate's longest path (the completing turn
    included): the attacker's current placements, then two defender and two attacker placements per further turn;
    a defender to move first plays out its `remaining` placements."""
    return remaining+(0 if attacker else 2)+4*(turns-1)


def mover(history):
    """The side to move after `history` placements (one opening placement, then turns of two)."""
    return ((len(history)+1)//2) % 2


def defence_hit(result):
    """A completed search failed to re-prove the attack. Operational failures are not evidence."""
    return result['status'] == 'UNKNOWN' and result.get('reason') in VERDICTS


def defence_turns(history, certificate, limit):
    """Rank complete turns from certificate actions, covered replies and winning completions.

    Deduplicate groups so a large reply expansion cannot multiply their weight. Keep the first attack turn,
    then pairs hitting the most distinct groups. Cross the 2*limit most frequent cells as well as the certificate's
    own pairs; this keeps pair construction bounded even for certificates covering thousands of free replies.
    Coordinate order breaks ties. Every returned turn is legal on the actual board.
    """
    occupied = set(map(tuple, history))
    groups = set()
    attacker = {tuple(c) for i, c in enumerate(history) if ((i+1)//2) % 2 != mover(history)}
    defender = occupied-attacker
    # The certificate may name only one way to finish a line. Include its other open completions too.
    segments = {tuple((q+(k-offset)*dq, r+(k-offset)*dr) for k in range(6))
                for q, r in attacker for dq, dr in ((1, 0), (0, 1), (1, -1)) for offset in range(6)}
    for segment in segments:
        gaps = set(segment)-attacker
        if 0 < len(gaps) <= 2 and not gaps & defender:
            groups.add(tuple(sorted(gaps)))
    for node in certificate['nodes']:
        actions = [node['action']] if 'action' in node else []
        actions += [r['action'] for r in node.get('responses', ())]
        actions += [r['action'] for r in node.get('alternatives', ())]
        actions += node.get('threats', [])
        for action in actions:
            group = tuple(sorted(set(map(tuple, action))-occupied))
            if group:
                groups.add(group)
    counts = Counter(c for group in groups for c in group)
    cells = sorted(counts, key=lambda c: (-counts[c], c))[:2*limit]
    if len(cells) == 1:
        # A single forced block still needs a second placement to make a queryable complete turn.
        game = Game(history)
        try:
            q, r = cells[0]
            filler = min((c for c in game.legal_moves() if c != cells[0]),
                         key=lambda c: (max(abs(c[0]-q), abs(c[1]-r), abs(sum(c)-q-r)), c))
            cells.append(filler)
            counts[filler] = 0
        finally:
            game.close()
    pairs = {g for g in groups if len(g) == 2} | {tuple(sorted(p)) for p in combinations(cells, 2)}
    masks = {c: 0 for c in counts}
    for i, group in enumerate(sorted(groups)):
        for c in group:
            masks[c] |= 1 << i
    first = tuple(sorted(map(tuple, certificate['nodes'][certificate['root']].get('action', ()))))
    rank = lambda p: (p != first, -(masks[p[0]] | masks[p[1]]).bit_count(), -sum(counts[c] for c in p), p)
    game, turns = Game(history), []
    try:
        for pair in sorted(pairs, key=rank):
            # Prefer the more frequent cell first; reverse if only the other order is legal.
            pair = tuple(sorted(pair, key=lambda c: (-counts[c], c)))
            if not game.legal(*pair[0]):
                pair = pair[::-1]
            if not game.legal(*pair[0]):
                continue
            game.play(*pair[0])
            if game.winner >= 0:
                game.undo()
                continue  # Own terminal wins belong to the exact tactical search.
            legal = game.legal(*pair[1])
            if legal:
                game.play(*pair[1])
                legal = game.winner < 0
                game.undo()
            game.undo()
            if legal:
                turns.append(pair)
                if len(turns) == limit:
                    break
    finally:
        game.close()
    return turns


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
        return self.walk(history)[:2]

    def action(self, history):
        """Winning placements still to play at this prefix, or None outside an attacker turn."""
        move = self.path(history)[1]
        return [list(a) for a in move[0]] if move and move[0] else None

    def reply(self, history):
        """The defender stones still to play in the first certificate reply that extends the defender's turn in
        progress at the end of `history`, or None (not at a covered defender turn on the certificate)."""
        _, move, node, played = self.walk(history)
        if move is None or move[0] or node['kind'] != 'defender_replies':
            return None
        for response in node['responses']:
            action = [tuple(a) for a in response['action']]
            if set(played) <= set(action):
                return [a for a in action if a not in played]
        return None

    def walk(self, history):
        """(labels, move, node, played) of `path`, plus the certificate node the walk ended at and the stones of the
        turn in progress there (None, () when it left the certificate)."""
        history, i = tuple(map(tuple, history)), len(self.base)
        if history[:i] != self.base:
            return [], None, None, ()
        labels, index, first = [], self.root, True
        while True:
            node, turns = self.nodes[index], self.turns(index)
            if node['kind'] in ('defender_replies', 'unstoppable'):
                labels += [(p, -1, turns) for p in range(i, min(i+2, len(history)))]
                reply = history[i:i+2]
                if len(reply) < 2:
                    return labels, ([], turns), node, reply
                i += 2
                if node['kind'] == 'unstoppable':
                    threat = next((t for t in node['threats'] if not set(map(tuple, t)) & set(reply)), None)
                    if threat is None:
                        return labels, None, None, ()
                    node, turns = dict(kind='immediate_win', action=threat), 1
                else:
                    index = self.replies[index].get(tuple(sorted(reply)))
                    if index is None:
                        return labels, None, None, ()
                    continue
            action = [tuple(a) for a in node['action']]
            played = history[i:i+len(action)]
            if i < len(history):
                labels.append((i, 1, turns))
            if played[:1] and played[0] in action and len(action) > 1 and i+1 < len(history):
                labels.append((i+1, 1, turns))
            if len(played) < len(action):
                if not set(played) <= set(action):
                    return labels, None, None, ()
                return labels, ([a for a in action if a not in played], turns), node, played
            if set(played) != set(action) or node['kind'] == 'immediate_win' or (first and self.first_turn_only):
                return labels, None, None, ()
            index, i, first = node['child'], i+len(action), False


class Plan:
    """The solver work of one Engine slot across its searches (module contract). `budgets` is re-read from the
    slot at every search start, so each colour of a match uses its own; the proof a side plays from and the pending
    deep queries are kept per side. With leaf_nodes, the Plan also keeps root certificates from in-process leaf
    checks. The solver can be None when no external queries are enabled, so following these proofs needs no
    worker processes."""

    def __init__(self, solver, schedule=None, leaf_nodes=0):
        self.solver, self.schedule = solver, solver.schedule if solver is not None else schedule or Schedule()
        self.leaf_nodes = leaf_nodes
        self.proofs, self.deep, self.late, self.found = {}, {}, [], []
        self.threat = self.root = self.finalists = None
        self.defence_queries, self.defences, self.defence_base = [], [], ()
        self.defence_limit = 8
        self.nodes, self.budget, self.turns, self.pruned, self.following, self.deferrals = 0, 0, 0, [], False, 0
        self.awaiting_finish = False

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
        self.awaiting_finish = False
        if not history:
            return False
        player, other = mover(history), 1-mover(history)
        if not schedule.fixed_budgets:
            self.poll(history)
        if self.leaf_nodes and schedule.follow and not self.alive(player, history):
            # An immediate child proof was found before this position became the played root.
            proof = next((p for p in reversed(self.found) if p.base == history), None)
            if proof is not None:
                self.proofs[player] = proof
        self.following = schedule.follow and (active(budgets, schedule) or self.leaf_nodes) and self.move(player, history) is not None
        if self.following or not active(budgets, schedule):
            return False
        if len(history) % 2 == 0:
            if budgets.defence and history[:-1] == self.defence_base:
                self.apply_defences(slot, history[-1])
            if budgets.finalists and self.move(player, history) is None:
                checked(native.hxg_hold(tree.ptr, 1))
            return False
        self.defences, self.defence_base = [], history
        self.defence_limit = budgets.defence_candidates
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
        """Yield this slot for missing fixed verdicts, or use adaptive overrun and late-proof rules."""
        if self.schedule.nonblocking_fixed:
            if all(query.ready() for query in queries):
                return True
            self.solver.stats['deferred'] += 1
            return False
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
        """Apply threat ordering or verified defence candidates, and retain finalist proofs. False defers
        the slot to its next visit."""
        ptr = slot.tree.ptr
        if self.awaiting_finish:
            history = tuple(map(tuple, slot.tree.history))
            player = mover(history)
            required = ([self.root] if self.root is not None else [])
            if len(history) % 2 and player in self.deep:
                required.append(self.deep[player])
            if not self.defer(required):
                return False
            self.awaiting_finish = False
        if self.threat is not None:
            if self.schedule.nonblocking_fixed and not self.defer([self.threat]):
                return False
            if self.solver.consume(self.threat, block=False):
                proven, result = self.threat.result()
                self.spent(result)
                if proven:
                    if slot.solver.defence:
                        history = tuple(map(tuple, slot.tree.history))
                        turns = defence_turns(history, result['certificate'], min(self.defence_limit, slot.budget))
                        # After our complete turn the original threat attacker is the actual mover.
                        self.defence_queries = [(turn, self.solver.submit('defence', history+turn, 'mover',
                                                                          self.threat.budget)) for turn in turns]
                    else:
                        cells = np.ascontiguousarray(result['moves'], np.int64).reshape(-1, 2)
                        checked(native.hxg_priority(ptr, cells, len(cells)))
            else:
                self.solver.stats['dropped'] += 1
                self.late.append(self.threat)   # accounted once it completes; its verdict no longer acts
            self.threat = None
        if self.defence_queries:
            if not self.defer([query for _, query in self.defence_queries]):
                return False
            for turn, query in self.defence_queries:
                if query in self.late:
                    continue
                result = query.result()[1]
                self.spent(result)
                if defence_hit(result):
                    self.defences.append(turn)
            self.defence_queries = []
            self.apply_defences(slot)
        if self.finalists is not None:
            if not self.defer([query for _, query in self.finalists]):
                return False
            for action, query in self.finalists:
                if query in self.late:
                    continue
                proven, result = query.result()
                self.spent(result)
                if proven:
                    distance = 1+proof_plies(len(result['moves']), int(result['proof_turns']))
                    checked(native.hxg_mark_exact(ptr, int(action[0]), int(action[1]), 1-mover(slot.tree.history), distance))
                    self.turns = max(self.turns, int(result['proof_turns']))
                    self.pruned.append(action)
                    self.proven(query, slot.tree.history)
            self.finalists = None
            checked(native.hxg_hold(ptr, 0))
        return True

    def apply_defences(self, slot, first=None):
        """Admit surviving first stones, or their compatible second stones, with 5/k per surviving turn.

        The bonus is at most one initial transformed-Q range. Search still chooses the move; no exact labels
        or pruning follow from an UNKNOWN. A very small search admits at most its simulation budget in cells.
        """
        counts = Counter(turn[0] if first is None else next(c for c in turn if c != first)
                         for turn in self.defences if first is None or first in turn)
        cells = sorted(counts, key=lambda c: (-counts[c], c))[:slot.budget]
        if cells:
            checked(native.hxg_defence(slot.tree.ptr, np.ascontiguousarray(cells, np.int64),
                                       np.asarray([5.*counts[c]/self.defence_limit for c in cells]), len(cells)))

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
        verdict at a turn start), play the kept proof's stone unless the tree proved a shorter win, and set result proven (+1 proof, -1 a root the
        finalist marks left exact-lost, else 0), proof_turns, solver_nodes (nodes spent on this search's queries),
        solver_budget (their granted budgets), pruned (the finalists marked lost), proof_action (the remaining
        winning placements for proven +1) and proof: with proven +1 the
        Proof played from, with proven -1 the opponent's Proof after the chosen action when a finalist query proved
        it (else None). False defers the slot to its next visit."""
        history = tuple(map(tuple, slot.tree.history))
        player = mover(history)
        if self.schedule.nonblocking_fixed:
            required = ([self.root] if self.root is not None else [])
            if self.schedule.fixed_budgets and len(history) % 2 and player in self.deep:
                required.append(self.deep[player])
            if not self.defer(required):
                self.awaiting_finish = True
                return False
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
        if self.pruned:
            # Finalist marks and leaf certificates (propagated natively) already exclude these losses in the tree.
            result.update(slot.tree.result(0, 0, 0, 0))
            if result['action'] is None:
                # All sampled candidates were lost; choose an unvisited survivor from the improved policy.
                result['action'] = result['actions'][int(np.argmax(result['policy']))].tolist()
        winner = result['exact_winner']
        result.update(proven=0 if winner < 0 else 1 if winner == player else -1, proof_turns=0)
        move = self.move(player, history) if active(slot.solver, self.schedule) or self.leaf_nodes else None
        bound = proof_plies(len(move[0]), move[1]) if move is not None else None
        if move is not None and winner == player and result.get('proof_plies', bound) < bound:
            # The tree already proved a shorter win than the certificate: play the tree's shortest winning move.
            move = None
        if move is not None:
            q, r = map(int, move[0][0])
            if winner < 0:
                # The certificate stone wins, so the improved policy is restricted to proven winning stones.
                checked(native.hxg_mark_exact(slot.tree.ptr, q, r, player, bound))
                result.update(slot.tree.result(0, 0, 0, 0))
            result.update(action=[q, r], proven=1, proof_turns=move[1], proof=self.proofs[player],
                          proof_action=[list(a) for a in move[0]], proof_plies=bound)
            if self.solver is not None:
                self.solver.stats['followed'] += self.following
        elif self.pruned and native.hxg_exact(slot.tree.ptr) == 1-mover(history):
            played = history+(tuple(map(int, result['action'])),)
            result.update(proven=-1, proof_turns=self.turns, proof=next((p for p in self.found if p.base == played), None))
        result.update(solver_nodes=self.nodes, solver_budget=self.budget, pruned=self.pruned)
        return True

    def pending(self):
        """Adaptive budgets with follow: whether a root, finalist or deep query that may still label this game's
        rows is running."""
        if self.schedule.fixed_budgets or not self.schedule.follow:
            return False
        queries = [*self.deep.values(), *(q for q in self.late if q.point not in ('threat', 'defence'))]
        return any(not q.future.done() for q in queries)

    def close(self, slot, moves):
        """The game of `slot` ended with `moves`: with follow, label the rows every kept proof decides through
        slot.label(ply, proven, proof_turns, proof_action) (adaptive budgets first take the deep and late verdicts already in;
        fixed budgets drop the ones never consumed). Queries still running pass to the Solver, which accounts
        them when they complete."""
        if not self.schedule.fixed_budgets:
            self.poll(moves)
        pending = [self.threat, self.root, *(q for _, q in self.finalists or ()),
                   *(q for _, q in self.defence_queries), *self.deep.values(), *self.late]
        if self.solver is not None:
            self.solver.orphans += [q for q in pending if q is not None and q.outcome is None]
        if not self.schedule.follow:
            return
        for proof in self.found:
            for ply, proven, turns in proof.path(moves)[0]:
                labelled = slot.label(ply, proven, turns, proof.action(moves[:ply]) if proven > 0 else None)
                if self.solver is not None:
                    self.solver.stats['labelled'] += labelled
