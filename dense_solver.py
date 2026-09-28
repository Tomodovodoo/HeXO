"""Solver queries inside the dense actor and evaluator searches (dense_selfplay.Engine drives them).

Points (settings solver_root_nodes, solver_finalists, solver_finalist_nodes, solver_threat_nodes; 0 = off):
  root       at a turn start (a search with two placements left): does the side to move have a forced win,
             within root_nodes? A proof decides the move played for the whole turn, the certificate's first turn
             (both stones); the search, its tree and its recorded policy are unchanged and the rows of both
             placements get the exact value +1 (`proven`).
  threat     at a turn start: would the opponent have a forced win if it moved now with a fresh turn, within
             threat_nodes? Root actions on the certificate's threat cells are sampled first in the search's
             opening phase (hxg_priority). Ordering only: nothing is pruned.
  finalists  in a mid-turn search, at its last halving boundary (its end when it never halves): for each of the
             `finalists` best candidates b (hxg_stats scores), does the opponent have a forced win after our turn
             ends with b, within finalist_nodes? A proof marks b exact-lost (hxg_mark_exact: Q -1, ineligible), so
             the remaining rounds and the final selection discard it and the improved policy gives it no mass.
Only native-verified PROVEN_WIN results act; UNKNOWN is never a loss. Budgets are node counts, so a verdict is a
function of (position, attacker, nodes, build). SAFETY_MS is only the wall-clock cap: a query that reaches it, or
any other UNKNOWN whose reason is not the search's own 'no verified strategy', is a failure (Solver.stats), not a
verdict.

Determinism: a search that submits queries always leaves its slot until the Engine's next visit, and every verdict
is awaited before the search goes on (threat and finalists) or before its move is played (root). The asynchronous
backend (one IsolatedTactics worker process fed by one thread) and the synchronous one (NativeTactics in process)
therefore produce identical games; the first overlaps the queries with the other games' search work.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, fields
import time

import numpy as np

from neural_search import checked, native
from tactical_proof import PROVEN_WIN, IsolatedTactics, NativeTactics, build_hash

SAFETY_MS = 5000
POINTS = ('root', 'threat', 'finalist')
SEARCHED = 'no verified strategy'  # the reason of an UNKNOWN the search itself decided


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


def record(budgets):
    """The episode-level `solver` record of a game played with `budgets`: the budgets and the tactical build hash
    (the backend is left out: both give the same games)."""
    return dict(**{f.name: getattr(budgets, f.name) for f in fields(budgets)}, build_hash=build_hash())


class Query:
    """One submitted solver query; `result()` waits for it once and returns (proven, result dict)."""

    def __init__(self, solver, point, future):
        self.solver, self.point, self.future, self.outcome = solver, point, future, None

    def result(self):
        if self.outcome is None:
            start = time.perf_counter()
            result = self.future.result()
            self.solver.account(self.point, result, time.perf_counter()-start)
            self.outcome = result['status'] == PROVEN_WIN and bool(result.get('native_verified')), result
        return self.outcome


class Solver:
    """One tactical backend per process: asynchronous (an IsolatedTactics worker fed by one thread) or synchronous
    (NativeTactics, each query answered inside submit).

    stats: per point {queries, hits, nodes, solver_ms (the backend's own elapsed time)}, plus wait_ms (time the
    calling thread spent in submit and result, the search loop's cost), failures and last_failure (module contract).
    """

    def __init__(self, asynchronous=True):
        self.asynchronous, self.build_hash = asynchronous, build_hash()
        self.engine = IsolatedTactics() if asynchronous else NativeTactics()
        self.pool = ThreadPoolExecutor(1, thread_name_prefix='solver') if asynchronous else None
        self.stats = dict(points={p: dict(queries=0, hits=0, nodes=0, solver_ms=0.) for p in POINTS}, wait_ms=0.,
                          failures=0, last_failure=None)

    def submit(self, point, history, attacker, nodes):
        start = time.perf_counter()
        query = dict(nodes=nodes, ms=SAFETY_MS, attacker=attacker)
        history = [list(p) for p in history]
        if self.pool:
            future = self.pool.submit(self.engine.history, history, **query)
        else:
            future = Future()
            future.set_result(self.engine.history(history, **query))
        self.stats['wait_ms'] += (time.perf_counter()-start)*1000
        return Query(self, point, future)

    def account(self, point, result, waited):
        stats = self.stats['points'][point]
        proven = result['status'] == PROVEN_WIN and bool(result.get('native_verified'))
        if result.get('build_hash') not in (None, self.build_hash):
            raise ValueError('Tactical build changed while the solver was running; rebuild and restart')
        stats['queries'] += 1
        stats['hits'] += proven
        stats['nodes'] += int(result.get('nodes_used') or 0)
        stats['solver_ms'] += float(result.get('elapsed_ms') or 0.)
        self.stats['wait_ms'] += waited*1000
        if not proven and result.get('reason') != SEARCHED:
            self.stats['failures'] += 1
            self.stats['last_failure'] = result.get('reason')

    def summary(self, seconds):
        """Status fields: queries per second over `seconds`, per point hit rate and mean backend ms, mean wait ms
        per query, failures."""
        points = self.stats['points']
        queries = sum(p['queries'] for p in points.values())
        rate = lambda a, b: a/b if b else None
        return dict(asynchronous=self.asynchronous, build_hash=self.build_hash, queries=queries,
                    queries_per_second=rate(queries, seconds), wait_ms_per_query=rate(self.stats['wait_ms'], queries),
                    failures=self.stats['failures'], last_failure=self.stats['last_failure'],
                    **{f'{k}_hit_rate': rate(p['hits'], p['queries']) for k, p in points.items()},
                    **{f'{k}_ms': rate(p['solver_ms'], p['queries']) for k, p in points.items()},
                    **{f'{k}_queries': p['queries'] for k, p in points.items()})

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True)
        if hasattr(self.engine, 'close'):
            self.engine.close()


def mover(history):
    """The side to move after `history` placements (one opening placement, then turns of two)."""
    return ((len(history)+1)//2) % 2


class Plan:
    """The solver work of one Engine slot across its searches (module contract). `budgets` is re-read from the
    slot at every search start, so each colour of a match uses its own."""

    def __init__(self, solver):
        self.solver, self.carry = solver, None
        self.threat = self.root = self.finalists = None
        self.nodes, self.turns, self.pruned = 0, 0, []

    def begin(self, slot):
        """After hxg_begin: submit the turn-start queries or arm the finalist hold. True when a verdict must be
        awaited on a later visit (`ready`) before the search requests anything."""
        budgets, tree = slot.solver, slot.tree
        history = tree.history
        self.threat = self.root = self.finalists = None
        self.nodes, self.turns, self.pruned = 0, 0, []
        if self.carry is not None and self.carry[0] != tuple(history):
            self.carry = None
        if budgets is None or not budgets.active or not history:
            return False
        if len(history) % 2:
            if budgets.threat_nodes:
                self.threat = self.solver.submit('threat', history, 'opponent', budgets.threat_nodes)
            if budgets.root_nodes:
                self.root = self.solver.submit('root', history, 'mover', budgets.root_nodes)
            return self.threat is not None
        if budgets.finalists and self.carry is None:
            checked(native.hxg_hold(tree.ptr, 1))
        return False

    def ready(self, slot):
        """Apply the verdicts awaited before the search continues: threat ordering, finalist marks."""
        tree = slot.tree
        ptr = tree.ptr
        if self.threat is not None:
            proven, result = self.threat.result()
            self.threat, self.nodes = None, self.nodes+int(result.get('nodes_used') or 0)
            if proven:
                cells = np.ascontiguousarray(result['moves'], np.int64).reshape(-1, 2)
                checked(native.hxg_priority(ptr, cells, len(cells)))
        if self.finalists is not None:
            winner = 1-mover(tree.history)
            for action, query in self.finalists:
                proven, result = query.result()
                self.nodes += int(result.get('nodes_used') or 0)
                if proven:
                    checked(native.hxg_mark_exact(ptr, int(action[0]), int(action[1]), winner))
                    self.turns = max(self.turns, int(result['proof_turns']))
                    self.pruned.append(action)
            self.finalists = None
            checked(native.hxg_hold(ptr, 0))

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
        """Before the move is played: await the root verdict, override the played action with the proof's turn,
        and set result proven (+1 proof, -1 a root the finalist marks left exact-lost, else 0), proof_turns,
        solver_nodes (nodes spent on this search's queries) and pruned (the finalists marked lost)."""
        history = tuple(slot.tree.history)
        result.update(proven=0, proof_turns=0)
        if self.root is not None:
            proven, answer = self.root.result()
            self.root, self.nodes = None, self.nodes+int(answer.get('nodes_used') or 0)
            if proven:
                moves = [tuple(m) for m in answer['moves']]
                self.carry = (history+(moves[0],), moves[1], answer['proof_turns']) if len(moves) > 1 else None
                result.update(action=list(moves[0]), proven=1, proof_turns=answer['proof_turns'])
        elif self.carry is not None and self.carry[0] == history:
            result.update(action=list(self.carry[1]), proven=1, proof_turns=self.carry[2])
            self.carry = None
        elif self.pruned and native.hxg_exact(slot.tree.ptr) == 1-mover(history):
            result.update(proven=-1, proof_turns=self.turns)
        result.update(solver_nodes=self.nodes, pruned=self.pruned)
