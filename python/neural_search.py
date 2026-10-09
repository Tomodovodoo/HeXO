"""Persistent native placement-tree search with batched neural leaf evaluation."""
import ctypes as C
from collections import OrderedDict
import time
import numpy as np
from hexo import Game, library

native = C.CDLL(str(library.with_name(library.name.replace('hexo', 'hexo_gumbel'))))
def bind(name, result, *args):
    fn = getattr(native, name)
    fn.restype, fn.argtypes = result, list(args)
    return fn
ptr = C.c_void_p
ints = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
bind('hxg_new', ptr, C.c_uint64)
bind('hxg_free', None, ptr)
bind('hxg_error', C.c_char_p)
bind('hxg_begin', C.c_int, ptr, C.c_int, C.c_int)
bind('hxg_next', C.c_int, ptr)
bind('hxg_history', C.c_int, ptr, C.c_int, ptr)
bind('hxg_legal', C.c_int, ptr, C.c_int, ptr)
if hasattr(native, 'hxg_encode'):
    bind('hxg_encode', C.c_int, ptr, C.c_int, ptr, C.c_int, ptr, ptr)
bind('hxg_fulfill', C.c_int, ptr, C.c_int, ints, doubles, doubles, C.c_int)
bind('hxg_cancel', None, ptr)
bind('hxg_advance', C.c_int, ptr, C.c_int64, C.c_int64)
bind('hxg_stats', C.c_int, ptr, ptr, ptr, ptr, ptr)
bind('hxg_policy', C.c_int, ptr, ptr)
bind('hxg_completed', C.c_int, ptr)
bind('hxg_done', C.c_int, ptr)
bind('hxg_tactics', C.c_int, ptr, C.c_int)
bind('hxg_graph', C.c_int, ptr, C.c_int)
bind('hxg_q_range_floor', C.c_int, ptr, C.c_double)
bind('hxg_root_noise', C.c_int, ptr, C.c_double)
bind('hxg_root_concentration', C.c_int, ptr, C.c_double)
bind('hxg_root_temperature', C.c_int, ptr, C.c_double)
bind('hxg_round_barrier', C.c_int, ptr, C.c_int)
bind('hxg_census', C.c_int, ptr, ptr)
bind('hxg_exact', C.c_int, ptr)
bind('hxg_distance', C.c_int, ptr)
bind('hxg_prove', C.c_int, ptr, C.c_int, ints, C.c_int, C.c_int, C.c_int, ints, C.c_int, C.c_int)
bind('hxg_hold', C.c_int, ptr, C.c_int)
bind('hxg_priority', C.c_int, ptr, ints, C.c_int)
if hasattr(native, 'hxg_defence'):
    bind('hxg_defence', C.c_int, ptr, ints, doubles, C.c_int)
bind('hxg_mark_exact', C.c_int, ptr, C.c_int64, C.c_int64, C.c_int, C.c_int)
bind('hxg_share', C.c_int, ptr, C.c_int64)
bind('hxg_archive', C.c_int, ptr, C.c_int64)
bind('hxg_archive_forward', C.c_int, ptr, C.c_int)
bind('hxg_archive_stats', C.c_int, ptr, ptr)
bind('hxg_root_at', C.c_int, ptr, ints, C.c_int)
bind('hxg_store', C.c_int, ptr, ptr)
bind('hxg_root_version', C.c_int64, ptr)
bind('hxg_q', C.c_int, ptr, ptr)
bind('hxg_facts', C.c_int, ptr, ptr, C.c_int)
bind('hxg_prove_loss', C.c_int, ptr, C.c_int, C.c_int)
bind('hxg_value', C.c_double, ptr)
if hasattr(native, 'hxg_view'):
    bind('hxg_view', ptr, ptr, ints, C.c_int, C.c_uint64)
    bind('hxg_view_counters', None, ptr, ptr)
    bind('hxg_root_credits', C.c_int, ptr, ptr)
HOLD = -3  # hxg_next: the search waits at its armed hold
GRAPH_LIMIT = 4096  # expanded nodes a GameGraph keeps between searches, about 56 KB each at 640 legal moves
PV_DROP = .05       # completed-Q fall (value units, -1 to 1) of the chosen stone that sends a checked search back

def checked(ok):
    if not ok:
        raise ValueError(native.hxg_error().decode())

class EvaluationCache:
    """Colored stones, turn context and model keys. No visit statistics are shared."""
    def __init__(self, capacity=4096):
        self.capacity, self.entries = capacity, OrderedDict()

    def key(self, history, version):
        # Dense inputs also identify this turn's first stone and the opponent's previous turn.
        size=len(history)
        start = size if size % 2 or size == 0 else size-1
        turn = tuple((int(q), int(r)) for q, r in history[start:size])
        previous = tuple(sorted((int(q), int(r)) for q, r in history[max(0, start-2):start]))
        return (version, ((size+1)//2)%2, 2 if size%2 else 1,
                tuple(sorted((int(q),int(r),((i+1)//2)%2) for i,(q,r) in enumerate(history))),
                turn, previous)

    def get(self, key):
        value = self.entries.get(key)
        if value is not None:
            self.entries.move_to_end(key)
        return value

    def put(self, key, value):
        self.entries[key] = value
        self.entries.move_to_end(key)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)

class NeuralSearch:
    """One native tree. `q_range_floor` is the least Q range of the completed-Q rescale (0 keeps mctx's
    min-max rescale); it applies to every search and policy target of the tree. `root_noise` e in [0, 1) is the
    uniform share of the root's candidate sampling: Gumbel-top-k draws the root samples from (1 - e) p + e / N over
    the N eligible moves instead of the prior p, while halving, the final choice and the policy target keep p.
    `root_concentration` c > 0 replaces the uniform 1 / N by KataGo's shaped Dirichlet noise, drawn once per search
    at total concentration c: half of c spread evenly over the eligible moves, half in proportion to how far each
    move's log min(p, 0.01) lies above their mean. `root_temperature` T > 0 divides the logits that sampling and
    that noise shape read, before the noise is mixed in; halving, the final choice and the policy target keep them.
    `limit`, when given, makes the tree a shared game graph (see GameGraph) keeping at most that many expanded nodes
    between searches (0: no bound). `archive_bytes` optionally retains up to 256 dormant expansions under a managed
    payload/index byte allowance (at least 64 KiB); it does not bound allocator residency or the active graph.
    `archive_forward` releases opposite-colour conflicts with the primary played board at safe owner points;
    leave it false when retaining analysis for undo. `round_barrier` keeps sampled root candidates fixed
    within each halving round while permitting concurrent work across visit layers; default false."""
    def __init__(self, evaluator, model_version, history=(), seed=0, cache=None,
                 tactics=False, proof_solver=None, proof_ms=100, graph=False, q_range_floor=0., root_noise=0.,
                 root_concentration=0., limit=None, archive_bytes=0, archive_forward=False, round_barrier=False,
                 root_temperature=1.):
        if not model_version:
            raise ValueError('A model version is required')
        self.evaluator, self.model_version = evaluator, model_version
        self.cache = cache if cache is not None else EvaluationCache()
        self.ptr = native.hxg_new(seed)
        if not self.ptr:
            raise MemoryError('Native tree allocation failed')
        self.history = []
        self.proof_solver, self.proof_ms = proof_solver, proof_ms
        checked(native.hxg_tactics(self.ptr, int(tactics)))
        checked(native.hxg_graph(self.ptr, int(graph)))
        try:
            if limit is not None:
                checked(native.hxg_share(self.ptr, int(limit)))
            if archive_bytes:
                checked(native.hxg_archive(self.ptr, int(archive_bytes)))
            if archive_forward:
                checked(native.hxg_archive_forward(self.ptr, 1))
            checked(native.hxg_q_range_floor(self.ptr, q_range_floor))
            checked(native.hxg_root_noise(self.ptr, root_noise))
            checked(native.hxg_root_concentration(self.ptr, root_concentration))
            checked(native.hxg_root_temperature(self.ptr, root_temperature))
            checked(native.hxg_round_barrier(self.ptr, int(round_barrier)))
            for point in history:
                self.advance(point)
        except Exception:
            self.close()
            raise

    def close(self):
        if self.ptr:
            native.hxg_free(self.ptr)
            self.ptr = None

    def advance(self, action):
        q, r = action
        if any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) for v in (q, r)):
            raise ValueError("Coordinates must be integers")
        checked(native.hxg_advance(self.ptr, int(q), int(r)))
        self.history.append((int(q), int(r)))

    def request(self):
        request = native.hxg_next(self.ptr)
        if request == -2:
            checked(False)
        if request <= 0:
            return request, None
        size = native.hxg_history(self.ptr, request, None)
        history = np.empty((size, 2), dtype=np.int64)
        native.hxg_history(self.ptr, request, history.ctypes.data)
        return request, history.tolist()

    def fulfill(self, request, prediction):
        raw_actions = np.asarray(prediction['actions'])
        if raw_actions.dtype.kind not in 'iu' or (raw_actions.size and
                (np.any(raw_actions < -10**12) or np.any(raw_actions > 10**12))):
            raise ValueError('Evaluator coordinates must be integers within +/- 10^12')
        actions = np.ascontiguousarray(raw_actions, dtype=np.int64)
        logits = np.ascontiguousarray(prediction['logits'], dtype=np.float64)
        q = np.ascontiguousarray(prediction['q'], dtype=np.float64)
        if actions.shape != (len(logits), 2) or q.shape != logits.shape or logits.ndim != 1:
            raise ValueError('Invalid evaluator shapes')
        checked(native.hxg_fulfill(self.ptr, request, actions, logits, q, len(q)))

    def search(self, simulations=128, root_samples=None, batch_size=16, milliseconds=None,
               *, stop=None, choice='policy', q_range_floor=None, root_noise=None):
        coordinator = SearchCoordinator(self.evaluator, self.model_version, self.cache)
        return coordinator.search_many([self], simulations, root_samples, batch_size, milliseconds,
                                       stop=stop, choice=choice, q_range_floor=q_range_floor,
                                       root_noise=root_noise)[0]

    def fulfill_proof(self, request, history, certificate, milliseconds=None):
        """Verify a certificate against this pending state before exact backup."""
        if self.proof_solver is None:
            raise ValueError('A native certificate verifier is required')
        result = self.proof_solver.history(history, ms=self.proof_ms if milliseconds is None else milliseconds,
                                          certificate=certificate)
        return self._install_verified_proof(request, history, result)

    def _install_verified_proof(self, request, history, result):
        """Install this leaf's already verified solver verdict; native checks its history, phase and turn."""
        if (result.get('status') != 'PROVEN_WIN' or not result.get('native_verified')
                or result.get('attacker') != 'mover'):
            return False
        moves = result.get('moves', [])
        if not moves:
            return False
        game = Game(history)
        try:
            h = np.ascontiguousarray(history, dtype=np.int64).reshape(-1, 2)
            checked(native.hxg_prove(self.ptr, request, h, len(h), game.player,
                                    game.remaining, np.ascontiguousarray(moves, dtype=np.int64), len(moves),
                                    int(result['proof_turns'])))
        finally:
            game.close()
        return True

    def expand(self):
        """Evaluate the root (through the cache) when it has no edges yet and the game goes on, so `mark` can
        settle its edges before a search."""
        if native.hxg_stats(self.ptr, None, None, None, None):
            return
        checked(native.hxg_begin(self.ptr, 1, 1))
        try:
            request, history = self.request()
            if request > 0:
                key = self.cache.key(history, self.model_version)
                prediction = self.cache.get(key)
                if prediction is None:
                    prediction = {k: np.asarray(v).copy() for k, v in self.evaluator.evaluate([history])[0].items()
                                  if k in ('actions', 'logits', 'q')}
                    self.cache.put(key, prediction)
                self.fulfill(request, prediction)
        finally:
            native.hxg_cancel(self.ptr)

    def mark(self, action, winner, distance):
        """Settle the expanded root's edge `action` as won by `winner` within `distance` placements, the stone
        itself included (hxg_mark_exact); ValueError when `action` is not a root edge."""
        checked(native.hxg_mark_exact(self.ptr, int(action[0]), int(action[1]), int(winner), int(distance)))

    def census(self):
        """{nodes, expanded, exact, duplicates} reachable from the root (native hxg_census)."""
        out = np.zeros(4, np.int64)
        native.hxg_census(self.ptr, out.ctypes.data)
        return dict(zip(('nodes', 'expanded', 'exact', 'duplicates'), map(int, out)))

    def facts(self):
        """Reachable exact outcomes as a bounded, worker-safe snapshot. Distances
        are placement upper bounds; no neural score is treated as a proof."""
        out = np.empty(100_000, np.int64)
        used = native.hxg_facts(self.ptr, out.ctypes.data, len(out))
        facts, i = [], 0
        while i < used and len(facts) < 2048:
            count, winner, plies = map(int, out[i:i+3])
            i += 3
            facts.append(dict(history=out[i:i+2*count].reshape(-1, 2).tolist(), winner=winner, plies=plies))
            i += 2*count
        return facts

    def prove_loss(self, winner, plies):
        """Install a verified loss at this root and propagate it to its parents."""
        checked(native.hxg_prove_loss(self.ptr, int(winner), int(plies)))

    def result(self, start, finished, evaluated, hits, *, choice='gumbel'):
        n = native.hxg_stats(self.ptr, None, None, None, None)
        actions = np.empty((n, 2), np.int64)
        visits = np.empty(n, np.int32)
        values, scores = np.empty(n), np.empty(n)
        native.hxg_stats(self.ptr, actions.ctypes.data, visits.ctypes.data, values.ctypes.data, scores.ctypes.data)
        policy = np.empty(n)
        native.hxg_policy(self.ptr, policy.ctypes.data)
        completed_q = np.empty(n)
        native.hxg_q(self.ptr, completed_q.ctypes.data)
        selected = int(np.argmax(scores)) if n and np.isfinite(scores).any() else None
        winner = native.hxg_exact(self.ptr)
        proven = 0 if winner < 0 else 1 if winner == ((len(self.history)+1)//2)%2 else -1
        if choice == 'policy' and not proven and n and policy.max() > 0:
            selected = int(np.argmax(policy))
        # A won root offers only its shortest winning moves, so those are the finite scores.
        shortest = [actions[i].tolist() for i in range(n) if np.isfinite(scores[i])] if proven > 0 else []
        return dict(action=actions[selected].tolist() if selected is not None else None,
                    actions=actions, visits=visits, values=values, policy=policy, scores=scores, completed_q=completed_q,
                    completed=native.hxg_completed(self.ptr), evaluated=evaluated, cache_hits=hits,
                    node_value=native.hxg_value(self.ptr),
                    elapsed_ms=(finished-start)*1000,
                    exact_winner=winner,
                    proven=proven, proof_turns=0, solver_nodes=0, solver_budget=0,
                    proof_plies=native.hxg_distance(self.ptr) if proven else 0, proof_action=shortest,
                    proof_status=('UNKNOWN' if winner < 0 else
                                  'PROVEN_WIN' if proven > 0 else 'PROVEN_LOSS'))


def pv_reserve(simulations, fraction):
    """Simulations a principal-variation check of share `fraction` (0 <= fraction < 0.5) reserves twice from a search
    of `simulations`: round(fraction * simulations), 0 when that is 0 or would leave the first pass none."""
    if not 0 <= fraction < .5:
        raise ValueError('pv_check must lie in [0, 0.5)')
    reserve = int(round(fraction*simulations))
    return reserve if reserve and simulations-2*reserve >= 1 else 0


class Recheck:
    """The principal-variation check of one search on a GameGraph, run as a sequence of searches at the graph's root.

    With reserve R = pv_reserve(simulations, fraction): the first pass searches the root A with simulations - 2R and
    chooses the turn (GameGraph.after_turn). The check moves the root to the position after that turn and searches it
    with R; its values reach A through the shared nodes. Back at A, when the chosen stone's completed Q fell by more
    than `drop`, A is searched again with the last R; otherwise those simulations are not spent. No check runs when
    R is 0, the first pass proved its root, its turn has no known second stone or the turn wins. `budget` is the
    first pass's simulations; give each finished search's result to `step`, which returns the next search's
    simulations (the root already moved to its position) or 0 once the root is back at A and the check is over."""

    def __init__(self, graph, simulations, fraction, drop=PV_DROP):
        self.graph, self.drop = graph, drop
        self.reserve = pv_reserve(simulations, fraction)
        self.budget = simulations-2*self.reserve
        self.root, self.phase, self.line, self.before, self.after = list(graph.history), 'first', None, None, None
        self.searched = False

    def step(self, result):
        if self.phase == 'first':
            self.phase = 'done'
            self.line = self.graph.after_turn(result) if self.reserve else None
            if self.line is None:
                return 0
            self.index = result['actions'].tolist().index(list(map(int, result['action'])))
            self.before = float(result['completed_q'][self.index])
            self.graph.at(self.line)
            self.phase = 'check'
            return self.reserve
        if self.phase == 'check':
            self.graph.at(self.root)
            self.after = float(self.graph.result(0, 0, 0, 0)['completed_q'][self.index])
            self.searched = self.before-self.after > self.drop
            self.phase = 'again' if self.searched else 'done'
            return self.reserve if self.searched else 0
        self.phase = 'done'
        return 0

    def abandon(self):
        """End the check before its next search: the root moves back to the first position."""
        if self.phase == 'check':
            self.graph.at(self.root)
        self.searched = self.searched and self.phase != 'again'
        self.phase = 'done'

    def summary(self):
        """{line: the checked turn's stones, before and after: the chosen stone's completed Q, searched: whether the
        root was searched again}, or None when no check ran."""
        if self.line is None:
            return None
        return dict(line=[list(p) for p in self.line[len(self.root):]], before=self.before, after=self.after,
                    searched=self.searched)


class GameGraph(NeuralSearch):
    """One game's shared search graph (native hxg_share). The store keeps every node the game's searches expanded,
    keyed by turn context, with its visits, values, exact marks and proof distances; an edge keeps its own visits
    and reads its stored child's value, so a search at a later position moves the values of every earlier position
    that reaches it, and each playout also counts as a visit of every stored edge along the root's history. `at(history)` moves the root to any position, stored or new; `advance` keeps the siblings of the played stone. Between
    searches at most `limit` expanded nodes are kept (0: no bound), the least recently used leaves leaving first.
    `search(..., pv_check=f)` adds the principal-variation check (Recheck)."""

    def __init__(self, evaluator, model_version, history=(), seed=0, cache=None, tactics=False, proof_solver=None,
                 proof_ms=100, q_range_floor=0., root_noise=0., root_concentration=0., limit=GRAPH_LIMIT, archive_bytes=0,
                 archive_forward=False, round_barrier=False, root_temperature=1.):
        super().__init__(evaluator, model_version, history, seed, cache, tactics, proof_solver, proof_ms,
                         q_range_floor=q_range_floor, root_noise=root_noise, root_concentration=root_concentration,
                         root_temperature=root_temperature,
                         limit=limit, archive_bytes=archive_bytes,
                         archive_forward=archive_forward, round_barrier=round_barrier)

    def at(self, history):
        """Move the root to the position after `history`, keeping every node's statistics."""
        cells = [(int(q), int(r)) for q, r in history]
        checked(native.hxg_root_at(self.ptr, np.asarray(cells, dtype=np.int64).reshape(-1, 2), len(cells)))
        self.history = cells

    def view(self, history=None, seed=0):
        """An independent search into this game's store. Calls from all views must use one owner thread.
        The view shares model/cache and position evidence, but has its own sampling, pending requests and credits.
        Closing either view leaves the other usable. Moving one view requires only its own requests to drain.
        """
        if not hasattr(native, 'hxg_view'):
            raise RuntimeError('Rebuild the native search library for independent graph views')
        if not self.ptr:
            raise ValueError('Graph is closed')
        cells = self.history if history is None else [(int(q), int(r)) for q, r in history]
        address = native.hxg_view(self.ptr, np.asarray(cells, dtype=np.int64).reshape(-1, 2), len(cells), seed)
        if not address:
            checked(False)
        view = object.__new__(GameGraph)
        view.ptr, view.history = address, list(cells)
        view.evaluator, view.model_version, view.cache = self.evaluator, self.model_version, self.cache
        view.proof_solver, view.proof_ms = self.proof_solver, self.proof_ms
        return view

    def counters(self):
        """This root comparison's issued/completed/cancelled simulations and this view's pending requests.
        `retired` counts late results since view creation; `views` counts live views into the same game.
        Shared visits are available separately in result().
        """
        out = np.zeros(6, np.uint64)
        native.hxg_view_counters(self.ptr, out.ctypes.data)
        return dict(zip(('issued', 'completed', 'cancelled', 'pending', 'retired', 'views'), map(int, out)))

    def credits(self):
        """Direct completed root comparison credits in result() action order, excluding inherited visits."""
        size = native.hxg_root_credits(self.ptr, None)
        out = np.empty(size, np.uint64)
        native.hxg_root_credits(self.ptr, out.ctypes.data)
        return out

    @property
    def root_version(self):
        """The number of root moves so far (native hxg_root_version): `at` and `advance` each add one."""
        return native.hxg_root_version(self.ptr)

    def store(self):
        """{nodes, expanded, evicted, limit, summaries: evicted nodes' statistics kept, outcomes: proven positions
        kept} of the store (native hxg_store)."""
        out = np.zeros(6, np.int64)
        native.hxg_store(self.ptr, out.ctypes.data)
        return dict(zip(('nodes', 'expanded', 'evicted', 'limit', 'summaries', 'outcomes'), map(int, out)))

    def archive(self):
        """Dormant evidence payload, estimated index bytes and reuse counts; excludes pool overhead/RSS."""
        out = np.zeros(10, np.int64)
        if not native.hxg_archive_stats(self.ptr, out.ctypes.data):
            return None
        return dict(zip(('nodes', 'bytes', 'limit', 'retained', 'reused', 'discarded', 'compatible',
                         'index_bytes', 'indexed_cells', 'focus_stones'), map(int, out)))

    def after_turn(self, result):
        """The history after the turn `result`, this root's finished search, chooses: its stone, then while the same
        side is to move the stone the graph's improved policy ranks first there. None when the result has no stone or
        a proof, the position after its stone was never expanded while the same side still moves, or the turn wins.
        The root is left where it was."""
        if result['action'] is None or result['proven']:
            return None
        root = list(self.history)
        line = [*root, tuple(map(int, result['action']))]
        game = Game(line)
        try:
            if game.winner >= 0:
                return None
            if game.player == player_of(len(root)):
                self.at(line)
                stats = self.result(0, 0, 0, 0)
                self.at(root)
                if not len(stats['policy']) or not stats['policy'].max() > 0:
                    return None
                line.append(tuple(map(int, stats['actions'][int(np.argmax(stats['policy']))])))
                game.play(*line[-1])
                if game.winner >= 0:
                    return None
        finally:
            game.close()
        return line

    def search(self, simulations=128, root_samples=None, batch_size=16, milliseconds=None, *, pv_check=0.,
               pv_drop=PV_DROP, **options):
        """NeuralSearch.search with the principal-variation check of share `pv_check` (Recheck). With a check the
        result is the root's after it, with `completed`, `evaluated`, `cache_hits` and `elapsed_ms` summed over every
        pass and `pv_check` (Recheck.summary). `milliseconds` caps all passes together: a pass gets what the earlier
        ones left, and the check ends, back at the root, when nothing is left."""
        start = time.perf_counter()
        check = Recheck(self, simulations, pv_check, pv_drop)
        passes = [super().search(check.budget, root_samples, batch_size, milliseconds, **options)]
        while budget := check.step(passes[-1]):
            left = None if milliseconds is None else milliseconds-(time.perf_counter()-start)*1000
            if left is not None and left <= 0:
                check.abandon()
                break
            passes.append(super().search(budget, root_samples, batch_size, left, **options))
        if len(passes) == 1:
            return passes[0]
        result = passes[-1] if check.searched else self.result(0, 0, 0, 0, choice=options.get('choice', 'policy'))
        result.update({key: sum(p[key] for p in passes) for key in ('completed', 'evaluated', 'cache_hits', 'elapsed_ms')},
                      pv_check=check.summary())
        return result


def player_of(length):
    """The side to move after `length` placements."""
    return (length+1)//2 % 2


class SearchCoordinator:
    """Persistent evaluator/cache serving independent native trees round-robin."""
    def __init__(self, evaluator, model_version, cache=None):
        self.evaluator, self.model_version = evaluator, model_version
        self.cache = cache if cache is not None else EvaluationCache()

    def search_many(self, searches, simulations=128, root_samples=None, batch_size=16, milliseconds=None,
                    *, stop=None, choice='policy', q_range_floor=None, root_noise=None):
        """Search every tree to its simulation budget, its `milliseconds` or until `stop()` returns true.

        `q_range_floor` and `root_noise`, when given, become every tree's floor and root noise (NeuralSearch) from
        this search on; None keeps each tree's own.

        The default choice is the highest improved policy; choice='gumbel' uses the
        final Gumbel score. Actors call result() directly and retain Gumbel exploration.
        """
        if choice not in ('policy', 'gumbel'):
            raise ValueError('choice must be policy or gumbel')
        self.last_stats = dict(inference_batches=0, unique_positions=0, largest_batch=0)
        searches = list(searches)
        if len({id(search) for search in searches}) != len(searches):
            raise ValueError('Each active search must be a distinct tree')
        if any(search.evaluator is not self.evaluator or search.model_version != self.model_version for search in searches):
            raise ValueError('A batch must share one frozen evaluator and model version')
        if batch_size < 1:
            raise ValueError('Positive batch size required')
        def expand(value):
            values = list(value) if isinstance(value, (list, tuple)) else [value]*len(searches)
            if len(values) != len(searches):
                raise ValueError('Per-tree budgets must match the tree count')
            return values
        budgets, samples, limits = expand(simulations), expand(root_samples), expand(milliseconds)
        if any(b < 1 for b in budgets) or any(t is not None and t <= 0 for t in limits):
            raise ValueError('Positive search budgets required')
        starts, finishes = [], [None]*len(searches)
        evaluated, hits = [0]*len(searches), [0]*len(searches)
        proof_spent = [0.]*len(searches)
        active = set()
        cursor = 0
        try:
            for i, search in enumerate(searches):
                starts.append(time.perf_counter())
                if q_range_floor is not None:
                    checked(native.hxg_q_range_floor(search.ptr, q_range_floor))
                if root_noise is not None:
                    checked(native.hxg_root_noise(search.ptr, root_noise))
                sample = max(2, int(budgets[i]**0.5)) if samples[i] is None else samples[i]
                checked(native.hxg_begin(search.ptr, budgets[i], sample))
                game = Game(search.history)
                try:
                    if game.winner < 0:
                        active.add(i)
                    else:
                        finishes[i] = time.perf_counter()
                finally:
                    game.close()
            def finished(i):
                now = time.perf_counter()
                done = native.hxg_done(searches[i].ptr)
                expired = (limits[i] is not None and (now-starts[i])*1000 >= limits[i]) or (stop is not None and stop())
                if done or expired:
                    active.discard(i)
                    finishes[i] = now
                    if expired:
                        native.hxg_cancel(searches[i].ptr)
                    return True
                return False
            while active:
                pending = []
                idle = 0
                while active and len(pending) < batch_size:
                    i = cursor % len(searches)
                    cursor += 1
                    if i not in active:
                        idle += 1
                    elif finished(i):
                        idle = 0
                    else:
                        request, history = searches[i].request()
                        if request == -1:
                            idle = 0
                        elif request == 0:
                            idle += 1
                        else:
                            idle = 0
                            search = searches[i]
                            if search.proof_solver is not None:
                                def proof_budget():
                                    return search.proof_ms if limits[i] is None else min(search.proof_ms,
                                        max(0, int(min(limits[i]/4-proof_spent[i],
                                            (limits[i]-(time.perf_counter()-starts[i])*1000)/4))))
                                allowance = proof_budget()
                                if allowance:
                                    proof_start = time.perf_counter()
                                    proof = search.proof_solver.history(history, ms=allowance)
                                    proof_spent[i] += (time.perf_counter()-proof_start)*1000
                                    if finished(i):
                                        continue
                                    if proof.get('status') == 'PROVEN_WIN' and proof.get('native_verified'):
                                        proof_start = time.perf_counter()
                                        fulfilled = search._install_verified_proof(request, history, proof)
                                        proof_spent[i] += (time.perf_counter()-proof_start)*1000
                                        if fulfilled:
                                            continue
                            key = self.cache.key(history, self.model_version)
                            cached = self.cache.get(key)
                            if cached is None:
                                pending.append((i, request, history, key))
                            else:
                                searches[i].fulfill(request, cached)
                                hits[i] += 1
                    if idle >= len(searches):
                        break
                # Expired requests never launch a new inference batch.
                for i in list(active):
                    finished(i)
                pending = [item for item in pending if item[0] in active]
                if pending:
                    grouped = OrderedDict()
                    for item in pending:
                        grouped.setdefault(item[3], []).append(item)
                    unique = list(grouped.values())
                    leaves = [(searches[items[0][0]].ptr, items[0][1], items[0][2]) for items in unique]
                    if hasattr(self.evaluator, 'evaluate_leaves'):
                        predictions = self.evaluator.evaluate_leaves(leaves)
                    else:
                        predictions = self.evaluator.evaluate([items[0][2] for items in unique])
                    self.last_stats["inference_batches"] += 1
                    self.last_stats["unique_positions"] += len(unique)
                    self.last_stats["largest_batch"] = max(self.last_stats["largest_batch"], len(unique))
                    if len(predictions) != len(unique):
                        raise ValueError('Evaluator returned the wrong batch size')
                    for items, prediction in zip(unique, predictions):
                        # IDs are local to each tree and are routed with its index.
                        for i, request, history, key in items:
                            searches[i].fulfill(request, prediction)
                            evaluated[i] += 1
                        self.cache.put(items[0][3], {k: np.asarray(prediction[k]).copy() for k in ('actions', 'logits', 'q')})
                elif active:
                    # Cache hits or terminal traversals can complete work without inference.
                    for i in list(active):
                        finished(i)
                    if active and idle >= len(searches):
                        raise RuntimeError('Native scheduler stalled without pending evaluations')
        finally:
            for search in searches:
                native.hxg_cancel(search.ptr)
        return [search.result(starts[i], finishes[i] or time.perf_counter(), evaluated[i], hits[i], choice=choice)
                for i, search in enumerate(searches)]
