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
bind('hxg_fulfill', C.c_int, ptr, C.c_int, ints, doubles, doubles, C.c_int)
bind('hxg_cancel', None, ptr)
bind('hxg_advance', C.c_int, ptr, C.c_int64, C.c_int64)
bind('hxg_stats', C.c_int, ptr, ptr, ptr, ptr, ptr)
bind('hxg_policy', C.c_int, ptr, ptr)
bind('hxg_completed', C.c_int, ptr)
bind('hxg_tactics', C.c_int, ptr, C.c_int)
bind('hxg_exact', C.c_int, ptr)
bind('hxg_prove', C.c_int, ptr, C.c_int, ints, C.c_int, C.c_int, C.c_int, ints, C.c_int)

def checked(ok):
    if not ok:
        raise ValueError(native.hxg_error().decode())

class EvaluationCache:
    """Exact colored-state/phase/model keys. No visit statistics are shared."""
    def __init__(self, capacity=4096):
        self.capacity, self.entries = capacity, OrderedDict()

    def key(self, history, version):
        game = Game(history)
        try:
            return (version, game.player, game.remaining, tuple(sorted(tuple(cell) for cell in game.cells)))
        finally:
            game.close()

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
    def __init__(self, evaluator, model_version, history=(), seed=0, cache=None,
                 tactics=False, proof_solver=None, proof_ms=100):
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
        try:
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

    def search(self, simulations=128, root_samples=None, batch_size=16, milliseconds=None):
        coordinator = SearchCoordinator(self.evaluator, self.model_version, self.cache)
        return coordinator.search_many([self], simulations, root_samples, batch_size, milliseconds)[0]

    def fulfill_proof(self, request, history, certificate, milliseconds=None):
        """Verify a certificate against this pending state before exact backup."""
        if self.proof_solver is None:
            raise ValueError('A native certificate verifier is required')
        result = self.proof_solver.history(history, ms=self.proof_ms if milliseconds is None else milliseconds,
                                          idtt_ms=0, certificate=certificate)
        if result.get('status') != 'PROVEN_WIN' or not result.get('native_verified'):
            return False
        moves = result.get('moves', [])
        if not moves:
            return False
        game = Game(history)
        try:
            h = np.ascontiguousarray(history, dtype=np.int64).reshape(-1, 2)
            checked(native.hxg_prove(self.ptr, request, h, len(h), game.player,
                                    game.remaining, np.ascontiguousarray(moves, dtype=np.int64), len(moves)))
        finally:
            game.close()
        return True

    def result(self, start, finished, evaluated, hits):
        n = native.hxg_stats(self.ptr, None, None, None, None)
        actions = np.empty((n, 2), np.int64)
        visits = np.empty(n, np.int32)
        values, scores = np.empty(n), np.empty(n)
        native.hxg_stats(self.ptr, actions.ctypes.data, visits.ctypes.data, values.ctypes.data, scores.ctypes.data)
        policy = np.empty(n)
        native.hxg_policy(self.ptr, policy.ctypes.data)
        selected = int(np.argmax(scores)) if n and np.isfinite(scores).any() else None
        winner = native.hxg_exact(self.ptr)
        return dict(action=actions[selected].tolist() if selected is not None else None,
                    actions=actions, visits=visits, values=values, policy=policy,
                    completed=native.hxg_completed(self.ptr), evaluated=evaluated, cache_hits=hits,
                    elapsed_ms=(finished-start)*1000,
                    exact_winner=winner,
                    proof_status=('UNKNOWN' if winner < 0 else
                                  'PROVEN_WIN' if winner == ((len(self.history)+1)//2)%2 else 'PROVEN_LOSS'))



class SearchCoordinator:
    """Persistent evaluator/cache serving independent native trees round-robin."""
    def __init__(self, evaluator, model_version, cache=None):
        self.evaluator, self.model_version = evaluator, model_version
        self.cache = cache if cache is not None else EvaluationCache()

    def search_many(self, searches, simulations=128, root_samples=None, batch_size=16, milliseconds=None):
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
                done = native.hxg_completed(searches[i].ptr) >= budgets[i]
                expired = limits[i] is not None and (now-starts[i])*1000 >= limits[i]
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
                                    proof = search.proof_solver.history(history, ms=allowance, idtt_ms=0)
                                    proof_spent[i] += (time.perf_counter()-proof_start)*1000
                                    if finished(i):
                                        continue
                                    allowance = proof_budget()
                                    if (allowance and proof.get('status') == 'PROVEN_WIN' and proof.get('native_verified')):
                                        proof_start = time.perf_counter()
                                        fulfilled = search.fulfill_proof(request, history, proof['certificate'], allowance)
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
        return [search.result(starts[i], finishes[i] or time.perf_counter(), evaluated[i], hits[i])
                for i, search in enumerate(searches)]
