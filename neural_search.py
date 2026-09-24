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
    def __init__(self, evaluator, model_version, history=(), seed=0, cache=None):
        if not model_version:
            raise ValueError('A model version is required')
        self.evaluator, self.model_version = evaluator, model_version
        self.cache = cache if cache is not None else EvaluationCache()
        self.ptr = native.hxg_new(seed)
        if not self.ptr:
            raise MemoryError('Native tree allocation failed')
        self.history = []
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
        if batch_size < 1 or simulations < 1 or (milliseconds is not None and milliseconds <= 0):
            raise ValueError('Positive search budgets required')
        root_samples = max(2, int(simulations**0.5)) if root_samples is None else root_samples
        start = time.perf_counter()
        checked(native.hxg_begin(self.ptr, simulations, root_samples))
        evaluated, hits = 0, 0
        try:
            while native.hxg_completed(self.ptr) < simulations:
                if milliseconds is not None and (time.perf_counter()-start)*1000 >= milliseconds:
                    break
                pending = []
                while len(pending) < batch_size:
                    if milliseconds is not None and (time.perf_counter()-start)*1000 >= milliseconds:
                        break
                    request, history = self.request()
                    if request == -1:
                        continue
                    if not request:
                        break
                    key = self.cache.key(history, self.model_version)
                    cached = self.cache.get(key)
                    if cached is not None:
                        self.fulfill(request, cached)
                        hits += 1
                    else:
                        pending.append((request, history, key))
                if milliseconds is not None and (time.perf_counter()-start)*1000 >= milliseconds:
                    break
                if pending:
                    predictions = self.evaluator.evaluate([x[1] for x in pending])
                    if len(predictions) != len(pending):
                        raise ValueError('Evaluator returned the wrong batch size')
                    for (request, history, key), prediction in zip(pending, predictions):
                        self.fulfill(request, prediction)
                        self.cache.put(key, {k: np.asarray(prediction[k]).copy() for k in ('actions', 'logits', 'q')})
                    evaluated += len(pending)
                elif native.hxg_completed(self.ptr) < simulations:
                    game = Game(self.history)
                    try:
                        if game.winner >= 0:
                            break
                    finally:
                        game.close()
        finally:
            native.hxg_cancel(self.ptr)
        n = native.hxg_stats(self.ptr, None, None, None, None)
        actions = np.empty((n, 2), np.int64)
        visits = np.empty(n, np.int32)
        values, scores = np.empty(n), np.empty(n)
        native.hxg_stats(self.ptr, actions.ctypes.data, visits.ctypes.data, values.ctypes.data, scores.ctypes.data)
        policy = np.empty(n)
        native.hxg_policy(self.ptr, policy.ctypes.data)
        selected = int(np.argmax(scores)) if n and np.isfinite(scores).any() else None
        return dict(action=actions[selected].tolist() if selected is not None else None,
                    actions=actions, visits=visits, values=values, policy=policy,
                    completed=native.hxg_completed(self.ptr), evaluated=evaluated, cache_hits=hits,
                    elapsed_ms=(time.perf_counter()-start)*1000, proof_status='UNKNOWN')

