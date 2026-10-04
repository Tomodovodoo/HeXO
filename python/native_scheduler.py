"""Native adaptive views and one frozen-model neural queue, called by one coordinator.

This is an explicit search API. Actors keep their existing target construction.
Graph evidence and comparison credits are exported separately.
"""
import ctypes as C
import numpy as np
from neural_search import native, bind, ptr, checked
from native_feed import NativeFeed


bind('hxgm_new', ptr, ptr, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
     C.c_uint64, C.c_char_p, C.c_uint64)
for name, result, args in (
    ('free', C.c_int, [ptr]), ('step', C.c_int, [ptr]), ('cancel', C.c_int, [ptr]),
    ('done', C.c_int, [ptr]), ('feed', ptr, [ptr]), ('admit', C.c_int, [ptr]),
    ('clock', C.c_int, [ptr, C.c_double]), ('owner', ptr, [ptr, C.c_int]),
    ('ready_limit', C.c_int, [ptr, C.c_int]),
    ('workers', C.c_int, [ptr, C.c_int]),
    ('cancel_game', C.c_int, [ptr, C.c_int]),
    ('retarget', C.c_int, [ptr, C.c_int, ptr, C.c_int, C.c_uint64, C.c_double]),
    ('stats', None, [ptr, ptr]), ('install', C.c_int, [ptr, ptr, C.c_int, ptr, ptr, ptr, ptr]),
):
    bind('hxgm_'+name, result, *args)
for name, result, args in (
    ('root', ptr, [ptr]), ('choice', C.c_int, [ptr, ptr]), ('stats', None, [ptr, ptr]),
    ('records', C.c_int, [ptr, ptr, ptr]), ('history', C.c_int, [ptr, ptr]),
    ('record_history', C.c_int, [ptr, C.c_int, ptr]), ('policy_audit', C.c_int, [ptr, ptr]),
):
    bind('hxgo_'+name, result, *args)
bind('hxgf_abandon_all', C.c_int, ptr)
bind('hxp_new', ptr, ptr, ptr, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int)
for name, result, args in (
    ('step', C.c_int, [ptr]), ('cancel', None, [ptr]), ('resume', None, [ptr]),
    ('drain', C.c_int, [ptr]), ('free', C.c_int, [ptr]),
    ('offer', C.c_int, [ptr, C.c_int, ptr, C.c_int, C.c_double]),
    ('stats', None, [ptr, ptr, ptr]), ('record', C.c_char_p, [ptr, C.c_int]),
):
    bind('hxp_'+name, result, *args)


class ProofLoop:
    """Native frontier and immutable jobs; Python never dispatches individual queries.

    A slice is a CPU scheduling quantum. UNKNOWN remains unknown. Resident tables
    survive slices and retargets; this does not restore a complete interrupted PN tree.
    """
    def __init__(self, pool, package=None, *, workers=2, queue=8, slice_ms=8, table_mb=4,
                 tasks=256, stamps=False):
        from tactical_proof import NativeTactics, PACKAGE
        self.pool = pool
        self.library = NativeTactics(PACKAGE if package is None else package)
        names = ('worker_new', 'worker_free', 'worker_answer', 'answer_info', 'answer_moves',
                 'answer_json', 'answer_free', 'free', 'prepare', 'cancel', 'release', 'worker_busy')
        functions = np.asarray([C.cast(getattr(self.library.lib, 'hexo_tactical_'+name), ptr).value
                                for name in names], np.uint64)
        self._ptr = native.hxp_new(pool.ptr, functions.ctypes.data, workers, queue, slice_ms,
                                   table_mb, tasks, bool(stamps))
        if not self._ptr:
            checked(False)

    @property
    def ptr(self):
        if not self._ptr:
            raise ValueError('Proof loop is closed')
        return self._ptr

    def step(self):
        checked(native.hxp_step(self.ptr))

    def cancel(self):
        native.hxp_cancel(self.ptr)

    def resume(self):
        native.hxp_resume(self.ptr)

    def drain(self):
        """Stop admission, cancel outstanding jobs and wait for their native completion."""
        checked(native.hxp_drain(self.ptr))

    def offer(self, game, history, relevance=1.):
        """Explicit analysis candidate. Normal candidates come from native graph evidence."""
        cells = np.ascontiguousarray(history, np.int64).reshape(-1, 2)
        checked(native.hxp_offer(self.ptr, game, cells.ctypes.data, len(cells), relevance))

    def stats(self):
        out, times = np.empty(16, np.uint64), np.empty(4, np.float64)
        native.hxp_stats(self.ptr, out.ctypes.data, times.ctypes.data)
        result = dict(zip(('ticks', 'submitted', 'started', 'finished', 'installed', 'cancelled',
                           'pruned', 'unknown', 'fresh_nodes', 'missing_fresh', 'queued', 'active',
                           'ready', 'tasks', 'facts', 'records'), map(int, out)))
        result.update(zip(('worker_service_ms', 'worker_idle_ms', 'snapshot_ms', 'install_ms'),
                          map(float, times)))
        return result

    def records(self):
        """Inspect verified evidence and its actual conditional request context."""
        import json
        return [json.loads(native.hxp_record(self.ptr, i)) for i in range(self.stats()['records'])]

    def close(self):
        if self._ptr:
            self.drain()
            checked(native.hxp_free(self._ptr))
            self._ptr = None
            self.library.close()
            self.pool.proofs = None


class _Feed(NativeFeed):
    def __init__(self, pool):
        self.pool = pool

    @property
    def ptr(self):
        return native.hxgm_feed(self.pool.ptr)

    def _install(self, ids, *outputs):
        checked(native.hxgm_install(self.pool.ptr, ids.ctypes.data, len(ids), *outputs))
        return ()

    def close(self):
        raise ValueError('The scheduler owns this feed')


class SearchView:
    """Read a scheduled game's focus and its completed, position-bound search records."""
    def __init__(self, pool, index):
        self.pool, self.index = pool, index

    @property
    def ptr(self):
        return native.hxgm_owner(self.pool.ptr, self.index)

    def history(self):
        count = native.hxgo_history(self.ptr, None)
        out = np.empty((count, 2), np.int64)
        native.hxgo_history(self.ptr, out.ctypes.data)
        return out.tolist()

    def choice(self):
        out = np.empty(2, np.int64)
        status = native.hxgo_choice(self.ptr, out.ctypes.data)
        if status < 0:
            checked(False)
        return out.tolist() if status else None

    def stats(self):
        out = np.empty(20, np.uint64)
        native.hxgo_stats(self.ptr, out.ctypes.data)
        return dict(zip(('ticks', 'completed', 'issued', 'cancelled', 'created', 'retired',
                         'candidates', 'live', 'pending', 'depth', 'root_completed', 'deadline',
                         'step_ns', 'discover_ns', 'records', 'slots', 'last_root_credits',
                         'root_passes', 'allocations', 'reclaimed'), map(int, out)))

    def evidence(self):
        """Complete legal set, shared Q evidence and three distinct sampling count bases.

        This reports inputs to target construction. It does not choose a teacher temperature.
        """
        count = native.hxgo_policy_audit(self.ptr, None)
        out = np.empty((count, 9), np.float64)
        native.hxgo_policy_audit(self.ptr, out.ctypes.data)
        return dict(actions=out[:, :2].astype(np.int64), logits=out[:, 2], completed_q=out[:, 3],
                    shared_visits=out[:, 4].astype(np.int64), current_credits=out[:, 5].astype(np.int64),
                    last_credits=out[:, 6].astype(np.int64), eligible=out[:, 7].astype(bool),
                    lifetime_credits=out[:, 8].astype(np.int64))

    def records(self):
        count = native.hxgo_records(self.ptr, None, None)
        metadata = np.empty((count, 10), np.uint64)
        values = np.empty((count, 3), np.float64)
        native.hxgo_records(self.ptr, metadata.ctypes.data, values.ctypes.data)
        rows = []
        for index, (a, b) in enumerate(zip(metadata, values)):
            size = native.hxgo_record_history(self.ptr, index, None)
            history = np.empty((size, 2), np.int64)
            native.hxgo_record_history(self.ptr, index, history.ctypes.data)
            rows.append(dict(game=self.index, view=int(a[0]), generation=int(a[1]),
                             comparison_credits=int(a[2]), shared_evidence=int(a[3]),
                             context=[int(a[4]), int(a[5])], depth=int(a[6]),
                             exact_winner=int(a[7])-1, history=history.tolist(),
                             root_estimate=float(b[0]) if a[9] else None,
                             raw_value=float(b[1]) if a[8] else None,
                             elapsed_ms=float(b[2])))
        return rows


class SearchPool:
    """Independent games share neural work; same-game views share graph evidence.

    `work` is an optional diagnostic ceiling, not the unit for speed acceptance.
    Arm a common clock after backend setup. Queued work can be abandoned only
    after every submitted GPU batch is fenced. close() rejects undrained work.
    Native host workers own independent games during joined phases; caller-side
    control, inspection, and proof delivery must stay between those phases.
    """
    def __init__(self, sources, quantum=64, views=8, depth=8, work=128, seed=220, cache=8192, workers=1):
        if not sources or any(not s.ptr for s in sources):
            raise ValueError('Open shared graphs are required')
        versions = {s.model_version for s in sources}
        if len(versions) != 1:
            raise ValueError('One search pool requires one fixed model version')
        self.model_version = versions.pop()
        pointers = np.asarray([s.ptr for s in sources], np.uintp)
        self._ptr = native.hxgm_new(pointers.ctypes.data, len(sources), cache, quantum, views,
                                   depth, work, self.model_version.encode(), seed)
        if not self._ptr:
            checked(False)
        self.games = [SearchView(self, i) for i in range(len(sources))]
        self.feed = _Feed(self)
        self.proofs = None
        try:
            checked(native.hxgm_workers(self.ptr, workers))
        except BaseException:
            self.close()
            raise

    def enable_proofs(self, package=None, **options):
        """Opt in to concurrent proving. Actor/learner target construction is unchanged."""
        if self.proofs is not None:
            raise ValueError('Search pool already has a proof loop')
        self.proofs = ProofLoop(self, package, **options)
        return self.proofs

    @property
    def ptr(self):
        if not self._ptr:
            raise ValueError('Search pool is closed')
        return self._ptr

    def clock(self, ms):
        checked(native.hxgm_clock(self.ptr, ms))

    def step(self):
        status = native.hxgm_step(self.ptr)
        if status < 0:
            checked(False)
        return status

    def admit(self):
        status = native.hxgm_admit(self.ptr)
        if status < 0:
            checked(False)
        return bool(status)

    def done(self):
        return bool(native.hxgm_done(self.ptr))

    def cancel(self, game=None):
        if game is None:
            checked(native.hxgm_cancel(self.ptr))
        else:
            checked(native.hxgm_cancel_game(self.ptr, game))

    def retarget(self, game, history, work=0, ms=0):
        cells = np.ascontiguousarray(history, np.int64).reshape(-1, 2)
        checked(native.hxgm_retarget(self.ptr, game, cells.ctypes.data, len(cells), work, ms))

    def stats(self):
        out = np.empty(5, np.uint64)
        native.hxgm_stats(self.ptr, out.ctypes.data)
        return dict(zip(('steps', 'games', 'active', 'failed', 'retargets'), map(int, out)))

    def abandon_fenced(self):
        """Caller has completed every GPU fence. No cancelled row may still read graph storage."""
        self.cancel()
        checked(native.hxgf_abandon_all(self.feed.ptr))

    def limit_ready(self, rows):
        """Pause between complete layer gathers; 0 drains every game and view."""
        checked(native.hxgm_ready_limit(self.ptr, rows))

    def run(self, evaluator, ms, batch_size=128, ready_limit=None, overlap=True):
        """Use dense_selfplay.Evaluator's packed forwards under one common clock.
        Native gathering runs while the previous forward is in flight; ready
        inference can run while decoded prior results install. overlap=False
        retains the serial handoff for timing comparisons.
        """
        if evaluator.model_version != self.model_version:
            raise ValueError('Evaluator does not match the pool weights')
        from native_dense import submit
        pending = completed = None
        if batch_size < 1:
            raise ValueError('A neural batch must contain at least one row')
        self.limit_ready(2*batch_size if ready_limit is None else ready_limit)
        self.clock(ms)
        if self.proofs is not None:
            self.proofs.resume()
        try:
            while self.admit():
                self.step()
                if pending is not None:
                    ids, handle = pending
                    completed = ids, handle.collect()
                    pending = None
                if completed is not None and not overlap:
                    ids, rows = completed
                    self.feed.install_packed(ids, rows)
                    completed = None
                if self.admit():
                    batch = self.feed.take_packed(batch_size)
                    if batch is not None:
                        ids, rows = batch
                        pending = ids, submit(evaluator, rows)
                if completed is not None:
                    ids, rows = completed
                    self.feed.install_packed(ids, rows)
                    completed = None
                if self.done():
                    break
            self.cancel()
            if pending is not None:
                ids, handle = pending
                self.feed.install_packed(ids, handle.collect())
                pending = None
        finally:
            try:
                try:
                    self.cancel()
                finally:
                    try:
                        if completed is not None:
                            completed[1].close()
                    finally:
                        if pending is not None:
                            pending[1].close()
                    self.abandon_fenced()
            finally:
                if self.proofs is not None:
                    self.proofs.drain()

    def close(self):
        if self._ptr:
            if self.proofs is not None:
                self.proofs.close()
            checked(native.hxgm_free(self._ptr))
            self._ptr = None
