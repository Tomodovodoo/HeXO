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
bind('hxb_new', ptr, C.c_int, C.c_int, C.c_int, C.c_double)
for name, result, args in (
    ('attach', C.c_int, [ptr, ptr, C.c_int]), ('start', C.c_int, [ptr, C.c_double]),
    ('detach', C.c_int, [ptr, C.c_int]), ('model_pending', C.c_int, [ptr, C.c_int]),
    ('workers', C.c_int, [ptr, C.c_int, C.c_int]),
    ('flights', C.c_int, [ptr, C.c_int]), ('flight_stats', None, [ptr, ptr]),
    ('profile', C.c_int, [ptr, C.c_int]), ('feedback', C.c_int, [ptr, C.c_int]), ('schedule_stats', None, [ptr, ptr]),
    ('reclaim_ready', C.c_int, [ptr]), ('reclaim', C.c_int, [ptr, ptr]),
    ('reclaim_stats', None, [ptr, ptr]), ('owner_reclaim_stats', None, [ptr, ptr]),
    ('cancel', None, [ptr]), ('take', C.c_int, [ptr, C.c_int, C.c_double, ptr, ptr, ptr]),
    ('complete', C.c_int, [ptr, C.c_uint64, ptr, ptr, ptr, ptr]),
    ('abort', C.c_int, [ptr, C.c_uint64]), ('done', C.c_int, [ptr]),
    ('join', C.c_int, [ptr]), ('free', C.c_int, [ptr]), ('stats', None, [ptr, ptr]),
    ('continuous', C.c_int, [ptr]), ('event', C.c_char_p, [ptr]),
    ('event_rows', C.c_int, [ptr, ptr, ptr, ptr]), ('wait_event', C.c_int, [ptr, C.c_double]),
    ('pause', C.c_int, [ptr, C.c_int]), ('paused', C.c_int, [ptr]),
    ('installed', C.c_uint64, [ptr]),
    ('retarget', C.c_int, [ptr, C.c_int, C.c_int, C.c_uint64, ptr, C.c_int,
                         C.c_uint64, C.c_double, C.c_int, C.c_int, C.c_double]),
    ('release', C.c_int, [ptr, C.c_int, C.c_int, C.c_uint64]),
    ('replace', C.c_int, [ptr, C.c_int, C.c_int, C.c_uint64, ptr, C.c_char_p, ptr, C.c_int,
                         C.c_uint64, C.c_double, C.c_int, C.c_int, C.c_double, C.c_uint64]),
):
    bind('hxb_'+name, result, *args)
bind('hxp_new', ptr, ptr, ptr, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int)
bind('hxp_neural', C.c_int, ptr, C.c_uint64, C.c_int)
bind('hxp_neural_stats', None, ptr, ptr)
bind('hxp_neural_record', C.c_char_p, ptr, C.c_int)
for name, result, args in (
    ('step', C.c_int, [ptr]), ('cancel', None, [ptr]), ('resume', None, [ptr]),
    ('drain', C.c_int, [ptr]), ('free', C.c_int, [ptr]),
    ('offer', C.c_int, [ptr, C.c_int, ptr, C.c_int, C.c_double]),
    ('stats', None, [ptr, ptr, ptr]), ('scope_stats', None, [ptr, ptr]), ('supply_stats', None, [ptr, ptr, ptr]), ('record', C.c_char_p, [ptr, C.c_int]),
    ('generation', C.c_uint64, [ptr, C.c_int]), ('effort', C.c_int, [ptr, C.c_int, ptr]),
):
    bind('hxp_'+name, result, *args)


class ProofLoop:
    """Native frontier and immutable jobs; Python never dispatches individual queries.

    A slice is a CPU scheduling quantum. UNKNOWN remains unknown. Resident tables
    and bounded best-first frontiers survive compatible slices and retargets.
    Recursive level-1 frames and kernel memos are still rebuilt per query.
    `queue` bounds queued plus running jobs; by default eight per worker, so
    workers keep work between the graph owner's refills.
    """
    def __init__(self, pool, package=None, *, workers=2, queue=None, slice_ms=8, table_mb=4,
                 tasks=256, stamps=False, endpoints=8, direct=False):
        queue = 8*workers if queue is None else queue
        if not isinstance(endpoints,int) or not 0<=endpoints<=8:
            raise ValueError('Neural frontier limit must be an integer from 0 to 8')
        from tactical_proof import NativeTactics, PACKAGE
        self.pool = pool
        self.library = NativeTactics(PACKAGE if package is None else package)
        names = ('worker_new', 'worker_free', 'worker_answer', 'answer_info', 'answer_moves',
                 'answer_json', 'answer_free', 'free', 'prepare', 'cancel', 'release', 'worker_busy')
        if direct:
            names = tuple(name+'_direct' if name in ('worker_new','worker_free','worker_answer','worker_busy')
                          else name for name in names)
        functions = np.asarray([C.cast(getattr(self.library.lib, 'hexo_tactical_'+name), ptr).value
                                for name in names], np.uint64)
        self._ptr = None
        try:
            callback=C.cast(self.library.lib.hexo_tactical_answer_frontier, ptr).value if endpoints else 0
            self._ptr = native.hxp_new(pool.ptr, functions.ctypes.data, workers, queue, slice_ms,
                                       table_mb, tasks, bool(stamps))
            if not self._ptr:
                checked(False)
            checked(native.hxp_neural(self._ptr, callback, endpoints))
        except BaseException:
            if self._ptr:
                checked(native.hxp_free(self._ptr));self._ptr=None
            self.library.close()
            raise

    @property
    def ptr(self):
        if not self._ptr:
            raise ValueError('Proof loop is closed')
        self.pool.ptr  # The native producer owns graph/proof delivery while attached.
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
        scope = np.empty(10, np.uint64)
        native.hxp_scope_stats(self.ptr, scope.ctypes.data)
        result['scope'] = dict(zip(('refreshes', 'changed', 'unchanged', 'refresh_ns', 'available_facts',
                                   'sent_facts', 'empty_jobs', 'quantum_ms', 'indexed_cells', 'closed_scopes'), map(int, scope)))
        frontier=np.empty(6, np.uint64)
        native.hxp_neural_stats(self.ptr, frontier.ctypes.data)
        result['neural_frontier']=dict(zip(('paths','candidates','rejected','bytes','install_ns','records'),map(int,frontier)))
        counts, idle = np.empty(12, np.uint64), np.empty(6, np.float64)
        native.hxp_supply_stats(self.ptr, counts.ctypes.data, idle.ctypes.data)
        result.update(zip(('supply_scans', 'supply_seen', 'supply_eligible', 'supply_deferred', 'supply_pending',
                           'supply_closed', 'supply_dormant', 'supply_full_exits', 'supply_held_exits',
                           'supply_empty_exits', 'supply_first_queries', 'supply_deferred_dispatched'), map(int, counts)))
        result.update(zip(('idle_capacity_ms', 'idle_held_ms', 'idle_pending_ms', 'idle_closed_ms',
                           'idle_dormant_ms', 'idle_empty_ms'), map(float, idle)))
        return result

    def records(self):
        """Inspect verified evidence and its actual conditional request context."""
        import json
        return [json.loads(native.hxp_record(self.ptr, i)) for i in range(self.stats()['records'])]

    def frontier_records(self):
        """CPU-to-neural paths are exploration records, never proof targets."""
        import json
        return [json.loads(native.hxp_neural_record(self.ptr, i)) for i in range(self.stats()['neural_frontier']['records'])]

    def effort(self, game):
        """Actual fresh work charged to the dispatched game/search generation.

        Read after closing the inference service. Late cancelled jobs still belong
        to their old generation; cache-hit historical counts never enter this ledger.
        """
        size = native.hxp_effort(self.ptr, game, None)
        out = np.empty((size, 4), np.uint64)
        native.hxp_effort(self.ptr, game, out.ctypes.data)
        return {int(r[0]): dict(fresh_nodes=int(r[1]), queries=int(r[2]), missing_fresh=int(r[3])) for r in out}

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
        self.ptr
        if self.proofs is not None:
            raise ValueError('Search pool already has a proof loop')
        self.proofs = ProofLoop(self, package, **options)
        return self.proofs

    @property
    def ptr(self):
        if not self._ptr:
            raise ValueError('Search pool is closed')
        if getattr(self, '_service', None) is not None:
            raise ValueError('Search pool is owned by its native inference service')
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
            self.ptr
            if self.proofs is not None:
                self.proofs.close()
            checked(native.hxgm_free(self._ptr))
            self._ptr = None


class InferenceService:
    """Native independent producers feeding one model-keyed GPU batch service.

    Pools and their source graphs are exclusively owned until close(). Python
    only launches/collects immutable packed batches. Producer threads install
    prediction and proof messages into their own graphs. Fixed-work mode stays
    available with ms=0; interactive comparisons use a common clock.
    """
    def __init__(self, pools, evaluators, *, batch_size=128, quantum=64, pending=2,
                 merge_cells=32768, latency_ms=.2, flights=2, profile=False, interleave_feedback=False):
        if not pools or batch_size<1 or batch_size>1024:
            raise ValueError('Open pools and a valid inference batch size are required')
        self.pools, self.models = [], list(evaluators)
        self.model_versions = versions = [e.model_version for e in self.models]
        self.calls = self.full_calls = 0
        if len(versions)!=len(set(versions)) or any(p.model_version not in versions for p in pools):
            raise ValueError('One frozen evaluator is required for every pool model version')
        self.batch_size, self.pending, self.leases, self._stats = batch_size, [], {}, None
        self.flight_limit = flights
        self._launcher = self._failure = None
        self._relaunch = False
        self.interleave_feedback = bool(interleave_feedback)
        self._ptr = native.hxb_new(quantum, pending, merge_cells, latency_ms)
        if not self._ptr:
            checked(False)
        try:
            checked(native.hxb_flights(self._ptr, flights))
            checked(native.hxb_profile(self._ptr, bool(profile)))
            checked(native.hxb_feedback(self._ptr, self.interleave_feedback))
            for pool in pools:
                self.attach(pool,self.models[versions.index(pool.model_version)])
        except BaseException:
            self.close()
            raise

    @property
    def ptr(self):
        if not self._ptr:
            raise ValueError('Inference service is closed')
        return self._ptr

    def attach(self, pool, evaluator):
        """Add a frozen model producer without stopping other continuous games."""
        if pool.model_version!=evaluator.model_version:
            raise ValueError('Evaluator does not match the producer weights')
        added = evaluator.model_version not in self.model_versions
        if added:
            self.model_versions.append(evaluator.model_version)
            self.models.append(evaluator)
        model = self.model_versions.index(evaluator.model_version)
        status = native.hxb_attach(self.ptr,pool.ptr,model)
        if not status:
            if added:
                self.model_versions.pop();self.models.pop()
            checked(False)
        producer = status-1
        while len(self.pools)<=producer:
            self.pools.append(None)
        self.pools[producer] = pool
        pool._service = self
        if self.models[model] is None:
            self.models[model] = evaluator
        return producer,model

    def detach(self, producer):
        """Join a fully retired producer; other models/games remain active."""
        pool = self.pools[producer]
        checked(native.hxb_detach(self.ptr,producer))
        pool._service = None
        self.pools[producer] = None

    def model_pending(self, model):
        """Keep weights alive while any producer or device task still uses them."""
        return bool(native.hxb_model_pending(self.ptr,model))

    def workers(self, producer, count):
        """Resize host workers on their owner thread between joined phases."""
        checked(native.hxb_workers(self.ptr,producer,count))

    def reclaim_ready(self):
        return bool(native.hxb_reclaim_ready(self.ptr))

    def reclaim(self, pool):
        """Transfer a joined, drained pool to bounded native background cleanup."""
        if pool.proofs is not None:
            pool.proofs.close()
        checked(native.hxb_reclaim(self.ptr,pool.ptr))
        pool._ptr = None

    def take(self, wait_ms=0):
        from native_dense import PackedRows
        token, model, pointer = C.c_uint64(), C.c_int(), ptr()
        count = native.hxb_take(self.ptr, self.batch_size, wait_ms, C.byref(token),
                                C.byref(model), C.byref(pointer))
        if count<0:
            checked(False)
        if not count:
            return None
        try:
            rows = PackedRows.from_native(pointer.value, count)
        except BaseException:
            native.hxgp_free(pointer.value)
            checked(native.hxb_abort(self.ptr, token.value))
            raise
        self.leases[token.value] = rows
        self.calls += 1
        self.full_calls += count==self.batch_size
        return token.value, model.value, rows

    def start(self, ms=0, *, continuous=False):
        if continuous:
            if ms:
                raise ValueError('Continuous roots carry their own clock or work limit')
            checked(native.hxb_continuous(self.ptr))
        checked(native.hxb_start(self.ptr, ms))

    def retarget(self, producer, game, history, *, expected=0, work=0, ms=0, samples=16, views=8, noise=0.):
        """Copy a next-root command to its native graph owner, never mutate it here."""
        cells = np.ascontiguousarray(history, np.int64).reshape(-1, 2)
        checked(native.hxb_retarget(self.ptr, producer, game, expected, cells.ctypes.data,
                                    len(cells), work, ms, samples, views, noise))

    def release(self, producer, game, *, expected):
        """Retire one game; its immutable completion includes all late proof effort."""
        checked(native.hxb_release(self.ptr, producer, game, expected))

    def replace(self, producer, game, source, *, expected, work=0, ms=0, samples=16, views=8, noise=0., seed=0):
        """Install a fresh game after release. Zero work/ms leaves it parked.

        Success transfers the source Tree and closes its caller wrapper. The
        command retains its store, not the caller's Tree address.
        Old caller trees must close after release acknowledgement and before
        replacement admission so their destructors cannot race the old owner.
        Predictions are reusable only within the slot's frozen model identity.
        """
        cells = np.ascontiguousarray(source.history, np.int64).reshape(-1, 2)
        checked(native.hxb_replace(self.ptr, producer, game, expected, source.ptr,
                                    source.model_version.encode(), cells.ctypes.data, len(cells),
                                    work, ms, samples, views, noise, seed))
        source.ptr = None

    def launch(self):
        """Run pump() on a launcher thread until pause() or close().

        The caller then never pumps itself: forwards keep launching and
        collecting while it handles events. A launcher failure cancels the
        service and is raised by the next event(), wait() or pause().
        """
        import threading
        if self._launcher is not None:
            raise ValueError('The inference launcher is already running')
        self.ptr
        halt = threading.Event()
        def run():
            try:
                while not halt.is_set():
                    self.pump()
                    if not self.pending and self.done():
                        halt.wait(.01)
            except BaseException as error:
                self._failure = error
                self.cancel()
        thread = threading.Thread(target=run, name='inference-launcher', daemon=True)
        self._launcher = thread, halt
        thread.start()

    def _halt(self):
        """Join the launcher; True when one was running."""
        if self._launcher is None:
            return False
        thread, halt = self._launcher
        halt.set()
        thread.join()
        self._launcher = None
        self._raise()
        return True

    def _raise(self):
        if self._failure is not None:
            raise self._failure

    def wait(self, ms):
        """Block up to `ms` for a root event; True when one is ready."""
        self._raise()
        return bool(native.hxb_wait_event(self.ptr, ms))

    def event(self):
        """Copy one immutable root/lifecycle completion; None while roots run.

        Numeric legal-action records bypass JSON and own their array storage,
        so later events, replacement and service close cannot invalidate them.
        """
        import json
        self._raise()
        text, edges, count = C.c_char_p(), C.POINTER(C.c_double)(), C.c_int()
        status = native.hxb_event_rows(self.ptr,C.byref(text),C.byref(edges),C.byref(count))
        if status<0:
            checked(False)
        if not status:
            self.done()  # Surface producer failures instead of silently waiting forever.
            return None
        result = json.loads(text.value)
        if count.value>=0:
            result['edges'] = np.ctypeslib.as_array(edges,shape=(count.value,9)).copy()
        return result

    def pause(self, *, timeout=None):
        """Fence neural forwards and retain queued work; native CPU proofs continue.

        Requires continuous mode. Active forwards are collected and installed,
        not abandoned. A failed fence retains its handle for close()/retry.
        Timed roots still use wall time; fixed-work roots retain their work.
        By default wait for acknowledgement. A caller deadline may be supplied
        in seconds; timing out retains the paused service and active handles.
        """
        import time
        self._relaunch = self._halt() or self._relaunch
        if set(self.leases)-{token for token, _ in self.pending}:
            raise ValueError('Complete or abandon_fenced manual batches before pausing')
        checked(native.hxb_pause(self.ptr, 1))
        while self.pending:
            token, handle = self.pending[0]
            self.complete(token, handle.collect())
            self.pending.pop(0)
        end = None if timeout is None else time.monotonic()+timeout
        while not self.paused():
            self.done()  # Surface an owner failure instead of waiting forever.
            if end is not None and time.monotonic()>=end:
                raise TimeoutError('Native producers have not acknowledged the neural pause')
            time.sleep(.001)

    def paused(self):
        """True only after GPU flights and native installation reach the pause."""
        status = native.hxb_paused(self.ptr)
        if status<0:
            checked(False)
        return bool(status)

    def resume(self):
        checked(native.hxb_pause(self.ptr, 0))
        if self._relaunch:
            self._relaunch = False
            self.launch()

    def complete(self, token, rows):
        if self.leases.get(token) is not rows:
            raise ValueError('Unknown inference service snapshot')
        checked(native.hxb_complete(self.ptr, token, *rows.outputs()))
        del self.leases[token]
        rows.close()

    def abandon_fenced(self, token):
        """Abandon a manual batch after its GPU readers have been fenced."""
        rows = self.leases[token]
        checked(native.hxb_abort(self.ptr, token))
        del self.leases[token]
        rows.close()

    def done(self):
        status = native.hxb_done(self.ptr)
        if status<0:
            checked(False)
        return bool(status)

    def stats(self):
        if not self._ptr and self._stats is not None:
            return dict(self._stats)
        out = np.empty(10, np.uint64)
        native.hxb_stats(self.ptr, out.ctypes.data)
        result = dict(zip(('unique_rows', 'coalesced_rows', 'launched_rows', 'subscriber_deliveries',
                         'withdrawn_ready_rows', 'batches', 'row_high_water', 'pending_rows',
                         'inflight_batches', 'active_producers'), map(int, out)))
        # Leased model batches include queued forwards, not simultaneous GPU kernels.
        # Producer snapshots have a separate limit and may span several batches.
        flights = np.empty(3, np.uint64)
        native.hxb_flight_stats(self.ptr, flights.ctypes.data)
        result.update(zip(('batch_flight_limit', 'batch_flight_high_water',
                           'snapshot_limit_per_producer'), map(int, flights)))
        # Opt-in observations and owner wall spans. Repeated queue samples are
        # not unique work; sums over concurrent producers are not CPU occupancy.
        timing = np.empty(39, np.uint64)
        native.hxb_schedule_stats(self.ptr, timing.ctypes.data)
        # Admission/gate counters are repeated observations, not unique rows or wait time.
        # Urgent-at-entry marks certificates, checked solver endpoints or errors
        # already awaiting collection; spans are whole installs, not exact delay overlap.
        # Burst spans include interleaved collection time; these totals overlap.
        names = ('profile_enabled', 'admission_samples', 'flight_capacity_samples', 'ready_samples',
                 'ready_row_samples', 'head_age_ns', 'head_age_max_ns', 'partial_head_samples',
                 'full_alternative_samples', 'larger_alternative_samples', 'lease_reservations',
                 'partial_leases', 'idle_partial_leases', 'aged_partial_leases', 'full_leases',
                 'partial_wait_samples', 'completion_packets', 'completion_rows',
                 'completion_queue_age_ns', 'completion_queue_age_max_ns', 'packet_install_ns',
                 'packet_install_max_ns', 'urgent_at_entry_packets', 'urgent_at_entry_packet_ns',
                 'completion_bursts', 'burst_packets_max', 'burst_install_ns', 'burst_install_max_ns',
                 'urgent_at_entry_bursts', 'urgent_at_entry_burst_ns', 'snapshot_gate_samples',
                 'snapshot_gate_ready_samples', 'snapshot_gate_ready_row_samples', 'snapshot_gate_ready_rows_max', 'urgent_collections', 'urgent_collection_ns', 'urgent_collection_max_ns',
                 'urgent_certificate_collections', 'urgent_certificates')
        result.update(('schedule_'+key, int(value)) for key, value in zip(names, timing))
        result['interleave_feedback'] = self.interleave_feedback
        # Native feed messages installed, including retired/empty messages.
        # This is not a neural-row or search-visit count.
        result['installed_message_rows'] = int(native.hxb_installed(self.ptr))
        retired = np.empty(4,np.uint64)
        native.hxb_reclaim_stats(self.ptr,retired.ctypes.data)
        result.update(zip(('reclaim_queued','reclaim_active','reclaimed_pools','reclaim_ns'),map(int,retired)))
        games = np.empty(6,np.uint64)
        native.hxb_owner_reclaim_stats(self.ptr,games.ctypes.data)
        result.update(zip(('owner_reclaim_queued','owner_reclaim_active','reclaimed_games',
                           'owner_reclaim_ns','reclaim_reserved','replacement_deferrals'),map(int,games)))
        return result

    def cancel(self):
        if self._ptr:
            native.hxb_cancel(self._ptr)

    def pump(self):
        """Launch/collect bulk forwards once, leaving per-game control to the caller."""
        import native_dense
        import time
        def launch(wait):
            batch = self.take(wait)
            if batch is None:
                return False
            token, model, rows = batch
            try:
                handle = native_dense.submit(self.models[model], rows)
            except BaseException:
                self.cancel()
                uncertain = [h for h in native_dense._quarantined if h.rows is rows]
                if uncertain:
                    self.pending.append((token, uncertain[0]))
                else:
                    self.abandon_fenced(token)
                raise
            self.pending.append((token, handle))
            return True
        while len(self.pending)<self.flight_limit:
            if not launch(0 if self.pending else 2.):
                break
        if self.pending:
            # A partial batch may become ready while the oldest forward runs.
            # Keep feeding its spare slot instead of blocking in collect().
            event = self.pending[0][1].event
            while len(self.pending)<self.flight_limit and event is not None and not event.query():
                if not launch(0):
                    # A timed take flushes a partial batch at its timeout.
                    # Preserve the configured batching latency while waiting.
                    time.sleep(.0001)
            token, handle = self.pending[0]
            self.complete(token, handle.collect())
            self.pending.pop(0)
            # Actor event processing can take longer than a forward. Leave
            # ready work running before handing control back to that caller.
            while len(self.pending)<self.flight_limit and launch(0):
                pass

    def run(self, ms=0):
        try:
            self.start(ms)
            while not self.done():
                self.pump()
        finally:
            self.close()

    def close(self, *, completions=False):
        """Fence and join; optionally retain final writer-owned root events."""
        final = []
        if self._ptr:
            self.cancel()
            if self._launcher is not None:
                thread, halt = self._launcher
                halt.set()
                thread.join()
                self._launcher = None
            failure = None
            for token, handle in tuple(self.pending):
                try:
                    handle.close()
                    if token in self.leases:
                        self.abandon_fenced(token)
                    self.pending.remove((token, handle))
                except BaseException as error:
                    if failure is None:
                        failure = error
            if failure is not None:
                raise failure
            if self.leases:
                raise ValueError('Complete or abandon_fenced every manually taken service batch before close')
            checked(native.hxb_join(self._ptr))
            self._stats = self.stats()
            try:
                if completions:
                    while (event := self.event()) is not None:
                        final.append(event)
            finally:
                checked(native.hxb_free(self._ptr))
                self._ptr = None
                for pool in self.pools:
                    if pool is not None:
                        pool._service = None
        return final if completions else None
