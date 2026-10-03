"""Batch bridge to the native neural frontier; one instance belongs to one frozen model.

Selection, exact context identity, prediction caching, queued/in-flight coalescing
and result installation run in C++. The owner thread calls this bridge between
GPU submissions. Trees are detached before they are closed or cancelled.
"""
import ctypes as C
import numpy as np
from neural_search import native, bind, ptr, checked


bind('hxgf_new', ptr, C.c_int)
bind('hxgf_free', None, ptr)
bind('hxgf_begin', C.c_int, ptr, ptr, ptr, C.c_int)
bind('hxgf_seed', C.c_int, ptr, ptr, ptr, ptr, ptr, C.c_int)
bind('hxgf_gather', C.c_int, ptr, ptr, ptr)
bind('hxgf_layout', C.c_int, ptr, C.c_int, ptr)
bind('hxgf_take', C.c_int, ptr, C.c_int, ptr, ptr, ptr, ptr, ptr, C.c_int64)
bind('hxgf_install', C.c_int, ptr, ptr, C.c_int, ptr, ptr, ptr, ptr, ptr, C.c_int)
bind('hxgf_detach', None, ptr, ptr)
bind('hxgf_root_value', C.c_int, ptr, ptr, ptr)
bind('hxgf_stats', None, ptr, ptr)
bind('hxgf_profile', None, ptr, C.c_int)
bind('hxgf_times', None, ptr, ptr)


class NativeFeed:
    def __init__(self, capacity):
        self.ptr = native.hxgf_new(capacity)
        if not self.ptr:
            checked(False)
        self.trees = {}
        self.root_versions = {}

    def begin(self, tree, prediction=None):
        history = np.ascontiguousarray(tree.history, np.int64).reshape(-1, 2)
        checked(native.hxgf_begin(self.ptr, tree.ptr, history.ctypes.data, len(history)))
        self.trees[tree.ptr] = tree
        self.root_versions[tree.ptr] = native.hxg_root_version(tree.ptr)
        if prediction is not None:
            actions = np.ascontiguousarray(prediction[0], np.int64)
            logits = np.ascontiguousarray(prediction[1], np.float64)
            values = np.ascontiguousarray(prediction[2], np.float64)
            if actions.size != 2*len(logits) or len(values) != len(logits):
                raise ValueError('Incomplete native feed root prediction')
            checked(native.hxgf_seed(self.ptr, tree.ptr, actions.ctypes.data, logits.ctypes.data,
                                    values.ctypes.data, len(logits)))

    def sync_root(self, tree):
        if tree.ptr in self.trees and self.root_versions[tree.ptr] != native.hxg_root_version(tree.ptr):
            self.begin(tree)

    def gather(self, tree):
        self.sync_root(tree)
        stats = np.empty(4, np.int64)
        status = native.hxgf_gather(self.ptr, tree.ptr, stats.ctypes.data)
        if status == -2:
            checked(False)
        return status, stats

    def take(self, limit):
        layout = np.empty(2, np.int64)
        checked(native.hxgf_layout(self.ptr, limit, layout.ctypes.data))
        count, size = map(int, layout)
        if not count:
            return None
        ids = np.empty(count, np.uint64)
        trees, requests = np.empty(count, np.uintp), np.empty(count, np.int32)
        offsets, history = np.empty(count+1, np.int64), np.empty((size, 2), np.int64)
        checked(native.hxgf_take(self.ptr, count, ids.ctypes.data, trees.ctypes.data, requests.ctypes.data,
                                offsets.ctypes.data, history.ctypes.data, size))
        leaves = [(int(tree), int(request), history[offsets[i]:offsets[i+1]])
                  for i, (tree, request) in enumerate(zip(trees, requests))]
        return ids, leaves

    def install(self, ids, predictions):
        if len(predictions) != len(ids):
            raise ValueError('Incomplete native feed batch')
        counts = [0 if p is None else len(p[0]) for p in predictions]
        offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
        valid = [p for p in predictions if p is not None]
        actions = np.ascontiguousarray(np.concatenate([p[0] for p in valid]) if valid else [], np.int64)
        logits = np.ascontiguousarray(np.concatenate([p[1] for p in valid]) if valid else [], np.float64)
        values = np.ascontiguousarray(np.concatenate([p[2] for p in valid]) if valid else [], np.float64)
        if len(logits) != offsets[-1] or len(values) != offsets[-1] or actions.size != 2*offsets[-1]:
            raise ValueError('Incomplete native feed prediction')
        return self._install(ids, offsets.ctypes.data, actions.ctypes.data, logits.ctypes.data, values.ctypes.data)

    def take_packed(self, limit):
        from native_dense import PackedRows
        layout = np.empty(2, np.int64)
        checked(native.hxgf_layout(self.ptr, limit, layout.ctypes.data))
        count = int(layout[0])
        if not count:
            return None
        ids = np.empty(count, np.uint64)
        trees, requests = np.empty(count, np.uintp), np.empty(count, np.int32)
        checked(native.hxgf_take(self.ptr, count, ids.ctypes.data, trees.ctypes.data, requests.ctypes.data,
                                None, None, 0))
        return ids, PackedRows(trees, requests)

    def install_packed(self, ids, rows):
        if rows.count != len(ids):
            raise ValueError('Incomplete native feed batch')
        try:
            return self._install(ids, *rows.outputs())
        finally:
            rows.close()

    def _install(self, ids, offsets, actions, logits, values):
        stopped = np.empty(len(self.trees), np.uintp)
        count = native.hxgf_install(self.ptr, ids.ctypes.data, len(ids), offsets, actions,
                                   logits, values, stopped.ctypes.data, len(stopped))
        if count < 0:
            checked(False)
        for tree in stopped[:count]:
            self.trees.pop(int(tree), None)
            self.root_versions.pop(int(tree), None)
        return stopped[:count]

    def root_value(self, tree):
        """The current root's raw prediction, or None when no prediction is retained for that context."""
        self.sync_root(tree)
        value = C.c_double()
        return value.value if native.hxgf_root_value(self.ptr, tree.ptr, C.byref(value)) else None

    def detach(self, tree):
        if tree.ptr in self.trees:
            native.hxgf_detach(self.ptr, tree.ptr)
            self.trees.pop(tree.ptr)
            self.root_versions.pop(tree.ptr)

    def stats(self):
        result = np.empty(6, np.int64)
        native.hxgf_stats(self.ptr, result.ctypes.data)
        return dict(zip(('new_rows', 'joined', 'cache_hits', 'installed', 'pending_rows', 'pending_requests'),
                        map(int, result)))

    def profile(self, enabled=True):
        native.hxgf_profile(self.ptr, enabled)

    def timings(self):
        result = np.empty(4, np.uint64)
        native.hxgf_times(self.ptr, result.ctypes.data)
        return dict(zip(('selection_ms', 'identity_ms', 'cached_install_ms', 'batch_install_ms'),
                        result.astype(np.float64)/1e6))

    def close(self):
        if self.ptr:
            for pointer, tree in self.trees.items():
                if tree.ptr == pointer:
                    native.hxgf_detach(self.ptr, pointer)
            native.hxgf_free(self.ptr)
            self.ptr = None
            self.trees.clear()
            self.root_versions.clear()
