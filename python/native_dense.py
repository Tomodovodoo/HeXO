"""Native-owned crop batches. Python launches forwards; C++ packs and decodes rows."""
import ctypes as C
import time
import weakref
import numpy as np
import torch
import hexnet
from neural_search import native, bind, ptr, checked


bind('hxgp_new', ptr, ptr, ptr, C.c_int, C.c_int)
bind('hxgp_free', None, ptr)
bind('hxgp_groups', C.c_int, ptr)
bind('hxgp_mixed', C.c_int, ptr)
bind('hxgp_group', C.c_int, ptr, C.c_int, ptr)
bind('hxgp_pack', C.c_int, ptr, C.c_int, ptr, C.c_int64)
bind('hxgp_decode', C.c_int, ptr, C.c_int, C.c_int, C.c_int, ptr, C.c_int64)
bind('hxgp_outputs', C.c_int, ptr, ptr)
bind('hxgp_costs_new', ptr)
bind('hxgp_costs_free', None, ptr)
bind('hxgp_costs_learn', C.c_int, ptr, ptr, C.c_int)
bind('hxgp_costs_stats', None, ptr, ptr)
bind('hxgp_plan', C.c_int, ptr, ptr, ptr, C.c_int, C.c_int)
_quarantined = []  # Keep host storage alive when GPU completion cannot be established.


class PackingCosts:
    """One bounded native cost model for a frozen evaluator's graph lifetime."""
    def __init__(self):
        self.ptr = native.hxgp_costs_new()
        if not self.ptr:
            checked(False)
        self._free = weakref.finalize(self, native.hxgp_costs_free, self.ptr)
        self.fitted, self.calls = False, 0

    def sample(self):
        self.calls += 1
        return not self.fitted or self.calls % 8 == 0

    def learn(self, observations):
        rows = np.ascontiguousarray(observations, np.float64).reshape(-1, 3)
        status = native.hxgp_costs_learn(self.ptr, rows.ctypes.data, len(rows))
        checked(status)
        self.fitted = status == 2

    def stats(self):
        out = np.empty(4, np.float64)
        native.hxgp_costs_stats(self.ptr, out.ctypes.data)
        return dict(samples=int(out[0]), cell_ms=out[1], launch_ms=out[2], fitted=bool(out[3]))


class PackedRows:
    def __init__(self, trees, requests, merge_cells=32768):
        trees, requests = np.ascontiguousarray(trees, np.uintp), np.ascontiguousarray(requests, np.int32)
        if trees.ndim != 1 or requests.ndim != 1 or len(trees) != len(requests):
            raise ValueError('Invalid packed batch handles')
        self.ptr = native.hxgp_new(trees.ctypes.data, requests.ctypes.data, len(requests), merge_cells)
        if not self.ptr:
            checked(False)
        self.count = len(requests)
        self._groups()

    @classmethod
    def from_native(cls, pointer, count):
        """Adopt an immutable service snapshot; close() owns its native deletion."""
        batch = cls.__new__(cls)
        batch.ptr, batch.count = pointer, count
        batch._groups()
        return batch

    def _groups(self):
        self.mixed = bool(native.hxgp_mixed(self.ptr))
        self.groups = []
        for index in range(native.hxgp_groups(self.ptr)):
            info = np.empty(2, np.int64)
            checked(native.hxgp_group(self.ptr, index, info.ctypes.data))
            self.groups.append(tuple(map(int, info)))

    def plan(self, costs, limits, step):
        if not self.ptr:
            raise ValueError('Packed batch closed')
        caps = np.ascontiguousarray(limits, np.int64).reshape(-1, 2)
        checked(native.hxgp_plan(self.ptr, costs.ptr, caps.ctypes.data, len(caps), step))
        self._groups()

    def pack(self, index, output):
        side, count = self._group(index)
        if output.dtype != np.uint8 or not output.flags.c_contiguous or not output.flags.writeable or output.shape != (count, 8, side, side):
            raise ValueError('Packed planes require a writable contiguous uint8 group')
        checked(native.hxgp_pack(self.ptr, index, output.ctypes.data, output.size))

    def decode(self, index, start, output):
        side, _ = self._group(index)
        if output.dtype != np.float32 or not output.flags.c_contiguous or output.ndim != 2 or output.shape[1] != side*side+2:
            raise ValueError('Packed predictions require contiguous float32 rows')
        checked(native.hxgp_decode(self.ptr, index, start, len(output), output.ctypes.data, output.size))

    def _group(self, index):
        if not self.ptr or index < 0 or index >= len(self.groups):
            raise ValueError('Invalid or closed packed group')
        return self.groups[index]

    def outputs(self):
        if not self.ptr:
            raise ValueError('Packed batch closed')
        result = (ptr*4)()
        checked(native.hxgp_outputs(self.ptr, result))
        return result

    def close(self):
        if self.ptr:
            native.hxgp_free(self.ptr)
            self.ptr = None


class Forward:
    """A submitted batch owns its staging until the GPU's completion event finishes."""
    def __init__(self, evaluator, rows):
        self.evaluator, self.rows = evaluator, rows
        self.staging = evaluator.free.pop() if evaluator.free else {}
        self.chunks, self.event = [], None
        self.costs, self.observations = None, {}

    def release(self):
        if self.evaluator.cuda and self.event is None:
            if self not in _quarantined:
                _quarantined.append(self)
            raise RuntimeError('GPU completion fence not established')
        if self.event is not None:
            try:
                self.event.synchronize()
            except BaseException:
                if self not in _quarantined:
                    _quarantined.append(self)
                raise
        self.chunks.clear()
        self.observations.clear()
        if self.staging is not None:
            self.evaluator.free.append(self.staging)
            self.staging = None
        if self in _quarantined:
            _quarantined.remove(self)

    def collect(self):
        try:
            if self.event is not None:
                self.event.synchronize()
            for index, start, output in self.chunks:
                began = time.perf_counter() if index in self.observations else 0.
                self.rows.decode(index, start, output.numpy())
                if index in self.observations:
                    self.observations[index][5] += (time.perf_counter()-began)*1000
            if self.costs is not None:
                samples = [[cells, launches, begin.elapsed_time(end)+pack_ms+decode_ms]
                           for cells, launches, begin, end, pack_ms, decode_ms in self.observations.values()]
                if samples:
                    self.costs.learn(samples)
            return self.rows
        finally:
            self.release()

    def close(self):
        try:
            self.release()
        finally:
            self.rows.close()


@torch.inference_mode()
def submit(evaluator, rows, max_cells=48*48*48):
    handle = Forward(evaluator, rows)
    try:
        graph = evaluator.graph
        if evaluator.cuda and graph is not None and getattr(evaluator, 'packing_adaptive', True):
            costs = getattr(graph, 'packing_costs', None)
            if costs is None:
                costs = graph.packing_costs = PackingCosts()
            if rows.mixed:
                limits = [] if graph.budget_exhausted else [(side, graph._limit(side, graph.max_batch)) for side in graph.CANVASES]
                rows.plan(costs, limits, evaluator.max_batch)
            handle.costs = costs
        observe = handle.costs is not None and handle.costs.sample()
        for index, (side, count) in enumerate(rows.groups):
            host = hexnet.staging_buffer(handle.staging, ('planes', side), count, (8, side, side),
                                         torch.uint8, evaluator.cuda)
            result = hexnet.staging_buffer(handle.staging, ('out', side), count, (side*side+2,),
                                           torch.float32, evaluator.cuda)
            began = time.perf_counter() if observe else 0.
            rows.pack(index, host.numpy())
            pack_ms = (time.perf_counter()-began)*1000 if observe else 0.
            graphed = evaluator.graph is not None and side in evaluator.graph.CANVASES
            step = evaluator.max_batch if graphed else max(1, min(evaluator.max_batch, max_cells//(side*side)))
            if observe and graphed:
                pieces = [cap for start in range(0, count, step)
                          for _, cap in graph._segments(min(step, count-start), graph._limit(side, graph.max_batch))]
                if all((side, cap) in graph.graphs for cap in pieces):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    handle.observations[index] = [sum(pieces)*side*side, len(pieces), begin, end, pack_ms, 0.]
            for start in range(0, count, step):
                size = min(step, count-start)
                x = host[start:start+size].to(evaluator.device, non_blocking=True)
                x = x.to(memory_format=evaluator.memory_format, dtype=torch.bfloat16 if evaluator.cuda else torch.float32)
                out = evaluator.predict(x)
                packed = torch.cat((out['policy'], out['far'][:, None], out['value_logit'][:, None]), 1)
                target = result[start:start+size].copy_(packed, non_blocking=True)
                handle.chunks.append((index, start, target))
            if index in handle.observations:
                handle.observations[index][3].record()
        if evaluator.cuda:
            event = torch.cuda.Event(blocking=True)
            event.record()
            handle.event = event
        return handle
    except BaseException:
        # A failure may follow an asynchronous copy. Do not recycle its host buffer early.
        if handle not in _quarantined:
            _quarantined.append(handle)
        try:
            if evaluator.cuda:
                event = torch.cuda.Event(blocking=True)
                event.record()
                handle.event = event
            handle.close()
        finally:
            handle.rows.close()
        raise
