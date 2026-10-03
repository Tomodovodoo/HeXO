"""Native-owned crop batches. Python launches forwards; C++ packs and decodes rows."""
import ctypes as C
import numpy as np
import torch
import hexnet
from neural_search import native, bind, ptr, checked


bind('hxgp_new', ptr, ptr, ptr, C.c_int, C.c_int)
bind('hxgp_free', None, ptr)
bind('hxgp_groups', C.c_int, ptr)
bind('hxgp_group', C.c_int, ptr, C.c_int, ptr)
bind('hxgp_pack', C.c_int, ptr, C.c_int, ptr, C.c_int64)
bind('hxgp_decode', C.c_int, ptr, C.c_int, C.c_int, C.c_int, ptr, C.c_int64)
bind('hxgp_outputs', C.c_int, ptr, ptr)
_quarantined = []  # Keep host storage alive when GPU completion cannot be established.


class PackedRows:
    def __init__(self, trees, requests, merge_cells=32768):
        trees, requests = np.ascontiguousarray(trees, np.uintp), np.ascontiguousarray(requests, np.int32)
        if trees.ndim != 1 or requests.ndim != 1 or len(trees) != len(requests):
            raise ValueError('Invalid packed batch handles')
        self.ptr = native.hxgp_new(trees.ctypes.data, requests.ctypes.data, len(requests), merge_cells)
        if not self.ptr:
            checked(False)
        self.count = len(requests)
        self.groups = []
        for index in range(native.hxgp_groups(self.ptr)):
            info = np.empty(2, np.int64)
            checked(native.hxgp_group(self.ptr, index, info.ctypes.data))
            self.groups.append(tuple(map(int, info)))

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
                self.rows.decode(index, start, output.numpy())
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
        for index, (side, count) in enumerate(rows.groups):
            host = hexnet.staging_buffer(handle.staging, ('planes', side), count, (8, side, side),
                                         torch.uint8, evaluator.cuda)
            result = hexnet.staging_buffer(handle.staging, ('out', side), count, (side*side+2,),
                                           torch.float32, evaluator.cuda)
            rows.pack(index, host.numpy())
            graphed = evaluator.graph is not None and side in evaluator.graph.CANVASES
            step = evaluator.max_batch if graphed else max(1, min(evaluator.max_batch, max_cells//(side*side)))
            for start in range(0, count, step):
                size = min(step, count-start)
                x = host[start:start+size].to(evaluator.device, non_blocking=True)
                x = x.to(memory_format=evaluator.memory_format, dtype=torch.bfloat16 if evaluator.cuda else torch.float32)
                out = evaluator.predict(x)
                packed = torch.cat((out['policy'], out['far'][:, None], out['value_logit'][:, None]), 1)
                target = result[start:start+size].copy_(packed, non_blocking=True)
                handle.chunks.append((index, start, target))
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
