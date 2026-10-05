"""Bounded same-canvas CUDA graphs for a frozen HexNet evaluator.

One instance belongs to one model. Instances reuse a device stream and lock so
BLAS workspaces do not accumulate with captures or checkpoint changes. Captures
share a model's memory pool; outputs are copied before another replay.
"""

import threading
import warnings

import torch


class ActorGraph:
    CANVASES = (24, 32, 40, 48, 64)
    BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
    LARGE_BATCHES = {24: 128, 32: 64, 40: 64}
    MAX_CELLS = 110592
    STREAMS = {}
    LOCK = threading.Lock()

    def __init__(self, model, max_incremental_bytes=384*1024*1024, max_batch=32):
        if model.training or model.net_kernels != 'fused':
            raise ValueError('ActorGraph requires a frozen fused eval model')
        device = next(model.parameters()).device
        if device.type != 'cuda':
            raise ValueError('ActorGraph requires a CUDA model')
        if max_batch not in self.BATCHES:
            raise ValueError('ActorGraph max_batch must be a capture capacity')
        self.max_batch = max_batch
        self.model = model
        self.device = device
        self.lock = self.LOCK
        with self.lock:
            if device not in self.STREAMS:
                self.STREAMS[device] = torch.cuda.Stream(device=device)
            self.stream = self.STREAMS[device]
            self.stream.wait_stream(torch.cuda.current_stream(device))
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs = {}
        self.before_reserved = torch.cuda.memory_reserved(device)
        self.max_incremental_bytes = max_incremental_bytes
        self.incremental_reserved_bytes = 0
        self.budget_exhausted = False
        self.capture_memory_error = None

    def _capture(self, side, capacity):
        if self.incremental_reserved_bytes >= self.max_incremental_bytes:
            raise MemoryError('actor graph memory budget exhausted')
        key = side, capacity
        # Outside capture: these addresses stay fixed, but can never alias the
        # shared graph pool. Dummy rows have one valid cell to avoid 0/0 pooling.
        template = torch.empty((capacity, 8, side, side), device=self.device,
                               dtype=torch.bfloat16, memory_format=torch.channels_last).zero_()
        template[:, 3, 0, 0] = 1
        static_input = template.clone(memory_format=torch.channels_last)
        static_packed = torch.empty((capacity, side*side+2), device=self.device, dtype=torch.float32)

        with torch.cuda.stream(self.stream), torch.inference_mode(), \
                torch.autocast('cuda', torch.bfloat16, cache_enabled=False):
            warm = self.model(static_input, static_input[:, 3:4], aux=False)
            static_packed.copy_(torch.cat((warm['policy'], warm['far'][:, None],
                                           warm['value_logit'][:, None]), dim=1))
        del warm

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16, cache_enabled=False), \
                torch.cuda.graph(graph, pool=self.pool, stream=self.stream):
            out = self.model(static_input, static_input[:, 3:4], aux=False)
            static_packed.copy_(torch.cat((out['policy'], out['far'][:, None],
                                           out['value_logit'][:, None]), dim=1))
        del out
        incremental = torch.cuda.memory_reserved(self.device)-self.before_reserved
        if incremental > self.max_incremental_bytes:
            del graph, static_packed, static_input, template
            raise MemoryError(f'actor graphs reserved {incremental/2**20:.1f} MiB; '
                              f'budget is {self.max_incremental_bytes/2**20:.1f} MiB')
        self.graphs[key] = graph, template, static_input, static_packed
        self.incremental_reserved_bytes = incremental
        return self.graphs[key]

    @staticmethod
    def _segments(rows, limit, side=24):
        while rows > limit:
            yield limit, limit
            rows -= limit
        # Partial large captures execute their padded rows too. Larger canvases
        # benefit from splitting a slightly wider tail before their 64-row replay.
        if limit >= 128 and 64 < rows <= 96:
            yield 64, 64
            rows -= 64
        if limit >= 64 and 32 < rows <= (56 if side == 40 else 48):
            yield 32, 32
            rows -= 32
        if 16 < rows <= 24:
            yield 16, 16
            rows -= 16
        if rows:
            yield rows, next(cap for cap in ActorGraph.BATCHES if rows <= cap <= limit)

    @classmethod
    def _limit(cls, side, max_batch=32):
        ceiling = min(max_batch, cls.LARGE_BATCHES.get(side, 32))
        return max(cap for cap in cls.BATCHES if cap <= ceiling and cap*side*side <= cls.MAX_CELLS)

    @torch.inference_mode()
    def __call__(self, planes):
        """Return caller-owned aux=False outputs for resident BF16 [B,8,S,S]."""
        packed = self.packed(planes)
        return {'policy': packed[:, :-2], 'far': packed[:, -2],
                'value_logit': packed[:, -1]}

    @torch.inference_mode()
    def packed(self, planes):
        """Caller-owned float32 policy, far-policy and value-logit rows."""
        return self._forward(planes)

    @torch.inference_mode()
    def copy_to(self, planes, target):
        """Queue packed predictions into caller-owned pinned CPU rows.

        The caller retains target until a completion fence on its CUDA stream.
        Downloads precede the next replay on the serialized graph stream.
        """
        b, _, side, _ = planes.shape
        if (target.device.type != 'cpu' or target.dtype != torch.float32
                or target.shape != (b, side*side+2) or not target.is_contiguous()
                or not target.is_pinned()):
            raise ValueError('expected pinned contiguous CPU float32 prediction rows')
        try:
            self._forward(planes, target)
        except BaseException:
            # Earlier segments may already be downloading when a later launch
            # fails. The caller's failure fence must cover those downloads too.
            caller = torch.cuda.current_stream(self.device)
            if caller.cuda_stream != self.stream.cuda_stream:
                caller.wait_stream(self.stream)
            raise
        return target

    def _forward(self, planes, target=None):
        b, channels, side, width = planes.shape
        if (channels != 8 or side != width or b < 1 or planes.device != self.device
                or planes.dtype != torch.bfloat16):
            raise ValueError('expected nonempty [B,8,S,S] BF16 planes on the model CUDA device')
        caller_stream = torch.cuda.current_stream(self.device)
        same_stream = caller_stream.cuda_stream == self.stream.cuda_stream
        with self.lock, torch.cuda.stream(self.stream), \
                torch.autocast('cuda', torch.bfloat16, cache_enabled=False):
            if not same_stream:
                self.stream.wait_stream(caller_stream)
                planes.record_stream(self.stream)
            if side not in self.CANVASES:
                packed = self._fallback(planes)
                if target is not None:
                    target.copy_(packed, non_blocking=True)
            else:
                limit = self._limit(side, self.max_batch)
                pieces = []
                start = 0
                for rows, capacity in self._segments(b, limit, side):
                    part = planes[start:start+rows]
                    record = self.graphs.get((side, capacity))
                    if record is None and not self.budget_exhausted:
                        try:
                            record = self._capture(side, capacity)
                        except (torch.OutOfMemoryError, MemoryError) as exc:
                            self.budget_exhausted = True
                            self.capture_memory_error = str(exc)
                            warnings.warn('CUDA graph memory limit reached; new shapes use eager inference',
                                          RuntimeWarning, stacklevel=2)
                    if record is None:
                        output = self._fallback(part)
                        if target is None:
                            pieces.append(output)
                        else:
                            target[start:start+rows].copy_(output, non_blocking=True)
                        start += rows
                        continue
                    graph, template, static_input, static_packed = record
                    if rows == capacity:
                        static_input.copy_(part)
                    else:
                        static_input.copy_(template)
                        static_input[:rows].copy_(part)
                    graph.replay()
                    # Copy before the next replay on this stream. Tensor callers
                    # own a clone; pinned targets own their queued download.
                    if target is None:
                        pieces.append(static_packed[:rows].clone())
                    else:
                        target[start:start+rows].copy_(static_packed[:rows], non_blocking=True)
                    start += rows
                if target is None:
                    packed = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
            if not same_stream:
                caller_stream.wait_stream(self.stream)
                if target is None:
                    packed.record_stream(caller_stream)
        return packed if target is None else target

    def _fallback(self, planes):
        side = planes.shape[-1]
        chunk = max(1, self.MAX_CELLS//(side*side))
        pieces = []
        for part in planes.split(chunk):
            out = self.model(part, part[:, 3:4], aux=False)
            pieces.append(torch.cat((out['policy'], out['far'][:, None],
                                     out['value_logit'][:, None]), dim=1))
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    def close(self):
        """Release graph and staging storage before replacing the model."""
        with self.lock:
            self.stream.synchronize()
            self.graphs.clear()
            torch.cuda.empty_cache()
