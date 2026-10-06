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

    @staticmethod
    def shape(canvas):
        return (canvas, canvas) if isinstance(canvas, int) else canvas

    @classmethod
    def supports(cls, canvas):
        height, width = cls.shape(canvas)
        return canvas in cls.CANVASES or (min(height, width) >= 24
                and max(height, width) <= 256 and height % 8 == width % 8 == 0)

    def _capture(self, side, capacity):
        if self.incremental_reserved_bytes >= self.max_incremental_bytes:
            raise MemoryError('actor graph memory budget exhausted')
        height, width = self.shape(side)
        key = side, capacity
        # Outside capture: these addresses stay fixed, but can never alias the
        # shared graph pool. Dummy rows have one valid cell to avoid 0/0 pooling.
        template = torch.empty((capacity, 8, height, width), device=self.device,
                               dtype=torch.bfloat16, memory_format=torch.channels_last).zero_()
        template[:, 3, 0, 0] = 1
        static_input = template.clone(memory_format=torch.channels_last)
        static_packed = torch.empty((capacity, height*width+2), device=self.device, dtype=torch.float32)

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
        height, width = cls.shape(side)
        ceiling = min(max_batch, cls.LARGE_BATCHES.get(side,
                      128 if height*width <= 32*32 else 64 if height*width <= 40*40 else 32))
        return max(cap for cap in cls.BATCHES if cap <= ceiling and cap*height*width <= cls.MAX_CELLS)

    @torch.inference_mode()
    def __call__(self, planes):
        """Return caller-owned aux=False outputs for resident BF16 [B,8,S,S]."""
        b, channels, height, width = planes.shape
        side = height if height == width else (height, width)
        if (channels != 8 or min(height, width) < 1 or b < 1 or planes.device != self.device
                or planes.dtype != torch.bfloat16):
            raise ValueError('expected nonempty [B,8,H,W] BF16 planes on the model CUDA device')
        caller_stream = torch.cuda.current_stream(self.device)
        same_stream = caller_stream.cuda_stream == self.stream.cuda_stream
        with self.lock, torch.cuda.stream(self.stream), \
                torch.autocast('cuda', torch.bfloat16, cache_enabled=False):
            if not same_stream:
                self.stream.wait_stream(caller_stream)
                planes.record_stream(self.stream)
            if not self.supports(side):
                packed = self._fallback(planes)
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
                        pieces.append(self._fallback(part))
                        start += rows
                        continue
                    graph, template, static_input, static_packed = record
                    if rows == capacity:
                        static_input.copy_(part)
                    else:
                        static_input.copy_(template)
                        static_input[:rows].copy_(part)
                    graph.replay()
                    # The clone is queued on the same stream before any graph
                    # can replay. Callers never observe reusable graph storage.
                    pieces.append(static_packed[:rows].clone())
                    start += rows
                packed = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
            if not same_stream:
                caller_stream.wait_stream(self.stream)
                packed.record_stream(caller_stream)
        return {'policy': packed[:, :-2], 'far': packed[:, -2],
                'value_logit': packed[:, -1]}

    def _fallback(self, planes):
        height, width = planes.shape[-2:]
        chunk = max(1, self.MAX_CELLS//(height*width))
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
