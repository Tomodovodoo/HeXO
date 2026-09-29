"""Bounded, model-only CUDA graphs for HexNet's serial training buckets.

Capture happens once, before the first optimizer step. Losses, gradient
clipping, optimizer updates, and EMA remain in the ordinary learner path.
Every bucket completes forward and backward before the next one starts.
"""

import gc
import warnings

import torch
from torch import nn


CAPACITIES = {24: 64, 32: 112, 40: 96, 48: 48, 64: 16}


def _canonical(planes, capacity):
    if planes.shape[0] == capacity:
        return planes
    shape = (capacity, *planes.shape[1:])
    layout = (torch.channels_last if planes.is_contiguous(memory_format=torch.channels_last)
              else torch.contiguous_format)
    padded = torch.empty(shape, device=planes.device, dtype=planes.dtype,
                         memory_format=layout).zero_()
    padded[:planes.shape[0]].copy_(planes)
    return padded


class _TupleForward(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        names = ['policy', 'far', 'value_logit']
        if model.config.aux_heads:
            names += ['short_value_logit', 'opponent_policy']
            names += ['future_masked' if model.future_target == 'masked' else 'future']
        self.names = tuple(names)

    def forward(self, planes):
        # Weight casts cached by an outer autocast context could otherwise die
        # after capture while the graph still holds their device addresses.
        with torch.autocast('cuda', torch.bfloat16, cache_enabled=False):
            out = self.model(planes, planes[:, 3:4], allow_empty=True)
        return tuple(out[name] for name in self.names)


class LearnerGraph:
    """One immutable train-mode model, five fixed graph shapes, one CUDA stream.

Call ``prepare(batch)`` with the first collated batch while parameter grads are
None, then ``begin_step()`` after each optimizer.zero_grad(set_to_none=True).
Pass this object to dense_learn.batch_losses instead of the model. Buckets
above their capacity and unsupported canvases execute eagerly as whole buckets.
"""

    def __init__(self, model, memory_format=torch.contiguous_format):
        device = next(model.parameters()).device
        if device.type != 'cuda' or not model.training or model.net_kernels != 'fused':
            raise ValueError('LearnerGraph requires a train-mode fused CUDA HexNet')
        if memory_format not in (torch.contiguous_format, torch.channels_last):
            raise ValueError('Expected contiguous NCHW or channels-last layout')
        if any(m.momentum is None for m in model.modules() if isinstance(m, nn.BatchNorm2d)):
            raise ValueError('Cumulative normalization has host state and cannot be captured')
        self.model = model
        self.config = model.config
        self.future_target = model.future_target
        self.net_kernels = model.net_kernels
        self.device = device
        self.memory_format = memory_format
        self._stream = torch.cuda.current_stream(device).cuda_stream
        self._params = tuple(p for p in model.parameters() if p.requires_grad)
        self._grads = ()
        self._graphs = {}
        self._pool = None
        self._names = _TupleForward(model).names
        self._prepared = False
        self.capture_memory_error = None

    def _check_stream(self):
        if torch.cuda.current_stream(self.device).cuda_stream != self._stream:
            raise RuntimeError('LearnerGraph capture and replay must use one CUDA stream')

    def _sample(self, side, capacity, batch):
        sample = torch.empty((capacity, 8, side, side), device=self.device,
                             dtype=torch.float32, memory_format=self.memory_format).zero_()
        bucket = batch.get(side)
        if bucket is None or len(bucket['planes']) == 0:
            sample[0, 3, side//2, side//2] = 1
        else:
            rows = min(len(bucket['planes']), capacity)
            sample[:rows].copy_(bucket['planes'][:rows].to(self.device).float())
        return sample

    def prepare(self, batch):
        """Warm all capacities, then capture each complete forward/backward.

        The real first batch supplies sample rows when available. Missing
        canvases use one valid synthetic cell. No graph is added later.
        """
        if self._prepared:
            return
        self._check_stream()
        if not self.model.training or any(p.grad is not None for p in self._params):
            raise ValueError('Prepare before training, with parameter grads cleared')
        saved_buffers = [(buf, buf.detach().clone()) for buf in self.model.buffers()]
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(self.device)
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            samples = {side: self._sample(side, cap, batch) for side, cap in CAPACITIES.items()}
            warm_wrapper = _TupleForward(self.model)
            self._grads = tuple(torch.zeros_like(p, memory_format=torch.preserve_format)
                                for p in self._params)
            for param, grad in zip(self._params, self._grads):
                param.grad = grad
            # Warm the largest working sets before reserving a graph pool.
            order = sorted(CAPACITIES, key=lambda s: CAPACITIES[s]*s*s, reverse=True)
            for side in order:
                outputs = warm_wrapper(samples[side])
                grad_outputs = tuple(torch.ones_like(out) for out in outputs)
                grads = torch.autograd.grad(outputs, self._params, grad_outputs)
                del outputs, grad_outputs, grads
            torch.cuda.synchronize(self.device)
            del warm_wrapper
            gc.collect()
            torch.cuda.empty_cache()
            self._pool = torch.cuda.graph_pool_handle()
            # All forwards finish backward before the next bucket starts, so
            # no graph-owned activation remains live across bucket replays.
            for side in order:
                # make_graphed_callables replaces an nn.Module's forward;
                # each capture needs its own unmodified wrapper.
                wrapper = _TupleForward(self.model)
                graph = torch.cuda.make_graphed_callables(
                    wrapper, (samples[side],), num_warmup_iters=0, pool=self._pool)
                self._graphs[side] = graph
            self._prepared = True
        except torch.OutOfMemoryError as exc:
            # A partially retained graph pool could leave too little room for
            # an eager oversized bucket. Drop every graph and use eager only.
            self.capture_memory_error = str(exc)
            self._graphs.clear()
            self._pool = None
            graph = wrapper = warm_wrapper = samples = None
            self._prepared = True
            warnings.warn('Training graph capture exceeded GPU memory; using eager training',
                          RuntimeWarning, stacklevel=2)
        finally:
            torch.cuda.synchronize(self.device)
            with torch.no_grad():
                for buffer, saved in saved_buffers:
                    buffer.copy_(saved)
            for module, training in modes:
                module.train(training)
            torch.random.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, self.device)
            for param in self._params:
                param.grad = None
            if not self._prepared:
                self._graphs.clear()
                self._pool = None
                self._grads = ()
            if self.capture_memory_error is not None:
                self._grads = ()
                gc.collect()
                torch.cuda.empty_cache()

    def begin_step(self):
        """Attach external gradients after optimizer.zero_grad, then zero them."""
        if not self._prepared:
            raise RuntimeError('Call prepare before begin_step')
        self._check_stream()
        if not self._graphs:
            return
        for param, grad in zip(self._params, self._grads):
            param.grad = grad
        torch._foreach_zero_(self._grads)

    def __call__(self, planes, mask, aux=True):
        if not self._prepared:
            raise RuntimeError('Call prepare before forwarding through LearnerGraph')
        if not self.model.training:
            raise ValueError('Training graphs cannot replay in eval mode')
        self._check_stream()
        rows, channels, side, width = planes.shape
        capacity = CAPACITIES.get(side)
        if (not aux or channels != 8 or side != width or capacity is None or rows > capacity
                or side not in self._graphs or mask.data_ptr() != planes[:, 3:4].data_ptr()):
            return self.model(planes, mask, aux=aux)
        if planes.device != self.device or planes.dtype != torch.float32:
            raise ValueError('Expected float32 CUDA planes on the captured model device')
        result = self._graphs[side](_canonical(planes, capacity))
        return {name: value[:rows] for name, value in zip(self._names, result)}

    def close(self):
        """Release graphs after the current training step has completed."""
        torch.cuda.synchronize(self.device)
        self._graphs.clear()
        self._pool = None
        for param, grad in zip(self._params, self._grads):
            if param.grad is grad:
                param.grad = None
        self._grads = ()
        self._prepared = False
        self.capture_memory_error = None
