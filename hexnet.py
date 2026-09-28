"""Dense hex-masked ResNet policy/value network over hexcrop bucketed crops.

Index layout is planes[c, y, x] with x = q', y = r' (see hexcrop). The six
axial neighbours (dq, dr) in {(+-1,0), (0,+-1), (1,-1), (-1,1)} become index
offsets (dy, dx) in {(0,+-1), (+-1,0), (-1,+1), (+1,-1)}. Of the eight 3x3
neighbours, (dy, dx) = (+1, +1) and (-1, -1) are hex distance 2, so HexConv
masks exactly those two taps: kernel[0, 0] and kernel[2, 2]. Line axes are the
index directions (dx, dy) = (1, 0), (0, 1) and (1, -1).
"""
from dataclasses import dataclass, asdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import hexcrop

SCHEMA = 'hexo-dense-policy-value-v1'
AXES = ((1, 0), (0, 1), (1, -1))    # index directions (dx, dy)
WINDOW = 6
FEATURES = len(hexcrop.PLANES)+2*len(AXES)+4
# Inference line convolutions run in batch chunks of at most this many crop cells: their matmul temporaries
# (about four chunk-sized activations, ~45 MB at 96 channels) then stay below the 3x3 convolutions' layout
# copies at dense_selfplay.MAX_CELLS; smaller chunks save no peak memory and add launches.
LINE_CHUNK_CELLS = 55296


@dataclass(frozen=True)
class HexNetConfig:
    blocks: int = 6
    channels: int = 96
    pool_every: int = 3
    line_length: int = 11
    value_hidden: int = 128
    head_channels: int = 32
    aux_heads: bool = True


def _line_kernel(length, taps):
    """[3, length, length] kernels holding `taps` cells from the centre along each axis."""
    kernel, c = torch.zeros(len(AXES), length, length), length//2
    for a, (dx, dy) in enumerate(AXES):
        for i in taps:
            kernel[a, c+i*dy, c+i*dx] = 1
    return kernel


class LineFeatures(nn.Module):
    """Fixed six-window features from own/opponent stones and the crop mask.

    Returns [B, 10, S, S]: for own then opponent and each axis, the best count
    over windows containing the cell that hold no enemy stone and lie fully
    inside the crop (divided by 6); then empty in-crop cells where the best own
    count is >= 4, opponent >= 4, own >= 5, opponent >= 5.
    """
    def __init__(self):
        super().__init__()
        window = _line_kernel(2*WINDOW-1, range(WINDOW))[:, None]   # window starting at the cell
        self.register_buffer('kernel', window.repeat(3, 1, 1, 1), persistent=False)

    def forward(self, own, opp, mask):
        size, pad = own.shape[-1], WINDOW-1
        counts = F.conv2d(torch.cat((own, opp, mask), 1), self.kernel.to(own.dtype), padding=pad, groups=3)
        o, p, m = counts.split(len(AXES), 1)
        full = m > WINDOW-0.5
        open_counts = F.pad(torch.cat((torch.where(full & (p < 0.5), o, 0), torch.where(full & (o < 0.5), p, 0)), 1),
                            (pad, pad, pad, pad))
        best = []
        for a, (dx, dy) in enumerate(AXES):
            # Windows containing the cell start at cell - i*(dx, dy), i = 0..5.
            best.append(torch.stack([open_counts[:, (a, a+3), pad-i*dy:pad-i*dy+size, pad-i*dx:pad-i*dx+size]
                                     for i in range(WINDOW)]).amax(0))
        best = torch.cat(best, 1)[:, (0, 2, 4, 1, 3, 5)]
        mine, theirs = best[:, :3].amax(1, keepdim=True), best[:, 3:].amax(1, keepdim=True)
        empty = mask*(1-own-opp)
        threats = torch.cat((mine > 3.5, theirs > 3.5, mine > 4.5, theirs > 4.5), 1).to(own.dtype)*empty
        return torch.cat((best/WINDOW, threats), 1)


class HexConv(nn.Conv2d):
    """3x3 convolution over the cell and its six hex neighbours (corners [0,0] and [2,2] masked)."""
    def __init__(self, inputs, outputs):
        super().__init__(inputs, outputs, 3, padding=1, bias=False)
        mask = torch.ones(1, 1, 3, 3)
        mask[..., 0, 0] = mask[..., 2, 2] = 0
        self.register_buffer('hex', mask, persistent=False)

    def forward(self, x):
        return F.conv2d(x, self.weight*self.hex, None, 1, 1)


def _toeplitz(weight, n):
    """[C, L] taps -> [C, n, n] with M[c, i, j] = weight[c, i-j+L//2] (zero outside the kernel)."""
    length = weight.shape[1]
    i = torch.arange(n, device=weight.device)
    d = i[:, None]-i[None, :]+length//2
    return torch.where((d >= 0) & (d < length), weight[:, d.clamp(0, length-1)], 0)


def _skewed(x):
    """View of a [C, H, B, H+W-1] buffer as [C, H, B, W] with view[c, y, b, x] = buffer[c, y, b, x+y]: in the
    buffer the anti-diagonals of every image are columns."""
    c, h, b, wide = x.shape
    w = wide-h+1
    return x.as_strided((c, h, b, w), (h*b*wide, b*wide+1, wide, 1), x.storage_offset())


class LineConv(nn.Module):
    """Depthwise length-L convolution along the three axes, summed.

    out[c, y, x] = sum_a sum_i weight[c, a, i] * in[c, y+(i-L//2)*dy_a, x+(i-L//2)*dx_a]
    for the index directions (dx, dy) = (1, 0), (0, 1), (1, -1). cuDNN depthwise
    kernels are slow at these sizes, so each axis is a per-channel Toeplitz
    matmul; the anti-diagonal runs as a column matmul on a skewed copy. The
    matrices are rebuilt on every call (no per-size cache holds GPU memory).
    The matmuls run in a [C, H, B, W] layout, where each is one bmm over the
    channels: a [B, C] batched matmul would copy the matrices B times.
    """
    def __init__(self, channels, length):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels, len(AXES), length))

    def matrices(self, size, dtype):
        """Horizontal, vertical and skewed-diagonal [C, size, size] matrices. With grad enabled they are built
        from _toeplitz (index backward); otherwise by one gather from the zero-padded taps."""
        weight = self.weight.to(dtype)
        if torch.is_grad_enabled():
            # In skewed columns the (1, -1) step is one row up, so the diagonal taps reverse.
            return (_toeplitz(weight[:, 0], size), _toeplitz(weight[:, 1], size).transpose(1, 2),
                    _toeplitz(weight[:, 2].flip(1), size).transpose(1, 2))
        taps = F.pad(weight.transpose(0, 1), (0, 1))     # [3, C, L+1], index L is zero
        index = _band_index(size, weight.shape[-1], weight.device)
        return tuple(taps.gather(2, index.expand(-1, taps.shape[1], -1)).unflatten(2, (size, size)))

    def lines(self, x, matrices):
        """The line convolution of x [B, C, H, W] with matrices from self.matrices, as a [B, C, H, W] view."""
        horizontal, vertical, diagonal = matrices
        b, c, h, w = x.shape
        # The skewed copy (two activations wide) is freed before the planar copy is made, so the temporaries
        # peak near four chunk-sized activations (see LINE_CHUNK_CELLS).
        skewed = x.new_zeros(c, h, b, h+w-1)
        _skewed(skewed).copy_(x.permute(1, 2, 0, 3))
        diagonal = torch.bmm(diagonal, skewed.view(c, h, b*(h+w-1))).view(c, h, b, h+w-1)
        del skewed
        planar = x.permute(1, 2, 0, 3).contiguous()     # [C, H, B, W]: the rows of every image side by side
        out = torch.bmm(planar.view(c, h*b, w), horizontal).view(c, h, b*w)
        # In place: autocast leaves baddbmm_ alone, so its operands take the dtype autocast gave the bmm.
        out.baddbmm_(vertical.to(out.dtype), planar.view(c, h, b*w).to(out.dtype))
        del planar
        return out.view(c, h, b, w).add_(_skewed(diagonal)).permute(2, 0, 1, 3)

    def forward(self, x):
        return self.lines(x, self.matrices(x.shape[-2], x.dtype))

    def add_to(self, x):
        """x += self(x) in place without autograd, in batch chunks of at most LINE_CHUNK_CELLS crop cells."""
        matrices = self.matrices(x.shape[-2], x.dtype)
        for part in x.split(max(1, LINE_CHUNK_CELLS//(x.shape[-2]*x.shape[-1]))):
            part += self.lines(part, matrices)
        return x


_BAND_INDEX = {}


def _band_index(size, length, device):
    """[3, 1, size*size] int64 tap indices of LineConv.matrices for one size; `length` (a zero tap) outside
    the band. Rows: horizontal M[i, j] = w[i-j+c], vertical w[j-i+c], skewed diagonal w[L-1-c+i-j]."""
    key = (size, length, device)
    if key not in _BAND_INDEX:
        i = torch.arange(size, device=device)
        d, c = i[:, None]-i[None, :], length//2
        index = torch.stack((d+c, c-d, length-1-c+d))
        _BAND_INDEX[key] = torch.where((index >= 0) & (index < length), index, length).flatten(1)[:, None]
    return _BAND_INDEX[key]


class _MaskedBatchNorm(torch.autograd.Function):
    """Batch norm over the cells where mask == 1, applied to every cell.

    Saves only tensors that already exist (x, mask) so training memory matches
    cuDNN batch norm. Statistics are reduced in at least float32.
    """
    @staticmethod
    def forward(ctx, x, mask, weight, bias, cells, eps):
        acc = torch.promote_types(x.dtype, torch.float32)
        mean = (x*mask).sum((0, 2, 3), dtype=acc)/cells
        # Centre on the mean rounded to x's dtype: x-shift is (near) exact in bf16, so a large channel
        # mean neither cancels the variance (E[x^2]-mean^2 lost ~10% at mean/std 20) nor the output.
        shift = mean.to(x.dtype)
        delta = mean-shift.to(acc)
        centred = x-shift[:, None, None]
        var = ((centred*mask*centred).sum((0, 2, 3), dtype=acc)/cells-delta*delta).clamp_min(0)
        invstd = torch.rsqrt(var+eps)
        scale = weight*invstd
        ctx.save_for_backward(x, mask, weight, mean, invstd, cells)
        return torch.addcmul((bias-delta*scale)[:, None, None].to(x.dtype), centred, scale[:, None, None].to(x.dtype)), mean, var

    @staticmethod
    def backward(ctx, grad, _mean, _var):
        x, mask, weight, mean, invstd, cells = ctx.saved_tensors
        acc = mean.dtype
        shift = mean.to(x.dtype)
        xhat = torch.addcmul(((shift.to(acc)-mean)*invstd)[:, None, None].to(x.dtype), x-shift[:, None, None],
                             invstd[:, None, None].to(x.dtype))
        # Every output depends on the masked statistics: dL/dx = w*invstd*(dy - mask*(sum dy + xhat*sum dy*xhat)/n).
        db = grad.sum((0, 2, 3), dtype=acc)
        dw = (grad*xhat).sum((0, 2, 3), dtype=acc)
        dx = grad-mask*torch.addcmul((db/cells)[:, None, None].to(x.dtype), xhat, (dw/cells)[:, None, None].to(x.dtype))
        dx = dx*(weight*invstd)[:, None, None].to(x.dtype)
        return dx, None, dw.to(weight.dtype), db.to(weight.dtype), None, None


class MaskedNorm(nn.BatchNorm2d):
    """BatchNorm whose training statistics cover only in-crop cells (mask is full-size, 0/1); with momentum None
    the running statistics are the mean and unbiased variance over every in-crop cell since reset_running_stats."""
    def reset_running_stats(self):
        super().reset_running_stats()
        self.cells_seen = 0.

    def forward(self, x, mask, cells):
        if not self.training:
            return super().forward(x)
        y, mean, var = _MaskedBatchNorm.apply(x, mask, self.weight, self.bias, cells, self.eps)
        with torch.no_grad():
            self.num_batches_tracked += 1
            if self.momentum is None:
                seen, self.cells_seen = self.cells_seen, self.cells_seen+float(cells)
                w, delta = float(cells)/self.cells_seen, mean-self.running_mean
                biased = self.running_var*max(seen-1, 0)/max(seen, 1)
                biased = biased*(1-w)+var*w+delta.square()*w*(1-w)
                self.running_mean.add_(delta*w)
                self.running_var.copy_(biased*self.cells_seen/max(self.cells_seen-1, 1))
            else:
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var*cells/(cells-1).clamp_min(1), self.momentum)
        return y


def act(x, ceiling):
    """relu(x) on in-crop cells, 0 on padding, in one pass: ceiling is +inf in the crop and 0 outside."""
    return torch.clamp(x, x.new_zeros(()), ceiling)


def pool(x, count):
    """Masked mean and max of non-negative, mask-zeroed features x: [B, 2C]."""
    return torch.cat((x.sum((2, 3))/count, x.amax((2, 3))), 1)


class Block(nn.Module):
    def __init__(self, config, pooled):
        super().__init__()
        c = config.channels
        self.norm1, self.conv1 = MaskedNorm(c), HexConv(c, c)
        self.line = LineConv(c, config.line_length) if config.line_length else None
        self.norm2, self.conv2 = MaskedNorm(c), HexConv(c, c)
        self.pool = nn.Linear(2*c, c) if pooled else None

    def forward(self, x, mask, ceiling, count, cells):
        """x may hold junk on padding cells; every convolution input is zero there."""
        y = self.conv1(act(self.norm1(x, mask, cells), ceiling))
        if self.pool is not None:
            y = y+self.pool(pool(act(y, ceiling), count))[:, :, None, None]
        if self.line is not None:
            y = y*mask
            # Recomputing the line matmuls in backward keeps training memory near the plain ResNet's.
            y = y+checkpoint(self.line, y, use_reentrant=False) if torch.is_grad_enabled() else self.line.add_to(y)
        return x+self.conv2(act(self.norm2(y, mask, cells), ceiling))


AUX_PREFIXES = ('aux_spatial.', 'short_value.')


class HexNet(nn.Module):
    """forward(planes [B, 8, S, S] float, mask [B, 1, S, S], aux=True) -> dict of float32 outputs:

    policy [B, S*S] logits; far [B] total logit of all far legal cells (hexcrop
    far mode); value_logit [B] win logit of the side to move, q = tanh(value_logit/2).
    With config.aux_heads and aux=True, also: short_value_logit [B] (side-to-move
    win logit of the searched root value 16 plies ahead); future [B, 2, S, S]
    logits that a cell is occupied after the next 6 / 20 placements;
    opponent_policy [B, S*S] logits of the improved policy recorded at the next ply.
    Aux heads share the policy's 1x1 hidden layer and the value's hidden layer.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or HexNetConfig()
        self.lines = LineFeatures()
        self.stem = HexConv(FEATURES, c.channels)
        self.blocks = nn.ModuleList(Block(c, (i+1) % c.pool_every == 0) for i in range(c.blocks))
        self.norm = MaskedNorm(c.channels)
        self.policy_hidden = nn.Conv2d(c.channels, c.head_channels, 1)
        self.policy = nn.Conv2d(c.head_channels, 1, 1)
        self.far = nn.Linear(2*c.channels, 1)
        self.value_hidden = nn.Linear(2*c.channels, c.value_hidden)
        self.value = nn.Linear(c.value_hidden, 1)
        if c.aux_heads:
            self.aux_spatial = nn.Conv2d(c.head_channels, 3, 1)    # opponent policy, future 6, future 20
            self.short_value = nn.Linear(c.value_hidden, 1)

    def forward(self, planes, mask, aux=True):
        count = mask.sum((2, 3), dtype=torch.float32)    # a bf16 sum rounds counts above 256
        cells = count.sum()
        x = self.stem(torch.cat((planes, self.lines(planes[:, :1], planes[:, 1:2], mask)), 1)*mask)
        if torch.is_grad_enabled() or not x.is_contiguous():
            # Full-size masks in x's memory format keep the elementwise passes vectorized.
            mask = torch.empty_like(x).copy_(mask.expand_as(x))
        else:
            # NCHW inference broadcasts over the outer channel dim at full speed; two fewer activations.
            mask = mask.to(x.dtype, memory_format=torch.contiguous_format)
        ceiling = torch.where(mask > 0, math.inf, 0).to(x.dtype)
        for block in self.blocks:
            x = block(x, mask, ceiling, count, cells)
        x = act(self.norm(x, mask, cells), ceiling)
        pooled = pool(x, count)
        hidden, value = F.relu(self.policy_hidden(x)), F.relu(self.value_hidden(pooled))
        out = dict(policy=self.policy(hidden).flatten(1).float(), far=self.far(pooled)[:, 0].float(),
                   value_logit=self.value(value)[:, 0].float())
        if aux and self.config.aux_heads:
            spatial = self.aux_spatial(hidden).float()
            out.update(short_value_logit=self.short_value(value)[:, 0].float(), future=spatial[:, 1:],
                       opponent_policy=spatial[:, 0].flatten(1))
        return out


def action_logits(policy, far, cells, counts):
    """Logits of each sample's legal list [B, N] from flat crop logits [B, S*S].

    Far cells (index -1) share far - log(far count) when far is given and are
    excluded otherwise; padding beyond counts is excluded. Returns (logits with
    -inf on excluded entries, included mask).
    """
    valid = torch.arange(cells.shape[1], device=cells.device) < counts[:, None]
    far_cells = valid & (cells < 0)
    logits = policy.gather(1, cells.clamp_min(0))
    if far is None:
        valid = valid & ~far_cells
    else:
        logits = torch.where(far_cells, far[:, None]-far_cells.sum(1, keepdim=True).clamp_min(1).log(), logits)
    return logits.masked_fill(~valid, -math.inf), valid


def _weighted_mean(loss, weight):
    """Per-sample losses averaged with weights; weight None means all ones. Zero-weight rows contribute 0."""
    if weight is None:
        return loss.mean()
    return (weight*loss).sum()/weight.sum().clamp_min(1e-8)


def _cross_entropy(policy, far, cells, counts, target_probs, weight):
    return _weighted_mean(policy_row_losses(policy, far, cells, counts, target_probs), weight)


def policy_row_losses(policy, far, cells, counts, target_probs):
    """Per-row cross-entropy [B] of target_probs [B, N] (zero beyond counts) against the legal-set softmax;
    0 for a row with no included entry (e.g. only far cells without a far logit)."""
    logits, valid = action_logits(policy, far, cells, counts)
    log_probs = logits.log_softmax(1).masked_fill(~valid, 0)
    return -(target_probs*log_probs).sum(1)


def policy_loss(policy, far, cells, counts, target_probs, weight=None):
    """Weighted mean of policy_row_losses.

    cells/counts are hexcrop.batch outputs; far cells share the far logit.
    """
    return _cross_entropy(policy, far, cells, counts, target_probs, weight)


def opponent_policy_loss(opponent_policy, cells, counts, target_probs, weight=None):
    """Cross-entropy of the next ply's improved policy over that position's legal cells.

    cells [B, N] are flat indices in this sample's crop (-1 = outside the crop,
    excluded, so their target mass is dropped); weight 0 marks rows without a target.
    """
    return _cross_entropy(opponent_policy, None, cells, counts, target_probs, weight)


def value_loss(value_logit, target_p_win, weight=None):
    """Weighted BCE of a side-to-move win logit against soft targets in [0, 1]."""
    return _weighted_mean(F.binary_cross_entropy_with_logits(value_logit, target_p_win, reduction='none'), weight)


short_value_loss = value_loss


def future_loss(future, target, mask, weight=None):
    """BCE of future occupancy logits [B, 2, S, S] against targets in [0, 1], averaged over
    in-crop cells (mask [B, 1, S, S]) per sample and horizon, then weighted by `weight` [B, 2]."""
    loss = F.binary_cross_entropy_with_logits(future, target, reduction='none')*mask
    per = loss.sum((2, 3))/mask.sum((1, 2, 3)).clamp_min(1)[:, None]
    return _weighted_mean(per.reshape(-1), None if weight is None else weight.reshape(-1))


def memory_format(config):
    """The line convolutions are matmuls that prefer NCHW; the plain trunk is faster channels_last."""
    return torch.contiguous_format if config.line_length else torch.channels_last


def model_digest(model):
    digest = hashlib.sha256(json.dumps(asdict(model.config), sort_keys=True).encode())
    for name, tensor in model.state_dict().items():
        digest.update(name.encode()+str(tensor.dtype).encode()+str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_model(path, model):
    """Write via a sibling temporary file and os.replace so readers never see a partial checkpoint."""
    path = Path(path)
    pending = path.with_name(path.name+'.tmp')
    torch.save(dict(schema=SCHEMA, config=asdict(model.config), state=model.state_dict()), pending)
    for attempt in range(8):
        try:
            os.replace(pending, path)
            return
        except PermissionError:     # Windows: a reader briefly holds the old file open
            if os.name != 'nt' or attempt == 7:
                raise
            time.sleep(.05*(attempt+1))


def load_model(path, device='cpu'):
    data = torch.load(path, map_location=device, weights_only=True)
    if data.get('schema') != SCHEMA:
        raise ValueError(f'{path} is not a {SCHEMA} checkpoint')
    model = HexNet(HexNetConfig(**data['config']))
    missing, unexpected = model.load_state_dict(data['state'], strict=False)
    missing = [k for k in missing if not k.startswith(AUX_PREFIXES)]
    if missing or unexpected:
        raise ValueError(f'{path} does not match its config: missing {missing}, unexpected {unexpected}')
    return model.to(device)


def staging_buffer(buffers, key, rows, shape, dtype, pin):
    """The first `rows` rows of the host buffer buffers[key] [capacity, *shape]: one reusable (pinned) staging
    buffer per key, reallocated when it holds fewer rows. Pinned blocks are rounded up to a power of two by
    torch's host allocator, so a pinned buffer takes every row that fits in its block."""
    buffer = buffers.get(key)
    if buffer is None or buffer.shape[0] < rows:
        row = math.prod(shape)*dtype.itemsize
        capacity = max(rows, (1 << (rows*row-1).bit_length())//row) if pin else rows
        buffer = buffers[key] = torch.empty((capacity, *shape), dtype=dtype, pin_memory=pin)
    return buffer[:rows]


def vram():
    """This process's CUDA caching allocator in MB, {'allocated_mb', 'reserved_mb'}, or None before CUDA is used.
    The CUDA context (about 0.3 GB) comes on top of reserved."""
    if not torch.cuda.is_initialized():
        return None
    return dict(allocated_mb=round(torch.cuda.memory_allocated()/2**20), reserved_mb=round(torch.cuda.memory_reserved()/2**20))


class DenseEvaluator:
    """Frozen bf16 evaluator with the NeuralEvaluator contract consumed by neural_search.

    evaluate(histories) returns, in input order, dicts with 'actions' int64 [N, 2]
    in native legal order, 'logits' float32 [N], 'q' float32 [N] (V(s) broadcast),
    'player', 'remaining' and 'model_version'.
    """
    def __init__(self, model, device='cuda', model_version=None, max_batch=512):
        model = copy.deepcopy(model).requires_grad_(False)
        self.model_version = model_version or model_digest(model)
        self.device = torch.device(device)
        self.cuda = self.device.type == 'cuda'
        self.memory_format = memory_format(model.config)
        self.model = model.to(self.device, memory_format=self.memory_format).eval()
        self.max_batch = max_batch
        self.staging = {}

    @torch.inference_mode()
    def evaluate(self, histories):
        samples = [hexcrop.encode(h) for h in histories]
        result = [None]*len(samples)
        for size, indices in hexcrop.group_by_size(samples).items():
            for start in range(0, len(indices), self.max_batch):
                chunk = indices[start:start+self.max_batch]
                host = staging_buffer(self.staging, size, len(chunk), (len(hexcrop.PLANES), size, size), torch.uint8, self.cuda)
                np.stack([samples[i].planes for i in chunk], out=host.numpy())
                x = host.to(self.device, non_blocking=True)
                x = x.to(memory_format=self.memory_format, dtype=torch.bfloat16 if self.cuda else torch.float32)
                with torch.autocast(self.device.type, torch.bfloat16, enabled=self.cuda):
                    out = self.model(x, x[:, 3:4], aux=False)
                packed = torch.cat((out['policy'], out['far'][:, None], out['value_logit'][:, None]), 1).cpu().numpy()
                if not np.isfinite(packed).all():
                    raise FloatingPointError('Nonfinite dense model predictions')
                for row, i in zip(packed, chunk):
                    s = samples[i]
                    logits = row[np.maximum(s.cells, 0)]
                    if s.far:
                        logits[s.cells < 0] = row[-2]-np.log(s.far)
                    result[i] = dict(actions=s.actions, logits=logits,
                                     q=np.full(len(logits), np.tanh(row[-1]/2), np.float32),
                                     player=s.player, remaining=s.remaining, model_version=self.model_version)
        return result
