"""Dense learner: trains one HexNet variant on a run's shards and exports EMA checkpoints (run layout: dense_config).

A checkpoint holds model.pt (raw weights) and ema.pt (what actors play), both hexnet.save_model, optimizer.pt and
manifest.json {variant, step, samples_seen, created_at, model_sha256, ema_sha256 (hexnet.model_digest), metrics,
learner (effective LearnerSettings), model (ModelSettings), copied_from, rows, pacing}. Events have source 'learner' and kind
export, info, error or replace. league.json is read for population replacement, never written. metrics/learner-
<variant>.jsonl gets a line every log_every steps (losses averaged since the previous averaging point, which also
falls on every tenth step) and one per export with validation_fields(metrics) of the manifest (Learner.export): EMA
losses on held-out rows of the window and on fixed per-source subsets (dense_data.ValidationSets), plus the value
loss of finished held-out games against their hard outcome by plies remaining (remaining_curve), the policy and
value losses against the ply from the start (ply_curve, ply_split), both over (ply from the start, plies remaining)
cells (surfaces), the value regret against what the search knew (calibration_reference, value_regret) and on rows
with a proof (value_regret_proven), and the replay window's count of rows with a proof (proven_rows). The EMA
averages parameters only; each export first recomputes its norm statistics (Learner.recalibrate), since the
raw model's running statistics do not describe the EMA weights.

VRAM: each export ends by returning the caching allocator's unused blocks to the driver (Learner.release), since
recalibration and validation raise the peak above what training needs. vram_reserved_mb > 0 caps the allocator
(Learner.cap_vram); learner-status.json and the metrics lines carry vram {allocated_mb, reserved_mb} (zeros off CUDA).

Every target is derived here from episodes (dense_data.examples), so value_target, td_lambda, outcome_lambda,
bootstrap_weight and short_value_horizon are learner settings; outcome_weight weighs a second value-logit loss, the
BCE against the hard outcome of finished games (head outcome_bce, always logged), KataGo-style. Rows with an exact
label (a nonzero `proven`) are left out of it, since their value target is the proven result; validation reports
the outcome BCE of held-out rows of finished games split into rows with and without one (outcome_split).
With --deblunder-weight > 0, eligible earlier losing-owner rows use the soft outcome for both value losses;
validation also reports value_bce_deblundered and deblundered_rows, while outcome splits and curves keep the
original game outcome. The value
target calibration map (Learner.calibrate) is fitted at startup and refitted at every export, recorded in the manifest as
metrics.calibration (calibration_report) and handed to the render workers; value_target 'calibrated' trains on the
newest map (hard outcomes while none is fitted). Batches are rendered by dense_data.Renderers worker processes
(--workers) with random hex symmetries; the window of this process and its workers share the variant's policy file
directory (policy_dir, dense_data.ReplayWindow). On CUDA this process runs torch on --threads CPU threads. learner-status.json reports
data_wait_fraction, the share of the recent step time spent waiting for a rendered batch (wait_fraction). The loss
of one optimizer step is, per head, the weighted mean over every row of the batch that has that target, summed with
the head coefficients; each crop bucket is a separate forward pass whose gradients accumulate (buckets padded to
QUANTUM rows with inert rows).
Pacing: at most samples_per_row * (retained trained rows in all shards; a historical opponent's plies are not
trained, see dense_data.trained) samples are presented since the pacing base (manifest pacing, Learner.rebase);
beyond that the learner waits. With cheap_row_fraction < 1 only that share of the ordinary cheap rows is retained
(full-search and exact rows always are, dense_data.retained), both for sampling and for this count, so
samples_per_row stays per retained row; held-out rows are validated whether retained or not. learner-status.json
reports retained_rows and retained_fraction of the window. The window
is sized in full-search rows (dense_data.ReplayWindow). With phase_rows > 0 the learner alternates phases (Phase):
it idles until the untrained backlog (backlog) reaches phase_rows, then trains until the pacing limit, so actors
following the phase (ActorSettings.phase_follow) have the GPU to themselves while it idles; exports still fall on
every export_every-th step.
"""
import argparse
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import shutil
import time
import zlib

import numpy as np
import torch

import dense_config
import dense_data
import hexnet
from train import write_json

HEADS = ('policy_ce', 'value_bce', 'short_value_bce', 'opponent_ce', 'future_bce', 'outcome_bce')
WEIGHTS = ('policy_weight', 'value_weight', 'short_weight', 'next_weight', 'future_weight', 'outcome_weight')
VALIDATION_ROWS = 2048
RECALIBRATION_ROWS = 4096
QUANTUM = 16  # bucket rows are padded to a multiple of this
STATUS_SECONDS = 2.
REFRESH_SECONDS = 30.
# Settings a replacement copy keeps from this learner rather than the source checkpoint's manifest.
KEEP = ('variant', 'protect_steps', 'replace_interval', 'replace_margin', 'validation_fraction', 'validation_rows',
        'validation_quota', 'export_every', 'log_every', 'vram_reserved_mb', 'phase_rows', 'deblunder_weight',
        'optimizer', 'proof_policy_weight', 'future_target', 'regret_fraction', 'cheap_row_fraction')
LOGGED = dict(zip(HEADS, ('policy_ce', 'value_bce', 'short_value_bce', 'next_ce', 'future_bce', 'outcome_bce')))  # metrics log names
REMAINING_GRID = tuple(range(0, 161, 4))  # plies remaining at which value curves are sampled
REMAINING_SIGMA = 4.
PLY_GRID = tuple(range(0, 385, 8))  # plies from the start at which by-ply curves are sampled
PLY_SIGMA = 4.
EARLY_PLY, LATE_PLY = 20, 60  # ply_split: early rows have ply < EARLY_PLY, late rows ply >= LATE_PLY
HORIZON_BCE = math.log(2)/2  # midpoint between a perfect and a chance value head
CURVE_SOURCES = ('fresh', 'newest')
CALIBRATION_V = tuple(np.linspace(-1, 1, 9).tolist())  # calibration_report table grid
CALIBRATION_H = tuple(range(0, 161, 8))
SURFACE_WIDTH, SURFACE_LIMIT, SURFACE_MIN = 16, 384, 8  # surfaces: cell width, axis limit, rows per reported cell
NEAR_END, FAR_END = 20, 60  # value_regret: early rows have remaining < NEAR_END, late rows remaining >= FAR_END
# (low, high) for replacement perturbations; td_lambda, outcome_lambda and ema are perturbed through 1 - x.
BOUNDS = dict(lr=(1e-5, 3e-3), weight_decay=(1e-5, 1e-1), bootstrap_weight=(0., 1.), td_lambda=(0., .995),
              outcome_lambda=(0., 1.), ema=(.99, .9999))
COMPLEMENTED = ('td_lambda', 'outcome_lambda', 'ema')


def validation_fields(metrics):
    """The metrics-log fields of an export's manifest metrics: metrics.validation (HEADS under LOGGED names, other
    keys as they are) plus the scalar entries of metrics.validation_sources (curves and surfaces stay in the
    manifest); None when both are null."""
    sources = {k: v for k, v in (metrics.get('validation_sources') or {}).items() if not isinstance(v, (list, dict))}
    fields = {LOGGED.get(h, h): v for h, v in (metrics['validation'] or {}).items()} | sources
    return fields or None


def policy_validation_rows(out, batch):
    """Policy CE, target entropy, KL and top-1 agreement for each collated row."""
    target = torch.zeros(batch['mask'].shape).masked_scatter_(batch['mask'], batch['policy'])
    policy, far = out['policy'].float().cpu(), out['far'].float().cpu()
    ce = hexnet.policy_row_losses(policy, far, batch['cells'], batch['counts'], target)
    entropy = -torch.xlogy(target, target).sum(1)
    logits, _ = hexnet.action_logits(policy, far, batch['cells'], batch['counts'])
    top1 = ((logits == logits.max(1, keepdim=True).values) &
            (target == target.max(1, keepdim=True).values)).any(1).float()
    return ce, entropy, ce-entropy, top1


def smoothed(x, ys, grid, sigma):
    """Per y in `ys`, a float array holding per grid point g the mean of y weighted by exp(-(x-g)^2 / (2 sigma^2)),
    nan where the weights sum below 1."""
    x, g = np.asarray(x, np.float64), np.asarray(grid, np.float64)
    k = np.exp(-.5*((g[:, None]-x[None])/sigma)**2)
    mass = k.sum(1)
    return [np.where(mass >= 1, k@np.asarray(y, np.float64)/np.maximum(mass, 1e-12), np.nan) for y in ys]


def compact(curve):
    """A curve as a JSON list: values rounded to 4 decimals, None where not finite."""
    return [round(float(v), 4) if np.isfinite(v) else None for v in curve]


def binary_entropy(rate):
    """Elementwise binary entropy in nats of an outcome rate (0 at rates 0 and 1, nan stays nan)."""
    rate = np.asarray(rate, np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((rate <= 0) | (rate >= 1), 0., -rate*np.log(rate)-(1-rate)*np.log(1-rate))


def surfaces(ply, remaining, columns, width=SURFACE_WIDTH, limit=SURFACE_LIMIT, min_cells=SURFACE_MIN):
    """Per-cell means over `width`-ply cells in (ply from the start, plies remaining), each axis covering
    [0, limit) (rows outside are dropped): (grid, means). grid is {ply_bins, remaining_bins (lower cell edges),
    counts ([ply bin][remaining bin] row counts)}; means holds per column a float array [ply bin][remaining bin],
    nan in cells with fewer than `min_cells` rows."""
    n = -(-limit//width)
    p, r = (np.asarray(x, np.float64)//width for x in (ply, remaining))
    keep = (p >= 0) & (p < n) & (r >= 0) & (r < n)
    index = (p[keep]*n+r[keep]).astype(np.int64)
    counts = np.bincount(index, minlength=n*n).reshape(n, n)
    means = [np.where(counts >= min_cells, np.bincount(index, np.asarray(y, np.float64)[keep], n*n).reshape(n, n)
                      / np.maximum(counts, 1), np.nan) for y in columns]
    bins = list(range(0, n*width, width))
    return dict(ply_bins=bins, remaining_bins=bins, counts=counts.tolist()), means


def searched_value(episode, ply):
    """Root value in [-1, 1] for the side to move at `ply` from the newest full search at or before it (a root
    value counts as full-search when the episode records no full_search flags), negated when that search's mover
    differs; nan without one."""
    roots, full = episode['root_values'], episode.get('full_search')
    for t in range(ply, -1, -1) if roots is not None else ():
        if roots[t] is not None and (full is None or full[t]):
            return roots[t] if dense_data.player_at(t) == dense_data.player_at(ply) else -roots[t]
    return math.nan


def calibration_reference(fit_value, fit_remaining, fit_outcome, value, remaining,
                          ridge=dense_data.CALIBRATION_RIDGE, steps=dense_data.CALIBRATION_ITERATIONS):
    """P(outcome = 1 | value, remaining) for the query rows from dense_data.fit_calibration_rows (the value target
    map's basis, shrinkage and fit, with `ridge` and at most `steps` Newton steps) on the fit rows, with base the
    mean fit outcome clipped to [1e-3, 1 - 1e-3]. Query rows without a finite value, and every row when no fit row
    has one, get the base rate; None without fit rows."""
    y = np.asarray(fit_outcome, np.float64)
    if not len(y): return None
    base = float(np.clip(y.mean(), 1e-3, 1-1e-3))
    return dense_data.fit_calibration_rows(fit_value, fit_remaining, y, base, ridge, steps).predict(value, remaining)


def value_regret(bce, outcome, reference, remaining):
    """{value_regret, value_regret_early, value_regret_late}: the mean of bce minus the BCE of `reference` against
    `outcome` over all rows, rows with remaining < NEAR_END and rows with remaining >= FAR_END; each None without
    rows (all None without a reference)."""
    keys = ('value_regret', 'value_regret_early', 'value_regret_late')
    if reference is None: return dict.fromkeys(keys)
    y, h = np.asarray(outcome, np.float64), np.asarray(remaining, np.float64)
    r = np.clip(np.asarray(reference, np.float64), 1e-7, 1-1e-7)
    regret = np.asarray(bce, np.float64)+y*np.log(r)+(1-y)*np.log(1-r)
    return {k: float(regret[m].mean()) if m.any() else None
            for k, m in zip(keys, (np.ones(len(h), bool), h < NEAR_END, h >= FAR_END))}


def outcome_split(bce, finished, exact):
    """{outcome_bce_exact, outcome_bce_exact_rows, outcome_bce_unproven, outcome_bce_unproven_rows}: the mean of
    `bce` (the value logit's BCE against the hard outcome) over rows of finished games with an exact label and over
    those without one, and the row counts; a mean is None without rows."""
    bce, finished, exact = np.asarray(bce, np.float64), np.asarray(finished, bool), np.asarray(exact, bool)
    out = {}
    for name, m in (('exact', finished & exact), ('unproven', finished & ~exact)):
        out |= {f'outcome_bce_{name}': float(bce[m].mean()) if m.any() else None, f'outcome_bce_{name}_rows': int(m.sum())}
    return out


def ply_curve(ply, loss, grid=PLY_GRID, sigma=PLY_SIGMA):
    """Loss against the ply from the start, smoothed like remaining_curve: a compact list over `grid`."""
    return compact(smoothed(ply, [loss], grid, sigma)[0])


def ply_split(ply, loss):
    """(mean loss of rows with ply < EARLY_PLY, mean loss of rows with ply >= LATE_PLY), each None without rows."""
    ply, loss = np.asarray(ply, np.float64), np.asarray(loss, np.float64)
    return tuple(float(loss[m].mean()) if m.any() else None for m in (ply < EARLY_PLY, ply >= LATE_PLY))


def remaining_curve(remaining, bce, target, grid=REMAINING_GRID, sigma=REMAINING_SIGMA):
    """Value loss against plies remaining over rows of finished games: {value_curve, value_excess_curve,
    value_bce_last20, value_horizon}. value_curve holds, per grid point g, the mean of the rows' BCE weighted by
    exp(-(remaining-g)^2 / (2 sigma^2)); value_excess_curve subtracts the binary entropy of the equally weighted
    mean target there (the loss of a predictor that only knows the outcome rate at that distance, so negative
    values beat the base rate); both rounded to 4 decimals, or None where the weights sum below 1. value_bce_last20
    is the mean BCE of rows with remaining <= 20. value_horizon is the plies remaining at which the BCE curve first
    exceeds HORIZON_BCE, linear between defined grid points (the grid point itself when the curve starts above).
    Each scalar is None when undefined."""
    r, b, t = (np.asarray(x, np.float64) for x in (remaining, bce, target))
    g = np.asarray(grid, np.float64)
    curve, rate = smoothed(r, (b, t), g, sigma)
    excess = curve-binary_entropy(rate)
    horizon = None
    for i in range(len(g)):
        if curve[i] > HORIZON_BCE:
            if i and np.isfinite(curve[i-1]):
                horizon = g[i-1]+(HORIZON_BCE-curve[i-1])/(curve[i]-curve[i-1])*(g[i]-g[i-1])
            else:
                horizon = g[i]
            break
    last = r <= 20
    return dict(value_curve=compact(curve), value_excess_curve=compact(excess),
                value_bce_last20=float(b[last].mean()) if last.any() else None,
                value_horizon=None if horizon is None else round(float(horizon), 2))


def calibration_report(calibration, games):
    """metrics.calibration of a manifest: {games (fitted games), fitted, base_rate, coef (dense_data.calibration_features
    order), h_knots, v_clip, table {v, h, p}} where p[i][j] is the map at h[i] plies remaining and root value v[j]; only games
    and fitted (False) when `calibration` is None."""
    if calibration is None:
        return dict(games=games, fitted=False)
    v, h = CALIBRATION_V, CALIBRATION_H
    p = calibration.predict(np.tile(v, len(h)), np.repeat(h, len(v))).reshape(len(h), len(v))
    return dict(games=games, fitted=True, base_rate=round(calibration.base, 6), coef=[round(c, 6) for c in calibration.coef],
                h_knots=list(dense_data.H_KNOTS), v_clip=dense_data.V_CLIP,
                table=dict(v=list(v), h=list(h), p=np.round(p, 4).tolist()))


def validation_sets(run, settings, seed):
    """The run's dense_data.ValidationSets sized by LearnerSettings validation_rows (limit) and validation_quota."""
    return dense_data.ValidationSets(run, settings.validation_fraction, seed, settings.validation_rows, settings.validation_quota)


def policy_dir(run, variant):
    """The variant's dense_data.ReplayWindow policy file directory; variants never share one, since each window deletes
    the files of shards outside itself."""
    return run/'cache'/'policies'/variant


def status_path(run, variant):
    return run/('learner-status.json' if variant == 'main' else f'learner-status-{variant}.json')


def wait_fraction(rate):
    """Share of step time spent waiting for a batch over `rate` entries (finished, samples, wait, train seconds);
    0. without entries."""
    total = sum(r[2]+r[3] for r in rate)
    return sum(r[2] for r in rate)/total if total > 0 else 0.


NO_BASE = dict(rows=0, samples=0)


def backlog(samples_seen, total_rows, samples_per_row, base=NO_BASE):
    """Untrained backlog in rows since the pacing base {rows, samples}: the part of the pacing budget
    samples_per_row * rows not yet presented, divided by samples_per_row, i.e. rows - samples / samples_per_row with
    rows = total_rows - base rows and samples = samples_seen - base samples: the rows still below the target if
    every samples_per_row presented samples had brought one whole row up to it. The learner cannot take another
    batch once it falls below batch / samples_per_row (paced)."""
    return total_rows-base['rows']-(samples_seen-base['samples'])/samples_per_row


def paced(samples_seen, total_rows, samples_per_row, batch, base=NO_BASE):
    """True when the next batch would exceed the pacing budget samples_per_row * (total_rows - base rows) for the
    samples presented since the base."""
    return samples_seen-base['samples']+batch > samples_per_row*(total_rows-base['rows'])


class Phase:
    """Phased training schedule. due(phase_rows, backlog_rows, paced) says whether the learner trains now, where
    `paced` means the pacing limit forbids the next batch. phase_rows 0: always due (the pacing limit alone decides).
    phase_rows > 0: a training phase starts once backlog_rows reaches phase_rows and lasts until `paced`; between
    phases the learner idles. `training` is the current phase."""

    def __init__(self):
        self.training = False

    def due(self, phase_rows, backlog_rows, paced):
        if phase_rows <= 0:
            return True
        if paced:
            self.training = False
        elif backlog_rows >= phase_rows:
            self.training = True
        return self.training


def pad(bucket, quantum):
    """Append inert rows up to a multiple of `quantum` rows so cuDNN sees few distinct batch shapes.
    Padding rows have one empty crop cell (keeps pooled means finite and adds one cell to the norm
    statistics), no legal or opponent cells and zero weight on every head."""
    extra = -len(bucket['counts']) % quantum
    if not extra:
        return bucket
    out = {}
    for k, v in bucket.items():
        if k == 'policy':
            out[k] = v
        elif k == 'offsets':
            out[k] = torch.cat((v, v[-1:].expand(extra)))
        else:
            fill = torch.full((extra, *v.shape[1:]), -1 if k in ('cells', 'next_cells') else 0, dtype=v.dtype)
            if k == 'planes':
                fill[:, 3, v.shape[2]//2, v.shape[3]//2] = 1
            out[k] = torch.cat((v, fill))
    return out


def forward(model, planes, device, memory_format):
    """(model outputs, mask [B,1,S,S]) for uint8 planes [B,8,S,S]; bf16 autocast on CUDA."""
    planes = planes.to(device, non_blocking=True).float().contiguous(memory_format=memory_format)
    mask = planes[:, 3:4]
    with torch.autocast(device.type, torch.bfloat16, enabled=device.type == 'cuda'):
        return model(planes, mask), mask


def deblunder_split(bce, changed):
    """Value BCE and row count on the soft targets introduced by deblundering."""
    bce, changed = np.asarray(bce), np.asarray(changed, bool)
    return dict(value_bce_deblundered=float(bce[changed].mean()) if changed.any() else None,
                deblundered_rows=int(changed.sum()))


def head_losses(model, batch, device, memory_format):
    """Per-head weighted means over one bucket and the bucket's weight sums, both [len(HEADS)] on device.
    outcome_bce uses the soft outcome_target when deblundering is enabled, otherwise the hard outcome."""
    b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    out, mask = forward(model, b['planes'], device, memory_format)
    target = torch.zeros(b['mask'].shape, device=device).masked_scatter_(b['mask'], b['policy'])
    losses = [hexnet.policy_loss(out['policy'], out['far'], b['cells'], b['counts'], target, b['policy_weight']),
              hexnet.value_loss(out['value_logit'], b['value'], b['value_weight'])]
    if model.config.aux_heads:
        losses += [hexnet.short_value_loss(out['short_value_logit'], b['short_value'], b['short_weight']),
                   hexnet.opponent_policy_loss(out['opponent_policy'], b['next_cells'], b['next_counts'],
                                               b['next_policy'], b['next_weight']),
                   hexnet.masked_future_loss(out['future_masked'], b['future'], b['planes'], b['future_weight'])
                   if model.future_target == 'masked' else
                   hexnet.future_loss(out['future'], b['future'].float(), mask, b['future_weight'])]
    else:
        losses += [torch.zeros((), device=device)]*3
    losses.append(hexnet.value_loss(out['value_logit'], b.get('outcome_target', b['outcome']), b['outcome_weight']))
    return torch.stack(losses), torch.stack([b[k].sum() for k in WEIGHTS])


def batch_losses(model, batch, coefficients, device, memory_format, train):
    """Head losses [len(HEADS)] of a collated {S: bucket} batch; with `train`, backpropagates the combined loss
    bucket by bucket (each bucket's head mean scaled by its share of that head's batch weight)."""
    totals = torch.stack([sum(b[k].sum() for b in batch.values()) for k in WEIGHTS]).to(device).clamp_min(1e-8)
    logged = torch.zeros(len(HEADS), device=device)
    for bucket in batch.values():
        with torch.set_grad_enabled(train):
            losses, weights = head_losses(model, pad(bucket, QUANTUM), device, memory_format)
            share = losses*weights/totals
            if train:
                (share*coefficients).sum().backward()
        logged += share.detach()
    return logged


def muon_parameters(model):
    """Split trunk hidden weights from the parameters retained by AdamW."""
    muon, adamw = [], []
    for name, param in model.named_parameters():
        parts = name.split('.')
        selected = (name in ('value_hidden.weight', 'policy_hidden.weight') or
                    len(parts) == 4 and parts[0] == 'blocks' and parts[1].isdigit() and
                    parts[2] in ('conv1', 'conv2', 'pool') and parts[3] == 'weight')
        (muon if selected else adamw).append(param)
    return muon, adamw


class MuonAdamW:
    """Muon on 2D weight copies, with AdamW on the remaining model parameters."""
    def __init__(self, model, s):
        self.weights, remaining = muon_parameters(model)
        # Reshape copies channels-last conv filters; 2D weights can share their storage.
        self.matrices = [p.detach().reshape(p.shape[0], -1).requires_grad_() for p in self.weights]
        self.muon = torch.optim.Muon(self.matrices, lr=s.lr, weight_decay=.1, momentum=.95,
                                     nesterov=True, adjust_lr_fn='match_rms_adamw')
        groups = [dict(params=[p for p in remaining if p.ndim > 1], weight_decay=s.weight_decay),
                  dict(params=[p for p in remaining if p.ndim <= 1], weight_decay=0.)]
        self.adamw = torch.optim.AdamW(groups, lr=s.lr, betas=(.9, .98), fused=next(model.parameters()).is_cuda)
        self.optimizers = (self.muon, self.adamw)

    @property
    def param_groups(self):
        return self.muon.param_groups+self.adamw.param_groups

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)
        for p in self.weights:
            p.grad = None

    @torch.no_grad()
    def step(self):
        for p, matrix in zip(self.weights, self.matrices):
            matrix.grad = None if p.grad is None else p.grad.detach().reshape_as(matrix)
        self.muon.step()
        for p, matrix in zip(self.weights, self.matrices):
            if p.data_ptr() != matrix.data_ptr():
                p.copy_(matrix.reshape_as(p))
        self.adamw.step()

    def state_dict(self):
        return dict(muon=self.muon.state_dict(), adamw=self.adamw.state_dict())

    def load_state_dict(self, state):
        self.muon.load_state_dict(state['muon'])
        self.adamw.load_state_dict(state['adamw'])


def make_optimizer(model, s):
    if s.optimizer == 'muon':
        return MuonAdamW(model, s)
    groups = [dict(params=[p for p in model.parameters() if p.ndim > 1], weight_decay=s.weight_decay),
              dict(params=[p for p in model.parameters() if p.ndim <= 1], weight_decay=0.)]
    return torch.optim.AdamW(groups, lr=s.lr, betas=(.9, .98), fused=next(model.parameters()).is_cuda)


@torch.no_grad()
def update_ema(ema, model, decay):
    pairs = list(zip(ema.parameters(), model.parameters()))
    if model.future_target == 'masked' and model.config.aux_heads:
        pairs = [(e[:1], p[:1]) if p is model.aux_spatial.weight or p is model.aux_spatial.bias else (e, p)
                 for e, p in pairs]  # only the opponent-policy channel of the legacy head is active
    torch._foreach_lerp_([e for e, _ in pairs], [p for _, p in pairs], 1-decay)


def perturb(settings, factor_rng, amount):
    """Multiply each BOUNDS setting (1 - x for COMPLEMENTED ones) by a factor in [1-amount, 1+amount], clipped to
    BOUNDS."""
    values = {}
    for name, (low, high) in BOUNDS.items():
        x, f = getattr(settings, name), factor_rng.uniform(1-amount, 1+amount)
        values[name] = float(np.clip(1-(1-x)*f if name in COMPLEMENTED else x*f, low, high))
    return replace(settings, **values)


def latest_rated(league):
    """{variant: league entry} of each variant's newest rated checkpoint that was not demoted for a regression."""
    latest = {}
    for c in league['checkpoints']:
        if c.get('elo') is not None and not c.get('demoted') and c['step'] >= latest.get(c['variant'], {'step': -1})['step']:
            latest[c['variant']] = c
    return latest


def checkpoints(run, variant):
    root = run/'checkpoints'/variant
    return sorted((p for p in root.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)) if root.exists() else []


class Learner:
    def __init__(self, run, settings, config, initial=None, overrides=None):
        """Resume from the newest checkpoint of settings.variant if any (its saved settings under the explicit
        `overrides`), else start from `initial` or random weights. The VRAM cap of the effective settings is
        installed before any CUDA allocation."""
        self.run, self.settings, self.config, self.overrides = run, settings, config, overrides or {}
        self.device = torch.device(config.device)
        self.memory_format = hexnet.memory_format(config.model)
        saved = checkpoints(run, settings.variant)
        manifest = json.loads((saved[-1]/'manifest.json').read_text(encoding='utf-8')) if saved else None
        if saved:
            # Settings saved by the last export (including replacement perturbations) under explicit CLI overrides.
            self.settings = replace(dense_config.section('learner', manifest['learner']), **self.overrides)
        self.cap_vram()
        self.model = self.place(hexnet.HexNet(hexnet.HexNetConfig(**asdict(config.model)), self.settings.future_target))
        self.step = self.samples_seen = self.optimizer_started = self.ema_updates = 0
        self.copied_from = None
        self.pacing, self.pacing_per_row, self.resumed_rows = dict(NO_BASE), self.settings.samples_per_row, None
        self.pacing_fraction = self.settings.cheap_row_fraction
        if saved:
            self.resume(saved[-1], manifest)
        else:
            if initial:
                self.load_weights(initial)
            self.ema = copy.deepcopy(self.model)
            self.optimizer = make_optimizer(self.model, self.settings)
        self.start_step = self.last_copy = self.step
        self.last_export = self.step if saved else None
        self.metrics = None
        self.calibration = self.calibration_report = None

    def cap_vram(self):
        """With settings.vram_reserved_mb > 0 on CUDA, cap this process's caching allocator at that many MB
        (torch.cuda.set_per_process_memory_fraction): the allocator frees cached blocks and retries before growing
        past it, and an allocation that still does not fit raises torch.OutOfMemoryError instead of spilling into
        shared system memory. Raises ValueError when the cap exceeds the device's memory."""
        mb = self.settings.vram_reserved_mb
        if mb <= 0 or self.device.type != 'cuda':
            return
        total = torch.cuda.get_device_properties(self.device).total_memory
        if mb*2**20 > total:
            raise ValueError(f'vram_reserved_mb {mb} exceeds the device memory of {total//2**20} MB')
        torch.cuda.set_per_process_memory_fraction(mb*2**20/total, torch.cuda.current_device() if self.device.index is None else self.device.index)

    def release(self):
        """Return the caching allocator's unused blocks to the driver (torch.cuda.empty_cache); a no-op off CUDA."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def vram(self):
        """{allocated_mb, reserved_mb} of this process's caching allocator (hexnet.vram), zeros off CUDA."""
        return hexnet.vram() if self.device.type == 'cuda' else dict(allocated_mb=0, reserved_mb=0)

    def optimizer_state_mb(self):
        """CUDA storage held by optimizer state tensors, in MiB."""
        optimizers = self.optimizer.optimizers if self.settings.optimizer == 'muon' else (self.optimizer,)
        return sum(value.numel()*value.element_size() for optimizer in optimizers for state in optimizer.state.values()
                   for value in state.values() if torch.is_tensor(value) and value.is_cuda)/2**20

    def place(self, model):
        return model.to(self.device, memory_format=self.memory_format)

    def load_weights(self, path):
        source = hexnet.load_model(path, future_target=self.settings.future_target)
        if source.config != self.model.config:
            raise ValueError(f'{path} has model {source.config}, the run uses {self.model.config}')
        self.model.load_state_dict(source.state_dict())

    def resume(self, path, manifest):
        """Load checkpoint weights and counters; reuse optimizer state only for the same kind and future target."""
        changed = manifest['learner'].get('future_target', 'legacy') != self.settings.future_target
        self.model = self.place(hexnet.load_model(path/'model.pt', future_target=self.settings.future_target))
        self.ema = self.place(hexnet.load_model(path/'ema.pt', future_target=self.settings.future_target))
        if changed and self.settings.future_target == 'masked' and self.model.config.aux_heads:
            self.ema.future_masked.load_state_dict(self.model.future_masked.state_dict())
        if self.model.config != hexnet.HexNetConfig(**asdict(self.config.model)):
            raise ValueError(f'{path} does not match the run model settings')
        state = torch.load(path/'optimizer.pt', map_location=self.device, weights_only=True)
        self.optimizer = make_optimizer(self.model, self.settings)
        saved_kind = manifest.get('optimizer_kind', manifest['learner'].get('optimizer', 'adamw'))
        if saved_kind == self.settings.optimizer and not changed:
            self.optimizer.load_state_dict(state['optimizer'])
        adamw = self.optimizer.adamw if self.settings.optimizer == 'muon' else self.optimizer
        for group, decay in zip(adamw.param_groups, (self.settings.weight_decay, 0.)):
            group['weight_decay'] = decay
        self.step, self.samples_seen = manifest['step'], manifest['samples_seen']
        self.optimizer_started = state['optimizer_started'] if saved_kind == self.settings.optimizer and not changed else self.step
        self.ema_updates = 0 if changed else state['ema_updates']
        if changed:
            dense_config.log_event(self.run, 'learner', 'info',
                                   f'future target switched to {self.settings.future_target}; optimizer and EMA update count reset')
        self.copied_from = manifest.get('copied_from')
        self.pacing, self.pacing_per_row = dict(manifest.get('pacing', NO_BASE)), manifest['learner']['samples_per_row']
        self.pacing_fraction = manifest['learner'].get('cheap_row_fraction', 1.)
        self.resumed_rows = manifest.get('rows')
        if saved_kind != self.settings.optimizer:
            dense_config.log_event(self.run, 'learner', 'optimizer_reset',
                  f'{self.settings.variant} optimizer changed from {saved_kind} to {self.settings.optimizer} at step {self.step}',
                  variant=self.settings.variant, step=self.step, old_optimizer=saved_kind, new_optimizer=self.settings.optimizer)
        dense_config.log_event(self.run, 'learner', 'info', f'{self.settings.variant} resumed from step {self.step}', variant=self.settings.variant, step=self.step)

    def rebase(self, total_rows):
        """Keep the pacing base {rows, samples} (self.pacing) tied to settings.samples_per_row and
        settings.cheap_row_fraction; called before every pacing check with the window's pacing count. When either
        differs from the one the base was set under (the resumed checkpoint's, or a replacement copy's), the base
        becomes (rows, samples_seen) and an info event records it with the old and new settings. rows is the
        resumed checkpoint's manifest rows on the first call after a resume when cheap_row_fraction is unchanged (so
        restarts from the same checkpoint agree on the base), else total_rows (a changed fraction changes how rows
        are counted)."""
        old, new = (self.pacing_per_row, self.pacing_fraction), (self.settings.samples_per_row, self.settings.cheap_row_fraction)
        rows = total_rows if self.resumed_rows is None or old[1] != new[1] else self.resumed_rows
        self.resumed_rows = None
        if old == new:
            return
        self.pacing, (self.pacing_per_row, self.pacing_fraction) = dict(rows=rows, samples=self.samples_seen), new
        dense_config.log_event(self.run, 'learner', 'info', f'{self.settings.variant} pacing base moved to {rows} rows, '
              f'{self.samples_seen} samples: samples_per_row {old[0]} -> {new[0]}, cheap_row_fraction {old[1]} -> {new[1]}',
              variant=self.settings.variant, step=self.step, pacing=self.pacing, old_samples_per_row=old[0],
              new_samples_per_row=new[0], old_cheap_row_fraction=old[1], new_cheap_row_fraction=new[1])

    def lr(self):
        """Linear warmup over warmup_steps from the optimizer's start, then constant."""
        return self.settings.lr*min(1., (self.step-self.optimizer_started+1)/max(1, self.settings.warmup_steps))

    def coefficients(self):
        s = self.settings
        return torch.tensor([1., s.value_weight, s.short_value_weight, s.opponent_policy_weight, s.future_weight, s.outcome_weight],
                            device=self.device)

    @property
    def heads(self):
        return tuple('future_masked_ce' if h == 'future_bce' and self.settings.future_target == 'masked' else h for h in HEADS)

    def targets(self):
        """dense_data.examples() keyword arguments of the current settings and calibration map."""
        return dense_data.target_options(self.settings, self.calibration)

    def calibrate(self, window):
        """Fit the value target map (dense_data.fit_calibration) to the window's newest calibration_games finished
        training games (dense_data.ReplayWindow.finished_games), over full-search root values only with
        bootstrap_full_only. Sets `calibration` (None below dense_data.CALIBRATION_MIN_GAMES games) and
        `calibration_report` (calibration_report)."""
        s = self.settings
        games = window.finished_games(s.calibration_games)
        self.calibration = dense_data.fit_calibration([(r, f if s.bootstrap_full_only else None, w) for r, f, w in games])
        self.calibration_report = calibration_report(self.calibration, len(games))

    def train_step(self, batch):
        """One optimizer step; returns the head losses [len(HEADS)], NaN for a head without target weight."""
        self.model.train()
        lr = self.lr()
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.optimizer.zero_grad(set_to_none=True)
        losses = batch_losses(self.model, batch, self.coefficients(), self.device, self.memory_format, True)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.grad_clip, error_if_nonfinite=True)
        # AdamW decays a whole parameter, including the two inactive channels sharing the opponent head.
        legacy = self.model.aux_spatial.weight[1:] if self.model.future_target == 'masked' and self.model.config.aux_heads else None
        saved = None if legacy is None else legacy.detach().clone()
        self.optimizer.step()
        if legacy is not None:
            with torch.no_grad():
                legacy.copy_(saved)
        self.ema_updates += 1
        update_ema(self.ema, self.model, min(self.settings.ema, (1+self.ema_updates)/(10+self.ema_updates)))
        self.step += 1
        self.samples_seen += sum(len(b['counts']) for b in batch.values())
        mass = torch.stack([sum(b[k].sum() for b in batch.values()) for k in WEIGHTS]).to(self.device)
        return losses.masked_fill(mass == 0, math.nan)

    def recalibrate(self, window):
        """Set the EMA's norm running statistics to their cumulative mean over train-mode passes on
        RECALIBRATION_ROWS training rows of the window, drawn like training batches (recency) with a fixed seed."""
        if not window.index:
            return
        norms = [m for m in self.ema.modules() if isinstance(m, hexnet.MaskedNorm)]
        momenta = [m.momentum for m in norms]
        for m in norms:
            m.reset_running_stats(); m.momentum = None
        self.ema.train()
        rng, s = np.random.default_rng([self.config.seed, 1]), self.settings
        with torch.no_grad():
            for _ in range(math.ceil(RECALIBRATION_ROWS/s.batch)):
                refs = window.sample(rng, s.batch, s.recency, regret_fraction=s.regret_fraction)
                batch = dense_data.collate(*dense_data.examples(window, refs, rng, **self.targets()))
                batch_losses(self.ema, batch, None, self.device, self.memory_format, False)
        for m, momentum in zip(norms, momenta):
            m.momentum = momentum
        self.ema.eval()

    def weighted_means(self, batches):
        """Per head of HEADS, the EMA's weighted mean loss over every row of the collated `batches` with that
        target; None for a head without target weight."""
        total = torch.zeros(len(HEADS), device=self.device); mass = torch.zeros(len(HEADS), device=self.device)
        with torch.no_grad():
            for batch in batches:
                weights = torch.stack([sum(b[w].sum() for b in batch.values()) for w in WEIGHTS]).to(self.device)
                total += batch_losses(self.ema, batch, None, self.device, self.memory_format, False)*weights; mass += weights
        return [float(total[h]/mass[h]) if mass[h] > 0 else None for h in range(len(HEADS))]

    def validate(self, window):
        """EMA weighted_means (eval mode) over VALIDATION_ROWS held-out rows drawn with a fixed sampling seed, by
        head, plus the outcome_split of the same rows; None without held-out rows."""
        if not window.validation:
            return None
        self.ema.eval()
        rng, s = np.random.default_rng(self.config.seed), self.settings
        batches = [dense_data.collate(*dense_data.examples(window, window.sample(rng, s.batch, validation=True), rng, **self.targets()))
                   for _ in range(math.ceil(VALIDATION_ROWS/s.batch))]
        rows, policy_rows, deblundered = [], [], []
        with torch.no_grad():
            for b in (b for batch in batches for b in batch.values()):
                out = forward(self.ema, b['planes'], self.device, self.memory_format)[0]
                logit = out['value_logit'].float().cpu()
                bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, b['outcome'], reduction='none')
                rows.append(np.stack([bce.numpy(), b['outcome'].numpy() != .5, b['exact'].numpy() > 0]))
                mask = b['policy_weight'] > 0
                if mask.any():
                    policy_rows.append(torch.stack(policy_validation_rows(out, b)[1:])[:, mask].numpy())
                if s.deblunder_weight:
                    value_bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, b['value'], reduction='none')
                    deblundered.append(np.stack([value_bce.numpy(), b['deblundered'].numpy()]))
        policy = np.concatenate(policy_rows, 1) if policy_rows else np.empty((3, 0))
        extra = dict(zip(('policy_target_entropy', 'policy_kl', 'policy_top1'),
                         (float(x.mean()) if x.size else None for x in policy)))
        result = dict(zip(self.heads, self.weighted_means(batches))) | outcome_split(*np.concatenate(rows, 1)) | extra
        if s.deblunder_weight:
            result.update(deblunder_split(*np.concatenate(deblundered, 1)))
        return result

    def subset_losses(self, sets, refs):
        """EMA weighted_means (policy_ce, value_bce, future loss) over `refs` under symmetries drawn from a fixed seed
        (a row keeps its symmetry while rows are appended)."""
        s = self.settings
        rng = np.random.default_rng(self.config.seed)
        batches = (dense_data.collate(*dense_data.examples(sets, refs[k:k+s.batch], rng, **self.targets()))
                   for k in range(0, len(refs), s.batch))
        means = self.weighted_means(batches)
        return tuple(means[i] for i in (0, 1, 4))

    def row_losses(self, sets, refs):
        """Per-row EMA losses over `refs` under symmetries drawn from a fixed seed: float arrays with one entry per
        ref, aligned across keys (not in the order of `refs`): ply (from the start), remaining (len(moves) - ply),
        finished (1. when winner >= 0, else 0.), value_bce, value (its target), outcome_bce, outcome (the hard
        outcome; .5 for capped games), policy_ce (against the improved policy; nan on rows without a policy
        target), policy_target_entropy, policy_kl, policy_top1 (nan without a policy target), searched
        (searched_value at the row), proven (the row's `proven`), proof_action (1 with a witness)
        and deblundered (0/1)."""
        s = self.settings
        rng = np.random.default_rng(self.config.seed)
        rows = []
        with torch.no_grad():
            for k in range(0, len(refs), s.batch):
                chunk = refs[k:k+s.batch]
                samples, targets = dense_data.examples(sets, chunk, rng, **self.targets())
                order = sorted(range(len(chunk)), key=lambda i: samples[i].size)    # collate's row order
                losses = []
                for b in dense_data.collate(samples, targets).values():
                    out = forward(self.ema, b['planes'], self.device, self.memory_format)[0]
                    ce, entropy, kl, top1 = policy_validation_rows(out, b)
                    logit = out['value_logit'].float().cpu()
                    bce = [torch.nn.functional.binary_cross_entropy_with_logits(logit, b[k], reduction='none') for k in ('value', 'outcome')]
                    policy = [torch.where(b['policy_weight'] > 0, x, math.nan).tolist() for x in (ce, entropy, kl, top1)]
                    losses += zip(bce[0].tolist(), b['value'].tolist(), bce[1].tolist(), b['outcome'].tolist(), *policy)
                for i, loss in zip(order, losses):
                    ref = chunk[i]
                    e, t = ref.episode, ref.row['ply']
                    rows.append((t, len(e['moves'])-t, float(e['winner'] >= 0), *loss, searched_value(e, t),
                                 float(ref.row.get('proven', 0)), float(bool(ref.row.get('proof_action'))),
                                 targets[i].get('deblundered', 0.)))
        keys = ('ply', 'remaining', 'finished', 'value_bce', 'value', 'outcome_bce', 'outcome', 'policy_ce',
                'policy_target_entropy', 'policy_kl', 'policy_top1', 'searched', 'proven', 'proof_action', 'deblundered')
        return dict(zip(keys, np.array(rows, np.float64).reshape(-1, len(keys)).T))

    def validate_sources(self, sets):
        """Refresh `sets` (dense_data.ValidationSets) and return, per source, <source>_policy_ce and
        <source>_value_bce and <source>_future_bce or <source>_future_masked_ce on its held subset,
        <source>_train_* on its train subset, <source>_gap_* = held minus
        train (None when either is), <source>_rows (held rows), <source>_outcome_bce_exact(_rows) and
        <source>_outcome_bce_unproven(_rows), the outcome_split of the row_losses of the held subset, plus
        newest_checkpoint. For CURVE_SOURCES, over the
        row_losses of the held subset: the remaining_curve of the outcome BCE of the rows of finished games as
        <source>_<key> (grid: remaining_grid); <source>_value_bce_by_ply, the ply_curve of the same rows' outcome
        BCE, and <source>_policy_ce_curve, the ply_curve of the policy CE of rows with a policy target (grid:
        ply_grid); <source>_policy_ce_early and <source>_policy_ce_late, the ply_split of that policy CE. Surfaces
        (surfaces grid dicts, cells compact rows): <source>_value_surface adds `value`, the mean outcome BCE of the
        rows of finished games, and <source>_value_excess_surface adds `excess`, that minus the binary entropy of
        the cell's outcome rate; <source>_policy_surface adds `policy`, the mean policy CE of rows with a policy
        target. <source>_value_regret(_early, _late) is the value_regret of the outcome BCE of the held rows of
        finished games against the calibration_reference fitted on the source's train rows of finished games
        (searched value, plies remaining, outcome for the side to move). <source>_value_regret_proven is the mean of
        1 - p over its held rows with a proof, p the EMA's probability of the proven result (the value target), and
        <source>_proven_rows their count; None without such rows. <source>_policy_ce_proof is policy CE against
        the configured mixed target on winning rows with a witness and a policy target, with its count in
        <source>_policy_ce_proof_rows; None without such rows. These use the fixed held panels."""
        sets.refresh()
        self.ema.eval()
        out = dict(newest_checkpoint=sets.newest_checkpoint)
        for source in dense_data.SOURCES:
            held, train = (self.subset_losses(sets, sets.subsets[source, split]) for split in ('held', 'train'))
            for name, v, w in zip(('policy_ce', 'value_bce', self.heads[4]), held, train):
                out.update({f'{source}_{name}': v, f'{source}_train_{name}': w,
                            f'{source}_gap_{name}': None if v is None or w is None else v-w})
            out[f'{source}_rows'] = len(sets.subsets[source, 'held'])
            r = self.row_losses(sets, sets.subsets[source, 'held'])
            proof = (r['proven'] > 0) & (r['proof_action'] > 0) & np.isfinite(r['policy_ce'])
            out[f'{source}_policy_ce_proof'] = float(r['policy_ce'][proof].mean()) if proof.any() else None
            out[f'{source}_policy_ce_proof_rows'] = int(proof.sum())
            p = np.isfinite(r['policy_ce'])
            for key in ('policy_target_entropy', 'policy_kl', 'policy_top1'):
                out[f'{source}_{key}'] = float(r[key][p].mean()) if p.any() else None
            out.update({f'{source}_{k}': v for k, v in outcome_split(r['outcome_bce'], r['finished'] > 0, r['proven'] != 0).items()})
            if self.settings.deblunder_weight:
                out.update({f'{source}_{k}': v for k, v in deblunder_split(r['value_bce'], r['deblundered']).items()})
            if source not in CURVE_SOURCES:
                continue
            f = r['finished'] > 0
            out.update({f'{source}_{k}': v for k, v in remaining_curve(r['remaining'][f], r['outcome_bce'][f], r['outcome'][f]).items()})
            early, late = ply_split(r['ply'][p], r['policy_ce'][p])
            out.update({f'{source}_value_bce_by_ply': ply_curve(r['ply'][f], r['outcome_bce'][f]),
                        f'{source}_policy_ce_curve': ply_curve(r['ply'][p], r['policy_ce'][p]),
                        f'{source}_policy_ce_early': early, f'{source}_policy_ce_late': late})
            grid, (bce, rate) = surfaces(r['ply'][f], r['remaining'][f], (r['outcome_bce'][f], r['outcome'][f]))
            out[f'{source}_value_surface'] = grid | dict(value=[compact(x) for x in bce])
            out[f'{source}_value_excess_surface'] = grid | dict(excess=[compact(x) for x in bce-binary_entropy(rate)])
            grid, (ce,) = surfaces(r['ply'][p], r['remaining'][p], (r['policy_ce'][p],))
            out[f'{source}_policy_surface'] = grid | dict(policy=[compact(x) for x in ce])
            fit = [(searched_value(e, t), len(e['moves'])-t, float(dense_data.player_at(t) == e['winner']))
                   for e, t in ((ref.episode, ref.row['ply']) for ref in sets.subsets[source, 'train']) if e['winner'] >= 0]
            fit = np.array(fit, np.float64).reshape(-1, 3).T
            reference = calibration_reference(*fit, r['searched'][f], r['remaining'][f])
            out.update({f'{source}_{k}': v for k, v in value_regret(r['outcome_bce'][f], r['outcome'][f], reference, r['remaining'][f]).items()})
            proven = r['proven'] != 0
            out[f'{source}_proven_rows'] = int(proven.sum())
            out[f'{source}_value_regret_proven'] = float(np.mean(1-np.exp(-r['value_bce'][proven]))) if proven.any() else None
        return out | dict(remaining_grid=list(REMAINING_GRID), ply_grid=list(PLY_GRID))

    def export(self, window, sets=None):
        """Write checkpoints/<variant>/<step:06d>/ atomically (staged in a hidden sibling, then renamed).
        The value target map is refitted first (calibrate; metrics.calibration), then the EMA is recalibrated;
        metrics.validation is validate(window) (the HEADS and the outcome_split; null without held-out rows in the window) and
        metrics.validation_sources is validate_sources(sets) (null without `sets`). The cache is released after
        these passes. rows is window.total_rows (the pacing count) and pacing the pacing base (rebase)."""
        s = self.settings
        root = self.run/'checkpoints'/s.variant
        root.mkdir(parents=True, exist_ok=True)
        final, stage = root/f'{self.step:06d}', root/f'.pending-{self.step:06d}'
        if final.exists():
            raise FileExistsError(f'{final} already exists')
        self.calibrate(window)
        self.recalibrate(window)
        validation, sources = self.validate(window), None if sets is None else self.validate_sources(sets)
        self.release()
        shutil.rmtree(stage, ignore_errors=True); stage.mkdir()
        hexnet.save_model(stage/'model.pt', self.model)
        hexnet.save_model(stage/'ema.pt', self.ema)
        torch.save(dict(kind=s.optimizer, optimizer=self.optimizer.state_dict(), optimizer_started=self.optimizer_started,
                        ema_updates=self.ema_updates), stage/'optimizer.pt')
        manifest = dict(variant=s.variant, step=self.step, samples_seen=self.samples_seen, created_at=time.time(),
                        optimizer_kind=s.optimizer,
                        model_sha256=hexnet.model_digest(self.model), ema_sha256=hexnet.model_digest(self.ema),
                        metrics=dict(self.metrics or {h: None for h in self.heads}, validation=validation, validation_sources=sources,
                                     calibration=self.calibration_report),
                        learner=asdict(s), model=asdict(self.config.model), copied_from=self.copied_from,
                        rows=window.total_rows, pacing=self.pacing)
        write_json(stage/'manifest.json', manifest)
        stage.rename(final)
        self.last_export = self.step
        dense_config.log_event(self.run, 'learner', 'export', f'{s.variant} exported step {self.step}', variant=s.variant, step=self.step,
              checkpoint=f'{s.variant}/{self.step:06d}', metrics=manifest['metrics'])
        return manifest

    def maybe_replace(self, factor_rng):
        """Population replacement (exploit/explore). Candidates are the latest rated, not demoted checkpoints of
        other variants (`latest_rated`); a candidate qualifies when league["differences"] holds its pair with
        this variant's latest such checkpoint and the lower bound of the (candidate minus mine) Elo interval
        exceeds replace_margin. The best qualifying candidate's raw weights and manifest learner settings are
        copied (KEEP fields stay this learner's), the continuous settings are perturbed from the copied values,
        the optimizer is reset and the EMA restarts from the copied weights. Checked before a step is trained,
        so a copy is always followed by training and recorded in the next manifest."""
        s = self.settings
        if self.step % s.replace_interval or self.step-self.last_copy < s.protect_steps or not (self.run/'league.json').exists():
            return False
        league = json.loads((self.run/'league.json').read_text(encoding='utf-8'))
        latest = latest_rated(league)
        mine = latest.get(s.variant)
        # Compare only once a checkpoint trained after the last copy has been rated.
        if mine is None or (self.copied_from and mine['step'] <= self.copied_from['at_step']):
            return False
        # (lo, hi, delta) of source minus mine, from either orientation of a difference entry.
        pairs = {}
        for d in league.get('differences', []):
            pairs[d['a'], d['b']] = (d['interval'][0], d['interval'][1], d['elo_delta'])
            pairs[d['b'], d['a']] = (-d['interval'][1], -d['interval'][0], -d['elo_delta'])
        others = [c for v, c in latest.items() if v != s.variant]
        missing = [c['id'] for c in others if (c['id'], mine['id']) not in pairs]
        if missing:
            dense_config.log_event(self.run, 'learner', 'info', f'{s.variant} replacement check at step {self.step}: no Elo difference entry for '
                  f'{", ".join(missing)} against {mine["id"]}', variant=s.variant, step=self.step, missing=missing)
        leaders = [c for c in others if (c['id'], mine['id']) in pairs and pairs[c['id'], mine['id']][0] > s.replace_margin]
        if not leaders:
            return False
        source = max(leaders, key=lambda c: pairs[c['id'], mine['id']][2])
        path = self.run/'checkpoints'/source['id']
        copied = json.loads((path/'manifest.json').read_text(encoding='utf-8'))['learner']
        self.load_weights(path/'model.pt')
        self.ema = copy.deepcopy(self.model)
        base = replace(dense_config.section('learner', copied), **{k: getattr(s, k) for k in KEEP})
        old, self.settings = s, perturb(base, factor_rng, s.perturb)
        self.optimizer = make_optimizer(self.model, self.settings)
        self.optimizer_started = self.last_copy = self.step
        self.ema_updates = 0
        lo, hi, delta = pairs[source['id'], mine['id']]
        self.copied_from = dict(checkpoint=source['id'], at_step=self.step, mine=mine['id'],
                                elo_delta=delta, interval=[lo, hi], source_learner=copied, perturbed=asdict(self.settings))
        dense_config.log_event(self.run, 'learner', 'replace', f'{s.variant} copied {self.copied_from["checkpoint"]} at step {self.step}',
              variant=s.variant, step=self.step, copied_from=self.copied_from, old=asdict(old), source_settings=copied,
              new=asdict(self.settings))
        return True


def main():
    parser = argparse.ArgumentParser(description='Train one dense HexNet variant on a run directory')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--initial', type=Path, help='hexnet checkpoint to warm start from (ignored when resuming)')
    parser.add_argument('--steps', type=int, help='stop once the step count reaches this (default: endless)')
    parser.add_argument('--workers', type=int, default=2, help='render worker processes')
    parser.add_argument('--threads', type=int, default=1, help='torch CPU threads of this process when training on CUDA')
    dense_config.add_arguments(parser, dense_config.LearnerSettings)
    args = parser.parse_args()
    config = dense_config.load(args.run)
    settings = dense_config.override(config.learner, args)
    overrides = {k: v for k, v in asdict(settings).items() if getattr(args, k, None) is not None}
    torch.manual_seed(config.seed)
    learner = Learner(args.run, settings, config, args.initial, overrides)
    if learner.device.type == 'cuda':
        torch.set_num_threads(args.threads)
    s = learner.settings
    status = dict(stage='training', variant=s.variant, error=None, samples_per_second=0.)

    def write_status(**fields):
        base = learner.pacing
        status.update(fields, updated_at=time.time(), step=learner.step, samples_seen=learner.samples_seen,
                      rows_available=window.total_rows, window_rows=window.rows, full_rows_available=window.total_full_rows,
                      window_full_rows=window.full_rows, retained_rows=window.retained_rows, retained_fraction=window.retained_fraction,
                      regret_rows=window.regret_rows, regret_effective_share=window.regret_share(
                          learner.settings.batch, learner.settings.regret_fraction, learner.settings.recency),
                      samples_per_row=(learner.samples_seen-base['samples'])/max(1, window.total_rows-base['rows']),
                      samples_per_row_target=learner.settings.samples_per_row, phase_rows=learner.settings.phase_rows,
                      backlog_rows=backlog(learner.samples_seen, window.total_rows, learner.settings.samples_per_row, base),
                      pacing_rows=base['rows'], pacing_samples=base['samples'],
                      lr=learner.lr(), data_wait_fraction=wait_fraction(rate),
                      last_export_step=learner.last_export, policy_ce=(learner.metrics or {}).get('policy_ce'),
                      value_bce=(learner.metrics or {}).get('value_bce'), vram=learner.vram(),
                      optimizer_state_mb=learner.optimizer_state_mb())
        write_json(status_path(args.run, s.variant), status)

    def speed():
        elapsed = rate[-1][0]-rate[0][0] if rate else 0.
        return sum(r[1] for r in rate[1:])/elapsed if elapsed > 0 else 0.

    def export():
        write_status(stage='exporting')
        fields = validation_fields(learner.export(window, sets)['metrics'])
        window.refresh_regret()
        stream.refresh_regret(window.regret_entries)
        stream.set_calibration(learner.calibration)
        if fields:
            dense_config.append_metrics(args.run, f'learner-{s.variant}', step=learner.step, samples_seen=learner.samples_seen,
                                        validation=True, vram=learner.vram(), proven_rows=window.proven_rows, **fields)

    rate = []
    try:
        torch.manual_seed(config.seed+learner.step)
        variant_seed = zlib.crc32(s.variant.encode())
        def replay():
            s = learner.settings
            return dense_data.ReplayWindow(args.run, s.window_capacity, s.window_min_rows, s.window_expand_per_row,
                                           s.window_taper, s.validation_fraction, policy_dir(args.run, s.variant),
                                           s.cheap_row_fraction, config.seed)
        window = replay()
        sets = validation_sets(args.run, s, config.seed)
        learner.calibrate(window)
        renderers = lambda: dense_data.Renderers(args.run, learner.settings, [config.seed, variant_seed, learner.step], args.workers,
                                                 calibration=learner.calibration, policy_dir=policy_dir(args.run, s.variant),
                                                 regret_entries=window.regret_entries, run_seed=config.seed)
        stream = renderers()
        factor_rng = np.random.default_rng([config.seed, variant_seed, learner.step, 1])
        dense_config.log_event(args.run, 'learner', 'info', f'{s.variant} learner started at step {learner.step}', variant=s.variant, step=learner.step,
              learner=asdict(s))
        print(f'future loss: {learner.heads[4]}', flush=True)
        print(f'{"step":>6} {"policy":>7} {"value":>7} {"short":>7} {"opp":>7} {"future":>7} {"outcome":>7} {"lr":>8} {"rows/s":>7} {"wait":>6} {"gpu":>6} {"mem":>6}', flush=True)
        sums = torch.zeros(len(HEADS), device=learner.device); counts = torch.zeros(len(HEADS), device=learner.device)
        last_status = last_refresh = time.time()
        phase = Phase()
        while args.steps is None or learner.step < args.steps:
            if learner.maybe_replace(factor_rng):
                stream.close(); window = replay(); learner.calibrate(window); stream = renderers(); last_refresh = time.time()
            s = learner.settings
            if time.time()-last_refresh > REFRESH_SECONDS:
                window.refresh(); last_refresh = time.time()
            learner.rebase(window.total_rows)
            limited = paced(learner.samples_seen, window.total_rows, s.samples_per_row, s.batch, learner.pacing)
            if window.index and not phase.due(s.phase_rows, backlog(learner.samples_seen, window.total_rows, s.samples_per_row, learner.pacing), limited):
                write_status(stage='phase-idle', samples_per_second=0.)
                rate = []; time.sleep(5.); window.refresh(); last_refresh = time.time()
                continue
            if not window.index or limited:
                write_status(stage='waiting-for-data', samples_per_second=0.)
                rate = []; time.sleep(5.); window.refresh(); last_refresh = time.time()
                continue
            started = time.perf_counter()
            batch = next(stream)
            ready = time.perf_counter()
            losses = learner.train_step(batch)
            sums += losses.nan_to_num(); counts += losses.isfinite()
            logged = learner.step % s.log_every == 0
            if learner.step % 10 == 0 or logged or learner.step % s.export_every == 0 or learner.step == args.steps:
                learner.metrics = {h: float(v/n) if n else None for h, v, n in zip(learner.heads, sums.tolist(), counts.tolist())}
                sums.zero_(); counts.zero_()
            finished = time.perf_counter()
            rate = (rate+[(finished, s.batch, ready-started, finished-ready)])[-50:]
            if logged:
                dense_config.append_metrics(args.run, f'learner-{s.variant}', step=learner.step, samples_seen=learner.samples_seen,
                                            lr=learner.lr(), **{LOGGED.get(h, h): v for h, v in learner.metrics.items()},
                                            samples_per_second=speed(), window_rows=window.rows, vram=learner.vram())
            if learner.step % 10 == 0:
                wait, gpu = np.mean([r[2] for r in rate]), np.mean([r[3] for r in rate])
                memory = torch.cuda.max_memory_allocated()/2**30 if learner.device.type == 'cuda' else 0.
                print(f'{learner.step:>6} ' + ' '.join(f'{"-":>7}' if v is None else f'{v:>7.4f}' for v in learner.metrics.values())
                      + f' {learner.lr():>8.2e} {s.batch/(wait+gpu):>7.0f} {1000*wait:>6.0f} {1000*gpu:>6.0f} {memory:>5.2f}G', flush=True)
            if time.time()-last_status > STATUS_SECONDS:
                write_status(stage='training', samples_per_second=speed())
                last_status = time.time()
            if learner.step % s.export_every == 0:
                export()
        if learner.last_export != learner.step:
            export()
        write_status(stage='idle', samples_per_second=0.)
    except KeyboardInterrupt:
        if 'window' in locals():
            write_status(stage='idle', samples_per_second=0.)
        raise
    except BaseException as error:
        dense_config.log_event(args.run, 'learner', 'error', f'{s.variant} learner failed: {error!r}', variant=s.variant, step=learner.step)
        if 'window' in locals():
            write_status(stage='failed', error=repr(error))
        raise
    finally:
        if 'stream' in locals():
            stream.close()


if __name__ == '__main__':
    main()
