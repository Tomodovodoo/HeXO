"""Dense learner: trains one HexNet variant on a run's shards and exports EMA checkpoints (run layout: dense_config).

A checkpoint holds model.pt (raw weights) and ema.pt (what actors play), both hexnet.save_model, optimizer.pt and
manifest.json {variant, step, samples_seen, created_at, model_sha256, ema_sha256 (hexnet.model_digest), metrics,
learner (effective LearnerSettings), model (ModelSettings), copied_from}. Events have source 'learner' and kind
export, info, error or replace. league.json is read for population replacement, never written. metrics/learner-
<variant>.jsonl gets a line every log_every steps (losses averaged since the previous averaging point, which also
falls on every tenth step) and one per export with validation_fields(metrics) of the manifest (Learner.export): EMA
losses on held-out rows of the window and on fixed per-source subsets (dense_data.ValidationSets). The EMA
averages parameters only; each export first recomputes its norm statistics (Learner.recalibrate), since the
raw model's running statistics do not describe the EMA weights.

Every target is derived here from episodes (dense_data.examples), so td_lambda, bootstrap_weight and
short_value_horizon are learner settings; batches are rendered by dense_data.Renderers worker processes
(--workers) with random hex symmetries. The loss of one optimizer step is, per head, the weighted mean
over every row of the batch that has that target, summed with the head coefficients; each crop bucket is
a separate forward pass whose gradients accumulate (buckets padded to QUANTUM rows with inert rows).
Pacing: at most samples_per_row * (trained rows in all shards, cheap rows included; a historical opponent's
plies are not trained, see dense_data.trained) samples are presented; beyond that the learner waits. The window
is sized in full-search rows (dense_data.ReplayWindow).
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

HEADS = ('policy_ce', 'value_bce', 'short_value_bce', 'opponent_ce', 'future_bce')
WEIGHTS = ('policy_weight', 'value_weight', 'short_weight', 'next_weight', 'future_weight')
VALIDATION_ROWS = 2048
RECALIBRATION_ROWS = 4096
QUANTUM = 16  # bucket rows are padded to a multiple of this
STATUS_SECONDS = 2.
REFRESH_SECONDS = 30.
# Settings a replacement copy keeps from this learner rather than the source checkpoint's manifest.
KEEP = ('variant', 'protect_steps', 'replace_interval', 'replace_margin', 'validation_fraction', 'export_every', 'log_every')
LOGGED = dict(zip(HEADS, ('policy_ce', 'value_bce', 'short_value_bce', 'next_ce', 'future_bce')))  # metrics log names
# (low, high) for replacement perturbations; td_lambda and ema are perturbed through 1 - x.
BOUNDS = dict(lr=(1e-5, 3e-3), weight_decay=(1e-5, 1e-1), bootstrap_weight=(0., 1.), td_lambda=(0., .995), ema=(.99, .9999))


def validation_fields(metrics):
    """The metrics-log fields of an export's manifest metrics: metrics.validation under LOGGED names plus
    metrics.validation_sources; None when both are null."""
    fields = {LOGGED[h]: v for h, v in (metrics['validation'] or {}).items()} | (metrics.get('validation_sources') or {})
    return fields or None


def status_path(run, variant):
    return run/('learner-status.json' if variant == 'main' else f'learner-status-{variant}.json')


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


def head_losses(model, batch, device, memory_format):
    """Per-head weighted means over one bucket and the bucket's weight sums, both [5] on device."""
    b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    planes = b['planes'].float().contiguous(memory_format=memory_format)
    future = b['future'].float()
    mask = planes[:, 3:4]
    with torch.autocast(device.type, torch.bfloat16, enabled=device.type == 'cuda'):
        out = model(planes, mask)
    target = torch.zeros(b['mask'].shape, device=device).masked_scatter_(b['mask'], b['policy'])
    losses = [hexnet.policy_loss(out['policy'], out['far'], b['cells'], b['counts'], target, b['policy_weight']),
              hexnet.value_loss(out['value_logit'], b['value'], b['value_weight'])]
    if model.config.aux_heads:
        losses += [hexnet.short_value_loss(out['short_value_logit'], b['short_value'], b['short_weight']),
                   hexnet.opponent_policy_loss(out['opponent_policy'], b['next_cells'], b['next_counts'],
                                               b['next_policy'], b['next_weight']),
                   hexnet.future_loss(out['future'], future, mask, b['future_weight'])]
    else:
        losses += [torch.zeros((), device=device)]*3
    return torch.stack(losses), torch.stack([b[k].sum() for k in WEIGHTS])


def batch_losses(model, batch, coefficients, device, memory_format, train):
    """Head losses [5] of a collated {S: bucket} batch; with `train`, backpropagates the combined loss
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


def make_optimizer(model, s):
    """AdamW with weight decay on conv/linear weights only (no decay on norm scales and biases)."""
    groups = [dict(params=[p for p in model.parameters() if p.ndim > 1], weight_decay=s.weight_decay),
              dict(params=[p for p in model.parameters() if p.ndim <= 1], weight_decay=0.)]
    return torch.optim.AdamW(groups, lr=s.lr, betas=(.9, .98), fused=next(model.parameters()).is_cuda)


@torch.no_grad()
def update_ema(ema, model, decay):
    torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1-decay)


def perturb(settings, factor_rng, amount):
    """Multiply lr, weight_decay, bootstrap_weight, 1-td_lambda and 1-ema by factors in [1-amount, 1+amount],
    clipped to BOUNDS."""
    values = {}
    for name, (low, high) in BOUNDS.items():
        x, f = getattr(settings, name), factor_rng.uniform(1-amount, 1+amount)
        values[name] = float(np.clip(1-(1-x)*f if name in ('td_lambda', 'ema') else x*f, low, high))
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
        `overrides`), else start from `initial` or random weights."""
        self.run, self.settings, self.config, self.overrides = run, settings, config, overrides or {}
        self.device = torch.device(config.device)
        self.memory_format = hexnet.memory_format(config.model)
        self.model = self.place(hexnet.HexNet(hexnet.HexNetConfig(**asdict(config.model))))
        self.step = self.samples_seen = self.optimizer_started = self.ema_updates = 0
        self.copied_from = None
        saved = checkpoints(run, settings.variant)
        if saved:
            self.resume(saved[-1])
        else:
            if initial:
                self.load_weights(initial)
            self.ema = copy.deepcopy(self.model)
            self.optimizer = make_optimizer(self.model, self.settings)
        self.start_step = self.last_copy = self.step
        self.last_export = self.step if saved else None
        self.metrics = None

    def place(self, model):
        return model.to(self.device, memory_format=self.memory_format)

    def load_weights(self, path):
        source = hexnet.load_model(path)
        if source.config != self.model.config:
            raise ValueError(f'{path} has model {source.config}, the run uses {self.model.config}')
        self.model.load_state_dict(source.state_dict())

    def resume(self, path):
        manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        self.model = self.place(hexnet.load_model(path/'model.pt'))
        self.ema = self.place(hexnet.load_model(path/'ema.pt'))
        if self.model.config != hexnet.HexNetConfig(**asdict(self.config.model)):
            raise ValueError(f'{path} does not match the run model settings')
        state = torch.load(path/'optimizer.pt', map_location=self.device, weights_only=True)
        # Settings saved by the last export (including replacement perturbations) under explicit CLI overrides.
        self.settings = replace(dense_config.LearnerSettings(**manifest['learner']), **self.overrides)
        self.optimizer = make_optimizer(self.model, self.settings)
        self.optimizer.load_state_dict(state['optimizer'])
        for group, decay in zip(self.optimizer.param_groups, (self.settings.weight_decay, 0.)):
            group['weight_decay'] = decay
        self.step, self.samples_seen = manifest['step'], manifest['samples_seen']
        self.optimizer_started, self.ema_updates = state['optimizer_started'], state['ema_updates']
        self.copied_from = manifest.get('copied_from')
        dense_config.log_event(self.run, 'learner', 'info', f'{self.settings.variant} resumed from step {self.step}', variant=self.settings.variant, step=self.step)

    def lr(self):
        """Linear warmup over warmup_steps from the optimizer's start, then constant."""
        return self.settings.lr*min(1., (self.step-self.optimizer_started+1)/max(1, self.settings.warmup_steps))

    def coefficients(self):
        s = self.settings
        return torch.tensor([1., s.value_weight, s.short_value_weight, s.opponent_policy_weight, s.future_weight], device=self.device)

    def train_step(self, batch):
        self.model.train()
        lr = self.lr()
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.optimizer.zero_grad(set_to_none=True)
        losses = batch_losses(self.model, batch, self.coefficients(), self.device, self.memory_format, True)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.grad_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.ema_updates += 1
        update_ema(self.ema, self.model, min(self.settings.ema, (1+self.ema_updates)/(10+self.ema_updates)))
        self.step += 1
        self.samples_seen += sum(len(b['counts']) for b in batch.values())
        return losses

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
                refs = window.sample(rng, s.batch, s.recency)
                batch = dense_data.collate(*dense_data.examples(window, refs, rng, **dense_data.target_options(s)))
                batch_losses(self.ema, batch, None, self.device, self.memory_format, False)
        for m, momentum in zip(norms, momenta):
            m.momentum = momentum
        self.ema.eval()

    def validate(self, window):
        """EMA losses (eval mode) on VALIDATION_ROWS held-out rows with a fixed sampling seed; None without held-out rows."""
        if not window.validation:
            return None
        self.ema.eval()
        rng = np.random.default_rng(self.config.seed)
        total = torch.zeros(len(HEADS), device=self.device); n = 0
        with torch.no_grad():
            for _ in range(math.ceil(VALIDATION_ROWS/self.settings.batch)):
                s = self.settings
                refs = window.sample(rng, s.batch, validation=True)
                batch = dense_data.collate(*dense_data.examples(window, refs, rng, **dense_data.target_options(s)))
                total += batch_losses(self.ema, batch, None, self.device, self.memory_format, False); n += 1
        return dict(zip(HEADS, (total/n).tolist()))

    def subset_losses(self, sets, refs):
        """EMA (policy_ce, value_bce) over `refs` of `sets`, each the weighted mean over every row with that
        target, under symmetries drawn from a fixed seed (a row keeps its symmetry while rows are appended);
        None for a head without target weight."""
        s = self.settings
        rng = np.random.default_rng(self.config.seed)
        total = torch.zeros(len(HEADS), device=self.device); mass = torch.zeros(len(HEADS), device=self.device)
        with torch.no_grad():
            for k in range(0, len(refs), s.batch):
                batch = dense_data.collate(*dense_data.examples(sets, refs[k:k+s.batch], rng, **dense_data.target_options(s)))
                weights = torch.stack([sum(b[w].sum() for b in batch.values()) for w in WEIGHTS]).to(self.device)
                total += batch_losses(self.ema, batch, None, self.device, self.memory_format, False)*weights; mass += weights
        return tuple(float(total[h]/mass[h]) if mass[h] > 0 else None for h in (0, 1))

    def validate_sources(self, sets):
        """Refresh `sets` (dense_data.ValidationSets) and return, per source, <source>_policy_ce and
        <source>_value_bce on its held subset, <source>_train_* on its train subset, <source>_gap_* = held minus
        train (None when either is), <source>_rows (held rows), plus newest_checkpoint."""
        sets.refresh()
        self.ema.eval()
        out = dict(newest_checkpoint=sets.newest_checkpoint)
        for source in dense_data.SOURCES:
            held, train = (self.subset_losses(sets, sets.subsets[source, split]) for split in ('held', 'train'))
            for name, v, w in zip(('policy_ce', 'value_bce'), held, train):
                out.update({f'{source}_{name}': v, f'{source}_train_{name}': w,
                            f'{source}_gap_{name}': None if v is None or w is None else v-w})
            out[f'{source}_rows'] = len(sets.subsets[source, 'held'])
        return out

    def export(self, window, sets=None):
        """Write checkpoints/<variant>/<step:06d>/ atomically (staged in a hidden sibling, then renamed).
        The EMA is recalibrated first; metrics.validation is validate(window) (the HEADS; null without held-out rows in the window) and
        metrics.validation_sources is validate_sources(sets) (null without `sets`)."""
        s = self.settings
        root = self.run/'checkpoints'/s.variant
        root.mkdir(parents=True, exist_ok=True)
        final, stage = root/f'{self.step:06d}', root/f'.pending-{self.step:06d}'
        if final.exists():
            raise FileExistsError(f'{final} already exists')
        self.recalibrate(window)
        validation, sources = self.validate(window), None if sets is None else self.validate_sources(sets)
        shutil.rmtree(stage, ignore_errors=True); stage.mkdir()
        hexnet.save_model(stage/'model.pt', self.model)
        hexnet.save_model(stage/'ema.pt', self.ema)
        torch.save(dict(optimizer=self.optimizer.state_dict(), optimizer_started=self.optimizer_started,
                        ema_updates=self.ema_updates), stage/'optimizer.pt')
        manifest = dict(variant=s.variant, step=self.step, samples_seen=self.samples_seen, created_at=time.time(),
                        model_sha256=hexnet.model_digest(self.model), ema_sha256=hexnet.model_digest(self.ema),
                        metrics=dict(self.metrics or {h: None for h in HEADS}, validation=validation, validation_sources=sources),
                        learner=asdict(s), model=asdict(self.config.model), copied_from=self.copied_from)
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
        base = replace(dense_config.LearnerSettings(**copied), **{k: getattr(s, k) for k in KEEP})
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
    dense_config.add_arguments(parser, dense_config.LearnerSettings)
    args = parser.parse_args()
    config = dense_config.load(args.run)
    settings = dense_config.override(config.learner, args)
    overrides = {k: v for k, v in asdict(settings).items() if getattr(args, k, None) is not None}
    torch.manual_seed(config.seed)
    learner = Learner(args.run, settings, config, args.initial, overrides)
    s = learner.settings
    status = dict(stage='training', variant=s.variant, error=None, samples_per_second=0.)

    def write_status(**fields):
        status.update(fields, updated_at=time.time(), step=learner.step, samples_seen=learner.samples_seen,
                      rows_available=window.total_rows, window_rows=window.rows, full_rows_available=window.total_full_rows,
                      window_full_rows=window.full_rows,
                      samples_per_row=learner.samples_seen/max(1, window.total_rows),
                      samples_per_row_target=learner.settings.samples_per_row, lr=learner.lr(),
                      last_export_step=learner.last_export, policy_ce=(learner.metrics or {}).get('policy_ce'),
                      value_bce=(learner.metrics or {}).get('value_bce'))
        write_json(status_path(args.run, s.variant), status)

    def speed():
        elapsed = rate[-1][0]-rate[0][0] if rate else 0.
        return sum(r[1] for r in rate[1:])/elapsed if elapsed > 0 else 0.

    def export():
        write_status(stage='exporting')
        fields = validation_fields(learner.export(window, sets)['metrics'])
        if fields:
            dense_config.append_metrics(args.run, f'learner-{s.variant}', step=learner.step, samples_seen=learner.samples_seen,
                                        validation=True, **fields)

    rate = []
    try:
        torch.manual_seed(config.seed+learner.step)
        variant_seed = zlib.crc32(s.variant.encode())
        def replay():
            s = learner.settings
            return dense_data.ReplayWindow(args.run, s.window_capacity, s.window_min_rows, s.window_expand_per_row,
                                           s.window_taper, s.validation_fraction, s.policy_cache_mb)
        window = replay()
        sets = dense_data.ValidationSets(args.run, s.validation_fraction, config.seed)
        renderers = lambda: dense_data.Renderers(args.run, learner.settings, [config.seed, variant_seed, learner.step], args.workers)
        stream = renderers()
        factor_rng = np.random.default_rng([config.seed, variant_seed, learner.step, 1])
        dense_config.log_event(args.run, 'learner', 'info', f'{s.variant} learner started at step {learner.step}', variant=s.variant, step=learner.step,
              learner=asdict(s))
        print(f'{"step":>6} {"policy":>7} {"value":>7} {"short":>7} {"opp":>7} {"future":>7} {"lr":>8} {"rows/s":>7} {"wait":>6} {"gpu":>6} {"mem":>6}', flush=True)
        sums = torch.zeros(len(HEADS), device=learner.device); count = 0
        last_status = last_refresh = time.time()
        while args.steps is None or learner.step < args.steps:
            if learner.maybe_replace(factor_rng):
                stream.close(); stream = renderers(); window = replay(); last_refresh = time.time()
            s = learner.settings
            if time.time()-last_refresh > REFRESH_SECONDS:
                window.refresh(); last_refresh = time.time()
            if not window.index or learner.samples_seen+s.batch > s.samples_per_row*window.total_rows:
                write_status(stage='waiting-for-data', samples_per_second=0.)
                rate = []; time.sleep(5.); window.refresh(); last_refresh = time.time()
                continue
            started = time.perf_counter()
            batch = next(stream)
            ready = time.perf_counter()
            losses = learner.train_step(batch)
            sums += losses; count += 1
            logged = learner.step % s.log_every == 0
            if learner.step % 10 == 0 or logged or learner.step % s.export_every == 0 or learner.step == args.steps:
                values = (sums/count).tolist()
                learner.metrics = dict(zip(HEADS, values))
                sums.zero_(); count = 0
            finished = time.perf_counter()
            rate = (rate+[(finished, s.batch, ready-started, finished-ready)])[-50:]
            if logged:
                dense_config.append_metrics(args.run, f'learner-{s.variant}', step=learner.step, samples_seen=learner.samples_seen,
                                            lr=learner.lr(), **{LOGGED[h]: v for h, v in learner.metrics.items()},
                                            samples_per_second=speed(), window_rows=window.rows)
            if learner.step % 10 == 0:
                wait, gpu = np.mean([r[2] for r in rate]), np.mean([r[3] for r in rate])
                memory = torch.cuda.max_memory_allocated()/2**30 if learner.device.type == 'cuda' else 0.
                print(f'{learner.step:>6} ' + ' '.join(f'{v:>7.4f}' for v in learner.metrics.values())
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
