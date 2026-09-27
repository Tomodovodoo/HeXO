"""Integration check for the dense HexNet stack; run before every training launch and after every change.

  python dense_check.py [--run DIR] [--shards runs/dense-v1/shards] [--model ckpt.pt]

Stages: CPU unit tests (tests/test_dense.py); GPU throughput of the default b6c96 model (inference and
training at batch 256 on bucket 32) against thresholds; bf16 vs fp32 agreement on real positions;
DenseEvaluator behind a SearchCoordinator (32 trees, 2 placements); symmetry invariance of a pointwise
model plus the real model's symmetry spread (diagnostic); and, once dense_selfplay.py, dense_learn.py and
dense_eval.py exist, an end-to-end smoke run in a scratch run directory. Prints a table and exits 1 on
any failure. Without --run a scratch run with a tiny configuration is created in the temp directory.
"""
import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np
import torch

import dense_config
import dense_data
import hexcrop
import hexnet
from neural_search import EvaluationCache, NeuralSearch, SearchCoordinator

ROOT = Path(__file__).resolve().parent
DRIVERS = ('dense_selfplay.py', 'dense_learn.py', 'dense_eval.py')
TINY_MODEL = dense_config.ModelSettings(blocks=2, channels=32, pool_every=2, line_length=5, value_hidden=32, head_channels=16)


class Report:
    def __init__(self):
        self.rows = []

    def add(self, stage, check, value, limit='', status='ok'):
        self.rows.append(dict(stage=stage, check=check, value=value, limit=limit, status=status))
        print(f'  [{status}] {stage}: {check} = {value}' + (f' (limit {limit})' if limit != '' else ''), flush=True)

    def gate(self, stage, check, value, limit, ok, fmt='{:.0f}'):
        self.add(stage, check, fmt.format(value), limit, 'PASS' if ok else 'FAIL')

    @property
    def failed(self):
        return any(r['status'] == 'FAIL' for r in self.rows)

    def table(self):
        widths = [max(len(str(r[k])) for r in self.rows+[dict(stage='stage', check='check', value='value', limit='limit',
                                                                  status='status')]) for k in ('stage', 'check', 'value', 'limit', 'status')]
        line = lambda r: '  '.join(str(v).ljust(w) for v, w in zip((r['stage'], r['check'], r['value'], r['limit'], r['status']), widths))
        print()
        print(line(dict(stage='stage', check='check', value='value', limit='limit', status='status')))
        print('  '.join('-'*w for w in widths))
        for r in self.rows:
            print(line(r))


def scratch_run(run):
    """Create (or reuse) a run directory holding a tiny configuration for the smoke stage."""
    run = Path(run) if run else Path(tempfile.mkdtemp(prefix='dense-check-'))
    if not (run/'config.json').exists():
        config = dense_config.RunConfig(
            created_at=time.time(), device='cuda' if torch.cuda.is_available() else 'cpu', model=TINY_MODEL,
            actor=dense_config.ActorSettings(games_in_flight=4, leaf_batch=32, full_sims=8, cheap_sims=4, root_samples=4,
                                             max_plies=24, shard_games=4, cache_positions=256),
            learner=dense_config.LearnerSettings(batch=16, warmup_steps=5, window_min_rows=64, export_every=10,
                                                 validation_fraction=.25, protect_steps=10, replace_interval=10),
            evaluation=dense_config.EvaluationSettings(games=2, sims=8, root_samples=4, max_plies=24, anchor_every=0,
                                                       anchor_games=0, seal_ms=10))
        dense_config.save(run, config)
    return run


def unit_stage(report):
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_dense')
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    bad = len(result.failures)+len(result.errors)
    report.add('cpu', 'unit tests (tests/test_dense.py)', f'{result.testsRun} run, {bad} failed in {time.perf_counter()-start:.1f}s',
               '', 'PASS' if bad == 0 and result.testsRun else 'FAIL')


def real_histories(shards, count, rng):
    """`count` distinct positions (histories before a recorded ply) sampled from shard rows."""
    paths = sorted(p for p in Path(shards).iterdir() if p.is_dir() and p.name.isdigit()) if Path(shards).exists() else []
    if not paths:
        raise FileNotFoundError(f'No dense shards under {shards}')
    seen, out = set(), []
    for path in rng.permutation(len(paths)):
        episodes = json.loads((paths[path]/'episodes.json').read_text(encoding='utf-8'))
        rows = json.loads((paths[path]/'rows.json').read_text(encoding='utf-8'))
        for i in rng.permutation(len(rows))[:max(1, count//4)]:
            row = rows[i]
            history = [tuple(m) for m in episodes[row['game']]['moves'][:row['ply']]]
            key = tuple(history)
            if key not in seen:
                seen.add(key)
                out.append(history)
        if len(out) >= 4*count:
            break
    return [out[i] for i in rng.permutation(len(out))]


def bucket_samples(histories, size, count):
    samples = []
    for h in histories:
        s = hexcrop.encode(h)
        if s.size == size:
            samples.append(s)
            if len(samples) == count:
                break
    if not samples:
        raise ValueError(f'No bucket-{size} positions found')
    while len(samples) < count:
        samples += samples[:count-len(samples)]
    return samples


def timed(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return time.perf_counter()-start


def memory_format(model):
    return torch.contiguous_format if model.config.line_length else torch.channels_last


def inference_stage(report, model, samples, args):
    device = torch.device('cuda')
    net = copy.deepcopy(model).requires_grad_(False).to(device, memory_format=memory_format(model)).eval()
    b = hexcrop.batch(samples)
    x = torch.from_numpy(b['planes']).to(device).to(dtype=torch.bfloat16, memory_format=memory_format(model))

    @torch.inference_mode()
    def step():
        with torch.autocast('cuda', torch.bfloat16):
            net(x, x[:, 3:4], aux=False)
    rate = len(samples)*args.iterations/timed(step, 5, args.iterations)
    report.gate('gpu', f'inference pos/s (batch {len(samples)}, bucket {b["size"]})', rate, args.min_inference,
                rate >= args.min_inference)


def training_stage(report, model, samples, args):
    device = torch.device('cuda')
    fmt = memory_format(model)
    net = copy.deepcopy(model).to(device, memory_format=fmt).train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5, weight_decay=1e-4)
    b = hexcrop.batch(samples)
    planes = torch.from_numpy(b['planes']).to(device).float().contiguous(memory_format=fmt)
    cells, counts = torch.from_numpy(b['cells']).to(device), torch.from_numpy(b['counts']).to(device)
    valid = (torch.arange(cells.shape[1], device=device) < counts[:, None]).float()
    target = valid/counts[:, None]
    half = torch.full((len(samples),), .5, device=device)
    future = torch.zeros(len(samples), 2, b['size'], b['size'], device=device)

    def step():
        with torch.autocast('cuda', torch.bfloat16):
            out = net(planes, planes[:, 3:4])
        loss = hexnet.policy_loss(out['policy'], out['far'], cells, counts, target) + hexnet.value_loss(out['value_logit'], half)
        if 'future' in out:
            loss = loss + hexnet.short_value_loss(out['short_value_logit'], half) \
                + hexnet.opponent_policy_loss(out['opponent_policy'], cells, counts, target) \
                + hexnet.future_loss(out['future'], future, planes[:, 3:4])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    rate = len(samples)*args.iterations/timed(step, 3, args.iterations)
    report.gate('gpu', f'training pos/s (batch {len(samples)}, bucket {b["size"]})', rate, args.min_training,
                rate >= args.min_training)
    report.add('gpu', 'training peak memory', f'{(torch.cuda.max_memory_allocated()-base)/2**30:.2f} GiB', '', 'info')
    del net, optimizer
    torch.cuda.empty_cache()


def legal_distribution(out, samples, device):
    b = hexcrop.batch(samples)
    logits, valid = hexnet.action_logits(out['policy'], out['far'], torch.from_numpy(b['cells']).to(device),
                                         torch.from_numpy(b['counts']).to(device))
    return logits.log_softmax(1).masked_fill(~valid, -torch.inf), valid


def precision_stage(report, model, histories, args):
    device = torch.device('cuda')
    fmt = memory_format(model)
    fp32 = copy.deepcopy(model).requires_grad_(False).to(device, memory_format=fmt).eval()
    samples = [hexcrop.encode(h) for h in histories]
    agree = total = 0
    kls, values, margins, flips = [], [], [], []
    with torch.inference_mode():
        for size, indices in hexcrop.group_by_size(samples).items():
            group = [samples[i] for i in indices]
            x = torch.from_numpy(hexcrop.batch(group)['planes']).to(device).contiguous(memory_format=fmt)
            reference = fp32(x.float(), x[:, 3:4].float(), aux=False)
            with torch.autocast('cuda', torch.bfloat16):
                low = fp32(x.to(torch.bfloat16), x[:, 3:4].to(torch.bfloat16), aux=False)
            p, valid = legal_distribution(reference, group, device)
            q, _ = legal_distribution(low, group, device)
            top = p.argmax(1)
            same = top == q.argmax(1)
            agree += int(same.sum()); total += len(group)
            kls.append(torch.where(valid, p.exp()*(p-q), 0).sum(1))
            values.append((torch.tanh(reference['value_logit']/2)-torch.tanh(low['value_logit']/2)).abs())
            second = p.masked_fill(torch.nn.functional.one_hot(top, p.shape[1]).bool(), -torch.inf).amax(1)
            margins.append(p.gather(1, top[:, None])[:, 0]-second)
            flips.append(~same)
    kl, value = torch.cat(kls), torch.cat(values)
    margin, flip = torch.cat(margins), torch.cat(flips)
    rate = agree/total
    report.gate('bf16', f'top-1 agreement vs fp32 ({total} real positions)', rate, args.min_agreement,
                rate >= args.min_agreement, '{:.3f}')
    report.gate('bf16', 'mean KL(fp32 || bf16)', float(kl.mean()), args.max_kl, float(kl.mean()) <= args.max_kl, '{:.2e}')
    report.add('bf16', 'max |q fp32 - q bf16|', f'{float(value.max()):.4f}', '', 'info')
    if flip.any():
        report.add('bf16', 'largest fp32 top-1 margin among flips', f'{float(margin[flip].max()):.4f} nats', '', 'info')


def search_stage(report, model, histories, args):
    evaluator = hexnet.DenseEvaluator(model, device='cuda')
    actor = dense_config.ActorSettings()
    searches = [NeuralSearch(evaluator, evaluator.model_version, history=h, seed=i, tactics=actor.tactics)
                for i, h in enumerate(histories[:args.trees])]
    coordinator = SearchCoordinator(evaluator, evaluator.model_version, EvaluationCache(actor.cache_positions))
    batches = unique = largest = placements = 0
    evaluator.evaluate(histories[:8])                           # warm the CUDA kernels
    start = time.perf_counter()
    try:
        for _ in range(2):
            results = coordinator.search_many(searches, actor.full_sims, actor.root_samples, actor.leaf_batch)
            stats = coordinator.last_stats
            batches += stats['inference_batches']; unique += stats['unique_positions']
            largest = max(largest, stats['largest_batch'])
            for search, result in zip(searches, results):
                if result['action'] is not None:
                    search.advance(result['action'])
                    placements += 1
    finally:
        for search in searches:
            search.close()
    elapsed = time.perf_counter()-start
    ok = placements == 2*len(searches)
    report.add('search', f'{len(searches)} trees x 2 placements ({actor.full_sims} sims, m={actor.root_samples})',
               f'{placements/elapsed:.1f} placements/s', '', 'PASS' if ok else 'FAIL')
    report.add('search', 'mean / largest inference batch', f'{unique/max(1, batches):.1f} / {largest}', '', 'info')


def symmetry_stage(report, model, histories):
    from tests.test_dense import pointwise_model
    device = torch.device('cuda')
    constant = pointwise_model(model.config).to(device)
    evaluator = hexnet.DenseEvaluator(model, device='cuda')
    worst, spreads = 0., []
    with torch.inference_mode():
        for h in histories:
            # Symmetries may land in different buckets, so evaluate them one at a time.
            values, q = [], []
            for k in range(12):
                x = torch.from_numpy(hexcrop.encode(h, symmetry=k).planes[None]).to(device)
                values.append(float(constant(x.float(), x[:, 3:4].float(), aux=False)['value_logit']))
                x = x.to(torch.bfloat16).contiguous(memory_format=evaluator.memory_format)
                with torch.autocast('cuda', torch.bfloat16):
                    q.append(float(torch.tanh(evaluator.model(x, x[:, 3:4], aux=False)['value_logit']/2)))
            worst = max(worst, max(values)-min(values))
            spreads.append(max(q)-min(q))
    report.gate('symmetry', f'pointwise-model value spread over 12 symmetries ({len(histories)} positions)',
                worst, 1e-4, worst <= 1e-4, '{:.2e}')
    report.add('symmetry', 'real-model q spread over 12 symmetries (mean / max)',
               f'{np.mean(spreads):.4f} / {np.max(spreads):.4f}', '', 'info')


def e2e_smoke(report, run):
    """TODO(dense_check e2e): fill in once dense_selfplay.py, dense_learn.py and dense_eval.py are final.

    Planned steps, all inside the scratch `run` (tiny config from scratch_run):
      1. one actor process: python dense_selfplay.py --run RUN --processes 1 --games 4  -> exactly 1 shard of 4 games
         (dense_data.read_shard verifies; counts.games == 4; every row replays via dense_data.examples);
      2. learner: python dense_learn.py --run RUN --steps 20 (export_every <= 20) -> checkpoints/<variant>/<step>/
         with model.pt, ema.pt, manifest.json whose 'learner' equals the effective LearnerSettings and whose
         model_sha256/ema_sha256 equal hexnet.model_digest of the saved files;
      3. evaluator: python dense_eval.py --run RUN loop --once -> league.json and champion.json exist and parse.
    Return normally after adding PASS/FAIL rows to `report`.
    """
    raise NotImplementedError


def e2e_stage(report, run):
    missing = [name for name in DRIVERS if not (ROOT/name).exists()]
    if missing:
        print(f'  drivers not present, skipped ({", ".join(missing)} missing)')
        report.add('e2e', 'smoke run', 'drivers not present, skipped', '', 'skip')
        return
    try:
        e2e_smoke(report, run)
    except NotImplementedError:
        print('  drivers present but e2e_smoke is not implemented yet (see dense_check.e2e_smoke)')
        report.add('e2e', 'smoke run', 'e2e_smoke not implemented', '', 'TODO')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', help='scratch run directory (created with a tiny config when missing)')
    parser.add_argument('--shards', type=Path, default=ROOT/'runs'/'dense-v1'/'shards', help='real positions')
    parser.add_argument('--model', type=Path, help='hexnet checkpoint (default: randomly initialised b6c96)')
    parser.add_argument('--min-inference', type=float, default=3000.)
    parser.add_argument('--min-training', type=float, default=900.)
    parser.add_argument('--min-agreement', type=float, default=.95)
    parser.add_argument('--max-kl', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=256, help='inference batch (thresholds assume 256)')
    parser.add_argument('--train-batch', type=int, default=256,
                        help='training batch; 256 peaks near 2.7 GiB, use 128 (~1.4 GiB) on a crowded GPU and scale the limit')
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--trees', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1740)
    parser.add_argument('--skip-unit', action='store_true')
    args = parser.parse_args()
    report = Report()
    run = scratch_run(args.run)
    print(f'run: {run}')
    if not args.skip_unit:
        print('== CPU unit tests')
        unit_stage(report)
    print('== GPU')
    if not torch.cuda.is_available():
        print('  CUDA unavailable: GPU stages skipped')
        report.add('gpu', 'all GPU stages', 'CUDA unavailable, skipped', '', 'skip')
    else:
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        model = hexnet.load_model(args.model) if args.model else hexnet.HexNet(hexnet.HexNetConfig())
        report.add('gpu', 'model', f'{args.model or "random init"} {asdict(model.config)["blocks"]}b'
                   f'{model.config.channels}c line {model.config.line_length}', '', 'info')
        histories = real_histories(args.shards, max(args.batch, 256), rng)
        samples = bucket_samples(histories, 32, max(args.batch, args.train_batch))
        for stage in (lambda: inference_stage(report, model, samples[:args.batch], args),
                      lambda: training_stage(report, model, samples[:args.train_batch], args),
                      lambda: precision_stage(report, model, histories[:256], args),
                      lambda: search_stage(report, model, histories, args),
                      lambda: symmetry_stage(report, model, histories[:32])):
            try:
                stage()
            except Exception as error:   # report and continue with the other stages
                report.add('gpu', getattr(error, '__qualname__', type(error).__name__), str(error)[:80], '', 'FAIL')
                import traceback
                traceback.print_exc()
        report.add('gpu', 'peak allocated (whole process)', f'{torch.cuda.max_memory_allocated()/2**30:.2f} GiB', '', 'info')
    print('== end-to-end')
    e2e_stage(report, run)
    report.table()
    out = run/'dense_check.json'
    pending = out.with_name(out.name+'.tmp')
    pending.write_text(json.dumps(dict(created_at=time.time(), args={k: str(v) for k, v in vars(args).items()},
                                       rows=report.rows, failed=report.failed), indent=2), encoding='utf-8')
    os.replace(pending, out)
    print(f'\n{"FAILED" if report.failed else "OK"}; report: {out}')
    sys.exit(1 if report.failed else 0)


if __name__ == '__main__':
    main()
