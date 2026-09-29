"""Benchmark real HexNet learner steps from a read-only checkpoint and replay snapshot.

First render consecutive window batches once, or reuse an existing snapshot::

    python tools/profile_learner.py prepare --run runs/dense-v1 --checkpoint runs/dense-v1/checkpoints/main/085000
    python tools/profile_learner.py measure --checkpoint runs/dense-v1/checkpoints/main/085000
    python tools/profile_learner.py measure --checkpoint runs/dense-v1/checkpoints/main/085000 --modes fused --profile

The GPU command runs in one 55-second child with a 60-second cooldown and a
12% allocator cap by default. Profile one mode per invocation. ``--memory-mib 3328`` requires that much free
GPU memory plus 384 MiB before launch. It never stops another process.
"""

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zlib

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.profile_hexnet import guard, trace_summary


def prepare(args):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    import dense_config
    import dense_data

    run, checkpoint, target = args.run.resolve(), args.checkpoint.resolve(), args.batches
    if target.exists():
        raise FileExistsError(f'Batch snapshot already exists: {target}')
    target.mkdir(parents=True)
    manifest = json.loads((checkpoint/'manifest.json').read_text(encoding='utf-8'))
    config = dense_config.load(run)
    settings = dense_config.section('learner', manifest['learner'])
    window = dense_data.ReplayWindow(
        run, settings.window_capacity, settings.window_min_rows,
        settings.window_expand_per_row, settings.window_taper,
        settings.validation_fraction, target/'policy-cache',
        settings.cheap_row_fraction, config.seed)
    games = window.finished_games(settings.calibration_games)
    calibration = dense_data.fit_calibration([
        (roots, full if settings.bootstrap_full_only else None, winner)
        for roots, full, winner in games])
    rng_seed = [config.seed, zlib.crc32(settings.variant.encode()), manifest['step'], 0]
    rng = np.random.default_rng(rng_seed)
    status = dict(state='rendering', run=str(run), checkpoint=str(checkpoint),
                  checkpoint_step=manifest['step'], rng_seed=rng_seed,
                  settings=asdict(settings), train_rows=len(window.index), batches=[])
    (target/'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    for index in range(args.steps):
        refs = window.sample(rng, settings.batch, settings.recency,
                             regret_fraction=settings.regret_fraction)
        samples, targets = dense_data.examples(
            window, refs, rng, **dense_data.target_options(settings, calibration))
        batch = dense_data.tensors(dense_data.collate_arrays(samples, targets))
        name = f'batch-{index:02d}.pt'
        torch.save(batch, target/name)
        status['batches'].append(dict(file=name, rows={str(s): len(b['counts']) for s, b in batch.items()},
                                      refs=[dict(shard=r.shard, row=r.index) for r in refs]))
        (target/'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    status['state'] = 'complete'
    (target/'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(json.dumps(dict(snapshot=str(target), steps=args.steps, train_rows=len(window.index)), indent=2))


def measure(args):
    import torch
    torch.cuda.set_per_process_memory_fraction(.12)
    if args.memory_mib:
        total = torch.cuda.get_device_properties(0).total_memory
        if args.memory_mib*2**20 > total:
            raise ValueError('Requested allocator cap exceeds device memory')
        torch.cuda.set_per_process_memory_fraction(args.memory_mib*2**20/total)
    torch.set_num_threads(2)

    import dense_config
    import dense_learn
    import hexnet
    import hexnet_kernels

    checkpoint = args.checkpoint.resolve()
    manifest = json.loads((checkpoint/'manifest.json').read_text(encoding='utf-8'))
    settings = dense_config.section('learner', manifest['learner'])
    status = json.loads((args.batches/'status.json').read_text(encoding='utf-8'))
    if status.get('state') != 'complete' or len(status['batches']) < args.steps:
        raise ValueError('Need a complete snapshot with enough successive batches')
    batches = [torch.load(args.batches/f'batch-{i:02d}.pt', map_location='cpu', weights_only=True)
               for i in range(args.steps)]
    rows = [sum(len(b['counts']) for b in batch.values()) for batch in batches]
    if len(set(rows)) != 1 or rows[0] != settings.batch:
        raise ValueError('Snapshot batch size differs from checkpoint learner settings')
    original_update = hexnet_kernels.norm_update
    device = torch.device('cuda')
    gpu_bytes = torch.cuda.get_device_properties(0).total_memory
    fraction = torch.cuda.get_per_process_memory_fraction()
    started = time.perf_counter()
    model_sha256 = hashlib.sha256((checkpoint/'model.pt').read_bytes()).hexdigest()
    settings_sha256 = hashlib.sha256(json.dumps(asdict(settings), sort_keys=True).encode()).hexdigest()
    snapshot_sha256 = hashlib.sha256((args.batches/'status.json').read_bytes()).hexdigest()
    report = dict(checkpoint=str(checkpoint), checkpoint_step=manifest['step'],
                  model_sha256=model_sha256, settings_sha256=settings_sha256,
                  batches=str(args.batches), snapshot_checkpoint=status.get('checkpoint'),
                  snapshot_step=status.get('checkpoint_step', status.get('source_checkpoint_step')),
                  snapshot_rng_seed=status.get('rng_seed'), snapshot_status_sha256=snapshot_sha256,
                  modes=args.modes, steps=args.steps,
                  rows_per_step=rows[0], requested_memory_mib=args.memory_mib,
                  allocator_fraction=fraction, allocator_cap_mib=fraction*gpu_bytes/2**20,
                  gpu=torch.cuda.get_device_name(),
                  phases=[], stage='setup')
    path = args.output/'learner-profile.json'

    def save(stage):
        report['stage'] = stage
        report['elapsed_seconds'] = time.perf_counter()-started
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(stage, flush=True)

    @torch.no_grad()
    def shipped_update(norm, mean, var, cells):
        norm.num_batches_tracked += 1
        norm.running_mean.lerp_(mean, norm.momentum)
        norm.running_var.lerp_(var*cells/(cells-1).clamp_min(1), norm.momentum)

    def create(mode):
        hexnet_kernels.norm_update = shipped_update if mode == 'shipped' else original_update
        learner = dense_learn.Learner.__new__(dense_learn.Learner)
        learner.settings = settings
        learner.device = device
        learner.net_kernels = 'reference' if mode == 'reference' else 'fused'
        raw = hexnet.load_model(checkpoint/'model.pt')
        learner.memory_format = hexnet.memory_format(raw.config)
        learner.model = learner.place(raw).train()
        learner.ema = learner.place(hexnet.load_model(checkpoint/'ema.pt'))
        if mode == 'shipped':
            for module in learner.model.modules():
                if isinstance(module, hexnet.LineConv):
                    module.net_kernels = 'reference'
        state = torch.load(checkpoint/'optimizer.pt', map_location=device, weights_only=True)
        learner.optimizer = dense_learn.make_optimizer(learner.model, settings)
        learner.optimizer.load_state_dict(state['optimizer'])
        learner.step, learner.samples_seen = manifest['step'], manifest['samples_seen']
        learner.optimizer_started = state['optimizer_started']
        learner.ema_updates = state['ema_updates']
        return learner

    def warm(learner):
        # Shape warmup runs forward/backward without updating weights or EMA.
        # Restore running statistics so measured steps start at the checkpoint.
        buffers = [(b, b.detach().clone()) for b in learner.model.buffers()]
        rng_cpu, rng_gpu = torch.random.get_rng_state(), torch.cuda.get_rng_state(device)
        unique = {}
        for batch in batches:
            for side, bucket in batch.items():
                key = side, (len(bucket['counts'])+dense_learn.QUANTUM-1)//dense_learn.QUANTUM*dense_learn.QUANTUM
                unique.setdefault(key, bucket)
        for (side, _), bucket in unique.items():
            learner.optimizer.zero_grad(set_to_none=True)
            dense_learn.batch_losses(learner.model, {side: bucket}, learner.coefficients(),
                                     device, learner.memory_format, True)
        learner.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
        torch.random.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state(rng_gpu, device)
        return len(unique)

    warmed_modes = {}
    try:
        for index, mode in enumerate(args.modes):
            if time.perf_counter()-started > 43:
                raise TimeoutError('Benchmark budget expired before all requested modes')
            save(f'{mode}_{index}_setup')
            learner = create(mode)
            phase = dict(mode=mode, seconds=[], rows=rows)
            report['phases'].append(phase)
            if mode in warmed_modes:
                phase['warmed_shapes'] = warmed_modes[mode]
                phase['warmup_reused'] = True
                phase['warmup_seconds'] = 0.
            else:
                warm_started = time.perf_counter()
                phase['warmed_shapes'] = warm(learner)
                phase['warmup_seconds'] = time.perf_counter()-warm_started
                warmed_modes[mode] = phase['warmed_shapes']
            torch.cuda.reset_peak_memory_stats()
            for step, batch in enumerate(batches):
                if time.perf_counter()-started > 48:
                    raise TimeoutError('Benchmark budget expired before all requested steps')
                tick = time.perf_counter()
                losses = learner.train_step(batch)
                torch.cuda.synchronize()
                phase['seconds'].append(time.perf_counter()-tick)
                if step == 0:
                    phase['first_losses'] = losses.detach().cpu().tolist()
                save(f'{mode}_{index}_step_{step}')
            steady = phase['seconds'][1:]
            phase['steady_samples_per_second'] = sum(rows[1:])/sum(steady)
            phase['peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
            phase['peak_reserved_mib'] = torch.cuda.max_memory_reserved()/2**20
            if args.profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
                    learner.train_step(batches[-1])
                    torch.cuda.synchronize()
                trace = args.output/f'learner-{index}-{mode}-trace.json'
                prof.export_chrome_trace(str(trace))
                phase['profile'] = trace_summary(trace)
                phase['trace'] = str(trace)
            del learner, losses
            gc.collect()
            torch.cuda.empty_cache()
            save(f'{mode}_{index}_complete')
        references = [p for p in report['phases'] if p['mode'] == 'reference']
        if references:
            baseline = sum(sum(p['rows'][1:]) for p in references)/sum(
                sum(p['seconds'][1:]) for p in references)
            report['baseline_reference_samples_per_second'] = baseline
            report['speedup_vs_reference'] = {
                p['mode']: p['steady_samples_per_second']/baseline
                for p in report['phases'] if p['mode'] != 'reference'}
        save('complete')
    except Exception as error:
        report['error'] = dict(type=type(error).__name__, message=str(error))
        save('failed')
        raise
    finally:
        hexnet_kernels.norm_update = original_update
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'measure'))
    parser.add_argument('--run', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/learner-profile')
    parser.add_argument('--batches', type=Path)
    parser.add_argument('--steps', type=int, choices=range(3, 9), default=3)
    parser.add_argument('--modes', nargs='+', choices=('reference', 'shipped', 'fused'),
                        default=('reference', 'shipped', 'fused', 'reference'))
    parser.add_argument('--memory-mib', type=int)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.batches = (args.batches or args.output/'batches').resolve()
    if not args.output.is_relative_to(ROOT/'artifacts') or not args.batches.is_relative_to(ROOT/'artifacts'):
        parser.error('All benchmark files must stay under this worktree\'s artifacts directory')
    if args.memory_mib is not None and args.memory_mib <= 0:
        parser.error('--memory-mib must be positive')
    if args.action == 'prepare' and args.run is None:
        parser.error('prepare requires --run')
    if args.profile and len(args.modes) != 1:
        parser.error('--profile requires exactly one --modes value to fit the 55-second child limit')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == 'prepare':
        prepare(args)
        return 0
    if not args.worker:
        return guard(args, entry=Path(sys.argv[0]).resolve())
    os.environ['OMP_NUM_THREADS'] = '2'
    measure(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
