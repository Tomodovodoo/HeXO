"""Benchmark real HexNet learner steps from a read-only checkpoint and replay snapshot.

First freeze committed replay shards and render consecutive window batches once;
later modes reuse that same worktree snapshot::

    python tools/profile_learner.py prepare --run runs/dense-v1 --checkpoint runs/dense-v1/checkpoints/main/085000
    python tools/profile_learner.py measure --checkpoint runs/dense-v1/checkpoints/main/085000
    python tools/profile_learner.py measure --checkpoint runs/dense-v1/checkpoints/main/085000 --modes fused --profile
    python tools/profile_learner.py prepare --run runs/dense-v1 --checkpoint runs/dense-v1/checkpoints/main/085000 --batches artifacts/learner-profile/live-batches
    python tools/profile_learner.py live --run runs/dense-v1 --checkpoint runs/dense-v1/checkpoints/main/085000 --batches artifacts/learner-profile/live-batches --modes fused --memory-mib 3328

The GPU command runs in one 55-second child with a 60-second cooldown and a
12% allocator cap by default. Profile one mode per invocation. ``--memory-mib 3328`` requires that much free
GPU memory plus 384 MiB before launch. ``measure`` is a short saved-batch
comparison; ``live`` times a sustained renderer and learner loop. It never
stops another process.
"""

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import zlib

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.profile_hexnet import guard, trace_summary
PROCESS_START = time.perf_counter()


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def replay_source_sha256(run, admitted):
    """Fingerprint replay metadata used to choose and label the fixed window."""
    import dense_data
    digest = hashlib.sha256()
    admitted_names = {name for name, _ in admitted}
    for shard in dense_data.shard_dirs(run):
        paths = [shard/'manifest.json']
        if shard.name in admitted_names:
            paths.append(shard/dense_data.SIDECAR)
        for path in paths:
            try:
                stat = path.stat()
                stamp = (shard.name, path.name, stat.st_size, stat.st_mtime_ns)
            except FileNotFoundError:
                stamp = (shard.name, path.name, None, None)
            digest.update(json.dumps(stamp).encode())
    path = run/'restarts.json'
    try:
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
    except FileNotFoundError:
        stamp = (None, None)
    digest.update(json.dumps(stamp).encode())
    return digest.hexdigest()


def freeze_replay(run, target):
    """Copy one committed replay corpus into the worktree, without linking to run files."""
    import dense_data
    frozen = target/'replay'
    (frozen/'shards').mkdir(parents=True)
    inventory = dict(source_run=str(run), files={}, shards={})
    for name in ('config.json', 'restarts.json'):
        source = run/name
        if source.exists():
            shutil.copy2(source, frozen/name)
            inventory['files'][name] = file_sha256(frozen/name)
        elif name == 'config.json':
            raise FileNotFoundError(source)
    for source in dense_data.shard_dirs(run):
        if not (source/'manifest.json').is_file():
            continue
        destination = frozen/'shards'/source.name
        destination.mkdir()
        copied = {}
        for name in ('manifest.json', *dense_data.FILES, dense_data.SIDECAR):
            path = source/name
            if not path.exists() and name == dense_data.SIDECAR:
                continue
            shutil.copy2(path, destination/name)
            copied[name] = file_sha256(destination/name)
        manifest = dense_data.manifest(destination)
        if any(copied[name] != manifest['files'][name] for name in dense_data.FILES):
            raise ValueError(f'Committed replay shard changed while freezing: {source.name}')
        inventory['shards'][source.name] = copied
    path = target/'frozen-corpus.json'
    path.write_text(json.dumps(inventory, sort_keys=True, indent=2), encoding='utf-8')
    return frozen, file_sha256(path), len(inventory['shards'])


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
    frozen_run, corpus_sha256, shard_count = freeze_replay(run, target)
    config = dense_config.load(frozen_run)
    settings = dense_config.section('learner', manifest['learner'])
    window = dense_data.ReplayWindow(
        frozen_run, settings.window_capacity, settings.window_min_rows,
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
                  frozen_run=str(frozen_run), frozen_corpus_sha256=corpus_sha256,
                  frozen_shards=shard_count,
                  checkpoint_step=manifest['step'], rng_seed=rng_seed,
                  checkpoint_manifest_sha256=file_sha256(checkpoint/'manifest.json'),
                  settings=asdict(settings), train_rows=len(window.index),
                  calibration=dense_data.pack_calibration(calibration),
                  admitted=[list(item) for item in window.admitted],
                  regret_entries=[[name, game, ply, weight]
                                  for (name, game, ply), weight in sorted(window.regret_entries.items())],
                  replay_source_sha256=replay_source_sha256(frozen_run, window.admitted),
                  batches=[])
    (target/'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    for index in range(args.steps):
        refs = window.sample(rng, settings.batch, settings.recency,
                             regret_fraction=settings.regret_fraction)
        samples, targets = dense_data.examples(
            window, refs, rng, **dense_data.target_options(settings, calibration))
        batch = dense_data.tensors(dense_data.collate_arrays(samples, targets))
        name = f'batch-{index:02d}.pt'
        torch.save(batch, target/name)
        status['batches'].append(dict(file=name, sha256=file_sha256(target/name),
                                      rows={str(s): len(b['counts']) for s, b in batch.items()},
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
    if status.get('settings') != asdict(settings):
        raise ValueError('Snapshot sampling/target settings differ from checkpoint manifest')
    batch_hashes = {}
    for index in range(args.steps):
        name = f'batch-{index:02d}.pt'
        if status['batches'][index]['file'] != name:
            raise ValueError('Snapshot batch order differs from its manifest')
        digest = file_sha256(args.batches/name)
        expected = status['batches'][index].get('sha256')
        if expected is not None and digest != expected:
            raise ValueError(f'Snapshot batch payload changed: {name}')
        batch_hashes[name] = digest
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
    model_sha256 = file_sha256(checkpoint/'model.pt')
    settings_sha256 = hashlib.sha256(json.dumps(asdict(settings), sort_keys=True).encode()).hexdigest()
    snapshot_sha256 = hashlib.sha256((args.batches/'status.json').read_bytes()).hexdigest()
    report = dict(checkpoint=str(checkpoint), checkpoint_step=manifest['step'],
                  checkpoint_manifest_sha256=file_sha256(checkpoint/'manifest.json'),
                  model_sha256=model_sha256, settings_sha256=settings_sha256,
                  ema_sha256=file_sha256(checkpoint/'ema.pt'),
                  optimizer_sha256=file_sha256(checkpoint/'optimizer.pt'),
                  batches=str(args.batches), snapshot_checkpoint=status.get('checkpoint'),
                  snapshot_step=status.get('checkpoint_step', status.get('source_checkpoint_step')),
                  snapshot_manifest_sha256=status.get('checkpoint_manifest_sha256'),
                  snapshot_rng_seed=status.get('rng_seed'), snapshot_status_sha256=snapshot_sha256,
                  batch_sha256=batch_hashes,
                  modes=args.modes, steps=args.steps,
                  timing_scope='short_saved_batch_window',
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
            phase['short_window_samples_per_second'] = sum(rows[1:])/sum(steady)
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
            report['baseline_reference_short_window_samples_per_second'] = baseline
            report['speedup_vs_reference_short_window'] = {
                p['mode']: p['short_window_samples_per_second']/baseline
                for p in report['phases'] if p['mode'] != 'reference'}
        save('complete')
    except Exception as error:
        report['error'] = dict(type=type(error).__name__, message=str(error))
        save('failed')
        raise
    finally:
        hexnet_kernels.norm_update = original_update
    print(json.dumps(report, indent=2), flush=True)


def live(args):
    """Time the deployed next(Renderers) + train_step loop, without run writes."""
    import torch
    torch.set_num_threads(2)
    import dense_config
    import dense_data
    import dense_learn
    import hexnet
    import hexnet_kernels

    run, checkpoint = args.run.resolve(), args.checkpoint.resolve()
    manifest = json.loads((checkpoint/'manifest.json').read_text(encoding='utf-8'))
    settings = dense_config.section('learner', manifest['learner'])
    frozen_run = args.batches/'replay'
    config = dense_config.load(frozen_run)
    mode = args.modes[0]
    original_update = hexnet_kernels.norm_update
    GPU_START = None

    @torch.no_grad()
    def shipped_update(norm, mean, var, cells):
        norm.num_batches_tracked += 1
        norm.running_mean.lerp_(mean, norm.momentum)
        norm.running_var.lerp_(var*cells/(cells-1).clamp_min(1), norm.momentum)

    report = dict(mode=mode, run=str(run), frozen_run=str(frozen_run), checkpoint=str(checkpoint),
                  timing_scope='live_renderer_and_learner_window',
                  checkpoint_step=manifest['step'],
                  checkpoint_manifest_sha256=file_sha256(checkpoint/'manifest.json'),
                  model_sha256=file_sha256(checkpoint/'model.pt'),
                  ema_sha256=file_sha256(checkpoint/'ema.pt'),
                  optimizer_sha256=file_sha256(checkpoint/'optimizer.pt'),
                  settings_sha256=hashlib.sha256(json.dumps(asdict(settings), sort_keys=True).encode()).hexdigest(),
                  workers=args.workers, seed=[config.seed, zlib.crc32(settings.variant.encode()), manifest['step']],
                  minimum_steps=args.steps, minimum_active_seconds=args.min_seconds,
                  steps=[], stage='setup')
    # Keep interleaved reference/fused/reference runs as separate observations.
    path = args.output/f'learner-live-{mode}-{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}.json'
    marker = args.output/f'gpu-start-{os.getpid()}.txt'
    marker.unlink(missing_ok=True)

    def save(stage):
        report['stage'] = stage
        report['process_seconds'] = time.perf_counter()-PROCESS_START
        if GPU_START is not None:
            report['gpu_seconds'] = time.perf_counter()-GPU_START
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(stage, flush=True)

    stream = None
    try:
        status_path = args.batches/'status.json'
        status = json.loads(status_path.read_text(encoding='utf-8'))
        if (status.get('state') != 'complete' or status.get('run') != str(run)
                or status.get('frozen_run') != str(frozen_run)
                or status.get('checkpoint_step') != manifest['step']
                or status.get('checkpoint_manifest_sha256') != report['checkpoint_manifest_sha256']
                or status.get('settings') != asdict(settings)
                or status.get('rng_seed', [])[:3] != report['seed']
                or not status.get('admitted') or not status.get('calibration')
                or status.get('regret_entries') is None):
            raise ValueError('Prepare a complete CPU replay snapshot for this run and checkpoint before live')
        corpus_sha256 = file_sha256(args.batches/'frozen-corpus.json')
        if corpus_sha256 != status.get('frozen_corpus_sha256'):
            raise ValueError('Frozen replay corpus manifest changed after preparation')
        source_sha256 = replay_source_sha256(frozen_run, status['admitted'])
        if source_sha256 != status.get('replay_source_sha256'):
            raise ValueError('Frozen replay corpus changed since CPU preparation')
        calibration = dense_data.unpack_calibration(status['calibration'])
        regret_entries = {(name, int(game), int(ply)): float(weight)
                          for name, game, ply, weight in status['regret_entries']}
        policy_dir = args.batches/'policy-cache'
        identity = dict(run=str(run), checkpoint_step=manifest['step'],
                        settings_sha256=report['settings_sha256'], workers=args.workers,
                        seed=report['seed'], admitted=status['admitted'],
                        train_rows=status['train_rows'], source_sha256=source_sha256,
                        corpus_sha256=corpus_sha256,
                        snapshot_sha256=file_sha256(status_path))
        identity_path = args.batches/'learner-live-window.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text(encoding='utf-8')) != identity:
                raise ValueError('Replay window or settings changed between live benchmark modes')
        else:
            identity_path.write_text(json.dumps(identity, indent=2), encoding='utf-8')
        report['window_sha256'] = file_sha256(identity_path)
        report['train_rows'] = status['train_rows']
        report['regret_entries'] = len(regret_entries)
        report['frozen_corpus_sha256'] = corpus_sha256
        report['frozen_shards'] = status['frozen_shards']
        report['snapshot_sha256'] = file_sha256(status_path)
        save('renderers_start')
        bootstrap_started = time.perf_counter()
        stream = dense_data.Renderers(frozen_run, settings, report['seed'], args.workers,
                                      calibration=calibration, policy_dir=policy_dir,
                                      regret_entries=regret_entries, run_seed=config.seed)
        if os.name == 'nt':
            import ctypes
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            for process in stream.processes:
                handle = kernel32.OpenProcess(0x0200, False, process.pid)
                if not handle:
                    raise OSError(ctypes.get_last_error(), 'Could not open render worker for priority change')
                try:
                    lowered = kernel32.SetPriorityClass(handle, 0x4000)
                finally:
                    kernel32.CloseHandle(handle)
                if not lowered:
                    raise OSError(ctypes.get_last_error(), 'Could not set render worker BelowNormal priority')
        elif hasattr(os, 'setpriority'):
            for process in stream.processes:
                os.setpriority(os.PRIO_PROCESS, process.pid, 10)

        warm_batch = next(stream)
        report['renderer_bootstrap_seconds'] = time.perf_counter()-bootstrap_started
        raw = hexnet.load_model(checkpoint/'model.pt')
        raw_ema = hexnet.load_model(checkpoint/'ema.pt')
        state = torch.load(checkpoint/'optimizer.pt', map_location='cpu', weights_only=True)
        while True:
            used, free, total = map(int, subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total',
                 '--format=csv,noheader,nounits'], text=True).strip().split(','))
            requested = args.memory_mib or int(.12*total+.999)
            if used <= 7400 and free >= requested+384:
                break
            report['preflight_gpu_mib'] = dict(used=used, free=free, requested=requested)
            save('waiting_for_gpu')
            time.sleep(10)
        report['preflight_gpu_mib'] = dict(used=used, free=free, requested=requested)
        save('gpu_ready')
        marker.write_text(str(time.time()), encoding='utf-8')
        GPU_START = time.perf_counter()
        torch.cuda.set_per_process_memory_fraction(.12)
        if args.memory_mib:
            cuda_total = torch.cuda.get_device_properties(0).total_memory
            if args.memory_mib*2**20 > cuda_total:
                raise ValueError('Requested allocator cap exceeds device memory')
            torch.cuda.set_per_process_memory_fraction(args.memory_mib*2**20/cuda_total)
        device = torch.device('cuda')
        report['gpu'] = torch.cuda.get_device_name()
        report['allocator_fraction'] = torch.cuda.get_per_process_memory_fraction()
        report['allocator_cap_mib'] = report['allocator_fraction']*torch.cuda.get_device_properties(0).total_memory/2**20

        hexnet_kernels.norm_update = shipped_update if mode == 'shipped' else original_update
        learner = dense_learn.Learner.__new__(dense_learn.Learner)
        learner.settings, learner.device = settings, device
        learner.net_kernels = 'reference' if mode == 'reference' else 'fused'
        learner.memory_format = hexnet.memory_format(raw.config)
        learner.model = learner.place(raw).train()
        learner.ema = learner.place(raw_ema)
        if mode == 'shipped':
            for module in learner.model.modules():
                if isinstance(module, hexnet.LineConv):
                    module.net_kernels = 'reference'
        learner.optimizer = dense_learn.make_optimizer(learner.model, settings)
        learner.optimizer.load_state_dict(state['optimizer'])
        learner.step, learner.samples_seen = manifest['step'], manifest['samples_seen']
        learner.optimizer_started, learner.ema_updates = state['optimizer_started'], state['ema_updates']

        sums = torch.zeros(len(dense_learn.HEADS), device=device)
        counts = torch.zeros(len(dense_learn.HEADS), device=device)
        report['metric_flush_steps'] = []

        def update_metrics(losses):
            sums.add_(losses.nan_to_num())
            counts.add_(losses.isfinite())
            logged = learner.step % settings.log_every == 0
            if learner.step % 10 == 0 or logged or learner.step % settings.export_every == 0:
                learner.metrics = {h: float(v/n) if n else None
                                   for h, v, n in zip(learner.heads, sums.tolist(), counts.tolist())}
                sums.zero_()
                counts.zero_()
                report['metric_flush_steps'].append(learner.step)

        # One complete step warms the exact renderer and learner path.
        warm_started = time.perf_counter()
        warm_ready = warm_started
        update_metrics(learner.train_step(warm_batch))
        torch.cuda.synchronize()
        report['warmup_seconds'] = time.perf_counter()-warm_started
        report['warm_train_seconds'] = time.perf_counter()-warm_ready
        if time.perf_counter()-GPU_START + args.min_seconds > 49:
            raise TimeoutError('Insufficient 55-second child budget after renderer startup')
        save('measuring')
        active_start = time.perf_counter()
        while len(report['steps']) < args.steps or time.perf_counter()-active_start < args.min_seconds:
            if time.perf_counter()-GPU_START > 49:
                raise TimeoutError('Could not complete the minimum live window within child budget')
            started = time.perf_counter()
            batch = next(stream)
            ready = time.perf_counter()
            losses = learner.train_step(batch)
            update_metrics(losses)
            finished = time.perf_counter()
            report['steps'].append(dict(rows=sum(len(b['counts']) for b in batch.values()),
                                        bucket_rows={str(side): len(b['counts']) for side, b in batch.items()},
                                        render_wait_seconds=ready-started,
                                        train_step_host_seconds=finished-ready))
        torch.cuda.synchronize()
        ended = time.perf_counter()
        if replay_source_sha256(frozen_run, status['admitted']) != source_sha256:
            raise ValueError('Frozen replay corpus changed during the live benchmark')
        report['active_seconds'] = ended-active_start
        report['tail_sync_seconds'] = ended-finished
        report['rows_per_second'] = sum(step['rows'] for step in report['steps'])/report['active_seconds']
        report['mean_render_wait_seconds'] = statistics.mean(step['render_wait_seconds'] for step in report['steps'])
        report['mean_train_step_host_seconds'] = statistics.mean(step['train_step_host_seconds'] for step in report['steps'])
        report['last_losses'] = losses.detach().cpu().tolist()
        report['peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
        report['peak_reserved_mib'] = torch.cuda.max_memory_reserved()/2**20
        save('complete')
        print(json.dumps(dict(report=str(path), rows_per_second=report['rows_per_second'],
                              steps=len(report['steps']), active_seconds=report['active_seconds'])), flush=True)
    except Exception as error:
        report['error'] = dict(type=type(error).__name__, message=str(error))
        save('failed')
        raise
    finally:
        if stream is not None:
            stream.close()
        hexnet_kernels.norm_update = original_update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'measure', 'live'))
    parser.add_argument('--run', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/learner-profile')
    parser.add_argument('--batches', type=Path)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--min-seconds', type=float, default=20.)
    parser.add_argument('--workers', type=int, default=2)
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
    if args.steps is None:
        args.steps = 10 if args.action == 'live' else 3
    if args.action in ('prepare', 'live') and args.run is None:
        parser.error(f'{args.action} requires --run')
    if args.action == 'live':
        if len(args.modes) != 1 or args.steps < 10 or args.min_seconds < 20 or args.workers < 1:
            parser.error('live requires one mode, at least 10 steps, 20 seconds and one worker')
        if args.profile:
            parser.error('profile live work in a separate measure invocation')
    elif not 3 <= args.steps <= 8:
        parser.error('prepare and measure require 3 to 8 steps')
    if args.profile and len(args.modes) != 1:
        parser.error('--profile requires exactly one --modes value to fit the 55-second child limit')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == 'prepare':
        prepare(args)
        return 0
    if not args.worker:
        return guard(args, entry=Path(sys.argv[0]).resolve())
    os.environ['OMP_NUM_THREADS'] = '2'
    (measure if args.action == 'measure' else live)(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
