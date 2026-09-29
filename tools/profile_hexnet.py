"""Reproduce bounded HeXO GPU comparisons from a frozen, read-only shard sample.

python tools/profile_hexnet.py prepare --run /path/to/runs/dense-v1
python tools/profile_hexnet.py eval --batch 256 --profile
python tools/profile_hexnet.py train --batch 256 --profile
python tools/profile_hexnet.py check

Each GPU invocation owns one child, caps its allocator at 12%, stops it at 55 s,
and waits 60 s after the preceding child exits. Outputs stay in this worktree.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def subset(batch, n):
    out, offset = {}, 0
    total = sum(len(b['counts']) for b in batch.values())
    for size, b in batch.items():
        end = offset+len(b['counts'])
        take = end*n//total-offset*n//total
        offset = end
        if take:
            out[size] = {k: v[:int(b['offsets'][take])] if k == 'policy' else v[:take+1] if k == 'offsets' else v[:take]
                         for k, v in b.items()}
    return out


def prepare(args):
    import numpy as np
    import torch
    import dense_data
    import dense_learn
    torch.set_num_threads(2)
    run = args.run.resolve()
    checkpoint = args.checkpoint or run/'checkpoints'/json.loads((run/'champion.json').read_text())['checkpoint']/'ema.pt'
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    paths = ([run/'shards'/name for name in args.shards] if args.shards else
             sorted(p for p in (run/'shards').iterdir() if (p/'manifest.json').exists())[-4:])
    refs, following, values = [], {}, {}
    for path in paths:
        episodes, rows = dense_data.read_shard(path)
        for i, row in enumerate(rows):
            ref = SimpleNamespace(episode=episodes[row['game']], row=row, shard=path.name, index=i)
            refs.append(ref)
            following[path.name, row['game'], row['ply']] = ref

    def targets(ref, lam, full, outcome_lam, calibration):
        key = ref.shard, ref.row['game']
        if key not in values:
            values[key] = dense_data.episode_value_targets(ref.episode, lam, full, outcome_lam, calibration)
        return values[key]

    window = SimpleNamespace(policy=lambda r: r.row['policy'], value_targets=targets,
                             following=lambda r: following.get((r.shard, r.row['game'], r.row['ply']+1)))
    rng = np.random.default_rng(args.seed)
    selected = [refs[i] for i in rng.choice(len(refs), 256, replace=False)]
    start = time.perf_counter()
    samples, target = dense_data.examples(window, selected, rng, future_target=weights.get('future_target', 'legacy'))
    render_ms = (time.perf_counter()-start)*1000
    times = defaultdict(list)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
        for _ in range(5):
            start = time.perf_counter()
            with torch.profiler.record_function('collate_256'):
                batch = dense_data.collate(samples, target)
            times['collate_ms'].append((time.perf_counter()-start)*1000)
            start = time.perf_counter()
            with torch.profiler.record_function('pad_256'):
                padded = {s: dense_learn.pad(b, dense_learn.QUANTUM) for s, b in batch.items()}
            times['pad_ms'].append((time.perf_counter()-start)*1000)
    prof.export_chrome_trace(str(args.output/'cpu-trace.json'))
    torch.save(batch, args.output/'batch.pt')
    torch.save(weights, args.output/'model.pt')
    return dict(seed=args.seed, shards=[p.name for p in paths], checkpoint=str(checkpoint), render_ms=render_ms,
                references=[dict(shard=r.shard, row=r.index) for r in selected], cpu_ms=times,
                mean_crop_cells=sum(s.size*s.size for s in samples)/256,
                shapes={s:list(b['planes'].shape) for s,b in batch.items()},
                padded_shapes={s:list(b['planes'].shape) for s,b in padded.items()},
                hashes={name:hashlib.sha256((args.output/name).read_bytes()).hexdigest() for name in ('batch.pt','model.pt')})


def trace_summary(path):
    events = json.loads(path.read_text())['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel']
    runtime = [e for e in events if e.get('cat') in ('cuda_runtime', 'cuda_driver')]
    launch = [e for e in runtime if 'LaunchKernel' in e['name'] or 'GraphLaunch' in e['name']]
    grouped = defaultdict(lambda: [0, 0.])
    for e in kernels:
        grouped[e['name']][0] += 1
        grouped[e['name']][1] += e['dur']/1000
    # Autocast can report both an outer and an inner conv2d with the same FLOPs.
    # Count only the inner operation. All HexNet convolutions preserve H and W.
    ops = [e for e in events if e.get('cat') == 'cpu_op' and e['name'] in
           ('aten::conv2d','aten::mm','aten::bmm','aten::addmm','aten::baddbmm_')]
    flops = 0
    for e in ops:
        if any(x is not e and x['name'] == e['name'] and x.get('tid') == e.get('tid') and x['ts'] >= e['ts']
               and x['ts']+x['dur'] <= e['ts']+e['dur'] and x['dur'] < e['dur'] for x in ops):
            continue
        dims, name = e['args']['Input Dims'], e['name']
        if name == 'aten::conv2d':
            b, _, h, w = dims[0]
            co, ci, kh, kw = dims[1]
            flops += 2*b*h*w*co*ci*kh*kw
        else:
            a, z = (dims[1],dims[2]) if name in ('aten::addmm','aten::baddbmm_') else (dims[0],dims[1])
            flops += 2*a[-2]*a[-1]*z[-1]*(a[0] if len(a) == 3 else 1)
    copies = [e for e in events if e.get('cat') == 'gpu_memcpy' and 'HtoD' in e['name']]
    return dict(kernel_count=len(kernels), kernel_ms=sum(e['dur'] for e in kernels)/1000,
                launch_calls=len(launch), launch_cpu_ms=sum(e['dur'] for e in launch)/1000,
                graph_launch_calls=sum('GraphLaunch' in e['name'] for e in launch),
                sync_cpu_ms=sum(e['dur'] for e in runtime if 'Synchronize' in e['name'])/1000,
                conv_gemm_flops=flops, h2d_bytes=sum(e['args']['bytes'] for e in copies),
                h2d_ms=sum(e['dur'] for e in copies)/1000,
                top_kernels=[dict(name=k, calls=v[0], ms=v[1]) for k,v in sorted(grouped.items(),key=lambda kv:-kv[1][1])[:12]])


def measure(args):
    import copy
    import torch
    torch.cuda.set_per_process_memory_fraction(.12)  # before every other CUDA call
    torch.set_num_threads(2)
    import dense_learn
    import hexnet
    source = torch.load(args.output/'batch.pt', weights_only=True)
    batch = subset(source, args.batch)
    device = torch.device('cuda')
    fmt = torch.contiguous_format  # b6c96 with LineConv uses this in origin/main
    report = dict(mode=args.mode, rows=args.batch, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(), fraction=torch.cuda.get_per_process_memory_fraction(),
                  shapes={s:list(b['planes'].shape) for s,b in batch.items()})
    if 'graphs' in args.kernels and args.mode != 'eval':
        raise ValueError('graphs is an inference-only comparison mode')
    models = {mode:hexnet.load_model(args.output/'model.pt', device=device,
                                   net_kernels='fused' if mode == 'graphs' else mode) for mode in args.kernels}
    formats = {mode:torch.channels_last if args.mode == 'eval' and mode != 'reference' else fmt for mode in models}
    for mode, model in models.items():
        model.to(memory_format=formats[mode])
        if args.mode == 'eval':
            model.eval().requires_grad_(False)
    graphs = {}
    if 'graphs' in models:
        from hexnet_graphs import ActorGraph
        graphs['graphs'] = ActorGraph(models['graphs'])
    optim = {m:torch.optim.AdamW(model.parameters(), lr=.0003, fused=True) for m,model in models.items()} if args.mode == 'train' else {}
    ema = {m:copy.deepcopy(model) for m,model in models.items()} if args.mode == 'train' else {}
    coeff = torch.tensor([1.,1.5,.5,.15,.5,0.], device=device)
    inputs = {mode:[b['planes'].to(device).to(dtype=torch.bfloat16,memory_format=layout) for b in batch.values()]
              for mode,layout in formats.items()} if args.mode == 'eval' else {}

    def step(mode):
        model = models[mode]
        if args.mode == 'train':
            model.train()
            optim[mode].zero_grad(set_to_none=True)
            dense_learn.batch_losses(model, batch, coeff, device, fmt, True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optim[mode].step()
            dense_learn.update_ema(ema[mode], model, .999)
        elif args.mode == 'transfer':
            for bucket in batch.values():
                with torch.profiler.record_function('cpu_pad'):
                    padded = dense_learn.pad(bucket, dense_learn.QUANTUM)
                with torch.profiler.record_function('h2d'):
                    b = {k:v.to(device, non_blocking=True) for k,v in padded.items()}
                with torch.profiler.record_function('cast_layout'):
                    planes = b['planes'].float().contiguous(memory_format=fmt)
        else:
            model.eval()
            with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
                for x in inputs[mode]:
                    if mode in graphs:
                        graphs[mode](x)
                    else:
                        for part in x.split(max(1,110592//x.shape[-1]**2)):
                            model(part, part[:,3:4], aux=False)

    try:
        for mode in models:
            step(mode)
            torch.cuda.synchronize()
        times = defaultdict(list)
        for mode in (args.kernels+args.kernels[::-1])*args.repeats:
            start = time.perf_counter()
            step(mode)
            torch.cuda.synchronize()
            times[mode].append(time.perf_counter()-start)
        medians = {m:statistics.median(v) for m,v in times.items()}
        report.update(seconds=times, median_seconds=medians, rows_per_second={m:args.batch/v for m,v in medians.items()})
        if len(medians) == 2:
            report['speedup'] = medians[args.kernels[0]]/medians[args.kernels[1]]
        (args.output/f'comparison-{args.mode}-{args.batch}.json').write_text(json.dumps(report,indent=2))
        if args.profile:
            report['profiles'] = {}
            for mode in models:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=True, with_flops=True) as prof:
                    step(mode)
                    torch.cuda.synchronize()
                path = args.output/f'{args.mode}-{args.batch}-{mode}-trace.json'
                prof.export_chrome_trace(str(path))
                report['profiles'][mode] = trace_summary(path)
    except torch.OutOfMemoryError as error:
        report['oom'] = str(error)
    report.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    for mode, graph in graphs.items():
        report[mode+'_reserved_mib'] = graph.incremental_reserved_bytes/2**20
        graph.close()
    return report


def guard(args):
    state = ROOT/'artifacts/gpu-kernels'
    state.mkdir(parents=True, exist_ok=True)
    stamp, lock = state/'last_gpu_end.txt', state/'gpu.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    try:
        while True:
            idle = 60-(time.time()-float(stamp.read_text())) if stamp.exists() else 0
            used = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            if idle <= 0 and used <= 7400:
                break
            print(json.dumps(dict(cooldown_seconds=max(0,round(idle)), gpu_used_mib=used)), flush=True)
            time.sleep(min(10,max(1,idle)))
        tmp = args.output/'tmp'
        tmp.mkdir(exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
                   TEMP=str(tmp), TMP=str(tmp), TRITON_CACHE_DIR=str(args.output/'triton-cache'),
                   TORCHINDUCTOR_CACHE_DIR=str(args.output/'inductor-cache'), CUDA_CACHE_PATH=str(args.output/'cuda-cache'))
        if args.mode == 'check':
            env.update(HEXO_TEST_CUDA='1', HEXO_TEST_BATCH=str(args.output/'batch.pt'), HEXO_TEST_MODEL=str(args.output/'model.pt'))
            command = [sys.executable,'-B','-m','unittest','tests.test_dense.FusedCudaTests']
        else:
            command = [sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:],'--output',str(args.output),'--worker']
        started = time.time()
        child = subprocess.Popen(command,cwd=ROOT,env=env,
                                 creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name=='nt' else 0)
        try:
            code = child.wait(timeout=55)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            code = 124
        finally:
            ended = time.time()
            stamp.write_text(str(ended))
            with (state/'sessions.jsonl').open('a') as f:
                f.write(json.dumps(dict(start=started,end=ended,elapsed=ended-started,code=code,gpu_used_mib=used,command=command))+'\n')
        return code
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','eval','train','transfer','check'])
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/gpu-kernels')
    parser.add_argument('--run',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--shards',nargs='+')
    parser.add_argument('--seed',type=int,default=3070)
    parser.add_argument('--batch',type=int,choices=[16,32,64,128,256],default=256)
    parser.add_argument('--kernels',choices=['reference','fused','graphs'],nargs='+',default=['reference','fused'])
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--repeats',type=int,choices=[1,2,3],default=3)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT/'artifacts'):
        parser.error('output must be under this worktree\'s artifacts directory')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.mode == 'prepare' and args.run is None:
        parser.error('prepare requires --run')
    if args.mode != 'prepare' and not args.worker:
        return guard(args)
    os.environ['OMP_NUM_THREADS'] = '2'
    report = prepare(args) if args.mode == 'prepare' else measure(args)
    path = args.output/('inputs.json' if args.mode == 'prepare' else f'comparison-{args.mode}-{args.batch}.json')
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return 2 if 'oom' in report else 0


if __name__ == '__main__':
    sys.exit(main())
