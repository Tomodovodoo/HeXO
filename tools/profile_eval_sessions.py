"""Compare complete evaluator sessions on a frozen copy of live league evidence.

The source run is read-only. Each mode gets its own copy of reports and opening
book. Fixed colour pairs run to completion through Evaluator.session, with the
live search settings. GPU windows target 40 seconds, with a 55-second watchdog, separated by at least
60 seconds idle. Outstanding proof work drains before each idle interval and is
included in active time. Report wall time too: segmented time is not an estimate
of uninterrupted live throughput.
"""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.profile_evaluator import sha, write


def memory():
    return tuple(map(int, subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total',
         '--format=csv,noheader,nounits'], text=True).strip().split(',')))


def prepare(args):
    import dense_config
    import dense_eval

    source, frozen = args.run.resolve(), args.output/'frozen'
    frozen.mkdir()
    for name in ('config.json', 'league.json', 'openings.json', 'evaluator-status.json'):
        shutil.copy2(source/name, frozen/name)
    for pattern in ('evaluations/*/report*.json', 'variant-requests/*.json'):
        for path in source.glob(pattern):
            target = frozen/path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    config = dense_config.load(frozen)
    live = json.loads((frozen/'evaluator-status.json').read_text())
    settings = dense_config.section('evaluation', live['settings'])
    evaluator = dense_eval.Evaluator(frozen, replace(config, device='cpu'), settings, dense_eval.Pacer(1.))
    tasks, excluded, names = [], set(), set()
    # Select two genuinely useful fill comparisons, using the production ranking.
    # Freeze them before either run so changed completion order cannot change the workload.
    while len(tasks) < 2:
        task = evaluator.fill(excluded)
        if task is None:
            raise ValueError('No two eligible neural comparisons fit in three models')
        entry, opponent, kind, games = task
        a, b = entry['id'], opponent
        excluded.update(((a, b), (b, a)))
        if b == dense_eval.SEAL or len(names | {a, b}) > 3:
            continue
        names.update((a, b))
        evaluator.open(a, b)
        tasks.append(dict(a=a, b=b, kind=kind, games=games,
                          before=len(evaluator.games(a, b)), first_pair=evaluator.next[a, b]))
    paths = {name: source/'checkpoints'/dense_eval.split_id(name)[0]/'ema.pt' for name in names}
    write(args.output/'inputs.json', dict(schema='hexo-eval-sessions-v1', source=str(source),
          settings=asdict(evaluator.settings), tasks=tasks, checkpoints={name: dict(path=str(path), sha256=sha(path))
          for name, path in paths.items()}, files={str(p.relative_to(frozen)): sha(p) for p in frozen.rglob('*.json')}))
    print(json.dumps(dict(tasks=tasks, models=sorted(names), settings=asdict(evaluator.settings))), flush=True)


def guard(args):
    state = ROOT/'artifacts/gpu-kernels'
    lock, stamp = state/'gpu.lock', state/'last_gpu_end.txt'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    child = None
    marker = args.output/f'{args.mode}-window.json'
    try:
        while True:
            wait = 60-(time.time()-float(stamp.read_text())) if stamp.exists() else 0
            used, free, total = memory()
            if wait <= 0 and used <= 7400 and free >= .12*total+384:
                break
            print(json.dumps(dict(wait_seconds=max(0, wait), used_mib=used)), flush=True)
            time.sleep(min(10, max(1, wait)) if wait > 0 else 10)
        if marker.exists():
            marker.unlink()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, '-B', str(Path(__file__).resolve()), 'run', '--mode', args.mode,
                   '--output', str(args.output), '--worker']
        child = subprocess.Popen(command, cwd=ROOT, env=env,
                                 start_new_session=os.name != 'nt',
                                 creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 0)
        started = time.time()
        while child.poll() is None:
            stage = json.loads(marker.read_text()) if marker.exists() else dict(phase='prepare', since=started)
            deadline = stage['since']+(55 if stage['phase'] == 'active' else 300 if stage['phase'] == 'prepare' else float('inf'))
            if time.time() > deadline:
                raise TimeoutError(f"Evaluator session worker exceeded {stage['phase']} deadline")
            time.sleep(.2)
        return child.returncode
    finally:
        if child is not None and child.poll() is None:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                import signal
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        stamp.write_text(str(time.time()))
        lock.unlink()


def run(args):
    import torch
    import dense_config
    import dense_eval
    import dense_solver

    torch.set_num_threads(2)
    inputs = json.loads((args.output/'inputs.json').read_text())
    frozen, target = args.output/'frozen', args.output/args.mode
    for name, digest in inputs['files'].items():
        if sha(frozen/name) != digest:
            raise ValueError(f'Frozen input changed: {name}')
    for name, record in inputs['checkpoints'].items():
        if sha(record['path']) != record['sha256']:
            raise ValueError(f'Checkpoint changed: {name}')
    shutil.copytree(frozen, target)
    config = dense_eval.kernel_config(dense_config.load(target), 'fused')
    settings = replace(dense_config.section('evaluation', inputs['settings']), pipeline=args.mode == 'pipeline')
    marker = args.output/f'{args.mode}-window.json'
    windows, pending, pools, shapes = [], [], [], Counter()
    active_start = time.perf_counter()
    wall_start = time.time()

    def phase(name):
        temp = marker.with_suffix('.tmp')
        write(temp, dict(phase=name, since=time.time()))
        temp.replace(marker)

    phase('active')
    torch.cuda.set_per_process_memory_fraction(.12)  # First CUDA call, before models/status.
    original_submit = dense_solver.Pool.submit
    original_pool = dense_eval.Pool

    def submit(pool, *a, **kw):
        pending[:] = [future for future in pending if not future.done()]
        future = original_submit(pool, *a, **kw)
        pending.append(future)
        return future

    def quiesce():
        # Main thread submits nothing here. All CPU proof work and GPU work finish
        # before the idle timer starts, so neither can make uncounted progress.
        for future in pending:
            future.result()
        pending.clear()
        torch.cuda.synchronize()

    def boundary(final=False):
        nonlocal active_start
        if not final and time.perf_counter()-active_start < 40:
            return
        quiesce()
        seconds = time.perf_counter()-active_start
        windows.append(dict(seconds=seconds, end=time.time(), final=final))
        print(json.dumps(dict(mode=args.mode, window=len(windows), seconds=seconds,
                              games=sum(pool.completed for pool in pools),
                              placements=sum(pool.placements+pool.moves() for pool in pools))), flush=True)
        phase('done' if final else 'idle')
        if final:
            return
        idle = time.time()
        while True:
            elapsed = time.time()-idle
            used, free, total = memory()
            if elapsed >= 60 and used <= 7400 and free >= .12*total+384:
                break
            time.sleep(min(10, max(1, 60-elapsed)) if elapsed < 60 else 10)
        active_start = time.perf_counter()
        phase('active')

    class MeasuredPool(original_pool):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.completed, self.placements, self.errors = 0, 0, 0
            self.occupancy, self.running_seconds, self.final_solver = 0., 0., None
            pools.append(self)

        def close(self):
            solver = self.engine.solver
            super().close()
            if solver is not None:
                solver.drain()
                self.final_solver = solver.summary(time.perf_counter()-self.started)

        def step(self):
            boundary()
            tick = time.perf_counter()
            held = self.running()
            result = super().step()
            seconds = time.perf_counter()-tick
            self.occupancy += held*seconds
            self.running_seconds += seconds
            for _, record in result:
                self.errors += 'error' in record
                self.completed += 1
                self.placements += record['plies']-len(record['opening'])
            return result

    class Evaluation(dense_eval.Evaluator):
        def weights(self, name):
            return Path(inputs['checkpoints'][name]['path'])

        def pipeline_ready(self):
            # The queue was chosen from frozen evidence. New live exports and
            # wall-clock book refreshes must not change this comparison's work.
            return True

        def use(self, *names):
            super().use(*names)
            for name, model in self.models.items():
                predictor = model.evaluator
                if getattr(predictor, '_profiled', False):
                    continue
                original = predictor.predict

                def predict(x, original=original, name=name):
                    shapes[f'{name}/{x.shape[0]}x{x.shape[-1]}'] += 1
                    return original(x)

                predictor.predict, predictor._profiled = predict, True

    dense_solver.Pool.submit = submit
    dense_eval.Pool = MeasuredPool
    evaluator = Evaluation(target, config, settings, dense_eval.Pacer(1.))
    tasks = inputs['tasks']
    targets = {(task['a'], task['b'], task['kind']): task['before']+task['games'] for task in tasks}

    def want(lane):
        left = targets[lane]-len(evaluator.games(*lane[:2]))
        return {lane: dense_eval.even(min(settings.pool_games, left))} if left > 0 else {}

    try:
        if args.mode == 'pipeline':
            first = next(iter(targets))

            def auxiliary(excluded, names):
                for lane, goal in targets.items():
                    if lane[:2] not in excluded and len(names | set(lane[:2])) <= 3:
                        return evaluator.entry(lane[0]), lane[1], lane[2], goal-len(evaluator.games(*lane[:2]))
                return None

            evaluator.session(lambda: want(first), targets[first], auxiliary=auxiliary)
        else:
            for lane, goal in targets.items():
                evaluator.session(lambda lane=lane: want(lane), goal)
        boundary(final=True)
        records = {}
        for task in tasks:
            lane = task['a'], task['b'], task['kind']
            games = evaluator.games(*lane[:2])[task['before']:]
            pairs = {}
            for game in games:
                pairs.setdefault(game['pair'], []).append(game)
            if len(games) != task['games'] or any(sorted(g['challenger_color'] for g in pair) != [0, 1]
                                                or pair[0]['opening'] != pair[1]['opening'] for pair in pairs.values()):
                raise AssertionError(f'Incomplete or mismatched colour pairs: {lane}')
            records['/'.join(lane)] = dict(games=len(games), pairs=len(pairs),
                 placements=dense_eval.placements(games), capped=sum(g['winner'] < 0 for g in games),
                 reasons=dict(Counter(g['reason'] for g in games)),
                 moves_sha256=__import__('hashlib').sha256(json.dumps(sorted(games, key=lambda g:
                     (g['pair'], g['challenger_color'])), sort_keys=True).encode()).hexdigest())
        active_seconds = sum(window['seconds'] for window in windows)
        wall_seconds = time.time()-wall_start
        placements = sum(record['placements'] for record in records.values())
        report = dict(mode=args.mode, input_sha256=sha(args.output/'inputs.json'), settings=asdict(settings),
                      wall_seconds=wall_seconds, active_seconds=active_seconds,
                      elapsed_placements_per_second=placements/wall_seconds,
                      segmented_placements_per_second=placements/active_seconds,
                      segmented_games_per_hour=sum(record['games'] for record in records.values())*3600/active_seconds,
                      windows=windows, records=records, forward_shapes=dict(shapes),
                      mean_running=sum(p.occupancy for p in pools)/sum(p.running_seconds for p in pools),
                      neural_evals=sum(p.engine.evals for p in pools), calls=sum(p.engine.calls for p in pools),
                      errors=sum(p.errors for p in pools),
                      solver=[p.final_solver for p in pools],
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        write(args.output/f'{args.mode}.json', report)
        print(json.dumps(report), flush=True)
    finally:
        for pool in pools:
            pool.close()
        dense_solver.Pool.submit, dense_eval.Pool = original_submit, original_pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['reference', 'pipeline'])
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT/'artifacts'):
        raise ValueError('Write benchmark copies only under this worktree/artifacts')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == 'prepare':
        prepare(args)
    elif args.worker:
        run(args)
    else:
        raise SystemExit(guard(args))


if __name__ == '__main__':
    main()
