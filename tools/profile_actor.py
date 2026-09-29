"""Bounded actor cold-start measurement, without writing to a run.

prepare reads the PR 174 snapshot's shard references and the run configuration.
actor keeps one worker and its games alive across 30-second windows, separated
by at least 60 seconds without GPU work. The parent enforces a 55-second limit
on each window and waits above 7400 MiB card usage. Outputs stay in artifacts.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cache_report(cache):
    counts, shapes, size = Counter(), [], 0
    for path in cache.rglob('*'):
        if path.is_file():
            size += path.stat().st_size
        if path.suffix == '.json' and not path.name.startswith('__grp'):
            counts[json.loads(path.read_text()).get('name', 'other')] += 1
        if path.name == '_windows.ttir':
            source = path.read_text()
            constants = dict(re.findall(r'(%\w+) = arith.constant dense<(\d+)> : tensor<256xi32>', source))
            area = re.search(r'arith.divsi %\w+, (%\w+)', source)
            bound = re.search(r'arith.cmpi slt, %i_\d+, (%\w+)', source)
            if area and bound and area[1] in constants and bound[1] in constants:
                cells = int(constants[area[1]])
                shapes.append((int(constants[bound[1]])//cells, int(cells**.5)))
    return dict(entries=len(list(cache.iterdir())), bytes=size, kernels=dict(counts),
                distinct_shapes=len(set(shapes)), shapes=sorted(set(shapes)),
                by_canvas=dict(Counter(s for b, s in shapes)))


def prepare(args):
    import dense_config
    import dense_data
    config = dense_config.load(args.run)
    snapshot = json.loads((args.output/'inputs.json').read_text())
    manifest = json.loads((args.run/'shards'/snapshot['shards'][0]/'manifest.json').read_text())
    settings = dense_config.section('actor', manifest['identity']['actor'])
    config = replace(config, actor=dense_config.override(settings, args))
    shards = {name: dense_data.read_shard(args.run/'shards'/name, policies=False)
              for name in snapshot['shards']}
    histories = []
    for ref in snapshot['references']:
        episodes, rows = shards[ref['shard']]
        row = rows[ref['row']]
        histories.append(episodes[row['game']]['moves'][:row['ply']])
    report = dict(config=asdict(config), histories=histories, source=snapshot)
    (args.output/'actor-inputs.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(histories=len(histories), actor=asdict(config.actor))))


def worker(args):
    # Nothing CUDA-related may precede the allocator limit.
    import torch
    torch.cuda.set_per_process_memory_fraction(.12)
    torch.set_num_threads(2)
    import dense_config
    import dense_selfplay as actor
    import dense_solver
    import hexnet
    source = json.loads((args.output/'actor-inputs.json').read_text())
    settings = dense_config.from_dict(source['config']).actor
    engine, model, games = None, None, set()
    next_game, placements = 0, 0
    shapes = Counter()

    def add_game():
        nonlocal next_game
        history = source['histories'][next_game % len(source['histories'])]
        game = actor.SelfPlayGame([model, model], settings, source['config']['seed']+next_game,
                                 restart=({'ply': len(history)}, history))
        original = game.searched

        def searched(result):
            nonlocal placements
            before = len(game.moves)
            alive = original(result)
            placements += len(game.moves)-before
            return alive

        game.searched = searched
        engine.add(game)
        games.add(game)
        next_game += 1

    def close_game(game):
        game.game.close()
        for tree in game.trees.values():
            tree.close()
        games.remove(game)

    try:
        for window in range(args.windows):
            if sys.stdin.readline().strip() != 'go':
                break
            start = time.perf_counter()
            before, evals, calls = placements, engine.evals if engine else 0, engine.calls if engine else 0
            if model is None:
                net = hexnet.load_model(args.output/'model.pt', net_kernels=args.kernels)
                model = actor.Model(net, 'snapshot', 'snapshot', 'cuda', settings.leaf_batch, settings.cache_positions)

                def record(module, inputs):
                    x = inputs[0]
                    shapes[tuple(x.shape)] += 1

                model.evaluator.model.register_forward_pre_hook(record)
                engine = actor.Engine(settings.leaf_batch, settings.solver_async, dense_solver.Schedule.of(settings))
                for _ in range(settings.games_in_flight):
                    add_game()
            while time.perf_counter()-start < 30:
                for game in engine.step():
                    close_game(game)
                    add_game()
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            report = dict(window=window, seconds=elapsed, placements=placements-before,
                          placements_per_second=(placements-before)/elapsed,
                          evals=engine.evals-evals, evals_per_second=(engine.evals-evals)/elapsed,
                          mean_batch=(engine.evals-evals)/max(1, engine.calls-calls),
                          shapes={','.join(map(str, k)): v for k, v in sorted(shapes.items())},
                          cache=cache_report(Path(os.environ['TRITON_CACHE_DIR'])),
                          peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            path = args.output/f'actor-{args.label}-{window}.json'
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(report, indent=2))
            temporary.replace(path)
            print(json.dumps({k:v for k,v in report.items() if k not in ('shapes','cache')}), flush=True)
    finally:
        if engine:
            engine.close()
        for game in list(games):
            close_game(game)


def guard(args):
    state = ROOT/'artifacts/gpu-kernels'
    state.mkdir(parents=True, exist_ok=True)
    stamp, lock = state/'last_gpu_end.txt', state/'gpu.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    child = None
    try:
        args.cache.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
                   TRITON_CACHE_DIR=str(args.cache.resolve()))
        for window in range(args.windows):
            while True:
                idle = 60-(time.time()-float(stamp.read_text())) if stamp.exists() else 0
                used = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used',
                                                   '--format=csv,noheader,nounits'], text=True).strip())
                if idle <= 0 and used <= 7400:
                    break
                print(json.dumps(dict(cooldown_seconds=max(0, round(idle)), gpu_used_mib=used)), flush=True)
                time.sleep(min(10, max(1, idle)))
            result = args.output/f'actor-{args.label}-{window}.json'
            if result.exists():
                raise FileExistsError(f'Choose a new --label: {result}')
            started = time.time()
            if child is None:
                child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
                                         cwd=ROOT, env=env, stdin=subprocess.PIPE, text=True,
                                         creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 0)
            child.stdin.write('go\n')
            child.stdin.flush()
            try:
                while not result.exists():
                    if child.poll() is not None:
                        raise RuntimeError(f'actor exited: {child.returncode}')
                    if time.time()-started >= 55:
                        raise TimeoutError('actor window exceeded 55 seconds')
                    time.sleep(.1)
            finally:
                ended = time.time()
                stamp.write_text(str(ended))
                with (state/'sessions.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(start=started, end=ended, elapsed=ended-started,
                                                 label=args.label, window=window, gpu_used_mib=used))+'\n')
        child.stdin.close()
        child.wait(timeout=15)
        stamp.write_text(str(time.time()))
        if child.returncode:
            raise RuntimeError(f'actor exited: {child.returncode}')
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()
            stamp.write_text(str(time.time()))
        lock.unlink()


def main():
    import dense_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['cache','prepare','actor'])
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/gpu-kernels')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--cache', type=Path, default=Path.home()/'.triton/cache')
    parser.add_argument('--kernels', choices=['reference','fused'], default='fused')
    parser.add_argument('--label', default='cold')
    parser.add_argument('--windows', type=int, default=10)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    dense_config.add_arguments(parser, dense_config.ActorSettings)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT/'artifacts'):
        parser.error('output must be under this worktree\'s artifacts directory')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'cache':
        report = cache_report(args.cache)
        (args.output/'cache-audit.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
    elif args.mode == 'prepare':
        prepare(args)
    elif args.worker:
        worker(args)
    else:
        guard(args)


if __name__ == '__main__':
    main()
