"""Benchmark real evaluator placements from frozen replay prefixes.

Prepare on CPU, then run one guarded live mode at a time:
  python tools/profile_hexnet.py prepare --run /path/to/runs/dense-v1
  python tools/profile_evaluator.py prepare --run /path/to/runs/dense-v1
  python tools/profile_evaluator.py live --kernels reference
  python tools/profile_evaluator.py live --kernels fused
"""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.profile_hexnet import guard


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf-8')


def prepare(args):
    import dense_data
    import dense_config

    if args.run is None:
        raise ValueError('prepare requires --run')
    run = args.run.resolve()
    config = dense_config.load(run)
    status_path = run/'evaluator-status.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    settings = dense_config.section('evaluation', status['settings'])
    source_path = args.references.resolve()
    source = json.loads(source_path.read_text(encoding='utf-8'))
    expected = {item['shard']: item['manifest'] for item in source.get('source_shards', [])}
    checkpoints = {name: run/'checkpoints'/'main'/name/'ema.pt' for name in ('085000', '075000')}
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    shards = {}
    prefixes = []
    seen = set()
    for ref in source['references']:
        name, index = ref['shard'], ref['row']
        if name not in shards:
            path = run/'shards'/name
            episodes, rows = dense_data.read_shard(path, policies=False)
            if name in expected and dense_data.manifest(path) != expected[name]:
                raise ValueError(f'Source shard manifest changed: {name}')
            shards[name] = episodes, rows
        episodes, rows = shards[name]
        row = rows[index]
        ply = int(row['ply'])
        key = name, row['game'], ply
        if key in seen or not 24 <= ply <= min(180, settings.max_plies-2):
            continue
        moves = episodes[row['game']]['moves'][:ply]
        if len(moves) != ply:
            raise ValueError(f'Invalid replay prefix: {key}')
        prefixes.append(dict(opening=moves, shard=name, game=row['game'], ply=ply, row=index))
        seen.add(key)
        if len(prefixes) == 32:
            break
    if len(prefixes) != 32:
        raise ValueError(f'Only {len(prefixes)} eligible midgame prefixes in inputs.json')
    payload = dict(schema='hexo-evaluator-profile-v1', run=str(run), seed=config.seed,
                   settings=asdict(settings), actor_leaf_batch=config.actor.leaf_batch,
                   checkpoint_sha256={name: sha(path) for name, path in checkpoints.items()},
                   checkpoint_paths={name: str(path) for name, path in checkpoints.items()},
                   run_config_sha256=sha(run/'config.json'), evaluator_status_sha256=sha(status_path),
                   inputs_sha256=sha(source_path),
                   source_shards={name: sha(run/'shards'/name/'manifest.json') for name in shards},
                   prefixes=prefixes)
    path = args.output/'prefixes.json'
    write(path, payload)
    print(json.dumps(dict(prefixes=len(prefixes), input_sha256=sha(path), output=str(path))))


def live(args):
    import torch
    import dense_config
    import dense_eval
    import dense_selfplay
    from dense_solver import Budgets, Schedule

    torch.set_num_threads(2)
    path = args.output/'prefixes.json'
    frozen = json.loads(path.read_text(encoding='utf-8'))
    if frozen['schema'] != 'hexo-evaluator-profile-v1' or len(frozen['prefixes']) != 32:
        raise ValueError('Prepare 32 frozen prefixes first')
    run = Path(frozen['run'])
    for name, checkpoint in frozen['checkpoint_paths'].items():
        if sha(checkpoint) != frozen['checkpoint_sha256'][name]:
            raise ValueError(f'Checkpoint changed: {checkpoint}')
    if sha(run/'config.json') != frozen['run_config_sha256']:
        raise ValueError('Run configuration changed since preparation')
    config = dense_eval.kernel_config(dense_config.load(run), args.kernels)
    settings = dense_config.section('evaluation', frozen['settings'])
    if config.actor.leaf_batch != frozen['actor_leaf_batch'] or settings.pool_games != 64:
        raise ValueError('Frozen evaluator settings do not match the 64-slot benchmark')

    # The guard owns the GPU lock and waits for card headroom.
    # Its 55-second GPU timer begins at this marker, after all CPU checks.
    marker = args.output/f'gpu-start-{os.getpid()}.txt'
    marker.write_text(str(time.time()), encoding='utf-8')
    gpu_start = time.perf_counter()
    torch.cuda.set_per_process_memory_fraction(.12)
    report = dict(stage='loading', kernels=args.kernels, input_sha256=sha(path),
                  run=str(run), settings=asdict(settings), checkpoint_sha256=frozen['checkpoint_sha256'],
                  gpu=torch.cuda.get_device_name())
    result_path = args.output/f'evaluator-{args.kernels}.json'
    write(result_path, report)
    pool = None
    original_searched = dense_eval.MatchGame.searched
    models = []
    phase = 'warmup'
    placements = 0
    shapes = Counter()
    generations = [0]*64
    lane = ('main/085000', 'main/075000', 'profile')
    try:
        a = dense_selfplay.load(run, config, source=('main/085000', Path(frozen['checkpoint_paths']['085000'])))
        b = dense_selfplay.load(run, config, source=('main/075000', Path(frozen['checkpoint_paths']['075000'])))
        models = [a, b]

        for model in models:
            evaluator = model.evaluator
            predict = evaluator.predict

            def traced_predict(x, predict=predict):
                if phase == 'measure':
                    shapes[f'{x.shape[0]}x{x.shape[-1]}'] += 1
                return predict(x)

            evaluator.predict = traced_predict

        def searched(game, result):
            nonlocal placements
            before = len(game.moves)
            more = original_searched(game, result)
            if phase == 'measure':
                placements += len(game.moves)-before
            return more

        dense_eval.MatchGame.searched = searched
        budget = Budgets.of(settings)

        def game_for(slot):
            pair, colour = divmod(slot, 2)
            prefix = frozen['prefixes'][pair]
            players = [b, b]
            players[colour] = a
            generation = generations[slot]
            generations[slot] += 1
            seed = dense_eval.pair_seed(config.seed, 'profile', pair+32*generation)
            return dense_eval.MatchGame(players, prefix['opening'], seed, settings.sims,
                                        settings.root_samples, settings.tactics, settings.max_plies,
                                        dict(slot=slot, pair=pair, challenger_color=colour,
                                             opening=prefix['opening']), solvers=(budget, budget))

        pool = dense_eval.Pool(config.actor.leaf_batch, Schedule.of(settings))
        pool.add(lane, [game_for(slot) for slot in range(64)])

        def step():
            for _, record in pool.step():
                if 'error' in record:
                    raise ValueError(f"Match game failed: {record['error']}")
                pool.add(lane, [game_for(record['slot'])])

        warm_start = time.perf_counter()
        while time.perf_counter()-warm_start < 8.:
            step()
        torch.cuda.synchronize()
        engine = pool.engine
        before_evals, before_calls, before_hits = engine.evals, engine.calls, engine.hits
        solver_before = None if engine.solver is None else {
            key: (sum(p[key] for p in engine.solver.stats['points'].values()) if key in ('queries', 'solver_ms')
                  else engine.solver.stats[key]) for key in ('queries', 'solver_ms', 'wait_ms')}
        torch.cuda.reset_peak_memory_stats()
        phase = 'measure'
        measured = time.perf_counter()
        while time.perf_counter()-measured < 20.:
            step()
        torch.cuda.synchronize()
        seconds = time.perf_counter()-measured
        phase = 'done'
        solver_after = None if engine.solver is None else engine.solver.summary(time.perf_counter()-pool.started)
        solver_delta = None if solver_before is None else {
            'queries': solver_after['queries']-solver_before['queries'],
            'backend_ms': sum(p['solver_ms'] for p in engine.solver.stats['points'].values())-solver_before['solver_ms'],
            'wait_ms': engine.solver.stats['wait_ms']-solver_before['wait_ms']}
        evals, calls = engine.evals-before_evals, engine.calls-before_calls
        report.update(stage='pass', seconds=seconds, warmup_seconds=measured-warm_start,
                      placements=placements, placements_per_second=placements/seconds,
                      neural_evals=evals, neural_evals_per_second=evals/seconds,
                      batch_calls=calls, mean_batch=evals/calls if calls else None,
                      cache_hits=engine.hits-before_hits, forward_shapes=dict(sorted(shapes.items())),
                      solver=solver_delta, solver_cumulative=solver_after,
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                      gpu_seconds=time.perf_counter()-gpu_start)
        write(result_path, report)
        print(json.dumps(dict(kernels=args.kernels, placements_per_second=report['placements_per_second'],
                              neural_evals_per_second=report['neural_evals_per_second'],
                              mean_batch=report['mean_batch'])), flush=True)
    except Exception as error:
        report.update(stage='failed', error=f'{type(error).__name__}: {error}')
        write(result_path, report)
        raise
    finally:
        phase = 'done'
        dense_eval.MatchGame.searched = original_searched
        if pool is not None:
            try:
                pool.close()
            finally:
                for _, game in list(pool.games.values()):
                    game.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'live'))
    parser.add_argument('--run', type=Path)
    parser.add_argument('--references', type=Path, default=ROOT/'artifacts/gpu-kernels/inputs.json')
    parser.add_argument('--kernels', choices=('reference', 'fused'), default='reference')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/evaluator-profile')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT/'artifacts'):
        parser.error('--output must be within the worktree artifacts directory')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == 'prepare':
        return prepare(args)
    if args.worker:
        return live(args)
    return guard(args, entry=Path(__file__).resolve())


if __name__ == '__main__':
    sys.exit(main())
