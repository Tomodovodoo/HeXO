"""Frozen direct relational players versus pinned Seal, with paired openings."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evaluate import sha, loaded_modules, verify_runtime, verify_trace, publish_failure
from hexo import Game, ROOT, library
from train import paired_metrics, task_opening, write_json

PLAYER = OPPONENT = None


def create_player(config, checkpoint):
    from relational_player import RelationalPlayer
    return RelationalPlayer(checkpoint, mode=config['mode'], device=config['device'],
        simulations=config['simulations'], root_samples=config['root_samples'], batch_size=config['batch_size'],
        milliseconds=config['neural_ms'], max_nodes=config['max_nodes'], max_edges=config['max_edges'], seed=config['seed'],
        proof_ms=config['proof_ms'])


def prepare(config, checkpoint):
    global PLAYER, OPPONENT
    from arena import Seal
    PLAYER = create_player(config, checkpoint)
    OPPONENT = Seal()
    # Warm the actual model and CUDA kernels before binding loaded dependencies.
    PLAYER.evaluator.evaluate([[(0, 0)]])


def worker_start(output):
    provenance = json.loads((output/'provenance.json').read_text())
    prepare(provenance['config'], output/'models/candidate.pt')
    verify_runtime(provenance['runtime_files_sha256'])


def heartbeat(output, index, phase, **extra):
    write_json(output/'workers'/f'{index:04d}.json', dict(index=index, phase=phase, heartbeat=time.time(), **extra))


def play(task):
    game = Game(task['opening'])
    PLAYER.set_history(task['opening'])
    trace = []
    output = Path(task['output'])
    start = time.perf_counter()
    def record():
        return dict(index=task['index'], pair=task['index']//2, seed=task['seed'],
            opening=task['opening'], family=task['family'], challenger_color=task['challenger_color'],
            winner=game.winner, reason='six-in-a-row' if game.winner >= 0 else 'truncated',
            cells=game.cells, search_trace=trace, seconds=time.perf_counter()-start)
    try:
        while game.winner < 0 and len(game.cells)+game.remaining <= task['max_stones']:
            side, remaining, ply = game.player, game.remaining, len(game.cells)
            heartbeat(output, task['index'], 'neural' if side == task['challenger_color'] else 'seal', ply=ply)
            turn_start = time.perf_counter()
            if side == task['challenger_color']:
                result = PLAYER.turn(game)
            else:
                result = dict(moves=OPPONENT(game, task['seal_ms']), backend='seal')
            wall = (time.perf_counter()-turn_start)*1000
            if not result['moves']:
                raise RuntimeError('Player returned an empty turn')
            trace.append(dict(ply=ply, player=side, remaining=remaining, wall_ms=wall, result=result))
            for action in result['moves']:
                if game.winner >= 0 or game.player != side:
                    raise ValueError('Player continued after its turn or a first-stone win')
                game.play(*action)
            if game.winner < 0 and game.player == side:
                raise ValueError('Player did not finish its complete turn')
        result = record()
        verify_trace(result)
        result['replay_verified'] = True
        heartbeat(output, task['index'], 'finished', ply=len(game.cells))
        return result
    except Exception as error:
        write_json(output/f'failed-game-{task["index"]:04d}.json', {**record(), 'error': repr(error)})
        heartbeat(output, task['index'], 'failed', error=repr(error))
        raise
    finally:
        game.close()


def freeze(args):
    if args.output.exists():
        raise ValueError('Output must be a new directory')
    if args.games < 2 or args.games % 2 or min(args.seal_ms, args.neural_ms, args.simulations, args.root_samples, args.batch_size) < 1 or args.max_stones < 5:
        raise ValueError('Even paired game count, positive budgets and cap >=5 required')
    output = args.output.resolve()
    checkpoint = args.checkpoint.resolve()
    config = {name: str(value.resolve()) if isinstance(value, Path) else value
              for name, value in vars(args).items() if name != 'execute_snapshot'}
    source_names = ['relational_evaluate.py', 'relational_player.py', 'evaluate.py', 'train.py', 'hexo.py',
        'curriculum.py', 'relational_train.py', 'relational_model.py', 'relational_encoder.py',
        'relational_native.py', 'klent.py', 'nnue_model.py', 'neural_search.py', 'arena.py',
        'CMakeLists.txt', 'tools/seal_adapter.cpp'] + [str(p.relative_to(ROOT)) for p in (ROOT/'src').glob('*') if p.is_file()]
    binaries = [library, library.with_name(library.name.replace('hexo', 'hexo_graph')),
                library.with_name(library.name.replace('hexo', 'hexo_gumbel'))]
    if args.mode == 'gumbel-proof':
        from tactical_proof import NativeTactics, PACKAGE
        proof = NativeTactics()
        source_names += ['tactical_proof.py', 'proof.py', 'tools/build_tactical.py']
        source_names += [str((PACKAGE/name).relative_to(ROOT)) for name in proof.metadata['sources']]
        binary = Path(proof.lib._name)
        binaries += [binary, binary.with_suffix(binary.suffix+'.json')]
    revision_file = args.seal_library.parent/'seal_revision.txt'
    if not revision_file.exists() and args.seal_library.parent.name in ('Release', 'Debug', 'RelWithDebInfo', 'MinSizeRel'):
        revision_file = args.seal_library.parent.parent/'seal_revision.txt'
    # Validate all inputs before creating the immutable destination.
    for path in [checkpoint, args.seal_library, revision_file, *binaries, *(ROOT/name for name in source_names)]:
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True)
    (output/'models').mkdir()
    (output/'workers').mkdir()
    shutil.copyfile(checkpoint, output/'models/candidate.pt')
    for name in source_names:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    for binary in binaries:
        target = output/'source'/binary.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(binary, target)
    seal = output/'source'/library.relative_to(ROOT).with_name(library.name.replace('hexo', 'hexo_seal'))
    shutil.copyfile(args.seal_library, seal)
    # Capture runtime from the copied tree after warm-up, not from a different host module path.
    config['seal_revision'] = revision_file.read_text().strip()
    identity = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
    write_json(output/'provenance.json', dict(schema='hexo-relational-evaluation-v1', config=config,
        files_sha256=identity, model_input_sha256={'candidate': sha(checkpoint), 'reference': sha(seal)},
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
        model_family='relational-policy-q', backend=args.mode, promotion=False,
        budget_comparison='unequal: neural simulations per placement with cooperative turn deadline; Seal milliseconds per turn'))
    return output


def execute(output):
    global PLAYER, OPPONENT
    provenance = json.loads((output/'provenance.json').read_text())
    config = provenance['config']
    def check_files():
        if any(sha(output/name) != value for name, value in provenance['files_sha256'].items()):
            raise ValueError('Frozen model/source/binary changed')
    check_files()
    prepare(config, output/'models/candidate.pt')
    copied = {p.resolve() for p in (output/'source').rglob('*') if p.is_file()}
    if 'runtime_files_sha256' in provenance:
        verify_runtime(provenance['runtime_files_sha256'])
    runtime = provenance.get('runtime_files_sha256') or {str(path): sha(path) for path in loaded_modules() if path not in copied}
    provenance['runtime_files_sha256'] = runtime
    provenance['runtime_scope'] = 'Loaded file-backed native dependencies after actual neural warmup; verified host paths, not whole-OS isolation'
    import torch
    provenance['hardware'] = dict(device=config['device'], torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(torch.device(config['device'])) if config['device'].startswith('cuda') else None)
    write_json(output/'provenance.json', provenance)
    PLAYER.close()
    PLAYER = OPPONENT = None
    if config['device'].startswith('cuda'):
        torch.cuda.empty_cache()
    records, started = [], time.perf_counter()
    tasks = [dict(index=i, seed=config['seed']+i//2, challenger_color=i%2, output=str(output),
                  max_stones=config['max_stones'], seal_ms=config['seal_ms'],
                  **task_opening(config['seed']+i//2, True, config['max_stones'], 'mixed-v1')) for i in range(config['games'])]
    write_json(output/'openings.json', tasks)
    def publish(finished=False):
        metrics = paired_metrics(records, config['games'])
        metrics.update(provisional_elo_delta=metrics['elo_delta'], elo_delta=None, rated=False, official_rating=False,
                       rating_status='DIAGNOSTIC_UNRATED_UNEQUAL_BUDGETS')
        status = dict(stage='finished' if finished else 'relational-evaluation', phase='evaluation',
            run=output.name, model_family='relational-policy-q', backend=config['mode'],
            candidate_sha256=provenance['model_input_sha256']['candidate'],
            reference_sha256=provenance['model_input_sha256']['reference'],
            source_sha256=provenance['files_sha256'], opponent={'backend': 'seal', 'ms': config['seal_ms'], 'revision': config['seal_revision']},
            heartbeat=time.time(), workers=[json.loads(p.read_text()) for p in sorted((output/'workers').glob('*.json'))],
            last_artifact='report.json', completed=len(records), total=config['games'],
            elapsed_seconds=time.perf_counter()-started, **metrics)
        write_json(output/'report.json', dict(provenance=provenance, status=status, metrics=metrics,
            openings_sha256=sha(output/'openings.json'), games=sorted(records, key=lambda g: g['index'])))
        write_json(output/'status.json', status)
    publish()
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_start, initargs=(output,)) as pool:
        pending = {pool.submit(play, task) for task in tasks}
        try:
            while pending:
                done, pending = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
                for future in done:
                    records.append(future.result())
                    print(f'{len(records)}/{config["games"]} completed', flush=True)
                publish()
        except BaseException as error:
            publish_failure(output, error)
            for future in pending:
                future.cancel()
            # Python 3.12 has no public Executor.terminate_workers method.
            # Terminate only this executor's processes before its context waits.
            for process in pool._processes.values():
                process.terminate()
            raise
    check_files()
    verify_runtime(runtime)
    publish(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--seal-library', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', choices=('pi', 'mu', 'gumbel', 'gumbel-proof'), default='gumbel')
    parser.add_argument('--device', default='cuda')
    for name, default in [('games',4),('seed',20261003),('simulations',16),('root-samples',8),('batch-size',4),
                          ('max-nodes',12000),('max-edges',1000000),('max-stones',80),('neural-ms',10000),('seal-ms',100),('proof-ms',1000)]:
        parser.add_argument('--'+name, type=int, default=default)
    parser.add_argument('--execute-snapshot', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute_snapshot:
        try:
            execute(args.execute_snapshot)
        except (Exception, KeyboardInterrupt) as error:
            publish_failure(args.execute_snapshot, error)
            raise
    else:
        if not all((args.checkpoint, args.seal_library, args.output)):
            parser.error('--checkpoint, --seal-library and --output are required')
        output = freeze(args)
        subprocess.run([sys.executable, str(output/'source/relational_evaluate.py'), '--execute-snapshot', str(output)], check=True)


if __name__ == '__main__':
    main()
