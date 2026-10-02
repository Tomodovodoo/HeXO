"""Play pinned Gumbel/PUCT/improved-policy challenges and colour-balanced comparisons.

Run independently of dense_eval loop: newer exports cannot supersede these games.
Completed games are saved after every round; resume replays only unfinished games.
Publish finished head-to-head reports after the evaluator loads benchmark_only support.
"""
import argparse
import ctypes
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
if os.name == 'nt' and not ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000):
    raise OSError('Could not set comparison priority to BelowNormal')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
import numpy as np
import torch
import dense_config
import dense_eval
import dense_openings
import dense_selfplay
import hexnet
from hexo import Game, library
from neural_search import NeuralSearch, SearchCoordinator, native
from puct_search import PUCTSearch, search_many
from legacy.train import paired_metrics, write_json

MODES = ('gumbel', 'puct', 'policy')
SOURCE_FILES = ('python/puct_search.py', 'python/neural_search.py', 'python/dense_selfplay.py',
                'python/hexnet.py', 'python/hexcrop.py', 'python/hexnet_kernels.py',
                'python/hexnet_graphs.py', 'tools/compare_search_modes.py')


class Evaluator:
    def __init__(self, model):
        self.model, self.evaluated = model, 0

    def evaluate(self, histories):
        predictions = self.model.evaluator.evaluate(histories)
        if any(p is None for p in predictions):
            raise ValueError('Search exceeded the model crop span')
        self.evaluated += len(histories)
        return [dict(zip(('actions', 'logits', 'q'), p)) for p in predictions]


class Match:
    def __init__(self, job, models, cpuct):
        self.job, self.models = job, models
        self.history = [tuple(m) for m in job['opening']]
        self.game, self.trees = Game(self.history), {}
        for colour, (checkpoint, mode) in enumerate(job['sides']):
            model, evaluator = models[checkpoint]
            cls = PUCTSearch if mode == 'puct' else NeuralSearch
            self.trees[colour] = cls(evaluator, model.sha, self.history, job['seed']*2+colour,
                                     model.cache, tactics=True, graph=True,
                                     **(dict(cpuct=cpuct) if mode == 'puct' else {}))

    def close(self):
        self.game.close()
        for tree in self.trees.values():
            tree.close()

    def play(self, result):
        action = result['action']
        mode = self.job['sides'][self.game.player][1]
        if mode == 'policy' and not result['proven']:
            action = result['actions'][np.argmax(result['policy'])]
        self.game.play(*map(int, action))
        self.history.append(tuple(map(int, action)))
        for tree in self.trees.values():
            tree.advance(action)

    def record(self):
        return dict(self.job, winner=self.game.winner, reason='six-in-a-row' if self.game.winner >= 0 else 'cap',
                    moves=[list(m) for m in self.history], plies=len(self.history))


def jobs(state):
    for mode in MODES:
        for i, case in enumerate(state['cases']):
            colour = ((len(case['history'])+1)//2)%2
            sides = [[state['opponent'], 'gumbel'] for _ in (0, 1)]
            sides[colour] = [state['checkpoint'], mode]
            yield dict(id=f'hard/{mode}/{i}', kind='hard', mode=mode, case=i, challenger_color=colour,
                       seed=20261001400+i, opening=case['history'], sides=sides)
    for a, b in itertools.combinations(MODES, 2):
        for pair, opening in enumerate(state['openings']):
            for colour in (0, 1):
                sides = [[state['checkpoint'], b] for _ in (0, 1)]
                sides[colour] = [state['checkpoint'], a]
                yield dict(id=f'{a}/{b}/{pair}/{colour}', kind='match', a=a, b=b, pair=pair,
                           challenger_color=colour, seed=state['seeds'][pair], opening=opening, sides=sides)


def publish(run, state):
    """Import archived reports and rating-only variants through the evaluator's existing request directory."""
    for checkpoint, sha in state['models'].items():
        path = run/'checkpoints'/checkpoint/'ema.pt'
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError('Target run checkpoint weights differ from the comparison')
    ids = {'gumbel': state['checkpoint'], **{m: state['checkpoint']+'@'+state['names'][m] for m in ('puct', 'policy')}}
    reports = []
    for a, b in itertools.combinations(MODES, 2):
        records = [dict(g, candidate=ids[a], opponent=ids[b]) for g in state['results']
                   if g['kind'] == 'match' and (g['a'], g['b']) == (a, b)]
        records.sort(key=lambda g: (g['pair'], g['challenger_color']))
        if len(records) != state['games']:
            raise ValueError('Publish requires all three complete head-to-head batches')
        settings = dense_config.EvaluationSettings(sims=state['sims'], root_samples=state['root_samples'],
                                                   max_plies=state['max_plies'], tactics=True, search_graph=True,
                                                   opening_suite='book', opening_book=state['book_digest'])
        overrides = {ids[m]: dict(search_choice=m) for m in (a, b)}
        report = dense_eval.make_report(ids[a], ids[b], records,
            {cid: state['models'][state['checkpoint']] for cid in ids.values()}, settings, overrides,
            report_id=state['id']+'-'+a+'-'+b)
        report.update(benchmark_only=True, search_modes=dict(a=a, b=b), cpuct=state['cpuct'],
                      source_head=state['source_head'], source_files=state['source_files'], native=state['native'], complete=True)
        path = dense_eval.report_path(run, ids[a], ids[b]).with_name('report-search-'+state['id']+'.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            old = json.loads(path.read_text(encoding='utf-8'))
            if old['id'] != report['id'] or old['games'] != report['games']:
                raise ValueError('An existing report differs')
            report = old
        else:
            write_json(path, report)
        reports.append(report)
    league = json.loads((run/'league.json').read_text(encoding='utf-8'))
    known = {v['id'] for v in league.get('variants', [])} | set(dense_eval.requests(run))
    for mode in ('puct', 'policy'):
        cid = ids[mode]
        if cid in known:
            continue
        matches = []
        for report in reports:
            if cid not in (report['candidate'], report['opponent']):
                continue
            other = report['opponent'] if cid == report['candidate'] else report['candidate']
            oriented = dense_eval.oriented(report['games'], report['candidate'], cid)
            viewed = dict(report, games=oriented, summary=dense_eval.summary(oriented), metrics=paired_metrics(oriented))
            matches.append(dense_eval.match_entry(other, viewed))
        entry = dict(id=cid, checkpoint=state['checkpoint'], base=state['checkpoint'], registered_as=cid,
                     name=state['names'][mode], settings=dict(search_choice=mode, sims=state['sims'], search_graph=True),
                     registered_at=time.time(), benchmark_only=True, benchmark=state['id'],
                     elo=None, elo_interval=None, matches=matches, verdict=dict(decision='fixed-batch-complete'))
        target = run/'variant-requests'/f'{cid.replace("/", "-")}.json'
        target.parent.mkdir(exist_ok=True)
        write_json(target, entry)
    return [r['id'] for r in reports]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', default='main/170000')
    parser.add_argument('--opponent', default='main/155000')
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--games', type=int, default=64)
    parser.add_argument('--sims', type=int, default=64)
    parser.add_argument('--root-samples', type=int, default=16)
    parser.add_argument('--max-plies', type=int, default=384)
    parser.add_argument('--cpuct', type=float, default=1.5)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.games < 2 or args.games%2:
        raise ValueError('games must be a positive colour-balanced pair count')
    native_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (library, Path(native._name))}
    source_hashes = {name: hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                     for name in SOURCE_FILES}
    if args.out.exists():
        state = json.loads(args.out.read_text(encoding='utf-8'))
        for key in ('checkpoint', 'opponent', 'games', 'sims', 'root_samples', 'max_plies', 'cpuct', 'device'):
            if state[key] != getattr(args, key):
                raise ValueError(f'{key} differs from the saved batch')
        if any(state['native'].get(name) != sha for name, sha in native_hashes.items()):
            raise ValueError('Native library differs from the saved batch')
        if not args.publish and state.get('source_files') != source_hashes:
            raise ValueError('Python search source differs from the saved batch; resume its original implementation')
    else:
        batch_id = uuid.uuid4().hex
        config = dense_config.load(args.run)
        book = dense_openings.Book(args.run, replace(config.evaluation, opening_suite='book'))
        seeds = [dense_eval.pair_seed(config.seed, 'search-mode-170000-20261002', i) for i in range(args.games//2)]
        state = dict(id=batch_id, checkpoint=args.checkpoint, opponent=args.opponent, games=args.games,
                     sims=args.sims, root_samples=args.root_samples, max_plies=args.max_plies, cpuct=args.cpuct,
                     created_at=time.time(), source_head=subprocess.check_output(
                         ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                     cases=json.loads(args.panel.read_text(encoding='utf-8'))['cases'],
                     openings=[book.draw(s) for s in seeds], seeds=seeds, book_digest=book.digest(),
                     names={m: f'{m}-{args.sims}-{batch_id[:8]}' for m in ('policy', 'puct')}, results=[], models={},
                     native=native_hashes, source_files=source_hashes,
                     external_solver=False, graph=True, tactics=True, device=args.device, complete=False)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, state)
    if args.publish:
        print(json.dumps(dict(imported=publish(args.run, state)), indent=2))
        return
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.065)
    models = {}
    for checkpoint in (args.checkpoint, args.opponent):
        path = args.run/'checkpoints'/checkpoint/'ema.pt'
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if checkpoint in state['models'] and state['models'][checkpoint] != sha:
            raise ValueError('Pinned checkpoint weights changed')
        net = hexnet.load_model(path, net_kernels='fused' if args.device == 'cuda' else 'reference')
        model = dense_selfplay.Model(net, sha, checkpoint, args.device, 32, 4096, cuda_graphs=args.device == 'cuda')
        models[checkpoint] = model, Evaluator(model)
        state['models'][checkpoint] = sha
    state.update(pid=os.getpid(), resumed_at=time.time(), complete=False)
    write_json(args.out, state)
    completed = {g['id'] for g in state['results']}
    pending = [g for g in jobs(state) if g['id'] not in completed]
    started = last = time.perf_counter()
    for start in range(0, len(pending), 32):
        batch_started = time.perf_counter()
        initial_evals = sum(e.evaluated for _, e in models.values())
        active = [Match(job, models, args.cpuct) for job in pending[start:start+32]]
        try:
            while active:
                groups = {}
                for match in active:
                    colour = match.game.player
                    checkpoint, mode = match.job['sides'][colour]
                    groups.setdefault((checkpoint, mode == 'puct'), []).append((match, match.trees[colour]))
                for (checkpoint, puct), group in groups.items():
                    model, evaluator = models[checkpoint]
                    trees = [tree for _, tree in group]
                    results = search_many(trees, args.sims) if puct else SearchCoordinator(
                        evaluator, model.sha, model.cache).search_many(trees, args.sims, args.root_samples, 32)
                    for (match, _), result in zip(group, results):
                        match.play(result)
                finished = [m for m in active if m.game.winner >= 0 or len(m.history) >= args.max_plies]
                for match in finished:
                    record = match.record()
                    state['results'].append(record)
                    print('FINISH', record['id'], 'winner', record['winner'], 'plies', record['plies'], flush=True)
                    match.close()
                    active.remove(match)
                if finished:
                    write_json(args.out, state)
                if time.perf_counter()-last >= 30:
                    last = time.perf_counter()
                    print('PROGRESS', len(state['results']), 'of', len(completed)+len(pending),
                          'evals', sum(e.evaluated for _, e in models.values()), 'seconds', round(last-started), flush=True)
        finally:
            for match in active:
                match.close()
        state.setdefault('batches', []).append(dict(jobs=[j['id'] for j in pending[start:start+32]],
            seconds=time.perf_counter()-batch_started,
            evaluations=sum(e.evaluated for _, e in models.values())-initial_evals))
        write_json(args.out, state)
    state.update(complete=True, finished_at=time.time(), seconds=time.perf_counter()-started,
                 evaluations=sum(e.evaluated for _, e in models.values()))
    write_json(args.out, state)
    print('DONE', len(state['results']), flush=True)


if __name__ == '__main__':
    main()
