"""Build the proof library or benchmark saved puzzles with the independent checker.

Run after tools/build_tactical.py: PYTHONPATH=python;. python tools/proof_stamps.py --out PATH
The output contains strategies, not imported shape verdicts. Native code checks them again before use.

--puzzles puzzles.txt --benchmark on.json --nodes 2000000 --ms 45000 compares
cold solver queries; repeat with --no-stamps and another output. Add --engine
ema.pt --simulations 2048 to run the full CPU player, or --cases 1 2 6 for a subset.
The engine mode also needs the native rules and search libraries built.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from tactical_proof import NativeTactics, independent_verify


SHAPES = {'line': [[1, 0], [2, 0]], 'triangle': [[1, 0], [0, 1]], 'chevron': [[1, 0], [0, -1]],
          'diamond': [[1, 0], [0, 1], [1, 1]]}


def build():
    engine, library = NativeTactics(), []
    for name, shape in SHAPES.items():
        history = [[0, 0], [0, 8], [8, 0], *shape[:2], [-8, 0], [0, -8]]
        if len(shape) == 3:
            history += [shape[2], [12, -8]]
        attacker = 'opponent' if len(shape) == 3 else 'mover'
        result = engine.history(history, nodes=100000, ms=10000, attacker=attacker)
        if result['status'] != 'PROVEN_WIN':
            raise ValueError(f"{name}: {result['reason']}")
        independent_verify(result['certificate'], history, attacker=attacker)
        engine.history(history, certificate=result['certificate'], nodes=100000, ms=10000,
                       stamps=True, library=[], attacker=attacker)
        reused = engine.history(history, nodes=1, ms=10000, stamps=True, library=[], attacker=attacker)
        node = reused['certificate']['nodes'][reused['certificate']['root']]
        if node['kind'] != 'stamp':
            raise ValueError(f'{name}: no checked local strategy')
        independent_verify(reused['certificate'], history, attacker=attacker)
        library.append(dict(name=name, source=node['source']))
        print(f"{name}: {len(node['source']['stones'])} supporting stones, {reused['proof_turns']} turns")
    return library


def benchmark():
    """Same-position CPU queries on the saved real-game fixtures, plus local reuse.

    Start this command in a fresh process. Cases share the bounded learned
    library, as a worker does. Wall times include verification and Python/native
    serialization. The reference result cache is isolated from the stamp bank.
    """
    fixture = json.loads((Path(__file__).resolve().parents[1]/'tests/fixtures/tactical_positions.json').read_text())
    output = []
    def query(engine, name, history, stamps, nodes=8192, **options):
        start = time.perf_counter()
        result = engine.history(history, stamps=stamps, nodes=nodes, ms=10000, **options)
        if result['reason'] in ('deadline', 'native worker busy finishing bounded prior query'):
            raise RuntimeError(f"{name}: benchmark did not finish its query: {result['reason']}")
        output.append(dict(case=name, stamps=stamps, budget=nodes, ms=(time.perf_counter()-start)*1000,
                           **{k: result[k] for k in ('status', 'nodes_fresh', 'native_verified', 'reason')},
                           hits=result.get('stamp_hits', 0), bytes=result.get('stamp_bytes', 0)))
        return result
    def real():
        engine = NativeTactics()
        for name, history in fixture['positions'].items():
            for enabled in (False, True):
                query(engine, name, history, enabled)
    def changed():
        engine = NativeTactics()
        for name in fixture['proving_nodes']:
            history = fixture['positions'][name]
            source = engine.history(history, nodes=50000, ms=3000)
            if source['status'] != 'PROVEN_WIN':
                continue
            engine.history(history, certificate=source['certificate'], stamps=True, nodes=50000, ms=3000)
            q, r = max(history)
            after = history + [[q+6,r], [q+12,r], [q+12,r+6], [q+6,r+6]]
            for enabled in (False, True):
                query(engine, name+'/four-remote-stones', after, enabled)
    def local():
        engine = NativeTactics()
        history = [[0,0],[0,8],[2,8],[1,0],[2,0],[4,8],[6,8]]
        query(engine, 'line/library-query', history, True)
        after = history + [[8,8],[10,8],[8,10],[10,10]]
        for enabled in (False, True):
            query(engine, 'line/four-remote-stones', after, enabled, nodes=50000)
    for group in (real, changed, local):
        group()
    return dict(source=fixture['source'], nodes=8192, ms=10000, rows=output)


def load_puzzles(path):
    """Concatenated HTTTX records, numbered in file order; mover is expected to win."""
    from hexo import Game
    from notation import loads
    raw = path.read_bytes()
    parts = re.split(r'(?=version\s*\[\s*1\s*\]\s*;)', raw.decode('utf-8-sig'))
    if parts[0].strip() or len(parts) == 1:
        raise ValueError('Expected one or more version[1]; HTTTX records')
    cases = []
    for number, text in enumerate(parts[1:], 1):
        record = loads(text)
        game = Game(record.history)
        try:
            if game.winner != -1:
                raise ValueError(f'Puzzle {number} is already terminal')
            state = json.dumps([sorted(game.cells), game.player, game.remaining], separators=(',', ':'))
            cases.append(dict(id=number, history=record.history, mover=game.player,
                              remaining=game.remaining, expected='PROVEN_WIN',
                              position_sha256=hashlib.sha256(state.encode()).hexdigest()))
        finally:
            game.close()
    return dict(source=path.name, input_sha256=hashlib.sha256(raw).hexdigest(), cases=cases)


def puzzle_benchmark(source, target, nodes, ms, stamps):
    """Cold independent workers, fixed budgets, checked wins and portable inputs.

    Search timing includes native checking and response decoding. The second,
    Python proof check is recorded separately. Unknowns remain in every summary.
    Write after each puzzle so an interrupted benchmark keeps completed results.
    """
    report = load_puzzles(source)
    report.update(nodes=nodes, ms=ms, stamps=stamps, cache='cold per puzzle', rows=[])
    for case in report['cases']:
        with NativeTactics(independent=True) as engine:
            start = time.perf_counter()
            result = engine.history(case['history'], nodes=nodes, ms=ms, depth=64, stamps=stamps)
            elapsed = (time.perf_counter()-start)*1000
        row = dict(id=case['id'], ms=elapsed, **{k: result.get(k) for k in
                   ('status', 'reason', 'nodes_fresh', 'proof_turns', 'stamp_hits', 'native_verified')})
        row['verification_ms'] = 0.
        if result['status'] == 'PROVEN_WIN':
            start = time.perf_counter()
            independent_verify(result['certificate'], case['history'], deadline_seconds=30.)
            row['verification_ms'] = (time.perf_counter()-start)*1000
        report['build_hash'] = result['build_hash']
        report['rows'].append(row)
        report['summary'] = dict(completed=len(report['rows']), total=len(report['cases']),
                               solved=sum(r['status']=='PROVEN_WIN' for r in report['rows']),
                               ms=sum(r['ms'] for r in report['rows']),
                               nodes=sum(r['nodes_fresh'] or 0 for r in report['rows']))
        target.write_text(json.dumps(report, separators=(',', ':'))+'\n', encoding='utf-8')
        print(json.dumps(row), flush=True)
    return report


def engine_benchmark(source, target, checkpoint, simulations, nodes, ms, stamps, case_ids=()):
    """The player's complete CPU analysis: root queries, graph search and reply checks.

    One resident engine reads the archive in order. Each puzzle has its own game
    graph; the network cache and bounded stamp library survive between puzzles.
    Verification of returned solver certificates is outside the timed analysis.
    """
    import torch
    from play import Engines
    from hexo import library
    from neural_search import native
    torch.set_num_threads(2)
    report = load_puzzles(source)
    cases = [c for c in report['cases'] if not case_ids or c['id'] in case_ids]
    if case_ids and set(case_ids) != {c['id'] for c in cases}:
        raise ValueError('Unknown puzzle number in --cases')
    report.update(checkpoint=str(checkpoint.resolve()), simulations=simulations, nodes=nodes, ms=ms,
                  stamps=stamps, device='cpu', threads=2, case_ids=[c['id'] for c in cases],
                  cache='resident engine; one graph per puzzle', rows=[])
    report['libraries'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (library, Path(native._name))}
    engines = Engines('cpu', proof_stamps=stamps)
    try:
        start = time.perf_counter()
        bubble = engines.bubble(checkpoint)
        prover, build = engines.solver()
        if prover is None:
            raise RuntimeError('The engine benchmark requires a built tactical solver')
        report.update(model_sha256=bubble.sha256, build_hash=build, load_ms=1000*(time.perf_counter()-start))
        report['summary'] = dict(completed=0, total=len(cases), solved=0, ms=0., nodes=0)
        target.write_text(json.dumps(report, separators=(',', ':'))+'\n', encoding='utf-8')
        original, pending = prover.history, []
        def query(history, **options):
            options['ms'] = min(options.get('ms', ms), ms)
            start = time.perf_counter()
            result = original(history, **options)
            pending.append((list(history), options, result, 1000*(time.perf_counter()-start)))
            return result
        prover.history = query
        budget = dict(simulations=simulations, solver_nodes=nodes)
        for case in cases:
            pending.clear()
            start = time.perf_counter()
            result, spent, identity = engines.evaluate(dict(path=checkpoint), '', budget, case['history'],
                                                       lambda n: None, game=case['id'])
            elapsed = 1000*(time.perf_counter()-start)
            verified, verification_ms, queries = 0, 0., []
            for history, options, answer, took in pending:
                queries.append(dict(attacker=options.get('attacker', 'mover'), ms=took,
                                    **{k: answer.get(k) for k in ('status', 'reason', 'nodes_fresh',
                                       'proof_turns', 'shortest', 'stamp_hits', 'native_verified')}))
                certificate = answer.get('certificate')
                if certificate is None and answer.get('certificate_json'):
                    certificate = json.loads(answer['certificate_json'])
                if answer.get('native_verified') and certificate is None:
                    raise RuntimeError('Verified solver answer is missing its certificate')
                if certificate is not None:
                    start = time.perf_counter()
                    independent_verify(certificate, history, attacker=options.get('attacker', 'mover'),
                                       known=options.get('known', ()), deadline_seconds=45.)
                    verification_ms += 1000*(time.perf_counter()-start)
                    verified += 1
            proof = result.get('proof')
            row = dict(id=case['id'], ms=elapsed, queries=queries, verification_ms=verification_ms,
                       verified_certificates=verified, status=('PROVEN_WIN' if proof['winner']==case['mover'] else 'PROVEN_LOSS') if proof else 'UNKNOWN',
                       proof=proof, moves=result['moves'], value=result['value'], budget=spent, engine=identity,
                       actual_completed=result.get('actual_completed'), actual_solver_nodes=result.get('actual_solver_nodes'))
            report['rows'].append(row)
            report['summary'] = dict(completed=len(report['rows']), total=len(cases),
                                    solved=sum(r['status']=='PROVEN_WIN' for r in report['rows']),
                                    ms=sum(r['ms'] for r in report['rows']),
                                    nodes=sum(q['nodes_fresh'] or 0 for r in report['rows'] for q in r['queries']))
            target.write_text(json.dumps(report, separators=(',', ':'))+'\n', encoding='utf-8')
            print(json.dumps({k: row[k] for k in ('id', 'ms', 'status', 'actual_completed', 'actual_solver_nodes')}), flush=True)
    finally:
        engines.close()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--out', type=Path)
    mode.add_argument('--benchmark', type=Path, metavar='JSON')
    parser.add_argument('--puzzles', type=Path, help='HTTTX archive for --benchmark; mover must win')
    parser.add_argument('--nodes', type=int, default=8192)
    parser.add_argument('--ms', type=int, default=10000)
    parser.add_argument('--stamps', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--engine', type=Path, metavar='EMA.PT', help='run the complete player analysis on CPU with these weights')
    parser.add_argument('--simulations', type=int, default=2048, help='search simulations per placement with --engine')
    parser.add_argument('--cases', type=int, nargs='+', default=[], help='archive puzzle numbers for --engine (default: all)')
    args = parser.parse_args()
    if not args.puzzles and (args.nodes != 8192 or args.ms != 10000 or not args.stamps):
        parser.error('--nodes, --ms and --no-stamps require --puzzles')
    target = args.out or args.benchmark
    if args.engine and not args.puzzles:
        parser.error('--engine requires --puzzles')
    if args.cases and not args.engine:
        parser.error('--cases requires --engine')
    if args.simulations < 1:
        parser.error('--simulations must be positive')
    if args.puzzles:
        if not args.benchmark:
            parser.error('--puzzles requires --benchmark')
        result = (engine_benchmark(args.puzzles, target, args.engine, args.simulations, args.nodes, args.ms, args.stamps, args.cases)
                  if args.engine else puzzle_benchmark(args.puzzles, target, args.nodes, args.ms, args.stamps))
        print(json.dumps(result['summary']))
    else:
        target.write_text(json.dumps(build() if args.out else benchmark(), separators=(',', ':'))+'\n', encoding='utf-8')
