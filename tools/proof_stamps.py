"""Build the proof library or benchmark saved puzzles with the independent checker.

Run after tools/build_tactical.py: PYTHONPATH=python;. python tools/proof_stamps.py --out PATH
The output contains strategies, not imported shape verdicts. Native code checks them again before use.
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--out', type=Path)
    mode.add_argument('--benchmark', type=Path, metavar='JSON')
    parser.add_argument('--puzzles', type=Path, help='HTTTX archive for --benchmark; mover must win')
    parser.add_argument('--nodes', type=int, default=8192)
    parser.add_argument('--ms', type=int, default=10000)
    parser.add_argument('--stamps', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not args.puzzles and (args.nodes != 8192 or args.ms != 10000 or not args.stamps):
        parser.error('--nodes, --ms and --no-stamps require --puzzles')
    target = args.out or args.benchmark
    if args.puzzles:
        if not args.benchmark:
            parser.error('--puzzles requires --benchmark')
        result = puzzle_benchmark(args.puzzles, target, args.nodes, args.ms, args.stamps)
        print(json.dumps(result['summary']))
    else:
        target.write_text(json.dumps(build() if args.out else benchmark(), separators=(',', ':'))+'\n', encoding='utf-8')
