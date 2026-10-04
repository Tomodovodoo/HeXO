"""Build the small proof library with our solver and the independent Python checker.

Run after tools/build_tactical.py: PYTHONPATH=python;. python tools/proof_stamps.py --out PATH
The output contains strategies, not imported shape verdicts. Native code checks them again before use.
"""
import argparse
import json
from pathlib import Path
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(build(), separators=(',', ':'))+'\n', encoding='utf-8')
