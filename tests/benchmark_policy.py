"""Compare scalar and bulk extraction on frozen prefixes from an arena report.

python -m tests.benchmark_policy --trace artifacts/seal-trained-fresh-40.json --output artifacts/bulk-policy.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from time import perf_counter

import numpy as np
from hexo import Game, ROOT, library
from klent import _baseline, observe


def scalar_observe(game):
    legal = game.legal_moves()
    codes, pairs = zip(*(game.nnue_policy_features(c) for c in legal))
    coords = np.asarray(legal, dtype='<i8')
    return {'centers': game.nnue_centers(), 'candidate_codes': np.asarray(codes, np.int32),
            'pairs': np.asarray(pairs, np.float32), 'phase': game.nnue_context(), 'player': game.player,
            'baseline': float(np.asarray(game.features(), np.int64)@_baseline)*(1 if game.player == 0 else -1),
            'legal': legal, 'legal_sha256': hashlib.sha256(coords.tobytes()).hexdigest()}


def run(trace, output, repeats=20):
    raw = trace.read_bytes()
    games = json.loads(raw)['games'][:4]
    histories = [[c[:2] for c in row['cells'][:n]] for row in games
                 for n in (10, 21, 32) if n < len(row['cells'])]
    if not histories or repeats < 1:
        raise ValueError('Need trace prefixes and positive repetitions')
    boards = [Game(history) for history in histories]
    samples = {'scalar_ms': [], 'bulk_ms': []}
    try:
        for board in boards:
            old, new = scalar_observe(board), observe(board)
            for key in old:
                if isinstance(old[key], np.ndarray):
                    assert old[key].dtype == new[key].dtype and old[key].tobytes() == new[key].tobytes(), key
                else:
                    assert old[key] == new[key], key
        for trial in range(6):
            order = [('scalar_ms', scalar_observe), ('bulk_ms', observe)]
            if trial % 2:
                order.reverse()
            for name, operation in order:
                start = perf_counter()
                for _ in range(repeats):
                    for board in boards:
                        operation(board)
                samples[name].append(1000*(perf_counter()-start)/(repeats*len(boards)))
        result = {'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'engine_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
                  'sources': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                              for name in ('src/hexo.cpp', 'src/hexo.hpp', 'hexo.py', 'klent.py')},
                  'trace_sha256': hashlib.sha256(raw).hexdigest(), 'histories': histories,
                  'legal_counts': [len(b.legal_moves()) for b in boards], 'repeats': repeats,
                  'ms_per_observation': samples,
                  'median_ms': {k: statistics.median(v) for k,v in samples.items()}}
        result['speedup'] = result['median_ms']['scalar_ms']/result['median_ms']['bulk_ms']
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k: result[k] for k in ('median_ms', 'speedup', 'legal_counts')}, indent=2))
    finally:
        for board in boards:
            board.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    run(args.trace, args.output, args.repeats)
