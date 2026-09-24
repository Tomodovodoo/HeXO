"""CPU end-to-end value-pass comparison using fixed rows of a saved KLENT corpus."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import statistics
import subprocess
from time import perf_counter

import numpy as np
import torch
from hexo import ROOT, library
from klent import Model, chunks, pack, rebuild
from nnue_model import SCORE_SCALE


def run(corpus, checkpoint, output, positions=128, trials=3):
    torch.set_num_threads(2)
    seed = 20261010
    rows = json.loads((corpus/'rows.json').read_text())
    episodes = {e['id']: e for e in json.loads((corpus/'episodes.json').read_text())}
    indices = np.random.default_rng(seed).choice(len(rows), min(positions, len(rows)), replace=False)
    selected = [rows[int(i)] for i in indices]
    with np.load(corpus/'policies.npz') as data:
        offsets = data['offsets']
        probabilities = data['probs']
        for i, row in zip(indices, selected):
            row['mu'] = probabilities[offsets[i]:offsets[i+1]].copy()
    args = SimpleNamespace(batch=32, cells=65536, centers=65536)
    initial = Model()
    initial.nnue.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    times = {'full': [], 'value_only': []}
    reference = None
    for trial in range(trials):
        for value_only in ((False, True) if trial % 2 == 0 else (True, False)):
            model = copy.deepcopy(initial)
            optimizer = torch.optim.Adam(model.nnue.value.parameters(), lr=.001)
            losses = []
            groups = []
            start = perf_counter()
            for start_row in range(0, len(selected), args.batch):
                part = selected[start_row:start_row+args.batch]
                observations = [rebuild(row, episodes, value_only=value_only) for row in part]
                optimizer.zero_grad(set_to_none=True)
                for ids in chunks(observations, args):
                    groups.append((start_row, ids))
                    batch = pack([observations[i] for i in ids], 'cpu', value_only=value_only)
                    with torch.no_grad():
                        inputs = model.nnue.position_features(batch) if value_only else model.nnue.features(batch)[0]
                    value = torch.tanh(batch['baseline']/SCORE_SCALE+model.nnue.value(inputs).squeeze(-1))
                    target = torch.tensor([part[i]['target'] for i in ids])
                    loss = (value-target).square().mean()
                    (loss*(len(ids)/len(part))).backward()
                    losses.append(loss.item())
                optimizer.step()
            times['value_only' if value_only else 'full'].append(perf_counter()-start)
            state = model.nnue.value.state_dict()
            optimizer_state = optimizer.state_dict()['state']
            if reference is None:
                reference = (state, optimizer_state, losses, groups)
            else:
                assert losses == reference[2] and groups == reference[3]
                for key in state:
                    torch.testing.assert_close(state[key], reference[0][key], rtol=0, atol=0)
                for key, item in optimizer_state.items():
                    for name, tensor in item.items():
                        torch.testing.assert_close(tensor, reference[1][key][name], rtol=0, atol=0)
    report = dict(device='cpu', threads=2, seed=seed, indices=list(map(int, indices)), rows=len(selected),
                  batch=args.batch, cells=args.cells, centers=args.centers, trials=trials,
                  source_revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                  source_hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('klent.py','nnue_model.py','hexo.py')},
                  engine_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                  corpus_hashes={name:hashlib.sha256((corpus/name).read_bytes()).hexdigest() for name in ('rows.json','episodes.json','policies.npz')},
                  model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  seconds=times, median_seconds={k:statistics.median(v) for k,v in times.items()},
                  parity='Exact losses, gradient grouping, value parameters, and Adam states on every trial')
    report['speedup'] = report['median_seconds']['full']/report['median_seconds']['value_only']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('rows','median_seconds','speedup','parity')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--positions', type=int, default=128)
    parser.add_argument('--trials', type=int, default=3)
    args = parser.parse_args()
    if args.positions < 1 or args.trials < 1:
        parser.error('positions and trials must be positive')
    run(args.corpus, args.model, args.output, args.positions, args.trials)
