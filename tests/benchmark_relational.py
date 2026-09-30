"""Reproduce primary-dimension CUDA forward/backward/Adam memory checks.

python -m tests.benchmark_relational --output artifacts/relational-production.json
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from hexo import Game, ROOT, library
from legacy.relational_encoder import pack
from legacy.relational_native import encode, _load
from legacy.relational_model import ModelConfig, RelationalNet


def history_fixture(count):
    rng, history, game = random.Random(20260924), [], Game()
    try:
        for index in range(count):
            choices = game.legal_moves()
            if index < 50:
                choices = sorted(choices,key=lambda c:max(abs(c[0]),abs(c[1]),abs(sum(c))))[:120]
            while True:
                action = rng.choice(choices)
                game.play(*action)
                if game.winner < 0:
                    history.append(action)
                    break
                game.undo()
                choices.remove(action)
    finally:
        game.close()
    return history


def run(output):
    if not torch.cuda.is_available():
        raise ValueError('This benchmark requires CUDA; CPU model tests are separate')
    torch.set_num_threads(2)
    torch.manual_seed(217)
    history = history_fixture(150)
    model = RelationalNet().cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
    rows = []
    for size in (1,150):
        graph = encode(history[:size],max_nodes=12000,max_edges=600000)
        batch = pack([graph],'cuda')
        free,total = torch.cuda.mem_get_info()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result = model(batch)
            loss = -torch.log_softmax(result['logits'],0)[0]+(result['q'][0]-.5).square()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        assert len(result['logits']) == len(graph.actions)
        assert all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all()) for p in model.parameters())
        rows.append(dict(stones=size,nodes=graph.node_count,relation_traversals=graph.edge_count,
                         legal_actions=len(graph.actions),seconds=elapsed,loss=loss.detach().item(),
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                         peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                         free_before_step_mib=free/2**20,total_mib=total/2**20))
    report = dict(config=asdict(ModelConfig()),parameters=sum(p.numel() for p in model.parameters()),
                  device=torch.cuda.get_device_name(),torch=torch.__version__,history=history,rows=rows,
                  precision='BF16 projections, FP32 softmax/normalization/reductions/residuals',
                  engine_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                  graph_engine_sha256=hashlib.sha256(Path(_load()._name).read_bytes()).hexdigest(),
                  source_hashes={name:hashlib.sha256((ROOT/('python/'+('legacy/'+name if name != 'hexo.py' else name) if name.endswith('.py') else name)).read_bytes()).hexdigest()
                                 for name in ('relational_encoder.py','relational_native.py','relational_model.py','src/relational_graph.cpp')})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'parameters':report['parameters'],'rows':rows},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
