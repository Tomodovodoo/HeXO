"""Fit only the KLENT Q head on verified human chosen-action terminal returns.

The matching NNUE representation is frozen. This is supervised initialization
under human continuations, not an on-policy KLENT return or an optimal-Q label.
"""
import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from legacy.corpus_warmstart import verified_shards
from hexo import ROOT, library
from legacy.klent import Model, SCHEMA, digest, publish
from legacy.nnue_model import batches_for, collate, load_replay
from legacy.train import write_json


def source_hashes(root):
    # load_replay calls train.merge_nnue; hexo selects the native library.
    return {name: digest(root/'python'/('legacy/'+name if name != 'hexo.py' else name)) for name in
            ("q_warmstart.py", "corpus_warmstart.py", "nnue_model.py", "klent.py", "train.py", "hexo.py")}


@torch.no_grad()
def chosen_features(nnue, replay, ids, device):
    """Reuse exact NNUE feature code, encoding only the human's chosen action."""
    batch = collate(replay, ids, device)
    rows = torch.arange(len(ids), device=device)
    chosen = batch["chosen"]
    batch["candidate_codes"] = batch["candidate_codes"][rows, chosen]
    batch["pairs"] = batch["pairs"][rows, chosen]
    batch["candidate_owner"] = rows
    _, features = nnue.features(batch)
    return features.detach()


def encode_split(nnue, paths, args, fraction):
    budgets = SimpleNamespace(replay_positions=max(1, int(args.positions*fraction)),
                              nnue_replay_centers=max(1, int(args.replay_centers*fraction)))
    replay, available = load_replay(paths, budgets, args.seed)
    features = []
    for ids in batches_for(np.arange(len(replay["family"])), replay, args.feature_batch, args.batch_centers):
        features.append(chosen_features(nnue, replay, ids, args.device).cpu())
    return torch.cat(features), torch.from_numpy(replay["outcome"].copy()), replay["family"].copy(), available


def fit_head(head, train, validation, args):
    """Optimize Q only; validation selects the epoch and never supplies gradients."""
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr, fused=args.device == "cuda")
    generator = np.random.default_rng(args.seed)
    tx, ty = (v.to(args.device) for v in train)
    vx, vy = (v.to(args.device) for v in validation)
    best, best_state, best_optimizer, metrics = math.inf, None, None, []
    for epoch in range(args.epochs):
        head.train()
        order = generator.permutation(len(tx))
        total = 0.
        for start in range(0, len(order), args.batch):
            ids = torch.as_tensor(order[start:start+args.batch], device=args.device)
            loss = (head(tx[ids]).squeeze(-1)-ty[ids]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite Q warm-start loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()*len(ids)
        head.eval()
        with torch.no_grad():
            prediction = head(vx).squeeze(-1)
            mse = (prediction-vy).square().mean().item()
        if not math.isfinite(mse) or any(not torch.isfinite(p).all() for p in head.parameters()):
            raise FloatingPointError("Nonfinite Q warm-start parameters/validation")
        metric = {"epoch": epoch+1, "train_mse": total/len(tx), "validation_mse": mse,
                  "validation_q_std": prediction.std(unbiased=False).item(),
                  "validation_abs_q_mean": prediction.abs().mean().item()}
        metrics.append(metric)
        print(json.dumps(metric), flush=True)
        if mse < best:
            best = mse
            best_state, best_optimizer = copy.deepcopy(head.state_dict()), copy.deepcopy(optimizer.state_dict())
    head.load_state_dict(best_state)
    return metrics, best_optimizer


def main(args):
    if args.output.exists():
        raise ValueError("Q warm-start output must be a new directory")
    if (min(args.epochs, args.batch, args.feature_batch, args.positions, args.replay_centers, args.batch_centers) < 1
            or args.positions < 2 or not math.isfinite(args.lr) or args.lr <= 0):
        raise ValueError("Positive budgets and at least two positions required")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    paths = verified_shards(args.corpus)
    source = args.model.read_bytes()
    model_sha = hashlib.sha256(source).hexdigest()
    model = Model().to(args.device)
    model.nnue.load_state_dict(torch.load(io.BytesIO(source), map_location=args.device, weights_only=True), strict=True)
    model.nnue.eval().requires_grad_(False)
    if any(not torch.isfinite(p).all() for p in model.nnue.parameters()):
        raise ValueError("Matching NNUE model has nonfinite parameters")
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    identity = {"kind": "human-chosen-terminal-q-v1", "config": config, "model_sha256": model_sha,
                "corpus_manifest_sha256": digest(args.corpus/"manifest.json"), "engine_sha256": digest(library),
                "shards": {split: [{"path": str(path.resolve()), "sha256": digest(path)} for path in files]
                           for split, files in paths.items()},
                "sources": source_hashes(ROOT),
                "target": "STM terminal outcome at the human chosen action; human continuation, not optimal or KLENT on-policy Q"}
    tx, ty, tf, ta = encode_split(model.nnue, paths["train"], args, .8)
    vx, vy, vf, va = encode_split(model.nnue, paths["validation"], args, .2)
    if np.any(tf % 5 == 0) or np.any(vf % 5 != 0) or set(tf) & set(vf):
        raise ValueError("Q warm-start family split disagrees with human corpus")
    metrics, optimizer = fit_head(model.q, (tx, ty), (vx, vy), args)
    # Refuse a corpus/model changed during fitting rather than publish false provenance.
    if digest(args.model) != model_sha or digest(args.corpus/"manifest.json") != identity["corpus_manifest_sha256"]:
        raise ValueError("Q warm-start source changed during fitting")
    verified_shards(args.corpus)
    report = {"kind": "human-chosen-terminal-q-v1", "promotion": False,
              "representation_frozen": True, "held_out_test_loaded": False, "excluded_loaded": False,
              "train_positions": len(tx), "validation_positions": len(vx),
              "available_positions": {"train": ta, "validation": va},
              "train_families": len(set(tf)), "validation_families": len(set(vf)),
              "epochs": metrics, "selected_epoch": min(metrics, key=lambda m: m["validation_mse"])["epoch"],
              "seconds": time.perf_counter()-started,
              "gpu_peak_mib": torch.cuda.max_memory_allocated()/2**20 if args.device == "cuda" else 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def writer(stage):
        (stage/"model.pt").write_bytes(source)
        model.nnue.export(stage/"model.nnue")
        torch.save({"schema": SCHEMA, "model_sha256": model_sha, "state": model.q.state_dict(),
                    "initialization": identity}, stage/"q.pt")
        torch.save(optimizer, stage/"q_optimizer.pt")
        write_json(stage/"report.json", report)
    publish(args.output, identity, writer, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=30000)
    parser.add_argument("--replay-centers", type=int, default=20000000)
    parser.add_argument("--batch-centers", type=int, default=65536)
    parser.add_argument("--feature-batch", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    main(parser.parse_args())
