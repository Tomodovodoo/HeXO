"""Fit NNUE on verified human train/validation shards, never held-out test data."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace


def verified_shards(directory):
    import numpy as np
    manifest = json.loads((directory/"manifest.json").read_text())
    result = {"train": [], "validation": []}
    for item in manifest["shards"]:
        relative = Path(item["path"])
        if relative.parts[0] not in result:
            continue
        path = (directory/relative).resolve()
        if relative.is_absolute() or path.parent != (directory/relative.parts[0]).resolve():
            raise ValueError("shard path escapes its declared split")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            if len(data["family"]) != item["positions"]:
                raise ValueError("shard row count mismatch")
            validation = data["family"] % 5 == 0
            if not (validation.all() if relative.parts[0] == "validation" else (~validation).all()):
                raise ValueError("shard family split mismatch")
            if data["search_valid"].any() or not data["policy_valid"].all():
                raise ValueError("expected human imitation labels without search")
            if not np.isin(data["outcome"], [-1, 1]).all() or not np.array_equal(data["search"], data["outcome"]):
                raise ValueError("expected verified terminal outcomes and inactive search copies")
        result[relative.parts[0]].append(path)
    if not all(result.values()):
        raise ValueError("both train and validation shards are required")
    return result


def fit(args):
    import numpy as np
    import torch
    from nnue_model import NNUE, batches_for, collate, load_replay, objective
    from train import merge_nnue
    if args.output.exists():
        raise ValueError("warmstart output must be a new directory")
    if (min(args.epochs, args.batch, args.positions, args.replay_centers, args.batch_centers) < 1
            or args.positions < 2 or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.policy_weight) or args.policy_weight < 0):
        raise ValueError("positive budgets and at least two replay positions required")
    paths = verified_shards(args.corpus)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    parts, available = [], {}
    # Pass every shard directly to the bounded loader, with no chronology window.
    for split, fraction in (("train", .8), ("validation", .2)):
        budget = SimpleNamespace(replay_positions=max(1, int(args.positions*fraction)),
                                 nnue_replay_centers=max(1, int(args.replay_centers*fraction)))
        part, available[split] = load_replay(paths[split], budget, args.seed)
        parts.append(part)
    replay = merge_nnue(parts)
    train = np.flatnonzero(replay["family"] % 5 != 0)
    validation = np.flatnonzero(replay["family"] % 5 == 0)
    model = NNUE().to(device)
    if args.initial_model:
        model.load_state_dict(torch.load(args.initial_model, map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.001, fused=device == "cuda")
    rng = np.random.default_rng(args.seed)
    best, best_model, best_optimizer, epochs = math.inf, None, None, []
    for epoch in range(args.epochs):
        model.train()
        total, count = 0., 0
        for ids in batches_for(rng.permutation(train), replay, args.batch, args.batch_centers):
            loss = objective(model, collate(replay, ids, device), args.policy_weight)[0]
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite human training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total += float(loss.detach())*len(ids)
            count += len(ids)
        model.eval()
        validation_total = 0.
        with torch.no_grad():
            for ids in batches_for(validation, replay, args.batch, args.batch_centers):
                loss = objective(model, collate(replay, ids, device), args.policy_weight)[0]
                validation_total += float(loss)*len(ids)
        metric = {"epoch": epoch+1, "train_loss": total/count, "validation_loss": validation_total/len(validation)}
        if not math.isfinite(metric["validation_loss"]):
            raise FloatingPointError("nonfinite human validation loss")
        epochs.append(metric)
        print(json.dumps(metric), flush=True)
        if metric["validation_loss"] < best:
            best = metric["validation_loss"]
            best_model, best_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    args.output.mkdir(parents=True)
    model.load_state_dict(best_model)
    torch.save(best_model, args.output/"model.pt")
    torch.save(best_optimizer, args.output/"optimizer.pt")
    model.export(args.output/"model.nnue")
    hashes = {name: hashlib.sha256((args.output/name).read_bytes()).hexdigest()
              for name in ("model.pt", "model.nnue", "optimizer.pt")}
    report = {"kind": "human-nnue-warmstart", "promotion": False,
              "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "corpus_manifest_sha256": hashlib.sha256((args.corpus/"manifest.json").read_bytes()).hexdigest(),
              "initial_model_sha256": hashlib.sha256(args.initial_model.read_bytes()).hexdigest() if args.initial_model else None,
              "shards": {split: [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in values]
                         for split, values in paths.items()},
              "available_positions": available, "train_positions": len(train), "validation_positions": len(validation),
              "held_out_test_loaded": False, "epochs": epochs,
              "selected_epoch": min(epochs, key=lambda x: x["validation_loss"])["epoch"],
              "output_sha256": hashes, "seconds": time.perf_counter()-started,
              "gpu_peak_mib": torch.cuda.max_memory_allocated()/2**20 if device == "cuda" else 0,
              "torch": torch.__version__, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report["training_source_sha256"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                         for name in ("corpus_warmstart.py", "nnue_model.py", "train.py")}
    (args.output/"manifest.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--positions", type=int, default=16384)
    parser.add_argument("--replay-centers", type=int, default=4000000)
    parser.add_argument("--batch-centers", type=int, default=32000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--policy-weight", type=float, default=.25)
    args = parser.parse_args()
    fit(args)


if __name__ == "__main__":
    main()
