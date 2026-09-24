"""Self-play -> fit challenger -> evaluate frozen opponents -> promote or reject."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np

from hexo import Game, ROOT, library
from arena import wilson


def family(opening):
    variants = []
    for reflected in (False, True):
        points = [(r, q) if reflected else (q, r) for q, r in opening[1:]]
        for _ in range(6):
            variants.append(tuple(sorted(points)))
            points = [(-r, q+r) for q, r in points]
    key = repr(min(variants)).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def opening_for(seed, evaluation=False):
    rng = random.Random(seed)
    radius = 3 if evaluation else 2
    cells = [(q, r) for q in range(-radius, radius+1) for r in range(-radius, radius+1)
             if 0 < max(abs(q), abs(r), abs(q+r)) <= radius]
    while True:
        opening = [(0, 0), *rng.sample(cells, 2)]
        if not evaluation or any(max(abs(q), abs(r), abs(q+r)) == 3 for q, r in opening[1:]):
            return opening


def load_table(path):
    table = np.load(path, allow_pickle=False)
    if table.shape != (729,) or table.dtype != np.int32:
        raise ValueError(f"Invalid native table: {path}")
    return table


def play_game(task):
    # Workers import no PyTorch and perform native CPU search only.
    tables = [load_table(p) for p in task["tables"]]
    game = Game(opening_for(task["seed"], task["evaluation"]))
    initial = [(q, r) for q, r, _ in game.cells]
    features, searches, times, depths = [], [], [[], []], []
    while game.winner < 0 and len(game.cells) < task["max_stones"]:
        if len(game.cells) + game.remaining > task["max_stones"]:
            break
        side = game.player
        game.load_table(tables[side])
        if not task["evaluation"]:
            features.append(game.features())
        start = time.perf_counter()
        result = game.search(task["ms"], width=task["width"])
        times[side].append((time.perf_counter()-start)*1000)
        depths.append(result["depth"])
        searches.append(math.tanh(result["score"]/6000) * (1 if side == 0 else -1))
        if not result["moves"]:
            raise RuntimeError("Nonterminal search returned no move")
        for move in result["moves"]:
            game.play(*move)
    record = {"seed": task["seed"], "family": family(initial), "winner": game.winner,
              "reason": "six-in-a-row" if game.winner >= 0 else "truncated",
              "cells": game.cells, "times": times, "depths": depths,
              "tables": task["tables"], "challenger_color": task.get("challenger_color")}
    samples = None
    if not task["evaluation"]:
        outcome = (1 if game.winner == 0 else -1) if game.winner >= 0 else float("nan")
        n = len(features)
        samples = {"features": np.asarray(features, dtype=np.int32).reshape(-1, 729),
                   "search": np.asarray(searches, dtype=np.float32),
                   "outcome": np.full(n, outcome, dtype=np.float32),
                   "family": np.full(n, family(initial), dtype=np.uint32)}
    game.close()
    return record, samples


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def event(run, kind, **fields):
    with (run / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": time.time(), "kind": kind, **fields}, allow_nan=False) + "\n")


def optimize(run, checkpoint, incumbent, replay_paths, args):
    import torch
    from learning_model import PatternModel, pattern_data
    torch.set_num_threads(2)
    torch.manual_seed(args.seed + checkpoint)
    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    if args.device == "cuda" and device != "cuda":
        raise RuntimeError("CUDA was requested but is unavailable")
    data = []
    for path in replay_paths[-args.replay_iterations:]:
        with np.load(path, allow_pickle=False) as saved:
            data.append({k: saved[k] for k in saved.files})
    merged = {k: np.concatenate([d[k] for d in data]) for k in data[0]}
    validation = merged["family"] % 5 == 0
    training = ~validation
    if not training.any():
        raise RuntimeError("No training opening families; collect more self-play games")
    features = torch.as_tensor(merged["features"], dtype=torch.float32, device=device)
    outcome = merged["outcome"]
    targets = np.where(np.isfinite(outcome), .75*np.nan_to_num(outcome)+.25*merged["search"], merged["search"])
    target = torch.as_tensor(targets, device=device)
    train_ids = torch.as_tensor(np.flatnonzero(training), device=device)
    val_ids = torch.as_tensor(np.flatnonzero(validation), device=device)
    model = PatternModel().to(device)
    model.load_state_dict(torch.load(run / incumbent["model"], map_location=device, weights_only=True))
    inputs, swap, reverse, baseline = pattern_data(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.001)
    losses = []
    generator = torch.Generator(device=device).manual_seed(args.seed+checkpoint)
    for epoch in range(args.epochs):
        model.train()
        order = train_ids[torch.randperm(len(train_ids), generator=generator, device=device)]
        total = 0
        for ids in order.split(args.batch):
            predicted, table = model(features[ids], inputs, swap, reverse, baseline)
            mse = (predicted-target[ids]).square().mean()
            loss = mse + 1e-7*table.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total += float(mse.detach()) * len(ids)
        model.eval()
        with torch.no_grad():
            val_loss = float((model(features[val_ids], inputs, swap, reverse, baseline)[0]-target[val_ids]).square().mean()) if len(val_ids) else None
        metrics = {"iteration": checkpoint, "epoch": epoch+1, "train_loss": total/len(train_ids),
                   "validation_loss": val_loss, "train_positions": len(train_ids),
                   "validation_positions": len(val_ids), "device": device}
        losses.append(metrics)
        event(run, "training", **metrics)
        print(f"Epoch {epoch+1}/{args.epochs}: loss {metrics['train_loss']:.4f}, validation {val_loss}", flush=True)
    directory = run / "checkpoints" / f"{checkpoint:04d}"
    directory.mkdir(exist_ok=True)
    model_path = directory / "model.pt"
    torch.save(model.cpu().state_dict(), model_path)
    model.eval()
    with torch.no_grad():
        inputs, swap, reverse, _ = pattern_data()
        unquantized = model.table(inputs, swap, reverse)
        table = torch.round(unquantized).to(torch.int32).numpy()
    assert table[0] == 0 and np.array_equal(table, -table[swap.numpy()])
    assert np.array_equal(table, table[reverse.numpy()])
    np.save(directory / "table.npy", table, allow_pickle=False)
    # Compare deployed integer inference with the training calculation on actual positions.
    expected = merged["features"].astype(np.int64) @ table.astype(np.int64)
    actual_float = merged["features"] @ unquantized.numpy()
    quantization_error = float(np.max(np.abs(expected-actual_float)))
    return {"id": checkpoint, "model": str(model_path.relative_to(run)),
            "table": str((directory / "table.npy").relative_to(run)), "loss": losses[-1],
            "quantization_max_score_error": quantization_error,
            "table_sha256": hashlib.sha256(table.tobytes()).hexdigest()}


def evaluate(pool, run, challenger, opponent, iteration, args):
    tasks = []
    for index in range(args.eval_games):
        color = index % 2
        tables = [str(run / opponent["table"])] * 2
        tables[color] = str(run / challenger["table"])
        tasks.append({"tables": tables, "seed": args.seed+1000000+iteration*10000+index//2,
                      "evaluation": True, "ms": args.eval_ms, "width": args.width,
                      "max_stones": args.max_stones, "challenger_color": color})
    records = []
    for future in as_completed([pool.submit(play_game, task) for task in tasks]):
        record, _ = future.result()
        records.append(record)
        event(run, "evaluation_game", iteration=iteration, opponent=opponent["id"],
              finished=len(records), total=len(tasks), winner=record["winner"],
              challenger_color=record["challenger_color"], reason=record["reason"])
    completed = [g for g in records if g["winner"] >= 0]
    wins = sum(g["winner"] == g["challenger_color"] for g in completed)
    n = len(completed)
    low, high = wilson(wins, n)
    rate = (wins+.5)/(n+1)
    pairs = {}
    for g in completed:
        pairs.setdefault(g["seed"], []).append(int(g["winner"] == g["challenger_color"]))
    pair_wins = sum(len(p) == 2 and sum(p) == 2 for p in pairs.values())
    pair_losses = sum(len(p) == 2 and sum(p) == 0 for p in pairs.values())
    decisive = pair_wins + pair_losses
    pair_p = sum(math.comb(decisive, k) for k in range(pair_wins, decisive+1))/2**decisive if decisive else 1
    logit = lambda p: 400*math.log10(max(1e-6,p)/max(1e-6,1-p))
    metrics = {"opponent": opponent["id"], "wins": wins, "losses": n-wins,
               "incomplete": len(records)-n, "win_rate": wins/n if n else None,
               "win_rate_95pct": [low, high], "elo_delta": logit(rate) if n else None,
               "elo_delta_95pct": [logit(low), logit(high)] if n else None,
               "opening_pair_wins": pair_wins, "opening_pair_losses": pair_losses,
               "opening_pair_p": pair_p}
    write_json(run / "matches" / f"{iteration:04d}-vs-{opponent['id']:04d}.json", {"metrics": metrics, "games": records})
    event(run, "evaluation", iteration=iteration, **metrics)
    return metrics


def run_training(args):
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    for folder in ("checkpoints", "data", "matches"):
        (run / folder).mkdir(exist_ok=True)
    summary_path = run / "summary.json"
    config = {k: v for k, v in vars(args).items() if k not in ("iterations", "run")}
    training_hash = hashlib.sha256(Path(__file__).read_bytes() + (ROOT / "learning_model.py").read_bytes()).hexdigest()
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["config"] != config:
            raise ValueError("Resume requires the same configuration. Use a new run directory for changed settings.")
        if summary["engine_sha256"] != hashlib.sha256(library.read_bytes()).hexdigest():
            raise ValueError("Native engine changed. Use a new run directory to keep ratings comparable.")
        if summary.get("training_sha256") != training_hash:
            raise ValueError("Training code changed. Use a new run directory to preserve run provenance.")
    else:
        import torch
        from learning_model import PatternModel
        torch.manual_seed(args.seed)
        initial = run / "checkpoints" / "0000"
        initial.mkdir()
        torch.save(PatternModel().state_dict(), initial / "model.pt")
        np.save(initial / "table.npy", np.zeros(729, dtype=np.int32), allow_pickle=False)
        summary = {"schema": 1, "started": time.time(), "status": "starting", "iteration": 0,
                   "positions": 0, "self_play_games": 0, "truncated_games": 0,
                   "config": config, "engine_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                   "training_sha256": training_hash,
                   "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "incumbent": 0, "checkpoints": [{"id": 0, "model": "checkpoints/0000/model.pt",
                   "table": "checkpoints/0000/table.npy", "promoted": True,
                   "anchor_elo": 0, "anchor_elo_95pct": None, "evaluations": []}]}
        write_json(summary_path, summary)
    lock = run / "training.lock"
    # Exclusive creation prevents two trainers from corrupting the same run.
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            start_iteration = summary["iteration"]+1
            for iteration in range(start_iteration, start_iteration+args.iterations):
                incumbent = next(c for c in summary["checkpoints"] if c["id"] == summary["incumbent"])
                pool_members = [c for c in summary["checkpoints"] if c["promoted"]][-4:]
                summary.update(status="self-play", active_iteration=iteration, stage_completed=0, stage_total=args.games)
                write_json(summary_path, summary)
                tasks = []
                for index in range(args.games):
                    rival = pool_members[index % len(pool_members)]
                    tables = [str(run / incumbent["table"]), str(run / rival["table"])]
                    if index % 2:
                        tables.reverse()
                    tasks.append({"tables": tables, "seed": args.seed+iteration*10000+index,
                                  "evaluation": False, "ms": args.ms, "width": args.width,
                                  "max_stones": args.max_stones})
                records, batches = [], []
                started = time.perf_counter()
                for future in as_completed([pool.submit(play_game, task) for task in tasks]):
                    record, samples = future.result()
                    records.append(record);batches.append(samples)
                    summary["stage_completed"] = len(records)
                    summary["games_per_hour"] = len(records)*3600/(time.perf_counter()-started)
                    write_json(summary_path, summary)
                    event(run, "self_play_game", iteration=iteration, completed=len(records),
                          total=args.games, reason=record["reason"], stones=len(record["cells"]))
                    print(f"Self-play {len(records)}/{args.games}: {record['reason']}, {len(record['cells'])} stones", flush=True)
                data_path = run / "data" / f"{iteration:04d}.npz"
                ordered = sorted(zip(records, batches), key=lambda pair: pair[0]["seed"])
                records, batches = map(list, zip(*ordered))
                merged = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
                np.savez_compressed(data_path, **merged)
                write_json(run / "data" / f"{iteration:04d}-games.json", records)
                summary["status"] = "training"
                write_json(summary_path, summary)
                challenger = optimize(run, iteration, incumbent, sorted((run / "data").glob("*.npz")), args)
                summary["status"] = "evaluation"
                write_json(summary_path, summary)
                opponents = [incumbent]
                anchor = summary["checkpoints"][0]
                if incumbent["id"] != 0:
                    opponents.append(anchor)
                older = [c for c in pool_members if c["id"] not in (0, incumbent["id"])]
                if older:
                    opponents.append(older[-1])
                evaluations = [evaluate(pool, run, challenger, opponent, iteration, args) for opponent in opponents]
                direct = evaluations[0]
                anchored = next(e for e in evaluations if e["opponent"] == 0)
                # Pair-level sign test accounts for the two colors of each opening.
                # Alpha spending bounds repeated promotion attempts across the run.
                alpha = .05/(iteration*(iteration+1))
                promoted = direct["incomplete"] == 0 and direct["win_rate_95pct"][0] > .5 and direct["opening_pair_p"] <= alpha
                # Reject obvious regressions against older frozen checkpoints as well.
                promoted = promoted and all(e["incomplete"] == 0 and e["win_rate_95pct"][1] >= .5 for e in evaluations[1:])
                challenger.update(promoted=promoted, promotion_alpha=alpha, evaluations=evaluations,
                                  anchor_elo=anchored["elo_delta"], anchor_elo_95pct=anchored["elo_delta_95pct"])
                summary["checkpoints"].append(challenger)
                if promoted:
                    summary["incumbent"] = iteration
                summary["positions"] += len(merged["features"])
                summary["self_play_games"] += len(records)
                summary["truncated_games"] += sum(g["winner"] < 0 for g in records)
                summary.update(iteration=iteration, status="iteration-complete")
                write_json(summary_path, summary)
                event(run, "checkpoint", iteration=iteration, promoted=promoted, incumbent=summary["incumbent"],
                      anchor_elo=challenger["anchor_elo"], anchor_elo_95pct=challenger["anchor_elo_95pct"])
                print(f"Checkpoint {iteration}: {'PROMOTED' if promoted else 'rejected'}, anchor Elo {challenger['anchor_elo']}", flush=True)
            summary["status"] = "finished"
            write_json(summary_path, summary)
    except BaseException as error:
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        write_json(summary_path, summary)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/selfplay")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--ms", type=int, default=50)
    parser.add_argument("--eval-ms", type=int, default=100)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--max-stones", type=int, default=200)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=.002)
    parser.add_argument("--replay-iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if any(getattr(args, k) < 1 for k in ("iterations", "games", "eval_games", "ms", "eval_ms", "workers", "epochs", "batch", "replay_iterations")):
        parser.error("Counts and search budgets must be positive")
    if args.eval_games % 2 or args.max_stones < 5 or not 2 <= args.width <= 128:
        parser.error("Evaluation games must be even, max-stones >= 5, width in 2..128")
    run_training(args)
