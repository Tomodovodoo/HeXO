"""Self-play -> fit challenger -> evaluate frozen opponents -> promote or reject."""
import argparse
import copy
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


def task_opening(seed, evaluation, max_stones, curriculum):
    if curriculum == "mixed-v1":
        from curriculum import opening_for as sampler, family as identify
    else:
        sampler, identify = opening_for, family
    for attempt in range(100):
        prefix = sampler(seed+attempt*15485863, evaluation)
        if len(prefix)+2 <= max_stones:
            return {"opening": prefix, "family": identify(prefix), "curriculum": curriculum}
    raise ValueError("No curriculum prefix fits the stone cap; increase the cap")


def load_table(path):
    table = np.load(path, allow_pickle=False)
    if table.shape != (729,) or table.dtype != np.int32:
        raise ValueError(f"Invalid native table: {path}")
    return table


def checkpoint_artifact(checkpoint):
    return checkpoint["nnue"] if checkpoint.get("kind") == "nnue" else checkpoint["table"]


def merge_nnue(batches):
    """Merge variable-length replay without padding the infinite-board features."""
    merged = {"nnue_schema": np.asarray([1], dtype=np.int32)}
    for key in batches[0]:
        if key in ("nnue_schema", "center_offsets", "candidate_offsets"):
            continue
        merged[key] = np.concatenate([b[key] for b in batches])
    for key, values in (("center_offsets", "centers"), ("candidate_offsets", "candidate_codes")):
        offsets, count = [np.asarray([0], dtype=np.int64)], 0
        for batch in batches:
            offsets.append(batch[key][1:]+count)
            count += len(batch[values])
        merged[key] = np.concatenate(offsets)
    return merged


def nnue_example(game, move, result, candidates):
    choices = game.candidates(candidates)[:candidates]
    if move not in choices:
        choices = choices[:max(0, candidates-1)] + [move]
    codes, pairs = zip(*(game.nnue_policy_features(c) for c in choices))
    powers = 3**np.arange(6)
    digits = np.arange(729)[:, None]//powers % 3
    black, white = (digits == 1).sum(1), (digits == 2).sum(1)
    weights = np.asarray([0, 1, 12, 150, 2400, 24000, 1000000])
    baseline = np.where(white == 0, weights[black], np.where(black == 0, -weights[white], 0))
    return {"centers": game.nnue_centers(), "candidate_codes": np.asarray(codes, dtype=np.int32),
            "pairs": np.asarray(pairs, dtype=np.float32), "candidate_coords": np.asarray(choices, dtype=np.int64),
            "phase": game.nnue_context(), "player": game.player,
            "baseline": float(np.asarray(game.features(), dtype=np.int64)@baseline)*(1 if game.player == 0 else -1),
            "chosen": choices.index(move), "search": math.tanh(result["score"]/6000),
            "ply": len(game.cells), "search_depth": result["depth"], "search_ms": result["elapsed_ms"],
            "search_valid": result["depth"] > 0, "policy_valid": result["depth"] > 0}


def pack_nnue(rows, outcome, fid):
    data = {"nnue_schema": np.asarray([1], dtype=np.int32)}
    for key in ("centers", "candidate_codes", "pairs", "candidate_coords"):
        data[key] = np.concatenate([r[key] for r in rows])
    for key, values in (("center_offsets", "centers"), ("candidate_offsets", "candidate_codes")):
        data[key] = np.concatenate(([0], np.cumsum([len(r[values]) for r in rows]))).astype(np.int64)
    for key, dtype in (("phase", np.float32), ("player", np.int64), ("baseline", np.float32),
                       ("chosen", np.int64), ("search", np.float32), ("search_valid", bool), ("policy_valid", bool)):
        data[key] = np.asarray([r[key] for r in rows], dtype=dtype)
    for key, dtype in (("ply", np.int32), ("search_depth", np.int16), ("search_ms", np.float32)):
        data[key] = np.asarray([r[key] for r in rows], dtype=dtype)
    data["outcome"] = np.where(data["player"] == 0, outcome, -outcome).astype(np.float32)
    data["family"] = np.full(len(rows), fid, dtype=np.uint32)
    return data


def play_game(task):
    # Workers import no PyTorch and perform native CPU search only.
    nnue = task.get("model_kind", "pattern") == "nnue"
    tables = task["tables"] if nnue else [load_table(p) for p in task["tables"]]
    game = Game(task.get("opening") or opening_for(task["seed"], task["evaluation"]))
    initial = [(q, r) for q, r, _ in game.cells]
    fid = task["family"] if "family" in task else family(initial)
    rng = random.Random(task["seed"] ^ 0xC0FFEE)
    explored_turns = 0
    features, searches, times, depths, examples = [], [], [[], []], [], []
    search_trace = []
    while game.winner < 0 and len(game.cells) < task["max_stones"]:
        if len(game.cells) + game.remaining > task["max_stones"]:
            break
        side = game.player
        if nnue:
            game.load_model(tables[side])
        else:
            game.load_table(tables[side])
        # Explore only after a searched turn and only when no immediate tactic
        # exists. Exploratory actions are omitted from teacher replay entirely.
        if (not task["evaluation"] and depths and rng.random() < task.get("native_exploration", 0)
                and not game.tactical()):
            while game.player == side and game.winner < 0:
                if game.tactical():
                    continuation = game.search(task["ms"], width=task["width"])
                    for move in continuation["moves"]:
                        game.play(*move)
                    break
                choices = game.legal_moves() if rng.random() < .2 else game.candidates(min(8, task["width"]))
                game.play(*rng.choice(choices))
            explored_turns += 1
            continue
        if not task["evaluation"] and not nnue:
            features.append(game.features())
        start = time.perf_counter()
        result = game.search(task["ms"], width=task["width"])
        times[side].append((time.perf_counter()-start)*1000)
        if task.get("record_searches"):
            search_trace.append({"ply": len(game.cells), "player": side, "remaining": game.remaining,
                                 "wall_ms": times[side][-1], "result": result})
        depths.append(result["depth"])
        searches.append(math.tanh(result["score"]/6000) * (1 if side == 0 else -1))
        if not result["moves"]:
            raise RuntimeError("Nonterminal search returned no move")
        for move in result["moves"]:
            if nnue and not task["evaluation"]:
                examples.append(nnue_example(game, move, result, task["policy_candidates"]))
            game.play(*move)
    record = {"seed": task["seed"], "family": fid, "winner": game.winner,
              "reason": "six-in-a-row" if game.winner >= 0 else "truncated",
              "cells": game.cells, "times": times, "depths": depths,
              "tables": task["tables"], "model_kind": task.get("model_kind", "pattern"),
              "opening": initial, "prefix_length": len(initial), "curriculum": task.get("curriculum", "legacy"),
              "native_exploration": task.get("native_exploration", 0), "exploratory_turns": explored_turns,
              "challenger_color": task.get("challenger_color")}
    if task.get("record_searches"):
        record["search_trace"] = search_trace
    samples = None
    if not task["evaluation"]:
        outcome = (1 if game.winner == 0 else -1) if game.winner >= 0 else float("nan")
        if nnue:
            samples = pack_nnue(examples, outcome, fid)
        else:
            n = len(features)
            samples = {"features": np.asarray(features, dtype=np.int32).reshape(-1, 729),
                       "search": np.asarray(searches, dtype=np.float32),
                       "outcome": np.full(n, outcome, dtype=np.float32),
                       "family": np.full(n, fid, dtype=np.uint32)}
    game.close()
    return record, samples


def digest(path):
    """Hex sha256 of the file at `path`."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    # Windows readers can briefly deny rename while the dashboard reads a file.
    # Retry only sharing/access errors, retaining atomic publication throughout.
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            if os.name != "nt" or error.winerror not in (5, 32) or attempt == 7:
                raise
            time.sleep(.01 * 2**attempt)


def event(run, kind, **fields):
    with (run / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": time.time(), "kind": kind, **fields}, allow_nan=False) + "\n")


def optimize(run, checkpoint, incumbent, replay_paths, args, progress=None):
    if args.model == "nnue":
        from nnue_model import optimize_nnue
        return optimize_nnue(run, checkpoint, incumbent, replay_paths, args, progress,
                             lambda **metrics: event(run, "training", **metrics))
    import torch
    from learning_model import PatternModel, pattern_data
    torch.set_num_threads(2)
    torch.manual_seed(args.seed + checkpoint)
    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    if args.device == "cuda" and device != "cuda":
        raise RuntimeError("CUDA was requested but is unavailable")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    counts = []
    for path in replay_paths[-args.replay_iterations:]:
        with np.load(path, allow_pickle=False) as saved:
            counts.append(len(saved["family"]))
    available_positions = sum(counts)
    if not available_positions:
        raise ValueError("Replay contains no positions")
    replay_limit = getattr(args, "replay_positions", 250000)
    if device == "cuda":
        free, _ = torch.cuda.mem_get_info()
        available_memory = free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        replay_limit = min(replay_limit, max(1, int(available_memory*.4/(729*4))))
    if available_positions > replay_limit:
        chosen = np.random.default_rng(args.seed+checkpoint).choice(available_positions, replay_limit, replace=False)
        chosen.sort()
    else:
        chosen = np.arange(available_positions)
    # Choose global indices first: keep at most one uncompressed replay file in
    # host memory alongside the selected positions, never the full replay window.
    parts, offset = {}, 0
    for path, count in zip(replay_paths[-args.replay_iterations:], counts):
        ids = chosen[np.searchsorted(chosen, offset):np.searchsorted(chosen, offset+count)]-offset
        if len(ids):
            with np.load(path, allow_pickle=False) as saved:
                for key in saved.files:
                    parts.setdefault(key, []).append(saved[key][ids])
        offset += count
    merged = {k: np.concatenate(v) for k, v in parts.items()}
    del parts
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.001,
                                  fused=device == "cuda")
    if incumbent.get("optimizer"):
        optimizer.load_state_dict(torch.load(run / incumbent["optimizer"], map_location=device, weights_only=True))
    # All replay tensors stay on the device. The network evaluates 729 patterns once
    # per minibatch, then one dense matrix product scores all positions in that batch.
    batch = args.batch or min(len(train_ids), 8192 if device == "cuda" else 2048)
    steps = getattr(args, "updates_per_epoch", 0) or math.ceil(len(train_ids)/(args.batch or 256))
    losses = []
    best_loss, best_epoch, best_state, best_optimizer = math.inf, 0, None, None
    generator = torch.Generator(device=device).manual_seed(args.seed+checkpoint)
    processed = 0
    for epoch in range(args.epochs):
        model.train()
        order = train_ids[torch.randperm(len(train_ids), generator=generator, device=device)]
        total = torch.zeros((), device=device)
        epoch_positions = 0
        minibatches = order.split(batch)
        for step in range(steps):
            ids = minibatches[step % len(minibatches)]
            predicted, table = model(features[ids], inputs, swap, reverse, baseline)
            mse = (predicted-target[ids]).square().mean()
            loss = mse + 1e-7*table.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total += mse.detach() * len(ids)
            epoch_positions += len(ids)
        processed += epoch_positions
        model.eval()
        with torch.no_grad():
            # Bound validation temporaries too, even when the replay grows large.
            val_total = torch.zeros((), device=device)
            for ids in val_ids.split(batch):
                val_total += (model(features[ids], inputs, swap, reverse, baseline)[0]-target[ids]).square().sum()
            val_loss = float(val_total / len(val_ids)) if len(val_ids) else None
        train_loss = float(total / epoch_positions)
        selection_loss = val_loss if val_loss is not None else train_loss
        if selection_loss < best_loss:
            best_loss, best_epoch = selection_loss, epoch+1
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())
        metrics = {"iteration": checkpoint, "epoch": epoch+1, "train_loss": train_loss,
                   "validation_loss": val_loss, "train_positions": len(train_ids),
                   "validation_positions": len(val_ids), "device": device, "batch": batch,
                   "replay_positions_available": available_positions,
                   "replay_positions_selected": len(features),
                   "optimizer_steps": (epoch+1)*steps, "examples_processed": processed,
                   "positions_per_second": processed/(time.perf_counter()-started),
                   "gpu_memory_peak_mb": torch.cuda.max_memory_allocated()/2**20 if device == "cuda" else 0}
        if not math.isfinite(train_loss) or not math.isfinite(selection_loss):
            raise FloatingPointError("Non-finite learner loss; checkpoint was not exported")
        losses.append(metrics)
        event(run, "training", **metrics)
        if progress:
            progress(epoch+1, args.epochs, metrics)
        print(f"Epoch {epoch+1}/{args.epochs}: loss {metrics['train_loss']:.4f}, validation {val_loss}", flush=True)
    directory = run / "checkpoints" / f"{checkpoint:04d}"
    directory.mkdir(exist_ok=True)
    model_path = directory / "model.pt"
    model.load_state_dict(best_state)
    torch.save(model.cpu().state_dict(), model_path)
    optimizer_path = directory / "optimizer.pt"
    torch.save(best_optimizer, optimizer_path)
    model.eval()
    with torch.no_grad():
        inputs, swap, reverse, _ = pattern_data()
        unquantized = model.table(inputs, swap, reverse)
        table = torch.round(unquantized).to(torch.int32).numpy()
    assert table[0] == 0 and np.array_equal(table, -table[swap.numpy()])
    assert np.array_equal(table, table[reverse.numpy()])
    np.save(directory / "table.npy", table, allow_pickle=False)
    # Compare deployed integer inference with the training calculation on actual positions.
    quantization_error = 0.0
    for offset in range(0, len(features), 8192):
        chunk = merged["features"][offset:offset+8192]
        expected = chunk.astype(np.int64) @ table.astype(np.int64)
        actual_float = chunk @ unquantized.numpy()
        quantization_error = max(quantization_error, float(np.max(np.abs(expected-actual_float))))
    return {"id": checkpoint, "model": str(model_path.relative_to(run)),
            "table": str((directory / "table.npy").relative_to(run)), "loss": losses[best_epoch-1],
            "optimizer": str(optimizer_path.relative_to(run)), "learner_parent": incumbent["id"],
            "selected_epoch": best_epoch, "selection": "validation" if len(val_ids) else "training-only",
            "training_seconds": time.perf_counter()-started,
            "replay": [{"path": str(p.relative_to(run)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in replay_paths[-args.replay_iterations:]],
            "quantization_max_score_error": quantization_error,
            "table_sha256": hashlib.sha256(table.tobytes()).hexdigest()}


def paired_metrics(records, planned_games=None):
    total = len(records) if planned_games is None else planned_games
    if total < 2 or total % 2 or len(records) > total:
        raise ValueError("An even planned game count of at least two is required")
    completed = [g for g in records if g["winner"] >= 0]
    wins = sum(g["winner"] == g["challenger_color"] for g in completed)
    n = len(completed)
    incomplete = len(records)-n
    pending = total-len(records)
    margin = math.sqrt(math.log(40)/(2*(total//2)))
    low = max(0, wins/total-margin)
    high = min(1, (wins+incomplete+pending)/total+margin)
    pairs = {}
    for game in records:
        pair = pairs.setdefault(game["seed"], {})
        color = game["challenger_color"]
        if color not in (0, 1) or color in pair:
            raise ValueError("Duplicate or invalid color within an opening pair")
        pair[color] = game
    if len(pairs) > total//2:
        raise ValueError("Too many distinct opening pairs")
    valid = [[int(g["winner"] == g["challenger_color"]) for g in pair.values()]
             for pair in pairs.values() if len(pair) == 2 and all(g["winner"] >= 0 for g in pair.values())]
    pair_wins = sum(sum(p) == 2 for p in valid)
    pair_losses = sum(sum(p) == 0 for p in valid)
    decisive = pair_wins+pair_losses
    pair_p = sum(math.comb(decisive, k) for k in range(pair_wins, decisive+1))/2**decisive if decisive else 1
    logit = lambda p: 400*math.log10(max(1e-6, p)/max(1e-6, 1-p))
    rate = (wins+.5)/(n+1)
    rated = not incomplete and not pending and n > 0
    return {"wins": wins, "losses": n-wins, "incomplete": incomplete, "pending": pending,
            "rated": rated, "win_rate": wins/n if rated else None,
            "completed_only_win_rate": wins/n if n else None,
            "completed_only_provisional_elo": logit(rate) if n else None,
            "win_rate_95pct": [low, high], "elo_delta": logit(rate) if rated else None,
            # Keep the legacy finite display interval; exact open bounds are explicit.
            "elo_delta_95pct": [logit(low), logit(high)],
            "elo_delta_95pct_open": [logit(low) if low > 0 else None, logit(high) if high < 1 else None],
            "elo_display_probability_floor": 1e-6,
            "played_games": len(records), "planned_games": total,
            "interval_method": "opening-pair Hoeffding 95%, censored and pending outcomes bounded",
            "opening_pair_wins": pair_wins, "opening_pair_losses": pair_losses,
            "opening_pair_ties": len(valid)-pair_wins-pair_losses,
            "incomplete_pairs": total//2-len(valid), "opening_pair_p": pair_p}


def evaluate(pool, run, challenger, opponent, iteration, args, progress=None):
    tasks = []
    # A fixed pair count eventually cannot pass the shrinking promotion threshold.
    # Always leave enough pairs for at least an all-wins result to be eligible.
    alpha = .05/(iteration*(iteration+1))
    game_count = max(args.eval_games, 2*math.ceil(math.log2(1/alpha)))
    for index in range(game_count):
        color = index % 2
        tables = [str(run / checkpoint_artifact(opponent))] * 2
        tables[color] = str(run / checkpoint_artifact(challenger))
        seed = args.seed+1000000+iteration*10000+index//2
        tasks.append({"tables": tables, "seed": seed,
                      "evaluation": True, "ms": args.eval_ms, "width": args.width,
                      "model_kind": args.model,
                      **task_opening(seed, True, args.eval_max_stones, args.curriculum),
                      "max_stones": args.eval_max_stones, "challenger_color": color})
    records = []
    for future in as_completed([pool.submit(play_game, task) for task in tasks]):
        record, _ = future.result()
        records.append(record)
        if progress:
            progress(len(records), len(tasks), {"opponent": opponent["id"]})
        event(run, "evaluation_game", iteration=iteration, opponent=opponent["id"],
              finished=len(records), total=len(tasks), winner=record["winner"],
              challenger_color=record["challenger_color"], reason=record["reason"])
    metrics = paired_metrics(records)
    metrics.update(opponent=opponent["id"], requested_games=args.eval_games)
    write_json(run / "matches" / f"{iteration:04d}-vs-{opponent['id']:04d}.json", {"metrics": metrics, "games": records})
    event(run, "evaluation", iteration=iteration, **metrics)
    return metrics


def run_training(args):
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = run / "training.lock"
    # Lock before reading the summary or creating the first checkpoint. A second
    # trainer must not read stale state while waiting for another run to finish.
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        _run_training(args)
    finally:
        lock.unlink(missing_ok=True)


def initial_artifacts(args):
    model = getattr(args, "initial_model", None)
    optimizer = getattr(args, "initial_optimizer", None)
    if optimizer and not model:
        raise ValueError("--initial-optimizer requires --initial-model")
    if (model or optimizer) and args.model != "nnue":
        raise ValueError("Initial checkpoint import currently requires --model nnue")
    result = {}
    for key, value in (("model", model), ("optimizer", optimizer)):
        if value:
            path = Path(value).resolve()
            setattr(args, "initial_"+key, str(path))
            result[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


def initialize_nnue(initial, args, artifacts, identity=None):
    """Validate before publishing checkpoint zero; never overwrite an existing one."""
    import io
    import tempfile
    import torch
    from nnue_model import NNUE
    if initial.exists():
        manifest_path = initial / "initialization.json"
        if identity is None or not manifest_path.is_file():
            raise ValueError("Checkpoint zero already exists; refusing to overwrite it without a matching initialization record")
        manifest = json.loads(manifest_path.read_text())
        expected_files = {"model.pt", "model.nnue"} | ({"optimizer.pt"} if "optimizer" in artifacts else set())
        if (manifest.get("schema") != 1 or manifest.get("identity") != identity
                or set(manifest.get("files", {})) != expected_files
                or {p.name for p in initial.iterdir()} != expected_files | {"initialization.json"}):
            raise ValueError("Checkpoint zero initialization identity or file set differs; refusing to overwrite it")
        for name, digest in manifest["files"].items():
            if hashlib.sha256((initial/name).read_bytes()).hexdigest() != digest:
                raise ValueError("Checkpoint zero initialization file hash changed; refusing to overwrite it")
        metadata = {"model_sha256": manifest["files"]["model.nnue"],
                    **({"initial_artifacts": artifacts} if artifacts else {}),
                    **({"optimizer": "checkpoints/0000/optimizer.pt"} if "optimizer" in artifacts else {})}
        if manifest.get("metadata") != metadata:
            raise ValueError("Checkpoint zero initialization metadata differs; refusing to overwrite it")
        return metadata
    contents = {key: Path(item["path"]).read_bytes() for key, item in artifacts.items()}
    if any(hashlib.sha256(contents[key]).hexdigest() != item["sha256"] for key, item in artifacts.items()):
        raise ValueError("Initial checkpoint changed while loading")
    model = NNUE()
    if artifacts:
        state = torch.load(io.BytesIO(contents["model"]), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("Initial model contains nonfinite weights")
    optimizer = None
    if "optimizer" in artifacts:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.001)
        optimizer.load_state_dict(torch.load(io.BytesIO(contents["optimizer"]), map_location="cpu", weights_only=True))
        for group in optimizer.param_groups:
            if any(key not in group for key in ("betas", "eps", "weight_decay", "amsgrad", "maximize")):
                raise ValueError("Initial optimizer lacks AdamW hyperparameters")
            beta1, beta2 = group["betas"]
            if not (0 <= beta1 < 1 and 0 <= beta2 < 1 and math.isfinite(group["eps"]) and group["eps"] >= 0
                    and math.isfinite(group["weight_decay"]) and group["weight_decay"] >= 0):
                raise ValueError("Invalid initial optimizer hyperparameters")
        for parameter, state in optimizer.state.items():
            if set(state) not in (set(), {"step", "exp_avg", "exp_avg_sq"}, {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}):
                raise ValueError("Initial optimizer is not a compatible AdamW state")
            for name, value in state.items():
                if not torch.is_tensor(value) or not torch.isfinite(value).all():
                    raise ValueError("Initial optimizer contains invalid state")
                if name == "step":
                    valid = value.numel() == 1 and value.item() >= 0
                else:
                    valid = value.shape == parameter.shape
                    if name in ("exp_avg_sq", "max_exp_avg_sq"):
                        valid = valid and bool((value >= 0).all())
                if not valid:
                    raise ValueError("Initial optimizer state shape/value is incompatible")
        for group in optimizer.param_groups:
            group["lr"] = args.lr
            group["fused"] = False
            group["capturable"] = False
    with tempfile.TemporaryDirectory(prefix="initial-", dir=initial.parent) as temporary:
        stage = Path(temporary)/"checkpoint"
        stage.mkdir()
        if artifacts:
            (stage/"model.pt").write_bytes(contents["model"])
        else:
            torch.save(model.state_dict(), stage/"model.pt")
        if optimizer:
            torch.save(optimizer.state_dict(), stage/"optimizer.pt")
        for item in artifacts.values():
            if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Initial checkpoint changed while loading")
        digest = model.export(stage/"model.nnue")
        metadata = {"model_sha256": digest, **({"initial_artifacts": artifacts} if artifacts else {}),
                    **({"optimizer": "checkpoints/0000/optimizer.pt"} if optimizer else {})}
        if identity is not None:
            manifest = {"schema": 1, "identity": identity, "metadata": metadata,
                        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.iterdir()}}
            write_json(stage/"initialization.json", manifest)
            for path in stage.iterdir():
                with path.open("r+b") as handle:
                    os.fsync(handle.fileno())
        stage.rename(initial)
    return metadata


def _run_training(args):
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    for folder in ("checkpoints", "data", "matches"):
        (run / folder).mkdir(exist_ok=True)
    summary_path = run / "summary.json"
    artifacts = initial_artifacts(args)
    if artifacts:
        args.initial_artifacts = artifacts
    if getattr(args, "reanalysis", []):
        if args.model != "nnue":
            raise ValueError("--reanalysis requires --model nnue")
        from reanalysis import external_replays
        args.reanalysis = [str(Path(p).resolve()) for p in args.reanalysis]
        args.external_replay = external_replays(args.reanalysis)
    config = {k: v for k, v in vars(args).items() if k not in ("iterations", "run")}
    source_files = [Path(__file__), ROOT / "learning_model.py", ROOT / "hexo.py"]
    if args.model == "nnue":
        source_files.append(ROOT / "nnue_model.py")
        source_files.append(ROOT / "src" / "nnue.hpp")
        source_files.append(ROOT / "reanalysis.py")
    if args.curriculum == "mixed-v1":
        source_files.append(ROOT / "curriculum.py")
    if args.selfplay_backend == "gpu":
        source_files.append(ROOT / "gpu_games.py")
        if args.model == "nnue":
            source_files.append(ROOT / "gpu_nnue.py")
    training_hash = hashlib.sha256(b"".join(p.read_bytes() for p in source_files)).hexdigest()
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
        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        initial = run / "checkpoints" / "0000"
        if args.model == "nnue":
            identity = {"run": str(run), "config": config, "training_sha256": training_hash,
                        "engine_sha256": hashlib.sha256(library.read_bytes()).hexdigest()}
            initial_metadata = initialize_nnue(initial, args, artifacts, identity)
            initial_table = "checkpoints/0000/model.nnue"
        else:
            # Preserve deterministic pattern initialization recovery after a
            # failed summary write; imported NNUE artifacts use the manifest above.
            initial.mkdir(exist_ok=True)
            initial_metadata = {}
            from learning_model import PatternModel
            torch.save(PatternModel().state_dict(), initial / "model.pt")
            np.save(initial / "table.npy", np.zeros(729, dtype=np.int32), allow_pickle=False)
            initial_table = "checkpoints/0000/table.npy"
        summary = {"schema": 2, "started": time.time(), "status": "starting", "iteration": 0,
                   "positions": 0, "self_play_games": 0, "truncated_games": 0,
                   "config": config, "engine_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                   "training_sha256": training_hash,
                   "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                               "cuda": torch.version.cuda,
                               "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None},
                   "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "incumbent": 0, "checkpoints": [{"id": 0, "model": "checkpoints/0000/model.pt",
                   ("nnue" if args.model == "nnue" else "table"): initial_table, "kind": args.model, "promoted": True,
                   "anchor_elo": 0, "anchor_elo_95pct": None, "evaluations": [], **initial_metadata}]}
        write_json(summary_path, summary)
    try:
        summary.pop("error", None)
        def progress(completed, total, metrics):
            summary.update(stage_completed=completed, stage_total=total, stage_metrics=metrics,
                           updated=time.time())
            write_json(summary_path, summary)

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            start_iteration = summary["iteration"]+1
            for iteration in range(start_iteration, start_iteration+args.iterations):
                incumbent = next(c for c in summary["checkpoints"] if c["id"] == summary["incumbent"])
                learner = summary["checkpoints"][-1]
                pool_members = [c for c in summary["checkpoints"] if c["promoted"]][-4:]
                selfplay_pool = pool_members if any(c["id"] == learner["id"] for c in pool_members) else [*pool_members, learner]
                summary.update(status="self-play", active_iteration=iteration, stage_completed=0, stage_total=args.games)
                summary.update(active_positions=0, active_truncated_games=0, stage_metrics={}, selfplay_model=learner["id"])
                write_json(summary_path, summary)
                tasks = []
                for index in range(args.games):
                    rival = selfplay_pool[(index//2) % len(selfplay_pool)]
                    tables = [str(run / checkpoint_artifact(learner)), str(run / checkpoint_artifact(rival))]
                    if index % 2:
                        tables.reverse()
                    seed = args.seed+iteration*10000+index
                    tasks.append({"tables": tables, "seed": seed,
                                  "evaluation": False, "ms": args.ms, "width": args.width,
                                  "model_kind": args.model, "policy_candidates": args.policy_candidates,
                                  "native_exploration": args.native_exploration,
                                  **task_opening(seed, False, args.max_stones, args.curriculum),
                                  "max_stones": args.max_stones})
                records, batches = [], []
                started = time.perf_counter()
                last_progress = started
                if args.selfplay_backend == "gpu":
                    if args.model == "nnue":
                        from gpu_nnue import generate_games, recommended_batch
                        backend_options = {"beam": args.gpu_beam}
                    else:
                        from gpu_games import generate_games, recommended_batch
                        backend_options = {}
                    gpu_batch = args.gpu_games_batch or min(len(tasks), recommended_batch(args.gpu_candidates, args.max_stones, **backend_options))
                    summary["gpu_games_batch"] = gpu_batch
                    write_json(summary_path, summary)
                    def gpu_results():
                        for offset in range(0, len(tasks), gpu_batch):
                            chunk = tasks[offset:offset+gpu_batch]
                            def gpu_progress(completed, total, metrics):
                                progress(offset+completed, len(tasks), metrics)
                            yield from generate_games(chunk, candidates=args.gpu_candidates,
                                                      epsilon=args.exploration, device="cuda", progress=gpu_progress,
                                                      **backend_options)
                    results = gpu_results()
                else:
                    results = (future.result() for future in as_completed([pool.submit(play_game, task) for task in tasks]))
                for record, samples in results:
                    record["training_positions"] = len(samples["family"])
                    records.append(record);batches.append(samples)
                    summary["active_positions"] += len(samples["family"])
                    summary["active_truncated_games"] += record["winner"] < 0
                    now = time.perf_counter()
                    if args.selfplay_backend == "native" or now-last_progress >= .5 or len(records) == len(tasks):
                        summary["stage_completed"] = len(records)
                        summary["games_per_hour"] = len(records)*3600/(now-started)
                        write_json(summary_path, summary)
                        event(run, "self_play_game", iteration=iteration, completed=len(records),
                              total=args.games, reason=record["reason"], stones=len(record["cells"]))
                        print(f"Self-play {len(records)}/{args.games}: {record['reason']}, {len(record['cells'])} stones", flush=True)
                        last_progress = now
                data_path = run / "data" / f"{iteration:04d}.npz"
                ordered = sorted(zip(records, batches), key=lambda pair: pair[0]["seed"])
                records, batches = map(list, zip(*ordered))
                merged = merge_nnue(batches) if args.model == "nnue" else {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
                np.savez_compressed(data_path, **merged)
                write_json(run / "data" / f"{iteration:04d}-games.json", records)
                summary.update(status="training", stage_completed=0, stage_total=args.epochs, stage_metrics={})
                write_json(summary_path, summary)
                # The learner keeps training across rejected promotion attempts.
                # Only the independently evaluated incumbent plays rated matches.
                challenger = optimize(run, iteration, learner, sorted((run / "data").glob("*.npz")), args, progress)
                summary.update(status="evaluation", stage_completed=0, stage_total=args.eval_games, stage_metrics={})
                write_json(summary_path, summary)
                opponents = [incumbent]
                anchor = summary["checkpoints"][0]
                if incumbent["id"] != 0:
                    opponents.append(anchor)
                older = [c for c in pool_members if c["id"] not in (0, incumbent["id"])]
                if older:
                    opponents.append(older[-1])
                evaluations = [evaluate(pool, run, challenger, opponent, iteration, args, progress) for opponent in opponents]
                direct = evaluations[0]
                anchored = next(e for e in evaluations if e["opponent"] == 0)
                # Pair-level sign test accounts for the two colors of each opening.
                # Alpha spending bounds repeated promotion attempts across the run.
                alpha = .05/(iteration*(iteration+1))
                promoted = direct["incomplete"] == 0 and direct["win_rate"] > .5 and direct["opening_pair_p"] <= alpha
                # Reject obvious regressions against older frozen checkpoints as well.
                promoted = promoted and all(e["incomplete"] == 0 and e["win_rate_95pct"][1] >= .5 for e in evaluations[1:])
                challenger.update(promoted=promoted, promotion_alpha=alpha, evaluations=evaluations,
                                  anchor_elo=anchored["elo_delta"], anchor_elo_95pct=anchored["elo_delta_95pct"])
                summary["checkpoints"].append(challenger)
                if promoted:
                    summary["incumbent"] = iteration
                summary["positions"] += len(merged["family"])
                summary["self_play_games"] += len(records)
                summary["truncated_games"] += sum(g["winner"] < 0 for g in records)
                summary.update(iteration=iteration, status="iteration-complete", active_positions=0,
                               active_truncated_games=0, learner=iteration)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/selfplay")
    parser.add_argument("--model", choices=["pattern", "nnue"], default="pattern")
    parser.add_argument("--initial-model", help="Initialize checkpoint zero of a new NNUE run from model.pt; ratings restart at zero")
    parser.add_argument("--initial-optimizer", help="Optional matching AdamW state; preserves moments and steps, uses --lr")
    parser.add_argument("--curriculum", choices=["legacy", "mixed-v1"], default="mixed-v1")
    parser.add_argument("--native-exploration", type=float, default=.05,
                        help="Quiet native turns to explore without adding teacher labels")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--ms", type=int, default=50)
    parser.add_argument("--eval-ms", type=int, default=100)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--max-stones", type=int, default=200)
    parser.add_argument("--eval-max-stones", type=int, default=800, help="Separate evaluation cap; unfinished matches remain unrated")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256, help="0: pattern uses up to 8192 CUDA positions with reference updates; NNUE uses 256 bounded by sparse centers")
    parser.add_argument("--updates-per-epoch", type=int, default=0, help="0 uses the batch-derived update count; positive values set an explicit optimizer-step budget")
    parser.add_argument("--selfplay-backend", choices=["native", "gpu"], default="native")
    parser.add_argument("--gpu-games-batch", type=int, default=0, help="0 sizes simultaneous GPU games from available VRAM")
    parser.add_argument("--gpu-candidates", type=int, default=32)
    parser.add_argument("--gpu-beam", type=int, default=4, help="NNUE complete-turn GPU beam width")
    parser.add_argument("--exploration", type=float, default=.1, help="Random legal candidate probability for GPU self-play")
    parser.add_argument("--lr", type=float, default=.002)
    parser.add_argument("--replay-iterations", type=int, default=8)
    parser.add_argument("--reanalysis", nargs="+", default=[], metavar="DIRECTORY",
                        help="Completed external NNUE reanalysis directories; retained outside the chronological replay window")
    parser.add_argument("--replay-positions", type=int, default=250000, help="Maximum uniformly sampled replay positions; CUDA also reserves memory for training")
    parser.add_argument("--nnue-replay-centers", type=int, default=2000000, help="Bound NNUE replay by sparse center count")
    parser.add_argument("--nnue-batch-centers", type=int, default=32768, help="Split NNUE batches by center count as well as positions")
    parser.add_argument("--policy-candidates", type=int, default=32)
    parser.add_argument("--policy-weight", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if any(getattr(args, k) < 1 for k in ("iterations", "games", "eval_games", "ms", "eval_ms", "workers", "epochs", "replay_iterations", "replay_positions")):
        parser.error("Counts and search budgets must be positive")
    if args.batch < 0 or args.updates_per_epoch < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Batch must be nonnegative and learning rate must be finite and positive")
    if args.gpu_games_batch < 0 or args.gpu_candidates < 1 or not 0 <= args.exploration <= 1:
        parser.error("GPU batch must be nonnegative, candidate count positive, exploration in [0,1]")
    if args.selfplay_backend == "gpu" and args.device == "cpu":
        parser.error("GPU self-play requires device auto or cuda")
    if not 1 <= args.gpu_beam <= 16:
        parser.error("GPU NNUE beam must be in 1..16")
    if args.nnue_replay_centers < 1 or args.nnue_batch_centers < 1 or not 2 <= args.policy_candidates <= 128:
        parser.error("NNUE center limits must be positive and policy candidates in 2..128")
    if not math.isfinite(args.policy_weight) or args.policy_weight < 0:
        parser.error("Policy weight must be finite and nonnegative")
    if not 0 <= args.native_exploration <= 1:
        parser.error("Native exploration must be in [0,1]")
    if args.eval_games % 2 or args.max_stones < 5 or args.eval_max_stones < 5 or not 2 <= args.width <= 128:
        parser.error("Evaluation games must be even, both stone caps >= 5, width in 2..128")
    run_training(args)
