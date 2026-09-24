"""Revisit saved NNUE histories with bounded native search; never edit source data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np

from hexo import Game, ROOT, library
from train import merge_nnue, nnue_example, pack_nnue, write_json


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_history(record, curriculum="legacy"):
    cells = record["cells"]
    opening = record.get("opening")
    prefix = record.get("prefix_length", len(opening) if opening else None)
    if prefix is None:
        if record.get("curriculum", curriculum) != "legacy":
            raise ValueError("Variable-prefix history is missing its opening length")
        prefix = 3
    if not 1 <= prefix <= len(cells):
        raise ValueError("Invalid prefix length")
    if opening is not None and [list(c[:2]) for c in cells[:prefix]] != opening:
        raise ValueError("Opening does not match history")
    game = Game()
    for q, r, owner in cells:
        if game.winner >= 0 or game.player != owner:
            raise ValueError("Illegal history or incorrect stone owner")
        game.play(q, r)
    if game.winner != record["winner"]:
        raise ValueError("Recorded winner disagrees with reconstructed board")
    if (game.winner >= 0) != (record["reason"] == "six-in-a-row"):
        raise ValueError("Terminal reason disagrees with reconstructed board")
    return prefix


def reconstruct(record, ply):
    if not 0 <= ply < len(record["cells"]):
        raise ValueError("Source ply must name a pre-placement state")
    return Game([c[:2] for c in record["cells"][:ply]])


def analyze(record, ply, model, ms, width, depth, candidates):
    game = reconstruct(record, ply)
    game.load_model(model)
    result = game.search(ms, depth=depth, width=width)
    if not result["moves"]:
        raise ValueError("Teacher returned no move for a nonterminal source state")
    outcome = float("nan") if record["winner"] < 0 else (1 if record["winner"] == 0 else -1)
    batches, row_metadata = [], []
    on_trajectory = True
    side = game.player
    for move in result["moves"]:
        if game.winner >= 0 or game.player != side:
            raise ValueError("Teacher returned a move beyond the current turn")
        row = nnue_example(game, tuple(move), result, candidates)
        batches.append(pack_nnue([row], outcome if on_trajectory else float("nan"), record["family"]))
        row_metadata.append({"ply": len(game.cells), "on_source_trajectory": on_trajectory})
        actual = record["cells"][len(game.cells)][:2] if len(game.cells) < len(record["cells"]) else None
        on_trajectory &= actual == list(move)
        game.play(*move)
    return merge_nnue(batches), {"moves": result["moves"], "score": result["score"],
        "depth": result["depth"], "nodes": result["nodes"], "elapsed_ms": result["elapsed_ms"],
        "rows": row_metadata, "score_semantics": "selective-root-estimate" if result["depth"] > 0 else "incomplete-search-fallback"}


def save_npz(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **data)
    os.replace(temporary, path)


def run(args):
    source = args.run.resolve()
    output = args.output.resolve()
    history_path = source / "data" / f"{args.iteration:04d}-games.json"
    summary = json.loads((source / "summary.json").read_text())
    checkpoint = next(c for c in summary["checkpoints"] if c["id"] == args.checkpoint)
    if checkpoint.get("kind") != "nnue":
        raise ValueError("Reanalysis requires an explicit NNUE evaluator checkpoint")
    model = source / checkpoint["nnue"]
    if output == source or output == history_path.parent or output == model.parent:
        raise ValueError("Choose a separate reanalysis output directory")
    records = json.loads(history_path.read_text())
    prefixes = [validate_history(r, summary["config"].get("curriculum", "legacy")) for r in records]
    counts = [len(r["cells"])-p for r, p in zip(records, prefixes)]
    offsets = np.cumsum([0]+counts)
    selected = sorted(random.Random(args.seed).sample(range(int(offsets[-1])), min(args.max_positions, int(offsets[-1]))))
    positions = []
    for index in selected:
        game = int(np.searchsorted(offsets, index, side="right")-1)
        positions.append([game, int(index-offsets[game]+prefixes[game])])
    if not positions:
        raise ValueError("No eligible source positions")
    provenance = {"schema": 1, "source_run": str(source), "iteration": args.iteration,
        "history_sha256": digest(history_path), "checkpoint": args.checkpoint,
        "model": str(model), "model_sha256": digest(model), "seed": args.seed,
        "max_positions": args.max_positions, "search": {"ms": args.ms, "width": args.width,
        "depth": args.depth, "policy_candidates": args.policy_candidates},
        "source_actor_config": summary["config"],
        "sources": {name: digest(ROOT/name) for name in ("reanalysis.py", "train.py", "hexo.py", "nnue_model.py")},
        "engine_sha256": digest(library), "positions": positions,
        "target_semantics": "Root score is a selective search estimate, shared by its conditional continuation; no exact proof labels. Outcomes retained only while teacher prefix matches source history."}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["provenance"] != provenance:
            raise ValueError("Resume provenance differs; choose a new output directory")
    else:
        if any(output.iterdir()):
            raise ValueError("Output directory must be empty or contain a matching manifest")
        manifest = {"provenance": provenance, "completed": [], "status": "running"}
        write_json(manifest_path, manifest)
    # One durable fragment per bounded root search permits restart without relabeling
    # completed positions. An uncommitted fragment is replaced on restart.
    for index, (game, ply) in enumerate(positions):
        path = output / f"position-{index:06d}.npz"
        if index < len(manifest["completed"]):
            if digest(path) != manifest["completed"][index]["sha256"]:
                raise ValueError("Completed replay fragment changed")
            continue
        data, metadata = analyze(records[game], ply, model, args.ms, args.width, args.depth, args.policy_candidates)
        save_npz(path, data)
        manifest["completed"].append({"game": game, "source_ply": ply, "family": records[game]["family"],
            "opening": [c[:2] for c in records[game]["cells"][:prefixes[game]]],
            "file": path.name, "sha256": digest(path), **metadata})
        write_json(manifest_path, manifest)
        print(f"Reanalyzed {index+1}/{len(positions)} roots", flush=True)
    batches = []
    for item in manifest["completed"]:
        with np.load(output/item["file"], allow_pickle=False) as saved:
            batches.append(dict(saved))
    replay = merge_nnue(batches)
    save_npz(output / "replay.npz", replay)
    manifest.update(status="finished", replay="replay.npz", replay_sha256=digest(output/"replay.npz"), rows=len(replay["family"]))
    write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--checkpoint", type=int, required=True, help="Frozen evaluator checkpoint ID")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-positions", type=int, default=256, help="Maximum source root searches; each emits up to two rows")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--ms", type=int, default=200)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--policy-candidates", type=int, default=32)
    args = parser.parse_args()
    if min(args.max_positions, args.ms, args.width, args.depth, args.policy_candidates) < 1:
        parser.error("Search and replay budgets must be positive")
    run(args)


if __name__ == "__main__":
    main()
