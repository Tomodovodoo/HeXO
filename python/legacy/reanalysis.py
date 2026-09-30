"""Revisit saved NNUE histories with bounded native search; never edit source data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np

from hexo import Game, ROOT, library
from legacy.train import merge_nnue, nnue_example, pack_nnue, write_json


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


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


def external_replays(directories):
    """Validate completed shards and return the immutable trainer input snapshot."""
    result = []
    for directory in directories:
        directory = Path(directory).resolve()
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        provenance = manifest["provenance"]
        if manifest.get("status") != "finished" or provenance.get("schema") != 1 or manifest.get("replay") != "replay.npz":
            raise ValueError(f"Reanalysis shard is not completed schema 1: {directory}")
        for key in ("history_sha256", "model_sha256", "engine_sha256"):
            value = provenance.get(key, "")
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Missing reanalysis provenance: {key}")
        if not provenance.get("sources") or not provenance.get("search") or not provenance.get("target_semantics"):
            raise ValueError("Incomplete reanalysis source/search provenance")
        for value in provenance["sources"].values():
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Malformed reanalysis source hash")
        if any(provenance["search"].get(key, 0) < 1 for key in ("ms", "width", "depth", "policy_candidates")):
            raise ValueError("Malformed reanalysis search budget")
        if len(manifest["completed"]) != len(provenance["positions"]):
            raise ValueError("Reanalysis manifest has unfinished positions")
        path = directory / "replay.npz"
        sha = digest(path)
        if sha != manifest["replay_sha256"]:
            raise ValueError(f"Reanalysis replay hash changed: {path}")
        with np.load(path, allow_pickle=False) as data:
            if data["nnue_schema"].tolist() != [1]:
                raise ValueError("Reanalysis requires NNUE replay schema 1")
            n = len(data["family"])
            if not n or n != manifest["rows"]:
                raise ValueError("Reanalysis row count disagrees with manifest")
            for offsets, key in (("center_offsets", "centers"), ("candidate_offsets", "candidate_codes")):
                values = data[offsets]
                if values.shape != (n+1,) or values[0] != 0 or values[-1] != len(data[key]) or np.any(np.diff(values) <= 0):
                    raise ValueError(f"Malformed reanalysis {offsets}")
            for key, width in (("centers", 3), ("candidate_codes", 3), ("pairs", 4), ("candidate_coords", 2)):
                if data[key].ndim != 2 or data[key].shape[1] != width:
                    raise ValueError(f"Malformed reanalysis {key}")
            for key in ("centers", "candidate_codes"):
                if data[key].dtype.kind not in "iu" or np.any(data[key] < 0) or np.any(data[key] >= 3**11):
                    raise ValueError(f"Invalid reanalysis centered-line codes: {key}")
            for key in ("pairs", "candidate_coords"):
                if len(data[key]) != len(data["candidate_codes"]):
                    raise ValueError("Mismatched reanalysis candidate arrays")
            for key in ("player", "baseline", "chosen", "search", "search_valid", "policy_valid", "ply", "search_depth", "search_ms", "outcome"):
                if data[key].shape != (n,):
                    raise ValueError(f"Malformed reanalysis {key}")
            if data["phase"].shape != (n, 4) or np.any(data["chosen"] < 0) or np.any(data["chosen"] >= np.diff(data["candidate_offsets"])):
                raise ValueError("Malformed reanalysis phase or teacher choice")
            for key in ("phase", "baseline", "pairs", "search", "search_ms"):
                if not np.isfinite(data[key]).all():
                    raise ValueError(f"Nonfinite reanalysis {key}")
            if not np.isin(data["player"], [0, 1]).all() or np.any(np.abs(data["search"]) > 1):
                raise ValueError("Invalid reanalysis player or value target")
            if np.any(~np.isnan(data["outcome"]) & ~np.isin(data["outcome"], [-1, 1])):
                raise ValueError("Invalid reanalysis outcome")
            recorded_rows = [(row["ply"], item["family"]) for item in manifest["completed"] for row in item["rows"]]
            if recorded_rows != list(zip(data["ply"].tolist(), data["family"].tolist())):
                raise ValueError("Replay geometry provenance disagrees with manifest")
        result.append({"path": str(path), "sha256": sha, "manifest": str(manifest_path),
                       "manifest_sha256": digest(manifest_path), "provenance": provenance})
    if len({x["path"] for x in result}) != len(result):
        raise ValueError("Duplicate reanalysis shard")
    return result


def replay_inputs(run, ordinary, args):
    """Apply the chronological window only to self-play, then append external shards."""
    paths = list(ordinary[-args.replay_iterations:])
    metadata = [{"path": str(p.relative_to(run)), "sha256": digest(p)} for p in paths]
    external = external_replays(getattr(args, "reanalysis", []))
    if external != getattr(args, "external_replay", []):
        raise ValueError("External reanalysis changed since run initialization")
    paths.extend(Path(item["path"]) for item in external)
    metadata.extend({"external": True, **item} for item in external)
    return paths, metadata


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
        "sources": {name: digest(ROOT/'python'/('legacy/'+name if name != 'hexo.py' else name)) for name in ("reanalysis.py", "train.py", "hexo.py", "nnue_model.py")},
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
