"""Reproducible local matches with alternating colors and paired openings."""
import argparse
import ctypes as C
import hashlib
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path
from hexo import Game, ROOT, library


class Seal:
    def __init__(self):
        adapter = library.with_name(library.name.replace("hexo", "hexo_seal"))
        self.lib = C.CDLL(str(adapter))
        self.fn = self.lib.seal_move
        self.fn.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_int)]
        self.fn.restype = C.c_int

    def __call__(self, game, ms):
        cells = game.cells
        data = (C.c_int * (3 * len(cells)))(*(v for cell in cells for v in cell))
        out = (C.c_int * 4)()
        count = self.fn(data, len(cells), game.player, game.remaining, ms, out)
        if count < 0:
            raise RuntimeError("Seal board range exceeded")
        return [(out[2*i], out[2*i+1]) for i in range(count)]


def wilson(wins, games):
    if not games:
        return [0, 1]
    z = 1.96
    p = wins / games
    d = 1 + z*z/games
    center = (p + z*z/(2*games))/d
    radius = z*math.sqrt(p*(1-p)/games + z*z/(4*games*games))/d
    return [center-radius, center+radius]


def run(args):
    rng = random.Random(args.seed)
    opponent = Seal() if args.opponent == "seal" else None
    opponent_metadata = None
    if args.opponent == "orca":
        from tools.orca_adapter import Orca
        opponent = Orca(args.orca_source, args.orca_checkpoint, args.orca_sims,
                        args.orca_device, args.max_stones)
        opponent_metadata = opponent.metadata
    games = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    model = None
    weights = [0]*729
    if args.run or args.table:
        import numpy as np
        if args.run:
            run_dir = Path(args.run).resolve()
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            checkpoint = summary["incumbent"] if args.checkpoint is None else args.checkpoint
            descriptor = next((c for c in summary["checkpoints"] if c["id"] == checkpoint), None)
            if descriptor is None:
                raise ValueError(f"Checkpoint {checkpoint} is not present in {run_dir}")
            path = run_dir / descriptor["table"]
        else:
            path = Path(args.table).resolve()
            checkpoint = None
        table = np.load(path, allow_pickle=False)
        if table.shape != (729,) or table.dtype != np.int32:
            raise ValueError("Expected a 729-entry int32 native pattern table")
        weights = table.tolist()
        model = {"checkpoint": checkpoint, "table": str(path),
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for index in range(args.games):
        if index % 2 == 0:
            opening = [(0, 0)]
            # Nearby legal openings, paired with colors exchanged.
            candidates = [(q, r) for q in range(-2, 3) for r in range(-2, 3)
                          if 0 < max(abs(q), abs(r), abs(q+r)) <= 2]
            opening += rng.sample(candidates, 2)
        game = Game(opening)
        our_color = index % 2
        timings = [[], []]
        searches = []
        reason = "truncated"
        error = None
        while game.winner < 0 and len(game.cells) < args.max_stones:
            if len(game.cells) + game.remaining > args.max_stones:
                break
            side = game.player
            before = time.perf_counter()
            try:
                if side == our_color:
                    game.load_table(weights)
                    search = game.search(args.ms, width=args.width)
                    searches.append(search)
                    moves = search["moves"]
                elif opponent:
                    moves = opponent(game, args.ms)
                elif args.opponent == "shallow":
                    game.load_table([0]*729)
                    moves = game.search(args.ms, depth=1, width=args.width)["moves"]
                else:
                    moves = []
                    for _ in range(game.remaining):
                        move = rng.choice(game.legal_moves())
                        moves.append(move)
                        game.play(*move)
                        if game.winner >= 0:
                            break
                    for _ in moves:
                        game.undo()
                timings[side].append((time.perf_counter()-before)*1000)
                if not moves:
                    raise ValueError("Opponent returned no moves")
                for move in moves:
                    game.play(*move)
                    if game.winner >= 0:
                        break
                if game.winner < 0 and game.player == side:
                    raise ValueError("Opponent did not complete its turn")
            except (ValueError, RuntimeError) as exc:
                reason, error = "invalid", str(exc)
                break
        if game.winner >= 0:
            reason = "six-in-a-row"
        record = {"index": index, "our_color": our_color, "winner": game.winner,
                  "reason": reason, "error": error, "cells": game.cells, "time_ms": timings,
                  "searches": searches}
        games.append(record)
        completed = [g for g in games if g["reason"] == "six-in-a-row"]
        wins = sum(g["winner"] == g["our_color"] for g in completed)
        pairs = (len(games)+1)//2
        paired_games = 2*pairs
        censored = paired_games-len(completed)
        margin = math.sqrt(math.log(40)/(2*pairs))
        report = {"revision": revision, "dirty": dirty, "platform": platform.platform(),
                  "model": model,
                  "engine_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                  "source_sha256": hashlib.sha256((ROOT / "src/hexo.cpp").read_bytes()).hexdigest(),
                  "opponent_revision": (ROOT / "build/seal_revision.txt").read_text().strip() if args.opponent == "seal" else (opponent_metadata or {}).get("revision"),
                  "opponent_metadata": opponent_metadata,
                  "config": vars(args), "wins": wins, "losses": len(completed)-wins,
                  "incomplete": len(games)-len(completed),
                  "unplayed_pair_partners": paired_games-len(games),
                  "win_rate_95pct": [max(0, wins/paired_games-margin), min(1, (wins+censored)/paired_games+margin)],
                  "interval_method": "Opening-pair Hoeffding 95%; incomplete games and unplayed pair partners bounded",
                  "completed_only_wilson_95pct": wilson(wins, len(completed)),
                  "games": games}
        output.write_text(json.dumps(report, indent=2))
        print(f"Game {index+1}: {reason}, winner {game.winner}; {wins} wins / {len(completed)-wins} losses / {len(games)-len(completed)} incomplete", flush=True)
        game.close()
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", choices=["seal", "orca", "shallow", "random"], default="shallow")
    parser.add_argument("--orca-source", help="External hexbot-building-framework checkout")
    parser.add_argument("--orca-checkpoint", help="Defaults to orca/checkpoint.pt within --orca-source")
    parser.add_argument("--orca-sims", type=int, default=200, help="MCTS simulations per placement, not a time budget")
    parser.add_argument("--orca-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--ms", type=int, default=100)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--max-stones", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", default="artifacts/arena.json")
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument("--run", help="Run directory; defaults to its promoted checkpoint")
    model_source.add_argument("--table", help="Exported native table.npy")
    parser.add_argument("--checkpoint", type=int, help="Specific checkpoint in --run, including rejected candidates")
    args = parser.parse_args()
    if args.games < 1 or args.ms < 1 or args.max_stones < 3:
        parser.error("games and ms must be positive; max-stones must be at least 3")
    if args.checkpoint is not None and not args.run:
        parser.error("--checkpoint requires --run")
    if args.opponent == "orca" and (not args.orca_source or args.orca_sims < 1):
        parser.error("Orca requires --orca-source and positive --orca-sims")
    run(args)
