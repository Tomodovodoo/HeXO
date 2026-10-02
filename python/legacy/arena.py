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
    def __init__(self, adapter=None):
        self.lib = C.CDLL(str(adapter or library.with_name(library.name.replace("hexo", "hexo_seal"))))
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
    probe = None
    learned = None
    if getattr(args, "strix_root_ms", 0):
        from legacy.strix_root import StrixRoot
        probe = StrixRoot(args.strix_root_ms, args.ms, nodes=args.strix_root_nodes,
                          depth=args.strix_root_depth, wide=args.strix_root_wide)
    try:
        if args.opponent == "strix":
            from tools.strix_learned_adapter import StrixLearned
            learned = StrixLearned(args.strix_model, args.strix_sims, args.strix_actions,
                                   args.strix_timeout_ms, args.seed)
            learned.warm_up()
        if probe:
            probe.warm_up()
        return run_matches(args, probe, learned)
    finally:
        if probe:
            probe.close()
        if learned:
            learned.close()


def run_matches(args, probe=None, learned=None):
    rng = random.Random(args.seed)
    opponent = learned or (Seal() if args.opponent == "seal" else None)
    opponent_metadata = learned.metadata if learned else None
    if args.opponent == "seal-current-best":
        from tools.seal_current import SealCurrent
        opponent = SealCurrent()
        opponent_metadata = opponent.metadata
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
    nnue_path = None
    if args.run or args.table or args.nnue:
        if args.run:
            run_dir = Path(args.run).resolve()
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            checkpoint = summary["incumbent"] if args.checkpoint is None else args.checkpoint
            descriptor = next((c for c in summary["checkpoints"] if c["id"] == checkpoint), None)
            if descriptor is None:
                raise ValueError(f"Checkpoint {checkpoint} is not present in {run_dir}")
            nnue_path = run_dir / descriptor["nnue"] if descriptor.get("kind") == "nnue" else None
            path = nnue_path or run_dir / descriptor["table"]
        else:
            nnue_path = Path(args.nnue).resolve() if args.nnue else None
            path = nnue_path or Path(args.table).resolve()
            checkpoint = None
        if nnue_path is None:
            import numpy as np
            table = np.load(path, allow_pickle=False)
            if table.shape != (729,) or table.dtype != np.int32:
                raise ValueError("Expected a 729-entry int32 native pattern table")
            weights = table.tolist()
        model = {"checkpoint": checkpoint, "kind": "nnue" if nnue_path else "pattern", "path": str(path),
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for index in range(args.games):
        if index % 2 == 0:
            opening = [(0, 0)]
            # Nearby legal openings, paired with colors exchanged.
            candidates = [(q, r) for q in range(-2, 3) for r in range(-2, 3)
                          if 0 < max(abs(q), abs(r), abs(q+r)) <= 2]
            opening += rng.sample(candidates, 2)
        game = Game(opening)
        if args.opponent == "seal-current-best":
            opponent.reset()
        our_color = index % 2
        timings = [[], []]
        searches = []
        turns = []
        opponent_searches = []
        reason = "truncated"
        error = None
        while game.winner < 0 and len(game.cells) < args.max_stones:
            if len(game.cells) + game.remaining > args.max_stones:
                break
            side = game.player
            before = time.perf_counter()
            try:
                if side == our_color:
                    if nnue_path:
                        game.load_model(nnue_path)
                    else:
                        game.load_table(weights)
                    if probe:
                        search = probe.search(game, args.ms, start=before, width=args.width,
                                              root_seconds=args.root_seconds, root_turns=args.root_turns,
                                              tt_injection=args.tt_injection)
                    else:
                        search = game.search(args.ms, width=args.width, root_seconds=args.root_seconds,
                                             root_turns=args.root_turns, tt_injection=args.tt_injection)
                    searches.append(search)
                    moves = search["moves"]
                elif opponent:
                    moves = opponent(game, args.ms)
                    if learned:
                        opponent_searches.append(learned.last_result)
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
                elapsed_ms = (time.perf_counter()-before)*1000
                timings[side].append(elapsed_ms)
                turns.append({"player": side, "ply": len(game.cells), "moves": moves,
                              "elapsed_ms": elapsed_ms, "requested_ms": args.ms})
                if probe and side == our_color:
                    overrun = elapsed_ms > args.ms
                    probe.counts["overruns"] += int(overrun)-int(search["strix"]["overrun"])
                    search["elapsed_ms"] = elapsed_ms
                    search["strix"]["overrun"] = overrun
                if not moves:
                    raise ValueError("Opponent returned no moves")
                for move in moves:
                    game.play(*move)
                    if game.winner >= 0:
                        break
                if game.winner < 0 and game.player == side:
                    raise ValueError("Opponent did not complete its turn")
            except (ValueError, RuntimeError) as exc:
                if learned and side != our_color and learned.last_result:
                    opponent_searches.append(learned.last_result)
                reason, error = "invalid", str(exc)
                break
        if game.winner >= 0:
            reason = "six-in-a-row"
        record = {"index": index, "our_color": our_color, "winner": game.winner,
                  "reason": reason, "error": error, "cells": game.cells, "time_ms": timings,
                  "searches": searches, "turns": turns, "opening": opening,
                  "opponent_searches": opponent_searches}
        games.append(record)
        completed = [g for g in games if g["reason"] == "six-in-a-row"]
        wins = sum(g["winner"] == g["our_color"] for g in completed)
        pairs = (len(games)+1)//2
        paired_games = 2*pairs
        censored = paired_games-len(completed)
        margin = math.sqrt(math.log(40)/(2*pairs))
        report = {"revision": revision, "dirty": dirty, "platform": platform.platform(),
                  "model": model,
                  "strix_root": probe.metadata() if probe else None,
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
    parser.add_argument("--opponent", choices=["seal", "seal-current-best", "orca", "strix", "shallow", "random"], default="shallow")
    parser.add_argument("--strix-model", help="Pinned public Strix safetensors checkpoint")
    parser.add_argument("--strix-sims", type=int, default=8, help="Gumbel MCTS simulations per placement; not equal wall time")
    parser.add_argument("--strix-actions", type=int, default=4)
    parser.add_argument("--strix-timeout-ms", type=int, default=5000, help="Safety deadline for one complete Strix turn")
    parser.add_argument("--orca-source", help="External hexbot-building-framework checkout")
    parser.add_argument("--orca-checkpoint", help="Defaults to orca/checkpoint.pt within --orca-source")
    parser.add_argument("--orca-sims", type=int, default=200, help="MCTS simulations per placement, not a time budget")
    parser.add_argument("--orca-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--ms", type=int, default=100)
    parser.add_argument("--strix-root-ms", type=float, default=0,
                        help="Optional Strix root probe budget; 0 disables, must be smaller than --ms")
    parser.add_argument("--strix-root-nodes", type=int, default=1000, help="IDTT node cap before native fallback")
    parser.add_argument("--strix-root-depth", type=int, default=8, help="Attacker-turn horizon including winning turn")
    parser.add_argument("--strix-root-wide", action="store_true", help="Use Strix's wider attacking-partner generator")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--tt-injection", action="store_true", help="Experimental previous-iteration TT turn admission; disables TT score cutoffs")
    parser.add_argument("--root-seconds", type=int, default=0, help="Experimental conditional-second budget; requires --root-turns")
    parser.add_argument("--root-turns", type=int, default=0, help="Experimental complete-turn root cap; requires --root-seconds")
    parser.add_argument("--max-stones", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", default="artifacts/arena.json")
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument("--run", help="Run directory; defaults to its promoted checkpoint")
    model_source.add_argument("--table", help="Exported native table.npy")
    model_source.add_argument("--nnue", help="Exported native model.nnue")
    parser.add_argument("--checkpoint", type=int, help="Specific checkpoint in --run, including rejected candidates")
    args = parser.parse_args()
    if args.games < 1 or args.ms < 1 or args.max_stones < 3:
        parser.error("games and ms must be positive; max-stones must be at least 3")
    if not math.isfinite(args.strix_root_ms) or not 0 <= args.strix_root_ms < args.ms:
        parser.error("Strix root budget must be finite, nonnegative and smaller than --ms")
    if not 1 <= args.strix_root_nodes < 1 << 64 or not 1 <= args.strix_root_depth <= 255:
        parser.error("Strix node cap must be positive unsigned64 and depth in 1..255")
    if not 2 <= args.width <= 128:
        parser.error("width must be in 2..128")
    if (args.root_seconds or args.root_turns) and not (max(6, args.width//2) <= args.root_seconds <= 128 and 2*args.width <= args.root_turns <= 1024):
        parser.error("Root budgets must both be zero, or seconds in max(6,width/2)..128 and turns in 2*width..1024")
    if args.checkpoint is not None and not args.run:
        parser.error("--checkpoint requires --run")
    if args.opponent == "orca" and (not args.orca_source or args.orca_sims < 1):
        parser.error("Orca requires --orca-source and positive --orca-sims")
    if args.opponent == "strix" and (not args.strix_model or args.max_stones > 800
            or not 1 <= args.strix_sims <= 100000 or not 1 <= args.strix_actions <= 1024
            or not 1 <= args.strix_timeout_ms <= 600000):
        parser.error("Strix requires --strix-model, max-stones<=800, simulations1..100000, actions1..1024 and timeout1..600000ms")
    run(args)
