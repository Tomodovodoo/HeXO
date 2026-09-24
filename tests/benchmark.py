"""Bounded, seed-controlled measurements; no pass/fail throughput threshold."""
import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

from hexo import Game, ROOT, library
from tests.reference import Reference


def measure(args):
    rng = random.Random(args.seed)
    samples = []
    for index in range(args.positions):
        game, reference = Game(), Reference()
        try:
            for _ in range(3+2*(index % 8)):
                while True:
                    q, r = rng.choice(reference.history) if reference.history else (0, 0)
                    move = (q+rng.randrange(-3, 4), r+rng.randrange(-3, 4)) if reference.history else (0, 0)
                    if reference.legal(*move):
                        break
                game.play(*move)
                reference.play(*move)
                if reference.winner >= 0:
                    break
            if game.winner >= 0:
                continue
            key, state, features = game.key, game.state(), game.features()
            start = time.perf_counter()
            cheap = game.search(args.ms, width=args.width)
            wall = (time.perf_counter()-start)*1000
            assert (game.key, game.state(), game.features()) == (key, state, features)
            wide = game.search(args.reference_ms, width=args.reference_width)
            assert (game.key, game.state(), game.features()) == (key, state, features)
            candidate_recall = None
            # Available once the NNUE/native candidate API lands. Comparing final
            # moves alone is not candidate recall and is reported separately.
            if hasattr(game, "candidates"):
                retained = True
                for move in wide["moves"]:
                    retained &= move in game.candidates(args.width)
                    game.play(*move)
                for _ in wide["moves"]:
                    game.undo()
                assert (game.key, game.state(), game.features()) == (key, state, features)
                candidate_recall = retained
            samples.append({"history": state["cells"], "cheap": cheap, "reference": wide,
                            "measured_ms": wall, "same_turn": set(cheap["moves"]) == set(wide["moves"]),
                            "reference_cells_in_candidate_lists": candidate_recall})
        finally:
            game.close()
    times = [s["measured_ms"] for s in samples]
    recalls = [s["reference_cells_in_candidate_lists"] for s in samples
               if s["reference_cells_in_candidate_lists"] is not None]
    return {"config": vars(args), "platform": platform.platform(), "processor": platform.processor(),
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
            "engine_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256((ROOT/"src/hexo.cpp").read_bytes()).hexdigest(),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "median_wall_ms": statistics.median(times), "maximum_wall_ms": max(times),
            "mean_nodes": statistics.mean(s["cheap"]["nodes"] for s in samples),
            "turn_agreement": statistics.mean(s["same_turn"] for s in samples),
            "candidate_cell_recall": statistics.mean(recalls) if recalls else None,
            "notes": "Wider search is a selective reference, not ground truth. Candidate-cell recall does not measure final turn-list retention.",
            "samples": samples}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--positions", type=int, default=12)
    parser.add_argument("--ms", type=int, default=5)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--reference-ms", type=int, default=25)
    parser.add_argument("--reference-width", type=int, default=32)
    args = parser.parse_args()
    if min(args.positions, args.ms, args.reference_ms) < 1 or not 2 <= args.width <= 128 or not 2 <= args.reference_width <= 128:
        parser.error("Counts/budgets must be positive and widths within 2..128")
    print(json.dumps(measure(args), indent=2))
