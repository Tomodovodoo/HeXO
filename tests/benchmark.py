"""Bounded, seed-controlled measurements; no pass/fail throughput threshold."""
import argparse
import hashlib
import io
import json
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

from hexo import Game, ROOT, library
from tests.reference import Reference


def pair_admission(args):
    """Measure final ordered-turn admission on recorded development positions."""
    import ctypes as C
    import numpy as np

    trace_bytes = Path(args.trace).read_bytes()
    trace_digest = hashlib.sha256(trace_bytes).hexdigest()
    trace = json.loads(trace_bytes)
    model = trace.get("model")
    table_path = None
    table_digest = None
    table = [0]*729
    if model:
        if model.get("kind", "pattern") != "pattern":
            raise ValueError("Trace benchmark requires a pattern-table or handwritten evaluator")
        # Older arena reports stored the pattern path under "table".
        stored_path = model.get("path", model.get("table"))
        if stored_path is None:
            raise ValueError("Trace model does not identify its pattern table")
        table_path = Path(stored_path)
        table_bytes = table_path.read_bytes()
        table_digest = hashlib.sha256(table_bytes).hexdigest()
        if table_digest != model.get("sha256"):
            raise ValueError("Trace pattern table SHA-256 does not match the recorded evaluator")
        array = np.load(io.BytesIO(table_bytes), allow_pickle=False)
        if array.shape != (729,) or array.dtype != np.int32:
            raise ValueError("Trace model must contain a 729-entry int32 pattern table")
        table = array.tolist()
    references = {}
    if args.reference_report:
        reference_report = json.loads(Path(args.reference_report).read_text())
        if reference_report.get("trace_sha256") != trace_digest:
            raise ValueError("Reference report does not match the current trace SHA-256")
        references = {(s["game"], s["stones"]): s
                      for s in reference_report["samples"]}
    seal = None
    if args.seal_library:
        seal_library = C.CDLL(str(Path(args.seal_library).resolve()))
        seal = seal_library.seal_move
        seal.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_int)]
        seal.restype = C.c_int
    samples = []
    for record in trace["games"]:
        if record["index"] < args.first_game or record["winner"] not in (0, 1) or record["winner"] == record["our_color"]:
            continue
        searches = record["searches"]
        if not searches:
            continue
        loss = next((i for i, s in enumerate(searches) if s["score"] <= -10000000), len(searches)-1)
        selected = searches[max(0, loss-2)]
        first = selected["moves"][0]
        count = next(i for i, cell in enumerate(record["cells"]) if cell[:2] == first)
        if count < 9:
            continue
        history = [tuple(cell[:2]) for cell in record["cells"][:count]]
        game = Game(history, table)
        try:
            before = (game.key, game.state(), game.features())
            if references:
                reference_sample = references[record["index"], count]
                if [tuple(point) for point in reference_sample["history"]] != history:
                    raise ValueError("Reference sample history differs from the current position")
                target = [tuple(point) for point in reference_sample["reference_turn"]]
            elif seal:
                cells = game.cells
                data = (C.c_int*(3*len(cells)))(*(v for cell in cells for v in cell))
                output = (C.c_int*4)()
                n = seal(data, len(cells), game.player, game.remaining, args.reference_ms, output)
                if n not in (1, 2):
                    raise RuntimeError("Seal reference failed")
                target = [(output[2*i], output[2*i+1]) for i in range(n)]
            else:
                target = game.search(args.reference_ms, width=args.reference_width)["moves"]
            for point in target:
                if not game.legal(*point):
                    raise ValueError("Reference returned an illegal ordered turn")
                game.play(*point)
            for _ in target:
                game.undo()
            variants = {"baseline": {}, "diverse": {"root_seconds": args.root_seconds,
                                                       "root_turns": args.root_turns}}
            selected_turns, generation_ms = {}, {}
            for name, settings in variants.items():
                start = time.perf_counter()
                selected_turns[name] = [tuple(t["moves"]) for t in game.turns(args.width, **settings)]
                generation_ms[name] = (time.perf_counter()-start)*1000
            assert set(selected_turns["baseline"]) <= set(selected_turns["diverse"])
            trials = []
            for repeat in range(args.repeats):
                order = ("baseline", "diverse") if (len(samples)+repeat) % 2 else ("diverse", "baseline")
                results = {}
                for name in order:
                    start = time.perf_counter()
                    result = game.search(args.ms, width=args.width, **variants[name])
                    result["wall_ms"] = (time.perf_counter()-start)*1000
                    results[name] = result
                    assert (game.key, game.state(), game.features()) == before
                trials.append(results)
            samples.append({"game": record["index"], "stones": count, "history": history,
                            "reference_turn": target, "recorded": selected,
                            "selected_turns": selected_turns, "generation_ms": generation_ms,
                            "ordered_recall": {k: tuple(target) in v for k, v in selected_turns.items()},
                            "position_recall": {k: any(set(t) == set(target) for t in v)
                                                for k, v in selected_turns.items()}, "trials": trials})
        finally:
            game.close()
        if len(samples) >= args.positions:
            break
    if not samples:
        raise ValueError("No eligible recorded loss positions")
    summary = {}
    for name in ("baseline", "diverse"):
        searches = [trial[name] for sample in samples for trial in sample["trials"]]
        summary[name] = {"ordered_recall": statistics.mean(s["ordered_recall"][name] for s in samples),
                         "position_recall": statistics.mean(s["position_recall"][name] for s in samples),
                         "mean_depth": statistics.mean(s["depth"] for s in searches),
                         "mean_nodes": statistics.mean(s["nodes"] for s in searches),
                         "mean_wall_ms": statistics.mean(s["wall_ms"] for s in searches),
                         "zero_depth": sum(s["depth"] == 0 for s in searches)}
    return {"config": vars(args), "summary": summary, "samples": samples,
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "platform": platform.platform(),
            "trace_sha256": trace_digest,
            "recall_scope": "Untimed generated-turn lists; a timed search may expire before admitting or searching these turns. See generation_ms, trial depth, and zero_depth.",
            "table_sha256": table_digest,
            "engine_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256((ROOT/"src/hexo.cpp").read_bytes()).hexdigest(),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "notes": "Alternating equal-time searches with separately measured untimed candidate recall. Recall does not measure searched turns, win rate, or proof of superior moves. Background CPU load affects completed depth."}


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
    parser.add_argument("--trace", help="Run ordered root-pair ablation on recorded arena losses")
    parser.add_argument("--first-game", type=int, default=10)
    parser.add_argument("--seal-library", help="Optional external Seal adapter for reference turns")
    parser.add_argument("--reference-report", help="Reuse exact reference turns from a prior trace benchmark")
    parser.add_argument("--root-seconds", type=int, default=16)
    parser.add_argument("--root-turns", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output")
    args = parser.parse_args()
    if min(args.positions, args.ms, args.reference_ms) < 1 or not 2 <= args.width <= 128 or not 2 <= args.reference_width <= 128:
        parser.error("Counts/budgets must be positive and widths within 2..128")
    if args.repeats < 1:
        parser.error("Repeats must be positive")
    if args.trace and (not max(6, args.width//2) <= args.root_seconds <= 128 or
                       not 2*args.width <= args.root_turns <= 1024):
        parser.error("Require max(6,width/2) <= root-seconds <= 128 and 2*width <= root-turns <= 1024")
    report = pair_admission(args) if args.trace else measure(args)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report.get("summary", {}), indent=2))
    else:
        print(json.dumps(report, indent=2))
