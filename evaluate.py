"""Paired NNUE checkpoint evaluation against a frozen, zero-Elo reference."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loaded_modules():
    """Resolved file-backed modules, including transitive native runtime libraries."""
    if os.name == "nt":
        import ctypes as C
        kernel = C.WinDLL("kernel32", use_last_error=True)
        psapi = C.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = C.c_void_p
        psapi.EnumProcessModulesEx.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.c_ulong, C.POINTER(C.c_ulong), C.c_ulong]
        kernel.GetModuleFileNameW.argtypes = [C.c_void_p, C.c_wchar_p, C.c_ulong]
        process = kernel.GetCurrentProcess()
        capacity = 256
        while True:
            modules = (C.c_void_p*capacity)()
            needed = C.c_ulong()
            if not psapi.EnumProcessModulesEx(process, modules, C.sizeof(modules), C.byref(needed), 3):
                raise C.WinError(C.get_last_error())
            if needed.value <= C.sizeof(modules):
                break
            capacity = needed.value//C.sizeof(C.c_void_p)
        paths = []
        for module in modules[:needed.value//C.sizeof(C.c_void_p)]:
            buffer = C.create_unicode_buffer(32768)
            if not kernel.GetModuleFileNameW(module, buffer, len(buffer)):
                raise C.WinError(C.get_last_error())
            paths.append(Path(buffer.value).resolve())
    elif sys.platform.startswith("linux"):
        paths = [Path(line.split(maxsplit=5)[5].strip()).resolve()
                 for line in Path("/proc/self/maps").read_text().splitlines()
                 if len(line.split(maxsplit=5)) == 6 and line.split(maxsplit=5)[5].startswith("/")]
    else:
        raise RuntimeError("Loaded runtime provenance currently supports Windows and Linux")
    return sorted(set(paths))


def runtime_identity():
    from hexo import library
    return {str(path): sha(path) for path in loaded_modules() if path != library.resolve()}


def verify_runtime(expected, loaded=None):
    actual = {os.path.normcase(str(path.resolve())) for path in (loaded_modules() if loaded is None else loaded)}
    for path, digest in expected.items():
        resolved = Path(path).resolve()
        if os.path.normcase(str(resolved)) not in actual or not resolved.is_file() or sha(resolved) != digest:
            raise ValueError(f"Native runtime dependency missing, relocated or changed: {path}")


def initialize_worker(runtime):
    import train  # Load the same native library and feature dependencies before checking.
    verify_runtime(runtime)


def publish_failure(output, error):
    from train import write_json
    provenance = json.loads((output/"provenance.json").read_text(encoding="utf-8"))
    status = json.loads((output/"status.json").read_text(encoding="utf-8")) if (output/"status.json").exists() else {}
    status.update(stage="failed", error=repr(error),
                  interrupted=isinstance(error, KeyboardInterrupt),
                  candidate_sha256=provenance["model_input_sha256"]["candidate"],
                  reference_sha256=provenance["model_input_sha256"]["reference"])
    write_json(output/"status.json", status)


def verify_trace(record):
    """Replay every reported turn in a fresh native rules state, without a model."""
    from hexo import Game
    game = Game(record["opening"])
    try:
        cells = [list(c) for c in record["cells"]]
        if cells[:len(game.cells)] != [list(c) for c in game.cells]:
            raise ValueError("Opening or opening owners disagree with trace")
        for turn in record["search_trace"]:
            if (turn["ply"], turn["player"], turn["remaining"]) != (len(game.cells), game.player, game.remaining):
                raise ValueError("Search phase disagrees with replay")
            side = game.player
            moves = turn["result"]["moves"]
            if not moves:
                raise ValueError("Empty search turn")
            for q, r in moves:
                if game.winner >= 0 or game.player != side:
                    raise ValueError("Search continued beyond its turn or first-stone win")
                if len(game.cells) >= len(cells) or cells[len(game.cells)] != [q, r, side]:
                    raise ValueError("Search moves disagree with ordered game history")
                game.play(q, r)
            if game.winner < 0 and game.player == side:
                raise ValueError("Search did not complete its turn")
        if [list(c) for c in game.cells] != cells or game.winner != record["winner"]:
            raise ValueError("Final board or winner disagrees with replay")
        if (game.winner >= 0) != (record["reason"] == "six-in-a-row"):
            raise ValueError("Truncation or terminal reason disagrees with replay")
    finally:
        game.close()


def play(task):
    """Current native-NNUE match adapter; reporting and statistics are actor-independent."""
    from train import play_game
    record, _ = play_game(task)
    verify_trace(record)
    record.update(index=task["index"], pair=task["index"]//2, replay_verified=True)
    return record


def freeze(args):
    from hexo import ROOT, library
    from train import write_json
    runtime = runtime_identity()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Evaluation output must be a new directory; existing evidence is never overwritten")
    if args.games < 2 or args.games % 2 or min(args.ms, args.workers) < 1 or args.max_stones < 5 or not 2 <= args.width <= 128:
        raise ValueError("Use an even game count, positive time/workers, stone cap >= 5 and native width in 2..128")
    # Read inputs before creating output, then use only immutable copied bytes.
    models = {name: path.resolve().read_bytes() for name, path in (("candidate", args.candidate), ("reference", args.reference))}
    sources = ["evaluate.py", "train.py", "hexo.py", "curriculum.py", "src/hexo.cpp", "src/hexo.hpp", "src/nnue.hpp", "CMakeLists.txt"]
    output.mkdir(parents=True)
    for name in sources:
        destination = output/"source"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
    target_library = output/"source"/library.relative_to(ROOT)
    target_library.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(library, target_library)
    (output/"models").mkdir()
    for name, content in models.items():
        (output/"models"/f"{name}.nnue").write_bytes(content)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    identity = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "execute_snapshot"}
    write_json(output/"provenance.json", {"schema": "hexo-paired-nnue-v1", "revision": revision, "dirty": dirty,
               "files_sha256": identity, "config": config, "python": sys.version, "platform": platform.platform(),
               "runtime_files_sha256": runtime,
               "runtime_scope": "Resolved file-backed modules loaded after native/Python imports; checked in parent and workers. Host files are verified, not copied; this is not whole-OS hermetic isolation.",
               "reference_elo": 0, "promotion": False,
               "rating_scope": "relative to this exact reference checkpoint, rules, search budget and opening distribution",
               "timing": "per-turn native search wall time; model attachment excluded; full search results retained",
               "model_input_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in models.items()}})
    return output


def execute(output):
    from hexo import Game
    from train import paired_metrics, task_opening, write_json
    provenance = json.loads((output/"provenance.json").read_text(encoding="utf-8"))
    config = provenance["config"]
    def check_identity():
        if any(sha(output/name) != value for name, value in provenance["files_sha256"].items()):
            raise ValueError("Frozen evaluation source, model or DLL changed")
    check_identity()
    verify_runtime(provenance["runtime_files_sha256"])
    model_paths = {name: str(output/"models"/f"{name}.nnue") for name in ("candidate", "reference")}
    game = Game()
    try:
        for path in model_paths.values():
            game.load_model(path)  # Reject malformed exports before scheduling any matches.
    finally:
        game.close()
    tasks = []
    for index in range(config["games"]):
        seed = config["seed"]+index//2
        color = index % 2
        tables = [model_paths["reference"]]*2
        tables[color] = model_paths["candidate"]
        tasks.append({"index": index, "tables": tables, "seed": seed, "evaluation": True,
                      "model_kind": "nnue", "ms": config["ms"], "width": config["width"],
                      "max_stones": config["max_stones"], "challenger_color": color, "record_searches": True,
                      **task_opening(seed, True, config["max_stones"], config["curriculum"])})
    write_json(output/"openings.json", [{k: t[k] for k in ("index", "seed", "opening", "family", "challenger_color")} for t in tasks])
    openings_sha256 = sha(output/"openings.json")
    records = []
    started = time.perf_counter()
    def publish(finished=False):
        metrics = paired_metrics(records, config["games"])
        # Ratings are withheld for pending or capped games. Open confidence endpoints
        # use null for negative/positive infinity, never artificial finite certainty.
        metrics["anchor_elo"] = 0
        metrics["anchored_elo"] = metrics["elo_delta"]
        metrics["anchored_elo_95pct"] = metrics["elo_delta_95pct_open"] if metrics["rated"] else None
        status = {"stage": "finished" if finished else "native", "total": config["games"],
                  "completed": len(records), "elapsed_seconds": time.perf_counter()-started,
                  "reference_sha256": provenance["model_input_sha256"]["reference"],
                  "candidate_sha256": provenance["model_input_sha256"]["candidate"],
                  **({"results": {"native": metrics}} if finished else metrics)}
        report = {"provenance": provenance, "openings_sha256": openings_sha256, "status": status, "metrics": metrics,
                  "games": sorted(records, key=lambda r: r["index"])}
        write_json(output/"report.json", report)
        write_json(output/"status.json", status)
        return metrics
    publish()
    with ProcessPoolExecutor(max_workers=config["workers"], initializer=initialize_worker,
                             initargs=(provenance["runtime_files_sha256"],)) as pool:
        for future in as_completed([pool.submit(play, task) for task in tasks]):
            records.append(future.result())
            metrics = publish()
            print(f"{len(records)}/{config['games']}: {metrics['wins']} W / {metrics['losses']} L / {metrics['incomplete']} capped", flush=True)
    check_identity()
    verify_runtime(provenance["runtime_files_sha256"])
    publish(finished=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--games", type=int, default=160)
    parser.add_argument("--ms", type=int, default=100)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--max-stones", type=int, default=800)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--curriculum", choices=("mixed-v1", "legacy"), default="mixed-v1")
    parser.add_argument("--backend", choices=("native-nnue",), default="native-nnue")
    parser.add_argument("--execute-snapshot", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute_snapshot:
        output = args.execute_snapshot.resolve()
        try:
            execute(output)
        except (Exception, KeyboardInterrupt) as error:
            publish_failure(output, error)
            raise
    else:
        if not all((args.candidate, args.reference, args.output)):
            parser.error("--candidate, --reference and --output are required")
        output = freeze(args)
        subprocess.run([sys.executable, str(output/"source/evaluate.py"), "--execute-snapshot", str(output)], check=True)


if __name__ == "__main__":
    main()
