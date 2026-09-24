"""Pinned human-game import, rules validation, family splits and NNUE replay."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import urllib.request

REPO = "timmyburn/hexo-bootstrap-corpus"
REVISION = "1a82e153ab313f0445a505dfc51289cbd805fe7d"
FILES = ("hexo_human_corpus.jsonl", "dataset_metadata.json", "SCHEMA.md", "README.md")
PINNED_HASHES = dict(zip(FILES, (
    "b2fe61eb360b91d77873a751446d28287955cad49e331fc32c156b4e1316840c",
    "aa2d65362c17ac103e107dbb2bcb2d44a7f81550aece114d62c1100feb438b8b",
    "880600e4c8a4873ddbca197f6b5155e700ec5bbb2d9aceac27ceeaffc8e18f9d",
    "84bf10a3606c4a7258727963a51ad2cc6389ff0ca5eec55a220a92b107e68ed3")))
AXES = ((1, 0), (0, 1), (1, -1))


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def owner(ply):
    return 0 if ply == 0 else 1-((ply-1)//2) % 2


def snapshot_key(moves):
    """Colored board and phase, invariant to translation and all 12 symmetries."""
    return colored_key([(q, r, owner(i)) for i, (q, r) in enumerate(moves)])


def colored_key(cells):
    """Arbitrary snapshot; exclusions conservatively cover every declared phase."""
    variants = []
    for reflect in (False, True):
        points = [(r, q, p) if reflect else (q, r, p) for q, r, p in cells]
        for _ in range(6):
            anchor = min(points)[:2]
            variants.append(sorted((q-anchor[0], r-anchor[1], p) for q, r, p in points))
            points = [(-r, q+r, p) for q, r, p in points]
    return digest((len(cells), min(variants)))


def validate(record):
    """Independent sparse replay; unfinished source games are rejected, never drawn."""
    if record.get("source") != "human" or type(record.get("winner")) is not int or record["winner"] not in (-1, 1):
        raise ValueError("invalid source or winner convention")
    if not re.fullmatch(r"[0-9a-f]{16}", record.get("game_hash", "")):
        raise ValueError("invalid source game_hash")
    board, winner = {}, -1
    for i, move in enumerate(record["moves"]):
        if not isinstance(move, list) or len(move) != 2 or any(type(x) is not int or abs(x) > 10**12 for x in move):
            raise ValueError("invalid axial coordinate")
        q, r = move
        if winner >= 0:
            raise ValueError("moves after first win")
        if (q, r) in board or (not board and (q, r) != (0, 0)):
            raise ValueError("occupied cell or noncentral opening")
        if board and not any(max(abs(q-a), abs(r-b), abs(q-a+r-b)) <= 8 for a, b in board):
            raise ValueError("placement exceeds radius eight")
        p = owner(i)
        board[q, r] = p
        for dq, dr in AXES:
            count = 1
            for sign in (-1, 1):
                k = 1
                while board.get((q+sign*k*dq, r+sign*k*dr)) == p:
                    count += 1
                    k += 1
            if count >= 6:
                winner = p
    if winner < 0:
        raise ValueError("unfinished history, not a draw")
    if winner != (0 if record["winner"] == 1 else 1):
        raise ValueError("winner disagrees with replay")
    return winner


def fetch(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = directory/name
        if not path.exists():
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}"
            path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    return provenance(directory)


def provenance(directory):
    metadata = json.loads((directory/"dataset_metadata.json").read_text())
    hashes = {name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in FILES}
    if hashes != PINNED_HASHES:
        raise ValueError("source snapshot does not match pinned file hashes")
    if hashes[FILES[0]] != metadata["sha256"] or (directory/FILES[0]).stat().st_size != metadata["bytes"]:
        raise ValueError("corpus bytes do not match pinned metadata")
    return {"repository": REPO, "revision": REVISION, "files_sha256": hashes,
            "license": "MIT, declared by pinned dataset card; no separate LICENSE file supplied",
            "metadata": metadata}


def excluded_keys(paths, minimum):
    keys, sources = set(), []
    for path in paths:
        value = json.loads(path.read_text())
        games = value.get("games", [value]) if isinstance(value, dict) else value
        for game in games:
            if "stones" in game:
                cells = [(q, r, {"P1": 0, "P2": 1}[p]) for q, r, p in game["stones"]]
                if len(cells) >= minimum:
                    keys.add(colored_key(cells))
                continue
            moves = game.get("moves") or game["cells"]
            for n in range(minimum, len(moves)+1):
                cells = [(m[0], m[1], {"P1": 0, "P2": 1}.get(m[2], m[2])) if len(m) == 3
                         else (m[0], m[1], owner(i)) for i, m in enumerate(moves[:n])]
                keys.add(colored_key(cells))
        sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "games": len(games)})
    return keys, sources


def prepare(directory, minimum=7, exclusions=()):
    if minimum < 3:
        raise ValueError("family minimum must be at least three stones")
    source = provenance(directory)
    records, rejected, duplicates = [], [], []
    seen_hash, seen_moves = {}, {}
    lines = (directory/FILES[0]).read_text().splitlines()
    if len(lines) != source["metadata"]["n_games"]:
        raise ValueError("corpus count disagrees with metadata")
    for line, text in enumerate(lines, 1):
        try:
            record = json.loads(text)
            validate(record)
            key = digest(record["moves"])
            if record["game_hash"] in seen_hash and seen_hash[record["game_hash"]] != key:
                raise ValueError("conflicting source hash")
            seen_hash[record["game_hash"]] = key
            if key in seen_moves:
                duplicates.append({"line": line, "duplicate_of": seen_moves[key]})
                continue
            seen_moves[key] = line
            records.append({**record, "content_sha256": key, "line": line})
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append({"line": line, "reason": str(exc)})
    excluded, sources = excluded_keys(exclusions, minimum)
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    first, blocked, evaluation = {}, set(), set()
    from curriculum import family
    for i, record in enumerate(records):
        moves = record["moves"]
        # Preserve the existing mixed-v1 evaluation buckets, even for earlier
        # prefixes that are deliberately omitted from the imported examples.
        if any(family(moves[:n]) % 10 == 9 for n in (3, 5, 7, 9, 11) if n <= len(moves)):
            evaluation.add(i)
        for n in range(minimum, len(moves)+1):
            key = snapshot_key(moves[:n])
            if key in excluded:
                blocked.add(i)
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
    groups = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)
    for ids in groups.values():
        key = min(records[i]["content_sha256"] for i in ids)
        fid = int(key[:8], 16)
        split = "validation" if fid % 5 == 0 else "train"
        if any(i in evaluation for i in ids):
            split = "test"
        if any(i in blocked for i in ids):
            split = "excluded"
        for i in ids:
            records[i].update(split=split, family=fid, family_sha256=key)
    report = {"source": source, "input_games": len(lines), "valid_unique_games": len(records),
              "winner_counts": dict(Counter(r["winner"] for r in records)),
              "total_placements": sum(len(r["moves"]) for r in records),
              "rejected": rejected, "duplicates": duplicates, "minimum_ply": minimum,
              "family_components": len(groups), "largest_component": max(map(len, groups.values()), default=0),
              "splits": dict(Counter(r["split"] for r in records)), "exclusion_sources": sources,
              "benchmark_matching_games": len(blocked), "mixed_evaluation_prefix_games": len(evaluation),
              "missing_benchmark": "The separately referenced 20-case tactical ZIP was not supplied; no exclusion claim for it",
              "family_rule": "Connected components sharing any symmetry/translation canonical colored position at or after minimum_ply; all component games share split. Earlier positions omitted. Mixed-v1 held-out prefix buckets force test."}
    return records, report


def examples(record, minimum=7, candidates=32, positions=8):
    """Both players, including conditional seconds; finite samples are not draws."""
    import numpy as np
    from hexo import Game
    from train import nnue_example, pack_nnue
    if candidates < 1 or positions < 4:
        raise ValueError("positive candidates and at least four positions required")
    moves = record["moves"]
    # Round-robin over player/turn-phase strata before filling the finite budget.
    strata = [[i for i in range(minimum, len(moves)) if (owner(i), (i-1) % 2) == (p, phase)]
              for p in (0, 1) for phase in (0, 1)]
    ids = set()
    for index, choices in enumerate(strata):
        budget = positions//4 + int(index < positions % 4)
        if choices:
            ids.update(int(choices[j]) for j in np.linspace(0, len(choices)-1, min(budget, len(choices)), dtype=int))
    # Fill unused slots if a short game has an unusually small phase stratum.
    remaining = sorted(set(range(minimum, len(moves)))-ids)
    if remaining and len(ids) < positions:
        ids.update(remaining[j] for j in np.linspace(0, len(remaining)-1, min(positions-len(ids), len(remaining)), dtype=int))
    game, rows = Game(), []
    try:
        for ply, move in enumerate(moves):
            if ply in ids:
                row = nnue_example(game, tuple(move), {"score": 0, "depth": 0, "elapsed_ms": 0}, candidates)
                row["policy_valid"] = True  # Human imitation, explicitly not a search teacher.
                rows.append(row)
            game.play(*move)
    finally:
        game.close()
    if not rows:
        return None
    data = pack_nnue(rows, record["winner"], record["family"])
    # The current loss gives invalid search fallbacks a small weight. Repeating
    # the outcome avoids inventing a zero-value target; this is not a search.
    data["search"] = data["outcome"].copy()
    return data


def convert(records, output, minimum=7, candidates=32, positions=8, shard_games=64, limit=0):
    import numpy as np
    from train import merge_nnue
    if shard_games < 1:
        raise ValueError("positive shard_games required")
    shards = []
    for split in ("train", "validation", "test"):
        selected = [r for r in records if r["split"] == split]
        if limit:
            selected = selected[:limit]
        for start in range(0, len(selected), shard_games):
            games = selected[start:start+shard_games]
            batches = [examples(r, minimum, candidates, positions) for r in games]
            row_counts = [0 if b is None else len(b["player"]) for b in batches]
            batches = [b for b in batches if b is not None]
            if not batches:
                continue
            path = output/split/f"{start//shard_games:05d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            data = merge_nnue(batches)
            np.savez_compressed(path, **data)
            shards.append({"path": str(path.relative_to(output)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "positions": len(data["player"]), "players": np.bincount(data["player"], minlength=2).tolist(),
                           "games": [g["content_sha256"] for g in games],
                           "game_row_counts": row_counts})
    return shards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/datasets/hexo-human")/REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--exclude-histories", nargs="*", type=Path, default=[])
    parser.add_argument("--minimum-ply", type=int, default=7)
    parser.add_argument("--positions-per-game", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--shard-games", type=int, default=64)
    parser.add_argument("--limit-games-per-split", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory, to prevent stale split/shard contamination")
    if args.download:
        fetch(args.source)
    records, report = prepare(args.source, args.minimum_ply, args.exclude_histories)
    report["importer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True)
    (args.output/"games.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    report["conversion"] = {"policy": "human imitation, both winners and losers; not optimal search labels", "search_valid": False,
                            "inactive_search": "duplicates outcome solely for compatibility with existing fallback loss; no teacher search",
                            "outcome": "terminal result from side-to-move perspective, no draws or fabricated truncated outcomes",
                            "positions_per_game": args.positions_per_game, "candidates": args.candidates,
                            "limit_games_per_split": args.limit_games_per_split}
    if args.convert:
        from hexo import library
        report["native_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
        report["encoding_source_sha256"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                             for name in ("human_corpus.py", "train.py", "hexo.py", "curriculum.py")}
        report["shards"] = convert(records, args.output, args.minimum_ply, args.candidates,
                                   args.positions_per_game, args.shard_games, args.limit_games_per_split)
    (args.output/"manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("input_games", "valid_unique_games", "family_components", "largest_component", "splits")}))


if __name__ == "__main__":
    main()
