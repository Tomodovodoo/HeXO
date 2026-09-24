"""Replay pinned public Strix puzzle snapshots through the independent process."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strix_reference import REVISION, StrixReference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="pinned external hexo-strix checkout")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=100000)
    parser.add_argument("--seconds", type=float, default=1)
    parser.add_argument("--wide", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        parser.error("source checkout does not match pinned solver")
    records = []
    with StrixReference() as reference:
        executable_hash = hashlib.sha256(Path(reference.executable).read_bytes()).hexdigest()
        paths = subprocess.check_output(["git", "-C", str(args.source), "ls-tree", "-r", "--name-only",
            REVISION, "scripts/fixtures/forcing_puzzles"], text=True).splitlines()
        for relative in sorted(name for name in paths if name.endswith(".json")):
            path = Path(relative)
            data = subprocess.check_output(["git", "-C", str(args.source), "show", f"{REVISION}:{relative}"])
            puzzle = json.loads(data)
            if "stones" not in puzzle:
                result = dict(status="SKIPPED_LINE", reason="game log has no declared query snapshot/attacker")
                records.append(dict(file=path.name, sha256=hashlib.sha256(data).hexdigest(), result=result))
                print(path.name, result["status"], flush=True)
                continue
            try:
                result = reference.solve(puzzle["stones"], puzzle["attacker"], puzzle["placements_remaining"],
                    depth=args.depth, nodes=args.nodes, timeout_s=args.seconds, wide=args.wide)
            except (KeyError, ValueError) as error:
                result = dict(status="UNKNOWN", reason=f"unsupported fixture: {error}")
            records.append(dict(file=path.name, sha256=hashlib.sha256(data).hexdigest(), result=result))
            print(path.name, result["status"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(revision=REVISION, executable_sha256=executable_hash,
        corpus="public Strix fixtures, not the quoted 20-case audit",
        records=records), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
