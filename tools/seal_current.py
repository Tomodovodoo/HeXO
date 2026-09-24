"""Build/load the pinned external SealBot best variant; no upstream code is vendored."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess

REVISION = "c94749c21c16c3b072fff6da49762dd5f92f3986"
WEIGHTS_SHA256 = "2819d28d6bce7baaacbd42c00cd8c5a21e171a95327245efe3f29eda79f434bd"
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(source, compiler="g++"):
    source = Path(source).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()
    if git("rev-parse", "HEAD") != REVISION:
        raise ValueError(f"SealBot checkout must be exactly {REVISION}")
    paths = git("ls-files", "best").splitlines()
    if git("status", "--porcelain", "--untracked-files=all", "--", "best"):
        raise ValueError("SealBot best source has local changes or untracked files")
    weights = (source / "best/pattern_data.h").read_bytes()
    if hashlib.sha256(weights.replace(b"\r\n", b"\n")).hexdigest() != WEIGHTS_SHA256:
        raise ValueError("SealBot best weights differ from pinned bytes")
    sources = {p: sha(source / p) for p in paths if p.endswith(".h")}
    adapter = ROOT / "tools/seal_current_adapter.cpp"
    out = ROOT / "build" / ("hexo_seal_current.dll" if os.name == "nt" else "libhexo_seal_current.so")
    out.parent.mkdir(exist_ok=True)
    command = [compiler, "-std=c++20", "-O3", "-shared", "-I", str(source / "best"), str(adapter), "-o", str(out)]
    command += ["-static-libgcc", "-static-libstdc++"] if os.name == "nt" else ["-fPIC"]
    version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    subprocess.run(command, check=True)
    if sources != {p: sha(source / p) for p in sources}:
        raise ValueError("SealBot sources changed during compilation")
    metadata = {"name": "seal-current-best", "repository": "https://github.com/Ramora0/SealBot",
                "revision": REVISION, "variant": "best", "license": "No project license found at pinned revision",
                "source_directory": str(source), "source_sha256": sources,
                "weights_sha256": WEIGHTS_SHA256, "weights_file_sha256": sha(source / "best/pattern_data.h"),
                "weights_hash_encoding": "canonical LF, with compiled file bytes separately hashed",
                "adapter_source_sha256": sha(adapter),
                "binary_sha256": sha(out), "compiler": version, "build_command": command,
                "budget": "milliseconds per complete turn, upstream best-effort deadline; not a hard timeout",
                "randomness": "Upstream random_device initialization, not seeded by arena seed"}
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(out)
    return out


class SealCurrent:
    def __init__(self):
        path = ROOT / "build" / ("hexo_seal_current.dll" if os.name == "nt" else "libhexo_seal_current.so")
        self.metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
        if self.metadata["revision"] != REVISION or self.metadata["weights_sha256"] != WEIGHTS_SHA256 or self.metadata["binary_sha256"] != sha(path):
            raise ValueError("SealBot build manifest mismatch; rebuild the pinned adapter")
        self.metadata["python_adapter_sha256"] = sha(__file__)
        self.lib = C.CDLL(str(path))
        self.lib.seal_current_reset.argtypes = []
        self.lib.seal_current_reset.restype = None
        self.fn = self.lib.seal_current_move
        self.fn.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_int)]
        self.fn.restype = C.c_int

    def reset(self):
        self.lib.seal_current_reset()

    def __call__(self, game, ms):
        cells = game.cells
        # Check before narrowing native int64 coordinates through ctypes c_int.
        if any(abs(q) > 55 or abs(r) > 55 for q, r, _ in cells):
            raise ValueError("SealBot board range exceeded")
        if game.winner >= 0 or not 1 <= ms <= 2**31-1:
            raise ValueError("SealBot requires a live position and positive int32 budget")
        if cells and game.remaining != 2:
            raise ValueError("Pinned SealBot supports complete-turn roots only")
        data = (C.c_int * (3 * len(cells)))(*(v for cell in cells for v in cell))
        out = (C.c_int * 4)()
        count = self.fn(data, len(cells), game.player, game.remaining, ms, out)
        if not 1 <= count <= game.remaining:
            raise ValueError(f"SealBot returned invalid placement count {count}")
        moves = [(out[2*i], out[2*i+1]) for i in range(count)]
        # Validate the entire ordered turn transactionally using exact native rules.
        played = 0
        side = game.player
        try:
            for move in moves:
                game.play(*move)
                played += 1
                if game.winner >= 0:
                    # Upstream always returns a fixed pair for nonempty roots.
                    # The actual turn ends on its first winning placement.
                    return moves[:played]
            if game.winner < 0 and game.player == side:
                raise ValueError("SealBot returned an incomplete turn")
        finally:
            for _ in range(played):
                game.undo()
        return moves


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="External sparse Ramora0/SealBot checkout at pinned revision")
    parser.add_argument("--compiler", default="g++", help="GCC-compatible C++ compiler")
    args = parser.parse_args()
    build(args.source, args.compiler)
