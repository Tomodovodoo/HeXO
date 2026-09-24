"""Thin ctypes interface to the native rules and search engine."""
import ctypes as C
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if os.name == "nt":
    import shutil
    compiler = shutil.which("g++")
    _dll_dir = os.add_dll_directory(str(Path(compiler).parent)) if compiler else None
    library = ROOT / "build" / "libhexo.dll"
    if not library.exists():
        library = ROOT / "build" / "Release" / "hexo.dll"
else:
    library = ROOT / "build" / ("libhexo.dylib" if os.uname().sysname == "Darwin" else "libhexo.so")
lib = C.CDLL(str(library))


class Cell(C.Structure):
    _fields_ = [("q", C.c_int64), ("r", C.c_int64), ("player", C.c_int32)]


class Result(C.Structure):
    _fields_ = [(name, C.c_int64) for name in ("q1", "r1", "q2", "r2")] + [
        ("nodes", C.c_uint64), ("elapsed_ms", C.c_double),
        ("count", C.c_int32), ("score", C.c_int32), ("depth", C.c_int32)]


def bind(name, result, *args):
    fn = getattr(lib, name)
    fn.restype, fn.argtypes = result, list(args)
    return fn


bind("hx_new", C.c_void_p)
bind("hx_free", None, C.c_void_p)
for name in ("player", "remaining", "winner", "size", "undo", "evaluate"):
    bind("hx_" + name, C.c_int, C.c_void_p)
for name in ("play", "legal"):
    bind("hx_" + name, C.c_int, C.c_void_p, C.c_int64, C.c_int64)
bind("hx_hash", C.c_uint64, C.c_void_p)
bind("hx_cell", C.c_int, C.c_void_p, C.c_int, C.POINTER(Cell))
bind("hx_moves", C.c_int, C.c_void_p, C.POINTER(Cell), C.c_int)
bind("hx_search", C.c_int, C.c_void_p, C.c_int, C.c_int, C.c_int, C.POINTER(Result))


class Game:
    def __init__(self, moves=()):
        self.ptr = lib.hx_new()
        if not self.ptr:
            raise MemoryError("Unable to allocate native board")
        for q, r in moves:
            self.play(q, r)

    def close(self):
        if self.ptr:
            lib.hx_free(self.ptr)
            self.ptr = None

    def __del__(self):
        self.close()

    @property
    def player(self):
        return lib.hx_player(self.ptr)

    @property
    def remaining(self):
        return lib.hx_remaining(self.ptr)

    @property
    def winner(self):
        return lib.hx_winner(self.ptr)

    @property
    def key(self):
        return lib.hx_hash(self.ptr)

    @property
    def evaluation(self):
        return lib.hx_evaluate(self.ptr)

    @property
    def cells(self):
        result = []
        for i in range(lib.hx_size(self.ptr)):
            c = Cell()
            lib.hx_cell(self.ptr, i, C.byref(c))
            result.append([c.q, c.r, c.player])
        return result

    def legal(self, q, r):
        return bool(lib.hx_legal(self.ptr, q, r))

    def legal_moves(self):
        n = lib.hx_moves(self.ptr, None, 0)
        cells = (Cell * n)()
        lib.hx_moves(self.ptr, cells, n)
        return [(c.q, c.r) for c in cells]

    def play(self, q, r):
        if type(q) is not int or type(r) is not int or abs(q) > 10**12 or abs(r) > 10**12:
            raise ValueError("Coordinates must be integers within +/- 10^12")
        if not lib.hx_play(self.ptr, q, r):
            raise ValueError(f"Illegal placement: {q}, {r}")

    def undo(self):
        return bool(lib.hx_undo(self.ptr))

    def search(self, ms=1000, depth=12, width=16):
        result = Result()
        if not lib.hx_search(self.ptr, ms, depth, width, C.byref(result)):
            raise ValueError("Search failed; require ms >= 1, depth >= 1, width in 2..128")
        moves = [(result.q1, result.r1), (result.q2, result.r2)][:result.count]
        return {"moves": moves, "score": result.score, "depth": result.depth,
                "nodes": result.nodes, "elapsed_ms": result.elapsed_ms}

    def state(self):
        return {"cells": self.cells, "player": self.player,
                "remaining": self.remaining, "winner": self.winner}
