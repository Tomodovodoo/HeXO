"""Thin ctypes interface to the native rules and search engine."""
import ctypes as C
from functools import lru_cache
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


class Turn(C.Structure):
    _fields_ = [(name, C.c_int64) for name in ("q1", "r1", "q2", "r2")] + [
        ("count", C.c_int32), ("score", C.c_int32)]


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
bind("hx_search_root", C.c_int, C.c_void_p, C.c_int, C.c_int, C.c_int,
     C.c_int, C.c_int, C.POINTER(Result))
bind("hx_search_tt", C.c_int, C.c_void_p, C.c_int, C.c_int, C.c_int,
     C.c_int, C.c_int, C.POINTER(Result))
bind("hx_turns", C.c_int, C.c_void_p, C.c_int, C.c_int, C.c_int, C.POINTER(Turn), C.c_int)
bind("hx_features", C.c_int, C.c_void_p, C.POINTER(C.c_int32), C.c_int)
bind("hx_load_table", C.c_int, C.c_void_p, C.POINTER(C.c_int32), C.c_int)
bind("hx_model_load", C.c_void_p, C.c_char_p)
bind("hx_model_free", None, C.c_void_p)
bind("hx_model_error", C.c_char_p)
bind("hx_set_model", C.c_int, C.c_void_p, C.c_void_p)
bind("hx_nnue_centers", C.c_int, C.c_void_p, C.POINTER(C.c_int64), C.POINTER(C.c_int32), C.c_int)
bind("hx_nnue_context", C.c_int, C.c_void_p, C.POINTER(C.c_float))
bind("hx_nnue_policy_features", C.c_int, C.c_void_p, C.c_int64, C.c_int64,
     C.POINTER(C.c_int32), C.POINTER(C.c_float))
bind("hx_nnue_policy_batch", C.c_int, C.c_void_p, C.POINTER(C.c_int64), C.c_int,
     C.POINTER(C.c_int32), C.POINTER(C.c_float))
bind("hx_nnue_inputs", C.c_int, C.c_void_p, C.POINTER(C.c_float), C.c_int)
bind("hx_nnue_rank", C.c_float, C.c_void_p, C.c_int64, C.c_int64)
bind("hx_candidates", C.c_int, C.c_void_p, C.c_int, C.POINTER(Cell), C.c_int)
bind("hx_tactical", C.c_int, C.c_void_p)


class NativeModel:
    def __init__(self, path):
        self.ptr = lib.hx_model_load(os.fsencode(path))
        if not self.ptr:
            raise ValueError(lib.hx_model_error().decode("utf-8"))

    def __del__(self):
        if self.ptr:
            lib.hx_model_free(self.ptr)
            self.ptr = None


@lru_cache(maxsize=8)
def _native_model(path, modified_ns, size):
    # File metadata also prevents reusing a handle after an explicit file update.
    return NativeModel(path)


def native_model(path):
    path = Path(path).resolve()
    info = path.stat()
    return _native_model(str(path), info.st_mtime_ns, info.st_size)


class Game:
    def __init__(self, moves=(), table=None):
        self.ptr = lib.hx_new()
        if not self.ptr:
            raise MemoryError("Unable to allocate native board")
        for q, r in moves:
            self.play(q, r)
        if table is not None:
            self.load_table(table)

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

    def search(self, ms=1000, depth=12, width=16, root_seconds=0, root_turns=0, tt_injection=False):
        result = Result()
        search = lib.hx_search_tt if tt_injection else lib.hx_search_root
        if not search(self.ptr, ms, depth, width, root_seconds, root_turns, C.byref(result)):
            raise ValueError("Invalid search budget or root admission settings")
        moves = [(result.q1, result.r1), (result.q2, result.r2)][:result.count]
        return {"moves": moves, "score": result.score, "depth": result.depth,
                "nodes": result.nodes, "elapsed_ms": result.elapsed_ms}

    def turns(self, width=16, root_seconds=0, root_turns=0):
        """Actual complete ordered turns selected for search, including tactics."""
        n = lib.hx_turns(self.ptr, width, root_seconds, root_turns, None, 0)
        if n < 0:
            raise ValueError("Invalid root admission settings")
        output = (Turn*n)()
        lib.hx_turns(self.ptr, width, root_seconds, root_turns, output, n)
        return [{"moves": [(t.q1, t.r1), (t.q2, t.r2)][:t.count], "score": t.score}
                for t in output]

    def state(self):
        return {"cells": self.cells, "player": self.player,
                "remaining": self.remaining, "winner": self.winner}

    def features(self):
        data = (C.c_int32 * 729)()
        lib.hx_features(self.ptr, data, 729)
        return list(data)

    def load_model(self, path):
        if not lib.hx_set_model(self.ptr, native_model(str(Path(path).resolve())).ptr):
            raise ValueError("Native NNUE model could not be attached")

    def candidates(self, limit=32):
        n = lib.hx_candidates(self.ptr, limit, None, 0)
        cells = (Cell*n)()
        lib.hx_candidates(self.ptr, limit, cells, n)
        return [(c.q, c.r) for c in cells]

    def tactical(self):
        return bool(lib.hx_tactical(self.ptr))

    def nnue_centers(self):
        import numpy as np
        n = lib.hx_nnue_centers(self.ptr, None, None, 0)
        coords, codes = (C.c_int64*(n*2))(), (C.c_int32*(n*3))()
        lib.hx_nnue_centers(self.ptr, coords, codes, n)
        return np.ctypeslib.as_array(codes).reshape(n, 3).copy()

    def nnue_context(self):
        output = (C.c_float*4)()
        lib.hx_nnue_context(self.ptr, output)
        return list(output)

    def nnue_policy_features(self, move):
        codes, pairs = (C.c_int32*3)(), (C.c_float*4)()
        if not lib.hx_nnue_policy_features(self.ptr, *move, codes, pairs):
            raise ValueError("NNUE policy features require a legal placement")
        return list(codes), list(pairs)

    def nnue_policy_batch(self, moves):
        """Exact ordered policy features in contiguous NumPy arrays (N,3)/(N,4)."""
        import numpy as np
        coords = np.asarray(moves)
        if coords.shape == (0,):
            coords = np.empty((0, 2), dtype=np.int64)
        if (coords.ndim != 2 or coords.shape[1] != 2 or coords.dtype.kind not in 'iu'
                or len(coords) > 2**31-1):
            raise ValueError("Coordinates must be an N-by-2 integer array")
        if coords.size and (coords.min() < -10**12 or coords.max() > 10**12):
            raise ValueError("Coordinates must be within +/- 10^12")
        coords = np.ascontiguousarray(coords, dtype=np.int64)
        codes = np.empty((len(coords), 3), dtype=np.int32)
        pairs = np.empty((len(coords), 4), dtype=np.float32)
        if not lib.hx_nnue_policy_batch(self.ptr, coords.ctypes.data_as(C.POINTER(C.c_int64)),
                len(coords), codes.ctypes.data_as(C.POINTER(C.c_int32)),
                pairs.ctypes.data_as(C.POINTER(C.c_float))):
            raise ValueError("NNUE policy features require legal placements")
        return codes, pairs

    def nnue_inputs(self):
        output = (C.c_float*68)()
        if lib.hx_nnue_inputs(self.ptr, output, 68) != 68:
            raise ValueError("NNUE inputs require an attached model")
        return list(output)

    def nnue_rank(self, move):
        return float(lib.hx_nnue_rank(self.ptr, *move))

    def load_table(self, weights):
        weights = list(weights)
        if len(weights) != 729 or any(int(w) != w for w in weights):
            raise ValueError("Pattern table must have 729 integer weights")
        if any(w < -10000 or w > 10000 for w in weights):
            raise ValueError("Pattern weights must be within +/- 10000")
        data = (C.c_int32 * 729)(*(int(w) for w in weights))
        if not lib.hx_load_table(self.ptr, data, len(weights)):
            raise ValueError("Pattern table requires zero empty baseline and weights within +/- 10000")
