"""Native accelerator with exactly the relational_encoder.Graph contract."""
import ctypes as C
import math
import os
from pathlib import Path
import numpy as np
from hexo import Game
from relational_encoder import Graph, WorkBudgetError

_library = None


def _load():
    global _library
    if _library is None:
        path = Path(__file__).parent/'build'/('libhexo_graph.dll' if os.name == 'nt' else ('libhexo_graph.dylib' if os.uname().sysname == 'Darwin' else 'libhexo_graph.so'))
        if os.name == 'nt' and not path.exists():
            path = path.parent/'Release/hexo_graph.dll'
        lib = C.CDLL(str(path))
        integer = C.POINTER(C.c_int64)
        lib.hgr_build.argtypes = [integer,C.c_int64,integer,C.c_int64,C.c_int64,C.c_int,C.POINTER(C.c_float),C.c_int64,C.c_int64]
        lib.hgr_build.restype = C.c_void_p
        lib.hgr_free.argtypes = [C.c_void_p]
        lib.hgr_array.argtypes = [C.c_void_p,C.c_int,C.POINTER(C.c_int64)]
        lib.hgr_array.restype = integer
        lib.hgr_features.argtypes = [C.c_void_p,C.POINTER(C.c_int64)]
        lib.hgr_features.restype = C.POINTER(C.c_float)
        lib.hgr_error.restype = C.c_char_p
        _library = lib
    return _library


def encode(history, *, global_tokens=16, max_nodes=None, max_edges=None):
    if type(global_tokens) is not int or global_tokens < 1:
        raise ValueError('Need positive global token count')
    game = Game(history)
    try:
        if game.winner >= 0:
            raise ValueError('Terminal positions are handled by exact search, not the network')
        return encode_game(game, global_tokens=global_tokens, max_nodes=max_nodes, max_edges=max_edges)
    finally:
        game.close()


def encode_game(game, *, global_tokens=16, max_nodes=None, max_edges=None):
    """Encode an existing native position without replaying its history."""
    if type(global_tokens) is not int or global_tokens < 1:
        raise ValueError('Need positive global token count')
    if game.winner >= 0:
        raise ValueError('Terminal positions are handled by exact search, not the network')
    if (max_nodes is not None and max_nodes < 0) or (max_edges is not None and max_edges < 0):
        raise WorkBudgetError('Position exceeds negative work budget')
    actions = np.asarray(game.legal_moves(),np.int64).reshape(-1,2)
    stones = np.asarray(game.cells,np.int64).reshape(-1,3)
    player, remaining = game.player, game.remaining
    ns = len(stones)
    phase = np.asarray([remaining == 1,remaining == 2,math.log1p(ns)/8,
                        np.count_nonzero(stones[:,2] == player)/max(1,ns)],np.float32)
    lib = _load()
    ptr = lib.hgr_build(stones.ctypes.data_as(C.POINTER(C.c_int64)),ns,
                        actions.ctypes.data_as(C.POINTER(C.c_int64)),len(actions),global_tokens,player,
                        phase.ctypes.data_as(C.POINTER(C.c_float)),
                        -1 if max_nodes is None else max_nodes,-1 if max_edges is None else max_edges)
    if not ptr:
        raise WorkBudgetError(lib.hgr_error().decode())
    try:
        arrays = []
        length = C.c_int64()
        for i in range(10):
            data = lib.hgr_array(ptr,i,C.byref(length))
            arrays.append(np.ctypeslib.as_array(data,shape=(length.value,)).copy() if length.value else np.empty(0,np.int64))
        features = lib.hgr_features(ptr,C.byref(length))
        features = np.ctypeslib.as_array(features,shape=(length.value,)).copy().reshape(-1,8)
        a,k,o,p,local,stone,read,write,coords,windows = arrays
        return Graph(a.reshape(-1,2),k,o,p,features,local.reshape(-1,5),stone.reshape(-1,5),
                     read.reshape(-1,5),write.reshape(-1,5),coords.reshape(-1,2),windows.reshape(-1,6,2),
                     player,remaining,global_tokens)
    finally:
        lib.hgr_free(ptr)
