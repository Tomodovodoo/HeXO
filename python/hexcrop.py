"""Bucketed axial board crops for the dense hex ResNet. NumPy only, no torch.

A position is transformed by one of the 12 hex symmetries and laid out as
planes[c, y, x] with x = q'-qmin+ox and y = r'-rmin+oy. Every symmetry maps the
axial neighbour set onto the index offsets (dx, dy) in {(+-1,0), (0,+-1),
(1,-1), (-1,1)} and the three line axes onto the index directions (1,0),
(0,1), (1,-1) (permuted, possibly negated), so the network's hex-masked 3x3
convolutions and fixed line windows see the same geometry under every
symmetry. Legal actions are always reported in native legal order.
"""
import ctypes as C
from dataclasses import dataclass
import numpy as np
import hexo
from hexo import Game
from legacy.relational_encoder import transform

BUCKETS = (24, 32, 40, 48, 64, 96, 128, 192, 256)
# own/opponent: stones of the side to move / the other side; legal: in-crop legal cells;
# crop: 1 on the bbox of stones and legal cells (far mode: stones plus halo), 0 on bucket padding;
# remaining_1/remaining_2: constant planes for 1 or 2 placements left this turn; turn_stone:
# the stone placed earlier this turn (remaining 1 only); opponent_turn: the opponent's previous turn.
PLANES = ['own', 'opponent', 'legal', 'crop', 'remaining_1', 'remaining_2', 'turn_stone', 'opponent_turn']
HALO = 4
RADIUS, LIMIT = 8, 10**12
# (q, r) -> (q, r) @ SYMMETRIES[k] reproduces relational_encoder.transform(., k).
SYMMETRIES = np.array([[transform((1, 0), k), transform((0, 1), k)] for k in range(12)], dtype=np.int64)
INVERSES = np.rint(np.linalg.inv(SYMMETRIES)).astype(np.int64)
# Transformed x and y are +-q, +-r or +-(q+r): index 0, 1 or 2 into the ranges of (q, r, q+r).
_AXIS_OF = np.array([[2 if m[0, j] and m[1, j] else int(m[1, j] != 0) for j in (0, 1)] for m in SYMMETRIES])
_POSITIVE = np.array([[(m[0, j] or m[1, j]) > 0 for j in (0, 1)] for m in SYMMETRIES])  # +q, +r or +(q+r)
_CELL = np.dtype([('q', '<i8'), ('r', '<i8'), ('player', '<i4')], align=True)
assert _CELL.itemsize == C.sizeof(hexo.Cell)


class SpanError(ValueError):
    """The stones plus halo span more cells than the largest bucket."""


@dataclass
class Sample:
    """One encoded position. cells[i] is the flat index y*size+x of actions[i], or -1 when far."""
    planes: np.ndarray      # uint8 [len(PLANES), size, size]
    size: int
    cells: np.ndarray       # int64 [N]
    actions: np.ndarray     # int64 [N, 2], untransformed, native legal order
    far: int
    player: int
    remaining: int
    symmetry: int
    offset: tuple           # (qmin, rmin, ox, oy)

    def point(self, index):
        """Original (q, r) of flat crop index `index`."""
        qmin, rmin, ox, oy = self.offset
        y, x = divmod(int(index), self.size)
        q, r = np.array([x-ox+qmin, y-oy+rmin]) @ INVERSES[self.symmetry]
        return int(q), int(r)


def native_legal(game):
    """Complete legal list from the engine as int64 [N, 2] in native order."""
    buffer = np.empty(4096, _CELL)
    n = hexo.lib.hx_moves(game.ptr, buffer.ctypes.data_as(C.POINTER(hexo.Cell)), len(buffer))
    if n > len(buffer):
        buffer = np.empty(n, _CELL)
        hexo.lib.hx_moves(game.ptr, buffer.ctypes.data_as(C.POINTER(hexo.Cell)), n)
    return np.stack((buffer['q'][:n], buffer['r'][:n]), 1)


class Position:
    """encode_game's and legal_array's view of a non-terminal native history int64 [n, 2] without replaying it:
    side to move and placements left follow from n. A Game is replayed only if legal_array needs the engine
    fallback (`ptr`)."""
    winner = -1

    def __init__(self, history):
        n = len(history)
        self.history, self.player, self.remaining = history, ((n+1)//2) % 2, 2 if n % 2 else 1

    @property
    def ptr(self):
        self.game = Game(self.history.tolist())
        return self.game.ptr


def _shift_or(grid, dq, dr):
    """grid[x] |= grid[x-(dq, dr)] in place."""
    h, w = grid.shape
    grid[max(dq, 0):h+min(dq, 0), max(dr, 0):w+min(dr, 0)] |= grid[max(-dq, 0):h-max(dq, 0), max(-dr, 0):w-max(dr, 0)]


def legal_array(game, moves):
    """Native legal order (lexicographic (q, r)): empty cells within hex distance 8 of a stone.

    Computed on a dense grid; spans over 4M cells or coordinates near the engine
    limit fall back to the engine's own generator.
    """
    if not len(moves):
        return np.zeros((1, 2), np.int64)
    low = moves.min(0)-RADIUS
    extent = moves.max(0)+RADIUS-low+1
    if extent[0]*extent[1] > 1 << 22 or np.abs(moves).max() > LIMIT-RADIUS:
        return native_legal(game)
    grid = np.zeros(extent, bool)
    p = moves-low
    grid[p[:, 0], p[:, 1]-RADIUS] = True
    # The radius-8 hex ball is the zonotope a(1,0)+b(0,1)+c(-1,1)-(0,8), a,b,c in 0..8.
    for dq, dr in ((1, 0), (0, 1), (-1, 1)):
        covered = 1
        while covered <= RADIUS:
            k = min(covered, RADIUS+1-covered)
            _shift_or(grid, k*dq, k*dr)
            covered += k
    grid[p[:, 0], p[:, 1]] = False
    q, r = np.nonzero(grid)
    return np.stack((q+low[0], r+low[1]), 1)


def _bounds(*arrays):
    """(low, high), each int64 [3], of q, r and q+r over the [n, 2] point arrays."""
    qrs = np.empty((3, sum(map(len, arrays))), np.int64)
    start = 0
    for a in arrays:
        qrs[:2, start:start+len(a)] = a.T
        start += len(a)
    np.add(qrs[0], qrs[1], out=qrs[2])
    return qrs.min(1), qrs.max(1)


def _sides(points, halo=0, bounds=None):
    """Required square side per symmetry, [12]; `bounds` is _bounds(points) when already known."""
    low, high = _bounds(points) if bounds is None else bounds
    return (high-low)[_AXIS_OF].max(1)+1+2*halo


def _bucket(side):
    return next((b for b in BUCKETS if b >= side), None)


def _choose(sides, symmetry, rng):
    if symmetry is not None:
        return symmetry
    best = int(np.argmin(sides))
    bucket = _bucket(sides[best])
    if rng is None or bucket is None:   # None: nothing fits, the caller switches to far mode or raises
        return best
    return int(rng.choice(np.flatnonzero(sides <= bucket)))


def encode(history, *, symmetry=None, rng=None):
    """Replay `history` and encode it; see encode_game."""
    game = Game(history)
    try:
        return encode_game(game, history, symmetry=symmetry, rng=rng)
    finally:
        game.close()


def encode_leaf(native, tree, request, history):
    """Encode an existing native search leaf; old libraries retain the non-replay Python path."""
    if not hasattr(native, 'hxg_encode'):
        return encode_game(Position(np.asarray(history, np.int64)), history)
    info = np.empty(9, np.int64)
    size = native.hxg_encode(tree, request, None, 0, None, info.ctypes.data)
    if size == -2:
        raise SpanError('Stones plus halo exceed the largest bucket')
    if size == 0:
        raise ValueError(native.hxg_error().decode())
    actions = np.empty((int(info[0]), 2), np.int64)
    native.hxg_legal(tree, request, actions.ctypes.data)
    planes = np.empty((len(PLANES), size, size), np.uint8)
    cells = np.empty(len(actions), np.int64)
    if native.hxg_encode(tree, request, planes.ctypes.data, planes.size, cells.ctypes.data, info.ctypes.data) != size:
        raise ValueError(native.hxg_error().decode())
    return Sample(planes, size, cells, actions, int(info[8]), int(info[1]), int(info[2]), int(info[3]),
                  tuple(map(int, info[4:8])))


def encode_leaves(native, leaves, *, allow_span=False):
    """Encode a pending batch in two native calls; returned action arrays own their cache storage."""
    if len(leaves) < 8 or not hasattr(native, 'hxg_encode_many'):
        samples = []
        for tree, request, history in leaves:
            try:
                samples.append(encode_leaf(native, tree, request, history))
            except SpanError:
                if not allow_span:
                    raise
                samples.append(None)
        return samples
    encode = native.hxg_encode_many
    encode.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_int64,
                       C.c_void_p, C.c_void_p, C.c_int64]
    encode.restype = C.c_int
    trees = np.asarray([tree for tree, _, _ in leaves], np.uintp)
    requests = np.asarray([request for _, request, _ in leaves], np.int32)
    info = np.zeros((len(leaves), 12), np.int64)
    if not encode(trees.ctypes.data, requests.ctypes.data, len(leaves), info.ctypes.data,
                  None, 0, None, None, 0):
        raise ValueError(native.hxg_error().decode())
    if not allow_span and np.any(info[:, 0] == -2):
        raise SpanError('Stones plus halo exceed the largest bucket')
    sides = np.maximum(info[:, 0], 0)
    planes = np.empty(int((len(PLANES)*sides*sides).sum()), np.uint8)
    count = int(info[sides > 0, 1].sum())
    cells, actions = np.empty(count, np.int64), np.empty((count, 2), np.int64)
    if not encode(trees.ctypes.data, requests.ctypes.data, len(leaves), info.ctypes.data,
                  planes.ctypes.data, len(planes), cells.ctypes.data, actions.ctypes.data, count):
        raise ValueError(native.hxg_error().decode())
    samples = []
    for row in info:
        size, n, player, remaining, symmetry, qmin, rmin, ox, oy, far, po, lo = map(int, row)
        samples.append(None if size == -2 else Sample(
            planes[po:po+len(PLANES)*size*size].reshape(len(PLANES), size, size), size,
            cells[lo:lo+n], actions[lo:lo+n].copy(), far, player, remaining, symmetry,
            (qmin, rmin, ox, oy)))
    return samples


def encode_game(game, history, *, symmetry=None, rng=None, actions=None):
    """Encode the position of `game`, whose placements are `history`; `actions` [N, 2], when given, must be its
    legal moves in native order (legal_array is skipped).

    symmetry=None picks the symmetry with the smallest crop (lowest id on ties);
    with `rng` it picks uniformly among symmetries that fit the same bucket.
    Terminal positions raise ValueError, positions too wide even for far mode SpanError.
    """
    if game.winner >= 0:
        raise ValueError('Terminal positions are not encoded')
    if symmetry is not None and not 0 <= symmetry < 12:
        raise ValueError('Symmetry must be in 0..11')
    player, remaining = game.player, game.remaining
    moves = np.asarray(history, dtype=np.int64).reshape(-1, 2)
    actions = legal_array(game, moves) if actions is None else actions
    n = len(moves)
    bounds = _bounds(moves, actions)
    sides = _sides(None, bounds=bounds)
    k = _choose(sides, symmetry, rng)
    halo = 0
    if sides[k] > BUCKETS[-1]:
        # Far mode: crop the stones plus a halo; legal cells outside are pooled.
        halo, bounds = HALO, _bounds(moves)
        sides = _sides(None, HALO, bounds)
        k = _choose(sides, symmetry, rng)
        if sides[k] > BUCKETS[-1]:
            raise SpanError(f'Stones span {sides[k]} cells with halo; the largest bucket is {BUCKETS[-1]}')
    size = _bucket(sides[k])
    # Transformed x and y are +-q, +-r or +-(q+r), so their bounds follow from `bounds`.
    axis, positive = _AXIS_OF[k], _POSITIVE[k]
    low = np.where(positive, bounds[0][axis], -bounds[1][axis])-halo
    extent = np.where(positive, bounds[1][axis], -bounds[0][axis])+halo-low+1
    ox, oy = (size-extent)//2
    shift = np.array([ox, oy])-low
    xy = moves @ SYMMETRIES[k]+shift
    a = actions @ SYMMETRIES[k]+shift
    cells = a[:, 1]*size+a[:, 0]
    if halo:
        inside = ((a >= (ox, oy)) & (a < (ox+extent[0], oy+extent[1]))).all(1)
        cells[~inside] = -1
    owner = ((np.arange(n)+1)//2) % 2
    planes = np.zeros((len(PLANES), size, size), np.uint8)
    own = owner == player
    planes[0, xy[own, 1], xy[own, 0]] = 1
    planes[1, xy[~own, 1], xy[~own, 0]] = 1
    planes[2].reshape(-1)[cells[cells >= 0]] = 1
    planes[3, oy:oy+extent[1], ox:ox+extent[0]] = 1
    planes[4 if remaining == 1 else 5] = 1
    # Ply 0 is a one-stone turn: no earlier stone this turn and no previous turn.
    start = n if remaining == 2 or n == 0 else n-1
    if start < n:
        planes[6, xy[n-1, 1], xy[n-1, 0]] = 1
    previous = xy[max(0, start-2):start]
    planes[7, previous[:, 1], previous[:, 0]] = 1
    return Sample(planes, size, cells, actions, int((cells < 0).sum()), player, remaining, k,
                  (int(low[0]), int(low[1]), int(ox), int(oy)))


def group_by_size(samples):
    """{bucket size: [sample indices]} in first-seen bucket order, indices ascending."""
    groups = {}
    for i, sample in enumerate(samples):
        groups.setdefault(sample.size, []).append(i)
    return groups


def batch(samples):
    """Stack samples of one bucket size; cells are padded with -1 beyond counts[b]."""
    size = samples[0].size
    if any(s.size != size for s in samples):
        raise ValueError('A batch must hold one bucket size')
    counts = np.array([len(s.cells) for s in samples], np.int64)
    planes = np.stack([s.planes for s in samples])
    cells = np.full((len(samples), counts.max(initial=0)), -1, np.int64)
    cells[np.arange(cells.shape[1]) < counts[:, None]] = np.concatenate([s.cells for s in samples])
    return dict(planes=planes, cells=cells, counts=counts, size=size,
                far=np.array([s.far for s in samples], np.int64),
                player=np.array([s.player for s in samples], np.int64),
                remaining=np.array([s.remaining for s in samples], np.int64))
