"""Versioned legal starting positions with a disjoint evaluation partition."""
import hashlib
import random

VERSION = "mixed-v1"
AXES = ((1, 0), (0, 1), (1, -1))


def owner(index):
    return 0 if index == 0 else 1 - ((index-1)//2) % 2


def family(opening):
    # Canonicalize the actual colored position, not just an unordered move list.
    # Complete prefixes end between turns; length also identifies player/phase.
    variants = []
    for reflected in (False, True):
        points = [(r, q, owner(i)) if reflected else (q, r, owner(i))
                  for i, (q, r) in enumerate(opening)]
        for _ in range(6):
            variants.append(tuple(sorted(points)))
            points = [(-r, q+r, p) for q, r, p in points]
    key = repr((len(opening), min(variants))).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def _distance(a, b):
    q, r = a[0]-b[0], a[1]-b[1]
    return max(abs(q), abs(r), abs(q+r))


def _wins(board, cell, player):
    for dq, dr in AXES:
        length = 1
        for sign in (-1, 1):
            q, r = cell
            while board.get((q+sign*dq, r+sign*dr)) == player:
                q += sign*dq
                r += sign*dr
                length += 1
        if length >= 6:
            return True
    return False


DISKS = {radius: [(q, r) for q in range(-radius, radius+1)
                 for r in range(-radius, radius+1)
                 if 0 < _distance((q, r), (0, 0)) <= radius]
         for radius in (2, 3, 5, 8)}


def opening_for(seed, evaluation=False):
    """Sample compact, broad, expanding, separated, and broken-line prefixes.

    Family bucket9 is used exclusively for evaluation. Training validation can
    retain the existing family%5==0 rule, yielding buckets0/5 within training.
    Seeds remain deterministic; rejection is by whole symmetry family.
    """
    rng = random.Random(seed)
    while True:
        mode = rng.randrange(5)
        length = rng.choice((9, 11)) if mode == 4 else rng.choice((3, 5, 7, 9))
        moves, board = [(0, 0)], {(0, 0): 0}
        while len(moves) < length:
            index = len(moves)
            player = owner(index)
            if mode == 0:
                cell = rng.choice(DISKS[3])
            elif mode == 1:
                cell = rng.choice(DISKS[8])
            elif mode == 2:
                anchor = moves[-1]
                dq, dr = rng.choice(AXES)
                distance = rng.choice((-8, -6, 6, 8))
                cell = anchor[0]+dq*distance, anchor[1]+dr*distance
            elif mode == 3:
                if index == 1:
                    cell = (8, 0)
                elif index == 2:
                    cell = (16, 0)
                else:
                    anchor = rng.choice(((0, 0), (16, 0)))
                    dq, dr = rng.choice(DISKS[2])
                    cell = anchor[0]+dq, anchor[1]+dr
            elif player == 0:
                # A broken five supplies first-stone wins or mandatory blocks.
                friendly = sum(p == 0 for p in board.values())
                cell = ((0, 1, 2, 4, 5)[friendly], 0)
            else:
                cell = rng.choice(DISKS[5])
                if cell[1] == 0:
                    continue
            if cell in board or not any(_distance(cell, p) <= 8 for p in board):
                continue
            if _wins(board, cell, player):
                continue
            moves.append(cell)
            board[cell] = player
        fid = family(moves)
        if (fid % 10 == 9) != evaluation:
            continue
        # Transform actual coordinates as well as grouping their family IDs.
        if rng.randrange(2):
            moves = [(r, q) for q, r in moves]
        for _ in range(rng.randrange(6)):
            moves = [(-r, q+r) for q, r in moves]
        return moves
