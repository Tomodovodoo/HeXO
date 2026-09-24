"""Slow rules oracle built from board snapshots, without native caches or search."""
from itertools import combinations

AXES = ((1, 0), (0, 1), (1, -1))
LIMIT = 10**12


class Reference:
    def __init__(self):
        self.cells = {}
        self.history = []

    @property
    def player(self):
        return 0 if not self.history else (1 + (len(self.history)-1)//2) % 2

    @property
    def remaining(self):
        return 1 if not self.history else 2-(len(self.history)-1) % 2

    @property
    def winner(self):
        for (q, r), owner in self.cells.items():
            for dq, dr in AXES:
                if all(self.cells.get((q+k*dq, r+k*dr)) == owner for k in range(6)):
                    return owner
        return -1

    def legal(self, q, r):
        if self.winner >= 0 or (q, r) in self.cells or abs(q) > LIMIT or abs(r) > LIMIT:
            return False
        if not self.cells:
            return (q, r) == (0, 0)
        return any((abs(q-a)+abs(r-b)+abs(q-a+r-b))//2 <= 8 for a, b in self.cells)

    def play(self, q, r):
        if not self.legal(q, r):
            raise ValueError((q, r))
        self.cells[q, r] = self.player
        self.history.append((q, r))

    def undo(self):
        if not self.history:
            return False
        del self.cells[self.history.pop()]
        return True

    def windows(self):
        segments = set()
        for q, r in self.cells:
            for dq, dr in AXES:
                for offset in range(6):
                    segments.add(tuple((q+(k-offset)*dq, r+(k-offset)*dr) for k in range(6)))
        return segments

    def features(self):
        counts = [0]*729
        for segment in self.windows():
            code = sum((self.cells.get(cell, -1)+1)*3**k for k, cell in enumerate(segment))
            counts[code] += 1
        return counts

    def completions(self, player, remaining=2):
        result = set()
        for segment in self.windows():
            if any(self.cells.get(cell) == 1-player for cell in segment):
                continue
            empty = frozenset(cell for cell in segment if cell not in self.cells)
            if 1 <= len(empty) <= remaining and all(self.legal(*cell) for cell in empty):
                result.add(empty)
        return result


def has_cover(threats, remaining):
    """Brute-force all subsets of the threat endpoints, independently of GPU branching."""
    if not threats:
        return True
    endpoints = sorted(set().union(*threats))
    return any(all(set(chosen) & threat for threat in threats)
               for count in range(1, remaining+1) for chosen in combinations(endpoints, count))


def interleave(players):
    """Build an ordinary legal move history from two explicit stone lists."""
    reference = Reference()
    used = [0, 0]
    while sum(used) < sum(map(len, players)):
        player = reference.player
        move = players[player][used[player]]
        reference.play(*move)
        used[player] += 1
    return reference.history
