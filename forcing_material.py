"""Cheap forcing-material gate for the tactical solver.

Reads the engine's incrementally maintained histogram of six-cell window patterns
(`Game.features()`, base-3 digits: 0 empty, 1 first player, 2 second player). A
window is live for a player when it holds none of the opponent's stones. A forced
win needs threats, and a threat is a live window with at least four stones, so a
turn can only start one from a live window that already holds three stones (one
placement) or two (both placements). The gate asks for a live three: every solver
win in the dense-v1 capped and scanned positions had one.
"""

_DIGITS = [[(index//3**k) % 3 for k in range(6)] for index in range(729)]
_STONES = [(digits.count(1), digits.count(2)) for digits in _DIGITS]
WEIGHTS = {2: 0.25, 3: 2.0, 4: 4.0, 5: 4.0}


def live_windows(game, player=None):
    """Counts of live windows for `player` (default: side to move) by own stone count 0-6."""
    player = game.player if player is None else player
    counts = [0]*7
    for index, number in enumerate(game.features()):
        if number:
            own, other = _STONES[index][player], _STONES[index][1-player]
            if other == 0:
                counts[own] += number
    return counts


def forcing_material(game, player=None):
    """Weighted live twos, threes and fours; orders gated positions for solver budget."""
    counts = live_windows(game, player)
    return sum(weight*counts[k] for k, weight in WEIGHTS.items())


def worth_solving(game, player=None):
    """True when `player` (default: side to move) holds a live window with three or more stones."""
    return sum(live_windows(game, player)[3:]) > 0
