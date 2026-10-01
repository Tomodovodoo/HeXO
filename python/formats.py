"""Game notations of other HeXO sites, converted to and from an HTTTX stone history.

Both formats below use the board frame of hexo.did.science, where HTTTX's (q, r) is (q + r, -r) and back.

Rectilinear notation (MineKing9534/HeXO, `board/parse`): a drawing of stones, then optionally a comma and BKE
turns. The drawing reads `x`, `o` or `.` per cell along a row, `-` for two empty cells, a number for that many,
`/` for the next row one step down-right; a leading `c` swaps rows and columns; `(...)` highlights and `[...]`
labels are skipped. BKE turns are `x` or `o` followed by moves; a move is a ring label (A = 1, ..., Z, AA) and an
offset around that ring, written `C2.1` for sector 2, cell 1. An optional prefix sets the ring origin
`@(q, r)` in the drawing's frame, the direction of offset 0 (`>`, `q`, `p`, `<`, `b`, `d`; `d` by default) and
`CW` or `CCW`. Without a drawing the origin holds one stone of the player who did not move first in the turns.

Tyto analysis links (SootyOwl/hexo-strix, `serving/static/shared.js`): `https://hexo.tyto.cc/analysis#c=` and the
stones after the origin as zigzag varints, base64url without padding.
"""
import base64
import re

from hexo import Game

DIRECTIONS = [('>', (1, 0)), ('q', (0, 1)), ('p', (-1, 1)), ('<', (-1, 0)), ('b', (0, -1)), ('d', (1, -1))]
SYMBOLS = [symbol for symbol, _ in DIRECTIONS]
MOVE = r'[A-Z]+(?:[0-5]\.)?\d+'
TURNS = rf'[xo](?:\s+{MOVE})+(?:\s+[xo](?:\s+{MOVE})+)*'
EXTENDED = re.compile(rf'^\s*([bdpq<>])?\s*(CW|CCW)?\s*(?:@\s*\((-?\d+),\s*(-?\d+)\)\s*:?)?\s*({TURNS})\s*$')
TYTO = 'https://hexo.tyto.cc/analysis#c='


def to_site(q, r):
    return q + r, -r


def from_site(x, y):
    return x + y, -y


def distance(a, b):
    dq, dr = a[0] - b[0], a[1] - b[1]
    return max(abs(dq), abs(dr), abs(dq + dr))


def ring_label(ring):
    label = ''
    while ring > 0:
        ring -= 1
        label = chr(ord('A') + ring % 26) + label
        ring //= 26
    return label


def ring_number(label):
    ring = 0
    for character in label:
        ring = ring * 26 + ord(character) - ord('A') + 1
    return ring


def ring_side(baseline, sector):
    return DIRECTIONS[(baseline + sector + 2) % 6][1]


def ring_cell(origin, baseline, clockwise, ring, offset):
    """The cell `offset` steps around ring `ring` of `origin`, from the `baseline` direction."""
    perimeter = 6 * ring
    offset = offset if clockwise else (perimeter - offset) % perimeter
    sector, step = divmod(offset, ring)
    q, r = origin[0] + DIRECTIONS[baseline][1][0] * ring, origin[1] + DIRECTIONS[baseline][1][1] * ring
    for side in range(sector):
        dq, dr = ring_side(baseline, side)
        q, r = q + dq * ring, r + dr * ring
    dq, dr = ring_side(baseline, sector)
    return q + dq * step, r + dr * step


def ring_offset(cell, origin, baseline):
    """(ring, sector, step) of `cell` around `origin`, clockwise from the `baseline` direction."""
    ring = distance(cell, origin)
    for sector in range(6):
        for step in range(ring):
            if ring_cell(origin, baseline, True, ring, sector * ring + step) == tuple(cell):
                return ring, sector, step
    raise ValueError(f'{cell} is not on ring {ring} of {origin}')


def drawing(text):
    """The stones of a Rectilinear drawing as {(q, r): 'x' or 'o'}."""
    columns = text.startswith('c')
    step, down = ((0, 1), (1, 0)) if columns else ((1, 0), (0, 1))
    cells, start, position, index = {}, (0, 0), (0, 0), 1 if columns else 0

    def advance(cell, n=1):
        return cell[0] + step[0] * n, cell[1] + step[1] * n

    while index < len(text):
        character = text[index]
        if character.isdigit():
            digits = re.match(r'\d+', text[index:])[0]
            position, index = advance(position, int(digits)), index + len(digits)
            continue
        if character in '([':
            depth, index = 0, index + 1
            closing = ')' if character == '(' else ']'
            while index < len(text) and (text[index] != closing or depth):
                if text[index] == '\\':
                    index += 1
                elif character == '[' and text[index] == '[':
                    depth += 1
                elif character == '[' and text[index] == ']':
                    depth -= 1
                index += 1
            if index >= len(text):
                raise ValueError('Unterminated highlight or label')
            index += 1
            continue
        index += 1
        if character == ' ':
            continue
        if character in '/\n':
            start = start[0] + down[0], start[1] + down[1]
            position = start
            continue
        if character in 'xXoO':
            cells[position] = character.lower()
        elif character == '-':
            position = advance(position)
        elif character not in '.!':
            raise ValueError(f'Unexpected character {character!r} in Rectilinear notation')
        position = advance(position)
    return cells


def bke_turns(text, implicit):
    """(origin, [(player, [cell, ...]), ...]) of extended BKE turns; ValueError when `text` is not BKE."""
    if text.strip() == '0':
        return None, []
    found = EXTENDED.match(text)
    if not found:
        raise ValueError('Not BKE turns')
    baseline, chirality, q, r, body = found.groups()
    if implicit and q is not None:
        raise ValueError('BKE turns without a drawing take no origin')
    origin = (0, 0) if q is None else (int(q), int(r))
    baseline = SYMBOLS.index(baseline or 'd')
    turns = []
    for word in body.split():
        if word in ('x', 'o'):
            turns.append((word, []))
            continue
        label = re.match(r'[A-Z]+', word)[0]
        ring, rest = ring_number(label), word[len(label):]
        sector, _, step = rest.rpartition('.')
        offset = int(sector) * ring + int(step) if sector else int(step)
        if not 0 <= offset < 6 * ring:
            raise ValueError(f'Offset {offset} is off ring {label}')
        turns[-1][1].append(ring_cell(origin, baseline, chirality != 'CCW', ring, offset))
    return origin, turns


def split_turns(text):
    """The drawing and the BKE part of Rectilinear notation, split at the first comma outside a label."""
    depth, escaped = 0, False
    for index, character in enumerate(text):
        if escaped:
            escaped = False
        elif character == '\\':
            escaped = True
        elif character == '[':
            depth += 1
        elif character == ']' and depth:
            depth -= 1
        elif character == ',' and not depth:
            return text[:index], text[index + 1:].strip()
    return text, None


def rectilinear_loads(text):
    """The HTTTX history of a Rectilinear position or game: the drawn stones as complete turns (the first
    player's stone, the one at the BKE origin when there is one, goes to the origin), then the BKE turns in order.
    The first player becomes cross whatever its letter. Raises ValueError for a drawing that is not a sequence of
    complete turns, turns out of order, or illegal stones."""
    text = text.strip()
    state, rest = split_turns(text)
    origin, turns = None, []
    if rest is None:
        try:
            origin, turns = bke_turns(text, implicit=True)
            first = {'x': 'o', 'o': 'x'}[turns[0][0]] if turns else 'x'
            cells = {(0, 0): first}
        except ValueError:
            cells = drawing(text)
    else:
        cells = drawing(state)
        origin, turns = bke_turns(rest, implicit=False)
    if not cells:
        raise ValueError('Rectilinear notation without stones')
    count = len(cells)
    if count % 2 == 0:
        raise ValueError('The drawn stones do not end a turn')
    later = (count - 1) // 2
    first_count = 1 + 2 * (later // 2)
    owners = {owner: sum(1 for o in cells.values() if o == owner) for owner in 'xo'}
    first = next((owner for owner in 'xo' if owners[owner] == first_count and count - first_count == owners[
        'o' if owner == 'x' else 'x']), None)
    if first is None:
        raise ValueError('The drawn stones are not a sequence of complete turns')
    second = 'o' if first == 'x' else 'x'
    reading = sorted(cells, key=lambda c: (c[1], c[0]))
    opening = origin if origin in cells and cells[origin] == first else next(c for c in reading if cells[c] == first)
    pools = {owner: [c for c in reading if cells[c] == owner and c != opening] for owner in 'xo'}
    stones, mover = [opening], second
    while pools['x'] or pools['o']:
        stones += [pools[mover].pop(0), pools[mover].pop(0)]
        mover = first if mover == second else second
    for player, moves in turns:
        if player != mover:
            raise ValueError(f'BKE turn by {player} where {mover} moves')
        stones += moves
        mover = first if mover == second else second
    if len(set(stones)) != len(stones):
        raise ValueError('Two stones on one cell')
    history = [from_site(x - opening[0], y - opening[1]) for x, y in stones]
    Game(history).close()
    return [list(p) for p in history]


def rectilinear_dumps(history):
    """`history` as Rectilinear notation: the opening stone drawn as `x`, every later turn as BKE around it, with
    the baseline that writes the smallest rings and offsets. Returns the text and, per stone, the (start, end)
    span of its token."""
    if not history:
        raise ValueError('Rectilinear notation needs a stone')
    cells = [to_site(*p) for p in history]
    origin = cells[0]
    turns = [(start, list(range(start, end))) for start, end in
             zip([1, *range(3, len(cells), 2)], [*range(3, len(cells), 2), len(cells)]) if start < len(cells)]
    if not turns:
        return 'x', [(0, 1)]

    def score(baseline):
        return [n for c in cells[1:] for ring, sector, step in [ring_offset(c, origin, baseline)]
                for n in (ring, sector * ring + step)]

    baseline = min(range(6), key=lambda b: (score(b), b))
    text, spans = f'x, {SYMBOLS[baseline]} @(0, 0)', [(0, 1)]
    for index, (start, stones) in enumerate(turns):
        text += ' ' + ('o' if index % 2 == 0 else 'x')
        for stone in stones:
            ring, sector, step = ring_offset(cells[stone], origin, baseline)
            token = ring_label(ring) + (str(sector) if ring == 1 else (f'{sector}.' if sector else '') + str(step))
            text += ' '
            spans.append((len(text), len(text) + len(token)))
            text += token
    return text, spans


def tyto_dumps(history):
    """The Tyto analysis link of `history`; ValueError for an empty board, which the link cannot hold."""
    if not history:
        raise ValueError('A Tyto link needs a stone')
    data = bytearray()
    for q, r in history[1:]:
        for value in to_site(q, r):
            value = value * 2 if value >= 0 else -value * 2 - 1
            while value > 0x7f:
                data.append(value & 0x7f | 0x80)
                value >>= 7
            data.append(value)
    return TYTO + base64.urlsafe_b64encode(bytes(data)).decode().rstrip('=')


def tyto_loads(code):
    """The HTTTX history of the code after `#c=` in a Tyto analysis link; ValueError for a broken one."""
    code = re.sub(r'\s+', '', code.replace('%20', ''))
    if not re.fullmatch(r'[A-Za-z0-9_-]*', code) or len(code) % 4 == 1:
        raise ValueError('Not a Tyto analysis link')
    try:
        data = base64.urlsafe_b64decode(code + '=' * (-len(code) % 4))
    except ValueError as error:
        raise ValueError('Not a Tyto analysis link') from error
    values, value, shift = [], 0, 0
    for byte in data:
        value += (byte & 0x7f) << shift
        shift += 7
        if not byte & 0x80:
            values.append(-(value + 1) // 2 if value % 2 else value // 2)
            value, shift = 0, 0
    if shift or len(values) % 2:
        raise ValueError('Truncated Tyto analysis link')
    history = [(0, 0)] + [from_site(values[i], values[i + 1]) for i in range(0, len(values), 2)]
    Game(history).close()
    return [list(p) for p in history]
