"""Hex-lattice geometry: distance, packing, D6 transforms, bias-row indexing.

Pure functions over axial coordinates (q, r); third cube axis s = -q - r.
All operations use exact integer arithmetic (no floats, no torch).
"""

from __future__ import annotations

from .constants import (
    BIAS_DISK_RADIUS,
    BIAS_EXACT_ROWS,
    BIAS_FAR_ROW,
    BIAS_OFF_AXIS_BASE,
    BIAS_ON_AXIS_BASE,
    BIAS_RING_MAX,
    BIAS_RING_MIN,
    COORD_OFFSET,
)

_COORD_MIN = -(1 << 15)
_COORD_MAX = (1 << 15) - 1


def hex_dist(dq: int, dr: int) -> int:
    """Cube-coordinate hex distance of an axial offset: max(|dq|, |dr|, |dq+dr|)."""

    return max(abs(dq), abs(dr), abs(dq + dr))


def pack_action_id(q: int, r: int) -> int:
    """Pack an axial coordinate into a single integer action id.

    Returns ((q + 2^15) << 16) | (r + 2^15); integer order matches ascending
    signed (q, r). Raises ValueError if q or r is outside i16 range.
    """

    if not (_COORD_MIN <= q <= _COORD_MAX and _COORD_MIN <= r <= _COORD_MAX):
        raise ValueError(f"coordinate out of i16 range: ({q}, {r})")
    return ((q + COORD_OFFSET) << 16) | (r + COORD_OFFSET)


def unpack_action_id(action_id: int) -> tuple[int, int]:
    """Inverse of :func:`pack_action_id`."""

    return ((action_id >> 16) & 0xFFFF) - COORD_OFFSET, (action_id & 0xFFFF) - COORD_OFFSET


# --- D6 about the origin --------------------------------------------------------
# Indices 0-5 = rot60^i; indices 6-11 = reflect-then-rotate:
# sigma_i = rot60^(i-6) ∘ reflect.


def rot60(q: int, r: int) -> tuple[int, int]:
    return -r, q + r


def reflect(q: int, r: int) -> tuple[int, int]:
    return q, -q - r


def apply_d6(index: int, q: int, r: int) -> tuple[int, int]:
    """Apply D6 transform `index` (0-11) to an axial coordinate."""

    if not 0 <= index < 12:
        raise ValueError(f"D6 index out of range: {index}")
    if index >= 6:
        q, r = reflect(q, r)
        index -= 6
    for _ in range(index):
        q, r = rot60(q, r)
    return q, r


def d6_inverse(index: int) -> int:
    """Index of the inverse transform. Rotations invert to 6-i mod 6;
    every reflect-then-rotate element is an involution."""

    if not 0 <= index < 12:
        raise ValueError(f"D6 index out of range: {index}")
    if index >= 6:
        return index
    return (6 - index) % 6


def on_win_axis(dq: int, dr: int) -> bool:
    """Whether an offset is collinear with one of the three win axes
    Q=(1,0), R=(0,1), QR=(1,-1): dq == 0 or dr == 0 or dq + dr == 0.

    Invariant under D6 (rotations 3-cycle the axes, reflections transpose them)."""

    return dq == 0 or dr == 0 or dq + dr == 0


# --- relative-position bias row index --------------------------------------------


def disk_offsets(radius: int) -> list[tuple[int, int]]:
    """All axial offsets with hex_dist <= radius, in ascending (dq, dr) order
    (which matches ascending packed-id order)."""

    offsets = [
        (dq, dr)
        for dq in range(-radius, radius + 1)
        for dr in range(-radius, radius + 1)
        if hex_dist(dq, dr) <= radius
    ]
    offsets.sort()
    return offsets


_EXACT_LUT: dict[tuple[int, int], int] = {
    offset: row for row, offset in enumerate(disk_offsets(BIAS_DISK_RADIUS))
}
assert len(_EXACT_LUT) == BIAS_EXACT_ROWS


def rel_bias_index(dq: int, dr: int) -> int:
    """Bias-table row for a cell-cell query/key offset.

    For d = hex_dist(dq, dr) <= BIAS_DISK_RADIUS, returns the exact LUT row.
    For BIAS_RING_MIN <= d <= BIAS_RING_MAX, returns base + (d - BIAS_RING_MIN),
    where base is BIAS_ON_AXIS_BASE if on_win_axis else BIAS_OFF_AXIS_BASE.
    Beyond that, returns BIAS_FAR_ROW."""

    d = hex_dist(dq, dr)
    if d <= BIAS_DISK_RADIUS:
        return _EXACT_LUT[(dq, dr)]
    if d <= BIAS_RING_MAX:
        base = BIAS_ON_AXIS_BASE if on_win_axis(dq, dr) else BIAS_OFF_AXIS_BASE
        return base + (d - BIAS_RING_MIN)
    return BIAS_FAR_ROW
