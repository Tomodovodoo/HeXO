"""shrimp constants: feature indices, geometry radii, bias-table layout.

Shared by the Python featurizer, the model, the wire ABI, and (via parity
fixtures) the Rust serve-time featurizer.
"""

from __future__ import annotations

import os

# --- engine-contract geometry -------------------------------------------------
# Legality: empty and hex-dist <= LEGAL_RADIUS of any stone; opening move is
# forced to {(0, 0)}. LEGAL_RADIUS matches engine legal.rs. The halo is the
# distance-(LEGAL_RADIUS + 1) shell.
LEGAL_RADIUS = 8
HALO_DIST = LEGAL_RADIUS + 1

# Fixed direction order D: the rotate60 orbit of (1, 0).
# rot60(D[i]) == D[(i + 1) % 6]; reflect(D[i]) == D[5 - i].
DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (0, -1),
    (1, -1),
)

# Packed action id: ((q + 2^15) << 16) | (r + 2^15). Matches engine legal.rs
# pack_coord and hexo_engine.types.pack_coord_id. Integer order of ids equals
# ascending signed (q, r) order.
COORD_OFFSET = 1 << 15

# Missing-neighbour sentinel on the u16 wire (`nbr` ABI buffer). The Python
# featurizer uses -1; the wire/batching layer maps -1 to the padded zero row.
NBR_SENTINEL_U16 = 0xFFFF

# --- node features (F = 15) ---------------------------------------------------
# Indices 0-12 are plane semantics; index 11 is distance-to-nearest-stone.
# Indices 13-14 are the standing-win planes.
F_OWN_STONE = 0
F_OPP_STONE = 1
F_EMPTY = 2
F_LEGAL = 3
F_PHASE_SECOND = 4
F_FIRST_STONE = 5
F_PLAYER_COLOUR = 6
F_OWN_RECENCY = 7
F_OPP_RECENCY = 8
F_OPP_HOT = 9
F_OWN_HOT = 10
F_DIST_TO_STONE = 11
F_OPP_LAST_TURN = 12
F_OPP_WIN_NOW = 13
F_OWN_WIN_NOW = 14
NUM_FEATURES = 15

# Window thresholds over length-WINDOW_LEN single-colour windows.
# hot: count >= HOT_MIN_COUNT (matches the TSS threat definition in
# threats_shared.rs). standing win: count == WIN_NOW_COUNT (single empty cell
# is a win-in-1). HOT_MIN_PLACEMENTS is the earliest placement at which a
# count-4 single-colour window can occur.
HOT_MIN_COUNT = 4
WIN_NOW_COUNT = 5
HOT_MIN_PLACEMENTS = 7
WINDOW_LEN = 6

# dist_to_stone feature scaling: stones -> 0, legal in (0, 1], halo -> 1.125
# (9/8, exactly representable in f16). Ply 0 => 0 everywhere.
DIST_SCALE = float(LEGAL_RADIUS)
HALO_DIST_FEATURE = HALO_DIST / DIST_SCALE  # 1.125

# --- heads / targets ------------------------------------------------------------
VALUE_BINS = 65
MOVES_LEFT_CAP = 209

# --- trunk ----------------------------------------------------------------------
# Trunk width from env SHRIMP_CHANNELS (default 96), read once at import.
# Must be divisible by ATTENTION_HEADS. A checkpoint loads only into a net built
# at the same width.
CHANNELS = int(os.environ.get("SHRIMP_CHANNELS", "").strip() or "96")
NUM_TOKENS = 8
# Head count from env SHRIMP_ATTENTION_HEADS (default 4). The published
# main_7 recipe uses 3 so head_dim lands on 64 (c=192), where attention
# kernels are actually fast; d=32 flex runs at ~1/3 the throughput.
ATTENTION_HEADS = int(os.environ.get("SHRIMP_ATTENTION_HEADS", "").strip() or "4")
if CHANNELS % ATTENTION_HEADS != 0:
    raise ValueError(
        f"SHRIMP_CHANNELS={CHANNELS} not divisible by "
        f"SHRIMP_ATTENTION_HEADS={ATTENTION_HEADS}"
    )
HEAD_DIM = CHANNELS // ATTENTION_HEADS  # 24 at c=96, 32 at c=128, 64 at c=192/h=3
MLP_RATIO = 2
# Trunk block order from env SHRIMP_TRUNK (default "CCCACCCACCA"). 'C' =
# ConvBlock (two hex convs), 'A' = attention block; the layout must end with
# 'A' (ln_final consumes the joint [tokens; cells] sequence the last attention
# block produced). The published main_7 recipe uses "CCACCACCACCACCA" (CC A x5).
# A checkpoint loads only into a net built with the same layout.
TRUNK_LAYOUT = os.environ.get("SHRIMP_TRUNK", "").strip() or "CCCACCCACCA"
if (
    not TRUNK_LAYOUT
    or set(TRUNK_LAYOUT) - {"C", "A"}
    or not TRUNK_LAYOUT.endswith("A")
):
    raise ValueError(
        f"SHRIMP_TRUNK={TRUNK_LAYOUT!r} must be a non-empty string of "
        "'C'/'A' ending with 'A'"
    )

# --- relative-position bias table (per-block learned tables) --------------------
# rows 0-216:  exact axial offsets with hex-dist <= 8 (the 217-offset disk LUT)
# rows 217-224: on-win-axis ring buckets, hex-dist 9-16
# rows 225-232: off-axis ring buckets, hex-dist 9-16
# row  233:    far bucket, hex-dist >= 17
# rows 234/235/236: (query=cell,key=token) / (query=token,key=cell) / (token,token)
BIAS_DISK_RADIUS = LEGAL_RADIUS
BIAS_EXACT_ROWS = 217
BIAS_RING_MIN = 9
BIAS_RING_MAX = 16
BIAS_ON_AXIS_BASE = 217
BIAS_OFF_AXIS_BASE = 225
BIAS_FAR_ROW = 233
BIAS_CELL_TOKEN_ROW = 234
BIAS_TOKEN_CELL_ROW = 235
BIAS_TOKEN_TOKEN_ROW = 236
BIAS_ROWS = 237
