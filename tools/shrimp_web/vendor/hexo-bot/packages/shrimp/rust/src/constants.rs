//! shrimp constants. Corresponding Python values are in python/shrimp/constants.py.

pub const LEGAL_RADIUS: i32 = 8;
pub const HALO_DIST: i32 = 9;

/// Fixed direction order D: the rotate60 orbit of (1, 0).
pub const DIRECTIONS: [(i16, i16); 6] = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)];

pub const NUM_FEATURES: usize = 15;
pub const F_OWN_STONE: usize = 0;
pub const F_OPP_STONE: usize = 1;
pub const F_EMPTY: usize = 2;
pub const F_LEGAL: usize = 3;
pub const F_PHASE_SECOND: usize = 4;
pub const F_FIRST_STONE: usize = 5;
pub const F_PLAYER_COLOUR: usize = 6;
pub const F_OWN_RECENCY: usize = 7;
pub const F_OPP_RECENCY: usize = 8;
pub const F_OPP_HOT: usize = 9;
pub const F_OWN_HOT: usize = 10;
pub const F_DIST_TO_STONE: usize = 11;
pub const F_OPP_LAST_TURN: usize = 12;
pub const F_OPP_WIN_NOW: usize = 13;
pub const F_OWN_WIN_NOW: usize = 14;

pub const HOT_MIN_COUNT: u32 = 4;
pub const WIN_NOW_COUNT: u32 = 5;
pub const HOT_MIN_PLACEMENTS: u32 = 7;
pub const WINDOW_LEN: usize = 6;

pub const DIST_SCALE: f32 = LEGAL_RADIUS as f32;
