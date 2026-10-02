//! Axial hex coordinate math.
//!
//! Hexo uses an unlimited hex grid. Axial coordinates store two cube axes
//! (`q`, `r`); the third cube axis is derived as `s = -q - r`.

use serde::{Deserialize, Serialize};
use std::ops::{Add, Neg, Sub};

/// One cell on the infinite Hexo board.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HexCoord {
    /// Axial q coordinate.
    pub q: i16,
    /// Axial r coordinate.
    pub r: i16,
}

impl HexCoord {
    /// Board center and the only legal opening placement.
    pub const ZERO: Self = Self { q: 0, r: 0 };

    /// Construct an axial coordinate.
    pub const fn new(q: i16, r: i16) -> Self {
        Self { q, r }
    }

    /// Derived cube coordinate used for distance math.
    pub const fn s(self) -> i16 {
        -self.q - self.r
    }

    /// Multiply both axial components by a scalar.
    ///
    /// This is useful when walking a line axis by N cells.
    pub fn scale(self, n: i16) -> Self {
        Self {
            q: self.q * n,
            r: self.r * n,
        }
    }
}

impl Add for HexCoord {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            q: self.q + rhs.q,
            r: self.r + rhs.r,
        }
    }
}

impl Sub for HexCoord {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            q: self.q - rhs.q,
            r: self.r - rhs.r,
        }
    }
}

impl Neg for HexCoord {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            q: -self.q,
            r: -self.r,
        }
    }
}

/// Hex-grid distance between two axial coordinates.
pub fn hex_distance(a: HexCoord, b: HexCoord) -> i16 {
    let dq = a.q - b.q;
    let dr = a.r - b.r;
    let ds = -dq - dr;
    dq.abs().max(dr.abs()).max(ds.abs())
}

/// Iterate every coordinate within `radius` hex steps of `center`.
///
/// The iterator walks a cube-coordinate diamond but yields axial coordinates.
pub fn coords_within_radius(center: HexCoord, radius: i16) -> impl Iterator<Item = HexCoord> {
    (-radius..=radius).flat_map(move |dq| {
        let r_min = (-radius).max(-dq - radius);
        let r_max = radius.min(-dq + radius);
        (r_min..=r_max).map(move |dr| HexCoord {
            q: center.q + dq,
            r: center.r + dr,
        })
    })
}
