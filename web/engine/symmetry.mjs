/* The 12 symmetries of the hex grid on HTTTX axial cells, as python/hexcrop.py SYMMETRIES lists them as a set.
 * Symmetry k turns a cell k % 6 times by 60 degrees, (q, r) -> (-r, q + r), after mirroring it, (q, r) -> (q + r, -r),
 * when k >= 6. With the play page's projection (q to the right, r up and to the right) a turn is counter-clockwise
 * and the mirror flips the board top to bottom. */
export const TURN_LEFT = 1, TURN_RIGHT = 5, MIRROR = 6;

/** The cell [q, r] under `symmetry`. */
export function turn([q, r], symmetry) {
  if (symmetry >= 6) [q, r] = [q + r, -r];
  for (let i = 0; i < symmetry % 6; i++) [q, r] = [-r, q + r];
  return [q, r];
}

/** Every cell of `moves` under `symmetry`. */
export function transform(moves, symmetry) {
  return moves.map(move => turn(move, symmetry));
}

/** The symmetry that applies `first`, then `then`. */
export function compose(first, then) {
  const image = transform(transform([[1, 0], [0, 1]], first), then).join();
  return [...Array(12).keys()].find(k => transform([[1, 0], [0, 1]], k).join() === image);
}

/** The symmetry that undoes `symmetry`. */
export function inverse(symmetry) {
  return [...Array(12).keys()].find(k => compose(symmetry, k) === 0);
}
