/* Bucketed axial crops: python/hexcrop.py encode_game with the legal actions given, smallest-crop symmetry. */

export const BUCKETS = [24, 32, 40, 48, 64, 96, 128, 192, 256];
export const PLANES = 8;
export const CHANNELS = 20;
const WINDOW = 6;
const AXES = [[1, 0], [0, 1], [1, -1]];
const HALO = 4;
const SYMMETRIES = [[[1, 0], [0, 1]], [[0, 1], [-1, 1]], [[-1, 1], [-1, 0]], [[-1, 0], [0, -1]], [[0, -1], [1, -1]],
  [[1, -1], [1, 0]], [[0, 1], [1, 0]], [[-1, 1], [0, 1]], [[-1, 0], [-1, 1]], [[0, -1], [-1, 0]], [[1, -1], [0, -1]],
  [[1, 0], [1, -1]]];
const AXIS_OF = [[0, 1], [1, 2], [2, 0], [0, 1], [1, 2], [2, 0], [1, 0], [0, 2], [2, 1], [1, 0], [0, 2], [2, 1]];
const POSITIVE = [[1, 1], [0, 1], [0, 1], [0, 0], [1, 0], [1, 0], [1, 1], [0, 1], [0, 1], [0, 0], [1, 0], [1, 0]];

function bounds(...lists) {
  const low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];
  for (const list of lists) for (const [q, r] of list) {
    const v = [q, r, q + r];
    for (let j = 0; j < 3; j++) { if (v[j] < low[j]) low[j] = v[j]; if (v[j] > high[j]) high[j] = v[j]; }
  }
  return [low, high];
}

function choose([low, high], halo) {
  let best = 0, side = Infinity;
  AXIS_OF.forEach((axes, k) => {
    const s = Math.max(...axes.map(a => high[a] - low[a])) + 1 + 2 * halo;
    if (s < side) { side = s; best = k; }
  });
  return [best, side];
}

/**
 * Encodes a non-terminal position: `history` [[q, r], ...], `actions` its legal moves in native order. Returns
 * {planes Uint8Array [8*size*size], size, cells Int32Array (flat crop index per action, -1 when far), far}.
 */
export function encode(history, actions, rectangular = false) {
  const n = history.length, player = ((n + 1) >> 1) % 2, remaining = n % 2 ? 2 : 1;
  let box = bounds(history, actions), halo = 0, [k, side] = choose(box, 0);
  if (side > BUCKETS[BUCKETS.length - 1]) {
    halo = HALO;
    box = bounds(history);
    [k, side] = choose(box, HALO);
    if (side > BUCKETS[BUCKETS.length - 1]) throw new Error(`Stones span ${side} cells with halo; the largest bucket is 256`);
  }
  const size = BUCKETS.find(b => b >= side), m = SYMMETRIES[k], low = [0, 0], extent = [0, 0];
  for (let j = 0; j < 2; j++) {
    const a = AXIS_OF[k][j];
    low[j] = (POSITIVE[k][j] ? box[0][a] : -box[1][a]) - halo;
    extent[j] = (POSITIVE[k][j] ? box[1][a] : -box[0][a]) + halo - low[j] + 1;
  }
  const width = rectangular ? Math.max(24, 8 * Math.ceil(extent[0] / 8)) : size;
  const height = rectangular ? Math.max(24, 8 * Math.ceil(extent[1] / 8)) : size;
  const ox = Math.floor((width - extent[0]) / 2), oy = Math.floor((height - extent[1]) / 2);
  const sx = ox - low[0], sy = oy - low[1], area = height * width;
  const at = ([q, r]) => [q * m[0][0] + r * m[1][0] + sx, q * m[0][1] + r * m[1][1] + sy];
  const planes = new Uint8Array(PLANES * area), cells = new Int32Array(actions.length);
  let far = 0;
  actions.forEach((action, i) => {
    const [x, y] = at(action);
    if (halo && !(x >= ox && x < ox + extent[0] && y >= oy && y < oy + extent[1])) { cells[i] = -1; far++; return; }
    cells[i] = y * width + x;
    planes[2 * area + cells[i]] = 1;
  });
  history.forEach((point, i) => {
    const [x, y] = at(point);
    planes[(((i + 1) >> 1) % 2 === player ? 0 : area) + y * width + x] = 1;
  });
  for (let y = oy; y < oy + extent[1]; y++) planes.fill(1, 3 * area + y * width + ox, 3 * area + y * width + ox + extent[0]);
  planes.fill(1, (remaining === 1 ? 4 : 5) * area, (remaining === 1 ? 5 : 6) * area);
  const start = remaining === 2 || n === 0 ? n : n - 1;
  if (start < n) { const [x, y] = at(history[n - 1]); planes[6 * area + y * width + x] = 1; }
  for (const point of history.slice(Math.max(0, start - 2), start)) { const [x, y] = at(point); planes[7 * area + y * width + x] = 1; }
  return {planes, size, height, width, cells, far};
}

/**
 * The network input of python/export_web.py inputs for one encoded sample, written into `out` (Float32Array of
 * CHANNELS*size*size, zero-filled) at `offset`: the planes, hexnet.LineFeatures (for own then opponent stones and
 * each axis, the best stone count over windows of six containing the cell that lie inside the crop and hold no
 * enemy stone, divided by six; then empty cells where the best own count is >= 4, opponent >= 4, own >= 5,
 * opponent >= 5) and two zero channels, all times the crop mask.
 */
export function features({planes, size, height = size, width = size}, out = new Float32Array(CHANNELS * height * width), offset = 0) {
  const area = height * width, own = planes.subarray(0, area), opp = planes.subarray(area, 2 * area), mask = planes.subarray(3 * area, 4 * area);
  for (let c = 0; c < PLANES; c++) for (let i = 0; i < area; i++) out[offset + c * area + i] = planes[c * area + i] * mask[i];
  const inside = (x, y) => x >= 0 && x < width && y >= 0 && y < height;
  const mine = new Uint8Array(area), theirs = new Uint8Array(area), open = [new Uint8Array(area), new Uint8Array(area)];
  AXES.forEach(([dx, dy], a) => {
    open[0].fill(0); open[1].fill(0);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      let o = 0, p = 0, m = 0;
      for (let i = 0; i < WINDOW; i++) {
        const u = x + i * dx, v = y + i * dy;
        if (!inside(u, v)) break;
        const j = v * width + u;
        o += own[j]; p += opp[j]; m += mask[j];
      }
      if (m < WINDOW) continue;
      if (!p) open[0][y * width + x] = o;
      if (!o) open[1][y * width + x] = p;
    }
    for (let side = 0; side < 2; side++) {
      const channel = offset + (PLANES + 3 * side + a) * area, best = side ? theirs : mine;
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        let b = 0;
        for (let i = 0; i < WINDOW; i++) {
          const u = x - i * dx, v = y - i * dy;
          if (inside(u, v) && open[side][v * width + u] > b) b = open[side][v * width + u];
        }
        const j = y * width + x;
        out[channel + j] = Math.fround(b / WINDOW) * mask[j];
        if (b > best[j]) best[j] = b;
      }
    }
  });
  const threats = offset + (PLANES + 6) * area;
  for (let j = 0; j < area; j++) {
    if (!mask[j] || own[j] || opp[j]) continue;
    out[threats + j] = mine[j] >= 4 ? 1 : 0;
    out[threats + area + j] = theirs[j] >= 4 ? 1 : 0;
    out[threats + 2 * area + j] = mine[j] >= 5 ? 1 : 0;
    out[threats + 3 * area + j] = theirs[j] >= 5 ? 1 : 0;
  }
  return out;
}
