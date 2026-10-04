/* The Play notations, in HTTTX coordinates. Legality is checked by the session's native rules. */
const directions = [[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]], symbols = '>qp<bd';
const site = ([q, r]) => [q + r, -r || 0], fromSite = ([x, y]) => [x + y, -y || 0];
const distance = (a, b) => Math.max(Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]), Math.abs(a[0] + a[1] - b[0] - b[1]));
const coord = p => p.join(',');

export function htttx(history) {
  if (!history.length) throw Error('HTTTX v1 needs the opening stone');
  let text = 'version[1];', spans = [];
  for (let start = 1, turn = 1; start < history.length; start += 2, turn++) {
    text += `\n${turn}. `;
    for (const p of history.slice(start, start + 2)) {
      const token = `[${p}]`;
      spans.push([text.length, text.length + token.length, ...p]);
      text += token;
    }
    text += ';';
  }
  return {text, spans};
}

function readHTTTX(text) {
  const metadata = {};
  text = text.trim();
  if (/^[a-z]/i.test(text)) {
    while (true) {
      const match = text.match(/^\s*((?:[a-z]\s*)+)\[([^\]]*)\]/);
      if (!match) throw Error('Malformed HTTTX metadata');
      const name = match[1].replace(/\s/g, '');
      if (name in metadata) throw Error('Duplicate metadata key');
      metadata[name] = match[2];
      text = text.slice(match[0].length).trimStart();
      if (text.startsWith(';')) { text = text.slice(1); break; }
    }
  }
  if ('version' in metadata && !/^0*1$/.test(metadata.version)) throw Error('Only HTTTX version 1 is supported');
  text = text.replace(/\s/g, '');
  const history = [[0, 0]];
  for (let turn = 1; text; turn++) {
    const m = text.match(/^(\d+)\.\[(-?\d+),(-?\d+)\](?:\[(-?\d+),(-?\d+)\])?!*;/);
    if (!m || +m[1] !== turn) throw Error('Expected consecutive numbered HTTTX turns');
    history.push([+m[2], +m[3]]);
    if (m[4] !== undefined) history.push([+m[4], +m[5]]);
    else if (m[0].length !== text.length) throw Error('A single-stone turn must be the final turn');
    text = text.slice(m[0].length);
  }
  return history;
}

function ringLabel(n) {
  let label = '';
  while (n) { n--; label = String.fromCharCode(65 + n % 26) + label; n = Math.floor(n / 26); }
  return label;
}
function ringOffset(cell, origin, baseline) {
  const ring = distance(cell, origin), d = directions[baseline];
  let q = origin[0] + d[0] * ring, r = origin[1] + d[1] * ring;
  for (let sector = 0; sector < 6; sector++) {
    const [dq, dr] = directions[(baseline + sector + 2) % 6], step = Math.max(Math.abs(cell[0] - q), Math.abs(cell[1] - r));
    if (step < ring && cell[0] - q === dq * step && cell[1] - r === dr * step) return [ring, sector, step];
    q += dq * ring; r += dr * ring;
  }
  throw Error('Invalid ring coordinate');
}
export function rectilinear(history) {
  if (!history.length) throw Error('Rectilinear notation needs a stone');
  if (history.length === 1) return {text: 'x', spans: [[0, 1, ...history[0]]]};
  const cells = history.map(site), origin = cells[0];
  const score = baseline => cells.slice(1).flatMap(c => { const [ring, sector, step] = ringOffset(c, origin, baseline); return [ring, sector * ring + step]; });
  const compare = (a, b) => { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] - b[i]; return 0; };
  const baseline = [0, 1, 2, 3, 4, 5].sort((a, b) => compare(score(a), score(b)) || a - b)[0];
  let text = `x, ${symbols[baseline]} @(0, 0)`, spans = [[0, 1, ...history[0]]];
  for (let i = 1; i < cells.length; i++) {
    if (i % 2) text += i % 4 === 1 ? ' o' : ' x';
    const [ring, sector, step] = ringOffset(cells[i], origin, baseline);
    const token = ringLabel(ring) + (ring === 1 ? sector : `${sector ? sector + '.' : ''}${step}`);
    text += ' '; spans.push([text.length, text.length + token.length, ...history[i]]); text += token;
  }
  return {text, spans};
}

function readTurns(text, implicit) {
  if (text.trim() === '0') return {origin: null, turns: []};
  const match = text.match(/^\s*([bdpq<>])?\s*(CW|CCW)?\s*(?:@\s*\((-?\d+),\s*(-?\d+)\)\s*:?)?\s*([xo](?:\s+[A-Z]+(?:[0-5]\.)?\d+)+(?:\s+[xo](?:\s+[A-Z]+(?:[0-5]\.)?\d+)+)*)\s*$/);
  if (!match || implicit && match[3] !== undefined) throw Error('Not BKE turns');
  const baseline = symbols.indexOf(match[1] || 'd'), origin = match[3] === undefined ? [0, 0] : [+match[3], +match[4]], turns = [];
  for (const token of match[5].split(/\s+/)) {
    if (token === 'x' || token === 'o') { turns.push({player: token, moves: []}); continue; }
    const m = token.match(/^([A-Z]+)(?:(\d)\.)?(\d+)$/);
    let ring = 0;
    for (const c of m[1]) ring = ring * 26 + c.charCodeAt(0) - 64;
    let offset = m[2] === undefined ? +m[3] : +m[2] * ring + +m[3];
    if (offset >= 6 * ring) throw Error('Offset is outside the ring');
    if (match[2] === 'CCW') offset = (6 * ring - offset) % (6 * ring);
    const sector = Math.floor(offset / ring), step = offset % ring, d = directions[baseline];
    let q = origin[0] + d[0] * ring, r = origin[1] + d[1] * ring;
    for (let s = 0; s <= sector; s++) { const [dq, dr] = directions[(baseline + s + 2) % 6], n = s < sector ? ring : step; q += dq * n; r += dr * n; }
    turns.at(-1).moves.push([q, r]);
  }
  return {origin, turns};
}
function drawing(text) {
  const columns = text.startsWith('c'), step = columns ? [0, 1] : [1, 0], down = columns ? [1, 0] : [0, 1];
  let q = 0, r = 0, row = [0, 0], i = columns ? 1 : 0;
  const cells = new Map(), advance = n => { q += step[0] * n; r += step[1] * n; };
  while (i < text.length) {
    const c = text[i++];
    if (/\d/.test(c)) { const n = text.slice(i - 1).match(/^\d+/)[0]; i += n.length - 1; advance(+n); continue; }
    if (c === '(' || c === '[') {
      const close = c === '(' ? ')' : ']'; let depth = 0, found = false;
      while (i < text.length) { const t = text[i++]; if (t === '\\') i++; else if (c === '[' && t === '[') depth++; else if (t === close) { if (!depth) { found = true; break; } depth--; } }
      if (!found) throw Error('Unterminated highlight or label');
      continue;
    }
    if (c === ' ' || c === '\r') continue;
    if (c === '/' || c === '\n') { row = [row[0] + down[0], row[1] + down[1]]; [q, r] = row; continue; }
    if (/[xo]/i.test(c)) cells.set(coord([q, r]), {point: [q, r], player: c.toLowerCase()});
    else if (c === '-') advance(1);
    else if (c !== '.' && c !== '!') throw Error(`Unexpected character ${c} in Rectilinear notation`);
    advance(1);
  }
  return cells;
}
function readRectilinear(text, native) {
  let depth = 0, split = -1;
  for (let i = 0; i < text.length; i++) { if (text[i] === '\\') i++; else if (text[i] === '[') depth++; else if (text[i] === ']') depth--; else if (text[i] === ',' && !depth) { split = i; break; } }
  let parsed, cells;
  if (split >= 0) { cells = drawing(text.slice(0, split)); parsed = readTurns(text.slice(split + 1), false); }
  else {
    try { parsed = readTurns(text, true); cells = drawing(parsed.turns[0]?.player === 'x' ? 'o' : 'x'); }
    catch { cells = drawing(text); parsed = {origin: null, turns: []}; }
  }
  const points = [...cells.values()], count = points.length, later = (count - 1) / 2, firstCount = 1 + 2 * Math.floor(later / 2);
  if (!count || count % 2 === 0) throw Error('The drawing must end a complete turn');
  const first = ['x', 'o'].find(p => points.filter(c => c.player === p).length === firstCount);
  if (!first) throw Error('The drawn stones are not a sequence of complete turns');
  let left = 20000;
  function extend(history, used, origin) {
    if (history.length === count) return history;
    const owner = ((history.length + 1) >> 1) % 2 === 0 ? first : first === 'x' ? 'o' : 'x';
    const near = c => Math.min(...history.map(p => { const s = site(p); return distance([s[0] + origin[0], s[1] + origin[1]], c.point); }));
    const options = points.filter(c => c.player === owner && !used.has(coord(c.point))).sort((a, b) => near(a) - near(b));
    for (const c of options) {
      if (--left < 0) return null;
      const point = fromSite([c.point[0] - origin[0], c.point[1] - origin[1]]), next = [...history, point];
      try { const state = native.game(next); if (state.winner >= 0 && (next.length < count || parsed.turns.length)) continue; }
      catch { continue; }
      const found = extend(next, new Set([...used, coord(c.point)]), origin);
      if (found) return found;
    }
    return null;
  }
  const openings = points.filter(c => c.player === first).sort((a, b) => (coord(a.point) === coord(parsed.origin || []) ? -1 : coord(b.point) === coord(parsed.origin || []) ? 1 : a.point[1] - b.point[1] || a.point[0] - b.point[0]));
  for (const c of openings) {
    let history = extend([[0, 0]], new Set([coord(c.point)]), c.point);
    if (!history) continue;
    let mover = later % 2 === 0 ? first === 'x' ? 'o' : 'x' : first;
    for (let i = 0; i < parsed.turns.length; i++) {
      const t = parsed.turns[i];
      if (t.player !== mover || t.moves.length > 2 || t.moves.length < 2 && i < parsed.turns.length - 1) throw Error('BKE turns are out of order or incomplete');
      history.push(...t.moves.map(p => fromSite([p[0] - c.point[0], p[1] - c.point[1]])));
      mover = mover === 'x' ? 'o' : 'x';
    }
    native.game(history);
    return history;
  }
  throw Error('The drawn stones cannot be played as legal turns');
}

export function tyto(history) {
  if (!history.length) throw Error('A Tyto link needs a stone');
  const bytes = [];
  for (const p of history.slice(1).map(site)) for (let v of p) {
    v = v >= 0 ? 2 * v : -2 * v - 1;
    while (v > 127) { bytes.push(v % 128 + 128); v = Math.floor(v / 128); }
    bytes.push(v);
  }
  return {text: 'https://hexo.tyto.cc/analysis#c=' + btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, ''), spans: []};
}
function readTyto(code) {
  code = code.replace(/%20|\s/g, '');
  if (!/^[\w-]*$/.test(code) || code.length % 4 === 1) throw Error('Not a Tyto analysis link');
  const data = atob(code.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - code.length % 4) % 4)), values = [];
  let value = 0, shift = 0;
  for (const c of data) { const byte = c.charCodeAt(0); value += (byte & 127) * 2 ** shift; shift += 7; if (shift > 53) throw Error('Coordinate is too large'); if (!(byte & 128)) { values.push(value % 2 ? -(value + 1) / 2 : value / 2); value = shift = 0; } }
  if (shift || values.length % 2) throw Error('Truncated Tyto analysis link');
  return [[0, 0], ...Array.from({length: values.length / 2}, (_, i) => fromSite(values.slice(2 * i, 2 * i + 2)))];
}

const MAX_GAME_BYTES = 32 * 1024 * 1024;

/** Lossless file compression, also used for immutable analysis records in IndexedDB. */
export async function compressFile(blob) {
  if (typeof CompressionStream !== 'function') return blob;
  const data = await new Response(blob.stream().pipeThrough(new CompressionStream('gzip'))).blob();
  return data.size < blob.size ? new Blob([data], {type: 'application/gzip'}) : blob;
}

/** Accept ordinary files and their gzip downloads. readGame limits individual games; backups may be larger. */
export async function readGameFile(file) {
  const header = new Uint8Array(await file.slice(0, 2).arrayBuffer());
  if (header[0] !== 31 || header[1] !== 139) return file.text();
  return new Response(file.stream().pipeThrough(new DecompressionStream('gzip'))).text();
}

export async function readGame(text, native, fetcher = fetch) {
  text = text.trim();
  if (text.length > MAX_GAME_BYTES) throw Error('Game text exceeds 32 MB');
  let history;
  if (/^https?:\/\//.test(text)) {
    const url = new URL(text), params = new URLSearchParams(url.hash.slice(1));
    if (url.hostname === 'hexo.tyto.cc' && params.has('c')) history = readTyto(params.get('c'));
    else {
      let response;
      try {
        if (url.hostname === 'hexo.tyto.cc' && params.has('g')) {
          throw Error('hexo.tyto.cc does not let other sites read its games');
        } else if (['hexo.did.science', 'hexo.mineking.dev'].includes(url.hostname)) {
          // Both sites serve the same API; only the mineking mirror lets other sites read it.
          const found = url.pathname.match(/^\/(?:account\/)?(games|sandbox)\/([\w-]{1,64})\/?$/);
          if (!found) throw Error('Not a game or sandbox link');
          const game = found[1] === 'games';
          response = await fetcher(`https://hexo.mineking.dev/proxy/api/${game ? `finished-games/${found[2]}` : `sandbox-positions/${found[2].toLowerCase()}`}`);
          if (!response.ok) throw Error(`HTTP ${response.status}`);
          const data = await response.json(), rows = game ? [...data.moves].sort((a, b) => a.moveNumber - b.moveNumber)
            : [...data.gamePosition.cells].sort((a, b) => a.moveId - b.moveId);
          if (!rows.length) throw Error('That link holds no stones');
          const origin = [rows[0].x, rows[0].y], owners = [];
          history = rows.map((row, i) => {
            const id = game ? row.playerId : row.player;
            if (!owners.includes(id)) owners.push(id);
            if (owners.indexOf(id) !== ((i + 1) >> 1) % 2) throw Error('Player order does not match HeXO turns');
            return fromSite([row.x - origin[0], row.y - origin[1]]);
          });
        } else throw Error('Unsupported game link');
      } catch (error) { throw Error(`Could not import this link: ${error.message}. You can paste its HTTTX or game file instead.`); }
    }
  } else if (/^[{[]/.test(text)) { const data = JSON.parse(text); history = data.history ?? data; }
  else if (text.includes(';') || /^version\s*\[/.test(text)) history = readHTTTX(text);
  else history = readRectilinear(text, native);
  if (!Array.isArray(history) || history.length > 4097 || history.some(p => !Array.isArray(p) || p.length !== 2 || p.some(v => !Number.isSafeInteger(v)))) throw Error('Expected at most 4097 integer coordinate pairs');
  native.game(history);
  return history;
}

export const exportGame = (history, format) => {
  if (format === 'htttx') return htttx(history);
  if (format === 'rectilinear') return rectilinear(history);
  if (format === 'tyto') return tyto(history);
  throw Error('Unknown notation format');
};
