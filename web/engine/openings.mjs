export function random(seed) {
  let state = seed >>> 0;
  return () => { state += 0x6D2B79F5; let n = Math.imul(state ^ state >>> 15, 1 | state); n ^= n + Math.imul(n ^ n >>> 7, 61 | n); return ((n ^ n >>> 14) >>> 0) / 4294967296; };
}
const shuffle = (items, rng) => { const out = [...items]; for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [out[i], out[j]] = [out[j], out[i]]; } return out; };
export function transform(moves, symmetry) {
  return moves.map(([q, r]) => {
    if (symmetry >= 6) [q, r] = [q + r, -r];
    for (let i = 0; i < symmetry % 6; i++) [q, r] = [-r, q + r];
    return [q, r];
  });
}
export class OpeningBook {
  constructor(data) { this.data = data; this.nodes = data.nodes; }
  pool(mode, count = 8) {
    if (!['narrow', 'wide', 'all'].includes(mode)) throw Error('Unknown opening book range');
    const nodes = this.nodes.filter(n => mode === 'all' || !n.off_policy).sort((a, b) => a.key.localeCompare(b.key));
    return mode === 'narrow' ? nodes.sort((a, b) => b.champion_probability - a.champion_probability || a.key.localeCompare(b.key)).slice(0, count) : nodes;
  }
  select(mode, count, seed = 0) {
    const pool = this.pool(mode, count), rng = random(seed);
    if (!Number.isInteger(count) || count < 1 || count > pool.length) throw Error(`The ${mode} book has ${pool.length} unique starts`);
    const selected = mode === 'narrow' ? pool.slice(0, count) : shuffle(pool, rng).slice(0, count);
    return shuffle(selected, rng).map(n => ({...n, symmetry: Math.floor(rng() * 12)})).map(n => ({...n, moves: transform(n.moves, n.symmetry)}));
  }
  /** A start of `mode` in a random orientation: `node` when given, else walking the branches that still hold the
   * start `side` has played least (`coverage` counts `side:key`). */
  pick(mode, coverage, side, rng = Math.random, node = null) {
    if (node) { const symmetry = Math.floor(rng() * 12); return {...node, symmetry, moves: transform(node.moves, symmetry)}; }
    let candidates = this.pool(mode), prefix = [];
    // Walk the book's branches that still contain an unplayed start, as Play does.
    const plays = n => coverage[`${side}:${n.key}`] || 0, least = Math.min(...candidates.map(plays));
    candidates = candidates.filter(n => plays(n) === least);
    while (true) {
      const ended = candidates.filter(n => n.moves.length === prefix.length);
      if (ended.length) { const n = ended[Math.floor(rng() * ended.length)], symmetry = Math.floor(rng() * 12); return {...n, symmetry, moves: transform(n.moves, symmetry)}; }
      const branches = [...new Set(candidates.map(n => n.moves[prefix.length].join(',')))];
      const chosen = branches[Math.floor(rng() * branches.length)];
      candidates = candidates.filter(n => n.moves[prefix.length].join(',') === chosen);
      prefix.push(chosen);
    }
  }
}
