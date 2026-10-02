/* The board overlay of an analysis record, for web/index.html (a classic script: it defines boardOverlay). */

/**
 * What web/index.html draws on the board for evaluation `ev` of the position whose stones are `stones`
 * ([q, r, player] in play order): {candidates: [{q, r, n, weight}], plies: [{q, r, n, player, fade}], six: [[q, r]]}.
 * Unproven (no `ev.proof`): every `ev.top` row is a candidate, `n` its rank and `weight` its policy share relative
 * to the best row's (1 for the best, so the strongest ghost marks the search's choice whatever its absolute share).
 * Proven: no candidates; `ev.pv` in its players' colours, each stone numbered by its ply (its fourth entry, so the
 * defender stones the line leaves out leave a gap; saves without one are numbered in order), `fade` from 1 at the
 * first ply down to .45 at the last, and `six` the completed line through its last stone (sorted by q, then r),
 * [] if none.
 */
function boardOverlay(ev, stones) {
  const empty = {candidates: [], plies: [], six: []};
  if (!ev) return empty;
  if (!ev.proof) {
    const top = ev.top || [], best = Math.max(...top.map(t => t[2]), 1e-9);
    return {...empty, candidates: top.map((t, i) => ({q: t[0], r: t[1], n: i + 1, weight: Math.min(1, t[2] / best)}))};
  }
  const pv = ev.pv || [], last = pv.at(-1);
  const ply = (p, i) => p[3] ?? i + 1, first = pv.length && ply(pv[0], 0), span = pv.length && ply(last, pv.length - 1) - first;
  const plies = pv.map((p, i) => ({q: p[0], r: p[1], n: ply(p, i), player: p[2], fade: span ? 1 - .55 * (ply(p, i) - first) / span : 1}));
  if (!last) return {...empty, plies};
  const at = (q, r) => `${q},${r}`, own = new Map([...stones, ...pv].map(p => [at(p[0], p[1]), p[2]]));
  for (const [dq, dr] of [[1, 0], [0, 1], [1, -1]]) {
    const run = [[last[0], last[1]]];
    for (const sign of [1, -1]) {
      for (let q = last[0] + sign * dq, r = last[1] + sign * dr; own.get(at(q, r)) === last[2]; q += sign * dq, r += sign * dr) run.push([q, r]);
    }
    if (run.length >= 6) return {...empty, plies, six: run.sort((a, b) => a[0] - b[0] || a[1] - b[1])};
  }
  return {...empty, plies};
}
