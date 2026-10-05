/* Rows of a browser analysis record: the candidate rows (python/play.py top_rows), the principal variation of a
 * verified solver certificate (python/play.py principal_variation) and the game's table of proven positions
 * (python/play.py Proofs). */

/** Most attacker turns on any path of `certificate` from each node, the completing turn included. */
function proofArena(certificate) {
  if (!certificate.nodes.some(n => n.kind === 'stamp' || n.kind === 'stamp_link')) return certificate;
  const nodes = [];
  const append = (document, depth = 0) => {
    if (depth > 32 || nodes.length + document.nodes.length > 200000) throw new Error('Proof source expansion limit');
    const offset = nodes.length;
    for (const node of document.nodes) nodes.push({...node});
    document.nodes.forEach((original, i) => {
      const node = nodes[offset + i];
      if (node.kind === 'stamp') nodes[offset + i] = {kind: 'link', child: append(node.source.certificate, depth + 1)};
      else if (node.kind === 'stamp_link') nodes[offset + i] = {kind: 'link', child: offset + node.source};
      else {
        if (node.child !== undefined) node.child += offset;
        if (node.fallback !== undefined) node.fallback += offset;
        for (const field of ['responses', 'alternatives']) {
          if (node[field]) node[field] = original[field].map(r => ({...r, child: r.child + offset}));
        }
      }
    });
    return offset + document.root;
  };
  const root = append(certificate);
  return {nodes, root};
}

function depths(certificate, known = []) {
  const memo = new Map();
  const turns = index => {
    if (!memo.has(index)) {
      const node = certificate.nodes[index];
      memo.set(index, node.kind === 'link' ? turns(node.child)
        : node.kind === 'zone_replies' ? Math.max(turns(node.fallback), ...node.responses.map(r => turns(r.child)))
        : node.kind === 'attacker_move' ? 1 + turns(node.child)
        : node.kind === 'defender_replies' ? Math.max(...node.responses.map(r => turns(r.child)))
        : node.kind === 'exact' ? proofTurns(known[node.fact].plies, known[node.fact].history.length % 2 ? 2 : 1,
          sideAt(known[node.fact].history.length) === known[node.fact].winner) : 1);
    }
    return memo.get(index);
  };
  return turns;
}

/** Steps between hex cells `a` and `b` ([q, r] axial). */
function distance(a, b) {
  const dq = a[0] - b[0], dr = a[1] - b[1];
  return Math.max(Math.abs(dq), Math.abs(dr), Math.abs(dq + dr));
}

/**
 * The principal variation of a verified certificate for the side to move at `history` (play.principal_variation):
 * {pv: [[q, r, player, ply], ...] up to the winning stone, plies: placements to it}. The attacker takes the
 * certificate's primary choice, the defender the covered reply lasting the most attacker turns, among equals the one
 * nearest the attacker's last stone (least summed hex distance), then the first listed; at an unstoppable fork the
 * defender's two stones are left out, their plies skipped, and the shortest threat completes.
 */
export function principalVariation(native, history, certificate, {attacker = native.game(history).player, known = []} = {}) {
  certificate = proofArena(certificate);
  const turns = depths(certificate, known), current = history.map(p => [...p]), pv = [];
  const near = reply => reply.action.reduce((sum, cell) => sum + distance(cell, pv.at(-1) || history.at(-1)), 0);
  let plies = 0, index = certificate.root;
  while (native.game(current).winner < 0) {
    const node = certificate.nodes[index];
    if (node.kind === 'link') { index = node.child; continue; }
    if (node.kind === 'exact') {
      const fact = known[node.fact];
      const after = node.after || [], line = fact.pv || [];
      if (JSON.stringify(line.slice(0, after.length).map(p => p.slice(0, 2))) === JSON.stringify(after)) {
        pv.push(...line.slice(after.length).map(([q, r, side, ply]) => [q, r, side, ply + plies - after.length]));
      }
      plies += fact.plies - after.length;
      break;
    }
    if (node.kind === 'unstoppable') {
      if (node.threats?.length) {
        const threat = node.threats.reduce((a, b) => b.length < a.length ? b : a);
        const remaining = native.game(current).remaining;
        threat.forEach(([q, r], i) => pv.push([q, r, attacker, plies + remaining + 1 + i]));
        plies += remaining + threat.length;
      }
      break;
    }
    let action;
    if (node.kind === 'defender_replies' || node.kind === 'zone_replies') {
      const longer = (a, b) => turns(b.child) - turns(a.child) || near(a) - near(b);
      const legal = node.kind === 'zone_replies' ? new Set(native.legal(current).map(p => p.join(','))) : null;
      const responses = legal ? node.responses.filter(r => legal.has(r.action[0].join(','))) : node.responses;
      if (!responses.length) break;
      const reply = responses.reduce((a, b) => longer(a, b) > 0 ? b : a);
      [action, index] = [reply.action, reply.child];
    } else [action, index] = [node.action, node.child];
    for (const [q, r] of action) {
      const state = native.game(current);
      if (state.winner >= 0) break;
      plies++;
      pv.push([q, r, state.player, plies]);
      current.push([q, r]);
    }
    if (index === undefined) break;
  }
  return {pv, plies};
}

/** A `top` row as python/play.py move_row: [q, r, probability], then from a search the mover's win probability after
 * the stone and 1 or -1 when the search proved that it wins or loses, else 0. */
function moveRow([q, r], probability, value) {
  const row = [q, r, Math.round(probability * 1e4) / 1e4];
  return value === undefined ? row : [...row, Math.round((value + 1) / 2 * 1e4) / 1e4, value >= 1 ? 1 : value <= -1 ? -1 : 0];
}

/** The five `top` rows as python/play.py top_rows (without `won`): `lead` first, then proven wins, the others by
 * policy and proven losses last; a stone below a policy share of 0.00005 only when the search proved it wins. */
export function topRows(actions, policy, values, lead) {
  const rank = i => !values ? 1 : values[i] >= 1 ? 0 : values[i] <= -1 ? 2 : 1;
  const first = actions.findIndex(a => a[0] === lead[0] && a[1] === lead[1]);
  const order = Array.from(policy, (p, i) => i).filter(i => i !== first && (policy[i] >= .00005 || rank(i) === 0))
    .sort((a, b) => rank(a) - rank(b) || policy[b] - policy[a]);
  return (first >= 0 ? [first, ...order] : order).slice(0, 5).map(i => moveRow(actions[i], policy[i], values?.[i]));
}

const sideAt = ply => ply === 0 ? 0 : ((ply - 1 >> 1) + 1) % 2;
const shifted = (pv, by) => pv.map(([q, r, side, ply]) => [q, r, side, ply + by]);

// A complete drawn line can break ties between equal proof bounds. A partial
// line is not a quicker win. Omitted, irrelevant defender stones keep their
// placement numbers, as in principalVariation's unstoppable ending.
function lineLength(history, winner, pv, offset = 0) {
  if (!pv.length || pv.some(p => p.length !== 4)) return Infinity;
  const own = new Set(history.filter((_, i) => sideAt(i) === winner).map(p => p.join(',')));
  for (const [q, r, side] of pv) if (side === winner) own.add(`${q},${r}`);
  const [q, r, side, ply] = pv.at(-1);
  if (side !== winner) return Infinity;
  for (const [dq, dr] of [[1, 0], [0, 1], [1, -1]]) {
    let count = 1;
    for (const sign of [-1, 1]) for (let i = 1; i < 6 && own.has(`${q + sign * i * dq},${r + sign * i * dr}`); i++) count++;
    if (count >= 6) return ply - offset;
  }
  return Infinity;
}

/** Compact a query's used premises for saving and later independent checking. */
export function proofEvidence(result) {
  const dependencies = result.dependencies || [], indices = new Map(dependencies.map((d, i) => [d.fact, i]));
  const certificate = {...result.certificate, nodes: result.certificate.nodes.map(n => n.kind === 'exact' ? {...n, fact: indices.get(n.fact)} : n)};
  return {certificate, dependencies: dependencies.map((d, i) => ({fact: i, outcome: d.outcome})), solver_build: result.build_hash};
}

/** The continuation and bound of an already verified mover certificate. */
export function winningLine(native, history, result, known = []) {
  const line = principalVariation(native, history, result.certificate, {known}), {player, remaining} = native.game(history);
  const reused = result.certificate.nodes.some(n => ['stamp', 'stamp_link', 'zone_replies'].includes(n.kind));
  return {moves: result.moves.map(m => [...m]), pv: line.pv,
    proof: {winner: player, turns: result.proof_turns, plies: reused ? remaining + 4 * (result.proof_turns - 1) : line.plies,
      ...proofEvidence(result)}};
}

/** The winner's turns in a proof `plies` placements long from a position whose mover has `remaining` stones left
 * (python/play.py proof_turns). */
export function proofTurns(plies, remaining, moverWins) {
  if (moverWins) return plies <= remaining ? 1 : 1 + Math.ceil((plies - remaining) / 4);
  return Math.ceil((plies - remaining) / 4);
}

/** `history`'s position whatever the order of its stones (python/play.py proof_key). */
export function proofKey(history) {
  const sides = [[], []];
  history.forEach(([q, r], i) => sides[sideAt(i)].push([q, r]));
  return sides.map(side => side.sort((a, b) => a[0] - b[0] || a[1] - b[1]).map(p => p.join(',')).join(' ')).join(' / ');
}

/**
 * The proven positions of a game (python/play.py Proofs): {winner, plies, pv} by `proofKey`, the winner completing six
 * within `plies` placements against any defence along the known line `pv` ([q, r, player, ply], ply from 1). `add`
 * indexes a saved evaluation holding a proof with the positions along its line; `list()` is the table as plain data
 * and `new Proofs(list)` rebuilds it, so a worker gets the session's table with each turn request.
 */
export class Proofs {
  constructor(list = []) {
    this.entries = new Map();
    this.sizes = new Map();
    this.seen = new Set();
    this.edgeCache = new Map();
    this.knownCache = new Map();
    this.choiceCache = new Map();
    this.strategies = new Map();
    for (const {history, winner, plies, pv} of list) this.put(history, winner, plies, pv);
  }
  list() {
    return [...this.entries.values()].map(({history, winner, plies, pv}) => ({history, winner, plies, pv}));
  }
  /** Saved strategies supply moves for checked replay, never changed-board verdicts. */
  replay(history) {
    const own = (h, side) => h.filter((_, i) => sideAt(i) === side).map(p => p.join(',')).sort().join(';');
    const at = [own(history, 0), own(history, 1)], applicable = new Set();
    for (const entry of this.strategies.values()) {
      const current = entry.history.slice();
      if (own(current, entry.winner) === at[entry.winner]) applicable.add(entry);
      for (let i = 0; i < entry.pv.length && current.length <= history.length + 2; i++) {
        const stone = entry.pv[i];
        if (stone.length !== 4 || stone[3] !== i + 1 || stone[2] !== sideAt(current.length)) break;
        current.push(stone.slice(0, 2));
        if (own(current, entry.winner) === at[entry.winner]) { applicable.add(entry); break; }
      }
    }
    const available = new Set([...applicable].map(entry => entry.winner));
    let cells = 0;
    return [...this.strategies.values()].reverse()
      .sort((a, b) => Number(applicable.has(b)) - Number(applicable.has(a))).filter(entry => {
      if (!available.has(entry.winner) || cells + entry.history.length + entry.pv.length > 50000) return false;
      cells += entry.history.length + entry.pv.length;
      return true;
    }).slice(0, 256);
  }
  /** Reachable exact solver premises, including the completed turns implied by lost half-turns. */
  facts(history) {
    const base = history.map(([q, r], i) => `${q},${r},${sideAt(i)}`), found = new Map();
    for (const [key, entry] of this.entries) if (base.every(p => entry.stones.has(p))) {
      const {history, winner, plies, pv} = entry;
      found.set(key, {history, winner, plies, pv});
    }
    if (history.length && history.length % 2 === 0) for (const {action, winner, outcome} of this.edges(history).values()) {
      if (winner === sideAt(history.length)) continue;
      const child = [...history, action], key = proofKey(child), old = found.get(key);
      if (!old || outcome.plies < old.plies) found.set(key, {history: child, ...outcome});
    }
    return [...found.values()].sort((a, b) => a.history.length - b.history.length).slice(0, 2048);
  }
  /** Index the root proof and the leaf continuations retained in `record.proofs`; `id` skips a record already indexed. A
   * proof without `plies` takes the bound its `turns` give (python/play.py proof_plies). */
  add(history, record, id = null) {
    if (!record || id !== null && this.seen.has(id)) return;
    if (id !== null) this.seen.add(id);
    for (const fact of record.proofs || []) this.add(fact.history, {proof: {winner: fact.winner, plies: fact.plies}, pv: fact.pv});
    const proof = record.proof;
    if (!proof) return;
    const key = proofKey(history), prior = this.strategies.get(key);
    const certificate = proof.certificate?.nodes.some(n => n.kind !== 'exact') ? proof.certificate : record.strategy || prior?.certificate;
    this.strategies.set(key, {history, winner: proof.winner, pv: record.pv || [], ...(certificate ? {certificate} : {})});
    if (this.strategies.size > 256) this.strategies.delete(this.strategies.keys().next().value);
    const current = history.map(([q, r]) => [q, r]), pv = record.pv || [];
    const remaining = current.length % 2 ? 2 : 1, mover = sideAt(current.length);
    const plies = proof.plies || remaining + (proof.winner === mover ? 0 : 2) + 4 * (proof.turns - 1);
    // The conservative bound can extend past the winning stone of this line.
    // A finished board is not a future solver premise.
    const end = Math.min(plies, lineLength(current, proof.winner, pv));
    this.put(current, proof.winner, plies, pv);
    for (let i = 0; i < pv.length; i++) {
      const [q, r, side, ply] = pv[i];
      if (pv[i].length !== 4 || ply !== i + 1 || side !== sideAt(current.length) || ply >= end) break;
      current.push([q, r]);
      this.put(current, proof.winner, plies - ply, shifted(pv.slice(i + 1), -ply));
    }
  }
  put(history, winner, plies, pv) {
    const key = proofKey(history), old = this.entries.get(key), witnessed = line => line.every(stone => stone.length === 4) ? line.length : 0;
    const length = lineLength(history, winner, pv);
    if (old && !(old.winner === winner && (plies < old.plies || plies === old.plies &&
        (length < old.length || length === old.length && witnessed(pv) > witnessed(old.pv))))) return;
    this.edgeCache.clear();
    this.knownCache.clear();
    this.choiceCache.clear();
    const stones = new Set(history.map(([q, r], i) => `${q},${r},${sideAt(i)}`));
    this.entries.set(key, {history: history.map(([q, r]) => [q, r]), winner, plies, pv, stones, length});
    if (!this.sizes.has(history.length)) this.sizes.set(history.length, new Set());
    this.sizes.get(history.length).add(key);
  }
  /** Shortest known attack, longest covered defence, within the proof's bound. A subset of defensive replies
   * cannot tighten the universal bound. Stop at omitted defender stones: their position is unknown. */
  line(history, outcome) {
    const current = history.map(p => [...p]);
    let pv = outcome.pv || [];
    for (let i = 0; i <= pv.length; i++) {
      const length = lineLength(current, outcome.winner, pv.slice(i), i);
      const ready = i && this.knownCache.get(proofKey(current));
      if (ready?.winner === outcome.winner && ready.plies + i <= outcome.plies
          && Number.isFinite(lineLength(current, ready.winner, ready.pv))
          && (ready.plies + i < outcome.plies || lineLength(current, ready.winner, ready.pv) <= length)) {
        return [...pv.slice(0, i), ...shifted(ready.pv, i)];
      }
      const entry = this.choice(current);
      const replacement = entry ? lineLength(current, entry.winner, entry.pv) : Infinity;
      if (entry?.winner === outcome.winner && entry.plies + i <= outcome.plies
          && (entry.plies + i < outcome.plies || replacement < length
            || replacement === length && entry.pv.length > pv.length - i)
          && entry.pv.every(p => p.length === 4)) {
        pv = [...pv.slice(0, i), ...shifted(entry.pv, i)];
      }
      if (sideAt(current.length) !== outcome.winner) {
        // A padded turn bound can outlast its drawn line. Resolve each child's
        // current continuation before comparing how long the defences last.
        const replies = [...this.edges(current).values()]
          .filter(e => e.winner === outcome.winner)
          .map(e => ({...e, outcome: this.known([...current, e.action]) || e.outcome}))
          .filter(e => e.outcome.plies + 1 + i <= outcome.plies && e.outcome.pv.length)
          .map(e => ({...e, length: (e.outcome.pv.at(-1)[3] ?? e.outcome.pv.length) + 1}))
          .sort((a, b) => b.length - a.length || b.outcome.plies - a.outcome.plies || a.action[0] - b.action[0] || a.action[1] - b.action[1]);
        if (replies.length) {
          const reply = replies[0], first = pv[i]?.slice(0, 2).join(',');
          if (first !== reply.action.join(',') && !replies.some(e => e.length === reply.length && e.outcome.plies === reply.outcome.plies && e.action.join(',') === first)) {
            pv = [...pv.slice(0, i), [...reply.action, sideAt(current.length), i + 1], ...shifted(reply.outcome.pv, i + 1)];
          }
        }
      }
      const stone = pv[i];
      if (!stone || stone.length !== 4 || stone[3] !== i + 1 || stone[2] !== sideAt(current.length)) break;
      current.push(stone.slice(0, 2));
    }
    return pv;
  }
  /** Map 'q,r' -> {action, winner, distance, outcome} for each stone from `history` whose position is proven: in the
   * table, or because one more stone by that position's mover reaches a position the mover wins. `distance` counts
   * the stone itself; `outcome` is the position's {winner, plies, pv}. */
  edges(history) {
    const at = proofKey(history);
    if (this.edgeCache.has(at)) return this.edgeCache.get(at);
    const size = history.length, base = history.map(([q, r], i) => `${q},${r},${sideAt(i)}`), own = new Set(base), found = new Map();
    // A lost half-turn after A covers A,B in either order. Its saved response
    // need not be B, so carry the verdict without inventing a new PV.
    // B may be earlier in the supplied history: only board and phase matter.
    if (size && size % 2 === 0) for (const key of this.sizes.get(size) || []) {
      const entry = this.entries.get(key), mover = sideAt(size);
      if (entry.winner === mover || entry.plies < 2) continue;
      const missing = [...entry.stones].filter(p => !own.has(p)), replaced = base.filter(p => !entry.stones.has(p));
      if (missing.length !== 1 || replaced.length !== 1 || Number(replaced[0].split(',')[2]) !== mover) continue;
      const [q, r, side] = missing[0].split(',').map(Number);
      if (side !== mover) continue;
      const outcome = {winner: entry.winner, plies: entry.plies - 1, pv: []};
      found.set(`${q},${r}`, {action: [q, r], winner: entry.winner, distance: entry.plies, outcome});
    }
    for (const extra of [1, 2]) for (const key of this.sizes.get(size + extra) || []) {
      const entry = this.entries.get(key);
      if (!base.every(stone => entry.stones.has(stone))) continue;
      const stones = [...entry.stones].filter(stone => !own.has(stone)).map(stone => stone.split(',').map(Number));
      for (const [first, second] of extra === 1 ? [[stones[0], null]] : [stones, [stones[1], stones[0]]]) {
        if (first[2] !== sideAt(size)) continue;
        let outcome;
        if (!second) outcome = {winner: entry.winner, plies: entry.plies, pv: entry.pv};
        else if (second[2] === sideAt(size + 1) && entry.winner === second[2]) {
          outcome = {winner: entry.winner, plies: entry.plies + 1, pv: [[second[0], second[1], entry.winner, 1], ...shifted(entry.pv, 1)]};
        } else continue;
        const action = `${first[0]},${first[1]}`, old = found.get(action);
        if (!old || old.outcome.plies > outcome.plies) found.set(action, {action: [first[0], first[1]], winner: outcome.winner, distance: outcome.plies + 1, outcome});
      }
    }
    this.edgeCache.set(at, found);
    if (this.edgeCache.size > 4096) this.edgeCache.delete(this.edgeCache.keys().next().value);
    return found;
  }
  /** The tightest stored guarantee, also considering shorter winning continuations. */
  choice(history) {
    const at = proofKey(history);
    if (this.choiceCache.has(at)) return this.choiceCache.get(at);
    const found = this.choose(history);
    this.choiceCache.set(at, found);
    if (this.choiceCache.size > 4096) this.choiceCache.delete(this.choiceCache.keys().next().value);
    return found;
  }
  choose(history) {
    let own = this.entries.get(proofKey(history));
    if (!own && history.length > 1 && history.length % 2 === 1) {
      const mover = sideAt(history.length), base = new Set(history.map(([q, r], i) => `${q},${r},${sideAt(i)}`));
      for (const key of this.sizes.get(history.length - 1) || []) {
        const loss = this.entries.get(key);
        if (loss.winner !== mover || loss.plies < 2 || ![...loss.stones].every(p => base.has(p))) continue;
        if (![...base].some(p => !loss.stones.has(p) && Number(p.split(',')[2]) === 1-mover)) continue;
        if (!own || own.plies > loss.plies - 1) own = {winner: mover, plies: loss.plies - 1, pv: []};
      }
    }
    const mover = sideAt(history.length);
    if (own && own.winner !== mover) return own;
    const wins = [...this.edges(history).values()].filter(e => e.winner === mover).map(e => {
      const child = this.known([...history, e.action]) || e.outcome;
      const pv = [[...e.action, mover, 1], ...shifted(child.pv, 1)];
      return {winner: mover, plies: child.plies + 1, pv, length: lineLength(history, mover, pv)};
    }).sort((a, b) => a.plies - b.plies || a.length - b.length || a.pv[0][0] - b.pv[0][0] || a.pv[0][1] - b.pv[0][1]);
    if (!wins.length || own && (own.plies < wins[0].plies || own.plies === wins[0].plies && own.pv.length
        && lineLength(history, mover, own.pv) <= wins[0].length)) return own || null;
    return wins[0];
  }
  /** Best known guarantee with its updated continuation, or null when nothing is proven. */
  known(history) {
    const at = proofKey(history);
    if (this.knownCache.has(at)) return this.knownCache.get(at);
    const found = this.choice(history);
    const outcome = found ? {winner: found.winner, plies: found.plies, pv: this.line(history, found)} : null;
    this.knownCache.set(at, outcome);
    if (this.knownCache.size > 4096) this.knownCache.delete(this.knownCache.keys().next().value);
    return outcome;
  }
}

/** The turn the table `known` gives at `history` without a search (python/play.py answered): when it proves a win for
 * the side to move whose line holds the rest of the turn, its stones, value 1, the winning stone as the only top row,
 * the proof and the line; else null. */
export function answered(native, history, known) {
  const outcome = known?.known(history), {player, remaining} = native.game(history);
  if (!outcome || outcome.winner !== player) return null;
  const moves = [];
  for (const stone of outcome.pv.slice(0, remaining)) {
    if (stone.length !== 4 || stone[2] !== player || stone[3] !== moves.length + 1) break;
    moves.push([stone[0], stone[1]]);
  }
  if (!moves.length || moves.length < remaining && native.game([...history, ...moves]).winner !== player) return null;
  return {moves, value: 1, top: [[...moves[0], 1, 1, 1]], proof: {winner: player, turns: proofTurns(outcome.plies, remaining, true), plies: outcome.plies},
    pv: outcome.pv, threat: [], solved: true, actual_completed: 0, actual_solver_nodes: 0};
}

/** A search `result` (search.mjs NeuralSearch.result) of the side `mover` with the stones `edges` (Proofs.edges)
 * proves settled: their values and completed Q become 1 or -1, a proven win is the choice (the shortest, unless the
 * search proved a win at most as long) and proves the position, a proven loss leaves the policy and the choice while a stone remains that is not proven lost, and when every stone is
 * proven lost the position is lost and the choice is the loss that lasts longest. */
export function settled(result, edges, mover) {
  if (!edges.size) return result;
  const values = [...result.values], completed_q = [...result.completed_q], policy = [...result.policy];
  let win = null;
  result.actions.forEach(([q, r], i) => {
    const edge = edges.get(`${q},${r}`);
    if (!edge) return;
    values[i] = completed_q[i] = edge.winner === mover ? 1 : -1;
    if (edge.winner !== mover) policy[i] = 0;
    else if (!win || edge.distance < win.distance) win = edge;
  });
  if (win && result.proven > 0 && result.proof_plies <= win.distance) return {...result, values, completed_q};
  if (win) return {...result, values, completed_q, action: win.action, proven: 1, exact_winner: mover, proof_plies: win.distance};
  const total = policy.reduce((a, b) => a + b, 0);
  if (!total && result.actions.every(([q, r]) => edges.has(`${q},${r}`))) {
    const longest = result.actions.map(([q, r]) => edges.get(`${q},${r}`)).reduce((a, b) => b.distance > a.distance ? b : a);
    return {...result, values, completed_q, policy, action: longest.action, proven: -1, exact_winner: 1 - mover, proof_plies: longest.distance};
  }
  if (!total) return {...result, values, completed_q};
  const shares = policy.map(p => p / total), lost = edges.get(result.action?.join(','));
  return {...result, values, completed_q, policy: shares, action: lost ? result.actions[shares.indexOf(Math.max(...shares))] : result.action};
}

/** `found` (an evaluation, or null) with what the table `known` proves of `history` (python/play.py Session.proven):
 * without its own proof, the position's proof, value and line; an existing proof's line extended from stored children;
 * each stone to a proven position as a top row marked
 * won or lost, proven wins first, the shortest leading and among equals the stone `played` next in the game, losses
 * last. A proven position without an evaluation gets one with no simulations; null when there is neither. `remaining`
 * is the mover's stones left in the turn. */
export function proven(known, history, found, remaining, played = null) {
  const outcome = known.known(history), edges = known.edges(history);
  if (!found && !outcome) return null;
  const mover = sideAt(history.length), shown = {...(found || {moves: [], top: [], threat: [], simulations: 0, solver_nodes: 0})};
  const oldPlies = shown.proof && (shown.proof.plies || remaining + (shown.proof.winner === mover ? 0 : 2) + 4 * (shown.proof.turns - 1));
  if (outcome && (!shown.proof || outcome.winner === shown.proof.winner && outcome.plies < oldPlies)) {
    const won = outcome.winner === mover;
    Object.assign(shown, {value: won ? 1 : 0, pv: outcome.pv,
      proof: {winner: outcome.winner, turns: proofTurns(outcome.plies, remaining, won), plies: outcome.plies}});
    if (won) {
      const moves = outcome.pv.slice(0, remaining).filter((p, i) => p[2] === mover && p[3] === i + 1).map(p => [p[0], p[1]]);
      shown.moves = moves.length === remaining || outcome.plies <= moves.length ? moves : [];
    }
  }
  if (shown.proof) {
    const {winner, turns} = shown.proof, plies = shown.proof.plies || remaining + (winner === mover ? 0 : 2) + 4 * (turns - 1);
    shown.pv = known.line(history, {winner, plies, pv: shown.pv || []});
  }
  const lost = shown.proof && shown.proof.winner !== mover;
  const turn = shown.proof ? (shown.pv || []).slice(0, remaining)
    .filter((p, i) => p.length === 4 && p[2] === mover && p[3] === i + 1).map(p => p.slice(0, 2)) : [];
  if (turn.length === remaining) shown.moves = turn;
  const defence = lost ? turn : [];
  const rows = (shown.top || []).map(row => [...row]);
  for (const {action, winner} of edges.values()) {
    let row = rows.find(r => r[0] === action[0] && r[1] === action[1]);
    if (!row && winner === mover) rows.push(row = [...action, 0]);
    if (row) row.splice(3, 2, winner === mover ? 1 : 0, winner === mover ? 1 : -1);
  }
  if (lost) {
    for (const row of rows) row.splice(3, 2, 0, -1);
    if (defence.length) {
      const at = rows.findIndex(r => r[0] === defence[0][0] && r[1] === defence[0][1]);
      rows.unshift(at < 0 ? [...defence[0], 0, 0, -1] : rows.splice(at, 1)[0]);
    }
  }
  const flag = row => row[4] ?? 0, distance = row => flag(row) > 0 ? edges.get(`${row[0]},${row[1]}`)?.distance ?? Infinity : 0;
  const other = row => flag(row) > 0 && played !== null && (row[0] !== played[0] || row[1] !== played[1]) ? 1 : 0;
  shown.top = rows.map((row, i) => [row, i]).sort(([a, i], [b, j]) => (1 - flag(a)) - (1 - flag(b)) || distance(a) - distance(b) || other(a) - other(b) || i - j)
    .map(([row]) => row).slice(0, 5);
  shown.refuted = !shown.proof && shown.top.length && shown.top.every(row => flag(row) < 0) ? shown.top.length : 0;
  if (shown.refuted && shown.node_value != null) shown.value = shown.node_value;
  return shown;
}
