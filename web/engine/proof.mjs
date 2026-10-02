/* Rows of a browser analysis record: the candidate rows (python/play.py top_rows) and the principal variation of a
 * verified solver certificate (python/play.py principal_variation). */

/** Most attacker turns on any path of `certificate` from each node, the completing turn included. */
function depths(certificate) {
  const memo = new Map();
  const turns = index => {
    if (!memo.has(index)) {
      const node = certificate.nodes[index];
      memo.set(index, node.kind === 'attacker_move' ? 1 + turns(node.child)
        : node.kind === 'defender_replies' ? Math.max(...node.responses.map(r => turns(r.child))) : 1);
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
export function principalVariation(native, history, certificate) {
  const turns = depths(certificate), current = history.map(p => [...p]), pv = [];
  const near = reply => reply.action.reduce((sum, cell) => sum + distance(cell, pv.at(-1)), 0);
  const attacker = native.game(current).player;
  let plies = 0, index = certificate.root;
  while (native.game(current).winner < 0) {
    const node = certificate.nodes[index];
    if (node.kind === 'unstoppable') {
      if (node.threats?.length) {
        const threat = node.threats.reduce((a, b) => b.length < a.length ? b : a);
        threat.forEach(([q, r], i) => pv.push([q, r, attacker, plies + 3 + i]));
        plies += 2 + threat.length;
      }
      break;
    }
    let action;
    if (node.kind === 'defender_replies') {
      const longer = (a, b) => turns(b.child) - turns(a.child) || near(a) - near(b);
      const reply = node.responses.reduce((a, b) => longer(a, b) > 0 ? b : a);
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
