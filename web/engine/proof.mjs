/* Walking a verified solver certificate: python/dense_solver.py Proof.walk and python/play.py winning_line. */

const same = (a, b) => a[0] === b[0] && a[1] === b[1];
const has = (list, point) => list.some(p => same(p, point));
const sorted = cells => cells.map(p => [...p]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);

export class Proof {
  /** A certificate of a forced win for the side to move at `base` ([[q, r], ...]). */
  constructor(base, certificate) {
    this.base = base;
    this.nodes = certificate.nodes;
    this.root = certificate.root;
    this.depth = new Map();
    this.replies = new Map();
    this.nodes.forEach((node, i) => {
      if (node.kind === 'defender_replies') this.replies.set(i, new Map(node.responses.map(r => [JSON.stringify(sorted(r.action)), r.child])));
    });
  }
  turns(index) {
    if (!this.depth.has(index)) {
      const node = this.nodes[index];
      this.depth.set(index, node.kind === 'attacker_move' ? 1 + this.turns(node.child)
        : this.replies.has(index) ? Math.max(...[...this.replies.get(index).values()].map(c => this.turns(c))) : 1);
    }
    return this.depth.get(index);
  }
  /** [move, node, played]: move is [stones still to play this attacker turn, turns] or null off the certificate. */
  walk(history) {
    let i = this.base.length;
    if (history.length < i || this.base.some((p, j) => !same(p, history[j]))) return [null, null, []];
    let index = this.root;
    for (;;) {
      let node = this.nodes[index], turns = this.turns(index);
      if (node.kind === 'defender_replies' || node.kind === 'unstoppable') {
        const reply = history.slice(i, i + 2);
        if (reply.length < 2) return [[[], turns], node, reply];
        i += 2;
        if (node.kind === 'unstoppable') {
          const threat = node.threats.find(t => !t.some(c => has(reply, c)));
          if (!threat) return [null, null, []];
          node = {kind: 'immediate_win', action: threat};
          turns = 1;
        } else {
          index = this.replies.get(index).get(JSON.stringify(sorted(reply)));
          if (index === undefined) return [null, null, []];
          continue;
        }
      }
      const action = node.action, played = history.slice(i, i + action.length);
      if (played.length < action.length) {
        if (!played.every(p => has(action, p))) return [null, null, []];
        return [[action.filter(a => !has(played, a)), turns], node, played];
      }
      if (!played.every(p => has(action, p)) || node.kind === 'immediate_win') return [null, null, []];
      index = node.child;
      i += action.length;
    }
  }
  /** Defender stones of the first covered reply extending this turn, or null. */
  reply(history) {
    const [move, node, played] = this.walk(history);
    if (!move || move[0].length || node.kind !== 'defender_replies') return null;
    for (const response of node.responses) {
      if (played.every(p => has(response.action, p))) return response.action.filter(a => !has(played, a));
    }
    return null;
  }
}

/** One legal continuation of `certificate` from `history` as [q, r, player] stones (play.winning_line). */
export function winningLine(native, history, certificate) {
  const proof = new Proof(history, certificate), current = history.map(p => [...p]), line = [];
  for (let state = native.game(current); state.winner < 0; state = native.game(current)) {
    const move = proof.walk(current)[0];
    if (!move) break;
    const actions = move[0].length ? move[0] : proof.reply(current) || native.legal(current).slice(0, state.remaining);
    for (const action of actions) {
      const player = native.game(current).player;
      line.push([action[0], action[1], player]);
      current.push([action[0], action[1]]);
      if (native.game(current).winner >= 0) break;
    }
  }
  return line;
}
