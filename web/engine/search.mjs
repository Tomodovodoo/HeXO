/* Native Gumbel search (gumbel.wasm, the hxg_* ABI of src/gumbel.cpp) driven like python/neural_search.py. */

export class Native {
  /** Wraps an instantiated gumbel.mjs module. */
  constructor(module) {
    this.m = module;
  }
  view(type, pointer, length) {
    return new type(this.m.HEAPU8.buffer, pointer, length);
  }
  alloc(bytes) {
    return this.m._malloc(Math.max(8, bytes));
  }
  error() {
    return this.m.UTF8ToString(this.m._hxg_error());
  }
  checked(ok) {
    if (!ok) throw new Error(this.error());
  }
  /** Copies pairs [[q, r], ...] into a new int64 buffer; the caller frees it. */
  cells(points) {
    const pointer = this.alloc(16 * points.length), out = this.view(BigInt64Array, pointer, 2 * points.length);
    points.forEach(([q, r], i) => { out[2 * i] = BigInt(q); out[2 * i + 1] = BigInt(r); });
    return pointer;
  }
  /** Reads `count` int64 pairs at `pointer` as [[q, r], ...]. */
  pairs(pointer, count) {
    const data = this.view(BigInt64Array, pointer, 2 * count), out = new Array(count);
    for (let i = 0; i < count; i++) out[i] = [Number(data[2 * i]), Number(data[2 * i + 1])];
    return out;
  }
  board(history, read) {
    const board = this.m._hx_new();
    try {
      for (const [q, r] of history) if (!this.m._hx_play(board, BigInt(q), BigInt(r))) throw new Error(`Illegal placement: ${q}, ${r}`);
      return read(board);
    } finally {
      this.m._hx_free(board);
    }
  }
  /** {winner, player, remaining} of a placement history, from the engine's own rules. */
  game(history) {
    return this.board(history, b => ({winner: this.m._hx_winner(b), player: this.m._hx_player(b), remaining: this.m._hx_remaining(b)}));
  }
  /** Legal moves of a placement history in native (sorted) order. */
  legal(history) {
    return this.board(history, b => {
      const count = this.m._hx_moves(b, 0, 0), pointer = this.alloc(24 * count);
      try {
        this.m._hx_moves(b, pointer, count);
        const data = this.view(BigInt64Array, pointer, 3 * count), out = new Array(count);
        for (let i = 0; i < count; i++) out[i] = [Number(data[3 * i]), Number(data[3 * i + 1])];
        return out;
      } finally {
        this.m._free(pointer);
      }
    });
  }
}

/** Evaluation cache keyed like neural_search.EvaluationCache: colored stones, turn context and model version. */
export class EvaluationCache {
  constructor(capacity = 4096) {
    this.capacity = capacity;
    this.entries = new Map();
  }
  key(history, version) {
    const size = history.length, start = size % 2 || size === 0 ? size : size - 1;
    const stones = history.map(([q, r], i) => [q, r, ((i + 1) >> 1) % 2]).sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
    const previous = history.slice(Math.max(0, start - 2), start).map(p => [...p]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    return JSON.stringify([version, ((size + 1) >> 1) % 2, size % 2 ? 2 : 1, stones, history.slice(start, size), previous]);
  }
  get(key) {
    const value = this.entries.get(key);
    if (value !== undefined) { this.entries.delete(key); this.entries.set(key, value); }
    return value;
  }
  put(key, value) {
    this.entries.delete(key);
    this.entries.set(key, value);
    while (this.entries.size > this.capacity) this.entries.delete(this.entries.keys().next().value);
  }
}

/**
 * One persistent search tree. `evaluate(requests)` takes [{history, actions}] (actions in native legal order) and
 * resolves to [{logits, q}] per request (array-likes of the action count; q is V(s) broadcast).
 */
export class NeuralSearch {
  constructor(native, {seed = 1740, tactics = false, graph = false, history = []} = {}) {
    this.n = native;
    this.m = native.m;
    this.ptr = this.m._hxg_new(BigInt(seed));
    if (!this.ptr) throw new Error('Native tree allocation failed');
    this.history = [];
    native.checked(this.m._hxg_tactics(this.ptr, tactics ? 1 : 0));
    native.checked(this.m._hxg_graph(this.ptr, graph ? 1 : 0));
    for (const point of history) this.advance(point);
  }
  close() {
    if (this.ptr) this.m._hxg_free(this.ptr);
    this.ptr = 0;
  }
  advance([q, r]) {
    this.n.checked(this.m._hxg_advance(this.ptr, BigInt(q), BigInt(r)));
    this.history.push([q, r]);
  }
  /** [id, {history, actions}] of the next leaf; id 0 (nothing now) or -1 (an exact simulation) carries none. */
  request() {
    const id = this.m._hxg_next(this.ptr);
    if (id === -2) this.n.checked(0);
    if (id <= 0) return [id, null];
    const size = this.m._hxg_history(this.ptr, id, 0), count = this.m._hxg_legal(this.ptr, id, 0);
    const pointer = this.n.alloc(16 * Math.max(size, count));
    try {
      this.m._hxg_history(this.ptr, id, pointer);
      const history = this.n.pairs(pointer, size);
      this.m._hxg_legal(this.ptr, id, pointer);
      return [id, {history, actions: this.n.pairs(pointer, count)}];
    } finally {
      this.m._free(pointer);
    }
  }
  fulfill(id, actions, prediction) {
    const n = actions.length, a = this.n.cells(actions), l = this.n.alloc(8 * n), q = this.n.alloc(8 * n);
    try {
      this.n.view(Float64Array, l, n).set(prediction.logits);
      this.n.view(Float64Array, q, n).set(prediction.q);
      this.n.checked(this.m._hxg_fulfill(this.ptr, id, a, l, q, n));
    } finally {
      this.m._free(a); this.m._free(l); this.m._free(q);
    }
  }
  /**
   * Runs one search like SearchCoordinator.search_many for a single tree; `stop()` true cancels it (the result is
   * then the partial search). Resolves to the result fields of NeuralSearch.result.
   */
  async search({simulations = 128, rootSamples = null, batchSize = 16, evaluate, cache = new EvaluationCache(), version = 'web',
    stop = () => false, onBatch = () => {}, choice = 'policy'} = {}) {
    if (batchSize < 1 || simulations < 1) throw new Error('Positive search budgets required');
    if (choice !== 'policy' && choice !== 'gumbel') throw new Error('choice must be policy or gumbel');
    const start = performance.now(), stats = {evaluated: 0, hits: 0, batches: 0, largest: 0, network_ms: 0};
    const sample = rootSamples ?? Math.max(2, Math.floor(Math.sqrt(simulations)));
    this.n.checked(this.m._hxg_begin(this.ptr, simulations, sample));
    let active = this.n.game(this.history).winner < 0, stopped = false;
    const finished = () => {
      const expired = stop();
      if (this.m._hxg_done(this.ptr) || expired) {
        active = false;
        if (expired) { stopped = true; this.m._hxg_cancel(this.ptr); }
        return true;
      }
      return false;
    };
    try {
      while (active) {
        let pending = [], idle = 0;
        while (active && pending.length < batchSize) {
          if (finished()) idle = 0;
          else {
            const [id, leaf] = this.request();
            if (id === -1) idle = 0;
            else if (id === 0) idle += 1;
            else {
              idle = 0;
              const key = cache.key(leaf.history, version), cached = cache.get(key);
              if (cached === undefined) pending.push({id, leaf, key});
              else { this.fulfill(id, leaf.actions, cached); stats.hits++; }
            }
          }
          if (idle >= 1) break;
        }
        if (active) finished();
        if (!active) pending = [];
        if (pending.length) {
          const groups = new Map();
          for (const item of pending) {
            if (!groups.has(item.key)) groups.set(item.key, []);
            groups.get(item.key).push(item);
          }
          const unique = [...groups.values()];
          const asked = performance.now(), predictions = await evaluate(unique.map(items => items[0].leaf));
          stats.network_ms += performance.now() - asked;
          stats.batches++;
          stats.largest = Math.max(stats.largest, unique.length);
          if (predictions.length !== unique.length) throw new Error('Evaluator returned the wrong batch size');
          unique.forEach((items, i) => {
            for (const item of items) { this.fulfill(item.id, item.leaf.actions, predictions[i]); stats.evaluated++; }
            cache.put(items[0].key, {logits: Float64Array.from(predictions[i].logits), q: Float64Array.from(predictions[i].q)});
          });
          onBatch(stats);
        } else if (active) {
          finished();
          if (active && idle >= 1) throw new Error('Native scheduler stalled without pending evaluations');
        }
      }
    } finally {
      if (this.ptr) this.m._hxg_cancel(this.ptr);
    }
    return {...this.result(choice), evaluated: stats.evaluated, cache_hits: stats.hits, inference_batches: stats.batches,
      largest_batch: stats.largest, network_ms: stats.network_ms, stopped, elapsed_ms: performance.now() - start};
  }
  /** Root statistics as NeuralSearch.result: action, actions, visits, values, policy, scores, exact fields. */
  result(choice = 'gumbel') {
    const m = this.m, n = m._hxg_stats(this.ptr, 0, 0, 0, 0);
    const a = this.n.alloc(16 * n), v = this.n.alloc(4 * n), q = this.n.alloc(8 * n), s = this.n.alloc(8 * n), p = this.n.alloc(8 * n);
    try {
      m._hxg_stats(this.ptr, a, v, q, s);
      m._hxg_policy(this.ptr, p);
      const actions = this.n.pairs(a, n), visits = Array.from(this.n.view(Int32Array, v, n));
      const values = Array.from(this.n.view(Float64Array, q, n)), scores = Array.from(this.n.view(Float64Array, s, n));
      const policy = Array.from(this.n.view(Float64Array, p, n));
      let selected = -1;
      scores.forEach((score, i) => { if (Number.isFinite(score) && (selected < 0 || score > scores[selected])) selected = i; });
      const winner = m._hxg_exact(this.ptr), mover = ((this.history.length + 1) >> 1) % 2;
      const proven = winner < 0 ? 0 : winner === mover ? 1 : -1;
      if (choice === 'policy' && !proven && policy.some(p => p > 0)) {
        selected = 0;
        policy.forEach((p, i) => { if (p > policy[selected]) selected = i; });
      }
      return {action: selected >= 0 ? actions[selected] : null, actions, visits, values, policy, scores,
        completed: m._hxg_completed(this.ptr), exact_winner: winner, proven,
        proof_plies: proven ? m._hxg_distance(this.ptr) : 0,
        proof_action: proven > 0 ? actions.filter((_, i) => Number.isFinite(scores[i])) : []};
    } finally {
      for (const pointer of [a, v, q, s, p]) m._free(pointer);
    }
  }
}
