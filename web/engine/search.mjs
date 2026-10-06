/* Native Gumbel search (gumbel.wasm, the hxg_* ABI of src/gumbel.cpp) driven like python/neural_search.py. */
import {nextTask} from './tasks.mjs';

export const GRAPH_LIMIT = 4096;  // expanded nodes a GameGraph keeps between searches (neural_search.GRAPH_LIMIT)
export const PV_DROP = .05;       // completed-Q fall that sends a checked search back (neural_search.PV_DROP)
export const PV_CHECK = .25;      // principal-variation check share of play and analysis searches (play.PV_CHECK)

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
 * resolves to [{logits, q}] per request (array-likes of the action count; q is V(s) broadcast). `qRangeFloor` is
 * the least Q range of the completed-Q rescale (0 keeps mctx's) and `rootNoise` the uniform share of the root's
 * candidate sampling (0 samples by the prior), as python/neural_search.py's q_range_floor and root_noise. `limit`,
 * when given, makes the tree a shared game graph keeping at most that many expanded nodes (see GameGraph).
 * `archiveBytes` optionally retains up to 256 dormant expansions under a payload/index allowance of at least 64 KiB;
 * active nodes and allocator residency are separate. `archiveForward` releases permanent colour conflicts with
 * the primary played board at safe owner points; leave it false to retain analysis for undo.
 */
export class NeuralSearch {
  constructor(native, {seed = 1740, tactics = false, graph = false, qRangeFloor = 0, rootNoise = 0, history = [], limit = null, archiveBytes = 0, archiveForward = false, roundBarrier = false} = {}) {
    this.n = native;
    this.m = native.m;
    this.ptr = this.m._hxg_new(BigInt(seed));
    if (!this.ptr) throw new Error('Native tree allocation failed');
    this.history = [];
    native.checked(this.m._hxg_tactics(this.ptr, tactics ? 1 : 0));
    native.checked(this.m._hxg_graph(this.ptr, graph ? 1 : 0));
    if (limit !== null) native.checked(this.m._hxg_share(this.ptr, BigInt(limit)));
    if (archiveBytes) native.checked(this.m._hxg_archive(this.ptr, BigInt(archiveBytes)));
    if (archiveForward) native.checked(this.m._hxg_archive_forward(this.ptr, 1));
    native.checked(this.m._hxg_q_range_floor(this.ptr, qRangeFloor));
    native.checked(this.m._hxg_root_noise(this.ptr, rootNoise));
    native.checked(this.m._hxg_round_barrier(this.ptr, roundBarrier ? 1 : 0));
    for (const point of history) this.advance(point);
  }
  close() {
    if (this.nativeOwner) throw new Error('Close the native owner before its graph');
    if (this.ptr) this.m._hxg_free(this.ptr);
    this.ptr = 0;
  }
  advance([q, r]) {
    if (this.nativeOwner) throw new Error('Close the native owner before advancing its graph');
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
  /** Installs a native-verified mover certificate at its pending leaf before exact backup. */
  fulfillProof(id, leaf, proof) {
    if (proof.attacker !== 'mover') throw new Error('A leaf proof must belong to the side to move');
    const state = this.n.game(leaf.history), history = this.n.cells(leaf.history), moves = this.n.cells(proof.moves);
    try {
      this.n.checked(this.m._hxg_prove(this.ptr, id, history, leaf.history.length, state.player, state.remaining,
        moves, proof.moves.length, proof.proof_turns));
    } finally {
      this.m._free(history); this.m._free(moves);
    }
  }
  /**
   * Settles the root's stones `edges` (proof.mjs Proofs.edges) before a search, as python/play.py TurnSearch.request:
   * evaluates the root through `cache` (else `evaluate`, as in search) when it has no edges yet, then marks each edge
   * exact for its winner within its distance, the stone itself included (hxg_mark_exact): the mover's losses first,
   * then its wins from the shortest. Later marks can tighten an already proven root and its stored parents.
   * Resolves to the edges the tree did not take, keyed as `edges`: those that are not root edges.
   */
  async settle(edges, {evaluate, cache = new EvaluationCache(), version = 'web'}) {
    const unmarked = new Map(edges);
    if (!edges.size) return unmarked;
    if (!this.m._hxg_stats(this.ptr, 0, 0, 0, 0)) {
      this.n.checked(this.m._hxg_begin(this.ptr, 1, 1));
      try {
        const [id, leaf] = this.request();
        if (id > 0) {
          const key = cache.key(leaf.history, version);
          let prediction = cache.get(key);
          if (prediction === undefined) {
            const [found] = await evaluate([leaf]);
            prediction = {logits: Float64Array.from(found.logits), q: Float64Array.from(found.q)};
            cache.put(key, prediction);
          }
          this.fulfill(id, leaf.actions, prediction);
        }
      } finally {
        this.m._hxg_cancel(this.ptr);
      }
    }
    const mover = ((this.history.length + 1) >> 1) % 2, won = edge => edge.winner === mover ? 1 : 0;
    const order = [...edges].sort(([, a], [, b]) => won(a) - won(b) || a.distance - b.distance || a.action[0] - b.action[0] || a.action[1] - b.action[1]);
    for (const [key, {action: [q, r], winner, distance}] of order) {
      if (this.m._hxg_mark_exact(this.ptr, BigInt(q), BigInt(r), winner, distance)) unmarked.delete(key);
    }
    return unmarked;
  }
  /**
   * Runs one search like SearchCoordinator.search_many for a single tree; `stop()` true cancels it (the result is
   * then the partial search). `prove(history)`, when supplied, returns a native-verified mover certificate or
   * UNKNOWN before cache lookup and neural evaluation. Resolves to the result fields of NeuralSearch.result.
   */
  async search({simulations = 128, rootSamples = null, batchSize = 16, evaluate, cache = new EvaluationCache(), version = 'web',
    stop = () => false, onBatch = () => {}, choice = 'policy', prove = null} = {}) {
    if (this.nativeOwner) throw new Error('Graph already has a native owner');
    if (batchSize < 1 || simulations < 1) throw new Error('Positive search budgets required');
    if (choice !== 'policy' && choice !== 'gumbel') throw new Error('choice must be policy or gumbel');
    const start = performance.now(), stats = {evaluated: 0, hits: 0, batches: 0, largest: 0, network_ms: 0};
    const sample = rootSamples ?? Math.max(2, Math.floor(Math.sqrt(simulations)));
    this.n.checked(this.m._hxg_begin(this.ptr, simulations, sample));
    let active = this.n.game(this.history).winner < 0, stopped = false, yielded = start;
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
          // Cached and exact leaves need no inference await, but must still receive cancellation messages.
          if (performance.now() - yielded >= 8) {
            onBatch({...stats, completed: this.m._hxg_completed(this.ptr)});
            await nextTask();
            yielded = performance.now();
          }
          if (finished()) idle = 0;
          else {
            const [id, leaf] = this.request();
            if (id === -1) idle = 0;
            else if (id === 0) idle += 1;
            else {
              idle = 0;
              if (prove) {
                const proof = await prove(leaf.history);
                if (finished()) continue;
                if (proof?.status === 'PROVEN_WIN' && proof.native_verified) {
                  this.fulfillProof(id, leaf, proof);
                  continue;
                }
              }
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
          onBatch({...stats, completed: this.m._hxg_completed(this.ptr)});
          await nextTask();
          yielded = performance.now();
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
  /** Root statistics as NeuralSearch.result: action, actions, visits, values, completed_q, policy, scores, exact fields. */
  facts() {
    const capacity = 100000, buffer = this.n.alloc(8 * capacity);
    try {
      const used = this.m._hxg_facts(this.ptr, buffer, capacity), words = this.n.view(BigInt64Array, buffer, used), facts = [];
      for (let i = 0; i < used && facts.length < 2048;) {
        const count = Number(words[i++]), winner = Number(words[i++]), plies = Number(words[i++]), history = [];
        for (let j = 0; j < count; j++) history.push([Number(words[i++]), Number(words[i++])]);
        facts.push({history, winner, plies});
      }
      return facts;
    } finally { this.m._free(buffer); }
  }
  proveLoss(winner, plies) { this.n.checked(this.m._hxg_prove_loss(this.ptr, winner, plies)); }

  result(choice = 'gumbel') {
    const m = this.m, n = m._hxg_stats(this.ptr, 0, 0, 0, 0);
    const a = this.n.alloc(16 * n), v = this.n.alloc(4 * n), q = this.n.alloc(8 * n), s = this.n.alloc(8 * n), p = this.n.alloc(8 * n);
    const c = this.n.alloc(8 * n);
    try {
      m._hxg_stats(this.ptr, a, v, q, s);
      m._hxg_policy(this.ptr, p);
      m._hxg_q(this.ptr, c);
      const actions = this.n.pairs(a, n), visits = Array.from(this.n.view(Int32Array, v, n));
      const values = Array.from(this.n.view(Float64Array, q, n)), scores = Array.from(this.n.view(Float64Array, s, n));
      const policy = Array.from(this.n.view(Float64Array, p, n)), completed_q = Array.from(this.n.view(Float64Array, c, n));
      let selected = -1;
      scores.forEach((score, i) => { if (Number.isFinite(score) && (selected < 0 || score > scores[selected])) selected = i; });
      const winner = m._hxg_exact(this.ptr), mover = ((this.history.length + 1) >> 1) % 2;
      const proven = winner < 0 ? 0 : winner === mover ? 1 : -1;
      if (choice === 'policy' && !proven && policy.some(p => p > 0)) {
        selected = 0;
        policy.forEach((p, i) => { if (p > policy[selected]) selected = i; });
      }
      return {action: selected >= 0 ? actions[selected] : null, actions, visits, values, completed_q, policy, scores,
        completed: m._hxg_completed(this.ptr), exact_winner: winner, proven, node_value: m._hxg_value(this.ptr),
        proof_plies: proven ? m._hxg_distance(this.ptr) : 0,
        proof_action: proven > 0 ? actions.filter((_, i) => Number.isFinite(scores[i])) : []};
    } finally {
      for (const pointer of [a, v, q, s, p, c]) m._free(pointer);
    }
  }
}

/**
 * One game's shared search graph (python/neural_search.py GameGraph): the store keeps every node the game's searches
 * expanded, a search reads each stored child's visits and value as its edge's, and `at(history)` moves the root to any
 * position. `search({..., pvCheck})` adds the principal-variation check (neural_search.Recheck).
 */
export class GameGraph extends NeuralSearch {
  /** `id` is unique to this graph, across reloads, so a session can tell a rebuilt graph from the one it searched. */
  constructor(native, options = {}) {
    super(native, {...options, limit: options.limit ?? GRAPH_LIMIT});
    this.id = globalThis.crypto.randomUUID();
  }
  /** Independent sampling into the same game store. All views belong to the same graph-owner worker. */
  view(history = this.history, seed = 1740) {
    if (this.nativeOwner) throw new Error('Graph already has a native owner');
    if (!this.ptr) throw new Error('Graph is closed');
    const cells = this.n.cells(history);
    try {
      const ptr = this.m._hxg_view(this.ptr, cells, history.length, BigInt(seed));
      this.n.checked(ptr);
      const view = Object.create(GameGraph.prototype);
      Object.assign(view, {n: this.n, m: this.m, ptr, history: history.map(p => [...p]), id: globalThis.crypto.randomUUID()});
      return view;
    } finally { this.m._free(cells); }
  }
  /** Current-root issued/completed/cancelled, view pending, lifetime retired results, live game views. */
  counters() {
    const pointer = this.n.alloc(48);
    try {
      this.m._hxg_view_counters(this.ptr, pointer);
      const values = Array.from(this.n.view(BigUint64Array, pointer, 6), Number);
      return Object.fromEntries(['issued', 'completed', 'cancelled', 'pending', 'retired', 'views'].map((name, i) => [name, values[i]]));
    } finally { this.m._free(pointer); }
  }
  /** Dormant payload/index estimates; excludes allocator pools and process memory. */
  archive() {
    const pointer = this.n.alloc(80);
    try {
      if (!this.m._hxg_archive_stats(this.ptr, pointer)) return null;
      const values = Array.from(this.n.view(BigInt64Array, pointer, 10), Number);
      return Object.fromEntries(['nodes', 'bytes', 'limit', 'retained', 'reused', 'discarded', 'compatible',
        'index_bytes', 'indexed_cells', 'focus_stones'].map((name, i) => [name, values[i]]));
    } finally { this.m._free(pointer); }
  }
  /** Direct completed comparison credits, excluding inherited visits, in result() action order. */
  credits() {
    const count = this.m._hxg_root_credits(this.ptr, 0), pointer = this.n.alloc(8 * count);
    try {
      this.m._hxg_root_credits(this.ptr, pointer);
      return Array.from(this.n.view(BigUint64Array, pointer, count), Number);
    } finally { this.m._free(pointer); }
  }
  /** Moves the root to the position after `history`, keeping every node's statistics. */
  at(history) {
    if (this.nativeOwner) throw new Error('Close the native owner before moving its graph');
    const cells = this.n.cells(history);
    try {
      this.n.checked(this.m._hxg_root_at(this.ptr, cells, history.length));
    } finally {
      this.m._free(cells);
    }
    this.history = history.map(([q, r]) => [q, r]);
  }
  /** The history after the turn `result` chooses at this root (GameGraph.after_turn), or null; the root stays. */
  afterTurn(result) {
    if (!result.action || result.proven) return null;
    const root = this.history.map(p => [...p]), line = [...root, [...result.action]], mover = ((root.length + 1) >> 1) % 2;
    const state = this.n.game(line);
    if (state.winner >= 0) return null;
    if (state.player === mover) {
      this.at(line);
      const {actions, policy} = this.result();
      this.at(root);
      if (!policy.length || !(Math.max(...policy) > 0)) return null;
      line.push([...actions[policy.indexOf(Math.max(...policy))]]);
      if (this.n.game(line).winner >= 0) return null;
    }
    return line;
  }
  /** NeuralSearch.search with the principal-variation check of share `pvCheck` (python/neural_search.py Recheck): with a
   * check the result is the root's after it, its counts summed over every pass, and `pv_check`. */
  async search({pvCheck = 0, ...options} = {}) {
    const simulations = options.simulations ?? 128, reserve = Math.round(pvCheck * simulations);
    if (!(pvCheck > 0 && reserve && simulations - 2 * reserve >= 1)) return super.search(options);
    let completed = 0;
    const onBatch = stats => options.onBatch?.({...stats, completed: completed + stats.completed});
    const root = this.history.map(p => [...p]), first = await super.search({...options, onBatch, simulations: simulations - 2 * reserve});
    completed += first.completed;
    const line = first.stopped ? null : this.afterTurn(first);
    if (!line) return first;
    const index = first.actions.findIndex(([q, r]) => q === first.action[0] && r === first.action[1]), before = first.completed_q[index];
    const passes = [first];
    this.at(line);
    try {
      passes.push(await super.search({...options, onBatch, simulations: reserve}));
      completed += passes[1].completed;
    } finally {
      this.at(root);
    }
    const after = this.result().completed_q[index], searched = before - after > PV_DROP && !passes[1].stopped;
    if (searched) passes.push(await super.search({...options, onBatch, simulations: reserve}));
    const result = searched ? passes[2] : {...passes[1], ...this.result(options.choice ?? 'policy')};
    for (const key of ['completed', 'evaluated', 'cache_hits', 'elapsed_ms']) result[key] = passes.reduce((sum, p) => sum + p[key], 0);
    return {...result, stopped: passes.some(p => p.stopped), pv_check: {line: line.slice(root.length), before, after, searched}};
  }
}

/** Immutable native row snapshots. Network.forward reads a JS-owned input copy,
 * so WASM memory growth while it awaits inference cannot detach that input. */
export class NativeBatch {
  constructor(owner, count) {
    this.owner = owner; this.n = owner.n; this.m = owner.m; this.count = count; this.ptr = 0;
    this.ids = this.n.alloc(8 * count);
    const trees = this.n.alloc(4 * count), requests = this.n.alloc(4 * count);
    try {
      this.n.checked(this.m._hxgf_take(owner.feed, count, this.ids, trees, requests, 0, 0, 0n));
      this.ptr = this.m._hxgp_new_rect(trees, requests, count, 0);
      this.n.checked(this.ptr);
      owner.batches.add(this);
      this.groups = [];
      const info = this.n.alloc(24);
      try {
        for (let i = 0; i < this.m._hxgp_groups(this.ptr); i++) {
          this.n.checked(this.m._hxgp_shape(this.ptr, i, info));
          const [height, width, rows] = Array.from(this.n.view(BigInt64Array, info, 3), Number);
          this.groups.push({size: Math.max(height, width), height, width, rows});
        }
      } finally { this.m._free(info); }
    } catch (error) {
      // A failed snapshot still leaves submitted feed IDs. Cancel detaches
      // subscribers; the caller then abandons only after inference has settled.
      if (this.ptr) this.m._hxgp_free(this.ptr);
      this.ptr = 0; this.m._free(this.ids); this.ids = 0; owner.cancel();
      owner.batches.delete(this);
      throw error;
    } finally { this.m._free(trees); this.m._free(requests); }
  }
  features(group, start, count) {
    if (!this.ptr) throw new Error('Native batch is closed');
    const {height, width} = this.groups[group];
    const length = count * 20 * height * width, buffer = this.n.alloc(4 * length);
    try {
      this.n.checked(this.m._hxgp_features(this.ptr, group, start, count, buffer, BigInt(length)));
      return this.n.view(Float32Array, buffer, length).slice();
    } finally { this.m._free(buffer); }
  }
  decode(group, start, count, {policy, far, value}) {
    if (!this.ptr) throw new Error('Native batch is closed');
    const {height, width} = this.groups[group], area = height * width;
    if (policy.length !== count * area || far.length !== count || value.length !== count) throw new Error('Wrong native prediction shape');
    // Allocate before making heap views, since any allocation may grow memory.
    const p = this.n.alloc(4 * policy.length), f = this.n.alloc(4 * count), v = this.n.alloc(4 * count);
    try {
      this.n.view(Float32Array, p, policy.length).set(policy);
      this.n.view(Float32Array, f, count).set(far);
      this.n.view(Float32Array, v, count).set(value);
      this.n.checked(this.m._hxgp_decode_split(this.ptr, group, start, count, p, f, v));
    } finally { this.m._free(p); this.m._free(f); this.m._free(v); }
  }
  install() {
    if (!this.ptr) throw new Error('Native batch is closed');
    const pointers = this.n.alloc(16);
    try {
      this.n.checked(this.m._hxgp_outputs(this.ptr, pointers));
      const [offsets, actions, logits, values] = this.n.view(Uint32Array, pointers, 4);
      this.n.checked(this.m._hxgm_install(this.owner.pool, this.ids, this.count, offsets, actions, logits, values));
    } finally { this.m._free(pointers); }
    this.installed = true; this.close();
  }
  close() {
    if (this.ptr && !this.installed) this.owner.cancel();
    if (this.ptr) this.m._hxgp_free(this.ptr);
    this.ptr = 0;
    if (this.ids) this.m._free(this.ids);
    this.ids = 0; this.owner.batches.delete(this);
  }
}

/** The desktop C++ scheduler in one browser graph-owner worker. JavaScript
 * crosses the boundary per neural batch or immutable proof slice. */
export class NativeOwner {
  constructor(graph, {capacity = 4096, quantum = 32, views = 8, depth = 8, work = 0, ms = 1000, seed = 1740} = {}) {
    if (!(graph instanceof GameGraph) || !graph.ptr || graph.nativeOwner) throw new Error('A free game graph is required');
    if (!work && !(ms > 0)) throw new Error('A work allowance or positive clock is required');
    this.graph = graph; this.n = graph.n; this.m = graph.m; this.batches = new Set(); this.busy = false; this.proofs = null;
    const source = this.n.alloc(4), version = new TextEncoder().encode(String(graph.model || 'web') + '\0'), name = this.n.alloc(version.length);
    try {
      this.n.view(Uint32Array, source, 1)[0] = graph.ptr; this.n.view(Uint8Array, name, version.length).set(version);
      this.pool = this.m._hxgm_new(source, 1, capacity, quantum, views, depth, BigInt(work || quantum), name, BigInt(seed));
      this.n.checked(this.pool);
      if (ms && work) {
        const history = this.n.cells(graph.history);
        try { this.n.checked(this.m._hxgm_retarget(this.pool, 0, history, graph.history.length, BigInt(work), ms)); }
        finally { this.m._free(history); }
      } else if (ms) this.n.checked(this.m._hxgm_clock(this.pool, ms));
      this.ptr = this.m._hxgm_owner(this.pool, 0); this.feed = this.m._hxgm_feed(this.pool);
    } catch (error) {
      if (this.pool) this.m._hxgm_free(this.pool); this.pool = 0; throw error;
    } finally { this.m._free(source); this.m._free(name); }
    this.root = Object.assign(Object.create(NeuralSearch.prototype), {n: this.n, m: this.m, ptr: this.m._hxgo_root(this.ptr), history: graph.history, nativeOwner: this});
    graph.nativeOwner = this;
  }
  step(readyLimit = 64) {
    if (!this.ptr) throw new Error('Native owner is closed');
    this.n.checked(this.m._hxgm_ready_limit(this.pool, readyLimit));
    const progress = this.m._hxgm_step(this.pool);
    if (progress < 0) this.n.checked(0);
    return progress;
  }
  done() { return !this.ptr || Boolean(this.m._hxgm_done(this.pool)); }
  cancel() { if (this.ptr) { try { this.n.checked(this.m._hxgm_cancel(this.pool)); } finally { this.proofs?.cancelActive(); } } }
  admit() {
    if (!this.ptr) return false;
    if (this.m._hxg_exact(this.root.ptr) >= 0) this.cancel();
    const admitted = this.m._hxgm_admit(this.pool);
    if (admitted < 0) this.n.checked(0);
    if (this.m._hxg_exact(this.root.ptr) >= 0) { this.cancel(); return false; }
    return Boolean(admitted);
  }
  take(limit = 64) {
    if (!this.ptr) throw new Error('Native owner is closed');
    const info = this.n.alloc(16);
    try {
      if (!this.admit()) return null;
      this.n.checked(this.m._hxgf_layout(this.feed, limit, info));
      const count = Number(this.n.view(BigInt64Array, info, 2)[0]);
      return count ? new NativeBatch(this, count) : null;
    } finally { this.m._free(info); }
  }
  stats() {
    if (!this.ptr) throw new Error('Native owner is closed');
    const buffer = this.n.alloc(160);
    try {
      this.m._hxgo_stats(this.ptr, buffer);
      const values = Array.from(this.n.view(BigUint64Array, buffer, 20), Number);
      const names = ['ticks', 'completed', 'issued', 'cancelled', 'created', 'retired', 'candidates', 'active', 'pending', 'depth',
        'root_completed', 'deadline', 'step_ns', 'discover_ns', 'records', 'views', 'last_credits', 'root_passes', 'allocations', 'reclaimed'];
      this.m._hxgf_stats(this.feed, buffer);
      const feed = Array.from(this.n.view(BigInt64Array, buffer, 6), Number);
      return {...Object.fromEntries(names.map((name, i) => [name, values[i]])),
        ...Object.fromEntries(['neural_rows', 'joins', 'cache_hits', 'installed', 'tasks', 'subscribers'].map((name, i) => [name, feed[i]]))};
    } finally { this.m._free(buffer); }
  }
  result(choice = 'gumbel') {
    if (!this.ptr) throw new Error('Native owner is closed');
    if (choice !== 'policy' && choice !== 'gumbel') throw new Error('choice must be policy or gumbel');
    const result = this.root.result(choice), action = this.n.alloc(16);
    try {
      const found = this.m._hxgo_choice(this.ptr, action);
      if (found < 0) this.n.checked(0);
      const stats = this.stats();
      return {...result, action: choice === 'policy' ? result.action : found ? this.n.pairs(action, 1)[0] : null, completed: stats.root_completed,
        all_view_completed: stats.completed};
    } finally { this.m._free(action); }
  }
  close() {
    if (!this.ptr) return;
    if (this.busy || this.batches.size || this.proofs) throw new Error('Wait for native producers before closing their owner');
    this.cancel();
    this.n.checked(this.m._hxgf_abandon_all(this.feed));
    this.n.checked(this.m._hxgm_free(this.pool));
    this.ptr = this.pool = 0; this.root.ptr = 0; this.graph.nativeOwner = null;
    // The scheduler root owned archive focus. Reclaim it before this graph
    // advances, so permanent colour conflicts follow the played position.
    this.graph.at(this.graph.history);
  }
  async search({network, batchSize = 64, stop = () => false, onBatch = () => {}, choice = 'gumbel', capture = false, proofs = null}) {
    if (this.busy || !this.ptr) throw new Error('Native owner is closed or already running');
    if (!(batchSize > 0)) throw new Error('Positive native batch size required');
    const frontier = proofs ? new NativeProofs(this, {...proofs, stop}) : null;
    this.busy = true; const started = performance.now(); let batches = 0, largest = 0, networkMs = 0, batch = null, flight = null;
    const inference = {rows: 0, physical: 0, forwards: 0};
    try {
      while (!this.done()) {
        if (stop()) { this.cancel(); break; }
        this.step(batchSize); frontier?.pump(); frontier?.check();
        if (this.done()) break;
        batch = this.take(batchSize);
        if (batch) {
          const begin = performance.now();
          let settled = false;
          flight = network.evaluateNative(batch, {capture, stop: () => {
            if (stop()) this.cancel();
            return !this.admit();
          }});
          // Observe rejection immediately while the owner prepares useful work
          // for the next batch. Await the same promise before releasing buffers.
          const observed = flight.then(() => { settled = true; }, () => { settled = true; });
          while (!settled && !this.done() && this.m._hxgf_queued(this.feed) < BigInt(batchSize)) {
            if (stop()) { this.cancel(); break; }
            const progress = this.step(batchSize); frontier?.pump(); frontier?.check();
            await nextTask();
            if (!progress) break;
          }
          await observed; const evaluated = await flight; flight = null;
          if (batch.inference) for (const key of Object.keys(inference)) inference[key] += batch.inference[key];
          networkMs += performance.now() - begin;
          await nextTask();
          if (stop()) this.cancel();
          if (evaluated === false || !this.admit()) {
            // Incomplete or no-longer-admitted batches cannot be installed.
            // The active forward has settled before their storage is released.
            this.cancel(); batch.close(); batch = null; break;
          }
          // Detached subscribers cannot install into a changed/cancelled view.
          batch.install(); batches++; largest = Math.max(largest, batch.count); batch = null;
          onBatch(this.stats());
        }
        await nextTask();
      }
      if (this.done()) this.n.checked(this.m._hxgf_abandon_all(this.feed));
      if (frontier) await frontier.close();
      return {...this.result(choice), scheduler: this.stats(), elapsed_ms: performance.now() - started,
        ...(frontier ? {proof_scheduler: frontier.finalStats, proof_records: frontier.records, neural_records: frontier.neuralRecords, solver_error: frontier.workerError} : {}),
        batches, largest, network_ms: networkMs, inference, ...(capture ? {capture_pool: network.captureStats()} : {})};
    } catch (error) {
      this.cancel(); throw error;
    } finally {
      // evaluateNative resolves/rejects only after its current forward settles.
      // A host callback/phase can also throw while that forward is still live.
      if (flight) await flight.catch(() => {});
      try { batch?.close(); } finally {
        try { if (frontier) await frontier.close(); } finally {
          this.busy = false;
          if (this.done()) this.n.checked(this.m._hxgf_abandon_all(this.feed));
        }
      }
    }
  }
}

/** Transport for the compiled proof frontier. query(worker, request) must keep
 * each worker's resident solver table and resolve only after its slice stops. */
export class NativeProofs {
  constructor(owner, {query, cancel = null, maxSlice = cancel ? 1000 : 64, workers = 1, slice = 8, table = 4, tasks = 256, stamps = false, endpoints = 8, stop = () => false} = {}) {
    if (!owner.pool || owner.proofs || typeof query !== 'function') throw new Error('A free native owner and proof transport are required');
    if (!Number.isInteger(endpoints) || endpoints < 0 || endpoints > 8) throw new Error('Neural frontier limit must be an integer from 0 to 8');
    if (maxSlice > 64 && typeof cancel !== 'function') throw new Error('Long solver slices require cooperative cancellation');
    this.owner = owner; this.n = owner.n; this.m = owner.m; this.query = query; this.workers = workers; this.stop = stop; this.signal = cancel;
    this.pending = new Map(); this.failure = null; this.workerError = null; this.closing = null; this.records = []; this.neuralRecords = [];
    this.ptr = this.m._hxpe_new(owner.pool, workers, workers, slice, table, tasks, Number(stamps), maxSlice);
    this.n.checked(this.ptr);
    try { this.n.checked(this.m._hxp_neural(this.ptr, 0n, endpoints)); }
    catch (error) { this.m._hxp_free(this.ptr); this.ptr = 0; throw error; }
    owner.proofs = this;
  }
  check() { if (this.failure) throw this.failure; }
  cancelActive() { for (const worker of this.pending.keys()) this.signal?.(worker); }
  offer(history, relevance = 1) {
    const cells = this.n.cells(history);
    try { this.n.checked(this.m._hxp_offer(this.ptr, 0, cells, history.length, relevance)); }
    finally { this.m._free(cells); }
  }
  complete(worker, id, found) {
    const data = found.info || Array(13).fill(0), cells = found.moves || [], text = new TextEncoder().encode((found.raw || '') + '\0');
    const info = this.n.alloc(104), moves = this.n.cells(cells), result = this.n.alloc(text.length);
    try {
      this.n.view(BigUint64Array, info, 13).set(data.map(BigInt)); this.n.view(Uint8Array, result, text.length).set(text);
      if (found.neural?.length) {
        const values = this.n.alloc(found.neural.length * 8);
        try {
          this.n.view(BigInt64Array, values, found.neural.length).set(found.neural.map(BigInt));
          this.n.checked(this.m._hxpe_neural(this.ptr, worker, id, values, found.neural.length));
        } finally { this.m._free(values); }
      }
      this.n.checked(this.m._hxpe_complete(this.ptr, worker, id, info, moves, cells.length, result, 0));
    } finally { this.m._free(info); this.m._free(moves); this.m._free(result); }
  }
  pump() {
    if (!this.ptr || this.closing) return;
    if (this.stop()) this.owner.cancel();
    const admitted = this.owner.admit();
    for (const worker of this.pending.keys()) if (this.m._hxpe_cancelled(this.ptr, worker)) this.signal?.(worker);
    if (!admitted) return;
    for (let worker = 0; worker < this.workers; worker++) {
      if (this.pending.has(worker)) continue;
      const id = this.m._hxpe_take(this.ptr, worker);
      if (id === 0xffffffffffffffffn) this.n.checked(0);
      if (!id) continue;
      const request = JSON.parse(this.m.UTF8ToString(this.m._hxpe_request(this.ptr, worker)));
      const cancelled = () => Boolean(this.closing || !this.ptr || this.m._hxpe_cancelled(this.ptr, worker));
      // Promise callbacks only deliver immutable completions. Installation and
      // renewed admission run through the single native pool between phases.
      const done = Promise.resolve().then(() => {
        if (!cancelled()) return this.query(worker, request, cancelled);
        const info = Array(13).fill(0), size = request.history.length;
        info[4] = 1; info[9] = Math.floor((size + 1) / 2) % 2; info[10] = !size || size % 2 === 0 ? 1 : 2;
        info[11] = request.attacker === 'defender' ? 1 : 0; info[12] = 1;
        return {info, moves: []};
      }).catch(error => {
        this.workerError = String(error.message || error);
        const info = Array(13).fill(0), size = request.history.length;
        info[9] = Math.floor((size + 1) / 2) % 2; info[10] = !size || size % 2 === 0 ? 1 : 2;
        info[11] = request.attacker === 'defender' ? 1 : 0; info[12] = 1;
        return {info, moves: [], raw: ''};
      }).then(found => {
        try { this.complete(worker, id, found); } catch (error) {
          this.failure ||= error;
          const info = Array(13).fill(0), size = request.history.length;
          info[9] = Math.floor((size + 1) / 2) % 2; info[10] = !size || size % 2 === 0 ? 1 : 2;
          info[11] = request.attacker === 'defender' ? 1 : 0; info[12] = 1;
          this.complete(worker, id, {info, moves: []});
        }
        this.pending.delete(worker);
        if (!this.closing && !this.failure) this.pump();
      }).catch(error => { this.failure ||= error; this.pending.delete(worker); });
      this.pending.set(worker, done);
    }
  }
  stats() {
    const out = this.n.alloc(128), times = this.n.alloc(32);
    try {
      this.m._hxp_stats(this.ptr, out, times);
      const values = Array.from(this.n.view(BigUint64Array, out, 16), Number), elapsed = Array.from(this.n.view(Float64Array, times, 4));
      this.m._hxp_neural_stats(this.ptr, out);
      const neural = Array.from(this.n.view(BigUint64Array, out, 6), Number);
      return {...Object.fromEntries(['ticks','submitted','started','finished','installed','cancelled','pruned','unknown','fresh_nodes','missing_fresh','queued','active','ready','tasks','facts','records'].map((name, i) => [name, values[i]])),
        neural_frontier: Object.fromEntries(['paths','candidates','rejected','bytes','install_ns','records'].map((name, i) => [name, neural[i]])),
        ...Object.fromEntries(['worker_service_ms','worker_idle_ms','snapshot_ms','install_ms'].map((name, i) => [name, elapsed[i]]))};
    } finally { this.m._free(out); this.m._free(times); }
  }
  close() {
    if (!this.ptr) return this.closing || Promise.resolve();
    if (this.closing) return this.closing;
    this.m._hxp_cancel(this.ptr);
    this.cancelActive();
    this.closing = (async () => {
      await Promise.all([...this.pending.values()]);
      this.n.checked(this.m._hxp_drain(this.ptr)); this.finalStats = this.stats();
      for (let i = 0; i < this.finalStats.records; i++) this.records.push(JSON.parse(this.m.UTF8ToString(this.m._hxp_record(this.ptr, i))));
      for (let i = 0; i < this.finalStats.neural_frontier.records; i++) this.neuralRecords.push(JSON.parse(this.m.UTF8ToString(this.m._hxp_neural_record(this.ptr, i))));
      this.n.checked(this.m._hxp_free(this.ptr)); this.ptr = 0; this.owner.proofs = null;
      this.check();
    })();
    return this.closing;
  }
}

/**
 * Game graphs that follow games, one per line (a key the page changes on undo, a new or loaded game and, for a seat,
 * a seat change). `graph(line, history, options)` is the line's GameGraph moved to `history`; it is built afresh from
 * `options` (GameGraph's) when the line is new or `options` differ. The `limit` most recently used lines keep their
 * graphs.
 */
export class GameGraphs {
  constructor(native, limit = 3) {
    this.native = native;
    this.limit = limit;
    this.graphs = new Map();
  }
  graph(line, history, options) {
    const key = JSON.stringify(options);
    let kept = this.graphs.get(line);
    this.graphs.delete(line);
    if (kept && kept.key !== key) {
      kept.graph.close();
      kept = null;
    }
    kept ??= {key, graph: new GameGraph(this.native, {...options, history})};
    this.graphs.set(line, kept);
    for (const [old, {graph}] of this.graphs) {
      if (this.graphs.size <= this.limit) break;
      graph.close();
      this.graphs.delete(old);
    }
    kept.graph.at(history);
    return kept.graph;
  }
}
