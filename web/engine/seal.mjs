/* Seal (Ramora0/HexTicTacToe) running in the browser: tools/seal_adapter.cpp compiled to seal/engine.wasm
 * (tools/build_web.py seal), searched in seal-worker.mjs. Seal's budget is a clock in ms. */

export const PRESETS = {lightning: {ms: 50}, quick: {ms: 100}, standard: {ms: 500}, strong: {ms: 2000}, deep: {ms: 8000},
  dangerous: {ms: 30000}};

const AXES = [[1, 0], [0, 1], [1, -1]];
const playerAt = ply => ply === 0 ? 0 : ((ply - 1 >> 1) + 1) % 2;

/** Whether the stone at `cell` completes six in a row for its owner in `stones` (a Map 'q,r' -> player). */
function wins(stones, [q, r]) {
  const owner = stones.get(`${q},${r}`);
  return AXES.some(([dq, dr]) => {
    let run = 1;
    for (const sign of [1, -1]) {
      for (let k = 1; stones.get(`${q + sign * k * dq},${r + sign * k * dr}`) === owner; k++) run++;
    }
    return run >= 6;
  });
}

/**
 * Seal's turn at `history` ([[q, r], ...]) with `ms` to think, through `module` (the instantiated engine.mjs), as
 * python/play.py plays it: the stones seal_move returns, cut where the turn ends or a stone wins. Returns
 * {moves, raw, ms}: `raw` is seal_move's answer uncut. Throws when Seal's board range is exceeded or its answer is
 * not a whole turn.
 */
export function sealTurn(module, history, ms) {
  const n = history.length, player = playerAt(n), remaining = n === 0 || n % 2 === 0 ? 1 : 2;
  const input = module._malloc(12 * Math.max(1, n)), output = module._malloc(16);
  try {
    module.HEAP32.set(history.flatMap(([q, r], i) => [q, r, playerAt(i)]), input >> 2);
    const start = performance.now();
    const count = module._seal_move(input, n, player, remaining, ms, output);
    const elapsed = performance.now() - start;
    if (count < 0) throw new Error('Seal board range exceeded');
    const raw = [];
    for (let i = 0; i < count; i++) raw.push([module.HEAP32[(output >> 2) + 2 * i], module.HEAP32[(output >> 2) + 2 * i + 1]]);
    const stones = new Map(history.map(([q, r], i) => [`${q},${r}`, playerAt(i)])), moves = [];
    for (const [q, r] of raw) {
      stones.set(`${q},${r}`, player);
      moves.push([q, r]);
      if (moves.length === remaining || wins(stones, [q, r])) return {moves, raw, ms: Math.round(elapsed)};
    }
    throw new Error('Seal returned an incomplete turn');
  } finally {
    module._free(input);
    module._free(output);
  }
}

/** The page-side handle of seal-worker.mjs. A cancelled turn ends its worker; the next call starts another. */
export class SealEngine {
  constructor() {
    this.worker = null;
    this.ready = null;
    this.abandon = null;
    this.pending = null;
    this.calls = 0;
  }

  /** Starts the worker and loads the wasm (Cache API, keyed by its SHA-256); `progress(fraction)` reports the download. */
  load(progress = () => {}) {
    if (this.ready) return this.ready;
    const worker = this.worker = new Worker(new URL('seal-worker.mjs', import.meta.url), {type: 'module'});
    const ready = this.ready = new Promise((resolve, reject) => {
      this.abandon = reject;
      worker.onmessage = ({data}) => {
        if (data.type === 'progress') progress(data.fraction);
        else if (data.type === 'ready') { this.abandon = null; resolve(data.revision); }
        else if (data.id === undefined) reject(new Error(data.message));
        else this.settle(data);
      };
      worker.onerror = event => this.fail(new Error(event.message || 'Seal worker failed'));
    });
    ready.catch(error => { if (this.ready === ready) this.fail(error); });
    worker.postMessage({type: 'load'});
    return ready;
  }

  settle(data) {
    const pending = this.pending;
    if (!pending || pending.id !== data.id) return;
    this.pending = null;
    clearInterval(pending.timer);
    if (data.type === 'result') pending.resolve(data.result);
    else pending.reject(new Error(data.message));
  }

  /**
   * Seal's turn at `history` with `budget` {ms} (a PRESETS entry): {moves, raw, ms}. `progress(fraction)` follows the
   * clock; aborting `signal` ends the worker and rejects with an AbortError. One turn runs at a time.
   */
  async turn(history, budget, {signal, progress = () => {}} = {}) {
    await this.load();
    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');
    if (this.pending) throw new Error('Seal is already thinking');
    const id = ++this.calls, start = performance.now();
    return new Promise((resolve, reject) => {
      const timer = setInterval(() => progress(Math.min(1, (performance.now() - start) / budget.ms)), 50);
      this.pending = {id, resolve, reject, timer};
      signal?.addEventListener('abort', () => { if (this.pending?.id === id) this.fail(new DOMException('Cancelled', 'AbortError')); }, {once: true});
      this.worker.postMessage({type: 'turn', id, history, ms: budget.ms});
    });
  }

  /** Ends the worker and rejects the pending load and call with `error`. */
  fail(error) {
    this.worker?.terminate();
    this.worker = null;
    this.ready = null;
    this.abandon?.(error);
    this.abandon = null;
    const pending = this.pending;
    this.pending = null;
    if (pending) {
      clearInterval(pending.timer);
      pending.reject(error);
    }
  }

  close() {
    this.fail(new DOMException('Closed', 'AbortError'));
  }
}
