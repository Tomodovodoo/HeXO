/* Seal (Ramora0/HexTicTacToe) running in the browser: tools/seal_adapter.cpp compiled to seal/engine.wasm
 * (tools/build_web.py seal), searched in seal-worker.mjs. Seal's budget is a clock in ms. */
import {NotOnSite, json, pins, published, workerUrl} from './assets.mjs';
import {watchdog, workerError} from './stages.mjs';

export const PRESETS = {lightning: {ms: 100}, quick: {ms: 250}, standard: {ms: 1000}, strong: {ms: 3000}, deep: {ms: 10000},
  dangerous: {ms: 60000}};

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

/** {revision, files}: Seal's revision and its module and wasm (assets.mjs records) as seal/manifest.json pins them;
 * NotOnSite when the build has no Seal. */
export async function files() {
  if (!await published('seal')) throw new NotOnSite('seal/manifest.json');
  const found = await json('seal/manifest.json'), {data, local} = found;
  const files = await pins('seal/manifest.json', found, other => other.revision === data.revision && other.files?.['engine.wasm'] === data.sha256);
  return {revision: data.revision, files: ['engine.mjs', 'engine.wasm'].map(name => ({path: `seal/${name}`,
    sha256: files[name] ?? (name === 'engine.wasm' ? data.sha256 : undefined), local}))};
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

  /** Starts the worker and loads Seal (assets.mjs); `progress(fraction, stage)` follows the download and the compile,
   * and a stage that stays silent for its stages.mjs LIMITS entry fails the load. */
  load(progress = () => {}) {
    if (this.ready) return this.ready;
    const worker = this.worker = new Worker(workerUrl('seal-worker.mjs'), {type: 'module'});
    const ready = this.ready = new Promise((resolve, reject) => {
      this.abandon = reject;
      const dog = watchdog(LABEL, reject);
      worker.onmessage = ({data}) => {
        if (data.type === 'progress') { dog.watch(data.stage); progress(data.fraction, data.stage); }
        else if (data.type === 'ready') { dog.stop(); this.abandon = null; resolve(data.revision); }
        else if (data.id === undefined) { dog.stop(); reject(workerError(LABEL, data)); }
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
   * Seal's turn at `history` with `budget` {ms} (a PRESETS entry), under a clock no longer than `ms`: {moves, raw, ms}.
   * `progress(fraction)` follows the clock; aborting `signal` ends the worker and rejects with an AbortError. One turn
   * runs at a time.
   */
  async turn(history, budget, {signal, progress = () => {}, ms = null} = {}) {
    await this.load();
    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');
    if (this.pending) throw new Error('Seal is already thinking');
    const id = ++this.calls, start = performance.now(), time = Math.max(1, Math.floor(Math.min(budget.ms, ms ?? Infinity)));
    return new Promise((resolve, reject) => {
      const timer = setInterval(() => progress(Math.min(1, (performance.now() - start) / time)), 50);
      this.pending = {id, resolve, reject, timer};
      signal?.addEventListener('abort', () => { if (this.pending?.id === id) this.fail(new DOMException('Cancelled', 'AbortError')); }, {once: true});
      this.worker.postMessage({type: 'turn', id, history, ms: time});
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

  async files() {
    return (await files()).files;
  }

  close() {
    this.fail(new DOMException('Closed', 'AbortError'));
  }
}

const ID = 'browser:seal', LABEL = 'Seal (browser)';

/** The evaluation record of Seal's turn `result` at `history`: its first stone as the only candidate and both stones
 * as the line. seal_move returns no score, so the record has no value. */
export function record(result, history, preset) {
  const player = playerAt(history.length);
  return {moves: result.moves, value: null, top: result.moves.slice(0, 1).map(([q, r]) => [q, r, 1]),
    line: result.moves.map(([q, r]) => [q, r, player]), threat: [], proof: null, ms: PRESETS[preset].ms, engine: ID};
}

export const seal = {entry: {id: ID, kind: 'seal', name: LABEL, label: LABEL, checkpoints: [], presets: PRESETS, analysis: true, clocks: true},
  engine: new SealEngine(), fresh: () => new SealEngine(), record, build: 'python tools/build_web.py seal'};
