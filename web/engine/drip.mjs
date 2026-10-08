/* "Drip (browser)": Drip (src/hexo.cpp) as WebAssembly in drip-worker.mjs, for the play page's browser engines
 * (seat.mjs). It plays as python/play.py's Drip: the same presets in ms, depth 12, width 16. */
import {DEPTH} from './native/search.mjs';
import {json, workerUrl} from './assets.mjs';
import {watchdog, workerError} from './stages.mjs';

export const PRESETS = {lightning: {ms: 100}, quick: {ms: 250}, standard: {ms: 1000}, strong: {ms: 3000},
  deep: {ms: 10000}, dangerous: {ms: 60000}};
const ID = 'browser:drip', LABEL = 'Drip (browser)', MATE = 10000000;
const SCALE = 1000;   // score units per factor e of the shown odds; Drip's score is a heuristic, not a probability
const playerAt = ply => ply === 0 ? 0 : ((ply - 1 >> 1) + 1) % 2;

/** Drip's files (assets.mjs records): native.wasm as web/engine/build.json pins it. */
export async function files() {
  const {data, local} = await json('build.json');
  return [{path: 'native/native.wasm', sha256: data.artefacts['native/native.wasm'], local}];
}

export class DripEngine {
  constructor() {
    this.worker = null;
    this.ready = null;
    this.calls = 0;
    this.waits = new Map();
  }

  /** Starts the worker and loads native.wasm; `progress(fraction, stage)` reports the download and the compile, and a
   * stage that stays silent for its stages.mjs LIMITS entry fails the load. */
  load(progress = () => {}) {
    if (this.ready) return this.ready;
    const worker = this.worker = new Worker(workerUrl('drip-worker.mjs'), {type: 'module'});
    const ready = this.ready = new Promise((resolve, reject) => {
      const dog = watchdog(LABEL, reject);
      this.waits.set(0, {resolve: value => { dog.stop(); resolve(value); }, reject: error => { dog.stop(); reject(error); }});
      worker.onmessage = ({data}) => {
        if (data.type === 'progress') { dog.watch(data.stage); progress(data.fraction, data.stage); return; }
        const id = data.type === 'ready' ? 0 : data.id ?? 0, wait = this.waits.get(id);
        this.waits.delete(id);
        if (data.type === 'error') wait?.reject(workerError(LABEL, data));
        else wait?.resolve(data.result);
      };
      worker.onerror = event => this.fail(new Error(event.message || 'Drip worker failed'));
      worker.postMessage({type: 'load'});
    });
    ready.catch(error => { if (this.ready === ready) this.fail(error); });
    return ready;
  }

  /**
   * Drip's turn at `history` ([[q, r], ...]) with `budget` {ms, depth?} (a PRESETS entry; depth defaults to the
   * server's 12): {moves, score, depth, nodes, elapsed_ms}. Under a clock `ms` caps the budget's time. `progress(fraction)`
   * follows the clock. Aborting `signal` ends the worker (a search cannot be interrupted inside it) and rejects with an
   * AbortError; the next call starts a new one.
   */
  async turn(history, budget, {signal, progress = () => {}, ms = null} = {}) {
    await this.load();
    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');
    const id = ++this.calls, start = performance.now(), time = Math.max(1, Math.floor(Math.min(budget.ms, ms ?? Infinity)));
    const timer = setInterval(() => progress(Math.min(1, (performance.now() - start) / time)), 50);
    const cancel = () => this.fail(new DOMException('Cancelled', 'AbortError'));
    signal?.addEventListener('abort', cancel, {once: true});
    try {
      return await new Promise((resolve, reject) => {
        this.waits.set(id, {resolve, reject});
        this.worker.postMessage({type: 'turn', id, history, ms: time, depth: budget.depth ?? DEPTH});
      });
    } finally {
      clearInterval(timer);
      signal?.removeEventListener('abort', cancel);
    }
  }

  files() {
    return files();
  }

  /** Ends the worker; pending calls reject with an AbortError. */
  close() {
    this.fail(new DOMException('Closed', 'AbortError'));
  }

  /** Ends the worker and rejects the pending load and calls with `error`; the next call starts a new worker. */
  fail(error) {
    this.worker?.terminate();
    this.worker = null;
    this.ready = null;
    for (const wait of this.waits.values()) wait.reject(error);
    this.waits.clear();
  }
}

/**
 * The evaluation record of Drip's turn `result` at `history` for the analysis panel: its first stone as the only
 * candidate, both stones as the line, and as the value the mover's odds logistic(score / SCALE), 1 or 0 for a proven
 * win or loss.
 */
export function record(result, history, preset) {
  const player = playerAt(history.length), proven = result.score >= MATE ? 1 : result.score <= -MATE ? -1 : 0;
  const value = proven ? (proven + 1) / 2 : Math.round(1e4 / (1 + Math.exp(-result.score / SCALE))) / 1e4;
  return {moves: result.moves, value, top: result.moves.slice(0, 1).map(([q, r]) => [q, r, 1, value, proven]),
    line: result.moves.map(([q, r]) => [q, r, player]), threat: [], proof: null, ms: PRESETS[preset].ms,
    score: result.score, depth: result.depth, nodes: result.nodes, engine: ID};
}

export const drip = {entry: {id: ID, kind: 'drip', name: LABEL, label: LABEL, checkpoints: [], presets: PRESETS, analysis: true, clocks: true},
  engine: new DripEngine(), record, build: 'python tools/build_web.py wasm'};
