/* "Shrimp (browser)": Shrimp (Cmiller132/hexo-bot main_7, MIT) in shrimp-worker.mjs, for the play page's browser
 * engines (seat.mjs). It plays as the server's Shrimp entry (tools/engines.json): Six's driver with these visits per
 * stone, mirrored into Six's frame. */
import {json, workerUrl} from './assets.mjs';
import {defaultThreads, loadFiles, probe} from './network.mjs';
import {ShrimpNetwork} from './shrimp/network.mjs';
import {NEURAL_PRESET} from './device.mjs';

/** The server entry's presets, as visits per stone; `simulations` is what the analysis panel shows. */
export const PRESETS = Object.fromEntries(Object.entries({lightning: 16, quick: 32, standard: 128, strong: 512, deep: 1024,
  dangerous: 4096}).map(([name, visits]) => [name, {visits, simulations: visits}]));
const ID = 'browser:shrimp', LABEL = 'Shrimp (browser)', READY_MS = 20000, STALLED = Symbol('stalled');
const playerAt = ply => ply === 0 ? 0 : ((ply - 1 >> 1) + 1) % 2;
const round = x => Math.round(x * 1e4) / 1e4;

export class ShrimpEngine {
  constructor(options = {}) {
    this.options = options;
    this.worker = null;
    this.ready = null;
    this.calls = 0;
    this.waits = new Map();
    this.device = null;
  }

  /**
   * Starts the worker, downloads the network and the search; `progress(fraction)` reports loading. Resolves to the
   * device ({provider, threads, ...}). As for Bubble, when the thread count was left to the loader and the runtime
   * has not come up READY_MS after the downloads, the worker is replaced by one on a single thread.
   */
  load(progress = () => {}) {
    if (this.ready) return this.ready;
    const ready = this.ready = this.start(progress).catch(error => {
      if (error !== STALLED) throw error;
      console.warn('Shrimp (browser): the runtime did not start with its thread workers; retrying on one thread');
      this.options = {...this.options, threads: 1};
      return this.start(progress);
    });
    ready.catch(error => { if (this.ready === ready) this.fail(error); });
    return ready;
  }

  start(progress) {
    return new Promise((resolve, reject) => {
      const worker = this.worker = new Worker(workerUrl('shrimp-worker.mjs'), {type: 'module'});
      const threaded = this.options.threads == null && defaultThreads({isolated: Boolean(globalThis.crossOriginIsolated),
        cores: navigator.hardwareConcurrency || 2}) > 1;
      let timer = null;
      const stall = armed => {
        clearTimeout(timer);
        if (armed && threaded) timer = setTimeout(() => { if (this.worker === worker) { worker.terminate(); reject(STALLED); } }, READY_MS);
      };
      this.waits.set(0, {resolve: device => { clearTimeout(timer); resolve(device); },
        reject: error => { clearTimeout(timer); reject(error); }, progress: f => { progress(f); stall(f >= .95); }});
      worker.onmessage = ({data}) => {
        const id = data.type === 'ready' ? 0 : data.id ?? 0, wait = this.waits.get(id);
        if (!wait) return;
        if (data.type === 'progress') { wait.progress(data.fraction); return; }
        this.waits.delete(id);
        if (data.type === 'ready') { this.device = data.device; wait.resolve(data.device); }
        else if (data.type === 'result') wait.resolve(data.result);
        else wait.reject(data.type === 'cancelled' ? new DOMException('Cancelled', 'AbortError') : new Error(data.message));
      };
      worker.onerror = event => { clearTimeout(timer); this.fail(new Error(event.message || 'Shrimp worker failed')); };
      worker.postMessage({type: 'load', options: this.options});
    });
  }

  /**
   * Shrimp's turn at `history` ([[q, r], ...]) with `budget` {visits} (a PRESETS entry): {moves, stones, ms, ...},
   * each stone {action, value (Shrimp's value for the side to move, -1 to 1), visits, moves: root moves with visits,
   * q and share}. Aborting `signal` stops the search after its current network batch and rejects with an AbortError.
   */
  async turn(history, budget, {signal, progress = () => {}} = {}) {
    await this.load();
    const id = ++this.calls, worker = this.worker;
    return new Promise((resolve, reject) => {
      if (signal?.aborted) { reject(new DOMException('Cancelled', 'AbortError')); return; }
      this.waits.set(id, {resolve, reject, progress});
      signal?.addEventListener('abort', () => worker.postMessage({type: 'cancel', id}), {once: true});
      worker.postMessage({type: 'turn', id, history, visits: budget.visits});
    });
  }

  /** The downloaded files (assets.mjs records) a load on this device may read: ONNX Runtime (with a WebGPU start's
   * WebAssembly fallback), the graph and shrimp.wasm. */
  async files() {
    const [{provider}, {file}, {data, local}] = await Promise.all([probe(this.options.prefer), ShrimpNetwork.files(), json('build.json')]);
    return [...await loadFiles(provider, this.options.prefer), file, {path: 'shrimp/shrimp.wasm', sha256: data.artefacts['shrimp/shrimp.wasm'], lines: true, local}];
  }

  /** Ends the worker; pending loads and calls reject with an AbortError. */
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
 * The evaluation record of Shrimp's turn `result` at `history` for the analysis panel: the first stone's search as
 * the candidates (share of visits, the mover's win chance (Q + 1) / 2), both stones as the line, and the root value
 * of the first stone's search as the mover's win chance.
 */
export function record(result, history, preset) {
  const [first] = result.stones, player = playerAt(history.length);
  return {moves: result.moves, value: round((first.value + 1) / 2),
    top: first.moves.slice(0, 5).map(m => [m.cell[0], m.cell[1], round(m.share), round((m.q + 1) / 2), 0]),
    line: result.moves.map(([q, r]) => [q, r, player]), threat: [], proof: null, solved: false,
    simulations: PRESETS[preset].visits, solver_nodes: 0, ms: result.ms, engine: ID};
}

export const shrimp = {entry: {id: ID, kind: 'six', badge: 'shrimp', name: LABEL, label: LABEL, checkpoints: [], presets: PRESETS, preset: NEURAL_PRESET, analysis: true, clocks: false},
  engine: new ShrimpEngine(), record, build: 'python tools/build_web.py ort shrimp'};
