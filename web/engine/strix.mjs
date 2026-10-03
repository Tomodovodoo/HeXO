/* "Strix (browser)": Strix's network and Gumbel search (tools/strix_web as strix/strix.wasm) in strix-worker.mjs, for
 * the play page's browser engines (seat.mjs). It plays as python/play.py's Strix: the same presets in simulations per
 * placement, with a network `python tools/build_web.py strix-network` placed in strix/ (listed in strix/networks.json,
 * the first is the default), or the site's when this origin has none. */
import {json, workerUrl} from './assets.mjs';

/** Simulations per placement, as python/play.py's Strix presets. */
export const PRESETS = {lightning: {simulations: 2}, quick: {simulations: 8}, standard: {simulations: 64},
  strong: {simulations: 128}, deep: {simulations: 512}, dangerous: {simulations: 4096}};
const ID = 'browser:strix', LABEL = 'Strix (browser)';

export class StrixEngine {
  /** `networks` are the strix/networks.json entries with their `path` under web/engine, the default first; with none,
   * the engine reads the list again before its next load. `checkpoints` lists their ids. */
  constructor(networks) {
    this.checkpoints = [];
    this.networks = new Map();
    this.network = null;
    this.use(networks);
    this.worker = null;
    this.ready = null;
    this.calls = 0;
  }

  /** Adds `networks` (strix/networks.json entries) to the choices; the first becomes current when none is. */
  use(networks) {
    for (const network of networks) this.networks.set(network.id, network);
    this.checkpoints.splice(0, Infinity, ...this.networks.keys());
    this.network ??= networks[0] ?? null;
  }

  /** Reads strix/networks.json again when the engine has no network yet; throws when it still has none. */
  async known() {
    if (!this.network) this.use(await networks());
  }

  /** Starts the worker with the current network; `progress(fraction)` reports the download. */
  load(progress = () => {}) {
    if (this.ready) return this.ready;
    if (!this.network) {
      const ready = this.ready = this.known().then(() => { this.ready = null; return this.load(progress); });
      ready.catch(() => { if (this.ready === ready) this.ready = null; });
      return ready;
    }
    const worker = this.worker = new Worker(workerUrl('strix-worker.mjs'), {type: 'module'});
    const ready = this.ready = new Promise((resolve, reject) => {
      worker.onmessage = ({data}) => {
        if (data.type === 'progress' && data.id === undefined) progress(data.fraction);
        else if (data.type === 'ready') resolve(data.info);
        else if (data.type === 'error') reject(new Error(data.message));
      };
      worker.onerror = event => reject(new Error(event.message || 'Strix worker failed'));
    });
    ready.catch(() => { if (this.ready === ready) this.close(); });
    worker.postMessage({type: 'load', network: {path: this.network.path, sha256: this.network.sha256, bytes: this.network.size}});
    return ready;
  }

  /** The downloaded files (assets.mjs records) a load reads: strix.wasm and the current network. */
  async files() {
    await this.known();
    const {data, local} = await json('build.json');
    return [{path: 'strix/strix.wasm', sha256: data.artefacts['strix/strix.wasm'], lines: true, local},
      {path: this.network.path, sha256: this.network.sha256, bytes: this.network.size, local: this.network.local}];
  }

  /** Ends the worker; the next load starts a new one. */
  close() {
    this.worker?.terminate();
    this.worker = this.ready = null;
  }

  /**
   * Strix's turn after `history` ([[q, r], ...]) at `budget.simulations` per placement with network `budget.checkpoint`
   * (the default when absent): {moves, value, top, simulations, eval_states, root_visits, ms}. Aborting `signal` ends
   * the worker (a search cannot be interrupted inside it) and rejects with an AbortError; the next call starts a new one.
   */
  async turn(history, budget, {signal, progress = () => {}} = {}) {
    const network = this.networks.get(budget.checkpoint) ?? this.networks.values().next().value ?? null;
    if (network !== this.network) {
      this.close();
      this.network = network;
    }
    await this.load();
    const worker = this.worker, id = ++this.calls;
    return new Promise((resolve, reject) => {
      const abort = () => { if (this.worker === worker) this.close(); reject(new DOMException('Cancelled', 'AbortError')); };
      if (signal?.aborted) { abort(); return; }
      signal?.addEventListener('abort', abort, {once: true});
      worker.onmessage = ({data}) => {
        if (data.id !== id) return;
        if (data.type === 'progress') { progress(data.fraction); return; }
        signal?.removeEventListener('abort', abort);
        data.type === 'result' ? resolve(data.result) : reject(new Error(data.message));
      };
      worker.onerror = event => { if (this.worker === worker) this.close(); reject(new Error(event.message || 'Strix worker failed')); };
      worker.postMessage({type: 'turn', id, history, simulations: budget.simulations});
    });
  }
}

/** The strix/networks.json entries, from the site when this origin has none; throws when neither answers. */
async function networks() {
  const {data, local} = await json('strix/networks.json');
  return data.networks.map(network => ({...network, path: `strix/${network.file}`, local}));
}

const engine = new StrixEngine([]);

/** The browser engine for seat.mjs; its checkpoints fill in once strix/networks.json is read. */
export const strix = {
  entry: {id: ID, kind: 'strix', name: LABEL, label: LABEL, checkpoints: engine.checkpoints, presets: PRESETS, analysis: true},
  engine,
  listed: engine.known().then(() => {}, () => {}),
  build: 'python tools/build_web.py strix-network',
  record: result => ({...result, proof: null, line: [], threat: [], engine: ID}),
};
