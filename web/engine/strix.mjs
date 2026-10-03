/* "Strix (browser)": Strix's network and Gumbel search (tools/strix_web as strix/strix.wasm) in strix-worker.mjs, for
 * the play page's browser engines (seat.mjs). It plays as python/play.py's Strix: the same presets in simulations per
 * placement, with a network `python tools/build_web.py strix-network` placed in strix/ (listed in strix/networks.json,
 * the first is the default), or the site's when this origin has none. */
import {json, workerUrl} from './assets.mjs';

import {NEURAL_PRESET} from './device.mjs';

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
    this.pending = new Set();   // the reject functions of the running load and turn
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

  /** Reads strix/networks.json again and replaces the choices, keeping the current network's id; a worker whose
   * network changed bytes ends, and the next load starts one on the new file. Throws when neither origin answers. */
  async refresh() {
    const list = await networks(), before = this.network;
    this.networks = new Map();
    this.network = null;
    this.use(list);
    this.network = this.networks.get(before?.id) ?? this.network;
    if (before && this.network?.sha256 !== before.sha256) this.close();
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
      const fail = error => { this.pending.delete(fail); reject(error); };
      this.pending.add(fail);
      worker.onmessage = ({data}) => {
        if (data.type === 'progress' && data.id === undefined) progress(data.fraction);
        else if (data.type === 'ready') { this.pending.delete(fail); resolve(data.info); }
        else if (data.type === 'error') fail(new Error(data.message));
      };
      worker.onerror = event => fail(new Error(event.message || 'Strix worker failed'));
    });
    ready.catch(() => { if (this.ready === ready) this.close(); });
    worker.postMessage({type: 'load', network: {path: this.network.path, sha256: this.network.sha256, bytes: this.network.size}});
    return ready;
  }

  /** The downloaded files (assets.mjs records) loads read: strix.wasm and the networks with ids in `checkpoints` (the
   * current one when none of them is known). */
  async files(checkpoints = []) {
    await this.refresh();
    const {data, local} = await json('build.json'), chosen = checkpoints.map(id => this.networks.get(id)).filter(Boolean);
    return [{path: 'strix/strix.wasm', sha256: data.artefacts['strix/strix.wasm'], lines: true, local},
      ...(chosen.length ? chosen : [this.network]).map(n => ({path: n.path, sha256: n.sha256, bytes: n.size, local: n.local}))];
  }

  /** Ends the worker and rejects its running load and turn with an AbortError; the next load starts a new one. */
  close() {
    this.worker?.terminate();
    this.worker = this.ready = null;
    const pending = [...this.pending];
    this.pending.clear();
    for (const reject of pending) reject(new DOMException('Closed', 'AbortError'));
  }

  /**
   * Strix's turn after `history` ([[q, r], ...]) at `budget.simulations` per placement with network `budget.checkpoint`
   * (the default when absent): {moves, value, top, simulations, eval_states, root_visits, ms}. Aborting `signal` ends
   * the worker (a search cannot be interrupted inside it) and rejects with an AbortError; the next call starts a new one.
   */
  async turn(history, budget, {signal, progress = () => {}} = {}) {
    await this.known();   // the requested network may be in a list that has not arrived yet
    const network = this.networks.get(budget.checkpoint) ?? this.networks.values().next().value ?? null;
    if (network !== this.network) {
      this.close();
      this.network = network;
    }
    await this.load();
    const worker = this.worker, id = ++this.calls;
    return new Promise((resolve, reject) => {
      const fail = error => { this.pending.delete(fail); signal?.removeEventListener('abort', abort); reject(error); };
      const abort = () => { fail(new DOMException('Cancelled', 'AbortError')); if (this.worker === worker) this.close(); };
      if (signal?.aborted) { abort(); return; }
      this.pending.add(fail);
      signal?.addEventListener('abort', abort, {once: true});
      worker.onmessage = ({data}) => {
        if (data.id !== id) return;
        if (data.type === 'progress') { progress(data.fraction); return; }
        if (data.type !== 'result') { fail(new Error(data.message)); return; }
        this.pending.delete(fail);
        signal?.removeEventListener('abort', abort);
        resolve(data.result);
      };
      worker.onerror = event => { fail(new Error(event.message || 'Strix worker failed')); if (this.worker === worker) this.close(); };
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
  entry: {id: ID, kind: 'strix', name: LABEL, label: LABEL, checkpoints: engine.checkpoints, presets: PRESETS, preset: NEURAL_PRESET, analysis: true, clocks: false},
  engine,
  listed: engine.known().then(() => {}, () => {}),
  build: 'python tools/build_web.py strix-network',
  record: result => ({...result, proof: null, line: [], threat: [], engine: ID}),
};
