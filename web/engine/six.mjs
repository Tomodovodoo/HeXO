/* "Six (browser)": Six's network search in six-worker.mjs, for the play page's browser engines (seat.mjs). It plays
 * as the server's Six: the same presets in Six protocol nodes, with the networks this site was built with. */
import {EngineWorker} from './engine-worker.mjs';
import {json, workerUrl} from './assets.mjs';
import {loadFiles, probe} from './network.mjs';
import {NEURAL_PRESET} from './device.mjs';

export const PRESETS = {lightning: {nodes: 240}, quick: {nodes: 960}, standard: {nodes: 3840}, strong: {nodes: 15360},
  deep: {nodes: 61440}, dangerous: {nodes: 2000000}};
const ID = 'browser:six', LABEL = 'Six (browser)', MANIFEST = 'six/networks/manifest.json';

export class SixEngine extends EngineWorker {
  /** `prefer` 'wasm' keeps the network off WebGPU; `threads` fixes ONNX Runtime's WebAssembly thread count.
   * `checkpoints` holds the manifest's network names, newest first, refreshed whenever files() reads it. */
  constructor({prefer = null, threads = null} = {}) {
    super(workerUrl('six-worker.mjs'), LABEL, {prefer, threads});
    this.checkpoints = [];
    this.networks = null;
  }

  /** Loads network `checkpoint` (the newest when null), so a timed turn does not spend its clock on it. */
  prepare(checkpoint = null, options = {}) {
    return this.call({type: 'use', network: checkpoint}, options);
  }

  /** Six's turn at `history` ([[q, r], ...]) within `budget.nodes` new positions, with network `budget.checkpoint` (a
   * manifest name; the newest when absent): the fields of python/play.py evaluate. Under a clock `options.ms` is Six's
   * movetime, the nodes a ceiling. Aborting `options.signal` cancels it at the search's next network batch and rejects
   * with an AbortError. */
  turn(history, budget, options = {}) {
    return this.call({type: 'turn', history, nodes: budget.nodes, network: budget.checkpoint ?? null,
      ms: options.ms == null ? 0 : Math.max(1, Math.floor(options.ms))}, options);
  }

  /** {data, local}: six/networks/manifest.json (assets.mjs json()), whose network names refresh `checkpoints`. A changed
   * list (names, files or digests) ends a running worker, which holds the manifest it loaded with; the next call starts
   * one on the new list. */
  async manifest() {
    const found = await json(MANIFEST), names = found.data.networks.map(n => n.name);
    const networks = JSON.stringify(found.data.networks.map(n => [n.name, n.file, n.sha256]));
    if (networks !== this.networks) {
      if (this.worker && this.networks) this.close();
      this.networks = networks;
      this.checkpoints.splice(0, Infinity, ...names);
    }
    return found;
  }

  /** The downloaded files (assets.mjs records) turns on this device may read: ONNX Runtime (with a WebGPU start's
   * WebAssembly fallback) and the networks named in
   * `networks` (manifest names; the newest when none of them is in the manifest). */
  async files(networks = []) {
    const [{provider}, {data, local}] = await Promise.all([probe(this.options.prefer), this.manifest()]);
    const chosen = data.networks.filter(n => networks.includes(n.name));
    return [...await loadFiles(provider, this.options.prefer), ...(chosen.length ? chosen : data.networks.slice(0, 1))
      .map(n => ({path: `six/networks/${n.file}`, sha256: n.sha256, bytes: n.bytes, local}))];
  }
}

const engine = new SixEngine();

/** Six (browser) for seat.mjs's ENGINES; its checkpoints fill in once the manifest is read (python tools/build_web.py
 * six builds it; the site's serves when this origin has none), which page startup does not wait for. */
export const six = {
  entry: {id: ID, kind: 'six', name: LABEL, label: LABEL, checkpoints: engine.checkpoints, presets: PRESETS, preset: NEURAL_PRESET, analysis: true, clocks: true},
  engine,
  listed: engine.manifest().then(() => {}, () => {}),
  record: (result, history, preset) => ({...result, engine: ID}),
  build: 'python tools/build_web.py ort six',
};
