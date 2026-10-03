/* "Six (browser)": Six's network search in six-worker.mjs, for the play page's browser engines (seat.mjs). It plays
 * as the server's Six: the same presets in Six protocol nodes, with the networks this site was built with. */
import {EngineWorker} from './engine-worker.mjs';
import {json, workerUrl} from './assets.mjs';
import {probe, runtimeFiles} from './network.mjs';

export const PRESETS = {lightning: {nodes: 1500}, quick: {nodes: 6000}, standard: {nodes: 30000}, strong: {nodes: 135000},
  deep: {nodes: 500000}, dangerous: {nodes: 2000000}};
const ID = 'browser:six', LABEL = 'Six (browser)', MANIFEST = 'six/networks/manifest.json';

export class SixEngine extends EngineWorker {
  /** `prefer` 'wasm' keeps the network off WebGPU; `threads` fixes ONNX Runtime's WebAssembly thread count.
   * `checkpoints` holds the manifest's network names, newest first, refreshed whenever files() reads it. */
  constructor({prefer = null, threads = null} = {}) {
    super(workerUrl('six-worker.mjs'), LABEL, {prefer, threads});
    this.checkpoints = [];
  }

  /** Six's turn at `history` ([[q, r], ...]) within `budget.nodes` new positions, with network `budget.checkpoint` (a
   * manifest name; the newest when absent): the fields of python/play.py evaluate. Aborting `options.signal` cancels
   * it at the search's next network batch and rejects with an AbortError. */
  turn(history, budget, options = {}) {
    return this.call({type: 'turn', history, nodes: budget.nodes, network: budget.checkpoint ?? null}, options);
  }

  /** {data, local}: six/networks/manifest.json (assets.mjs json()), whose network names refresh `checkpoints`. */
  async manifest() {
    const found = await json(MANIFEST);
    this.checkpoints.splice(0, Infinity, ...found.data.networks.map(n => n.name));
    return found;
  }

  /** The downloaded files (assets.mjs records) a first turn on this device reads: ONNX Runtime and the newest network. */
  async files() {
    const [{provider}, {data, local}] = await Promise.all([probe(this.options.prefer), this.manifest()]), [newest] = data.networks;
    return [...await runtimeFiles(provider), {path: `six/networks/${newest.file}`, sha256: newest.sha256, bytes: newest.bytes, local}];
  }
}

const engine = new SixEngine();
await engine.manifest().catch(() => {});

/** Six (browser) for seat.mjs's ENGINES; its checkpoints fill in once the manifest is read (python tools/build_web.py
 * six builds it; the site's serves when this origin has none). */
export const six = {
  entry: {id: ID, kind: 'six', name: LABEL, label: LABEL, checkpoints: engine.checkpoints, presets: PRESETS, analysis: true},
  engine,
  record: (result, history, preset) => ({...result, engine: ID}),
  build: 'python tools/build_web.py ort six',
};
