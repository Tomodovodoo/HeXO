/* "Six (browser)": Six's network search in six-worker.mjs, for the play page's browser engines (seat.mjs). It plays
 * as the server's Six: the same presets in Six protocol nodes, with the networks this site was built with. */
import {EngineWorker} from './engine-worker.mjs';
import {json, workerUrl} from './assets.mjs';
import {probe, runtimeFiles} from './network.mjs';

export const PRESETS = {lightning: {nodes: 1500}, quick: {nodes: 6000}, standard: {nodes: 30000}, strong: {nodes: 135000},
  deep: {nodes: 500000}, dangerous: {nodes: 2000000}};
const ID = 'browser:six', LABEL = 'Six (browser)', MANIFEST = 'six/networks/manifest.json';

export class SixEngine extends EngineWorker {
  /** `prefer` 'wasm' keeps the network off WebGPU; `threads` fixes ONNX Runtime's WebAssembly thread count. */
  constructor({prefer = null, threads = null} = {}) {
    super(workerUrl('six-worker.mjs'), LABEL, {prefer, threads});
  }

  /** Six's turn at `history` ([[q, r], ...]) within `budget.nodes` new positions, with network `budget.checkpoint` (a
   * manifest name; the newest when absent): the fields of python/play.py evaluate. Aborting `options.signal` cancels
   * it at the search's next network batch and rejects with an AbortError. */
  turn(history, budget, options = {}) {
    return this.call({type: 'turn', history, nodes: budget.nodes, network: budget.checkpoint ?? null}, options);
  }

  /** The downloaded files (assets.mjs records) a first turn on this device reads: ONNX Runtime and the newest network. */
  async files() {
    const [{provider}, {data, local}] = await Promise.all([probe(this.options.prefer), json(MANIFEST)]), [newest] = data.networks;
    return [...await runtimeFiles(provider), {path: `six/networks/${newest.file}`, sha256: newest.sha256, bytes: newest.bytes, local}];
  }
}

/** The networks this site was built with (python tools/build_web.py six), newest first, from the site when this origin
 * has none; none when neither answers. */
async function networks() {
  try {
    return (await json(MANIFEST)).data.networks.map(n => n.name);
  } catch {
    return [];
  }
}

const checkpoints = await networks();

/** Six (browser) for seat.mjs's ENGINES. */
export const six = {
  entry: {id: ID, kind: 'six', name: LABEL, label: LABEL, checkpoints, presets: PRESETS, analysis: true},
  engine: new SixEngine(),
  record: (result, history, preset) => ({...result, engine: ID}),
};
