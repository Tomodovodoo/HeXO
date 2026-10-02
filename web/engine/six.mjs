/* "Six (browser)": Six's network search in six-worker.mjs, for the play page's browser engines (seat.mjs). It plays
 * as the server's Six: the same presets in Six protocol nodes, with the networks this site was built with. */
import {EngineWorker} from './engine-worker.mjs';

export const PRESETS = {lightning: {nodes: 1500}, quick: {nodes: 6000}, standard: {nodes: 30000}, strong: {nodes: 135000},
  deep: {nodes: 500000}, dangerous: {nodes: 2000000}};
const ID = 'browser:six', LABEL = 'Six (browser)';

export class SixEngine extends EngineWorker {
  /** `prefer` 'wasm' keeps the network off WebGPU; `threads` fixes ONNX Runtime's WebAssembly thread count. */
  constructor({prefer = null, threads = null} = {}) {
    super(new URL('six-worker.mjs', import.meta.url), LABEL, {prefer, threads});
  }

  /** Six's turn at `history` ([[q, r], ...]) within `budget.nodes` new positions, with network `budget.checkpoint` (a
   * manifest name; the newest when absent): the fields of python/play.py evaluate. Aborting `options.signal` cancels
   * it at the search's next network batch and rejects with an AbortError. */
  turn(history, budget, options = {}) {
    return this.call({type: 'turn', history, nodes: budget.nodes, network: budget.checkpoint ?? null}, options);
  }
}

/** The networks this site was built with (python tools/build_web.py six), newest first; none without the build. */
async function networks() {
  try {
    const response = await fetch(new URL('six/networks/manifest.json', import.meta.url), {cache: 'no-cache'});
    return response.ok ? (await response.json()).networks.map(n => n.name) : [];
  } catch {
    return [];
  }
}

const checkpoints = await networks();

/** Six (browser) for seat.mjs's ENGINES, or null when the site has no Six networks. */
export const six = checkpoints.length ? {
  entry: {id: ID, kind: 'six', name: LABEL, label: LABEL, checkpoints, presets: PRESETS, analysis: true},
  engine: new SixEngine(),
  record: (result, history, preset) => ({...result, engine: ID}),
} : null;
