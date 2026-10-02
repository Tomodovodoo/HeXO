/* Six running in the browser: the page-side handle of six-worker.mjs and its entry for seat.mjs. */
import {EngineWorker} from './engine-worker.mjs';

/** The Six protocol nodes of each preset, as python/play.py gives them to the server's Six. */
export const PRESETS = {lightning: {nodes: 1500}, quick: {nodes: 6000}, standard: {nodes: 30000}, strong: {nodes: 135000},
  deep: {nodes: 500000}, dangerous: {nodes: 2000000}};

export class SixEngine extends EngineWorker {
  /** `prefer` 'wasm' keeps the network off WebGPU; `threads` fixes ONNX Runtime's WebAssembly thread count. */
  constructor({prefer = null, threads = null} = {}) {
    super(new URL('six-worker.mjs', import.meta.url), 'Six (browser)', {prefer, threads});
  }

  /** Six's turn at `history` ([[q, r], ...]) within `nodes` new positions with network `network` (a manifest name, the
   * newest when null): the fields of python/play.py evaluate. Aborting `options.signal` cancels it. */
  turn(history, nodes, network = null, options = {}) {
    return this.call({type: 'turn', history, nodes, network}, options);
  }
}

/** Six (browser) for seat.mjs's engine list, with the networks the site was built with (newest first); null when
 * the site has none (python tools/build_web.py six). */
export async function browserEngine() {
  let manifest;
  try {
    const response = await fetch(new URL('six/networks/manifest.json', import.meta.url), {cache: 'no-cache'});
    if (!response.ok) return null;
    manifest = await response.json();
  } catch {
    return null;
  }
  const engine = new SixEngine(), label = 'Six (browser)', checkpoints = manifest.networks.map(n => n.name);
  return {
    entry: {id: 'browser:six', kind: 'six', name: label, label, checkpoints, presets: PRESETS},
    get device() { return engine.device; },
    load: progress => engine.load(progress),
    turn: (history, {preset, checkpoint}, options) => engine.turn(history, PRESETS[preset].nodes, checkpoint, options),
  };
}
