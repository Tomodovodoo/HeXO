/* Strix in a Web Worker (strix/core.mjs). A turn runs synchronously, so the page cancels it by terminating the worker.
 * In:  {type: 'load', network: {path, sha256, bytes}} | {type: 'turn', id, history, simulations}.
 * Out: {type: 'progress', id?, fraction, stage?} | {type: 'ready', info} | {type: 'result', id, result}
 *     | {type: 'error', id?, message, stage?} (stages.mjs's download and compile while loading).
 */
import {loadStrix} from './strix/core.mjs';
import {cached, json} from './assets.mjs';
import {Stages, errorReport} from './stages.mjs';

let strix = null, current = null;

/** strix.wasm (checked against the digest web/engine/build.json records) and `network` ({path, sha256}). */
async function load(network) {
  const stages = new Stages(postMessage);
  return stages.run(async () => {
    const wasmFile = {path: 'strix/strix.wasm', sha256: (await json('build.json')).data.artefacts['strix/strix.wasm'], lines: true};
    const [wasm, weights] = await Promise.all([cached(wasmFile, stages.file(wasmFile.path)), cached(network, stages.file(network.path))]);
    stages.enter('compile');
    const engine = await loadStrix(wasm, {progress: fraction => postMessage({type: 'progress', id: current, fraction})});
    const info = engine.load(weights);
    strix = engine;
    return {source_checkpoint: info.source_checkpoint};
  });
}
onmessage = async ({data}) => {
  try {
    if (data.type === 'load') postMessage({type: 'ready', info: await load(data.network)});
    else if (data.type === 'turn') {
      current = data.id;
      const start = performance.now(), result = strix.turn(data.history, data.simulations);
      postMessage({type: 'result', id: data.id, result: {...result, ms: Math.round(performance.now() - start)}});
    }
  } catch (error) {
    postMessage(errorReport(error, data.id));
  }
};
