/* Drip in a Web Worker. A search runs to its end inside the module, so the page cancels a turn by ending this worker.
 * In: {type: 'load'} | {type: 'turn', id, history, ms, depth}.
 * Out: {type: 'progress', fraction, stage} | {type: 'ready'} | {type: 'result', id, result} | {type: 'error', id?, message, stage?}
 * (stages.mjs's download and compile).
 */
import {DripSearch} from './native/search.mjs';
import {cached, wasmOptions} from './assets.mjs';
import {files} from './drip.mjs';
import {Stages, errorReport} from './stages.mjs';

let drip;

/** native.wasm through assets.mjs, checked against the digest web/engine/build.json records for it. */
async function load() {
  const stages = new Stages(postMessage);
  await stages.run(async () => {
    stages.enter('download');
    const [wasm] = await files(), bytes = await cached(wasm, stages.file(wasm.path));
    stages.enter('compile');
    drip = await DripSearch.create(wasmOptions(bytes));
  });
}

onmessage = async ({data}) => {
  try {
    if (data.type === 'load') {
      await load();
      postMessage({type: 'ready'});
    } else if (data.type === 'turn') {
      postMessage({type: 'result', id: data.id, result: drip.turn(data.history, data.ms, data.depth)});
    }
  } catch (error) {
    postMessage(errorReport(error, data.id));
  }
};
