/* Seal in a Web Worker. Its search blocks the worker for the turn's ms (Seal's clock is performance.now()), so a
 * cancel ends the worker from the page (seal.mjs).
 * In: {type: 'load'} | {type: 'turn', id, history, ms}.
 * Out: {type: 'progress', fraction, stage} | {type: 'ready', revision} | {type: 'result', id, result}
 *     | {type: 'error', id?, message, stage?} (stages.mjs's download and compile).
 */
import {cached, moduleUrl, wasmOptions} from './assets.mjs';
import {files, sealTurn} from './seal.mjs';
import {Stages, errorReport} from './stages.mjs';

let module = null;

/** Seal's module and wasm through assets.mjs (checked against seal/manifest.json when they come from the site). */
async function load() {
  const stages = new Stages(postMessage);
  return stages.run(async () => {
    const {revision, files: [glue, wasm]} = await files();
    const [{default: createModule}, wasmBinary] = await Promise.all([moduleUrl(glue, stages.file(glue.path)).then(url => import(url)),
      cached(wasm, stages.file(wasm.path))]);
    stages.enter('compile');
    module = await createModule(wasmOptions(wasmBinary));
    return revision;
  });
}

onmessage = async ({data}) => {
  try {
    if (data.type === 'load') postMessage({type: 'ready', revision: await load()});
    else if (data.type === 'turn') postMessage({type: 'result', id: data.id, result: sealTurn(module, data.history, data.ms)});
  } catch (error) {
    postMessage(errorReport(error, data.id));
  }
};
