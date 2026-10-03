/* Native in a Web Worker. A search runs to its end inside the module, so the page cancels a turn by ending this worker.
 * In: {type: 'load'} | {type: 'turn', id, history, ms, depth}.
 * Out: {type: 'progress', fraction} | {type: 'ready'} | {type: 'result', id, result} | {type: 'error', id?, message}.
 */
import {NativeSearch} from './native/search.mjs';
import {cached, wasmOptions} from './assets.mjs';
import {files} from './native.mjs';

let native;

/** native.wasm through assets.mjs, checked against the digest web/engine/build.json records for it. */
async function load() {
  const [wasm] = await files();
  native = await NativeSearch.create(wasmOptions(await cached(wasm, fraction => postMessage({type: 'progress', fraction}))));
}

onmessage = async ({data}) => {
  try {
    if (data.type === 'load') {
      await load();
      postMessage({type: 'ready'});
    } else if (data.type === 'turn') {
      postMessage({type: 'result', id: data.id, result: native.turn(data.history, data.ms, data.depth)});
    }
  } catch (error) {
    postMessage({type: 'error', id: data.id, message: String(error.message || error)});
  }
};
