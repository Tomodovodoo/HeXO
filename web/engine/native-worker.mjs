/* Native in a Web Worker. A search runs to its end inside the module, so the page cancels a turn by ending this worker.
 * In: {type: 'load'} | {type: 'turn', id, history, ms, depth}.
 * Out: {type: 'progress', fraction} | {type: 'ready'} | {type: 'result', id, result} | {type: 'error', id?, message}.
 */
import {NativeSearch} from './native/search.mjs';
import {cached} from './network.mjs';

let native;

/** native.wasm from the Cache API under the digest web/engine/build.json records for it. */
async function load() {
  const record = await (await fetch(new URL('build.json', import.meta.url), {cache: 'no-cache'})).json();
  const url = new URL('native/native.wasm', import.meta.url).href;
  const wasmBinary = await cached(url, record.artefacts['native/native.wasm'], fraction => postMessage({type: 'progress', fraction}));
  native = await NativeSearch.create({wasmBinary});
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
