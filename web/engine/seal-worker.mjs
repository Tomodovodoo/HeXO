/* Seal in a Web Worker. Its search blocks the worker for the turn's ms (Seal's clock is performance.now()), so a
 * cancel ends the worker from the page (seal.mjs).
 * In: {type: 'load'} | {type: 'turn', id, history, ms}.
 * Out: {type: 'progress', fraction} | {type: 'ready', revision} | {type: 'result', id, result} | {type: 'error', id?, message}.
 */
import createModule from './seal/engine.mjs';
import {cached} from './network.mjs';
import {sealTurn} from './seal.mjs';

let module = null;

async function load() {
  const manifest = await (await fetch(new URL('seal/manifest.json', import.meta.url), {cache: 'no-store'})).json();
  const wasmBinary = await cached(new URL('seal/engine.wasm', import.meta.url).href, manifest.sha256,
    fraction => postMessage({type: 'progress', fraction}));
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', wasmBinary)), b => b.toString(16).padStart(2, '0')).join('');
  if (digest !== manifest.sha256) throw new Error('seal/engine.wasm does not match seal/manifest.json');
  module = await createModule({wasmBinary});
  return manifest.revision;
}

onmessage = async ({data}) => {
  try {
    if (data.type === 'load') postMessage({type: 'ready', revision: await load()});
    else if (data.type === 'turn') postMessage({type: 'result', id: data.id, result: sealTurn(module, data.history, data.ms)});
  } catch (error) {
    postMessage({type: 'error', id: data.id, message: String(error.message || error)});
  }
};
