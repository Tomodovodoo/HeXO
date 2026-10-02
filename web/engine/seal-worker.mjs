/* Seal in a Web Worker. Its search blocks the worker for the turn's ms (Seal's clock is performance.now()), so a
 * cancel ends the worker from the page (seal.mjs).
 * In: {type: 'load'} | {type: 'turn', id, history, ms}.
 * Out: {type: 'progress', fraction} | {type: 'ready', revision} | {type: 'result', id, result} | {type: 'error', id?, message}.
 */
import createModule from './seal/engine.mjs';
import {sealTurn} from './seal.mjs';

const CACHE = 'seal-engine-v1';
let module = null;

const sha256 = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
  b => b.toString(16).padStart(2, '0')).join('');

/** seal/engine.wasm with SHA-256 `digest`: from the Cache API, else fetched by that digest and kept once it matches. */
async function wasm(digest) {
  const url = new URL(`seal/engine.wasm?v=${digest}`, import.meta.url);
  let store = null;
  try { store = await caches.open(CACHE); } catch {}
  const hit = await store?.match(url);
  if (hit) {
    const bytes = await hit.arrayBuffer();
    if (await sha256(bytes) === digest) return bytes;
  }
  const response = await fetch(url, {cache: 'no-store'});
  if (!response.ok) throw new Error(`seal/engine.wasm: ${response.status}`);
  const bytes = await response.arrayBuffer();
  if (await sha256(bytes) !== digest) throw new Error('seal/engine.wasm does not match seal/manifest.json');
  if (store) {
    for (const old of await store.keys()) await store.delete(old);
    await store.put(url, new Response(bytes.slice(0)));
  }
  return bytes;
}

async function load() {
  const manifest = await (await fetch(new URL('seal/manifest.json', import.meta.url), {cache: 'no-store'})).json();
  module = await createModule({wasmBinary: await wasm(manifest.sha256)});
  postMessage({type: 'progress', fraction: 1});
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
