/* Strix in a Web Worker (strix/core.mjs). A turn runs synchronously, so the page cancels it by terminating the worker.
 * In:  {type: 'load', network: {url, sha256}} | {type: 'turn', id, history, simulations}.
 * Out: {type: 'progress', id?, fraction} | {type: 'ready', info} | {type: 'result', id, result} | {type: 'error', id?, message}.
 */
import {loadStrix} from './strix/core.mjs';
import {cached, forget} from './network.mjs';

let strix = null, current = null;

/** SHA-256 hex of `bytes`, with CRLF read as LF when `lines` (as tools/build_web.py's digest records files). */
async function sha256(bytes, lines = false) {
  let data = new Uint8Array(bytes);
  if (lines) data = data.filter((b, i) => !(b === 13 && data[i + 1] === 10));
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', data))].map(b => b.toString(16).padStart(2, '0')).join('');
}

/** The body of `url` from the Cache API under `version`, checked against `digest`; bytes that fail are dropped from
 * the cache and fail the load. */
async function verified(url, version, digest, lines, progress) {
  const bytes = await cached(url, version, progress);
  if (await sha256(bytes, lines) === digest) return bytes;
  await forget(url);
  throw new Error(`${url} does not match its SHA-256`);
}

/** strix.wasm (checked against the digest web/engine/build.json records) and the network (against its SHA-256). */
async function load({url, sha256: network}) {
  const record = await (await fetch(new URL('build.json', import.meta.url), {cache: 'no-cache'})).json();
  const parts = [0, 0], report = i => fraction => { parts[i] = fraction; postMessage({type: 'progress', fraction: (parts[0] + parts[1]) / 2}); };
  const wasmDigest = record.artefacts['strix/strix.wasm'];
  const [wasm, weights] = await Promise.all([
    verified(new URL('strix/strix.wasm', import.meta.url).href, wasmDigest, wasmDigest, true, report(0)),
    verified(new URL(url, import.meta.url).href, network, network, false, report(1))]);
  const engine = await loadStrix(wasm, {progress: fraction => postMessage({type: 'progress', id: current, fraction})});
  const info = engine.load(weights);
  strix = engine;
  return {source_checkpoint: info.source_checkpoint};
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
    postMessage({type: 'error', id: data.id, message: String(error.message || error)});
  }
};
