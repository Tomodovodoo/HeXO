/* Strix in a Web Worker (strix/core.mjs). A turn runs synchronously, so the page cancels it by terminating the worker.
 * In:  {type: 'load', network: {url, sha256}} | {type: 'turn', id, history, simulations}.
 * Out: {type: 'progress', id?, fraction} | {type: 'ready', info} | {type: 'result', id, result} | {type: 'error', id?, message}.
 */
import {loadStrix} from './strix/core.mjs';
import {cached} from './network.mjs';

let strix = null, current = null;

/** strix.wasm (under the digest web/engine/build.json records for it) and the network (under its SHA-256), both from
 * the Cache API; the network's bytes must match its SHA-256. */
async function load({url, sha256}) {
  const record = await (await fetch(new URL('build.json', import.meta.url), {cache: 'no-cache'})).json();
  const parts = [0, 0], report = i => fraction => { parts[i] = fraction; postMessage({type: 'progress', fraction: (parts[0] + parts[1]) / 2}); };
  const [wasm, weights] = await Promise.all([
    cached(new URL('strix/strix.wasm', import.meta.url).href, record.artefacts['strix/strix.wasm'], report(0)),
    cached(new URL(url, import.meta.url).href, sha256, report(1))]);
  const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', weights))].map(b => b.toString(16).padStart(2, '0')).join('');
  if (digest !== sha256) throw new Error(`Strix network ${url} does not match its SHA-256`);
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
