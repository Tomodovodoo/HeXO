/* Strix in a Web Worker (strix/core.mjs). A turn runs synchronously, so the page cancels it by terminating the worker.
 * In:  {type: 'load', network: {url, sha256}} | {type: 'turn', id, history, simulations, analysis}.
 * Out: {type: 'progress', id?, fraction} | {type: 'ready', info} | {type: 'result', id, result} | {type: 'error', id?, message}.
 */
import {loadStrix} from './strix/core.mjs';
import {cached} from './network.mjs';

let strix = null, current = null;

async function load({url, sha256}) {
  const [engine, weights] = await Promise.all([
    loadStrix(new URL('strix/strix.wasm', import.meta.url).href, {progress: fraction => postMessage({type: 'progress', id: current, fraction})}),
    cached(new URL(url, import.meta.url).href, sha256, fraction => postMessage({type: 'progress', fraction}))]);
  const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', weights))].map(b => b.toString(16).padStart(2, '0')).join('');
  if (digest !== sha256) throw new Error(`Strix network ${url} does not match its SHA-256`);
  const info = engine.load(weights);
  strix = engine;
  return {source_checkpoint: info.source_checkpoint, build: engine.build_hash};
}

onmessage = async ({data}) => {
  try {
    if (data.type === 'load') postMessage({type: 'ready', info: await load(data.network)});
    else if (data.type === 'turn') {
      current = data.id;
      const start = performance.now(), result = strix.turn(data.history, data.simulations, {analysis: data.analysis});
      postMessage({type: 'result', id: data.id, result: {...result, ms: Math.round(performance.now() - start)}});
    }
  } catch (error) {
    postMessage({type: 'error', id: data.id, message: String(error.message || error)});
  }
};
