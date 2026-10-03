/* Shrimp in a Web Worker: the network (shrimp/network.mjs) and the search (shrimp/shrimp.wasm via shrimp/search.mjs).
 * In: {type: 'load', options} | {type: 'turn', id, history, visits} | {type: 'cancel', id}.
 * Out: {type: 'progress', id?, fraction, stage?} | {type: 'ready', device} | {type: 'result', id, result} | {type: 'cancelled', id}
 *     | {type: 'error', id?, message, stage?}: stages.mjs's loading stages; a stage in an error is where it stopped. */
import {cached, json} from './assets.mjs';
import {probe} from './network.mjs';
import {Stages, errorReport, stall} from './stages.mjs';
import {ShrimpNetwork} from './shrimp/network.mjs';
import {Cancelled, ShrimpSearch, loadModule} from './shrimp/search.mjs';

let network, module, profile, queue = Promise.resolve();
const searches = new Map(), cancelled = new Set();

/** One search engine per visit budget, as the server runs one driver process per preset: each counts its own games. */
async function engine(visits) {
  if (!searches.has(visits)) searches.set(visits, await ShrimpSearch.create(module, profile));
  return searches.get(visits);
}

/** Turns run one at a time: they share the network session. */
function turn(data) {
  const run = queue.then(() => play(data));
  queue = run.catch(() => {});
  return run;
}

async function play({id, history, visits}) {
  if (cancelled.has(id)) throw new Cancelled();
  const start = performance.now(), search = await engine(visits);
  const before = {batches: network.batches, ms: network.ms};
  let result;
  try {
    result = await search.turn(history, visits, {evaluate: rows => network.evaluate(rows), stop: () => cancelled.has(id),
      progress: fraction => postMessage({type: 'progress', id, fraction})});
  } catch (error) {
    if (error instanceof Cancelled) searches.delete(visits);   // the server restarts a driver whose search it stopped
    throw error;
  }
  return {...result, ms: Math.round(performance.now() - start), batches: network.batches - before.batches,
    network_ms: Math.round(network.ms - before.ms)};
}

/** Probes the device, starts ONNX Runtime with the network and the search, and plays one visit, reporting each stage. */
async function load(options = {}) {
  const stages = new Stages(postMessage);
  return stages.run(async () => {
    stages.enter('probe');
    const build = (await json('build.json')).data, device = await probe(options.prefer);
    stages.probed(device);
    const file = {path: 'shrimp/shrimp.wasm', sha256: build.artefacts['shrimp/shrimp.wasm'], lines: true};
    const wasm = cached(file, stages.file(file.path)).then(loadModule);
    wasm.catch(() => {});
    network = await ShrimpNetwork.create({device, stages, threads: options.threads});
    module = await wasm;
    profile = network.manifest.search;
    stages.enter('warmup', device.provider);
    const t = performance.now();
    const warm = await engine(16);
    await stall('warmup', () => warm.turn([], 1, {evaluate: rows => network.evaluate(rows)}));
    searches.clear();
    return {provider: network.provider, adapter: device.adapter, threads: network.threads,
      isolated: Boolean(globalThis.crossOriginIsolated), warmup_ms: Math.round(performance.now() - t), model: network.version};
  });
}

onmessage = async ({data}) => {
  if (data.type === 'cancel') {
    cancelled.add(data.id);
    return;
  }
  try {
    if (data.type === 'load') postMessage({type: 'ready', device: await load(data.options)});
    else if (data.type === 'turn') postMessage({type: 'result', id: data.id, result: await turn(data)});
  } catch (error) {
    postMessage(error instanceof Cancelled ? {type: 'cancelled', id: data.id} : errorReport(error, data.id));
  } finally {
    cancelled.delete(data.id);
  }
};
