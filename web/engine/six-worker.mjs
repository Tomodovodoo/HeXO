/* Six (browser) in a Web Worker: Six's search (six/search.mjs) with a network from six/networks under ONNX Runtime
 * Web, WebGPU when the device has it, else WebAssembly. The protocol is engine-worker.mjs's:
 * In: {type: 'load', options: {threads, prefer}} | {type: 'use', id, network} | {type: 'turn', id, history, nodes, network} | {type: 'cancel', id}.
 * Out: {type: 'progress', id?, fraction} | {type: 'ready', device} | {type: 'result', id, result} | {type: 'cancelled', id}
 *     | {type: 'error', id?, message}.
 * Turns run one at a time; a cancel stops the running one at its next network batch. */
import {cached, json} from './assets.mjs';
import {probe, runtime} from './network.mjs';
import {SixSearch} from './six/search.mjs';

const WIN = 1000000;
/** A one-node graph (Identity on one float) whose session starts the runtime before any network is chosen. */
const START = Uint8Array.from(atob('CAgSADo7ChAKAXgSAXkiCElkZW50aXR5EgVzdGFydFoPCgF4EgoKCAgBEgQKAggBYg8KAXkSCgoICAESBAoCCAFCBAoAEBE='), c => c.charCodeAt(0));
const cancelled = new Set();
let ort = null, device = null, settings = {}, manifest = null, search = null, current = null, running = null;
let queue = Promise.resolve();

class Cancelled extends Error {}

/** Starts ONNX Runtime on `device.provider` with a one-node session and a new search, holding no network. */
async function start(report = () => {}) {
  ort = await runtime(device.provider, settings.threads, report);
  const session = await ort.InferenceSession.create(START, {executionProviders: [device.provider], logSeverityLevel: 3});
  await session.release();
  search = await SixSearch.create(ort);
  current = null;
}

/** Moves to WebAssembly after WebGPU failed with `error`, unless WebGPU was asked for; otherwise throws `error`. */
async function fallback(error) {
  if (device.provider !== 'webgpu' || settings.prefer) throw error;
  device = {provider: 'wasm', adapter: '', fallback: String(error.message || error)};
  await start();
}

/** Searches with network `name` (a manifest entry) from the next turn on; `report(fraction)` follows its download.
 * Only the network in use is held. A graph that WebGPU cannot run moves the engine to WebAssembly. */
async function use(name, report = () => {}) {
  if (current?.name === name) return;
  const entry = manifest.data.networks.find(n => n.name === name);
  if (!entry) throw new Error(`Six has no network ${name}`);
  const bytes = await cached({path: `six/networks/${entry.file}`, sha256: entry.sha256, bytes: entry.bytes}, report);
  const create = () => ort.InferenceSession.create(new Uint8Array(bytes), {executionProviders: [device.provider],
    graphOptimizationLevel: 'all', logSeverityLevel: 3});
  let session;
  try {
    session = await create();
  } catch (error) {
    await fallback(error);
    session = await create();
  }
  const old = current;
  current = {name, session};
  search.use(session);
  await old?.session.release();
}

async function load(options = {}) {
  settings = options;
  manifest = await json('six/networks/manifest.json');
  device = await probe(options.prefer);
  try {
    await start(fraction => postMessage({type: 'progress', fraction: .95 * fraction}));
  } catch (error) {
    await fallback(error);
  }
  postMessage({type: 'progress', fraction: 1});
  return {provider: device.provider, adapter: device.adapter, fallback: device.fallback, threads: ort.env.wasm.numThreads,
    isolated: Boolean(globalThis.crossOriginIsolated), networks: manifest.data.networks.map(n => n.name)};
}

/** Six's turn at `history` within `nodes` new positions, with the fields of python/play.py evaluate: its stones as
 * `moves` and `line`, the first as the one `top` row, the mover's win probability from its score (1 when its threat
 * solver proved a win, whose distance Six does not report, so there is no `proof`) and the positions searched. The
 * network (the newest when null) is fetched on its first turn; progress follows that download, then the search. */
async function turn({id, history, nodes, network, ms = 0}) {
  if (cancelled.has(id)) throw new Cancelled();
  await use(network ?? manifest.data.networks[0].name, fraction => postMessage({type: 'progress', id, fraction}));
  if (cancelled.has(id)) throw new Cancelled();
  const start = performance.now(), player = history.length === 0 ? 0 : ((history.length - 1 >> 1) + 1) % 2;
  running = id;
  let result;
  try {
    result = await search.turn(history, nodes, ms, count => postMessage({type: 'progress', id, fraction: Math.min(1, count / nodes)}));
  } finally {
    running = null;
  }
  if (result.stopped) throw new Cancelled();
  const won = result.score === WIN, value = won ? 1 : Math.round((Math.max(-1, Math.min(1, result.score / 1000)) + 1) / 2 * 1e4) / 1e4;
  const [first] = result.moves;
  return {moves: result.moves, value, top: first ? [[first[0], first[1], 1, value, won ? 1 : 0]] : [],
    line: result.moves.map(([q, r]) => [q, r, player]), proof: null, threat: [], solved: false,
    ms: Math.round(performance.now() - start), network: current.name, nodes: result.nodes};
}

onmessage = ({data}) => {
  if (data.type === 'cancel') {
    cancelled.add(data.id);
    if (running === data.id) search.stop();
    return;
  }
  const task = async () => {
    try {
      if (data.type === 'load') postMessage({type: 'ready', device: await load(data.options)});
      else if (data.type === 'turn') postMessage({type: 'result', id: data.id, result: await turn(data)});
      else if (data.type === 'use') { await use(data.network ?? manifest.data.networks[0].name, fraction => postMessage({type: 'progress', id: data.id, fraction})); postMessage({type: 'result', id: data.id, result: null}); }
    } catch (error) {
      postMessage(error instanceof Cancelled ? {type: 'cancelled', id: data.id} : {type: 'error', id: data.id, message: String(error.message || error)});
    } finally {
      cancelled.delete(data.id);
    }
  };
  queue = queue.then(task);
};
