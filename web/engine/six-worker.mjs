/* Six (browser) in a Web Worker: Six's search (six/search.mjs) with a network from six/networks under ONNX Runtime
 * Web, WebGPU when the device has it, else WebAssembly. The protocol is engine-worker.mjs's:
 * In: {type: 'load', options: {threads, prefer}} | {type: 'turn', id, history, nodes, network} | {type: 'cancel', id}.
 * Out: {type: 'progress', id?, fraction} | {type: 'ready', device} | {type: 'result', id, result} | {type: 'cancelled', id}
 *     | {type: 'error', id?, message}.
 * Turns run one at a time; a cancel stops the running one at its next network batch. */
import {cached, probe, defaultThreads} from './network.mjs';
import {SixSearch} from './six/search.mjs';

const BASE = new URL('./', import.meta.url), WIN = 1000000;
const cancelled = new Set();
let ort = null, device = null, manifest = null, search = null, current = null, running = null, queue = Promise.resolve();

class Cancelled extends Error {}

/** ONNX Runtime Web for `provider`, its WebAssembly binary from the Cache API; `report(fraction)` follows the download. */
async function runtime(provider, threads, report) {
  const ortBase = new URL('ort/', BASE), version = (await (await fetch(new URL('version.json', ortBase))).json()).version;
  const gpu = provider === 'webgpu', binary = gpu ? 'ort-wasm-simd-threaded.asyncify' : 'ort-wasm-simd-threaded';
  const [module, wasmBinary] = await Promise.all([import(new URL(gpu ? 'ort.webgpu.min.mjs' : 'ort.wasm.min.mjs', ortBase).href),
    cached(new URL(`${binary}.wasm`, ortBase).href, version, report)]);
  module.env.wasm.wasmPaths = {mjs: new URL(`${binary}.mjs`, ortBase).href};
  module.env.wasm.wasmBinary = wasmBinary;
  module.env.wasm.numThreads = threads ?? defaultThreads({isolated: Boolean(globalThis.crossOriginIsolated), cores: navigator.hardwareConcurrency || 2});
  module.env.wasm.proxy = false;
  module.env.logLevel = 'error';
  return module;
}

/** Searches with network `name` (a manifest entry) from the next turn on; `report(fraction)` follows its download. */
async function use(name, report = () => {}) {
  if (current?.name === name) return;
  const entry = manifest.networks.find(n => n.name === name);
  if (!entry) throw new Error(`Six has no network ${name}`);
  const bytes = await cached(new URL(`six/networks/${entry.file}`, BASE).href, entry.sha256, report);
  const session = await ort.InferenceSession.create(new Uint8Array(bytes), {executionProviders: [device.provider],
    graphOptimizationLevel: 'all', logSeverityLevel: 3});
  const old = current;
  current = {name, session};
  search.use(session);
  await old?.session.release();
}

async function load(options = {}) {
  manifest = await (await fetch(new URL('six/networks/manifest.json', BASE), {cache: 'no-cache'})).json();
  const shares = [0, 0], report = i => f => { shares[i] = f; postMessage({type: 'progress', fraction: .95 * (.3 * shares[0] + .7 * shares[1])}); };
  const start = async () => {
    ort = await runtime(device.provider, options.threads, report(0));
    search ??= await SixSearch.create(ort);
    current = null;
    await use(manifest.networks[0].name, report(1));
  };
  device = await probe(options.prefer);
  try {
    await start();
  } catch (error) {
    if (device.provider !== 'webgpu' || options.prefer) throw error;
    device = {provider: 'wasm', adapter: '', fallback: String(error.message || error)};
    search = null;
    await start();
  }
  postMessage({type: 'progress', fraction: 1});
  return {provider: device.provider, adapter: device.adapter, fallback: device.fallback, threads: ort.env.wasm.numThreads,
    isolated: Boolean(globalThis.crossOriginIsolated), networks: manifest.networks.map(n => n.name)};
}

/** Six's turn at `history` within `nodes` new positions, with the fields of python/play.py evaluate: its stones as
 * `moves` and as `top` rows, the mover's win probability from its score, and a proof when it found a forced win. */
async function turn({id, history, nodes, network}) {
  await use(network ?? manifest.networks[0].name);
  if (cancelled.has(id)) throw new Cancelled();
  const start = performance.now(), player = history.length === 0 ? 0 : ((history.length - 1 >> 1) + 1) % 2;
  running = id;
  let result;
  try {
    result = await search.turn(history, nodes, count => postMessage({type: 'progress', id, fraction: Math.min(1, count / nodes)}));
  } finally {
    running = null;
  }
  if (result.stopped) throw new Cancelled();
  const won = result.score >= WIN - 1000, value = won ? 1 : Math.round((Math.max(-1, Math.min(1, result.score / 1000)) + 1) / 2 * 1e4) / 1e4;
  return {moves: result.moves, value, top: result.moves.map(([q, r]) => [q, r, 1, value, won ? 1 : 0]),
    proof: won ? {winner: player, turns: Math.max(1, WIN - result.score)} : null, line: [], threat: [], solved: false,
    ms: Math.round(performance.now() - start), network: current.name, nodes};
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
    } catch (error) {
      postMessage(error instanceof Cancelled ? {type: 'cancelled', id: data.id} : {type: 'error', id: data.id, message: String(error.message || error)});
    } finally {
      cancelled.delete(data.id);
    }
  };
  queue = queue.then(task);
};
