/* Six (browser) in a Web Worker: Six's search (six/search.mjs) with a network from six/networks under ONNX Runtime
 * Web, WebGPU when the device has it, else WebAssembly. The protocol is engine-worker.mjs's:
 * In: {type: 'load', options: {threads, prefer}} | {type: 'use', id, network} | {type: 'turn', id, history, nodes, network} | {type: 'cancel', id}.
 * Out: {type: 'progress', id?, fraction, stage?} | {type: 'ready', device} | {type: 'result', id, result} | {type: 'cancelled', id}
 *     | {type: 'error', id?, message, stage?}: stages.mjs's loading stages; a stage in an error is where it stopped.
 * Turns run one at a time; a cancel stops the running one at its next network batch. */
import {cached, json} from './assets.mjs';
import {probe, runtime, session} from './network.mjs';
import {Stages, errorReport, stall} from './stages.mjs';
import {SixSearch} from './six/search.mjs';

const WIN = 1000000;
const cancelled = new Set();
let ort = null, device = null, manifest = null, search = null, current = null, running = null;
let queue = Promise.resolve();

class Cancelled extends Error {}

/** Searches with network `name` (a manifest entry) from the next turn on, reporting its download, session and a first
 * batch (which compiles a WebGPU session's shaders) to `stages`. Only the network in use is held. */
async function use(name, stages) {
  if (current?.name === name) return;
  const entry = manifest.data.networks.find(n => n.name === name);
  if (!entry) throw new Error(`Six has no network ${name}`);
  const created = await stages.run(async () => {
    const path = `six/networks/${entry.file}`, bytes = await cached({path, sha256: entry.sha256, bytes: entry.bytes}, stages.file(path));
    const made = await session(ort, bytes, device.provider, stages), shape = [1, search.planeCount, search.crop, search.crop];
    stages.enter('warmup', device.provider);
    const input = new ort.Tensor('float32', new Float32Array(shape.reduce((a, b) => a * b)), shape);
    for (const tensor of Object.values(await stall('warmup', () => made.run({planes: input})))) tensor.dispose?.();
    input.dispose?.();
    return made;
  });
  const old = current;
  current = {name, session: created};
  search.use(created);
  await old?.session.release();
}

/** Probes the device and starts ONNX Runtime and Six's search, holding no network yet. */
async function load(options = {}) {
  const stages = new Stages(postMessage);
  return stages.run(async () => {
    stages.enter('probe');
    manifest = await json('six/networks/manifest.json');
    device = await probe(options.prefer);
    stages.probed(device);
    ort = await runtime(device.provider, options.threads, stages);
    search = await SixSearch.create(ort);
    current = null;
    return {provider: device.provider, adapter: device.adapter, threads: ort.env.wasm.numThreads,
      isolated: Boolean(globalThis.crossOriginIsolated), networks: manifest.data.networks.map(n => n.name)};
  });
}

/** Six's turn at `history` within `nodes` new positions, with the fields of python/play.py evaluate: its stones as
 * `moves` and `line`, the first as the one `top` row, the mover's win probability from its score (1 when its threat
 * solver proved a win, whose distance Six does not report, so there is no `proof`) and the positions searched. The
 * network (the newest when null) is fetched on its first turn; progress follows that download, then the search. */
async function turn({id, history, nodes, network, ms = 0}) {
  if (cancelled.has(id)) throw new Cancelled();
  await use(network ?? manifest.data.networks[0].name, new Stages(postMessage, id));
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
      else if (data.type === 'use') { await use(data.network ?? manifest.data.networks[0].name, new Stages(postMessage, data.id)); postMessage({type: 'result', id: data.id, result: null}); }
    } catch (error) {
      postMessage(error instanceof Cancelled ? {type: 'cancelled', id: data.id} : errorReport(error, data.id));
    } finally {
      cancelled.delete(data.id);
    }
  };
  queue = queue.then(task);
};
