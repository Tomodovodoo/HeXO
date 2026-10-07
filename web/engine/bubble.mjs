/* Bubble running in the browser: the page-side handle of worker.mjs. */

export const PRESETS = {lightning: {simulations: 8, solver_nodes: 2048}, quick: {simulations: 32, solver_nodes: 2048},
  standard: {simulations: 128, solver_nodes: 32768}, strong: {simulations: 512, solver_nodes: 131072},
  deep: {simulations: 2048, solver_nodes: 524288}, dangerous: {simulations: 65536, solver_nodes: 4000000}};

/** The proof workers of every Bubble search with solver nodes, the solver preset included: half the browser's threads
 * less one, from 1 to 8, so the page's main thread and inference keep the other half. */
export const PROOF_WORKERS = Math.max(1, Math.min(8, Math.floor((globalThis.navigator?.hardwareConcurrency || 4) / 2) - 1));

/**
 * Registers ../coi-sw.js (cross-origin isolation for static hosts that cannot send headers) and reloads once, so
 * ONNX Runtime's WebAssembly backend can use threads. Resolves false when the page is already isolated or cannot be.
 */
export async function isolate() {
  const workers = globalThis.navigator?.serviceWorker, script = new URL('../coi-sw.js', import.meta.url).href;
  if (globalThis.crossOriginIsolated || !workers || !isSecureContext || workers.controller?.scriptURL === script) return false;
  let registration;
  try {
    registration = await workers.register(script);
  } catch (error) {   // a host that refuses service workers still gets the engine, without WASM threads
    console.warn('Bubble (browser): no cross-origin isolation,', error.message);
    return false;
  }
  if (registration.active?.scriptURL === script) { location.reload(); return true; }
  const reload = worker => worker?.addEventListener('statechange', () => { if (worker.state === 'activated') location.reload(); });
  reload(registration.installing || registration.waiting);
  registration.addEventListener('updatefound', () => reload(registration.installing));
  return true;
}

import {EngineWorker} from './engine-worker.mjs';
import {json, workerUrl} from './assets.mjs';
import {loadFiles, modelFiles, probe} from './network.mjs';

/** The exported networks (python tools/build_web.py model), newest first, as model/networks.json (here, else the
 * public site's) lists them:
 * [{name, manifest (relative to model/), model_version}]; empty when the site holds a single model/manifest.json. */
export const NETWORKS = await json('model/networks.json').then(found => found.data.networks, () => []);

/** The manifest of network `name` relative to web/engine: the newest when null, model/manifest.json without a list. */
export function networkManifest(name = null) {
  const network = name === null ? NETWORKS[0] : NETWORKS.find(n => n.name === name);
  if (name !== null && !network) throw new Error(`Bubble has no network ${name}`);
  return network ? `model/${network.manifest}` : 'model/manifest.json';
}

export class BubbleEngine extends EngineWorker {
  /** `model` is the default manifest's path under web/engine; `prefer` 'wasm', 'webgpu-fp32' or 'webgpu-fp16' narrows the device choice. */
  constructor({model = networkManifest(), prefer = null, threads = null} = {}) {
    super(workerUrl('worker.mjs'), 'Bubble (browser)', {model, prefer, threads});
  }

  /** The downloaded files (assets.mjs records) a load on this device may read: ONNX Runtime and the model graphs,
   * with the WebAssembly runtime and the fp32 graph a WebGPU fallback needs. */
  async files() {
    const {provider, precisions} = await probe(this.options.prefer), {prefer} = this.options;
    const graphs = provider === 'webgpu' && !prefer ? [...new Set([...precisions, 'fp32'])] : precisions;
    return [...await loadFiles(provider, prefer), ...(await modelFiles(graphs, this.options.model)).files];
  }

  /**
   * Bubble's turn at `history` ([[q, r], ...]) with `budget` {simulations, solver_nodes, optional q_range_floor and
   * checkpoint, a NETWORKS name} (a PRESETS entry): the fields of python/play.py evaluate. Every stone runs the hybrid
   * scheduler (worker.mjs playTurn): `simulations` is its work per stone, and `solver_nodes` above 0 adds the root
   * queries at that node budget and the owner's proof frontier on PROOF_WORKERS proof workers (`solver_workers`
   * overrides the count). Optional solver_ms (the SOLVER preset) spends up to that long on proof work alone before the
   * turn (worker.mjs proveRoot). Under a clock `options.ms` is the turn's time and the budget a ceiling (see
   * worker.mjs). `options.line`, a seat's game key, continues that game's search graph; `options.known` is the game's
   * proof table (proof.mjs Proofs.list()) the turn may use. Checked local proof reuse is on by default;
   * `options.proofStamps = false` disables it. Aborting `signal` cancels it (rejects with an AbortError).
   */
  turn(history, budget, options = {}) {
    return this.call({type: 'turn', history, model: budget.checkpoint ? networkManifest(budget.checkpoint) : this.options.model,
      simulations: budget.simulations, solverNodes: budget.solver_nodes, solverWorkers: budget.solver_workers ?? PROOF_WORKERS,
      solverSlice: budget.solver_slice_ms ?? 8, solverTable: budget.solver_table_mb ?? 4, proveMs: budget.solver_ms ?? 0,
      proofStamps: options.proofStamps ?? true,
      batchSize: budget.batch_size ?? 16, choice: options.choice ?? 'policy', qRangeFloor: budget.q_range_floor ?? 0,
      ms: options.ms ?? null, line: options.line ?? null, known: options.known ?? null, replay: options.replay ?? []}, options);
  }

  /** Loads network `checkpoint` (a NETWORKS name, the default when null), so a timed turn does not spend its clock on it. */
  prepare(checkpoint = null, options = {}) {
    return this.call({type: 'use', model: checkpoint ? networkManifest(checkpoint) : this.options.model}, options);
  }

  /** Network predictions [{actions, logits, q}] for each history, as hexnet.DenseEvaluator gives them (q broadcast). */
  evaluate(histories, options = {}) {
    return this.call({type: 'evaluate', histories}, options);
  }

  /** Forward latency in ms per batch, {size: {batch: ms}}. */
  bench(options = {}) {
    return this.call({type: 'bench', ...options});
  }
}
