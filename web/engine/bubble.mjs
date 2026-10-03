/* Bubble running in the browser: the page-side handle of worker.mjs. */

export const PRESETS = {lightning: {simulations: 8, solver_nodes: 2048}, quick: {simulations: 32, solver_nodes: 2048},
  standard: {simulations: 128, solver_nodes: 32768}, strong: {simulations: 512, solver_nodes: 131072},
  deep: {simulations: 2048, solver_nodes: 524288}, dangerous: {simulations: 65536, solver_nodes: 4000000}};

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

/** The exported networks (python tools/build_web.py model), newest first, as model/networks.json lists them:
 * [{name, manifest (relative to model/), model_version}]; empty when the site holds a single model/manifest.json. */
export const NETWORKS = await (async () => {
  try {
    const response = await fetch(new URL('model/networks.json', import.meta.url), {cache: 'no-cache'});
    return response.ok ? (await response.json()).networks : [];
  } catch {
    return [];
  }
})();

/** The manifest of network `name` relative to web/engine: the newest when null, model/manifest.json without a list. */
export function networkManifest(name = null) {
  const network = name === null ? NETWORKS[0] : NETWORKS.find(n => n.name === name);
  if (name !== null && !network) throw new Error(`Bubble has no network ${name}`);
  return network ? `model/${network.manifest}` : 'model/manifest.json';
}

export class BubbleEngine extends EngineWorker {
  /** `model` is the default manifest URL relative to web/engine; `prefer` 'wasm', 'webgpu-fp32' or 'webgpu-fp16'
   * narrows the device choice. */
  constructor({model = networkManifest(), prefer = null, threads = null} = {}) {
    super(new URL('worker.mjs', import.meta.url), 'Bubble (browser)', {model, prefer, threads});
  }

  /**
   * Bubble's turn at `history` ([[q, r], ...]) with `budget` {simulations, solver_nodes, optional q_range_floor and
   * checkpoint, a NETWORKS name} (a PRESETS entry): the fields of python/play.py evaluate. Under a clock
   * `options.ms` is the turn's time and the budget a ceiling (see worker.mjs). Aborting `signal` cancels it (rejects
   * with an AbortError).
   */
  turn(history, budget, options = {}) {
    return this.call({type: 'turn', history, model: budget.checkpoint ? networkManifest(budget.checkpoint) : this.options.model,
      simulations: budget.simulations, solverNodes: budget.solver_nodes,
      batchSize: budget.batch_size ?? 16, choice: options.choice ?? 'policy', qRangeFloor: budget.q_range_floor ?? 0,
      ms: options.ms ?? null}, options);
  }

  /** Loads network `checkpoint` (a NETWORKS name, the default when null), so a timed turn does not spend its clock on it. */
  prepare(checkpoint = null, options = {}) {
    return this.call({type: 'use', model: checkpoint ? networkManifest(checkpoint) : this.options.model}, options);
  }

  /** One search of `simulations` from `history` (no solver): action, completed, elapsed_ms, evaluated, batches, policy.
   * `options.qRangeFloor` is the tree's Q range floor (0 by default). Aborting `options.signal` cancels it (rejects
   * with an AbortError). */
  search(history, simulations, options = {}) {
    return this.call({type: 'search', history, simulations, batchSize: options.batchSize ?? 16,
      choice: options.choice ?? 'policy', qRangeFloor: options.qRangeFloor ?? 0}, options);
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
