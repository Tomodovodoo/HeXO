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

export class BubbleEngine extends EngineWorker {
  /** `model` is the manifest URL relative to web/engine; `prefer` 'wasm', 'webgpu-fp32' or 'webgpu-fp16' narrows the device choice. */
  constructor({model = 'model/manifest.json', prefer = null, threads = null} = {}) {
    super(new URL('worker.mjs', import.meta.url), 'Bubble (browser)', {model, prefer, threads});
  }

  /**
   * Bubble's turn at `history` ([[q, r], ...]) with `budget` {simulations, solver_nodes, optional q_range_floor} (a
   * PRESETS entry): the fields of python/play.py evaluate. `options.known` is the game's proof table (proof.mjs
   * Proofs.list()) the turn may use. Aborting `signal` cancels it (rejects with an AbortError).
   */
  turn(history, budget, options = {}) {
    return this.call({type: 'turn', history, simulations: budget.simulations, solverNodes: budget.solver_nodes,
      batchSize: budget.batch_size ?? 16, choice: options.choice ?? 'policy', qRangeFloor: budget.q_range_floor ?? 0,
      known: options.known ?? null}, options);
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
