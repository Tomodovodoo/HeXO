/* Bubble running in the browser: the page-side handle of worker.mjs. */

export const PRESETS = {lightning: {simulations: 8, solver_nodes: 2048}, quick: {simulations: 32, solver_nodes: 2048},
  standard: {simulations: 128, solver_nodes: 32768}, strong: {simulations: 512, solver_nodes: 131072},
  deep: {simulations: 2048, solver_nodes: 524288}, dangerous: {simulations: 65536, solver_nodes: 4000000}};

/**
 * Registers ../coi-sw.js (cross-origin isolation for static hosts that cannot send headers) and reloads once, so
 * ONNX Runtime's WebAssembly backend can use threads. Resolves false when the page is already isolated or cannot be.
 */
export async function isolate() {
  if (globalThis.crossOriginIsolated || !navigator.serviceWorker || !isSecureContext) return false;
  const registration = await navigator.serviceWorker.register(new URL('../coi-sw.js', import.meta.url));
  if (registration.active && !navigator.serviceWorker.controller) { location.reload(); return true; }
  registration.addEventListener('updatefound', () => registration.installing?.addEventListener('statechange', event => {
    if (event.target.state === 'activated') location.reload();
  }));
  return true;
}

export class BubbleEngine {
  /** `model` is the manifest URL relative to web/engine; `prefer` 'wasm', 'webgpu-fp32' or 'webgpu-fp16' narrows the device choice. */
  constructor({model = 'model/manifest.json', prefer = null, threads = null} = {}) {
    this.options = {model, prefer, threads};
    this.calls = 0;
    this.waits = new Map();
    this.worker = null;
    this.ready = null;
    this.device = null;
  }

  /** Starts the worker and loads the model; `progress(fraction)` reports loading. Resolves to the chosen device. */
  load(progress = () => {}) {
    this.ready ??= new Promise((resolve, reject) => {
      this.worker = new Worker(new URL('worker.mjs', import.meta.url), {type: 'module'});
      this.worker.onmessage = ({data}) => {
        if (data.type === 'ready') { this.device = data.device; resolve(data.device); return; }
        if (data.id === undefined) {
          if (data.type === 'progress') progress(data.fraction);
          else if (data.type === 'error') reject(new Error(data.message));
          return;
        }
        const wait = this.waits.get(data.id);
        if (!wait) return;
        if (data.type === 'progress') { wait.progress(data.fraction); return; }
        this.waits.delete(data.id);
        if (data.type === 'result') wait.resolve(data.result);
        else wait.reject(data.type === 'cancelled' ? new DOMException('Cancelled', 'AbortError') : new Error(data.message));
      };
      this.worker.onerror = event => reject(new Error(event.message || 'Engine worker failed'));
      this.worker.postMessage({type: 'load', options: this.options});
    });
    this.ready.catch(() => { this.ready = null; this.worker?.terminate(); });
    return this.ready;
  }

  async call(message, {signal, progress = () => {}} = {}) {
    await this.load();
    const id = ++this.calls;
    return new Promise((resolve, reject) => {
      if (signal?.aborted) { reject(new DOMException('Cancelled', 'AbortError')); return; }
      this.waits.set(id, {resolve, reject, progress});
      signal?.addEventListener('abort', () => this.worker.postMessage({type: 'cancel', id}), {once: true});
      this.worker.postMessage({...message, id});
    });
  }

  /**
   * Bubble's turn at `history` ([[q, r], ...]) with `budget` {simulations, solver_nodes} (a PRESETS entry): the fields
   * of python/play.py evaluate. Aborting `signal` cancels it (rejects with an AbortError).
   */
  turn(history, budget, options = {}) {
    return this.call({type: 'turn', history, simulations: budget.simulations, solverNodes: budget.solver_nodes,
      batchSize: budget.batch_size ?? 16}, options);
  }

  /** One search of `simulations` from `history` (no solver): action, completed, elapsed_ms, evaluated, batches, policy. */
  search(history, simulations, options = {}) {
    return this.call({type: 'search', history, simulations, batchSize: options.batchSize ?? 16}, options);
  }

  /** Network predictions [{actions, logits, q}] for each history, as hexnet.DenseEvaluator gives them. */
  evaluate(histories, options = {}) {
    return this.call({type: 'evaluate', histories}, options);
  }

  /** Forward latency in ms per batch, {size: {batch: ms}}. */
  bench(options = {}) {
    return this.call({type: 'bench', ...options});
  }

  close() {
    this.worker?.terminate();
    this.worker = null;
    this.ready = null;
  }
}
