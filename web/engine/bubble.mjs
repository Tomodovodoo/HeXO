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

export class BubbleEngine {
  /** `model` is the manifest URL relative to web/engine; `prefer` 'wasm', 'webgpu-fp32' or 'webgpu-fp16' narrows the device choice. */
  constructor({model = 'model/manifest.json', prefer = null, threads = null} = {}) {
    this.options = {model, prefer, threads};
    this.calls = 0;
    this.waits = new Map();
    this.worker = null;
    this.ready = null;
    this.abandon = null;
    this.device = null;
  }

  /** Starts the worker and loads the model; `progress(fraction)` reports loading. Resolves to the chosen device. */
  load(progress = () => {}) {
    if (this.ready) return this.ready;
    const ready = this.ready = new Promise((resolve, reject) => {
      this.abandon = reject;
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
      this.worker.onerror = event => this.fail(new Error(event.message || 'Engine worker failed'));
      this.worker.postMessage({type: 'load', options: this.options});
    });
    ready.catch(error => { if (this.ready === ready) this.fail(error); });
    return ready;
  }

  async call(message, {signal, progress = () => {}} = {}) {
    await this.load();
    const id = ++this.calls;
    return new Promise((resolve, reject) => {
      if (signal?.aborted || !this.worker) { reject(new DOMException(signal?.aborted ? 'Cancelled' : 'Closed', 'AbortError')); return; }
      const worker = this.worker;
      this.waits.set(id, {resolve, reject, progress});
      signal?.addEventListener('abort', () => worker.postMessage({type: 'cancel', id}), {once: true});
      worker.postMessage({...message, id});
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

  /** One search of `simulations` from `history` (no solver): action, completed, elapsed_ms, evaluated, batches, policy.
   * Aborting `options.signal` cancels it (rejects with an AbortError). */
  search(history, simulations, options = {}) {
    return this.call({type: 'search', history, simulations, batchSize: options.batchSize ?? 16}, options);
  }

  /** Network predictions [{actions, logits, q}] for each history, as hexnet.DenseEvaluator gives them (q broadcast). */
  evaluate(histories, options = {}) {
    return this.call({type: 'evaluate', histories}, options);
  }

  /** Forward latency in ms per batch, {size: {batch: ms}}. */
  bench(options = {}) {
    return this.call({type: 'bench', ...options});
  }

  /** Ends the worker; pending loads and calls reject with an AbortError. */
  close() {
    this.fail(new DOMException('Closed', 'AbortError'));
  }

  /** Ends the worker and rejects the pending load and calls with `error`; the next call starts a new worker. */
  fail(error) {
    this.worker?.terminate();
    this.worker = null;
    this.ready = null;
    this.abandon?.(error);
    this.abandon = null;
    for (const wait of this.waits.values()) wait.reject(error);
    this.waits.clear();
  }
}
