/* The page side of a browser engine's Web Worker. The worker answers {type: 'load', options} with 'progress'
 * ({fraction}) messages and then 'ready' ({device}) or 'error' ({message}); each call {type, id, ...} with 'progress'
 * ({id, fraction, live?}, `live` being a running search's root rows, passed to progress as its second argument)
 * messages and then 'result' ({id, result}), 'error' ({id, message}) or, after {type: 'cancel', id}, 'cancelled' ({id}). */
import {defaultThreads} from './network.mjs';

const READY_MS = 20000, STALLED = Symbol('stalled');

export class EngineWorker {
  /** `script` is the worker module URL, `name` the engine's name in messages, `options` go to the worker's load
   * (`options.threads` null leaves the ONNX Runtime thread count to the worker). */
  constructor(script, name, options = {}) {
    this.script = script;
    this.name = name;
    this.options = {threads: null, ...options};
    this.calls = 0;
    this.waits = new Map();
    this.worker = null;
    this.ready = null;
    this.abandon = null;
    this.device = null;
  }

  /**
   * Starts the worker; `progress(fraction)` reports loading. Resolves to the worker's device. When the thread count
   * was left to the worker and the runtime has not come up READY_MS after the downloads finished (its thread workers
   * never start on some hosts), the worker is replaced by one running on a single thread; a further download (the
   * WebAssembly runtime after a failed WebGPU start) suspends that watchdog.
   */
  load(progress = () => {}) {
    if (this.ready) return this.ready;
    const ready = this.ready = this.start(progress).catch(error => {
      if (error !== STALLED) throw error;
      console.warn(`${this.name}: the runtime did not start with its thread workers; retrying on one thread`);
      this.options = {...this.options, threads: 1};
      return this.start(progress);
    });
    ready.catch(error => { if (this.ready === ready) this.fail(error); });
    return ready;
  }

  start(progress) {
    return new Promise((resolve, reject) => {
      const worker = this.worker = new Worker(this.script, {type: 'module'});
      let timer = null;
      const threaded = this.options.threads === null && defaultThreads({isolated: Boolean(globalThis.crossOriginIsolated),
        cores: navigator.hardwareConcurrency || 2}) > 1;   // a single-threaded start has nothing to fall back to
      const stall = armed => {   // the worker reports below .95 while downloading, at .95 while the runtime starts
        clearTimeout(timer);
        if (armed && threaded) timer = setTimeout(() => { if (this.worker === worker) { worker.terminate(); reject(STALLED); } }, READY_MS);
      };
      this.abandon = reject;
      worker.onmessage = ({data}) => {
        if (data.type === 'ready') { clearTimeout(timer); this.device = data.device; resolve(data.device); return; }
        if (data.id === undefined) {
          if (data.type === 'progress') { progress(data.fraction); stall(data.fraction >= .95); }
          else if (data.type === 'error') { clearTimeout(timer); reject(new Error(data.message)); }
          return;
        }
        const wait = this.waits.get(data.id);
        if (!wait) return;
        if (data.type === 'progress') { wait.progress(data.fraction, data.live); return; }
        this.waits.delete(data.id);
        if (data.type === 'result') wait.resolve(data.result);
        else wait.reject(data.type === 'cancelled' ? new DOMException('Cancelled', 'AbortError') : new Error(data.message));
      };
      worker.onerror = event => { clearTimeout(timer); this.fail(new Error(event.message || `${this.name} worker failed`)); };
      worker.postMessage({type: 'load', options: this.options});
    });
  }

  /** Sends `message` once the worker is ready; aborting `signal` cancels it (rejects with an AbortError). */
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
