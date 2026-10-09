/* The page side of a browser engine's Web Worker. The worker answers {type: 'load', options} with 'progress'
 * ({fraction, stage}) messages and then 'ready' ({device}) or 'error' ({message, stage?}); each call {type, id, ...}
 * with 'progress' ({id, fraction, live?, stage?, placed?}: `live` a running search's root rows, passed to progress as
 * its second argument, `stage` a loading stage while the call loads a network, its third, `placed` a turn's stones
 * decided so far, its fourth) messages and then 'result'
 * ({id, result}), 'error' ({id, message, stage?}) or, after {type: 'cancel', id}, 'cancelled' ({id}). Stages are
 * stages.mjs's. */
import {deviceThreads} from './network.mjs';
import {LIMITS, stageText} from './stages.mjs';

/** Stages whose failure another device or thread count may avoid; a download fails the same way on any device. */
const RETRIED = new Set(['probe', 'compile', 'session', 'timing', 'warmup']);
const CANCEL_GRACE_MS = 2000, CANCEL_LIMIT_MS = 10000;

/** The engines' fallback notices for the page: 'notice' events whose `detail` is {engine (the EngineWorker), text (one
 * short line), cpu (true when the engine left WebGPU for WebAssembly)}. */
export const notices = new EventTarget();

/** A stage that stalled or failed: `reason` 'timed out' or 'failed', `detail` the worker's message. */
class Stopped extends Error {
  constructor(stage, reason, detail = '') {
    super(`${stageText(stage)} ${reason}`);
    this.stage = stage;
    this.reason = reason;
    this.detail = detail;
  }
}

export class EngineWorker {
  /** `script` is the worker module URL, `name` the engine's name in messages, `options` go to the worker's load
   * (`options.threads` null leaves the ONNX Runtime thread count to the worker, `options.prefer` null the device). */
  constructor(script, name, options = {}) {
    this.script = script;
    this.name = name;
    this.options = {threads: null, prefer: null, ...options};
    this.calls = 0;
    this.waits = new Map();
    this.worker = null;
    this.ready = null;
    this.device = null;
    this.progress = () => {};
    this.watched = new Map();   // the load ('load') or a call's id -> {stage, timer} while it is in a loading stage
    this.provider = null;
    this.fallback = null;
    this.booting = false;
    this.abandon = null;
  }

  /**
   * Starts the worker; `progress(fraction, stage)` reports loading (a stage carries `provider` once the probe has
   * finished). Resolves to the worker's device, with `fallback` when the load left its first device. A stage that
   * reports nothing for its LIMITS entry, or fails while RETRIED holds it, moves the engine one step down and starts
   * a new worker: WebGPU to WebAssembly unless `prefer` fixed the device, then WebAssembly threads to one thread unless
   * `threads` fixed them. Each step is logged and posts one notice naming the engine and the fallback. With no step
   * left, or a failure in another stage, the load rejects with an error naming the engine and the stage.
   */
  load(progress = () => {}) {
    if (this.ready) return this.ready;
    this.progress = progress;
    this.fallback = null;
    const ready = this.ready = this.boot();
    ready.catch(error => { if (this.ready === ready) this.fail(error); });
    return ready;
  }

  /** The next step down from the current options: {options, text}, or null. One thread is a step only on WebAssembly,
   * whose thread count it is. */
  step() {
    if (!this.options.prefer && this.provider !== 'wasm') return {options: {prefer: 'wasm'}, text: 'on CPU'};
    const cpu = this.provider === 'wasm' || this.options.prefer === 'wasm';
    if (cpu && this.options.threads === null && deviceThreads() > 1) return {options: {threads: 1}, text: 'on one thread'};
    return null;
  }

  /** Starts workers until one is ready, from `stopped` (a Stopped that ended the last worker) when given. */
  async boot(stopped = null) {
    this.booting = true;
    try {
      for (;;) {
        if (stopped) {
          const step = this.step(), what = `${this.name}: ${stopped.message}`;
          console.warn(`${what}${stopped.detail ? ` (${stopped.detail})` : ''}${step ? `; running ${step.text}` : ''}`);
          if (!step) throw new Error(what);
          this.notice(`${what}, running ${step.text}`, step.options.prefer === 'wasm');
          this.options = {...this.options, ...step.options};
          this.fallback = stopped.message;
        }
        try {
          const device = await this.start();
          console.info(`${this.name} device:`, device);
          return this.fallback ? {...device, fallback: this.fallback} : device;
        } catch (error) {
          if (!(error instanceof Stopped)) throw error;
          stopped = error;
        }
      }
    } finally {
      this.booting = false;
    }
  }

  /** Ends the worker after `stopped` during a call, starts the next step's worker and sends it the waiting calls. */
  restart(stopped) {
    this.halt();
    const ready = this.ready = this.boot(stopped).then(device => {
      for (const [id, wait] of this.waits) {
        if (!wait.cancelled) this.worker.postMessage({...wait.message, id});
        else { this.waits.delete(id); wait.reject(new DOMException('Cancelled', 'AbortError')); }
      }
      return device;
    });
    ready.catch(error => { if (this.ready === ready) this.fail(error); });
  }

  /** Re-arms the watchdog of `key` (the load, or a call's id) for `stage` (null: none), which hands `stalled` the
   * stage's failure as 'timed out'. Each load and call has its own, so one call's progress leaves another's alone. */
  watch(key, stage, stalled) {
    clearTimeout(this.watched.get(key)?.timer);
    this.watched.delete(key);
    if (stage?.provider) this.provider = stage.provider;
    if (!stage || !LIMITS[stage.name]) return;
    this.watched.set(key, {stage, timer: setTimeout(() => stalled(this.failure('timed out', stage, '')), LIMITS[stage.name])});
  }

  /** The loading stage the worker was last watched in, or null when nothing is loading. */
  get stage() {
    return [...this.watched.values()].at(-1)?.stage ?? null;
  }

  /** The error for `reason` ('failed' or 'timed out', with the worker's `detail`) in `stage`: a Stopped when another
   * step may avoid it, else an Error naming the engine and the stage. */
  failure(reason, stage, detail) {
    if (stage && RETRIED.has(stage.name)) return new Stopped(stage, reason, detail);
    return new Error(stage ? `${this.name}: ${stageText(stage)} ${reason}${detail ? `: ${detail}` : ''}` : detail);
  }

  start() {
    return new Promise((resolve, reject) => {
      const worker = this.worker = new Worker(this.script, {type: 'module'}), first = {name: 'probe'};
      this.abandon = reject;
      this.provider = null;
      const report = (fraction, stage) => {
        this.progress(fraction, stage);
        for (const wait of this.waits.values()) wait.progress(fraction, undefined, stage);
      };
      const loading = error => { if (this.worker === worker) { this.halt(); reject(error); } };
      const calling = error => { if (this.worker === worker) error instanceof Stopped ? this.restart(error) : this.fail(error); };
      let ready = false;
      worker.onmessage = ({data}) => {
        if (this.worker !== worker) return;
        if (data.type === 'ready') { ready = true; this.watch('load', null); this.abandon = null; this.device = data.device; resolve(data.device); return; }
        if (data.id === undefined) {
          if (data.type === 'progress') {
            if (data.stage?.fallback && !this.fallback) this.noticeProbe(data.stage);
            this.watch('load', data.stage ?? null, loading);
            report(data.fraction, data.stage);
          } else if (data.type === 'error') {
            loading(this.failure('failed', data.stage, data.message));
          }
          return;
        }
        const wait = this.waits.get(data.id);
        if (!wait) return;
        if (data.type === 'progress') {
          if (!wait.cancelled) { this.watch(data.id, data.stage ?? null, calling); wait.progress(data.fraction, data.live, data.stage, data.placed); }
          else wait.grace?.();   // a worker still reporting is winding down, not stuck
          return;
        }
        this.watch(data.id, null);
        if (!wait.cancelled && data.type === 'error' && RETRIED.has(data.stage?.name)) { calling(this.failure('failed', data.stage, data.message)); return; }
        this.waits.delete(data.id);
        if (data.type === 'result' && !wait.cancelled) wait.resolve(data.result);
        else {
          // A worker that stopped a search names the search graph it changed (worker.mjs), for the session's count.
          const error = wait.cancelled || data.type === 'cancelled' ? new DOMException('Cancelled', 'AbortError') : this.failure('failed', data.stage, data.message);
          const graph = data.graph ?? (wait.cancelled ? data.result?.graph_id : null);
          if (graph) error.graph = graph;
          wait.reject(error);
        }
      };
      worker.onerror = event => {
        if (this.worker !== worker) return;
        const error = this.failure('failed', this.stage, event.message || `${this.name} worker failed`);
        (ready ? calling : loading)(error);
      };
      this.watch('load', first, loading);
      report(0, first);
      worker.postMessage({type: 'load', options: this.options});
    });
  }

  /** Posts the notice for a probe that fell back to WebAssembly (stage.fallback, its reason), and keeps later workers
   * off WebGPU. */
  noticeProbe(stage) {
    const what = `${this.name}: ${stageText({name: 'probe'})} ${stage.fallback}`;
    console.warn(`${what}; running on CPU`);
    this.notice(`${what}, running on CPU`, true);
    this.fallback = `${stageText({name: 'probe'})} ${stage.fallback}`;
    this.options = {...this.options, prefer: this.options.prefer ?? 'wasm'};
  }

  /** Dispatches a notice: `text` for the page, `cpu` when the engine left WebGPU for WebAssembly. */
  notice(text, cpu) {
    notices.dispatchEvent(new CustomEvent('notice', {detail: {engine: this, text, cpu}}));
  }

  /** Sends `message` once the worker is ready; aborting `signal` cancels it (rejects with an AbortError). A call that
   * a restart catches goes to the new worker. */
  async call(message, {signal, progress = () => {}} = {}) {
    let ready;
    do {
      ready = this.load();
      await ready;
    } while (ready !== this.ready);
    const id = ++this.calls;
    return new Promise((resolve, reject) => {
      if (signal?.aborted || !this.worker) { reject(new DOMException(signal?.aborted ? 'Cancelled' : 'Closed', 'AbortError')); return; }
      let timer;
      const clean = () => { clearTimeout(timer); signal?.removeEventListener('abort', cancel); };
      const wait = {resolve: value => { clean(); resolve(value); }, reject: error => { clean(); reject(error); }, progress, message, cancelled: false};
      const cancel = () => {
        if (this.waits.get(id) !== wait) return;
        wait.cancelled = true;
        this.watch(id, null);
        if (!this.booting) {
          this.worker?.postMessage({type: 'cancel', id});
          // A search stuck in WASM or inference cannot acknowledge cancellation, nor report anything. End its worker
          // when it stays silent for CANCEL_GRACE_MS before releasing the session's job slot; the next job then loads
          // a fresh engine on the same device. Each message of the call starts the grace again, for CANCEL_LIMIT_MS at most.
          const asked = performance.now();
          wait.grace = () => {
            if (performance.now() - asked > CANCEL_LIMIT_MS) return;
            clearTimeout(timer);
            timer = setTimeout(() => {
              if (this.waits.get(id) !== wait) return;
              this.fail(new DOMException('Cancelled', 'AbortError'));
              this.notice(`${this.name}: cancelled search did not stop; restarting engine`, false);
            }, CANCEL_GRACE_MS);
          };
          wait.grace();
        } else { this.waits.delete(id); wait.reject(new DOMException('Cancelled', 'AbortError')); }
      };
      this.waits.set(id, wait);
      signal?.addEventListener('abort', cancel, {once: true});
      this.worker.postMessage({...message, id});
    });
  }

  /** Ends the worker; pending loads and calls reject with an AbortError. */
  close() {
    this.fail(new DOMException('Closed', 'AbortError'));
  }

  /** Ends the current worker and its watchdog. */
  halt() {
    for (const {timer} of this.watched.values()) clearTimeout(timer);
    this.watched.clear();
    this.worker?.terminate();
    this.worker = null;
    for (const [id, wait] of this.waits) if (wait.cancelled) {
      this.waits.delete(id); wait.reject(new DOMException('Cancelled', 'AbortError'));
    }
  }

  /** Ends the worker and rejects the pending load and calls with `error`; the next call starts a new worker. */
  fail(error) {
    this.halt();
    this.ready = null;
    this.abandon?.(error);
    this.abandon = null;
    for (const wait of this.waits.values()) wait.reject(error);
    this.waits.clear();
  }
}
