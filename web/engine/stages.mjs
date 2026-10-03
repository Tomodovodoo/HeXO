/* The stages of a browser engine's load. A worker reports each stage in its progress messages as `stage`
 * {name, provider?, fallback?, file?, received?, total?}: probe, download, compile (the runtime starting), session (a
 * network's session), timing (Bubble's fp32 and fp16 batches) and warmup. A call that loads a network reports the same
 * stages; a call's progress without one is its search. The page shows stageText(stage) next to the busy bar, and
 * engine-worker.mjs stops a stage that stays silent for LIMITS[name] ms. */

/** Silence allowed per stage in ms; a stage that reports nothing for that long has stalled. A download reports every
 * chunk, the other stages once. Inside a stage, `adapter` bounds the WebGPU adapter request of the probe and `idle` a
 * download's wait for its next chunk (assets.mjs). */
export const LIMITS = {probe: 20000, download: 45000, compile: 30000, session: 60000, timing: 30000, warmup: 60000,
  adapter: 8000, idle: 30000};

/** Hosts on which the page's test parameters (assets.mjs `assets`, `stall`) are honoured: a development server on
 * this machine. */
const LOOPBACK = /^(localhost|127\.0\.0\.1|\[::1\])$|\.localhost$/;

/** Query parameter `name` of a page on a loopback host (so a link cannot set it on a deployed page), or of a worker's
 * script URL, which only assets.mjs workerUrl fills; null elsewhere. */
export function localParam(name) {
  const value = new URLSearchParams(globalThis.location?.search).get(name);
  return globalThis.document && !LOOPBACK.test(location.hostname) ? null : value;
}

/** The load's share of the bar when a stage starts; downloads fill the bar up to the first of them. */
const FRACTIONS = {probe: .02, compile: .9, session: .93, timing: .96, warmup: .98};

const megabytes = (bytes, total) => (bytes / 1e6).toFixed(total < 1e7 ? 1 : 0);

/** The words the page shows for `stage`: "downloading 12 of 27 MB", "starting GPU", "thinking" (no stage). */
export function stageText(stage) {
  if (!stage) return 'thinking';
  if (stage.name === 'download') {
    return stage.total ? `downloading ${megabytes(stage.received, stage.total)} of ${megabytes(stage.total, stage.total)} MB`
      : stage.received ? `downloading ${megabytes(stage.received, stage.received)} MB` : 'downloading';
  }
  if (stage.name === 'session') return stage.provider === 'webgpu' ? 'starting GPU' : 'starting CPU';
  return {probe: 'checking GPU', compile: 'compiling', timing: 'warming up', warmup: 'warming up'}[stage.name] ?? stage.name;
}

const STALLS = new Set((localParam('stall') ?? '').split(',').filter(Boolean));

/** `step()`, or a promise that never settles (and step not run) when the page asked for step `name` to hang:
 * `?stall=adapter,session` on a loopback host, which assets.mjs workerUrl hands to the workers (adapter, compile,
 * session, timing, warmup). */
export function stall(name, step) {
  return STALLS.has(name) ? new Promise(() => {}) : step();
}

/** A worker's reports of one load or network change: `post` sends {type: 'progress', id?, fraction, stage}. */
export class Stages {
  constructor(post, id) {
    this.post = message => post(message);   // a worker's postMessage, called on the worker's global
    this.id = id;
    this.current = null;
    this.fraction = 0;
    this.device = {};
    this.files = new Map();
  }

  /** Enters stage `name` (probe, download before a manifest or file request, compile, session, timing, warmup);
   * `provider` is the device a session runs on. */
  enter(name, provider) {
    this.current = {name, ...this.device, ...(provider ? {provider} : {})};
    this.send(FRACTIONS[name] ?? this.fraction);
  }

  /** Records the probe's device {provider, fallback?}; the stages after it carry them, so the page can tag the engine. */
  probed({provider, fallback}) {
    this.device = {provider, ...(fallback ? {fallback} : {})};
  }

  /** An assets.cached progress callback for `file`: its bytes join the download stage, which reports the bytes of all
   * files downloading in this load, with their total while every file's size is known. A file read from the Cache API
   * reports no bytes and no stage. */
  file(path) {
    return (fraction, received, total) => {
      if (received === undefined) return;
      this.files.set(path, [received, total]);
      let got = 0, sized = 0, all = 0, known = true;
      for (const [r, t] of this.files.values()) {
        got += r;
        if (t) { sized += r; all += t; } else known = false;
      }
      this.current = {name: 'download', ...this.device, file: path, received: got, total: known ? all : 0};
      this.send(all ? FRACTIONS.compile * Math.min(1, sized / all) : 0);
    };
  }

  send(fraction) {
    this.fraction = fraction;
    this.post({type: 'progress', ...(this.id === undefined ? {} : {id: this.id}), fraction, stage: this.current});
  }

  /** Runs `steps()`; an error it throws carries the stage it stopped in as `error.stage`. */
  async run(steps) {
    try {
      return await steps();
    } catch (error) {
      if (error instanceof Error) error.stage ??= this.current;
      throw error;
    }
  }
}

/** The error message a worker posts for `error`, with the stage it stopped in when it has one. */
export function errorReport(error, id) {
  return {type: 'error', id, message: String(error?.message || error), ...(error?.stage ? {stage: error.stage} : {})};
}

/** The page's Error for engine `name`'s worker error {message, stage?}: it names the stage the worker stopped in. */
export function workerError(name, {message, stage}) {
  return new Error(stage ? `${name}: ${stageText(stage)} failed: ${message}` : message);
}
