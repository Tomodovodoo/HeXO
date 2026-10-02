/* Strix (browser): the page handle of strix-worker.mjs, as an engine of engines.mjs. Its networks are the ones
 * `python tools/build_web.py strix` placed in strix/ and listed in strix/networks.json; without that file the page
 * does not offer it. */

/** Simulations per placement, as python/play.py's Strix presets. */
export const PRESETS = {lightning: {simulations: 2}, quick: {simulations: 8}, standard: {simulations: 64},
  strong: {simulations: 128}, deep: {simulations: 512}, dangerous: {simulations: 4096}};

const MANIFEST = new URL('strix/networks.json', import.meta.url);

export class StrixEngine {
  constructor() {
    this.networks = new Map();
    this.worker = null;
    this.network = null;
    this.ready = null;
    this.calls = 0;
  }

  /** The network ids of strix/networks.json, the default first, or null when it is missing. */
  async catalogue() {
    const response = await fetch(MANIFEST, {cache: 'no-store'});
    if (!response.ok) return null;
    const {networks} = await response.json();
    for (const network of networks) this.networks.set(network.id, {...network, url: new URL(network.file, MANIFEST).href});
    return networks.map(network => network.id);
  }

  /** Starts the worker with network `id` (null for the default); `progress(fraction)` reports the download. */
  load(id, progress = () => {}) {
    const network = this.networks.get(id ?? this.networks.keys().next().value);
    if (!network) return Promise.reject(new Error(`No Strix network ${id}`));
    if (this.ready && this.network === network) return this.ready;
    this.close();
    const worker = this.worker = new Worker(new URL('strix-worker.mjs', import.meta.url), {type: 'module'});
    this.network = network;
    const ready = this.ready = new Promise((resolve, reject) => {
      worker.onmessage = ({data}) => {
        if (data.type === 'progress' && data.id === undefined) progress(data.fraction);
        else if (data.type === 'ready') resolve(data.info);
        else if (data.type === 'error') reject(new Error(data.message));
      };
      worker.onerror = event => reject(new Error(event.message || 'Strix worker failed'));
    });
    ready.catch(() => { if (this.ready === ready) this.close(); });
    worker.postMessage({type: 'load', network: {url: network.url, sha256: network.sha256}});
    return ready;
  }

  /** Ends the worker; the next load starts a new one. */
  close() {
    this.worker?.terminate();
    this.worker = this.network = this.ready = null;
  }

  /**
   * Strix's turn after `history` at `budget.simulations` per placement with network `checkpoint`. Aborting `signal`
   * ends the worker and rejects with an AbortError.
   */
  async turn(history, budget, {checkpoint = null, analysis = false, signal, progress = () => {}} = {}) {
    await this.load(checkpoint);
    const worker = this.worker, id = ++this.calls;
    const result = await new Promise((resolve, reject) => {
      const abort = () => { if (this.worker === worker) this.close(); reject(new DOMException('Cancelled', 'AbortError')); };
      if (signal?.aborted) { abort(); return; }
      signal?.addEventListener('abort', abort, {once: true});
      worker.onmessage = ({data}) => {
        if (data.id !== id) return;
        if (data.type === 'progress') { progress(data.fraction); return; }
        signal?.removeEventListener('abort', abort);
        data.type === 'result' ? resolve(data.result) : reject(new Error(data.message));
      };
      worker.onerror = event => { if (this.worker === worker) this.close(); reject(new Error(event.message || 'Strix worker failed')); };
      worker.postMessage({type: 'turn', id, history, simulations: budget.simulations, analysis});
    });
    return {...result, proof: null, line: [], threat: [], simulations: budget.simulations};
  }
}

const engine = new StrixEngine();

export default {
  id: 'browser:strix', kind: 'strix', label: 'Strix (browser)', presets: PRESETS,
  catalogue: () => engine.catalogue(),
  load: (checkpoint, progress) => engine.load(checkpoint, progress),
  turn: (history, budget, options) => engine.turn(history, budget, options),
};
