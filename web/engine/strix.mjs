/* "Strix (browser)": Strix's network and Gumbel search (tools/strix_web as strix/strix.wasm) in strix-worker.mjs, for
 * the play page's browser engines (seat.mjs). It plays as python/play.py's Strix: the same presets in simulations per
 * placement, and the network `python tools/build_web.py strix-network` placed in strix/ (listed in
 * strix/networks.json, the first one is used). Without that file `strix` is null and the page does not offer it. */

/** Simulations per placement, as python/play.py's Strix presets. */
export const PRESETS = {lightning: {simulations: 2}, quick: {simulations: 8}, standard: {simulations: 64},
  strong: {simulations: 128}, deep: {simulations: 512}, dangerous: {simulations: 4096}};
const ID = 'browser:strix', LABEL = 'Strix (browser)', MANIFEST = new URL('strix/networks.json', import.meta.url);

export class StrixEngine {
  /** `network` is a strix/networks.json entry with its `url`. */
  constructor(network) {
    this.network = network;
    this.worker = null;
    this.ready = null;
    this.calls = 0;
  }

  /** Starts the worker with the network; `progress(fraction)` reports the download. */
  load(progress = () => {}) {
    if (this.ready) return this.ready;
    const worker = this.worker = new Worker(new URL('strix-worker.mjs', import.meta.url), {type: 'module'});
    const ready = this.ready = new Promise((resolve, reject) => {
      worker.onmessage = ({data}) => {
        if (data.type === 'progress' && data.id === undefined) progress(data.fraction);
        else if (data.type === 'ready') resolve(data.info);
        else if (data.type === 'error') reject(new Error(data.message));
      };
      worker.onerror = event => reject(new Error(event.message || 'Strix worker failed'));
    });
    ready.catch(() => { if (this.ready === ready) this.close(); });
    worker.postMessage({type: 'load', network: {url: this.network.url, sha256: this.network.sha256}});
    return ready;
  }

  /** Ends the worker; the next load starts a new one. */
  close() {
    this.worker?.terminate();
    this.worker = this.ready = null;
  }

  /**
   * Strix's turn after `history` ([[q, r], ...]) at `budget.simulations` per placement: {moves, value, top, simulations,
   * eval_states, root_visits, ms}. Aborting `signal` ends the worker (a search cannot be interrupted inside it) and
   * rejects with an AbortError; the next call starts a new one.
   */
  async turn(history, budget, {signal, progress = () => {}} = {}) {
    await this.load();
    const worker = this.worker, id = ++this.calls;
    return new Promise((resolve, reject) => {
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
      worker.postMessage({type: 'turn', id, history, simulations: budget.simulations});
    });
  }
}

async function networks() {
  try {
    const response = await fetch(MANIFEST, {cache: 'no-store'});
    if (!response.ok) return [];
    return (await response.json()).networks.map(network => ({...network, url: new URL(network.file, MANIFEST).href}));
  } catch {
    return [];
  }
}

const found = await networks();

/** The browser engine for seat.mjs, or null when no network was built. */
export const strix = found.length ? {
  entry: {id: ID, kind: 'strix', name: LABEL, label: LABEL, checkpoints: found.map(n => n.id), presets: PRESETS, analysis: true},
  engine: new StrixEngine(found[0]),
  record: result => ({...result, proof: null, line: [], threat: [], engine: ID}),
} : null;
