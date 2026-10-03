import {BrowserSession} from './play-session.mjs';
import {PlayStorage} from './storage.mjs';
import {notePace} from './device.mjs';

/** The tag the page shows for the device an engine's load resolved to: GPU when its network runs on WebGPU, CPU
 * otherwise (ONNX Runtime on WebAssembly, or an engine with no network). The detail goes to the console. */
export function deviceLabel(device) {
  console.info('Browser engine device:', device);
  return device?.provider === 'webgpu' ? 'GPU' : 'CPU';
}

export async function mountPlay(engines, legacy) {
  const firstEngine = [...engines.values()].find(e => e.entry.kind === 'bubble') || engines.values().next().value;
  const entry = firstEngine.entry;
  const params = new URLSearchParams(location.search), study = params.has('study');
  const book = await (await fetch(new URL('openings.json', import.meta.url))).json();
  let storage;
  try { storage = await PlayStorage.open(); }
  catch (error) { globalThis.toast(`Browser storage is unavailable: ${error.message}`); storage = new PlayStorage(null); }
  const id = study ? `study:${params.get('batch')}:${params.get('game')}` : 'live';
  const start = entry.preset === 'lightning' ? 'lightning' : 'quick';
  const session = await BrowserSession.create({engine: entry.id, preset: start, budget: entry.presets[start], auto: true}, {storage, id, book});
  session.initializing = true;
  const first = !await storage.get('sessions', id);
  if (first && study && params.has('batch')) await session.openGame(params.get('batch'), +params.get('game'));
  for (const {entry, engine, record} of engines.values()) session.registerEngine(entry, {
    ready: async (f, checkpoint) => { entry.device = deviceLabel(await engine.load(f)); await engine.prepare?.(checkpoint); },
    turn: async (history, budget, options) => {
      const started = performance.now();
      const result = await engine.turn(history, {...budget, ...(options.checkpoint ? {checkpoint: options.checkpoint} : {})}, options), preset = options.preset === 'custom' ? 'standard' : options.preset;
      if (options.ms == null && options.preset !== 'custom') notePace(entry, options.preset, performance.now() - started, result.moves?.length);
      return record ? {...record(result, history, preset), ...result} : result;
    }
  });
  if (first && !study) {
    const choices = legacy?.seats?.some(Boolean) ? legacy.seats : [null, {engine: entry.id, preset: entry.preset || 'standard'}];
    session.seats = choices.map(choice => choice ? session.spec(typeof choice === 'string' ? {engine: entry.id, preset: choice} : choice) : {engine: 'human'});
    if (legacy?.analysis) session.analysis = session.spec({...typeof legacy.analysis === 'string' ? {engine: entry.id, preset: legacy.analysis} : legacy.analysis, auto: true});
    session.book.enabled = session.seats.some(s => s.engine !== 'human');
  }
  const networkFetch = globalThis.fetch;
  globalThis.fetch = async (input, init = {}) => {
    const url = new URL(typeof input === 'string' ? input : input.url, location.href);
    if (url.origin !== location.origin || !BrowserSession.handles(url.pathname)) return networkFetch(input, init);
    const method = init.method || input.method || 'GET';
    const body = init.body ? JSON.parse(init.body) : {};
    // Opening a saved game from the live board returns its study URL without replacing the live game.
    if (url.pathname === '/matches/open' && !study) return Response.json({url: `?study=1&batch=${encodeURIComponent(body.batch)}&game=${body.game}`});
    const [status, data, type] = await session.request(url.pathname + url.search, body, method);
    return new Response(type === 'application/json' ? JSON.stringify(data) : data, {status, headers: {'Content-Type': type}});
  };
  let revision = -1, frame = null, pending;
  session.onchange = data => {
    const hint = document.getElementById('browser-storage');
    if (hint && data.storage.error) hint.textContent = data.storage.error;
    pending = data;
    if (frame !== null) return;
    frame = requestAnimationFrame(() => {
      frame = null;
      const value = pending;
      globalThis.accept(value.revision === revision ? {instance: value.instance, revision: value.revision, jobs: value.jobs, clock: value.clock} : value);
      revision = value.revision;
    });
  };
  // Engine adapters can register without changing storage, routes or the Play UI.
  globalThis.browserPlay = session;
  dispatchEvent(new CustomEvent('hexo-browser-ready', {detail: session}));
  document.documentElement.classList.add('browser-play');
  document.addEventListener('click', async event => {
    const link = event.target.closest('a');
    if (!link) return;
    const url = new URL(link.href);
    if (url.origin !== location.origin) return;
    if (link.hasAttribute('download') && BrowserSession.handles(url.pathname)) {
      event.preventDefault();
      try {
        const response = await fetch(url.href);
        if (!response.ok) throw Error((await response.json()).error);
        const object = URL.createObjectURL(await response.blob()), a = document.createElement('a');
        a.href = object; a.download = link.download || `game-${url.searchParams.get('game') || 'saved'}.htttx`;
        document.body.append(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(object), 1000);
      } catch (error) { globalThis.toast(error.message); }
    } else if (url.pathname === '/' && (url.searchParams.has('study') || link.getAttribute('aria-label') === 'Live board')) {
      event.preventDefault();
      const target = new URL('./', new URL('../', import.meta.url)); target.search = url.search;
      if (link.target === '_blank') window.open(target.href, '_blank', 'noopener'); else location.href = target.href;
    }
  });
  const leave = () => { session.freezeClock(); session.cancelJobs(); if (session.match) session.match.active = false; session.paused = true; if (session.dirty) session.persist(); };
  addEventListener('pagehide', leave);
  const hint = document.getElementById('browser-storage');
  if (hint) hint.textContent = storage.db ? 'Games and analysis are saved in this browser.' : 'Browser storage is unavailable. Download your games before leaving.';
  session.initializing = false;
  if (first) session.persist();
  globalThis.accept(session.state()); session.pump();
  return session;
}
