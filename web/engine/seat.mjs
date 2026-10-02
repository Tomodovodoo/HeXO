/* The browser engines (engines.mjs) for web/index.html: seats and an analysis engine that run entirely in this
 * browser. The server sees a browser seat as a human seat and receives its stones through /play; like server engines a
 * browser seat waits while the game is paused, cancelling its move pauses the game, and Undo steps back over its
 * turns to the people's last turn. Browser analyses are shown in place of the server's for the positions they cover.
 * A task that failed is not retried until the position, preset, checkpoint or engine changes. Choices persist per
 * browser (localStorage) as {engine, checkpoint, preset} per seat and for the analysis. */
import {isolate} from './bubble.mjs';
import {ENGINES} from './engines.mjs';
import {OfflineSession} from './offline.mjs';

const STORE = 'browser-engines';
const HOOKS = ['accept', 'post', 'shown', 'renderSeat', 'renderEngineHead', 'renderJobs', 'canPlace', 'renderPanels', 'draw',
  'openMenu', 'el', 'toast', 'badge', 'seatControls', 'pickItems'];
const page = globalThis, original = Object.fromEntries(HOOKS.map(name => [name, page[name]]));
const analyses = new Map(), hk = history => history.map(p => p.join(',')).join(';');
const available = new Map(), entries = [];
let config = {seats: [null, null], analysis: null}, job = null, failed = null, posting = false;
let fresh = true;
try { const saved = localStorage.getItem(STORE); fresh = saved === null; config = {...config, ...JSON.parse(saved)}; } catch {}
const save = () => { failed = null; try { localStorage.setItem(STORE, JSON.stringify(config)); } catch {} };
const state = () => typeof S === 'undefined' ? null : S;
const viewed = () => typeof view === 'undefined' ? 0 : view;
const closeIcon = () => typeof icon === 'function' ? icon('close') : '×';
const specKey = spec => `${spec.engine}|${spec.checkpoint ?? ''}|${spec.preset}`;

/** The spec for choosing engine `id`: the current one's preset and checkpoint when it is that engine already. */
function choose(id, current, change = {}) {
  const engine = available.get(id), base = current?.engine === id ? current : {preset: 'standard', checkpoint: null};
  const spec = {engine: id, preset: base.preset, checkpoint: base.checkpoint, ...change};
  if (!engine.checkpoints.includes(spec.checkpoint)) spec.checkpoint = engine.checkpoints[0] ?? null;
  if (!engine.presets[spec.preset]) spec.preset = 'standard';
  return spec;
}

/** Adds the browser entries to a state's engines and drops browser seats the server has given another engine. */
function adopt(data) {
  if (data.engines) for (const entry of entries) if (!data.engines.some(e => e.id === entry.id)) data.engines.push(entry);
  if (data.seats && data.seats.some((seat, side) => config.seats[side] && seat.engine !== 'human')) {
    config.seats = config.seats.map((spec, side) => data.seats[side].engine === 'human' ? spec : null);
    save();
  }
}

/** Browser analyses of the positions in `data.history`, as server evaluation records. */
function inject(data) {
  if (!config.analysis || !data.history || !data.evaluations) return;
  for (let ply = 0; ply <= data.history.length; ply++) {
    const record = analyses.get(`${specKey(config.analysis)}|${hk(data.history.slice(0, ply))}`);
    if (record) data.evaluations[ply] = record;
  }
}

function bar(element, fraction) {
  if (!element) return;
  element.style.visibility = fraction === null ? 'hidden' : 'visible';
  element.classList.toggle('running', fraction !== null);
  if (fraction !== null) element.firstChild.style.transform = `scaleX(${fraction})`;
}

function progress() {
  const s = state();
  if (!s) return;
  const fraction = () => job.loading ? job.loaded : job.fraction;
  for (const side of [0, 1]) {
    if (!config.seats[side]) continue;
    const active = job?.kind === 'move' && job.side === side, cancel = document.getElementById('cancel' + side);
    bar(document.getElementById('busy' + side), active ? fraction() : null);
    document.getElementById('dot' + side)?.classList.toggle('thinking', active);
    if (!cancel) continue;
    if (!active) { cancel.replaceChildren(); continue; }
    if (!cancel.children.length) {
      cancel.append(original.el('button', {class: 'x', 'aria-label': 'Cancel', html: closeIcon(),
        onclick: () => page.post('/pause', {paused: true})}));
    }
  }
  if (config.analysis) {
    const active = job?.kind === 'analyse' && job.ply === viewed();
    bar(document.getElementById('analysis-progress'), active ? fraction() : null);
    document.getElementById('evalbar')?.classList.toggle('pending', active);
  }
}

function schedule() {
  const s = state();
  if (!s || posting) return;
  const side = s.player, spec = config.seats[side];
  const move = s.winner < 0 && spec && !s.paused && !s.match?.active && s.seats[side].engine === 'human';
  const prefix = s.history.slice(0, viewed());
  const analyse = config.analysis && !(s.winner >= 0 && prefix.length === s.history.length)
    && !analyses.has(`${specKey(config.analysis)}|${hk(prefix)}`);
  const task = move ? {kind: 'move', side, spec, history: s.history.map(p => [...p])}
    : analyse ? {kind: 'analyse', ply: prefix.length, spec: config.analysis, history: prefix.map(p => [...p])} : null;
  const key = task && `${task.kind}|${specKey(task.spec)}|${hk(task.history)}`;
  if (job?.key === key || (key && key === failed)) return;
  job?.controller.abort();
  job = null;
  if (task) run(key, task);
  progress();
}

async function run(key, task) {
  const controller = new AbortController(), current = job = {...task, key, controller, fraction: 0, loading: true, loaded: 0};
  const engine = available.get(task.spec.engine), budget = engine.presets[task.spec.preset];
  try {
    await engine.load(task.spec.checkpoint, f => { current.loaded = f; progress(); });
    current.loading = false;
    const result = await engine.turn(task.history, budget, {checkpoint: task.spec.checkpoint, analysis: task.kind === 'analyse', signal: controller.signal,
      progress: f => { current.fraction = f; progress(); }});
    if (job !== current) return;
    job = null;
    if (task.kind === 'move') {
      posting = true;
      try {
        for (const [q, r] of result.moves) {
          const spec = config.seats[task.side];
          if (state().paused || !spec || specKey(spec) !== specKey(task.spec) || hk(state().history) !== hk(task.history)) break;
          if (!(await original.post('/play', {q, r}))) {
            failed = key;
            break;
          }
          task.history.push([q, r]);
        }
      } finally {
        posting = false;
      }
    } else {
      const record = {...budget, ...result, engine: engine.id, checkpoint: task.spec.checkpoint};
      analyses.set(`${specKey(task.spec)}|${hk(task.history)}`, record);
      const s = state();
      if (s && hk(s.history.slice(0, task.history.length)) === hk(task.history)) {
        s.evaluations[task.history.length] = record;
        page.renderPanels();
        original.draw();
      }
    }
  } catch (error) {
    if (error.name !== 'AbortError') {
      failed = key;
      original.toast(error.message);
    }
    if (job === current) job = null;
  }
  schedule();
}

/** The page's seat controls (checkpoint select, strength) for a browser spec; `send(spec)` stores a change. */
function controls(spec, send, id) {
  const seat = {engine: spec.engine, checkpoint: spec.checkpoint, preset: spec.preset, budget: available.get(spec.engine).presets[spec.preset]};
  return original.seatControls(seat, change => send(choose(spec.engine, spec, change)), id, null);
}

function install() {
  page.accept = data => {
    adopt(data);
    inject(data);
    original.accept(data);
    schedule();
  };
  page.post = (path, body = {}) => {
    const s = state();
    if (path === '/seat' && body.engine !== undefined) {
      config.seats[body.side] = available.has(body.engine) ? choose(body.engine, config.seats[body.side]) : null;
      save();
      if (job?.kind === 'move' && job.side === body.side && !config.seats[body.side]) {
        job.controller.abort();
        job = null;
      }
      if (available.has(body.engine)) return original.post('/seat', {side: body.side, engine: 'human'});
    }
    if (path === '/analysis' && body.engine !== undefined) {
      config.analysis = available.has(body.engine) ? choose(body.engine, config.analysis) : null;
      save();
      if (available.has(body.engine)) {
        if (s?.analysis?.auto) return original.post('/analysis', {...body, engine: s.analysis.engine, checkpoint: s.analysis.checkpoint, auto: false});
        page.renderPanels();
        return Promise.resolve(s);
      }
    }
    if (path === '/analyse' && config.analysis) {
      if (body.force && s) {
        const ply = Number.isInteger(body.ply) ? body.ply : viewed();
        analyses.delete(`${specKey(config.analysis)}|${hk(s.history.slice(0, ply))}`);
        failed = null;
        schedule();
      }
      return Promise.resolve(s);
    }
    if (path === '/undo' && s && config.seats.some(Boolean)) {
      return original.post('/undo', {people: [0, 1].filter(side => s.seats[side].engine === 'human' && !config.seats[side])});
    }
    return original.post(path, body);
  };
  page.shown = seat => {
    const s = state(), side = s ? s.seats.indexOf(seat) : -1;
    const spec = side >= 0 && seat.engine === 'human' ? config.seats[side] : s && seat === s.analysis ? config.analysis : null;
    if (spec) return [available.get(spec.engine).kind, available.get(spec.engine).label];
    return original.shown(seat);
  };
  page.canPlace = () => original.canPlace() && !config.seats[state().player];
  page.renderSeat = side => {
    original.renderSeat(side);
    const box = document.getElementById('seat' + side), s = state(), spec = config.seats[side];
    if (!spec || !box || s.saved_game) return;
    const send = change => { config.seats[side] = change; save(); page.renderPanels(); };
    const pick = box.querySelector('.pick');
    if (pick) {
      const items = [{id: 'human', ids: ['human'], kind: 'you', label: null}, ...original.pickItems(() => true)];
      pick.onclick = () => original.openMenu(pick, items, spec.engine, it => page.post('/seat', {side, engine: it.id}));
    }
    box.append(original.el('div', {class: 'more'}, original.el('div', {}, ...controls(spec, send, 'seat' + side))));
    progress();
  };
  page.renderEngineHead = () => {
    original.renderEngineHead();
    const head = document.getElementById('engine-head'), s = state(), spec = config.analysis;
    if (!spec || !head || !s?.analysis) return;
    const engine = available.get(spec.engine);
    const pick = original.el('button', {class: 'pick'}, ...original.badge(engine.kind, engine.label));
    const items = original.pickItems(e => e.kind === 'bubble' || e.browser);
    pick.onclick = () => original.openMenu(pick, items, spec.engine, it => page.post('/analysis', {engine: it.id, checkpoint: null, preset: 'standard', auto: true}));
    const send = change => { config.analysis = change; save(); page.renderPanels(); };
    head.replaceChildren(original.el('div', {class: 'head'}, pick), ...controls(spec, send, 'analysis'));
  };
  page.renderPanels = () => {
    original.renderPanels();
    const pause = document.getElementById('pause');
    if (pause && config.seats.some(Boolean)) pause.disabled = false;
    schedule();
  };
  page.renderJobs = () => {
    original.renderJobs();
    progress();
  };
  const s = state();
  if (s) {
    adopt(s);
    page.renderPanels();
  }
}

/**
 * Without a play server (a static host) the page's server requests are answered by an OfflineSession, server-only
 * controls are hidden, Bubble (browser) takes seat O and analysis, and the page asks for cross-origin isolation.
 * Resolves true when it took over (or is reloading for isolation).
 */
async function serverless() {
  try {
    const response = await fetch('/state', {cache: 'no-store'});
    if (response.ok && (response.headers.get('Content-Type') || '').includes('json')) return false;
  } catch {}
  if (await isolate()) return true;
  const bubble = ENGINES[0];
  const session = await OfflineSession.create({engine: bubble.id, preset: 'quick', budget: bubble.presets.quick});
  const server = globalThis.fetch;
  globalThis.fetch = (input, init) => {
    const url = new URL(typeof input === 'string' ? input : input.url, location.href);
    if (url.origin !== location.origin || !OfflineSession.handles(url.pathname)) return server(input, init);
    const [status, data] = session.answer(url.pathname, init?.body ? JSON.parse(init.body) : {});
    return Promise.resolve(new Response(JSON.stringify(data), {status, headers: {'Content-Type': 'application/json'}}));
  };
  if (fresh) config.seats = [null, choose(bubble.id, null)];
  config.analysis ??= choose(bubble.id, null, {preset: 'quick'});
  save();
  document.head.append(original.el('style', {}, ['#review-go', '#copy', '#more', '#import', '[aria-label="Tournaments"]']
    .map(selector => `.serverless ${selector}`).join(',') + '{display:none!important}'));
  document.documentElement.classList.add('serverless');
  page.accept(session.state());
  return true;
}

/** Registers the engines this page can run and drops stored choices of the others. */
async function discover() {
  const found = await Promise.all(ENGINES.map(engine => engine.catalogue().catch(() => null)));
  ENGINES.forEach((engine, i) => {
    if (found[i] === null) return;
    available.set(engine.id, {...engine, checkpoints: found[i]});
    entries.push({id: engine.id, kind: engine.kind, name: engine.label, label: engine.label, checkpoints: found[i],
      presets: engine.presets, browser: true});
  });
  const valid = spec => spec && available.has(spec.engine) ? choose(spec.engine, spec) : null;
  config = {seats: config.seats.map(valid), analysis: valid(config.analysis)};
}

if (HOOKS.every(name => typeof original[name] === 'function')) {
  await discover();
  install();
  serverless();
} else console.warn('Browser engines need the play page functions:', HOOKS.filter(name => typeof original[name] !== 'function'));
