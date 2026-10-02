/* Browser engines for web/index.html: seats and analysis engines that run entirely in this browser, one entry per
 * engine in ENGINES. The server sees a browser seat as a human seat and receives its stones through /play; like
 * server engines a browser seat waits while the game is paused, cancelling its move pauses the game, and Undo steps
 * back over its turns to the people's last turn. Browser analyses are shown in place of the server's for the
 * positions they cover. A task that failed is not retried until the position, preset or engine choice changes.
 * Choices persist per browser (localStorage). */
import {BubbleEngine, PRESETS as BUBBLE, isolate} from './bubble.mjs';
import {SealEngine, PRESETS as SEAL} from './seal.mjs';
import {OfflineSession} from './offline.mjs';

/**
 * Each engine has `load(progress)` and `turn(history, budget, {signal, progress})`, which resolves to a turn with
 * `moves` and rejects with an AbortError when `signal` aborts; `record(result, budget)` is that turn as the server's
 * evaluation record ({value, top, line, proof, threat} and the budget it spent).
 */
const ENGINES = [
  {id: 'browser:bubble', kind: 'bubble', label: 'Bubble (browser)', presets: BUBBLE, engine: new BubbleEngine(),
    record: (result, {simulations, solver_nodes}) => ({...result, simulations, solver_nodes: result.solved ? solver_nodes : 0})},
  {id: 'browser:seal', kind: 'seal', label: 'Seal (browser)', presets: SEAL, engine: new SealEngine(),
    record: ({moves}, {ms}) => ({moves, value: null, top: moves.map(([q, r]) => [q, r, 1]), line: [], proof: null, threat: [], ms})},
];
const BY_ID = new Map(ENGINES.map(e => [e.id, e]));
const ENTRIES = ENGINES.map(({id, kind, label, presets}) => ({id, kind, name: label, label, checkpoints: [], presets, browser: true}));
const STORE = 'browser-engines', BUBBLE_STORE = 'bubble-browser';
const HOOKS = ['accept', 'post', 'shown', 'renderSeat', 'renderEngineHead', 'renderJobs', 'canPlace', 'renderPanels', 'draw',
  'openMenu', 'el', 'toast', 'badge', 'strength', 'pickItems'];
const page = globalThis, original = Object.fromEntries(HOOKS.map(name => [name, page[name]]));
const analyses = new Map(), hk = history => history.map(p => p.join(',')).join(';');
/* seats[side] and analysis are null or {engine: an ENGINES id, preset} */
let config = {seats: [null, null], analysis: null}, job = null, failed = null, posting = false, fresh = true;
try {
  /* choices saved before Seal joined (key BUBBLE_STORE) hold Bubble preset names */
  const bubble = preset => typeof preset === 'string' ? {engine: 'browser:bubble', preset} : null;
  const earlier = JSON.parse(localStorage.getItem(BUBBLE_STORE));
  const saved = JSON.parse(localStorage.getItem(STORE))
    ?? (earlier && {seats: [0, 1].map(side => bubble(earlier.seats?.[side])), analysis: bubble(earlier.analysis)});
  fresh = saved === null;
  const valid = choice => BY_ID.get(choice?.engine)?.presets[choice.preset] ? choice : null;
  if (saved) config = {seats: [0, 1].map(side => valid(saved.seats?.[side])), analysis: valid(saved.analysis)};
  localStorage.removeItem(BUBBLE_STORE);
  if (earlier) localStorage.setItem(STORE, JSON.stringify(config));
} catch {}
const save = () => { failed = null; try { localStorage.setItem(STORE, JSON.stringify(config)); } catch {} };
const state = () => typeof S === 'undefined' ? null : S;
const viewed = () => typeof view === 'undefined' ? 0 : view;
const closeIcon = () => typeof icon === 'function' ? icon('close') : '×';
const label = choice => BY_ID.get(choice.engine).label;
/** `choice` moved to engine `id`: the same preset on the same engine, else standard. */
const choose = (choice, id) => ({engine: id, preset: choice?.engine === id ? choice.preset : 'standard'});
const analysisKey = (choice, history) => `${choice.engine}|${choice.preset}|${hk(history)}`;

/** Adds the browser entries to a state's engines and drops browser seats the server has given another engine. */
function adopt(data) {
  if (data.engines) for (const entry of ENTRIES) if (!data.engines.some(e => e.id === entry.id)) data.engines.push(entry);
  if (data.seats && data.seats.some((seat, side) => config.seats[side] && seat.engine !== 'human')) {
    config.seats = config.seats.map((choice, side) => data.seats[side].engine === 'human' ? choice : null);
    save();
  }
}

/** Browser analyses of the positions in `data.history`, as server evaluation records. */
function inject(data) {
  if (!config.analysis || !data.history || !data.evaluations) return;
  for (let ply = 0; ply <= data.history.length; ply++) {
    const record = analyses.get(analysisKey(config.analysis, data.history.slice(0, ply)));
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
  const fraction = () => job.loading ?? job.fraction;
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
  const side = s.player, choice = config.seats[side];
  const move = s.winner < 0 && choice && !s.paused && !s.match?.active && s.seats[side].engine === 'human';
  const prefix = s.history.slice(0, viewed());
  const analyse = config.analysis && !(s.winner >= 0 && prefix.length === s.history.length)
    && !analyses.has(analysisKey(config.analysis, prefix));
  const task = move ? {kind: 'move', side, choice, history: s.history.map(p => [...p])}
    : analyse ? {kind: 'analyse', ply: prefix.length, choice: config.analysis, history: prefix.map(p => [...p])} : null;
  const key = task && `${task.kind}|${analysisKey(task.choice, task.history)}`;
  if (job?.key === key || (key && key === failed)) return;
  job?.controller.abort();
  job = null;
  if (task) run(key, task);
  progress();
}

async function run(key, task) {
  const controller = new AbortController(), current = job = {...task, key, controller, fraction: 0, loading: 0};
  const {engine, presets, record} = BY_ID.get(task.choice.engine), budget = presets[task.choice.preset];
  try {
    await engine.load(f => { current.loading = f; progress(); });
    current.loading = null;
    const result = await engine.turn(task.history, budget, {signal: controller.signal,
      progress: f => { current.fraction = f; progress(); }});
    if (job !== current) return;
    job = null;
    if (task.kind === 'move') {
      posting = true;
      try {
        for (const [q, r] of result.moves) {
          if (state().paused || config.seats[task.side] !== task.choice || hk(state().history) !== hk(task.history)) break;
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
      const evaluation = {...record(result, budget), engine: task.choice.engine};
      analyses.set(analysisKey(task.choice, task.history), evaluation);
      const s = state();
      if (s && hk(s.history.slice(0, task.history.length)) === hk(task.history)) {
        s.evaluations[task.history.length] = evaluation;
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
      config.seats[body.side] = BY_ID.has(body.engine) ? choose(config.seats[body.side], body.engine) : null;
      save();
      if (job?.kind === 'move' && job.side === body.side && job.choice !== config.seats[body.side]) {
        job.controller.abort();
        job = null;
      }
      if (BY_ID.has(body.engine)) return original.post('/seat', {side: body.side, engine: 'human'});
    }
    if (path === '/analysis' && body.engine !== undefined) {
      config.analysis = BY_ID.has(body.engine) ? choose(config.analysis, body.engine) : null;
      save();
      if (BY_ID.has(body.engine)) {
        if (s?.analysis?.auto) return original.post('/analysis', {...body, engine: s.analysis.engine, auto: false});
        page.renderPanels();
        return Promise.resolve(s);
      }
    }
    if (path === '/analyse' && config.analysis) {
      if (body.force && s) {
        const ply = Number.isInteger(body.ply) ? body.ply : viewed();
        analyses.delete(analysisKey(config.analysis, s.history.slice(0, ply)));
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
    if (side >= 0 && seat.engine === 'human' && config.seats[side]) return [BY_ID.get(config.seats[side].engine).kind, label(config.seats[side])];
    if (s && seat === s.analysis && config.analysis) return [BY_ID.get(config.analysis.engine).kind, label(config.analysis)];
    return original.shown(seat);
  };
  page.canPlace = () => original.canPlace() && !config.seats[state().player];
  page.renderSeat = side => {
    original.renderSeat(side);
    const box = document.getElementById('seat' + side), s = state(), choice = config.seats[side];
    if (!choice || !box || s.saved_game) return;
    const send = change => { config.seats[side] = {...choice, preset: change.preset}; save(); page.renderPanels(); };
    const pick = box.querySelector('.pick');
    if (pick) {
      const items = [{id: 'human', ids: ['human'], kind: 'you', label: null}, ...original.pickItems(() => true)];
      pick.onclick = () => original.openMenu(pick, items, choice.engine, it => page.post('/seat', {side, engine: it.id}));
    }
    box.append(original.el('div', {class: 'more'}, original.el('div', {}, original.strength({preset: choice.preset}, send, 'seat' + side, null))));
    progress();
  };
  page.renderEngineHead = () => {
    original.renderEngineHead();
    const head = document.getElementById('engine-head'), s = state(), choice = config.analysis;
    if (!choice || !head || !s?.analysis) return;
    const pick = original.el('button', {class: 'pick'}, ...original.badge(BY_ID.get(choice.engine).kind, label(choice)));
    const items = original.pickItems(e => e.kind === 'bubble' || e.browser);
    pick.onclick = () => original.openMenu(pick, items, choice.engine, it => page.post('/analysis', {engine: it.id, checkpoint: null, preset: 'standard', auto: true}));
    const send = change => { config.analysis = {...choice, preset: change.preset}; save(); page.renderPanels(); };
    head.replaceChildren(original.el('div', {class: 'head'}, pick), original.strength({preset: choice.preset}, send, 'analysis', null));
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
 * controls are hidden, Bubble (browser) takes seat O and analysis on a first visit, and the page asks for
 * cross-origin isolation. Resolves true when it took over (or is reloading for isolation).
 */
async function serverless() {
  try {
    const response = await fetch('/state', {cache: 'no-store'});
    if (response.ok && (response.headers.get('Content-Type') || '').includes('json')) return false;
  } catch {}
  if (await isolate()) return true;
  const session = await OfflineSession.create({engine: 'browser:bubble', preset: 'quick', budget: BUBBLE.quick});
  const server = globalThis.fetch;
  globalThis.fetch = (input, init) => {
    const url = new URL(typeof input === 'string' ? input : input.url, location.href);
    if (url.origin !== location.origin || !OfflineSession.handles(url.pathname)) return server(input, init);
    const [status, data] = session.answer(url.pathname, init?.body ? JSON.parse(init.body) : {});
    return Promise.resolve(new Response(JSON.stringify(data), {status, headers: {'Content-Type': 'application/json'}}));
  };
  if (fresh) config.seats = [null, {engine: 'browser:bubble', preset: 'standard'}];
  config.analysis ??= {engine: 'browser:bubble', preset: 'quick'};
  save();
  document.head.append(original.el('style', {}, ['#review-go', '#copy', '#more', '#import', '[aria-label="Tournaments"]']
    .map(selector => `.serverless ${selector}`).join(',') + '{display:none!important}'));
  document.documentElement.classList.add('serverless');
  page.accept(session.state());
  return true;
}

if (HOOKS.every(name => typeof original[name] === 'function')) {
  install();
  serverless();
} else console.warn('Browser engines need the play page functions:', HOOKS.filter(name => typeof original[name] !== 'function'));
