/* Browser engines for web/index.html: seats and analysis engines that run entirely in this browser.
 * The server sees a browser seat as a human seat and receives its stones through /play; like server engines a
 * browser seat waits while the game is paused, cancelling its move pauses the game, and Undo steps back over its
 * turns to the people's last turn. Browser analyses are shown in place of the server's for the positions they cover.
 * A task that failed is not retried until the position, preset or engine choice changes. Choices persist per browser
 * (localStorage).
 *
 * ENGINES lists them. Each is {entry, engine, record}: `entry` is its picker entry ({id, kind, name, label,
 * checkpoints, presets}, with `badge` when the bot is not its kind and `analysis: true` when it can analyse), `engine.load(progress)` starts it (progress(fraction)
 * while it downloads) and `engine.turn(history, budget, {signal, progress})` resolves to its turn {moves, ...} at a
 * preset's budget, plus `checkpoint` (one of `entry.checkpoints`, chosen in a select when there are several) when the
 * entry lists any, rejecting with an AbortError when `signal` aborts; `record(result, history, preset)` is the
 * evaluation record the analysis panel shows for that turn. */
import {BubbleEngine, PRESETS, isolate} from './bubble.mjs';
import {native} from './native.mjs';
import {shrimp} from './shrimp.mjs';
import {mountPlay} from './browser-play.mjs';
import {seal} from './seal.mjs';
import {six} from './six.mjs';
import {strix} from './strix.mjs';

const BUBBLE = 'browser:bubble', bubbleLabel = 'Bubble (browser)';
const bubble = {entry: {id: BUBBLE, kind: 'bubble', name: bubbleLabel, label: bubbleLabel, checkpoints: [], presets: PRESETS, analysis: true},
  engine: new BubbleEngine(),
  record: (result, history, preset) => ({...result, simulations: PRESETS[preset].simulations,
    solver_nodes: result.solved ? PRESETS[preset].solver_nodes : 0, engine: BUBBLE})};
const ENGINES = new Map([bubble, native, shrimp, seal, six, strix].filter(Boolean).map(e => [e.entry.id, e]));
const STORE = 'browser-engines';
const HOOKS = ['accept', 'post', 'shown', 'renderSeat', 'renderEngineHead', 'renderJobs', 'canPlace', 'renderPanels', 'draw',
  'openMenu', 'el', 'toast', 'badge', 'strength', 'pickItems'];
const page = globalThis, original = Object.fromEntries(HOOKS.map(name => [name, page[name]]));
const analyses = new Map(), loads = new Map(), hk = history => history.map(p => p.join(',')).join(';');
/** Each seat and the analysis: null, or {engine: an ENGINES id, preset, checkpoint}. */
let config = {seats: [null, null], analysis: null}, job = null, failed = null, posting = false;
let fresh = true;
try {
  const saved = localStorage.getItem(STORE), known = choice => ENGINES.has(choice?.engine) ? pickEngine(choice.engine, choice, choice.checkpoint) : null;
  fresh = saved === null;
  const stored = {...config, ...JSON.parse(saved)};
  config = {seats: stored.seats.map(known), analysis: known(stored.analysis)};
} catch {}
const save = () => { failed = null; try { localStorage.setItem(STORE, JSON.stringify(config)); } catch {} };
const state = () => typeof S === 'undefined' ? null : S;
const viewed = () => typeof view === 'undefined' ? 0 : view;
const closeIcon = () => typeof icon === 'function' ? icon('close') : '×';
/** The choice for browser engine `id`, keeping `current`'s preset and network when it chose the same engine. */
function pickEngine(id, current, checkpoint = null) {
  const same = current?.engine === id, {checkpoints} = ENGINES.get(id).entry;
  const network = [checkpoint, same ? current.checkpoint : null].find(c => checkpoints.includes(c)) ?? checkpoints[0] ?? null;
  return {engine: id, preset: same ? current.preset : 'standard', checkpoint: network};
}
const analysable = e => e.kind === 'bubble' || e.analysis;
const analysisKey = (choice, history) => `${choice.engine}|${choice.preset}|${choice.checkpoint}|${hk(history)}`;

/**
 * Adds the browser entries to a state's engines and drops browser seats the server has given another engine. A server
 * without an analysis engine (no Bubble model) gets the browser's: Native (browser) at quick unless another was chosen.
 */
function adopt(data) {
  if (data.engines) for (const {entry} of ENGINES.values()) if (!data.engines.some(e => e.id === entry.id)) data.engines.push(entry);
  if (data.seats && data.seats.some((seat, side) => config.seats[side] && seat.engine !== 'human')) {
    config.seats = config.seats.map((choice, side) => data.seats[side].engine === 'human' ? choice : null);
    save();
  }
  if (data.analysis === null) {
    if (!config.analysis) {
      config.analysis = {engine: native.entry.id, preset: 'quick'};
      save();
    }
    const {engine, preset} = config.analysis;
    data.analysis = {engine, checkpoint: null, preset, budget: ENGINES.get(engine).entry.presets[preset], auto: false};
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
  const fraction = () => job.loading ? loads.get(job.engine) ?? 0 : job.fraction;
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
  const side = s.player, seat = config.seats[side], a = config.analysis;
  const move = s.winner < 0 && seat && !s.paused && !s.match?.active && s.seats[side].engine === 'human';
  const prefix = s.history.slice(0, viewed());
  const analyse = a && !(s.winner >= 0 && prefix.length === s.history.length) && !analyses.has(analysisKey(a, prefix));
  const task = move ? {kind: 'move', side, ...seat, history: s.history.map(p => [...p])}
    : analyse ? {kind: 'analyse', ply: prefix.length, ...a, history: prefix.map(p => [...p])} : null;
  const key = task && `${task.kind}|${task.engine}|${task.preset}|${task.checkpoint}|${hk(task.history)}`;
  if (job?.key === key || (key && key === failed)) return;
  job?.controller.abort();
  job = null;
  if (task) run(key, task);
  progress();
}

async function run(key, task) {
  const controller = new AbortController(), current = job = {...task, key, controller, fraction: 0};
  const {engine, entry, record} = ENGINES.get(task.engine);
  try {
    current.loading = true;
    await engine.load(f => { loads.set(task.engine, f); progress(); });
    current.loading = false;
    const budget = {...entry.presets[task.preset], ...(task.checkpoint ? {checkpoint: task.checkpoint} : {})};
    const result = await engine.turn(task.history, budget, {signal: controller.signal,
      progress: f => { current.fraction = f; progress(); }});
    if (job !== current) return;
    job = null;
    if (task.kind === 'move') {
      posting = true;
      try {
        for (const [q, r] of result.moves) {
          const seat = config.seats[task.side];
          if (state().paused || seat?.engine !== task.engine || seat.preset !== task.preset || seat.checkpoint !== task.checkpoint
            || hk(state().history) !== hk(task.history)) break;
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
      const evaluation = record(result, task.history, task.preset);
      analyses.set(analysisKey(task, task.history), evaluation);
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

/** A choice's network select (when its engine offers several) and strength slider; `send` takes {preset} or {checkpoint}. */
function controls(choice, send, id) {
  const {checkpoints} = ENGINES.get(choice.engine).entry, out = [];
  if (checkpoints.length > 1) {
    const select = original.el('select', {'aria-label': 'Network'}, checkpoints.map(c => original.el('option', {value: c}, c)));
    select.value = choice.checkpoint;
    select.onchange = () => send({checkpoint: select.value});
    out.push(select);
  }
  out.push(original.strength(choice, change => send({preset: change.preset}), id, null));
  return out;
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
      const browser = ENGINES.has(body.engine);
      config.seats[body.side] = browser ? pickEngine(body.engine, config.seats[body.side], body.checkpoint) : null;
      save();
      if (job?.kind === 'move' && job.side === body.side && job.engine !== config.seats[body.side]?.engine) {
        job.controller.abort();
        job = null;
      }
      if (browser) return original.post('/seat', {side: body.side, engine: 'human'});
    }
    if (path === '/analysis' && body.engine !== undefined) {
      const browser = ENGINES.has(body.engine);
      config.analysis = browser ? pickEngine(body.engine, config.analysis, body.checkpoint) : null;
      save();
      if (browser) {
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
    const choice = side >= 0 && seat.engine === 'human' ? config.seats[side] : s && seat === s.analysis ? config.analysis : null;
    if (!choice) return original.shown(seat);
    const {entry} = ENGINES.get(choice.engine);
    return [entry.badge || entry.kind, entry.label];
  };
  page.canPlace = () => original.canPlace() && !config.seats[state().player];
  page.renderSeat = side => {
    original.renderSeat(side);
    const box = document.getElementById('seat' + side), s = state();
    if (!config.seats[side] || !box || s.saved_game) return;
    const send = change => { config.seats[side] = {...config.seats[side], ...change}; save(); page.renderPanels(); };
    const pick = box.querySelector('.pick');
    if (pick) {
      const items = [{id: 'human', ids: ['human'], kind: 'you', label: null}, ...original.pickItems(() => true)];
      pick.onclick = () => original.openMenu(pick, items, config.seats[side].engine, it => page.post('/seat', {side, engine: it.id}));
    }
    box.append(original.el('div', {class: 'more'}, original.el('div', {}, ...controls(config.seats[side], send, 'seat' + side))));
    progress();
  };
  page.renderEngineHead = () => {
    original.renderEngineHead();
    const head = document.getElementById('engine-head'), s = state();
    if (!config.analysis || !head || !s?.analysis) return;
    const {entry} = ENGINES.get(config.analysis.engine);
    const pick = original.el('button', {class: 'pick'}, ...original.badge(entry.badge || entry.kind, entry.label));
    const items = original.pickItems(analysable);
    pick.onclick = () => original.openMenu(pick, items, entry.id, it => page.post('/analysis', {engine: it.id, checkpoint: null, preset: 'standard', auto: true}));
    const send = change => { config.analysis = {...config.analysis, ...change}; save(); page.renderPanels(); };
    head.replaceChildren(original.el('div', {class: 'head'}, pick), ...controls(config.analysis, send, 'analysis'));
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
 * Without a play server (a static host), mount the browser Play session and ask for cross-origin isolation.
 * Resolves true when it took over (or is reloading for isolation).
 */
async function serverless() {
  try {
    const response = await fetch('/state', {cache: 'no-store'});
    if (response.ok && (response.headers.get('Content-Type') || '').includes('json')) return false;
  } catch {}
  if (await isolate()) return true;
  Object.assign(page, original);
  const [manifest, build] = await Promise.all(['model/manifest.json', 'build.json'].map(async path => (await fetch(new URL(path, import.meta.url), {cache:'no-cache'})).json()));
  bubble.entry.version = [manifest.model_version, build.artefacts['gumbel.wasm'], build.artefacts['tactical.wasm']].join(':');
  native.entry.version = build.artefacts['native/native.wasm'];
  for (const {entry} of ENGINES.values()) entry.version ||= JSON.stringify(build.artefacts);
  await mountPlay(ENGINES, config);
  return true;
}

if (HOOKS.every(name => typeof original[name] === 'function')) {
  install();
  serverless().then(active=>{if(!active||page.browserPlay)page.resolvePlayReady?.()}).catch(error=>{original.toast(error.message)});
} else console.warn('The browser engines need the play page functions:', HOOKS.filter(name => typeof original[name] !== 'function'));
