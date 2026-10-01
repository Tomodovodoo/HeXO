/* "Bubble (browser)" for web/index.html: a seat and an analysis engine that run entirely in this browser.
 * The server sees a browser seat as a human seat and receives its stones through /play; like server engines a
 * browser seat waits while the game is paused, and cancelling its move pauses the game. Browser analyses are shown
 * in place of the server's for the positions they cover. A task that failed is not retried until the position, preset
 * or engine choice changes. Choices persist per browser (localStorage). */
import {BubbleEngine, PRESETS} from './bubble.mjs';

const ID = 'browser:bubble', LABEL = 'Bubble (browser)', ENTRY = {id: ID, kind: 'bubble', name: LABEL, label: LABEL, checkpoints: []};
const STORE = 'bubble-browser';
const HOOKS = ['accept', 'post', 'shown', 'renderSeat', 'renderEngineHead', 'renderJobs', 'canPlace', 'renderPanels', 'draw',
  'openMenu', 'el', 'toast', 'badge', 'strength', 'pickItems'];
const page = globalThis, original = Object.fromEntries(HOOKS.map(name => [name, page[name]]));
const engine = new BubbleEngine(), analyses = new Map(), hk = history => history.map(p => p.join(',')).join(';');
let config = {seats: [null, null], analysis: null}, job = null, failed = null, posting = false, loaded = 0;
try { config = {...config, ...JSON.parse(localStorage.getItem(STORE))}; } catch {}
const save = () => { failed = null; try { localStorage.setItem(STORE, JSON.stringify(config)); } catch {} };
const state = () => typeof S === 'undefined' ? null : S;
const viewed = () => typeof view === 'undefined' ? 0 : view;
const closeIcon = () => typeof icon === 'function' ? icon('close') : '×';

/** Adds the browser entry to a state's engines and drops browser seats the server has given another engine. */
function adopt(data) {
  if (data.engines && !data.engines.some(e => e.id === ID)) data.engines.push(ENTRY);
  if (data.seats && data.seats.some((seat, side) => config.seats[side] && seat.engine !== 'human')) {
    config.seats = config.seats.map((preset, side) => data.seats[side].engine === 'human' ? preset : null);
    save();
  }
}

/** Browser analyses of the positions in `data.history`, as server evaluation records. */
function inject(data) {
  if (!config.analysis || !data.history || !data.evaluations) return;
  for (let ply = 0; ply <= data.history.length; ply++) {
    const record = analyses.get(`${config.analysis}|${hk(data.history.slice(0, ply))}`);
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
  const fraction = () => engine.device ? job.fraction : loaded;
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
  const side = s.player, preset = config.seats[side];
  const move = s.winner < 0 && preset && !s.paused && !s.match?.active && s.seats[side].engine === 'human';
  const prefix = s.history.slice(0, viewed());
  const analyse = config.analysis && !(s.winner >= 0 && prefix.length === s.history.length)
    && !analyses.has(`${config.analysis}|${hk(prefix)}`);
  const task = move ? {kind: 'move', side, preset, history: s.history.map(p => [...p])}
    : analyse ? {kind: 'analyse', ply: prefix.length, preset: config.analysis, history: prefix.map(p => [...p])} : null;
  const key = task && `${task.kind}|${task.preset}|${hk(task.history)}`;
  if (job?.key === key || (key && key === failed)) return;
  job?.controller.abort();
  job = null;
  if (task) run(key, task);
  progress();
}

async function run(key, task) {
  const controller = new AbortController(), current = job = {...task, key, controller, fraction: 0};
  try {
    await engine.load(f => { loaded = f; progress(); });
    const result = await engine.turn(task.history, PRESETS[task.preset], {signal: controller.signal,
      progress: f => { current.fraction = f; progress(); }});
    if (job !== current) return;
    job = null;
    if (task.kind === 'move') {
      posting = true;
      try {
        for (const [q, r] of result.moves) {
          if (state().paused || hk(state().history) !== hk(task.history) || !(await original.post('/play', {q, r}))) break;
          task.history.push([q, r]);
        }
      } finally {
        posting = false;
      }
    } else {
      const {simulations, solver_nodes} = PRESETS[task.preset];
      const record = {...result, simulations, solver_nodes: result.solved ? solver_nodes : 0, engine: ID};
      analyses.set(`${task.preset}|${hk(task.history)}`, record);
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
      config.seats[body.side] = body.engine === ID ? config.seats[body.side] || 'standard' : null;
      save();
      if (body.engine === ID) return original.post('/seat', {side: body.side, engine: 'human'});
    }
    if (path === '/analysis' && body.engine !== undefined) {
      config.analysis = body.engine === ID ? config.analysis || 'standard' : null;
      save();
      if (body.engine === ID) {
        if (s?.analysis?.auto) return original.post('/analysis', {...body, engine: s.analysis.engine, auto: false});
        page.renderPanels();
        return Promise.resolve(s);
      }
    }
    if (path === '/analyse' && config.analysis) return Promise.resolve(s);
    return original.post(path, body);
  };
  page.shown = seat => {
    const s = state(), side = s ? s.seats.indexOf(seat) : -1;
    if ((side >= 0 && seat.engine === 'human' && config.seats[side]) || (s && seat === s.analysis && config.analysis)) return [ENTRY.kind, LABEL];
    return original.shown(seat);
  };
  page.canPlace = () => original.canPlace() && !config.seats[state().player];
  page.renderSeat = side => {
    original.renderSeat(side);
    const box = document.getElementById('seat' + side), s = state();
    if (!config.seats[side] || !box || s.saved_game) return;
    const send = change => { config.seats[side] = change.preset; save(); page.renderPanels(); };
    box.append(original.el('div', {class: 'more'}, original.el('div', {}, original.strength({preset: config.seats[side]}, send, 'seat' + side, null))));
    progress();
  };
  page.renderEngineHead = () => {
    original.renderEngineHead();
    const head = document.getElementById('engine-head'), s = state();
    if (!config.analysis || !head || !s?.analysis) return;
    const pick = original.el('button', {class: 'pick'}, ...original.badge(ENTRY.kind, LABEL));
    const items = original.pickItems(e => e.kind === 'bubble');
    pick.onclick = () => original.openMenu(pick, items, ID, it => page.post('/analysis', {engine: it.id, checkpoint: null, preset: 'standard', auto: true}));
    const send = change => { config.analysis = change.preset; save(); page.renderPanels(); };
    head.replaceChildren(original.el('div', {class: 'head'}, pick), original.strength({preset: config.analysis}, send, 'analysis', null));
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

if (HOOKS.every(name => typeof original[name] === 'function')) install();
else console.warn('Bubble (browser) needs the play page functions:', HOOKS.filter(name => typeof original[name] !== 'function'));
