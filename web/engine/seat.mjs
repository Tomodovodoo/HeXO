/* "Bubble (browser)" for web/index.html: a seat and an analysis engine that run entirely in this browser.
 * The server sees a browser seat as a human seat and receives its stones through /play; browser analyses are
 * shown in place of the server's for the positions they cover. Choices persist per browser (localStorage). */
import {BubbleEngine, PRESETS} from './bubble.mjs';

const ID = 'browser:bubble', ENTRY = {id: ID, kind: 'bubble', name: 'Bubble (browser)', checkpoints: []};
const SHORT = {lightning: 'L', quick: 'Q', standard: 'S', strong: 'St', deep: 'D', dangerous: 'X'}, STORE = 'bubble-browser';
const HOOKS = ['accept', 'post', 'engineName', 'renderSeat', 'renderEngineHead', 'renderJobs', 'canPlace', 'renderPanels', 'draw',
  'openMenu', 'closeMenu', 'el', 'toast'];
const page = globalThis, original = Object.fromEntries(HOOKS.map(name => [name, page[name]]));
const engine = new BubbleEngine(), analyses = new Map(), hk = history => history.map(p => p.join(',')).join(';');
let config = {seats: [null, null], analysis: null}, job = null, posting = false, loaded = 0;
try { config = {...config, ...JSON.parse(localStorage.getItem(STORE))}; } catch {}
const save = () => { try { localStorage.setItem(STORE, JSON.stringify(config)); } catch {} };
const state = () => typeof S === 'undefined' ? null : S;
const viewed = () => typeof view === 'undefined' ? 0 : view;

function presets(current, choose) {
  return original.el('div', {class: 'seg', style: `grid-template-columns:repeat(${Object.keys(PRESETS).length},1fr)`},
    Object.keys(PRESETS).map(name => {
      const button = original.el('button', {class: name === current ? 'on' : '', 'aria-label': name}, SHORT[name]);
      button.onclick = () => choose(name);
      return button;
    }));
}

/** Browser analyses of the positions in `data.history`, as server evaluation records. */
function inject(data) {
  if (!config.analysis || !data.history || !data.evaluations) return;
  for (let ply = 0; ply <= data.history.length; ply++) {
    const record = analyses.get(`${config.analysis}|${hk(data.history.slice(0, ply))}`);
    if (record) data.evaluations[ply] = record;
  }
}

function progress() {
  const s = state();
  if (!s) return;
  for (const side of [0, 1]) {
    const box = document.getElementById('busy' + side);
    if (!box || !config.seats[side]) continue;
    const active = job?.kind === 'move' && job.side === side;
    if (!active) { if (box.dataset.browser) { box.replaceChildren(); delete box.dataset.browser; } continue; }
    let bar = box.querySelector('.bar');
    if (!bar || !box.dataset.browser) {
      bar = original.el('div', {class: 'bar'}, original.el('i'));
      const cancel = original.el('button', {class: 'x', 'aria-label': 'Cancel'}, '×');
      cancel.onclick = () => { config.seats[side] = null; save(); job?.controller.abort(); original.renderPanels(); };
      box.replaceChildren(bar, cancel);
      box.dataset.browser = '1';
    }
    bar.firstChild.style.transform = `scaleX(${engine.device ? job.fraction : loaded})`;
  }
  document.getElementById('evalbar')?.classList.toggle('pending', job?.kind === 'analyse' && job.ply === viewed());
}

function schedule() {
  const s = state();
  if (!s || posting) return;
  const side = s.player, preset = config.seats[side];
  const move = s.winner < 0 && preset && !s.match?.active && s.seats[side].engine === 'human';
  const prefix = s.history.slice(0, viewed());
  const analyse = config.analysis && !(s.winner >= 0 && prefix.length === s.history.length)
    && !analyses.has(`${config.analysis}|${hk(prefix)}`);
  const task = move ? {kind: 'move', side, preset, history: s.history.map(p => [...p])}
    : analyse ? {kind: 'analyse', ply: prefix.length, preset: config.analysis, history: prefix.map(p => [...p])} : null;
  const key = task && `${task.kind}|${task.preset}|${hk(task.history)}`;
  if (job?.key === key) return;
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
          if (hk(state().history) !== hk(task.history) || !(await original.post('/play', {q, r}))) break;
          task.history.push([q, r]);
        }
      } finally {
        posting = false;
      }
    } else {
      const {simulations, solver_nodes} = PRESETS[task.preset];
      analyses.set(`${task.preset}|${hk(task.history)}`, {...result, simulations, solver_nodes, engine: ID});
      const s = state();
      if (s && hk(s.history.slice(0, task.history.length)) === hk(task.history)) {
        s.evaluations[task.history.length] = analyses.get(`${task.preset}|${hk(task.history)}`);
        original.renderPanels();
        original.draw();
      }
    }
  } catch (error) {
    if (error.name !== 'AbortError') original.toast(error.message);
    if (job === current) job = null;
  }
  schedule();
}

/** Adds the browser entry to a state's engines and drops browser seats the server has given another engine. */
function adopt(data) {
  if (data.engines && !data.engines.some(e => e.id === ID)) data.engines.push(ENTRY);
  if (data.seats && data.seats.some((seat, side) => config.seats[side] && seat.engine !== 'human')) {
    config.seats = config.seats.map((preset, side) => data.seats[side].engine === 'human' ? preset : null);
    save();
  }
}

function install() {
  page.accept = data => {
    adopt(data);
    inject(data);
    original.accept(data);
    schedule();
  };
  page.post = (path, body = {}) => {
    if (path === '/seat' && body.engine !== undefined) {
      config.seats[body.side] = body.engine === ID ? config.seats[body.side] || 'standard' : null;
      save();
      if (body.engine === ID) return original.post('/seat', {side: body.side, engine: 'human'});
    }
    if (path === '/analysis' && body.engine !== undefined) {
      config.analysis = body.engine === ID ? config.analysis || 'standard' : null;
      save();
      const s = state();
      if (body.engine === ID) {
        if (s?.analysis?.auto) return original.post('/analysis', {...body, engine: s.analysis.engine, auto: false});
        page.renderPanels();
        return Promise.resolve(s);
      }
    }
    return original.post(path, body);
  };
  page.engineName = seat => {
    const s = state(), side = s ? s.seats.indexOf(seat) : -1;
    if ((side >= 0 && seat.engine === 'human' && config.seats[side]) || (s && seat === s.analysis && config.analysis)) return [ENTRY.kind, ENTRY.name];
    return original.engineName(seat);
  };
  page.canPlace = () => original.canPlace() && !config.seats[state().player];
  page.renderSeat = side => {
    original.renderSeat(side);
    const box = document.getElementById('seat' + side), busy = document.getElementById('busy' + side);
    if (config.seats[side] && box && busy && !state().saved_game) {
      box.insertBefore(presets(config.seats[side], name => { config.seats[side] = name; save(); page.renderSeat(side); schedule(); }), busy);
    }
  };
  page.renderEngineHead = () => {
    original.renderEngineHead();
    const head = document.getElementById('engine-head'), s = state();
    if (!config.analysis || !head || !s) return;
    const pick = original.el('button', {class: 'pick'}, original.el('span', {class: 'kind'}, ENTRY.kind), original.el('span', {}, ENTRY.name));
    const items = s.engines.filter(e => e.kind === 'bubble').map(e => ({id: e.id, kind: e.kind, name: e.name}));
    pick.onclick = () => original.openMenu(pick, items, ID, it => page.post('/analysis', {engine: it.id, checkpoint: null, preset: 'standard', auto: true}));
    head.replaceChildren(original.el('div', {class: 'stack', style: 'flex:1;min-width:0'}, original.el('div', {class: 'head'}, pick),
      presets(config.analysis, name => { config.analysis = name; save(); page.renderEngineHead(); schedule(); })));
  };
  page.renderPanels = () => {
    original.renderPanels();
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
