/* Browser engines for web/index.html: seats and analysis engines that run entirely in this browser.
 * The server sees a browser seat as a human seat and receives its stones through /play; like server engines a
 * browser seat waits while the game is paused, cancelling its move pauses the game, and Undo steps back over its
 * turns to the people's last turn. Browser analyses are shown in place of the server's for the positions they cover.
 * A task that failed is not retried until the position, preset or engine choice changes. Choices persist per browser
 * (localStorage).
 *
 * ENGINES lists them. Each is {entry, engine, record, build, listed?}: `listed` settles once `entry.checkpoints` is
 * read (the saved choices are then checked against it and the page redraws), `build` is the command that builds its files into this
 * checkout (named when a saved choice needs files the public site does not serve; no engine list shows such an engine),
 * `entry` is its picker entry ({id, kind, name, label,
 * checkpoints, presets}, with `badge` when the bot is not its kind and `analysis: true` when it can analyse), `engine.load(progress)` starts it (progress(fraction,
 * stage) while it loads, stages.mjs's stages), `engine.files()` lists the files it downloads (assets.mjs records, for the picker's download
 * button) and `engine.turn(history, budget, {signal, progress})` (progress(fraction, live, stage)) resolves to its turn {moves, ...} at a
 * preset's budget, plus `checkpoint` (one of `entry.checkpoints`, chosen in a select when there are several) when the
 * entry lists any, rejecting with an AbortError when `signal` aborts; `record(result, history, preset)` is the
 * evaluation record the analysis panel shows for that turn. A seat or the analysis shows the stage of its job next to
 * its bar, and an engine's fallback notice (engine-worker.mjs `notices`) is a toast; a fallback to the CPU moves the
 * engine's choices to lightning. */
import {BubbleEngine, NETWORKS, PRESETS, isolate, networkManifest} from './bubble.mjs';
import {drip} from './drip.mjs';
import {shrimp} from './shrimp.mjs';
import {mountPlay, deviceLabel} from './browser-play.mjs';
import {turnTime} from './clock.mjs';
import {NEURAL_PRESET, notePace} from './device.mjs';
import {seal} from './seal.mjs';
import {six} from './six.mjs';
import {strix} from './strix.mjs';
import {NotOnSite, install as download, json, status} from './assets.mjs';
import {notices} from './engine-worker.mjs';
import {stageText} from './stages.mjs';
import {savedIds} from './storage.mjs';

const BUBBLE = 'browser:bubble', bubbleLabel = 'Bubble (browser)';
const bubble = {entry: {id: BUBBLE, kind: 'bubble', name: bubbleLabel, label: bubbleLabel, checkpoints: NETWORKS.map(n => n.name), presets: PRESETS, preset: NEURAL_PRESET, analysis: true, clocks: true},
  engine: new BubbleEngine(),
  record: (result, history, preset) => ({...result, simulations: PRESETS[preset].simulations,
    solver_nodes: result.solved ? PRESETS[preset].solver_nodes : 0, engine: BUBBLE}),
  build: 'python tools/build_web.py ort model'};
const ENGINES = new Map([bubble, drip, shrimp, seal, six, strix].map(e => [e.entry.id, e]));
const STORE = 'browser-engines';
const HOOKS = ['accept', 'post', 'shown', 'renderSeat', 'renderEngineHead', 'renderJobs', 'canPlace', 'renderPanels', 'draw',
  'openMenu', 'el', 'toast', 'badge', 'strength', 'pickItems', 'isHuman', 'setupRing', 'clockPicker'];
const page = globalThis, original = Object.fromEntries(HOOKS.map(name => [name, page[name]]));
const analyses = new Map(), loads = new Map(), hk = history => history.map(p => p.join(',')).join(';');
/** Each seat and the analysis: null, or {engine: an ENGINES id, preset, checkpoint}. */
let config = {seats: [null, null], analysis: null}, job = null, failed = null, posting = false;
/** Each seat's line, the key of the game tree its Bubble searches (worker.mjs); undo, a new or loaded game and a seat change renew it. */
const lines = [0, 1];
let nextLine = 2;
const renew = (...sides) => { for (const side of sides.length ? sides : [0, 1]) lines[side] = nextLine++; };
let fresh = true, notice = null;
/** Engine and checkpoint pairs whose network is loaded, with the engine's `ready` promise of the worker that loaded it (an
 * engine that discards its worker on cancel starts a new one): a timed move of any other holds the server's clock while it
 * loads. */
const warmed = new Map();
/** While a timed move holds the server's clock to load its engine: {paused}, set when the person pauses meanwhile. */
let holding = null;
try {
  const saved = localStorage.getItem(STORE), known = choice => ENGINES.has(choice?.engine) ? pickEngine(choice.engine, choice, choice.checkpoint) : null;
  fresh = saved === null;
  const stored = {...config, ...savedIds(JSON.parse(saved))};
  config = {seats: stored.seats.map(known), analysis: known(stored.analysis)};
} catch {}
const save = () => { failed = null; try { localStorage.setItem(STORE, JSON.stringify(config)); } catch {} };
const state = () => typeof S === 'undefined' ? null : S;
const viewed = () => typeof view === 'undefined' ? 0 : view;
const closeIcon = () => typeof icon === 'function' ? icon('close') : '×';
/** The choice for browser engine `id`, keeping `current`'s preset and network when it chose the same engine, else
 * at the engine's starting preset. A network
 * is checked against the entry's checkpoints once they are known; before its manifest arrives it is kept as given. */
function pickEngine(id, current, checkpoint = null) {
  const same = current?.engine === id, {checkpoints} = ENGINES.get(id).entry, wanted = [checkpoint, same ? current.checkpoint : null];
  const network = checkpoints.length ? wanted.find(c => checkpoints.includes(c)) ?? checkpoints[0] : wanted.find(Boolean) ?? null;
  return {engine: id, preset: same ? current.preset : ENGINES.get(id).entry.preset || 'standard', checkpoint: network};
}
const analysable = e => e.kind === 'bubble' || e.analysis;
const analysisKey = (choice, history) => `${choice.engine}|${choice.preset}|${choice.checkpoint}|${hk(history)}`;

/**
 * Adds the browser entries to a state's engines and drops browser seats the server has given another engine. A server
 * without an analysis engine (no Bubble model) gets the browser's: Drip (browser) at quick unless another was chosen.
 */
function adopt(data) {
  if (data.engines) for (const {entry} of ENGINES.values()) if (!data.engines.some(e => e.id === entry.id)) data.engines.push(entry);
  if (data.seats && data.seats.some((seat, side) => config.seats[side] && seat.engine !== 'human')) {
    config.seats = config.seats.map((choice, side) => data.seats[side].engine === 'human' ? choice : null);
    save();
  }
  if (data.analysis === null) {
    if (!config.analysis) {
      config.analysis = {engine: drip.entry.id, preset: 'quick'};
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
  const words = (id, active) => { const label = document.getElementById(id); if (label) label.textContent = active ? job.stage ?? '' : ''; };
  for (const side of [0, 1]) {
    if (!config.seats[side]) continue;
    const active = job?.kind === 'move' && job.side === side, cancel = document.getElementById('cancel' + side);
    bar(document.getElementById('busy' + side), active ? fraction() : null);
    words('stage' + side, active);
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
    words('analysis-stage', active);
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
  const task = move ? {kind: 'move', side, ...seat, line: lines[side], history: s.history.map(p => [...p])}
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
    const timed = task.kind === 'move' && state()?.clock_spec && state().clock_spec.mode !== 'fixed', warm = `${task.engine}|${task.checkpoint}`;
    const hold = timed && (!engine.ready || warmed.get(warm) !== engine.ready);
    if (hold) { posting = true; holding = {paused: false}; await original.post('/pause', {paused: true}); }
    const report = (f, stage) => {
      if (stage?.provider) entry.device = deviceLabel(stage);   // once the probe has finished
      loads.set(task.engine, f);
      current.stage = stageText(stage);
      progress();
    };
    try {
      entry.device = deviceLabel(await engine.load(report));
      checks.delete(task.engine);
      await engine.prepare?.(task.checkpoint, {signal: controller.signal, progress: (f, live, stage) => report(f, stage)});
      warmed.set(warm, engine.ready);
    } finally {
      if (hold) {
        const kept = holding.paused;
        holding = null;
        if (!kept) await original.post('/pause', {paused: false});
        posting = false;
      }
    }
    if (job !== current || hold && state()?.paused) {
      if (job === current) job = null;
      else if (!job) schedule();   // a CPU fallback ended this job while the clock was held: start its replacement
      return;
    }
    current.loading = false;
    current.stage = stageText(null);
    const budget = {...entry.presets[task.preset], ...(task.checkpoint ? {checkpoint: task.checkpoint} : {})}, s = state();
    const ms = task.kind === 'move' && s?.clock && s.clock_spec?.mode !== 'fixed' ? turnTime(s.clock_spec, s.clock, task.side) : null;
    const started = performance.now();
    const result = await engine.turn(task.history, budget, {signal: controller.signal, ms, line: task.line,
      progress: (f, live, stage) => { current.fraction = f; current.stage = stageText(stage); progress(); }});
    if (ms == null) notePace(entry, task.preset, performance.now() - started, result.moves?.length);
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
      if (result.solver_error && notice !== result.solver_error) original.toast(`The solver could not run in this browser (${notice = result.solver_error}), so evaluations have no proofs`);
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
      if ((await check(task.engine)).state === 'unpublished') unpublished(task.engine);   // a worker's error is a plain message
      else original.toast(error.message);
      // The server counts a browser seat as a person, so its clock would run on into a loss on time.
      if (task.kind === 'move' && state()?.clock_spec && state().clock_spec.mode !== 'fixed') original.post('/pause', {paused: true});
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
  out.push(original.strength(choice, change => send({preset: change.preset}), id, null, -1, ENGINES.get(choice.engine).entry));
  return out;
}

/* Engine id -> a promise of its files' assets.mjs status(), {state: 'unpublished'} when neither this origin nor the
 * site has one of them (only a local build provides it), or {state: 'failed', error} for another failure. Its
 * `stamp` names the networks it covered; a different saved choice checks again. */
const checks = new Map(), downloads = new Map();
const failure = error => error instanceof NotOnSite ? {state: 'unpublished'} : {state: 'failed', error: error.message};
/** The networks the seat and analysis choices pick for engine `id`, which its files() should cover: the browser
 * session's on a static page, else the saved ones. */
const chosen = id => {
  const session = page.browserPlay, choices = session ? [...session.seats, session.analysis] : [...config.seats, config.analysis];
  return [...new Set(choices.filter(c => c?.engine === id && c.checkpoint).map(c => c.checkpoint))];
};
const remember = (id, promise) => { checks.set(id, Object.assign(promise, {stamp: chosen(id).join(',')})); return promise; };
const check = id => checks.get(id)?.stamp === chosen(id).join(',') ? checks.get(id)
  : remember(id, ENGINES.get(id).engine.files(chosen(id)).then(files => { if (ENGINES.get(id).listed) recheck(ENGINES.get(id).entry); return status(files); }).catch(failure));
/** Ids of the browser engines whose files are on neither this origin nor the site: only a local build provides them. */
const absent = new Set();
/** Whether an engine list offers engine entry `entry`: not when it is an absent browser engine. A choice saved earlier
 * keeps its entry in the play state, so its seat still shows its name and controls. */
export const listed = entry => !absent.has(entry.id);
/** The page's pickItems(filter), whose items every engine list (seats, analysis, tournament) shows, keeping listed entries. */
const pickItems = filter => original.pickItems(entry => filter(entry) && listed(entry));
/** Checks every browser engine's files and takes those on neither origin off the engine lists, redrawing the page when
 * it took one off; resolves once all are checked. */
export async function survey() {
  const states = await Promise.all([...ENGINES.keys()].map(async id => [id, (await check(id)).state]));
  const gone = states.filter(([id, found]) => found === 'unpublished' && !absent.has(id));
  for (const [id] of gone) absent.add(id);
  if (gone.length && state()) page.renderPanels();
}
const unpublished = id => original.toast(`${ENGINES.get(id).entry.label} is not on the public site; build it here with ${ENGINES.get(id).build}`);
/* States in which a browser engine plays without a download first: its files are here, cached, or cannot be cached. */
const READY = new Set(['local', 'cached', 'uncached']);
const megabytes = bytes => `${(bytes / 1e6).toFixed(bytes < 1e7 ? 1 : 0)} MB`;

/** Draws picker row `row` of a browser engine: as it is when its files are here or cached, else as a download with
 * its size, and while downloading with its progress. A row whose files are on neither origin leaves the menu. */
async function paint(row) {
  const id = row.dataset.engine, found = await check(id), running = downloads.get(id);
  if (!row.isConnected) return;
  row.querySelectorAll('.get, .size').forEach(node => node.remove());
  const ready = !running && READY.has(found.state);
  row.classList.remove('setup', 'running', 'failed');
  row.removeAttribute('aria-label');
  if (ready) return;
  if (found.state === 'unpublished') {
    row.remove();
    survey();
    return;
  }
  row.classList.add('setup', ...(running ? ['running'] : found.state === 'failed' ? ['failed'] : []));
  const total = found.bytes ? megabytes(found.bytes) : '';
  const size = !running ? total : total ? `${megabytes(running.fraction * found.bytes)} / ${total}` : `${Math.round(running.fraction * 100)}%`;
  row.setAttribute('aria-label', `Download ${ENGINES.get(id).entry.label}`);
  row.append(original.el('span', {class: 'size', style: 'white-space:nowrap'}, size), original.setupRing(running ? {state: 'running', progress: running.fraction} : null));
}

/** Downloads browser engine `id`'s missing files into the Cache API; a failure shows its reason. */
async function fetchEngine(id) {
  if (downloads.has(id)) return;
  const slot = {fraction: 0};
  downloads.set(id, slot);
  const repaint = () => document.querySelectorAll(`#menu [data-engine="${id}"]`).forEach(paint);
  try {
    if (['failed', 'unpublished'].includes((await check(id)).state)) checks.delete(id);   // the site or a build may have changed
    repaint();
    const found = await check(id), stamp = checks.get(id).stamp;
    if (found.state === 'unpublished') { unpublished(id); return; }
    if (found.state === 'failed') throw new Error(found.error);
    if (found.state === 'missing') await download(found.files, fraction => { slot.fraction = fraction; repaint(); });
    checks.delete(id);
    const after = await check(id);
    if (checks.get(id).stamp === stamp && !READY.has(after.state)) {   // the same files did not stay (a full quota): each load downloads them
      remember(id, Promise.resolve({state: 'uncached'}));
      original.toast(`${ENGINES.get(id).entry.label}: the browser did not keep the files, so each start downloads them`);
    }
    if (failed?.split('|')[1] === id) failed = null;   // let this engine's work that failed for want of its files run again
    const session = page.browserPlay;
    if (session) {
      session.jobs = session.jobs.filter(job => job.status !== 'failed' || job.spec.engine !== id);
      session.changed();
      session.pump();
    } else schedule();
  } catch (error) {
    remember(id, Promise.resolve(failure(error)));
    if (error instanceof NotOnSite) unpublished(id);
    else original.toast(`${ENGINES.get(id).entry.label}: ${error.message}`);
  } finally {
    downloads.delete(id);
    repaint();
  }
}

/** The page's engine picker, with a browser engine's row offering its download until its files are here. */
function openMenu(anchor, items, current, choose) {
  original.openMenu(anchor, items, current, choose);
  const rows = [...document.getElementById('menu').children];
  items.forEach((item, i) => {
    if (!ENGINES.has(item.id)) return;
    const row = rows[i], pick = row.onclick;
    row.dataset.engine = item.id;
    row.onclick = async event => {
      const {state} = await check(item.id);
      if (!row.isConnected) return;   // the menu closed or changed while the status was read
      if (!downloads.has(item.id) && READY.has(state)) pick(event);
      else fetchEngine(item.id);
    };
    paint(row);
  });
}

/** Checks the saved choices and the static page's session choices (seats, analysis, a stored match's players) for
 * engine `entry` against its loaded network list: a network the list no longer has becomes its newest. Then redraws.
 * Runs when the list changed since the last check, or always with `force`. */
const rechecked = new Map();
function recheck(entry, force = false) {
  const list = entry.checkpoints.join(',');
  if (!force && rechecked.get(entry.id) === list) return;   // nothing new to check against
  rechecked.set(entry.id, list);
  const fix = choice => choice?.engine === entry.id ? pickEngine(entry.id, choice, choice.checkpoint) : choice;
  const fixed = {seats: config.seats.map(fix), analysis: fix(config.analysis)};
  if (JSON.stringify(fixed) !== JSON.stringify(config)) {   // save() also clears the failed key, so only on a change
    config = fixed;
    save();
  }
  if (!page.browserPlay?.relist(entry.id) && state()) page.renderPanels();
}

/** After browser engine `engine` left WebGPU for WebAssembly: lightning becomes its starting preset, and the saved
 * choices (or the static page's session) that use it at another preset move to lightning, ending a job of it at
 * another preset. */
function lighten(engine) {
  const found = [...ENGINES.values()].find(e => e.engine === engine);
  if (!found) return;
  const id = found.entry.id;
  found.entry.preset = 'lightning';
  if (page.browserPlay) { page.browserPlay.lighten(id); return; }
  const light = choice => choice?.engine === id ? {...choice, preset: 'lightning'} : choice;
  config = {seats: config.seats.map(light), analysis: light(config.analysis)};
  save();
  if (job?.engine === id && job.preset !== 'lightning') {
    job.controller.abort();
    job = null;
  }
  if (state()) page.renderPanels();
}

function install() {
  notices.addEventListener('notice', ({detail}) => {
    original.toast(detail.text);
    if (detail.cpu) lighten(detail.engine);
  });
  page.openMenu = openMenu;
  page.pickItems = pickItems;
  page.accept = data => {
    adopt(data);
    inject(data);
    original.accept(data);
    schedule();
  };
  page.post = (path, body = {}) => {
    const s = state();
    if (['/undo', '/new', '/book', '/import', '/retry', '/match'].includes(path)) renew();
    if (path === '/seat') renew(body.side);
    if (path === '/pause' && holding) holding.paused = !!body.paused;
    const fixed = id => ENGINES.has(id) && !ENGINES.get(id).entry.clocks ? ENGINES.get(id).entry.name : null;
    const refused = name => { original.toast(`${name} plays a fixed budget; it cannot keep a clock`); return Promise.resolve(null); };
    if (path === '/clock' && body.mode !== 'fixed') {
      const name = config.seats.map(choice => choice && fixed(choice.engine)).find(Boolean);
      if (name) return refused(name);
    }
    if (path === '/seat' && fixed(body.engine) && s?.clock_spec && s.clock_spec.mode !== 'fixed') return refused(fixed(body.engine));
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
    if (['/undo', '/new', '/book'].includes(path) && s && config.seats.some(Boolean)) {
      return original.post(path, {...body, people: [0, 1].filter(side => s.seats[side].engine === 'human' && !config.seats[side])});
    }
    return original.post(path, body);
  };
  page.shown = seat => {
    const s = state(), side = s ? s.seats.indexOf(seat) : -1;
    const choice = side >= 0 && seat.engine === 'human' ? config.seats[side] : s && seat === s.analysis ? config.analysis : null;
    if (!choice) return original.shown(seat);
    const {entry} = ENGINES.get(choice.engine);
    return [entry.badge || entry.kind, entry.label, entry.device];
  };
  page.isHuman = seat => {
    const s = state(), side = s ? s.seats.indexOf(seat) : -1;
    return original.isHuman(seat) && !(side >= 0 && config.seats[side]);
  };
  page.canPlace = () => original.canPlace() && !config.seats[state().player];
  page.renderSeat = side => {
    original.renderSeat(side);
    const box = document.getElementById('seat' + side), s = state();
    if (!config.seats[side] || !box || s.saved_game) return;
    const send = change => { config.seats[side] = {...config.seats[side], ...change}; renew(side); save(); page.renderPanels(); };
    const pick = box.querySelector('.pick');
    if (pick) {
      const items = [{id: 'human', ids: ['human'], kind: 'human', label: null}, ...pickItems(() => true)];
      pick.onclick = () => page.openMenu(pick, items, config.seats[side].engine, it => page.post('/seat', {side, engine: it.id}));
    }
    const clock = ENGINES.get(config.seats[side].engine).entry.clocks && s.clock_spec
      ? [original.clockPicker(s.clock_spec, spec => page.post('/clock', spec), 'clock-seat' + side)] : [];
    box.append(original.el('div', {class: 'more'}, original.el('div', {}, ...controls(config.seats[side], send, 'seat' + side), ...clock)));
    progress();
  };
  page.renderEngineHead = () => {
    original.renderEngineHead();
    const head = document.getElementById('engine-head'), s = state();
    if (!config.analysis || !head || !s?.analysis) return;
    const {entry} = ENGINES.get(config.analysis.engine);
    const pick = original.el('button', {class: 'pick'}, ...original.badge(entry.badge || entry.kind, entry.label, entry.device));
    const items = pickItems(analysable);
    pick.onclick = () => page.openMenu(pick, items, entry.id, it => page.post('/analysis', {engine: it.id, checkpoint: null, preset: ENGINES.get(it.id)?.entry.preset || 'standard', auto: true}));
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
  Object.assign(page, original, {openMenu, pickItems});
  const [manifest, build] = await Promise.all([json(networkManifest()).then(found => found.data, () => ({})), json('build.json').then(found => found.data)]);
  // 'hybrid' keeps evaluations of the hybrid scheduler apart from those an older page saved with its per-leaf search.
  bubble.entry.version = ['hybrid', build.artefacts['gumbel.wasm'], build.artefacts['tactical.wasm']].join(':');
  bubble.entry.models = NETWORKS.length ? Object.fromEntries(NETWORKS.map(n => [n.name, n.model_version])) : {'': manifest.model_version};
  drip.entry.version = build.artefacts['native/native.wasm'];
  for (const {entry} of ENGINES.values()) entry.version ||= JSON.stringify(build.artefacts);
  await mountPlay(ENGINES, config);
  for (const {entry, listed} of ENGINES.values()) listed?.then(() => recheck(entry, true));   // the session's own choices
  return true;
}

if (HOOKS.every(name => typeof original[name] === 'function')) {
  install();
  survey();
  for (const {entry, listed} of ENGINES.values()) listed?.then(() => recheck(entry));
  serverless().then(active=>{if(!active||page.browserPlay)page.resolvePlayReady?.()}).catch(error=>{original.toast(error.message)});
} else console.warn('The browser engines need the play page functions:', HOOKS.filter(name => typeof original[name] !== 'function'));
