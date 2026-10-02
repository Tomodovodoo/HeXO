/* Static-host Play session. Engines register a turn callback; the session owns jobs, books, clocks and saves. */
import {OfflineSession} from './offline.mjs';
import {PlayStorage} from './storage.mjs';
import {OpeningBook} from './openings.mjs';
import {readGame, exportGame, htttx} from './notation.mjs';

const playerAt = ply => ply === 0 ? 0 : ((ply - 1 >> 1) + 1) % 2;
const copy = value => structuredClone(value), position = history => history.map(p => p.join(',')).join(';');
const uid = () => globalThis.crypto.randomUUID(), human = () => ({engine: 'human'});
const REVIEW_PRESET = 'standard';
const starts = length => [0, ...Array.from({length: Math.ceil(Math.max(0, length - 1) / 2)}, (_, i) => 2 * i + 1)];

export function review(history, lookup, winner = -1) {
  const turns = [], ss = starts(history.length);
  for (let i = 0; i < ss.length; i++) {
    const ply = ss[i], end = ss[i + 1] ?? history.length, me = playerAt(ply), stones = history.slice(ply, end);
    if (stones.length < (ply ? 2 : 1) && !(end === history.length && winner === me)) break;
    const turn = {ply, player: me, stones, label: null, before: null, after: null, better: null, line: null};
    turns.push(turn);
    if (end === history.length && winner === me) { turn.label = 'win'; continue; }
    const before = lookup(history.slice(0, ply)), after = lookup(history.slice(0, end));
    if (!before || !after) continue;
    turn.before = before.value; turn.after = 1 - after.value;
    const had = before.proof?.winner, has = after.proof?.winner, loss = turn.before - turn.after;
    const best = before.moves?.length && JSON.stringify(before.moves.map(p => p.join(',')).sort()) === JSON.stringify(stones.map(p => p.join(',')).sort());
    turn.label = had === 1 - me ? 'lost' : had === me ? has === me || best ? 'kept' : 'missed' : has === 1 - me ? 'allowed'
      : has === me ? 'found' : best ? 'best' : loss < .05 ? 'good' : loss < .1 ? 'inaccuracy' : loss < .2 ? 'mistake' : 'blunder';
    if (['inaccuracy', 'mistake', 'blunder', 'missed', 'allowed'].includes(turn.label) && before.moves?.length) {
      turn.better = before.moves;
      turn.line = before.pv?.length ? before.pv : before.line?.length ? before.line : [...before.moves.map(p => [...p, me]), ...(lookup([...history.slice(0, ply), ...before.moves])?.moves || []).map(p => [...p, 1 - me])];
    }
  }
  return turns;
}

export function pairElo(results) {
  const ordered = [...results].sort((a, b) => a.game - b.game), pairs = [];
  for (let i = 0; i + 1 < ordered.length; i += 2) if (ordered[i].game % 2 === 1 && ordered[i + 1].game === ordered[i].game + 1) {
    pairs.push((ordered[i].winner === 0 ? 1 : ordered[i].winner === 1 ? 0 : .5) + (ordered[i + 1].winner === 0 ? 1 : ordered[i + 1].winner === 1 ? 0 : .5));
  }
  if (!pairs.length) return null;
  const p = (pairs.reduce((a, b) => a + b, 0) + .5) / (2 * pairs.length + 1), elo = 400 * Math.log10(p / (1 - p));
  const variance = pairs.reduce((n, v) => n + (v / 2 - p) ** 2, .25) / (pairs.length + 1);
  const sd = 400 / Math.LN10 / (p * (1 - p)) * Math.sqrt(variance / pairs.length);
  return {a_minus_b: elo, interval: [elo - 1.96 * sd, elo + 1.96 * sd], pairs: pairs.length, method: 'paired approximation'};
}

export class BrowserSession extends OfflineSession {
  static async create(analysis, {storage, id = 'live', book} = {}) {
    const rules = await OfflineSession.create(analysis), session = new BrowserSession(rules.native, analysis);
    session.storage = storage ?? await PlayStorage.open(); session.id = id;
    if (book) session.bookData = new OpeningBook(book);
    await session.restore();
    return session;
  }
  constructor(native, analysis = {}) {
    super(native, analysis);
    this.storage = new PlayStorage(null); this.id = 'live'; this.seats = [human(), human()];
    this.entries = new Map(); this.adapters = new Map(); this.cache = new Map(); this.index = new Map(); this.jobs = [];
    this.bookData = null; this.book = {enabled: false, mode: 'wide', opening: null}; this.coverage = {};
    this.match = null; this.saved_game = null; this.clock = null; this.gameId = uid(); this.gameCreated = new Date().toISOString(); this.records = []; this.gameSignature = null;
    this.running = null; this.idle = Promise.resolve(); this.importing = false; this.nextJob = 0; this.onchange = () => {}; this.saving = Promise.resolve(); this.storageError = null;
  }
  registerEngine(entry, adapter) {
    this.entries.set(entry.id, entry); this.adapters.set(entry.id, adapter);
    for (const seat of [...this.seats, this.analysis].filter(Boolean)) if (seat.engine === entry.id) Object.assign(seat, this.spec(seat));
    this.changed(); this.pump();
  }
  spec(input) {
    if (input.engine === 'human') return human();
    const entry = this.entries.get(input.engine);
    if (!entry) throw Error('This engine is not installed in the browser');
    const preset = input.preset || 'standard', budget = preset === 'custom' ? {...entry.presets.standard, ...input.custom, ...input.budget} : entry.presets[preset];
    if (!budget) throw Error('Unknown strength preset');
    for (const [name, value] of Object.entries(budget)) if (typeof value === 'number' && (!Number.isInteger(value) || value < 0 || value > ({simulations: 65536, solver_nodes: 4000000, ms: 120000, nodes: 50000000}[name] ?? 50000000))) throw Error(`Invalid ${name} budget`);
    return {engine: input.engine, checkpoint: input.checkpoint ?? null, preset, budget: copy(budget), auto: input.auto ?? false};
  }
  engineKey(spec) { return [spec.engine, spec.checkpoint, this.entries.get(spec.engine)?.version || ''].join('|'); }
  cacheKey(history, spec) { return `${this.engineKey(spec)}|${JSON.stringify(spec.budget)}|${position(history)}`; }
  lookup(history, spec = this.analysis, exact = false) {
    if (!spec) return null;
    const hit = this.cache.get(this.cacheKey(history, spec));
    if (exact) return hit || null;
    return (this.index.get(`${this.engineKey(spec)}|${position(history)}`) || [])
      .sort((a, b) => Boolean(b.proof) - Boolean(a.proof) || b.simulations - a.simulations || b.solver_nodes - a.solver_nodes || (b.budget?.ms || 0) - (a.budget?.ms || 0) || (b.budget?.nodes || 0) - (a.budget?.nodes || 0))[0] || null;
  }
  state() {
    const {winner, player, remaining} = this.native.game(this.history), evaluations = {};
    for (let ply = 0; ply <= this.history.length; ply++) {
      const prefix = this.history.slice(0, ply), record = this.lookup(prefix) || this.records.findLast(r => r.position === position(prefix) && (!this.analysis || r.engine_key === this.engineKey(this.analysis)));
      if (record) evaluations[ply] = record;
    }
    return {instance: `browser:${this.id}`, revision: this.revision, history: copy(this.history), player, remaining, winner,
      paused: this.paused, seats: copy(this.seats), analysis: copy(this.analysis), engines: [...this.entries.values()], match: this.match,
      clock: this.clockNow(), saved_game: this.saved_game, models_folder: null, importing: this.importing, storage: {persistent: !!this.storage.db, error: this.storageError},
      book: {available: !!this.bookData, ...this.book, count: this.bookData?.nodes.length, on_policy: this.bookData?.pool('wide').length, refreshed_by: this.bookData?.data.refreshed_by},
      evaluations, review: review(this.history, h => this.lookup(h, this.reviewSpec(), true), winner), review_preset: REVIEW_PRESET,
      jobs: this.jobs.filter(j => !j.controller.signal.aborted).map(({id, kind, status, done, total, error, history, side}) => ({id, kind, status, done, total, error, ply: history.length, side}))};
  }
  static handles(path) { path = path.replace(/^\/study/, ''); return OfflineSession.handles(path) || ['/storage', '/openings'].some(p => path === p || path.startsWith(p + '/')); }
  answer(path, body = {}) {
    try { this.apply(path.replace(/^\/study/, ''), body); return [200, this.state()]; }
    catch (error) { return [400, {error: error.message}]; }
  }
  changed() { this.revision++; this.deepen(); this.onchange(this.state()); this.persist(); }
  reviewSpec() { return this.analysis && this.entries.has(this.analysis.engine) ? this.spec({...this.analysis, preset: REVIEW_PRESET}) : null; }
  deepening() {
    return this.analysis?.auto && !this.paused && this.native.game(this.history).winner < 0 && !this.match?.active && this.seats.some(s => this.adapters.has(s.engine));
  }
  deepen() {
    const active = this.deepening(), current = position(this.history);
    this.cancelJobs(j => j.tier && (!active || position(j.history) !== current));
    if (!active || this.jobs.some(j => j.tier && j.status !== 'failed') || this.lookup(this.history)?.proof) return;
    const entry = this.entries.get(this.analysis.engine);
    if (!entry || !this.adapters.has(entry.id)) return;
    for (const tier of Object.keys(entry.presets)) {
      const spec = this.spec({...this.analysis, preset: tier});
      if (!this.lookup(this.history, spec, true) && !this.jobs.some(j => j.tier === tier && j.key === `analyse|${this.cacheKey(this.history, spec)}` && j.status === 'failed')) {
        this.enqueue('analyse', this.history, spec, {tier}); return;
      }
    }
  }
  snapshot() {
    return {id: this.id, history: copy(this.history), seats: copy(this.seats), analysis: copy(this.analysis), paused: this.paused,
      book: copy(this.book), match: copy(this.match), saved_game: copy(this.saved_game), clock: this.clockNow(), gameId: this.gameId, gameCreated: this.gameCreated, records: copy(this.records)};
  }
  persist() {
    if (this.importing) return this.saving;
    const snapshot = this.snapshot(), freeplay = this.freeplay();
    this.saving = this.saving.then(async () => {
      await this.storage.put('sessions', snapshot);
      if (freeplay) { await this.storage.put('games', freeplay.game); await this.storage.put('matches', freeplay.summary); }
    }).catch(error => { this.gameSignature = null; this.storageError = `Could not save in this browser: ${error.message}`; this.onchange(this.state()); });
    return this.saving;
  }
  async restore() {
    const [saved, coverage, evaluations] = await Promise.all([this.storage.get('sessions', this.id), this.storage.get('coverage', 'book'), this.storage.all('evaluations')]);
    this.cache.clear(); this.index.clear();
    for (const r of evaluations) this.indexRecord(r);
    this.coverage = coverage?.counts || {};
    if (saved) {
      this.native.game(saved.history); Object.assign(this, {...saved, paused: true});
      if (this.match) this.match.active = false;
      if (this.clock) { delete this.clock.started; delete this.clock.side; }
      const game = this.gameId && await this.storage.get('games', this.gameId);
      if (game) { const {saved_at, ...content} = game; this.gameSignature = JSON.stringify(content); }
    }
  }
  cancelJobs(predicate = () => true) {
    for (const job of this.jobs) if (predicate(job)) job.controller.abort();
    this.jobs = this.jobs.filter(j => !j.controller.signal.aborted);
  }
  editable() { if (this.match?.active) throw Error('Stop the match before changing its players or position'); }
  load(history, paused = false) {
    this.native.game(history); this.cancelJobs(); this.history = copy(history); this.records = []; this.paused = paused; this.saved_game = null;
    this.clock = null; this.gameId = uid(); this.gameCreated = new Date().toISOString(); this.gameSignature = null;
  }
  forkGame() {
    if (this.saved_game || !this.gameId || this.match) {
      this.saved_game = null; this.match = null; this.clock = null; this.gameId = uid(); this.gameCreated = new Date().toISOString(); this.gameSignature = null;
    }
  }
  freeplay() {
    if (this.match || this.saved_game || !this.gameId) return null;
    const winner = this.native.game(this.history).winner, players = this.seats.map(s => ({...s, name: this.entries.get(s.engine)?.name || 'Human'}));
    const game = {id: this.gameId, format: 'bubble-replay', version: 1, history: copy(this.history), players, winner: winner < 0 ? null : winner,
      reason: winner < 0 ? 'saved' : 'six', records: copy(this.records), evaluations: this.state().evaluations, opening: copy(this.book.opening), created_at: this.gameCreated};
    const signature = JSON.stringify(game);
    if (signature === this.gameSignature) return null;
    this.gameSignature = signature; game.saved_at = new Date().toISOString();
    const summary = {id: this.gameId, name: game.created_at.slice(0, 19).replace('T', ' '), games: 1, completed: 1,
      players: players.map(p => p.name), wins: winner < 0 ? [0, 0] : [winner === 0 ? 1 : 0, winner === 1 ? 1 : 0], capped: 0,
      results: [{game: 1, winner: game.winner, reason: game.reason, placements: this.history.length, id: game.id}], single: true, kind: 'freeplay'};
    return {game, summary};
  }
  async saveGame() { await this.persist(); }
  apply(path, body) {
    if (['/new', '/undo', '/retry', '/seat'].includes(path)) this.editable();
    if (path === '/play') {
      const point = [body.q, body.r];
      if (!point.every(Number.isSafeInteger)) throw Error('Coordinates must be integers');
      if (this.native.game(this.history).winner >= 0) throw Error('The game has finished');
      this.native.game([...this.history, point]); this.forkGame(); this.history.push(point); this.paused = false;
    } else if (path === '/undo') {
      this.cancelJobs(); this.forkGame(); const people = body.people || [0, 1].filter(i => this.seats[i].engine === 'human'); this.history.pop();
      while (people.length && this.history.length && !(people.includes(playerAt(this.history.length)) && this.history.length % 2)) this.history.pop();
      this.paused = true; this.saved_game = null;
    } else if (path === '/new') {
      this.match = null; let history = []; this.book.opening = null;
      if (this.book.enabled && this.bookData) {
        const side = this.seats.findIndex(s => s.engine === 'human'), n = this.bookData.pick(this.book.mode, this.coverage, side);
        history = n.moves; this.book.opening = {mode: this.book.mode, key: n.key, symmetry: n.symmetry};
        const k = `${side}:${n.key}`; this.coverage[k] = (this.coverage[k] || 0) + 1;
        this.storage.put('coverage', {id: 'book', counts: copy(this.coverage)}).catch(e => { this.storageError = e.message; });
      }
      this.load(history);
    } else if (path === '/retry') {
      if (!Number.isInteger(body.ply) || body.ply < 0 || body.ply > this.history.length) throw Error('Invalid retry position');
      this.load(this.history.slice(0, body.ply)); this.match = null;
    } else if (path === '/seat') {
      if (![0, 1].includes(body.side)) throw Error('Invalid seat');
      this.cancelJobs(j => j.kind === 'move'); this.seats[body.side] = this.spec({...this.seats[body.side], ...body, budget: body.preset === 'custom' ? body.custom : undefined});
    } else if (path === '/analysis') {
      this.cancelJobs(j => j.kind !== 'move'); this.analysis = this.spec({...this.analysis, ...body, budget: body.preset === 'custom' ? body.custom : undefined});
    } else if (path === '/analyse') {
      const ply = body.ply ?? this.history.length;
      if (!Number.isInteger(ply) || ply < 0 || ply > this.history.length) throw Error('Invalid analysis position');
      if (body.force || ply !== this.history.length || !this.deepening()) this.enqueue('analyse', this.history.slice(0, ply), this.analysis, {force: !!body.force});
    } else if (path === '/review') {
      if (!this.analysis || !this.adapters.has(this.analysis.engine)) throw Error('Choose an analysis engine');
      const history = copy(this.history), plies = starts(history.length);
      if (this.native.game(history).winner < 0 && !plies.includes(history.length)) plies.push(history.length);
      this.enqueue('review', history, this.reviewSpec(), {plies, cursor: 0, total: plies.length});
    } else if (path === '/cancel') {
      if (this.jobs.some(j => j.id === body.id && j.kind === 'move')) { this.paused = true; this.freezeClock(); }
      this.cancelJobs(j => j.id === body.id);
    } else if (path === '/pause') {
      this.paused = Boolean(body.paused);
      if (this.paused) { this.freezeClock(); this.cancelJobs(j => j.kind === 'move'); }
      else if (this.match && this.match.completed < this.match.games) this.resumeMatch();
    } else if (path === '/book') {
      if (!this.bookData) throw Error('The opening book has not loaded');
      if (body.mode) { this.bookData.pool(body.mode); this.book.mode = body.mode; }
      if ('enabled' in body) this.book.enabled = !!body.enabled;
    } else if (!['/state', '/rescan', '/models'].includes(path)) throw Error('Unknown Play request');
    if (path !== '/state') { this.changed(); queueMicrotask(() => this.pump()); }
  }
  enqueue(kind, history, spec, fields = {}) {
    if (this.importing) return;
    if (!spec || !this.adapters.has(spec.engine)) throw Error('Choose a browser engine');
    const key = `${kind}|${this.cacheKey(history, spec)}`;
    if (fields.force) this.cancelJobs(j => j.key === key);
    if (this.jobs.some(j => j.key === key)) return;
    if (kind === 'analyse' && !fields.force && this.lookup(history, spec, true)) return;
    if (kind === 'analyse') this.cancelJobs(j => j.kind === 'analyse' && j.status !== 'failed' && (fields.tier ? j.tier : true));
    this.jobs.push({id: ++this.nextJob, kind, history: copy(history), spec: copy(spec), key, controller: new AbortController(), status: 'queued', done: 0, total: 1, ...fields});
  }
  async record(history, spec, result) {
    const record = {...result, id: this.cacheKey(history, spec), position: position(history), engine: spec.engine, engine_key: this.engineKey(spec),
      simulations: spec.budget.simulations ?? result.simulations ?? spec.budget.visits ?? 0, solver_nodes: result.solved === false ? 0 : spec.budget.solver_nodes ?? result.solver_nodes ?? 0, budget: copy(spec.budget), saved_at: new Date().toISOString()};
    this.indexRecord(record); await this.storage.put('evaluations', record);
    if (position(this.history.slice(0, history.length)) === position(history)) this.records.push(record);
    return record;
  }
  indexRecord(record) {
    this.cache.set(record.id, record);
    const key = `${record.engine_key}|${record.position}`, values = (this.index.get(key) || []).filter(r => r.id !== record.id);
    values.push(record); this.index.set(key, values);
  }
  checkedTurn(history, moves) {
    const player = this.native.game(history).player, out = [];
    for (const p of moves) {
      if (!Array.isArray(p) || p.length !== 2 || !p.every(Number.isSafeInteger)) throw Error('Engine returned invalid coordinates');
      out.push(p); const state = this.native.game([...history, ...out]);
      if (state.winner >= 0 || state.player !== player) return out;
    }
    throw Error('Engine returned an incomplete turn');
  }
  clockNow() {
    if (!this.clock) return null;
    const clock = {...this.clock};
    if (clock.started != null) { const field = clock.side ? 'circle_ms' : 'cross_ms'; clock[field] = Math.max(0, clock[field] - (Date.now() - clock.started)); }
    return clock;
  }
  freezeClock() { this.clock = this.clockNow(); if (this.clock) { delete this.clock.started; delete this.clock.side; } }
  async pump() {
    if (this.running || this.importing) return;
    const state = this.native.game(this.history), seat = this.seats[state.player];
    if (!this.paused && state.winner < 0 && this.adapters.has(seat.engine) && !this.jobs.some(j => j.kind === 'move')) this.enqueue('move', this.history, seat, {side: state.player});
    const job = this.jobs.find(j => j.status === 'queued' && j.kind === 'move') || this.jobs.find(j => j.status === 'queued' && j.kind === 'analyse' && !j.tier)
      || this.jobs.find(j => j.status === 'queued' && j.kind === 'review') || this.jobs.find(j => j.status === 'queued' && j.tier);
    if (!job) return;
    let settled;
    this.idle = new Promise(resolve => { settled = resolve; });
    this.running = job; job.status = 'running'; this.changed();
    let timer;
    const match = job.kind === 'move' && this.match?.active ? this.match : null;
    const history = job.kind === 'review' ? job.history.slice(0, job.plies[job.cursor]) : job.history;
    try {
      const adapter = this.adapters.get(job.spec.engine);
      await adapter.ready?.(f => { job.done = f * .1; this.onchange(this.state()); });
      if (job.controller.signal.aborted) throw new DOMException('Cancelled', 'AbortError');
      const start = Date.now(); let timeout = false, limit = null;
      if (job.kind === 'move' && this.match?.active) {
        const c = this.match.clock;
        limit = c.mode === 'move' ? c.ms : c.mode === 'game' ? this.clock[job.side ? 'circle_ms' : 'cross_ms'] : null;
        if (c.mode === 'game') Object.assign(this.clock, {started: start, side: job.side});
        if (limit != null) timer = setTimeout(() => { timeout = true; job.controller.abort(); }, Math.max(1, limit));
        job.controller.signal.addEventListener('abort', () => clearTimeout(timer), {once: true});
      }
      let result = job.kind !== 'move' && !job.force ? this.lookup(history, job.spec, true) : null;
      try {
        result ||= await adapter.turn(copy(history), copy(job.spec.budget), {signal: job.controller.signal, checkpoint: job.spec.checkpoint, preset: job.spec.preset, ms: limit,
          progress: f => { job.done = job.kind === 'review' ? job.cursor + f : f; this.onchange(this.state()); }});
      } catch (e) { if (!timeout) throw e; }
      clearTimeout(timer);
      const elapsed = Date.now() - start;
      if (job.kind === 'move') {
        if (match && (this.match !== match || !match.active || this.paused || !this.jobs.includes(job))) return;
        this.freezeClock();
        if (timeout || limit != null && elapsed > limit) { await this.finishMatch(1 - job.side, 'timeout'); return; }
      }
      if (job.controller.signal.aborted) throw new DOMException('Cancelled', 'AbortError');
      await this.record(history, job.spec, result);
      if (job.kind === 'move') {
        if (job.controller.signal.aborted || position(this.history) !== position(history) || this.paused) return;
        if (!this.match?.active) this.forkGame();
        this.history.push(...copy(this.checkedTurn(history, result.moves)));
        if (this.clock) this.clock[job.side ? 'circle_ms' : 'cross_ms'] += this.match.clock.increment_ms;
        if (this.match?.active) {
          this.match.timings.push({ply: history.length, side: job.side, engine: job.spec.engine, elapsed_ms: elapsed});
          const winner = this.native.game(this.history).winner;
          if (winner >= 0) await this.finishMatch(winner, 'six');
        }
      } else if (job.kind === 'review') { job.cursor++; job.done = job.cursor; }
    } catch (error) {
      if (error.name !== 'AbortError') { job.status = 'failed'; job.error = error.message; if (job.kind === 'move') this.paused = true; if (this.match) this.match.error = error.message; }
    } finally {
      clearTimeout(timer); this.running = null;
      if (job.status !== 'failed' && (job.kind !== 'review' || job.cursor >= job.plies.length || job.controller.signal.aborted)) this.jobs = this.jobs.filter(j => j !== job);
      else if (job.status !== 'failed') job.status = 'queued';
      this.changed(); await this.saving;
      if (!this.importing && this.match && !this.match.active) await this.storage.put('matches', copy(this.match)).catch(error => { this.storageError = error.message; });
      settled();
      queueMicrotask(() => this.pump());
    }
  }
  async startMatch(body) {
    if (this.match?.active) throw Error('Stop the current match first');
    if (!Array.isArray(body.players) || body.players.length !== 2 || !Number.isInteger(body.games) || body.games < 2 || body.games > 10000 || body.games % 2) throw Error('Choose two engines and an even number of games');
    const players = body.players.map(p => { const s = this.spec(p), entry = this.entries.get(s.engine); return {...s, name: entry.name, version: entry.version}; });
    const seed = body.seed ?? Math.floor(Math.random() * 4294967296), mode = body.opening_range || 'origin';
    const openings = mode === 'origin' ? (body.openings || [[[0, 0]]]).map(moves => ({moves})) : this.bookData?.select(mode, body.unique_openings || body.games / 2, seed);
    if (!openings?.length) throw Error('No opening book is available');
    openings.forEach(n => this.native.game(n.moves));
    const clock = {...body.clock || {mode: 'fixed'}};
    if (!['fixed', 'move', 'game'].includes(clock.mode)) throw Error('Unknown clock mode');
    if (clock.mode === 'move' && (!Number.isInteger(clock.ms) || clock.ms < 1)) throw Error('Seconds per turn must be positive');
    if (clock.mode === 'game') {
      const m = String(clock.tc).match(/^(\d+(?:\.\d+)?)(?:\+(\d+(?:\.\d+)?))?$/);
      if (!m || +m[1] <= 0) throw Error('Use seconds+increment, such as 180+2');
      clock.initial_ms = +m[1] * 1000; clock.increment_ms = +(m[2] || 0) * 1000;
    }
    await this.saveGame(); this.cancelJobs();
    this.match = {id: uid(), name: new Date().toISOString().slice(0, 19).replace('T', ' '), active: true, players, games: body.games,
      completed: 0, current: 1, wins: [0, 0], capped: 0, results: [], openings, opening_range: mode, seed, clock, timings: [], elo: null};
    this.beginMatchGame(); await this.storage.put('matches', copy(this.match)); this.changed(); this.pump();
  }
  beginMatchGame() {
    const match = this.match, number = match.completed + 1;
    this.load(match.openings[Math.floor((number - 1) / 2) % match.openings.length].moves);
    this.seats = copy(number % 2 ? match.players : [...match.players].reverse()); match.current = number; match.timings = []; match.pending_game = false;
    if (match.clock.mode === 'game') this.clock = {cross_ms: match.clock.initial_ms, circle_ms: match.clock.initial_ms};
  }
  resumeMatch() {
    if (!this.match || this.match.completed >= this.match.games) throw Error('No unfinished match');
    for (const p of this.match.players) if (p.version && this.entries.get(p.engine)?.version !== p.version) throw Error('This match used a different engine version. Start a new match.');
    if (this.match.pending_game) this.beginMatchGame();
    this.jobs = this.jobs.filter(j => j.status !== 'failed'); this.match.error = null; this.match.active = true; this.paused = false;
  }
  async finishMatch(winner, reason) {
    const m = this.match, game = m.current, aWinner = winner == null ? null : game % 2 ? winner : 1 - winner, id = `${m.id}:${game}`;
    const record = {id, format: 'bubble-replay', version: 1, game, history: copy(this.history), players: copy(this.seats), winner, reason,
      evaluations: this.state().evaluations, records: copy(this.records), timings: copy(m.timings), saved_at: new Date().toISOString()};
    await this.storage.put('games', record);
    m.results.push({id, game, winner: aWinner, reason, placements: this.history.length, opening: Math.floor((game - 1) / 2)});
    if (winner == null) m.capped++; else m.wins[aWinner]++;
    m.completed++; m.elo = pairElo(m.results);
    if (m.completed >= m.games) { m.active = false; if (this.match === m) { this.paused = true; this.clock = null; } }
    else if (this.match === m && m.active) this.beginMatchGame();
    else { m.pending_game = true; m.current = m.completed + 1; }
    await this.storage.put('matches', copy(m));
  }
  async catalogue() {
    await this.saving;
    return (await this.storage.all('matches')).sort((a, b) => b.name.localeCompare(a.name)).map(m => ({...m, players: m.players.map(p => typeof p === 'string' ? p : p.name)}));
  }
  async savedReplay(id, game) {
    await this.saving;
    const match = await this.storage.get('matches', id), result = match?.results.find(r => r.game === +game), record = result && await this.storage.get('games', result.id);
    if (!record) throw Error('No such saved game in this browser');
    return record;
  }
  async openGame(id, number) {
    const game = await this.savedReplay(id, number);
    this.load(game.history, true); this.seats = [human(), human()];
    this.saved_game = {batch: id, game: +number, players: game.players.map(p => p.name || 'Human'), winner: game.winner, reason: game.reason};
    this.records = game.records || Object.values(game.evaluations || {});
    if (this.analysis) this.analysis.auto = false;
    this.changed();
  }
  async request(input, body = {}, method = 'GET') {
    const url = new URL(input, 'https://play.invalid'), path = url.pathname.replace(/^\/study/, ''), q = url.searchParams;
    try {
      let data, type = 'application/json';
      if (path === '/state' && method === 'GET' && q.has('since') && +q.get('since') === this.revision) {
        const state = this.state(); data = {instance: state.instance, revision: state.revision, jobs: state.jobs, clock: state.clock};
      } else if (path === '/matches' && method === 'GET') data = {matches: await this.catalogue()};
      else if (path === '/matches/game') { data = await this.savedReplay(q.get('batch'), q.get('game')); if (q.get('format') === 'htttx') { data = htttx(data.history).text; type = 'text/plain'; } }
      else if (path === '/matches/open') { await this.openGame(body.batch, body.game); data = {url: `?study=1&batch=${encodeURIComponent(body.batch)}&game=${body.game}`}; }
      else if (path === '/matches/delete') {
        if (this.match?.id === body.batch && this.match.active) throw Error('Stop the match before deleting it');
        if (this.match?.id === body.batch) { this.cancelJobs(); await this.idle; this.match = null; this.gameId = null; this.changed(); }
        if (this.gameId === body.batch) { this.gameId = null; this.changed(); }
        await this.saving;
        const match = await this.storage.get('matches', body.batch);
        for (const r of match?.results || []) await this.storage.delete('games', r.id);
        await this.storage.delete('matches', body.batch); data = this.state();
      } else if (path === '/export') data = exportGame(this.history.slice(0, q.has('ply') ? +q.get('ply') : this.history.length), q.get('format') || 'htttx');
      else if (path === '/htttx' || path === '/match/replay') { data = htttx(path === '/htttx' ? this.history : (await this.savedReplay(this.match.id, q.get('game'))).history).text; type = 'text/plain'; }
      else if (path === '/replay') data = {format: 'bubble-replay', version: 1, history: this.history, players: this.seats.map(p => ({...p, name: this.entries.get(p.engine)?.name || 'Human'})), evaluations: this.state().evaluations, records: this.records};
      else if (path === '/evaluations') { data = this.records.map(r => JSON.stringify(r)).join('\n'); type = 'application/x-ndjson'; }
      else if (path === '/storage/backup') { await this.saving; data = await this.storage.backup(); }
      else if (path === '/storage/save') { await this.saveGame(); data = this.state(); }
      else if (path === '/openings') data = {refreshed_by: this.bookData?.data.refreshed_by, nodes: this.bookData.select(q.get('range') || 'wide', +(q.get('count') || 8), +(q.get('seed') || 0))};
      else if (path === '/import') {
        this.editable(); const json = body.text.trim().startsWith('{') ? JSON.parse(body.text) : null;
        if (json?.format === 'hexo-browser-save') {
          this.importing = true; this.paused = true; this.cancelJobs();
          try {
            await this.idle; await this.saving; await this.storage.restore(json, this.native); await this.restore();
            for (const spec of [...this.seats, this.analysis].filter(Boolean)) if (this.entries.has(spec.engine)) Object.assign(spec, this.spec(spec));
          } finally { this.importing = false; }
          this.changed(); data = this.state();
        }
        else {
          const history = await readGame(body.text, this.native); await this.saveGame(); this.match = null; this.load(history, true);
          this.records = json?.records || Object.values(json?.evaluations || {});
          for (const r of this.records) if (r.id && r.engine_key) { this.indexRecord(r); await this.storage.put('evaluations', r); }
          this.changed(); data = this.state();
        }
      } else if (path === '/match/results' || path === '/match' && method === 'GET') data = this.match;
      else if (path === '/match') {
        if (!body.action || body.action === 'start') await this.startMatch(body);
        else if (body.action === 'resume') {
          if (body.batch && this.match?.id !== body.batch) {
            this.editable(); const saved = await this.storage.get('matches', body.batch);
            if (!saved || saved.single || saved.completed >= saved.games) throw Error('No unfinished match');
            for (const p of saved.players) if (p.version && this.entries.get(p.engine)?.version !== p.version) throw Error('This match used a different engine version. Start a new match.');
            this.match = saved;
            this.match.players = this.match.players.map(p => ({...p, ...this.spec(p)})); this.beginMatchGame();
          }
          this.resumeMatch();
        } else if (body.action === 'pause' || body.action === 'stop') {
          this.freezeClock(); this.cancelJobs(); this.paused = true; if (this.match) this.match.active = false;
        } else throw Error('Unknown match action');
        if (this.match) await this.storage.put('matches', copy(this.match));
        this.changed(); this.pump(); data = this.state();
      } else {
        if (path === '/new') await this.saveGame();
        const [status, value] = this.answer(path, body);
        if (path === '/play' && this.native.game(this.history).winner >= 0) await this.saveGame();
        await this.saving; return [status, value, type];
      }
      await this.saving; return [200, data, type];
    } catch (error) { return [400, {error: error.message}, 'application/json']; }
  }
}
