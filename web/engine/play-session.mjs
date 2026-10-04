/* Static-host Play session. Engines register a turn callback; the session owns jobs, books, clocks and saves. */
import {OfflineSession} from './offline.mjs';
import {PlayStorage} from './storage.mjs';
import {OpeningBook} from './openings.mjs';
import {readGame, exportGame, htttx} from './notation.mjs';
import {clockSpec, turnTime} from './clock.mjs';
import {Proofs, proven, proofKey} from './proof.mjs';
import {stageText} from './stages.mjs';
import {PV_CHECK} from './search.mjs';

const REFRESH_PLIES = 4;  // earlier placements a finished analysis refreshes (python/play.py REFRESH_PLIES)
const REFRESH_ROUNDS = 3, REFRESH_MOVE = .05;  // further refreshes of one position while each still moves its result (python/play.py)
/** True when the evaluation `found` differs from the saved evaluation `before` in its stones or by more than REFRESH_MOVE in value. */
const moved = (found, before) => JSON.stringify((found.moves || []).map(p => p.join(',')).sort()) !== JSON.stringify((before.moves || []).map(p => p.join(',')).sort()) || Math.abs(found.value - before.value) > REFRESH_MOVE;

const playerAt = ply => ply === 0 ? 0 : ((ply - 1 >> 1) + 1) % 2;
const copy = value => structuredClone(value), position = history => history.map(p => p.join(',')).join(';');
const stones = text => text ? text.split(';').map(p => p.split(',').map(Number)) : [];
const uid = () => globalThis.crypto.randomUUID(), human = () => ({engine: 'human'});
/** The longest delay setTimeout keeps; a longer clock is checked again when it fires. */
const MAX_TIMER = 2 ** 31 - 1;
/** The engines take budgets as 32-bit signed integers. */
const MAX_BUDGET = 2 ** 31 - 1;
const FIELD_NAMES = {simulations: 'Search', solver_nodes: 'Solver', nodes: 'Positions'};
const starts = length => [0, ...Array.from({length: Math.ceil(Math.max(0, length - 1) / 2)}, (_, i) => 2 * i + 1)];

/** `promise`, or an AbortError as soon as `signal` aborts, so a job never waits on a load it no longer needs. */
function abortable(promise, signal) {
  return new Promise((resolve, reject) => {
    const abort = () => reject(new DOMException('Cancelled', 'AbortError'));
    if (signal.aborted) abort();
    signal.addEventListener('abort', abort, {once: true});
    Promise.resolve(promise).then(resolve, reject).finally(() => signal.removeEventListener('abort', abort));
  });
}

const UNGRADED = {label: null, before: null, after: null, better: null, line: null};

/** Labels every turn of `history`, and each of its stones in `grades`, as python/play.py review does; `lookup(prefix,
 * ply)` is the evaluation of a position, given `ply` when the prefix is `history.slice(0, ply)`. */
export function review(history, lookup, winner = -1) {
  const seen = new Map(), look = (prefix, ply) => {
    const key = position(prefix);
    if (!seen.has(key)) seen.set(key, lookup(prefix, ply));
    return seen.get(key);
  };
  const judge = (s, e, me) => {
    const stones = history.slice(s, e), grade = {...UNGRADED};
    if (e === history.length && winner === me) return {...grade, label: 'win'};
    const before = look(history.slice(0, s), s), after = look(history.slice(0, e), e);
    if (!before || !after) return grade;
    grade.before = before.value; grade.after = playerAt(e) === me ? after.value : 1 - after.value;
    const had = before.proof?.winner, has = after.proof?.winner, loss = grade.before - grade.after, engine = (before.moves || []).map(p => p.join(','));
    const best = engine.length > 0 && stones.every(p => engine.includes(p.join(',')));
    grade.label = had === 1 - me ? 'lost' : had === me ? has === me || best ? 'kept' : 'missed' : has === 1 - me ? 'allowed'
      : has === me ? 'found' : best ? 'best' : loss < .05 ? 'good' : loss < .1 ? 'inaccuracy' : loss < .2 ? 'mistake' : 'blunder';
    if (['inaccuracy', 'mistake', 'blunder', 'missed', 'allowed'].includes(grade.label) && engine.length && !best) {
      grade.better = before.moves.slice(0, stones.length);
      grade.line = before.pv?.length ? before.pv : before.line?.length ? before.line : [...before.moves.map(p => [...p, me]), ...(look([...history.slice(0, s), ...before.moves])?.moves || []).map(p => [...p, 1 - me])];
    }
    return grade;
  };
  const turns = [], ss = starts(history.length);
  for (let i = 0; i < ss.length; i++) {
    const ply = ss[i], end = ss[i + 1] ?? history.length, me = playerAt(ply);
    if (end === ply) break;
    const complete = end - ply === (ply ? 2 : 1) || end === history.length && winner === me;
    turns.push({ply, player: me, stones: history.slice(ply, end), ...(complete ? judge(ply, end, me) : UNGRADED),
      grades: Array.from({length: end - ply}, (_, j) => judge(ply + j, ply + j + 1, me))});
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
    this.bookData = null; this.book = {enabled: false, mode: 'narrow', opening: null}; this.coverage = {};
    this.match = null; this.saved_game = null; this.clock = null; this.timeControl = {mode: 'fixed'}; this.clockTurns = []; this.outcome = null; this.flag = null; this.clockPartial = 0; this.gameId = uid(); this.gameCreated = new Date().toISOString(); this.records = []; this.gameSignature = null;
    this.running = null; this.idle = Promise.resolve(); this.importing = false; this.nextJob = 0; this.onchange = () => {}; this.saving = Promise.resolve(); this.storageError = null;
    this.storageToken = null; this.initializing = false; this.dirty = false; this.conflicted = false; this.evaluationsVersion = 0; this.studied = null; this.notice = null; this.lines = [uid(), uid()]; this.analysisLine = uid(); this.graph = {generation: null, searches: 0};
    this.proofs = new Proofs(); this.provenRecords = new Map();
  }
  /** Adds a browser engine: `adapter.ready(progress, checkpoint)` loads it with that checkpoint's network (a timed
   * move's clock starts after it; `progress(fraction, stage)` with stages.mjs's stages) and `adapter.turn(history,
   * budget, options)` plays (`options.progress(fraction, live, stage)`). */
  registerEngine(entry, adapter) {
    this.entries.set(entry.id, entry); this.adapters.set(entry.id, adapter);
    for (const seat of [...this.seats, this.analysis].filter(Boolean)) if (seat.engine === entry.id) Object.assign(seat, this.spec(seat));
    this.changed(); this.pump();
  }
  /** After engine `id` left WebGPU for WebAssembly: lightning becomes its starting preset, and the seats and the
   * analysis that use it at another preset move to lightning (outside a running match), ending their jobs. */
  lighten(id) {
    const entry = this.entries.get(id);
    if (!entry) return;
    entry.preset = 'lightning';
    if (this.match?.active) return;
    const light = spec => spec?.engine === id && spec.preset !== 'lightning' ? this.spec({...spec, preset: 'lightning'}) : spec;
    this.seats = this.seats.map(light); this.analysis = light(this.analysis);
    this.cancelJobs(job => job.spec.engine === id && job.spec.preset !== 'lightning' && !job.tier && job.kind !== 'review');
    this.changed(); this.pump();
  }
  spec(input) {
    if (input.engine === 'human') return human();
    const entry = this.entries.get(input.engine);
    if (!entry) throw Error('This engine is not installed in the browser');
    const preset = input.preset || entry.preset || 'standard', budget = preset === 'custom' ? {...entry.presets.standard, ...input.custom, ...input.budget} : entry.presets[preset];
    if (!budget) throw Error('Unknown strength preset');
    for (const [name, value] of Object.entries(budget)) {
      const least = {ms: 10, nodes: 1, visits: 1}[name] ?? (entry.kind === 'strix' ? 1 : 0);
      if (typeof value === 'number' && (!Number.isInteger(value) || value < least || value > MAX_BUDGET)) throw Error(`${FIELD_NAMES[name] || name} must be a whole number from ${least} to ${MAX_BUDGET}`);
    }
    const checkpoint = input.checkpoint ?? entry.checkpoints?.[0] ?? null;
    if (entry.checkpoints?.length && !entry.checkpoints.includes(checkpoint)) throw Error('Unknown checkpoint');
    return {engine: input.engine, checkpoint, preset, budget: copy(budget), auto: input.auto ?? false};
  }
  /** Evaluations are keyed by the engine, its checkpoint, its build `version` and the checkpoint's weights (`models`). */
  engineKey(spec) {
    const entry = this.entries.get(spec.engine);
    return [spec.engine, spec.checkpoint, entry?.version || '', entry?.models?.[spec.checkpoint ?? ''] ?? ''].join('|');
  }
  cacheKey(history, spec) { return `${this.engineKey(spec)}|${JSON.stringify(spec.budget)}|${position(history)}`; }
  lookup(history, spec = this.analysis, exact = false) { return this.lookupAt(position(history), spec, exact); }
  /** The evaluation of the position whose `position()` text is `at`: by `spec` at exactly its budget when `exact`,
   * else the deepest by its engine. */
  lookupAt(at, spec = this.analysis, exact = false) {
    if (!spec) return null;
    if (exact) return this.cache.get(`${this.engineKey(spec)}|${JSON.stringify(spec.budget)}|${at}`) || null;
    return (this.index.get(`${this.engineKey(spec)}|${at}`) || [])
      .sort((a, b) => Boolean(b.proof) - Boolean(a.proof) || b.simulations - a.simulations || b.solver_nodes - a.solver_nodes || (b.budget?.ms || 0) - (a.budget?.ms || 0) || (b.budget?.nodes || 0) - (a.budget?.nodes || 0))[0] || null;
  }
  /** The evaluations shown per ply and the review of the game, recomputed only after the game or the saved
   * evaluations change. */
  /** Indexes into the game's proof table (`proofs`, proof.mjs Proofs) every saved evaluation of a position of the game
   * that holds a proof, and the game's own records. */
  extendProofs() {
    for (const record of this.records) this.proofs.add(stones(record.position), record, `${record.id}|${record.saved_at}`);
    for (let ply = 0; ply <= this.history.length; ply++) {
      for (const id of this.provenRecords.get(position(this.history.slice(0, ply))) || []) {
        const record = this.cache.get(id);
        if (record) this.proofs.add(this.history.slice(0, ply), record, `${record.id}|${record.saved_at}`);
      }
    }
  }
  /** `record` (or null) at `history`, `played` the game's next stone, with what the game's proof table proves there
   * (proof.mjs proven). */
  withProofs(history, record, played = null) {
    return proven(this.proofs, history, record, this.native.game(history).remaining, played);
  }
  study(winner) {
    const sig = `${this.revision}|${this.evaluationsVersion}|${this.records.length}`;
    if (this.studied?.sig === sig) return this.studied;
    const positions = [''], evaluations = {}, engine = this.analysis && this.engineKey(this.analysis), spec = this.reviewSpec();
    for (const [q, r] of this.history) positions.push(positions.length > 1 ? `${positions.at(-1)};${q},${r}` : `${q},${r}`);
    this.extendProofs();
    positions.forEach((at, ply) => {
      const found = this.lookupAt(at) || this.records.findLast(r => r.position === at && (!engine || r.engine_key === engine));
      const record = this.withProofs(this.history.slice(0, ply), found || null, this.history[ply] ?? null);
      if (record) evaluations[ply] = record;
    });
    const turns = review(this.history, (h, ply) => this.withProofs(h, this.lookupAt(ply === undefined ? position(h) : positions[ply], spec, true)), winner);
    return this.studied = {sig, evaluations, review: turns};
  }
  state() {
    const board = this.native.game(this.history), {player, remaining} = board, winner = board.winner >= 0 ? board.winner : this.outcome?.winner ?? -1;
    const {evaluations, review: turns} = this.study(winner);
    return {instance: `browser:${this.id}`, revision: this.revision, history: copy(this.history), player, remaining, winner,
      paused: this.paused, seats: copy(this.seats), analysis: copy(this.analysis), engines: [...this.entries.values()], match: this.match,
      clock: this.clockNow(), clock_spec: this.control(), outcome: this.outcome, saved_game: this.saved_game, models_folder: null, notice: this.notice, importing: this.importing, storage: {persistent: !!this.storage.db, error: this.storageError},
      book: {available: !!this.bookData, ...this.book, count: this.bookData?.nodes.length, on_policy: this.bookData?.pool('wide').length, refreshed_by: this.bookData?.data.refreshed_by},
      evaluations, stale: Object.keys(evaluations).map(Number).filter(ply => this.stale(evaluations[ply], ply)), review: turns,
      review_preset: this.analysis?.preset ?? null,
      jobs: this.jobs.filter(j => !j.controller.signal.aborted).map(({id, kind, status, done, total, error, history, side, live, stage}) => ({id, kind, status, done, total, error, ply: history.length, side, live, stage}))};
  }
  static handles(path) { path = path.replace(/^\/study/, ''); return OfflineSession.handles(path) || ['/storage', '/openings', '/clock'].some(p => path === p || path.startsWith(p + '/')); }
  answer(path, body = {}) {
    try { this.apply(path.replace(/^\/study/, ''), body); return [200, this.state()]; }
    catch (error) { return [400, {error: error.message}]; }
  }
  changed() { this.revision++; this.runClock(); this.deepen(); this.onchange(this.state()); this.persist(); }
  /** The review's engine, checkpoint and strength: the analysis slot's, so every verdict compares one budget. */
  reviewSpec() { return this.analysis && this.entries.has(this.analysis.engine) ? this.spec({...this.analysis, custom: this.analysis.budget}) : null; }
  deepening() {
    return this.analysis?.auto && !this.paused && this.native.game(this.history).winner < 0 && !this.match?.active && this.seats.some(s => this.adapters.has(s.engine));
  }
  /** While Auto is on and an engine seat plays, evaluates the current position at each preset in turn, fastest first,
   * up to strong unless the analysis engine runs on WebGPU. */
  deepen() {
    const active = this.deepening(), current = position(this.history);
    this.cancelJobs(j => j.tier && (!active || position(j.history) !== current));
    if (!active || this.jobs.some(j => j.tier && j.status !== 'failed') || this.lookup(this.history)?.proof) return;
    const entry = this.entries.get(this.analysis.engine);
    if (!entry || !this.adapters.has(entry.id)) return;
    const tiers = Object.keys(entry.presets), last = entry.device === 'GPU' ? tiers.length : tiers.indexOf('strong') + 1;
    for (const tier of tiers.slice(0, last || tiers.length)) {
      const spec = this.spec({...this.analysis, preset: tier});
      if (!this.lookup(this.history, spec, true) && !this.jobs.some(j => j.tier === tier && j.key === `analyse|${this.cacheKey(this.history, spec)}` && j.status === 'failed')) {
        this.enqueue('analyse', this.history, spec, {tier, line: this.analysisLine}); return;
      }
    }
  }
  snapshot() {
    return {id: this.id, history: copy(this.history), seats: copy(this.seats), analysis: copy(this.analysis), paused: this.paused,
      book: copy(this.book), match: copy(this.match), saved_game: copy(this.saved_game), clock: this.clockNow(), timeControl: copy(this.timeControl),
      clockTurns: copy(this.clockTurns), clockPartial: this.clockPartial, outcome: copy(this.outcome), taken: Date.now(), gameId: this.gameId, gameCreated: this.gameCreated, records: copy(this.records)};
  }
  storageConflict() {
    this.conflicted = true; this.paused = true; this.freezeClock(); this.cancelJobs();
    if (this.match) this.match.active = false;
    this.gameSignature = null; this.storageError = 'This game changed in another tab. Reload to continue.'; this.revision++; this.onchange(this.state());
  }
  persist(games = []) {
    if (this.importing || this.initializing || this.conflicted) return this.saving;
    this.dirty = true;
    if (this.match && !this.match.pending_game && this.match.completed < this.match.games) {
      this.match.position = {game: this.match.current, history: copy(this.history), records: copy(this.records), clock: this.clockNow(), clockTurns: copy(this.clockTurns),
        clockPartial: this.clockPartial + (this.clock?.started != null ? Date.now() - this.clock.started : 0), opening: copy(this.book.opening)};
    }
    const snapshot = this.snapshot(), freeplay = this.freeplay();
    this.saving = this.saving.then(async () => {
      if (this.conflicted) return;
      snapshot._write_token = uid();
      if (!await this.storage.saveSession(snapshot, freeplay, this.storageToken, games)) this.storageConflict();
      else this.storageToken = snapshot._write_token;
    }).catch(error => { this.gameSignature = null; this.storageError = `Could not save in this browser: ${error.message}`; this.onchange(this.state()); });
    return this.saving;
  }
  /** Loads the saved session. A budget game resumes as it was left unless `paused` asks otherwise; a clocked game or a
   * match always waits for Resume so no side's time runs unattended. */
  async restore({paused = false} = {}) {
    const [saved, coverage, evaluations] = await Promise.all([this.storage.get('sessions', this.id), this.storage.get('coverage', 'book'), this.storage.all('evaluations')]);
    this.cache.clear(); this.index.clear(); this.provenRecords.clear(); this.proofs = new Proofs();
    for (const r of evaluations) this.indexRecord(r);
    this.coverage = coverage?.counts || {};
    this.storageToken = saved?._write_token ?? null; this.conflicted = false; this.dirty = false; this.renewLines();
    if (saved) {
      const {taken, ...fields} = saved;
      this.native.game(saved.history); Object.assign(this, {...fields, paused: paused || Boolean(fields.paused) || Boolean(fields.clock)});
      if (this.match) { this.match.active = false; this.paused = true; }
      if (this.clock?.started != null && taken) {
        // The side was thinking when the page went away: the time since the snapshot counts against it.
        const field = this.clock.side ? 'circle_ms' : 'cross_ms', now = Date.now();
        this.clock[field] = Math.max(0, this.clock[field] - (now - taken)); this.clockPartial = (this.clockPartial || 0) + now - this.clock.started;
      }
      if (this.clock) { delete this.clock.started; delete this.clock.side; delete this.clock.running; }
      const game = this.gameId && await this.storage.get('games', this.gameId);
      if (game) { const {saved_at, ...content} = game; this.gameSignature = JSON.stringify(content); }
    }
  }
  cancelJobs(predicate = () => true) {
    for (const job of this.jobs) if (predicate(job)) job.controller.abort();
    this.jobs = this.jobs.filter(j => !j.controller.signal.aborted);
  }
  editable() { if (this.conflicted) throw Error(this.storageError); if (this.match?.active) throw Error('Stop the match before changing its players or position'); }
  load(history, paused = false, opening = null) {
    this.native.game(history); this.cancelJobs(); this.history = copy(history); this.records = []; this.proofs = new Proofs(); this.paused = paused; this.saved_game = null; this.renewLines();
    this.gameId = uid(); this.gameCreated = new Date().toISOString(); this.gameSignature = null;
    this.book.opening = copy(opening); this.freshClock();
  }
  /** Gives `sides` (both when none) a new line, the key of the game tree a Bubble seat searches (worker.mjs). */
  renewLines(...sides) {
    for (const side of sides.length ? sides : [0, 1]) this.lines[side] = uid();
    if (!sides.length) { this.analysisLine = uid(); this.graph = {generation: null, plies: []}; }
  }
  forkGame() {
    if (this.saved_game || !this.gameId || this.match) {
      this.saved_game = null; this.match = null; this.freshClock(); this.gameId = uid(); this.gameCreated = new Date().toISOString(); this.gameSignature = null;
    }
  }
  freeplay() {
    if (this.match || this.saved_game || !this.gameId) return null;
    const board = this.native.game(this.history).winner, winner = board >= 0 ? board : this.outcome?.winner ?? -1, players = this.seats.map(s => ({...s, name: this.entries.get(s.engine)?.name || 'Human'}));
    const game = {id: this.gameId, format: 'bubble-replay', version: 1, history: copy(this.history), players, winner: winner < 0 ? null : winner,
      reason: this.outcome ? this.outcome.reason : winner >= 0 ? 'six' : 'saved', records: copy(this.records),
      ...(this.timeControl.mode === 'fixed' ? {} : {clock: copy(this.timeControl), turns: copy(this.clockTurns)}), evaluations: this.state().evaluations, opening: copy(this.book.opening), created_at: this.gameCreated};
    const signature = JSON.stringify(game);
    if (signature === this.gameSignature) return null;
    this.gameSignature = signature; game.saved_at = new Date().toISOString();
    const summary = {id: this.gameId, name: game.created_at.slice(0, 19).replace('T', ' '), games: 1, completed: 1,
      players: players.map(p => p.name), player_specs: players, created_at: game.created_at, wins: winner < 0 ? [0, 0] : [winner === 0 ? 1 : 0, winner === 1 ? 1 : 0], capped: 0,
      results: [{game: 1, winner: game.winner, reason: game.reason, placements: this.history.length, id: game.id}], single: true, kind: 'freeplay'};
    return {game, summary};
  }
  async saveGame() { await this.persist(); }
  apply(path, body) {
    if (this.conflicted && path !== '/state') throw Error(this.storageError);
    if (['/new', '/undo', '/retry', '/seat', '/clock'].includes(path)) this.editable();
    if (path === '/play') {
      const point = [body.q, body.r];
      if (!point.every(Number.isSafeInteger)) throw Error('Coordinates must be integers');
      const before = this.native.game(this.history);
      if (before.winner < 0 && !this.outcome && this.clock?.side === before.player && this.clockNow()[before.player ? 'circle_ms' : 'cross_ms'] <= 0) {
        this.chargeTurn(before.player); this.changed();
      }
      if (before.winner >= 0 || this.outcome) throw Error('The game has finished');
      const after = this.native.game([...this.history, point]); this.forkGame(); this.history.push(point); this.paused = false;
      if (after.winner >= 0 || after.player !== before.player) this.chargeTurn(before.player);
    } else if (path === '/undo') {
      this.cancelJobs(); this.forkGame(); this.renewLines(); const people = body.people || [0, 1].filter(i => this.seats[i].engine === 'human'); this.history.pop();
      while (people.length && this.history.length && !(people.includes(playerAt(this.history.length)) && this.history.length % 2)) this.history.pop();
      this.saved_game = null; this.freshClock();
    } else if (path === '/new') {
      this.match = null; this.newGame(body.people);
    } else if (path === '/retry') {
      if (!Number.isInteger(body.ply) || body.ply < 0 || body.ply > this.history.length) throw Error('Invalid retry position');
      this.load(this.history.slice(0, body.ply), false, body.ply >= this.book.opening?.ply ? this.book.opening : null); this.match = null;
    } else if (path === '/seat') {
      if (![0, 1].includes(body.side)) throw Error('Invalid seat');
      const seat = this.spec({...this.seats[body.side], ...body, budget: body.preset === 'custom' ? body.custom : undefined});
      if (this.timeControl.mode !== 'fixed') this.clockable(seat);
      this.cancelJobs(j => j.kind === 'move' && j.side === body.side); this.renewLines(body.side); this.seats[body.side] = seat;
      if (this.clock?.side === body.side) this.freezeClock();
    } else if (path === '/clock') {
      if (this.saved_game) throw Error('A saved game has no clock');
      const control = clockSpec(body);
      if (control.mode !== 'fixed') this.seats.forEach(seat => this.clockable(seat));
      this.cancelJobs(j => j.kind === 'move'); this.match = null; this.timeControl = control; this.freshClock();
    } else if (path === '/analysis') {
      this.cancelJobs(j => j.kind !== 'move'); this.analysis = this.spec({...this.analysis, ...body, budget: body.preset === 'custom' ? body.custom : undefined});
    } else if (path === '/analyse') {
      const ply = body.ply ?? this.history.length;
      if (!Number.isInteger(ply) || ply < 0 || ply > this.history.length) throw Error('Invalid analysis position');
      const history = this.history.slice(0, ply), saved = this.analysis && this.lookup(history);
      // A position viewed again whose graph another analysis has searched since: search the graph there again.
      if (!body.force && this.stale(saved, ply) && !saved.proof) this.enqueue('analyse', history, {...this.analysis, budget: copy(saved.budget)}, {force: true, refresh: saved, line: this.analysisLine});
      else if (body.force || ply !== this.history.length || !this.deepening()) this.enqueue('analyse', history, this.analysis, {force: !!body.force, line: this.analysisLine});
    } else if (path === '/review') {
      if (!this.analysis || !this.adapters.has(this.analysis.engine)) throw Error('Choose an analysis engine');
      // From the last position backwards: what a later position proves is known when an earlier one is searched.
      const history = copy(this.history), plies = Array.from({length: history.length + (this.native.game(history).winner < 0)}, (_, i) => i).reverse();
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
      if (this.book.enabled && !this.history.length && !this.match?.active) this.newGame(body.people);
    } else if (!['/state', '/rescan', '/models'].includes(path)) throw Error('Unknown Play request');
    if (path !== '/state') { this.changed(); queueMicrotask(() => this.pump()); }
  }
  /** Starts a new game; with the book on, from one of its openings in a random orientation: against one person
   * (`people`, the sides people play, else the human seats) the least played line on that side, otherwise uniformly. */
  newGame(people = [0, 1].filter(i => this.seats[i].engine === 'human')) {
    let history = [], opening = null;
    if (this.book.enabled && this.bookData) {
      const side = people.length === 1 ? people[0] : null, pool = this.bookData.pool(this.book.mode);
      const n = side === null ? this.bookData.pick(this.book.mode, {}, -1, Math.random, pool[Math.floor(Math.random() * pool.length)])
        : this.bookData.pick(this.book.mode, this.coverage, side);
      history = n.moves; opening = {mode: this.book.mode, key: n.key, symmetry: n.symmetry, ply: history.length};
      if (side !== null) {
        const k = `${side}:${n.key}`; this.coverage[k] = (this.coverage[k] || 0) + 1;
        this.storage.put('coverage', {id: 'book', counts: copy(this.coverage)}).catch(e => { this.storageError = e.message; });
      }
    }
    this.load(history, false, opening);
  }
  enqueue(kind, history, spec, fields = {}) {
    if (this.importing) return;
    if (!spec || !this.adapters.has(spec.engine)) throw Error('Choose a browser engine');
    const key = `${kind}|${this.cacheKey(history, spec)}`;
    if (fields.force) this.cancelJobs(j => j.key === key);
    if (this.jobs.some(j => j.key === key)) return;
    if (kind === 'analyse' && !fields.force && this.lookup(history, spec, true)) return;
    if (kind === 'analyse' && !fields.refresh) this.cancelJobs(j => j.kind === 'analyse' && j.status !== 'failed' && (fields.tier ? j.tier : true));
    this.jobs.push({id: ++this.nextJob, kind, history: copy(history), spec: copy(spec), key, controller: new AbortController(), status: 'queued', done: 0, total: 1, ...fields});
  }
  /** Saves the evaluation `result` of `history` by `spec`. One whose solver could not run (`solver_error`) is kept for
   * this visit only and raises the session's `notice`, so a later visit evaluates the position again with proofs. A
   * `kept` one (a seat's move on its game tree, or one cut short by a clock) is shown but never reused as a fresh
   * evaluation. A saved evaluation holding a proof is kept over an unproven result of the same id. Resolves to the saved
   * evaluation. */
  async record(history, spec, result, kept = false) {
    const id = this.cacheKey(history, spec) + (kept ? '|kept' : ''), saved = this.cache.get(id);
    const source = saved?.proof && !result.proof ? saved : result, facts = new Map();
    for (const fact of [...(saved?.proofs || []), ...(result.proofs || [])]) {
      const key = proofKey(fact.history), old = facts.get(key);
      if (!old || fact.plies < old.plies || fact.plies === old.plies && fact.pv.length > old.pv.length) facts.set(key, fact);
    }
    const record = source === saved && !result.proofs?.length ? saved : {...source,
      ...(facts.size ? {proofs: [...facts.values()]} : {}), id, position: position(history), engine: spec.engine, engine_key: this.engineKey(spec),
      simulations: spec.budget.simulations ?? result.simulations ?? spec.budget.visits ?? 0, solver_nodes: result.solved === false ? 0 : spec.budget.solver_nodes ?? result.solver_nodes ?? 0, budget: copy(spec.budget), saved_at: new Date().toISOString()};
    if (record !== saved) {
      this.indexRecord(record); this.proofs.add(history, record, `${record.id}|${record.saved_at}`);
      if (result.solver_error) this.notice = `The solver could not run in this browser (${result.solver_error}), so evaluations have no proofs`;
      else await this.storage.put('evaluations', record);
    }
    if (position(this.history.slice(0, history.length)) === position(history)) {
      this.records = this.records.filter(r => r.id !== record.id); this.records.push(record);
    }
    return record;
  }
  /** Notes a search at `ply` on the analysis graph `id` (GameGraph.id, unique to each graph the worker builds) and
   * returns the stamp its evaluation is saved with, [id, searches so far]; a new id starts a new count. A primary
   * analysis counts and records its ply; a refresh re-reads evidence the graph already holds, so it takes the current
   * count without adding to it and never stales another position (python/play.py Session.searched). */
  graphSearched(id, ply, count = true) {
    if (this.graph.generation !== id) this.graph = {generation: id, plies: []};
    if (count) this.graph.plies.push(ply);
    return [id, this.graph.plies.length];
  }
  /** True when `record`, the saved evaluation of the position at `ply`, predates a primary analysis of a deeper position
   * on the graph analysis searched last (python/play.py Session.stale): a stamped record is older than the searches
   * after its stamp, an unstamped one (a review's, or an earlier graph's) than every search. A refresh never stales a
   * record and a proven record is never stale. */
  stale(record, ply) {
    if (!record || record.proof || this.graph.generation == null) return false;
    const since = record.graph?.[0] === this.graph.generation ? record.graph[1] : 0;
    return this.graph.plies.slice(since).some(searched => searched > ply);
  }
  /** Queues a refresh of each position up to REFRESH_PLIES placements before `history` whose shown evaluation by `spec`'s
   * engine (the deepest, a tier included) is stale and holds no proof, nearest first, at that evaluation's budget: the search on the game graph `line` moved the values those positions reach
   * (python/play.py Session.refresh). A refresh searches that graph again with the PV_CHECK share of the simulations
   * and no solver query, keeps the saved threat and replaces the saved evaluation. */
  refresh(history, spec, line) {
    for (let ply = history.length - 1; ply >= Math.max(0, history.length - REFRESH_PLIES); ply--) {
      const saved = this.lookup(history.slice(0, ply), spec);
      if (this.stale(saved, ply) && !saved.proof) this.enqueue('analyse', history.slice(0, ply), {...spec, budget: copy(saved.budget)}, {force: true, refresh: saved, line});
    }
  }
  indexRecord(record) {
    this.evaluationsVersion++;
    this.cache.set(record.id, record);
    if (record.proof || record.proofs?.length) {
      if (!this.provenRecords.has(record.position)) this.provenRecords.set(record.position, new Set());
      this.provenRecords.get(record.position).add(record.id);
    }
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
    clock.running = clock.started != null ? ['x', 'o'][clock.side] : null;
    return clock;
  }
  freezeClock() {
    clearTimeout(this.flag);
    if (this.clock?.started != null) this.clockPartial += Date.now() - this.clock.started;
    this.clock = this.clockNow(); if (this.clock) { delete this.clock.started; delete this.clock.side; delete this.clock.running; }
  }
  /** The time control in force: the match's, a saved game's recorded one, else the freeplay one. */
  control() { return this.match ? this.match.clock : this.saved_game ? this.saved_game.clock || {mode: 'fixed'} : this.timeControl; }
  /** Refuses `seat` under a clock when its engine plays a fixed budget. */
  clockable(seat) {
    const entry = this.entries.get(seat.engine);
    if (seat.engine !== 'human' && !entry?.clocks) throw Error(`${entry?.name || seat.engine} plays a fixed budget; it cannot keep a clock`);
  }
  /** Full balances for the game under `control()` (none for a fixed budget), no outcome and an empty turn log. */
  freshClock() {
    clearTimeout(this.flag);
    const c = this.control(), base = c.mode === 'move' ? c.ms : c.base_ms ?? c.initial_ms;
    this.clock = c.mode === 'fixed' ? null : {cross_ms: base, circle_ms: base, increment_ms: c.mode === 'game' ? c.increment_ms : 0};
    this.clockTurns = []; this.outcome = null; this.clockPartial = 0;
  }
  /** Starts the clock of the side to move while the game runs on one. An engine side's clock starts only from its move
   * job, once `ready` (its engine and the checkpoint's network loaded for that job). Arms the loss on time. */
  runClock(ready = false) {
    const c = this.clock;
    if (!c || c.started != null || this.paused || this.outcome || this.saved_game || this.conflicted || this.importing) return;
    if (this.match && (!this.match.active || this.match.pending_game)) return;
    const {winner, player} = this.native.game(this.history), seat = this.seats[player];
    if (winner >= 0 || seat.engine !== 'human' && !ready) return;
    Object.assign(c, {started: Date.now(), side: player});
    this.armFlag();
  }
  /** Arms the loss on time for the running clock. */
  armFlag() {
    clearTimeout(this.flag);
    const c = this.clock;
    if (c?.started != null) this.flag = setTimeout(() => this.checkTime(), Math.min(MAX_TIMER, Math.max(0, c[c.side ? 'circle_ms' : 'cross_ms'] - (Date.now() - c.started)) + 20));
  }
  /** Charges `side`'s completed turn at time `at`, with what it spent before a pause (`clockPartial`): finished late it loses on time; otherwise it gains its increment, or
   * a whole turn again on a per-turn clock. Logs the balances in `clockTurns`; true when the turn stands. */
  chargeTurn(side, at = Date.now()) {
    const c = this.clock;
    if (!c || c.started == null || c.side !== side) return true;
    clearTimeout(this.flag);
    const field = side ? 'circle_ms' : 'cross_ms', left = c[field] - (at - c.started), spent = at - c.started + this.clockPartial, control = this.control();
    this.clockPartial = 0;
    delete c.started; delete c.side;
    if (left <= 0) { c[field] = 0; this.outcome = {winner: 1 - side, reason: 'time'}; }
    else c[field] = control.mode === 'move' ? control.ms : left + (c.increment_ms || 0);
    this.clockTurns.push({ply: this.history.length, side, spent_ms: spent, cross_ms: c.cross_ms, circle_ms: c.circle_ms});
    return left > 0;
  }
  /** The side to move whose clock ran out loses on time; its engine's search is cancelled. */
  async checkTime() {
    const c = this.clock;
    if (!c || c.started == null) return;
    const side = c.side, left = this.clockNow()[side ? 'circle_ms' : 'cross_ms'];
    if (left > 0) { this.flag = setTimeout(() => this.checkTime(), Math.min(MAX_TIMER, left + 20)); return; }
    this.cancelJobs(j => j.kind === 'move');
    this.chargeTurn(side);
    if (this.match?.active) await this.finishMatch(1 - side, 'timeout');
    this.changed();
  }
  /** Runs the next job: an engine move first, then analysis, review and deepening. A queued move interrupts a running
   * job of another kind, which goes back to the queue (a review keeps its finished positions). */
  async pump() {
    if (this.importing || this.conflicted) return;
    const state = this.native.game(this.history), seat = this.seats[state.player];
    if (!this.paused && state.winner < 0 && this.adapters.has(seat.engine) && !this.jobs.some(j => j.kind === 'move')) this.enqueue('move', this.history, seat, {side: state.player, line: this.lines[state.player]});
    if (this.running) {
      if (this.running.kind !== 'move' && this.jobs.some(j => j.kind === 'move' && j.status === 'queued')) this.running.attempt.abort();
      return;
    }
    const job = this.jobs.find(j => j.status === 'queued' && j.kind === 'move') || this.jobs.find(j => j.status === 'queued' && j.kind === 'analyse' && !j.tier)
      || this.jobs.find(j => j.status === 'queued' && j.kind === 'review') || this.jobs.find(j => j.status === 'queued' && j.tier);
    if (!job) return;
    let settled;
    this.idle = new Promise(resolve => { settled = resolve; });
    this.running = job; job.status = 'running'; job.attempt = new AbortController(); this.changed();
    const signal = AbortSignal.any([job.controller.signal, job.attempt.signal]);
    let timer, interrupted = false;
    const match = job.kind === 'move' && this.match?.active ? this.match : null;
    const history = job.kind === 'review' ? job.history.slice(0, job.plies[job.cursor]) : job.history;
    try {
      const adapter = this.adapters.get(job.spec.engine);
      await abortable(adapter.ready?.((f, stage) => { job.done = f * .1; job.stage = stageText(stage); this.onchange(this.state()); }, job.spec.checkpoint), signal);
      if (signal.aborted) throw new DOMException('Cancelled', 'AbortError');
      job.stage = stageText(null);
      if (job.kind === 'move') this.runClock(true);
      let timeout = false, limit = null, ms = null;
      if (job.kind === 'move' && this.clock) {
        const clock = this.clockNow();
        limit = clock[job.side ? 'circle_ms' : 'cross_ms']; ms = turnTime(this.control(), clock, job.side);
        const field = job.side ? 'circle_ms' : 'cross_ms', expire = () => {
          const left = this.clockNow()?.[field] ?? 0;
          if (left > 0) { timer = setTimeout(expire, Math.min(MAX_TIMER, left)); return; }
          timeout = true; job.attempt.abort();
        };
        timer = setTimeout(expire, Math.min(MAX_TIMER, Math.max(1, limit)));
      }
      let result = job.kind !== 'move' && !job.force ? this.lookup(history, job.spec, true) : null;
      const budget = job.refresh ? {...job.spec.budget, simulations: Math.max(1, Math.round(PV_CHECK * job.spec.budget.simulations)), solver_nodes: 0}
        : job.spec.budget;
      try {
        result ||= await adapter.turn(copy(history), copy(budget), {signal, checkpoint: job.spec.checkpoint, preset: job.spec.preset, ms, line: job.line,
          known: job.kind === 'move' ? null : (this.extendProofs(), this.proofs.list()),
          progress: (f, live, stage) => { job.done = job.kind === 'review' ? job.cursor + f : f; job.stage = stageText(stage); if (live && job.kind !== 'review') job.live = live; this.onchange(this.state()); }});
      } catch (e) { if (!timeout) throw e; }
      clearTimeout(timer);
      const at = Date.now(), elapsed = this.clock?.started != null ? at - this.clock.started : null;
      if (job.kind === 'move') {
        if (match && (this.match !== match || !match.active || this.paused || !this.jobs.includes(job))) return;
        if (timeout || !this.outcome && this.clock && this.clockNow()[job.side ? 'circle_ms' : 'cross_ms'] <= 0) {
          this.chargeTurn(job.side, at);
          if (match) await this.finishMatch(1 - job.side, 'timeout');
          return;
        }
      }
      if (job.controller.signal.aborted) throw new DOMException('Cancelled', 'AbortError');
      // The move came back at `at`; saving it must not run its clock out.
      if (job.kind === 'move') clearTimeout(this.flag);
      if (job.refresh) result = {...result, threat: job.refresh.threat ?? []};
      const {graph_id: graph, ...answer} = result;
      result = answer;
      if (job.kind === 'analyse' && job.line != null && graph) { result = {...result, graph: this.graphSearched(graph, history.length, !job.refresh)}; job.counted = true; }
      await this.record(history, job.spec, result, job.kind === 'move' && (ms != null || this.entries.get(job.spec.engine)?.kind === 'bubble'));
      if (job.kind === 'analyse' && job.line != null && this.entries.get(job.spec.engine)?.kind === 'bubble') {
        if (!job.refresh) this.refresh(history, job.spec, job.line);
        else if (graph && moved(result, job.refresh)) {
          // The refresh changed this position's result: new evidence for the positions before it, and another round here.
          this.graphSearched(graph, history.length); this.refresh(history, job.spec, job.line);
          const rounds = (job.rounds || 0) + 1;
          if (rounds < REFRESH_ROUNDS) this.enqueue('analyse', history, job.spec, {force: true, refresh: this.lookup(history, job.spec), line: job.line, rounds});
        }
      }
      if (job.kind === 'move') {
        if (job.controller.signal.aborted || position(this.history) !== position(history) || this.paused) { this.armFlag(); return; }
        if (!this.match?.active) this.forkGame();
        this.history.push(...copy(this.checkedTurn(history, result.moves)));
        this.chargeTurn(job.side, at);
        if (this.match?.active) {
          this.match.timings.push({ply: history.length, side: job.side, engine: job.spec.engine, elapsed_ms: elapsed});
          const winner = this.native.game(this.history).winner;
          if (winner >= 0) await this.finishMatch(winner, 'six');
          else if (this.match.max_placements && this.history.length >= this.match.max_placements) await this.finishMatch(null, 'capped');
        }
      } else if (job.kind === 'review') { job.cursor++; job.done = job.cursor; }
    } catch (error) {
      // A cancelled or failed primary analysis that touched its game graph (the worker names the graph) changed it.
      if (job.kind === 'analyse' && job.line != null && error.graph && !job.counted) this.graphSearched(error.graph, history.length, !job.refresh);
      interrupted = error.name === 'AbortError' && !job.controller.signal.aborted;
      if (error.name !== 'AbortError') { job.status = 'failed'; job.error = error.message; if (job.kind === 'move') { this.freezeClock(); this.paused = true; } if (this.match) this.match.error = error.message; }
    } finally {
      clearTimeout(timer); this.running = null;
      if (interrupted) job.status = 'queued';
      else if (job.status !== 'failed' && (job.kind !== 'review' || job.cursor >= job.plies.length || job.controller.signal.aborted)) this.jobs = this.jobs.filter(j => j !== job);
      else if (job.status !== 'failed') job.status = 'queued';
      this.changed(); await this.saving;
      settled();
      queueMicrotask(() => this.pump());
    }
  }
  async startMatch(body) {
    if (this.conflicted) throw Error(this.storageError);
    if (this.match?.active) throw Error('Stop the current match first');
    if (!Array.isArray(body.players) || body.players.length !== 2 || !Number.isInteger(body.games) || body.games < 2 || body.games > 10000 || body.games % 2) throw Error('Choose two engines and an even number of games');
    const max_placements = body.max_placements ?? 512;
    if (!Number.isSafeInteger(max_placements) || max_placements < 0 || max_placements === 1) throw Error('Stone limit must be at least 2, or 0 for uncapped');
    const players = body.players.map(p => { const s = this.spec(p), entry = this.entries.get(s.engine); return {...s, name: entry.name, version: this.engineKey(s)}; });
    const seed = body.seed ?? Math.floor(Math.random() * 4294967296), mode = body.opening_range || 'origin';
    const openings = mode === 'origin' ? (body.openings || [[[0, 0]]]).map(moves => ({moves})) : this.bookData?.select(mode, body.unique_openings || body.games / 2, seed);
    if (!openings?.length) throw Error('No opening book is available');
    openings.forEach(n => { if (this.native.game(n.moves).winner >= 0 || max_placements && n.moves.length >= max_placements) throw Error('Match openings must be unfinished and shorter than the stone limit'); });
    const clock = clockSpec(body.clock);
    if (clock.mode !== 'fixed') players.forEach(p => this.clockable(p));
    await this.saveGame(); this.cancelJobs();
    if (this.analysis) this.analysis.auto = false;
    this.match = {id: uid(), name: new Date().toISOString().slice(0, 19).replace('T', ' '), active: true, players, games: body.games,
      completed: 0, current: 1, wins: [0, 0], capped: 0, results: [], openings, opening_range: mode, seed, clock, timings: [], elo: null, max_placements};
    this.beginMatchGame(); this.changed(); await this.saving; this.pump();
  }
  beginMatchGame() {
    const match = this.match, number = match.completed + 1;
    const opening = match.openings[Math.floor((number - 1) / 2) % match.openings.length];
    this.load(opening.moves, false, opening.key ? {mode: match.opening_range, key: opening.key, symmetry: opening.symmetry, ply: opening.moves.length} : null);
    this.seats = copy(number % 2 ? match.players : [...match.players].reverse()); match.current = number; match.timings = []; match.pending_game = false;
    this.freshClock();
  }
  resumeMatch() {
    if (!this.match || this.match.completed >= this.match.games) throw Error('No unfinished match');
    for (const p of this.match.players) if (p.version && this.engineKey(p) !== p.version) throw Error('This match used a different engine version. Start a new match.');
    if (this.match.pending_game) this.beginMatchGame();
    this.seats = copy(this.match.current % 2 ? this.match.players : [...this.match.players].reverse());
    this.jobs = this.jobs.filter(j => j.status !== 'failed'); this.match.error = null; this.match.active = true; this.paused = false;
  }
  async finishMatch(winner, reason) {
    const m = this.match, game = m.current, aWinner = winner == null ? null : game % 2 ? winner : 1 - winner, id = `${m.id}:${game}`;
    const record = {id, format: 'bubble-replay', version: 1, game, history: copy(this.history), players: copy(this.seats), winner, reason,
      evaluations: this.state().evaluations, records: copy(this.records), timings: copy(m.timings), opening: copy(this.book.opening), saved_at: new Date().toISOString(),
      ...(m.clock.mode === 'fixed' ? {} : {clock: copy(m.clock), turns: copy(this.clockTurns)})};
    m.results.push({id, game, winner: aWinner, reason, placements: this.history.length, opening: Math.floor((game - 1) / 2)});
    if (winner == null) m.capped++; else m.wins[aWinner]++;
    m.completed++; m.elo = pairElo(m.results);
    if (m.completed >= m.games) { m.active = false; if (this.match === m) { this.paused = true; this.freezeClock(); } }
    else { m.pending_game = true; m.current = m.completed + 1; }
    await this.persist([record]);
    if (!this.conflicted && !this.paused && this.match === m && m.active && m.completed < m.games) this.beginMatchGame();
  }
  async catalogue() {
    await this.saving;
    return Promise.all((await this.storage.all('matches')).sort((a, b) => b.name.localeCompare(a.name)).map(async m => {
      const game = m.single && !m.player_specs ? await this.storage.get('games', m.id) : null;
      return {...m, player_specs: m.player_specs || game?.players || m.players, created_at: m.created_at || game?.created_at,
        players: m.players.map(p => typeof p === 'string' ? p : p.name)};
    }));
  }
  async savedReplay(id, game) {
    await this.saving;
    const match = await this.storage.get('matches', id), result = match?.results.find(r => r.game === +game), record = result && await this.storage.get('games', result.id);
    if (!record) throw Error('No such saved game in this browser');
    return record;
  }
  async openGame(id, number) {
    const game = await this.savedReplay(id, number);
    this.load(game.history, true, game.opening || null); this.seats = [human(), human()];
    this.saved_game = {batch: id, game: +number, players: game.players.map(p => p.name || 'Human'), winner: game.winner, reason: game.reason, clock: game.clock || null};
    this.clockTurns = copy(game.turns || []);
    this.outcome = ['time', 'timeout'].includes(game.reason) && game.winner != null ? {winner: game.winner, reason: game.reason} : null;
    const last = this.clockTurns.at(-1), base = game.clock && (game.clock.mode === 'move' ? game.clock.ms : game.clock.base_ms);
    this.clock = game.clock ? {cross_ms: last?.cross_ms ?? base, circle_ms: last?.circle_ms ?? base, increment_ms: game.clock.increment_ms || 0} : null;
    this.records = game.records || Object.values(game.evaluations || {});
    if (this.analysis) this.analysis.auto = false;
    this.changed();
  }
  async request(input, body = {}, method = 'GET') {
    const url = new URL(input, 'https://play.invalid'), path = url.pathname.replace(/^\/study/, ''), q = url.searchParams;
    const interrupt = path === '/cancel' || path === '/pause' && body.paused || path === '/match' && ['pause', 'stop'].includes(body.action);
    try {
      if (method !== 'GET' && !interrupt && !['/state', '/export', '/replay', '/evaluations', '/storage/backup'].includes(path)) {
        await this.saving;
        const known = this.storageToken, stored = (await this.storage.get('sessions', this.id))?._write_token ?? null;
        if (stored !== known && stored !== this.storageToken) this.storageConflict();
        if (this.conflicted) throw Error(this.storageError);
      }
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
      else if (path === '/replay') data = {format: 'bubble-replay', version: 1, history: this.history, players: this.seats.map(p => ({...p, name: this.entries.get(p.engine)?.name || 'Human'})), evaluations: this.state().evaluations, records: this.records, opening: this.book.opening};
      else if (path === '/evaluations') { data = this.records.map(r => JSON.stringify(r)).join('\n'); type = 'application/x-ndjson'; }
      else if (path === '/storage/backup') { await this.saving; data = await this.storage.backup(); }
      else if (path === '/storage/save') { await this.saveGame(); data = this.state(); }
      else if (path === '/openings') data = {refreshed_by: this.bookData?.data.refreshed_by, nodes: this.bookData.select(q.get('range') || 'wide', +(q.get('count') || 8), +(q.get('seed') || 0))};
      else if (path === '/import') {
        this.editable(); const json = body.text.trim().startsWith('{') ? JSON.parse(body.text) : null;
        if (json?.format === 'hexo-browser-save') {
          this.importing = true; this.paused = true; this.cancelJobs();
          try {
            await this.idle; await this.saving; await this.storage.restore(json, this.native); await this.restore({paused: true});
            for (const spec of [...this.seats, this.analysis].filter(Boolean)) if (this.entries.has(spec.engine)) Object.assign(spec, this.spec(spec));
          } finally { this.importing = false; }
          this.changed(); data = this.state();
        }
        else {
          const history = await readGame(body.text, this.native); await this.saveGame(); this.match = null; this.load(history, true, json?.opening || null);
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
            for (const p of saved.players) if (p.version && this.engineKey(p) !== p.version) throw Error('This match used a different engine version. Start a new match.');
            this.match = saved;
            this.match.players = this.match.players.map(p => ({...p, ...this.spec(p)}));
            if (!saved.pending_game && saved.position?.game === saved.completed + 1) {
              this.load(saved.position.history, true, saved.position.opening);
              this.records = copy(saved.position.records); this.clock = copy(saved.position.clock);
              this.clockTurns = copy(saved.position.clockTurns || []); this.clockPartial = saved.position.clockPartial || 0;
              if (this.clock) { delete this.clock.started; delete this.clock.side; }
              this.seats = copy(saved.current % 2 ? saved.players : [...saved.players].reverse());
            } else this.beginMatchGame();
          }
          this.resumeMatch();
        } else if (body.action === 'pause' || body.action === 'stop') {
          this.freezeClock(); this.cancelJobs(); this.paused = true; if (this.match) this.match.active = false;
        } else throw Error('Unknown match action');
        this.changed(); this.pump(); data = this.state();
      } else {
        if (path === '/new') await this.saveGame();
        const [status, value] = this.answer(path, body);
        if (path === '/play' && this.native.game(this.history).winner >= 0) await this.saveGame();
        if (!interrupt) await this.saving; return [status, value, type];
      }
      if (!interrupt) await this.saving; return [200, data, type];
    } catch (error) { return [400, {error: error.message}, 'application/json']; }
  }
}
