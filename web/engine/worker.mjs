/* Bubble in a Web Worker: network (network.mjs), native search (gumbel.wasm) and the tactical solver (solver-worker.mjs).
 * In: {type: 'load', options} | {type: 'use', id, model}
 *     | {type: 'turn', id, history, model, simulations, solverNodes, solverWorkers, batchSize, qRangeFloor, ms, line, known}
 *     | {type: 'cancel', id} | {type: 'bench', id, batches, sizes, repeats} | {type: 'evaluate', id, histories}.
 * Out: {type: 'progress', id?, fraction, stage?} | {type: 'ready', device} | {type: 'result', id, result} | {type: 'cancelled', id}
 *     | {type: 'error', id?, message, stage?}: stages.mjs's loading stages; a stage in an error is where it stopped.
 */
import createModule from './gumbel.mjs';
import {Native, EvaluationCache, GameGraph, GameGraphs, NativeOwner} from './search.mjs';
import {Network, probe, runtime} from './network.mjs';
import {Stages, errorReport, stall} from './stages.mjs';
import {principalVariation, topRows, Proofs, answered, settled, proofTurns, proofKey, proofEvidence, winningLine, proven} from './proof.mjs';

const VERDICTS = new Set(['no verified strategy', 'quiet defender unsupported', 'defender counterwin',
  'candidate has unproved defender continuation', 'candidate defense expansion budget', 'candidate certificate size limit',
  'free-second coverage work limit', 'no fallback strategy', 'zone proof budget', 'zone certificate byte limit', 'proof zone size limit']);
let native, ort, network, cache, games, device, solver = null, solverCalls = 0;
let frontierWorkers = null;
/** The latest game-tree turn: they run one at a time, so a cancelled turn still awaiting the network settles before
 * another turn advances, searches or evicts a game tree. */
let gameTurn = Promise.resolve();
let modelUse = Promise.resolve();
const held = new Map();
const cancelled = new Set(), solverWaits = new Map(), cancelWaits = new Map();

class Cancelled extends Error {}

/** `promise`, or a Cancelled rejection as soon as job `id` is cancelled; the promise itself runs on. */
function unlessCancelled(id, promise) {
  return new Promise((resolve, reject) => {
    if (cancelled.has(id)) { reject(new Cancelled()); return; }
    cancelWaits.set(id, () => reject(new Cancelled()));
    promise.then(resolve, reject).finally(() => cancelWaits.delete(id));
  });
}

const GRACE_MS = 500;
const unknown = reason => ({status: 'UNKNOWN', native_verified: false, moves: [], nodes_used: 0, reason});

/** One outstanding slice per worker. Ordinary cancellation drains its short
 * query and retains the resident table; only a broken worker is replaced. */
class SolverWorkers {
  constructor(count) { this.count = count; this.entries = Array(count).fill(null); this.next = 0; }
  retire(index, error) {
    const entry = this.entries[index]; if (!entry) return;
    entry.worker.terminate(); this.entries[index] = null;
    if (entry.wait) { clearTimeout(entry.wait.timer); entry.wait.reject(error); entry.wait = null; }
  }
  ask(index, data, ms) {
    const entry = this.entries[index];
    if (!entry || entry.wait) return Promise.reject(new Error('Solver worker is unavailable or busy'));
    return new Promise((resolve, reject) => {
      const id = ++this.next, timer = setTimeout(() => this.retire(index, new Error('solver slice hard deadline')), ms + GRACE_MS);
      entry.wait = {id, timer, resolve, reject}; entry.worker.postMessage({id, ...data});
    });
  }
  prepare(index) {
    let entry = this.entries[index];
    if (!entry) {
      const worker = new Worker(new URL('solver-worker.mjs', import.meta.url), {type: 'module'});
      entry = this.entries[index] = {worker, wait: null, ready: null,
        control: typeof SharedArrayBuffer === 'function' ? new Int32Array(new SharedArrayBuffer(4)) : null};
      worker.onmessage = ({data}) => {
        const wait = entry.wait;
        if (!wait || wait.id !== data.id) return;
        clearTimeout(wait.timer); entry.wait = null; wait.resolve(data);
      };
      worker.onerror = event => this.retire(index, new Error(`${FAILED}: ${event.message || 'error'}`));
      entry.ready = this.ask(index, {prepare: true}, 10000).then(data => {
        if (!data.ready) { const error = new Error(`${FAILED}: ${data.result?.reason || 'preparing'}`); this.retire(index, error); throw error; }
      });
    }
    return entry.ready;
  }
  async query(index, request, cancelled = () => false) {
    await this.prepare(index);
    if (cancelled()) {
      const info = Array(13).fill(0), size = request.history.length;
      info[4] = 1; info[9] = Math.floor((size + 1) / 2) % 2; info[10] = !size || size % 2 === 0 ? 1 : 2;
      info[11] = request.attacker === 'defender' ? 1 : 0; info[12] = 1;
      return {info, moves: []};
    }
    const control = this.entries[index].control;
    if (control) Atomics.store(control, 0, 0);
    const data = await this.ask(index, {request, cancel: control}, request.ms);
    if (!data.answer) throw new Error('Missing typed solver completion');
    return data.answer;
  }
  cancel(index) { const flag = this.entries[index]?.control; if (flag) Atomics.store(flag, 0, 1); }
  get cooperative() { return this.entries.every(entry => entry?.control); }
  close() { for (let i = 0; i < this.count; i++) this.retire(i, new Error('Solver worker closed')); }
}

async function proofWorkers(count) {
  if (!Number.isInteger(count) || count < 1 || count > 16) throw new Error('Invalid solver worker count');
  if (frontierWorkers && frontierWorkers.count !== count) { frontierWorkers.close(); frontierWorkers = null; }
  frontierWorkers ??= new SolverWorkers(count);
  await Promise.all(Array.from({length: count}, (_, i) => frontierWorkers.prepare(i)));
  return frontierWorkers;
}

/**
 * The solver's answer to one query of turn `owner`; queries run one at a time in the solver worker. A query still
 * unanswered GRACE_MS after its `ms` (its own deadline, as IsolatedTactics' hard deadline) answers UNKNOWN and the
 * solver worker is replaced, since certificate reconstruction cannot be interrupted inside it.
 */
function solve(owner, history, options) {
  return new Promise((resolve, reject) => {
    const id = ++solverCalls;
    const timer = setTimeout(() => { settle(id, unknown('deadline')); replace(); }, options.ms + GRACE_MS);
    solverWaits.set(id, {owner, history, options, resolve, reject, timer});
    send(id);
  });
}

function settle(id, result) {
  const wait = solverWaits.get(id);
  if (!wait) return;
  clearTimeout(wait.timer);
  solverWaits.delete(id);
  result instanceof Error ? wait.reject(result) : wait.resolve(result);
}

function send(id) {
  if (!solver) {
    try {
      solver = new Worker(new URL('solver-worker.mjs', import.meta.url), {type: 'module'});
    } catch (error) {   // a browser that forbids nested workers: the query has no answer
      settle(id, unknown(`${FAILED}: ${error.message}`));
      return;
    }
    solver.onmessage = ({data}) => settle(data.id, data.result);
    solver.onerror = event => {
      for (const id of [...solverWaits.keys()]) settle(id, unknown(`${FAILED}: ${event.message || 'error'}`));
      replace();
    };
  }
  const {history, options} = solverWaits.get(id);
  solver.postMessage({id, history, options});
}

/** A new solver worker for the queries still waiting. */
function replace() {
  solver?.terminate();
  solver = null;
  for (const id of solverWaits.keys()) send(id);
}

/** Ends the solver queries of turn `owner` and replaces the solver worker when it was working for that turn. */
function stopSolver(owner) {
  const ids = [...solverWaits].filter(([, wait]) => wait.owner === owner).map(([id]) => id);
  if (!ids.length) return;
  for (const id of ids) settle(id, new Cancelled());
  replace();
}

const verified = r => r.status === 'PROVEN_WIN' && r.native_verified;
const FAILED = 'solver worker failed';
const searched = r => verified(r) || VERDICTS.has(r.reason);

/** The root of the first stone's running search as the analysis panel shows it: {value (the mover's win chance), top}. */
function rootRows(tree, choice) {
  if (!tree) return null;
  const {action, actions, policy, values, completed_q} = tree.result(choice);
  if (!action) return null;
  const top = topRows(actions, policy, completed_q, action);
  const refuted = top.length && top.every(row => row[4] < 0) && tree.m._hxg_exact(tree.ptr) < 0;
  const value = refuted ? tree.m._hxg_value(tree.ptr) : policy.reduce((sum, p, i) => sum + p * values[i], 0);
  return {value: Math.round((value + 1) / 2 * 1e4) / 1e4, top, refuted: refuted ? top.length : 0};
}

/** Indexes verified frontier answers `records` ({request, result}, NativeProofs.records) into the game's proof table and
 * the turn's frontier proofs (`found`, keyed by proofKey). */
function frontierProofs(records, table, found) {
  for (const {request, result} of records) {
    const winner = result.winner,
      plies = result.status === 'PROVEN_WIN' ? result.moves.length + 4 * (result.proof_turns - 1) : 4 * result.proof_turns + 2;
    const pv = principalVariation(native, request.history, result.certificate, {attacker: winner, known: request.known}).pv;
    found.set(proofKey(request.history), {history: request.history, winner, plies, pv});
    table.add(request.history, {proof: {winner, plies, turns: result.proof_turns, ...proofEvidence(result)}, pv});
  }
}

/**
 * The solver preset (python/play.py prove): proof work alone on `history` for up to `ms`, until a verified proof for
 * either side. The tactical solver asks the root for a win of `player` (the side to move) with 32,768 nodes, then four
 * times as many each round, while a native owner's neural search feeds `workers` proof workers (`pool`, null when they
 * could not start) the positions it reaches. Posts progress with live {solver: {elapsed_ms, root_nodes, frontier, busy,
 * workers, proof}} at most twice a second, after a neural batch. Resolves to {mine (a verified root answer, or null), proof ({winner, turns, plies} the native search proved,
 * or null), records (the frontier's verified answers), used (nodes), solver (totals), error}.
 */
async function proveRoot(id, history, player, {ms, workers, facts, stamps, batchSize, pool}) {
  const start = performance.now(), end = start + ms, remaining = native.game(history).remaining;
  const premises = facts.filter(f => f.history.length !== history.length || f.winner !== player).map(({history, winner, plies}) => ({history, winner, plies}));
  const root = {nodes: 0, mine: null, done: false};
  const asking = (async () => {
    for (let nodes = 32768; !root.done && !cancelled.has(id) && performance.now() < end;) {
      const found = await solve(id, history, {attacker: 'mover', nodes, shortest: true, stamps, known: premises,
        ms: Math.max(1, Math.floor(Math.min(end - performance.now(), 60000, Math.max(10000, nodes / 8))))});
      root.nodes += found.nodes_used || 0;
      if (verified(found) && found.moves.length) { root.mine = found; return; }
      if (!searched(found)) continue;   // the solver worker was replaced: ask again
      if ((found.nodes_used || 0) < nodes) return;   // the solver ruled the root out before spending its nodes
      nodes = Math.min(4 * nodes, 10000000);
    }
  })().catch(error => { if (!(error instanceof Cancelled)) throw error; });
  const graph = new GameGraph(native, {seed: 1740, tactics: true, model: network.version, roundBarrier: true, history: history.map(p => [...p])});
  let result = null, shown = 0, frontier = 0, busy = 0;
  try {
    const owner = new NativeOwner(graph, {work: 0, ms});
    const report = () => {
      if (performance.now() < shown) return;
      shown = performance.now() + 500;
      const stats = owner.proofs?.ptr ? owner.proofs.stats() : null;
      if (stats) { frontier = stats.queued + stats.active; busy = stats.active; }
      postMessage({type: 'progress', id, fraction: Math.min(1, (performance.now() - start) / ms), stage: {name: 'proving'},
        live: {solver: {elapsed_ms: Math.round(performance.now() - start), root_nodes: root.nodes, frontier, busy, workers, proof: null}}});
    };
    try {
      result = await owner.search({network, batchSize, choice: 'policy',
        proofs: pool && {workers, slice: 8, table: 4, stamps, cancel: pool.cooperative ? worker => pool.cancel(worker) : null,
          query: (worker, request, stopped) => pool.query(worker, request, stopped)},
        stop: () => cancelled.has(id) || root.mine !== null, onBatch: report});
    } finally { owner.close(); }
  } finally {
    graph.close();
    root.done = true;
    stopSolver(id);
    await asking;
  }
  const exact = result.proven ? (result.proven > 0 ? player : 1 - player) : -1;
  const proof = !root.mine && exact >= 0 ? {winner: exact, turns: proofTurns(result.proof_plies, remaining, exact === player), plies: result.proof_plies} : null;
  const nativeNodes = result.proof_scheduler?.fresh_nodes || 0, certificates = result.proof_records?.length || 0;
  return {mine: root.mine, proof, records: result.proof_records || [], used: root.nodes + nativeNodes, error: result.solver_error,
    solver: {elapsed_ms: Math.round(performance.now() - start), root_nodes: root.nodes, native_nodes: nativeNodes, certificates, workers}};
}

/** Bubble's turn from `history` with the fields of python/play.py evaluate (moves, value, top, proof, pv, threat, solved, ms),
 * plus `solver_error` when the solver's worker could not run, so the turn has no proof or threat. Under a clock `ms` is the
 * turn's time, as the timed engine spends it: the solver gets at most a quarter, the first stone 60% of the rest and the
 * simulations are a ceiling; a stone whose search has not finished by its time plays the search's choice so far, or the
 * network's policy before any. */
async function turn(request) {
  const previous = gameTurn;
  let release = null;
  gameTurn = new Promise(resolve => { release = resolve; });
  try {
    await previous;
    if (cancelled.has(request.id)) throw new Cancelled();
    return await playTurn(request);
  } finally {
    release?.();
  }
}

/** The search of `turn`: each stone searches a GameGraph with a NativeOwner (the hybrid scheduler) for `simulations`
 * completed simulations over all its views, in quanta of 64, or of the stone's work when that is smaller (4 at
 * least). `line` (a seat's or the analysis board's game, see GameGraphs) searches that game's graph; without it the
 * turn searches a graph of its own. With `solverNodes` the root queries (mover win, opponent threat, defender) run
 * first at that node budget, and the owner's proof frontier runs on `solverWorkers` proof workers during each search.
 * `known` (Proofs.list() of the game's table, or null) answers a position it proves won for the mover without solver
 * or search, gives a position it proves lost for the mover its proof and line, and marks the proven stones of each
 * search root exact before it searches (NeuralSearch.settle); a stone the graph does not take is applied to the
 * search's result (proof.mjs settled). */
async function playTurn({id, history, model, simulations, solverNodes, batchSize = 16, choice = 'policy', qRangeFloor = 0, ms = null, line = null, known = null, replay = [], proofStamps = true, solverWorkers = 1, solverSlice = 8, solverTable = 4, proveMs = 0}) {
  await use(model, new Stages(postMessage, id));
  let failure = null, proofPool = null;
  // A browser that cannot start the proof workers still searches; the turn reports why it has no frontier proofs.
  if (solverNodes || proveMs) {
    proofPool = await unlessCancelled(id, proofWorkers(solverWorkers).catch(error => { failure = error.message; return null; }));
  }
  const start = performance.now(), check = () => { if (cancelled.has(id)) throw new Cancelled(); };
  check();
  const state = native.game(history), player = state.player;
  if (state.winner >= 0) throw new Error('The game has finished');
  const table = new Proofs(known || []), given = answered(native, history, table), frontier = new Map();
  if (given) return {...given, ms: Math.round(performance.now() - start)};
  let moves = [], top = [], value = null, proof = null, pv = [], threat = [], solved = true, completed = 0, solverUsed = 0, tree = null, touched = null;
  let solverStats = null;
  const scheduler = [], graphOptions = {seed: 1740, tactics: true, qRangeFloor, model: network.version, roundBarrier: true};
  if (line != null) tree = games.graph(line, history, graphOptions);
  const merged = new Map((tree?.facts() || []).map(f => [proofKey(f.history), f]));
  for (const fact of table?.facts(history) || []) {
    const key = proofKey(fact.history), old = merged.get(key);
    if (old && old.winner !== fact.winner) throw new Error('Contradictory graph and stored proofs');
    if (!old || fact.plies <= old.plies) merged.set(key, fact);
  }
  const facts = [];
  let cells = 0;
  for (const fact of [...merged.values()].sort((a, b) => a.history.length - b.history.length)) {
    if (facts.length >= 2048 || cells + fact.history.length > 200000) break;
    facts.push(fact); cells += fact.history.length;
  }
  const premises = facts.map(({history, winner, plies}) => ({history, winner, plies}));
  let nodeValue = null;
  const note = r => { if (r.reason?.startsWith(FAILED)) failure = r.reason; return r; };
  const timed = ms != null, end = start + (ms ?? 0), solverEnd = start + .25 * (ms ?? 0);
  const predict = async (history, actions) => {
    const key = cache.key(history, network.version);
    let prediction = cache.get(key);
    if (prediction === undefined) {
      const [found] = await network.evaluate([{history, actions}]);
      prediction = {logits: Float64Array.from(found.logits), q: Float64Array.from(found.q)};
      cache.put(key, prediction);
    }
    check();
    return prediction;
  };
  const deadline = Math.min(60000, Math.max(10000, Math.floor(solverNodes / 8)));
  const solverMs = () => timed ? Math.max(1, Math.floor(Math.min(deadline, solverEnd - performance.now()))) : deadline;
  try {
    if (proveMs) {
      const found = await proveRoot(id, history, player, {ms: proveMs, workers: solverWorkers, facts, stamps: proofStamps, batchSize, pool: proofPool});
      check();
      solverUsed += found.used; solverStats = found.solver; failure ||= found.error;
      frontierProofs(found.records, table, frontier);
      // A root win joins the proof table, so the Standard search below still runs and the table gives the turn.
      if (found.mine) {
        const record = winningLine(native, history, found.mine, facts.filter(f => f.history.length !== history.length || f.winner !== player));
        table.add(history, record);
        ({pv, proof} = record);
      } else if (found.proof) proof = found.proof;
    } else if (solverNodes) {
      postMessage({type: 'progress', id, fraction: 0, stage: {name: 'checking proof'}});
      if (!timed) {
        const actions = native.legal(history), prediction = await predict(history, actions);
        const maximum = Math.max(...prediction.logits), weights = Array.from(prediction.logits, l => Math.exp(l - maximum));
        const total = weights.reduce((a, b) => a + b, 0), policy = weights.map(w => w / total);
        const action = actions[policy.indexOf(Math.max(...policy))];
        postMessage({type: 'progress', id, fraction: 0, stage: {name: 'checking proof'},
          live: rootRows(tree, choice) || {value: (prediction.q[0] + 1) / 2, top: topRows(actions, policy, null, action)}});
      }
      let replayed = null;
      if (proofStamps && !timed) for (const winner of [player, 1 - player]) {
        const lines = replay.filter(r => r.winner === winner);
        if (!lines.length) continue;
        const found = note(await solve(id, history, {attacker: winner === player ? 'mover' : 'defender',
          nodes: Math.min(solverNodes, 20000), ms: Math.min(15000, solverMs()), stamps: true, replay: lines}));
        check();
        solverUsed += found.nodes_used || 0;
        if (found.native_verified && (found.status === 'PROVEN_WIN' || found.status === 'PROVEN_LOSS')) { replayed = found; break; }
      }
      const mineFacts = facts.filter(f => f.history.length !== history.length || f.winner !== player);
      const mine = replayed || note(await solve(id, history, {attacker: 'mover', nodes: solverNodes, ms: solverMs(), shortest: true,
        stamps: proofStamps,
        known: mineFacts.map(({history, winner, plies}) => ({history, winner, plies}))}));
      check();
      solved = replayed !== null || searched(mine);
      if (!replayed) solverUsed += mine.nodes_used || 0;
      if (verified(mine) && mine.moves.length) {
        ({moves, pv, proof} = winningLine(native, history, mine, mineFacts));
        top = [[...moves[0], 1, 1, 1]];
      } else {
        postMessage({type: 'progress', id, fraction: 0, stage: {name: 'checking threats'}});
        const theirs = replayed || note(await solve(id, history, {attacker: 'opponent', nodes: solverNodes, ms: solverMs(), known: premises, stamps: proofStamps}));
        check();
        solved = solved && (replayed !== null || searched(theirs));
        if (!replayed) solverUsed += theirs.nodes_used || 0;
        if (verified(theirs)) threat = theirs.moves.map(m => [...m]);
        if ((replayed || verified(theirs) || premises.length) && (!timed || performance.now() < solverEnd)) {
          postMessage({type: 'progress', id, fraction: 0, stage: {name: 'checking defence'}});
          const defended = replayed || note(await solve(id, history, {attacker: 'defender', nodes: solverNodes, ms: solverMs(), known: premises, stamps: proofStamps}));
          check();
          const lost = defended.status === 'PROVEN_LOSS' && defended.native_verified;
          solved = solved && (lost || searched(defended));
          if (!replayed) solverUsed += defended.nodes_used || 0;
          if (lost) {
            pv = principalVariation(native, history, defended.certificate, {attacker: 1 - player, known: facts}).pv;
            proof = {winner: 1 - player, turns: defended.proof_turns, plies: state.remaining + 2 + 4 * (defended.proof_turns - 1),
              ...proofEvidence(defended)};
          }
        }
      }
    }
    const outcome = proof === null ? table?.known(history) : null;
    if (outcome && outcome.winner !== player) {
      proof = {winner: outcome.winner, turns: proofTurns(outcome.plies, state.remaining, false), plies: outcome.plies};
      pv = outcome.pv;
    }
    const given = moves.length > 0, current = history.map(p => [...p]), searchStart = performance.now();
    for (let local = native.game(current); !given && local.player === player && local.winner < 0; local = native.game(current)) {
      let action, policy, actions, stoneValue, values = null;
      const stone = moves.length, stoneEnd = !timed || local.remaining === 1 || stone ? end : searchStart + .6 * (end - searchStart);
      const raw = async () => {
        actions = native.legal(current);
        const prediction = await predict(current, actions);
        const maximum = Math.max(...prediction.logits), weights = Array.from(prediction.logits, l => Math.exp(l - maximum));
        const total = weights.reduce((a, b) => a + b, 0);
        policy = weights.map(w => w / total);
        action = actions[policy.indexOf(Math.max(...policy))];
        stoneValue = prediction.q[0];
      };
      if (simulations) {
        tree ??= line == null ? new GameGraph(native, {...graphOptions, history: current})
          : games.graph(line, current, graphOptions);
        if (proof && proof.winner !== player) {
          tree.proveLoss(proof.winner, Math.max(1, proof.plies - moves.length));
          if (line != null) touched = tree.id;
        }
        const evaluate = leaves => network.evaluate(leaves), edges = table ? table.edges(current) : new Map();
        // `touched` names the game graph once this turn changed its statistics: marks settled, or a batch backed up.
        const unmarked = await tree.settle(edges, {evaluate, cache, version: network.version});
        if (line != null && edges.size) touched = tree.id;
        check();
        let searchedResult;
        const owner = new NativeOwner(tree, {quantum: Math.max(4, Math.min(64, simulations)), work: simulations,
          ms: timed ? Math.max(1, stoneEnd - performance.now()) : 0});
        try {
          searchedResult = await owner.search({network, batchSize, choice,
            proofs: proofPool && {workers: solverWorkers, slice: solverSlice, table: solverTable, stamps: proofStamps,
              cancel: proofPool.cooperative ? worker => proofPool.cancel(worker) : null,
              query: (worker, request, stopped) => proofPool.query(worker, request, stopped)},
            stop: () => cancelled.has(id), onBatch: stats => {
              if (line != null) touched = tree.id;
              postMessage({type: 'progress', id, fraction: timed ? Math.min(1, (performance.now() - start) / ms)
                : Math.min(1, (stone + stats.completed / simulations) / state.remaining),
                ...(stone ? {} : {live: rootRows(owner.root, choice)})});
            }});
          scheduler.push({...searchedResult.scheduler, inference: searchedResult.inference,
            ...(proofPool ? {proof: searchedResult.proof_scheduler} : {})});
          if (proofPool) {
            solverUsed += searchedResult.proof_scheduler.fresh_nodes;
            failure ||= searchedResult.solver_error;
            frontierProofs(searchedResult.proof_records, table, frontier);
          }
        } finally { owner.close(); }
        const result = settled(searchedResult, unmarked, local.player);
        if (line != null && result.completed) touched = tree.id;
        check();
        completed += result.completed;
        if (result.action) {
          ({action, policy, actions, completed_q: values} = result);
          if (!moves.length) nodeValue = (result.node_value + 1) / 2;
          stoneValue = result.proven ? result.proven : result.exact_winner >= 0 ? (result.exact_winner === local.player ? 1 : -1)
            : policy.reduce((sum, p, i) => sum + p * result.values[i], 0);
          if (proof === null && (result.proven > 0 || (result.proven < 0 && !moves.length))) {
            proof = {winner: result.proven > 0 ? player : 1 - player, turns: proofTurns(result.proof_plies, local.remaining, result.proven > 0),
              plies: result.proof_plies + moves.length};
          }
        } else await raw();
      } else await raw();
      if (!moves.length) {
        top = topRows(actions, policy, values, action);
        value = (stoneValue + 1) / 2;
      }
      moves.push([action[0], action[1]]);
      current.push([action[0], action[1]]);
      tree?.advance(action);
    }
    if (line != null && tree && moves.length > 1 && !given) {
      // The later stones' searches changed the graph under the first root: read that root again.
      tree.at(history.map(p => [...p]));
      const root = tree.result(choice);
      top = topRows(root.actions, root.policy, root.completed_q, moves[0]);
      value = (root.policy.reduce((sum, p, i) => sum + p * root.values[i], 0) + 1) / 2;
      nodeValue = (root.node_value + 1) / 2;
    }
    if (proof) value = proof.winner === player ? 1 : 0;
    if (!proof && top.length && top.every(row => row[4] < 0) && nodeValue != null) value = nodeValue;
    if (proof && !pv.length) {
      const after = table?.known([...history, ...moves]);
      pv = moves.map(([q, r], i) => [q, r, player, i + 1]);
      if (after?.winner === proof.winner) pv.push(...after.pv.map(([q, r, side, ply]) => [q, r, side, ply + moves.length]));
    }
    const knownTurn = answered(native, history, table);
    if (knownTurn && (!proof || knownTurn.proof.plies <= proof.plies)) ({moves, value, top, proof, pv} = knownTurn);
    else if (proof) pv = table.line(history, {winner: proof.winner, plies: proof.plies, pv});
    return proven(table, history, {moves, value: Math.round(value * 1e4) / 1e4, node_value: nodeValue, top, proof, pv, threat, solved, ms: Math.round(performance.now() - start),
      actual_completed: completed, actual_solver_nodes: solverUsed, graph_id: touched,
      scheduler, ...(frontier.size ? {proofs: [...frontier.values()]} : {}),
      ...(solverStats ? {solver: {...solverStats, proof}} : {}),
      ...(failure ? {solver_error: failure} : {})}, state.remaining);
  } catch (error) {
    // A turn that stops after touching its game graph names the graph, so the session can count that search.
    if (error && typeof error === 'object') error.graph = touched;
    throw error;
  } finally {
    if (line == null) tree?.close();
  }
}

/** Forward latency in ms per batch {size: {batch: ms}} over `repeats` timed runs after two warm-up runs. */
async function bench({batches = [1, 16, 64], sizes = [24, 32], repeats = 10}) {
  const out = {};
  for (const size of sizes) {
    out[size] = {};
    for (const count of batches) out[size][count] = await network.time(count, size, repeats);
  }
  return out;
}

/** Searches with the network whose manifest is `model` (relative to web/engine) from now on, reporting its download and
 * session to `stages`; the two most recently used stay loaded. */
async function use(model, stages) {
  const previous = modelUse; let release;
  modelUse = new Promise(resolve => { release = resolve; });
  await previous;
  try {
    if (held.has(model)) {
      const kept = held.get(model);
      if (kept.closed) {
        await kept.close();
        held.delete(model);
      } else {
        network = kept;
        held.delete(model);
        held.set(model, network);
        return;
      }
    }
    // Retire before admitting another model. A failed close retains its retry
    // owner without allowing subsequent requests to grow the cache.
    while (held.size >= 2) {
      const [old, released] = held.entries().next().value;
      await released.close();
      held.delete(old);
    }
    network = await stages.run(() => Network.create({model, device, ort, stages}));
    held.set(model, network);
  } finally { release(); }
}

/** Probes the device, starts ONNX Runtime and the search, loads `options.model` and warms both up, reporting each
 * stage. */
async function load(options = {}) {
  const stages = new Stages(postMessage);
  return stages.run(async () => {
    stages.enter('probe');
    device = await probe(options.prefer);
    stages.probed(device);
    stages.enter('download');   // gumbel.mjs fetches gumbel.wasm
    native = new Native(await createModule());
    ort = await runtime(device.provider, options.threads, stages);
    await use(options.model, stages);
    cache = new EvaluationCache(4096);
    games = new GameGraphs(native);
    stages.enter('warmup', device.provider);
    const t = performance.now();
    for (const history of [[[0, 0]], [[0, 0], [1, 0], [0, 1], [5, 0], [6, 0]]]) {
      const leaf = {history, actions: native.legal(history)};
      await stall('warmup', () => network.evaluate([leaf]));
      await network.evaluate(new Array(16).fill(leaf));
    }
    return {provider: device.provider, precision: network.precision, timings: network.timings, adapter: device.adapter,
      threads: network.threads, isolated: Boolean(globalThis.crossOriginIsolated), warmup_ms: Math.round(performance.now() - t),
      model: network.version};
  });
}

onmessage = async ({data}) => {
  if (data.type === 'cancel') {
    cancelled.add(data.id);
    cancelWaits.get(data.id)?.();
    stopSolver(data.id);
    return;
  }
  try {
    if (data.type === 'load') postMessage({type: 'ready', device: await load(data.options)});
    else if (data.type === 'turn') postMessage({type: 'result', id: data.id, result: await turn(data)});
    else if (data.type === 'use') { await gameTurn; await use(data.model, new Stages(postMessage, data.id)); postMessage({type: 'result', id: data.id, result: null}); }
    else if (data.type === 'bench') postMessage({type: 'result', id: data.id, result: await bench(data)});
    else if (data.type === 'evaluate') {
      const leaves = data.histories.map(history => ({history, actions: native.legal(history)}));
      const predictions = await network.evaluate(leaves);
      if (cancelled.has(data.id)) throw new Cancelled();
      postMessage({type: 'result', id: data.id, result: predictions.map((p, i) => ({actions: leaves[i].actions, logits: Array.from(p.logits), q: Array.from(p.q)}))});
    }
  } catch (error) {
    const graph = error?.graph ? {graph: error.graph} : {};
    postMessage(error instanceof Cancelled ? {type: 'cancelled', id: data.id, ...graph} : {...errorReport(error, data.id), ...graph});
  } finally {
    cancelled.delete(data.id);
  }
};
