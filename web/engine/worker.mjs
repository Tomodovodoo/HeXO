/* Bubble in a Web Worker: network (network.mjs), native search (gumbel.wasm) and the tactical solver (solver-worker.mjs).
 * In: {type: 'load', options} | {type: 'use', id, model} | {type: 'turn', id, history, model, simulations, solverNodes, leafNodes, batchSize, qRangeFloor, ms, line, known}
 *     | {type: 'cancel', id} | {type: 'bench', id, batches, sizes, repeats}
 *     | {type: 'search', id, history, simulations, batchSize, qRangeFloor}
 *     | {type: 'evaluate', id, histories}.
 * Out: {type: 'progress', id?, fraction, stage?} | {type: 'ready', device} | {type: 'result', id, result} | {type: 'cancelled', id}
 *     | {type: 'error', id?, message, stage?}: stages.mjs's loading stages; a stage in an error is where it stopped.
 */
import createModule from './gumbel.mjs';
import {Native, NeuralSearch, EvaluationCache, GameGraphs, PV_CHECK} from './search.mjs';
import {Network, probe, runtime} from './network.mjs';
import {Stages, errorReport, stall} from './stages.mjs';
import {principalVariation, topRows, Proofs, answered, settled, proofTurns} from './proof.mjs';

const VERDICTS = new Set(['no verified strategy', 'quiet defender unsupported', 'defender counterwin',
  'candidate has unproved defender continuation', 'candidate defense expansion budget', 'candidate certificate size limit',
  'free-second coverage work limit']);
let native, ort, network, cache, games, device, solver = null, solverCalls = 0;
/** The latest game-tree turn: they run one at a time, so a cancelled turn still awaiting the network settles before
 * another turn advances, searches or evicts a game tree. */
let gameTurn = Promise.resolve();
const held = new Map();
const cancelled = new Set(), solverWaits = new Map();

class Cancelled extends Error {}

const GRACE_MS = 500;
const unknown = reason => ({status: 'UNKNOWN', native_verified: false, moves: [], nodes_used: 0, reason});

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
    solver = new Worker(new URL('solver-worker.mjs', import.meta.url), {type: 'module'});
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
  const {action, actions, policy, values, completed_q} = tree.result(choice);
  if (!action) return null;
  const value = policy.reduce((sum, p, i) => sum + p * values[i], 0);
  return {value: Math.round((value + 1) / 2 * 1e4) / 1e4, top: topRows(actions, policy, completed_q, action)};
}

/** Bubble's turn from `history` with the fields of python/play.py evaluate (moves, value, top, proof, pv, threat, solved, ms),
 * plus `solver_error` when the solver's worker could not run, so the turn has no proof or threat. Under a clock `ms` is the
 * turn's time, as the timed engine spends it: the solver gets at most a quarter, the first stone 60% of the rest and the
 * simulations are a ceiling; a stone whose search has not finished by its time plays the search's choice so far, or the
 * network's policy before any. */
async function turn(request) {
  const previous = gameTurn;
  let release = null;
  if (request.line != null) gameTurn = new Promise(resolve => { release = resolve; });
  try {
    if (request.line != null) {
      await previous;
      if (cancelled.has(request.id)) throw new Cancelled();
    }
    return await playTurn(request);
  } finally {
    release?.();
  }
}

/** The search of `turn`, with `line` (a seat's or the analysis board's game, see GameGraphs) searching that game's graph
 * with the principal-variation check PV_CHECK as play.py does; without it the turn searches a tree of its own. `known` (Proofs.list() of the game's table, or null) answers a
 * position it proves won for the mover without solver or search, gives a position it proves lost for the mover its
 * proof and line, and marks the proven stones of each search root exact before it searches (NeuralSearch.settle); a
 * stone the tree does not take is applied to the search's result (proof.mjs settled). */
async function playTurn({id, history, model, simulations, solverNodes, leafNodes: leafBudget = 0, leafQueryMs = 10, batchSize = 16, choice = 'policy', qRangeFloor = 0, ms = null, line = null, known = null}) {
  await use(model, new Stages(postMessage, id));
  const start = performance.now(), check = () => { if (cancelled.has(id)) throw new Cancelled(); };
  const state = native.game(history), player = state.player;
  if (state.winner >= 0) throw new Error('The game has finished');
  const table = known ? new Proofs(known) : null, given = answered(native, history, table);
  if (given) return {...given, ms: Math.round(performance.now() - start)};
  let moves = [], top = [], value = null, proof = null, pv = [], threat = [], solved = true, completed = 0, solverUsed = 0, tree = null, touched = null;
  let failure = null;
  const note = r => { if (r.reason?.startsWith(FAILED)) failure = r.reason; return r; };
  const timed = ms != null, end = start + (ms ?? 0), solverEnd = start + .25 * (ms ?? 0);
  const deadline = Math.min(60000, Math.max(10000, Math.floor(solverNodes / 8)));
  const solverMs = () => timed ? Math.max(1, Math.floor(Math.min(deadline, solverEnd - performance.now()))) : deadline;
  let leafNodes = leafBudget, leafMs = Math.min(60000, Math.max(10000, Math.floor(leafBudget / 8)));
  const prove = leafBudget ? async leaves => {
    const ms = Math.min(leafQueryMs, Math.floor(leafMs), timed ? Math.floor(solverEnd - performance.now()) : leafQueryMs);
    if (!leafNodes || ms < 1) return null;
    check();
    const before = performance.now();
    const found = note(await solve(id, leaves, {nodes: Math.min(2048, leafNodes), ms}));
    const used = found.nodes_used || 0;
    leafNodes = Math.max(0, leafNodes - used);
    leafMs -= performance.now() - before;
    solverUsed += used;
    check();
    return found;
  } : null;
  try {
    if (solverNodes) {
      const mine = note(await solve(id, history, {attacker: 'mover', nodes: solverNodes, ms: solverMs(), shortest: true}));
      check();
      solved = searched(mine);
      solverUsed += mine.nodes_used || 0;
      if (verified(mine)) {
        moves = mine.moves.map(m => [...m]);
        const found = principalVariation(native, history, mine.certificate);
        pv = found.pv;
        proof = {winner: player, turns: mine.proof_turns, plies: found.plies};
        top = [[...moves[0], 1, 1, 1]];
      } else {
        const theirs = note(await solve(id, history, {attacker: 'opponent', nodes: solverNodes, ms: solverMs()}));
        check();
        solved = solved && searched(theirs);
        solverUsed += theirs.nodes_used || 0;
        if (verified(theirs)) threat = theirs.moves.map(m => [...m]);
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
        const [prediction] = await network.evaluate([{history: current, actions}]);
        check();
        const maximum = Math.max(...prediction.logits), weights = Array.from(prediction.logits, l => Math.exp(l - maximum));
        const total = weights.reduce((a, b) => a + b, 0);
        policy = weights.map(w => w / total);
        action = actions[policy.indexOf(Math.max(...policy))];
        stoneValue = prediction.q[0];
      };
      if (simulations) {
        tree ??= line == null ? new NeuralSearch(native, {seed: 1740, tactics: true, qRangeFloor, history: current})
          : games.graph(line, current, {seed: 1740, tactics: true, qRangeFloor, model: network.version});
        const evaluate = leaves => network.evaluate(leaves), edges = table ? table.edges(current) : new Map();
        // `touched` names the game graph once this turn changed its statistics: marks settled, or a batch backed up.
        const unmarked = await tree.settle(edges, {evaluate, cache, version: network.version});
        if (line != null && edges.size) touched = tree.id;
        check();
        const result = settled(await tree.search({simulations, rootSamples: 16, batchSize, cache, version: network.version, choice,
          ...(line == null ? {} : {pvCheck: PV_CHECK}),
          evaluate, prove, stop: () => cancelled.has(id) || timed && performance.now() >= stoneEnd,
          onBatch: () => (line != null && (touched = tree.id), postMessage({type: 'progress', id, fraction: Math.min(1, (stone + tree.m._hxg_completed(tree.ptr) / simulations) / state.remaining),
            ...(stone ? {} : {live: rootRows(tree, choice)})}))}), unmarked, local.player);
        if (line != null && result.completed) touched = tree.id;
        check();
        completed += result.completed;
        if (result.action) {
          ({action, policy, actions, completed_q: values} = result);
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
    }
    if (proof) value = proof.winner === player ? 1 : 0;
    if (proof && !pv.length) {
      const after = table?.known([...history, ...moves]);
      pv = moves.map(([q, r], i) => [q, r, player, i + 1]);
      if (after?.winner === proof.winner) pv.push(...after.pv.map(([q, r, side, ply]) => [q, r, side, ply + moves.length]));
    }
    return {moves, value: Math.round(value * 1e4) / 1e4, top, proof, pv, threat, solved, ms: Math.round(performance.now() - start),
      actual_completed: completed, actual_solver_nodes: solverUsed, graph_id: touched,
      ...(failure ? {solver_error: failure} : {})};
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
  if (held.has(model)) {
    network = held.get(model);
    held.delete(model);
    held.set(model, network);
    return;
  }
  network = await stages.run(() => Network.create({model, device, ort, stages}));
  held.set(model, network);
  while (held.size > 2) {
    const [old, released] = held.entries().next().value;
    held.delete(old);
    await released.session.release();
  }
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
    stopSolver(data.id);
    return;
  }
  try {
    if (data.type === 'load') postMessage({type: 'ready', device: await load(data.options)});
    else if (data.type === 'turn') postMessage({type: 'result', id: data.id, result: await turn(data)});
    else if (data.type === 'use') { await use(data.model, new Stages(postMessage, data.id)); postMessage({type: 'result', id: data.id, result: null}); }
    else if (data.type === 'bench') postMessage({type: 'result', id: data.id, result: await bench(data)});
    else if (data.type === 'evaluate') {
      const leaves = data.histories.map(history => ({history, actions: native.legal(history)}));
      const predictions = await network.evaluate(leaves);
      if (cancelled.has(data.id)) throw new Cancelled();
      postMessage({type: 'result', id: data.id, result: predictions.map((p, i) => ({actions: leaves[i].actions, logits: Array.from(p.logits), q: Array.from(p.q)}))});
    }
    else if (data.type === 'search') {
      const tree = new NeuralSearch(native, {seed: 1740, tactics: true, qRangeFloor: data.qRangeFloor ?? 0, history: data.history});
      try {
        const result = await tree.search({simulations: data.simulations, rootSamples: 16, batchSize: data.batchSize ?? 16,
          choice: data.choice ?? 'policy',
          cache: new EvaluationCache(4096), version: network.version, evaluate: leaves => network.evaluate(leaves),
          stop: () => cancelled.has(data.id)});
        if (result.stopped) throw new Cancelled();
        postMessage({type: 'result', id: data.id, result: {action: result.action, completed: result.completed, elapsed_ms: result.elapsed_ms,
          evaluated: result.evaluated, batches: result.inference_batches, network_ms: result.network_ms, policy: result.policy,
          actions: result.actions}});
      } finally {
        tree.close();
      }
    }
  } catch (error) {
    const graph = error?.graph ? {graph: error.graph} : {};
    postMessage(error instanceof Cancelled ? {type: 'cancelled', id: data.id, ...graph} : {...errorReport(error, data.id), ...graph});
  } finally {
    cancelled.delete(data.id);
  }
};
