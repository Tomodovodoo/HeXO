/* Bubble in a Web Worker: network (network.mjs), native search (gumbel.wasm) and the tactical solver (solver-worker.mjs).
 * In: {type: 'load', options} | {type: 'turn', id, history, simulations, solverNodes, batchSize} | {type: 'cancel', id}
 *     | {type: 'bench', id, batches, sizes, repeats} | {type: 'search', id, history, simulations, batchSize}
 *     | {type: 'evaluate', id, histories}.
 * Out: {type: 'progress', id?, fraction} | {type: 'ready', device} | {type: 'result', id, result} | {type: 'cancelled', id}
 *     | {type: 'error', id?, message}.
 */
import createModule from './gumbel.mjs';
import {Native, NeuralSearch, EvaluationCache} from './search.mjs';
import {Network, probe} from './network.mjs';
import {principalVariation, topRows} from './proof.mjs';

const VERDICTS = new Set(['no verified strategy', 'quiet defender unsupported', 'defender counterwin',
  'candidate has unproved defender continuation', 'candidate defense expansion budget', 'candidate certificate size limit',
  'free-second coverage work limit']);
let native, network, cache, solver = null, solverCalls = 0;
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
      for (const id of [...solverWaits.keys()]) settle(id, unknown(`solver worker failed: ${event.message || 'error'}`));
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
const searched = r => verified(r) || VERDICTS.has(r.reason);

function proofTurns(plies, remaining, moverWins) {
  if (moverWins) return plies <= remaining ? 1 : 1 + Math.ceil((plies - remaining) / 4);
  return Math.ceil((plies - remaining) / 4);
}

/** Bubble's turn from `history` with the fields of python/play.py evaluate (moves, value, top, proof, pv, threat, solved, ms). */
async function turn({id, history, simulations, solverNodes, batchSize = 16, choice = 'policy'}) {
  const start = performance.now(), check = () => { if (cancelled.has(id)) throw new Cancelled(); };
  const state = native.game(history), player = state.player;
  if (state.winner >= 0) throw new Error('The game has finished');
  let moves = [], top = [], value = null, proof = null, pv = [], threat = [], solved = true, completed = 0, solverUsed = 0, tree = null;
  const deadline = Math.min(60000, Math.max(10000, Math.floor(solverNodes / 8)));
  try {
    if (solverNodes) {
      const mine = await solve(id, history, {attacker: 'mover', nodes: solverNodes, ms: deadline, shortest: true});
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
        const theirs = await solve(id, history, {attacker: 'opponent', nodes: solverNodes, ms: deadline});
        check();
        solved = solved && searched(theirs);
        solverUsed += theirs.nodes_used || 0;
        if (verified(theirs)) threat = theirs.moves.map(m => [...m]);
      }
    }
    const given = moves.length > 0, current = history.map(p => [...p]);
    for (let local = native.game(current); !given && local.player === player && local.winner < 0; local = native.game(current)) {
      let action, policy, actions, stoneValue, values = null;
      if (simulations) {
        tree ??= new NeuralSearch(native, {seed: 1740, tactics: true, history: current});
        const stone = moves.length;
        const result = await tree.search({simulations, rootSamples: 16, batchSize, cache, version: network.version, choice,
          evaluate: leaves => network.evaluate(leaves), stop: () => cancelled.has(id),
          onBatch: () => postMessage({type: 'progress', id, fraction: Math.min(1, (stone + tree.m._hxg_completed(tree.ptr) / simulations) / state.remaining)})});
        check();
        ({action, policy, actions, values} = result);
        completed += result.completed;
        stoneValue = result.proven ? result.proven : result.exact_winner >= 0 ? (result.exact_winner === local.player ? 1 : -1)
          : policy.reduce((sum, p, i) => sum + p * result.values[i], 0);
        if (proof === null && (result.proven > 0 || (result.proven < 0 && !moves.length))) {
          proof = {winner: result.proven > 0 ? player : 1 - player, turns: proofTurns(result.proof_plies, local.remaining, result.proven > 0),
            plies: result.proof_plies + moves.length};
        }
      } else {
        actions = native.legal(current);
        const [prediction] = await network.evaluate([{history: current, actions}]);
        check();
        const maximum = Math.max(...prediction.logits), weights = Array.from(prediction.logits, l => Math.exp(l - maximum));
        const total = weights.reduce((a, b) => a + b, 0);
        policy = weights.map(w => w / total);
        action = actions[policy.indexOf(Math.max(...policy))];
        stoneValue = prediction.q[0];
      }
      if (!moves.length) {
        top = topRows(actions, policy, values, action);
        value = (stoneValue + 1) / 2;
      }
      moves.push([action[0], action[1]]);
      current.push([action[0], action[1]]);
      tree?.advance(action);
    }
    if (proof) value = proof.winner === player ? 1 : 0;
    if (proof && !pv.length) pv = moves.map(([q, r]) => [q, r, player]);
    return {moves, value: Math.round(value * 1e4) / 1e4, top, proof, pv, threat, solved, ms: Math.round(performance.now() - start),
      actual_completed: completed, actual_solver_nodes: solverUsed};
  } finally {
    tree?.close();
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

async function load(options = {}) {
  native = new Native(await createModule());
  let device = await probe(options.prefer);
  const report = fraction => postMessage({type: 'progress', fraction: .95 * fraction});
  const create = () => Network.create(new URL('./', import.meta.url), {model: options.model, device, progress: report, threads: options.threads});
  try {
    network = await create();
  } catch (error) {
    if (device.provider !== 'webgpu' || options.prefer) throw error;
    device = {provider: 'wasm', precisions: ['fp32'], adapter: '', fallback: String(error.message || error)};
    network = await create();
  }
  cache = new EvaluationCache(4096);
  const t = performance.now();
  for (const history of [[[0, 0]], [[0, 0], [1, 0], [0, 1], [5, 0], [6, 0]]]) {
    const leaf = {history, actions: native.legal(history)};
    await network.evaluate([leaf]);
    await network.evaluate(new Array(16).fill(leaf));
  }
  postMessage({type: 'progress', fraction: 1});
  return {provider: device.provider, precision: network.precision, timings: network.timings, adapter: device.adapter, fallback: device.fallback,
    threads: network.threads, isolated: Boolean(globalThis.crossOriginIsolated), warmup_ms: Math.round(performance.now() - t),
    model: network.version};
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
    else if (data.type === 'bench') postMessage({type: 'result', id: data.id, result: await bench(data)});
    else if (data.type === 'evaluate') {
      const leaves = data.histories.map(history => ({history, actions: native.legal(history)}));
      const predictions = await network.evaluate(leaves);
      if (cancelled.has(data.id)) throw new Cancelled();
      postMessage({type: 'result', id: data.id, result: predictions.map((p, i) => ({actions: leaves[i].actions, logits: Array.from(p.logits), q: Array.from(p.q)}))});
    }
    else if (data.type === 'search') {
      const tree = new NeuralSearch(native, {seed: 1740, tactics: true, history: data.history});
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
    postMessage(error instanceof Cancelled ? {type: 'cancelled', id: data.id} : {type: 'error', id: data.id, message: String(error.message || error)});
  } finally {
    cancelled.delete(data.id);
  }
};
