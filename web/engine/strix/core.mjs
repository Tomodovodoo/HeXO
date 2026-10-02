/** Strix (tools/strix_web compiled to strix.wasm) for browser workers and node.
 *
 * `loadStrix(source, {progress})` instantiates the module; `load(weights)` reads safetensors bytes and `turn(history,
 * simulations)` plays one turn the way python/play.py's Strix child does through tools/strix_learned_adapter.py: the
 * stones go to Strix's frame (HTTTX (q, r) is (q + r, -r)), each placement is searched with `simulations`, 4 root
 * actions and seed 0 (the search uses no randomness), and the moves come back in HTTTX coordinates. A trap
 * re-instantiates the module and reloads the weights before it is reported.
 */
import {compile, wasi, Exit} from '../tactical.mjs';

const ACTIONS = 4;
export const mirror = ([q, r]) => [q + r, -r];
const playerAt = ply => ply === 0 ? 0 : ((ply - 1 >> 1) + 1) % 2;

/** The request of tools/strix_learned for the position after `history`: stones in Strix's frame, mover, placements left. */
export function request(history, simulations) {
  const n = history.length;
  return {stones: history.map((point, ply) => [...mirror(point), playerAt(ply)]), player: playerAt(n),
    remaining: n % 2 === 1 ? 2 : 1, simulations, actions: ACTIONS, seed: 0};
}

/** Load from a URL string, Response, bytes or WebAssembly.Module; `progress(fraction)` hears each network batch. */
export async function loadStrix(source, {progress = () => {}} = {}) {
  const [module, buildHash] = await compile(source);
  const state = {}, shim = wasi(state);
  const imports = {hexo_strix: {progress: fraction => progress(fraction)}, wasi_snapshot_preview1: {}};
  for (const {module: space, name} of WebAssembly.Module.imports(module)) {
    if (space === 'wasi_snapshot_preview1') imports[space][name] = shim[name] ?? (() => 52);
  }
  let weights = null;
  const bind = instance => { state.exports = instance.exports; state.memory = instance.exports.memory; };
  bind(await WebAssembly.instantiate(module, imports));

  function call(name, bytes) {
    const {strix_alloc: alloc, strix_free: free, strix_free_text: release} = state.exports;
    const input = alloc(bytes.length);
    new Uint8Array(state.memory.buffer, input, bytes.length).set(bytes);
    const output = state.exports[name](input, bytes.length);
    free(input, bytes.length);
    const memory = new Uint8Array(state.memory.buffer, output);
    const text = new TextDecoder().decode(memory.subarray(0, memory.indexOf(0)));
    release(output);
    return JSON.parse(text);
  }

  function guarded(name, bytes) {
    try {
      return call(name, bytes);
    } catch (error) {
      if (!(error instanceof WebAssembly.RuntimeError || error instanceof Exit)) throw error;
      bind(new WebAssembly.Instance(module, imports));
      if (weights) call('strix_load', weights);
      throw new Error(`Strix failed: ${error.message}`);
    }
  }

  /** Reads safetensors `bytes` (ArrayBuffer or view): {status: 'READY', source_checkpoint, metadata}. */
  function load(bytes) {
    const view = new Uint8Array(bytes.buffer ?? bytes, bytes.byteOffset ?? 0, bytes.byteLength);
    const result = guarded('strix_load', view);
    if (result.status !== 'READY') throw new Error(`Strix model: ${result.error}`);
    weights = view;
    return result;
  }

  const round = x => Math.round(x * 1e4) / 1e4;
  const ask = (name, history, simulations) => {
    const answer = guarded(name, new TextEncoder().encode(JSON.stringify(request(history, simulations))));
    if (answer.status !== 'OK') throw new Error(`Strix: ${answer.error}`);
    return answer;
  };

  /**
   * Strix's turn after `history` ([[q, r], ...], HTTTX) at `simulations` per placement: {moves, value (the mover's win
   * probability), top ([q, r, policy, mover's win probability, 0] for up to five first stones), simulations,
   * eval_states, root_visits}. The empty board's only stone is the origin, played without search (simulations 0) and
   * valued by the network's evaluation of the position after it.
   */
  function turn(history, simulations) {
    if (!history.length) {
      const value = round(1 - ask('strix_value', [[0, 0]], 1).value);
      return {moves: [[0, 0]], value, top: [[0, 0, 1, value, 0]], simulations: 0, eval_states: 1, root_visits: []};
    }
    const answer = ask('strix_turn', history, simulations);
    return {moves: answer.moves.map(mirror), value: round(answer.value),
      top: answer.top.map(([q, r, p, v]) => [...mirror([q, r]), round(p), round(v), 0]),
      simulations, eval_states: answer.eval_states, root_visits: answer.root_visits};
  }

  return {load, turn, build_hash: buildHash};
}
