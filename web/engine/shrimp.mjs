/* "Shrimp (browser)": Shrimp (Cmiller132/hexo-bot main_7, MIT) in shrimp-worker.mjs, for the play page's browser
 * engines (seat.mjs). It plays as the server's Shrimp entry (tools/engines.json): Six's driver with these visits per
 * stone, mirrored into Six's frame. */
import {json, workerUrl} from './assets.mjs';
import {EngineWorker} from './engine-worker.mjs';
import {loadFiles, probe} from './network.mjs';
import {ShrimpNetwork} from './shrimp/network.mjs';
import {NEURAL_PRESET} from './device.mjs';

/** The server entry's presets, as visits per stone; `simulations` is what the analysis panel shows. */
export const PRESETS = Object.fromEntries(Object.entries({lightning: 16, quick: 32, standard: 128, strong: 512, deep: 1024,
  dangerous: 4096}).map(([name, visits]) => [name, {visits, simulations: visits}]));
const ID = 'browser:shrimp', LABEL = 'Shrimp (browser)';
const playerAt = ply => ply === 0 ? 0 : ((ply - 1 >> 1) + 1) % 2;
const round = x => Math.round(x * 1e4) / 1e4;

export class ShrimpEngine extends EngineWorker {
  /** `prefer` 'wasm' keeps the network off WebGPU; `threads` fixes ONNX Runtime's WebAssembly thread count. */
  constructor({prefer = null, threads = null} = {}) {
    super(workerUrl('shrimp-worker.mjs'), LABEL, {prefer, threads});
  }

  /**
   * Shrimp's turn at `history` ([[q, r], ...]) with `budget` {visits} (a PRESETS entry): {moves, stones, ms, ...},
   * each stone {action, value (Shrimp's value for the side to move, -1 to 1), visits, moves: root moves with visits,
   * q and share}. Aborting `signal` stops the search after its current network batch and rejects with an AbortError.
   */
  turn(history, budget, options = {}) {
    return this.call({type: 'turn', history, visits: budget.visits}, options);
  }

  /** The downloaded files (assets.mjs records) a load on this device may read: ONNX Runtime (with a WebGPU start's
   * WebAssembly fallback), the graph and shrimp.wasm. */
  async files() {
    const [{provider}, {file}, {data, local}] = await Promise.all([probe(this.options.prefer), ShrimpNetwork.files(), json('build.json')]);
    return [...await loadFiles(provider, this.options.prefer), file, {path: 'shrimp/shrimp.wasm', sha256: data.artefacts['shrimp/shrimp.wasm'], lines: true, local}];
  }
}

/**
 * The evaluation record of Shrimp's turn `result` at `history` for the analysis panel: the first stone's search as
 * the candidates (share of visits, the mover's win chance (Q + 1) / 2), both stones as the line, and the root value
 * of the first stone's search as the mover's win chance.
 */
export function record(result, history, preset) {
  const [first] = result.stones, player = playerAt(history.length);
  return {moves: result.moves, value: round((first.value + 1) / 2),
    top: first.moves.slice(0, 5).map(m => [m.cell[0], m.cell[1], round(m.share), round((m.q + 1) / 2), 0]),
    line: result.moves.map(([q, r]) => [q, r, player]), threat: [], proof: null, solved: false,
    simulations: PRESETS[preset].visits, solver_nodes: 0, ms: result.ms, engine: ID};
}

export const shrimp = {entry: {id: ID, kind: 'six', badge: 'shrimp', name: LABEL, label: LABEL, checkpoints: [], presets: PRESETS, preset: NEURAL_PRESET, analysis: true, clocks: false},
  engine: new ShrimpEngine(), record, build: 'python tools/build_web.py ort shrimp'};
