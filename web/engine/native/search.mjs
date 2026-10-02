/* Native's search (src/hexo.cpp, built as native.wasm by tools/build_web.py) on placement histories, as
 * python/play.py's search child runs it: hx_search with a time budget, depth 12 and width 16. */
import createModule from './native.mjs';

export const DEPTH = 12, WIDTH = 16;
const RESULT_BYTES = 64;   // HxResult: q1, r1, q2, r2, nodes (int64), elapsed_ms (double), count, score, depth (int32)

export class NativeSearch {
  /** An instantiated native.mjs module; `options` go to it (`wasmBinary` supplies the module's bytes). */
  static async create(options = {}) {
    return new NativeSearch(await createModule(options));
  }

  constructor(module) {
    this.m = module;
    this.result = module._malloc(RESULT_BYTES);
  }

  /**
   * Native's turn at `history` ([[q, r], ...]) within `ms` milliseconds and `depth` complete turns:
   * {moves: [[q, r], ...], score, depth, nodes, elapsed_ms}. The proven winning plan it finds is kept for the
   * next call, as the server's persistent search child keeps it. Throws on an illegal history or a finished game.
   */
  turn(history, ms, depth = DEPTH) {
    const m = this.m, board = m._hx_new();
    try {
      for (const [q, r] of history) if (!m._hx_play(board, BigInt(q), BigInt(r))) throw new Error(`Illegal placement: ${q}, ${r}`);
      if (m._hx_winner(board) >= 0) throw new Error('The game has finished');
      if (!m._hx_search(board, ms, depth, WIDTH, this.result)) throw new Error('Invalid search budget');
    } finally {
      m._hx_free(board);
    }
    const v = new DataView(m.HEAPU8.buffer, this.result, RESULT_BYTES), cell = i => [0, 8].map(o => Number(v.getBigInt64(16 * i + o, true)));
    return {moves: [cell(0), cell(1)].slice(0, v.getInt32(48, true)), score: v.getInt32(52, true), depth: v.getInt32(56, true),
      nodes: Number(v.getBigUint64(32, true)), elapsed_ms: v.getFloat64(40, true)};
  }
}
