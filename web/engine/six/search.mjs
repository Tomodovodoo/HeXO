/* Six's network search (six.wasm, built from tools/six) with a Six network under ONNX Runtime Web. A turn takes and
 * gives HTTTX cells and plays like python/six_engine.py SixEngine against sixengine: the same search, defaults and
 * radius, `go nodes N` with no time limit, and the tree kept while the game continues. */
import createModule from './six.mjs';
import {nextTask} from '../tasks.mjs';

/** Six's frame for an HTTTX cell (q, r) and back: (q + r, -r) is its own inverse. */
export const mirror = ([q, r]) => [q + r, -r];

const RADIUS = 8;

export class SixSearch {
  static async create(ort) {
    return new SixSearch(ort, await createModule());
  }

  constructor(ort, module) {
    this.ort = ort;
    this.module = module;
    this.session = null;
    this.played = null;
    this.evaluations = 0;
    this.cropCells = module.ccall('six_crop_cells', 'number', [], []);
    this.planeCount = module.ccall('six_plane_count', 'number', [], []);
    this.crop = module.ccall('six_crop', 'number', [], []);
    module.evaluateBatch = (planes, batch, out) => this.evaluate(planes, batch, out);
  }

  /** Searches with `session` (an InferenceSession of a six/networks graph) from the next turn on, on a new tree. */
  use(session) {
    this.session = session;
    this.forget();
  }

  forget() {
    this.module.ccall('six_new_game', null, [], []);
    this.played = null;
  }

  /** Six's engine/web evaluateBatch: policy logits, tanh of half the win and loss logit difference, and the score. */
  async evaluate(planesPtr, batch, outPtr) {
    const size = batch * this.planeCount * this.cropCells, stride = this.cropCells + 2;
    const planes = this.module.HEAPF32.slice(planesPtr >> 2, (planesPtr >> 2) + size);
    const input = new this.ort.Tensor('float32', planes, [batch, this.planeCount, this.crop, this.crop]);
    const result = await this.session.run({planes: input});
    input.dispose?.();
    const policy = result.policy.data, value = result.value.data, score = result.score.data;
    const heap = this.module.HEAPF32;
    for (let b = 0; b < batch; b++) {
      const at = (outPtr >> 2) + b * stride;
      heap.set(policy.subarray(b * this.cropCells, (b + 1) * this.cropCells), at);
      heap[at + this.cropCells] = Math.tanh(0.5 * (value[2 * b] - value[2 * b + 1]));
      heap[at + this.cropCells + 1] = score[b];
    }
    for (const tensor of Object.values(result)) tensor.dispose?.();
    this.evaluations += batch;
    await nextTask();
  }

  /**
   * The rest of the turn at `history` ([[q, r], ...]) within `nodes` new positions: {moves, score, nodes, stopped},
   * with `score` 1000 times the mover's value or 1000000 for a proven win and `nodes` the positions searched. `progress(nodes)` reports the search a few
   * times a second. After stop() the turn ends early with `stopped` set, and the next turn starts a new tree.
   */
  async turn(history, nodes, progress = null) {
    const continues = this.played && this.played.length <= history.length
      && this.played.every(([q, r], i) => history[i][0] === q && history[i][1] === r);
    if (!continues) this.forget();
    this.stopped = false;
    this.module.onProgress = progress && (count => progress(count));
    let reply;
    try {
      reply = await this.module.ccall('six_turn', 'string', ['string', 'number', 'number', 'number'],
        [history.map(mirror).flat().join(' '), RADIUS, 0, nodes], {async: true});
    } finally {
      this.module.onProgress = null;
    }
    if (reply.startsWith('error')) throw new Error(`Six: ${reply.slice(6)}`);
    const numbers = reply.trim().split(/\s+/).filter(Boolean).map(Number), moves = [];
    for (let i = 0; i + 1 < numbers.length; i += 2) moves.push(mirror([numbers[i], numbers[i + 1]]));
    const stopped = this.stopped;
    if (stopped) this.forget();
    else this.played = [...history.map(p => [...p]), ...moves];
    return {moves, score: this.module.ccall('six_score', 'number', [], []), nodes: this.module.ccall('six_nodes', 'number', [], []),
      stopped};
  }

  /** Ends the running turn early. */
  stop() {
    this.stopped = true;
    this.module._six_stop();
  }
}
