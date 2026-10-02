/* Shrimp's search (shrimp.wasm, built from tools/shrimp_web) and the turn loop of Six's Shrimp driver
 * (CixMango/Six arena/drivers/shrimp_driver.py), for browser workers and node.
 *
 * A turn takes HTTTX coordinates. Like the play server, which runs Shrimp mirrored, it hands the driver Six's frame
 * (q + r, -r); the driver moves the first stone to the origin, searches each stone of the turn with the same session
 * (the second stone reuses the first one's tree), and maps the stones back. Every turn is a new game to the driver
 * (the server sends `newgame` before each turn), so its session starts empty and its game key counts up per engine. */

const SEED_MASK = (1n << 63n) - 1n;
const FEATURES = 15;

class Exit extends Error {}

function wasi(state) {
  const view = () => new DataView(state.memory.buffer);
  const none = (count, size) => { view().setUint32(count, 0, true); view().setUint32(size, 0, true); return 0; };
  return {
    random_get(ptr, len) { crypto.getRandomValues(new Uint8Array(state.memory.buffer, ptr, len)); return 0; },
    environ_get: () => 0,
    environ_sizes_get: none,
    fd_write(fd, iovs, count, written) {
      const v = view();
      let text = '', total = 0;
      for (let i = 0; i < count; i++) {
        const ptr = v.getUint32(iovs + 8 * i, true), len = v.getUint32(iovs + 8 * i + 4, true);
        text += new TextDecoder().decode(new Uint8Array(state.memory.buffer, ptr, len));
        total += len;
      }
      console.error(text.trimEnd());
      v.setUint32(written, total, true);
      return 0;
    },
    proc_exit(code) { throw new Exit(`shrimp module exited with ${code}`); },
  };
}

/** HTTTX (q, r) as Six's frame and back: its own inverse. */
export const mirror = ([q, r]) => [q + r, -r];

/** The search module from a URL string, Response, ArrayBuffer (or view) or WebAssembly.Module. */
export async function loadModule(source) {
  if (source instanceof WebAssembly.Module) return source;
  let bytes;
  if (typeof source === 'string') {
    const url = new URL(source, globalThis.location?.href);
    bytes = url.protocol === 'file:' ? await (await import('node:fs/promises')).readFile(url) : await (await fetch(url)).arrayBuffer();
  } else if (source instanceof Response) bytes = await source.arrayBuffer();
  else bytes = source;
  return WebAssembly.compile(bytes);
}

export class Cancelled extends Error {
  constructor() { super('Cancelled'); this.name = 'AbortError'; }
}

export class ShrimpSearch {
  /** `module` from loadModule; `profile` the manifest's `search` ({search_parity_mode, cache_states, support_radius,
   * settings}). */
  static async create(module, profile) {
    const state = {}, imports = {wasi_snapshot_preview1: {}};
    const shim = wasi(state);
    for (const {module: space, name} of WebAssembly.Module.imports(module)) (imports[space] ??= {})[name] = shim[name] ?? (() => 52);
    const instance = await WebAssembly.instantiate(module, imports);
    state.memory = instance.exports.memory;
    return new ShrimpSearch(instance.exports, profile);
  }

  constructor(exports, profile) {
    this.x = exports;
    this.profile = profile;
    this.gameKey = 0n;
  }

  get memory() { return this.x.memory.buffer; }

  check(code) {
    if (code >= 0) return code;
    const message = new TextDecoder().decode(new Uint8Array(this.memory, this.x.sh_error(), this.x.sh_error_len()));
    throw new Error(`Shrimp: ${message}`);
  }

  /** Runs `body(pointer)` with `values` (Int32Array) copied into the module's memory. */
  withInts(values, body) {
    const bytes = Math.max(4, values.length * 4), pointer = this.x.sh_alloc(bytes);
    try {
      new Int32Array(this.memory, pointer, values.length).set(values);
      return body(pointer);
    } finally {
      this.x.sh_free(pointer, bytes);
    }
  }

  /** {player, over, winner} of the position after `moves` (Shrimp's frame); winner -1 while it is not won. */
  position(moves) {
    const code = this.check(this.withInts(moves.flat(), pointer => this.x.sh_position(pointer, moves.length)));
    return {player: code & 1, over: code >= 2, winner: code >= 2 ? code >> 2 : -1};
  }

  newSession() {
    const {cache_states, support_radius, search_parity_mode, settings} = this.profile;
    const handle = this.check(this.x.sh_session_new(cache_states, support_radius, search_parity_mode ? 1 : 0));
    const encoder = new TextEncoder();
    for (const [key, value] of Object.entries(settings)) {
      const name = encoder.encode(key), pointer = this.x.sh_alloc(name.length);
      try {
        new Uint8Array(this.memory, pointer, name.length).set(name);
        this.check(this.x.sh_set(handle, pointer, name.length, value));
      } finally {
        this.x.sh_free(pointer, name.length);
      }
    }
    return handle;
  }

  /** The rows the session waits for: {count, nodes, offsets, legal, features, coords, neighbours}, copied out. */
  rows(handle, count) {
    const x = this.x, nodes = this.check(x.sh_rows_nodes(handle)), copy = (Type, pointer, length) => new Type(this.memory, pointer, length).slice();
    return {count, nodes, offsets: copy(Int32Array, x.sh_rows_offsets(handle), count + 1), legal: copy(Int32Array, x.sh_rows_legal(handle), count),
      features: copy(Float32Array, x.sh_rows_features(handle), nodes * FEATURES), coords: copy(Int32Array, x.sh_rows_coords(handle), nodes * 2),
      neighbours: copy(Int32Array, x.sh_rows_neighbours(handle), nodes * 6)};
  }

  fulfill(handle, {values, movesLeft, logits}) {
    const floats = [values, movesLeft, logits].map(a => Float32Array.from(a));
    const sizes = floats.map(a => Math.max(4, a.length * 4)), pointers = sizes.map(size => this.x.sh_alloc(size));
    try {
      floats.forEach((a, i) => new Float32Array(this.memory, pointers[i], a.length).set(a));
      this.check(this.x.sh_fulfill(handle, ...pointers));
    } finally {
      pointers.forEach((pointer, i) => this.x.sh_free(pointer, sizes[i]));
    }
  }

  /** The chosen stone and the root of the finished search, in Shrimp's frame: {action, value, visits, moves: [{cell,
   * visits, q, share}]} with the chosen stone first. */
  result(handle) {
    const x = this.x, count = this.check(x.sh_result_count(handle));
    const cells = new Int32Array(this.memory, x.sh_result_moves(handle), 3 * count).slice();
    const stats = new Float32Array(this.memory, x.sh_result_stats(handle), 2 * count).slice();
    const moves = [];
    for (let i = 0; i < count; i++) moves.push({cell: [cells[3 * i], cells[3 * i + 1]], visits: cells[3 * i + 2], q: stats[2 * i], share: stats[2 * i + 1]});
    return {action: moves[0].cell, value: x.sh_result_value(handle), visits: this.check(x.sh_result_visits(handle)), moves};
  }

  /**
   * One stone of `visits` from `moves` (Shrimp's frame, the first stone at the origin) in session `handle`.
   * `evaluate(rows)` answers {values, movesLeft, logits} (logits of each row's legal cells, rows in order);
   * `stop()` true cancels; `progress(completed)` follows the visits.
   */
  async stone(handle, moves, visits, seed, {evaluate, stop = () => false, progress = () => {}}) {
    this.check(this.withInts(moves.flat(), pointer => this.x.sh_begin(handle, pointer, moves.length, visits, seed, this.gameKey)));
    for (let count = this.check(this.x.sh_step(handle)); count > 0; count = this.check(this.x.sh_step(handle))) {
      const answer = await evaluate(this.rows(handle, count));
      if (stop()) throw new Cancelled();
      this.fulfill(handle, answer);
      progress(this.check(this.x.sh_completed(handle)));
    }
    return this.result(handle);
  }

  /**
   * Shrimp's turn from `history` (HTTTX [[q, r], ...]) with `visits` per stone, as the server's Shrimp plays it:
   * {moves (HTTTX), stones: [result per stone, cells in HTTTX]}. `options` as for `stone`, with `progress(fraction)`
   * over the turn and `seed` the driver's --seed (1).
   */
  async turn(history, visits, {evaluate, stop = () => false, progress = () => {}, seed = 1} = {}) {
    const six = history.map(mirror), [oq, or] = six.length ? six[0] : [0, 0];
    const local = six.map(([q, r]) => [q - oq, r - or]);
    const back = ([q, r]) => mirror([q + oq, r + or]);
    let position = this.position(local);
    if (position.over) throw new Error('The game has finished');
    const entry = position.player, stones = [], moves = [], total = history.length === 0 ? 1 : 2;
    this.gameKey += 1n;
    const handle = this.newSession();
    try {
      for (let ply = local.length; !position.over && position.player === entry; ply++) {
        const done = stones.length;
        const stoneSeed = (BigInt(seed) * 5003n + BigInt(ply)) & SEED_MASK;
        const result = await this.stone(handle, local, visits, BigInt.asUintN(64, stoneSeed), {evaluate, stop,
          progress: completed => progress(Math.min(1, (done + completed / visits) / total))});
        local.push(result.action);
        moves.push(back(result.action));
        stones.push({...result, action: back(result.action), moves: result.moves.map(m => ({...m, cell: back(m.cell)}))});
        position = this.position(local);
      }
    } finally {
      this.x.sh_session_free(handle);
    }
    return {moves, stones};
  }
}
