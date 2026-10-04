/** The native tactical solver (tools/tactical) as WebAssembly, for browser workers and node.
 *
 * `query(request)` takes the request object NativeTactics sends and returns the parsed result.
 * `history(history, options)` mirrors NativeTactics.history in python/tactical_proof.py without gates.
 * Both are synchronous and run one query at a time; a trap (a Rust panic) re-instantiates the module.
 */
const MAX_NODES = 10000000, MAX_TABLE_MB = 256, REQUEST_LIMIT = 64 * 1024 * 1024;
const ENOSYS = 52, EINVAL = 28;

/** Thrown by the shim's proc_exit. */
export class Exit extends Error {}

/** The WASI preview1 calls a wasm32-wasip1 library uses, over `state.memory`; stderr goes to console.error. */
export function wasi(state) {
  const view = () => new DataView(state.memory.buffer);
  const zeroSizes = (count, size) => { view().setUint32(count, 0, true); view().setUint32(size, 0, true); return 0; };
  return {
    clock_time_get(id, precision, out) {
      const ns = id === 0 ? BigInt(Date.now()) * 1000000n : id === 1 ? BigInt(Math.round(performance.now() * 1e6)) : null;
      if (ns === null) return EINVAL;
      view().setBigUint64(out, ns, true);
      return 0;
    },
    random_get(ptr, len) { crypto.getRandomValues(new Uint8Array(state.memory.buffer, ptr, len)); return 0; },
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
    environ_get: () => 0,
    environ_sizes_get: zeroSizes,
    args_get: () => 0,
    args_sizes_get: zeroSizes,
    proc_exit(code) { throw new Exit(`wasm module exited with ${code}`); },
    sched_yield: () => 0,
  };
}

/** [WebAssembly.Module, SHA-256 hex of its bytes or null] from a URL string (file: in node), Response, bytes or Module. */
export async function compile(source) {
  if (source instanceof WebAssembly.Module) return [source, null];
  let bytes;
  if (typeof source === 'string') {
    const url = new URL(source, globalThis.location?.href);
    bytes = url.protocol === 'file:'
      ? await (await import('node:fs/promises')).readFile(url)
      : await (await fetch(url)).arrayBuffer();
  } else if (source instanceof Response) bytes = await source.arrayBuffer();
  else bytes = source;
  const digest = globalThis.crypto?.subtle
    ? [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(b => b.toString(16).padStart(2, '0')).join('')
    : null;
  return [await WebAssembly.compile(bytes), digest];
}

function checkBudgets(ms, nodes, idtt_nodes, depth, attacker, table_mb) {
  const int = (x, lo, hi) => Number.isInteger(x) && lo <= x && x <= hi;
  if (!int(ms, 1, 60000) || !int(nodes, 1, MAX_NODES) || !int(idtt_nodes, 0, nodes - 1) || !int(depth, 1, 64)
      || !['mover', 'opponent', 'defender'].includes(attacker) || !int(table_mb, 0, MAX_TABLE_MB)) {
    throw new RangeError('Invalid tactical budgets');
  }
}

/** Load the solver from a URL string, Response, ArrayBuffer (or view) or WebAssembly.Module. */
export async function loadTactical(source) {
  const [module, buildHash] = await compile(source);
  const state = {};
  const shim = wasi(state);
  const imports = { wasi_snapshot_preview1: {} };
  for (const { module: space, name } of WebAssembly.Module.imports(module)) {
    (imports[space] ??= {})[name] = shim[name] ?? (() => ENOSYS);
  }
  const bind = instance => { state.exports = instance.exports; state.memory = instance.exports.memory; };
  bind(await WebAssembly.instantiate(module, imports));

  function query(request) {
    const payload = new TextEncoder().encode(JSON.stringify(request));
    const { hexo_tactical_alloc: alloc, hexo_tactical_query: run, hexo_tactical_free: free } = state.exports;
    try {
      const input = alloc(payload.length);
      new Uint8Array(state.memory.buffer, input, payload.length).set(payload);
      const output = run(input);
      free(input);
      const bytes = new Uint8Array(state.memory.buffer, output);
      const text = new TextDecoder().decode(bytes.subarray(0, bytes.indexOf(0)));
      free(output);
      return JSON.parse(text);
    } catch (error) {
      if (!(error instanceof WebAssembly.RuntimeError || error instanceof Exit)) throw error;
      bind(new WebAssembly.Instance(module, imports));
      return { status: 'UNKNOWN', native_verified: false, reason: 'native panic', moves: [] };
    }
  }

  function history(history, { nodes = 2500, ms = 1000, idtt_nodes = 0, depth = 8, attacker = 'mover',
                               certificate, root_moves, table_mb = 0, shortest = false, bounds = false, resume = false, known = [],
                               stamps = false, library = null, replay = [] } = {}) {
    checkBudgets(ms, nodes, idtt_nodes, depth, attacker, table_mb);
    if (resume && !table_mb) throw new RangeError('Solver resume requires a positive table_mb');
    const start = performance.now();
    const unknown = reason => ({
      status: 'UNKNOWN', native_verified: false, moves: [], certificate: null, proof_turns: null, shortest: false, nodes_used: 0, nodes_fresh: 0,
      attacker, build_hash: buildHash, reason, elapsed_ms: performance.now() - start, budget: nodes, gate_score: null,
      ...(bounds ? { proof_numbers: null } : {}),
    });
    const remaining = Math.floor(ms - (performance.now() - start));
    if (remaining < 1) return unknown('deadline');
    const request = { history, ms: remaining, nodes, idtt_nodes, depth, attacker, table_mb };
    if (known.length) request.known = known;
    if (stamps) request.stamps = true;
    if (library !== null) request.library = library;
    if (replay.length) request.replay = replay;
    if (certificate != null) request.certificate = certificate;
    if (root_moves != null) request.root_moves = root_moves;
    if (shortest) request.shortest = true;
    if (bounds) request.bounds = true;
    if (resume) request.resume = true;
    if (JSON.stringify(request).length > REQUEST_LIMIT) return unknown('request size limit');
    const result = { ...unknown('native error'), nodes_fresh: null, ...query(request) };
    if (performance.now() - start >= ms) Object.assign(result, {
      status: 'UNKNOWN', native_verified: false, moves: [], certificate: null, proof_turns: null, shortest: false, reason: 'deadline',
    });
    return Object.assign(result, { elapsed_ms: performance.now() - start, budget: nodes, gate_score: null });
  }

  return { query, history, build_hash: buildHash };
}
