/* Shrimp's network (tools/shrimp_web/export.py) under ONNX Runtime Web: WebGPU when the device has it, else
 * WebAssembly. */
import {cached, defaultThreads} from '../network.mjs';

const FEATURES = 15;

/**
 * The graph inputs for the rows of one request (search.mjs `rows`), padded to the longest row, as export.py `pack`:
 * {feats, index, mask, pair, batch, nodes}. `bias` is the manifest's bias layout.
 */
export function pack(rows, bias) {
  const b = rows.count, sizes = Array.from({length: b}, (_, i) => rows.offsets[i + 1] - rows.offsets[i]), n = Math.max(...sizes);
  const t = bias.tokens, s = t + n, m = bias.span, w = 2 * m + 1, clamp = x => x < -m ? 0 : x > m ? w - 1 : x + m;
  const feats = new Float32Array(b * n * FEATURES), index = new Int32Array(b * n * 7).fill(b * n);
  const mask = new Float32Array(b * n), pair = new Int32Array(b * s * s).fill(bias.pad);
  for (let i = 0; i < b; i++) {
    const start = rows.offsets[i], k = sizes[i], row = i * n, square = i * s * s;
    feats.set(rows.features.subarray(start * FEATURES, (start + k) * FEATURES), row * FEATURES);
    mask.fill(1, row, row + k);
    for (let j = 0; j < n; j++) index[(row + j) * 7] = row + j;
    for (let j = 0; j < k; j++) {
      for (let d = 0; d < 6; d++) {
        const neighbour = rows.neighbours[(start + j) * 6 + d];
        if (neighbour >= 0) index[(row + j) * 7 + 1 + d] = row + neighbour;
      }
    }
    for (let q = 0; q < t; q++) {
      pair.fill(bias.token_token, square + q * s, square + q * s + t);
      pair.fill(bias.token_cell, square + q * s + t, square + q * s + t + k);
    }
    for (let q = 0; q < n; q++) {
      const at = square + (t + q) * s, cq = q < k ? rows.coords[(start + q) * 2] : 0, cr = q < k ? rows.coords[(start + q) * 2 + 1] : 0;
      pair.fill(bias.cell_token, at, at + t);
      for (let key = 0; key < k; key++) {
        pair[at + t + key] = bias.lut[clamp(rows.coords[(start + key) * 2] - cq) * w + clamp(rows.coords[(start + key) * 2 + 1] - cr)];
      }
    }
  }
  return {feats, index, mask, pair, batch: b, nodes: n};
}

export class ShrimpNetwork {
  /**
   * Loads ONNX Runtime and the graph named by the manifest at `model` (relative to `base`, the web/engine URL) on
   * `device` (network.mjs probe(); WebGPU runs the fp32 graph too). `progress(fraction)` reports downloads.
   */
  static async create(base, {model = 'shrimp/model/manifest.json', device, progress = () => {}, threads = null} = {}) {
    const manifestUrl = new URL(model, base), manifest = await (await fetch(manifestUrl, {cache: 'no-cache'})).json();
    const gpu = device.provider === 'webgpu';
    const ortBase = new URL('ort/', base), ortVersion = (await (await fetch(new URL('version.json', ortBase))).json()).version;
    const runtime = gpu ? 'ort-wasm-simd-threaded.asyncify' : 'ort-wasm-simd-threaded';
    const file = Object.keys(manifest.files)[0], shares = [0, 0];
    const report = (i, f) => { shares[i] = f; progress(.3 * shares[0] + .7 * shares[1]); };
    const [ort, wasmBinary, graph] = await Promise.all([
      import(new URL(gpu ? 'ort.webgpu.min.mjs' : 'ort.wasm.min.mjs', ortBase).href),
      cached(new URL(`${runtime}.wasm`, ortBase).href, ortVersion, f => report(0, f)),
      cached(new URL(file, manifestUrl).href, manifest.files[file].sha256, f => report(1, f))]);
    ort.env.wasm.wasmPaths = {mjs: new URL(`${runtime}.mjs`, ortBase).href};
    ort.env.wasm.wasmBinary = wasmBinary;
    ort.env.wasm.numThreads = threads ?? defaultThreads({isolated: Boolean(globalThis.crossOriginIsolated),
      cores: navigator.hardwareConcurrency || 2});
    ort.env.wasm.proxy = false;
    ort.env.logLevel = 'error';
    const session = await ort.InferenceSession.create(new Uint8Array(graph), {executionProviders: [gpu ? 'webgpu' : 'wasm'],
      graphOptimizationLevel: 'all', enableCpuMemArena: true, logSeverityLevel: 3});
    return new ShrimpNetwork(ort, session, manifest, gpu ? 'webgpu' : 'wasm', ort.env.wasm.numThreads);
  }

  constructor(ort, session, manifest, provider, threads) {
    this.ort = ort;
    this.session = session;
    this.manifest = manifest;
    this.provider = provider;
    this.threads = threads;
    this.version = manifest.model_version;
    this.batches = 0;
    this.ms = 0;
  }

  /** {values, movesLeft, logits} for the rows of one request: logits of each row's legal cells, rows in order. */
  async evaluate(rows) {
    const start = performance.now(), {feats, index, mask, pair, batch, nodes} = pack(rows, this.manifest.bias), T = this.ort.Tensor;
    const size = this.manifest.bias.tokens + nodes;
    const inputs = {feats: new T('float32', feats, [batch, nodes, FEATURES]), index: new T('int32', index, [batch, nodes, 7]),
      mask: new T('float32', mask, [batch, nodes]), pair: new T('int32', pair, [batch, size, size])};
    const out = await this.session.run(inputs);
    for (const tensor of Object.values(inputs)) tensor.dispose?.();
    const policy = out.policy.data, total = rows.legal.reduce((a, b) => a + b, 0), logits = new Float32Array(total);
    for (let i = 0, at = 0; i < batch; i++) {
      logits.set(policy.subarray(i * nodes, i * nodes + rows.legal[i]), at);
      at += rows.legal[i];
    }
    const answer = {values: Float32Array.from(out.value.data), movesLeft: Float32Array.from(out.moves_left.data), logits};
    for (const tensor of Object.values(out)) tensor.dispose?.();
    this.batches++;
    this.ms += performance.now() - start;
    return answer;
  }
}
