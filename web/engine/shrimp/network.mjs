/* Shrimp's network (tools/shrimp_web/export.py) under ONNX Runtime Web: WebGPU when the device has it, else
 * WebAssembly. */
import {cached, json} from '../assets.mjs';
import {runtime, session} from '../network.mjs';

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
  /** {manifest, file}: the manifest at `model` (a path under web/engine) and the graph it pins (an assets.mjs record). */
  static async files(model = 'shrimp/model/manifest.json') {
    const {data, local} = await json(model), name = Object.keys(data.files)[0];
    return {manifest: data, file: {path: model.slice(0, model.lastIndexOf('/') + 1) + name, ...data.files[name], local}};
  }

  /**
   * Starts ONNX Runtime on `device` (network.mjs probe(); WebGPU runs the fp32 graph too) and loads the graph named by
   * the manifest at `model`, reporting each stage to `stages` (stages.mjs).
   */
  static async create({model = 'shrimp/model/manifest.json', device, stages, threads = null}) {
    const {manifest, file} = await ShrimpNetwork.files(model);
    const [ort, graph] = await Promise.all([runtime(device.provider, threads, stages), cached(file, stages.file(file.path))]);
    return new ShrimpNetwork(ort, await session(ort, graph, device.provider, stages), manifest, device.provider, ort.env.wasm.numThreads);
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
