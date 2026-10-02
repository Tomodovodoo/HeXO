/* The exported network (python/export_web.py) under ONNX Runtime Web: WebGPU when the device has it, else WebAssembly. */
import {encode, features, CHANNELS} from './encode.mjs';

const CACHE = 'bubble-engine-v1';

/** The body of `url` as an ArrayBuffer, from the Cache API when it holds `url` at `version`; `progress(fraction)`. */
export async function cached(url, version, progress = () => {}) {
  const key = new URL(url, location.href);
  key.searchParams.set('v', version);
  let store = null;
  try { store = await caches.open(CACHE); } catch {}
  const hit = store && await store.match(key);
  if (hit) { progress(1); return hit.arrayBuffer(); }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const total = Number(response.headers.get('Content-Length')) || 0, parts = [];
  let received = 0;
  for (const reader = response.body.getReader(); ;) {
    const {done, value} = await reader.read();
    if (done) break;
    parts.push(value);
    received += value.length;
    if (total) progress(Math.min(1, received / total));
  }
  const body = await new Blob(parts).arrayBuffer();
  if (store) {
    for (const old of await store.keys()) if (old.url.split('?')[0] === key.href.split('?')[0]) await store.delete(old);
    await store.put(key, new Response(body.slice(0)));
  }
  progress(1);
  return body;
}

/**
 * The device to run on: {provider: 'webgpu' | 'wasm', precisions: candidate graphs, adapter}. WebGPU offers fp16
 * (shader-f16 adapters) and fp32, WebAssembly fp32. `prefer` 'wasm', 'webgpu-fp32' or 'webgpu-fp16' narrows it.
 */
export async function probe(prefer = null) {
  const wasm = {provider: 'wasm', precisions: ['fp32'], adapter: ''};
  if (prefer === 'wasm' || !globalThis.navigator?.gpu) return wasm;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return wasm;
    const f16 = adapter.features.has('shader-f16');
    const precisions = prefer === 'webgpu-fp32' || !f16 ? ['fp32'] : prefer === 'webgpu-fp16' ? ['fp16'] : ['fp32', 'fp16'];
    return {provider: 'webgpu', precisions, adapter: adapter.info?.description || adapter.info?.vendor || ''};
  } catch {
    return wasm;
  }
}

const halves = new Map();

/** IEEE half bits of each value (round to nearest even); the inputs hold only a handful of distinct values. */
function toHalf(values) {
  const out = new Uint16Array(values.length), word = new Uint32Array(1), float = new Float32Array(word.buffer);
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    let bits = halves.get(v);
    if (bits === undefined) {
      float[0] = v;
      const x = word[0], sign = (x >>> 16) & 0x8000, exponent = ((x >>> 23) & 0xff) - 112, mantissa = x & 0x7fffff;
      if (exponent <= 0) bits = sign;
      else if (exponent >= 31) bits = sign | 0x7c00;
      else {
        bits = sign | (exponent << 10) | (mantissa >>> 13);
        const rest = mantissa & 0x1fff;
        if (rest > 0x1000 || (rest === 0x1000 && (bits & 1))) bits++;
      }
      halves.set(v, bits);
    }
    out[i] = bits;
  }
  return out;
}

export class Network {
  /**
   * Loads ONNX Runtime and the model from `base` (the web/engine URL): `model` is the manifest URL, `device` a probe()
   * result, `progress(fraction)` reports downloads. With two candidate precisions it times a batch of 16 under each
   * and keeps fp16 only when it is at least FASTER times quicker, since fp32 reproduces the server's evaluations.
   */
  static async create(base, {model = 'model/manifest.json', device, progress = () => {}, threads = null} = {}) {
    const manifestUrl = new URL(model, base), manifest = await (await fetch(manifestUrl, {cache: 'no-cache'})).json();
    const gpu = device.provider === 'webgpu';
    const ortBase = new URL('ort/', base), ortVersion = (await (await fetch(new URL('version.json', ortBase))).json()).version;
    const runtime = gpu ? 'ort-wasm-simd-threaded.asyncify' : 'ort-wasm-simd-threaded';
    const shares = new Array(1 + device.precisions.length).fill(0), weights = [.8, ...device.precisions.map(() => .2 / device.precisions.length)];
    const report = (i, f) => { shares[i] = f; progress(shares.reduce((sum, s, j) => sum + s * weights[j], 0)); };
    const [ort, wasmBinary, ...graphs] = await Promise.all([
      import(new URL(gpu ? 'ort.webgpu.min.mjs' : 'ort.wasm.min.mjs', ortBase).href),
      cached(new URL(`${runtime}.wasm`, ortBase).href, ortVersion, f => report(0, f)),
      ...device.precisions.map((precision, i) => {
        const file = `bubble-${precision}.onnx`;
        return cached(new URL(file, manifestUrl).href, manifest.files[file].sha256, f => report(i + 1, f));
      })]);
    ort.env.wasm.wasmPaths = {mjs: new URL(`${runtime}.mjs`, ortBase).href};
    ort.env.wasm.wasmBinary = wasmBinary;
    ort.env.wasm.numThreads = threads ?? (globalThis.crossOriginIsolated ? Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 2) - 1)) : 1);
    ort.env.wasm.proxy = false;
    ort.env.logLevel = 'error';
    const networks = [];
    for (const [i, precision] of device.precisions.entries()) {
      const session = await ort.InferenceSession.create(new Uint8Array(graphs[i]), {executionProviders: [gpu ? 'webgpu' : 'wasm'],
        graphOptimizationLevel: 'all', enableCpuMemArena: true, logSeverityLevel: 3});
      networks.push(new Network(ort, session, precision, manifest, ort.env.wasm.numThreads));
    }
    if (networks.length === 1) return networks[0];
    const timings = {};
    for (const network of networks) timings[network.precision] = await network.time(16, 24, 5);
    const chosen = timings.fp16 * Network.FASTER <= timings.fp32 ? networks[1] : networks[0];
    for (const network of networks) if (network !== chosen) await network.session.release();
    chosen.timings = timings;
    return chosen;
  }

  static FASTER = 1.25;

  constructor(ort, session, precision, manifest, threads) {
    this.ort = ort;
    this.session = session;
    this.precision = precision;
    this.version = manifest.model_version;
    this.threads = threads;
    this.maxBatch = 64;
    this.timings = null;
  }

  /** Mean ms of `repeats` forwards of `count` inputs of side `size` (crop mask only), after two untimed ones. */
  async time(count, size, repeats) {
    const area = size * size, input = new Float32Array(count * CHANNELS * area);
    for (let i = 0; i < count; i++) input.fill(1, (i * CHANNELS + 3) * area, (i * CHANNELS + 4) * area);
    for (let i = 0; i < 2; i++) await this.forward(input, count, size);
    const start = performance.now();
    for (let i = 0; i < repeats; i++) await this.forward(input, count, size);
    return (performance.now() - start) / repeats;
  }

  /** policy [B*S*S], far [B], value [B] (Float32Arrays) for `count` stacked inputs (encode.mjs features) of side `size`. */
  async forward(input, count, size) {
    const half = this.precision === 'fp16';
    const tensor = new this.ort.Tensor(half ? 'float16' : 'float32', half ? toHalf(input) : input, [count, CHANNELS, size, size]);
    const out = await this.session.run({features: tensor});
    tensor.dispose?.();
    const result = {policy: out.policy.data, far: out.far.data, value: out.value.data};
    for (const value of Object.values(out)) value.dispose?.();
    return result;
  }

  /** Predictions [{logits, q}] for leaves [{history, actions}], as hexnet.DenseEvaluator.evaluate gives them. */
  async evaluate(leaves) {
    const samples = leaves.map(leaf => encode(leaf.history, leaf.actions)), result = new Array(leaves.length), groups = new Map();
    samples.forEach((s, i) => { if (!groups.has(s.size)) groups.set(s.size, []); groups.get(s.size).push(i); });
    for (const [size, indices] of groups) {
      const area = size * size, stride = CHANNELS * area;
      for (let start = 0; start < indices.length; start += this.maxBatch) {
        const chunk = indices.slice(start, start + this.maxBatch), input = new Float32Array(chunk.length * stride);
        chunk.forEach((i, row) => features(samples[i], input, row * stride));
        const {policy, far, value} = await this.forward(input, chunk.length, size);
        chunk.forEach((i, row) => {
          const s = samples[i], logits = new Float64Array(s.cells.length), farLogit = Math.fround(far[row] - Math.log(s.far || 1));
          for (let j = 0; j < logits.length; j++) logits[j] = s.cells[j] < 0 ? farLogit : policy[row * area + s.cells[j]];
          const q = Math.fround(Math.tanh(Math.fround(value[row] / 2)));
          if (!Number.isFinite(q) || logits.some(v => !Number.isFinite(v))) throw new Error('Nonfinite network predictions');
          result[i] = {logits, q: new Float64Array(logits.length).fill(q)};
        });
      }
    }
    return result;
  }
}
