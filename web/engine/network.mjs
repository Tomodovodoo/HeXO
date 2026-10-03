/* The exported network (python/export_web.py) under ONNX Runtime Web: WebGPU when the device has it, else WebAssembly. */
import {encode, features, CHANNELS} from './encode.mjs';
import {cached, json, moduleUrl} from './assets.mjs';

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

/** ONNX Runtime's WebAssembly thread count: on an isolated page the cores but one, at most 8; otherwise one. */
export function defaultThreads({isolated, cores}) {
  return isolated ? Math.max(1, Math.min(8, cores - 1)) : 1;
}

const RUNTIMES = {webgpu: ['ort.webgpu.min.mjs', 'ort-wasm-simd-threaded.asyncify'], wasm: ['ort.wasm.min.mjs', 'ort-wasm-simd-threaded']};

/** ONNX Runtime Web's files for `provider` ('webgpu' or 'wasm') as ort/version.json pins them: the API module, the
 * runtime module and its wasm (assets.mjs file records, cached under the runtime's version). */
export async function runtimeFiles(provider) {
  const {data, local} = await json('ort/version.json'), [api, runtime] = RUNTIMES[provider];
  return [api, `${runtime}.mjs`, `${runtime}.wasm`].map(name => ({path: `ort/${name}`, sha256: data.files?.[name], version: data.version, local}));
}

/** ONNX Runtime Web for `provider` on `threads` WebAssembly threads (null: defaultThreads); `progress(fraction)`
 * follows the wasm download. */
export async function runtime(provider, threads, progress = () => {}) {
  const [api, glue, wasm] = await runtimeFiles(provider);
  const [ort, mjs, wasmBinary] = await Promise.all([moduleUrl(api).then(url => import(url)), moduleUrl(glue), cached(wasm, progress)]);
  ort.env.wasm.wasmPaths = {mjs};
  ort.env.wasm.wasmBinary = wasmBinary;
  ort.env.wasm.numThreads = threads ?? defaultThreads({isolated: Boolean(globalThis.crossOriginIsolated), cores: navigator.hardwareConcurrency || 2});
  ort.env.wasm.proxy = false;
  ort.env.logLevel = 'error';
  return ort;
}

/** The model graphs for `precisions` that the Bubble manifest at `model` (a path under web/engine) pins. */
export async function modelFiles(precisions, model = 'model/manifest.json') {
  const {data, local} = await json(model), folder = model.slice(0, model.lastIndexOf('/') + 1);
  return {manifest: data, files: precisions.map(precision => {
    const name = `bubble-${precision}.onnx`, {sha256, bytes} = data.files[name];
    return {path: folder + name, sha256, bytes, local};
  })};
}

export class Network {
  /**
   * Loads ONNX Runtime and the model: `model` is the manifest's path under web/engine, `device` a probe() result,
   * `progress(fraction)` reports downloads. With two candidate precisions it times a batch of 16 under each and keeps
   * fp16 only when it is at least FASTER times quicker, since fp32 reproduces the server's evaluations.
   */
  static async create({model = 'model/manifest.json', device, progress = () => {}, threads = null} = {}) {
    const {manifest, files} = await modelFiles(device.precisions, model);
    const shares = new Array(1 + files.length).fill(0), weights = [.8, ...files.map(() => .2 / files.length)];
    const report = (i, f) => { shares[i] = f; progress(shares.reduce((sum, s, j) => sum + s * weights[j], 0)); };
    const [ort, ...graphs] = await Promise.all([runtime(device.provider, threads, f => report(0, f)),
      ...files.map((file, i) => cached(file, f => report(i + 1, f)))]);
    const networks = [];
    for (const [i, precision] of device.precisions.entries()) {
      const session = await ort.InferenceSession.create(new Uint8Array(graphs[i]), {executionProviders: [device.provider],
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
