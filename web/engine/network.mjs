/* The exported network (python/export_web.py) under ONNX Runtime Web: WebGPU when the device has it, else WebAssembly. */
import {encode, features, CHANNELS} from './encode.mjs';
import {cached, json, moduleUrl, pins} from './assets.mjs';
import {LIMITS, stall} from './stages.mjs';

/** An adapter whose largest buffer is at most the WebGPU default (256 MiB), as phone GPUs report, is limited. */
const LIMITED_BYTES = 2 ** 28;
const probes = new Map(), TIMED_OUT = Symbol('timed out');

/**
 * The device to run on, decided once per context and `prefer`: {provider: 'webgpu' | 'wasm', precisions: candidate
 * graphs, adapter, fallback?}. WebGPU offers fp16 (shader-f16 adapters) and fp32, WebAssembly fp32; a limited adapter
 * offers fp16 alone when it has shader-f16, so it never holds both sessions. `prefer` 'wasm', 'webgpu-fp32' or
 * 'webgpu-fp16' narrows it. An adapter request that rejects or takes over LIMITS.adapter ms gives WebAssembly with
 * `fallback`, the reason ('timed out' or 'failed (message)').
 */
export function probe(prefer = null) {
  if (!probes.has(prefer)) probes.set(prefer, detect(prefer));
  return probes.get(prefer);
}

async function detect(prefer) {
  const wasm = {provider: 'wasm', precisions: ['fp32'], adapter: ''};
  if (prefer === 'wasm' || !globalThis.navigator?.gpu) return wasm;
  let adapter, timer;
  try {
    adapter = await Promise.race([stall('adapter', () => navigator.gpu.requestAdapter()), new Promise((_, reject) => {
      timer = setTimeout(() => reject(TIMED_OUT), LIMITS.adapter);
    })]);
  } catch (error) {
    const reason = error === TIMED_OUT ? 'timed out' : `failed (${error?.message || error})`;
    console.warn(`WebGPU adapter request ${reason}`);
    return {...wasm, fallback: reason};
  } finally {
    clearTimeout(timer);
  }
  if (!adapter) return wasm;
  const f16 = adapter.features.has('shader-f16'), limited = (adapter.limits?.maxBufferSize ?? Infinity) <= LIMITED_BYTES;
  const precisions = prefer === 'webgpu-fp32' || !f16 ? ['fp32'] : prefer === 'webgpu-fp16' || limited ? ['fp16'] : ['fp32', 'fp16'];
  return {provider: 'webgpu', precisions, adapter: adapter.info?.description || adapter.info?.vendor || '', limited};
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

/**
 * ONNX Runtime's WebAssembly thread count: on an isolated page the cores but one, at most 8, and at most 2 under 4 GB
 * of device memory or 4 under 8 GB (`memory` is navigator.deviceMemory, absent outside Chromium); otherwise one. Each
 * thread is a worker on the runtime's shared memory, which a phone with many cores and little RAM cannot afford.
 */
export function defaultThreads({isolated, cores, memory}) {
  const cap = !memory ? 8 : memory < 4 ? 2 : memory < 8 ? 4 : 8;
  return isolated ? Math.max(1, Math.min(cap, cores - 1)) : 1;
}

/** defaultThreads for this context. */
export function deviceThreads() {
  return defaultThreads({isolated: Boolean(globalThis.crossOriginIsolated), cores: globalThis.navigator?.hardwareConcurrency || 2,
    memory: globalThis.navigator?.deviceMemory});
}

const RUNTIMES = {webgpu: ['ort.webgpu.min.mjs', 'ort-wasm-simd-threaded.asyncify'], wasm: ['ort.wasm.min.mjs', 'ort-wasm-simd-threaded']};

/** ONNX Runtime Web's files for `provider` ('webgpu' or 'wasm') as ort/version.json pins them: the API module, the
 * runtime module and its wasm (assets.mjs file records, cached under the runtime's version, with their sizes when the
 * manifest lists them). */
export async function runtimeFiles(provider) {
  const found = await json('ort/version.json'), {data, local} = found, [api, runtime] = RUNTIMES[provider];
  const files = await pins('ort/version.json', found, other => other.version === data.version);
  return [api, `${runtime}.mjs`, `${runtime}.wasm`].map(name => ({path: `ort/${name}`, sha256: files[name], bytes: data.sizes?.[name],
    version: data.version, local}));
}

/** The runtime files a load on `provider` may read: runtimeFiles(provider), plus the WebAssembly runtime that a
 * failed WebGPU start falls back to unless `prefer` fixed the device. */
export async function loadFiles(provider, prefer) {
  const files = await runtimeFiles(provider);
  return provider === 'webgpu' && !prefer ? [...files, ...await runtimeFiles('wasm')] : files;
}

/** A one-node graph (Identity on one float) whose session starts the runtime before any network is loaded. */
const START = Uint8Array.from(atob('CAgSADo7ChAKAXgSAXkiCElkZW50aXR5EgVzdGFydFoPCgF4EgoKCAgBEgQKAggBYg8KAXkSCgoICAESBAoCCAFCBAoAEBE='), c => c.charCodeAt(0));

/**
 * ONNX Runtime Web for `provider`, started (its wasm compiled, its thread workers or WebGPU device up) on `threads`
 * WebAssembly threads (null: deviceThreads()). `stages` (stages.mjs) follows the manifest and the downloads, then the
 * compile stage.
 */
export async function runtime(provider, threads, stages) {
  stages.enter('download');
  const [api, glue, wasm] = await runtimeFiles(provider);
  const [ort, mjs, wasmBinary] = await Promise.all([moduleUrl(api, stages.file(api.path)).then(url => import(url)),
    moduleUrl(glue, stages.file(glue.path)), cached(wasm, stages.file(wasm.path))]);
  ort.env.wasm.wasmPaths = {mjs};
  ort.env.wasm.wasmBinary = wasmBinary;
  ort.env.wasm.numThreads = threads ?? deviceThreads();
  ort.env.wasm.proxy = false;
  ort.env.logLevel = 'error';
  stages.enter('compile', provider);
  const session = await stall('compile', () => ort.InferenceSession.create(START, {executionProviders: [provider], logSeverityLevel: 3}));
  await session.release();
  return ort;
}

/** A session of `graph` (bytes) on `provider`, in the session stage of `stages`. */
export function session(ort, graph, provider, stages) {
  stages.enter('session', provider);
  return stall('session', () => ort.InferenceSession.create(new Uint8Array(graph), {executionProviders: [provider],
    graphOptimizationLevel: 'all', enableCpuMemArena: true, logSeverityLevel: 3}));
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
   * Loads the model whose manifest is `model` (a path under web/engine) on `device` (a probe() result) with `ort` (a
   * runtime() for its provider), reporting to `stages`. With two candidate precisions it times a batch of 16 under each
   * and keeps fp16 only when it is at least FASTER times quicker, since fp32 reproduces the server's evaluations.
   */
  static async create({model = 'model/manifest.json', device, ort, stages}) {
    stages.enter('download');
    const {manifest, files} = await modelFiles(device.precisions, model);
    const graphs = await Promise.all(files.map(file => cached(file, stages.file(file.path))));
    const networks = [];
    for (const [i, precision] of device.precisions.entries()) {
      networks.push(new Network(ort, await session(ort, graphs[i], device.provider, stages), precision, manifest, ort.env.wasm.numThreads));
    }
    if (networks.length === 1) return networks[0];
    stages.enter('timing', device.provider);
    const timings = {};
    for (const network of networks) timings[network.precision] = await stall('timing', () => network.time(16, 24, 5));
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

  /** Compiled graph owner batches already own their crop/context mappings.
   * Keep JavaScript work at submission granularity and bound feature staging
   * on large canvases. The ONNX forward still returns its three output arrays. */
  async evaluateNative(batch, {stop = () => batch.owner.done()} = {}) {
    for (let group = 0; group < batch.groups.length; group++) {
      const {size, rows} = batch.groups[group];
      const limit = Math.min(this.maxBatch, Math.max(1, Math.floor(64 * 32 * 32 / (size * size))));
      for (let start = 0; start < rows; start += limit) {
        if (stop()) return false;
        const count = Math.min(limit, rows - start), input = batch.features(group, start, count);
        if (stop()) return false;
        const prediction = await this.forward(input, count, size);
        batch.decode(group, start, count, prediction);
        if (stop()) return false;
      }
    }
    return true;
  }
}
