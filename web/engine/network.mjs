/* The exported network (python/export_web.py) under ONNX Runtime Web: WebGPU when the device has it, else WebAssembly. */
import {encode, features, CHANNELS} from './encode.mjs';
import {cached, json, moduleUrl, pins} from './assets.mjs';
import {LIMITS, stall} from './stages.mjs';
import {nextTask} from './tasks.mjs';

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
      networks.push(new Network(ort, await session(ort, graphs[i], device.provider, stages), precision, manifest, ort.env.wasm.numThreads,
        {graph: graphs[i], provider: device.provider}));
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

  constructor(ort, session, precision, manifest, threads, {graph = null, provider = null} = {}) {
    this.ort = ort;
    this.session = session;
    this.precision = precision;
    this.version = manifest.model_version;
    this.threads = threads;
    this.maxBatch = 64;
    this.timings = null;
    this.graph = graph;
    this.provider = provider;
    this.captures = null;
    this.closed = false;
    this.closing = null;
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

  /** Opt-in native batches use bounded static GPU sessions. Session creation
   * and draining remain in the caller's clock; no implicit CPU fallback. */
  async forwardCaptured(input, count, size, stop = () => false) {
    if (this.closed) throw new Error('Native captures are closed');
    if (this.provider !== 'webgpu' || !this.graph) throw new Error('Native captures require a WebGPU model');
    this.captures ??= new Captures(this);
    return this.captures.forward(input, count, size, stop);
  }

  captureStats() { return this.captures?.stats() ?? {rows: 0, physical: 0, forwards: 0, creates: 0, evictions: 0, setup_ms: 0,
    entries: 0, cells: 0, owned_gpu_bytes: 0, staging_bytes: 0, unreleased_outputs: 0}; }

  async close() {
    if (this.closing) return this.closing;
    this.closed = true;
    this.closing = (async () => {
      let error;
      try { await this.captures?.close(); } catch (failed) { error = failed; }
      try {
        if (this.session) { await this.session.release(); this.session = null; }
      } catch (failed) { error ??= failed; }
      if (error) throw error;
    })();
    try { await this.closing; } finally { this.closing = null; }
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
  async evaluateNative(batch, {stop = () => batch.owner.done(), capture = false} = {}) {
    for (let group = 0; group < batch.groups.length; group++) {
      const {size, rows} = batch.groups[group];
      const limit = Math.min(this.maxBatch, Math.max(1, Math.floor(64 * 32 * 32 / (size * size))));
      for (let start = 0; start < rows; start += limit) {
        if (stop()) return false;
        const count = Math.min(limit, rows - start), input = batch.features(group, start, count);
        if (stop()) return false;
        const prediction = capture ? await this.forwardCaptured(input, count, size, stop) : await this.forward(input, count, size);
        if (!prediction) return false;
        batch.inference ??= {rows: 0, physical: 0, forwards: 0};
        batch.inference.rows += count; batch.inference.physical += prediction.physical_rows ?? count; batch.inference.forwards++;
        batch.decode(group, start, count, prediction);
        // WASM forwards may resolve only through microtasks. Give worker
        // cancellation messages a task boundary before admitting more work.
        await nextTask();
        if (stop()) return false;
      }
    }
    return true;
  }
}

/** One network/model owns this cache. Bounds cover retained capture count and
 * canvas-row capacity, not the runtime allocator or driver residency. GPU
 * buffers and sessions are released only after their forward/download drains. */
class Captures {
  constructor(network) {
    this.network = network;
    this.entries = new Map();
    this.outputs = new Set();
    this.cells = 0;
    this.tail = Promise.resolve();
    this.closed = false;
    this.counts = {rows: 0, physical: 0, forwards: 0, creates: 0, evictions: 0, setup_ms: 0};
  }
  stats() {
    let gpu = 0, staging = 0;
    for (const entry of this.entries.values()) {
      gpu += (entry.input?.size ?? 0) + (entry.download?.size ?? 0); staging += entry.values.byteLength;
    }
    return {...this.counts, entries: this.entries.size, cells: this.cells, owned_gpu_bytes: gpu, staging_bytes: staging,
      unreleased_outputs: this.outputs.size};
  }
  drainOutputs() {
    let error;
    for (const tensor of this.outputs) {
      try { tensor.dispose(); this.outputs.delete(tensor); } catch (failed) { error ??= failed; }
    }
    if (error) throw error;
  }
  async release(entry) {
    let error;
    for (const [name, method] of [['session', 'release'], ['tensor', 'dispose'], ['input', 'destroy'], ['download', 'destroy']]) {
      try {
        if (entry[name]) { await entry[name][method](); entry[name] = null; }
      } catch (failed) { error ??= failed; }
    }
    if (error) throw error;
  }
  async entry(count, size) {
    const maximum = Math.floor(64 * 32 * 32 / (size * size));
    if (!(count > 0 && count <= maximum)) throw new Error('Captured batch exceeds canvas capacity');
    const quantum = size >= 40 ? 8 : 16;
    const rows = Math.min(maximum, count <= 16 ? 2 ** Math.ceil(Math.log2(count)) : quantum * Math.ceil(count / quantum));
    const key = `${size}/${rows}`;
    if (this.entries.has(key)) {
      const entry = this.entries.get(key); this.entries.delete(key); this.entries.set(key, entry); return entry;
    }
    const device = await this.network.ort.env.webgpu.device;
    if (!device) throw new Error('WebGPU device is unavailable');
    const limited = device.limits.maxBufferSize <= LIMITED_BYTES, limit = limited ? 65536 : 524288, slots = limited ? 8 : 24;
    const cells = rows * size * size;
    while (this.entries.size && (this.entries.size >= slots || this.cells + cells > limit)) {
      const [old, entry] = this.entries.entries().next().value;
      try { await this.release(entry); }
      catch (error) { this.closed = this.network.closed = true; throw error; }
      this.entries.delete(old); this.cells -= entry.cells; this.counts.evictions++;
    }
    const half = this.network.precision === 'fp16', length = rows * (size * size + 2), features = rows * CHANNELS * size * size;
    const entry = {size, rows, cells, device, session: null, input: null, tensor: null, download: null,
      values: half ? new Uint16Array(features) : new Float32Array(features)};
    this.entries.set(key, entry); this.cells += cells;
    const align = bytes => 16 * Math.ceil(bytes / 16), start = performance.now();
    try {
      entry.input = device.createBuffer({size: align(entry.values.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
      entry.tensor = this.network.ort.Tensor.fromGpuBuffer(entry.input, {dataType: half ? 'float16' : 'float32', dims: [rows, CHANNELS, size, size]});
      entry.download = device.createBuffer({size: align(length * 4), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
      entry.session = await this.network.ort.InferenceSession.create(new Uint8Array(this.network.graph), {
        executionProviders: [{name: 'webgpu', device}], freeDimensionOverrides: {batch: rows, size},
        preferredOutputLocation: 'gpu-buffer', enableGraphCapture: true, graphOptimizationLevel: 'all', logSeverityLevel: 3});
      this.counts.creates++;
      return entry;
    } catch (error) {
      try { await this.release(entry); }
      catch (failed) { this.closed = this.network.closed = true; throw new AggregateError([error, failed], 'Capture creation and cleanup failed'); }
      this.entries.delete(key); this.cells -= cells;
      throw error;
    } finally { this.counts.setup_ms += performance.now() - start; }
  }
  async forward(input, count, size, stop) {
    // Concurrent requests cannot overwrite a captured input or map its download
    // twice. Close joins this same queue before releasing sessions.
    if (this.closed) throw new Error('Native captures are closed');
    const previous = this.tail; let release;
    this.tail = new Promise(resolve => { release = resolve; });
    await previous;
    let outputs = null, entry = null, mapped = false;
    try {
      if (this.closed) throw new Error('Native captures are closed');
      this.drainOutputs();
      if (stop()) return null;
      entry = await this.entry(count, size);
      if (stop()) return null;
      const values = this.network.precision === 'fp16' ? toHalf(input) : input, stride = CHANNELS * size * size;
      entry.values.set(values);
      // Padded rows copy a real input so every mask and reduction remains valid.
      for (let row = count; row < entry.rows; row++) entry.values.set(values.subarray(0, stride), row * stride);
      if (stop()) return null;
      entry.device.queue.writeBuffer(entry.input, 0, entry.values);
      this.counts.rows += count; this.counts.physical += entry.rows; this.counts.forwards++;
      outputs = await entry.session.run({features: entry.tensor});
      const command = entry.device.createCommandEncoder(); let offset = 0;
      for (const name of ['policy', 'far', 'value']) {
        if (outputs[name].type !== 'float32') throw new Error('Captured output must be float32');
        const length = name === 'policy' ? entry.rows * size * size : entry.rows;
        command.copyBufferToBuffer(outputs[name].gpuBuffer, 0, entry.download, offset, length * 4); offset += length * 4;
      }
      entry.device.queue.submit([command.finish()]);
      await entry.download.mapAsync(GPUMapMode.READ); mapped = true;
      const data = new Float32Array(entry.download.getMappedRange()).slice(0, offset / 4);
      return {physical_rows: entry.rows, policy: data.slice(0, count * size * size), far: data.slice(entry.rows * size * size, entry.rows * size * size + count),
        value: data.slice(entry.rows * (size * size + 1), entry.rows * (size * size + 1) + count)};
    } finally {
      // A rejected run may already have submitted device work. Its fence must
      // settle before output tensors can release buffers referenced by commands.
      let error;
      try { if (entry) await entry.device.queue.onSubmittedWorkDone(); } catch (failed) { error = failed; }
      try { if (mapped) entry.download.unmap(); } catch (failed) { error ??= failed; }
      if (outputs) for (const tensor of Object.values(outputs)) {
        try { tensor.dispose(); } catch (failed) { this.outputs.add(tensor); error ??= failed; }
      }
      release();
      if (error) throw error;
    }
  }
  async close() {
    this.closed = true;
    await this.tail;
    let error;
    try { this.drainOutputs(); } catch (failed) { error = failed; }
    for (const [key, entry] of this.entries) {
      try { await this.release(entry); this.entries.delete(key); this.cells -= entry.cells; }
      catch (failed) { error ??= failed; }
    }
    if (error) throw error;
  }
}
