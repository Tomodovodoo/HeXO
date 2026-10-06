// Node runner for tests/test_web_engine.py Loading: the browser engines' loading stages, watchdogs, thread cap and
// fallback chain, with scripted workers, fake WebGPU adapters and a fake ONNX Runtime. Writes one JSON object of
// observations to stdout.
import {createHash} from 'node:crypto';

Object.defineProperty(globalThis, 'navigator', {value: {hardwareConcurrency: 8, deviceMemory: 8}, configurable: true});
globalThis.crossOriginIsolated = true;
console.info = () => {};   // engines log their device; stdout carries the answer
const {LIMITS, Stages, errorReport, stageText} = await import('../../web/engine/stages.mjs');
Object.assign(LIMITS, {probe: 60, download: 60, compile: 60, session: 60, timing: 60, warmup: 60, adapter: 60, idle: 60});
const {EngineWorker, notices} = await import('../../web/engine/engine-worker.mjs');
const {defaultThreads} = await import('../../web/engine/network.mjs');
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
const out = {};

/** Workers that play `script`: script.load(options, post) for a load, script.call(message, post, options) for a call. */
let script = null;
const starts = [];
globalThis.Worker = class {
  constructor() { this.terminated = false; }
  postMessage(message) {
    const post = data => setTimeout(() => { if (!this.terminated) this.onmessage?.({data}); }, 1);
    if (message.type === 'load') { this.options = message.options; starts.push({...message.options}); script.load(message.options, post); }
    else if (message.type === 'cancel') script.cancel?.(message, post, this.options);
    else script.call(message, post, this.options);
  }
  terminate() { this.terminated = true; }
};

/** A load that reports the probe, a 10 MB download, compile, session, timing and warmup; `fate(device, stage)` says
 * whether a stage goes on, 'hang's (reports nothing more) or 'fail's. `probe` is the probe's device for `options`. */
const loader = (fate, probe = options => ({provider: options.prefer === 'wasm' ? 'wasm' : 'webgpu'})) => ({
  load(options, post) {
    const device = probe(options), stages = new Stages(post);
    (async () => {
      stages.enter('probe');
      stages.probed(device);
      for (const received of [5e6, 1e7]) {
        stages.file('ort/runtime.wasm')(received / 1e7, received, 1e7);
        if (fate({...options, ...device}, 'download') === 'fail') throw Object.assign(new Error('network error'), {stage: stages.current});
      }
      for (const name of ['compile', 'session', 'timing', 'warmup']) {
        stages.enter(name, device.provider);
        const next = fate({...options, ...device}, name);
        if (next === 'hang') return;
        if (next === 'fail') throw Object.assign(new Error('device lost'), {stage: stages.current});
        await wait(1);
      }
      post({type: 'ready', device: {provider: device.provider, threads: options.threads ?? 7}});
    })().catch(error => post(errorReport(error)));
  },
  call(message, post, options) {
    if (message.type === 'use') {
      const stages = new Stages(post, message.id), provider = options.prefer === 'wasm' ? 'wasm' : 'webgpu';
      stages.enter('session', provider);
      if (fate({...options, provider}, 'use') === 'hang') return;
      post({type: 'result', id: message.id, result: provider});
    } else {
      post({type: 'progress', id: message.id, fraction: .5});
      post({type: 'result', id: message.id, result: 'done'});
    }
  },
});

/** Loads an engine under `chosen` and records its stages (the words the page shows), notices, worker starts and device. */
async function observe(chosen, options = {}, act = null) {
  script = chosen;
  starts.length = 0;
  const said = [], heard = [], listen = ({detail}) => heard.push({text: detail.text, cpu: detail.cpu});
  notices.addEventListener('notice', listen);
  const engine = new EngineWorker('worker.mjs', 'Bubble', options), tags = [];
  const note = stage => {
    const words = stageText(stage);
    if (said.at(-1) !== words) said.push(words);
    if (stage?.provider && tags.at(-1) !== stage.provider) tags.push(stage.provider);
  };
  let device, error = null, result = null;
  try {
    device = await engine.load((fraction, stage) => note(stage));
    if (act) result = await act(engine, note);
  } catch (e) {
    error = e.message;
  }
  notices.removeEventListener('notice', listen);
  engine.close();
  return {stages: said, notices: heard, starts: starts.map(s => ({prefer: s.prefer, threads: s.threads})), tags,
    device: device ?? null, error, result};
}

const gpu = (stage, how = 'hang') => (device, name) => device.provider === 'webgpu' && name === stage ? how : 'go';
out.clean = await observe(loader(() => 'go'));
out.gpu_session_hangs = await observe(loader(gpu('session')));
out.gpu_compile_hangs = await observe(loader(gpu('compile')));
out.gpu_timing_hangs = await observe(loader(gpu('timing')));
out.gpu_warmup_hangs = await observe(loader(gpu('warmup')));
out.gpu_session_fails = await observe(loader(gpu('session', 'fail')));
out.threads_stall = await observe(loader((device, name) => device.threads === null && name === 'compile' ? 'hang' : 'go'), {prefer: 'wasm'});
out.everything_hangs = await observe(loader((device, name) => name === 'session' ? 'hang' : 'go'));
out.download_fails = await observe(loader((device, name) => name === 'download' ? 'fail' : 'go'));
out.probe_fell_back = await observe(loader(() => 'go', options => ({provider: 'wasm', ...(options.prefer ? {} : {fallback: 'timed out'})})));
out.silent = await observe({load() {}, call() {}});
out.fixed_device = await observe(loader(gpu('session')), {prefer: 'webgpu-fp32', threads: 2});
out.fixed_gpu = await observe(loader(gpu('session')), {prefer: 'webgpu-fp16'});
const said = [];
out.call_restarts = await observe(loader(gpu('use')), {}, async (engine, note) => engine.call({type: 'use'},
  {progress: (fraction, live, stage) => { note(stage); said.push(stageText(stage)); }}));
out.call_restarts.call_stages = [...new Set(said)];
script = loader(() => 'go');
out.call_first = await new EngineWorker('worker.mjs', 'Bubble').call({type: 'evaluate'}).catch(error => error.message);
out.calls_overlap = await observe(loader(gpu('use')), {}, async engine => {
  const use = engine.call({type: 'use'});
  await wait(5);
  return [await engine.call({type: 'evaluate'}), await use];
});

// Native, Seal and Strix have no fallback, but a load that goes silent in a stage still ends with an error naming it.
script = {load(options, post) { const stages = new Stages(post); stages.enter('download'); stages.enter('compile'); }, call() {}};
const silent = async engine => engine.load().then(() => 'loaded', error => error.message);
out.single = {
  native: await silent((await import('../../web/engine/native.mjs')).native.engine),
  seal: await silent((await import('../../web/engine/seal.mjs')).seal.engine),
  strix: await silent(new (await import('../../web/engine/strix.mjs')).StrixEngine([{id: 'net', path: 'strix/net.safetensors', sha256: 'x', size: 1}])),
};
script = {load(options, post) {   // a compile that keeps reporting for longer than its limit
  const stages = new Stages(post);
  stages.enter('compile');
  (async () => { for (let i = 0; i < 6; i++) { await wait(25); stages.send(.9); } post({type: 'ready', info: {}}); })();
}, call() {}};
out.single.reporting = await silent(new (await import('../../web/engine/strix.mjs')).StrixEngine([{id: 'net', path: 'strix/net.safetensors', sha256: 'x', size: 1}]));

out.threads = [{isolated: true, cores: 8}, {isolated: true, cores: 8, memory: 2}, {isolated: true, cores: 8, memory: 4},
  {isolated: true, cores: 8, memory: 8}, {isolated: true, cores: 2, memory: 2}, {isolated: false, cores: 8, memory: 8},
  {isolated: true, cores: 24, memory: 0.5}].map(defaultThreads);

out.words = [null, {name: 'probe'}, {name: 'download', received: 12.4e6, total: 27e6}, {name: 'download', received: 2.5e6, total: 4.6e6},
  {name: 'download', received: 17.1e6, total: 0}, {name: 'download', received: 0, total: 0}, {name: 'compile'}, {name: 'session', provider: 'webgpu'}, {name: 'session', provider: 'wasm'}, {name: 'timing'}, {name: 'warmup'}].map(stageText);

// The probe with fake adapters: a fresh network.mjs each time, since probe() keeps its answer.
let fresh = 0;
async function probeWith(gpuApi, prefer = null) {
  Object.defineProperty(globalThis, 'navigator', {value: {hardwareConcurrency: 8, deviceMemory: 8, gpu: gpuApi}, configurable: true});
  const network = await import(`../../web/engine/network.mjs?probe=${fresh++}`);
  const device = await network.probe(prefer);
  return {provider: device.provider, precisions: device.precisions, fallback: device.fallback ?? null, same: device === await network.probe(prefer)};
}
const adapter = (f16, maxBufferSize) => ({features: new Set(f16 ? ['shader-f16'] : []), limits: {maxBufferSize}, info: {vendor: 'fake'}});
out.probe = {
  hangs: await probeWith({requestAdapter: () => new Promise(() => {})}),
  rejects: await probeWith({requestAdapter: async () => { throw new Error('blocked'); }}),
  none: await probeWith({requestAdapter: async () => null}),
  desktop: await probeWith({requestAdapter: async () => adapter(true, 2 ** 32)}),
  phone: await probeWith({requestAdapter: async () => adapter(true, 2 ** 28)}),
  phone_without_f16: await probeWith({requestAdapter: async () => adapter(false, 2 ** 28)}),
  phone_asked_fp32: await probeWith({requestAdapter: async () => adapter(true, 2 ** 28)}, 'webgpu-fp32'),
};
Object.defineProperty(globalThis, 'navigator', {value: {hardwareConcurrency: 8, deviceMemory: 8}, configurable: true});

// Network.create with a fake ONNX Runtime and a fake origin that serves the model's manifest and graphs.
const bubbleGraph = Buffer.from('CAo6hwEKHAoIZmVhdHVyZXMSBnBvbGljeSIISWRlbnRpdHkSB2ZpeHR1cmVaLwoIZmVhdHVyZXMSIwohCAESHQoHEgViYXRjaAoCCBQKBhIEc2l6ZQoGEgRzaXplYi0KBnBvbGljeRIjCiEIARIdCgcSBWJhdGNoCgIIFAoGEgRzaXplCgYSBHNpemVCBAoAEBE=', 'base64');
const graphs = {'bubble-fp32.onnx': bubbleGraph, 'bubble-fp16.onnx': bubbleGraph, 'shrimp.onnx': 'shrimp graph'};
const hash = text => createHash('sha256').update(text).digest('hex');
const pinned = names => Object.fromEntries(names.map(name => [name, {sha256: hash(graphs[name]), bytes: graphs[name].length}]));
const manifests = {'model/manifest.json': {model_version: 'm', files: pinned(['bubble-fp32.onnx', 'bubble-fp16.onnx'])},
  'shrimp/model/manifest.json': {model_version: 's', files: pinned(['shrimp.onnx'])}};
globalThis.fetch = async input => {
  const url = String(input), name = url.split('/').pop(), found = Object.keys(manifests).find(path => url.endsWith(path));
  if (found) return new Response(JSON.stringify(manifests[found]));
  if (!graphs[name]) return new Response('missing', {status: 404});
  return new Response(graphs[name], {headers: {'Content-Length': String(graphs[name].length)}});
};
globalThis.caches = {open: async () => { throw new Error('no Cache API'); }};
const fakeOrt = (fp16Ms, fp32Ms) => ({
  env: {wasm: {numThreads: 4}},
  Tensor: class { constructor(type, data, dims) { Object.assign(this, {type, data, dims}); } },
  InferenceSession: {create: async () => ({
    async run({features}) {
      const [count, , size] = features.dims, ms = features.type === 'float16' ? fp16Ms : fp32Ms, start = performance.now();
      while (performance.now() - start < ms);
      return {policy: {data: new Float32Array(count * size * size)}, far: {data: new Float32Array(count)}, value: {data: new Float32Array(count)}};
    },
    release: async () => {},
  })},
});
const {Network} = await import('../../web/engine/network.mjs');
async function created(precisions, ort) {
  const seen = [], stages = new Stages(({stage}) => { if (seen.at(-1) !== stage.name) seen.push(stage.name); });
  const network = await Network.create({device: {provider: 'webgpu', precisions}, ort, stages});
  return {stages: seen, precision: network.precision};
}
out.network = {both: await created(['fp32', 'fp16'], fakeOrt(0, 20)), fp16_only: await created(['fp16'], fakeOrt(0, 0))};
const {ShrimpNetwork} = await import('../../web/engine/shrimp/network.mjs');
const shrimpStages = [];
await ShrimpNetwork.create({device: {provider: 'wasm'}, ort: fakeOrt(0, 0),
  stages: new Stages(({stage}) => { if (shrimpStages.at(-1) !== stage.name) shrimpStages.push(stage.name); })});
out.network.shrimp = shrimpStages;
const lost = new Stages(() => {});
out.network.missing_manifest = await lost.run(() => Network.create({model: 'model/missing/manifest.json', device: {provider: 'wasm', precisions: ['fp32']},
  ort: fakeOrt(0, 0), stages: lost})).then(() => null, error => error.stage?.name ?? null);

// A browser session job whose engine never finishes loading gives way when its seat changes engine, and an engine
// that left the GPU moves its choices to lightning.
const {Native} = await import('../../web/engine/search.mjs');
const {default: createModule} = await import('../../web/engine/gumbel.mjs');
const {BrowserSession} = await import('../../web/engine/play-session.mjs');
const session = new BrowserSession(new Native(await createModule()));
const entry = id => ({id, name: id, kind: 'bubble', version: 'v1', checkpoints: [], presets: {lightning: {simulations: 1, solver_nodes: 0}, standard: {simulations: 2, solver_nodes: 0}}, clocks: true});
const loads = [];
session.registerEngine(entry('stuck'), {ready: report => { report(.5, {name: 'session', provider: 'webgpu'}); loads.push('stuck'); return new Promise(() => {}); }, turn: async () => ({moves: [[0, 0]], value: .5, top: []})});
session.registerEngine(entry('quick'), {ready: async () => { loads.push('quick'); }, turn: async () => ({moves: [[0, 0]], value: .5, top: []})});
await session.request('/seat', {side: 0, engine: 'stuck', preset: 'standard'}, 'POST');
await wait(20);
const stuck = session.state().jobs.map(j => ({kind: j.kind, status: j.status, stage: j.stage}));
await session.request('/seat', {side: 0, engine: 'quick', preset: 'standard'}, 'POST');
for (let i = 0; i < 500 && !session.history.length; i++) await wait(2);
out.session = {stuck, loads, history: session.history};
await session.request('/seat', {side: 1, engine: 'quick', preset: 'standard'}, 'POST');
session.paused = true;
session.lighten('quick');
out.lighten = {seats: session.seats.map(s => s.preset ?? null), preset: session.entries.get('quick').preset};
session.cancelJobs();

// An abandoned search must release the session even when its worker ignores Cancel.
const stalled = new BrowserSession(session.native), engine = new EngineWorker('worker.mjs', 'Bubble');
const history = [[0,0],[0,1],[1,0],[2,-1],[-1,-5]];
for (let k=0;k<12;k++) {
  history.push([-3-4*k,4+4*k],[-4-4*k,5+4*k]);
  if(k<11)history.push([-2-4*k,3+4*k],[-4-4*k,4+4*k]);
}
let started, firstId, requests=0, finished;
const began = new Promise(resolve=>{started=resolve;}), completed = new Promise(resolve=>{finished=resolve;});
script = {
  load(options,post) { post({type:'ready',device:{provider:'webgpu'}}); },
  call(message,post) {
    if (++requests===1) { firstId=message.id; started(); return; }
    post({type:'result',id:message.id,result:{value:.75,moves:[],top:[],solved:true}});
  },
  cancel(message,post) { post({type:'progress',id:message.id,fraction:.2,stage:{name:'session',provider:'webgpu'}}); }
};
stalled.registerEngine(entry('cancel'), {ready:()=>engine.load(),turn:(history,budget,options)=>engine.call({type:'turn',history},options)});
stalled.history=history; stalled.analysis=stalled.spec({engine:'cancel',preset:'standard',auto:false});
stalled.onchange=state=>{if(state.evaluations[history.length]?.value===.75)finished();};
stalled.enqueue('analyse',history.slice(0,26),stalled.analysis); stalled.pump(); await began;
const abandoned = engine.worker;
stalled.enqueue('analyse',history,stalled.analysis); stalled.pump();
const blocked = {queued:stalled.state().jobs.map(j=>({ply:j.ply,status:j.status})),aborted:stalled.running.controller.signal.aborted};
const recovered = await Promise.race([completed.then(()=>true),wait(4000).then(()=>false)]);
if(recovered)await stalled.idle;
abandoned.onmessage({data:{type:'result',id:firstId,result:{value:.1,moves:[],top:[],solved:true}}});
out.cancel_stalled={blocked,recovered,requests,terminated:abandoned.terminated,sameDevice:engine.device?.provider==='webgpu',
  value:stalled.lookup(history)?.value,oldSaved:!!stalled.lookup(history.slice(0,26)),running:!!stalled.running};
stalled.cancelJobs();engine.close();

// A prompt acknowledgement or a result racing cancellation keeps the healthy worker and its graph.
out.cancel_ack=[];
for(const type of ['cancelled','result']) {
  let entered;
  const ready = new Promise(resolve=>{entered=resolve;}), control = new AbortController();
  script={load(options,post){post({type:'ready',device:{provider:'webgpu'}});},call(){entered();},
    cancel(message,post){post({type,id:message.id,graph:'kept',result:{graph_id:'kept'}});}};
  const engine=new EngineWorker('worker.mjs','Bubble'), result=engine.call({type:'turn'},{signal:control.signal})
    .then(()=>({resolved:true}),error=>({name:error.name,graph:error.graph}));
  await ready;const worker=engine.worker;control.abort();const found=await result;
  await wait(2100);
  out.cancel_ack.push({...found,kept:engine.worker===worker&&!worker.terminated,waits:engine.waits.size});
  engine.close();
}

process.stdout.write(JSON.stringify(out));
process.exit(0);
