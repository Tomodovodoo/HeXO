// Node runner for tests/test_web_assets.py: runs web/engine/assets.mjs and the engines' file lists against a fake
// fetch (this origin and the public site, each a map of path -> body) and a fake Cache API, and writes one JSON
// object of observations to stdout.
import {createHash} from 'node:crypto';

const BASE = new URL('../../web/engine/', import.meta.url).href, SITE = 'https://tomodovodoo.github.io/HeXO/engine/';
const hash = text => createHash('sha256').update(text).digest('hex');
const requests = [], here = new Map(), site = new Map(), store = new Map();
let offline = false;

globalThis.fetch = async (input, init = {}) => {
  const url = String(input), method = init.method ?? 'GET';
  requests.push(`${method} ${url}`);
  if (url.startsWith(BASE)) {
    if (offline) throw new TypeError('Failed to fetch');
    const body = here.get(url.slice(BASE.length));
    return body === undefined ? new Response('missing', {status: 404}) : new Response(body);
  }
  const body = site.get(url.slice(SITE.length));
  if (body === undefined) return new Response('missing', {status: 404});
  return new Response(method === 'HEAD' ? null : body, {headers: {'Content-Length': String(Buffer.byteLength(body))}});
};
globalThis.caches = {open: async () => ({
  match: async key => store.has(String(key)) ? new Response(store.get(String(key))) : undefined,
  put: async (key, response) => { store.set(String(key), await response.arrayBuffer()); },
  delete: async key => store.delete(String(key.url ?? key)),
  keys: async () => [...store.keys()].map(url => ({url})),
})};

const assets = await import('../../web/engine/assets.mjs');
const text = buffer => new TextDecoder().decode(buffer);
const attempt = async run => { try { return {value: await run()}; } catch (error) { return {error: error.message}; } };
const reset = () => { requests.length = 0; here.clear(); site.clear(); store.clear(); offline = false; };
const out = {};

reset();
here.set('a.wasm', 'local bytes');
out.same_origin = {body: text(await assets.cached({path: 'a.wasm', sha256: hash('local bytes')})), requests: [...requests],
  keys: [...store.keys()]};

reset();
site.set('model/b.onnx', 'site bytes');
out.fallback = {body: text(await assets.cached({path: 'model/b.onnx', sha256: hash('site bytes')})), requests: [...requests],
  keys: [...store.keys()]};

reset();
offline = true;
site.set('b.onnx', 'site bytes');
out.unreachable = {body: text(await assets.cached({path: 'b.onnx', sha256: hash('site bytes')})), requests: [...requests]};

reset();
site.set('c.onnx', 'tampered');
out.mismatch = {...await attempt(() => assets.cached({path: 'c.onnx', sha256: hash('original')})), keys: [...store.keys()]};

reset();
site.set('d.mjs', 'export default 1;');
out.unpinned = await attempt(() => assets.cached({path: 'd.mjs'}));

reset();
here.set('e.wasm', 'line one\r\nline two');
out.lines = text(await assets.cached({path: 'e.wasm', sha256: hash('line one\nline two'), lines: true}));

reset();
site.set('f.onnx', 'first');
await assets.cached({path: 'f.onnx', sha256: hash('first')});
requests.length = 0;
out.reuse = {body: text(await assets.cached({path: 'f.onnx', sha256: hash('first')})), requests: [...requests]};
site.set('f.onnx', 'second');
await assets.cached({path: 'f.onnx', sha256: hash('second')});
out.reuse.keys = [...store.keys()];

reset();
site.set('ort/x.mjs', 'export default 7;');
const url = await assets.moduleUrl({path: 'ort/x.mjs', sha256: hash('export default 7;'), local: true});
out.module = {text: await (await import('node:buffer')).resolveObjectURL(url).text(), requests: [...requests]};

reset();
site.set('m.json', '{"version": 3}');
out.json = await assets.json('m.json');

reset();
here.set('g.wasm', 'here');
site.set('h.onnx', '12345');
site.set('i.onnx', 'abc');
const files = [{path: 'g.wasm', local: true}, {path: 'h.onnx', sha256: hash('12345'), local: false},
  {path: 'i.onnx', sha256: hash('abc'), bytes: 3, local: false}];
const before = await assets.status(files), fractions = [];
await assets.install(before.files, fraction => fractions.push(fraction));
out.status = {before: {state: before.state, bytes: before.bytes}, after: await assets.status(files), local: await assets.status(files.slice(0, 1)),
  last: fractions.at(-1)};

site.set('gone.wasm', 'gone');
out.partial = await assets.status([{path: 'g.wasm', local: true}, {path: 'gone.wasm', sha256: hash('gone'), bytes: 4, local: true}]);

const open = caches.open;
caches.open = async () => { throw new Error('no Cache API'); };
out.uncached = await assets.status(files);
caches.open = open;

// The engines' file lists on an origin without build outputs: everything comes from the site's manifests.
reset();
const pins = {};
for (const path of ['ort/ort.wasm.min.mjs', 'ort/ort-wasm-simd-threaded.mjs', 'ort/ort-wasm-simd-threaded.wasm',
  'model/bubble-fp32.onnx', 'six/networks/gen-2.onnx', 'strix/net.safetensors', 'shrimp/model/shrimp.onnx',
  'seal/engine.mjs', 'seal/engine.wasm', 'native/native.wasm', 'strix/strix.wasm', 'shrimp/shrimp.wasm']) {
  site.set(path, `bytes of ${path}`);
  pins[path] = hash(`bytes of ${path}`);
}
const name = path => path.slice(path.lastIndexOf('/') + 1);
const manifests = {
  'ort/version.json': {version: '1.30.0', files: Object.fromEntries(Object.entries(pins).filter(([p]) => p.startsWith('ort/')).map(([p, h]) => [name(p), h]))},
  'model/manifest.json': {model_version: 'm', files: {'bubble-fp32.onnx': {sha256: pins['model/bubble-fp32.onnx'], bytes: 40}}},
  'six/networks/manifest.json': {networks: [{name: 'gen-2', file: 'gen-2.onnx', sha256: pins['six/networks/gen-2.onnx'], bytes: 41}]},
  'strix/networks.json': {networks: [{id: 'net', file: 'net.safetensors', sha256: pins['strix/net.safetensors'], size: 39}]},
  'shrimp/model/manifest.json': {files: {'shrimp.onnx': {sha256: pins['shrimp/model/shrimp.onnx'], bytes: 42}}},
  'seal/manifest.json': {revision: 'r', files: {'engine.mjs': pins['seal/engine.mjs'], 'engine.wasm': pins['seal/engine.wasm']}},
  'build.json': {artefacts: Object.fromEntries(['native/native.wasm', 'strix/strix.wasm', 'shrimp/shrimp.wasm'].map(p => [p, pins[p]]))},
};
for (const [path, data] of Object.entries(manifests)) site.set(path, JSON.stringify(data));
const engines = {
  bubble: new (await import('../../web/engine/bubble.mjs')).BubbleEngine(),
  native: (await import('../../web/engine/native.mjs')).native.engine,
  six: (await import('../../web/engine/six.mjs')).six.engine,
  strix: (await import('../../web/engine/strix.mjs')).strix.engine,
  shrimp: (await import('../../web/engine/shrimp.mjs')).shrimp.engine,
  seal: (await import('../../web/engine/seal.mjs')).seal.engine,
};
out.engines = {};
for (const [id, engine] of Object.entries(engines)) {
  const listed = await engine.files();
  requests.length = 0;
  await assets.install(listed);
  out.engines[id] = {files: listed.map(f => ({path: f.path, local: f.local, pinned: f.sha256 === pins[f.path]})),
    downloads: requests.filter(r => !r.endsWith('.json')), cached: (await assets.status(listed)).state};
}

const late = new (await import('../../web/engine/strix.mjs')).StrixEngine([]);
out.strix_retry = {files: (await late.files()).map(f => f.path), checkpoints: late.checkpoints};

process.stdout.write(JSON.stringify(out));
