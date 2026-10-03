// Node runner for tests/test_web_assets.py: runs web/engine/assets.mjs and the engines' file lists against a fake
// fetch (this origin and the public site, each a map of path -> body) and a fake Cache API, and writes one JSON
// object of observations to stdout.
import {createHash} from 'node:crypto';

const BASE = new URL('../../web/engine/', import.meta.url).href, SITE = 'https://tomodovodoo.github.io/HeXO/engine/';
const hash = text => createHash('sha256').update(text).digest('hex');
const requests = [], here = new Map(), site = new Map(), store = new Map();
let offline = false, noHead = false, broken = 0;

globalThis.fetch = async (input, init = {}) => {
  const url = String(input), method = init.method ?? 'GET';
  requests.push(`${method} ${url}`);
  if (url.startsWith(BASE)) {
    if (offline) throw new TypeError('Failed to fetch');
    if (noHead && method === 'HEAD') return new Response('unsupported', {status: 501});
    const body = here.get(url.slice(BASE.length));
    return body === undefined ? new Response('missing', {status: 404}) : new Response(body);
  }
  const body = site.get(url.slice(SITE.length));
  if (body === undefined) return new Response('missing', {status: 404});
  const stream = broken-- > 0 && method === 'GET'
    ? new ReadableStream({pull: controller => controller.error(new TypeError('input stream'))}) : method === 'HEAD' ? null : body;
  return Object.defineProperty(new Response(stream, {headers: {'Content-Length': String(Buffer.byteLength(body))}}), 'url', {value: url});
};
globalThis.caches = {open: async () => ({
  match: async key => store.has(String(key)) ? new Response(store.get(String(key))) : undefined,
  put: async (key, response) => { store.set(String(key), await response.arrayBuffer()); },
  delete: async key => store.delete(String(key.url ?? key)),
  keys: async () => [...store.keys()].map(url => ({url})),
})};

const assets = await import('../../web/engine/assets.mjs');
const text = buffer => new TextDecoder().decode(buffer);
const kind = async run => { try { await run(); return 'ok'; } catch (error) { return error instanceof assets.NotOnSite ? 'not on site' : 'error'; } };
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
site.set('s.onnx', 'streamed');
broken = 1;
const fractions2 = [];
out.stream_fallback = {body: text(await assets.cached({path: 's.onnx', sha256: hash('streamed')}, f => fractions2.push(f))),
  requests: requests.filter(r => r.includes('s.onnx')).length, reset: fractions2.includes(0)};

reset();
here.set('k.wasm', 'versioned');
await assets.cached({path: 'k.wasm', version: '1.0', sha256: hash('versioned')});
out.digest_key = [...store.keys()][0].split('v=')[1] === hash('versioned');

reset();
site.set('m.json', '{"version": 3}');
out.json = await assets.json('m.json');
const online = globalThis.fetch;
globalThis.fetch = async () => { throw new TypeError('Failed to fetch'); };
out.json_offline = await assets.json('m.json');
globalThis.fetch = online;

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

noHead = true;
out.no_head = {here: (await assets.status([{path: 'g.wasm', local: true}])).state,
  partial: (await assets.status([{path: 'g.wasm', local: true}, {path: 'gone.wasm', sha256: hash('gone'), bytes: 4, local: true}])).state};
noHead = false;

reset();
here.set('ort/version.json', JSON.stringify({version: '1.30.0'}));
site.set('ort/version.json', JSON.stringify({version: '1.30.0', files: {'ort.wasm.min.mjs': 'a', 'ort-wasm-simd-threaded.mjs': 'b', 'ort-wasm-simd-threaded.wasm': 'c'}}));
out.legacy_pins = (await (await import('../../web/engine/network.mjs')).runtimeFiles('wasm')).map(f => [f.sha256, f.local]);
site.delete('ort/version.json');
out.legacy_pins_kept = (await (await import('../../web/engine/network.mjs')).runtimeFiles('wasm')).map(f => f.sha256);

const open = caches.open;
caches.open = async () => { throw new Error('no Cache API'); };
site.set('h.onnx', '12345');
out.uncached = await assets.status([{path: 'h.onnx', sha256: hash('12345'), local: false}]);
out.uncached_unpublished = await kind(() => assets.status([{path: 'nowhere.onnx', sha256: 'x', local: false}]));
caches.open = open;

// The engines' file lists on an origin without build outputs: everything comes from the site's manifests.
reset();
const pins = {};
for (const path of ['ort/ort.wasm.min.mjs', 'ort/ort-wasm-simd-threaded.mjs', 'ort/ort-wasm-simd-threaded.wasm',
  'model/bubble-fp32.onnx', 'six/networks/gen-2.onnx', 'six/networks/gen-1.onnx', 'strix/net.safetensors', 'shrimp/model/shrimp.onnx',
  'seal/engine.mjs', 'seal/engine.wasm', 'native/native.wasm', 'strix/strix.wasm', 'shrimp/shrimp.wasm']) {
  site.set(path, `bytes of ${path}`);
  pins[path] = hash(`bytes of ${path}`);
}
const name = path => path.slice(path.lastIndexOf('/') + 1);
const manifests = {
  'ort/version.json': {version: '1.30.0', files: Object.fromEntries(Object.entries(pins).filter(([p]) => p.startsWith('ort/')).map(([p, h]) => [name(p), h]))},
  'model/manifest.json': {model_version: 'm', files: {'bubble-fp32.onnx': {sha256: pins['model/bubble-fp32.onnx'], bytes: 40}}},
  'six/networks/manifest.json': {networks: [{name: 'gen-2', file: 'gen-2.onnx', sha256: pins['six/networks/gen-2.onnx'], bytes: 41},
    {name: 'gen-1', file: 'gen-1.onnx', sha256: pins['six/networks/gen-1.onnx'], bytes: 41}]},
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

out.six_chosen = (await engines.six.files(['gen-1'])).map(f => f.path).filter(p => p.startsWith('six/'));
site.delete('seal/manifest.json');
site.delete('strix/networks.json');
out.unpublished = {
  manifest: await kind(() => assets.json('nowhere/manifest.json')),
  status: await kind(() => assets.status([{path: 'nowhere.onnx', sha256: 'x', local: false}])),
  seal: await kind(() => engines.seal.files()),
  strix: await kind(async () => new (await import('../../web/engine/strix.mjs')).StrixEngine([]).files()),
};
offline = true;
const realFetch = globalThis.fetch;
globalThis.fetch = async (input, init) => { if (!String(input).startsWith(BASE)) throw new TypeError('Failed to fetch'); return realFetch(input, init); };
out.unpublished.unreachable = await kind(() => assets.json('nowhere/manifest.json'));
globalThis.fetch = realFetch;
offline = false;
site.set('strix/networks.json', JSON.stringify(manifests['strix/networks.json']));

const late = new (await import('../../web/engine/strix.mjs')).StrixEngine([]);
out.strix_retry = {files: (await late.files()).map(f => f.path), checkpoints: late.checkpoints};
out.strix_chosen = (await late.files(['net', 'unknown'])).map(f => f.path);

const sixLate = new (await import('../../web/engine/six.mjs')).SixEngine();
await sixLate.files();
out.six_retry = sixLate.checkpoints;

// The page's engine lists while Seal's files are on neither origin and the site has every other engine's.
const seat = await import('../../web/engine/seat.mjs');
await seat.survey();
out.offered = seat.offer({engines: ['browser:bubble', 'browser:seal', 'browser:strix', 'browser:six', 'six'].map(id => ({id}))}).engines.map(e => e.id);

const page = host => {
  globalThis.document = {querySelector: () => null};
  globalThis.location = {hostname: host, href: `http://${host}/`, search: '?assets=https://other.example/engine'};
  const chosen = assets.site();
  delete globalThis.document;
  delete globalThis.location;
  return chosen;
};
out.override = {public: page('tomodovodoo.github.io'), loopback: page('127.0.0.1')};

process.stdout.write(JSON.stringify(out));
