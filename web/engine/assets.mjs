/* Engine files by their path under web/engine. A file comes from this origin when it is here, else from the public
 * site, which serves every build output with `Access-Control-Allow-Origin: *`. Downloads are kept in the Cache API
 * under this origin's URL of the file, so a local copy of web/ that lacks the build outputs downloads each file once.
 * A file from the site must match a SHA-256 from this origin's manifest or, when this origin has none, the site's. */

export const SITE = 'https://tomodovodoo.github.io/HeXO/engine/';
const BASE = new URL('./', import.meta.url).href, CACHE = 'bubble-engine-v1';

/** Hosts on which the page's `assets` query parameter is honoured: a development server on this machine. */
const LOOPBACK = /^(localhost|127\.0\.0\.1|\[::1\])$|\.localhost$/;

/**
 * The engine folder files come from when this origin lacks them. In a page: the `data-assets` attribute of a page
 * script, else the `assets` query parameter when the page is on a loopback host (so a link cannot point another
 * site's page at foreign code), else SITE. In a worker: the `assets` parameter of its script URL, which only
 * workerUrl sets.
 */
export function site() {
  const query = new URLSearchParams(globalThis.location?.search).get('assets'), page = globalThis.document;
  const chosen = page ? page.querySelector('script[data-assets]')?.dataset.assets ?? (LOOPBACK.test(location.hostname) ? query : null) : query;
  if (!chosen) return SITE;
  const href = new URL(chosen, globalThis.location?.href).href;
  return href.endsWith('/') ? href : href + '/';
}

/** The URL of worker script `path` (under web/engine), carrying a site other than SITE to the worker's site(). */
export function workerUrl(path) {
  const url = new URL(path, BASE), chosen = site();
  if (chosen !== SITE) url.searchParams.set('assets', chosen);
  return url;
}

/** The site's URL of `path`, or null when the site is this folder. */
function remote(path) {
  const there = site();
  return there === BASE ? null : new URL(path, there).href;
}

/** Thrown for a file that neither this origin nor the site has (both answer 404): the site does not publish it, so
 * only a local build provides it. Other failures (a network error, another status) are plain Errors. */
export class NotOnSite extends Error {
  constructor(path) {
    super(`${path} is neither here nor on ${site()}`);
    this.path = path;
  }
}

/** {response, local}: `path` from this origin, else (after a 404 or a network error) from the site. */
async function locate(path, init = {}) {
  const here = await fetch(new URL(path, BASE), init).catch(() => null);
  if (here?.ok) return {response: here, local: true};
  const there = remote(path);
  if (!there) throw here?.status === 404 ? new NotOnSite(path) : new Error(`${path}: ${here ? here.status : 'unreachable'}`);
  const response = await fetch(there, {...init, mode: 'cors'}).catch(error => { throw new Error(`${path}: ${error.message} (${there})`); });
  if (response.status === 404) throw new NotOnSite(path);
  if (!response.ok) throw new Error(`${path}: ${response.status} (${there})`);
  return {response, local: false};
}

/**
 * {data, local}: the JSON manifest at `path`, revalidated on each call; `local` is false when it came from the site.
 * A manifest from the site is kept in the Cache API and answers when neither origin can be reached (not when the site
 * answers 404), so installed engines start offline.
 */
export async function json(path) {
  const store = await open(), id = `${new URL(path, BASE).href}?manifest`;
  try {
    const {response, local} = await locate(path, {cache: 'no-cache'}), data = await response.json();
    if (!local) await store?.put(id, new Response(JSON.stringify(data))).catch(() => {});   // keeping it is best effort
    return {data, local};
  } catch (error) {
    const kept = !(error instanceof NotOnSite) && await store?.match(id);
    if (!kept) throw error;
    return {data: await kept.json(), local: false};
  }
}

/** SHA-256 hex of `bytes`, with CRLF read as LF when `lines` (as tools/build_web.py records build.json's files). */
export async function sha256(bytes, lines = false) {
  let data = new Uint8Array(bytes);
  if (lines) data = data.filter((b, i) => !(b === 13 && data[i + 1] === 10));
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', data)), b => b.toString(16).padStart(2, '0')).join('');
}

async function open() {
  try { return await caches.open(CACHE); } catch { return null; }
}

/** The Cache API key of `file`: this origin's URL of its path at `?v=` its SHA-256, or its version when it has none. */
function key(file) {
  const url = new URL(file.path, BASE);
  url.searchParams.set('v', file.sha256 ?? file.version ?? '');
  return url.href;
}

/**
 * The bytes of engine file `file` {path, sha256?, version?, lines?} as an ArrayBuffer: from the Cache API when it holds
 * the file's key, else fetched (this origin first, then the site), checked against `sha256` (CRLF read as LF when
 * `lines`) and stored there, replacing other versions of the path. A file from the site needs a `sha256`, and bytes
 * that do not match it throw. `progress(fraction)` follows the download.
 */
export async function cached(file, progress = () => {}) {
  const store = await open(), id = key(file), hit = store && await store.match(id);
  if (hit) { progress(1); return hit.arrayBuffer(); }
  const {response, local} = await locate(file.path, {cache: 'no-cache'});   // a Cache API miss means new bytes: revalidate
  if (!local && !file.sha256) throw new Error(`${file.path}: the site's manifest has no SHA-256 for it`);
  const total = Number(response.headers.get('Content-Length')) || file.bytes || 0, parts = [];
  let received = 0;
  for (const reader = response.body.getReader(); ;) {
    const {done, value} = await reader.read();
    if (done) break;
    parts.push(value);
    received += value.length;
    if (total) progress(Math.min(1, received / total));
  }
  const body = await new Blob(parts).arrayBuffer();
  if (file.sha256 && await sha256(body, file.lines) !== file.sha256) {
    throw new Error(`${file.path} from ${local ? 'this site' : site()} does not match its SHA-256`);
  }
  if (store) {   // keeping the bytes is best effort: a full quota still returns them
    const base = id.split('?')[0];
    try {
      for (const old of await store.keys()) if (old.url.split('?')[0] === base) await store.delete(old);
      await store.put(id, new Response(body.slice(0)));
    } catch {}
  }
  progress(1);
  return body;
}

/** Options for an Emscripten module factory that make it instantiate `bytes`: its `locateFile` answers an object URL
 * of them, since a factory built without `wasmBinary` in its incoming API fetches the wasm itself. */
export function wasmOptions(bytes) {
  const url = URL.createObjectURL(new Blob([bytes], {type: 'application/wasm'}));
  return {wasmBinary: bytes, locateFile: () => url};
}

/** A URL to import JavaScript engine file `file` from: an object URL of its bytes from cached() (this origin's copy,
 * else the site's, checked), which module workers and ONNX Runtime's thread workers can load under isolation. */
export async function moduleUrl(file, progress) {
  return URL.createObjectURL(new Blob([await cached(file, progress)], {type: 'text/javascript'}));
}

/** Whether this origin serves `path`: a HEAD request, or a GET whose body is dropped when HEAD is not supported (as
 * python/play.py answers 501). */
async function present(path) {
  const url = new URL(path, BASE), head = await fetch(url, {method: 'HEAD'}).catch(() => null);
  if (head && head.status !== 405 && head.status !== 501) return head.ok;
  const get = await fetch(url).catch(() => null);
  get?.body?.cancel();
  return Boolean(get?.ok);
}

/**
 * Where an engine's `files` are: {state: 'local'} when this origin has them all (a file marked `local` counts when
 * present() finds it), {state: 'cached'} when the Cache API holds the others,
 * {state: 'uncached'} when there is no Cache API (each load downloads them), else
 * {state: 'missing', files, bytes}: the files still to download, each with its size in `bytes` (from its manifest,
 * else the site's Content-Length), and their total.
 */
export async function status(files) {
  const here = await Promise.all(files.map(file => file.local && present(file.path)));
  const away = files.filter((file, i) => !here[i]), store = await open();
  if (!away.length) return {state: 'local'};
  if (!store) return {state: 'uncached'};
  const missing = [];
  for (const file of away) if (!await store.match(key(file))) missing.push(file);
  if (!missing.length) return {state: 'cached'};
  const sized = await Promise.all(missing.map(async file => {
    const there = remote(file.path), head = there && await fetch(there, {method: 'HEAD', mode: 'cors'}).catch(() => null);
    if (!there || head?.status === 404) throw new NotOnSite(file.path);
    return {...file, bytes: file.bytes || Number(head?.headers.get('Content-Length')) || 0};
  }));
  return {state: 'missing', files: sized, bytes: sized.reduce((sum, file) => sum + file.bytes, 0)};
}

/** Downloads `files` into the Cache API (cached()); `progress(fraction)` follows them together, weighted by `bytes`.
 * Settles once every download has, rejecting with the first failure. */
export async function install(files, progress = () => {}) {
  const shares = files.map(() => 0), weights = files.map(file => file.bytes || 1), total = weights.reduce((a, b) => a + b, 0);
  const settled = await Promise.allSettled(files.map((file, i) => cached(file, fraction => {
    shares[i] = fraction;
    progress(shares.reduce((sum, share, j) => sum + share * weights[j], 0) / total);
  })));
  const failed = settled.find(result => result.status === 'rejected');
  if (failed) throw failed.reason;
}
