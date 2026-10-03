/* Engine files by their path under web/engine. A file comes from this origin when it is here, else from the public
 * site, which serves every build output with `Access-Control-Allow-Origin: *`. Downloads are kept in the Cache API
 * under this origin's URL of the file, so a local copy of web/ that lacks the build outputs downloads each file once.
 * A file from the site must match a SHA-256 from this origin's manifest or, when this origin has none, the site's. */
import {LIMITS, localParam} from './stages.mjs';

export const SITE = 'https://tomodovodoo.github.io/HeXO/engine/';
const BASE = new URL('./', import.meta.url).href, CACHE = 'bubble-engine-v1';

/**
 * The engine folder files come from when this origin lacks them: the `data-assets` attribute of a page script, else
 * localParam('assets'), else SITE.
 */
export function site() {
  const chosen = globalThis.document?.querySelector('script[data-assets]')?.dataset.assets ?? localParam('assets');
  if (!chosen) return SITE;
  const href = new URL(chosen, globalThis.location?.href).href;
  return href.endsWith('/') ? href : href + '/';
}

/** The URL of worker script `path` (under web/engine), carrying a site other than SITE and the `stall` parameter to
 * the worker's localParam. */
export function workerUrl(path) {
  const url = new URL(path, BASE), chosen = site(), stalls = localParam('stall');
  if (chosen !== SITE) url.searchParams.set('assets', chosen);
  if (stalls) url.searchParams.set('stall', stalls);
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

/** fetch(url, init), rejecting when no response arrives within LIMITS.idle ms; the body is the reader's to bound. */
async function request(url, init = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error(`no answer in ${LIMITS.idle / 1000} s`)), LIMITS.idle);
  try {
    return await fetch(url, {...init, signal: controller.signal});
  } finally {
    clearTimeout(timer);
  }
}

/** The JSON body of `response` (for `path`), rejecting when it is not read within LIMITS.idle ms. */
function body(response, path) {
  let timer;
  const idle = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`${path}: no answer in ${LIMITS.idle / 1000} s`)), LIMITS.idle); });
  return Promise.race([response.json(), idle]).finally(() => clearTimeout(timer));
}

/** {response, local}: `path` from this origin, else (after a 404, a network error or no answer) from the site. */
async function locate(path, init = {}) {
  const here = await request(new URL(path, BASE), init).catch(() => null);
  if (here?.ok) return {response: here, local: true};
  const there = remote(path);
  if (!there) throw here?.status === 404 ? new NotOnSite(path) : new Error(`${path}: ${here ? here.status : 'unreachable'}`);
  const response = await request(there, {...init, mode: 'cors'}).catch(error => { throw new Error(`${path}: ${error.message} (${there})`); });
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
    const {response, local} = await locate(path, {cache: 'no-cache'}), data = await body(response, path);
    if (!local) await store?.put(id, new Response(JSON.stringify(data))).catch(() => {});   // keeping it is best effort
    return {data, local};
  } catch (error) {
    const kept = !(error instanceof NotOnSite) && await store?.match(id);
    if (!kept) throw error;
    return {data: await kept.json(), local: false};
  }
}

/** The `files` pins ({name: SHA-256}) of manifest `found` (a json() result for `path`). A local manifest written before
 * builds pinned their files borrows the site's pins when `same(site manifest)` says both describe one build. */
export async function pins(path, {data, local}, same) {
  const there = remote(path);
  if (data.files || !local || !there) return data.files ?? {};
  const store = await open(), id = `${new URL(path, BASE).href}?site`;   // kept so the pins also answer offline
  const response = await request(there, {cache: 'no-cache', mode: 'cors'}).catch(() => null);
  let other = response?.ok ? await body(response, path).catch(() => null) : null;
  if (other) await store?.put(id, new Response(JSON.stringify(other))).catch(() => {});
  else other = await (await store?.match(id).catch(() => null))?.json() ?? null;
  return other && same(other) ? other.files ?? {} : {};
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

/** Bytes after which an unfinished download stores what it has as a part; a later load resumes after the parts. */
const PART = 1 << 22;
class Stalled extends Error {}

/** The Cache API key of part `n` of the download of the file at key `id`. */
const partKey = (id, n) => `${id}&part=${n}`;

/** The stored parts (ArrayBuffers, in order) of an unfinished download of the file at key `id`. */
async function heldParts(store, id) {
  const parts = [];
  for (let hit; store && (hit = await store.match(partKey(id, parts.length)).catch(() => null));) parts.push(await hit.arrayBuffer());
  return parts;
}

/**
 * The rest of `response` (a body that starts at byte `start` of the file) as chunks, read so `progress(fraction,
 * received, total)` follows the file's bytes. `total` is the manifest's `bytes`, else the Content-Length, else 0
 * (unknown); a compressed response's Content-Length counts the compressed bytes, so a total the body outgrows becomes
 * 0. Every PART bytes `keep(blob)` stores the chunks since the last part. Throws Stalled when no chunk arrives for
 * LIMITS.idle ms.
 */
async function download(response, {path, start, bytes, keep}, progress) {
  const length = Number(response.headers.get('Content-Length'));
  let total = bytes || (length ? start + length : 0);
  const chunks = [], reader = response.body.getReader();
  let received = start, pending = 0, from = 0;
  for (;;) {
    let timer;
    const idle = new Promise((_, reject) => { timer = setTimeout(() => reject(new Stalled(`${path}: the download stalled`)), LIMITS.idle); });
    const {done, value} = await Promise.race([reader.read(), idle]).catch(error => { reader.cancel().catch(() => {}); throw error; })
      .finally(() => clearTimeout(timer));
    if (done) return chunks;
    chunks.push(value);
    received += value.length;
    pending += value.length;
    if (pending >= PART) {
      await keep(new Blob(chunks.slice(from)));
      from = chunks.length;
      pending = 0;
    }
    if (received > total) total = 0;
    progress(total ? received / total : 0, received, total);
  }
}

/**
 * The bytes of engine file `file` {path, sha256?, version?, lines?} as an ArrayBuffer: from the Cache API when it holds
 * the file's key, else fetched (this origin first, then the site), checked against `sha256` (CRLF read as LF when
 * `lines`) and stored there, replacing other versions of the path and the parts of unfinished downloads. A file from
 * the site needs a `sha256`, and bytes that do not match it throw. A download stores its bytes in parts of PART bytes
 * as they arrive, so a load after an interrupted one asks for the rest only (an HTTP range; a server that answers
 * with the whole file starts over, keeping the held parts until that body completes). `progress(fraction, received, total)` follows the download's bytes (`total` 0
 * when unknown), from before the request is sent; a file from the Cache API reports progress(1) alone.
 */
export async function cached(file, progress = () => {}) {
  const store = await open(), id = key(file), hit = store && await store.match(id);
  if (hit) {
    try {
      const body = await hit.arrayBuffer();
      progress(1);
      return body;
    } catch {   // an entry the browser can no longer read is fetched afresh
      await store.delete(id).catch(() => {});
    }
  }
  const base = id.split('?')[0], forget = async () => {
    for (const old of await store.keys()) if (old.url.split('?')[0] === base) await store.delete(old);
  };
  let parts = await heldParts(store, id), offset = parts.reduce((sum, part) => sum + part.byteLength, 0);
  progress(0, offset, file.bytes || 0);   // the download stage covers a request that never answers too
  let found = offset ? await locate(file.path, {cache: 'no-cache', headers: {Range: `bytes=${offset}-`}}).catch(() => null) : null;
  const replacing = parts.length > 0 && found?.response.status !== 206;   // the held parts stay until this body completes
  if (found?.response.status !== 206) {   // a new download, also when the server ignores the range
    found ??= await locate(file.path, {cache: 'no-cache'});   // a Cache API miss means new bytes: revalidate
    parts = [];
    offset = 0;
  }
  const {response, local} = found;
  if (!local && !file.sha256) throw new Error(`${file.path}: the site's manifest has no SHA-256 for it`);
  let stored = parts.length;
  const keep = replacing ? () => {} : blob => store?.put(partKey(id, stored), new Response(blob)).then(() => { stored++; }, () => {});
  const body = !response.body ? await new Blob([...parts, await response.arrayBuffer()]).arrayBuffer()
    : await download(response, {path: file.path, start: offset, bytes: file.bytes, keep}, progress).then(chunks => new Blob([...parts, ...chunks]).arrayBuffer(),
      async error => {   // a browser whose body stream fails still delivers the whole body at once
        if (error instanceof Stalled) throw error;
        progress(0, 0, 0);
        const again = await request(response.url, {cache: 'reload', ...(local ? {} : {mode: 'cors'})});
        if (!again.ok) throw new Error(`${file.path}: ${again.status}`);
        return again.arrayBuffer();
      });
  if (file.sha256 && await sha256(body, file.lines) !== file.sha256) {
    await forget().catch(() => {});
    throw new Error(`${file.path} from ${local ? 'this site' : site()} does not match its SHA-256`);
  }
  if (store) {   // keeping the bytes is best effort: a full quota still returns them
    try {
      await forget();
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
  const url = new URL(path, BASE), head = await request(url, {method: 'HEAD'}).catch(() => null);
  if (head && head.status !== 405 && head.status !== 501) return head.ok;
  const get = await request(url).catch(() => null);
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
  const missing = [];
  for (const file of away) if (!await store?.match(key(file))) missing.push(file);
  if (!missing.length) return {state: 'cached'};
  const sized = await Promise.all(missing.map(async file => {   // also tells files the site does not publish
    const there = remote(file.path), head = there && await request(there, {method: 'HEAD', mode: 'cors'}).catch(() => null);
    if (!there || head?.status === 404) throw new NotOnSite(file.path);
    return {...file, bytes: file.bytes || Number(head?.headers.get('Content-Length')) || 0};
  }));
  if (!store) return {state: 'uncached'};
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
