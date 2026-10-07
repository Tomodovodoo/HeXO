/* Games and evaluations belong to this browser. Model downloads use the versioned Cache API in network.mjs. */
import {compressFile, readGameFile} from './notation.mjs';

const OLDER = 'browser:native', DRIP = 'browser:drip';
/** `text` with Drip's id in place of the older one at its start (an engine id, or an engine key, evaluation id or version). */
const currentId = text => text === OLDER || text.startsWith(OLDER + '|') ? DRIP + text.slice(OLDER.length) : text;

/** `value` read back from this browser's saved sessions, games, matches, evaluations or engine choices, with Drip's ids
 * and name in place of the ones older saves store for it. The name and kind change only in an object that is Drip's:
 * one whose engine or id is the older id, or whose kind is the older kind. */
export function savedIds(value) {
  if (Array.isArray(value)) return value.map(savedIds);
  if (!value || Object.getPrototypeOf(value) !== Object.prototype) return value;
  const found = Object.fromEntries(Object.entries(value).map(([key, v]) => [key, typeof v === 'string' ? currentId(v) : savedIds(v)]));
  if (value.engine === OLDER || value.id === OLDER || value.kind === 'native') {
    if (found.kind === 'native') found.kind = 'drip';
    if (found.name === 'Native (browser)') found.name = 'Drip (browser)';
  }
  return found;
}

export class PlayStorage {
  static async open() {
    if (!globalThis.indexedDB) return new PlayStorage(null);
    const db = await new Promise((resolve, reject) => {
      const request = indexedDB.open('hexo-play', 1);
      request.onupgradeneeded = () => { for (const name of ['sessions', 'games', 'matches', 'evaluations', 'coverage']) request.result.createObjectStore(name, {keyPath: 'id'}); };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    return new PlayStorage(db);
  }
  constructor(db) { this.db = db; this.memory = new Map(); this.packed = new WeakMap(); this.unpacked = new Map(); this.unpacking = new Map(); this.unpackedBytes = 0; }
  /** Record versions are immutable. Blobs let snapshots share their bytes without cloning proof trees. */
  packRecord(record) {
    if (!record || typeof record !== 'object') return Promise.resolve(record);
    if (!this.packed.has(record)) this.packed.set(record, (async () => {
      const blob = new Blob([JSON.stringify(record)], {type: 'application/json'});
      if (blob.size < 16384) return record;
      const packed = {id: record.id, packed_record: await compressFile(blob), record_version: crypto.randomUUID(), record_bytes: blob.size};
      this.remember(packed, record);
      return packed;
    })());
    return this.packed.get(record);
  }
  remember(packed, record) {
    if (this.unpacked.has(packed.record_version)) return;
    while (this.unpacked.size && this.unpackedBytes + packed.record_bytes > 32 * 1024 * 1024) {
      const key = this.unpacked.keys().next().value;
      this.unpackedBytes -= this.unpacked.get(key).bytes; this.unpacked.delete(key);
    }
    if (packed.record_bytes <= 32 * 1024 * 1024) {
      this.unpacked.set(packed.record_version, {record, bytes: packed.record_bytes}); this.unpackedBytes += packed.record_bytes;
    }
  }
  async unpackRecord(record) {
    if (!(record?.packed_record instanceof Blob)) return record;
    const cached = this.unpacked.get(record.record_version);
    if (cached) return cached.record;
    if (!this.unpacking.has(record.record_version)) this.unpacking.set(record.record_version, (async () => {
      try {
        const unpacked = JSON.parse(await readGameFile(record.packed_record));
        this.packed.set(unpacked, Promise.resolve(record)); this.remember(record, unpacked);
        return unpacked;
      } finally { this.unpacking.delete(record.record_version); }
    })());
    return this.unpacking.get(record.record_version);
  }
  async records(row, encode) {
    if (!row) return row;
    const convert = r => encode ? this.packRecord(r) : this.unpackRecord(r);
    if (row.packed_record instanceof Blob) return encode ? row : convert(row);
    const out = {...row};
    if (row.records) out.records = await Promise.all(row.records.map(convert));
    if (row.position?.records) out.position = {...row.position, records: await Promise.all(row.position.records.map(convert))};
    if (row.match) out.match = await this.records(row.match, encode);
    if (row.evaluations) out.evaluations = Object.fromEntries(await Promise.all(Object.entries(row.evaluations).map(async ([key, value]) => [key, await convert(value)])));
    return out;
  }
  encode(store, row) { return store === 'evaluations' ? this.packRecord(row) : this.records(row, true); }
  async decode(store, row) { return savedIds(await (store === 'evaluations' ? this.unpackRecord(row) : this.records(row, false))); }
  request(store, mode, operation) {
    if (!this.db) return Promise.resolve(operation(null).result);
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(store, mode), result = operation(tx.objectStore(store));
      tx.oncomplete = () => resolve(result.result);
      tx.onabort = tx.onerror = () => reject(tx.error || result.error);
    });
  }
  async get(store, id) { return this.decode(store, await this.request(store, 'readonly', s => s ? s.get(id) : {result: this.memory.get(`${store}:${id}`)})); }
  /** Every row of `store`, decoded; a row stored under an older id is left out once a row holds its current id. */
  async all(store) {
    const rows = await this.request(store, 'readonly', s => s ? s.getAll() : {result: [...this.memory].filter(([k]) => k.startsWith(store + ':')).map(([, v]) => v)});
    const ids = new Set(rows.map(row => row.id));
    return Promise.all(rows.filter(row => currentId(row.id) === row.id || !ids.has(currentId(row.id))).map(row => this.decode(store, row)));
  }
  async put(store, value) {
    value = await this.encode(store, value);
    return this.request(store, 'readwrite', s => { if (s) return s.put(value); this.memory.set(`${store}:${value.id}`, structuredClone(value)); return {result: value.id}; });
  }
  async saveSession(snapshot, freeplay, expected, games = []) {
    let rows = [['sessions', snapshot], ...(snapshot.match ? [['matches', snapshot.match]] : []),
      ...(freeplay ? [['games', freeplay.game], ['matches', freeplay.summary]] : []), ...games.map(game => ['games', game])];
    rows = await Promise.all(rows.map(async ([store, row]) => [store, await this.encode(store, row)]));
    if (!this.db) {
      if ((this.memory.get(`sessions:${snapshot.id}`)?._write_token ?? null) !== expected) return Promise.resolve(false);
      for (const [store, value] of rows) this.memory.set(`${store}:${value.id}`, structuredClone(value));
      return Promise.resolve(true);
    }
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(['sessions', 'games', 'matches'], 'readwrite'), current = tx.objectStore('sessions').get(snapshot.id);
      let saved = false;
      current.onsuccess = () => {
        if ((current.result?._write_token ?? null) !== expected) return;
        for (const [store, value] of rows) tx.objectStore(store).put(value);
        saved = true;
      };
      tx.oncomplete = () => resolve(saved);
      tx.onabort = tx.onerror = () => reject(tx.error || current.error);
    });
  }
  delete(store, id) { return this.request(store, 'readwrite', s => { if (s) return s.delete(id); this.memory.delete(`${store}:${id}`); return {result: undefined}; }); }
  async backup() {
    return {format: 'hexo-browser-save', version: 1, saved_at: new Date().toISOString(), ...Object.fromEntries(await Promise.all(['sessions', 'games', 'matches', 'evaluations', 'coverage'].map(async s => [s, await this.all(s)])))};
  }
  async restore(data, native) {
    if (data.format !== 'hexo-browser-save' || data.version !== 1) throw Error('Not a HeXO browser backup');
    for (const name of ['sessions', 'games', 'matches', 'evaluations', 'coverage']) {
      if (!Array.isArray(data[name])) throw Error(`Missing ${name} in backup`);
      for (const row of data[name]) {
        if (typeof row.id !== 'string') throw Error('Invalid saved record');
        if (row.history) native.game(row.history);
      }
    }
    for (const name of ['sessions', 'games', 'matches', 'evaluations', 'coverage']) for (const row of data[name])
      await this.put(name, name === 'sessions' ? {...row, _write_token: crypto.randomUUID()} : row);
  }
}
