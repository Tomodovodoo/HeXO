/* Games and evaluations belong to this browser. Model downloads use the versioned Cache API in network.mjs. */
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
  constructor(db) { this.db = db; this.memory = new Map(); }
  request(store, mode, operation) {
    if (!this.db) return Promise.resolve(operation(null).result);
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(store, mode), result = operation(tx.objectStore(store));
      tx.oncomplete = () => resolve(result.result);
      tx.onabort = tx.onerror = () => reject(tx.error || result.error);
    });
  }
  get(store, id) { return this.request(store, 'readonly', s => s ? s.get(id) : {result: this.memory.get(`${store}:${id}`)}); }
  all(store) { return this.request(store, 'readonly', s => s ? s.getAll() : {result: [...this.memory].filter(([k]) => k.startsWith(store + ':')).map(([, v]) => v)}); }
  put(store, value) { return this.request(store, 'readwrite', s => { if (s) return s.put(value); this.memory.set(`${store}:${value.id}`, structuredClone(value)); return {result: value.id}; }); }
  saveSession(snapshot, freeplay, expected, games = []) {
    const rows = [['sessions', snapshot], ...(snapshot.match ? [['matches', snapshot.match]] : []),
      ...(freeplay ? [['games', freeplay.game], ['matches', freeplay.summary]] : []), ...games.map(game => ['games', game])];
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
