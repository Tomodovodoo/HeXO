/* HTTTX v2 records. Variations replace complete turns, including when only the second stone differs. */
const integer = '(?:0|-?[1-9][0-9]*)', point = `${integer},${integer}`;
const clone = x => structuredClone(x);
export const positionKey = h => h.map(p => p.join(',')).join(';');

export function parseV2(source) {
  if (source.length > 1048576) throw Error('Notation exceeds 1 MB');
  let text = source.trim(), metadata = {};
  while (!text.startsWith(';')) {
    const m = text.match(/^((?:[a-z]\s*)+)\[([^\]]*)\]\s*/);
    if (!m) throw Error('Malformed v2 metadata');
    const key = m[1].replace(/\s/g, '');
    if (key in metadata) throw Error('Duplicate metadata tag');
    metadata[key] = m[2]; text = text.slice(m[0].length);
  }
  if (metadata.version !== '2' || Object.keys(metadata)[0] !== 'version') throw Error('Expected version[2] first');
  text = text.slice(1).replace(/\s/g, '');
  let at = 0, count = 0;
  const take = pattern => { const m = text.slice(at).match(pattern); if (!m) throw Error(`Malformed v2 notation near ${text.slice(at, at + 32)}`); at += m[0].length; return m; };
  const visuals = () => { const out = []; while (text[at] === '<') out.push(take(/^<[^<>]*>/)[0]); return out; };
  function move(terminal = false, setup = false) {
    const m = take(new RegExp(`^\\[(${point}${terminal ? '|/' : ''})\\]`));
    const stone = m[1] === '/' ? null : m[1].split(',').map(Number);
    if (stone && (!stone.every(Number.isSafeInteger) || ++count > 4096)) throw Error('Too many stones or coordinate outside safe integer range');
    let info = null;
    if (!setup && text[at] === '{') info = take(new RegExp(`^\\{(?:@[0-9]+(?::[%#]${integer})?|[%#]${integer})\\}`))[0];
    return {stone, info, visuals: setup ? [] : visuals()};
  }
  let setup = null;
  if (/[xo]/.test(text[at] || '!')) {
    setup = {first: text[at++], x: [], o: [], visuals: []};
    while (text[at] === '[') setup[setup.first].push(move(false, true).stone);
    take(/^:/); const second = setup.first === 'x' ? 'o' : 'x'; take(new RegExp('^' + second));
    while (text[at] === '[') setup[second].push(move(false, true).stone);
    setup.visuals = visuals(); take(/^;/);
    const cells = [...setup.x, ...setup.o].map(p => p.join(','));
    if (new Set(cells).size !== cells.length) throw Error('Duplicate setup cell');
  }
  function line(start, depth = 0) {
    if (depth > 64) throw Error('Variations exceed 64 levels');
    const out = []; let ended = false;
    while (at < text.length && text[at] !== ')') {
      if (ended) throw Error('A terminator must end its line');
      const number = Number(take(/^(0|[1-9][0-9]*)\./)[1]);
      if (number !== start + out.length) throw Error('Expected consecutive turn numbers');
      const moves = [move(), move(true)], variations = [];
      while (text[at] === '(') { at++; variations.push(line(number, depth + 1)); take(/^\)/); }
      take(/^;/); out.push({number, moves, variations}); ended = !moves[1].stone;
    }
    if (!out.length && (!setup || depth)) throw Error('Expected a line');
    return out;
  }
  const turns = line(1);
  if (at !== text.length) throw Error('Unexpected trailing notation');
  return {metadata, setup, turns};
}

export function writeV2(record) {
  const {metadata = {}, setup, turns} = record;
  const tags = [['version', '2'], ...Object.entries(metadata).filter(([key]) => key !== 'version')].map(([k, v]) => {
    if (!/^[a-z]+$/.test(k) || typeof v !== 'string' || v.includes(']')) throw Error('Invalid metadata');
    return `${k}[${v}]`;
  }).join('');
  const stone = m => `[${m.stone ?? '/'}]${m.info || ''}${(m.visuals || []).join('')}`;
  const line = ts => ts.map(t => `${t.number}. ${t.moves.map(stone).join(' ')}${t.variations.map(v => ` (${line(v)})`).join('')};`).join('\n');
  let prefix = tags + ';\n';
  if (setup) {
    const second = setup.first === 'x' ? 'o' : 'x';
    prefix += `${setup.first}${setup[setup.first].map(p => `[${p}]`).join('')}:${second}${setup[second].map(p => `[${p}]`).join('')}${setup.visuals.join('')};\n`;
  }
  const text = prefix + line(turns);
  parseV2(text); // Export must obey the same grammar as import.
  return {text, spans: []};
}

export class Study {
  constructor(data = {}) { this.metadata = data.metadata || {version: '2'}; this.setup = data.setup || null; this.lines = data.lines || []; this.annotations = data.annotations || {}; }
  static fromRecord(record, native) {
    const study = new Study(record);
    if (record.setup?.first === 'x' && !record.setup.x.length && !record.setup.o.length && !record.turns.length) return study;
    if (record.setup && (record.setup.first !== 'o' || record.setup.o.length || JSON.stringify(record.setup.x) !== '[[0,0]]')) throw Error('This board cannot analyse custom setups yet. The notation parser supports them.');
    function visit(turns, prefix) {
      let h = clone(prefix);
      for (const t of turns) {
        const before = clone(h);
        for (const m of t.moves) {
          if (m.stone) h.push(m.stone);
          if (m.stone || m.info || m.visuals.length) {
            const key = positionKey(h), saved = study.annotations[key];
            study.annotations[key] = {stone: clone(h.at(-1)), info: m.info || saved?.info || null,
              visuals: clone(m.visuals.length ? m.visuals : saved?.visuals || [])};
          }
        }
        native.game(h);
        for (const v of t.variations) visit(v, before);
      }
      study.lines.push(h);
    }
    visit(record.turns, [[0, 0]]);
    // Main line is first; recursive traversal above discovers side lines first.
    const main = [[0, 0], ...record.turns.flatMap(t => t.moves.filter(m => m.stone).map(m => m.stone))];
    study.lines = [main, ...study.lines.filter(h => positionKey(h) !== positionKey(main))];
    return study;
  }
  capture(history) {
    if (!history.length) return;
    if (this.setup?.first === 'x' && !this.setup.x.length && !this.setup.o.length) {
      this.setup = {first: 'o', x: [[0, 0]], o: [], visuals: this.setup.visuals};
    }
    const key = positionKey(history);
    if (this.lines.some(h => positionKey(h) === key || positionKey(h).startsWith(key + ';'))) return;
    const i = this.lines.findIndex(h => key.startsWith(positionKey(h) + ';'));
    if (i >= 0) this.lines[i] = clone(history); else this.lines.push(clone(history));
  }
  record(lookup = () => null) {
    const root = [];
    for (const h of this.lines) {
      let children = root;
      for (let i = 1; i < h.length; i += 2) {
        const pair = h.slice(i, i + 2), key = positionKey(pair);
        let node = children.find(n => n.key === key);
        if (!node) { node = {key, pair, prefix: h.slice(0, i), children: []}; children.push(node); }
        children = node.children;
      }
    }
    const annotate = (p, prefix) => {
      const saved = this.annotations[positionKey(prefix)], ev = lookup(prefix);
      const m = {stone: p, info: saved?.info || null, visuals: saved?.visuals || []};
      if (ev?.value != null) {
        const x = ((prefix.length + 1) >> 1) % 2 === 0 ? ev.value : 1 - ev.value;
        const clock = m.info?.match(/@[0-9]+/)?.[0];
        m.info = `{${clock ? clock + ':' : ''}%${Math.round(100 * (2 * x - 1))}}`;
      }
      return m;
    };
    function branch(node, alternatives = []) {
      const result = [];
      while (true) {
      const moves = node.pair.map((p, j) => annotate(p, [...node.prefix, ...node.pair.slice(0, j + 1)]));
      if (moves.length === 1) moves.push({stone: null, info: null, visuals: []});
      const turn = {number: (node.prefix.length + 1) / 2, moves, variations: alternatives.map(n => branch(n))};
      result.push(turn);
      if (!node.children.length) return result;
      alternatives = node.children.slice(1); node = node.children[0];
      }
    }
    return {metadata: this.metadata, setup: this.setup || (!root.length ? {first: this.lines.length ? 'o' : 'x', x: this.lines.length ? [[0, 0]] : [], o: [], visuals: []} : null), turns: root.length ? branch(root[0], root.slice(1)) : []};
  }
}
