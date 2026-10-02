// Node runner for tests/test_web_engine.py: reads one JSON job from stdin, writes one JSON answer to stdout.
// {kind: 'encode', positions: [{history, actions}]} -> [{size, cells, far, ones: [flat plane indices], features: base64 float32}]
// {kind: 'search', cases: [{history, seed, tactics, steps: [{simulations, root_samples, batch_size}], batches}]}
//   replays the recorded evaluations batch by batch -> [[{action, policy, visits, completed}] per step]
// {kind: 'pv', history, certificate} -> {pv, plies} of the principal variation
// {kind: 'rows', actions, policy, values, lead} -> top rows
// {kind: 'overlay', cases: [{ev, stones}]} -> [boardOverlay(ev, stones)] from web/engine/overlay.js
// {kind: 'offline', requests: [[path, body]]} -> [[status, history or error, paused]] from an OfflineSession
// {kind: 'threads', contexts: [{isolated, cores}]} -> the WebAssembly thread count the loader would pick
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {encode, features} from '../../web/engine/encode.mjs';
import {Native, NeuralSearch, EvaluationCache} from '../../web/engine/search.mjs';
import {principalVariation, topRows} from '../../web/engine/proof.mjs';
import createModule from '../../web/engine/gumbel.mjs';
import {OfflineSession} from '../../web/engine/offline.mjs';
import {defaultThreads} from '../../web/engine/network.mjs';
import {BrowserSession} from '../../web/engine/play-session.mjs';
import {PlayStorage} from '../../web/engine/storage.mjs';
import {OpeningBook} from '../../web/engine/openings.mjs';
import {exportGame, readGame} from '../../web/engine/notation.mjs';

const job = JSON.parse(readFileSync(0, 'utf8'));
const native = new Native(await createModule());

async function search(item) {
  const batches = item.batches.slice(), tree = new NeuralSearch(native, {seed: item.seed, tactics: item.tactics, history: item.history});
  const cache = new EvaluationCache(), out = [];
  try {
    for (const step of item.steps) {
      const result = await tree.search({simulations: step.simulations, rootSamples: step.root_samples, batchSize: step.batch_size, cache,
        ...(step.choice ? {choice: step.choice} : {}),
        evaluate: async leaves => {
          const batch = batches.shift();
          if (!batch || batch.length !== leaves.length) throw new Error('Batch shape differs from the native run');
          return leaves.map((leaf, i) => {
            if (JSON.stringify(leaf.history) !== JSON.stringify(batch[i].history)) throw new Error('Leaf differs from the native run');
            return batch[i];
          });
        }});
      out.push({action: result.action, policy: result.policy, visits: result.visits, completed: result.completed});
      if (!result.action) break;
      tree.advance(result.action);
    }
  } finally {
    tree.close();
  }
  return out;
}

let answer;
if (job.kind === 'encode') {
  answer = job.positions.map(({history, actions}) => {
    const s = encode(history, actions), ones = [];
    s.planes.forEach((v, i) => { if (v) ones.push(i); });
    return {size: s.size, cells: Array.from(s.cells), far: s.far, ones, features: Buffer.from(features(s).buffer).toString('base64')};
  });
} else if (job.kind === 'search') {
  answer = [];
  for (const item of job.cases) answer.push(await search(item));
} else if (job.kind === 'pv') {
  answer = principalVariation(native, job.history, job.certificate);
} else if (job.kind === 'rows') {
  answer = topRows(job.actions, job.policy, job.values, job.lead);
} else if (job.kind === 'overlay') {
  const page = {};
  runInNewContext(readFileSync(new URL('../../web/engine/overlay.js', import.meta.url), 'utf8'), page);
  answer = job.cases.map(({ev, stones}) => page.boardOverlay(ev, stones));
} else if (job.kind === 'threads') {
  answer = job.contexts.map(defaultThreads);
} else if (job.kind === 'offline') {
  const session = new OfflineSession(native, {engine: 'browser:bubble'});
  answer = job.requests.map(([path, body]) => {
    const [status, data] = session.answer(path, body);
    return [status, status === 200 ? data.history : data.error, status === 200 && data.paused];
  });
} else if (job.kind === 'notation') {
  answer = [];
  for (const history of job.histories) {
    const formats = {};
    for (const name of ['htttx', 'rectilinear', 'tyto']) {
      const data = exportGame(history, name);
      formats[name] = {...data, history: await readGame(data.text, native)};
    }
    answer.push(formats);
  }
} else if (job.kind === 'play') {
  const s = new BrowserSession(native), data = JSON.parse(readFileSync(new URL('../../web/engine/openings.json', import.meta.url)));
  s.bookData = new OpeningBook(data);
  const entry = {id: 'browser:test', name: 'Test', kind: 'bubble', version: 'v1', checkpoints: [],
    presets: {standard: {simulations: 1, solver_nodes: 0}, quick: {simulations: 1, solver_nodes: 0}}};
  s.registerEngine(entry, {turn: async history => ({moves: job.history.slice(history.length, history.length + (history.length ? 2 : 1)), value: .5, top: [], proof: null, line: []})});
  s.analysis = s.spec({engine: entry.id, preset: 'standard'});
  answer = [];
  for (const [path, body] of job.requests) {
    const [status, data] = await s.request(path, body, 'POST');
    while (s.running || s.jobs.some(j => j.status === 'queued')) await new Promise(r => setTimeout(r, 1));
    answer.push({status, data: path === '/state' ? s.state() : data});
  }
  answer.push({backup: await s.storage.backup(), catalogue: await s.catalogue()});
} else if (job.kind === 'book') {
  const book = new OpeningBook(JSON.parse(readFileSync(new URL('../../web/engine/openings.json', import.meta.url))));
  answer = ['narrow', 'wide', 'all'].map(mode => {
    const nodes = book.select(mode, book.pool(mode).length, 42);
    nodes.forEach(n => native.game(n.moves));
    return {mode, count: nodes.length, unique: new Set(nodes.map(n => n.key)).size, off_policy: nodes.filter(n => n.off_policy).length};
  });
} else if (job.kind === 'freeplay') {
  const s = new BrowserSession(native), entry = {id: 'test', name: 'Test', kind: 'bubble', version: 'v1',
    presets: {quick: {simulations: 1, solver_nodes: 0}, standard: {simulations: 2, solver_nodes: 0}, deep: {simulations: 4, solver_nodes: 0}}};
  const calls = [];
  const adapter = {turn: async (history, budget) => { calls.push([history.length, budget.simulations]); return {moves: history.length ? [[1, 0], [2, 0]] : [[0, 0]], value: .5, top: []}; }};
  s.registerEngine(entry, adapter);
  s.seats = [s.spec({engine: 'test'}), {engine: 'human'}]; s.analysis = s.spec({engine: 'test', auto: true}); s.changed(); s.pump();
  for (let i = 0; s.running || s.jobs.some(j => j.status === 'queued'); i++) { if (i > 1000) throw Error('Analysis did not finish'); await new Promise(r => setTimeout(r, 1)); }
  await s.saving;
  const id = s.gameId, original = JSON.stringify(await s.savedReplay(id, 1));
  const study = new BrowserSession(native); study.storage = s.storage; study.id = 'study'; await study.openGame(id, 1); study.registerEngine(entry, adapter);
  await study.request('/play', {q: 1, r: 0}, 'POST');
  const reopened = new BrowserSession(native); reopened.storage = s.storage; await reopened.restore(); reopened.registerEngine(entry, adapter); await reopened.saving;
  for (let i = 0; i < 40; i++) await reopened.persist();
  const imported = new BrowserSession(native); imported.registerEngine(entry, adapter); imported.analysis = imported.spec({engine: 'test'});
  await imported.request('/import', {text: JSON.stringify((await s.request('/replay'))[1])}, 'POST');
  const label = imported.state().review[0].label;
  imported.registerEngine({...entry, version: 'v2'}, adapter);
  answer = {calls, history: reopened.history, simulations: reopened.state().evaluations[1].simulations, catalogue: await s.catalogue(), preserved: JSON.stringify(await s.savedReplay(id, 1)) === original,
    variation: (await study.savedReplay(study.gameId, 1)).history, imported_label: label, changed_version: imported.state().evaluations, restored_identity: reopened.gameId === id};
} else if (job.kind === 'resume') {
  const s = new BrowserSession(native), wait = ms => new Promise(r => setTimeout(r, ms));
  const until = async condition => { for (let i = 0; !condition(); i++) { if (i > 3000) throw Error('Move did not start'); await wait(1); } };
  s.bookData = new OpeningBook(JSON.parse(readFileSync(new URL('../../web/engine/openings.json', import.meta.url))));
  s.registerEngine({id:'test', name:'Test', kind:'bubble', version:'v1', presets:{standard:{simulations:1,solver_nodes:0}}}, {
    turn: (history, budget, options) => history.length === 1 ? Promise.resolve({moves:[[0,2],[1,2]],value:.5}) : new Promise((resolve,reject) => {
      options.signal.addEventListener('abort', () => reject(new DOMException('Cancelled','AbortError')), {once:true});
    })
  });
  s.analysis=s.spec({engine:'test',auto:true});
  await s.startMatch({players:[{engine:'test'},{engine:'test'}],games:2,clock:{mode:'game',tc:'180+2'}});
  await until(() => s.history.length === 3 && s.running?.history.length === 3);
  await s.request('/match',{action:'stop'},'POST'); await s.idle; await s.saving;
  const id=s.match.id, before={history:structuredClone(s.history),clock:s.clockNow(),timings:structuredClone(s.match.timings)};
  s.apply('/seat',{side:0,engine:'human'}); s.apply('/seat',{side:1,engine:'human'}); s.apply('/book',{enabled:true});
  await s.request('/new',{},'POST'); const bookStart=s.book.opening;
  await s.request('/import',{text:JSON.stringify({history:[[0,0]]})},'POST'); const imported=s.book.opening;
  await s.request('/match',{action:'resume',batch:id},'POST');
  await until(() => !!s.running); await s.request('/match',{action:'stop'},'POST'); await s.idle;
  answer={before,after:{history:s.history,clock:s.clockNow(),timings:s.match.timings},bookStart,imported,auto:s.analysis.auto};
} else if (job.kind === 'lifecycle') {
  const entry = {id: 'test', name: 'Test', kind: 'bubble', version: 'v1', presets: {standard: {simulations: 1, solver_nodes: 0}}};
  const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
  const until = async condition => { for (let i = 0; !condition(); i++) { if (i > 3000) throw Error('Job did not settle'); await wait(1); } };
  const s = new BrowserSession(native);
  s.registerEngine(entry, {turn: async history => ({moves: job.history.slice(history.length, history.length + (history.length ? 2 : 1)), value: .5, top: []})});
  let release, saving = false;
  const save = s.storage.saveSession.bind(s.storage);
  s.storage.saveSession = async (...args) => { if (args[3]?.some(game => game.game === 1)) { saving = true; await new Promise(resolve => { release = resolve; }); } return save(...args); };
  await s.startMatch({players: [{engine: 'test'}, {engine: 'test'}], games: 2, clock: {mode: 'fixed'}});
  await until(() => saving);
  await s.request('/match', {action: 'stop'}, 'POST'); release(); await s.idle;
  answer = {stopped_save: {paused: s.paused, active: s.match.active, completed: s.match.completed}};
  s.storage.saveSession = save;
  await s.request('/match', {action: 'resume'}, 'POST'); await until(() => !s.match.active);
  answer.resumed = {wins: s.match.wins, completed: s.match.completed};
  answer.paused_save = [];
  for (const path of ['/pause', '/cancel']) {
    const paused = new BrowserSession(native); let commit;
    paused.registerEngine(entry,{turn:async history=>({moves:job.history.slice(history.length,history.length+(history.length?2:1)),value:.5})});
    const commitSession = paused.storage.saveSession.bind(paused.storage);
    paused.storage.saveSession = async (...args) => { if(args[3]?.length) await new Promise(resolve=>{commit=resolve;}); return commitSession(...args); };
    await paused.startMatch({players:[{engine:'test'},{engine:'test'}],games:2,openings:[job.history.slice(0,-1)]});
    await until(()=>!!commit);
    await paused.request(path,path==='/pause'?{paused:true}:{id:paused.running.id},'POST'); commit(); await paused.idle;
    answer.paused_save.push({paused:paused.paused,completed:paused.match.completed,pending:paused.match.pending_game,current:paused.match.current});
  }
  const id = s.match.id;
  await s.request('/matches/delete', {batch: id}, 'POST');
  s.analysis = s.spec({engine: 'test'}); s.apply('/analyse', {ply: 1}); await until(() => !s.running && !s.jobs.some(j => j.status === 'queued'));
  answer.deleted = {current: s.match, catalogue: await s.catalogue()};
  const t = new BrowserSession(native);
  t.registerEngine(entry, {turn: (history, budget, options) => new Promise((resolve, reject) => {
    options.signal.addEventListener('abort', () => setTimeout(() => reject(new DOMException('Cancelled', 'AbortError')), 20));
  })});
  t.analysis = t.spec({engine: 'test'}); t.apply('/analyse', {ply: 0}); await until(() => !!t.running); await wait(2);
  const backup = await t.storage.backup(); backup.sessions = [{...t.snapshot(), history: [[0, 0]], records: []}];
  const [status] = await t.request('/import', {text: JSON.stringify(backup)}, 'POST');
  answer.imported = {status, history: t.history, saved: (await t.storage.get('sessions', 'live')).history};
  const u = new BrowserSession(native); let expired = false;
  u.registerEngine(entry, {turn: (history, budget, options) => new Promise((resolve, reject) => {
    options.signal.addEventListener('abort', () => { expired = true; setTimeout(() => reject(new DOMException('Cancelled', 'AbortError')), 20); });
  })});
  await u.startMatch({players: [{engine: 'test'}, {engine: 'test'}], games: 2, clock: {mode: 'move', ms: 5}});
  await until(() => expired); await u.request('/match', {action: 'stop'}, 'POST'); await u.idle;
  answer.stopped_timeout = {paused: u.paused, active: u.match.active, completed: u.match.completed};
  answer.finished_opening_status=(await s.request('/match',{players:[{engine:'test'},{engine:'test'}],games:2,openings:[job.history]},'POST'))[0];
  s.match = {active: false}; s.clock = {cross_ms: 0, circle_ms: 0}; s.apply('/undo', {});
  answer.forked_clock = {match: s.match, clock: s.clock};
  const capped = new BrowserSession(native);
  capped.registerEngine(entry, {turn: async history => ({moves: job.history.slice(history.length, history.length + (history.length ? 2 : 1)), value: .5})});
  await capped.startMatch({players:[{engine:'test'},{engine:'test'}],games:2,max_placements:3});
  await until(() => !capped.match.active);
  answer.capped = {completed:capped.match.completed,capped:capped.match.capped,results:capped.match.results};
  await capped.startMatch({players:[{engine:'test'},{engine:'test'}],games:2,max_placements:0});
  await until(() => !capped.match.active);
  answer.uncapped = {completed:capped.match.completed,capped:capped.match.capped,wins:capped.match.wins};
  const failed = new BrowserSession(native);
  failed.registerEngine(entry, {turn: async () => { throw Error('Engine failed'); }});
  await failed.startMatch({players:[{engine:'test'},{engine:'test'}],games:2,clock:{mode:'game',tc:'180+2'}});
  await until(() => !failed.running);
  const frozen = failed.clockNow(); await wait(25);
  answer.failure_clock_frozen = failed.paused && !('started' in failed.clock) && JSON.stringify(frozen) === JSON.stringify(failed.clockNow());
  const writer = new BrowserSession(native);
  writer.registerEngine(entry, {turn:async()=>null});
  await writer.request('/play',{q:0,r:0},'POST'); await writer.saving;
  const stale = new BrowserSession(native); stale.storage = writer.storage; stale.initializing = true;
  await stale.restore(); stale.registerEngine(entry,{turn:async()=>null}); stale.initializing = false;
  await writer.request('/play',{q:1,r:0},'POST'); await writer.saving;
  await stale.persist();
  const restored = new BrowserSession(native); restored.storage = writer.storage; await restored.restore();
  answer.stale_tab = {conflicted:stale.conflicted,history:restored.history,archive:(await writer.savedReplay(writer.gameId,1)).history,
    games:(await writer.catalogue()).length,identity:restored.gameId===writer.gameId,mutation_status:(await stale.request('/play',{q:0,r:1},'POST'))[0]};
  const older = new BrowserSession(native); let late;
  older.registerEngine(entry, {turn: () => new Promise(resolve => { late = resolve; })});
  await older.startMatch({players:[{engine:'test'},{engine:'test'}],games:2,openings:[job.history.slice(0,-1)]});
  await until(() => !!late); await older.saving;
  const owner = new BrowserSession(native); owner.storage = older.storage; owner.initializing = true; await owner.restore();
  owner.registerEngine(entry, {turn: (history,budget,options) => new Promise((resolve,reject) => {
    options.signal.addEventListener('abort',()=>reject(new DOMException('Cancelled','AbortError')),{once:true});
  })}); owner.initializing = false;
  await owner.request('/match',{action:'resume'},'POST'); await owner.saving;
  late({moves:job.history.slice(-1),value:1}); await older.idle; await older.saving;
  answer.stale_match = {conflicted:older.conflicted,session_completed:(await owner.storage.get('sessions','live')).match.completed,
    archive_completed:(await owner.storage.get('matches',owner.match.id)).completed,
    archived_games:(await owner.storage.all('games')).filter(g=>g.game).length};
  await owner.request('/match',{action:'stop'},'POST'); await owner.idle;
}
process.stdout.write(JSON.stringify(answer));
