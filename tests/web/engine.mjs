// Node runner for tests/test_web_engine.py: reads one JSON job from stdin, writes one JSON answer to stdout.
// {kind: 'encode', positions: [{history, actions}]} -> [{size, cells, far, ones: [flat plane indices], features: base64 float32}]
// {kind: 'search', cases: [{history, seed, tactics, q_range_floor, root_noise, limit, steps: [{simulations, root_samples, batch_size, marks, at}], batches}]}
//   replays the recorded evaluations batch by batch, a step's `marks` ([q, r, winner, distance]) settled before its search;
//   with `limit` the tree is a GameGraph and a step's `at` (a history) moves its root before the search, else each step
//   advances by the previous step's action -> [[{action, policy, visits, completed, proven, unmarked}] per step]
// {kind: 'game', simulations} -> turns of seats on GameGraphs lines with a ranked network: the root visits each stone's
//   search started from on the first and second turn of one line, back at the first position and on a new line, and
//   the lines kept
// {kind: 'revisit', history} -> A -> B -> A on one GameGraph with ranked priors: A searched, its chosen B searched as
//   a root where every position below B is lost for A's mover, A read again and searched again ->
//   {first, back, again: {action, visits, policy, completed_q} of A}
// {kind: 'pv', history, certificate} -> {pv, plies} of the principal variation
// {kind: 'rows', actions, policy, values, lead} -> top rows
// {kind: 'glimpse', history, simulations, nodes: 0} -> live snapshots and results of two worker turns on one graph
// {kind: 'analysis-bar', cases: [{history, value, live}]} -> the page's X/O labels and bar transform
// {kind: 'overlay', cases: [{ev, stones}]} -> [boardOverlay(ev, stones)] from web/engine/overlay.js
// {kind: 'offline', requests: [[path, body]]} -> [[status, history or error, paused]] from an OfflineSession
// {kind: 'threads', contexts: [{isolated, cores}]} -> the WebAssembly thread count the loader would pick
// {kind: 'table', records: [[history, record]], queries: [history], result, lost, exact, mover} -> {known, edges} per query
//   from a proof.mjs Proofs and `settled` of `result`, `lost` and `exact` with the edges of the first query
// {kind: 'retired-checkpoints'} -> the checkpoints a BrowserSession gives Strix and Bubble seats naming networks it does not offer
// {kind: 'proofs', history, ply, found} -> a BrowserSession whose engine proves `found` at `ply` and searches other positions
//   with the stones the proof table it is sent proves marked (see `searched`): the evaluations at ply - 1 after analysis,
//   undo, another preset and a reload, and `given`, the turn proof.mjs answered gives at ply - 1 from `found`'s table
import {readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {runInNewContext} from 'node:vm';
import {encode, features} from '../../web/engine/encode.mjs';
import {Native, NeuralSearch, EvaluationCache, GameGraph, GameGraphs, NativeOwner, NativeProofs} from '../../web/engine/search.mjs';
import {principalVariation, topRows, Proofs, answered, settled, proofTurns, proofKey, proofEvidence, winningLine, proven} from '../../web/engine/proof.mjs';
import createModule from '../../web/engine/gumbel.mjs';
import {OfflineSession} from '../../web/engine/offline.mjs';
import {defaultThreads, Network} from '../../web/engine/network.mjs';
import {Stages, errorReport, stall} from '../../web/engine/stages.mjs';
import {BrowserSession, review} from '../../web/engine/play-session.mjs';
import {PlayStorage} from '../../web/engine/storage.mjs';
import {OpeningBook} from '../../web/engine/openings.mjs';
import {exportGame, readGame} from '../../web/engine/notation.mjs';
import {loadTactical, proofAnswer} from '../../web/engine/tactical.mjs';

const job = JSON.parse(readFileSync(0, 'utf8'));
const native = new Native(await createModule());

/** One search of `tree` (a NeuralSearch, a GameGraph or one of its views) through the per-leaf hxg_* ABI, batched as
 * python/neural_search.py NeuralSearch.search batches it, so the wasm build can be checked against the native library.
 * `evaluate([{history, actions}])` resolves to [{logits, q}]; cached leaves are answered from `cache`. Resolves to
 * the tree's result(choice). */
async function drive(tree, {simulations = 128, rootSamples = null, batchSize = 16, evaluate, cache = new EvaluationCache(),
  version = 'web', choice = 'policy'}) {
  const m = tree.m;
  native.checked(m._hxg_begin(tree.ptr, simulations, rootSamples ?? Math.max(2, Math.floor(Math.sqrt(simulations)))));
  try {
    for (let active = native.game(tree.history).winner < 0; active;) {
      const pending = [];
      let idle = false;
      while (pending.length < batchSize && !idle) {
        if (m._hxg_done(tree.ptr)) { active = false; break; }
        const [id, leaf] = tree.request();
        if (id === 0) idle = true;
        else if (id > 0) {
          const key = cache.key(leaf.history, version), cached = cache.get(key);
          if (cached === undefined) pending.push({id, leaf, key});
          else tree.fulfill(id, leaf.actions, cached);
        }
      }
      if (m._hxg_done(tree.ptr)) active = false;
      if (!active) break;
      if (!pending.length) throw new Error('Native scheduler stalled without pending evaluations');
      const groups = new Map();
      for (const item of pending) groups.set(item.key, [...groups.get(item.key) ?? [], item]);
      const unique = [...groups.values()], predictions = await evaluate(unique.map(items => items[0].leaf));
      unique.forEach((items, i) => {
        for (const item of items) tree.fulfill(item.id, item.leaf.actions, predictions[i]);
        cache.put(items[0].key, {logits: Float64Array.from(predictions[i].logits), q: Float64Array.from(predictions[i].q)});
      });
    }
  } finally { m._hxg_cancel(tree.ptr); }
  return tree.result(choice);
}

async function search(item) {
  const options = {seed: item.seed, tactics: item.tactics, qRangeFloor: item.q_range_floor ?? 0, rootNoise: item.root_noise ?? 0,
    history: item.history};
  const batches = item.batches.slice(), tree = item.limit == null ? new NeuralSearch(native, options) : new GameGraph(native, {...options, limit: item.limit});
  const cache = new EvaluationCache(), out = [];
  const evaluate = async leaves => {
    if (item.uniform) return leaves.map(({actions}) => ({logits: actions.map(() => 0), q: actions.map(() => 0)}));
    const batch = batches.shift();
    if (!batch || batch.length !== leaves.length) throw new Error('Batch shape differs from the native run');
    return leaves.map((leaf, i) => {
      if (JSON.stringify(leaf.history) !== JSON.stringify(batch[i].history)) throw new Error('Leaf differs from the native run');
      return batch[i];
    });
  };
  try {
    for (const step of item.steps) {
      if (step.at) tree.at(step.at);
      const marks = new Map((step.marks || []).map(([q, r, winner, distance]) => [`${q},${r}`, {action: [q, r], winner, distance}]));
      const unmarked = await tree.settle(marks, {cache, evaluate});
      const result = await drive(tree, {simulations: step.simulations, rootSamples: step.root_samples, batchSize: step.batch_size, cache,
        ...(step.choice ? {choice: step.choice} : {}), evaluate});
      out.push({action: result.action, policy: result.policy, visits: result.visits, completed: result.completed, proven: result.proven,
        proof_plies: result.proof_plies, native_distance: tree.m._hxg_distance(tree.ptr), unmarked: unmarked.size});
      if (!result.action) break;
      if (item.limit == null) tree.advance(result.action);
    }
  } finally {
    tree.close();
  }
  return out;
}

/** The turn at `history` as the worker searches it, on a network with fixed priors and 8 simulations a stone: each
 * root's stones `table` (a Proofs) proves are marked exact before its search, and the first stone's root gives the
 * value, top rows and proof; a proof's line is the turn's stones, then the table's line from there. Throws when the
 * tree does not take a mark. */
async function searched(history, table) {
  const evaluate = async leaves => leaves.map(({actions}) => ({logits: actions.map((_, i) => -2 * i), q: actions.map(() => 0)}));
  const {player, remaining} = native.game(history), current = history.map(p => [...p]), moves = [];
  let first = null;
  while (native.game(current).player === player && native.game(current).winner < 0) {
    const tree = new NeuralSearch(native, {seed: 1740, tactics: true, history: current});
    try {
      if ((await tree.settle(table.edges(current), {evaluate})).size) throw new Error('The tree did not take a proven stone');
      const result = await drive(tree, {simulations: 8, rootSamples: 16, evaluate});
      first ??= result;
      moves.push(result.action);
      current.push(result.action);
    } finally {
      tree.close();
    }
  }
  const won = first.proven > 0, after = table.known(current)?.pv ?? [];
  return {moves, value: won ? 1 : .5, top: topRows(first.actions, first.policy, first.values, first.action), threat: [],
    proof: won ? {winner: player, turns: proofTurns(first.proof_plies, remaining, true), plies: first.proof_plies} : null,
    pv: won ? [...moves.map(([q, r], i) => [q, r, player, i + 1]), ...after.map(([q, r, side, ply]) => [q, r, side, ply + moves.length])] : []};
}

let answer;
if (job.kind === 'encode') {
  answer = job.positions.map(({history, actions}) => {
    const s = encode(history, actions), ones = [];
    s.planes.forEach((v, i) => { if (v) ones.push(i); });
    return {size: s.size, cells: Array.from(s.cells), far: s.far, ones, features: Buffer.from(features(s).buffer).toString('base64')};
  });
} else if (job.kind === 'native-features') {
  answer = [];
  for (const history of job.histories) {
    const graph = new GameGraph(native, {history, roundBarrier: true}), owner = new NativeOwner(graph, {work: 1, ms: 0, views: 1});
    let batch;
    try {
      owner.step(); batch = owner.take();
      if (!batch || batch.count !== 1 || batch.groups.length !== 1) throw new Error('Expected one root encoding');
      const input = batch.features(0, 0, 1), sample = encode(history, native.legal(history), true);
      answer.push({size: batch.groups[0].size, features: Buffer.from(input.buffer).toString('base64'),
        reference: Buffer.from(features(sample).buffer).toString('base64'), far: sample.far});
    } finally { batch?.close(); owner.close(); graph.close(); }
  }
} else if (job.kind === 'native-owner') {
  const graph = new GameGraph(native, {history: job.history, roundBarrier: true}), owner = new NativeOwner(graph, {
    work: job.work ?? (job.ms ? 0 : 256), ...(job.defaultClock ? {} : {ms: job.ms ?? 0}), views: job.views ?? 8, quantum: 32});
  const network = new Network(null, null, 'fp32', {model_version: 'test'}, 1), snapshots = [];
  network.maxBatch = job.maxBatch ?? network.maxBatch;
  let sent = 0, rejected = false, lateProof = false, blockedClose = false, readyBeforeResult = 0, forwardsFinished = 0;
  let forwardCalls = 0, decoded = false, packed = false, messageCancelled = false;
  if (job.stopOnDecode || job.stopOnFeatures || job.expireBeforeInstall) network.evaluateNative = async (batch, options) => {
    const decode = batch.decode.bind(batch), features = batch.features.bind(batch);
    batch.decode = (...args) => {decode(...args); decoded = true;};
    batch.features = (...args) => {const input = features(...args); packed = true; return input;};
    const result = await Network.prototype.evaluateNative.call(network, batch, options);
    if (job.expireBeforeInstall) await new Promise(resolve => setTimeout(resolve, 2 * job.ms));
    return result;
  };
  network.forward = async (input, count, size) => {
    forwardCalls++;
    sent += count;
    if (job.cancelByMessage && forwardCalls === (job.messageCancelAfter ?? 1)) {
      const channel = new MessageChannel();
      channel.port1.onmessage = () => {messageCancelled = true;channel.port1.close();channel.port2.close();};
      channel.port2.postMessage('cancel');
    }
    if (job.delay && forwardCalls <= (job.delayedForwards ?? Infinity)) {
      await new Promise(resolve => setTimeout(resolve, job.delay));
      readyBeforeResult = Math.max(readyBeforeResult, Number(owner.m._hxgf_queued(owner.feed)));
    }
    if (job.cancel && forwardCalls >= (job.cancelAfter ?? 1)) {
      owner.cancel();
      try { owner.close(); } catch { blockedClose = true; }
    }
    if (job.prove && !lateProof && forwardCalls >= (job.proveAfter ?? 1)) {graph.proveLoss(1-native.game(job.history).player, 7);lateProof = true;}
    const policy = new Float32Array(count * (Array.isArray(size) ? size[0]*size[1] : size*size)), far = new Float32Array(count), value = new Float32Array(count);
    if (job.nonfinite) policy[0] = NaN;
    forwardsFinished++;
    return {policy, far, value};
  };
  try {
    let result;
    try { result = await owner.search({network, choice: job.choice ?? 'gumbel', batchSize: job.batchSize ?? 16, onBatch: stats => snapshots.push(stats),
      stop: () => {if (job.throwWhilePending && sent) throw new Error('Control failed during inference');return Boolean(messageCancelled || job.stopOnDecode && decoded || job.stopOnFeatures && packed || job.stopAfter && forwardsFinished >= job.stopAfter);}}); }
    catch (error) { if (!job.nonfinite && !job.throwWhilePending) throw error; rejected = /Nonfinite|Control failed/.test(error.message); }
    answer = {result, stats: owner.stats(), sent, snapshots, rejected, lateProof, blockedClose, readyBeforeResult, forwardsFinished, forwardCalls};
    owner.close();
    graph.at(job.history);
    answer.graph = graph.counters();
    answer.remainingViews = graph.counters().views;
  } finally { owner.close(); graph.close(); }
} else if (job.kind === 'native-proofs') {
  const {Worker: Thread} = await import('node:worker_threads'), workers = [], controls = [], waits = new Map(), events = [];
  let next = 0, inForward = false, cancelled = false, forwards = 0, error = null;
  const script = `import {parentPort} from 'node:worker_threads';
    import {loadTactical,proofAnswer} from ${JSON.stringify(new URL('../../web/engine/tactical.mjs',import.meta.url).href)};
    const solver=await loadTactical(${JSON.stringify(new URL('../../web/engine/tactical.wasm',import.meta.url).href)});
    parentPort.postMessage({ready:true});
    parentPort.on('message',({id,request,cancel})=>{const {history,...options}=request;
      const start=performance.now(), result=solver.history(history,{...options,cancel});
      parentPort.postMessage({id,answer:proofAnswer(result,request),ms:performance.now()-start});});`;
  try {
    await Promise.all(Array.from({length:job.workers??1},(_,index)=>new Promise((resolve,reject)=>{
      const worker=new Thread(new URL('data:text/javascript,'+encodeURIComponent(script)),{type:'module'});workers.push(worker);controls.push(new Int32Array(new SharedArrayBuffer(4)));
      worker.on('error',reject);
      worker.on('message',data=>{
        if(data.ready){resolve();return;}
        const wait=waits.get(data.id);waits.delete(data.id);
        events.push({worker:index,request:wait.request,ms:data.ms,info:data.answer.info,resident:data.answer.resident_reused,reused:data.answer.frontier_reused_nodes,duringForward:inForward});
        wait.resolve(data.answer);
      });
    })));
    const graph = new GameGraph(native,{history:job.history,roundBarrier:true});
    let middle = null, peer = null;
    if (job.offer || job.cancelBeforeDispatch || job.cooldown) {
      const evaluate = async leaves => leaves.map(({history,actions}) => ({logits:actions.map(()=>0),q:actions.map(()=>job.cooldown && history.length>job.history.length ? .8 : 0)}));
      await drive(graph,{simulations:4,rootSamples:4,evaluate});
      if(job.offer){middle=graph.view(job.offer.history.slice(0,-1));await drive(middle,{simulations:4,rootSamples:4,evaluate});
        peer=graph.view(job.offer.peer || job.offer.history);}
    }
    const owner = new NativeOwner(graph,{work:job.ms?0:4096,ms:job.ms??0,views:job.views??4,depth:job.depth??6});
    const rejectedEndpoints=[];
    if(job.rejectEndpoints)for(const endpoints of [-1,9,1.5]){
      try{new NativeProofs(owner,{query:()=>Promise.resolve({info:Array(13).fill(0)}),endpoints});}
      catch(error){rejectedEndpoints.push(String(error));}
    }
    const network = new Network(null,null,'fp32',{model_version:'test'},1);
    network.forward = async (input,count,size)=>{
      forwards++;inForward=true;await new Promise(resolve=>setTimeout(resolve,job.delay??20));inForward=false;
      return {policy:new Float32Array(count*(Array.isArray(size) ? size[0]*size[1] : size*size)),far:new Float32Array(count),value:new Float32Array(count)};
    };
    const query = (index,request)=>new Promise(resolve=>{
      // This fixture tests delivery from the offered interior position. A root
      // proof can otherwise finish first and correctly cancel that query.
      if(job.offer && JSON.stringify(request.history)!==JSON.stringify(job.offer.history)){
        const info=proofAnswer({status:'UNKNOWN',nodes_fresh:0},request);
        events.push({request,info:info.info});resolve(info);return;
      }
      if(job.cooldown){
        const info=proofAnswer({status:'UNKNOWN',nodes_fresh:0,proof_numbers:request.attacker==='mover'?{scope:'wide-forcing',game_exact:false,pn:1073741824,dn:0}:null},request);
        events.push({request,info:info.info});resolve(info);return;
      }
      const id=++next;waits.set(id,{resolve,request});Atomics.store(controls[index],0,0);workers[index].postMessage({id,request,cancel:controls[index]});
      if(job.cancelOnDispatch)cancelled=true;
      if(job.cancelAfterMs) setTimeout(()=>{cancelled=true;owner.cancel();},job.cancelAfterMs);
    });
    try {
      if(job.cancelBeforeDispatch || job.cooldown){
        const frontier=new NativeProofs(owner,{query,cancel:i=>Atomics.store(controls[i],0,1),workers:1,slice:1000,stamps:false});
        try{
          frontier.pump();
          if(job.cancelBeforeDispatch)owner.cancel();
          else {for(let i=0;i<30;i++){await Promise.resolve();frontier.pump();}}
          answer={queries:events.length,discrepancy:Math.abs(native.m._hxg_value(graph.ptr))};
        }finally{await frontier.close();if(answer)answer.proof=frontier.finalStats;owner.cancel();}
      }else if(job.offer){
        const frontier = new NativeProofs(owner,{query,cancel:i=>Atomics.store(controls[i],0,1),workers:workers.length,slice:job.slice??8,stamps:false});
        try {
          try {
            frontier.offer(job.offer.history,10);
            const until=performance.now()+2000;
            while(native.m._hxg_exact(peer.ptr)<0 && performance.now()<until){frontier.pump();await new Promise(resolve=>setTimeout(resolve,2));}
            answer={peerExact:native.m._hxg_exact(peer.ptr),rootExact:native.m._hxg_exact(graph.ptr),middleExact:native.m._hxg_exact(middle.ptr)};
          } finally {peer.close();}
        } finally {await frontier.close();if(answer){answer.proof=frontier.finalStats;answer.records=frontier.records;}owner.cancel();}
      } else {
        try {
          const result=await owner.search({network,batchSize:16,proofs:{query,cancel:i=>Atomics.store(controls[i],0,1),workers:workers.length,slice:job.slice??8,stamps:false,endpoints:job.endpoints??8},stop:()=>cancelled});
          answer={result,proof:result.proof_scheduler,records:result.proof_records,neuralRecords:result.neural_records};
        } catch(e){error=String(e);answer={error};}
      }
      answer.stats=owner.stats();answer.events=events;answer.forwards=forwards;answer.waits=waits.size;answer.cancelled=cancelled;answer.rejectedEndpoints=rejectedEndpoints;
      answer.viewHistories=[];
      for(let i=0;i<answer.stats.records;i++){
        const count=native.m._hxgo_record_history(owner.ptr,i,0), buffer=native.alloc(count*16);
        try{native.m._hxgo_record_history(owner.ptr,i,buffer);answer.viewHistories.push(native.pairs(buffer,count));}
        finally{native.m._free(buffer);}
      }
      owner.close();graph.at(job.history);answer.graph=graph.counters();
    } finally {owner.close();peer?.close();middle?.close();graph.close();}
  } finally {await Promise.all(workers.map(w=>w.terminate()));}
} else if (job.kind === 'solver-preparation-cancel') {
  const messages=[],workerUrl=new URL('../../web/engine/worker.mjs',import.meta.url);let ready;
  const context={URL,performance,setTimeout,clearTimeout,SharedArrayBuffer,Int32Array,Atomics,onmessage:null,
    Worker:class{postMessage(data){messages.push(data);if(data.prepare)ready=()=>this.onmessage({data:{id:data.id,ready:true}});}terminate(){}}};
  const source=readFileSync(workerUrl,'utf8').replace(/^import .*;\r?$/gm,'').replaceAll('import.meta.url',JSON.stringify(workerUrl.href));
  runInNewContext(source+'\nglobalThis.SolverWorkers=SolverWorkers;',context);
  const pool=new context.SolverWorkers(1);let cancelled=false;
  const pending=pool.query(0,{history:[[0,0]],attacker:'mover',ms:1000},()=>cancelled);
  cancelled=true;pool.cancel(0);ready();const found=await pending;
  answer={messages:messages.length,query_messages:messages.filter(m=>m.request).length,flag:Atomics.load(pool.entries[0].control,0),info:found.info};pool.close();
} else if (job.kind === 'owner-adapter') {
  const fetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({networks: []}));
  const {BubbleEngine, PRESETS} = await import('../../web/engine/bubble.mjs');
  globalThis.fetch = fetch;
  const engine = new BubbleEngine({model:'test'});engine.call = async request => request;
  answer = await Promise.all(Object.values(PRESETS).map(budget => engine.turn([[0,0]], budget)));
} else if (job.kind === 'worker-model-cache') {
  const messages=[],created=[],live=new Set(),attempts=[],workerUrl=new URL('../../web/engine/worker.mjs',import.meta.url);
  let fail=true,largest=0;
  const context={onmessage:null,Stages,errorReport,URL,performance,setTimeout,clearTimeout,postMessage:m=>messages.push(m),
    Network:{async create({model}){
      await new Promise(resolve=>setTimeout(resolve,1));created.push(model);live.add(model);largest=Math.max(largest,live.size);
      return {closed:false,async time(){if(this.closed)throw Error('Closed active network');return model;},
        async close(){this.closed=true;attempts.push(model);if(model==='A'&&fail)throw Error('Persistent session release failure');live.delete(model);}};
    }}};
  const source=readFileSync(workerUrl,'utf8').replace(/^import .*;\r?$/gm,'').replaceAll('import.meta.url',JSON.stringify(workerUrl.href));
  runInNewContext(source,context);
  let id=0;const use=model=>context.onmessage({data:{type:'use',id:++id,model}});
  try {
    await use('A');await use('B');await Promise.all(['C','D','E'].map(use));await use('A');
    const bench=++id;await context.onmessage({data:{type:'bench',id:bench,batches:[1],sizes:[24],repeats:1}});
    answer={blocked:{created:[...created],live:[...live],errors:messages.filter(m=>m.type==='error').map(m=>m.message)}};
    answer.active_model=messages.find(m=>m.id===bench&&m.type==='result')?.result['24']['1'];
    fail=false;await Promise.all(['F','G','H'].map(use));
    answer.recovered={created:[...created],live:[...live],largest,attempts:[...attempts]};
  }finally{fail=false;await runInNewContext('Promise.all([...held.values()].map(n=>n.close()))',context);}
  answer.remaining=live.size;
} else if (job.kind === 'search') {
  answer = [];
  for (const item of job.cases) answer.push(await search(item));
} else if (job.kind === 'rounds') {
  const tree = new GameGraph(native, {history: [[0,0]], seed: 23, roundBarrier: true});
  const predict = leaf => ({logits: leaf.actions.map(() => 0), q: leaf.actions.map(() => 0)});
  const install = ([id, leaf]) => tree.fulfill(id, leaf.actions, predict(leaf));
  try {
    native.checked(native.m._hxg_begin(tree.ptr,128,8));install(tree.request());
    const first = Array.from({length:8}, () => tree.request());
    const blocked = tree.request()[0];install(first[0]);const extra = tree.request();
    const lost = first[1][1].history.at(-1);
    native.checked(native.m._hxg_mark_exact(tree.ptr, BigInt(lost[0]), BigInt(lost[1]),0,4));
    const replacement = tree.request();install(first[1]);
    native.m._hxg_cancel(tree.ptr);
    const retired = tree.counters();
    const result = await drive(tree, {simulations:32,rootSamples:8,batchSize:128,evaluate:async leaves => leaves.map(predict)});
    answer = {blocked,extra:extra[0]>0,replacement:replacement[0]>0,retired,
      completed:result.completed,mass:result.policy.reduce((a,b) => a+b,0),action:result.action};
  } finally { tree.close(); }
} else if (job.kind === 'archive') {
  const original = [[0,0],[1,1],[2,1],[2,0],[0,3],[0,2],[1,2],[-1,2],[3,1],[-1,0],[0,-1]];
  const current = [...original.slice(0,3), ...original.slice(7,9), ...original.slice(5,7)];
  const returned = [...current, ...original.slice(3,5), ...original.slice(9,11)];
  const graph = new GameGraph(native, {history: original, seed: 51, limit: 4, archiveBytes: 262144, archiveForward: true});
  const evaluate = async leaves => leaves.map(({actions}) => ({logits: actions.map(() => 0), q: actions.map(() => 0)}));
  const cache = new EvaluationCache(0);
  try {
    const first = await drive(graph, {simulations:128, rootSamples:16, batchSize:16, evaluate, cache});
    graph.at(current);
    await drive(graph, {simulations:128, rootSamples:16, batchSize:16, evaluate, cache});
    await drive(graph, {simulations:4, rootSamples:4, batchSize:16, evaluate, cache});
    graph.at(returned);const reused = graph.result();
    const after = await drive(graph, {simulations:8, rootSamples:8, batchSize:16, evaluate, cache});
    answer = {first:first.visits, reused:reused.visits, after:after.visits, credits:graph.credits(), archive:graph.archive(), counters:graph.counters()};
  } finally {graph.close();}
  answer.conflicts = [];
  for (const forward of [false,true]) for (const opposite of [false,true]) {
    const retained = new GameGraph(native, {history: [[0,0]], seed: 7, limit: 1, archiveBytes: 65536, archiveForward: forward});
    try {
      for (const history of [[[0,0]], [[0,0],[1,0],[2,0]], [[0,0],[1,0],[2,0],[3,0]]]) {
        retained.at(history); native.checked(native.m._hxg_begin(retained.ptr,1,1));
        const [id,leaf] = retained.request(); retained.fulfill(id,leaf.actions,(await evaluate([leaf]))[0]);
      }
      retained.at([[0,0]]); const before = retained.archive();
      retained.at(opposite ? [[0,0],[4,0],[5,0],[1,0]] : [[0,0],[3,0]]);
      answer.conflicts.push({forward,opposite,before,after:retained.archive(),counters:retained.counters()});
    } finally {retained.close();}
  }
  answer.ownerConflicts = [];
  for (const forward of [false,true]) {
    const retained = new GameGraph(native, {history: [[0,0]], seed: 7, limit: 1, archiveBytes: 65536, archiveForward: forward});
    let owner;
    try {
      for (const history of [[[0,0]], [[0,0],[1,0],[2,0]], [[0,0],[1,0],[2,0],[3,0]]]) {
        retained.at(history); native.checked(native.m._hxg_begin(retained.ptr,1,1));
        const [id,leaf] = retained.request(); retained.fulfill(id,leaf.actions,(await evaluate([leaf]))[0]);
      }
      retained.at([[0,0]]);
      owner = new NativeOwner(retained, {work: 32, ms: 0});
      owner.close(); const before = retained.archive();
      for (const point of [[4,0],[5,0],[1,0]]) retained.advance(point);
      answer.ownerConflicts.push({forward,before,after:retained.archive(),counters:retained.counters()});
    } finally {owner?.close();retained.close();}
  }
  const second = original.map(c => [...c]), third = original.map(c => [...c]);
  [second[5],second[9]] = [second[9],second[5]];[second[6],second[10]] = [second[10],second[6]];
  [third[1],third[9]] = [third[9],third[1]];[third[2],third[10]] = [third[10],third[2]];
  const bounded = new GameGraph(native, {history: original, seed: 7, limit: 1, archiveBytes: 65536});
  try {
    for (const history of [original,second,third]) {
      bounded.at(history);await drive(bounded, {simulations:1,rootSamples:1,batchSize:1,evaluate,cache});
    }
    bounded.at(third);const before = bounded.archive(), winner = 1-native.game(third).player;
    bounded.proveLoss(winner,7);
    const after = bounded.archive();
    bounded.at(original);
    answer.proofGrowth = {before,after,winner,returnedWinner:bounded.m._hxg_exact(bounded.ptr)};
  } finally {bounded.close();}
} else if (job.kind === 'views') {
  const parent = new GameGraph(native, {history: job.history, seed: 3});
  const evaluate = async leaves => leaves.map(({actions}) => ({logits: actions.map(() => 0), q: actions.map(() => 0)}));
  let view, proof;
  try {
    await drive(parent, {simulations: 16, rootSamples: 4, batchSize: 4, evaluate});
    const before = parent.counters(), beforeCredits = parent.credits();
    view = parent.view(job.history, 19);
    await drive(view, {simulations: 32, rootSamples: 8, batchSize: 8, evaluate});
    answer = {before, after: parent.counters(), beforeCredits, afterCredits: parent.credits(), viewCredits: view.credits()};
    parent.close();
    await drive(view, {simulations: 8, rootSamples: 4, batchSize: 4, evaluate});
    answer.survived = view.counters();
    native.checked(native.m._hxg_begin(view.ptr, 8, 4));
    const [id, leaf] = view.request();
    if (id <= 0) throw new Error('Expected a pending neural leaf');
    proof = view.view(leaf.history, 4);
    proof.proveLoss(1 - native.game(leaf.history).player, 7);
    view.fulfill(id, leaf.actions, (await evaluate([leaf]))[0]);
    answer.retired = view.counters();
    answer.proofValue = proof.result().node_value;
  } finally { proof?.close(); view?.close(); parent.close(); }
} else if (job.kind === 'game') {
  const trees = new GameGraphs(native), cache = new EvaluationCache(), options = {seed: 1740, tactics: true, qRangeFloor: 0};
  const evaluate = async leaves => leaves.map(({actions}) => ({logits: actions.map((_, i) => -2 * i), q: actions.map(() => 0)}));
  const turn = async (line, history) => {
    const player = native.game(history).player, current = history.map(p => [...p]), carried = [];
    while (native.game(current).player === player && native.game(current).winner < 0) {
      const tree = trees.graph(line, current, options);
      carried.push(tree.result().visits.reduce((a, b) => a + b, 0));
      current.push((await drive(tree, {simulations: job.simulations, rootSamples: 16, cache, evaluate})).action);
    }
    return {history: current, carried, tree: trees.graphs.get(line).graph};
  };
  const first = await turn('a', [[0, 0]]), reply = [...first.history];
  for (let i = 0; i < 2; i++) reply.push(native.legal(reply)[0]);
  const second = await turn('a', reply);
  answer = {first: first.carried, second: second.carried, same: second.tree === first.tree,
    undone: (await turn('a', [[0, 0]])).carried, fresh: (await turn('b', reply)).carried};
  await turn('c', [[0, 0]]);
  await turn('d', [[0, 0]]);
  answer.lines = [...trees.graphs.keys()];
} else if (job.kind === 'revisit') {
  const a = job.history, mover = native.game(a).player;
  let refuted = null;
  const evaluate = async leaves => leaves.map(({history, actions}) => {
    const below = refuted && history.length > refuted.length && refuted.every(([q, r], i) => history[i][0] === q && history[i][1] === r);
    const value = below ? ((((history.length + 1) >> 1) % 2) === mover ? -.9 : .9) : 0;
    return {logits: actions.map((_, i) => -2 * i), q: actions.map(() => value)};
  });
  const graph = new GameGraph(native, {seed: 5, tactics: true, history: a}), cache = new EvaluationCache();
  const pick = ({action, visits, policy, completed_q}) => ({action, visits, policy, completed_q});
  try {
    const first = await drive(graph, {simulations: 64, rootSamples: 8, batchSize: 8, cache, evaluate});
    refuted = [...a, first.action];
    graph.at(refuted);
    await drive(graph, {simulations: 1024, rootSamples: 16, batchSize: 32, cache: new EvaluationCache(), evaluate});
    graph.at(a);
    const back = graph.result('policy');
    const again = await drive(graph, {simulations: 32, rootSamples: 8, batchSize: 8, cache: new EvaluationCache(), evaluate});
    answer = {first: pick(first), back: pick(back), again: pick(again)};
  } finally { graph.close(); }
} else if (job.kind === 'proof-answer') {
  try{answer={result:proofAnswer(job.result,job.request)};}catch(error){answer={error:String(error)};}
} else if (job.kind === 'tactical') {
  const solver = await loadTactical(new URL('../../web/engine/tactical.wasm', import.meta.url).href);
  answer = job.queries.map(({history, options}) => solver.history(history, options));
} else if (job.kind === 'worker-turn' || job.kind === 'glimpse') {
  const solver = await loadTactical(new URL('../../web/engine/tactical.wasm', import.meta.url).href);
  const messages = [], queries = [], evaluations = [], workerUrl = new URL('../../web/engine/worker.mjs', import.meta.url);
  const glimpsing = job.kind === 'glimpse', mover = native.game(job.history).player;
  let prepares = 0, constructed = 0;
  let graph = null;
  const context = {Native, EvaluationCache, GameGraph, NativeOwner, createModule, principalVariation, topRows,
    GameGraphs: class extends GameGraphs { graph(...args) { return graph = super.graph(...args); } },
    Proofs, answered, settled, proofTurns, proofKey, proofEvidence, winningLine, proven,
    URL, performance, setTimeout, clearTimeout, onmessage: null,
    postMessage: message => messages.push({...message, root: graph?.history.map(p => [...p]), at: performance.now()}),
    probe: async () => ({provider: 'wasm', precisions: ['fp32']}), runtime: async () => ({env: {wasm: {numThreads: 1}}}),
    Stages, errorReport, stall,
    Network: {create: async () => ({version: 'uniform', precision: 'fp32', threads: 1,
      evaluateNative: Network.prototype.evaluateNative, maxBatch: 64,
      // `peaked`: logits fall by 16 per cell away from the crop's centre, so searches keep meeting the same few
      // stones, as a trained network's do; otherwise uniform.
      forward: async (input, count, size) => {
        const [height, width] = Array.isArray(size) ? size : [size, size], policy = new Float32Array(count*height*width);
        if (job.peaked) policy.forEach((_, i) => { const c = i % (height*width); policy[i] = -16 * Math.hypot(Math.floor(c / width) - height / 2, c % width - width / 2); });
        return {policy, far: new Float32Array(count), value: new Float32Array(count)};
      },
      evaluate: async leaves => leaves.map(({history, actions}) => {
        if (messages.some(m => m.type === 'ready')) evaluations.push(history);
        const value = glimpsing ? (native.game(history).player === mover ? .86 : -.86) : 0;
        return {logits: actions.map((_, i) => glimpsing ? -2 * i : 0), q: actions.map(() => value)};
      })})},
    Worker: class {
      constructor() { constructed++; if (job.noWorkers) throw new Error('nested workers are not allowed'); }
      postMessage({id, history, options, prepare, request, cancel}) {
        if (prepare) { prepares++; if (!job.stallPrepare) queueMicrotask(() => this.onmessage({data:{id,ready:true}})); return; }
        if (request) {
          const {history,...options} = request;
          // A worker's answer arrives as a task, so the owner's turn can run between proof slices.
          const found = solver.history(history,{...options,cancel});
          setTimeout(() => this.onmessage({data:{id,answer:proofAnswer(found,request)}})); return;
        }
        queries.push({preview: messages.some(m => m.live?.top?.length), stage: messages.at(-1)?.stage});
        const result = job.replayMiss && options.replay?.length
          ? {status: 'UNKNOWN', native_verified: false, moves: [], nodes_used: options.nodes, reason: 'replay work limit'}
          : solver.history(history, options);
        queueMicrotask(() => this.onmessage({data: {id, result}}));
      }
      terminate() {}
    }};
  const source = readFileSync(workerUrl, 'utf8').replace(/^import .*;\r?$/gm, '')
    .replaceAll('import.meta.url', JSON.stringify(workerUrl.href));
  runInNewContext(source, context);
  await context.onmessage({data: {type: 'load', options: {prefer: 'wasm'}}});
  if (job.adapter) {
    const fetch = globalThis.fetch;
    globalThis.fetch = async () => new Response(JSON.stringify({networks: []}));
    const {BubbleEngine, PRESETS} = await import('../../web/engine/bubble.mjs');
    globalThis.fetch = fetch;
    const adapter = new BubbleEngine({model: 'test'}), session = new BrowserSession(native);
    let calls = 0;
    adapter.call = async request => {
      const id = ++calls;
      await context.onmessage({data: {...request, id}});
      const reply = messages.find(m => m.id === id && (m.type === 'result' || m.type === 'error'));
      if (reply?.type !== 'result') throw Error(reply?.message || 'Worker did not answer');
      return reply.result;
    };
    // Without a preset the job's simulations and root nodes are the standard level, so a test can set both.
    session.registerEngine({id: 'test', kind: 'bubble', name: 'Bubble',
      presets: job.preset ? PRESETS : {...PRESETS, standard: {simulations: job.simulations, solver_nodes: job.nodes}}}, adapter);
    const imported = await session.request('/import', {text: JSON.stringify({history: job.history, records: job.restore ? [] : job.records || []})}, 'POST');
    if (imported[0] !== 200) throw Error(JSON.stringify(imported));
    if (job.restore) {
      session.records = job.records;
      for (const record of job.records) await session.storage.put('evaluations', record);
      await session.persist(); await session.saving; await session.restore({paused: true});
    }
    const configured = await session.request('/analysis', {engine: 'test', preset: job.preset ?? 'standard', auto: false}, 'POST');
    if (configured[0] !== 200) throw Error(JSON.stringify(configured));
    const requested = await session.request('/analyse', {ply: job.history.length}, 'POST');
    if (requested[0] !== 200) throw Error(JSON.stringify(requested));
    while (session.running || session.jobs.some(j => j.status === 'queued')) await new Promise(r => setTimeout(r, 1));
    answer = session.lookup(job.history);
    if (job.preset && answer) answer = {...answer, analysis: session.state().analysis, solver_frames: messages.filter(m => m.live?.solver).map(m => m.live.solver)};
    if (!answer) throw Error(JSON.stringify(session.state().jobs));
  } else if (job.warm) {
    // The page's prepare call starts the proof workers before any clock runs; the turn then reuses them.
    await context.onmessage({data: {type: 'use', id: 1, model: 'test', proofWorkers: 2}});
    const warmed = prepares, tried = constructed;
    // A browser that could not start them is not asked again.
    await context.onmessage({data: {type: 'use', id: 3, model: 'test', proofWorkers: 2}});
    const retried = constructed - tried;
    await context.onmessage({data: {type: 'turn', id: 2, history: job.history, simulations: job.simulations, solverNodes: job.nodes, solverWorkers: 2}});
    answer = {warmed, after: prepares, tried, retried, moves: messages.find(m => m.id === 2 && m.type === 'result')?.result.moves};
  } else if (job.stallPrepare) {
    // The proof workers never answer their preparation; a cancel 50 ms in must still end the turn.
    const started = performance.now();
    setTimeout(() => context.onmessage({data: {type: 'cancel', id: 1}}), 50);
    await context.onmessage({data: {type: 'turn', id: 1, history: job.history, simulations: job.simulations, solverNodes: job.nodes}});
    answer = {replies: messages.filter(m => m.id === 1 && m.type !== 'progress').map(m => m.type), ms: performance.now() - started};
    runInNewContext('frontierWorkers.close()', context);
  } else if (job.cancelAfter) {
    // A long search on a game's graph, cancelled `cancelAfter` ms in: when its replies came, and the live rows it sent.
    const started = performance.now();
    setTimeout(() => context.onmessage({data: {type: 'cancel', id: 1}}), job.cancelAfter);
    await context.onmessage({data: {type: 'turn', id: 1, history: job.history, line: 'game', simulations: job.simulations, solverNodes: 0}});
    const mine = messages.filter(m => m.id === 1), times = [started, ...mine.map(m => m.at)];
    answer = {replies: mine.filter(m => m.type !== 'progress').map(m => m.type), ms: mine.at(-1).at - started,
      gap: Math.max(...times.slice(1).map((t, i) => t - times[i])), live: mine.filter(m => m.live).map(m => m.live)};
  } else for (let id = 1; id <= (glimpsing ? 2 : 1); id++) {
    await context.onmessage({data: {type: 'turn', id, history: job.history, line: glimpsing ? 'live' : null,
      simulations: job.simulations, solverNodes: job.nodes, solverSlice: job.solverSlice ?? 8, ms: job.ms ?? null, proveMs: job.proveMs ?? 0,
      known: job.known || null, replay: job.replay || [], proofStamps: job.proofStamps}});
  }
  const error = messages.find(m => m.type === 'error');
  if (error) throw new Error(error.message);
  answer ??= glimpsing ? [1, 2].map(id => ({result: messages.find(m => m.id === id && m.type === 'result').result,
    progress: messages.filter(m => m.id === id && m.type === 'progress').map(m => ({fraction: m.fraction, stage: m.stage})), queries, evaluations,
    live: messages.filter(m => m.id === id && m.live).map(m => ({...m.live, root: m.root}))}))
    : {...messages.find(m => m.type === 'result').result, solver_frames: messages.filter(m => m.live?.solver).length,
      solver_live: messages.filter(m => m.live?.solver).map(m => m.live.solver)};
} else if (job.kind === 'page-dismissal') {
  // The page with Auto on: the analysis requests it sends after Cancel, after stepping back and after stepping forward.
  const source = readFileSync(new URL('../../web/index.html', import.meta.url), 'utf8'), session = new BrowserSession(native);
  session.registerEngine({id:'test',kind:'bubble',presets:{standard:{simulations:4,solver_nodes:0}}},{turn:(history,budget,options)=>
    new Promise((_,reject)=>options.signal.addEventListener('abort',()=>reject(new DOMException('cancelled','AbortError'))))});
  session.history = [[0,0]]; session.analysis = session.spec({engine:'test',preset:'standard',auto:true});
  const elements = new Map(), requests = [];
  const element = () => ({classList:{toggle(){}},style:{setProperty(){}},firstChild:{style:{}},children:[],
    replaceChildren(){},append(){},setAttribute(k,v){this[k]=v;}});
  const page = {S:null,view:1,COLORS:['yellow','blue'],STOPS:['standard'],STOP_ICON:[''],asked:new Map(),
    performance,placing:[],placed:new Set(),landed:new Map(),seenLabels:new Map(),fitted:false,
    $:id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id);},
    setIcon:(e,icon)=>{e.icon=icon;},toast(){},fit(){},draw(){},tickClocks(){},renderPanels(){},renderAnalysis(){},
    post:async(path,body)=>{requests.push([path,body]);return (await session.request(path,body,'POST'))[1];},key:(q,r)=>`${q},${r}`};
  runInNewContext(source.match(/^const playerAt=.*$/m)[0]+'\n'
    + source.slice(source.indexOf('const failures='),source.indexOf('/* board geometry:'))
    + source.slice(source.indexOf('function finished('),source.indexOf('function renderAnalysis('))
    + source.slice(source.indexOf('function autoAnalyse('),source.indexOf('function renderGraph('))
    + source.slice(source.indexOf('function renderJobs('),source.indexOf('/* game actions */')),page);
  let shown = 0;
  const show = () => { page.accept({...session.state(),revision:session.revision+ ++shown}); };
  const sent = () => requests.splice(0).map(([path,body]) => path === '/analyse' ? `${path} ${body.ply}` : path);
  show(); await new Promise(resolve => setTimeout(resolve, 5));
  answer = {opened: sent()};
  // Each step clears the page's 3 s spacing between requests for one position, so only Auto's rules decide.
  show(); await elements.get('again').onclick(); page.asked.clear(); show();
  answer.cancelled = sent();
  answer.auto = session.analysis.auto;
  page.view = 0; page.asked.clear(); show(); answer.back = sent();
  page.view = 1; page.asked.clear(); show(); answer.forward = sent();
  session.cancelJobs();
} else if (job.kind === 'analysis-failure') {
  const source = readFileSync(new URL('../../web/index.html', import.meta.url), 'utf8'), session = new BrowserSession(native);
  let calls = 0;
  const entry = {id:'test',kind:'bubble',presets:{standard:{simulations:4,solver_nodes:0}}};
  session.registerEngine(entry,{turn:async()=>{
    if (++calls === 1) throw Error('Temporary inference failure');
    return {value:.5,moves:[[0,1],[1,0]],top:[[0,1,1,.5]],solved:true};
  }});
  session.history = [[0,0]]; session.analysis = session.spec({engine:'test',preset:'standard',auto:true});
  session.enqueue('analyse',session.history,session.analysis); await session.pump();
  const elements = new Map(), requests = [], notices = [];
  const element = () => ({classList:{toggle(){}},style:{setProperty(){}},firstChild:{style:{}},children:[],
    replaceChildren(){},append(){},setAttribute(k,v){this[k]=v;}});
  const page = {S:null,view:0,COLORS:['yellow','blue'],STOPS:['standard'],STOP_ICON:[''],asked:new Map(),
    performance,placing:[],placed:new Set(),landed:new Map(),seenLabels:new Map(),fitted:false,
    $:id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id);},
    setIcon:(e,icon)=>{e.icon=icon;},toast:text=>notices.push(text),
    fit(){},draw(){},tickClocks(){},renderPanels(){},renderAnalysis(){},
    post:async(path,body)=>{requests.push([path,body]);},key:(q,r)=>`${q},${r}`};
  runInNewContext(source.match(/^const playerAt=.*$/m)[0]+'\n'
    + source.slice(source.indexOf('const failures='),source.indexOf('/* board geometry:'))
    + source.slice(source.indexOf('function finished('),source.indexOf('function renderAnalysis('))
    + source.slice(source.indexOf('function autoAnalyse('),source.indexOf('function renderGraph('))
    + source.slice(source.indexOf('function renderJobs('),source.indexOf('/* game actions */')),page);
  page.accept(session.state());
  // Polling the same revision must keep the error visible without resubmitting it.
  page.accept(session.state());
  page.accept({instance:session.instance,revision:session.revision,jobs:[]});
  const failure = {stage:elements.get('analysis-stage').textContent,title:elements.get('analysis-stage').title,
    retry:elements.get('again')['aria-label'],requests:requests.length,notices:[...notices],calls};
  await elements.get('again').onclick();
  const retry = requests.at(-1);
  const response = await session.request(...retry,'POST'); page.accept(response[1]); await session.idle;
  page.accept(session.state());
  const recovered = {calls,value:session.lookup(session.history)?.value,stage:elements.get('analysis-stage').textContent,
    label:elements.get('again')['aria-label']};
  page.accept({...session.state(),revision:session.revision+1,jobs:[
    {id:1,kind:'analyse',status:'failed',ply:1,error:'Temporary inference failure'},
    {id:99,kind:'analyse',status:'queued',ply:1,done:0,total:1}]});
  const queued = {stage:elements.get('analysis-stage').textContent,
    progress:elements.get('analysis-progress').style.visibility,label:elements.get('again')['aria-label']};
  page.accept({...session.state(),revision:session.revision+1,jobs:[{id:1,kind:'analyse',status:'failed',ply:1,error:'Temporary inference failure'}]});
  answer = {failure,retry,recovered,queued,afterRetry:elements.get('analysis-stage').textContent};
} else if (job.kind === 'analysis-bar') {
  const source = readFileSync(new URL('../../web/index.html', import.meta.url), 'utf8');
  answer = job.cases.map(({history, value, live, top = [], node_value}) => {
    const elements = new Map(), element = () => ({classList: {toggle() {}}, style: {}, firstChild: {style: {}}, replaceChildren() {}});
    const evaluation = {value, node_value, top, threat: []};
    const page = {view: history.length, COLORS: ['yellow', 'blue'],
      $: id => { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); },
      S: {history, winner: -1, evaluations: live ? {} : {[history.length]: evaluation},
        jobs: live ? [{kind: 'analyse', ply: history.length, status: 'running', live: evaluation}] : []}};
    runInNewContext(source.match(/^const playerAt=.*$/m)[0] + '\n' + source.match(/^const pct=.*$/m)[0] + '\n'
      + source.slice(source.indexOf('function liveAt('), source.indexOf('function renderStudy('))
      + '\nconst e=shownEval(view); renderAnalysis.sig = JSON.stringify([view, false, e.top, e.threat, false]); renderAnalysis();', page);
    return {x: elements.get('xv').textContent, o: elements.get('ov').textContent,
      transform: elements.get('evalbar').firstChild.style.transform};
  });
} else if (job.kind === 'pv') {
  answer = principalVariation(native, job.history, job.certificate, job.options || {});
} else if (job.kind === 'review') {
  answer = job.cases.map(({history, evaluations, winner}) => {
    const table = new Map(evaluations.map(([prefix, record]) => [JSON.stringify(prefix), record]));
    return review(history, prefix => table.get(JSON.stringify(prefix)) ?? null, winner);
  });
} else if (job.kind === 'rows') {
  answer = topRows(job.actions, job.policy, job.values, job.lead);
} else if (job.kind === 'overlay') {
  const page = {};
  runInNewContext(readFileSync(new URL('../../web/engine/overlay.js', import.meta.url), 'utf8'), page);
  answer = job.cases.map(({ev, stones}) => page.boardOverlay(ev, stones));
} else if (job.kind === 'table') {
  const table = new Proofs();
  for (const [history, record] of job.records) {
    for (const h of job.queries) table.known(h);
    table.add(history, record);
  }
  const rebuilt = new Proofs(table.list());
  answer = {queries: job.queries.map(h => ({known: rebuilt.known(h), facts: rebuilt.facts(h),
    shown: proven(rebuilt, h, {moves: [], top: []}, h.length % 2 ? 2 : 1),
    edges: [...rebuilt.edges(h).values()].map(e => [...e.action, e.winner, e.distance])})),
    settled: settled(job.result, rebuilt.edges(job.queries[0]), job.mover), lost: settled(job.lost, rebuilt.edges(job.queries[0]), job.mover),
    exact: settled(job.exact, rebuilt.edges(job.queries[0]), job.mover)};
} else if (job.kind === 'proofs') {
  const s = new BrowserSession(native), sent = [], wait = () => new Promise(resolve => setTimeout(resolve, 1));
  const settle = async () => { for (let i = 0; s.running || s.jobs.some(j => j.status === 'queued'); i++) { if (i > 3000) throw Error('Analysis did not finish'); await wait(); } await s.saving; };
  const entry = {id: 'test', name: 'Test', kind: 'bubble', version: 'v1', checkpoints: [],
    presets: {quick: {simulations: 1, solver_nodes: 0}, standard: {simulations: 2, solver_nodes: 0}}};
  const adapter = {turn: async (history, budget, options) => {
    sent.push(options.known?.length ?? null);
    if (history.length === job.ply) return job.found;
    return searched(history, new Proofs(options.known));
  }};
  s.registerEngine(entry, adapter);
  s.analysis = s.spec({engine: 'test', preset: 'standard'});
  await s.request('/import', {text: JSON.stringify({history: job.history})}, 'POST');
  for (const [history, record] of job.records || []) await s.record(history, s.analysis, record);
  const shown = () => s.state().evaluations[job.ply - 1];
  for (const ply of [job.ply, job.ply - 1]) { await s.request('/analyse', {ply}, 'POST'); await settle(); }
  await s.record(s.history.slice(0, job.ply - 1), s.analysis, {moves: [], value: .5, top: [], proof: null, pv: [], threat: []});
  answer = {analysed: shown(), parent: s.state().evaluations[job.ply - 2], sent: [...sent], kept: s.lookup(s.history.slice(0, job.ply - 1))?.proof ?? null};
  const study = s.state().evaluations; s.changed(); answer.idleReuse = study === s.state().evaluations;
  await s.request('/undo', {people: []}, 'POST'); answer.undone = {length: s.history.length, shown: shown()};
  await s.request('/analysis', {engine: 'test', preset: 'quick'}, 'POST'); await s.request('/analyse', {ply: job.ply - 1}, 'POST'); await settle();
  answer.quick = {shown: shown(), saved: s.lookup(s.history.slice(0, job.ply - 1))};
  const reopened = new BrowserSession(native); reopened.storage = s.storage; await reopened.restore(); reopened.registerEngine(entry, adapter);
  answer.reloaded = reopened.state().evaluations[job.ply - 1];
  answer.reloadedParent = reopened.state().evaluations[job.ply - 2];
  const table = new Proofs();
  table.add(job.history, job.found);
  answer.given = answered(native, job.history.slice(0, job.ply - 1), table);
} else if (job.kind === 'custom-forms') {
  // Custom seats on the browser session: what each runs as the page edits Time, Nodes and Width, and what a reload keeps.
  const s = new BrowserSession(native), turn = async () => ({moves: [[0, 0]], value: .5, top: []});
  const {PRESETS: BUBBLE} = await import('../../web/engine/bubble.mjs'), {PRESETS: SIX} = await import('../../web/engine/six.mjs');
  s.registerEngine({id: 'bubble', name: 'Bubble', kind: 'bubble', presets: BUBBLE}, {turn});
  s.registerEngine({id: 'six', name: 'Six', kind: 'six', presets: SIX}, {turn});
  const seat = async (custom, engine = 'bubble') => {
    await s.request('/seat', {side: 1, engine, preset: 'custom', custom}, 'POST');
    return {budget: s.seats[1].budget, custom: s.seats[1].custom};
  };
  answer = {old: await seat({simulations: 300, solver_nodes: 9})};
  answer.time = await seat({...answer.old.custom, ms: 2500, active: 'ms'});
  answer.nodes = await seat({...answer.time.custom, simulations: 600, views: 4, active: 'simulations'});
  answer.six = await seat({nodes: 700, ms: 900, active: 'ms'}, 'six');
  answer.sixBack = await seat({...answer.six.custom, nodes: 800, active: 'nodes'}, 'six');
  await s.request('/seat', {side: 1, engine: 'bubble', preset: 'custom', custom: answer.time.custom}, 'POST'); await s.saving;
  const back = new BrowserSession(native); back.storage = s.storage; await back.restore();
  back.registerEngine({id: 'bubble', name: 'Bubble', kind: 'bubble', presets: BUBBLE}, {turn});
  answer.reloaded = {budget: back.seats[1].budget, custom: back.seats[1].custom};
  answer.bad = (await s.request('/seat', {side: 1, engine: 'bubble', preset: 'custom', custom: {views: 40}}, 'POST'))[0];
  // Shrimp speaks Six's protocol but keeps its own budget.
  const {PRESETS: SHRIMP} = await import('../../web/engine/shrimp.mjs'), other = new BrowserSession(native);
  other.registerEngine({id: 'shrimp', name: 'Shrimp', kind: 'six', badge: 'shrimp', presets: SHRIMP}, {turn});
  await other.request('/seat', {side: 1, engine: 'shrimp', preset: 'custom', custom: {...SHRIMP.standard, visits: 64}}, 'POST');
  answer.shrimp = {budget: other.seats[1].budget, custom: other.seats[1].custom};
} else if (job.kind === 'retired-checkpoints') {
  // Checkpoints the engine no longer offers: a saved Strix seat and analysis naming a retired network, Strix chosen
  // while its list is still empty and after it fills, and Bubble's networks.
  const wait = () => new Promise(resolve => setTimeout(resolve, 5)), played = [];
  const strix = checkpoints => ({id: 'strix', name: 'Strix', kind: 'strix', checkpoints, presets: {standard: {simulations: 1}}});
  const bubble = checkpoints => ({id: 'bubble', name: 'Bubble', kind: 'bubble', checkpoints, presets: {standard: {simulations: 1, solver_nodes: 0}}});
  const adapter = {turn: async (history, budget, options) => { played.push(options.checkpoint); return {moves: [[0, 0]], value: .5, top: []}; }};
  const saved = new BrowserSession(native), retired = {engine: 'strix', checkpoint: 'pulsatrix-10-best', preset: 'standard', budget: {simulations: 1}, auto: false};
  await saved.storage.saveSession({id: 'live', history: [], seats: [retired, {engine: 'human'}], analysis: {...retired, auto: true}, paused: false, _write_token: 'a'}, null, null);
  const s = new BrowserSession(native); s.storage = saved.storage; await s.restore();
  s.registerEngine(strix(['strix-237000']), adapter);
  for (const end = Date.now() + 30000; !s.history.length && Date.now() < end;) await wait();
  answer = {restored: {seat: s.seats[0].checkpoint, analysis: s.analysis.checkpoint, played: played.slice(), history: s.history}};
  const list = [], late = new BrowserSession(native);
  late.registerEngine(strix(list), adapter); late.registerEngine(bubble(['b2', 'b1']), adapter); late.registerEngine({...bubble([]), id: 'plain'}, adapter);
  const seat = async (side, body) => (await late.request('/seat', {side, ...body}, 'POST'))[0];
  answer.early = [await seat(0, {engine: 'strix'}), await seat(1, {engine: 'strix', checkpoint: 'pulsatrix-10-best'})];
  answer.empty = late.seats.map(spec => spec.checkpoint);
  list.push('strix-237000');
  answer.moved = late.relist('strix'); answer.again = late.relist('strix');
  answer.filled = late.seats.map(spec => spec.checkpoint);
  await late.saving;
  const back = new BrowserSession(native); back.storage = late.storage; await back.restore();
  answer.reloaded = back.seats.map(spec => spec.checkpoint);
  answer.bubble = [await seat(0, {engine: 'bubble', checkpoint: 'b1'}), late.seats[0].checkpoint, await seat(0, {engine: 'bubble', preset: 'standard'}), late.seats[0].checkpoint,
    await seat(1, {engine: 'bubble'}), late.seats[1].checkpoint, await seat(1, {engine: 'plain'}), late.seats[1].checkpoint];
  answer.untouched = late.relist('bubble');
} else if (job.kind === 'dismissal') {
  // Auto deepening while an engine seat plays, with analyses that run until cancelled: which positions have an
  // analysis running or queued after a cancel, an analysis request and a move.
  const s = new BrowserSession(native), wait = () => new Promise(resolve => setTimeout(resolve, 5));
  const entry = {id: 'test', name: 'Test', kind: 'bubble', version: 'v1', checkpoints: [], device: 'GPU',
    presets: {quick: {simulations: 1, solver_nodes: 0}, standard: {simulations: 2, solver_nodes: 0}}};
  s.registerEngine(entry, {turn: (history, budget, options) => new Promise((_, reject) =>
    options.signal.addEventListener('abort', () => reject(new DOMException('cancelled', 'AbortError'))))});
  await s.request('/import', {text: JSON.stringify({history: [[0, 0], [1, 0], [1, 1]]})}, 'POST');
  s.seats = [{engine: 'human'}, s.spec({engine: 'test'})]; s.paused = false;
  await s.request('/analysis', {engine: 'test', preset: 'standard', auto: true}, 'POST');
  const analysing = () => s.jobs.filter(j => j.kind === 'analyse').map(j => j.history.length);
  const settle = async () => { for (let i = 0; i < 20; i++) await wait(); return analysing(); };
  answer = {before: await settle()};
  await s.request('/cancel', {id: s.jobs.find(j => j.kind === 'analyse').id}, 'POST');
  answer.cancelled = await settle(); answer.auto = s.analysis.auto;
  await s.request('/analyse', {ply: 3}, 'POST');
  answer.asked = await settle();
  await s.request('/cancel', {id: s.jobs.find(j => j.kind === 'analyse').id}, 'POST');
  answer.again = await settle();
  await s.request('/play', {q: 2, r: 2}, 'POST');
  answer.moved = await settle();
  s.cancelJobs();
} else if (job.kind === 'restore-pause') {
  const make = async clock => {
    const s = new BrowserSession(native), entry = {id: 'test', name: 'Test', kind: 'bubble', version: 'v1', clocks: true, presets: {quick: {simulations: 1, solver_nodes: 0}, standard: {simulations: 1, solver_nodes: 0}}};
    s.registerEngine(entry, {turn: async history => ({moves: history.length ? [[1, 0], [2, 0]] : [[0, 0]], value: .5, top: []})});
    s.seats = [{engine: 'human'}, s.spec({engine: 'test'})]; s.changed();
    if (clock) await s.request('/clock', clock, 'POST');
    await s.request('/play', {q: 0, r: 0}, 'POST'); await s.saving;
    const before = s.paused;
    const back = new BrowserSession(native); back.storage = s.storage; await back.restore();
    clearTimeout(s.flag); clearTimeout(back.flag);  // the running game clocks would keep node alive for a minute
    return {before, after: back.paused, clock: Boolean(back.clock)};
  };
  answer = {budget: await make(null), clocked: await make(job.clock)};
} else if (job.kind === 'stale-loop') {
  const s = new BrowserSession(native), asked = [];
  const entry = {id: 'browser:test', name: 'Test', kind: 'bubble', version: 'v1', checkpoints: [], presets: {standard: {simulations: 4, solver_nodes: 0}, quick: {simulations: 4, solver_nodes: 0}}};
  s.registerEngine(entry, {turn: async (history, budget) => { asked.push([history.length, budget.simulations]); const value = budget.simulations < 4 && job.drift ? .5 + job.drift * asked.length : .5; return {moves: history.length ? [[9, 9], [9, 8]] : [[0, 0]], value, top: [], proof: null, line: [], graph_id: 'g'}; }});
  s.seats = [{engine: 'human'}, {engine: 'human'}]; s.analysis = s.spec({engine: entry.id, preset: 'standard'});
  const idle = async () => { while (s.running || s.jobs.some(j => j.status === 'queued')) await new Promise(r => setTimeout(r, 1)); };
  for (const [q, r] of job.history) await s.request('/play', {q, r}, 'POST');
  await s.request('/analyse', {ply: 2, force: true}, 'POST'); await idle();
  await s.request('/analyse', {ply: 4, force: true}, 'POST'); await idle();
  const after = {stale: s.state().stale, asked: [...asked]};
  await s.request('/analyse', {ply: 2}, 'POST'); await idle();
  answer = {...after, asked_after_view: [...asked], stale_after_view: s.state().stale};
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
} else if (job.kind === 'links') {
  answer = [];
  for (const [url, data] of job.links) {
    const asked = [], fetcher = async (api, init = {}) => { asked.push([api, init.method || 'GET']); return {ok: true, json: async () => data}; };
    try { answer.push({history: await readGame(url, native, fetcher), asked}); }
    catch (error) { answer.push({error: error.message, asked}); }
  }
} else if (job.kind === 'clock') {
  const wait = ms => new Promise(r => setTimeout(r, ms)), s = new BrowserSession(native), asked = [];
  s.registerEngine({id: 'fixed', name: 'Fixed', kind: 'strix', presets: {standard: {simulations: 1}}, clocks: false}, {turn: async () => ({moves: []})});
  s.registerEngine({id: 'timed', name: 'Timed', kind: 'native', presets: {standard: {ms: 1000}}, clocks: true}, {
    turn: async (history, budget, options) => { asked.push(options.ms); return {moves: [[history.length, 3], [history.length, 4]].slice(0, history.length ? 2 : 1)}; }});
  answer = {set: s.answer('/clock', {mode: 'game', tc: '0.3+1'})[0]};
  s.answer('/play', {q: 0, r: 0});
  answer.after_turn = s.clockNow();
  answer.fixed_seat = s.answer('/seat', {side: 1, engine: 'fixed'});
  s.answer('/seat', {side: 1, engine: 'timed'});
  await wait(60);
  answer.engine_turn = {history: s.history.length, asked, clock: s.clockNow()};
  clearTimeout(s.flag); await wait(s.clockNow().cross_ms + 50);
  answer.late = s.answer('/play', {q: 9, r: 9})[0];
  answer.timed_out = {state: (({winner, outcome}) => ({winner, outcome}))(s.state()), game: (await s.saving, await s.storage.get('games', s.gameId)), play: s.answer('/play', {q: 9, r: 8})[0]};
  { const v = new BrowserSession(native); v.storage = s.storage; await v.openGame(s.gameId, 1); answer.study_winner = v.state().winner; }
  s.answer('/new', {});
  answer.long_clock = s.answer('/clock', {mode: 'game', tc: '3000000'})[0];
  s.answer('/clock', {mode: 'game', tc: '0.3+1'});
  answer.fresh = {outcome: s.outcome, clock: s.clockNow()};
  s.answer('/seat', {side: 1, engine: 'human'}); s.answer('/clock', {mode: 'game', tc: '60'});
  await wait(100); s.answer('/pause', {paused: true}); await wait(100); s.answer('/pause', {paused: false}); await wait(100);
  s.answer('/play', {q: 0, r: 0});
  answer.paused_turn = s.clockTurns[0].spent_ms;
  const slow = new BrowserSession(native); let readyAt = 0;
  slow.registerEngine({id: 'slow', name: 'Slow', kind: 'native', presets: {standard: {ms: 1000}}, clocks: true}, {
    ready: async () => { await wait(300); readyAt = Date.now(); },
    turn: async history => ({moves: [[history.length, 3], [history.length, 4]]})});
  slow.answer('/clock', {mode: 'game', tc: '60'}); slow.answer('/play', {q: 0, r: 0});
  const seated = Date.now(); slow.answer('/seat', {side: 1, engine: 'slow'}); await wait(500);
  answer.load_charged = {spent: slow.clockTurns[1]?.spent_ms ?? null, load: readyAt - seated};
  s.answer('/pause', {paused: true}); slow.answer('/pause', {paused: true});
  { const m = new BrowserSession(native), entry = {id: 'net', name: 'Net', kind: 'bubble', checkpoints: ['1'], models: {'1': 'weights-a'}, presets: {standard: {simulations: 1, solver_nodes: 0}}};
    m.registerEngine(entry, {turn: () => new Promise(() => {})});
    await m.startMatch({players: [{engine: 'net'}, {engine: 'net'}], games: 2}); await m.request('/match', {action: 'stop'}, 'POST');
    entry.models = {'1': 'weights-b'};
    answer.rebuilt_model = (await m.request('/match', {action: 'resume'}, 'POST'))[1].error ?? null; m.cancelJobs(); }
  const r = new BrowserSession(native); r.answer('/clock', {mode: 'game', tc: '60'}); await wait(150);
  await r.persist(); await wait(250);
  const back = new BrowserSession(native); back.storage = r.storage; await back.restore();
  answer.reloaded = {balance: back.clock.cross_ms, partial: back.clockPartial}; r.freezeClock();
} else if (job.kind === 'play') {
  const s = new BrowserSession(native), data = JSON.parse(readFileSync(new URL('../../web/engine/openings.json', import.meta.url)));
  s.bookData = new OpeningBook(data);
  const entry = {id: 'browser:test', name: 'Test', kind: 'bubble', version: 'v1', checkpoints: [],
    presets: {standard: {simulations: 1, solver_nodes: 0}, quick: {simulations: 1, solver_nodes: 0}}};
  const asked = [];
  s.registerEngine(entry, {turn: async history => { asked.push(history.length); return {moves: job.history.slice(history.length, history.length + native.game(history).remaining), value: .5, top: [], proof: null, line: []}; }});
  s.analysis = s.spec({engine: entry.id, preset: 'standard'});
  answer = [];
  for (const [path, body] of job.requests) {
    const [status, data] = await s.request(path, body, 'POST');
    while (s.running || s.jobs.some(j => j.status === 'queued')) await new Promise(r => setTimeout(r, 1));
    answer.push({status, data: path === '/state' ? s.state() : data});
  }
  answer.push({backup: await s.storage.backup(), catalogue: await s.catalogue(), asked});
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
  const moved = s.lookup([], s.spec({engine: 'test', preset: 'standard'}), true);
  await s.request('/analyse', {ply: 0}, 'POST');
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
    variation: (await study.savedReplay(study.gameId, 1)).history, imported_label: label, move_reused: moved !== null, changed_version: imported.state().evaluations, restored_identity: reopened.gameId === id};
} else if (job.kind === 'resume') {
  const s = new BrowserSession(native), wait = ms => new Promise(r => setTimeout(r, ms));
  const until = async condition => { for (let i = 0; !condition(); i++) { if (i > 3000) throw Error('Move did not start'); await wait(1); } };
  s.bookData = new OpeningBook(JSON.parse(readFileSync(new URL('../../web/engine/openings.json', import.meta.url))));
  s.registerEngine({id:'test', name:'Test', kind:'bubble', version:'v1', presets:{standard:{simulations:1,solver_nodes:0}}, clocks:true}, {
    turn: (history, budget, options) => history.length === 1 ? Promise.resolve({moves:[[0,2],[1,2]],value:.5}) : new Promise((resolve,reject) => {
      options.signal.addEventListener('abort', () => reject(new DOMException('Cancelled','AbortError')), {once:true});
    })
  });
  s.analysis=s.spec({engine:'test',auto:true});
  await s.startMatch({players:[{engine:'test'},{engine:'test'}],games:2,clock:{mode:'game',tc:'180+2'}});
  await until(() => s.history.length === 3 && s.running?.history.length === 3);
  await s.request('/match',{action:'stop'},'POST'); await s.idle; await s.saving;
  const id=s.match.id, before={history:structuredClone(s.history),clock:s.clockNow(),timings:structuredClone(s.match.timings),turns:s.clockTurns.length};
  s.apply('/seat',{side:0,engine:'human'}); s.apply('/seat',{side:1,engine:'human'});
  await s.request('/match',{action:'resume'},'POST'); await until(()=>!!s.running);
  const sameBatchSeats=s.seats.map(s=>s.engine);
  await s.request('/match',{action:'stop'},'POST'); await s.idle;
  s.apply('/seat',{side:0,engine:'human'}); s.apply('/seat',{side:1,engine:'human'}); s.apply('/book',{enabled:true});
  await s.request('/new',{},'POST'); const bookStart=s.book.opening;
  await s.request('/import',{text:JSON.stringify({history:[[0,0]]})},'POST'); const imported=s.book.opening;
  await s.request('/match',{action:'resume',batch:id},'POST');
  await until(() => !!s.running); await s.request('/match',{action:'stop'},'POST'); await s.idle;
  answer={before,after:{history:s.history,clock:s.clockNow(),timings:s.match.timings,turns:s.clockTurns.length},bookStart,imported,auto:s.analysis.auto,sameBatchSeats};
} else if (job.kind === 'lifecycle') {
  const entry = {id: 'test', name: 'Test', kind: 'bubble', version: 'v1', presets: {standard: {simulations: 1, solver_nodes: 0}}, clocks: true};
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
    const adapter={turn:async history=>({moves:job.history.slice(history.length,history.length+(history.length?2:1)),value:.5})};
    paused.registerEngine(entry,adapter); paused.registerEngine({...entry,id:'other',name:'Other'},adapter);
    const commitSession = paused.storage.saveSession.bind(paused.storage);
    paused.storage.saveSession = async (...args) => { if(args[3]?.length) await new Promise(resolve=>{commit=resolve;}); return commitSession(...args); };
    await paused.startMatch({players:[{engine:'test'},{engine:'other'}],games:2,openings:[job.history.slice(0,-1)]});
    await until(()=>!!commit);
    await paused.request(path,path==='/pause'?{paused:true}:{id:paused.running.id},'POST'); commit(); await paused.idle;
    const result={paused:paused.paused,completed:paused.match.completed,pending:paused.match.pending_game,current:paused.match.current};
    paused.storage.saveSession=commitSession;
    await paused.request('/match',{action:'stop'},'POST');
    paused.apply('/seat',{side:0,engine:'human'}); paused.apply('/seat',{side:1,engine:'human'});
    await paused.request('/match',{action:'resume'},'POST'); await until(()=>!paused.match.active);
    result.resumedSeats=paused.seats.map(s=>s.engine); answer.paused_save.push(result);
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
  const backup = await t.storage.backup(); backup.sessions = [{...t.snapshot(), history: [[0, 0]], records: [], paused: false}];
  const lines = [...t.lines], [status] = await t.request('/import', {text: JSON.stringify(backup)}, 'POST');
  answer.imported = {status, history: t.history, paused: t.paused, saved: (await t.storage.get('sessions', 'live')).history,
    renewed: t.lines.every((line, side) => line !== lines[side])};
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
  answer.timed_replay = await (async () => { const v = new BrowserSession(native); v.registerEngine(entry, {turn: async history => ({moves: job.history.slice(history.length, history.length + (history.length ? 2 : 1)), value: .5})});
    await v.startMatch({players: [{engine: 'test'}, {engine: 'test'}], games: 2, max_placements: 3, clock: {mode: 'game', tc: '60+1'}}); await until(() => v.match.completed === 2);
    const g = await v.storage.get('games', v.match.results[0].id); await v.openGame(v.match.id, 1); const st = v.state();
    return {clock: g.clock, turns: g.turns.length, opened: {spec: st.clock_spec, turns: v.clockTurns.length, running: st.clock?.running ?? null}}; })();
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
