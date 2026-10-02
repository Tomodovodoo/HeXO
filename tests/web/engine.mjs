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
}
process.stdout.write(JSON.stringify(answer));
