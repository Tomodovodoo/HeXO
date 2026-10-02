// Node runner for tests/test_web_six.py: reads one JSON job from stdin, writes one JSON answer to stdout. Six's
// search (web/engine/six) runs its network under ONNX Runtime Web's WebAssembly build on one thread.
// {kind: 'turns', network, cases: [{history, nodes}]} -> [{moves, score}], each case on a new tree
// {kind: 'game', network, history, nodes, turns} -> [{moves, stopped}]: turns played in a row on one tree
// {kind: 'stop', network, history, nodes, after_ms} -> {moves, stopped, ms}: stop() called after `after_ms`
import {readFileSync} from 'node:fs';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {SixSearch} from '../../web/engine/six/search.mjs';

const job = JSON.parse(readFileSync(0, 'utf8'));
const ortDir = fileURLToPath(new URL('../../web/engine/ort/', import.meta.url));
const ort = await import(pathToFileURL(ortDir + 'ort.wasm.min.mjs').href);
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = {mjs: pathToFileURL(ortDir + 'ort-wasm-simd-threaded.mjs').href};
ort.env.wasm.wasmBinary = readFileSync(ortDir + 'ort-wasm-simd-threaded.wasm');
ort.env.logLevel = 'error';
const search = await SixSearch.create(ort);
search.use(await ort.InferenceSession.create(readFileSync(job.network), {executionProviders: ['wasm']}));

let answer;
if (job.kind === 'turns') {
  answer = [];
  for (const {history, nodes} of job.cases) {
    search.forget();
    const {moves, score} = await search.turn(history, nodes);
    answer.push({moves, score});
  }
} else if (job.kind === 'game') {
  const history = job.history.map(p => [...p]);
  answer = [];
  for (let i = 0; i < job.turns; i++) {
    const {moves, stopped} = await search.turn(history, job.nodes);
    answer.push({moves, stopped});
    history.push(...moves);
  }
} else if (job.kind === 'stop') {
  const start = performance.now();
  setTimeout(() => search.stop(), job.after_ms);
  const {moves, stopped} = await search.turn(job.history, job.nodes);
  answer = {moves, stopped, ms: performance.now() - start};
}
process.stdout.write(JSON.stringify(answer));
