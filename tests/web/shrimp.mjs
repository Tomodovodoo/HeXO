// Node runner for tests/test_web_shrimp.py: reads one JSON job from stdin, writes one JSON answer to stdout.
// {kind: 'turns', profile, cases: [{history, visits}], rows: {key: {value, moves_left, logits}}}
//   plays each turn with the recorded network answers (key: sha1 of a row's int32 coords and float32 features)
// {kind: 'network', profile, bias, model, cases} plays each turn with the exported graph under ONNX Runtime Web (WebAssembly)
// -> [{moves, stones: [{action, value, visits}]}] per case, the cases in order on one engine (one game key sequence)
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {ShrimpSearch, loadModule} from '../../web/engine/shrimp/search.mjs';
import {ShrimpNetwork} from '../../web/engine/shrimp/network.mjs';

const job = JSON.parse(readFileSync(0, 'utf8'));
const engine = await ShrimpSearch.create(await loadModule(new URL('../../web/engine/shrimp/shrimp.wasm', import.meta.url).href), job.profile);

function recorded(rows) {
  const values = [], movesLeft = [], logits = [];
  for (let i = 0; i < rows.count; i++) {
    const [start, end] = [rows.offsets[i], rows.offsets[i + 1]];
    const key = createHash('sha1').update(rows.coords.subarray(2 * start, 2 * end)).update(rows.features.subarray(15 * start, 15 * end)).digest('hex');
    const row = job.rows[key];
    if (!row) throw new Error(`Row ${key} was not evaluated by the driver`);
    if (row.logits.length !== rows.legal[i]) throw new Error(`Row ${key} has ${rows.legal[i]} legal cells, the driver's ${row.logits.length}`);
    values.push(row.value);
    movesLeft.push(row.moves_left);
    logits.push(...row.logits);
  }
  return {values, movesLeft, logits};
}

let evaluate = recorded;
if (job.kind === 'network') {
  const ort = await import(new URL('../../web/engine/ort/ort.wasm.min.mjs', import.meta.url).href);
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = 'error';
  const session = await ort.InferenceSession.create(new Uint8Array(readFileSync(job.model)), {executionProviders: ['wasm']});
  const network = new ShrimpNetwork(ort, session, {model_version: 'test', bias: job.bias}, 'wasm', 1);
  evaluate = rows => network.evaluate(rows);
}
const answer = [];
for (const {history, visits} of job.cases) {
  const {moves, stones} = await engine.turn(history, visits, {evaluate});
  answer.push({moves, stones: stones.map(({action, value, visits}) => ({action, value, visits}))});
}
process.stdout.write(JSON.stringify(answer));
