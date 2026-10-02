// Node runner for tests/test_web_native.py: reads [{history, ms, depth}] from stdin and writes Native's turn for each,
// searched in that order by one module, as [{moves, score, depth, nodes, elapsed_ms}].
import {readFileSync} from 'node:fs';
import {NativeSearch} from '../../web/engine/native/search.mjs';

const native = await NativeSearch.create();
const cases = JSON.parse(readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(cases.map(({history, ms, depth}) => native.turn(history, ms, depth))));
