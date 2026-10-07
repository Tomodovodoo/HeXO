// Node runner for tests/test_web_drip.py: reads [{history, ms, depth}] from stdin and writes Drip's turn for each,
// searched in that order by one module, as [{moves, score, depth, nodes, elapsed_ms}].
import {readFileSync} from 'node:fs';
import {DripSearch} from '../../web/engine/native/search.mjs';

const drip = await DripSearch.create();
const cases = JSON.parse(readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(cases.map(({history, ms, depth}) => drip.turn(history, ms, depth))));
