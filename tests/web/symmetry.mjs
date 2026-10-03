// Node runner for tests/test_web_engine.py: reads {cells: [[q, r]]} from stdin and writes, for web/engine/symmetry.mjs,
// {images: [transform(cells, k) for k < 12], undone: [transform(transform(cells, k), inverse(k)) for k < 12],
//  reached: [every symmetry the page's turn and mirror steps reach from 0, as compose gives it]}
import {readFileSync} from 'node:fs';
import {MIRROR, TURN_LEFT, TURN_RIGHT, compose, inverse, transform} from '../../web/engine/symmetry.mjs';

const {cells} = JSON.parse(readFileSync(0, 'utf8')), keys = [...Array(12).keys()], reached = new Set([0]);
for (const from of reached) for (const step of [TURN_LEFT, TURN_RIGHT, MIRROR]) reached.add(compose(from, step));
process.stdout.write(JSON.stringify({images: keys.map(k => transform(cells, k)),
  undone: keys.map(k => transform(transform(cells, k), inverse(k))), reached: [...reached].sort((a, b) => a - b)}));
