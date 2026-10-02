// Strix (web/engine/strix/strix.wasm) with the network at argv[2]: one JSON request {history, simulations}
// per stdin line, one JSON turn per stdout line.
import { readFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { loadStrix } from '../../web/engine/strix/core.mjs';

const strix = await loadStrix(await readFile(new URL('../../web/engine/strix/strix.wasm', import.meta.url)));
strix.load(await readFile(process.argv[2]));
for await (const line of createInterface({ input: process.stdin })) {
  if (!line.trim()) continue;
  const { history, simulations } = JSON.parse(line);
  process.stdout.write(JSON.stringify(strix.turn(history, simulations)) + '\n');
}
