// One JSON request ({history, ...options} of history()) per stdin line, one JSON result per stdout line.
import { readFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { loadTactical } from '../../web/engine/tactical.mjs';

const tactics = await loadTactical(await readFile(new URL('../../web/engine/tactical.wasm', import.meta.url)));
for await (const line of createInterface({ input: process.stdin })) {
  if (!line.trim()) continue;
  const { history, ...options } = JSON.parse(line);
  process.stdout.write(JSON.stringify(tactics.history(history, options)) + '\n');
}
