// One JSON request ({history, ms}) per stdin line, one JSON result ({moves, raw, ms} or {error}) per stdout line.
import { readFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import createModule from '../../web/engine/seal/engine.mjs';
import { sealTurn } from '../../web/engine/seal.mjs';

const module = await createModule({ wasmBinary: await readFile(new URL('../../web/engine/seal/engine.wasm', import.meta.url)) });
for await (const line of createInterface({ input: process.stdin })) {
  if (!line.trim()) continue;
  const { history, ms } = JSON.parse(line);
  let result;
  try {
    result = sealTurn(module, history, ms);
  } catch (error) {
    result = { error: error.message };
  }
  process.stdout.write(JSON.stringify(result) + '\n');
}
