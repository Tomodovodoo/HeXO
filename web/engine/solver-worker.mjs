/* The tactical solver in its own worker, so a cancelled query ends with the worker. Messages: {id, history, options}. */
import {loadTactical} from './tactical.mjs';

const solver = loadTactical(new URL('tactical.wasm', import.meta.url).href);

onmessage = async ({data}) => {
  try {
    postMessage({id: data.id, result: (await solver).history(data.history, data.options)});
  } catch (error) {
    postMessage({id: data.id, result: {status: 'UNKNOWN', native_verified: false, moves: [], reason: String(error.message || error)}});
  }
};
