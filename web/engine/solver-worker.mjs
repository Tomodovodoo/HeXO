/* The tactical solver in its own worker, so a cancelled query ends with the worker. Messages: {id, history, options}. */
import {loadTactical, proofAnswer} from './tactical.mjs';

const solver = loadTactical(new URL('tactical.wasm', import.meta.url).href);

onmessage = async ({data}) => {
  try {
    const ready = await solver;
    if (data.prepare) { postMessage({id: data.id, ready: true}); return; }
    if (data.request) {
      const {history, ...options} = data.request;
      postMessage({id: data.id, answer: proofAnswer(ready.history(history, {...options, cancel: data.cancel}), data.request)});
    } else postMessage({id: data.id, result: ready.history(data.history, data.options)});
  } catch (error) {
    const result = {status: 'UNKNOWN', native_verified: false, moves: [], reason: String(error.message || error)};
    postMessage(data.request ? {id: data.id, answer: proofAnswer(result, data.request)} : {id: data.id, result});
  }
};
