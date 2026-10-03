/* Searches in a worker await their network through microtasks alone on WebAssembly, which would hold a cancel
 * message back until the turn ends; awaiting nextTask() between batches lets the worker read it. */

/** Resolves after the tasks already queued (a worker's messages, timers) have run. */
export function nextTask() {
  return new Promise(resolve => {
    const channel = new MessageChannel();
    channel.port1.onmessage = () => { channel.port1.close(); resolve(); };
    channel.port2.postMessage(null);
  });
}
