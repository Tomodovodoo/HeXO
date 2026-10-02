/* The engines that run in this browser, as seat.mjs offers them. An engine is an object with:
 *   id          'browser:<name>', its id in the page's engine list
 *   kind, label the server registry's kind (bubble, strix, ...) and the picker's name, '<Name> (browser)'
 *   presets     {preset: budget}, the server's presets for its kind
 *   catalogue() resolves to its checkpoints or networks, the default first ([] when it has none to choose), or null
 *               when this page cannot run it (its files were not built)
 *   load(checkpoint, progress)  starts it with that checkpoint (null for the default); progress(fraction) while loading
 *   turn(history, budget, {checkpoint, analysis, signal, progress})  resolves to its turn after `history` with the
 *               evaluation fields of python/play.py evaluate (moves, value, top, proof, line, threat, ms) and the budget
 *               it spent; `analysis` is true when the result is shown as an evaluation rather than played, and
 *               aborting `signal` rejects with an AbortError
 */
import {BubbleEngine, PRESETS as BUBBLE_PRESETS} from './bubble.mjs';
import strix from './strix.mjs';

const bubbleEngine = new BubbleEngine();

const bubble = {
  id: 'browser:bubble', kind: 'bubble', label: 'Bubble (browser)', presets: BUBBLE_PRESETS,
  catalogue: async () => [],
  load: (checkpoint, progress) => bubbleEngine.load(progress),
  async turn(history, budget, options) {
    const result = await bubbleEngine.turn(history, budget, options);
    return {...result, simulations: budget.simulations, solver_nodes: result.solved ? budget.solver_nodes : 0};
  },
};

export const ENGINES = [bubble, strix];
