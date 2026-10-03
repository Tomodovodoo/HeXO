/* What the browser engines can expect of this device, decided once per page. */
import {probe} from './network.mjs';

/** True when the page can run ONNX Runtime on WebGPU; without it the neural engines run on WebAssembly. */
export const WEBGPU = (await probe()).provider === 'webgpu';

/** The preset a neural engine starts at: its server-sized default on WebGPU, the lightest one on the CPU. */
export const NEURAL_PRESET = WEBGPU ? 'standard' : 'lightning';

/** Records that `entry` played `stones` stones at `preset` in `ms`; the page shows `entry.pace` (ms per stone at each
 * preset it has played) next to the strength slider. */
export function notePace(entry, preset, ms, stones) {
  if (!stones || !entry.presets[preset]) return;
  entry.pace = {...entry.pace, [preset]: ms / stones};
}
