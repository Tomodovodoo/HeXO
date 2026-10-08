/* What the browser engines can expect of this device, decided once per page. */
import {probe} from './network.mjs';

/** True when the page can run ONNX Runtime on WebGPU; without it the neural engines run on WebAssembly. */
export const WEBGPU = (await probe()).provider === 'webgpu';

/** The preset a neural engine starts at: its server-sized default on WebGPU, the lightest one on the CPU. */
export const NEURAL_PRESET = WEBGPU ? 'standard' : 'lightning';

/** Records that `entry` searched a turn of `stones` stones at `preset` in `ms`. The page shows `entry.pace` (ms per
 * two-stone turn at each preset, a one-stone turn counted twice) next to the strength slider; each turn moves it
 * halfway to the new time. */
export function notePace(entry, preset, ms, stones) {
  if (!stones || !entry.presets[preset]) return;
  const turn = ms * 2 / stones, old = entry.pace?.[preset];
  entry.pace = {...entry.pace, [preset]: old == null ? turn : (old + turn) / 2};
}
