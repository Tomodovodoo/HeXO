# Browser play

## Comparison

Looked at on 2026-09-30: Strix at hexo.tyto.cc, Six at playsix.cixmango.workers.dev and its web source, Mantis
Shrimp's play deck from its repository, and the official HeXO client for the stone colours.

| | Strix | Six | Mantis Shrimp | Bubble before | Bubble after |
|---|---|---|---|---|---|
| Theme | dark only | dark default, five other themes including light | dark only | light only | dark only |
| First player (X) | orange | yellow `#f6d04a` | blue `#5a9dff` | blue `#226b95` | amber `#fbbf24` |
| Second player (O) | blue | light blue `#87d1f7` | red `#f0605a` | red `#c35539` | sky `#38bdf8` |
| Stones | filled hex | inset hex with glow | filled hex | filled hex | filled hex, landing animation |
| Engine runs | in the browser | browser worker (WASM, ONNX on WebGPU) or native process | Python server | inside the HTTP request | background worker thread with job ids |
| Page while thinking | usable | usable, progress in positions per second | usable, stale reads dropped | frozen, every control disabled | usable, progress per job, cancel |
| Engines per side | Strix versions against a human | Six levels, bot against bot | random, checkpoints, SealBot | human against Bubble or native | Bubble, native, Seal, Strix, Six, Shrimp per side from a models folder, bot against bot live |
| Strength | Instant, Quick, Standard, Strong, Deep | levels by positions per turn (6k to 135k) or time | argmax, sample, improved policy | simulations and proof nodes lists | Quick, Standard, Strong, Deep, custom, per side |
| Analysis | vertical eval bar, top 5 moves with scores, shaded candidates | win-chance bar, best turn as ghost stones | value, entropy, candidate table, heat maps | on click: win chance, top 5, proof, threat | continuous: eval bar, top moves, best turn, proof and threat, saved |
| Forced wins | check on request, winning line overlay | proven scores, threat outlines | none | solver proof and threat | solver proof and threat, winning line |
| Review labels | best, good, mistake, blunder per turn | best to blunder bands, missed win, allowed win, rating 1 to 100 | none | none | best, good, inaccuracy, mistake, blunder, missed and allowed forced win, per turn |
| Better move | suggested line and difference | outlined stones, 4-turn follow-up | none | none | better turn and its line on the board |
| Value graph | timeline with label dots | win-chance graph | per-ply trace charts | list of saved win chances | graph with label marks, click to jump |
| Stepping | timeline, arrows | arrows, Home, End | arrows, Shift for 10, Home, End | none, undo only | arrows, Home, End, click on list or graph |
| Retry from here | play any empty hex in analysis | yes | no | no | yes |
| Import | HTTTX, sandbox link | HTTTX, HeXO links, position string, replay file | none | none | HTTTX, replay file |
| Export | HTTTX | HTTTX, position string, replay file | none | HTTTX | HTTTX, replay file |
| Saved evaluations | none across visits | none across visits | none | JSON file, whole file rewritten on each save | append-only JSON lines per run, indexed in memory by position and settings, deepest shown |
| 1366x768 | panel covers part of the board, nested scrolling | board shrinks to a strip, page scrolls | three fixed columns | sidebar scrolls | fixed three-column grid, no page scroll |
| Phone | board with a bottom sheet | toolbar clipped, page scrolls | stacked columns | page scrolls | board over a tabbed panel, no page scroll |

The official client (`board-renderer/src/themes/darkColors.ts`) draws player 0, the X who opens at the origin, in amber
`#fbbf24` and player 1, O, in sky blue `#38bdf8` on `#0f172a`. Strix and Six follow the same warm-first, blue-second
order. Bubble had it the other way round.

## Bubble in the browser

Feasibility of a static page that runs Bubble with no install: HexNet on WebGPU through ONNX Runtime Web, the search
and the solver in WebAssembly. Measured on CPU on 2026-10-01 with `main/122500`; GPU speeds are estimates.

| Part | Finding | Work left |
|---|---|---|
| ONNX export | Fails as written: `as_strided` in LineConv breaks the dynamo exporter, and the legacy exporter produces wrong logits (off by up to 188). Rewritten at export time (LineConv as a depthwise 11x11 conv, the clamp as `Relu * Clip`, masks as `Clip`/`Mul`), it matches torch to 2e-5. One file covers all crop buckets with dynamic batch and size. | export script and parity tests |
| WebGPU ops | 295 nodes after optimisation, all in ONNX Runtime Web's WebGPU operator list; only `Shape` runs on the CPU. About 150 nodes are small line-feature slices, which means many small dispatches. | move line features into the encoder later |
| Model size | fp32 4.7 MB (3.5 MB gzipped), fp16 2.4 MB (1.8 MB gzipped, policy logits within 0.06) | none |
| Encoder | `hexcrop.encode` (legal set, 12 symmetries, bucket or far mode, 8 planes) has no C++ version; about 150 lines to port | C++ in the WASM shim, golden fixtures from Python |
| Search | `src/gumbel.cpp` and `src/hexo.cpp` compile unchanged with Emscripten: 93 KB wasm (42 KB gzipped). The tree costs 0.1 ms per simulation. JS drives the request and fulfill loop, so no ASYNCIFY. Average network batch is about 4. | port the Python search coordinator to JS, int32 wrappers for int64 arguments |
| Solver | The Rust crate builds for `wasm32-unknown-unknown` (255 KB gzipped). It starts a thread and reads `Instant::now()`, which both fail there. | a wasm32 path that runs in its own worker, cancelled by terminating it |
| Runtime | ONNX Runtime Web with WebGPU is 6.7 MB gzipped (over 25 MB raw, so served from a CDN as Six does) | COOP and COEP headers for WASM threads |

Speed, network evaluations per second:

| Backend | 24x24 crop | 32x32 | 48x48 |
|---|---|---|---|
| torch CPU, 2 threads | 36 to 57 | 28 | 13 |
| ONNX Runtime CPU, 2 threads | 78 | 45 | 22 |
| ONNX Runtime Web WASM, 2 threads (Node) | 19 | 10 | 6 |
| WebGPU fp16, estimated | 300 to 500 discrete, 100 to 150 integrated | | |

Seconds per two-stone turn in the browser:

| Simulations per stone | Discrete GPU | Integrated GPU | WASM fallback |
|---|---|---|---|
| 32 | 0.2 | 0.5 | 4 |
| 128 | 0.7 | 2 | 17 |
| 512 | 3 | 8 | 70 |
| 2048 | 12 | 30 | too slow |

The page would weigh about 9 MB gzipped: model 1.8 MB, runtime 6.7 MB from a CDN, search 47 KB, solver 255 KB.
Six ships the same shape (435 KB engine wasm, onnxruntime-web, fp16 when the adapter has `shader-f16`). Strix
ships a 785 KB wasm with its network, search and solver on the CPU and no ONNX.

Order: export script with parity tests, encoder in C++, Emscripten build and JS coordinator tested in Node against
the Python search, browser worker with WebGPU and WASM fallback, solver worker, static hosting.
