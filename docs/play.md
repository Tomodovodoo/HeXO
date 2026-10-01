# Browser play

```sh
python python/bubble.py play                       # the newest release, downloaded into runs/play
python python/bubble.py play --model path/to/ema.pt
python python/play.py --dense-run runs/bubble --device cpu
```

Open <http://127.0.0.1:8765>. Either side is a person or an engine, so a person can play Bubble, Bubble can
play another checkpoint, and two engines play each other live while the page stays usable. Engine work runs on one
background thread as jobs; the page polls `/state` and every job can be cancelled. Cancelling an engine's move
pauses the game until Resume.

## Engines

The picker lists everything the server finds on start:

- Bubble runs: every directory in `runs/` (or `--runs`) and `models/` (or `--models`) with
  `checkpoints/<variant>/<step>/ema.pt`, plus `--dense-run`. The champion comes first, then the newest.
- Bubble exports: any other `.pt` file under `models/`.
- `models/<name>.json` for a model stored elsewhere: `{"name": "old", "kind": "bubble", "path": "../runs/x"}`.
- Native, the handwritten engine, always.
- Seal, when `build/libhexo_seal.dll` (or `.so`) is built with `-DHEXO_SEAL_SOURCE`.
- Six: a folder in `models/` holding `sixengine.exe` and `gen-NNNN.onnx` networks shows one entry per network,
  labelled with the backend it runs on: TensorRT, CUDA, DirectML or CPU, the first whose libraries are found.
  Six seats search by nodes, so a preset plays the same strength on any hardware.
- Any engine speaking the Six protocol: `models/<name>.json` with `{"name", "kind": "six", "command", "mirrored"}`,
  `command` a list of arguments.
  Set `"mirrored": true` for engines in Six's frame, where HTTTX `(q, r)` is `(q + r, -r)`.
- Strix: `models/<name>.json` with `{"name", "kind": "strix", "model": "model.safetensors"}`.

Engines you build or download yourself:

- Six: the release zip and a network from github.com/CixMango/Six releases, unpacked into `models/six/`. The
  release's engine is the DirectML build, which ships `DirectML.dll` next to `sixengine.exe`.
- Six on CUDA: Six's engine built against ONNX Runtime's GPU package, with that package's DLLs (among them
  `onnxruntime_providers_cuda.dll`) beside `sixengine.exe`, and `cudart64_12.dll` and `cudnn64_9.dll` (CUDA 12,
  cuDNN 9) on PATH or beside it; an installed PyTorch with CUDA also has them, and its `torch/lib` is added to the
  engine's PATH.
- Six on TensorRT: `onnxruntime_providers_tensorrt.dll` beside the engine and `nvinfer_10.dll` as well, for example
  from `pip install tensorrt` (its `tensorrt_libs` is added too). The first game builds the TensorRT plan beside the
  network, which takes a few minutes.
- Strix: `python tools/build_strix_learned.py <hexo-strix checkout>` (Rust and MinGW), and the public model
  from `https://hexo.tyto.cc/model.safetensors` next to the JSON.
- Mantis Shrimp: build Cmiller132/hexo-bot with its `scripts/build_native.sh` into a Six checkout's `rivals/shrimp`,
  then point a mirrored `six` entry at Six's `arena/drivers/shrimp_driver.py`, run by that build's Python. Its
  strength is `--visits`, so give each preset its own `args`, for example
  `"presets": {"quick": {"nodes": 1, "args": ["--visits", "32"]}}`.
- Seal: build with `-DHEXO_SEAL_SOURCE=<HexTicTacToe checkout>`.

| Preset | Bubble simulations per stone | Bubble solver nodes | Native and Seal ms | Six protocol nodes | Strix simulations |
|---|---|---|---|---|---|
| Quick (Q) | 32 | 2,048 | 250 and 100 | 6,000 | 8 |
| Standard (S) | 128 | 32,768 | 1,000 and 500 | 30,000 | 64 |
| Strong (St) | 512 | 131,072 | 3,000 and 2,000 | 135,000 | 128 |
| Deep (D) | 2,048 | 524,288 | 10,000 and 8,000 | 500,000 | 512 |

On a Ryzen 9 5900X with two threads, Bubble takes about 2, 3, 13 and 75 seconds per turn at these presets. Custom
(`⋯`) takes any simulations from 0 (raw policy) to 16,384, solver nodes up to 1,500,000 (0 turns the solver off; the solver gets up to a minute)
and 10 to 120,000 ms.

## Analysis and review

The analysis engine is a Bubble checkpoint with its own preset. With Auto on it evaluates every position where a
turn starts, plus any position you step to. Engine moves by the same checkpoint count as evaluations, so a game
against Bubble costs nothing extra on Bubble's turns. Review evaluates whatever is missing and labels each turn
from the mover's win probability before and after it:

| Label | Meaning |
|---|---|
| ★ best | the engine's own turn, stones in either order |
| ✓ good | lost under 5% |
| ?! inaccuracy | lost 5% to 10% |
| ? mistake | lost 10% to 20% |
| ?? blunder | lost 20% or more |
| ✗ missed | had a proven win and lost it |
| ⚑ allowed | handed the opponent a proven win |
| ! found, = kept | proved a win, or kept one |
| · lost | the opponent already had a proven win |
| ◆ | six in a row |

For inaccuracies and worse the board outlines the engine's turn and the panel lists its line. Keys: ← and →
step one stone, ↑ and ↓ one turn, Home and End, F fits the board. Retry plays on from the shown position.
Import takes HTTTX or a game file; Copy HTTTX, Game file and Evaluations export.

## Saved evaluations

Evaluations go to `play-evaluations.jsonl` in the `--dense-run` directory (else `models/`, or `--evaluations`),
one JSON line each: position, the engine (the first 16 hex digits of the weights' SHA-256 and 8 of the
solver build's, plus a readable name), simulations, solver nodes, value, the engine's turn, top five
first stones, proof winner and distance in turns, winning line and time. The file is only appended to. On start
the server copies it to `play-evaluations.jsonl.<time>.bak` and keeps the newest three copies, then indexes the
newest 200,000 evaluations in memory by position and model; the deepest one is shown, and a saved evaluation
is reused when both its simulations and its solver nodes reach the requested budget. A line is about 300 bytes
plus 8 per stone, so 200,000 evaluations from 60-stone games take about 150 MB on disk and a similar amount of
memory. Self-play and evaluator data never go here.

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
| Engines per side | Strix versions against a human | Six levels, bot against bot | random, checkpoints, SealBot | human against Bubble or native | Bubble, native, Seal, Six, Strix or Shrimp per side, found in `runs/` and `models/`, bot against bot live |
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
