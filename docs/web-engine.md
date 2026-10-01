# Browser engine

Bubble runs entirely in the browser: the network under ONNX Runtime Web (WebGPU, else WebAssembly), the native
Gumbel search (`src/gumbel.cpp`) and the tactical solver (`tools/tactical`) compiled to WebAssembly, in Web Workers.
On the play page pick **Bubble (browser)** for a seat or for analysis; presets are lightning 8/2048, quick 32/2048,
standard 128/32768, strong 512/131072, deep 2048/524288 and dangerous 65536/4000000 (simulations / solver nodes).

```sh
python tools/build_web.py wasm                      # gumbel.wasm, tactical.wasm (em++ and cargo, committed)
python tools/build_web.py ort                       # onnxruntime-web 1.30.0 into web/engine/ort
python tools/build_web.py model                     # newest release, or --checkpoint path/to/ema.pt
python -m unittest tests.test_web_engine tests.test_web_tactical
```

`wasm` needs [emsdk](https://emscripten.org/docs/getting_started/downloads.html) (`em++` on PATH or `--emxx`) and
`rustup target add wasm32-wasip1`; `web/engine/build.json` binds the committed artefacts to their sources and the
tests fail when they are stale. `ort` and `model` write ignored files; a static deployment runs all three.

## Layout

| File | Role |
|---|---|
| `python/export_web.py` | HexNet to ONNX (opset 17, dynamic batch and crop size, fp32 and fp16) with a parity report |
| `web/engine/bubble.mjs` | Page API: `BubbleEngine.load/turn/search/evaluate/bench`, `PRESETS`, `isolate()` |
| `web/engine/worker.mjs` | Engine worker: a turn exactly like `python/play.py evaluate` |
| `web/engine/network.mjs` | Device probe, cached downloads (Cache API), sessions, batched evaluation |
| `web/engine/encode.mjs` | `hexcrop.encode` and `hexnet.LineFeatures` |
| `web/engine/search.mjs` | `hxg_*` driver: the `neural_search` loop, evaluation cache, root statistics |
| `web/engine/tactical.mjs`, `solver-worker.mjs` | Solver with a WASI shim, in a worker that a cancel terminates |
| `web/engine/proof.mjs` | Certificate walk for the winning line |
| `web/engine/seat.mjs` | Play page hook: the browser seat and analysis engine |
| `web/coi-sw.js` | Cross-origin isolation on static hosts (`isolate()`), for WebAssembly threads |

## Device choice

WebGPU with `shader-f16` loads both graphs, times a batch of 16 at crop 24 under each and keeps fp16 only when it is
1.25 times faster; fp32 reproduces the server's evaluations, fp16 does not. Without WebGPU it runs the WebAssembly
build with SIMD, and threads when the page is cross-origin isolated (`play.py` sends COOP/COEP; static hosts use
`isolate()`). The search batch stays 16, as on the server, so a browser search is the server's search. A larger
batch raises WebGPU throughput (64 leaves in 63 ms against 16 in 26 ms) but changes which leaves are searched;
`BubbleEngine.turn` takes `batch_size` in its budget for that. WebAssembly time grows linearly with the batch.

The ONNX Runtime binary (27 MB WebGPU, 14 MB WebAssembly) and the model (4.6 MB fp32, 2.3 MB fp16) are fetched
once and kept in the Cache API under their version and SHA-256. GitHub release downloads send no CORS headers, so
the model is served from the same origin: `build_web.py model` downloads the release and exports it.

## Parity

| Check | Result |
|---|---|
| ONNX vs PyTorch reference path, 47 recorded positions (CPU) | fp32 max abs 3.6e-5 policy, 9.1e-6 value; fp16 0.069 / 0.013 |
| Browser evaluation vs PyTorch, same positions | WebGPU fp32 3.8e-5, WebAssembly 4.0e-5, WebGPU fp16 0.32; argmax 47/47 for all |
| Encoder and line features vs `hexcrop` and `LineFeatures` (node) | identical |
| Search vs native library, same seed and evaluations, 13 cases up to 512 simulations (node) | identical actions, visits, policy (1e-12) |
| Solver wasm vs native library, 8 queries (node) | identical status, moves, certificate, nodes used |
| Full turn, standard preset, two positions (WebGPU fp32, WebAssembly vs server CPU) | identical moves, value and top moves |

## Speed

RTX 3070 Ti and Ryzen 9 5900X, Chrome 154 on Windows, while the training run kept the GPU about 93% busy. The server
numbers are `play.Bubble` with the same search on the same machine in the same session. One search from a 9-stone
position, root samples 16, batch 16, no solver, median of three (`web/engine/bench.html`).

Forward latency, ms per batch (crop 24 / crop 32):

| Device | Batch 1 | Batch 16 | Batch 64 |
|---|---|---|---|
| WebGPU fp32 (fp16 within 3%) | 17 / 14 | 26 / 36 | 63 / 105 |
| WebAssembly, 8 threads | 25 / 31 | 289 / 389 | 966 / 1631 |
| WebAssembly, 1 thread | 78 / 130 | 1279 / 2003 | 4870 / 8115 |

One search, ms (placements per second):

| Preset | WebGPU fp32 | WebAssembly 8 threads | WebAssembly 1 thread | Server CPU, 8 threads | Server CUDA (bf16) |
|---|---|---|---|---|---|
| lightning 8 | 44 (181) | 144 (56) | 649 (12) | 194 (41) | 386 (21) |
| quick 32 | 111 (289) | 508 (63) | 2105 (15) | 458 (70) | 1090 (29) |
| standard 128 | 599 (214) | 1912 (67) | 6673 (19) | 2056 (62) | 5978 (21) |
| strong 512 | 1384 (370) | 4420 (116) | 17354 (30) | 3270 (157) | 12864 (40) |
| deep 2048 | 13885 (147) | 15657 (131) | not run | 11159 (184) | 47837 (43) |

Every run chose the same first stone in every column. Over 95% of a browser search is network time. The shared GPU
makes the WebGPU and CUDA columns noisy (WebGPU deep took 11 to 18 s across runs): with the training run on the card
a one-operator graph already takes 6 ms, so a batch costs mostly dispatch and time slicing, not arithmetic. Graph
capture saved 15% at batch 16 and is not used. WebAssembly with one thread is what a page without cross-origin
isolation gets.
