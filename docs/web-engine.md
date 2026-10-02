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

## GitHub Pages

`.github/workflows/pages.yml` builds the bundle on every push to `main` (ONNX Runtime, and the model exported from the
newest release) and publishes `web/` as the site, so the play page opens at <https://tomodovodoo.github.io/HeXO/>.
It runs once the repository is public and Settings > Pages > Build and deployment > Source is set to GitHub Actions.
Without a play server the page answers its own game requests (`web/engine/offline.mjs`): Bubble (browser) holds seat O
and the analysis, server-only controls (review, tournaments, import and export) are hidden, and `web/coi-sw.js`, scoped
to the site's path, adds the cross-origin isolation headers after one reload.

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
| `web/engine/seat.mjs` | Play page hook: the browser engines (`ENGINES`) as seats and analysis |
| `web/engine/native.mjs`, `native-worker.mjs` | Native (browser): page API and worker |
| `web/engine/native/` | `native.wasm` and its loader, built from `src/hexo.cpp`; `search.mjs` runs a turn |
| `web/engine/strix.mjs`, `strix-worker.mjs` | Strix (browser): page API and worker |
| `web/engine/strix/` | `strix.wasm` and its loader `core.mjs`, built from `tools/strix_web`; the network when built |
| `web/engine/offline.mjs` | The play server's game requests answered in the page, for static hosting |
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

## Native (browser)

Native, the handwritten engine in `src/hexo.cpp`, also runs in the page. Pick **Native (browser)** for a seat or
the analysis. `build_web.py wasm` compiles `hexo.cpp` with the same em++ flags as `gumbel.wasm` into
`web/engine/native/native.wasm` (204 KB) and its loader `native.mjs` (10 KB). `build.json` records both. A worker
(`native-worker.mjs`) keeps the module between turns, as the server keeps its search child, so a proven winning plan
carries over to the next turn. The worker fetches `native.wasm` once and keeps it in the Cache API under the digest
`build.json` records for it.

A turn is `hx_search` with the server's settings: depth 12, width 16, and the preset's milliseconds (lightning 100,
quick 250, standard 1,000, strong 3,000, deep 10,000, dangerous 60,000). The C++ clock is `performance.now()` in
the worker. The search cannot stop part way inside the module, so cancelling a move ends the worker and the next
turn starts a fresh one. The server kills its search child the same way. As analysis, Native shows its turn as the
top move and the line. It is the analysis engine when the play server has none (no Bubble model). The bar shows the mover's odds as logistic(score / 1000), and 1 or 0 once the search proves a
win or a loss. The score is a heuristic, not a probability.

Parity (`tests/test_web_native.py`): with the deadline out of reach, the wasm build and the native library pick the
same turn with the same score and completed depth in 169 depth-bounded searches (depth 2, 3 and 4) of 75 recorded
positions. Node counts differ in 4 of them, by at most 1.2%. `std::sort` orders equal-scored turns differently in
libc++ (Emscripten) and libstdc++ (MinGW); with two `std::sort` calls in `turns()` replaced by `std::stable_sort`
the counts match as well. In Chrome 152 on the Ryzen 9 5900X the browser searches 78,000 nodes per second from a 7-stone
position, against 106,000 for the server library. At 100, 1,000 and 3,000 ms both reached the same depth and chose
the same turn.

What the bundle contains: `src/hexo.cpp` (this repository, MIT), Emscripten's runtime and loader (MIT or NCSA),
and libc++ and libc++abi (Apache 2.0 with LLVM exception).

## Strix (browser)

Strix ([SootyOwl/hexo-strix](https://github.com/SootyOwl/hexo-strix) at `5a771e5`) with the network hexo.tyto.cc
lists as `pulsatrix-10-best` also runs in the page as **Strix (browser)**, for a seat or the analysis, with the
server's presets in simulations per placement (lightning 2, quick 8, standard 64, strong 128, deep 512, dangerous
4,096). hexo.tyto.cc runs Strix the same way: the Rust network (`hexo-infer`) and Gumbel search (`hexo-mcts`)
compiled to WebAssembly in a worker that a cancel terminates. Here `tools/strix_web` wraps the same calls as the
server's wrapper `tools/strix_learned` (4 root actions, `c_visit` 50, `c_scale` 1, no Gumbel noise, the same request
and Strix's mirrored frame) as a `wasm32-wasip1` library with SIMD, run through the WASI shim of `tactical.mjs`.
`build_web.py strix` builds `web/engine/strix/strix.wasm` (676 KB) and records it in `build.json`; `build_web.py wasm`
builds it with the rest.

The network is not committed. `python tools/build_web.py strix-network` downloads the file pinned in
`tools/engines.json` (2.8 MB, checked against its SHA-256) into `web/engine/strix/` with `networks.json`; without
that file the page does not offer Strix. The worker keeps `strix.wasm` and the network in the Cache API under their
digests. Its licence is unstated: hexo.tyto.cc serves it publicly, and neither the site nor the repository grants
permission to redistribute it. The Pages workflow therefore fetches it only when the repository variable
`PUBLISH_STRIX_NETWORK` is `true`. The engine code is MIT (`web/engine/strix/LICENSE-hexo-strix.txt`); the Rust
crates it links (serde, serde_json, rand, safetensors, rayon, rustc-hash) are MIT or Apache 2.0.

A search runs on one thread. On the Ryzen 9 5900X under node 24, from a 19-stone position, a turn takes 1.0 s at
lightning, 3.0 s at quick, 23 s at standard, 44 s at strong and 178 s at deep, against 0.8, 1.9, 10.7, 19.3 and 76 s
for the server wrapper, which evaluates a batch on all cores. From a 5-stone position standard takes 6 s.

Parity (`tests/test_web_strix.py`, against `tools/strix_learned_adapter.py`): the wasm build plays the server's turn in
all 156 recorded positions at 8 simulations, and in 40 of 46 at 64. The network's outputs differ from the native
build's in the last bits of a float (about 1e-9; the network calls `exp` and `tanh`, which the two math libraries
round differently), and that decides near-equal choices: five of the six differences are the position after the
origin, where the browser plays the mirror image of the server's turn, and one is a different stone in an 8-stone
position. The 19-stone timing position above also got a different second stone at 128. The test checks exact turns
on positions without such ties, and the origin position up to its symmetry.
