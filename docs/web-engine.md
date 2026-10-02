# Browser engine

Bubble runs entirely in the browser: the network under ONNX Runtime Web (WebGPU, else WebAssembly), the native
Gumbel search (`src/gumbel.cpp`) and the tactical solver (`tools/tactical`) compiled to WebAssembly, in Web Workers.
On the play page pick **Bubble (browser)** for a seat or for analysis; presets are lightning 8/2048, quick 32/2048,
standard 128/32768, strong 512/131072, deep 2048/524288 and dangerous 65536/4000000 (simulations / solver nodes).

```sh
python tools/build_web.py wasm                      # gumbel, native, tactical and six wasm (committed)
python tools/build_web.py ort                       # onnxruntime-web 1.30.0 into web/engine/ort
python tools/build_web.py model                     # newest release, or --checkpoint path/to/ema.pt
python tools/build_web.py six                       # Six's networks (pinned) into web/engine/six/networks
python -m unittest tests.test_web_engine tests.test_web_tactical tests.test_web_six
```

`wasm` needs [emsdk](https://emscripten.org/docs/getting_started/downloads.html) (`em++` on PATH or `--emxx`) and
`rustup target add wasm32-wasip1`; `web/engine/build.json` binds the committed artefacts to their sources and the
tests fail when they are stale. `ort`, `model` and `six` write ignored files; a static deployment runs all four.

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
| `web/engine/engine-worker.mjs` | `EngineWorker`: a browser engine's worker from the page, with loading, cancellable calls and the one-thread retry |
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
| `web/engine/shrimp.mjs`, `shrimp-worker.mjs` | Shrimp (browser): page API and worker |
| `web/engine/shrimp/` | `shrimp.wasm` (built from `tools/shrimp_web`), `search.mjs` (the driver's turn), `network.mjs` (the graph's inputs) |
| `web/engine/seal.mjs`, `seal-worker.mjs` | Seal (browser): page API and worker; `seal/` holds its build |
| `web/engine/six.mjs`, `six-worker.mjs`, `six/search.mjs` | Six (browser): page API and seat entry, worker, and Six's search driving its network |
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

## Shrimp (browser)

Shrimp is Cmiller132/hexo-bot's main_7 network (epoch 18, 8.1M parameters) with hexo-bot's own Gumbel search, as
the server runs it through Six's Shrimp driver. Pick **Shrimp (browser)** for a seat or the analysis. Its presets are
the server entry's visits per stone: lightning 16, quick 32, standard 128, strong 512, deep 1,024, dangerous 4,096.

What runs where:

- The search is hexo-bot's Rust, compiled to `web/engine/shrimp/shrimp.wasm` (380 KB) from `tools/shrimp_web`. The
  tree, the threat-space search, the featurizer and the evaluation cache are hexo-bot's files at `6251fc6`, vendored
  unchanged under `tools/shrimp_web/vendor/hexo-bot`. `src/search.rs` is the driver's `ShrimpMctsSession.search`
  with the evaluator call turned into a pause: `sh_step` returns when leaves need the network and `sh_fulfill`
  takes the answers. Selection, early stop, LCB and the tactical guard are hexo-bot's functions.
- `shrimp/search.mjs` plays a turn the way the driver does: the history mirrored into Six's frame, the first stone
  moved to the origin, one search per stone with seed 5003 + ply, the second stone reusing the first one's tree, and a
  new game key per turn. The server sends `newgame` before every turn and runs one driver process per preset, so the
  worker keeps one game counter per preset; a cancelled turn resets it, as the server restarts the driver.
- The network is `shrimp-fp32.onnx` (29 MB) under ONNX Runtime Web, WebGPU when the device has it, else WebAssembly.
  `tools/shrimp_web/export.py` writes it from the pinned weights: the evaluator's CPU forward with the moves-left
  head and the value decode. The page builds the inputs (`shrimp/network.mjs`): node features, the hex
  convolution's gather rows and each attention pair's bias row, so the graph has no integer arithmetic left for the
  WebGPU provider to hand back to the CPU. With the pair index computed in the graph, a batch took 1.2 s on WebGPU
  whatever its size; built in the page it takes 46 ms for 16 positions.

`build_web.py shrimp` downloads the weights from hexo-bot's Git LFS at their pinned SHA-256 (or takes
`--shrimp-weights`) and exports them into `web/engine/shrimp/model/` (ignored) with a manifest holding the search
profile from the pinned `shrimp_main_7.toml` and the graph's parity. The Pages workflow runs it.

Parity (`tests/test_web_shrimp.py`, which needs the server's Shrimp installed in `models/`): on 10 turns from
openings, the tactical fixtures and recorded middle games, at 16 to 64 visits, the browser search fed the driver's
own network answers plays the driver's stones with bit-identical root values and visit counts. The exported graph
under ONNX Runtime Web (WebAssembly, in node) plays the driver's stones on the first 6 of those turns. On the
driver's rows the graph's logits are within 1e-4 of PyTorch's; on synthetic rows the export reports 7.6e-6 for
the policy, 0 for the value and 1.9e-5 for the moves left.

Seconds per two-stone turn from a 9-stone position, Chrome in the Claude desktop pane on the RTX 3070 Ti and Ryzen 9
5900X, the GPU shared with the training run. The server is the driver with PyTorch on two CPU threads.

| Preset | WebGPU | WebAssembly, 1 thread | Server CPU |
|---|---|---|---|
| lightning 16 | 0.2 | 5.0 | 1.8 |
| quick 32 | 0.3 | 11.3 | 4.7 |
| standard 128 | 1.0 | | 16.5 |
| strong 512 | 3.5 | | 61 |

WebGPU and WebAssembly chose the same stones. The pane does not start ONNX Runtime's thread workers, so threaded
WebAssembly is not measured; as with Bubble, a stalled start falls back to one thread after 20 s.

What the bundle contains: hexo-bot's Rust and the weights (MIT, Colton Miller), the crates they use (ahash, half,
serde, thiserror; MIT or Apache 2.0) and ONNX Runtime Web (MIT). Mantis Shrimp (Cmiller132/Hexo-Shrimp-Bot) is not
included: it has no licence and no published weights.

## Seal

Seal is the alpha-beta bot by Ramora0 ([HexTicTacToe](https://github.com/Ramora0/HexTicTacToe), revision `3474edb`),
the community's reference bot. **Seal (browser)** in the picker plays a seat or the analysis with the server's
presets: lightning 50, quick 100, standard 500, strong 2000, deep 8000 and dangerous 30000 ms per turn.

```sh
python tools/build_web.py seal --emxx path/to/em++   # seal/engine.mjs, engine.wasm, manifest.json (ignored)
python -m unittest tests.test_web_seal
```

`build_web.py seal` downloads Seal's four headers at the revision pinned in `tools/engines.json`, checks each
against its SHA-256 and compiles them with `tools/seal_adapter.cpp`, the server's adapter, using the same em++ flags
as `gumbel.wasm`. The headers are never committed; the Pages workflow installs emsdk 6.0.10 and builds Seal on every
deployment. HexTicTacToe has no licence file; this site serves the compiled Seal regardless.

The wasm is 115 KB and its glue 10 KB. The worker fetches `seal/manifest.json`, then the wasm from the Cache API
under its SHA-256, and calls `seal_move` exactly as the server does; Seal's clock is `performance.now()` in the
worker. The search blocks the worker, so a cancel terminates it and the next turn starts a new one, which loses
Seal's transposition table. The page cuts the answer to the stones left in the turn and stops at a winning stone,
as `play.checked_turn` does. As analysis, Seal shows its first stone as the top move and both stones as the line.
`seal_move` returns no score, so the evaluation bar stays empty.

Seal searches to a clock and adds one random far candidate at the root, so the same position can get different
turns. `tests/test_web_seal.py` compares turns at 1000 ms on 11 recorded positions where the server library gave
one answer over repeated runs at 300 and 1500 ms; of 32 recorded positions, every one of the 16 with a stable
server answer got the same turn in the browser build. `.github/workflows/web.yml` builds the native library and the
wasm and runs the test. On the Ryzen 9 5900X in node 24 the wasm searches 70 to 90% of the native library's nodes
per second (525,000 against 662,000 a second on a three-stone position at 2 s) and reaches the same depth or one
less.

What the bundle contains: Seal's `engine.h`, `types.h` and `pattern_data.h` (no licence), ankerl's
`unordered_dense.h` as vendored in HexTicTacToe (MIT), `tools/seal_adapter.cpp` (this repository, MIT),
Emscripten's runtime and loader (MIT or NCSA), and libc++ and libc++abi (Apache 2.0 with LLVM exception).

## Six (browser)

**Six (browser)** is CixMango/Six's own browser build, as its site playsix.cixmango.workers.dev runs it: Six's MCTS
and threat solver (`tools/six`, its `engine/src` and `engine/web/web_bot.cpp` at `fbc7087`, MIT) compiled to
WebAssembly with Six's flags (`web/engine/six/six.wasm`, 435 kB, committed through `build.json`), with each batch of
positions sent out to ONNX Runtime Web. HeXO adds three exports to `web_bot.cpp`: `six_stop`, so a cancel ends a
turn at its next batch, `six_score` and `six_nodes`. Both seats and the analysis panel can use it. As analysis it
shows its turn: the first stone as the top move, both stones as the line, the win chance from its score (100% when
its threat solver proves a win, whose distance Six does not report) and the positions searched; the presets give it the server's
Six protocol nodes (lightning 1,500 to dangerous 2,000,000) and the network select lists the site's networks, newest
first. It plays like the server's Six (`python/six_engine.py` driving `sixengine`): Six's default search settings,
radius 8, mirrored coordinates, `go nodes N` with no time limit, and the tree kept while the game continues.

`build_web.py six` downloads Six v1.3.3's smallest archive (macOS, 90 MB) for gen-0455 and gen-0400, 0300, 0200 and 0100 from Six's
`networks` release, checks each against the SHA-256 pinned in `tools/build_web.py`, and rewrites each graph as Six's
`engine/web/prepare_net.py` does for its site, so WebGPU runs all of it: masks become numbers, And becomes Mul, the
policy reshape gets a fixed shape and the unused opponent head is dropped. The build fails when the rewritten graph's
outputs differ from the original's under onnxruntime. The networks are 24 MB each (gen-0100 18 MB), fp32, and kept in
the Cache API under their SHA-256 after the first load. Six's site runs an fp16 export of its newest network; the
releases publish only fp32 graphs, so this build stays fp32. The engine is MIT, and so are the networks (the release
notes of `networks` and Six's `NOTICE.txt`).

`tests/test_web_six.py` plays recorded positions with the browser search in node (ONNX Runtime Web's WebAssembly
build, one thread) and with `sixengine --cpu` through `SixEngine`, at 48 and 160 nodes, and three turns of one game on
one tree: the moves are identical. It needs `models/six` (the play page's Six download) and takes about four minutes,
since one thread evaluates about 2.7 positions per second. The standard preset (30,000 nodes) from the first stone gave the
same two stones on WebGPU in the browser (102 s) and from `sixengine --cpu` on the Ryzen 9 5900X (720 s).

Speed on the RTX 3070 Ti (otherwise idle) in the Claude desktop browser pane, WebGPU, from one stone: lightning 1.3 s,
quick 22 s, standard 102 s. The search itself and its threat solver run on one thread in the worker, so turns slow
down as the tree grows. On WebAssembly a position costs about 0.4 s on one thread, so without WebGPU only lightning
is practical.

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
digests. The network's licence is unstated in the repository, and its author allows this use; the Pages workflow fetches it
when the repository variable `PUBLISH_STRIX_NETWORK` is `true`, which it is. The engine code is MIT (`web/engine/strix/LICENSE-hexo-strix.txt`); the Rust
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
