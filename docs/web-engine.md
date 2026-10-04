# Browser engine

Bubble runs entirely in the browser: the network under ONNX Runtime Web (WebGPU, else WebAssembly), the native
Gumbel search (`src/gumbel.cpp`) and the tactical solver (`tools/tactical`) compiled to WebAssembly, in Web Workers.
On the play page pick **Bubble (browser)** for a seat or for analysis; presets are lightning 8/2048, quick 32/2048,
standard 128/32768, strong 512/131072, deep 2048/524288 and dangerous 65536/4000000 (simulations / solver nodes).

```sh
python tools/build_web.py wasm                      # gumbel, native, tactical and six wasm (committed)
python tools/build_web.py ort                       # onnxruntime-web 1.30.0 into web/engine/ort
python tools/build_web.py model                     # every bubble-<step> release, or --checkpoint path/to/ema.pt
python tools/build_web.py six                       # Six's networks (pinned) into web/engine/six/networks
python -m unittest tests.test_web_engine tests.test_web_tactical tests.test_web_six
```

`wasm` needs [emsdk](https://emscripten.org/docs/getting_started/downloads.html) (`em++` on PATH or `--emxx`) and
`rustup target add wasm32-wasip1`; `web/engine/build.json` binds the committed artefacts to their sources and the
tests fail when they are stale. `ort`, `model` and `six` write ignored files; a static deployment runs all four.

## GitHub Pages

`.github/workflows/pages.yml` builds the bundle on every push to `main` (ONNX Runtime, and every Bubble release
exported as a network) and publishes `web/` as the site, so the play page opens at <https://tomodovodoo.github.io/HeXO/>.
It runs once the repository is public and Settings > Pages > Build and deployment > Source is set to GitHub Actions.
`web/coi-sw.js`, scoped to the site's path, adds the cross-origin isolation headers after one reload.

Without a play server the page answers its own requests in a browser session (`web/engine/play-session.mjs` on
`offline.mjs`'s rules): seats, analysis with auto-deepening, review, saved games and evaluations in IndexedDB with a
backup file, import and export, the opening book, tournaments and clocks. It differs from the server in these ways:

- One job runs at a time. An engine move goes first and sends a running analysis, review step or deepening back to
  the queue; deepening stops at strong unless the analysis engine runs on WebGPU.
- A search reads a cancel between network batches, so Cancel, Pause and seat changes take effect within one batch.
- When the solver's worker cannot start (some embedded browsers forbid a worker inside a worker), the page says so once
  and keeps those evaluations, which have no proofs, for the visit only.
- Game links are read through hexo.mineking.dev's API mirror, the only one of the sites that lets another site read
  it; hexo.tyto.cc game links cannot be read cross-site, so the page asks for the game's HTTTX. Tyto analysis links
  are decoded in the page.

Large analysis records are stored as lossless compressed blobs in IndexedDB. Snapshots reuse immutable record
versions, so saving a deeply analysed game does not copy its proof trees on the UI thread. Earlier plain records
still load, and updating an evaluation does not replace the older version in a saved game. The decoded-record
cache holds at most 32 MiB of JSON by byte count, separate from the session's active analysis.

Game and backup downloads of at least 256 KiB use `.json.gz` when compression makes them smaller. Import accepts
these files directly, as well as ordinary JSON, up to 32 MiB after decompression. The JSON inside has the same
format and retains all analysis records and certificates. The HTTP replay response remains ordinary JSON.

On a 5,288,454-byte study with 153 records, a Chrome IndexedDB check measured repeated save submission at
0.52 ms, down from 308 ms, and completion at 8.53 ms, down from 534 ms. A fresh storage instance restored the
records in 81 ms and built their first display state in 249 ms. The previous restore and first state took
202 ms and 332 ms. Importing through the page and downloading then reimporting its 364 KB compressed export
preserved all 153 records exactly. This is a large analysis study, not a typical game's storage cost.

## Bubble networks

`build_web.py model` downloads every GitHub release named `bubble-<step>` that carries an `ema.pt` and exports each
into `model/<step>/` (both graphs and a manifest); `--release TAG` takes one release and `--checkpoint path/ema.pt`
a local file, named by its step folder (`checkpoints/<variant>/<step>/ema.pt`) or else by the first 12 hex digits of
its SHA-256. A network already exported from the same file is kept. `model/networks.json` lists the exported networks
newest step first, and the seat and analysis pickers offer them in the network select, as for Six. A new release
reaches the site with the next push to `main`, whose Pages build fetches all releases. A bundle with only the older
single `model/manifest.json` still loads it, with no select.

## Devices and clocks

Once an engine's probe has finished, the seat and the analysis head tag it with its device:

| Engine | WebGPU | Otherwise | Clock |
|---|---|---|---|
| Bubble (browser) | yes, fp32 or fp16 | WASM, threads when isolated | turn time: solver at most a quarter, first stone 60% of the rest, simulations a ceiling |
| Six (browser) | yes, fp32 | WASM | Six's movetime, nodes a ceiling |
| Shrimp (browser) | yes, fp32 | WASM | none, fixed visits |
| Native (browser) | no | CPU, one thread | ms capped to the turn time |
| Seal (browser) | no | CPU, one thread | ms capped to the turn time |
| Strix (browser) | no | CPU, one thread | none, fixed simulations |

Without WebGPU (the probe finds no adapter) the neural engines start at Lightning for seats and analysis, and the
strength row shows the expected seconds per stone at the chosen stop, from the engine's own turns at that or another stop. Seconds
per stone on WebAssembly, taken from the measurements in the sections below (Bubble with 8 threads and no solver, the
others on one thread; Six from its 0.4 s per position):

| Engine | Lightning | Quick | Standard |
|---|---|---|---|
| Bubble | 0.07 | 0.25 | 1 |
| Six | 48 | 190 | 770 |
| Shrimp | 2.5 | 5.7 | not measured |
| Strix | 0.5 | 1.5 | 11.5 |

Six stays slow without WebGPU even at its lightest stop.

The turn time is `time_control.allowance`'s normal share (`web/engine/clock.mjs`); a per-turn clock gives the whole
turn less 10 ms. A side whose clock runs out loses on time. Engines without a clock are refused while one is on.

## Loading

A browser engine's worker reports the stage it is in (`web/engine/stages.mjs`), and the seat and the analysis head
show it next to the busy bar: "checking GPU" (the WebGPU probe), "downloading 12 of 27 MB" (all files of the load
that are not in the Cache API), "compiling" (the runtime's wasm, its thread workers or its WebGPU device, started on a
one-node graph), "starting GPU" or "starting CPU" (a network's session), "warming up" (Bubble's fp32 and fp16 timing
batch and a first batch for every engine) and then "thinking" while it searches. Six loads its network on its first
turn or network change, so those calls show the download, session and warm-up stages too.

Each stage has a limit on silence (`LIMITS`): 20 s for the probe, 45 s between download chunks, 30 s to compile,
60 s for a session, 30 s for the timing batch and 60 s to warm up. The adapter request inside the probe gets 8 s, and
a request or download that receives nothing for 30 s stops with an error. Manifests are fetched in the download
stage, so a network failure is never taken for a device failure. When a stage stays silent past its limit, or fails,
the page ends the worker and starts the next one down the chain, logs the reason to the console and shows one toast
such as "Bubble (browser): starting GPU timed out, running on CPU". A call that was waiting is sent again to the new
worker. When no step is left, or a download fails (another device would not help), the move or analysis fails with a
toast naming the engine and the stage. A job whose engine is still loading gives way when its seat or the analysis
changes engine.

| Engine | Fallback chain |
|---|---|
| Bubble, Six, Shrimp | WebGPU, then WebAssembly with threads, then WebAssembly on one thread, then an error |
| Bubble, Six, Shrimp without WebGPU | WebAssembly with threads, then one thread, then an error |
| Native, Seal, Strix | WebAssembly on one thread; a stage that stalls or fails ends in an error naming it |

A probe whose adapter request fails or times out counts as no WebGPU and shows the same toast ("checking GPU timed
out, running on CPU"). An engine that falls back from WebGPU to WebAssembly starts at Lightning from then on, and the
seats and analysis that use it move to Lightning (outside a running match), since Six on WebAssembly is slow even
there (see Six below). `prefer` or `threads` given to an engine fix that step of the chain.

Phones get two more rules. ONNX Runtime's threaded WebAssembly build runs one worker per thread on shared memory, and
`defaultThreads` (the cores but one, at most 8) gave an 8-core phone with 4 GB seven of them. With
`navigator.deviceMemory` (Chromium only, rounded down to 0.25, 0.5, 1, 2, 4 or 8) the count is at most 2 under 4 GB
and at most 4 under 8 GB, so the Vivo Y52 and the OnePlus Nord N20 (both report 4) run 4 threads. And an adapter whose
`maxBufferSize` is at most 256 MiB, the WebGPU default that phone GPUs report (desktop GPUs report gigabytes), is
limited: Bubble loads only the fp16 graph there when the adapter has `shader-f16` (else fp32), so the phone never
holds both sessions and skips the timing batch.

Every download keeps what it has received in the Cache API in parts of 4 MiB, so a load after a reload or a dropped
connection asks for the rest with an HTTP range and joins the parts; a server that answers the range with the whole
file starts over. The complete file then replaces its parts.

To test this without a phone, a page on localhost, 127.0.0.1 or [::1] takes `?stall=adapter,compile,session,timing,warmup`
(any of them): that step never answers, in the page and in the workers, and the watchdogs take over. Chrome's device
emulation shows the layout and the stage words at phone width.

## Running a local copy

A copy of `web/` (a static server, or the page `python/play.py` serves) usually lacks the build outputs: ONNX
Runtime, the Bubble model, Six's networks, Shrimp's model, Seal and the Strix network. Every browser engine stays in
the picker anyway. One whose files are neither on this origin nor already downloaded shows a download button with
its size; click it to fetch the files from the public site, after which the engine plays as usual. Picking such an
engine for a seat without clicking downloads the same files on its first move. When the site answers 404 for one of
an engine's files too (the site serves Seal and the Strix network only while `PUBLISH_SEAL` and `PUBLISH_STRIX_NETWORK` are true), the row reads "local build" and a
click shows the `tools/build_web.py` command that builds them here; nothing is downloaded or retried. Once the site
serves the files again, the download works with no code change.

`web/engine/assets.mjs` resolves each file. It asks this origin first and, on a 404 or a network error, fetches the
file from <https://tomodovodoo.github.io/HeXO/engine/>, which allows any origin to read it. A file from the site must
match the SHA-256 that this origin's manifest gives (`build.json`, `model/manifest.json`, `ort/version.json` and the
other manifests), or the site's own manifest when this origin has none. A mismatch, or a file the manifest does not
pin, fails the download with a message on the page. The bytes go into the Cache API store `bubble-engine-v1` of
this origin, keyed by the file's URL here and its SHA-256 (or its version when no manifest pins it), so a reload or a later visit reads them from the cache. Manifests that came from the site are kept there too and
answer when neither origin can be reached, so an installed engine starts offline.
The public site finds every file on its own origin and never fetches from elsewhere. The wasm that is committed and
loaded by its own glue (`gumbel.wasm`, `tactical.wasm`, `six/six.wasm`) is always present and loads as before.

For development, a `data-assets` attribute on the page's `seat.mjs` script tag points the fallback at another
engine folder. `?assets=<url of an engine folder>` does the same, but only on a page served from localhost,
127.0.0.1 or [::1], so a link cannot make a deployed page import code from a site of its choosing. Workers receive
the choice on their script URL.

## Layout

| File | Role |
|---|---|
| `python/export_web.py` | HexNet to ONNX (opset 17, dynamic batch and crop size, fp32 and fp16) with a parity report |
| `web/engine/engine-worker.mjs` | `EngineWorker`: a browser engine's worker from the page, with loading, cancellable calls, the stage watchdogs and the fallback chain |
| `web/engine/stages.mjs` | Loading stages: the worker's reports, the words the page shows, the limits and the `?stall=` test hook |
| `web/engine/bubble.mjs` | Page API: `BubbleEngine.load/turn/search/evaluate/bench`, `PRESETS`, `isolate()` |
| `web/engine/worker.mjs` | Engine worker: a turn exactly like `python/play.py evaluate` |
| `web/engine/assets.mjs` | Engine files from this origin or the public site, checked and kept in the Cache API |
| `web/engine/network.mjs` | Device probe, ONNX Runtime loading, sessions, batched evaluation |
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
| `web/engine/book.mjs` | Play page hook: the opening-book default that follows the seats |
| `web/engine/offline.mjs` | The game rules behind the browser session |
| `web/engine/play-session.mjs`, `browser-play.mjs` | The browser session that answers the page's requests on a static host |
| `web/engine/clock.mjs` | Clock requests and turn allowances |
| `web/engine/tasks.mjs` | `nextTask()`, which lets a worker read a cancel between network batches |
| `web/coi-sw.js` | Cross-origin isolation on static hosts (`isolate()`), for WebAssembly threads |

## Device choice

WebGPU with `shader-f16` loads both graphs, times a batch of 16 at crop 24 under each and keeps fp16 only when it is
1.25 times faster; fp32 reproduces the server's evaluations, fp16 does not. A limited adapter (see Loading) loads
fp16 alone. Without WebGPU it runs the WebAssembly
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
WebAssembly is not measured; as with Bubble, a compile that stalls for 30 s falls back to one thread (see Loading).

What the bundle contains: hexo-bot's Rust and the weights (MIT, Colton Miller), the crates they use (ahash, half,
serde, thiserror; MIT or Apache 2.0) and ONNX Runtime Web (MIT). Mantis Shrimp (Cmiller132/Hexo-Shrimp-Bot) is not
included: it has no licence and no published weights.

## Seal

Seal is the alpha-beta bot by Ramora0 ([HexTicTacToe](https://github.com/Ramora0/HexTicTacToe), revision `3474edb`),
the community's reference bot. **Seal (browser)** in the picker plays a seat or the analysis with the server's
presets: lightning 100, quick 250, standard 1000, strong 3000, deep 10000 and dangerous 60000 ms per turn, the same
ladder as Native.

```sh
python tools/build_web.py seal --emxx path/to/em++   # seal/engine.mjs, engine.wasm, manifest.json (ignored)
python -m unittest tests.test_web_seal
```

`build_web.py seal` downloads Seal's four headers at the revision pinned in `tools/engines.json`, checks each
against its SHA-256 and compiles them with `tools/seal_adapter.cpp`, the server's adapter, using the same em++ flags
as `gumbel.wasm`. The headers are never committed; the Pages workflow installs emsdk 6.0.10 and builds Seal on every
deployment. HexTicTacToe has no licence file, so the site serves the compiled Seal only while the repository variable
`PUBLISH_SEAL` is `true`; a local build always makes it.

The wasm is 115 KB and its glue 10 KB. The worker fetches `seal/manifest.json`, which pins both, then the glue and
the wasm through `assets.mjs`, and calls `seal_move` exactly as the server does; Seal's clock is `performance.now()` in the
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
Six positions (lightning 240 to dangerous 2,000,000) and the network select lists the site's networks, newest
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
since one thread evaluates about 2.7 positions per second. A 30,000-position search from the first stone gave the
same two stones on WebGPU in the browser (102 s) and from `sixengine --cpu` on the Ryzen 9 5900X (720 s).

Speed on the RTX 3070 Ti (otherwise idle) in the Claude desktop browser pane, WebGPU, from one stone, measured before
the ladder was rebased: 1,500 positions 1.3 s, 6,000 positions 22 s, 30,000 positions 102 s. On that scale the current
presets cost about 0.5 s (lightning, 240), 3 s (quick, 960), 13 s (standard, 3,840) and a minute (strong, 15,360).
The search itself and its threat solver run on one thread in the worker, so turns slow down as the tree grows. On
WebAssembly a position costs about 0.4 s on one thread, so without WebGPU lightning takes about two minutes and the
heavier presets are impractical. A device whose WebGPU cannot start, create or run Six's graph falls back to
WebAssembly at lightning (see Loading).

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
that file the page downloads the public site's when it publishes one, and otherwise asks for a local build
([Running a local copy](#running-a-local-copy)). The worker keeps `strix.wasm` and the network in the Cache API under their
digests. The network's licence is unstated in the repository, so the Pages workflow fetches it only while the repository
variable `PUBLISH_STRIX_NETWORK` is `true`; a local build always fetches it. The engine code is MIT (`web/engine/strix/LICENSE-hexo-strix.txt`); the Rust
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
