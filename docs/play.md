# Browser play

```sh
python python/bubble.py play                       # the newest release, downloaded into runs/play
python python/bubble.py play --model path/to/ema.pt
python python/play.py --dense-run runs/bubble --device cpu
```

Open <http://127.0.0.1:8765>. Either side is a person or an engine, so a person can play Bubble, Bubble can
play another checkpoint, and two engines play each other live while the page stays usable. Engine work runs on one
background thread as jobs; the page polls `/state` and every job can be cancelled. Cancelling an engine's move,
an engine failure and an import pause the game: the engine to move shows a pause mark and the play button pulses.
Placing a stone or pressing play resumes it.

The board fills the page. The top-left card holds both seats; hover it, or click its header, to choose engines,
checkpoints and strength. New, Undo and Pause sit at the bottom right, the move stepper at the bottom centre. The
right panel holds the analysis engine, the evaluation bar, the candidate stones (policy share, then the mover's win
chance after that stone), Import, Copy HTTTX, Review and the move list.

## Watch a bot match

The match command is a thin HTTP client. It uses an existing player on the port, or starts an idle one,
then submits the batch and prints the watch URL:

```sh
python python/bubble.py match "dense-v1@champion@strong" "Six@standard" --run runs/dense-v1 --openings narrow --unique-openings 16 --port 8772
```

Open <http://127.0.0.1:8772> to watch all 32 games, with the running score, analysis and move timeline.
The server plays each opening twice with colours swapped. It keeps running when the browser is closed,
and leaves the final board available when the batch finishes. Pause holds play and the next game; Stop match
ends the batch. Position edits and player changes become available again after stopping.

`python python/bubble.py models --port 8772` lists the same catalogue as the browser picker. Match names are
case-insensitive catalogue names, exact ids, unique engine kinds, or paths to `.pt` exports. Use
`dense-v1@main/150000`, `dense-v1@150000` or `dense-v1@champion` for checkpoints. `bubble:150000` also works
when that step identifies one available Bubble run. Ambiguous names are rejected.

A seat combines an engine, preset or custom budget, and a device. For example:

```sh
python python/bubble.py match "dense-v1@150000{simulations=512,solver_nodes=131072}" "Six@standard" --port 8772 --unique-openings 16 --a-device cpu
```

`@lightning`, `@quick`, `@standard`, `@strong`, `@deep` and `@dangerous` use the UI's presets; `--preset` supplies the default for both seats.
Custom keys are the engine's own: Bubble has `simulations` and `solver_nodes`, Six has `nodes` (positions),
Strix/Pulsatrix has `simulations`, and Native/Seal has `ms`. Unknown keys are rejected. `--a-device` and
`--b-device` choose CPU or CUDA for a Bubble seat. Six uses the backend in its catalogue entry.

The clock belongs to the match and applies to both seats:

| Mode | Command option | Behaviour |
|---|---|---|
| Fixed | No clock option | Each seat spends its configured search budget |
| Per turn | `--move 5s` | Five seconds covers both stones; no banking |
| Game | `--tc 180+2` | 180 seconds per player, plus two seconds after a complete turn |

Engines warm before clocks start. Under a clock, search budgets are ceilings: Bubble caps simulations
across the whole turn and keeps solver work inside the allowance; Six receives nodes plus movetime or
both clocks and increments; Native and Seal receive the smaller of their ms ceiling and the allocated time.
Strix/Pulsatrix clock requests are rejected because their current adapter cannot return an interrupted
search's best move. Fixed-budget games remain supported. Timing uses the clock/controller from the existing
timed engine; CPU/GPU and backend identities are saved, since they affect clocked strength.

Opening selection reads the player's existing `openings.json` from `--run`, or an explicit `--book path`:

| Range | Selection of N unique openings |
|---|---|
| `narrow` | The N highest recorded champion policy probabilities; ties use the canonical opening key |
| `wide` | A seeded sample without replacement from all active in-policy openings, the default |
| `all` | A seeded sample without replacement including active off-policy openings |

The probabilities are the book's saved whole-opening reach probabilities. N sets the resulting cutoff;
no model inference is needed to select the set. Retired and separately labelled tactical cases stay outside
these comparison pools. The book file, its digest, scoring checkpoint, cutoff and selected positions are
saved with the batch. The selection stays fixed if the live book refreshes. `--seed` controls sampling and
order and a random hex symmetry. Both colour assignments use the same orientation. `--games 32` also selects
16 unique book openings; an impossible opening count fails before play starts.
An explicit `--games` larger than twice `--unique-openings` repeats the selected opening cycle.

Without a book the default is the origin opening. `--opening start.htttx` supplies a custom position; repeat
it for several paired positions. `--max-placements 512` caps a game after the current turn, reported separately
from wins. Repeating identical openings and deterministic settings can repeat identical games.

Results go to a fresh `artifacts/play/<match>/` directory, or `--out <new-directory>`. Every completed game
gets a replay JSON and HTTTX file before the next game starts. `summary.json` records scores, player sources,
full Bubble weight digests, engine/network file digests, backend/device, budgets and opening selection.
Per-turn records include elapsed time and work counts where the adapter reports them; unavailable counts are
null. Complete colour pairs use the evaluator's pentanomial scoring and `dense_posterior.Posterior` for a
relative Elo estimate and 95% interval. These exploratory results stay in the batch directory and do not
change the training league or its calibrated-opponent scoreboard.

The **Tournaments** button lists saved batches and every completed game's result. Click a game to open it
in the player's analysis board in another tab, or download its HTTTX. The analysis board has its own CPU
engine queue, move timeline, analysis controls and Review button, so browsing and analysing a saved game
do not replace the live tournament board or spend its clock. One analysis board is shared by the player
port. Retry creates a variation without changing the saved tournament game.

The tournament catalogue also remembers custom `--out` directories and survives server restarts.
Analysis is saved in `artifacts/play/study-<port>.jsonl`; reopening a game restores it and also reuses
evaluations recorded during the tournament. Bookmarked game links reopen the saved game after a restart.

```sh
python python/bubble.py match status --port 8772
python python/bubble.py match pause --port 8772
python python/bubble.py match resume --port 8772
python python/bubble.py match stop --port 8772
python python/bubble.py match --resume artifacts/play/<match> --port 8772
```

Every completed turn saves `current.json`. A saved batch resumes from its last completed turn, preserving
completed games, colour pairing and clock balances. An interrupted search may need to run again. An OS lock
prevents two players resuming the same batch. Resuming rejects changed checkpoint or engine files.
Engine failures pause and report an error. A clock overrun loses on time. A batch cannot take over an
unfinished human game; choose another port or reset that board explicitly.

The HTTP API uses the same session that the browser displays. On an existing player:

```sh
curl "http://127.0.0.1:8772/openings?range=narrow&count=16"
curl -X POST http://127.0.0.1:8772/match -H "Content-Type: application/json" -d '{"players":["dense-v1","Six"],"opening_range":"narrow","unique_openings":16,"preset":"standard","seed":0}'
```

| Request | Result |
|---|---|
| `GET /state` | Engines, current board, jobs, match progress and score |
| `GET /models` | Player catalogue, checkpoints, budgets and clock support |
| `GET /openings?range=narrow&count=16&seed=0` | Preview the exact selected opening set, read-only |
| `POST /match` | Start with `players`, `games` or `unique_openings`, `opening_range`, `seed`, and optional `book` or `output` |
| `GET /match` | Match specification, score, completed results and paused state |
| `POST /match` with `{"action":"pause"}`, `resume` or `stop` | Control the current batch |
| `POST /match` with `{"action":"resume","batch":"path/to/batch"}` | Load and resume a saved batch |
| `GET /match/results` | Download the batch specification and completed results |
| `GET /match/replay?game=1` | Download a completed game's HTTTX |
| `GET /matches` | Saved tournaments and their completed game results |
| `GET /matches/game?batch=<id>&game=1` | Saved replay JSON; add `&format=htttx` for notation |
| `POST /matches/open` with `{"batch":"<id>","game":1}` | Open a saved game on the separate analysis board |
| `GET /replay`, `GET /htttx` | Export the visible game |

A player specification can also be `{"engine":"dense-v1","checkpoint":"main/150000","preset":"custom","custom":{"simulations":128,"solver_nodes":32768}}`.
Clock JSON is `{"mode":"fixed"}`, `{"mode":"move","ms":5000}`, or `{"mode":"game","tc":"180+2"}`.
The API is loopback-only. Each player port holds one visible game; use a separate port for another simultaneous match.

## Engines

Native always plays. Everything else comes from the models folder: `models/` in the checkout, or the folder given
with `--models` (the picker shows its path when it holds no models). Put files there, then press the rescan button
at the bottom of the picker, or restart the server.

| Engine | What goes in the models folder | Picker |
|---|---|---|
| Bubble | a run folder with `checkpoints/<variant>/<step>/ema.pt` (champion first), or any `.pt` export, for example `models/old/ema.pt` | the folder or file name; checkpoints in the select |
| Six | a folder, for example `models/six/`, with `sixengine.exe` (`sixengine` on Linux) and `gen-NNNN.onnx` networks | Six; networks and backend in the select |
| Strix | `strix.json` and the model file beside it | the JSON's name |
| Shrimp, or any engine speaking the Six protocol | `shrimp.json` | the JSON's name |

The run given with `--dense-run` or `--model` is shown as Bubble; `runs/` (or `--runs`) adds every other run under
its folder name. Seal appears when `build/libhexo_seal.dll` (or `.so`) is built with `-DHEXO_SEAL_SOURCE`.

A JSON entry names one engine, with paths relative to the JSON file:

```json
{"name": "Strix", "kind": "strix", "model": "strix.safetensors"}
{"name": "old", "kind": "bubble", "path": "../runs/old"}
{"name": "Shrimp", "kind": "six", "mirrored": true,
 "command": ["six-checkout/rivals/shrimp/.venv/Scripts/python.exe", "six-checkout/arena/drivers/shrimp_driver.py"],
 "presets": {"quick": {"nodes": 1, "args": ["--visits", "32"]}, "deep": {"nodes": 1, "args": ["--visits", "800"]}}}
```

`command` is a list of arguments. `"mirrored": true` is for engines in Six's frame, where HTTTX `(q, r)` is
`(q + r, -r)`. `presets` overrides a preset's budget; `args` are extra arguments for engines whose strength is set
at launch.

Getting the files:

- Six: the release zip and a network from github.com/CixMango/Six releases, unpacked into `models/six/`. The
  release's engine is the DirectML build, which ships `DirectML.dll` next to `sixengine.exe`. A CUDA build needs ONNX
  Runtime's GPU DLLs (among them `onnxruntime_providers_cuda.dll`) beside `sixengine.exe`, and `cudart64_12.dll` and
  `cudnn64_9.dll` on PATH, beside it, or in an installed PyTorch's `torch/lib`. TensorRT adds
  `onnxruntime_providers_tensorrt.dll` beside the engine and `nvinfer_10.dll`, for example from `pip install tensorrt`;
  the first game builds the plan beside the network, which takes a few minutes. The fastest backend whose libraries
  are found wins: TensorRT, CUDA, DirectML, CPU. Six searches by nodes, so a preset plays the same on any hardware.
- Strix: `python tools/build_strix_learned.py <hexo-strix checkout>` (Rust and MinGW) builds the engine into
  `tools/strix_learned/target/release`; the public model is `https://hexo.tyto.cc/model.safetensors`.
- Shrimp: clone CixMango/Six into `models/six-checkout` and build Cmiller132/hexo-bot with its
  `scripts/build_native.sh` into its `rivals/shrimp`; the entry above runs Six's Shrimp driver with that build's
  Python. Engines run in the models folder, so relative arguments resolve there.
- Seal: build with `-DHEXO_SEAL_SOURCE=<HexTicTacToe checkout>`.

Strength is a slider from Faster to Smarter with six stops, marked by growing hexagons and a hazard sign for the
last; arrow keys move it one stop. The sliders button beside it opens the custom budget.

| Preset | Bubble simulations per stone | Bubble solver nodes | Native and Seal ms | Six protocol nodes | Strix simulations |
|---|---|---|---|---|---|
| Lightning | 8 | 2,048 | 100 and 50 | 1,500 | 2 |
| Quick | 32 | 2,048 | 250 and 100 | 6,000 | 8 |
| Standard | 128 | 32,768 | 1,000 and 500 | 30,000 | 64 |
| Strong | 512 | 131,072 | 3,000 and 2,000 | 135,000 | 128 |
| Deep | 2,048 | 524,288 | 10,000 and 8,000 | 500,000 | 512 |
| Dangerous | 65,536 | 4,000,000 | 60,000 and 30,000 | 2,000,000 | 4,096 |

On a Ryzen 9 5900X with two threads, Bubble takes about 2, 3, 13 and 75 seconds per turn at Quick to Deep;
Dangerous takes many minutes per stone on a CPU. A thinking engine's seat shows a progress line (a moving one when
the engine reports no progress) and its cancel button. The custom budget shows the engine's own fields: Search
(simulations, 0 plays the raw policy, up to 65,536) and Solver (nodes, 0 turns it off, up to 4,000,000; the solver
gets up to a minute) for Bubble, Positions for Six, Search for Strix, and 10 to 120,000 ms for Native and Seal.

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
Changing the analysis engine, checkpoint or strength evaluates the shown position again at once.

Each cell has fixed places for its marks, so none hides another: a candidate or line stone fills the cell with its
rank in the middle, a threat is a badge at the lower left, a review glyph a badge at the upper right, the last stone
a dot, and the hovered cell a ring. Candidate rows carry the same threat badge.

Import reads whatever is pasted:

| Pasted | Read as |
|---|---|
| `version[1]; 1. [1,0][2,0];` | HTTTX, or a replay file or JSON list of `[q, r]` |
| `x, d @(0, 0) o A0 A1 x B2 B3` | Rectilinear notation (MineKing9534/HeXO): drawn stones, then BKE turns |
| `https://hexo.tyto.cc/analysis#c=BAE` | a Tyto analysis link, decoded on the spot |
| `https://hexo.tyto.cc/#g=<id>` | a Tyto game, from the site's `POST /game_htttx` |
| `https://hexo.did.science/games/<id>`, `/account/games/<id>`, `/sandbox/<id>` | a finished game or saved sandbox position, from `/api/finished-games/<id>` or `/api/sandbox-positions/<id>` |
| `https://hexo.mineking.dev/games/<id>`, `/sandbox/<id>` | the same, from the API mirror under `/proxy/api` |

The server does the fetching, without accounts or tokens. These sites draw HTTTX's `(q, r)` at `(q + r, -r)`; the
first stone moves to the origin, and in Rectilinear notation the player who moved first becomes cross. Drawn stones
must form complete turns, since a drawing has no move order.

The clipboard button above the move list copies the shown position as HTTTX; the menu beside it shows it as HTTTX,
Rectilinear and a Tyto analysis link, each with its own copy button. Hovering a stone's token there rings that cell
on the board, and hovering a stone on the board marks its tokens. `GET /export?format=htttx|rectilinear|tyto&ply=N`
returns the same text with each stone's span.

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
| Import | HTTTX, sandbox link | HTTTX, HeXO links, position string, replay file | none | none | HTTTX, replay file, HeXO game and sandbox links |
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
