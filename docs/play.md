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
right panel holds the analysis engine, the evaluation bar, the candidate stones (share of the improved policy, then
the stone's completed Q as the mover's win chance), Import, Copy HTTTX, Review and the move list.

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
when that step identifies one available Bubble run. A Six folder's network is its checkpoint: `Six@gen-0455`, the
newest by default. Ambiguous names are rejected.

A seat combines an engine, preset or custom budget, and a device. For example:

```sh
python python/bubble.py match "dense-v1@150000{simulations=512,views=4}" "Six@standard" --port 8772 --unique-openings 16 --a-device cpu
```

`@lightning`, `@quick`, `@standard`, `@strong`, `@deep` and `@dangerous` use the UI's presets; `--preset` supplies the default for both seats.
Custom keys are the engine's own: Bubble has `simulations` (Nodes), `ms` (Time, `active=ms` or `ms` alone applies it)
and `views` (Width), Six has `nodes` (positions) or `ms`,
Strix/Pulsatrix has `simulations`, and Drip/Seal has `ms`. Unknown keys are rejected. `--a-device` and
`--b-device` choose CPU or CUDA for a Bubble seat. Six uses the backend in its catalogue entry.

The clock belongs to the match and applies to both seats:

| Mode | Command option | Behaviour |
|---|---|---|
| Fixed | No clock option | Each seat spends its configured search budget |
| Per turn | `--move 5s` | Five seconds covers both stones; no banking |
| Game | `--tc 180+2` | 180 seconds per player, plus two seconds after a complete turn |

Engines warm before clocks start. Under a clock, search budgets are ceilings: Bubble caps simulations
across the whole turn, and its solver nodes only switch its proof work on or off, since a timed turn's proofs run
in time slices inside the allowance; Six receives nodes plus movetime or
both clocks and increments; Drip and Seal receive the smaller of their ms ceiling and the allocated time.
Strix/Pulsatrix and Six-protocol drivers of other bots (Shrimp) are refused under a clock: their adapters play a
fixed budget and cannot return an interrupted search's best move. Fixed-budget games remain supported. Timing uses the clock/controller from the existing
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

### A single game on a clock

An engine seat's strength panel holds the same four clock choices under its slider: Budget, Turn, Absolute and
Fischer, with the base time and the increment in seconds. The clock belongs to the game and covers both sides,
people included; each seat head shows its remaining time and the side to move counts down. Choosing a clock
starts both balances full at the shown position; New, Undo, Retry and Import start them again. Engine seats play
through the timed engines above, which warm before their clock runs, so Bubble, Drip, Seal and Six spend the turn's
allowance with their budget as a ceiling. A seat whose engine cannot keep a clock (Strix, Shrimp) is refused while a
clock is on, and a clock is refused while such an engine plays. A side whose time runs out loses on time, also while a
person thinks. The saved game records the control (`clock`) and, after each complete turn, the time spent and both
balances (`turns`).

The trophy button opens Tournaments. New starts a batch on this board between two engines (A and B, each with
its checkpoint and strength), for a number of games, from the origin or from narrow, wide or all book openings
(each played twice with colours swapped). Each entrant has the seat's six strength stops and its custom budget
(the sliders button). The clock is Budget (the seats' own budgets), Turn (seconds per complete turn), Absolute
(seconds per game) or Fischer (seconds per game plus seconds after each complete turn). An unfinished game
against a person is cleared first. Results lists saved batches with the score, the Elo of A over B
with its 95% interval from complete colour pairs, and every completed game's result. Click a game to open it
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
| `POST /match` | Start with `players`, `games` or `unique_openings`, `opening_range`, `seed`, and optional `book`, `output` or `replace` (clear an unfinished game against a person) |
| `GET /match` | Match specification, score, completed results and paused state |
| `POST /match` with `{"action":"pause"}`, `resume` or `stop` | Control the current batch |
| `POST /match` with `{"action":"resume","batch":"path/to/batch"}` | Load and resume a saved batch |
| `GET /match/results` | Download the batch specification and completed results |
| `GET /match/replay?game=1` | Download a completed game's HTTTX |
| `GET /matches` | Saved tournaments and their completed game results |
| `GET /matches/game?batch=<id>&game=1` | Saved replay JSON; add `&format=htttx` for notation |
| `POST /matches/open` with `{"batch":"<id>","game":1}` | Open a saved game on the separate analysis board |
| `GET /replay`, `GET /htttx` | Export the visible game |

A player specification can also be `{"engine":"dense-v1","checkpoint":"main/150000","preset":"custom","custom":{"simulations":128,"ms":2000,"active":"ms","views":8}}`; root query nodes follow the
work or the time (see the custom budget above), and a `solver_nodes` key is rejected here.
Clock JSON is `{"mode":"fixed"}`, `{"mode":"move","ms":5000}`, or `{"mode":"game","tc":"180+2"}`.
`POST /clock` with the same JSON puts the single game on this board on that clock.
The API is loopback-only. Each player port holds one visible game; use a separate port for another simultaneous match.

## Engines

Drip always plays, and Drip (browser) runs the same engine as WebAssembly in the page
(docs/web-engine.md). The engine picker also lists every engine below; one that is not installed yet has a download
button. Click it and the server installs the engine into the models folder (`models/` in the checkout, or the folder
given with `--models`). The row fills as files download, turns while something compiles, and becomes the engine
once it is registered. A failure shows its error for a few seconds and the button comes back.

The browser engines (Bubble, Drip, Six, Strix, Shrimp and Seal, each marked "(browser)") install themselves. When
this checkout lacks their build outputs, the picker shows a download button with the size; it fetches the files from
the public site, checks them against their SHA-256 and keeps them in the browser's cache, so only the first use
downloads ([Running a local copy](web-engine.md#running-a-local-copy)). An engine the public site does not serve
shows "local build" instead, and clicking it shows the `python tools/build_web.py` command to run.

| Engine | Source, licence | One click |
|---|---|---|
| Six | newest release of CixMango/Six, MIT | downloads the release archive for Windows, Linux or macOS (arm64), checks it against the SHA-256 GitHub publishes for it, and keeps only the engine folder and the network, as `models/six/`. Also runs in the browser as Six (browser), which installs itself ([web-engine.md](web-engine.md#six-browser)) |
| Strix | SootyOwl/hexo-strix at `5a771e5`, MIT; the network is `strix-237000`, the Strix network at training step 237000, hosted in this repository's `strix-networks-v1` release with the permission of its author, Tyto, who trained it | builds the wrapper with `tools/build_strix_learned.py` when cargo and git are found, else downloads it from the `engines-v1` release; downloads the network from `strix-networks-v1` and checks its pinned SHA-256; writes `models/strix/` and `models/strix.json`, which lists the network as the entry's checkpoint. Also runs in the page as Strix (browser) once `python tools/build_web.py strix-network` has placed the network; the public site serves the network only while it publishes it (see docs/web-engine.md) |
| Shrimp | Cmiller132/hexo-bot at `6251fc6` (main_7, epoch 18), MIT, run by Six's Shrimp driver | downloads the driver, the weights and the search profile, each at a pinned SHA-256; builds `hexo_engine` and `shrimp` with maturin when cargo is found, else downloads the release's wheels for this Python; writes `models/shrimp/` and `models/shrimp.json`, which run the driver with the server's Python and its PyTorch on two CPU threads. Needs Python 3.11 or newer with PyTorch and NumPy. The page also runs it with no server, as Shrimp (browser), which installs itself |
| Seal | Ramora0/HexTicTacToe at `3474edb`, no licence file | downloads four pinned headers and compiles `tools/seal_adapter.cpp` with g++ or clang++. It has no licence file, so the engines release does not carry it and it needs a C++ compiler; a local build serves it compiled to WebAssembly ([browser engine](web-engine.md#seal)); the public site only while it publishes it, and otherwise the picker asks for a local build |

Pinned revisions, URLs and hashes live in `tools/engines.json`. A setup works in `models/.setup/<engine>` and moves
the finished folder into place with a `setup.json` marker; it replaces only folders and entries that an earlier setup
made. `.pt` files inside such folders are never offered as Bubble models. Over HTTP, `GET /setup` lists the engines,
whether each is installed (the id of its entry) and the last setup's progress; `POST /setup` with
`{"engine": "six"}` starts one.

The prebuilt Strix and Shrimp files come from this repository's `engines-v<N>` release.
`gh workflow run engines.yml -f tag=engines-v1` builds them for Windows x64, Linux x64 and arm64, and macOS arm64 (Shrimp's wheels for Python 3.11 to
3.14) and creates the release with a `SHA256SUMS` file. Downloads are checked against the hashes pinned in
`tools/engines.json`, or against the release's `SHA256SUMS` while none are pinned; `python tools/build_engines.py pin
engines-v1` copies those hashes into the manifest. `python tools/build_engines.py build --out dist` builds the same
files on one machine.

Kraken (Ramora0/KrakenBot) is not offered: its code has no licence and its weights are not public. Neither is
MantisNet (Cmiller132/Hexo-Shrimp-Bot), which has no licence and no weights.

Strength is a slider from Faster to Smarter with six stops: a spark, an open hexagon, a filled one, one in a ring, a
stack and the hazard sign; arrow keys move it one stop. The sliders button beside it opens the custom budget.

When the run has an opening book (`openings.json` in `--dense-run`, or `--book`), the seats card offers Opening
book, on when the player starts, and its set: narrow (the openings above the policy cutoff, as for batches),
on-policy (every active opening) or all (with the imported off-policy ones). New then starts from one of them in a random
orientation, shown as played stones; the move list names the set. Between engines the opening is drawn uniformly.
Against a person it prefers lines that person has not played on that side: walking the book's tree, each step
picks among branches that still hold an unplayed opening, so a branch counts as played only once all of its
openings have been; when all have, one of the least played is picked. What was played is kept per side in
`play-openings.jsonl` beside the saved evaluations, appended like them. A browser seat is a human seat to the
server, so the page names the sides people play (`people`) when it asks for a new game or the book.

Until the switch or its set is changed in this browser session, each seat change sets the book from who plays
the seats: off when both are Human, on otherwise (a seat a browser engine plays is an engine).

Bubble's and Six's levels are amounts of work, not time: Bubble's nodes per stone (the hybrid owner's completed
simulations) with its root query nodes, and Six's positions. The counts were picked so each ladder spans about the
same wall time on its own device, under half a second at Lightning to a few seconds at Deep. A CPU server gives Bubble
its own ladder (`CPU_PRESETS`); the browser uses the GPU one. The dial's tooltip names each level's count.

| Preset | Bubble nodes, GPU | Bubble nodes, CPU server | Bubble root query nodes, GPU / CPU | Drip and Seal ms | Six positions | Strix simulations |
|---|---|---|---|---|---|---|
| Lightning | 16 | 4 | 1,024 / 512 | 100 | 120 | 2 |
| Quick | 64 | 8 | 2,048 / 1,024 | 250 | 240 | 8 |
| Standard | 256 | 16 | 4,096 / 2,048 | 1,000 | 480 | 64 |
| Strong | 512 | 32 | 8,192 / 4,096 | 3,000 | 960 | 128 |
| Deep | 1,024 | 64 | 16,384 / 8,192 | 10,000 | 1,920 | 512 |
| Dangerous | 65,536 | 1,024 | 4,000,000 / 131,072 | 60,000 | 2,000,000 | 4,096 |

Seconds per analysis turn on main/185000, from 3, 5 and 9 stones (the range over the three), RTX 3070 Ti and Ryzen 9
5900X, measured on 2026-10-08 while the training run shared the GPU and CPU. Old is the ladder before this one
(8, 32, 128, 512 and 2,048 nodes with 2,048 to 524,288 root nodes; Six 240 to 61,440 positions).

| Preset | Served GPU | Served GPU, old | Browser WebGPU | Browser, old | Served CPU | Six (browser) | Six, old |
|---|---|---|---|---|---|---|---|
| Lightning | 0.3 to 0.8 | 0.5 to 1.0 | 0.08 to 0.10 | 0.07 to 0.10 | 0.5 to 1.0 | 0.3 to 0.4 | 0.7 to 0.8 |
| Quick | 0.7 to 1.8 | 1.3 to 1.7 | 0.33 to 0.36 | 0.18 to 0.22 | 0.5 to 1.2 | 0.7 to 0.8 | 1.6 to 2.1 |
| Standard | 1.6 to 2.4 | 2.0 to 10 | 1.0 to 1.1 | 0.49 to 0.57 | 1.5 to 1.9 | 1.0 to 1.1 | 5.8 to 7.6 |
| Strong | 1.7 to 3.8 | 3.1 to 19 | 1.8 to 2.3 | 1.8 to 2.2 | 2.8 to 3.4 | 1.6 to 2.1 | 24 to 31 |
| Deep | 3.3 to 11 | 6.0 to 65 | 3.4 to 4.6 | 7.7 to 9.5 | 6.3 to 6.9 | 2.7 to 3.6 | 89 to 104 |

On the old ladder the root queries made most of the served time: a query that cannot spend its nodes runs to its
10 s deadline, which the five-stone position hit from Standard up. The CPU server's old ladder took about 1, 8, 36
and 125 s at Quick to Deep. The CPU column was measured with 1,024 root nodes at Lightning. Six was measured only in
the browser (WebGPU); a served Six on TensorRT or CUDA runs the same counts faster. Dangerous stays a stress level.

A thinking engine's seat shows a progress line (a moving one when the engine reports no progress) and its cancel
button. The custom budget shows the engine's own fields. Bubble has Time (ms per turn) and Nodes (work per stone),
and exactly one of them applies: editing one makes it the budget and greys the other, which keeps its value for
later. Width is the number of views the owner searches per position (1 to 16, 8 by default). The root queries get 16
nodes per node of work, or 4 per ms, between 1,024 and 4,000,000; under Time they get at most a quarter of the
clock, the first of two stones 60% of the rest and Nodes is not a limit (65,536 per stone at most). Six has Positions
or Time on the same rule. Strix has Search (at least 1), Drip and Seal ms (at least 10). The choice is saved with the
custom budget. An older custom Bubble budget opens with its simulations as Nodes; its solver nodes are dropped. Any
larger whole number up to 2,147,483,647 (the engines take 32-bit budgets) is accepted on both pages; it only takes
longer. A time budget's evaluations are saved under their own key (`~ms2500`), as are a width other than 8 (`~views4`).
A Bubble seat, served or in the browser, keeps one search graph for its game (a `GameGraph`, see
[neural-search.md](neural-search.md)), adding each turn's simulations to the visits already under the position until
undo, a new or loaded game or a seat change, so its moves are saved with the kept-tree evaluations and never read
back from the store.

### How Bubble searches

Every Bubble search on the served page runs on the hybrid scheduler ([search-scheduler-design.md](search-scheduler-design.md)):
analysis, Auto, deepening, refresh, review, engine turns and matches. Each stone of a turn is one search of
the game graph at that position. A native graph owner spends the preset's simulations there as completed
simulations over up to eight views: the position itself and positions below it that its own evidence ranks, at most
eight placements down. Each view searches in quanta of up to 64 simulations (fewer when the budget is smaller), with
16 sampled root candidates and the round barrier, and the network batches of every view share one queue of up to
128 rows. A position's simulations therefore mean neural work spent around it, not visits at its root alone.

A preset's solver nodes keep two jobs. They are the budget of the root queries (a forced win for the side to move,
the opponent's threat if it moved now, and the defence when a threat or a known proof makes one worth asking), and
when they are above 0 each search also runs a proof frontier: the owner hands the positions its search reaches to
native proof workers, each answer checked before it enters the graph, and every position the frontier proves joins
the game's proof table. Each lane (engine moves, and analysis with review) has one set of proof workers, four fewer
than the machine's threads and between 2 and 12, shared by all of its searches; the frontier may take the owner's
whole time. Solver nodes 0 turns both off, and so does a missing tactical build.

| Preset | Work per stone (GPU) | Owner quantum | Root query nodes | Proof frontier |
|---|---|---|---|---|
| Lightning | 16 | 16 | 1,024 | on |
| Quick | 64 | 64 | 2,048 | on |
| Standard | 256 | 64 | 4,096 | on |
| Strong | 512 | 64 | 8,192 | on |
| Deep | 1,024 | 64 | 16,384 | on |
| Dangerous | 65,536 | 64 | 4,000,000 | on |

### By hand

The models folder is scanned at start and by the rescan button at the bottom of the picker:

| Engine | What goes in the models folder | Picker |
|---|---|---|
| Bubble | a run folder with `checkpoints/<variant>/<step>/ema.pt` (champion first), or any `.pt` export, for example `models/old/ema.pt` | the folder or file name; checkpoints in the select |
| Six | a folder with `sixengine.exe` (`sixengine` on Linux) and `gen-NNNN.onnx` networks | Six with its backend; networks in the select, newest first |
| Strix, Seal, Shrimp or any engine speaking the Six protocol | a JSON entry | the JSON's name |

The run given with `--dense-run` or `--model` is shown as Bubble; `runs/` (or `--runs`) adds every other run under
its folder name. Seal also appears when `build/libhexo_seal.dll` (or `.so`) is built with
`-DHEXO_SEAL_SOURCE=<HexTicTacToe checkout>`. A JSON entry names one engine, with paths relative to the JSON file:

```json
{"name": "Strix", "kind": "strix", "model": "strix/model.safetensors", "engine": "strix/hexo-strix-learned.exe"}
{"name": "Seal", "kind": "seal", "library": "seal/hexo_seal.dll"}
{"name": "old", "kind": "bubble", "path": "../runs/old"}
{"name": "old-flat", "kind": "bubble", "path": "../runs/old", "q_range_floor": 0.5}
{"name": "Shrimp", "kind": "six", "badge": "shrimp", "mirrored": true, "command": ["python", "shrimp/launch.py", "--threads", "2"],
 "presets": {"quick": {"nodes": 1, "args": ["--visits", "32"]}, "deep": {"nodes": 1, "args": ["--visits", "1024"]}}}
```

`command` is a list of arguments, run in the models folder; a first argument `python` is the server's own Python.
`"mirrored": true` is for engines in Six's frame, where HTTTX `(q, r)` is `(q + r, -r)`. `presets` overrides a
preset's budget; `args` are extra arguments for engines whose strength is set at launch, and `files` lists further
files a match records the hashes of. Strix's `engine` defaults to `tools/strix_learned/target/release`.
`kind` is how the server runs an engine; the picker's badge says which bot it is. A Six-protocol entry for a bot
other than Six names it in `badge` (a lowercase word, `six` when left out): Shrimp's setup writes `"badge": "shrimp"`,
and a Strix behind Six's protocol would say `"strix"`. `--match` also takes a unique badge. A Shrimp installed before
badges existed shows its download button again; setting it up once more writes the badge. A Bubble entry's
`q_range_floor` (0 to 2) is its searches' Q range floor ([neural-search.md](neural-search.md)), saved under its
own evaluation key.

Six picks the fastest backend whose libraries it finds: TensorRT, CUDA, DirectML, CPU. The release's engine is the
DirectML build, which ships `DirectML.dll`. A CUDA build needs ONNX Runtime's GPU DLLs (among them
`onnxruntime_providers_cuda.dll`) beside `sixengine.exe`, and `cudart64_12.dll` and `cudnn64_9.dll` on PATH, beside it,
or in an installed PyTorch's `torch/lib`. TensorRT adds `onnxruntime_providers_tensorrt.dll` beside the engine and
`nvinfer_10.dll`, for example from `pip install tensorrt`; the first game builds the plan beside the network, which
takes a few minutes. A running Six keeps its process across presets; choosing another network starts one for it, and
the two most recently used networks stay running. Six searches by nodes, so a preset plays the same on any hardware.

## Analysis and review

The analysis engine is a Bubble checkpoint with its own preset. Analysis and review run on their own worker and
model, beside the one that plays engine moves, so they keep up during play. With Auto on it evaluates every
position where a turn starts, plus any position you step to; while an engine seat plays it also deepens the current
position through every preset, Lightning first, showing each as it lands and starting again when the position
changes. That deepening runs last in the queue, gives way to any other analysis and holds its network work while
an engine seat searches. Cancelling an analysis leaves Auto on: Auto and deepening only stop asking for that position
at those settings until something changes, a move, a step back or forward, another preset or engine, or Analyse
again. Each preset continues the search of the one before on the same position, adding only the
simulations it lacks, and keeps a solver proof it already has; these evaluations are saved apart from fresh ones
(their engine key ends in `:kept`), shown like them, and never used by review. All analysis of one game searches one
game graph, kept until undo or a new or loaded game: a position reached from several analysed positions is one node,
and the visits and values a search finds there count for every position before it. When an analysis lands, the saved analyses of the four
placements before it that came from the same graph are searched again with a quarter of their budget, so stepping
back shows what the later search found; any other position of the game the graph has changed since its analysis is
searched again the same way when you step to it. That search goes on from the visits the graph holds there. The share of the improved policy saturates at high budgets: its Q weight grows with the visits, so a
deep search gives one stone nearly all of it. A position without a saved
evaluation shows the search of the engine or analysis working on it as it goes. The search of a turn's second
stone is saved as the evaluation of the position after its first stone, so every placement has rows. The slider
rings the stop of the evaluation shown. Engine moves by the same checkpoint count as
evaluations, so a game against Bubble costs nothing extra on Bubble's turns.

A proof, once found, stays with the game. Each game keeps a table of proven positions: every saved evaluation of
one of its positions that ends proven, by the solver or by the search's exact root, and every position along that
proof's line, each with its winner, its distance in placements and the line from there. It is built from the saved
evaluations, so it survives a restart, an undo, another preset or another checkpoint. An analysis or review
consults it first. A position it proves won for the side to move gets that turn and line at once, with no solver
query or search. Otherwise each search starts with the stones it proves already settled, one stone ahead or two
when the second stone wins for the player placing it: a proven loss leaves the candidates, and a proven win ends
the search. Stepping back from a proven position therefore shows the stone that led there as proven. When that
stone wins for the player who placed it, the position before it is proven too: the stone leads the candidates,
the eval bar shows 100 for that player and the line runs on into the known one. The page shows
what the table proves even for an evaluation saved before the proof was found, and review labels use it. Engine
seats never read the table. The static page keeps the same table in its session, rebuilt from the evaluations it
saves in the browser, and sends it with each analysis request. Its search settles the proven stones in the tree
before searching, as the server does.

Review evaluates with the analysis engine, network and strength the analysis slot is set to (its preset or custom
budget), and labels each turn only from evaluations at exactly that budget, so a verdict never compares a deep
evaluation with a shallow one. It evaluates every position of the game, the ones after a turn's first stone too, so
a review costs about twice the turns. The Review button carries that preset's mark and counts the positions done. It evaluates the missing positions from the
last one backwards, so what a later position proves is already known when an earlier one is searched, in pooled
steps: the solver queries run on four tactical workers at once, each distinct position solved once and its proof
added to the game's proof table, then fresh game graphs, one per position, search together with that table on one
hybrid scheduler pool, so their owners share network batches and proof workers. Between
steps it gives way to more urgent analysis, and it holds its network work while an engine seat searches. It labels each turn
from the mover's win probability before and after it, and each stone the same way on its own. A first stone is
judged against the engine's stones from the turn start, a second stone against the engine's second stone given
the first one actually played. A position after a first stone lists the candidates for the second stone and its
value, like any other.

| Label | Meaning |
|---|---|
| ★ best | the engine's own turn, stones in either order; for a stone, one of the engine's stones |
| ✓ good | lost under 5% |
| ?! inaccuracy | lost 5% to 10% |
| ? mistake | lost 10% to 20% |
| ?? blunder | lost 20% or more |
| ✗ missed | had a proven win and lost it |
| ⚑ allowed | handed the opponent a proven win |
| ! found, = kept | proved a win, or kept one |
| · lost | the opponent already had a proven win |
| ◆ | six in a row |

The move list shows the turn's label on its row and each stone's own label beside the stone. The box under the list
names the turn, its line, and each stone's label, loss and the engine's stone in its place; the stone of the shown
position is highlighted, so stepping one stone at a time reads every grade. A turn still missing its second stone
gets a row with its first stone graded. For inaccuracies and worse the board outlines the engine's turn, beside the candidates of the position reached, and the panel lists its line. Keys: ← and →
step one stone, ↑ and ↓ one turn, Home and End, F fits the board. Retry plays on from the shown position. The
board draws HTTTX's `[q, r]` as the notation's diagram does: q to the right, r up and to the right, so a position
looks the same here as on the sites that use the notation. The buttons at the board's lower left turn it 60 degrees
either way, mirror it top to bottom and reset it, which shows a position from any of its 12 sides. The keys are [
and ], M and 0. Only the drawing changes. The move list, exports, analysis and saved games keep HTTTX coordinates,
and the browser remembers the view without putting it in links or saved games.
Changing the analysis engine, checkpoint or strength evaluates the shown position again at once.

The Deep Solve switch beside Auto sets the analysis to the solver preset, for puzzles and positions where a
forced win is the answer you want. Each analysis then spends up to two minutes on proof work alone and stops as soon
as a verified proof for either side arrives. Two provers run side by side: the tactical solver asks the root for a
win of the side to move with 32,768 nodes and four times as many each round, and a hybrid scheduler search of the
position feeds the analysis lane's proof workers (up to 12, four fewer than the machine's threads) the positions its
neural search reaches, with the whole owner budget. The panel shows that work in place of the evaluation bar, read
from the running solver and proof loop about twice a second: time, the root query's nodes and its current budget,
the frontier's answered jobs and nodes, its queued and running jobs and busy workers. When the root query stops
before spending its nodes, the side to move has no forcing win there; the panel says so and goes on showing the
frontier's work on the replies. Then it shows the winner and distance, or the nodes and certificates spent without
a proof. The turn and candidates are searched afterwards at the Standard simulations, from the proven turn
when there is one. Solver evaluations are saved under their own key (`~solver120000`) and their proofs join the
game's proof table like any other; engine seats cannot use the preset. The static page has the same switch. Its
proof workers are Web Workers, half the browser's threads less one, from one to eight (`PROOF_WORKERS`, see
[web-engine.md](web-engine.md)).

Each cell has fixed places for its marks, so none hides another. A candidate is a ring with its rank and, below,
the mover's win chance after it: rank 1 green, the others blue to red by how far they fall behind it. A proven line
is drawn as translucent stones in each player's colour numbered from the shown position; on a candidate's cell
its number moves to a badge at the lower right. A threat is a badge at the lower left, a stone the search proved
wins or loses a check or cross at the upper left, a review glyph a badge at the upper right, and the hovered cell
a white ring. Proven positions still list candidates, proven wins first and proven losses last. The proof shows
as the winner's colour and the placements of its line above the rows. Candidate rows carry the same badges.

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
must form complete turns; a drawing has no move order, so the importer looks for one that plays them legally.

Above the move list sit Import, the copy button (HTTTX of the shown position) with its format arrow, and Review.
The arrow opens HTTTX, Rectilinear, Tyto link and Game file: a click on a name copies it (or downloads the replay
file), and the chevron beside it opens its full text in a panel, with its own copy button. Hovering a stone's
token in the text rings that cell on the board, and hovering a stone on the board marks its tokens. In the move
list the stone of the shown position is outlined within its turn and the later one dimmed; hovering a stone rings
it on the board. `GET /export?format=htttx|rectilinear|tyto&ply=N`
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
| Engines per side | Strix versions against a human | Six levels, bot against bot | random, checkpoints, SealBot | human against Bubble or Drip | Bubble, Drip, Seal, Six, Strix or Shrimp per side, found in `runs/` and `models/`, bot against bot live |
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

On the static page every Bubble search runs the hybrid scheduler on the game's graph, as the served page does: a
preset's simulations are the owner's work per stone, and its solver nodes the root queries' budget, with the
owner's proof frontier on the page's proof workers whenever that budget is above 0. A refresh searches with a quarter
of the saved evaluation's simulations and no proof work. [web-engine.md](web-engine.md) has the settings.

On an isolated page ONNX Runtime runs on the cores but one WebAssembly threads, at most 8. On hosts where the
runtime's thread workers never come up (the Claude desktop browser pane is one), the engine notices the stalled
start 20 s after the downloads finish and replaces its worker with one running on a single thread.
`new BubbleEngine({threads})` fixes the count instead.

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
