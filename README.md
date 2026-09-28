# HeXO

## Native verified tactical strategies

The optional `tactical_proof.NativeTactics` library runs wide Strix IDTT followed by PDS-PN. The solver crates are vendored in `tools/tactical/vendor/hexo-strix` from [SootyOwl/hexo-strix](https://github.com/SootyOwl/hexo-strix) at `5a771e572553a8bd8e010112b2ce65f16e5afa1b` (MIT) and changed locally; the history of that directory lists the changes. IDTT principal variations are only hints. An exact positive requires a complete PDS-PN strategy DAG accepted by a separate raw-coordinate Rust checker. `independent_verify` also rechecks the exported strategy through the independent Python rules implementation.

```sh
python tools/build_tactical.py
python -m unittest tests.test_tactical_proof -v
```

This needs Rust/Cargo supporting edition 2024 and uses locked dependencies. The build manifest binds the native binary to its wrapper sources, the vendored sources and the Cargo lockfile. `NativeTactics().solve(game, ms=100)` returns `PROVEN_WIN` with `native_verified=true` and the complete current turn only after verification; unresolved searches return `UNKNOWN`, never a global loss. Partial-turn roots are supported. The exact board, player, remaining placements, fixed rules and verifier scope identify cached facts; neural model evaluations and visit counts are not stored here.

The certificate checker covers both mandatory two-cell defenses and a mandatory single block followed by every legal free second placement. For singleton covers it enumerates the entire radius-eight frontier after the block, including newly legal fillers, and preserves a legal order for each resulting pair. `solve(game, root_moves=[first, second], ...)` can verify a proposed attacker turn by constructing all these defensive branches and proving every continuation with bounded PDS-PN. The default upstream generator still searches fully forcing attacks; entirely quiet defender nodes and unresolved continuations remain `UNKNOWN`. A legal open-three fixture yields a 1,295-node wide-search strategy accepted by both checkers, whereas tight IDTT finds no forcing win. On the development host, a fresh verified proof took about 0.7 seconds and cached re-verification about 18 ms; this is a tactical correctness result, not a strength result.

The search checks its deadline on every PDS-PN level-1 node, before every level-2 expansion and every 16 IDTT nodes, and its position memos are capped. Upstream checked only every 8,192 nodes, so abandoned queries ran 10 to 40 seconds past a 2-second budget. With these checks they stop within about 10 ms, and a 15-second proof search peaks at 66 MB. Certificate reconstruction and verification scale with the proof, not the search. The kernel reads windows from the board's incremental window index, which also keeps each player's live windows (two or more stones, no opponent stone). The generator builds the wide builder list only when some threat cell can pair with a builder. That held at 16 of 20,187 generator calls in a six-turn win from dense-v1 self-play. Unexpanded attacker nodes are seeded with their threat-window count instead of a full move list, which cut generator calls to about 6,400 there. That win's 398-node proof takes 1.4 s instead of 14.7 s. The PDS-PN table is sized at one megabyte per 64 ms of budget, up to 16 MB, so a 5 ms query no longer spends 3 ms clearing a table. At 5 ms, 38 of 400 random dense-v1 turn starts had a verified forced win, at a mean cost of 1.2 ms per query. One persistent native worker bounds how long callers wait and rejects overlapping requests as `UNKNOWN`. Reports expose background-worker state, completed-late counts, elapsed worker time and Windows thread CPU time. These are **not equal-compute tournament clocks**. Late or partial certificates never become exact search values. Primary neural MCTS integration remains a separate task. The optional candidate route proved a legal 27-stone fixture with one mandatory block and 745 distinct legal free-placement replies. Both independent checkers accepted its 45,063-node strategy, and deleting one reply invalidated it. That proof took about ten seconds on the development host; it is not a 100 ms tactical result. All legal free placements are covered when a positive is returned, but finding a strategy remains selective and budget-limited.

`IsolatedTactics(package)` runs the same library in a disposable child process. A query returns within its budget plus `grace_ms` (default 100). A child that misses that deadline, or reports abandoned native work, is killed and replaced, so an overrun never delays the next query. Committed memory is capped at `memory_mb` (default 1536) through a Windows job object or `RLIMIT_AS`. Results return the strategy as undecoded JSON text in `certificate_json`, so decoding a multi-megabyte certificate never runs inside the deadline. On the development host the pipe adds about 0.2 ms per query, and a replacement child is ready about 80 ms after a kill. `forcing_material.worth_solving(game)` is a cheap gate built from the engine's six-cell window histogram. It requires a live window holding three of the mover's stones. All 42 solver wins among 1,030 labelled dense-v1 positions passed it, and 59% of turn starts in the same shards do.

C++20 Hexo rules and search engine, Python interface, and local browser game.

The current priority is a self-play learning loop with a local experiment dashboard. Checkpoints earn promotion through matches against frozen opponents; fitting loss alone never replaces the incumbent.

An optional Strix root-probe experiment is available in `arena.py` after building
the [separate reference executable](tools/strix/README.md). `--strix-root-ms 20`
allocates up to 20 ms of each turn to that reference; its default is zero.
`--strix-root-nodes 1000`, `--strix-root-depth 8`, and `--strix-root-wide` expose
the solver limits and generator. The probe must have less time than `--ms`.
Only a sequentially legal winning PV can supply the current turn, recorded as
`source=strix_reference`, `score=null`, and `independent_proof=false`. All other
results fall back to native PVS with the remaining wall budget. No training
targets or native tactical rules change.

Each arena worker warms one persistent reference process during setup and
records that latency separately. Hard timeouts kill the process; later restart
costs count against the next turn. No work runs during the opponent's turn.
Reports include combined timings, overruns, call/result counts, executable and
adapter hashes. Native fallback receives at least 1 ms even after a scheduler
or cleanup overrun, which remains visible in the report.

A bounded operational check used the same trained pattern checkpoint, Seal at
100 ms, seeds 20260929/20260930, alternating which configuration ran first,
two color-swapped games per seed, and an 800-stone cap. Both probe-off and
20-ms/1,000-node probe-on lost all four games. The probe made 47 calls with
43 scoped negatives, four unknowns, and zero reference selections. Both
configurations stayed below 105 ms per measured turn; initial probe setup took
118–120 ms. This small check establishes neither a strength advantage nor a
strength equivalence. The experiment remains disabled by default.

Player 1 opens at the origin. Players then alternate two placements. Each placement must be empty and within hex distance eight of an existing stone of either color. Six or more connected stones along any of the three axes wins immediately, including on the first placement of a turn.

## Build and play

Requires Python 3.10+, CMake 3.20+, and a C++20 compiler. The Python interface has no third-party dependencies.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 4
python play.py
```

Open <http://127.0.0.1:8765>. Click cells or enter axial coordinates. Drag to pan, scroll to zoom, and choose either color or control both players. The engine button plays one complete turn. Undo removes one stone. All browser tabs share one local game.

On Windows with MinGW, add `-G "MinGW Makefiles"` to the configure command. Python locates the MinGW runtime through `g++` on PATH. With Visual Studio, it loads `build/Release/hexo.dll`.

```python
from hexo import Game

game = Game()
game.play(0, 0)
assert len(game.legal_moves()) == 216
turn = game.search(ms=1000)
for q, r in turn["moves"]:
    game.play(q, r)
print(game.state())
game.close()
```

`Game.legal_moves()` enumerates the complete legal frontier. `play()` checks each placement against the updated board. `undo()` restores turn phase, winner, evaluation, and position hash. Players are 0 and 1; winner -1 means ongoing. A search never changes the supplied position. A terminal position returns no suggested moves.

## Self-play learning

```sh
pip install -r requirements-learning.txt
python train.py --run runs/selfplay --iterations 10 --device cuda
```

In another terminal:

```sh
python dashboard.py --run runs/selfplay
```

Open <http://127.0.0.1:8766>. The dashboard shows self-play progress, positions collected, training and validation loss, checkpoint decisions, and Elo estimates against checkpoint 0. Everything stays in the run directory; no W&B account or external telemetry service is required.

The defaults use four CPU actors, 64 self-play games per iteration, a 50 ms self-play budget, and 40 evaluation games per opponent at 100 ms per turn. PyTorch trains on the GPU when available. Use `--device cpu` for a CPU learner. Each game cap is a truncation, not a draw. Truncated positions retain search targets but have no outcome label.

To generate games on the GPU and use larger learner batches:

```sh
python train.py --run runs/gpu-selfplay --selfplay-backend gpu --games 2048 --gpu-games-batch 0 --batch 0 --device cuda
python dashboard.py --run runs/gpu-selfplay
```

`--gpu-games-batch 0` chooses a batch from available VRAM and the episode cap. `--batch 0` increases the learner batch while preserving the reference batch-256 optimizer update count. `--updates-per-epoch` sets an explicit update budget. CUDA training uses fused AdamW and keeps replay tensors on the device. Replay sampling occurs before concatenating shards, and `--replay-positions` bounds retained positions. The limit also accounts for available device memory. Best validation weights and the matching optimizer state are saved.

The GPU `pattern` actor uses exact sparse rules with growing coordinate storage and incremental native-compatible pattern features. They first take exact one- or two-stone wins, then choose a placement belonging to a complete cover of all immediate opponent threats when such a cover exists. These choices override exploration. Quiet candidates combine nearby cells, axial development, and broader exploration, scored one placement at a time by the learned evaluator. They do not run native PVS; `--ms` and `--width` do not control this actor. Replay records identify the actor and one-placement target semantics. Promotion always measures the native deployed engine at equal wall-clock budgets.

On this RTX 3070 Ti, the tactical zero-residual GPU actor generated about 184 games/second and 28,300 placements/second at batch 512 with 32 quiet candidates and a 256-stone cap. Of those 512 games, 388 finished and 124 reached the cap; peak PyTorch allocation was about 920 MiB. These timings include generation and replay transfer, but not training or native evaluation. The earlier actor was faster but frequently missed immediate wins. Its shorter games make raw games/second an unfair performance comparison. Reproduce on your hardware:

```sh
python gpu_benchmark.py --actor --batches 64 256 512 2048 --placements 256
python gpu_benchmark.py --batches 64 512 2048 --placements 96
```

The second command compares only exact rule transitions and feature updates with C++. Small GPU batches are slower than native code. The dashboard exposes device utilization, VRAM, power, and temperature; telemetry includes other applications using the GPU. A small pattern learner cannot productively saturate every GPU unit at all times, so larger game batches and measured throughput guide settings.

The default `pattern` evaluator uses an `18 -> 32 -> 1` network to map six-cell ternary patterns to a bounded residual on top of the handwritten evaluator. It exports a 729-entry integer table. The `nnue` evaluator uses centered eleven-cell lines, nonlinear combinations of all three axes, an incremental 64-channel position summary, and separate value and conditional move-ranking heads. Both run entirely in C++ during search.

To train the full NNUE model with native search actors and a CUDA learner:

```sh
python train.py --model nnue --run runs/nnue --selfplay-backend native --device cuda --batch 0
```

NNUE replay stores sparse center patterns and search-selected first and second placements. `--nnue-replay-centers` bounds retained center features, and `--nnue-batch-centers` splits batches by feature count. NNUE's native format and feature definitions are documented below.

The GPU NNUE actor loads the same exported model and maintains the same sparse
center features. It ranks candidate first placements, evaluates conditional
second placements, and chooses a complete turn using the value head. Exact
immediate wins and defensive obligations override the quiet beam search.
`--gpu-beam 4` uses four first candidates and four conditional second candidates;
this remains selective search, not exhaustive play or native PVS. Exploration
does not create valid policy-teacher labels. Native matches still decide promotion.

```sh
python train.py --model nnue --selfplay-backend gpu --run runs/nnue-gpu --games 512 --gpu-beam 4 --device cuda
```

For NNUE, `--batch 0` means at most 256 positions, further limited by sparse
center count. Each epoch visits the selected training positions once unless
`--updates-per-epoch` explicitly requests another update budget. Metrics record
optimizer steps, examples processed, retained positions and retained centers.

New runs use `--curriculum mixed-v1`: legal prefixes of 3, 5, 7, 9 or 11 stones,
including compact fights, broad placements, separated groups and chained distant
placements. Families preserve stone owners and turn phase under all 12 board
symmetries. Evaluation reserves a separate family bucket that neither training
nor loss validation can use. `--curriculum legacy` retains the original narrow
three-stone distribution for explicit comparisons.

Native actors explore 5% of quiet turns by default. `--native-exploration` controls
this probability. Wins and mandatory defenses override exploration, including a
fresh check before the second placement. Exploratory actions do not become
teacher labels. Replay retains each sampled position's absolute ply and every
game's complete history, so omitted exploratory turns do not lose geometry.

Each iteration:

1. Play the latest learner against itself and earlier promoted checkpoints, with varied openings. The rated incumbent remains separate from the current learner.
2. Store positions, search targets, game outcomes, and complete move histories.
3. Continue the latest learner and its optimizer on recent replay data, even if it was rejected for deployment. Entire opening families stay together; the 12 hex symmetries determine family identity. One fifth of family hash buckets is reserved for validation.
4. Evaluate the challenger against the incumbent, the original checkpoint, and an older promoted checkpoint when available. These openings include radius-three placements, distinct from radius-two training openings, and are paired with colors exchanged.
5. Promote only after a positive observed advantage and an exact opening-pair sign test. The significance budget shrinks across successive attempts to limit false promotions; evaluation sample sizes grow when needed so promotion does not become mathematically impossible. Incomplete evaluation games prevent promotion. Clear losses against older checkpoints also prevent promotion.

Elo is estimated against the frozen original checkpoint, whose rating is zero. It is a within-run estimate, not a site leaderboard rating. Graphs include rejected challengers and conservative opening-pair confidence intervals. An incomplete evaluation has no Elo point estimate; bounds account for the unknown outcomes. `--eval-max-stones` defaults to 800 independently of the shorter training cap. The pair test, rather than the graph alone, governs promotion. A small evaluation may be unable to establish improvement. No run is guaranteed to produce a stronger checkpoint.

Run directories contain `summary.json`, append-only `events.jsonl`, replay batches, match histories, and checkpoint model/table files. Repeat the same command to perform additional iterations with the same configuration. Engine or training-source changes require a new run directory, keeping ratings comparable. A lock prevents concurrent trainers from writing the same run. If a process is forcibly killed, verify it has stopped before removing its stale `training.lock`.

To play the latest promoted checkpoint:

```sh
python play.py --run runs/selfplay
```

For a short end-to-end run, use a separate directory:

```sh
python train.py --run runs/short-run --iterations 1 --games 12 --eval-games 8 --ms 10 --eval-ms 20 --epochs 4 --workers 2
```

The evaluator increases undersized requests enough to make the pair test possible. This short command exercises collection, training, evaluation, and checkpoint decisions; it cannot by itself establish competitive strength.

## Engine

- Sparse axial board with signed 64-bit coordinates. The public API accepts coordinates within +/- 10^12 to keep arithmetic safe. There is no fixed board crop.
- Incremental counts and evaluation for the 18 six-cell windows touched by each placement. Tactical completion sets include broken lines anywhere on the board.
- Immediate wins take priority. Defensive covers intersect every opponent one-turn completion set; unused defensive placements are searched for development and counterattacks.
- Conditional first and second placements, followed by deduplication of resulting positions. A move at `(8, 0)` can make `(16, 0)` legal on the same turn.
- Iterative deepening over complete turns, principal variation search, and transposition bounds.
- A hand-written window evaluator plus an optional learned pattern residual, updated on make/unmake and loaded through `Game.load_table()`.

The legal environment is exact within its integer representation. Search is selective: ordinary candidates come from nearby cells and promising lines, and quiet turns are shortlisted. It does not prove game-theoretic wins. A mate-like search score is not a proof certificate. Deadlines are checked during search; setup and an individual candidate-generation operation can exceed very small budgets. Search metadata includes total native elapsed time.

Window storage uses a contiguous growing hash table with compact counts and pattern codes. Benchmarked completed searches improved by 8-22% compared with the initial node-based table; the measured make/undo workload improved by 31%. Moves, scores, depths and node counts matched at completed depths. Differential verification covered 28,723 states, 292 timeout/restoration checks, and 80 full legal-frontier comparisons.

## Opponent matches

```sh
python arena.py --opponent shallow --games 20 --ms 100
python arena.py --opponent random --games 20 --ms 100
```

The arena alternates colors and reuses each opening for a pair of games. Results contain moves, actual decision times, engine hashes, and conservative opening-pair confidence bounds. Truncations, invalid games and unplayed partners of a partial pair contribute unknown outcomes to those bounds. A completed-games-only Wilson interval is retained separately and must not be used as an overall strength estimate.

To compare against Seal, clone its source outside this repository, then configure the optional adapter:

```sh
git clone https://github.com/Ramora0/HexTicTacToe.git ../seal-reference
cmake -S . -B build -DHEXO_SEAL_SOURCE=../seal-reference
cmake --build build --config Release -j 4
python arena.py --opponent seal --games 20 --ms 100 --output artifacts/seal.json
python arena.py --opponent seal --run runs/gpu-selfplay --checkpoint 1 --games 20 --ms 100
```

The adapter compiles the external engine without vendoring it. Seal's fixed array has a smaller coordinate range; games outside the adapter's safe range are marked invalid rather than counted as victories. Equal requested budgets are used, and both engines' actual elapsed times are retained. `--run` loads the promoted checkpoint, `--checkpoint` selects another saved candidate, and `--table` or `--nnue` loads a standalone export. Reports identify the loaded model and its hash. Without a model option the arena uses the original evaluator.

The separate `seal-current-best` opponent uses [Ramora0/SealBot at c94749c](https://github.com/Ramora0/SealBot/tree/c94749c21c16c3b072fff6da49762dd5f92f3986), with its `best` pattern table. This is a newer search implementation than `seal`, which uses HexTicTacToe. The best table itself is unchanged from oldSeal. No project license was found at the pinned SealBot revision; upstream code stays in an external checkout.

```sh
git clone --depth 1 --filter=blob:none --sparse https://github.com/Ramora0/SealBot.git ../seal-current-reference
git -C ../seal-current-reference sparse-checkout set best
git -C ../seal-current-reference fetch origin c94749c21c16c3b072fff6da49762dd5f92f3986
git -C ../seal-current-reference checkout --detach c94749c21c16c3b072fff6da49762dd5f92f3986
python tools/seal_current.py ../seal-current-reference
python arena.py --opponent seal-current-best --nnue runs/example/checkpoints/0001/model.nnue --games 40 --ms 100 --max-stones 800 --output artifacts/seal-current-best.json
python -m unittest tests.test_seal_current -v
```

The optional adapter build requires a GCC-compatible C++20 compiler. Its manifest records every compiled upstream header hash, canonical and on-disk weight hashes, adapter and binary hashes, compiler, and build command. Arena verifies the binary against that manifest, resets upstream search state between games, and records both sides' ordered turns and elapsed times. The 100 ms setting is an upstream best-effort full-turn deadline, not a hard timeout. Upstream initializes randomness independently of the arena opening seed. Coordinates outside ±55, illegal moves, and incomplete turns are rejected. Upstream returns fixed pairs; when its first placement wins under native rules, only that winning prefix is played. This upstream engine does not correctly support non-opening partial-turn roots, so the adapter rejects those explicitly; arena calls start at complete-turn boundaries.

Other public references were checked on 2026-09-24. [Mantis at 9c94b95](https://github.com/Cmiller132/Hexo-Shrimp-Bot/tree/9c94b95ce5e3ccf4f892eeadca20524c522d0629) provides maintained Rust/Python inference and match entry points, but no public trained checkpoint or project license was found. [Strix at 5a771e5](https://github.com/SootyOwl/hexo-strix/tree/5a771e572553a8bd8e010112b2ce65f16e5afa1b) is MIT-licensed and publishes a [2,810,120-byte safetensors model](https://hexo.tyto.cc/model.safetensors), SHA256 `aec92391c66050e737d9b769757248b520ffc1bf44fa039db7c8abd3ef720185`. Its metadata says `checkpoint_000010.pt`, step 10, not the private `pulsatrix-246` checkpoint. Its relational graph requires direct `InferModel::eval_states` with Gumbel MCTS; the pinned HX04 server drops the required relational edge fields. [HextocZero at dc1be7b](https://codeberg.org/Kubuxu/HextocZero/src/commit/dc1be7b175dd9f27b6db8481e00153c9a0f2e3ae) is MIT-licensed heuristic MCTS with neural inference unimplemented; its documented legality omits the radius-eight restriction. None of these source discoveries establishes comparative strength.

The first current-Seal comparison used 40 games, 20 color-swapped opening pairs, seed `20261003`, width 16, and 100 ms requested per turn. The internally promoted `nnue-reanalysis-native-v1` checkpoint 1 scored **5 wins and 35 losses**, with no invalid or truncated games. Actual turn means were 87.72 ms for HeXO and 79.63 ms for SealBot, with maxima 117.79 and 103.76 ms. The conservative opening-pair 95% win-rate interval was [0, 0.429]. All 1,938 post-opening placements were independently replayed through the Python rules reference. This run used adapter commit `a4cc08a`, NNUE SHA256 `6c0599e380b3b87a764f7b95c76b21e43863f81fdc1a273bb4865434aed353ae`, and trace SHA256 `4b23e04428505d54411b5b1464919e7f5bb374ad8fb2c9f02e578926375575d5`. A subsequent adapter fix accepts upstream fixed pairs whose first placement wins; independent replay confirmed none of the 492 Seal turns in this trace had that case, so this recorded result is unaffected. It does not establish superiority over current SealBot.

To compare against the published Orca model, use an external checkout:

```sh
git clone https://github.com/Saiki77/hexbot-building-framework.git ../orca-reference
python arena.py --opponent orca --orca-source ../orca-reference --orca-sims 200 --games 20 --ms 100 --max-stones 800 --output artifacts/orca.json
```

This requires PyTorch. The adapter strictly loads the checkout's seven-channel `orca/checkpoint.pt` without adding random weights. Use `--orca-checkpoint` to select another compatible checkpoint and `--orca-device cuda` for GPU inference. The report records source revision, checkpoint hash, simulation budget and actual turn times. Orca receives simulations per placement; our engine receives milliseconds per complete turn. This comparison does not use equal time budgets. Native rules validate every returned move, and replay disagreements remain invalid games rather than wins.

The optional [learned Strix adapter](tools/strix_learned/README.md) loads the
pinned public `checkpoint_000010.pt` safetensors artifact and runs direct
relational graph inference plus Gumbel MCTS in a persistent CPU process.
Use `--opponent strix --strix-model PATH` after its separate build. Reports
identify the model, source patch, executable and actual timings. This public
step-10 model is not the private Pulsatrix checkpoint; its checkpoint license
is unknown. Its simulation budget is not an equal-time match against native PVS.

## Correctness tests and local benchmarks

Build the native library with the CMake commands above, then run:

```sh
python -m unittest discover -s tests -v
python -m tests.benchmark --positions 12 --ms 5 --reference-ms 25
```

The tests compare native rules and incremental features with an independent Python board reference. They cover sequential radius-eight legality, turn phase, both colors, all three winning axes, first-placement wins, overlines, distant expansion, make/unmake and hash restoration, residual-table bounds, tactical wins and defensive covers. With PyTorch installed, the same batched-environment checks run on CPU and on CUDA when available, including capacity growth, reset and truncation. GPU checks are skipped when PyTorch is unavailable. The NNUE implementation adds separate export/value/policy checks. Pull requests run the native rules subset on Linux; CPU/CUDA tensor parity is also run locally.

The benchmark prints JSON with seeded positions, source and library hashes, hardware, actual search times, nodes and agreement with a wider search. Timing-dependent search results can vary across runs. There are no machine-specific speed assertions. A wider search is a selective reference, not a proof of the best move. Candidate-cell recall is reported only when the native candidate API is available; it does not measure whether the final pruned turn list retained that pair.

## Bounded forcing certificates

`proof.py` searches continuous double-threat attacks and returns `PROVEN_WIN`, `PROVEN_LOSS`, or `UNKNOWN`. Every winning certificate covers all relevant defensive branches. Immediate counterwins take priority; a defense with a free second stone is unsupported and returns unknown. The independent verifier reconstructs rules and covers from raw coordinates. Ordinary search scores are never treated as certificates.

```sh
python proof.py --history position.json --ms 100 --output proof-result.json
python proof.py --history position.json --verify proof-result.json
python proof.py --benchmark
```

A history is a JSON list of `[q, r]` placements in play order. Verification needs a returned certificate; an unknown result has none. The solver is separate from deployed PVS and has no demonstrated Elo benefit. Its deadline is cooperative: a synchronous native candidate call can overrun it, and late results become unknown. In one benchmark a 13-stone forcing win verified in 35 ms, while a 1001-stone sparse board took 236 ms under a requested 100 ms budget.

## Experimental root turn coverage

Quiet root widening is opt-in. It retains every existing selected turn and adds complete pairs by conditional rank, with separate second-placement and final-turn budgets. Immediate wins and mandatory defenses keep their exact handling. Deeper search keeps its existing candidate restrictions.

```sh
python arena.py --opponent seal --games 40 --ms 100 --width 16 --root-seconds 16 --root-turns 48 --max-stones 800 --output artifacts/seal-widened.json
python -m tests.benchmark --trace artifacts/seal-trained-fresh-40.json --first-game 10 --positions 12 --ms 100 --width 16 --reference-ms 1000 --root-seconds 16 --root-turns 48 --output artifacts/pair-admission.json
```

The trace benchmark reports complete ordered-turn lists, resulting-position recall, depth, nodes and actual time. An optional `--seal-library` uses a separately built Seal adapter as reference; `--reference-report` reuses frozen reference turns for another ablation. On 12 development positions, the 48-turn setting raised Seal-reference result recall from 4/12 to 7/12, while mean completed depth fell from 2.50 to 2.42 at 100 ms. These traces include a repeated position family, so this is a development diagnostic rather than independent validation. Both default and widened settings later scored 4 wins and 36 losses against Seal on the same 40 fresh games at 100 ms. No playing-strength gain is established. Search clocks are best-effort; generation and legal fallback can exceed very short budgets.

## Status and remaining work

The local game, native engine, self-play trainer, checkpoint evaluation and dashboard are playable. The committed suite currently has 22 tests covering reference rules, CPU/CUDA parity, NNUE inference and undo, curriculum partitioning and training-target semantics. Native search now orders a stored transposition move first when it is already in the selected legal turn list. The table is still local to each search.

The old GPU actor produced 6,144 games but missed immediate wins. Its best early candidate scored 28/40, then only 78/160 on fresh confirmation games. The tactical replacement produced 6,144 games and 1,060,519 positions, with 526 capped games whose outcomes remain unlabeled. Its first candidate passed the paired promotion test at 104 wins and 56 losses against the frozen reference. Later candidates were rejected against that incumbent, including one that scored 110/160 against the reference but only 66/160 against the incumbent. Self-Elo is opponent-dependent; more training has not consistently improved the deployed checkpoint.

Fresh independent matches used 40 games and seed 20260924. The promoted pattern engine scored **4 wins and 36 losses against Seal** at 100 ms per turn for both engines. The handwritten engine scored **39 wins and 1 loss against the published Orca checkpoint** using 100 ms per turn versus Orca's 200 simulations per placement on CUDA. Orca averaged about 434 ms per turn versus HeXO's 89 ms, so this is not an equal-time comparison. There is no claim of superiority over all known bots.

The first native NNUE experiment collected 15,558 positions from 128 games and scored 39 wins and 41 losses against its zero-head NNUE reference. It was rejected. That reference pays NNUE inference costs; this experiment does not establish an improvement over the faster bare handwritten engine. The GPU NNUE pipeline has completed training, native evaluation and resumed training, with separate policy/value losses and actual optimizer exposure counts on the dashboard.

Remaining work includes complete-turn candidate recall and controlled widening, stronger-search reanalysis, bounded tactical search, broader independent equal-time matches, and architecture/compute comparisons. The current NNUE has width 32 and nonlinear fusion of crossing lines, but no neighboring-cell mixing block. Match clocks, rated lobbies, and online account play are not part of the local board yet.

## References

Rules and turn semantics were checked against the [official HeXO source](https://github.com/HeXO-Game/HeXO/blob/1aea2b676733f8cdc53f8f92cb00b367af72111c/packages/shared/src/sharedTypes.ts).

Independent opponents: [Seal](https://github.com/Ramora0/HexTicTacToe) and [Orca framework](https://github.com/Saiki77/hexbot-building-framework). The site bundles a Seal WebAssembly build; the optional native adapter currently compares against the selected external source revision, which may differ from that build.

The intended learned evaluator follows ideas from [Rapfi](https://github.com/dhbloo/rapfi) and [NNUE](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html), adapted to Hexo's three axes and turn semantics. Their existing game-specific weights are not used.
## Centered-line NNUE contract

The `nnue` model uses three centered eleven-cell ternary patterns at every center
whose lines contain a stone. A placement changes 33 directional patterns at 31
centers. The exported shared table has 177147 rows and 32 signed int16 channels,
scaled by 256. Channels 0–15 are odd under color swap; channels 16–31 are even.
All channels are invariant under line reversal. The learned mapping is
33 → 64 → 32 with ReLU and tanh. Training uses straight-through quantization.

For each center, sum its three table rows into `u`. Its 64-channel contribution
is `[ReLU(u), ReLU(-u)] - [ReLU(3*table[0]), ReLU(-3*table[0])]`. This subtraction
makes empty space contribute zero. C++ maintains exact int64 pooled sums. Divide
the pool by `256*sqrt(max(1, number_of_active_centers))`. For player 1, swap the
positive and negative halves of the first 16 channels. The other channels stay
in place. There is no board crop.

The value head is 68 → 32 → 1 with ReLU. Its inputs are the pooled 64 channels
and four context values: `remaining==1`, `remaining==2`, `log1p(stones)/8`, and
`(own_stones-other_stones)/max(1,stones)`. Its output times 6000 is a residual
added to the handwritten evaluation from the current player's perspective.
The resulting ordinary search score is clamped to ±500000; terminal scores
remain separate.

The conditional policy head is 104 → 16 → 1 with ReLU. Its inputs are pooled
64 channels, the 32-channel sum of the candidate's three lines **after** its
placement divided by 256, the four context values, and four pair values.
The candidate channels use the current player's perspective. Pair values are
`has_first`, `shares_axis`, `min(hex_distance,8)/8`, and
`shares_axis*max(0,6-hex_distance)/5`. All four are zero without a first stone
from the current turn. Second-stone examples are collected after the first
stone is applied. Policy scores order candidates; tactical inclusions remain
mandatory.

Native files begin with a packed little-endian 60-byte header: magic
`HXNNUE1\0`; nine uint32 values `1, 0x01020304, 177147, 32, 64, 4, 4, 32, 16`;
float32 scales `256, 6000`; and uint64 payload length. The payload contains
the row-major int16 table, then float32 value `W1,b1,W2,b2`, then policy
`W1,b1,W2,b2`. Shapes follow the heads above. Checkpoint metadata records the
whole-file SHA256. Native loading validates dimensions, sizes, finite weights,
table bounds and table symmetries. A model handle is immutable and shared by
attached boards. Loading a legacy table detaches NNUE and vice versa.

After an engine change, start a new rating run while retaining learned NNUE weights:

```sh
python train.py --run runs/nnue-new-engine --model nnue --initial-model runs/nnue-old/checkpoints/0001/model.pt --initial-optimizer runs/nnue-old/checkpoints/0001/optimizer.pt --device cuda
```

`--initial-optimizer` is optional. When supplied, AdamW moments and step counters
are retained, and `--lr` sets the new run's learning rate. The imported model is
validated, copied into checkpoint zero, and exported with the current native
format. Source paths and hashes are recorded; resume requires the same arguments
and unchanged source files. Ratings start at zero against the imported anchor.
Existing runs and checkpoint files are never reinitialized by these options.

Saved NNUE histories can be revisited with a frozen evaluator and a larger native
search budget, then supplied to an ordinary training run:

```sh
python reanalysis.py --run runs/nnue-selfplay --iteration 1 --checkpoint 0 --max-positions 256 --ms 200 --width 32 --output runs/reanalysis-0001
python train.py --run runs/nnue-with-reanalysis --model nnue --reanalysis runs/reanalysis-0001 --device cuda
```

`--reanalysis` accepts multiple completed shard directories. Their positions join
the replay sampling pool independently of `--replay-iterations`, which still
limits only chronological self-play shards. External files are read in place;
their manifests, search provenance and hashes are recorded in the run and each
trained checkpoint. Modified shards or manifests prevent resume. Repeating the
reanalysis command reuses completed root searches when its provenance matches.
Targets remain selective search estimates. A conditional second-stone row loses
the source game's outcome label when the teacher's first move diverges.

### Experimental KLENT training

`klent.py` runs a separate policy/Q experiment; ordinary `train.py` is unchanged.
It shares the centered-line NNUE representation and adds a scalar candidate Q
head. Neural collection and fitting batch on CUDA, while exact native rules and
feature reconstruction run on the CPU. Every legal cell is included, including
newly reachable cells after the first placement. Memory limits split batches;
they never crop the legal action set.

```sh
python klent.py --run runs/klent --initial-model runs/nnue-selfplay/checkpoints/0001/model.pt --games 16 --envs 8 --max-plies 128 --device cuda
```

One frozen actor collects each fresh corpus. With `pi = softmax(policy_logits)`
and `Q = tanh(q_head)`, the acting policy is
`mu = softmax((Q + beta * log(pi)) / (alpha + beta))`. One shuffled fitting pass
minimizes `CE(mu, pi_new) + (Q_new(taken_action) - G)^2`; stored targets are
detached. This follows the [KLENT paper's policy and scalar-Q objectives](https://arxiv.org/html/2602.10894v2#S4).
The defaults are alpha 0.03, beta 0.1, gamma 1, and placement-level lambda
`exp(-1/16)`.

Returns use the actual mover change: the sign is positive between two placements
by the same player and negative when the opponent acts next. The winning
placement has target +1. Capped games bootstrap the final nonterminal state from
that same frozen actor and remain explicitly unfinished; they are never draws.
Both colors' records are retained. A separate, reported value-head-only pass
distills the returns for native PVS with shared features detached; this auxiliary
pass is skipped when every critic target is zero.

Q starts at zero. With no terminal episodes, zero tail bootstraps can leave the
critic without a learning signal. Check `critic_targets_informative`,
`nonzero_return_fraction`, and `terminal_fraction`; throughput is not evidence of
improvement. `--initial-q` accepts a saved `q.pt` only with its exact matching
`--initial-model` file. Q depends on the learned shared representation, so an
unrelated Q head is rejected. No handwritten value is silently substituted for Q.

An explicit human-corpus Q warm-start can provide a nonzero critic before fresh
self-play. It freezes the matching NNUE features and fits only the Q head on
verified human chosen actions and their terminal outcomes in the acting player's
frame. The existing hashed family train/validation split is retained; test and
excluded shards are never loaded. These targets describe human continuations,
not optimal actions or on-policy KLENT returns. Values for unchosen actions are
model extrapolations that self-play must test.

```sh
python q_warmstart.py --corpus artifacts/datasets/human-warmstart-v1 --model artifacts/models/human-warmstart-12/model.pt --output artifacts/models/human-q-warmstart-12 --positions 40000 --epochs 12 --device cuda
python klent.py --run runs/human-initialized-klent --initial-model artifacts/models/human-q-warmstart-12/model.pt --initial-q artifacts/models/human-q-warmstart-12/q.pt --device cuda
```

Q initialization selects the best validation epoch and publishes into a new
directory atomically. Its copied `model.pt` and native export remain unchanged;
`q.pt` records their representation identity plus corpus and source provenance.
The tool does not resume partial fits or overwrite an existing output.

Checkpoint directories retain standard `model.pt` and `model.nnue` deployment
artifacts, plus `klent.pt` with Q/optimizer state and a representation-bound
`q.pt`. Complete corpus directories preserve every sampled move, legal-set hash,
acting distribution/value, mover/phase and return target. Manifests bind them to
the actor, code, engine, configuration and content hashes. Resume validates these
artifacts and reuses a completed corpus after an interrupted fit; unfinished
collection restarts deterministically. Each iteration consumes only its own
corpus. This path does not promote checkpoints or assign Elo; use paired external
evaluation of its native exports.

The inspected [Mantis implementation](https://github.com/Cmiller132/Hexo-Shrimp-Bot/blob/9c94b95ce5e3ccf4f892eeadca20524c522d0629/python/mantisnet/mantisnet/klent/train.py)
instead trains a categorical critic. Its [acting operator](https://github.com/Cmiller132/Hexo-Shrimp-Bot/blob/9c94b95ce5e3ccf4f892eeadca20524c522d0629/python/mantisnet/mantisnet/klent/improve.py)
also permits a mass-normalized Q score. Those adaptations are not enabled here.

NNUE replay is versioned separately from legacy six-cell histograms. It stores
ragged center-code triples and candidate-code triples with offsets, candidate
coordinates, pair context, turn context, player, handwritten baseline, chosen
candidate, search depth validity, outcome and opening family. Policy labels are
the search-selected first and conditional second placements, not alpha-beta
visit counts. Unfinished outcomes remain missing. Native PVS and the GPU
complete-turn beam identify their teacher semantics in the recorded games;
their search targets should not be treated as interchangeable depths.

Native depth-zero evaluation checks whether the opponent's immediate completion
sets can be covered by the remaining placements, after checking our own immediate
win. An impossible cover is a mate loss; a cover with a spare placement remains
unresolved. This fixes three observed leaf misvaluations. In a fresh paired
100 ms comparison against the pinned public Seal engine (20 opening pairs per
build, seed 20260929), the baseline scored 3-37 and the guard scored 2-38, with no
truncated games. This experiment did not demonstrate a playing-strength gain.

The trace benchmark's ordered and resulting-position recall measure the complete
**untimed** generated lists. Timed searches can expire during generation, so these
figures do not claim that a turn was searched within the budget. The report records
generation time, completed depth, and zero-depth trials separately. Reused reference
reports must match both the trace hash and the exact position history.

Experimental TT turn admission is available through
`Game.search(..., tt_injection=True)` and `arena.py --tt-injection`.
The arena records the flag in its configuration and applies it only to the
contender. It validates and reserves a previous-iteration complete turn before
candidate truncation, preserving immediate wins and mandatory defenses. Hints
are frozen throughout each iteration, including PVS re-searches; TT score-bound
reuse is disabled in this mode. The table remains local to one search call,
with no persistent entries or cross-model score reuse. The default remains off.
A 12-position development benchmark showed identical depth-three results with
4.6% more elapsed time; this experiment has no demonstrated playing-strength gain.

## Official notation and local bot API

`notation.py` imports and exports [notation v1](https://github.com/hex-tic-tac-toe/hexagonal-tic-tac-toe-notation/tree/15bb7877ae020d661497e332adf0810d00d24e3e).
Cross is native player 0; its origin placement is implicit in the text and present
in imported histories. Turn numbers, sequential radius-8 legality, and terminal
states are checked. Metadata and `!` annotations are preserved without inferring
their meaning. The upstream example uses `datetime` rather than `utcdatetime`,
and a named time control that differs from its numeric grammar; these values are
kept intact. Python callers can use `loads(text)` and `dumps(record_or_history)`.

```sh
python notation.py import match.txt > match.json
python notation.py export match.json > match-roundtrip.txt
python bot_api.py --port 8790 --ms 100
```

The loopback HTTP adapter exposes `GET /capabilities.json` and
`POST /stateless/v1-alpha/turn` according to the
[published API definitions](https://github.com/hex-tic-tac-toe/htttx-bot-api/tree/37d2385f1016abe8b25798238a7d0c4a17a25dda/definitions).
For example, send `{"board":{"to_move":"o","cells":[{"q":0,"r":0,"p":"x"}]},"request_id":1}`
with content type `application/json`. Responses contain `move.pieces` objects
with `q` and `r`, and echo an optional request ID. Add `--model path/to/model.bin`
to use a native NNUE export; capabilities identify its SHA256. Otherwise the
adapter uses the handwritten evaluator. This command does not register a bot
with any external service.

Both published formats require exactly two placements per recorded turn or API
move, while the API also forbids placements after a win. A first-placement win
cannot satisfy both requirements. Notation export raises `NotationConflict` for
that case, partial turns, and empty boards. The API returns HTTP 409 for origin
turns, partial turns, already-terminal boards, or a chosen first-placement win;
it never pads a winning move. Full two-placement wins are supported.

The stateless board is unordered. The adapter reconstructs a legal ordering of
exactly the supplied cells and checks counts and `to_move`; it never assumes array
order is history. Unreachable positions return 400, and a reconstruction exceeding
one second or 20,000 visited states returns 503. Local limits are 1 MiB request
bodies and 1,025 board cells; notation accepts 4,097 placements and 1 MiB text.
`time_limit` is used as an advisory search cap, with zero returning 408. Native
search and reconstruction cannot guarantee a hard response deadline, so
`move_time_limit` is false. Websocket and matchmaking capabilities are not declared.
Run `python -m unittest tests.test_notation_api -v` for the protocol checks.

The local adapter bounds the request line and headers together to two seconds,
including clients that keep sending bytes. Request-body transfer has a separate
two-second deadline. These transport limits are separate from its advisory search
budget. A configured model that disappears or becomes unreadable returns JSON 503.

## Relational policy/Q training

`relational_warmstart.py` initializes both heads of the primary relational model
from full human histories. Geometry is rebuilt from chronological moves. Existing
hashed corpus shard membership supplies the family split; NNUE features and
weights are not transferred. Only the recorded action receives a Q target, using
the actual terminal outcome from its mover's perspective. Human actions are
imitation labels, not claims of optimal play. Test and excluded games never
supply training or validation examples.

```powershell
python relational_warmstart.py --corpus artifacts/datasets/human-warmstart-v1 --output artifacts/models/relational-human --epochs 12 --device cuda
python relational_train.py --run runs/relational-v1 --initial-model artifacts/models/relational-human/model.pt --games 256 --envs 8 --iterations 5 --max-plies 300 --device cuda
```

The default network uses width 256, eight blocks, eight attention heads, feed-forward
width 1024 and sixteen global tokens. `--positions 0` uses all available human
positions after the corpus family-prefix boundary; a positive value selects a
seeded bounded sample. Graph microbatches are bounded by `--max-nodes` and
`--max-edges`. A single position exceeding a budget raises an explicit error;
legal actions are never cropped. CUDA uses BF16 matrix operations and FP32
normalization, losses and return targets.

KLENT collects each fresh corpus with frozen weights and fits it in exactly one
shuffled pass. It uses the existing scalar policy/Q objective and signed lambda
returns, including actual-player sign changes and explicit cap bootstrap. It
neither mixes old replay into the fit nor distills an NNUE value/export. Proofs
are not substituted for historical returns. New runs reset Adam; resuming an
existing run restores its model and optimizer and verifies source, engine,
initial model and previously consumed corpus hashes. A completed pending corpus
is reused after an interrupted fit.

`model.pt` contains schema `hexo-relational-policy-q-v1`, the complete model
configuration and state dictionary, including Q. `relational_train.load_model`
loads this format for the neural evaluator/player. Native NNUE loading is not a
supported deployment path. Training metrics remain unrated until the neural
player is measured against pinned independent opponents under stated budgets.
KLENT's separate deployment-value pass reconstructs and validates the same full
legal action ordering as the actor pass, retaining its row order and chunk
boundaries, but skips candidate feature encoding and embedding. It trains only
the value head against the existing return targets. A CPU comparison on 128
saved corpus rows measured 0.857 to 0.423 seconds for this pass (2.03x), with
identical losses, value-head parameters and Adam states in that experiment.
This is a value-pass measurement, not an overall training or GPU speedup.
Reproduce it with `python -m tests.benchmark_value --corpus <directory>
--model <model.pt> --output <report.json>`.

## Primary relational policy/Q model

`relational_model.RelationalNet(ModelConfig())` implements the new primary
representation: width 256, eight residual blocks, eight attention heads,
feed-forward width 1,024, and 16 learned global tokens. It has **31,860,930
trainable parameters**. Each block performs local relational attention, stone
attention, global read/write attention, then local attention again. Legal-cell
representations persist through every block. Policy and bounded scalar-Q heads
return every legal action in native `Game.legal_moves()` order.

The encoder shares stone/cell identity across every nonempty six-cell window.
Window occupancy and incidence slots are tied under reversal; no absolute axis
embedding is used. Radius-eight stone/cell links and local legal-cell neighbors
retain geometry outside window coverage. Both the independent Python reference
and the native accelerator preserve all 12 board symmetries. The native encoder
is the explicit default of `NeuralEvaluator`; `backend="reference"` selects the
reference implementation, and a missing native library does not silently fall
back. Normal CMake builds include `hexo_graph`.

Training calls `encode(history)`, `pack(graphs, device)`, and `model(batch)`.
Outputs are flat FP32 `logits`, bounded FP32 `q`, and `action_offsets`.
`NeuralEvaluator(model, device).evaluate(histories)` returns ordered action,
logit, and Q arrays per position. Histories contain all placements, including
the origin. Native rules determine side and remaining placements; terminal
positions belong to the search implementation.

Default work budgets are 12,000 nodes and 600,000 relation traversals per batch
(local edges count twice). Whole graphs are batched in order; a single graph
exceeding the configured budget raises `WorkBudgetError`, never truncates its
actions. These are explicit resource limits, not game rules. Increase them only
after checking memory. Edge chunks control temporary allocations; autograd still
retains work proportional to edges times width. Block checkpointing, BF16
projections, and FP32 normalization, softmax, reductions, and residuals keep the
production model practical on the RTX 3070 Ti.

The production model completed forward/backward/Adam on a 150-stone position
with 6,166 nodes, 442,732 relation traversals, and all 3,938 legal outputs:
2.045 seconds, 2,272 MiB peak allocated and 3,072 MiB peak reserved. The device
reported 6,949 MiB free before the step; this was not a competing-load benchmark.
These measurements establish executable capacity, not playing strength.
`python -m tests.benchmark_relational --output <report.json>` reproduces the
fixture and records configuration, source hashes, and memory measurements.
Geometry, action indexing, 12 symmetries, batch isolation, native/reference
parity, and checkpoint gradients are covered by `tests.test_relational`.
Held-out prediction gains and superiority against independent bots remain to
be established by training and external evaluation.

### Shared relational action/value contract

The model, collector and neural search consume chronological axial histories,
replayed by the exact native rules. The resulting position is identified by
`position_key`: SHA256 of sorted `(q, r, absolute_owner)` stones plus the native
player to move and placements remaining. History order within an equivalent
position is not a distinct neural state. Coordinates retain their actual values;
this identity is not a symmetry-canonical cache key.

`Graph.actions` contains **all** native legal coordinates as `int64[N, 2]`, in
`Game.legal_moves()` order (lexicographic q, then r). `Graph.player` and
`Graph.remaining` are authoritative native phase values. Packing preserves that
order and supplies `action_owner[N]` and `action_offsets[B+1]`; offsets start at
zero, end at N, and delimit each position's complete action set. Newly reachable
second placements are encoded from the position after the first placement.

The differentiable network returns finite FP32 `logits[N]`, bounded scalar
`q[N]` in [-1, 1], and the same `action_offsets`. Each Q value uses the player to
move's perspective. The inference adapter echoes `position_key`, `player`,
`remaining`, aligned `actions`, and `model_version` for every record. Deployed
`model_version` is the SHA256 of the exact loaded relational `model.pt` bytes.
The loader validates those bytes before deserialization. No adapter may fall back
to a native NNUE model, handwritten values, or reordered/truncated actions.

Value constructions are deliberately distinct:

- KLENT collection uses `mu = softmax((Q + beta * log(pi)) / (alpha + beta))`
  and `V_mu = sum(mu * Q)` for frozen-actor return bootstraps.
- Neural search uses the network prior `pi = softmax(logits)` and
  `V_pi = sum(pi * Q)` for unresolved leaves.

Neither is a proven value. Actual terminal outcomes come from the rules; proofs
retain their separate verified scope. Backups keep the sign when the same player
continues a turn and reverse it only when control changes. A first-placement win
terminates immediately. Training stores the acting distribution and action index
with the full legal-coordinate hash, player and phase, then checks that identity
when reconstructing replay. These requirements apply equally to raw-policy,
KLENT-improved-policy, Gumbel-search and verified-tactics modes of the same player.
Each graph exposes `position_key`, a SHA256 of the rule identifier, native
player/remaining phase, and sorted absolute stone coordinates and owners.
Evaluator records echo that identity, `player`, `remaining`, and `model_version`
alongside the full native-order actions, logits, and Q. Deployed callers pass the
checkpoint SHA as `model_version`; standalone callers receive a configuration
and parameter digest. The evaluator copies and freezes the supplied model so
later training or checkpoint loads on the caller's model cannot change its identity.
Terminal inference is rejected because the rules and search own terminal values.


## Search-trained policy/value self-play

`search_train.py` trains the relational policy/value model from Gumbel MCTS self-play. The policy target is the full legal-action Gumbel completed-Q improved policy. The value target is the final game result from the player-to-move perspective. Capped games keep their policy targets but supply no outcome target. This path does not use KLENT action-Q targets or external bots for checkpoint selection.

```powershell
python search_train.py --run runs/search-selfplay --initial-model path/to/model.pt --games 128 --eval-games 32 --evaluate-every 4 --replay-positions 200000 --reuse-ratio 4 --iterations 100 --device cuda
python dashboard.py --run runs/search-selfplay --port 8766
```

A policy/Q warm start preserves its backbone and policy and initializes a new scalar value head at zero. Native trees batch neural leaves across games. Graph memory limits split whole positions without cropping legal actions. The collector enables immediate tactical constraints in search and its normalized policy targets. Ordinary Gumbel play also enables those constraints; deep proof search remains optional. The nominal 16 simulations are retained, including inexpensive exact backups. Evaluation keeps tactics disabled, preserving the original fixed protocol for measuring network progress.

Checkpoint 0 is the internal Elo anchor at zero. Every fourth candidate by default plays equal-search-budget, color-swapped games against the previous checkpoint, incumbent champion, and a distinct checkpoint near 20% behind the previous one, plus a smaller anchor comparison when the anchor is not already required. Duplicate opponents are played once. Promotion requires full matches and more wins than losses against both the incumbent and selected older checkpoint after counting capped games as candidate losses for this decision. There is no p-value promotion gate. When early history has no distinct older checkpoint, promotion waits. Learning continues from the latest candidate regardless of champion selection. Ratings describe this internal league and search budget, not an external leaderboard.

Rerun the same command to resume completed artifacts; only the requested iteration count may change. Saved corpora and optimizer checkpoints are verified. A partial fitting pass restarts from the preceding checkpoint rather than silently applying the same targets twice. Active run source files must remain unchanged.

### Recent replay and learning credit

Each collection defaults to 128 attempted games. Capped episodes keep searched policy rows; their value targets are masked. Up to 200,000 eligible positions are read from the most recent corpora, including earlier actors in the same run. The oldest admitted corpus is trimmed at the position limit. Corpus manifests bind actor hashes and targets; fitting records all source manifest hashes and the number of presentations by target age.

`--reuse-ratio 4` permits exactly four example presentations per fresh admitted position. The learner samples shuffled passes across replay until that budget is spent, including a smaller final minibatch. It does not run repeated full replay epochs. The replay learner uses all eligible rows, so it has no compulsory 25% holdout; historical held-out losses remain visible and are not extended with training losses. Model weights and Adam state continue from the latest checkpoint.

`--evaluate-every 4` runs internal matches periodically. Collection, fitting and evaluation remain synchronous on the single GPU, but evaluation no longer follows every fitting cycle. Intermediate checkpoints may be unrated until they participate in a later comparison. This is a throughput choice, not evidence that those checkpoints improved.

Run `python search_evaluate.py --run runs/gumbel-policy-value-v1 --from-checkpoint 13 --threads 4`
from a separate frozen checkout to evaluate intervening checkpoints while the GPU
learner continues. It uses the same search budget and previous/champion/older/reference
opponent allocation on CPU float32. Each complete color-swapped pair updates
`background-league.json`; the dashboard overlays these estimates without modifying
the trainer's league, champion, models, optimizer or corpus. Partial match sets are
labeled provisional and report played/planned counts. Fully terminal opening pairs
from a capped comparison still inform provisional Elo; pairs containing a cap do not.
CPU results show promotion evidence only after full
incumbent and distinct older matches both have winning conservative scores;
actual champion promotion still uses scheduled GPU matches.

CPU float32 and CUDA mixed-precision comparisons remain separate rating protocols.
Their conditional 95% credible intervals do not account for backend differences or
selection caused by capped games. Background evaluation can lag checkpoint creation;
queued checkpoints remain unrated until games supply evidence. Keep the worker source
unchanged while it runs. It resumes saved pairs, rejects changed identities and owns
an exclusive `background-evaluation.lock`.

The worker also samples up to 32 terminal games from each next collection, taking
three positions per game. It measures that collection's actor before it has trained
on those games. Fresh validation policy cross-entropy uses recorded search targets;
value MSE uses terminal outcomes, with a zero-value baseline of 1. These diagnostics
are conditional on the actor's terminal games and are separate from Elo and training
loss. They appear after the next corpus is complete. The dashboard labels historical
held-out validation separately from fresh-collection validation.

Exact cache keys now derive colored stones and turn phase from native legal histories without constructing another rules board. Full tuples still distinguish hash collisions. Persistent trees and model-versioned predictions retain their existing reuse semantics. Selective higher-budget targets and incremental neural trunks are not enabled.

### Moving internal opponents and league ratings

Search-training evaluation uses `--eval-games 32` against the previous checkpoint,
the current champion, and the checkpoint nearest 20% behind the previous one that
is distinct from both. It uses `--reference-games 8` against checkpoint 0 unless
that checkpoint is already required, in which case it gets the full allocation.
Duplicate opponents are played once. Self-play training still uses
the latest network on both sides with Gumbel search targets and terminal outcomes.

Displayed joint Elo fits fully terminal color-swapped pairs from current-protocol
comparisons, fixing checkpoint 0 at zero. The exact search-run4 (57df979) to
search-run5 (79751dd) promotion-only migration also accepts prior CUDA reports
after checking the hash-bound migration history, original report manifests,
model hashes, unchanged evaluation settings and the two audited source maps.
Their original protocols remain in the reports; CPU and earlier source revisions
are excluded. Completed pairs contribute even when other
pairs in that match cap; no outcome is invented for capped or unsaved games. Historical
ratings can change when new results arrive. Raw match scores and reference-only
estimates remain in `league.json`.

The displayed 95% intervals are approximate Bayesian credible intervals for the
joint model conditional on pair completion. The dashboard labels them provisional
where capped or unsaved pairs are present. Match score bounds run from known wins
divided by planned games to known wins plus all unknown outcomes divided by planned
games. These are deterministic observed-score bounds, not 95% sampling intervals.
New comparisons use distinct opponent-specific seeds; older comparisons may
share opening schedules, a dependence this approximation does not model across
comparisons. Promotion uses the separately recorded conservative two-opponent scores.

For live sparse-checkpoint estimates, run `python paired_rating.py --run runs/gumbel-policy-value-v1`
beside the evaluator. The dashboard prefers its `paired-ratings.json` output.
This fits actual opening-pair counts jointly, replacing the older projection of
independently smoothed matchup scores. In particular, one swept pair against a weak
reference is not converted into an artificial 70% observation that drags down a model
which tied a stronger champion. No extra games or pseudo-wins enter this likelihood.

For an Elo difference `d` in log-odds units, pair outcomes 0, 1, 2 wins have probabilities
`softmax(-h, tau, h)`, with `h = d/2 + asinh(exp(tau)*sinh(d/2)/2)`.
The expected game score is exactly `logistic(d)` for every `tau`. A shared dispersion
parameter permits more or fewer split pairs than independent games. Ratings have a
weak Normal(0, 1000 Elo) prior with checkpoint 0 fixed; `tau` has a Normal(log(2), 2)
prior. The point estimate is the joint posterior mode. A multivariate Student-t proposal
around that mode supplies 32,768 importance-weighted posterior samples for the 95%
credible interval; publication requires at least 1,000 effective samples. The dashboard
shows the number of rated and censored pairs involving each checkpoint, including
games as opponent.

These are model-dependent estimates. A single split pair means zero head-to-head Elo
difference with large uncertainty; other observed matchups can still change the joint
estimate. Only fully terminal pairs enter the likelihood. Completion may depend on
model strength, so conditional Elo and its credible interval can be biased for the
full match. CPU/CUDA differences and historical shared-opening dependence across
comparisons remain outside this uncertainty model.
The rating worker reads saved results only and changes neither promotion nor training.

To upgrade a completed older search run, reuse its original learning and search
arguments, add `--upgrade-run --reference-games 8`, and increase `--iterations`. Collection size, replay capacity/reuse, evaluation cadence and actor tactics may change during this explicit upgrade; network, optimizer hyperparameters and evaluation search settings must match. The default replay options apply to the upgrade.
The trainer requires a finished checkpoint boundary and an exclusive run lock.
It verifies and binds the old manifests in `history.json`, preserves model and
optimizer files, and records the new source identity. Normal later resumes omit
`--upgrade-run`. Never change the source checkout of an active trainer.

## Dense learner value targets

`dense_learn.py` derives the value target of every row from its episode
(`dense_data.value_targets`). Capped games always use TD(`td_lambda`) over the
searched root values. For finished games, `--value-target` chooses the target:

- `outcome` (the default): the hard result, 1 for the side that won and 0 for
  the side that lost.
- `td`: TD(`outcome_lambda`, default 0.98). The recursion is the capped-game one,
  started from the outcome at the last ply.
- `calibrated`: P(side to move wins | v, h), where v is the searched root value
  at the ply and h is the plies remaining. A null value carries the previous one
  forward along the game. Before the first search value, the target is the base
  rate.

The `calibrated` map is a ridge-regularised logistic regression that is shrunk
toward the base rate. Its inputs are the product of degree-1 B-splines in
log2(h) (knots 1, 2, 4, ..., 256) with [1, logit((1+v)/2)]. The learner fits it
from the newest `calibration_games` (4000) finished training games in the
replay window, at startup and again at every export, in about 2 s of CPU. Each
export trains the render workers on its new map until the next one. Where
search values carry no information, the map returns the base rate. Near the
end of a game it returns about the outcome. With fewer than 200 finished games,
`calibrated` falls back to `outcome`. Each checkpoint records its map as
`metrics.calibration`: the coefficients, plus a table over v in -1..1 in steps
of 0.25 and h in 0..160 in steps of 8.

With `--bootstrap-full-only`, all three chains (the capped TD chain, `td` and
`calibrated`) use only full-search root values. `--outcome-weight w` adds a
KataGo-style value-logit BCE against the hard outcome, with weight w. That head,
`outcome_bce`, is always logged. The validation curves by plies remaining
always score finished games against their hard outcome.

## Dense evaluator

`dense_eval.py loop` rates each new dense checkpoint against the champion and keeps the league in
`league.json`; its module docstring is the full contract, and every setting is an `EvaluationSettings` field in
`dense_config.py` (override per process with `--eval-*`).

- **Promotion** (`decision`, default `posterior`). One Bradley-Terry posterior covers every rated checkpoint, the
  candidate and Seal, and it uses every report: direct games, games against the previous champion, against Seal
  and against panel members. Each pair also gets a matchup deviation (prior sd `matchup_prior_elo`, default 30),
  so a pair's own games outweigh the transitive picture when the two disagree. The candidate needs at least
  `sprt_min_games` direct games, and its rating sd may be at most `uncertainty_parity` (1.5) times the champion's.
  It is promoted when it has the highest posterior rating and P(candidate - champion > `sprt_elo0`) is at least
  `promote_confidence`. It is rejected when that probability is at most 1 - `promote_confidence`. Neither
  happens while the direct-only and pooled estimates disagree beyond their intervals. `decision sprt` keeps the
  sequential test (`sprt_elo0` 0, `sprt_elo1` 25).
- **Continuous pool.** Like the actors, the evaluator keeps `pool_games` (64) games in flight on one engine. When
  a game ends, the next opening of its pairing starts at once (both colours together), so the GPU batch stays
  full. Each completed colour pair is written to its report immediately, so a restart loses only the games in
  flight, and the evaluation resumes where it stopped.
- **Direct games first.** While a promotion decision is pending, direct games against the champion take the
  whole pool until `sprt_min_games` (64) of them are complete. After that, at most `evidence_share` (1/4) of
  the pool may go to the evidence pairing whose games most reduce the posterior variance of the decision: the
  candidate or champion against the previous champion or Seal. The pairing is re-chosen after every
  completed colour pair. A newer checkpoint or a pairing change stops new games of the old pairing; its
  running games finish and count. A superseded decision settles on all of them: the candidate is promoted when
  P(candidate - champion > `sprt_elo0`) is at least `promote_confidence`, whether or not the other readiness
  conditions hold (`decision sprt` settles the same way). On every start the evaluator re-applies the rule to
  the reports on disk, and a rated checkpoint that already passes it is crowned at once.
- **Streaming.** `evaluator-status.json` carries the pool composition, the running tally of the current
  comparison (updated per finished game) and the pending verdict. The dashboard shows all three, including a
  provisional league row for the candidate.
- **Idle work.** After the decision, the evaluator plays the champion's Seal anchor, the adaptive panel (the
  rated checkpoints closest to the champion) and other optional comparisons. It then plays fill games until the
  next checkpoint appears: the champion against Seal until their interval is `anchor_target_halfwidth` narrow,
  `games` of the newest checkpoint against the previous champion, then the widest pair among the top
  `fill_top`. Pairings where either side's expected score exceeds `max_expected_score` are never played.
