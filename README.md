# HeXO

C++20 Hexo rules and search engine, Python interface, and local browser game.

The current priority is a self-play learning loop with a local experiment dashboard. Checkpoints earn promotion through matches against frozen opponents; fitting loss alone never replaces the incumbent.

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

GPU actors use exact sparse rules with growing coordinate storage and incremental native-compatible pattern features. They first take exact one- or two-stone wins, then choose a placement belonging to a complete cover of all immediate opponent threats when such a cover exists. These choices override exploration. Quiet candidates combine nearby cells, axial development, and broader exploration, scored one placement at a time by the learned evaluator. They do not run native PVS; `--ms` and `--width` do not control this actor. Replay records identify the actor and one-placement target semantics. Promotion always measures the native deployed engine at equal wall-clock budgets.

On this RTX 3070 Ti, the tactical zero-residual GPU actor generated about 184 games/second and 28,300 placements/second at batch 512 with 32 quiet candidates and a 256-stone cap. Of those 512 games, 388 finished and 124 reached the cap; peak PyTorch allocation was about 920 MiB. These timings include generation and replay transfer, but not training or native evaluation. The earlier actor was faster but frequently missed immediate wins. Its shorter games make raw games/second an unfair performance comparison. Reproduce on your hardware:

```sh
python gpu_benchmark.py --actor --batches 64 256 512 2048 --placements 256
python gpu_benchmark.py --batches 64 512 2048 --placements 96
```

The second command compares only exact rule transitions and feature updates with C++. Small GPU batches are slower than native code. The dashboard exposes device utilization, VRAM, power, and temperature; telemetry includes other applications using the GPU. A small pattern learner cannot productively saturate every GPU unit at all times, so larger game batches and measured throughput guide settings.

The first learned evaluator is deliberately small. An `18 -> 32 -> 1` network maps six-cell ternary patterns to a bounded residual on top of the hand-written evaluator. Training pools every occupied six-cell window without cropping the board. Color antisymmetry and line reversal are enforced. The 729-entry integer table is exported for native incremental evaluation and move ordering. No Python or GPU calls occur inside search. This is the initial six-cell learner, not yet the larger length-11 NNUE model from the project plan.

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

The adapter compiles the external engine without vendoring it. Seal's fixed array has a smaller coordinate range; games outside the adapter's safe range are marked invalid rather than counted as victories. Equal requested budgets are used, and both engines' actual elapsed times are retained. `--run` loads the promoted checkpoint, `--checkpoint` selects another saved candidate, and `--table` loads a standalone export. Reports identify the loaded table and its hash. Without a model option the arena uses the original evaluator.

To compare against the published Orca model, use an external checkout:

```sh
git clone https://github.com/Saiki77/hexbot-building-framework.git ../orca-reference
python arena.py --opponent orca --orca-source ../orca-reference --orca-sims 200 --games 20 --ms 100 --max-stones 800 --output artifacts/orca.json
```

This requires PyTorch. The adapter strictly loads the checkout's seven-channel `orca/checkpoint.pt` without adding random weights. Use `--orca-checkpoint` to select another compatible checkpoint and `--orca-device cuda` for GPU inference. The report records source revision, checkpoint hash, simulation budget and actual turn times. Orca receives simulations per placement; our engine receives milliseconds per complete turn. This comparison does not use equal time budgets. Native rules validate every returned move, and replay disagreements remain invalid games rather than wins.

## Correctness tests and local benchmarks

Build the native library with the CMake commands above, then run:

```sh
python -m unittest discover -s tests -v
python -m tests.benchmark --positions 12 --ms 5 --reference-ms 25
```

The tests compare native rules and incremental features with an independent Python board reference. They cover sequential radius-eight legality, turn phase, both colors, all three winning axes, first-placement wins, overlines, distant expansion, make/unmake and hash restoration, residual-table bounds, tactical wins and defensive covers. With PyTorch installed, the same batched-environment checks run on CPU and on CUDA when available, including capacity growth, reset and truncation. GPU checks are skipped when PyTorch is unavailable. The NNUE implementation adds separate export/value/policy checks. Pull requests run the native rules subset on Linux; CPU/CUDA tensor parity is also run locally.

The benchmark prints JSON with seeded positions, source and library hashes, hardware, actual search times, nodes and agreement with a wider search. Timing-dependent search results can vary across runs. There are no machine-specific speed assertions. A wider search is a selective reference, not a proof of the best move. Candidate-cell recall is reported only when the native candidate API is available; it does not measure whether the final pruned turn list retained that pair.

## Status and remaining work

The initial non-neural version is playable. Direct runtime checks have covered the radius-eight frontier, sequential expansion, immediate wins, and 800 make/unmake comparisons. A differential run matched 2,019 transitions against the official TypeScript rules, including rejected moves, turn phase, cells, and winner. Sparse expansion to coordinate 800 also passed. An initial eight-game development comparison against Seal scored two wins and six losses at 100 ms per turn. This is an initial measurement, not a competitive-strength claim. Two tactical-extension experiments scored zero wins in the same eight openings and were removed.

The first complete learning run collected 516 positions from 12 games and trained on the RTX 3070 Ti. Its challenger scored three wins, four losses, and one incomplete evaluation game and was rejected. That demonstrates the loop, not a strength gain. Larger runs are needed to measure improvement.

A subsequent native-actor run collected 10,130 positions from 192 games. None of its three candidates earned promotion. The GPU pipeline has also completed generation, fitting, native evaluation, checkpoint persistence, and resumed training with optimizer lineage. GPU trajectories and feature targets have been replayed against the native engine. No convincing Elo gain has been established yet. The first 6,144-game GPU run produced 138,652 positions. Its best initial candidate scored 28 wins in 40 games, then only 78 wins in 160 fresh confirmation games. An audit found that 9.03% of its replay positions had a provable win that the shallow actor later lost. The tactical actor fixes those missed wins; use a fresh run directory for its data.

Further work includes training-quality improvements, search profiling, stronger threat search, larger held-out opponent matches, Orca integration, and game-data import. Match clocks, rated lobbies, and online account play are not part of the local board yet.

## References

Rules and turn semantics were checked against the [official HeXO source](https://github.com/HeXO-Game/HeXO/blob/1aea2b676733f8cdc53f8f92cb00b367af72111c/packages/shared/src/sharedTypes.ts).

Independent opponents: [Seal](https://github.com/Ramora0/HexTicTacToe) and [Orca framework](https://github.com/Saiki77/hexbot-building-framework). The site bundles a Seal WebAssembly build; the optional native adapter currently compares against the selected external source revision, which may differ from that build.

The intended learned evaluator follows ideas from [Rapfi](https://github.com/dhbloo/rapfi) and [NNUE](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html), adapted to Hexo's three axes and turn semantics. Their existing game-specific weights are not used.
