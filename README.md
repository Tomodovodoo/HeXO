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

The first learned evaluator is deliberately small. An `18 -> 32 -> 1` network maps six-cell ternary patterns to a bounded residual on top of the hand-written evaluator. Training pools every occupied six-cell window without cropping the board. Color antisymmetry and line reversal are enforced. The 729-entry integer table is exported for native incremental evaluation and move ordering. No Python or GPU calls occur inside search. This is the initial six-cell learner, not yet the larger length-11 NNUE model from the project plan.

Each iteration:

1. Play the incumbent against itself and earlier promoted checkpoints, with varied openings.
2. Store positions, search targets, game outcomes, and complete move histories.
3. Fit a challenger on the recent replay data. Entire opening families stay together; the 12 hex symmetries determine family identity. One fifth of family hash buckets is reserved for validation.
4. Evaluate the challenger against the incumbent, the original checkpoint, and an older promoted checkpoint when available. These openings include radius-three placements, distinct from radius-two training openings, and are paired with colors exchanged.
5. Promote only after a positive lower game-level confidence bound and an opening-pair sign test. The significance budget shrinks across successive attempts to limit false promotions. Incomplete evaluation games prevent promotion. Clear losses against older checkpoints also prevent promotion.

Elo is estimated against the frozen original checkpoint, whose rating is zero. It is a within-run estimate, not a site leaderboard rating. Graphs include rejected challengers and approximate game-level confidence intervals. The pair test, rather than the graph alone, governs promotion. A small evaluation may be unable to establish improvement. No run is guaranteed to produce a stronger checkpoint.

Run directories contain `summary.json`, append-only `events.jsonl`, replay batches, match histories, and checkpoint model/table files. Repeat the same command to perform additional iterations with the same configuration. Engine or training-source changes require a new run directory, keeping ratings comparable. A lock prevents concurrent trainers from writing the same run. If a process is forcibly killed, verify it has stopped before removing its stale `training.lock`.

To play the latest promoted checkpoint:

```sh
python play.py --run runs/selfplay
```

For a short end-to-end run, use a separate directory:

```sh
python train.py --run runs/short-run --iterations 1 --games 12 --eval-games 8 --ms 10 --eval-ms 20 --epochs 4 --workers 2
```

Eight evaluation games cannot pass the default first-iteration pair-test significance threshold. This command checks collection, training, evaluation, and rejection behavior; it cannot establish competitive strength.

## Engine

- Sparse axial board with signed 64-bit coordinates. The public API accepts coordinates within +/- 10^12 to keep arithmetic safe. There is no fixed board crop.
- Incremental counts and evaluation for the 18 six-cell windows touched by each placement. Tactical completion sets include broken lines anywhere on the board.
- Immediate wins take priority. Defensive covers intersect every opponent one-turn completion set; unused defensive placements are searched for development and counterattacks.
- Conditional first and second placements, followed by deduplication of resulting positions. A move at `(8, 0)` can make `(16, 0)` legal on the same turn.
- Iterative deepening over complete turns, principal variation search, and transposition bounds.
- A hand-written window evaluator plus an optional learned pattern residual, updated on make/unmake and loaded through `Game.load_table()`.

The legal environment is exact within its integer representation. Search is selective: ordinary candidates come from nearby cells and promising lines, and quiet turns are shortlisted. It does not prove game-theoretic wins. A mate-like search score is not a proof certificate. Deadlines are checked during search; setup and an individual candidate-generation operation can exceed very small budgets. Search metadata includes total native elapsed time.

## Opponent matches

```sh
python arena.py --opponent shallow --games 20 --ms 100
python arena.py --opponent random --games 20 --ms 100
```

The arena alternates colors and reuses each opening for a pair of games. Results contain moves, actual decision times, engine hashes, and a Wilson interval. Move-cap truncations and invalid games are recorded separately from wins and losses. The current interval treats games as independent; use a larger opening-family analysis before making strength claims.

To compare against Seal, clone its source outside this repository, then configure the optional adapter:

```sh
git clone https://github.com/Ramora0/HexTicTacToe.git ../seal-reference
cmake -S . -B build -DHEXO_SEAL_SOURCE=../seal-reference
cmake --build build --config Release -j 4
python arena.py --opponent seal --games 20 --ms 100 --output artifacts/seal.json
```

The adapter compiles the external engine without vendoring it. Seal's fixed array has a smaller coordinate range; games outside the adapter's safe range are marked invalid rather than counted as victories. Equal requested budgets are used, and both engines' actual elapsed times are retained.

## Status and remaining work

The initial non-neural version is playable. Direct runtime checks have covered the radius-eight frontier, sequential expansion, immediate wins, and 800 make/unmake comparisons. A differential run matched 2,019 transitions against the official TypeScript rules, including rejected moves, turn phase, cells, and winner. Sparse expansion to coordinate 800 also passed. An initial eight-game development comparison against Seal scored two wins and six losses at 100 ms per turn. This is an initial measurement, not a competitive-strength claim. Two tactical-extension experiments scored zero wins in the same eight openings and were removed.

The first complete learning run collected 516 positions from 12 games and trained on the RTX 3070 Ti. Its challenger scored three wins, four losses, and one incomplete evaluation game and was rejected. That demonstrates the loop, not a strength gain. Larger runs are needed to measure improvement.

Further work includes training-quality improvements, search profiling, stronger threat search, larger held-out opponent matches, Orca integration, and game-data import. Match clocks, rated lobbies, and online account play are not part of the local board yet.

## References

Rules and turn semantics were checked against the [official HeXO source](https://github.com/HeXO-Game/HeXO/blob/1aea2b676733f8cdc53f8f92cb00b367af72111c/packages/shared/src/sharedTypes.ts).

Independent opponents: [Seal](https://github.com/Ramora0/HexTicTacToe) and [Orca framework](https://github.com/Saiki77/hexbot-building-framework). The site bundles a Seal WebAssembly build; the optional native adapter currently compares against the selected external source revision, which may differ from that build.

The intended learned evaluator follows ideas from [Rapfi](https://github.com/dhbloo/rapfi) and [NNUE](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html), adapted to Hexo's three axes and turn semantics. Their existing game-specific weights are not used.
