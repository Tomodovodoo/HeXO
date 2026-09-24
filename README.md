# HeXO

C++20 Hexo rules and search engine, Python interface, and local browser game.

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

## Engine

- Sparse axial board with signed 64-bit coordinates. The public API accepts coordinates within +/- 10^12 to keep arithmetic safe. There is no fixed board crop.
- Incremental counts and evaluation for the 18 six-cell windows touched by each placement. Tactical completion sets include broken lines anywhere on the board.
- Immediate wins take priority. Defensive covers intersect every opponent one-turn completion set; unused defensive placements are searched for development and counterattacks.
- Conditional first and second placements, followed by deduplication of resulting positions. A move at `(8, 0)` can make `(16, 0)` legal on the same turn.
- Iterative deepening over complete turns, principal variation search, and transposition bounds.
- A hand-written window evaluator. No neural model is loaded or trained yet.

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

Further work includes search profiling, stronger threat search, larger held-out opponent matches, Orca integration, game-data import, and learned incremental evaluation. Match clocks, rated lobbies, and online account play are not part of the local board yet.

## References

Rules and turn semantics were checked against the [official HeXO source](https://github.com/HeXO-Game/HeXO/blob/1aea2b676733f8cdc53f8f92cb00b367af72111c/packages/shared/src/sharedTypes.ts).

Independent opponents: [Seal](https://github.com/Ramora0/HexTicTacToe) and [Orca framework](https://github.com/Saiki77/hexbot-building-framework). The site bundles a Seal WebAssembly build; the optional native adapter currently compares against the selected external source revision, which may differ from that build.

The intended learned evaluator follows ideas from [Rapfi](https://github.com/dhbloo/rapfi) and [NNUE](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html), adapted to Hexo's three axes and turn semantics. Their existing game-specific weights are not used.
