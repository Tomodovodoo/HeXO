# Native engine

`src/hexo.cpp` is the rules engine every other part uses, exposed to Python through `python/hexo.py` as `Game`. The board is a sparse axial grid with 64-bit coordinates and no edge. Each placement updates the counts of the 18 six-cell windows it touches, which is what the one-turn tactics read: immediate wins, and the covers that block every opponent completion.

`Game.search(ms)` is the handwritten player: iterative deepening over complete turns with principal variation search, a window evaluator and an optional learned pattern table. It is what `python python/play.py` and the bot API serve when no checkpoint is given. It does not prove anything; its scores are never used as certificates.

```python
from hexo import Game

game = Game()
try:
    game.play(0, 0)
    for q, r in game.search(ms=100)["moves"]:
        game.play(q, r)
    print(game.state())
finally:
    game.close()
```

## Seal

Seal is the external bot the evaluator rates champions against. Build its adapter from an external checkout of [Ramora0/HexTicTacToe](https://github.com/Ramora0/HexTicTacToe):

```sh
git clone https://github.com/Ramora0/HexTicTacToe.git ../seal-reference
cmake -S . -B build -DHEXO_SEAL_SOURCE=../seal-reference
cmake --build build --config Release --parallel 2
```

This compiles `libhexo_seal` next to the engine without vendoring Seal. Then start training with `python python/bubble.py train --seal`, or give the evaluator its anchor settings by hand (`anchor_games`, `anchor_on_promotion`, `seal_ms` in `EvaluationSettings`). Seal's fixed array has a smaller coordinate range than ours; a game that leaves it is marked invalid, not won.

## Tests

```sh
python -m unittest discover -s tests -v
```

`tests/reference.py` is an independent Python implementation of the rules. The engine tests compare native rules and incremental features against it: radius-eight legality, turn phase, both colours, all three axes, first-placement wins, overlines, make and unmake with hash restoration, tactical wins and covers. `tests/gumbel.cpp` checks the search tree's halving schedule, sign handling and proof propagation against an independent enumeration, and CI runs the native subset on Linux.
