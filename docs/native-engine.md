# Native engine

`src/hexo.cpp` is the rules engine every other part uses, exposed to Python through `python/hexo.py` as `Game`. The board is a sparse axial grid with 64-bit coordinates and no edge. Each placement updates the counts of the 18 six-cell windows it touches, which is what the one-turn tactics read: immediate wins, and the covers that block every opponent completion.

`Game.search(ms)` is the handwritten player: iterative deepening over complete turns with principal variation search, a window evaluator and an optional learned pattern table. It is what `python python/play.py` and the bot API serve when no checkpoint is given. Its bounded tactical search proves continuous-threat wins and retains their winning continuations between turns. Ordinary heuristic scores are not certificates.

Native updates candidate scores only along the lines changed by a placement. It
ranks the second stone from an exact evaluation delta, avoiding a full make/unmake
for every proposed pair. The candidate cache lasts for one search. Ordinary
Native search uses no transposition table. The experimental Python
`tt_injection=True` option keeps its separate previous-iteration move hints;
it is off by default and never caches score bounds.
The existing line-count arithmetic is precomputed into 1 KiB of immutable
constants. Without optional pattern adjustments, candidate updates skip that
adjustment arithmetic and windows whose gains did not change. Candidate selection
sorts only the retained cells, preserving their scores and tie order.
The root prover can also cover an attack that leaves the defender a free second
stone. It first finds a strategy against the mandatory block and records which
empty cells can affect its moves or threats. Fillers outside that set share the
strategy; every remaining legal filler is checked separately. These probes share
the existing root proof time budget, and retained proof storage stays capped at
256 KiB.

No weights, model, external solver or new dependency is required.
The optional NNUE search keeps its existing candidate evaluation.

## Play on HeXO Arena

Native can play through the [HeXO Bot API](https://github.com/TimmyBurn2/Hexo-Bot-Api).
The engine runs on your computer; the site hosts the games and their live boards.
Build Native as above and install the existing API extra with
`python -m pip install -e ".[api]"`, then run:

```sh
python python/arena_bot.py https://hexo.seeligto.de --ms 100
```

Enter the bot token at the hidden prompt, or supply it in `HEXO_TOKEN`.
The client declares its supported clocks, opens its presence stream, accepts
challenges and plays games started from its bot page. It uses CPU only, with
100 ms per complete turn by default, leaving time for transport when the server
supplies a smaller allowance. `--width` and `--depth` set Native's search limits.
Each game's worker retains Native's proven continuations. Keep the client running
to stay online; Ctrl+C disconnects it. `HEXO_NATIVE_DIR` selects a particular build.
The client prints game IDs and results; the site keeps the games for viewing.

Native values immediate pressure by the placements needed to block it. Several
four- or five-stone windows that share a blocking cell count as one obligation;
otherwise the heuristic counts two. Search checks unavoidable wins separately.
The score also rewards the mover's three-stone setups, limited by the placements
left after mandatory defence. This accounts for initiative without rewarding an
attack the mover has no time to develop. These are positional estimates, not
proofs. The tactical prover and the public `Game.turns` and `Game.evaluation`
APIs keep their previous feature/table scores.

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

Compare independently built Native versions in one process, with equal time per
turn and swapped colours on each opening:

```sh
PYTHONPATH=python:. HEXO_NATIVE_DIR=build python -m tests.benchmark \
  --compare-library ../old-native/build/libhexo.so \
  --games 64 --ms 25 --width 16 --seed 810223 \
  --output artifacts/native-comparison.json
```

Use `--seal-library path/to/libhexo_seal.so` instead for Seal. On Windows set
`PYTHONPATH` to `python;.` and `HEXO_NATIVE_DIR` to the candidate build directory,
and supply the opponent DLL path. Both libraries stay loaded between turns;
there is no subprocess or server startup for each move. The report saves library
digests, every placement and search timing, paired scores and capped games. Caps
are reported separately and counted as half a point, not as proven draws.
Seal range exits remain in the report as invalid games; the comparison continues
and excludes incomplete pairs from paired statistics.

With `--compare-library` and no `--games`, the comparison uses saved real
positions. Seal comparisons require paired games or a recorded `--trace`.
Add `--depth 2
--ms 10000 --positions 28 --repeats 2` to compare completed work rather than a
time budget; check identical moves, scores, nodes and depths before interpreting
the timing difference. The same report measures complete-turn generation.

```sh
python -m unittest discover -s tests -v
```

`tests/reference.py` is an independent Python implementation of the rules. The engine tests compare native rules and incremental features against it: radius-eight legality, turn phase, both colours, all three axes, first-placement wins, overlines, make and unmake with hash restoration, tactical wins and covers. `tests/gumbel.cpp` checks the search tree's halving schedule, sign handling and proof propagation against an independent enumeration, and CI runs the native subset on Linux.
