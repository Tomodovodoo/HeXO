# Bubble

Bubble is a self-play bot for [HeXO](https://github.com/HeXO-Game/HeXO), the hexagonal connect-six game this repository is named after.

## What Bubble is

Bubble is a KataGo-style asynchronous self-play engine that trains on one GPU. Separate processes share a run directory:

- **Network.** A hex-masked ResNet with policy and value heads over bucketed board crops (`python/hexnet.py`, `python/hexcrop.py`).
- **Actors.** Gumbel MCTS self-play with many games batched through the network (`python/dense_selfplay.py`, `src/gumbel.cpp`).
- **Learner.** Trains on the actors' game shards through a KataGo-style replay window and exports checkpoints (`python/dense_learn.py`, `python/dense_data.py`).
- **Evaluator.** Plays paired games from an opening book, rates checkpoints with a posterior Bradley-Terry model and decides which checkpoint the actors use (`python/dense_eval.py`, `python/dense_posterior.py`, `python/dense_openings.py`).
- **Tactical solver.** A Rust forced-win solver with an independent certificate checker (`tools/tactical/`). The search calls it during play (`python/dense_solver.py`), and an offline proof pass labels finished games with proven results (`python/dense_solve.py`).
- **Dashboard.** A local web page with training, evaluation and proof-pass status (`python/dashboard.py`).

## Build

Requirements:

- Python 3.10 or newer
- CMake 3.20 or newer and a C++20 compiler
- Rust and Cargo with edition 2024 support, for the tactical solver
- PyTorch 2.11 or newer, for training and neural play

```sh
git clone https://github.com/Tomodovodoo/HeXO.git
cd HeXO
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` in PowerShell, `.venv\Scripts\activate.bat` in cmd.exe, `source .venv/bin/activate` on Linux and macOS), then build:

```sh
python -m pip install -e .
python -m pip install -r requirements/learning.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 2
python tools/build_tactical.py
```

For CUDA, install the PyTorch wheel for your hardware from the [PyTorch selector](https://pytorch.org/get-started/locally/). `requirements/tested.txt` pins the tested library versions. With MinGW on Windows, add `-G "MinGW Makefiles"` to the configure command and keep `g++` on `PATH`.

## The game

HeXO is played by two players, Cross (X) and Circle (O), on an unbounded grid of hexagons. Cells use axial coordinates `(q, r)`. The distance between two cells is

```
distance = max(|q1 - q2|, |r1 - r2|, |(q1 + r1) - (q2 + r2)|)
```

so the six neighbours of `(0, 0)` are `(1, 0)`, `(-1, 0)`, `(0, 1)`, `(0, -1)`, `(1, -1)` and `(-1, 1)`.

Turns:

1. Cross opens with a single stone, and it must go on the origin `(0, 0)`.
2. From then on the players alternate, starting with Circle, and each turn is two placements by the same player.
3. A placement must go on an empty cell within distance 8 of any stone already on the board, of either colour. The board grows as stones spread; there is no fixed edge.

Winning:

- A player wins with six or more of their own stones in an unbroken line along one of the three hex axes: `(1, 0)`, `(0, 1)` or `(1, -1)`.
- The win counts the moment the sixth stone lands. If the first placement of a turn completes a line, the game ends there and the second placement is never made.
- There are no captures and no passes.

A short example. Cross opens at `(0, 0)`. Circle plays `(1, 0)` and `(0, 1)`. Cross plays `(-1, 0)` and `(-2, 0)`, making three in a row on the `(1, 0)` axis. Circle's stone at `(1, 0)` already blocks that line on the right, so Cross can only reach six by extending left to `(-5, 0)`. Because each turn adds two stones, a player who has four in a row with both ends open threatens to finish on the next turn, and the opponent needs both of their placements to block. Much of the game is about building two such threats at once.

The rules engine is `src/hexo.cpp`. `tests/reference.py` is an independent Python implementation used to check it.


## Test

```sh
python -m unittest tests.test_engine tests.test_proof tests.test_notation_api tests.test_neural_search -v
python -m unittest tests.test_dense tests.test_dense_solve tests.test_openings -v
```

The first line needs the native build and NumPy. The second also needs PyTorch.

## Train

Create a run and its first checkpoint:

```sh
python python/dense_config.py --run runs/bubble --device cuda
python python/dense_learn.py --run runs/bubble --steps 0
```

Then start each process in its own terminal:

```sh
python python/dense_learn.py --run runs/bubble
python python/dense_selfplay.py --run runs/bubble --processes 4
python python/dense_eval.py loop --run runs/bubble --eval-anchor-games 0 --no-eval-anchor-on-promotion --eval-anchor-target-halfwidth 0
python python/dense_solve.py --run runs/bubble
python python/dashboard.py --run runs/bubble
```

The dashboard is at <http://127.0.0.1:8766>. `config.json` in the run directory holds the settings of the actors, learner and evaluator. A flag on one process overrides the matching setting for that process only. The proof pass takes its settings from its own flags only, so record them with the run.

The three evaluator flags turn off rating games against Seal, an external bot that needs its own setup. Use `--device cpu` in `dense_config.py` to train without a GPU. The proof pass needs the tactical build. The solver inside the search is off by default; see [docs/tactical-solver.md](docs/tactical-solver.md) to enable it.

## Play

```sh
python python/play.py                                         # handwritten native evaluator
python python/play.py --dense-run runs/bubble --device cpu    # trained network from a run
```

Open <http://127.0.0.1:8765>. Each side is a person or any engine found under `models/` and `runs/`; see [docs/play.md](docs/play.md).

From Python:

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

## Notation and bot API

`python/notation.py` reads and writes [HTTTX notation v1](https://github.com/hex-tic-tac-toe/hexagonal-tic-tac-toe-notation/tree/15bb7877ae020d661497e332adf0810d00d24e3e) and checks every move for legality:

```sh
python python/notation.py import match.txt > match.json
python python/notation.py export match.json > match.txt
```

`python/bot_api.py` serves the [HTTTX stateless bot API](https://github.com/hex-tic-tac-toe/htttx-bot-api/tree/37d2385f1016abe8b25798238a7d0c4a17a25dda/definitions) on localhost with `GET /capabilities.json` and `POST /stateless/v1-alpha/turn`:

```sh
python python/bot_api.py --port 8790 --ms 100
```

Both formats record two placements per turn, so they cannot express a win on the first placement of a turn. Export raises `NotationConflict` and the API returns HTTP 409 in that case. Details are in [docs/notation-api.md](docs/notation-api.md).

## Documentation

- [Dense training and value targets](docs/dense-training.md)
- [Dense evaluation, opening books and ratings](docs/dense-evaluation.md)
- [Six engine protocol and arena matches](docs/six-engine.md)
- [Tactical solver](docs/tactical-solver.md)
- [Neural search](docs/neural-search.md)
- [GPU kernels](docs/gpu-kernels.md)
- [Human corpus import](docs/human-corpus.md)
- [Native engine and opponent matches](docs/native-engine.md)
- [Notation and bot API](docs/notation-api.md)
- [Earlier NNUE and relational models](docs/earlier-models.md)

## Repository layout

| Path | Contents |
|---|---|
| `src/` | C++ rules engine, native Gumbel search and graph encoder |
| `python/` | Network, actors, learner, evaluator, proof pass, dashboard, browser game and bindings |
| `python/legacy/` | Earlier NNUE and relational models; the dense code still imports a few shared helpers from here |
| `tools/tactical/` | Rust tactical solver, certificate checker and the vendored hexo-strix crates |
| `tools/` | Build scripts, profilers and adapters for external opponents |
| `web/` | Browser game and dashboard pages |
| `openings/` | Fixed opening suite for evaluation |
| `tests/` | Unit tests, fixtures and the Python reference rules |
| `docs/` | Detailed notes on training, evaluation, the solver, GPU kernels and earlier models |
| `requirements/` | Training dependencies and tested versions |

`build/`, `runs/` and `artifacts/` hold native builds, training data and local results. Git ignores all three.

## References

- Danihelka et al., [Policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO), ICLR 2022. The root search and completed-Q policy targets follow this paper and [DeepMind's mctx](https://github.com/google-deepmind/mctx).
- Wu, [Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565), 2019 (KataGo). Playout-cap randomization and the replay window follow this paper.
- [SootyOwl/hexo-strix](https://github.com/SootyOwl/hexo-strix), MIT licensed. Its solver crates are vendored in `tools/tactical/vendor/hexo-strix` and modified.
- [Official HeXO source](https://github.com/HeXO-Game/HeXO/blob/1aea2b676733f8cdc53f8f92cb00b367af72111c/packages/shared/src/sharedTypes.ts), used to check the rules.
