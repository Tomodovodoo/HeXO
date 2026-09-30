# HeXO · Bubble

Bubble is a bot for [HeXO](https://github.com/HeXO-Game/HeXO), with a native rules engine, neural search, a verified tactical solver and self-play training. HeXO remains the game and repository name. Bubble's name comes from the Pokémon move; future bots can follow the same theme, such as Wave and Waterfall.

Player one opens at the origin. Players then alternate two placements. Each stone must be empty and within hex distance eight of an existing stone. Six or more consecutive stones along any axis wins, including on the first placement of a turn.

## Build and play

Install Python 3.10 or newer, CMake 3.20 or newer, and a C++20 compiler. Run these commands from the checkout root:

```sh
git clone https://github.com/Tomodovodoo/HeXO.git
cd HeXO
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on Linux/macOS. Then:

```sh
python -m pip install -e .
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 2
python python/play.py
```

Open <http://127.0.0.1:8765>. The initial opponent uses the handwritten native evaluator. Click a cell or enter axial coordinates, drag to pan, and scroll to zoom. Bubble plays one complete turn; undo removes one stone.

On Windows with MinGW, add `-G "MinGW Makefiles"` to the configure command and keep `g++` on PATH. Visual Studio builds load from `build/Release/`. The editable install registers the Python modules while keeping native libraries and browser assets in this checkout. A checkout or ZIP download contains source, fixtures and the opening suite. Trained checkpoints and native binaries are generated locally.

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

## Train Bubble

The current stack uses the dense HexNet policy/value model. Install its dependencies:

```sh
python -m pip install -r requirements/learning.txt
```

For NVIDIA training, install a compatible CUDA PyTorch wheel using the [PyTorch installation selector](https://pytorch.org/get-started/locally/). The tested Windows environment uses PyTorch 2.11.0 with CUDA 12.6, NumPy 2.4.3 and SciPy 1.17.1. [requirements/tested.txt](requirements/tested.txt) pins those Python library versions; choose the PyTorch wheel index for your hardware.

Create the run and its initial checkpoint before starting actors:

```sh
python python/dense_config.py --run runs/bubble --device cuda
python python/dense_learn.py --run runs/bubble --steps 0
```

Start each long-running command in a separate terminal with the same activated environment:

```sh
python python/dense_selfplay.py --run runs/bubble --processes 4
python python/dense_learn.py --run runs/bubble
python python/dense_eval.py loop --run runs/bubble --anchor-games 0 --no-anchor-on-promotion --anchor-target-halfwidth 0
python python/dashboard.py --run runs/bubble
```

The dashboard opens at <http://127.0.0.1:8766>. The actors write game shards, the learner resumes from its latest export, and the evaluator decides checkpoint promotion. Generation and training share the GPU according to the configured pacing settings. Use `--device cpu` when creating a CPU run. A short CPU example is in [dense training](docs/dense-training.md#short-cpu-run).

For the supported GPU speedups, add `--net-kernels fused` to the actor, learner and evaluator commands. Actors can also add `--cuda-graphs`. Fused kernels need Triton; Windows uses the [triton-windows package](https://github.com/triton-lang/triton-windows). Installation, measured costs and reproduction commands are in [GPU kernels](docs/gpu-kernels.md#enable-and-reproduce). Reference kernels work without Triton.

The optional tactical solver needs Rust/Cargo with edition 2024 support:

```sh
python tools/build_tactical.py
```

This writes the native solver and its source/binary manifest under `tools/tactical/target/release/`. Solver budgets default to zero in a new run. See [solver scheduling](docs/tactical-solver.md#dense-actor-solver-scheduling) for enabling proof queries and proof following. External opponents such as Seal and Strix have separate setup instructions; they are optional.

## Play a trained checkpoint

```sh
python python/play.py --dense-run runs/bubble --device cpu
```

The model picker offers up to four available exports, including the champion and newest. Search and solver have separate toggles and budgets. Suggested moves, verified winning lines and opponent threats appear through the analysis button. Use `--device cuda` for GPU inference. Proof analysis requires the tactical build above; turn the solver off to play without it.

## Reproduce a run

Keep the Git revision, dependency versions, `config.json`, launch flags and native build metadata with your results. Flags override settings for a process, so record them alongside the configuration. The actor and learner must use the same validation fraction. Restart sampling excludes validation games and descendants of those games.

`runs/<name>/` contains game shards, model/EMA/optimizer checkpoints, evaluation games, metrics and status files. Use a new directory for a new experiment. Resuming the learner restores weights and optimizer state from the newest complete export; steps since that export must be trained again. Configurations, seeds, search budgets and model hashes identify the inputs to a comparison. Adaptive solver scheduling depends on available CPU time, so identical seeds alone do not guarantee identical games.

The local training run, downloaded opponent weights and research results are gitignored. To repeat a published measurement, you need the named checkpoint and shard inputs as well as the code. The [evaluation documentation](docs/dense-evaluation.md) describes ratings and decision checks. Training games reaching their placement cap have no outcome label. Evaluation scores capped games as half a point without claiming a proven draw.

## Repository layout

| Folder | Contents |
|---|---|
| `python/` | Current dense training, HexNet, rules bindings, search and server commands |
| `python/legacy/` | Earlier pattern, NNUE and relational pipelines, including numerical helpers still shared with dense training |
| `src/` | C++ rules, graph encoder and Gumbel search |
| `tools/` | Build helpers, profilers and external opponent adapters |
| `tools/tactical/` | Rust tactical solver, independent checker and vendored Strix sources |
| `tests/` | Existing correctness tests and committed fixtures |
| `web/` | Local game and training dashboard assets |
| `openings/` | Committed evaluation opening suite |
| `docs/` | Model, solver, data and evaluation details |
| `requirements/` | Learning dependencies and tested versions |
| `build/`, `runs/`, `artifacts/` | Generated native files, training data and local results, all gitignored |

## Tests

After the native build and editable install, install NumPy for the CPU search tests:

```sh
python -m pip install numpy
python -m unittest tests.test_curriculum tests.test_proof tests.test_notation_api tests.test_neural_search -v
```

Dense tests also need the learning dependencies:

```sh
python -m unittest tests.test_dense tests.test_dense_solve tests.test_openings -v
```

For local development beside a live run, use `OMP_NUM_THREADS=2`, hide CUDA from test processes and run them at BelowNormal priority on Windows. Native binaries can be copied from a compatible existing build instead of rebuilt. Keep the tactical DLL and its matching JSON manifest together. Never remove a worktree that supplies a running process or a loaded native library.

## More documentation

- [Dense training and value targets](docs/dense-training.md)
- [Dense evaluation, opening books and ratings](docs/dense-evaluation.md)
- [Verified tactical solver](docs/tactical-solver.md)
- [Neural search](docs/neural-search.md)
- [GPU kernels and profiling](docs/gpu-kernels.md)
- [Human corpus import](docs/human-corpus.md)
- [Native engine and opponent matches](docs/native-engine.md)
- [Notation and stateless API](docs/notation-api.md)
- [Earlier pattern, NNUE and relational experiments](docs/earlier-models.md)
