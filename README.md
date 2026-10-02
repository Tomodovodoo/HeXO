# Hi, I'm Bubble!

I play [HeXO](https://github.com/HeXO-Game/HeXO), connect six on an infinite hexagonal board.
I learned the game from scratch on one RTX 3070 Ti. Read further on how to play me, run other bots, handle my api, or use my training stack

Code and released Bubble weights are licensed under [MIT](LICENSE).

## How I made Bubble

Bubble is a hex-masked residual network with a policy head and a value head. It trains on one GPU in a loop that runs as separate processes on one run directory:

- Actors play Bubble against itself with Gumbel MCTS, AlphaZero-like using Gumbel with a root search that improves the policy from a handful of simulations per stone.
- A learner trains on those games from a replay window sized the KataGo way, and exports a checkpoint every few thousand steps.
- An evaluator plays every checkpoint against the champion on colour-swapped opening pairs and rates them with a posterior Bradley-Terry model. Promotion is a posterior probability, not an SPRT.
- A tactical solver proves forced wins with absolute labels to train on. It started as the solver of [Strix](https://github.com/SootyOwl/hexo-strix) (MIT), edited and changed in many places.
- A proof pass uses the spare CPU to re-check finished games, and a dashboard shows the run.

## Decisions that made Bubble stronger

The rules make HeXO a tactical game: two stones per turn means a four with open ends is already a win. Most of what worked came from taking that seriously.

- Proven outcomes live on the search graph. A proof propagates up the tree with its distance, so the search plays the shortest win, resists longest when lost, and never spends simulations on a settled node. The search also shares evaluations between positions that only differ in move order.
- Spare CPU hunts blunders. The proof pass scans finished games for forced wins that the search missed, labels those positions exactly, and feeds the worst misses back as restart positions for new games.
- Value targets know how far the end is. A calibrated map turns the search value and the plies remaining into a win probability, blended with the outcome. Rows with an exact label get double weight and no outcome noise.
- The opening book pushes play off policy. A symmetry-reduced DAG of openings with paired colour statistics retires skewed or nested openings, and imported off-policy and known-loss openings put the actors in positions the policy would avoid, so the value head has to learn them.
- Policy and value must agree. Value targets follow the improved search policy, positions the network gets most wrong are resampled, and full-search rows carry the policy target while cheap-search rows only carry value.
- The evaluator is honest. Each opening pair is one pentanomial observation, a superseded trial settles on the games it played, fill games sharpen intervals, and anchor matches against Seal keep the scale grounded.
- Kernels. On advice from Vladdy, who wrote [Mantis Shrimp](https://github.com/Cmiller132/Hexo-Shrimp-Bot), the network runs on fused Triton kernels with CUDA graphs: actors got 2.8 times faster and the learner 1.5 times.
- One GPU, shared. Actors and learner alternate in phases paced by rows produced, and the evaluator yields whenever either of them is busy.

## Play Bubble

```sh
git clone https://github.com/Tomodovodoo/HeXO.git
cd HeXO
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` in PowerShell, `.venv\Scripts\activate.bat` in cmd.exe, `source .venv/bin/activate` on Linux and macOS), then:

```sh
python -m pip install -e . -r requirements/learning.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 2
python python/bubble.py play
```

The last command downloads the newest [released Bubble](https://github.com/Tomodovodoo/HeXO/releases) (4.4 MB) into `runs/play` the first time and opens the game at <http://127.0.0.1:8765>. Click a cell to place a stone, drag to pan, scroll to zoom. Either side can be you, Bubble, the native engine or Seal, so engines can also play each other while you watch. The page shows Bubble's win estimate and suggested moves, saves every evaluation, labels each turn from best to blunder, and imports and exports HTTTX, Rectilinear notation and links of hexo.did.science, hexo.mineking.dev and hexo.tyto.cc. Six, Strix, Shrimp and Seal show in the engine picker with a download button that installs them in one click. [docs/play.md](docs/play.md) has the details.

To play a particular checkpoint, point it at the file or at a run you trained:

```sh
python python/bubble.py play --model path/to/ema.pt
python python/bubble.py play --run runs/dense-v1
```

The engine picker also lists every run under `runs/` and every model under `models/`, champion first. A GPU is used when PyTorch sees one; add `--device cpu` otherwise. The solver needs the Rust build from the next section; without it Bubble plays on search alone. Without any weights, `python python/play.py` serves the handwritten native engine.

Other engines go in `models/` (or `--models`), then the rescan button in the picker: a Six folder with `sixengine.exe` and its `gen-*.onnx` networks, or a JSON entry such as `{"name": "Strix", "kind": "strix", "model": "strix.safetensors"}` beside its model file. [docs/play.md#engines](docs/play.md#engines) lists the files per engine and the entry format for Strix, Shrimp and other Six-protocol engines.

## Build Bubble

For games with a clock, bot connections and live remaining times, see
[timed matches and the API](docs/time-controls.md). Bubble supports Absolute and Fischer controls, shares one
deadline across solving and both stones, and exposes HTTTX HTTP and WebSocket routes.

You need Python 3.10 or newer, CMake 3.20 or newer with a C++20 compiler, and PyTorch 2.11 or newer. Install the CUDA build of PyTorch for your card from the [PyTorch selector](https://pytorch.org/get-started/locally/); `requirements/tested.txt` pins the versions this run used. With MinGW on Windows, add `-G "MinGW Makefiles"` to the configure command and keep `g++` on `PATH`.

The tactical solver needs Rust and Cargo with edition 2024 support:

```sh
python tools/build_tactical.py
```

Without it the search plays without proofs and the launcher skips the proof pass until the solver is built. Check the build with the tests:

```sh
python -m unittest tests.test_engine tests.test_neural_search tests.test_proof tests.test_notation_api -v
python -m unittest tests.test_dense tests.test_dense_solve tests.test_openings tests.test_bubble -v
```

## Train Bubble

```sh
python python/bubble.py train --run runs/bubble
```

That creates the run and its first checkpoint, then starts the learner, four actors, the evaluator, the proof pass (once the solver is built) and the dashboard at <http://127.0.0.1:8766>. Everything the run produces stays under `runs/bubble`: game shards, checkpoints, ratings, logs.

```sh
python python/bubble.py status --run runs/bubble
python python/bubble.py stop --run runs/bubble
```

Stopping and starting again resumes from the newest checkpoint. Settings live in `config.json` of the run; pass `--help` to any of the `python/dense_*.py` scripts for the flags that override them. Two options worth knowing: `--net-kernels fused` uses the Triton kernels ([docs/gpu-kernels.md](docs/gpu-kernels.md)), and `--seal` rates champions against Seal once its adapter is built ([docs/native-engine.md](docs/native-engine.md)). A CPU-only run works with `--device cpu`, slowly.

## Notation

`python/notation.py` reads and writes [HTTTX notation v1](https://github.com/hex-tic-tac-toe/hexagonal-tic-tac-toe-notation/tree/15bb7877ae020d661497e332adf0810d00d24e3e), and `python/bot_api.py` serves the [HTTTX stateless bot API](https://github.com/hex-tic-tac-toe/htttx-bot-api/tree/37d2385f1016abe8b25798238a7d0c4a17a25dda/definitions) on localhost:

```sh
python python/notation.py import match.txt > match.json
python python/notation.py export match.json > match.txt
python python/bot_api.py --port 8790 --ms 100
```

A final turn may hold one stone, whether it won or the turn is still open, and the origin-only board is `version[1];`. The API's reply must carry two pieces, so it answers 409 to a first-stone win. See [docs/notation-api.md](docs/notation-api.md).

## Documentation

- [Training: value targets, cheap rows, book starts](docs/dense-training.md)
- [Evaluation: promotion, opening books, variants](docs/dense-evaluation.md)
- [Search: Gumbel MCTS, tree reuse](docs/neural-search.md)
- [Outcomes on the search graph](docs/search-outcomes.md)
- [Tactical solver and its scheduling](docs/tactical-solver.md)
- [Six: Bubble as a Six engine, Six as an opponent](docs/six-engine.md)
- [GPU kernels](docs/gpu-kernels.md)
- [Native engine, Seal adapter, tests](docs/native-engine.md)
- [Browser engine: WebGPU, WebAssembly search and solver](docs/web-engine.md)
- [Notation and bot API](docs/notation-api.md)
