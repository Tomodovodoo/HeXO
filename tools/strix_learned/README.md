# Public learned Strix opponent

This adapter uses `InferModel::eval_states` and `gumbel_mcts` directly from
[SootyOwl/hexo-strix at 5a771e57](https://github.com/SootyOwl/hexo-strix/tree/5a771e572553a8bd8e010112b2ce65f16e5afa1b).
The direct evaluator constructs the relational graph, including edge types,
distances, and global relations required by this checkpoint. The HX04 server
path is not used because its batch reconstruction omits those relations.

The source is MIT-licensed. The separately downloaded checkpoint's license is
unknown. We do not redistribute its weights or claim that public availability
grants redistribution permission. Include upstream and dependency license
notices if distributing compiled code.

## Pinned checkpoint

Download [the public model](https://hexo.tyto.cc/model.safetensors) to an external
file. The adapter requires exactly these bytes:

```text
SHA256 aec92391c66050e737d9b769757248b520ffc1bf44fa039db7c8abd3ef720185
Size   2810120 bytes
Source checkpoint_000010.pt
Steps  10
```

Its metadata identifies a four-layer, hidden-128 relational graph network with
JK concatenation, axis window8, compact stone encoding and no node coordinates.
This is the public step-10 artifact, not a private Pulsatrix-246 checkpoint or
a claim about the strongest Strix bot. If the URL changes, the hash check fails;
do not silently substitute another model.

## Build

Install Rust with edition 2024 support and a native linker. From the HeXO root:

```sh
python tools/build_strix_learned.py EXTERNAL_STRIX_BUILD_CHECKOUT
```

The setup clones the pinned dependency when that external directory is absent.
It refuses a different revision, unexpected tracked changes, and untracked
files in the compiled dependency tree, including automatically discovered
`build.rs` files. The only source
patch changes the unused `inference_subprocess` module gate from `not(wasm32)`
to `target_os = "linux"`. Upstream otherwise tries to compile Linux pipe and `fcntl`
calls on Windows and macOS. Search, graph construction, model arithmetic and rules are
unchanged. `build-local.toml` contains local dependency paths and is ignored;
Cargo.lock pins the other dependencies. The generated `build-provenance.json`
records the source revision, original/patched module hashes and executable
hash. The adapter checks that executable hash when the report is present.

The local build used Rust 1.98.1 and MinGW on Windows. Inference is native CPU;
the adapter does not use PyTorch, a GPU, or a remote service.

## Run and verify

```sh
python python/legacy/arena.py --opponent strix --strix-model EXTERNAL_MODEL.safetensors --strix-sims 2 --strix-actions 2 --strix-timeout-ms 5000 --games 2 --ms 100 --max-stones 80 --seed 20261002 --output artifacts/strix-learned.json
```

`--strix-sims` controls simulations per placement; the second placement is
searched from the updated board. Gumbel noise is disabled, `c_visit=50`, and
`c_scale=1`. Upstream's root forcing shortcut remains enabled for both turn
phases, with wide generation, depth 6 and a 2,000-node cap. Leaf forcing is
disabled. These settings are recorded and do not constitute independently
verified proof results. A safety
deadline bounds the whole turn, including any process restart. This does not
make the simulation budget equal to the native engine's wall-clock budget.
The report explicitly sets `equal_wall_budget=false` and includes both sides'
actual timings, neural evaluation counts, root visits, source/model/executable
identity, loaded metadata and setup latency.

One persistent worker loads immutable verified model bytes from a private
temporary file. Its executable also uses the verified private-image mechanism.
Private files are removed after the worker exits. A timeout or invalid response
closes the worker and raises an arena error, producing an incomplete/invalid
game rather than inventing moves or scoring a loss. The next call may restart.
Calls serialize; queued time counts toward the deadline. No inference runs
during the opponent's turn.

Every returned stone is replayed through HeXO's native legality checks and then
undone. This handles newly reachable second cells, one-placement turn phases,
and immediate wins on the first stone. Coordinates are limited to ±1000000 and
positions to 800 stones by this adapter. There is no cropped board or artificial
draw inside the external search. The arena's cap remains a truncation.

Set `HEXO_STRIX_PUBLIC_MODEL` to the external pinned file, then run:

```sh
python -m unittest tests.test_strix_learned -v
```

Six tests cover loaded relational metadata, actual neural evaluation,
persistent complete/conditional turns, exact sequential legality and restoration,
first-stone termination, wrong-model rejection, timeout cleanup and recovery.
Tests requiring the optional model/binary skip when those are absent.

An initial two-game color-swapped operational check used the promoted pattern
checkpoint 1 versus this public model at two simulations per placement.
HeXO lost both games. Mean measured time was 82.5 ms for HeXO and 183.8 ms for Strix;
Strix's maximum was 324.3 ms. The games ended at 67 and 45 stones with no invalid
moves. This establishes working integration, not a controlled strength ranking
or a result against any other Strix checkpoint.
