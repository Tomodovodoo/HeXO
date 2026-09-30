# Native engine

Build instructions are in [the README](../README.md#build).

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
python python/legacy/arena.py --opponent shallow --games 20 --ms 100
python python/legacy/arena.py --opponent random --games 20 --ms 100
```

The arena alternates colors and reuses each opening for a pair of games. Results contain moves, actual decision times, engine hashes, and conservative opening-pair confidence bounds. Truncations, invalid games and unplayed partners of a partial pair contribute unknown outcomes to those bounds. A completed-games-only Wilson interval is retained separately and must not be used as an overall strength estimate.

To compare against Seal, clone its source outside this repository, then configure the optional adapter:

```sh
git clone https://github.com/Ramora0/HexTicTacToe.git ../seal-reference
cmake -S . -B build -DHEXO_SEAL_SOURCE=../seal-reference
cmake --build build --config Release -j 4
python python/legacy/arena.py --opponent seal --games 20 --ms 100 --output artifacts/seal.json
python python/legacy/arena.py --opponent seal --run runs/gpu-selfplay --checkpoint 1 --games 20 --ms 100
```

The adapter compiles the external engine without vendoring it. Seal's fixed array has a smaller coordinate range; games outside the adapter's safe range are marked invalid rather than counted as victories. Equal requested budgets are used, and both engines' actual elapsed times are retained. `--run` loads the promoted checkpoint, `--checkpoint` selects another saved candidate, and `--table` or `--nnue` loads a standalone export. Reports identify the loaded model and its hash. Without a model option the arena uses the original evaluator.

The separate `seal-current-best` opponent uses [Ramora0/SealBot at c94749c](https://github.com/Ramora0/SealBot/tree/c94749c21c16c3b072fff6da49762dd5f92f3986), with its `best` pattern table. This is a newer search implementation than `seal`, which uses HexTicTacToe. The best table itself is unchanged from oldSeal. No project license was found at the pinned SealBot revision; upstream code stays in an external checkout.

```sh
git clone --depth 1 --filter=blob:none --sparse https://github.com/Ramora0/SealBot.git ../seal-current-reference
git -C ../seal-current-reference sparse-checkout set best
git -C ../seal-current-reference fetch origin c94749c21c16c3b072fff6da49762dd5f92f3986
git -C ../seal-current-reference checkout --detach c94749c21c16c3b072fff6da49762dd5f92f3986
python tools/seal_current.py ../seal-current-reference
python python/legacy/arena.py --opponent seal-current-best --nnue runs/example/checkpoints/0001/model.nnue --games 40 --ms 100 --max-stones 800 --output artifacts/seal-current-best.json
python -m unittest tests.test_seal_current -v
```

The optional adapter build requires a GCC-compatible C++20 compiler. Its manifest records every compiled upstream header hash, canonical and on-disk weight hashes, adapter and binary hashes, compiler, and build command. Arena verifies the binary against that manifest, resets upstream search state between games, and records both sides' ordered turns and elapsed times. The 100 ms setting is an upstream best-effort full-turn deadline, not a hard timeout. Upstream initializes randomness independently of the arena opening seed. Coordinates outside ±55, illegal moves, and incomplete turns are rejected. Upstream returns fixed pairs; when its first placement wins under native rules, only that winning prefix is played. This upstream engine does not correctly support non-opening partial-turn roots, so the adapter rejects those explicitly; arena calls start at complete-turn boundaries.

Other public references were checked on 2026-09-24. [Mantis at 9c94b95](https://github.com/Cmiller132/Hexo-Shrimp-Bot/tree/9c94b95ce5e3ccf4f892eeadca20524c522d0629) provides maintained Rust/Python inference and match entry points, but no public trained checkpoint or project license was found. [Strix at 5a771e5](https://github.com/SootyOwl/hexo-strix/tree/5a771e572553a8bd8e010112b2ce65f16e5afa1b) is MIT-licensed and publishes a [2,810,120-byte safetensors model](https://hexo.tyto.cc/model.safetensors), SHA256 `aec92391c66050e737d9b769757248b520ffc1bf44fa039db7c8abd3ef720185`. Its metadata says `checkpoint_000010.pt`, step 10, not the private `pulsatrix-246` checkpoint. Its relational graph requires direct `InferModel::eval_states` with Gumbel MCTS; the pinned HX04 server drops the required relational edge fields. [HextocZero at dc1be7b](https://codeberg.org/Kubuxu/HextocZero/src/commit/dc1be7b175dd9f27b6db8481e00153c9a0f2e3ae) is MIT-licensed heuristic MCTS with neural inference unimplemented; its documented legality omits the radius-eight restriction. None of these source discoveries establishes comparative strength.

The first current-Seal comparison used 40 games, 20 color-swapped opening pairs, seed `20261003`, width 16, and 100 ms requested per turn. The internally promoted `nnue-reanalysis-native-v1` checkpoint 1 scored **5 wins and 35 losses**, with no invalid or truncated games. Actual turn means were 87.72 ms for HeXO and 79.63 ms for SealBot, with maxima 117.79 and 103.76 ms. The conservative opening-pair 95% win-rate interval was [0, 0.429]. All 1,938 post-opening placements were independently replayed through the Python rules reference. This run used adapter commit `a4cc08a`, NNUE SHA256 `6c0599e380b3b87a764f7b95c76b21e43863f81fdc1a273bb4865434aed353ae`, and trace SHA256 `4b23e04428505d54411b5b1464919e7f5bb374ad8fb2c9f02e578926375575d5`. A subsequent adapter fix accepts upstream fixed pairs whose first placement wins; independent replay confirmed none of the 492 Seal turns in this trace had that case, so this recorded result is unaffected. It does not establish superiority over current SealBot.

To compare against the published Orca model, use an external checkout:

```sh
git clone https://github.com/Saiki77/hexbot-building-framework.git ../orca-reference
python python/legacy/arena.py --opponent orca --orca-source ../orca-reference --orca-sims 200 --games 20 --ms 100 --max-stones 800 --output artifacts/orca.json
```

This requires PyTorch. The adapter strictly loads the checkout's seven-channel `orca/checkpoint.pt` without adding random weights. Use `--orca-checkpoint` to select another compatible checkpoint and `--orca-device cuda` for GPU inference. The report records source revision, checkpoint hash, simulation budget and actual turn times. Orca receives simulations per placement; our engine receives milliseconds per complete turn. This comparison does not use equal time budgets. Native rules validate every returned move, and replay disagreements remain invalid games rather than wins.

The optional [learned Strix adapter](../tools/strix_learned/README.md) loads the
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
python python/proof.py --history position.json --ms 100 --output proof-result.json
python python/proof.py --history position.json --verify proof-result.json
python python/proof.py --benchmark
```

A history is a JSON list of `[q, r]` placements in play order. Verification needs a returned certificate; an unknown result has none. The solver is separate from deployed PVS and has no demonstrated Elo benefit. Its deadline is cooperative: a synchronous native candidate call can overrun it, and late results become unknown. In one benchmark a 13-stone forcing win verified in 35 ms, while a 1001-stone sparse board took 236 ms under a requested 100 ms budget.

## Experimental root turn coverage

Quiet root widening is opt-in. It retains every existing selected turn and adds complete pairs by conditional rank, with separate second-placement and final-turn budgets. Immediate wins and mandatory defenses keep their exact handling. Deeper search keeps its existing candidate restrictions.

```sh
python python/legacy/arena.py --opponent seal --games 40 --ms 100 --width 16 --root-seconds 16 --root-turns 48 --max-stones 800 --output artifacts/seal-widened.json
python -m tests.benchmark --trace artifacts/seal-trained-fresh-40.json --first-game 10 --positions 12 --ms 100 --width 16 --reference-ms 1000 --root-seconds 16 --root-turns 48 --output artifacts/pair-admission.json
```

The trace benchmark reports complete ordered-turn lists, resulting-position recall, depth, nodes and actual time. An optional `--seal-library` uses a separately built Seal adapter as reference; `--reference-report` reuses frozen reference turns for another ablation. On 12 development positions, the 48-turn setting raised Seal-reference result recall from 4/12 to 7/12, while mean completed depth fell from 2.50 to 2.42 at 100 ms. These traces include a repeated position family, so this is a development diagnostic rather than independent validation. Both default and widened settings later scored 4 wins and 36 losses against Seal on the same 40 fresh games at 100 ms. No playing-strength gain is established. Search clocks are best-effort; generation and legal fallback can exceed very short budgets.

## References

Rules and turn semantics were checked against the [official HeXO source](https://github.com/HeXO-Game/HeXO/blob/1aea2b676733f8cdc53f8f92cb00b367af72111c/packages/shared/src/sharedTypes.ts).

Independent opponents: [Seal](https://github.com/Ramora0/HexTicTacToe) and [Orca framework](https://github.com/Saiki77/hexbot-building-framework). The site bundles a Seal WebAssembly build; the optional native adapter currently compares against the selected external source revision, which may differ from that build.

The intended learned evaluator follows ideas from [Rapfi](https://github.com/dhbloo/rapfi) and [NNUE](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html), adapted to Hexo's three axes and turn semantics. Their existing game-specific weights are not used.
