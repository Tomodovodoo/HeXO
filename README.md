# HeXO

C++20 Hexo rules and search engine, Python interface, and local browser game.

The current priority is a self-play learning loop with a local experiment dashboard. Checkpoints earn promotion through matches against frozen opponents; fitting loss alone never replaces the incumbent.

An optional Strix root-probe experiment is available in `arena.py` after building
the [separate reference executable](tools/strix/README.md). `--strix-root-ms 20`
allocates up to 20 ms of each turn to that reference; its default is zero.
`--strix-root-nodes 1000`, `--strix-root-depth 8`, and `--strix-root-wide` expose
the solver limits and generator. The probe must have less time than `--ms`.
Only a sequentially legal winning PV can supply the current turn, recorded as
`source=strix_reference`, `score=null`, and `independent_proof=false`. All other
results fall back to native PVS with the remaining wall budget. No training
targets or native tactical rules change.

Each arena worker warms one persistent reference process during setup and
records that latency separately. Hard timeouts kill the process; later restart
costs count against the next turn. No work runs during the opponent's turn.
Reports include combined timings, overruns, call/result counts, executable and
adapter hashes. Native fallback receives at least 1 ms even after a scheduler
or cleanup overrun, which remains visible in the report.

A bounded operational check used the same trained pattern checkpoint, Seal at
100 ms, seeds 20260929/20260930, alternating which configuration ran first,
two color-swapped games per seed, and an 800-stone cap. Both probe-off and
20-ms/1,000-node probe-on lost all four games. The probe made 47 calls with
43 scoped negatives, four unknowns, and zero reference selections. Both
configurations stayed below 105 ms per measured turn; initial probe setup took
118–120 ms. This small check establishes neither a strength advantage nor a
strength equivalence. The experiment remains disabled by default.

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

The GPU `pattern` actor uses exact sparse rules with growing coordinate storage and incremental native-compatible pattern features. They first take exact one- or two-stone wins, then choose a placement belonging to a complete cover of all immediate opponent threats when such a cover exists. These choices override exploration. Quiet candidates combine nearby cells, axial development, and broader exploration, scored one placement at a time by the learned evaluator. They do not run native PVS; `--ms` and `--width` do not control this actor. Replay records identify the actor and one-placement target semantics. Promotion always measures the native deployed engine at equal wall-clock budgets.

On this RTX 3070 Ti, the tactical zero-residual GPU actor generated about 184 games/second and 28,300 placements/second at batch 512 with 32 quiet candidates and a 256-stone cap. Of those 512 games, 388 finished and 124 reached the cap; peak PyTorch allocation was about 920 MiB. These timings include generation and replay transfer, but not training or native evaluation. The earlier actor was faster but frequently missed immediate wins. Its shorter games make raw games/second an unfair performance comparison. Reproduce on your hardware:

```sh
python gpu_benchmark.py --actor --batches 64 256 512 2048 --placements 256
python gpu_benchmark.py --batches 64 512 2048 --placements 96
```

The second command compares only exact rule transitions and feature updates with C++. Small GPU batches are slower than native code. The dashboard exposes device utilization, VRAM, power, and temperature; telemetry includes other applications using the GPU. A small pattern learner cannot productively saturate every GPU unit at all times, so larger game batches and measured throughput guide settings.

The default `pattern` evaluator uses an `18 -> 32 -> 1` network to map six-cell ternary patterns to a bounded residual on top of the handwritten evaluator. It exports a 729-entry integer table. The `nnue` evaluator uses centered eleven-cell lines, nonlinear combinations of all three axes, an incremental 64-channel position summary, and separate value and conditional move-ranking heads. Both run entirely in C++ during search.

To train the full NNUE model with native search actors and a CUDA learner:

```sh
python train.py --model nnue --run runs/nnue --selfplay-backend native --device cuda --batch 0
```

NNUE replay stores sparse center patterns and search-selected first and second placements. `--nnue-replay-centers` bounds retained center features, and `--nnue-batch-centers` splits batches by feature count. NNUE's native format and feature definitions are documented below.

The GPU NNUE actor loads the same exported model and maintains the same sparse
center features. It ranks candidate first placements, evaluates conditional
second placements, and chooses a complete turn using the value head. Exact
immediate wins and defensive obligations override the quiet beam search.
`--gpu-beam 4` uses four first candidates and four conditional second candidates;
this remains selective search, not exhaustive play or native PVS. Exploration
does not create valid policy-teacher labels. Native matches still decide promotion.

```sh
python train.py --model nnue --selfplay-backend gpu --run runs/nnue-gpu --games 512 --gpu-beam 4 --device cuda
```

For NNUE, `--batch 0` means at most 256 positions, further limited by sparse
center count. Each epoch visits the selected training positions once unless
`--updates-per-epoch` explicitly requests another update budget. Metrics record
optimizer steps, examples processed, retained positions and retained centers.

New runs use `--curriculum mixed-v1`: legal prefixes of 3, 5, 7, 9 or 11 stones,
including compact fights, broad placements, separated groups and chained distant
placements. Families preserve stone owners and turn phase under all 12 board
symmetries. Evaluation reserves a separate family bucket that neither training
nor loss validation can use. `--curriculum legacy` retains the original narrow
three-stone distribution for explicit comparisons.

Native actors explore 5% of quiet turns by default. `--native-exploration` controls
this probability. Wins and mandatory defenses override exploration, including a
fresh check before the second placement. Exploratory actions do not become
teacher labels. Replay retains each sampled position's absolute ply and every
game's complete history, so omitted exploratory turns do not lose geometry.

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

The adapter compiles the external engine without vendoring it. Seal's fixed array has a smaller coordinate range; games outside the adapter's safe range are marked invalid rather than counted as victories. Equal requested budgets are used, and both engines' actual elapsed times are retained. `--run` loads the promoted checkpoint, `--checkpoint` selects another saved candidate, and `--table` or `--nnue` loads a standalone export. Reports identify the loaded model and its hash. Without a model option the arena uses the original evaluator.

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

## Bounded forcing certificates

`proof.py` searches continuous double-threat attacks and returns `PROVEN_WIN`, `PROVEN_LOSS`, or `UNKNOWN`. Every winning certificate covers all relevant defensive branches. Immediate counterwins take priority; a defense with a free second stone is unsupported and returns unknown. The independent verifier reconstructs rules and covers from raw coordinates. Ordinary search scores are never treated as certificates.

```sh
python proof.py --history position.json --ms 100 --output proof-result.json
python proof.py --history position.json --verify proof-result.json
python proof.py --benchmark
```

A history is a JSON list of `[q, r]` placements in play order. Verification needs a returned certificate; an unknown result has none. The solver is separate from deployed PVS and has no demonstrated Elo benefit. Its deadline is cooperative: a synchronous native candidate call can overrun it, and late results become unknown. In one benchmark a 13-stone forcing win verified in 35 ms, while a 1001-stone sparse board took 236 ms under a requested 100 ms budget.

## Experimental root turn coverage

Quiet root widening is opt-in. It retains every existing selected turn and adds complete pairs by conditional rank, with separate second-placement and final-turn budgets. Immediate wins and mandatory defenses keep their exact handling. Deeper search keeps its existing candidate restrictions.

```sh
python arena.py --opponent seal --games 40 --ms 100 --width 16 --root-seconds 16 --root-turns 48 --max-stones 800 --output artifacts/seal-widened.json
python -m tests.benchmark --trace artifacts/seal-trained-fresh-40.json --first-game 10 --positions 12 --ms 100 --width 16 --reference-ms 1000 --root-seconds 16 --root-turns 48 --output artifacts/pair-admission.json
```

The trace benchmark reports complete ordered-turn lists, resulting-position recall, depth, nodes and actual time. An optional `--seal-library` uses a separately built Seal adapter as reference; `--reference-report` reuses frozen reference turns for another ablation. On 12 development positions, the 48-turn setting raised Seal-reference result recall from 4/12 to 7/12, while mean completed depth fell from 2.50 to 2.42 at 100 ms. These traces include a repeated position family, so this is a development diagnostic rather than independent validation. Both default and widened settings later scored 4 wins and 36 losses against Seal on the same 40 fresh games at 100 ms. No playing-strength gain is established. Search clocks are best-effort; generation and legal fallback can exceed very short budgets.

## Status and remaining work

The local game, native engine, self-play trainer, checkpoint evaluation and dashboard are playable. The committed suite currently has 22 tests covering reference rules, CPU/CUDA parity, NNUE inference and undo, curriculum partitioning and training-target semantics. Native search now orders a stored transposition move first when it is already in the selected legal turn list. The table is still local to each search.

The old GPU actor produced 6,144 games but missed immediate wins. Its best early candidate scored 28/40, then only 78/160 on fresh confirmation games. The tactical replacement produced 6,144 games and 1,060,519 positions, with 526 capped games whose outcomes remain unlabeled. Its first candidate passed the paired promotion test at 104 wins and 56 losses against the frozen reference. Later candidates were rejected against that incumbent, including one that scored 110/160 against the reference but only 66/160 against the incumbent. Self-Elo is opponent-dependent; more training has not consistently improved the deployed checkpoint.

Fresh independent matches used 40 games and seed 20260924. The promoted pattern engine scored **4 wins and 36 losses against Seal** at 100 ms per turn for both engines. The handwritten engine scored **39 wins and 1 loss against the published Orca checkpoint** using 100 ms per turn versus Orca's 200 simulations per placement on CUDA. Orca averaged about 434 ms per turn versus HeXO's 89 ms, so this is not an equal-time comparison. There is no claim of superiority over all known bots.

The first native NNUE experiment collected 15,558 positions from 128 games and scored 39 wins and 41 losses against its zero-head NNUE reference. It was rejected. That reference pays NNUE inference costs; this experiment does not establish an improvement over the faster bare handwritten engine. The GPU NNUE pipeline has completed training, native evaluation and resumed training, with separate policy/value losses and actual optimizer exposure counts on the dashboard.

Remaining work includes complete-turn candidate recall and controlled widening, stronger-search reanalysis, bounded tactical search, broader independent equal-time matches, and architecture/compute comparisons. The current NNUE has width 32 and nonlinear fusion of crossing lines, but no neighboring-cell mixing block. Match clocks, rated lobbies, and online account play are not part of the local board yet.

## References

Rules and turn semantics were checked against the [official HeXO source](https://github.com/HeXO-Game/HeXO/blob/1aea2b676733f8cdc53f8f92cb00b367af72111c/packages/shared/src/sharedTypes.ts).

Independent opponents: [Seal](https://github.com/Ramora0/HexTicTacToe) and [Orca framework](https://github.com/Saiki77/hexbot-building-framework). The site bundles a Seal WebAssembly build; the optional native adapter currently compares against the selected external source revision, which may differ from that build.

The intended learned evaluator follows ideas from [Rapfi](https://github.com/dhbloo/rapfi) and [NNUE](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html), adapted to Hexo's three axes and turn semantics. Their existing game-specific weights are not used.
## Centered-line NNUE contract

The `nnue` model uses three centered eleven-cell ternary patterns at every center
whose lines contain a stone. A placement changes 33 directional patterns at 31
centers. The exported shared table has 177147 rows and 32 signed int16 channels,
scaled by 256. Channels 0–15 are odd under color swap; channels 16–31 are even.
All channels are invariant under line reversal. The learned mapping is
33 → 64 → 32 with ReLU and tanh. Training uses straight-through quantization.

For each center, sum its three table rows into `u`. Its 64-channel contribution
is `[ReLU(u), ReLU(-u)] - [ReLU(3*table[0]), ReLU(-3*table[0])]`. This subtraction
makes empty space contribute zero. C++ maintains exact int64 pooled sums. Divide
the pool by `256*sqrt(max(1, number_of_active_centers))`. For player 1, swap the
positive and negative halves of the first 16 channels. The other channels stay
in place. There is no board crop.

The value head is 68 → 32 → 1 with ReLU. Its inputs are the pooled 64 channels
and four context values: `remaining==1`, `remaining==2`, `log1p(stones)/8`, and
`(own_stones-other_stones)/max(1,stones)`. Its output times 6000 is a residual
added to the handwritten evaluation from the current player's perspective.
The resulting ordinary search score is clamped to ±500000; terminal scores
remain separate.

The conditional policy head is 104 → 16 → 1 with ReLU. Its inputs are pooled
64 channels, the 32-channel sum of the candidate's three lines **after** its
placement divided by 256, the four context values, and four pair values.
The candidate channels use the current player's perspective. Pair values are
`has_first`, `shares_axis`, `min(hex_distance,8)/8`, and
`shares_axis*max(0,6-hex_distance)/5`. All four are zero without a first stone
from the current turn. Second-stone examples are collected after the first
stone is applied. Policy scores order candidates; tactical inclusions remain
mandatory.

Native files begin with a packed little-endian 60-byte header: magic
`HXNNUE1\0`; nine uint32 values `1, 0x01020304, 177147, 32, 64, 4, 4, 32, 16`;
float32 scales `256, 6000`; and uint64 payload length. The payload contains
the row-major int16 table, then float32 value `W1,b1,W2,b2`, then policy
`W1,b1,W2,b2`. Shapes follow the heads above. Checkpoint metadata records the
whole-file SHA256. Native loading validates dimensions, sizes, finite weights,
table bounds and table symmetries. A model handle is immutable and shared by
attached boards. Loading a legacy table detaches NNUE and vice versa.

After an engine change, start a new rating run while retaining learned NNUE weights:

```sh
python train.py --run runs/nnue-new-engine --model nnue --initial-model runs/nnue-old/checkpoints/0001/model.pt --initial-optimizer runs/nnue-old/checkpoints/0001/optimizer.pt --device cuda
```

`--initial-optimizer` is optional. When supplied, AdamW moments and step counters
are retained, and `--lr` sets the new run's learning rate. The imported model is
validated, copied into checkpoint zero, and exported with the current native
format. Source paths and hashes are recorded; resume requires the same arguments
and unchanged source files. Ratings start at zero against the imported anchor.
Existing runs and checkpoint files are never reinitialized by these options.

Saved NNUE histories can be revisited with a frozen evaluator and a larger native
search budget, then supplied to an ordinary training run:

```sh
python reanalysis.py --run runs/nnue-selfplay --iteration 1 --checkpoint 0 --max-positions 256 --ms 200 --width 32 --output runs/reanalysis-0001
python train.py --run runs/nnue-with-reanalysis --model nnue --reanalysis runs/reanalysis-0001 --device cuda
```

`--reanalysis` accepts multiple completed shard directories. Their positions join
the replay sampling pool independently of `--replay-iterations`, which still
limits only chronological self-play shards. External files are read in place;
their manifests, search provenance and hashes are recorded in the run and each
trained checkpoint. Modified shards or manifests prevent resume. Repeating the
reanalysis command reuses completed root searches when its provenance matches.
Targets remain selective search estimates. A conditional second-stone row loses
the source game's outcome label when the teacher's first move diverges.

### Experimental KLENT training

`klent.py` runs a separate policy/Q experiment; ordinary `train.py` is unchanged.
It shares the centered-line NNUE representation and adds a scalar candidate Q
head. Neural collection and fitting batch on CUDA, while exact native rules and
feature reconstruction run on the CPU. Every legal cell is included, including
newly reachable cells after the first placement. Memory limits split batches;
they never crop the legal action set.

```sh
python klent.py --run runs/klent --initial-model runs/nnue-selfplay/checkpoints/0001/model.pt --games 16 --envs 8 --max-plies 128 --device cuda
```

One frozen actor collects each fresh corpus. With `pi = softmax(policy_logits)`
and `Q = tanh(q_head)`, the acting policy is
`mu = softmax((Q + beta * log(pi)) / (alpha + beta))`. One shuffled fitting pass
minimizes `CE(mu, pi_new) + (Q_new(taken_action) - G)^2`; stored targets are
detached. This follows the [KLENT paper's policy and scalar-Q objectives](https://arxiv.org/html/2602.10894v2#S4).
The defaults are alpha 0.03, beta 0.1, gamma 1, and placement-level lambda
`exp(-1/16)`.

Returns use the actual mover change: the sign is positive between two placements
by the same player and negative when the opponent acts next. The winning
placement has target +1. Capped games bootstrap the final nonterminal state from
that same frozen actor and remain explicitly unfinished; they are never draws.
Both colors' records are retained. A separate, reported value-head-only pass
distills the returns for native PVS with shared features detached; this auxiliary
pass is skipped when every critic target is zero.

Q starts at zero. With no terminal episodes, zero tail bootstraps can leave the
critic without a learning signal. Check `critic_targets_informative`,
`nonzero_return_fraction`, and `terminal_fraction`; throughput is not evidence of
improvement. `--initial-q` accepts a saved `q.pt` only with its exact matching
`--initial-model` file. Q depends on the learned shared representation, so an
unrelated Q head is rejected. No handwritten value is silently substituted for Q.

An explicit human-corpus Q warm-start can provide a nonzero critic before fresh
self-play. It freezes the matching NNUE features and fits only the Q head on
verified human chosen actions and their terminal outcomes in the acting player's
frame. The existing hashed family train/validation split is retained; test and
excluded shards are never loaded. These targets describe human continuations,
not optimal actions or on-policy KLENT returns. Values for unchosen actions are
model extrapolations that self-play must test.

```sh
python q_warmstart.py --corpus artifacts/datasets/human-warmstart-v1 --model artifacts/models/human-warmstart-12/model.pt --output artifacts/models/human-q-warmstart-12 --positions 40000 --epochs 12 --device cuda
python klent.py --run runs/human-initialized-klent --initial-model artifacts/models/human-q-warmstart-12/model.pt --initial-q artifacts/models/human-q-warmstart-12/q.pt --device cuda
```

Q initialization selects the best validation epoch and publishes into a new
directory atomically. Its copied `model.pt` and native export remain unchanged;
`q.pt` records their representation identity plus corpus and source provenance.
The tool does not resume partial fits or overwrite an existing output.

Checkpoint directories retain standard `model.pt` and `model.nnue` deployment
artifacts, plus `klent.pt` with Q/optimizer state and a representation-bound
`q.pt`. Complete corpus directories preserve every sampled move, legal-set hash,
acting distribution/value, mover/phase and return target. Manifests bind them to
the actor, code, engine, configuration and content hashes. Resume validates these
artifacts and reuses a completed corpus after an interrupted fit; unfinished
collection restarts deterministically. Each iteration consumes only its own
corpus. This path does not promote checkpoints or assign Elo; use paired external
evaluation of its native exports.

The inspected [Mantis implementation](https://github.com/Cmiller132/Hexo-Shrimp-Bot/blob/9c94b95ce5e3ccf4f892eeadca20524c522d0629/python/mantisnet/mantisnet/klent/train.py)
instead trains a categorical critic. Its [acting operator](https://github.com/Cmiller132/Hexo-Shrimp-Bot/blob/9c94b95ce5e3ccf4f892eeadca20524c522d0629/python/mantisnet/mantisnet/klent/improve.py)
also permits a mass-normalized Q score. Those adaptations are not enabled here.

NNUE replay is versioned separately from legacy six-cell histograms. It stores
ragged center-code triples and candidate-code triples with offsets, candidate
coordinates, pair context, turn context, player, handwritten baseline, chosen
candidate, search depth validity, outcome and opening family. Policy labels are
the search-selected first and conditional second placements, not alpha-beta
visit counts. Unfinished outcomes remain missing. Native PVS and the GPU
complete-turn beam identify their teacher semantics in the recorded games;
their search targets should not be treated as interchangeable depths.

Native depth-zero evaluation checks whether the opponent's immediate completion
sets can be covered by the remaining placements, after checking our own immediate
win. An impossible cover is a mate loss; a cover with a spare placement remains
unresolved. This fixes three observed leaf misvaluations. In a fresh paired
100 ms comparison against the pinned public Seal engine (20 opening pairs per
build, seed 20260929), the baseline scored 3-37 and the guard scored 2-38, with no
truncated games. This experiment did not demonstrate a playing-strength gain.

The trace benchmark's ordered and resulting-position recall measure the complete
**untimed** generated lists. Timed searches can expire during generation, so these
figures do not claim that a turn was searched within the budget. The report records
generation time, completed depth, and zero-depth trials separately. Reused reference
reports must match both the trace hash and the exact position history.

Experimental TT turn admission is available through
`Game.search(..., tt_injection=True)` and `arena.py --tt-injection`.
The arena records the flag in its configuration and applies it only to the
contender. It validates and reserves a previous-iteration complete turn before
candidate truncation, preserving immediate wins and mandatory defenses. Hints
are frozen throughout each iteration, including PVS re-searches; TT score-bound
reuse is disabled in this mode. The table remains local to one search call,
with no persistent entries or cross-model score reuse. The default remains off.
A 12-position development benchmark showed identical depth-three results with
4.6% more elapsed time; this experiment has no demonstrated playing-strength gain.

## Official notation and local bot API

`notation.py` imports and exports [notation v1](https://github.com/hex-tic-tac-toe/hexagonal-tic-tac-toe-notation/tree/15bb7877ae020d661497e332adf0810d00d24e3e).
Cross is native player 0; its origin placement is implicit in the text and present
in imported histories. Turn numbers, sequential radius-8 legality, and terminal
states are checked. Metadata and `!` annotations are preserved without inferring
their meaning. The upstream example uses `datetime` rather than `utcdatetime`,
and a named time control that differs from its numeric grammar; these values are
kept intact. Python callers can use `loads(text)` and `dumps(record_or_history)`.

```sh
python notation.py import match.txt > match.json
python notation.py export match.json > match-roundtrip.txt
python bot_api.py --port 8790 --ms 100
```

The loopback HTTP adapter exposes `GET /capabilities.json` and
`POST /stateless/v1-alpha/turn` according to the
[published API definitions](https://github.com/hex-tic-tac-toe/htttx-bot-api/tree/37d2385f1016abe8b25798238a7d0c4a17a25dda/definitions).
For example, send `{"board":{"to_move":"o","cells":[{"q":0,"r":0,"p":"x"}]},"request_id":1}`
with content type `application/json`. Responses contain `move.pieces` objects
with `q` and `r`, and echo an optional request ID. Add `--model path/to/model.bin`
to use a native NNUE export; capabilities identify its SHA256. Otherwise the
adapter uses the handwritten evaluator. This command does not register a bot
with any external service.

Both published formats require exactly two placements per recorded turn or API
move, while the API also forbids placements after a win. A first-placement win
cannot satisfy both requirements. Notation export raises `NotationConflict` for
that case, partial turns, and empty boards. The API returns HTTP 409 for origin
turns, partial turns, already-terminal boards, or a chosen first-placement win;
it never pads a winning move. Full two-placement wins are supported.

The stateless board is unordered. The adapter reconstructs a legal ordering of
exactly the supplied cells and checks counts and `to_move`; it never assumes array
order is history. Unreachable positions return 400, and a reconstruction exceeding
one second or 20,000 visited states returns 503. Local limits are 1 MiB request
bodies and 1,025 board cells; notation accepts 4,097 placements and 1 MiB text.
`time_limit` is used as an advisory search cap, with zero returning 408. Native
search and reconstruction cannot guarantee a hard response deadline, so
`move_time_limit` is false. Websocket and matchmaking capabilities are not declared.
Run `python -m unittest tests.test_notation_api -v` for the protocol checks.

The local adapter bounds the request line and headers together to two seconds,
including clients that keep sending bytes. Request-body transfer has a separate
two-second deadline. These transport limits are separate from its advisory search
budget. A configured model that disappears or becomes unreadable returns JSON 503.

KLENT's separate deployment-value pass reconstructs and validates the same full
legal action ordering as the actor pass, retaining its row order and chunk
boundaries, but skips candidate feature encoding and embedding. It trains only
the value head against the existing return targets. A CPU comparison on 128
saved corpus rows measured 0.857 to 0.423 seconds for this pass (2.03x), with
identical losses, value-head parameters and Adam states in that experiment.
This is a value-pass measurement, not an overall training or GPU speedup.
Reproduce it with `python -m tests.benchmark_value --corpus <directory>
--model <model.pt> --output <report.json>`.

## Primary relational policy/Q model

`relational_model.RelationalNet(ModelConfig())` implements the new primary
representation: width 256, eight residual blocks, eight attention heads,
feed-forward width 1,024, and 16 learned global tokens. It has **31,860,930
trainable parameters**. Each block performs local relational attention, stone
attention, global read/write attention, then local attention again. Legal-cell
representations persist through every block. Policy and bounded scalar-Q heads
return every legal action in native `Game.legal_moves()` order.

The encoder shares stone/cell identity across every nonempty six-cell window.
Window occupancy and incidence slots are tied under reversal; no absolute axis
embedding is used. Radius-eight stone/cell links and local legal-cell neighbors
retain geometry outside window coverage. Both the independent Python reference
and the native accelerator preserve all 12 board symmetries. The native encoder
is the explicit default of `NeuralEvaluator`; `backend="reference"` selects the
reference implementation, and a missing native library does not silently fall
back. Normal CMake builds include `hexo_graph`.

Training calls `encode(history)`, `pack(graphs, device)`, and `model(batch)`.
Outputs are flat FP32 `logits`, bounded FP32 `q`, and `action_offsets`.
`NeuralEvaluator(model, device).evaluate(histories)` returns ordered action,
logit, and Q arrays per position. Histories contain all placements, including
the origin. Native rules determine side and remaining placements; terminal
positions belong to the search implementation.

Default work budgets are 12,000 nodes and 600,000 relation traversals per batch
(local edges count twice). Whole graphs are batched in order; a single graph
exceeding the configured budget raises `WorkBudgetError`, never truncates its
actions. These are explicit resource limits, not game rules. Increase them only
after checking memory. Edge chunks control temporary allocations; autograd still
retains work proportional to edges times width. Block checkpointing, BF16
projections, and FP32 normalization, softmax, reductions, and residuals keep the
production model practical on the RTX 3070 Ti.

The production model completed forward/backward/Adam on a 150-stone position
with 6,166 nodes, 442,732 relation traversals, and all 3,938 legal outputs:
2.045 seconds, 2,272 MiB peak allocated and 3,072 MiB peak reserved. The device
reported 6,949 MiB free before the step; this was not a competing-load benchmark.
These measurements establish executable capacity, not playing strength.
`python -m tests.benchmark_relational --output <report.json>` reproduces the
fixture and records configuration, source hashes, and memory measurements.
Geometry, action indexing, 12 symmetries, batch isolation, native/reference
parity, and checkpoint gradients are covered by `tests.test_relational`.
Held-out prediction gains and superiority against independent bots remain to
be established by training and external evaluation.

