# Earlier training models

The pattern, NNUE and relational pipelines below are retained for earlier experiments. Bubble currently trains the dense HexNet stack described in [the README](../README.md#train). Measurements here belong to the named historical configurations.

## Self-play learning

```sh
pip install -r requirements/learning.txt
python python/legacy/train.py --run runs/selfplay --iterations 10 --device cuda
```

In another terminal:

```sh
python python/dashboard.py --run runs/selfplay
```

Open <http://127.0.0.1:8766>. The dashboard shows self-play progress, positions collected, training and validation loss, checkpoint decisions, and Elo estimates against checkpoint 0. Everything stays in the run directory; no W&B account or external telemetry service is required.

The defaults use four CPU actors, 64 self-play games per iteration, a 50 ms self-play budget, and 40 evaluation games per opponent at 100 ms per turn. PyTorch trains on the GPU when available. Use `--device cpu` for a CPU learner. Each game cap is a truncation, not a draw. Truncated positions retain search targets but have no outcome label.

To generate games on the GPU and use larger learner batches:

```sh
python python/legacy/train.py --run runs/gpu-selfplay --selfplay-backend gpu --games 2048 --gpu-games-batch 0 --batch 0 --device cuda
python python/dashboard.py --run runs/gpu-selfplay
```

`--gpu-games-batch 0` chooses a batch from available VRAM and the episode cap. `--batch 0` increases the learner batch while preserving the reference batch-256 optimizer update count. `--updates-per-epoch` sets an explicit update budget. CUDA training uses fused AdamW and keeps replay tensors on the device. Replay sampling occurs before concatenating shards, and `--replay-positions` bounds retained positions. The limit also accounts for available device memory. Best validation weights and the matching optimizer state are saved.

The GPU `pattern` actor uses exact sparse rules with growing coordinate storage and incremental native-compatible pattern features. They first take exact one- or two-stone wins, then choose a placement belonging to a complete cover of all immediate opponent threats when such a cover exists. These choices override exploration. Quiet candidates combine nearby cells, axial development, and broader exploration, scored one placement at a time by the learned evaluator. They do not run native PVS; `--ms` and `--width` do not control this actor. Replay records identify the actor and one-placement target semantics. Promotion always measures the native deployed engine at equal wall-clock budgets.

On this RTX 3070 Ti, the tactical zero-residual GPU actor generated about 184 games/second and 28,300 placements/second at batch 512 with 32 quiet candidates and a 256-stone cap. Of those 512 games, 388 finished and 124 reached the cap; peak PyTorch allocation was about 920 MiB. These timings include generation and replay transfer, but not training or native evaluation. The earlier actor was faster but frequently missed immediate wins. Its shorter games make raw games/second an unfair performance comparison. Reproduce on your hardware:

```sh
python python/legacy/gpu_benchmark.py --actor --batches 64 256 512 2048 --placements 256
python python/legacy/gpu_benchmark.py --batches 64 512 2048 --placements 96
```

The second command compares only exact rule transitions and feature updates with C++. Small GPU batches are slower than native code. The dashboard exposes device utilization, VRAM, power, and temperature; telemetry includes other applications using the GPU. A small pattern learner cannot productively saturate every GPU unit at all times, so larger game batches and measured throughput guide settings.

The default `pattern` evaluator uses an `18 -> 32 -> 1` network to map six-cell ternary patterns to a bounded residual on top of the handwritten evaluator. It exports a 729-entry integer table. The `nnue` evaluator uses centered eleven-cell lines, nonlinear combinations of all three axes, an incremental 64-channel position summary, and separate value and conditional move-ranking heads. Both run entirely in C++ during search.

To train the full NNUE model with native search actors and a CUDA learner:

```sh
python python/legacy/train.py --model nnue --run runs/nnue --selfplay-backend native --device cuda --batch 0
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
python python/legacy/train.py --model nnue --selfplay-backend gpu --run runs/nnue-gpu --games 512 --gpu-beam 4 --device cuda
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

For a short end-to-end run, use a separate directory:

```sh
python python/legacy/train.py --run runs/short-run --iterations 1 --games 12 --eval-games 8 --ms 10 --eval-ms 20 --epochs 4 --workers 2
```

The evaluator increases undersized requests enough to make the pair test possible. This short command exercises collection, training, evaluation, and checkpoint decisions; it cannot by itself establish competitive strength.

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
python python/legacy/train.py --run runs/nnue-new-engine --model nnue --initial-model runs/nnue-old/checkpoints/0001/model.pt --initial-optimizer runs/nnue-old/checkpoints/0001/optimizer.pt --device cuda
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
python python/legacy/reanalysis.py --run runs/nnue-selfplay --iteration 1 --checkpoint 0 --max-positions 256 --ms 200 --width 32 --output runs/reanalysis-0001
python python/legacy/train.py --run runs/nnue-with-reanalysis --model nnue --reanalysis runs/reanalysis-0001 --device cuda
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
python python/legacy/klent.py --run runs/klent --initial-model runs/nnue-selfplay/checkpoints/0001/model.pt --games 16 --envs 8 --max-plies 128 --device cuda
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
python python/legacy/q_warmstart.py --corpus artifacts/datasets/human-warmstart-v1 --model artifacts/models/human-warmstart-12/model.pt --output artifacts/models/human-q-warmstart-12 --positions 40000 --epochs 12 --device cuda
python python/legacy/klent.py --run runs/human-initialized-klent --initial-model artifacts/models/human-q-warmstart-12/model.pt --initial-q artifacts/models/human-q-warmstart-12/q.pt --device cuda
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

## Relational policy/Q training

`relational_warmstart.py` initializes both heads of the primary relational model
from full human histories. Geometry is rebuilt from chronological moves. Existing
hashed corpus shard membership supplies the family split; NNUE features and
weights are not transferred. Only the recorded action receives a Q target, using
the actual terminal outcome from its mover's perspective. Human actions are
imitation labels, not claims of optimal play. Test and excluded games never
supply training or validation examples.

```powershell
python python/legacy/relational_warmstart.py --corpus artifacts/datasets/human-warmstart-v1 --output artifacts/models/relational-human --epochs 12 --device cuda
python python/legacy/relational_train.py --run runs/relational-v1 --initial-model artifacts/models/relational-human/model.pt --games 256 --envs 8 --iterations 5 --max-plies 300 --device cuda
```

The default network uses width 256, eight blocks, eight attention heads, feed-forward
width 1024 and sixteen global tokens. `--positions 0` uses all available human
positions after the corpus family-prefix boundary; a positive value selects a
seeded bounded sample. Graph microbatches are bounded by `--max-nodes` and
`--max-edges`. A single position exceeding a budget raises an explicit error;
legal actions are never cropped. CUDA uses BF16 matrix operations and FP32
normalization, losses and return targets.

KLENT collects each fresh corpus with frozen weights and fits it in exactly one
shuffled pass. It uses the existing scalar policy/Q objective and signed lambda
returns, including actual-player sign changes and explicit cap bootstrap. It
neither mixes old replay into the fit nor distills an NNUE value/export. Proofs
are not substituted for historical returns. New runs reset Adam; resuming an
existing run restores its model and optimizer and verifies source, engine,
initial model and previously consumed corpus hashes. A completed pending corpus
is reused after an interrupted fit.

`model.pt` contains schema `hexo-relational-policy-q-v1`, the complete model
configuration and state dictionary, including Q. `relational_train.load_model`
loads this format for the neural evaluator/player. Native NNUE loading is not a
supported deployment path. Training metrics remain unrated until the neural
player is measured against pinned independent opponents under stated budgets.
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

### Shared relational action/value contract

The model, collector and neural search consume chronological axial histories,
replayed by the exact native rules. The resulting position is identified by
`position_key`: SHA256 of sorted `(q, r, absolute_owner)` stones plus the native
player to move and placements remaining. History order within an equivalent
position is not a distinct neural state. Coordinates retain their actual values;
this identity is not a symmetry-canonical cache key.

`Graph.actions` contains **all** native legal coordinates as `int64[N, 2]`, in
`Game.legal_moves()` order (lexicographic q, then r). `Graph.player` and
`Graph.remaining` are authoritative native phase values. Packing preserves that
order and supplies `action_owner[N]` and `action_offsets[B+1]`; offsets start at
zero, end at N, and delimit each position's complete action set. Newly reachable
second placements are encoded from the position after the first placement.

The differentiable network returns finite FP32 `logits[N]`, bounded scalar
`q[N]` in [-1, 1], and the same `action_offsets`. Each Q value uses the player to
move's perspective. The inference adapter echoes `position_key`, `player`,
`remaining`, aligned `actions`, and `model_version` for every record. Deployed
`model_version` is the SHA256 of the exact loaded relational `model.pt` bytes.
The loader validates those bytes before deserialization. No adapter may fall back
to a native NNUE model, handwritten values, or reordered/truncated actions.

Value constructions are deliberately distinct:

- KLENT collection uses `mu = softmax((Q + beta * log(pi)) / (alpha + beta))`
  and `V_mu = sum(mu * Q)` for frozen-actor return bootstraps.
- Neural search uses the network prior `pi = softmax(logits)` and
  `V_pi = sum(pi * Q)` for unresolved leaves.

Neither is a proven value. Actual terminal outcomes come from the rules; proofs
retain their separate verified scope. Backups keep the sign when the same player
continues a turn and reverse it only when control changes. A first-placement win
terminates immediately. Training stores the acting distribution and action index
with the full legal-coordinate hash, player and phase, then checks that identity
when reconstructing replay. These requirements apply equally to raw-policy,
KLENT-improved-policy, Gumbel-search and verified-tactics modes of the same player.
Each graph exposes `position_key`, a SHA256 of the rule identifier, native
player/remaining phase, and sorted absolute stone coordinates and owners.
Evaluator records echo that identity, `player`, `remaining`, and `model_version`
alongside the full native-order actions, logits, and Q. Deployed callers pass the
checkpoint SHA as `model_version`; standalone callers receive a configuration
and parameter digest. The evaluator copies and freezes the supplied model so
later training or checkpoint loads on the caller's model cannot change its identity.
Terminal inference is rejected because the rules and search own terminal values.


## Search-trained policy/value self-play

`search_train.py` trains the relational policy/value model from Gumbel MCTS self-play. The policy target is the full legal-action Gumbel completed-Q improved policy. The value target is the final game result from the player-to-move perspective. Capped games keep their policy targets but supply no outcome target. This path does not use KLENT action-Q targets or external bots for checkpoint selection.

```powershell
python python/legacy/search_train.py --run runs/search-selfplay --initial-model path/to/model.pt --games 128 --eval-games 32 --evaluate-every 4 --replay-positions 200000 --reuse-ratio 4 --iterations 100 --device cuda
python python/dashboard.py --run runs/search-selfplay --port 8766
```

A policy/Q warm start preserves its backbone and policy and initializes a new scalar value head at zero. Native trees batch neural leaves across games. Graph memory limits split whole positions without cropping legal actions. The collector enables immediate tactical constraints in search and its normalized policy targets. Ordinary Gumbel play also enables those constraints; deep proof search remains optional. The nominal 16 simulations are retained, including inexpensive exact backups. Evaluation keeps tactics disabled, preserving the original fixed protocol for measuring network progress.

Checkpoint 0 is the internal Elo anchor at zero. Every fourth candidate by default plays equal-search-budget, color-swapped games against the previous checkpoint, incumbent champion, and a distinct checkpoint near 20% behind the previous one, plus a smaller anchor comparison when the anchor is not already required. Duplicate opponents are played once. Promotion requires full matches and more wins than losses against both the incumbent and selected older checkpoint after counting capped games as candidate losses for this decision. There is no p-value promotion gate. When early history has no distinct older checkpoint, promotion waits. Learning continues from the latest candidate regardless of champion selection. Ratings describe this internal league and search budget, not an external leaderboard.

Rerun the same command to resume completed artifacts; only the requested iteration count may change. Saved corpora and optimizer checkpoints are verified. A partial fitting pass restarts from the preceding checkpoint rather than silently applying the same targets twice. Active run source files must remain unchanged.

### Recent replay and learning credit

Each collection defaults to 128 attempted games. Capped episodes keep searched policy rows; their value targets are masked. Up to 200,000 eligible positions are read from the most recent corpora, including earlier actors in the same run. The oldest admitted corpus is trimmed at the position limit. Corpus manifests bind actor hashes and targets; fitting records all source manifest hashes and the number of presentations by target age.

`--reuse-ratio 4` permits exactly four example presentations per fresh admitted position. The learner samples shuffled passes across replay until that budget is spent, including a smaller final minibatch. It does not run repeated full replay epochs. The replay learner uses all eligible rows, so it has no compulsory 25% holdout; historical held-out losses remain visible and are not extended with training losses. Model weights and Adam state continue from the latest checkpoint.

`--evaluate-every 4` runs internal matches periodically. Collection, fitting and evaluation remain synchronous on the single GPU, but evaluation no longer follows every fitting cycle. Intermediate checkpoints may be unrated until they participate in a later comparison. This is a throughput choice, not evidence that those checkpoints improved.

Run `python python/legacy/search_evaluate.py --run runs/gumbel-policy-value-v1 --from-checkpoint 13 --threads 4`
from a separate frozen checkout to evaluate intervening checkpoints while the GPU
learner continues. It uses the same search budget and previous/champion/older/reference
opponent allocation on CPU float32. Each complete color-swapped pair updates
`background-league.json`; the dashboard overlays these estimates without modifying
the trainer's league, champion, models, optimizer or corpus. Partial match sets are
labeled provisional and report played/planned counts. Fully terminal opening pairs
from a capped comparison still inform provisional Elo; pairs containing a cap do not.
CPU results show promotion evidence only after full
incumbent and distinct older matches both have winning conservative scores;
actual champion promotion still uses scheduled GPU matches.

CPU float32 and CUDA mixed-precision comparisons remain separate rating protocols.
Their conditional 95% credible intervals do not account for backend differences or
selection caused by capped games. Background evaluation can lag checkpoint creation;
queued checkpoints remain unrated until games supply evidence. Keep the worker source
unchanged while it runs. It resumes saved pairs, rejects changed identities and owns
an exclusive `background-evaluation.lock`.

The worker also samples up to 32 terminal games from each next collection, taking
three positions per game. It measures that collection's actor before it has trained
on those games. Fresh validation policy cross-entropy uses recorded search targets;
value MSE uses terminal outcomes, with a zero-value baseline of 1. These diagnostics
are conditional on the actor's terminal games and are separate from Elo and training
loss. They appear after the next corpus is complete. The dashboard labels historical
held-out validation separately from fresh-collection validation.

Exact cache keys now derive colored stones and turn phase from native legal histories without constructing another rules board. Full tuples still distinguish hash collisions. Persistent trees and model-versioned predictions retain their existing reuse semantics. Selective higher-budget targets and incremental neural trunks are not enabled.

### Moving internal opponents and league ratings

Search-training evaluation uses `--eval-games 32` against the previous checkpoint,
the current champion, and the checkpoint nearest 20% behind the previous one that
is distinct from both. It uses `--reference-games 8` against checkpoint 0 unless
that checkpoint is already required, in which case it gets the full allocation.
Duplicate opponents are played once. Self-play training still uses
the latest network on both sides with Gumbel search targets and terminal outcomes.

Displayed joint Elo fits fully terminal color-swapped pairs from current-protocol
comparisons, fixing checkpoint 0 at zero. The exact search-run4 (57df979) to
search-run5 (79751dd) promotion-only migration also accepts prior CUDA reports
after checking the hash-bound migration history, original report manifests,
model hashes, unchanged evaluation settings and the two audited source maps.
Their original protocols remain in the reports; CPU and earlier source revisions
are excluded. Completed pairs contribute even when other
pairs in that match cap; no outcome is invented for capped or unsaved games. Historical
ratings can change when new results arrive. Raw match scores and reference-only
estimates remain in `league.json`.

The displayed 95% intervals are approximate Bayesian credible intervals for the
joint model conditional on pair completion. The dashboard labels them provisional
where capped or unsaved pairs are present. Match score bounds run from known wins
divided by planned games to known wins plus all unknown outcomes divided by planned
games. These are deterministic observed-score bounds, not 95% sampling intervals.
New comparisons use distinct opponent-specific seeds; older comparisons may
share opening schedules, a dependence this approximation does not model across
comparisons. Promotion uses the separately recorded conservative two-opponent scores.

For live sparse-checkpoint estimates, run `python python/legacy/paired_rating.py --run runs/gumbel-policy-value-v1`
beside the evaluator. The dashboard prefers its `paired-ratings.json` output.
This fits actual opening-pair counts jointly, replacing the older projection of
independently smoothed matchup scores. In particular, one swept pair against a weak
reference is not converted into an artificial 70% observation that drags down a model
which tied a stronger champion. No extra games or pseudo-wins enter this likelihood.

For an Elo difference `d` in log-odds units, pair outcomes 0, 1, 2 wins have probabilities
`softmax(-h, tau, h)`, with `h = d/2 + asinh(exp(tau)*sinh(d/2)/2)`.
The expected game score is exactly `logistic(d)` for every `tau`. A shared dispersion
parameter permits more or fewer split pairs than independent games. Ratings have a
weak Normal(0, 1000 Elo) prior with checkpoint 0 fixed; `tau` has a Normal(log(2), 2)
prior. The point estimate is the joint posterior mode. A multivariate Student-t proposal
around that mode supplies 32,768 importance-weighted posterior samples for the 95%
credible interval; publication requires at least 1,000 effective samples. The dashboard
shows the number of rated and censored pairs involving each checkpoint, including
games as opponent.

These are model-dependent estimates. A single split pair means zero head-to-head Elo
difference with large uncertainty; other observed matchups can still change the joint
estimate. Only fully terminal pairs enter the likelihood. Completion may depend on
model strength, so conditional Elo and its credible interval can be biased for the
full match. CPU/CUDA differences and historical shared-opening dependence across
comparisons remain outside this uncertainty model.
The rating worker reads saved results only and changes neither promotion nor training.

To upgrade a completed older search run, reuse its original learning and search
arguments, add `--upgrade-run --reference-games 8`, and increase `--iterations`. Collection size, replay capacity/reuse, evaluation cadence and actor tactics may change during this explicit upgrade; network, optimizer hyperparameters and evaluation search settings must match. The default replay options apply to the upgrade.
The trainer requires a finished checkpoint boundary and an exclusive run lock.
It verifies and binds the old manifests in `history.json`, preserves model and
optimizer files, and records the new source identity. Normal later resumes omit
`--upgrade-run`. Never change the source checkout of an active trainer.
