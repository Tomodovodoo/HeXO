# Dense training

The current Bubble learner uses HexNet. Start a new run using the commands in [the README](../README.md#train). Run layouts and settings are defined in [dense_config.py](../python/dense_config.py).

## Short CPU run

Run from the checkout root after the native build, editable install and learning dependency install. This exercises collection, checkpoint export and resuming in a small new directory.

```sh
python python/dense_config.py --run runs/bubble-short --device cpu --blocks 2 --channels 32 --pool-every 2 --batch 32 --window-min-rows 8 --warmup-steps 1 --export-every 1 --validation-rows 16 --validation-quota 16 --games-in-flight 4 --leaf-batch 8 --full-sims 4 --cheap-sims 4 --root-samples 4 --max-plies 16 --shard-games 4
python python/dense_learn.py --run runs/bubble-short --steps 0 --workers 1
python python/dense_selfplay.py --run runs/bubble-short --games 4
python python/dense_learn.py --run runs/bubble-short --steps 1 --workers 1
```

The short games can reach their placement cap. Their outcomes stay masked; search values supply the bootstrap target. This example checks that the pipeline runs, without establishing model strength. Solver queries and external opponents are disabled by default.

## Dense learner value targets

`dense_learn.py` derives the value target of every row from its episode
(`dense_data.value_targets`). Capped games always use TD(`td_lambda`) over the
searched root values. For finished games, `--value-target` chooses the target:

- `outcome` (the default): the hard result, 1 for the side that won and 0 for
  the side that lost.
- `td`: TD(`outcome_lambda`, default 0.98). The recursion is the capped-game one,
  started from the outcome at the last ply.
- `calibrated`: P(side to move wins | v, h), where v is the searched root value
  at the ply and h is the plies remaining. A null value carries the previous one
  forward along the game. Before the first search value, the target is the base
  rate.

The `calibrated` map is a ridge-regularised logistic regression that is shrunk
toward the base rate. Its inputs are the product of degree-1 B-splines in
log2(h) (knots 1, 2, 4, ..., 256) with [1, logit((1+v)/2)]. The learner fits it
from the newest `calibration_games` (4000) finished training games in the
replay window, at startup and again at every export, in about 2 s of CPU. Each
export trains the render workers on its new map until the next one. Where
search values carry no information, the map returns the base rate. Near the
end of a game it returns about the outcome. With fewer than 200 finished games,
`calibrated` falls back to `outcome`. Each checkpoint records its map as
`metrics.calibration`: the coefficients, plus a table over v in -1..1 in steps
of 0.25 and h in 0..160 in steps of 8.

With `--bootstrap-full-only`, all three chains (the capped TD chain, `td` and
`calibrated`) use only full-search root values. `--outcome-weight w` adds a
KataGo-style value-logit BCE against the hard outcome, with weight w. That head,
`outcome_bce`, is always logged. The validation curves by plies remaining
always score finished games against their hard outcome.

`--deblunder-weight w`, default 0, uses transient wins found by `dense_solve.py`
to soften a losing owner's earlier value targets. The proof pass writes
`deblunder` records automatically on newly solved shards. For that owner's
rows before the window, after the previous proof window or from game start,
the learner uses `w * 1 + (1 - w) * original_outcome` as the win probability
for both value losses. This replaces the calibrated or TD target on those
rows. Exact labels take precedence, and these soft rows remain unproven.
The original outcome stays available for diagnostics. Validation reports
`value_bce_deblundered` and `deblundered_rows`, also per source.
Restart the proof pass with its existing flags and the learner with, for
example, `--deblunder-weight 0.25`. Existing sidecars are not rewritten.

### Cheap rows

Most self-play rows come from cheap searches: they have no policy target and
their value weight is `cheap_value_weight` (0.25). `--cheap-row-fraction f`
(default 1) keeps only a share f of these ordinary cheap rows in training.
Full-search rows and rows with an exact label (a proof, including forced-line
rows) are always kept. Each row is kept or dropped by a hash of the run seed,
the shard name and the row index, so the learner, its render workers and every
restart agree. f = 0 matches KataGo, which does not train on cheap rows.

Pacing counts kept rows only, so `samples_per_row` stays the number of
presentations per kept row: at f = 0.5 a shard adds fewer rows to the pacing
budget, and each kept row is seen as often as before. Changing f on a restart
moves the pacing base to the current count, as a change of `samples_per_row`
does. Held-out validation rows are all scored whatever f is.
`learner-status.json` reports `retained_rows` (the kept training rows of the
window) and `retained_fraction` (their share of the window's training rows).
At f < 1 the learner reads every shard's rows once at startup to count them.

## Dense learner future occupancy

`--future-target legacy` is the default. It keeps the occupancy BCE at 6 and
20 placements, including stones already on the board. `--future-target masked`
uses a fresh three-class head at 20 placements: empty, own, or opponent from
the current mover's view. Cross-entropy is averaged over currently empty cells
inside each crop, then over rows with a known target. Finished games use their
final board when they end before 20 placements; capped games need all 20.
`--future-weight` keeps its default coefficient of 0.5.

Restart the learner with its existing arguments plus `--future-target masked`.
The first mode switch preserves shared and legacy head weights, training
counters, shared Adam moments and the EMA update count, and adds the new head.
The new head starts without optimizer moments; its raw and EMA weights match.
Later resumes restore the saved mode, head, and optimizer. To switch back,
pass `--future-target legacy`. Actors and evaluators read either checkpoint
format without extra flags.

The new loss is `future_masked_ce`; legacy remains `future_bce`. Each fixed
validation source reports the active metric on held and training panels, plus
their gap, under separate names such as `newest_future_masked_ce` and
`newest_future_bce`. These are different objectives, not comparable loss values.

## Certified winning moves

`--proof-policy-weight w` teaches moves from verified winning certificates,
including solver roots and generated proof continuations that have no search
policy. Its default is 0. With `--proof-policy-missing-only`, it supplies targets
only for those empty rows; existing search policies keep their targets and loss
weights. For example, `--proof-policy-weight 0.25 --proof-policy-missing-only`
gives the added rows policy loss weight 0.25. This uses the existing heads and
rendered positions, with no additional search or inference.

The certificate identifies known winning placements rather than every winning
move. A missing witness or a proven losing position supplies no new policy
target. The optional mode's playing benefit still needs measurement.

## Dense actor batches

`--book-fraction 0.25 --restart-fraction 0.1` allocates 25% of newly started games to the live off-policy
opening pool, 10% to the restart buffer, and the remaining 65% to ordinary starts. The book fraction defaults
to 0. Classes in the eligible off-policy pool are sampled uniformly, with a random hex symmetry; ordinary
book entries do not consume this allocation. These start shares also apply with historical opponents enabled.
An empty eligible pool or unavailable restart source produces an ordinary start. The actor heartbeat reports
the configured fractions and eligible pool size; shard events and manifests count completed book/restart games.

Actors reload the read-only book snapshot at shard and learner-phase boundaries. Book games store origin
`book` and `{suite, key, digest, ply, off_policy}` metadata. The preset stones have no training rows and null
root values. Search and training begin after that prefix, using the normal full/cheap search schedule and
solver settings. The manifest's `forced_plies` includes both book and restart prefixes.

Both the learner and actors accept `--net-kernels fused` for optional Triton GPU kernels. The default is
`reference`; checkpoints load in either mode. See [GPU kernels](gpu-kernels.md) for installation, the paired
benchmarks, profiling commands, and the batch-256 training validation that remains blocked by the shared-card
memory cap.

`dense_selfplay.py` accepts `--games-in-flight` and `--leaf-batch` per worker, alongside `--games` per process. The
actor heartbeat and metrics log report `mean_batch` and `full_batch_fraction`, the share of model submissions with
exactly `leaf_batch` distinct positions. More games can supply more leaves to each call; increasing `leaf_batch`
alone only raises the limit. These are engine submission counts; the evaluator splits each submission into model
forwards by crop size and `MAX_CELLS`.

Each game owns a native CPU tree, with no fixed per-game GPU allocation. On this Windows host, 128 trees at a
20-ply position used about 30 KiB of private memory each before search and 3.39 MiB each after a 64-simulation
search with tactics enabled, measured without a model or GPU. An additional inference row makes an 8-plane bf16
GPU input of `8 * size * size * 2` bytes, or 36 KiB at a 48x48 crop. One 96-channel bf16 activation at that size
is 432 KiB per row; the network also needs other activations and temporary buffers. Crop size changes the cost
quadratically, so these tensor sizes are not a measured peak VRAM increase.
