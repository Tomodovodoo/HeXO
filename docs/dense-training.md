# Training

The learner (`python/dense_learn.py`) trains HexNet on the actors' game shards. Every setting is a field of the dataclasses in `python/dense_config.py`, saved in the run's `config.json`; a flag on the command line overrides it for that process only. Start a run with `python python/bubble.py train`, as in the README.

## A short CPU run

This checks that collection, export and resume work, in a few minutes and without a GPU:

```sh
python python/dense_config.py --run runs/bubble-short --device cpu --blocks 2 --channels 32 --pool-every 2 --batch 32 --window-min-rows 8 --warmup-steps 1 --export-every 1 --validation-rows 16 --validation-quota 16 --games-in-flight 4 --leaf-batch 8 --full-sims 4 --cheap-sims 4 --root-samples 4 --max-plies 16 --shard-games 4
python python/dense_learn.py --run runs/bubble-short --steps 0 --workers 1
python python/dense_selfplay.py --run runs/bubble-short --games 4
python python/dense_learn.py --run runs/bubble-short --steps 1 --workers 1
```

The games hit their placement cap, so their outcomes stay masked and the search values supply the value target. It proves nothing about strength.

## Rows

An actor writes one row per placement. A full-search row (the `full_fraction` of placements, 128 simulations in the live run) carries the improved search policy as its policy target and a value target. A cheap-search row (12 simulations) carries a value target at weight `cheap_value_weight` (0.25) and no policy target; `--cheap-row-fraction f` keeps a hash-chosen share `f` of them, and `f = 0` is KataGo's choice of not training on them at all. Rows with an exact label, from a proof or a forced line, are always kept at value weight `proven_value_weight` (2) and their outcome loss is masked.

Actors also store the network's own value prediction for each root, before search. The proof pass uses it to find positions where the network was wrong although the solver knew better.

## Value targets

`--value-target` picks how a finished game labels its rows:

- `outcome`: the hard result.
- `td`: TD(`outcome_lambda`) over the searched root values, started from the outcome.
- `calibrated`: P(side to move wins | search value, plies remaining). The map is a shrunk logistic regression on B-splines of log2(plies remaining) times the search value, refitted from the newest `calibration_games` (4000) finished games at every export. Far from the end it returns the base rate; near the end it returns the outcome. Below 200 finished games it falls back to `outcome`.

Capped games always use TD over root values. `--bootstrap-full-only` restricts every chain to full-search values. `--outcome-weight w` adds a separate value-logit loss against the hard outcome, logged as `outcome_bce` and always reported on validation by plies remaining.

`--deblunder-weight w` softens the earlier value targets of a player who later blundered a proven win, using the proof pass's `deblunder` records: those rows get `w` toward a win. Exact labels take precedence.

## Policy targets

The policy target is the improved policy of the root search over the moves it considered. Three sources add to it:

- `--proof-policy-weight w` mixes the certificate's winning stones into winning rows, `(search + w * proof) / (1 + w)`; with `--proof-policy-missing-only` it only fills rows that have no search policy, such as solver roots and forced-line rows.
- `--regret-fraction f` draws a share `f` of each batch from the proof pass's regret buffer, positions where the network's value was furthest from a proven result.
- `--future-target masked` adds a three-class occupancy head (empty, own, opponent) at 20 placements ahead, weight `--future-weight` (0.5). The default `legacy` keeps the older occupancy targets at 6 and 20 placements. Switching keeps the shared weights and optimizer state.

## Window and pacing

The replay window follows KataGo: at least `window_min_rows` full-search rows, then it grows by `window_expand_per_row` times the extra rows, tapered by the exponent `window_taper`. Pacing keeps `samples_per_row` presentations per kept row; changing it, or the cheap-row fraction, resets the pacing base at the current row count. The learner and actors alternate in phases: actors play until `phase_rows` new rows exist, then pause while the learner trains through them, and the evaluator yields while either is busy. `learner-status.json` reports the window size, retained rows, pacing backlog and phase state.

## Book and restart starts

`--book-fraction` starts that share of new games from the opening book's off-policy pool, with a random hex symmetry, and `--restart-fraction` from the proof pass's restart buffer. Preset stones produce no rows; search and training start after them. A tactical opening from `openings/tactical/` fixes the value target of the first position after its prefix, so a later blunder cannot contradict the opening's known result.

## Kernels

`--net-kernels fused` on the learner, actors and evaluator selects the Triton kernels described in [gpu-kernels.md](gpu-kernels.md). Checkpoints load in either mode.
