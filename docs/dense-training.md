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

An actor writes one row per placement. A full-search row (the `full_fraction` of placements, 128 simulations in the live run) carries the improved search policy as its policy target and a value target. A cheap-search row (12 simulations) carries a value target at weight `cheap_value_weight` (0.25) and no policy target; `--cheap-row-fraction f` keeps a hash-chosen share `f` of them, and `f = 0` is KataGo's choice of not training on them at all. Each placement draws its kind independently, so both stones of a turn are full searches in only `full_fraction` squared of turns; the actor flag `--full-turns` lets a turn's second stone repeat its first stone's draw, which keeps the full share and makes every full first stone a pair that `--pair-policy-weight` can use. Rows with an exact label, from a proof or a forced line, are always kept at value weight `proven_value_weight` (2) and their outcome loss is masked.

Actors also store the network's own value prediction for each root, before search. Completed games batch any missing predictions, including trained solver continuation rows. The existing bounded regret sampler prioritizes exact rows whose predictions disagree with their proven result, so these continuations can receive extra value and proof-policy training. Older rows without predictions keep their ordinary sampling weight. The proof pass also uses these predictions to find positions where the network was wrong although the solver knew better.

## Value targets

`--value-target` picks how a finished game labels its rows:

- `outcome`: the hard result.
- `td`: TD(`outcome_lambda`) over the searched root values, started from the outcome.
- `calibrated`: P(side to move wins | search value, plies remaining). The map is a shrunk logistic regression on B-splines of log2(plies remaining) times the search value, refitted from the newest `calibration_games` (4000) finished games at every export. Far from the end it returns the base rate; near the end it returns the outcome. Below 200 finished games it falls back to `outcome`.

Capped games always use TD over root values. `--bootstrap-full-only` restricts every chain to full-search values. `--outcome-weight w` adds a separate value-logit loss against the hard outcome, logged as `outcome_bce` and always reported on validation by plies remaining.

`--deblunder-weight w` softens the earlier value targets of a player who later blundered a proven win, using the proof pass's `deblunder` records: those rows get `w` toward a win. Exact labels take precedence.

## Policy targets

The policy target is the root search's improved policy over every eligible legal move. Unvisited moves receive a mixed value estimate. Three sources add to it:

- `--proof-policy-weight w` mixes the certificate's winning stones into winning rows, `(search + w * proof) / (1 + w)`; with `--proof-policy-missing-only` it only fills rows that have no search policy, such as solver roots and forced-line rows.
- `--pair-policy-weight w` mixes the same turn's second-stone search policy into a first stone's target, `(search + w * second) / (1 + w)`, where it exists. A turn's two stones reach the same position in either order, so the complement the second search found is also a good first stone, and the first-stone prior should rank both stones of a good pair near the top. Cells that only became legal after the first stone are dropped and the rest renormalised. Second stones and rows without both searches keep their targets. `w = 1` gives the played pair equal weight when both searches are sharp. Replay validation's policy metrics then score first stones against the mixed target.
- `--regret-fraction f` draws a share `f` of each batch from the proof pass's regret buffer, positions where the network's value was furthest from a proven result.
- `--future-target masked` adds a three-class occupancy head (empty, own, opponent) at 20 placements ahead, weight `--future-weight` (0.5). The default `legacy` keeps the older occupancy targets at 6 and 20 placements. Switching keeps the shared weights and optimizer state.

At export, held replay validation reports `certified_policy_mass`, the probability assigned to its certified winning stones, and `certified_policy_top1`, whether any stone tied for highest policy belongs to that certificate. These include winning rows with a witness even when their policy training target is empty. `certified_policy_first_*` and `certified_policy_second_*` split the measurements by placements remaining in the turn; each group includes its row count. Source-specific fixed panels report the same fields on their existing full-search rows. They measure recognition of a verified continuation independently of proof-policy weighting, rather than agreement with a changing search target.

Replay validation's policy CE, target entropy, KL and top-1 use the same policy row weights, so CE equals target entropy plus KL even when certificate rows have a smaller loss weight. `policy_top2` is the share of rows where the network's argmax is among the target's two highest-mass moves (ties at the second mass count), so a pair-mixed first stone scores when the network picks either stone of the pair. `policy_argmax_mass` is the mean target probability on the network's argmax. `policy_pair_top1` is the share of first-stone rows trained on a pair target where the network's argmax is one of the two stones the turn played, with its count in `policy_pair_rows`. `policy_first_*` and `policy_second_*` split top-1, top-2 and argmax mass by placements remaining in the turn, with row counts, as the certified metrics do. Certified-move recognition remains an unweighted mean over positions with a winning witness.

## Window and pacing

The replay window follows KataGo: at least `window_min_rows` full-search rows, then it grows by `window_expand_per_row` times the extra rows, tapered by the exponent `window_taper`. Pacing keeps `samples_per_row` presentations per kept row; changing it, or the cheap-row fraction, resets the pacing base at the current row count. Phases are off in a new run. With `--phase-rows N` on the learner and `--phase-follow` on the actors, the two alternate: actors play until N new rows exist, then pause while the learner trains through them, and the evaluator yields while either is busy. That is how the live run shared one GPU; without it, all three compete for the card at once. `learner-status.json` reports the window size, retained rows, pacing backlog and phase state.

## Book and restart starts

`--book-fraction` starts that share of new games from the opening book's off-policy pool, with a random hex symmetry, and `--restart-fraction` from the proof pass's restart buffer. Preset stones produce no rows; search and training start after them. A tactical opening from `openings/tactical/` fixes the value target of the first position after its prefix, so a later blunder cannot contradict the opening's known result.

## Actor search

Actors search on the hybrid scheduler ([search-scheduler-design.md](search-scheduler-design.md)) unless `--no-hybrid-scheduler` is given: native game graphs that keep up to `HYBRID_GAME_GRAPH` (1024) expanded nodes per game, one model-keyed inference queue, and `hybrid_proof_workers` (12) CPU proof workers that may take `hybrid_proof_budget` (0.1) of each graph owner's time, their proof loops reusing the solver's stamps of earlier certificates (`hybrid_proof_stamps`, on; `--no-hybrid-proof-stamps` turns it off). The older per-query solver budgets (`solver_*_nodes`), `pv_check` and `proven_line_rows` belong to the Python-coordinated search of `python/neural_search.py` and are refused unless `hybrid_scheduler` is off. An actor without the tactical solver build runs with no proof workers and logs a warning event.

## Kernels

`--net-kernels fused` on the learner, actors and evaluator selects the Triton kernels described in [gpu-kernels.md](gpu-kernels.md). Checkpoints load in either mode.

## Throughput

Measured on the RTX 3070 Ti with a read-only copy of the run at `reset/195000` (batch 256, `--net-kernels fused`, idle machine):

| | samples/s |
|---|---:|
| learner at 0341850 (three runs) | 746, 785, 808 |
| learner at 267e9ef (two runs) | 1010, 1011 |
| same day as the next row, at df8f8ee (two runs) | 926, 943 |
| channels-last training (two runs) | 1135, 1158 |

A steady export takes 31 s instead of 35 s, so a 2500-step export cycle trains about 960 samples/s instead of 750. Before, the training thread stalled 2.8 s every 30 seconds rebuilding replay priorities, buckets were padded to 16 rows (25% more cells than the rows hold), each batch was read from the worker pipe on the training thread, and the line convolution and pooling took 29% of the GPU step. Two 1500-step runs from the same checkpoint give the same training losses and validation metrics within their noise.

With `--net-kernels fused` the learner trains channels-last: cuDNN no longer converts every convolution's operands to NHWC and back (about a tenth of the GPU step), and the tap gradients of the line convolution are reduced over position tiles of all taps at once (178 ms per ten steps instead of 436). Saved-batch steps run at 1210 to 1218 samples/s against 985 to 989 for main. Two 1500-step runs per arm from the same checkpoint give training losses within the spread of their 20-step logs and the same exported validation metrics within the arms' own run-to-run spread.

A restart reads the shards through an index in `<run>/cache/shards/` (about 2 GB for the live run's 6,071 shards). The first learner start builds it with eight worker processes; later starts reuse it and rebuild only shards whose files changed. On the same copy, launch to the first logged step took 217 and 381 s without the index and 32 to 44 s with it (88 s while it was being built); the first export, which builds the validation sets, took 276 and 292 s without it and 37 to 40 s with it.
