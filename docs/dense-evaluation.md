# Evaluation

`python python/dense_eval.py loop --run R` rates every new checkpoint against the champion and keeps the league in `league.json`. Every setting is a field of `EvaluationSettings` in `python/dense_config.py`, overridden per process with `--eval-*` flags. `python python/dense_eval.py settle --run R --checkpoint main/032500` asks the running evaluator to settle that checkpoint on its completed games.

## Games

The evaluator keeps `pool_games` (64) games in flight on one engine, so the GPU batch stays full. Every pairing draws colour-swapped openings from the opening book, and each finished colour pair is written to its report at once, so a restart loses only the games in flight. Games that reach the placement cap count half a point each way.

While a promotion decision is pending, direct games against the champion take the whole pool until `sprt_min_games` (64) are complete. After that, at most `evidence_share` (1/4) of the pool goes to whichever other pairing most reduces the posterior variance of the decision. A newer checkpoint supersedes the trial: its running games finish and count, and the verdict settles on all of them.

The anchor opponent is Seal by default; `--eval-external-engine` with `--eval-external-name` rates champions against any engine that speaks the Six protocol under its own league id ([six-engine.md](six-engine.md)). After a decision the evaluator plays the champion's anchor, a panel of the closest rated checkpoints, and fill games that sharpen the widest intervals, until the next checkpoint appears. Pairings whose expected score exceeds `max_expected_score` (0.85, about 300 Elo) are skipped.

## Ratings and promotion

One Bradley-Terry posterior covers every rated checkpoint, the candidate and Seal, using every report. Each colour-swapped opening pair is one observation with five outcomes (0 to 2 points), because both games share an opening, and each pairing's likelihood is divided by its dispersion, the observed variance of pair points over the binomial one. Pairs that sweep more often than chance widen the interval; pairs that split more often narrow it. Each pair of players also gets a matchup deviation with prior sd `matchup_prior_elo` (30), so a pair's own games outweigh the transitive picture when they disagree.

A candidate is promoted when it has the highest posterior rating and P(candidate beats champion by more than `sprt_elo0`) is at least `promote_confidence` (0.9); rejected when that probability is at most `1 - promote_confidence`; and left running while the direct-only and pooled estimates disagree beyond their intervals. On start the evaluator re-applies the rule to the reports on disk. `decision sprt` keeps a generalised SPRT over the same pair outcomes instead.

`league.json` also records a calibration diagnostic: for every posterior verdict, the stated sd of the rating difference against how far it later moved.

The dashboard also shows provisional external opponents in the scoreboard and as dashed Elo references. When a match uses different search settings from the league, compare those settings against the same checkpoint's normal evaluator settings first. Fit the calibration and opponent comparisons jointly, then translate the resulting opponent difference through that checkpoint's current league Elo. The displayed interval includes uncertainty from both comparisons and the reference checkpoint. Hash-bound `matches/*/*-elo-estimate.json` artifacts retain the match and calibration reports; changing either report invalidates the displayed estimate. Match games played at different budgets stay separate.

## Variants

A variant is a rated checkpoint's weights with overridden per-side settings (`sims`, `root_samples`, `tactics`, `search_graph`, `search_choice`, `solver_*`), rated in the league as `<checkpoint>@<name>`:

```sh
python python/dense_eval.py variant --run R --checkpoint champion --name solver --set solver_root_nodes=135 --set solver_finalists=2 --set solver_finalist_nodes=135 --set solver_threat_nodes=135
```

The evaluator decides it against its own checkpoint with the promotion rule. Variants never become champion and never reach the actors; they are how a search setting earns its place before it is switched on for the run.

To test selecting the highest improved search policy instead of the final Gumbel score:

```sh
python python/dense_eval.py variant --run R --checkpoint champion --name policy --set search_choice=policy
```

`search_choice` defaults to `gumbel`. `policy` changes only the final move at an unproven root; internal search,
simulation budgets and policy targets stay the same. It can choose an eligible move removed in an earlier
halving round. Proven roots retain shortest wins and longest resistance. `--eval-search-choice policy` sets it
for both model sides of ordinary matches. Historical reports without this field used `gumbel`.

## Opening book

`python/dense_openings.py` keeps `openings.json`, a DAG of symmetry-reduced positions. Each completed pair is recorded on every node its opening passed through. The live book holds `book_size` (512) openings of 3 to `book_plies` (5) placements, each the shortest plausible unused prefix of a line sampled from the champion's own searches. At every champion change and every `book_refresh_hours` (6) the champion re-scores the book and retires openings that are implausible (policy below `book_min_prob`), skewed toward one colour beyond `book_max_skew` Elo, short and skewed, or nested inside a child; retired openings are replaced. A refresh that changes the openings starts comparisons afresh.

Imported openings are different: `python python/dense_openings.py import --run R --source openings/off-policy-107500-p2-ge47p5-v1.json` adds two-stone openings outside the sampled DAG that the actors draw through `--book-fraction`. They are marked `off_policy`, cannot be retired by policy reach, and only the value floor `book_min_p2_value` (45%) and played-game skew can retire them. `openings/tactical/known-loss-v1.json` imports openings with a known result as a separate `tactical` pool: actors train on them, evaluation never draws them, and the dashboard reports how often the actors convert the known win. Run the writing commands (`import`, `refresh`, `prune`) while the evaluator is stopped.

`league.json` `openings` holds first and second player results overall and per player; `/api/openings` on the dashboard serves the DAG with its statistics.
