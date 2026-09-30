## Dense evaluator

`dense_eval.py loop` rates each new dense checkpoint against the champion and keeps the league in
`league.json`; its module docstring is the full contract, and every setting is an `EvaluationSettings` field in
`dense_config.py` (override per process with `--eval-*`).
`python python/dense_eval.py settle --run runs/dense-v1 --checkpoint main/032500` asks the running evaluator to settle that checkpoint on its completed games at its next step.

- **Promotion** (`decision`, default `posterior`). One Bradley-Terry posterior covers every rated checkpoint, the
  candidate and Seal, and it uses every report: direct games, games against the previous champion, against Seal
  and against panel members. Each pair also gets a matchup deviation (prior sd `matchup_prior_elo`, default 30),
  so a pair's own games outweigh the transitive picture when the two disagree. Each colour-swapped opening pair
  is one observation with five outcomes (0, 1/2, 1, 3/2 or 2 points; a capped game is half a point), because the
  two games of a pair share their opening. The likelihood of each pair of players is divided by its dispersion:
  the observed variance of the pair points over what two independent games at the same score would give, shrunk
  toward 1 by four pseudo-pairs. Pairs that sweep (2-0 or 0-2) more often than chance widen the interval; pairs
  that split 1-1 more often than chance narrow it, since the opening then decides the colour rather than the
  player. The dispersion scales the weight of a pair's games, so it moves the point estimate only where priors or
  other pairings compete with them. `evaluator-status.json` reports the effective pair count (pairs over
  dispersion) of the direct games. The candidate needs at least
  `sprt_min_games` direct games. No separate bound applies to its rating sd: P(better) already accounts for it.
  It is promoted when it has the highest posterior rating and P(candidate - champion > `sprt_elo0`) is at least
  `promote_confidence`. It is rejected when that probability is at most 1 - `promote_confidence`. Neither
  happens while the direct-only and pooled estimates disagree beyond their intervals. `decision sprt` keeps the
  sequential test (`sprt_elo0` 0, `sprt_elo1` 25), the generalized SPRT over the same five pair outcomes, so its
  log-likelihood ratio carries the pair-level variance too. Each verdict records its likelihood as `model`
  (`pentanomial`). A decision recorded under another model stays settled: the start-up review does not re-judge it
  and the calibration diagnostic leaves it out.
- **Calibration diagnostic.** Each posterior verdict records the sd of delta it stated. Once the checkpoint has
  three later comparisons, `league.json` `calibration` compares the realised RMS shift of delta with what a
  calibrated posterior expects (root mean of sd then squared minus sd now squared). A realised RMS well below the
  expected one means the posterior overstates its variance. No decision reads it.
- **Continuous pool.** Like the actors, the evaluator keeps `pool_games` (64) games in flight on one engine. When
  a game ends, the next opening of its pairing starts at once (both colours together), so the GPU batch stays
  full. Each completed colour pair is written to its report immediately, so a restart loses only the games in
  flight, and the evaluation resumes where it stopped.
- **Direct games first.** While a promotion decision is pending, direct games against the champion take the
  whole pool until `sprt_min_games` (64) of them are complete. After that, at most `evidence_share` (1/4) of
  the pool may go to the evidence pairing whose games most reduce the posterior variance of the decision: the
  candidate or champion against the previous champion or Seal. The pairing is re-chosen after every
  completed colour pair. A newer checkpoint or a pairing change stops new games of the old pairing; its
  running games finish and count. A superseded decision settles on all of them: the candidate is promoted when
  P(candidate - champion > `sprt_elo0`) is at least `promote_confidence`, whether or not the other readiness
  conditions hold (`decision sprt` settles the same way). On every start the evaluator re-applies the rule to
  the reports on disk, and a rated checkpoint that already passes it is crowned at once.
- **Streaming.** `evaluator-status.json` carries the pool composition, the running tally of the current
  comparison (every recorded game of the pair, in either role and across restarts, updated per finished game) and the pending verdict. The dashboard shows all three, including a
  provisional league row for the candidate.
- **Idle work.** After the decision, the evaluator plays the champion's Seal anchor, the adaptive panel (the
  rated checkpoints closest to the champion) and other optional comparisons. It then plays fill games until the
  next checkpoint appears: the champion against Seal until their interval is `anchor_target_halfwidth` narrow,
  `games` of the newest checkpoint against the previous champion, then the widest pair among the top
  `fill_top`. Automatic anchors, panels, evidence and idle pairings are excluded when either side's expected
  score exceeds `max_expected_score`, even if an anchor quota is still owed. The default 0.85 permits about
  301 Elo difference; `--eval-max-expected-score 0.9090909090909091` permits up to 400 Elo. Unknown ratings
  remain eligible so new opponents can be rated. Direct promotion/variant trials and explicit `match`
  commands retain their own game budgets. Historical Seal games remain part of the rating evidence.
- **Variants.** An A/B test of search settings runs through the same pool, reports and posterior. A variant is
  a rated checkpoint's weights with overridden per-side settings (`sims`, `root_samples`, `tactics`,
  `solver_*`), league id `<checkpoint>@<name>`:

  ```text
  python python/dense_eval.py variant --run runs/dense-v1 --checkpoint main/032500 --name solver --set solver_root_nodes=135 --set solver_finalists=2 --set solver_finalist_nodes=135 --set solver_threat_nodes=135
  ```

  The running evaluator picks it up once no checkpoint waits and decides it against its checkpoint: 'better'
  once P(variant - checkpoint > `sprt_elo0`) reaches `promote_confidence`, 'worse' once it falls to 1 -
  `promote_confidence`, 'max-games' at `sprt_max_games`. Variants are rated in the league and shown in the
  checkpoint history, but they never become champion and never reach the actors. `--checkpoint champion` registers against whichever checkpoint is champion when the comparison starts; with `rebase_on_promotion` (default on) a variant registered against the champion also follows a new champion until it starts.
- **Opening books** (`dense_openings.py`). Every pairing, Seal anchors included, draws its colour-swapped openings from
  the book of `opening_suite`. Each completed pair is recorded on every node its opening passed through, so the
  statistics of a node cover its whole subtree. A book is a DAG of symmetry-reduced positions, and a frozen suite
  is a book file too: `openings/standard-v1.json` is the old evaluation suite with its original distribution.
  With `--eval-opening-suite book`, the live book `openings.json` holds `book_size` (512) settled openings of 3 to
  `book_plies` (5) placements. Each is the shortest plausible, unused prefix of a line sampled at
  `book_temperature` from the visit counts of `book_sims` (16) searches. Generation skips prefixes of active openings;
  when a child extends an opening, the parent retires as `nested` at refresh. At every champion change and every
  `book_refresh_hours` (6) the champion re-scores the openings. It retires the implausible ones (policy
  probability below `book_min_prob`) and the skewed ones (first-player skew interval beyond ±`book_max_skew`
  Elo after `book_min_games` pairs); a skewed opening makes way for a child. It also retires an opening with at least
  `book_short_min_games` (6) decisive games when its first-player win z-score reaches `book_short_skew_z` (2.5) and its
  mean game length is below the `book_short_quantile` (0.25) of played openings' mean lengths. The champion also challenges
  `book_revisit_fraction` of the settled openings with alternatives at the same depth, and the more balanced one
  stays. Retired openings are replaced until the target is met again. On start a book counts every report pair it
  has not counted yet (the first time, a live book imports the other suites' reports too), and a refresh adopts
  the plausible, balanced positions among them. A report is reused
  only under the book state it was played in: a refresh that changes the openings starts comparisons afresh.
  `league.json` `openings` holds P1/P2 results overall and per player, and each book's counts, depths and skew
  histogram. `/api/openings` serves the DAG with its statistics. `python python/dense_openings.py refresh|stats|prune
  --run R` refreshes, inspects or prunes a book; run the writing commands while the evaluator is stopped.
