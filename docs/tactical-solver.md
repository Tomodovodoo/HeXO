## Native verified tactical strategies

The optional `tactical_proof.NativeTactics` library runs wide Strix IDTT followed by PDS-PN. The solver crates are vendored in `tools/tactical/vendor/hexo-strix` from [SootyOwl/hexo-strix](https://github.com/SootyOwl/hexo-strix) at `5a771e572553a8bd8e010112b2ce65f16e5afa1b` (MIT) and changed locally; the history of that directory lists the changes. IDTT principal variations are only hints. An exact positive requires a complete PDS-PN strategy DAG accepted by a separate raw-coordinate Rust checker. `independent_verify` also rechecks the exported strategy through the independent Python rules implementation.

```sh
python tools/build_tactical.py
python -m unittest tests.test_tactical_proof -v
```

This needs Rust/Cargo supporting edition 2024 and uses locked dependencies. The build manifest binds the native binary to its wrapper sources, the vendored sources and the Cargo lockfile. `NativeTactics().solve(game, nodes=2500)` returns `PROVEN_WIN` with `native_verified=true` and the complete current turn only after verification; unresolved searches return `UNKNOWN`, never a global loss. Partial-turn roots are supported. The exact board, player, remaining placements, fixed rules and verifier scope identify cached facts; neural model evaluations and visit counts are not stored here.

`nodes` is the primary budget: one meter counts every IDTT node, PDS-PN level-1 node and level-2 expansion, charged before the work runs, so a query never exceeds it and its verdict, certificate and `nodes_used` depend only on the position, the attacker, the budgets and the build (`build_hash`). The native checker allows at most `min(200000, max(50000, 8*nodes))` certificate nodes and visits, using the granted budget after any gate adjustment. Candidate proof construction uses the same size limit. This deterministic limit preserves fixed-budget verdicts, including evaluator and determinism-test queries. `ms` is only a safety cap; a query that reaches it returns `UNKNOWN`. Certificate reconstruction and checking after a proof are not metered; they are deterministic and bounded by the certificate size. Proven facts are cached per process under the exact position and budgets, so a cache hit returns what a fresh search would. `solve(game, attacker='opponent')` asks whether the opponent, moving now with a fresh two-placement turn on the current stones, has a forced win; both checkers verify that certificate from the flipped phase, and `threat_cells(certificate)` returns its first attacking turn. `proof_turns` is the most attacker turns on any certificate path, counting the completing turn.

The certificate checker covers both mandatory two-cell defenses and a mandatory single block followed by every legal free second placement. For singleton covers it enumerates the entire radius-eight frontier after the block, including newly legal fillers, and preserves a legal order for each resulting pair. `solve(game, root_moves=[first, second], ...)` can verify a proposed attacker turn by constructing all these defensive branches and proving every continuation with bounded PDS-PN. The default upstream generator still searches fully forcing attacks; entirely quiet defender nodes and unresolved continuations remain `UNKNOWN`. A legal open-three fixture yields a 1,295-node wide-search strategy accepted by both checkers, whereas tight IDTT finds no forcing win. On the development host, a fresh verified proof took about 0.7 seconds and cached re-verification about 18 ms; this is a tactical correctness result, not a strength result.

The search checks its deadline on every PDS-PN level-1 node, before every level-2 expansion and every 16 IDTT nodes, and its position memos are capped. Upstream checked only every 8,192 nodes, so abandoned queries ran 10 to 40 seconds past a 2-second budget. With these checks they stop within about 10 ms, and a 15-second proof search peaks at 66 MB. Certificate reconstruction and verification scale with the proof, not the search. The kernel reads windows from the board's incremental window index, which also keeps each player's live windows (two or more stones, no opponent stone). The generator builds the wide builder list only when some threat cell can pair with a builder. That held at 16 of 20,187 generator calls in a six-turn win from dense-v1 self-play. Unexpanded attacker nodes are seeded with their threat-window count instead of a full move list, which cut generator calls to about 6,400 there. That win's 398-node proof takes 1.4 s instead of 14.7 s. The PDS-PN table is sized at one megabyte per 2,048 budgeted nodes, up to 16 MB, so a small query does not spend milliseconds clearing a table. At 5 ms, 38 of 400 random dense-v1 turn starts had a verified forced win, at a mean cost of 1.2 ms per query. One persistent native worker bounds how long callers wait and rejects overlapping requests as `UNKNOWN`. Reports expose background-worker state, completed-late counts, elapsed worker time and Windows thread CPU time. These are **not equal-compute tournament clocks**. Late or partial certificates never become exact search values. Primary neural MCTS integration remains a separate task. The optional candidate route proved a legal 27-stone fixture with one mandatory block and 745 distinct legal free-placement replies. Both independent checkers accepted its 45,063-node strategy, and deleting one reply invalidated it. That proof took about ten seconds on the development host; it is not a 100 ms tactical result. All legal free placements are covered when a positive is returned, but finding a strategy remains selective and budget-limited.

`IsolatedTactics(package)` runs the same library in a disposable child process. A query returns within its budget plus `grace_ms` (default 100). A child that misses that deadline, or reports abandoned native work, is killed and replaced, so an overrun never delays the next query. Committed memory is capped at `memory_mb` (default 1536) through a Windows job object or `RLIMIT_AS`. Results return the strategy as undecoded JSON text in `certificate_json`, so decoding a multi-megabyte certificate never runs inside the deadline. On the development host the pipe adds about 0.2 ms per query, and a replacement child is ready about 80 ms after a kill. `forcing_material.worth_solving(game)` is a cheap gate built from the engine's six-cell window histogram. It requires a live window holding three of the mover's stones. All 42 solver wins among 1,030 labelled dense-v1 positions passed it, and 59% of turn starts in the same shards do.


## Bounded forcing certificates

`proof.py` searches continuous double-threat attacks and returns `PROVEN_WIN`, `PROVEN_LOSS`, or `UNKNOWN`. Every winning certificate covers all relevant defensive branches. Immediate counterwins take priority; a defense with a free second stone is unsupported and returns unknown. The independent verifier reconstructs rules and covers from raw coordinates. Ordinary search scores are never treated as certificates.

```sh
python python/proof.py --history position.json --ms 100 --output proof-result.json
python python/proof.py --history position.json --verify proof-result.json
python python/proof.py --benchmark
```

A history is a JSON list of `[q, r]` placements in play order. Verification needs a returned certificate; an unknown result has none. The solver is separate from deployed PVS and has no demonstrated Elo benefit. Its deadline is cooperative: a synchronous native candidate call can overrun it, and late results become unknown. In one benchmark a 13-stone forcing win verified in 35 ms, while a 1001-stone sparse board took 236 ms under a requested 100 ms budget.


## Dense actor solver scheduling

`dense_solver.Schedule` decides how the solver queries of the dense searches run (settings `solver_*` in `ActorSettings` and `EvaluationSettings`).

Defence search defaults off. Actors enable it with `--solver-defence --solver-threat-nodes 27000`.
Evaluator loops use
`--eval-solver-defence --eval-solver-threat-nodes 27000`; a match can enable just one side with
`--a-solver-defence --a-solver-threat-nodes 27000`. `solver_defence_candidates` defaults to 8.
Each proven threat supplies candidate complete turns from its placements, replies and line completions.
The solver checks each turn at the threat budget. A completed UNKNOWN keeps the turn as a search candidate,
without claiming safety. Surviving placements enter the root sample set and receive a bonus proportional to
their surviving turns. Matching second stones receive that support on the next placement. Status reports
`defence_queries`, `defence_hits` and `defence_nodes`. See [the measured Seal positions](defence-search.md).

- **Fixed budgets** (`solver_fixed_budgets`, default on): every verdict is awaited where it is needed. Seeded self-play repeats exactly on both backends. Evaluation, engine verification and the determinism test run this way. Evaluation spends each point's flat node budget by default; `--eval-solver-gate-cap-nodes` enables position-only gate scaling with weight 3 and the point budget as its floor.
- **Adaptive budgets** (`--no-solver-fixed-budgets`, actors): a query's budget is `solver_slack_fraction` of its point's measured lead time (20th percentile, minus 50 ms), net of the work already queued, at the measured worker rate, clamped to `[solver_min_nodes, solver_cap_nodes]`. With `solver_gate_weight` the worker scales it by the attacker's forcing material up to `solver_gate_cap_nodes`; positions below the gate's lower level get the floor. Verdicts are polled. A root or finalist verdict may cost at most `solver_overrun_fraction` of the step time in waits; past that its game is skipped for one step while the others build the batch, and then goes on without it. Threat verdicts that miss the next visit are dropped.
- **Following** (`solver_follow`): a side with a proof plays the certificate's turns while the game stays on it and asks no further queries; every proof labels the rows it decides (`proven` +1 for the winner, -1 for the loser), including proofs that arrived too late to decide a move.
- **Deep proofs** (`solver_deep_nodes`, needs following): at each turn start, a `root_moves` query asks whether the side that just moved wins against every defence of its turn. Adaptive deep queries run on one extra idle-priority worker, up to `solver_deep_cap_nodes`.

- **Adjudication** (`adjudicate_proven`): a search whose proof decides the game (a root proof for the side to move, or a root the finalist checks left lost) plays its move and ends the game with reason `proven`. With `proven_line_rows` the certificate's forced line is appended first: attacker stones from the certificate, the defender's first covered reply, one row per placement with the exact value, no policy and `line` true, and no inference. Shard manifests count `proven_games`, `line_rows` and `adjudicated_plies`.
- **Resident tables** (`solver_table_mb`, adaptive budgets only): each worker keeps its solver transposition table and proven-node set per attacker colour across queries.

Foreground workers (`solver_workers`) run at below-normal priority. Actor status `solver` reports queries per second, budget mean and p95, hit rates by point and by budget band, the share of steps with a verdict wait and its mean, the overrun (wait time over step time), lead times, worker rate and utilisation, idle fraction per pool, slack utilisation (busy time over the collect time plus overrun allowance the scheduler targets), and deferred, late, dropped, followed and labelled counts.

Winning certificates also record `proof_action` on actor rows, including late proof labels and forced-line rows.
The proof pass writes an action for each proven ply in its sidecar. Readers accept older rows and sidecars without
actions. Restart actors with their existing solver flags to collect these targets; no new actor flag is needed.

The learner's `--proof-policy-weight` defaults to `0.0`, preserving existing training. A positive weight `w` uses
`(search + w * proof) / (1 + w)` on winning rows with a witness. The proof distribution splits mass equally across
the certificate's two placements at turn start and uses the remaining stone at mid-turn. Other searched moves
keep their mass. Without a search policy, the proof supplies the target with policy loss weight `w`. Proven
losing rows keep their existing policy. For a first trial, restart the learner with `--proof-policy-weight 0.5`.
Fixed validation panels report `<source>_policy_ce_proof` and `<source>_policy_ce_proof_rows` for winning witness
rows with a policy target, using the configured mix.
