# Tactical solver

`tools/tactical` is a Rust solver for forced wins: iterative deepening threat-space search (IDTT) followed by PDS-PN. It started as the solver crates of [SootyOwl/hexo-strix](https://github.com/SootyOwl/hexo-strix) at `5a771e5` (MIT), vendored under `tools/tactical/vendor/hexo-strix` and changed; that directory's history lists the changes.

```sh
python tools/build_tactical.py
python -m unittest tests.test_tactical_proof -v
```

The build needs Rust and Cargo with edition 2024 support and writes the library and a manifest that binds it to its sources under `tools/tactical/target/release/`.

## Contract

`tactical_proof.NativeTactics().solve(game, nodes=2500)` returns `PROVEN_WIN` only with a complete strategy against every defence, verified by a separate Rust checker that replays the certificate from raw coordinates, and `UNKNOWN` otherwise, never a loss. `independent_verify` rechecks a certificate through the Python rules. The certificate covers every defender reply: both mandatory two-cell blocks and a single block followed by every legal free second stone. Only continuous-threat attacks are found; a win that needs a quiet turn stays unknown.

`nodes` is the budget. One meter counts every node of both searches, charged before the work, so the verdict, the certificate and `nodes_used` depend only on the position, the attacker, the budget and the build. `ms` is a safety cap that turns the result into `UNKNOWN`. `solve(game, attacker='opponent')` asks whether the opponent would win with a fresh two-stone turn on the current stones; `threat_cells` returns that certificate's first turn. `solve(game, root_moves=[a, b])` verifies a proposed turn by proving every defence to it. Certificates carry `proof_turns`, the most attacker turns on any path.

`attacker='defender'` starts at the actual defender turn, with its actual one or two stones remaining. A verified `PROVEN_LOSS` proves the opponent wins against every legal defence. The checker enumerates the complete cover set, including free second stones when one block suffices, and checks counterwins first. A quiet unresolved defender root returns `UNKNOWN`.

The Python, isolated-worker and browser wrappers accept `known=[{history, winner, plies}]`, a snapshot of **trusted exact game outcomes** from the graph or saved proof table. `history` identifies all coloured stones and the turn; `plies` is a placement upper bound until that winner completes six. Neural estimates and scoped forcing-search failures are not outcomes. Both PDS-PN levels use matching facts as terminal nodes. A proof of a losing first stone also covers any legal completion of that same losing turn.

A certificate's `exact` leaf names a fact index; optional `after` stones finish the losing defender's turn from that fact. Both raw-board checkers verify the board, mover, remaining placements, winner and continuation. Such a certificate is conditional on its supplied facts. `independent_verify(..., known=...)` rejects it without those separately supplied premises. The response's `dependencies` lists the indices and complete outcomes used. An imported arbitrary winner claim must never be promoted to a trusted premise. Rules are fixed to six in a row and placement radius eight, and the wrapper reports its checker revision and binary identity.

Queries with `known` disable cross-query resident/result-cache reuse and shortest-proof tightening. Their inputs include the snapshot. Cancellation retires it; a later query cannot inherit its proof numbers. `nodes_fresh` counts search work, while elapsed time includes snapshot validation and certificate checking.

The player passes native graph facts and saved proofs to root queries. After a threat or with existing facts, it tries the defender root. A verified loss settles the graph through `hxg_prove_loss`, propagates to parents and equivalent rule positions, and is saved with its certificate and dependencies. `hxg_facts` exports a bounded read-only snapshot of exact nodes and edges, including edges without allocated children. Both interfaces require the graph owner to serialize access; installation rejects outstanding neural requests. An unproven position whose displayed replies are all refuted shows the node estimate and "replies refuted, not proven".

`IsolatedTactics` runs the library in a disposable child process with a deadline and a memory cap (1536 MB); a child that overruns is killed and replaced, so one slow query never delays the next. `forcing_material.worth_solving(game)` is the cheap gate the searches use before asking: it needs a live window with three of the mover's stones.

Both wrappers expose `cancel()` to stop the current query cooperatively. Search, certificate reconstruction, guided shortening, and both native checkers share its stop token. Request IDs prevent a late cancellation from stopping the next query. Cooperative cancellation retains the isolated child and its resident tables; `abort()` and the hard deadline still replace a child that cannot stop. Rebuild the tactical library to enable cooperative cancellation.

## In the search

`dense_solver.Schedule` runs the solver queries of the actor and evaluator searches (settings `solver_*` in `ActorSettings` and `EvaluationSettings`). Three queries per turn start: a root query (can the mover force a win), a threat query (could the opponent, moving now), and finalist queries on the second stone (does each of the top candidates lose by force). A root proof decides the move; a threat certificate's cells are searched first; a finalist proof marks its candidate lost in the tree the moment the verdict arrives.

- Fixed budgets (`solver_fixed_budgets`, the evaluator): every verdict is awaited, so seeded games repeat exactly. `--eval-solver-gate-cap-nodes` scales the budget with the attacker's forcing material up to that cap.
- Adaptive budgets (`--no-solver-fixed-budgets`, the actors): a query's budget is what the worker can finish in the slack before the GPU needs the answer, clamped to `[solver_min_nodes, solver_cap_nodes]` and scaled by forcing material up to `solver_gate_cap_nodes`. Verdicts are polled; a late one is skipped.
- Following (`solver_follow`): a side with a proof plays the certificate's turns and asks nothing more while the game stays on it. Every proof labels the rows it decides (`proven` +1 or -1, `proof_plies`, `proof_action`).
- Adjudication (`adjudicate_proven`): a proof that decides the game ends it, and with `proven_line_rows` the forced line is appended as rows with exact values and no search policy. The actor's seeded RNG samples covered defender replies, either stone order, and retained attacker alternatives. A chosen alternative must pass the independent raw-board checker at its actual position within 100 ms, including certificate conversion; otherwise the primary strategy is kept. The line then follows the checked alternative's certificate. Winning rows carry that strategy's remaining stones for proof-policy supervision. A longer descendant increases earlier line distance bounds conservatively. The check adds no neural evaluations or solver searches. Analysis and play retain the primary attacker choices and first covered reply.
- Defence search (`solver_defence`, off): for each proven threat, check up to `solver_defence_candidates` complete defending turns at the threat budget and give the survivors a bonus at the root.

Actor status reports the solver's query rate, budgets, hit rates by query and budget band, waits and worker utilisation.

## Offline proof pass

`python python/dense_solve.py --run R` walks finished shards with the spare CPU, solves turn starts at a larger budget than the actors could afford, and writes sidecars: exact labels for proven windows, `deblunder` records where a proven win was later thrown away, and a restart buffer of the positions where the network's value was furthest from the proof. The actors and the learner read those files; nothing in the shards is rewritten. A verification timeout is not a rejection, and a failed shard worker does not stop the pass.
