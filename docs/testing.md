# Testing

Tests check what the system answers or can do from the outside: legal turns, verified proofs, rows the learner can
read, a clock that ends a turn. They do not pin implementation details such as node counts, call orders, hashes of
outputs or bit-identical results against an older build. A test that fails because a legitimate change moved such a
detail gets rewritten as a capability check or deleted.

## Two tiers

The fast tier holds every contract and invariant test. It runs on every pull request and before every push.

The slow tier holds tests whose cost is the point: real games, real solver searches, ONNX exports, Seal matches,
spawned worker processes and load reproduction. A slow test carries the `@slow` decorator from `tests/__init__.py`
on its class or method and runs only with `HEXO_SLOW=1`. When a capability can only be shown by a real search, the
search goes in the slow tier and a cheap fast-tier test checks the surrounding contract.

## Commands

Build the native libraries in the checkout first (`cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release`,
`cmake --build build -j 4`, `python tools/build_tactical.py`). From the checkout root, with `python/` and the root on
`PYTHONPATH` (or the package installed with `pip install -e .`):

```sh
python -m tests                       # the fast tier; run this before every push
HEXO_SLOW=1 python -m tests           # every test
python -m tests 'test_play' 'test_dense*'   # some modules; a pattern starting with ! excludes
python -m unittest tests.test_play.Matches -v   # one class while working on it
```

`python -m tests` runs each module in its own process, four at a time by default (`-j N`), largest first, and
prints one line per module. It exits with status 1 and prints the failing modules' output when anything fails.
Without node, onnxruntime, Seal or the optional engines the tests that need them skip.

The native tests outside Python:

```sh
c++ -std=c++20 -O2 tests/tt_injection.cpp -o build/tt_injection && build/tt_injection
c++ -std=c++20 -O2 tests/gumbel.cpp -o build/gumbel_test && build/gumbel_test
cargo test --release --manifest-path tools/tactical/Cargo.toml
```

## CI

| Workflow | Job | Pull requests | Manual dispatch |
| --- | --- | --- | --- |
| Native rules (`native.yml`) | `rules` | `tt_injection.cpp`, `gumbel.cpp`, the Rust solver tests, and `NativeRules` plus `test_leaf_tactics` against the profile-guided build | same |
| Contracts (`contracts.yml`) | `contracts` | fast tier of every Python module except `test_web_*` | every test of those modules |
| Browser engine (`web.yml`) | `parity` | fast tier of `test_web_*`, with Seal built | every test of those modules |

No Python test runs in two jobs. The `rules` job runs two modules again, against a different build of the library.

## Times

Measured on 2026-10-07 on this repository's Windows desktop, at BelowNormal priority with OMP_NUM_THREADS=2 and CUDA
hidden, while other work shared the machine. CI times come from GitHub's ubuntu-latest runners.

| | Before (80018fb) | After |
| --- | ---: | ---: |
| Test methods | 1331 in 42 modules | 1263 in 32 modules, 21 of them slow |
| Whole suite, one module at a time | 1026 s, 7 tests failing in 4 modules no CI job ran | 865 s, all passing |
| Fast tier, `python -m tests` | not separable | 90 to 94 s |
| Slow tier alone, one at a time | | 510 s, 367 s of it one browser solve |
| CI `rules` job | 2:45 to 3:50 | 2:04 |
| CI `contracts` job | 3:41 to 4:36 (3 modules) | 2:47 to 4:35 (24 modules) |
| CI `parity` job | 8:03 to 14:24 (test step 696 s) | 2:18 to 3:25 (test step 34 to 48 s) |

The fast tier's wall time is test_dense's own (75 to 82 s). The other modules finish beside it.

## Inventory

One row per test class. The columns:
- "Fast s" and "Slow s": the seconds the class's fast-tier and slow-tier tests took, one module at a time, on the
  machine above. 0.0 means under 50 ms, skipped here (Seal, CUDA and the optional engines are not built in a fresh
  worktree), or work done in `setUpClass`, which no row counts (the browser `Resolver` and `Loading` classes run
  their node process there).
- "Overlaps": another test that checks the same capability.
- "Decision": what happened to the class in #457, #458 and #459.

The modules deleted in #457 tested legacy code that no live module imports. They covered relational play and
training, the GPU NNUE, GPU self-play rules, Q warm start, the old trainer's warm start, reanalysis, the human
corpus, klent fitting, the curriculum, Strix root search, and the old evaluator's freeze and trace checks: 74 tests.
The code is still in `python/legacy/`. `tests/benchmark_policy.py`, `benchmark_relational.py` and
`benchmark_value.py` are measurement scripts for the same code, not tests.

### Rules, solver and search

| Class | Tests | What it protects | Fast s | Slow s | What costs time | Overlaps | Decision |
| --- | ---: | --- | ---: | ---: | --- | --- | --- |
| `test_dense_solver.NativeEntries` | 5 | Native hold, exact marks and priority entries in the search tree | 0.0 | 0.0 | - | gumbel.cpp halving schedule | keep |
| `test_dense_solver.DefenceSearch` | 8 (1 slow) | Defence candidates are admitted at both placements, queries are capped, the bonus is cleared for reuse | 0.5 | 4.3 | 27000-node threat queries | Determinism | keep; query-count pins rewritten (#458); seeded shard repeat slow |
| `test_dense_solver.InjectionPoints` | 4 | Root proofs, finalist pruning, threat ordering and per-side budgets inside real games | 0.6 | 0.0 | tiny-model games | - | keep; pruned-count pin rewritten (#458) |
| `test_dense_solver.LostRoots` | 2 | A proven-lost side still plays a legal move | 0.0 | 0.0 | tiny-model games | neural_search exact loss | keep |
| `test_dense_solver.GraphSearch` | 6 | Graph and Q-floor options reach actors and the evaluator | 0.7 | 0.0 | tiny-model games | - | keep |
| `test_dense_solver.Proofs` | 5 | The Proof walker: action, reply, path labels, followed proofs | 0.7 | 0.0 | 5000-node solves | - | keep |
| `test_dense_solver.Adjudication` | 7 | Proven adjudication, line rows and certificate replay | 1.0 | 0.0 | 13 seeded solves | - | keep |
| `test_dense_solver.Scheduler` | 16 (1 slow) | Nonblocking fixed budgets, allocation from measured leads, late verdicts, pool abort | 2.2 | 8.9 | match batches at 2048 nodes | tactical_proof Gate | keep; seeded match equality between modes slow |
| `test_dense_solver.Determinism` | 1 (slow) | Seeded self-play shards repeat across runs and solver backends | 0.0 | 6.1 | three self-play runs | actor_determinism | slow |
| `test_dense_solver.ProvenTargets` | 5 | Proven rows get the right labels and learner targets | 0.1 | 0.0 | - | - | keep |
| `test_dense_solver.Protocol` | 3 | Solver settings validation | 0.0 | 0.0 | - | - | keep |
| `test_engine.NativeRules` | 11 | Native rules match the Python reference: radius, turn phase, axes, first-stone wins, overlines, make/unmake, TT hints, root admission | 1.6 | 0.0 | 6 axes of short searches | tt_injection.cpp (line storage, hints) | keep |
| `test_forcing_material.ForcingMaterial` | 4 | Live-window counts and the gate level the solver budget follows | 0.0 | 0.0 | - | - | keep |
| `test_isolated_tactics.Isolation` | 19 | Solver child process: reuse, hard kill at the deadline, abort, cancel, memory cap, size limits | 6.0 | 0.0 | a Python child per test, 0.1 to 0.3 s sleeps, 512 MB allocations | tactical_proof isolated cancel | keep |
| `test_leaf_tactics.LeafTactics` | 5 | Depth-1 search finds forced wins and avoids forced losses on development positions, checked by the reference | 3.0 | 0.0 | 1 s searches | tt_injection.cpp development positions | keep |
| `test_neural_search.NeuralTree` | 40 | Native Gumbel tree: noise, Q floor, choice, exact wins and losses, proof install, reuse, cache keys, batching, deadlines, input checks | 1.4 | 0.0 | 8 to 128 uniform-evaluator simulations | gumbel.cpp (shared turn node, shortest win, round barrier) | keep; legal-count pins rewritten (#458) |
| `test_neural_search.SharedGraph` | 25 | Shared game graph across views: archive, reconvergence, credits, eviction, proofs reaching earlier roots, PV recheck | 6.7 | 0.0 | one 2048-simulation search | gumbel.cpp archive and eviction blocks | keep |
| `test_neural_search.HybridScheduler` | 42 | SearchPool and InferenceService: batching, coalescing, retargeting, cancellation, watermarks, fenced teardown, GPU launcher overlap | 6.1 | 0.0 | service threads with condition waits | - | keep |
| `test_neural_search.NativeProofs` | 36 (1 slow) | Proof loop over the real solver: admission, scope closing, facts, shared workers, owner budget, cancellation, verified certificates | 7.1 | 3.1 | real solver, condition waits | Rust token tests | keep; the 1.5 s owner-budget load comparison is slow; slice pin rewritten (#458) |
| `test_proof.ForcingProof` | 6 | Python proof.solve/verify: certificates verify, mutations are rejected, timeouts stop | 0.1 | 0.0 | 1 s deadlines | Rust checker tests | keep |
| `test_tactical_proof.NativeStrategy` | 32 (1 slow) | Rust solver through ctypes: replay evidence, stamps, worker isolation, cancellation, certificate checks, shortest proofs, deadlines | 8.2 | 7.3 | 5000 to 1M node solves | Rust lib.rs tests; free-filler enumeration also in check.rs | keep; free-second filler enumeration slow; shortest and cap pins rewritten (#458) |
| `test_tactical_proof.SlowStrategy` | 1 (slow) | An unstoppable shape covers a complete quiet turn | 0.0 | 42.4 | a 40 s solve | - | slow |
| `test_tactical_proof.NodeBudget` | 7 | Budgets bound work, resume reuses nodes, the same budget gives the same result in fresh processes | 4.3 | 0.0 | subprocess pair at 540 nodes | Rust resume tests | keep; exact node spend rewritten to at most (#458) |
| `test_tactical_proof.FlippedTurnThreats` | 4 | Opponent-attacker proofs are proved and checked by both checkers | 0.4 | 0.0 | an isolated child | - | keep |
| `test_tactical_proof.Gate` | 4 | The solver budget follows the attacker's forcing material within floor and caps | 0.2 | 0.0 | an isolated child | dense_solver Scheduler allocation | keep; budget pins rewritten to gated_nodes (#458) |
| `test_tactical_proof.IndependentCheckerBounds` | 1 | The independent checker rejects negative indices and malformed DAGs | 0.0 | 0.0 | - | - | keep |

### Dense actor, learner and evaluator

| Class | Tests | What it protects | Fast s | Slow s | What costs time | Overlaps | Decision |
| --- | ---: | --- | ---: | ---: | --- | --- | --- |
| `test_actor_determinism.ActorDeterminismTests` | 1 | The same seed and budget play the same actor and evaluator games | 1.2 | 0.0 | two short tiny-model runs | dense_solver Determinism | rewritten (#457): was a recorded-fixture comparison |
| `test_dense.HexcropTests` | 13 | Crop encoding: planes, actions and cells match the engine's legal moves; native, Python and packed paths agree | 3.4 | 0.0 | about 100 replayed positions, 12 symmetries | - | keep |
| `test_dense.HexNetTests` | 20 | HexNet forward, losses, norms, line convolutions and save/load match reference maths; old checkpoints load | 1.7 | 0.0 | pure-Python line reference | - | keep |
| `test_dense.FusedCudaTests` | 8 | Fused CUDA kernels and the actor graph runner match the reference within bf16 tolerance | 0.0 | 0.0 | GPU only (HEXO_TEST_CUDA=1) | - | keep, explicit GPU run |
| `test_dense.DenseConfigTests` | 20 | Settings parse, override, validate and reach actor workers; old configs resume with the same defaults; a run is created from the command line | 1.0 | 0.0 | one subprocess, one HTTP server | - | keep; the learner speed tile is checked in page source |
| `test_dense.DenseDataTests` | 32 (1 slow) | Shards write, read and reject tampering; value, outcome, calibration and policy targets; replay window and regret priority; render workers | 11.5 | 4.4 | spawned render workers | - | keep; spawned-children check slow |
| `test_dense.WindowMemoryTests` | 8 | Replay window index and policy memory-map cache are correct, shared between processes and pruned | 1.2 | 0.0 | synthetic runs | - | keep |
| `test_dense.CheapRowTests` | 10 | cheap_row_fraction drops only ordinary cheap rows, consistently in learner, render workers and bench tool | 5.0 | 0.0 | two exports | - | keep |
| `test_dense.DenseBootstrapTests` | 1 | An old corpus converts into loadable shards; a tampered one is refused | 0.1 | 0.0 | - | - | keep |
| `test_dense.MaskedFutureLearnerTests` | 2 | Switching future_target on resume keeps step, optimizer and model state | 2.0 | 0.0 | three exports per test | LearnerPipelineTests | keep |
| `test_dense.ValidationSourceTests` | 27 (1 slow) | Per-source validation subsets; export reports per-source metrics, curves and regret; EMA recalibration; VRAM cap | 6.2 | 6.5 | Learner.export | - | keep; full-count EMA recalibration slow |
| `test_dense.EvaluatorSearchTests` | 12 | Match games finish legally and propagate failures; evaluator logits equal a model forward | 0.5 | 0.0 | tiny-model searches | - | keep |
| `test_dense.EngineTests` | 67 | Actor engines produce legal, labelled rows; pause, fence, retire and failure without leaks; solver proofs label rows | 14.1 | 0.0 | hybrid self-play with a tiny model | - | keep |
| `test_dense.YieldTests` | 17 | The actor pause gate follows learner heartbeats on a fake clock; phase tokens are acknowledged after a drain | 0.1 | 0.0 | - | - | keep |
| `test_dense.LearnerPipelineTests` | 4 | Muon/AdamW split covers every parameter; both train on CPU and reset on a kind change | 0.7 | 0.0 | two exports | MaskedFutureLearnerTests | keep |
| `test_dense.PhaseTests` | 13 | Backlog and pacing base; phased training; the learner waits for every actor's acknowledgement | 2.0 | 0.0 | exports, dense_learn.main in process | CheapRowTests rebase | keep |
| `test_dense.ActorModelTests` | 6 | The model pointer resolves; workers switch checkpoints between games; hybrid workers publish rows the learner reads | 3.5 | 0.0 | hybrid workers, 6 games | EngineTests | keep |
| `test_dense.PacerTests` | 3 | Evaluator busy and share pacing on fake clocks | 0.0 | 0.0 | - | - | keep |
| `test_dense.PosteriorTests` | 9 | Rating posterior: direct and pooled evidence, sweeps, value of information | 0.0 | 0.0 | 128-sample posteriors | - | keep |
| `test_dense.OpponentSchedulerTests` | 8 | Report reload, payoff matrix, panel and veto, PFSP weights, opponent plies masked from training | 0.6 | 0.0 | one 4-game engine run | - | keep |
| `test_dense.EvaluatorLoopTests` | 96 | League evaluator: SPRT and posterior trials, anchors, variants, panels, fills, rematches, supersession, restarts, reports | 20.0 | 0.0 | tiny-model games (Seal ones need the library) | openings EvaluatorBookTests | keep |
| `test_dense.SlowDenseTests` | 2 (slow) | Window memory per row stays under budget; the learner CLI heartbeat reports its target | 0.0 | 26.9 | a 50,000-row window; two subprocesses | - | slow |
| `test_dense.DenseTimedWorker` | 2 (1 slow) | Timed hybrid turns are legal and complete, report proofs, honour deadlines and swap models | 0.0 | 7.6 | 64-simulation turns with 30 s clocks | - | keep; the clocked CPU turn is slow |
| `test_dense.DenseBrowser` | 8 | play.evaluate returns complete legal turns, keeps proven second stones, reloads changed weights, cancels promptly | 0.7 | 0.0 | tiny model, one solve | - | keep |
| `test_dense_solve.PassTests` | 10 (1 slow) | Offline proof pass: windows, lookback, buffer entries, gate, verification events | 13.8 | 4.7 | real solves; worker processes | - | keep; worker processes slow |
| `test_dense_solve.WorkerCountTests` | 3 | Book prefixes are excluded; worker count follows the learner phase | 0.1 | 0.0 | - | - | keep |
| `test_dense_solve.RestartBufferTests` | 6 | The priority restart buffer | 0.0 | 0.0 | - | - | keep |
| `test_dense_solve.RestartActorTests` | 11 | Restart draws and restart games in the actor | 1.2 | 0.0 | tiny-model games | - | keep |
| `test_dense_solve.ProvenLabelTests` | 11 | Proof sidecar labels reach the replay, validation and learner targets | 6.9 | 0.0 | Learner validation | - | keep |
| `test_dense_solve.DashboardTests` | 1 | The actor tile's restart and verification fields | 0.0 | 0.0 | - | - | keep (two fields checked in training.html source) |

### Play, API, dashboard and books

| Class | Tests | What it protects | Fast s | Slow s | What costs time | Overlaps | Decision |
| --- | ---: | --- | ---: | ---: | --- | --- | --- |
| `test_bubble.CommandTests` | 4 | The match command and services target the run; flags reach GPU services | 0.1 | 0.0 | - | - | keep |
| `test_bubble.ModelTests` | 3 | Install and download place the weights and champion | 0.0 | 0.0 | - | - | keep |
| `test_bubble.MembersTests` | 2 | Process-tree membership | 0.0 | 0.0 | - | - | keep |
| `test_bubble.MatchTests` | 1 | A match process is a member of the tree | 0.0 | 0.0 | - | - | keep |
| `test_bubble.LauncherTests` | 20 | prepare, start, stop, status, locking, cleanup after a failed start | 2.2 | 0.0 | - | - | keep |
| `test_dashboard.GameLengths` | 2 | Game-length histogram windows, start types and endings | 0.5 | 0.0 | - | - | keep |
| `test_dashboard.TacticalResults` | 1 | Tactical results grouped by opening and model | 0.0 | 0.0 | - | - | keep |
| `test_dashboard.ExternalRatings` | 2 | External Elo follows the reference rating and checks saved report hashes | 0.1 | 0.0 | - | test_dense external rating | keep |
| `test_dashboard.EvaluationBinding` | 1 | An evaluation binds to the latest model.nnue on the legacy run page | 0.0 | 0.0 | - | - | keep |
| `test_dashboard.OpeningBookPages` | 12 | Book API: stats, prefix matching, paging, sort and filters, cache invalidation, DAG, validation | 6.6 | 0.0 | HTTP server per test | - | keep |
| `test_engine_setup.Recipes` | 18 | One-click Six, Strix, Shrimp and Seal setups from recorded downloads: hashes, layout, registry, fallbacks, locking | 1.0 | 0.0 | a subprocess | play Registry | keep |
| `test_engine_setup.Pieces` | 4 | Six member, unsafe archive paths, source trees, wheel tags, job progress | 0.0 | 0.0 | - | - | keep |
| `test_engine_setup.Http` | 1 | /setup lists and starts setups and rejects unknown engines | 0.5 | 0.0 | HTTP server | - | keep |
| `test_evaluation.PairedEvaluation` | 1 | Paired match statistics rate only complete colour pairs | 0.0 | 0.0 | - | - | keep; the legacy evaluate checks were deleted (#457) |
| `test_notation_api.OfficialNotation` | 3 | htttx parse and dump round trip, strict turns, bounds, first-stone terminal | 0.0 | 0.0 | - | - | keep |
| `test_notation_api.OfficialAPI` | 7 | Stateless HTTP API: board reconstruction, 400/408/409/503, request deadlines | 1.9 | 0.0 | HTTP server, 1 ms searches | - | keep |
| `test_notation_api.TimedClocks` | 15 | Side settings, paired openings, clock arithmetic, the hybrid controller's allowance, HTTP opponent failures | 2.4 | 0.0 | hybrid timed turn; a handler that sleeps 1 s | six_engine opponent failure | keep |
| `test_notation_api.TimedAPI` | 8 | Match REST and websocket: clock stream, interrupt, shutdown, engine release, resume | 1.8 | 0.0 | aiohttp server | - | keep; the shutdown test found a session engine left open, fixed in #460 and #462 |
| `test_notation_api.ArenaDrip` | 6 | Arena bot: gateway retries, cleanup, socket redial, replay without duplicate moves | 4.7 | 0.0 | real searches, a 1.2 s retry | - | keep |
| `test_openings.CanonicalTests` | 5 | Canonical keys over symmetries and turn orders; parents; tempered weights; reach | 0.1 | 0.0 | - | - | keep |
| `test_openings.SkewTests` | 3 | Skew estimate and interval; reconcile fills lengths | 0.0 | 0.0 | - | - | keep |
| `test_openings.FilterTests` | 3 | Retirement rules by value, skew, probability | 0.0 | 0.0 | - | - | keep |
| `test_openings.RefreshTests` | 20 | Book refresh, generation, retirement, import and adoption, draws, prune | 1.9 | 0.0 | - | - | keep |
| `test_openings.GenerationTests` | 3 | Reach sums every play order; refresh and continuations with a tiny model | 2.3 | 0.0 | tiny-model searches | - | keep |
| `test_openings.FrozenTests` | 2 | standard-v1 is frozen and matches train.opening_for | 0.8 | 0.0 | 4770 draws | - | keep |
| `test_openings.SummaryTests` | 2 | Per-colour statistics; reading books does not load torch | 0.3 | 0.0 | a subprocess | - | keep |
| `test_openings.SettingsTests` | 1 | Book command-line flags parse and validate | 0.0 | 0.0 | - | - | keep |
| `test_openings.CommandLineTests` | 2 | dense_openings refresh, stats and prune; report stamps | 0.9 | 0.0 | - | - | keep |
| `test_openings.EvaluatorBookTests` | 13 (1 slow) | The evaluator uses the book; archive, calibration and anchors under the evaluator | 7.7 | 0.0 | real Evaluator on tiny checkpoints | test_dense EvaluatorLoopTests | keep; the Seal anchor test is slow and needs Seal (#457) |
| `test_play.Store` | 8 | The evaluation store appends, reloads, prefers the deepest record, survives a torn line, evicts past its limit | 0.3 | 0.0 | - | - | keep |
| `test_play.Review` | 6 | Turn labels and per-stone grading | 0.0 | 0.0 | - | - | keep |
| `test_play.Jobs` | 40 | The Session worker: replies, cancel, pause, analysis reuse and refresh, deepening, rescans, undo, failures, budgets | 5.0 | 0.0 | spawned engine processes, short sleeps | - | keep |
| `test_play.GameGraphs` | 5 | A seat's game graph reuses visits across turns; proofs propagate on the graph | 3.2 | 0.0 | 10000-node solves | - | keep |
| `test_play.GraphAnalysis` | 2 (slow) | Analysis with the real network on one graph lowers the right stones | 0.0 | 0.0 | real network, 512 simulations; needs the run's champion | - | slow (#459) |
| `test_play.Http` | 5 | Engine bundle with COOP/COEP, responsive page, match API, import/export, origin and Host checks | 4.7 | 0.0 | HTTP server, one fake match | SiteImport, Formats | keep |
| `test_play.SiteImport` | 3 | hexo.did.science, mineking and tyto links become histories | 0.0 | 0.0 | - | - | keep |
| `test_play.Formats` | 5 | Rectilinear, ring and tyto round trips and export spans | 0.1 | 0.0 | - | - | keep |
| `test_play.Matches` | 23 | Batch matches: colour swap, pause, failure, book openings, resume, solver build checks, saved games, clocks | 25.7 | 0.0 | fake-engine matches with the 2 s pause between games | - | keep |
| `test_play.FreeplayClock` | 7 | Fischer clock, pause, a failed engine stops the clock, timed engines kept | 1.3 | 0.0 | 0.1 to 0.5 s sleeps | - | keep |
| `test_play.Proofs` | 10 | The proof table: bound padding, shorter child replaces the root, defender lines, turn orders | 0.1 | 0.0 | - | - | keep |
| `test_play.TurnTrees` | 26 (1 slow) | play.evaluate and TurnSearch: the second stone in the first stone's tree, proofs settle edges, leaf allowance, cancellation | 12.5 | 9.6 | tiny-model forwards, solver leaves | web Bundle solver leaves | keep; losing half-turn leaf proof slow; exact moves rewritten (#459) |
| `test_play.GameProofs` | 4 | A proof at ply 80 carries back to earlier plies, survives a reload and works inside a line | 1.7 | 0.0 | 32768-node solves | web BrowserProofs | keep |
| `test_play.PrincipalVariation` | 6 | A principal variation from a certificate; solve asks for the shortest win | 0.2 | 0.0 | 32768-node solve | tactical shortest | keep; exact line rewritten (#459) |
| `test_play.Registry` | 4 | Six backend choice, model-folder and runs scan, Q-range floor | 0.4 | 0.0 | - | engine_setup Recipes | keep |

### Engines and adapters

| Class | Tests | What it protects | Fast s | Slow s | What costs time | Overlaps | Decision |
| --- | ---: | --- | ---: | ---: | --- | --- | --- |
| `test_nnue.NNUETest` | 6 | An exported NNUE gives the native engine the same evaluations | 0.2 | 0.0 | - | - | keep |
| `test_relational.RelationalTests` | 7 | Relational encoder: symmetry transforms and native encoder parity | 2.4 | 0.0 | - | - | keep; moved from its own workflow into Contracts |
| `test_seal_current.SealCurrentContract` | 10 | Seal's current adapter: build provenance and turn validation | 0.1 | 0.0 | one test needs the built adapter | - | keep |
| `test_six_engine.SixProtocolTests` | 15 | Six protocol server and client: stop during search, clocks, handshake, restarts, cancel, coordinates, match side | 4.2 | 0.0 | spawned fake engines, a 2 s sleep | notation_api opponent failure | keep; stale search result fixed (#457) |
| `test_strix_learned.TurnValidation` | 3 | Strix turn validation and the checkpoint hash check before launch | 0.0 | 0.0 | - | strix_reference SequentialReplay | keep |
| `test_strix_learned.Frame` | 1 | Stones and moves mirror into Strix's frame | 0.0 | 0.0 | - | - | keep |
| `test_strix_learned.LearnedProcess` | 3 | The learned Strix process plays two moves then one and is reused | 0.0 | 0.0 | needs HEXO_STRIX_PUBLIC_MODEL and the built engine | - | keep, optional engine |
| `test_strix_reference.SequentialReplay` | 3 | Principal-variation validation rules | 0.0 | 0.0 | - | strix_learned TurnValidation | keep |
| `test_strix_reference.CorpusProvenance` | 1 | The Strix corpus tool reports the executable it ran | 0.0 | 0.0 | - | - | keep |
| `test_strix_reference.NativeReference` | 9 | The Strix reference process: timeouts and recovery | 0.0 | 0.0 | needs the built Strix reference | - | keep, optional engine |

### Browser

| Class | Tests | What it protects | Fast s | Slow s | What costs time | Overlaps | Decision |
| --- | ---: | --- | ---: | ---: | --- | --- | --- |
| `test_web_assets.Resolver` | 20 | assets.mjs: local first, then the public site, checked hashes, cache, resume, timeouts | 0.0 | 0.0 | one node process | - | keep |
| `test_web_assets.Loaders` | 1 | Each engine loader fetches through assets.mjs, so downloads get hash checks, the cache and the site fallback | 0.0 | 0.0 | - | - | keep; reads module source, no loader behaviour test exists yet |
| `test_web_drip.WebDripParity` | 2 | wasm Drip plays the library's turn at a fixed depth and keeps a timed turn legal | 4.0 | 0.0 | about 150 searches each side | - | keep (now in CI) |
| `test_web_engine.Export` | 2 (slow) | The ONNX export matches PyTorch; the rectangular rewrite runs | 0.0 | 7.9 | ONNX export | - | slow |
| `test_web_engine.PlayPage` | 10 | The opening book follows the seats until the user touches the switch; the human seat label | 1.0 | 0.0 | node per test | - | keep; the seat label is checked in page source |
| `test_web_engine.BoardPerspective` | 1 | The 12 board views are the hex symmetries and invert | 0.1 | 0.0 | - | - | keep |
| `test_web_engine.Overlay` | 2 | Candidate ranking and the proven-line display | 0.1 | 0.0 | - | - | keep |
| `test_web_engine.BrowserProofs` | 16 | The browser proof table, storage and replay match play.Proofs and verified certificates | 4.1 | 0.0 | native solves | web_tactical parity | keep |
| `test_web_engine.Loading` | 16 | Load stages, watchdogs and the GPU, WASM and one-thread fallbacks | 0.0 | 0.0 | fake workers | - | keep |
| `test_web_engine.Bundle` | 64 (1 slow) | The browser bundle: native owner, proofs, sessions, matches, clocks and search parity with the native library | 44.2 | 367.4 | node per test, wasm searches | play TurnTrees, web_tactical | keep; wasm solver-leaves test slow; parity uses a small network, one position of each kind in the fast tier (#457) |
| `test_web_seal.WebSealParity` | 3 | Both Seal backends play legal turns, take the immediate win and enforce the range | 0.0 | 0.0 | needs Seal; 11 positions at 1 s | - | keep (CI builds Seal) |
| `test_web_shrimp.Export` | 2 (slow) | The Shrimp ONNX graph matches the evaluator forward; profile constants | 0.0 | 2.9 | ONNX export | - | slow |
| `test_web_shrimp.Presets` | 1 | Browser Shrimp presets equal engines.json | 0.1 | 0.0 | - | - | keep |
| `test_web_shrimp.DriverParity` | 3 | The exported graph plays the driver's turns | 0.0 | 0.0 | needs models/shrimp | - | keep, optional engine |
| `test_web_six.Browser` | 1 | Six in the browser stops a turn early | 0.0 | 0.0 | needs ORT and the Six networks | - | keep, optional engine |
| `test_web_six.Parity` | 2 | Browser Six plays sixengine's moves | 0.0 | 0.0 | needs models/six | - | keep, optional engine |
| `test_web_strix.WebStrixParity` | 2 | Browser Strix plays the adapter's moves up to symmetry | 0.0 | 0.0 | needs the Strix weights and binary | - | keep, optional engine |
| `test_web_tactical.WebTacticalParity` | 8 | The wasm solver's fields equal native, certificates verify, resume works | 5.2 | 0.0 | native solves up to 100k nodes | web BrowserProofs | keep |

## Known flaky tests

Running every module in CI from #457 on surfaced timing races that the old three-module jobs never ran. These are
fixed:
- six_engine's clock test sent `quit` before the search answered.
- play's review dedupe asked twice without holding the fake engine.
- The timed API's bot session could skip its engine close at server shutdown (#460, #462).

These failed once each and are not explained yet:
- `test_play.Jobs.test_rescans_cancel_work_for_a_vanished_analysis_model` waited the full 60 s for the old analysis
  job to report `cancelled` (CI run 37663090184).
- `test_neural_search.HybridScheduler.test_replaced_slots_finish_cleanup_and_keep_retirement_bounded` failed once in
  a local parallel fast-tier run. 40 repeats beside three other modules passed.
