# HeXO search scheduler: design

Status: design for review, 2026-10-03. Base: main at c0eecc9, PR 328 head 3554bf5 (search/game-graph), the Codex
lane's uncommitted native frontier (codex/220-native-frontier, worktree hexo-native-frontier-20261003), issue 220
comments of 2026-10-03, and the GPU schedule artifacts in artifacts/live/gpu-schedule-20261003 and
artifacts/live/pipeline-profile-20261003.

Every number says where it comes from. "Measured" means a file or a log I read. "Estimate" means arithmetic on
measured numbers, or a default to be tuned.

## 0. Summary

The engine becomes one loop per process. A native graph owner thread holds one persistent search graph per game.
It fills GPU batches from many open leaves across several search views, and it feeds a pool of CPU solver workers
from a priority queue of graph nodes. Results from both come back as messages. Only the owner writes node fields.

The main decisions:

1. One native owner thread per process owns all graphs, views, the next batch and the solver frontier. Python
   configures, runs torch, talks to solver processes and writes rows. Python never touches node fields.
2. Gumbel's root contract stays: Gumbel-top-k sampling, sequential halving with the same visits per survivor per
   phase, the improved policy as target. What changes is the barrier: a view waits at halving phase boundaries,
   not after every visit layer. A 128-simulation search goes from 30 launches to 4.
3. Values flow by the MCGS rule with edge visits, not child visits (KataGo). PR 328 weights edges by child visits;
   this design changes that. Playouts at a deeper root credit the game-line edges above it.
4. Depth comes from line views: the turn-end position of each strong root candidate gets its own root sampling and
   halving (PR 328's principal-variation check, generalised). K = number of candidates with improved-policy share at
   least 0.15, at most 2 in play, 3 in analysis, 0 in actors. Breadth comes from root widening and from prefetch
   rows that fill spare batch slots.
5. The solver pool takes jobs from a frontier queue scored as view weight x reach x decision sensitivity x
   probability of progress / cost, plus aging. Jobs are time slices that return proof and disproof numbers. A
   continued job goes back to the worker that holds its table state.
6. Proofs install on any node through a new native call (`hxg_prove_at`). Proofs always beat neural values, even
   when they arrive late. A node whose query is running stays searchable.
7. Batches are assembled per canvas size, launched on size, oldest-request age, deadline or no useful work, two in
   flight at most. At canvas 32 (93% of measured actor rows) the GPU saturates near 64 rows; 128 rows only pays at
   canvas 24.
8. The host cost per row must fall from about 207 us (measured) to about 100 us for the GPU to be the limit. The
   native feeder alone gave 10% (measured). The rest has to come from encoding off the owner thread, decoding in
   native code, and compact edges.
9. Presets become time per stone for play and analysis. Fixed budgets stay for the evaluator and for reproducible
   analysis. Actors get flags whose defaults keep today's behaviour: equal strength and equal targets within the seed-to-seed spread.
10. A training row's value target is always its own root's value at the end of its own search. Later evidence
    never rewrites it, except an exact proof through the existing label path.

## 1. What the research changes

Only findings that change a decision are listed. Sources are at the end.

| Source | Finding | Decision it drives |
|---|---|---|
| KataGo GraphSearch.md, MCGS (Czech et al.) | Q(n) = (U(n) + sum_a N(n,a) Q(child_a)) / (1 + sum_a N(n,a)) with edge visits N(n,a). Weighting by child visits corrupts the parent, because a shared child gets visits this parent never chose. The update is idempotent, so stale values from other paths are safe. | Section B: edge visits, not child visits. Late results can be applied in any order. |
| mctx, Gumbel MuZero | Sequential halving gives each survivor a fixed number of visits per phase. The guarantee is about the root choice given good Q estimates. mctx batches across environments only, never inside one tree. No published batched Gumbel inside one tree. | Section C: keep per-phase visit counts; batch inside a phase with pending visits counted in N; measure the cost (open question 2). |
| Lc0 search flags | Collisions (a descent reaching a leaf in flight) are capped per batch and cancelled. Cache and terminal hits are backed up at once ("out of order eval"). Spare batch slots are filled with likely future positions ("max-prefetch"). Smart pruning stops visiting a root move that cannot catch the leader in the remaining budget. | Section D: collision cap, immediate installs, prefetch fill. Section G: early stop. |
| Stockfish search.cpp, timeman.cpp, tt.h | Time = optimum x falling-eval factor x best-move stability x best-move effort. The TT replaces low-depth, old-generation entries first. | Section G: time multipliers from best-move changes, falling root Q and the best move's visit share. Section B: eviction by age then visits. |
| KataGo playout cap randomisation | Only full searches write policy targets. | Section H: line views and prefetch never write policy targets by default; actors keep full and cheap searches. |
| SPDFPN (Pawlewicz, Hayward) | One owner picks solver jobs centrally, each job has a work cap, jobs resume from a shared table, and running jobs are marked virtually so other workers go elsewhere. Parallel efficiency 0.8 on 4 threads. | Section E: central frontier, sliced jobs, resume, soft virtual marks. |
| PDS-PN (Winands et al.) | After each second-level search only its root goes into the table; the rest is discarded. | Section E: resident tables help mostly at the first level, so continuation gain must be measured, not assumed (open question 5). |
| DFPN-E (Kishimoto et al.), FDFPN-CNN (Gao et al.) | Policy-derived edge costs or policy ordering cut solver work by 36 to 47% (Hex) and 3.3x (synthesis planning). | Section K: a later stage may pass network priors to the solver. Not in the first delivery. |
| PN-MCTS (Doe, Winands et al.) | pn/dn ranks added to selection win 93 to 96% in Lines of Action, 66% in high-branching games, at 72 to 90% of the simulation rate. | Section K: a pn-rank selection bonus is an open question, not a default. |
| MoHex / Benzene | Solver and MCTS run in parallel but race; solver results do not enter the tree. | Our solver writes into the graph. That is the gap MoHex left. |
| Rapfi | VCF runs as the quiescence search with a shared TT, inline at every leaf, because its evaluation is CPU-cheap (Mixnet, 146 kFLOPs). Its MCTS mode uses a sharded transposition table with mate bounds. | With a GPU net the leaf solver cannot be inline; it belongs on CPU workers whose results land as bounds on nodes. |
| Six (CixMango/Six, vendored in tools/six) | PUCT MCTS, batches of 32 with virtual loss, top-40 children per expansion, a 64-node 2-turn threat solver at new leaves, a 20,000-node root solver capped at 25% of the move time, an expansion cache keyed by position (43% hits). "Positions" counts root visit increments including cache hits and terminal backups. Measured here: 3.3k to 7.3k visits/s, 13k to 17k network rows per 30k visits, 14 to 18 rows per batch (shared GPU). | Section D: Six's rate is the speed target. Section E: tiny leaf budgets plus a large root budget is the proven split. |

Two corrections to the inputs file. Six is not Tyto's engine: hexo.tyto.cc hosts Strix; Six is CixMango/Six. Six is
not alpha-beta in play: its network player is PUCT MCTS (search.cpp has a negamax searcher, but web_bot.cpp uses
`Mcts`). Six's 5.9M-parameter net is about a 20-block, 128-channel ResNet on a 25x25 crop.

Why about 30 Six positions per Bubble simulation at equal strength: not settled. Three measured contributors are
known. Six counts cache hits (43% of visits) and terminal backups as positions. A Bubble expansion runs the
one-turn tactics oracle over every legal move, which settles hundreds of replies per network call. Gumbel planning
is designed to be efficient at small budgets. Their shares are an open question; the design does not depend on
them.

## 2. Audit: where the inputs disagree, and what this design decides

| Topic | phase2-spec | Sol | PR 328 | Lane feeder | Decision and reason |
|---|---|---|---|---|---|
| Where the scheduler lives | Python object next to the coordinator; JS twin | One native owner | Native store, Python GameGraph | Native feeder, Python Engine loop | Native owner thread. Host cost is the measured limit; a Python frontier over many nodes per batch would add to it. The browser runs the same C++ compiled to wasm. |
| Edge weights in value flow | not addressed | "shared values, shared visits and root credits are different quantities" | edge.visits = child->n (child visits) | not addressed | Edge visits (MCGS). Child visits double count a child reached by both turn orders and inflate N in selection and in the completed-Q scale. |
| Visits credited above the root | not addressed | no double claims | every playout increments n at each stored prefix of the root's history | n/a | Credit only the edges of the single game-line path above the root, once per playout. |
| Root change with requests pending | not addressed | | root_at refuses pending requests | feeder keeps requests across in-flight batches | Allow. Requests carry a view id; backup runs along the stored path, which stays valid; per-view credits are skipped if the view is gone. |
| Eviction | not addressed | profile maintenance | full scan and sort, only with no requests pending, at begin/root_at | keeps requests pending | Incremental eviction inside the loop, skipping pending and solver-busy nodes. Under the lane's feeder PR 328's eviction would never run. |
| Solver table per game | "one table per game in the pool" | | n/a | n/a | Not possible as stated: the resident table is per worker thread (dfpn.rs `thread_local`), one per IsolatedTactics process. Decision: per-worker tables plus continuation affinity. |
| Cancellation at deadline | running query finishes its slice | | n/a | n/a | Cancel at once (PR 337) and keep the returned bounds. |
| PV check | n/a | | Recheck: S - 2R at A, R at C, R again at A if Q fell by 0.05 | n/a | Becomes the line-view rule with K = 1. Recheck stays as the actor flag until the scheduler is validated. |
| Visit scale of the target | n/a | | max visits includes inherited visits (docs note the bias) | n/a | Flag `target_scale`: 'edge' (today) or 'search' (this search's visits). Play and analysis switch after the stability test. |
| 128-row batches | inputs: 9.6 ms at 24x24 | | n/a | captures 128 opt-in, default 32 | 9.6 ms is right but 24x24 is 4% of actor rows. At 32x32 (93% of rows) 128 rows take 17.0 ms and 64 take 8.6 ms (measured, large-capture-benchmark.json), so the target is 64 rows at canvas 32. |
| Expected gain from a native feeder | "5x plausible only with full batches AND bulk feeding" | same | n/a | measured -10% CPU replay, -2% GPU end to end | Native feeding is necessary, not sufficient. The 5x for play comes from phase barriers and views; for actors the host cost must halve. |
| Six | "C++ alpha-beta style, Tyto's" | | | | PUCT MCTS by CixMango; see section 1. |

Conflicts between PR 328 and the lane's feeder are listed in section J.4 so the two owners can coordinate.

## A. Components and ownership

### A.1 Three graphs, three identities

Sol's point stands: there are three graphs. The game-position graph (this document's "search graph"), the task
graph between CPU and GPU work, and the CUDA kernel graph. Sharing positions removes neither of the other two.

Three identities, and where each lives:

| Identity | Key | Lives in | Used for |
|---|---|---|---|
| Rule position | 128-bit position key: stones by colour, mover, remaining placements (gumbel.cpp `keys`) | Outcome table (owner), solver requests (as history), resident solver tables (as the prover's own hash) | Exact values, proofs, distances. Independent of turn order and of the network. |
| Neural context | 128-bit context key: position plus the stone placed earlier this turn plus the opponent's previous turn; plus the model version | Node store (owner), evaluation cache (owner, per model) | Network inputs and outputs; one node per context. |
| Search view | View id | View table (owner) | Root, kind, deadline or budget, Gumbel noise, halving schedule, per-view credits, in-flight count. |

Today the same neural context identity is implemented four times: `gumbel::keys` (128-bit, board), PR 328's
`keys(history)` (128-bit, history), `EvaluationCache.key` and `position_key` (Python tuples), and the lane's exact
`feeding::context` vector key. Decision: the owner uses `gumbel::keys` only. The feeder reads a request's context key
from the owner instead of recomputing it. The exact vector key stays as a debug assertion.

### A.2 Components

```
                 Python shell (one thread, plus I/O threads)
   config, clocks, UI, model load, rows, solver process I/O, torch feeder
        |  views, moves, stop          ^ events: view results, proofs, stats
        v                              |
 +------------------------------------------------------------------+
 |  Graph owner thread (native, one per process)                    |
 |  node store per game | outcome table | views | requests          |
 |  next batch per canvas (mutable) | solver frontier | eval cache  |
 +------------------------------------------------------------------+
   | batch ready (id, canvas, rows,      ^ batch done (id, output ptr)
   |  staging slot already filled)       |
   v                                     |
 GPU feeder thread (Python + torch)      |
   H2D copy, graph replay, D2H copy, event --+
   | job (id, node ref, history,          ^ result (id, status, moves, turns,
   |  attacker, slice, resume hint)       |  pn, dn, fresh nodes, cache hit)
   v                                      |
 Solver I/O threads (Python, one per worker, mostly blocked)
   |                                      ^
   v                                      |
 Solver worker processes (IsolatedTactics, PDS-PN, resident table each)
```

| Component | Owns | Receives | Sends | Never touches |
|---|---|---|---|---|
| Graph owner (native thread) | Node store, outcome table, eval cache, views, request table, next batch, staging slot assignment, solver frontier, counters | Commands from Python; batch completions; solver results | Batch descriptors; solver jobs; events | torch, solver processes, files |
| GPU feeder (Python thread) | CUDA graphs, pinned staging buffers (registered with the owner), streams, events | Batch descriptors | Completion with output buffer pointer | Node fields, keys, histories |
| Solver I/O threads (Python) | One IsolatedTactics process each | Jobs | Results (parsed JSON, verified flag) | Node fields |
| Solver workers (processes) | Resident PDS-PN table per attacker colour | Request JSON | Response JSON | Anything on the graph |
| Python shell | Settings, clocks, rows, UI state, model versions | Events, snapshots | Commands | Node fields; it reads only bulk snapshots |

What crosses: node references as (id, generation), histories as int64 arrays, encoded planes in staging slots,
prediction buffers, solver JSON. What never crosses: node pointers, edge arrays, partial trees, network values to
the solver (until a later stage adds prior hints), solver tables to the owner.

Threads per process: the owner does not need the GIL; the feeder holds it only during torch calls (about 1.6 ms per
launch, estimate from the profiled 2.2 ms `F5.gpu_submit` over the 1.41 profiling slowdown). Selection and
installation therefore overlap torch launch and decode, which today run on the same Python thread.

### A.3 The owner loop

```
loop:
  1. apply completions:  GPU batches (install, backup), solver results (prove, bounds)
  2. maintenance budget:  eviction step (<= 256 nodes), dirty cleanup counters
  3. GPU:    while fewer than 2 batches in flight:
               fill the next batch per canvas from views (section C, D)
               if a launch trigger fires: hand the batch to the feeder
  4. solver: while a worker is idle and the frontier has an eligible job: dispatch the top job
  5. views:  finish views whose budget or deadline is reached; emit results
  6. wait on the completion condition, timeout = earliest of (oldest request age limit, view deadline)
```

Every step is bounded so the loop never stalls the GPU for more than one batch time.

## B. The graph

### B.1 Node and edge fields

Today an Edge is 88 bytes and a Node 80 bytes, and an expanded node holds one edge per legal move, 640 on average
(measured, docs/search-outcomes.md). That is about 56 KB per expanded node. At 7k rows per second an analysis graph
would grow by 390 MB per second. Compact storage is a prerequisite for long pondering.

Proposed layout (estimate of sizes):

| Field | Type | Bytes | Notes |
|---|---|---|---|
| context key, position key | 2 x 128-bit | 32 | identities A.1 |
| player, remaining, stones | packed | 4 | |
| network value U | float | 4 | |
| q (MCGS value, mover's view) | float | 4 | recomputed, section B.3 |
| visits n = 1 + sum of edge visits | uint32 | 4 | own evidence, section B.4 |
| carried visits, carried sum | uint32, float | 8 | PR 328: evicted child statistics kept on the edge |
| exact winner, distance, bound | int8, int16, bit | 4 | today's proof marks |
| proof number, disproof number | uint32 x 2 | 8 | last solver attempt at this node, attacker = mover |
| solver nodes spent (fresh only) | uint32 | 4 | E.6 |
| solver state, attempts, last worker | uint8 x 3 | 3 | idle, queued, running |
| gpu pending (request id) | uint32 | 4 | a reservation, 0 when none |
| dirty, expanded flags | bits | 1 | |
| used clock | uint32 | 4 | eviction generation |
| Q movement (EMA of absolute change per backup) | float | 4 | section C |
| parents | small vector of node refs | 16 + 8 per extra | usually 1 or 2 |
| legal list: action (int32 q, int32 r) and logit (float) | per legal move | 12 x L | sorted by prior, descending; coordinates stay wide, a translated compact board can exceed int16 (lane review) |
| touched edges: index, visits, pending, exact, distance, bound, eligible, child ref, value sum (float) | per edge with a child, a visit, a pending descent or a proof | 24 x T | sparse; the sum is the edge's own running mean, which B.3's v(n,a) falls back to when the child is evicted, so an evicted child's statistics stay on its own edge (Codex review) |

Total for L = 640 and T = 10: about 8 KB, seven times smaller than today. Unvisited edges with no stored child
need only their prior, because every such eligible edge shares the same completed Q; the deterministic interior
rule therefore picks the highest-prior one, which the sorted legal list gives at once. An edge with zero visits
from this parent but a stored child that holds evidence from another parent is a touched edge: it reads the
child's value and is never part of the untouched group (lane review).

Today's expansion also draws one random number per edge to "advance the stream" for every node, root or not
(gumbel.cpp `fulfill`). Only root edges read Gumbel noise and `begin` redraws it. The compact path drops those draws.
That changes the random stream, so it is part of the new path only, never the legacy one.

Node references are (32-bit arena index, 32-bit generation). A request or solver job holding a reference to an
evicted node sees a generation mismatch; section E.8 says what happens then. This replaces `shared_ptr`/`weak_ptr`,
whose atomic counts buy nothing with a single owner thread.

### B.2 Keys, tables and eviction

- Node store: context key to node ref, one per game graph (PR 328 `store`).
- Outcome table: position key to outcome (winner, distance, bound, witness edges), never evicted while the game can
  return to the position. PR 328 keeps it for the whole game in shared mode; that is right.
- Evaluation cache: context key plus model version to compact prediction, one per model per process. With the store
  holding every expanded node's prior, the cache matters for re-expansion after eviction and across games (actors).
  One cache, owned by the owner. The lane's C++ cache and the Python `EvaluationCache` become this one.
- Children index: PR 328's `link` probes the position table once per legal move at every expansion (640 hash
  probes). Replace it with an index from a node's position key to the stored positions one stone later, filled when a
  node is created. Expansion then attaches only children that exist.

Eviction (Stockfish replacement, adapted): when expanded nodes exceed the game's limit, the loop removes up to 256
leaf nodes per pass (no stored children) in order of oldest `used` clock, then fewest visits. It skips nodes with a
GPU reservation, a running or queued solver job, and nodes on any view's root path. A removed child's last visits
and value stay on the parent edge (PR 328's carried statistics). Exact outcomes stay in the outcome table. No full
scan: the owner keeps an intrusive list of leaves ordered by `used`.

Limits (defaults, tunable): play and analysis 200,000 expanded nodes per game (1.1 GB compact, estimate; today's
layout would need 11 GB); actors 2,000 per game (11 MB compact per game, 1.4 GB per 128-game process, estimate).
Actors keep today's tree behaviour until the compact layout lands.

### B.3 Value flow (MCGS with edge visits)

For every node:

```
q(n) = exact value                                   if n is exact
q(n) = (U(n) + carried_sum(n) + sum_a N(n,a) * v(n,a)) / (1 + carried(n) + sum_a N(n,a))   otherwise
v(n,a) = +-1 if edge a is exact; else q(child) from n's mover's view if the child has evidence; else edge mean
```

N(n,a) are edge visits: playouts that went from n through a. Child visits are not used as weights. This is the
change from PR 328, where `clean` and `current` set `e.visits = e.child->n`.

Why it matters, with an example. After a search at the turn-end position C, PR 328 gives both turn orders B and B'
an edge to C with C's full visit count, and A's value then counts C's evidence twice, once through each order. The
interior rule also reads those foreign visits as N(a), so it treats C as explored and steers away from it, and the
completed-Q scale `50 + max visits` grows with visits that this node never chose, which sharpens the target.

Inherited evidence still flows: `v(n,a)` reads the child's current q whenever the child has evidence, so the
completed Q of edge a at B' uses C's deep value even if B' never visited a. What does not flow is the visit count.

Propagation after a backup:
- Nodes on the backup path are refreshed at once (the path is short).
- Other parents of changed nodes are marked dirty. A dirty node's ancestors are dirty, so the walk stops at the
  first node already dirty (PR 328 `stale`). Amortised O(1) per backup.
- A dirty node is cleaned when read (selection, stats, snapshot), children first (PR 328 `clean`).
- Exact verdicts propagate eagerly (they are rare and they change eligibility).
- Counters: dirty marks, nodes cleaned, clean time. Sol is right that this can become the next CPU cost.

### B.4 What counts as a visit for a root

For a view rooted at R, four kinds of evidence exist. Only the first two are visits of R's edges.

| Kind | Counted in N(R,a) | Example |
|---|---|---|
| Own visits | yes, and in the view's credits | playouts of this view since its begin |
| Retained visits | yes | playouts of earlier searches at R, or that passed through R from an ancestor root |
| Lineage credits | yes, along one path | playouts of a view rooted below R on the game line credit each edge of the line from R down to that root, once |
| Inherited evidence | no | a child's q built from visits through other parents (transpositions, other turn order) |

Lineage replaces PR 328's prefix counting. PR 328 increments `n` at every stored prefix of the root's history; with
child-visit edges that reaches both turn orders. Here the owner keeps the game line (the moves actually played, or
the analysis line) and a playout at a view rooted at C credits only the edges of that line above C. A view whose
root is off the line (a line view at a candidate's turn end) credits the path from its parent view's root to its
own root.

Halving uses each root edge's per-view epoch credits, as today, never N. The final choice uses the Gumbel score over
survivors. The completed-Q scale for the decision keeps `50 + max N` (today's rule). The scale for the recorded
policy target is a flag (section H).

### B.5 The principal-variation check

PR 328's rule: search A with S - 2R, move to the turn end C and search it with R, return to A and spend the last R
only if the chosen stone's completed Q fell by more than 0.05. In this design that is the line-view rule with K = 1,
f = 0.25, delta = 0.05 (section C.4), run concurrently instead of in sequence, and without moving the game's root.

### B.6 What a training row records

Two measurements, kept apart from the hypothesis that joins them: during the 207500 regression the value targets
came from full searches 9.4 plies older than the row and the calibration slope fell (issue 220, 06:47 comment);
the lane's teacher study found current cheap roots (BCE 0.3217) better than carried full roots (0.4022) at 17 to
48 remaining plies (recovery-data-check-20261003.json). Whether target age caused the regression is not
established; what is established is that carried targets measure worse as teachers at those horizons. The graph
must not bring back carried targets in a new form. Rules:

1. A row records its own root's q at the end of its own search as the root search estimate, with provenance.
   Today's teacher (policy-weighted raw child values with the fallback) stays the default; using the root q as
   the teacher is a separate flag with its own strength and learning check (lane review). Retained and lineage visits are evidence
   about that same position, so they count, as tree reuse already does today.
2. No row ever takes a value computed at another root, even an ancestor or descendant on the game line.
3. A row is written once. Later evidence (a deeper search elsewhere that changes this root's q) never rewrites it.
   Exact proofs still relabel through the existing path (`SelfPlayGame.label`).
4. The row records provenance: `own_visits` (this search), `root_visits` (N at the end), `q_own` (value from own
   playouts only, kept in the view), `q_root` (the target), and `model` (the graph is dropped on a model change).
   The learner can then measure whether inherited evidence helps or hurts, by horizon, before using it.

### B.9 The store, worked backwards from what one simulation touches

What the loop needs per step: selection reads, for each child, logit, completed Q, visits, pending and the exact
state, and per node the net value, total and max visits (Gumbel noise and the halving round at the root only);
backup updates visits and pending along the path, the node's mean and dirty marks on parents; settle needs the
exact state of every child and whether the children are the complete legal set; the graph needs the position key,
the parents, re-root and eviction; the solver needs proof and disproof numbers, fresh nodes spent, a slice
generation and an in-flight bit; the root target needs the improved policy over every legal move.

What the engine constrains: legal moves are radius 8 around stones, about 217 cells early and 600 to 1000
mid-game, and today every one becomes a 100-byte edge (gumbel.cpp:355), so one node is 50 to 100 KB and
transformed() rebuilds the completed-Q logits over all of them on every pass. Two stones per turn make
transpositions the norm, so graph mode is the only mode and most nodes have exactly two parents. A node with a
capped child set cannot be declared lost by exhaustion; only the solver, which sees every legal move, or a node
holding the complete set may do that, so completeness is a per-node bit that settle respects. One writer thread, no
atomics. The browser runs the same code on wasm32 and phones with a node budget of a few hundred MB. The evaluator
needs the same seed and budget to give the same move, which index-addressed storage with fixed child order gives
and pointer-addressed storage does not.

Precision: node mean value float64 (updated a million times at the root; float32 drifts); mirrored child Q and the
cached transformed logit float32 (they feed an argmax and a softmax); logits float32 (the root target is built from
them); visits int32; pending int16 (bounded by the batch); exact state packed in 16 bits (winner 2, bound 1,
eligible 1, distance 12); action two int32, since a translated compact board can exceed int16; position key 128 bits; proof and disproof numbers uint32, spent uint16
saturating, generation uint8.

Layout: children are 32-byte records in the parent's contiguous block (node index, action, logit, mirrored Q,
cached transformed logit, visits, pending, exact state); a node is one 64-byte line (key, float64 value, net value,
visits, pending, first child and count, two inline parents plus an overflow index, proof and disproof numbers,
spent, flags expanded/complete/dirty/solver-in-flight, generation). Flat arrays addressed by 32-bit index. No
shared pointers, no vectors.

Adaptive K: at expansion keep children in prior order until the kept mass reaches 0.995, bounded to 8..48, plus
every forced cell from the leaf classify; with today's policy the median is about 12. The dropped mass is one rest
entry with its aggregate prior so the improved policy normalises; if its improved share exceeds the weakest kept
child it is materialised from the node's own compact legal list (B.1: action and logit per legal move, sorted by
prior), which every node keeps for its lifetime. The evaluation cache is an LRU of 4,096 entries shared across
games (python/neural_search.py) and cannot be relied on for this, so the legal list, not the cache, is the source of
omitted logits; a node promoted to a root therefore has its full legal set at hand (Codex review). Child records
hold what selection reads. The root keeps the full legal set, with Gumbel noise and the halving round in root-only
side arrays.

Mirrors instead of chases: a child's Q is copied into the parent's record on backup through that edge; when another
parent's backup refreshes a shared child, the child's parents are marked dirty and refreshed before their next
selection. Selection never touches a child node. The completed-Q logits are cached per node and recomputed only for
nodes a backup touched. Re-root marks the subtree reachable from the new root and copies it into a fresh arena with
remapped indices between batches when the store exceeds its budget; solver tables keep their own keys.

Cost: selection at a node touches one line plus K times 32 bytes (six lines at K 12), about 140 lines or 9 KB over a
20-ply path, L1-resident; today about one megabyte per simulation. Memory per expanded node falls from 50 to 100 KB
to about 450 bytes at the median K, so two million simulations fit in one gigabyte, and the phone budget sets K.

### B.10 Order of the store changes (after Sol's second audit)

First group the untouched moves: every unvisited edge without shared evidence gets the same mixed completed Q, so
their improved-policy weight is one maintained sum of exp(logit) times exp(sigma(q_u)), and the best untouched
candidate follows the stored logit order. The hot scan then covers only visited, pending, transposed and proven
edges while every legal move stays available. Adaptive K (B.9) is the second step, measured separately, so the
savings from layout are not confused with the savings from searching less. A rest entry is never a legal move and
never evidence that omitted replies lose; the completeness bit governs loss by exhaustion.

Rows record a root search estimate with provenance (kind, root, budget, visits of its own, inherited share); the
learner constructs the value target from it as it does today (outcome, bootstrapped and calibrated forms), and a
later qualified reanalysis may add a versioned target for the same position without changing the original.

## C. The scheduler's decision rule

### C.1 Views

| View kind | Root | Created by | Weight w_v | Writes rows |
|---|---|---|---|---|
| main | the position to play or the position shown | Python | 1.0 | root comparison (actors, play logs) |
| line | turn-end position of a strong main candidate | owner, rule C.4 | 0.5 | no (flag for analysis export) |
| ponder | our turn-end positions after the opponent's likely moves | owner, while the opponent thinks | 0.3 | no |
| actor | one per game in flight | Python | 1.0 | as today |

Each view has its own Gumbel state: noise per root edge, epoch per root edge, halving schedule, `started`,
`completed`, hold, priority and defence lists. Today `gumbel` and `epoch` sit on the shared Edge. They move into a
per-view array indexed by root edge. Interior selection reads neither, so a node can be one view's root and another
view's interior node.

### C.2 Inside a view: what stays and what changes

Stays (the Gumbel root contract):
- Gumbel-top-k draws m candidates without replacement from logits plus noise (with `root_noise`, from the mixed
  sampling distribution).
- Sequential halving: each phase gives each survivor `max(1, floor(S / (ceil(log2 m) x survivors)))` visits, then
  keeps the top half by g + logit + sigma(completed Q).
- The final choice and the improved policy target are computed exactly as today.
- Below the root, the deterministic rule argmax_a [pi'(a) - (N(a) + pending(a)) / (1 + sum_b (N(b) + pending(b)))].

Changes:
- Barrier. Today `request()` refuses to start a new visit layer while the previous layer has requests pending
  (gumbel.cpp line 315). That gives 30 launches for 128 simulations and 240 for 1000 (computed from `schedule`), with
  4.2 to 4.3 rows each. New: the barrier is the phase boundary. Within a phase every survivor's visits are known in
  advance, so all of them may be requested at once. 128 simulations at m = 16 become 4 phases of 32 rows; 1000
  become 4 phases of about 240.
- Concurrency inside a candidate's subtree is capped: at most c_max descents in flight per root candidate, with
  c_max = clamp(B_target / survivors, 4, 64). Pending descents count in N(a) (no value penalty), so parallel descents
  spread out the way Lc0's collision handling intends, and results still feed back between waves.
- Collisions. A descent that reaches a pending child skips that edge (today's interior behaviour). At most 2 x c_max
  collisions per view per fill pass; then the view yields.
- Transposition early stop (KataGo): with edge visits restored, a playout that reaches a child holding more visits
  than its edge takes the child's q without a network call (today's graph-mode stop, disabled in PR 328's shared
  mode). It counts as an own visit.

Flag: `barrier = 'layer' | 'phase'`, default 'layer' for actors and the evaluator.

### C.3 Per node: breadth, depth, solver, prune

The deterministic interior rule already decides breadth against depth at every interior node: a new child when its
improved-policy share exceeds its visit share, a deeper descent otherwise. The scheduler does not override it.
Overriding it would break the policy-improvement reasoning the targets rely on. The scheduler adds four levers.

Quantities at node n (all from the owner's fields):

| Symbol | Meaning | Unit |
|---|---|---|
| N_n | edge visits into n from its view parent | visits |
| r_n | reach: product of edge visit shares along the view path, N(p,a)/N(p) at each step; 1 at the root | 0..1 |
| q_n, dq_n | value, and EMA (factor 0.2) of absolute change per backup | value units, -1..1 |
| H_n | entropy of the prior | nats |
| U_n | unexpanded prior mass: sum of priors of eligible edges with no visit | 0..1 |
| d_n | placements from the view root | placements |
| g_n | forcing material of the mover (tactical_proof gate level); none below forcing_material.LOW | 0..1 |
| pn_n, dn_n, s_n | last attempt's proof and disproof numbers, fresh solver nodes spent | numbers, nodes |
| busy_n | GPU reservation or running solver job | flag |

Lever 1, prefetch breadth (GPU rows). For an expanded node n on a view's principal line or line-view line, with
d_n <= 6 and N_n >= 4, the next unvisited eligible edges by prior are prefetch candidates with priority

```
P_fetch(n,a) = w_v * r_n * pi'_n(a)                      (unitless, 0..1)
```

They fill spare batch rows while the main phase waits at its barrier, in decreasing priority, while
P_fetch >= 0.02, and never more than 25% of a batch. A prefetch row expands the child and stores the prediction. It
adds no visit and no credit; the child's evaluation is used when the search gets there.

Lever 2, depth by line views (section C.4).

Lever 3, the solver (section E). A node becomes a solver candidate when it is expanded or a turn start, not exact,
not busy, r_n >= 0.002, and either g_n is at least LOW or n is on a main or line principal line within 4 placements
of the view root.

Lever 4, prune. Only proofs prune: a proven-lost edge becomes ineligible while an alternative exists, a won node
offers only its shortest wins, an exact node is never expanded or evaluated again (today's rules). No soft pruning:
the deterministic rule already gives low-pi' edges no visits, and Gumbel's guarantee assumes that survivors get
their phase visits. Memory eviction (B.2) is not pruning: an evicted subtree can be rebuilt.

### C.4 Depth: K line views per main view

After each halving phase of the main view, for the top candidates a by improved policy:

```
K      = number of candidates with pi'(a) >= p_line, at most K_max
A      = f_line * S                                        the line-view allowance of the whole main view
R_a    = round((A - spent so far on line views) / phases left * pi'(a) / sum over the K candidates of pi')
C_a    = position after a and, while the same side moves, the improved-policy best next stone (PR 328's turn rule)
```

The allowance A is cumulative over the main view: with four halving phases each phase hands out a quarter of what
is left, so line views never exceed f_line x S in total and the stated budget comparison holds (Codex review).
Defaults: p_line = 0.15, K_max = 2 (play), 3 (analysis), 0 (actors and evaluator); f_line = 0.25. S is the main
view's budget, or for a time-based view the estimated simulations in its allotment (rows per second x seconds). All
tunable.

A line view at C_a samples its own m candidates and runs halving with budget R_a, so the opponent's replies at C_a
get the root treatment that the interior rule never gives them. This is what found the refutation in Tom's position
(PR 328 description: A at 87%, C at about 50% when searched as a root).

After a line view finishes: if candidate a's completed Q at the main root fell by more than delta = 0.05 (value
units) since the line view started, the main view gets a refresh pass of round(f_line x S / 2) simulations: a new
halving over current counts, as PR 328's third step does.

Root widening (breadth at the root). The next main pass samples m' = min(2m, 32) candidates when either:
- every line view of the last round lowered its candidate's Q by more than delta, or
- the unexpanded prior mass at the root U_R > 0.3 and the best survivor's completed Q is below the mixed value by
  more than 0.1.

The issue 220 audit found Six's choices inside Bubble's width-16 sample 93 to 96% of the time but only 71 to 80% at
width 4 (measured, 02:32 comment). Widening is cheap insurance at play strength; actors keep fixed m.

### C.5 GPU slots across views

Every fill pass orders candidate rows by class, then by priority within the class:

1. Mandatory: requests of a main or actor view's current phase (root contract). Priority by view age.
2. Line and ponder views' phase requests, priority w_v x pi'(a) of the candidate they belong to.
3. Prefetch rows, priority P_fetch.

Fill stops at the launch trigger (section D). Classes 2 and 3 never delay a launch that class 1 could make.

Note: inside a halving round the second visit to a candidate needs the child's policy from the first and pending collisions force flushes, so 128 simulations take about 8 to 12 launches, not 4. Six's self-play search is Gumbel top-m with sequential halving at the root and PUCT below (tools/six/src/mcts.cpp:516) and is the reference for batching under that contract. The phase-barrier prototype runs on the current store, in parallel with the store work, with three counters: issued work, completed work and comparison credits.

## D. Batching

### D.1 Assembly

- Rows are grouped per canvas size (24, 32, 40, 48, 64 and the eager sizes above). Each canvas has its own next
  batch and its own trigger. Merging a small group into a larger canvas stays as today (MERGE_CELLS rule) at launch
  time.
- Dedup by context key before a row enters a batch: a key already in the next batch or in flight gets a subscriber,
  not a row (the lane's coalescing across queued and in-flight batches). Cache hits install at once (Lc0 out of order
  eval).
- The owner encodes each row straight into the canvas's pinned staging slot when the row is admitted (or hands it to
  an encoder helper thread, D.6). The feeder never encodes.
- The owner decodes: the feeder hands back the packed output pointer; the owner maps each row's cells to legal logits
  (today done per row in Python, `Evaluator.collect`) and installs.

### D.2 Targets and triggers

| Use | Target rows at canvas 24 | at canvas 32 | at 40 and up | Oldest-request age limit | Deadline guard |
|---|---|---|---|---|---|
| Play on a clock | 64 | 64 | 32 | 2 ms | launch now if time left < 2 batch times |
| Analysis and pondering | 128 | 64 | 32 | 5 ms | none |
| Actors | 128 | 128 | 64 | 20 ms | none |

Launch a canvas batch when any holds: its rows reach the target; its oldest row is older than the age limit; a
deadline is within the guard; or no view can add a row (all wait at barriers, nothing to prefetch) and fewer than two
batches are in flight. Fill counts unique rows after dedup.

Double buffering: at most two batches in flight. The next batch stays mutable until handed to the feeder; a submitted
batch is immutable (Sol). A third batch fills while two run, so each canvas needs three staging sets, or an
input-release event after the copy to device frees a set for the filling batch (lane review).

Captures: CUDA graphs for 24 up to 128 rows and 32 up to 64 rows (PR 338 sizes), 32 above. Tails go to the next
capture size up; padding is counted.

### D.3 Measured GPU cost

Resident-input CUDA graph replay, frozen main/200000, uncontended (large-capture-benchmark.json):

| Rows | 24x24 ms | rows/s | 32x32 ms | rows/s |
|---|---|---|---|---|
| 16 | 2.0 | 8,000 | 3.2 | 4,950 |
| 32 | 3.05 | 10,500 | 4.85 | 6,600 |
| 64 | 5.1 | 12,600 | 8.6 | 7,300 |
| 128 | 9.6 | 13,300 | 17.0 | 7,500 |

Linear fits (estimate): 24x24, 0.91 + 0.068 n ms; 32x32, 1.23 + 0.123 n ms. At 32x32 the GPU is saturated near 64
rows; 128 rows buys 3%. In the measured actor run 6033 of 6469 rows were 32x32, 258 were 24x24 and 178 were 40x40
(actor-optimized-p7-n64.cuda-summary.json). So 32x32 is the case to optimise.

### D.4 Measured host cost

Uncontended actor search, 128 games, 64 simulations, 6469 rows, 54 submissions (native-feed-gpu-comparison.json):
1.41 to 1.44 s wall, 1.34 to 1.38 s process CPU. That is 207 us of CPU per row, and the search is host-bound: the
GPU part of the same work is about 1.0 s (estimate from D.3).

Where the host time goes (profiled run, 1.41x profiling slowdown, actor-optimized-p7-n64.cuda-summary.json):

| Stage | Share of host CPU |
|---|---|
| `hxg_fulfill` (expansion, tactics classify, backup) | 24% |
| leaf encoding | 22% |
| `hxg_next` (selection, completions for tactics) | 16% |
| torch submit | 11% |
| legal lists, history copies | 11% |
| cache keys, gets, puts | 12% |
| decode in collect | 3% |

The lane's native feeder removes most of the Python glue around these: CPU replay 1.035 to 0.936 s (-10%), GPU end
to end 1.437 to 1.413 s (-2%) (native-feed-comparison.json, native-feed-gpu-comparison.json). Fulfill, encoding and
selection remain.

### D.5 Timing model and predicted throughput

Pipelined loop with two batches in flight: time per batch = max(H x n, G(n)), where H is host CPU per row on the
owner thread and G the GPU time. Rows per second (computed from the fits above):

| H (host us per row) | n = 64 at 32x32 | n = 128 at 32x32 | n = 128 at 24x24 |
|---|---|---|---|
| 207 (today, measured) | 4,800 | 4,800 | 4,800 |
| 150 | 6,700 | 6,700 | 6,700 |
| 100 | 7,000 | 7,500 | 10,000 |
| 75 | 7,000 | 7,500 | 13,300 |

What the bulk calls must cost: for a 128-row batch at 32x32 to keep the GPU busy, all owner work per row (select,
encode or hand off, decode, install, backup, dedup) must stay under 133 us, and under 75 us at 24x24. Target H = 100
us. Route to it (estimates, each to be measured): encoding on two helper threads (-22%), decoding and caching in the
owner (-15%), compact expansion without per-edge random draws and with the children index (-10%), feeder glue
(-10%, measured).

Predicted rows per second (estimates unless marked):

| Use | Today | After phase barrier and views | After H = 100 us |
|---|---|---|---|
| Actors, one process, 128 games | 4,600 (measured, uncontended) | same | 7,000 to 7,500 at 32x32 |
| Actors, 4 processes plus learner | 3,600 total (measured, live actor-status files) | same | bounded by the GPU share left by the learner |
| Play, one game | about 1,000: 4.3 rows per launch, synchronous (estimate) | about 4,000 at 128 simulations (4 phases of 32, barrier-bound) | 5,000 to 7,000 with line views filling barrier waits |
| Analysis, one game | about 1,000 (estimate) | 5,000 to 7,000 | 7,000 to 7,500 |
| Six, for reference | 3,300 to 7,300 visits/s, 4,900 network rows/s at ply 7 (measured, six-work-profile) | | |

For context, smaller cohorts today (measured, actor-pool-comparison.json, shared load): 8 games 2,700 rows/s at 14
rows per batch; 32 games 3,800 at 38.5; 128 games 4,600 at 120.

Play's wall time today is not dominated by the network at all. At the standard preset `solve` asks two root queries
of 32,768 nodes each before any search (play.py `solve`), about 1.3 s each at the calibrated 26 nodes per ms per
worker (dense_solver RATE), against well under a second for 128 simulations. Running the solver concurrently
(section F) is the largest single latency gain for play.

### D.6 Encoder helpers

Encoding is a pure function of a request's history and legal list (gumbel.cpp `encode`). Two helper threads take
admitted rows from a lock-free queue and write planes into the staging slot; the owner marks the batch launchable
when all its rows are encoded. Helpers read only the request's own copied history and legal list, never node fields.

### D.7 VRAM

The 3070 Ti has 8 GB shared with the learner and the desktop.

| Consumer | VRAM | Source |
|---|---|---|
| Learner | about 3.8 GB at an allocator cap of 3840 MiB; 3328 MiB OOMed (capacity study); production cap `vram_reserved_mb` = 0 (unlimited) in the run config | issue 220 06:04 comment; runs/dense-v1/config.json |
| Each actor process | 416 to 430 MB reserved plus about 0.3 GB CUDA context | actor-status*.json; hexnet.vram docstring |
| Evaluator worker | about 0.7 GB each | dense-run-launch memory note |
| CUDA graph captures per evaluator | 22 to 186 MB measured increments, budget 384 MiB (`ActorGraph.max_incremental_bytes`) | large-capture-benchmark.json; hexnet_graphs.py |
| Staging per batch | pinned host, 1 MB at 128 x 8 x 32 x 32 bytes; device BF16 input 2 MB | arithmetic |
| Desktop browser | up to 2 GB observed | issue 220 03:30 comment |

Decisions: keep the 384 MiB capture budget; capture 24/128 and 32/64; three staging sets per canvas (two in flight, one filling), or two with an input-release event after the device copy.
With H = 100 us one actor process can feed most of the GPU, so the actor count can drop from 4 to 2, saving about
0.7 GB of context each (estimate). That is a run-owner decision after measurement.

## E. Solver policy

### E.1 What the solver is today

tools/tactical wraps the vendored hexo-strix PDS-PN prover. Facts that constrain the design (read in lib.rs,
dfpn.rs, tactical_proof.py, dense_solver.py):
- One query is a function of (position, attacker, budget, build) unless `table_mb` > 0. Then a thread-local resident
  table per attacker colour carries state across queries of that worker (dfpn.rs `set_resident`).
- The per-query table size follows the budget: one MiB per 2048 budgeted nodes, 1 to 16 MiB (lib.rs
  NODES_PER_TT_MB).
- A process-wide cache keyed by position and budgets replays a stored search, and returns the stored node count as
  `nodes_used`.
- Only a verified PROVEN_WIN acts; selective negatives stay UNKNOWN.
- PR 337: cooperative cancellation through certificate reconstruction (staged, not yet active in production).
- The pool runs IsolatedTactics worker processes, earliest deadline first, at a measured rate capped at 26 nodes per
  ms per worker.
- Nothing returns proof or disproof numbers.

### E.2 The frontier queue

One frontier per game, inside the owner. Entries are (node ref, attacker, slice nodes, priority). The attacker is the
node's mover, so a proof marks the node won, and settle and propagation turn proven children into a lost parent.
Threat queries (attacker = opponent, flipped turn) stay at view roots only, where they order root sampling
(`hxg_priority`) and feed defence candidates as today.

At view creation two jobs are queued with a priority boost: the mover's win at the root and the opponent's threat.
They are today's root and threat points. Measured live hit rates: root 1.6%, threat 62% (solver block of
actor-status.json).

### E.3 Priority

```
S_n = w_v * r_n * D_n * P_n / C_n  +  alpha * age_n

D_n = max(0.1, 1 - |q_n|)                       decision sensitivity, 0.1..1
P_n = h(band(g_n), slice) * rho_n               probability that this slice proves the node
rho_n = 1 for a fresh node; clamp(2 * dn / (pn + dn), 0.5, 2) for a continued one
C_n = slice_nodes / rate + 0.5                  ms, rate = measured pool rate (<= 26 nodes/ms)
age_n                                            seconds in the queue
alpha = 0.02 per second                          in the same units as the first term
```

Units: S_n is expected decision value per millisecond of one worker. The hit-rate table h starts from the pool's
measured hit rates by granted budget band (dense_solver BANDS 250, 1000, 4000 and above) and updates online per
gate band. rho_n and alpha are estimates to tune with the counters of section I. A node whose nearest ancestor
within 4 placements has a running job gets S_n x 0.25 (a soft virtual mark, after SPDFPN), so workers spread over
different subtrees.

The job with the highest S_n goes to the next idle worker. Entries are rescored lazily: when popped, a stale entry is
rescored and pushed back if it lost more than 20%.

### E.4 Slices

| Use | First slice | Next slices | Cap per slice | Cap per node |
|---|---|---|---|---|
| Play and analysis | 2,048 nodes (about 80 ms, estimate) | doubling per attempt | 32,768 (gate_cap_nodes) | the preset's solver budget; 4,000,000 at dangerous |
| Actors | 512 nodes | doubling | 2,048 (live cap_nodes) | 32,768 (live gate_cap_nodes) |
| Leaf filler (new leaves with forcing material, only when workers idle) | 64 nodes (Six's leaf budget) | none | 64 | 64 |

Near a deadline the slice shrinks to at most half the time left.

### E.5 Continuation and table affinity

A per-game table shared across workers does not exist and would need shared memory between processes. Decision:
each worker keeps its resident table (`table_mb` 64 for play and analysis, 32 for actors), and a continued job goes
to the worker that ran the node's last attempt (`last worker` field). If that worker stays busy longer than one
slice, any idle worker takes the job cold. For play, where all workers serve one game, the first job under each main
candidate picks a worker by candidate, so a candidate's subtree tends to stay on one table.

The partial bounds stored on the node (pn, dn, fresh nodes) serve three purposes: priority (rho_n), the page ("tried,
N nodes"), and accounting. They do not let a cold worker resume; only the table does. PDS-PN discards its
second-level trees, so the warm gain is mostly first-level reuse; lib.rs has a test that warm proves with less work,
but the size of the gain on real positions is open question 5.

### E.6 Budget accounting

- Fresh work only: a result reports `nodes_fresh`, the meter's own count for this query. A cache hit reports 0 fresh
  nodes and `cache_hit` true; today it returns the stored count, which would double count.
- The node's `solver nodes spent` adds `nodes_fresh` only.
- The view's solver total, the row's `solver_nodes` and the page's counter all read fresh nodes.
- A cancelled job reports the fresh nodes it spent and its bounds.

### E.7 Solver API changes (tools/tactical, tactical_proof.py)

Request, new optional fields (absent fields keep today's behaviour, and the evaluator's fixed mode is
untouched):
- `bounds: true`: report the root's proof and disproof numbers at the end, for UNKNOWN too.
- `resume: true`: keep and reuse the resident table across this worker's queries without dropping it when the size
  differs (today a new size drops the state).

Response, new fields: `pn`, `dn`, `nodes_fresh`, `cache_hit` (exists), `reason` 'cancelled' with bounds.

### E.8 Installing results, and late results

New native call (PR 328's file, section J):

```
hxg_prove_at(tree, history, n, player, remaining, moves, move_count, turns)  -> 1 or 0
```

It checks the history, phase and first turn like `hxg_prove`, records the outcome for the position (winner, distance
= move_count + 4 x (turns - 1), bound) with the witness edge(s), and installs it on every live node of that position
(PR 328's `record`, `apply`, `share`, `revise`). It needs no pending request and works on interior nodes, roots and
unexpanded nodes. A second call, `hxg_bounds_at(tree, node ref, pn, dn, fresh)`, stores partial bounds.

Rules:
1. A proof always installs. If the node was evicted (generation mismatch), the outcome still goes into the outcome
   table and applies when the node is created again.
2. A proof beats a neural value that arrives later: when a GPU result arrives for a node that became exact while
   pending, the request completes with the exact value, the prediction goes to the cache, and no edge statistics
   change. Counter: late neural results discarded.
3. A node whose query is running stays searchable. The GPU may expand and visit it; when the proof lands, the mark
   replaces the network value and propagation corrects the ancestors.
4. No overlap the other way: the frontier never dispatches a node with a GPU reservation. The network's opinion
   should exist first, because it sets the priority.
5. A proof under a node the GPU is evaluating can settle an ancestor (Sol). The ancestor's pending requests complete
   as in rule 2.
6. In an advancing game, a proof for a position the game can no longer reach (fewer stones than the board) is
   dropped. An analysis graph keeps every proof for the game's lifetime, since analysis revisits earlier positions
   (lane review); the two expiry rules are separate settings. Counter: proofs
   arrived unreachable.
7. UNKNOWN never marks a loss.

Off-game proofs (positions not on the game line) are training row candidates with their own histories (section H).

### E.9 Pool size and CPU

The Ryzen 9 5900X has 12 cores and 24 threads (memory note). Per process the owner, the feeder and two encoder
helpers want about 3 cores (estimate: the feeder is mostly blocked). Defaults:

| Setting | Solver workers |
|---|---|
| Play or analysis while training runs | 4 (today's REVIEW_SOLVERS) |
| Play or analysis on an idle machine | 8 |
| Actors (per process) | 2 (live setting) |

Never more than cores minus the owner, feeder and encoder threads of all engine processes. Solver processes run at
below-normal priority so they never starve the owner.

## F. Keeping the GPU and the solver busy

Two queues, one owner: the next GPU batch (per canvas) and the solver frontier. The owner refills both on every loop
pass, so neither waits on the other.

```
time ->
GPU     [batch 1 ][batch 2 ][batch 3 ][batch 4 ] ...           at most 2 in flight
owner    fill 2 | install 1, fill 3 | install 2, fill 4 | ...   dispatches solver jobs between fills
solver  w1 [job a 80ms      ][job d ...
        w2 [job b    ][job c        ][job e ...
                         ^ result c -> owner installs proof -> subtree leaves the next fills
```

Keeping the GPU busy under a single game: the main view waits at each halving phase boundary. During the wait the
batch fills from line views, ponder views and prefetch rows (section C.5). If a single view with no line views
cannot fill a batch, the age trigger launches what exists, so latency stays bounded.

Keeping the solver busy: when the frontier holds nothing above the gate, idle workers take leaf filler jobs (64
nodes on newly expanded leaves with forcing material), off the critical path. A newly expanded node enters the
frontier when its prediction installs, never before.

Reserving the feeder's CPU: solver processes are below-normal priority and their count is capped by section E.9.
The owner measures its own loop time per row; if H rises above its target for 2 seconds the scheduler lowers solver
workers by one (down to 1) and raises them back after 10 seconds below target.

At the deadline (or a stop):
1. The owner stops filling. Rows not yet submitted are dropped and their reservations released.
2. In-flight batches complete and install (at most 2 x 17 ms at 32x32, 128 rows; measured G). A batch that cannot
   finish before the hard limit is abandoned; its rows are released when it returns.
3. Queued solver jobs are dropped. Running jobs are cancelled (PR 337) and their bounds kept.
4. The view's result is taken from the last finished halving phase (today's hold snapshot rule) plus every proof
   installed so far.

Target: both producers stopped within one batch time plus the solver's cancellation latency. The cancellation
latency is not measured yet (artifacts has probe_solver_cancel.py but no result I could find); gate 5 ms.

## G. Time and budgets

Presets for play and analysis become time per stone. The simulation and solver node counts become reported results.

| Preset | Today (Bubble) | Proposed time per stone (estimate, to be calibrated so no preset gets weaker) |
|---|---|---|
| lightning | 8 sims, 2,048 solver nodes | 50 ms |
| quick | 32 sims, 2,048 | 150 ms |
| standard | 128 sims, 32,768 | 0.5 s |
| strong | 512 sims, 131,072 | 2 s |
| deep | 2,048 sims, 524,288 | 8 s |
| dangerous | 65,536 sims, 4,000,000 | 60 s |

Calibration: measure today's wall time per stone at each preset on the GPU play path, and set the new time no
higher. The equal-time harness (section I) then has to show the new search at least as strong.

Clock play: base allotment t = remaining / max(10, estimated stones left) + increment (estimate). Multipliers after
Stockfish, each measured per finished halving phase:
- best-move changes since the last phase: x (1 + 0.5 x changes), at most x 2;
- falling root Q (drop of more than 0.05 since the previous phase): x 1.5;
- best move's share of root visits above 0.75: x 0.7.
Never beyond 3 x t or the clock's maximum. Stop early when the root is proven, or in the last phase when the leader's
Gumbel score cannot be passed by the second with the remaining visits at the extreme completed Q (Lc0 smart pruning).

Fixed budgets stay:
- The evaluator keeps fixed simulations, the layer barrier and fixed per-point solver budgets (reproducibility
  between checkpoints). Out of scope for this design.
- Reproducible analysis: `deterministic = true` uses fixed simulations, fixed solver nodes per job instead of time
  slices, a pinned seed, GPU completions applied in submission order, and solver results applied only at phase
  boundaries in job-id order. Resident solver tables are off in this mode (every job runs cold) and jobs go to
  workers by job id modulo worker count, never to "the worker that holds the table" or "an idle worker", because
  with warm tables and timing-dependent assignment two runs could prove different things (Codex review). Same
  inputs then give the same graph.

Pondering: while a game is open and the opponent thinks, the main view sits at the current position, and ponder views
run at our turn-end positions after the opponent's top replies (by its improved policy, K_max = 2). When the opponent
moves, the owner re-roots the main view onto the played position. Re-rooting does not stop the loop: requests in
flight keep their paths and back up when they return (section B, root change decision); the old main view's
credits end; the new main view begins with fresh Gumbel noise and retained visits as priors.

## H. Training interface

Row kinds:

| Kind | Source | Policy target | Value target | Default for actors |
|---|---|---|---|---|
| root comparison | a main or actor view's search of a position on the game line | yes for full searches (as today) | records its own root q as the root search estimate (B.6); the learner's teacher is unchanged (policy-weighted raw child values with the fallback) unless `teacher=root_q` | on (today's rows) |
| depth extension | a line view's root (off the game line) | only if it ran at least full_sims own visits with its own root sampling; off by default | records its own root q; teacher as above | off |
| breadth fill | prefetch rows | never | never | n/a |
| off-game proof | a proof installed on a node off the game line | the witness stones for a win (pair targets, PR 238 style); none for a loss | exact +1/-1 with distance | off |

Two masks, kept apart (Sol):
- Search eligibility: a proven-lost edge is not sampled, gets no halving visits and has improved-policy mass 0.
- Loss legality: every legal move stays in the policy softmax. A proven-lost move keeps target 0, so the gradient
  pushes down a confident prior on it. This is today's behaviour (`hxg_policy` gives ineligible edges 0, and the
  learner's policy loss runs over all legal cells); the scheduler must not change it.

Staleness rule for carried values: B.6. In short, a value target is the row's own root value at the end of its own
search, written once, with provenance fields; exact proofs relabel through the existing path.

Actor flags (all default off, which keeps today's behaviour; the test is equal strength and equal targets within spread, not identical rows). `teacher=root_q` is the separate flag B.6 asks for: with it the learner takes the recorded root search estimate as the value teacher; without it the teacher is today's. Other flags:

| Flag | Default | Effect when on |
|---|---|---|
| `native_feed` | False | lane's feeder (exact results today) |
| `game_graph` (PR 328) | 0 | shared store with that node limit |
| `pv_check` (PR 328) | 0 | Recheck on full searches |
| `barrier` | 'layer' | 'phase': phase-level barrier |
| `edge_visits` | False | MCGS edge visits in shared mode |
| `line_views` | 0 | K_max |
| `prefetch` | False | breadth fill rows |
| `solver_frontier` | False | frontier jobs instead of fixed points |
| `target_scale` | 'edge' | 'search': completed-Q scale from this search's visits for the recorded target |
| `row_provenance` | False | write own_visits, root_visits, q_own |
| `offgame_rows` | False | write off-game proof rows |

## I. Measurement and acceptance

Tests check behaviour, never implementation identity. No bit-for-bit, hash or identity comparisons against the old
code. Acceptance is: correctness invariants (a proven node is never expanded again, settle never marks a capped
node lost, proofs reach every parent, re-root and eviction keep every reachable node and proof, the same seed and
budget give the same move twice); quality at fixed budgets (tactical suite, Tom's position, policy-target agreement
within the seed-to-seed spread, paired matches at 128 and 512 simulations); and cost thresholds (microseconds per
row, bytes per simulation).


### I.1 Counters (owner, per view and per process)

| Counter | Definition |
|---|---|
| unique rows installed | network rows installed after dedup |
| padding rows | capture rows minus real rows |
| prevented duplicates | subscribers joined to a queued or in-flight row |
| cache hits | rows answered by the evaluation cache |
| early stops | playouts ended at a transposed child with more visits than its edge |
| prefetch rows, prefetch used | rows from lever 1, and how many were later visited |
| collisions | descents that hit a pending child |
| barrier waits | ms a main view waited at a phase boundary with the GPU idle |
| GPU idle | ms with no batch in flight while work existed, from CUDA events (a gap in our stream is not an idle device; Sol) |
| host us per row | owner loop CPU per installed row, split by stage (select, encode, decode, install, backup, dirty cleanup, eviction) |
| late neural results discarded | rule E.8.2 |
| late proofs | proofs installed after their node's view ended |
| proofs unreachable | rule E.8.6 |
| solver fresh nodes, solver CPU ms | fresh only (E.6) |
| solver cache hits | |
| proofs that changed a decision | a proof that changed a view's final choice or removed a survivor |
| continuation gain | fresh nodes to a verdict, warm against cold, on repeated nodes |
| deadline overruns | stops later than one batch time plus 5 ms after the deadline |
| dirty marks, nodes cleaned | value maintenance cost |

### I.2 Strength harness

Build from what exists:
- `dense_eval.py` has paired games over the standard-v1 opening suite, variants with `--set`, and SPRT. Use it for
  fixed-work comparisons: same checkpoint, flag on against flag off, fixed simulations.
- `timed_match.py` runs clocked games (default 180+2) against a checkpoint, the native player or a Six-protocol
  command, one game at a time. Extend it so both sides can be Bubble with different search settings, then run equal
  time per stone, paired openings from standard-v1, alternating colours, one game at a time so the two arms never
  share the GPU or the solver cores.
- Report Elo with the paired Hoeffding interval and SPRT (elo0 0, elo1 30, alpha = beta = 0.05, max 400 games).
- Tactical suite: Tom's position (PR 328), the user-supplied quiet pair, and the 32 independently verified cases in
  heldout-proven-20261001.json. Report proofs found, time to proof, and choice.

### I.3 Gates per stage

| Stage | Gate |
|---|---|
| 0 counters | no behaviour change; counter overhead below 1% of host time |
| 1 PR 328 merge | its own tests; actors with flags off play at equal strength and produce targets within the seed-to-seed spread |
| 2 views and edge visits | legacy path at equal strength; shared mode: Tom's position still shows the refutation at A after C; fixed-simulation match edge visits against child visits at 512 simulations not worse than -10 Elo (lower bound) |
| 3 interior proofs | a proof installed at a depth-4 node settles the root in the tactical suite; no regression in the 32 verified cases |
| 4 feeder on the owner thread | exact rows with flags on at the layer barrier; host us per row at most 150 uncontended; GPU idle with work pending below 15% |
| 5 phase barrier | fixed-simulation match phase against layer at 128 and 512 simulations: lower bound above -15 Elo; recorded policy target KL to the layer version within the seed-to-seed KL; play latency at 128 simulations at most 40% of today |
| 6 solver frontier and API | fixed mode deterministic at the same seed and budget; tactical suite at equal time: at least as many proofs; deadline overruns 0 in 1,000 stones |
| 7 scheduler in play and analysis | equal-time match against today's play at standard and strong: SPRT accepts +30 Elo; no preset slower than today |
| 8 actors | flags on: rows per GPU-second at least 1.3x; teacher study metric (unproved outcome BCE at 17 to 48 plies) not worse; calibration slope b(h=64) at least 0.5; then a strength trial of the resulting checkpoint against the incumbent |

## J. Delivery plan

### J.1 Ownership

As agreed in issue 220: PR 328's owner (Claude lane) owns src/gumbel.cpp: shared store, views, arbitrary proofs. The
Codex lane owns the feeder module and the scheduler. This plan keeps that split and puts the boundary in a C ABI:
gumbel.cpp exposes views, requests, installs and proofs; the scheduler (in its own native module, the lane's
gumbel_feed.cpp grown into a scheduler) drives them. Sol reviews each PR's design before Codex review.

### J.2 Ordered PRs

| # | PR | Owner | Scope | Tests | Gate |
|---|---|---|---|---|---|
| 1 | Counters and harness | Codex lane | counters in Engine and search_many; timed_match Bubble-vs-Bubble settings; tactical suite runner | unit tests for counters; harness smoke test | I.3 stage 0 |
| 2 | PR 328 merge | Claude lane | as described in the PR | its tests | stage 1 |
| 3 | Views, edge visits, node-based install | Claude lane | View struct with per-view Gumbel state; requests carry a view id; `root_at` and `advance` with requests pending; edge visits in shared mode; lineage path credits; transposition early stop in shared mode; `barrier` option in `request`; incremental eviction; node refs with generations | bit-for-bit legacy; shared-graph tests from PR 328 updated; A to B to A; re-root with requests in flight | stage 2 |
| 4 | Interior proofs and bounds | Claude lane | `hxg_prove_at`, `hxg_bounds_at`, node solver fields, children index | proof at depth, evicted-node proof, late neural result after proof | stage 3 |
| 5 | Feeder on the owner thread | Codex lane | rebase the native frontier on PR 3: requests by node ref, context key from the owner, one cache, per-canvas next batch, triggers, double buffering, encode in staging, native decode, encoder helpers | the lane's exact-replay tests; feeder with phase barrier and views | stage 4 |
| 6 | Solver API | Codex lane | `bounds`, `resume`, `nodes_fresh`, cancel with bounds in lib.rs and tactical_proof.py | fixed mode unchanged; warm continuation spends fewer fresh nodes; cache hit reports 0 fresh | part of stage 6 |
| 7 | Scheduler core | Codex lane | frontier queue and priority; line views; prefetch; root widening; time management; deadline; pondering; Python shell for play and analysis | fake-search and fake-pool unit tests (no overlap with reservations, priority order, continuation affinity, slices return bounds, deadline stop within one batch); CPU integration on Tom's position | stages 5 to 7 |
| 8 | Training rows and actor flags | Claude lane | row kinds, provenance fields, flags, learner reads `kind` | flags off: equal strength and targets within spread; rows with flags on carry post-proof targets | stage 8 (data checks) |
| 9 | Compact edges | Claude lane | the B.1 layout | memory and host time benchmarks; exact search results against PR 3 with the same random stream policy | memory per node at most 7 KB; host time not worse |
| 10 | Browser twin | Claude lane | owner core in wasm, solver Web Workers | tests/web parity | browser parity |

PRs 3 and 5 can proceed in parallel once the C ABI between them is written down (J.3). PR 9 can start any time after
PR 3 and is what makes long analysis affordable.

What must not change: the evaluator's fixed budgets and layer barrier; the Gumbel root contract (sampling, per-phase
visits, final choice, improved-policy target) until stage 5 validates the phase barrier; today's actor rows with
flags off; the solver's fixed-budget mode; proof semantics (only verified PROVEN_WIN acts; UNKNOWN never a loss).

### J.3 The C ABI between gumbel.cpp and the scheduler (to agree before PRs 3 and 5)

```
hxg_view_new(tree, kind, root history, simulations or 0, samples, seed) -> view id
hxg_view_next(tree, view, max rows, out requests)     -> count; respects the view's barrier and c_max
hxg_view_done(tree, view) / hxg_view_result(tree, view, out stats, policy, q)
hxg_request_info(tree, request, out context key, history, legal, canvas hint)
hxg_install(tree, request, actions, logits, values, count)   -> node-based install, backup along the stored path
hxg_prove_at(...), hxg_bounds_at(...)
hxg_frontier_candidates(tree, out node refs and quantities)  -> for the scheduler's solver queue
hxg_evict(tree, max nodes)
```

### J.4 Conflicts between PR 328 and the lane's feeder

1. Pending requests and root moves. PR 328's `root_at` throws while any request is pending. The feeder keeps
   requests alive across queued and in-flight batches. Today a tree begins only after its pending requests drain,
   so this is not a leak in PR 340; it is the reason continuous views need re-root and incremental eviction that
   work with requests pending.
2. Eviction precondition. PR 328's `evict()` runs only when `requests.empty()`, at `begin` and `root_at`. With the
   feeder's cross-batch coalescing a game graph may never be idle at those points, so the store grows without bound.
3. Visit weights. PR 328 sets edge visits to child visits in shared mode. The feeder's backups go through
   `hxg_fulfill`, so whatever weighting PR 328 has applies to the feeder's installs. The scheduler needs edge visits
   (section B); this is a PR 328-file change the feeder depends on.
4. The layer barrier. The feeder's gather "drains to the existing tree barrier" (gumbel_feed.cpp comment). The
   phase barrier and views live in `Tree::request`, which is PR 328's file. The lane's multi-depth frontier needs
   that change from the other owner.
5. Root tracking. The feeder records a root key per tree at `hxgf_begin` for `root_value`. PR 328's Recheck calls
   `hxg_begin` directly and GameGraph moves roots with `root_at`; the feeder is not told, so its root key goes stale.
   Harmless today (Recheck keeps the first pass's network value) but wrong once the feeder serves play.
6. Three key implementations. The feeder computes its own exact context key; PR 328 adds `keys(history)`; gumbel.cpp
   has `keys(board)`; Python has two tuple keys. Same identity, four codes.
7. Two caches. The feeder keeps its own LRU prediction cache and does not write Python's `EvaluationCache`;
   `Engine.begin` still reads the Python cache for the root's network value. PR 328's store also keeps every
   expanded node's prior. Up to three copies of the same predictions (78 MB per 4096-entry cache, measured in docs).
8. Install cost. In shared mode every `hxg_fulfill` also runs `link` (one hash probe per legal move), `renew`,
   `propagate(*root)` and the lineage increments. The feeder's measured gains are on the non-shared path; they will
   shrink in shared mode until the children index (PR 4 here) lands.
9. Same Python functions. Both edit `dense_selfplay.Engine.__init__`, `begin` and `step`: PR 328 adds the
   `checking`/`recheck` path in the done branch and skips leaf proofs while checking; the feeder adds the
   `native_feed` gather branch, a feed per model, and `retire_feeds`. The feeder refuses `leaf_nodes`; PR 328's
   check searches skip leaf proofs. A merged `step` needs both rules.
10. Detach and cancel. The feeder's `detach` calls `hxg_cancel(tree)`, which clears every pending request of the
    game graph, including requests of other views once views exist.
11. Build files. The feeder adds gumbel_feed.cpp to `hexo_gumbel` in CMakeLists; PR 328 changes `GUMBEL_EXPORTS` in
    tools/build_web.py and the wasm. The feeder is absent from the browser build.

No conflict on node fields today: the feeder adds none. The overlap will start with the lane's planned multi-depth
frontier and solver producer, which need reservation and solver fields on nodes. Those fields are in PR 4 of this
plan, on the Claude lane's side.

## K. Open questions

| # | Question | Experiment that settles it |
|---|---|---|
| 1 | Can the owner reach 100 us host CPU per row at 32x32? Without it actors stay host-bound and nothing above 4,800 rows/s is possible. | CPU replay of the recorded 6469-row actor workload (the lane's harness) after each step of D.5's route, stage timings from the owner's counters. |
| 2 | Does the phase barrier (requests inside a phase issued together, pending counted in N) cost strength at fixed simulations? | Fixed-simulation paired match, phase against layer, at 128 and 512 simulations, plus policy-target KL and top-move agreement against the seed-to-seed baseline (the q_range_floor study's method). |
| 3 | Edge visits against child visits in the shared graph: which gives better analysis and better targets? | Tom's position and the tactical suite (refutation found, A's value after C); fixed-simulation match at 512 simulations; target stability across seeds. |
| 4 | Do line views beat a wider root at equal time, and what K_max, p_line and f_line? | Equal-time grid K_max in {0, 1, 2, 3}, f_line in {0.15, 0.25, 0.4} at the standard preset time, 200 paired games per cell, then SPRT on the best against today. |
| 5 | How much does a warm resident table save on real positions, given that PDS-PN discards second-level work? | Replay solver jobs from the tactical suite and from recorded frontier logs: fresh nodes to verdict warm against cold, and the share of continuations that reach a verdict. |
| 6 | Is `target_scale = 'search'` better than counting retained visits? | Target stability (two seeds, top-move agreement) and fixed-simulation strength; for actors, the teacher-study BCE on rows written both ways. |
| 7 | Does inherited evidence in a root's q help or hurt value targets? | With `row_provenance`, the lane's held-out calibration fit split by the share of own visits, per horizon band. |
| 8 | What is the solver's cancellation latency with PR 337 active? | probe_solver_cancel.py on 100 running jobs at different budgets: time from cancel to result. Gate 5 ms. |
| 9 | Do network priors help the solver (DFPN-E edge costs or policy ordering)? | Pass the node's top priors with the job; compare fresh nodes to verdict on the tactical suite. |
| 10 | Does a pn-rank bonus in interior selection (PN-MCTS) help? | Fixed-simulation match with a bonus C_pn in {0, 0.5, 1} on nodes with solver bounds. |
| 11 | Fewer actor processes with more games each once H drops? | Rows per GPU-second and VRAM with 2 x 256 games against 4 x 128 games, learner running. |
| 12 | What explains about 30 Six positions per Bubble simulation? | Count Six's cache and terminal visits per position, and Bubble's tactics-settled replies per expansion, on the same positions. Informational only. |

### K.13 Depth on an over-focused policy

Tom's observation (2026-10-03): the current policy puts almost all its mass on one move, so any deeper search
below the root follows one line, and the sharpened target then teaches the next network that the line was the
whole story. Depth confirms the over-focus instead of correcting it. Two experiments settle what the scheduler
must assume:

1. Interior exploration floor. Add a flag that mixes a temperature or a small uniform share into interior logits
   (below the root only; the root keeps Gumbel's sampling). Measure at fixed simulations: paired match at 128 and
   512, policy entropy of the targets, and whether line views and prefetch rows change any root decision. Without
   a floor the line views in section C assume a spread that does not exist today.
2. Target sharpening. The completed-Q transform scales by 0.1 times (50 + max visits), which turns a small value
   gap into a 100.0 target at deeper budgets. Train from the same checkpoint with the scale capped, and with the
   target mixed with the visit distribution; judge on policy entropy, the certified-stone metrics and a strength
   match. This is a learner experiment and is separate from the play-time q_range_floor, which measured worse.

Until one of these lands, the scheduler's depth work (line views, depth extensions) should be measured on the
tactical suite and Tom's position only, not expected to move Elo.

## References

- KataGo, Monte-Carlo Graph Search from First Principles: https://github.com/lightvector/KataGo/blob/master/docs/GraphSearch.md
- KataGo methods (uncertainty, variance-scaled cPUCT): https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md
- D. Wu, Accelerating Self-Play Learning in Go (playout cap randomisation): https://arxiv.org/abs/1902.10565
- J. Czech, P. Korus, K. Kersting, Monte-Carlo Graph Search for AlphaZero: https://arxiv.org/abs/2012.11045
- Lc0 search flags (minibatch, collisions, out-of-order eval, prefetch, smart pruning): https://lczero.org/play/flags/
- Lc0 classic search: https://github.com/LeelaChessZero/lc0/blob/master/src/search/classic/search.cc
- Stockfish search, time management, transposition table: https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp, https://github.com/official-stockfish/Stockfish/blob/master/src/timeman.cpp, https://github.com/official-stockfish/Stockfish/blob/master/src/tt.cpp
- mctx (Gumbel MuZero reference, sequential halving, completed Q): https://github.com/google-deepmind/mctx
- Danihelka et al., Policy improvement by planning with Gumbel (ICLR 2022): https://iclr.cc/virtual/2022/poster/6418
- T. Cazenave, Batch Monte Carlo Tree Search: https://www.lamsade.dauphine.fr/~cazenave/papers/BatchMCTSFinal.pdf
- M. Winands et al., PDS-PN and PN search chapter: https://dke.maastrichtuniversity.nl/m.winands/documents/pnchapter.pdf
- J. Pawlewicz, R. Hayward, Scalable Parallel DFPN Search: https://webdocs.cs.ualberta.ca/~hayward/papers/pawlhayw.pdf
- 1+epsilon trick (Pawlewicz, Lew): https://webdocs.cs.ualberta.ca/~hayward/talks/hex.epstrick.pdf
- A. Kishimoto et al., DFPN with heuristic edge cost (NeurIPS 2019): https://papers.nips.cc/paper_files/paper/2019/file/4fc28b7093b135c21c7183ac07e928a6-Paper.pdf
- C. Gao, M. Müller, R. Hayward, Focused depth-first proof number search using CNNs for Hex (IJCAI 2017): https://www.ijcai.org/proceedings/2017/513
- Kowalski, Doe, Winands et al., Proof Number Based Monte-Carlo Tree Search: https://arxiv.org/abs/2303.09449
- MoHex, Benzene PlayAndSolve: https://webdocs.cs.ualberta.ca/~hayward/papers/leiden17.pdf, https://benzene.sourceforge.net/benzene-doc/html/PlayAndSolve_8cpp_source.html
- Rapfi paper: https://arxiv.org/html/2503.13178v1 ; source: https://github.com/dhbloo/rapfi
- M. Winands, Y. Björnsson, J.-T. Saito, Monte-Carlo Tree Search Solver: https://dke.maastrichtuniversity.nl/m.winands/documents/uctloa.pdf
- Six: CixMango/Six, vendored at tools/six (release v1.3.3 per tools/build_web.py)
- Local measurements: artifacts/live/gpu-schedule-20261003 (large-capture-benchmark.json, native-feed-comparison.json,
  native-feed-gpu-comparison.json, capacity-search-comparison.json, actor-optimized-p7-n64*.cuda-summary.json),
  artifacts/live/pipeline-profile-20261003 (actor-pool-comparison.json, six-work-profile/measurements.json),
  runs/dense-v1/actor-status*.json, docs/search-outcomes.md, issue 220 comments of 2026-10-03.
