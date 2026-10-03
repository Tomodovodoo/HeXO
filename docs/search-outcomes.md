# Outcomes on the search graph

How Bubble's placement search (`src/gumbel.cpp`) represents, propagates and reuses proven game results, and what it
takes from chess and Go engines. HeXO differs from those games in three ways that shape every rule below: a turn is
two placements by the same player (the opening turn is one), so consecutive plies often share a mover; an exact
one-turn tactics oracle (`classify`) proves immediate wins and forced blocks for every legal placement at once; and an
external certificate solver (`tools/tactical`) proves continuous-threat wins against every defence.

## What other engines do

| Engine | Outcome handling | Taken here | Not taken, and why |
|---|---|---|---|
| Stockfish | Mate scores are `VALUE_MATE - ply`, a distance from the root. `value_to_tt`/`value_from_tt` convert them to and from a distance from the stored node, so a transposed position reached at another depth reads a correct mate distance. Mate-distance pruning narrows `[alpha, beta]` to `[mated_in(ply), mate_in(ply+1)]`, so no line can be searched for a longer mate than one already found. Transposition-table entries carry bound types (exact, lower, upper); a mate bound found in one iteration keeps cutting in later ones. | Distances stored relative to the node itself, so a shared or retained node never needs conversion. Proven values as bounds: a certificate gives "wins within at most d placements", combined by min at the winner's choices and max at the loser's. The root stops as soon as it is proven. | Alpha-beta windows: Gumbel search has no window, so mate-distance pruning reduces to "an exact node is never descended" and "a won node offers only its shortest wins". |
| Leela Chess Zero (classic search) | Nodes keep a lower and an upper game-result bound; `MaybeSetBounds` combines them from the children with the sign flip, so a child loss makes the parent a win and all-child wins make it a loss ("certainty propagation"). With `StickyEndgames` a newly terminal node updates its ancestors at once. Final move choice prefers the shortest terminal win and the longest terminal loss by the moves-left estimate `M`. A separate `dag_classic` backend shares transposed nodes. ("Solid" in lc0 is a memory layout of the children, not a proof concept.) | Win/loss results propagated at backup (#212), shortest win and longest loss at the root, a transposition graph as an opt-in backend. | Draw bounds: HeXO has no draw except the ply cap, which proofs ignore (a proven win longer than the remaining cap is reported but not realisable). |
| KataGo | Terminal positions are leaves with a fixed utility; there is no solver. Its graph search (docs/GraphSearch.md) derives the MCGS backup from first principles: `Q(n) = (U(n) + sum_a N(n,a) Q(child_a)) / (1 + sum_a N(n,a))`, with edge visits `N(n,a)` kept separate from child visits, so a child reached by several paths is not over-weighted; a playout may stop at a child that already has more visits than its edge. | That backup rule and the early stop, in the opt-in graph search keyed on the evaluation cache's turn-context key. | Graph-history interaction handling: HeXO stones are never removed, so every path strictly adds stones and the graph is acyclic. |
| MCTS-Solver (Winands, Bjornsson, Saito 2008) | A proven win at the mover's choice proves the node; a loss needs every child proven lost; proven-lost children leave selection. | Exactly this (#212), with absolute winners so same-mover plies need no sign rule. | Its "average of non-proven children" fallback for losses: here a lost node simply stops the search. |
| Monte-Carlo Graph Search (Czech, Korus, Kersting 2021) | MCTS on a DAG with transposition-aware Q, a revised terminal solver, epsilon-greedy exploration and domain constraints. | Proven outcomes kept on the search structure and reused. | Its epsilon-greedy exploration: Gumbel root sampling already guarantees coverage of the sampled set. |
| PN-MCTS (Kowalski, Doe, Winands, Gorski, Soemers 2023) | Proof and disproof numbers kept in the MCTS tree steer final move selection, solving subtrees and UCB selection. | Proof numbers are implicit: `classify` settles hundreds of replies per node at once, and the solver adds certificates. | Proof-number steering in selection: every unproven node here has hundreds of unresolved replies, so disproof numbers carry no signal until the tactics oracle has pruned; revisit after measurement. |

## Outcomes in this search

**Representation.** An outcome is a winner (player id, never a sign) and a distance: the number of further placements
within which the winner can complete six from this position against any defence, stored on the node and on each edge
(edge distance = 1 + child distance). Terminal nodes have distance 0. `classify` distances are exact: a completing
edge is 1, a first stone of a two-cell completion is 2, and a forced loss adds the opponent's fastest uncovered
completion after the mover's best remaining cover. Certificates give an upper bound from `proof_turns`: the first
turn's placements plus `4 * (proof_turns - 1)` from the certificate's root. A `bound` flag marks distances that are
only upper bounds and travels with them.

**Propagation.** A node is won when any edge is won for its mover, with the least winning distance: the shortest
guaranteed win. It is lost when every legal edge is lost, with the largest distance. The same-mover case needs nothing
special because winners are absolute. Upper bounds cannot rank losses by themselves: a loose certificate bound may
hide a faster loss. So a lost node keeps as candidates every loss whose distance reaches the largest lower bound
among its losses. An exact distance is its own lower bound. A bounded loss that `classify` did not mark is not a
one-turn loss, so it resists at least the mover's remaining placements, the opponent's turn, the mover's next turn and
one more opponent stone: `remaining + 5`. Without tactics a bound's lower bound is 1.

**Selection.** Proven-lost edges are ineligible while an alternative exists; a won node offers only its shortest
winning edges; an exact node or edge ends the descent without inference; an exact root stops the search. Completed Q
is taken over eligible edges only.

**Final move and targets.** A won root plays a shortest guaranteed win. A lost root plays among the losses that may
resist longest, then by search score. A tree win shorter than a followed certificate replaces the certificate's move. The policy target of a won root covers only its shortest winning moves; a lost root records none.

**Sharing (graph search, `search_graph`, off by default).** Nodes are keyed by the evaluation cache's turn-context
key (stones by colour, mover, remaining placements, the stone placed earlier in this turn, the opponent's previous
turn; `dense_selfplay.position_key`), so the two orders of a turn meet at the next turn start and share statistics
and proofs. Backup follows the MCGS rule: a node's value is recomputed from its network value and its edges' visits
times their children's current values, and a playout stops without inference at a child that already holds more
visits than its edge. Proven outcomes are also kept by position key (stones, mover, remaining), which ignores the
turn context because the game value does not depend on it, so a new node of a proven position starts exact. A shared game
graph (`hxg_share`) keeps this store across roots: an edge takes its child's visits and value, so values found at one
root reach every stored position that leads to it, and proofs propagate to every stored parent as they do within a
search. HeXO
never removes stones, so the graph is acyclic and a position with fewer stones than the board can not recur; both
tables drop such entries when the tree advances. Keys are two independent 64-bit sums of mixed cell hashes.

**Measured** (CPU, champion `main/110000`, 20 real positions per row, sequential leaves, tree against graph):

| Simulations | Evaluations per simulation | Expanded nodes | Repeated turn contexts | Playouts reusing a transposed child | Policy entropy | Same move |
|---|---|---|---|---|---|---|
| 16 | 1.062 / 1.019 | 17.0 / 16.3 | 0.7 / 0 | 0 / 0.7 | 0.665 / 0.671 | 20/20 |
| 64 | 1.016 / 0.902 | 65.0 / 57.7 | 7.9 / 0 | 0 / 7.3 | 0.698 / 0.685 | 20/20 |
| 128 | 1.008 / 0.867 | 129.0 / 111.0 | 19.4 / 0 | 0 / 18.1 | 0.581 / 0.558 | 18/20 |

Equal-simulation CPU matches, graph against tree, same checkpoint, colour-paired openings from recent games:
- 16 simulations: 9.5/20 (0.475 ± 0.166). Games are capped at 120 placements, and 9 of them were capped.
- 64 simulations: 9.0/16 (0.562 ± 0.217). The graph used 53.8 evaluations per search against 64.1.

Together that is 18.5/36. The samples are small and show no strength difference either way, so `search_graph` stays off until the evaluator's paired comparison, for example `dense_eval.py variant ... --set search_graph=true`.

## Keeping analysis across placements

**What is kept today.** `advance` (`src/gumbel.cpp`, `Tree::advance`) moves the root to the played edge's child and keeps that whole subtree: its visits, values, network priors and proofs. The siblings of the played move are freed. They are unreachable, because stones are never removed.
- **Actors.** A game keeps one tree per distinct model (`SelfPlayGame.trees`). In self-play both colours are the same model, so the opponent's search extends the same tree the next own search reads from. Every placement advances it, both stones of a turn and the opponent's turn alike.
- **Evaluator.** One tree per model and graph setting (`MatchGame.trees`). Two checkpoints therefore keep separate trees. That is required: one network's Q estimates must not steer the other's search.
- **Play page.** Each Bubble seat and the analysis board keep one game graph per game (`GameGraph`, [neural-search.md](neural-search.md)): its store keeps the siblings of played stones and every earlier position, so analysis at any position reads the visits and values of every other searched position it reaches.
- **Evaluation cache.** An actor process holds one LRU cache of 4096 positions per model, shared by its 128 games: about 32 positions per game. It removes duplicates within a batch but is mostly evicted by the next search. The tree itself keeps the network outputs of every reachable expanded node.

**Measured reuse.** CPU, champion `main/110000`, 8 turn starts from recent games, 5 placements each (ours, ours, theirs, theirs, ours):

| Simulations | Kept after one placement | Visits at our next turn start (shared tree, self-play) | Same with one tree per side (evaluator) |
|---|---|---|---|
| 64 | 27% of root visits | 29.7 (46% of the previous turn start's search) | 4.9 (8%) |
| 128 | 24% | 43.3 (34%) | 6.4 (5%) |

The other 73 to 76% of a search's visits went to moves that were not played, and nothing can reach them again. Within a tree nothing reachable is thrown away. What the tree cannot reach are transpositions: 7.9 (64 simulations) and 19.4 (128) of the turn contexts it expands per search are repeats that the graph search shares.

**Determinism and correctness of reuse.** mctx builds a fresh tree on every call. The Gumbel MuZero paper states the algorithm for one fresh search per move. KataGo and Leela Chess Zero reuse the subtree of the played move, re-apply root noise, and keep visits and values as priors. Here reuse works the same way, and the search stays deterministic given seed and history. A shared tree is consistent for both sides: edge sums and graph node values are stored from the mover's view, and proofs carry player ids.

The reuse rules:
- `begin` redraws the root Gumbel noise and resets the per-search epochs, so sequential halving schedules the new budget from scratch.
- Retained visits and Q values act as priors: completed Q and interior selection read them.
- Proven outcomes and distances are kept verbatim.

One bias remains. The completed-Q visit scale `50 + max visits` counts retained visits, so a root that inherits many visits gets a sharper improved policy than a fresh root at the same budget. Counting only this search's visits would remove it.

**Memory.**
- **Per node.** An `Edge` is 88 bytes and a `Node` 80 bytes. An expanded node owns one edge per legal move: 640 on average in recent games (median 635, 90th percentile 980). So an expanded node costs about 56 KB.
- **Per game.** Expanded nodes held by a game's tree are roughly the root's visits: up to about 170 right after a 128-simulation search (43 retained plus 128), about 55 after a 12-simulation one. With the actors' 25% full searches that averages about 60 to 100 nodes, 3 to 6 MB per game.
- **Per actor process** (128 games): 0.4 to 0.7 GB at today's budgets, 0.8 to 1.4 GB at 2x and 1.6 to 2.8 GB at 4x. The live actor processes hold about 1.3 to 1.4 GB working set, which includes torch.
- **Evaluation cache.** About 19 KB per entry (int64 actions, float64 logits and values for 640 moves), so 78 MB per model per process at 4096 entries.
- **Keeping every node a game ever expanded** (118 placements at about 41 nodes each) would cost about 270 MB per game and 35 GB per actor process. Unreachable nodes must therefore be pruned.
- **Analytics** need only a compact summary per played position: the top 16 children's visits, Q and prior, the proof winner and distance, and the principal line. That is under 1 KB per placement, about 100 KB per game.
- **VRAM** is unaffected: network outputs are copied to host memory when a batch is collected, and nothing is kept on the device.

**Where the gains are.** Retained search is already free in self-play and gives the next search 24 to 34% extra visits. Remaining gains, in order:
1. **Transposition sharing** (the graph search above): 11% fewer evaluations per simulation at 64 simulations and 14% at 128.
2. **A persistent tree in the play server**, which today searches every placement from scratch. The same tree would give the dashboard per-position visit counts, principal lines and proof status for the whole game, at no extra search.
3. **Compact edges** (int32 offsets, float32 priors, statistics only for visited edges, about 16 bytes per legal move). This would cut node memory about five times, which is what makes 2x to 4x budgets affordable.

Reuse yields nothing for the moves not played, nothing across two different networks, and little for the 12-simulation cheap searches: their own 12 simulations add little, but they still read what the previous full search retained.

## What rows carry

Every row with `proven` +1 or -1 also carries `proof_plies`: the distance above, from that row's position, for the
side the proof favours (+1: the side to move wins within that many placements; -1: the opponent does). Sources: the
tree's exact root, a certificate (`remaining + 4 * (turns - 1)` for the attacker, `remaining + 2 + 4 * (turns - 1)`
for the defender) and forced-line rows. It is exact for terminal and tactical proofs and an upper bound whenever a
certificate is involved. Tree-proven wins also carry `proof_action`: the shortest winning moves at that
row. The learner does not read `proof_plies` yet; it is the target for a moves-left style head and for weighting
exact rows by how near the result is.

## References

- Stockfish, `src/search.cpp` (`value_to_tt`, `value_from_tt`, mate-distance pruning): https://github.com/official-stockfish/Stockfish
- Leela Chess Zero, `src/search/classic/search.cc` (`MaybeSetBounds`, `GetBestChildrenNoTemperature`), `src/search/dag_classic`: https://github.com/LeelaChessZero/lc0
- KataGo, Monte-Carlo Graph Search from First Principles: https://github.com/lightvector/KataGo/blob/master/docs/GraphSearch.md
- M. Winands, Y. Bjornsson, J.-T. Saito, Monte-Carlo Tree Search Solver, CG 2008: https://staff.ru.is/yngvi/pdf/WinandsBS08.pdf
- J. Czech, P. Korus, K. Kersting, Improving AlphaZero Using Monte-Carlo Graph Search, ICAPS 2021: https://arxiv.org/abs/2012.11045
- J. Kowalski, E. Doe, M. Winands, D. Gorski, D. Soemers, Proof Number Based Monte-Carlo Tree Search: https://arxiv.org/abs/2303.09449
