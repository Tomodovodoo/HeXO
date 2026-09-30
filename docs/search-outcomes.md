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
| Leela Chess Zero (classic search) | Nodes keep a lower and an upper game-result bound; `MaybeSetBounds` combines them from the children with the sign flip, so a child loss makes the parent a win and all-child wins make it a loss ("certainty propagation"). With `StickyEndgames` a newly terminal node updates its ancestors at once. Final move choice prefers the shortest terminal win and the longest terminal loss by the moves-left estimate `M`. A separate `dag_classic` backend shares transposed nodes. ("Solid" in lc0 is a memory layout of the children, not a proof concept.) | Win/loss results propagated at backup (#212), shortest win and longest loss at the root. | Draw bounds: HeXO has no draw except the ply cap, which proofs ignore (a proven win longer than the remaining cap is reported but not realisable). |
| KataGo | Terminal positions are leaves with a fixed utility; there is no solver. Its graph search (docs/GraphSearch.md) derives the MCGS backup from first principles: `Q(n) = (U(n) + sum_a N(n,a) Q(child_a)) / (1 + sum_a N(n,a))`, with edge visits `N(n,a)` kept separate from child visits, so a child reached by several paths is not over-weighted; a playout may stop at a child that already has more visits than its edge. | Candidate for transposition sharing, keyed on the evaluation cache's turn-context key. | Graph-history interaction handling: HeXO stones are never removed, so every path strictly adds stones and the graph is acyclic. |
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
