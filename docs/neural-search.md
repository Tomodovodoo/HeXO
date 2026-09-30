# Native neural placement search

`NeuralSearch` is the placement-level neural player. The separate `hexo_gumbel` library compiles the exact rules and owns a tree with edge-local visits. No Board pointer crosses library boundaries.

Rebuild `hexo_gumbel` when updating this Python API: completion now uses the native `hxg_done` entry.

```python
from neural_search import NeuralSearch
search = NeuralSearch(evaluator, model_version="checkpoint-sha256", history=[(0, 0)])
result = search.search(simulations=128, root_samples=16, batch_size=16)
search.advance(result["action"])
# The next search retains this placement's subtree, including its second stone.
search.close()
```

The persistent evaluator supplies `evaluate(histories)`, returning one dictionary per position. `actions` must contain every native legal cell in native sorted order, shape `[N, 2]`. `logits` and bounded `q` have shape `[N]`. Terminal positions never reach inference. Each expansion computes the state estimate as `sum(softmax(logits) * q)`. The evaluator must depend only on the colored board and turn phase, not the order of equivalent histories. The evaluator must remain frozen for this search object's lifetime. Construct a new search object for a changed model version. The cache key contains the full colored board, player, remaining placements and model version; tree visits are never cached across positions.

The root uses independent Gumbel samples and sequential halving. Its initial sample uses only logits and Gumbels even when old subtree values exist; later allocation uses completed Q values. Each new search receives its own requested number of simulations even when the subtree retains old visit statistics. The default considered-action count grows with the square root of the simulation budget; callers may increase it to the full legal count. Unselected cells remain legal and unresolved. Interior nodes select from the improved policy using mixed-value completion and visit-frequency matching. Value signs change only when the acting player changes.

The behavioral references are DeepMind Mctx's [action selection](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/action_selection.py), [sequential halving](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/seq_halving.py), and [completed Q transformation](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/qtransforms.py). This is a C++ implementation of those algorithms, with pending reservations and fresh per-search root allocation for reused trees. Pending leaves reserve their traversed edges. Batches finish a visit layer before values select the next halving layer. Batch sizes can change interior traversal as outstanding visits affect allocation.

`milliseconds` is a cooperative budget checked between batches. It cannot interrupt an in-flight evaluator call. Reports include actual elapsed time. A budget exhausted before any simulation starts returns `action=None`. When proofs remove every visited root move, the unvisited survivors compete for the final choice. Callers must handle this explicitly. Errors cancel outstanding reservations before propagating.

Verified tactical certificates and terminal outcomes supply exact absolute winners. Each backup copies a solved child's winner to its incoming edge and settles the ancestors: one winning edge proves a win for that node's mover; a loss requires every legal edge to be proven for the opponent. This uses the actual mover, including consecutive placements by the same player. Unknown and unvisited actions remain obligations even when their neural values are exactly ±1. Proven losing edges leave selection and policy while alternatives remain. Exact outcomes override earlier numerical averages in both Q transforms and reports.

An exact root stops requesting simulations immediately. Already reserved evaluations drain before `hxg_done` permits tree advancement; `completed` remains the number actually backed up, which can be below the requested cap or zero for an immediately proven root. Exact roots can select a witnessed action without any visits; actor rows record the policy of such a zero-simulation root only when it is won and its winning moves are a strict subset of the legal moves. Completed Q (mixed value, visit scale and min-max range) is taken over eligible edges only, so proven losses do not flatten the improved policy among the surviving moves. Retained winning subtrees and certified second placements survive advancement. Immediate tactical second placements can be reconstructed without inference. A principal variation or an unsuccessful solver query cannot establish an exact result; sampled losses alone do not certify a position.

Actors and evaluators use this completion condition. Actor rows and root values retain the exact outcome, including with a solver Plan active. With proven adjudication enabled, a tree proof can end the game; certificate-generated continuation rows still require an actual strategy certificate. A tree-derived proof does not claim a solver certificate or a measured proof length (`proof_turns=0`).

Validation includes complete legal support, root sampling without replacement, phase transitions, subtree reuse, fresh budgets, pending cancellation, exact cache keys, malformed action-order rejection and first-stone terminal handling. The native test exhausts player/sign assignments for paths through four edges and checks the halving schedule and mixed-Q arithmetic. A finite independent payoff-tree test also verifies that recursive opponent replies overturn an optimistic shallow estimate. These are algorithm checks, not a claim of exhaustive solution of infinite-grid Hexo or demonstrated search-strength scaling. A diagnostic static-evaluator game finished in 29 placements with independent legality and winner validation. Frozen relational-model strength and scaling remain to be measured.

## Multiple games on one inference worker

Keep one `SearchCoordinator(evaluator, model_version)` alive for a cohort of games. `search_many(trees, simulations=[128, 256], root_samples=16, batch_size=32, milliseconds=[100, 200])` returns results in input-tree order. Scalar budgets apply to every tree. After applying each result, call that tree's `advance(action)` and reuse the coordinator for the next cohort.

The coordinator gathers leaves round-robin across native trees. It deduplicates identical neural evaluations within a batch without sharing tree visits. Request identifiers are routed together with their tree, so equal numeric request IDs from different games cannot collide. One tree reaching its deadline does not cancel another; inference failures cancel every outstanding reservation in the cohort. In-flight inference is still cooperative and its actual time is reported. `last_stats` records inference batch count, unique evaluated positions and largest batch. Per-tree `evaluated` counts fulfilled requests, including requests sharing one inference result.

All trees in a coordinator must use the same evaluator object and model version. Partition different checkpoints into separate coordinators. The evaluator controls token/edge work limits inside each requested position batch. Single-tree `search()` uses this same coordinator path.
