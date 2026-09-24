# Native neural placement search

`NeuralSearch` is the placement-level neural player. The separate `hexo_gumbel` library compiles the exact rules and owns a tree with edge-local visits. The old NNUE/PVS remains unchanged. No Board pointer crosses library boundaries.

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

`milliseconds` is a cooperative budget checked between batches. It cannot interrupt an in-flight evaluator call. Reports include actual elapsed time. A budget exhausted before any sampled action completes returns `action=None`. Callers must handle this explicitly. Errors cancel outstanding reservations before propagating.

Proof status is separate and currently always `UNKNOWN`. Exact terminal outcomes back up numeric values, but sampled losses never certify a position. Native tactical strategy verification is a separate integration; no principal variation or restricted negative can override this player as a proof.

Validation includes complete legal support, root sampling without replacement, phase transitions, subtree reuse, fresh budgets, pending cancellation, exact cache keys, malformed action-order rejection and first-stone terminal handling. The native test exhausts player/sign assignments for paths through four edges and checks the halving schedule and mixed-Q arithmetic. A finite independent payoff-tree test also verifies that recursive opponent replies overturn an optimistic shallow estimate. These are algorithm checks, not a claim of exhaustive solution of infinite-grid Hexo or demonstrated search-strength scaling. A diagnostic static-evaluator game finished in 29 placements with independent legality and winner validation. Frozen relational-model strength and scaling remain to be measured.
