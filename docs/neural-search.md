# Search

`NeuralSearch` (`python/neural_search.py`) is the placement-level player. The tree lives in the `hexo_gumbel` library (`src/gumbel.cpp`), which compiles the exact rules and owns every node; Python only supplies network evaluations. Rebuild the library whenever its API changes.

```python
from neural_search import NeuralSearch
search = NeuralSearch(evaluator, model_version="checkpoint-sha256", history=[(0, 0)])
result = search.search(simulations=128, root_samples=16, batch_size=16)
search.advance(result["action"])
search.close()
```

`evaluator.evaluate(histories)` returns, per position, every legal cell in native order with a logit and a bounded value. It must depend only on the coloured board and the turn context, which is what the evaluation cache is keyed on: the stones with colours, the stone placed earlier this turn, the opponent's previous turn and the model version. Tree visits are never cached across positions.

## Gumbel MCTS

The root samples `root_samples` moves without replacement by Gumbel-top-k over the logits, then spends the simulation budget in sequential halving rounds; with 64 simulations and 16 samples that is 16x1, 8x2, 4x4, 2x8. Below the root, selection is deterministic: the improved policy (softmax of logits plus the completed Q) minus the visit frequency, as in DeepMind's [mctx](https://github.com/google-deepmind/mctx). Completed Q gives unvisited moves the mixed value and normalises over the eligible moves only. The chosen action is the sampled move with the highest score after the last round. The policy target covers every eligible legal move, including unvisited moves whose Q is completed with the mixed value.

Values are stored from the mover's view and flip sign only when the mover changes, so a turn's two placements back up without a sign flip. Terminal boards and the one-turn tactics oracle give exact values, and the solver adds certificates; how those propagate is in [search-outcomes.md](search-outcomes.md).

`advance(action)` keeps the played move's subtree, with its visits, values and proofs, and frees the rest. In self-play both colours share one tree per model, so the opponent's search deepens the tree the next turn reads. Fixed-budget play in `play.py` and `dense_player.py` keeps one tree for the whole turn, so the second stone's search starts from the continuation the first stone's search grew. Reused visits act as priors: the halving schedule restarts and the Gumbel noise is redrawn at every search.

`NeuralSearch(..., graph=True)` (actor and evaluation setting `search_graph`, off by default) turns the tree into a transposition graph with Monte-Carlo graph search backups, shared proofs by position and `census()` diagnostics; see [search-outcomes.md](search-outcomes.md).

`milliseconds` is a cooperative cap checked between batches. `action` is `None` only when no simulation started.

## Many games, one evaluator

`SearchCoordinator(evaluator, model_version)` runs the trees of a cohort of games together: `search_many(trees, simulations, root_samples, batch_size)` gathers leaves round-robin, deduplicates identical evaluations within a batch, and returns results in input order. All trees of a coordinator share one evaluator and model version; different checkpoints get separate coordinators. The actors and the evaluator both run this way.
