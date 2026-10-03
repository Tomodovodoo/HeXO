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

`advance(action)` keeps the played move's subtree, with its visits, values and proofs, and frees the rest. In self-play both colours share one tree per model, so the opponent's search deepens the tree the next turn reads. A Bubble seat on the play page, served by `play.py` or in the browser, keeps a game graph (below) across turns until undo, a new game or a seat change, and so does the page's analysis until undo or a new game; `dense_player.py` keeps one tree for the whole turn, so the second stone's search starts from the continuation the first stone's search grew. Reused visits act as priors: the halving schedule restarts and the Gumbel noise is redrawn at every search.

`NeuralSearch(..., graph=True)` (actor and evaluation setting `search_graph`, off by default) turns the tree into a transposition graph with Monte-Carlo graph search backups, shared proofs by position and `census()` diagnostics; see [search-outcomes.md](search-outcomes.md).

`NeuralSearch(..., q_range_floor=x)` (actor and evaluation setting `q_range_floor`, 0 by default, or a `q_range_floor` field in a Bubble entry's `models/<name>.json` for the play page) rescales the completed Q by at least x instead of its own min-max range, so a root whose moves differ by less than x keeps a flatter target. Without it a spread made of estimation noise gets the same confident target as a real gap: on checkpoint 200000, in the half of 40 positions whose Q spread was below the median of 0.24 (value scale -1 to 1), two independent searches chose different top target moves 25 percent of the time at 128 simulations and 35 percent at 256. A floor of 0.5 raised top-move agreement over all 40 positions from 0.80 to 0.875 at 128 simulations and from 0.825 to 1.00 at 256, still moved the target's top move away from the prior's in 28 percent of them, and picked solver-certified winning moves as often as the plain rescale.

`NeuralSearch(..., root_noise=e)` (actor setting `root_noise`, 0 by default, `--root-noise` on dense_selfplay, applied to full searches only) draws the root's Gumbel-top-k candidates from (1 - e) p + e / N over the N eligible moves instead of the prior p. Halving, the final choice, the completed Q, the improved policy target and every node below the root keep the network's logits, so the noise changes which moves get searched but not how a searched move is scored. A policy audit found the search's chosen move inside the prior's top 10 in 98.6 percent of positions, so a move the prior ranks low was never evaluated and never entered a target; the uniform share lets self-play search some of them without extra network calls.

`milliseconds` is a cooperative cap checked between batches. `action` is `None` only when no simulation started.

## Game graphs

`GameGraph(evaluator, model_version, history, limit=4096)` is a tree whose node store outlives a root (native `hxg_share`). Nodes are keyed by turn context, the evaluation cache's key, and the store keeps each one with its visits, value, exact marks and proof distances until it is evicted. `at(history)` moves the root to any position, stored or new; a new node joins every stored node it extends by one stone, in both orders of a completed turn. `advance` keeps the siblings of the played stone. A search reads each child in the store as its edge: the edge's visits are the child's and its value is the child's value, so a 2048-simulation search at C gives B's edge to C those visits and that value, and B's value, a visit-weighted mean of its children (the graph search's backup), carries them to A's edge to B. Each playout also counts one visit at every stored position the root's history passes through, so A's edge to B holds C's visits too. A node attached later under a new parent adds none of its visits to that parent or to the positions before it; its own searches already counted them wherever those positions were stored. A node whose descendants changed is marked stale and recomputed when it is next read. Between searches the store keeps at most `limit` expanded nodes (0: no bound): the least recently visited nodes without children leave first, down to seven eighths of the limit, and their last visits and value stay on the parent's edge, where a node later created for that edge takes them over. Without `hxg_share` nothing of this runs and a tree searches bit for bit as before.

`GameGraph.search(..., pv_check=f)` (0 by default) adds a principal-variation check. With R = round(f * S) for a budget of S simulations and 0 <= f < 0.5:

1. The root A is searched with S - 2R simulations, and the search chooses its stone.
2. The turn is that stone, then, while the same side is still to move, the stone the graph's improved policy ranks first at the next position. The root moves to the position C after the turn, which is searched with R simulations.
3. Back at A, when the chosen stone's completed Q (value units, -1 to 1) fell by more than 0.05 (`PV_DROP`), A is searched again with the last R simulations; otherwise they are not spent.

No check runs when R is 0, when S - 2R would be below 1, when the first search proved its root, when the position after its stone was never expanded while the same side still moves, or when the turn wins. The result is A's after the check: its policy, values, choice and `completed` (all passes), plus `pv_check` (the turn, the stone's Q before and after, and whether A was searched again). The play page and analysis use f = 0.25 (`play.PV_CHECK`); actors and the evaluator do not check.

Actors take both behind settings that are off by default: `game_graph` (`--game-graph N` on dense_selfplay) makes each game's trees game graphs keeping at most N expanded nodes, and `pv_check` (`--pv-check f`, needs `game_graph`) checks every full search. A checked search's row records the policy target and value of the root after the check and any second search, never of its first pass, and keeps the first pass's solver node counts; its proof fields come from the root after the check.

## Many games, one evaluator

`SearchCoordinator(evaluator, model_version)` runs the trees of a cohort of games together: `search_many(trees, simulations, root_samples, batch_size)` gathers leaves round-robin, deduplicates identical evaluations within a batch, and returns results in input order. All trees of a coordinator share one evaluator and model version; different checkpoints get separate coordinators. The actors and the evaluator both run this way.
