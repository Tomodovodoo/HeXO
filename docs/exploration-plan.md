# Self-play exploration plan

KataGo spreads its self-play with seven techniques, described in its docs/KataGoMethods.md, its `cpp/configs/training/selfplay1.cfg` and its paper, arXiv 1902.10565. This note checks each one against Bubble's actors, which run Gumbel search on the hybrid scheduler, and says what we take, what we skip and why. Every change lands behind a flag whose default keeps the live run as it is.

The live actors run 128-simulation full searches on a quarter of placements and 32-simulation cheap searches on the rest. They sample 16 root moves, give 10% of the root sampling distribution to a uniform share (`root_noise 0.1`), sample a mean of 2 opening placements from the search policy, and start 25% of games from the opening book and 10% from the restart buffer.

## 1. Shaped root noise: do it

KataGo mixes 25% Dirichlet noise into the root prior. The Dirichlet concentration totals 10.83, and half of it is spread evenly over the legal moves. The other half goes to moves whose log prior lies above the mean log prior, in proportion to the excess, with priors capped at 1% first.

In Gumbel search the prior plays three roles at the root. It draws the sampled set through Gumbel-top-k, it ranks candidates in the halving rounds, and it forms the policy target together with completed Q. Today `root_noise` changes only the draw, which is the right place for noise, because the target and the final choice then still read the network's prior. Our uniform share does almost nothing for a blind spot, though. With 300 legal moves a 10% share adds 0.03% to each move, so a move at 0.05% stays far below the top 16. A Dirichlet draw at concentration 10.83 lands most of its mass on about ten moves at a few percent each, and the shaped half picks those mostly among moves the network already rates above the crowd. Those moves then enter the sampled set and get a real Q.

Change: `root_noise_shape` with `uniform`, the default and today's mix, or `dirichlet`, KataGo's shaped draw at total concentration `root_noise_concentration`, 10.83 by default. `root_noise` stays the noise weight. The draw changes only the sampling logits of the opening Gumbel-top-k, as the uniform share does now. A hybrid root starts a fresh search pass, with fresh Gumbel draws, each time its view is rescheduled, so it draws fresh noise per pass too. Descendant views copy the root's sampling settings, as they already copy `root_noise`.

## 2. Root policy temperature: do it, for sampling only

KataGo divides root log priors by 1.25 early in the game, decaying to 1.1, before adding noise. It does this for two reasons. PUCT visit counts come out sharper than the prior when moves are equally good, and the temperature counters that drift. It also keeps opening policies from settling too early.

The first reason does not apply to us. Gumbel's improved policy is softmax(logit + sigma(completed Q)). Moves with equal Q keep their prior ratios, so nothing sharpens and there is nothing to restore. Tempering the logits inside the target would flatten every generation's policy without any evidence from search. The second reason does apply, and it belongs in the same place as the noise.

Change: `root_temperature_early` and `root_temperature`, both 1, decaying with a half-life of `root_temperature_halflife` placements. Its default of 19 copies KataGo's 19 turns on a 19x19 board. The temperature divides the logits that the opening Gumbel-top-k draws on, before the noise mix. Halving, the final choice and the policy target keep the raw logits.

## 3. Policy target pruning: moot

KataGo's target is the visit distribution. Noise and forced playouts add visits to moves that do not deserve them, so it subtracts those visits from the target. Our target is the completed-Q improved policy, and visits enter it only through the sigma scale, (50 + max visits) * 0.1. A move that noise brings into the sampled set gets its own Q in place of the mixed value. If that Q is poor, its target mass falls below its prior, and visits never add to it.

This holds under the hybrid scheduler too. The broker builds the recorded policy with the same formula, `gumbel_broker.hpp` near line 790, over root edges whose Q comes from the shared graph's child values. The visit scale there counts the root view's direct credits only. Descendant views, which pick lines with a 20% uniform exploration share, only make child values more accurate. They do not add target mass. There is nothing to prune.

## 4. Game forking: do it

After a game ends, KataGo forks it with probability 4% early and 1% anywhere. An early fork takes the position at an exponentially distributed move with mean 2.5% of the board area, about 9 moves. A late fork takes a uniform move of the game. At that position it draws between 3 and 12 random legal moves, or up to 36 for a late fork, plays the one the value head likes best, and queues the result as the start of the next game. The forced prefix is not recorded again.

Our restart buffer already starts games from stored positions, and the book starts 25% of games off-policy. Neither tries an odd move inside the current model's own games, so this is new.

Change: `fork_early_fraction` and `fork_anywhere_fraction`, both 0, with `fork_early_plies` as the mean of the exponential fork point, `fork_early_choices` 12, `fork_anywhere_choices` 36 and `fork_min_choices` 3. Candidates are uniform random legal moves at the fork position, scored in one batch by the actor's value head from the forking side's point of view. A first stone leaves the same side to move, a second stone hands the move over. A fork game is an ordinary self-play game from the fork position, with the same full and cheap search mix, value targets and proofs as any other game. KataGo's forks are ordinary games too. Like a restart, the replayed prefix and the odd move get no rows. The episode records origin `fork` and `fork` {kind, ply, choices}, where `ply` is the index of the odd move, so the learner and the dashboard can tell fork rows apart. The manifest counts fork games, and their forced placements go into `forced_plies`. Forks from validation games are dropped, as restarts from them are. A finished fork game can fork again, as in KataGo. Forked starts are taken before book and restart draws, so a fork is the next game its worker starts. No opening placements are sampled after a fork, just as KataGo turns off its policy openings in forked games.

## 5. Cheap searches for value-only rows: have it

Playout cap randomization is in place: `full_fraction 0.25`, 128 against 32 simulations, and cheap rows carry no policy. We go further than KataGo here. KataGo gives cheap positions training weight 0, while we keep them as value rows at `cheap_value_weight 1`, and the training audit measured that this helps, in its finding 2 and recommendation 2. No change.

## 6. Early-move temperature and policy-sampled openings: skip

KataGo picks its played move from the visit counts at temperature 0.75, decaying to 0.15 with a half-life of 19 moves. Our actors play the Gumbel choice, the argmax of Gumbel draw + logit + sigma(Q) over the last halving round. That choice is already a sample, and its spread comes from the same draws that chose the sampled set. A visit temperature on top would count the randomness twice and break the link between the draws and the move. KataGo also opens games with a few moves sampled from the raw policy at high temperature. `opening_random_plies` and the 25% book share cover that. No change.

## 7. Policy surprise weighting: do it, measured

KataGo writes each full-search position to its training data more often when its target is far from the prior. A game's full-search rows keep half their total weight uniform and get the other half in proportion to KL(target || prior). The rows are repeated, not reweighted.

The audit's finding 1 was that priority draws of exact rows over-trained the value head. Surprise weighting draws policy rows by policy evidence, but every drawn row also trains the value head, so it gets the same test as those draws.

Change: actors store `surprise`, the KL from the root's raw network prior to the recorded policy target, on every row with a policy target. The actor stores the field always, since it changes no search. `surprise_weight` on the learner, 0 by default, is the share of each game's full-row sampling weight that follows that KL, with KataGo at 0.5. The sampler draws rows in proportion to the weight, and rows without a stored surprise keep weight 1. Before any live use, the audit's procedure measures it. That means 2,500 steps from a fixed EMA on the same shards, with the KL taken from each shard's own checkpoint, outcome BCE on the panels and 64 games against main/185000, compared with a baseline arm trained the same way.

## Not applicable

Komi, handicap, rule randomization and seki forks have no counterpart, because HeXO has none of these rules. KataGo's value-surprise weighting is skipped for the reason in 7. Its visit reduction in won positions has a counterpart already, since proofs end decided games. Asymmetric playouts and side positions are outside this list and left out.

## Order

This note lands first. Shaped noise, root temperature, forking and surprise weighting follow as separate pull requests, each with a behaviour test. A switch happens only at an export boundary.
