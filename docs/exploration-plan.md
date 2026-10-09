# Self-play exploration plan

KataGo spreads its self-play with seven techniques, described in its docs/KataGoMethods.md, its `cpp/configs/training/selfplay1.cfg` and its paper, arXiv 1902.10565. This note checks each one against Bubble's actors, which run Gumbel search on the hybrid scheduler, and says what we take, what we skip and why. Every change lands behind a flag whose default keeps the live run as it is.

The live actors run 128-simulation full searches on a quarter of placements and 32-simulation cheap searches on the rest. They sample 16 root moves, give 10% of the root sampling distribution to a uniform share (`root_noise 0.1`), sample a mean of 2 opening placements from the search policy, and start 25% of games from the opening book and 10% from the restart buffer.

## 1. Shaped root noise: do it

KataGo mixes 25% Dirichlet noise into the root prior. The Dirichlet concentration totals 10.83, and half of it is spread evenly over the legal moves. The other half goes to moves whose log prior lies above the mean log prior, in proportion to the excess, with priors capped at 1% first.

In Gumbel search the prior plays three roles at the root. It draws the sampled set through Gumbel-top-k, it ranks candidates in the halving rounds, and it forms the policy target together with completed Q. Today `root_noise` changes only the draw, which is the right place for noise, because the target and the final choice then still read the network's prior. Our uniform share does almost nothing for a blind spot, though. With 300 legal moves a 10% share adds 0.03% to each move, so a move at 0.05% stays far below the top 16. A Dirichlet draw at concentration 10.83 lands most of its mass on about ten moves at a few percent each, and the shaped half picks those mostly among moves the network already rates above the crowd. Those moves then enter the sampled set and get a real Q.

Change: `root_noise_concentration`. At 0, the default, the noise stays today's uniform share. Above 0 it is KataGo's shaped Dirichlet draw at that total concentration, 10.83 in KataGo. `root_noise` stays the noise weight. The draw changes only the sampling logits of the opening Gumbel-top-k, as the uniform share does now. A hybrid root starts a fresh search pass, with fresh Gumbel draws, each time its view is rescheduled, so it draws fresh noise per pass too. Descendant views copy the root's sampling settings, as they already copy `root_noise`.

## 2. Root policy temperature: do it, for sampling only

KataGo divides root log priors by 1.25 early in the game, decaying to 1.1, before adding noise. It does this for two reasons. PUCT visit counts come out sharper than the prior when moves are equally good, and the temperature counters that drift. It also keeps opening policies from settling too early.

The first reason does not apply to us. Gumbel's improved policy is softmax(logit + sigma(completed Q)). Moves with equal Q keep their prior ratios, so nothing sharpens and there is nothing to restore. Tempering the logits inside the target would flatten every generation's policy without any evidence from search. The second reason does apply, and it belongs in the same place as the noise.

Change: `root_temperature_early` and `root_temperature`, both 1, decaying with a half-life of `root_temperature_halflife` placements. KataGo's half-life on a 19x19 board is 19 moves, and a HeXO turn places two stones, so the default is 38 placements. The temperature divides the logits that the opening Gumbel-top-k draws on, before the noise mix. Halving, the final choice and the policy target keep the raw logits.

## 3. Policy target pruning: moot

KataGo's target is the visit distribution. Noise and forced playouts add visits to moves that do not deserve them, so it subtracts those visits from the target. Our target is the completed-Q improved policy, and visits enter it only through the sigma scale, (50 + max visits) * 0.1. A move that noise brings into the sampled set gets its own Q in place of the mixed value. If that Q is poor, its target mass falls below its prior, and visits never add to it.

This holds under the hybrid scheduler too. The broker builds the recorded policy with the same formula, `gumbel_broker.hpp` near line 790, over root edges whose Q comes from the shared graph's child values. The visit scale there counts the root view's direct credits only. Descendant views, which pick lines with a 20% uniform exploration share, only make child values more accurate. They do not add target mass. There is nothing to prune.

## 4. Game forking: do it

After a game ends, KataGo forks it with probability 4% early and 1% anywhere. An early fork takes the position at an exponentially distributed move with mean 2.5% of the board area, about 9 moves. A late fork takes a uniform move of the game. At that position it draws between 3 and 12 random legal moves, or up to 36 for a late fork, plays the one the value head likes best, and queues the result as the start of the next game. The forced prefix is not recorded again.

Our restart buffer already starts games from stored positions, and the book starts 25% of games off-policy. Neither tries an odd move inside the current model's own games, so this is new.

Change: `fork_early_fraction` and `fork_anywhere_fraction`, both 0, with `fork_early_plies` as the mean of the exponential fork point, `fork_early_choices` 12, `fork_anywhere_choices` 36 and `fork_min_choices` 3. Candidates are uniform random legal moves at the fork position, scored in one batch by the actor's value head from the forking side's point of view. A first stone leaves the same side to move, a second stone hands the move over. A fork game is an ordinary self-play game from the fork position, with the same full and cheap search mix, value targets and proofs as any other game. KataGo's forks are ordinary games too. Like a restart, the replayed prefix and the odd move get no rows. The episode records origin `fork` and `fork` {kind, ply, choices}, where `ply` is the index of the odd move, so the learner and the dashboard can tell fork rows apart. The manifest counts fork games, and their forced placements go into `forced_plies`. Forks from validation games are dropped, as restarts from them are, and so is a fork whose odd move ends the game, as in KataGo, or whose forced placements reach `max_plies`, as restart entries at the cap are. A finished fork game can fork again, as in KataGo. Forked starts are taken before book and restart draws, so a fork is the next game its worker starts. No opening placements are sampled after a fork, just as KataGo turns off its policy openings in forked games.

## 5. Cheap searches for value-only rows: have it

Playout cap randomization is in place: `full_fraction 0.25`, 128 against 32 simulations, and cheap rows carry no policy. We go further than KataGo here. KataGo gives cheap positions training weight 0, while we keep them as value rows at `cheap_value_weight 1`, and the training audit measured that this helps, in its finding 2 and recommendation 2. No change.

## 6. Early-move temperature and policy-sampled openings: skip

KataGo picks its played move from the visit counts at temperature 0.75, decaying to 0.15 with a half-life of 19 moves. Our actors play the Gumbel choice, the argmax of Gumbel draw + logit + sigma(Q) over the last halving round. That choice is already a sample, and its spread comes from the same draws that chose the sampled set. A visit temperature on top would count the randomness twice and break the link between the draws and the move. KataGo also opens games with a few moves sampled from the raw policy at high temperature. We leave that out. Our nearest tools differ from it. `opening_random_plies` samples the improved search policy, with no temperature, and the live book's lines come from 16-simulation searches sampled at temperature 1.5. Both spread openings, but neither draws from the raw prior. Forking covers the case KataGo's raw draws serve best, a move the policy ranks low, and it does so at every depth. If opening spread still looks thin after forking, a raw-prior temperature for `opening_random_plies` is the next thing to add.

## 7. Policy surprise weighting: do it, measured

KataGo writes each full-search position to its training data more often when its target is far from the prior. A game's full-search rows keep half their total weight uniform and get the other half in proportion to KL(target || prior). The rows are repeated, not reweighted.

The audit's finding 1 was that priority draws of exact rows over-trained the value head. Surprise weighting draws policy rows by policy evidence, but every drawn row also trains the value head, so it gets the same test as those draws.

Change: actors store `surprise`, the KL from the root's raw network prior to the recorded policy target, on every row with a policy target whose root a network evaluation expanded. A root expanded by an immediate win or a proof has no prior, so its row carries none and samples at weight 1. Storing the field changes no search, so actors do it whatever the learner flag says. The KL compares the search with the network at that position. Pair mixing and certificate policies change the trained target later in the learner, and the scalar does not follow them on purpose. The sampler asks where search found something the network missed, and that stays true whatever target the learner builds from the row. Under the live `proof_policy_missing_only` the certificate policy goes only to rows without a search policy, which carry no surprise anyway. `surprise_weight` on the learner, 0 by default, is the share of each game's full-row sampling weight that follows that KL, with KataGo at 0.5. The sampler draws rows in proportion to the weight, and rows without a stored surprise keep weight 1. Before any live use, the audit's procedure measures it. That means 2,500 steps from a fixed EMA on the same shards, outcome BCE on the panels and 64 games against main/185000, compared with a baseline arm trained the same way.

## Not applicable

Komi, handicap, rule randomization and seki forks have no counterpart, because HeXO has none of these rules. KataGo's value-surprise weighting is skipped for the reason in 7. Its visit reduction in won positions has a counterpart already, since proofs end decided games. Asymmetric playouts and side positions are outside this list and left out.

## What landed

Each change is off by default, and the live run is unchanged until a flag is set.

| technique | PR | flags, default | recommended live value |
|---|---|---|---|
| shaped root noise | #485 | `root_noise_concentration` 0 | 10.83, keeping `root_noise` 0.1 |
| root temperature | #487 | `root_temperature_early` 1, `root_temperature` 1, `root_temperature_halflife` 38 | leave at 1 |
| game forks | #486 | `fork_early_fraction` 0, `fork_anywhere_fraction` 0, `fork_early_plies` 6, `fork_early_choices` 12, `fork_anywhere_choices` 36, `fork_min_choices` 3 | 0.04 and 0.01, the rest as default |
| surprise weighting | #488 | `surprise_weight` 0 on the learner; actors store `surprise` wherever the root has a network prior | leave at 0 |

#485, #487 and #488 change native code. The main checkout's `build/libhexo_gumbel.dll` was rebuilt and swapped after each merge, so new processes load a library that matches the Python. Running processes keep their old library until they restart.

## Root sampling on real priors

This check draws the opening Gumbel-top-16 set 100 times for each of 1,500 full-search positions from new-schedule/197500's window. It uses that checkpoint's priors and the same arithmetic as `sampling()`, with no search. The columns count, per sampled set, moves with prior from 0.05% to 1%, moves below 0.05% that still lie above the mean log prior, and moves below the mean log prior, which are hopeless moves.

| setting | 0.05% to 1% | below 0.05%, above mean | hopeless |
|---|---:|---:|---:|
| no noise | 3.77 | 2.37 | 0.58 |
| uniform 0.1 (live) | 1.36 | 1.07 | 5.00 |
| uniform 0.25 | 0.80 | 1.24 | 6.08 |
| shaped 0.05 | 2.41 | 1.73 | 2.23 |
| shaped 0.1 | 2.03 | 1.88 | 2.56 |
| shaped 0.25 (KataGo) | 1.59 | 2.18 | 3.06 |
| shaped 0.1, temperature 1.25 | 2.40 | 2.04 | 2.32 |
| temperature 1.1 | 3.63 | 2.36 | 0.76 |
| temperature 1.25 | 3.39 | 2.37 | 1.06 |

The hopeless label comes from the prior, so it was checked against search. 320 positions from the newest 400 live shards were searched on the CPU at 64 simulations with 16 samples and the live uniform share of 0.1, using the same checkpoint. Each sampled move then carries its own searched Q.

| sampled moves | count | best searched Q | within 0.05 of the best |
|---|---:|---:|---:|
| below the mean log prior | 1,246 | 0.3% | 2.3% |
| below 0.05%, above the mean | 811 | 0.7% | 3.9% |
| 0.05% to 1% | 416 | 1.4% | 19.2% |
| 1% and up | 1,871 | 15.7% | 39.3% |

Moves below 0.05% are rarely competitive, whether they sit below the mean log prior or above it. Moves from 0.05% to 1% come within 0.05 of the best in one search of five. So the useful column is the 0.05% to 1% band. Counting the two groups below 0.05% as slots that search will mostly reject, the uniform 0.1 spends 6.1 of 16 samples there, shaped 0.1 spends 4.4 and no noise 2.9. Shaped 0.1 also samples 2.0 band moves against 1.4.

Gumbel-top-16 already explores. With no noise at all, nearly four of the sixteen samples come from the 0.05% to 1% band. Every kind of noise trades some of those for moves below 0.05%. The live uniform 0.1 is the worst of the realistic settings: it cuts band coverage by two thirds and fills five slots with moves below the mean log prior. Shaped noise at the same weight keeps more of the band and spends fewer slots below 0.05%, so it should replace the uniform share at the next actor restart. No noise does better still on both counts. Whether that holds up in play needs a self-play test that was not run here, so the next experiment is `root_noise 0` against shaped 0.1. Temperature alone changes little and only adds hopeless samples, so it stays at 1.

## Surprise weighting, measured

new-schedule/197500's training data from before the October 8 restart is overwritten, so the audit's seed is gone. Both arms start from the current new-schedule/197500 EMA, which was trained with the live settings. They train 2,500 steps on copies of that checkpoint's window, 1,119 shards and 870,063 policy rows, with the live learner flags (`--value-target outcome --regret-fraction 0 --cheap-value-weight 1`, pair weight 0.5, certificate policy 0.25 on missing rows only). The surprise arm adds `--surprise-weight 0.5`. The KL comes from the seed EMA's prior for every row, not from each shard's own checkpoint as KataGo does it, because most of those checkpoints were overwritten too. Mean KL is 0.57 nats, and the top 10% of rows carry 37% of it. The panels are the audit's: held-out rows, outcome BCE on unproven rows of finished games, and policy KL against the stored target for first and second stones.

| arm | vs main/185000, W-L-capped | outcome BCE A170 / B190 / H | first-stone KL A170 / B190 / H | second-stone KL A170 / B190 / H |
|---|---:|---:|---:|---:|
| seed, no training | | 0.647 / 0.650 / 0.638 | 0.775 / 0.713 / 0.757 | 0.512 / 0.483 / 0.367 |
| baseline | 41-21-2 | 0.638 / 0.642 / 0.636 | 0.778 / 0.709 / 0.754 | 0.511 / 0.481 / 0.365 |
| surprise 0.5 | 40-24-0 | 0.638 / 0.642 / 0.636 | 0.811 / 0.745 / 0.736 | 0.522 / 0.497 / 0.379 |

This arm tests a proxy. Its KL comes from one later network, so on older shards it also measures how far that network has drifted from the one that played, and it picks a different set of rows than actor-stored KL would. Within that limit, the proxy leaves outcome BCE where it is, within 0.001 on every panel, so unlike the regret draws it does not over-train the value head. It also buys nothing. The match is a tie within its noise. Policy KL against the average target rises by 0.03 nats on two of three first-stone panels and falls by 0.02 on the third. On every second-stone panel it rises by 0.01 to 0.02. That is what drawing surprising rows more often should do to a fit of the average. The audit found policy KL does not predict Elo, so that cost may not matter either, but there is no gain to pay it for. `surprise_weight` stays at 0 until an arm on shards whose actors stored their own KL shows a gain. Actors now store it, so once a window of such shards exists, that arm costs one learner flag.
