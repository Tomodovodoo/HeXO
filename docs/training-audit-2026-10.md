# Training audit, October 2026

This audit covers the dense learner on `runs/dense-v1` after `main/185000`, which nothing trained with the old settings beat. Up to the settings switch on 2026-10-08, every later export lost to it. It uses the run's shards, manifests, logs and league, plus five fixed panels of 6,000 held-out rows each (validation-split games, never trained on). P150 is main 150k to 157.5k, A170 is main 170k to 182.5k, B190 is main 190k to 200k, R is the `reset` actors (128/32) and H is the hybrid actor. Every rated checkpoint since 150k was scored on every panel. The scores are outcome BCE on unproven rows of finished games, BCE on exact-labelled rows, and policy KL against the stored search policy. Ablation arms each trained 2,500 steps from new-schedule/197500's EMA on one fixed shard set and seed, then played 64 games against main/185000 at the evaluator's settings.

## The decline is real, and it is in the value head

185000 is not a lucky snapshot. Over two matches it beat 182500 142 to 113. The main successors from 187.5k to 200k scored 477 to 596 against it over 1,073 games, which is -38 Elo. 182500, which is not the champion, also beat 202500 (36 to 27) and 205000 (34 to 27), so a book biased toward the champion cannot explain the drop.

Across the 18 checkpoints with a direct result against 185000, Elo correlates with mean unproven-outcome BCE on the A170, B190, R and H panels at r = -0.79, about -50 Elo per +0.01 nat. It correlates with exact-row BCE at r = +0.79: the better a checkpoint fits proven rows, the weaker it plays. Policy KL does not predict Elo (r = -0.06 for first stones). Checkpoints trained on B190-era games predict that era's outcomes no better than 165000 (0.668 against 0.664), while their policy KL on the same rows kept falling (0.764 to 0.678). From 175k to 215k the training value BCE against the calibrated target fell from 0.465 to 0.435, and BCE against the outcome rose from 0.648 to 0.671.

| checkpoint | Elo vs 185000 | outcome BCE A170 / B190 | exact BCE A170 / B190 |
|---|---:|---:|---:|
| main/165000 | | 0.658 / 0.664 | 0.109 / 0.141 |
| main/185000 | 0 | 0.664 / 0.668 | 0.099 / 0.128 |
| main/200000 | -50 (244 games) | 0.670 / 0.668 | 0.095 / 0.118 |
| main/215000 | -124 (98) | 0.698 / 0.689 | 0.069 / 0.084 |
| new-schedule/197500 | -119 (208) | 0.691 / 0.684 | 0.075 / 0.090 |

## Ablations

| arm (2,500 steps from 197500 EMA) | vs main/185000, W-L-capped | outcome BCE A170 / B190 / H | exact BCE B190 |
|---|---:|---:|---:|
| seed, no training | 16-46-2 | 0.691 / 0.684 / 0.660 | 0.090 |
| current settings | 12-49-3 | 0.689 / 0.685 / 0.658 | 0.095 |
| `--value-target outcome` | 39-21-4 | 0.675 / 0.669 / 0.651 | 0.109 |
| `--regret-fraction 0` | 31-27-6 | 0.666 / 0.669 / 0.645 | 0.126 |
| both | 43-17-4 | 0.652 / 0.655 / 0.641 | 0.149 |
| both, `--cheap-value-weight 1` | 43-21-0 | 0.641 / 0.646 / 0.637 | 0.181 |
| main 175k to 185k shards only | 35-23-6 | 0.672 / 0.670 / 0.651 | 0.113 |
| main 190k to 200k shards only | 33-29-2 | 0.674 / 0.672 / 0.653 | 0.113 |

Each match has 64 games, so a single arm's Elo is only good to about 80 either way. The panel scores move the same way on every panel, and the value-target and regret changes add up. The two era arms are alike on every panel, so the actors' data after 185000 is not the cause. Both era arms beat the current-settings arm, which fits the rest of the table: their windows hold one tenth as many regret rows (2 to 3% of each batch against 8%) and no hybrid rows.

## Findings, ranked

**1. Regret sampling over-trains tactical rows (remove).** In the arms it is the largest single effect on outcome BCE (-0.013 to -0.022 against the current-settings arm). The sampler draws 8% of every batch from 64,346 priority rows in the 197500 window. At least 44,000 of those are certified rows whose prediction missed the proven result. The rest are matches from the 20,000-entry restart buffer, 71% of them `defence` entries at lookback plies, which may be unproven. The two kinds were not ablated separately. The 4x cap binds on every one of them, so regret magnitudes have no effect, and every priority row is drawn at exactly 4/W. Exact rows already carry about a third of the value loss (in the outcome arm, value BCE 0.46 mixes exact rows near 0.09 with outcome rows at 0.65), and they are excluded from the outcome loss.

**2. The calibrated value target carries almost no outcome information (switch to `outcome`).** A finished game's target is the fitted P(win | v, h). Here v is the last full-search root value at or before the row (`carried_values`), and h is plies to the end. Over 2,500 games per era, unproven rows more than 64 plies from the end (52% of them) get 0.50 whatever happens. Their BCE against the outcome equals the outcome entropy, 0.69. A cheap row's target comes from a full search a median 6 plies earlier (90th percentile 18), and 7% have none and get the base rate. Refitting the calibration on other inputs gives these held-out BCEs on cheap rows: carried value 0.60, the row's own cheap root value 0.58, the next full search 0.55. `outcome_lambda` acts only with `value_target td`, so it does nothing in this run. `td_lambda` labels capped games only (0 to 6%). With `--value-target outcome`, a check on 1,426 hybrid rows found every finished unproven row's target equal to its outcome. No calibration or outcome-lambda blend remains. The fit itself is stable: target sd over the last 20 exports is 0.004 to 0.012. It just knows little.

**3. One calibration fit mixes two meanings of h (moot under `outcome`).** Legacy games end at six in a row after about 14 forced-line rows. Hybrid games end at the proof, so h is 13 to 26 plies shorter. The legacy fit's intercept at h = 1 is +3.4 (the mover wins). At new-schedule/197500 it was -0.4. The legacy map scores hybrid rows 0.03 to 0.10 nats worse than a hybrid-only fit for h up to 32. If calibration returns, fit it per game-end kind.

**4. Hybrid rows lose most tactical supervision (test).** Exact rows are 3.9% of hybrid rows against 15.0% in A170, and line rows went from 14.4 per game to none. `proof_policy_missing_only` gives the certificate policy to proven winning rows without a search policy: forced-line rows and cheap proven roots. Those rows fell from 8.2 per game (8.1% of rows) in A170 to 2.0 per game (2.1%) in H. On proven wins the hybrid first-stone target puts 0.56 of its mass on certificate stones (top-1 0.65), against 0.90 to 0.98 in legacy rows. The recorded policy is the root's improved policy over root edges, not a view aggregate. The root view gets 32 of a full search's 128 simulations, and the top move's median direct credit is 4 (12 in cheap searches). The sigma scale (50 + max credits) * 0.1 is therefore about 5.4. Hybrid first-stone targets come out flatter (entropy 1.48 against 1.02, top mass 0.52 against 0.64). Second stones are unchanged.

**5. Raw weights trail the EMA (keep EMA, never use raw).** Raw weights lose 0.013 to 0.023 nats of outcome BCE against their own EMA at 182500, 195000 and 197500, which is the ~70 Elo gap at the slope above. At the evaluator's settings the raw weights scored 25-35-4 against their EMA at 182500 and 26-38-0 at 195000, which is -54 and -65 Elo. 185000's raw weights are unusually close to its EMA (0.003 to 0.008). EMA 0.999 averages about 1,000 steps, 0.4 of an export interval, so it does not span the interval.

**6. Pair mixing rewrites a third of first-stone targets (inconclusive).** At weight 0.5 it changes the argmax of 27 to 40% of first-stone targets. After mixing, the played second stone is the top move in 41 to 52% of rows. Policy KL kept improving after it was switched on and does not track Elo. A 0 against 0.5 arm would decide it.

**7. Not the cause.** Window and pacing: a retained row is drawn about 2.75 times over a window of about 29,000 steps. 52% of the 197500 window came from models 65 to 186 Elo below the champion, but the era arms show no data effect, so an Elo filter has no support. Policy plateau: target mass outside the 16 sampled moves is 0.2 to 1.3%. The moving-window CE of 1.4 is target entropy 0.8 plus KL 0.6, and fixed-panel KL keeps falling, so root sampling is not capping it. Cheap rows score the same held-out outcome BCE as full rows (0.64 to 0.69). Gradient norms are 2.5 to 4.5 against `grad_clip 1.0`, so every step is clipped. Under Adam that mostly reweights steps, and its effect is untested.

## Code review

I found no sign or perspective errors. These read correctly: `player_at`, `carried_values` and `value_targets` by player and stone, the short-value and masked-future targets, the backward TD recursion, the cheap-row and proof-row weights, pair-mix normalisation, legal masking, policy CE with far cells, and per-head weighting across buckets. AdamW applies no decay to norm or bias parameters, a resume with the same optimizer kind skips warmup (a changed kind restarts it), and the EMA averages parameters only and rebuilds norm statistics at export.

PRs #430 to #449 were checked against reference computations on real shards. Regret priorities match the old code at all 2,052 positions. The receiver thread drops and repeats no batch. The rebuild rule gives the same index as a fresh window. The shard index matches 40 real shards, sidecars included. The fused kernels agree with the reference ops to bf16 rounding, though only on CPU through Triton's interpreter.

One small effect remains. Each padding row (`pad`, QUANTUM 4) adds one crop cell to the masked norm statistics in training and in EMA recalibration. That moves a bucket's norm variance by up to 4.8% and the gradient by 0.6 to 2.7%. The fix is to leave pad cells out of the norm mask and cell count while keeping the pooling count at 1. With regret sampling off, two older behaviours stop mattering: retention dropping 378 of 1,335 restart-buffer rows, and the saturated 4x cap.

## Recommendations for the live run

1. `--value-target outcome --regret-fraction 0`, applied at 15:10 UTC on 2026-10-08 with a restart from main/185000's EMA. Its exports already bear this out. new-schedule/205000 beat main/185000 48-15-1 and became champion. 207500 scored 39-25-0 against 185000. Keep these settings.
2. `--cheap-value-weight 1`. Under the outcome target a cheap row's label is as good as a full row's. The arm cut outcome BCE by a further 0.004 to 0.011 on every panel, and its match tied the arm without it (43-21-0 against 43-17-4). Switch at an export boundary and watch exact-row BCE, which rose to 0.18.
3. Leave `td_lambda`, `outcome_lambda` and `bootstrap_full_only` as they are. Under `outcome`, `outcome_lambda` is inactive and `td_lambda` labels capped games only. `bootstrap_full_only` still limits capped-game chains and the short-value head's `average` target to full-search root values.
4. Next tests, one arm each: pair policy weight 0 against 0.5, and certificate policy on all winning rows (`--no-proof-policy-missing-only`; a resume keeps the saved setting unless that flag is given) for hybrid data.
5. Keep EMA 0.999. Start seeds from another checkpoint's EMA weights, as #476 does, and never evaluate or serve raw weights. An ordinary resume of the same variant should keep loading `model.pt` with its optimizer state and `ema.pt` alongside it, because the saved moments belong to the raw parameters.
