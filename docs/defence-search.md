# Defence search on the Seal losses

The useful intervention is a small set of checked complete turns. A flipped-turn proof now supplies candidate
cells from attacker placements, including alternatives, covered defender replies and winning completions.
The candidate generator also includes open line completions on the current board, since a certificate may name
only one way to finish a line. It deduplicates these groups, keeps the certificate's first pair, then ranks pairs
by how many groups they touch. Cross-pairs use the most frequent `2 * defence_candidates` cells; the
certificate's own pairs remain available. Coordinates break ties. This bounds pair construction without
needing a network evaluation to start verification.

Each legal complete turn gets one follow-up query at the original threat budget. After our two placements,
the original attacker is the actual mover, so that query uses `attacker='mover'`. A completed UNKNOWN,
including the prover's defender-counterwin result, counts as a survivor. A timeout or verification failure
does not. This is evidence that the budget can no longer establish the threat, not a proof of safety.

At most eight turns are checked by default. Each surviving first stone enters the Gumbel opening sample set,
with a bonus of `5 * surviving_turns_for_this_stone / defence_candidates`. The total bonus is at most 5,
the initial transformed-Q range. It participates in halving, final selection and the policy target. The
sample count can rise to admit the set, while the simulation budget stays fixed. Searches with fewer
simulations than the candidate limit check at most that many turns. On the second placement, matching
survivors supply the remaining stones with the same bonus, limited to that search's simulation budget.
Exact tactical results still govern eligibility. No UNKNOWN result changes an exact value or proof label.

The bonus belongs to the current root search, so tree reuse clears it without changing stored network
logits or priors. Fixed budgets use no resident solver state or random draws for defence generation.
The disabled path preserves the existing threat ordering and episode records.

Actors use `--solver-defence --solver-threat-nodes 27000 --solver-defence-candidates 8`.
Evaluator loops use `--eval-solver-defence --eval-solver-threat-nodes 27000 --eval-solver-defence-candidates 8`.
Individual matches can use `--a-solver-defence --a-solver-threat-nodes 27000`, and likewise for side b.
All default to off. Solver status includes `defence_queries`, `defence_hits` and `defence_nodes`.

## Measurement

The input was the three recorded losses in `runs/dense-v1/evaluations/main-065000-vs-seal/report.json`.
The report was read only. No network, GPU or run process was used. Each query had 27,000 nodes, no resident
table and a 60-second safety deadline. There were no deadline or verification failures.

| Ply | Pair 15 survivors / queries | Extra nodes | Pair 28 survivors / queries | Extra nodes |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 2 / 8 | 18,464 | 8 / 8 | 2,331 |
| 11 | 0 / 8 | 6,246 | 8 / 8 | 80,269 |
| 15 | 5 / 8 | 590 | 5 / 8 | 135,413 |
| 19 | 8 / 8 | 442 | 8 / 8 | 69,536 |
| 23 | 8 / 8 | 1,022 | 1 / 8 | 29,392 |
| 27 | 2 / 3 | 4,514 | 1 / 8 | 5,621 |
| 31 | 5 / 8 | 1,034 | 1 / 8 | 4,414 |
| Total | 30 / 51 | 32,312 | 32 / 56 | 326,976 |

That finds at least one survivor in 13 of the 14 requested positions. The original threat queries cost
6,918 nodes for pair 15 and 3,636 for pair 28. The combined verification cost is 359,288 extra nodes,
about 34 times the threat-detection work. Eight is a useful bounded shortlist here, but the node cost is
substantial when several queries exhaust their budget. This measurement establishes candidate discovery,
not an improvement in match win rate.

| Candidate cap | Positions with a survivor, out of 14 | Verification queries | Extra nodes |
| ---: | ---: | ---: | ---: |
| 1 | 11 | 14 | 6,577 |
| 4 | 13 | 55 | 226,837 |
| 8 | 13 | 107 | 359,288 |

Four candidates are the better cost tradeoff on these positions: the same position coverage with 37% fewer
nodes than eight. Eight remains the requested default and finds more alternatives for the root to compare.
The one-candidate result verifies the first pair instead of assuming that occupying it is safe.

The third loss, pair 26, was checked at its own turn starts in the same interval.

| Ply | Survivors / queries | Extra nodes |
| ---: | ---: | ---: |
| 5 | 1 / 8 | 19,433 |
| 9 | 5 / 8 | 30,586 |
| 13 | 3 / 8 | 4,513 |
| 17 | 3 / 3 | 30 |
| 21 | 5 / 8 | 99,507 |
| 25 | 0 / 8 | 505 |
| 29 | 0 / 8 | 392 |
| Total | 17 / 51 | 154,966 |

The copied tactical DLL has SHA-256
`a47b611762ce82572dc103fb7257b86e93d8020cd4a3e5c7f771558ccf1c5712`.
It predates main's larger certificate limit. The measurements and local solver tests use a temporary
package containing that DLL and its matching source files from `verify-timeout`; source identity checks
remain enabled. The tactical library was not rebuilt. Only `src/gumbel.cpp` changed among native sources,
and only its C++ DLL was rebuilt, in this worktree.

Reproduce with `python tools/measure_defence.py --report <report.json> --package <matching tactical package>`.
The tool writes JSON to stdout, runs BelowNormal with `OMP_NUM_THREADS=2`, and applies a 1900 MiB Windows
process memory limit. The eight-candidate pass over all 21 positions took 21.03 seconds, peaked at 93.1 MiB
working set and 151.0 MiB committed memory, and recorded priority class 16384, BelowNormal. Repeated passes
returned the same node counts and survivor counts.

## Validation

The synthetic open-four position supplies the breaking turn `[0,2] + [5,2]`. The test checks that its first
stone enters the root even with one requested root sample, and that its second stone enters the next root
without another solver query. Other checks cover the bonus through halving and tree reuse, fixed-budget
query limits, repeated search arrays and complete seeded self-play shards.

With the flag off, a four-game CPU run produced 104 rows and byte-identical `episodes.json`, `rows.json`
and `targets.npz` against main using the supplied prebuilt libraries. The tests intentionally exclude
`test_engine`, `test_klent` and `test_gpu_nnue`.
