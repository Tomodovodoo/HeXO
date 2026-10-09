# Local proof reuse

The Python player and browser Bubble player enable this path by default,
including analysis, saved-game review and clocked games. Use
`python/play.py --no-proof-stamps` or `Bubble.turn(..., {proofStamps: false})`
to disable it. Direct solver calls remain opt-in: the Python solver API accepts
`stamps=True`, and the browser solver accepts the same option.

A stamp is a complete checked strategy plus conditions under which that same
strategy still wins. It is not a cached evaluation or a verdict copied from a
shape catalogue. Both levels of PDS-PN consult stamps as terminal game outcomes,
alongside the exact graph facts introduced for issue #347. Search tables and the
whole-position result cache are isolated from this optional path.

## What a proof needs

The compiler walks the primary strategy after the raw-coordinate checker has
verified it. It keeps the initial friendly stones supporting its winning
windows and legal attack placements, and marks every initially empty played
cell and completion gap as protected. Every defender branch must spend all its
remaining placements answering the attack. A strategy that grants an
unaccounted free stone cannot become a forcing stamp.

There is no fixed spatial radius. For every defender turn, consider each
six-cell window touched by a defensive stone placed earlier in the strategy.
If an earlier guaranteed friendly placement blocks the window, it is safe.
Otherwise, with `k` such defensive stones and `r` placements left this turn,
the initial board may contain at most `5 - r - k` enemy stones in that window,
unless an initial friendly stone blocks it. These are the local counter-threat
guards. Windows untouched by a later defensive stone are checked globally,
after the friendly prefix common to every defender branch. This catches a
counterwin on the other side of the board as well as one next to the shape.

These conditions preserve at least the source proof's attacker threats.
Additional friendly threats can shrink the defender's reply set. They cannot
add a missing reply. Protected cells keep the recorded moves available, and
the counter-threat guards prevent a defender from winning instead of answering.
The resulting induction follows the checked strategy to six in a row.

The compiler shares identical proof branches before applying the strategy size
limit. The checker still verifies each branch in its own board context.

The compiler also tries the proof again with only its supporting stones left.
It uses that smaller source only if its stamp still matches the original board.
Removing enemy blockers can otherwise create a shorter strategy that cannot
work with those blockers present. A composed strategy keeps its checked child
stamps. Their required and protected cells become parent conditions. Guards
subtract enemy stones added along the parent path, and guaranteed friendly
prefixes block the corresponding counter-threat windows. The raw checker
still checks the complete composition. Nested sources have a limit of 32;
the existing source size and library accounting limits also apply.

## Replaying an analysed variation

Browser analysis retains certificates and recorded continuations across
refresh and reload. If the attacker has the same stones at a saved root or
along a recorded continuation, these records can supply attack suggestions
on the changed board. The replay follows legal suggested attacks and derives
every required defensive cover from the current board. A changed counterwin,
missing reply or quiet unresolved continuation leaves the result UNKNOWN.
Saved exact verdicts never prove a changed board.

During replay, complete sub-strategies become checked local stamps. Subsequent
defensive branches can reuse those stamps after checking their conditions.
This avoids replaying every stone of the same forcing continuation for every
irrelevant defence. Composing their conditions avoids expanding that repeated
work again when the parent strategy becomes a stamp.

The optional tactical API argument is `replay`, a list of records containing
`history`, `winner`, `pv` and optionally `certificate`. It requires `stamps`.
At most 256 records and 50,000 history/PV cells are accepted. Evidence scanning
has a 200,000-node limit, and replay shares the query's node meter and deadline.
The browser tries at most 20,000 nodes and 15 seconds per candidate winner
before falling back to ordinary search with its full configured budget. Replay
work is reported in the total used nodes but does not reduce that budget.
This archive replay runs in untimed
analysis and review; ordinary play and timed games keep their existing path.
Disabling proof stamps also disables archive replay.

The result is a standalone certificate. Refreshes that keep a tighter scalar
proof retain the previous strategy separately for later replay. Its longer
bound does not replace the tighter displayed result.

## Tempo and identity

The key includes the required stones, protected cells, counter-threat guards,
the common friendly prefix, the winner relative to the mover, and whether one
or two placements remain. Geometry is canonicalized over translations and all
12 hex symmetries. Small primitives also match after exchanging colours.
Keys organize proofs; a hash match never establishes an outcome.

An extra tempo is an actual game transition that has been proved. No query
grants a player three placements on a normal turn. A failed forcing search is
UNKNOWN, including when it finds a plausible holding reply. It is never a
proof of a draw or an opponent win.

## Proving a quiet defender turn

For a checked attacker strategy, derive the cells where up to the defender's
remaining one or two placements could invalidate it. This region contains the
protected cells and the empty cells of every counter-threat window whose
allowed count could be exceeded. It includes relevant remote threats.

Every defender turn wholly outside the region preserves the fallback strategy.
For every cell inside it, the solver separately proves the position after one
defensive stone there. If a stone remains, it repeats the procedure with a new
checked fallback and region. If none remains, it asks the ordinary attacker
prover. Existing exact graph facts can close any of these obligations.

Within a two-placement turn the stones commute. A relevant second stone is
therefore also considered first. To make this sound even when the other stone
extends the legal frontier, the proof checker permits these abstract defensive
first stones beyond the current frontier: it proves a superset of legal turns.
Immediate defender wins are checked before making this split. A deadline,
missing branch or unproved continuation leaves the root UNKNOWN.

The certificate records the split, its fallback strategy, and every relevant
reply. Repeated strategies share certificate nodes. The native checker derives
the region again. The independent Python checker instead replays each fallback
with universally quantified enemy stones outside the supplied region, checking
attack cells and every possible counterwin window. It does not trust the native
compiler's masks or its danger calculation. This also works for a proposed quiet
attacking turn supplied through `root_moves`.

## Positions the mover loses

A stamp is a local shape, not a remembered position. A position that the side to move loses is proved by the
`defender` query (the section above): a fallback win for the opponent, then every reply that could disturb it. The
query learns stamps as it goes, and the worker keeps them, so a query that runs out of nodes leaves the next one
closer. After 25. [4,7] of `tests/fixtures/lost-quiet-defence.htttx` the first `defender` query needs about 10,500
nodes on a cold worker (4,096 fails, 8,192 fails, 16,384 proves in 2.9 s); a worker that has already tried at 4,096
proves it on the fourth try, and the same query on a worker that holds the proof takes 60 nodes. The positions
after the opponent's replies are wins for the opponent that the stamps recognise at once (59 nodes, 0.3 s).

Analysis asks for this proof in three places. A Standard analysis (`solve`, and its browser twin in `worker.mjs`)
asks it after the opponent's threat is verified, and when a query without a clock spends all its nodes it asks
once more with four times as many (at most 65,536). Deep Solve asks it as soon as the root has no forcing win,
in rounds from 16,384 nodes. A timed turn keeps its single query inside its clock. Before this change Deep Solve
asked only for a win of the side to move, so a lost position ran the frontier for the full two minutes (6.9 million
frontier nodes, no proof) while the stamps that proved it in about five seconds were never used.

## Library and limits

`tools/tactical/stamps.json` contains primitive strategies produced by HeXO.
`tools/proof_stamps.py --out PATH` regenerates them and runs the independent
checker. The native worker checks these sources on first use. If a short query
expires during loading, the next query resumes at the first unfinished entry. An explicit
`library` list supplies other checked sources; `library=[]` skips builtin seeding
without clearing strategies already learned in that worker.

A locally holding reply is not a game-theoretic draw. The library stores
complete wins for either colour; a hit for the defender can disprove an
attacker's branch. An unsuccessful forcing search supplies no library verdict
and cannot justify pruning all other moves. Likewise, the suggested abstract
"three-stone tempo" must be realized by legal exchanges before a proof can use
it. The board and its real remaining placement count carry that information.

The worker retains at most 32 compiled templates under a 4 MiB accounting
budget, including an allowance for compiled masks and containers. Matching
instances and negative lookup memoization are bounded and expire after each
query. Existing small shapes are indexed once at the query root. Imported
shapes are also matched around newly added stones, comparing actual cell
contents when solver contexts reorder the board. Candidate geometries survive
backtracking within a query, with at most 512 per template. A failed candidate
watches one missing supporting stone or occupied protected cell and sleeps
until that cell changes. Removing stones during backtracking wakes the same
watches. Candidates are grouped by template and colour so a lookup visits only
the current turn phase. Every accepted use still checks the full mask and
global counter-threat guards. There is at most one watched cell per candidate;
the index expires with the query. Larger game-local stamps are
discarded when the current branch permanently occupies a required empty cell
or gives a required friendly cell to the opponent. Every hit is checked against
the current board and tempo. No result survives merely because it matched an
earlier position.

On insertion, the worker discards a stamp only when another checked strategy
has the same winner relative to the mover and the same remaining placements,
covers all its matching boards, and has an equal or shorter win bound. This compares
supporting stones, protected empty cells, counter-threat guards and the common
attacker prefix. Strategies compare in their actual game coordinates. Distinct
fixed frames remain available even if capped geometric matching cannot discover
them again. Small relocatable entries retain the same supporting geometry; an
imported entry also retains its root mask filtering. Broader-but-slower strategies remain separate. Equal
conditions and bounds keep the smaller compiled strategy. Portable status
follows the retained entry; an active query keeps its original sources.
Repeated imports remember the complete validated source identity and the
retained strategy that covers it. These aliases share the 4 MiB accounting
budget and stop applying when that strategy leaves the library, so deduplication
does not force the client to recheck a discarded certificate on every query.

This comparison runs only when learning or importing a stamp. The library
remains local to one solver worker. Ordinary player analysis uses one worker;
Python batch review can use four processes with independent libraries. The
browser solver uses one Web Worker. Deduplication adds no threads or locks.

A quiet query permits at most 256 relevant cells at each split, the existing
certificate node limit, and 8 MiB of distinct source strategies. Search work
shares the query's node meter. Compilation and checking share its cancellation
token and deadline. The first enabled query also pays for library verification.
The player saves the certificate with the exact outcome; its existing proof
table propagates that result into the game graph.

## CPU measurements

The October 5 lookup change watches failed candidate conditions instead of
rescanning every geometry retained from earlier branches. The following paired
measurements compare the preceding production build, `2c392c9`, with that
change. Both builds enable stamps. CPU is a Ryzen 9 5900X, BelowNormal priority,
two OpenMP threads, no GPU. Browser timings use the shipped WASM in Node.

| Workload | Before | After | Speed-up |
|---|---:|---:|---:|
| Complete `tree` win after `[14,0]`, native | 25.68 s | 12.54 s | 2.05x |
| Same complete proof, browser WASM | 36.46 s | 14.48 s | 2.52x |
| Six recorded games, 546 native queries | 29.27 s | 23.00 s | 1.27x |
| 28 original real positions, native | 1.49 s | 1.33 s | 1.13x |

The complete solves allow 524,288 nodes and 59 seconds. All four finish at
203,408 fresh nodes with identical certificates and a 14-attacker-turn bound;
the independent Python checker accepts them. The six games use the same shard
and query sequence documented below, at 8,192 nodes and 5 seconds per query.
Every query's verdict, node count and stamp hits agree between builds: 210
proven wins, 329,076 fresh nodes and 97 stamp hits. None hit a deadline.

Disabling stamps still completes the six-game replay faster, in 15.20 seconds,
with 341,767 nodes and the same 210 wins. This change reduces miss overhead;
it does not make stamps beneficial on every workload. The eight changed real
positions still prove in 24 ms total using eight fresh nodes after their
source strategies have been learned, versus 485 ms, 9,471 nodes and seven
wins without stamps. These recorded-position tests do not measure games/hour.

Earlier measurements follow for the preceding reuse changes.

The analysed turn-17 variation replacing Blue's `[-2,0]` with `[-1,0]`
was checked with the saved 153-record study in a fresh WASM worker. Both
queries allowed 524,288 nodes and 15 seconds on the same CPU, at BelowNormal
priority. The reference received the archive's applicable exact graph facts;
replay received its move suggestions and certificates.

| Changed turn-17 position | Time | Fresh nodes | Result |
|---|---:|---:|---|
| Previous production solver | 15.01 s | 84,890 | UNKNOWN, deadline |
| Checked strategy replay | 5.99 s | 1,509 | Proven win, 62-placement bound |
| Same worker querying the resulting stamp again | 12 ms | 1 | Proven win |

Restoring the whole study through `BrowserSession` and running the normal
worker analysis path took 5.92 seconds of analysis, with the same 1,509 nodes
and proof bound. This integration check used a uniform test network; the
tactical solver and certificates were real. The independent Python verifier
also accepted the full changed-board certificate. These are one puzzle's
measurements, not a general solver speed-up claim. The baseline did not finish,
so the table does not assign it a time-to-solve ratio.

The next optimization retains exact child-strategy compilations for the duration
of one parent compilation. A child can be absent from the persistent library
because another stamp covers it, while its specific moves still need checking
inside the parent. The temporary memo compares the complete serialized source,
keeps the usual library insertion and dominance rules, and clears on completion
or failure. It permits 128 entries and 4 MiB of accounted source and stamp bytes.

The following paired WASM run compares that memo against PR #369. The fixed set
contains the existing 28 real positions, an open three, a quiet defender root,
and the study's cold solve, warm reuse and saved-proof reload. Ordinary queries
allow 65,536 nodes and 10 seconds. The study allows 524,288 nodes and 15 seconds;
warm reuse allows one node and reload allows 20,000. Each version uses the same
inputs and limits, at BelowNormal priority. Times include verification and JSON
serialization. No query reaches its deadline.

| Complete query | Before | After | Fresh nodes, both |
|---|---:|---:|---:|
| 30 ordinary positions, total | 7.502 s | 7.312 s | 64,012 |
| Study, changed defence | 6.045 s | 4.585 s | 1,509 |
| Study, warm reuse | 8.4 ms | 8.3 ms | 1 |
| Study, fresh worker loading saved proof | 2.440 s | 1.147 s | 1 |
| All 33 queries | 15.995 s | 13.052 s | 65,523 |

All outcomes, node counts, bounds and certificates are identical. Both solve 14
queries; the other 19 remain UNKNOWN. The study's cold query uses 4.77 CPU seconds
in 4.58 wall seconds, consistent with one busy solver thread. Its bound is still
62 placements, not a claim of a globally shortest win. The 2.6% difference on
ordinary positions is too small for a general speed claim from this one run.

Run `PYTHONPATH=python:. python tools/proof_stamps.py --benchmark results.json`
in a fresh process (use `python;.` on Windows). The input file is the existing
28-position fixture from dense-v1 shards. Queries have equal node budgets and
10-second deadlines; wall time includes checking and serialization. The
changed-board cases first learn the source strategy, then add four legal remote
stones, preserving the mover and turn phase. They exercise local reuse rather
than the existing cache for identical boards.

| Queries | Reference | Stamps | Result |
|---|---:|---:|---|
| 28 original positions, total | 825.9 ms | 1472.4 ms | Same 7 wins; no stamp hits |
| 8 changed real positions, total | 482.4 ms | 683.2 ms | Same 7 wins; 3 stamp hits |
| Changed shard `1790600287230040:30:248` | 66.20 ms / 1003 fresh nodes | 2.93 ms / 1 node | 22.6x; both proven |
| Changed shard `1790604657706760:28:77` | 24.43 ms / 24 fresh nodes | 1.65 ms / 1 node | 14.9x; both proven |
| Open three after four remote stones | 534.79 ms / 8972 fresh nodes | 4.97 ms / 1 node | 107.7x; both proven |

The real-position queries use 8192 nodes; the open-three comparison uses 50000.
These are solver timings, not actor or GPU throughput claims. The library uses
about 219 KiB of its accounting budget in this run. Reuse pays when a strategy
matches; misses cost time. A quiet diamond is also
proved with a full two-stone defender turn, and independently checked, where
the reference defender query returns UNKNOWN.

Six recorded games were also replayed chronologically from dense-v1 shard
`1791050712484684` (episodes 0–5), querying the mover and hypothetical opponent
at every two-placement turn start through adjudication. One worker and its
stamp cache lived through all six games in each mode, with 8192 nodes per query
and no deadline hits. This includes learning and checking stamps. Both modes
used the recorded moves, without network inference or new move selection.

| Six-game replay | Reference | Stamps |
|---|---:|---:|
| Queries | 546 | 546 |
| Solver time | 15.504 s | 31.374 s |
| Fresh nodes | 341,767 | 341,727 |
| Proven-win queries | 210 | 210 |

All query outcomes agreed. The accepted certificates contained 77 stamp hits;
the cache peaked at 497.4 KiB. Reuse across these turns saved only 40 fresh
nodes, and total solver time rose 102%. The player default is enabled at the
owner's request despite that measured cost; these figures do not establish a
game-throughput benefit.

After proof compaction, another paired replay of those same six games measured
insertion-time deduplication. Both versions used stamps, one resident worker,
8192 nodes per query, a 5-second safety cap, BelowNormal priority and two CPU
threads. No query hit its deadline. All 546 outcomes still agreed.

| Six-game replay | Before deduplication | After deduplication |
|---|---:|---:|
| Solver time | 28.922 s | 28.595 s |
| Fresh nodes | 329,076 | 329,076 |
| Proven-win queries | 210 | 210 |
| Stamp hits | 97 | 97 |
| Peak compiled entries | 32 | 32 |
| Mean compiled entries | 26.6 | 26.0 |
| Peak accounted library size | 234.1 KiB | 179.6 KiB |

The final version took 1.1% less time and used 23.3% fewer peak accounted bytes
in this comparison. Fresh nodes and stamp hits were unchanged. This small timing
difference is one paired measurement, not evidence of a broad throughput gain.
These recorded continuations do not measure newly selected moves or games/hour.

The distinction between forcing, holding and unstoppable shapes is also useful
in [Six's shape guide](https://github.com/CixMango/Six/blob/main/guide/shapes.md).
The boundary construction follows the purpose of
[relevance-zone proof search](https://scholar.nycu.edu.tw/en/publications/relevance-zone-oriented-proof-search-for-connect6/),
with HeXO's actual one/two-placement phase, legal frontier, and independently
checked strategy as the obligations. Neither source supplies our verdicts.

## Puzzle archive benchmark

The proof search now shares turn-context nodes inside each bounded level-2
search. Changes propagate to every parent. Expanded estimates and proved
subpositions survive in the existing bounded table even when the seed root
remains unresolved. Positive entries retain their reconstruction witnesses.
Consecutive searches leave their shared board prefix in place, and table hits
use the child hash without applying and undoing its stones. Stamp lookup runs
when a candidate is expanded and before discarding a forcing dead end.

The October 5 archive has 66 HTTTX entries, representing 64 distinct boards.
Each expects a win for the actual mover, including one-stone turn contexts
when supplied. This expectation is a benchmark label, never a proof premise.
The archive SHA-256 is
`49e6823458f15797184ac8fe6f2d05d20cd2498f48e956451a9999556d8c5e2c`.

Against main `caa20c8`, on a Ryzen 9 5900X at BelowNormal priority with two CPU
threads allowed and no GPU, the native results were:

| Workload | Before | After | Verified wins |
|---|---:|---:|---:|
| All 66 puzzles, 8,192 nodes each | 19.38 s / 219,913 nodes | 18.15 s / 185,074 nodes | 37 → 43 |
| All 66 puzzles, 131,072 nodes each | 96.25 s / 1,578,571 nodes | 58.02 s / 1,263,228 nodes | 47 → 47 |
| The 47 solved puzzles at the larger budget | 35.13 s / 388,860 nodes | 18.43 s / 170,126 nodes | All 47 retained |
| Six recorded games, 546 queries at 8,192 nodes | 22.39 s / 329,076 nodes | 16.69 s / 265,044 nodes | 210 → 213 |
| Browser WASM, all 66 puzzles at 8,192 nodes | 28.31 s / 219,913 nodes | 28.67 s / 185,074 nodes | 37 → 43 |

Each puzzle starts with a fresh worker and stamps enabled. The safety caps are
5 seconds for the small budget and 15 seconds for the larger one; none fired.
Times include native checking, stamp compilation and response decoding. Every
positive result also passed the independent Python checker outside the timed
query. These are solver measurements, not new games/hour measurements.
The six-game replay keeps one worker through the recorded positions from shard
`1791050712484684`, episodes 0–5, as in the earlier comparisons. It retains every
previous win and the same 97 stamp hits.
The browser row uses the player's WASM library through Node, with the same
cold-worker policy and a 15-second safety cap. Its total time is slightly higher
while producing six additional checked proofs. Its per-puzzle verdicts and
fresh node counts agree with native; it is not a measurement of page rendering.

The unresolved puzzles remain in both totals. Eight still exhaust 524,288
nodes in a follow-up probe; the others stop without a complete forcing proof.
These results do not establish that the selective forcing model can solve the
entire archive. There are no puzzle-specific search rules or larger default
budgets in this change.

Run the archive with the existing benchmark tool:

```powershell
$env:PYTHONPATH = 'python;.'
python tools/proof_stamps.py --puzzles "$env:USERPROFILE/Downloads/puzzles-HeXO.txt" --benchmark artifacts/puzzles.json --nodes 131072 --ms 15000
```

Use `--no-stamps` for the reference stamp mode. The JSON output preserves every
position and its turn context, input and build hashes, caps, per-puzzle time,
fresh nodes, verdict and independent-check time. It is rewritten after each
completed puzzle. The same positions and labels can be reused for the later
GPU-assisted solver; unknown results stay visible in the score.

## Checking and stamp compilation

The raw-coordinate checker and stamp compiler now scan each six-cell window
from its first friendly stone. They reject blocked or too-empty windows before
allocating gap lists, and keep the same completion sets and stamp footprints.
Completion gaps are at most five steps from their anchor, so they need no
separate radius-eight legality scan. Cancellation is checked at each anchor.
The checker also avoids repeating the defender counterwin test before the
defense enumerator performs it.

On the same archive and CPU, including the full query cost:

| Workload | After graph reuse | After checking changes | Verdicts |
|---|---:|---:|---|
| All 66 puzzles, 8,192 nodes | 18.36 s | 12.76 s | Same 43 wins |
| All 66 puzzles, 131,072 nodes | 58.02 s | 52.44 s | Same 47 wins |
| The 47 solved puzzles at 131,072 nodes | 18.43 s | 12.33 s | All retained |
| Six recorded games, 546 queries | 16.69 s | 14.91 s | Same 213 wins |
| Browser WASM, all 66 puzzles at 8,192 nodes | 28.67 s | 17.58 s | Same 43 wins |

Compared with the original reference, the combined changes give a 1.84x
speedup over the entire larger-budget archive and 2.85x over its solved
positions. Fresh-node counts, proof bounds and the returned puzzle
certificates are unchanged by the checking changes. The recorded-game replay
also retains identical stamp hits and accounted library sizes. All benchmark
wins pass the independent Python checker. These measurements include unresolved
cases and use the same caps; no query reaches its deadline.
