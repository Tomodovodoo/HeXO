# Local proof reuse

`python/play.py --proof-stamps` enables this path for the player, including its
analysis workers. The Python solver API accepts `stamps=True`; the browser
solver accepts the same option, and `Bubble.turn(..., {proofStamps: true})`
passes it through the browser search worker. The default remains unchanged.

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

The compiler also tries the proof again with only its supporting stones left.
When that replay succeeds, it saves this smaller source position. Composed
proofs are expanded and rechecked before saving, so learning cannot create
ever-growing chains of imported strategies.

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

## Library and limits

`tools/tactical/stamps.json` contains primitive strategies produced by HeXO.
`tools/proof_stamps.py --out PATH` regenerates them and runs the independent
checker. The native worker checks these sources on first use. An explicit
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
query. Existing small shapes are indexed once at the query root. Only imported
primitives are also matched around the last two added stones. Candidate
geometries survive backtracking within a query, with at most 512 per template;
every use checks the actual stones and masks again. Larger game-local stamps are
discarded when the current branch permanently occupies a required empty cell
or gives a required friendly cell to the opponent. Every hit is checked against
the current board and tempo. No result survives merely because it matched an
earlier position.

A quiet query permits at most 256 relevant cells at each split, the existing
certificate node limit, and 8 MiB of distinct source strategies. Search work
shares the query's node meter. Compilation and checking share its cancellation
token and deadline. The first enabled query also pays for library verification.
The player saves the certificate with the exact outcome; its existing proof
table propagates that result into the game graph.

## CPU measurements

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
matches; misses cost time, so this remains opt-in. A quiet diamond is also
proved with a full two-stone defender turn, and independently checked, where
the reference defender query returns UNKNOWN.

The distinction between forcing, holding and unstoppable shapes is also useful
in [Six's shape guide](https://github.com/CixMango/Six/blob/main/guide/shapes.md).
The boundary construction follows the purpose of
[relevance-zone proof search](https://scholar.nycu.edu.tw/en/publications/relevance-zone-oriented-proof-search-for-connect6/),
with HeXO's actual one/two-placement phase, legal frontier, and independently
checked strategy as the obligations. Neither source supplies our verdicts.
