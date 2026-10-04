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

The worker retains at most 32 compiled templates under a 4 MiB accounting
budget, including an allowance for compiled masks and containers. Matching
instances and negative lookup memoization are bounded and expire after each
query. Existing small shapes are indexed once at the query root. Only imported
primitives are also matched around newly played stones, so every learned proof
does not add another per-node geometry scan. Larger game-local stamps are
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

The distinction between forcing, holding and unstoppable shapes is also useful
in [Six's shape guide](https://github.com/CixMango/Six/blob/main/guide/shapes.md).
The boundary construction follows the purpose of
[relevance-zone proof search](https://scholar.nycu.edu.tw/en/publications/relevance-zone-oriented-proof-search-for-connect6/),
with HeXO's actual one/two-placement phase, legal frontier, and independently
checked strategy as the obligations. Neither source supplies our verdicts.
