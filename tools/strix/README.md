# Strix tactical reference

This separate executable calls the MIT-licensed [SootyOwl/hexo-strix](https://github.com/SootyOwl/hexo-strix/tree/5a771e572553a8bd8e010112b2ce65f16e5afa1b)
`hexo-solver` library at revision `5a771e572553a8bd8e010112b2ce65f16e5afa1b`.
Cargo pins that revision and the checked-in lockfile pins Rust dependencies.
No Strix implementation is copied into HeXO. The external source carries its
[MIT license](https://github.com/SootyOwl/hexo-strix/blob/5a771e572553a8bd8e010112b2ce65f16e5afa1b/LICENSE).
If distributing a linked executable, include that license, copyright notice,
and dependency notices. No executable is committed here.

Install Rust with edition 2024 support and a native linker, then from the HeXO root:

```sh
cargo build --release --locked --manifest-path tools/strix/Cargo.toml
python -m unittest tests.test_strix_reference -v
```

The local Windows verification used Rust 1.98.1, the GNU Windows target, and
MinGW. CUDA, PyTorch, MCTS, and neural checkpoints are unnecessary.

```python
from strix_reference import StrixReference

with StrixReference() as solver:
    result = solver.solve([[q, 0, "P1"] for q in range(4)], "P1", 2,
                          depth=8, nodes=100000, wide=False, timeout_s=1)
```

The process stays alive across successful queries. Each query gets a fresh
solver state, so no negative result leaks between generator modes or budgets.
Calls on one client serialize. Close the client when finished. A wall timeout
kills and reaps the process, returns `UNKNOWN`, and starts a new process on the
next call. Transport runs in a worker thread. Absolute deadline checks discard
late replies, including those already queued or delayed by PV validation.
The timeout covers solver startup/search/PV extraction, with normal
OS scheduling and cleanup overhead. IDTT does not expose a usable node counter;
responses report `nodes: null` instead of claiming zero search work.
Direct responses carry the executable SHA256 recorded before first launch.
The client refuses to restart if those bytes change, so a replacement binary
requires a new client and produces distinct provenance. Malformed non-object
responses become `UNKNOWN`, and the client always sets the independent-proof
flag to false itself.

## Result scope

Only IDTT is exposed in this first adapter. Tight and wide are explicit options.
Depth counts attacker turns including the completing turn. Every response
includes attacker, remaining placements, generator, budgets, and fixed rules:
six or more in a line, radius 8, no match move cap. Negative results apply only
to the selected forcing-search domain and horizon.

- `REFERENCE_WIN_WITHIN_SCOPE` is Strix's positive result with a locally checked
  legal sequential principal variation ending in an attacker win. This is not
  an independently checked proof over every defender branch.
- `NO_FORCING_WIN_WITHIN_SCOPE` is a scoped negative, never a losing-game label.
- `UNKNOWN` covers node/depth exhaustion, coordinate spread limits, wall timeout,
  process failures, and rejected output. Invalid client inputs raise `ValueError`.

All results have `independently_verified_proof: false`. Nothing connects this
adapter to native PVS, promotion, or training targets.

The inspected upstream `forcing.rs::def_within` checks the defender's immediate
counterwin first. It refuses continuations with a cover smaller than two cells,
enumerates two-cell covers, and recognizes un-coverable threats. Thus a spare
defender placement is excluded from this forcing domain, never silently filled
with an arbitrary move. Wide mode broadens attacking partners; it does not make
the domain unrestricted. Principal-variation replay checks each placement in
order, alternates players only after the declared phase expires, and stops on
the first winning stone. A PV alone cannot verify all defensive branches.

The adapter limits requests to 4096 stones and 512 KiB. Input snapshots may be
unreachable; they are not advertised as legal game
histories. Duplicate stones, invalid players/phases, existing sixes, and
coordinates outside ±2^30 are rejected. Strix's dense-grid spread/allocation
guards may still return `UNKNOWN` for valid unbounded HeXO positions. The HeXO
engine's wider coordinate support is unchanged.

## Public fixture runner

```sh
git clone https://github.com/SootyOwl/hexo-strix EXTERNAL_STRIX
git -C EXTERNAL_STRIX checkout 5a771e572553a8bd8e010112b2ce65f16e5afa1b
python tools/strix_corpus.py EXTERNAL_STRIX --depth 8 --nodes 10000 --seconds 0.5 --output artifacts/strix-public-tight.json
```

The pinned repository supplies 25 JSON files: 19 declared snapshots and 6 game
logs. The runner reads committed bytes, records SHA256 values, queries the 19
snapshots, and explicitly skips logs lacking a query phase/attacker. It neither
invents winning labels nor treats game logs as independent puzzle snapshots.
These public files are distinct from the unavailable 20-case ZIP quoted in the
user's audit. No claim is made to have reproduced that audit.

Local bounded interface check at depth 8, 10,000 nodes, and 0.5 seconds per snapshot:

| Generator | Reference wins | Scoped negatives | Unknown |
| --- | ---: | ---: | ---: |
| Tight | 5 | 3 | 11 |
| Wide | 4 | 2 | 13 |

All returned winning lines passed sequential replay. Wide mode spends its
budget on more candidates, so this small-budget count is not a comparison of
the generators' eventual completeness. Ten adapter tests pass. This is a
bounded interface check, not a solver-strength or proof-soundness benchmark.
