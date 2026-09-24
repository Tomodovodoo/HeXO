# Human corpus import

`human_corpus.py` imports the explicit JSONL file from
[timmyburn/hexo-bootstrap-corpus](https://huggingface.co/datasets/timmyburn/hexo-bootstrap-corpus/tree/1a82e153ab313f0445a505dfc51289cbd805fe7d).
The revision and SHA-256 of all four source files are pinned in the module. The
dataset card declares MIT; that snapshot supplies no separate LICENSE file.
Raw files and generated arrays belong under ignored `artifacts/datasets/`.

```powershell
python human_corpus.py --download --convert --output artifacts/datasets/human-warmstart --exclude-histories artifacts/seal-trained-fresh-40.json artifacts/nnue-mixed-vs-bare-80.json
python -m unittest tests.test_human_corpus -v
```

Build the native engine first for conversion. Validation and family grouping
use independent Python rules and require no native library or PyTorch. A new
output directory is required so stale shards cannot contaminate a later split.
Each manifest records the snapshot hashes, rejected records, duplicate records,
exclusion inputs, family counts and shard hashes. `games.jsonl` preserves source
ratings, source IDs, complete histories and full move-content hashes. Source
16-character IDs are preserved rather than guessed from an undocumented hash
serialization.

Every placement must be empty, within radius eight of an existing stone, and
follow the central opening and one-then-two turn sequence. Replay stops on the
first six-or-longer line. A wrong winner, extra post-win move or unfinished game
is rejected. The corpus winner convention is +1 for player zero and -1 for
player one. There are no draw labels.

Families are connected components of games sharing any colored board with at
least seven stones, canonicalized over translation and all twelve hex-grid
symmetries. This includes same-color placement-order transpositions. The whole
component receives one split; early positions before seven stones are omitted.
Using the origin or two-stone prefixes as a grouping key would connect nearly
everything and leave no useful split. Seven is configurable with
`--minimum-ply`, and the manifest records it. No claim is made that every
strategically similar position is recognized.

Any component touching a supplied benchmark position is excluded altogether.
`--exclude-histories` accepts arena reports with `games`, individual ordered
`moves`/`cells`, and Strix snapshots with `stones` containing P1/P2 labels.
History prefixes and snapshots are matched with their actual colors; snapshot
matches exclude every phase conservatively. Supply all benchmark files that
must remain held out. The separately referenced 20-case tactical ZIP is absent
from this workspace, so it cannot be excluded by identity. Public Strix
fixtures are a separate dataset.

Components containing a prefix of length 3, 5, 7, 9 or 11 in the existing
mixed-v1 evaluation bucket are assigned to `test`, unless already excluded.
Remaining components whose stable family ID is divisible by five go to
`validation`; the rest go to `train`. IDs are derived from the smallest full
content hash in each component. Adding source games can change components and
IDs, which is why the source snapshot is fixed. Never pass the `test` directory
to an optimizer. Training and validation directories follow the existing
`family % 5` convention.

Conversion defaults to eight positions per game, stratified across both players
and first/second placements, spread over each game's history. Each chosen
human move is inserted into the legal candidate list if the native shortlist
omits it. States after a first placement are reconstructed before encoding the
conditional second action. Shards contain at most 64 games, so the importer
does not accumulate the entire sparse feature dataset in memory.

The output uses the existing NNUE ragged replay schema. `outcome` is the actual
finished result from the player-to-move perspective. `policy_valid=true` means
human imitation, not an optimal action or a completed search. The manifest
marks these shards as human policy data and stores their full game content IDs
and row counts in order. Keeping provenance out of arrays preserves native
replay merging compatibility. `search_valid=false`, depth and timing are zero. The inactive
`search` field repeats the outcome solely because the current objective gives
invalid search fallbacks a small weight; this avoids introducing artificial
zero targets. No teacher search was run.

Run a bounded warm start with:

```powershell
python corpus_warmstart.py --corpus artifacts/datasets/human-warmstart --output artifacts/models/human-initial --device cuda --epochs 3 --positions 16384
```

This checks hashes and split/label semantics before loading only `train` and
`validation`. Every eligible shard is considered, with no default eight-shard
chronology window. The randomized loader bounds total positions and center
features; batches also have an explicit center budget. CUDA is required when
requested, with no silent CPU fallback. `--initial-model` optionally loads a
previous `model.pt`; optimizer state starts fresh. The best validation epoch
produces `model.pt`, native `model.nnue`, `optimizer.pt` and a manifest of hashes,
metrics and actual memory usage. Test shards are never loaded. These commands
do not alter self-play, promotion or evaluation settings. Human imitation
quality and playing strength still require measured follow-up experiments.
