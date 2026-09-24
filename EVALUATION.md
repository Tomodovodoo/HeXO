# Frozen paired evaluation

```sh
python evaluate.py --candidate runs/example/checkpoints/0005/model.nnue --reference runs/example/checkpoints/0000/model.nnue --output runs/example/evaluation --games 160 --ms 100 --width 16 --max-stones 800 --workers 2 --seed 20261002
```

The output directory must not exist. The CLI copies and hashes both checkpoints,
the loaded native library, and its Python/C++ sources. Matches run in that copied
source tree. Each fresh mixed-v1 held-out opening is played with swapped colors.
On Windows and Linux, provenance also records resolved paths and hashes for loaded
file-backed runtime modules, including transitive native dependencies. These host
files are checked before matches, in each worker, and after completion. They are
verified in place rather than copied; the snapshot does not isolate the whole OS.
Choose the game count and seed before inspecting results; repeated exploratory
comparisons do not constitute a prespecified confirmation.

`report.json` retains every placement, complete native search result, actual
per-turn search wall time, model hashes, protocol, and opening schedule hash.
Each game is replay-validated before publication. Model attachment time is excluded
from search timing. `status.json` is updated atomically for the training dashboard;
put this output under the run's `evaluation` directory to attach it. The candidate
hash must match the latest checkpoint before the dashboard shows its evidence.
Failed status retains both checkpoint hashes and the last progress counts.

The reference has an arbitrary anchor of zero Elo. The reported relative estimate
uses a half-win/half-loss continuity correction and a conservative 95% opening-pair
Hoeffding interval. An open interval endpoint is JSON `null` (negative infinity for
the lower bound, positive infinity for the upper). Pending or capped matches have
no rating; censored outcomes remain bounded in the win-rate interval. This is a
fixed-budget comparison against one checkpoint, not a global rating or promotion.
The shared statistics helper is also used by native training evaluations.

The implemented backend is `native-nnue`: both models use the same native search
budget and engine. Match execution is isolated in `evaluate.play`; paired
statistics, snapshots, traces, and reporting do not depend on NNUE internals.
A future neural/MCTS backend must supply complete-turn traces through this adapter
and snapshot its own executable dependencies. It is not implemented by this CLI.
