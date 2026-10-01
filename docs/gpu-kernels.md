# GPU kernels

`--net-kernels fused` replaces the parts of HexNet around the hex convolutions with Triton kernels: the fixed line features, masked normalisation with its activation and backward pass, and the line convolution. The hex convolutions stay on cuDNN; they are the largest cost, but the avoidable work was around them (index transfers that synchronised the stream, layout copies, normalisation intermediates and the matrices the inference line convolution built). Actors run channels-last in this mode; the learner keeps NCHW and the cuBLAS line products with fused staging. Checkpoints do not record the mode and load either way. `reference` is the default and needs no Triton.

`--cuda-graphs` on the actors additionally captures the champion's forward passes for batches of 8, 16 and 32 rows per crop size and replays them. Historical opponents and uncaptured shapes use eager inference; captures stop at a 384 MiB reservation budget.

Kernel dimensions are runtime arguments, so an actor compiles four kernels once and reuses them from `~/.triton/cache` (or `TRITON_CACHE_DIR`) for every batch and crop size.

## Measured on the RTX 3070 Ti

Paired reference/fused/fused/reference windows on frozen real inputs, with the training run sharing the card.

| Work | Reference | Fused | Gain |
|---|---:|---:|---:|
| Actor placements/s (search, encoding, transfers, 128 games) | 26.3 | 72.5 | 2.76x |
| Actor evaluated positions/s | 473 | 1428 | 3.02x |
| Evaluator placements/s (64 games, 2048-node solver budgets) | 34.8 | 60.3 | 1.73x |
| Learner samples/s (batch 256, full training step) | 442.6 | 652.7 | 1.47x |
| Actor positions/s, fused with CUDA graphs, batch 128 | 671 | 1051 | 1.57x |

Fused outputs match the reference within two bf16 epsilons on every check, including a real 256-row training step's gradients, model and EMA tensors.

## Enable

```sh
python -m pip install triton-windows==3.6.0.post26
python python/bubble.py train --run runs/bubble --net-kernels fused
```

On Linux install `triton` instead. The [Windows wheel](https://github.com/triton-lang/triton-windows) bundles its compiler; no CUDA toolkit or MSVC is needed. Started by hand, the learner, actors and evaluator each take `--net-kernels fused`, and the actors `--cuda-graphs`.

## Reproduce

`tools/profile_hexnet.py`, `tools/profile_actor.py`, `tools/profile_evaluator.py` and `tools/profile_learner.py` prepare frozen inputs from a run and time each mode in child processes that run at BelowNormal priority with a memory fraction of 12%, a 55-second deadline and a 60-second idle gap, so they can run beside a live training run:

```sh
python tools/profile_hexnet.py prepare --run runs/bubble --checkpoint runs/bubble/checkpoints/main/085000/ema.pt
python tools/profile_hexnet.py eval --kernels fused graphs --batch 256 --repeats 2 --profile
python tools/profile_actor.py prepare --run runs/bubble
python tools/profile_actor.py actor --kernels fused --windows 10
python tools/profile_learner.py prepare --run runs/bubble --checkpoint runs/bubble/checkpoints/main/085000 --batches artifacts/learner-profile/batches
python tools/profile_learner.py live --run runs/bubble --checkpoint runs/bubble/checkpoints/main/085000 --batches artifacts/learner-profile/batches --modes fused --memory-mib 3328 --warmup-steps 10
```

Reports, traces and input hashes go under `artifacts/gpu-kernels`.
