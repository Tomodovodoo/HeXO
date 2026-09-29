# Opt-in dense GPU kernels

`--net-kernels fused` selects Triton kernels for fixed line features, masked normalization with activation and its backward pass, and LineConv. Actors also use channels-last with this mode. Training keeps NCHW and the reference cuBLAS line products, with fused staging, residual additions and tap-gradient reductions. `reference` remains the default.

Keep cuDNN for HexConv. This checkout already implements each hex convolution as one masked 3x3 convolution. Its tensor-core kernels are the largest single cost, but the avoidable work is around them: small synchronous index transfers, layout copies, full-activation normalization intermediates, and the matrices and skewed copies used by inference LineConv. The direct LineConv kernel tiles pixels and channels together, which makes channels-last usable without those matrices. Normalization preserves the reference's bf16 rounding points and centred variance computation.

Checkpoint parameter names, config, digest and serialized tensors do not include the execution mode. Loading a checkpoint defaults to `reference`, including a checkpoint saved while training with `fused`. The learner flag is process-local. CPU execution uses the existing operations.

## Evaluator inference

Actor command-line overrides do not select kernels for the evaluator. Add
`--net-kernels fused` to `dense_eval.py loop`, `match` or `calibrate` to use the
same fused inference in that process. `dense_openings.py refresh` accepts the
flag too. Without an override, each command keeps the saved actor kernel setting,
which defaults to `reference`. The override never writes `config.json`.
Historical checkpoint models continue to use eager inference without CUDA graphs.

On September 29, 2026, the running evaluator still inherited `reference` from its
saved configuration. Its previously reported roughly 12 placements/s therefore
did not include the fused kernels. The following comparison uses the real
`dense_eval.Pool`, `MatchGame` and solver path on the shared RTX 3070 Ti:

| Mode, in measurement order | Placements/s | Neural positions/s | Mean neural batch |
|---|---:|---:|---:|
| Reference, first window | 33.48 | 444.83 | 41.14 |
| Fused | 60.25 | 791.75 | 42.06 |
| Reference, closing window | 36.17 | 469.42 | 40.18 |

Pooling the reference windows gives 34.83 placements/s, so fused improves
placement throughput by **1.73x**. Neural throughput also improves by 1.73x.
The fixed inputs are 32 saved replay prefixes, played with both colours in 64
concurrent slots, using the `085000` and `075000` EMA checkpoints. Finished slots
refill from those prefixes. Both modes use 64 simulations, 16 root samples,
the live evaluator's 2048-node root/finalist/threat budgets, two finalists,
three solver workers and a 32768-node gate cap. Each fresh process warms for
eight seconds and measures twenty seconds. The same frozen input file has
SHA-256 `b5596b5892e1f39ec12f122c571c00e83fc3f6cdf4ce0318188b354fdfa6a141`.

This measures actual search, inference and solver work on those midgame
positions. It excludes league bookkeeping and opening-book refresh, and its
absolute rate is not a forecast for the live evaluator's changing game mix.
Solver wait time was 1.69 and 2.43 seconds in the reference windows and 7.04
seconds with fused inference. Faster inference exposes more solver waiting.
Peak allocated/reserved GPU memory was 142/232 MiB for reference and 119/220 MiB
for fused. GPU occupancy lasted 29.4 to 30.2 seconds per child, with at least
60 seconds idle between children, BelowNormal priority and two CPU threads.
The allocator cap stayed at 12%; the live evaluator and run files were untouched.

Enable with `python dense_eval.py loop --run runs/dense-v1 --net-kernels fused`,
retaining the evaluator's other flags. Existing Triton installation suffices.
To reproduce the table, prepare replay references with
`python tools/profile_hexnet.py prepare --run runs/dense-v1`, then freeze the
evaluator inputs with `python tools/profile_evaluator.py prepare --run runs/dense-v1`.
Run `python tools/profile_evaluator.py live --kernels reference`, then `fused`,
then `reference`. The shared guard enforces headroom, cooldown and a 55-second
GPU deadline, and closes the measurement's own solver processes on timeout.

## Training LineConv and running statistics

The learner's fused mode now packs planar and skewed line inputs in Triton,
uses the existing cuBLAS products, and gathers the result with the residual
addition. Backward retains the reference order of bf16 additions and reduces
tap gradients directly. The custom backward saves the input and taps, so it
avoids recomputing the line forward through an inner activation checkpoint.
Normalization updates its running mean, variance and batch counter in one
kernel. Cumulative recalibration retains the existing implementation.

These changes use the existing `--net-kernels fused` flag. They add no learner
graph flag, checkpointing flag, dependency or checkpoint format change.

Production-path validation uses `Learner.train_step`, model/optimizer/EMA state
from export 85000, and successive 256-row batches sampled through the real
replay window with the learner's sampling and target settings. The source run
is read-only. The owner authorized stopping the learner after export 85000
completed so these measurements could use its 3328 MiB memory allowance.
Actors and the evaluator continue running. Every GPU child still has a
55-second limit and a 60-second cooldown.

The early LineConv-only comparison measured 146.9 samples/s versus 70.3 for
reference. A later comparison measured 145.3 versus 126.7 for the already
deployed fused implementation. These short windows expose warmup and shared
GPU load, so they do not establish the final twofold target. The final warmed
comparison and profiler table are pending.

Learner graphs were tested on these changing crop shapes and removed from
this PR. Five private graphs with block recomputation reached 131.9 samples/s
versus 146.9 for the eager kernel. A single 40x40 graph reached 140.7 versus
145.3 and reserved 3324 MiB. The eager kernel is the better measured choice
for this learner workload. Actor graphs remain available as described below.

## Actor CUDA graphs

`--net-kernels fused --cuda-graphs` adds a second opt-in acceleration for actors.
It keeps each crop's canvas and replays batches of 8, 16 or 32 rows, with at most
seven inert rows added per crop group. The 64x64 canvas uses at most 16 rows.
Fourteen possible captures share one memory pool and one stream. Inputs and
packed outputs live outside that pool; every returned output owns its storage.
Captures disable autocast's weight cache so graph pointers remain valid after
the capture context exits.

Only the current champion owns graphs. Historical opponents use eager fused
inference. A champion switch releases the old graphs before loading the new
champion; games using the old model continue with eager inference. New captures
stop when the measured reservation increase reaches the 384 MiB budget, or an
allocation fails, and uncaptured shapes use eager inference. The allocator cap
still applies to all development measurements.

The following paired comparison uses the same frozen real inputs and weights as
the earlier measurements. It compares the current fused implementation with
graphs, using four alternating observations per mode after capture and warmup.
These are resident-input model rates under the live run's shared GPU load.

| Batch | Fused positions/s | Fused + graphs positions/s | Additional speed-up |
|---|---:|---:|---:|
| 64 | 488.4 | 676.8 | 1.39x |
| 128 | 671.3 | 1051.0 | 1.57x |
| 256 | 845.3 | 1135.3 | 1.34x |

All fourteen shapes were captured, adding 292 MiB of reserved memory. Peak
allocated/reserved memory for the whole comparison was 274/686 MiB. Outputs
matched eager inference to the existing bf16 tolerance. Holding outputs across
later replays and reversing the canvas order preserved them exactly. Rates
exclude first-use capture; captures belong to each process and model.

At batch 256, eager execution submitted 831 individual kernel launches. Graph
execution submitted 12 graph launches and 4 individual kernel launches. Measured
CPU time in those launch calls fell from 9.01 to 4.03 ms. The graph contained
1,780 kernels versus 831 in eager execution: smaller batches and padding raised
the sum of kernel durations from 48.29 to 56.32 ms. Removing host submission work
still improved wall throughput. GPU gaps include contention from the live run
and are not counted as measured launch overhead.

| Top kernel group at batch 256 | Fused ms | Fused + graphs ms |
|---|---:|---:|
| Dominant cuDNN bf16 convolution | 25.20 | 22.53 |
| Direct LineConv | 10.10 | 10.82 |
| Norm plus activation | 3.69 | 3.95 |
| Elementwise residual add | 2.67 | 2.56 |

Graph replay lacks eager operator FLOP annotations; zero in that profiler field
does not mean zero arithmetic. The CUDA trace still records kernel durations.
These gains are measured against fused in the same window, and are not multiplied
by earlier reference comparisons made under different shared-card load.

Enable actors with `python dense_selfplay.py --run runs/dense-v1 --processes 4
--net-kernels fused --cuda-graphs`. No additional dependency beyond the existing
Triton installation is needed. The learner continues to use `--net-kernels fused`.
To reproduce each graph comparison, run `python tools/profile_hexnet.py eval
--kernels fused graphs --batch 256 --repeats 2 --profile`, replacing 256 with 64
or 128. The tool enforces the same bounded GPU sessions and cooldowns.

## Variable actor shapes

The original fixed-shape measurements below missed a live regression. On September 29,
four actors using `fused` spent much of their time compiling new shapes. CPU inspection
of `%USERPROFILE%\.triton\cache` found 1,191 directories, 377,623,192 bytes and 1,184
compiled HeXO kernels: 297 `_windows`, 296 `_features`, 296 `_eval` and 295
`_line_add_nhwc`. Reading the constants from `_windows.ttir` recovered 297 distinct
`(batch, canvas)` pairs: 143 for 24x24, 44 for 32x32, 64 for 40x40 and 46 for 48x48.
These are distinct compiled variants, not a frequency histogram of forwards.

`dense_selfplay.Evaluator.submit` merges small crop groups into the next canvas when
the added area is below `MERGE_CELLS`, then splits at `MAX_CELLS = 110592`. For those
four canvases, the maximum forward batches are 192, 108, 69 and 48. The last chunk
can have any smaller positive batch. Warming one fixed batch per canvas does not
cover this workload.

Batch, height, width and strides now enter the Triton kernels as runtime arguments.
The [JIT `do_not_specialize` option](https://triton-lang.org/main/python-api/generated/triton.jit.html)
also prevents scalar value/alignment specialization. Launch grids use the actual
sizes and the existing masks cover partial tiles. Model constants, layout and tile
sizes remain compile-time choices; no canvas or batch padding was added. Training
uses runtime dimensions too, with bounded power-of-two tiles for the final reductions.
Triton 3.6 retains alignment specialization inside stride tuples; all supported
actor canvases share those alignments. The CUDA reuse check covers the actual actor
buckets and varying batch tails, while the existing checks cover both layouts.

The default cache is explicitly `%USERPROFILE%\.triton\cache` (or `~/.triton/cache`
on Linux). All actor processes and subsequent launches use the same persistent
directory. An explicit `TRITON_CACHE_DIR` takes precedence; set it once in the
supervisor's environment if a different shared location is wanted. No cache is
created in a run, and existing cache entries need not be deleted. Kernel source
changes produce new cache keys.

`tools/profile_actor.py cache` reproduces the CPU cache audit. For an actor run,
first prepare the frozen inputs with `tools/profile_hexnet.py prepare` as below,
then run `tools/profile_actor.py prepare --run <run>`. Preparation reads the
effective actor settings from the first shard's manifest; CLI overrides are also
accepted. Preparation only reads the run. The actor benchmark uses `Engine`,
`SelfPlayGame` and `Evaluator.submit` directly, with 128 games started from real
shard positions and the frozen checkpoint; it records actual forward shapes.
It does not publish shards or update run status. Each completed game is replaced
from the same frozen position pool. This measures an isolated actor under shared
GPU load, including search, crop encoding and transfers.

Use `python tools/profile_actor.py actor --kernels fused --cache
artifacts/gpu-kernels/actor-cold-cache --label cold --windows 10` with a new cache
directory for a cold run. Each window targets 30 seconds and finishes its current
engine step. The worker retains its games, CUDA context and loaded kernels through
the required 60-second idle breaks. Rates exclude the breaks; the first window
includes model construction and JIT compilation, after Python/CUDA initialization.
The parent stops a window at 55 seconds and waits above 7400 MiB card usage.
Subsequent processes should reuse that cache with a new `--label`; use
`--kernels reference` for the paired baseline. Every GPU worker uses BelowNormal
priority, `OMP_NUM_THREADS=2` and the 12% allocator cap before other GPU work.

The cold run used the same four shards and `main/065000/ema.pt` as PR 174. Search
used 64/12 simulations, 128 games, leaf batch 256, 135-node root/threat/finalist
base budgets, two finalists, two adaptive solver workers, node caps 2048/32768,
gate weight 3, proof following, adjudication and proven-line rows. All ten windows
stayed below 30.41 seconds; the table labels their nominal active-time intervals.
They totalled 302.02 active seconds plus the mandatory idle breaks.

| Active seconds | Placements/s | Evaluated positions/s |
|---|---:|---:|
| 0-30 | 65.8 | 1312 |
| 30-60 | 87.0 | 1780 |
| 60-90 | 85.1 | 1744 |
| 90-120 | 52.5 | 1102 |
| 120-150 | 57.3 | 1089 |
| 150-180 | 54.2 | 1139 |
| 180-210 | 53.0 | 1119 |
| 210-240 | 79.7 | 1630 |
| 240-270 | 50.1 | 1013 |
| 270-300 | 47.5 | 976 |

The actor made 9,198 forwards with 251 distinct `(batch, canvas)` pairs. Canvas
frequencies were 9 at 24x24, 1,048 at 32x32, 3,637 at 40x40, 2,447 at 48x48 and
2,057 at 64x64. Mean request batches ranged from 225 to 239. The isolated cold
cache contained exactly four compiled kernels after the first window and after
every subsequent window: nine directories, 1,441,782 bytes. Peak allocated/reserved
PyTorch memory was 117/254 MiB. The last five windows had medians of 53.0 placements/s
and 1,119 evaluated positions/s. Shared-card load varied during the run; those rates
are not a four-actor deployment result.

The actor comparison then ran reference/fused/fused/reference, each in a fresh
process using the same position pool, seed, settings and checkpoint. Fused reused
the disk cache. Each 30-second window includes actor setup; the median of the two
windows per mode gives the following result. Reference windows were 24.2 and 28.3
placements/s; fused windows were 85.1 and 59.9. This variability is why the order
is paired and the individual results are retained.

| Actor work | Reference /s | Fused /s | Speed-up |
|---|---:|---:|---:|
| Placements | 26.3 | 72.5 | 2.76x |
| Evaluated positions | 473 | 1428 | 3.02x |

The cold worker and three subsequent fused worker processes shared the same four
kernel files without changing their write times. The fourth process measured
79.3 placements/s. Processes ran sequentially to obey the one-GPU-process limit;
this verifies persistent reuse without launching four benchmark actors together.
Twenty-seven focused CPU tests and all four CUDA tests passed, including changing
actor batches/canvases without adding compiled variants, both layouts, full-model
forward/backward agreement and checkpoint compatibility.

The PR 174 resident-input inference benchmark was repeated after warmup with its
unchanged frozen batch and reference/fused/fused/reference order, three repetitions.
These are steady-state model submissions, excluding actor search and transfers.

| Batch | Reference positions/s | Fused positions/s | Speed-up |
|---|---:|---:|---:|
| 64 | 254.0 | 752.8 | 2.96x |
| 128 | 383.6 | 911.2 | 2.38x |
| 256 | 500.2 | 1185.6 | 2.37x |

At batch 256 the profiler counted 2,184 versus 828 kernel launches and 72.90 versus
45.83 ms in kernels. Launch-call CPU time was 17.77 versus 7.72 ms. The runtime
dimensions retained the original kernel-time improvement while bounding compilation.
The batch-64 training comparison (two paired repetitions) measured 51.3 reference
versus 74.6 fused samples/s, a 1.45x gain, with peak allocated/reserved memory of
762/982 MiB. The original batch-256 training OOM limit remains; it was not rerun or
extrapolated from batch 64.

## Original fixed-shape measurements (PR 174)

Measured on 2026-09-29 with RTX 3070 Ti, driver 560.94, Python 3.14, torch 2.11.0+cu126 and triton-windows 3.6.0.post26. All GPU children ran at BelowNormal priority with two OpenMP threads and `torch.cuda.set_per_process_memory_fraction(0.12)` before GPU work. A parent stopped children at 55 seconds and enforced at least 60 seconds between them. It waited whenever the preflight `nvidia-smi` reading exceeded 7400 MiB. The live run kept running throughout.

The fixed batch uses seed 3070 and 256 rows from shards `1790686740142252`, `1790686744197376`, `1790687074623980`, and `1790687075148972`. Weights are the frozen `main/065000/ema.pt` checkpoint. These shard reads do not construct ReplayWindow or write a run cache. The sampled mean canvas is 1471 cells, smaller than the approximately 2700-cell figure in the brief. Results apply to this sample.

The learner's batch-256 bucket shapes are 52x24x24, 75x32x32, 72x40x40, 44x48x48 and 13x64x64. Its existing pad function rounds those row counts to 64, 80, 80, 48 and 16. Actor forwards retain the existing 110592-cell limit. Inference below measures resident-input model submissions grouped by crop size, without search, crop encoding or transfers. It does not measure games per second. Training includes pad, transfers, losses, backward, gradient clipping, AdamW and EMA updates.

The table uses alternating reference/fused/fused/reference order, repeated three times for inference and twice for training, after warmup. Rates are samples or positions divided by the median synchronized wall time. Each pair uses identical rows and starting weights. These are shared-card measurements; the supplied 420 samples/s figure is not used as a baseline.

| Work | Batch | Reference /s | Fused /s | Speed-up |
|---|---:|---:|---:|---:|
| Inference | 64 | 273.6 | 833.5 | 3.05x |
| Inference | 128 | 394.1 | 1003.6 | 2.55x |
| Inference | 256 | 877.8 | 2364.1 | 2.69x |
| Training | 64 | 51.4 | 73.3 | 1.43x |
| Training | 256 | OOM at 12% cap | OOM at 12% cap | Not established |

The batch-256 reference learner exhausted the 983 MiB allowance while requesting another 30 MiB. The fused learner also exhausted it. No smaller-batch result is extrapolated to batch 256. The twofold training target remains unproven. The batch-64 comparison reached 765 MiB allocated and 982 MiB reserved, so it also reflects the imposed memory limit. A full batch-256 comparison needs a separately authorized validation window that can fit the reference step.

## Profiler evidence

CUDA and CPU activities were recorded with input shapes. Timing excludes the profiler. The batch-256 inference trace changed as follows.

| Metric per 256-position submission | Reference | Fused |
|---|---:|---:|
| Kernel launches | 2184 | 828 |
| Sum of kernel durations | 75.29 ms | 46.20 ms |
| CPU time in runtime/driver kernel launch calls | 20.81 ms | 7.08 ms |
| CPU time in stream/device synchronization calls | 195.86 ms | 40.81 ms |
| Small host-to-device index traffic | 2016 bytes | 0 bytes |

Kernel time improved 1.63x. Wall time improved 2.69x. Launch-call time is measured host submission time, not a claim that all gaps on the GPU are launch overhead. Other live processes share those gaps. The original LineFeatures indexing caused 114 stream synchronizations across these six forwards, despite transferring only about 2 KiB.

The largest kernel groups are below. Names are shortened from the CUDA trace.

| Kernel group | Reference time | Fused time |
|---|---:|---:|
| Dominant cuDNN bf16 tensor-core convolution | 24.00 ms | 23.95 ms |
| Inference LineConv stencil | absent | 9.27 ms |
| Norm plus activation | separate kernels | 3.68 ms |
| NCHW/NHWC conversions | 7.91 ms | absent from dominant kernels |

The initial fitting training diagnostic, 16 real 32x32 rows, launched 2191 kernels over forward, backward and AdamW. It spent 17.56 ms in kernels and 20.14 ms in launch calls. The leading groups were convolution weight gradient at 1.90 ms, convolution forward at 1.68 ms, elementwise multiply at 1.25 ms, convolution input gradient at 1.13 ms, and Toeplitz indexing backward at 0.98 ms. This diagnostic identified the normalization passes without changing the batch-256 learner's statistics by splitting its buckets.

The final fused trace of that same 16-row diagnostic launched 1590 kernels, with 13.29 ms of kernel time and 11.55 ms in launch calls. Its leading kernels were convolution weight gradient at 1.70 ms, convolution forward at 1.68 ms, convolution input gradient at 1.12 ms, and Toeplitz indexing backward at 0.98 ms. Peak allocated memory fell from 207 to 169 MiB. This is a diagnostic comparison, not the missing batch-256 training result.

CPU collation took a median 6.38 ms for 256 rows and padding 6.64 ms, with CPU profiling enabled. Rendering the sampled positions and targets took 140 ms in the one recorded preparation pass. The separate pad/transfer/cast measurement had a median 20.21 ms wall time. Its CUDA trace transferred 9,961,200 bytes in 0.532 ms, approximately 18.74 GB/s of host-to-device copy throughput. The dtype/layout cast took 0.045 ms in GPU kernels. Host staging and waiting are included in the wall measurement, not in the copy-engine duration.

The rated reference peaks are 608 GB/s of memory bandwidth, 21.75 TFLOP/s of FP32 CUDA arithmetic, and approximately 43.5 TFLOP/s of dense bf16 tensor arithmetic with FP32 accumulation. The GeForce FP32-accumulation rate is lower than the often-quoted FP16-accumulation or sparse rates. These figures follow the [NVIDIA 3070 Ti specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3070-3070ti/), [GA10x throughput table](https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf), and [19 Gb/s, 256-bit memory specifications](https://www.msi.com/Graphics-Card/GeForce-RTX-3070-Ti-VENTUS-3X-8G-OC/Specification).

Counting each inner convolution/GEMM once gives 0.838 trillion operations for reference inference and 0.764 trillion for fused inference, excluding the custom stencil's approximately 0.014 trillion operations. The corresponding conv/GEMM estimates divided by all kernel time are 11.13 and 16.53 TFLOP/s, or 25.6% and 38.0% of the rated dense bf16 peak. Dividing by submission wall time gives 6.6% and 16.2%. These are arithmetic estimates, not hardware-counter utilization. Autocast duplicates some outer profiler FLOP annotations, so summing the default annotations would overcount. The profiling tool removes those duplicate wrappers.

Achieved DRAM bandwidth was not measured. Requesting `dram__bytes_read.sum` and `dram__bytes_write.sum` through the installed profiler returned no DRAM counters. The transfer figure above measures PCIe copies, not DRAM traffic. Reporting a DRAM utilization percentage from tensor sizes would be an unsupported claim.

A fixed-input CUDA graph prototype reduced a sliced-reference submission from 460 to 94 ms in separate windows, but increased peak reserved memory from 266 to 554 MiB. Those numbers omit variable-shape capture and reuse costs. Graphs and torch.compile are not included in the shipped mode. The current change keeps the measured improvement without a graph cache in every actor.

## Enable and reproduce

Install the optional dependency in the Python environment that will run the flag:

```text
python -m pip install triton-windows==3.6.0.post26
python dense_learn.py --run runs/dense-v1 --net-kernels fused
python dense_selfplay.py --run runs/dense-v1 --processes 4 --net-kernels fused
```

These are enablement examples, not commands run against the live run during development. No CUDA toolkit, nvcc or MSVC installation is needed. The [Windows wheel](https://github.com/triton-lang/triton-windows) bundles its tools and pairs Triton 3.6 with torch 2.11. Development installed it under this worktree's ignored artifacts directory; the main Python environment was not changed. Initial calls compile kernels, so throughput measurements exclude compilation.

To reproduce the snapshot, use the four shard names above with `--shards` and the frozen checkpoint with `--checkpoint`:

```text
python tools/profile_hexnet.py prepare --run /path/to/runs/dense-v1 --shards 1790686740142252 1790686744197376 1790687074623980 1790687075148972 --checkpoint /path/to/runs/dense-v1/checkpoints/main/065000/ema.pt
python tools/profile_hexnet.py check
python tools/profile_hexnet.py eval --batch 64 --profile
python tools/profile_hexnet.py eval --batch 128 --profile
python tools/profile_hexnet.py eval --batch 256 --profile
python tools/profile_hexnet.py train --batch 64 --repeats 2
```

Outputs, input hashes, per-iteration times, traces, memory peaks and child-session logs stay under this worktree's `artifacts/gpu-kernels`. The tool imposes the same limits on each invocation. `train --batch 256` records an OOM result under this cap rather than changing the model's batch semantics.

CPU tests check reference/fused forward and backward equivalence and checkpoint loading in both directions. Explicit CUDA tests compare normalization and full-network training/inference on random and real inputs, including the channels-last actor path. GPU comparisons bound both relative L2 error and peak-scaled error by two bf16 epsilon units, 1.5625%, plus a small absolute allowance. Direct line-feature comparisons require exact equality. No Triton dependency is required for reference mode.
