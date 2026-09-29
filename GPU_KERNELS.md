# Opt-in dense GPU kernels

`--net-kernels fused` selects Triton kernels for fixed line features, masked normalization with activation and its backward pass, and inference LineConv. Actors also use channels-last with this mode. The learner keeps the current NCHW layout and the reference differentiable LineConv. `reference` remains the default for both commands.

Keep cuDNN for HexConv. This checkout already implements each hex convolution as one masked 3x3 convolution. Its tensor-core kernels are the largest single cost, but the avoidable work is around them: small synchronous index transfers, layout copies, full-activation normalization intermediates, and the matrices and skewed copies used by inference LineConv. The direct LineConv kernel tiles pixels and channels together, which makes channels-last usable without those matrices. Normalization preserves the reference's bf16 rounding points and centred variance computation.

Checkpoint parameter names, config, digest and serialized tensors do not include the execution mode. Loading a checkpoint defaults to `reference`, including a checkpoint saved while training with `fused`. The learner flag is process-local. CPU execution uses the existing operations.

## Measurements

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
