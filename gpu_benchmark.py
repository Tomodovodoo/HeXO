"""Measure sparse GPU rules against the native CPU rules on identical actions.

Run: python gpu_benchmark.py --batches 1 64 512 2048 --placements 96
Numbers exclude initialization and device transfers, include exact incremental
729-pattern updates, and use wall time with CUDA synchronization. This does not
compare policy strength or native search with a one-ply GPU policy.
"""
import argparse
import json
import time

import torch

from gpu_games import BatchedHexo, generate_games, recommended_batch
from hexo import Game


def benchmark(batch, placements, device):
    actions = [(8*i, 0) for i in range(placements)]
    tensors = torch.tensor(actions, device=device)[:, None, :].expand(-1, batch, -1)
    gpu = BatchedHexo(batch, device=device, capacity=placements)
    # Warm up allocation and kernels with the same tensor shapes.
    for action in tensors[:min(8, placements)]:
        gpu.step(action)
    gpu.reset()
    if gpu.device.type == "cuda":
        torch.cuda.synchronize(gpu.device)
        torch.cuda.reset_peak_memory_stats(gpu.device)
    started = time.perf_counter()
    for action in tensors:
        gpu.step(action)
    if gpu.device.type == "cuda":
        torch.cuda.synchronize(gpu.device)
    gpu_seconds = time.perf_counter()-started
    gpu_memory = torch.cuda.max_memory_allocated(gpu.device) if gpu.device.type == "cuda" else None
    native = [Game() for _ in range(batch)]
    started = time.perf_counter()
    for q, r in actions:
        for game in native:
            game.play(q, r)
    native_seconds = time.perf_counter()-started
    assert torch.all(gpu.counts == placements).item()
    assert gpu.features[0].cpu().tolist() == native[0].features()
    for game in native:
        game.close()
    return {"batch": batch, "placements": placements, "device": str(gpu.device),
            "gpu_seconds": gpu_seconds, "native_seconds": native_seconds,
            "gpu_placements_per_second": batch*placements/gpu_seconds,
            "native_placements_per_second": batch*placements/native_seconds,
            "speed_ratio_native_over_gpu": native_seconds/gpu_seconds,
            "gpu_peak_allocated_bytes": gpu_memory}


def benchmark_actor(batch, placements, device, candidates, epsilon):
    import tempfile
    from pathlib import Path
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="hexo-actor-benchmark-") as temporary:
        path = str(Path(temporary) / "table.npy")
        np.save(path, np.zeros(729, dtype=np.int32))
        tasks = [dict(seed=i+741, evaluation=False, tables=[path, path], max_stones=placements)
                 for i in range(batch)]
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        results = generate_games(tasks, candidates=candidates, epsilon=epsilon, device=device)
        elapsed = time.perf_counter()-started
        completed = sum(record["winner"] >= 0 for record, _ in results)
        stones = sum(len(record["cells"]) for record, _ in results)
        memory = torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None
        # Validate resulting games against the native rules, outside timing.
        for record, samples in results[:8]:
            game = Game()
            for index, (q, r, owner) in enumerate(record["cells"]):
                if index >= 3:
                    assert np.array_equal(samples["features"][index-3], game.features())
                assert owner == game.player
                game.play(q, r)
            assert record["winner"] == game.winner
            game.close()
        return {"actor": "gpu-pattern-one-ply", "batch": batch, "max_placements": placements,
                "device": device, "candidates": candidates, "epsilon": epsilon,
                "seconds": elapsed, "games_per_second": batch/elapsed,
                "placements_per_second": stones/elapsed, "completed": completed,
                "truncated": batch-completed, "gpu_peak_allocated_bytes": memory,
                "native_replayed_games": min(8, batch)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 64, 512, 2048])
    parser.add_argument("--placements", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--actor", action="store_true", help="Measure full sampled-policy self-play instead of rules only")
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=.1)
    args = parser.parse_args()
    if args.placements < 1 or any(batch < 1 for batch in args.batches):
        parser.error("placements and batches must be positive")
    torch.set_num_threads(2)
    print(json.dumps({"torch": torch.__version__, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}))
    if args.actor:
        print(json.dumps({"recommended_batch": recommended_batch(args.candidates, args.placements, args.device)}))
    for batch in args.batches:
        result = (benchmark_actor(batch, args.placements, args.device, args.candidates, args.epsilon)
                  if args.actor else benchmark(batch, args.placements, args.device))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
