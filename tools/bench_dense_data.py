"""CPU benchmark of the dense learner's batch pipeline on a run's shards (the run is only read).

--shards N copies the N newest shards into a temporary run; --shards 0 reads the whole run in place. Policy files
go to a temporary directory either way. --workers 0 renders in this process and reports examples/s with the time
per example of each stage (window.sample, dense_data.examples, dense_data.collate_arrays); --workers W consumes a
dense_data.Renderers pool of W processes and reports examples/s, the consumer's share of time spent waiting for a
batch, its time per batch in dense_learn.pad and the CPU cores it used (torch on --threads threads). Timing covers
--batches batches after one warm-up batch; the learner settings are the run's config.json with --batch as the
batch size, and cheap rows are retained as the learner retains them (cheap_row_fraction keyed by the run seed).
"""
import argparse
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dense_config
import dense_data
import dense_learn


def copy_shards(run, count, target):
    """Copy the files of the `count` newest shards of `run` into target/shards; returns their names."""
    names = [p.name for p in dense_data.shard_dirs(run)[-count:]]
    for name in names:
        source, dest = Path(run)/'shards'/name, Path(target)/'shards'/name
        dest.mkdir(parents=True)
        for file in (*dense_data.FILES, 'manifest.json', dense_data.SIDECAR):
            if (source/file).exists():
                shutil.copyfile(source/file, dest/file)
    return names


def in_process(run, settings, count, policy_dir, seed=0, run_seed=0):
    """{examples_per_second, <stage>_ms per example} of rendering `count` batches after a warm-up batch here, from a
    window retaining cheap rows as the learner's (settings.cheap_row_fraction keyed by `run_seed`)."""
    window = dense_data.ReplayWindow(run, settings.window_capacity, settings.window_min_rows, settings.window_expand_per_row,
                                     settings.window_taper, settings.validation_fraction, policy_dir,
                                     settings.cheap_row_fraction, run_seed)
    rng, options = np.random.default_rng(seed), dense_data.target_options(settings)
    stages = dict(sample=0., examples=0., collate=0.)
    for k in range(count+1):
        t0 = time.perf_counter()
        refs = window.sample(rng, settings.batch, settings.recency)
        t1 = time.perf_counter()
        rendered = dense_data.examples(window, refs, rng, **options)
        t2 = time.perf_counter()
        dense_data.collate_arrays(*rendered)
        t3 = time.perf_counter()
        if k:
            for stage, seconds in zip(stages, (t1-t0, t2-t1, t3-t2)):
                stages[stage] += seconds
    total = count*settings.batch
    return dict(examples_per_second=total/sum(stages.values()), **{f'{k}_ms': 1000*v/total for k, v in stages.items()})


def pooled(run, settings, count, workers, policy_dir, seed=0, run_seed=0):
    """{examples_per_second, wait_fraction, pad_ms per batch, consumer_cores} of consuming `count` batches of a
    Renderers pool (cheap rows retained as the learner's, keyed by `run_seed`) after a warm-up batch, padding each
    bucket as the learner does."""
    stream = dense_data.Renderers(run, settings, [seed], workers, policy_dir=policy_dir, run_seed=run_seed)
    try:
        next(stream)
        wait = pad = 0.; examples = 0
        started, cpu = time.perf_counter(), time.process_time()
        for _ in range(count):
            t0 = time.perf_counter()
            batch = next(stream)
            t1 = time.perf_counter()
            for bucket in batch.values():
                dense_learn.pad(bucket, dense_learn.QUANTUM)
            wait += t1-t0; pad += time.perf_counter()-t1
            examples += sum(len(b['counts']) for b in batch.values())
        elapsed = time.perf_counter()-started
        return dict(examples_per_second=examples/elapsed, wait_fraction=wait/elapsed, pad_ms=1000*pad/count,
                    consumer_cores=(time.process_time()-cpu)/elapsed)
    finally:
        stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', type=Path, default=Path('runs/dense-v1'))
    parser.add_argument('--shards', type=int, default=8, help='newest shards to copy; 0 reads the whole run in place')
    parser.add_argument('--batches', type=int, default=8)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threads', type=int, help="torch CPU threads of this process (default: torch's own)")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    config = dense_config.load(args.run)
    settings = replace(config.learner, batch=args.batch)
    with tempfile.TemporaryDirectory(prefix='bench-dense-', ignore_cleanup_errors=True) as temporary:
        run = args.run
        if args.shards:
            run = Path(temporary)/'run'
            copy_shards(args.run, args.shards, run)
        policy_dir = Path(temporary)/'policies'
        if args.workers:
            result = pooled(run, settings, args.batches, args.workers, policy_dir, run_seed=config.seed)
        else:
            result = in_process(run, settings, args.batches, policy_dir, run_seed=config.seed)
    print(f'shards {args.shards or "all"}, {args.batches} batches of {args.batch}, workers {args.workers}: '
          + ', '.join(f'{k} {v:.3f}' for k, v in result.items()))


if __name__ == '__main__':
    main()
