"""Seed a deeper learner variant from a checkpoint without changing what the network computes.

python tools/grow_checkpoint.py RUN SOURCE VARIANT --blocks N

SOURCE is a checkpoint id (variant/step). The new checkpoint RUN/checkpoints/VARIANT/<step> holds the source's EMA
weights with blocks appended up to N; each new block's second convolution is zero, so every new block adds
nothing to its residual stream and the grown network's outputs equal the source's. Its manifest names the source as
seed_source (dense_learn.is_seed): the learner trains from these weights with a fresh optimizer.
With --adopt, RUN/live-variant.json then names VARIANT and its model settings, so the learner, actors, evaluator
and reanalysis (dense_config.load) all follow the grown variant once restarted; without it, run the learner with
--variant VARIANT --model-blocks N for an experiment beside the live variant.
"""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
import dense_config  # noqa: E402
import hexnet  # noqa: E402


def grown(source, blocks):
    """`source` with blocks appended up to `blocks`, the new blocks' conv2 zero."""
    if blocks <= source.config.blocks:
        raise ValueError(f'the source already has {source.config.blocks} blocks')
    model = hexnet.HexNet(replace(source.config, blocks=blocks), source.future_target)
    missing, unexpected = model.load_state_dict(source.state_dict(), strict=False)
    added = tuple(f'blocks.{i}.' for i in range(source.config.blocks, blocks))
    if unexpected or not all(k.startswith(added) for k in missing):
        raise ValueError(f'unexpected parameter layout: missing {missing}, unexpected {unexpected}')
    with torch.no_grad():
        for block in model.blocks[source.config.blocks:]:
            block.conv2.weight.zero_()
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('run', type=Path)
    parser.add_argument('source')
    parser.add_argument('variant')
    parser.add_argument('--blocks', type=int, required=True)
    parser.add_argument('--adopt', action='store_true', help='make VARIANT the live learner variant of the run')
    args = parser.parse_args()
    origin = args.run/'checkpoints'/args.source
    step = int(args.source.split('/')[1])
    target = args.run/'checkpoints'/args.variant/f'{step:06d}'
    if target.exists():
        raise SystemExit(f'{target} exists')
    manifest = json.loads((origin/'manifest.json').read_text(encoding='utf-8'))
    source = hexnet.load_model(origin/'ema.pt', future_target=manifest['learner'].get('future_target', 'legacy'))
    model = grown(source, args.blocks)
    probe = torch.zeros(2, 8, 24, 24)
    probe[:, 3], probe[:, 2], probe[:, 5], probe[0, 0, 11, 11], probe[0, 2, 11, 11] = 1, 1, 1, 1, 0
    with torch.no_grad():
        a, b = source.eval()(probe, probe[:, 3:4]), model(probe, probe[:, 3:4])
    worst = max(float((a[k]-b[k]).abs().max()) for k in a)
    if worst > 1e-4:
        raise SystemExit(f'grown network differs from the source by {worst}')
    staging = target.with_name('.'+target.name+'.pending')
    shutil.rmtree(staging, ignore_errors=True)  # an interrupted earlier attempt
    staging.mkdir(parents=True)
    for name in ('ema.pt', 'model.pt'):
        hexnet.save_model(staging/name, model)
    optimizer = torch.optim.AdamW(model.parameters())
    torch.save(dict(optimizer=optimizer.state_dict(), optimizer_started=step, ema_updates=0), staging/'optimizer.pt')
    manifest.update(variant=args.variant, seed_source=args.source, model=hexnet.saved_config(model.config),
                    optimizer_kind='seed', ema_sha256=hexnet.model_digest(model), model_sha256=hexnet.model_digest(model))
    (staging/'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    staging.rename(target)
    print(f'{target}: {model.config.blocks} blocks, max output change {worst:.2e}')
    if args.adopt:
        live = args.run/dense_config.LIVE_VARIANT
        pending = live.with_name(live.name+'.tmp')
        pending.write_text(json.dumps(dict(variant=args.variant, model=asdict(model.config))), encoding='utf-8')
        os.replace(pending, live)
        print(f'{live}: live learner variant {args.variant}')


if __name__ == '__main__':
    main()
