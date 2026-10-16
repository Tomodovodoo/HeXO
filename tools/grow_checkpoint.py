"""Seed a deeper learner variant from a checkpoint without changing what the network computes.

python tools/grow_checkpoint.py RUN SOURCE VARIANT --blocks N

SOURCE is a checkpoint id (variant/step). The new checkpoint RUN/checkpoints/VARIANT/<step> holds the source's EMA
weights with blocks appended up to N; each new block's second convolution is zero, so every new block adds
nothing to its residual stream and the grown network's outputs equal the source's. Its manifest names the source as
seed_source (dense_learn.is_seed): the learner trains from these weights with a fresh optimizer.
With --adopt, RUN/live-variant.json then names VARIANT and its model settings, so the learner, actors, evaluator
and reanalysis (dense_config.load) all follow the grown variant once restarted; an --adopt retry after an
interruption adopts the already published seed. Without --adopt, the tool prints the learner flags (--variant and
every --model-* setting the grown network differs in) for an experiment beside the live variant.
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
        published = json.loads((target/'manifest.json').read_text(encoding='utf-8'))
        if not (args.adopt and published.get('seed_source') == args.source
                and published['model'].get('blocks') == args.blocks):
            raise SystemExit(f'{target} exists')
        adopt(args.run, args.variant, hexnet.load_model(target/'ema.pt').config)
        return
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
        adopt(args.run, args.variant, model.config)
    else:
        live = asdict(dense_config.load(args.run).model)
        flags = [f'--model-{k.replace("_", "-")} {v}' for k, v in asdict(model.config).items() if live.get(k) != v]
        print('learner flags:', ' '.join([f'--variant {args.variant}', *flags]))


def adopt(run, variant, config):
    """Atomically make `variant`, with model settings `config`, the run's live learner variant."""
    live = run/dense_config.LIVE_VARIANT
    pending = live.with_name(live.name+'.tmp')
    pending.write_text(json.dumps(dict(variant=variant, model=asdict(config))), encoding='utf-8')
    os.replace(pending, live)
    print(f'{live}: live learner variant {variant}')


if __name__ == '__main__':
    main()
