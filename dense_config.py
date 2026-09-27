"""Run configuration for the dense self-play stack; every tunable lives here.

`config.json` in the run directory is the single source of settings for the
actor, learner and evaluator processes. Learner variants override the `learner`
section per process and record the effective values in each checkpoint manifest.
"""
import argparse
from dataclasses import dataclass, asdict, field, fields
import json
from pathlib import Path
import time

SCHEMA = 'hexo-dense-run-v1'


@dataclass(frozen=True)
class ModelSettings:
    blocks: int = 6
    channels: int = 96
    pool_every: int = 3
    line_length: int = 11
    value_hidden: int = 128
    head_channels: int = 32
    aux_heads: bool = True


@dataclass(frozen=True)
class ActorSettings:
    games_in_flight: int = 128
    leaf_batch: int = 256
    full_sims: int = 64          # recorded policy targets come from these searches
    cheap_sims: int = 12         # value-only positions; no policy row
    full_fraction: float = .25   # KataGo playout-cap randomization share
    root_samples: int = 16       # Gumbel m
    max_plies: int = 256
    tactics: bool = True         # exact win/must-block classification inside the tree
    cache_positions: int = 4096
    shard_games: int = 32
    opening_random_plies: float = 2.  # mean of an exponential; sampled from the raw policy
    resign_disabled: bool = True


@dataclass(frozen=True)
class LearnerSettings:
    variant: str = 'main'
    batch: int = 256
    lr: float = 3e-4
    warmup_steps: int = 300
    weight_decay: float = 1e-4
    ema: float = .999
    samples_per_row: float = 4.  # train presentations per generated row (KataGo ~4)
    window_min_rows: int = 20000
    window_expand_per_row: float = .4
    window_taper: float = .65
    window_capacity: int = 2000000
    recency: float = 0.
    bootstrap_weight: float = 1.  # weight of TD(lambda) value rows from capped games; 0 = mask
    td_lambda: float = .9
    short_value_horizon: int = 16
    value_weight: float = 1.5
    short_value_weight: float = .5
    opponent_policy_weight: float = .15
    future_weight: float = .5
    validation_fraction: float = .03
    export_every: int = 500
    protect_steps: int = 3000     # no replacement for this many steps after start or copy
    replace_interval: int = 2000  # steps between replacement checks
    replace_margin: float = 50.   # Elo the source must lead by beyond interval overlap
    perturb: float = .2           # relative perturbation of copied continuous settings


@dataclass(frozen=True)
class EvaluationSettings:
    games: int = 64               # per internal comparison, colour-swapped opening pairs
    sims: int = 64
    root_samples: int = 16
    max_plies: int = 256
    tactics: bool = True
    anchor_every: int = 5         # rate against external anchors every N checkpoints
    anchor_games: int = 32
    seal_ms: int = 100
    opening_suite: str = 'standard-v1'


@dataclass(frozen=True)
class RunConfig:
    schema: str = SCHEMA
    created_at: float = 0.
    seed: int = 1740
    device: str = 'cuda'
    model: ModelSettings = field(default_factory=ModelSettings)
    actor: ActorSettings = field(default_factory=ActorSettings)
    learner: LearnerSettings = field(default_factory=LearnerSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)


SECTIONS = dict(model=ModelSettings, actor=ActorSettings, learner=LearnerSettings, evaluation=EvaluationSettings)


def from_dict(data):
    if data.get('schema') != SCHEMA:
        raise ValueError(f'Expected a {SCHEMA} configuration')
    parts = {name: cls(**data[name]) for name, cls in SECTIONS.items()}
    return RunConfig(**{k: v for k, v in data.items() if k not in SECTIONS}, **parts)


def load(run):
    return from_dict(json.loads((Path(run)/'config.json').read_text(encoding='utf-8')))


def save(run, config):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    path = run/'config.json'
    if path.exists():
        raise FileExistsError(f'{path} exists; runs never change configuration in place')
    pending = path.with_suffix('.json.tmp')
    pending.write_text(json.dumps(asdict(config), indent=2), encoding='utf-8')
    pending.replace(path)
    return config


def add_arguments(parser, section, prefix=''):
    """Expose one settings dataclass as --name flags (dest prefix+name); None means "keep the config value"."""
    for item in fields(section):
        kind, dest = type(item.default), prefix+item.name
        if kind is bool:
            parser.add_argument('--'+dest.replace('_', '-'), dest=dest, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument('--'+dest.replace('_', '-'), dest=dest, type=kind, default=None)


def override(settings, args, prefix=''):
    """Apply non-None parsed flags onto a settings dataclass instance."""
    values = {item.name: getattr(args, prefix+item.name) for item in fields(settings)
              if getattr(args, prefix+item.name, None) is not None}
    return type(settings)(**{**asdict(settings), **values})


def main():
    parser = argparse.ArgumentParser(description='Create a dense run directory with its configuration')
    parser.add_argument('--run', required=True)
    parser.add_argument('--seed', type=int, default=RunConfig.seed)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=RunConfig.device)
    # actor and evaluation share root_samples/max_plies/tactics, so evaluation flags are --eval-*.
    prefixes = dict(evaluation='eval_')
    for name, cls in SECTIONS.items():
        group = parser.add_argument_group(name)
        add_arguments(group, cls, prefixes.get(name, ''))
    args = parser.parse_args()
    parts = {name: override(cls(), args, prefixes.get(name, '')) for name, cls in SECTIONS.items()}
    config = RunConfig(created_at=time.time(), seed=args.seed, device=args.device, **parts)
    save(args.run, config)
    print(json.dumps(asdict(config), indent=2))


if __name__ == '__main__':
    main()
