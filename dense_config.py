"""Run configuration for the dense self-play stack; every tunable lives here.

Run layout, shared by dense_selfplay (actor), dense_learn (learner), dense_eval (evaluator) and dashboard.py:
  config.json                          RunConfig, written once; the single source of settings for every process
  shards/<name>/                       immutable self-play shards (format in dense_data); names sort oldest first
  checkpoints/<variant>/<step:06d>/    model.pt, ema.pt, optimizer.pt, manifest.json (dense_learn); complete once
                                       manifest.json exists; the checkpoint id is '<variant>/<step:06d>'
  champion.json                        {checkpoint, ema_sha256, updated_at}: the checkpoint actors play (dense_eval)
  league.json                          ratings, champion and Elo differences of variant heads; checkpoints the
                                       evaluator passed over have skipped true and elo null (dense_eval)
  evaluator-status.json                evaluator heartbeat and progress (dense_eval.Evaluator), rewritten about
                                       every 2 s while it plays
  evaluations/<a>-vs-<b>/report.json   paired match records, ids with '/' written as '-' (dense_eval)
  actor-status[-<k>].json              heartbeat of actor worker k (none for k = 0), rewritten about every 2 s
  learner-status[-<variant>].json      heartbeat of a learner variant (none for main), rewritten about every 2 s
  events.jsonl                         one line per event: {time, source, kind, message, ...} (log_event)
  metrics/learner-<variant>.jsonl      {time, step, samples_seen, lr, policy_ce, value_bce, short_value_bce, next_ce,
                                       future_bce, samples_per_second, window_rows} every log_every steps, plus
                                       {time, step, samples_seen, <the five losses>, validation: true} per export
  metrics/actor-<k>.jsonl              {time, positions, games_completed, placements_per_second, evals_per_second,
                                       mean_batch, terminal_fraction, mean_plies, checkpoint, paused_seconds} about
                                       every 30 s and at every pause or resume; counters restart with the worker
                                       process (dense_selfplay)
  metrics/gpu.jsonl                    {time, utilization, used_mib, watts, temperature} about every 10 s while a
                                       dashboard watches the run (dashboard.py)
Metrics logs are append-only, one JSON line per write (append_metrics); readers skip a partial last line.
Learner variants override the `learner` section per process and record the effective values in each checkpoint
manifest.
"""
import argparse
from dataclasses import dataclass, asdict, field, fields
import json
from pathlib import Path
import time

from train import write_json

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
    opening_random_plies: float = 2.  # mean of an exponential; sampled from the search policy
    # Cooperative GPU sharing (dense_selfplay.Yield): workers stop searching while a training learner's
    # samples_per_row is below yield_below * learner.samples_per_row and resume at yield_resume * it.
    yield_below: float = .9       # 0 disables
    yield_resume: float = .975
    yield_check_seconds: float = 30.


@dataclass(frozen=True)
class LearnerSettings:
    variant: str = 'main'
    batch: int = 256
    lr: float = 3e-4
    warmup_steps: int = 300
    weight_decay: float = 1e-2    # decoupled (AdamW), conv and linear weights only
    grad_clip: float = 1.
    ema: float = .999
    samples_per_row: float = 4.   # training presentations per generated row (KataGo ~4)
    window_min_rows: int = 100000  # counts full-search rows only
    window_expand_per_row: float = .4
    window_taper: float = .65
    window_capacity: int = 2000000
    recency: float = 0.
    bootstrap_weight: float = 1.  # weight of TD(lambda) value rows from capped games; 0 = mask
    bootstrap_full_only: bool = False  # True: chain TD(lambda) through full-search root values only
    cheap_value_weight: float = .25    # value weight of cheap-search rows (KataGo: 0)
    td_lambda: float = .9
    short_value_horizon: int = 16
    value_weight: float = 1.5
    short_value_weight: float = .5
    opponent_policy_weight: float = .15
    future_weight: float = .5
    validation_fraction: float = .03
    export_every: int = 500
    log_every: int = 20           # steps per metrics/learner-<variant>.jsonl line
    protect_steps: int = 3000     # no replacement for this many steps after start or copy
    replace_interval: int = 2000  # steps between replacement checks
    replace_margin: float = 50.   # Elo the source must lead by beyond interval overlap
    perturb: float = .2           # relative perturbation of copied continuous settings


@dataclass(frozen=True)
class EvaluationSettings:
    games: int = 64               # games per round of every comparison, colour-swapped opening pairs
    previous_games: int = 0       # vs the previous rated checkpoint of the variant, only while idle; 0 = never
    sims: int = 64
    root_samples: int = 16
    max_plies: int = 256
    tactics: bool = True
    anchor_every: int = 5         # rate against external anchors every N checkpoints
    anchor_games: int = 100       # vs Seal, only while idle; 0 = never
    seal_ms: int = 100
    sprt_elo0: float = 0.         # promotion SPRT bounds on candidate minus champion
    sprt_elo1: float = 50.
    sprt_alpha: float = .05
    sprt_beta: float = .05
    sprt_max_games: int = 200     # 'max-games' does not promote
    opening_suite: str = 'standard-v1'
    eval_share: float = .12       # ceiling on the evaluator's playing share of wall time (dense_eval.Pacer)


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


def append_line(path, record):
    """Append `record` to `path` as one JSON line in a single unbuffered write."""
    with Path(path).open('ab', buffering=0) as stream:
        stream.write((json.dumps(record, allow_nan=False)+'\n').encode())


def log_event(run, source, kind, message, **fields):
    """Append one line to <run>/events.jsonl."""
    append_line(Path(run)/'events.jsonl', dict(time=time.time(), source=source, kind=kind, message=message, **fields))


def append_metrics(run, name, **fields):
    """Append {time, **fields} to <run>/metrics/<name>.jsonl."""
    path = Path(run)/'metrics'/f'{name}.jsonl'
    path.parent.mkdir(exist_ok=True)
    append_line(path, dict(time=time.time(), **fields))


def from_dict(data):
    if data.get('schema') != SCHEMA:
        raise ValueError(f'Expected a {SCHEMA} configuration')
    parts = {name: cls(**data[name]) for name, cls in SECTIONS.items()}
    return RunConfig(**{k: v for k, v in data.items() if k not in SECTIONS}, **parts)


def load(run):
    return from_dict(json.loads((Path(run)/'config.json').read_text(encoding='utf-8')))


def save(run, config):
    """Write <run>/config.json; an existing configuration is never replaced."""
    path = Path(run)/'config.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'{path} exists; runs never change configuration in place')
    write_json(path, asdict(config))
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
