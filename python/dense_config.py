"""Configuration, status and event helpers shared by dense processes.

config.json is written once. CLI overrides apply only to their process.
Run files and launch commands are documented in docs/dense-training.md.
"""
import argparse
from dataclasses import dataclass, asdict, field, fields
import json
from pathlib import Path
import time

from legacy.train import write_json

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
    cheap_root_samples: int = 4  # leave cheap searches enough visits to descend beyond the root
    max_plies: int = 256
    tactics: bool = True         # exact win/must-block classification inside the tree
    cache_positions: int = 4096
    shard_games: int = 32
    opening_random_plies: float = 2.  # mean of an exponential; sampled from the search policy
    # Checkpoint the actor plays (dense_selfplay.resolve): 'champion' (champion.json), 'newest' (newest complete
    # checkpoint of learner.variant) or 'newest_veto' (actor.json: newest unless the evaluator vetoed it).
    model_source: str = 'newest_veto'
    historical_fraction: float = 0.   # share of games in flight against a frozen rated checkpoint
    historical_weighting: str = 'pfsp'  # 'pfsp': weight (1-p)^2, p = champion's expected score; 'uniform'
    historical_pool: int = 8          # distinct historical opponents loaded at once, redrawn per shard
    # Cooperative GPU sharing (dense_selfplay.Yield): workers stop searching while a training learner's
    # samples_per_row is below yield_below * learner.samples_per_row and resume at yield_resume * it.
    yield_below: float = .9       # 0 disables
    yield_resume: float = .975
    yield_check_seconds: float = 30.
    # True: also pause while a learner with phase_rows > 0 is waiting for actors, training or exporting,
    # acknowledging token-held phases after queued GPU work finishes (dense_selfplay.Yield).
    phase_follow: bool = False
    # Solver points inside the search (dense_solver), node budgets; 0 = off. root: forced-win check at each turn
    # start, a proof decides the turn played; finalists: defence check of the k best mid-turn candidates at the last
    # halving boundary, a proven opponent win eliminates the candidate; threat: the opponent's forced win on a
    # flipped turn orders the root samples. solver_async: one awaited worker process instead of in-process queries
    # (identical results).
    solver_root_nodes: int = 0
    solver_finalists: int = 0
    solver_finalist_nodes: int = 0
    solver_threat_nodes: int = 0
    solver_leaf_nodes: int = 0  # opt-in verified forced-win check before evaluating a new leaf; 10 ms per query
    solver_defence: bool = False  # verify certificate-derived turns, admit survivors and bonus them at the root
    solver_defence_candidates: int = 8
    solver_async: bool = True
    # Scheduling of those queries (dense_solver.Schedule). solver_fixed_budgets: every query spends its point's fixed
    # budget and every verdict is awaited (reproducible; evaluation always runs this way). Off: budgets follow the
    # measured slack (solver_slack_fraction of each point's lead time less GUARD_MS, net of queued work), clamped to
    # [solver_min_nodes, solver_cap_nodes]; with solver_gate_weight > 0 forcing material scales them by
    # 1 + weight * g up to solver_gate_cap_nodes; a verdict may cost the loop solver_overrun_fraction of the step
    # time in waits before its game is deferred. solver_workers: foreground worker processes. solver_follow: a side
    # with a proof plays its certificate and every proof labels the rows it decides; solver_deep_nodes (needs
    # solver_follow): background proof of each committed turn, its fixed budget and adaptive minimum, adaptive cap
    # solver_deep_cap_nodes, on one extra idle-priority worker.
    solver_fixed_budgets: bool = True
    solver_workers: int = 1
    solver_slack_fraction: float = .95
    solver_overrun_fraction: float = .05
    solver_min_nodes: int = 32
    solver_cap_nodes: int = 512
    solver_gate_cap_nodes: int = 8192
    solver_gate_weight: float = 0.
    solver_deep_nodes: int = 0
    solver_deep_cap_nodes: int = 65536
    solver_follow: bool = False
    solver_table_mb: int = 32   # adaptive budgets: resident solver table per worker and attacker colour (tactical_proof)
    # End a game at a verified proof (dense_selfplay.SelfPlayGame.adjudicate, reason 'proven'); proven_line_rows also
    # appends the certificate's forced line as search-free rows with exact values.
    adjudicate_proven: bool = False
    proven_line_rows: bool = False
    # Restarts (dense_selfplay.Restarts): a self-play game starts with probability restart_fraction from a position
    # of restarts.json, drawn with probability proportional to regret^(1/restart_temperature); 0 = never.
    restart_fraction: float = 0.
    restart_temperature: float = 1.
    net_kernels: str = 'reference'  # opt-in Triton features, normalization and inference LineConv
    cuda_graphs: bool = False  # reuse bounded CUDA graphs for frozen fused actor models

    def __post_init__(self):
        if self.cheap_root_samples < 1 or self.solver_leaf_nodes < 0:
            raise ValueError('cheap_root_samples must be positive and solver_leaf_nodes nonnegative')
        if self.net_kernels not in ('reference', 'fused'):
            raise ValueError('net_kernels must be reference or fused')
        if self.cuda_graphs and self.net_kernels != 'fused':
            raise ValueError('cuda_graphs requires net_kernels=fused')
        if not 0 <= self.restart_fraction <= 1 or not self.restart_temperature > 0:
            raise ValueError('restart_fraction must lie in [0, 1] and restart_temperature must be positive')


VALUE_TARGETS = ('outcome', 'td', 'calibrated')


@dataclass(frozen=True)
class LearnerSettings:
    variant: str = 'main'
    batch: int = 256
    optimizer: str = 'adamw'
    lr: float = 3e-4
    warmup_steps: int = 300
    weight_decay: float = 1e-2    # decoupled (AdamW), conv and linear weights only
    grad_clip: float = 1.
    ema: float = .999
    samples_per_row: float = 4.   # training presentations per retained row
    # Phased schedule (dense_learn.Phase); 0 = train whenever the pacing allows. > 0: idle (stage 'phase-idle')
    # until the untrained backlog (dense_learn.backlog) reaches phase_rows rows, then train until the pacing limit.
    phase_rows: int = 0
    phase_export: bool = False  # collect enough pacing credit to train through the next checkpoint export
    phase_actors: int = 0       # wait for workers 0..N-1 to acknowledge drained GPU work before a phase
    window_min_rows: int = 100000  # counts full-search rows only
    window_expand_per_row: float = .4
    window_taper: float = .65
    window_capacity: int = 2000000
    recency: float = 0.
    regret_fraction: float = 0.  # share of training batches drawn from the proof restart buffer
    bootstrap_weight: float = 1.  # weight of TD(lambda) value rows from capped games; 0 = mask
    bootstrap_full_only: bool = False  # True: chain TD(lambda) through full-search root values only
    cheap_value_weight: float = .25    # value weight of cheap-search rows
    # Probability that an ordinary cheap row (no full search, no exact label) enters the training index and the
    # pacing count; decided per row from the run seed (dense_data.ReplayWindow). 1 = every row.
    cheap_row_fraction: float = 1.
    td_lambda: float = .9
    # Value target of finished games (dense_data.value_targets): 'outcome' = the hard outcome; 'td' = TD(outcome_lambda)
    # from the outcome through the root values, KataGo-style; 'calibrated' = P(win | root value, plies remaining)
    # refitted from the newest calibration_games finished games at every export (dense_learn.Learner.calibrate).
    # 'td' and 'calibrated' honour bootstrap_full_only like the capped-game chain.
    value_target: str = 'outcome'
    outcome_lambda: float = .98
    calibration_games: int = 4000
    outcome_weight: float = 0.    # coefficient of an extra value-logit BCE against the hard outcome (finished games, rows without a proof)
    deblunder_weight: float = 0.  # blend earlier losing-owner rows toward a transient proof's win; 0 disables
    short_value_horizon: int = 16
    short_value_target: str = 'future'  # future: one root at horizon; average: exponential future roots and outcome
    value_weight: float = 1.5
    short_value_weight: float = .5
    opponent_policy_weight: float = .15
    future_weight: float = .5
    future_target: str = 'legacy'  # legacy: occupancy BCE at 6/20; masked: empty/own/opponent CE at 20 on empty cells
    proven_value_weight: float = 2.  # value weight of rows the solver proved; their target is the proven value
    proof_policy_weight: float = 0.  # mix (search + weight * proof)/(1 + weight); proof-only rows have this loss weight
    validation_fraction: float = .03
    validation_rows: int = 8192   # rows per per-source validation subset (dense_data.ValidationSets limit)
    validation_quota: int = 128   # rows each shard may contribute to a subset (ValidationSets quota)
    export_every: int = 500
    log_every: int = 20           # steps per metrics/learner-<variant>.jsonl line
    protect_steps: int = 3000     # no replacement for this many steps after start or copy
    replace_interval: int = 2000  # steps between replacement checks
    replace_margin: float = 50.   # Elo the source must lead by beyond interval overlap
    perturb: float = .2           # relative perturbation of copied continuous settings
    # Cap in MB on the learner's CUDA caching allocator (dense_learn.Learner.cap_vram); 0 = unlimited. Reaching it
    # raises out-of-memory instead of spilling into shared system memory (the intended failure on Windows).
    vram_reserved_mb: int = 0

    def __post_init__(self):
        if self.optimizer not in ('adamw', 'muon'):
            raise ValueError(f'optimizer must be adamw or muon, not {self.optimizer!r}')
        if self.value_target not in VALUE_TARGETS:
            raise ValueError(f'value_target must be one of {VALUE_TARGETS}, not {self.value_target!r}')
        if self.future_target not in ('legacy', 'masked'):
            raise ValueError(f'future_target must be legacy or masked, not {self.future_target!r}')
        if self.short_value_target not in ('future', 'average') or (self.short_value_target == 'average' and self.short_value_horizon < 1):
            raise ValueError('short_value_target must be future or average, with a positive averaging horizon')
        if not 0. <= self.deblunder_weight <= 1.:
            raise ValueError('deblunder_weight must be between 0 and 1')
        if '@' in self.variant or '/' in self.variant:
            raise ValueError(f"variant {self.variant!r}: '@' marks a search-settings variant (dense_eval) and '/' a step")
        if not 0 <= self.cheap_row_fraction <= 1:
            raise ValueError(f'cheap_row_fraction must lie in [0, 1], not {self.cheap_row_fraction}')
        if self.phase_rows < 0:
            raise ValueError(f'phase_rows must be 0 (off) or positive, not {self.phase_rows}')
        if self.phase_actors < 0 or self.phase_actors and not (self.phase_rows or self.phase_export):
            raise ValueError('phase_actors must be nonnegative and requires phased training')
        if self.phase_export and min(self.export_every, self.batch, self.samples_per_row) <= 0:
            raise ValueError('phase_export requires positive export_every, batch and samples_per_row')
        if not 0 <= self.proof_policy_weight < float('inf'):
            raise ValueError('proof_policy_weight must be finite and nonnegative')
        if not 0 <= self.regret_fraction <= 1:
            raise ValueError('regret_fraction must lie in [0, 1]')


@dataclass(frozen=True)
class EvaluationSettings:
    games: int = 64               # games of each optional comparison (panel top-up, fill), colour-swapped opening pairs
    pool_games: int = 64          # evaluation games in flight; a finished game is replaced at once
    pipeline: bool = False        # use free slots for independent idle comparisons while a session drains
    evidence_share: float = .25   # posterior: most of the pool evidence games may take while a decision is pending
    previous_games: int = 0       # vs the previous rated checkpoint of the variant, only while idle; 0 = never
    sims: int = 64
    root_samples: int = 16
    max_plies: int = 256
    tactics: bool = True
    anchor_every: int = 5         # the champion owes anchor_games more vs Seal per N checkpoints rated during its reign
    anchor_games: int = 100       # champion vs Seal, played before optional work; 0 = never
    anchor_session_games: int = 20  # most Seal anchor games before a pending trial gets its turn
    anchor_on_promotion: bool = True  # every new champion owes anchor_games vs Seal
    seal_ms: int = 100
    decision: str = 'posterior'   # promotion rule: 'posterior' (dense_eval.Evaluator.verdict) or 'sprt'
    promote_confidence: float = .9  # posterior: P(candidate - champion > sprt_elo0) needed to promote (1 - it rejects)
    matchup_prior_elo: float = 30.  # posterior: prior sd of a pair's deviation from the transitive rating difference
    sprt_min_games: int = 64      # direct games vs the champion before any decision or evidence game
    sprt_elo0: float = 0.         # promotion SPRT bounds on candidate minus champion
    sprt_elo1: float = 25.
    sprt_alpha: float = .05
    sprt_beta: float = .05
    sprt_max_games: int = 200     # 'max-games' does not promote
    opening_suite: str = 'standard-v1'  # 'book' (the live book) or a frozen openings/<suite>.json (dense_openings)
    # Live opening book (dense_openings.Book.refresh), refreshed at every champion change and every
    # book_refresh_hours: book_size settled openings of book_min_plies..book_plies placements, each the shortest
    # plausible, unused prefix of a line sampled at book_temperature from the visit counts of book_sims-simulation
    # searches (0: from the policy). An opening retires once book_min_games colour-swapped pairs put its first-player
    # skew interval wholly beyond +-book_max_skew Elo, or after book_short_min_games decisive games when its P1 win
    # z-score reaches book_short_skew_z and its mean length falls below the book_short_quantile of played openings.
    # Skewed openings are replaced by a child below book_plies. An opening also retires when the
    # champion's policy probability of it is below book_min_prob; each refresh the champion challenges a random
    # book_revisit_fraction of the settled openings with an alternative at the same depth. book_weighting 'uniform'
    # or 'least_played' (weight 1 / (1 + pairs)) chooses how matches draw openings.
    book_plies: int = 5
    book_min_plies: int = 3
    book_temperature: float = 1.5
    book_sims: int = 16
    book_size: int = 512
    book_revisit_fraction: float = .25
    book_refresh_hours: float = 6.
    book_max_skew: float = 50.
    book_min_games: int = 16
    book_short_min_games: int = 6
    book_short_skew_z: float = 2.5
    book_short_quantile: float = .25
    book_min_prob: float = 1e-4
    book_weighting: str = 'uniform'
    opening_book: str = ''        # Book.digest of the live book the games are played under; stamped by the evaluator
    eval_share: float = .12       # ceiling on the evaluator's playing share of wall time (dense_eval.Pacer)
    busy_share: float = 1.        # short-burst playing ceiling while actors or learners are active; 1 disables it
    extra_opponents: int = 2      # panel opponents drawn per rated checkpoint with probability ~ p(1-p), idle only
    idle_rematch: bool = True     # replay decision-relevant comparisons while no checkpoint awaits rating
    idle_fill: bool = True        # after all other work, play fill games until a checkpoint awaits rating
    anchor_target_halfwidth: float = 25.  # include champion-Seal uncertainty in fill until this 95% half-width; 0 = no Seal fill
    fill_top: int = 3             # fill comparisons that most narrow these top checkpoints' 95% rating intervals
    veto_margin: float = -30.     # actor.json skips the newest checkpoint once its Elo interval vs its champion lies below this
    max_expected_score: float = .85  # panel, optional and fill pairings only while either side's expected score is at most this
    rebase_on_promotion: bool = True  # a variant registered against the champion follows a new champion until it starts
    # Solver node budgets of both sides of every evaluation game, as ActorSettings.solver_*; 0 = off.
    solver_root_nodes: int = 0
    solver_finalists: int = 0
    solver_finalist_nodes: int = 0
    solver_threat_nodes: int = 0
    solver_defence: bool = False
    solver_defence_candidates: int = 8
    solver_workers: int = 1      # foreground tactical processes; fixed budgets remain unchanged
    solver_gate_cap_nodes: int = 0  # 0 keeps fixed budgets flat; otherwise gate scales toward this cap at weight 3


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
# Settings of earlier versions, ignored when a config or a checkpoint manifest is read.
RETIRED = dict(evaluation=('round_games', 'model_cache', 'uncertainty_parity'), learner=('policy_cache_mb',))


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
    parts = {name: section(name, data[name]) for name in SECTIONS}
    return RunConfig(**{k: v for k, v in data.items() if k not in SECTIONS}, **parts)


def section(name, values):
    """The SECTIONS[name] settings of dict `values`, without its RETIRED keys."""
    return SECTIONS[name](**{k: v for k, v in values.items() if k not in RETIRED.get(name, ())})


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
