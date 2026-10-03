"""Paired checkpoint evaluation, promotion and league reporting.

The loop owns league.json, actor.json and the champion pointer. It plays
colour-swapped opening pairs, checks promotion at decision boundaries, and
refreshes the direct decision summary as finished games change the tally.
Capped games score half a point without establishing a draw.

Commands and rating rules are documented in docs/dense-evaluation.md.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import shlex
from pathlib import Path
import time
import uuid

import numpy as np
import torch

import dense_config
from dense_config import log_event
from dense_stats import pair_scores, pentanomial, summary, sprt, tally
import dense_data
import dense_openings
from dense_posterior import MODEL, Posterior, parents
from dense_solver import Budgets, Schedule
import hexnet
from legacy.arena import Seal
from six_engine import SixEngine
from dense_selfplay import Engine, checkpoints, expected, load, resolve
from hexo import Game
from legacy.klent import digest
from legacy.train import paired_metrics, write_json

SEAL = 'seal'


def anchor_name(settings):
    return settings.external_name if settings.external_engine else SEAL


def anchor_engine(settings):
    return SixEngine(shlex.split(settings.external_engine)) if settings.external_engine else Seal()

PACE_WINDOW = 3600.  # a Pacer banks at most share * PACE_WINDOW seconds of idle credit
STATUS_SECONDS = 2.
# Settings a reused report must share; a report without one of PROTOCOL_DEFAULTS was played at that value.
PROTOCOL = ('sims', 'root_samples', 'max_plies', 'tactics', 'search_graph', 'search_choice', 'q_range_floor', 'opening_suite', 'opening_book', 'seal_ms',
            'external_engine', 'external_name',
            'solver_root_nodes', 'solver_finalists', 'solver_finalist_nodes', 'solver_threat_nodes',
            'solver_defence', 'solver_defence_candidates', 'solver_gate_cap_nodes', 'pipeline')
PROTOCOL_DEFAULTS = dict(opening_book='', search_graph=False, search_choice='gumbel', q_range_floor=0., external_engine='', external_name='seal', solver_root_nodes=0, solver_finalists=0, solver_finalist_nodes=0,
                         solver_threat_nodes=0, solver_defence=False, solver_defence_candidates=8,
                         solver_gate_cap_nodes=0, pipeline=False)
CHAMPION = 'champion'  # the symbolic base of a variant, bound to the champion when its comparison starts
# The PROTOCOL fields of one side's search, which a variant may override; max_plies, opening_suite, seal_ms
# and opening_book belong to the game.
SIDE = ('sims', 'root_samples', 'tactics', 'search_graph', 'search_choice', 'q_range_floor', 'solver_root_nodes', 'solver_finalists', 'solver_finalist_nodes',
        'solver_threat_nodes', 'solver_defence', 'solver_defence_candidates')
REMATCH_SPRT_LIMIT = 2    # a continued champion SPRT stops at this many times sprt_max_games
CALIBRATION_LATER = 3     # later comparisons of a decided checkpoint before `calibration` counts its verdict
RATING_NOTE = ('Bradley-Terry over paired comparisons (caps count half a point to each side); each opening pair is one '
               'observation, with counts pooled across reports and uncertainty adjusted for pair correlation; weak '
               '1000 Elo Gaussian priors on rating differences between adjacent checkpoints; 95% Laplace credible '
               'intervals from joint posterior draws; the first evaluated checkpoint is fixed at 0. Each anchor is '
               'rated jointly from its games, so its Elo is estimated, not assumed.')


class MatchGame:
    """One evaluation game. sides[colour] is a dense_selfplay.Model or SEAL; model sides search `sims`
    placements, choosing by Gumbel score or improved search policy (one tree per distinct model, advanced on every placement); a Seal side
    plays complete turns inline, validated with Game.legal. Ends at a win, `max_plies` placements or when the
    Engine stops it (`reason` 'span'). sims, samples and tactics are one value for both colours or a pair per
    colour; solvers[colour] is that colour's dense_solver.Budgets (None: off). graphs, one value or a pair, selects
    graph search per colour. A model playing both colours with one graph setting shares one tree, searched with the
    first such colour's tactics. choices, one value or a pair, selects the final move per colour; 'policy'
    takes the largest improved policy at unproven roots. Exact roots keep their shortest win or longest resistance.
    floors, one value or a pair, is each colour's q_range_floor (neural_search); colours share a tree only when
    their graph and floor settings agree."""

    def __init__(self, sides, opening, seed, sims, samples, tactics, max_plies, record, seal=None, seal_ms=0,
                 solvers=(None, None), anchor=SEAL, graphs=False, choices='policy', floors=0.):
        self.sides, self.max_plies, self.seal, self.seal_ms, self.record = sides, max_plies, seal, seal_ms, record
        self.anchor = anchor
        self.solvers = solvers
        self.reason, self.error = None, None
        per = lambda value: tuple(value) if isinstance(value, (tuple, list)) else (value, value)
        self.budgets, self.sample_counts, tactics, self.graphs = per(sims), per(samples), per(tactics), per(graphs)
        self.choices, self.floors = per(choices), per(floors)
        self.game, self.moves = Game([tuple(m) for m in opening]), [list(m) for m in opening]
        self.trees = {}
        for colour, side in enumerate(sides):
            key = id(side), self.graphs[colour], self.floors[colour]
            if side != self.anchor and key not in self.trees:
                args = [tuple(m) for m in opening], seed*2+colour, tactics[colour]
                options = {k: v for k, v in dict(graph=self.graphs[colour], q_range_floor=self.floors[colour]).items()
                           if v}
                self.trees[key] = side.tree(*args, **options)
        try:
            self.seal_turns()
        except Exception as error:
            if self.anchor not in sides:
                raise
            self.error = f'{type(error).__name__}: {error}'

    @property
    def budget(self):
        return self.budgets[self.game.player]

    @property
    def samples(self):
        return self.sample_counts[self.game.player]

    @property
    def model(self):
        return self.sides[self.game.player]

    @property
    def tree(self):
        return self.trees[id(self.model), self.graphs[self.game.player], self.floors[self.game.player]]

    @property
    def solver(self):
        return self.solvers[self.game.player]

    def over(self):
        return self.error is not None or self.game.winner >= 0 or len(self.moves) >= self.max_plies

    def play(self, q, r):
        self.game.play(q, r)
        for tree in self.trees.values():
            tree.advance((q, r))
        self.moves.append([q, r])

    def seal_turns(self):
        while not self.over() and self.sides[self.game.player] == self.anchor:
            side, turn = self.game.player, self.seal(self.game, self.seal_ms)
            if not 1 <= len(turn) <= 2:
                raise ValueError(f'{self.anchor} returned {len(turn)} moves for {self.game.remaining} placements')
            for q, r in turn[:self.game.remaining]:
                if not self.game.legal(q, r):
                    raise ValueError(f'{self.anchor} played an illegal placement {q}, {r}')
                self.play(int(q), int(r))
                if self.over():
                    break
            if not self.over() and self.game.player == side:
                raise ValueError(f'{self.anchor} did not complete its turn')

    def searched(self, result):
        action = result['action']
        if self.choices[self.game.player] == 'policy' and not result.get('proven'):
            action = result['actions'][np.argmax(result['policy'])]
        self.play(*map(int, action))
        try:
            self.seal_turns()
        except Exception as error:
            if self.anchor not in self.sides:
                raise
            self.error = f'{type(error).__name__}: {error}'
        return not self.over()

    def label(self, ply, proven, turns, proof_action=None):
        """Match games have no training rows to label when a followed proof arrives."""
        return 0

    def finish(self):
        winner = self.game.winner
        self.game.close()
        for tree in self.trees.values():
            tree.close()
        return dict(self.record, winner=winner, reason=self.reason or ('six-in-a-row' if winner >= 0 else 'cap'),
                    plies=len(self.moves), moves=self.moves, **(dict(error=self.error) if self.error else {}))


def play(games, leaf_batch, heartbeat=lambda finished: None, schedule=None):
    """Run MatchGames to completion in one engine; returns their records in input order. heartbeat(records of
    the finished games, in finishing order) is called after every engine step."""
    engine, records = Engine(leaf_batch, schedule=schedule), {}
    try:
        for game in games:
            if game.over():
                records[id(game)] = game.finish()
            else:
                engine.add(game)
        if records:
            heartbeat(list(records.values()))
        while engine.slots or engine.closing:
            try:
                finished = engine.step()
            except BaseException:
                for game in engine.completed:
                    if id(game) not in records:
                        records[id(game)] = game.finish()
                heartbeat(list(records.values()))
                raise
            for game in finished:
                records[id(game)] = game.finish()
            heartbeat(list(records.values()))
    finally:
        engine.close()
    for record in records.values():
        if 'error' in record:
            raise ValueError(f'Match game failed: {record["error"]}')
    return [records[id(g)] for g in games]


def pair_seed(run_seed, label, pair):
    return int.from_bytes(hashlib.sha256(f'{run_seed}/{label}/{pair}'.encode()).digest()[:4], 'little')


def paired_games(challenger, rival, games, label, config, settings, seal, book, first_pair=0, sides=None, **record):
    """`games` MatchGames: games/2 openings drawn from `book` (dense_openings.Book.draw) by the seed of run seed,
    `label` and pair index (pairs numbered from `first_pair`), each played once with the candidate as colour 0 and
    once as colour 1. `sides` is (challenger's, rival's) EvaluationSettings of their search (sims, root_samples,
    tactics and solver budgets), by default both `settings`; max_plies and seal_ms come from `settings`."""
    sides = sides or (settings, settings)
    if games < 2 or games % 2:
        raise ValueError('Paired matches need an even game count of at least two')
    out = []
    for pair in range(first_pair, first_pair+games//2):
        seed = pair_seed(config.seed, label, pair)
        opening = book.draw(seed)
        for colour in (0, 1):
            players, search = [rival, rival], [sides[1], sides[1]]
            players[colour], search[colour] = challenger, sides[0]
            out.append(MatchGame(players, opening, seed, [s.sims for s in search], [s.root_samples for s in search],
                                 [s.tactics for s in search], settings.max_plies,
                                 dict(record, pair=pair, seed=seed, opening=[list(m) for m in opening], challenger_color=colour),
                                 seal, settings.seal_ms, [Budgets.of(s) for s in search], anchor_name(settings),
                                 [s.search_graph for s in search], choices=[s.search_choice for s in search],
                                 floors=[s.q_range_floor for s in search]))
    return out


def split_id(cid):
    """(checkpoint, variant name) of league id cid: `<checkpoint>@<name>` for a variant, name None for a
    checkpoint."""
    checkpoint, _, name = cid.partition('@')
    return checkpoint, name or None


def side_settings(settings, overrides):
    """One side's EvaluationSettings: `settings` with `overrides` (SIDE fields); root_samples, unless overridden,
    is at most the side's sims."""
    out = replace(settings, **overrides)
    out = out if 'root_samples' in overrides else replace(out, root_samples=min(out.root_samples, out.sims))
    Schedule.of(out)
    return out


def parse_settings(assignments):
    """{field: value} of `key=value` strings over SIDE, typed like the EvaluationSettings field (booleans: true,
    false, 1, 0); a variant must override at least one field."""
    defaults, out = dense_config.EvaluationSettings(), {}
    for text in assignments:
        key, sep, value = text.partition('=')
        key = key.strip().replace('-', '_')
        if not sep or key not in SIDE:
            raise ValueError(f'{text!r}: variant settings are key=value over {", ".join(SIDE)}')
        kind = type(getattr(defaults, key))
        if kind is bool:
            if value.strip().lower() not in ('true', 'false', '1', '0'):
                raise ValueError(f'{text!r}: {key} is true or false')
            out[key] = value.strip().lower() in ('true', '1')
        else:
            out[key] = kind(value)
    if not out:
        raise ValueError('A variant overrides at least one setting')
    if out.get('sims', 1) < 1 or out.get('root_samples', 1) < 1:
        raise ValueError('sims and root_samples are at least 1')
    return out


def requests(run):
    """{variant id: path} of the registrations waiting in <run>/variant-requests (`register`)."""
    return {json.loads(path.read_text())['id']: path for path in sorted((Path(run)/'variant-requests').glob('*.json'))}


def adopt(league, run):
    """Append to league['variants'] the registrations waiting in <run>/variant-requests (`requests`) whose id it
    lacks, in registration order; the entries already in `league` stay as they are. Returns how many were added.
    `write_league` removes the request files once league.json holds them."""
    known = {v.get('registered_as', v['id']) for v in league.setdefault('variants', [])}
    new = [json.loads(path.read_text()) for cid, path in requests(run).items() if cid not in known]
    league['variants'] += sorted(new, key=lambda v: v['registered_at'])
    return len(new)


def register(run, checkpoint, name, settings):
    """Register variant `<checkpoint>@<name>` of a rated league checkpoint with overrides `settings`
    (`parse_settings`) and log a 'variant' event; returns its entry. `checkpoint` CHAMPION registers against the
    symbolic champion: id `champion@<name>` with checkpoint None until the evaluator binds it (`Evaluator.bind`).
    The entry records base (`checkpoint` as given), registered_as (its id at registration) and on_champion
    (whether `checkpoint` was the champion then). The registration is written as
    <run>/variant-requests/<id>.json, never to league.json: the evaluator, league.json's only writer, adopts it
    (`adopt`) on its next step. The name is letters, digits, '.', '_' or '-'. Registering an id again (in the
    league, by its current id or registered_as, or waiting) with the same settings returns the existing entry, with
    other settings raises ValueError; a new champion variant whose `<champion>@<name>` is already registered raises
    ValueError too."""
    run = Path(run)
    if not name or not all(ch.isalnum() or ch in '._-' for ch in name):
        raise ValueError(f'{name!r}: a variant name is letters, digits, ".", "_" or "-"')
    path = run/'league.json'
    league = json.loads(path.read_text()) if path.exists() else dict(champion=None, checkpoints=[])
    if checkpoint == CHAMPION:
        if league.get('champion') is None:
            raise ValueError('The league has no champion yet')
    else:
        base = next((c for c in league['checkpoints'] if c['id'] == checkpoint), None)
        if base is None or base.get('skipped') or base.get('elo') is None:
            raise ValueError(f'{checkpoint} is not a rated league checkpoint')
    config = dense_config.load(run)
    Budgets.of(side_settings(config.evaluation, settings))
    cid = f'{checkpoint}@{name}'
    waiting = requests(run)
    existing = next((v for v in league.get('variants', []) if cid in (v['id'], v.get('registered_as'))), None) \
        or (json.loads(waiting[cid].read_text()) if cid in waiting else None)
    if existing is not None:
        if existing['settings'] != settings:
            raise ValueError(f'{cid} is registered with settings {existing["settings"]}; register another name')
        return existing
    taken = f'{league.get("champion")}@{name}'
    if checkpoint == CHAMPION and (any(taken in (v['id'], v.get('registered_as')) for v in league.get('variants', []))
                                   or taken in waiting):
        raise ValueError(f'{taken} is already registered; register the champion variant under another name')
    entry = dict(id=cid, checkpoint=None if checkpoint == CHAMPION else checkpoint, name=name, settings=settings,
                 base=checkpoint, registered_as=cid, on_champion=checkpoint in (CHAMPION, league.get('champion')),
                 registered_at=time.time(), elo=None, elo_interval=None, matches=[])
    target = run/'variant-requests'/f'{cid.replace("/", "-")}.json'
    target.parent.mkdir(exist_ok=True)
    write_json(target, entry)
    log_event(run, 'evaluator', 'variant', f'{cid} registered: ' + ', '.join(f'{k}={v}' for k, v in settings.items()),
              candidate=cid, checkpoint=checkpoint, settings=settings)
    return entry


def settle_path(run, checkpoint):
    return Path(run)/'settle-requests'/f'{hashlib.sha256(checkpoint.encode()).hexdigest()}.json'


def request_settle(run, checkpoint):
    request = dict(checkpoint=checkpoint, requested_at=time.time())
    path = settle_path(run, checkpoint)
    path.parent.mkdir(exist_ok=True)
    write_json(path, request)
    return request


def observations(reports):
    """The `Posterior` results of `reports`: (candidate, opponent, `pentanomial` of its games) per report."""
    return [(r['candidate'], r['opponent'], pentanomial(r['games'])) for r in reports]


def ready(settings, games, disagree):
    """Whether a posterior verdict may decide: at least sprt_min_games direct games and the direct and pooled
    intervals do not `disagree`. No uncertainty bound beyond that: P(better) already carries the sd of delta."""
    return games >= settings.sprt_min_games and not disagree


def judge(settings, games, disagree, leader, p_better):
    """The posterior decision once `ready`: 'promote' when the candidate is the pooled `leader` and p_better
    (P(delta > sprt_elo0)) >= promote_confidence, 'reject' when p_better <= 1 - promote_confidence, else None."""
    if not ready(settings, games, disagree):
        return None
    return 'promote' if leader and p_better >= settings.promote_confidence else \
        'reject' if p_better <= 1-settings.promote_confidence else None


def oriented(games, candidate, side):
    """`games` of a report whose candidate is `candidate`, seen from `side` (either player): challenger_color is
    side's colour, and seed is (candidate, seed) so pairs of different reports never merge in `tally`."""
    flip = side != candidate
    return [dict(g, seed=(candidate, g['seed']), challenger_color=1-g['challenger_color'] if flip else g['challenger_color'])
            for g in games]


def report_path(run, candidate, opponent):
    return Path(run)/'evaluations'/f'{candidate.replace("/", "-")}-vs-{opponent.replace("/", "-")}'/'report.json'


def games_digest(games):
    """Identity of a report's game list: a digest of each game's pair, colour, winner and length, in order."""
    rows = [(g['pair'], g['challenger_color'], g['winner'], g['plies']) for g in games]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


def report_name(report):
    """The directory name of `report` under evaluations/."""
    return report_path('', report['candidate'], report['opponent']).parent.name


def make_report(candidate, opponent, records, shas, settings, overrides=None, report_id=None):
    """Report of a finished comparison; `shas` maps checkpoint ids to their ema.pt digests; `overrides` ({id:
    settings} of its variant sides) is stored when not empty; `report_id` is the report's `id` (a new one when
    None), kept while pairs are appended."""
    return dict(id=report_id or uuid.uuid4().hex, candidate=candidate, opponent=opponent, created_at=time.time(),
                candidate_sha256=shas[candidate],
                opponent_sha256=opponent if opponent == anchor_name(settings) else shas[opponent],
                settings=asdict(settings), **({'overrides': overrides} if overrides else {}),
                metrics=paired_metrics(records), summary=summary(records), games=records)


def solve(ids, anchor, edges):
    """Bradley-Terry Elo per id from edges (a, b, points of a, games); the anchor is 0 and ids outside its
    connected component get None. Newton iterations with backtracking (checkpoint_league.solve_ratings)."""
    connected = {anchor}
    while True:
        before = len(connected)
        for a, b, _, _ in edges:
            if a in connected or b in connected:
                connected.update((a, b))
        if len(connected) == before:
            break
    variables = [i for i in ids if i in connected and i != anchor]
    index = {name: k for k, name in enumerate(variables)}
    rows = [(a, b, w, n) for a, b, w, n in edges if a in connected]
    matrix = np.zeros((len(rows), len(variables)))
    for r, (a, b, _, _) in enumerate(rows):
        if a != anchor: matrix[r, index[a]] = 1.
        if b != anchor: matrix[r, index[b]] = -1.
    wins = np.array([w for _, _, w, _ in rows]); totals = np.array([n for _, _, _, n in rows])
    objective = lambda v: float(np.sum(totals*np.logaddexp(0., matrix@v)-wins*(matrix@v)))
    x = np.zeros(len(variables))
    for _ in range(100):
        if not len(x):
            break
        p = 1/(1+np.exp(-np.clip(matrix@x, -700, 700)))
        gradient = matrix.T@(totals*p-wins)
        if np.max(np.abs(gradient)) < 1e-9:
            break
        step = np.linalg.solve(matrix.T@(matrix*(totals*p*(1-p))[:, None]), gradient)
        scale, old = 1., objective(x)
        while objective(x-scale*step) > old and scale > 1e-8:
            scale *= .5
        x -= scale*step
        if np.max(np.abs(scale*step)) < 1e-9:
            break
    ratings = {name: None for name in ids}
    ratings[anchor] = 0.
    ratings.update({name: float(x[k]*400/math.log(10)) for name, k in index.items()})
    return ratings


def rate(ids, anchor, reports, samples=2048, seed=1740):
    """(point ratings, 95% intervals, joint posterior draws per rated id) from actual paired scores.

    The existing pentanomial Posterior pools counts across reports, adjusts for pair correlation and uses weak
    rating priors to keep perfect sweeps finite. No prior pseudo-games enter the observed score. Matchup deviations
    are disabled for the global Bradley-Terry table; decision posteriors are fitted separately. The anchor is 0
    and ids outside its connected component remain unrated. Draws use the joint Laplace approximation.
    """
    results = [(a, b, c) for a, b, c in observations(reports) if sum(c)]
    connected = {anchor}
    while True:
        before = len(connected)
        for a, b, _ in results:
            if a in connected or b in connected:
                connected.update((a, b))
        if len(connected) == before:
            break
    rated = [name for name in ids if name in connected]
    point = {name: 0. if name == anchor else None for name in ids}
    draws = {name: [] for name in rated}
    if len(rated) > 1:
        fit = Posterior(rated, anchor, [(a, b, c) for a, b, c in results if a in connected],
                        matchup_prior=0., parents=parents(rated))
        point.update({name: fit.rating(name) for name in rated})
        rng = np.random.default_rng(seed)
        joint = rng.multivariate_normal(fit.mode, fit.cov, size=samples, method='cholesky')
        draws = {name: [0.]*samples if name == anchor else joint[:, fit.index[name]].tolist() for name in rated}
    return point, {name: np.quantile(v, [.025, .975]).tolist() if v else [0., 0.] for name, v in draws.items()}, draws


_reports = {}


def load_reports(run, settings=None):
    """Every evaluations/*/report*.json: each pairing's report.json and the reports `Evaluator.open` archived beside it
    (cached per path and file identity), which pool into the league ratings. With `settings`, only the current
    report.json files played under its PROTOCOL: decisions read a pairing's games from its current report alone
    (`Evaluator.games`), so an archive whose protocol matches again (a setting changed and restored) stays out of them."""
    reports = []
    for path in sorted((Path(run)/'evaluations').glob('*/report.json' if settings else '*/report*.json')):
        stat = path.stat()
        stamp = stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        if _reports.get(path, (None,))[0] != stamp:
            _reports[path] = stamp, json.loads(path.read_text())
        reports.append(_reports[path][1])
    return [r for r in reports if settings is None or same_protocol(r, settings)]


def same_protocol(report, settings):
    """Whether `report` was played under the PROTOCOL of `settings`. Under the live book that includes opening_book
    (dense_openings.Book.digest of its openings), which changes only at a book refresh: a report is reused while the
    book keeps its openings, and a refresh that changes them starts every comparison afresh. A frozen suite's name
    fixes its openings (opening_book ''); a report without a PROTOCOL_DEFAULTS field was played at its default."""
    return all(report['settings'].get(k, PROTOCOL_DEFAULTS.get(k)) == getattr(settings, k) for k in PROTOCOL)


def payoff(reports, ratings=None):
    """Payoff matrix {a: {b: {wins, losses, capped, games, p}}}: a's results against b summed over every report in
    which they met, in either role (matrix[b][a] mirrors matrix[a][b]); p is the Bradley-Terry expected score of a
    against b under `ratings` (caps half a point), None when either is unrated or no ratings are given."""
    matrix = {}
    for r in reports:
        s = r['summary']
        for a, b, wins, losses in ((r['candidate'], r['opponent'], s['wins'], s['losses']),
                                   (r['opponent'], r['candidate'], s['losses'], s['wins'])):
            cell = matrix.setdefault(a, {}).setdefault(b, dict(wins=0, losses=0, capped=0, games=0))
            cell['wins'] += wins; cell['losses'] += losses; cell['capped'] += s['capped']; cell['games'] += s['games']
    for a, row in matrix.items():
        for b, cell in row.items():
            known = ratings is not None and ratings.get(a) is not None and ratings.get(b) is not None
            cell['p'] = expected(ratings[a], ratings[b]) if known else None
    return matrix


def informative(p, cap):
    """Whether a pairing with expected score p (either side's) can inform the ratings: 1 - cap <= p <= cap."""
    return max(p, 1-p) <= cap


def closeness(p):
    """Ranking key of a pairing with expected score p: p(1-p) to two decimals, so near-equal pairings tie."""
    return round(p*(1-p), 2)


def panel_members(league, candidate, incumbent, count, cap=1.):
    """The `count` most informative panel opponents of `candidate` and `incumbent`, derived from the league as it
    is now: rated checkpoints other than those two and the current champion, neither skipped nor demoted, whose
    expected score p against the champion (league Elo, Bradley-Terry) is `informative` under `cap`; ranked by
    `closeness`, then wider elo_interval, then newer league entry."""
    entries = league['checkpoints']
    champion = league.get('champion')
    reference = next((c['elo'] for c in entries if c['id'] == champion and c.get('elo') is not None), 0.)
    pool = [(k, c, expected(reference, c['elo'])) for k, c in enumerate(entries)
            if c.get('elo') is not None and not c.get('skipped') and not c.get('demoted') and c['id'] not in (candidate, incumbent, champion)]
    width = lambda c: (lambda i: i[1]-i[0])(c.get('elo_interval') or [0., 0.])
    ranked = sorted((x for x in pool if informative(x[2], cap)), key=lambda x: (-closeness(x[2]), -width(x[1]), -x[0]))
    return [c['id'] for _, c, _ in ranked[:count]]


def panel_result(members, candidate, incumbent, matrix):
    """Panel veto. candidate_score and incumbent_score are the decisive win shares of each against the `members`
    both have decisive games with (summed over those members); z is the two-proportion statistic with pooled
    variance and `veto` holds when the candidate is worse at 95% (z < -1.96). Scores and z are None without
    decisive games on either side."""
    wc = nc = wi = ni = 0
    for m in members:
        c, i = matrix.get(candidate, {}).get(m, {}), matrix.get(incumbent, {}).get(m, {})
        dc, di = c.get('wins', 0)+c.get('losses', 0), i.get('wins', 0)+i.get('losses', 0)
        if dc and di:
            wc += c['wins']; nc += dc; wi += i['wins']; ni += di
    out = dict(members=list(members), candidate_score=wc/nc if nc else None, incumbent_score=wi/ni if ni else None,
               z=None, veto=False)
    if nc and ni:
        pooled = (wc+wi)/(nc+ni)
        se = math.sqrt(pooled*(1-pooled)*(1/nc+1/ni))
        out['z'] = (wc/nc-wi/ni)/se if se else 0.
        out['veto'] = out['z'] < -1.96
    return out


def rematch_pair(run, a, b, settings):
    """Report orientation for more a-b games: a-vs-b, else b-vs-a, skipping a report that holds an SPRT record or a
    posterior verdict (it only grows through its own decision) or was played under another protocol; None when
    neither can grow."""
    for x, y in ((a, b), (b, a)):
        path = report_path(run, x, y)
        report = json.loads(path.read_text()) if path.exists() else None
        if report is None or (same_protocol(report, settings) and not {'sprt', 'posterior'} & report['metrics'].keys()):
            return x, y
    return None


def variant_heads(entries, rated=lambda c: True):
    """{variant: entry}: each variant's newest (highest step) entry that is `rated`, not skipped and not demoted;
    the heads of league differences, panels and rematches."""
    heads = {}
    for c in entries:
        if rated(c) and not c.get('skipped') and not c.get('demoted') and c['step'] >= heads.get(c['variant'], c)['step']:
            heads[c['variant']] = c
    return heads


def calibration(league, reports):
    """Diagnostic of the posterior's stated uncertainty; no decision reads it. For every checkpoint whose posterior
    verdict holds its `Evaluator.snapshot` (opponent, protocol, matchup_prior, reports) and was stated under the
    current likelihood (model dense_posterior.MODEL), whose snapshotted reports
    (every input of the verdict's posterior, the direct one among them) all still begin with the games it was decided
    on (`games_digest` of that prefix: extended at most, not replaced) and which has at least CALIBRATION_LATER later
    comparisons (reports with it under that protocol that are new or have grown since the verdict; a pairing's archive
    that matches the protocol again is a report of its own), shift = delta now - delta at the verdict, both r_cid -
    r_opponent + their matchup deviation over the reports of that protocol with the verdict's matchup prior. Reports
    the snapshot lists as `earlier` (present at the verdict but not among its inputs, such as older archives) are
    neither later comparisons nor part of the posterior now. A calibrated Gaussian posterior expects E[shift^2] = sd_then^2 - sd_now^2 (the variance the later games
    resolved). Returns {count, predicted_sd (mean delta_sd at the verdicts), expected_rms (root mean sd_then^2 -
    sd_now^2), realised_rms (root mean shift^2)}, the three None without a counted verdict: realised_rms well below
    expected_rms means the posterior overstates its variance."""
    rated = [c['id'] for c in league['checkpoints']+league.get('variants', []) if c.get('elo') is not None and not c.get('skipped')]
    protocol = lambda settings: tuple(settings.get(k, PROTOCOL_DEFAULTS.get(k)) for k in PROTOCOL)
    posteriors, then, now, shifts = {}, [], [], []
    for entry in league['checkpoints']:
        verdict, cid = entry.get('verdict') or {}, entry['id']
        if not {'opponent', 'protocol', 'matchup_prior', 'reports'} <= verdict.keys() or verdict.get('delta_sd') is None \
                or verdict.get('model') != MODEL or cid not in rated or verdict['opponent'] not in rated:
            continue
        key = protocol(verdict['protocol'])
        earlier = frozenset(verdict.get('earlier', ()))
        group = [r for r in reports if protocol(r['settings']) == key and r.get('id') not in earlier]
        named = {}  # report name -> its reports of the protocol: the current one and archives it matches again
        for r in group:
            named.setdefault(report_name(r), []).append(r)
        direct = report_path('', cid, verdict['opponent']).parent.name
        intact = lambda name, snap: next((r for r in named.get(name, []) if len(r['games']) >= snap['games']
                                          and games_digest(r['games'][:snap['games']]) == snap['digest']), None)
        decided = {name: intact(name, snap) for name, snap in verdict['reports'].items()}
        if direct not in decided or None in decided.values():
            continue
        later = sum(cid in (r['candidate'], r['opponent'])
                    and len(r['games']) > (verdict['reports'][name]['games'] if r is decided.get(name) else 0)
                    for name, rs in named.items() for r in rs)
        if later < CALIBRATION_LATER:
            continue
        model = key, verdict['matchup_prior'], earlier
        if model not in posteriors:
            ids = rated+list(dict.fromkeys(r['opponent'] for r in group if r['opponent'] not in rated))
            posteriors[model] = Posterior(ids, ids[0], observations(group), verdict['matchup_prior'], parents(ids))
        mean, sd = posteriors[model].difference(cid, verdict['opponent'])
        then.append(verdict['delta_sd']); now.append(sd); shifts.append(mean-verdict['delta'])
    root = lambda values: math.sqrt(max(0., float(np.mean(values)))) if values else None
    return dict(count=len(shifts), predicted_sd=float(np.mean(then)) if then else None,
                expected_rms=root([a*a-b*b for a, b in zip(then, now)]), realised_rms=root([x*x for x in shifts]))


def write_league(run, league, config, top=None):
    """Adopt waiting variant registrations (`adopt`; their request files are removed after the write), recompute ratings, the payoff matrix and the ladder (`top`
    checkpoints, default config.evaluation.fill_top) from every report among rated (not skipped) ids, the variants
    of those ids and Seal, then publish league.json; skipped entries keep elo and elo_interval null. The ladder and
    differences hold checkpoints only. The ratings pool the reports of every protocol, opening-book states included,
    so a refresh never empties the ladder; the promotion posterior, panels and rematches use protocol-matching
    reports only (`same_protocol`)."""
    run = Path(run)
    adopt(league, run)
    ids = [c['id'] for c in league['checkpoints'] if not c.get('skipped')]
    variants = [v for v in league['variants'] if v['checkpoint'] in ids]
    rated = ids+[v['id'] for v in variants]
    all_ids = {c['id'] for c in league['checkpoints']+league['variants']}
    all_reports = load_reports(run)
    known_anchors = {SEAL, anchor_name(config.evaluation), *league.get('anchors', {}),
                     *(r['opponent'] for r in all_reports if r['candidate'] in rated and r['opponent'] not in all_ids)}
    reports = [r for r in all_reports if r['candidate'] in rated and
               r['opponent'] in rated+list(known_anchors)]
    anchors = list(dict.fromkeys(r['opponent'] for r in reports if r['opponent'] not in rated)) or [anchor_name(config.evaluation)]
    names = rated+anchors
    point, intervals, draws = rate(names, ids[0], reports, seed=config.seed) if ids else ({}, {}, {})
    for c in league['checkpoints']+league['variants']:
        c['elo'], c['elo_interval'] = point.get(c['id']), intervals.get(c['id'])
    # a-b intervals of the variant heads come from the same joint draws.
    heads = [c['id'] for _, c in sorted(variant_heads(league['checkpoints'], lambda c: point.get(c['id']) is not None).items())]
    difference = lambda a, b: dict(a=a, b=b, elo_delta=point[a]-point[b], interval=np.quantile(
        np.subtract(draws[a], draws[b]), [.025, .975]).tolist() if draws[a] else [0., 0.])
    league['differences'] = [difference(a, b) for i, a in enumerate(heads) for b in heads[i+1:]]
    league['ladder_top'] = config.evaluation.fill_top if top is None else top
    best = sorted((c['id'] for c in league['checkpoints'] if point.get(c['id']) is not None and not c.get('demoted')),
                  key=lambda k: -point[k])[:league['ladder_top']]
    league['ladder'] = [difference(a, b) for i, a in enumerate(best) for b in best[i+1:]]
    points = lambda t: t['wins']+t['capped']/2
    league['anchors'] = {}
    for anchor in anchors:
        anchored = {}
        for r in sorted((r for r in reports if r['opponent'] == anchor and r['candidate'] in ids),
                        key=lambda r: ids.index(r['candidate'])):
            total = anchored.setdefault(r['candidate'], dict(wins=0, losses=0, capped=0, games=0))
            for k in total:
                total[k] += r['summary'][k]
        matches = [dict(checkpoint=cid, **t, elo_delta=400*math.log10((points(t)+.5)/(t['games']-points(t)+.5)))
                   for cid, t in anchored.items()]
        league['anchors'][anchor] = dict(elo=point.get(anchor), elo_interval=intervals.get(anchor),
                                         games=sum(len(r['games']) for r in reports if r['opponent'] == anchor),
                                         matches=matches, latest_delta=matches[-1]['elo_delta'] if matches else None)
    league['matrix'] = payoff(reports, point)
    league['calibration'] = calibration(league, reports)
    league['openings'] = dense_openings.summary(run, reports)
    league['rating_note'] = RATING_NOTE
    league['updated_at'] = time.time()
    write_json(run/'league.json', league)
    adopted = {v.get('registered_as', v['id']) for v in league['variants']}
    for cid, path in requests(run).items():
        if cid in adopted:
            path.unlink()


class Pacer:
    """Ceiling on the evaluator's playing share of wall time, a token bucket: credit accrues at `share` per
    wall second, capped at `share * window` while not playing (a new pacer starts full), playing spends
    `weight` per second, and wait(tick) sleeps while credit is negative, calling tick()
    before each sleep of at most a second and returning early once tick() returns true; share 1 with weight 1
    never waits. used() is the weighted playing
    share of the last `window` seconds (of the pacer's lifetime while shorter)."""

    def __init__(self, share, window=PACE_WINDOW, clock=time.monotonic, sleep=time.sleep):
        if not 0 < share <= 1:
            raise ValueError('eval_share must be in (0, 1]')
        self.share, self.window, self.clock, self.sleep = share, window, clock, sleep
        self.started = self.last = clock()
        self.credit, self.intervals = share*window, []

    def refill(self, now):
        self.credit = min(self.share*self.window, self.credit+self.share*(now-self.last))
        self.last = now

    def played(self, start, end, weight=1):
        self.refill(start)
        self.credit -= (weight-self.share)*(end-start)
        self.last = end
        self.intervals = [i for i in self.intervals if i[1] > end-self.window]+[(start, end, weight)]

    def used(self):
        now = self.clock()
        since = max(self.started, now-self.window)
        return sum(w*max(0., min(b, now)-max(a, since)) for a, b, w in self.intervals)/max(now-since, 1e-9)

    def ready(self):
        """Whether a new game may start now: the credit is not negative."""
        self.refill(self.clock())
        return self.credit >= -1e-6

    def wait(self, tick=lambda: None):
        self.refill(self.clock())
        while self.credit < -1e-6 and not tick():
            self.sleep(min(1., -self.credit/self.share))
            self.refill(self.clock())


class BusyPacer:
    """Limit playing duty under fresh actor/learner activity, without banking idle credit.

    After at least 100 ms of charged work, wait for the evaluator's queued GPU work and then yield
    work * (1/share - 1) seconds, calling tick() before each sleep of at most a second and ending the yield
    early once tick() returns true. This controls wall time, not a measured fraction of GPU capacity.
    """

    def __init__(self, run, share, clock=time.monotonic, sleep=time.sleep, now=time.time):
        if not 0 < share <= 1:
            raise ValueError('busy_share must be in (0, 1]')
        self.run, self.share, self.clock, self.sleep, self.now = Path(run), share, clock, sleep, now
        self.work, self.checked, self.busy = 0., -float('inf'), False

    def occupied(self):
        if self.share == 1:
            return False
        if self.clock()-self.checked >= 1.:
            self.checked, self.busy = self.clock(), False
            for pattern, stages in (('actor-status*.json', ('playing',)),
                                    ('learner-status*.json', ('training', 'exporting'))):
                for path in self.run.glob(pattern):
                    try:
                        status = json.loads(path.read_text(encoding='utf-8'))
                    except (OSError, ValueError):
                        continue
                    if self.now()-float(status.get('updated_at') or 0.) <= 120. and status.get('stage') in stages:
                        self.busy = True
                        return True
        return self.busy

    def played(self, start, end):
        if self.share < 1:
            self.work += end-start

    def wait(self, synchronize, tick=lambda: None):
        if not self.occupied():
            self.work = 0.
            return
        if self.work < .1:
            return
        started = self.clock()
        synchronize()
        delay = (self.work+self.clock()-started)*(1/self.share-1)
        self.work = 0.
        until = self.clock()+delay
        while self.clock() < until and not tick():
            self.sleep(min(1., max(0., until-self.clock())))


class Pool:
    """A continuous pool of MatchGames on one Engine, the evaluator's counterpart of the actors' games in flight.
    Games belong to lanes (any hashable pairing key); add(lane, games) starts them at once, step() advances the
    engine once and returns [(lane, record)] of the games that finished, so a finished game's slot can be refilled
    before the next step. running() counts games in flight (`running`); moves() is the
    placements played after their openings by the games in flight. close() stops the engine's solver and
    releases the games still in flight, which are discarded."""

    def __init__(self, leaf_batch, schedule=None):
        self.engine, self.games, self.ready = Engine(leaf_batch, schedule=schedule), {}, []
        self.started = time.perf_counter()

    def solver(self):
        """dense_solver.Solver.summary of the pool's queries so far, None before its first solver search."""
        solver = self.engine.solver
        return solver.summary(time.perf_counter()-self.started) if solver else None

    def close(self):
        self.engine.close()
        for _, game in self.games.values():
            game.finish()
        self.games.clear()

    def synchronize(self):
        """Finish queued GPU work without collecting predictions or advancing any game."""
        for _, _, _, handle in self.engine.inflight:
            if handle[2] is not None:
                handle[2].synchronize()

    def add(self, lane, games):
        for game in games:
            if game.over():
                self.ready.append((lane, game.finish()))
            else:
                self.engine.add(game)
                self.games[id(game)] = lane, game

    def step(self):
        out, self.ready = self.ready, []
        if self.engine.slots:
            for game in self.engine.step():
                out.append((self.games.pop(id(game))[0], game.finish()))
        return out

    def running(self, lane=None, kind=None):
        """Games in flight, including those that finished on creation and wait for step(): of `lane`, else of
        lanes (a, b, kind) of `kind`, else all."""
        held = [lane for lane, _ in self.games.values()]+[lane for lane, _ in self.ready]
        return sum(h == lane if lane else h[2] == kind if kind else True for h in held)

    def moves(self):
        return sum(len(game.moves)-len(game.record['opening']) for _, game in self.games.values())


def even(games):
    """`games` rounded down to whole colour pairs."""
    return int(games)//2*2


def placements(records):
    """Placements played in `records` after their openings."""
    return sum(r['plies']-len(r['opening']) for r in records)


def public(verdict):
    """A `verdict` for status, reports and events: without its posterior, and without the numbers of a candidate
    that has no direct game yet (delta, delta_sd, p_better and pooled None)."""
    out = {k: v for k, v in verdict.items() if k != 'posterior'}
    if not out['direct']['games']:
        out.update(delta=None, delta_sd=None, p_better=None, pooled=None)
    return out


def brief(cid, opponent, kind, verdict=None):
    """One status `pending` entry: {candidate, opponent, kind, p_better, delta, delta_sd, direct {games, wins,
    losses, capped, effective_pairs}} of `verdict` (`public` rules: the numbers are None without a direct game),
    without a verdict the numbers None and the direct record 0."""
    shown = public(verdict) if verdict else {}
    direct = shown.get('direct', {})
    return dict(candidate=cid, opponent=opponent, kind=kind, p_better=shown.get('p_better'), delta=shown.get('delta'),
                delta_sd=shown.get('delta_sd'), direct={k: direct.get(k, 0) for k in ('games', 'wins', 'losses', 'capped', 'effective_pairs')})


def match_entry(opponent, report):
    s = report['summary']
    return dict(opponent=opponent, wins=s['wins'], losses=s['losses'], capped=s['capped'], games=s['games'],
                opening_pair_p=report['metrics']['opening_pair_p'], elo_delta=s['elo_delta'],
                pair_score_lower=s['pair_score_lower'])


class Evaluator:
    """The `loop` evaluator of one run (module contract) with `settings` in place of config.evaluation.
    step() does one unit of work, played as a `session` of the continuous pool (`Pool`, pool_games in flight).
    publish() maintains evaluator-status.json: {stage ('idle', 'playing', 'throttled' or 'failed'), updated_at,
    comparison ({candidate, opponent, kind 'champion', 'variant', 'evidence', 'previous', 'anchor', 'panel',
    'incumbent', 'sprt', 'replacement', 'fill' or 'generalization'} of the session's main lane, or null), pool ([{candidate,
    opponent, kind, running, share}] per lane: games in flight and the games in flight it wants), started_at
    (epoch seconds when the session began), games_played (the main lane's report games plus its finished games
    whose colour partner still runs), games_planned (sprt_max_games for the champion, which plays until decided,
    superseded or that many; else the comparison's target), tally (`tally` of every recorded game of the main
    lane's pair, `direct`, plus its finished games whose colour partner still runs, with the SPRT fields for kinds
    'champion' and 'sprt'; refreshed with the status as games finish; null while idle), decision (the
    newest `verdict` of a checkpoint or variant decision, `public`, with candidate, opponent and, while pending,
    next [[a, b], ...] lanes; its direct score includes finished games awaiting their colour partner, while
    p_better, delta, delta_sd, effective_pairs and the pooled fit refresh with the status from complete pairs;
    a finished game awaiting its colour partner changes only the direct tally. Status refreshes preserve the
    decision result until the next decision check; null before any), pending ([`brief`] of every pending decision: the unrated
    checkpoints against the champion, then the variants without a verdict against their checkpoints; refreshed
    every step and, for the one being decided, after every completed colour pair; it leaves once decided),
    placements_played and
    placements_per_second (the session's placements after openings, including the
    games in flight, and per second), mean_placements (per finished game of the main lane in the session, null
    before any),
    settings (the effective EvaluationSettings), net_kernels (this process's model kernel mode), fill_uncertainty
    (the current fill plan's targets and predicted 95% half-widths; None for other work),
    eval_share (the Pacer's share: 1 with --once), backlog (unrated
    checkpoint ids at the last step), eval_share_used (Pacer.used), vram (hexnet.vram()), error, solver (the solver
    settings' dense_solver.Budgets, `sides` {player: Budgets} of the session's players with their own budgets
    (variants), and the session's query statistics, Pool.solver, as `queries`; None while every budget in use is
    0)}. Match events,
    one per lane at the end of a session, carry the lane's games and placements, the session's seconds and
    worker_seconds (seconds times the lane's share of the session's placements)."""

    def __init__(self, run, config, settings, pacer):
        self.openings = dense_openings.Book(run, settings)
        missed = self.openings.reconcile(dense_openings.stamp(run))
        settings = replace(settings, opening_book=self.openings.digest())
        config = replace(config, evaluation=settings)
        self.run, self.config, self.settings, self.pacer = Path(run), config, settings, pacer
        self.anchor_id = anchor_name(settings)
        self.busy_pacer = BusyPacer(run, settings.busy_share, clock=pacer.clock, sleep=pacer.sleep)
        path = self.run/'league.json'
        self.league = json.loads(path.read_text()) if path.exists() else dict(champion=None, checkpoints=[])
        if self.league.get('champion') and self.league.get('reign_anchor', SEAL) != self.anchor_id:
            self.league.update(reign_anchor=self.anchor_id, reign_pooled=True,
                               reign_games=sum(len(r['games']) for r in self.seal_reports(self.league['champion'])))
            write_json(path, self.league)
        self.pool_reign()
        if self.league['checkpoints'] and (missed or 'matrix' not in self.league or 'calibration' not in self.league
                                           or self.league.get('ladder_top') != settings.fill_top):
            write_league(self.run, self.league, self.config, self.settings.fill_top)
        self.models, self.seal, self.written, self.fill_target, self.deciding, self.reviewed = {}, None, 0., None, None, False
        previous_status = self.run/'evaluator-status.json'
        previous = json.loads(previous_status.read_text()) if previous_status.exists() else {}
        self.anchor_turn = previous.get('anchor_turn', True) if previous.get('anchor_champion') == self.league['champion'] else True
        self.book, self.next, self.ids, self.shas = {}, {}, {}, {}
        self.failed_seal = set()
        self.status = dict(stage='idle', updated_at=None, comparison=None, pool=[], started_at=None, games_played=0,
                           games_planned=0, tally=None, decision=None, pending=[], placements_played=0, mean_placements=None,
                           placements_per_second=None, settings=asdict(settings), eval_share=pacer.share, backlog=[],
                           eval_share_used=0., error=None, solver=self.solver_status(None),
                           net_kernels=config.actor.net_kernels, fill_uncertainty=None,
                           anchor_turn=self.anchor_turn, anchor_champion=self.league['champion'])

    def set_anchor_turn(self, turn):
        """Persist which task gets the next turn, including across evaluator restarts."""
        champion = self.league['champion']
        if (self.anchor_turn, self.status['anchor_champion']) != (turn, champion):
            self.anchor_turn = turn
            self.publish(True, anchor_turn=turn, anchor_champion=champion)

    def solver_status(self, pool, names=()):
        """The status `solver` field for `pool` (None: no session) playing the players `names`: the evaluation
        settings' Budgets, `sides` {name: Budgets} of the players whose own budgets (`side`) differ from them, and
        `queries`; None while every budget in use is 0."""
        budgets = Budgets.of(self.settings)
        sides = {name: Budgets.of(self.side(name)) for name in dict.fromkeys(names) if name != self.anchor_id}
        sides = {name: b for name, b in sides.items() if b != budgets}
        if not budgets.active and not any(b.active for b in sides.values()):
            return None
        return dict(asdict(budgets), sides={name: asdict(b) for name, b in sides.items()},
                    queries=pool.solver() if pool else None)

    def publish(self, force=False, **fields):
        """Update the status; rewrite the file when forced or STATUS_SECONDS after the last write."""
        self.status.update(fields)
        if force or time.monotonic()-self.written >= STATUS_SECONDS:
            self.written = time.monotonic()
            ceiling = self.busy_pacer.share if self.busy_pacer.occupied() else 1.
            write_json(self.run/'evaluator-status.json',
                       dict(self.status, updated_at=time.time(), eval_share=min(self.pacer.share, ceiling),
                            eval_share_used=self.pacer.used(), vram=hexnet.vram()))
            self.point()

    def point(self):
        """Rewrite actor.json {checkpoint, reason, updated_at, vetoed} and log an 'actor_model' event when its
        checkpoint or vetoed list changes. The checkpoint is the newest complete checkpoint of config.learner.variant
        (reason 'newest'); once that checkpoint is demoted or the Elo interval (`tally`) of its games against the
        champion it met (`met`, else the current champion; its `direct` games, counted once their colour pair is complete) lies entirely below
        veto_margin, it joins `vetoed` (the last 16 kept) for good and the checkpoint is the highest-rated one
        neither demoted, skipped nor vetoed (else the champion) until a newer export. Without a checkpoint of the
        variant nothing is written."""
        own = [e[0] for e in checkpoints(self.run) if e[0].split('/')[0] == self.config.learner.variant]
        if not own:
            return
        path, newest = self.run/'actor.json', own[-1]
        pointer = json.loads(path.read_text()) if path.exists() else dict(checkpoint=None, vetoed=[])
        vetoed, entry = list(pointer['vetoed']), self.entry(newest) or {}
        rival = self.met(newest) or self.league['champion']
        why = None
        if newest not in vetoed and rival not in (None, newest):
            interval = tally(self.direct(newest, rival))['elo_interval']
            if entry.get('demoted'):
                why = 'demoted by its panel'
            elif interval and interval[1] < self.settings.veto_margin:
                why = f'Elo {interval[0]:+.0f} to {interval[1]:+.0f} vs {rival}, below {self.settings.veto_margin:+.0f}'
            if why:
                vetoed = (vetoed+[newest])[-16:]
        checkpoint, reason = newest, 'newest'
        if newest in vetoed:
            rated = [c for c in self.league['checkpoints'] if c.get('elo') is not None and not c.get('demoted')
                     and not c.get('skipped') and c['id'] not in vetoed]
            checkpoint = max(rated, key=lambda c: c['elo'])['id'] if rated else self.league['champion']
            reason = f'{newest} vetoed' + (f': {why}' if why else '') + f'; best-rated {checkpoint}'
        if checkpoint != pointer['checkpoint'] or vetoed != pointer['vetoed']:
            write_json(path, dict(checkpoint=checkpoint, reason=reason, updated_at=time.time(), vetoed=vetoed))
            log_event(self.run, 'evaluator', 'actor_model', f'actors play {checkpoint}: {reason}', checkpoint=checkpoint,
                      previous=pointer['checkpoint'], reason=reason, vetoed=vetoed)

    def use(self, *names):
        """Keep exactly the named checkpoints and variants loaded (Seal once, lazily), a variant as its own model
        instance of its checkpoint's weights; games in flight hold their own models, so a dropped model lives
        until its last game finishes."""
        for name in names:
            if name == self.anchor_id:
                self.seal = self.seal or anchor_engine(self.settings)
            elif name not in self.models:
                self.models[name] = load(self.run, self.config, source=(name, self.weights(name)))
        for name in [n for n in self.models if n not in names]:
            del self.models[name]

    def weights(self, name):
        """The ema.pt of checkpoint or variant `name`."""
        return self.run/'checkpoints'/split_id(name)[0]/'ema.pt'

    def overrides(self, name):
        """The settings overrides of variant `name`, {} for a checkpoint or Seal."""
        return (self.entry(name) or {}).get('settings', {}) if split_id(name)[1] else {}

    def side(self, name):
        """The EvaluationSettings that `name` searches with (`side_settings`)."""
        return side_settings(self.settings, self.overrides(name))

    def games(self, a, b):
        """The games of report a-vs-b: the session's copy when it is open, else the file's when it was played
        under the current protocol ([] without one or under another protocol)."""
        if (a, b) in self.book:
            return self.book[a, b]
        path = report_path(self.run, a, b)
        report = json.loads(path.read_text()) if path.exists() else None
        return report['games'] if report and same_protocol(report, self.settings) else []

    def direct(self, a, b):
        """Every recorded game of a against b under the current protocol, from a's side (`oriented`): the games of
        report a-vs-b and of report b-vs-a (`games`, so an open report includes the session's completed pairs)."""
        return oriented(self.games(a, b), a, a)+oriented(self.games(b, a), b, a)

    def open(self, a, b):
        """Load report a-vs-b for appending and number its next opening pair after the highest one it holds, so a
        restart resumes the pairing; the report keeps its id (dense_openings.report_id). A report played under another
        protocol is kept as report-<created_at>-<id>.json beside it (unique per report; still pooled by load_reports)
        with an 'info' event, and a new one starts with a new id."""
        if (a, b) in self.book:
            return
        path = report_path(self.run, a, b)
        old = json.loads(path.read_text()) if path.exists() else None
        if old and not same_protocol(old, self.settings):
            kept = path.with_name(f'report-{int(old["created_at"])}-{dense_openings.report_id(old)}.json')
            path.replace(kept)
            log_event(self.run, 'evaluator', 'info', f'{a} vs {b}: the report played under another protocol is kept as '
                      f'{kept.name}; a new one starts', candidate=a, opponent=b)
            old = None
        self.book[a, b] = list(old['games']) if old else []
        self.next[a, b] = max((g['pair'] for g in self.book[a, b]), default=-1)+1
        self.ids[a, b] = dense_openings.report_id(old) if old else uuid.uuid4().hex

    def start(self, pool, lane):
        """Start the next opening pair of lane (a, b, kind) in `pool`: both colours at once (`paired_games`), each
        side searching with its own settings (`side`)."""
        a, b, _ = lane
        pair, self.next[a, b] = self.next[a, b], self.next[a, b]+1
        pool.add(lane, paired_games(self.models[a], self.anchor_id if b == self.anchor_id else self.models[b], 2, a, self.config, self.settings,
                                    self.seal, self.openings, pair, (self.side(a), self.side(b)), candidate=a, opponent=b))

    def persist(self, a, b, kind, pair):
        """Append a completed colour pair to report a-vs-b and rewrite it (kind 'sprt' recomputes metrics.sprt,
        decision or 'max-games'), so a restart loses only the games in flight, then record it in the opening book; a
        game stopped by 'span' is logged."""
        records = self.book[a, b] = self.book[a, b]+pair
        for name in (a, b):
            if name != self.anchor_id and name not in self.shas:
                self.shas[name] = digest(self.weights(name))
        report = make_report(a, b, records, self.shas, self.settings,
                             {name: self.overrides(name) for name in (a, b) if self.overrides(name)}, self.ids[a, b])
        if kind == 'sprt':
            result = self.test(records)
            report['metrics']['sprt'] = dict(result, decision=result['decision'] or 'max-games')
        path = report_path(self.run, a, b)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
        self.openings.record(pair, self.ids[a, b])
        for record in pair:
            if record['reason'] == 'span':
                log_event(self.run, 'evaluator', 'error', f'{a} vs {b} pair {record["pair"]}: game counted as capped at ply '
                          f'{record["plies"]}, a searched position spans more than the largest crop', candidate=a, opponent=b)

    def session(self, want, planned, stop=None, auxiliary=None, budget=None):
        """Play the pool until want() asks for nothing and the games in flight have finished; returns {lane:
        games finished}. want() -> {(a, b, kind): games in flight wanted, even}, asked at the start and after every
        completed colour pair; a finished game's slot is refilled before the next engine step, and a lane want()
        drops (or shrinks) starts no new games while its running games finish and count. Games in flight never
        exceed pool_games; a lane's games (in flight or finished while their colour partner runs) never exceed
        the games it wants, nor per kind those wanted of that kind, so a draining lane's games count against its
        replacement's and a half-finished pair's slot is not refilled past a budget. In pipeline mode, budget()
        may give {(a, b): games remaining}: those pairings use physical slots for their lane/kind shares, while
        running games and finished halves together must fit the remaining budget. This keeps decision pools
        full when one colour finishes sooner; at most pool_games finished halves await their partners, and
        only complete pairs enter a verdict. The Pacer is charged for
        engine steps and for starting games (Seal plays its first turns then); no game starts while its credit
        is negative, and with nothing running the session then waits and asks want() again. Every completed
        pair is persisted at once. stop(), when supplied (a newer export, settle request, or a checkpoint
        interrupting a variant), is asked after every engine step that finishes a game and at least once a
        second while the session plays or either pacer holds it; once it holds, the session abandons every game in flight and every finished
        half awaiting its colour partner, publishes the idle status at once and logs an 'abandon' event with
        the main lane, games_abandoned and halves_discarded.
        Status shows the session's pairing from its first pass. When auxiliary is supplied, free slots
        may launch independent idle lanes up to two poolfuls per session; admissions stop for higher-priority work,
        but every launched colour pair still drains and is persisted."""
        pool, waiting, added, failed = Pool(self.config.actor.leaf_batch, Schedule.of(self.settings)), {}, {}, {}
        placed, completed, shown_completed = 0, 0, 0
        gate_at, gate_open, miss, miss_at, checked_at = -float('inf'), False, None, 0., -float('inf')
        start, wall = self.pacer.clock(), time.time()
        primary, extra, extra_started = want(), {}, 0
        used_pairs = {(x, y) for a, b, _ in primary for x, y in ((a, b), (b, a))}
        lanes = dict(primary)
        shown, fresh, abandoned = dict(lanes), True, None

        def show(stage, force=False):
            nonlocal shown_completed
            if not force and time.monotonic()-self.written < STATUS_SECONDS:
                return  # the tally is computed only for a write
            main = next(iter(shown), None)
            a, b, kind = main or (None, None, None)
            halves = [r for group in waiting.get(main, {}).values() for r in group if 'error' not in r]
            played = len(self.games(a, b))+len(halves) if a else 0
            done = self.direct(a, b)+oriented(halves, a, a) if a else []
            live = placed+pool.moves()
            score = tally(done, self.test if kind in ('champion', 'sprt') else None)
            decision = self.status['decision']
            if decision and (decision['candidate'], decision['opponent']) == (a, b) and \
                    (decision['direct']['games'] != score['games'] or completed != shown_completed):
                current = public(self.verdict(a, b))
                direct = dict(current['direct'], games=score['games'], wins=score['wins'], losses=score['losses'],
                              capped=score['capped'], elo=score['elo_delta'], interval=score['elo_interval'])
                decision = decision | current | dict(decision=decision['decision'], direct=direct)
                shown_completed = completed
            self.publish(True, stage=stage, comparison=dict(candidate=a, opponent=b, kind=kind) if a else None,
                         pool=[dict(candidate=x, opponent=y, kind=k, running=pool.running((x, y, k)), share=lanes.get((x, y, k), 0))
                               for x, y, k in shown], started_at=wall, games_played=played, games_planned=planned,
                         tally=score, decision=decision, placements_played=live,
                         mean_placements=added[main][1]/added[main][0] if main in added else None,
                         placements_per_second=live/max(self.pacer.clock()-start, 1e-9),
                         solver=self.solver_status(pool, [name for lane in shown for name in lane[:2]]))

        def halt(now=False):
            nonlocal checked_at, abandoned
            if stop is not None and abandoned is None and (now or self.pacer.clock()-checked_at >= 1.):
                checked_at = self.pacer.clock()
                if stop():
                    abandoned = pool.running(), sum(len(group) for groups in waiting.values() for group in groups.values())
            return abandoned is not None

        def throttle(force=False):
            show('throttled', force)
            return halt()
        while True:
            self.busy_pacer.wait(pool.synchronize, throttle)
            if abandoned:
                break
            lanes = dict(primary)
            now = self.pacer.clock()
            if auxiliary is not None and now-gate_at >= 1.:
                gate_open, gate_at = self.pipeline_ready(), now
            may_refill = auxiliary is not None and gate_open
            if may_refill and extra_started < 2*self.settings.pool_games:
                lanes.update({lane: even(min(self.settings.pool_games, target-len(self.games(*lane[:2]))))
                              for lane, target in extra.items() if len(self.games(*lane[:2])) < target
                              and (*lane, self.settings.opening_book) not in self.failed_seal})
            for a, b, _ in lanes:
                self.open(a, b)
            live = {lane for lane, _ in pool.games.values()} | {lane for lane, _ in pool.ready}
            live.update(lane for lane, groups in waiting.items() if any(groups.values()))
            held_lanes = (*lanes, *live) if auxiliary else lanes
            self.use(*dict.fromkeys(name for lane in held_lanes for name in lane[:2]))
            shown.update(lanes)
            ready = self.pacer.ready()
            if lanes and ready:
                remaining = budget() if self.settings.pipeline and budget is not None else {}
                kinds = {kind: sum(n for lane, n in lanes.items() if lane[2] == kind) for _, _, kind in lanes}
                halves = lambda match: sum(len(g) for held, groups in waiting.items() if match(held) for g in groups.values())
                held = lambda lane: pool.running(lane)+halves(lambda h: h == lane)
                occupied = lambda lane: pool.running(lane) if lane[:2] in remaining else held(lane)
                kind_held = lambda kind: pool.running(kind=kind)+halves(lambda h: h[2] == kind and h[:2] not in remaining)
                tick = self.pacer.clock()  # starting games plays Seal's first turns: playing time
                for lane, share in lanes.items():
                    while occupied(lane)+2 <= share and kind_held(lane[2])+2 <= kinds[lane[2]] \
                            and held(lane)+2 <= remaining.get(lane[:2], share) \
                            and pool.running()+2 <= self.settings.pool_games \
                            and (lane in primary or extra_started+2 <= 2*self.settings.pool_games):
                        self.start(pool, lane)
                        if lane in extra:
                            extra_started += 2
                while may_refill and extra_started+2 <= 2*self.settings.pool_games \
                        and pool.running()+2 <= self.settings.pool_games:
                    active = {lane for lane, _ in pool.games.values()} | {lane for lane, _ in pool.ready}
                    active.update(lane for lane, groups in waiting.items() if any(groups.values()))
                    active.update(lanes)
                    names = {name for lane in active for name in lane[:2] if name != self.anchor_id}
                    key = completed, tuple(sorted(names)), len(used_pairs)
                    if key == miss and self.pacer.clock()-miss_at < 1.:
                        break
                    task = auxiliary(used_pairs, names)
                    if task is None:
                        miss, miss_at = key, self.pacer.clock()
                        break
                    miss = None
                    entry, opponent, kind, games = task
                    lane = (entry['id'], opponent, kind)
                    target = games if kind == 'previous' else len(self.games(*lane[:2]))+games
                    if target-len(self.games(*lane[:2])) < 2:
                        used_pairs.update((lane[:2], (opponent, entry['id'])))
                        continue
                    used_pairs.update((lane[:2], (opponent, entry['id'])))
                    extra[lane] = target
                    self.open(*lane[:2])
                    self.use(*dict.fromkeys(name for held_lane in (*active, lane) for name in held_lane[:2]))
                    lanes[lane] = even(min(self.settings.pool_games, target-len(self.games(*lane[:2]))))
                    shown[lane] = lanes[lane]
                    while held(lane)+2 <= lanes[lane] and pool.running()+2 <= self.settings.pool_games \
                            and extra_started+2 <= 2*self.settings.pool_games:
                        self.start(pool, lane)
                        extra_started += 2
                end = self.pacer.clock()
                self.pacer.played(tick, end)
                self.busy_pacer.played(tick, end)
            if not pool.running():
                if lanes and not ready:
                    self.pacer.wait(lambda: throttle(True))
                    if abandoned:
                        break
                    primary = want()  # the wait may have outlasted the pairing (a newer checkpoint)
                    continue
                break
            show('playing', fresh)
            fresh = False
            self.busy_pacer.wait(pool.synchronize, throttle)
            if abandoned:
                break
            tick = self.pacer.clock()
            results = pool.step()
            end = self.pacer.clock()
            self.pacer.played(tick, end)
            self.busy_pacer.played(tick, end)
            paired = False
            for lane, record in results:
                moves = record['plies']-len(record['opening'])
                placed += moves
                group = waiting.setdefault(lane, {}).setdefault(record['pair'], [])
                group.append(record)
                if 'error' in record:
                    log_event(self.run, 'evaluator', 'error', f'{lane[0]} vs {lane[1]} pair {record["pair"]}: '
                              f'game discarded after {record["error"]}', candidate=lane[0], opponent=lane[1])
                if len(group) == 2:
                    del waiting[lane][record['pair']]
                    if any('error' in game for game in group):
                        failed[lane] = failed.get(lane, 0)+1
                        if lane[1] == self.anchor_id and failed[lane] == 2:
                            self.failed_seal.add((*lane, self.settings.opening_book))
                            log_event(self.run, 'evaluator', 'error', f'{lane[0]} vs {self.anchor_id} ({lane[2]}): paused after '
                                      f'{failed[lane]} failed pairs', candidate=lane[0], opponent=self.anchor_id)
                    else:
                        self.persist(*lane, sorted(group, key=lambda r: r['challenger_color']))
                        completed += 1
                        count = added.setdefault(lane, [0, 0])
                        count[0] += 2
                        count[1] += sum(game['plies']-len(game['opening']) for game in group)
                    paired = True
            if paired:
                primary = want()
            if halt(bool(results)):
                break
        if abandoned is None:
            show('playing', True)
        pool.close()
        if abandoned is not None:
            self.publish(True, stage='idle', comparison=None, pool=[], tally=None)
            a, b, kind = next(iter(shown))
            log_event(self.run, 'evaluator', 'abandon', f'{a} vs {b} ({kind}): abandoned {abandoned[0]} games in flight and '
                      f'{abandoned[1]} halves awaiting their colour partner', candidate=a, opponent=b, comparison=kind,
                      games_abandoned=abandoned[0], halves_discarded=abandoned[1])
        seconds = self.pacer.clock()-start
        for (a, b, kind), (games, moves) in added.items():
            records = self.games(a, b)
            s, test = summary(records), self.test(records) if kind in ('champion', 'sprt') else None
            log_event(self.run, 'evaluator', 'match', f'{a} vs {b} ({kind}): +{s["wins"]} -{s["losses"]} ={s["capped"]} after '
                      f'{games} more games' + (f', LLR {test["llr"]:.2f}' if test else ''), candidate=a, opponent=b,
                      comparison=kind, **s, sprt=test, games_added=games, seconds=seconds,
                      worker_seconds=seconds*moves/max(placed, 1), placements=moves)
        self.book.clear()
        return {lane: count[0] for lane, count in added.items()}

    def backlog(self):
        """Whether any unrated checkpoint exists."""
        known = {c['id'] for c in self.league['checkpoints']}
        return any(e[0] not in known for e in checkpoints(self.run))

    def newer(self, cid):
        """Whether an unrated checkpoint of cid's variant other than cid exists (it is newer: cid was chosen as
        the newest)."""
        known = {c['id'] for c in self.league['checkpoints']}
        variant = cid.split('/')[0]
        return any(f'{variant}/{path.name}' not in known and f'{variant}/{path.name}' != cid
                   and (path/'manifest.json').exists() and (path/'ema.pt').exists()
                   for path in (self.run/'checkpoints'/variant).glob('*'))

    def requested(self, cid):
        return settle_path(self.run, cid).exists()

    def dismiss_settle_requests(self, resumed):
        for path in sorted((self.run/'settle-requests').glob('*.json')):
            cid = json.loads(path.read_text())['checkpoint']
            if cid in resumed:
                continue
            log_event(self.run, 'evaluator', 'info', f'{cid}: settle request ignored; no running trial', candidate=cid)
            path.unlink()

    def variants(self):
        """Automatically playable variants, in registration order.

        Imported fixed-size benchmarks remain in league.json for ratings and
        checkpoint history. Their separate workers own their games.
        """
        return [v for v in self.league.setdefault('variants', []) if not v.get('benchmark_only')]

    def entry(self, cid):
        """The league entry of checkpoint or variant cid, or None."""
        return next((c for c in self.league['checkpoints']+self.variants() if c['id'] == cid), None)

    def close(self, a, b):
        """Whether the league's current Elo of a and b (Seal: anchors.seal.elo) makes their pairing `informative`
        under max_expected_score; a side without an Elo counts as informative."""
        elo = lambda k: self.league.get('anchors', {}).get(self.anchor_id, {}).get('elo') if k == self.anchor_id else (self.entry(k) or {}).get('elo')
        ea, eb = elo(a), elo(b)
        return ea is None or eb is None or informative(expected(ea, eb), self.settings.max_expected_score)

    def record(self, a, b, report):
        """Record report as a's match against b (replacing an earlier one) and republish the league."""
        entry = self.entry(a)
        if entry is not None:
            entry['matches'] = [m for m in entry['matches'] if m['opponent'] != b]+[match_entry(b, report)]
        write_league(self.run, self.league, self.config, self.settings.fill_top)

    def promote(self, cid, previous, panel=None):
        self.crown(cid)
        write_json(self.run/'champion.json', dict(checkpoint=cid, ema_sha256=digest(self.run/'checkpoints'/cid/'ema.pt'),
                                                  updated_at=time.time()))
        log_event(self.run, 'evaluator', 'promotion', f'{cid} promoted to champion' + (f' over {previous}' if previous else ''),
                  checkpoint=cid, previous_champion=previous, panel=panel)

    def members(self, entry):
        """The panel members of `entry`: those judged, else `panel_members` of the league now."""
        panel = entry.get('panel', {})
        if 'veto' in panel:
            return panel['members']
        return panel_members(self.league, entry['id'], panel.get('incumbent'), self.settings.extra_opponents,
                             self.settings.max_expected_score) if panel else []

    def needs(self, entry):
        """The panel comparisons `entry` still lacks: [(a, b, kind, games)] topping up its own ('panel') and then
        the incumbent's ('incumbent') protocol-matching games against each current member (`members`) to `games`,
        appended to an existing report where `rematch_pair` allows (a member no report can grow for, or not
        `close` to that side, is left as it is; the veto then judges the members both sides met)."""
        s, panel = self.settings, entry.get('panel', {})
        matrix, out = payoff(load_reports(self.run, s)), []
        for side, kind in ((entry['id'], 'panel'), (panel.get('incumbent'), 'incumbent')):
            for m in self.members(entry):
                short = s.games-matrix.get(side, {}).get(m, {}).get('games', 0)
                pair = rematch_pair(self.run, side, m, s) if short > 0 and self.close(side, m) else None
                if pair:
                    out.append((*pair, kind, short+short % 2))
        return out

    def settle(self):
        """Judge every panel that became complete and is not judged yet. A vetoed checkpoint is marked demoted
        (never promoted or restored again) with a 'regression' event; when it is the champion, the champion becomes
        its most recent predecessor, following panel incumbents, that is neither demoted nor skipped, and stays
        unchanged when there is none."""
        entries = {c['id']: c for c in self.league['checkpoints']}
        for entry in self.league['checkpoints']:
            panel = entry.get('panel')
            if not panel or 'veto' in panel or self.needs(entry):
                continue
            panel.update(panel_result(self.members(entry), entry['id'], panel['incumbent'], payoff(load_reports(self.run, self.settings))))
            score = lambda v: '-' if v is None else f'{v:.3f}'
            log_event(self.run, 'evaluator', 'panel', f'{entry["id"]} vs panel of {len(panel["members"])}: decisive score '
                      f'{score(panel["candidate_score"])}, incumbent {panel["incumbent"]} {score(panel["incumbent_score"])}'
                      + (' - veto' if panel['veto'] else ''), candidate=entry['id'], **panel)
            if panel['veto']:
                entry['demoted'] = True
                restored, seen = panel['incumbent'], {entry['id']}
                while restored in entries and restored not in seen and (entries[restored].get('demoted') or entries[restored].get('skipped')):
                    seen.add(restored)
                    restored = entries[restored].get('panel', {}).get('incumbent')
                restored = restored if restored in entries and restored not in seen else None
                if self.league['champion'] == entry['id'] and restored:
                    self.crown(restored)
                    write_json(self.run/'champion.json', dict(checkpoint=restored, updated_at=time.time(),
                                                              ema_sha256=digest(self.run/'checkpoints'/restored/'ema.pt')))
                champion = self.league['champion'] == restored
                log_event(self.run, 'evaluator', 'regression', f'{entry["id"]} demoted: its panel score is significantly below '
                          f'{panel["incumbent"]}\'s' + (f'; {restored} is champion again' if champion else ''),
                          checkpoint=entry['id'], restored=restored if champion else None, panel=panel)
            write_league(self.run, self.league, self.config, self.settings.fill_top)

    def rematches(self):
        """Idle rematches that can change a decision, first applicable only: the newest checkpoint of a variant
        whose SPRT against the current champion ended 'max-games' continues (up to REMATCH_SPRT_LIMIT *
        sprt_max_games games), then two variant heads whose Elo difference interval straddles +-replace_margin
        play more (up to sprt_max_games between them) while `close`; reports under another protocol are left alone.
        [(a, b, kind, games left)]."""
        s, champion = self.settings, self.league['champion']
        heads = self.heads()
        for c in heads:
            path = report_path(self.run, c['id'], champion)
            report = json.loads(path.read_text()) if c['id'] != champion and path.exists() else None
            if report and same_protocol(report, s) and report['metrics'].get('sprt', {}).get('decision') == 'max-games' \
                    and len(report['games']) < REMATCH_SPRT_LIMIT*s.sprt_max_games:
                return [(c['id'], champion, 'sprt', REMATCH_SPRT_LIMIT*s.sprt_max_games-len(report['games']))]
        matrix, margin = payoff(load_reports(self.run, s)), self.config.learner.replace_margin
        for d in self.league.get('differences', []):
            lo, hi = d['interval']
            played = matrix.get(d['a'], {}).get(d['b'], {}).get('games', 0)
            pair = rematch_pair(self.run, d['a'], d['b'], s)
            if (lo < margin < hi or lo < -margin < hi) and played < s.sprt_max_games and pair and self.close(d['a'], d['b']):
                return [(*pair, 'replacement', s.sprt_max_games-played)]
        return []

    def heads(self):
        """`variant_heads` of the league, newest first."""
        heads = variant_heads(self.league['checkpoints']).values()
        return sorted(heads, key=lambda c: -self.league['checkpoints'].index(c))

    def test(self, records):
        s = self.settings
        return sprt(records, s.sprt_elo0, s.sprt_elo1, s.sprt_alpha, s.sprt_beta)

    def verdict(self, cid, champion):
        """The posterior decision on candidate cid against the champion (module contract), or on variant cid
        against its checkpoint `champion`, from every protocol-matching report among the league's rated
        checkpoints and variants, cid, the opponent and Seal, each rating's prior centred on its parent's
        (dense_posterior.parents), so a candidate without games sits at its previous export: {decision ('promote', 'reject' or None; for a
        variant 'better', 'worse' or None), rule and threshold (the rule in force: 'posterior' with
        promote_confidence, or with decision 'sprt' for a checkpoint 'sprt' with sprt_elo1), model
        (dense_posterior.MODEL, the likelihood the numbers come from), comparison ('champion' or
        'variant'), delta and delta_sd (r_cid - r_champion + their matchup deviation), p_better (P(delta >
        sprt_elo0)), pooled (95% interval of r_cid - r_champion without it), direct {games, wins, losses, capped,
        elo, interval} (`tally` of their `direct` games) with effective_pairs (`Posterior.effective_pairs`), disagree
        (the direct and pooled intervals do not overlap), spread (rating sd of cid and of the champion about the league mean), leader (the rated or candidate
        checkpoint of highest posterior rating; variants never lead)}, plus `posterior` (the Posterior) for
        pairing. A variant needs no lead."""
        s = self.settings
        ids, reports = self.inputs(cid, champion)
        post = Posterior(ids, ids[0], observations(reports), s.matchup_prior_elo, parents(ids))
        mean, sd = post.difference(cid, champion)
        pooled_mean, pooled_sd = post.difference(cid, champion, False)
        pooled = [pooled_mean-1.96*pooled_sd, pooled_mean+1.96*pooled_sd]
        t = tally(self.direct(cid, champion))
        interval = t['elo_interval']
        disagree = bool(interval) and (interval[1] < pooled[0] or interval[0] > pooled[1])
        spread = [post.spread(cid), post.spread(champion)]
        leader = max((i for i in ids if i != self.anchor_id and not split_id(i)[1] and not (self.entry(i) or {}).get('demoted')),
                     key=post.rating)
        p_better = .5*math.erfc((s.sprt_elo0-mean)/(max(sd, 1e-9)*math.sqrt(2)))
        if split_id(cid)[1]:
            kind, rule, threshold = 'variant', 'posterior', s.promote_confidence
            decision = dict(promote='better', reject='worse').get(judge(s, t['games'], disagree, True, p_better))
        else:
            kind, rule = 'champion', s.decision
            threshold = s.sprt_elo1 if rule == 'sprt' else s.promote_confidence
            decision = judge(s, t['games'], disagree, leader == cid, p_better)
        return dict(decision=decision, rule=rule, threshold=threshold, model=MODEL, comparison=kind, delta=mean, delta_sd=sd,
                    p_better=p_better, pooled=pooled, direct=dict(games=t['games'], wins=t['wins'], losses=t['losses'],
                    capped=t['capped'], effective_pairs=post.effective_pairs(cid, champion), elo=t['elo_delta'],
                    interval=interval), disagree=disagree, spread=spread, leader=leader, posterior=post)

    def inputs(self, cid, champion):
        """(ids, reports) of the `verdict` posterior on cid against the champion: the league's rated ids, the
        champion, cid and Seal when it has a report among them; every protocol-matching report among those."""
        rated = [c['id'] for c in self.league['checkpoints']+self.variants() if c.get('elo') is not None and not c.get('skipped')]
        ids = list(dict.fromkeys(rated+[champion, cid]))
        reports = [r for r in load_reports(self.run, self.settings) if r['candidate'] in ids+[self.anchor_id] and r['opponent'] in ids+[self.anchor_id]]
        if any(self.anchor_id in (r['candidate'], r['opponent']) for r in reports):
            ids.append(self.anchor_id)
        return ids, reports

    def snapshot(self, cid, champion):
        """What a decided verdict records for `calibration`: {opponent (the champion), protocol ({PROTOCOL setting:
        value}), matchup_prior (the effective matchup_prior_elo), reports ({report name: {games, digest
        (`games_digest`)}} of every report the verdict's posterior uses, `inputs`), earlier (the ids of every other
        report present, archives included)}."""
        s, inputs = self.settings, self.inputs(cid, champion)[1]
        used = {r.get('id') for r in inputs}
        return dict(opponent=champion, protocol={k: getattr(s, k) for k in PROTOCOL}, matchup_prior=s.matchup_prior_elo,
                    reports={report_name(r): dict(games=len(r['games']), digest=games_digest(r['games'])) for r in inputs},
                    earlier=sorted(r['id'] for r in load_reports(self.run) if r.get('id') and r['id'] not in used))

    def evidence(self, verdict, cid, champion, games):
        """(a, b) of the evidence pairing for a pending posterior decision, or None: of cid and the champion each
        vs the previous champion (`met` of the champion) and vs Seal, the one whose `games` most reduce the
        posterior variance of delta (value of information, `Posterior.after`), when that beats as many direct
        games; only pairings whose report can grow (`rematch_pair`) and that are `close`. The decision posterior
        also checks the score limit, including for a candidate whose league rating is not published yet."""
        s, post, previous = self.settings, verdict['posterior'], self.met(champion)
        options = []
        for a, b in ((cid, previous), (champion, previous), (cid, self.anchor_id), (champion, self.anchor_id)):
            if not b or b == a or (a, b, 'evidence', s.opening_book) in self.failed_seal or not self.close(a, b):
                continue
            if b == self.anchor_id:
                path = report_path(self.run, a, self.anchor_id)
                if not path.exists() or same_protocol(json.loads(path.read_text()), s):
                    options.append((a, self.anchor_id))
            elif pair := rematch_pair(self.run, a, b, s):
                options.append(pair)
        known = lambda x: x in post.index or x == post.anchor
        after = lambda pair: post.after((cid, champion, True), pair, games)
        options = [o for o in options if known(o[0]) and known(o[1])
                   and informative(expected(post.rating(o[0]), post.rating(o[1])), s.max_expected_score)]
        best = min(options, key=after, default=None)
        return best if best and after(best) < after((cid, champion)) else None

    def lanes(self, verdict, cid, champion):
        """The pool's lanes for a pending posterior decision: direct games of cid vs the champion (lane kind
        verdict['comparison']) fill the pool (never beyond sprt_max_games direct games) until sprt_min_games of them are
        complete and while the direct and pooled estimates disagree; after that an `evidence` pairing that beats
        direct games gets evidence_share of the pool and the direct games the rest."""
        s, direct = self.settings, verdict['direct']['games']
        room = even(min(s.pool_games, s.sprt_max_games-direct))
        share = even(s.pool_games*s.evidence_share)
        other = self.evidence(verdict, cid, champion, share) if share and direct >= s.sprt_min_games and not verdict['disagree'] else None
        lanes = {(cid, champion, verdict['comparison']): min(room, s.pool_games-share) if other else room}
        if other:
            lanes[(*other, 'evidence')] = share
        return {lane: n for lane, n in lanes.items() if n >= 2}

    def decide(self, cid, champion):
        """Posterior mode: a session whose lanes (`lanes`) follow the verdict after every completed colour pair
        until `verdict` decides or sprt_max_games direct games are complete, after which the games in flight
        finish and count (play resumes when they leave the verdict undecided), or until a newer checkpoint of
        cid's variant exists or a settle request names cid, which abandons the games in flight (`session`).
        The final verdict settles it: superseded, cid is
        promoted when p_better >= promote_confidence (settled true), readiness aside. Returns ({opponent: report of
        cid against it}, verdict with decision 'promote', 'reject', 'max-games' or 'superseded'), or ({}, None)
        without a game in its report against the champion. The verdict (`public`) is published as status decision, stored as the direct
        report's metrics.posterior and logged as a 'decision' event."""
        s = self.settings

        def want():
            verdict = self.verdict(cid, champion)
            self.refresh(cid, champion, verdict)
            if verdict['decision'] or verdict['direct']['games'] >= s.sprt_max_games or self.newer(cid) \
                    or self.requested(cid) and verdict['direct']['games']:
                return {}
            lanes = self.lanes(verdict, cid, champion)
            self.publish(decision=dict(public(verdict), candidate=cid, opponent=champion, next=[list(l[:2]) for l in lanes]))
            return lanes
        while True:  # the games in flight can undo a verdict that stopped the session: then play on
            added = self.session(want, s.sprt_max_games,
                                 stop=lambda: self.newer(cid) or self.requested(cid) and self.games(cid, champion),
                                 budget=lambda: {(cid, champion): s.sprt_max_games-len(self.direct(cid, champion))})
            for (a, b, _), games in added.items():
                if a != cid and games:
                    self.record(a, b, json.loads(report_path(self.run, a, b).read_text()))
            verdict = self.verdict(cid, champion)
            if verdict['decision'] or not added or verdict['direct']['games'] >= s.sprt_max_games or self.newer(cid) \
                    or self.requested(cid) and verdict['direct']['games']:
                break
        if not self.games(cid, champion):
            return {}, None
        requested, superseded = self.requested(cid), self.newer(cid)
        verdict = dict(public(verdict), candidate=cid, **self.snapshot(cid, champion))
        if not verdict['decision'] and (superseded or requested) and verdict['p_better'] >= s.promote_confidence:
            verdict.update(decision='promote', settled=True)
        verdict['decision'] = verdict['decision'] or ('superseded' if superseded or requested else 'max-games')
        path = report_path(self.run, cid, champion)
        report = json.loads(path.read_text())
        report['metrics']['posterior'] = verdict
        write_json(path, report)
        self.publish(True, decision=verdict, pending=[p for p in self.status['pending'] if p['candidate'] != cid])
        direct, g = verdict['direct'], lambda v: f'{v:+.0f}' if v is not None else '-'
        log_event(self.run, 'evaluator', 'decision', f'{cid} vs {champion}: {verdict["decision"]}'
                  + (f' (settled on {"request" if requested else "supersession"})' if requested or superseded else '')
                  + f' after {direct["games"]} direct games: '
                  f'P(delta > {s.sprt_elo0:g}) {verdict["p_better"]:.3f}, delta {g(verdict["delta"])} +- {verdict["delta_sd"]:.0f}, '
                  f'direct {g(direct["elo"])}, pooled [{verdict["pooled"][0]:+.0f}, {verdict["pooled"][1]:+.0f}]',
                  **verdict)
        if requested:
            settle_path(self.run, cid).unlink(missing_ok=True)
        found = {r['opponent']: r for r in load_reports(self.run, s) if r['candidate'] == cid}
        return {champion: found[champion], **found}, verdict

    def refresh(self, cid, opponent, verdict):
        """Replace cid's entry of status `pending` by `brief` of its current `verdict`."""
        self.status['pending'] = [brief(cid, opponent, verdict['comparison'], verdict) if p['candidate'] == cid else p
                                  for p in self.status['pending']]

    def queue(self, unrated):
        """Set status `pending` (class contract) for the unrated checkpoint ids `unrated`; a verdict is computed
        only for a pairing with direct games."""
        champion = self.league['champion']
        pairs = [(cid, champion, 'champion') for cid in unrated] + \
            [(v['id'], v['checkpoint'], 'variant') for v in self.variants() if 'verdict' not in v and v['checkpoint']]
        self.status['pending'] = [brief(cid, opponent, kind, self.verdict(cid, opponent) if opponent and self.direct(cid, opponent) else None)
                                  for cid, opponent, kind in pairs]

    def bind(self):
        """Record bound_at for a variant whose direct report exists (its comparison has started), then point every
        variant whose comparison has not started (no bound_at) at the current champion when it follows the champion: base CHAMPION always, and with rebase_on_promotion a variant registered against the
        then champion (on_champion) whose checkpoint is no longer champion. Its id becomes `<champion>@<name>`
        with a 'variant' event. When another entry holds that id, an entry of base CHAMPION is dropped, with its
        request file if still waiting, and an 'error' event (it may compare against the champion only), and a
        rebased one keeps its checkpoint. Returns
        whether any entry changed."""
        champion, changed = self.league['champion'], False
        for entry in list(self.variants()):
            if 'bound_at' not in entry and entry['checkpoint'] and report_path(self.run, entry['id'], entry['checkpoint']).exists():
                entry['bound_at'], changed = time.time(), True
            follows =entry.get('base') == CHAMPION or (self.settings.rebase_on_promotion and entry.get('on_champion'))
            if 'bound_at' in entry or not follows or champion is None or entry['checkpoint'] == champion:
                continue
            if self.entry(f'{champion}@{entry["name"]}'):
                if entry.get('base') == CHAMPION:
                    self.league['variants'].remove(entry)
                    request = requests(self.run).get(entry.get('registered_as', entry['id']))
                    if request:
                        request.unlink()
                    log_event(self.run, 'evaluator', 'error', f'{entry["id"]} dropped: {champion}@{entry["name"]} is already '
                              'registered; register it under another name', candidate=entry['id'])
                    changed = True
                continue
            before = entry['id']
            entry.update(id=f'{champion}@{entry["name"]}', checkpoint=champion)
            log_event(self.run, 'evaluator', 'variant', f'{before} now plays as {entry["id"]}: {champion} is champion',
                      candidate=entry['id'], previous=before, checkpoint=champion)
            changed = True
        return changed

    def trial(self, entry):
        """Decide the pending variant `entry` against its checkpoint (module contract): a session like `decide`'s
        whose lanes (`lanes`, kind 'variant') follow the verdict after every completed colour pair until it
        decides ('better' or 'worse') or sprt_max_games direct games are complete, after which the games in
        flight finish and count (play resumes when they leave the verdict undecided), or until a checkpoint
        awaits rating, which abandons the games in flight (`session`). Stopped
        for a waiting checkpoint the variant stays pending and resumes on a later step; otherwise the final
        verdict (`public`, decision 'better', 'worse' or 'max-games', with candidate and opponent) is stored as
        the entry's verdict and the direct report's metrics.posterior, published as status decision and logged
        as a 'decision' event with P(better), delta, delta_sd and its 95% interval. Every report played is
        recorded in the league (`record`). The first trial of an entry binds it to the champion of that moment
        (`bind`, after any promotion earlier in the step; status `pending` is rebuilt when that moves an entry, and
        an entry `bind` drops ends the trial). Its first persisted game fixes the binding: bound_at (epoch
        seconds) is recorded (`bind` also records it for an entry whose direct report exists) and `bind` leaves it
        alone from then on."""
        if 'bound_at' not in entry and self.bind():
            self.queue(self.status['backlog'])
            write_league(self.run, self.league, self.config, self.settings.fill_top)
            if entry not in self.variants():
                return
        s, cid, base = self.settings, entry['id'], entry['checkpoint']

        def want():
            verdict = self.verdict(cid, base)
            self.refresh(cid, base, verdict)
            if verdict['decision'] or verdict['direct']['games'] >= s.sprt_max_games or self.backlog():
                return {}
            lanes = self.lanes(verdict, cid, base)
            self.publish(decision=dict(public(verdict), candidate=cid, opponent=base, next=[list(l[:2]) for l in lanes]))
            return lanes
        while True:
            added = self.session(want, s.sprt_max_games, stop=self.backlog,
                                 budget=lambda: {(cid, base): s.sprt_max_games-len(self.direct(cid, base))})
            if 'bound_at' not in entry and report_path(self.run, cid, base).exists():
                entry['bound_at'] = time.time()
                write_league(self.run, self.league, self.config, self.settings.fill_top)
            for (a, b, _), games in added.items():
                if games:
                    self.record(a, b, json.loads(report_path(self.run, a, b).read_text()))
            verdict = self.verdict(cid, base)
            if verdict['decision'] or not added or verdict['direct']['games'] >= s.sprt_max_games or self.backlog():
                break
        if not verdict['decision'] and verdict['direct']['games'] < s.sprt_max_games:
            return
        verdict = dict(public(verdict), candidate=cid, opponent=base)
        verdict['decision'] = verdict['decision'] or 'max-games'
        path = report_path(self.run, cid, base)
        report = json.loads(path.read_text())
        report['metrics']['posterior'] = verdict
        write_json(path, report)
        entry['verdict'] = verdict
        self.record(cid, base, report)
        self.publish(True, decision=verdict, pending=[p for p in self.status['pending'] if p['candidate'] != cid])
        low, high = verdict['delta']-1.96*verdict['delta_sd'], verdict['delta']+1.96*verdict['delta_sd']
        log_event(self.run, 'evaluator', 'decision', f'{cid} vs {base}: {verdict["decision"]} after {verdict["direct"]["games"]} '
                  f'direct games: P(delta > {s.sprt_elo0:g}) {verdict["p_better"]:.3f}, delta {verdict["delta"]:+.0f} +- '
                  f'{verdict["delta_sd"]:.0f} [{low:+.0f}, {high:+.0f}]', interval=[low, high], **verdict)

    def review(self):
        """Posterior mode, once per evaluator (so after every restart or settings change): re-apply the promotion
        rule to the existing reports. Of the rated checkpoints, neither skipped nor demoted, with at least
        sprt_min_games direct games against the champion (`direct`), those whose direct reports' decisions
        (metrics.posterior of either orientation under the current protocol), if any, were taken under the current
        likelihood (model dense_posterior.MODEL: a settled decision is never re-judged under another) and whose
        `verdict` is `ready` with P(delta > sprt_elo0) >= promote_confidence are eligible; the one of highest
        posterior rating among them is promoted when it also out-rates the champion and every other checkpoint with
        those direct games ('decision' event 'promote on review', then the 'promotion' event; its Seal anchor is
        scheduled as for any promotion), and its entry keeps that verdict with its `snapshot`. A higher-rated
        checkpoint without those direct games does not block it: it has not met the champion."""
        s, champion = self.settings, self.league['champion']
        if s.decision != 'posterior' or champion is None:
            return
        eligible, met = [], [champion]
        for c in self.league['checkpoints']:
            if c['id'] == champion or c.get('elo') is None or c.get('skipped') or c.get('demoted') \
                    or len(self.direct(c['id'], champion)) < s.sprt_min_games:
                continue
            met.append(c['id'])
            if any(r['metrics']['posterior'].get('model') != MODEL for r in load_reports(self.run, s)
                   if {r['candidate'], r['opponent']} == {c['id'], champion} and 'posterior' in r['metrics']):
                continue
            verdict = self.verdict(c['id'], champion)
            if ready(s, verdict['direct']['games'], verdict['disagree']) \
                    and verdict['p_better'] >= s.promote_confidence:
                eligible.append((verdict['posterior'].rating(c['id']), c['id'], verdict))
        if not eligible:
            return
        _, cid, verdict = max(eligible)
        if max(met, key=verdict['posterior'].rating) != cid:
            return
        verdict = dict(public(verdict), candidate=cid, decision='promote', review=True, **self.snapshot(cid, champion))
        log_event(self.run, 'evaluator', 'decision', f'{cid} vs {champion}: promote on review of the existing reports '
                  f'({verdict["direct"]["games"]} direct games, P(delta > {s.sprt_elo0:g}) {verdict["p_better"]:.3f})',
                  **verdict)
        self.entry(cid)['verdict'] = verdict
        self.promote(cid, champion)
        write_league(self.run, self.league, self.config, self.settings.fill_top)

    def sequential(self, cid, champion):
        """SPRT mode: a session of cid vs the champion until the SPRT decides or sprt_max_games games are
        complete (the games in flight finish and count; a bound crossed before them stays the decision), or
        until a newer checkpoint of cid's variant exists or a settle request names cid, which abandons the games
        in flight (`session`). Returns ({champion:
        report}, metrics.sprt) with decision 'H1', 'H0', 'max-games' or 'superseded', settled as the module
        contract states ('settle' event: promoted when the posterior p_better of `verdict` is at least
        promote_confidence), or ({}, None) without a game. The status decision is the posterior `verdict` of
        the direct games with the SPRT's decision (None while pending), published after every completed colour
        pair and at the end."""
        s, decided = self.settings, []

        def shown(decision=None):
            """The status decision: the posterior `verdict` numbers with the SPRT's decision (None while pending)."""
            verdict = self.verdict(cid, champion)
            self.refresh(cid, champion, verdict)
            return dict(public(verdict), decision=decision, candidate=cid, opponent=champion)

        def want():
            games = self.games(cid, champion)
            if games and (decision := self.test(games)['decision']):
                decided.append(decision)
            if decided or len(games) >= s.sprt_max_games or self.newer(cid) or self.requested(cid) and games:
                return {}
            self.publish(decision=dict(shown(), next=[[cid, champion]]))
            return {(cid, champion, 'champion'): even(min(s.pool_games, s.sprt_max_games-len(games)))}
        self.session(want, s.sprt_max_games,
                     stop=lambda: self.newer(cid) or self.requested(cid) and self.games(cid, champion),
                     budget=lambda: {(cid, champion): s.sprt_max_games-len(self.games(cid, champion))})
        path = report_path(self.run, cid, champion)
        if not self.games(cid, champion):  # no game under the active protocol
            return {}, None
        report = json.loads(path.read_text())
        result, n = self.test(report['games']), len(report['games'])
        requested = self.requested(cid)
        decision = decided[0] if decided else 'superseded' if n < s.sprt_max_games and (self.newer(cid) or requested) else 'max-games'
        test = report['metrics']['sprt'] = dict(result, decision=decision)
        final = shown(decision)
        self.publish(True, decision=final, pending=[p for p in self.status['pending'] if p['candidate'] != cid])
        summary_ = report['summary']
        if test['decision'] == 'superseded':
            p_better = final['p_better']
            settled = test['settled'] = dict(pair_score=summary_['pair_score'], p_better=p_better,
                                             promote=p_better >= s.promote_confidence)
            log_event(self.run, 'evaluator', 'settle', f'{cid} vs {champion} settled on {"request" if requested else "supersession"} after {n} games: '
                      f'+{summary_["wins"]} -{summary_["losses"]} ={summary_["capped"]}, pair score {summary_["pair_score"]:.3f}, '
                      f'P(delta > {s.sprt_elo0:g}) {p_better:.3f}, LLR {test["llr"]:.2f}: '
                      + ('promoted' if settled['promote'] else 'not promoted'),
                      candidate=cid, opponent=champion, games=n, llr=test['llr'], rule=final['rule'],
                      threshold=final['threshold'], **settled)
        write_json(path, report)
        if requested:
            settle_path(self.run, cid).unlink(missing_ok=True)
        return {champion: report}, test

    def rate(self, entry):
        """Rate a chosen checkpoint against the champion (none for the first) by the configured `decision`
        (`sequential` for 'sprt': promote on H1 or a promoting settlement; `decide` for 'posterior': promote on
        'promote'); add it to the league (superseded true when a newer checkpoint cut its evaluation short) and
        promote."""
        cid, path, _ = entry
        variant, step = cid.split('/')
        champion, s = self.league['champion'], self.settings
        self.deciding = cid
        try:
            reports, test = ({}, {}) if champion is None else self.decide(cid, champion) if s.decision == 'posterior' \
                else self.sequential(cid, champion)
        finally:
            self.deciding = None
        report = reports.get(champion)
        if champion is not None and report is None:
            self.league['checkpoints'].append(dict(id=cid, variant=variant, step=int(step), skipped=True,
                                                   elo=None, elo_interval=None, matches=[]))
            log_event(self.run, 'evaluator', 'skip', f'skipped {cid}: a newer checkpoint appeared before its first champion game',
                      checkpoints=[cid], candidate=cid)
            write_league(self.run, self.league, self.config, self.settings.fill_top)
            return
        decision = test.get('decision')
        entry = dict(id=cid, variant=variant, step=int(step), ema_sha256=digest(path/'ema.pt'),
                     elo=None, elo_interval=None, **({'superseded': True} if decision == 'superseded' else {}),
                     **({'panel': dict(incumbent=champion)} if champion is not None and s.extra_opponents > 0 else {}),
                     **({'verdict': test} if 'p_better' in (test or {}) else {}),
                     matches=[match_entry(o, r) for o, r in reports.items()])
        self.league['checkpoints'].append(entry)
        if champion is None:
            self.promote(cid, None)
        elif decision in ('H1', 'promote') or test.get('settled', {}).get('promote'):
            self.promote(cid, champion)
        promoted = self.league['champion'] == cid
        write_league(self.run, self.league, self.config, self.settings.fill_top)
        print(f'{cid}: ' + ', '.join(f'vs {o} +{r["summary"]["wins"]} -{r["summary"]["losses"]} ={r["summary"]["capped"]}'
                                      for o, r in reports.items()) + (' -> champion' if promoted else f' ({decision})' if decision else ''), flush=True)

    def sealed(self, cid):
        """The champion-vs-Seal report of cid, or None."""
        path = report_path(self.run, cid, self.anchor_id)
        return json.loads(path.read_text()) if path.is_file() else None

    def seal_reports(self, cid):
        """cid's reports against Seal under every protocol: its report.json and those archived beside it."""
        return [r for r in load_reports(self.run) if r['candidate'] == cid and r['opponent'] == self.anchor_id]

    def crown(self, cid):
        """Make cid champion and start its reign: reign_from, reign_games and reign_pooled (league contract)."""
        self.league.update(champion=cid, reign_from=len(self.league['checkpoints']), reign_pooled=True,
                           reign_anchor=self.anchor_id,
                           reign_games=sum(len(r['games']) for r in self.seal_reports(cid)))

    def pool_reign(self):
        """Once, for a reign begun before archived reports were pooled (no reign_pooled): add the champion's archived
        Seal games to reign_games, so they count as played before the reign, capped at all its Seal games (the
        report reign_games was counted on may itself be archived by now, its prefix then counted once). Games played
        during the reign in an archive are then owed again; no game is ever taken as played twice and the anchor
        never goes negative. The two fields are written to league.json at once, so the migration happens once."""
        champion = self.league.get('champion')
        if champion is None or self.league.get('reign_pooled'):
            return
        current = self.sealed(champion)
        total = sum(len(r['games']) for r in self.seal_reports(champion))
        archived = total-(len(current['games']) if current else 0)
        self.league.update(reign_games=min(self.league.get('reign_games', 0)+archived, total), reign_pooled=True)
        write_json(self.run/'league.json', self.league)

    def anchor(self):
        """(champion entry, self.anchor_id, 'anchor', games left) while the current champion owes Seal games, else None. In
        its current reign it owes anchor_games once (anchor_on_promotion) and anchor_games more per `anchor_every`
        checkpoints rated during the reign (entries from `reign_from` on), counted against the games its Seal reports
        (`seal_reports`, every protocol) gained since reign_games, so an anchor owed when the protocol changes (a book
        refresh) is played under the new one. A newer champion supersedes the old one's unfinished anchor.
        Quotas apply only while the champion and Seal are `close`, as for other automatic opponents."""
        s, champion = self.settings, self.entry(self.league['champion'])
        if not s.anchor_games or champion is None or not self.close(champion['id'], self.anchor_id) \
                or (champion['id'], self.anchor_id, 'anchor', s.opening_book) in self.failed_seal:
            return None
        entries = self.league['checkpoints']
        later = sum(not c.get('skipped') for c in entries[self.league.get('reign_from', entries.index(champion)+1):])
        played = sum(len(r['games']) for r in self.seal_reports(champion['id']))-self.league.get('reign_games', 0)
        left = s.anchor_games*(s.anchor_on_promotion+later//s.anchor_every)-played
        return (champion, self.anchor_id, 'anchor', left) if left > 0 else None

    def optional(self, exclude=(), models=None):
        """(league entry of the candidate side, opponent, kind, games) of the first optional comparison, else None:
        the current champion's panel ('panel', 'incumbent'), idle rematches (`rematches`, with idle_rematch),
        the panels of the variant heads, newest first, then the newest rated checkpoint missing its
        previous-checkpoint comparison (when previous_games > 0 and the two are `close`)."""
        s, excluded = self.settings, set(exclude)
        champion = [c for c in self.league['checkpoints'] if c['id'] == self.league['champion']]
        for a, b, kind, games in [n for c in champion for n in self.needs(c)] + (self.rematches() if s.idle_rematch else []) \
                + [n for c in self.heads() for n in self.needs(c)]:
            if (a, b) not in excluded and (models is None or len(models | {name for name in (a, b) if name != self.anchor_id}) <= 3):
                return self.entry(a), b, kind, games
        rated = [c for c in self.league['checkpoints'] if not c.get('skipped')]
        for index in reversed(range(len(rated))):
            c = rated[index]
            earlier = [p for p in rated[:index] if p['variant'] == c['variant'] and p['step'] < c['step']]
            if earlier and s.previous_games and (previous := max(earlier, key=lambda p: p['step'])['id']) \
                    not in {m['opponent'] for m in c['matches']} and self.close(c['id'], previous) \
                    and (c['id'], previous) not in excluded \
                    and (models is None or len(models | {c['id'], previous}) <= 3):
                return c, previous, 'previous', s.previous_games
        return None

    def met(self, cid):
        """The champion that checkpoint cid met when it was rated (its panel incumbent, else its first match's
        opponent), or None."""
        entry = self.entry(cid) or {}
        return entry.get('panel', {}).get('incumbent') or next((m['opponent'] for m in entry.get('matches', [])), None)

    def fill(self, exclude=(), models=None):
        """Next fill comparison, maximizing the expected reduction in summed 95% interval half-widths.

        Targets are the top `fill_top` checkpoints' ratings on the published league scale, plus champion minus
        Seal while its half-width exceeds anchor_target_halfwidth. Fit the same pooled, zero-matchup posterior
        as `rate`, including archived protocols. Any rated league pair may reduce those targets through shared
        covariance, even when neither player is a target. Only `close`, growable comparisons may play; Seal is
        allowed with anchor_target_halfwidth > 0, excluding failed pairs. Rechoose after each `games` round.
        `fill_uncertainty` records the selected pair's current and predicted half-widths, not a guaranteed result.
        """
        s, champion = self.settings, self.entry(self.league['champion'])
        excluded = set(exclude)
        if not excluded:
            self.status['fill_uncertainty'] = None
        if not s.idle_fill or champion is None:
            return None
        checkpoints = [c for c in self.league['checkpoints'] if c.get('elo') is not None and not c.get('skipped')]
        ids = [c['id'] for c in checkpoints]
        ids += [v['id'] for v in self.variants() if v['checkpoint'] in ids and v.get('elo') is not None]
        if not ids:
            return None
        reports = [r for r in load_reports(self.run) if r['candidate'] in ids and r['opponent'] in ids+[self.anchor_id]]
        names = ids+([self.anchor_id] if s.anchor_target_halfwidth > 0 or any(r['opponent'] == self.anchor_id for r in reports) else [])
        if len(names) < 2:
            return None
        post = Posterior(names, ids[0], observations(reports), 0., parents(names))
        best = sorted((c for c in checkpoints if not c.get('demoted')), key=lambda c: -post.rating(c['id']))[:s.fill_top]
        targets = [(c['id'], post.anchor, False) for c in best if c['id'] != post.anchor]
        if s.anchor_target_halfwidth > 0 and 1.96*post.difference(champion['id'], self.anchor_id, False)[1] > s.anchor_target_halfwidth:
            targets.append((champion['id'], self.anchor_id, False))
        if not targets:
            return None
        before = [1.96*post.difference(a, b, False)[1] for a, b, _ in targets]
        chosen, gain, predicted = None, 0., None
        for i, a in enumerate(names):
            for b in names[i+1:]:
                if not self.close(a, b) or models is not None \
                        and len(models | {name for name in (a, b) if name != self.anchor_id}) > 3:
                    continue
                if b == self.anchor_id:
                    if s.anchor_target_halfwidth <= 0 or (a, b, 'fill', s.opening_book) in self.failed_seal:
                        continue
                    pair = a, b
                else:
                    pair = rematch_pair(self.run, a, b, s)
                if pair is None or pair in excluded:
                    continue
                after = [1.96*math.sqrt(max(0., post.after(t, pair, s.games))) for t in targets]
                reduction = sum(x-y for x, y in zip(before, after))
                if reduction > gain:
                    chosen, gain, predicted = pair, reduction, after
        if chosen is None:
            return None
        if not excluded:
            self.status['fill_uncertainty'] = dict(candidate=chosen[0], opponent=chosen[1], games=s.games,
                                                   expected_reduction=gain, targets=[dict(a=a, b=b, halfwidth=x,
                                                   expected_halfwidth=y) for (a, b, _), x, y in zip(targets, before, predicted)])
        return self.entry(chosen[0]), chosen[1], 'fill', s.games

    def pipeline_ready(self):
        """New idle work must yield to a checkpoint, variant trial, or due opening-book refresh."""
        return not self.backlog() and not any('verdict' not in v and v['checkpoint'] for v in self.variants()) \
            and not any((self.run/'variant-requests').glob('*.json')) \
            and not self.openings.due(self.league['champion'], time.time())

    def pipeline_task(self, blocked, names):
        """Next independent idle comparison fitting at most three live neural models."""
        excluded = set(blocked)
        while task := self.optional(excluded, names):
            entry, opponent, kind, _ = task
            if kind != 'sprt':
                return task
            excluded.update(((entry['id'], opponent), (opponent, entry['id'])))
        return self.fill(excluded, names)

    def filling(self, target):
        """Log a 'fill' event when fill work starts, changes target ('seal', '<a> vs <b>' or 'generalization <a> vs
        <b>') or ends (None)."""
        if target is None:
            self.status['fill_uncertainty'] = None
        if target != self.fill_target:
            log_event(self.run, 'evaluator', 'fill', f'fill: {target}' if target else f'fill ended: {self.fill_target}',
                      target=target)
            self.fill_target = target

    def refresh_openings(self):
        """Refresh a live book with the champion's model when it is due (dense_openings.Book.due: the champion
        changed or book_refresh_hours passed), stamp settings.opening_book with its new digest, log a 'book' event
        and republish the league. A new digest drops the open pairings' cached games, so each reopens (`open`) under
        the new state."""
        champion, now = self.league['champion'], time.time()
        if champion is None or not self.openings.due(champion, now):
            return
        self.use(champion)
        rng = np.random.default_rng(pair_seed(self.config.seed, f'book/{champion}', int(now)))
        result = self.openings.refresh(self.models[champion], champion, rng, now, self.config.actor.leaf_batch)
        if result['digest'] != self.settings.opening_book:
            for cache in (self.book, self.next, self.ids):
                cache.clear()
        self.settings = replace(self.settings, opening_book=result['digest'])
        self.status['settings'] = asdict(self.settings)
        log_event(self.run, 'evaluator', 'book', f'opening book refreshed by {champion}: {result["added"]} added, retired '
                  + ', '.join(f'{n} {r}' for r, n in result['retired'].items()) + f'; {result["openings"]} openings, '
                  f'{result["challengers"]} of them challengers', checkpoint=champion, **result)
        write_league(self.run, self.league, self.config, self.settings.fill_top)

    def step(self):
        """One unit of work; False when there is none. First judges every panel already complete on disk
        (`settle`, e.g. after a restart), so no candidate meets a regressed champion. A checkpoint with games
        against the champion is never left skipped: a skipped league entry with such games is removed again
        ('info' event), and an unrated checkpoint with such games is rated first (an evaluation cut short by a
        restart resumes there, or settles on its games when superseded). Otherwise a due refresh of the live opening
        book runs (`refresh_openings`; never while such a candidate or a variant trial with games waits, as it would
        restart its games). Then, on
        the first step, the promotion
        rule is re-applied to the existing reports (`review`). Then it rates the newest unrated checkpoint of the
        variant whose newest unrated checkpoint is oldest, skipping that variant's older unrated checkpoints (none
        of them has games against the champion), or else decides the first pending variant (`trial`, in
        registration order; waiting registrations are adopted into league.json first). While the champion owes
        Seal games, it plays at most anchor_session_games before a pending trial, then gives the trial a turn.
        Without a trial it plays the anchor, then optional and fill work, each until its games are complete
        or a checkpoint waits (the games in flight then finish and count)."""
        self.settle()
        if self.status['anchor_champion'] != self.league['champion']:
            self.set_anchor_turn(True)
        champion = self.league['champion']
        revived = [c for c in self.league['checkpoints'] if c.get('skipped') and champion and self.games(c['id'], champion)]
        for c in revived:
            self.league['checkpoints'].remove(c)
            log_event(self.run, 'evaluator', 'info', f'{c["id"]} was skipped with {len(self.games(c["id"], champion))} games '
                      f'against {champion}; it is rated on them', candidate=c['id'])
        if revived:
            write_league(self.run, self.league, self.config, self.settings.fill_top)
        if adopt(self.league, self.run) + self.bind():
            write_league(self.run, self.league, self.config, self.settings.fill_top)
        known = {c['id'] for c in self.league['checkpoints']}
        unrated = [e for e in checkpoints(self.run) if e[0] not in known]
        self.status['backlog'] = [e[0] for e in unrated]
        self.queue(self.status['backlog'])
        resumed = [e for e in unrated if champion and self.games(e[0], champion)]
        self.dismiss_settle_requests({e[0] for e in resumed})
        # A refresh that changes the book would restart the games of a candidate or variant trial waiting to resume.
        trials = [v for v in self.variants() if 'verdict' not in v and v['checkpoint'] and self.games(v['id'], v['checkpoint'])]
        if not resumed and not trials:
            self.refresh_openings()
        anchor_task = self.anchor() if self.anchor_turn else None
        if resumed and not anchor_task:
            self.filling(None)
            self.rate(resumed[0])
            self.set_anchor_turn(True)
            return True
        if not self.reviewed:
            self.reviewed = True
            self.review()
            if self.status['anchor_champion'] != self.league['champion']:
                self.set_anchor_turn(True)
            if not trials:
                self.refresh_openings()  # a champion crowned on review refreshes the book before any game
        anchor_task = self.anchor() if self.anchor_turn else None
        if unrated and not anchor_task:
            self.filling(None)
            variant = lambda e: e[0].split('/')[0]
            heads = {variant(e): e for e in unrated}
            head = next(e for e in unrated if heads[variant(e)] is e)
            skipped = [e[0] for e in unrated if variant(e) == variant(head) and e is not head]
            if skipped:
                self.league['checkpoints'] += [dict(id=cid, variant=variant(head), step=int(cid.split('/')[1]), skipped=True,
                                                    elo=None, elo_interval=None, matches=[]) for cid in skipped]
                write_league(self.run, self.league, self.config, self.settings.fill_top)
                log_event(self.run, 'evaluator', 'skip', f'skipped {", ".join(skipped)} for {head[0]}',
                          checkpoints=skipped, candidate=head[0])
            self.rate(head)
            self.set_anchor_turn(True)
            return True
        if not anchor_task:
            for entry in self.variants():
                if 'verdict' not in entry and entry['checkpoint']:
                    self.filling(None)
                    self.trial(entry)
                    self.set_anchor_turn(True)
                    return True
        task = anchor_task or self.anchor() or self.optional() or self.fill()
        self.filling(None if task is None or task[2] not in ('fill', 'generalization') else self.anchor_id if task[1] == self.anchor_id
                     else f'{"generalization " if task[2] == "generalization" else ""}{task[0]["id"]} vs {task[1]}')
        if task is None:
            return False
        entry, opponent, kind, games = task
        a, s = entry['id'], self.settings
        initial_games = len(self.games(a, opponent))
        if kind == 'anchor':
            games = min(games, s.anchor_session_games)
            self.set_anchor_turn(False)  # commit the handoff before any anchor pair is written
        target, decided = games if kind == 'previous' else initial_games+games, []
        waiting = {e[0] for e in unrated}

        def arrived():
            return (any(e[0] not in waiting and self.entry(e[0]) is None for e in checkpoints(self.run))
                    or any((self.run/'variant-requests').glob('*.json')))

        def want():
            done = self.games(a, opponent)
            if kind == 'sprt' and done and (decision := self.test(done)['decision']):
                decided.append(decision)  # an idle SPRT rematch keeps the first bound it crosses
            new_trial = arrived() if kind == 'anchor' else self.backlog()
            if new_trial or len(done) >= target or decided or (a, opponent, kind, s.opening_book) in self.failed_seal:
                return {}
            return {(a, opponent, kind): even(min(s.pool_games, target-len(done)))}
        added = self.session(want, target, auxiliary=self.pipeline_task if s.pipeline and kind != 'sprt' else None,
                             budget=(lambda: {(a, opponent): target-len(self.games(a, opponent))}) if kind == 'sprt' else None)
        path = report_path(self.run, a, opponent)
        if path.exists():
            report = json.loads(path.read_text())
            if decided:
                report['metrics']['sprt']['decision'] = decided[0]
                write_json(path, report)
            self.record(a, opponent, report)
            if kind == 'sprt' and report['metrics']['sprt']['decision'] == 'H1' and self.league['champion'] == opponent \
                    and not entry.get('demoted'):
                self.promote(entry['id'], opponent)
                write_league(self.run, self.league, self.config, self.settings.fill_top)
        for lane in added:
            if lane[:2] == (a, opponent):
                continue
            other = report_path(self.run, *lane[:2])
            if other.exists():
                self.record(*lane[:2], json.loads(other.read_text()))
        if path.exists() or added:
            self.settle()
        return True


def loop(args):
    run = Path(args.run)
    config = kernel_config(dense_config.load(run), getattr(args, 'net_kernels', None))
    settings = dense_config.override(config.evaluation, args, 'eval_')
    if any(getattr(settings, name) % 2 for name in ('games', 'previous_games', 'anchor_games', 'anchor_session_games', 'sprt_max_games', 'pool_games', 'sprt_min_games')):
        raise ValueError('Evaluation game counts must be even: every opening is played with both colours')
    if min(settings.games, settings.sprt_max_games, settings.pool_games) < 2 or any(0 < getattr(settings, name) < 2 or getattr(settings, name) < 0
                                                                for name in ('previous_games', 'anchor_games')):
        raise ValueError('games, sprt_max_games and pool_games need at least one opening pair; optional totals are 0 or at least 2')
    if not 0 <= settings.evidence_share < 1:
        raise ValueError('evidence_share must be in [0, 1)')
    if settings.decision not in ('posterior', 'sprt') or not .5 < settings.promote_confidence < 1 \
            or settings.matchup_prior_elo < 0 or settings.sprt_min_games < 2:
        raise ValueError("decision is 'posterior' or 'sprt', promote_confidence in (0.5, 1), matchup_prior_elo >= 0 "
                         'and sprt_min_games at least one opening pair')
    if settings.anchor_games and settings.anchor_every < 1:
        raise ValueError('anchor_every must be at least 1 while anchor games are enabled')
    if settings.anchor_session_games < 2:
        raise ValueError('anchor_session_games must be at least one opening pair')
    if settings.anchor_target_halfwidth < 0 or settings.fill_top < 0 or not .5 <= settings.max_expected_score <= 1:
        raise ValueError('anchor_target_halfwidth and fill_top must be at least 0, max_expected_score in [0.5, 1]')
    dense_openings.check(settings)
    Budgets.of(settings)
    Schedule.of(settings)
    if args.once:
        settings = replace(settings, idle_fill=False)  # fill work never runs out
    evaluator = Evaluator(run, config, settings, Pacer(1. if args.once else settings.eval_share))
    legacy = 'decision' not in json.loads((run/'config.json').read_text())['evaluation']
    log_event(run, 'evaluator', 'info', f'promotion rule: {settings.decision}'
              + (' (default; config.json predates the setting)' if legacy and args.eval_decision is None else ''),
              decision=settings.decision)
    try:
        while True:
            if evaluator.step():
                continue
            evaluator.publish(True, stage='idle', comparison=None, pool=[], tally=None)
            if args.once:
                break
            time.sleep(args.poll)
    except Exception as error:
        message = f'{type(error).__name__}: {error}'
        evaluator.publish(True, stage='failed', error=message)
        log_event(run, 'evaluator', 'error', message, comparison=evaluator.status['comparison'])
        raise
    finally:
        if evaluator.seal is not None and hasattr(evaluator.seal, 'close'):
            evaluator.seal.close()


def calibrate(args):
    """Continue the newest capped games from their cap with the champion (both sides, `--sims`) for up to
    --extra plies, then score each original ply's TD(lambda) target (dense_data.value_targets), the raw root
    value and the masked baseline (p = 1/2) against the realised result, overall and per continuation length."""
    run = Path(args.run)
    config = kernel_config(dense_config.load(run), getattr(args, 'net_kernels', None))
    model = load(run, config, source=resolve(run))
    chosen = []
    for path in reversed(dense_data.shard_dirs(run)):
        episodes, _ = dense_data.read_shard(path, policies=False)
        chosen += [e for e in episodes if e['winner'] < 0 and e['root_values'] is not None
                   and all(v is not None for v in e['root_values'])][:args.games-len(chosen)]
        if len(chosen) >= args.games:
            break
    settings = config.evaluation
    samples = min(settings.root_samples, args.sims)
    games = [MatchGame([model, model], e['moves'], k, args.sims, samples, settings.tactics, len(e['moves'])+args.extra,
                       dict(index=k), graphs=settings.search_graph, choices=settings.search_choice,
                       floors=settings.q_range_floor) for k, e in enumerate(chosen)]
    started = time.perf_counter()
    records = play(games, config.actor.leaf_batch)
    methods = ['masked', 'root']+[f'td{lam:g}' for lam in args.lambdas]
    buckets = [(a, min(b, args.extra)) for a, b in ((1, 32), (33, 96), (97, args.extra)) if a <= args.extra]
    scores = {}  # (method, bucket) -> [log loss sum, brier sum, n]
    for e, record in zip(chosen, records):
        winner = record['winner']
        if winner < 0:
            continue
        extra = record['plies']-len(e['moves'])
        bucket = next(f'{a}-{b}' for a, b in buckets if a <= extra <= b)
        players = [dense_data.player_at(t) for t in range(len(e['moves']))]
        outcome = np.array([float(p == winner) for p in players])
        predictions = dict(masked=np.full(len(players), .5), root=(1+np.array(e['root_values']))/2)
        for lam in args.lambdas:
            predictions[f'td{lam:g}'] = np.array(dense_data.value_targets(players, e['root_values'], -1, lam)[0])
        for method, p in predictions.items():
            p = np.clip(p, 1e-6, 1-1e-6)
            loss = -(outcome*np.log(p)+(1-outcome)*np.log(1-p)); brier = (p-outcome)**2
            for key in ((method, 'all'), (method, bucket)):
                s = scores.setdefault(key, [0., 0., 0])
                s[0] += loss.sum(); s[1] += brier.sum(); s[2] += len(p)
    labels = ['all']+[f'{a}-{b}' for a, b in buckets]
    table = {m: {b: dict(log_loss=scores[m, b][0]/scores[m, b][2], brier=scores[m, b][1]/scores[m, b][2],
                         plies=scores[m, b][2]) for b in labels if (m, b) in scores} for m in methods}
    resolved = sum(r['winner'] >= 0 for r in records)
    counts = {b: sum(1 for e, r in zip(chosen, records) if r['winner'] >= 0 and a <= r['plies']-len(e['moves']) <= c)
              for b, (a, c) in zip(labels[1:], buckets)}
    out = dict(created_at=time.time(), checkpoint=model.checkpoint, actor_sha256=model.sha, sims=args.sims, extra=args.extra,
               lambdas=args.lambdas, games=len(chosen), resolved=resolved, games_per_bucket=counts,
               seconds=time.perf_counter()-started, table=table,
               note='buckets are continuation lengths (plies after the cap until the result); unresolved games are excluded',
               games_detail=[dict(moves_at_cap=len(e['moves']), winner=r['winner'], plies=r['plies']) for e, r in zip(chosen, records)])
    (run/'evaluations').mkdir(exist_ok=True)
    target = run/'evaluations'/f'calibration-{time.strftime("%Y%m%d-%H%M%S")}.json'
    write_json(target, out)
    print(f'{len(chosen)} capped games, {resolved} resolved within {args.extra} plies; per bucket {counts}')
    print(f'{"method":>8} ' + ' '.join(f'{b:>17}' for b in labels))
    for m in methods:
        print(f'{m:>8} ' + ' '.join(f'{table[m][b]["log_loss"]:8.4f}/{table[m][b]["brier"]:.4f}' if b in table[m] else f'{"-":>17}'
                                    for b in labels))
    print(f'(log loss / Brier per ply) -> {target}')


def match(args):
    """Paired match of --a against --b; each side has its own model instance, trees and solver budgets."""
    run = Path(args.run)
    config = kernel_config(dense_config.load(run), getattr(args, 'net_kernels', None))
    settings = replace(config.evaluation,
                       external_engine=getattr(args, 'eval_external_engine', None) or config.evaluation.external_engine,
                       external_name=getattr(args, 'eval_external_name', None) or config.evaluation.external_name)
    anchor = anchor_name(settings)
    if args.sims:
        settings = replace(settings, sims=args.sims, root_samples=min(settings.root_samples, args.sims))
    sides = {side: replace(settings, **{'solver_'+f: getattr(args, f'{side}_solver_{f}') for f in asdict(Budgets())
                                        if getattr(args, f'{side}_solver_{f}') is not None}) for side in 'ab'}
    budgets = {side: Budgets.of(s) for side, s in sides.items()}
    for side in sides.values():
        Schedule.of(side)
    if args.a == anchor:
        raise ValueError('Seal plays as --b; pass the checkpoint as --a')
    if args.a == args.b and budgets['a'] == budgets['b']:
        raise ValueError('A match needs two distinct players: other checkpoints or other solver budgets')
    models = {}
    for side, name in (('a', args.a), ('b', args.b)):
        if name != anchor:
            path = Path(name) if Path(name).is_file() else run/'checkpoints'/name/'ema.pt'
            models[side] = load(run, config, source=(name, path))
    book = dense_openings.Book(run, settings)
    settings = replace(settings, opening_book=book.digest())
    started = time.perf_counter()
    target = run/'matches'/f'{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:8]}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    report = dict(id=uuid.uuid4().hex, candidate=args.a, opponent=args.b, created_at=time.time(),
                  candidate_sha256=models['a'].sha,
                  opponent_sha256=anchor if args.b == anchor else models['b'].sha,
                  settings=asdict(settings), solver={side: asdict(b) for side, b in budgets.items()}, games=[])

    def record(finished):
        finished = [g for g in finished if 'error' not in g]
        if len(finished) == len(report['games']):
            return
        report.update(games=finished, summary=tally(finished), metrics=paired_metrics(finished, args.games))
        write_json(target, report)

    external = anchor_engine(settings) if args.b == anchor else None
    try:
        records = play(paired_games(models['a'], anchor if args.b == anchor else models['b'], args.games,
                                    f'match/{args.a}/{args.b}', config, settings, external, book,
                                    sides=(sides['a'], sides['b']), candidate=args.a, opponent=args.b),
                       config.actor.leaf_batch, heartbeat=record, schedule=Schedule.of(settings))
    finally:
        if external is not None and hasattr(external, 'close'):
            external.close()
    record(records)
    print(json.dumps(dict(summary=report['summary'], metrics=report['metrics'], seconds=time.perf_counter()-started,
                          solver=report['solver'], report=str(target)), indent=2))


def variant(args):
    """Register --checkpoint@--name with the --set overrides (`register`) and print its entry."""
    print(json.dumps(register(args.run, args.checkpoint, args.name, parse_settings(args.set)), indent=2))


def settle(args):
    print(json.dumps(request_settle(args.run, args.checkpoint), indent=2))


def kernel_config(config, mode):
    """Override evaluator model kernels in this process, leaving the run config untouched."""
    return (config if mode is None else replace(
        config, actor=replace(config.actor, net_kernels=mode, cuda_graphs=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('loop'); p.add_argument('--run', required=True); p.add_argument('--once', action='store_true')
    p.add_argument('--poll', type=float, default=30.)
    p.add_argument('--net-kernels', choices=('reference', 'fused'), help='model kernels for this evaluator process')
    dense_config.add_arguments(p.add_argument_group('evaluation overrides for this process'), dense_config.EvaluationSettings, 'eval_')
    p = sub.add_parser('calibrate'); p.add_argument('--run', required=True)
    p.add_argument('--net-kernels', choices=('reference', 'fused'), help='model kernels for this process')
    p.add_argument('--lambdas', type=float, nargs='+', default=[.5, .7, .9, 1.])
    p.add_argument('--games', type=int, default=100); p.add_argument('--sims', type=int, default=64)
    p.add_argument('--extra', type=int, default=256)
    p = sub.add_parser('match'); p.add_argument('--run', required=True); p.add_argument('--a', required=True)
    p.add_argument('--net-kernels', choices=('reference', 'fused'), help='model kernels for this process')
    p.add_argument('--b', required=True); p.add_argument('--games', type=int, default=32); p.add_argument('--sims', type=int)
    p.add_argument('--eval-external-engine'); p.add_argument('--eval-external-name')
    for side in 'ab':
        for name, default in asdict(Budgets()).items():
            kind = dict(action=argparse.BooleanOptionalAction) if isinstance(default, bool) else dict(type=int)
            p.add_argument(f'--{side}-solver-{name.replace("_", "-")}', dest=f'{side}_solver_{name}', **kind,
                           help=f'solver_{name} of --{side} (default: the evaluation setting)')
    p = sub.add_parser('variant'); p.add_argument('--run', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help=f'one of {", ".join(SIDE)}; repeatable')
    p = sub.add_parser('settle'); p.add_argument('--run', required=True); p.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    dict(loop=loop, calibrate=calibrate, match=match, variant=variant, settle=settle)[args.command](args)


if __name__ == '__main__':
    main()
