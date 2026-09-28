"""Dense checkpoint evaluator: paired league games, Bradley-Terry ratings, champion promotion, calibration.

Run layout: dense_config. Subcommands
  loop       rate the newest unrated checkpoint of a variant (dense_selfplay.checkpoints order) against the
             champion; that variant's older unrated checkpoints enter league.json with skipped true and elo null
             and are never played (KataGo's gatekeeper: only the newest candidate meets the champion). Only when
             no checkpoint waits does it play the optional comparisons, newest rated checkpoint first, one round
             of `games` at a time: `previous_games` vs the previous rated checkpoint of the variant and, for every
             `anchor_every`-th rated checkpoint, `anchor_games` vs Seal. A champion SPRT that sees a newer
             checkpoint of its variant after a round stops there (decision 'superseded', entry superseded
             true, no promotion). All play is paced by `eval_share` (Pacer); --processes splits each round
             over worker subprocesses (each about 0.7 GB of VRAM: on an 8 GB card two workers leave room for
             three actor processes, not four); --eval-* flags override evaluation settings for this process
             (reports record the effective settings). Writes evaluations/<a>-vs-<b>/report.json (reused when present),
             league.json, champion.json on promotion, evaluator-status.json (Evaluator.publish) and events.
  calibrate  continue capped self-play games with the champion and score TD(lambda) value targets against
             the realised results.
  pairs      internal: one worker process of loop --processes.
  match      ad hoc paired match between two checkpoints (run ids or paths) or a checkpoint and Seal.

Scoring: a capped game (at the ply limit, or reason 'span' when a searched position does not fit the largest crop) is
half a point for each side. Pair score = candidate points / 2 over its two games.
The comparison with the champion is a sequential test (see `sprt`) played in rounds of `games` games until it
accepts H0 or H1 or reaches `sprt_max_games` (decision 'max-games'); only H1 promotes. The first rated
checkpoint becomes champion unopposed.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

import dense_config
from dense_config import log_event
import dense_data
from arena import Seal
from dense_selfplay import Engine, checkpoints, load
from hexo import Game
from klent import digest
from train import paired_metrics, task_opening, write_json

SEAL = 'seal'
PACE_WINDOW = 3600.  # a Pacer banks at most share * PACE_WINDOW seconds of idle credit
STATUS_SECONDS = 2.
RATING_NOTE = ('Bradley-Terry over paired comparisons (caps count half a point to each side); each opening pair is one '
               'observation with a Jeffreys Dirichlet prior over the five pair scores 0..2 in half points; 95% '
               'credible intervals from posterior draws; the first evaluated checkpoint is fixed at 0. Seal is one '
               'more node rated jointly from its anchor games, so its Elo is estimated, not assumed.')


class MatchGame:
    """One evaluation game. sides[colour] is a dense_selfplay.Model or SEAL; model sides search `sims`
    placements with argmax play (one tree per distinct model, advanced on every placement); a Seal side
    plays complete turns inline, validated with Game.legal. Ends at a win, `max_plies` placements or when the
    Engine stops it (`reason` 'span')."""

    def __init__(self, sides, opening, seed, sims, samples, tactics, max_plies, record, seal=None, seal_ms=0):
        self.sides, self.max_plies, self.seal, self.seal_ms, self.record = sides, max_plies, seal, seal_ms, record
        self.reason = None
        self.budget, self.samples = sims, samples
        self.game, self.moves = Game([tuple(m) for m in opening]), [list(m) for m in opening]
        self.trees = {}
        for colour, side in enumerate(sides):
            if side != SEAL and id(side) not in self.trees:
                self.trees[id(side)] = side.tree([tuple(m) for m in opening], seed*2+colour, tactics)
        self.seal_turns()

    @property
    def model(self):
        return self.sides[self.game.player]

    @property
    def tree(self):
        return self.trees[id(self.model)]

    def over(self):
        return self.game.winner >= 0 or len(self.moves) >= self.max_plies

    def play(self, q, r):
        self.game.play(q, r)
        for tree in self.trees.values():
            tree.advance((q, r))
        self.moves.append([q, r])

    def seal_turns(self):
        while not self.over() and self.sides[self.game.player] == SEAL:
            side, turn = self.game.player, self.seal(self.game, self.seal_ms)
            if not 1 <= len(turn) <= self.game.remaining:
                raise ValueError(f'Seal returned {len(turn)} moves for {self.game.remaining} placements')
            for q, r in turn:
                if not self.game.legal(q, r):
                    raise ValueError(f'Seal played an illegal placement {q}, {r}')
                self.play(int(q), int(r))
                if self.over():
                    break
            if not self.over() and self.game.player == side:
                raise ValueError('Seal did not complete its turn')

    def searched(self, result):
        self.play(*map(int, result['action']))
        self.seal_turns()
        return not self.over()

    def finish(self):
        winner = self.game.winner
        self.game.close()
        for tree in self.trees.values():
            tree.close()
        return dict(self.record, winner=winner, reason=self.reason or ('six-in-a-row' if winner >= 0 else 'cap'),
                    plies=len(self.moves), moves=self.moves)


def play(games, leaf_batch, heartbeat=lambda finished: None):
    """Run MatchGames to completion in one engine; returns their records in input order. heartbeat(finished
    games) is called after every engine step."""
    engine, records = Engine(leaf_batch), {}
    for game in games:
        if game.over():
            records[id(game)] = game.finish()
        else:
            engine.add(game)
    while engine.slots:
        for game in engine.step():
            records[id(game)] = game.finish()
        heartbeat(len(records))
    return [records[id(g)] for g in games]


def pair_seed(run_seed, label, pair):
    return int.from_bytes(hashlib.sha256(f'{run_seed}/{label}/{pair}'.encode()).digest()[:4], 'little')


def paired_games(challenger, rival, games, label, config, settings, seal, first_pair=0, **record):
    """`games` MatchGames: games/2 openings (train.task_opening from run seed, `label` and pair index, pairs
    numbered from `first_pair`), each played once with the candidate as colour 0 and once as colour 1."""
    if games < 2 or games % 2:
        raise ValueError('Paired matches need an even game count of at least two')
    out = []
    for pair in range(first_pair, first_pair+games//2):
        seed = pair_seed(config.seed, label, pair)
        opening = task_opening(seed, True, settings.max_plies, settings.opening_suite)['opening']
        for colour in (0, 1):
            sides = [rival, rival]; sides[colour] = challenger
            out.append(MatchGame(sides, opening, seed, settings.sims, settings.root_samples, settings.tactics,
                                 settings.max_plies, dict(record, pair=pair, seed=seed, opening=[list(m) for m in opening],
                                 challenger_color=colour), seal, settings.seal_ms))
    return out


def pair_scores(records):
    """{pair seed: candidate points / 2} with wins 1, caps 1/2, losses 0 per game; pairs must be intact."""
    pairs = {}
    for g in records:
        pairs.setdefault(g['seed'], []).append(1. if g['winner'] == g['challenger_color'] else .5 if g['winner'] < 0 else 0.)
    if any(len(p) != 2 for p in pairs.values()):
        raise ValueError('Rating requires intact colour-swapped opening pairs')
    return {seed: sum(p)/2 for seed, p in pairs.items()}


def summary(records):
    """Draw-aware match summary: decisive wins/losses, caps, pair score, its paired Hoeffding lower bound
    and an Elo delta from the points with a +1/2 continuity correction."""
    scores = list(pair_scores(records).values())
    wins = sum(g['winner'] == g['challenger_color'] for g in records)
    capped = sum(g['winner'] < 0 for g in records)
    losses = len(records)-wins-capped
    score = sum(scores)/len(scores)
    points = wins+capped/2
    return dict(wins=wins, losses=losses, capped=capped, games=len(records), pair_score=score,
                pair_score_lower=score-math.sqrt(math.log(40)/(2*len(scores))),
                elo_delta=400*math.log10((points+.5)/(len(records)-points+.5)))


def sprt(records, elo0, elo1, alpha, beta):
    """Generalized SPRT (Van den Bergh; Fishtest's pentanomial test) on colour-swapped opening pairs.

    Each pair is one observation x in {0, 1/4, 1/2, 3/4, 1}: the candidate's points over its two games / 2,
    with caps half a point (without caps this is the trinomial 0/1/2 wins per pair), so within-pair
    correlation never inflates the evidence. Let p^ be the empirical category frequencies over N pairs
    (plus 1e-3 pseudo-counts so every category is supported) and s_j = 1/(1 + 10^(-elo_j/400)) the expected
    score under H_j (candidate = champion + elo_j). p_j is the maximum-likelihood category distribution with
    mean s_j: p_j,k = p^_k / (1 + l_j (x_k - s_j)), with l_j solving sum_k p^_k (x_k - s_j)/(1 + l_j (x_k - s_j)) = 0.
    LLR = N sum_k p^_k log(p_1,k / p_0,k). H1 (promote) when LLR >= log((1-beta)/alpha), H0 when
    LLR <= log(beta/(1-alpha)), otherwise decision None.
    """
    x = np.arange(5)/4
    counts = np.zeros(5)
    for score in pair_scores(records).values():
        counts[round(4*score)] += 1
    n = counts.sum()
    phat = (counts+1e-3)/(n+5e-3)

    def fit(elo):
        d = x-1/(1+10**(-elo/400))
        low, high = -1/d.max()+1e-12, -1/d.min()-1e-12
        for _ in range(200):
            lam = (low+high)/2
            if np.sum(phat*d/(1+lam*d)) > 0:
                low = lam
            else:
                high = lam
        return phat/(1+lam*d)

    llr = float(n*np.sum(phat*np.log(fit(elo1)/fit(elo0))))
    lower, upper = math.log(beta/(1-alpha)), math.log((1-beta)/alpha)
    return dict(llr=llr, bound_lower=lower, bound_upper=upper, games=int(2*n), elo0=elo0, elo1=elo1, alpha=alpha, beta=beta,
                pair_counts=counts.astype(int).tolist(), decision='H1' if llr >= upper else 'H0' if llr <= lower else None)


def report_path(run, candidate, opponent):
    return Path(run)/'evaluations'/f'{candidate.replace("/", "-")}-vs-{opponent.replace("/", "-")}'/'report.json'


def make_report(candidate, opponent, records, shas, settings):
    """Report of a finished comparison; `shas` maps checkpoint ids to their ema.pt digests."""
    return dict(candidate=candidate, opponent=opponent, created_at=time.time(),
                candidate_sha256=shas[candidate], opponent_sha256=SEAL if opponent == SEAL else shas[opponent],
                settings=asdict(settings),
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
    """(point ratings, 95% intervals, posterior draws per rated id), the model of checkpoint_league.rate_league
    over string ids with caps as half points: per comparison a Dirichlet(counts + 1/2) posterior over the pair
    scores 0, 1/2, 1, 3/2, 2, each draw solved jointly so draws of different ids are paired."""
    comparisons = []
    for report in reports:
        counts = np.zeros(5)
        for score in pair_scores(report['games']).values():
            counts[round(4*score)] += 1
        comparisons.append((report['candidate'], report['opponent'], counts+.5, len(report['games'])))
    edge = lambda a, b, p, n: (a, b, n*(p@np.arange(5))/4, n)
    point = solve(ids, anchor, [edge(a, b, alpha/alpha.sum(), n) for a, b, alpha, n in comparisons])
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in ids if point[name] is not None}
    for _ in range(samples if comparisons else 0):
        ratings = solve(ids, anchor, [edge(a, b, rng.dirichlet(alpha), n) for a, b, alpha, n in comparisons])
        for name in draws:
            draws[name].append(ratings[name])
    return point, {name: np.quantile(v, [.025, .975]).tolist() if v else [0., 0.] for name, v in draws.items()}, draws


def write_league(run, league, config):
    """Recompute ratings from every report among rated (not skipped) ids and Seal, then publish league.json;
    skipped entries keep elo and elo_interval null."""
    run = Path(run)
    ids = [c['id'] for c in league['checkpoints'] if not c.get('skipped')]
    reports = [json.loads(p.read_text()) for p in sorted((run/'evaluations').glob('*/report.json'))]
    reports = [r for r in reports if r['candidate'] in ids and (r['opponent'] in ids or r['opponent'] == SEAL)]
    seal_games = sum(len(r['games']) for r in reports if r['opponent'] == SEAL)
    names = ids+([SEAL] if seal_games else [])
    point, intervals, draws = rate(names, ids[0], reports, seed=config.seed) if ids else ({}, {}, {})
    for c in league['checkpoints']:
        c['elo'], c['elo_interval'] = point.get(c['id']), intervals.get(c['id'])
    # Latest rated checkpoint of each variant; a-b intervals come from the same joint draws.
    latest = {}
    for c in league['checkpoints']:
        if point.get(c['id']) is not None and c['step'] >= latest.get(c['variant'], c)['step']:
            latest[c['variant']] = c
    heads = [c['id'] for _, c in sorted(latest.items())]
    league['differences'] = [
        dict(a=a, b=b, elo_delta=point[a]-point[b],
             interval=np.quantile(np.subtract(draws[a], draws[b]), [.025, .975]).tolist() if draws[a] else [0., 0.])
        for i, a in enumerate(heads) for b in heads[i+1:]]
    league['anchors'] = {SEAL: dict(elo=point.get(SEAL), elo_interval=intervals.get(SEAL), games=seal_games)}
    league['rating_note'] = RATING_NOTE
    league['updated_at'] = time.time()
    write_json(run/'league.json', league)


class Pacer:
    """Ceiling on the evaluator's playing share of wall time, a token bucket: credit accrues at `share` per
    wall second, capped at `share * window` while not playing (a new pacer starts full), playing spends
    `weight` (worker processes) per second, and wait(tick) sleeps while credit is negative, calling tick()
    before each sleep of at most 10 s; share 1 with weight 1 never waits. used() is the weighted playing
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

    def wait(self, tick=lambda: None):
        self.refill(self.clock())
        while self.credit < -1e-6:
            tick()
            self.sleep(min(10., -self.credit/self.share))
            self.refill(self.clock())


def play_pairs(config, settings, candidate, opponent, first_pair, games, seal=None,
               heartbeat=lambda finished, placements: None):
    """Records of `games` paired games (paired_games, labelled by the candidate's checkpoint id, pairs numbered
    from `first_pair`) of a candidate Model vs a Model or SEAL; heartbeat(finished games, placements played)
    after every engine step."""
    name = lambda side: SEAL if side == SEAL else side.checkpoint
    matches = paired_games(candidate, opponent, games, candidate.checkpoint, config, settings, seal, first_pair,
                           candidate=candidate.checkpoint, opponent=name(opponent))
    opened = sum(len(g.moves) for g in matches)
    return play(matches, config.actor.leaf_batch,
                lambda finished: heartbeat(finished, sum(len(g.moves) for g in matches)-opened))


def match_entry(opponent, report):
    s = report['summary']
    return dict(opponent=opponent, wins=s['wins'], losses=s['losses'], capped=s['capped'], games=s['games'],
                opening_pair_p=report['metrics']['opening_pair_p'], elo_delta=s['elo_delta'],
                pair_score_lower=s['pair_score_lower'])


class Evaluator:
    """The `loop` evaluator of one run (module contract) with `settings` in place of config.evaluation.
    step() does one unit of work. Each round runs in this process (processes 1) or is split into contiguous
    shares of its opening pairs over `processes` `pairs` subprocesses (each loads both models, about 0.7 GB
    of VRAM on CUDA); records, seeds and openings do not depend on `processes`. publish() maintains
    evaluator-status.json: {stage ('idle', 'playing', 'throttled' or 'failed'), updated_at, comparison
    ({candidate, opponent, kind 'champion', 'previous' or 'anchor'} or null), games_played, games_planned
    (sprt_max_games for the champion), placements_per_second (current round, both sides, all processes),
    backlog (unrated checkpoint ids at the last step), processes, eval_share_used (Pacer.used), error}."""

    def __init__(self, run, config, settings, pacer, processes=1):
        self.run, self.config, self.settings, self.pacer, self.processes = Path(run), config, settings, pacer, processes
        path = self.run/'league.json'
        self.league = json.loads(path.read_text()) if path.exists() else dict(champion=None, checkpoints=[])
        self.models, self.partial, self.seal, self.written = {}, {}, None, 0.
        self.status = dict(stage='idle', updated_at=None, comparison=None, games_played=0, games_planned=0,
                           placements_per_second=None, backlog=[], processes=processes, eval_share_used=0., error=None)

    def publish(self, force=False, **fields):
        """Update the status; rewrite the file when forced or STATUS_SECONDS after the last write."""
        self.status.update(fields)
        if force or time.monotonic()-self.written >= STATUS_SECONDS:
            self.written = time.monotonic()
            write_json(self.run/'evaluator-status.json',
                       dict(self.status, updated_at=time.time(), eval_share_used=self.pacer.used()))

    def use(self, *names):
        """Load the named checkpoints (Seal once, lazily) and release every other model."""
        dropped = [n for n in self.models if n not in names]
        for name in dropped:
            del self.models[name]
        if dropped and self.config.device == 'cuda':
            torch.cuda.empty_cache()
        for name in names:
            if name == SEAL:
                self.seal = self.seal or Seal()
            elif name not in self.models:
                self.models[name] = load(self.run, self.config, source=(name, self.run/'checkpoints'/name/'ema.pt'))

    def spread(self, cid, opponent, first_pair, games, heartbeat):
        """play_pairs split over `processes` `pairs` subprocesses; records in pair order. heartbeat(finished,
        placements) summed over the workers about every second."""
        pairs, n = games//2, self.processes
        shares = [s for s in (pairs//n+(k < pairs % n) for k in range(n)) if s]
        with tempfile.TemporaryDirectory() as tmp:
            jobs, pair = [], first_pair
            for k, share in enumerate(shares):
                out = Path(tmp)/f'{k}.json'
                jobs.append((subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'pairs', '--run', str(self.run),
                                               '--candidate', cid, '--opponent', opponent, '--first-pair', str(pair),
                                               '--games', str(2*share), '--settings', json.dumps(asdict(self.settings)),
                                               '--out', str(out)]), out))
                pair += share
            while any(job.poll() is None for job, _ in jobs):
                time.sleep(1.)
                progress = [json.loads(p.read_text()) for _, out in jobs if (p := out.with_suffix('.progress')).exists()]
                heartbeat(sum(p['finished'] for p in progress), sum(p['placements'] for p in progress))
            if any(job.returncode for job, _ in jobs):
                raise RuntimeError(f'evaluation worker exit codes {[job.returncode for job, _ in jobs]}')
            return [record for _, out in jobs for record in json.loads(out.read_text())]

    def round(self, cid, opponent, kind, planned):
        """After pacing, play the next min(games, planned - played) games of cid vs opponent, opening pairs
        numbered on from earlier rounds; returns the comparison's {records, seconds} so far. A champion round
        is not started once a newer checkpoint of the variant exists."""
        done = self.partial.setdefault((cid, opponent), dict(records=[], seconds=0.))
        records = done['records']
        self.pacer.wait(lambda: self.publish(True, stage='throttled'))
        if kind == 'champion' and self.newer(cid):
            return done
        count, start = min(self.settings.games, planned-len(records)), self.pacer.clock()
        self.publish(True, stage='playing', comparison=dict(candidate=cid, opponent=opponent, kind=kind),
                     games_played=len(records), games_planned=planned, placements_per_second=None)
        heartbeat = lambda finished, placements: self.publish(
            games_played=len(records)+finished, placements_per_second=placements/max(self.pacer.clock()-start, 1e-9))
        if self.processes == 1:
            self.use(cid, opponent)
            results = play_pairs(self.config, self.settings, self.models[cid], SEAL if opponent == SEAL else self.models[opponent],
                                 len(records)//2, count, self.seal, heartbeat)
        else:
            results = self.spread(cid, opponent, len(records)//2, count, heartbeat)
        self.pacer.played(start, self.pacer.clock(), min(self.processes, count//2))
        done['seconds'] += self.pacer.clock()-start
        for record in results:
            if record['reason'] == 'span':
                log_event(self.run, 'evaluator', 'error', f'{cid} vs {opponent} pair {record["pair"]}: game counted as '
                          f'capped at ply {record["plies"]}, a searched position spans more than the largest crop',
                          candidate=cid, opponent=opponent)
        records += results
        placements = sum(r['plies']-len(r['opening']) for r in results)
        self.publish(True, games_played=len(records), placements_per_second=placements/max(self.pacer.clock()-start, 1e-9))
        return done

    def newer(self, cid):
        """Whether an unrated checkpoint of cid's variant other than cid exists (it is newer: cid was chosen as
        the newest)."""
        known = {c['id'] for c in self.league['checkpoints']}
        return any(e[0] != cid and e[0].split('/')[0] == cid.split('/')[0] and e[0] not in known for e in checkpoints(self.run))

    def report(self, cid, opponent, kind, planned, complete):
        """The report of cid vs opponent, reused when on disk. Otherwise the champion comparison plays rounds
        until complete(records) or, after a round, a newer checkpoint of the variant exists (SPRT decision
        'superseded'); optional comparisons play one round per call and return None until complete. A
        finished report is written and logged; a champion comparison superseded before any game returns None."""
        path = report_path(self.run, cid, opponent)
        if path.exists():
            return json.loads(path.read_text())
        while True:
            done = self.round(cid, opponent, kind, planned)
            if complete(done['records']) or (kind == 'champion' and self.newer(cid)):
                break
            if kind != 'champion':
                return None
        if not done['records']:
            return None
        shas = {name: digest(self.run/'checkpoints'/name/'ema.pt') for name in (cid, opponent) if name != SEAL}
        report = make_report(cid, opponent, done['records'], shas, self.settings)
        if kind == 'champion':
            result = self.test(done['records'])
            report['metrics']['sprt'] = dict(result, decision=result['decision'] or ('max-games' if complete(done['records']) else 'superseded'))
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
        del self.partial[cid, opponent]
        s, test = report['summary'], report['metrics'].get('sprt')
        log_event(self.run, 'evaluator', 'match', f'{cid} vs {opponent}: +{s["wins"]} -{s["losses"]} ={s["capped"]}'
                  + (f' (SPRT {test["decision"]}, LLR {test["llr"]:.2f})' if test else ''),
                  candidate=cid, opponent=opponent, comparison=kind, **s, sprt=test, seconds=done['seconds'])
        return report

    def test(self, records):
        s = self.settings
        return sprt(records, s.sprt_elo0, s.sprt_elo1, s.sprt_alpha, s.sprt_beta)

    def rate(self, entry):
        """Play the champion SPRT of a chosen checkpoint (none for the first), add it to the league (superseded
        true when a newer checkpoint cut its SPRT short) and promote on H1."""
        cid, path, _ = entry
        variant, step = cid.split('/')
        champion, s = self.league['champion'], self.settings
        report = None if champion is None else self.report(
            cid, champion, 'champion', s.sprt_max_games,
            lambda records: self.test(records)['decision'] is not None or len(records) >= s.sprt_max_games)
        if champion is not None and report is None:
            self.league['checkpoints'].append(dict(id=cid, variant=variant, step=int(step), skipped=True,
                                                   elo=None, elo_interval=None, matches=[]))
            log_event(self.run, 'evaluator', 'skip', f'skipped {cid}: a newer checkpoint appeared before its first champion round',
                      checkpoints=[cid], candidate=cid)
            write_league(self.run, self.league, self.config)
            return
        reports = {} if report is None else {champion: report}
        decision = reports[champion]['metrics']['sprt']['decision'] if reports else None
        promoted = champion is None or decision == 'H1'
        self.league['checkpoints'].append(dict(id=cid, variant=variant, step=int(step), ema_sha256=digest(path/'ema.pt'),
                                               elo=None, elo_interval=None, **({'superseded': True} if decision == 'superseded' else {}),
                                               matches=[match_entry(o, r) for o, r in reports.items()]))
        if promoted:
            self.league['champion'] = cid
            write_json(self.run/'champion.json', dict(checkpoint=cid, ema_sha256=digest(path/'ema.pt'), updated_at=time.time()))
            log_event(self.run, 'evaluator', 'promotion', f'{cid} promoted to champion' + (f' over {champion}' if champion else ''),
                      checkpoint=cid, previous_champion=champion)
        write_league(self.run, self.league, self.config)
        print(f'{cid}: ' + ', '.join(f'vs {o} +{r["summary"]["wins"]} -{r["summary"]["losses"]} ={r["summary"]["capped"]}'
                                      for o, r in reports.items()) + (' -> champion' if promoted else f' ({decision})' if decision else ''), flush=True)

    def optional(self):
        """(league entry, opponent, kind, games) of the newest rated checkpoint missing its previous-checkpoint
        comparison (when previous_games > 0) or its due Seal anchor (when anchor_games > 0), else None."""
        s = self.settings
        rated = [c for c in self.league['checkpoints'] if not c.get('skipped')]
        for index in reversed(range(len(rated))):
            c = rated[index]
            earlier = [p for p in rated[:index] if p['variant'] == c['variant'] and p['step'] < c['step']]
            wanted = [(max(earlier, key=lambda p: p['step'])['id'], 'previous', s.previous_games)] if earlier and s.previous_games else []
            wanted += [(SEAL, 'anchor', s.anchor_games)] if s.anchor_games and index % s.anchor_every == 0 else []
            played = {m['opponent'] for m in c['matches']}
            for opponent, kind, games in wanted:
                if opponent not in played:
                    return c, opponent, kind, games
        return None

    def step(self):
        """One unit of work; False when there is none. Rates the newest unrated checkpoint of the variant whose
        newest unrated checkpoint is oldest, skipping that variant's older unrated checkpoints; otherwise plays
        one round of an optional comparison."""
        known = {c['id'] for c in self.league['checkpoints']}
        unrated = [e for e in checkpoints(self.run) if e[0] not in known]
        self.status['backlog'] = [e[0] for e in unrated]
        if unrated:
            variant = lambda e: e[0].split('/')[0]
            heads = {variant(e): e for e in unrated}
            head = next(e for e in unrated if heads[variant(e)] is e)
            skipped = [e[0] for e in unrated if variant(e) == variant(head) and e is not head]
            if skipped:
                self.league['checkpoints'] += [dict(id=cid, variant=variant(head), step=int(cid.split('/')[1]), skipped=True,
                                                    elo=None, elo_interval=None, matches=[]) for cid in skipped]
                write_league(self.run, self.league, self.config)
                log_event(self.run, 'evaluator', 'skip', f'skipped {", ".join(skipped)} for {head[0]}',
                          checkpoints=skipped, candidate=head[0])
            self.rate(head)
            return True
        task = self.optional()
        if task is None:
            return False
        entry, opponent, kind, games = task
        report = self.report(entry['id'], opponent, kind, games, lambda records: len(records) >= games)
        if report is not None:
            entry['matches'].append(match_entry(opponent, report))
            write_league(self.run, self.league, self.config)
        return True


def loop(args):
    run = Path(args.run)
    config = dense_config.load(run)
    settings = dense_config.override(config.evaluation, args, 'eval_')
    evaluator = Evaluator(run, config, settings, Pacer(1. if args.once else settings.eval_share), args.processes)
    try:
        while True:
            if evaluator.step():
                continue
            evaluator.publish(True, stage='idle', comparison=None)
            if args.once:
                break
            time.sleep(args.poll)
    except Exception as error:
        message = f'{type(error).__name__}: {error}'
        evaluator.publish(True, stage='failed', error=message)
        log_event(run, 'evaluator', 'error', message, comparison=evaluator.status['comparison'])
        raise


def pairs(args):
    """Worker of Evaluator.spread: plays its share with --settings (EvaluationSettings as JSON), writes the
    records to --out and {finished, placements} to <out>.progress about every STATUS_SECONDS."""
    run = Path(args.run)
    config = dense_config.load(run)
    model = lambda name: load(run, config, source=(name, run/'checkpoints'/name/'ema.pt'))
    out, written = Path(args.out), [0.]

    def heartbeat(finished, placements):
        if time.monotonic()-written[0] >= STATUS_SECONDS:
            written[0] = time.monotonic()
            write_json(out.with_suffix('.progress'), dict(finished=finished, placements=placements))
    records = play_pairs(config, dense_config.EvaluationSettings(**json.loads(args.settings)), model(args.candidate),
                         SEAL if args.opponent == SEAL else model(args.opponent), args.first_pair, args.games,
                         Seal() if args.opponent == SEAL else None, heartbeat)
    write_json(out, records)


def calibrate(args):
    """Continue the newest capped games from their cap with the champion (both sides, `--sims`) for up to
    --extra plies, then score each original ply's TD(lambda) target (dense_data.value_targets), the raw root
    value and the masked baseline (p = 1/2) against the realised result, overall and per continuation length."""
    run = Path(args.run)
    config = dense_config.load(run)
    model = load(run, config)
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
                       dict(index=k)) for k, e in enumerate(chosen)]
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
    run = Path(args.run)
    config = dense_config.load(run)
    settings = config.evaluation
    if args.sims:
        settings = replace(settings, sims=args.sims, root_samples=min(settings.root_samples, args.sims))
    models = {}
    for name in (args.a, args.b):
        if name != SEAL and name not in models:
            path = Path(name) if Path(name).is_file() else run/'checkpoints'/name/'ema.pt'
            models[name] = load(run, config, source=(name, path))
    if args.a == args.b:
        raise ValueError('A match needs two distinct players')
    if args.a == SEAL:
        raise ValueError('Seal plays as --b; pass the checkpoint as --a')
    started = time.perf_counter()
    records = play(paired_games(models[args.a], SEAL if args.b == SEAL else models[args.b], args.games,
                                f'match/{args.a}/{args.b}', config, settings, Seal() if args.b == SEAL else None,
                                candidate=args.a, opponent=args.b), config.actor.leaf_batch)
    report = make_report(args.a, args.b, records, {n: m.sha for n, m in models.items()}, settings)
    print(json.dumps(dict(summary=report['summary'], metrics=report['metrics'], seconds=time.perf_counter()-started), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('loop'); p.add_argument('--run', required=True); p.add_argument('--once', action='store_true')
    p.add_argument('--poll', type=float, default=30.)
    p.add_argument('--processes', type=int, default=1, help='worker processes per round (about 0.7 GB of VRAM each)')
    dense_config.add_arguments(p.add_argument_group('evaluation overrides for this process'), dense_config.EvaluationSettings, 'eval_')
    p = sub.add_parser('pairs', help='internal: one worker of loop --processes')
    for flag in ('--run', '--candidate', '--opponent', '--settings', '--out'):
        p.add_argument(flag, required=True)
    p.add_argument('--first-pair', type=int, required=True); p.add_argument('--games', type=int, required=True)
    p = sub.add_parser('calibrate'); p.add_argument('--run', required=True)
    p.add_argument('--lambdas', type=float, nargs='+', default=[.5, .7, .9, 1.])
    p.add_argument('--games', type=int, default=100); p.add_argument('--sims', type=int, default=64)
    p.add_argument('--extra', type=int, default=256)
    p = sub.add_parser('match'); p.add_argument('--run', required=True); p.add_argument('--a', required=True)
    p.add_argument('--b', required=True); p.add_argument('--games', type=int, default=32); p.add_argument('--sims', type=int)
    args = parser.parse_args()
    dict(loop=loop, pairs=pairs, calibrate=calibrate, match=match)[args.command](args)


if __name__ == '__main__':
    main()
