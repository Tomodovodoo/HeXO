"""Dense checkpoint evaluator: paired league games, Bradley-Terry ratings, champion promotion, calibration.

Run layout: dense_config. Subcommands
  loop       rate the newest unrated checkpoint of a variant (dense_selfplay.checkpoints order) against the
             champion; that variant's older unrated checkpoints enter league.json with skipped true and elo null
             and are never played (KataGo's gatekeeper: only the newest candidate meets the champion). When
             no checkpoint waits it plays the current champion's scheduled Seal anchor (`Evaluator.anchor`), then
             the optional comparisons (`Evaluator.optional`): panels and idle rematches, then, newest rated
             checkpoint first, `previous_games` vs the previous rated checkpoint of the variant; then, with
             idle_fill (off with --once), fill work until a checkpoint waits (`Evaluator.fill`): the champion vs
             Seal until their Elo interval is anchor_target_halfwidth narrow, then `games` of the newest rated
             checkpoint vs the previous champion ('generalization': does its edge carry beyond the incumbent it
             met), then the widest pair of the league ladder. Fill games are ordinary reports; the champion's fill
             games vs Seal also count toward its scheduled anchors. Panel, optional and fill pairings are played
             only while `informative` (expected score at most max_expected_score for either side).
             Play: one continuous pool of pool_games games in flight on one engine (`Pool`, `Evaluator.session`),
             like the actors' games: a finished game's slot is refilled at once with the next opening pair of its
             pairing (both colours start together), so the batch stays full and results stream per game. Every
             completed colour pair is appended to its report at once, so a restart loses only the games in flight
             and resumes the pairing (a candidate whose report against the champion exists is rated first). A
             pairing change or a newer checkpoint stops new games of the old pairing while its running games
             finish and count. All play is paced by `eval_share` (Pacer: no new game while its credit is
             negative); --eval-* flags override evaluation settings for this process (reports record the
             effective settings). Writes evaluations/<a>-vs-<b>/report.json, league.json, champion.json on
             promotion, evaluator-status.json (Evaluator.publish) and events. Every pairing, Seal anchors included,
             draws its openings from the opening book of opening_suite (dense_openings: the live book 'book' or a
             frozen suite such as 'standard-v1'); each completed pair is recorded on the book's nodes, a live book is
             refreshed between steps (`Evaluator.refresh_openings`), and reports are reused only under the book state
             they were played in (`same_protocol`). On start every report gets an `id` if it lacks one
             (dense_openings.stamp) and the book counts the pairs its file misses (dense_openings.Book.reconcile).
  calibrate  continue capped self-play games with the champion and score TD(lambda) value targets against
             the realised results.
  match      ad hoc paired match between two checkpoints (run ids or paths) or a checkpoint and Seal.

Scoring: a capped game (at the ply limit, or reason 'span' when a searched position does not fit the largest crop) is
half a point for each side. Pair score = candidate points / 2 over its two games; decisions use completed pairs.
Promotion (`decision`): 'posterior' (default) decides on the league-wide posterior (dense_posterior: Bradley-Terry
ratings of every rated checkpoint, the candidate and Seal from every protocol-matching report, with a per-pair
matchup deviation of prior sd matchup_prior_elo so a pair's own games dominate when they disagree with the
transitive picture). Delta = r_candidate - r_champion + their deviation. After at least sprt_min_games direct
games, while the candidate's rating sd about the league mean is at most uncertainty_parity times the champion's and
the direct-only and pooled 95% intervals overlap, the candidate is promoted when it has the highest posterior rating
of the rated checkpoints and P(delta > sprt_elo0) >= promote_confidence, and rejected when that probability is at
most 1 - promote_confidence (`Evaluator.verdict`), re-judged after every completed colour pair; until then direct
games keep playing. On start the rule is re-applied to the existing reports (`Evaluator.review`). Direct games fill
the pool until sprt_min_games are complete; after that at most evidence_share of the pool may go to the evidence
pairing that most reduces Var(delta) (value of information: the candidate or champion vs the previous champion or
Seal, `Evaluator.evidence`), up to sprt_max_games direct games. Supersession settles once the games in flight have
finished: the candidate is promoted when P(delta > sprt_elo0) >= promote_confidence, however the readiness
conditions stand, else 'superseded'. Evidence games are ordinary reports, kind 'evidence'.
With 'sprt', the comparison with the champion is a sequential test (see `sprt`) until it accepts H0 or H1, reaches
`sprt_max_games` (decision 'max-games') or is superseded by a newer checkpoint of its variant. H1 promotes; a
superseded comparison is settled on the same posterior: it promotes when P(delta > sprt_elo0) >= promote_confidence,
recorded as metrics.sprt.settled {pair_score, p_better, promote} with a 'settle' event. The first rated checkpoint becomes champion unopposed.

Panels: a checkpoint rated while a champion exists (with extra_opponents > 0) gets entry `panel` {incumbent}, the
champion it met. Its members are re-derived from the current ladder whenever the panel is scheduled or judged:
the extra_opponents most informative rated checkpoints against the current champion (`panel_members`). Its games
and the incumbent's top-ups against the members are optional comparisons, the current champion's first. SPRT H1
promotes at once; the panel is a post-hoc regression check: once every current member has its games, the panel
records members, candidate_score, incumbent_score, z and veto in the entry and a 'panel' event. A vetoed checkpoint is marked
`demoted` ('regression' event) and never promoted or restored again; a vetoed champion is replaced by its most
recent non-demoted, non-skipped predecessor along panel incumbents (`Evaluator.settle`).

Actor pointer: whenever the status file is written the evaluator keeps actor.json (`Evaluator.point`), the model of
actors with model_source 'newest_veto': the newest export of learner.variant unless vetoed, then the best-rated
checkpoint until the next export.

league.json: {champion, reign_from, reign_games, checkpoints: [{id, variant, step, ema_sha256, elo, elo_interval, matches, skipped?,
superseded?, panel?, demoted?, verdict? (the posterior verdict that rated it)}], differences, ladder, ladder_top, anchors, matrix, openings
(dense_openings.summary: P1/P2 results overall and per player, and the state of each opening book), rating_note, updated_at}.
differences and ladder are [{a, b, elo_delta, interval}] over pairs of the variant heads and of the ladder_top
(fill_top) best rated, not demoted checkpoints, a above b, intervals from the joint rating draws. anchors.seal is {elo,
elo_interval, games, matches: [{checkpoint, wins, losses, capped, games, elo_delta}] in league order,
latest_delta}: elo_delta is each checkpoint's direct-match Elo minus Seal, latest_delta the newest one's. reign_from
is the number of checkpoint entries and reign_games the champion's Seal games when it was promoted or restored
(absent: its own position + 1 and 0). `matrix` (`payoff`) holds
only pairs that met; readers compute p for other rated pairs from the ratings (dense_selfplay.expected). Leagues
written before `matrix`, `ladder`, `panel` and `openings` existed lack those keys; the Evaluator adds `matrix` on start, and
rebuilds the league when its ladder_top differs from the effective fill_top or its book counted pairs it had missed.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np
import torch

import dense_config
from dense_config import log_event
import dense_data
import dense_openings
from dense_posterior import Posterior
import hexnet
from arena import Seal
from dense_selfplay import Engine, checkpoints, expected, load, resolve
from hexo import Game
from klent import digest
from train import paired_metrics, write_json

SEAL = 'seal'
PACE_WINDOW = 3600.  # a Pacer banks at most share * PACE_WINDOW seconds of idle credit
STATUS_SECONDS = 2.
PROTOCOL = ('sims', 'root_samples', 'max_plies', 'tactics', 'opening_suite', 'opening_book', 'seal_ms')  # settings a reused report must share
REMATCH_SPRT_LIMIT = 2    # a continued champion SPRT stops at this many times sprt_max_games
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
    """Run MatchGames to completion in one engine; returns their records in input order. heartbeat(records of
    the finished games, in finishing order) is called after every engine step."""
    engine, records = Engine(leaf_batch), {}
    for game in games:
        if game.over():
            records[id(game)] = game.finish()
        else:
            engine.add(game)
    while engine.slots:
        for game in engine.step():
            records[id(game)] = game.finish()
        heartbeat(list(records.values()))
    return [records[id(g)] for g in games]


def pair_seed(run_seed, label, pair):
    return int.from_bytes(hashlib.sha256(f'{run_seed}/{label}/{pair}'.encode()).digest()[:4], 'little')


def paired_games(challenger, rival, games, label, config, settings, seal, book, first_pair=0, **record):
    """`games` MatchGames: games/2 openings drawn from `book` (dense_openings.Book.draw) by the seed of run seed,
    `label` and pair index (pairs numbered from `first_pair`), each played once with the candidate as colour 0 and
    once as colour 1."""
    if games < 2 or games % 2:
        raise ValueError('Paired matches need an even game count of at least two')
    out = []
    for pair in range(first_pair, first_pair+games//2):
        seed = pair_seed(config.seed, label, pair)
        opening = book.draw(seed)
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


def tally(records, test=None):
    """Running score of a comparison's finished games: wins, losses, capped and games over all of them; over its
    complete opening pairs: pairs, pair_score, pair_interval (`summary`'s paired Hoeffding 95% bounds, clipped to
    [0, 1]), elo_delta and elo_interval (`summary`'s continuity-corrected Elo of the points at the score and at its
    bounds), and llr, bound_lower and bound_upper of test(complete records) (an `sprt` result) when given. Pair
    fields are None before the first complete pair."""
    wins = sum(g['winner'] == g['challenger_color'] for g in records)
    capped = sum(g['winner'] < 0 for g in records)
    out = dict(wins=wins, losses=len(records)-wins-capped, capped=capped, games=len(records), pairs=0, pair_score=None,
               pair_interval=None, elo_delta=None, elo_interval=None, llr=None, bound_lower=None, bound_upper=None)
    seeds = [g['seed'] for g in records]
    complete = [g for g in records if seeds.count(g['seed']) == 2]
    if complete:
        s, n = summary(complete), len(complete)
        half = s['pair_score']-s['pair_score_lower']
        bounds = [max(0., s['pair_score']-half), min(1., s['pair_score']+half)]
        elo = lambda p: 400*math.log10((p*n+.5)/(n-p*n+.5))
        out.update(pairs=n//2, pair_score=s['pair_score'], pair_interval=bounds, elo_delta=s['elo_delta'],
                   elo_interval=[elo(p) for p in bounds])
        if test:
            result = test(complete)
            out.update(llr=result['llr'], bound_lower=result['bound_lower'], bound_upper=result['bound_upper'])
    return out


def report_path(run, candidate, opponent):
    return Path(run)/'evaluations'/f'{candidate.replace("/", "-")}-vs-{opponent.replace("/", "-")}'/'report.json'


def make_report(candidate, opponent, records, shas, settings, report_id=None):
    """Report of a finished comparison; `shas` maps checkpoint ids to their ema.pt digests and `report_id` is the
    report's `id` (a new one when None), kept while pairs are appended."""
    return dict(id=report_id or uuid.uuid4().hex, candidate=candidate, opponent=opponent, created_at=time.time(),
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


_reports = {}


def load_reports(run, settings=None):
    """Every evaluations/*/report*.json: each pairing's report.json and the reports `Evaluator.open` archived beside it
    (cached per path and modification time); with `settings`, only the reports played under its PROTOCOL."""
    reports = []
    for path in sorted((Path(run)/'evaluations').glob('*/report*.json')):
        stamp = path.stat().st_mtime_ns
        if _reports.get(path, (None,))[0] != stamp:
            _reports[path] = stamp, json.loads(path.read_text())
        reports.append(_reports[path][1])
    return [r for r in reports if settings is None or same_protocol(r, settings)]


def same_protocol(report, settings):
    """Whether `report` was played under the PROTOCOL of `settings`. Under the live book that includes opening_book
    (dense_openings.Book.digest of its openings), which changes only at a book refresh: a report is reused while the
    book keeps its openings, and a refresh that changes them starts every comparison afresh. A frozen suite's name
    fixes its openings (opening_book ''); a report written before opening_book existed was played under one."""
    played = {'opening_book': '', **report['settings']}
    return all(played.get(k) == getattr(settings, k) for k in PROTOCOL)


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
    return 1-cap <= p <= cap


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


def write_league(run, league, config, top=None):
    """Recompute ratings, the payoff matrix and the ladder (`top` checkpoints, default config.evaluation.fill_top)
    from every report among rated (not skipped) ids and Seal, then publish league.json; skipped entries keep elo
    and elo_interval null. The ratings pool the reports of every protocol, opening-book states included, so a
    refresh never empties the ladder; the promotion posterior, panels and rematches use protocol-matching reports only
    (`same_protocol`)."""
    run = Path(run)
    ids = [c['id'] for c in league['checkpoints'] if not c.get('skipped')]
    reports = [r for r in load_reports(run) if r['candidate'] in ids and (r['opponent'] in ids or r['opponent'] == SEAL)]
    seal_games = sum(len(r['games']) for r in reports if r['opponent'] == SEAL)
    names = ids+([SEAL] if seal_games else [])
    point, intervals, draws = rate(names, ids[0], reports, seed=config.seed) if ids else ({}, {}, {})
    for c in league['checkpoints']:
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
    anchored = sorted((r for r in reports if r['opponent'] == SEAL), key=lambda r: ids.index(r['candidate']))
    matches = [dict(checkpoint=r['candidate'], **{k: r['summary'][k] for k in ('wins', 'losses', 'capped', 'games', 'elo_delta')})
               for r in anchored]
    league['anchors'] = {SEAL: dict(elo=point.get(SEAL), elo_interval=intervals.get(SEAL), games=seal_games, matches=matches,
                                    latest_delta=matches[-1]['elo_delta'] if matches else None)}
    league['matrix'] = payoff(reports, point)
    league['openings'] = dense_openings.summary(run, reports)
    league['rating_note'] = RATING_NOTE
    league['updated_at'] = time.time()
    write_json(run/'league.json', league)


class Pacer:
    """Ceiling on the evaluator's playing share of wall time, a token bucket: credit accrues at `share` per
    wall second, capped at `share * window` while not playing (a new pacer starts full), playing spends
    `weight` per second, and wait(tick) sleeps while credit is negative, calling tick()
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

    def ready(self):
        """Whether a new game may start now: the credit is not negative."""
        self.refill(self.clock())
        return self.credit >= -1e-6

    def wait(self, tick=lambda: None):
        self.refill(self.clock())
        while self.credit < -1e-6:
            tick()
            self.sleep(min(10., -self.credit/self.share))
            self.refill(self.clock())


class Pool:
    """A continuous pool of MatchGames on one Engine, the evaluator's counterpart of the actors' games in flight.
    Games belong to lanes (any hashable pairing key); add(lane, games) starts them at once, step() advances the
    engine once and returns [(lane, record)] of the games that finished, so a finished game's slot can be refilled
    before the next step. running() counts games in flight (`running`); moves() is the
    placements played after their openings by the games in flight."""

    def __init__(self, leaf_batch):
        self.engine, self.games, self.ready = Engine(leaf_batch), {}, []

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


def match_entry(opponent, report):
    s = report['summary']
    return dict(opponent=opponent, wins=s['wins'], losses=s['losses'], capped=s['capped'], games=s['games'],
                opening_pair_p=report['metrics']['opening_pair_p'], elo_delta=s['elo_delta'],
                pair_score_lower=s['pair_score_lower'])


class Evaluator:
    """The `loop` evaluator of one run (module contract) with `settings` in place of config.evaluation.
    step() does one unit of work, played as a `session` of the continuous pool (`Pool`, pool_games in flight).
    publish() maintains evaluator-status.json: {stage ('idle', 'playing', 'throttled' or 'failed'), updated_at,
    comparison ({candidate, opponent, kind 'champion', 'evidence', 'previous', 'anchor', 'panel', 'incumbent',
    'sprt', 'replacement', 'fill' or 'generalization'} of the session's main lane, or null), pool ([{candidate,
    opponent, kind, running, share}] per lane: games in flight and the games in flight it wants), started_at
    (epoch seconds when the session began), games_played (the main lane's report games plus its finished games
    whose colour partner still runs), games_planned (sprt_max_games for the champion, which plays until decided,
    superseded or that many; else the comparison's target), tally (`tally` of those games, with the SPRT fields
    for kinds 'champion' and 'sprt'; refreshed with the status as games finish; null while idle), decision (the
    newest posterior `verdict` without its posterior, with next [[a, b], ...] lanes while pending; null before
    any), placements_played and placements_per_second (the session's placements after openings, including the
    games in flight, and per second), mean_placements (per finished game of the main lane in the session, null
    before any),
    settings (the effective EvaluationSettings), eval_share (the Pacer's share: 1 with --once), backlog (unrated
    checkpoint ids at the last step), eval_share_used (Pacer.used), vram (hexnet.vram()), error}. Match events,
    one per lane at the end of a session, carry the lane's games and placements, the session's seconds and
    worker_seconds (seconds times the lane's share of the session's placements)."""

    def __init__(self, run, config, settings, pacer):
        self.openings = dense_openings.Book(run, settings)
        missed = self.openings.reconcile(dense_openings.stamp(run))
        settings = replace(settings, opening_book=self.openings.digest())
        self.run, self.config, self.settings, self.pacer = Path(run), config, settings, pacer
        path = self.run/'league.json'
        self.league = json.loads(path.read_text()) if path.exists() else dict(champion=None, checkpoints=[])
        if self.league['checkpoints'] and (missed or 'matrix' not in self.league or self.league.get('ladder_top') != settings.fill_top):
            write_league(self.run, self.league, self.config, self.settings.fill_top)
        self.models, self.seal, self.written, self.fill_target, self.deciding, self.reviewed = {}, None, 0., None, None, False
        self.book, self.next, self.ids, self.shas = {}, {}, {}, {}
        self.status = dict(stage='idle', updated_at=None, comparison=None, pool=[], started_at=None, games_played=0,
                           games_planned=0, tally=None, decision=None, placements_played=0, mean_placements=None,
                           placements_per_second=None, settings=asdict(settings), eval_share=pacer.share, backlog=[],
                           eval_share_used=0., error=None)

    def publish(self, force=False, **fields):
        """Update the status; rewrite the file when forced or STATUS_SECONDS after the last write."""
        self.status.update(fields)
        if force or time.monotonic()-self.written >= STATUS_SECONDS:
            self.written = time.monotonic()
            write_json(self.run/'evaluator-status.json',
                       dict(self.status, updated_at=time.time(), eval_share_used=self.pacer.used(), vram=hexnet.vram()))
            self.point()

    def point(self):
        """Rewrite actor.json {checkpoint, reason, updated_at, vetoed} and log an 'actor_model' event when its
        checkpoint or vetoed list changes. The checkpoint is the newest complete checkpoint of config.learner.variant
        (reason 'newest'); once that checkpoint is demoted or the Elo interval (`tally`) of its games against the
        champion it met (`met`, else the current champion; games count once their colour pair is complete) lies entirely below
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
            interval = tally(self.games(newest, rival))['elo_interval']
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
        """Keep exactly the named checkpoints loaded (Seal once, lazily); games in flight hold their own models,
        so a dropped model lives until its last game finishes."""
        for name in names:
            if name == SEAL:
                self.seal = self.seal or Seal()
            elif name not in self.models:
                self.models[name] = load(self.run, self.config, source=(name, self.run/'checkpoints'/name/'ema.pt'))
        for name in [n for n in self.models if n not in names]:
            del self.models[name]

    def games(self, a, b):
        """The games of report a-vs-b: the session's copy when it is open, else the file's when it was played
        under the current protocol ([] without one or under another protocol)."""
        if (a, b) in self.book:
            return self.book[a, b]
        path = report_path(self.run, a, b)
        report = json.loads(path.read_text()) if path.exists() else None
        return report['games'] if report and same_protocol(report, self.settings) else []

    def open(self, a, b):
        """Load report a-vs-b for appending and number its next opening pair after the highest one it holds, so a
        restart resumes the pairing; the report keeps its id (dense_openings.report_id). A report played under another
        protocol is kept as report-<created_at>.json beside it (still pooled by load_reports) with an 'info' event, and
        a new one starts with a new id."""
        if (a, b) in self.book:
            return
        path = report_path(self.run, a, b)
        old = json.loads(path.read_text()) if path.exists() else None
        if old and not same_protocol(old, self.settings):
            kept = path.with_name(f'report-{int(old["created_at"])}.json')
            path.replace(kept)
            log_event(self.run, 'evaluator', 'info', f'{a} vs {b}: the report played under another protocol is kept as '
                      f'{kept.name}; a new one starts', candidate=a, opponent=b)
            old = None
        self.book[a, b] = list(old['games']) if old else []
        self.next[a, b] = max((g['pair'] for g in self.book[a, b]), default=-1)+1
        self.ids[a, b] = dense_openings.report_id(old) if old else uuid.uuid4().hex

    def start(self, pool, lane):
        """Start the next opening pair of lane (a, b, kind) in `pool`: both colours at once (`paired_games`)."""
        a, b, _ = lane
        pair, self.next[a, b] = self.next[a, b], self.next[a, b]+1
        pool.add(lane, paired_games(self.models[a], SEAL if b == SEAL else self.models[b], 2, a, self.config, self.settings,
                                    self.seal, self.openings, pair, candidate=a, opponent=b))

    def persist(self, a, b, kind, pair):
        """Append a completed colour pair to report a-vs-b and rewrite it (kind 'sprt' recomputes metrics.sprt,
        decision or 'max-games'), so a restart loses only the games in flight, then record it in the opening book; a
        game stopped by 'span' is logged."""
        records = self.book[a, b] = self.book[a, b]+pair
        for name in (a, b):
            if name != SEAL and name not in self.shas:
                self.shas[name] = digest(self.run/'checkpoints'/name/'ema.pt')
        report = make_report(a, b, records, self.shas, self.settings, self.ids[a, b])
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

    def session(self, want, planned):
        """Play the pool until want() asks for nothing and the games in flight have finished; returns {lane:
        games finished}. want() -> {(a, b, kind): games in flight wanted, even}, asked at the start and after every
        completed colour pair; a finished game's slot is refilled before the next engine step, and a lane want()
        drops (or shrinks) starts no new games while its running games finish and count. Games in flight never
        exceed pool_games; a lane's games (in flight or finished while their colour partner runs) never exceed
        the games it wants, nor per kind those wanted of that kind, so a draining lane's games count against its
        replacement's and a half-finished pair's slot is not refilled past a budget. The Pacer is charged for
        engine steps and for starting games (Seal plays its first turns then); no game starts while its credit
        is negative, and with nothing running the session then waits and asks want() again. Every completed
        pair is persisted at once."""
        pool, waiting, added = Pool(self.config.actor.leaf_batch), {}, {}
        placed = 0
        start, wall = self.pacer.clock(), time.time()
        lanes = want()
        shown = dict(lanes)

        def show(stage, force=False):
            if not force and time.monotonic()-self.written < STATUS_SECONDS:
                return  # the tally is computed only for a write
            main = next(iter(shown), None)
            a, b, kind = main or (None, None, None)
            done = self.games(a, b)+[r for group in waiting.get(main, {}).values() for r in group] if a else []
            live = placed+pool.moves()
            self.publish(True, stage=stage, comparison=dict(candidate=a, opponent=b, kind=kind) if a else None,
                         pool=[dict(candidate=x, opponent=y, kind=k, running=pool.running((x, y, k)), share=lanes.get((x, y, k), 0))
                               for x, y, k in shown], started_at=wall, games_played=len(done), games_planned=planned,
                         tally=tally(done, self.test if kind in ('champion', 'sprt') else None), placements_played=live,
                         mean_placements=added[main][1]/added[main][0] if main in added else None,
                         placements_per_second=live/max(self.pacer.clock()-start, 1e-9))
        while True:
            for a, b, _ in lanes:
                self.open(a, b)
            self.use(*dict.fromkeys(name for lane in lanes for name in lane[:2]))
            shown.update(lanes)
            ready = self.pacer.ready()
            if lanes and ready:
                kinds = {kind: sum(n for lane, n in lanes.items() if lane[2] == kind) for _, _, kind in lanes}
                halves = lambda match: sum(len(g) for held, groups in waiting.items() if match(held) for g in groups.values())
                held = lambda lane: pool.running(lane)+halves(lambda h: h == lane)
                kind_held = lambda kind: pool.running(kind=kind)+halves(lambda h: h[2] == kind)
                tick = self.pacer.clock()  # starting games plays Seal's first turns: playing time
                for lane, share in lanes.items():
                    while held(lane)+2 <= share and kind_held(lane[2])+2 <= kinds[lane[2]] \
                            and pool.running()+2 <= self.settings.pool_games:
                        self.start(pool, lane)
                self.pacer.played(tick, self.pacer.clock())
            if not pool.running():
                if lanes and not ready:
                    self.pacer.wait(lambda: show('throttled', True))
                    lanes = want()  # the wait may have outlasted the pairing (a newer checkpoint)
                    continue
                break
            show('playing')
            tick = self.pacer.clock()
            results = pool.step()
            self.pacer.played(tick, self.pacer.clock())
            paired = False
            for lane, record in results:
                moves = record['plies']-len(record['opening'])
                placed += moves
                count = added.setdefault(lane, [0, 0])
                count[0] += 1; count[1] += moves
                group = waiting.setdefault(lane, {}).setdefault(record['pair'], [])
                group.append(record)
                if len(group) == 2:
                    del waiting[lane][record['pair']]
                    self.persist(*lane, sorted(group, key=lambda r: r['challenger_color']))
                    paired = True
            if paired:
                lanes = want()
        show('playing', True)
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
        return any(e[0] != cid and e[0].split('/')[0] == cid.split('/')[0] and e[0] not in known for e in checkpoints(self.run))

    def entry(self, cid):
        return next((c for c in self.league['checkpoints'] if c['id'] == cid), None)

    def close(self, a, b):
        """Whether the league's current Elo of a and b (Seal: anchors.seal.elo) makes their pairing `informative`
        under max_expected_score; a side without an Elo counts as informative."""
        elo = lambda k: self.league.get('anchors', {}).get(SEAL, {}).get('elo') if k == SEAL else (self.entry(k) or {}).get('elo')
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
        """The posterior decision on candidate cid against the champion (module contract), from every
        protocol-matching report among the league's rated ids, cid and Seal: {decision ('promote', 'reject' or
        None), delta and delta_sd (r_cid - r_champion + their matchup deviation), p_better (P(delta >
        sprt_elo0)), pooled (95% interval of r_cid - r_champion without it), direct {games, elo, interval}
        (`tally` of their own games), disagree (the direct and pooled intervals do not overlap), spread (rating sd
        of cid and of the champion about the league mean), leader (the rated or candidate checkpoint of highest
        posterior rating)}, plus `posterior` (the Posterior) for pairing."""
        s = self.settings
        rated = [c['id'] for c in self.league['checkpoints'] if c.get('elo') is not None and not c.get('skipped')]
        ids = list(dict.fromkeys(rated+[champion, cid]))
        reports = [r for r in load_reports(self.run, s) if r['candidate'] in ids+[SEAL] and r['opponent'] in ids+[SEAL]]
        if any(SEAL in (r['candidate'], r['opponent']) for r in reports):
            ids.append(SEAL)
        post = Posterior(ids, ids[0], [(r['candidate'], r['opponent'], r['summary']['wins']+r['summary']['capped']/2,
                                       r['summary']['games']) for r in reports], s.matchup_prior_elo)
        mean, sd = post.difference(cid, champion)
        pooled_mean, pooled_sd = post.difference(cid, champion, False)
        pooled = [pooled_mean-1.96*pooled_sd, pooled_mean+1.96*pooled_sd]
        t = tally(self.games(cid, champion))
        interval = t['elo_interval']
        disagree = bool(interval) and (interval[1] < pooled[0] or interval[0] > pooled[1])
        spread = [post.spread(cid), post.spread(champion)]
        leader = max((i for i in ids if i != SEAL and not (self.entry(i) or {}).get('demoted')), key=post.rating)
        p_better = .5*math.erfc((s.sprt_elo0-mean)/(max(sd, 1e-9)*math.sqrt(2)))
        ready = t['games'] >= s.sprt_min_games and spread[0] <= spread[1]*s.uncertainty_parity and not disagree
        decision = 'promote' if ready and leader == cid and p_better >= s.promote_confidence else \
            'reject' if ready and p_better <= 1-s.promote_confidence else None
        return dict(decision=decision, delta=mean, delta_sd=sd, p_better=p_better, pooled=pooled,
                    direct=dict(games=t['games'], elo=t['elo_delta'], interval=interval), disagree=disagree,
                    spread=spread, leader=leader, posterior=post)

    def evidence(self, verdict, cid, champion, games):
        """(a, b) of the evidence pairing for a pending posterior decision, or None: of cid and the champion each
        vs the previous champion (`met` of the champion) and vs Seal, the one whose `games` most reduce the
        posterior variance of delta (value of information, `Posterior.after`), when that beats as many direct
        games; only pairings whose report can grow (`rematch_pair`) and that are `close`."""
        s, post, previous = self.settings, verdict['posterior'], self.met(champion)
        options = []
        for a, b in ((cid, previous), (champion, previous), (cid, SEAL), (champion, SEAL)):
            if not b or b == a or not self.close(a, b):
                continue
            if b == SEAL:
                path = report_path(self.run, a, SEAL)
                if not path.exists() or same_protocol(json.loads(path.read_text()), s):
                    options.append((a, SEAL))
            elif pair := rematch_pair(self.run, a, b, s):
                options.append(pair)
        known = lambda x: x in post.index or x == post.anchor
        after = lambda pair: post.after((cid, champion, True), pair, games)
        options = [o for o in options if known(o[0]) and known(o[1])]
        best = min(options, key=after, default=None)
        return best if best and after(best) < after((cid, champion)) else None

    def lanes(self, verdict, cid, champion):
        """The pool's lanes for a pending posterior decision: direct games of cid vs the champion fill the pool
        (never beyond sprt_max_games direct games) until sprt_min_games of them are complete and while the direct
        and pooled estimates disagree; after that an `evidence` pairing that beats direct games gets
        evidence_share of the pool and the direct games the rest."""
        s, direct = self.settings, verdict['direct']['games']
        room = even(min(s.pool_games, s.sprt_max_games-direct))
        share = even(s.pool_games*s.evidence_share)
        other = self.evidence(verdict, cid, champion, share) if share and direct >= s.sprt_min_games and not verdict['disagree'] else None
        lanes = {(cid, champion, 'champion'): min(room, s.pool_games-share) if other else room}
        if other:
            lanes[(*other, 'evidence')] = share
        return {lane: n for lane, n in lanes.items() if n >= 2}

    def decide(self, cid, champion):
        """Posterior mode: a session whose lanes (`lanes`) follow the verdict after every completed colour pair
        until `verdict` decides, sprt_max_games direct games are complete or a newer checkpoint of cid's variant
        exists; the games in flight then finish and count (play resumes when they leave the verdict undecided)
        and the final verdict settles it: superseded, cid is
        promoted when p_better >= promote_confidence (settled true), readiness aside. Returns ({opponent: report of
        cid against it}, verdict with decision 'promote', 'reject', 'max-games' or 'superseded'), or ({}, None)
        without a direct game. The verdict (`public`) is published as status decision, stored as the direct
        report's metrics.posterior and logged as a 'decision' event."""
        s = self.settings

        def want():
            verdict = self.verdict(cid, champion)
            if verdict['decision'] or verdict['direct']['games'] >= s.sprt_max_games or self.newer(cid):
                return {}
            lanes = self.lanes(verdict, cid, champion)
            self.publish(decision=dict(public(verdict), candidate=cid, next=[list(l[:2]) for l in lanes]))
            return lanes
        while True:  # the games in flight can undo a verdict that stopped the session: then play on
            added = self.session(want, s.sprt_max_games)
            for (a, b, _), games in added.items():
                if a != cid and games:
                    self.record(a, b, json.loads(report_path(self.run, a, b).read_text()))
            verdict = self.verdict(cid, champion)
            if verdict['decision'] or not added or verdict['direct']['games'] >= s.sprt_max_games or self.newer(cid):
                break
        if not verdict['direct']['games']:
            return {}, None
        superseded = self.newer(cid)
        verdict = dict(public(verdict), candidate=cid)
        if not verdict['decision'] and superseded and verdict['p_better'] >= s.promote_confidence:
            verdict.update(decision='promote', settled=True)
        verdict['decision'] = verdict['decision'] or ('superseded' if superseded else 'max-games')
        path = report_path(self.run, cid, champion)
        report = json.loads(path.read_text())
        report['metrics']['posterior'] = verdict
        write_json(path, report)
        self.publish(True, decision=verdict)
        direct, g = verdict['direct'], lambda v: f'{v:+.0f}' if v is not None else '-'
        log_event(self.run, 'evaluator', 'decision', f'{cid} vs {champion}: {verdict["decision"]}'
                  + (' (settled on supersession)' if superseded else '') + f' after {direct["games"]} direct games: '
                  f'P(delta > {s.sprt_elo0:g}) {verdict["p_better"]:.3f}, delta {g(verdict["delta"])} +- {verdict["delta_sd"]:.0f}, '
                  f'direct {g(direct["elo"])}, pooled [{verdict["pooled"][0]:+.0f}, {verdict["pooled"][1]:+.0f}]',
                  opponent=champion, **verdict)
        found = {r['opponent']: r for r in load_reports(self.run, s) if r['candidate'] == cid}
        return {champion: found[champion], **found}, verdict

    def review(self):
        """Posterior mode, once per evaluator (so after every restart or settings change): re-apply the promotion
        rule to the existing reports. Of the rated checkpoints, neither skipped nor demoted, with at least
        sprt_min_games direct games against the champion (their report against it), those whose `verdict`
        meets uncertainty parity, P(delta > sprt_elo0) >= promote_confidence and agreeing direct and pooled
        intervals are eligible; the one of highest posterior rating among them is promoted ('decision' event
        'promote on review', then the 'promotion' event; its Seal anchor is scheduled as for any promotion). A
        higher-rated checkpoint without those direct games does not block it: it has not met the champion."""
        s, champion = self.settings, self.league['champion']
        if s.decision != 'posterior' or champion is None:
            return
        eligible = []
        for c in self.league['checkpoints']:
            if c['id'] == champion or c.get('elo') is None or c.get('skipped') or c.get('demoted') \
                    or len(self.games(c['id'], champion)) < s.sprt_min_games:
                continue
            verdict = self.verdict(c['id'], champion)
            if verdict['spread'][0] <= verdict['spread'][1]*s.uncertainty_parity and not verdict['disagree'] \
                    and verdict['p_better'] >= s.promote_confidence:
                eligible.append((verdict['posterior'].rating(c['id']), c['id'], verdict))
        if not eligible:
            return
        _, cid, verdict = max(eligible)
        verdict = dict(public(verdict), candidate=cid, decision='promote', review=True)
        log_event(self.run, 'evaluator', 'decision', f'{cid} vs {champion}: promote on review of the existing reports '
                  f'({verdict["direct"]["games"]} direct games, P(delta > {s.sprt_elo0:g}) {verdict["p_better"]:.3f})',
                  opponent=champion, **verdict)
        self.promote(cid, champion)
        write_league(self.run, self.league, self.config, self.settings.fill_top)

    def sequential(self, cid, champion):
        """SPRT mode: a session of cid vs the champion until the SPRT decides, sprt_max_games games are complete
        or a newer checkpoint of cid's variant exists (the games in flight finish and count; a bound crossed
        before them stays the decision). Returns ({champion:
        report}, metrics.sprt) with decision 'H1', 'H0', 'max-games' or 'superseded', settled as the module
        contract states ('settle' event: promoted when the posterior p_better of `verdict` is at least
        promote_confidence), or ({}, None) without a game."""
        s, decided = self.settings, []

        def want():
            games = self.games(cid, champion)
            if games and (decision := self.test(games)['decision']):
                decided.append(decision)
            if decided or len(games) >= s.sprt_max_games or self.newer(cid):
                return {}
            return {(cid, champion, 'champion'): even(min(s.pool_games, s.sprt_max_games-len(games)))}
        self.session(want, s.sprt_max_games)
        path = report_path(self.run, cid, champion)
        if not self.games(cid, champion):  # no game under the active protocol
            return {}, None
        report = json.loads(path.read_text())
        result, n = self.test(report['games']), len(report['games'])
        decision = decided[0] if decided else 'superseded' if n < s.sprt_max_games and self.newer(cid) else 'max-games'
        test = report['metrics']['sprt'] = dict(result, decision=decision)
        summary_ = report['summary']
        if test['decision'] == 'superseded':
            p_better = self.verdict(cid, champion)['p_better']
            settled = test['settled'] = dict(pair_score=summary_['pair_score'], p_better=p_better,
                                             promote=p_better >= s.promote_confidence)
            log_event(self.run, 'evaluator', 'settle', f'{cid} vs {champion} settled on supersession after {n} games: '
                      f'+{summary_["wins"]} -{summary_["losses"]} ={summary_["capped"]}, pair score {summary_["pair_score"]:.3f}, '
                      f'P(delta > {s.sprt_elo0:g}) {p_better:.3f}, LLR {test["llr"]:.2f}: '
                      + ('promoted' if settled['promote'] else 'not promoted'),
                      candidate=cid, opponent=champion, games=n, llr=test['llr'], **settled)
        write_json(path, report)
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
        path = report_path(self.run, cid, SEAL)
        return json.loads(path.read_text()) if path.is_file() else None

    def seal_reports(self, cid):
        """cid's reports against Seal under every protocol: its report.json and those archived beside it."""
        return [r for r in load_reports(self.run) if r['candidate'] == cid and r['opponent'] == SEAL]

    def crown(self, cid):
        """Make cid champion and start its reign: reign_from and reign_games (league contract)."""
        self.league.update(champion=cid, reign_from=len(self.league['checkpoints']),
                           reign_games=sum(len(r['games']) for r in self.seal_reports(cid)))

    def anchor(self):
        """(champion entry, SEAL, 'anchor', games left) while the current champion owes Seal games, else None. In
        its current reign it owes anchor_games once (anchor_on_promotion) and anchor_games more per `anchor_every`
        checkpoints rated during the reign (entries from `reign_from` on), counted against the games its Seal reports
        (`seal_reports`, every protocol) gained since reign_games, so an anchor owed when the protocol changes (a book
        refresh) is played under the new one. A newer champion supersedes the old one's unfinished anchor."""
        s, champion = self.settings, self.entry(self.league['champion'])
        if not s.anchor_games or champion is None:
            return None
        entries = self.league['checkpoints']
        later = sum(not c.get('skipped') for c in entries[self.league.get('reign_from', entries.index(champion)+1):])
        played = sum(len(r['games']) for r in self.seal_reports(champion['id']))-self.league.get('reign_games', 0)
        left = s.anchor_games*(s.anchor_on_promotion+later//s.anchor_every)-played
        return (champion, SEAL, 'anchor', left) if left > 0 else None

    def optional(self):
        """(league entry of the candidate side, opponent, kind, games) of the first optional comparison, else None:
        the current champion's panel ('panel', 'incumbent'), idle rematches (`rematches`, with idle_rematch),
        the panels of the variant heads, newest first, then the newest rated checkpoint missing its
        previous-checkpoint comparison (when previous_games > 0 and the two are `close`)."""
        s = self.settings
        champion = [c for c in self.league['checkpoints'] if c['id'] == self.league['champion']]
        for a, b, kind, games in [n for c in champion for n in self.needs(c)] + (self.rematches() if s.idle_rematch else []) \
                + [n for c in self.heads() for n in self.needs(c)]:
            return self.entry(a), b, kind, games
        rated = [c for c in self.league['checkpoints'] if not c.get('skipped')]
        for index in reversed(range(len(rated))):
            c = rated[index]
            earlier = [p for p in rated[:index] if p['variant'] == c['variant'] and p['step'] < c['step']]
            if earlier and s.previous_games and (previous := max(earlier, key=lambda p: p['step'])['id']) \
                    not in {m['opponent'] for m in c['matches']} and self.close(c['id'], previous):
                return c, previous, 'previous', s.previous_games
        return None

    def met(self, cid):
        """The champion that checkpoint cid met when it was rated (its panel incumbent, else its first match's
        opponent), or None."""
        entry = self.entry(cid) or {}
        return entry.get('panel', {}).get('incumbent') or next((m['opponent'] for m in entry.get('matches', [])), None)

    def fill(self):
        """(league entry, opponent, kind, games) of the next fill work with idle_fill, else None; only `close`
        pairings, `games` each. (1) The champion vs Seal ('fill') while anchor_target_halfwidth > 0 and the
        half-width of the 95% interval of their Elo difference (`rate` over the champion's Seal reports alone,
        `seal_reports`) exceeds it (no report yet counts as wide). (2) Once, the
        newest rated, not demoted checkpoint vs the previous champion ('generalization'): the champion that the
        champion it met had met (`met` twice), when the two have no games yet. (3) The league ladder pair with
        the widest interval whose report can grow (`rematch_pair`, 'fill')."""
        s, champion = self.settings, self.entry(self.league['champion'])
        if not s.idle_fill or champion is None:
            return None
        reports = self.seal_reports(champion['id'])
        if s.anchor_target_halfwidth > 0 and self.close(champion['id'], SEAL):
            low, high = rate([champion['id'], SEAL], champion['id'], reports, seed=self.config.seed)[1][SEAL] if reports \
                else (-math.inf, math.inf)
            if (high-low)/2 > s.anchor_target_halfwidth:
                return champion, SEAL, 'fill', s.games
        newest = [c for c in self.league['checkpoints'] if c.get('elo') is not None and not c.get('demoted')][-1:]
        for c in newest:
            previous = self.met(self.met(c['id']))
            if previous and previous != c['id'] and (self.entry(previous) or {}).get('elo') is not None \
                    and previous not in self.league.get('matrix', {}).get(c['id'], {}) and self.close(c['id'], previous):
                return c, previous, 'generalization', s.games
        for d in sorted(self.league.get('ladder', []), key=lambda d: d['interval'][0]-d['interval'][1]):
            if self.close(d['a'], d['b']) and (pair := rematch_pair(self.run, d['a'], d['b'], s)):
                return self.entry(pair[0]), pair[1], 'fill', s.games
        return None

    def filling(self, target):
        """Log a 'fill' event when fill work starts, changes target ('seal', '<a> vs <b>' or 'generalization <a> vs
        <b>') or ends (None)."""
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
        book runs (`refresh_openings`; never while such a candidate waits, as it would restart its games). Then, on
        the first step, the promotion
        rule is re-applied to the existing reports (`review`). Then it rates the newest unrated checkpoint of the
        variant whose newest unrated checkpoint is oldest, skipping that variant's older unrated checkpoints (none
        of them has games against the champion), or else plays a session of the champion's Seal anchor
        (`anchor`), else of an optional comparison, else of fill work (`fill`), each until its games are complete
        or a checkpoint waits (the games in flight then finish and count)."""
        self.settle()
        champion = self.league['champion']
        revived = [c for c in self.league['checkpoints'] if c.get('skipped') and champion and self.games(c['id'], champion)]
        for c in revived:
            self.league['checkpoints'].remove(c)
            log_event(self.run, 'evaluator', 'info', f'{c["id"]} was skipped with {len(self.games(c["id"], champion))} games '
                      f'against {champion}; it is rated on them', candidate=c['id'])
        if revived:
            write_league(self.run, self.league, self.config, self.settings.fill_top)
        known = {c['id'] for c in self.league['checkpoints']}
        unrated = [e for e in checkpoints(self.run) if e[0] not in known]
        self.status['backlog'] = [e[0] for e in unrated]
        resumed = [e for e in unrated if champion and self.games(e[0], champion)]
        if not resumed:
            self.refresh_openings()
        if resumed:
            self.filling(None)
            self.rate(resumed[0])
            return True
        if not self.reviewed:
            self.reviewed = True
            self.review()
        if unrated:
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
            return True
        task = self.anchor() or self.optional() or self.fill()
        self.filling(None if task is None or task[2] not in ('fill', 'generalization') else 'seal' if task[1] == SEAL
                     else f'{"generalization " if task[2] == "generalization" else ""}{task[0]["id"]} vs {task[1]}')
        if task is None:
            return False
        entry, opponent, kind, games = task
        a, s = entry['id'], self.settings
        target, decided = games if kind == 'previous' else len(self.games(a, opponent))+games, []

        def want():
            done = self.games(a, opponent)
            if kind == 'sprt' and done and (decision := self.test(done)['decision']):
                decided.append(decision)  # an idle SPRT rematch keeps the first bound it crosses
            if self.backlog() or len(done) >= target or decided:
                return {}
            return {(a, opponent, kind): even(min(s.pool_games, target-len(done)))}
        self.session(want, target)
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
            self.settle()
        return True


def loop(args):
    run = Path(args.run)
    config = dense_config.load(run)
    settings = dense_config.override(config.evaluation, args, 'eval_')
    if any(getattr(settings, name) % 2 for name in ('games', 'previous_games', 'anchor_games', 'sprt_max_games', 'pool_games', 'sprt_min_games')):
        raise ValueError('Evaluation game counts must be even: every opening is played with both colours')
    if min(settings.games, settings.sprt_max_games, settings.pool_games) < 2 or any(0 < getattr(settings, name) < 2 or getattr(settings, name) < 0
                                                                for name in ('previous_games', 'anchor_games')):
        raise ValueError('games, sprt_max_games and pool_games need at least one opening pair; optional totals are 0 or at least 2')
    if not 0 <= settings.evidence_share < 1:
        raise ValueError('evidence_share must be in [0, 1)')
    if settings.decision not in ('posterior', 'sprt') or not .5 < settings.promote_confidence < 1 \
            or settings.uncertainty_parity <= 0 or settings.matchup_prior_elo < 0 or settings.sprt_min_games < 2:
        raise ValueError("decision is 'posterior' or 'sprt', promote_confidence in (0.5, 1), uncertainty_parity > 0, "
                         'matchup_prior_elo >= 0 and sprt_min_games at least one opening pair')
    if settings.anchor_games and settings.anchor_every < 1:
        raise ValueError('anchor_every must be at least 1 while anchor games are enabled')
    if settings.anchor_target_halfwidth < 0 or settings.fill_top < 0 or not .5 <= settings.max_expected_score <= 1:
        raise ValueError('anchor_target_halfwidth and fill_top must be at least 0, max_expected_score in [0.5, 1]')
    dense_openings.check(settings)
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


def calibrate(args):
    """Continue the newest capped games from their cap with the champion (both sides, `--sims`) for up to
    --extra plies, then score each original ply's TD(lambda) target (dense_data.value_targets), the raw root
    value and the masked baseline (p = 1/2) against the realised result, overall and per continuation length."""
    run = Path(args.run)
    config = dense_config.load(run)
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
    book = dense_openings.Book(run, settings)
    settings = replace(settings, opening_book=book.digest())
    started = time.perf_counter()
    records = play(paired_games(models[args.a], SEAL if args.b == SEAL else models[args.b], args.games,
                                f'match/{args.a}/{args.b}', config, settings, Seal() if args.b == SEAL else None, book,
                                candidate=args.a, opponent=args.b), config.actor.leaf_batch)
    report = make_report(args.a, args.b, records, {n: m.sha for n, m in models.items()}, settings)
    print(json.dumps(dict(summary=report['summary'], metrics=report['metrics'], seconds=time.perf_counter()-started), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('loop'); p.add_argument('--run', required=True); p.add_argument('--once', action='store_true')
    p.add_argument('--poll', type=float, default=30.)
    dense_config.add_arguments(p.add_argument_group('evaluation overrides for this process'), dense_config.EvaluationSettings, 'eval_')
    p = sub.add_parser('calibrate'); p.add_argument('--run', required=True)
    p.add_argument('--lambdas', type=float, nargs='+', default=[.5, .7, .9, 1.])
    p.add_argument('--games', type=int, default=100); p.add_argument('--sims', type=int, default=64)
    p.add_argument('--extra', type=int, default=256)
    p = sub.add_parser('match'); p.add_argument('--run', required=True); p.add_argument('--a', required=True)
    p.add_argument('--b', required=True); p.add_argument('--games', type=int, default=32); p.add_argument('--sims', type=int)
    args = parser.parse_args()
    dict(loop=loop, calibrate=calibrate, match=match)[args.command](args)


if __name__ == '__main__':
    main()
