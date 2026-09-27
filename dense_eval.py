"""Dense checkpoint evaluator: paired league games, Bradley-Terry ratings, champion promotion, calibration.

Run layout: dense_config. Subcommands
  loop       rate every complete checkpoint (dense_selfplay.checkpoints order) once: colour-swapped opening
             pairs vs the previous checkpoint of its variant and vs the champion, every `anchor_every`-th
             evaluated checkpoint also vs Seal; writes evaluations/<a>-vs-<b>/report.json (skipped when present),
             league.json, champion.json on promotion, and events.
  calibrate  continue capped self-play games with the champion and score TD(lambda) value targets against
             the realised results.
  match      ad hoc paired match between two checkpoints (run ids or paths) or a checkpoint and Seal.

Scoring: a capped game (at the ply limit, or reason 'span' when a searched position does not fit the largest crop) is
half a point for each side. Pair score = candidate points / 2 over its two games.
The comparison with the previous checkpoint of the variant plays `games` games. The comparison with the
champion is a sequential test (see `sprt`) played in rounds of `games` games until it accepts H0 or H1 or
reaches `sprt_max_games`; only H1 promotes. The first evaluated checkpoint becomes champion unopposed.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

import dense_config
from dense_config import log_event
import dense_data
from arena import Seal
from dense_selfplay import Engine, checkpoints, load
from hexo import Game
from klent import digest
from train import paired_metrics, task_opening, write_json

SEAL = 'seal'
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


def play(games, leaf_batch):
    """Run MatchGames to completion in one engine; returns their records in input order."""
    engine, records = Engine(leaf_batch), {}
    for game in games:
        if game.over():
            records[id(game)] = game.finish()
        else:
            engine.add(game)
    while engine.slots:
        for game in engine.step():
            records[id(game)] = game.finish()
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


def make_report(candidate, opponent, records, models, settings):
    return dict(candidate=candidate, opponent=opponent, created_at=time.time(),
                candidate_sha256=models[candidate].sha, opponent_sha256=SEAL if opponent == SEAL else models[opponent].sha,
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
    """Recompute ratings from every report among rated ids and Seal, then publish league.json."""
    run = Path(run)
    ids = [c['id'] for c in league['checkpoints']]
    reports = [json.loads(p.read_text()) for p in sorted((run/'evaluations').glob('*/report.json'))]
    reports = [r for r in reports if r['candidate'] in ids and (r['opponent'] in ids or r['opponent'] == SEAL)]
    seal_games = sum(len(r['games']) for r in reports if r['opponent'] == SEAL)
    names = ids+([SEAL] if seal_games else [])
    point, intervals, draws = rate(names, ids[0], reports, seed=config.seed) if ids else ({}, {}, {})
    for c in league['checkpoints']:
        c['elo'], c['elo_interval'] = point[c['id']], intervals.get(c['id'])
    # Latest rated checkpoint of each variant; a-b intervals come from the same joint draws.
    latest = {}
    for c in league['checkpoints']:
        if point[c['id']] is not None and c['step'] >= latest.get(c['variant'], c)['step']:
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


def evaluate_checkpoint(run, config, league, entry, models, seal):
    """Play (or reuse) this checkpoint's comparisons, append it to the league and decide promotion.
    All missing comparisons play their first round together; further champion rounds follow until the SPRT
    decides or `sprt_max_games` is reached (then decision 'max-games', no promotion)."""
    cid, path, manifest = entry
    settings = config.evaluation
    variant, step = cid.split('/')
    rated = {c['id']: c for c in league['checkpoints']}
    previous = [c for c in rated.values() if c['variant'] == variant and c['step'] < int(step)]
    previous = max(previous, key=lambda c: c['step'])['id'] if previous else None
    champion = league['champion']
    opponents = [o for o in dict.fromkeys([previous, champion]) if o is not None]
    if len(rated) % settings.anchor_every == 0:
        opponents.append(SEAL)

    def model(name):
        if name not in models:
            models[name] = load(run, config, source=(name, Path(run)/'checkpoints'/name/'ema.pt'))
        return models[name]

    test = lambda records: sprt(records, settings.sprt_elo0, settings.sprt_elo1, settings.sprt_alpha, settings.sprt_beta)
    reports, records, todo = {}, {}, {}
    for opponent in opponents:
        if report_path(run, cid, opponent).exists():
            reports[opponent] = json.loads(report_path(run, cid, opponent).read_text())
        else:
            records[opponent] = []
            todo[opponent] = settings.anchor_games if opponent == SEAL else \
                min(settings.games, settings.sprt_max_games) if opponent == champion else settings.games
    started = time.perf_counter()
    while todo:
        games = {o: paired_games(model(cid), SEAL if o == SEAL else model(o), n, cid, config, settings, seal,
                                 len(records[o])//2, candidate=cid, opponent=o) for o, n in todo.items()}
        results = play([g for items in games.values() for g in items], config.actor.leaf_batch)
        for record in results:
            if record['reason'] == 'span':
                log_event(run, 'evaluator', 'error', f'{cid} vs {record["opponent"]} pair {record["pair"]}: game counted as '
                          f'capped at ply {record["plies"]}, a searched position spans more than the largest crop',
                          candidate=cid, opponent=record['opponent'])
        for o, items in games.items():
            records[o] += results[:len(items)]; results = results[len(items):]
        todo = {}
        if champion in records and test(records[champion])['decision'] is None and len(records[champion]) < settings.sprt_max_games:
            todo[champion] = min(settings.games, settings.sprt_max_games-len(records[champion]))
    for opponent, items in records.items():
        report = make_report(cid, opponent, items, models, settings)
        if opponent == champion:
            result = test(items)
            report['metrics']['sprt'] = dict(result, decision=result['decision'] or 'max-games')
        report_path(run, cid, opponent).parent.mkdir(parents=True, exist_ok=True)
        write_json(report_path(run, cid, opponent), report)
        reports[opponent] = report
        s = report['summary']
        log_event(run, 'evaluator', 'match', f'{cid} vs {opponent}: +{s["wins"]} -{s["losses"]} ={s["capped"]}'
                  + (f' (SPRT {report["metrics"]["sprt"]["decision"]}, LLR {report["metrics"]["sprt"]["llr"]:.2f})' if opponent == champion else ''),
                  candidate=cid, opponent=opponent, **s, sprt=report['metrics'].get('sprt'), seconds=time.perf_counter()-started)
    promoted = champion is None or (champion in reports and reports[champion]['metrics']['sprt']['decision'] == 'H1')
    league['checkpoints'].append(dict(
        id=cid, variant=variant, step=int(step), ema_sha256=digest(path/'ema.pt'), elo=None, elo_interval=None,
        matches=[dict(opponent=o, wins=r['summary']['wins'], losses=r['summary']['losses'], capped=r['summary']['capped'],
                      games=r['summary']['games'], opening_pair_p=r['metrics']['opening_pair_p'],
                      elo_delta=r['summary']['elo_delta'], pair_score_lower=r['summary']['pair_score_lower'])
                 for o, r in reports.items()]))
    if promoted:
        league['champion'] = cid
        write_json(Path(run)/'champion.json', dict(checkpoint=cid, ema_sha256=digest(path/'ema.pt'), updated_at=time.time()))
        log_event(run, 'evaluator', 'promotion', f'{cid} promoted to champion' + (f' over {champion}' if champion else ''),
                  checkpoint=cid, previous_champion=champion)
    write_league(run, league, config)
    print(f'{cid}: ' + ', '.join(f'vs {o} +{r["summary"]["wins"]} -{r["summary"]["losses"]} ={r["summary"]["capped"]}'
                                  for o, r in reports.items()) + (' -> champion' if promoted else ''), flush=True)


def loop(args):
    run = Path(args.run)
    config = dense_config.load(run)
    seal = Seal()
    while True:
        league_path = run/'league.json'
        league = json.loads(league_path.read_text()) if league_path.exists() else dict(champion=None, checkpoints=[])
        rated = {c['id'] for c in league['checkpoints']}
        models = {}
        for entry in checkpoints(run):
            if entry[0] not in rated:
                try:
                    evaluate_checkpoint(run, config, league, entry, models, seal)
                except BaseException as error:
                    log_event(run, 'evaluator', 'error', f'{entry[0]}: {type(error).__name__}: {error}', checkpoint=entry[0])
                    raise
        if args.once:
            break
        time.sleep(args.poll)


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
    report = make_report(args.a, args.b, records, models, settings)
    print(json.dumps(dict(summary=report['summary'], metrics=report['metrics'], seconds=time.perf_counter()-started), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('loop'); p.add_argument('--run', required=True); p.add_argument('--once', action='store_true')
    p.add_argument('--poll', type=float, default=30.)
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
