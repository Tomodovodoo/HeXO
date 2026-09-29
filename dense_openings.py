"""Opening books of the dense evaluator: a DAG of symmetry-reduced positions with paired colour statistics per node.

Nodes are positions after `depth` placements from the origin stone, reduced by the 12 hex symmetries about the origin
(`canonical`: key and a legal play order `moves` of the least image); edges are placements (`parents`: the positions
one stone of the last turn earlier). An opening is a node with status 'opening'; openings sit at any depth from
book_min_plies to book_plies and games start from their `moves`. Every completed colour pair is recorded on every node
its opening passed through (the canonical prefixes of the played moves, `Book.record`), so a node's statistics
aggregate the games of its subtree along the paths played.

Two kinds share one format and one code path (`Book`):
  live    opening_suite 'book', <run>/openings.json. `Book.refresh` (at every champion change and every
          book_refresh_hours) re-scores every opening under the champion, retires nested parents, the implausible and
          the skewed ones, settles contests, lets the champion challenge a random book_revisit_fraction of the settled
          openings with an alternative at the same depth, and replaces retired openings until book_size are settled.
  frozen  any other suite: the repo's openings/<suite>.json, drawn in proportion to each opening's `weight` and never
          refreshed; its statistics are kept in <run>/openings-<suite>.json. openings/standard-v1.json is
          train.opening_for's evaluation suite.

File {schema, suite, frozen, refreshed_by, refreshed_at, counted, imported, weighting (live), nodes: [node]}, nodes
ordered by depth then key; counted maps `report_id` to the pairs of that report the book has counted
(`Book.reconcile`) and weighting is the live book's draw rule (EvaluationSettings.book_weighting).
node {key, moves, depth, weight (frozen openings), status, reason, challenges, probability, visit_share, checkpoint,
      created_at, retired_at, champion_probability, champion_value, scored_by, games, p1_wins, p2_wins, capped,
      plies_sum, plies_games, mean_plies, pairs, skew}:
  status        'opening' (drawn), 'retired' (was an opening; `reason` 'probability', 'skew', 'short_skew', 'nested' or
                'replaced') or null (a prefix, or a position imported from reports that never was an opening).
  challenges    the key of the opening an open challenger competes with (null when settled).
  probability   `reach` policy probability of the node's class under `checkpoint`, the champion that made it an
                opening, and visit_share the product of the search visit shares its generator sampled (`generate`);
                champion_probability and champion_value (the value head's P1 expected score) under `scored_by`, the
                champion of the newest refresh; null until scored.
  games, p1_wins, p2_wins, capped  games through the node; P1 is player 0, who places the origin stone.
  mean_plies    mean final game length through the node, from plies_sum / plies_games.
  pairs         colour-swapped pairs through the node by P1's points: counts of 0, 1/2, 1, 3/2 and 2.
  skew          `skew` of pairs: P1's advantage in Elo with its 95% interval.
The openings change only at a refresh; `Book.digest` names that state for the match protocol.
"""
import argparse
from dataclasses import replace
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np

import dense_config
import hexcrop
from train import write_json

SCHEMA = 'hexo-opening-book-v2'
LIVE = 'book'
FROZEN = Path(__file__).resolve().parent/'openings'
ORIGIN = (0, 0)
PAIR_SCORES = np.arange(5)/4  # P1's pair score (points over the pair / 2) of each `pairs` category
ALTERNATIVES = 4    # lines sampled per generation start; the plausible one the value head finds most balanced wins
GROW_ROUNDS = 8     # generation rounds per refresh while settled openings are missing
SCORE_BATCH = 512   # positions per policy call
SKEW_EDGES = list(range(-200, 201, 25))  # histogram bins of `Book.stats`, the outer bins open-ended
MAX_PLIES = 10      # longest book line: P2's sixth stone is the 11th placement, so no book position is terminal
REASONS = ('probability', 'skew', 'short_skew', 'nested', 'replaced')


def turns(moves):
    """The placements of `moves` grouped by turn: the first placement alone, then pairs (the last may be partial)."""
    return [moves[:1]]+[moves[k:k+2] for k in range(1, len(moves), 2)]


def canonical(moves):
    """(key, representative) of a position's class under the 12 hex symmetries about the origin. Per symmetry the
    image's turns as sorted stone tuples; key is the least of them as text 'q,r q,r|...' (turns joined by '|') and
    the representative is the image of `moves` under the first symmetry attaining it, in the play order of `moves`
    (a legal order stays legal)."""
    return _canonical(tuple(tuple(int(v) for v in m) for m in moves))


@lru_cache(maxsize=1 << 16)
def _canonical(moves):
    images = [np.asarray(moves, np.int64).reshape(-1, 2) @ m for m in hexcrop.SYMMETRIES]
    forms = [tuple(tuple(sorted(map(tuple, turn.tolist()))) for turn in turns(image)) for image in images]
    k = min(range(len(forms)), key=forms.__getitem__)
    return '|'.join(' '.join(f'{q},{r}' for q, r in turn) for turn in forms[k]), images[k].tolist()


def parents(moves):
    """Keys of the positions one stone of the last turn before `moves` (none for the origin alone)."""
    if len(moves) < 2:
        return []
    last = turns(list(moves))[-1]
    head = list(moves[:len(moves)-len(last)])
    return sorted({canonical(head+last[:i]+last[i+1:])[0] for i in range(len(last))})


def images(moves):
    """The distinct positions the 12 symmetries map `moves` to (turns as stone sets), each as the image of `moves` in
    its play order."""
    points = np.asarray(moves, np.int64).reshape(-1, 2)
    out = {}
    for m in hexcrop.SYMMETRIES:
        image = [tuple(p) for p in (points @ m).tolist()]
        out.setdefault(tuple(frozenset(turn) for turn in turns(image)), image)
    return list(out.values())

def policy(model, histories):
    """[({(q, r): probability}, P1 value)] of `model` (a dense_selfplay.Model) at each history: its policy and its
    value head's expected score for player 0."""
    out = []
    for start in range(0, len(histories), SCORE_BATCH):
        chunk = [np.asarray(h, np.int64).reshape(-1, 2) for h in histories[start:start+SCORE_BATCH]]
        for h, (actions, logits, q) in zip(chunk, model.evaluator.evaluate(chunk)):
            p = np.exp(logits-logits.max())
            mover = (len(h)+1)//2 % 2
            out.append((dict(zip(map(tuple, actions.tolist()), p/p.sum())), float((1+q[0]*(1-2*mover))/2)))
    return out


def reach(model, positions):
    """Per position (a move list), (probability, P1 value): the probability that `model`'s policy plays into its class,
    summed over the class's distinct symmetric `images`, and the value head's P1 expected score averaged over them.
    An image's probability is a product over turns of the turn's probability summed over its (at most two) placement
    orders (the origin placement has probability 1). Summing per turn is exact because a finished turn's order is not
    part of the network's input (hexcrop planes), so later turns do not depend on it; each turn is evaluated after
    the earlier turns in their stored order."""
    steps = []  # per position, per image: (image, per turn: [(history, placement) per order of the turn])
    for m in positions:
        per_image = []
        for image in images(m):
            per, played = [], 1
            for turn in turns(image)[1:]:
                head = image[:played]
                per.append([[(tuple(head+list(order[:i])), order[i]) for i in range(len(order))]
                            for order in itertools.permutations(turn)])
                played += len(turn)
            per_image.append((tuple(image), per))
        steps.append(per_image)
    histories = list(dict.fromkeys([h for per_image in steps for image, per in per_image
                                    for h in [image]+[h for orders in per for order in orders for h, _ in order]]))
    table = dict(zip(histories, policy(model, histories)))
    turn = lambda orders: sum(math.prod(table[h][0].get(move, 0.) for h, move in order) for order in orders)
    return [(sum(math.prod(map(turn, per)) for _, per in per_image), float(np.mean([table[image][1] for image, _ in per_image])))
            for per_image in steps]

def tempered(p, temperature):
    """p^(1/temperature), normalised, computed in log space so that no temperature underflows or overflows the
    weights; zeros keep weight 0."""
    with np.errstate(divide='ignore'):
        log = np.log(np.asarray(p, np.float64))/temperature
    w = np.exp(log-log.max())
    return w/w.sum()


class Line:
    """A dense_selfplay.Engine slot that continues `start` by `sims`-simulation Gumbel searches of `model`, each
    placement sampled from the root visit counts at `temperature` (from the search policy when no visit was made),
    until it holds `plies` placements; `shares` holds each sampled placement's visit share (None without visits).
    Book lines are searched without the tactical solver (`solver` None)."""

    def __init__(self, model, start, plies, sims, samples, tactics, temperature, seed):
        self.model, self.plies, self.temperature, self.reason, self.solver = model, plies, temperature, None, None
        self.budget, self.samples = sims, min(samples, sims)
        self.rng, self.moves, self.shares = np.random.default_rng(seed), [tuple(m) for m in start], []
        self.tree = model.tree(list(self.moves), seed, tactics)

    def searched(self, result):
        visits = result['visits'].astype(np.float64)
        total = visits.sum()
        k = self.rng.choice(len(visits), p=tempered(visits if total else result['policy'], self.temperature))
        move = tuple(int(v) for v in result['actions'][k])
        self.tree.advance(move)
        self.moves.append(move)
        self.shares.append(visits[k]/total if total else None)
        return len(self.moves) < self.plies


def continuations(model, starts, settings, rng, leaf_batch=256):
    """(lines, shares): each start (a move list) continued to settings.book_plies placements, each placement sampled
    at book_temperature from the visit counts of a book_sims-simulation search of `model` (`Line`), or with book_sims
    0 from its policy; shares[i] lists the visit share of each placement line i gained (None in policy mode)."""
    s = settings
    lines = [[tuple(m) for m in start] for start in starts]
    if s.book_sims <= 0:
        grown = [len(line) for line in lines]
        while pending := [line for line in lines if len(line) < s.book_plies]:
            for line, (p, _) in zip(pending, policy(model, pending)):
                line.append(list(p)[rng.choice(len(p), p=tempered(list(p.values()), s.book_temperature))])
        return lines, [[None]*(len(line)-n) for line, n in zip(lines, grown)]
    from dense_selfplay import Engine  # torch loads only where a search runs; readers such as the dashboard skip it
    engine = Engine(leaf_batch)
    slots = [Line(model, line, s.book_plies, s.book_sims, s.root_samples, s.tactics, s.book_temperature,
                  int(rng.integers(2**31))) if len(line) < s.book_plies else None for line in lines]
    for slot in filter(None, slots):
        engine.add(slot)
    while engine.slots:
        engine.step()
    engine.close()
    for slot in filter(None, slots):
        slot.tree.close()
    return [slot.moves if slot else line for slot, line in zip(slots, lines)], [slot.shares if slot else [] for slot in slots]


def skew(counts):
    """P1's advantage from colour-swapped pairs `counts` (pairs by P1's points 0, 1/2, 1, 3/2, 2): the league's
    pair model (dense_eval.rate), a Jeffreys Dirichlet(counts + 1/2) over the five pair scores, summarised by the
    posterior mean m and sd of P1's pair score. {elo, interval, pairs}: elo is 400 log10(m / (1 - m)); the interval
    maps m -+ 1.96 sd, clipped to (0, 1), the same way. Each pair has both players on both colours, so the pair score
    is half of 1 plus the paired difference of a player's scores as P1 and as P2: player strength cancels."""
    counts = np.asarray(counts, np.float64)
    alpha = counts+.5
    total = alpha.sum()
    mean = PAIR_SCORES@alpha/total
    sd = math.sqrt(max(PAIR_SCORES**2@alpha/total-mean**2, 0.)/(total+1))
    elo = lambda p: 400*math.log10(p/(1-p))
    clip = lambda p: min(max(p, 1e-6), 1-1e-6)
    return dict(elo=elo(mean), interval=[elo(clip(mean-1.96*sd)), elo(clip(mean+1.96*sd))], pairs=int(counts.sum()))


def skewed(node, settings):
    """Whether the node's pairs show significant skew: at least book_min_games pairs and a skew interval wholly
    beyond +-book_max_skew Elo."""
    low, high = node['skew']['interval']
    return node['skew']['pairs'] >= settings.book_min_games and (low > settings.book_max_skew or high < -settings.book_max_skew)


def short_skewed(node, settings, short_limit):
    """Decisive first-player skew on a line shorter than the played openings' mean-length quantile."""
    decisive = node.get('p1_wins', 0)+node.get('p2_wins', 0)
    return short_limit is not None and decisive >= settings.book_short_min_games and node.get('mean_plies') is not None \
            and node['mean_plies'] < short_limit \
            and abs(node['p1_wins']-decisive/2)/math.sqrt(decisive/4) >= settings.book_short_skew_z


def judge(node, settings, short_limit=None):
    """Why an opening must retire, else None. `short_limit` is the played openings' mean-length quantile."""
    if skewed(node, settings):
        return 'skew'
    if short_skewed(node, settings, short_limit):
        return 'short_skew'
    p = node['champion_probability']
    return 'probability' if p is not None and p < settings.book_min_prob else None


def worst(node):
    """The largest |skew| inside the node's interval: the balance evidence two contestants are compared on."""
    return max(abs(v) for v in node['skew']['interval'])


def new_node(moves, now=0.):
    key, moves = canonical(moves)
    return dict(key=key, moves=moves, depth=len(moves), status=None, reason=None, challenges=None, probability=None,
                visit_share=None, checkpoint=None, created_at=now, retired_at=None, champion_probability=None, champion_value=None,
                scored_by=None, games=0, p1_wins=0, p2_wins=0, capped=0, plies_sum=0, plies_games=0,
                mean_plies=None, pairs=[0]*5, skew=skew([0]*5))


def suites():
    """The opening suites: the live 'book' and every frozen openings/<suite>.json."""
    return (LIVE,)+tuple(sorted(p.stem for p in FROZEN.glob('*.json')))


def book_path(run, suite):
    return Path(run)/('openings.json' if suite == LIVE else f'openings-{suite}.json')


def check(settings):
    """Raise ValueError unless the opening settings are usable: opening_suite one of `suites`, opening_book empty
    outside 'book' (the evaluator stamps it), 2 <= book_min_plies <= book_plies <= MAX_PLIES and book_plies <
    max_plies (a one-placement opening is the empty start), book_temperature > 0,
    book_sims and book_max_skew >= 0, book_size, book_min_games and book_short_min_games >= 1,
    book_short_skew_z >= 0, book_short_quantile in [0, 1], book_revisit_fraction in [0, 1],
    book_refresh_hours > 0, book_min_prob in [0, 1) and book_weighting 'uniform' or 'least_played'."""
    s = settings
    if s.opening_suite not in suites():
        raise ValueError(f'opening_suite must be one of {suites()}, not {s.opening_suite!r}')
    if s.opening_suite != LIVE and s.opening_book:
        raise ValueError("opening_book names a book state; it stays empty outside opening_suite 'book'")
    if not 2 <= s.book_min_plies <= s.book_plies <= MAX_PLIES or s.book_plies >= s.max_plies or s.book_temperature <= 0 \
            or min(s.book_sims, s.book_max_skew, s.book_short_skew_z) < 0 \
            or min(s.book_size, s.book_min_games, s.book_short_min_games) < 1 \
            or not 0 <= s.book_short_quantile <= 1 or not 0 <= s.book_revisit_fraction <= 1 or s.book_refresh_hours <= 0 \
            or not 0 <= s.book_min_prob < 1 or s.book_weighting not in ('uniform', 'least_played'):
        raise ValueError(f'need 2 <= book_min_plies <= book_plies <= {MAX_PLIES}, book_plies < max_plies, '
                         'book_temperature > 0, book_sims and '
                         'book_max_skew and book_short_skew_z >= 0, book_size, book_min_games and '
                         'book_short_min_games >= 1, book_short_quantile and book_revisit_fraction in [0, 1], '
                         "book_refresh_hours > 0, book_min_prob in [0, 1) and book_weighting 'uniform' or 'least_played'")


def pairs_of(report):
    """The complete colour pairs of an evaluator report, [[game, game]] grouped by pair seed in the order the report
    holds them (reports only append)."""
    groups = {}
    for g in report['games']:
        groups.setdefault(g['seed'], []).append(g)
    return [p for p in groups.values() if len(p) == 2]


def report_id(report):
    """A report's identity for `Book.reconcile`: its `id` (`stamp` gives every report one)."""
    return report['id']


def stamp(run):
    """Every evaluation report of `run` (evaluations/*/report*.json, the reports archived beside a pairing's
    report.json included), after writing a new unique `id` into each one that lacks it (reports written before ids)."""
    reports = []
    for path in sorted((Path(run)/'evaluations').glob('*/report*.json')):
        report = json.loads(path.read_text())
        if 'id' not in report:
            report['id'] = uuid.uuid4().hex
            write_json(path, report)
        reports.append(report)
    return reports

class Book:
    """The opening book of `suite` (default settings.opening_suite) in `run` (module contract) under EvaluationSettings
    `settings`. A missing live file is a book holding only the origin; a missing frozen file starts from the repo's
    openings/<suite>.json. A live book's draw rule is part of its state (file field `weighting`): opening it with
    `settings` adopts settings.book_weighting; without settings the book is a view of the file for reading (stats,
    graph, digest), whose draw rule and refresh need settings. Nothing is written before `save`, `record`,
    `reconcile` or `refresh`."""

    def __init__(self, run, settings=None, suite=None):
        self.settings, self.suite = settings, suite or settings.opening_suite
        self.path, self.frozen = book_path(run, self.suite), self.suite != LIVE
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        elif self.frozen:
            source = FROZEN/f'{self.suite}.json'
            if not source.exists():
                raise ValueError(f'No frozen opening suite {self.suite!r}: {source} is missing')
            self.data = json.loads(source.read_text())
        else:
            self.data = dict(schema=SCHEMA, suite=LIVE, frozen=False, refreshed_by=None, refreshed_at=None, counted={},
                             imported=False, weighting='uniform', nodes=[new_node([ORIGIN])])
        if self.data['schema'] != SCHEMA or self.data['suite'] != self.suite:
            raise ValueError(f'{self.path} is not a {SCHEMA} book of suite {self.suite!r}')
        if settings and not self.frozen:
            self.data['weighting'] = settings.book_weighting
        self.nodes = {n['key']: n for n in self.data['nodes']}
        self._missing_plies = {n['key'] for n in self.nodes.values() if 'plies_games' not in n}
        self._legacy_counted = dict(self.data['counted'])

    def openings(self):
        """The openings (status 'opening'), by key."""
        return sorted((n for n in self.nodes.values() if n['status'] == 'opening'), key=lambda n: n['key'])

    def digest(self):
        """The state reports are played under: '' for a frozen book (its suite fixes its openings and their weights),
        else the sha256 of its draw rule (`weighting`) and the opening keys, so a change of either names a new state."""
        if self.frozen:
            return ''
        state = [self.data['weighting'], [n['key'] for n in self.openings()]]
        return hashlib.sha256(json.dumps(state).encode()).hexdigest()

    def due(self, champion, now):
        """Whether a live book needs a refresh: `champion` did not make the newest one or book_refresh_hours passed."""
        return not self.frozen and (self.data['refreshed_by'] != champion
                                    or now-self.data['refreshed_at'] >= self.settings.book_refresh_hours*3600)

    def draw(self, seed):
        """The moves of an opening drawn by `seed`: in proportion to `weight` in a frozen book, else uniformly, or with
        book_weighting 'least_played' in proportion to 1 / (1 + its pairs); played in one of the 12 symmetric
        orientations, chosen uniformly by `seed`, so every image of a class is equally likely (a frozen weight of
        class's number of images makes every physical opening equally likely). ValueError without openings."""
        openings = self.openings()
        if not openings:
            raise ValueError(f'{self.path} has no opening; the evaluator refreshes a live book once a champion exists')
        w = np.array([n['weight'] if self.frozen else 1/(1+n['skew']['pairs']) if self.data['weighting'] == 'least_played'
                      else 1. for n in openings], np.float64)
        rng = np.random.default_rng(seed)
        moves = np.array(openings[rng.choice(len(openings), p=w/w.sum())]['moves'], np.int64)
        return (moves @ hexcrop.SYMMETRIES[rng.integers(len(hexcrop.SYMMETRIES))]).tolist()

    def add(self, moves, now):
        """The node of `moves`, adding it and its missing prefix nodes."""
        for k in range(1, len(moves)+1):
            key = canonical(moves[:k])[0]
            if key not in self.nodes:
                self.nodes[key] = new_node(moves[:k], now)
                self.data['nodes'].append(self.nodes[key])
        return self.nodes[canonical(moves)[0]]

    def tally(self, pair):
        """Count a completed colour pair on every node its opening passed through that is in the book."""
        moves = pair[0]['opening']
        points = sum(1. if g['winner'] == 0 else .5 if g['winner'] < 0 else 0. for g in pair)
        for k in range(1, len(moves)+1):
            node = self.nodes.get(canonical(moves[:k])[0])
            if node is None:
                continue
            node['games'] += len(pair)
            node['p1_wins'] += sum(g['winner'] == 0 for g in pair)
            node['p2_wins'] += sum(g['winner'] == 1 for g in pair)
            node['capped'] += sum(g['winner'] < 0 for g in pair)
            self._count_plies(node, pair)
            node['pairs'][round(2*points)] += 1
            node['skew'] = skew(node['pairs'])

    @staticmethod
    def _count_plies(node, pair):
        node['plies_sum'] = node.get('plies_sum', 0)+sum(g['plies'] for g in pair)
        node['plies_games'] = node.get('plies_games', 0)+len(pair)
        node['mean_plies'] = node['plies_sum']/node['plies_games']

    def record(self, pair, report):
        """`tally` a completed colour pair (two game records of one opening, as dense_eval writes them) that was just
        appended to the report of `report_id` `report`, count it in `counted` and rewrite the file. Statuses wait for a
        refresh."""
        self.tally(pair)
        self.data['counted'][report] = self.data['counted'].get(report, 0)+1
        self.save()

    def reconcile(self, reports):
        """`tally` the pairs of `reports` (evaluator report dicts) that the book has not counted yet: per report, its
        complete pairs beyond `counted` (so a pair written to its report but not to the book, as when the evaluator
        stopped between the two writes, is counted once on the next start). Reports of the book's own suite count
        always; a live book counts the reports of other suites too the first time (`imported`), adding the positions of
        their openings (status null: a refresh may make them openings). Returns the pairs counted."""
        if self._missing_plies:
            for key in self._missing_plies:
                node = self.nodes[key]
                node.setdefault('plies_sum', 0)
                node.setdefault('plies_games', 0)
                node.setdefault('mean_plies', None)
            for report in reports:
                for pair in pairs_of(report)[:self._legacy_counted.get(report_id(report), 0)]:
                    for k in range(1, len(pair[0]['opening'])+1):
                        key = canonical(pair[0]['opening'][:k])[0]
                        if key in self._missing_plies:
                            self._count_plies(self.nodes[key], pair)
            self._missing_plies.clear()
        added = 0
        for report in reports:
            if report['settings'].get('opening_suite') != self.suite and (self.frozen or self.data['imported']):
                continue
            pairs = pairs_of(report)
            key = report_id(report)
            for pair in pairs[self.data['counted'].get(key, 0):]:
                if not self.frozen:
                    self.add(pair[0]['opening'], self.data['refreshed_at'] or 0.)
                self.tally(pair)
                added += 1
            self.data['counted'][key] = max(len(pairs), self.data['counted'].get(key, 0))
        self.data['imported'] = True
        self.save()
        return added

    def score(self, model, checkpoint, nodes):
        """Set champion_probability, champion_value and scored_by of `nodes` under `model` (`reach`)."""
        for node, (p, value) in zip(nodes, reach(model, [n['moves'] for n in nodes])):
            node.update(champion_probability=p, champion_value=value, scored_by=checkpoint)

    def retire(self, node, reason, now):
        node.update(status='retired', reason=reason, retired_at=now, challenges=None)

    def refresh(self, model, checkpoint, rng, now=None, leaf_batch=256):
        """Refresh a live book with the champion `model` (checkpoint id `checkpoint`), in order:
        1. score every opening (`reach`), retire nested parents, and retire each remaining one `judge` rejects;
        2. settle contests: once a challenger and the opening it challenges both have book_min_games pairs, the one
           of larger `worst` |skew| retires ('replaced'; the challenger on a tie) and the other is settled; a challenger
           whose opponent retired is settled;
        3. challenge a random book_revisit_fraction of the settled openings: each gets one challenger at its own
           depth, sampled from its parent position (`generate`);
        4. while fewer than book_size openings are settled, add settled replacements: first the imported positions
           with pairs that no opening passes through (most pairs first; at least book_min_plies deep, plausible and
           free of either skew), then one child of every opening retired for skew in step 1 below book_plies, then fresh
           openings from the origin (`generate`, up to GROW_ROUNDS rounds); retire parents extended during adoption or
           generation.
        Writes the file; returns {digest, openings, challengers, retired {reason: count} of this refresh, added}."""
        if self.frozen:
            raise ValueError(f'The frozen suite {self.suite!r} is never refreshed')
        s, now = self.settings, time.time() if now is None else now
        retired = dict.fromkeys(REASONS, 0)
        openings = self.openings()
        self.score(model, checkpoint, openings)
        lengths = [n['mean_plies'] for n in openings if n.get('mean_plies') is not None]
        short_limit = float(np.quantile(lengths, s.book_short_quantile)) if lengths else None

        def retire_nested():
            covered = {canonical(n['moves'][:d])[0] for n in self.openings() for d in range(1, n['depth'])}
            retired_keys = set()
            for node in self.openings():
                if node['key'] in covered:
                    self.retire(node, 'nested', now)
                    retired_keys.add(node['key'])
                    retired['nested'] += 1
            for node in self.openings():
                if node['challenges'] in retired_keys:
                    node['challenges'] = None

        retire_nested()
        extend = []
        for node in openings:
            if node['status'] != 'opening':
                continue
            if reason := judge(node, s, short_limit):
                self.retire(node, reason, now)
                retired[reason] += 1
                if reason in ('skew', 'short_skew') and node['depth'] < s.book_plies:
                    extend.append(node)
        for node in self.openings():
            rival = self.nodes.get(node['challenges'])
            if rival is None:
                continue
            if rival['status'] != 'opening':
                node['challenges'] = None
            elif min(node['skew']['pairs'], rival['skew']['pairs']) >= s.book_min_games:
                loser, winner = (node, rival) if worst(node) >= worst(rival) else (rival, node)
                self.retire(loser, 'replaced', now)
                retired['replaced'] += 1
                winner['challenges'] = None
        challenged = {n['challenges'] for n in self.openings()}
        settled = [n for n in self.openings() if n['challenges'] is None and n['key'] not in challenged and n['depth'] > 1]
        chosen = rng.permutation(len(settled))[:round(s.book_revisit_fraction*len(settled))]
        added = self.generate(model, checkpoint, [(settled[k]['moves'][:-1], settled[k]['depth'], settled[k]['key'])
                                                  for k in sorted(chosen)], rng, now, leaf_batch)
        missing = lambda: s.book_size-sum(n['challenges'] is None for n in self.openings())
        added += self.adopt(model, checkpoint, missing(), now, short_limit)
        retire_nested()
        if missing() > 0:
            added += self.generate(model, checkpoint, [(n['moves'], n['depth']+1, None) for n in extend][:missing()],
                                   rng, now, leaf_batch)
            retire_nested()
        for _ in range(GROW_ROUNDS):
            if missing() <= 0:
                break
            added += self.generate(model, checkpoint, [([ORIGIN], s.book_min_plies, None)]*missing(), rng, now, leaf_batch)
            retire_nested()
        self.data.update(refreshed_by=checkpoint, refreshed_at=now)
        self.save()
        return dict(digest=self.digest(), openings=len(self.openings()),
                    challengers=sum(n['challenges'] is not None for n in self.openings()), retired=retired, added=added)

    def adopt(self, model, checkpoint, count, now, short_limit=None):
        """Make up to `count` imported positions openings (step 4 of `refresh`), none of them a prefix of an opening,
        a retired opening or an earlier adoption, and neither skew rule rejects them; returns how many."""
        s = self.settings
        if count <= 0:
            return 0
        used = {canonical(n['moves'][:d])[0] for n in self.nodes.values() if n['status'] for d in range(1, n['depth']+1)}
        pool = sorted((n for n in self.nodes.values() if n['status'] is None and n['key'] not in used and n['skew']['pairs']
                       and s.book_min_plies <= n['depth'] <= s.book_plies and not skewed(n, s)
                       and not short_skewed(n, s, short_limit)),
                      key=lambda n: (-n['skew']['pairs'], n['key']))
        self.score(model, checkpoint, pool)
        taken = 0
        for node in pool:
            if taken < count and node['key'] not in used and node['champion_probability'] >= s.book_min_prob:
                node.update(status='opening', probability=node['champion_probability'], checkpoint=checkpoint, created_at=now)
                used.update(canonical(node['moves'][:d])[0] for d in range(1, node['depth']+1))
                taken += 1
        return taken

    def generate(self, model, checkpoint, starts, rng, now, leaf_batch=256):
        """New openings, one per start (moves, depth, challenged key or None) where possible: ALTERNATIVES lines continue
        the start's moves to book_plies (`continuations`); on each line the opening is the shortest prefix of at least
        `depth` placements (and longer than the start) whose position is not, and never was, an opening, was not
        taken by an earlier start and is plausible (policy probability at least book_min_prob); of the lines with
        one, the shallowest wins, among equals the one whose value-head P1 expected score is nearest 1/2. A candidate
        cannot be a proper prefix of an active opening; a line may extend through an active opening.
        It records probability and visit_share (the product of the visit shares of the placements the search sampled
        after the start; null in policy mode). A challenger (third field set) competes with that opening: it is taken
        at exactly `depth` placements (a line whose position there is used is discarded, never extended), so the two
        share no games. Returns how many were added."""
        s = self.settings
        if not starts:
            return 0
        starting = [m for m, _, _ in starts for _ in range(ALTERNATIVES)]
        lines, shares = continuations(model, starting, s, rng, leaf_batch)
        covered = {canonical(n['moves'][:d])[0] for n in self.openings() for d in range(1, n['depth'])}
        prefixes = []  # per line: [(key, moves, visit share)] of the unused prefixes deep enough, shortest first
        for i, line in enumerate(lines):
            start, depth, challenged = starts[i//ALTERNATIVES]
            prefixes.append([])
            for d in [depth] if challenged else range(max(depth, len(start)+1), len(line)+1):
                key, moves = canonical(line[:d])
                if key not in covered and (self.nodes.get(key) or {}).get('status') is None:
                    gained = shares[i][:d-len(start)]
                    prefixes[-1].append((key, moves, None if None in gained else math.prod(gained)))
        flat = list({key: moves for options in prefixes for key, moves, _ in options}.items())
        scores = dict(zip((key for key, _ in flat), reach(model, [moves for _, moves in flat])))
        added, taken = 0, set()
        for k, (_, _, challenged) in enumerate(starts):
            fit = []
            for options in prefixes[k*ALTERNATIVES:(k+1)*ALTERNATIVES]:
                key, moves, share = next((o for o in options if o[0] not in covered and o[0] not in taken
                                          and scores[o[0]][0] >= s.book_min_prob),
                                         (None, None, None))
                if key is not None:
                    fit.append((len(moves), abs(scores[key][1]-.5), key, moves, share))
            if not fit:
                continue
            *_, key, moves, share = min(fit, key=lambda f: f[:3])
            taken.add(key)
            covered.update(canonical(moves[:d])[0] for d in range(1, len(moves)))
            node = self.add(moves, now)
            node.update(status='opening', challenges=challenged, probability=scores[key][0], visit_share=share,
                        checkpoint=checkpoint, created_at=now, champion_probability=scores[key][0],
                        champion_value=scores[key][1], scored_by=checkpoint)
            added += 1
        return added
    def prune(self):
        """Remove the retired openings without games and then the null-status nodes without games that no remaining
        opening or retired opening passes through; returns how many nodes were removed."""
        before = len(self.nodes)
        keep = {k: n for k, n in self.nodes.items() if not (n['status'] == 'retired' and not n['games'])}
        needed = {canonical(n['moves'][:d])[0] for n in keep.values() if n['status'] for d in range(1, n['depth']+1)}
        self.nodes = {k: n for k, n in keep.items() if n['status'] or n['games'] or k in needed or n['depth'] == 1}
        self.data['nodes'] = list(self.nodes.values())
        self.save()
        return before-len(self.nodes)

    def stats(self):
        """{suite, frozen, digest, refreshed_by, refreshed_at, nodes, openings, challengers, retired {reason: count},
        depths {depth: openings}, mean_abs_skew (mean |skew elo| over openings with pairs, null without),
        histogram {edges, counts} of those skews (SKEW_EDGES, outer bins open-ended)}."""
        openings = self.openings()
        played = [n['skew']['elo'] for n in openings if n['skew']['pairs']]
        edges = SKEW_EDGES
        counts = [0]*(len(edges)-1)
        for v in played:
            counts[min(max(int(np.searchsorted(edges, v, 'right'))-1, 0), len(counts)-1)] += 1
        return dict(suite=self.suite, frozen=self.frozen, digest=self.digest(), refreshed_by=self.data['refreshed_by'],
                    refreshed_at=self.data['refreshed_at'], nodes=len(self.nodes), openings=len(openings),
                    challengers=sum(n['challenges'] is not None for n in openings),
                    retired={r: sum(n['reason'] == r for n in self.nodes.values()) for r in REASONS},
                    depths={str(d): sum(n['depth'] == d for n in openings) for d in sorted({n['depth'] for n in openings})},
                    mean_abs_skew=float(np.mean(np.abs(played))) if played else None,
                    histogram=dict(edges=edges, counts=counts))

    def graph(self):
        """The DAG with its statistics: {stats, nodes (every node), edges [[parent key, child key]] between book nodes}."""
        return dict(stats=self.stats(), nodes=list(self.nodes.values()),
                    edges=[[p, n['key']] for n in self.nodes.values() for p in parents(n['moves']) if p in self.nodes])

    def save(self):
        self.data['nodes'] = sorted(self.nodes.values(), key=lambda n: (n['depth'], n['key']))
        write_json(self.path, self.data)


def books(run):
    """{suite: Book} of the book files in `run`, as views for reading."""
    return {suite: Book(run, suite=suite) for suite in suites() if book_path(run, suite).exists()}


def summary(run, reports):
    """league.json `openings` over `reports` and the book files of `run`: {games, p1_wins, p2_wins, capped, players:
    {id: {p1_games, p1_wins, p2_games, p2_wins, mean_abs_skew}}, books: {suite: `Book.stats`}}, each report's candidate
    and opponent (Seal as 'seal') a player. mean_abs_skew is a player's mean |skew elo| over its games whose opening
    node, in the book of the report's suite, has pairs (null without any)."""
    found = books(run)
    out = dict(games=0, p1_wins=0, p2_wins=0, capped=0, players={}, books={k: b.stats() for k, b in found.items()})
    skews = {}
    for report in reports:
        book = found.get(report['settings'].get('opening_suite'))
        for g in report['games']:
            out['games'] += 1
            out['p1_wins'] += g['winner'] == 0
            out['p2_wins'] += g['winner'] == 1
            out['capped'] += g['winner'] < 0
            node = book.nodes.get(canonical(g['opening'])[0]) if book else None
            for name, colour in ((report['candidate'], g['challenger_color']), (report['opponent'], 1-g['challenger_color'])):
                player = out['players'].setdefault(name, dict(p1_games=0, p1_wins=0, p2_games=0, p2_wins=0))
                seat = 'p1' if colour == 0 else 'p2'
                player[f'{seat}_games'] += 1
                player[f'{seat}_wins'] += g['winner'] == colour
                if node and node['skew']['pairs']:
                    skews.setdefault(name, []).append(abs(node['skew']['elo']))
    for name, player in out['players'].items():
        player['mean_abs_skew'] = float(np.mean(skews[name])) if name in skews else None
    return out


def main():
    parser = argparse.ArgumentParser(description='Inspect or maintain a dense run\'s opening books. refresh and prune '
                                                 'write the book (refresh also stamps report ids, stamp): run '
                                                 'them while the evaluator is stopped.')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('refresh', 'stats', 'prune'):
        p = sub.add_parser(name)
        p.add_argument('--run', required=True)
        p.add_argument('--suite', default=LIVE)
        dense_config.add_arguments(p.add_argument_group('evaluation overrides'), dense_config.EvaluationSettings, 'eval_')
        if name == 'refresh':
            p.add_argument('--device', default='cpu')
            p.add_argument('--net-kernels', choices=('reference', 'fused'), help='model kernels for this process')
        if name == 'stats':
            p.add_argument('--nodes', action='store_true', help='also print the DAG (nodes and edges)')
    args = parser.parse_args()
    config = dense_config.load(args.run)
    settings = replace(dense_config.override(config.evaluation, args, 'eval_'), opening_suite=args.suite)
    check(replace(settings, opening_book=''))
    book = Book(args.run, settings)
    if args.command == 'refresh':
        champion = json.loads((Path(args.run)/'champion.json').read_text())['checkpoint']
        from dense_selfplay import load
        actor = config.actor if args.net_kernels is None else replace(
            config.actor, net_kernels=args.net_kernels, cuda_graphs=False)
        model = load(args.run, replace(config, device=args.device, actor=actor),
                                    source=(champion, Path(args.run)/'checkpoints'/champion/'ema.pt'))
        book.reconcile(stamp(args.run))
        now = time.time()
        out = book.refresh(model, champion, np.random.default_rng(int(now)), now, config.actor.leaf_batch)
    elif args.command == 'prune':
        out = dict(removed=book.prune(), **book.stats())
    else:
        out = book.graph() if args.nodes else book.stats()
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
