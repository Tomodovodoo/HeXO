"""CPU checks for the dense evaluator's opening books (dense_openings)."""
import argparse
from collections import Counter
import contextlib
from dataclasses import asdict, replace
import io
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

import numpy as np
import torch

import dense_config
import dense_openings
import dense_selfplay
import hexcrop
import hexnet
import train
from tests.test_dense import TINY

CHAMPION = 'main/000010'


class Uniform:
    """A dense_selfplay.Model stand-in: its policy is uniform over the legal cells within hex distance `radius` of the
    origin (all legal cells when None), other cells get relative probability e^-60; its value is always 0."""

    def __init__(self, radius=None):
        self.radius, self.evaluator = radius, self

    def evaluate(self, histories):
        out = []
        for h in histories:
            actions = hexcrop.legal_array(None, np.asarray(h, np.int64).reshape(-1, 2))
            near = np.ones(len(actions), bool) if self.radius is None else \
                np.max(np.abs(np.c_[actions, actions.sum(1)]), 1) <= self.radius
            out.append((actions, np.where(near, 0., -60.), np.zeros(len(actions))))
        return out


def settings(**values):
    return dense_config.EvaluationSettings(**{**dict(opening_suite='book', book_sims=0, book_size=4), **values})


def game(opening, winner, colour):
    return dict(opening=[list(m) for m in opening], winner=winner, challenger_color=colour)


def pair(opening, points):
    """A colour pair from `opening` in which P1 scores `points` (0, 0.5, 1, 1.5 or 2)."""
    winners = {0: (1, 1), .5: (1, -1), 1: (0, 1), 1.5: (0, -1), 2: (0, 0)}[points]
    return [game(opening, winners[0], 0), game(opening, winners[1], 1)]


def opening(book, moves, pairs=(0, 0, 0, 0, 0)):
    """Make `moves` an opening of `book` holding `pairs` (P1 point categories)."""
    node = book.add(moves, 0.)
    node.update(status='opening', probability=1., champion_probability=1., checkpoint=CHAMPION)
    for category, count in enumerate(pairs):
        for _ in range(count):
            book.tally(pair(node['moves'], category/2))
    return node


def orders(moves):
    """Every play order of `moves` that permutes stones within a turn only."""
    return [[tuple(p) for turn in order for p in turn] for order in
            itertools.product(*(itertools.permutations(map(tuple, t)) for t in dense_openings.turns(list(moves))))]


class CanonicalTests(unittest.TestCase):
    def test_every_symmetric_image_and_turn_order_has_one_key(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            moves = [(0, 0)]+[tuple(int(v) for v in rng.integers(-4, 5, 2)) for _ in range(4)]
            if len(set(moves)) < 5:
                continue
            key, representative = dense_openings.canonical(moves)
            self.assertEqual(dense_openings.canonical(representative), (key, representative))
            for m in hexcrop.SYMMETRIES:
                image = [tuple(int(v) for v in np.array(p) @ m) for p in moves]
                for order in orders(image):
                    self.assertEqual(dense_openings.canonical(order)[0], key)
        count = lambda m: len(dense_openings.images(m))
        self.assertEqual(count([(0, 0)]), 1)
        self.assertEqual(count([(0, 0), (1, 0), (-1, 0)]), 3)                  # a line through the origin: 3 axes
        self.assertEqual(count([(0, 0), (1, 0), (2, 0)]), 6)
        self.assertEqual(dense_openings.images([(0, 0), (1, 0), (2, 0)])[0], [(0, 0), (1, 0), (2, 0)])

        # The same stones split differently between turns are different positions.
        self.assertNotEqual(dense_openings.canonical([(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)])[0],
                            dense_openings.canonical([(0, 0), (1, 0), (0, 1), (2, 0), (0, 2)])[0])

    def test_the_representative_keeps_a_legal_play_order(self):
        moves = [(0, 0), (8, 0), (14, 0)]                    # (14, 0) is legal only after (8, 0)
        _, representative = dense_openings.canonical(moves)
        game = train.Game([tuple(m) for m in representative])
        game.close()

    def test_parents_remove_one_stone_of_the_last_turn(self):
        key = lambda m: dense_openings.canonical(m)[0]
        self.assertEqual(dense_openings.parents([(0, 0)]), [])
        self.assertEqual(dense_openings.parents([(0, 0), (1, 0), (2, 0)]), sorted({key([(0, 0), (1, 0)]), key([(0, 0), (2, 0)])}))
        self.assertEqual(dense_openings.parents([(0, 0), (1, 0), (2, 0), (0, 1)]), [key([(0, 0), (1, 0), (2, 0)])])

    def test_tempered_weights_survive_extreme_temperatures(self):
        np.testing.assert_allclose(dense_openings.tempered([1e-5, 2e-5, 0.], 1e-3), [0., 1., 0.], atol=1e-12)
        np.testing.assert_allclose(dense_openings.tempered([0., 3., 1.], .5), [0., .9, .1])
        np.testing.assert_allclose(dense_openings.tempered([10**6, 10**6], 1e-3), [.5, .5])

    def test_reach_sums_turn_orders_and_images(self):
        moves = [(0, 0), (1, 0), (2, 0)]
        first = len(hexcrop.legal_array(None, np.array(moves[:1])))
        after = lambda cell: len(hexcrop.legal_array(None, np.array([(0, 0), cell])))
        expected = 6*(1/(first*after((1, 0)))+1/(first*after((2, 0))))
        [(p, value)] = dense_openings.reach(Uniform(), [moves])
        self.assertAlmostEqual(p, expected)
        self.assertEqual(value, .5)
        self.assertEqual(dense_openings.reach(Uniform(), [[(0, 0)]]), [(1., .5)])


class SkewTests(unittest.TestCase):
    def test_estimate_and_interval_on_synthetic_pairs(self):
        even = dense_openings.skew([0, 0, 0, 0, 0])
        self.assertAlmostEqual(even['elo'], 0.)
        self.assertAlmostEqual(even['interval'][0], -even['interval'][1])
        self.assertEqual(even['pairs'], 0)
        counts = [1, 2, 6, 5, 6]
        alpha = np.array(counts)+.5
        x = np.arange(5)/4
        mean = x@alpha/alpha.sum()
        sd = math.sqrt((x**2@alpha/alpha.sum()-mean**2)/(alpha.sum()+1))
        elo = lambda p: 400*math.log10(p/(1-p))
        got = dense_openings.skew(counts)
        self.assertAlmostEqual(got['elo'], elo(mean))
        np.testing.assert_allclose(got['interval'], [elo(mean-1.96*sd), elo(mean+1.96*sd)])
        self.assertGreater(got['interval'][0], 0.)
        mirrored = dense_openings.skew(counts[::-1])
        self.assertAlmostEqual(mirrored['elo'], -got['elo'])
        np.testing.assert_allclose(mirrored['interval'], [-got['interval'][1], -got['interval'][0]])
        self.assertLess(dense_openings.skew([8, 0, 0, 0, 8])['interval'][0], 0.)    # split colours: no skew
        wide, narrow = dense_openings.skew([1, 1, 2, 2, 2]), dense_openings.skew([10, 10, 20, 20, 20])
        self.assertLess(narrow['interval'][1]-narrow['interval'][0], wide['interval'][1]-wide['interval'][0])

    def test_a_pair_counts_on_every_node_it_passed_through(self):
        with tempfile.TemporaryDirectory() as run:
            book = dense_openings.Book(run, settings())
            deep = [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)]
            leaf = opening(book, deep)
            image = [[int(v) for v in np.array(p) @ hexcrop.SYMMETRIES[5]] for p in deep]
            book.record(pair(image, 2), 'r')                               # P1 won both
            book.record([game(image, 0, 0), game(image, -1, 1)], 'r')      # 1.5
            book.record(pair([(0, 0), (1, 0), (2, 0), (3, 0)], 1), 'r')    # shares the first three placements
            book.record(pair([(0, 0), (5, 0), (6, 0)], 0), 'r')             # not in the book
            saved = {n['key']: n for n in json.loads(book.path.read_text())['nodes']}
            node = lambda k: saved[dense_openings.canonical(deep[:k])[0]]
            self.assertEqual((node(5)['games'], node(5)['p1_wins'], node(5)['p2_wins'], node(5)['capped']), (4, 3, 0, 1))
            self.assertEqual(node(5)['pairs'], [0, 0, 0, 1, 1])
            self.assertEqual(node(5)['skew'], dense_openings.skew([0, 0, 0, 1, 1]))
            self.assertEqual(node(4)['pairs'], [0, 0, 0, 1, 1])
            self.assertEqual(node(3)['pairs'], [0, 0, 1, 1, 1])           # its subtree, along the paths played
            self.assertEqual(node(1)['games'], 8)                         # every game starts at the origin
            self.assertEqual(leaf['status'], 'opening')                    # statuses wait for a refresh


class FilterTests(unittest.TestCase):
    def test_skew_needs_min_games_pairs_and_an_interval_beyond_max_skew(self):
        s = settings(book_min_games=16, book_max_skew=50., book_min_prob=1e-4)
        node = lambda counts, p=1.: dict(skew=dense_openings.skew(counts), champion_probability=p)
        self.assertIsNone(dense_openings.judge(node((0, 0, 0, 0, 15)), s))                       # too few pairs
        self.assertEqual(dense_openings.judge(node((0, 0, 0, 0, 16)), s), 'skew')
        self.assertEqual(dense_openings.judge(node((16, 0, 0, 0, 0)), s), 'skew')
        self.assertIsNone(dense_openings.judge(node((2, 4, 8, 4, 2)), s))
        lean = node((0, 0, 20, 30, 0))
        low = lean['skew']['interval'][0]
        self.assertGreater(low, 0.)
        self.assertEqual(dense_openings.judge(lean, replace(s, book_max_skew=low-1)), 'skew')
        self.assertIsNone(dense_openings.judge(lean, replace(s, book_max_skew=low+1)))
        self.assertEqual(dense_openings.judge(node((2, 4, 8, 4, 2), 5e-5), s), 'probability')
        self.assertIsNone(dense_openings.judge(node((2, 4, 8, 4, 2), None), s))                  # not scored yet


class RefreshTests(unittest.TestCase):
    def book(self, run, **values):
        return dense_openings.Book(run, settings(**values))

    def test_a_first_refresh_fills_the_book_with_the_shortest_distinct_openings(self):
        with tempfile.TemporaryDirectory() as run:
            book = self.book(run, book_size=6)
            result = book.refresh(Uniform(radius=2), CHAMPION, np.random.default_rng(0), now=5.)
            openings = book.openings()
            self.assertEqual((result['openings'], result['added'], result['challengers']), (6, 6, 0))
            self.assertEqual({n['depth'] for n in openings}, {3})
            self.assertEqual(len({n['key'] for n in openings}), 6)
            for n in openings:
                self.assertEqual(n['moves'][0], [0, 0])
                self.assertEqual((n['checkpoint'], n['scored_by'], n['created_at'], n['visit_share']), (CHAMPION, CHAMPION, 5., None))
                self.assertGreaterEqual(n['probability'], 1e-4)
                self.assertEqual(n['champion_value'], .5)
            saved = json.loads(book.path.read_text())
            self.assertEqual((saved['refreshed_by'], saved['refreshed_at']), (CHAMPION, 5.))
            self.assertEqual(result['digest'], book.digest())
            self.assertFalse(book.due(CHAMPION, 5.+3600))
            self.assertTrue(book.due(CHAMPION, 5.+6*3600))
            self.assertTrue(book.due('main/000020', 6.))

    def test_duplicates_extend_by_a_placement(self):
        """Only three positions of two neighbours of the origin exist up to symmetry, so a book of eight neighbour-only
        openings needs openings deeper than three placements."""
        with tempfile.TemporaryDirectory() as run:
            book = self.book(run, book_size=8)
            book.refresh(Uniform(radius=1), CHAMPION, np.random.default_rng(1), now=0.)
            openings = book.openings()
            depths = Counter(n['depth'] for n in openings)
            self.assertEqual((len(openings), len({n['key'] for n in openings})), (8, 8))
            self.assertLessEqual(depths[3], 3)
            self.assertGreater(depths[4]+depths[5], 0)
            self.assertEqual(book.stats()['depths'], {str(d): c for d, c in sorted(depths.items())})
    def test_skewed_openings_retire_and_a_child_replaces_them(self):
        with tempfile.TemporaryDirectory() as run:
            book = self.book(run, book_size=2, book_revisit_fraction=0.)
            skewed = opening(book, [(0, 0), (1, 0), (1, -1)], (0, 0, 2, 4, 14))
            balanced = opening(book, [(0, 0), (1, 0), (-1, 0)], (0, 4, 12, 4, 0))
            result = book.refresh(Uniform(radius=2), CHAMPION, np.random.default_rng(2), now=1.)
            self.assertEqual((skewed['status'], skewed['reason'], skewed['retired_at']), ('retired', 'skew', 1.))
            self.assertEqual(balanced['status'], 'opening')
            [child] = [n for n in book.openings() if n is not balanced]
            self.assertEqual(child['depth'], 4)
            self.assertEqual(dense_openings.canonical(child['moves'][:3])[0], skewed['key'])
            self.assertEqual((result['retired']['skew'], result['added'], result['openings']), (1, 1, 2))
            # At book_plies a skewed opening retires without a child: a fresh opening replaces it.
            deep = self.book(tempfile.mkdtemp(dir=run), book_size=1, book_plies=3, book_revisit_fraction=0.)
            last = opening(deep, [(0, 0), (1, 0), (1, -1)], (0, 0, 0, 0, 20))
            deep.refresh(Uniform(radius=2), CHAMPION, np.random.default_rng(3), now=1.)
            [fresh] = deep.openings()
            self.assertEqual((last['reason'], fresh['depth']), ('skew', 3))

    def test_implausible_openings_retire(self):
        with tempfile.TemporaryDirectory() as run:
            book = self.book(run, book_size=1, book_revisit_fraction=0.)
            far = opening(book, [(0, 0), (5, 0), (6, 0)])
            result = book.refresh(Uniform(radius=2), CHAMPION, np.random.default_rng(0), now=1.)
            self.assertEqual((far['status'], far['reason'], far['scored_by']), ('retired', 'probability', CHAMPION))
            self.assertLess(far['champion_probability'], 1e-4)
            self.assertEqual(far['probability'], 1.)                           # the generator's own stays
            self.assertEqual((result['retired']['probability'], result['openings']), (1, 1))

    def test_challengers_compete_on_balance_and_the_worse_retires(self):
        with tempfile.TemporaryDirectory() as run:
            book = self.book(run, book_size=2, book_revisit_fraction=1., book_min_games=4)
            first = opening(book, [(0, 0), (1, 0), (1, -1)])
            second = opening(book, [(0, 0), (1, 0), (-1, 0)])
            result = book.refresh(Uniform(radius=2), CHAMPION, np.random.default_rng(4), now=1.)
            challengers = [n for n in book.openings() if n['challenges']]
            self.assertEqual((result['openings'], result['challengers'], len(challengers)), (4, 2, 2))
            self.assertEqual({n['challenges'] for n in challengers}, {first['key'], second['key']})
            self.assertTrue(all(n['depth'] == 3 for n in challengers))
            # A revisit never extends a used position: with only three two-neighbour positions (all taken) a challenge
            # at depth three finds nothing, where a fresh opening would extend to depth four.
            tight = self.book(tempfile.mkdtemp(dir=run), book_size=3, book_revisit_fraction=1.)
            tight.refresh(Uniform(radius=1), CHAMPION, np.random.default_rng(0), now=0.)
            self.assertEqual([n['depth'] for n in tight.openings()], [3, 3, 3])
            again = tight.refresh(Uniform(radius=1), 'main/000020', np.random.default_rng(1), now=1.)
            self.assertEqual((again['challengers'], again['added']), (0, 0))
            rival = next(n for n in challengers if n['challenges'] == first['key'])
            for points in (1.5, 1.5, 1, 1):
                book.tally(pair(first['moves'], points))                       # the incumbent leans toward P1
                book.tally(pair(rival['moves'], 1))
            again = book.refresh(Uniform(radius=2), 'main/000020', np.random.default_rng(5), now=2.)
            self.assertEqual((first['status'], first['reason']), ('retired', 'replaced'))
            self.assertIsNone(rival['challenges'])
            self.assertEqual(again['retired']['replaced'], 1)
            self.assertEqual(sum(n['challenges'] is None for n in book.openings()), 2)      # book_size settled

    def test_reports_of_another_suite_are_imported_once_and_refresh_adopts_them(self):
        with tempfile.TemporaryDirectory() as run:
            played = [(0, 0), (1, 0), (-1, 0)]
            games = [dict(g, seed=7) for g in pair(played, 1)]+[dict(g, seed=8) for g in pair(played, 2)]
            report = dict(id='r0', candidate='a', opponent='b', settings=dict(opening_suite='standard-v1'),
                          games=games+[dict(game(played, 0, 0), seed=9)])             # seed 9 is a half pair
            book = self.book(run, book_size=1, book_revisit_fraction=0.)
            self.assertEqual(book.reconcile([report]), 2)
            report['games'] += [dict(g, seed=10) for g in pair(played, 0)]
            self.assertEqual(book.reconcile([report]), 0)                              # other suites: the first time only
            node = book.nodes[dense_openings.canonical(played)[0]]
            self.assertEqual((node['status'], node['pairs']), (None, [0, 0, 1, 0, 1]))
            book.refresh(Uniform(radius=2), CHAMPION, np.random.default_rng(0), now=1.)
            self.assertEqual([n['key'] for n in book.openings()], [node['key']])
            self.assertEqual((node['status'], node['checkpoint']), ('opening', CHAMPION))

    def test_draws_follow_the_weighting(self):
        with tempfile.TemporaryDirectory() as run:
            book = self.book(run, book_weighting='least_played')
            fresh = opening(book, [(0, 0), (1, 0), (-1, 0)])
            played = opening(book, [(0, 0), (1, 0), (1, -1)], (0, 0, 99, 0, 0))
            draws = Counter(dense_openings.canonical(book.draw(seed))[0] for seed in range(2000))
            self.assertGreater(draws[fresh['key']], 20*draws[played['key']])
            uniform = dense_openings.Book(run, settings(book_weighting='uniform'))
            opening(uniform, fresh['moves']), opening(uniform, played['moves'])
            draws = Counter(dense_openings.canonical(uniform.draw(seed))[0] for seed in range(2000))
            self.assertLess(abs(draws[fresh['key']]-1000), 120)
            with self.assertRaises(ValueError):
                self.book(tempfile.mkdtemp(dir=run)).draw(1)

    def test_prune_stats_and_graph(self):
        with tempfile.TemporaryDirectory() as run:
            book = self.book(run)
            kept = opening(book, [(0, 0), (1, 0), (-1, 0), (0, 1)], (0, 1, 2, 1, 0))
            idle = opening(book, [(0, 0), (4, 0), (4, 1)])
            played = opening(book, [(0, 0), (2, 0), (2, 1)], (0, 0, 0, 1, 1))
            for node, reason in ((idle, 'probability'), (played, 'skew')):
                book.retire(node, reason, 1.)
            stats = book.stats()
            self.assertEqual((stats['openings'], stats['retired'], stats['depths']),
                             (1, dict(probability=1, skew=1, replaced=0), {'4': 1}))
            self.assertAlmostEqual(stats['mean_abs_skew'], abs(kept['skew']['elo']))
            self.assertEqual(sum(stats['histogram']['counts']), 1)
            graph = book.graph()
            self.assertIn([dense_openings.canonical(kept['moves'][:3])[0], kept['key']], graph['edges'])
            removed = book.prune()
            self.assertEqual(removed, 2)                                       # idle and its prefix (4, 0)
            self.assertNotIn(idle['key'], book.nodes)
            self.assertIn(played['key'], book.nodes)
            self.assertEqual(len(json.loads(book.path.read_text())['nodes']), len(book.nodes))


class GenerationTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(3)
        self.model = dense_selfplay.Model(hexnet.HexNet(TINY), 'tiny', CHAMPION, 'cpu', 64, 256)

    def test_reach_per_turn_equals_the_sum_over_every_play_order(self):
        moves = [(0, 0), (1, 0), (2, -1), (0, 1), (-1, 2), (3, 0), (1, 1)]
        sequences = [o for image in dense_openings.images(moves) for o in orders(image)]
        histories = list(dict.fromkeys(tuple(o[:k]) for o in sequences for k in range(1, len(moves)+1)))
        policy = dict(zip(histories, dense_openings.policy(self.model, histories)))
        brute = sum(math.prod(policy[tuple(o[:k])][0].get(o[k], 0.) for k in range(1, len(o))) for o in sequences)
        [(p, value)] = dense_openings.reach(self.model, [moves])
        self.assertAlmostEqual(p, brute, delta=1e-6*brute)
        self.assertLessEqual(p, 1.)
        self.assertAlmostEqual(value, np.mean([policy[tuple(i)][1] for i in dense_openings.images(moves)]))

    def test_search_and_policy_generation_with_a_tiny_model(self):
        for sims in (2, 0):
            with tempfile.TemporaryDirectory() as run:
                book = dense_openings.Book(run, settings(book_sims=sims, root_samples=2, book_min_prob=0.))
                self.assertEqual(book.refresh(self.model, CHAMPION, np.random.default_rng(sims), now=0.)['added'], 4)
                for n in book.openings():
                    self.assertEqual(dense_openings.canonical(n['moves']), (n['key'], n['moves']))
                    self.assertTrue(3 <= n['depth'] <= 5)
                    self.assertGreater(n['probability'], 0.)
                    self.assertTrue(0 <= n['champion_value'] <= 1)
                    if sims:
                        self.assertTrue(0 < n['visit_share'] <= 1)
                    else:
                        self.assertIsNone(n['visit_share'])

    def test_continuations_sample_visit_counts(self):
        s = settings(book_sims=4, root_samples=4, book_plies=3, book_temperature=1e-3)
        lines, shares = dense_openings.continuations(self.model, [[(0, 0)]]*3+[[(0, 0), (1, 0), (2, 0)]], s,
                                                     np.random.default_rng(0))
        self.assertEqual([len(l) for l in lines], [3, 3, 3, 3])
        self.assertEqual(shares[3], [])
        for share in shares[:3]:
            self.assertEqual(len(share), 2)
            self.assertTrue(all(0 < v <= 1 for v in share))


class FrozenTests(unittest.TestCase):
    def test_standard_v1_is_train_opening_for_as_a_frozen_book(self):
        ring = lambda q, r: max(abs(q), abs(r), abs(q+r))
        cells = [(q, r) for q in range(-3, 4) for r in range(-3, 4) if 0 < ring(q, r) <= 3]
        classes = Counter(dense_openings.canonical([(0, 0), a, b])[0] for a, b in itertools.combinations(cells, 2)
                          if 3 in (ring(*a), ring(*b)))
        with tempfile.TemporaryDirectory() as run:
            book = dense_openings.Book(run, dense_config.EvaluationSettings())
            self.assertTrue(book.frozen)
            self.assertEqual({n['key']: n['weight'] for n in book.openings()}, dict(classes))
            for n in book.openings():
                self.assertEqual(n['weight'], len(dense_openings.images(n['moves'])))
            for seed in range(200):
                self.assertIn(dense_openings.canonical(train.opening_for(seed, True))[0], classes)
            self.assertEqual(book.digest(), '')
            self.assertFalse(book.due(CHAMPION, 0.))
            with self.assertRaises(ValueError):
                book.refresh(Uniform(), CHAMPION, np.random.default_rng(0))
            # Draws follow train.opening_for's distribution: classes in proportion to their unordered cell pairs.
            draws = Counter(dense_openings.canonical(book.draw(seed))[0] for seed in range(4770))
            top = max(classes, key=classes.get)
            self.assertLess(abs(draws[top]-10*classes[top]), 4*math.sqrt(10*classes[top]))
            # Each draw is played in a seed-chosen orientation: every physical image of a class turns up.
            images = {frozenset(map(tuple, m[1:])) for m in map(book.draw, range(4770)) if dense_openings.canonical(m)[0] == top}
            self.assertEqual(len(images), classes[top])
            self.assertFalse(book.path.exists())                                # nothing written before a record
            book.record(pair(book.openings()[0]['moves'], 1), 'r')
            self.assertEqual(json.loads(book.path.read_text())['nodes'][0]['games'], 2)
            self.assertEqual(json.loads((dense_openings.FROZEN/'standard-v1.json').read_text())['nodes'][0]['games'], 0)

    def test_reconcile_counts_each_pair_of_the_own_suite_once(self):
        """A pair the report holds but the book missed (the evaluator stopped between the two writes) is counted on the
        next reconcile; recorded pairs are not counted twice; a frozen book adds no node and skips other suites."""
        with tempfile.TemporaryDirectory() as run:
            book = dense_openings.Book(run, dense_config.EvaluationSettings())
            inside = book.openings()[0]['moves']
            own = dict(id='r1', candidate='a', opponent='b', settings=dict(opening_suite='standard-v1'),
                       games=[dict(g, seed=1) for g in pair(inside, 2)]+[dict(g, seed=2) for g in pair([(0, 0), (7, 0), (8, 0)], 2)])
            other = dict(id='r2', candidate='a', opponent='c', settings=dict(opening_suite='book'),
                         games=[dict(g, seed=1) for g in pair(inside, 2)])
            self.assertEqual(book.reconcile([own, other]), 2)
            self.assertEqual(len(book.nodes), 50)
            root = lambda: book.nodes[dense_openings.canonical([(0, 0)])[0]]['pairs']
            self.assertEqual(root(), [0, 0, 0, 0, 2])
            late = [dict(g, seed=3) for g in pair(inside, 0)]
            own['games'] += late
            book.record(late, 'r1')                                            # written to both: counted once
            own['games'] += [dict(g, seed=4) for g in pair(inside, 1)]         # in the report only
            reopened = dense_openings.Book(run, dense_config.EvaluationSettings())
            self.assertEqual(reopened.reconcile([own, other]), 1)
            self.assertEqual(reopened.nodes[dense_openings.canonical([(0, 0)])[0]]['pairs'], [1, 0, 1, 0, 2])
            self.assertEqual(reopened.data['counted'], {'r1': 4})



class SummaryTests(unittest.TestCase):
    def test_colour_statistics_per_player_and_books(self):
        with tempfile.TemporaryDirectory() as run:
            book = dense_openings.Book(run, settings())
            node = opening(book, [(0, 0), (1, 0), (-1, 0)], (0, 0, 1, 0, 1))
            book.save()
            played = [dict(g, seed=1) for g in pair(node['moves'], 1.5)]
            report = dict(candidate='main/000020', opponent='seal', settings=asdict(settings()), games=played)
            out = dense_openings.summary(run, [report], settings())
            self.assertEqual((out['games'], out['p1_wins'], out['p2_wins'], out['capped']), (2, 1, 0, 1))
            self.assertEqual(out['players']['main/000020'], dict(p1_games=1, p1_wins=1, p2_games=1, p2_wins=0,
                                                                 mean_abs_skew=abs(node['skew']['elo'])))
            self.assertEqual(out['players']['seal']['p2_games'], 1)
            self.assertEqual(set(out['books']), {'book'})
            self.assertEqual(out['books']['book']['openings'], 1)


    def test_reading_books_does_not_load_torch(self):
        code = 'import sys, dense_openings; sys.exit("torch" in sys.modules)'
        self.assertEqual(subprocess.run([sys.executable, '-c', code], cwd=Path(dense_openings.__file__).parent).returncode, 0)


class SettingsTests(unittest.TestCase):
    def test_book_flags_parse_and_validate(self):
        parser = argparse.ArgumentParser()
        dense_config.add_arguments(parser, dense_config.EvaluationSettings, 'eval_')
        args = parser.parse_args(['--eval-opening-suite', 'book', '--eval-book-plies', '7', '--eval-book-min-plies', '4',
                                  '--eval-book-temperature', '2', '--eval-book-sims', '8', '--eval-book-size', '256',
                                  '--eval-book-revisit-fraction', '.5', '--eval-book-refresh-hours', '2.5',
                                  '--eval-book-max-skew', '40', '--eval-book-min-games', '8', '--eval-book-min-prob', '1e-5',
                                  '--eval-book-weighting', 'least_played'])
        s = dense_config.override(dense_config.EvaluationSettings(), args, 'eval_')
        self.assertEqual((s.opening_suite, s.book_plies, s.book_min_plies, s.book_temperature, s.book_sims, s.book_size,
                          s.book_revisit_fraction, s.book_refresh_hours, s.book_max_skew, s.book_min_games, s.book_min_prob,
                          s.book_weighting), ('book', 7, 4, 2., 8, 256, .5, 2.5, 40., 8, 1e-5, 'least_played'))
        dense_openings.check(s)
        d = dense_config.EvaluationSettings()
        self.assertEqual((d.opening_suite, d.book_plies, d.book_min_plies, d.book_temperature, d.book_sims, d.book_size,
                          d.book_revisit_fraction, d.book_refresh_hours, d.book_max_skew, d.book_min_games, d.book_min_prob,
                          d.book_weighting, d.opening_book),
                         ('standard-v1', 5, 3, 1.5, 16, 512, .25, 6., 50., 16, 1e-4, 'uniform', ''))
        dense_openings.check(d)
        self.assertEqual(dense_openings.suites(), ('book', 'standard-v1'))
        for bad in (dict(opening_suite='mixed-v1'), dict(opening_book='abc'), dict(book_plies=0), dict(book_plies=256),
                    dict(book_min_plies=6), dict(book_min_plies=1), dict(book_plies=11), dict(book_temperature=0.), dict(book_sims=-1),
                    dict(book_size=0), dict(book_min_games=0), dict(book_revisit_fraction=1.5), dict(book_refresh_hours=0.),
                    dict(book_min_prob=1.), dict(book_max_skew=-1.), dict(book_weighting='other')):
            with self.assertRaises(ValueError, msg=bad):
                dense_openings.check(replace(d, **bad))
        data = asdict(dense_config.RunConfig())
        data['evaluation'] = {k: v for k, v in data['evaluation'].items() if not k.startswith('book_') and k != 'opening_book'}
        self.assertEqual(dense_config.from_dict(data).evaluation, dense_config.EvaluationSettings())


class CommandLineTests(unittest.TestCase):
    def test_refresh_stats_and_prune(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            config = dense_config.RunConfig(device='cpu', model=dense_config.ModelSettings(**{k: getattr(TINY, k) for k in (
                'blocks', 'channels', 'pool_every', 'line_length', 'value_hidden', 'head_channels')}))
            dense_config.save(run, config)
            path = run/'checkpoints'/'main'/'000010'
            path.mkdir(parents=True)
            hexnet.save_model(path/'ema.pt', hexnet.HexNet(TINY))
            (run/'champion.json').write_text(json.dumps(dict(checkpoint=CHAMPION)))

            def cli(*args):
                out = io.StringIO()
                with unittest.mock.patch('sys.argv', ['dense_openings.py', *args, '--run', str(run)]), contextlib.redirect_stdout(out):
                    dense_openings.main()
                return json.loads(out.getvalue())
            refreshed = cli('refresh', '--eval-book-size', '3', '--eval-book-sims', '0', '--eval-book-min-prob', '0')
            self.assertEqual((refreshed['openings'], refreshed['added']), (3, 3))
            self.assertEqual(cli('stats', '--eval-book-size', '3')['openings'], 3)
            self.assertEqual(len(cli('stats', '--nodes')['stats']['depths']), 1)
            self.assertEqual(cli('prune')['removed'], 0)
            self.assertEqual(cli('stats', '--suite', 'standard-v1')['openings'], 47)

    def test_stamp_gives_legacy_and_archived_reports_distinct_ids_once(self):
        """An archived report and its pairing's current report.json can hold the same games (a pairing reopened under
        another protocol restarts its pair numbering); each gets its own id, so reconcile counts both."""
        with tempfile.TemporaryDirectory() as run:
            folder = Path(run)/'evaluations'/'a-vs-b'
            folder.mkdir(parents=True)
            games = [dict(g, seed=1) for g in pair([(0, 0), (1, 0), (-1, 0)], 2)]
            for name in ('report.json', 'report-5.json'):
                (folder/name).write_text(json.dumps(dict(candidate='a', opponent='b', settings=dict(opening_suite='book'),
                                                         games=games)))
            first = dense_openings.stamp(run)
            self.assertEqual(len({r['id'] for r in first}), 2)
            self.assertEqual([r['id'] for r in dense_openings.stamp(run)], [r['id'] for r in first])
            book = dense_openings.Book(run, settings())
            self.assertEqual(book.reconcile(first), 2)

if __name__ == '__main__':
    unittest.main()
