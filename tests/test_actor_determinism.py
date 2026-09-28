"""Seeded actor and evaluator games with a tiny CPU model must reproduce the recorded fixture exactly: moves,
root values and search policies. The opening is an 80-ply prefix of a
recorded dense-v1 game. Regenerate with `python -m tests.test_actor_determinism --write` only when search
behaviour is meant to change."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

import dense_config
import dense_eval
import dense_selfplay
import hexnet

FIXTURE = Path(__file__).with_name('fixtures')/'actor_selfplay.npz'
TINY = hexnet.HexNetConfig(blocks=2, channels=16, pool_every=2, line_length=5, value_hidden=16, head_channels=8)


def play(opening):
    """Three self-play games from the empty board and two evaluator games from `opening` [n, 2]; returns the
    fixture arrays (moves and game lengths, root values, concatenated full-search policies and their lengths)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(11)
        model = dense_selfplay.Model(hexnet.HexNet(TINY).eval(), 'tiny', 'test', 'cpu', 64, 256)
        settings = replace(dense_config.ActorSettings(), full_sims=8, cheap_sims=4, root_samples=4, max_plies=40,
                           full_fraction=.5, opening_random_plies=3., leaf_batch=64)
        engine = dense_selfplay.Engine(settings.leaf_batch)
        games = [dense_selfplay.SelfPlayGame([model, model], settings, seed) for seed in (1, 2, 3)]
        for game in games:
            engine.add(game)
        while engine.slots:
            engine.step()
        episodes = [game.episode() for game in games]
        opening = opening.tolist()
        matches = dense_eval.play([dense_eval.MatchGame([model, model], opening, k, 8, 4, True, len(opening)+16,
                                                        dict(index=k)) for k in range(2)], 64)
    finally:
        torch.set_num_threads(threads)
    moves = [e['moves'] for e, _ in episodes]+[m['moves'] for m in matches]
    policies = [r['policy'] for _, rows in episodes for r in rows if r['policy'] is not None]
    return dict(opening=np.asarray(opening, np.int64), lengths=np.array([len(m) for m in moves]),
                moves=np.concatenate([np.asarray(m, np.int64) for m in moves]),
                root_values=np.array([v for e, _ in episodes for v in e['root_values']]),
                policy_lengths=np.array([len(p) for p in policies]), policies=np.concatenate(policies))


class ActorDeterminismTests(unittest.TestCase):
    def test_seeded_games_match_the_fixture(self):
        expected = np.load(FIXTURE)
        got = play(expected['opening'])
        for key in ('lengths', 'moves', 'policy_lengths'):
            np.testing.assert_array_equal(got[key], expected[key], err_msg=key)
        np.testing.assert_allclose(got['root_values'], expected['root_values'], rtol=0, atol=1e-9)
        np.testing.assert_allclose(got['policies'], expected['policies'], rtol=0, atol=1e-6)


if __name__ == '__main__':
    if sys.argv[1:] == ['--write']:
        np.savez_compressed(FIXTURE, **play(np.load(FIXTURE)['opening']))
    else:
        unittest.main()
