"""Seeded actor and evaluator games with a tiny CPU model repeat: the same seed and budget give the same moves, root
values and search policies, which paired evaluation and reproducible runs rely on."""
from dataclasses import replace
from pathlib import Path
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
    """One self-play game from the empty board and one evaluator game from `opening` [n, 2]: their moves, the
    self-play root values and full-search policies."""
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(11)
        model = dense_selfplay.Model(hexnet.HexNet(TINY).eval(), 'tiny', 'test', 'cpu', 64, 256)
        settings = replace(dense_config.ActorSettings(hybrid_scheduler=False), full_sims=8, cheap_sims=4,
                           root_samples=4, max_plies=24, full_fraction=.5, opening_random_plies=3., leaf_batch=64)
        engine = dense_selfplay.Engine(settings.leaf_batch)
        game = dense_selfplay.SelfPlayGame([model, model], settings, 1)
        engine.add(game)
        while engine.slots:
            engine.step()
        episode, rows = game.episode()
        opening = opening.tolist()
        match, = dense_eval.play([dense_eval.MatchGame([model, model], opening, 0, 8, 4, True, len(opening)+8,
                                                       dict(index=0))], 64)
    finally:
        torch.set_num_threads(threads)
    return dict(moves=np.asarray(episode['moves']), match=np.asarray(match['moves']),
                root_values=np.asarray(episode['root_values']),
                policies=np.concatenate([r['policy'] for r in rows if r['policy'] is not None]))


class ActorDeterminismTests(unittest.TestCase):
    def test_the_same_seed_plays_the_same_games(self):
        opening = np.load(FIXTURE)['opening']
        first, second = play(opening), play(opening)
        self.assertGreater(len(first['match']), len(opening))
        for key in first:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)


if __name__ == '__main__':
    unittest.main()
