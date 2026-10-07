import random
import unittest

from hexo import Game
from tests.reference import AXES, Reference, has_cover, interleave

try:
    import torch
    from legacy.gpu_games import BatchedHexo
except ImportError:
    torch = None


class NativeRules(unittest.TestCase):
    def make_game(self, moves=()):
        game = Game(moves)
        self.addCleanup(game.close)
        return game

    def test_benchmark_records_only_executed_placements(self):
        import contextlib
        import io
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from hexo import library
        from tests.benchmark import drip_comparison

        history = [(0,0),(1,0),(0,-8),(-1,0),(-2,0),(-2,-8),(-4,-8),
                   (-3,0),(-4,0),(-6,-8),(-8,-8)]
        args = SimpleNamespace(seed=1, games=2, book="selected-book", opening_range="wide",
                               compare_library=None, seal_library=library, ms=100, depth=1,
                               width=16, max_stones=301, output=None)
        for wins_first in (True, False):
            submitted = [(-5,0) if wins_first else (-2,1), (56,0)]
            engine = Mock(path=Path(library))
            engine.search.side_effect = [dict(moves=submitted.copy()) for _ in range(2)]
            with patch("tests.benchmark.DripLibrary", return_value=engine), \
                 patch("tests.benchmark.SealLibrary", return_value=engine), \
                 patch("play.book_openings", return_value={"nodes": [{"moves": history}]}), \
                 contextlib.redirect_stdout(io.StringIO()):
                report = drip_comparison(args)
            self.assertEqual(report["summary"]["invalid_pairs"], 0 if wins_first else 1)
            if wins_first:
                self.assertEqual((report["summary"]["wins"], report["summary"]["losses"]), (1, 1))
            else:
                self.assertIsNone(report["summary"]["score"])
            for game in report["games"]:
                self.assertEqual(game["winner"], 0 if wins_first else -1)
                self.assertEqual(game["invalid"], None if wins_first else "Seal coordinate range exceeded")
                self.assertEqual(game["history"], history + submitted[:1])
                self.assertEqual(game["searches"][0]["moves"], submitted[:1])
                self.assertEqual(game["searches"][0]["submitted_moves"], submitted)

    def test_tt_injection_preserves_board_and_returns_complete_legal_turn(self):
        from legacy.curriculum import opening_for
        for seed in range(12):
            game = self.make_game(opening_for(20261001+seed, evaluation=True))
            before = (game.key, game.state(), game.evaluation, game.features())
            for ms in (1, 10):
                result = game.search(ms, depth=4, width=4, tt_injection=True)
                self.assertEqual((game.key, game.state(), game.evaluation, game.features()), before)
                side = game.player
                for point in result["moves"]:
                    self.assertTrue(game.legal(*point))
                    game.play(*point)
                self.assertTrue(game.winner >= 0 or game.player != side)
                for _ in result["moves"]:
                    game.undo()
        history = interleave([[(q, 0) for q in range(6)], [(2*q, 6) for q in range(6)]])
        game = self.make_game(history[:-1])
        result = game.search(100, width=4, tt_injection=True)
        self.assertEqual(len(result["moves"]), 1)
        game.play(*result["moves"][0])
        self.assertGreaterEqual(game.winner, 0)
        with self.assertRaises(ValueError):
            game.search(100, width=1, tt_injection=True)

    def test_root_admission_retains_baseline_and_exact_ordered_turns(self):
        game = self.make_game([(0, 0), (-2, 0), (-1, 0)])
        before = (game.key, game.state(), game.evaluation, game.features())
        base = game.turns(width=4)
        wider = game.turns(width=4, root_seconds=12, root_turns=24)
        base_moves = {tuple(t["moves"]) for t in base}
        self.assertTrue(base_moves <= {tuple(t["moves"]) for t in wider})
        self.assertGreater(len(wider), len(base))
        self.assertLessEqual(len(wider), 24)
        hashes = set()
        for turn in wider:
            side = game.player
            for point in turn["moves"]:
                self.assertTrue(game.legal(*point))
                game.play(*point)
            self.assertTrue(game.winner >= 0 or game.player != side)
            self.assertNotIn(game.key, hashes)
            hashes.add(game.key)
            for _ in turn["moves"]:
                game.undo()
        self.assertEqual((game.key, game.state(), game.evaluation, game.features()), before)
        for budget in (1, 5):
            result = game.search(budget, width=4, root_seconds=12, root_turns=24)
            self.assertEqual((game.key, game.state(), game.evaluation, game.features()), before)
            for point in result["moves"]:
                game.play(*point)
            for _ in result["moves"]:
                game.undo()

    def test_root_admission_preserves_immediate_first_stone_win(self):
        moves = interleave([[(q, 0) for q in range(6)], [(2*q, 6) for q in range(6)]])
        game = self.make_game(moves[:-1])
        base = game.turns(width=4)
        self.assertEqual(base, game.turns(width=4, root_seconds=12, root_turns=24))
        self.assertEqual(len(base[0]["moves"]), 1)
        game.play(*base[0]["moves"][0])
        self.assertEqual(game.winner, 0)
        game.undo()
        for options in ({"root_seconds": 5, "root_turns": 24},
                        {"root_seconds": 12, "root_turns": 7}):
            with self.assertRaises(ValueError):
                game.search(1, width=4, **options)

    def assert_state(self, game, reference):
        self.assertEqual((game.player, game.remaining, game.winner),
                         (reference.player, reference.remaining, reference.winner))
        self.assertEqual({(q, r): p for q, r, p in game.cells}, reference.cells)
        self.assertEqual(game.features(), reference.features())

    def test_sequential_radius_and_turn_phase(self):
        game = self.make_game()
        self.assertFalse(game.legal(1, 0))
        game.play(0, 0)
        self.assertEqual((game.player, game.remaining), (1, 2))
        expected = {(q, r) for q in range(-8, 9) for r in range(-8, 9)
                    if 0 < (abs(q)+abs(r)+abs(q+r))//2 <= 8}
        self.assertEqual(set(game.legal_moves()), expected)
        self.assertFalse(game.legal(16, 0))
        self.assertFalse(game.legal(9, 0))
        self.assertTrue(game.legal(8, -8))
        game.play(8, 0)
        self.assertEqual((game.player, game.remaining), (1, 1))
        self.assertTrue(game.legal(16, 0))
        game.play(16, 0)
        self.assertEqual((game.player, game.remaining), (0, 2))

    def test_all_axes_first_stone_terminal_and_overlines(self):
        for dq, dr in AXES:
            # Player zero's sixth stone wins before its second placement.
            for offsets in ([0, 1, 2, 3, 4, 5], [0, 1, 2, 4, 5, 6, 8, 3]):
                ours = [(k*dq, k*dr) for k in offsets]
                # A line parallel to the winning axis, separated by six cells.
                shift = (0, 6) if dq else (6, 0)
                theirs = [(2*k*dq+shift[0], 2*k*dr+shift[1]) for k in range(len(ours))]
                moves = interleave([ours, theirs])
                game = self.make_game(moves[:-1])
                self.assertEqual(game.remaining, 2)
                self.assertEqual(game.winner, -1)
                game.play(*moves[-1])
                self.assertEqual(game.winner, 0)
                self.assertEqual(game.remaining, 1)
                self.assertEqual(game.legal_moves(), [])
                self.assertEqual(game.search(1)["moves"], [])
                with self.assertRaises(ValueError):
                    game.play(0, 100)
                self.assertTrue(game.undo())
                self.assertEqual(game.winner, -1)

    def test_distant_expansion_has_no_board_crop(self):
        game, reference = self.make_game(), Reference()
        for q in range(0, 1601, 8):
            game.play(q, 0)
            reference.play(q, 0)
        self.assert_state(game, reference)
        self.assertTrue(game.legal(1608, 0))
        self.assertFalse(game.legal(1609, 0))
        for point in ((10**12+1, 0), (2**63, 0), (0.5, 0), (True, 0)):
            with self.assertRaises(ValueError):
                game.play(*point)

    def test_seeded_differential_make_undo_and_search(self):
        rng = random.Random(20260924)
        weights = [rng.randrange(-100, 101) for _ in range(729)]
        weights[0] = 0
        baseline = []
        for code in range(729):
            digits = [(code//3**k) % 3 for k in range(6)]
            n0, n1 = digits.count(1), digits.count(2)
            w = [0, 1, 12, 150, 2400, 24000, 1000000]
            baseline.append(w[n0] if n1 == 0 else -w[n1] if n0 == 0 else 0)
        for trial in range(4):
            game, reference = self.make_game(), Reference()
            game.load_table(weights)
            snapshots = []
            for step in range(32):
                self.assert_state(game, reference)
                snapshots.append((game.key, game.state(), game.evaluation, game.features()))
                features = reference.features()
                score = max(-500000, min(500000, sum(n*(a+b) for n, a, b in zip(features, baseline, weights))))
                self.assertEqual(game.evaluation, score if game.player == 0 else -score)
                for _ in range(4):
                    point = (rng.randrange(-20, 21), rng.randrange(-20, 21))
                    self.assertEqual(game.legal(*point), reference.legal(*point))
                if step % 8 == 0:
                    result = game.search(1, width=4)
                    self.assertEqual((game.key, game.state(), game.evaluation, game.features()), snapshots[-1])
                    for move in result["moves"]:
                        reference.play(*move)
                    for _ in result["moves"]:
                        reference.undo()
                if not reference.history:
                    move = (0, 0)
                else:
                    while True:
                        q, r = rng.choice(reference.history)
                        move = (q+rng.randrange(-3, 4), r+rng.randrange(-3, 4))
                        if reference.legal(*move):
                            break
                reference.play(*move)
                game.play(*move)
                if reference.winner >= 0:
                    break
            for expected in reversed(snapshots):
                self.assertTrue(game.undo())
                self.assertTrue(reference.undo())
                self.assertEqual((game.key, game.state(), game.evaluation, game.features()), expected)
            self.assertFalse(game.undo())

    def test_legacy_table_rejects_overflow_and_bad_empty_entry(self):
        game = self.make_game([(0, 0)])
        original = (game.key, game.evaluation)
        for value in (-10001, 10001, 2**32, -(2**32), 1.5):
            table = [0]*729
            table[1] = value
            with self.assertRaises(ValueError):
                game.load_table(table)
            self.assertEqual((game.key, game.evaluation), original)
        with self.assertRaises(ValueError):
            game.load_table([1]+[0]*728)

    def test_timed_search_across_dense_line_storage_edge(self):
        # Search stores lines starting within 32 cells of the origin in a
        # temporary array. Self-play over its edge on every axis must keep
        # legal complete turns and exact restoration, and end near their allowance.
        for direction in ((1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1)):
            game, reference = self.make_game(), Reference()
            q, r = direction
            for step in range(5):
                move = (8*step*q, 8*step*r)
                game.play(*move)
                reference.play(*move)
            for _ in range(10):
                if game.winner >= 0:
                    break
                before = (game.key, game.state(), game.features())
                result = game.search(30)
                self.assertEqual((game.key, game.state(), game.features()), before)
                # A guard against runaway searches, not a timing benchmark.
                self.assertLess(result["elapsed_ms"], 1000)
                side = game.player
                for move in result["moves"]:
                    self.assertTrue(reference.legal(*move))
                    game.play(*move)
                    reference.play(*move)
                self.assertTrue(game.winner >= 0 or game.player != side)
                self.assert_state(game, reference)

    def test_drip_full_turn_win_and_two_cell_defense(self):
        for theirs in ([(0, 2), (2, 2), (3, 2), (5, 2)],
                       [(0, 3), (2, 3), (4, 3), (6, 3)]):
            history = interleave([[(q, 0) for q in range(5)], theirs])
            game, reference = self.make_game(history), Reference()
            for point in history:
                reference.play(*point)
            winning = bool(reference.completions(1))
            self.assertEqual(game.turns(width=4), game.turns(width=4, root_seconds=12, root_turns=24))
            result = game.search(5, width=4)
            self.assertEqual(len(result["moves"]), 2)
            for move in result["moves"]:
                reference.play(*move)
                game.play(*move)
            if winning:
                self.assertEqual(game.winner, 1)
            else:
                self.assertFalse(reference.completions(0))


@unittest.skipIf(torch is None, "PyTorch is optional; install requirements/learning.txt for GPU rules checks")
class BatchedRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def devices(self):
        return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    def test_parity_growth_rejections_reset_and_truncation(self):
        for device in self.devices():
            env = BatchedHexo(3, device, capacity=2, max_placements=20)
            references = [Reference() for _ in range(3)]
            rng = random.Random(92)
            for step in range(23):
                actions = []
                expected = []
                for ref in references:
                    if not ref.history:
                        point = (0, 0)
                    elif step % 5 == 0:
                        point = rng.choice(ref.history)  # Occupied and rejected.
                    else:
                        q, r = rng.choice(ref.history)
                        point = (q+rng.randrange(-8, 9), r+rng.randrange(-8, 9))
                    accept = len(ref.history) < 20 and ref.legal(*point)
                    expected.append(accept)
                    actions.append(point)
                    if accept:
                        ref.play(*point)
                result = env.step(torch.tensor(actions, device=device))
                self.assertEqual(result.accepted.cpu().tolist(), expected)
                self.assertEqual(env.player.cpu().tolist(), [r.player for r in references])
                self.assertEqual(env.remaining.cpu().tolist(), [r.remaining for r in references])
                self.assertEqual(env.winner.cpu().tolist(), [r.winner for r in references])
                self.assertEqual(env.features.cpu().tolist(), [r.features() for r in references])
            env.reset([False, True, False])
            self.assertEqual(env.counts[1].item(), 0)
            self.assertEqual(env.features[1].sum().item(), 0)
            bounded = BatchedHexo(1, device, capacity=1, max_placements=3)
            for point in ((0, 0), (8, 0), (16, 0)):
                self.assertTrue(bounded.step(torch.tensor([point], device=device)).accepted.item())
            self.assertTrue(bounded.truncated.item())
            self.assertFalse(bounded.step(torch.tensor([[24, 0]], device=device)).accepted.item())
            bounded.reset()
            for point in ((2**63-1, 0), (-2**63, -2**63)):
                self.assertFalse(bounded.legal(torch.tensor([point], device=device)).item())

    def test_exact_wins_and_defensive_covers(self):
        # Both colors threaten: player one's broken two-stone win has priority.
        own = [(0, 2), (2, 2), (3, 2), (5, 2)]
        other = [(i, 0) for i in range(5)]
        win = interleave([other, own])
        # Two intersecting five-stone lines share the same blocking cell (2,2).
        cross = [(0, 0)] + [(q, 2) for q in (0, 1, 3, 4, 5)] + [(2, r) for r in (0, 1, 3, 4, 5)]
        filler = [(-4+2*(i % 4), 6+2*(i//4)) for i in range(10)]
        shared = interleave([cross, filler])
        two_blocks = interleave([other, [(0, 3), (2, 3), (4, 3), (6, 3)]])
        for device in self.devices():
            for history in (win, shared, two_blocks):
                env, ref = BatchedHexo(1, device, capacity=2), Reference()
                for point in history:
                    ref.play(*point)
                    self.assertTrue(env.step(torch.tensor([point], device=device)).accepted.item())
                side, remaining = ref.player, ref.remaining
                for _ in range(remaining):
                    own_wins = ref.completions(side, ref.remaining)
                    threats = ref.completions(1-side)
                    action = env.tactical_action()
                    self.assertEqual(action.winning.item(), bool(own_wins))
                    self.assertEqual(action.defending.item(), bool(threats) and has_cover(threats, ref.remaining) and not own_wins)
                    if not action.forced.item():
                        break
                    point = tuple(action.action[0].cpu().tolist())
                    self.assertTrue(ref.legal(*point))
                    if history == shared:
                        self.assertEqual(point, (2, 2))
                    ref.play(*point)
                    env.step(action.action)
                    if ref.winner >= 0:
                        break
                if history == win:
                    self.assertEqual(ref.winner, side)
                else:
                    self.assertFalse(ref.completions(1-side))

    def test_first_stone_overline_all_axes(self):
        base = interleave([[(0, 0)]+[(2*k, 6) for k in range(6)],
                           [(k, 2) for k in (0, 1, 2, 4, 5, 6, 3)]])
        transforms = (lambda q, r: (q, r), lambda q, r: (r, q), lambda q, r: (q+r, -q))
        for device in self.devices():
            for transform in transforms:
                env, native = BatchedHexo(1, device, capacity=2), Game()
                self.addCleanup(native.close)
                for point in base[:-1]:
                    action = transform(*point)
                    native.play(*action)
                    env.step(torch.tensor([action], device=device))
                self.assertEqual(native.player, 1)
                self.assertEqual(native.remaining, 2)
                self.assertEqual(native.winner, -1)
                last = transform(*base[-1])
                native.play(*last)
                result = env.step(torch.tensor([last], device=device))
                self.assertTrue(result.terminated.item())
                self.assertEqual(env.winner.item(), 1)
                self.assertEqual(env.remaining.item(), 1)
                self.assertEqual(env.features[0].cpu().tolist(), native.features())
                self.assertFalse(env.step(torch.tensor([[100, 100]], device=device)).accepted.item())


if __name__ == "__main__":
    unittest.main()
