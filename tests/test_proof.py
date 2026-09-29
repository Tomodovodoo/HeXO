import copy
import json
import time
import unittest
from unittest.mock import patch

from hexo import Game
from proof import PROVEN_LOSS, PROVEN_WIN, UNKNOWN, VerificationTimeout, solve, verify
from tests.reference import AXES, interleave


class ForcingProof(unittest.TestCase):
    def game(self, history=()):
        game = Game(history)
        self.addCleanup(game.close)
        return game

    def run_proof(self, game, **kwargs):
        before = game.state(), game.key
        result = solve(game, deadline=time.perf_counter()+1, **kwargs)
        self.assertEqual((game.state(), game.key), before)
        if result['certificate']:
            # Certificates must survive serialization, with no native state/cache.
            cert = json.loads(json.dumps(result['certificate']))
            self.assertEqual(verify(cert, [c[:2] for c in game.cells]), result['status'])
        return result

    def test_immediate_wins_axes_first_stone_overline_and_counterwin(self):
        for dq, dr in AXES:
            for offsets in ([0, 1, 2, 3, 4, 5], [0, 1, 2, 4, 5, 6, 8, 3]):
                shift = (0, 6) if dq else (6, 0)
                ours = [(k*dq, k*dr) for k in offsets]
                theirs = [(2*k*dq+shift[0], 2*k*dr+shift[1]) for k in range(len(ours))]
                history = interleave([ours, theirs])[:-1]
                result = self.run_proof(self.game(history))
                self.assertEqual(result['status'], PROVEN_WIN)
                self.assertEqual(len(result['certificate']['tree']['moves']), 1)
        # Both players threaten: the side to move wins before needing to block.
        history = interleave([[(q, 0) for q in range(5)], [(0, 2), (2, 2), (3, 2), (5, 2)]])
        result = self.run_proof(self.game(history))
        self.assertEqual(result['status'], PROVEN_WIN)
        self.assertEqual(len(result['certificate']['tree']['moves']), 2)
        bad = copy.deepcopy(result['certificate'])
        bad['attacker'] = 0
        bad['tree'] = {'kind': 'uncovered'}
        with self.assertRaises(ValueError):
            verify(bad, history)  # A defender counterwin invalidates a loss claim.

    def test_no_cover_loss_and_partial_turn(self):
        ours = [(q, r) for r in (0, 3) for q in range(4)] + [(10, 0)]
        theirs = [(6+(i % 3)*3, 3*(i//3)+1) for i in range(8)]
        game = self.game(interleave([ours, theirs]))
        self.assertEqual(game.player, 1)
        self.assertEqual(self.run_proof(game)['status'], PROVEN_LOSS)
        game.play(15, 7)
        self.assertEqual(game.remaining, 1)
        self.assertEqual(self.run_proof(game)['status'], PROVEN_LOSS)

    def fork_position(self, blocked=False):
        ours = [(q, r) for r in (0, 3, 6) for q in range(3)]
        theirs = [(6+(i % 3)*3, 3*(i//3)+1) for i in range(10)]
        if blocked:
            theirs = [(-1, 0)] + theirs[:-1]
        return self.game(interleave([ours, theirs]))

    def test_complete_and_tree_and_independent_verifier(self):
        game = self.fork_position()
        def proposals(width):
            moves = [(3, 0), (12, -3)] if len(game.cells) == 19 else [(3, 3), (3, 6)]
            return [{'moves': moves}]
        with patch.object(game, 'turns', proposals):
            result = self.run_proof(game)
        self.assertEqual(result['status'], PROVEN_WIN)
        cert = result['certificate']
        branches = cert['tree']['child']['branches']
        self.assertEqual(len(branches), 3)
        history = [c[:2] for c in game.cells]
        # A valid witness remains checkable even when solver helpers are broken.
        with patch('proof._completions', side_effect=AssertionError), patch('proof._covers', side_effect=AssertionError):
            self.assertEqual(verify(cert, history), PROVEN_WIN)
        for mutation in ('missing', 'duplicate', 'illegal', 'fake_leaf', 'history'):
            bad = copy.deepcopy(cert)
            children = bad['tree']['child']['branches']
            if mutation == 'missing':
                children.pop()
            elif mutation == 'duplicate':
                children.append(copy.deepcopy(children[0]))
            elif mutation == 'illegal':
                bad['tree']['moves'][1] = (10**6, 10**6)
            elif mutation == 'fake_leaf':
                children[0]['child'] = {'kind': 'terminal'}
            else:
                bad['history'][0] = [1, 0]
            with self.assertRaises(ValueError, msg=mutation):
                verify(bad, history)
        # Running out of time is not a rejection: a valid and an invalid certificate time out alike.
        for certificate in (cert, bad):
            with self.assertRaises(VerificationTimeout) as caught:
                verify(certificate, history, deadline=time.perf_counter())
            self.assertNotIsInstance(caught.exception, ValueError)

    def test_free_filler_quiet_and_depth_exhaustion_are_unknown(self):
        self.assertEqual(self.run_proof(self.game(), attack_turns=0)['status'], UNKNOWN)
        game = self.fork_position()
        # No selective attacker proposals is not a loss certificate.
        with patch.object(game, 'turns', return_value=[]):
            self.assertEqual(self.run_proof(game)['status'], UNKNOWN)
        # A singleton-cover defender node with a seductive fork next turn.
        game = self.fork_position(blocked=True)
        with patch.object(game, 'turns', return_value=[{'moves': [(3, 0), (5, 0)]}]):
            # Five stones with one internal gap has a singleton cover.
            self.assertEqual(self.run_proof(game)['status'], UNKNOWN)
        history = [c[:2] for c in game.cells]
        bad = {'version': 1, 'history': history, 'attacker': 0, 'tree': {
            'kind': 'move', 'moves': [(3, 0), (5, 0)], 'child': {
                'kind': 'defenses', 'branches': [{'moves': [(4, 0), (15, 7)],
                                                'child': {'kind': 'uncovered'}}]}}}
        with self.assertRaises(ValueError):
            verify(bad, history)  # Sampling just one free filler is not coverage.

    def test_native_proposals_and_deadline_node_budget_restore(self):
        history = interleave([[(0,0),(1,0),(2,0),(0,2),(1,2),(2,2),(10,1)],
                              [(0,6),(2,6),(4,6),(6,6),(8,6),(10,6)]])
        result = self.run_proof(self.game(history), width=8)
        self.assertEqual(result['status'], PROVEN_WIN)
        game = self.fork_position()
        before = game.state(), game.key
        for options in ({'deadline': time.perf_counter()-1},
                        {'deadline': time.perf_counter()+1, 'node_limit': 0},
                        {'deadline': time.perf_counter()+1, 'node_limit': 3}):
            result = solve(game, **options)
            self.assertEqual(result['status'], UNKNOWN)
            self.assertIsNone(result['certificate'])
            self.assertEqual((game.state(), game.key), before)
        # Native proposal generation is cooperative, not preemptible. A late
        # answer is discarded rather than mislabeled as meeting the deadline.
        def slow(width):
            time.sleep(.015)
            return []
        with patch.object(game, 'turns', slow):
            result = solve(game, deadline=time.perf_counter()+.005)
        self.assertEqual(result['reason'], 'deadline')
        self.assertGreater(result['elapsed_ms'], 5)

    def test_checker_rejects_wrong_phase_and_post_win_moves(self):
        history = interleave([[(q, 0) for q in range(5)], [(0, 2), (2, 2), (3, 2), (5, 2)]])
        game = self.game(history)
        cert = self.run_proof(game)['certificate']
        cert['tree']['moves'] = cert['tree']['moves'][:1]
        with self.assertRaises(ValueError):
            verify(cert, [c[:2] for c in game.cells])
        history = interleave([[(q, 0) for q in range(6)], [(2*q, 6) for q in range(6)]])[:-1]
        cert = self.run_proof(self.game(history))['certificate']
        cert['tree']['moves'].append((15, 7))
        with self.assertRaises(ValueError):
            verify(cert, history)


if __name__ == '__main__':
    unittest.main()
