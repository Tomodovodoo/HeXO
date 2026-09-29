"""CPU checks for the dense hex ResNet stack: hexcrop, hexnet, dense_config, dense_data, dense_bootstrap, the
learner's validation and the actor/evaluator engine."""
import argparse
import copy
import dataclasses
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
import time
from collections import Counter
from types import SimpleNamespace
import unittest
import unittest.mock

import numpy as np
import torch
from torch.nn import functional as F

from hexo import Game
import hexcrop
import hexnet
import dense_config
import dense_data
import dense_bootstrap
import dense_eval
import dense_learn
import dense_selfplay
from neural_search import NeuralSearch

ROOT = Path(__file__).resolve().parents[1]
TINY = hexnet.HexNetConfig(blocks=2, channels=16, pool_every=2, line_length=5, value_hidden=16, head_channels=8)
AXIAL_AXES = ((1, 0), (0, 1), (1, -1))


def legal(history):
    game = Game(history)
    try:
        return np.asarray(game.legal_moves(), np.int64).reshape(-1, 2)
    finally:
        game.close()


def random_game(rng, plies, local=True):
    """Random legal placements; `local` prefers cells within distance 2 of the last stone (compact games)."""
    game, history = Game(), []
    try:
        while len(history) < plies and game.winner < 0:
            moves = game.legal_moves()
            if local and history:
                q, r = history[-1]
                near = [m for m in moves if max(abs(m[0]-q), abs(m[1]-r), abs(m[0]-q+m[1]-r)) <= 2]
                moves = near or moves
            move = tuple(int(v) for v in moves[rng.integers(len(moves))])
            game.play(*move)
            history.append(move)
        return history, game.winner
    finally:
        game.close()


def line_history(stones, step=8):
    """Stones spaced `step` apart along q: legal, never a six, and as wide as needed."""
    return [(step*k, 0) for k in range(stones)]


def fixed_positions():
    rng = np.random.default_rng(7)
    positions = [[], [(0, 0)], [(0, 0), (1, 0)], [(0, 0), (8, 0), (0, 8)]]
    while len(positions) < 96:
        history, _ = random_game(rng, int(rng.integers(3, 40)), local=bool(rng.integers(2)))
        positions.append(history[:int(rng.integers(len(history)+1))])
    positions += [line_history(6), line_history(10), line_history(12), line_history(15)]
    return positions


def active(history):
    game = Game(history)
    try:
        return game.winner < 0
    finally:
        game.close()


POSITIONS = [h for h in fixed_positions() if active(h)]


class HexcropTests(unittest.TestCase):
    def test_legal_array_matches_engine_on_100_positions(self):
        self.assertGreaterEqual(len(POSITIONS), 95)
        sizes = set()
        for history in POSITIONS:
            game = Game(history)
            try:
                expected = np.asarray(game.legal_moves(), np.int64).reshape(-1, 2)
                got = hexcrop.legal_array(game, np.asarray(history, np.int64).reshape(-1, 2))
                np.testing.assert_array_equal(got, expected, err_msg=str(history))
                np.testing.assert_array_equal(hexcrop.native_legal(game), expected)
            finally:
                game.close()
            s = hexcrop.encode(history)
            np.testing.assert_array_equal(s.actions, expected)
            sizes.add(s.size)
        self.assertTrue(sizes & {64, 96}, sizes)
        self.assertEqual(hexcrop.encode(line_history(6)).size, 64)
        self.assertEqual(hexcrop.encode(line_history(10)).size, 96)

    def check_sample(self, history, s):
        n = len(history)
        game = Game(history)
        player, remaining = game.player, game.remaining
        game.close()
        self.assertEqual((s.player, s.remaining), (player, remaining))
        planes = s.planes.reshape(len(hexcrop.PLANES), -1)
        inside = s.cells >= 0
        self.assertEqual(int(inside.sum())+s.far, len(s.actions))
        self.assertEqual(int(planes[2].sum()), int(inside.sum()))
        for cell, action in zip(s.cells, s.actions):
            if cell >= 0:
                self.assertEqual(s.point(cell), tuple(int(v) for v in action))
                self.assertEqual((planes[0, cell], planes[1, cell], planes[2, cell], planes[3, cell]), (0, 0, 1, 1))
        owner = {tuple(m): ((i+1)//2) % 2 for i, m in enumerate(history)}
        for plane, mine in ((0, True), (1, False)):
            points = [s.point(i) for i in np.flatnonzero(planes[plane])]
            self.assertEqual(sorted(points), sorted(p for p, o in owner.items() if (o == player) == mine))
        turn = [s.point(i) for i in np.flatnonzero(planes[6])]
        self.assertEqual(turn, [tuple(history[-1])] if remaining == 1 and n > 0 else [])
        start = n if remaining == 2 or n == 0 else n-1
        previous = sorted(s.point(i) for i in np.flatnonzero(planes[7]))
        self.assertEqual(previous, sorted(tuple(m) for m in history[max(0, start-2):start]))
        self.assertTrue(np.all(planes[4] == (remaining == 1)) and np.all(planes[5] == (remaining == 2)))
        # Stones always lie on the crop plane; padding carries nothing.
        stones = planes[0] | planes[1]
        self.assertTrue(np.all(planes[3][stones > 0] == 1))
        for c in (0, 1, 2, 6, 7):
            self.assertTrue(np.all(planes[c][planes[3] == 0] == 0))

    def test_every_symmetry_encodes_the_same_physical_cells(self):
        for history in POSITIONS[:24]+POSITIONS[-4:-1]:
            for k in range(12):
                s = hexcrop.encode(history, symmetry=k)
                self.assertEqual(s.symmetry, k)
                self.check_sample(history, s)

    def test_line_axes_and_neighbours_map_to_index_directions(self):
        index_axes = {(1, 0): 0, (0, 1): 1, (1, -1): 2}
        neighbours = {(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)}
        for k in range(12):
            m = hexcrop.SYMMETRIES[k]
            images = []
            for d in AXIAL_AXES:
                x, y = (int(v) for v in np.array(d) @ m)
                axis = index_axes.get((x, y), index_axes.get((-x, -y)))
                self.assertIsNotNone(axis, (k, d, (x, y)))
                images.append(axis)
            self.assertEqual(sorted(images), [0, 1, 2], k)
            self.assertEqual({tuple(int(v) for v in np.array(d) @ m) for d in neighbours}, neighbours)
            np.testing.assert_array_equal(m @ hexcrop.INVERSES[k], np.eye(2, dtype=np.int64))
        # hexnet's line axes and HexConv mask agree with the same index geometry.
        self.assertEqual(set(hexnet.AXES), set(index_axes))
        conv = hexnet.HexConv(1, 1)
        taps = {(ky-1, kx-1) for ky in range(3) for kx in range(3) if conv.hex[0, 0, ky, kx]}
        self.assertEqual(taps-{(0, 0)}, {(dy, dx) for dx, dy in neighbours})

    def test_symmetry_choice_is_deterministic_without_rng(self):
        for history in POSITIONS[4:30]:
            a, b = hexcrop.encode(history), hexcrop.encode(history)
            self.assertEqual(a.symmetry, b.symmetry)
            np.testing.assert_array_equal(a.planes, b.planes)
            moves = np.asarray(history, np.int64).reshape(-1, 2)
            points = np.concatenate((moves, legal(history)))
            sides = hexcrop._sides(points)
            self.assertEqual(a.symmetry, int(np.argmin(sides)))
            rng = np.random.default_rng(3)
            seen = set()
            for _ in range(24):
                s = hexcrop.encode(history, rng=rng)
                self.assertEqual(s.size, a.size)
                seen.add(s.symmetry)
            self.assertGreater(len(seen), 1)

    def test_far_mode_covers_every_legal_cell(self):
        self.assertEqual(hexcrop.encode(line_history(15)).size, 192)
        history = line_history(31)          # span 240: legal cells need 257, stones plus halo 249
        for kwargs in ({}, {'rng': np.random.default_rng(1)}, {'symmetry': 5}):
            s = hexcrop.encode(history, **kwargs)
            self.assertEqual(s.size, 256)
            self.assertGreater(s.far, 0)
            self.check_sample(history, s)
            qmin, rmin, ox, oy = s.offset
            xy = s.actions @ hexcrop.SYMMETRIES[s.symmetry]+(ox-qmin, oy-rmin)
            crop = s.planes[3]
            inside = (xy >= 0).all(1) & (xy < s.size).all(1)
            inside[inside] = crop[xy[inside, 1], xy[inside, 0]] > 0
            np.testing.assert_array_equal(inside, s.cells >= 0)
        with self.assertRaises(hexcrop.SpanError):
            hexcrop.encode(line_history(33))  # stones plus halo exceed the largest bucket

    def test_point_inverts_every_crop_index(self):
        for history in (POSITIONS[20], line_history(6)):
            for k in (0, 3, 7, 11):
                s = hexcrop.encode(history, symmetry=k)
                qmin, rmin, ox, oy = s.offset
                for index in np.flatnonzero(s.planes[3].reshape(-1))[::7]:
                    q, r = s.point(index)
                    x, y = np.array([q, r]) @ hexcrop.SYMMETRIES[k]+(ox-qmin, oy-rmin)
                    self.assertEqual(y*s.size+x, index)

    def test_batch_pads_cells(self):
        samples = [hexcrop.encode(h) for h in POSITIONS[:40]]
        size, indices = next(iter(hexcrop.group_by_size(samples).items()))
        b = hexcrop.batch([samples[i] for i in indices])
        for row, i in enumerate(indices):
            n = len(samples[i].cells)
            self.assertEqual(b['counts'][row], n)
            np.testing.assert_array_equal(b['cells'][row, :n], samples[i].cells)
            self.assertTrue(np.all(b['cells'][row, n:] == -1))


def line_reference(own, opp, mask):
    """Slow LineFeatures over [S, S] 0/1 arrays indexed [y, x]."""
    size = own.shape[0]
    out = np.zeros((10, size, size), np.float32)
    best = np.zeros((2, 3, size, size))
    for a, (dx, dy) in enumerate(hexnet.AXES):
        for y in range(size):
            for x in range(size):
                for i in range(hexnet.WINDOW):
                    cells = [(y+(j-i)*dy, x+(j-i)*dx) for j in range(hexnet.WINDOW)]
                    if any(not (0 <= cy < size and 0 <= cx < size) or not mask[cy, cx] for cy, cx in cells):
                        continue
                    o, p = sum(own[c] for c in cells), sum(opp[c] for c in cells)
                    if p == 0:
                        best[0, a, y, x] = max(best[0, a, y, x], o)
                    if o == 0:
                        best[1, a, y, x] = max(best[1, a, y, x], p)
    out[:6] = best.reshape(6, size, size)/hexnet.WINDOW
    mine, theirs = best[0].max(0), best[1].max(0)
    empty = mask*(1-own-opp)
    for k, value in enumerate((mine >= 4, theirs >= 4, mine >= 5, theirs >= 5)):
        out[6+k] = value*empty
    return out


def line_conv_reference(x, weight):
    c, length = weight.shape[0], weight.shape[2]
    h, w = x.shape[-2:]
    out = torch.zeros_like(x)
    for a, (dx, dy) in enumerate(hexnet.AXES):
        for i in range(length):
            oy, ox = (i-length//2)*dy, (i-length//2)*dx
            shifted = torch.zeros_like(x)
            ys, xs = slice(max(0, -oy), min(h, h-oy)), slice(max(0, -ox), min(w, w-ox))
            shifted[..., ys, xs] = x[..., max(0, oy):min(h, h+oy), max(0, ox):min(w, w+ox)]
            out += weight[:, a, i][:, None, None]*shifted
    return out


def same_bucket(n, size=24):
    return [h for h in POSITIONS if hexcrop.encode(h).size == size][:n]


def planes_batch(histories):
    samples = [hexcrop.encode(h) for h in histories]
    b = hexcrop.batch(samples)
    return samples, b, torch.from_numpy(b['planes']).float()


class HexNetTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.threads)
        torch.manual_seed(1740)

    def test_hexconv_masks_distance_two_taps(self):
        conv = hexnet.HexConv(3, 4)
        x = torch.randn(2, 3, 7, 7, requires_grad=True)
        out = conv(x)
        out.square().sum().backward()
        effective = conv.weight*conv.hex
        for ky, kx in ((0, 0), (2, 2)):
            self.assertTrue(torch.all(effective[..., ky, kx] == 0))
            self.assertTrue(torch.all(conv.weight.grad[..., ky, kx] == 0))
        self.assertTrue(torch.all(conv.weight.grad[..., 0, 2] != 0))
        torch.testing.assert_close(out, F.conv2d(x, effective, padding=1))

    def test_line_features_match_reference(self):
        size = 16
        own, opp, mask = np.zeros((3, size, size), np.float32)
        mask[1:15, 2:14] = 1
        own[8, 4:8] = 1                      # open four along (1, 0)
        for i in range(5):
            opp[13-i, 3+i] = 1               # open five along (1, -1)
        own[3:6, 10] = 1                     # three along (0, 1)
        opp[6, 5] = 1
        cases = [(own, opp, mask)]
        rng = np.random.default_rng(5)
        for _ in range(2):
            m = np.zeros((size, size), np.float32)
            m[rng.integers(0, 3):rng.integers(12, 17), rng.integers(0, 3):rng.integers(12, 17)] = 1
            stones = rng.random((size, size))
            cases.append(((stones < .2)*m, ((stones > .2) & (stones < .35))*m, m))
        lines = hexnet.LineFeatures()
        for own, opp, mask in cases:
            t = [torch.from_numpy(np.ascontiguousarray(v, np.float32))[None, None] for v in (own, opp, mask)]
            got = lines(*t)[0].numpy()
            np.testing.assert_allclose(got, line_reference(own, opp, mask), atol=1e-6)
        got = lines(*[torch.from_numpy(v)[None, None] for v in (cases[0])])[0].numpy()
        self.assertEqual(got[6, 8, 3], 1)    # empty end of the open four
        self.assertEqual(got[6, 8, 9], 1)
        self.assertEqual(got[9, 14, 2], 1)   # empty end of the opponent five
        self.assertEqual(got[9, 8, 8], 1)
        self.assertEqual(got[8, 8, 3], 0)    # own four is not a five

    def test_line_conv_matches_direct_sum(self):
        conv = hexnet.LineConv(3, 5)
        with torch.no_grad():
            conv.weight.normal_()
        x = torch.randn(2, 3, 9, 9)          # crops are square
        expected = line_conv_reference(x, conv.weight.detach())
        torch.testing.assert_close(conv(x), expected, atol=1e-5, rtol=1e-5)
        with torch.no_grad():
            torch.testing.assert_close(conv(x), expected, atol=1e-5, rtol=1e-5)
            conv.weight.mul_(2)
            torch.testing.assert_close(conv(x), 2*expected, atol=1e-5, rtol=1e-5)   # matrices follow updates
            wide = torch.randn(1, 3, 130, 130)
            torch.testing.assert_close(conv(wide), line_conv_reference(wide, conv.weight), atol=1e-5, rtol=1e-5)

    def test_line_conv_gradients_match_direct_sum(self):
        conv = hexnet.LineConv(3, 5)
        with torch.no_grad():
            conv.weight.normal_()
        x = torch.randn(4, 3, 9, 9, dtype=torch.float64, requires_grad=True)
        grad = torch.randn(4, 3, 9, 9, dtype=torch.float64)
        conv.double()
        got = torch.autograd.grad(conv(x), (x, conv.weight), grad)
        expected = torch.autograd.grad(line_conv_reference(x, conv.weight), (x, conv.weight), grad)
        for g, e in zip(got, expected):
            torch.testing.assert_close(g, e)

    def test_line_conv_under_autocast_with_float_input(self):
        conv = hexnet.LineConv(3, 5)
        with torch.no_grad():
            conv.weight.normal_()
        x = torch.randn(2, 3, 9, 9)
        expected = line_conv_reference(x, conv.weight.detach())
        for grad in (True, False):
            with torch.set_grad_enabled(grad), torch.autocast('cpu', torch.bfloat16):
                got = conv(x)
            self.assertEqual(got.dtype, torch.bfloat16)
            torch.testing.assert_close(got.float(), expected, atol=.1, rtol=.02)

    def test_line_conv_inference_matrices_match_training_matrices(self):
        conv = hexnet.LineConv(4, 11)
        with torch.no_grad():
            conv.weight.normal_()
        for size in (7, 11, 24, 48, 131):
            trained = conv.matrices(size, torch.float32)       # grad enabled: the _toeplitz path
            with torch.no_grad():
                gathered = conv.matrices(size, torch.float32)
                for old, new in zip(trained, gathered):
                    self.assertEqual(new.shape, (4, size, size))
                    torch.testing.assert_close(new, old.detach(), atol=1e-5, rtol=0)
                x = torch.randn(5, 4, size, size)
                expected = line_conv_reference(x, conv.weight)
                torch.testing.assert_close(conv(x), expected, atol=1e-5, rtol=1e-5)
                with unittest.mock.patch.object(hexnet, 'LINE_CHUNK_CELLS', 2*size*size):    # chunks of 2, 2, 1
                    y = x.clone()
                    self.assertIs(conv.add_to(y), y)
                torch.testing.assert_close(y, x+expected, atol=1e-5, rtol=1e-5)

    def test_inference_forward_matches_training_path(self):
        model = hexnet.HexNet(TINY).eval()
        planes = torch.from_numpy(np.stack([hexcrop.encode(h).planes for h in same_bucket(3)])).float()
        with torch.no_grad():
            expected = model(planes, planes[:, 3:4])
        for key, value in model(planes, planes[:, 3:4]).items():   # grad enabled: full-size masks, no chunks
            torch.testing.assert_close(value.detach(), expected[key], atol=1e-5, rtol=1e-5)

    def test_masked_norm_ignores_padding(self):
        norm = hexnet.MaskedNorm(4).double()
        with torch.no_grad():
            norm.weight.uniform_(.5, 2)
            norm.bias.normal_()
        x = (torch.randn(3, 4, 6, 6, dtype=torch.float64)*torch.tensor([1., 2, 3, 4], dtype=torch.float64)[:, None, None]
             + torch.tensor([0., 5, -3, 20], dtype=torch.float64)[:, None, None])
        mask = torch.zeros(3, 1, 6, 6, dtype=torch.float64)
        mask[0, :, :4, :5] = mask[1, :, 1:, :] = mask[2, :, 2:5, 1:4] = 1
        x = torch.where(mask > 0, x, 1e3).requires_grad_()
        cells = mask.sum()
        full = mask.expand_as(x).contiguous()
        y = norm(x, full, cells)
        xr = x.detach().clone().requires_grad_()
        mean = (xr*mask).sum((0, 2, 3))/cells
        var = (((xr-mean[:, None, None])**2)*mask).sum((0, 2, 3))/cells
        reference = (xr-mean[:, None, None])/torch.sqrt(var[:, None, None]+norm.eps)*norm.weight[:, None, None]+norm.bias[:, None, None]
        torch.testing.assert_close(y, reference)
        m = norm.momentum
        torch.testing.assert_close(norm.running_mean, m*mean.detach())
        torch.testing.assert_close(norm.running_var, (1-m)+m*var.detach()*cells/(cells-1))
        g = torch.randn_like(y)*mask                    # blocks zero the gradient on padding
        (y*g).sum().backward()
        (reference*g).sum().backward()
        torch.testing.assert_close(x.grad, xr.grad)
        norm.eval()
        torch.testing.assert_close(norm(x.detach(), full, cells), F.batch_norm(
            x.detach(), norm.running_mean, norm.running_var, norm.weight, norm.bias, False, 0., norm.eps))

    def test_masked_norm_cumulative_statistics_weight_cells(self):
        """momentum None: a 1-row and a 7-row bucket give the statistics of one pass over all their cells."""
        x = torch.randn(8, 3, 5, 5, dtype=torch.float64)*torch.tensor([1., 3, .5], dtype=torch.float64)[:, None, None]
        x[0] += 10
        mask = (torch.rand(8, 1, 5, 5) < .7).double()
        full = mask.expand_as(x).contiguous()

        def stats(parts):
            norm = hexnet.MaskedNorm(3).double().train()
            norm.momentum = None
            with torch.no_grad():
                for k in parts:
                    norm(x[k], full[k], mask[k].sum())
            return norm.running_mean, norm.running_var
        split, whole = stats([slice(0, 1), slice(1, 8)]), stats([slice(0, 8)])
        torch.testing.assert_close(split, whole)
        cells = mask.sum()
        mean = (x*mask).sum((0, 2, 3))/cells
        torch.testing.assert_close(whole, (mean, (((x-mean[:, None, None])**2)*mask).sum((0, 2, 3))/(cells-1)))

    def test_masked_norm_bf16_statistics(self):
        x = (torch.randn(8, 4, 16, 16)+torch.tensor([0., 5, 20, 60])[:, None, None]).bfloat16()
        mask = torch.ones_like(x)
        mask[..., 12:, :] = 0
        cells = mask[:, :1].sum()
        _, mean, var = hexnet._MaskedBatchNorm.apply(x, mask, torch.ones(4), torch.zeros(4), cells, 1e-5)
        xd = x.double()
        ref_mean = (xd*mask).sum((0, 2, 3))/cells
        ref_var = (((xd-ref_mean[:, None, None])**2)*mask).sum((0, 2, 3))/cells
        torch.testing.assert_close(var.double(), ref_var, rtol=2e-3, atol=1e-4)

    def test_forward_shapes_for_two_buckets(self):
        model = hexnet.HexNet(TINY)
        for history, size in ((POSITIONS[4], None), (line_history(6), 64)):
            samples, b, planes = planes_batch([history, history])
            if size:
                self.assertEqual(b['size'], size)
            s = b['size']
            for training in (True, False):
                out = model.train(training)(planes, planes[:, 3:4])
                self.assertEqual(out['policy'].shape, (2, s*s))
                self.assertEqual(out['far'].shape, (2,))
                self.assertEqual(out['value_logit'].shape, (2,))
                self.assertEqual(out['short_value_logit'].shape, (2,))
                self.assertEqual(out['future'].shape, (2, 2, s, s))
                self.assertEqual(out['opponent_policy'].shape, (2, s*s))
                self.assertTrue(all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in out.values()))
            self.assertEqual(set(model(planes, planes[:, 3:4], aux=False)), {'policy', 'far', 'value_logit'})

    def cross_entropy_case(self):
        policy, far = torch.randn(3, 16, requires_grad=True), torch.randn(3, requires_grad=True)
        cells = torch.tensor([[3, 7, -1, 12, -1], [0, 5, -1, -1, -1], [9, 2, -1, -1, -1]])
        counts = torch.tensor([5, 3, 2])
        target = torch.rand(3, 5)*(torch.arange(5) < counts[:, None])
        return policy, far, cells, counts, target/target.sum(1, keepdim=True)

    def test_policy_loss_matches_manual_cross_entropy(self):
        policy, far, cells, counts, target = self.cross_entropy_case()
        weight = torch.tensor([1., .5, 2.])
        for use_far in (True, False):
            losses = []
            for b in range(3):
                n, nfar = int(counts[b]), int((cells[b, :counts[b]] < 0).sum())
                logits, probs = [], []
                for i in range(n):
                    c = int(cells[b, i])
                    if c < 0 and not use_far:
                        continue
                    logits.append(far[b]-math.log(nfar) if c < 0 else policy[b, c])
                    probs.append(target[b, i])
                log_probs = torch.stack(logits).log_softmax(0)
                losses.append(-(torch.stack(probs)*log_probs).sum())
            losses = torch.stack(losses)
            f = far if use_far else None
            got = hexnet.policy_loss(policy, f, cells, counts, target)
            torch.testing.assert_close(got, losses.mean())
            torch.testing.assert_close(hexnet.policy_loss(policy, f, cells, counts, target, weight),
                                       (weight*losses).sum()/weight.sum())
            if not use_far:
                torch.testing.assert_close(hexnet.opponent_policy_loss(policy, cells, counts, target), losses.mean())
        logits, valid = hexnet.action_logits(policy, far, cells, counts)
        self.assertEqual(valid.tolist(), (torch.arange(5) < counts[:, None]).tolist())
        # Far cells together carry exp(far) of softmax mass.
        torch.testing.assert_close(torch.logsumexp(logits[0][cells[0] < 0], 0), far[0])

    def test_zero_weights_give_zero_loss_and_gradients(self):
        policy, far, cells, counts, target = self.cross_entropy_case()
        cells[2] = torch.tensor([-1, -1, -1, -1, -1])     # far-only row: no entry without a far logit
        value = torch.randn(3, requires_grad=True)
        future = torch.randn(3, 2, 4, 4, requires_grad=True)
        mask = (torch.rand(3, 1, 4, 4) > .3).float()
        zero = torch.zeros(3)
        losses = [hexnet.policy_loss(policy, far, cells, counts, target, zero),
                  hexnet.opponent_policy_loss(policy, cells, counts, target, zero),
                  hexnet.value_loss(value, torch.rand(3), zero), hexnet.short_value_loss(value, torch.rand(3), zero),
                  hexnet.future_loss(future, torch.rand(3, 2, 4, 4), mask, torch.zeros(3, 2))]
        for loss in losses:
            self.assertEqual(float(loss), 0.)
        sum(losses).backward()
        for tensor in (policy, far, value, future):
            self.assertTrue(torch.all(tensor.grad == 0), tensor.grad)
        # A far-only row without a far logit is 0, not NaN, even with weight.
        self.assertTrue(torch.isfinite(hexnet.opponent_policy_loss(policy, cells, counts, target)))

    def test_save_load_round_trip_and_digest(self):
        model = hexnet.HexNet(TINY)
        _, _, planes = planes_batch(same_bucket(4))
        model.train()(planes, planes[:, 3:4])            # nontrivial running statistics
        model.eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pt'
            hexnet.save_model(path, model)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ['model.pt'])
            loaded = hexnet.load_model(path)
            hexnet.save_model(path, loaded)                 # overwrite in place works
        loaded.eval()
        a, b = model(planes, planes[:, 3:4]), loaded(planes, planes[:, 3:4])
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)
        digest = hexnet.model_digest(model)
        self.assertEqual(digest, hexnet.model_digest(loaded))
        self.assertEqual(digest, hexnet.model_digest(copy.deepcopy(model).to(memory_format=torch.channels_last)))
        with torch.no_grad():
            loaded.blocks[0].conv1.weight[0, 0, 1, 1] += 1e-3
        self.assertNotEqual(hexnet.model_digest(loaded), digest)
        self.assertNotEqual(hexnet.model_digest(hexnet.HexNet(replace(TINY, aux_heads=False))), hexnet.model_digest(
            hexnet.HexNet(TINY)))

    def test_checkpoint_without_aux_heads(self):
        plain = hexnet.HexNet(replace(TINY, aux_heads=False)).eval()
        _, _, planes = planes_batch(same_bucket(2))
        expected = plain(planes, planes[:, 3:4])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'plain.pt'
            hexnet.save_model(path, plain)
            loaded = hexnet.load_model(path).eval()
            self.assertFalse(loaded.config.aux_heads)
            self.assertEqual(set(loaded(planes, planes[:, 3:4])), {'policy', 'far', 'value_logit'})
            # A checkpoint whose config predates aux_heads loads into the default (aux) config;
            # the aux heads stay freshly initialised and the shared outputs are unchanged.
            data = torch.load(path, weights_only=True)
            del data['config']['aux_heads']
            torch.save(data, path)
            upgraded = hexnet.load_model(path).eval()
            self.assertTrue(upgraded.config.aux_heads)
            out = upgraded(planes, planes[:, 3:4])
            for key in expected:
                self.assertTrue(torch.equal(out[key], expected[key]), key)
            self.assertIn('future', out)
            # Extra weights are still refused.
            data['state']['surplus'] = torch.zeros(1)
            torch.save(data, path)
            with self.assertRaises(ValueError):
                hexnet.load_model(path)

    def test_pointwise_model_is_symmetry_invariant(self):
        model = pointwise_model(TINY)
        for history in POSITIONS[8:16]+[line_history(6)]:
            values = []
            for k in range(12):
                s = hexcrop.encode(history, symmetry=k)
                planes = torch.from_numpy(s.planes[None]).float()
                values.append(float(model(planes, planes[:, 3:4], aux=False)['value_logit'].detach()))
            self.assertLess(max(values)-min(values), 1e-5, (history, values))


def pointwise_model(config):
    """HexNet whose value is invariant under every hex symmetry: centre taps only, axis-tied line inputs,
    no line convolutions, no mean pooling. Any symmetry spread then comes from the encoder."""
    model = hexnet.HexNet(config).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, hexnet.HexConv):
                centre = module.weight[..., 1, 1].clone()
                module.weight.zero_()
                module.weight[..., 1, 1] = centre
            if isinstance(module, hexnet.LineConv):
                module.weight.zero_()
            if isinstance(module, hexnet.MaskedNorm):
                module.running_mean.normal_()
                module.running_var.uniform_(.5, 2)
        base = len(hexcrop.PLANES)
        w = model.stem.weight
        for side in (0, 3):
            w[:, base+side:base+side+3] = w[:, base+side:base+1+side].clone()
        for block in model.blocks:
            if block.pool is not None:
                block.pool.weight.zero_()
                block.pool.bias.zero_()
        model.value_hidden.weight[:, :config.channels] = 0
    return model


class DenseConfigTests(unittest.TestCase):
    def test_pages_serve_from_step_control(self):
        """The project page and a dense run page carry the "from step" header control."""
        import dashboard
        from http.server import ThreadingHTTPServer
        import threading
        import urllib.request
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'r').mkdir()
            (Path(tmp)/'r'/'config.json').write_text('{}')
            handler = type('Handler', (dashboard.Handler,), dict(runs=Path(tmp)))
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                for query in ('', '?run=r'):
                    with urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/{query}') as response:
                        page = response.read().decode()
                    self.assertIn('from step <input id="from-step" type="number"', page)
            finally:
                server.shutdown()
                server.server_close()

    def test_metrics_log_series(self):
        """Partial last lines are skipped until completed, resumed steps replace the rewound ones and
        downsampling keeps the first, last and extreme points."""
        import dashboard
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for step in (1, 2, 3, 2, 3, 4):
                dense_config.append_metrics(run, 'learner-main', step=step, policy_ce=float(step))
            path = run/'metrics'/'learner-main.jsonl'
            with path.open('ab') as stream:
                stream.write(b'{"time": 1, "step": 5, "policy')
            config = dict(created_at=0.)
            self.assertEqual(dashboard.series(run, config, 'main', 'policy_ce')['points'], [[1, 1.], [2, 2.], [3, 3.], [4, 4.]])
            with path.open('ab') as stream:
                stream.write(b'_ce": 5.0}\n')
            self.assertEqual(dashboard.series(run, config, 'main', 'policy_ce')['points'][-1], [5, 5.])
            self.assertEqual(dashboard.series(run, config, 'main', 'policy_ce', from_step=4)['points'], [[4, 4.], [5, 5.]])
            self.assertEqual(len(dashboard.series(run, config, 'main', 'policy_ce', 'hours', from_step=4)['points']), 5)
            points = [(x, 100. if x == 777 else math.sin(x)) for x in range(5000)]
            kept = dashboard.downsample(points, 100)
            self.assertLessEqual(len(kept), 100)
            self.assertEqual((kept[0], kept[-1]), (points[0], points[-1]))
            self.assertIn((777, 100.), kept)

    def test_provisional_league_row(self):
        """The league row of an unrated candidate under evaluation comes from the status tally, offset by the
        opponent's league Elo (Seal: its anchor Elo); rated candidates and idle evaluators have none."""
        import dashboard
        league = dict(checkpoints=[dict(id='main/000010', elo=12.)], anchors=dict(seal=dict(elo=-300.)))
        tally = dict(wins=5, losses=2, capped=1, games=8, elo_delta=40., elo_interval=[-10., 90.])
        status = dict(stage='playing', comparison=dict(candidate='main/000020', opponent='main/000010', kind='champion'),
                      games_planned=200, tally=tally)
        self.assertEqual(dashboard.provisional(league, status), dict(
            id='main/000020', opponent='main/000010', wins=5, losses=2, capped=1, games=8, games_planned=200, elo=52.,
            elo_interval=[2., 102.]))
        seal = dashboard.provisional(league, dict(status, stage='throttled', comparison=dict(candidate='main/000020', opponent='seal')))
        self.assertEqual((seal['elo'], seal['elo_interval']), (-260., [-310., -210.]))
        unknown = dashboard.provisional(league, dict(status, tally=dict(tally, elo_delta=None, elo_interval=None)))
        self.assertEqual((unknown['elo'], unknown['elo_interval'], unknown['wins']), (None, None, 5))
        self.assertIsNone(dashboard.provisional(league, dict(status, comparison=dict(candidate='main/000010', opponent='seal'))))
        self.assertIsNone(dashboard.provisional(league, dict(status, stage='idle')))
        self.assertIsNone(dashboard.provisional(league, dict(status, tally=None)))
        variant = dict(status, comparison=dict(candidate='main/000010@solver', opponent='main/000010', kind='variant'))
        league['variants'] = [dict(id='main/000010@solver', checkpoint='main/000010', elo=30.)]
        self.assertEqual(dashboard.provisional(league, variant)['elo'], 52.)        # pending: live against its checkpoint
        league['variants'][0]['verdict'] = dict(decision='better')
        self.assertIsNone(dashboard.provisional(league, variant))

    def test_tally(self):
        game = lambda pair, colour, winner: dict(seed=pair, challenger_color=colour, winner=winner)
        records = [game(1, 0, 0), game(1, 1, -1), game(2, 0, 1), game(2, 1, 1), game(3, 0, 0)]  # pair 3 unfinished
        t = dense_eval.tally(records, lambda complete: dict(llr=len(complete), bound_lower=-1., bound_upper=1.))
        self.assertEqual({k: t[k] for k in ('wins', 'losses', 'capped', 'games', 'pairs', 'pair_score', 'llr')},
                         dict(wins=3, losses=1, capped=1, games=5, pairs=2, pair_score=.625, llr=4))
        half = math.sqrt(math.log(40)/4)
        self.assertEqual(t['pair_interval'], [max(0., .625-half), min(1., .625+half)])
        self.assertAlmostEqual(t['elo_delta'], 400*math.log10(3/2))
        self.assertIsNone(dense_eval.tally(records[:1])['pair_score'])
        self.assertIsNone(dense_eval.tally(records)['llr'])

    def test_configs_with_retired_settings_load(self):
        data = asdict(dense_config.RunConfig(created_at=1.))
        data['evaluation'].update(round_games=8, model_cache=6, uncertainty_parity=1.5)
        self.assertEqual(dense_config.from_dict(data), dense_config.RunConfig(created_at=1.))
        data['evaluation']['typo_games'] = 1
        with self.assertRaises(TypeError):
            dense_config.from_dict(data)

    def test_posterior_judge_needs_only_leadership_games_and_confidence(self):
        """The live main/032500 vs main/030000 verdict (100 direct games, P(better) .99994, the pooled leader, rating
        sd 56 against the champion's 33) promotes: the candidate's sd is no condition. So does a small real edge
        once P(better) clears promote_confidence."""
        s = dense_config.EvaluationSettings()
        self.assertEqual(dense_eval.judge(s, 100, False, True, .99994), 'promote')
        self.assertEqual(dense_eval.judge(s, 200, False, True, .91), 'promote')
        self.assertIsNone(dense_eval.judge(s, 200, False, True, .89))
        self.assertEqual(dense_eval.judge(s, 100, False, False, .01), 'reject')
        self.assertIsNone(dense_eval.judge(s, 100, False, False, .99994))     # not the leader
        self.assertIsNone(dense_eval.judge(s, 62, False, True, .99994))       # too few direct games
        self.assertIsNone(dense_eval.judge(s, 100, True, True, .99994))       # direct and pooled disagree

    def test_round_trip_and_no_overwrite(self):
        config = dense_config.RunConfig(created_at=12.5, seed=3, device='cpu',
                                        model=dense_config.ModelSettings(blocks=2, aux_heads=False),
                                        learner=dense_config.LearnerSettings(lr=1e-3, recency=.5))
        with tempfile.TemporaryDirectory() as tmp:
            dense_config.save(tmp, config)
            self.assertEqual(dense_config.load(tmp), config)
            with self.assertRaises(FileExistsError):
                dense_config.save(tmp, replace(config, seed=4))
            self.assertEqual(dense_config.load(tmp).seed, 3)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ['config.json'])
        with self.assertRaises(ValueError):
            dense_config.from_dict(dict(asdict(config), schema='other'))
        # ModelSettings and HexNetConfig must stay field-for-field identical.
        self.assertEqual(asdict(dense_config.ModelSettings()), asdict(hexnet.HexNetConfig()))

    def test_flags_override_bool_and_float(self):
        parser = argparse.ArgumentParser()
        dense_config.add_arguments(parser, dense_config.ActorSettings)
        base = dense_config.ActorSettings()
        self.assertEqual(dense_config.override(base, parser.parse_args([])), base)
        args = parser.parse_args(['--no-tactics', '--full-fraction', '0.5', '--full-sims', '32'])
        got = dense_config.override(base, args)
        self.assertEqual((got.tactics, got.full_fraction, got.full_sims), (False, .5, 32))
        self.assertIsInstance(got.full_fraction, float)
        self.assertTrue(dense_config.override(got, parser.parse_args(['--tactics'])).tactics)
        self.assertEqual(dense_selfplay.actor_flags(parser.parse_args(['--historical-weighting', 'uniform'])),
                         ['--historical-weighting', 'uniform'])
        prefixed = argparse.ArgumentParser()
        dense_config.add_arguments(prefixed, dense_config.ActorSettings)
        dense_config.add_arguments(prefixed, dense_config.EvaluationSettings, 'eval_')
        args = prefixed.parse_args(['--root-samples', '8', '--eval-root-samples', '4', '--no-eval-tactics'])
        self.assertEqual(dense_config.override(base, args).root_samples, 8)
        evaluation = dense_config.override(dense_config.EvaluationSettings(), args, 'eval_')
        self.assertEqual((evaluation.root_samples, evaluation.tactics), (4, False))

    def test_learner_target_and_validation_flags(self):
        parser = argparse.ArgumentParser()
        dense_config.add_arguments(parser, dense_config.LearnerSettings)
        base = dense_config.LearnerSettings()
        self.assertEqual((base.value_target, base.outcome_lambda, base.outcome_weight, base.calibration_games, base.validation_rows,
                          base.validation_quota), ('outcome', .98, 0., 4000, 8192, 128))
        fit = dense_data.Calibration((0.,)*dense_data.CALIBRATION_FEATURES, .5)
        pick = lambda s, c=fit: {k: v for k, v in dense_data.target_options(s, c).items() if k in ('outcome_lam', 'calibration', 'full_only')}
        self.assertEqual(pick(base), dict(outcome_lam=1., calibration=None, full_only=False))
        args = parser.parse_args(['--value-target', 'td', '--outcome-lambda', '0.95', '--bootstrap-full-only', '--outcome-weight', '0.5',
                                  '--validation-rows', '4096', '--validation-quota', '64', '--calibration-games', '1000'])
        got = dense_config.override(base, args)
        self.assertEqual((got.value_target, got.outcome_lambda, got.bootstrap_full_only, got.outcome_weight, got.validation_rows,
                          got.validation_quota, got.calibration_games), ('td', .95, True, .5, 4096, 64, 1000))
        self.assertEqual(pick(got), dict(outcome_lam=.95, calibration=None, full_only=True))
        calibrated = dense_config.override(got, parser.parse_args(['--value-target', 'calibrated']))
        self.assertEqual(pick(calibrated), dict(outcome_lam=1., calibration=fit, full_only=True))
        self.assertEqual(pick(calibrated, None), dict(outcome_lam=1., calibration=None, full_only=True))
        with self.assertRaises(ValueError):
            dense_config.override(base, parser.parse_args(['--value-target', 'soft']))
        # Manifests written before these settings load with the defaults.
        old = {k: v for k, v in asdict(base).items() if k not in ('value_target', 'outcome_lambda', 'outcome_weight', 'calibration_games',
                                                                   'validation_rows', 'validation_quota')}
        self.assertEqual(dense_config.LearnerSettings(**old), base)
        rng = np.random.default_rng(0)
        self.assertEqual(dense_learn.perturb(replace(base, outcome_lambda=1.), rng, .2).outcome_lambda, 1.)
        for _ in range(20):
            x = dense_learn.perturb(base, rng, .2).outcome_lambda
            self.assertTrue(.98-.02*.2-1e-12 <= x <= .98+.02*.2+1e-12)
        self.assertIn('validation_rows', dense_learn.KEEP)

    def test_command_line_creates_a_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/'run'
            done = subprocess.run([sys.executable, str(ROOT/'dense_config.py'), '--run', str(run), '--device', 'cpu',
                                   '--blocks', '2', '--lr', '0.001', '--no-aux-heads', '--eval-games', '8'],
                                  capture_output=True, text=True, cwd=ROOT, timeout=60)
            self.assertEqual(done.returncode, 0, done.stderr)
            config = dense_config.load(run)
            self.assertEqual((config.model.blocks, config.learner.lr, config.model.aux_heads, config.evaluation.games),
                             (2, .001, False, 8))


def episode_rows(moves, winner, root_values=None, rng=None, policy_every=1):
    """Shard episode and rows for a real game: policies are random over the native legal list."""
    rng = rng or np.random.default_rng(0)
    game, rows = Game(), []
    for ply in range(len(moves)):
        actions = np.asarray(game.legal_moves(), np.int64).reshape(-1, 2)
        policy = None
        if ply % policy_every == 0:
            p = rng.random(len(actions)).astype(np.float32)
            policy = p/p.sum()
        rows.append(dict(game=0, ply=ply, player=game.player, remaining=game.remaining,
                         legal_sha256=hashlib.sha256(actions.tobytes()).hexdigest(), policy=policy))
        game.play(*moves[ply])
    game.close()
    episode = dict(moves=[list(m) for m in moves], winner=winner, reason='test', opening_plies=0, actor='test',
                   root_values=root_values, full_search=[True]*len(moves))
    return episode, rows


def winning_game():
    """Player 0 completes (0,0)..(5,0) at ply 11."""
    p0 = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
    p1 = [(0, 5), (1, 5), (3, 5), (4, 5), (6, 6), (7, 7)]
    moves = [p0[0], p1[0], p1[1], p0[1], p0[2], p1[2], p1[3], p0[3], p0[4], p1[4], p1[5], p0[5]]
    game = Game(moves)
    assert game.winner == 0, game.winner
    game.close()
    return moves


def write_games(path, games, identity=None):
    episodes, rows = [], []
    for g, (moves, winner, root_values) in enumerate(games):
        e, r = episode_rows(moves, winner, root_values, np.random.default_rng(g))
        episodes.append(e)
        rows += [dict(x, game=g) for x in r]
    return dense_data.write_shard(path, identity or dict(actor_sha256='a'*64), episodes, rows)


class DenseDataTests(unittest.TestCase):
    def test_value_targets(self):
        players = [dense_data.player_at(t) for t in range(7)]
        self.assertEqual(players, [0, 1, 1, 0, 0, 1, 1])
        targets, weights = dense_data.value_targets(players, None, 1)
        self.assertEqual(targets, [0., 1., 1., 0., 0., 1., 1.])
        self.assertEqual(weights, [1.]*7)
        # lam = 1 with a decisive final root value reproduces the terminal targets of that winner.
        roots = [.3, -.2, .1, .5, -.4, 1., 1.]          # the last mover (player 1) is winning
        targets, weights = dense_data.value_targets(players, roots, -1, lam=1.)
        self.assertEqual(targets, dense_data.value_targets(players, None, 1)[0])
        # lam = 0: one-step bootstrap from the next ply's root value in the mover's frame.
        targets, _ = dense_data.value_targets(players, roots, -1, lam=0.)
        for t in range(6):
            u = roots[t+1] if players[t+1] == players[t] else -roots[t+1]
            self.assertAlmostEqual(targets[t], (1+u)/2)
        self.assertAlmostEqual(targets[6], (1+roots[6])/2)
        # Null root values are skipped; all-null means no target.
        targets, _ = dense_data.value_targets(players, [.3, None, .1, .5, None, None, None], -1, lam=0.)
        u5 = .5                                         # ply 3 value, player 0 frame
        self.assertAlmostEqual(targets[6], (1-u5)/2)    # player 1 at the trailing null
        self.assertAlmostEqual(targets[0], (1+(-.1))/2)  # skips ply 1, bootstraps from ply 2 (player 1)
        targets, weights = dense_data.value_targets(players, [None]*7, -1)
        self.assertEqual((targets, weights), ([None]*7, [0.]*7))
        self.assertEqual(dense_data.value_targets(players, None, -1), ([None]*7, [0.]*7))
        with self.assertRaises(ValueError):
            dense_data.value_targets(players, [.1]*6, -1)

    def test_outcome_lambda_targets(self):
        players = [dense_data.player_at(t) for t in range(4)]
        self.assertEqual(players, [0, 1, 1, 0])
        roots = [.2, -.4, .6, .8]                       # side to move's frame
        u = [.2, .4, -.6, .8]                           # player 0's frame
        for winner in (0, 1):
            hard = dense_data.value_targets(players, None, winner)
            self.assertEqual(dense_data.value_targets(players, roots, winner, outcome_lam=1.), hard)
            self.assertEqual(dense_data.value_targets(players, None, winner, outcome_lam=.9), hard)
            lam, z = .9, 1. if winner == 0 else -1.
            targets, weights = dense_data.value_targets(players, roots, winner, outcome_lam=lam)
            self.assertEqual(weights, [1.]*4)
            self.assertEqual(targets[3], hard[0][3])    # the last ply is exactly the outcome
            for t in range(4):
                g = (1-lam)*sum(lam**(k-1)*u[t+k] for k in range(1, 4-t)) + lam**(3-t)*z
                s = 1 if players[t] == 0 else -1
                self.assertAlmostEqual(targets[t], (1+s*g)/2)
                searched = [(1+s*u[k])/2 for k in range(t+1, 4)]
                self.assertTrue(min([hard[0][t], *searched]) <= targets[t] <= max([hard[0][t], *searched]))
            self.assertNotEqual(targets[:3], hard[0][:3])
        # bootstrap_full_only: a cheap ply's root value is skipped, exactly like a null one.
        full = [True, True, False, True]
        cheap = dense_data.value_targets(players, roots, 0, full=full, outcome_lam=.9)[0]
        self.assertEqual(cheap, dense_data.value_targets(players, [.2, -.4, None, .8], 0, outcome_lam=.9)[0])
        self.assertAlmostEqual(cheap[1], (1-(.1*.8+.9))/2)
        self.assertNotEqual(cheap, dense_data.value_targets(players, roots, 0, outcome_lam=.9)[0])
        # Capped games ignore outcome_lam.
        self.assertEqual(dense_data.value_targets(players, roots, -1, .8, full, outcome_lam=.5),
                         dense_data.value_targets(players, roots, -1, .8, full))
        with self.assertRaises(ValueError):
            dense_data.value_targets(players, roots[:3], 0, outcome_lam=.9)

    def test_carried_values(self):
        self.assertTrue(np.array_equal(dense_data.carried_values([None, .5, None, None]), [np.nan, .5, .5, -.5], equal_nan=True))
        self.assertTrue(np.allclose(dense_data.carried_values([.3, .5, np.nan, None], [True, False, True, True]), [.3, -.3, -.3, .3]))

    def test_calibration_learns_where_search_values_inform(self):
        """Root values carry the outcome only in the last 20 plies: the map is ~z there and ~the base rate earlier."""
        rng = np.random.default_rng(4)
        games = []
        for _ in range(600):
            T, winner = int(rng.integers(60, 160)), int(rng.integers(2))
            sign = np.where([dense_data.player_at(t) == winner for t in range(T)], 1., -1.)
            late = T-np.arange(T) <= 20
            roots = np.where(late, np.clip(.8*sign+rng.normal(0, .1, T), -1, 1), rng.uniform(-1, 1, T))
            games.append((roots, rng.random(T) < .7, winner))
        self.assertIsNone(dense_data.fit_calibration(games[:199]))
        fit = dense_data.fit_calibration(games)
        self.assertEqual(fit, dense_data.fit_calibration(games))
        self.assertAlmostEqual(fit.base, .5, delta=.03)
        near = fit.predict([.8, -.8], [5, 5])
        self.assertGreater(near[0], .95); self.assertLess(near[1], .05)
        for v in (-.8, 0., .8):
            self.assertAlmostEqual(float(fit.predict([v], [100])[0]), fit.base, delta=.05)
        self.assertEqual(fit.predict([np.nan], [5]).tolist(), [fit.base])
        self.assertEqual(dense_data.unpack_calibration(dense_data.pack_calibration(fit)), fit)
        self.assertIsNone(dense_data.unpack_calibration(dense_data.pack_calibration(None)))
        # Finished games get the map at (carried v, plies remaining); capped games keep their TD chain.
        roots, full, winner = games[0]
        roots = [None if not f else float(v) for v, f in zip(roots, full)]
        players = [dense_data.player_at(t) for t in range(len(roots))]
        targets, weights = dense_data.value_targets(players, roots, winner, calibration=fit)
        self.assertEqual(targets, fit.predict(dense_data.carried_values(roots), len(roots)-np.arange(len(roots))).tolist())
        self.assertEqual(weights, [1.]*len(roots))
        self.assertEqual(dense_data.value_targets(players, roots, -1, calibration=fit), dense_data.value_targets(players, roots, -1))
        self.assertEqual(dense_data.value_targets(players, None, winner, calibration=fit)[0], [fit.base]*len(roots))

    def test_learner_fits_the_calibration_from_the_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            synthetic_run(run, 4, 90, 30)
            window = dense_data.ReplayWindow(run, 100000, 10**6)
            games = window.finished_games(10**6)
            episodes = [e for p in dense_data.shard_dirs(run) for e in dense_data.read_shard(p, policies=False)[0]]
            self.assertEqual(len(games), sum(e['winner'] >= 0 for e in episodes))
            newest = [e for e in episodes if e['winner'] >= 0][-1]
            self.assertEqual((games[0][2], len(games[0][0])), (newest['winner'], len(newest['moves'])))
            episodes, rows = [], []
            for g, (moves, winner) in enumerate([(winning_game(), 0)]):
                e, r = episode_rows(moves, winner, [.5]*len(moves))
                episodes.append(dict(e, full_search=None)); rows += [dict(x, game=g) for x in r]
            dense_data.write_shard(run/'shards'/'000009', dict(actor_sha256='a'*64), episodes, rows)
            unflagged = dense_data.ReplayWindow(run, 100000, 10**6).finished_games(1)[0]
            self.assertEqual((unflagged[1], unflagged[2], len(unflagged[0])), (None, 0, len(winning_game())))
            shutil.rmtree(run/'shards'/'000009')
            cut = dense_data.ReplayWindow(run, 1500, 10**6)    # the row budget cuts through an admitted shard
            self.assertGreater(cut.starts[cut.admitted[0][0]], 0)
            inside = {(n, int(cut.shards[n].game[i])) for n, i in cut.index if cut.shards[n].winner[cut.shards[n].game[i]] >= 0}
            self.assertEqual(len(cut.finished_games(10**6)), len(inside))
            config = dense_config.RunConfig(device='cpu', model=dense_config.ModelSettings(**asdict(TINY)))
            learner = dense_learn.Learner(run/'learner', replace(config.learner, calibration_games=220, bootstrap_full_only=True), config)
            learner.calibrate(window)
            games = window.finished_games(220)
            self.assertEqual(learner.calibration, dense_data.fit_calibration([(r, f, w) for r, f, w in games]))
            report = learner.calibration_report
            self.assertEqual((report["games"], report["fitted"], len(report["coef"])), (220, True, dense_data.CALIBRATION_FEATURES))
            self.assertEqual(np.array(report['table']['p']).shape, (len(dense_learn.CALIBRATION_H), len(dense_learn.CALIBRATION_V)))
            self.assertEqual(learner.targets()['calibration'], None)    # value_target 'outcome'
            learner.settings = replace(learner.settings, value_target='calibrated')
            self.assertEqual(learner.targets()['calibration'], learner.calibration)
            learner.settings = replace(learner.settings, calibration_games=150)
            learner.calibrate(window)
            self.assertEqual((learner.calibration, learner.calibration_report), (None, dict(games=150, fitted=False)))
            self.assertEqual(learner.targets()['calibration'], None)

    def test_outcome_targets_in_examples(self):
        moves = winning_game()
        roots = [float(v) for v in np.random.default_rng(3).uniform(-1, 1, len(moves))]
        with tempfile.TemporaryDirectory() as tmp:
            write_games(Path(tmp)/'shards'/'000001', [(moves, 0, roots), (moves[:8], -1, roots[:8])])
            window = dense_data.ReplayWindow(tmp, capacity_rows=1000)
            refs = [window.ref('000001', i) for i in range(len(window.index))]
            hard = dense_data.examples(window, refs, np.random.default_rng(0))[1]
            soft = dense_data.examples(window, refs, np.random.default_rng(0), outcome_lam=.9)[1]
            for ref, h, t in zip(refs, hard, soft):
                e, ply = ref.episode, ref.row['ply']
                if e['winner'] < 0:
                    self.assertEqual((t['value'], t['outcome_weight']), (h['value'], 0.))
                    continue
                won = float(dense_data.player_at(ply) == 0)
                self.assertEqual((h['value'], h['outcome'], t['outcome'], t['outcome_weight']), (won, won, won, t['value_weight']))
                self.assertEqual(t['value'], dense_data.episode_value_targets(e, .9, False, .9)[0][ply])
                self.assertEqual(t['value'] == won, ply == len(moves)-1)

    def test_outcome_weight_adds_a_value_term(self):
        torch.manual_seed(0)
        model = hexnet.HexNet(TINY)
        moves = winning_game()
        roots = [float(v) for v in np.random.default_rng(3).uniform(-1, 1, len(moves))]
        with tempfile.TemporaryDirectory() as tmp:
            write_games(Path(tmp)/'shards'/'000001', [(moves, 0, roots)])
            window = dense_data.ReplayWindow(tmp, capacity_rows=1000)
            refs = [window.ref('000001', i) for i in range(len(window.index))]
            batch = dense_data.collate(*dense_data.examples(window, refs, np.random.default_rng(0), outcome_lam=.9))
        base = dense_config.LearnerSettings()
        def run(settings):
            learner = SimpleNamespace(settings=settings, device=torch.device('cpu'))
            coefficients = dense_learn.Learner.coefficients(learner)
            m = copy.deepcopy(model)
            logged = dense_learn.batch_losses(m, batch, coefficients, learner.device, torch.contiguous_format, True)
            return logged, (logged*coefficients).sum().item(), [p.grad.clone() for p in m.parameters() if p.grad is not None]
        off, loss_off, grad_off = run(base)
        on, loss_on, grad_on = run(replace(base, outcome_weight=.5))
        self.assertEqual(len(off), len(dense_learn.HEADS))
        self.assertTrue(torch.equal(off, on))
        self.assertGreater(off[-1].item(), 0)
        self.assertNotAlmostEqual(off[-1].item(), off[1].item())    # soft value target, hard outcome
        self.assertAlmostEqual(loss_off, (off[:-1]*torch.tensor([1., base.value_weight, base.short_value_weight,
                                                                   base.opponent_policy_weight, base.future_weight])).sum().item(), places=5)
        self.assertAlmostEqual(loss_on-loss_off, .5*off[-1].item(), places=5)
        self.assertFalse(all(torch.allclose(a, b) for a, b in zip(grad_off, grad_on)))

    def test_shard_round_trip_and_tamper(self):
        rng = np.random.default_rng(2)
        moves, _ = random_game(rng, 10)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'shards'/'000001'
            manifest = write_games(path, [(moves, -1, None), (winning_game(), 0, None)])
            self.assertEqual(manifest['counts']['games'], 2)
            self.assertEqual(manifest['counts']['rows'], len(moves)+12)
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ['000001'])
            episodes, rows = dense_data.read_shard(path)
            self.assertEqual(episodes[1]['moves'], [list(m) for m in winning_game()])
            _, source = episode_rows(moves, -1, None, np.random.default_rng(0))
            np.testing.assert_allclose(rows[3]['policy'], source[3]['policy'])
            self.assertTrue(all(abs(float(r['policy'].sum())-1) < 1e-5 for r in rows))
            with self.assertRaises(ValueError):
                write_games(path, [(moves, -1, None)])
            with self.assertRaises(ValueError):
                dense_data.write_shard(Path(tmp)/'shards'/'000002', dict(actor_sha256='a'), [episodes[0]],
                                       [dict(game=0, ply=0, player=0, remaining=1, legal_sha256='', policy=[.5, .6])])
            text = (path/'rows.json').read_text(encoding='utf-8')
            (path/'rows.json').write_text(text.replace('"ply": 3', '"ply": 4', 1), encoding='utf-8')
            with self.assertRaises(ValueError):
                dense_data.read_shard(path)

    def test_window_size(self):
        self.assertEqual(dense_data.window_size(0), 0)
        self.assertEqual(dense_data.window_size(20000), 20000)
        for total in (40000, 1000000):
            expected = 20000*(1+.4*((total/20000)**.65-1)/.65)
            self.assertEqual(dense_data.window_size(total), int(expected))
        self.assertEqual(dense_data.window_size(30, 12, .4, .65), 18)

    def test_replay_window_newest_rows_and_uniform_sampling(self):
        rng = np.random.default_rng(4)
        moves, _ = random_game(rng, 10)
        self.assertEqual(len(moves), 10)
        with tempfile.TemporaryDirectory() as tmp:
            for k in (1, 2, 3):
                write_games(Path(tmp)/'shards'/f'{k:06d}', [(moves, -1, None)])
            window = dense_data.ReplayWindow(tmp, capacity_rows=1000, min_rows=12)
            self.assertEqual(window.rows, 18)
            self.assertEqual(list(window.index), [('000002', i) for i in range(2, 10)]+[('000003', i) for i in range(10)])
            capped = dense_data.ReplayWindow(tmp, capacity_rows=5, min_rows=12)
            self.assertEqual(list(capped.index), [('000003', i) for i in range(5, 10)])
            counts = {}
            for ref in window.sample(np.random.default_rng(0), 36000):
                counts[(ref.shard, ref.index)] = counts.get((ref.shard, ref.index), 0)+1
                self.assertEqual(ref.row['ply'], ref.index)
            self.assertEqual(set(counts), set(window.index))
            self.assertLess(max(abs(c-2000) for c in counts.values()), 200)
            recent = window.sample(np.random.default_rng(0), 9000, recency=2.)
            first = sum(r == ('000002', 2) for r in ((x.shard, x.index) for x in recent))
            last = sum(r == ('000003', 9) for r in ((x.shard, x.index) for x in recent))
            self.assertGreater(last, 20*first)
            write_games(Path(tmp)/'shards'/'000004', [(moves, -1, None)])
            window.refresh()
            self.assertNotIn('000001', window.shards)
            self.assertEqual(window.index[-1], ('000004', 9))

    def test_examples_and_collate(self):
        rng = np.random.default_rng(6)
        compact, _ = random_game(rng, 14)
        wide = line_history(8)
        with tempfile.TemporaryDirectory() as tmp:
            write_games(Path(tmp)/'shards'/'000001', [(compact, -1, [float(v) for v in rng.uniform(-1, 1, len(compact))]),
                                                      (wide, -1, None), (winning_game(), 0, None)])
            window = dense_data.ReplayWindow(tmp, capacity_rows=1000)
            refs = [window.ref('000001', i) for i in range(len(window.index))]
            samples, targets = dense_data.examples(window, refs, np.random.default_rng(0))
            batches = dense_data.collate(samples, targets)
            self.assertGreater(len(batches), 1)
            seen = 0
            for size, b in batches.items():
                n = len(b['counts'])
                seen += n
                self.assertEqual(tuple(b['planes'].shape), (n, len(hexcrop.PLANES), size, size))
                self.assertEqual(b['planes'].dtype, torch.uint8)
                self.assertEqual(b['offsets'].tolist(), [0]+np.cumsum(b['counts'].numpy()).tolist())
                self.assertEqual(int(b['mask'].sum()), len(b['policy']))
                self.assertEqual(b['mask'].sum(1).tolist(), b['counts'].tolist())
                for row in range(n):
                    part = b['policy'][b['offsets'][row]:b['offsets'][row+1]]
                    self.assertAlmostEqual(float(part.sum()), float(b['policy_weight'][row]), places=5)
                    self.assertTrue(torch.all(b['cells'][row, b['counts'][row]:] == -1))
                self.assertEqual(tuple(b['future'].shape), (n, 2, size, size))
                # future includes the stones already on the board.
                stones = (b['planes'][:, 0]+b['planes'][:, 1]).float()
                self.assertTrue(torch.all(b['future'][:, 0][stones > 0] == 1))
                self.assertTrue(torch.all((b['value_weight'] == 0) | ((b['value'] >= 0) & (b['value'] <= 1))))
            self.assertEqual(seen, len(refs))
            by_game = {}
            for ref, t in zip(refs, targets):
                by_game.setdefault(ref.row['game'], []).append(t)
            self.assertTrue(all(t['value_weight'] == 0 for t in by_game[1]))       # capped, no root values
            self.assertEqual([t['value'] for t in by_game[2]],
                             [float(dense_data.player_at(t) == 0) for t in range(12)])
            self.assertTrue(all(t['value_weight'] == 1 for t in by_game[0]+by_game[2]))
            for t in by_game[0][:-1]:
                if t['next_weight']:
                    self.assertAlmostEqual(float(t['next_policy'].sum()), 1, places=5)


def synthetic_run(run, shards, games, plies, seed=0):
    """Shards of fake games (positions are never replayed): random moves, digests, root values (some null),
    trained sides and winners; every other ply has a short policy; rows are stored in shuffled order."""
    rng = np.random.default_rng(seed)
    for k in range(shards):
        episodes, rows = [], []
        for g in range(games):
            T = int(rng.integers(plies//2, plies+1)); side = [None, None, 0, 1][int(rng.integers(4))]
            roots = None if g % 5 == 0 else [None if rng.random() < .2 else float(rng.uniform(-1, 1)) for _ in range(T)]
            episodes.append(dict(moves=rng.integers(-50, 50, (T, 2)).tolist(), winner=int(rng.integers(-1, 2)), reason='test',
                                 opening_plies=0, actor='a'*64, root_values=roots, trained_side=side,
                                 full_search=[bool(t % 2 == 0) for t in range(T)]))
            for t in range(T):
                p = rng.random(int(rng.integers(1, 4))) if t % 2 == 0 else None
                rows.append(dict(game=g, ply=t, player=dense_data.player_at(t), remaining=1 if t == 0 else 2-(t+1) % 2,
                                 legal_sha256=rng.bytes(32).hex(), policy=None if p is None else p/p.sum()))
        rows = [rows[i] for i in rng.permutation(len(rows))]
        dense_data.write_shard(Path(run)/'shards'/f'{k+1:06d}', dict(actor_sha256='a'*64), episodes, rows)


def reference_window(run, capacity, min_rows, fraction):
    """(training index, validation index, {name: (episodes, rows)}) of the window by a direct walk of the shards."""
    data = {p.name: dense_data.read_shard(p) for p in dense_data.shard_dirs(run)}
    full = {n: [i for i, r in enumerate(rows) if len(r['policy'])] for n, (_, rows) in data.items()}
    want = min(capacity, dense_data.window_size(sum(map(len, full.values())), min_rows))
    admitted, have = [], 0
    for name in sorted(data, reverse=True):
        if have >= want:
            break
        take = min(len(full[name]), want-have); admitted.insert(0, (name, take)); have += take
    index, held = [], []
    for name, take in admitted:
        episodes, rows = data[name]
        start = 0 if take >= len(full[name]) else full[name][-take] if take else len(rows)
        for i in range(start, len(rows)):
            e = episodes[rows[i]['game']]
            if dense_data.trained(e, rows[i]['ply']):
                (held if dense_data.holdout(e, fraction) else index).append((name, i))
    return index, held, data


class WindowMemoryTests(unittest.TestCase):
    def test_sampling_matches_the_reference_walk(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_run(tmp, 6, 8, 30)
            window = dense_data.ReplayWindow(tmp, 10**6, 150, validation_fraction=.2)
            index, held, data = reference_window(Path(tmp), 10**6, 150, .2)
            self.assertEqual((list(window.index), list(window.validation)), (index, held))
            self.assertLess(len(window.shards), 6)
            self.assertEqual(window.rows, len(index)+len(held))
            for recency, validation in ((0., False), (2., False), (0., True)):
                keys = held if validation else index
                rng = np.random.default_rng(9)
                if recency:
                    w = (np.arange(1, len(keys)+1)/len(keys))**recency
                    picks = rng.choice(len(keys), 500, p=w/w.sum())
                else:
                    picks = rng.integers(len(keys), size=500)
                refs = window.sample(np.random.default_rng(9), 500, recency, validation)
                self.assertEqual([(r.shard, r.index) for r in refs], [keys[k] for k in picks])
                for ref in refs:
                    episodes, rows = data[ref.shard]
                    row, e = rows[ref.index], episodes[rows[ref.index]['game']]
                    self.assertEqual(ref.row, {k: row.get(k, 0) for k in ('game', 'ply', 'player', 'remaining', 'proven', 'legal_sha256')})
                    self.assertEqual(ref.episode, {k: e[k] for k in ('moves', 'winner', 'root_values', 'full_search', 'trained_side')})
                    np.testing.assert_array_equal(window.policy(ref), row['policy'])
                    for lam, full_only in ((.9, False), (.5, True)):
                        self.assertEqual(window.value_targets(ref, lam, full_only), dense_data.value_targets(
                            [dense_data.player_at(t) for t in range(len(e['moves']))], e['root_values'], e['winner'], lam,
                            e['full_search'] if full_only else None))

    def test_following_row_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_run(tmp, 2, 6, 20)
            window = dense_data.ReplayWindow(tmp, 10**6, 10**6)
            for name, (_, rows) in reference_window(Path(tmp), 10**6, 10**6, 0.)[2].items():
                where = {(r['game'], r['ply']): i for i, r in enumerate(rows)}
                for i, r in enumerate(rows):
                    following = window.following(window.ref(name, i))
                    j = where.get((r['game'], r['ply']+1))
                    self.assertEqual(None if following is None else following.index, j)
                    if j is not None:
                        self.assertEqual((following.row['game'], following.row['ply']), (r['game'], r['ply']+1))

    def test_policy_cache_evicts_least_recently_used_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_run(tmp, 3, 6, 20)
            names = [p.name for p in dense_data.shard_dirs(tmp)]
            sizes = {n: sum(a.nbytes for a in dense_data.load_policies(Path(tmp)/'shards'/n, len(dense_data.read_shard(Path(tmp)/'shards'/n)[1])))
                     for n in names}
            budget = (sorted(sizes.values())[-1]+sorted(sizes.values())[-2]+1)/2**20
            window = dense_data.ReplayWindow(tmp, 10**6, 10**6, policy_cache_mb=budget)
            _, _, data = reference_window(Path(tmp), 10**6, 10**6, 0.)
            first = {n: window.ref(n, int(np.flatnonzero([len(r['policy']) for r in data[n][1]])[0])) for n in names}
            for n in (names[0], names[1], names[0], names[2]):
                np.testing.assert_array_equal(window.policy(first[n]), data[n][1][first[n].index]['policy'])
            self.assertEqual(list(window.policies), [names[0], names[2]])
            self.assertLessEqual(window.policy_bytes(), budget*2**20)
            tiny = dense_data.ReplayWindow(tmp, 10**6, 10**6, policy_cache_mb=1e-6)
            for n in names:
                for i, r in enumerate(data[n][1]):
                    np.testing.assert_array_equal(tiny.policy(tiny.ref(n, i)), r['policy'])
                self.assertEqual(list(tiny.policies), [n])

    def test_held_policies_do_not_pin_evicted_shards(self):
        import gc
        import weakref
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_run(tmp, 5, 6, 20)
            window = dense_data.ReplayWindow(tmp, 10**6, 10**6, policy_cache_mb=1e-6)
            _, _, data = reference_window(Path(tmp), 10**6, 10**6, 0.)
            held, arrays = [], []
            for ref in window.sample(np.random.default_rng(1), 200):
                policy = window.policy(ref)
                arrays += [weakref.ref(a) for a in window.policies[ref.shard]]
                np.testing.assert_array_equal(policy, data[ref.shard][1][ref.index]['policy'])
                self.assertEqual(policy.dtype, np.float32)
                held.append(policy)
            self.assertGreater(len({r.shard for r in window.sample(np.random.default_rng(1), 200)}), 1)
            gc.collect()
            cached = {id(a) for entry in window.policies.values() for a in entry}
            self.assertTrue(all(r() is None or id(r()) in cached for r in arrays))
            # A view keeps its buffer's owner (the npz member's bytes, not the array) alive: policies must own theirs.
            self.assertTrue(all(p.base is None for p in held))

    def test_resident_bytes_per_row(self):
        import gc
        import tracemalloc
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_run(tmp, 40, 20, 130, seed=1)
            gc.collect(); tracemalloc.start()
            try:
                before = tracemalloc.get_traced_memory()[0]
                window = dense_data.ReplayWindow(tmp, 10**7, 10**7, validation_fraction=.03)
                gc.collect()
                used = tracemalloc.get_traced_memory()[0]-before
            finally:
                tracemalloc.stop()
            self.assertGreater(window.rows, 50000)
            self.assertLess(used/window.rows, 150)


def old_corpus(path):
    """A tiny gumbel-policy-value-v1 style corpus: one finished game (id 5) and one capped game (id 9)."""
    rng = np.random.default_rng(11)
    capped, _ = random_game(rng, 9)
    games = [(5, winning_game(), 0, 'six-in-a-row'), (9, capped, -1, 'max-plies')]
    episodes, rows, policies = [], [], []
    for gid, moves, winner, reason in games:
        episodes.append(dict(id=gid, moves=[list(m) for m in moves], opening=[], winner=winner, reason=reason, actors=[0, 0]))
        game = Game()
        for ply, move in enumerate(moves):
            actions = np.asarray(game.legal_moves(), np.int64).reshape(-1, 2)
            p = rng.random(len(actions)).astype(np.float32)
            policies.append(p/p.sum())
            target = None if winner < 0 else (1. if game.player == winner else -1.)
            rows.append(dict(game=gid, ply=ply, player=game.player, remaining=game.remaining, action=list(move),
                             legal_sha256=hashlib.sha256(actions.tobytes()).hexdigest(), simulations=16,
                             evaluated=17, target=target))
            game.play(*move)
        game.close()
    path.mkdir(parents=True)
    (path/'episodes.json').write_text(json.dumps(episodes), encoding='utf-8')
    (path/'rows.json').write_text(json.dumps(rows), encoding='utf-8')
    np.savez_compressed(path/'targets.npz', offsets=np.cumsum([0]+[len(p) for p in policies]).astype(np.int64),
                        probabilities=np.concatenate(policies))
    files = {name: dense_bootstrap.digest(path/name) for name in ('episodes.json', 'rows.json', 'targets.npz')}
    manifest = dict(schema=dense_bootstrap.OLD_SCHEMA, identity=dict(actor_sha256='b'*64, policy_target='test'),
                    metrics=None, files=files)
    (path/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return games


class DenseBootstrapTests(unittest.TestCase):
    def test_convert_old_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, run = Path(tmp)/'old'/'corpus'/'0001', Path(tmp)/'run'
            games = old_corpus(source)
            manifest, episodes, rows = dense_bootstrap.read_corpus(source)
            identity, new_episodes, new_rows = dense_bootstrap.convert(
                manifest, dense_bootstrap.digest(source/'manifest.json'), episodes, rows)
            target = run/'shards'/'000001'
            written = dense_data.write_shard(target, identity, new_episodes, new_rows, 'converted')
            self.assertEqual((written['actor'], written['origin']), ('b'*64, 'converted'))
            self.assertEqual(dense_data.origin(dict(written, origin=None)), 'converted')    # inferred from the identity
            self.assertEqual(written['counts'], dict(games=2, rows=21, policy_rows=21, opponent_rows=0, terminal_games=1, capped_games=1,
                                                 proven_rows=0, proven_games=0, line_rows=0, adjudicated_plies=0,
                                                 restart_games=0, forced_plies=0))
            self.assertEqual(dense_bootstrap.check(target), 21)
            _, stored = dense_data.read_shard(target)
            self.assertEqual({r['game'] for r in stored}, {0, 1})
            window = dense_data.ReplayWindow(run, capacity_rows=1000)
            refs = [window.ref('000001', i) for i in range(21)]
            samples, targets = dense_data.examples(window, refs, np.random.default_rng(0))
            for ref, s, t in zip(refs, samples, targets):
                np.testing.assert_array_equal(s.actions, legal(games[ref.row['game']][1][:ref.row['ply']]))
                self.assertEqual(len(t['policy']), len(s.actions))
                if ref.row['game'] == 0:
                    self.assertEqual((t['value'], t['value_weight']), (float(s.player == 0), 1.))
                else:
                    self.assertEqual(t['value_weight'], 0.)
            dense_data.collate(samples, targets)
            # A tampered corpus is refused.
            (source/'rows.json').write_text('[]', encoding='utf-8')
            with self.assertRaises(ValueError):
                dense_bootstrap.read_corpus(source)


def source_shard(path, seed, actor, origin='actor', checkpoint=None, games=6, policy_every=2, publisher=None, winner=-1):
    """A shard of `games` random games recorded with `winner` (default capped); game g is played by actor[g % len(actor)] for a list, else by `actor`.
    The identity's actor_sha256 is `publisher`, default the last actor. Actor shards get a dense_selfplay-like identity."""
    rng = np.random.default_rng(seed)
    actors = actor if isinstance(actor, list) else [actor]
    episodes, rows = [], []
    for g in range(games):
        moves, _ = random_game(rng, 8)
        e, r = episode_rows(moves, winner, [float(v) for v in rng.uniform(-1, 1, len(moves))], rng, policy_every)
        episodes.append(dict(e, actor=actors[g % len(actors)]))
        rows += [dict(x, game=g) for x in r]
    publisher = publisher or actors[-1]
    identity = dict(source='gumbel-policy-value-v1', actor_sha256=publisher) if origin == 'converted' else \
        dict(actor_sha256=publisher, actors=sorted(set(actors)), checkpoint=checkpoint)
    return dense_data.write_shard(path, identity, episodes, rows, origin)


class ValidationSourceTests(unittest.TestCase):
    def test_origin_inference(self):
        self.assertEqual(dense_data.origin(dict(origin='actor', identity=dict(source='x'))), 'actor')
        self.assertEqual(dense_data.origin(dict(identity=dict(source='gumbel-policy-value-v1', actor_sha256='a'))), 'converted')
        self.assertEqual(dense_data.origin(dict(identity=dict(actor_sha256='a', actors=['a'], checkpoint='main/000500'))), 'actor')
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(source_shard(Path(tmp)/'shards'/'1', 0, 'a')['origin'], 'actor')
            with self.assertRaises(ValueError):
                dense_data.write_shard(Path(tmp)/'shards'/'2', dict(actor_sha256='a'), [], [], 'other')

    def test_fixed_subsets_per_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for k in (1, 2):
                source_shard(run/'shards'/f'{k:06d}', k, 'old', 'converted')
            source_shard(run/'shards'/'1000000000001', 3, 'x', checkpoint='main/000010')
            sets = dense_data.ValidationSets(run, .5, 5, limit=10, quota=4)
            sets.refresh()
            def keys():
                return {k: [(r.shard, r.index) for r in refs] for k, refs in sets.subsets.items()}
            first = keys()
            for (source, split), refs in sets.subsets.items():
                self.assertTrue(refs, (source, split))
                self.assertLessEqual(len(refs), 10)
                for shard in {r.shard for r in refs}:
                    self.assertLessEqual(sum(r.shard == shard for r in refs), 4)
                for r in refs:
                    self.assertEqual(dense_data.holdout(r.episode, .5), split == 'held')
                    self.assertTrue(len(sets.policy(r)))
                    self.assertEqual(len(r.shard) == 6, source == 'converted')
            self.assertEqual((sets.newest, sets.newest_checkpoint), ('x', 'main/000010'))
            self.assertEqual(first['fresh', 'held'], first['newest', 'held'])
            again = dense_data.ValidationSets(run, .5, 5, limit=10, quota=4)
            again.refresh()
            self.assertEqual({k: [(r.shard, r.index) for r in refs] for k, refs in again.subsets.items()}, first)
            # A newer shard of the same actor only appends; a new actor restarts only the newest subsets.
            source_shard(run/'shards'/'1000000000002', 4, 'x', checkpoint='main/000010')
            sets.refresh()
            grown = keys()
            for key, before in first.items():
                self.assertEqual(grown[key][:len(before)], before)
            self.assertEqual(grown['converted', 'held'], first['converted', 'held'])
            source_shard(run/'shards'/'1000000000003', 5, 'y', checkpoint='main/000020')
            sets.refresh()
            self.assertEqual(sets.newest_checkpoint, 'main/000020')
            self.assertEqual({r.shard for r in sets.subsets['newest', 'train']}, {'1000000000003'})
            self.assertEqual(keys()['fresh', 'train'][:len(grown['fresh', 'train'])], grown['fresh', 'train'])
            refs = sets.subsets['fresh', 'train']
            samples, targets = dense_data.examples(sets, refs, np.random.default_rng(0))
            for r, t in zip(refs, targets):
                self.assertEqual(t['policy_weight'], 1.)
                following = sets.following(r)
                self.assertEqual(t['next_weight'] > 0, following is not None)
                if following is not None:
                    self.assertEqual((following.row['game'], following.row['ply']), (r.row['game'], r.row['ply']+1))
            dense_data.collate(samples, targets)

    def test_learner_settings_size_the_subsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for k in (1, 2, 3):
                source_shard(run/'shards'/f'10000000000{k:02d}', k, 'x', games=12, policy_every=1)
            settings = dense_config.LearnerSettings(validation_fraction=.5, validation_rows=9, validation_quota=4)
            sets = dense_learn.validation_sets(run, settings, 5)
            sets.refresh()
            self.assertEqual((sets.limit, sets.quota), (9, 4))
            for (source, _), refs in sets.subsets.items():
                if source == 'converted':
                    continue
                self.assertEqual(len(refs), 9)
                self.assertEqual(sorted(Counter(r.shard for r in refs).values()), [1, 4, 4])

    def test_newest_actor_comes_from_episodes(self):
        """Right after a checkpoint switch the newest shard's publisher may have played none of its games."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            source_shard(run/'shards'/'1000000000001', 1, 'x', checkpoint='main/000010')
            source_shard(run/'shards'/'1000000000002', 2, 'x', checkpoint='main/000020', publisher='y')
            sets = dense_data.ValidationSets(run, .5, 5, limit=10, quota=4)
            sets.refresh()
            self.assertEqual((sets.newest, sets.newest_checkpoint), ('x', 'main/000010'))
            for split in ('held', 'train'):
                refs = sets.subsets['newest', split]
                self.assertTrue(refs)
                self.assertTrue(all(r.episode['actor'] == 'x' for r in refs))
            source_shard(run/'shards'/'1000000000003', 3, ['x', 'y'], checkpoint='main/000020')
            sets.refresh()
            self.assertEqual((sets.newest, sets.newest_checkpoint), ('y', 'main/000020'))
            self.assertTrue(sets.subsets['newest', 'train'])
            self.assertTrue(all(r.episode['actor'] == 'y' for r in sets.subsets['newest', 'train']))

    def test_lagging_worker_does_not_move_newest_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            source_shard(run/'shards'/'1000000000001', 1, 'x', checkpoint='main/000010')
            source_shard(run/'shards'/'1000000000002', 2, 'y', checkpoint='main/000020')
            sets = dense_data.ValidationSets(run, .5, 5, limit=10, quota=4)
            sets.refresh()
            self.assertEqual((sets.newest, sets.newest_checkpoint), ('y', 'main/000020'))
            before = [(r.shard, r.index) for r in sets.subsets['newest', 'train']]
            source_shard(run/'shards'/'1000000000003', 3, 'x', checkpoint='main/000010')    # the lagging worker
            sets.refresh()
            self.assertEqual((sets.newest, sets.newest_checkpoint), ('y', 'main/000020'))
            self.assertEqual([(r.shard, r.index) for r in sets.subsets['newest', 'train']], before)
            again = dense_data.ValidationSets(run, .5, 5, limit=10, quota=4)
            again.refresh()
            self.assertEqual(again.newest, 'y')

    def test_retained_state_is_bounded(self):
        """Many shards: full subsets stop consuming shards, and per shard only actor row counts are kept."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            sets = dense_data.ValidationSets(run, .5, 5, limit=6, quota=3)
            for k in range(12):
                source_shard(run/'shards'/f'{1000000000001+k}', k, 'x', checkpoint='main/000010')
                sets.refresh()
            self.assertEqual({k: len(p) for k, p in sets.picks.items() if k[0] != 'converted'},
                             {(s, split): 6 for s in ('fresh', 'newest') for split in ('held', 'train')})
            self.assertLessEqual(max(len(w) for w in sets.walked.values()), 6)
            self.assertLessEqual(len(sets.entries), 2*6*len(sets.picks))
            self.assertEqual(sorted(sets.actors), [f'{1000000000001+k}' for k in range(12)])
            self.assertTrue(all(list(c) == ['x'] for c in sets.actors.values()))

    def test_newest_change_selects_a_cached_successor(self):
        """A row cached only as a chosen row's next-ply successor can be chosen after the newest actor changes."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            source_shard(run/'shards'/'1000000000001', 1, ['x', 'y'], checkpoint='main/000010', games=8, policy_every=1,
                         publisher='x')
            source_shard(run/'shards'/'1000000000002', 2, 'x', checkpoint='main/000010', policy_every=1)
            sets = dense_data.ValidationSets(run, .3, 5, limit=40, quota=12)
            sets.refresh()
            successors = set(sets.entries)-set(sets.following_index)
            source_shard(run/'shards'/'1000000000003', 3, 'y', checkpoint='main/000020', policy_every=1)
            sets.refresh()
            self.assertEqual(sets.newest, 'y')
            chosen = {(r.shard, r.index) for refs in sets.subsets.values() for r in refs}
            self.assertTrue(successors & chosen)
            for refs in sets.subsets.values():
                for r in refs:
                    following = sets.following(r)
                    if following is not None:
                        self.assertEqual((following.row['game'], following.row['ply']), (r.row['game'], r.row['ply']+1))
                        self.assertTrue(len(sets.policy(following)))

    def test_export_logs_per_source_validation(self):
        import dashboard
        torch.set_num_threads(2)
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            source_shard(run/'shards'/'000001', 1, 'old', 'converted')
            source_shard(run/'shards'/'1000000000001', 2, 'x', checkpoint='main/000010')
            config = dense_config.RunConfig(device='cpu', model=dense_config.ModelSettings(**asdict(TINY)),
                                            learner=dense_config.LearnerSettings(batch=8, validation_fraction=.5))
            learner = dense_learn.Learner(run, config.learner, config)
            window = dense_data.ReplayWindow(run, 1000, 10, validation_fraction=.5)
            sets = dense_data.ValidationSets(run, .5, config.seed, limit=12, quota=6)
            manifest = learner.export(window, sets)
            self.assertEqual(set(manifest), {'variant', 'step', 'samples_seen', 'created_at', 'model_sha256', 'ema_sha256',
                                             'metrics', 'learner', 'model', 'copied_from', 'rows', 'pacing'})
            aggregate, v = manifest['metrics']['validation'], manifest['metrics']['validation_sources']
            self.assertEqual(v['newest_checkpoint'], 'main/000010')
            for h in ('policy_ce', 'value_bce', 'opponent_ce', 'future_bce'):
                self.assertTrue(math.isfinite(aggregate[h]))
            # Capped 8-ply games: no short-value row (the game ends within the horizon) and no outcome row.
            self.assertEqual((aggregate['short_value_bce'], aggregate['outcome_bce']), (None, None))
            self.assertEqual(manifest['metrics']['calibration'], dict(games=0, fitted=False))
            for source in dense_data.SOURCES:
                self.assertEqual(v[f'{source}_rows'], len(sets.subsets[source, 'held']))
                for name in ('policy_ce', 'value_bce'):
                    self.assertAlmostEqual(v[f'{source}_gap_{name}'], v[f'{source}_{name}']-v[f'{source}_train_{name}'])
            self.assertEqual((aggregate, v), (learner.validate(window), learner.validate_sources(sets)))
            path = run/'checkpoints'/'main'/'000000'
            self.assertEqual(sorted(p.name for p in path.iterdir()), ['ema.pt', 'manifest.json', 'model.pt', 'optimizer.pt'])
            self.assertEqual(dense_learn.Learner(run, config.learner, config).step, 0)
            fields = dense_learn.validation_fields(manifest['metrics'])
            self.assertEqual(fields['next_ce'], aggregate['opponent_ce'])
            dense_config.append_metrics(run, 'learner-main', step=10, validation=True, **fields)
            points = dashboard.series(run, dict(created_at=0.), 'main', 'validation_newest_gap_policy_ce')['points']
            self.assertEqual(points, [[10, v['newest_gap_policy_ce']]])
            dense_config.append_metrics(run, 'learner-main', step=20, outcome_bce=.5)
            self.assertEqual(dashboard.series(run, dict(created_at=0.), 'main', 'outcome_bce')['points'], [[20, .5]])
            self.assertIn('validation_outcome_bce', dashboard.LEARNER_METRICS)
            refs = window.sample(np.random.default_rng(0), 8)
            losses = learner.train_step(dense_data.collate(*dense_data.examples(window, refs, np.random.default_rng(0))))
            self.assertTrue(math.isnan(losses[2]) and math.isnan(losses[5]) and torch.isfinite(losses[:2]).all())

    def test_remaining_curve_on_outcomes_decided_in_the_last_ten_plies(self):
        """Games whose outcome is fixed only in their last 10 plies: a predictor that knows it there and says 0.5
        before has a curve near 0 at the end and near ln 2 far away; the horizon (curve crossing ln 2 / 2) is the step."""
        rng = np.random.default_rng(0)
        remaining, bce, target = [], [], []
        for _ in range(300):
            y, length = rng.integers(2), int(rng.integers(40, 121))
            for left in range(1, length+1):
                q = .01+.98*y if left <= 10 else .5
                remaining.append(left); bce.append(-math.log(q if y else 1-q)); target.append(float(y))
        c = dense_learn.remaining_curve(remaining, bce, target)
        curve = dict(zip(dense_learn.REMAINING_GRID, c['value_curve']))
        excess = dict(zip(dense_learn.REMAINING_GRID, c['value_excess_curve']))
        self.assertEqual(len(c['value_curve']), 41)
        self.assertLess(curve[0], .1)
        self.assertAlmostEqual(excess[0], curve[0]-math.log(2), delta=.02)
        self.assertAlmostEqual(excess[100], 0., delta=.01)
        self.assertIsNone(excess[160])
        self.assertEqual([curve[g] for g in range(0, 28, 4)], sorted(curve[g] for g in range(0, 28, 4)))
        self.assertAlmostEqual(curve[60], math.log(2), places=3)
        self.assertAlmostEqual(curve[100], math.log(2), places=3)
        self.assertIsNone(curve[160])
        self.assertAlmostEqual(c['value_horizon'], 10.5, delta=1.)
        mask = np.array(remaining) <= 20
        self.assertAlmostEqual(c['value_bce_last20'], float(np.mean(np.array(bce)[mask])))
        sure = dense_learn.remaining_curve([5, 5], [.8, .8], [1., 1.], grid=(5,))
        self.assertEqual((sure['value_curve'], sure['value_excess_curve'], sure['value_horizon']), ([.8], [.8], 5.))
        split = dense_learn.remaining_curve([5, 5], [.8, .8], [1., 0.], grid=(5,))
        self.assertEqual(split['value_excess_curve'], [round(.8-math.log(2), 4)])
        empty = dense_learn.remaining_curve([], [], [])
        self.assertEqual((set(empty['value_curve']), empty['value_bce_last20'], empty['value_horizon']), ({None}, None, None))

    def test_policy_curve_rises_when_the_policy_is_sharp_early_and_flat_late(self):
        """Improved policies that are sharp before ply 40 and flat over 30 cells after, matched exactly by the
        network: the by-ply policy CE is the target entropy, near 0 early and near ln 30 late, rising in between,
        and None where no row lies within reach of a grid point."""
        plies = np.arange(300).repeat(3)
        n = 30
        target = torch.full((len(plies), n), 1/n)
        sharp = torch.tensor(plies < 40)
        target[sharp] = .001/n
        target[sharp, 0] += .999
        cells = torch.arange(n).repeat(len(plies), 1)
        ce = hexnet.policy_row_losses(target.log(), None, cells, torch.full((len(plies),), n), target).numpy()
        curve = dict(zip(dense_learn.PLY_GRID, dense_learn.ply_curve(plies, ce)))
        self.assertEqual(len(curve), 49)
        self.assertLess(curve[0], .05)
        self.assertAlmostEqual(curve[96], math.log(n), places=3)
        ys = [curve[g] for g in range(0, 300, 8)]
        self.assertEqual(ys, sorted(ys))
        self.assertIsNone(curve[384])
        early, late = dense_learn.ply_split(plies, ce)
        self.assertAlmostEqual(early, float(ce[plies < 20].mean()))
        self.assertAlmostEqual(late, math.log(n), places=4)
        self.assertEqual(dense_learn.ply_split([30], [1.]), (None, None))

    def test_export_reports_value_by_plies_remaining(self):
        import dashboard
        torch.set_num_threads(2)
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            source_shard(run/'shards'/'1000000000001', 2, 'x', checkpoint='main/000010', games=8, winner=0)
            source_shard(run/'shards'/'1000000000002', 3, 'x', checkpoint='main/000010', games=4)
            config = dense_config.RunConfig(device='cpu', model=dense_config.ModelSettings(**asdict(TINY)),
                                            learner=dense_config.LearnerSettings(batch=8, validation_fraction=.5))
            learner = dense_learn.Learner(run, config.learner, config)
            window = dense_data.ReplayWindow(run, 1000, 10, validation_fraction=.5)
            sets = dense_data.ValidationSets(run, .5, config.seed, limit=40, quota=20)
            metrics = learner.export(window, sets)['metrics']
            v = metrics['validation_sources']
            self.assertEqual(v['remaining_grid'], list(dense_learn.REMAINING_GRID))
            held = sets.subsets['fresh', 'held']
            finished = [r for r in held if r.episode['winner'] >= 0]
            self.assertTrue(finished and len(finished) < len(held))
            r = learner.row_losses(sets, held)
            f, p = r['finished'] > 0, np.isfinite(r['policy_ce'])
            self.assertEqual(sorted(r['remaining'][f]), sorted(len(x.episode['moves'])-x.row['ply'] for x in finished))
            self.assertEqual(sorted(r['ply']), sorted(x.row['ply'] for x in held))
            self.assertTrue(set(r['outcome'][f].tolist()) <= {0., 1.})
            self.assertTrue(np.array_equal(r['value'][f], r['outcome'][f]) and np.array_equal(r['value_bce'][f], r['outcome_bce'][f]))
            self.assertAlmostEqual(v['fresh_value_bce_last20'], float(r['outcome_bce'][f].mean()))
            self.assertTrue(p.all())
            self.assertAlmostEqual(float(r['policy_ce'][p].mean()), v['fresh_policy_ce'], delta=1e-4)
            self.assertEqual(v['ply_grid'], list(dense_learn.PLY_GRID))
            self.assertEqual(v['fresh_policy_ce_curve'], dense_learn.ply_curve(r['ply'][p], r['policy_ce'][p]))
            self.assertEqual(v['fresh_value_bce_by_ply'], dense_learn.ply_curve(r['ply'][f], r['outcome_bce'][f]))
            self.assertEqual((v['fresh_policy_ce_early'], v['fresh_policy_ce_late']), dense_learn.ply_split(r['ply'][p], r['policy_ce'][p]))
            self.assertIsNotNone(v['fresh_policy_ce_early'])
            for source in dense_learn.CURVE_SOURCES:
                for key in ('value_curve', 'value_excess_curve'):
                    self.assertEqual(len(v[f'{source}_{key}']), len(dense_learn.REMAINING_GRID))
                for key in ('policy_ce_curve', 'value_bce_by_ply'):
                    self.assertEqual(len(v[f'{source}_{key}']), len(dense_learn.PLY_GRID))
                for key in ('value_horizon', 'policy_ce_early', 'policy_ce_late'):
                    self.assertIn(f'{source}_{key}', v)
            for source in dense_learn.CURVE_SOURCES:
                for key, field in (('value_surface', 'value'), ('value_excess_surface', 'excess'), ('policy_surface', 'policy')):
                    grid = v[f'{source}_{key}']
                    self.assertEqual(set(grid), {'ply_bins', 'remaining_bins', 'counts', field})
                    self.assertEqual(len(grid[field]), len(grid['ply_bins']))
            fg, (bce,) = dense_learn.surfaces(r['ply'][f], r['remaining'][f], (r['value_bce'][f],))
            self.assertEqual(v['fresh_value_surface'], fg | dict(value=[dense_learn.compact(x) for x in bce]))
            self.assertEqual(sum(map(sum, v['fresh_policy_surface']['counts'])), int(p.sum()))
            self.assertTrue(any(y is not None for row in v['fresh_value_surface']['value'] for y in row))
            self.assertNotIn('converted_value_curve', v)
            self.assertNotIn('converted_value_surface', v)
            for source in dense_learn.CURVE_SOURCES:
                for key in ('value_regret', 'value_regret_early', 'value_regret_late'):
                    self.assertIn(f'{source}_{key}', v)
            self.assertIsNotNone(v['fresh_value_regret'])
            self.assertTrue(np.isfinite(r['searched']).all())
            self.assertNotIn('converted_policy_ce_curve', v)
            fields = dense_learn.validation_fields(metrics)
            self.assertFalse([k for k, x in fields.items() if isinstance(x, (list, dict))])
            self.assertEqual(fields['fresh_value_bce_last20'], v['fresh_value_bce_last20'])
            self.assertEqual(fields['fresh_policy_ce_early'], v['fresh_policy_ce_early'])
            self.assertEqual(fields['fresh_value_regret'], v['fresh_value_regret'])
            dense_config.append_metrics(run, 'learner-main', step=7, validation=True, **fields)
            config = dict(created_at=0.)
            self.assertEqual(dashboard.series(run, config, 'main', 'validation_fresh_value_bce_last20')['points'],
                             [[7, v['fresh_value_bce_last20']]])
            self.assertEqual(dashboard.series(run, config, 'main', 'validation_fresh_policy_ce_early')['points'],
                             [[7, v['fresh_policy_ce_early']]])
            late = dashboard.series(run, config, 'main', 'validation_newest_policy_ce_late')['points']
            self.assertEqual(late, [] if v['newest_policy_ce_late'] is None else [[7, v['newest_policy_ce_late']]])
            horizon = dashboard.series(run, config, 'main', 'validation_newest_value_horizon')['points']
            self.assertEqual(horizon, [] if v['newest_value_horizon'] is None else [[7, v['newest_value_horizon']]])
            curve = dashboard.series(run, config, 'main', 'fresh_value_curve', 'remaining')
            self.assertEqual(curve['checkpoint'], 'main/000000')
            self.assertEqual(curve['points'], [list(q) for q in zip(v['remaining_grid'], v['fresh_value_curve'])])
            self.assertTrue(any(y is not None for _, y in curve['points']))
            self.assertEqual(dashboard.series(run, config, 'side', 'fresh_value_curve', 'remaining')['points'], [])
            with self.assertRaises(ValueError):
                dashboard.series(run, config, 'main', 'fresh_value_curve')
            for key in ('policy_ce_curve', 'value_bce_by_ply'):
                curve = dashboard.series(run, config, 'main', f'fresh_{key}', 'ply')
                self.assertEqual(curve['points'], [list(q) for q in zip(v['ply_grid'], v[f'fresh_{key}'])])
                self.assertTrue(any(y is not None for _, y in curve['points']))
                self.assertTrue(any(y is None for _, y in curve['points']))
                with self.assertRaises(ValueError):
                    dashboard.series(run, config, 'main', f'fresh_{key}', 'remaining')
            json.dumps(curve, allow_nan=False)

    def test_surfaces_bin_rows_into_cells(self):
        """Rows land in the (ply // 16, remaining // 16) cell; cells under min_cells rows are nan but keep their
        count; rows beyond the limit are dropped; the excess subtracts the entropy of the cell's outcome rate."""
        ply = [0]*8+[20]*8+[40]*3+[400]
        remaining = [15]*8+[33]*8+[5]*3+[0]
        loss = [.5]*8+[1., 2.]*4+[9.]*3+[9.]
        target = [1.]*8+[1., 0.]*4+[0.]*4
        grid, (mean, rate) = dense_learn.surfaces(ply, remaining, (loss, target))
        self.assertEqual(grid['ply_bins'], list(range(0, 384, 16)))
        self.assertEqual(grid['remaining_bins'], grid['ply_bins'])
        counts = np.array(grid['counts'])
        self.assertEqual(counts.shape, (24, 24))
        self.assertEqual((counts[0, 0], counts[1, 2], counts[2, 0], counts.sum()), (8, 8, 3, 19))
        self.assertEqual((mean[0, 0], mean[1, 2]), (.5, 1.5))
        self.assertTrue(np.isnan(mean[2, 0]) and np.isnan(mean[5, 5]))
        excess = mean-dense_learn.binary_entropy(rate)
        self.assertAlmostEqual(excess[0, 0], .5)
        self.assertAlmostEqual(excess[1, 2], 1.5-math.log(2))
        self.assertTrue(np.isnan(excess[2, 0]))
        few = dense_learn.surfaces(ply, remaining, (loss,), min_cells=3)[1][0]
        self.assertEqual(few[2, 0], 9.)

    def test_value_regret_against_the_calibrated_search(self):
        """Outcomes drawn from a known function of the searched value and plies remaining: a net predicting that
        probability has regret near zero; one that is confidently wrong far from the end has positive late regret."""
        rng = np.random.default_rng(0)
        def rows(n):
            v, h = rng.uniform(-1, 1, n), rng.integers(1, 200, n).astype(float)
            p = 1/(1+np.exp(-np.arctanh(np.clip(v, -.995, .995))*(3-np.log1p(h)/3)))
            v[:n//20] = np.nan
            p[:n//20] = .5
            return v, h, (rng.random(n) < p).astype(float), p
        fv, fh, fy, _ = rows(20000)
        v, h, y, truth = rows(20000)
        reference = dense_learn.calibration_reference(fv, fh, fy, v, h)
        self.assertAlmostEqual(float(reference[0]), float(np.clip(fy.mean(), 1e-3, 1-1e-3)))
        bce = lambda q: -(y*np.log(q)+(1-y)*np.log(1-q))
        matched = dense_learn.value_regret(bce(truth), y, reference, h)
        for k in ('value_regret', 'value_regret_early', 'value_regret_late'):
            self.assertLess(abs(matched[k]), .02, k)
        wrong = np.where(h >= 60, np.where(truth > .5, .01, .99), truth)
        off = dense_learn.value_regret(bce(wrong), y, reference, h)
        self.assertGreater(off['value_regret_late'], .5)
        self.assertLess(abs(off['value_regret_early']), .02)
        self.assertEqual(dense_learn.value_regret([.1], [1.], None, [5]), dict.fromkeys(matched))
        self.assertIsNone(dense_learn.calibration_reference([], [], [], [0.], [5]))
        episode = dict(root_values=[.2, None, .4, .6], full_search=[True, True, True, False])
        self.assertEqual([dense_learn.searched_value(episode, t) for t in range(4)], [.2, -.2, .4, -.4])
        self.assertTrue(math.isnan(dense_learn.searched_value(dict(root_values=None), 3)))

    def test_value_target_map_and_regret_reference_share_one_fit(self):
        """The same finished games as value target map input (fit_calibration) and as regret reference fit rows
        (one row per ply: carried value, plies remaining, outcome for the side to move) give the same map."""
        rng = np.random.default_rng(7)
        games = []
        for _ in range(250):
            T, winner = int(rng.integers(30, 120)), int(rng.integers(2))
            sign = np.where([dense_data.player_at(t) == winner for t in range(T)], 1., -1.)
            roots = [None if rng.random() < .2 else float(np.clip(sign[t]*rng.uniform(0, 1)+rng.normal(0, .5), -1, 1))
                     for t in range(T)]
            games.append((roots, rng.random(T) < .8, winner))
        fit = dense_data.fit_calibration(games)
        rows = np.concatenate([np.stack([dense_data.carried_values(roots, full), len(roots)-np.arange(len(roots)),
                                         [float(dense_data.player_at(t) == winner) for t in range(len(roots))]])
                               for roots, full, winner in games], 1)
        v, h = np.append(rng.uniform(-1, 1, 500), [np.nan, -1., 1.]), np.append(rng.integers(0, 400, 500), [5, 5, 5])
        reference = dense_learn.calibration_reference(*rows, v, h)
        self.assertEqual(fit, dense_data.fit_calibration_rows(*rows, fit.base))
        loose = dense_data.fit_calibration(games, ridge=100., iterations=3)
        self.assertEqual(loose, dense_data.fit_calibration_rows(*rows, fit.base, 100., 3))
        self.assertNotEqual(loose, fit)
        self.assertTrue(np.array_equal(dense_learn.calibration_reference(*rows, v, h, ridge=100., steps=3), loose.predict(v, h)))
        self.assertTrue(np.allclose(reference, fit.predict(v, h), rtol=0, atol=1e-12))
        self.assertEqual(reference[500], fit.base)
        self.assertLess(reference[501], .5); self.assertGreater(reference[502], .5)
        few = dense_learn.calibration_reference([.5, -.5, .2], [5, 5, 5], [1., 0., 1.], [.5, 0.], [5, 100], ridge=0.)
        self.assertTrue(np.all(np.isfinite(few)))
        prior = dense_data.fit_calibration_rows([np.nan]*3, [5, 9, 40], [1., 0., 1.], .25)
        self.assertTrue(np.allclose(prior.predict([-1., 0., 1.], [2, 50, 300]), .25))

    def test_surface_endpoint_returns_the_newest_grid(self):
        import dashboard
        from http.server import HTTPServer
        from urllib.error import HTTPError
        from urllib.request import urlopen
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root/'r'
            run.mkdir()
            (run/'config.json').write_text(json.dumps(dict(schema=dense_config.SCHEMA, created_at=0.)))
            bins = [0, 16]
            for step, value in ((5, .9), (10, .4)):
                path = run/'checkpoints'/'main'/f'{step:06d}'
                path.mkdir(parents=True)
                grid = dict(ply_bins=bins, remaining_bins=bins, counts=[[9, 2], [0, 12]], value=[[value, None], [None, .7]])
                (path/'manifest.json').write_text(json.dumps(dict(step=step, metrics=dict(validation_sources=dict(newest_value_surface=grid)))))
            out = dashboard.surface(run, 'main', 'newest_value_surface')
            self.assertEqual(out['checkpoint'], 'main/000010')
            self.assertEqual((out['ply_bins'], out['remaining_bins'], out['counts']), (bins, bins, [[9, 2], [0, 12]]))
            self.assertEqual(out['values'], [[.4, None], [None, .7]])
            empty = dashboard.surface(run, 'main', 'fresh_policy_surface')
            self.assertEqual((empty['checkpoint'], empty['values'], empty['counts']), (None, [], []))
            with self.assertRaises(ValueError):
                dashboard.surface(run, 'main', 'fresh_value_curve')
            server = HTTPServer(('127.0.0.1', 0), type('Handler', (dashboard.Handler,), dict(runs=root)))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                url = f'http://127.0.0.1:{server.server_port}/api/surface?run=r&variant=main&metric='
                with urlopen(url+'newest_value_surface', timeout=5) as response:
                    self.assertEqual(json.loads(response.read()), out)
                with self.assertRaises(HTTPError) as error:
                    urlopen(url+'nope', timeout=5)
                self.assertEqual(error.exception.code, 400)
                error.exception.close()
            finally:
                server.shutdown(); server.server_close()

    def test_value_curve_series_keeps_unsupported_gaps(self):
        """A curve supported on two separate ranges keeps null points between them in the response, so the chart
        breaks the line there instead of bridging the gap."""
        import dashboard
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            grid = list(dense_learn.REMAINING_GRID)
            curve = [.1, .2, None, None, None, .6, .65]+[None]*(len(grid)-7)
            for step, values in ((5, [.9]*len(grid)), (10, curve)):
                path = run/'checkpoints'/'main'/f'{step:06d}'
                path.mkdir(parents=True)
                (path/'manifest.json').write_text(json.dumps(dict(step=step, metrics=dict(
                    validation_sources=dict(remaining_grid=grid, newest_value_curve=values)))))
            out = dashboard.series(run, dict(created_at=0.), 'main', 'newest_value_curve', 'remaining')
            self.assertEqual(out['checkpoint'], 'main/000010')
            self.assertEqual(out['points'], [[g, y] for g, y in zip(grid, curve)])
            self.assertEqual([y for _, y in out['points'][1:6]], [.2, None, None, None, .6])
            json.dumps(out, allow_nan=False)

    def test_export_recalibrates_ema_norm_statistics(self):
        """The raw model drifts (here: perturbed weights) after the EMA was taken. The exported EMA must carry norm
        statistics of its own weights, so its eval-mode losses match its train-mode (batch statistics) losses,
        which the raw model's statistics do not achieve. Calibration rows are drawn with the training recency."""
        torch.set_num_threads(2)
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            source_shard(run/'shards'/'1000000000001', 2, 'x', games=12)
            config = dense_config.RunConfig(device='cpu', model=dense_config.ModelSettings(**asdict(TINY)),
                                            learner=dense_config.LearnerSettings(batch=16, validation_fraction=0., recency=.5))
            learner = dense_learn.Learner(run, config.learner, config)
            window = dense_data.ReplayWindow(run, 1000, 10)
            rng, s = np.random.default_rng(3), learner.settings
            batch = dense_data.collate(*dense_data.examples(window, window.sample(rng, 64), rng, **dense_data.target_options(s)))
            learner.model.train()
            with torch.no_grad():
                for p in learner.model.parameters():
                    p.normal_().mul_(.3)
                learner.ema = copy.deepcopy(learner.model)
                for p in learner.model.parameters():
                    p.add_(torch.randn_like(p)*.3)
                for _ in range(20):
                    dense_learn.batch_losses(learner.model, batch, None, learner.device, learner.memory_format, False)
            dense_learn.update_ema(learner.ema, learner.model, .999)

            def gap(model):
                with torch.no_grad():
                    losses = [dense_learn.batch_losses(copy.deepcopy(model).train(mode), batch, None, learner.device,
                                                       learner.memory_format, False)[:2] for mode in (False, True)]
                return float((losses[0]-losses[1]).abs().max())
            copied = copy.deepcopy(learner.ema)
            for e, m in zip(copied.buffers(), learner.model.buffers()):
                e.copy_(m)
            with unittest.mock.patch.object(window, 'sample', wraps=window.sample) as sample:
                learner.export(window)
            self.assertTrue(sample.call_args_list and all(c.args[2] == .5 for c in sample.call_args_list))
            ema = hexnet.load_model(run/'checkpoints'/'main'/'000000'/'ema.pt')
            self.assertGreater(gap(copied), .05)
            self.assertLess(gap(ema), .01)
            self.assertEqual(learner.ema.blocks[0].norm1.momentum, .1)

    def test_vram_cap_and_release_on_cuda(self):
        """vram_reserved_mb caps the allocator at that share of the device, installed from the resumed settings
        before any weights are placed; export releases the cache once after its recalibration and validation
        passes; neither touches CUDA on the CPU."""
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            source_shard(run/'shards'/'1000000000001', 2, 'x')
            config = dense_config.RunConfig(device='cpu', model=dense_config.ModelSettings(**asdict(TINY)),
                                            learner=dense_config.LearnerSettings(batch=8, validation_fraction=0., vram_reserved_mb=2048))
            with unittest.mock.patch.object(torch.cuda, 'set_per_process_memory_fraction') as cap,                  unittest.mock.patch.object(torch.cuda, 'empty_cache') as empty:
                learner = dense_learn.Learner(run, config.learner, config)
                window = dense_data.ReplayWindow(run, 1000, 10)
                with unittest.mock.patch.object(learner, 'release', wraps=learner.release) as release,                      unittest.mock.patch.object(learner, 'validate', wraps=learner.validate) as validate:
                    validate.side_effect = lambda w: release.assert_not_called()
                    learner.export(window)
                release.assert_called_once_with()
                cap.assert_not_called(); empty.assert_not_called()
                self.assertEqual(learner.vram(), dict(allocated_mb=0, reserved_mb=0))
                learner.device = torch.device('cuda', 0)
                with unittest.mock.patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(total_memory=8*2**30)),                      unittest.mock.patch.object(torch.cuda, 'current_device', return_value=1):
                    learner.cap_vram()
                    cap.assert_called_once_with(.25, 0)
                    learner.device = torch.device('cuda')
                    learner.cap_vram()
                    cap.assert_called_with(.25, 1)
                    learner.device = torch.device('cuda', 0)
                    learner.release()
                    empty.assert_called_once_with()
                    learner.settings = replace(learner.settings, vram_reserved_mb=9000)
                    with self.assertRaises(ValueError):
                        learner.cap_vram()
            seen = []
            def cap_vram(resumed):
                seen.append((hasattr(resumed, 'model'), resumed.settings.vram_reserved_mb, resumed.settings.batch))
            with unittest.mock.patch.object(dense_learn.Learner, 'cap_vram', autospec=True, side_effect=cap_vram):
                dense_learn.Learner(run, replace(config.learner, vram_reserved_mb=0), config, overrides=dict(batch=4))
            self.assertEqual(seen, [(False, 2048, 4)])  # before any weights exist, with the resumed settings

    def test_export_without_held_out_games(self):
        """No held-out game in the window: metrics.validation stays null, the sources are still reported, and the
        dashboard series skips the null aggregate."""
        import dashboard
        torch.set_num_threads(2)
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            source_shard(run/'shards'/'1000000000001', 2, 'x', checkpoint='main/000010')
            config = dense_config.RunConfig(device='cpu', model=dense_config.ModelSettings(**asdict(TINY)),
                                            learner=dense_config.LearnerSettings(batch=8, validation_fraction=0.))
            learner = dense_learn.Learner(run, config.learner, config)
            window = dense_data.ReplayWindow(run, 1000, 10)
            first = learner.export(window)['metrics']
            self.assertEqual((first['validation'], first['validation_sources']), (None, None))
            self.assertIsNone(dense_learn.validation_fields(first))
            learner.step = 1
            sets = dense_data.ValidationSets(run, 0., config.seed, limit=12, quota=6)
            metrics = learner.export(window, sets)['metrics']
            self.assertIsNone(metrics['validation'])
            self.assertEqual(metrics['validation_sources']['fresh_rows'], 0)
            self.assertTrue(math.isfinite(metrics['validation_sources']['fresh_train_policy_ce']))
            fields = dense_learn.validation_fields(metrics)
            self.assertNotIn('policy_ce', fields)
            dense_config.append_metrics(run, 'learner-main', step=1, validation=True, policy_ce=None, **fields)
            config = dict(created_at=0.)
            self.assertEqual(dashboard.series(run, config, 'main', 'validation_policy_ce')['points'], [])
            self.assertEqual(dashboard.series(run, config, 'main', 'validation_fresh_policy_ce')['points'], [])
            self.assertEqual(dashboard.series(run, config, 'main', 'validation_fresh_train_policy_ce')['points'],
                             [[1, fields['fresh_train_policy_ce']]])


class EvaluatorSearchTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.threads)
        torch.manual_seed(3)
        self.model = hexnet.HexNet(TINY).eval()
        self.evaluator = hexnet.DenseEvaluator(self.model, device='cpu')

    def test_evaluator_matches_model_including_far_cells(self):
        histories = [POSITIONS[10], [], line_history(31), line_history(6)]
        results = self.evaluator.evaluate(histories)
        for history, result in zip(histories, results):
            s = hexcrop.encode(history)
            np.testing.assert_array_equal(result['actions'], legal(history))
            planes = torch.from_numpy(s.planes[None]).float()
            with torch.no_grad():
                out = self.model(planes, planes[:, 3:4], aux=False)
            expected = out['policy'][0][torch.from_numpy(np.maximum(s.cells, 0))].numpy()
            if s.far:
                expected[s.cells < 0] = float(out['far'][0])-np.log(s.far)
            np.testing.assert_allclose(result['logits'], expected, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(result['q'], np.tanh(float(out['value_logit'][0])/2), rtol=1e-5, atol=1e-6)
            self.assertEqual((result['player'], result['remaining']), (s.player, s.remaining))
        self.assertGreater(hexcrop.encode(histories[2]).far, 0)

    def test_actor_evaluator_reuses_one_staging_set_per_pending_batch(self):
        evaluator = dense_selfplay.Evaluator(self.model, 'cpu', 'tiny', 64)
        histories = [np.asarray(h, np.int64).reshape(-1, 2) for h in [POSITIONS[10], POSITIONS[12], line_history(31)]]
        expected = evaluator.evaluate(histories)
        self.assertEqual(len(evaluator.free), 1)
        staging = evaluator.free[0]
        self.assertEqual({kind for kind, _ in staging}, {'planes', 'out'})    # one input and one output buffer per size
        pointers = {k: v.data_ptr() for k, v in staging.items()}
        # Pipelined like Engine: a second batch is submitted before the first is collected.
        first, second = evaluator.submit(histories), evaluator.submit(histories[::-1])
        self.assertIs(first[-1], staging)
        self.assertIsNot(second[-1], staging)
        for got, want in zip(evaluator.collect(first)+evaluator.collect(second)[::-1], expected+expected):
            for a, b in zip(got, want):
                np.testing.assert_array_equal(a, b)
        self.assertEqual(len(evaluator.free), 2)
        evaluator.evaluate(histories)
        self.assertEqual(len(evaluator.free), 2)
        self.assertEqual({k: v.data_ptr() for k, v in staging.items()}, pointers)

    def test_native_search_returns_legal_actions_in_native_order(self):
        for history in (POSITIONS[12], [(0, 0)], line_history(31)):
            search = NeuralSearch(self.evaluator, self.evaluator.model_version, history=history, seed=1)
            try:
                result = search.search(simulations=8, root_samples=4)
            finally:
                search.close()
            actions = legal(history)
            np.testing.assert_array_equal(result['actions'], actions)
            self.assertIn(tuple(result['action']), {tuple(a) for a in actions.tolist()})
            self.assertGreaterEqual(result['completed'], 8)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.threads)
        torch.manual_seed(5)

    def test_position_wider_than_the_largest_crop_ends_the_game(self):
        model = dense_selfplay.Model(hexnet.HexNet(TINY), 'tiny', 'test', 'cpu', 64, 256)
        openings = (line_history(33), POSITIONS[12])
        games = [dense_eval.MatchGame([model, model], h, k, 8, 4, True, len(h)+2, dict(index=k)) for k, h in enumerate(openings)]
        wide, narrow = dense_eval.play(games, 64)
        self.assertEqual((wide['winner'], wide['reason'], wide['plies']), (-1, 'span', len(openings[0])))
        self.assertNotEqual(narrow['reason'], 'span')
        self.assertGreater(narrow['plies'], len(openings[1]))

    def test_published_counts_one_worker_since_its_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            since = time.time()
            write_games(Path(tmp)/'shards'/'000001', [(winning_game(), 0, None)], dict(actor_sha256='a'*64, process=1))
            write_games(Path(tmp)/'shards'/'000002', [(winning_game(), 0, None)]*2, dict(actor_sha256='a'*64, process=0))
            self.assertEqual(dense_selfplay.published(tmp, 1, since)['games_completed'], 1)
            self.assertEqual(dense_selfplay.published(tmp, 0, since)['games_completed'], 2)
            self.assertEqual(dense_selfplay.published(tmp, 1, time.time()+1)['games_completed'], 0)


class YieldTests(unittest.TestCase):
    """dense_selfplay.Yield: the actors' cooperative pause against fake learner heartbeats and a fake clock."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run, self.clock, self.wall = Path(tmp.name), [0.], [1000.]

    def heartbeat(self, rate, variant='main', stage='training', age=0., target=None, phase_rows=None):
        name = 'learner-status.json' if variant == 'main' else f'learner-status-{variant}.json'
        extra = {} if target is None else dict(samples_per_row_target=target)
        if phase_rows is not None:
            extra['phase_rows'] = phase_rows
        (self.run/name).write_text(json.dumps(dict(stage=stage, variant=variant, samples_per_row=rate,
                                                   updated_at=self.wall[0]-age, **extra)))

    def gate(self, below=.9, resume=.975, check=30., follow=False):
        return dense_selfplay.Yield(self.run, 4., below, resume, check, follow, clock=lambda: self.clock[0],
                                    now=lambda: self.wall[0])

    def advance(self, seconds):
        self.clock[0] += seconds; self.wall[0] += seconds

    def test_pause_and_resume_with_hysteresis(self):
        gate = self.gate()
        states = []
        for rate in (3.8, 3.59, 3.7, 3.85, 3.9, 3.95, 3.8, 3.59):
            self.heartbeat(rate)
            self.advance(30.)
            states.append(gate.paused())
        # pause below 3.6, resume at 3.9, run on between the bounds in whichever state it was
        self.assertEqual(states, [False, True, True, True, False, False, False, True])
        self.assertIn('3.59', gate.reason)

    def test_checks_the_heartbeat_only_every_check_seconds(self):
        gate = self.gate(check=30.)
        self.heartbeat(2.)
        self.assertTrue(gate.paused())
        self.heartbeat(4.)
        self.advance(29.)
        self.assertTrue(gate.paused())
        self.advance(1.)
        self.assertFalse(gate.paused())

    def test_missing_stale_or_idle_learners_never_pause(self):
        gate = self.gate(check=0.)
        self.assertFalse(gate.paused())
        self.assertEqual(gate.reason, 'no training learner heartbeat')
        self.heartbeat(1., stage='waiting-for-data')
        self.assertFalse(gate.paused())
        self.heartbeat(1., age=dense_selfplay.STALE_SECONDS+1)
        self.assertFalse(gate.paused())
        (self.run/'learner-status.json').write_text('{"stage": "trai')  # torn write
        self.assertFalse(gate.paused())
        self.heartbeat(1.)
        self.assertTrue(gate.paused())
        (self.run/'learner-status.json').unlink()  # a paused actor resumes when the learner goes away
        self.assertFalse(gate.paused())

    def test_the_furthest_behind_training_learner_decides(self):
        gate = self.gate(check=0.)
        self.heartbeat(3.95)
        self.heartbeat(3.0, variant='wide')
        self.assertTrue(gate.paused())
        self.assertIn('wide', gate.reason)
        self.heartbeat(3.0, variant='wide', stage='failed')
        self.assertFalse(gate.paused())

    def test_each_learner_is_compared_with_its_own_target(self):
        gate = self.gate(check=0.)  # config target 4
        self.heartbeat(3.0, target=3.2)  # 0.94 of its own target: not behind, though below 0.9 * 4
        self.assertFalse(gate.paused())
        self.heartbeat(3.95, variant='wide', target=8.)  # 0.49 of its target: behind, though above 3.6
        self.assertTrue(gate.paused())
        self.assertIn('wide', gate.reason)
        self.assertIn('pause below 7.20', gate.reason)
        self.heartbeat(7.9, variant='wide', target=8.)
        self.assertTrue(gate.paused())  # main at 0.94 of its target is below the resume bound
        self.heartbeat(3.15, target=3.2)
        self.assertFalse(gate.paused())
        self.heartbeat(3.5)  # no samples_per_row_target: the config target applies
        self.assertTrue(gate.paused())

    def test_the_pacing_rule_alone_does_not_follow_the_phases(self):
        gate = self.gate(check=0.)
        self.heartbeat(3.88, stage='training', phase_rows=12000)  # the live ratio: a training phase, no pause
        self.assertFalse(gate.paused())
        self.heartbeat(3.0, stage='phase-idle', phase_rows=12000)  # far behind, but idling: no pause
        self.assertFalse(gate.paused())

    def test_phase_follow_pauses_exactly_during_the_training_phase(self):
        gate = self.gate(below=0., check=0., follow=True)
        states = []
        for stage in ('phase-idle', 'training', 'exporting', 'training', 'phase-idle', 'waiting-for-data', 'training'):
            self.heartbeat(3.99, stage=stage, phase_rows=12000)
            states.append(gate.paused())
        self.assertEqual(states, [False, True, True, True, False, False, True])
        self.assertIn('main in its training phase', gate.reason)
        self.heartbeat(3.99, stage='training', phase_rows=12000, age=dense_selfplay.STALE_SECONDS+1)
        self.assertFalse(gate.paused())
        self.heartbeat(3.99, stage='training', phase_rows=0)  # an unphased learner is not followed
        self.assertFalse(gate.paused())
        self.heartbeat(3.99, stage='training')  # nor one whose heartbeat predates phase_rows
        self.assertFalse(gate.paused())

    def test_phase_follow_and_the_pacing_rule_combine(self):
        gate = self.gate(check=0., follow=True)
        self.heartbeat(3.99, stage='training', phase_rows=12000)
        self.heartbeat(3.0, variant='wide')
        self.assertTrue(gate.paused())
        self.heartbeat(3.99, stage='phase-idle', phase_rows=12000)
        self.assertTrue(gate.paused())  # wide is still behind its pacing
        self.assertIn('wide', gate.reason)
        self.heartbeat(3.95, variant='wide')
        self.assertFalse(gate.paused())
        self.assertFalse(self.gate(below=0., check=0.).paused())  # neither rule enabled

    def test_metrics_lines_on_every_stage_change_and_periodically(self):
        due = dense_selfplay.metrics_due
        self.assertTrue(due('paused', 'playing', 0.))
        self.assertTrue(due('playing', 'paused', 0.))  # the resume is logged at once
        self.assertFalse(due('paused', 'paused', dense_selfplay.METRICS_SECONDS-1))
        self.assertTrue(due('paused', 'paused', dense_selfplay.METRICS_SECONDS))
        self.assertFalse(due('playing', 'playing', 1.))

    def test_learner_heartbeat_reports_its_effective_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/'run'
            made = subprocess.run([sys.executable, str(ROOT/'dense_config.py'), '--run', str(run), '--device', 'cpu',
                                   '--blocks', '1', '--channels', '16'], capture_output=True, text=True, cwd=ROOT, timeout=60)
            self.assertEqual(made.returncode, 0, made.stderr)
            write_games(run/'shards'/'000001', [(winning_game(), 0, None)]*8)
            done = subprocess.run([sys.executable, str(ROOT/'dense_learn.py'), '--run', str(run), '--steps', '1',
                                   '--workers', '1', '--batch', '8', '--window-min-rows', '1', '--samples-per-row', '7.5',
                                   '--validation-fraction', '0', '--log-every', '1', '--vram-reserved-mb', '1500'],
                                  capture_output=True, text=True, cwd=ROOT, timeout=300)
            self.assertEqual(done.returncode, 0, done.stderr)
            status = json.loads((run/'learner-status.json').read_text())
            self.assertEqual(status['samples_per_row_target'], 7.5)
            self.assertEqual(status['phase_rows'], 0)
            self.assertAlmostEqual(status['backlog_rows'], status['rows_available']-status['samples_seen']/7.5)
            self.assertEqual((status['pacing_rows'], status['pacing_samples']), (0, 0))
            zeros = dict(allocated_mb=0, reserved_mb=0)
            self.assertEqual(status['vram'], zeros)
            lines = [json.loads(line) for line in (run/'metrics'/'learner-main.jsonl').read_text().splitlines()]
            self.assertEqual([(r.get('validation', False), r['vram']) for r in lines], [(False, zeros), (True, zeros)])
            manifest = json.loads((run/'checkpoints'/'main'/'000001'/'manifest.json').read_text())
            self.assertEqual(manifest['learner']['vram_reserved_mb'], 1500)

    def test_disabled_and_invalid_bounds(self):
        self.heartbeat(0.)
        self.assertFalse(self.gate(below=0.).paused())
        with self.assertRaises(ValueError):
            self.gate(below=.99, resume=.9)

    def test_actor_flags_round_trip_through_the_worker_parser(self):
        parser = argparse.ArgumentParser()
        dense_config.add_arguments(parser, dense_config.ActorSettings)
        args = parser.parse_args(['--games-in-flight', '256', '--no-tactics', '--yield-below', '0.8'])
        flags = dense_selfplay.actor_flags(args)
        self.assertEqual(flags, ['--games-in-flight', '256', '--no-tactics', '--yield-below', '0.8'])
        settings = dense_config.override(dense_config.ActorSettings(), parser.parse_args(flags))
        self.assertEqual((settings.games_in_flight, settings.tactics, settings.yield_below, settings.leaf_batch),
                         (256, False, .8, dense_config.ActorSettings.leaf_batch))

    def test_configs_written_before_the_yield_settings_load_with_the_defaults(self):
        data = asdict(dense_config.RunConfig())
        for name in ('yield_below', 'yield_resume', 'yield_check_seconds'):
            del data['actor'][name]
        self.assertEqual(dense_config.from_dict(data).actor.yield_below, dense_config.ActorSettings.yield_below)


class PhaseTests(unittest.TestCase):
    """dense_learn.backlog and dense_learn.Phase: the phased schedule against synthetic row arrivals."""

    def test_backlog_is_the_unspent_pacing_budget_in_rows(self):
        self.assertEqual(dense_learn.backlog(0, 100, 4.), 100)
        self.assertEqual(dense_learn.backlog(400, 100, 4.), 0)
        self.assertEqual(dense_learn.backlog(200, 100, 4.), 50)
        self.assertEqual(dense_learn.backlog(10496000, 2705320, 4.), 81320)  # the live heartbeat at step 41000

    def test_backlog_and_pacing_count_from_the_base(self):
        base = dict(rows=3_800_000, samples=14_800_000)
        self.assertEqual(dense_learn.backlog(14_800_000, 3_800_000, 3., base), 0)
        self.assertEqual(dense_learn.backlog(14_800_300, 3_800_400, 3., base), 300)
        self.assertLess(dense_learn.backlog(14_800_000, 3_800_000, 3.), -1e6)  # without the base: a deficit
        self.assertTrue(dense_learn.paced(14_800_000, 3_800_000, 3., 1, base))
        self.assertFalse(dense_learn.paced(14_800_000, 3_800_100, 3., 300, base))
        self.assertTrue(dense_learn.paced(14_800_000, 3_800_100, 3., 301, base))
        for seen, rows in ((0, 0), (400, 100), (401, 100), (100, 30)):
            self.assertEqual(dense_learn.paced(seen, rows, 4., 8), seen+8 > 4.*rows)
            self.assertEqual(dense_learn.backlog(seen, rows, 4.), rows-seen/4.)

    def test_resume_moves_the_pacing_base_only_when_samples_per_row_changes(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            source_shard(run/'shards'/'1000000000001', 2, 'x')
            config = dense_config.RunConfig(device='cpu', model=dense_config.ModelSettings(**asdict(TINY)),
                                            learner=dense_config.LearnerSettings(batch=8, validation_fraction=0.))
            window = dense_data.ReplayWindow(run, 1000, 10)
            first = dense_learn.Learner(run, config.learner, config)
            first.samples_seen = 5000
            manifest = first.export(window)
            rows = window.total_rows
            self.assertEqual((manifest['rows'], manifest['pacing']), (rows, dict(rows=0, samples=0)))
            same = dense_learn.Learner(run, config.learner, config)
            same.rebase(rows+100)
            self.assertEqual(same.pacing, dict(rows=0, samples=0))
            events = lambda: [e for e in map(json.loads, (run/'events.jsonl').read_text().splitlines()) if 'pacing' in e]
            self.assertEqual(events(), [])
            # The base is the checkpoint's rows, whatever arrived before each restart that did not export.
            for arrived in (100, 300):
                lower = dense_learn.Learner(run, config.learner, config, overrides=dict(samples_per_row=3.))
                lower.rebase(rows+arrived)
                self.assertEqual(lower.pacing, dict(rows=rows, samples=5000))
                self.assertLess(dense_learn.backlog(5000, rows+arrived, 4.), 0)
                self.assertEqual(dense_learn.backlog(5000, rows+arrived, 3., lower.pacing), arrived)
            event = events()[-1]
            self.assertEqual((event['pacing'], event['old_samples_per_row'], event['new_samples_per_row']),
                             (dict(rows=rows, samples=5000), 4., 3.))
            lower.rebase(rows+500)
            self.assertEqual(lower.pacing, dict(rows=rows, samples=5000))
            lower.settings = replace(lower.settings, samples_per_row=2.)  # a replacement copy's setting
            lower.rebase(rows+500)
            self.assertEqual(lower.pacing, dict(rows=rows+500, samples=5000))
            lower.settings = replace(lower.settings, samples_per_row=3.)
            lower.rebase(rows+500)
            lower.step = 1
            self.assertEqual(lower.export(window)['pacing'], dict(rows=rows+500, samples=5000))
            kept = dense_learn.Learner(run, config.learner, config, overrides=dict(samples_per_row=3.))
            kept.rebase(rows+900)
            self.assertEqual(kept.pacing, dict(rows=rows+500, samples=5000))
            self.assertEqual(len(events()), 4)

    def test_manifests_without_a_pacing_base_pace_from_zero(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            source_shard(run/'shards'/'1000000000001', 2, 'x')
            config = dense_config.RunConfig(device='cpu', model=dense_config.ModelSettings(**asdict(TINY)),
                                            learner=dense_config.LearnerSettings(batch=8, validation_fraction=0.))
            learner = dense_learn.Learner(run, config.learner, config)
            learner.samples_seen = 5000
            learner.export(dense_data.ReplayWindow(run, 1000, 10))
            path = run/'checkpoints'/'main'/'000000'/'manifest.json'
            manifest = json.loads(path.read_text())
            del manifest['pacing'], manifest['rows']
            path.write_text(json.dumps(manifest))
            resumed = dense_learn.Learner(run, config.learner, config)
            resumed.rebase(1000)
            self.assertEqual(resumed.pacing, dense_learn.NO_BASE)
            self.assertEqual(dense_learn.backlog(5000, 1000, 4., resumed.pacing), dense_learn.backlog(5000, 1000, 4.))
            lower = dense_learn.Learner(run, config.learner, config, overrides=dict(samples_per_row=3.))
            lower.rebase(1000)  # no manifest rows: the rows at the rebase
            self.assertEqual(lower.pacing, dict(rows=1000, samples=5000))

    def simulate(self, phase_rows, ticks, arrivals=90, batch=256, per_row=4., rows=0, seen=0):
        """One tick: the learner takes one batch if Phase says so and the pacing allows it, otherwise actors
        publish `arrivals` rows (they pause while the learner trains). Returns (stage, backlog) per tick."""
        phase, trace = dense_learn.Phase(), []
        for _ in range(ticks):
            paced = seen+batch > per_row*rows
            if not phase.due(phase_rows, dense_learn.backlog(seen, rows, per_row), paced):
                trace.append(('phase-idle', dense_learn.backlog(seen, rows, per_row)))
                rows += arrivals
            elif paced:
                trace.append(('waiting-for-data', dense_learn.backlog(seen, rows, per_row)))
                rows += arrivals
            else:
                trace.append(('training', dense_learn.backlog(seen, rows, per_row)))
                seen += batch
                if not phase_rows:
                    rows += arrivals
        return trace

    def test_phases_alternate_on_the_backlog(self):
        trace = self.simulate(1000, 200)
        stages = [s for s, _ in trace]
        runs = [stages[0]]
        for s in stages[1:]:
            if s != runs[-1]:
                runs.append(s)
        self.assertEqual(runs[:5], ['phase-idle', 'training', 'phase-idle', 'training', 'phase-idle'])
        self.assertNotIn('waiting-for-data', stages)
        for (stage, backlog), (after, _) in zip(trace, trace[1:]):
            if stage == 'phase-idle' and after == 'training':
                self.assertGreaterEqual(backlog+90, 1000)  # a phase starts once the backlog reaches phase_rows
            if stage == 'phase-idle':
                self.assertLess(backlog, 1000)
            if stage == 'training' and after == 'phase-idle':
                self.assertLess(backlog-256/4., 256/4.)  # and ends only when the next batch is out of budget
        first = stages.index('training')
        self.assertEqual(stages[first:first+16], ['training']*16)  # ~1000 rows * 4 / 256 batches, uninterrupted

    def test_a_backlog_above_phase_rows_trains_at_once(self):
        trace = self.simulate(1000, 3, rows=5000)
        self.assertEqual([s for s, _ in trace], ['training']*3)

    def test_phase_rows_zero_trains_whenever_the_pacing_allows(self):
        stages = [s for s, _ in self.simulate(0, 40)]
        self.assertEqual(stages[0], 'waiting-for-data')
        self.assertNotIn('phase-idle', stages)
        self.assertIn('training', stages)
        phase = dense_learn.Phase()
        self.assertTrue(phase.due(0, 0., True))
        self.assertTrue(phase.due(0, 1e9, False))

    def test_defaults_keep_the_unphased_schedule(self):
        self.assertEqual(dense_config.LearnerSettings().phase_rows, 0)
        self.assertIs(dense_config.ActorSettings().phase_follow, False)
        data = asdict(dense_config.RunConfig())
        del data['learner']['phase_rows'], data['actor']['phase_follow']
        config = dense_config.from_dict(data)
        self.assertEqual((config.learner.phase_rows, config.actor.phase_follow), (0, False))
        self.assertIn('phase_rows', dense_learn.KEEP)
        with self.assertRaises(ValueError):
            dense_config.LearnerSettings(phase_rows=-1)


class ActorModelTests(unittest.TestCase):
    """dense_selfplay.resolve per model_source and the worker's switch between games."""

    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.threads)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run = Path(tmp.name)

    def export(self, cid, created_at):
        path = self.run/'checkpoints'/cid
        path.mkdir(parents=True)
        hexnet.save_model(path/'ema.pt', hexnet.HexNet(TINY))
        (path/'manifest.json').write_text(json.dumps(dict(created_at=created_at)))

    def pick(self, source):
        return dense_selfplay.resolve(self.run, None, source, 'main')[0]

    def test_sources(self):
        self.assertEqual(dense_config.ActorSettings().model_source, 'newest_veto')
        data = asdict(dense_config.RunConfig())
        del data['actor']['model_source'], data['evaluation']['veto_margin']
        loaded = dense_config.from_dict(data)
        self.assertEqual((loaded.actor.model_source, loaded.evaluation.veto_margin), ('newest_veto', -30.))
        self.export('main/000010', 1.)
        self.export('main/000020', 2.)
        self.export('wide/000030', 3.)
        self.assertEqual([self.pick(s) for s in ('champion', 'newest', 'newest_veto')], ['wide/000030', 'main/000020', 'wide/000030'])
        (self.run/'champion.json').write_text(json.dumps(dict(checkpoint='main/000010')))
        self.assertEqual([self.pick(s) for s in ('champion', 'newest', 'newest_veto')], ['main/000010', 'main/000020', 'main/000010'])
        (self.run/'actor.json').write_text(json.dumps(dict(checkpoint='main/000020', reason='newest', vetoed=[])))
        self.assertEqual([self.pick(s) for s in ('champion', 'newest', 'newest_veto')], ['main/000010', 'main/000020', 'main/000020'])
        with self.assertRaises(ValueError):
            self.pick('latest')

    def test_worker_switches_between_games(self):
        """The pointer moves as the first shard is written: the game started before it keeps the first model, the
        next game plays the second, and the switch is an 'actor_model' event."""
        self.export('main/000010', 1.)
        self.export('main/000020', 2.)
        config = dense_config.RunConfig(
            device='cpu', model=dense_config.ModelSettings(**{k: getattr(TINY, k) for k in (
                'blocks', 'channels', 'pool_every', 'line_length', 'value_hidden', 'head_channels')}),
            actor=dense_config.ActorSettings(games_in_flight=1, leaf_batch=64, full_sims=2, cheap_sims=2, root_samples=2,
                                             max_plies=6, cache_positions=256, shard_games=1, opening_random_plies=0.))
        dense_config.save(self.run, config)
        pointer = lambda cid: (self.run/'actor.json').write_text(json.dumps(dict(checkpoint=cid, reason='newest', vetoed=[])))
        pointer('main/000010')
        write_shard = dense_data.write_shard

        def publish(path, identity, *args):
            pointer('main/000020')
            return write_shard(path, identity, *args)
        with unittest.mock.patch.object(dense_data, 'write_shard', publish):
            dense_selfplay.worker(SimpleNamespace(run=str(self.run), worker=0, games=2, initial_model=None))
        shards = [dense_data.manifest(path)['identity'] for path in dense_data.shard_dirs(self.run)]
        self.assertEqual([s['checkpoint'] for s in shards], ['main/000010', 'main/000020'])
        self.assertEqual(json.loads((self.run/'actor-status.json').read_text())['checkpoint'], 'main/000020')
        events = [json.loads(line) for line in (self.run/'events.jsonl').read_text().splitlines()]
        self.assertEqual([(e['previous'], e['checkpoint']) for e in events if e['kind'] == 'actor_model'],
                         [('main/000010', 'main/000020')])


class PacerTests(unittest.TestCase):
    def test_share_ceiling_with_a_fake_clock(self):
        now, slept, ticks = [0.], [], []
        def sleep(seconds):
            slept.append(seconds); now[0] += seconds
        pacer = dense_eval.Pacer(.25, window=100., clock=lambda: now[0], sleep=sleep)
        pacer.wait(lambda: ticks.append(now[0]))
        self.assertEqual(slept, [])  # a new pacer starts with share * window = 25 s of credit
        now[0] = 40.
        pacer.played(0., 40.)
        self.assertAlmostEqual(pacer.used(), 1.)
        pacer.wait(lambda: ticks.append(now[0]))
        # 25 s of credit + 10 s accrued while playing - 40 s played: 5 s of debt repaid at a quarter per second.
        self.assertAlmostEqual(sum(slept), 20.)
        self.assertLessEqual(max(slept), 10.)
        self.assertEqual(len(ticks), len(slept))
        self.assertAlmostEqual(pacer.used(), 40/60)
        now[0] = 1000.
        pacer.wait()
        self.assertAlmostEqual(sum(slept), 20.)
        self.assertAlmostEqual(pacer.used(), 0.)
        pacer.played(1000., 1200.)  # idle credit is capped at 25 s
        now[0] = 1200.
        pacer.wait()
        self.assertAlmostEqual(sum(slept), 20.+(150-25)/.25)
        full = dense_eval.Pacer(1., clock=lambda: now[0], sleep=sleep)
        now[0] = 9000.
        full.played(1200., 9000.)
        full.wait()
        self.assertAlmostEqual(sum(slept), 20.+500.)
        with self.assertRaises(ValueError):
            dense_eval.Pacer(0.)


def fake_report(candidate, opponent, wins, losses, capped, settings=None):
    games = wins+losses+capped
    return dict(candidate=candidate, opponent=opponent, settings=asdict(settings or dense_config.EvaluationSettings()),
                summary=dict(wins=wins, losses=losses, capped=capped, games=games), metrics={}, games=[])


def league_of(elos, champion=None, matrix=None):
    """A league.json dict with checkpoints '<variant>/<step:06d>' -> Elo (None: unrated)."""
    league = dict(champion=champion, checkpoints=[dict(id=k, variant=k.split('/')[0], step=int(k.split('/')[1]), elo=e, matches=[])
                                                  for k, e in elos.items()])
    if matrix is not None:
        league['matrix'] = matrix
    return league


class PosteriorTests(unittest.TestCase):
    """dense_posterior.Posterior on synthetic results (a, b, points of a, games)."""

    def test_pooled_and_direct_estimates(self):
        from dense_posterior import Posterior
        results = [('a', 'b', 12, 20), ('a', 'c', 30, 40), ('b', 'c', 20, 40)]
        direct = Posterior(['a', 'b', 'c'], 'c', results, 1e4)          # a free deviation: a-b from its own games
        pooled = Posterior(['a', 'b', 'c'], 'c', results, 0.)           # transitive Bradley-Terry
        mean, sd = direct.difference('a', 'b')
        self.assertAlmostEqual(mean, 400*math.log10(12/8), delta=2)
        pooled_mean, pooled_sd = pooled.difference('a', 'b')
        self.assertTrue(400*math.log10(12/8) < pooled_mean < 400*math.log10(3))  # pulled toward the indirect a-c-b path
        self.assertLess(pooled_sd, sd)
        self.assertEqual(Posterior(['a', 'b', 'c'], 'c', results, 30.).difference('a', 'b', False)[0] > 0, True)

    def test_direct_games_dominate_a_non_transitive_triangle(self):
        from dense_posterior import Posterior
        results = [('a', 'b', 900, 1000), ('b', 'c', 900, 1000), ('c', 'a', 900, 1000)]
        post = Posterior(['a', 'b', 'c'], 'c', results, 30.)
        direct = 400*math.log10(9)
        self.assertAlmostEqual(post.difference('a', 'b', False)[0], 0., delta=1)  # the transitive picture: a tie
        self.assertGreater(post.difference('a', 'b')[0], direct/2)            # its own games dominate
        self.assertLess(post.difference('b', 'a')[0], -direct/2)

    def test_a_fresh_candidate_is_centred_on_its_previous_export(self):
        from dense_posterior import Posterior, parents
        ids = ['main/000010', 'main/000020', 'main/000030', 'main/000040', 'main/000030@solver', 'side/000005', 'seal']
        self.assertEqual(parents(ids), {'main/000020': 'main/000010', 'main/000030': 'main/000020', 'main/000040': 'main/000030',
                                        'main/000030@solver': 'main/000030'})
        # The champion main/000030 sits far above the anchor; the candidate splits 20-20 with it.
        results = [('main/000020', 'main/000010', 380, 400), ('main/000030', 'main/000020', 380, 400),
                   ('main/000040', 'main/000030', 20, 40)]
        ids = ['main/000010', 'main/000020', 'main/000030', 'main/000040']
        centred = Posterior(ids, ids[0], results, 30., parents(ids)).difference('main/000040', 'main/000030', False)[0]
        self.assertLess(abs(centred), 3.)
        self.assertLess(Posterior(ids, ids[0], results, 30.).difference('main/000040', 'main/000030', False)[0], centred-3)

    def test_value_of_information_prefers_the_pairing_that_resolves_delta(self):
        from dense_posterior import Posterior
        best = lambda post: min((('cand', 'champ'), ('cand', 'prev'), ('champ', 'prev')),
                                key=lambda pair: post.after(('cand', 'champ', True), pair, 8))
        # No indirect evidence about the candidate: only direct games inform delta.
        post = Posterior(['champ', 'prev', 'cand'], 'champ', [('champ', 'prev', 5, 10)], 30.)
        self.assertEqual(best(post), ('cand', 'champ'))
        # The candidate is lopsided against the champion (p ~ .95) but even with the well-measured previous
        # champion: a round against it resolves delta faster than another lopsided direct round.
        results = [('prev', 'champ', 950, 1000), ('cand', 'champ', 38, 40)]
        post = Posterior(['champ', 'prev', 'cand'], 'champ', results, 1.)
        self.assertEqual(best(post), ('cand', 'prev'))


class OpponentSchedulerTests(unittest.TestCase):
    def test_payoff_matrix_from_reports(self):
        reports = [fake_report('main/000002', 'main/000001', 5, 2, 1), fake_report('main/000001', 'main/000002', 3, 3, 2),
                   fake_report('main/000002', 'seal', 1, 6, 1)]
        ratings = {'main/000001': 0., 'main/000002': 100., 'seal': None}
        matrix = dense_eval.payoff(reports, ratings)
        self.assertEqual({k: v for k, v in matrix['main/000002']['main/000001'].items() if k != 'p'},
                         dict(wins=8, losses=5, capped=3, games=16))
        self.assertEqual(matrix['main/000001']['main/000002']['wins'], 5)
        self.assertAlmostEqual(matrix['main/000002']['main/000001']['p'], 1/(1+10**(-100/400)))
        self.assertAlmostEqual(matrix['main/000001']['main/000002']['p']+matrix['main/000002']['main/000001']['p'], 1)
        self.assertIsNone(matrix['seal']['main/000002']['p'])
        self.assertEqual(matrix['seal']['main/000002']['wins'], 6)
        self.assertIsNone(dense_eval.payoff(reports)['main/000001']['main/000002']['p'])

    def test_panel_picks_the_closest_rated_checkpoints(self):
        """Members are the rated checkpoints closest to the current champion (Elo 0) by p(1-p), ties toward wider
        intervals and newer entries; lopsided ones (expected .95) are excluded; the set follows the ratings."""
        elos = {'main/000001': 0., 'main/000002': -400*math.log10(19), 'main/000003': -400*math.log10(7/3),
                'main/000004': 30., 'main/000005': -30., 'main/000006': 150., 'main/000007': None}
        league = league_of(elos, 'main/000001')
        members = lambda count, cap=.85: dense_eval.panel_members(league, 'main/000008', 'main/000006', count, cap)
        self.assertEqual(members(5), ['main/000005', 'main/000004', 'main/000003'])   # 4 and 5 tie: newer first
        league['checkpoints'][3]['elo_interval'] = [-100., 160.]
        self.assertEqual(members(2), ['main/000004', 'main/000005'])                  # the wider interval breaks the tie
        league['checkpoints'][3]['elo'], league['checkpoints'][4]['elo'] = -250., -300.  # ratings moved
        self.assertEqual(members(1), ['main/000003'])
        league['checkpoints'][4]['demoted'] = True
        self.assertEqual(members(5), ['main/000003', 'main/000004'])                  # main/000002 (expected .95) never
        self.assertIn('main/000002', members(5, 1.))

    def test_panel_veto_arithmetic(self):
        cell = lambda w, l: dict(wins=w, losses=l, capped=3, games=w+l+3)
        matrix = {'cand': {'x': cell(10, 30), 'y': cell(5, 15), 'z': cell(9, 0)},
                  'inc': {'x': cell(25, 15), 'y': cell(12, 8)}}
        got = dense_eval.panel_result(['x', 'y', 'z'], 'cand', 'inc', matrix)
        wc, nc, wi, ni = 15, 60, 37, 60                                  # z has no incumbent games: excluded
        pooled = (wc+wi)/(nc+ni)
        z = (wc/nc-wi/ni)/math.sqrt(pooled*(1-pooled)*(2/60))
        self.assertAlmostEqual(got['candidate_score'], .25)
        self.assertAlmostEqual(got['incumbent_score'], 37/60)
        self.assertAlmostEqual(got['z'], z)
        self.assertTrue(got['veto'])
        self.assertEqual(got['members'], ['x', 'y', 'z'])
        close = dense_eval.panel_result(['x'], 'cand', 'inc', {'cand': {'x': cell(18, 22)}, 'inc': {'x': cell(22, 18)}})
        self.assertGreater(close['z'], -1.96)
        self.assertFalse(close['veto'])
        empty = dense_eval.panel_result(['x'], 'cand', 'inc', {'cand': {'x': cell(3, 1)}})
        self.assertEqual((empty['incumbent_score'], empty['z'], empty['veto']), (None, None, False))

    def test_pfsp_weights(self):
        elos = {'main/000001': 0., 'main/000002': 100., 'main/000003': 100., 'side/000001': None}
        matrix = {'main/000003': {'main/000002': dict(wins=0, losses=16, capped=0, games=16)}}
        league = league_of(elos, 'main/000003', matrix)
        weights = dense_selfplay.opponent_weights(league, 'main/000003', 'pfsp')
        self.assertEqual(set(weights), {'main/000001', 'main/000002'})
        self.assertAlmostEqual(weights['main/000001'], (1-dense_selfplay.expected(100., 0.))**2)
        self.assertAlmostEqual(weights['main/000002'], (1-(16*.5+0)/32)**2)     # a recorded 0-16 raises the weight
        self.assertEqual(dense_selfplay.opponent_weights(league, 'main/000003', 'uniform'), {'main/000001': 1., 'main/000002': 1.})
        self.assertEqual(dense_selfplay.opponent_weights({}, 'fresh', 'pfsp'), {})
        unrated = dense_selfplay.opponent_weights(league_of(elos, 'side/000001'), 'side/000001', 'pfsp')
        self.assertEqual(set(unrated.values()), {.25})
        with self.assertRaises(ValueError):
            dense_selfplay.opponent_weights(league, 'main/000003', 'other')

    def test_opponent_pool_and_grouping(self):
        rng = np.random.default_rng(2)
        weights = {'a': 1., 'b': 3., 'c': 0., 'd': 2.}
        pool = dense_selfplay.draw_pool(weights, 2, rng)
        self.assertEqual(len(set(pool)), 2)
        self.assertNotIn('c', pool)
        self.assertEqual(sorted(dense_selfplay.draw_pool(weights, 8, rng)), ['a', 'b', 'd'])
        config = replace(dense_config.RunConfig(), actor=dense_config.ActorSettings(games_in_flight=16, historical_fraction=.5))
        with tempfile.TemporaryDirectory() as tmp:
            historical = dense_selfplay.Historical(tmp, config, rng)
            historical.redraw('main/000001', 'a'*64)                  # no league yet: no opponents
            self.assertEqual(historical.models, {})
        self.assertEqual((historical.target, historical.block), (8, 4))
        historical.models, historical.weights = {'a': 'A', 'b': 'B'}, {'a': 1., 'b': 1.}
        drawn, colours = zip(*(historical.next() for _ in range(40)))
        self.assertEqual(set(drawn), {'A', 'B'})
        for k in range(0, 40, 4):
            self.assertEqual(len(set(drawn[k:k+4])), 1)               # blocks of target/BLOCKS games per opponent
        self.assertEqual(colours, (0, 1)*20)                           # alternating over historical games only
        # A run shorter than games_in_flight keeps the fraction: 8 games, half historical.
        self.assertEqual(dense_selfplay.Historical('.', config, rng, games=8).target, 4)
        self.assertEqual(dense_selfplay.Historical('.', config, rng, games=100).target, 8)

    def test_pool_excludes_opponents_with_the_champion_digest(self):
        config = replace(dense_config.RunConfig(device='cpu'), model=dense_config.ModelSettings(**asdict(TINY)),
                         actor=dense_config.ActorSettings(historical_fraction=.5))
        with tempfile.TemporaryDirectory() as tmp:
            for step in (1, 2):
                path = Path(tmp)/'checkpoints'/'main'/f'{step:06d}'
                path.mkdir(parents=True)
                hexnet.save_model(path/'ema.pt', hexnet.HexNet(TINY))
            (Path(tmp)/'league.json').write_text(json.dumps(dict(champion='main/000003', checkpoints=[
                dict(id='main/000001', variant='main', step=1, elo=0., ema_sha256='c'*64),     # re-exported champion weights
                dict(id='main/000002', variant='main', step=2, elo=0., ema_sha256='d'*64),
                dict(id='main/000003', variant='main', step=3, elo=0., ema_sha256='c'*64)])))
            historical = dense_selfplay.Historical(tmp, config, np.random.default_rng(0))
            historical.redraw('main/000003', 'c'*64)
            self.assertEqual(list(historical.models), ['main/000002'])

    def test_mixed_games_mask_the_opponent_plies(self):
        torch.manual_seed(8)
        champion = dense_selfplay.Model(hexnet.HexNet(TINY), 'c'*64, 'main/000002', 'cpu', 64, 256)
        opponent = dense_selfplay.Model(hexnet.HexNet(TINY), 'o'*64, 'main/000001', 'cpu', 64, 256)
        twin = dense_selfplay.Model(hexnet.HexNet(TINY), 'c'*64, 'main/000001', 'cpu', 64, 256)   # same digest as the champion
        settings = dense_config.ActorSettings(full_sims=4, cheap_sims=2, root_samples=2, max_plies=14, full_fraction=.5,
                                              opening_random_plies=0.)
        games = [dense_selfplay.SelfPlayGame([champion, opponent], settings, 1, 0, 'main/000001'),
                 dense_selfplay.SelfPlayGame([opponent, champion], settings, 2, 1, 'main/000001'),
                 dense_selfplay.SelfPlayGame([champion, champion], settings, 3),
                 dense_selfplay.SelfPlayGame([twin, champion], settings, 4, 1, 'main/000001')]
        engine = dense_selfplay.Engine(64)
        for g in games:
            engine.add(g)
        finished = []
        while engine.slots:
            finished += engine.step()
        self.assertEqual(engine.calls >= 2, True)
        episodes, rows = [], []
        for g in games:
            e, items = g.episode()
            episodes.append(e)
            rows += [dict(r, game=len(episodes)-1) for r in items]
        mixed, swapped, plain, twinned = episodes
        self.assertEqual([e['trained_side'] for e in episodes], [0, 1, None, 1])
        self.assertEqual((mixed['actors'], mixed['actor'], mixed['opponent']), ({'0': 'c'*64, '1': 'o'*64}, 'c'*64, 'main/000001'))
        self.assertEqual((swapped['actors'], swapped['actor']), ({'0': 'o'*64, '1': 'c'*64}, 'c'*64))
        self.assertEqual((plain['actors'], plain['opponent']), ({'0': 'c'*64, '1': 'c'*64}, None))
        for g, e in enumerate(episodes):
            for r in (r for r in rows if r['game'] == g):
                mine = dense_data.trained(e, r['ply'])
                self.assertEqual(mine, g == 2 or dense_data.player_at(r['ply']) == [0, 1, None, 1][g])
                if not mine:
                    self.assertIsNone(r['policy'])
                    self.assertIsNone(e['root_values'][r['ply']])
                    self.assertFalse(e['full_search'][r['ply']])
            self.assertEqual([r['ply'] for r in rows if r['game'] == g], list(range(len(e['moves']))))
        with tempfile.TemporaryDirectory() as tmp:
            manifest = dense_data.write_shard(Path(tmp)/'shards'/'000001', dict(actor_sha256='c'*64), episodes, rows)
            skipped = sum(not dense_data.trained(episodes[r['game']], r['ply']) for r in rows)
            self.assertGreater(skipped, 0)
            self.assertEqual(manifest['counts']['opponent_rows'], skipped)
            self.assertEqual(dense_bootstrap.check(Path(tmp)/'shards'/'000001'), len(rows))
            window = dense_data.ReplayWindow(tmp, capacity_rows=1000, min_rows=1000)
            self.assertEqual(window.total_rows, len(rows)-skipped)
            refs = [window.ref(*k) for k in window.index]
            self.assertEqual(len(refs), len(rows)-skipped)
            self.assertTrue(all(dense_data.trained(r.episode, r.row['ply']) for r in refs))
            _, targets = dense_data.examples(window, refs, np.random.default_rng(0))
            for ref, t in zip(refs, targets):
                if ref.episode['winner'] < 0:
                    self.assertTrue(t['value_weight'] == 0 or t['value'] is not None)
                # The opponent's next ply never supplies an opponent-policy target.
                if ref.row['game'] != 2 and ref.row['ply']+1 < len(ref.episode['moves']) and not dense_data.trained(ref.episode, ref.row['ply']+1):
                    self.assertEqual(t['next_weight'], 0.)


class Crash(Exception):
    """Raised by a scripted pool to stop the evaluator mid-session, as a killed process would."""


def scripted(winner=lambda record: record['challenger_color'], hook=lambda pool, steps: None, moves=lambda record: 5,
             order=lambda record: 0):
    """A dense_eval.Pool stand-in whose step() first calls hook(pool, steps so far) and then finishes the game in
    flight of least order(record) (the oldest among equals) after moves(record) placements, won by colour
    winner(record) (-1: a cap; default the candidate)."""
    class Scripted(dense_eval.Pool):
        def step(self):
            self.steps = getattr(self, 'steps', 0)+1
            hook(self, self.steps)
            out, self.ready = self.ready, []
            if self.games:
                lane, game = self.games.pop(min(self.games, key=lambda k: order(self.games[k][1].record)))
                self.engine.slots.remove(game)
                game.game.close()
                for tree in game.trees.values():
                    tree.close()
                out.append((lane, dict(game.record, winner=winner(game.record), reason='six-in-a-row',
                                       plies=len(game.record['opening'])+moves(game.record), moves=[])))
            return out
    return Scripted


class EvaluatorLoopTests(unittest.TestCase):
    """Evaluator.step on a CPU run of TINY checkpoints with two-sim, eight-ply games."""

    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.threads)
        torch.manual_seed(11)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run = Path(tmp.name)

    def start(self, **evaluation):
        """An Evaluator of two-sim, eight-ply games whose pool holds one colour pair (pool_games 2) unless set."""
        settings = dict(games=2, pool_games=2, sims=2, root_samples=2, max_plies=8, anchor_games=0, sprt_max_games=2,
                        extra_opponents=0, decision='sprt', sprt_min_games=2, idle_rematch=False, idle_fill=False)
        config = dense_config.RunConfig(
            device='cpu', model=dense_config.ModelSettings(**{k: getattr(TINY, k) for k in (
                'blocks', 'channels', 'pool_every', 'line_length', 'value_hidden', 'head_channels')}),
            actor=dense_config.ActorSettings(leaf_batch=64, cache_positions=256),
            evaluation=dense_config.EvaluationSettings(**{**settings, **evaluation}))
        if not (self.run/'config.json').exists():
            dense_config.save(self.run, config)
        return dense_eval.Evaluator(self.run, config, config.evaluation, dense_eval.Pacer(1.))

    def export(self, *steps):
        for step in steps:
            path = self.run/'checkpoints'/'main'/f'{step:06d}'
            path.mkdir(parents=True)
            hexnet.save_model(path/'ema.pt', hexnet.HexNet(TINY))
            (path/'manifest.json').write_text(json.dumps(dict(variant='main', step=step, created_at=float(step))))

    def league(self):
        return json.loads((self.run/'league.json').read_text())

    def test_newest_first_skips_older_and_needs_only_the_champion(self):
        evaluator = self.start()
        self.export(10, 20, 30)
        self.assertTrue(evaluator.step())
        league = self.league()
        self.assertEqual(league['champion'], 'main/000030')
        self.assertEqual([(c['id'], c.get('skipped', False), c['elo']) for c in league['checkpoints']],
                         [('main/000010', True, None), ('main/000020', True, None), ('main/000030', False, 0.)])
        self.assertFalse(evaluator.step())
        self.export(40, 50)
        self.assertTrue(evaluator.step())
        league = self.league()
        self.assertEqual([c['id'] for c in league['checkpoints'] if c.get('skipped')], ['main/000010', 'main/000020', 'main/000040'])
        self.assertEqual([m['opponent'] for m in league['checkpoints'][-1]['matches']], ['main/000030'])
        self.assertFalse(evaluator.step())  # previous_games 0: nothing optional is played
        self.assertEqual(sorted(p.parent.name for p in (self.run/'evaluations').glob('*/report.json')),
                         ['main-000050-vs-main-000030'])
        status = json.loads((self.run/'evaluator-status.json').read_text())
        self.assertEqual(status['comparison'], dict(candidate='main/000050', opponent='main/000030', kind='champion'))
        self.assertEqual((status['stage'], status['games_played'], status['games_planned']), ('playing', 2, 2))
        self.assertEqual(status['backlog'], ['main/000040', 'main/000050'])
        self.assertGreater(status['placements_per_second'], 0)
        self.assertTrue(0 < status['eval_share_used'] <= 1)
        self.assertGreater(status['updated_at'], 0)
        self.assertLessEqual(status['started_at'], status['updated_at'])
        self.assertEqual(status['settings'], asdict(evaluator.settings))
        games = json.loads(dense_eval.report_path(self.run, 'main/000050', 'main/000030').read_text())['games']
        placed = sum(g['plies']-len(g['opening']) for g in games)
        self.assertEqual((status['placements_played'], status['mean_placements']), (placed, placed/2))
        events = [json.loads(line) for line in (self.run/'events.jsonl').read_text().splitlines()]
        match = next(e for e in events if e['kind'] == 'match')
        self.assertEqual((match['placements'], match['worker_seconds']), (placed, match['seconds']))
        self.assertEqual([e['checkpoints'] for e in events if e['kind'] == 'skip'], [['main/000010', 'main/000020'], ['main/000040']])

    def test_throttled_status_names_the_waiting_pairing(self):
        """While pacing holds back the first games, the status shows the pairing about to play with none running."""
        evaluator = self.start(sprt_max_games=4)
        self.export(10)
        self.assertTrue(evaluator.step())
        self.export(20)
        ready, seen = iter([False]), []
        evaluator.pacer.ready = lambda: next(ready, True)
        def wait(tick):
            tick()
            seen.append(json.loads((self.run/'evaluator-status.json').read_text()))
        evaluator.pacer.wait = wait
        self.assertTrue(evaluator.step())
        self.assertEqual([(s['stage'], s['comparison']['candidate'], s['pool'][0]['running'], s['pool'][0]['share'], s['eval_share'])
                          for s in seen], [('throttled', 'main/000020', 0, 2, 1.)])
        self.assertEqual(len(evaluator.games('main/000020', 'main/000010')), 4)

    def test_previous_comparison_plays_while_idle(self):
        self.export(10, 30, 50)
        match = lambda opponent: dict(opponent=opponent, wins=1, losses=1, capped=0, games=2)
        # An older league: entries without `skipped`, main/000010 still champion.
        old = dict(champion='main/000010', checkpoints=[
            dict(id='main/000010', variant='main', step=10, elo=0., matches=[]),
            dict(id='main/000030', variant='main', step=30, elo=None, matches=[match('main/000010')]),
            dict(id='main/000050', variant='main', step=50, elo=None, matches=[match('main/000010')])])
        (self.run/'league.json').write_text(json.dumps(old))
        evaluator = self.start(previous_games=4)
        self.assertIn('matrix', self.league())                  # rewritten once on start with the payoff matrix
        self.assertEqual(evaluator.optional()[1:], ('main/000030', 'previous', 4))
        self.assertTrue(evaluator.step())
        league = self.league()
        self.assertEqual([m['opponent'] for m in league['checkpoints'][-1]['matches']], ['main/000010', 'main/000030'])
        self.assertEqual(league['checkpoints'][-1]['matches'][-1]['games'], 4)
        self.assertIsNone(evaluator.optional())  # main/000030's previous is its champion comparison

    def test_league_without_skipped_rates_before_anchoring(self):
        self.export(10, 30)
        (self.run/'league.json').write_text(json.dumps(dict(champion='main/000010', checkpoints=[
            dict(id='main/000010', variant='main', step=10, elo=0., matches=[]),
            dict(id='main/000020', variant='main', step=20, skipped=True, elo=None, matches=[])])))
        evaluator = self.start(anchor_every=1, anchor_games=2, seal_ms=5)
        self.assertEqual(evaluator.anchor()[1:], ('seal', 'anchor', 2))
        self.assertTrue(evaluator.step())
        league = self.league()
        self.assertEqual([(c['id'], c['elo'] is None) for c in league['checkpoints']],
                         [('main/000010', False), ('main/000020', True), ('main/000030', False)])
        self.assertEqual(league['checkpoints'][0]['elo'], 0.)
        self.assertEqual(league['anchors']['seal'], dict(elo=None, elo_interval=None, games=0, matches=[], latest_delta=None))
        # main/000030 was not promoted: no anchor of its own; the champion owes 2 more per rated checkpoint.
        entry, opponent, kind, games = evaluator.anchor()
        self.assertEqual((entry['id'], opponent, kind, games), ('main/000010', 'seal', 'anchor', 4))
        evaluator = self.start(anchor_every=1, anchor_games=2, anchor_on_promotion=False)
        self.assertEqual(evaluator.anchor()[1:], ('seal', 'anchor', 2))

    def anchored(self):
        """An evaluator (anchor_games 4, idle rematches) that rated main/000010 (champion) before main/000020 was
        exported."""
        evaluator = self.start(anchor_games=4, seal_ms=5, idle_rematch=True)
        self.export(10)
        self.assertTrue(evaluator.step())
        self.assertEqual(evaluator.anchor()[1:], ('seal', 'anchor', 4))
        self.export(20)
        return evaluator

    def test_promotion_anchors_after_the_next_sprt_and_before_optional_work(self):
        import dashboard
        evaluator = self.anchored()
        self.assertTrue(evaluator.step())                                   # the waiting candidate's SPRT first
        self.assertEqual(self.league()['checkpoints'][-1]['matches'][0]['opponent'], 'main/000010')
        path = dense_eval.report_path(self.run, 'main/000010', 'seal')
        self.assertFalse(path.exists())
        self.assertEqual(evaluator.optional()[1:], ('main/000010', 'sprt', 2))
        self.assertTrue(evaluator.step())                                   # all 4 anchor games in one session
        self.assertEqual(len(json.loads(path.read_text())['games']), 4)
        status = json.loads((self.run/'evaluator-status.json').read_text())
        self.assertEqual(status['comparison'], dict(candidate='main/000010', opponent='seal', kind='anchor'))
        self.assertIsNone(evaluator.anchor())
        league = self.league()
        seal, report = league['anchors']['seal'], json.loads(path.read_text())['summary']
        self.assertEqual(seal['matches'], [dict(checkpoint='main/000010', **{k: report[k] for k in
                                                ('wins', 'losses', 'capped', 'games', 'elo_delta')})])
        self.assertEqual((seal['games'], seal['latest_delta']), (4, report['elo_delta']))
        self.assertIsNotNone(seal['elo'])
        self.assertEqual(league['checkpoints'][0]['matches'][-1]['opponent'], 'seal')
        self.assertTrue(evaluator.step())                                   # then the optional rematch
        self.assertEqual(len(json.loads(dense_eval.report_path(self.run, 'main/000020', 'main/000010').read_text())['games']), 4)
        for x, at in (('step', 10), ('hours', 10/3600)):
            self.assertEqual(dashboard.series(self.run, dict(created_at=0.), 'main', 'seal_delta', x)['points'],
                             [[at, report['elo_delta']]])
        self.assertEqual(dashboard.series(self.run, dict(created_at=0.), 'side', 'seal_delta')['points'], [])

    def test_late_promotion_and_restoration_owe_anchors_from_their_reign(self):
        self.vetoed_on_disk()
        with unittest.mock.patch.object(dense_eval, 'write_league'):
            evaluator = self.start(anchor_every=1, anchor_games=2, extra_opponents=1)
            evaluator.settle()
        self.assertEqual((evaluator.league['champion'], evaluator.league['reign_from']), ('main/000010', 3))
        self.assertEqual(evaluator.anchor()[1:], ('seal', 'anchor', 2))
        # Restored with 200 Seal games from an earlier reign: a fresh anchor per anchor_every rated checkpoints.
        report = dense_eval.report_path(self.run, 'main/000010', 'seal')
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps(dict(candidate='main/000010', opponent='seal', settings=asdict(evaluator.settings),
                                          games=[{}]*200)))
        evaluator.crown('main/000010')
        self.assertEqual((evaluator.league['reign_games'], evaluator.anchor()[3]), (200, 2))
        evaluator.league['checkpoints'].append(dict(id='main/000050', variant='main', step=50, elo=0., matches=[]))
        self.assertEqual(evaluator.anchor()[3], 4)
        report.write_text(json.dumps(dict(json.loads(report.read_text()), games=[{}]*204)))
        self.assertIsNone(evaluator.anchor())
        report.unlink()
        self.export(30)
        evaluator.league = dict(champion='main/000010', checkpoints=[
            dict(id=f'main/{k:06d}', variant='main', step=k, elo=0., matches=[]) for k in (10, 20, 30, 40)])
        self.assertEqual(evaluator.anchor()[3], 8)                  # legacy league: from the champion's entry on
        evaluator.promote('main/000020', 'main/000010')             # e.g. an idle SPRT rematch reaching H1
        self.assertEqual((evaluator.league['reign_from'], evaluator.anchor()[3]), (4, 2))
        evaluator.league['checkpoints'].append(dict(id='main/000050', variant='main', step=50, elo=0., matches=[]))
        self.assertEqual(evaluator.anchor()[3], 4)

    def test_a_restart_loses_only_the_games_in_flight(self):
        """Every completed colour pair is on disk at once: an evaluator killed mid-session resumes the pairing."""
        evaluator = self.anchored()
        self.assertTrue(evaluator.step())                                   # main/000020's SPRT
        def crash(pool, steps):
            if steps == 3:                                                  # pair 0 is persisted, pair 1 is in flight
                raise Crash
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(hook=crash)), self.assertRaises(Crash):
            evaluator.step()
        report = json.loads(dense_eval.report_path(self.run, 'main/000010', 'seal').read_text())
        self.assertEqual([g['pair'] for g in report['games']], [0, 0])
        evaluator = self.start(anchor_games=4, seal_ms=5, idle_rematch=True)
        self.assertEqual(evaluator.anchor()[1:], ('seal', 'anchor', 2))
        self.assertTrue(evaluator.step())
        report = json.loads(dense_eval.report_path(self.run, 'main/000010', 'seal').read_text())
        self.assertEqual([g['pair'] for g in report['games']], [0, 0, 1, 1])
        self.assertIsNone(evaluator.anchor())

    def test_a_restart_resumes_the_candidate_it_was_rating(self):
        evaluator = self.start(sprt_max_games=6)
        self.export(10)
        self.assertTrue(evaluator.step())
        self.export(20)
        def crash(pool, steps):
            if steps == 3:
                raise Crash
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(winner=lambda r: -1, hook=crash)), self.assertRaises(Crash):
            evaluator.step()
        self.assertEqual(len(evaluator.games('main/000020', 'main/000010')), 2)
        self.export(30)                                                     # a newer export while it was down
        evaluator = self.start(sprt_max_games=6)
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(winner=lambda r: -1)):
            self.assertTrue(evaluator.step())                               # settles main/000020 on its 2 games
        entry = evaluator.entry('main/000020')
        report = json.loads(dense_eval.report_path(self.run, 'main/000020', 'main/000010').read_text())
        self.assertEqual((entry.get('skipped'), entry.get('superseded'), len(report['games'])), (None, True, 2))
        self.assertEqual(report['metrics']['sprt']['decision'], 'superseded')

    def test_newer_champion_supersedes_an_unfinished_anchor(self):
        evaluator = self.anchored()
        evaluator.step()
        def export(pool, steps):
            if steps == 2 and not (self.run/'checkpoints'/'main'/'000030').exists():
                self.export(30)                                             # stops the anchor after its first pair
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(hook=export)):
            evaluator.step()                                                # 2 of main/000010's 4 anchor games
        test = evaluator.test
        evaluator.test = lambda records: dict(test(records), decision='H1')
        self.assertTrue(evaluator.step())
        self.assertEqual(self.league()['champion'], 'main/000030')
        entry, opponent, kind, games = evaluator.anchor()
        self.assertEqual((entry['id'], opponent, kind, games), ('main/000030', 'seal', 'anchor', 4))
        self.assertTrue(evaluator.step())
        seal = self.league()['anchors']['seal']
        self.assertEqual([(m['checkpoint'], m['games']) for m in seal['matches']], [('main/000010', 2), ('main/000030', 4)])
        self.assertEqual(seal['latest_delta'], seal['matches'][-1]['elo_delta'])
        self.assertEqual(len(json.loads(dense_eval.report_path(self.run, 'main/000010', 'seal').read_text())['games']), 2)

    def test_newer_checkpoint_before_the_first_game_skips_the_candidate(self):
        evaluator = self.start(sprt_max_games=8)
        self.export(10)
        evaluator.step()
        self.export(30)
        newer = evaluator.newer
        def export_then_check(cid):
            if not (self.run/'checkpoints'/'main'/'000040').exists():
                self.export(40)
            return newer(cid)
        evaluator.newer = export_then_check
        self.assertTrue(evaluator.step())
        self.assertFalse(dense_eval.report_path(self.run, 'main/000030', 'main/000010').exists())
        entry = self.league()['checkpoints'][-1]
        self.assertEqual((entry['id'], entry.get('skipped'), entry['matches']), ('main/000030', True, []))
        self.assertTrue(evaluator.step())
        self.assertTrue(dense_eval.report_path(self.run, 'main/000040', 'main/000010').exists())

    def test_supersession_drains_the_pool_and_counts_its_games(self):
        """A newer export stops new games; the games in flight finish and count, none is discarded."""
        evaluator = self.start(sprt_max_games=40, pool_games=8)
        self.export(10)
        evaluator.step()
        self.export(30)
        def export(pool, steps):
            if steps == 3 and not (self.run/'checkpoints'/'main'/'000040').exists():
                self.export(40)
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(winner=lambda r: -1, hook=export)):
            self.assertTrue(evaluator.step())
        report = json.loads(dense_eval.report_path(self.run, 'main/000030', 'main/000010').read_text())
        started = evaluator.next['main/000030', 'main/000010']
        self.assertEqual((report['metrics']['sprt']['decision'], len(report['games'])), ('superseded', 2*started))
        self.assertEqual(sorted({g['pair'] for g in report['games']}), list(range(started)))
        self.assertGreater(started, 4)                                     # refilled pairs, then drained ones
        entry = self.league()['checkpoints'][-1]
        self.assertEqual((entry['id'], entry.get('superseded'), self.league()['champion']), ('main/000030', True, 'main/000010'))
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(winner=lambda r: -1)):
            self.assertTrue(evaluator.step())
        report = json.loads(dense_eval.report_path(self.run, 'main/000040', 'main/000010').read_text())
        self.assertEqual((report['metrics']['sprt']['decision'], len(report['games'])), ('max-games', 40))

    def test_the_pool_refills_as_games_finish(self):
        """Games stream one at a time: a finished game's slot is refilled before the next step, and the status
        tally and games played follow every finished game."""
        evaluator = self.start(sprt_max_games=20, pool_games=6, sprt_alpha=1e-9, sprt_beta=1e-9)
        self.export(10)
        evaluator.step()
        self.export(20)
        seen = []
        def watch(pool, steps):
            if steps > 1:
                status = json.loads((self.run/'evaluator-status.json').read_text())
                seen.append((pool.running(), status['games_played'], status['tally']['wins'], status['pool'][0]['running']))
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(hook=watch)), \
                unittest.mock.patch.object(dense_eval, 'STATUS_SECONDS', 0.):
            self.assertTrue(evaluator.step())
        self.assertEqual([s[1] for s in seen], list(range(1, 20)))            # one more finished game per step
        self.assertEqual([s[2] for s in seen], list(range(1, 20)))
        self.assertTrue(all(s[0] >= 5 for s in seen[:13]))                 # kept full until the budget runs out
        self.assertEqual(len(evaluator.games('main/000020', 'main/000010')), 20)

    def test_superseded_sprt_settles_on_its_games(self):
        """A newer checkpoint stops the SPRT; the settlement promotes because the paired 95% lower bound of its pair
        score exceeds 1/2."""
        evaluator = self.start(sprt_max_games=64, pool_games=16, sprt_alpha=1e-9, sprt_beta=1e-9)
        self.export(10)
        evaluator.step()
        self.export(30)
        def export(pool, steps):
            if steps == 16 and not (self.run/'checkpoints'/'main'/'000040').exists():
                self.export(40)
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(hook=export)):
            self.assertTrue(evaluator.step())
        report = json.loads(dense_eval.report_path(self.run, 'main/000030', 'main/000010').read_text())
        test, n = report['metrics']['sprt'], len(report['games'])
        self.assertEqual((test['decision'], test['settled']['promote'], n % 2), ('superseded', True, 0))
        self.assertGreater(n, 16)
        league = self.league()
        self.assertEqual((league['champion'], league['checkpoints'][-1].get('superseded')), ('main/000030', True))
        events = [json.loads(line) for line in (self.run/'events.jsonl').read_text().splitlines()]
        settle = next(e for e in events if e['kind'] == 'settle')
        self.assertIn(f'settled on supersession after {n} games', settle['message'])
        self.assertEqual((settle['promote'], settle['games']), (True, n))

    def test_posterior_decision_promotes_a_clear_winner(self):
        """Posterior mode: rounds go to the direct pairing (no other evidence), the verdict promotes once the
        candidate leads with P(delta > 0) >= promote_confidence; status, report and event carry the verdict."""
        evaluator = self.start(decision='posterior', sprt_max_games=12, promote_confidence=.9)
        self.export(10)
        evaluator.step()
        self.export(20)
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted()):
            self.assertTrue(evaluator.step())
        league = self.league()
        self.assertEqual(league['champion'], 'main/000020')
        report = json.loads(dense_eval.report_path(self.run, 'main/000020', 'main/000010').read_text())
        verdict = report['metrics']['posterior']
        self.assertEqual((verdict['decision'], verdict['leader'], verdict['direct']['games']), ('promote', 'main/000020', len(report['games'])))
        self.assertGreaterEqual(verdict['p_better'], .9)
        self.assertLessEqual(len(report['games']), 12)
        status = json.loads((self.run/'evaluator-status.json').read_text())
        self.assertEqual(status['decision']['decision'], 'promote')
        events = [json.loads(line) for line in (self.run/'events.jsonl').read_text().splitlines()]
        decision = next(e for e in events if e['kind'] == 'decision')
        self.assertIn('main/000020 vs main/000010: promote after', decision['message'])
        self.assertEqual(dense_eval.rematch_pair(self.run, 'main/000020', 'main/000010', evaluator.settings),
                         ('main/000010', 'main/000020'))                        # the decided report never grows

    def test_loop_logs_the_promotion_rule_of_a_legacy_config(self):
        self.start()
        config = json.loads((self.run/'config.json').read_text())
        del config['evaluation']['decision']
        (self.run/'config.json').write_text(json.dumps(config))
        flags = {f'eval_{f.name}': None for f in dataclasses.fields(dense_config.EvaluationSettings)}
        dense_eval.loop(argparse.Namespace(run=str(self.run), once=True, poll=0., processes=1, **flags))
        events = [json.loads(line) for line in (self.run/'events.jsonl').read_text().splitlines()]
        self.assertEqual([e['message'] for e in events if e['kind'] == 'info'],
                         ['promotion rule: posterior (default; config.json predates the setting)'])

    def test_posterior_direct_games_stop_at_sprt_max_games(self):
        evaluator = self.start(decision='posterior', sprt_max_games=10, pool_games=8)
        self.export(10)
        evaluator.step()
        self.export(20)
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(winner=lambda r: -1)):
            self.assertTrue(evaluator.step())
        report = json.loads(dense_eval.report_path(self.run, 'main/000020', 'main/000010').read_text())
        self.assertEqual((len(report['games']), report['metrics']['posterior']['decision']), (10, 'max-games'))

    def test_posterior_decision_rejects_a_clear_loser(self):
        evaluator = self.start(decision='posterior', sprt_max_games=12, promote_confidence=.9)
        self.export(10)
        evaluator.step()
        self.export(20)
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(winner=lambda r: 1-r['challenger_color'])):
            self.assertTrue(evaluator.step())
        self.assertEqual(self.league()['champion'], 'main/000010')
        report = json.loads(dense_eval.report_path(self.run, 'main/000020', 'main/000010').read_text())
        self.assertEqual(report['metrics']['posterior']['decision'], 'reject')

    def test_variant_ids_registration_and_cli(self):
        self.assertEqual(dense_eval.split_id('main/032500@solver'), ('main/032500', 'solver'))
        self.assertEqual(dense_eval.split_id('main/032500'), ('main/032500', None))
        self.assertEqual(dense_eval.parse_settings(['sims=20', 'tactics=false', 'solver-root-nodes=135']),
                         dict(sims=20, tactics=False, solver_root_nodes=135))
        for bad in (['max_plies=10'], ['sims'], [], ['tactics=maybe'], ['sims=0']):
            with self.assertRaises(ValueError):
                dense_eval.parse_settings(bad)
        evaluator = self.start(decision='posterior')
        self.export(10, 20)
        evaluator.step()                                                    # main/000020 champion, main/000010 skipped
        for checkpoint, name, settings in (('main/000010', 'x', dict(sims=1)), ('main/000020', 'a@b', dict(sims=1)),
                                           ('main/000020', 'x', dict(solver_finalists=2))):
            with self.assertRaises(ValueError):
                dense_eval.register(self.run, checkpoint, name, settings)
        argv = ['dense_eval.py', 'variant', '--run', str(self.run), '--checkpoint', 'main/000020', '--name', 'solver',
                '--set', 'solver_root_nodes=135', '--set', 'solver_finalists=2', '--set', 'solver_finalist_nodes=135',
                '--set', 'solver_threat_nodes=135']
        with unittest.mock.patch.object(sys, 'argv', argv), unittest.mock.patch('builtins.print'):
            dense_eval.main()
        settings = dict(solver_root_nodes=135, solver_finalists=2, solver_finalist_nodes=135, solver_threat_nodes=135)
        self.assertEqual(self.league()['variants'], [])                     # only the evaluator writes league.json
        entry = json.loads(dense_eval.requests(self.run)['main/000020@solver'].read_text())
        self.assertEqual({k: entry[k] for k in ('id', 'checkpoint', 'name', 'settings', 'elo', 'matches')},
                         dict(id='main/000020@solver', checkpoint='main/000020', name='solver', settings=settings, elo=None, matches=[]))
        self.assertEqual(dense_eval.register(self.run, 'main/000020', 'solver', settings), entry)
        with self.assertRaises(ValueError):
            dense_eval.register(self.run, 'main/000020', 'solver', dict(sims=20))
        dense_eval.write_league(self.run, evaluator.league, evaluator.config)   # the evaluator's next write adopts it
        self.assertEqual([v['id'] for v in self.league()['variants']], ['main/000020@solver'])
        self.assertEqual(dense_eval.requests(self.run), {})
        self.assertEqual(dense_eval.register(self.run, 'main/000020', 'solver', settings)['id'], 'main/000020@solver')
        self.assertEqual(dense_eval.requests(self.run), {})
        with self.assertRaises(ValueError):
            dense_eval.register(self.run, 'main/000020', 'solver', dict(sims=20))
        self.assertEqual(evaluator.side('main/000020@solver').solver_threat_nodes, 135)
        events = [json.loads(line) for line in (self.run/'events.jsonl').read_text().splitlines()]
        self.assertEqual([e['candidate'] for e in events if e['kind'] == 'variant'], ['main/000020@solver'])

    def test_pool_games_give_each_side_its_settings(self):
        evaluator = self.start(decision='posterior', sims=4, root_samples=4)
        self.export(10)
        evaluator.step()
        dense_eval.register(self.run, 'main/000010', 'fast', dict(sims=2, solver_root_nodes=16))
        dense_eval.adopt(evaluator.league, self.run)
        a, b = 'main/000010@fast', 'main/000010'
        evaluator.open(a, b)
        evaluator.use(a, b)
        self.assertIsNot(evaluator.models[a], evaluator.models[b])
        started = []
        evaluator.start(SimpleNamespace(add=lambda lane, games: started.extend(games)), (a, b, 'variant'))
        self.assertEqual(sorted(g.record['challenger_color'] for g in started), [0, 1])
        for game in started:
            mine = game.record['challenger_color']
            self.assertEqual((game.budgets[mine], game.budgets[1-mine]), (2, 4))
            self.assertEqual((game.sample_counts[mine], game.sample_counts[1-mine]), (2, 4))   # root_samples at most sims
            self.assertEqual((game.solvers[mine].root_nodes, game.solvers[1-mine].active), (16, False))
            self.assertEqual(game.budget, game.budgets[game.game.player])
            game.finish()
        solver = evaluator.solver_status(None, [a, b])                       # the baseline's budgets are all 0
        self.assertEqual((solver['root_nodes'], solver['sides'][a]['root_nodes'], list(solver['sides'])), (0, 16, [a]))
        self.assertIsNone(evaluator.solver_status(None, [b, 'seal']))
        for name in ('ab@test', 'ab/1'):
            with self.assertRaises(ValueError):
                dense_config.LearnerSettings(variant=name)

    def test_variant_decisions_stop_on_p_and_never_promote(self):
        """A variant is decided against its checkpoint by P(better) alone, well before sprt_max_games; it is rated
        in the league but never becomes champion, leads a checkpoint verdict, enters the ladder or the actor pointer."""
        evaluator = self.start(decision='posterior', sprt_max_games=40, sprt_min_games=4, pool_games=4, promote_confidence=.9)
        self.export(10)
        evaluator.step()
        strong, base = 'main/000010@strong', 'main/000010'
        dense_eval.register(self.run, base, 'strong', dict(sims=1))
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted()):
            self.assertTrue(evaluator.step())
        league = self.league()
        entry = league['variants'][0]
        verdict = entry['verdict']
        self.assertEqual({k: verdict[k] for k in ('decision', 'comparison', 'rule', 'threshold', 'candidate', 'opponent')},
                         dict(decision='better', comparison='variant', rule='posterior', threshold=.9, candidate=strong, opponent=base))
        self.assertGreaterEqual(verdict['p_better'], .9)
        self.assertLess(verdict['direct']['games'], 40)
        self.assertEqual(verdict['direct']['wins'], verdict['direct']['games'])
        self.assertEqual([(m['opponent'], m['wins'], m['games']) for m in entry['matches']],
                         [(base, verdict['direct']['games'], verdict['direct']['games'])])
        self.assertGreater(entry['elo'], 0)
        self.assertLess(entry['elo_interval'][0], entry['elo'])
        report = json.loads(dense_eval.report_path(self.run, strong, base).read_text())
        self.assertEqual((report['overrides'], report['metrics']['posterior']['decision']), ({strong: dict(sims=1)}, 'better'))
        status = json.loads((self.run/'evaluator-status.json').read_text())
        self.assertEqual((status['decision']['decision'], status['decision']['rule'], status['pending']), ('better', 'posterior', []))
        events = [json.loads(line) for line in (self.run/'events.jsonl').read_text().splitlines()]
        decision = next(e for e in events if e['kind'] == 'decision')
        self.assertIn(f'{strong} vs {base}: better after', decision['message'])
        self.assertEqual((decision['rule'], decision['threshold'], len(decision['interval'])), ('posterior', .9, 2))
        self.assertEqual(league['champion'], base)
        self.assertNotIn(strong, {d[k] for d in league['ladder']+league['differences'] for k in ('a', 'b')})
        self.assertEqual(json.loads((self.run/'actor.json').read_text())['checkpoint'], base)
        self.assertFalse(evaluator.step())                                   # nothing pending
        weak = 'main/000010@weak'
        dense_eval.register(self.run, base, 'weak', dict(sims=1, tactics=False))
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(winner=lambda r: 1-r['challenger_color'])):
            self.assertTrue(evaluator.step())
        self.assertEqual(self.league()['variants'][1]['verdict']['decision'], 'worse')
        self.export(20)
        self.assertEqual(evaluator.verdict('main/000020', base)['leader'], base)  # the variant rated above it never leads

    def test_a_variant_trial_yields_to_a_waiting_checkpoint(self):
        evaluator = self.start(decision='posterior', sprt_max_games=40, sprt_min_games=4, pool_games=2,
                               promote_confidence=.999999)
        self.export(10)
        evaluator.step()
        variant, base = 'main/000010@x', 'main/000010'
        dense_eval.register(self.run, base, 'x', dict(sims=1))
        hook = lambda pool, steps: self.export(20) if steps == 4 else None
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(hook=hook, winner=lambda r: -1)):
            self.assertTrue(evaluator.step())
        played = len(evaluator.games(variant, base))
        self.assertTrue(0 < played < 40)
        self.assertNotIn('verdict', self.league()['variants'][0])
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(winner=lambda r: -1)):
            self.assertTrue(evaluator.step())                                # the checkpoint is rated first
            pending = json.loads((self.run/'evaluator-status.json').read_text())['pending']
            self.assertEqual([(p['candidate'], p['kind'], p['direct']['games']) for p in pending], [(variant, 'variant', played)])
            self.assertIsNotNone(pending[0]['p_better'])
            self.assertIn('main/000020', [c['id'] for c in self.league()['checkpoints']])
            self.assertTrue(evaluator.step())                                # then the trial resumes
        self.assertEqual(self.league()['variants'][0]['verdict']['decision'], 'max-games')
        self.assertEqual(len(evaluator.games(variant, base)), 40)

    def crown(self, evaluator, step):
        """Make exported checkpoint main/<step> a rated league entry and the champion, as a promotion would."""
        evaluator.league['checkpoints'].append(dict(id=f'main/{step:06d}', variant='main', step=step, elo=0., matches=[]))
        evaluator.crown(f'main/{step:06d}')

    def test_a_champion_variant_binds_when_its_comparison_starts(self):
        evaluator = self.start(decision='posterior', sprt_max_games=4, sprt_min_games=4, pool_games=2)
        with self.assertRaises(ValueError):
            dense_eval.register(self.run, 'champion', 'solver', dict(sims=1))  # no champion yet
        self.export(10, 20, 30)
        evaluator.step()                                                    # main/000030 champion
        entry = dense_eval.register(self.run, 'champion', 'solver', dict(sims=1))
        self.assertEqual((entry['id'], entry['checkpoint'], entry['base']), ('champion@solver', None, 'champion'))
        self.assertEqual(dense_eval.adopt(evaluator.league, self.run), 1)
        self.assertTrue(evaluator.bind())                                    # follows the champion until it starts
        self.assertEqual(evaluator.variants()[0]['id'], 'main/000030@solver')
        self.crown(evaluator, 20)
        self.assertTrue(evaluator.bind())
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(winner=lambda r: -1)):
            self.assertTrue(evaluator.step())
        entry = self.league()['variants'][0]
        self.assertEqual((entry['id'], entry['checkpoint'], entry['registered_as']), ('main/000020@solver', 'main/000020', 'champion@solver'))
        self.assertIn('bound_at', entry)
        self.assertEqual(entry['matches'][0]['opponent'], 'main/000020')
        self.assertEqual(dense_eval.requests(self.run), {})
        self.crown(evaluator, 30)
        self.assertFalse(evaluator.bind())                                   # started: the binding stays
        self.assertEqual(evaluator.variants()[0]['id'], 'main/000020@solver')
        self.assertEqual(dense_eval.register(self.run, 'champion', 'solver', dict(sims=1))['id'], 'main/000020@solver')
        events = [json.loads(line) for line in (self.run/'events.jsonl').read_text().splitlines()]
        self.assertEqual([e['candidate'] for e in events if e['kind'] == 'variant'],
                         ['champion@solver', 'main/000030@solver', 'main/000020@solver'])

    def test_a_review_promotion_rebinds_a_variant_before_it_starts(self):
        evaluator = self.start(decision='posterior', sprt_max_games=4, sprt_min_games=4, pool_games=2)
        self.export(10, 20, 30)
        evaluator.step()                                                    # main/000030 champion
        dense_eval.register(self.run, 'champion', 'solver', dict(sims=1))
        evaluator.reviewed = False
        evaluator.review = lambda: self.crown(evaluator, 20)                 # the startup review promotes main/000020
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(winner=lambda r: -1)):
            self.assertTrue(evaluator.step())
        entry = self.league()['variants'][0]
        self.assertEqual((entry['id'], entry['checkpoint'], entry['matches'][0]['opponent']),
                         ('main/000020@solver', 'main/000020', 'main/000020'))
        status = json.loads((self.run/'evaluator-status.json').read_text())
        self.assertEqual((status['decision']['candidate'], status['pending']), ('main/000020@solver', []))
        seen = []
        evaluator.publish = lambda force=False, **fields: seen.append(list(evaluator.status['pending']))
        evaluator.league['variants'].append(dict(id='champion@fast', checkpoint=None, name='fast', settings=dict(sims=1),
                                                 base='champion', registered_as='champion@fast', matches=[]))
        evaluator.bind()
        evaluator.queue([])
        self.assertEqual([p['candidate'] for p in evaluator.status['pending']], ['main/000020@fast'])
        self.crown(evaluator, 10)
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(winner=lambda r: -1)):
            evaluator.trial(evaluator.variants()[-1])
        self.assertEqual({p['candidate'] for p in seen[0]}, {'main/000010@fast'})     # rebuilt after the rebinding

    def test_a_trial_without_games_leaves_the_binding_open(self):
        evaluator = self.start(decision='posterior')
        self.export(10, 20)
        evaluator.step()                                                    # main/000020 champion
        dense_eval.register(self.run, 'champion', 'solver', dict(sims=1))
        dense_eval.adopt(evaluator.league, self.run)
        evaluator.bind()
        self.export(30)                                                     # a checkpoint waits: no game starts
        evaluator.trial(evaluator.variants()[0])
        self.assertNotIn('bound_at', evaluator.variants()[0])
        self.crown(evaluator, 30)
        self.assertTrue(evaluator.bind())
        self.assertEqual(evaluator.variants()[0]['id'], 'main/000030@solver')

    def test_a_champion_variant_never_collides_with_a_registered_id(self):
        evaluator = self.start(decision='posterior')
        self.export(10, 20, 30)
        evaluator.step()                                                    # main/000030 champion
        dense_eval.register(self.run, 'main/000030', 'solver', dict(sims=1))
        with self.assertRaises(ValueError):
            dense_eval.register(self.run, 'champion', 'solver', dict(sims=1))
        dense_eval.register(self.run, 'champion', 'fast', dict(sims=1))
        dense_eval.register(self.run, 'main/000030', 'y', dict(sims=2))
        dense_eval.adopt(evaluator.league, self.run)
        self.assertTrue(evaluator.bind())                                   # champion@fast -> main/000030@fast
        evaluator.variants()[0]['bound_at'] = 0.                            # main/000030@solver has started
        evaluator.variants()[2].update(id='main/000020@fast', checkpoint='main/000020', name='fast', bound_at=0.)
        self.crown(evaluator, 20)                                           # main/000020@fast is taken: dropped
        self.assertTrue(evaluator.bind())
        self.assertEqual([v['id'] for v in evaluator.variants()], ['main/000030@solver', 'main/000020@fast'])
        events = [json.loads(line) for line in (self.run/'events.jsonl').read_text().splitlines()]
        self.assertIn('main/000030@fast dropped', events[-1]['message'])
        self.assertNotIn('champion@fast', dense_eval.requests(self.run))    # the drop is never re-adopted
        dense_eval.write_league(self.run, evaluator.league, evaluator.config)
        self.assertEqual([v['id'] for v in self.league()['variants']], ['main/000030@solver', 'main/000020@fast'])

    def test_a_pending_variant_of_the_champion_rebases_on_promotion(self):
        evaluator = self.start(decision='posterior')
        self.export(10, 20)
        evaluator.step()                                                    # main/000020 champion
        dense_eval.register(self.run, 'main/000020', 'x', dict(sims=1))
        dense_eval.adopt(evaluator.league, self.run)
        self.assertFalse(evaluator.bind())
        self.export(30)
        self.crown(evaluator, 30)
        evaluator.settings = replace(evaluator.settings, rebase_on_promotion=False)
        self.assertFalse(evaluator.bind())
        evaluator.settings = replace(evaluator.settings, rebase_on_promotion=True)
        self.assertTrue(evaluator.bind())
        self.assertEqual((evaluator.variants()[0]['id'], evaluator.variants()[0]['checkpoint']), ('main/000030@x', 'main/000030'))
        events = [json.loads(line) for line in (self.run/'events.jsonl').read_text().splitlines()]
        self.assertIn('main/000020@x now plays as main/000030@x', events[-1]['message'])
        evaluator.league['variants'].append(dict(id='main/000020@y', checkpoint='main/000020', name='y', settings=dict(sims=1),
                                                 base='main/000020', on_champion=False, matches=[]))
        self.assertFalse(evaluator.bind())                                   # registered against a non-champion: stays

    def pending(self, **evaluation):
        """A posterior evaluator whose champion main/000020 beat main/000010, with main/000030 unrated; the
        verdict of main/000030 is kept pending (promote_confidence .999999) so the lanes can be inspected."""
        evaluator = self.start(decision='posterior', promote_confidence=.999999, sprt_max_games=400, pool_games=16,
                               sprt_min_games=8, **evaluation)
        self.export(10)
        evaluator.step()
        test = evaluator.test
        evaluator.settings = replace(evaluator.settings, decision='sprt')
        evaluator.test = lambda records: dict(test(records), decision='H1')
        self.export(20)
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted()):
            evaluator.step()
        evaluator.test, evaluator.settings = test, replace(evaluator.settings, decision='posterior')
        self.assertEqual(self.league()['champion'], 'main/000020')
        self.export(30)
        return evaluator

    def test_direct_games_take_the_pool_until_sprt_min_games(self):
        evaluator = self.pending()
        evaluator.evidence = lambda *args: ('main/000030', 'main/000010')  # evidence always looks better
        lanes = lambda: evaluator.lanes(evaluator.verdict('main/000030', 'main/000020'), 'main/000030', 'main/000020')
        self.assertEqual(lanes(), {('main/000030', 'main/000020', 'champion'): 16})
        seen = []
        def watch(pool, steps):
            seen.append({k: pool.running(k) for k in {held for held, _ in pool.games.values()}})
            if steps == 40:
                raise Crash
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(winner=lambda r: -1, hook=watch)), self.assertRaises(Crash):
            evaluator.step()
        direct = [i for i, lanes in enumerate(seen) if any(k[2] == 'evidence' for k in lanes)]
        self.assertTrue(direct)
        self.assertGreaterEqual(direct[0], 8)                           # no evidence game before 8 complete direct games
        self.assertEqual(lanes(), {('main/000030', 'main/000020', 'champion'): 12, ('main/000030', 'main/000010', 'evidence'): 4})

    def test_evidence_never_exceeds_its_share(self):
        evaluator = self.pending(evidence_share=.25)
        evaluator.evidence = lambda *args: ('main/000030', 'main/000010')
        report = dense_eval.report_path(self.run, 'main/000030', 'main/000020')
        seen = []
        def watch(pool, steps):
            lanes = {held for held, _ in pool.games.values()}
            seen.append(sum(pool.running(k) for k in lanes if k[2] == 'evidence'))
            if steps == 80:
                raise Crash
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(winner=lambda r: -1, hook=watch)), self.assertRaises(Crash):
            evaluator.step()
        self.assertEqual(max(seen), 4)                                  # 1/4 of 16 games in flight
        self.assertGreater(len(json.loads(report.read_text())['games']), 8)
        evaluator.settings = replace(evaluator.settings, evidence_share=0.)
        verdict = evaluator.verdict('main/000030', 'main/000020')
        self.assertEqual(evaluator.lanes(verdict, 'main/000030', 'main/000020'), {('main/000030', 'main/000020', 'champion'): 16})

    def test_fill_order_seal_then_generalization_then_the_widest_ladder_pair(self):
        """Fill priority: the champion vs Seal while their interval is wide, then one round of the newest rated
        checkpoint vs the previous champion, then the widest ladder pair; a waiting checkpoint interrupts."""
        evaluator = self.start(idle_fill=True, anchor_target_halfwidth=1e-3, seal_ms=5, fill_top=3)
        self.export(10)
        self.assertTrue(evaluator.step())
        test = evaluator.test
        evaluator.test = lambda records: dict(test(records), decision='H1')
        self.export(20)
        self.assertTrue(evaluator.step())                          # main/000020 beats main/000010
        evaluator.test = test
        self.export(30)
        self.assertTrue(evaluator.step())                          # main/000030 meets main/000020
        self.assertEqual(self.league()['champion'], 'main/000020')
        self.assertEqual(evaluator.fill()[1:], ('seal', 'fill', 2))  # no Seal games yet: the interval is unbounded
        self.assertTrue(evaluator.step())
        self.assertEqual(len(evaluator.sealed('main/000020')['games']), 2)
        status = json.loads((self.run/'evaluator-status.json').read_text())
        self.assertEqual((status['comparison']['kind'], status['comparison']['opponent']), ('fill', 'seal'))
        evaluator.settings = replace(evaluator.settings, anchor_target_halfwidth=1e9)
        entry, opponent, kind, _ = evaluator.fill()
        self.assertEqual((entry['id'], opponent, kind), ('main/000030', 'main/000010', 'generalization'))
        self.assertTrue(evaluator.step())
        self.assertIn('main/000010', self.league()['matrix']['main/000030'])
        self.assertEqual({frozenset((d['a'], d['b'])) for d in self.league()['ladder']},
                         {frozenset(p) for p in (('main/000010', 'main/000020'), ('main/000010', 'main/000030'), ('main/000020', 'main/000030'))})
        entry, opponent, kind, _ = evaluator.fill()                # generalization is played once
        self.assertEqual(kind, 'fill')
        self.assertNotEqual(opponent, 'seal')
        growable = [d for d in self.league()['ladder'] if dense_eval.rematch_pair(self.run, d['a'], d['b'], evaluator.settings)]
        self.assertEqual({entry['id'], opponent}, {max(growable, key=lambda d: d['interval'][1]-d['interval'][0])[k] for k in 'ab'})
        self.assertTrue(evaluator.step())
        self.assertTrue(dense_eval.report_path(self.run, entry['id'], opponent).exists())
        targets = [f'{entry["id"]} vs {opponent}']
        entry, opponent, _, _ = evaluator.fill()                   # re-chosen from the updated ladder
        targets = list(dict.fromkeys(targets+[f'{entry["id"]} vs {opponent}']))
        path = dense_eval.report_path(self.run, entry['id'], opponent)
        before = len(json.loads(path.read_text())['games']) if path.exists() else 0
        def export(pool, steps):
            if steps == 1 and not (self.run/'checkpoints'/'main'/'000040').exists():
                self.export(40)
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(hook=export)):
            self.assertTrue(evaluator.step())                      # a new export drains the fill work
        self.assertEqual(len(json.loads(path.read_text())['games']), before+2)   # the pair in flight counts
        self.assertTrue(evaluator.step())
        self.assertIn('main/000040', {c['id'] for c in self.league()['checkpoints']})
        events = [json.loads(line) for line in (self.run/'events.jsonl').read_text().splitlines()]
        self.assertEqual([e['target'] for e in events if e['kind'] == 'fill'],
                         ['seal', 'generalization main/000030 vs main/000010', *targets, None])

    def report(self, a, b, results):
        """Write report a-vs-b of colour pairs whose candidate results (1 win, 0 loss, .5 cap) are `results`."""
        records = [dict(candidate=a, opponent=b, pair=k//2, seed=dense_eval.pair_seed(1740, a, k//2), opening=[[0, 0]],
                        challenger_color=k % 2, winner=k % 2 if r == 1 else 1-k % 2 if r == 0 else -1, reason='six-in-a-row',
                        plies=6, moves=[]) for k, r in enumerate(results)]
        shas = {name: dense_eval.digest(self.run/'checkpoints'/name/'ema.pt') for name in (a, b)}
        path = dense_eval.report_path(self.run, a, b)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dense_eval.make_report(a, b, records, shas, dense_config.load(self.run).evaluation)))

    def test_a_superseded_candidate_that_clearly_beats_the_champion_is_promoted(self):
        """The live case: main/025000 beat champion main/019500 +38 -11 =15 in 64 games while the champion had 300
        games of its own; main/027500 arrives before the verdict is ready (64 of sprt_min_games 128 direct games), and
        the settlement promotes on the posterior P(delta > 0), not on the Hoeffding pair-score bound (0.47 here)."""
        self.export(17000, 19500, 25000, 27500)
        self.start(decision='posterior', sprt_max_games=200, pool_games=64, sprt_min_games=64)       # writes config.json
        self.report('main/019500', 'main/017000', [1]*170+[0]*130)
        self.report('main/025000', 'main/019500', [1, 1]*19+[0, 0]*5+[0, .5]+[.5]*14)
        entry = lambda step, elo: dict(id=f'main/{step:06d}', variant='main', step=step, elo=elo, elo_interval=None, matches=[])
        (self.run/'league.json').write_text(json.dumps(dict(champion='main/019500', checkpoints=[entry(17000, 0.), entry(19500, 0.)])))
        evaluator = self.start(decision='posterior', sprt_max_games=200, pool_games=64, sprt_min_games=128)
        summary = json.loads(dense_eval.report_path(self.run, 'main/025000', 'main/019500').read_text())['summary']
        self.assertEqual((summary['wins'], summary['losses'], summary['capped']), (38, 11, 15))
        self.assertLess(summary['pair_score_lower'], .5)
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted()):
            self.assertTrue(evaluator.step())
        league = self.league()
        self.assertEqual(league['champion'], 'main/025000')
        verdict = evaluator.entry('main/025000')['verdict']
        self.assertEqual((verdict['decision'], verdict['settled'], verdict['direct']['games']), ('promote', True, 64))
        self.assertGreaterEqual(verdict['p_better'], .9)
        events = [json.loads(line) for line in (self.run/'events.jsonl').read_text().splitlines()]
        self.assertIn('main/025000 vs main/019500: promote (settled on supersession) after 64 direct games',
                      next(e for e in events if e['kind'] == 'decision')['message'])
        fresh = dense_eval.public(evaluator.verdict('main/027500', 'main/025000'))   # no direct game yet: no numbers
        self.assertEqual((fresh['direct']['games'], fresh['delta'], fresh['p_better'], fresh['pooled']), (0, None, None, None))

    def test_starting_games_is_charged_to_the_pacer(self):
        """Seal plays its first turns while a game is built, so the time spent starting games counts as playing."""
        evaluator = self.start(sprt_max_games=4)
        self.export(10)
        evaluator.step()
        self.export(20)
        now, charged, start = [evaluator.pacer.clock()], [], evaluator.start
        evaluator.pacer.clock = lambda: now[0]
        evaluator.pacer.played = lambda a, b, weight=1: charged.append(b-a)
        def slow(pool, lane):
            now[0] += 5.
            start(pool, lane)
        evaluator.start = slow
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted()):
            evaluator.step()
        self.assertEqual(sum(charged), 10.)                                 # two pairs started, 5 s each

    def test_a_long_throttle_rechecks_the_pairing(self):
        evaluator = self.start(sprt_max_games=4)
        self.export(10)
        evaluator.step()
        self.export(30)
        ready = iter([False])
        evaluator.pacer.ready = lambda: next(ready, True)
        evaluator.pacer.wait = lambda tick: self.export(40)                   # a newer export while throttled
        self.assertTrue(evaluator.step())
        self.assertEqual((evaluator.entry('main/000030')['skipped'], evaluator.games('main/000030', 'main/000010')), (True, []))

    def test_an_sprt_bound_crossed_before_draining_stays_the_decision(self):
        evaluator = self.start(sprt_max_games=8, pool_games=4)
        self.export(10)
        evaluator.step()
        test = evaluator.test
        evaluator.test = lambda records: dict(test(records), decision='H1' if len(records) == 2 else None)
        self.export(20)
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(winner=lambda r: -1)):
            self.assertTrue(evaluator.step())
        report = json.loads(dense_eval.report_path(self.run, 'main/000020', 'main/000010').read_text())
        self.assertEqual((len(report['games']), report['metrics']['sprt']['decision']), (4, 'H1'))   # the pair in flight drains
        self.assertEqual(self.league()['champion'], 'main/000020')

    def test_an_idle_sprt_rematch_keeps_the_bound_it_crossed(self):
        evaluator = self.start(sprt_max_games=2, pool_games=4, games=4, idle_rematch=True)
        for step in (10, 20):
            self.export(step)
            self.assertTrue(evaluator.step())
        self.assertEqual(evaluator.optional()[1:], ('main/000010', 'sprt', 2))
        test = evaluator.test
        evaluator.test = lambda records: dict(test(records), decision='H1' if len(records) == 4 else None)
        evaluator.settings = replace(evaluator.settings, sprt_max_games=4)
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(winner=lambda r: -1)):
            self.assertTrue(evaluator.step())
        report = json.loads(dense_eval.report_path(self.run, 'main/000020', 'main/000010').read_text())
        self.assertEqual((len(report['games']), report['metrics']['sprt']['decision']), (6, 'H1'))   # a drained pair after H1
        self.assertEqual(self.league()['champion'], 'main/000020')

    def test_a_posterior_verdict_undone_by_draining_games_plays_on(self):
        evaluator = self.start(decision='posterior', sprt_max_games=12, pool_games=4)
        self.export(10)
        evaluator.step()
        self.export(20)
        verdict, calls = evaluator.verdict, [0]
        def flicker(cid, champion):                                    # decided on exactly one look, then undecided
            calls[0] += 1
            return dict(verdict(cid, champion), decision='promote' if calls[0] == 2 else None)
        evaluator.verdict = flicker
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(winner=lambda r: -1)):
            self.assertTrue(evaluator.step())
        report = json.loads(dense_eval.report_path(self.run, 'main/000020', 'main/000010').read_text())
        self.assertEqual((len(report['games']), report['metrics']['posterior']['decision']), (12, 'max-games'))

    def test_a_candidate_with_direct_games_is_never_left_skipped(self):
        """The live case after a restart: main/027500 completed 64 games against champion main/019500 (+42 -10
        =12) but an older evaluator marked it skipped when main/030000 appeared. A fresh evaluator rates it on
        those games and promotes it."""
        self.export(17000, 19500, 27500, 30000)
        self.start()
        self.report('main/019500', 'main/017000', [1]*12+[0]*8)
        self.report('main/027500', 'main/019500', [1, 1]*21+[0, 0]*5+[.5, .5]*6)
        entry = lambda step, elo, **extra: dict(id=f'main/{step:06d}', variant='main', step=step, elo=elo, elo_interval=None,
                                                matches=[], **extra)
        (self.run/'league.json').write_text(json.dumps(dict(champion='main/019500', checkpoints=[
            entry(17000, 0.), entry(19500, 60.), entry(27500, None, skipped=True)])))
        evaluator = self.start(decision='posterior', sprt_max_games=200, pool_games=64, sprt_min_games=64)
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted()):
            self.assertTrue(evaluator.step())
        rated = evaluator.entry('main/027500')
        self.assertEqual((rated.get('skipped'), rated['verdict']['decision'], rated['verdict']['direct']['games']), (None, 'promote', 64))
        self.assertEqual(self.league()['champion'], 'main/027500')
        self.assertEqual(len(evaluator.games('main/027500', 'main/019500')), 64)      # no game was added or lost
        events = [json.loads(line) for line in (self.run/'events.jsonl').read_text().splitlines()]
        self.assertIn('main/027500 was skipped with 64 games against main/019500', next(e for e in events if e['kind'] == 'info')['message'])

    def test_the_direct_tally_is_cumulative_over_every_report_of_the_pair_and_a_restart(self):
        """main/037500's comparison with champion main/032500 is cut by a restart. The status tally, the decision's
        direct record and the pending entry count every recorded game of the pair (its report, the reverse report
        and the pool's completed pairs), before and after the restart, with pair score and LLR over all its pairs."""
        self.export(30000, 32500, 37500)
        self.start()
        self.report('main/032500', 'main/030000', [1]*12+[0]*8)
        self.report('main/037500', 'main/032500', [1, 1]*10+[0, 0]*5)          # +20 -10 before the restart
        self.report('main/032500', 'main/037500', [1, 0]*3)                    # reverse role: +3 -3 for main/037500
        entry = lambda step, elo: dict(id=f'main/{step:06d}', variant='main', step=step, elo=elo, elo_interval=None, matches=[])
        (self.run/'league.json').write_text(json.dumps(dict(champion='main/032500', checkpoints=[entry(30000, 0.), entry(32500, 60.)])))
        settings = dict(decision='posterior', promote_confidence=.999999, sprt_max_games=400, pool_games=4, sprt_min_games=400)
        status = lambda: json.loads((self.run/'evaluator-status.json').read_text())
        record = lambda d: (d['games'], d['wins'], d['losses'])
        self.assertEqual(record(self.start(**settings).verdict('main/037500', 'main/032500')['direct']), (36, 23, 13))

        def crash(after):
            def hook(pool, steps):
                if steps > after:
                    raise Crash
            return hook
        with unittest.mock.patch.object(dense_eval, 'STATUS_SECONDS', 0.), \
                unittest.mock.patch.object(dense_eval, 'Pool', scripted(hook=crash(4))), self.assertRaises(Crash):
            self.start(**settings).step()                                      # four games won, then the process dies
        played = status()
        self.assertEqual(record(played['tally']), (40, 27, 13))
        self.assertEqual(record(played['decision']['direct']), (40, 27, 13))
        self.assertEqual([record(p['direct']) for p in played['pending']], [(40, 27, 13)])
        self.assertEqual(played['games_played'], 34)                           # the lane's own report

        with unittest.mock.patch.object(dense_eval, 'STATUS_SECONDS', 0.), \
                unittest.mock.patch.object(dense_eval, 'Pool', scripted(hook=crash(0))), self.assertRaises(Crash):
            self.start(**settings).step()                                      # restarted: the count carries over
        resumed = status()
        self.assertEqual(record(resumed['tally']), (40, 27, 13))
        self.assertEqual(record(resumed['decision']['direct']), (40, 27, 13))
        self.assertEqual([record(p['direct']) for p in resumed['pending']], [(40, 27, 13)])
        self.assertEqual(resumed['tally']['pairs'], 20)
        pairs = [(1, 1)]*12+[(0, 0)]*5+[(1, 0)]*3
        score = sum(map(sum, pairs))/len(pairs)/2
        self.assertAlmostEqual(resumed['tally']['pair_score'], score)
        s = self.start(**settings).settings
        games = [dict(seed=k, challenger_color=c, winner=c if r else 1-c) for k, pair in enumerate(pairs) for c, r in enumerate(pair)]
        self.assertAlmostEqual(resumed['tally']['llr'], dense_eval.sprt(games, s.sprt_elo0, s.sprt_elo1, s.sprt_alpha, s.sprt_beta)['llr'])

    def test_a_fresh_evaluator_promotes_a_rated_leader_from_existing_reports(self):
        """On start the promotion rule is re-applied to the reports on disk: main/025000, rated but not promoted,
        leads with +38 -11 =15 in 64 direct games against the champion and is crowned without a new game."""
        self.export(17000, 19500, 25000)
        self.start()
        self.report('main/019500', 'main/017000', [1]*12+[0]*8)                     # the champion beat its predecessor
        self.report('main/025000', 'main/019500', [1, 1]*19+[0, 0]*5+[0, .5]+[.5]*14)
        entry = lambda step, elo: dict(id=f'main/{step:06d}', variant='main', step=step, elo=elo, elo_interval=None, matches=[])
        (self.run/'league.json').write_text(json.dumps(dict(champion='main/019500', checkpoints=[
            entry(17000, 0.), entry(19500, 60.), entry(25000, 200.)])))
        evaluator = self.start(decision='sprt', sprt_min_games=64)
        evaluator.review()
        self.assertEqual(evaluator.league['champion'], 'main/019500')                 # the rule applies in posterior mode
        evaluator = self.start(decision='posterior', sprt_min_games=64, sims=3)
        self.assertEqual(evaluator.games('main/025000', 'main/019500'), [])          # another protocol's games do not count
        evaluator.review()
        self.assertEqual(evaluator.league['champion'], 'main/019500')
        evaluator = self.start(decision='posterior', sprt_min_games=66, anchor_games=2)
        evaluator.review()
        self.assertEqual(evaluator.league['champion'], 'main/019500')                 # too few direct games
        evaluator = self.start(decision='posterior', sprt_min_games=64, anchor_games=2)
        verdict, ratings = evaluator.verdict, {'main/017000': 0., 'main/019500': 50., 'main/025000': 40.}
        with unittest.mock.patch.object(evaluator, 'verdict', lambda cid, champion: dict(
                verdict(cid, champion), posterior=SimpleNamespace(rating=ratings.get))):
            evaluator.review()
        self.assertEqual(evaluator.league['champion'], 'main/019500')                 # P(better) alone, not out-rating it
        evaluator.review()
        self.assertEqual((self.league()['champion'], json.loads((self.run/'champion.json').read_text())['checkpoint']),
                         ('main/025000', 'main/025000'))
        self.assertEqual(evaluator.anchor()[0]['id'], 'main/025000')                  # its Seal anchor is scheduled
        events = [json.loads(line) for line in (self.run/'events.jsonl').read_text().splitlines()]
        self.assertEqual([e['kind'] for e in events if e['kind'] in ('decision', 'promotion')], ['decision', 'promotion'])
        self.assertIn('promote on review', events[-2]['message'])
        verdict = next(c for c in self.league()['checkpoints'] if c['id'] == 'main/025000')['verdict']
        self.assertEqual((verdict['review'], verdict['opponent'], verdict['reports']['main-025000-vs-main-019500']['games']),
                         (True, 'main/019500', 64))

    def test_a_restart_under_another_protocol_starts_the_candidate_afresh(self):
        evaluator = self.start(sprt_max_games=6)
        self.export(10)
        self.assertTrue(evaluator.step())
        self.export(20)
        def crash(pool, steps):
            if steps == 3:
                raise Crash
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(winner=lambda r: -1, hook=crash)), self.assertRaises(Crash):
            evaluator.step()
        path = dense_eval.report_path(self.run, 'main/000020', 'main/000010')
        first = json.loads(path.read_text())
        evaluator = self.start(sprt_max_games=6, sims=3)                        # e.g. restarted with --eval-sims 3
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(winner=lambda r: -1)):
            self.assertTrue(evaluator.step())
        report = json.loads(path.read_text())
        self.assertEqual((report['settings']['sims'], len(report['games'])), (3, 6))
        kept = path.with_name(f"report-{int(first['created_at'])}-{first['id']}.json")
        self.assertEqual(len(json.loads(kept.read_text())['games']), 2)

    def test_an_old_protocol_report_never_rates_a_candidate_without_games(self):
        evaluator = self.start(sprt_max_games=6)
        self.export(10)
        self.assertTrue(evaluator.step())
        self.export(20)
        def crash(pool, steps):
            if steps == 3:
                raise Crash
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(winner=lambda r: -1, hook=crash)), self.assertRaises(Crash):
            evaluator.step()
        evaluator = self.start(sprt_max_games=6, sims=3)
        newer = evaluator.newer
        def export_then_check(cid):
            if not (self.run/'checkpoints'/'main'/'000030').exists():
                self.export(30)
            return newer(cid)
        evaluator.newer = export_then_check
        evaluator.rate(next(e for e in dense_eval.checkpoints(self.run) if e[0] == 'main/000020'))
        self.assertEqual((evaluator.entry('main/000020')['skipped'], self.league()['champion']), (True, 'main/000010'))

    def test_half_finished_pairs_hold_their_slots_against_the_budget(self):
        """Every colour-0 game finishes before any colour-1 game: the finished halves still count against the
        comparison's budget, so exactly sprt_max_games games are played."""
        evaluator = self.start(sprt_max_games=8, pool_games=8)
        self.export(10)
        evaluator.step()
        self.export(20)
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(winner=lambda r: -1, order=lambda r: r['challenger_color'])):
            self.assertTrue(evaluator.step())
        self.assertEqual(len(evaluator.games('main/000020', 'main/000010')), 8)

    def test_games_finished_on_creation_occupy_the_pool(self):
        pool = dense_eval.Pool(64)
        lane = ('a', 'b', 'evidence')
        pool.ready.append((lane, {}))                                      # e.g. Seal won inside MatchGame.__init__
        self.assertEqual((pool.running(), pool.running(lane), pool.running(kind='evidence')), (1, 1, 1))
        self.assertEqual(pool.step(), [(lane, {})])
        self.assertEqual(pool.running(), 0)

    def test_draining_lanes_count_against_the_pool_and_their_kind(self):
        """Switching the evidence pairing every pair never lifts the games in flight above pool_games, nor the
        evidence games above the evidence wanted, while the old pairing drains."""
        evaluator = self.start(pool_games=8)
        self.export(10, 20, 30, 40)
        evaluator.league['checkpoints'] = [dict(id=f'main/{k:06d}', variant='main', step=k, elo=None, elo_interval=None,
                                                matches=[]) for k in (10, 20, 30, 40)]
        rivals, calls, seen = ['main/000020', 'main/000030'], [0], []
        def want():
            calls[0] += 1
            if calls[0] > 12:
                return {}
            return {('main/000040', 'main/000010', 'champion'): 4, ('main/000040', rivals[calls[0] % 2], 'evidence'): 4}
        def watch(pool, steps):
            seen.append((pool.running(), pool.running(kind='evidence')))
        long = lambda record: 30 if record['opponent'] == 'main/000010' else 5   # main lane games are longer
        with unittest.mock.patch.object(dense_eval, 'Pool', scripted(hook=watch, moves=long)):
            evaluator.session(want, 0)
        status = json.loads((self.run/'evaluator-status.json').read_text())
        self.assertEqual((status['comparison']['opponent'], status['mean_placements']), ('main/000010', 30.))
        self.assertEqual(max(total for total, _ in seen), 8)
        self.assertEqual(max(evidence for _, evidence in seen), 4)
        self.assertTrue(all(evaluator.games('main/000040', r) for r in rivals))

    def test_models_stay_loaded_while_their_pairing_is_in_the_pool(self):
        evaluator = self.start()
        self.export(10, 20, 30)
        evaluator.use('main/000010', 'main/000020')
        first = evaluator.models['main/000010']
        evaluator.use('main/000020', 'main/000010')
        self.assertIs(evaluator.models['main/000010'], first)       # no reload
        evaluator.use('main/000030', 'main/000010')                 # main/000020 has no lane left
        self.assertEqual(set(evaluator.models), {'main/000010', 'main/000030'})
        self.assertIs(evaluator.models['main/000010'], first)

    def test_evaluation_defaults(self):
        s = dense_config.EvaluationSettings()
        self.assertEqual((s.sprt_elo0, s.sprt_elo1, s.pool_games, s.sprt_min_games, s.evidence_share, s.idle_fill,
                          s.max_expected_score), (0., 25., 64, 64, .25, True, .85))

    def test_existing_panels_skip_uninformative_members(self):
        """Panel members are re-derived (a stored list is ignored) and include only those within
        max_expected_score."""
        entry = lambda step, elo, **extra: dict(id=f'main/{step:06d}', variant='main', step=step, elo=elo, matches=[], **extra)
        (self.run/'league.json').write_text(json.dumps(dict(champion='main/000001', matrix={}, ladder=[], ladder_top=3, calibration={}, checkpoints=[
            entry(1, 0.), entry(2, -400*math.log10(19)), entry(3, -400*math.log10(7/3)),
            entry(4, 0., panel=dict(members=['main/000002', 'main/000003'], incumbent='main/000001'))])))
        evaluator = self.start(extra_opponents=2)
        self.assertEqual(evaluator.needs(evaluator.entry('main/000004')),
                         [('main/000004', 'main/000003', 'panel', 2), ('main/000001', 'main/000003', 'incumbent', 2)])
        evaluator.settings = replace(evaluator.settings, max_expected_score=1.)
        self.assertEqual(len(evaluator.needs(evaluator.entry('main/000004'))), 4)

    def test_restart_rebuilds_the_ladder_for_a_new_fill_top(self):
        evaluator = self.start()
        for step in (10, 20):
            self.export(step)
            self.assertTrue(evaluator.step())
        self.assertEqual((self.league()['ladder_top'], len(self.league()['ladder'])), (3, 1))
        self.start(fill_top=1)
        self.assertEqual((self.league()['ladder_top'], self.league()['ladder']), (1, []))

    def test_fill_is_off_without_idle_fill(self):
        evaluator = self.start(anchor_target_halfwidth=25.)
        self.export(10)
        self.assertTrue(evaluator.step())
        self.assertIsNone(evaluator.fill())
        self.assertFalse(evaluator.step())

    def test_pool_size_does_not_change_the_games(self):
        self.export(10, 30)
        played = []
        for pool in (2, 6):
            shutil.rmtree(self.run/'evaluations', ignore_errors=True)
            evaluator = self.start(games=6, pool_games=pool, max_plies=16)
            evaluator.league['checkpoints'] = [dict(id=cid, variant='main', step=int(cid.split('/')[1]), elo=None,
                                                    elo_interval=None, matches=[]) for cid in ('main/000010', 'main/000030')]
            lane = ('main/000030', 'main/000010', 'previous')
            evaluator.session(lambda: {lane: even for even in [min(pool, 6-len(evaluator.games(*lane[:2])))] if even}, 6)
            played.append(sorted(evaluator.games(*lane[:2]), key=lambda r: (r['pair'], r['challenger_color'])))
        self.assertEqual([r['pair'] for r in played[1]], [0, 0, 1, 1, 2, 2])
        self.assertEqual(played[0], played[1])

    def test_panel_and_sprt_rematch_are_optional_rounds(self):
        evaluator = self.start(extra_opponents=1, idle_rematch=True)
        for step in (10, 20, 30):
            self.export(step)
            self.assertTrue(evaluator.step())
        self.assertEqual(self.league()['checkpoints'][-1]['panel'], dict(incumbent='main/000010'))
        # A league.json from before the matrix: a new evaluator adds it.
        league = self.league(); del league['matrix']
        (self.run/'league.json').write_text(json.dumps(league))
        evaluator = self.start(extra_opponents=1, idle_rematch=True)
        self.assertIn('main/000030', self.league()['matrix'])
        self.export(40)
        self.assertTrue(evaluator.step())                                  # the champion SPRT only
        entry = self.league()['checkpoints'][-1]
        self.assertEqual(entry['panel'], dict(incumbent='main/000010'))
        self.assertEqual([m['opponent'] for m in entry['matches']], ['main/000010'])
        sprt = json.loads(dense_eval.report_path(self.run, 'main/000040', 'main/000010').read_text())
        self.assertEqual((sprt['metrics']['sprt']['decision'], len(sprt['games'])), ('max-games', 2))
        # Idle: the undecided SPRT continues by one round first, then the panel game, then nothing.
        self.assertEqual(evaluator.optional()[1:], ('main/000010', 'sprt', 2))
        self.assertTrue(evaluator.step())
        sprt = json.loads(dense_eval.report_path(self.run, 'main/000040', 'main/000010').read_text())
        self.assertEqual(([g['pair'] for g in sprt['games']], sprt['metrics']['sprt']['games']), ([0, 0, 1, 1], 4))
        member = evaluator.optional()[1]
        self.assertEqual(evaluator.optional()[1:], (member, 'panel', 2))
        self.assertIn(member, {'main/000020', 'main/000030'})
        for _ in range(8):                                                 # members follow the ratings as games land
            if not evaluator.step():
                break
        league = self.league()
        panel = league['checkpoints'][-1]['panel']
        self.assertEqual(set(panel), {'members', 'incumbent', 'candidate_score', 'incumbent_score', 'z', 'veto'})
        self.assertTrue(panel['members'] and set(panel['members']) <= {'main/000020', 'main/000030'})
        self.assertTrue(all(league['matrix']['main/000040'][m]['games'] >= 2 for m in panel['members']))
        self.assertFalse(evaluator.step())
        kinds = [e.get('comparison') for e in map(json.loads, (self.run/'events.jsonl').read_text().splitlines()) if e['kind'] == 'match']
        self.assertEqual((kinds[-1], kinds.count('sprt')), ('panel', 1))

    def test_h1_promotes_before_any_panel_game(self):
        evaluator = self.start(extra_opponents=1)
        for step in (10, 20, 30):
            self.export(step)
            self.assertTrue(evaluator.step())
        self.assertEqual(self.league()['champion'], 'main/000010')
        test = evaluator.test
        evaluator.test = lambda records: dict(test(records), decision='H1')
        self.export(40)
        self.assertTrue(evaluator.step())
        league = self.league()
        self.assertEqual(league['champion'], 'main/000040')
        self.assertEqual(json.loads((self.run/'champion.json').read_text())['checkpoint'], 'main/000040')
        self.assertEqual(league['checkpoints'][-1]['panel'], dict(incumbent='main/000010'))
        entry, member, kind, games = evaluator.optional()                        # the champion's panel comes first
        self.assertEqual((entry['id'], kind, games), ('main/000040', 'panel', 2))
        self.assertIn(member, {'main/000020', 'main/000030'})
        self.assertFalse(dense_eval.report_path(self.run, 'main/000040', member).exists())

    def test_panel_regression_demotes_the_champion(self):
        self.export(10, 20, 30, 40)
        evaluator = self.start(extra_opponents=1)
        cell = lambda a, b, w, l: dict(candidate=a, opponent=b, settings=asdict(evaluator.settings),
                                       summary=dict(wins=w, losses=l, capped=0, games=w+l), metrics={}, games=[])
        for regressed in (True, False):
            evaluator.league = dict(champion='main/000040', checkpoints=[
                dict(id=f'main/{k:06d}', variant='main', step=k, elo=0., matches=[]) for k in (10, 20)])
            evaluator.league['checkpoints'].append(dict(id='main/000040', variant='main', step=40, elo=0., matches=[],
                                                        panel=dict(incumbent='main/000010')))
            reports = [cell('main/000040', 'main/000020', 0 if regressed else 2, 20 if regressed else 0),
                       cell('main/000010', 'main/000020', 15, 5)]
            with unittest.mock.patch.object(dense_eval, 'write_league'), \
                    unittest.mock.patch.object(dense_eval, 'load_reports', lambda *args: reports), \
                    unittest.mock.patch.object(dense_eval, 'report_path', lambda run, a, b: self.run):
                evaluator.settle()
            entry = evaluator.league['checkpoints'][-1]
            self.assertEqual((entry['panel']['veto'], entry.get('demoted', False)), (regressed, regressed))
            self.assertEqual(evaluator.league['champion'], 'main/000010' if regressed else 'main/000040')
        self.assertEqual(json.loads((self.run/'champion.json').read_text())['checkpoint'], 'main/000010')
        events = [json.loads(line) for line in (self.run/'events.jsonl').read_text().splitlines()]
        self.assertEqual([(e['checkpoint'], e['restored']) for e in events if e['kind'] == 'regression'], [('main/000040', 'main/000010')])

    def test_rematch_rounds_stop_at_their_budget(self):
        evaluator = self.start(games=4, idle_rematch=True)
        for step in (10, 20):
            self.export(step)
            self.assertTrue(evaluator.step())
        path = dense_eval.report_path(self.run, 'main/000020', 'main/000010')
        self.assertEqual(len(json.loads(path.read_text())['games']), 2)             # sprt_max_games 2
        self.assertEqual(evaluator.optional()[1:], ('main/000010', 'sprt', 2))      # REMATCH_SPRT_LIMIT * 2 - 2 left
        self.assertTrue(evaluator.step())
        self.assertEqual(len(json.loads(path.read_text())['games']), 4)             # not 2 + games
        self.assertIsNone(evaluator.optional())
        # A replacement rematch is offered only the games left below sprt_max_games.
        evaluator = self.start(games=4, sprt_max_games=6, idle_rematch=True)
        evaluator.league['checkpoints'] = []
        evaluator.league['differences'] = [dict(a='main/000020', b='side/000010', elo_delta=0., interval=[-100., 100.])]
        played = dict(candidate='side/000010', opponent='main/000020', settings=asdict(evaluator.settings),
                      summary=dict(wins=1, losses=1, capped=0, games=2), metrics={}, games=[])
        with unittest.mock.patch.object(dense_eval, 'load_reports', lambda *args: [played]):
            self.assertEqual(evaluator.rematches(), [('main/000020', 'side/000010', 'replacement', 4)])

    def test_rematches_skip_reports_of_another_protocol(self):
        evaluator = self.start(idle_rematch=True)
        for step in (10, 20):
            self.export(step)
            self.assertTrue(evaluator.step())
        path = dense_eval.report_path(self.run, 'main/000020', 'main/000010')
        report = json.loads(path.read_text())
        self.assertEqual(evaluator.rematches(), [('main/000020', 'main/000010', 'sprt', 2)])
        path.write_text(json.dumps(dict(report, settings=dict(report['settings'], sims=64))))   # an --eval-sims restart
        self.assertEqual(evaluator.rematches(), [])
        self.assertIsNone(dense_eval.rematch_pair(self.run, 'main/000020', 'main/000010', evaluator.settings)) \
            if (dense_eval.report_path(self.run, 'main/000010', 'main/000020').exists()) else None
        self.assertEqual(dense_eval.rematch_pair(self.run, 'main/000020', 'main/000010', evaluator.settings),
                         ('main/000010', 'main/000020'))

    def test_incumbent_tops_up_an_undersized_report(self):
        self.export(10, 20, 40)
        evaluator = self.start(games=4, extra_opponents=1)
        evaluator.league = dict(champion='main/000040', checkpoints=[
            dict(id=f'main/{k:06d}', variant='main', step=k, elo=0., matches=[]) for k in (10, 20)])
        evaluator.league['checkpoints'].append(dict(id='main/000040', variant='main', step=40, elo=0., matches=[],
                                                    panel=dict(incumbent='main/000010')))
        lane = ('main/000010', 'main/000020', 'incumbent')                          # e.g. played before --eval-games 4
        evaluator.session(lambda: {} if evaluator.games(*lane[:2]) else {lane: 2}, 2)
        entry = evaluator.league['checkpoints'][-1]
        self.assertEqual(evaluator.needs(entry), [('main/000040', 'main/000020', 'panel', 4),
                                                  ('main/000010', 'main/000020', 'incumbent', 2)])
        lane = ('main/000040', 'main/000020', 'panel')
        evaluator.session(lambda: {} if len(evaluator.games(*lane[:2])) >= 4 else {lane: 2}, 4)
        self.assertEqual(evaluator.needs(entry), [('main/000010', 'main/000020', 'incumbent', 2)])  # not complete yet
        self.assertEqual(evaluator.optional()[1:], ('main/000020', 'incumbent', 2))
        self.assertTrue(evaluator.step())
        report = json.loads(dense_eval.report_path(self.run, 'main/000010', 'main/000020').read_text())
        self.assertEqual([g['pair'] for g in report['games']], [0, 0, 1, 1])
        self.assertEqual(evaluator.needs(entry), [])
        self.assertIn('veto', entry['panel'])                                        # judged once complete

    def test_successive_vetoed_champions_restore_the_last_sound_predecessor(self):
        self.export(10, 20, 30, 40)
        evaluator = self.start(extra_opponents=1)
        cell = lambda a, w, l: dict(candidate=a, opponent='main/000020', settings=asdict(evaluator.settings),
                                    summary=dict(wins=w, losses=l, capped=0, games=w+l), metrics={}, games=[])
        reports = [cell('main/000010', 15, 5), cell('main/000030', 5, 15), cell('main/000040', 0, 20)]
        for chained in (True, False):
            panel = lambda incumbent: dict(incumbent=incumbent)
            evaluator.league = dict(champion='main/000040', checkpoints=[
                dict(id='main/000010', variant='main', step=10, elo=0., matches=[]),
                dict(id='main/000020', variant='main', step=20, elo=0., matches=[]),
                dict(id='main/000030', variant='main', step=30, elo=0., matches=[], **({'panel': panel('main/000010')} if chained else {})),
                dict(id='main/000040', variant='main', step=40, elo=0., matches=[], panel=panel('main/000030'))])
            if not chained:
                evaluator.league['checkpoints'][2]['demoted'] = True       # vetoed earlier, predecessor unknown
            with unittest.mock.patch.object(dense_eval, 'write_league'), \
                    unittest.mock.patch.object(dense_eval, 'load_reports', lambda *args: reports), \
                    unittest.mock.patch.object(dense_eval, 'report_path', lambda run, a, b: self.run):
                evaluator.settle()                                          # one pass judges both panels, oldest first
            entries = {c['id']: c for c in evaluator.league['checkpoints']}
            self.assertTrue(entries['main/000030']['demoted'] and entries['main/000040']['demoted'])
            self.assertEqual(evaluator.league['champion'], 'main/000010' if chained else 'main/000040')
        events = [json.loads(line) for line in (self.run/'events.jsonl').read_text().splitlines() if '"regression"' in line]
        self.assertEqual([(e['checkpoint'], e['restored']) for e in events],
                         [('main/000030', None), ('main/000040', 'main/000010'), ('main/000040', None)])
        self.assertEqual(json.loads((self.run/'champion.json').read_text())['checkpoint'], 'main/000010')

    def vetoed_on_disk(self):
        """A league whose champion main/000040 has a complete, vetoing panel on disk (incumbent main/000010), as
        left by an evaluator that exited before settling it."""
        self.export(10, 20, 40)
        settings = self.start().settings
        (self.run/'league.json').write_text(json.dumps(dict(champion='main/000040', matrix={}, checkpoints=[
            dict(id='main/000010', variant='main', step=10, elo=0., matches=[]),
            dict(id='main/000020', variant='main', step=20, elo=0., matches=[]),
            dict(id='main/000040', variant='main', step=40, elo=0., matches=[],
                 panel=dict(incumbent='main/000010'))])))
        for a, wins, losses in (('main/000040', 0, 20), ('main/000010', 15, 5)):
            path = dense_eval.report_path(self.run, a, 'main/000020')
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(candidate=a, opponent='main/000020', settings=asdict(settings), metrics={}, games=[],
                                            summary=dict(wins=wins, losses=losses, capped=0, games=wins+losses))))

    def test_restart_settles_panels_completed_on_disk(self):
        self.vetoed_on_disk()
        with unittest.mock.patch.object(dense_eval, 'write_league'):
            evaluator = self.start(extra_opponents=1)
            self.assertEqual(evaluator.needs(evaluator.entry('main/000040')), [])       # nothing left to play
            self.assertFalse(evaluator.step())
        entry = evaluator.entry('main/000040')
        self.assertTrue(entry['panel']['veto'] and entry['demoted'])
        self.assertEqual(evaluator.league['champion'], 'main/000010')
        self.assertEqual(json.loads((self.run/'champion.json').read_text())['checkpoint'], 'main/000010')

    def test_restart_settles_before_rating_a_new_checkpoint(self):
        self.vetoed_on_disk()
        self.export(50)
        with unittest.mock.patch.object(dense_eval, 'write_league'):
            evaluator = self.start(extra_opponents=1)
            self.assertTrue(evaluator.step())
        self.assertTrue(evaluator.entry('main/000040')['demoted'])
        self.assertTrue(dense_eval.report_path(self.run, 'main/000050', 'main/000010').exists())    # the restored champion
        self.assertFalse(dense_eval.report_path(self.run, 'main/000050', 'main/000040').exists())
        self.assertEqual([m['opponent'] for m in evaluator.entry('main/000050')['matches']], ['main/000010'])

    def test_demoted_checkpoints_are_not_variant_heads(self):
        league = dict(champion='main/000010', checkpoints=[
            dict(id='main/000010', variant='main', step=10, elo=0., matches=[]),
            dict(id='main/000020', variant='main', step=20, elo=90., matches=[], demoted=True),
            dict(id='main/000030', variant='main', step=30, skipped=True, elo=None, matches=[]),
            dict(id='side/000010', variant='side', step=10, elo=10., matches=[])])
        self.assertEqual({v: c['id'] for v, c in dense_learn.latest_rated(league).items()},
                         {'main': 'main/000010', 'side': 'side/000010'})
        rated = lambda names, anchor, reports, seed: ({n: 0. for n in names}, {n: [0., 0.] for n in names}, {n: [0.] for n in names})
        self.start()
        with unittest.mock.patch.object(dense_eval, 'rate', rated), unittest.mock.patch.object(dense_eval, 'write_json'):
            dense_eval.write_league(self.run, league, dense_config.load(self.run))
        self.assertEqual([(d['a'], d['b']) for d in league['differences']], [('main/000010', 'side/000010')])

    def test_panel_of_the_head_below_a_demoted_checkpoint_is_scheduled(self):
        evaluator = self.start(extra_opponents=1)
        evaluator.league = dict(champion='main/000010', checkpoints=[
            dict(id='main/000010', variant='main', step=10, elo=0., matches=[]),
            dict(id='main/000020', variant='main', step=20, elo=0., matches=[]),
            dict(id='main/000030', variant='main', step=30, elo=0., matches=[], panel=dict(incumbent='main/000010')),
            dict(id='main/000040', variant='main', step=40, elo=0., matches=[], demoted=True,
                 panel=dict(members=['main/000020'], incumbent='main/000030', veto=True))])
        self.assertEqual([c['id'] for c in evaluator.heads()], ['main/000030'])
        entry, opponent, kind, games = evaluator.optional()
        self.assertEqual((entry['id'], opponent, kind, games), ('main/000030', 'main/000020', 'panel', 2))

    def test_calibration_compares_the_stated_sd_with_the_later_shift(self):
        """league.json calibration counts a verdict once its checkpoint has three later comparisons: the delta sd it
        stated, the RMS a calibrated posterior expects (sd then^2 - sd now^2) and the realised shift of delta."""
        self.export(10, 20, 30, 40, 50)
        self.start()
        self.report('main/000020', 'main/000010', [1, 1, 0, 1]*4)
        entry = lambda step: dict(id=f'main/{step:06d}', variant='main', step=step, elo=0., elo_interval=None, matches=[])
        evaluator = self.start(decision='posterior', sprt_min_games=16, matchup_prior_elo=20.)       # not config.json's
        evaluator.league = league = dict(champion='main/000010', checkpoints=[entry(10), entry(20)])
        verdict = dense_eval.public(evaluator.verdict('main/000020', 'main/000010'))
        snapshot = evaluator.snapshot('main/000020', 'main/000010')
        self.assertEqual({k: v['games'] for k, v in snapshot['reports'].items()}, {'main-000020-vs-main-000010': 16})
        league['checkpoints'][1]['verdict'] = dict(verdict, candidate='main/000020', **snapshot)
        config = dense_config.load(self.run)
        dense_eval.write_league(self.run, league, config)
        self.assertEqual(league['calibration'], dict(count=0, predicted_sd=None, expected_rms=None, realised_rms=None))
        for step, results in ((30, [0, 0, 1, 0]*2), (40, [0, 1]*4)):
            league['checkpoints'].append(entry(step))
            self.report(f'main/{step:06d}', 'main/000020', results)
        dense_eval.write_league(self.run, league, config)
        self.assertEqual(league['calibration']['count'], 0)                            # two later comparisons
        league['checkpoints'].append(entry(50))
        self.report('main/000020', 'main/000050', [1, 0]*4)
        dense_eval.write_league(self.run, league, config)
        reports = dense_eval.load_reports(self.run)
        post = dense_eval.Posterior([f'main/{s:06d}' for s in (10, 20, 30, 40, 50)], 'main/000010',
                                    [(r['candidate'], r['opponent'], r['summary']['wins']+r['summary']['capped']/2,
                                      r['summary']['games']) for r in reports], 20.,
                                    dense_eval.parents([f'main/{s:06d}' for s in (10, 20, 30, 40, 50)]))
        mean, sd = post.difference('main/000020', 'main/000010')
        calibration = league['calibration']
        self.assertEqual(calibration['count'], 1)
        self.assertAlmostEqual(calibration['predicted_sd'], verdict['delta_sd'])
        self.assertAlmostEqual(calibration['expected_rms'], math.sqrt(verdict['delta_sd']**2-sd**2))
        self.assertAlmostEqual(calibration['realised_rms'], abs(mean-verdict['delta']))
        self.assertGreater(calibration['realised_rms'], 0.)
        reports = league['checkpoints'][1]['verdict']['reports']
        digest = reports['main-000020-vs-main-000010']['digest']
        reports['main-000020-vs-main-000010']['digest'] = '0'*16
        dense_eval.write_league(self.run, league, config)
        self.assertEqual(league['calibration']['count'], 0)                            # the direct report was replaced
        reports['main-000020-vs-main-000010']['digest'] = digest
        reports['main-000030-vs-main-000040'] = dict(games=2, digest='0'*16)
        dense_eval.write_league(self.run, league, config)
        self.assertEqual(league['calibration']['count'], 0)                            # an input report is gone
        del reports['main-000030-vs-main-000040']
        self.report('main/000020', 'main/000010', [1, 1, 0, 1]*4+[1, 0])              # extended: still counted
        dense_eval.write_league(self.run, league, config)
        self.assertEqual(league['calibration']['count'], 1)
        league['checkpoints'][1]['verdict']['protocol']['sims'] += 1                  # decided under another protocol
        dense_eval.write_league(self.run, league, config)
        self.assertEqual(league['calibration']['count'], 0)
        del league['calibration']
        (self.run/'league.json').write_text(json.dumps(league))
        self.start()
        self.assertIn('calibration', self.league())                                    # added on start

if __name__ == '__main__':
    unittest.main()
