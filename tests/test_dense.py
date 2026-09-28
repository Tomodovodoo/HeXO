"""CPU checks for the dense hex ResNet stack: hexcrop, hexnet, dense_config, dense_data, dense_bootstrap, the
learner's validation and the actor/evaluator engine."""
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

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
            torch.testing.assert_close(conv(x), 2*expected, atol=1e-5, rtol=1e-5)   # cache follows updates
            wide = torch.randn(1, 3, hexnet.CACHED_SIZE+1, hexnet.CACHED_SIZE+1)
            torch.testing.assert_close(conv(wide), line_conv_reference(wide, conv.weight), atol=1e-5, rtol=1e-5)
            self.assertEqual({k[0] for k in conv.cache}, {9})

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
            points = [(x, 100. if x == 777 else math.sin(x)) for x in range(5000)]
            kept = dashboard.downsample(points, 100)
            self.assertLessEqual(len(kept), 100)
            self.assertEqual((kept[0], kept[-1]), (points[0], points[-1]))
            self.assertIn((777, 100.), kept)

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
        prefixed = argparse.ArgumentParser()
        dense_config.add_arguments(prefixed, dense_config.ActorSettings)
        dense_config.add_arguments(prefixed, dense_config.EvaluationSettings, 'eval_')
        args = prefixed.parse_args(['--root-samples', '8', '--eval-root-samples', '4', '--no-eval-tactics'])
        self.assertEqual(dense_config.override(base, args).root_samples, 8)
        evaluation = dense_config.override(dense_config.EvaluationSettings(), args, 'eval_')
        self.assertEqual((evaluation.root_samples, evaluation.tactics), (4, False))

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
            self.assertEqual(window.index, [('000002', i) for i in range(2, 10)]+[('000003', i) for i in range(10)])
            capped = dense_data.ReplayWindow(tmp, capacity_rows=5, min_rows=12)
            self.assertEqual(capped.index, [('000003', i) for i in range(5, 10)])
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
            self.assertEqual(written['counts'], dict(games=2, rows=21, policy_rows=21, terminal_games=1, capped_games=1))
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


def source_shard(path, seed, actor, origin='actor', checkpoint=None, games=6, policy_every=2):
    """A shard of `games` random capped games; game g is played by actor[g % len(actor)] for a list, else by `actor`
    (the identity's actor_sha256 is the last one). Actor shards get a dense_selfplay-like identity."""
    rng = np.random.default_rng(seed)
    actors = actor if isinstance(actor, list) else [actor]
    episodes, rows = [], []
    for g in range(games):
        moves, _ = random_game(rng, 8)
        e, r = episode_rows(moves, -1, [float(v) for v in rng.uniform(-1, 1, len(moves))], rng, policy_every)
        episodes.append(dict(e, actor=actors[g % len(actors)]))
        rows += [dict(x, game=g) for x in r]
    identity = dict(source='gumbel-policy-value-v1', actor_sha256=actors[-1]) if origin == 'converted' else \
        dict(actor_sha256=actors[-1], actors=sorted(set(actors)), checkpoint=checkpoint)
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

    def test_newest_change_selects_a_cached_successor(self):
        """A row cached only as a chosen row's next-ply successor can be chosen after the newest actor changes."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            source_shard(run/'shards'/'1000000000001', 1, ['x', 'y'], checkpoint='main/000010', games=8, policy_every=1)
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
                                             'metrics', 'learner', 'model', 'copied_from'})
            v = manifest['metrics']['validation']
            self.assertEqual(v['newest_checkpoint'], 'main/000010')
            for h in dense_learn.HEADS:
                self.assertTrue(math.isfinite(v[h]))
            for source in dense_data.SOURCES:
                self.assertEqual(v[f'{source}_rows'], len(sets.subsets[source, 'held']))
                for name in ('policy_ce', 'value_bce'):
                    self.assertAlmostEqual(v[f'{source}_gap_{name}'], v[f'{source}_{name}']-v[f'{source}_train_{name}'])
            self.assertEqual(v, learner.validate(window) | learner.validate_sources(sets))
            path = run/'checkpoints'/'main'/'000000'
            self.assertEqual(sorted(p.name for p in path.iterdir()), ['ema.pt', 'manifest.json', 'model.pt', 'optimizer.pt'])
            self.assertEqual(dense_learn.Learner(run, config.learner, config).step, 0)
            dense_config.append_metrics(run, 'learner-main', step=10, validation=True,
                                        **{dense_learn.LOGGED.get(h, h): x for h, x in v.items()})
            points = dashboard.series(run, dict(created_at=0.), 'main', 'validation_newest_gap_policy_ce')['points']
            self.assertEqual(points, [[10, v['newest_gap_policy_ce']]])


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


if __name__ == '__main__':
    unittest.main()
