"""Exact sparse batched Hexo rules on torch devices.

Coordinates have the same +/-10**12 representation limit as native Hexo, with
no board crop. Storage grows with stone count. Each step places one stone;
call twice for a normal turn, checking termination between placements.
GPU actors can rank candidates with exact changes in the native 729 patterns.
"""
from dataclasses import dataclass

import torch


@dataclass
class StepResult:
    accepted: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class BatchedHexo:
    def __init__(self, batch_size, device="cuda", capacity=128, max_placements=None):
        if batch_size < 1 or capacity < 1:
            raise ValueError("batch_size and capacity must be positive")
        if max_placements is not None and max_placements < 1:
            raise ValueError("max_placements must be positive or None")
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.max_placements = max_placements
        self.coords = torch.zeros((batch_size, capacity, 2), device=device, dtype=torch.int64)
        self.owners = torch.full((batch_size, capacity), -1, device=device, dtype=torch.int8)
        self.counts = torch.zeros(batch_size, device=device, dtype=torch.int64)
        self.player = torch.zeros_like(self.counts)
        self.remaining = torch.ones_like(self.counts)
        self.winner = torch.full_like(self.counts, -1)
        self.truncated = torch.zeros(batch_size, device=device, dtype=torch.bool)
        self.features = torch.zeros((batch_size, 729), device=device, dtype=torch.int32)
        self._rows = torch.arange(batch_size, device=device)
        self._upper_count = 0
        self._powers = 3 ** torch.arange(6, device=device)
        self._windows = 5 - torch.arange(6, device=device)[:, None] + torch.arange(6, device=device)[None, :]
        offsets = [(q, r) for q in range(-8, 9) for r in range(-8, 9)
                   if max(abs(q), abs(r), abs(q+r)) <= 8]
        local = [(q, r) for q, r in offsets if max(abs(q), abs(r), abs(q+r)) <= 2]
        axial = [(q*k, r*k) for q, r in ((1, 0), (0, 1), (1, -1))
                 for k in range(-5, 6) if k]
        banks = (local, local, axial, offsets)
        self._sampling_counts = torch.tensor([len(bank) for bank in banks], device=device)
        self._sampling_offsets = torch.tensor([bank + [(0, 0)]*(217-len(bank)) for bank in banks], device=device)

    @property
    def active(self):
        return (self.winner < 0) & ~self.truncated

    def reset(self, mask=None):
        """Reset all games or selected rows without moving observations to the CPU."""
        if mask is None:
            mask = torch.ones_like(self.truncated)
            self._upper_count = 0
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        self.owners.masked_fill_(mask[:, None], -1)
        self.counts.masked_fill_(mask, 0)
        self.player.masked_fill_(mask, 0)
        self.remaining.masked_fill_(mask, 1)
        self.winner.masked_fill_(mask, -1)
        self.truncated.masked_fill_(mask, False)
        self.features.masked_fill_(mask[:, None], 0)

    def _actions(self, actions):
        actions = torch.as_tensor(actions, device=self.device)
        if actions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("Coordinates must be integers")
        if actions.ndim == 2:
            actions = actions[:, None, :]
        if actions.ndim != 3 or actions.shape[0] != self.batch_size or actions.shape[-1] != 2:
            raise ValueError("Expected [batch, candidates, 2] coordinates")
        return actions.to(torch.int64)

    def _differences(self, actions):
        # Clamp before subtracting so arbitrary int64 caller input cannot overflow.
        actions = actions.clamp(-10**12-16, 10**12+16)
        return self.coords[:, None, :, :] - actions[:, :, None, :]

    def legal(self, actions):
        actions = self._actions(actions)
        delta = self._differences(actions)
        q, r = delta.unbind(-1)
        occupied = self.owners[:, None, :] >= 0
        empty = ~((q == 0) & (r == 0) & occupied).any(-1)
        near = ((q.abs() <= 8) & (r.abs() <= 8) & ((q+r).abs() <= 8) & occupied).any(-1)
        opening = (actions == 0).all(-1)
        bounded = ((actions >= -10**12) & (actions <= 10**12)).all(-1)
        return empty & torch.where(self.counts[:, None] == 0, opening, near) & bounded & self.active[:, None]

    def candidate_codes(self, actions):
        """Return before/after codes [B,K,3,6] for windows through each candidate.

        The hypothetical replacement assumes an empty cell. Use legal() to mask
        illegal candidates. Colors are native player 0 -> digit 1, player 1 -> 2.
        """
        actions = self._actions(actions)
        q, r = self._differences(actions).unbind(-1)
        offset = torch.stack((q, r, q), dim=2)
        on_axis = torch.stack((r == 0, q == 0, q+r == 0), dim=2)
        valid = on_axis & (offset.abs() <= 5) & (self.owners[:, None, None, :] >= 0)
        digits = (self.owners[:, None, None, :].to(torch.int64) + 1) * valid
        lines = torch.zeros((*offset.shape[:-1], 11), device=self.device, dtype=torch.int64)
        lines.scatter_add_(-1, (offset+5).clamp(0, 10), digits)
        windows = lines[..., self._windows]
        before = (windows * self._powers).sum(-1)
        after = before + (self.player[:, None, None, None]+1) * self._powers
        return before, after

    def candidate_features(self, actions):
        """Exact native histogram deltas [B,K,729], for legal candidates only."""
        before, after = self.candidate_codes(actions)
        return self._feature_delta(before, after)

    def _feature_delta(self, before, after):
        before, after = before.flatten(-2), after.flatten(-2)
        delta = torch.zeros((*before.shape[:-1], 729), device=self.device, dtype=torch.int32)
        delta.scatter_add_(-1, before, -(before != 0).to(torch.int32))
        # Occupied candidates are not hypothetical placements; clamp codes so
        # their masked-out feature rows cannot cause out-of-bounds accesses.
        delta.scatter_add_(-1, after.clamp_max(728), torch.ones_like(after, dtype=torch.int32))
        return delta

    def sampled_candidates(self, count=32, generator=None):
        """Sample local candidates and return coordinates plus their legal mask.

        Half the samples are within radius two, a quarter lie along a board
        axis within five cells, and a quarter use the full legal radius eight.
        Duplicates are allowed. The final candidate extends the rightmost stone
        by one cell, giving every active game a legal fallback. At the positive
        coordinate representation limit it instead extends the leftmost stone.
        This is a policy candidate subset, not an enumeration of legal moves.
        """
        if count < 1:
            raise ValueError("count must be positive")
        anchor_index = (torch.rand((self.batch_size, count), device=self.device, generator=generator)
                        * self.counts[:, None].clamp_min(1)).to(torch.int64)
        anchors = self.coords[self._rows[:, None], anchor_index]
        groups = torch.arange(count, device=self.device) % 4
        offsets = (torch.rand((self.batch_size, count), device=self.device, generator=generator)
                   * self._sampling_counts[groups]).to(torch.int64)
        candidates = anchors + self._sampling_offsets[groups, offsets]
        occupied = self.owners >= 0
        q = self.coords[..., 0]
        right = q.masked_fill(~occupied, -10**12-1).argmax(-1)
        left = q.masked_fill(~occupied, 10**12+1).argmin(-1)
        right_cell = self.coords[self._rows, right]
        left_cell = self.coords[self._rows, left]
        use_left = right_cell[:, 0] >= 10**12
        fallback = torch.where(use_left[:, None], left_cell, right_cell).clone()
        fallback[:, 0] += torch.where(use_left, -1, 1)
        candidates[:, -1] = fallback
        candidates = torch.where((self.counts == 0)[:, None, None], 0, candidates)
        return candidates, self.legal(candidates)

    def _grow(self):
        if self._upper_count < self.coords.shape[1]:
            return
        # A sync only at potential capacity boundaries, never each placement.
        self._upper_count = int(self.counts.max().item())
        if self._upper_count < self.coords.shape[1]:
            return
        self.coords = torch.cat((self.coords, torch.zeros_like(self.coords)), dim=1)
        self.owners = torch.cat((self.owners, torch.full_like(self.owners, -1)), dim=1)

    @torch.no_grad()
    def step(self, actions):
        actions = self._actions(actions)
        if actions.shape[1] != 1:
            raise ValueError("step takes exactly one placement per game")
        self._grow()
        accepted = self.legal(actions).squeeze(1)
        before, after = self.candidate_codes(actions)
        won = (after == torch.where(self.player == 0, 364, 728)[:, None, None, None]).flatten(1).any(1) & accepted
        self.features += self._feature_delta(before, after).squeeze(1) * accepted[:, None]
        index = self.counts.clamp_max(self.coords.shape[1]-1)
        self.coords[self._rows, index] = torch.where(accepted[:, None], actions[:, 0], self.coords[self._rows, index])
        self.owners[self._rows, index] = torch.where(accepted, self.player, self.owners[self._rows, index]).to(torch.int8)
        self.counts += accepted
        self.winner = torch.where(won, self.player, self.winner)
        remaining = self.remaining - accepted.to(torch.int64)
        switched = accepted & (remaining == 0)
        self.player = torch.where(switched, 1-self.player, self.player)
        self.remaining = torch.where(switched, 2, remaining)
        if self.max_placements is not None:
            self.truncated |= (self.counts >= self.max_placements) & (self.winner < 0)
        self._upper_count += 1
        return StepResult(accepted, self.winner >= 0, self.truncated.clone())


@torch.no_grad()
def generate_games(tasks, *, candidates=32, epsilon=0.1, device="cuda", progress=None):
    """Generate replay with a GPU sampled-candidate, one-placement policy.

    This actor is separate from native search. `ms` and `width` do not apply;
    frozen-opponent promotion evaluation must continue using native search.
    Values and outcomes use player 0's sign, matching train.play_game.
    """
    import time
    import numpy as np
    from learning_model import pattern_data
    from train import family, load_table, opening_for

    if not tasks:
        return []
    if any(task["evaluation"] for task in tasks):
        raise ValueError("GPU policy generation is self-play only; evaluate with native search")
    if candidates < 1 or not 0 <= epsilon <= 1:
        raise ValueError("candidates must be positive and epsilon must be in [0,1]")
    if any(task["max_stones"] < 3 for task in tasks):
        raise ValueError("max_stones must accommodate the three-stone opening")
    started = time.perf_counter()
    batch = len(tasks)
    cap = max(task["max_stones"] for task in tasks)
    env = BatchedHexo(batch, device, capacity=min(cap, 128), max_placements=cap)
    openings = [opening_for(task["seed"], False) for task in tasks]
    opening_tensor = torch.tensor(openings, device=device)
    for i in range(3):
        env.step(opening_tensor[:, i])
    caps = torch.tensor([task["max_stones"] for task in tasks], device=device)
    env.truncated |= env.counts >= caps
    baseline = pattern_data(device)[3]
    table_cache = {str(path): load_table(path) for task in tasks for path in task["tables"]}
    residuals = torch.tensor(np.stack([[table_cache[str(path)] for path in task["tables"]]
                                     for task in tasks]), device=device, dtype=torch.float32)
    tables = residuals + baseline
    batch_seed = sum((i+1)*int(task["seed"]) for i, task in enumerate(tasks)) % (2**63-1)
    rng = torch.Generator(device=device).manual_seed(batch_seed)
    saved_features, saved_values, saved_masks = [], [], []
    rows = torch.arange(batch, device=device)
    for ply in range(3, cap):
        active = env.active
        actions, legal = env.sampled_candidates(candidates, generator=rng)
        before, after = env.candidate_codes(actions)
        table = tables[rows, env.player]
        before_value = table.gather(1, before.flatten(1)).reshape_as(before)
        after_value = table.gather(1, after.clamp_max(728).flatten(1)).reshape_as(after)
        current_value = (env.features * table).sum(1)
        values = current_value[:, None] + (after_value-before_value).sum((-1, -2))
        signed = values * torch.where(env.player[:, None] == 0, 1, -1)
        winning = (after == torch.where(env.player == 0, 364, 728)[:, None, None, None]).flatten(2).any(2)
        # An immediate sampled win is mandatory even under exploration.
        signed = torch.where(winning, float("inf"), signed).masked_fill(~legal, -float("inf"))
        best = signed.argmax(1)
        random_pick = torch.rand((batch, candidates), device=device, generator=rng).masked_fill(~legal, -1).argmax(1)
        explore = (torch.rand(batch, device=device, generator=rng) < epsilon) & ~(winning & legal).any(1)
        chosen = torch.where(explore, random_pick, best)
        saved_features.append(env.features.clone())
        saved_values.append(torch.tanh(values[rows, chosen]/6000))
        saved_masks.append(active.clone())
        env.step(actions[rows, chosen])
        env.truncated |= (env.counts >= caps) & (env.winner < 0)
        if (ply-2) % 16 == 0 or ply == cap-1:
            completed = int((~env.active).sum().item())
            if progress:
                progress(completed, batch, {"actor": "gpu-pattern-one-ply", "placements": ply+1})
            if completed == batch:
                break
    if env.device.type == "cuda":
        torch.cuda.synchronize(env.device)
    elapsed = time.perf_counter()-started
    coords, owners = env.coords.cpu().numpy(), env.owners.cpu().numpy()
    winners = env.winner.cpu().tolist()
    counts = env.counts.cpu().tolist()
    if saved_features:
        features = torch.stack(saved_features).cpu().numpy()
        values = torch.stack(saved_values).cpu().numpy()
        masks = torch.stack(saved_masks).cpu().numpy()
    results = []
    for i, task in enumerate(tasks):
        winner = winners[i]
        fid = family(openings[i])
        record = {"seed": task["seed"], "family": fid, "winner": winner,
                  "reason": "six-in-a-row" if winner >= 0 else "truncated",
                  "cells": [[int(q), int(r), int(owner)] for (q, r), owner in zip(coords[i, :counts[i]], owners[i, :counts[i]])],
                  "times": [[], []], "depths": [], "tables": task["tables"],
                  "challenger_color": task.get("challenger_color"),
                  "actor": "gpu-pattern-one-ply", "actor_candidates": candidates,
                  "actor_epsilon": epsilon, "actor_batch_seed": batch_seed,
                  "actor_batch_seconds": elapsed}
        f = features[masks[:, i], i] if saved_features else np.empty((0, 729), dtype=np.int32)
        v = values[masks[:, i], i] if saved_features else np.empty(0, dtype=np.float32)
        outcome = (1 if winner == 0 else -1) if winner >= 0 else float("nan")
        samples = {"features": f, "search": v,
                   "outcome": np.full(len(f), outcome, dtype=np.float32),
                   "family": np.full(len(f), fid, dtype=np.uint32)}
        results.append((record, samples))
    return results

def recommended_batch(candidates=32, max_stones=256, device="cuda"):
    """Size an actor batch from free VRAM, leaving 35% for other allocations.

    Accounts conservatively for candidate/stone comparisons, temporary pattern
    codes, and both history lists and the final stacked replay. The final task
    count should also cap the caller's batch. CPU operation uses 64 games.
    """
    if candidates < 1 or max_stones < 3:
        raise ValueError("candidates must be positive and max_stones at least three")
    device = torch.device(device)
    if device.type != "cuda":
        return 64
    free, _ = torch.cuda.mem_get_info(device)
    capacity = max(128, 1 << (max_stones-1).bit_length())
    per_game = capacity*candidates*192 + max_stones*729*12 + 65536
    count = max(1, min(2048, int(free*.65/per_game)))
    return max(1, count//64*64) if count >= 64 else count
