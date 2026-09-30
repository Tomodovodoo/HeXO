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


@dataclass
class TacticalAction:
    action: torch.Tensor
    forced: torch.Tensor
    winning: torch.Tensor
    defending: torch.Tensor


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
        # Each occupied anchor stores its 18 six-cell windows. Duplicates avoid
        # hashing and do not change completion or defensive-cover decisions.
        self._window_codes = torch.zeros((batch_size, capacity, 3, 6), device=device, dtype=torch.int16)
        self._axes = torch.tensor(((1, 0), (0, 1), (1, -1)), device=device)
        codes = torch.arange(729, device=device)
        digits = codes[:, None] // (3 ** torch.arange(6, device=device)) % 3
        empty = digits == 0
        missing = empty.sum(1)
        self._missing_first = empty.to(torch.int64).argmax(1)
        self._missing_last = 5 - empty.flip(1).to(torch.int64).argmax(1)
        self._completion_size = torch.stack([
            torch.where(((digits == 0) | (digits == color)).all(1) & (missing >= 1) & (missing <= 2), missing, 7)
            for color in (1, 2)])
        self._tactical_codes = ((self._completion_size <= 2).any(0)).nonzero().flatten()
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
        self._window_codes.masked_fill_(mask[:, None, None, None], 0)

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

    def tactical_action(self):
        """Find an exact win this turn, otherwise a complete next-turn defense.

        A pure four/five-stone six-cell window needs at most two empty cells.
        Those cells are within five of an existing stone, so either placement
        order is legal. Windows crossing the coordinate bound are excluded.
        Defense hits every opponent one/two-cell completion set using at most
        `remaining` stones. Recompute after placement to finish a chosen pair.
        This is exact immediate tactics, not a search of later turns.
        """
        moves = torch.zeros((self.batch_size, 2), dtype=torch.int64, device=self.device)
        win = torch.zeros_like(self.truncated)
        defend = torch.zeros_like(self.truncated)
        # Histograms cheaply identify quiet games. Only tactical rows allocate
        # endpoint tensors; nonzero synchronizes once to compact the batch.
        potential = self.active & (self.features[:, self._tactical_codes].sum(1) > 0)
        ids = potential.nonzero().flatten()
        if not ids.numel():
            return TacticalAction(moves, win | defend, win, defend)
        codes = self._window_codes[ids].to(torch.int64)
        owner_valid = (self.owners[ids] >= 0)[:, :, None, None]
        shift = torch.arange(6, device=self.device)
        first = self.coords[ids, :, None, None, :] + self._axes[None, None, :, None, :] * (self._missing_first[codes] - shift)[..., None]
        last = self.coords[ids, :, None, None, :] + self._axes[None, None, :, None, :] * (self._missing_last[codes] - shift)[..., None]
        bounded = ((first >= -10**12) & (first <= 10**12) & (last >= -10**12) & (last <= 10**12)).all(-1)
        valid = (owner_valid & bounded).flatten(1)
        first, last = first.flatten(1, 3), last.flatten(1, 3)
        codes = codes.flatten(1)
        rows = torch.arange(len(ids), device=self.device)
        side = self.player[ids]
        needed = self._completion_size[side[:, None], codes]
        own = valid & (needed <= self.remaining[ids, None])
        own_index = needed.masked_fill(~own, 7).argmin(1)
        has_win = own.any(1)
        own_move = first[rows, own_index]

        threat = valid & (self._completion_size[1-side[:, None], codes] <= 2)
        threat_index = threat.to(torch.int8).argmax(1)
        # Any cover must contain one endpoint of the first threat.
        branch = torch.stack((first[rows, threat_index], last[rows, threat_index]), dim=1)
        hit = ((branch[:, :, None, :] == first[:, None, :, :]).all(-1)
               | (branch[:, :, None, :] == last[:, None, :, :]).all(-1))
        uncovered = threat[:, None, :] & ~hit
        single = ~uncovered.any(-1)
        # Once an endpoint is chosen, any second stone must hit the first
        # remaining threat. Its two endpoints exhaust the possible covers.
        next_index = uncovered.to(torch.int8).argmax(-1)
        second = torch.stack((first[rows[:, None], next_index], last[rows[:, None], next_index]), dim=2)
        second_hit = ((second[:, :, :, None, :] == first[:, None, None, :, :]).all(-1)
                      | (second[:, :, :, None, :] == last[:, None, None, :, :]).all(-1))
        pairs = ~(uncovered[:, :, None, :] & ~second_hit).any(-1)
        cover = single | ((self.remaining[ids, None] == 2) & pairs.any(-1))
        has_cover = cover.any(1) & threat.any(1)
        defend_move = branch[rows, cover.to(torch.int8).argmax(1)]
        moves[ids] = torch.where(has_win[:, None], own_move, defend_move)
        win[ids] = has_win
        defend[ids] = has_cover & ~has_win
        return TacticalAction(moves, win | defend, win, defend)

    def _grow(self):
        if self._upper_count < self.coords.shape[1]:
            return
        # A sync only at potential capacity boundaries, never each placement.
        self._upper_count = int(self.counts.max().item())
        if self._upper_count < self.coords.shape[1]:
            return
        self.coords = torch.cat((self.coords, torch.zeros_like(self.coords)), dim=1)
        self.owners = torch.cat((self.owners, torch.full_like(self.owners, -1)), dim=1)
        self._window_codes = torch.cat((self._window_codes, torch.zeros_like(self._window_codes)), dim=1)

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
        # Update every previously registered window that contains this move.
        q, r = (actions[:, 0, None, :].clamp(-10**12-16, 10**12+16) - self.coords).unbind(-1)
        position = torch.stack((q, r, q), dim=-1)[..., None] + torch.arange(6, device=self.device)
        aligned = torch.stack((r == 0, q == 0, q+r == 0), dim=-1)[..., None]
        changed = aligned & (position >= 0) & (position < 6) & (self.owners >= 0)[:, :, None, None] & accepted[:, None, None, None]
        increment = self._powers[position.clamp(0, 5)] * (self.player+1)[:, None, None, None] * changed
        self._window_codes += increment.to(torch.int16)
        index = self.counts.clamp_max(self.coords.shape[1]-1)
        self._window_codes[self._rows, index] = torch.where(accepted[:, None, None], after[:, 0], self._window_codes[self._rows, index]).to(torch.int16)
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
    """Generate replay with exact immediate tactics and sampled quiet moves.

    This actor is separate from native search. `ms` and `width` do not apply;
    frozen-opponent promotion evaluation must continue using native search.
    Values and outcomes use player 0's sign, matching train.play_game.
    """
    import time
    import numpy as np
    from legacy.learning_model import pattern_data
    from legacy.train import family, load_table, opening_for

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
    openings = [task["opening"] if "opening" in task else opening_for(task["seed"], False) for task in tasks]
    if any(not p or len(p) > task["max_stones"] for p, task in zip(openings, tasks)):
        raise ValueError("Opening prefixes must be nonempty and fit their episode caps")
    prefix_lengths = torch.tensor([len(p) for p in openings], device=device)
    longest = max(map(len, openings))
    opening_tensor = torch.tensor([p + [(0, 0)]*(longest-len(p)) for p in openings], device=device)
    for i in range(longest):
        accepted = env.step(opening_tensor[:, i]).accepted
        if not torch.equal(accepted, i < prefix_lengths):
            raise ValueError("An opening prefix contains an illegal or post-terminal placement")
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
    for ply in range(min(map(len, openings)), cap):
        active = env.active
        actions, legal = env.sampled_candidates(candidates, generator=rng)
        tactical = env.tactical_action()
        actions = torch.cat((actions, tactical.action[:, None, :]), dim=1)
        legal = torch.cat((legal, tactical.forced[:, None]), dim=1)
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
        random_pick = torch.rand((batch, candidates+1), device=device, generator=rng).masked_fill(~legal, -1).argmax(1)
        explore = (torch.rand(batch, device=device, generator=rng) < epsilon) & ~(winning & legal).any(1)
        chosen = torch.where(tactical.forced, candidates, torch.where(explore, random_pick, best))
        saved_features.append(env.features.clone())
        target_value = torch.tanh(values[rows, chosen]/6000)
        target_value = torch.where(tactical.winning, torch.where(env.player == 0, 1., -1.), target_value)
        saved_values.append(target_value)
        saved_masks.append(active.clone())
        env.step(actions[rows, chosen])
        env.truncated |= (env.counts >= caps) & (env.winner < 0)
        if (ply-2) % 16 == 0 or ply == cap-1:
            completed = int((~env.active).sum().item())
            if progress:
                progress(completed, batch, {"actor": "gpu-pattern-tactical", "placements": int(env.counts.max().item())})
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
        fid = task["family"] if "family" in task else family(openings[i])
        record = {"seed": task["seed"], "family": fid, "winner": winner,
                  "reason": "six-in-a-row" if winner >= 0 else "truncated",
                  "cells": [[int(q), int(r), int(owner)] for (q, r), owner in zip(coords[i, :counts[i]], owners[i, :counts[i]])],
                  "times": [[], []], "depths": [], "tables": task["tables"],
                  "challenger_color": task.get("challenger_color"),
                  "actor": "gpu-pattern-tactical", "actor_candidates": candidates,
                  "opening": openings[i], "opening_length": len(openings[i]),
                  "curriculum": task.get("curriculum", "legacy"),
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
    per_game = capacity*(candidates*192 + 4096) + max_stones*729*12 + 65536
    count = max(1, min(2048, int(free*.65/per_game)))
    return max(1, count//64*64) if count >= 64 else count
