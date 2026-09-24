"""Frozen NNUE inference and complete-turn GPU self-play.

The sparse center cache uses the native eleven-cell codes and exported int16
LUT. Exact immediate tactics precede a selective, conditional two-placement
beam. Replay snapshots move to host memory every placement.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from gpu_games import BatchedHexo
from nnue_model import HEADER, PATTERNS


class ModelBank:
    def __init__(self, paths, device="cuda"):
        from hexo import native_model
        self.device = torch.device(device)
        self.paths = list(dict.fromkeys(str(Path(p).resolve()) for p in paths))
        tables, heads = [], []
        for path in self.paths:
            native_model(path)  # Full native format, bounds and symmetry validation.
            content = Path(path).read_bytes()
            header = HEADER.unpack_from(content)
            if header[0] != b"HXNNUE1\0" or len(content) != HEADER.size + header[-1]:
                raise ValueError("Invalid NNUE artifact")
            start = HEADER.size
            tables.append(np.frombuffer(content, dtype="<i2", count=PATTERNS*32, offset=start).reshape(PATTERNS, 32).copy())
            heads.append(np.frombuffer(content, dtype="<f4", offset=start+PATTERNS*64).copy())
        self.table = torch.as_tensor(np.stack(tables), device=device)
        data = torch.as_tensor(np.stack(heads), device=device)
        offset = 0
        for name, shape in (("vw", (32, 68)), ("vb", (32,)), ("vo", (32,)), ("vob", (1,)),
                            ("pw", (16, 104)), ("pb", (16,)), ("po", (16,)), ("pob", (1,))):
            size = int(np.prod(shape))
            setattr(self, name, data[:, offset:offset+size].reshape(-1, *shape))
            offset += size
        if offset != data.shape[1]:
            raise ValueError("Unexpected NNUE head size")

    def rows(self, model, codes):
        shape = (len(model),) + (1,)*(codes.ndim-1)
        return self.table[model.reshape(shape), codes.long().clamp(0, PATTERNS-1)].to(torch.int32)

    def head(self, inputs, model, policy=False):
        w, b, out, bias = ((self.pw, self.pb, self.po, self.pob) if policy else (self.vw, self.vb, self.vo, self.vob))
        # Explicit reductions avoid TF32 changing the frozen native head.
        hidden = (inputs[..., None, :] * w[model, None]).sum(-1) + b[model, None]
        return (hidden.relu()*out[model, None]).sum(-1) + bias[model]


@dataclass
class Overlay:
    points: torch.Tensor
    codes: torch.Tensor
    pool: torch.Tensor
    count: torch.Tensor
    stones: torch.Tensor
    player: torch.Tensor
    remaining: torch.Tensor
    baseline: torch.Tensor
    won: torch.Tensor
    action: torch.Tensor

    def column(self, index):
        return Overlay(**{name: value[:, index] for name, value in vars(self).items()})


class NNUEBatch:
    def __init__(self, model_paths, device="cuda", capacity=128, max_placements=None):
        self.env = BatchedHexo(len(model_paths), device, capacity, max_placements)
        self.device = self.env.device
        self.bank = ModelBank([p for pair in model_paths for p in pair], device)
        self.model_ids = torch.tensor([[self.bank.paths.index(str(Path(p).resolve())) for p in pair]
                                       for pair in model_paths], device=device)
        batch = len(model_paths)
        self.shared_models = torch.equal(self.model_ids[:, 0], self.model_ids[:, 1])
        self.codes = torch.zeros((batch, capacity*31, 3), device=device, dtype=torch.int32)
        self.alias = torch.zeros((batch, capacity, 31), device=device, dtype=torch.int32)
        self.center_counts = torch.zeros(batch, device=device, dtype=torch.int64)
        self.pool = torch.zeros((batch, 2, 64), device=device, dtype=torch.int64)
        self.stones = torch.zeros((batch, 2), device=device, dtype=torch.int64)
        self.last = torch.zeros((batch, 2), device=device, dtype=torch.int64)
        self.last_player = torch.full((batch,), -1, device=device, dtype=torch.int64)
        self.axes = torch.tensor(((1, 0), (0, 1), (1, -1)), device=device)
        offsets, increments = [(0, 0)], [[3**5]*3]
        for axis, (q, r) in enumerate(((1, 0), (0, 1), (1, -1))):
            for k in range(-5, 6):
                if k:
                    offsets.append((q*k, r*k))
                    delta = [0, 0, 0]
                    delta[axis] = 3**(5-k)
                    increments.append(delta)
        self.offsets = torch.tensor(offsets, device=device)
        self.increments = torch.tensor(increments, device=device, dtype=torch.int32)
        self.perspective = torch.tensor([*range(32, 48), *range(16, 32), *range(16), *range(48, 64)], device=device)
        digits = torch.arange(729, device=device)[:, None] // (3**torch.arange(6, device=device)) % 3
        p0, p1 = (digits == 1).sum(1), (digits == 2).sum(1)
        weights = torch.tensor([0, 1, 12, 150, 2400, 24000, 1000000], device=device)
        self.baseline_table = torch.where(p1 == 0, weights[p0], torch.where(p0 == 0, -weights[p1], 0))

    def lookup(self, points, parent=None):
        """Resolve arbitrary centers through an occupied stone's 31 aliases."""
        q, r = (points[:, :, None] - self.env.coords[:, None]).unbind(-1)
        horizontal = (r == 0) & (q.abs() <= 5)
        vertical = (q == 0) & (r.abs() <= 5)
        diagonal = (q+r == 0) & (q.abs() <= 5)
        match = (horizontal | vertical | diagonal) & (self.env.owners[:, None] >= 0)
        found = match.any(-1)
        anchor = match.to(torch.int8).argmax(-1)
        dq, dr = q.gather(2, anchor[..., None]).squeeze(-1), r.gather(2, anchor[..., None]).squeeze(-1)
        axis = torch.where(dr == 0, 0, torch.where(dq == 0, 1, 2))
        k = torch.where(axis == 1, dr, dq)
        slot = torch.where((dq == 0) & (dr == 0), 0, 1+axis*10+torch.where(k < 0, k+5, k+4)).clamp(0, 30)
        ids = self.alias[self.env._rows[:, None], anchor, slot]
        codes = self.codes[self.env._rows[:, None], ids]
        codes = torch.where(found[..., None], codes, 0)
        if parent is not None:
            equal = (points[:, :, None] == parent.points[:, None]).all(-1)
            updated = equal.any(-1)
            index = equal.to(torch.int8).argmax(-1)
            codes = torch.where(updated[..., None], parent.codes[self.env._rows[:, None], index], codes)
            found = found | updated
        return codes, found, ids

    def phase(self, player, remaining, stones):
        n = stones.sum(-1)
        side = stones.gather(-1, player[..., None]).squeeze(-1)
        return torch.stack(((remaining == 1).float(), (remaining == 2).float(),
                            n.float().log1p()/8, (2*side-n).float()/n.clamp_min(1)), -1)

    def inputs(self, pool, count, player, remaining, stones, actor):
        rows = self.env._rows[:, None]
        selected = pool[rows, torch.arange(pool.shape[1], device=self.device)[None], actor[:, None]]
        values = selected.float()/(256*count.clamp_min(1).float().sqrt()[..., None])
        values = torch.where((player == 0)[..., None], values, values[..., self.perspective])
        return torch.cat((values, self.phase(player, remaining, stones)), -1)

    def baseline(self):
        return (self.env.features.to(torch.int64)*self.baseline_table).sum(-1)

    def state_inputs(self, actor=None, parent=None):
        if actor is None:
            actor = self.env.player
        if parent is None:
            return self.inputs(self.pool[:, None], self.center_counts[:, None], self.env.player[:, None],
                               self.env.remaining[:, None], self.stones[:, None], actor)[:, 0]
        return self.inputs(parent.pool[:, None], parent.count[:, None], parent.player[:, None],
                           parent.remaining[:, None], parent.stones[:, None], actor)[:, 0]

    def legal(self, actions, parent=None):
        if parent is None:
            return self.env.legal(actions)
        codes = self.lookup(actions, parent)[0]
        empty = codes[..., 0] // (3**5) % 3 == 0
        q, r = (actions-parent.action[:, None]).unbind(-1)
        near = (q.abs() <= 8) & (r.abs() <= 8) & ((q+r).abs() <= 8)
        bounded = ((actions >= -10**12) & (actions <= 10**12)).all(-1)
        return (self.env.legal(actions) | near) & empty & bounded & self.env.active[:, None] & ~parent.won[:, None]

    def policy(self, actions, actor=None, parent=None):
        if actor is None:
            actor = self.env.player
        player = self.env.player if parent is None else parent.player
        remaining = self.env.remaining if parent is None else parent.remaining
        previous = self.last if parent is None else parent.action
        previous_player = self.last_player if parent is None else self.env.player
        codes = self.lookup(actions, parent)[0] + (player+1)[:, None, None]*(3**5)
        local = self.bank.rows(self.model_ids[self.env._rows, actor], codes).sum(-2).float()/256
        local[..., :16] *= torch.where(player == 0, 1, -1)[:, None, None]
        q, r = (actions-previous[:, None]).unbind(-1)
        distance = torch.maximum(torch.maximum(q.abs(), r.abs()), (q+r).abs())
        axis = (q == 0) | (r == 0) | (q+r == 0)
        pairs = torch.stack((torch.ones_like(distance), axis, distance.clamp_max(8).float()/8,
                             torch.where(axis, (6-distance).clamp_min(0).float()/5, 0)), -1).float()
        pairs *= ((remaining == 1) & (previous_player == player))[:, None, None]
        state = self.state_inputs(actor, parent)
        inputs = torch.cat((state[:, None, :64].expand(-1, actions.shape[1], -1), local,
                            state[:, None, 64:].expand(-1, actions.shape[1], -1), pairs), -1)
        rank = self.bank.head(inputs, self.model_ids[self.env._rows, actor], policy=True)
        return rank, codes, pairs

    def virtual(self, actions, parent=None):
        """Evaluate immutable placement overlays; the live cache is unchanged."""
        batch, width, _ = actions.shape
        points = actions[:, :, None]+self.offsets
        old, exists, _ = self.lookup(points.flatten(1, 2), parent)
        old, exists = old.reshape(batch, width, 31, 3), exists.reshape(batch, width, 31)
        player = self.env.player if parent is None else parent.player
        remaining = self.env.remaining if parent is None else parent.remaining
        pool = self.pool if parent is None else parent.pool
        count = self.center_counts if parent is None else parent.count
        stones = self.stones if parent is None else parent.stones
        baseline = self.baseline() if parent is None else parent.baseline
        new = old + (player+1)[:, None, None, None]*self.increments
        updated = []
        for side in range(1 if self.shared_models else 2):
            before = self.bank.rows(self.model_ids[:, side], old).sum(-2)
            after = self.bank.rows(self.model_ids[:, side], new).sum(-2)
            delta = torch.cat((after.relu()-before.relu(), (-after).relu()-(-before).relu()), -1).sum(-2)
            updated.append(pool[:, side, None]+delta)
        if self.shared_models:
            updated.append(updated[0])
        pools = torch.stack(updated, dim=2)
        counts = count[:, None] + (~exists).sum(-1)
        stone_counts = stones[:, None].expand(-1, width, -1) + torch.nn.functional.one_hot(player, 2)[:, None]
        switch = remaining == 1
        next_player = torch.where(switch, 1-player, player)[:, None].expand(-1, width)
        next_remaining = torch.where(switch, 2, remaining-1)[:, None].expand(-1, width)
        before, _ = self.env.candidate_codes(actions)
        if parent is not None:
            q, r = (parent.action[:, None]-actions).unbind(-1)
            offset = torch.stack((q, r, q), -1)[..., None]+torch.arange(6, device=self.device)
            aligned = torch.stack((r == 0, q == 0, q+r == 0), -1)[..., None]
            included = aligned & (offset >= 0) & (offset < 6)
            before += (self.env.player+1)[:, None, None, None] * self.env._powers[offset.clamp(0, 5)] * included
        after = before+(player+1)[:, None, None, None]*self.env._powers
        score = baseline[:, None] + (self.baseline_table[after.clamp(0, 728)]-self.baseline_table[before.clamp(0, 728)]).sum((-1, -2))
        won = (after == torch.where(player == 0, 364, 728)[:, None, None, None]).flatten(2).any(-1)
        return Overlay(points, new, pools, counts, stone_counts, next_player, next_remaining, score, won, actions)

    def value(self, overlay, actor):
        inputs = self.inputs(overlay.pool, overlay.count, overlay.player, overlay.remaining, overlay.stones, actor)
        residual = self.bank.head(inputs, self.model_ids[self.env._rows, actor])
        sign = torch.where(overlay.player == 0, 1, -1)
        score = (overlay.baseline + (residual*6000).clamp(-1000000, 1000000).trunc()*sign).clamp(-500000, 500000)
        root_sign = torch.where(self.env.player == 0, 1, -1)[:, None]
        result = torch.tanh(score*root_sign/6000)
        # All leaves belong to the original actor's current turn.
        return torch.where(overlay.won, 1., result)

    def step(self, actions):
        accepted = self.env.legal(actions).squeeze(1)
        overlay = self.virtual(actions[:, None]).column(0)
        _, exists, old_ids = self.lookup(overlay.points)
        ids = torch.where(exists, old_ids, self.center_counts[:, None]+(~exists).cumsum(1)-1)
        capacity = max(self.env.coords.shape[1], self.env._upper_count+1)
        if capacity > self.alias.shape[1]:
            self.alias = torch.cat((self.alias, torch.zeros_like(self.alias)), 1)
            self.codes = torch.cat((self.codes, torch.zeros_like(self.codes)), 1)
        # New center rows are unique. Inactive rows preserve all cache entries.
        safe_ids = ids.clamp(0, self.codes.shape[1]-1)
        prior = self.codes[self.env._rows[:, None], safe_ids]
        self.codes[self.env._rows[:, None], safe_ids] = torch.where(accepted[:, None, None], overlay.codes, prior).to(torch.int32)
        slot = self.env.counts.clamp_max(self.alias.shape[1]-1)
        self.alias[self.env._rows, slot] = torch.where(accepted[:, None], ids, self.alias[self.env._rows, slot]).to(torch.int32)
        self.pool = torch.where(accepted[:, None, None], overlay.pool, self.pool)
        self.center_counts = torch.where(accepted, overlay.count, self.center_counts)
        self.stones = torch.where(accepted[:, None], overlay.stones, self.stones)
        self.last = torch.where(accepted[:, None], actions, self.last)
        self.last_player = torch.where(accepted, self.env.player, self.last_player)
        return self.env.step(actions)

    def sample(self, count, generator, parent=None):
        actions, valid = self.env.sampled_candidates(count, generator)
        if parent is None:
            return actions, valid
        groups = torch.arange(count, device=self.device) % 4
        index = (torch.rand((self.env.batch_size, count), device=self.device, generator=generator)
                 * self.env._sampling_counts[groups]).long()
        around_first = parent.action[:, None]+self.env._sampling_offsets[groups, index]
        actions[:, :max(1, count//2)] = around_first[:, :max(1, count//2)]
        occupied = self.env.owners >= 0
        q = self.env.coords[..., 0]
        right = self.env.coords[self.env._rows, q.masked_fill(~occupied, -10**12-1).argmax(1)]
        left = self.env.coords[self.env._rows, q.masked_fill(~occupied, 10**12+1).argmin(1)]
        right = torch.where((parent.action[:, 0] > right[:, 0])[:, None], parent.action, right)
        left = torch.where((parent.action[:, 0] < left[:, 0])[:, None], parent.action, left)
        use_left = right[:, 0] >= 10**12
        fallback = torch.where(use_left[:, None], left, right).clone()
        fallback[:, 0] += torch.where(use_left, -1, 1)
        actions[:, -1] = fallback
        return actions, self.legal(actions, parent)

    @staticmethod
    def unique(actions, valid):
        count = actions.shape[1]
        previous = torch.arange(count, device=actions.device)[None, :] < torch.arange(count, device=actions.device)[:, None]
        same = (actions[:, :, None] == actions[:, None, :]).all(-1)
        duplicate = (same & previous & valid[:, None, :]).any(-1)
        return valid & ~duplicate

    def choose(self, candidates, generator, epsilon=.1, beam=4, planned=None):
        env = self.env
        actions, valid = self.sample(candidates, generator)
        tactics = env.tactical_action()
        actions = torch.cat((actions, tactics.action[:, None]), 1)
        valid = torch.cat((valid, tactics.forced[:, None]), 1)
        if planned is not None:
            actions = torch.cat((actions, planned[:, None]), 1)
            valid = torch.cat((valid, self.legal(planned[:, None]) & (env.remaining == 1)[:, None]), 1)
        valid = self.unique(actions, valid)
        rank, codes, pairs = self.policy(actions)
        first_ids = rank.masked_fill(~valid, -float("inf")).topk(min(beam, actions.shape[1]), 1).indices
        if planned is not None:
            planned_match = (actions == planned[:, None]).all(-1) & valid
            planned_id = planned_match.to(torch.int8).argmax(1)
            reserve = planned_match.any(1) & (env.remaining == 1) & ~(first_ids == planned_id[:, None]).any(1)
            first_ids[:, -1] = torch.where(reserve, planned_id, first_ids[:, -1])
        first = actions[env._rows[:, None], first_ids]
        first_valid = valid.gather(1, first_ids)
        overlays = self.virtual(first)
        values = self.value(overlays, env.player)
        next_moves = first.clone()
        two = (env.remaining == 2) & env.active & ~tactics.forced
        if two.any().item():
            for k in range(first.shape[1]):
                parent = overlays.column(k)
                second, mask = self.sample(candidates, generator, parent)
                mask = self.unique(second, mask)
                second_rank = self.policy(second, parent=parent)[0]
                second_ids = second_rank.masked_fill(~mask, -float("inf")).topk(min(beam, candidates), 1).indices
                second = second[env._rows[:, None], second_ids]
                leaves = self.virtual(second, parent)
                leaf_values = self.value(leaves, env.player).masked_fill(~mask.gather(1, second_ids), -float("inf"))
                best = leaf_values.argmax(1)
                complete = leaf_values[env._rows, best]
                values[:, k] = torch.where(two, complete, values[:, k])
                next_moves[:, k] = second[env._rows, best]
        values = values.masked_fill(~first_valid, -float("inf"))
        best = values.argmax(1)
        selected = first_ids[env._rows, best]
        search = values[env._rows, best]
        planned = next_moves[env._rows, best]
        explore = (torch.rand(env.batch_size, device=self.device, generator=generator) < epsilon) & ~tactics.forced & env.active
        random_pick = torch.rand(valid.shape, device=self.device, generator=generator).masked_fill(~valid, -1).argmax(1)
        selected = torch.where(explore, random_pick, selected)
        forced_index = (((actions == tactics.action[:, None]).all(-1)) & valid).to(torch.int8).argmax(1)
        selected = torch.where(tactics.forced, forced_index, selected)
        search = torch.where(tactics.winning, 1., search)
        # Defensive labels are exact covers, but their value is only an estimate.
        teacher = env.active & ~explore
        search_valid = teacher & (~tactics.defending)
        search = torch.nan_to_num(search, nan=0., neginf=-1., posinf=1.)
        return {"actions": actions, "valid": valid, "codes": codes, "pairs": pairs,
                "chosen": selected, "search": search, "search_valid": search_valid,
                "policy_valid": teacher, "planned": planned,
                "move": actions[env._rows, selected]}

@torch.no_grad()
def generate_games(tasks, *, candidates=32, epsilon=.1, device="cuda", progress=None, beam=4):
    """Generate NNUE replay with selective complete-turn search as the teacher.

    Exploration labels are invalid, exact tactical covers label the policy,
    and proven wins label both heads. Native search remains the evaluator.
    No learned six-cell table is substituted for an NNUE model.
    """
    import time
    from train import opening_for, family, pack_nnue
    if not tasks:
        return []
    if any(t.get("model_kind") != "nnue" or t["evaluation"] for t in tasks):
        raise ValueError("GPU NNUE generation requires NNUE self-play tasks")
    if candidates < 1 or beam < 1 or not 0 <= epsilon <= 1:
        raise ValueError("Invalid candidate count, beam, or exploration")
    openings = [t.get("opening") or opening_for(t["seed"], False) for t in tasks]
    if any(t["max_stones"] < len(opening) for t, opening in zip(tasks, openings)):
        raise ValueError("Placement cap is shorter than the opening")
    start = time.perf_counter()
    cap = max(t["max_stones"] for t in tasks)
    state = NNUEBatch([t["tables"] for t in tasks], device, min(cap, 128), cap)
    env = state.env
    caps = torch.tensor([t["max_stones"] for t in tasks], device=device)
    for index in range(max(map(len, openings))):
        moves = [opening[index][:2] if index < len(opening) else (0, 0) for opening in openings]
        expected = torch.tensor([index < len(opening) for opening in openings], device=device)
        result = state.step(torch.tensor(moves, device=device))
        if not torch.equal(result.accepted, expected):
            raise ValueError("Illegal or terminal opening prefix")
    env.truncated |= env.counts >= caps
    seed = sum((i+1)*int(t["seed"]) for i, t in enumerate(tasks)) % (2**63-1)
    generator = torch.Generator(device=device).manual_seed(seed)
    examples = [[] for _ in tasks]
    planned = None
    steps = max(t["max_stones"]-len(o) for t, o in zip(tasks, openings))
    for iteration in range(steps):
        if not env.active.any().item():
            break
        decision = state.choose(candidates, generator, epsilon, beam, planned)
        planned = decision["planned"]
        # Stream ragged snapshots to CPU. GPU memory holds only this position.
        active = env.active
        counts = state.center_counts*active
        keep = torch.arange(state.codes.shape[1], device=device)[None] < counts[:, None]
        centers = state.codes[keep].cpu().numpy()
        sizes = counts.cpu().tolist()
        copied = {key: decision[key].cpu().numpy() for key in ("actions", "valid", "codes", "pairs", "chosen", "search", "search_valid", "policy_valid")}
        phase = state.phase(env.player, env.remaining, state.stones).cpu().numpy()
        players, plies = env.player.cpu().tolist(), env.counts.cpu().tolist()
        baseline = (state.baseline()*torch.where(env.player == 0, 1, -1)).cpu().numpy()
        cursor = 0
        for i, count in enumerate(sizes):
            if not count:
                continue
            choices = np.flatnonzero(copied["valid"][i])
            selected = int(copied["chosen"][i])
            local_index = np.flatnonzero(choices == selected)
            if len(local_index) != 1:
                raise RuntimeError("NNUE actor selected an illegal placement")
            examples[i].append({"centers": centers[cursor:cursor+count],
                "candidate_codes": copied["codes"][i, choices].astype(np.int32),
                "pairs": copied["pairs"][i, choices], "candidate_coords": copied["actions"][i, choices],
                "phase": phase[i], "player": players[i], "baseline": baseline[i],
                "chosen": int(local_index[0]), "search": copied["search"][i],
                "search_valid": copied["search_valid"][i], "policy_valid": copied["policy_valid"][i],
                "ply": plies[i], "search_depth": 0, "search_ms": 0.})
            cursor += count
        result = state.step(decision["move"])
        if not torch.equal(result.accepted, active):
            raise RuntimeError("NNUE actor generated an illegal placement")
        env.truncated |= (env.counts >= caps) & (env.winner < 0)
        if progress and ((iteration+1) % 16 == 0 or iteration+1 == steps):
            progress(int((~env.active).sum().item()), len(tasks), {"actor": "gpu-nnue-turn-beam", "placements": iteration+1})
    elapsed = time.perf_counter()-start
    coords, owners, lengths = env.coords.cpu().numpy(), env.owners.cpu().numpy(), env.counts.cpu().tolist()
    winners = env.winner.cpu().tolist()
    results = []
    for i, task in enumerate(tasks):
        fid = task.get("family")
        if fid is None:
            fid = family(openings[i])
        outcome = (1 if winners[i] == 0 else -1) if winners[i] >= 0 else float("nan")
        if examples[i]:
            samples = pack_nnue(examples[i], outcome, fid)
        else:
            samples = {"nnue_schema": np.asarray([1], dtype=np.int32),
                "centers": np.empty((0, 3), np.int32), "candidate_codes": np.empty((0, 3), np.int32),
                "candidate_coords": np.empty((0, 2), np.int64), "pairs": np.empty((0, 4), np.float32),
                "center_offsets": np.zeros(1, np.int64), "candidate_offsets": np.zeros(1, np.int64),
                "phase": np.empty((0, 4), np.float32)}
            for key, dtype in (("player", np.int64), ("baseline", np.float32), ("chosen", np.int64),
                ("search", np.float32), ("search_valid", bool), ("policy_valid", bool), ("outcome", np.float32),
                ("family", np.uint32), ("ply", np.int32), ("search_depth", np.int16), ("search_ms", np.float32)):
                samples[key] = np.empty(0, dtype=dtype)
        record = {"seed": task["seed"], "family": fid, "winner": winners[i],
            "reason": "six-in-a-row" if winners[i] >= 0 else "truncated",
            "cells": [[int(q), int(r), int(p)] for (q, r), p in zip(coords[i, :lengths[i]], owners[i, :lengths[i]])],
            "times": [[], []], "depths": [], "tables": task["tables"], "model_kind": "nnue",
            "challenger_color": task.get("challenger_color"), "actor": "gpu-nnue-turn-beam",
            "teacher": "conditional-complete-turn-value-beam", "actor_beam": [beam, beam],
            "actor_candidates": candidates, "actor_epsilon": epsilon, "actor_batch_seed": seed,
            "actor_batch_seconds": elapsed, "opening": openings[i], "prefix_length": len(openings[i]),
            "curriculum": task.get("curriculum", "legacy")}
        results.append((record, samples))
    return results


def recommended_batch(candidates=32, max_stones=256, device="cuda", beam=4):
    """Reserve half the currently free VRAM for center lookup/beam temporaries."""
    if candidates < 1 or max_stones < 3 or beam < 1:
        raise ValueError("Invalid NNUE actor batch dimensions")
    if torch.device(device).type != "cuda":
        return 8
    free, _ = torch.cuda.mem_get_info(device)
    capacity = max(128, 1 << (max_stones-1).bit_length())
    per_game = capacity*(31*beam*64+1024+candidates*64)+131072
    count = max(1, min(512, int(max(0, free*.5-32*1024**2)/per_game)))
    return max(1, count//16*16) if count >= 16 else count
