"""Centered eleven-cell NNUE, quantization-aware training and native export."""
import hashlib
import copy
import math
from pathlib import Path
import struct
import time

import numpy as np
import torch
from torch import nn

PATTERNS = 3**11
LUT_SCALE = 256
SCORE_SCALE = 6000
HEADER = struct.Struct("<8s9I2fQ")


class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mapping = nn.Sequential(nn.Linear(33, 64), nn.ReLU(), nn.Linear(64, 32), nn.Tanh())
        self.value = nn.Sequential(nn.Linear(68, 32), nn.ReLU(), nn.Linear(32, 1))
        self.policy = nn.Sequential(nn.Linear(104, 16), nn.ReLU(), nn.Linear(16, 1))
        self.register_buffer("powers", 3**torch.arange(11), persistent=False)
        self.register_buffer("perspective", torch.tensor([*range(32, 48), *range(16, 32),
                                                          *range(16), *range(48, 64)]), persistent=False)
        for head in (self.value, self.policy):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def embeddings(self, codes):
        """Canonical orientation enforces exact reversal and color identities.

        First 16 channels change sign on color swap; the other 16 do not.
        Straight-through rounding trains the same int16 table used by C++.
        """
        shape = codes.shape
        digits = codes.reshape(-1, 1).long() // self.powers % 3
        swapped = torch.where(digits == 0, 0, 3-digits)
        direct = torch.minimum((digits*self.powers).sum(1), (digits.flip(1)*self.powers).sum(1))
        other = torch.minimum((swapped*self.powers).sum(1), (swapped.flip(1)*self.powers).sum(1))
        canonical = torch.minimum(direct, other)
        unique, inverse = torch.unique(canonical, return_inverse=True)
        canonical_digits = unique[:, None] // self.powers % 3
        inputs = nn.functional.one_hot(canonical_digits, 3).float().flatten(1)
        mapped = self.mapping(inputs)[inverse]
        sign = torch.sign((other-direct).float())[:, None]
        mapped = torch.cat((mapped[:, :16]*sign, mapped[:, 16:]), 1)
        rounded = torch.round(mapped*LUT_SCALE)/LUT_SCALE
        result = mapped + (rounded-mapped).detach()
        return result.reshape(*shape, 32)

    @staticmethod
    def local(lines):
        summed = lines.sum(-2)
        return torch.cat((summed.relu(), (-summed).relu()), -1)

    def position_inputs(self, centers, center_owner, center_counts, phase, player, empty):
        local = self.local(centers) - self.local(empty.expand(3, -1))
        pool = torch.zeros((len(phase), 64), device=phase.device)
        pool.index_add_(0, center_owner, local)
        pool = pool / center_counts.clamp_min(1).sqrt()[:, None]
        pool = torch.where(player[:, None] == 0, pool, pool[:, self.perspective])
        return torch.cat((pool, phase), 1)

    def position_features(self, batch):
        """Value inputs without encoding or embedding policy candidates."""
        centers = batch["centers"]
        embedded = self.embeddings(torch.cat((centers.flatten(), centers.new_zeros(1))))
        return self.position_inputs(embedded[:-1].reshape(-1, 3, 32), batch["center_owner"],
                                    batch["center_counts"], batch["phase"], batch["player"], embedded[-1])

    def features(self, batch):
        """Shared position/candidate features; support padded or flat ragged actions."""
        centers = batch["centers"]
        candidates = batch["candidate_codes"]
        all_codes = torch.cat((centers.flatten(), candidates.flatten(), centers.new_zeros(1)))
        embedded = self.embeddings(all_codes)
        split = centers.numel()
        center_emb = embedded[:split].reshape(-1, 3, 32)
        candidate_emb = embedded[split:-1].reshape(*candidates.shape, 32)
        inputs = self.position_inputs(center_emb, batch["center_owner"], batch["center_counts"],
                                      batch["phase"], batch["player"], embedded[-1])
        local = candidate_emb.sum(-2)
        if candidates.ndim == 2:
            owner = batch["candidate_owner"]
            sign = torch.where(batch["player"][owner] == 0, 1, -1).to(local.dtype)
            local = torch.cat((local[:, :16]*sign[:, None], local[:, 16:]), -1)
            policy_input = torch.cat((inputs[owner, :64], local, batch["phase"][owner], batch["pairs"]), -1)
        else:
            sign = torch.where(batch["player"] == 0, 1, -1).to(local.dtype)
            local = torch.cat((local[..., :16]*sign[:, None, None], local[..., 16:]), -1)
            policy_input = torch.cat((inputs[:, None, :64].expand(-1, local.shape[1], -1), local,
                                      batch["phase"][:, None].expand(-1, local.shape[1], -1), batch["pairs"]), -1)
        return inputs, policy_input

    def forward(self, batch):
        inputs, policy_input = self.features(batch)
        residual = self.value(inputs).squeeze(1)
        value = torch.tanh(batch["baseline"]/SCORE_SCALE + residual)
        logits = self.policy(policy_input).squeeze(-1).masked_fill(~batch["candidate_mask"], -float("inf"))
        return value, logits

    @torch.no_grad()
    def export(self, path):
        self.eval()
        device = next(self.parameters()).device
        rows = []
        for start in range(0, PATTERNS, 4096):
            codes = torch.arange(start, min(start+4096, PATTERNS), device=device)
            rows.append(torch.round(self.embeddings(codes)*LUT_SCALE).to(torch.int16).cpu().numpy())
        table = np.concatenate(rows).astype("<i2")
        codes = np.arange(PATTERNS)
        powers = 3**np.arange(11)
        digits = codes[:, None]//powers % 3
        swap = (np.where(digits == 0, 0, 3-digits)*powers).sum(1)
        reverse = (digits[:, ::-1]*powers).sum(1)
        if not (np.array_equal(table, table[reverse]) and
                np.array_equal(table[:, :16], -table[swap, :16]) and
                np.array_equal(table[:, 16:], table[swap, 16:]) and np.abs(table).max() <= LUT_SCALE):
            raise ValueError("NNUE table failed symmetry or quantization checks")
        payload = bytearray(table.tobytes())
        for head in (self.value, self.policy):
            for layer in (head[0], head[2]):
                for parameter in (layer.weight, layer.bias):
                    array = parameter.detach().cpu().numpy().astype("<f4")
                    if not np.isfinite(array).all():
                        raise ValueError("Non-finite NNUE head")
                    payload.extend(array.tobytes())
        header = HEADER.pack(b"HXNNUE1\0", 1, 0x01020304, PATTERNS, 32, 64, 4, 4, 32, 16,
                             LUT_SCALE, SCORE_SCALE, len(payload))
        content = header + payload
        Path(path).write_bytes(content)
        return hashlib.sha256(content).hexdigest()


def collate(replay, ids, device):
    """Collate ragged sparse centers and variable candidate lists on the CPU."""
    ids = np.asarray(ids, dtype=np.int64)
    center_counts = replay["center_offsets"][ids+1]-replay["center_offsets"][ids]
    counts = replay["candidate_offsets"][ids+1]-replay["candidate_offsets"][ids]
    width = int(counts.max())
    candidates = np.zeros((len(ids), width, 3), dtype=np.int32)
    pairs = np.zeros((len(ids), width, 4), dtype=np.float32)
    mask = np.zeros((len(ids), width), dtype=bool)
    centers = []
    for row, index in enumerate(ids):
        centers.append(replay["centers"][replay["center_offsets"][index]:replay["center_offsets"][index+1]])
        lo, hi = replay["candidate_offsets"][index:index+2]
        candidates[row, :hi-lo] = replay["candidate_codes"][lo:hi]
        pairs[row, :hi-lo] = replay["pairs"][lo:hi]
        mask[row, :hi-lo] = True
    arrays = {key: replay[key][ids] for key in ("phase", "player", "baseline", "search", "search_valid",
                                               "outcome", "policy_valid", "chosen")}
    arrays.update(centers=np.concatenate(centers), center_owner=np.repeat(np.arange(len(ids)), center_counts),
                  center_counts=center_counts.astype(np.float32), candidate_codes=candidates,
                  pairs=pairs, candidate_mask=mask)
    return {key: torch.as_tensor(value, device=device) for key, value in arrays.items()}


def objective(model, batch, policy_weight):
    value, logits = model(batch)
    outcome_known = torch.isfinite(batch["outcome"])
    z = batch["outcome"].nan_to_num()
    result_weight = outcome_known.float()*.75
    search_weight = torch.where(outcome_known, .25, 1.)*torch.where(batch["search_valid"], 1., .1)
    value_loss = ((value-z).square()*result_weight + (value-batch["search"]).square()*search_weight)
    # With no outcome, a depth-zero fallback remains a low-confidence target.
    # Dividing by its own .1 weight would otherwise cancel that downweighting.
    normalizer = torch.where(outcome_known, result_weight+search_weight, 1.)
    value_loss = (value_loss/normalizer).mean()
    labels = batch["policy_valid"].float()
    ce = nn.functional.cross_entropy(logits, batch["chosen"].long(), reduction="none")
    policy_loss = (ce*labels).sum()/labels.sum().clamp_min(1)
    return value_loss + policy_weight*policy_loss, value_loss.detach(), policy_loss.detach()


def load_replay(paths, args, seed):
    from legacy.train import merge_nnue
    counts, center_counts = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            if "nnue_schema" not in saved or saved["nnue_schema"].tolist() != [1]:
                raise ValueError(f"NNUE requires centered-line replay schema 1: {path}")
            counts.append(len(saved["family"]))
            center_counts.append(np.diff(saved["center_offsets"]))
    available = sum(counts)
    if not available:
        raise ValueError("NNUE replay has no positions")
    rng = np.random.default_rng(seed)
    order = rng.permutation(available)[:args.replay_positions]
    sizes = np.concatenate(center_counts)[order]
    # A random prefix bounds replay storage without preferring short positions.
    keep = int(np.searchsorted(np.cumsum(sizes), args.nnue_replay_centers, side="right"))
    if not keep:
        raise ValueError("A complete position exceeds --nnue-replay-centers; increase the limit")
    chosen = np.sort(order[:keep])
    parts, offset = [], 0
    for path, count in zip(paths, counts):
        ids = chosen[np.searchsorted(chosen, offset):np.searchsorted(chosen, offset+count)]-offset
        offset += count
        if not len(ids):
            continue
        with np.load(path, allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
        part = {"nnue_schema": np.asarray([1], dtype=np.int32)}
        ragged = {"centers", "candidate_codes", "pairs", "candidate_coords", "center_offsets", "candidate_offsets", "nnue_schema"}
        for key in data.keys()-ragged:
            part[key] = data[key][ids]
        for offsets, keys in (("center_offsets", ("centers",)),
                              ("candidate_offsets", ("candidate_codes", "pairs", "candidate_coords"))):
            spans = [slice(*data[offsets][i:i+2]) for i in ids]
            part[offsets] = np.concatenate(([0], np.cumsum([s.stop-s.start for s in spans]))).astype(np.int64)
            for key in keys:
                part[key] = np.concatenate([data[key][s] for s in spans])
        parts.append(part)
        del data
    return merge_nnue(parts), available


def batches_for(ids, replay, positions, center_limit):
    sizes = np.diff(replay["center_offsets"])
    current, centers = [], 0
    for index in ids:
        if sizes[index] > center_limit:
            raise ValueError("A complete position exceeds --nnue-batch-centers; increase the limit")
        if current and (len(current) >= positions or centers+sizes[index] > center_limit):
            yield np.asarray(current)
            current, centers = [], 0
        current.append(index)
        centers += sizes[index]
    if current:
        yield np.asarray(current)


def optimize_nnue(run, checkpoint, incumbent, replay_paths, args, progress, log):
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed+checkpoint)
    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    if args.device == "cuda" and device != "cuda":
        raise RuntimeError("CUDA requested but unavailable")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    from legacy.reanalysis import replay_inputs
    paths, replay_metadata = replay_inputs(run, replay_paths, args)
    replay, available = load_replay(paths, args, args.seed+checkpoint)
    train_ids = np.flatnonzero(replay["family"] % 5 != 0)
    val_ids = np.flatnonzero(replay["family"] % 5 == 0)
    if not len(train_ids):
        raise ValueError("No training opening families; collect more games")
    model = NNUE().to(device)
    model.load_state_dict(torch.load(run / incumbent["model"], map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.001, fused=device == "cuda")
    if incumbent.get("optimizer"):
        optimizer.load_state_dict(torch.load(run / incumbent["optimizer"], map_location=device, weights_only=True))
        for group in optimizer.param_groups:
            group["fused"] = device == "cuda"
    batch_size = args.batch or 256
    generator = np.random.default_rng(args.seed+checkpoint)
    best, best_state, best_optimizer, best_metrics = math.inf, None, None, None
    processed, updates = 0, 0
    for epoch in range(args.epochs):
        model.train()
        batches = list(batches_for(generator.permutation(train_ids), replay, batch_size, args.nnue_batch_centers))
        steps = args.updates_per_epoch or len(batches)
        totals = torch.zeros(3, device=device)
        seen = 0
        for step in range(steps):
            ids = batches[step % len(batches)]
            batch = collate(replay, ids, device)
            loss, value_loss, policy_loss = objective(model, batch, args.policy_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            totals += torch.stack((loss.detach(), value_loss, policy_loss))*len(ids)
            seen += len(ids)
        processed += seen
        updates += steps
        train_loss, value_loss, policy_loss = (totals/seen).tolist()
        model.eval()
        with torch.no_grad():
            total = torch.zeros((), device=device)
            for ids in batches_for(val_ids, replay, batch_size, args.nnue_batch_centers):
                total += objective(model, collate(replay, ids, device), args.policy_weight)[0]*len(ids)
            validation = float(total/len(val_ids)) if len(val_ids) else None
        selected_loss = validation if validation is not None else train_loss
        if not math.isfinite(train_loss) or not math.isfinite(selected_loss):
            raise FloatingPointError("Non-finite NNUE loss; model not exported")
        metrics = {"iteration": checkpoint, "epoch": epoch+1, "train_loss": train_loss,
                   "value_loss": value_loss, "policy_loss": policy_loss, "validation_loss": validation,
                   "train_positions": len(train_ids), "validation_positions": len(val_ids),
                   "device": device, "batch": batch_size, "replay_positions_available": available,
                   "replay_positions_selected": len(replay["family"]), "replay_centers": len(replay["centers"]),
                   "policy_positions": int(replay["policy_valid"].sum()), "optimizer_steps": updates,
                   "examples_processed": processed, "positions_per_second": processed/(time.perf_counter()-started),
                   "gpu_memory_peak_mb": torch.cuda.max_memory_allocated()/2**20 if device == "cuda" else 0,
                   "model_kind": "nnue"}
        if selected_loss < best:
            best, best_metrics = selected_loss, metrics
            best_state, best_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        log(**metrics)
        if progress:
            progress(epoch+1, args.epochs, metrics)
        print(f"Epoch {epoch+1}/{args.epochs}: NNUE value {value_loss:.4f}, policy {policy_loss:.4f}, validation {validation}", flush=True)
    model.load_state_dict(best_state)
    directory = run / "checkpoints" / f"{checkpoint:04d}"
    directory.mkdir(exist_ok=True)
    torch.save(model.state_dict(), directory / "model.pt")
    torch.save(best_optimizer, directory / "optimizer.pt")
    digest = model.export(directory / "model.nnue")
    return {"id": checkpoint, "kind": "nnue", "model": str((directory / "model.pt").relative_to(run)),
            "nnue": str((directory / "model.nnue").relative_to(run)),
            "optimizer": str((directory / "optimizer.pt").relative_to(run)), "learner_parent": incumbent["id"],
            "loss": best_metrics, "selected_epoch": best_metrics["epoch"],
            "selection": "validation" if len(val_ids) else "training-only",
            "training_seconds": time.perf_counter()-started, "model_sha256": digest,
            "replay": replay_metadata}
