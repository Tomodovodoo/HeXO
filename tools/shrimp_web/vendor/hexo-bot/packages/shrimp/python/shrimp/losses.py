"""Losses and 65-bin helpers.

- Policy CE is a segment soft cross-entropy over each row's legal prefix
  (scatter-logsumexp, fp32). Legality masking is structural: the logit support
  is the legal set. Target mass off the legal prefix raises.
- Loss reduction is mean over rows. Every reduction accepts an optional explicit
  denominator, so gradient accumulation over micro-buckets with step-global
  denominators matches a monolithic batch.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F

from .constants import MOVES_LEFT_CAP, VALUE_BINS

POLICY_WEIGHT = 1.0
VALUE_WEIGHT = 1.0
OPP_POLICY_WEIGHT = 0.25
# Auxiliary soft-policy target loss weight. The soft target is built in
# batching.py. Mirrored by config.TrainingSection.soft_policy_weight.
SOFT_POLICY_WEIGHT = 4.0
SHORT_TERM_VALUE_WEIGHT = 0.1
MOVES_LEFT_WEIGHT = 0.1
Q_HEAD_WEIGHT = 0.1


def _at_least_fp32(x: torch.Tensor) -> torch.Tensor:
    """fp32 floor for the loss math: upcast float16/bfloat16, pass fp32/fp64
    through unchanged."""

    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
    return x


def value_bins(*, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    """The VALUE_BINS scalar support points, linspace(-1, 1), shared by every
    binned head."""

    return torch.linspace(-1.0, 1.0, VALUE_BINS, device=device, dtype=dtype)


def decode_binned_value(logits: torch.Tensor) -> torch.Tensor:
    """Softmax expectation over the bins, clamped to [-1, 1]."""

    bins = value_bins(device=logits.device, dtype=logits.dtype)
    return ((torch.softmax(logits, dim=-1) * bins).sum(dim=-1)).clamp(-1.0, 1.0)


def decode_moves_left(logits: torch.Tensor) -> torch.Tensor:
    """Softmax expectation over the bins, clamped to [-1, 1], then affine-mapped
    to decisions in [0, MOVES_LEFT_CAP]."""

    bins = value_bins(device=logits.device, dtype=logits.dtype)
    scalar = (torch.softmax(logits, dim=-1) * bins).sum(dim=-1).clamp(-1.0, 1.0)
    return (scalar + 1.0) * 0.5 * MOVES_LEFT_CAP


def scalar_to_binned_target(values: torch.Tensor | float) -> torch.Tensor:
    """Scalars in [-1, 1] -> adjacent-bin soft targets."""

    target = torch.as_tensor(values)
    if not bool(torch.isfinite(target).all().item()):
        raise ValueError("value targets must be finite")
    if bool(((target < -1.0) | (target > 1.0)).any().item()):
        raise ValueError("value targets must be in [-1, 1]")
    original_shape = target.shape
    flat = target.reshape(-1)
    position = (flat + 1.0) * ((VALUE_BINS - 1) / 2.0)
    lower = torch.floor(position).to(dtype=torch.long)
    upper = torch.ceil(position).to(dtype=torch.long)
    upper_weight = position - lower.to(dtype=position.dtype)
    lower_weight = 1.0 - upper_weight
    out = torch.zeros((flat.numel(), VALUE_BINS), device=flat.device, dtype=target.dtype)
    rows = torch.arange(flat.numel(), device=flat.device)
    out[rows, lower] += lower_weight
    out[rows, upper] += upper_weight
    return out.reshape(*original_shape, VALUE_BINS)


def segment_policy_ce(
    logits: torch.Tensor,
    legal_counts: torch.Tensor,
    target: torch.Tensor,
    *,
    allow_zero_rows: bool = False,
    denominator: float | None = None,
    row_weight: torch.Tensor | None = None,
    weight_denominator: float | None = None,
) -> torch.Tensor:
    """Soft CE over each row's legal prefix; mean over rows.

    logits/target: (B, Npad); per row g only slots [0, L_g) participate, where
    L_g = legal_counts[g]. Target mass outside the prefix raises. With
    ``allow_zero_rows``, zero-mass rows contribute 0 but stay in the denominator;
    otherwise a zero-mass row raises.
    """

    if logits.shape != target.shape:
        raise ValueError(
            f"policy target shape {tuple(target.shape)} != logits {tuple(logits.shape)}"
        )
    b, npad = logits.shape
    if bool((legal_counts <= 0).any().item()):
        raise ValueError("policy rows must have at least one legal move")
    prefix = torch.arange(npad, device=logits.device).unsqueeze(0) < legal_counts.unsqueeze(1)
    target = target.to(device=logits.device)
    if not bool(torch.isfinite(target).all().item()):
        raise ValueError("policy targets must be finite")
    if bool((target < 0).any().item()):
        raise ValueError("policy targets must be nonnegative")
    if bool((target[~prefix] > 0).any().item()):
        raise ValueError("policy target mass off the legal prefix is a hard error")

    row_sum = target.sum(dim=-1)
    positive = row_sum > 0
    if not allow_zero_rows and not bool(positive.all().item()):
        raise ValueError("policy targets must contain positive probability mass")

    flat_logits = _at_least_fp32(logits[prefix])
    flat_target = _at_least_fp32(target[prefix])
    row_ids = prefix.nonzero(as_tuple=True)[0]

    # Per-row logsumexp: row max, then sum of exp(logit - max) over the prefix.
    row_max = torch.full((b,), float("-inf"), device=logits.device, dtype=flat_logits.dtype)
    row_max = row_max.scatter_reduce(0, row_ids, flat_logits, reduce="amax")
    shifted = (flat_logits - row_max[row_ids]).exp()
    row_expsum = torch.zeros(b, device=logits.device, dtype=flat_logits.dtype)
    row_expsum = row_expsum.index_add(0, row_ids, shifted)
    lse = row_max + row_expsum.log()

    log_probs = flat_logits - lse[row_ids]
    normalizer = _at_least_fp32(torch.where(positive, row_sum, torch.ones_like(row_sum)))
    weighted = flat_target / normalizer[row_ids] * log_probs
    per_row = torch.zeros(b, device=logits.device, dtype=flat_logits.dtype)
    per_row = per_row.index_add(0, row_ids, weighted).neg()

    if row_weight is not None:
        per_row = per_row * row_weight.to(device=per_row.device, dtype=per_row.dtype)
        denom = float(b) if weight_denominator is None else float(weight_denominator)
    else:
        denom = float(b) if denominator is None else float(denominator)
    if denom <= 0.0:
        # Denominator of 0: return 0 without dividing.
        return logits.sum() * 0.0
    return per_row.sum() / denom


def binned_value_loss(
    logits: torch.Tensor,
    target: torch.Tensor | float,
    *,
    mask: torch.Tensor | None = None,
    denominator: float | None = None,
) -> torch.Tensor:
    """CE against scalar or distributional 65-bin targets.

    Scalar targets (shape != logits) are converted via scalar_to_binned_target.
    Masked rows contribute 0. The denominator defaults to the masked row count
    (or the item count when unmasked) and is overridable."""

    # Targets stay fp32 end-to-end: under train autocast ``logits.dtype`` is fp16,
    # which would quantize continuous scalar targets on entry AND compute the
    # two-hot position (in [0, 64], fp16 ulp ~1/32 near the top) in fp16,
    # mis-splitting adjacent-bin weights by up to ~3% of a bin (short-term
    # values, cell_q, moves_left). The CE below already lifts logits to >= fp32
    # via ``_at_least_fp32``, so an fp32 target costs nothing.
    target_tensor = torch.as_tensor(target, device=logits.device, dtype=torch.float32)
    if target_tensor.shape != logits.shape:
        target_tensor = scalar_to_binned_target(target_tensor).to(device=logits.device)
    if logits.shape != target_tensor.shape:
        raise ValueError(
            f"value target shape {tuple(target_tensor.shape)} != logits {tuple(logits.shape)}"
        )
    if not bool(torch.isfinite(target_tensor).all().item()):
        raise ValueError("value distribution targets must be finite")
    if bool((target_tensor < 0).any().item()):
        raise ValueError("value distribution targets must be nonnegative")
    target_sum = target_tensor.sum(dim=-1, keepdim=True)
    if not bool((target_sum > 0).all().item()):
        raise ValueError("value distribution targets must contain positive probability mass")
    target_tensor = target_tensor / target_sum
    per_item = -(target_tensor * F.log_softmax(_at_least_fp32(logits), dim=-1)).sum(dim=-1)
    if mask is None:
        denom = float(per_item.numel()) if denominator is None else float(denominator)
        return per_item.sum() / denom
    mask_tensor = torch.as_tensor(mask, device=logits.device, dtype=per_item.dtype)
    while mask_tensor.ndim < per_item.ndim:
        mask_tensor = mask_tensor.unsqueeze(-1)
    mask_tensor = mask_tensor.expand_as(per_item)
    denom = float(mask_tensor.sum().item()) if denominator is None else float(denominator)
    if denom <= 0.0:
        return logits.sum() * 0.0
    return (per_item * mask_tensor).sum() / denom


def shrimp_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    policy_weight: float = POLICY_WEIGHT,
    value_weight: float = VALUE_WEIGHT,
    opp_policy_weight: float = OPP_POLICY_WEIGHT,
    soft_policy_weight: float = SOFT_POLICY_WEIGHT,
    short_term_value_weight: float = SHORT_TERM_VALUE_WEIGHT,
    moves_left_weight: float = MOVES_LEFT_WEIGHT,
    q_head_weight: float = Q_HEAD_WEIGHT,
    policy_target: str = "visit",
    denominators: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted sum of the per-head losses; returns (total, components) where
    components maps each head name (plus "total") to its scalar loss.

    Head weights default to the module constants: policy, value,
    opp_policy, soft_policy, short_term_value (each stvalue_<h>), moves_left,
    q_head (cell_q).

    ``denominators`` supplies step-global row counts. Recognized keys: ``rows``,
    ``policy_ce_weight_sum``, ``policy_rows``, and per-masked-head counts
    (``value``, ``stvalue_<h>``, ``moves_left``, ``cell_q``). When absent, each
    reduction uses this batch's own counts. The ``value`` count excludes
    value_mask==0 rows from both numerator and denominator.
    """

    denoms = dict(denominators or {})
    rows = denoms.get("rows")
    components: dict[str, torch.Tensor] = {}

    # policy_valid==0 rows carry all-zero policy targets. policy_valid is
    # multiplied into the per-row CE weight so those rows contribute 0;
    # allow_zero_rows lets their zero-mass targets pass. weight_denominator is the
    # full-row weight sum (policy_ce_weight_sum), so the reduction is a mean over
    # policy_valid==1 rows.
    _pol_weight = batch.get("policy_ce_weight")
    _pv = batch.get("policy_valid")
    if _pol_weight is not None and _pv is not None:
        _pol_weight = _pol_weight * _pv
    # Select the main-policy CE target per row. When policy_target == "gumbel"
    # and both "gumbel_policy" and "gumbel_policy_valid" are present, rows with
    # gumbel_policy_valid > 0 use the gumbel_policy target; all other rows use the
    # "policy" (visit) target. Computed as a per-row blend, so mixed batches and
    # visit-only batches are both handled.
    _policy_target = batch["policy"]
    if (
        policy_target == "gumbel"
        and "gumbel_policy" in batch
        and "gumbel_policy_valid" in batch
    ):
        _use_gumbel = (batch["gumbel_policy_valid"] > 0.0).to(_policy_target.dtype)
        _use_gumbel = _use_gumbel.unsqueeze(1)  # (B,1) broadcast over the legal prefix
        _policy_target = (
            _use_gumbel * batch["gumbel_policy"] + (1.0 - _use_gumbel) * batch["policy"]
        )
    components["policy"] = segment_policy_ce(
        outputs["policy"],
        batch["legal_counts"],
        _policy_target,
        allow_zero_rows=True,
        row_weight=_pol_weight,
        weight_denominator=denoms.get("policy_ce_weight_sum"),
        denominator=rows,
    )
    # value_mask==0 rows are excluded from the value head. The denominator uses
    # denoms['value'] if present, else `rows`, else the masked row count inside
    # binned_value_loss.
    components["value"] = binned_value_loss(
        outputs["value"],
        batch["value"],
        mask=batch.get("value_mask"),
        denominator=denoms.get("value", rows),
    )
    total = policy_weight * components["policy"] + value_weight * components["value"]

    if "opp_policy" in outputs and "opp_policy" in batch:
        # allow_zero_rows keeps zero-target rows in the denominator while they
        # contribute 0. row_weight=policy_valid zeroes policy_valid==0 rows in the
        # numerator; weight_denominator=policy_rows makes the reduction a mean over
        # policy_valid==1 rows.
        components["opp_policy"] = segment_policy_ce(
            outputs["opp_policy"],
            batch["legal_counts"],
            batch["opp_policy"],
            allow_zero_rows=True,
            row_weight=batch.get("policy_valid"),
            weight_denominator=denoms.get("policy_rows", rows),
            denominator=denoms.get("policy_rows", rows),
        )
        total = total + opp_policy_weight * components["opp_policy"]

    if "soft_policy" in outputs and "soft_policy" in batch:
        # CE of the soft-policy head logits against the "soft_policy" target
        # (built in collate_training). No row_weight, so the denominator is a flat
        # row count. policy_valid==0 rows carry all-zero soft_policy;
        # allow_zero_rows lets them contribute 0. Denominator is policy_rows.
        components["soft_policy"] = segment_policy_ce(
            outputs["soft_policy"],
            batch["legal_counts"],
            batch["soft_policy"],
            allow_zero_rows=True,
            denominator=denoms.get("policy_rows", rows),
        )
        total = total + soft_policy_weight * components["soft_policy"]

    for key, output in outputs.items():
        if key.startswith("stvalue_") and key in batch:
            components[key] = binned_value_loss(
                output,
                batch[key],
                mask=batch.get(f"{key}_mask"),
                denominator=denoms.get(key),
            )
            total = total + short_term_value_weight * components[key]

    if "moves_left" in outputs and "moves_left" in batch:
        components["moves_left"] = binned_value_loss(
            outputs["moves_left"],
            batch["moves_left"],
            mask=batch.get("moves_left_mask"),
            denominator=denoms.get("moves_left"),
        )
        total = total + moves_left_weight * components["moves_left"]

    if "cell_q" in outputs and "cell_q" in batch:
        components["cell_q"] = binned_value_loss(
            outputs["cell_q"],
            batch["cell_q"],
            mask=batch["cell_q_mask"],
            denominator=denoms.get("cell_q"),
        )
        total = total + q_head_weight * components["cell_q"]

    components["total"] = total
    return total, components
