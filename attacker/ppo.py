"""Turn-PPO loss: three-level aggregation (§6).

  ① Segment  -> segment weights w_k spread A_t across tokens inside one action
  ② Turn     -> GAE gives A_t (§5); clip acts on the WHOLE turn (all-or-nothing)
  ③ Batch    -> normalize by N (episode count), a constant. NOT sum_i T_i.

Effective per-token advantage coefficient (§6.4):

    A_eff(i,t,j) = (1/N) * A_{i,t} * w_k(j) / n_k(j)
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch

# Default segment weights (§6.1). Training passes the values from cfg.ppo;
# keeping an equal default makes direct/test callers match the current recipe.
W = {"next_position": 0.5, "message": 0.5}


def segment_weighted_mean(values: torch.Tensor, seg_masks: Dict[str, torch.Tensor],
                          weights: Optional[Mapping[str, float]] = None) -> torch.Tensor:
    """Equal-length-independent aggregation shared by PPO, KL, and entropy."""
    weights = W if weights is None else weights
    present = {k: m for k, m in seg_masks.items() if m.any()}
    if not present:
        return values.sum() * 0.0
    Z = sum(weights[k] for k in present)
    if Z <= 0:
        raise ValueError("sum of present segment weights must be positive")
    result = values.new_zeros(())
    for k, mask in present.items():
        result = result + (weights[k] / Z) * values[mask].mean()
    return result


def turn_log_ratio(
    logp: torch.Tensor,          # [L_t] new log-probs at generated value tokens
    old_logp: torch.Tensor,      # [L_t] old log-probs (recomputed, §10.A)
    seg_masks: Dict[str, torch.Tensor],  # name -> BoolTensor[L_t]
    weights: Optional[Mapping[str, float]] = None,
) -> torch.Tensor:
    """Segment-weighted turn log-ratio D_t (§6.1).

        D_t = sum_k (w_k / Z) * mean_{j in seg k}(logp_j - old_logp_j)

    [TRAP] When a segment is missing (e.g. message failed to parse) the
    present weights are RE-NORMALIZED (Z = sum of present w_k), not treated as
    0. Otherwise D_t is systematically shrunk toward 0 and the turn is
    silently skipped.

    Only value tokens are passed in; JSON scaffolding must already be masked
    out by the caller ([TRAP] §6.1).
    """
    d = logp - old_logp
    return segment_weighted_mean(d, seg_masks, weights)


def ppo_turn_loss(
    D_t: torch.Tensor,   # scalar turn log-ratio
    A_t: float,          # scalar turn advantage (detached constant)
    clip_eps: float,
) -> torch.Tensor:
    """PPO clipped surrogate on the whole turn (§6.2).

    clip acts on the entire turn: when clipped, every token in the turn gets
    zero gradient (all-or-nothing, intended).
    """
    ratio = torch.exp(D_t)
    A = D_t.new_tensor(float(A_t))
    unclipped = ratio * A
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A
    return torch.minimum(unclipped, clipped)


def approx_kl_k3(D_t: torch.Tensor) -> torch.Tensor:
    """k3 estimator, per turn: (ratio - 1 - log ratio), non-negative (§7.d)."""
    ratio = torch.exp(D_t)
    return ratio - 1.0 - D_t


def is_clipped(D_t: torch.Tensor, A_t: float, clip_eps: float) -> bool:
    """True if this turn was clipped (for turn_clip_frac metric)."""
    r = float(torch.exp(D_t))
    if A_t >= 0:
        return r > 1.0 + clip_eps
    return r < 1.0 - clip_eps
