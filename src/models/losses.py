"""Regime-aware logit-adjusted BCE for single-head fire segmentation.

Splits the per-pixel loss by regime — 1 = ignition (no fire nearby at t), 2 = spread
(fire nearby at t), 0 = sea/invalid (ignored) — applies a PER-REGIME logit adjustment
(Menon et al. 2020, "Long-tail learning via logit adjustment") using each regime's own
class prior, and combines:

    loss = alpha * L_ignition + (1 - alpha) * L_spread

Two orthogonal knobs:
  * the per-regime **logit adjustment** corrects each regime's positive/negative imbalance
    (ignition positives are far rarer than spread, so they get a stronger negative shift);
  * **alpha** is the gradient-budget lean toward the hard ignition regime (recommend ~0.6 —
    a gentle lean; not the imbalance correction).

Kept as logit-adjusted BCE (not focal) to change one thing at a time vs. the prior pipeline.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

IGNITION, SPREAD = 1, 2  # regime codes (0 = sea/invalid)


def logit_adjustment_from_counts(pos: int, total: int, eps: float = 1e-8) -> float:
    """log(pi_pos / pi_neg) for a class prior estimated from counts."""
    pi_pos = max(pos / max(total, 1), eps)
    pi_neg = max(1.0 - pi_pos, eps)
    return math.log(pi_pos / pi_neg)


def compute_regime_priors(loader) -> Tuple[float, float, dict]:
    """Scan a (X, y, regime) loader once; return (adj_ignition, adj_spread, info).

    info carries the raw positive/total counts + positive rates per regime (for logging).
    """
    counts = {IGNITION: [0, 0], SPREAD: [0, 0]}  # code -> [pos, total]
    for _, y, regime in loader:
        for code in (IGNITION, SPREAD):
            m = regime == code
            counts[code][1] += int(m.sum().item())
            counts[code][0] += int(y[m].sum().item())
    adj_ign = logit_adjustment_from_counts(*counts[IGNITION])
    adj_spr = logit_adjustment_from_counts(*counts[SPREAD])
    info = {
        "ignition_pos": counts[IGNITION][0], "ignition_total": counts[IGNITION][1],
        "spread_pos": counts[SPREAD][0], "spread_total": counts[SPREAD][1],
        "ignition_pos_rate": counts[IGNITION][0] / max(counts[IGNITION][1], 1),
        "spread_pos_rate": counts[SPREAD][0] / max(counts[SPREAD][1], 1),
        "adj_ignition": adj_ign, "adj_spread": adj_spr,
    }
    return adj_ign, adj_spr, info


class RegimeLogitAdjustedBCE(nn.Module):
    """alpha * L_ignition + (1-alpha) * L_spread, each a per-regime logit-adjusted BCE.

    Args:
        alpha: weight on the ignition regime's loss (0..1; ~0.6 recommended).
        adj_ignition, adj_spread: per-regime logit adjustments (from compute_regime_priors).
    """

    def __init__(self, alpha: float, adj_ignition: float, adj_spread: float):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = float(alpha)
        self.register_buffer("adj_ign", torch.tensor(float(adj_ignition)))
        self.register_buffer("adj_spread", torch.tensor(float(adj_spread)))
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets, regime):
        """logits/targets/regime: [B,1,H,W]; regime int in {0,1,2}. Returns (loss, components)."""
        targets = targets.float()
        ign = regime == IGNITION
        spr = regime == SPREAD
        adj = torch.zeros_like(logits)
        adj = torch.where(ign, self.adj_ign.to(logits.dtype), adj)
        adj = torch.where(spr, self.adj_spread.to(logits.dtype), adj)
        per_px = self.bce(logits + adj, targets)

        def regime_mean(mask):
            m = mask.to(per_px.dtype)
            denom = m.sum()
            if denom <= 0:
                return logits.new_tensor(0.0)
            return (per_px * m).sum() / denom

        l_ign = regime_mean(ign)
        l_spr = regime_mean(spr)
        loss = self.alpha * l_ign + (1.0 - self.alpha) * l_spr
        return loss, {"L_ignition": l_ign.detach(), "L_spread": l_spr.detach()}
