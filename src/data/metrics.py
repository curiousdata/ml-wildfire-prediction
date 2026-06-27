"""Shared, torch-free metrics + paths for the FGDC GBT pipeline.

Extracted from the legacy U-Net ``scripts/train.py`` (which imports torch + the whole U-Net stack) so the v2
trainers, calibrator, baseline panel, and serving can import ``regime_metrics`` / ``project_root`` WITHOUT
pulling torch onto the GBT/serving path (the v1 lesson: torch is off the live path).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

project_root = Path(__file__).resolve().parents[2]


def regime_metrics(prob, tgt, reg, neg_ratio: int = 15, seed: int = 0) -> dict:
    """Interpretable, GBT-comparable metrics over land cells (``reg > 0``).

      * ``roc``         : full-prevalence ROC-AUC (prevalence-independent honest ranker).
      * ``{regime}_ap`` : AP at MATCHED prevalence (negatives subsampled to ``neg_ratio:1``, same ``seed`` each
                          call → comparable across epochs / models). ``reg``: 1 = new-ignition, 2 = spread.
      * ``prec_at_k``   : R-precision (precision in the top-K cells, K = #fire) — "of the K we'd flag, how many burn".

    ``prob``/``tgt``/``reg`` are flat per-cell arrays for one eval set.
    """
    rng = np.random.default_rng(seed)
    land = reg > 0
    p, t, r = prob[land], tgt[land], reg[land]
    out = {"roc": float("nan"), "overall_ap": float("nan"),
           "new_ignition_ap": float("nan"), "spread_ap": float("nan"),
           "prec_at_k": float("nan"), "n_pos": int(t.sum())}
    if t.sum() == 0 or t.size == 0:
        return out
    try:
        out["roc"] = float(roc_auc_score(t, p))
    except ValueError:
        pass
    neg_p = p[t == 0]

    def matched_ap(pos_mask):
        n_pos = int(pos_mask.sum())
        if n_pos == 0 or neg_p.size == 0:
            return float("nan")
        k = min(neg_p.size, neg_ratio * n_pos)
        sel = rng.choice(neg_p.size, size=k, replace=False)
        pp = np.concatenate([p[pos_mask], neg_p[sel]])
        tt = np.concatenate([np.ones(n_pos), np.zeros(k)])
        return float(average_precision_score(tt, pp))

    out["overall_ap"] = matched_ap(t == 1)
    out["new_ignition_ap"] = matched_ap((t == 1) & (r == 1))
    out["spread_ap"] = matched_ap((t == 1) & (r == 2))
    k = int(t.sum())                                  # R-precision: top-K most-confident cells
    topk = np.argpartition(p, -k)[-k:]
    out["prec_at_k"] = float(t[topk].sum() / k)
    return out
