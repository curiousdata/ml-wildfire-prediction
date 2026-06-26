"""Diagnostic: can the GroupNorm U-Net OVERFIT a tiny batch of fire days?

If even a handful of fire-containing days can't be memorized (train-AP stays ~0, fire
cells not ranked above no-fire), the loss/gradient is broken — most likely the mean
reduction over overwhelmingly-negative pixels diluting the positive gradient, which the
logit adjustment shifts the boundary for but does NOT un-dilute. We A/B a few recipes so
the test also points at the fix.

Recipes: current (logit-adj BCE, lr 3e-5) | higher lr | + focal | tempered adjustment.
Reports, per recipe: train overall/new-ign/spread AP + mean predicted prob on fire vs
no-fire cells (the ranking-direction check that explains the ROC<0.5 we saw).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch
import xarray as xr
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.datasets import StackedRegimeIberFireDataset
from src.data.features import build_segmentation_features
from src.models.cnn import build_unet
from src.models.losses import RegimeLogitAdjustedBCE

CUBE = "data/gold/IberFire_coarse4.zarr"; STATS = "stats/coarse4_norm_stats_train.json"; DYN = "data/gold/IberFire_coarse4_dyn.zarr"
ADJ_IGN, ADJ_SPR, ALPHA = -10.90, -2.67, 0.6


def aps(prob, y, reg):
    land = reg.ravel() > 0; p = prob.ravel()[land]; t = y.ravel()[land]; r = reg.ravel()[land]
    out = {}
    out["overall"] = average_precision_score(t, p) if t.sum() else float("nan")
    out["roc"] = roc_auc_score(t, p) if 0 < t.sum() < t.size else float("nan")
    for nm, c in (("new_ign", 1), ("spread", 2)):
        pos = (t == 1) & (r == c); keep = pos | (t == 0)
        out[nm] = average_precision_score(t[keep], p[keep]) if pos.sum() else float("nan")
    out["meanp_fire"] = float(p[t == 1].mean()) if (t == 1).any() else float("nan")
    out["meanp_nofire"] = float(p[t == 0].mean())
    return out


def main():
    dev = torch.device("mps")
    feats = build_segmentation_features(xr.open_zarr(CUBE, consolidated=True).data_vars)
    ds = StackedRegimeIberFireDataset(dyn_zarr_path=DYN, zarr_path=CUBE, time_start="2008-01-01",
        time_end="2018-12-31", feature_vars=feats, label_var="is_fire", lead_time=1,
        compute_stats=False, stats_path=STATS, mode="all")
    # grab 8 fire-containing days
    picks = []
    for i in range(len(ds)):
        _, y, _ = ds[i]
        if y.sum() > 0:
            picks.append(i)
        if len(picks) >= 8:
            break
    Xs, ys, rs = zip(*[ds[i] for i in picks])
    X = torch.stack(Xs).to(dev).float(); y = torch.stack(ys).to(dev).float(); reg = torch.stack(rs).to(dev)
    C = X.shape[1]
    print(f"overfit batch: {X.shape}  fire pixels={int(y.sum())}  C={C}", flush=True)

    # Now test the REAL production loss (focal-weight-mass normalization) at lr candidates.
    # Goal: find an lr that still overfits (AP high) AND is stable (loss not NaN/exploding) at
    # the new loss scale — the ignition term is ~O(10) now (was ~1e-4), so lr must come down.
    recipes = [("focal-mass lr5e-5", 5e-5, 2.0), ("focal-mass lr2e-5", 2e-5, 2.0),
               ("focal-mass lr1e-5", 1e-5, 2.0), ("focal-mass lr5e-6", 5e-6, 2.0)]
    for name, lr, gamma in recipes:
        torch.manual_seed(0)
        m = build_unet(in_channels=C, encoder_name="resnet34", encoder_weights="imagenet",
                       decoder_dropout=0.0, norm="group").to(dev)
        loss_fn = RegimeLogitAdjustedBCE(ALPHA, ADJ_IGN, ADJ_SPR, focal_gamma=gamma).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=lr)
        m.train()
        l0 = lN = float("nan")
        for step in range(120):
            opt.zero_grad(); logit = m(X); loss, _ = loss_fn(logit, y, reg)
            if step == 0:
                l0 = float(loss.item())
            loss.backward(); opt.step()
        lN = float(loss.item())
        m.eval()
        with torch.no_grad():
            prob = torch.sigmoid(m(X)).cpu().numpy()
        a = aps(prob, y.cpu().numpy(), reg.cpu().numpy())
        print(f"  {name:18}: loss {l0:.3f}->{lN:.3f} | train overall AP={a['overall']:.3f} "
              f"new-ign={a['new_ign']:.3f} roc={a['roc']:.3f} | meanP fire={a['meanp_fire']:.3f} "
              f"nofire={a['meanp_nofire']:.3f}", flush=True)


if __name__ == "__main__":
    main()
