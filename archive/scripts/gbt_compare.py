"""GBT benchmark on the EXACT same split/features/eval as the U-Net — directly comparable.

The measurement_floor GBT (new-ign ~0.50) used its own subsampled eval, NOT identical to the U-Net's
`regime_metrics`. This script removes every methodological difference: it reuses the U-Net's dataset
(same 146 features, same normalization, same regime codes), trains a point-wise HistGBT on the train
split, predicts per-cell on the SAME strided val/test days + land cells the U-Net evaluates on, and scores
with train.py's `regime_metrics`. The resulting GBT numbers are produced by the identical function on
identical cells → apples-to-apples with v4/v5/v6.

Point-wise GBT vs spatial U-Net on the same yardstick isolates exactly what spatial context buys.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.train as T
from src.data.features import build_segmentation_features

MAX_TRAIN_ROWS = 400_000   # subsample train cells (all positives + negatives) for tractable GBT fit
NEG_PER_POS = 15           # training imbalance (eval prevalence is matched separately in regime_metrics)
TRAIN_DAY_STRIDE = 3       # subsample train days (4018 -> ~1340) before per-cell sampling


def collect_cells(ds, day_indices, all_land=False, rng=None):
    """Flatten (X,y,reg) over the given days into per-cell rows.
    all_land=True keeps every land cell (for EVAL, matches the U-Net). Otherwise sample positives +
    NEG_PER_POS negatives per day (for TRAIN)."""
    Xs, ys, rs = [], [], []
    for i in day_indices:
        X, y, reg = ds[i]
        X = X.numpy(); y = y.numpy().ravel(); reg = reg.numpy().ravel()
        C = X.shape[0]
        Xf = X.reshape(C, -1).T            # [H*W, C]
        land = reg > 0
        if all_land:
            keep = land
        else:
            pos = land & (y == 1)
            neg = land & (y == 0)
            negidx = np.where(neg)[0]
            k = min(negidx.size, NEG_PER_POS * max(int(pos.sum()), 1))
            sel = rng.choice(negidx, size=k, replace=False) if k else np.array([], int)
            keepidx = np.concatenate([np.where(pos)[0], sel])
            keep = np.zeros_like(land); keep[keepidx] = True
        Xs.append(Xf[keep]); ys.append(y[keep]); rs.append(reg[keep])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(rs)


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("gbt_compare")
    smoke = "--smoke" in sys.argv
    rng = np.random.default_rng(0)

    feats = build_segmentation_features(xr.open_zarr(str(T.CUBE), consolidated=True).data_vars)
    log.info(f"features: {len(feats)}")
    train_ds = T.make_dataset(*T.SPLITS["train"], feats, use_stack=True)
    val_ds = T.make_dataset(*T.SPLITS["val"], feats, use_stack=True)
    test_ds = T.make_dataset(*T.SPLITS["test"], feats, use_stack=True)

    # --- training rows (subsampled, same day-grid spirit as the U-Net's train split) ---
    tstride = (40 if smoke else TRAIN_DAY_STRIDE)
    tdays = list(range(0, len(train_ds), tstride))
    log.info(f"collecting train cells from {len(tdays)} days...")
    Xtr, ytr, _ = collect_cells(train_ds, tdays, all_land=False, rng=rng)
    if Xtr.shape[0] > MAX_TRAIN_ROWS:
        sel = rng.choice(Xtr.shape[0], size=MAX_TRAIN_ROWS, replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]
    log.info(f"train matrix {Xtr.shape}, pos rate {ytr.mean():.4f}")

    gbt = HistGradientBoostingClassifier(
        max_iter=(50 if smoke else 400), learning_rate=0.05, max_leaf_nodes=63,
        l2_regularization=1.0, validation_fraction=0.1, early_stopping=True, random_state=0)
    t0 = time.time(); gbt.fit(Xtr, ytr); log.info(f"GBT fit in {time.time()-t0:.0f}s, {gbt.n_iter_} iters")

    # --- eval on the EXACT cells the U-Net uses: strided days, ALL land cells, regime_metrics ---
    # Predict PER DAY and accumulate only prob/y/regime (small) — never materialize the full ~9GB
    # feature matrix for all eval cells.
    for name, ds in (("val", val_ds), ("test", test_ds)):
        stride = max(1, len(ds) // (20 if smoke else 365))
        days = list(range(0, len(ds), stride))
        probs, ys, rs = [], [], []
        for i in days:
            X, y, reg = ds[i]
            C = X.shape[0]
            Xf = X.numpy().reshape(C, -1).T
            land = reg.numpy().ravel() > 0
            probs.append(gbt.predict_proba(Xf[land])[:, 1])
            ys.append(y.numpy().ravel()[land]); rs.append(reg.numpy().ravel()[land])
        m = T.regime_metrics(np.concatenate(probs), np.concatenate(ys), np.concatenate(rs))
        log.info(f"{name.upper()} GBT: new-ign AP={m['new_ignition_ap']:.4f} (bar 0.50) "
                 f"spread AP={m['spread_ap']:.4f} overall={m['overall_ap']:.4f} "
                 f"prec@K={m['prec_at_k']:.4f} roc={m['roc']:.4f}  (matched 15:1, same as U-Net)")


if __name__ == "__main__":
    main()
