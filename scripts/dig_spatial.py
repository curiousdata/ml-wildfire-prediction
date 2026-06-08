"""Does SPATIAL context add over point-wise — tested with the strong learner (GBT)?

Fast, decisive screen before committing to a neural spatial model: give GBT explicit neighbourhood
context as features — the 3x3 and 5x5 mean of EVERY feature map (full-res, no downsampling) — and see if
test new-ignition AP beats the point-wise GBT (~0.63). If maximal hand-crafted spatial context can't lift
the strongest learner, spatial genuinely doesn't add here. If it does, a learned non-downsampling spatial
model (dilated FCN) is justified.

Neighbourhood means are a hand-crafted spatial prior (not a learned filter), so this is a strong positive
indicator but a soft negative — a learned spatial model could still find structure aggregates miss. Read
accordingly.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, xarray as xr
from scipy.ndimage import uniform_filter
from sklearn.ensemble import HistGradientBoostingClassifier
import scripts.train as T
from src.data.features import build_segmentation_features

MAX_TRAIN = 400_000; NEG = 15; TRAIN_STRIDE = 3
SCALES = [3, 5]  # neighbourhood window sizes (cells); 4km grid → 12km, 20km windows


def aug(X):
    """X[C,H,W] -> [C*(1+len(SCALES)), H, W] with neighbourhood means appended (full-res, no downsample)."""
    X = np.nan_to_num(X, nan=0.0)
    outs = [X]
    for k in SCALES:
        outs.append(uniform_filter(X, size=(1, k, k), mode="nearest"))
    return np.concatenate(outs, axis=0)


def collect(ds, days, all_land, rng=None):
    Xs, ys, rs = [], [], []
    for i in days:
        X, y, reg = ds[i]
        Xa = aug(X.numpy())
        C = Xa.shape[0]
        Xf = Xa.reshape(C, -1).T
        y = y.numpy().ravel(); reg = reg.numpy().ravel(); land = reg > 0
        if all_land:
            keep = land
        else:
            pos = land & (y == 1); negidx = np.where(land & (y == 0))[0]
            k = min(negidx.size, NEG * max(int(pos.sum()), 1))
            sel = rng.choice(negidx, k, replace=False) if k else np.array([], int)
            keep = np.zeros_like(land); keep[np.concatenate([np.where(pos)[0], sel])] = True
        Xs.append(Xf[keep]); ys.append(y[keep]); rs.append(reg[keep])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(rs)


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("dig_spatial")
    smoke = "--smoke" in sys.argv
    rng = np.random.default_rng(0)
    feats = build_segmentation_features(xr.open_zarr(str(T.CUBE), consolidated=True).data_vars)
    log.info(f"base feats {len(feats)} -> augmented {len(feats)*(1+len(SCALES))} (+{SCALES} neighbourhood means)")
    train_ds = T.make_dataset(*T.SPLITS["train"], feats, use_stack=True)
    test_ds = T.make_dataset(*T.SPLITS["test"], feats, use_stack=True)

    tdays = list(range(0, len(train_ds), 40 if smoke else TRAIN_STRIDE))
    log.info(f"collecting {len(tdays)} train days (with neighbourhood aug)...")
    Xtr, ytr, _ = collect(train_ds, tdays, all_land=False, rng=rng)
    if Xtr.shape[0] > MAX_TRAIN:
        s = rng.choice(Xtr.shape[0], MAX_TRAIN, replace=False); Xtr, ytr = Xtr[s], ytr[s]
    log.info(f"train matrix {Xtr.shape}")
    gbt = HistGradientBoostingClassifier(max_iter=50 if smoke else 400, learning_rate=0.05,
        max_leaf_nodes=63, l2_regularization=1.0, validation_fraction=0.1, early_stopping=True, random_state=0)
    t0 = time.time(); gbt.fit(Xtr, ytr); log.info(f"GBT(+spatial) fit {gbt.n_iter_} iters in {time.time()-t0:.0f}s")

    days = list(range(0, len(test_ds), max(1, len(test_ds) // (10 if smoke else 365))))
    probs, ys, rs = [], [], []
    for i in days:
        X, y, reg = test_ds[i]
        Xa = aug(X.numpy()); Xf = Xa.reshape(Xa.shape[0], -1).T
        land = reg.numpy().ravel() > 0
        probs.append(gbt.predict_proba(Xf[land])[:, 1])
        ys.append(y.numpy().ravel()[land]); rs.append(reg.numpy().ravel()[land])
    m = T.regime_metrics(np.concatenate(probs), np.concatenate(ys), np.concatenate(rs))
    log.info(f"TEST GBT+SPATIAL: new-ign AP={m['new_ignition_ap']:.4f}  spread={m['spread_ap']:.4f}  "
             f"prec@K={m['prec_at_k']:.4f}  roc={m['roc']:.4f}")
    log.info(f"  vs point-wise GBT new-ign 0.6330 / prec@K 0.4534  → "
             f"spatial {'HELPS (+%.3f)' % (m['new_ignition_ap']-0.633) if m['new_ignition_ap']>0.633 else 'does NOT add (%.3f)' % (m['new_ignition_ap']-0.633)}")


if __name__ == "__main__":
    main()
