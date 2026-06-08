"""Train + persist the PRODUCTION point-wise GBT (the model, post-2026-06-07 pivot).

After the segmentation U-Net was shelved (CHANGES.md 2026-06-07), the model is a point-wise
HistGradientBoostingClassifier on the 146 per-cell features. This trains it on the train split (reusing
the U-Net dataset so features/normalization/regime are IDENTICAL to the eval), saves the artifact +
feature list + metadata, and reports the official val/test metrics via train.py's `regime_metrics`.

Serving = load the joblib, predict_proba per land cell (NO logit adjustment — GBT outputs probabilities
directly; the per-regime adjustment was a U-Net training artifact).

Output: models/gbt_coarse4.joblib + models/gbt_coarse4.meta.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingClassifier

import scripts.train as T
import scripts.gbt_compare as G  # reuse collect_cells + constants (single source for the GBT data path)
from src.data.features import build_segmentation_features


def evaluate(gbt, ds, smoke=False):
    """Per-day predict over ALL land cells on the strided eval days; score with regime_metrics."""
    stride = max(1, len(ds) // (20 if smoke else 365))
    probs, ys, rs = [], [], []
    for i in range(0, len(ds), stride):
        X, y, reg = ds[i]
        Xf = X.numpy().reshape(X.shape[0], -1).T
        land = reg.numpy().ravel() > 0
        probs.append(gbt.predict_proba(Xf[land])[:, 1])
        ys.append(y.numpy().ravel()[land]); rs.append(reg.numpy().ravel()[land])
    return T.regime_metrics(np.concatenate(probs), np.concatenate(ys), np.concatenate(rs))


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("train_gbt")
    smoke = "--smoke" in sys.argv
    rng = np.random.default_rng(0)

    feats = build_segmentation_features(xr.open_zarr(str(T.CUBE), consolidated=True).data_vars)
    train_ds = T.make_dataset(*T.SPLITS["train"], feats, use_stack=True)
    val_ds = T.make_dataset(*T.SPLITS["val"], feats, use_stack=True)
    test_ds = T.make_dataset(*T.SPLITS["test"], feats, use_stack=True)

    tdays = list(range(0, len(train_ds), 40 if smoke else G.TRAIN_DAY_STRIDE))
    log.info(f"collecting {len(tdays)} train days ({len(feats)} features)...")
    Xtr, ytr, _ = G.collect_cells(train_ds, tdays, all_land=False, rng=rng)
    if Xtr.shape[0] > G.MAX_TRAIN_ROWS:
        s = rng.choice(Xtr.shape[0], G.MAX_TRAIN_ROWS, replace=False); Xtr, ytr = Xtr[s], ytr[s]
    log.info(f"train matrix {Xtr.shape}, pos rate {ytr.mean():.4f}")

    params = dict(max_iter=50 if smoke else 400, learning_rate=0.05, max_leaf_nodes=63,
                  l2_regularization=1.0, validation_fraction=0.1, early_stopping=True, random_state=0)
    gbt = HistGradientBoostingClassifier(**params)
    t0 = time.time(); gbt.fit(Xtr, ytr); log.info(f"GBT fit {gbt.n_iter_} iters in {time.time()-t0:.0f}s")

    val = evaluate(gbt, val_ds, smoke); test = evaluate(gbt, test_ds, smoke)
    for nm, m in (("VAL", val), ("TEST", test)):
        log.info(f"{nm}: new-ign AP={m['new_ignition_ap']:.4f} (GBT bar 0.50) spread={m['spread_ap']:.4f} "
                 f"overall={m['overall_ap']:.4f} prec@K={m['prec_at_k']:.4f} roc={m['roc']:.4f}")

    out = T.project_root / "models" / "gbt_coarse4.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": gbt, "features": list(feats)}, out)
    meta = {
        "model": "HistGradientBoostingClassifier (point-wise, post-2026-06-07 pivot)",
        "cube": str(T.CUBE), "n_features": len(feats), "features": list(feats),
        "params": params, "n_iter": int(gbt.n_iter_),
        "train_rows": int(Xtr.shape[0]), "train_pos_rate": float(ytr.mean()),
        "splits": {k: f"{v[0]}..{v[1]}" for k, v in T.SPLITS.items()},
        "val": val, "test": test,
        "note": "serve = predict_proba on normalized per-cell features (same dataset pipeline); no logit adj.",
    }
    (out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2))
    log.info(f"saved {out.name} + {out.with_suffix('.meta.json').name}")


if __name__ == "__main__":
    main()
