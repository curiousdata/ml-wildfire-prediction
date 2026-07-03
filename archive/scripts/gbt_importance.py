"""Per-regime permutation importance for the production GBT → models/gbt_coarse4.importance.json.

The app's "top drivers" panel needs to say *what* most drives tomorrow's risk, split by regime
(ignition = no nearby fire today; spread = fire within REGIME_KM). HistGBT exposes no native
feature_importances_, so we use permutation importance (average-precision drop when a feature is
shuffled) computed SEPARATELY on the ignition-regime rows and the spread-regime rows of the test
split. AP is rank-based, so the (monotonic) isotonic calibrator wouldn't change the ranking — we use
the raw GBT. Persisted once; the app reads the JSON (drivers are ~stationary, recompute on retrain).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from sklearn.inspection import permutation_importance

import scripts.train as T
import scripts.gbt_compare as G
from src.data.features import build_segmentation_features
import xarray as xr

TOPK = 8
NEG_PER_POS = 30  # keep ALL positives, subsample negatives (ignition positives are very rare — uniform
                  # subsampling drops them all and the AP becomes degenerate)


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("gbt_importance")
    art = joblib.load(T.project_root / "models" / "gbt_coarse4.joblib")
    gbt, feats = art["model"], art["features"]
    ds = T.make_dataset(*T.SPLITS["test"], feats, use_stack=True)
    days = list(range(0, len(ds), max(1, len(ds) // 60)))  # ~60 strided test days
    X, y, reg = G.collect_cells(ds, days, all_land=True)
    log.info(f"collected {X.shape[0]} land-cell rows over {len(days)} test days ({X.shape[1]} features)")
    rng = np.random.default_rng(0)
    out = {"features": list(feats), "regimes": {}}
    for name, code in [("ignition", 1), ("spread", 2)]:
        m = reg == code
        Xr, yr = X[m], y[m]
        if yr.sum() < 5:
            log.warning(f"{name}: only {int(yr.sum())} positives — skipping"); continue
        # keep ALL positives + a NEG_PER_POS sample of negatives (preserves the rare ignitions)
        pos = np.where(yr == 1)[0]; neg = np.where(yr == 0)[0]
        k = min(neg.size, NEG_PER_POS * pos.size)
        sel = np.concatenate([pos, rng.choice(neg, k, replace=False)])
        Xr, yr = Xr[sel], yr[sel]
        pi = permutation_importance(gbt, Xr, yr, scoring="average_precision",
                                    n_repeats=4, random_state=0, n_jobs=-1)
        order = np.argsort(pi.importances_mean)[::-1][:TOPK]
        top = [{"feature": feats[j], "drop": float(pi.importances_mean[j]),
                "std": float(pi.importances_std[j])} for j in order]
        out["regimes"][name] = {"n_rows": int(Xr.shape[0]), "n_pos": int(yr.sum()), "top": top}
        log.info(f"{name} ({int(yr.sum())} pos / {Xr.shape[0]} rows) top drivers: "
                 + ", ".join(f"{t['feature']}({t['drop']:.3f})" for t in top[:5]))
    p = T.project_root / "models" / "gbt_coarse4.importance.json"
    p.write_text(json.dumps(out, indent=2))
    log.info(f"wrote {p}")


if __name__ == "__main__":
    main()
