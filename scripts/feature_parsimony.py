"""Feature parsimony for the production GBT — fewer features, same test performance.

Post-pivot the val→test gap is already solved (GBT generalizes), so this is for PARSIMONY (simpler model,
smaller dyn stack, faster serving), NOT to fix generalization. Method that respects correlation:
  1. Cluster the 146 features by Spearman correlation (Ward linkage; cut so |corr|>~0.7 groups together).
  2. Per-feature new-ignition permutation importance on VAL (to rank + pick cluster representatives).
  3. Reduced set = one representative (highest-importance member) per cluster.
  4. RETRAIN GBT on the reduced set, eval TEST — confirm new-ign/spread hold vs full (0.633 / 0.998).
Validation is the retrain-on-reduced TEST number, never importance alone (correlated features under-credit).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import xarray as xr
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

import scripts.train as T
import scripts.gbt_compare as G
from src.data.features import build_segmentation_features

CORR_THRESH = 0.30  # cluster cut on (1-|spearman|): groups features with |corr| >~ 0.70


def fit_gbt(X, y, smoke):
    g = HistGradientBoostingClassifier(max_iter=50 if smoke else 400, learning_rate=0.05,
        max_leaf_nodes=63, l2_regularization=1.0, validation_fraction=0.1, early_stopping=True, random_state=0)
    g.fit(X, y); return g


def eval_test(gbt, ds, cols, smoke):
    stride = max(1, len(ds) // (20 if smoke else 365))
    ps, ys, rs = [], [], []
    for i in range(0, len(ds), stride):
        X, y, reg = ds[i]
        land = reg.numpy().ravel() > 0
        Xf = X.numpy().reshape(X.shape[0], -1).T[land][:, cols]
        ps.append(gbt.predict_proba(Xf)[:, 1]); ys.append(y.numpy().ravel()[land]); rs.append(reg.numpy().ravel()[land])
    return T.regime_metrics(np.concatenate(ps), np.concatenate(ys), np.concatenate(rs))


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("parsimony")
    smoke = "--smoke" in sys.argv
    rng = np.random.default_rng(0)
    feats = build_segmentation_features(xr.open_zarr(str(T.CUBE), consolidated=True).data_vars)
    F = len(feats)
    train_ds = T.make_dataset(*T.SPLITS["train"], feats, use_stack=True)
    val_ds = T.make_dataset(*T.SPLITS["val"], feats, use_stack=True)
    test_ds = T.make_dataset(*T.SPLITS["test"], feats, use_stack=True)

    tdays = list(range(0, len(train_ds), 40 if smoke else G.TRAIN_DAY_STRIDE))
    Xtr, ytr, _ = G.collect_cells(train_ds, tdays, all_land=False, rng=rng)
    if Xtr.shape[0] > G.MAX_TRAIN_ROWS:
        s = rng.choice(Xtr.shape[0], G.MAX_TRAIN_ROWS, replace=False); Xtr, ytr = Xtr[s], ytr[s]
    log.info(f"train matrix {Xtr.shape}")

    # (1) Spearman -> Ward clusters
    samp = Xtr[rng.choice(Xtr.shape[0], min(50_000, Xtr.shape[0]), replace=False)]
    rho, _ = spearmanr(samp)
    rho = np.nan_to_num(rho, nan=0.0)
    dist = 1.0 - np.abs(rho); np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average")
    cl = fcluster(Z, t=CORR_THRESH, criterion="distance")
    nC = cl.max()
    log.info(f"{F} features -> {nC} correlation clusters (|corr|>~{1-CORR_THRESH:.2f} grouped)")

    # (2) full GBT + per-feature new-ign permutation importance on VAL
    full = fit_gbt(Xtr, ytr, smoke)
    vdays = list(range(0, len(val_ds), max(1, len(val_ds) // (10 if smoke else 60))))
    Xv, yv, rv = G.collect_cells(val_ds, vdays, all_land=True)
    ipos = (yv == 1) & (rv == 1); neg = np.where((yv == 0) & (rv > 0))[0]
    keep = np.concatenate([np.where(ipos)[0], rng.choice(neg, min(neg.size, 15 * int(ipos.sum())), replace=False)])
    Xe, ye = Xv[keep], yv[keep]
    base = average_precision_score(ye, full.predict_proba(Xe)[:, 1])
    imp = np.zeros(F)
    for j in range(F):
        col = Xe[:, j].copy(); Xe[:, j] = rng.permutation(Xe[:, j])
        imp[j] = base - average_precision_score(ye, full.predict_proba(Xe)[:, 1]); Xe[:, j] = col
    log.info(f"full GBT new-ign AP on val subset = {base:.4f}")

    # (3) representative per cluster = highest-importance member
    reps = []
    for c in range(1, nC + 1):
        members = np.where(cl == c)[0]
        reps.append(int(members[np.argmax(imp[members])]))
    reps = sorted(reps)
    log.info(f"reduced set = {len(reps)} representatives (1 per cluster). Dropped {F-len(reps)} redundant.")

    # (4) retrain on reduced set, eval TEST
    red = fit_gbt(Xtr[:, reps], ytr, smoke)
    mt = eval_test(red, test_ds, reps, smoke)
    log.info(f"REDUCED ({len(reps)} feats) TEST: new-ign AP={mt['new_ignition_ap']:.4f} spread={mt['spread_ap']:.4f} "
             f"prec@K={mt['prec_at_k']:.4f} roc={mt['roc']:.4f}  (full was 0.6330 / 0.9975)")

    kept = [feats[j] for j in reps]
    dropped = [feats[j] for j in range(F) if j not in set(reps)]
    (T.project_root / "models" / "gbt_parsimony.json").write_text(json.dumps({
        "n_full": F, "n_reduced": len(reps), "corr_thresh": CORR_THRESH,
        "reduced_test": mt, "full_test_newign": 0.6330,
        "kept_features": kept, "dropped_features": dropped,
        "top_importance": [(feats[j], float(imp[j])) for j in np.argsort(imp)[::-1][:20]],
    }, indent=2))
    log.info(f"saved gbt_parsimony.json (kept {len(kept)}, dropped {len(dropped)})")


if __name__ == "__main__":
    main()
