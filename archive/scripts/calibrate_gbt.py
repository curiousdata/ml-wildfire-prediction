"""Calibrate the production GBT's probabilities for true-prevalence risk maps.

The GBT is trained on NEG-subsampled cells (~3.8% positive), so predict_proba is calibrated to that
inflated prevalence — far too high for the true rate (~0.005% new-ign / ~6% spread over all land cells).
For a risk product, "0.3" must mean ~30% burn frequency. We fit **isotonic regression** on the true-
prevalence VAL set (raw GBT prob -> observed label) — it learns the correction directly, is monotonic
(so ranking / AP / ROC are unchanged), and handles the subsample bias without assuming a parametric form.

Reports reliability (binned), ECE, Brier on TEST before vs after; confirms AP unchanged; saves the calibrator.
Output: models/gbt_coarse4.calibrator.joblib + models/gbt_coarse4.calibration.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import xarray as xr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import scripts.train as T
from src.data.features import build_segmentation_features


def collect_probs(gbt, ds, smoke=False):
    """Per-day predict_proba over ALL land cells (true prevalence); return (prob, label)."""
    stride = max(1, len(ds) // (20 if smoke else 365))
    ps, ys = [], []
    for i in range(0, len(ds), stride):
        X, y, reg = ds[i]
        land = reg.numpy().ravel() > 0
        Xf = X.numpy().reshape(X.shape[0], -1).T[land]
        ps.append(gbt.predict_proba(Xf)[:, 1]); ys.append(y.numpy().ravel()[land])
    return np.concatenate(ps), np.concatenate(ys)


def reliability(prob, y, bins=12):
    """Binned reliability + ECE + Brier. Log-spaced low-end bins (signal lives at low prob)."""
    edges = np.concatenate([[0], np.geomspace(1e-4, 1.0, bins)])
    idx = np.clip(np.digitize(prob, edges) - 1, 0, len(edges) - 2)
    rows, ece, N = [], 0.0, len(prob)
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        mp, fp = float(prob[m].mean()), float(y[m].mean())
        ece += n / N * abs(mp - fp)
        rows.append((edges[b], edges[b + 1], n, mp, fp))
    brier = float(np.mean((prob - y) ** 2))
    return rows, ece, brier


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("calibrate_gbt")
    smoke = "--smoke" in sys.argv
    rng = np.random.default_rng(0)

    art = joblib.load(T.project_root / "models" / "gbt_fireguard.joblib")
    gbt, feats = art["model"], art["features"]
    _ = build_segmentation_features(xr.open_zarr(str(T.CUBE), consolidated=True).data_vars)
    val_ds = T.make_dataset(*T.SPLITS["val"], feats, use_stack=True)
    test_ds = T.make_dataset(*T.SPLITS["test"], feats, use_stack=True)

    log.info("collecting VAL probs (true prevalence)...")
    pv, yv = collect_probs(gbt, val_ds, smoke)
    log.info(f"VAL cells {pv.size}, pos rate {yv.mean():.6f}")
    # fit isotonic on a subsample (keep all positives + large neg sample)
    pos = np.where(yv == 1)[0]; neg = np.where(yv == 0)[0]
    keep = np.concatenate([pos, rng.choice(neg, min(neg.size, 4_000_000), replace=False)])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(pv[keep], yv[keep])

    log.info("collecting TEST probs...")
    pt, yt = collect_probs(gbt, test_ds, smoke)
    pt_cal = iso.predict(pt)

    for nm, p in (("RAW", pt), ("CALIBRATED", pt_cal)):
        rows, ece, brier = reliability(p, yt)
        log.info(f"--- TEST reliability [{nm}] ECE={ece:.5f} Brier={brier:.6f} ---")
        for lo, hi, n, mp, fp in rows:
            log.info(f"   p[{lo:.4f},{hi:.4f}) n={n:>8} mean_pred={mp:.4f} obs_freq={fp:.4f}")
    # monotonic => ranking preserved
    log.info(f"ranking preserved: AP raw={average_precision_score(yt, pt):.4f} cal={average_precision_score(yt, pt_cal):.4f} "
             f"| ROC raw={roc_auc_score(yt, pt):.4f} cal={roc_auc_score(yt, pt_cal):.4f}")

    joblib.dump(iso, T.project_root / "models" / "gbt_fireguard.calibrator.joblib")
    rows, ece_r, brier_r = reliability(pt, yt); _, ece_c, brier_c = reliability(pt_cal, yt)
    (T.project_root / "models" / "gbt_fireguard.calibration.json").write_text(json.dumps({
        "method": "isotonic on true-prevalence VAL (corrects neg-subsample bias)",
        "test_ece_raw": ece_r, "test_ece_calibrated": ece_c,
        "test_brier_raw": brier_r, "test_brier_calibrated": brier_c,
        "test_ap_raw": float(average_precision_score(yt, pt)), "test_ap_calibrated": float(average_precision_score(yt, pt_cal)),
    }, indent=2))
    log.info(f"saved calibrator + calibration.json (ECE {ece_r:.5f} -> {ece_c:.5f})")


if __name__ == "__main__":
    main()
