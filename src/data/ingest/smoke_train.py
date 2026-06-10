"""P1 gate — prove the FGDC pipeline end-to-end: load FGDC gold (4 km), predict next-day fire with a
HistGBT, report that it trains + a sanity metric.

This is NOT the production model (no vegetation yet — P2; tiny window). It only proves silver→coarsen→
features→train works on recollected FGDC data. Label = is_fire(t+1) from features(t). dist_to_fire(t) is
computed inline (top driver) via EDT on is_fire; vegetation/full engineered features arrive in P2/P4.

CLI: [--cube gold/FireGuard_coarse4.zarr]
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.ingest import grid

GOLD = grid.ROOT / "data" / "gold" / "FireGuard_coarse4.zarr"
# dynamic features the FGDC has (weather + soil + vegetation; filtered to those present in the cube)
DYN = ["t2m_mean", "t2m_max", "t2m_min", "t2m_range", "RH_mean", "RH_min", "RH_max",
       "surface_pressure_mean", "wind_speed_mean", "wind_speed_max", "wind_u_mean", "wind_v_mean",
       "total_precipitation_mean", "soil_moisture_mean", "soil_temperature_mean",
       "NDVI", "EVI", "LAI", "FAPAR", "LST", "popdens", "built_s"]   # popdens/built_s = GHS, interpolated daily
STAT = ["elevation_mean", "slope_mean", "dist_to_roads_mean",
        "CLC_2018_forest_and_semi_natural_proportion", "CLC_2018_scrub_proportion",
        "CLC_2018_artificial_proportion"]


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("smoke_train")
    a = sys.argv
    cube = a[a.index("--cube") + 1] if "--cube" in a else GOLD
    z = xr.open_zarr(str(cube), consolidated=True)
    dyn = [v for v in DYN if v in z]; stat = [v for v in STAT if v in z]
    log.info(f"FGDC gold {dict(z.sizes)} | dynamic feats {len(dyn)} | static feats {len(stat)}")

    isf = z["is_fire"].values  # [T,H,W]
    T, H, W = isf.shape
    land = (np.nan_to_num(z["is_spain"].values) > 0.5) if "is_spain" in z else np.ones((H, W), bool)
    cell_km = 4.0
    HORIZONS = [1, 3, 7]                          # predict fire within {1,3,7} days (user's N-day idea)
    Hmax = max(HORIZONS)

    # static block (constant over time). Keep NaN (HistGBT handles missing natively — no nan_to_num,
    # which would turn cloud-gapped veg / non-land into a misleading 0).
    Sblk = np.stack([z[v].values.astype(np.float32) for v in stat], -1) if stat else np.zeros((H, W, 0))

    # build features(t) once + a label per horizon; restrict t so every horizon has a full lookahead window
    tmax = T - 1 - Hmax
    if tmax < 1:
        log.warning(f"window too short for {Hmax}-day horizon (T={T}); reducing horizons")
        HORIZONS = [h for h in HORIZONS if h <= T - 2] or [1]; Hmax = max(HORIZONS); tmax = T - 1 - Hmax
    X, ys = [], {h: [] for h in HORIZONS}
    for t in range(tmax + 1):
        fire_t = isf[t] > 0.5
        d2f = (distance_transform_edt(~fire_t) * cell_km) if fire_t.any() else np.full((H, W), 1e3)
        dyn_t = np.stack([z[v].isel(time=t).values.astype(np.float32) for v in dyn], -1)
        feat = np.concatenate([dyn_t, d2f[..., None], Sblk], -1)[land]   # [n_land, F]
        X.append(feat)
        for h in HORIZONS:
            lab = (isf[t + 1:t + 1 + h] > 0.5).any(0).astype(np.int8)    # fire within next h days
            ys[h].append(lab[land])
    X = np.concatenate(X)
    names = dyn + ["dist_to_fire"] + stat
    log.info(f"rows={X.shape[0]:,} | features={X.shape[1]} ({len(dyn)} weather + dist_to_fire + {len(stat)} static)")

    n_pairs = tmax + 1
    n_per = X.shape[0] // n_pairs
    cut = int(n_pairs * 0.8) * n_per                # chronological 80/20 split
    log.info("HORIZON COMPARISON (predict fire within N days; same features):")
    for h in HORIZONS:
        y = np.concatenate(ys[h])
        Xtr, ytr, Xv, yv = X[:cut], y[:cut], X[cut:], y[cut:]
        gbt = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=63,
                                             l2_regularization=1.0, class_weight="balanced", random_state=0)
        gbt.fit(Xtr, ytr)
        msg = f"  within {h}d: pos-rate {100*y.mean():.3f}% | trained {gbt.n_iter_} it"
        if yv.sum() > 0 and yv.min() == 0:
            p = gbt.predict_proba(Xv)[:, 1]
            msg += f" | val AP={average_precision_score(yv,p):.3f} ROC-AUC={roc_auc_score(yv,p):.3f} (val pos {int(yv.sum())})"
        log.info(msg)
    log.info("GATE — FGDC silver→coarsen→features→multi-horizon GBT works end-to-end on recollected data.")


if __name__ == "__main__":
    main()
