"""Multi-satellite TRAINING-LABEL matrix (see ABLATIONS.md, 2026-07-04) — target-definition ablation for the
3-VIIRS-bird era. The live serve is FIXED 6-pass (S-NPP∪NOAA-20∪NOAA-21), so ALL configs are scored on the SAME
6-pass held-out set (2025-06→2026-04) vs 6-pass truth; only the TRAINING data differs. Answers: is a dense 6-pass
label from 2024-only better than long S-NPP-all-years, and does stitching (add sats as available) beat both?

  fire6 = S-NPP ∪ N20(≥2018) ∪ N21(≥2024)   — progressive "add sats as available" (stitch) timeline.
The 4 label-dependent features (dist_to_fire, fire_upwind_exposure, time_since_last_fire, burn_frequency_365d)
are recomputed on the config's fire timeline exactly as scripts/build_features.py does; every other feature is
label-independent and reused from the cube.

Configs (all tested on 6-pass truth):
  snpp    — train S-NPP feats+label, ALL years (= what we ship: 2p-train / 6p-serve = config D)
  stitch  — train fire6 feats+label, ALL years
  crop    — train fire6 feats+label, 2024+ only

Same GBT hyperparams as scripts/train_gbt.py (neg 30:1 per day, seed 0). Metric = src.data.metrics.regime_metrics.
Prereq: run scripts/experiments/multisat_fire_fetch.py first (caches fire_n20.npy / fire_n21.npy).

  python scripts/experiments/multisat_label_matrix.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import HistGradientBoostingClassifier

from src.data import metrics as T
from src.data.features import FGDC_FEATURE_VARS
from src.data.feature_engineering import fire_distance_and_exposure, days_since_rain, rolling_sum_time

CUBE = T.project_root / "data" / "gold" / "FireGuard_coarse4_t200.zarr"
CACHE = T.project_root / "data" / "cache" / "multisat"
REPORT = T.project_root / "reports" / "multisat_label_matrix.json"
FIRE = {"dist_to_fire", "fire_upwind_exposure", "time_since_last_fire", "burn_frequency_365d"}
REGIME_KM, NEG_RATIO = 6.0, 30
PARAMS = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=1.0,
              validation_fraction=0.1, early_stopping=True, random_state=0)


def main():
    t0 = time.time()
    z = xr.open_zarr(str(CUBE), consolidated=True)
    tidx = pd.DatetimeIndex(z["time"].values); N = len(tidx)
    gx, gy = z["x"].values.astype(float), z["y"].values.astype(float)
    land = np.nan_to_num(z["is_spain"].values) > 0.5
    cell_km = abs(gx[1] - gx[0]) / 1000.0
    nofire = float(np.hypot(z.sizes["y"], z.sizes["x"]) * cell_km)
    idx = lambda s: int(np.where(tidx.date == pd.Timestamp(s).date())[0][0])
    TR_CUT, TE_END, I2024 = idx("2025-06-01"), idx("2026-04-30"), idx("2024-01-17")
    print(f"days={N} land={int(land.sum())} | train t≤{TR_CUT-2} | test {tidx[TR_CUT].date()}..{tidx[TE_END-1].date()} "
          f"| crop from {tidx[I2024].date()}", flush=True)

    fire2 = z["is_fire"].values > 0.5
    n20 = np.load(CACHE / "fire_n20.npy")[:N]; n21 = np.load(CACHE / "fire_n21.npy")[:N]
    fire6 = fire2 | n20 | n21
    print(f"fire2={int(fire2.sum())} fire6={int(fire6.sum())} "
          f"(+{100*(fire6.sum()-fire2.sum())/fire2.sum():.0f}%) n20={int(n20.sum())} n21={int(n21.sum())}", flush=True)
    del n20, n21
    tsl6 = days_since_rain(fire6.astype("float32"), 0.5).astype("float32")
    bf6 = rolling_sum_time(fire6.astype("float32"), 365).astype("float32")

    feats = [f for f in FGDC_FEATURE_VARS if f in z]
    dyn = [f for f in feats if "time" in z[f].dims]
    stat_vals = {f: z[f].values.astype("float32")[land] for f in feats if "time" not in z[f].dims}

    def fire6_over(block, lt, t):
        di, ei = fire_distance_and_exposure(fire6[t], block["wind_u_mean"].isel(time=lt).values,
                                            block["wind_v_mean"].isel(time=lt).values, gx, gy, nofire)
        return {"dist_to_fire": di, "fire_upwind_exposure": ei,
                "time_since_last_fire": tsl6[t], "burn_frequency_365d": bf6[t]}, di

    def vec(block, lt, over):
        cols = []
        for f in feats:
            if f in FIRE:
                cols.append(over[f][land])
            elif f in stat_vals:
                cols.append(stat_vals[f])
            else:
                cols.append(block[f].isel(time=lt).values.astype("float32")[land])
        return np.stack(cols, -1)

    rng = np.random.default_rng(0)
    Xtr = {k: [] for k in ("snpp", "stitch", "crop")}; ytr = {k: [] for k in ("snpp", "stitch", "crop")}

    def add(bucket, feat, y):
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
        if neg.size > NEG_RATIO * pos.size:
            neg = rng.choice(neg, NEG_RATIO * max(pos.size, 1), replace=False)
        keep = np.concatenate([pos, neg]); Xtr[bucket].append(feat[keep]); ytr[bucket].append(y[keep])

    print("building train matrices …", flush=True)
    for b0 in range(0, TR_CUT - 1, 200):
        block = z[dyn].isel(time=slice(b0, min(b0 + 200, TR_CUT - 1))).load()
        for lt, t in enumerate(range(b0, min(b0 + 200, TR_CUT - 1))):
            snpp_over = {f: block[f].isel(time=lt).values for f in FIRE}
            add("snpp", vec(block, lt, snpp_over), (fire2[t + 1]).astype(np.int8)[land])
            f6, _ = fire6_over(block, lt, t)
            v6 = vec(block, lt, f6); y6 = (fire6[t + 1]).astype(np.int8)[land]
            add("stitch", v6, y6)
            if t >= I2024:
                add("crop", v6, y6)
        del block

    models = {}
    for k in ("snpp", "stitch", "crop"):
        X = np.concatenate(Xtr[k]); y = np.concatenate(ytr[k]); Xtr[k] = ytr[k] = None
        g = HistGradientBoostingClassifier(**PARAMS); tf = time.time(); g.fit(X, y); models[k] = g
        print(f"  fit [{k}] X{X.shape} pos={y.mean():.4f} → {g.n_iter_} iters ({time.time()-tf:.0f}s)", flush=True)
        del X, y

    print("evaluating on 6-pass held-out …", flush=True)
    probs = {k: [] for k in models}; ys, regs = [], []
    for b0 in range(TR_CUT, TE_END, 200):
        block = z[dyn].isel(time=slice(b0, min(b0 + 200, TE_END))).load()
        for lt, t in enumerate(range(b0, min(b0 + 200, TE_END))):
            f6, di = fire6_over(block, lt, t)
            X = vec(block, lt, f6)
            ys.append((fire6[t + 1]).astype(np.int8)[land])
            regs.append(np.where(di[land] <= REGIME_KM, 2, 1).astype(np.int8))
            for k, g in models.items():
                probs[k].append(g.predict_proba(X)[:, 1])
        del block
    y = np.concatenate(ys); reg = np.concatenate(regs)

    desc = {"snpp": "S-NPP, all years (SHIPPED: 2p-train/6p-serve)", "stitch": "fire6 stitch, all years",
            "crop": "fire6, 2024+ only"}
    res = {k: T.regime_metrics(np.concatenate(probs[k]).astype(float), y, reg) for k in models}
    print("\n" + "=" * 88)
    print("MULTI-SAT TRAINING-LABEL MATRIX — all scored on 6-pass truth, held-out 2025-06→2026-04")
    print("=" * 88)
    print(f"{'config':9}{'train label / years':44}{'newIgn':>9}{'spread':>9}{'ovr':>8}{'ROC':>8}{'p@K':>8}")
    for k in ("snpp", "stitch", "crop"):
        m = res[k]
        print(f"{k:9}{desc[k]:44}{m['new_ignition_ap']:>9.4f}{m['spread_ap']:>9.4f}"
              f"{m['overall_ap']:>8.4f}{m['roc']:>8.4f}{m['prec_at_k']:>8.4f}")
    b = res["snpp"]["new_ignition_ap"]
    print("=" * 88)
    print(f"  stitch−snpp {res['stitch']['new_ignition_ap']-b:+.4f}  crop−snpp {res['crop']['new_ignition_ap']-b:+.4f}  "
          f"stitch−crop {res['stitch']['new_ignition_ap']-res['crop']['new_ignition_ap']:+.4f}  (new-ign AP)")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "experiment": "multisat_label_matrix", "date": "2026-07-04",
        "test_window": f"{tidx[TR_CUT].date()}..{tidx[TE_END-1].date()}", "eval_truth": "6-pass (SNPP∪N20∪N21)",
        "test_cells": int(y.size), "test_pos": int(y.sum()),
        "configs": {k: {kk: float(vv) for kk, vv in res[k].items()} for k in res}}, indent=2))
    print(f"wrote {REPORT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
