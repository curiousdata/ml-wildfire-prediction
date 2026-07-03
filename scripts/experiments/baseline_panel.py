"""Non-ML + linear baselines vs the production GBT on the SAME held-out FGDC val split.

Answers the credibility question the v1/v2 A/B never did: does the ML model beat the OPERATIONAL
STATUS QUO (a fire-weather index ranker) and the trivial PERSISTENCE / CLIMATOLOGY floors? Every
score is ranked through ``scripts.train.regime_metrics`` (rank-based: full-prevalence ROC, matched-
prevalence regime AP, R-precision @K) on the identical val split + regime decomposition as
``train_gbt_fgdc.py`` — so the rows are directly comparable to the GBT's reported numbers.

Floors (no training):
  * climatology  : causal per-(cell, day-of-year) fire-rate learned from TRAIN days only, scored at
                   the TARGET day's doy — the seasonal+spatial null ("you knew this from the calendar").
  * persistence  : -dist_to_fire(t) — proximity to today's fire. Dominates the spread regime, ~0 on
                   new ignition; contextualizes why spread AP is ~0.98.
  * ffwi / kbdi / hdw : single fire-weather index alone — what a fire service ranks by today.
Plus:
  * logistic     : balanced LogisticRegression on the full feature set (linear-ML floor → isolates the
                   nonlinearity gain). Imputed+scaled (LR can't take NaN; the GBT handles it natively).
  * gbt          : the saved production model (models/gbt_fireguard.joblib) — the thing we defend.

Read-only on the cube. Writes reports/baseline_panel.json.  Usage: python scripts/baseline_panel.py [--smoke]
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import joblib
import numpy as np
import pandas as pd
import xarray as xr

from src.data import metrics as T          # torch-free regime_metrics + project_root
from scripts.train_gbt_fgdc import CUBE, REGIME_KM, NEG_RATIO

MODEL = T.project_root / "models" / "gbt_fireguard.joblib"
REPORTS = T.project_root / "reports"


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("baseline_panel")
    smoke = "--smoke" in sys.argv
    horizon = 1
    rng = np.random.default_rng(0)

    datacube = xr.open_zarr(str(CUBE), consolidated=True)
    art = joblib.load(MODEL)
    gbt, feats = art["model"], art["features"]            # the EXACT feature list the GBT was trained on
    log.info(f"loaded GBT ({len(feats)} features) from {MODEL.name}")

    isf = datacube["is_fire"].values
    Tn, H, W = isf.shape
    land = np.nan_to_num(datacube["is_spain"].values) > 0.5
    nland = int(land.sum())
    tmax = Tn - 1 - horizon
    cut = int((tmax + 1) * 0.8)                           # identical split to train_gbt_fgdc
    log.info(f"{Tn} days, {nland} land cells; train ≤ {cut}, val > {cut}")

    dynamic = [f for f in feats if "time" in datacube[f].dims]
    dyn_set = set(dynamic)
    stat = [f for f in feats if "time" not in datacube[f].dims]
    stat_vals = {f: datacube[f].values.astype(np.float32)[land] for f in stat}

    def build_feat(block, lt):
        dvals = {f: block[f].isel(time=lt).values.astype(np.float32)[land] for f in dynamic}
        return np.stack([dvals[f] if f in dyn_set else stat_vals[f] for f in feats], -1)

    def label(t):
        return (isf[t + 1:t + 1 + horizon] > 0.5).any(0).astype(np.int8)[land]

    # target-day day-of-year (climatology is the seasonality of the day we PREDICT, t+1) ----------
    doy = pd.DatetimeIndex(datacube["time"].values).dayofyear.values
    doy_tp1 = pd.DatetimeIndex(datacube["time"].values + np.timedelta64(1, "D")).dayofyear.values

    # --- CLIMATOLOGY: per-(cell, doy) fire-rate from TRAIN days only (causal) --------------------
    log.info("building causal climatology (train per-cell×doy fire-rate)...")
    isf_train = isf[:cut][:, land].astype(np.float32)     # (cut, nland)
    clim = np.zeros((367, nland), np.float32)
    for d in np.unique(doy[:cut]):
        clim[d] = isf_train[doy[:cut] == d].mean(0)
    del isf_train

    # --- LOGISTIC floor: fit on a TRAIN subsample (NEG_RATIO:1), full feature set ----------------
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    log.info("building logistic train matrix (subsampled)...")
    Xtr, ytr = [], []
    for t0 in range(0, cut, 200):
        block = datacube[dynamic].isel(time=slice(t0, t0 + 200)).load()
        for lt, t in enumerate(range(t0, min(t0 + 200, cut))):
            yt = label(t)
            pos = np.where(yt == 1)[0]; neg = np.where(yt == 0)[0]
            if neg.size > NEG_RATIO * pos.size:
                neg = rng.choice(neg, NEG_RATIO * max(pos.size, 1), replace=False)
            keep = np.concatenate([pos, neg])
            Xtr.append(build_feat(block, lt)[keep]); ytr.append(yt[keep])
        if smoke:
            break
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0))
    t0 = time.time(); lr.fit(Xtr, ytr)
    log.info(f"logistic fit on {Xtr.shape} in {time.time()-t0:.0f}s")
    del Xtr, ytr

    # --- VAL: accumulate every score on the SAME cells ------------------------------------------
    cols = ["gbt", "logistic", "climatology", "persistence", "ffwi", "kbdi", "hdw"]
    acc = {k: [] for k in cols}
    ys, regs = [], []
    # floor-index columns that may NOT be model features (e.g. hdw is newer than the GBT-135) must be
    # loaded into the block too, on top of the model's dynamic features.
    val_read = list(dict.fromkeys(dynamic + ["dist_to_fire", "ffwi", "kbdi", "hdw"]))
    vstart = time.time()
    for t0 in range(cut, tmax + 1, 200):
        block = datacube[val_read].isel(time=slice(t0, t0 + 200)).load()
        for lt, t in enumerate(range(t0, min(t0 + 200, tmax + 1))):
            feat = build_feat(block, lt)
            ys.append(label(t))
            d2f = block["dist_to_fire"].isel(time=lt).values[land]
            regs.append(np.where(d2f <= REGIME_KM, 2, 1).astype(np.int8))
            acc["gbt"].append(gbt.predict_proba(feat)[:, 1])
            acc["logistic"].append(lr.predict_proba(feat)[:, 1])
            acc["climatology"].append(clim[doy_tp1[t]])
            acc["persistence"].append(-d2f)                          # closer to today's fire = higher risk
            for idx in ("ffwi", "kbdi", "hdw"):
                acc[idx].append(block[idx].isel(time=lt).values[land])
        if smoke:
            break
    y = np.concatenate(ys); reg = np.concatenate(regs)
    log.info(f"val scored in {time.time()-vstart:.0f}s  ({y.size} cell-days, {int(y.sum())} pos)")

    # --- METRICS: same regime_metrics for every score (NaN scores → low risk) -------------------
    rows = {}
    for k in cols:
        s = np.nan_to_num(np.concatenate(acc[k]), nan=-1e9)
        rows[k] = T.regime_metrics(s, y, reg)

    order = ["climatology", "persistence", "ffwi", "kbdi", "hdw", "logistic", "gbt"]
    hdr = f"{'baseline':<14}{'new_ign_ap':>11}{'spread_ap':>11}{'overall_ap':>12}{'prec@K':>9}{'roc':>8}"
    print("\n" + hdr); print("-" * len(hdr))
    for k in order:
        m = rows[k]
        print(f"{k:<14}{m['new_ignition_ap']:>11.4f}{m['spread_ap']:>11.4f}"
              f"{m['overall_ap']:>12.4f}{m['prec_at_k']:>9.4f}{m['roc']:>8.4f}")

    REPORTS.mkdir(exist_ok=True)
    outp = REPORTS / "baseline_panel.json"
    outp.write_text(json.dumps(
        {"cube": str(CUBE), "model": MODEL.name, "n_features": len(feats),
         "split": f"chrono 80/20 of {tmax + 1} days (val > {cut})",
         "regime_km": REGIME_KM, "n_pos": int(y.sum()), "rows": rows}, indent=2, default=float))
    print(f"\nsaved {outp}")


if __name__ == "__main__":
    main()
