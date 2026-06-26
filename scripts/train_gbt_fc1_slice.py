"""Slice experiment — does GEFS d+1 forecast weather (+ CAPE) lift new-ignition AP?

Reads the main cube's 135 features at day t, and the GEFS forecast-for-t+1 channels (`*_fc1` + `cape_*_fc1`)
regridded on the fly from the weather_fc1 bronze (no separate materialization). Internal chronological split
on the reforecast slice (default train 2016-2018 / val 2019). Trains BASELINE (135, t-weather only) and
COMPLEMENT (135 + fc1), reports regime metrics for both → the decision gate for the full build.

Honest framing: real, errorful GEFS d+1 forecast (vs the perfect-foresight reanalysis ceiling), trained the
way it would serve, on a different period than the production val (2023-26) — a signal test, not the final
number. CAPE is along for the ride as a Tier-2 probe.

Usage: python scripts/train_gbt_fc1_slice.py [--train-end 2018] [--val 2019] [--smoke]
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import HistGradientBoostingClassifier

import scripts.train as T
import scripts.fetch_openmeteo as OM
from src.data.features_fireguard import FGDC_FEATURE_VARS
from src.data.ingest import grid, ingest_weather as IW
from src.data.ingest import ingest_weather_gefs as G

CUBE = T.project_root / "data" / "gold" / "FireGuard_coarse4_t200.zarr"
REGIME_KM = 6.0
NEG_RATIO = 30


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("fc1_slice")
    a = sys.argv
    smoke = "--smoke" in a
    train_end = int(a[a.index("--train-end") + 1]) if "--train-end" in a else 2018
    val_year = int(a[a.index("--val") + 1]) if "--val" in a else 2019
    horizon = 1
    rng = np.random.default_rng(0)

    z = xr.open_zarr(str(CUBE), consolidated=True)
    gx, gy = z["x"].values.astype(float), z["y"].values.astype(float)
    isf = z["is_fire"].values
    land = np.nan_to_num(z["is_spain"].values) > 0.5
    times = pd.DatetimeIndex(z["time"].values)
    yr = times.year.values

    feats = [f for f in FGDC_FEATURE_VARS if f in z]
    dynamic = [f for f in feats if "time" in z[f].dims]
    dyn_set = set(dynamic)
    stat = [f for f in feats if "time" not in z[f].dims]
    stat_vals = {f: z[f].values.astype(np.float32)[land] for f in stat}

    # fc1 setup — regridder from the first available bronze day; feature order frozen here
    bronze = {p.stem for p in G.BRONZE.glob("*.npz")}
    if not bronze:
        raise SystemExit("no weather_fc1 bronze yet — run ingest_weather_gefs --backfill first")
    w0 = np.load(G.BRONZE / f"{sorted(bronze)[0]}.npz")
    fc_feats = [k for k in w0.files if not k.startswith("__")]
    fc_mode = sys.argv[sys.argv.index("--fc") + 1] if "--fc" in sys.argv else "all"   # all | weather | cape
    if fc_mode == "weather":
        fc_feats = [f for f in fc_feats if not f.startswith("cape")]
    elif fc_mode == "cape":
        fc_feats = [f for f in fc_feats if f.startswith("cape")]
    fc_names = [f"{k}_fc1" for k in fc_feats]
    wregrid = OM.make_regridder(w0[IW.WLON], w0[IW.WLAT], gx, gy)
    log.info(f"{len(feats)} base + {len(fc_names)} fc1 channels; {len(bronze)} bronze days available")

    # valid day indices: in [2016, val_year], label t+1 exists, and the fc1 bronze for date(t) is present
    def date_of(t):
        return str(times[t])[:10]
    cut_lo = int(np.argmax((yr >= 2016)))
    days = [t for t in range(cut_lo, len(times) - horizon)
            if 2016 <= yr[t] <= val_year and date_of(t) in bronze]
    tr_days = [t for t in days if yr[t] <= train_end]
    va_days = [t for t in days if yr[t] == val_year]
    if smoke:
        tr_days, va_days = tr_days[:60], va_days[:60]
    log.info(f"train days {len(tr_days)} (≤{train_end}) | val days {len(va_days)} (={val_year})")

    d2f_col = feats.index("dist_to_fire")        # regime read straight from the base matrix (no extra isel)

    def fc_feat(t):
        w = np.load(G.BRONZE / f"{date_of(t)}.npz")
        return np.stack([wregrid(w[f])[land] for f in fc_feats], -1)

    def label(t):
        return (isf[t + 1:t + 1 + horizon] > 0.5).any(0).astype(np.int8)[land]

    def block_iter(day_list, blk=100):
        """Yield (t, base[nland, n_feats]) reading dynamic feats in `blk`-day blocks — the 28× speedup over
        per-day isel on the 200-chunked cube. 100-day blocks bound memory (~1.2 GB/block)."""
        by_block = {}
        for t in sorted(day_list):
            by_block.setdefault((t // blk) * blk, []).append(t)
        for b0 in sorted(by_block):
            block = z[dynamic].isel(time=slice(b0, b0 + blk)).load()
            for t in by_block[b0]:
                dv = {f: block[f].isel(time=t - b0).values.astype(np.float32)[land] for f in dynamic}
                yield t, np.stack([dv[f] if f in dyn_set else stat_vals[f] for f in feats], -1)

    # --- TRAIN: subsampled matrix via block-read; fit baseline + complement ---
    bt0 = time.time()
    Xb, Xf, y = [], [], []
    for t, base in block_iter(tr_days):
        fc, yt = fc_feat(t), label(t)
        pos = np.where(yt == 1)[0]; neg = np.where(yt == 0)[0]
        if neg.size > NEG_RATIO * pos.size:
            neg = rng.choice(neg, NEG_RATIO * max(pos.size, 1), replace=False)
        keep = np.concatenate([pos, neg])
        Xb.append(base[keep]); Xf.append(fc[keep]); y.append(yt[keep])
    Xb = np.concatenate(Xb); Xf = np.concatenate(Xf); y_tr = np.concatenate(y)
    log.info(f"train built in {time.time()-bt0:.0f}s  {Xb.shape} (+fc {Xf.shape[1]}), pos {int(y_tr.sum())}")

    params = dict(max_iter=50 if smoke else 400, learning_rate=0.05, max_leaf_nodes=63,
                  l2_regularization=1.0, validation_fraction=0.1, early_stopping=True, random_state=0)
    base_model = HistGradientBoostingClassifier(**params).fit(Xb, y_tr)
    comp_model = HistGradientBoostingClassifier(**params).fit(np.hstack([Xb, Xf]), y_tr)
    del Xb, Xf

    # --- VAL: per-day eval via block-read (memory-safe — never holds the full val matrix) ---
    pb, pc, ys, rs = [], [], [], []
    vt = time.time()
    for t, base in block_iter(va_days):
        fc = fc_feat(t)
        pb.append(base_model.predict_proba(base)[:, 1])
        pc.append(comp_model.predict_proba(np.hstack([base, fc]))[:, 1])
        ys.append(label(t))
        rs.append(np.where(base[:, d2f_col] <= REGIME_KM, 2, 1).astype(np.int8))
    y_va = np.concatenate(ys); reg_va = np.concatenate(rs)
    mb = T.regime_metrics(np.concatenate(pb), y_va, reg_va)
    mc = T.regime_metrics(np.concatenate(pc), y_va, reg_va)
    log.info(f"val scored in {time.time()-vt:.0f}s  ({y_va.size} cell-days, {int(y_va.sum())} pos)")

    log.info(f"=== {val_year} val · BASELINE vs COMPLEMENT (GEFS d+1 forecast + CAPE) ===")
    for tag, r in [("baseline (135)", mb), (f"complement (135+{len(fc_names)})", mc)]:
        log.info(f"{tag:24} new-ign={r['new_ignition_ap']:.4f}  spread={r['spread_ap']:.4f}  "
                 f"overall={r['overall_ap']:.4f}  prec@K={r['prec_at_k']:.4f}  roc={r['roc']:.4f}")
    log.info(f"Δ new-ign (complement − baseline) = {mc['new_ignition_ap']-mb['new_ignition_ap']:+.4f}  "
             f"| Δ prec@K = {mc['prec_at_k']-mb['prec_at_k']:+.4f}")


if __name__ == "__main__":
    main()
