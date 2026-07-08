"""Resolution ablation — does a finer grid predict fire better? Train the SAME GBT at a given coarsening factor
(4 km / 2 km / …), evaluate PER-REGIME (new-ign vs spread) AND with a resolution-fair LOCALIZATION metric.

Why localization: AP mechanically DROPS at finer res (denser negatives), so it under-sells resolution. Localization
is measured in KM at a FIXED physical tolerance, so 2 km can't win merely by having a finer grid — it must actually
place its alerts closer to real fire. Alert set = top ALERT_Q fraction of land cells by prob (fraction → resolution-
comparable). Report (a) HIT-RATE = fraction of actual next-day fires with an alert within {4,8} km, and
(b) ALERT-PRECISION = fraction of alert cells within {4,8} km of an actual fire.

  python scripts/experiments/resolution_ablation.py --factor 4   # baseline (uses existing gold)
  python scripts/experiments/resolution_ablation.py --factor 2   # after build_features --factor 2
Compare reports/resolution_ablation_{F}km.json side by side.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import distance_transform_edt
from sklearn.ensemble import HistGradientBoostingClassifier

from src.data import metrics as T
from src.data.features import FGDC_FEATURE_VARS

REGIME_KM, NEG_RATIO, ALERT_Q = 6.0, 30, 0.003        # alert set = top 0.3% of land cells by prob
PARAMS = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=1.0,
              validation_fraction=0.1, early_stopping=True, random_state=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, required=True)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--start", default=None, help="only use days >= this date (chrono 80/20 within the window)")
    ap.add_argument("--seed", type=int, default=0, help="neg-subsample + GBT seed (multi-seed de-risk)")
    args = ap.parse_args()
    PARAMS["random_state"] = args.seed
    F = args.factor
    cell_km = float(F)                                             # silver is 1 km → factor F = F-km cells
    block = args.block or max(12, int(200 * (F / 4.0) ** 2))       # scale block by area → ~constant RAM/block
    cube = T.project_root / "data" / "gold" / f"FireGuard_coarse{F}.zarr"
    t0 = time.time()
    z = xr.open_zarr(str(cube), consolidated=True)
    feats = [f for f in FGDC_FEATURE_VARS if f in z]
    dynset = {f for f in feats if "time" in z[f].dims}
    dyn = [f for f in feats if f in dynset]
    stat_vals = {f: z[f].values.astype(np.float32)[np.nan_to_num(z["is_spain"].values) > 0.5]
                 for f in feats if f not in dynset}
    isf = z["is_fire"].values
    Tn, H, W = isf.shape
    land = np.nan_to_num(z["is_spain"].values) > 0.5
    landflat = land.ravel()
    d2f = z["dist_to_fire"].values
    t0i = int(np.searchsorted(pd.DatetimeIndex(z["time"].values), pd.Timestamp(args.start))) if args.start else 0
    tmax = Tn - 2                                                  # last usable feature day (need t+1)
    cut = t0i + int((tmax + 1 - t0i) * 0.8)                        # chrono 80/20 WITHIN [t0i, tmax]
    rng = np.random.default_rng(args.seed)
    print(f"[{F}km seed{args.seed}] {cube.name}: {Tn}d {H}x{W}, {int(land.sum())} land, block={block}, "
          f"window t[{t0i},{tmax}] train<{cut} ({len(feats)} feats)", flush=True)

    def feat_day(blk, lt):
        return np.stack([blk[f].isel(time=lt).values.astype(np.float32)[land] if f in dynset else stat_vals[f]
                         for f in feats], -1)

    # --- train (subsample negatives NEG_RATIO:1 per day) ---
    Xtr, ytr = [], []
    for b0 in range(t0i, cut, block):
        blk = z[dyn].isel(time=slice(b0, min(b0 + block, cut))).load()
        for lt, t in enumerate(range(b0, min(b0 + block, cut))):
            X = feat_day(blk, lt); y = (isf[t + 1][land] > 0.5).astype(np.int8)
            pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
            if neg.size > NEG_RATIO * pos.size:
                neg = rng.choice(neg, NEG_RATIO * max(pos.size, 1), replace=False)
            keep = np.concatenate([pos, neg]); Xtr.append(X[keep]); ytr.append(y[keep])
        del blk
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    g = HistGradientBoostingClassifier(**PARAMS); tf = time.time(); g.fit(Xtr, ytr)
    print(f"[{F}km] train X{Xtr.shape} pos={ytr.mean():.4f} → {g.n_iter_} it ({time.time()-tf:.0f}s)", flush=True)
    del Xtr, ytr

    # --- eval: per-regime metrics + resolution-fair localization ---
    probs, ys, regs = [], [], []
    hit4 = hit8 = nfire = ap4 = ap8 = nalert = 0
    hig4 = hig8 = nig = hsp4 = hsp8 = nsp = 0                  # regime-split hit-rate (ignition vs spread fires)
    for b0 in range(cut, tmax + 1, block):
        blk = z[dyn].isel(time=slice(b0, min(b0 + block, tmax + 1))).load()
        for lt, t in enumerate(range(b0, min(b0 + block, tmax + 1))):
            X = feat_day(blk, lt); p = g.predict_proba(X)[:, 1]
            probs.append(p); ys.append((isf[t + 1][land] > 0.5).astype(np.int8))
            regs.append(np.where(d2f[t][land] <= REGIME_KM, 2, 1).astype(np.int8))
            fire = isf[t + 1] > 0.5
            if not (fire & land).any():
                continue
            pg = np.zeros(H * W, np.float32); pg[landflat] = p; pg = pg.reshape(H, W)
            k = max(1, int(ALERT_Q * land.sum()))
            alert = (pg >= np.partition(p, -k)[-k]) & land
            df = distance_transform_edt(~fire) * cell_km          # km to nearest actual fire
            da = distance_transform_edt(~alert) * cell_km         # km to nearest alert
            fl = fire & land
            regf = np.where(d2f[t] <= REGIME_KM, 2, 1)        # full-grid regime → split the next-day fire cells
            fig = fl & (regf == 1); fsp = fl & (regf == 2)
            nfire += int(fl.sum()); hit4 += int((da[fl] <= 4).sum()); hit8 += int((da[fl] <= 8).sum())
            nig += int(fig.sum()); hig4 += int((da[fig] <= 4).sum()); hig8 += int((da[fig] <= 8).sum())
            nsp += int(fsp.sum()); hsp4 += int((da[fsp] <= 4).sum()); hsp8 += int((da[fsp] <= 8).sum())
            nalert += int(alert.sum()); ap4 += int((df[alert] <= 4).sum()); ap8 += int((df[alert] <= 8).sum())
        del blk
    m = T.regime_metrics(np.concatenate(probs), np.concatenate(ys), np.concatenate(regs))
    loc = dict(hit_rate_4km=hit4 / max(nfire, 1), hit_rate_8km=hit8 / max(nfire, 1),
               alert_prec_4km=ap4 / max(nalert, 1), alert_prec_8km=ap8 / max(nalert, 1),
               ign_hit_4km=hig4 / max(nig, 1), ign_hit_8km=hig8 / max(nig, 1),
               spread_hit_4km=hsp4 / max(nsp, 1), spread_hit_8km=hsp8 / max(nsp, 1),
               n_fire=int(nfire), n_fire_ign=int(nig), n_fire_spread=int(nsp), n_alert=int(nalert))
    print(f"\n=== {F}km RESULT ===")
    print(f"  new-ign AP {m['new_ignition_ap']:.4f}  spread AP {m['spread_ap']:.4f}  overall {m['overall_ap']:.4f}  "
          f"ROC {m['roc']:.4f}  prec@K {m['prec_at_k']:.4f}")
    print(f"  LOCALIZATION (top {ALERT_Q*100:.1f}% alerts, resolution-fair): "
          f"hit@4km {loc['hit_rate_4km']:.3f} hit@8km {loc['hit_rate_8km']:.3f} | "
          f"alertPrec@4km {loc['alert_prec_4km']:.3f} @8km {loc['alert_prec_8km']:.3f}")
    print(f"    by regime — IGNITION hit@4km {loc['ign_hit_4km']:.3f} @8km {loc['ign_hit_8km']:.3f} (n={nig}) · "
          f"SPREAD hit@4km {loc['spread_hit_4km']:.3f} @8km {loc['spread_hit_8km']:.3f} (n={nsp})")
    out = T.project_root / "reports" / (f"resolution_ablation_{F}km.json" if args.seed == 0
                                        else f"resolution_ablation_{F}km_seed{args.seed}.json")
    out.write_text(json.dumps({"factor": F, "cell_km": cell_km,
                               "regime_metrics": {k: float(v) for k, v in m.items()},
                               "localization": {k: float(v) for k, v in loc.items()},
                               "train_days": int(cut), "test_days": int(tmax + 1 - cut)}, indent=2))
    print(f"  wrote {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
