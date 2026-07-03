"""Train + evaluate the production FGDC (v2) GBT on the enriched FireGuard cube — the clean IberFire A/B.

Reads FGDC_FEATURE_VARS (frozen, leak-free, fixed order) from the materialized gold cube; chronological
80/20 split; TRAIN negatives subsampled (rare-event); per-day VAL eval (memory-safe — never holds the full
val matrix) → train.regime_metrics (new-ign vs spread AP at MATCHED 15:1 prevalence, exactly v1's recipe).

Target = next-day (horizon=1) to match v1's new-ignition AP ≈ 0.63 bar. NB the comparison is directional:
FGDC label = VIIRS active-fire vs v1 EFFIS burned-area, and the val window is the held-out recent ~20%.

Output: models/gbt_fireguard.joblib (+ .meta.json).  Use --smoke for a fast pipeline check.
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

from src.data import metrics as T          # torch-free regime_metrics + project_root
from src.data.features import FGDC_FEATURE_VARS

_WX_PREFIX = ("t2m_", "RH_", "surface_pressure_", "wind_", 
              "total_precipitation_", "soil_")
WEATHER_LEAD_VARS = {f for f in FGDC_FEATURE_VARS
                     if f.startswith(_WX_PREFIX) 
                     or f in ("ffwi", "emc_peak")}

CUBE = T.project_root / "data" / "gold" / "FireGuard_coarse4_t200.zarr"
REGIME_KM = 6.0          # v1's regime_dist_cells=1.5 × 4 km cell → spread if dist_to_fire(t) ≤ 6 km
NEG_RATIO = 30           # train negatives kept per positive (per day), to bound the rare-event matrix


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("train_gbt")
    smoke = "--smoke" in sys.argv
    # --drop f1,f2,... : train on FGDC_FEATURE_VARS minus these (feature ablation); --tag NAME suffixes the
    # output model so an ablation run never clobbers the production gbt_fireguard.joblib slot.
    drop = set(sys.argv[sys.argv.index("--drop") + 1].split(",")) if "--drop" in sys.argv else set()
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
    lead = int(sys.argv[sys.argv.index("--weather-lead") + 1]) if "--weather-lead" in sys.argv else 0
    # --complement: ADD the weather vars at t+lead as EXTRA columns (keeping them at t too) — the deployment-
    # honest "does tomorrow's forecast help ON TOP OF today's weather?" ceiling. Without it, --weather-lead
    # SUBSTITUTES (replaces t weather with t+lead), which only tests "is t+lead as good as t".
    complement = "--complement" in sys.argv
    horizon = 1
    rng = np.random.default_rng(0)

    datacube = xr.open_zarr(str(CUBE), consolidated=True)
    feats = [f for f in FGDC_FEATURE_VARS if f in datacube and f not in drop]
    miss = [f for f in FGDC_FEATURE_VARS if f not in datacube]
    if miss:
        log.warning(f"{len(miss)} features missing from cube (skipped): {miss}")
    if drop:
        log.info(f"--drop: ablating {len(drop)} features → {sorted(drop)}")
    dynamic_features = [f for f in feats if "time" in datacube[f].dims]
    stat = [f for f in feats if "time" not in datacube[f].dims]
    dynamic_feature_set = set(dynamic_features)
    log.info(f"{len(feats)} features = {len(dynamic_features)} dynamic + {len(stat)} static; horizon={horizon}d")

    lead_feats = [f for f in feats if f in WEATHER_LEAD_VARS]        # weather vars (feats order) to also read at t+lead
    if complement and lead == 0:
        lead = 1; log.info("--complement set with no --weather-lead → defaulting lead=1")
    out_features = feats + [f"{f}_fc{lead}" for f in lead_feats] if complement else feats
    if complement:
        log.info(f"--complement: +{len(lead_feats)} weather channels at t+{lead} → {len(out_features)} total features")

    isf = datacube["is_fire"].values
    Tn, H, W = isf.shape
    land = np.nan_to_num(datacube["is_spain"].values) > 0.5
    stat_vals = {f: datacube[f].values.astype(np.float32)[land] for f in stat}   # static layers read once
    tmax = Tn - 1 - horizon
    cut = int((tmax + 1) * 0.8)
    log.info(f"{Tn} days, {int(land.sum())} land cells; train ≤ day {cut}, val > {cut}")

    def build_feat(block, local_t, lead=0, complement=False):
        """Stack the feature matrix for day t.
        substitute (default): the weather vars are read at t+lead IN PLACE of t.
        complement: ALL features at t, PLUS the weather vars ALSO at t+lead appended as extra columns."""
        if complement:
            base = {f: block[f].isel(time=local_t).values.astype(np.float32)[land] for f in dynamic_features}
            cols = [base[f] if f in dynamic_feature_set else stat_vals[f] for f in feats]
            cols += [block[f].isel(time=local_t + lead).values.astype(np.float32)[land] for f in lead_feats]
            return np.stack(cols, -1)
        dvals = {f: block[f].isel(time=local_t + (lead if f in WEATHER_LEAD_VARS else 0)).values
                 .astype(np.float32)[land] for f in dynamic_features}
        return np.stack([dvals[f] if f in dynamic_feature_set else stat_vals[f] for f in feats], -1)

    def label(t):
        return (isf[t + 1:t + 1 + horizon] > 0.5).any(0).astype(np.int8)[land]

    # --- TRAIN (first 80% of days; subsample negatives to NEG_RATIO:1) ---
    Xtr, ytr = [], []
    build_start = time.time()
    for t0 in range(0, cut, 200):
        block = datacube[dynamic_features].isel(time=slice(t0, t0+200+lead)).load()
        for local_t, t in enumerate(range(t0, min(t0 + 200, cut))):

            # Build the feature matrix and label vector for the current day t, subsampling negatives 
            feat = build_feat(block, local_t, lead=lead, complement=complement)
            yt = label(t)

            pos = np.where(yt == 1)[0]; neg = np.where(yt == 0)[0]
            if neg.size > NEG_RATIO * pos.size:
                neg = rng.choice(neg, NEG_RATIO * max(pos.size, 1), replace=False)

            keep = np.concatenate([pos, neg])
            Xtr.append(feat[keep]); ytr.append(yt[keep])
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    log.info(f"train matrix {Xtr.shape}, pos rate {ytr.mean():.4f} (built in {time.time()-build_start:.0f}s)")

    # --- FIT GBT ---

    params = dict(max_iter=50 if smoke else 400, learning_rate=0.05, max_leaf_nodes=63,
                  l2_regularization=1.0, validation_fraction=0.1, early_stopping=True, random_state=0)
    gbt = HistGradientBoostingClassifier(**params)
    t0 = time.time(); gbt.fit(Xtr, ytr)
    log.info(f"GBT fit {gbt.n_iter_} iters in {time.time()-t0:.0f}s")
    del Xtr, ytr

    # --- VAL (last 20% of days; per-day eval, full prevalence accumulated) ---
    probs, ys, regs = [], [], []
    val_start = time.time()
    for t0 in range(cut, tmax + 1, 200):
        block = datacube[dynamic_features].isel(time=slice(t0, t0+200+lead)).load()
        for local_t, t in enumerate(range(t0, min(t0 + 200, tmax + 1))):
            feat = build_feat(block, local_t, lead=lead, complement=complement)
            yt = label(t)
            probt = gbt.predict_proba(feat)[:, 1]
            regt = np.where(block["dist_to_fire"].isel(time=local_t).values[land] <= REGIME_KM, 2, 1).astype(np.int8)
            probs.append(probt); ys.append(yt); regs.append(regt)


    prob = np.concatenate(probs); y = np.concatenate(ys); reg = np.concatenate(regs)
    m = T.regime_metrics(prob, y, reg)
    log.info(f"VAL built in {time.time()-val_start:.0f}s")
    log.info(f"VAL next-day:  new-ign AP={m['new_ignition_ap']:.4f} (v1 bar≈0.63)  spread={m['spread_ap']:.4f}  "
             f"overall={m['overall_ap']:.4f}  prec@K={m['prec_at_k']:.4f}  roc={m['roc']:.4f}")

    out = T.project_root / "models" / (f"gbt_fireguard_{tag}.joblib" if tag else "gbt_fireguard.joblib")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": gbt, "features": out_features}, out)
    meta = {"model": "HistGradientBoostingClassifier (FGDC v2, point-wise)", "cube": str(CUBE),
            "n_features": len(out_features), "features": out_features, "horizon": horizon, "regime_km": REGIME_KM,
            "params": params, "n_iter": int(gbt.n_iter_), "val": m,
            "split": f"chrono 80/20 of {tmax + 1} days (train ≤ {cut})",
            "note": "v1-comparable new-ign AP at matched 15:1 prevalence; label=VIIRS active-fire (vs v1 EFFIS)."}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=float))
    log.info(f"saved {out.name} + {out.with_suffix('.meta.json').name}")


if __name__ == "__main__":
    main()
