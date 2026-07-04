"""Daily collect→infer→log job — the FGDC v2 operational pipeline.

For one issue date t: build the RAW FGDC feature slice, run the production GBT (`gbt_fireguard`) + isotonic
calibrator, classify regime (`dist_to_fire` ≤ 6 km → spread, else new-ignition), and append to a
date-partitioned store:
  serving_store/inference/issue_date=YYYY-MM-DD.parquet  — per-region ignition/spread summary (alerts log)
  serving_store/grids/YYYY-MM-DD.npz                      — full prob/regime/today-fire grids (per-cell eval)
  serving_store/feature_stats/date=YYYY-MM-DD.parquet     — per-feature land stats (drift / pipeline health)

Modes:
  --mode replay (default): pull the day from the existing gold cube — proves the v2 plumbing, seeds the store.
  --mode live: the **no-cold-start append loop** (fetch feeds → append cube → recompute engineered features
               over a trailing window → predict). Issue date is gated by `latest_complete_fire_date()` — `t` is
               scoreable only after its afternoon SNPP pass settles. [phase B — not built yet; raises w/ guidance.]

v2 vs the legacy v1 job: `gbt_fireguard` + `FGDC_FEATURE_VARS`, **raw** features (no normalization — the GBT is
trained raw), **torch-free** (`src.data.metrics`), reads `FireGuard_coarse4.zarr`, and **no warm-start**.
Idempotent (skips a logged date unless --overwrite). Inspect with --show.
"""
from __future__ import annotations
import argparse
import datetime as _dt
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import xarray as xr

from src.data import metrics as M                        # torch-free project_root

CUBE = M.project_root / "data" / "gold" / "FireGuard_coarse4.zarr"
STORE = M.project_root / "data" / "serving_store"
MODEL = M.project_root / "models" / "gbt_fireguard.joblib"
CALIBRATOR = M.project_root / "models" / "gbt_fireguard.calibrator.joblib"
REGIME_KM = 6.0
# Live completeness gate: the served is_fire[t] is the UTC-day UNION of all VIIRS passes (S-NPP + NOAA-20 + NOAA-21
# = 6/day), so day t is COMPLETE as a feature day only after its LAST pass — the ~13:30 UTC SNPP afternoon overpass —
# settles in FIRMS (~3 h). Scoring t before that has only the earlier passes = a partial, off-distribution label (see
# the `fire-label-timing` memory). NOAA-20 (~12:40) and NOAA-21 (leads, ~12:00) are earlier, so SNPP still sets the gate.
FIRMS_AFTERNOON_SETTLE_UTC = 17     # UTC hour after which day t's afternoon SNPP pass is reliably in FIRMS
from src.data.regions import CCAA_NAMES as CCAA        # shared region map (code -> display name)


def latest_complete_fire_date(now_utc=None):
    """Latest UTC date usable as a COMPLETE feature day `t` (then predict t+1). is_fire[t] is the whole-UTC-day
    union of all VIIRS passes (S-NPP + NOAA-20), so `t` isn't complete until its last (~13:30 UTC SNPP afternoon)
    pass settles in FIRMS (~FIRMS_AFTERNOON_SETTLE_UTC UTC). Before that, today has only its earlier passes → latest
    complete is yesterday (a same-day nowcast); after, it's today (a true next-day forecast). See `fire-label-timing`."""
    now = now_utc or datetime.now(timezone.utc)
    today = now.date()
    return today if now.hour >= FIRMS_AFTERNOON_SETTLE_UTC else today - _dt.timedelta(days=1)


def _load():
    """Open the gold cube + the production model/calibrator; precompute land mask, region codes, static cols."""
    z = xr.open_zarr(str(CUBE), consolidated=True)
    art = joblib.load(MODEL)
    gbt, feats = art["model"], art["features"]            # the model's exact feature list + order
    calib = joblib.load(CALIBRATOR) if CALIBRATOR.exists() else None
    land = np.nan_to_num(z["is_spain"].values) > 0.5
    ccaa = np.rint(np.nan_to_num(z["AutonomousCommunities"].values)).astype(int)
    dyn_set = {f for f in feats if "time" in z[f].dims}
    stat_vals = {f: z[f].values.astype(np.float32)[land] for f in feats if f not in dyn_set}
    return z, gbt, feats, calib, land, ccaa, dyn_set, stat_vals


def build_feat(z, t_idx, feats, land, dyn_set, stat_vals):
    """Raw FGDC features for day t over land → (n_land, n_feat) in the model's order (no normalization)."""
    cols = [z[f].isel(time=t_idx).values.astype(np.float32)[land] if f in dyn_set else stat_vals[f]
            for f in feats]
    return np.stack(cols, -1)


def day_drivers(gbt, X, reg_land, p, feats, topk_cells=200, topk=6):
    """Per-day occlusion attribution: for the highest-risk cells per regime, set each feature to the day's
    MEAN (raw — NOT 0; the v1 bug was zeroing normalized inputs) and measure the mean predicted-risk DROP.
    A large positive drop ⇒ that feature's value today drives risk up. Occlusion-to-baseline (ignores
    interactions); dependency-free (no SHAP)."""
    fmean = X.mean(0)
    out = {}
    for name, code in (("ignition", 1), ("spread", 2)):
        idx = np.where(reg_land == code)[0]
        if idx.size == 0:
            continue
        k = min(topk_cells, idx.size)
        focus = idx[np.argsort(p[idx])[::-1][:k]]         # the relatively-highest-risk cells of this regime
        Xfoc = X[focus]
        base = gbt.predict_proba(Xfoc)[:, 1].mean()
        drops = np.empty(len(feats))
        for j in range(len(feats)):
            Xo = Xfoc.copy(); Xo[:, j] = fmean[j]
            drops[j] = base - gbt.predict_proba(Xo)[:, 1].mean()
        order = np.argsort(drops)[::-1][:topk]
        out[name] = [{"feature": feats[j], "drop": float(drops[j])} for j in order if drops[j] > 1e-5]
    return out


def _log(issue, target, p, reg_land, X, today_fire, feats, land, ccaa, gbt, alert_thr, source_tag, refreshed=None):
    """Write inference (region summary) + grid + feature-stats for one issued prediction."""
    import logging
    log = logging.getLogger("serve")
    H, W = land.shape
    flat = land.ravel()
    prob_grid = np.zeros(H * W, np.float32); prob_grid[flat] = p
    reg_grid = np.zeros(H * W, np.int8); reg_grid[flat] = reg_land
    prob_grid, reg_grid = prob_grid.reshape(H, W), reg_grid.reshape(H, W)
    now = datetime.now(timezone.utc).isoformat()

    rows = []
    for code, name in CCAA.items():
        rm = ccaa == code
        for rl, rcode in (("ignition", 1), ("spread", 2)):
            cells = prob_grid[rm & (reg_grid == rcode)]
            if cells.size == 0:
                continue
            rows.append(dict(issue_date=issue, target_date=target, region_code=code, region_name=name,
                             regime=rl, n_cells=int(cells.size), mean_prob=float(cells.mean()),
                             max_prob=float(cells.max()), expected_count=float(cells.sum()),
                             n_alert=int((cells >= alert_thr).sum()), source=source_tag, logged_at=now))
    (STORE / "inference").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(STORE / "inference" / f"issue_date={issue}.parquet", index=False)

    drivers = day_drivers(gbt, X, reg_land, p, feats)
    (STORE / "grids").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(STORE / "grids" / f"{issue}.npz", prob=prob_grid, regime=reg_grid,
                        today_fire=today_fire, issue_date=issue, target_date=target, source=source_tag,
                        fetched_at=now, refreshed=json.dumps(refreshed or []), drivers=json.dumps(drivers))
    fs = [dict(date=issue, feature=feats[j], mean=float(np.nanmean(X[:, j])), std=float(np.nanstd(X[:, j])),
               min=float(np.nanmin(X[:, j])), max=float(np.nanmax(X[:, j])),
               nan_frac=float(np.isnan(X[:, j]).mean()), logged_at=now) for j in range(len(feats))]
    (STORE / "feature_stats").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fs).to_parquet(STORE / "feature_stats" / f"date={issue}.parquet", index=False)
    log.info(f"{issue} [{source_tag}]: logged {len(rows)} region-rows, grid, {len(feats)} feature-stats (target {target})")


def predict_day(z, t_idx, gbt, feats, calib, land, ccaa, dyn_set, stat_vals, alert_thr, overwrite, source_tag):
    """Build raw features for day t_idx, predict + calibrate, classify regime, log to the store."""
    import logging
    log = logging.getLogger("serve")
    times = pd.DatetimeIndex(z["time"].values)
    issue = str(times[t_idx].date())
    target = str((times[t_idx] + pd.Timedelta(days=1)).date())
    if (STORE / "inference" / f"issue_date={issue}.parquet").exists() and not overwrite:
        log.info(f"{issue}: already logged (skip)"); return
    X = build_feat(z, t_idx, feats, land, dyn_set, stat_vals)
    p = gbt.predict_proba(X)[:, 1]
    p = calib.predict(p) if calib is not None else p      # true-prevalence calibrated risk
    d2f = z["dist_to_fire"].isel(time=t_idx).values[land]
    reg_land = np.where(d2f <= REGIME_KM, 2, 1).astype(np.int8)
    today_fire = (z["is_fire"].isel(time=t_idx).values > 0.5).astype(np.float32)
    _log(issue, target, p, reg_land, X, today_fire, feats, land, ccaa, gbt, alert_thr, source_tag)


def show():
    inf = sorted((STORE / "inference").glob("*.parquet"))
    print(f"store: {STORE}")
    print(f"  inference days: {len(inf)} | grids: {len(list((STORE/'grids').glob('*.npz')))} | "
          f"feature_stats days: {len(list((STORE/'feature_stats').glob('*.parquet')))}")
    if inf:
        df = pd.concat([pd.read_parquet(f) for f in inf[-3:]], ignore_index=True)
        for _, r in df.sort_values("max_prob", ascending=False).head(8).iterrows():
            print(f"    {r.issue_date}->{r.target_date} {r.regime:<9} {r.region_name:<18} "
                  f"max={r.max_prob:.3f} exp_cells={r.expected_count:.1f} alerts={r.n_alert}")


def serve_live(date_arg, alert_thr):
    """Progressive-refinement live serve: predict t+1 from TODAY's forecast+NRT edge via the ephemeral serve
    engine (serve_engine.serve_edge — fetch forecast/NRT, seed from the cube tail, compute engineered, NO cube
    write). Stamped PRELIMINARY until today's afternoon SNPP pass settles (~17 UTC); re-running later refines
    (overwrites the logged prediction). See the `fire-label-timing` memory."""
    import logging
    import scripts.serve_engine as EC
    log = logging.getLogger("serve")
    now = datetime.now(timezone.utc)
    t = date_arg or now.date().isoformat()                          # predict from TODAY's edge (partial fire ok)
    issue, fields = EC.serve_edge(CUBE, t)
    z, gbt, feats, calib, land, ccaa, dyn_set, stat_vals = _load()
    if fields is None:                                              # t already settled in the cube → replay it
        i = int(np.where(pd.DatetimeIndex(z["time"].values).date == _dt.date.fromisoformat(issue))[0][0])
        predict_day(z, i, gbt, feats, calib, land, ccaa, dyn_set, stat_vals, alert_thr, True, "live-settled")
        return
    X = np.stack([fields[f][land] if f in dyn_set else stat_vals[f] for f in feats], -1)
    p = gbt.predict_proba(X)[:, 1]
    p = calib.predict(p) if calib is not None else p
    reg_land = np.where(fields["dist_to_fire"][land] <= REGIME_KM, 2, 1).astype(np.int8)
    today_fire = (fields["is_fire"] > 0.5).astype(np.float32)
    target = str((pd.Timestamp(issue) + pd.Timedelta(days=1)).date())
    prelim = (issue == now.date().isoformat()) and (now.hour < FIRMS_AFTERNOON_SETTLE_UTC)
    tag = "live-prelim" if prelim else "live"
    _log(issue, target, p, reg_land, X, today_fire, feats, land, ccaa, gbt, alert_thr, tag)
    log.info(f"LIVE [{tag}]: issue {issue} → predict {target} "
             f"({'PRELIMINARY (today afternoon pass not settled)' if prelim else 'fire complete'})")


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["replay", "live"], default="replay")
    ap.add_argument("--date", help="issue date YYYY-MM-DD (replay); default = latest available")
    ap.add_argument("--backfill", type=int, default=0, help="replay the last N available days")
    ap.add_argument("--alert-thr", type=float, default=0.25)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if args.show:
        show(); return
    if args.mode == "live":
        serve_live(args.date, args.alert_thr); show(); return

    z, gbt, feats, calib, land, ccaa, dyn_set, stat_vals = _load()
    times = pd.DatetimeIndex(z["time"].values)
    if args.backfill > 0:
        idxs = list(range(len(times) - args.backfill, len(times)))
    elif args.date:
        idxs = [int(np.where(times.date == _dt.date.fromisoformat(args.date))[0][0])]
    else:
        idxs = [len(times) - 1]
    for i in idxs:
        predict_day(z, i, gbt, feats, calib, land, ccaa, dyn_set, stat_vals, args.alert_thr, args.overwrite, "replay")
    show()


if __name__ == "__main__":
    main()
