"""Daily collect→infer→log job + store — the operational pipeline skeleton (IberFire-v2 + tracking).

Each run, for one "issue date" (t): build the feature slice, run the GBT, and append to a date-partitioned
store:
  serving_store/inference/issue_date=YYYY-MM-DD.parquet  — per-region ignition/spread summary (alerts log)
  serving_store/grids/YYYY-MM-DD.npz                      — full prob/regime/today-fire grids (for later
                                                            per-cell eval vs what actually burned at t+1)
  serving_store/feature_stats/date=YYYY-MM-DD.parquet     — per-feature land stats (DRIFT / pipeline-health)

Two modes:
  --mode replay (default): pulls the day from the existing cube (seeds the store now; proves the plumbing).
  --mode live: fetch AEMET+FIRMS+CLMS → assemble slice → same logging. Gated on API keys + slice assembly
               (the next build) — currently raises with guidance.

Idempotent (skips a date already logged unless --overwrite). Run daily via cron, or --backfill N to seed.
Inspect with --show.
"""
from __future__ import annotations
import argparse
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
import scripts.train as T
from src.data.features import build_segmentation_features

STORE = T.project_root / "data" / "serving_store"
CCAA = {1: "Andalucía", 2: "Aragón", 3: "Asturias", 4: "Baleares", 6: "Cantabria",
        7: "Castilla y León", 8: "Castilla-La Mancha", 9: "Cataluña", 10: "C. Valenciana",
        11: "Extremadura", 12: "Galicia", 13: "Madrid", 14: "Murcia", 15: "Navarra",
        16: "País Vasco", 17: "La Rioja"}


def _load():
    feats = build_segmentation_features(xr.open_zarr(str(T.CUBE), consolidated=True).data_vars)
    ds = T.make_dataset("2008-01-01", "2024-12-31", feats, use_stack=True)  # full range so any date is reachable
    art = joblib.load(T.project_root / "models" / "gbt_coarse4.joblib")
    calib_path = T.project_root / "models" / "gbt_coarse4.calibrator.joblib"
    calib = joblib.load(calib_path) if calib_path.exists() else None
    ccaa = np.rint(np.nan_to_num(xr.open_zarr(str(T.CUBE), consolidated=True)["AutonomousCommunities"].values)).astype(int)
    return feats, ds, art["model"], calib, ccaa


def _log(issue, target, prob, regg, today_fire, Xn, feats, ccaa, alert_thr, source_tag):
    """Write inference (region summary) + grid + feature-stats for one issued prediction."""
    import logging
    log = logging.getLogger("daily_job")
    C = Xn.shape[0]
    land2d = regg > 0
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for code, name in CCAA.items():
        rm = ccaa == code
        for rl, rcode in (("ignition", 1), ("spread", 2)):
            cells = prob[rm & (regg == rcode)]
            if cells.size == 0:
                continue
            rows.append(dict(issue_date=issue, target_date=target, region_code=code, region_name=name,
                             regime=rl, n_cells=int(cells.size), mean_prob=float(cells.mean()),
                             max_prob=float(cells.max()), expected_count=float(cells.sum()),
                             n_alert=int((cells >= alert_thr).sum()), source=source_tag, logged_at=now))
    (STORE / "inference").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(STORE / "inference" / f"issue_date={issue}.parquet", index=False)
    (STORE / "grids").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(STORE / "grids" / f"{issue}.npz", prob=prob, regime=regg,
                        today_fire=today_fire, issue_date=issue, target_date=target, source=source_tag)
    Xland = Xn.reshape(C, -1)[:, land2d.ravel()]
    fs = [dict(date=issue, feature=feats[j], mean=float(np.nanmean(Xland[j])), std=float(np.nanstd(Xland[j])),
               min=float(np.nanmin(Xland[j])), max=float(np.nanmax(Xland[j])),
               nan_frac=float(np.isnan(Xland[j]).mean()), logged_at=now) for j in range(C)]
    (STORE / "feature_stats").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fs).to_parquet(STORE / "feature_stats" / f"date={issue}.parquet", index=False)
    log.info(f"{issue} [{source_tag}]: logged {len(rows)} region-rows, grid, {C} feature-stats (target {target})")


def run_day(idx, feats, ds, gbt, calib, ccaa, overwrite, alert_thr):
    import logging
    log = logging.getLogger("daily_job")
    issue = str(ds.get_time_value(idx))[:10]
    target = str(ds.get_time_value(min(idx + 1, len(ds) - 1)))[:10]
    if (STORE / "inference" / f"issue_date={issue}.parquet").exists() and not overwrite:
        log.info(f"{issue}: already logged (skip)"); return
    X, y, reg = ds[idx]
    C, H, W = X.shape
    Xn = X.numpy(); regf = reg[0].numpy().ravel(); land = regf > 0
    p = gbt.predict_proba(Xn.reshape(C, -1).T[land])[:, 1]
    p = calib.predict(p) if calib is not None else p
    prob = np.zeros(H * W, np.float32); prob[land] = p
    prob, regg = prob.reshape(H, W), reg[0].numpy()
    today_fire = (ds.ds["is_fire"].sel(time=ds.get_time_value(idx)).values > 0.5).astype(np.float32)
    _log(issue, target, prob, regg, today_fire, Xn, feats, ccaa, alert_thr, "replay")


def run_live(date, feats, ds, gbt, calib, ccaa, overwrite, alert_thr):
    """Real live prediction for `date`: warm-start from the latest cube slice, overwrite live features
    (Open-Meteo temp + FIRMS fire), predict, log. target = date+1."""
    import datetime as _dt
    import logging
    import scripts.live_slice as LS
    log = logging.getLogger("daily_job")
    if (STORE / "inference" / f"issue_date={date}.parquet").exists() and not overwrite:
        log.info(f"{date}: already logged (skip)"); return
    stats = __import__("json").loads(Path(T.STATS).read_text())
    # warm-start from the cube slice with the nearest DAY-OF-YEAR (seasonal match for the features we don't
    # yet refresh live — antecedent dryness, vegetation, etc.), latest year among ties. Avoids applying e.g.
    # winter dryness to a summer prediction. (Full fix = live antecedents from a rolling history.)
    tgt_doy = _dt.date.fromisoformat(date).timetuple().tm_yday
    cdoy = np.array([pd.Timestamp(ds.get_time_value(i)).dayofyear for i in range(len(ds))])
    circ = np.minimum(np.abs(cdoy - tgt_doy), 366 - np.abs(cdoy - tgt_doy))
    base_idx = int(np.where(circ == circ.min())[0][-1])  # nearest day-of-year, latest year
    log.info(f"{date}: warm-start from cube {str(ds.get_time_value(base_idx))[:10]} (Δdoy={int(circ.min())})")
    src = "archive" if date < "2025-01-01" else "forecast"
    Xn, regg, today_fire, refreshed = LS.build_live_slice(date, feats, ds, stats, base_idx,
                                                          source=src, use_firms=True)
    log.info(f"{date} live: refreshed {refreshed}")
    prob = LS.predict(gbt, calib, Xn, regg)
    target = (_dt.date.fromisoformat(date) + _dt.timedelta(days=1)).isoformat()
    _log(date, target, prob, regg, today_fire, Xn, feats, ccaa, alert_thr, f"live:{src}+firms")


def show():
    inf = sorted((STORE / "inference").glob("*.parquet"))
    print(f"store: {STORE}")
    print(f"  inference days: {len(inf)}  | grids: {len(list((STORE/'grids').glob('*.npz')))} | "
          f"feature_stats days: {len(list((STORE/'feature_stats').glob('*.parquet')))}")
    if inf:
        df = pd.concat([pd.read_parquet(f) for f in inf[-3:]], ignore_index=True)
        top = df.sort_values("max_prob", ascending=False).head(8)
        print("  recent top region-risk rows:")
        for _, r in top.iterrows():
            print(f"    {r.issue_date}->{r.target_date} {r.regime:<9} {r.region_name:<18} "
                  f"max={r.max_prob:.3f} exp_cells={r.expected_count:.1f} alerts={r.n_alert}")


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
    feats, ds, gbt, calib, ccaa = _load()
    if args.mode == "live":
        if not args.date:
            import datetime as _dt
            args.date = (_dt.date.today() - _dt.timedelta(days=1)).isoformat()  # yesterday (feeds settled)
        run_live(args.date, feats, ds, gbt, calib, ccaa, args.overwrite, args.alert_thr)
        show(); return
    dates = [str(ds.get_time_value(i))[:10] for i in range(len(ds))]
    if args.backfill > 0:
        idxs = list(range(len(ds) - args.backfill, len(ds)))
    elif args.date:
        idxs = [dates.index(args.date)]
    else:
        idxs = [len(ds) - 1]
    for i in idxs:
        run_day(i, feats, ds, gbt, calib, ccaa, args.overwrite, args.alert_thr)
    show()


if __name__ == "__main__":
    main()
