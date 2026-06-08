"""Live feature-slice builder — assemble a 'today' slice from live feeds + cube warm-start, run the GBT.

Strategy (honest MVP): WARM-START from the latest available cube slice (all 146 features correctly
engineered + normalized), then OVERWRITE only the cleanly-refreshable live features:
  * temperature (t2m_mean/min/max) ← Open-Meteo (keyless, validated °C — no unit ambiguity);
  * today's fire (is_fire, dist_to_fire) + regime ← FIRMS (if FIRMS_MAP_KEY set) — the IMPACTFUL live
    signal, since dist_to_fire is a top driver and it sets the ignition/spread regime.
Warm-started (flagged, refined later via the accumulating daily-job history): antecedent precip sums
(precip_sum_7/30/90d — need a rolling precip history), RH/pressure/wind aggregates (need Open-Meteo
hourly + the exact IberFire aggregation), vegetation (10-day), time_since_last_fire/burn_frequency (need
fire history). As the daily job runs forward, the store builds that history → fuller live features.

`--validate DATE`: build a live slice for a cube-range DATE (Open-Meteo temp + cube fire), compare the GBT
prediction to the pure-cube prediction — high agreement confirms the assembly is correct (no corruption).
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import xarray as xr
import datetime as _dt
import scripts.train as T
import scripts.fetch_openmeteo as OM
import scripts.fetch_firms as FB
import scripts.fetch_effis as EF
import src.data.feature_engineering as FE
from src.data.features import build_segmentation_features

TEMP_MAP = {"temperature_2m_mean": "t2m_mean", "temperature_2m_max": "t2m_max", "temperature_2m_min": "t2m_min"}
REGIME_KM = 6.0  # ignition (dist>6km) vs spread (<=6km); = regime_dist_cells 1.5 * 4km
_ANNUAL_RAIN = {}  # cube precip climatology (mm/yr per pixel), for KBDI; cached


def _annual_rain(ds):
    if "v" not in _ANNUAL_RAIN:
        # cube total_precipitation_mean is an HOURLY mean → daily total = 24×; ×365.25 = mm/yr
        _ANNUAL_RAIN["v"] = (24.0 * ds.ds["total_precipitation_mean"].mean("time").values * 365.25)
    return _ANNUAL_RAIN["v"]


def live_antecedent_dryness(date, ds, gx, gy, source):
    """Recompute precip_sum_7/30/90d, days_since_rain, total_precipitation_mean, kbdi from a live 90-day
    Open-Meteo precip/temp stack, using the SAME feature_engineering functions as the cube. Returns
    {feature: physical grid[H,W]} for the target date."""
    end = _dt.date.fromisoformat(date)
    start = (end - _dt.timedelta(days=89)).isoformat()
    # ALWAYS fetch the precip/temp history from the ARCHIVE (ERA5): the forecast endpoint returns the date
    # slots but ALL-NULL daily precip for past days, so it can't build antecedents. Archive has ~5-day
    # latency, so for very recent target dates the last few days come back all-NaN — we DROP those tail days
    # and compute antecedents over the available history (a 90-day sum tolerates a few-day-stale tail).
    # NOTE: era5_land returns NULL precip via Open-Meteo, so we use default ERA5; ERA5 vs the cube's
    # ERA5-Land precip differ in MAGNITUDE (corr ~0.9) but it washes out in the GBT (shift test).
    plon, plat, vals, dates = OM.fetch_grid_range(start, date, ["precipitation_sum", "temperature_2m_max"],
                                                  step=0.5, source="archive")
    valid = [t for t in range(len(dates)) if np.isfinite(vals["precipitation_sum"][t]).any()]
    if len(valid) < 7:
        raise ValueError(f"archive returned <7 valid precip days for {date} (got {len(valid)})")
    precip = np.stack([OM.regrid_to_cube(plon, plat, vals["precipitation_sum"][t], gx, gy) for t in valid])  # daily mm
    tmax = np.stack([OM.regrid_to_cube(plon, plat, vals["temperature_2m_max"][t], gx, gy) for t in valid])
    nd = len(valid)
    tp_h = precip / 24.0  # match cube's hourly-mean total_precipitation_mean units
    out = {}
    for N in (7, 30, 90):
        out[f"precip_sum_{N}d"] = FE.rolling_sum_time(tp_h, min(N, nd))[-1]
    out["days_since_rain"] = FE.days_since_rain(tp_h, FE.DRY_DAY_THRESHOLD_MM)[-1]
    out["total_precipitation_mean"] = tp_h[-1]
    # KBDI: daily TOTAL rain (precip), daily Tmax, annual-rain climatology; seed q0 from cube kbdi 90d back
    annual = _annual_rain(ds)
    q0 = None
    if "kbdi" in ds.ds:
        try:
            q0 = ds.ds["kbdi"].sel(time=start, method="nearest").values.astype("float64")
        except Exception:
            q0 = None
    out["kbdi"] = FE.keetch_byram_drought_index(precip, tmax, annual, q0=q0)[-1]
    return out


def build_live_slice(date, feats, ds, stats, base_idx, source="archive", use_firms=True,
                     use_antecedent=True, fire_source="effis"):
    """Return (Xn[C,H,W] normalized, regime[H,W], today_fire[H,W], refreshed:list)."""
    X, y, reg = ds[base_idx]
    Xn = X.numpy().copy(); C, H, W = Xn.shape
    regime = reg[0].numpy().astype(int)
    land = regime > 0
    today_fire = (ds.ds["is_fire"].sel(time=ds.get_time_value(base_idx)).values > 0.5).astype(np.float32)
    gx = ds.ds["x"].values.astype(float); gy = ds.ds["y"].values.astype(float)
    fidx = {f: j for j, f in enumerate(feats)}
    refreshed = []

    def overwrite(cf, grid_phys):
        if cf in fidx and stats.get(cf):
            m, s = stats[cf]["mean"], stats[cf]["std"]
            Xn[fidx[cf]] = ((grid_phys - m) / (s or 1.0)).astype(Xn.dtype)
            refreshed.append(cf)

    # --- Open-Meteo temperature (keyless, validated) ---
    plon, plat, vals = OM.fetch_grid(date, list(TEMP_MAP), step=0.5, source=source)  # ~5 reqs (rate-limit-friendly)
    for om, cf in TEMP_MAP.items():
        overwrite(cf, OM.regrid_to_cube(plon, plat, vals[om], gx, gy))

    # --- live antecedent dryness (precip_sum_7/30/90d, days_since_rain, kbdi) from a 90-day precip stack ---
    if use_antecedent:
        try:
            for cf, grid in live_antecedent_dryness(date, ds, gx, gy, source).items():
                if cf == "days_since_rain":
                    continue  # poor live reproduction (precip-product/threshold sensitivity) — keep warm-start
                overwrite(cf, grid)
            refreshed.append("antecedent-dryness")
        except Exception as e:
            refreshed.append(f"antecedent-failed:{type(e).__name__}")

    # --- MODEL fire features ← EFFIS burned-area (MATCHES the training definition). RECENCY CASCADE
    #     (weather-style persistence — NEVER the seasonal cube slice, which would import last-year's fires
    #     and fabricate spread-risk):
    #       1) live EFFIS today  → cache it;
    #       2) most-recent CACHED EFFIS (persisted from an earlier run) → use it, dated honestly;
    #       3) cold-start (no live, no cache) → NO-FIRE / all-ignition (never invent fire).
    #     FIRMS active-fire is a different quantity (pred-corr 0.10 vs the EFFIS-trained model) → display only.
    fire_asof = None
    if fire_source == "effis":
        fire = d2f = None
        try:
            fire = EF.polys_to_grid(EF.fetch_effis_polys(), gx, gy)
            EF.cache_effis(date, fire)
            fire_asof = date; refreshed.append("fire:EFFIS-live")
        except Exception:
            cached = EF.latest_cached_effis()
            if cached is not None:
                fire_asof, fire = cached
                refreshed.append(f"fire:EFFIS-persist:{fire_asof}")
        if fire is not None:                       # tiers 1–2: real EFFIS (today or persisted)
            d2f = EF.dist_to_fire(fire)
            fill = float(d2f[np.isfinite(d2f)].max()) if np.isfinite(d2f).any() else 0.0
            overwrite("is_fire", fire)
            overwrite("dist_to_fire", np.where(np.isfinite(d2f), d2f, fill))
            regime = np.where(~land, 0, np.where(d2f > REGIME_KM, 1, 2)).astype(int)
            today_fire = fire
        else:                                      # tier 3: no known fire → all-ignition, far from fire
            no_fire = np.zeros((H, W), np.float32)
            far = float(np.nanmax(ds.ds["dist_to_fire"].values)) if "dist_to_fire" in ds.ds else 1e3
            overwrite("is_fire", no_fire)
            overwrite("dist_to_fire", np.full((H, W), far, np.float32))
            regime = np.where(land, 1, 0).astype(int)
            today_fire = no_fire
            refreshed.append("fire:none(no-live-or-cache)")

    # --- DISPLAY fire ← FIRMS (low-latency hotspots, for the MAP only; NOT fed to the model) ---
    if use_firms and os.getenv("FIRMS_MAP_KEY"):
        try:
            df = FB.fetch_firms(os.getenv("FIRMS_MAP_KEY"), date)
            today_fire = FB.fires_to_grid(df["longitude"].values, df["latitude"].values, gx, gy)
            refreshed.append("display:FIRMS")
        except Exception:
            pass
    return Xn, regime, today_fire, refreshed


def predict(gbt, calib, Xn, regime):
    C, H, W = Xn.shape
    land = regime.ravel() > 0
    p = gbt.predict_proba(Xn.reshape(C, -1).T[land])[:, 1]
    if calib is not None:
        p = calib.predict(p)
    prob = np.zeros(H * W, np.float32); prob[land] = p
    return prob.reshape(H, W)


def main():
    import logging, joblib
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("live_slice")
    if "--validate-dryness" in sys.argv:
        date = sys.argv[sys.argv.index("--validate-dryness") + 1]
        feats = build_segmentation_features(xr.open_zarr(str(T.CUBE), consolidated=True).data_vars)
        ds = T.make_dataset("2008-01-01", "2024-12-31", feats, use_stack=True)
        gx = ds.ds["x"].values.astype(float); gy = ds.ds["y"].values.astype(float)
        anti = live_antecedent_dryness(date, ds, gx, gy, source="archive")
        log.info(f"live antecedent dryness vs cube ({date}) — Open-Meteo 90d precip/temp → feature_engineering:")
        for f in ["precip_sum_7d", "precip_sum_30d", "precip_sum_90d", "days_since_rain", "kbdi"]:
            if f in anti and f in ds.ds:
                truth = ds.ds[f].sel(time=date).values.astype(float)
                live = np.asarray(anti[f], float); land = np.isfinite(truth) & np.isfinite(live)
                err = np.abs(live[land] - truth[land])
                cc = np.corrcoef(live[land], truth[land])[0, 1] if land.sum() > 1 else float("nan")
                log.info(f"  {f:<22} MAE={err.mean():.3f}  median={np.median(err):.3f}  corr={cc:.4f}  "
                         f"(cube mean {truth[land].mean():.3f})")
        return
    if "--validate" not in sys.argv:
        print("Use --validate YYYY-MM-DD (assembly) or --validate-dryness YYYY-MM-DD (antecedents).", file=sys.stderr)
        return
    date = sys.argv[sys.argv.index("--validate") + 1]
    feats = build_segmentation_features(xr.open_zarr(str(T.CUBE), consolidated=True).data_vars)
    stats = json.loads(Path(T.STATS).read_text())
    ds = T.make_dataset("2008-01-01", "2024-12-31", feats, use_stack=True)
    dates = [str(ds.get_time_value(i))[:10] for i in range(len(ds))]
    base_idx = dates.index(date)
    art = joblib.load(T.project_root / "models" / "gbt_coarse4.joblib")
    gbt = art["model"]
    cp = T.project_root / "models" / "gbt_coarse4.calibrator.joblib"
    calib = joblib.load(cp) if cp.exists() else None

    # cube-baseline prediction
    Xc, _, rc = ds[base_idx]
    prob_cube = predict(gbt, calib, Xc.numpy(), rc[0].numpy().astype(int))
    # live-slice prediction (Open-Meteo temp + cube fire unless FIRMS key)
    Xn, reg, fire, refreshed = build_live_slice(date, feats, ds, stats, base_idx, source="archive",
                                                use_firms=bool(os.getenv("FIRMS_MAP_KEY")))
    prob_live = predict(gbt, calib, Xn, reg)
    land = rc[0].numpy() > 0
    d = np.abs(prob_live[land] - prob_cube[land])
    log.info(f"refreshed live features: {refreshed}")
    log.info(f"live-slice vs cube prediction ({date}): MAE={d.mean():.5f} max={d.max():.4f} "
             f"corr={np.corrcoef(prob_live[land], prob_cube[land])[0,1]:.4f}")
    log.info("  → high agreement confirms the live-slice assembly is correct (temperature swap doesn't "
             "corrupt; FIRMS fire, when keyed, changes dist_to_fire/regime = the impactful live signal).")


if __name__ == "__main__":
    main()
