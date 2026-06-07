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
import scripts.train as T
import scripts.fetch_openmeteo as OM
import scripts.fetch_firms as FB
from src.data.features import build_segmentation_features

TEMP_MAP = {"temperature_2m_mean": "t2m_mean", "temperature_2m_max": "t2m_max", "temperature_2m_min": "t2m_min"}
REGIME_KM = 6.0  # ignition (dist>6km) vs spread (<=6km); = regime_dist_cells 1.5 * 4km


def build_live_slice(date, feats, ds, stats, base_idx, source="archive", use_firms=True):
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

    # --- FIRMS today's fire → is_fire, dist_to_fire, regime (impactful; needs FIRMS_MAP_KEY) ---
    if use_firms and os.getenv("FIRMS_MAP_KEY"):
        try:
            df = FB.fetch_firms(os.getenv("FIRMS_MAP_KEY"), date)
            fire = FB.fires_to_grid(df["longitude"].values, df["latitude"].values, gx, gy)
            d2f = FB.dist_to_fire(fire)              # km
            today_fire = fire
            overwrite("is_fire", fire)
            overwrite("dist_to_fire", np.where(np.isfinite(d2f), d2f, d2f[np.isfinite(d2f)].max() if np.isfinite(d2f).any() else 0))
            regime = np.where(~land, 0, np.where(d2f > REGIME_KM, 1, 2)).astype(int)
            refreshed.append("regime(FIRMS)")
        except Exception as e:
            refreshed.append(f"FIRMS-failed:{type(e).__name__}")
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
    if "--validate" not in sys.argv:
        print("Use --validate YYYY-MM-DD (cube-range) to check the live-slice assembly vs the cube.", file=sys.stderr)
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
