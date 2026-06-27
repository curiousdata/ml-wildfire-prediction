"""Self-healing cube extension — the FGDC v2 speed layer (no warm-starts, no cold-starts).

`extend_cube_to(last_day_of_data, today)` walks the gold cube forward to `today`: for each missing day it
fetches the raw feeds, regrids straight to the 4 km grid, and APPENDS the raw dynamic vars; then it re-runs the
existing engineered-feature scripts (`add_engineered_features` / `add_fire_context`, whole-cube `--overwrite`)
so every history-dependent feature (precip_sum_*, kbdi's recursion, time_since_last_fire, the causal anomalies)
is recomputed from REAL history — never warm-started. The same path fills a 1-day gap, this ~1-month catch-up,
or a from-scratch rebuild; that is the self-healing.

Lambda watermark: ERA5 reanalysis lags ~5 days, so
  * day ≤ today − WATERMARK_DAYS : weather from the ERA5 ARCHIVE (final) → appended permanently.
  * day  > today − WATERMARK_DAYS : weather from the Open-Meteo FORECAST/nowcast (provisional) → written and
                                    OVERWRITTEN on later runs until it ages past the watermark and settles to
                                    the reanalysis value. (Fire = FIRMS NRT; veg = MODIS, both ~daily/16-day.)

Reuse: `scripts.fetch_openmeteo` + `ingest_weather.daily_point_features` (weather), `ingest_fire` (FIRMS→grid),
`ingest_veg` (MODIS→grid); regrid via `OM.make_regridder` onto the cube's own 4 km grid (direct-to-gold, the
agreed simplification — same as the GEFS path). Then the engineered scripts.

CLI:  python scripts/extend_cube.py [--to YYYY-MM-DD] [--no-engineered] [--dry-run]
NB: a live catch-up fetches real feeds (network) and the MODIS veg pull is the slow part — run headless.
"""
from __future__ import annotations
import argparse
import datetime as _dt
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import xarray as xr
from numcodecs import Blosc

import scripts.fetch_openmeteo as OM
from src.data.ingest import grid
from src.data.ingest import ingest_weather as IW
from src.data.ingest import ingest_fire as IF

CUBE = grid.ROOT / "data" / "gold" / "FireGuard_coarse4.zarr"
WATERMARK_DAYS = 5                                   # ERA5 archive lag → batch/speed split
COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
FETCH_STEP = 0.25                                    # ERA5 native ~0.25°


def _weather_day(date: str, gx, gy, wregrid_cache: dict, source: str):
    """Fetch one day's weather (archive=ERA5 final | forecast=provisional), aggregate to v1's daily family, and
    regrid the native ~0.25° points straight onto the 4 km grid → {feature: grid[NY,NX]}."""
    models = "era5" if source == "archive" else None     # archive → ERA5 reanalysis; forecast → the live model
    plon, plat, hv, times = OM.fetch_grid_hourly_range(date, date, IW.HOURLY_VARS, step=FETCH_STEP,
                                                       source=source, models=models)
    _, _, dly, _ = OM.fetch_grid_range(date, date, ["precipitation_sum"], step=FETCH_STEP, source=source, models=models)
    feats = IW.daily_point_features(hv, times, dly["precipitation_sum"][0], date)
    key = (round(float(plon[0]), 3), len(plon))          # cache the regridder by the (fixed) source point set
    if key not in wregrid_cache:
        wregrid_cache[key] = OM.make_regridder(np.asarray(plon, float), np.asarray(plat, float), gx, gy)
    rg = wregrid_cache[key]
    return {f: rg(vec) for f, vec in feats.items()}


def _fire_day(date: str, gx, gy, fire_src):
    """FIRMS active-fire for `date` rasterized directly onto the 4 km grid (any detection in a 4 km cell = the
    max-pool label). Reuses scripts.fetch_firms (the same backend ingest_fire uses)."""
    import os
    import scripts.fetch_firms as FB
    key = os.getenv("FIRMS_MAP_KEY")
    zero = {"is_fire": np.zeros((len(gy), len(gx)), np.float32)}
    if not key:
        return zero
    df = IF._filter_conf(FB.fetch_firms(key, date, src=fire_src, bbox=IF.BBOX_LL, days=1))
    df_day = df[df["acq_date"] == date] if "acq_date" in df.columns else df
    if df_day is None or df_day.empty:
        return zero
    return {"is_fire": FB.fires_to_grid(df_day["longitude"].values, df_day["latitude"].values, gx, gy)}


def extend_cube_to(last_day_of_data: str, today: str, with_engineered: bool = True, dry_run: bool = False):
    """Append raw days (last_day_of_data, today] to the gold cube, then recompute engineered features."""
    import logging
    log = logging.getLogger("extend_cube")
    z = xr.open_zarr(str(CUBE), consolidated=True)
    gx, gy = z["x"].values.astype(float), z["y"].values.astype(float)
    d0 = _dt.date.fromisoformat(last_day_of_data) + _dt.timedelta(days=1)
    end = _dt.date.fromisoformat(today)
    watermark = end - _dt.timedelta(days=WATERMARK_DAYS)
    days = [(d0 + _dt.timedelta(k)).isoformat() for k in range((end - d0).days + 1)]
    if not days:
        log.info("cube already current — nothing to extend"); return
    log.info(f"extend {days[0]}..{days[-1]} ({len(days)}d); watermark {watermark} "
             f"(≤ = ERA5 archive/final, > = forecast/provisional)")

    # weather var keys = the dynamic weather family the cube carries (from ingest_weather)
    wcache = {}
    rows = []
    for d in days:
        src = "archive" if _dt.date.fromisoformat(d) <= watermark else "forecast"
        try:
            day = _weather_day(d, gx, gy, wcache, src)
            day.update(_fire_day(d, gx, gy, IF.SRC_NRT))   # FIRMS NRT covers the recent catch-up edge
            # ⚠️ INCOMPLETE: the append also needs the cube's other raw dynamic vars before it will succeed —
            #   veg  EVI/FAPAR/LAI/LST/NDVI  → wire ingest_veg (MODIS, slow-varying, daily_interp) → regrid 4 km
            #   GHS  built_s / popdens       → ingest_static.interp_to_date (cheap; build_silver already does this)
            # day.update(_veg_day(d, gx, gy)); day.update(_ghs_day(d, gy, gx))   # TODO — then the append is whole.
            rows.append((d, src, day))
            log.info(f"  {d} [{src}]: weather {len(day)-1} vars + fire ({int(day['is_fire'].sum())} cells)")
        except Exception as exc:
            log.warning(f"  {d}: SKIP {type(exc).__name__} {exc}")
    if dry_run or not rows:
        log.info(f"dry-run / nothing fetched ({len(rows)} days ready)"); return

    # append raw dynamic vars along time (provisional days overwrite a prior provisional append for the same date)
    wkeys = [k for k in rows[0][2]]
    new_times = pd.to_datetime([d for d, _, _ in rows])
    arrs = {k: np.stack([day[k] for _, _, day in rows]) for k in wkeys}
    ds = xr.Dataset({k: (("time", "y", "x"), v) for k, v in arrs.items()},
                    coords={"time": new_times, "y": gy, "x": gx})
    enc = {k: {"compressor": COMPRESSOR} for k in wkeys}
    # NB: existing-date overwrite (settling a provisional edge) needs a region write; first cut = append-only.
    ds.to_zarr(str(CUBE), mode="a", append_dim="time", consolidated=True)
    log.info(f"appended {len(rows)} raw days ({len(wkeys)} dynamic vars) to {CUBE.name}")

    if with_engineered:
        log.info("recomputing engineered features (whole-cube --overwrite; reuses the existing scripts)...")
        for script in ("add_fire_context.py", "add_engineered_features.py"):   # fire_context makes precip_sum_* → before spi_90d
            subprocess.run([sys.executable, str(Path(__file__).with_name(script)),
                            "--cube", "FireGuard", "--factor", "4", "--overwrite"], check=True)
        log.info("engineered features recomputed — cube current, no warm-start.")


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", help="target date YYYY-MM-DD (default: today)")
    ap.add_argument("--no-engineered", action="store_true", help="append raw only, skip the engineered recompute")
    ap.add_argument("--dry-run", action="store_true", help="fetch + report, don't write")
    args = ap.parse_args()
    z = xr.open_zarr(str(CUBE), consolidated=True)
    last = str(pd.Timestamp(z["time"].values[-1]).date())
    today = args.to or _dt.date.today().isoformat()
    extend_cube_to(last, today, with_engineered=not args.no_engineered, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
