"""FGDC weather ingester — daily meteorology on the 1 km grid from Open-Meteo (ERA5), backfill + append.

Design decision (see CHANGES.md FGDC entry): use the **uniform ERA5** model for ALL weather. Open-Meteo's
era5_land exposes only temperature/RH/soil/dewpoint (surface_pressure, wind, precip come back NULL), so a
v1-style era5_land cube would seam two reanalyses. ERA5 (default, ~0.25°) has every variable, is internally
consistent, and is identical at train and serve — the whole point of the FGDC. Native ~25 km is regridded
to the 1 km grid (effective resolution lives in the source, not the target).

Per day we produce v1's meteorology family on the 1 km grid:
  t2m_{mean,min,max,range}, RH_{mean,min,max,range}, surface_pressure_{mean,min,max,range},
  wind_speed_{mean,max}, wind_{u,v}_mean, wind_{u,v}_atmaxspeed, total_precipitation_mean,
  soil_moisture_mean, soil_temperature_mean   (soil = SWI proxy; mapped to SWI_* later).
Wind is decomposed to u/v at the hourly level then averaged — pooling-safe (no circular-mean error), the
same fix v1 applied at coarsen time.

CLI:
  --validate DATE         build one day, compare regridded t2m_mean/RH_mean to the v1 cube (sanity)
  --backfill START END    write per-day bronze npz caches for the range (resumable; skips existing)
  --append [DATE]         write the single latest day (default: 5 days ago = archive horizon)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import datetime as _dt
import os
import numpy as np

import scripts.fetch_openmeteo as OM
import src.data.feature_engineering as FE
from src.data.ingest import grid

BRONZE = grid.ROOT / "data" / "bronze" / "fireguard" / "weather"
HOURLY_VARS = ["temperature_2m", "relative_humidity_2m", "surface_pressure",
               "wind_speed_10m", "wind_direction_10m", "soil_moisture_7_to_28cm",
               "soil_temperature_7_to_28cm"]
FETCH_STEP = 0.25  # ERA5 native ~0.25°; fetch grid then regrid to 1 km

# Weather bronze is stored at NATIVE ERA5 resolution (the ~0.25° source points), NOT upsampled to 1 km —
# storing 1 km was ~71 MB/day (~370 GB for 13 yr) of pure interpolation redundancy (CHANGES.md 2026-06-14).
# Native = ~200 KB/day; build_silver regrids to 1 km on read via the cached make_regridder. Each npz carries
# the source point coords under these reserved keys so the regridder can be rebuilt; build_silver excludes them.
WLON, WLAT = "__lon__", "__lat__"


def atomic_savez(path, **arrays):
    """Write a compressed npz atomically: to a temp file, then os.replace → rename. A crash (e.g. ENOSPC)
    leaves only the temp file, never a truncated 'present' partition that skip-existing would treat as done.
    The temp name ends in .npz so np.savez_compressed doesn't silently append its own .npz suffix."""
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _stats(arr2d):
    """arr2d[n_hours, n_pts] → dict of per-point daily mean/min/max/range (NaN-safe)."""
    return {"mean": np.nanmean(arr2d, 0), "min": np.nanmin(arr2d, 0),
            "max": np.nanmax(arr2d, 0), "range": np.nanmax(arr2d, 0) - np.nanmin(arr2d, 0)}


def daily_point_features(hv, times, daily_precip, date):
    """Aggregate one day's hourly per-point arrays → {feature: per-point vector}. `hv` maps var→[n_hours,n_pts]
    for the full range; `times` the matching ISO hours; `daily_precip` the era5 daily precipitation_sum row."""
    idx = [i for i, t in enumerate(times) if t[:10] == date]
    if not idx:
        raise ValueError(f"no hourly data for {date}")
    sl = slice(idx[0], idx[-1] + 1)
    out = {}
    t = _stats(hv["temperature_2m"][sl]); out.update({f"t2m_{k}": v for k, v in t.items()})
    rh = _stats(hv["relative_humidity_2m"][sl]); out.update({f"RH_{k}": v for k, v in rh.items()})
    sp = _stats(hv["surface_pressure"][sl]); out.update({f"surface_pressure_{k}": v for k, v in sp.items()})
    spd = hv["wind_speed_10m"][sl] / 3.6                  # Open-Meteo wind is km/h → m/s (v1/FFWI/HDW use m/s)
    dirn = hv["wind_direction_10m"][sl]
    u, v = FE.wind_to_uv(spd, dirn)                       # per-hour u/v (pooling-safe)
    out["wind_speed_mean"] = np.nanmean(spd, 0)
    out["wind_speed_max"] = np.nanmax(spd, 0)
    out["wind_u_mean"] = np.nanmean(u, 0); out["wind_v_mean"] = np.nanmean(v, 0)
    jmax = np.nanargmax(np.where(np.isfinite(spd), spd, -np.inf), 0)
    cols = np.arange(spd.shape[1])
    out["wind_u_atmaxspeed"] = u[jmax, cols]; out["wind_v_atmaxspeed"] = v[jmax, cols]
    out["soil_moisture_mean"] = np.nanmean(hv["soil_moisture_7_to_28cm"][sl], 0)
    out["soil_temperature_mean"] = np.nanmean(hv["soil_temperature_7_to_28cm"][sl], 0)
    # precip: v1 stores total_precipitation_mean as an HOURLY mean (mm) → daily_sum / 24
    out["total_precipitation_mean"] = np.asarray(daily_precip, float) / 24.0
    return out


def build_day(date, plon=None, plat=None, hv=None, times=None, dprecip=None, step=FETCH_STEP):
    """Fetch (if not supplied) + aggregate + regrid one day → {feature: grid[NY,NX]} on the 1 km grid.
    (Regridded output — used by --validate and any live 1 km consumer; the bronze WRITE path stores native.)"""
    gx, gy = grid.x_coords(), grid.y_coords()
    if hv is None:
        plon, plat, hv, times = OM.fetch_grid_hourly_range(date, date, HOURLY_VARS, step=step, models="era5")
        _, _, dly, _ = OM.fetch_grid_range(date, date, ["precipitation_sum"], step=step, models="era5")
        dprecip = dly["precipitation_sum"][0]
    feats = daily_point_features(hv, times, dprecip, date)
    return {f: OM.regrid_to_cube(plon, plat, vec, gx, gy) for f, vec in feats.items()}


def native_day(date, step=FETCH_STEP):
    """Fetch + aggregate one day → NATIVE per-point dict {feature: vec[n_pts], __lon__, __lat__} (no regrid).
    This is what the bronze stores; build_silver regrids to 1 km on read."""
    plon, plat, hv, times = OM.fetch_grid_hourly_range(date, date, HOURLY_VARS, step=step, models="era5")
    _, _, dly, _ = OM.fetch_grid_range(date, date, ["precipitation_sum"], step=step, models="era5")
    feats = daily_point_features(hv, times, dly["precipitation_sum"][0], date)
    out = {k: np.asarray(v, np.float32) for k, v in feats.items()}
    out[WLON] = np.asarray(plon, np.float64); out[WLAT] = np.asarray(plat, np.float64)
    return out


def write_day(date):
    BRONZE.mkdir(parents=True, exist_ok=True)
    out = native_day(date)
    atomic_savez(BRONZE / f"{date}.npz", **out)
    return out


def backfill_range(start, end, chunk_days=60, step=FETCH_STEP):
    """Efficient backfill: Open-Meteo serves a whole date RANGE in one request per coord-batch, so we fetch
    the hourly+daily stack in chunk_days windows (one fetch per chunk, not per day), then aggregate+regrid
    +write each day. ~chunk_days× fewer requests than per-day. Resumable (skips days whose npz exists).

    NOTE — Open-Meteo free tier is QUOTA-LIMITED (per-minute/hour/day). Large multi-year backfills must run
    on a fresh quota; dev validate runs consume it. Bigger chunk_days = fewer requests = friendlier. If 429s
    persist (daily cap), just re-run later — it resumes from the first missing day."""
    import logging
    log = logging.getLogger("ingest_weather")
    BRONZE.mkdir(parents=True, exist_ok=True)
    d = _dt.date.fromisoformat(start); last = _dt.date.fromisoformat(end)
    while d <= last:
        cend = min(d + _dt.timedelta(days=chunk_days - 1), last)
        days = [(d + _dt.timedelta(days=k)).isoformat() for k in range((cend - d).days + 1)]
        if all((BRONZE / f"{x}.npz").exists() for x in days):
            log.info(f"{d}..{cend}: all exist, skip"); d = cend + _dt.timedelta(days=1); continue
        plon, plat, hv, times = OM.fetch_grid_hourly_range(d.isoformat(), cend.isoformat(), HOURLY_VARS,
                                                           step=step, models="era5")
        _, _, dly, ddates = OM.fetch_grid_range(d.isoformat(), cend.isoformat(), ["precipitation_sum"],
                                                step=step, models="era5")
        didx = {dt: k for k, dt in enumerate(ddates)}
        lonf, latf = np.asarray(plon, np.float64), np.asarray(plat, np.float64)
        for x in days:
            if (BRONZE / f"{x}.npz").exists():
                continue
            feats = daily_point_features(hv, times, dly["precipitation_sum"][didx[x]], x)
            out = {f: np.asarray(vec, np.float32) for f, vec in feats.items()}
            out[WLON] = lonf; out[WLAT] = latf                       # native: store source coords, regrid on read
            atomic_savez(BRONZE / f"{x}.npz", **out)
        log.info(f"{d}..{cend}: wrote {len(days)} days native ({step}° fetch, 1 chunk request)")
        d = cend + _dt.timedelta(days=1)


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("ingest_weather")
    a = sys.argv
    if "--validate" in a:
        import xarray as xr
        date = a[a.index("--validate") + 1]
        out = build_day(date, step=0.5)  # lighter fetch for a quick sanity check
        z = xr.open_zarr(str(grid.V1_CUBE), consolidated=True)
        log.info(f"FGDC weather (ERA5) vs v1 coarse4 (ERA5-Land) {date} — block-mean FGDC ×4 then compare:")
        for f in ["t2m_mean", "t2m_max", "RH_mean", "surface_pressure_mean", "wind_speed_max", "total_precipitation_mean"]:
            if f not in z:
                continue
            fg = out[f].reshape(grid.NY // 4, 4, grid.NX // 4, 4).mean((1, 3))  # to 4 km
            tv = z[f].sel(time=date).values.astype(float)
            m = np.isfinite(fg) & np.isfinite(tv)
            mae = np.abs(fg[m] - tv[m]).mean(); cc = np.corrcoef(fg[m], tv[m])[0, 1]
            log.info(f"  {f:24} MAE={mae:.3f} corr={cc:.4f} (v1 mean {tv[m].mean():.2f}, FGDC {fg[m].mean():.2f})")
        return
    if "--append" in a:
        i = a.index("--append")
        date = a[i + 1] if len(a) > i + 1 and not a[i + 1].startswith("-") else \
            (_dt.date.today() - _dt.timedelta(days=5)).isoformat()
        write_day(date); log.info(f"wrote weather {date}"); return
    if "--backfill" in a:
        i = a.index("--backfill"); start, end = a[i + 1], a[i + 2]
        step = float(a[a.index("--step") + 1]) if "--step" in a else FETCH_STEP
        backfill_range(start, end, step=step); return
    print("Use --validate DATE | --backfill START END | --append [DATE]", file=sys.stderr)


if __name__ == "__main__":
    main()
