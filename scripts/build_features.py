"""Build all engineered features on a coarse gold cube, in the correct order — one command.

Merges the former `add_fire_context.py` (§E fire-context) and `add_engineered_features.py` (Group-A) into a
single ordered pass (2026-07-03 refactor). **Order is load-bearing:** `fire_context()` MUST run first because
`engineered()`'s `spi_90d` reads `precip_sum_90d` that `fire_context()` produces — running them out of order
raised `KeyError: precip_sum_90d`. Merging them removes that footgun. Both stages are whole-cube (the features
are causal/recursive: kbdi / precip_sum_* / time_since_last_fire / seasonal anomalies), append incrementally
(mode='a') to bound memory, and are idempotent-ish via --overwrite.

  fire_context : dist_to_fire, fire_upwind_exposure, precip_sum_{7,30,90,180,365}d
  engineered   : tpi, terrain_curvature, kbdi, spi_90d, ndvi/lai_anomaly, emc_peak, ffwi, vpd_peak, hdw, fvc,
                 dist_to_urban, aspect_southness/eastness, hli, time_since_last_fire, burn_frequency_365d,
                 doy/dow sincos, national/regional holidays

Usage:  python scripts/build_features.py --factor 4 --cube FireGuard [--overwrite]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numcodecs import Blosc

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.data.feature_engineering import (
    fire_distance_and_exposure,
    days_since_rain,
    equilibrium_moisture_content,
    fosberg_ffwi,
    fractional_vegetation_cover,
    heat_load_index,
    keetch_byram_drought_index,
    rolling_sum_time,
    seasonal_anomaly,
    terrain_curvature,
    topographic_position_index,
    vpd_kpa,
    hdw_index,
    day_of_year_sincos,
    day_of_week_sincos,
)
from src.data.regions import CCAA_TO_SUBDIV

COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=2)
TP_TO_DAILY_MM = 24   # FGDC ingester writes total_precipitation_mean = daily_sum/24 → ×24 = daily total mm.
                      # (Was a stale v1 2.9 → land-mean 81 mm/yr; ×24 → 674 mm/yr, matches AEMET ~640. Audit 2026-07-07.)
FIRE_CONTEXT_VARS = ("dist_to_fire", "fire_upwind_exposure",
                     "precip_sum_7d", "precip_sum_30d", "precip_sum_90d", "precip_sum_180d", "precip_sum_365d")


def fire_context(path: Path, overwrite: bool) -> None:
    """§E spatial fire-context + trailing precip windows (former add_fire_context.py). Causal: from is_fire(t)."""
    c = xr.open_zarr(path, consolidated=True)
    if any(v in c.data_vars for v in FIRE_CONTEXT_VARS) and not overwrite:
        raise SystemExit(f"{FIRE_CONTEXT_VARS} already present in {path}. Use --overwrite.")

    xc = c["x"].values.astype("float64")
    yc = c["y"].values.astype("float64")
    cell_km = abs(xc[1] - xc[0]) / 1000.0
    no_fire_dist_km = float(np.hypot(c.sizes["y"], c.sizes["x"]) * cell_km)  # grid diagonal
    nt, ny, nx = c.sizes["time"], c.sizes["y"], c.sizes["x"]
    print(f"[fire-context] {path.name}: {nt} days, grid {ny}x{nx}, cell {cell_km:.0f} km, "
          f"no-fire cap {no_fire_dist_km:.0f} km")

    fire_all = c["is_fire"].values  # (T,y,x) uint8 — cheap (~0.4 GB)
    dist = np.empty((nt, ny, nx), dtype="float32")
    expo = np.empty((nt, ny, nx), dtype="float32")
    for t in range(nt):
        u = c["wind_u_mean"].isel(time=t).values  # per-day read (keeps memory low)
        v = c["wind_v_mean"].isel(time=t).values
        dist[t], expo[t] = fire_distance_and_exposure(fire_all[t], u, v, xc, yc, no_fire_dist_km)
        if t % 1000 == 0:
            print(f"  {t}/{nt}")

    # Memory-bounded write: each whole-cube var is ~1.4 GB at 4 km, so bundling all 7 + a float64 cumsum in one
    # Dataset peaked ~16 GB on a 17 GB box → swap/freeze. Write each var SEPARATELY (mode='a') and free between.
    coords = {"time": c["time"], "y": c["y"], "x": c["x"]}

    def _append(name: str, arr: np.ndarray, attrs: dict | None = None) -> None:
        if arr.dtype.kind == "f":
            arr = arr.astype("float16")   # storage: all fire-context outputs are fp16-safe (km dist, precip mm, exposure)
        ds = xr.Dataset({name: (("time", "y", "x"), arr)}, coords=coords)
        if attrs:
            ds[name].attrs = attrs
        ds = ds.chunk({"time": 1, "y": ny, "x": nx})
        ds.to_zarr(path, mode="a", encoding={name: {"compressor": COMPRESSOR}}, consolidated=True, zarr_format=2)
        print(f"  appended {name} {arr.shape} {arr.dtype}")

    _append("dist_to_fire", dist,
            {"units": "km", "description": "Distance to nearest fire cell on day t (0 on fire cells)."})
    _append("fire_upwind_exposure", expo,
            {"description": "(W.d)/|d|^2 downwind-exposure to nearest fire; >0 downwind, <0 upwind."})
    del dist, expo, fire_all

    # float32 cumsum: ~7 sig-digits at ~1e4 mm totals → sub-0.001 mm window resolution, and half the
    # RAM of float64 (5.8 vs 11.6 GB whole-cube). Input .values is already fp16 (2.9 GB).
    precip_sum = np.cumsum(c["total_precipitation_mean"].values, axis=0, dtype="float32")  # fp16 in, f32 accum (no fp32 copy)
    for N in (7, 30, 90, 180, 365):
        win = precip_sum.copy()                                  # rows < N keep the partial cumsum
        win[N:] = precip_sum[N:] - precip_sum[:-N]               # N-day trailing window
        _append(f"precip_sum_{N}d", win)
        del win
    del precip_sum
    print("[fire-context] done.")


def engineered(path: Path, overwrite: bool) -> None:
    """Group-A engineered features (former add_engineered_features.py). Reads precip_sum_90d from fire_context()."""
    c = xr.open_zarr(path, consolidated=True)

    coords = {"time": c["time"], "y": c["y"], "x": c["x"]}
    ny, nx = c.sizes["y"], c.sizes["x"]
    doy = c["time"].dt.dayofyear.values
    nyears = (c["time"].values[-1] - c["time"].values[0]).astype("timedelta64[D]").astype(int) / 365.25

    existing = set(c.data_vars)

    F32_KEEP = {"time_since_last_fire"}   # can exceed 2048 days → fp16 loses integer precision; keep float32

    def append(name, dims, data, attrs=None):
        if name in existing and not overwrite:
            print(f"  skip {name} (exists; use --overwrite)"); return
        if data.dtype.kind == "f" and name not in F32_KEEP:
            data = data.astype("float16")   # storage: engineered floats are fp16-safe for HistGBT (255-bin)
        if name in existing:  # mode='a' won't replace an existing var -> delete it first
            import zarr
            g = zarr.open_group(str(path), mode="a")
            if name in g:
                del g[name]
        sub_coords = {d: coords[d] for d in dims}
        ds = xr.Dataset({name: (dims, data)}, coords=sub_coords)
        if attrs:
            ds[name].attrs = attrs
        chunks = {"time": 1, "y": ny, "x": nx} if "time" in dims else {"y": ny, "x": nx}
        ds = ds.chunk({k: v for k, v in chunks.items() if k in dims})
        ds.to_zarr(path, mode="a", encoding={name: {"compressor": COMPRESSOR}},
                   consolidated=True, zarr_format=2)
        print(f"  appended {name} {tuple(data.shape)} {data.dtype}")

    print(f"[enrich] {path.name}  grid {ny}x{nx}, {len(doy)} days")

    # --- static terrain (cheap) ---
    elev = c["elevation_mean"].values
    append("tpi", ("y", "x"), topographic_position_index(elev, size=5),
           {"description": "Topographic Position Index (elev - local 5x5 mean); ridge>0, valley<0."})
    append("terrain_curvature", ("y", "x"), terrain_curvature(elev),
           {"description": "Terrain curvature (Laplacian of elevation); convex>0, concave<0."})

    # --- KBDI (needs tp + t2m_max) ---
    tp = c["total_precipitation_mean"].values
    daily_rain = (TP_TO_DAILY_MM * tp).astype("float16")     # keep fp16 (kbdi upcasts per-slice); was float64 11.6 GB
    del tp                                                   # free the fp16 input before allocating more
    R = np.nansum(daily_rain, axis=0, dtype="float64") / nyears   # per-grid accumulator stays float64
    tmax = c["t2m_max"].values                               # fp16 whole-cube (kbdi upcasts per-slice)
    append("kbdi", ("time", "y", "x"),
           keetch_byram_drought_index(daily_rain, tmax, R),   # already returns float16 (KBDI ≤ 203.2)
           {"units": "mm", "description": f"Keetch-Byram Drought Index (daily_rain={TP_TO_DAILY_MM}*tp, approx)."})
    del daily_rain, tmax

    # --- seasonal anomalies (SPI proxy + greenness) — CAUSAL climatology (prior years only): no train/test
    #     leakage and identical to what's available at serve time. ---
    append("spi_90d", ("time", "y", "x"), seasonal_anomaly(c["precip_sum_90d"].values, doy, causal=True),
           {"description": "Standardized 90-day precip anomaly vs PRIOR-year day-of-year climatology (causal SPI proxy)."})
    append("ndvi_anomaly", ("time", "y", "x"), seasonal_anomaly(c["NDVI"].values, doy, causal=True),
           {"description": "NDVI anomaly vs PRIOR-year day-of-year climatology (causal z-score)."})
    append("lai_anomaly", ("time", "y", "x"), seasonal_anomaly(c["LAI"].values, doy, causal=True),
           {"description": "LAI anomaly vs PRIOR-year day-of-year climatology (causal z-score)."})

    # --- fire-weather / fuel / vegetation (pointwise, from coarse vars) ---
    emc_peak = equilibrium_moisture_content(c["t2m_max"].values, c["RH_min"].values)
    append("emc_peak", ("time", "y", "x"), emc_peak.astype("float32"),
           {"units": "percent", "description": "Equilibrium (1-hr dead-fuel) moisture from t2m_max/RH_min; lower=drier."})
    append("ffwi", ("time", "y", "x"),
           fosberg_ffwi(emc_peak, c["wind_speed_max"].values).astype("float32"),
           {"description": "Fosberg Fire Weather Index (EMC_peak + wind_speed_max); higher=worse."})
    del emc_peak
    vpd_peak = vpd_kpa(c["t2m_max"].values, c["RH_min"].values)
    append("vpd_peak", ("time", "y", "x"), vpd_peak.astype("float32"),
           {"units": "kPa", "description": "Vapor Pressure Deficit (VPD) from t2m_max/RH_min; higher=drier."})
    hdw = hdw_index(vpd_peak, c["wind_speed_max"].values)
    append("hdw", ("time", "y", "x"), hdw.astype("float32"),
           {"description": "Hot-Dry-Windy Index (VPD + wind_speed_max); higher=worse."})
    append("fvc", ("time", "y", "x"),
           fractional_vegetation_cover(c["NDVI"].values).astype("float32"),
           {"description": "Fractional vegetation cover [0,1] from NDVI (Carlson & Ripley)."})

    # --- WUI proximity: distance (km) to nearest 'urban' cell (CLC_2018 artificial > 0.5) ---
    from scipy.ndimage import distance_transform_edt
    cell_km = abs(float(c["x"].values[1] - c["x"].values[0])) / 1000.0
    art = c["CLC_2018_artificial_proportion"].values
    urban = np.where(np.isfinite(art), art > 0.5, False)
    if urban.any():
        dist_urban = distance_transform_edt(~urban).astype("float32") * cell_km
    else:
        dist_urban = np.full(art.shape, np.hypot(*art.shape) * cell_km, dtype="float32")
    dist_urban = np.where(np.isfinite(art), dist_urban, np.nan).astype("float32")
    append("dist_to_urban", ("y", "x"), dist_urban,
           {"units": "km", "description": "WUI proxy: distance to nearest CLC_2018 artificial(>0.5) cell."})

    # --- aspect orientation + heat-load. aspect_1..8 = 0-45,45-90,...,315-360 deg
    #     => sector CENTERS at 22.5 + k*45 (0deg = North, clockwise). ---
    centers = np.deg2rad(np.arange(8) * 45.0 + 22.5)
    asp = np.stack([c[f"aspect_{i}"].values for i in range(1, 9)], axis=0)  # (8,y,x) fractions
    east = np.nansum(asp * np.sin(centers)[:, None, None], axis=0)
    north = np.nansum(asp * np.cos(centers)[:, None, None], axis=0)
    finite_asp = np.isfinite(c["aspect_1"].values)
    append("aspect_southness", ("y", "x"), np.where(finite_asp, -north, np.nan).astype("float32"),
           {"description": "Continuous southness from aspect one-hots (+1=S, -1=N); sector centers 22.5+k*45."})
    append("aspect_eastness", ("y", "x"), np.where(finite_asp, east, np.nan).astype("float32"),
           {"description": "Continuous eastness from aspect one-hots (+1=E, -1=W)."})
    # HLI: reconstruct a continuous aspect angle, add latitude (pyproj) + slope.
    from pyproj import Transformer
    aspect_deg = np.mod(np.degrees(np.arctan2(east, north)), 360.0)
    xx, yy = np.meshgrid(c["x"].values, c["y"].values)
    _, lat = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True).transform(xx, yy)
    hli = heat_load_index(c["slope_mean"].values, aspect_deg, lat)
    append("hli", ("y", "x"), np.where(finite_asp, hli, np.nan).astype("float32"),
           {"description": "McCune-Keon Heat Load Index (slope + reconstructed aspect + latitude); terrain solar load."})

    # --- fire history (reuse dryness/rolling helpers on is_fire) ---
    fire = c["is_fire"].values.astype("float32")
    append("time_since_last_fire", ("time", "y", "x"),
           days_since_rain(fire, threshold=0.5).astype("float32"),
           # NB: no units="days" attr — it makes xarray decode this as timedelta64, not float32.
           {"description": "Consecutive days (count) since the cell last burned (0 on fire days)."})
    append("burn_frequency_365d", ("time", "y", "x"),
           rolling_sum_time(fire, 365).astype("float32"),
           {"description": "Number of fire-days in the trailing 365 days (NaN for first 364)."})

    # --- calendar features (doy, dow, holidays) ---
    dates = c["time"].values; dates_tp1 = dates + np.timedelta64(1, "D")
    pd_dates = pd.DatetimeIndex(dates)
    pd_dates_tp1 = pd.DatetimeIndex(dates_tp1)

    # The model predicts is_fire(t+1) from the row at t, so we expose calendar context for BOTH the feature day t
    # AND the target day t+1. Per-day scalars are written as constant (time,y,x) planes via zero-copy broadcast:
    # zstd crushes a constant-per-day plane to ~nothing, and the (time,y,x) shape needs no trainer special-casing.
    import holidays as _hol

    T = len(dates)
    dow = pd_dates.dayofweek
    doy_tp1 = pd_dates_tp1.dayofyear
    dow_tp1 = pd_dates_tp1.dayofweek

    def _plane(vec):
        """(time,) per-day scalar -> constant-in-space (time,y,x) float32 view (zero-copy broadcast)."""
        return np.broadcast_to(np.asarray(vec, "float32")[:, None, None], (T, ny, nx))

    # cyclic sin/cos: doy for the target day t+1; dow for BOTH t and t+1 (sincos returns a (sin, cos) tuple).
    doy_sin, doy_cos = day_of_year_sincos(doy_tp1)
    dow_sin, dow_cos = day_of_week_sincos(dow)
    dow_sin_tp1, dow_cos_tp1 = day_of_week_sincos(dow_tp1)
    for name, vec, desc in [
        ("doy_sin", doy_sin, "Day-of-year sine (target day t+1; cyclic, no New-Year seam)."),
        ("doy_cos", doy_cos, "Day-of-year cosine (target day t+1)."),
        ("dow_sin", dow_sin, "Day-of-week sine (feature day t; weekly human-ignition rhythm)."),
        ("dow_cos", dow_cos, "Day-of-week cosine (feature day t)."),
        ("dow_sin_tp1", dow_sin_tp1, "Day-of-week sine (target day t+1)."),
        ("dow_cos_tp1", dow_cos_tp1, "Day-of-week cosine (target day t+1)."),
    ]:
        append(name, ("time", "y", "x"), _plane(vec), {"description": desc})

    # Holidays. national = Spain-wide (constant plane); regional = community-specific EXTRA days (subdiv holidays
    # MINUS the national set, so channels are non-redundant), painted by AutonomousCommunities code. Both for t
    # and t+1. CCAA_TO_SUBDIV (code -> ISO 3166-2:ES) is imported from src.data.regions (shared with update_edge).
    years = range(int(pd_dates.year.min()), int(pd_dates_tp1.year.max()) + 1)
    nat_days = set(_hol.Spain(years=years).keys())

    def _is_in(idx, dayset):
        """boolean (time,): is each timestamp's calendar date in `dayset`?"""
        return np.array([ts.date() in dayset for ts in idx], dtype=bool)

    append("is_holiday_national", ("time", "y", "x"), _plane(_is_in(pd_dates, nat_days)),
           {"description": "1 if day t is a Spain-wide national public holiday."})
    append("is_holiday_national_tp1", ("time", "y", "x"), _plane(_is_in(pd_dates_tp1, nat_days)),
           {"description": "1 if the target day t+1 is a Spain-wide national public holiday."})

    ac = np.rint(np.nan_to_num(c["AutonomousCommunities"].values)).astype(int)  # (y,x) region codes
    for tag, idx in (("", pd_dates), ("_tp1", pd_dates_tp1)):
        reg = np.zeros((T, ny, nx), dtype=bool)
        for code, sub in CCAA_TO_SUBDIV.items():
            mask = ac == code
            if not mask.any():
                continue
            reg_only = set(_hol.Spain(subdiv=sub, years=years).keys()) - nat_days
            reg |= _is_in(idx, reg_only)[:, None, None] & mask[None, :, :]
        append(f"is_holiday_regional{tag}", ("time", "y", "x"), reg.astype("float32"),
               {"description": "1 if the day (t or t+1) is a region-specific autonomous-community holiday "
                               "(beyond national); painted by AutonomousCommunities code."})
        del reg

    print("[enrich] done.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build all engineered features on a coarse cube, in order "
                                             "(fire_context THEN engineered).")
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--cube", type=str, default="FireGuard", help="Cube name (default: FireGuard).")
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if the vars already exist.")
    args = ap.parse_args()

    path = project_root / "data" / "gold" / f"{args.cube}_coarse{args.factor}.zarr"
    if not path.exists():
        raise FileNotFoundError(path)
    fire_context(path, args.overwrite)   # FIRST — produces precip_sum_* + dist_to_fire
    engineered(path, args.overwrite)     # consumes precip_sum_90d for spi_90d


if __name__ == "__main__":
    main()
