"""Compute the Group-A SOTA-gap-fill features on a coarse cube and append them.

Implemented here (validated functions in src/data/feature_engineering.py):
  kbdi                  - Keetch-Byram Drought Index (daily_rain ~= 2.9*tp, calibrated to AEMET).
  spi_90d               - standardized 90-day precip anomaly vs day-of-year climatology (SPI proxy).
  ndvi_anomaly, lai_anomaly - greenness anomaly vs seasonal climatology.
  tpi, terrain_curvature    - DEM-derived terrain position / curvature (static).
  time_since_last_fire  - days since the cell last burned (days_since_rain on is_fire).
  burn_frequency_365d   - fire-days in the trailing 365 days.

Deferred (need extra deps/handling, NOT here): WUI proximity (year-aware CLC), TWI (flow
accumulation), HLI/solar-insolation (per-pixel latitude -> pyproj, absent locally).

Appends each variable incrementally (mode='a') to bound memory. Idempotent-ish: use --overwrite
to recompute. Usage:  python scripts/add_engineered_features.py --factor 4 [--overwrite]
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
    day_of_week_sincos
)

COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=2)
TP_TO_DAILY_MM = 2.9  # calibrated cumulative-mean -> daily-total factor (AEMET national ~640 mm/yr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cube", type=str, default="IberFire", help="Name of the cube to append to (default: IberFire).")

    args = ap.parse_args()
    factor, cube, overwrite = args.factor, args.cube, args.overwrite

    args = ap.parse_args()
    path = project_root / "data" / "gold" / f"{cube}_coarse{factor}.zarr"
    c = xr.open_zarr(path, consolidated=True)

    coords = {"time": c["time"], "y": c["y"], "x": c["x"]}
    ny, nx = c.sizes["y"], c.sizes["x"]
    doy = c["time"].dt.dayofyear.values
    nyears = (c["time"].values[-1] - c["time"].values[0]).astype("timedelta64[D]").astype(int) / 365.25

    existing = set(c.data_vars)

    def append(name, dims, data, attrs=None):
        if name in existing and not overwrite:
            print(f"  skip {name} (exists; use --overwrite)"); return
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
    daily_rain = (TP_TO_DAILY_MM * tp).astype("float64")
    R = np.nansum(daily_rain, axis=0) / nyears
    tmax = c["t2m_max"].values
    append("kbdi", ("time", "y", "x"),
           keetch_byram_drought_index(daily_rain, tmax, R).astype("float32"),
           {"units": "mm", "description": f"Keetch-Byram Drought Index (daily_rain={TP_TO_DAILY_MM}*tp, approx)."})
    del tp, daily_rain, tmax

    # --- seasonal anomalies (SPI proxy + greenness) — CAUSAL climatology (prior years only): no train/test
    #     leakage and identical to what's available at serve time (CHANGES.md / code-review #1). ---
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
    hdw = hdw_index(c["t2m_max"].values, c["RH_min"].values, c["wind_speed_max"].values)
    append("hdw", ("time", "y", "x"), hdw.astype("float32"),
           {"description": "High-Desert Weather Index (t2m_max + RH_min + wind_speed_max); higher=worse."})
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

    doy = pd_dates.dayofyear; dow = pd_dates.dayofweek
    doy_tp1 = pd_dates_tp1.dayofyear; dow_tp1 = pd_dates_tp1.dayofweek

    append("day_of_year_sincos", ("time",), np.broadcast_to(v[:, None, None], (T, ny, nx))
           # day_of_year_sincos(doy).astype("float32"), (len(doy), 2)),
           # TODO: for correct memory-efficient broadcast with np
           {"description": "Day-of-year sine/cosine encoding (2 channels)."})
    append("day_of_week_sincos", ("time",), day_of_week_sincos(dow).astype("float32"),
           {"description": "Day-of-week sine/cosine encoding (2 channels)."})
    append("day_of_year_sincos_tp1", ("time",), day_of_year_sincos(doy_tp1).astype("float32"),
           {"description": "Day-of-year sine/cosine encoding (2 channels) for the next day."})
    append("day_of_week_sincos_tp1", ("time",), day_of_week_sincos(dow_tp1).astype("float32"),
           {"description": "Day-of-week sine/cosine encoding (2 channels) for the next day."})


    print("[enrich] done.")


if __name__ == "__main__":
    main()
