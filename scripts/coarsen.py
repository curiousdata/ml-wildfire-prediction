"""Coarsen the silver IberFire cube to a gold analysis cube — semantic pooling +
inline feature engineering.

Reads silver (1 km) READ-ONLY (never modified) and writes
``data/gold/IberFire_coarse{F}.zarr``.

=============================================================================
MAP SAFETY (read before editing) — the Streamlit monolith georeferences cells
purely from the cube's ``x``/``y`` DIM COORDINATES (EPSG:3035 metres), via
min/max -> ``rasterio.from_bounds`` -> reproject to WGS84. The coords MUST stay
in the convention the app already renders correctly at coarse32, namely
block-MEAN cell centres. That is exactly what ``xarray.coarsen`` produces:
``coord_func`` defaults to ``"mean"`` for ALL reductions (.mean/.max/.min), so
every variable group below shares identical block-mean ``x``/``y`` coords.
  -> verified: coarse32 ``x[0]`` == ``mean(silver x[0:32])``.
DO NOT override ``coord_func``, post-edit ``x``/``y``, or mix coarsen factors
across groups, or the cells will shift on the map.
=============================================================================

Per-variable pooling (NOT mean-everything):
  is_fire (label)            -> max   (any sub-cell fire => fire; preserves rare positives)
  *_max                      -> max
  *_min                      -> min
  wind_direction_*           -> DROPPED; replaced by mean-pooled wind u/v components
  AutonomousCommunities      -> mode  (categorical region code; mean would be invalid)
  x_index/y_index/x_coordinate/y_coordinate -> DROPPED (grid metadata; the x/y DIM
                                coords carry georeferencing). Absolute position is also
                                a memorisation risk as a feature.
  is_near_fire               -> DROPPED (leakage-removed; superseded by the §E
                                dist_to_fire feature computed AFTER coarsening)
  everything else            -> mean  (incl. CLC/aspect one-hots -> fractional composition)

Derived features computed from the COARSE variables (src/data/feature_engineering.py):
  wind_u_mean / wind_v_mean              (from wind_speed_mean + wind_direction_mean, at 1 km then pooled)
  wind_u_atmaxspeed / wind_v_atmaxspeed  (from wind_speed_max  + wind_direction_at_max_speed)
  VPD_mean = VPD(t2m_mean, RH_mean)
  VPD_peak = VPD(t2m_max,  RH_min)       (hottest+driest corner of the cell)
  HDW      = VPD_peak * wind_speed_max
  precip_sum_{7,30,90}d                  (trailing rolling sums of coarse precip)
  days_since_rain                        (dry-spell length on coarse precip)
  doy_sin/doy_cos, dow_sin/dow_cos       (time-only -> stored as (time,) vars, not broadcast)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from numcodecs import Blosc

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.feature_engineering import (
    ANTECEDENT_WINDOWS_DAYS,
    DRY_DAY_THRESHOLD_MM,
    day_of_week_sincos,
    day_of_year_sincos,
    days_since_rain,
    hdw_index,
    vpd_kpa,
    wind_to_uv,
)

SILVER = project_root / "data" / "silver" / "IberFire.zarr"
OUT_DIR = project_root / "data" / "gold"
COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=2)

LABEL_VAR = "is_fire"
WIND_DIR_VARS = {"wind_direction_mean", "wind_direction_at_max_speed"}
CATEGORICAL_VARS = {"AutonomousCommunities"}
DROP_VARS = {"x_index", "y_index", "x_coordinate", "y_coordinate", "is_near_fire"}

# LST (Kelvin) carries ~0.07% physically-impossible cloud/edge artifacts from its
# multi-source satellite origin (min ~156 K, max ~409 K). Clip to a physical land-
# surface band BEFORE pooling so 4x4 mean-pooling doesn't ingest the garbage.
# (Real values span ~262-314 K; this only touches the artifact tails.)
LST_CLIP_K = (250.0, 340.0)


def classify(name: str) -> str:
    """Pooling op for a silver variable: 'mean' | 'max' | 'min' | 'special'."""
    if name in WIND_DIR_VARS or name in CATEGORICAL_VARS or name in DROP_VARS:
        return "special"  # handled out-of-band (or dropped)
    if name == LABEL_VAR or name.endswith("_max"):
        return "max"
    if name.endswith("_min"):
        return "min"
    return "mean"


def mode_pool_static(arr2d: np.ndarray, factor: int) -> np.ndarray:
    """Mode-pool a 2D (y, x) categorical field over factor×factor blocks.

    Scipy-free and robust for small cardinality (Spain: ~19 communities + NaN sea).
    All-NaN blocks -> NaN.
    """
    ny, nx = arr2d.shape
    if ny % factor or nx % factor:
        raise ValueError("mode_pool_static requires dims divisible by factor")
    blocks = (
        arr2d.reshape(ny // factor, factor, nx // factor, factor)
        .transpose(0, 2, 1, 3)
        .reshape(ny // factor, nx // factor, factor * factor)
    )
    codes = np.unique(blocks[np.isfinite(blocks)])
    best_count = np.zeros(blocks.shape[:2], dtype=np.int64)
    best_code = np.full(blocks.shape[:2], np.nan, dtype="float32")
    for code in codes:
        cnt = (blocks == code).sum(axis=2)
        upd = cnt > best_count
        best_code[upd] = code
        best_count[upd] = cnt[upd]
    return best_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Coarsen silver -> gold (semantic pooling + engineered features).")
    parser.add_argument("--factor", type=int, default=4, help="Spatial coarsening factor (default: 4).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output Zarr if it exists.")
    parser.add_argument("--max-time", type=int, default=0,
                        help="Debug/smoke: keep only the first N time steps (0 = all).")
    args = parser.parse_args()
    F = args.factor

    if not SILVER.exists():
        raise FileNotFoundError(f"Source Zarr not found: {SILVER}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"IberFire_coarse{F}.zarr"
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} already exists. Re-run with --overwrite to replace it.")
        shutil.rmtree(out)

    ds = xr.open_zarr(SILVER, consolidated=True, decode_times=True)
    if args.max_time:
        ds = ds.isel(time=slice(0, args.max_time))
        print(f"[coarsen] DEBUG: restricted to first {args.max_time} time steps")

    # LST quality clip (see LST_CLIP_K). Reassigns the in-memory lazy var only;
    # silver on disk is never modified.
    if "LST" in ds:
        lo, hi = LST_CLIP_K
        ds["LST"] = ds["LST"].clip(min=lo, max=hi)
        print(f"[coarsen] LST clipped to physical band {LST_CLIP_K} K before pooling")

    ckw = dict(y=F, x=F, boundary="trim")  # same factor+boundary for every group (map-safe coords)

    allv = list(ds.data_vars)
    mean_vars = [v for v in allv if classify(v) == "mean"]
    max_vars = [v for v in allv if classify(v) == "max"]
    min_vars = [v for v in allv if classify(v) == "min"]

    print(f"[coarsen] factor={F}  mean={len(mean_vars)} max={len(max_vars)} min={len(min_vars)} "
          f"special={len(allv) - len(mean_vars) - len(max_vars) - len(min_vars)}")

    # --- semantic spatial pooling (coords reduced by coord_func='mean' for all) ---
    parts = [
        ds[mean_vars].coarsen(**ckw).mean(),
        ds[max_vars].coarsen(**ckw).max(),
        ds[min_vars].coarsen(**ckw).min(),
    ]
    coarse = xr.merge(parts)

    # --- wind: reconstruct u/v at 1 km (can't pool a direction), then mean-pool ---
    u_mean, v_mean = wind_to_uv(ds["wind_speed_mean"], ds["wind_direction_mean"])
    u_amx, v_amx = wind_to_uv(ds["wind_speed_max"], ds["wind_direction_at_max_speed"])
    wind = xr.Dataset(
        {"wind_u_mean": u_mean, "wind_v_mean": v_mean,
         "wind_u_atmaxspeed": u_amx, "wind_v_atmaxspeed": v_amx}
    ).coarsen(**ckw).mean()
    coarse = xr.merge([coarse, wind])

    # --- AutonomousCommunities: mode-pool (categorical) ---
    if "AutonomousCommunities" in ds:
        ac = mode_pool_static(ds["AutonomousCommunities"].values.astype("float32"), F)
        coarse["AutonomousCommunities"] = (("y", "x"), ac)

    # --- derived features from the COARSE variables ---
    coarse["VPD_mean"] = vpd_kpa(coarse["t2m_mean"], coarse["RH_mean"])
    coarse["VPD_peak"] = vpd_kpa(coarse["t2m_max"], coarse["RH_min"])
    coarse["HDW"] = hdw_index(coarse["VPD_peak"], coarse["wind_speed_max"])

    cp = coarse["total_precipitation_mean"]
    for w in ANTECEDENT_WINDOWS_DAYS:
        coarse[f"precip_sum_{w}d"] = cp.rolling(time=w, min_periods=w).sum()

    # days_since_rain is a non-linear cumulative-along-time op -> compute eagerly on the
    # (small) coarse precip series, defined on the coarse grid. Tiled over x and kept in
    # float32 to bound peak memory (the int cumsum is the hog).
    cp_vals = cp.transpose("time", "y", "x").astype("float32").values
    nt, ny, nx = cp_vals.shape
    dsr = np.empty((nt, ny, nx), dtype="float32")
    step = max(1, nx // 6)
    for x0 in range(0, nx, step):
        dsr[:, :, x0:x0 + step] = days_since_rain(
            cp_vals[:, :, x0:x0 + step], threshold=DRY_DAY_THRESHOLD_MM
        ).astype("float32")
    coarse["days_since_rain"] = (("time", "y", "x"), dsr)
    del cp_vals, dsr

    # --- calendar features (time-only; stored as (time,) — broadcast at read time) ---
    doy_sin, doy_cos = day_of_year_sincos(coarse["time"].dt.dayofyear.values)
    dow_sin, dow_cos = day_of_week_sincos(coarse["time"].dt.dayofweek.values)
    for nm, a in [("doy_sin", doy_sin), ("doy_cos", doy_cos),
                  ("dow_sin", dow_sin), ("dow_cos", dow_cos)]:
        coarse[nm] = (("time",), a.astype("float32"))

    # --- attrs, chunking, write ---
    coarse.attrs.update(ds.attrs)
    coarse.attrs["coarsen_factor"] = F
    coarse.attrs["coarsen_pooling"] = "semantic: label=max, *_max=max, *_min=min, wind->u/v, AC=mode, else=mean"

    chunks = {"time": 1, "y": coarse.sizes["y"], "x": coarse.sizes["x"]}
    coarse = coarse.chunk(chunks)
    encoding = {n: {"compressor": COMPRESSOR} for n in coarse.data_vars}

    print(f"[coarsen] writing {out}  dims={dict(coarse.sizes)}  vars={len(coarse.data_vars)}")
    with ProgressBar():
        # zarr_format=2: matches the existing silver/coarse cubes and accepts the
        # numcodecs.Blosc compressor (zarr 3.x's default v3 format rejects it).
        coarse.to_zarr(out, mode="w", encoding=encoding, consolidated=True, zarr_format=2)
    print("[coarsen] done:", out)


if __name__ == "__main__":
    main()
