"""Compute the §E spatial fire-context features on a coarse cube and append them.

Resolution-coupled features (see src/data/feature_engineering.fire_distance_and_exposure):
  - dist_to_fire           : km to the nearest fire cell that day (0 on fire cells).
  - fire_upwind_exposure   : (W . d)/|d|^2 downwind-exposure (>0 downwind of a nearby fire).

Causal: derived from is_fire(t), used to predict is_fire(t+1). Appended to the existing
gold cube in place (mode='a'); silver is untouched.

Usage:  python scripts/add_fire_context.py --factor 4 [--overwrite]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from numcodecs import Blosc

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.data.feature_engineering import fire_distance_and_exposure

COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=2)
NEW_VARS = ("dist_to_fire", "fire_upwind_exposure")


def main() -> None:
    ap = argparse.ArgumentParser(description="Append §E fire-context features to a coarse cube.")
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true", help="Re-append even if the vars already exist.")
    args = ap.parse_args()

    path = project_root / "data" / "gold" / f"IberFire_coarse{args.factor}.zarr"
    if not path.exists():
        raise FileNotFoundError(path)

    c = xr.open_zarr(path, consolidated=True)
    if any(v in c.data_vars for v in NEW_VARS) and not args.overwrite:
        raise SystemExit(f"{NEW_VARS} already present in {path}. Use --overwrite.")

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
        dist[t], expo[t] = fire_distance_and_exposure(
            fire_all[t], u, v, xc, yc, no_fire_dist_km
        )
        if t % 1000 == 0:
            print(f"  {t}/{nt}")

    ctx = xr.Dataset(
        {"dist_to_fire": (("time", "y", "x"), dist),
         "fire_upwind_exposure": (("time", "y", "x"), expo)},
        coords={"time": c["time"], "y": c["y"], "x": c["x"]},
    )
    ctx["dist_to_fire"].attrs = {"units": "km", "description": "Distance to nearest fire cell on day t (0 on fire cells)."}
    ctx["fire_upwind_exposure"].attrs = {"description": "(W.d)/|d|^2 downwind-exposure to nearest fire; >0 downwind, <0 upwind."}
    ctx = ctx.chunk({"time": 1, "y": ny, "x": nx})
    enc = {v: {"compressor": COMPRESSOR} for v in ctx.data_vars}

    print("[fire-context] appending to cube...")
    with ProgressBar():
        ctx.to_zarr(path, mode="a", encoding=enc, consolidated=True, zarr_format=2)
    print("[fire-context] done.")


if __name__ == "__main__":
    main()
