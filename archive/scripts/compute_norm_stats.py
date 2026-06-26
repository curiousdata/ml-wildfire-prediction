"""Per-feature normalization stats (mean/std) for the coarse cube, TRAIN split only.

Computed over LAND cells (finite t2m_mean) on the training period to avoid val/test
leakage. One mean/std per data variable (year-aware CLC/popdens get per-year stats;
the dataset resolves the right year + its stats at read time). The label (`is_fire`)
and the categorical region code (`AutonomousCommunities`) are excluded — not normalized
continuous features.

Output: stats/coarse4_norm_stats_train.json  ->  {var: {mean, std, n_finite}}

Usage:  python scripts/compute_norm_stats.py [--factor 4] [--n-days 250]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

project_root = Path(__file__).resolve().parents[1]
TRAIN = ("2008-01-01", "2018-12-31")
SKIP = {"is_fire", "AutonomousCommunities"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--n-days", type=int, default=250)
    args = ap.parse_args()

    cube = project_root / "data" / "gold" / f"IberFire_coarse{args.factor}.zarr"
    out = project_root / "stats" / f"coarse{args.factor}_norm_stats_train.json"
    c = xr.open_zarr(cube, consolidated=True)

    time = c["time"].values
    train_idx = np.where((time >= np.datetime64(TRAIN[0])) & (time <= np.datetime64(TRAIN[1])))[0]
    sample_days = np.unique(np.linspace(train_idx[0], train_idx[-1], args.n_days).astype(int))
    land = np.isfinite(c["t2m_mean"].isel(time=int(sample_days[len(sample_days) // 2])).values)
    print(f"[stats] {cube.name}: train {TRAIN[0]}..{TRAIN[1]}, {sample_days.size} sampled days, "
          f"{int(land.sum())} land cells", flush=True)

    sel = xr.DataArray(sample_days, dims="t")
    stats, skipped = {}, []
    for k, v in enumerate(sorted(c.data_vars)):
        if v in SKIP:
            skipped.append(v); continue
        da = c[v]
        if "time" in da.dims and {"y", "x"} <= set(da.dims):
            arr = da.isel(time=sel).values.astype("float64")          # (t,y,x)
            vals = arr[:, land]                                       # land cells, all sampled days
        elif set(da.dims) == {"y", "x"}:
            vals = da.values.astype("float64")[land]
        elif set(da.dims) == {"time"}:
            vals = da.isel(time=sel).values.astype("float64")        # calendar (time,)
        else:
            skipped.append(v); continue
        finite = np.isfinite(vals)
        if finite.sum() == 0:
            skipped.append(v); continue
        mean = float(np.nanmean(vals[finite]))
        std = float(np.nanstd(vals[finite]))
        stats[v] = {"mean": mean, "std": max(std, 1e-6), "n_finite": int(finite.sum())}
        if k % 40 == 0:
            print(f"  {k}/{len(c.data_vars)}", flush=True)

    out.write_text(json.dumps(stats, indent=2))
    print(f"\n[stats] {len(stats)} vars -> {out}  | skipped: {skipped}", flush=True)
    print("\nsanity (known vars):", flush=True)
    for v in ["t2m_mean", "RH_mean", "wind_speed_mean", "total_precipitation_mean", "NDVI",
              "FWI", "kbdi", "elevation_mean", "doy_sin", "dist_to_fire"]:
        if v in stats:
            print(f"  {v:24} mean={stats[v]['mean']:10.3f} std={stats[v]['std']:10.3f}", flush=True)


if __name__ == "__main__":
    main()
