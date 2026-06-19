"""Materialize a training-optimized DYNAMIC feature stack for fast loading.

Per training sample, only the ~48 dynamic features hit disk (statics are RAM-cached,
calendar is broadcast). Reading them as 48 separate zarr chunks is the I/O bottleneck
(GPU ~24% utilized). This pre-stacks them into ONE chunk per day:

  dyn    (time, channel, y, x)  float16, PRE-NORMALIZED (train stats), Blosc-lz4, chunks (1,C,y,x)
  regime (time, y, x)           int8  (0=sea, 1=ignition, 2=spread)

So a day = 1 contiguous read + 1 decompress (vs 48), in half the bytes (fp16), with no
per-sample normalization. The loader (StackedRegimeIberFireDataset) reads `dyn[t]` for the
dynamic channels and assembles RAM-cached statics + broadcast calendar around them.

Reuses RegimeIberFireDataset._build_X so the stored values exactly match what the model
would otherwise compute. One-time (~40 min). Derived artifact (no new information).

Usage:  python scripts/build_training_array.py [--factor 4] [--block 100]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from numcodecs import Blosc

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from src.data.datasets import RegimeIberFireDataset
from src.data.features import build_segmentation_features

CODEC = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)  # fast decompress for read-heavy training


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--block", type=int, default=100)
    args = ap.parse_args()

    cube = project_root / "data" / "gold" / f"IberFire_coarse{args.factor}.zarr"
    stats = project_root / "stats" / f"coarse{args.factor}_norm_stats_train.json"
    out = project_root / "data" / "gold" / f"IberFire_coarse{args.factor}_dyn.zarr"
    feats = build_segmentation_features(xr.open_zarr(cube, consolidated=True).data_vars)

    ds = RegimeIberFireDataset(zarr_path=cube, time_start="2007-01-01", time_end="2025-01-01",
                               feature_vars=feats, label_var="is_fire", lead_time=1,
                               compute_stats=False, stats_path=stats, mode="all")
    dyn_idx = [i for i, v in enumerate(feats) if v in ds.dynamic_vars]
    dyn_names = [feats[i] for i in dyn_idx]
    T = int(ds.ds.sizes["time"]); H, W = ds.H, ds.W; Cd = len(dyn_idx)
    dist = ds.root["dist_to_fire"]
    print(f"[stack] T={T} days, C_dyn={Cd}, grid {H}x{W} -> {out.name}", flush=True)

    if out.exists():
        import shutil; shutil.rmtree(out)

    time_coord = ds.ds["time"]
    first = True
    for t0 in range(0, T, args.block):
        t1 = min(t0 + args.block, T)
        bd = np.empty((t1 - t0, Cd, H, W), dtype="float16")
        br = np.empty((t1 - t0, H, W), dtype="int8")
        for k, t in enumerate(range(t0, t1)):
            X = ds._build_X(t)  # (C, H, W) normalized — reuses the validated assembly
            bd[k] = X[dyn_idx].astype("float16")
            d = np.asarray(dist[t, :, :], dtype="float32")
            near = np.isfinite(d) & (d <= ds.regime_dist_km)
            br[k] = np.where(ds.land_mask, np.where(near, 2, 1), 0).astype("int8")
        block = xr.Dataset(
            {"dyn": (("time", "channel", "y", "x"), bd), "regime": (("time", "y", "x"), br)},
            coords={"time": time_coord.isel(time=slice(t0, t1)),
                    "channel": np.array(dyn_names, dtype=object),
                    "y": ds.ds["y"], "x": ds.ds["x"]},
        )
        enc = {"dyn": {"compressor": CODEC}, "regime": {"compressor": CODEC}}
        if first:
            block.attrs["dyn_features"] = ",".join(dyn_names)
            block.attrs["normalized"] = "true"
            block.attrs["regime_dist_km"] = float(ds.regime_dist_km)
            block.to_zarr(out, mode="w", encoding=enc, consolidated=True, zarr_format=2)
            first = False
        else:
            block.to_zarr(out, mode="a", append_dim="time", consolidated=True, zarr_format=2)
        print(f"  {t1}/{T}", flush=True)

    print(f"[stack] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
