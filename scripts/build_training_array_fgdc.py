"""
Builds a training array for the FGDC cube and HistGBT.
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
from src.data.features_fireguard import FGDC_FEATURE_VARS

CODEC = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)  # fast decompress for read-heavy training


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--cube", type=str, default="FireGuard", help="path to cube zarr (overrides factor)")
    args = ap.parse_args()

    cube = project_root / "data" / "gold" / f"{args.cube}_coarse{args.factor}.zarr"
    out = project_root / "data" / "gold" / f"{args.cube}_coarse{args.factor}_dyn.zarr"
    feats = FGDC_FEATURE_VARS

    # open the cube directly as z
    z = xr.open_zarr(cube, consolidated=True)

    # define dynamic variables
    dyn_names = [f for f in feats if f in z and "time" in z[f].dims]

    T = int(z.sizes["time"]); H, W = z.sizes["y"], z.sizes["x"]; Cd = len(dyn_names)

    if out.exists():
        import shutil; shutil.rmtree(out)

    time_coord = z["time"]
    first = True
    for t0 in range(0, T, args.block):
        t1 = min(t0 + args.block, T)
        bd = np.empty((t1 - t0, Cd, H, W), dtype="float16")

        for k, t in enumerate(range(t0, t1)):
            bd[k] = np.stack([z[f].isel(time=t).values for f in dyn_names], axis=0).astype("float16")

        block = xr.Dataset(
            {"dyn": (("time", "channel", "y", "x"), bd)},
            coords={"time": time_coord.isel(time=slice(t0, t1)),
                    "channel": np.array(dyn_names, dtype=object),
                    "y": z["y"], "x": z["x"]},
        )
        enc = {"dyn": {"compressor": CODEC}}
        if first:
            block.attrs["dyn_features"] = ",".join(dyn_names)
            block.attrs["normalized"] = "false"
            block.to_zarr(out, mode="w", encoding=enc, consolidated=True, zarr_format=2)
            first = False
        else:
            block.to_zarr(out, mode="a", append_dim="time", consolidated=True, zarr_format=2)
        print(f"  {t1}/{T}", flush=True)

    print(f"[stack] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
