"""Pre-stack the FGDC cube's DYNAMIC features into one chunk per day for fast training reads.

Output: data/gold/<cube>_coarse<F>_dyn.zarr  — dyn(time, channel, y, x) float16, RAW (not normalized;
HistGBT splits on raw values), chunks (1, C, y, x), lz4. `channel` = the dynamic FGDC_FEATURE_VARS in
frozen order (the trainer maps columns by this). Statics aren't stored — the trainer RAM-caches them.
Derived artifact (no new info); rebuild when the feature set changes. ~30 min, one-time.
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
    # parse arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--factor", type=int, default=4)
    argument_parser.add_argument("--block", type=int, default=100)
    argument_parser.add_argument("--cube", type=str, default="FireGuard", help="path to cube zarr (overrides factor)")
    args = argument_parser.parse_args()

    # define input/output paths and features
    input_path = project_root / "data" / "gold" / f"{args.cube}_coarse{args.factor}.zarr"
    output_path = project_root / "data" / "gold" / f"{args.cube}_coarse{args.factor}_dyn.zarr"
    feature_names = FGDC_FEATURE_VARS

    # open the cube directly as zarr_open
    zarr_opened = xr.open_zarr(input_path, consolidated=True)

    # define dynamic variables
    dynamic_feature_names = [
        feature for feature in feature_names if feature in zarr_opened and "time" in zarr_opened[feature].dims  
    ]

    # get input sizes: time, height, width, number of dynamic features
    time_size = int(zarr_opened.sizes["time"])
    height_size, width_size = zarr_opened.sizes["y"], zarr_opened.sizes["x"]
    dynamic_channel_size = len(dynamic_feature_names)

    if output_path.exists():
        import shutil; shutil.rmtree(output_path)

    # build the training array in blocks of time
    time_coord = zarr_opened["time"]
    is_first_block = True

    for t0 in range(0, time_size, args.block):
        t1 = min(t0 + args.block, time_size)
        block_array = np.empty((t1 - t0, dynamic_channel_size, height_size, width_size), dtype="float16")

        # stack the dynamic features for the current block of time
        for k, t in enumerate(range(t0, t1)):
            array = np.stack([zarr_opened[f].isel(time=t).values for f in dynamic_feature_names], axis=0)
            block_array[k] = np.clip(array, -6e4, 6e4).astype("float16")  # clip to avoid overflow in float16

        block = xr.Dataset(
            {"dyn": (("time", "channel", "y", "x"), block_array)},
            coords={"time": time_coord.isel(time=slice(t0, t1)),
                    "channel": np.array(dynamic_feature_names, dtype=object),
                    "y": zarr_opened["y"], "x": zarr_opened["x"]},
        )
        # define encoding for the dynamic variable and write to zarr
        enc = {"dyn": {"compressor": CODEC}}
        if is_first_block:
            block.attrs["dyn_features"] = ",".join(dynamic_feature_names)
            block.attrs["normalized"] = "false"
            block.to_zarr(output_path, mode="w", encoding=enc, consolidated=True, zarr_format=2)
            is_first_block = False
        else:
            block.to_zarr(output_path, mode="a", append_dim="time", consolidated=True, zarr_format=2)
        print(f"  {t1}/{time_size}", flush=True)

    print(f"[stack] done -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
