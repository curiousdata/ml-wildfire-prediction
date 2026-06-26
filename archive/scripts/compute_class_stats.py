#!/usr/bin/env python
import json
from pathlib import Path

import numpy as np
import xarray as xr

# Project root is parent of /scripts
project_root = Path(__file__).resolve().parents[1]

# Same Zarr + period as in train.py
ZARR_PATH = project_root / "data" / "gold" / "IberFire_coarse32.zarr"
LABEL_VAR = "is_fire"
TRAIN_TIME_START = "2008-01-01"
TRAIN_TIME_END = "2022-12-31"
COARSEN_FACTOR = 32  # for logging only

OUT_PATH = project_root / "stats" / "train_class_stats_is_fire.json"


def main():
    print(f"Opening Zarr: {ZARR_PATH}")
    ds = xr.open_zarr(ZARR_PATH, consolidated=True)

    if LABEL_VAR not in ds.data_vars:
        raise KeyError(f"Label variable '{LABEL_VAR}' not found in dataset.")

    da = ds[LABEL_VAR].sel(time=slice(TRAIN_TIME_START, TRAIN_TIME_END))

    # Binary mask 0/1 → positives = sum, total = number of finite pixels
    arr = da.values
    # If NaNs exist, ignore them in counts
    valid_mask = np.isfinite(arr)
    total_pixels = int(valid_mask.sum())
    positives = int(arr[valid_mask].sum())
    negatives = int(total_pixels - positives)

    pos_ratio = float(positives / total_pixels) if total_pixels > 0 else 0.0
    neg_ratio = float(negatives / total_pixels) if total_pixels > 0 else 0.0

    stats = {
        "zarr_path": str(ZARR_PATH),
        "label_var": LABEL_VAR,
        "time_start": TRAIN_TIME_START,
        "time_end": TRAIN_TIME_END,
        "coarsen_factor": COARSEN_FACTOR,
        "total_pixels": total_pixels,
        "positives": positives,
        "negatives": negatives,
        "pos_ratio": pos_ratio,
        "neg_ratio": neg_ratio,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved train class stats to {OUT_PATH}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()