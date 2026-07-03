import xarray as xr
import numpy as np
from pathlib import Path
import json

project_root = Path(__file__).resolve().parents[1]
ZARR_PATH = project_root / "data" / "silver" / "IberFire.zarr"
OUT_PATH = project_root / "stats" / "fire_day_indices.json"

LABEL_NAME = "is_fire"

def main():
    print(f"Opening Zarr: {ZARR_PATH}")
    ds = xr.open_zarr(ZARR_PATH, consolidated=True)

    label_da = ds[LABEL_NAME]  # dims: (time, y, x)
    print("Dataset dims:", dict(ds.dims))

    # Count positive pixels per day
    # (if label is bool or 0/1, this is number of fire pixels)
    fire_pixels_per_day = label_da.sum(dim=("y", "x")).values  # shape: (time,)

    # Define fire-day vs no-fire-day
    fire_days = np.where(fire_pixels_per_day > 0)[0].tolist()
    no_fire_days = np.where(fire_pixels_per_day == 0)[0].tolist()

    print(f"Total time steps: {len(fire_pixels_per_day)}")
    print(f"Fire days: {len(fire_days)}")
    print(f"No-fire days: {len(no_fire_days)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fire_days": fire_days,
        "no_fire_days": no_fire_days,
    }
    with OUT_PATH.open("w") as f:
        json.dump(payload, f)

    print(f"Saved fire/no-fire day indices to: {OUT_PATH}")

if __name__ == "__main__":
    main()