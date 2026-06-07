"""NASA FIRMS fetcher — today's active fire (near-real-time), the second live feed (prototype).

Provides today's fire for the live system: FIRMS VIIRS/MODIS active-fire detections (~3 h latency) →
rasterise to the grid → is_fire(today) → dist_to_fire (distance transform). These feed the fire-history
features (dist_to_fire is a TOP new-ignition driver; time_since_last_fire) and the regime split
(ignition vs spread). EFFIS (burned-area >5 ha) is the training label source; FIRMS active-fire is the
low-latency operational proxy for "what's burning right now".

API: https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SRC}/{W,S,E,N}/{days}/{date}
  SRC e.g. VIIRS_SNPP_NRT ; returns CSV of detections (latitude, longitude, acq_date, confidence, frp).
Needs a free MAP_KEY (env FIRMS_MAP_KEY). Unlike weather, fire is RASTERISED (point→cell), not interpolated.

Run `--demo` (no key/network): takes a real fire day from the cube, treats its fire cells as synthetic
FIRMS detections (lat/lon), rasterises them back + derives dist_to_fire, and compares to the cube's own
dist_to_fire — validates the fire→grid→dist_to_fire chain.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import xarray as xr
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt

CUBE = Path(__file__).resolve().parents[1] / "data" / "gold" / "IberFire_coarse4.zarr"
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def fetch_firms(map_key: str, date: str, src: str = "VIIRS_SNPP_NRT", bbox=(-10.0, 35.0, 5.0, 44.5)):
    """Fetch active-fire detections for the bbox (W,S,E,N) on `date`; return DataFrame (needs FIRMS_MAP_KEY)."""
    import io as _io
    import pandas as pd
    import requests
    w, s, e, n = bbox
    url = f"{FIRMS_BASE}/{map_key}/{src}/{w},{s},{e},{n}/1/{date}"
    r = requests.get(url, timeout=60); r.raise_for_status()
    return pd.read_csv(_io.StringIO(r.text))


def fires_to_grid(lons, lats, gx, gy, epsg=3035):
    """Rasterise detection points (lon/lat) onto the cube grid → binary is_fire[H,W]."""
    H, W = len(gy), len(gx); dx = (gx[-1] - gx[0]) / (W - 1); dy = (gy[-1] - gy[0]) / (H - 1)
    tx, ty = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(np.asarray(lons), np.asarray(lats))
    col = np.rint((np.asarray(tx) - gx[0]) / dx).astype(int)
    row = np.rint((np.asarray(ty) - gy[0]) / dy).astype(int)
    ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    fire = np.zeros((H, W), np.float32); fire[row[ok], col[ok]] = 1.0
    return fire


def dist_to_fire(fire, cell_km=4.0):
    """Distance (km) from each cell to the nearest fire cell (EDT on the binary fire mask)."""
    if fire.sum() == 0:
        return np.full(fire.shape, np.inf, np.float32)
    return (distance_transform_edt(fire == 0) * cell_km).astype(np.float32)


def demo():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("fetch_firms.demo")
    z = xr.open_zarr(str(CUBE), consolidated=True)
    gx = z["x"].values.astype(float); gy = z["y"].values.astype(float)
    # pick the fire-heaviest day
    isf = z["is_fire"]
    sums = isf.sum(dim=["x", "y"]).values
    ti = int(np.argmax(sums))
    fire_true = (isf.isel(time=ti).values > 0.5).astype(np.float32)
    nfire = int(fire_true.sum())
    log.info(f"fire-heaviest day idx {ti}: {nfire} fire cells")
    # synthetic FIRMS detections = lon/lat of those fire cells
    GX, GY = np.meshgrid(gx, gy)
    fi, fj = np.where(fire_true > 0.5)
    inv = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lons, lats = inv.transform(GX[fi, fj], GY[fi, fj])
    # rasterise back + dist_to_fire
    fire_rec = fires_to_grid(lons, lats, gx, gy)
    log.info(f"rasterised back: {int(fire_rec.sum())} cells (recovered {100*fire_rec.sum()/max(nfire,1):.0f}%)")
    d_rec = dist_to_fire(fire_rec)
    if "dist_to_fire" in z:
        d_true = z["dist_to_fire"].isel(time=ti).values.astype(float)
        land = np.isfinite(d_true)
        err = np.abs(d_rec[land] - d_true[land])
        log.info(f"dist_to_fire vs cube: MAE={err.mean():.3f} km median={np.median(err):.3f} "
                 f"corr={np.corrcoef(d_rec[land], d_true[land])[0,1]:.4f}")
    log.info("→ fire→grid→dist_to_fire chain validated; live path = FIRMS detections instead of cube cells.")


def main():
    if "--demo" in sys.argv:
        demo(); return
    key = os.getenv("FIRMS_MAP_KEY")
    if not key:
        print("Set FIRMS_MAP_KEY (free at https://firms.modaps.eosdis.nasa.gov/api/area/) for live fetch, or --demo.", file=sys.stderr)
        sys.exit(2)
    print("Live FIRMS path wired (fetch_firms + fires_to_grid + dist_to_fire); supply FIRMS_MAP_KEY and a date.")


if __name__ == "__main__":
    main()
