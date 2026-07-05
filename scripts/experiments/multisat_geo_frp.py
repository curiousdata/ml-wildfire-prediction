"""Geostationary fire (MTG-FCI FDeM, "Active Fire Monitoring") fetch + grid — for the VIIRS-vs-geostationary
complementarity study (does a continuous-cadence geo sensor add signal the 3 polar VIIRS birds miss?).

Access: EUMETSAT Data Store via `eumdac` (EUMETSAT_KEY / EUMETSAT_SECRET in .env).
Collection: EO:EUM:DAT:0682 (Active Fire Monitoring, netCDF, MTG 0°). Each granule is a full-disk L2 grid
(`fire_result[5568,5568]` on the geostationary projection): values {1,2,3}=fire, 0=no-fire, 4=space. Native
cadence ~10 min; we SAMPLE (default hourly) to keep the ~144-granule/day firehose tractable — fine for the
correlation study since VIIRS gaps are multi-hour windows.

  --probe DATE            pull one granule near DATE 12:00 UTC, report disk & Spain-bbox fire counts (sanity).
  --fetch START END       grid fire pixels to the cube's 4 km grid per cube-day, accumulate:
                            geo_any — cell lit by MTG at any sampled slot that day
                            geo_gap — cell lit ONLY in a between-VIIRS-pass slot (the geostationary payoff)
                          → data/cache/multisat/geo_{any,gap}.npy (aligned to cube time; git-ignored).
  --step-min N            sampling stride in minutes (default 60).

NB coarse geo (~2 km) localisation is harmless at our 4 km grid; its value is TEMPORAL (fires that ignite+die
between the ~01:30/13:30 VIIRS overpasses), traded against lower small-fire sensitivity.
"""
from __future__ import annotations
import argparse
import datetime as dt
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from dotenv import load_dotenv; load_dotenv(dotenv_path=str(Path(__file__).resolve().parents[2] / ".env"))
except Exception:
    pass
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

from src.data import fetch as FB
from src.data import metrics as T
from src.data.ingest import grid

COLLECTION = "EO:EUM:DAT:0682"
CACHE = T.project_root / "data" / "cache" / "multisat"
BBOX = (-10.0, 35.0, 5.0, 44.5)                                  # W,S,E,N — Spain + margin
FIRE_CLASSES = (1, 2, 3)                                          # fire_result values that mean "fire"
# VIIRS overpass UTC windows over Spain (night + afternoon clusters, ±~1 h). An MTG slot OUTSIDE these is a
# "gap" slot — fire the polar birds structurally cannot see that day.
VIIRS_WINDOWS = [(0.0, 2.5), (11.7, 14.5), (23.7, 24.0)]         # hours UTC


def _collection():
    import eumdac
    key, secret = os.getenv("EUMETSAT_KEY"), os.getenv("EUMETSAT_SECRET")
    if not (key and secret):
        raise SystemExit("set EUMETSAT_KEY and EUMETSAT_SECRET in .env")
    return eumdac.DataStore(eumdac.AccessToken((key, secret))).get_collection(COLLECTION)


def _fire_lonlat(product, td: Path, tr: list):
    """Fire-pixel (lon, lat) from one MTG FDeM granule: fire_result∈{1,2,3} → geostationary x/y (rad) → lon/lat."""
    ent = next((e for e in product.entries if e.lower().endswith((".nc", ".nc4"))), None)
    if ent is None:
        return np.array([]), np.array([])
    dst = td / Path(ent).name
    with product.open(entry=ent) as fs, open(dst, "wb") as fo:
        fo.write(fs.read())
    x = xr.open_dataset(dst)
    fr = x["fire_result"].values
    rows, cols = np.where(np.isin(fr, FIRE_CLASSES))
    if rows.size == 0:
        return np.array([]), np.array([])
    h = float(x["mtg_geos_projection"].attrs["perspective_point_height"])
    if tr[0] is None:
        tr[0] = Transformer.from_crs(f"+proj=geos +h={h} +lon_0=0 +a=6378137 +b=6356752.3 +sweep=y",
                                     "EPSG:4326", always_xy=True)
    # MTG stores the x scan-angle with the OPPOSITE sign to the PROJ geos convention → negate x (verified against
    # VIIRS truth: with the negation, geo fires land a median 4 km from VIIRS fires; without, ~300 km off onto sea).
    lon, lat = tr[0].transform(-x["x"].values[cols] * h, x["y"].values[rows] * h)
    return np.asarray(lon, float), np.asarray(lat, float)


def probe(date: str):
    d = dt.datetime.fromisoformat(date + "T12:00:00")
    coll = _collection()
    prods = list(coll.search(dtstart=d, dtend=d + dt.timedelta(minutes=10)))
    print(f"{COLLECTION}: {len(prods)} product(s) near {d}")
    if not prods:
        return
    with tempfile.TemporaryDirectory() as td:
        lon, lat = _fire_lonlat(prods[0], Path(td), [None])
    m = ((lon >= BBOX[0]) & (lon <= BBOX[2]) & (lat >= BBOX[1]) & (lat <= BBOX[3])) if lon.size else np.array([])
    print(f"  {prods[0]}\n  {lon.size} disk fires, {int(m.sum()) if lon.size else 0} in Spain bbox")


def _is_gap(hour: float) -> bool:
    return not any(a <= hour < b for a, b in VIIRS_WINDOWS)


def fetch(start: str, end: str, step_min: int):
    CACHE.mkdir(parents=True, exist_ok=True)
    z = xr.open_zarr(str(grid.ROOT / "data" / "gold" / "FireGuard_coarse4.zarr"), consolidated=True)
    gx, gy = z["x"].values.astype(float), z["y"].values.astype(float)
    tindex = {d.date().isoformat(): i for i, d in enumerate(pd.DatetimeIndex(z["time"].values))}
    T_, Y, X = z.sizes["time"], z.sizes["y"], z.sizes["x"]
    any_ = np.zeros((T_, Y, X), bool); gap = np.zeros((T_, Y, X), bool)
    coll = _collection(); tr = [None]
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    t0 = time.time(); nslot = ndl = 0
    day = d0
    while day <= d1:
        i = tindex.get(day.isoformat())
        if i is None:
            day += dt.timedelta(days=1); continue
        with tempfile.TemporaryDirectory() as td:
            for minute in range(0, 24 * 60, step_min):
                h0 = dt.datetime.combine(day, dt.time()) + dt.timedelta(minutes=minute)
                prods = list(coll.search(dtstart=h0, dtend=h0 + dt.timedelta(minutes=10)))
                if not prods:
                    continue
                try:
                    lon, lat = _fire_lonlat(prods[0], Path(td), tr); ndl += 1
                except Exception as e:
                    print(f"    {h0:%Y-%m-%d %H:%M} parse fail {type(e).__name__}", flush=True); continue
                if lon.size == 0:
                    continue
                m = (lon >= BBOX[0]) & (lon <= BBOX[2]) & (lat >= BBOX[1]) & (lat <= BBOX[3])
                if not m.any():
                    continue
                g = FB.fires_to_grid(lon[m], lat[m], gx, gy) > 0.5
                any_[i] |= g
                if _is_gap(h0.hour + h0.minute / 60.0):
                    gap[i] |= g
                nslot += 1
        if (day - d0).days % 5 == 0:
            print(f"  {day} ({ndl} granules, {nslot} with Spain fire, any={int(any_.sum())} cell-days, "
                  f"{time.time()-t0:.0f}s)", flush=True)
        day += dt.timedelta(days=1)
    np.save(CACHE / "geo_any.npy", any_); np.save(CACHE / "geo_gap.npy", gap)
    print(f"DONE: {ndl} granules, any={int(any_.sum())} gap={int(gap.sum())} cell-days → {CACHE}/geo_*.npy "
          f"({time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description="MTG-FCI FDeM fetch/grid for the geo-vs-VIIRS study")
    ap.add_argument("--probe", metavar="DATE")
    ap.add_argument("--fetch", nargs=2, metavar=("START", "END"))
    ap.add_argument("--step-min", type=int, default=60)
    a = ap.parse_args()
    if a.probe:
        probe(a.probe)
    elif a.fetch:
        fetch(a.fetch[0], a.fetch[1], a.step_min)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
