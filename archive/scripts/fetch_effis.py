"""EFFIS burned-area fetcher — the fire source that MATCHES the model's training definition.

The IberFire `is_fire`/`dist_to_fire` features were built from EFFIS burned-area polygons (>5 ha), so for
live serving the model's fire features must come from EFFIS too (NOT FIRMS active-fire — different quantity,
caused a prediction-corr drop to 0.10). FIRMS stays for the low-latency "burning now" DISPLAY only.

Source: EFFIS WFS (open) at ies-ows.jrc.ec.europa.eu/effis, layer `ercc.ba` (burned-area perimeters,
daily NRT). Fetch GeoJSON polygons (EPSG:4326) → transform to the cube CRS (EPSG:3035, via pyproj) →
rasterize onto the 4 km grid (rasterio.features, no PROJ db needed) → is_fire → dist_to_fire (EDT).

NOTE (2026-06-07): the open WFS backend was returning an OracleSpatial connection error (server-side,
likely transient). `fetch_effis_polys` retries with backoff and raises if the endpoint stays down — callers
(live_slice) then fall back to the cube's EFFIS-consistent warm-start (matched definition, just not today's).

Run `--demo` to test the rasterize→dist_to_fire chain offline (synthetic polygon) + probe the live endpoint.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import xarray as xr
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt

CUBE = Path(__file__).resolve().parents[1] / "data" / "gold" / "IberFire_coarse4.zarr"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "serving_store" / "effis_cache"
WFS = "https://ies-ows.jrc.ec.europa.eu/effis"
LAYER = "ercc.ba"
SPAIN_BBOX_LL = (35.5, -9.5, 44.0, 4.5)  # minlat, minlon, maxlat, maxlon (WFS 1.1.0 EPSG:4326 axis order)


def cache_effis(date, fire_grid):
    """Persist a successfully-fetched EFFIS fire grid so a later run can fall back to the freshest KNOWN
    fire state (weather-style persistence) instead of a year-old seasonal cube slice."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_DIR / f"{date}.npz", fire=fire_grid.astype(np.float32), date=str(date))


def latest_cached_effis():
    """(date, fire_grid) of the most recent cached EFFIS fetch, or None if the cache is empty."""
    if not CACHE_DIR.exists():
        return None
    files = sorted(CACHE_DIR.glob("*.npz"))
    if not files:
        return None
    d = np.load(files[-1], allow_pickle=True)
    return str(d["date"]), d["fire"].astype(np.float32)


def fetch_effis_polys(bbox_ll=SPAIN_BBOX_LL, tries=4):
    """GeoJSON burned-area features (EPSG:4326) for the bbox; retries the transient backend error."""
    import requests
    params = {"service": "WFS", "version": "1.1.0", "request": "GetFeature", "typename": LAYER,
              "outputFormat": "geojson", "srsName": "EPSG:4326",
              "bbox": ",".join(map(str, bbox_ll)) + ",EPSG:4326"}
    for i in range(tries):
        r = requests.get(WFS, params=params, timeout=60)
        if r.status_code == 200 and "ServiceException" not in r.text[:400] and r.text.lstrip().startswith("{"):
            return r.json().get("features", [])
        time.sleep(2 ** i * 4)  # backoff (server-side Oracle error is often transient)
    raise RuntimeError("EFFIS WFS unavailable (endpoint/backend error after retries)")


def _grid_transform(gx, gy):
    dx = (gx[-1] - gx[0]) / (len(gx) - 1); dy = (gy[-1] - gy[0]) / (len(gy) - 1)
    return Affine(dx, 0, gx[0] - dx / 2, 0, dy, gy[0] - dy / 2)  # matches the cube array orientation


def _to_3035(geom, tr):
    def conv(ring):
        xs, ys = tr.transform([p[0] for p in ring], [p[1] for p in ring])
        return [[x, y] for x, y in zip(xs, ys)]
    t = geom["type"]; c = geom["coordinates"]
    if t == "Polygon":
        return {"type": "Polygon", "coordinates": [conv(r) for r in c]}
    if t == "MultiPolygon":
        return {"type": "MultiPolygon", "coordinates": [[conv(r) for r in poly] for poly in c]}
    return None


def polys_to_grid(features, gx, gy, epsg=3035):
    """Rasterise EFFIS burned-area polygons (lon/lat) onto the cube grid → binary is_fire[H,W]."""
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    shapes = []
    for f in features:
        g = _to_3035(f.get("geometry") or {}, tr)
        if g:
            shapes.append((g, 1))
    H, W = len(gy), len(gx)
    if not shapes:
        return np.zeros((H, W), np.float32)
    return rasterize(shapes, out_shape=(H, W), transform=_grid_transform(gx, gy), fill=0).astype(np.float32)


def dist_to_fire(fire, cell_km=4.0):
    if fire.sum() == 0:
        return np.full(fire.shape, np.inf, np.float32)
    return (distance_transform_edt(fire == 0) * cell_km).astype(np.float32)


def demo():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("fetch_effis.demo")
    z = xr.open_zarr(str(CUBE), consolidated=True)
    gx = z["x"].values.astype(float); gy = z["y"].values.astype(float)
    # offline rasterize test: a synthetic burned polygon near Madrid (lon/lat) → grid → dist
    poly = {"type": "Polygon", "coordinates": [[[-3.8, 40.3], [-3.8, 40.6], [-3.4, 40.6], [-3.4, 40.3], [-3.8, 40.3]]]}
    fire = polys_to_grid([{"geometry": poly}], gx, gy)
    d = dist_to_fire(fire)
    log.info(f"offline rasterize: synthetic polygon → {int(fire.sum())} burned cells; "
             f"dist_to_fire range [{np.nanmin(d):.0f},{np.nanmax(d[np.isfinite(d)]):.0f}] km — chain OK")
    # probe live endpoint
    try:
        feats = fetch_effis_polys()
        fire_live = polys_to_grid(feats, gx, gy)
        log.info(f"LIVE EFFIS: {len(feats)} burned-area features → {int(fire_live.sum())} cells")
    except Exception as e:
        log.warning(f"LIVE EFFIS unavailable: {e} (callers fall back to cube EFFIS warm-start)")


def main():
    if "--demo" in sys.argv:
        demo(); return
    print("Module: fetch_effis_polys() + polys_to_grid() + dist_to_fire(). Run --demo to test.", file=sys.stderr)


if __name__ == "__main__":
    main()
