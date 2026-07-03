"""Open-Meteo weather feed — keyless, gridded, live forecast + ERA5 archive. The chosen weather source.

Replaces the AEMET-station path (flaky key, station→grid interpolation, distribution shift). Open-Meteo is:
  * keyless + free (non-commercial), no signup;
  * already GRIDDED (we query a regular lat/lon grid and bilinear-regrid to the cube — no scattered-station
    interpolation);
  * ERA5-based, so it MATCHES our model's training distribution (minimal train/serve shift);
  * forecast API (live "today") + archive API (ERA5, for backfill / IberFire-v2).

Open-Meteo daily → our feature mapping (units already match the cube: °C, mm, hPa):
  temperature_2m_mean/min/max -> t2m_mean/min/max ; precipitation_sum -> tp ;
  wind_speed_10m_max -> wind_speed ; wind_direction_10m_dominant -> wind_direction ;
  (pressure / RH need hourly→daily aggregation — add when wiring the full slice).

archive:  https://archive-api.open-meteo.com/v1/archive   (start_date/end_date; ERA5)
forecast: https://api.open-meteo.com/v1/forecast           (past_days/forecast_days)

`--demo`: fetch a cube-range date from the archive, regrid t2m_mean to the cube grid, and compare to the
cube's own t2m_mean for that date — validates the fetch→regrid produces cube-compatible features.
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import griddata

CUBE = Path(__file__).resolve().parents[1] / "data" / "gold" / "IberFire_coarse4.zarr"
SPAIN_BBOX = (-9.5, 35.5, 4.5, 44.0)  # W, S, E, N
# Commercial key (OPENMETEO_API_KEY in .env) is scoped to the FORECAST product only (customer-archive → 403), so
# only the forecast endpoint uses it (higher rate limits — the free tier 429s on large hourly forecast requests).
# ARCHIVE stays on the FREE endpoint: the key doesn't cover it, and archive responses are immutable+cached+low-
# volume (weekly batch) so the free tier is fine. Everything degrades to free endpoints when no key is set.
OPENMETEO_KEY = os.getenv("OPENMETEO_API_KEY") or None
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://customer-api.open-meteo.com/v1/forecast" if OPENMETEO_KEY else "https://api.open-meteo.com/v1/forecast"
DAILY_MAP = {"temperature_2m_mean": "t2m_mean", "temperature_2m_max": "t2m_max",
             "temperature_2m_min": "t2m_min", "precipitation_sum": "tp",
             "wind_speed_10m_max": "wind_speed", "wind_direction_10m_dominant": "wind_direction"}


CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "openmeteo"
CACHE_SAFE_DAYS = 7   # only cache archive ranges ending ≥ this old — the archive is seamless-to-today and its
                      # recent ~5 d are IFS/ERA5T-preliminary that REVISE daily (caching them would serve stale data)


def _cache_path(url, params):
    """Disk-cache key for a request. OLD archive days are IMMUTABLE ERA5 (a past date's reanalysis is stable), so
    caching them is correctness-safe AND dodges the free-tier 429 wall on repeated backfills/backtests. But the
    archive is seamless-to-today and its RECENT ~5 d are IFS/ERA5T-preliminary that revise daily → never cache a
    range reaching within CACHE_SAFE_DAYS of today. Forecast responses also never cached (returns None)."""
    if not url.startswith(ARCHIVE):
        return None
    end = params.get("end_date")
    if end:
        import datetime as _d
        if (_d.date.today() - _d.date.fromisoformat(str(end))).days < CACHE_SAFE_DAYS:
            return None
    import hashlib
    import json
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True, default=str)).encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _get(url, params, tries=6, timeout=120, use_cache=True):
    """GET with backoff on 429/5xx AND on network timeouts/connection errors (hourly multi-var requests are
    large and the free tier is flaky), honoring Retry-After. Archive responses are disk-cached (immutable →
    repeated backtests/backfills cost no quota)."""
    import json
    import requests
    cp = _cache_path(url, params) if use_cache else None
    if cp is not None and cp.exists():
        return json.loads(cp.read_text())
    # apikey only on the (commercial) customer forecast endpoint — archive is free + would 403 with the key
    req_params = {**params, "apikey": OPENMETEO_KEY} if (OPENMETEO_KEY and "customer-" in url) else params
    last_exc = None
    for i in range(tries):
        try:
            r = requests.get(url, params=req_params, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e; time.sleep(min(2 ** i * 5, 60)); continue
        if r.status_code == 429 or r.status_code >= 500:
            wait = int(r.headers.get("Retry-After", 0)) or min(2 ** i * 5, 60)  # 5,10,20,40,60,60s
            time.sleep(wait)
            continue
        r.raise_for_status()
        js = r.json()
        if cp is not None:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(js))
        return js
    if last_exc is not None:
        raise last_exc
    r.raise_for_status()


def fetch_grid(date: str, daily_vars, bbox=SPAIN_BBOX, step=0.25, source="archive", batch=100):
    """Query Open-Meteo on a regular lat/lon grid for `date`; return (lons, lats, {var: values}).

    Large batch = few requests (Open-Meteo accepts many coords/call) — the key to not getting rate-limited."""
    w, s, e, n = bbox
    lons = np.arange(w, e + 1e-9, step); lats = np.arange(s, n + 1e-9, step)
    LON, LAT = np.meshgrid(lons, lats)
    plon, plat = LON.ravel(), LAT.ravel()
    url = ARCHIVE if source == "archive" else FORECAST
    out = {v: np.full(plon.size, np.nan) for v in daily_vars}
    for b0 in range(0, plon.size, batch):
        sl = slice(b0, b0 + batch)
        params = {"latitude": ",".join(f"{x:.4f}" for x in plat[sl]),
                  "longitude": ",".join(f"{x:.4f}" for x in plon[sl]),
                  "daily": ",".join(daily_vars), "timezone": "UTC"}
        params["start_date"] = params["end_date"] = date  # forecast endpoint also serves recent past + forecast window
        js = _get(url, params)
        js = js if isinstance(js, list) else [js]  # multi-coord returns a list
        for k, item in enumerate(js):
            d = item.get("daily", {})
            for v in daily_vars:
                val = d.get(v, [None])
                out[v][b0 + k] = (val[0] if val and val[0] is not None else np.nan)
        if b0 + batch < plon.size:
            time.sleep(1.0)  # be polite between batches
    return plon, plat, out


def fetch_grid_range(start: str, end: str, daily_vars, bbox=SPAIN_BBOX, step=0.5, source="archive",
                     batch=100, models=None):
    """Like fetch_grid but for a DATE RANGE — returns (lons, lats, {var: array[n_days, n_points]}, dates).
    The archive endpoint serves the whole [start,end] window in ONE request per coord-batch (cheap).
    `models` (e.g. 'era5_land') selects the reanalysis model — use era5_land to match the IberFire cube."""
    w, s, e, n = bbox
    lons = np.arange(w, e + 1e-9, step); lats = np.arange(s, n + 1e-9, step)
    LON, LAT = np.meshgrid(lons, lats); plon, plat = LON.ravel(), LAT.ravel()
    url = ARCHIVE if source == "archive" else FORECAST
    out, dates = None, None
    for b0 in range(0, plon.size, batch):
        sl = slice(b0, b0 + batch)
        params = {"latitude": ",".join(f"{x:.4f}" for x in plat[sl]),
                  "longitude": ",".join(f"{x:.4f}" for x in plon[sl]),
                  "daily": ",".join(daily_vars), "timezone": "UTC", "start_date": start, "end_date": end}
        if models:
            params["models"] = models
        js = _get(url, params); js = js if isinstance(js, list) else [js]
        if out is None:
            dates = js[0]["daily"]["time"]
            out = {v: np.full((len(dates), plon.size), np.nan) for v in daily_vars}
        for k, item in enumerate(js):
            d = item.get("daily", {})
            for v in daily_vars:
                a = d.get(v)
                if a:
                    out[v][:, b0 + k] = [(x if x is not None else np.nan) for x in a]
        if b0 + batch < plon.size:
            time.sleep(1.0)
    return plon, plat, out, dates


def fetch_grid_hourly_range(start: str, end: str, hourly_vars, bbox=SPAIN_BBOX, step=0.5, source="archive",
                            batch=60, models=None):
    """Hourly counterpart of fetch_grid_range — returns (lons, lats, {var: array[n_hours, n_points]}, times).
    Needed for the variables Open-Meteo does NOT expose as daily aggregates (RH, surface_pressure, hourly
    wind speed/direction → daily mean/min/max/range and at-max-speed). Smaller default batch since each
    point returns 24×n_days values."""
    w, s, e, n = bbox
    lons = np.arange(w, e + 1e-9, step); lats = np.arange(s, n + 1e-9, step)
    LON, LAT = np.meshgrid(lons, lats); plon, plat = LON.ravel(), LAT.ravel()
    url = ARCHIVE if source == "archive" else FORECAST
    out, times = None, None
    for b0 in range(0, plon.size, batch):
        sl = slice(b0, b0 + batch)
        params = {"latitude": ",".join(f"{x:.4f}" for x in plat[sl]),
                  "longitude": ",".join(f"{x:.4f}" for x in plon[sl]),
                  "hourly": ",".join(hourly_vars), "timezone": "UTC", "start_date": start, "end_date": end}
        if models:
            params["models"] = models
        js = _get(url, params); js = js if isinstance(js, list) else [js]
        if out is None:
            times = js[0]["hourly"]["time"]
            out = {v: np.full((len(times), plon.size), np.nan) for v in hourly_vars}
        for k, item in enumerate(js):
            h = item.get("hourly", {})
            for v in hourly_vars:
                a = h.get(v)
                if a:
                    out[v][:, b0 + k] = [(x if x is not None else np.nan) for x in a]
        if b0 + batch < plon.size:
            time.sleep(1.0)
    return plon, plat, out, times


def regrid_to_cube(plon, plat, vals, gx, gy, epsg=3035):
    """Bilinear-regrid scattered lat/lon point values onto the cube grid (EPSG:3035)."""
    tx, ty = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(plon, plat)
    ok = np.isfinite(vals)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([tx[ok], ty[ok]])
    lin = griddata(pts, vals[ok], (GX, GY), method="linear")
    nn = griddata(pts, vals[ok], (GX, GY), method="nearest")  # fill edges/holes
    return np.where(np.isfinite(lin), lin, nn).astype(np.float32)


def make_regridder(plon, plat, gx, gy, epsg=3035):
    """Precompute the source→cube interpolation ONCE and return a fast callable `f(vals) -> cube_grid`.

    `regrid_to_cube` rebuilds the Delaunay triangulation AND a KD-tree on every call (twice — linear+nearest).
    In a backfill the source point set (the fixed ERA5/Open-Meteo grid) and the target (the fixed 1 km cube
    grid) are identical for every day × feature, so all that geometry is recomputed thousands of times for
    nothing. Here we do it once: build the triangulation, locate every cube cell in its simplex, cache the
    barycentric weights + vertex indices (linear interp) and the nearest-neighbour index (hull-edge fill).
    Each subsequent regrid is then a gather + weighted-sum — O(cells), not O(triangulate). Turns the
    per-feature cost from ~hundreds of ms into ~tens of ms.

    Assumes the SAME finite source points for every call (true for ERA5 reanalysis over our box — no NaNs).
    If a value vector contains NaN it transparently falls back to the scattered `regrid_to_cube`."""
    from scipy.spatial import Delaunay, cKDTree
    tx, ty = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(plon, plat)
    src = np.column_stack([np.asarray(tx, float), np.asarray(ty, float)])
    GX, GY = np.meshgrid(gx, gy)
    shape = GX.shape
    tgt = np.column_stack([GX.ravel(), GY.ravel()])
    tri = Delaunay(src)
    simplex = tri.find_simplex(tgt)
    inside = simplex >= 0
    T = tri.transform[simplex[inside]]                 # (n_in, 3, 2): [:, :2]=affine inv, [:, 2]=offset
    d = tgt[inside] - T[:, 2]
    bary = np.einsum("ijk,ik->ij", T[:, :2], d)        # first 2 barycentric coords
    weights = np.column_stack([bary, 1.0 - bary.sum(axis=1)]).astype(np.float64)  # (n_in, 3)
    verts = tri.simplices[simplex[inside]]             # (n_in, 3) → indices into src
    nn_idx = cKDTree(src).query(tgt)[1]                # nearest source point per cube cell

    def regrid(vals):
        vals = np.asarray(vals, float)
        if not np.isfinite(vals).all():
            return regrid_to_cube(plon, plat, vals, gx, gy, epsg)   # robust fallback (rare for ERA5)
        out = vals[nn_idx]                                          # nearest everywhere → hull-edge fill
        out[inside] = np.einsum("ij,ij->i", vals[verts], weights)   # barycentric-linear inside the hull
        return out.reshape(shape).astype(np.float32)

    regrid.n_src = src.shape[0]
    regrid.n_cells = tgt.shape[0]
    return regrid


def demo():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("fetch_openmeteo.demo")
    z = xr.open_zarr(str(CUBE), consolidated=True)
    cube_var = next((v for v in ["t2m_mean"] if v in z), None)
    date = "2024-07-15"  # cube-range, in archive
    gx = z["x"].values.astype(float); gy = z["y"].values.astype(float)
    t0 = time.time()
    plon, plat, vals = fetch_grid(date, ["temperature_2m_mean"], step=0.5, source="archive")
    log.info(f"fetched {plon.size} Open-Meteo grid points for {date} in {time.time()-t0:.1f}s "
             f"({np.isfinite(vals['temperature_2m_mean']).sum()} valid)")
    grid = regrid_to_cube(plon, plat, vals["temperature_2m_mean"], gx, gy)
    if cube_var:
        truth = z[cube_var].sel(time=date).values.astype(float)
        land = np.isfinite(truth)
        err = np.abs(grid[land] - truth[land])
        log.info(f"Open-Meteo regridded t2m_mean vs cube '{cube_var}' ({date}): "
                 f"MAE={err.mean():.3f}°C median={np.median(err):.3f} corr={np.corrcoef(grid[land], truth[land])[0,1]:.4f}")
        log.info("  → fetch+regrid produces cube-compatible features (both ERA5-based); this IS the live weather slice.")
    else:
        log.info(f"grid built {grid.shape}; cube t2m var not found for direct comparison.")


def main():
    if "--demo" in sys.argv:
        demo(); return
    print("Use as a module: fetch_grid(date, vars, source='forecast'|'archive') + regrid_to_cube(...).", file=sys.stderr)
    print("Run --demo to validate the fetch+regrid chain against the cube.", file=sys.stderr)


if __name__ == "__main__":
    main()
