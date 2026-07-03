"""Unified external-data fetch primitives — the live feeds' HTTP + rasterise + regrid layer.

Consolidates the former `scripts/fetch_openmeteo.py` + `scripts/fetch_firms.py` (2026-07-03 refactor) into one
`src.data`-rooted module, since these are LIBRARIES imported across the ingest + serve paths (they were reaching
back into `scripts/` from `src/`, backwards). Two feeds:

  * **Open-Meteo** (weather) — keyless/free, ERA5-based so it matches the model's training distribution.
    Query a regular lat/lon grid → bilinear-regrid to the cube (EPSG:3035). Archive endpoint for backfill/serve
    (immutable past = disk-cached; recent ~5 d revise so are never cached), forecast endpoint for live "today".
    Commercial `OPENMETEO_API_KEY` (in .env) is FORECAST-scoped only (customer-archive → 403); archive stays free.
  * **FIRMS** (fire) — NASA VIIRS/MODIS active-fire CSV (~3 h latency). Point detections → rasterise to the grid
    (`fires_to_grid`) → is_fire. Needs a free `FIRMS_MAP_KEY` (env).

(The old single-date `fetch_grid`, `dist_to_fire`, and the `--demo` self-checks were dropped in the merge — the
demos opened the deleted v1 IberFire cube, and callers use only the range fetchers + regridder + rasteriser.
The EFFIS burned-area fetcher moved to `archive/scripts/fetch_effis.py` — v1-legacy, offline-eval only.)
"""
from __future__ import annotations
import os
import time
from pathlib import Path
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

import numpy as np
from pyproj import Transformer
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt

_ROOT = Path(__file__).resolve().parents[2]              # src/data/fetch.py → project root

# ── Open-Meteo (weather) ────────────────────────────────────────────────────────────────────────────────────
SPAIN_BBOX = (-9.5, 35.5, 4.5, 44.0)                     # W, S, E, N
# Commercial key is scoped to the FORECAST product only (customer-archive → 403), so only the forecast endpoint
# uses it; ARCHIVE stays on the FREE endpoint (immutable + cached + low-volume). Degrades to free when unset.
OPENMETEO_KEY = os.getenv("OPENMETEO_API_KEY") or None
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://customer-api.open-meteo.com/v1/forecast" if OPENMETEO_KEY else "https://api.open-meteo.com/v1/forecast"
CACHE_DIR = _ROOT / "data" / "cache" / "openmeteo"
CACHE_SAFE_DAYS = 7   # only cache archive ranges ending ≥ this old — the archive is seamless-to-today and its
                      # recent ~5 d are IFS/ERA5T-preliminary that REVISE daily (caching them would serve stale data)

# ── FIRMS (fire) ────────────────────────────────────────────────────────────────────────────────────────────
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


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


def fetch_grid_range(start: str, end: str, daily_vars, bbox=SPAIN_BBOX, step=0.5, source="archive",
                     batch=100, models=None):
    """Query Open-Meteo daily aggregates on a regular lat/lon grid for [start,end] — returns
    (lons, lats, {var: array[n_days, n_points]}, dates). The archive endpoint serves the whole window in ONE
    request per coord-batch (cheap). `models` (e.g. 'era5_land') selects the reanalysis model to match the cube."""
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
    Needed for variables Open-Meteo does NOT expose as daily aggregates (RH, surface_pressure, hourly wind
    speed/direction → daily mean/min/max/range). Smaller default batch since each point returns 24×n_days values."""
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
    In a backfill the source point set (fixed ERA5/Open-Meteo grid) and the target (fixed cube grid) are identical
    for every day × feature, so all that geometry is recomputed for nothing. Here we do it once: build the
    triangulation, locate every cube cell in its simplex, cache the barycentric weights + vertex indices (linear
    interp) and the nearest-neighbour index (hull-edge fill). Each subsequent regrid is a gather + weighted-sum —
    O(cells), not O(triangulate). Assumes the SAME finite source points every call (true for ERA5 over our box);
    a value vector with NaN transparently falls back to the scattered `regrid_to_cube`."""
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


def fetch_firms(map_key: str, date: str, src: str = "VIIRS_SNPP_NRT", bbox=(-10.0, 35.0, 5.0, 44.5),
                days: int = 1, tries: int = 5):
    """Fetch active-fire detections for the bbox (W,S,E,N) starting at `date` for `days` days (FIRMS area API
    caps day_range at 5); return DataFrame. `src` selects sensor+latency, e.g. VIIRS_SNPP_NRT (recent) or
    VIIRS_SNPP_SP (standard-processing archive, 2012-01-20→). Needs FIRMS_MAP_KEY. Retries on transient errors."""
    import io as _io
    import time as _t
    import pandas as pd
    import requests
    w, s, e, n = bbox
    url = f"{FIRMS_BASE}/{map_key}/{src}/{w},{s},{e},{n}/{int(days)}/{date}"
    for i in range(tries):
        try:
            r = requests.get(url, timeout=90)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            _t.sleep(min(2 ** i * 4, 45)); continue
        if r.status_code == 429 or r.status_code >= 500:
            _t.sleep(min(2 ** i * 4, 45)); continue
        r.raise_for_status()
        txt = r.text
        if txt.lstrip().lower().startswith(("invalid", "error")) or "Invalid MAP_KEY" in txt[:200]:
            raise RuntimeError(f"FIRMS error: {txt[:120]}")
        return pd.read_csv(_io.StringIO(txt))
    raise RuntimeError("FIRMS unavailable after retries")


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
    """Distance (km) from each cell to the nearest fire cell (EDT on the binary fire mask). (Kept as a small
    reusable primitive; the cube's dist_to_fire FEATURE is built via feature_engineering.fire_distance_and_exposure.)"""
    if fire.sum() == 0:
        return np.full(fire.shape, np.inf, np.float32)
    return (distance_transform_edt(fire == 0) * cell_km).astype(np.float32)
