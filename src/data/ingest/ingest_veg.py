"""FGDC vegetation ingester — MODIS NDVI/EVI (+ LAI/FAPAR) → 1 km daily, via Microsoft Planetary Computer.

Source decision (refines the plan): use **Microsoft Planetary Computer** STAC (keyless, anonymous-read +
SAS-signed COGs) for the MODIS spine instead of NASA Earthdata — no login, fits the FGDC keyless principle.
Collections: modis-13A1-061 (NDVI/EVI, 500 m, 16-day) [+ modis-15A2H-061 LAI/FAPAR, modis-09A1 reflectance
for NDWI/NDMI — same pattern, added next]. MODIS covers the whole 2012→~2026 span; the VIIRS VNP13/15
forward-bridge (post-MODIS-deorbit) is a later task and would harmonize on the 2012–2026 overlap.

Pipeline per variable: search Spain tiles (sinusoidal h17/h18 × v04/v05) for composites bracketing the
window → mosaic → reproject onto the canonical 1 km EPSG:3035 grid (reproject_match) → linearly interpolate
the 16-day composites to daily. NDVI/EVI scale ×0.0001.

PROJ note: the shell leaks a broken conda PROJ_DATA; we pin it to rasterio's bundled proj_data so GDAL warp
works in this .venv (pyproj's own proj.db is version-incompatible with rasterio's GDAL).

CLI: --validate DATE | --backfill START END
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# --- pin PROJ to rasterio's bundled db BEFORE importing rasterio/rioxarray (conda leak fix) ---
import rasterio as _rio
_RIO_PROJ = os.path.join(os.path.dirname(_rio.__file__), "proj_data")
if os.path.isdir(_RIO_PROJ):
    os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = _RIO_PROJ
# --- harden GDAL streaming of MPC COGs (transient partial-read / TIFFReadEncodedTile failures) ---
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "6")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import datetime as _dt
import numpy as np
import rioxarray  # noqa: F401  (registers .rio)
import xarray as xr

from src.data.ingest import grid

BRONZE = grid.ROOT / "data" / "bronze" / "fireguard" / "veg"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BBOX_LL = (-9.6, 35.4, 4.6, 44.1)
# MPC collection → {feature: (asset, scale, valid_min_DN, valid_max_DN)}. Each composite is mosaicked over
# Spain tiles, reprojected to 1 km, and linearly interpolated composite→daily. Reaches v1 veg parity:
# NDVI/EVI, LAI/FAPAR, LST. Valid DN ranges mask MODIS QA fill values (e.g. 15A2H 249–255 = water/cloud/
# fill) BEFORE scaling — otherwise fills survive ×scale (FAPAR>1, LAI~10). (SWI proxy = ERA5 soil moisture
# from ingest_weather; NDWI/NDMI from 09A1 are a later ablation.)
PRODUCTS = {
    "modis-13A1-061": {"NDVI": ("500m_16_days_NDVI", 1e-4, -2000, 10000),
                       "EVI": ("500m_16_days_EVI", 1e-4, -2000, 10000)},
    "modis-15A2H-061": {"LAI": ("Lai_500m", 0.1, 0, 100), "FAPAR": ("Fpar_500m", 0.01, 0, 100)},
    "modis-11A2-061": {"LST": ("LST_Day_1km", 0.02, 7500, 65535)},   # Kelvin, like v1
}
_TARGET = None


def _target():
    """An empty DataArray on the canonical 1 km EPSG:3035 grid for reproject_match."""
    global _TARGET
    if _TARGET is None:
        da = xr.DataArray(np.zeros(grid.shape(), np.float32),
                          coords={"y": grid.y_coords(), "x": grid.x_coords()}, dims=("y", "x"))
        _TARGET = da.rio.write_crs("EPSG:3035")
    return _TARGET


def _client():
    import planetary_computer as pc
    import pystac_client
    return pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)


def _search(cat, collection, start, end, tries=6):
    import time
    for i in range(tries):
        try:
            return list(cat.search(collections=[collection], bbox=BBOX_LL,
                                   datetime=f"{start}/{end}").items())
        except Exception:
            time.sleep(min(2 ** i * 3, 40))
    raise RuntimeError(f"MPC search failed for {collection} {start}/{end}")


def _read_tile(href, tries=5):
    """Open + fully read one COG tile into memory, retrying transient streaming errors (partial reads)."""
    import time
    from rasterio.errors import RasterioIOError
    for i in range(tries):
        try:
            da = rioxarray.open_rasterio(href, masked=True).squeeze("band", drop=True)
            da.load()                       # force the read NOW so we can retry on a streaming failure
            return da
        except (RasterioIOError, Exception) as e:
            if i == tries - 1:
                raise
            time.sleep(min(2 ** i * 3, 30))


def _mosaic_reproject(items, asset, vmin=None, vmax=None):
    """Mosaic the Spain sinusoidal tiles for one composite, reproject onto the 1 km grid → [NY,NX] float32.
    Masks DN outside [vmin,vmax] to NaN on the RAW tiles (before reproject) so MODIS QA fill values don't
    blend in or survive scaling."""
    from concurrent.futures import ThreadPoolExecutor
    from rioxarray.merge import merge_arrays

    def _one(it):
        t = _read_tile(it.assets[asset].href)
        if vmin is not None:
            t = t.where((t >= vmin) & (t <= vmax))   # fills/out-of-range → NaN
        return t.rio.write_nodata(np.nan)            # so reproject propagates NaN, not 0, to gaps
    with ThreadPoolExecutor(max_workers=4) as ex:   # tile reads are I/O-bound → parallelize (~4× faster)
        tiles = list(ex.map(_one, items))
    mos = merge_arrays(tiles, nodata=np.nan) if len(tiles) > 1 else tiles[0]
    out = mos.rio.reproject_match(_target(), nodata=np.nan)
    return out.values.astype(np.float32)


def build_range(start, end):
    """Return {feature: (dates_list, stack[n_comp,NY,NX])} of reprojected composites bracketing [start,end]."""
    import logging
    log = logging.getLogger("ingest_veg")
    cat = _client()
    s = (_dt.date.fromisoformat(start) - _dt.timedelta(days=20)).isoformat()
    e = (_dt.date.fromisoformat(end) + _dt.timedelta(days=20)).isoformat()
    res = {}
    for coll, assets in PRODUCTS.items():
        items = _search(cat, coll, s, e)
        bycomp = {}                                   # group items by composite (item.id: PRODUCT.AYYYYDDD.tile…)
        for it in items:
            bycomp.setdefault(it.id.split(".")[1], []).append(it)
        comp_dates = sorted(bycomp)
        log.info(f"{coll}: {len(items)} tiles over {len(comp_dates)} composites [{comp_dates[0]}..{comp_dates[-1]}]")
        for feat, (asset, scale, vmin, vmax) in assets.items():
            stack, dts = [], []
            for cd in comp_dates:
                try:
                    arr = _mosaic_reproject(bycomp[cd], asset, vmin, vmax) * scale
                except Exception as e:           # one bad/expired-URL composite must not abort the run
                    log.warning(f"  skip {asset} composite {cd}: {type(e).__name__} {e}"); continue
                yr, doy = int(cd[1:5]), int(cd[5:8])
                dts.append(_dt.date(yr, 1, 1) + _dt.timedelta(days=doy - 1)); stack.append(arr)
            if stack:
                res[feat] = (dts, np.stack(stack))
    return res


def daily_interp(dates_comp, stack, day):
    """Linear-interp the composite stack to a single calendar `day` (per-pixel over time)."""
    t = np.array([(d - _dt.date(2000, 1, 1)).days for d in dates_comp], float)
    td = (_dt.date.fromisoformat(day) - _dt.date(2000, 1, 1)).days
    if td <= t[0]:
        return stack[0]
    if td >= t[-1]:
        return stack[-1]
    j = np.searchsorted(t, td)
    w = (td - t[j - 1]) / (t[j] - t[j - 1])
    return (1 - w) * stack[j - 1] + w * stack[j]


def write_range(start, end, chunk_days=92):
    """Chunked + resumable backfill. Process in ~quarterly windows so MODIS COG search+read stay close in
    time (MPC signed URLs expire after hours) and days persist per chunk. Skips chunks already fully written."""
    import logging
    log = logging.getLogger("ingest_veg")
    BRONZE.mkdir(parents=True, exist_ok=True)
    d = _dt.date.fromisoformat(start); last = _dt.date.fromisoformat(end)
    while d <= last:
        cend = min(d + _dt.timedelta(days=chunk_days - 1), last)
        days = [(d + _dt.timedelta(days=k)).isoformat() for k in range((cend - d).days + 1)]
        if all((BRONZE / f"{x}.npz").exists() for x in days):
            log.info(f"{d}..{cend}: all exist, skip"); d = cend + _dt.timedelta(days=1); continue
        res = build_range(d.isoformat(), cend.isoformat())
        for ds in days:
            if (BRONZE / f"{ds}.npz").exists():
                continue
            out = {feat: daily_interp(dts, stk, ds).astype(np.float32) for feat, (dts, stk) in res.items()}
            np.savez_compressed(BRONZE / f"{ds}.npz", **out)
        log.info(f"{d}..{cend}: wrote {len(days)} days ({len(res)} veg features)")
        d = cend + _dt.timedelta(days=1)


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("ingest_veg")
    a = sys.argv
    if "--validate" in a:
        date = a[a.index("--validate") + 1]
        res = build_range(date, date)
        ndvi = daily_interp(*res["NDVI"], date)
        z = xr.open_zarr(str(grid.V1_CUBE), consolidated=True)
        fg = ndvi.reshape(grid.NY // 4, 4, grid.NX // 4, 4).mean((1, 3))   # to 4 km
        tv = z["NDVI"].sel(time=date, method="nearest").values.astype(float)
        m = np.isfinite(fg) & np.isfinite(tv)
        log.info(f"FGDC NDVI (MODIS/MPC) vs v1 NDVI {date}: MAE={np.abs(fg[m]-tv[m]).mean():.3f} "
                 f"corr={np.corrcoef(fg[m],tv[m])[0,1]:.4f} (v1 mean {tv[m].mean():.3f}, FGDC {fg[m].mean():.3f})")
        return
    if "--backfill" in a:
        i = a.index("--backfill"); write_range(a[i + 1], a[i + 2])
        log.info(f"wrote veg {a[i+1]}..{a[i+2]}"); return
    print("Use --validate DATE | --backfill START END", file=sys.stderr)


if __name__ == "__main__":
    main()
