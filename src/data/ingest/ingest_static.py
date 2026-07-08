"""FGDC slow-layer ingester — GHS-POP (population) + GHS-BUILT-S (built-up surface), with TEMPORAL
INTERPOLATION between editions (the "linear-growth" representation, user-prioritized).

Why GHS-POP instead of inheriting v1's popdens: v1's WorldPop stops at 2020 → stale for the FGDC's 2021→
live edge, and `popdens` is a top-2/3 ignition driver (ablation: human group +0.030 AP). GHS-POP R2023A is
5-yearly **1975–2030 incl. 2025/2030 projections** → covers the whole span + the live edge with no gap.
GHS-BUILT-S adds continuous WUI/structure exposure (ablation candidate; keep only if it earns it).

**Temporal interpolation (the key idea):** rather than step-snapping a cell to the nearest 5-yearly edition
(a Jan-1 discontinuity the model can spuriously key on), we LINEARLY INTERPOLATE between bracketing editions
so population/built-up grow smoothly day-to-day. Applies to these continuous layers (and, in P4, to CLC
*proportions*); categorical layers stay nearest-edition. Whether interpolation actually helps is an ABLATION
(step-snap vs interp) — but it can only show an effect on a MULTI-YEAR span, so that ablation runs on the
full backfill, not the 1-month dev slice (a single month barely moves between 5-yearly editions).

Source: JRC GHSL FTP, global 1 km Mollweide (ESRI:54009) GeoTIFF per epoch → clip Spain → reproject to the
1 km EPSG:3035 grid. PROJ pinned to rasterio's bundled db (conda-leak fix, as in ingest_veg).

CLI: --epochs (download+reproject+cache all epochs) | --validate DATE (interp popdens, compare to v1)
"""
from __future__ import annotations
import os
import sys
import zipfile
from pathlib import Path

import rasterio as _rio
_RIO_PROJ = os.path.join(os.path.dirname(_rio.__file__), "proj_data")
if os.path.isdir(_RIO_PROJ):
    os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = _RIO_PROJ

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import datetime as _dt
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from src.data.ingest import grid

BRONZE = grid.ROOT / "data" / "bronze" / "fireguard" / "ghsl"
EPOCHS = [2010, 2015, 2020, 2025, 2030]          # R2023A 5-yearly (2025/2030 = projections)
PRODUCTS = {"popdens": "GHS_POP", "built_s": "GHS_BUILT_S"}
RES = 1000
BASE = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
_TARGET = None


def _target():
    global _TARGET
    if _TARGET is None:
        da = xr.DataArray(np.zeros(grid.shape(), np.float32),
                          coords={"y": grid.y_coords(), "x": grid.x_coords()}, dims=("y", "x"))
        _TARGET = da.rio.write_crs("EPSG:3035")
    return _TARGET


def _url(prod, year):
    stem = f"{prod}_E{year}_GLOBE_R2023A_54009_{RES}_V1_0"
    return f"{BASE}/{prod}_GLOBE_R2023A/{prod}_E{year}_GLOBE_R2023A_54009_{RES}/V1-0/{stem}.zip", stem


def _download(prod, year):
    """Download + extract the global 1 km GeoTIFF for one product/epoch (cached in bronze)."""
    import requests
    BRONZE.mkdir(parents=True, exist_ok=True)
    url, stem = _url(prod, year)
    tif = BRONZE / f"{stem}.tif"
    if tif.exists():
        return tif
    zp = BRONZE / f"{stem}.zip"
    if not zp.exists():
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(zp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    with zipfile.ZipFile(zp) as z:
        name = next(n for n in z.namelist() if n.endswith(".tif"))
        z.extract(name, BRONZE)
        (BRONZE / name).rename(tif)
    zp.unlink(missing_ok=True)
    return tif


def _spain_mollweide_box():
    """Grid bounds (EPSG:3035) → ESRI:54009 (Mollweide) clip box, with margin."""
    from pyproj import Transformer
    w, s, e, n = grid.bounds()
    tr = Transformer.from_crs("EPSG:3035", "ESRI:54009", always_xy=True)
    xs, ys = tr.transform([w, w, e, e], [s, n, s, n])
    pad = 20000.0
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def reproject_epoch(prod, year):
    """Download → clip Spain (Mollweide window) → reproject to the 1 km EPSG:3035 grid → [NY,NX] float32."""
    tif = _download(prod, year)
    da = rioxarray.open_rasterio(tif, masked=True).squeeze("band", drop=True)
    da = da.rio.clip_box(*_spain_mollweide_box())          # windowed: avoids reprojecting the globe
    da = da.where(da >= 0)                                  # GHSL nodata is negative
    out = da.rio.reproject_match(_target(), nodata=np.nan)
    return out.values.astype(np.float32)


def build_epochs(prod):
    """Cache per-epoch reprojected layers for one product → {year: grid[NY,NX]}."""
    import logging
    log = logging.getLogger("ingest_static")
    out = {}
    for y in EPOCHS:
        cache = BRONZE / f"{PRODUCTS[prod]}_{prod}_{y}_1km3035.npz"
        if cache.exists():
            out[y] = np.load(cache)["a"]
        else:
            out[y] = reproject_epoch(PRODUCTS[prod], y)
            grid.atomic_savez(cache, a=out[y])
        log.info(f"  {prod} {y}: mean={np.nanmean(out[y]):.2f} max={np.nanmax(out[y]):.1f}")
    return out


def load_cached_epochs(prod):
    """{year: grid[NY,NX]} for whatever epoch caches already exist (no download) — used by build_silver."""
    import re
    out = {}
    if not BRONZE.exists():
        return out
    for p in BRONZE.glob(f"*_{prod}_*_1km3035.npz"):
        m = re.search(rf"_{prod}_(\d{{4}})_1km3035", p.name)
        if m:
            out[int(m.group(1))] = np.load(p)["a"]
    return out


def interp_to_date(epoch_arrays, date):
    """Linear-interpolate the epoch stack to a calendar `date` (the temporal-interpolation feature).
    Clamps outside the epoch range (no extrapolation)."""
    yrs = sorted(epoch_arrays)
    frac_year = _dt.date.fromisoformat(date).timetuple().tm_yday / 365.25
    yd = _dt.date.fromisoformat(date).year + frac_year
    if yd <= yrs[0]:
        return epoch_arrays[yrs[0]]
    if yd >= yrs[-1]:
        return epoch_arrays[yrs[-1]]
    hi = next(y for y in yrs if y >= yd); lo = max(y for y in yrs if y <= yd)
    if hi == lo:
        return epoch_arrays[hi]
    w = (yd - lo) / (hi - lo)
    return ((1 - w) * epoch_arrays[lo] + w * epoch_arrays[hi]).astype(np.float32)


def nearest_to_date(epoch_arrays, date):
    """Step-snap to the nearest edition (the v1-style baseline, for the interpolation ablation)."""
    yrs = sorted(epoch_arrays)
    yd = _dt.date.fromisoformat(date).year
    return epoch_arrays[min(yrs, key=lambda y: abs(y - yd))]


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("ingest_static")
    a = sys.argv
    if "--validate" in a:
        date = a[a.index("--validate") + 1]
        # only need the bracketing epochs for one date
        global EPOCHS
        yd = _dt.date.fromisoformat(date).year
        EPOCHS = sorted({min([e for e in [2010, 2015, 2020, 2025, 2030]], key=lambda e: abs(e - yd)),
                         *[e for e in [2010, 2015, 2020] if e <= yd][-1:],
                         *[e for e in [2015, 2020, 2025] if e >= yd][:1]})
        pop = build_epochs("popdens")
        interp = interp_to_date(pop, date); step = nearest_to_date(pop, date)
        z = xr.open_zarr(str(grid.V1_CUBE), consolidated=True)
        fg = interp.reshape(grid.NY // 4, 4, grid.NX // 4, 4).mean((1, 3))
        if "popdens" in z:
            tv = z["popdens"].values.astype(float)
            m = np.isfinite(fg) & np.isfinite(tv) & (tv >= 0)
            cc = np.corrcoef(np.log1p(fg[m]), np.log1p(tv[m]))[0, 1]
            log.info(f"GHS-POP interp {date} vs v1 popdens: log-corr={cc:.4f} "
                     f"(GHS mean {np.nanmean(fg):.1f}, v1 {np.nanmean(tv[m]):.1f})")
        log.info(f"interp vs step-snap mean abs diff: {np.nanmean(np.abs(interp-step)):.3f} "
                 f"(near-zero within one edition window; the interpolation ablation needs a MULTI-YEAR span)")
        return
    if "--epochs" in a:
        for prod in PRODUCTS:
            log.info(f"building {prod} epochs {EPOCHS}"); build_epochs(prod)
        return
    print("Use --epochs | --validate DATE", file=sys.stderr)


if __name__ == "__main__":
    main()
