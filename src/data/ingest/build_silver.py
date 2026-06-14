"""Assemble the FGDC silver cube — 1 km daily EPSG:3035 Zarr — from bronze dynamic feeds + inherited static.

Composition (see CHANGES.md FGDC entry):
  * DYNAMIC (recollected from operational providers, per-day bronze npz): weather (ingest_weather) + fire
    (ingest_fire) [+ vegetation in P2]. These are the families that DRIFT between train and serve, so they
    must come from the same source we serve from.
  * STATIC (inherited from v1, refined ×4 to 1 km): terrain, CORINE one-hots/proportions, popdens, masks,
    dist_to_* — 234 time-invariant layers. Static features do NOT drift train↔serve, and since gold
    coarsens 1 km→4 km by block-mean, refining v1's 4 km static ×4 is LOSSLESS at the gold target. (P3 can
    re-derive them natively from DEM/CORINE if ever needed; it would reproduce ~the same values.)

Engineered DYNAMIC features (kbdi, vpd, ffwi, precip_sum_*, dist_to_fire, time_since_last_fire, anomalies,
doy/dow…) are NOT stored here — they're computed downstream by coarsen.py + add_engineered_features.py (P4),
exactly as for v1, so the same machinery and feature-set apply.

Writes zarr_format=2 + Blosc(zstd) to match the v1 cubes (zarr 3.x's default v3 rejects numcodecs.Blosc).

CLI: --start YYYY-MM-DD --end YYYY-MM-DD [--out PATH] [--no-static]
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
import xarray as xr

from src.data.ingest import grid
from src.data.ingest import ingest_weather as IW
from src.data.ingest import ingest_fire as IF
from src.data.ingest import ingest_veg as IV
from src.data.ingest import ingest_static as IS
import scripts.fetch_openmeteo as OM

SILVER = grid.ROOT / "data" / "silver" / "FireGuard.zarr"


def _dates_present():
    """Dates with BOTH a weather and a fire bronze npz (the dynamic families needed per day)."""
    wd = {p.stem for p in IW.BRONZE.glob("*.npz")} if IW.BRONZE.exists() else set()
    fd = {p.stem for p in IF.BRONZE.glob("*.npz")} if IF.BRONZE.exists() else set()
    return sorted(wd & fd)


def _load_static(refine=True):
    """v1 static vars (dims y,x) refined ×4 onto the 1 km grid → {name: array[NY,NX]}. EXCLUDES v1's
    popdens_* (WorldPop, stops 2020) — the FGDC sources population from GHS-POP instead (added per-day,
    temporally interpolated, in build())."""
    z = xr.open_zarr(str(grid.V1_CUBE), consolidated=True)
    out = {}
    for v in z.data_vars:
        if "time" not in z[v].dims and not v.startswith("popdens"):
            arr = np.asarray(z[v].values, np.float32)
            out[v] = grid.refine_to_1km(arr) if refine else arr
    return out


def _dyn_chunk(cdates, wkeys, vkeys, ghs, wregrid=None):
    """Build the dynamic [time,y,x] arrays for a small set of dates (bounded memory). Weather bronze is
    stored at native ~0.25°; `wregrid` (OM.make_regridder) upsamples each native vector to the 1 km grid
    on read. (Legacy 1 km bronze, where each value is already a grid, passes through when wregrid is None.)"""
    dyn = {k: np.empty((len(cdates), grid.NY, grid.NX), np.float32) for k in wkeys + ["is_fire"] + vkeys}
    for i, d in enumerate(cdates):
        w = np.load(IW.BRONZE / f"{d}.npz")
        for k in wkeys:
            dyn[k][i] = wregrid(w[k]) if wregrid is not None else w[k]
        dyn["is_fire"][i] = np.load(IF.BRONZE / f"{d}.npz")["is_fire"]
        if vkeys:
            vv = np.load(IV.BRONZE / f"{d}.npz")
            for k in vkeys:
                dyn[k][i] = vv[k] if k in vv.files else np.nan
    for p, epochs in ghs.items():                         # GHS interpolated per day
        arr = np.empty((len(cdates), grid.NY, grid.NX), np.float32)
        for i, d in enumerate(cdates):
            arr[i] = IS.interp_to_date(epochs, d)
        dyn[p] = arr
    return dyn


def build(start, end, out=SILVER, with_static=True, chunk_days=20):
    """Assemble silver, writing INCREMENTALLY along time (chunk_days at a time) so memory stays bounded —
    a full multi-year build can't hold all days at once (184 days × 27 vars ≈ 22 GB)."""
    import logging
    import shutil
    import zarr
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("build_silver")
    from numcodecs import Blosc

    dates = [d for d in _dates_present() if start <= d <= end]
    if not dates:
        raise SystemExit(f"no bronze days with both weather+fire in [{start},{end}] — run the ingesters first")
    gx, gy = grid.x_coords(), grid.y_coords()
    w0 = np.load(IW.BRONZE / f"{dates[0]}.npz")
    wkeys = [k for k in w0.files if k not in (IW.WLON, IW.WLAT)]      # feature keys only (drop coord arrays)
    wregrid = OM.make_regridder(w0[IW.WLON], w0[IW.WLAT], gx, gy) if IW.WLON in w0.files else None  # native→1km
    vkeys = list(np.load(IV.BRONZE / f"{dates[0]}.npz").files) if (IV.BRONZE / f"{dates[0]}.npz").exists() else []
    ghs = {p: e for p, e in ((p, IS.load_cached_epochs(p)) for p in ("popdens", "built_s")) if e}
    static = _load_static() if with_static else {}
    n_dyn = len(wkeys) + 1 + len(vkeys) + len(ghs)
    log.info(f"assembling {len(dates)} days [{dates[0]}..{dates[-1]}] | {n_dyn} dynamic "
             f"(weather {len(wkeys)}{' native→1km' if wregrid is not None else ''} + fire + veg {len(vkeys)} "
             f"+ GHS {list(ghs)}) + {len(static)} static; chunked {chunk_days}d")
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    for ci, c0 in enumerate(range(0, len(dates), chunk_days)):
        cdates = dates[c0:c0 + chunk_days]
        dyn = _dyn_chunk(cdates, wkeys, vkeys, ghs, wregrid)
        ds = xr.Dataset({k: (("time", "y", "x"), v) for k, v in dyn.items()},
                        coords={"time": pd.to_datetime(cdates), "y": gy, "x": gx})
        if ci == 0:                                       # first write: + static + encoding + attrs
            for k, v in static.items():
                ds[k] = (("y", "x"), v)
            ds.attrs.update(title="Fire Guard Datacube (silver, 1 km)", crs="EPSG:3035",
                            dynamic_source="Open-Meteo ERA5 + FIRMS VIIRS + MODIS/MPC veg + GHS-POP/BUILT",
                            static_source="inherited from IberFire v1 (refined ×4, lossless at 4 km gold)")
            enc = {v: {"compressor": comp, "chunks": (1, grid.NY, grid.NX) if "time" in ds[v].dims
                       else (grid.NY, grid.NX)} for v in ds.data_vars}
            ds.to_zarr(str(out), mode="w", zarr_format=2, encoding=enc, consolidated=False)
        else:
            ds.to_zarr(str(out), append_dim="time", consolidated=False)
        log.info(f"  chunk {cdates[0]}..{cdates[-1]} ({len(cdates)}d) written")
    zarr.consolidate_metadata(str(out))
    log.info(f"wrote {out}  ({n_dyn} dynamic + {len(static)} static, {len(dates)} days, chunked)")
    return out


def main():
    a = sys.argv
    start = a[a.index("--start") + 1] if "--start" in a else "2016-08-01"
    end = a[a.index("--end") + 1] if "--end" in a else "2016-08-31"
    build(start, end, with_static="--no-static" not in a)


if __name__ == "__main__":
    main()
