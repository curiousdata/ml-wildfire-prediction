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

SILVER = grid.ROOT / "data" / "silver" / "FireGuard.zarr"


def _dates_present():
    """Dates with BOTH a weather and a fire bronze npz (the dynamic families needed per day)."""
    wd = {p.stem for p in IW.BRONZE.glob("*.npz")} if IW.BRONZE.exists() else set()
    fd = {p.stem for p in IF.BRONZE.glob("*.npz")} if IF.BRONZE.exists() else set()
    return sorted(wd & fd)


def _load_static(refine=True):
    """v1 static vars (dims y,x) refined ×4 onto the 1 km grid → {name: array[NY,NX]}."""
    z = xr.open_zarr(str(grid.V1_CUBE), consolidated=True)
    out = {}
    for v in z.data_vars:
        if "time" not in z[v].dims:                       # static layer
            arr = np.asarray(z[v].values, np.float32)
            out[v] = grid.refine_to_1km(arr) if refine else arr
    return out


def build(start, end, out=SILVER, with_static=True):
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("build_silver")
    from numcodecs import Blosc

    dates = [d for d in _dates_present() if start <= d <= end]
    if not dates:
        raise SystemExit(f"no bronze days with both weather+fire in [{start},{end}] — run the ingesters first")
    log.info(f"assembling {len(dates)} days [{dates[0]}..{dates[-1]}] on the 1 km grid {grid.shape()}")
    gx, gy = grid.x_coords(), grid.y_coords()
    times = pd.to_datetime(dates)

    # --- dynamic: stack per-day bronze npz into [time, y, x] per feature ---
    wkeys = list(np.load(IW.BRONZE / f"{dates[0]}.npz").files)
    veg_present = (IV.BRONZE / f"{dates[0]}.npz").exists()
    vkeys = list(np.load(IV.BRONZE / f"{dates[0]}.npz").files) if veg_present else []
    if veg_present:
        log.info(f"including {len(vkeys)} vegetation features: {vkeys}")
    dyn = {k: np.empty((len(dates), grid.NY, grid.NX), np.float32) for k in wkeys + ["is_fire"] + vkeys}
    for i, d in enumerate(dates):
        w = np.load(IW.BRONZE / f"{d}.npz")
        for k in wkeys:
            dyn[k][i] = w[k]
        dyn["is_fire"][i] = np.load(IF.BRONZE / f"{d}.npz")["is_fire"]
        if vkeys:
            vv = np.load(IV.BRONZE / f"{d}.npz")
            for k in vkeys:
                dyn[k][i] = vv[k] if k in vv.files else np.nan

    data_vars = {k: (("time", "y", "x"), v) for k, v in dyn.items()}
    if with_static:
        stat = _load_static()
        log.info(f"inheriting {len(stat)} static layers from v1 (refined ×4)")
        for k, v in stat.items():
            data_vars[k] = (("y", "x"), v)

    ds = xr.Dataset(data_vars, coords={"time": times, "y": gy, "x": gx})
    ds.attrs.update(title="Fire Guard Datacube (silver, 1 km)", crs="EPSG:3035",
                    dynamic_source="Open-Meteo ERA5 (weather) + FIRMS VIIRS_SNPP (fire)",
                    static_source="inherited from IberFire v1 (refined ×4, lossless at 4 km gold)")
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    enc = {v: {"compressor": comp, "chunks": (1, grid.NY, grid.NX) if "time" in ds[v].dims
               else (grid.NY, grid.NX)} for v in ds.data_vars}
    out = Path(out)
    if out.exists():
        import shutil; shutil.rmtree(out)
    ds.to_zarr(str(out), mode="w", zarr_format=2, encoding=enc, consolidated=True)
    log.info(f"wrote {out}  ({len(ds.data_vars)} vars: {len(dyn)} dynamic + "
             f"{len(ds.data_vars)-len(dyn)} static, {len(dates)} days)")
    return out


def main():
    a = sys.argv
    start = a[a.index("--start") + 1] if "--start" in a else "2016-08-01"
    end = a[a.index("--end") + 1] if "--end" in a else "2016-08-31"
    build(start, end, with_static="--no-static" not in a)


if __name__ == "__main__":
    main()
