"""Multi-satellite VIIRS fire grids — shared dependency for the multi-sat source ablation (see ABLATIONS.md,
2026-07-04). Fetches NOAA-20 (2018-04→2026-04, SP archive) and NOAA-21 (2024-01→2026-04, NRT — no SP archive
exists, but NRT retains N21's full history) active fire, filters conf≥nominal, grids to the cube's 4 km grid,
aligns to cube time → caches bool arrays [T,y,x] under data/cache/multisat/ (git-ignored).

The cube already holds S-NPP as `is_fire`; these two arrays let `multisat_label_matrix.py` build the union
timelines (4-pass = S-NPP∪N20, 6-pass = ∪N21) without touching the cube. Idempotent-ish: re-run to refresh;
transient FIRMS HTTPErrors skip a 5-day window (re-run to backfill the gaps).

  python scripts/experiments/multisat_fire_fetch.py
"""
from __future__ import annotations
import datetime as dt
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass
import numpy as np
import pandas as pd
import xarray as xr

from src.data import fetch as FB
from src.data import metrics as T
from src.data.ingest import grid

CACHE = T.project_root / "data" / "cache" / "multisat"
BBOX = (-10.0, 35.0, 5.0, 44.5)
CONF = {"n", "h"}
END = dt.date(2026, 4, 30)                              # S-NPP/N20 SP-archive horizon
JOBS = [("VIIRS_NOAA20_SP", "n20", dt.date(2018, 4, 1)),
        ("VIIRS_NOAA21_NRT", "n21", dt.date(2024, 1, 17))]


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("multisat_fetch")
    key = os.getenv("FIRMS_MAP_KEY")
    if not key:
        raise SystemExit("set FIRMS_MAP_KEY")
    CACHE.mkdir(parents=True, exist_ok=True)
    # --factor F: grid to data/gold/FireGuard_coarse{F}.zarr → data/cache/multisat/fire_{name}_{F}km.npy
    factor = int(sys.argv[sys.argv.index("--factor") + 1]) if "--factor" in sys.argv else 4
    suffix = "" if factor == 4 else f"_{factor}km"
    z = xr.open_zarr(str(grid.ROOT / "data" / "gold" / f"FireGuard_coarse{factor}.zarr"), consolidated=True)
    gx, gy = z["x"].values.astype(float), z["y"].values.astype(float)
    tindex = {d.date().isoformat(): i for i, d in enumerate(pd.DatetimeIndex(z["time"].values))}
    T_, Y, X = z.sizes["time"], z.sizes["y"], z.sizes["x"]

    for src, name, start in JOBS:
        arr = np.zeros((T_, Y, X), bool); t0 = time.time(); d = start; win = ncell = nfail = 0
        log.info(f"[{name}] {src} {start}..{END}")
        while d <= END:
            w = min(5, (END - d).days + 1)
            try:
                df = FB.fetch_firms(key, d.isoformat(), src=src, bbox=BBOX, days=w)
            except Exception as e:
                nfail += 1; log.warning(f"  {d} fetch fail ({type(e).__name__}) — window skipped")
                d += dt.timedelta(days=w); win += 1; continue
            if df is not None and not df.empty and "confidence" in df.columns:
                c = df["confidence"].astype(str).str.strip().str.lower(); df = df[c.isin(CONF)]
            for k in range(w):
                day = (d + dt.timedelta(days=k)).isoformat(); i = tindex.get(day)
                if i is None:
                    continue
                sub = df[df["acq_date"] == day] if (df is not None and "acq_date" in df.columns) else None
                if sub is not None and not sub.empty:
                    g = FB.fires_to_grid(sub["longitude"].values, sub["latitude"].values, gx, gy) > 0.5
                    arr[i] = g; ncell += int(g.sum())
            win += 1; d += dt.timedelta(days=w)
            if win % 40 == 0:
                log.info(f"  {name} {day} ({win} win, {time.time()-t0:.0f}s, {ncell} cell-hits)")
        np.save(CACHE / f"fire_{name}{suffix}.npy", arr)
        log.info(f"[{name}] DONE: {ncell} cell-hits, {nfail} window-fails → {CACHE/f'fire_{name}{suffix}.npy'} "
                 f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
