"""Coarsen the FGDC silver (1 km) → gold (4 km) with semantic block-pooling.

Separate from v1's `scripts/coarsen.py` (non-destructive): v1 decomposes wind_direction→u/v AT coarsen time,
but the FGDC silver already stores wind u/v directly (decomposed hourly, the pooling-safe form), so we just
mean-pool them. Pooling rules mirror v1:
  * is_fire (label) and any *_max feature  → MAX  (preserve rare positives / daily extremes)
  * any *_min feature                       → MIN
  * everything else (incl. u/v, proportions, terrain, masks) → MEAN
The 1 km grid is aligned to v1, so factor-4 block-means land exactly on v1's coarse4 grid.
Engineered features (kbdi, vpd, ffwi, precip_sum_*, dist_to_fire, time_since_last_fire, doy/dow, anomalies…)
are added AFTER, by add_engineered_features (P4) — same as v1.

CLI: --factor 4 [--in silver/FireGuard.zarr] [--out gold/FireGuard_coarse{F}.zarr] [--overwrite]
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import xarray as xr

from src.data.ingest import grid

SILVER = grid.ROOT / "data" / "silver" / "FireGuard.zarr"
GOLD_DIR = grid.ROOT / "data" / "gold"


def _pool_rule(name: str) -> str:
    if name == "is_fire" or name.endswith("_max"):
        return "max"
    if name.endswith("_min"):
        return "min"
    return "mean"


def coarsen(infile=SILVER, factor=4, out=None, overwrite=False):
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("coarsen")
    from numcodecs import Blosc

    out = Path(out) if out else GOLD_DIR / f"FireGuard_coarse{factor}.zarr"
    if out.exists():
        if not overwrite:
            raise SystemExit(f"{out} exists — pass --overwrite")
        import shutil; shutil.rmtree(out)
    ds = xr.open_zarr(str(infile), consolidated=True)
    assert grid.NY % factor == 0 and grid.NX % factor == 0, "grid not divisible by factor"

    pooled = {}
    counts = {"max": 0, "min": 0, "mean": 0}
    for v in ds.data_vars:
        rule = _pool_rule(v); counts[rule] += 1
        c = ds[v].coarsen(y=factor, x=factor, boundary="exact")
        pooled[v] = getattr(c, rule)()
    g = xr.Dataset(pooled)
    g.attrs.update(ds.attrs); g.attrs["coarsen_factor"] = factor
    g.attrs["coarsen_pooling"] = "is_fire/*_max=max, *_min=min, else mean"

    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    enc = {v: {"compressor": comp,
               "chunks": (1, g.sizes["y"], g.sizes["x"]) if "time" in g[v].dims else (g.sizes["y"], g.sizes["x"])}
           for v in g.data_vars}
    g.to_zarr(str(out), mode="w", zarr_format=2, encoding=enc, consolidated=True)
    log.info(f"wrote {out}  {dict(g.sizes)}  ({len(g.data_vars)} vars: "
             f"{counts['max']} max-pooled, {counts['min']} min, {counts['mean']} mean)")
    return out


def main():
    a = sys.argv
    factor = int(a[a.index("--factor") + 1]) if "--factor" in a else 4
    infile = a[a.index("--in") + 1] if "--in" in a else SILVER
    out = a[a.index("--out") + 1] if "--out" in a else None
    coarsen(infile, factor, out, overwrite="--overwrite" in a)


if __name__ == "__main__":
    main()
