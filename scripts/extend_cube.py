"""Ephemeral SERVE engine — the FGDC v2 live edge (writes NOTHING to the cube).

To predict t+1 at the forecast edge, build day t's feature vector in memory: fetch **forecast** weather + **FIRMS
NRT** fire for the band beyond the settled cube `(cube_last, t]`, **carry** the slow feeds (veg, GHS) forward from
the cube's last row, seed the engine from the cube tail, and run `update_edge.compute_edge_engineered` over the
band. All in memory — the cube is never touched. This is the **serve** tier (see the `lambda-architecture-fgdc`
memory): forecast + cube-tail seed → t+1, ephemeral. It **retires the old Option C** (which appended provisional
rows to gold) — no provisional cube state to reconcile.

`serve_edge(cube, t)` → `(issue_date, {var: grid[NY,NX]})` of day t's raw+engineered fields (None if t is already
in the settled cube → caller predicts from the cube row). `daily_job --mode live` calls it, predicts, logs.

Reuse: `src.data.fetch` (Open-Meteo + FIRMS) + `ingest_weather.daily_point_features` (forecast weather → 4 km),
`ingest_fire` (NRT fire → 4 km), `update_edge.compute_edge_engineered` (the one engine).
NB: a live serve hits the network (Open-Meteo forecast + FIRMS NRT). Veg/GHS are slow → carried, not fetched.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import xarray as xr

from src.data import fetch as OM
import scripts.update_edge as UE
from src.data.ingest import grid
from src.data.ingest import ingest_weather as IW
from src.data.ingest import ingest_fire as IF

CUBE = grid.ROOT / "data" / "gold" / "FireGuard_coarse4.zarr"
FETCH_STEP = 0.25                                    # native ~0.25°


def _band_raw(z, band, gx, gy):
    """Raw dynamic fields for the edge band → {var: (n_band, NY, NX)}. Weather from the Open-Meteo **ARCHIVE**
    (free endpoint, seamless to ~today — the recent ~5 d are IFS-model-filled), fire=FIRMS NRT (both fetched as a
    SINGLE range request — not per-day — to stay under rate limits), everything else (veg, GHS, and engineered
    placeholders) CARRIED from the cube's last row (the engine overwrites the engineered; veg/GHS are slow-varying
    so carry-forward is the right estimate for a ≤~5 d edge). Archive (not forecast) because the model predicts
    t+1 from TODAY's features — today's weather is in the archive — so serve needs NO commercial forecast key, and
    it matches the product family the batch later settles (better train/serve consistency)."""
    import os
    from src.data import fetch as FB
    gold_time = [v for v in z.data_vars if "time" in z[v].dims]
    last = {v: z[v].isel(time=-1).values for v in gold_time}
    start, end = band[0].date().isoformat(), band[-1].date().isoformat()
    # ONE archive-weather range fetch + ONE precip range fetch (whole band; seamless-to-today, free endpoint)
    plon, plat, hv, times = OM.fetch_grid_hourly_range(start, end, IW.HOURLY_VARS, step=FETCH_STEP, source="archive")
    _, _, dly, _ = OM.fetch_grid_range(start, end, ["precipitation_sum"], step=FETCH_STEP, source="archive")
    rg = OM.make_regridder(np.asarray(plon, float), np.asarray(plat, float), gx, gy)
    precip = dly["precipitation_sum"]                                # (n_band, n_pts)
    # FIRMS NRT in ≤5-day windows (the area API caps day_range at 5), concat, split per acq_date
    key = os.getenv("FIRMS_MAP_KEY")
    fdf = None
    if key:
        bd = [d.date() for d in band]
        parts, i = [], 0
        while i < len(bd):
            w = min(5, len(bd) - i)
            parts.append(IF._filter_conf(FB.fetch_firms(key, bd[i].isoformat(), src=IF.SRC_NRT, bbox=IF.BBOX_LL, days=w)))
            i += w
        fdf = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if parts else None

    raw = {v: np.empty((len(band), len(gy), len(gx)), np.float32) for v in gold_time}
    for k, d in enumerate(band):
        ds = d.date().isoformat()
        wday = {f: rg(vec) for f, vec in IW.daily_point_features(hv, times, precip[k], ds).items()}
        fd = fdf[fdf["acq_date"] == ds] if (fdf is not None and "acq_date" in fdf.columns) else None
        fire = (FB.fires_to_grid(fd["longitude"].values, fd["latitude"].values, gx, gy)
                if (fd is not None and not fd.empty) else np.zeros((len(gy), len(gx)), np.float32))
        for v in gold_time:
            raw[v][k] = wday[v] if v in wday else (fire if v == "is_fire" else last[v])
    return raw


def serve_edge(cube_path=CUBE, t=None):
    """Ephemeral feature build for day `t` (predict t+1). Returns (issue_date, {var: grid}) of day t's raw+
    engineered fields, computed over the forecast band seeded by the cube tail — NO cube write. Returns
    (issue_date, None) if `t` is already in the settled cube (caller should predict from the cube row)."""
    import logging
    log = logging.getLogger("serve")
    z = xr.open_zarr(str(cube_path), consolidated=True)
    gx, gy = z["x"].values.astype(float), z["y"].values.astype(float)
    cube_last = pd.Timestamp(z["time"].values[-1])
    t = pd.Timestamp(t)
    if t <= cube_last:
        log.info(f"serve: {t.date()} already settled in cube — predict from cube row"); return str(t.date()), None
    import time
    band = pd.date_range(cube_last + pd.Timedelta(days=1), t)
    n = len(band); i0 = z.sizes["time"]
    log.info(f"serve: forecast band {band[0].date()}..{band[-1].date()} ({n}d) after settled {cube_last.date()} — fetching")
    tf = time.time()
    raw = _band_raw(z, band, gx, gy)
    log.info(f"  band raw fetched ({time.time()-tf:.0f}s); running engine")
    gold_time = list(raw.keys())
    band_ds = xr.Dataset({v: (("time", "y", "x"), raw[v]) for v in gold_time},
                         coords={"time": band, "y": z["y"], "x": z["x"]})
    z_ext = xr.concat([z[gold_time], band_ds], dim="time")            # virtual cube: history + forecast band
    for sv in [v for v in z.data_vars if "time" not in z[v].dims]:
        z_ext[sv] = z[sv]
    te = time.time()
    eng = UE.compute_edge_engineered(z_ext, i0, n)                   # engineered over the band (ephemeral)
    log.info(f"  engine done ({time.time()-te:.0f}s)")
    jt = n - 1                                                        # day t = last band index
    fields = {v: (eng[v][jt] if v in eng else raw[v][jt]) for v in gold_time}
    return str(t.date()), fields


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Ephemeral serve: build day-t edge fields (predict t+1). No cube write.")
    ap.add_argument("--to", help="issue date t YYYY-MM-DD (default: latest_complete_fire_date)")
    args = ap.parse_args()
    from scripts.daily_job import latest_complete_fire_date
    t = args.to or str(latest_complete_fire_date())
    issue, fields = serve_edge(CUBE, t)
    if fields is None:
        print(f"{issue}: already settled — no forecast band"); return
    land = np.nan_to_num(xr.open_zarr(str(CUBE), consolidated=True)["is_spain"].values) > 0.5
    print(f"issue {issue} → predict {pd.Timestamp(issue)+pd.Timedelta(days=1):%Y-%m-%d} | "
          f"{len(fields)} fields; sample over land:")
    for v in ("dist_to_fire", "kbdi", "spi_90d", "ffwi", "is_fire", "t2m_max"):
        if v in fields:
            a = fields[v][land]
            print(f"  {v:14} finite={np.isfinite(a).mean():.2f} mean={np.nanmean(a):.3f}")


if __name__ == "__main__":
    main()
