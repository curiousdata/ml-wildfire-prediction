"""Multi-satellite label DENSITY measurement (see ABLATIONS.md, 2026-07-04) — does each added VIIRS bird
densify the rare fire-positive label? Math-only cell-day count over 2024-01-17→2026-04-30 on the 4 km FireGuard
land grid (conf≥nominal), mirroring the earlier NOAA-20 density study for the 3rd bird.

S-NPP & NOAA-20 use the SP science archive; NOAA-21 has NO SP archive → uses NRT (retained for its full history).
Reports 2-pass (S-NPP) → 4-pass (+N20) → 6-pass (+N21) cell-days and each bird's marginal add.

  python scripts/experiments/multisat_density.py
"""
from __future__ import annotations
import datetime as dt
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass
import numpy as np
import xarray as xr

from src.data import fetch as FB
from src.data.ingest import grid

START, END = dt.date(2024, 1, 17), dt.date(2026, 4, 30)          # N21 start … SP-archive horizon
SRC = {"snpp": "VIIRS_SNPP_SP", "n20": "VIIRS_NOAA20_SP", "n21": "VIIRS_NOAA21_NRT"}
BBOX = (-10.0, 35.0, 5.0, 44.5)
CONF = {"n", "h"}


def main():
    key = os.getenv("FIRMS_MAP_KEY")
    if not key:
        raise SystemExit("set FIRMS_MAP_KEY")
    z = xr.open_zarr(str(grid.ROOT / "data" / "gold" / "FireGuard_coarse4.zarr"), consolidated=True)
    gx, gy = z["x"].values.astype(float), z["y"].values.astype(float)
    land = np.nan_to_num(z["is_spain"].values) > 0.5
    print(f"window {START}..{END} | 4km land cells = {int(land.sum())}", flush=True)

    def grid_day(df, day):
        if df is None or df.empty:
            return np.zeros(land.shape, bool)
        sub = df[df["acq_date"] == day] if "acq_date" in df.columns else df
        if sub.empty:
            return np.zeros(land.shape, bool)
        c = sub["confidence"].astype(str).str.strip().str.lower()
        sub = sub[c.isin(CONF)]
        return (FB.fires_to_grid(sub["longitude"].values, sub["latitude"].values, gx, gy) > 0.5) & land

    acc = defaultdict(lambda: dict(cd2=0, cd4=0, cd6=0, add21=0, resc=0, ndays=0))
    tot = dict(cd2=0, cd4=0, cd6=0, add21=0, resc=0, ndays=0, fail=0)
    t0 = time.time(); d = START; win = 0
    while d <= END:
        wdays = min(5, (END - d).days + 1)
        dates = [(d + dt.timedelta(days=k)).isoformat() for k in range(wdays)]
        try:
            dfs = {kb: FB.fetch_firms(key, d.isoformat(), src=s, bbox=BBOX, days=wdays) for kb, s in SRC.items()}
        except Exception as e:
            tot["fail"] += 1; print(f"  [win {win}] {d} fail {type(e).__name__}", flush=True)
            d += dt.timedelta(days=wdays); win += 1; continue
        for day in dates:
            a, b, c = grid_day(dfs["snpp"], day), grid_day(dfs["n20"], day), grid_day(dfs["n21"], day)
            u4, u6 = a | b, a | b | c
            for bucket in (acc[day[:4]], tot):
                bucket["cd2"] += int(a.sum()); bucket["cd4"] += int(u4.sum()); bucket["cd6"] += int(u6.sum())
                bucket["add21"] += int((u6 & ~u4).sum())
                bucket["resc"] += int((not u4.any()) and c.any())
                bucket["ndays"] += 1
        win += 1; d += dt.timedelta(days=wdays)
        if win % 20 == 0:
            print(f"  … {day} ({win} win, {time.time()-t0:.0f}s, cd6={tot['cd6']})", flush=True)

    pct = lambda n, dvd: 100.0 * n / dvd if dvd else 0.0
    print("\n" + "=" * 74)
    print("MULTI-SAT DENSITY — fire-positive cell-days, 4km FireGuard land grid (conf≥nominal)")
    print("=" * 74)
    print(f"days={tot['ndays']} window-fails={tot['fail']}")
    print(f"  2-pass (S-NPP)     {tot['cd2']:>8,}")
    print(f"  4-pass (+N20)      {tot['cd4']:>8,}  (+{pct(tot['cd4']-tot['cd2'],tot['cd2']):.1f}% vs 2-pass)")
    print(f"  6-pass (+N21)      {tot['cd6']:>8,}  (+{pct(tot['cd6']-tot['cd4'],tot['cd4']):.1f}% vs 4-pass)")
    print(f"  >>> N21 marginal over 4-pass: +{tot['cd6']-tot['cd4']:,} cell-days ({pct(tot['cd6']-tot['cd4'],tot['cd4']):.1f}%), "
          f"rescued {tot['resc']} empty days")
    print("  per-year N21 marginal over 4-pass:")
    for yr in sorted(acc):
        a = acc[yr]
        print(f"    {yr}: days={a['ndays']:3d} cd4={a['cd4']:>7,} cd6={a['cd6']:>7,} "
              f"N21+={a['cd6']-a['cd4']:>6,} (+{pct(a['cd6']-a['cd4'],a['cd4']):.1f}%)")
    print("=" * 74)


if __name__ == "__main__":
    main()
