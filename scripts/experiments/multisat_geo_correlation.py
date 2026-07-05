"""Geostationary (MTG-FCI) vs VIIRS-6pass complementarity — does the continuous-cadence geo sensor add
fire-positive cell-days the three polar VIIRS birds miss? Compares the daily 4 km geo masks (from
multisat_geo_frp.py) against the 6-pass VIIRS union over the fetched window.

Metrics: overlap / Jaccard, geo's recall of VIIRS (sensitivity), geo-UNIQUE cell-days (complementary signal),
the between-pass GAP-unique subset (the geostationary payoff — fire VIIRS structurally can't see that day),
density gain (fire6 ∪ geo vs fire6), and per-day cell-count correlation.

  python scripts/experiments/multisat_geo_correlation.py --window 2025-08-01 2025-08-31
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import xarray as xr

from src.data import metrics as T
from src.data.ingest import grid

CACHE = T.project_root / "data" / "cache" / "multisat"
REPORT = T.project_root / "reports" / "multisat_geo_correlation.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", nargs=2, metavar=("START", "END"), required=True)
    a = ap.parse_args()
    z = xr.open_zarr(str(grid.ROOT / "data" / "gold" / "FireGuard_coarse4.zarr"), consolidated=True)
    tidx = pd.DatetimeIndex(z["time"].values); N = len(tidx)
    land = np.nan_to_num(z["is_spain"].values) > 0.5
    w0, w1 = pd.Timestamp(a.window[0]).date(), pd.Timestamp(a.window[1]).date()
    days = np.array([i for i, d in enumerate(tidx) if w0 <= d.date() <= w1])
    print(f"window {w0}..{w1} → {len(days)} cube-days, {int(land.sum())} land cells")

    geo_any = np.load(CACHE / "geo_any.npy")[days][:, land]
    geo_gap = np.load(CACHE / "geo_gap.npy")[days][:, land]
    fire2 = (z["is_fire"].values[days][:, land] > 0.5)
    n20 = np.load(CACHE / "fire_n20.npy")[:N][days][:, land]
    n21 = np.load(CACHE / "fire_n21.npy")[:N][days][:, land]
    fire6 = fire2 | n20 | n21                                    # 3-bird VIIRS union

    V, G = int(fire6.sum()), int(geo_any.sum())
    both = int((geo_any & fire6).sum())
    geo_only = int((geo_any & ~fire6).sum())
    viirs_only = int((fire6 & ~geo_any).sum())
    union = int((geo_any | fire6).sum())
    gap_unique = int((geo_gap & ~fire6).sum())                   # between-pass geo detections VIIRS missed
    pct = lambda n, d: 100.0 * n / d if d else 0.0
    # per-day cell-count correlation
    gd, vd = geo_any.sum(1).astype(float), fire6.sum(1).astype(float)
    r = float(np.corrcoef(gd, vd)[0, 1]) if len(days) > 2 and gd.std() > 0 and vd.std() > 0 else float("nan")

    print("\n" + "=" * 74)
    print("MTG geostationary  vs  VIIRS 6-pass — daily 4 km cell-days over land")
    print("=" * 74)
    print(f"  VIIRS 6-pass cell-days           {V:>7,}")
    print(f"  MTG geo cell-days                {G:>7,}")
    print(f"  overlap (both)                   {both:>7,}   (geo recall of VIIRS {pct(both,V):.0f}%,  "
          f"VIIRS recall of geo {pct(both,G):.0f}%)")
    print(f"  Jaccard(geo,VIIRS)               {pct(both,union)/100:>7.3f}")
    print(f"  VIIRS-unique (geo misses)        {viirs_only:>7,}   ({pct(viirs_only,V):.0f}% of VIIRS — small-fire sensitivity gap)")
    print(f"  >>> geo-UNIQUE (complementary)   {geo_only:>7,}   (+{pct(geo_only,V):.1f}% density over VIIRS-6pass)")
    print(f"      of which BETWEEN-pass (gap)  {gap_unique:>7,}   (fire VIIRS structurally can't see that day)")
    print(f"  density: fire6 {V:,} → fire6∪geo {union:,}  (+{pct(union-V,V):.1f}%)")
    print(f"  per-day cell-count correlation   r={r:.3f}")
    print("=" * 74)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "experiment": "multisat_geo_correlation", "source": "MTG-FCI FDeM (EO:EUM:DAT:0682), hourly sample",
        "window": f"{w0}..{w1}", "n_days": int(len(days)),
        "viirs6_cell_days": V, "geo_cell_days": G, "overlap": both,
        "geo_recall_of_viirs_pct": pct(both, V), "viirs_recall_of_geo_pct": pct(both, G),
        "jaccard": pct(both, union) / 100, "viirs_unique": viirs_only,
        "geo_unique": geo_only, "geo_unique_pct_over_viirs": pct(geo_only, V),
        "geo_gap_unique": gap_unique, "density_gain_pct": pct(union - V, V),
        "per_day_count_corr": r}, indent=2))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
