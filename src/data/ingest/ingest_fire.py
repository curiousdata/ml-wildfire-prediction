"""FGDC fire ingester — VIIRS 375 m active fire (NASA FIRMS) → 1 km daily is_fire, backfill + append.

This is the FGDC's fire SOURCE for BOTH the label and the fire-context features (dist_to_fire,
time_since_last_fire, burn_frequency are derived from this same is_fire history at coarsen/engineering
time). Using one source for label+features is the train=serve identity that fixes v1's 0.10-corr bug;
VIIRS is also a markedly more learnable next-day target than MODIS active-fire (Karlsson et al. 2025,
arXiv:2503.08580). A cell is is_fire=1 if ≥1 VIIRS detection at confidence ≥ nominal falls in it that day.

Sources (FIRMS area API):
  * backfill  → VIIRS_SNPP_SP  (standard-processing archive, 2012-01-20 → ~2 months ago)
  * append    → VIIRS_SNPP_NRT (near-real-time, last ~2 months)
  (NOAA-20/21 VIIRS could be pooled in for 2018+ density — a later enhancement; SNPP covers all of 2012→.)

CLI:
  --validate DATE        fetch + rasterize one day, report detections/cells (sanity)
  --backfill START END   write per-day is_fire npz for the range (10-day windows; resumable)
  --append [DATE]        write a single recent day via NRT (default: yesterday)
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import datetime as _dt
import numpy as np

from src.data import fetch as FB
from src.data.ingest import grid

BRONZE = grid.ROOT / "data" / "bronze" / "fireguard" / "fire"
BBOX_LL = (-10.0, 35.0, 5.0, 44.5)            # W,S,E,N — Spain + margin
SRC_ARCHIVE, SRC_NRT = "VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"
SRC_NRT2 = "VIIRS_NOAA20_NRT"                  # 2nd VIIRS bird (~50 min offset)
SRC_NRT3 = "VIIRS_NOAA21_NRT"                  # 3rd VIIRS bird (leads the group) → serve unions all 3 = 6 daily passes.
                                              # Proven +25.6% cell-days over the 2-bird baseline, stable ~26%/yr 2024-26
                                              # (n21_density.py); NOAA-21 is NRT-only (no SP archive) but NRT keeps full history.
CONF_KEEP = {"n", "h"}                         # VIIRS confidence: keep nominal + high (drop low)


def _filter_conf(df):
    """Keep detections at confidence ≥ nominal. VIIRS uses 'l'/'n'/'h'; be robust to missing/odd values."""
    if "confidence" not in df.columns or df.empty:
        return df
    c = df["confidence"].astype(str).str.strip().str.lower()
    return df[c.isin(CONF_KEEP)]


def rasterize_day(df_day):
    """Rasterize one day's detections onto the 1 km FGDC grid → binary is_fire[NY,NX]."""
    gx, gy = grid.x_coords(), grid.y_coords()
    if df_day is None or df_day.empty:
        return np.zeros((grid.NY, grid.NX), np.float32)
    return FB.fires_to_grid(df_day["longitude"].values, df_day["latitude"].values, gx, gy)


def write_day(date, src=SRC_NRT, df_day=None):
    BRONZE.mkdir(parents=True, exist_ok=True)
    if df_day is None:
        key = os.getenv("FIRMS_MAP_KEY")
        df = _filter_conf(FB.fetch_firms(key, date, src=src, bbox=BBOX_LL, days=1))
        df_day = df[df["acq_date"] == date] if "acq_date" in df.columns else df
    fire = rasterize_day(df_day)
    grid.atomic_savez(BRONZE / f"{date}.npz", is_fire=fire, n_det=np.int32(0 if df_day is None else len(df_day)))
    return fire, (0 if df_day is None else len(df_day))


def backfill(start, end, src=SRC_ARCHIVE, window=5):
    """Fetch the range in ≤window-day requests (FIRMS area API caps day_range at 5); split by acq_date;
    rasterize + write each day. Resumable."""
    import logging
    log = logging.getLogger("ingest_fire")
    key = os.getenv("FIRMS_MAP_KEY")
    if not key:
        raise SystemExit("set FIRMS_MAP_KEY")
    d = _dt.date.fromisoformat(start); last = _dt.date.fromisoformat(end)
    while d <= last:
        win_days = min(window, (last - d).days + 1)
        wdates = [(d + _dt.timedelta(days=k)).isoformat() for k in range(win_days)]
        if all((BRONZE / f"{wd}.npz").exists() for wd in wdates):
            log.info(f"{d}..+{win_days}d: all exist, skip"); d += _dt.timedelta(days=win_days); continue
        df = _filter_conf(FB.fetch_firms(key, d.isoformat(), src=src, bbox=BBOX_LL, days=win_days))
        for wd in wdates:
            sub = df[df["acq_date"] == wd] if "acq_date" in df.columns else df.iloc[0:0]
            fire, n = write_day(wd, df_day=sub)
            log.info(f"  {wd}: {n} detections → {int(fire.sum())} cells")
        d += _dt.timedelta(days=win_days)


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("ingest_fire")
    a = sys.argv
    if "--validate" in a:
        date = a[a.index("--validate") + 1]
        key = os.getenv("FIRMS_MAP_KEY")
        if not key:
            log.error("FIRMS_MAP_KEY not set"); return
        src = SRC_ARCHIVE if date < (_dt.date.today() - _dt.timedelta(days=60)).isoformat() else SRC_NRT
        df = FB.fetch_firms(key, date, src=src, bbox=BBOX_LL, days=1)
        dff = _filter_conf(df[df["acq_date"] == date] if "acq_date" in df.columns else df)
        fire = rasterize_day(dff)
        log.info(f"VIIRS {src} {date}: {len(df)} raw → {len(dff)} conf≥nominal detections → "
                 f"{int(fire.sum())} of {grid.NY*grid.NX} cells lit (1 km grid)")
        return
    if "--backfill" in a:
        i = a.index("--backfill"); backfill(a[i + 1], a[i + 2]); return
    if "--append" in a:
        i = a.index("--append")
        date = a[i + 1] if len(a) > i + 1 and not a[i + 1].startswith("-") else \
            (_dt.date.today() - _dt.timedelta(days=1)).isoformat()
        fire, n = write_day(date, src=SRC_NRT); log.info(f"wrote fire {date}: {n} det → {int(fire.sum())} cells"); return
    print("Use --validate DATE | --backfill START END | --append [DATE]", file=sys.stderr)


if __name__ == "__main__":
    main()
