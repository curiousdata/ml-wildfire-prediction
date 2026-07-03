"""Batch job — the FGDC v2 authoritative cadence (settled data only; touches silver). Cadence: WEEKLY.

⚠️ NAME vs LAMBDA ROLE: in the Lambda-architecture vocabulary this is actually the **SPEED tier** (weekly ERA5T
preliminary reanalysis, appends the cube up to the settle edge). The *true* monthly "batch" tier (final ERA5,
overwrite behind a `final_watermark` seam) is BACKLOGGED — final≈ERA5T, not worth building (see the
`lambda-architecture-fgdc` memory). The file/agent keep the "batch" name to avoid churning the live launchd
wiring (`run_batch.sh` → `com.fireguard.batch`); read "batch" here as "the weekly settled-data refresh".


The cube's data flows bronze → silver → gold → engineered, and **bronze is the source of truth**: every
raw day is cached as an npz that `build_silver` reads. So the batch job is:

  1. TOP UP BRONZE with the new SETTLED days (weather=Open-Meteo/ERA5, fire=FIRMS VIIRS, veg=MODIS/MPC),
     up to the watermark where the reanalysis/archive feeds are final.
  2. EXTEND the medallion — silver (1 km) → gold (4 km, coarsen_fgdc) → engineered (add_engineered_features +
     add_fire_context, whole-cube):
       * silver = **incremental APPEND** of just the new settled days (`build_silver --append`); ~10 s for a
         weekly window vs ~47 min for a full re-regrid of 14 yr. `--full` forces the one-time whole-rebuild
         baseline (e.g. first run, or after a schema change).
       * gold = full `coarsen_fgdc` for now (reads 150 GB silver→4 km; far cheaper than re-regridding, and
         proven). TODO: incremental `coarsen_fgdc.append_new` once measured + tested.
       * engineered MUST stay whole-cube — kbdi/precip_sum_*/time_since_last_fire are causal/recursive — but
         that's on the small 4 km gold. (The provisional ≤7-day edge to *today* is the DAILY job's concern —
         Option C; see the `fgdc-extend-cadence` memory.)

This is the ONLY thing that mutates silver, by design. It runs on the DATA MACHINE (silver is 150 GB; it can't
live on GH Actions/HF). Retraining/recalibration is deliberately NOT here — the model is stable; rerun
train_gbt_fgdc/calibrate explicitly when you want to retrain.

Per-feed settling lag (the watermark is the slowest feed that bounds "final"):
  * weather (ERA5-Land via Open-Meteo): archive lag ~5 d.
  * fire (FIRMS VIIRS): SP archive is final but lags ~2 months; NRT is good for the recent edge → we use
    ARCHIVE where available and NRT for the last ~FIRE_ARCHIVE_LAG_DAYS, so a recent edge isn't blocked 2 mo.
  * veg (MODIS 16-day composite): published with a few weeks' lag; daily_interp extrapolates the latest
    composite, so a missing newest composite degrades gracefully (NaN → GBT-native).

CLI:
  python scripts/batch_job.py [--to YYYY-MM-DD] [--watermark-days N] [--from YYYY-MM-DD] [--full]
                              [--skip-ingest] [--skip-silver] [--skip-gold] [--skip-engineered]
                              [--dry-run] [--force]
Run headless (MODIS veg fetch + the whole-cube rebuild are the slow parts).
"""
from __future__ import annotations
import argparse
import datetime as _dt
import logging
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv; load_dotenv()          # FIRMS_MAP_KEY / EDH_TOKEN live in .env (gitignored)
except Exception:
    pass

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
import xarray as xr

from src.data.ingest import grid
from src.data.ingest import ingest_weather as IW
from src.data.ingest import ingest_fire as IF
from src.data.ingest import ingest_veg as IV

GOLD = PROJECT / "data" / "gold" / "FireGuard_coarse4.zarr"
WATERMARK_DAYS = 6                 # ERA5T settle edge (~5 d) with 1 d safety margin. TODO (Delta 1): make this
                                   # data-driven — probe the archive (models=era5, which GAPS at the ERA5T edge)
                                   # for the last non-NaN day; fall back to this fixed value. Blocked on a
                                   # quota-available test of that gap behavior (the seamless archive fills to today).
FIRE_ARCHIVE_LAG_DAYS = 60         # FIRMS VIIRS SP archive horizon; newer than this → NRT
FACTOR = 4
log = logging.getLogger("batch_job")


def _run(stage: str, argv: list[str]):
    """Run a pipeline stage in its own process (memory isolation; each is a proven CLI). Fail loud."""
    log.info(f"[{stage}] $ {' '.join(argv)}")
    subprocess.run(argv, check=True, cwd=str(PROJECT))
    log.info(f"[{stage}] done")


def _ingest(new_start: str, end: str, today: _dt.date):
    """Top up bronze for (new_start..end): weather + fire + veg. Each ingester is resumable (skips existing)."""
    log.info(f"[ingest] topping up bronze {new_start}..{end}")
    # weather — Open-Meteo ERA5 archive (≤ watermark is final); range-efficient + skips existing.
    IW.backfill_range(new_start, end)

    # fire — VIIRS: ARCHIVE (SP, final) where settled, NRT for the recent edge so we don't wait ~2 months.
    archive_end = (today - _dt.timedelta(days=FIRE_ARCHIVE_LAG_DAYS)).isoformat()
    if new_start <= archive_end:
        IF.backfill(new_start, min(end, archive_end), src=IF.SRC_ARCHIVE)
    if end > archive_end:
        IF.backfill(max(new_start, (_dt.date.fromisoformat(archive_end) + _dt.timedelta(1)).isoformat()),
                    end, src=IF.SRC_NRT)

    # veg — MODIS/MPC 16-day composites interpolated to daily; latest composite may lag (graceful → NaN).
    IV.write_range(new_start, end)


def _verify(target_end: str, new_start: str):
    """Reopen the rebuilt gold cube and sanity-check the new edge: currency, fire seasonality, veg coverage."""
    z = xr.open_zarr(str(GOLD), consolidated=True)
    times = pd.DatetimeIndex(z["time"].values)
    last = str(times[-1].date())
    land = np.nan_to_num(z["is_spain"].values) > 0.5 if "is_spain" in z else None
    new_mask = times >= pd.Timestamp(new_start)
    log.info(f"[verify] gold last day {last} (target {target_end}); {int(new_mask.sum())} new days; "
             f"{z.sizes['time']} total")
    if int(new_mask.sum()):
        isf = z["is_fire"].isel(time=new_mask).values
        fire_days = int((isf.reshape(isf.shape[0], -1).sum(1) > 0).sum())
        log.info(f"[verify] fire-positive days in new window: {fire_days}/{int(new_mask.sum())}")
        for v in ("NDVI", "popdens"):                      # catch a silently-missing slow feed at the edge
            if v in z:
                sl = z[v].isel(time=new_mask).values
                cov = float(np.isfinite(sl[:, land] if (land is not None and "time" in z[v].dims) else sl).mean())
                log.info(f"[verify] {v} finite-fraction over new window: {cov:.3f}"
                         + ("   ⚠️ near-empty — check the feed" if cov < 0.5 else ""))
    if last != target_end:
        log.warning(f"[verify] last day {last} != target {target_end} — some edge days had no bronze (feed lag?)")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="FGDC weekly batch: top up bronze → extend silver→gold→engineered.")
    ap.add_argument("--to", help="target end date YYYY-MM-DD (default: today − watermark-days)")
    ap.add_argument("--watermark-days", type=int, default=WATERMARK_DAYS, help="settling margin (default 7)")
    ap.add_argument("--from", dest="from_", help="silver rebuild start (default: existing cube's first day)")
    ap.add_argument("--full", action="store_true",
                    help="one-time baseline: whole-rebuild silver from all bronze (else: incremental append)")
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--skip-silver", action="store_true")
    ap.add_argument("--skip-gold", action="store_true")
    ap.add_argument("--skip-engineered", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the window + planned stages, do nothing")
    ap.add_argument("--force", action="store_true", help="run even if the cube already reaches the target")
    args = ap.parse_args()

    today = _dt.date.today()
    target_end = args.to or (today - _dt.timedelta(days=args.watermark_days)).isoformat()

    z = xr.open_zarr(str(GOLD), consolidated=True)
    times = pd.DatetimeIndex(z["time"].values)
    first_day = args.from_ or str(times[0].date())
    last_day = str(times[-1].date())
    new_start = (times[-1] + pd.Timedelta(days=1)).date().isoformat()
    z.close()

    log.info(f"cube {first_day}..{last_day} ({len(times)}d) | target_end {target_end} "
             f"(today {today} − {args.watermark_days}d) | new window {new_start}..{target_end}")
    if target_end < new_start and not args.force:
        log.info("cube already reaches the target watermark — nothing to do (use --force to rebuild anyway).")
        return
    if args.dry_run:
        plan = [s for s, sk in (("ingest", args.skip_ingest), ("silver", args.skip_silver),
                                ("gold", args.skip_gold), ("engineered", args.skip_engineered)) if not sk]
        silver_plan = (f"FULL-rebuild silver {first_day}..{target_end}" if args.full
                       else f"APPEND silver {new_start}..{target_end} (incremental)")
        log.info(f"[dry-run] would ingest {new_start}..{target_end}, {silver_plan}, then gold+engineered. "
                 f"stages: {plan}")
        return

    py = sys.executable
    if not args.skip_ingest:
        _ingest(new_start, target_end, today)
    if not args.skip_silver:
        if args.full:                                      # one-time baseline: whole-rebuild from all bronze
            _run("silver-full", [py, "-m", "src.data.ingest.build_silver", "--start", first_day, "--end", target_end])
        else:                                              # weekly: regrid + APPEND only the new settled days (~10 s)
            _run("silver-append", [py, "-m", "src.data.ingest.build_silver", "--append", target_end])
    if args.full:                                          # one-time baseline: whole-cube coarsen + engineered (heavy)
        if not args.skip_gold:
            _run("gold-full", [py, "-m", "src.data.ingest.coarsen_fgdc", "--factor", str(FACTOR), "--overwrite"])
        if not args.skip_engineered:
            for script in ("add_fire_context.py", "add_engineered_features.py"):   # fire_context FIRST (makes precip_sum_*)
                _run("engineered", [py, str(PROJECT / "scripts" / script),
                                    "--cube", "FireGuard", "--factor", str(FACTOR), "--overwrite"])
    elif not (args.skip_gold and args.skip_engineered):    # weekly: coarsen ONLY new days + engineer via edge
        _run("gold-edge", [py, str(PROJECT / "scripts" / "update_edge.py"), "--run", "--to", target_end])

    _verify(target_end, new_start)
    log.info("batch job complete — cube current to the watermark with settled data (silver mutated).")


if __name__ == "__main__":
    main()
