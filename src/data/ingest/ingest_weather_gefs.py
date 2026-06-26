"""FGDC forecast-weather ingester — GEFSv12 reforecast d+1 (next-day) forecast → daily features on the Spain
grid. This is the *forecast* counterpart of ingest_weather (which stores OBSERVED ERA5 weather): for each
init date t (00 UTC run), we take the forecast VALID FOR t+1 (lead 24-45 h) — i.e. "tomorrow's weather as
known today" — to test/serve the forecast-weather complement (see ABLATIONS / CHANGES).

Source decision: GEFSv12 reforecast (Hamill et al. 2022), AWS `noaa-gefs-retrospective` (anonymous HTTPS,
GRIB2), 2000-2019, 0.25°, 3-hourly to +10 d, control member c00. Chosen over Open-Meteo previous-runs because
the full weather set only reaches back to 2024 there (untrainable on the 2012-2023 split), whereas GEFS covers
the whole history; train→serve seam-free if served from operational GEFS too. CAPE rides along for free
(Tier-2 dry-lightning signal). Soil is DROPPED from the forecast block: it's slow-varying (tomorrow ≈ today,
already in the cube as observed t-soil), its fast change is rain-driven (captured by the precip forecast we
keep), and GEFS's 0.1-0.4 m layer mismatches the cube's 7-28 cm — lowest value, highest friction.

Efficiency (fits a tight disk budget): we never keep GRIBs. Per var-file we (1) read the tiny `.idx`, (2)
byte-range GET only the 8 d+1 messages (~10 % of the file; the rest is leads we don't use), (3) aggregate to
daily native points, (4) discard the bytes. Persistent bronze ≈ 0.6 GB for the whole history; peak disk < 1 GB.

CLI: --validate DATE | --backfill START END
"""
from __future__ import annotations
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import datetime as _dt
import time

import numpy as np
import requests
import xarray as xr

from src.data.ingest import grid
from src.data.ingest import ingest_weather as IW          # reuse atomic_savez + WLON/WLAT native convention
import src.data.feature_engineering as FE                 # saturation_vapour_pressure_kpa for RH

BRONZE = grid.ROOT / "data" / "bronze" / "fireguard" / "weather_fc1"
REFO = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
MEMBER = "c00"
D1_HOURS = (24, 27, 30, 33, 36, 39, 42, 45)              # forecast valid-hours covering the NEXT calendar day
SPAIN = (-9.5, 4.5, 35.5, 44.0)                          # W, E, S, N  (GEFS lon is 0..360 → wrapped on read)

GEFS_VARS = ("tmp_2m", "tmax_2m", "tmin_2m", "spfh_2m", "pres_sfc",
             "ugrd_hgt", "vgrd_hgt", "apcp_sfc", "cape_sfc")
WIND_VARS = ("ugrd_hgt", "vgrd_hgt")                     # share their file with 100 m wind → filter to 10 m on read


def _stem(init: str, var: str) -> str:
    ymd = init.replace("-", "") + "00"                    # 2017-07-01 → 2017070100 (00 UTC run)
    return f"{REFO}/{init[:4]}/{ymd}/{MEMBER}/Days:1-10/{var}_{ymd}_{MEMBER}.grib2"


def _retry(fn, tries=5):
    last = None
    for i in range(tries):
        try:
            return fn()
        except (requests.exceptions.RequestException,) as e:
            last = e; time.sleep(min(2 ** i * 3, 30))
    raise last


def _d1_spans(stem: str):
    """Read the .idx and return the d+1 messages coalesced into contiguous byte runs (usually one — messages
    are step-ordered) so the slice downloads in 1-2 range-GETs, not 8. `valid hour` = the LAST number in the
    GRIB field (handles `24 hour fcst` and accumulation windows `21-24 hour acc fcst`). d+1 is mid-file (runs
    to +240 h) so the next message always bounds the last."""
    txt = _retry(lambda: requests.get(stem + ".idx", timeout=60).text)
    lines = [l for l in txt.splitlines() if l.strip()]
    starts = [int(l.split(":")[1]) for l in lines]
    sel = []
    for i, l in enumerate(lines):
        if i + 1 >= len(lines):
            continue
        nums = re.findall(r"\d+", l.split(":")[5])
        if (int(nums[-1]) if nums else -1) in D1_HOURS:
            sel.append((starts[i], starts[i + 1] - 1))
    sel.sort()
    runs = []
    for s, e in sel:
        if runs and s == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], e)
        else:
            runs.append((s, e))
    return runs


def _fetch_field(init: str, var: str):
    """Byte-range fetch the d+1 block (1-2 coalesced runs) for one var → (lon, lat, vals[steps, lat, lon]).
    Wind files also hold 100 m wind, so filter to the 10 m level on read."""
    stem = _stem(init, var)
    runs = _d1_spans(stem)
    if not runs:
        raise ValueError(f"no d+1 messages for {var} {init}")
    buf = b""
    for s, e in runs:
        buf += _retry(lambda s=s, e=e: requests.get(stem, headers={"Range": f"bytes={s}-{e}"}, timeout=120).content)
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(buf); tmp = f.name
    try:
        bk = {"indexpath": ""}
        if var in WIND_VARS:
            bk["filter_by_keys"] = {"typeOfLevel": "heightAboveGround", "level": 10}
        ds = xr.open_dataset(tmp, engine="cfgrib", backend_kwargs=bk)
        v = ds[list(ds.data_vars)[0]]
        vals = v.values
        if vals.ndim == 2:                                # single message → add a step axis
            vals = vals[None]
        return ds.longitude.values, ds.latitude.values, vals
    finally:
        os.remove(tmp)


def _spain(lon, lat):
    LON, LAT = np.meshgrid(lon, lat)
    lon180 = np.where(LON > 180, LON - 360, LON)
    m = (lon180 >= SPAIN[0]) & (lon180 <= SPAIN[1]) & (LAT >= SPAIN[2]) & (LAT <= SPAIN[3])
    return m.ravel(), lon180[m], LAT[m]


def native_day(init: str):
    """Fetch all d+1 GEFS vars, aggregate to the cube's daily weather features over Spain native points.
    Returns {feature: vec[n_pts], __lon__, __lat__}. Feature names match the OBSERVED weather block (the
    forecast suffix `_fc1` is applied later, at gold materialization)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(GEFS_VARS)) as ex:    # I/O-bound S3 fetches → all vars at once
        raw = dict(zip(GEFS_VARS, ex.map(lambda v: _fetch_field(init, v), GEFS_VARS)))
    glon, glat, _ = raw["tmp_2m"]
    mask, plon, plat = _spain(glon, glat)
    F = {var: vals.reshape(vals.shape[0], -1)[:, mask] for var, (_, _, vals) in raw.items()}

    out = {}
    T = F["tmp_2m"] - 273.15                              # K → °C
    out["t2m_mean"] = np.nanmean(T, 0)
    out["t2m_max"] = np.nanmax(F["tmax_2m"] - 273.15, 0)  # native daily-max field (no 3-h under-sampling)
    out["t2m_min"] = np.nanmin(F["tmin_2m"] - 273.15, 0)
    out["t2m_range"] = out["t2m_max"] - out["t2m_min"]
    # RH from specific humidity: e = q·p/(0.622+0.378q) [Pa]; es(T) [Pa]; RH = 100·e/es
    q, p = F["spfh_2m"], F["pres_sfc"]
    e = q * p / (0.622 + 0.378 * q)
    es = FE.saturation_vapour_pressure_kpa(T) * 1000.0    # kPa → Pa
    RH = np.clip(100.0 * e / es, 0.0, 100.0)
    out["RH_mean"] = np.nanmean(RH, 0); out["RH_min"] = np.nanmin(RH, 0)
    out["RH_max"] = np.nanmax(RH, 0); out["RH_range"] = out["RH_max"] - out["RH_min"]
    P = F["pres_sfc"] / 100.0                             # Pa → hPa (cube/Open-Meteo unit)
    out["surface_pressure_mean"] = np.nanmean(P, 0); out["surface_pressure_min"] = np.nanmin(P, 0)
    out["surface_pressure_max"] = np.nanmax(P, 0); out["surface_pressure_range"] = out["surface_pressure_max"] - out["surface_pressure_min"]
    u, v = F["ugrd_hgt"], F["vgrd_hgt"]; spd = np.hypot(u, v)
    out["wind_speed_mean"] = np.nanmean(spd, 0); out["wind_speed_max"] = np.nanmax(spd, 0)
    out["wind_u_mean"] = np.nanmean(u, 0); out["wind_v_mean"] = np.nanmean(v, 0)
    jmax = np.nanargmax(np.where(np.isfinite(spd), spd, -np.inf), 0); cols = np.arange(spd.shape[1])
    out["wind_u_atmaxspeed"] = u[jmax, cols]; out["wind_v_atmaxspeed"] = v[jmax, cols]
    out["total_precipitation_mean"] = np.nansum(F["apcp_sfc"], 0) / 24.0   # daily total (mm) → hourly-mean, as the cube stores it
    out["cape_mean"] = np.nanmean(F["cape_sfc"], 0); out["cape_max"] = np.nanmax(F["cape_sfc"], 0)  # NEW (Tier-2)

    out = {k: np.asarray(val, np.float32) for k, val in out.items()}
    out[IW.WLON] = np.asarray(plon, np.float64); out[IW.WLAT] = np.asarray(plat, np.float64)
    return out


def write_day(init: str):
    BRONZE.mkdir(parents=True, exist_ok=True)
    IW.atomic_savez(BRONZE / f"{init}.npz", **native_day(init))


def backfill_range(start: str, end: str, workers: int = 4):
    """Backfill (skips existing). Each day streams ~10 % of 9 GRIB files via idx byte-ranges, aggregates,
    writes a small npz, keeps nothing — peak disk < 1 GB. Days run `workers`-wide (each already fans out 9
    fetches), so wall time is a few× the per-day cost, not the sum."""
    import logging
    from concurrent.futures import ThreadPoolExecutor, as_completed
    log = logging.getLogger("ingest_weather_gefs")
    BRONZE.mkdir(parents=True, exist_ok=True)
    d, last = _dt.date.fromisoformat(start), _dt.date.fromisoformat(end)
    todo = []
    while d <= last:
        iso = d.isoformat()
        if not (BRONZE / f"{iso}.npz").exists():
            todo.append(iso)
        d += _dt.timedelta(days=1)
    log.info(f"{len(todo)} days to fetch ({workers} concurrent)")

    def one(iso):
        t = time.time()
        try:
            write_day(iso); return iso, time.time() - t, None
        except Exception as exc:
            return iso, 0.0, exc

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(one, iso) for iso in todo]):
            iso, dt, exc = fut.result(); done += 1
            if exc is not None:
                log.warning(f"{iso}: SKIP {type(exc).__name__} {exc}")
            elif done % 25 == 0 or done == len(todo):
                log.info(f"[{done}/{len(todo)}] {iso} ({dt:.0f}s)")


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("ingest_weather_gefs")
    a = sys.argv
    if "--validate" in a:
        date = a[a.index("--validate") + 1]
        out = native_day(date)
        log.info(f"GEFS d+1 native_day({date}): {len([k for k in out if not k.startswith('__')])} features, "
                 f"{out[IW.WLON].size} Spain points")
        for k in sorted(out):
            if k.startswith("__"):
                continue
            v = out[k]
            log.info(f"  {k:24} min={np.nanmin(v):8.2f} max={np.nanmax(v):8.2f} mean={np.nanmean(v):8.2f}")
        return
    if "--backfill" in a:
        i = a.index("--backfill"); backfill_range(a[i + 1], a[i + 2]); return
    print("Use --validate DATE | --backfill START END", file=sys.stderr)


if __name__ == "__main__":
    main()
