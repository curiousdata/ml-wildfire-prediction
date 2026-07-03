"""FGDC weather backfill from **raw ERA5** — the free, bulk source for the multi-year historical backfill
(Open-Meteo's *archive* endpoint is paywalled). The live daily append still uses Open-Meteo *forecast*
(ingest_weather); this module is ONLY for the one-time historical backfill. Three interchangeable sources:

  edh  (PREFERRED) — Earth Data Hub (DestinE) ERA5 Zarr mirror, chunked *for time analysis* (small spatial
        chunks), so a Spain slice downloads only Spain → ~GBs total. Needs a free DestinE API key (see SETUP).
  arco — public ARCO-ERA5 Zarr (anonymous, no account), BUT chunked (time=1, full-globe): a regional pull
        downloads whole global fields (~24 GB/month → ~TBs for 13 yr). Use only as a no-auth fallback.
  cds  — Copernicus CDS NetCDF via cdsapi (queued; needs ~/.cdsapirc). Legacy fallback.

Design — produces bronze npz **identical to the Open-Meteo path** so `build_silver` consumes it unchanged:
ERA5 single-levels hourly → convert each field to the Open-Meteo-named hourly point arrays → reuse
`ingest_weather.daily_point_features` (the SAME daily aggregation) → cached `fetch_openmeteo.make_regridder`
onto the 1 km grid → `data/bronze/fireguard/weather/<date>.npz`. The regridder + the store handle are each
built ONCE per run and reused for every day × feature (the two backfill-speed optimizations).

ERA5 → our mapping (units converted to match the Open-Meteo feed daily_point_features expects):
  t2m (K)→temperature_2m(°C) · d2m (K)+t2m→relative_humidity_2m(%, Magnus) · sp (Pa)→surface_pressure(hPa) ·
  u10,v10 (m/s)→wind_speed_10m(km/h)+wind_direction_10m(deg, via FE.uv_to_direction_deg so it round-trips) ·
  swvl2 (m³/m³)→soil_moisture_7_to_28cm · stl2 (K)→soil_temperature_7_to_28cm(°C) · tp (m, hourly accum)→
  daily precip sum (mm). Store var names (short vs long) are auto-resolved + renamed to short per source.

SETUP for 'edh' (one-time, user — I cannot do this part):
  1. Register at https://earthdatahub.destine.eu (free DestinE Platform account).
  2. Account settings → copy your default API key.
  3. Add it to this repo's .env as:  EDH_TOKEN=<your-key>   (gitignored; never commit it).
SETUP for 'cds':  free account at https://cds.climate.copernicus.eu → accept the ERA5 licence → write
  ~/.cdsapirc (url: https://cds.climate.copernicus.eu/api  /  key: <token>).

CLI: --probe [DATE] [--source edh|arco|cds]   pull 1 day, PRINT the structure, run the full per-day parse
                                              (validates var names/units/regrid + EDH chunk efficiency)
     --backfill START END [--source ...]      monthly-chunked, resumable backfill into the weather bronze
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
try:
    from dotenv import load_dotenv; load_dotenv()   # EDH_TOKEN lives in .env (gitignored)
except Exception:
    pass
import logging
import dask
import numpy as np
import pandas as pd
import xarray as xr

# ARCO chunks are (time=1, full-globe), so a regional pull still downloads whole global fields → it's
# bandwidth-bound. Benchmarking showed ~16 concurrent chunk fetches saturates throughput (48 ≈ no better).
ARCO_WORKERS = 16

from src.data.ingest import grid
from src.data.ingest import ingest_weather as IW
from src.data import fetch as OM
import src.data.feature_engineering as FE

log = logging.getLogger("ingest_weather_cds")

DATASET = "reanalysis-era5-single-levels"
# ERA5 CDS variable names → the NetCDF short names they arrive as.
CDS_VARS = ["2m_temperature", "2m_dewpoint_temperature", "surface_pressure",
            "10m_u_component_of_wind", "10m_v_component_of_wind", "total_precipitation",
            "volumetric_soil_water_layer_2", "soil_temperature_level_2"]
SHORT = {"t2m": "2m_temperature", "d2m": "2m_dewpoint_temperature", "sp": "surface_pressure",
         "u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind", "tp": "total_precipitation",
         "swvl2": "volumetric_soil_water_layer_2", "stl2": "soil_temperature_level_2"}
AREA = [44.6, -10.0, 35.2, 5.0]   # N, W, S, E — covers mainland Spain + Balearics with margin
HOURS = [f"{h:02d}:00" for h in range(24)]
RAW = grid.ROOT / "data" / "cache" / "era5"   # keep downloaded NetCDF (gitignored) → re-parse without re-download
# Route 'arco': the public ARCO-ERA5 Zarr (analysis-ready, cloud-optimized) — anonymous, no account. BUT it is
# chunked (time=1, full-globe), so a regional pull downloads whole global fields (~24 GB/month) → ~TBs total.
ARCO_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
# Route 'edh' (PREFERRED for the multi-year backfill): Earth Data Hub (DestinE) ERA5 Zarr mirror, chunked
# *for time analysis* (small spatial chunks) → a Spain slice downloads only Spain (~GBs total, not TBs). Needs
# a free DestinE account → personal API key in .env as EDH_TOKEN (or a ~/.netrc entry). Zarr v3 (zarr>=3).
EDH_STORE = "https://api.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr"
EDH_TOKEN = os.getenv("EDH_TOKEN")

_STORE_CACHE = {}   # opening a global ERA5 Zarr reads lots of metadata — do it ONCE per (source) per process


def _resolve_vars(ds):
    """Map our 8 needed fields (keyed by ERA5 short name) to whatever names this store actually uses
    (short like 't2m' or long like '2m_temperature'), so we can subset+rename to short names before load."""
    out = {}
    for short, long in SHORT.items():
        if short in ds.data_vars:
            out[short] = short
        elif long in ds.data_vars:
            out[short] = long
        else:
            raise KeyError(f"store has neither '{short}' nor '{long}'; vars: {list(ds.data_vars)[:20]}…")
    return out


def _open_store(source):
    """Open a global ERA5 Zarr ONCE (per source) and cache the handle + Spain longitude selection
    (0–360 → −180..180; Spain straddles the 0-meridian) + the var name map. Reused for every monthly slice
    so the expensive store-open + metadata read isn't repaid 150+ times across a multi-year backfill."""
    if source not in _STORE_CACHE:
        n, w, s, e = AREA
        if source == "arco":
            ds = xr.open_zarr(ARCO_STORE, chunks={}, storage_options={"token": "anon"})
        elif source == "edh":
            if not EDH_TOKEN:
                raise RuntimeError("EDH route needs a DestinE API key in .env as EDH_TOKEN "
                                   "(register at https://earthdatahub.destine.eu → account settings).")
            url = EDH_STORE.replace("https://", f"https://edh:{EDH_TOKEN}@")
            ds = xr.open_dataset(url, engine="zarr", chunks={})
        else:
            raise ValueError(f"unknown zarr source '{source}'")
        lonname = "longitude" if "longitude" in ds.coords else "lon"
        latname = "latitude" if "latitude" in ds.coords else "lat"
        lon = ds[lonname]
        lon180 = ((lon + 180) % 360) - 180 if float(lon.max()) > 180 else lon  # ECMWF native is 0–360
        keep = np.where(((lon180 >= w) & (lon180 <= e)).values)[0]
        latv = ds[latname].values
        latkeep = np.where((latv >= s) & (latv <= n))[0]          # order-agnostic (ERA5 lat is descending)
        _STORE_CACHE[source] = dict(ds=ds, keep=keep, lon180=lon180.isel({lonname: keep}), latkeep=latkeep,
                                    lonname=lonname, latname=latname, vmap=_resolve_vars(ds))
    c = _STORE_CACHE[source]
    return c


def _open_arco(start, end, source="arco"):
    """Slice the (cached) zarr store → the Spain slice for [start,end], shaped like the CDS NetCDF
    (time/latitude/longitude + ERA5 var names `_process_day` reads via `_var`); columns re-expressed as
    −180..180 and sorted ascending, matching what regrid_to_cube expects. Works for 'arco' and 'edh'."""
    c = _open_store(source)
    lonname, latname, vmap = c["lonname"], c["latname"], c["vmap"]
    tname = "valid_time" if "valid_time" in c["ds"].coords or "valid_time" in c["ds"].dims else "time"
    sub = (c["ds"][list(vmap.values())]
           .sel({tname: slice(start, end)})
           .isel({latname: c["latkeep"], lonname: c["keep"]}))
    sub = sub.assign_coords({lonname: c["lon180"]}).sortby(lonname)
    rename = {v: k for k, v in vmap.items() if v != k}           # long → short, so `_var` finds them
    rename.update({lonname: "longitude", latname: "latitude"})
    sub = sub.rename({k: v for k, v in rename.items() if k in sub.variables or k in sub.coords})
    return sub, tname


def _es(tc):
    """Saturation vapour pressure (hPa) from temperature in °C (Magnus)."""
    return 6.112 * np.exp(17.67 * tc / (tc + 243.5))


def _open(nc):
    """Open an ERA5 NetCDF, normalizing the time coord name and any expver dim (recent ERA5T months)."""
    ds = xr.open_dataset(nc)
    tname = "valid_time" if "valid_time" in ds.coords or "valid_time" in ds.dims else "time"
    if "expver" in ds.dims:                    # 1=ERA5, 5=ERA5T overlap → collapse (one is non-nan per time)
        ds = ds.ffill("expver").isel(expver=-1)
    return ds, tname


def _var(ds, short):
    """Fetch a field by ERA5 short name, tolerating the long CDS name if that's what the file uses."""
    if short in ds:
        return ds[short]
    long = SHORT[short]
    if long in ds:
        return ds[long]
    raise KeyError(f"ERA5 file has neither '{short}' nor '{long}'; vars present: {list(ds.data_vars)}")


def _native_day(ds, tname, day):
    """One day's 24 hourly ERA5 fields → native per-point bronze dict (reusing daily_point_features).
    Returns the NATIVE per-point dict {feature: vec[n_pts], __lon__, __lat__} — NO regrid (build_silver
    regrids to 1 km on read). Storing native keeps bronze ~200 KB/day vs ~71 MB/day upsampled to 1 km."""
    tdays = pd.to_datetime(ds[tname].values)
    sel = np.where(tdays.strftime("%Y-%m-%d") == day)[0]
    if sel.size != 24:
        raise ValueError(f"{day}: expected 24 hours, got {sel.size}")
    sub = ds.isel({tname: sel})
    lo = sub["longitude"].values.astype(float); la = sub["latitude"].values.astype(float)
    LO, LA = np.meshgrid(lo, la); plon, plat = LO.ravel(), LA.ravel()

    def hr(short):                               # [24, n_pts]
        return np.asarray(_var(sub, short).values, float).reshape(24, -1)

    t2m_c = hr("t2m") - 273.15
    d2m_c = hr("d2m") - 273.15
    rh = np.clip(100.0 * _es(d2m_c) / _es(t2m_c), 0.0, 100.0)
    u, v = hr("u10"), hr("v10")
    spd_kmh = np.sqrt(u ** 2 + v ** 2) * 3.6     # daily_point_features divides by 3.6 → m/s
    hv = {"temperature_2m": t2m_c, "relative_humidity_2m": rh, "surface_pressure": hr("sp") / 100.0,
          "wind_speed_10m": spd_kmh, "wind_direction_10m": FE.uv_to_direction_deg(u, v),
          "soil_moisture_7_to_28cm": hr("swvl2"), "soil_temperature_7_to_28cm": hr("stl2") - 273.15}
    daily_precip_mm = hr("tp").sum(0) * 1000.0   # m → mm, daily sum (matches Open-Meteo precipitation_sum)
    times = [t.strftime("%Y-%m-%dT%H:%M") for t in tdays[sel]]
    feats = IW.daily_point_features(hv, times, daily_precip_mm, day)
    out = {k: np.asarray(vec, np.float32) for k, vec in feats.items()}
    out[IW.WLON] = plon.astype(np.float64); out[IW.WLAT] = plat.astype(np.float64)
    return out


def _qc_native(out):
    """Light data-quality gate before an atomic write: every feature must carry some finite data (catches a
    bad parse). Crash/ENOSPC truncation is handled separately by the atomic temp→rename write."""
    bad = [k for k in out if not k.startswith("__") and not np.isfinite(out[k]).any()]
    if bad:
        raise ValueError(f"QC: all-non-finite features {bad}")
    return out


def _retrieve_month(year, month, target):
    """Download one month of hourly ERA5 over the Spain area as NetCDF (skips if already on disk)."""
    if target.exists():
        log.info(f"  {target.name} cached, skip download"); return
    import cdsapi
    target.parent.mkdir(parents=True, exist_ok=True)
    ndays = pd.Period(f"{year}-{month:02d}").days_in_month
    req = {"product_type": ["reanalysis"], "variable": CDS_VARS, "year": str(year),
           "month": [f"{month:02d}"], "day": [f"{d:02d}" for d in range(1, ndays + 1)],
           "time": HOURS, "area": AREA, "data_format": "netcdf", "download_format": "unarchived"}
    log.info(f"  CDS retrieve {year}-{month:02d} ({ndays}d, {len(CDS_VARS)} vars) → {target.name} (queued)…")
    cdsapi.Client().retrieve(DATASET, req, str(target))


def backfill(start, end, overwrite=False, source="arco"):
    """Monthly-chunked, resumable ERA5 backfill → weather bronze npz (Open-Meteo-compatible).
    source='edh': Earth Data Hub Zarr (token, time-chunked → light regional pull — PREFERRED for multi-year).
    'arco': public ARCO Zarr (no auth, but whole-globe chunks → heavy). 'cds': CDS NetCDF (queued)."""
    IW.BRONZE.mkdir(parents=True, exist_ok=True)
    # EDH time-chunks are 4320 h (~6 months): read a whole chunk-span at once so each chunk downloads ONCE
    # (monthly reads would re-fetch the same 6-month chunk 6×). ARCO/CDS are time=1 / per-month → block=1.
    block_months = 6 if source == "edh" else 1
    periods = pd.period_range(start=start[:7], end=end[:7], freq="M")
    for i in range(0, len(periods), block_months):
        block = periods[i:i + block_months]
        days = []
        for per in block:
            days += [d.strftime("%Y-%m-%d")
                     for d in pd.period_range(f"{per.year}-{per.month:02d}", periods=per.days_in_month, freq="D")]
        days = [d for d in days if start <= d <= end]
        todo = [d for d in days if overwrite or not (IW.BRONZE / f"{d}.npz").exists()]
        tag = f"{block[0]}" if len(block) == 1 else f"{block[0]}..{block[-1]}"
        if not todo:
            log.info(f"{tag}: all {len(days)} days present, skip"); continue
        if source in ("arco", "edh"):
            ds, tname = _open_arco(days[0], days[-1], source)
            with dask.config.set(scheduler="threads", num_workers=ARCO_WORKERS):
                ds = ds.load()                                          # parallel chunk fetch
        else:
            per = block[0]; nc = RAW / f"era5_{per.year}-{per.month:02d}.nc"
            _retrieve_month(per.year, per.month, nc); ds, tname = _open(nc)
        nok = 0
        for d in todo:
            try:
                IW.atomic_savez(IW.BRONZE / f"{d}.npz", **_qc_native(_native_day(ds, tname, d))); nok += 1
            except Exception as e:
                log.warning(f"  {d}: parse failed ({type(e).__name__}: {e})")
        ds.close()
        log.info(f"{tag}: wrote {nok}/{len(todo)} day(s) native [{source}]")


def probe(date="2015-08-01", source="arco"):
    """Pull ONE day, print the raw structure (so var-name/units assumptions are verified against reality),
    then run the full per-day parse + regrid and sanity-report ranges. Default source='arco' (no auth)."""
    if source in ("arco", "edh"):
        ds, tname = _open_arco(date, date, source)
        with dask.config.set(scheduler="threads", num_workers=ARCO_WORKERS):
            ds = ds.load()
    else:
        nc = RAW / f"probe_{date}.nc"
        if not nc.exists():
            import cdsapi
            nc.parent.mkdir(parents=True, exist_ok=True)
            y, m, d = date.split("-")
            log.info(f"probe: retrieving {date} (1 day, {len(CDS_VARS)} vars)…")
            cdsapi.Client().retrieve(DATASET, {"product_type": ["reanalysis"], "variable": CDS_VARS,
                "year": y, "month": [m], "day": [d], "time": HOURS, "area": AREA,
                "data_format": "netcdf", "download_format": "unarchived"}, str(nc))
        ds, tname = _open(nc)
    log.info(f"[{source}] dims={dict(ds.sizes)} time_coord='{tname}' data_vars={list(ds.data_vars)}")
    out = _qc_native(_native_day(ds, tname, date))                 # native per-point dict (what bronze stores)
    gx, gy = grid.x_coords(), grid.y_coords()
    regrid = OM.make_regridder(out[IW.WLON], out[IW.WLAT], gx, gy)  # the regrid build_silver does on read
    log.info(f"  native pts={out[IW.WLON].size} → cube {regrid.n_cells} cells (regrid-on-read at silver build)")
    grids = {k: regrid(v) for k, v in out.items() if not k.startswith("__")}
    for k in ("t2m_mean", "RH_mean", "RH_min", "surface_pressure_mean", "wind_speed_max",
              "wind_u_mean", "total_precipitation_mean", "soil_moisture_mean"):
        if k in grids:
            v = grids[k]; fin = np.isfinite(v)
            log.info(f"  {k:<24} finite%={fin.mean()*100:4.0f} range=[{np.nanmin(v):.3f},{np.nanmax(v):.3f}]")
    log.info(f"probe OK — {len(grids)} features; native bronze (~200 KB/day) + regrid-on-read validated")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    a = sys.argv
    src = a[a.index("--source") + 1] if "--source" in a else "arco"   # 'edh' | 'arco' (no auth) | 'cds'
    if "--probe" in a:
        i = a.index("--probe")
        probe(a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else "2015-08-01", source=src); return
    if "--backfill" in a:
        i = a.index("--backfill"); backfill(a[i + 1], a[i + 2], overwrite="--overwrite" in a, source=src); return
    print("Use --probe [DATE] | --backfill START END  [--source edh|arco|cds]", file=sys.stderr)


if __name__ == "__main__":
    main()
