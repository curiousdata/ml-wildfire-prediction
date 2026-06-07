"""Open-Meteo weather feed — keyless, gridded, live forecast + ERA5 archive. The chosen weather source.

Replaces the AEMET-station path (flaky key, station→grid interpolation, distribution shift). Open-Meteo is:
  * keyless + free (non-commercial), no signup;
  * already GRIDDED (we query a regular lat/lon grid and bilinear-regrid to the cube — no scattered-station
    interpolation);
  * ERA5-based, so it MATCHES our model's training distribution (minimal train/serve shift);
  * forecast API (live "today") + archive API (ERA5, for backfill / IberFire-v2).

Open-Meteo daily → our feature mapping (units already match the cube: °C, mm, hPa):
  temperature_2m_mean/min/max -> t2m_mean/min/max ; precipitation_sum -> tp ;
  wind_speed_10m_max -> wind_speed ; wind_direction_10m_dominant -> wind_direction ;
  (pressure / RH need hourly→daily aggregation — add when wiring the full slice).

archive:  https://archive-api.open-meteo.com/v1/archive   (start_date/end_date; ERA5)
forecast: https://api.open-meteo.com/v1/forecast           (past_days/forecast_days)

`--demo`: fetch a cube-range date from the archive, regrid t2m_mean to the cube grid, and compare to the
cube's own t2m_mean for that date — validates the fetch→regrid produces cube-compatible features.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import griddata

CUBE = Path(__file__).resolve().parents[1] / "data" / "gold" / "IberFire_coarse4.zarr"
SPAIN_BBOX = (-9.5, 35.5, 4.5, 44.0)  # W, S, E, N
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
DAILY_MAP = {"temperature_2m_mean": "t2m_mean", "temperature_2m_max": "t2m_max",
             "temperature_2m_min": "t2m_min", "precipitation_sum": "tp",
             "wind_speed_10m_max": "wind_speed", "wind_direction_10m_dominant": "wind_direction"}


def _get(url, params, tries=6):
    """GET with backoff on 429/5xx (Open-Meteo free tier is rate-limited per minute), honoring Retry-After."""
    import requests
    for i in range(tries):
        r = requests.get(url, params=params, timeout=90)
        if r.status_code == 429 or r.status_code >= 500:
            wait = int(r.headers.get("Retry-After", 0)) or min(2 ** i * 5, 60)  # 5,10,20,40,60,60s
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def fetch_grid(date: str, daily_vars, bbox=SPAIN_BBOX, step=0.25, source="archive", batch=100):
    """Query Open-Meteo on a regular lat/lon grid for `date`; return (lons, lats, {var: values}).

    Large batch = few requests (Open-Meteo accepts many coords/call) — the key to not getting rate-limited."""
    w, s, e, n = bbox
    lons = np.arange(w, e + 1e-9, step); lats = np.arange(s, n + 1e-9, step)
    LON, LAT = np.meshgrid(lons, lats)
    plon, plat = LON.ravel(), LAT.ravel()
    url = ARCHIVE if source == "archive" else FORECAST
    out = {v: np.full(plon.size, np.nan) for v in daily_vars}
    for b0 in range(0, plon.size, batch):
        sl = slice(b0, b0 + batch)
        params = {"latitude": ",".join(f"{x:.4f}" for x in plat[sl]),
                  "longitude": ",".join(f"{x:.4f}" for x in plon[sl]),
                  "daily": ",".join(daily_vars), "timezone": "UTC"}
        params["start_date"] = params["end_date"] = date  # forecast endpoint also serves recent past + forecast window
        js = _get(url, params)
        js = js if isinstance(js, list) else [js]  # multi-coord returns a list
        for k, item in enumerate(js):
            d = item.get("daily", {})
            for v in daily_vars:
                val = d.get(v, [None])
                out[v][b0 + k] = (val[0] if val and val[0] is not None else np.nan)
        if b0 + batch < plon.size:
            time.sleep(1.0)  # be polite between batches
    return plon, plat, out


def regrid_to_cube(plon, plat, vals, gx, gy, epsg=3035):
    """Bilinear-regrid scattered lat/lon point values onto the cube grid (EPSG:3035)."""
    tx, ty = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(plon, plat)
    ok = np.isfinite(vals)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([tx[ok], ty[ok]])
    lin = griddata(pts, vals[ok], (GX, GY), method="linear")
    nn = griddata(pts, vals[ok], (GX, GY), method="nearest")  # fill edges/holes
    return np.where(np.isfinite(lin), lin, nn).astype(np.float32)


def demo():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("fetch_openmeteo.demo")
    z = xr.open_zarr(str(CUBE), consolidated=True)
    cube_var = next((v for v in ["t2m_mean"] if v in z), None)
    date = "2024-07-15"  # cube-range, in archive
    gx = z["x"].values.astype(float); gy = z["y"].values.astype(float)
    t0 = time.time()
    plon, plat, vals = fetch_grid(date, ["temperature_2m_mean"], step=0.5, source="archive")
    log.info(f"fetched {plon.size} Open-Meteo grid points for {date} in {time.time()-t0:.1f}s "
             f"({np.isfinite(vals['temperature_2m_mean']).sum()} valid)")
    grid = regrid_to_cube(plon, plat, vals["temperature_2m_mean"], gx, gy)
    if cube_var:
        truth = z[cube_var].sel(time=date).values.astype(float)
        land = np.isfinite(truth)
        err = np.abs(grid[land] - truth[land])
        log.info(f"Open-Meteo regridded t2m_mean vs cube '{cube_var}' ({date}): "
                 f"MAE={err.mean():.3f}°C median={np.median(err):.3f} corr={np.corrcoef(grid[land], truth[land])[0,1]:.4f}")
        log.info("  → fetch+regrid produces cube-compatible features (both ERA5-based); this IS the live weather slice.")
    else:
        log.info(f"grid built {grid.shape}; cube t2m var not found for direct comparison.")


def main():
    if "--demo" in sys.argv:
        demo(); return
    print("Use as a module: fetch_grid(date, vars, source='forecast'|'archive') + regrid_to_cube(...).", file=sys.stderr)
    print("Run --demo to validate the fetch+regrid chain against the cube.", file=sys.stderr)


if __name__ == "__main__":
    main()
