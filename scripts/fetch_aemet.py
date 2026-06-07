"""AEMET OpenData fetcher + station→grid interpolation — the first real LIVE feed (prototype).

Replaces ERA5-Land (5-day latency, unusable for real-time) with near-real-time AEMET station data, exactly
as the IberFire authors intended (they validated ERA5 vs AEMET per-station; see their
process_aemet_station_data). AEMET gives POINT stations; our model needs the GRID — so the new piece here
is point→grid interpolation (IDW), which upstream never did (their AEMET use was point-wise validation).

AEMET→our-units mapping (from upstream process_aemet_station_data):
  TMEDIA/TMIN/TMAX -> t2m mean/min/max ;  PRECIPITACION ("Ip"->0, "Acum"->NaN, /24) -> tp ;
  VELMEDIA + DIR(*10) -> wind speed + direction (-> u/v) ;  PRESMAX/PRESMIN -> surface pressure.

API: opendata.aemet.es needs a free key (env AEMET_API_KEY). Two-step: GET endpoint -> {"datos": url} -> GET url.
  daily climate: /api/valores/climatologicos/diarios/datos/fechaini/{ini}T00:00:00UTC/fechafin/{fin}T23:59:59UTC/estacion/{id}
  near-real-time: /api/observacion/convencional/datos/estacion/{id}  (last ~24h hourly -> aggregate to daily)

Run `--demo` (NO key/network): validates the GRIDDING by sampling a cube meteo field at N synthetic
"stations" and IDW-reconstructing the grid — quantifies the station-sparsity error (the real accuracy risk).
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

CUBE = Path(__file__).resolve().parents[1] / "data" / "gold" / "IberFire_coarse4.zarr"
AEMET_BASE = "https://opendata.aemet.es/opendata"
# upstream's validated meteo fields  ->  cube variable family
AEMET_TO_CUBE = {"TMEDIA": "t2m_mean", "TMIN": "t2m_min", "TMAX": "t2m_max",
                 "PRECIPITACION": "tp", "VELMEDIA": "wind_speed", "DIR": "wind_direction",
                 "PRESMAX": "surface_pressure_max", "PRESMIN": "surface_pressure_min"}


# ---------------- AEMET API (needs key + network) ----------------
def _aemet_get(path: str, key: str):
    import requests
    r = requests.get(f"{AEMET_BASE}{path}", params={"api_key": key}, timeout=30)
    r.raise_for_status()
    meta = r.json()
    if "datos" not in meta:
        raise RuntimeError(f"AEMET response has no 'datos' url: {meta}")
    return requests.get(meta["datos"], timeout=60).json()  # AEMET 2-step indirection


def normalize_aemet(df):
    """Apply upstream process_aemet_station_data transforms (units/sentinels) on a per-station daily frame."""
    import pandas as pd
    df = df.copy()
    if "PRECIPITACION" in df:
        df["PRECIPITACION"] = (df["PRECIPITACION"].replace("Ip", 0.0).replace("Acum", np.nan)
                               .astype(float) / 24.0)
    if "DIR" in df:
        df["DIR"] = df["DIR"].replace(99, np.nan).astype(float) * 10.0
        df.loc[df["DIR"] > 360, "DIR"] = np.nan
    for c in ["TMEDIA", "TMIN", "TMAX", "PRESMAX", "PRESMIN", "VELMEDIA", "RACHA"]:
        if c in df:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
    return df


def fetch_today(key: str, date: str):
    """Fetch daily climate for `date` for all stations; return normalized DataFrame (needs AEMET_API_KEY)."""
    import pandas as pd
    rows = _aemet_get(f"/api/valores/climatologicos/diarios/datos/fechaini/{date}T00:00:00UTC/"
                      f"fechafin/{date}T23:59:59UTC/todasestaciones/", key)
    return normalize_aemet(pd.DataFrame(rows))


# ---------------- station → grid (the new piece) ----------------
def idw_to_grid(pts_xy, vals, gx, gy, power=2.0, k=12):
    """Inverse-distance-weighted interpolation of scattered station values to a regular grid.
    pts_xy: (n,2) station coords (same CRS as gx/gy). gx,gy: 1-D grid axes. Returns grid[len(gy),len(gx)]."""
    ok = np.isfinite(vals)
    pts_xy, vals = pts_xy[ok], vals[ok]
    tree = cKDTree(pts_xy)
    GX, GY = np.meshgrid(gx, gy)
    q = np.column_stack([GX.ravel(), GY.ravel()])
    kk = min(k, len(vals))
    d, i = tree.query(q, k=kk)
    d = np.atleast_2d(d.T).T if kk == 1 else d
    i = np.atleast_2d(i.T).T if kk == 1 else i
    w = 1.0 / np.clip(d, 1.0, None) ** power
    out = (w * vals[i]).sum(1) / w.sum(1)
    out[(d[:, 0] == 0)] = vals[i[d[:, 0] == 0, 0]]  # exact hit
    return out.reshape(GY.shape)


# ---------------- demo: validate gridding on cube data (no key) ----------------
def demo(n_stations=250, seed=0):
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("fetch_aemet.demo")
    rng = np.random.default_rng(seed)
    z = xr.open_zarr(str(CUBE), consolidated=True)
    var = next((v for v in ["t2m_mean", "t2m_max", "t2m_min"] if v in z), None)
    if var is None:
        log.error("no t2m_* var in cube"); return
    gx = z["x"].values.astype(float); gy = z["y"].values.astype(float)
    field = z[var].isel(time=int(len(z.time) // 2)).values.astype(float)  # one day's true grid
    land = np.isfinite(field)
    GX, GY = np.meshgrid(gx, gy)
    li, lj = np.where(land)
    pick = rng.choice(li.size, min(n_stations, li.size), replace=False)  # synthetic "stations" on land
    sx, sy = GX[li[pick], lj[pick]], GY[li[pick], lj[pick]]
    svals = field[li[pick], lj[pick]]
    t0 = time.time()
    recon = idw_to_grid(np.column_stack([sx, sy]), svals, gx, gy)
    dt = time.time() - t0
    err = np.abs(recon[land] - field[land])
    log.info(f"DEMO gridding '{var}': {n_stations} synthetic stations -> {land.sum()} land cells in {dt:.2f}s")
    log.info(f"  reconstruction MAE={err.mean():.3f}  median={np.median(err):.3f}  p95={np.percentile(err,95):.3f} "
             f"(units of {var}; field range {np.nanmin(field):.1f}..{np.nanmax(field):.1f})")
    log.info(f"  corr(recon, true)={np.corrcoef(recon[land], field[land])[0,1]:.4f}")
    log.info("  → this is the station-sparsity error budget for the live weather feed (real AEMET ~250 stns).")


def main():
    if "--demo" in sys.argv:
        n = 250
        for a in sys.argv:
            if a.startswith("--n="):
                n = int(a.split("=")[1])
        demo(n_stations=n); return
    key = os.getenv("AEMET_API_KEY")
    if not key:
        print("Set AEMET_API_KEY (free at https://opendata.aemet.es) for live fetch, or run --demo.", file=sys.stderr)
        sys.exit(2)
    import datetime as _dt  # noqa
    print("Live AEMET fetch path is wired (fetch_today + idw_to_grid); supply AEMET_API_KEY and a date.")


if __name__ == "__main__":
    main()
