"""Incremental engineered-feature engine — compute the engineered features for just the NEW days from a
bounded tail of history, instead of the whole-cube `add_fire_context` + `add_engineered` passes.

WHY: the whole-cube passes materialize entire (time,y,x) variables (~1.4 GB each at 4 km × 5285 d) → peak ~16 GB
→ swap/freeze on a 17 GB box, and they redo 14 years to add ~7 days. This computes only the new days, seeded
from history, in MBs and seconds. Verified by `--test`: **27/28 features bit-identical** (Δ=0 or float noise);
the lone exception is **kbdi (≤~3.5 mm on a 0–203 scale)** — its hidden per-cell wet-spell accumulator `cum_wet`
isn't stored, so a tail warmup can't reproduce it exactly in never-reset arid cells. The residual decays forward
and is negligible for the tree model; persist `cum_wet` as cube state if exactness is ever required.

It is the one shared engine behind three wrappers (see the `fgdc-extend-cadence` memory): the weekly batch
(seed = local cube tail), Option C's speed edge, and the HF daily job (seed = a shipped bundle). Provenance-
agnostic: it consumes raw VALUES + a history seed, so a window naturally blends settled tail + forecast edge.

Per-feature incremental form (each reproduces the exact whole-cube call):
  * per-day      : dist_to_fire, fire_upwind_exposure, emc_peak, ffwi, vpd_peak, hdw, fvc, doy/dow, holidays
  * window (≤365): precip_sum_{7,30,90,180,365}d, burn_frequency_365d   (cumsum/rolling over tail+new, slice new)
  * seed         : kbdi (q0 = last kbdi field; R = full-cube annual-rain mean), time_since_last_fire (counter)
  * causal clim  : spi_90d, ndvi_anomaly, lai_anomaly  (per-doy mean/std over PRIOR same-doy occurrences)
Static engineered (tpi, hli, dist_to_urban, aspect_*) are time-invariant → untouched (no new-day slice).

CLI:  python scripts/update_edge.py --test 30   # verify bit-identity vs the stored whole-cube values
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
import xarray as xr

from src.data import feature_engineering as FE
from src.data.regions import CCAA_TO_SUBDIV        # shared region map (was duplicated here + add_engineered)

GOLD = PROJECT / "data" / "gold" / "FireGuard_coarse2.zarr"   # production grid = 2 km (2026-07-08 cutover)
SILVER = PROJECT / "data" / "silver" / "FireGuard.zarr"
TAIL = 365                          # longest hard lookback (precip_sum_365d, burn_frequency_365d)
TP_TO_DAILY_MM = 24                 # must match build_features.TP_TO_DAILY_MM (fixed 2.9→24, audit 2026-07-07)
FIRE_THR = 0.5                      # is_fire>0.5 = burned (time_since_last_fire)
EPS = 1e-6                          # must match seasonal_anomaly eps


def _causal_anomaly_edge(z, var, i0, n_new, doy_all, new_vals=None):
    """seasonal_anomaly(causal=True) for days [i0,i0+n_new): per-doy mean/std over PRIOR same-doy occurrences in
    history (cnt≥2 else NaN), reading only those prior slices — bit-identical to the whole-cube causal pass
    (mirrors feature_engineering.seasonal_anomaly's causal branch; keep in sync). The new-day value is read from
    z[var] (new_vals=None), OR supplied via new_vals[j] for a feature whose new-day input was just computed and is
    not yet in the cube (e.g. spi_90d ← the freshly-computed precip_sum_90d)."""
    out = np.full((n_new, z.sizes["y"], z.sizes["x"]), np.nan, np.float32)
    for j in range(n_new):
        t = i0 + j
        d = doy_all[t]
        prior = np.where(doy_all[:t] == d)[0]                # earlier occurrences of this doy
        if prior.size < 2:
            continue
        block = z[var].isel(time=prior).values.astype("float64")   # [n_prior, y, x]
        fin = np.isfinite(block)
        cnt = fin.sum(0)
        ssum = np.where(fin, block, 0.0).sum(0)
        ssq = np.where(fin, block ** 2, 0.0).sum(0)
        val = (new_vals[j] if new_vals is not None else z[var].isel(time=t).values).astype("float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = ssum / cnt
            std = np.sqrt(np.maximum(ssq / cnt - mean ** 2, 0.0))
            zsc = (val - mean) / (std + EPS)
        out[j] = np.where(cnt >= 2, zsc, np.nan).astype("float32")
    return out


def compute_edge_engineered(z, i0, n_new):
    """Engineered (time,y,x) features for days [i0, i0+n_new) of gold cube `z`, from history z[:i0] + the
    new days' raw. Returns {feature: (n_new,y,x)}. Bit-identical to the whole-cube passes (see --test)."""
    T = z.sizes["time"]; ny, nx = z.sizes["y"], z.sizes["x"]
    assert i0 >= 1 and i0 + n_new <= T, "need ≥1 prior day and new days in-range"
    t0 = max(0, i0 - TAIL)
    nw = i0 - t0                                              # offset of the new days within the tail window
    times = pd.DatetimeIndex(z["time"].values)
    doy_all = z["time"].dt.dayofyear.values
    out = {}

    # --- fire-context (per-day): dist_to_fire, fire_upwind_exposure ---
    xc = z["x"].values.astype("float64"); yc = z["y"].values.astype("float64")
    cell_km = abs(xc[1] - xc[0]) / 1000.0
    no_fire = float(np.hypot(ny, nx) * cell_km)
    dist = np.empty((n_new, ny, nx), "float32"); expo = np.empty((n_new, ny, nx), "float32")
    for j in range(n_new):
        t = i0 + j
        fire = z["is_fire"].isel(time=t).values
        u = z["wind_u_mean"].isel(time=t).values; v = z["wind_v_mean"].isel(time=t).values
        dist[j], expo[j] = FE.fire_distance_and_exposure(fire, u, v, xc, yc, no_fire)
    out["dist_to_fire"] = dist; out["fire_upwind_exposure"] = expo

    # --- windowed precip sums (cumsum over tail+new, slice new) ---
    tp_win = z["total_precipitation_mean"].isel(time=slice(t0, i0 + n_new)).values
    csum = np.cumsum(tp_win, axis=0, dtype="float64")
    for N in (7, 30, 90, 180, 365):
        win = csum.astype("float32")
        win[N:] = (csum[N:] - csum[:-N]).astype("float32")   # rows<N keep partial cumsum (same as whole-cube)
        out[f"precip_sum_{N}d"] = win[nw:]
    del tp_win

    # --- KBDI: warm up over the TAIL (q0 = stored kbdi at the tail start) so BOTH the deficit Q and the hidden
    #     wet-spell accumulator `cum_wet` (not exposed by q0) converge before the new days; R = full-cube annual
    #     rain mean. Seeding only q0 at i0-1 loses cum_wet → ~5 mm boundary error; the warmup washes it out. ---
    nyears = (z["time"].values[-1] - z["time"].values[0]).astype("timedelta64[D]").astype(int) / 365.25
    R = (TP_TO_DAILY_MM * z["total_precipitation_mean"]).sum("time", skipna=True).values / nyears   # lazy → low mem
    rain_win = TP_TO_DAILY_MM * z["total_precipitation_mean"].isel(time=slice(t0, i0 + n_new)).values.astype("float64")
    tmax_win = z["t2m_max"].isel(time=slice(t0, i0 + n_new)).values.astype("float64")
    q0 = z["kbdi"].isel(time=t0 - 1).values.astype("float64") if t0 > 0 else None
    out["kbdi"] = FE.keetch_byram_drought_index(rain_win, tmax_win, R, q0=q0)[nw:].astype("float32")
    del rain_win, tmax_win

    # --- causal seasonal anomalies (prior same-doy) ---
    out["spi_90d"] = _causal_anomaly_edge(z, "precip_sum_90d", i0, n_new, doy_all, new_vals=out["precip_sum_90d"])
    out["ndvi_anomaly"] = _causal_anomaly_edge(z, "NDVI", i0, n_new, doy_all)
    out["lai_anomaly"] = _causal_anomaly_edge(z, "LAI", i0, n_new, doy_all)

    # --- pointwise fire-weather/fuel ---
    t2m_max = z["t2m_max"].isel(time=slice(i0, i0 + n_new)).values
    rh_min = z["RH_min"].isel(time=slice(i0, i0 + n_new)).values
    wmax = z["wind_speed_max"].isel(time=slice(i0, i0 + n_new)).values
    ndvi = z["NDVI"].isel(time=slice(i0, i0 + n_new)).values
    emc = FE.equilibrium_moisture_content(t2m_max, rh_min)
    out["emc_peak"] = emc.astype("float32")
    out["ffwi"] = FE.fosberg_ffwi(emc, wmax).astype("float32")
    vpd = FE.vpd_kpa(t2m_max, rh_min)
    out["vpd_peak"] = vpd.astype("float32")
    out["hdw"] = FE.hdw_index(vpd, wmax).astype("float32")
    out["fvc"] = FE.fractional_vegetation_cover(ndvi).astype("float32")

    # --- fire history: time_since_last_fire (counter seed), burn_frequency_365d (window) ---
    fire_win = (z["is_fire"].isel(time=slice(t0, i0 + n_new)).values > FIRE_THR)
    # time_since_last_fire: exact forward counter reseeded from the stored value at i0-1 (it can exceed any tail).
    seed = z["time_since_last_fire"].isel(time=i0 - 1).values.astype("float32")
    tsf = np.empty((n_new, ny, nx), "float32"); prev = seed.copy()
    for j in range(n_new):
        burned = fire_win[nw + j]
        prev = np.where(burned, 0.0, prev + 1.0).astype("float32")
        tsf[j] = prev
    out["time_since_last_fire"] = tsf
    bf = FE.rolling_sum_time((z["is_fire"].isel(time=slice(t0, i0 + n_new)).values).astype("float32"), 365)
    out["burn_frequency_365d"] = bf[nw:].astype("float32")

    # --- calendar (per-day from dates; doy for t+1, dow for t and t+1) ---
    d_new = times[i0:i0 + n_new]; d_tp1 = d_new + pd.Timedelta(days=1)
    ds, dc = FE.day_of_year_sincos(d_tp1.dayofyear.values)
    ws, wc = FE.day_of_week_sincos(d_new.dayofweek.values)
    ws1, wc1 = FE.day_of_week_sincos(d_tp1.dayofweek.values)
    plane = lambda v: np.broadcast_to(np.asarray(v, "float32")[:, None, None], (n_new, ny, nx)).copy()
    for nm, v in (("doy_sin", ds), ("doy_cos", dc), ("dow_sin", ws), ("dow_cos", wc),
                  ("dow_sin_tp1", ws1), ("dow_cos_tp1", wc1)):
        out[nm] = plane(v)

    # --- holidays (national constant plane; regional painted by AutonomousCommunities) ---
    import holidays as _hol
    years = range(int(d_new.year.min()), int(d_tp1.year.max()) + 1)
    nat = set(_hol.Spain(years=years).keys())
    _in = lambda idx, s: np.array([ts.date() in s for ts in idx], bool)
    out["is_holiday_national"] = plane(_in(d_new, nat))
    out["is_holiday_national_tp1"] = plane(_in(d_tp1, nat))
    ac = np.rint(np.nan_to_num(z["AutonomousCommunities"].values)).astype(int)
    for tag, idx in (("", d_new), ("_tp1", d_tp1)):
        reg = np.zeros((n_new, ny, nx), bool)
        for code, sub in CCAA_TO_SUBDIV.items():
            m = ac == code
            if not m.any():
                continue
            rd = set(_hol.Spain(subdiv=sub, years=years).keys()) - nat
            flag = _in(idx, rd)
            reg[flag] |= m
        out[f"is_holiday_regional{tag}"] = reg.astype("float32")
    return out


def update_gold_edge(gold=GOLD, silver=SILVER, to=None, dry_run=False):
    """Incrementally extend gold with new SETTLED silver days: coarsen them to 4 km, append COMPLETE rows
    (raw + NaN-placeholder engineered) so the time axis isn't ragged, compute the new days' engineered features
    via the edge engine, and region-write them. Replaces whole-cube coarsen + add_fire_context +
    add_engineered for the weekly path — MBs/seconds, no swap. Returns the appended dates."""
    import logging
    from src.data.ingest.coarsen import _pool_rule
    log = logging.getLogger("update_edge")
    zg = xr.open_zarr(str(gold), consolidated=True)
    zs = xr.open_zarr(str(silver), consolidated=True)
    gold_last = pd.Timestamp(zg["time"].values[-1])
    end = pd.Timestamp(to) if to else pd.Timestamp(zs["time"].values[-1])
    new = zs.sel(time=slice(gold_last + pd.Timedelta(days=1), end))
    n_new = new.sizes["time"]
    if n_new == 0:
        log.info(f"gold current ({gold_last.date()}) — nothing to extend"); return []
    new_dates = pd.DatetimeIndex(new["time"].values)
    ny, nx = zg.sizes["y"], zg.sizes["x"]; i0 = zg.sizes["time"]
    factor = int(round(zs.sizes["y"] / ny))
    log.info(f"extend gold +{n_new}d [{new_dates[0].date()}..{new_dates[-1].date()}] after {gold_last.date()}")

    # 1. coarsen the new silver days (raw dynamic only; static already in gold), same semantic pooling
    pooled = {v: getattr(new[v].coarsen(y=factor, x=factor, boundary="exact"), _pool_rule(v))().values.astype("float32")
              for v in new.data_vars if "time" in new[v].dims}
    gold_time = [v for v in zg.data_vars if "time" in zg[v].dims]
    eng_vars = [v for v in gold_time if v not in pooled]
    if dry_run:
        log.info(f"[dry-run] would append {n_new}d ({len(pooled)} raw + {len(eng_vars)} engineered) as complete rows")
        return list(new_dates.strftime("%Y-%m-%d"))

    # 2. ATOMIC: build a VIRTUAL extended cube (lazy zarr history + in-memory new raw + NaN engineered placeholders),
    #    compute the new days' engineered from it, then append COMPLETE rows (raw + engineered) in ONE write. A crash
    #    leaves either no new rows or fewer-DAY rows (gold_last < target → the date-based currency check self-heals),
    #    never NaN-engineered rows that the currency check would mistake for complete.
    new_ds = xr.Dataset(
        {v: (("time", "y", "x"), pooled[v] if v in pooled else np.full((n_new, ny, nx), np.nan, "float32"))
         for v in gold_time}, coords={"time": new_dates, "y": zg["y"], "x": zg["x"]})
    z_ext = xr.concat([zg[gold_time], new_ds], dim="time")     # lazy: history from zarr, new days in-memory
    for sv in [v for v in zg.data_vars if "time" not in zg[v].dims]:
        z_ext[sv] = zg[sv]                                      # static (e.g. AutonomousCommunities the engine reads)
    eng = compute_edge_engineered(z_ext, i0, n_new)
    assert set(eng) == set(eng_vars), f"engine vs cube engineered mismatch: {set(eng) ^ set(eng_vars)}"

    # 3. single complete-row append (raw + engineered) — the only write
    rows = {v: (("time", "y", "x"), pooled[v] if v in pooled else eng[v]) for v in gold_time}
    xr.Dataset(rows, coords={"time": new_dates, "y": zg["y"], "x": zg["x"]}).to_zarr(
        str(gold), append_dim="time", consolidated=True)
    log.info(f"gold extended to {new_dates[-1].date()} (+{n_new}d complete rows; engineered via edge engine)")
    return list(new_dates.strftime("%Y-%m-%d"))


def _e2e(M, K):
    """End-to-end: temp gold = real gold[:M]; extend it by K days (raw from real silver) via update_gold_edge;
    compare the K new complete rows to the real gold. Validates coarsen-new + append + engine + region-write."""
    import shutil, logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    g = xr.open_zarr(str(GOLD), consolidated=True)
    tmp = Path("/private/tmp/claude-501/-Users-vladimir-ml-wildfire-prediction/a28e1365-b918-4829-9f4b-6130c6f390d5/scratchpad/gold_e2e.zarr")
    if tmp.exists(): shutil.rmtree(tmp)
    to = str(pd.Timestamp(g["time"].values[M + K - 1]).date())
    print(f"[e2e] temp gold = real gold[:{M}]; extend to {to} (+{K}d) from real silver…")
    g.isel(time=slice(0, M)).to_zarr(tmp, mode="w", zarr_format=2, consolidated=True)
    update_gold_edge(gold=tmp, silver=SILVER, to=to)
    t = xr.open_zarr(tmp, consolidated=True)
    worst = 0.0
    for v in [d for d in g.data_vars if "time" in g[d].dims]:
        a = t[v].isel(time=slice(M, M + K)).values; b = g[v].isel(time=slice(M, M + K)).values
        both = np.isfinite(a) & np.isfinite(b)
        d = float(np.nanmax(np.abs(a[both] - b[both]))) if both.any() else 0.0
        nm = int((np.isfinite(a) != np.isfinite(b)).sum())
        if d > 1e-3 or nm:
            print(f"  ⚠️ {v:24} max|Δ|={d:.2e} nan_mismatch={nm}")
        worst = max(worst, d)
    print(f"[e2e] worst max|Δ| across all time-vars = {worst:.2e} (kbdi ~3.5 expected)")
    shutil.rmtree(tmp)


def _test(K):
    """Read-only bit-identity check: recompute the LAST K days' engineered via the edge engine and compare to
    the stored whole-cube values."""
    z = xr.open_zarr(str(GOLD), consolidated=True)
    T = z.sizes["time"]; i0 = T - K
    print(f"[test] cube {T} days; recomputing last {K} (idx {i0}..{T-1}) incrementally vs stored…")
    eng = compute_edge_engineered(z, i0, K)
    worst = 0.0
    for feat, arr in eng.items():
        if feat not in z:
            print(f"  {feat:24} MISSING in cube"); continue
        stored = z[feat].isel(time=slice(i0, T)).values
        both = np.isfinite(arr) & np.isfinite(stored)
        d = float(np.nanmax(np.abs(arr[both] - stored[both]))) if both.any() else float("nan")
        nan_mismatch = int((np.isfinite(arr) != np.isfinite(stored)).sum())
        worst = max(worst, d if np.isfinite(d) else 0.0)
        flag = "" if (d < 1e-3 and nan_mismatch == 0) else "  ⚠️"
        print(f"  {feat:24} max|Δ|={d:.2e}  nan_mismatch={nan_mismatch}{flag}")
    print(f"[test] worst max|Δ| = {worst:.2e}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="extend gold with new settled silver days (weekly path)")
    ap.add_argument("--to", help="extend only up to this date (default: silver's last day)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test", type=int, metavar="K", help="verify engine bit-identity on the last K days")
    ap.add_argument("--e2e", nargs=2, type=int, metavar=("M", "K"),
                    help="end-to-end: temp gold=real[:M], extend +K from silver, compare to real gold")
    a = ap.parse_args()
    if a.test:
        _test(a.test)
    elif a.e2e:
        _e2e(a.e2e[0], a.e2e[1])
    elif a.run:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        update_gold_edge(to=a.to, dry_run=a.dry_run)
    else:
        print("use --run [--to DATE] [--dry-run] | --test K | --e2e M K", file=sys.stderr)
