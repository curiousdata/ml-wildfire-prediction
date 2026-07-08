"""Derived-feature engineering for the IberFire cube.

Functions here take native (1 km, silver) variables and produce new derived
features. They are pure array functions (NumPy / xarray DataArrays both work),
so they can run on a single time slice for validation or be mapped across the
whole cube when materializing the derived feature store.

Keep `silver/IberFire.zarr` immutable: write outputs to a separate derived store.

--------------------------------------------------------------------------------
Wind convention (confirmed 2026-06-05 against the cube attrs, the IberFire paper
arXiv:2505.00837 eq. (3), and the author's repo):

  `wind_direction_*` is the METEOROLOGICAL convention — the direction the wind
  blows *FROM*, in degrees measured CLOCKWISE from true north (0°=N, 90°=E,
  180°=S, 270°=W). The dataset derived it from ERA5-Land u/v and then discarded
  the components; `wind_to_uv` reconstructs them.

`wind_to_uv` returns the standard ERA5 components of the vector the wind blows
*TOWARD* (u = eastward, v = northward) — i.e. the original components before the
direction-from was computed. Getting the sign wrong here would flip every
downwind/upwind feature by 180°; `uv_to_direction_deg` is the exact inverse and
was round-trip-validated against the stored `wind_direction_mean` on a real
silver slice (max |Δ| = 0.0°, 2026-06-05).
"""

from __future__ import annotations

import numpy as np


def wind_to_uv(speed, direction_deg):
    """Reconstruct (u, v) wind components from speed + meteorological direction.

    Args:
        speed: Wind speed magnitude (m/s). Array-like (NumPy or xarray).
        direction_deg: Wind direction in degrees, meteorological "from"
            convention (clockwise from true north — the direction the wind
            comes FROM), as stored in `wind_direction_mean` /
            `wind_direction_at_max_speed`.

    Returns:
        (u, v): eastward and northward components of the vector the wind blows
        TOWARD (ERA5 convention). NaNs in the inputs propagate unchanged.

    Convention check (wind FROM θ blows TOWARD θ+180):
        θ=0   (from N) -> u=0,        v=-speed  (toward S)
        θ=90  (from E) -> u=-speed,   v=0       (toward W)
        θ=180 (from S) -> u=0,        v=+speed  (toward N)
        θ=270 (from W) -> u=+speed,   v=0       (toward E)
    """
    theta = np.deg2rad(direction_deg)
    u = -speed * np.sin(theta)  # eastward component the wind blows toward
    v = -speed * np.cos(theta)  # northward component the wind blows toward
    return u, v


def uv_to_direction_deg(u, v):
    """Inverse of `wind_to_uv`: recover meteorological "from" direction (degrees).

    Mirrors the dataset's derivation (IberFire paper eq. 3): the bearing of the
    wind vector clockwise from north is `atan2(u, v)`; the "from" direction is
    that minus 180°. Used to round-trip-validate `wind_to_uv` against the stored
    `wind_direction_*`.
    """
    bearing = np.rad2deg(np.arctan2(u, v))          # direction wind blows TOWARD
    return np.mod(bearing - 180.0, 360.0)            # direction wind comes FROM


# ---------------------------------------------------------------------------
# Fire-weather: vapour pressure deficit (VPD) and Hot-Dry-Windy (HDW)
# ---------------------------------------------------------------------------
# Pure functions of existing IberFire variables (t2m in °C, RH in %, wind in m/s).
# VPD is the atmosphere's drying power (the direct driver of dead-fuel moisture);
# HDW = VPD × wind couples that with spread potential (Srock et al. 2018).
# Both partially overlap the existing FWI — step-3 analysis must check whether
# they add signal *beyond* FWI, not just that they're predictive.

# Magnus / Alduchov–Eskridge (1996) coefficients, over water (fire season is warm,
# so the over-ice branch is irrelevant). The choice vs. Tetens is a <1% difference.
_MAGNUS_A = 17.625
_MAGNUS_B = 243.04  # °C


def saturation_vapour_pressure_kpa(t_celsius):
    """Saturation vapour pressure e_s(T) in kPa, T in °C (Magnus formula)."""
    return 0.6108 * np.exp(_MAGNUS_A * t_celsius / (t_celsius + _MAGNUS_B))


def vpd_kpa(t_celsius, rh_pct):
    """Vapour pressure deficit (kPa) from temperature (°C) and RH (%).

    VPD = e_s(T) · (1 − RH/100). Clamped to ≥ 0 (RH can nudge just past 100 from
    rounding/supersaturation, which would otherwise give a tiny negative deficit).

    Two physically distinct variants are worth computing because e_s is exponential
    in T, so VPD(mean inputs) ≠ mean(VPD):
      - VPD_mean: vpd_kpa(t2m_mean, RH_mean)  — the day's average drying.
      - VPD_peak: vpd_kpa(t2m_max,  RH_min)   — the hot-dry afternoon extreme
        (usually the more fire-relevant; hottest and driest typically co-occur).
    """
    es = saturation_vapour_pressure_kpa(t_celsius)
    vpd = es * (1.0 - rh_pct / 100.0)
    return np.maximum(vpd, 0.0)


def equilibrium_moisture_content(t_celsius, rh_pct):
    """Equilibrium moisture content (%) — Simard (1968) / NFDRS, a proxy for 1-hr
    dead-fuel moisture (LOWER = drier fuel = more flammable).

    Piecewise in RH; temperature converted to °F internally. Pair with the daytime
    extreme (t2m_max, RH_min) for the most-flammable "peak" value, mirroring VPD_peak.
    """
    rh = np.asarray(rh_pct, dtype="float32")
    t_f = np.asarray(t_celsius, dtype="float32") * 9.0 / 5.0 + 32.0
    emc = np.where(
        rh < 10.0,
        0.03229 + 0.281073 * rh - 0.000578 * rh * t_f,
        np.where(
            rh <= 50.0,
            2.22749 + 0.160107 * rh - 0.014784 * t_f,
            21.0606 + 0.005565 * rh ** 2 - 0.00035 * rh * t_f - 0.483199 * rh,
        ),
    )
    return np.maximum(emc, 0.0)


def keetch_byram_drought_index(daily_rain_mm, t_max_c, annual_rain_mm, q0=None):
    """Keetch-Byram Drought Index (mm of soil-moisture deficit, 0..203.2), metric form.

    A recursive daily drought accumulator — repeatedly among the strongest fire
    predictors. Builds with hot/dry days, drops after rain. HIGHER = drier.

    Args:
        daily_rain_mm: (time, y, x) daily TOTAL rainfall in mm (note: from this cube,
            ``24 * total_precipitation_mean`` since that var is an hourly mean).
        t_max_c: (time, y, x) daily max temperature (°C).
        annual_rain_mm: (y, x) mean annual rainfall (mm) per pixel.
        q0: optional (y, x) initial deficit (default 0 = saturated; ~1 month spin-up
            before the 2008 training start).

    Returns:
        (time, y, x) float64 KBDI. NaNs (sea) propagate.

    Rainfall handling follows Keetch-Byram: the first 5.08 mm (0.20 in) of each wet
    spell is intercepted (no effect); rain beyond that reduces the deficit. The
    drought factor is clamped >= 0 (drying cannot be negative on cold days).
    """
    # memory-bound: keep whole-cube inputs at their (fp16) dtype and upcast only the per-day SLICE inside the
    # loop (a few MB) — this is a per-grid time-recursion, so it never needs the inputs as full f32 arrays
    # (peak ~8.7 not ~14.5 GB). Output float16 (KBDI clamped 0..203.2 → fp16 exact enough); the RECURSIVE
    # per-grid accumulators Q/cum_wet stay float64 for drift-free accumulation.
    rain = np.asarray(daily_rain_mm)
    tmax = np.asarray(t_max_c)
    R = np.asarray(annual_rain_mm, dtype="float64")
    nt = rain.shape[0]
    Q = (np.zeros(rain.shape[1:], dtype="float64") if q0 is None
         else np.asarray(q0, dtype="float64").copy())
    cum_wet = np.zeros_like(Q)
    denom = 1.0 + 10.88 * np.exp(-0.001736 * R)
    THRESH = 5.08
    out = np.empty(rain.shape, dtype="float16")
    for t in range(nt):
        r = rain[t].astype("float32")   # per-day slice upcast (a few MB), not the whole cube
        wet = r > 0
        prev_excess = np.maximum(cum_wet - THRESH, 0.0)
        cum_wet = np.where(wet, cum_wet + r, 0.0)          # reset wet spell on dry days
        # net rain only reduces Q on WET days (on a dry day the spell resets, which
        # would otherwise make this difference negative and spuriously add deficit).
        net_rain = np.where(wet, np.maximum(cum_wet - THRESH, 0.0) - prev_excess, 0.0)
        Q = np.maximum(Q - net_rain, 0.0)
        dq = (203.2 - Q) * (0.968 * np.exp(0.0875 * tmax[t].astype("float32") + 1.5552) - 8.30) / denom * 1e-3
        Q = np.minimum(Q + np.maximum(dq, 0.0), 203.2)
        out[t] = Q
    return out


def fosberg_ffwi(emc_pct, wind_speed_ms):
    """Fosberg Fire Weather Index from EMC (%) and wind speed (m/s).

    Fast-reacting fire-weather index (complements the slow FWI). Higher = more
    dangerous (dry fuel + strong wind). Unbounded-ish but typically 0–100.
    """
    m = np.minimum(np.asarray(emc_pct, dtype="float32"), 30.0) / 30.0  # moisture damping, capped
    eta = 1.0 - 2.0 * m + 1.5 * m ** 2 - 0.5 * m ** 3
    u_mph = np.asarray(wind_speed_ms, dtype="float32") * 2.236936
    return eta * np.sqrt(1.0 + u_mph ** 2) / 0.3002


def fractional_vegetation_cover(ndvi, ndvi_soil: float = 0.05, ndvi_veg: float = 0.86):
    """Fractional vegetation cover [0, 1] from NDVI (Carlson & Ripley 1997)."""
    n = np.asarray(ndvi, dtype="float32")
    fvc = ((n - ndvi_soil) / (ndvi_veg - ndvi_soil)) ** 2
    return np.clip(fvc, 0.0, 1.0)


def hdw_index(vpd_kpa_value, wind_speed_ms):
    """Hot-Dry-Windy index ≈ VPD × wind speed (Srock et al. 2018).

    Surface, daily-extreme proxy: pair VPD_peak with wind_speed_max. NOTE this
    over-states the true index slightly — the original maxes the VPD×wind product
    through the lowest ~500 m from sub-daily data, and peak VPD and peak wind need
    not co-occur within the day. Fine as a daily fire-danger *ordering*; the
    absolute scale is irrelevant since training normalizes the channel.
    """
    return vpd_kpa_value * wind_speed_ms


# ---------------------------------------------------------------------------
# Seasonality: day-of-year as a cyclic (sin, cos) pair
# ---------------------------------------------------------------------------
# Fire risk in Spain is strongly annual; this gives the model a smooth seasonal
# prior. sin/cos removes the Dec-31 -> Jan-1 discontinuity that raw day-number has,
# and the pair (vs. a single sinusoid) uniquely identifies the phase around the year.
#
# NOTE: this varies in TIME only (one scalar pair per day, identical across all
# pixels). Storing it as a full (time, y, x) field would duplicate one value across
# ~1.1M cells per day — at materialization, prefer a (time,) variable broadcast at
# read time, or compute it on-the-fly from the cube's `time` coordinate.

_DAYS_PER_YEAR = 365.25  # fractional, so the phase doesn't drift on leap years


def day_of_year_sincos(day_of_year, period: float = _DAYS_PER_YEAR):
    """Encode day-of-year (1..366) as a cyclic (sin, cos) pair on the unit circle.

    Args:
        day_of_year: integer day-of-year, 1-based (e.g. from
            ``xarray.DataArray.dt.dayofyear``). Array-like.
        period: length of the cycle in days (default 365.25).

    Returns:
        (doy_sin, doy_cos): two arrays in [-1, 1] with ``sin**2 + cos**2 == 1``.
        Angle is 0 at Jan 1, advancing through the year — so Dec 31 and Jan 1 are
        adjacent on the circle (no New-Year seam).
    """
    angle = 2.0 * np.pi * (np.asarray(day_of_year) - 1.0) / period
    return np.sin(angle), np.cos(angle)


def day_of_week_sincos(day_of_week, period: float = 7.0):
    """Encode day-of-week as a cyclic (sin, cos) pair (weekly human-activity rhythm).

    Captures the weekly ignition cadence (weekday vs weekend) that human-caused
    fires follow — orthogonal to the annual cycle (`day_of_year_sincos`) and to
    `is_holiday`. Sunday sits adjacent to Monday on the circle (no week seam).

    Args:
        day_of_week: 0-based day index, Monday=0 .. Sunday=6 (e.g. from
            ``xarray.DataArray.dt.dayofweek``). Array-like.
        period: length of the cycle in days (default 7).

    Returns:
        (dow_sin, dow_cos): arrays in [-1, 1] with ``sin**2 + cos**2 == 1``.

    Caveat: the EFFIS `is_fire` label is a burned-area polygon stamped across each
    fire's start–end range, so the weekly *ignition* signal is smeared over
    multi-day fire durations. Expect this feature's value to show up mainly in the
    new-ignition vs. continuation evaluation, not the blended label.
    """
    angle = 2.0 * np.pi * np.asarray(day_of_week) / period
    return np.sin(angle), np.cos(angle)


# ---------------------------------------------------------------------------
# Spatial fire-context (§E) — computed on the COARSE fire mask, post-coarsen
# ---------------------------------------------------------------------------
# Resolution-coupled (distance/advection only mean something at the working cell
# size), so these live on the coarse grid, not on 1 km silver. Causal: derived
# from is_fire(t) to predict is_fire(t+1). Both strongly favour *continuation*
# (cells near today's fire burn tomorrow) -> read their value through the §A
# new-ignition vs. continuation split.


def seasonal_anomaly(values, doy, eps: float = 1e-6, causal: bool = False, clip: float = 10.0):
    """Standardized anomaly vs the day-of-year climatology (per pixel).

    For each calendar day-of-year d, z = (value − mean_d) / std_d. Captures "wetter/greener/drier than
    normal for the season." Unit-independent (standardized), so it doubles as:
      - SPI-like  : seasonal_anomaly(precip_sum_90d, doy)
      - greenness : seasonal_anomaly(NDVI, doy) / seasonal_anomaly(LAI, doy)

    `causal=False` (default): mean_d/std_d over ALL years sharing that doy — convenient but it leaks the
    test period into train features AND is not reproducible at serve time (no future years). Use only for
    one-shot exploratory stats.
    `causal=True`: for each occurrence of doy d, the climatology uses ONLY PRIOR years (expanding window) —
    no train/test leakage, and identical to what's available when serving "today". The first occurrence of
    each doy has no prior → NaN (cold start; the model handles NaN).

    Args:
        values: (time, y, x) array. doy: (time,) integer day-of-year (1..366).
    Returns:
        (time, y, x) float32 z-scores. NaNs propagate.
    """
    values = np.asarray(values, dtype="float32")   # whole-cube → float32 (z-score precision fine); per-doy blocks small
    doy = np.asarray(doy)
    if not causal:
        out = np.empty_like(values, dtype="float32")
        for d in np.unique(doy):
            sel = doy == d
            block = values[sel]
            mean = np.nanmean(block, axis=0)
            std = np.nanstd(block, axis=0)
            # clip degenerate z-scores: near-constant cells (std≈0) otherwise blow up past ±1e4 (fp16 → inf);
            # |z|>clip σ is a numerical artifact, not seasonal signal. NaN is preserved by np.clip.
            out[sel] = np.clip((block - mean) / (std + eps), -clip, clip).astype("float32")
        return out
    out = np.full(values.shape, np.nan, dtype="float32")          # causal: NaN where no prior year exists
    for d in np.unique(doy):
        idx = np.where(doy == d)[0]                               # this doy's occurrences, chronological
        block = values[idx]                                      # [n_occ, y, x]
        if block.shape[0] < 2:
            continue
        fin = np.isfinite(block)                                 # NaN-aware expanding mean/std (O(n))
        cnt = np.cumsum(fin, axis=0).astype("float64")
        ssum = np.cumsum(np.where(fin, block, 0.0), axis=0)
        ssq = np.cumsum(np.where(fin, block ** 2, 0.0), axis=0)
        for k in range(1, block.shape[0]):                       # occurrence k uses priors 0..k-1 (index k-1)
            c = cnt[k - 1]
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = ssum[k - 1] / c
                std = np.sqrt(np.maximum(ssq[k - 1] / c - mean ** 2, 0.0))
                z = np.clip((block[k] - mean) / (std + eps), -clip, clip)   # clip degenerate near-constant-cell blowups
            out[idx[k]] = np.where(c >= 2, z, np.nan).astype("float32")   # need ≥2 priors (1 → std=0 → blowup)
    return out


def topographic_position_index(elevation, size: int = 5):
    """TPI: elevation minus the local (size×size) mean — ridge (>0) vs valley (<0).

    NaN-aware (sea cells stay NaN; the local mean ignores NaN neighbours).
    """
    from scipy.ndimage import uniform_filter

    e = np.asarray(elevation, dtype="float64")
    finite = np.isfinite(e)
    e0 = np.where(finite, e, 0.0)
    num = uniform_filter(e0, size=size, mode="nearest")
    den = uniform_filter(finite.astype("float64"), size=size, mode="nearest")
    local_mean = num / np.maximum(den, 1e-6)
    return np.where(finite, e - local_mean, np.nan).astype("float32")


def terrain_curvature(elevation):
    """Terrain curvature (Laplacian of elevation) — convex (>0) vs concave (<0)."""
    from scipy.ndimage import laplace

    e = np.asarray(elevation, dtype="float64")
    finite = np.isfinite(e)
    curv = laplace(np.where(finite, e, 0.0))
    return np.where(finite, curv, np.nan).astype("float32")


def heat_load_index(slope_deg, aspect_deg, lat_deg):
    """McCune & Keon (2002) Heat Load Index (eq. 3) — terrain solar load.

    Captures how much solar energy a slope receives (south/SW-facing + steep +
    lower-latitude => hotter, drier fuels). Folds aspect about 225° (SW = warmest).
    All inputs in degrees; returns heat load (~0.03–1.1, dimensionless).

    Note: in this cube aspect is only 8-sector one-hots, so `aspect_deg` is a
    reconstructed continuous angle — HLI here is approximate. It varies by region
    (terrain + latitude), so it's kept as a candidate for the spatial model to use.
    """
    s = np.deg2rad(np.asarray(slope_deg, dtype="float64"))
    lat = np.deg2rad(np.asarray(lat_deg, dtype="float64"))
    folded = np.deg2rad(np.abs(180.0 - np.abs(np.asarray(aspect_deg, dtype="float64") - 225.0)))
    ln_hl = (-1.467 + 1.582 * np.cos(lat) * np.cos(s)
             - 1.500 * np.cos(folded) * np.sin(s) * np.sin(lat)
             - 0.262 * np.sin(lat) * np.sin(s)
             + 0.607 * np.sin(folded) * np.sin(s))
    return np.exp(ln_hl)


def fire_distance_and_exposure(fire_mask, wind_u, wind_v, x_coords, y_coords, no_fire_dist_km):
    """Spatial fire-context for ONE day.

    Args:
        fire_mask: (H, W) truthy where fire on day t.
        wind_u, wind_v: (H, W) wind components the wind blows TOWARD (m/s); NaN over sea.
        x_coords: (W,) easting per column (m, EPSG:3035, increasing).
        y_coords: (H,) northing per row (m; decreasing downward — handled via values).
        no_fire_dist_km: value to fill `dist_to_fire` on days with no fire anywhere.

    Returns:
        (dist_km (H, W) float32, exposure (H, W) float32):
          dist_km  = distance to nearest fire cell (km), 0 on fire cells.
          exposure = (W . d) / |d|^2  (downwind-exposure; >0 downwind of a nearby fire,
                     <0 upwind, ~0 far; 0 on fire cells). NaN where wind is NaN (sea).
    """
    from scipy.ndimage import distance_transform_edt

    mask = np.asarray(fire_mask) > 0
    H, W = mask.shape
    if not mask.any():
        return (np.full((H, W), no_fire_dist_km, dtype="float32"),
                np.zeros((H, W), dtype="float32"))

    # distance_transform_edt measures distance to the nearest ZERO; invert so fire=0.
    dist_cells, (rf, cf) = distance_transform_edt(~mask, return_indices=True)
    cell_m = abs(float(x_coords[1] - x_coords[0]))
    dist_m = dist_cells * cell_m

    # fire -> cell displacement in metres, from COORDINATE VALUES (sign-safe):
    east = x_coords[np.newaxis, :] - x_coords[cf]   # x_cell - x_nearestfire
    north = y_coords[:, np.newaxis] - y_coords[rf]   # y_cell - y_nearestfire
    dot = wind_u * east + wind_v * north
    with np.errstate(divide="ignore", invalid="ignore"):
        exposure = np.where(dist_m > 0, dot / (dist_m ** 2), 0.0)

    return (dist_m / 1000.0).astype("float32"), exposure.astype("float32")


# ---------------------------------------------------------------------------
# Antecedent dryness (temporal-window precipitation features)
# ---------------------------------------------------------------------------
# Trailing windows look BACKWARD (days t-N+1 .. t, inclusive of today) -> causal
# for a t+1 target. They are functions of past INPUTS only, so computing them on
# the full series before any train/val/test split does NOT leak the label.
#
# These partially overlap FWI's drought codes (DC/DMC are exponential precip
# accumulators) -> step-3 analysis must check incremental value over FWI.
#
# Memory: time is axis 0. On the full (time, y, x) cube these can't fit in RAM at
# once -> at materialization, run per spatial chunk (full time each). The
# xarray-lazy equivalent of `rolling_sum_time` is
# `da.rolling(time=window, min_periods=window).sum()`.

# Trailing windows to materialize (days). Short-term surface dryness -> drought.
ANTECEDENT_WINDOWS_DAYS = (7, 30, 90)

# "Dry day" threshold calibrated to `total_precipitation_mean` (an hourly-mean in
# mm): < 1 mm/day total == hourly-mean < 1/24 mm. ~46% of land-days qualify
# (verified on 2020-2022, 2026-06-05). Pass this to `days_since_rain`.
DRY_DAY_THRESHOLD_MM = 1.0 / 24.0


def rolling_sum_time(arr, window: int):
    """Trailing-window sum along axis 0 (time), inclusive of the current index.

    Args:
        arr: ndarray with time as axis 0 (e.g. precipitation (time, y, x)).
        window: number of days in the trailing window (>= 1).

    Returns:
        Same-shape float64 array. Rows before a full window is available (the
        first ``window - 1``) are NaN. NaNs in the input propagate (sea cells are
        NaN throughout and are masked downstream).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    arr = np.asarray(arr, dtype="float32")   # counts (fire-days) — float32 is exact up to 2^24; whole-cube memory
    csum = np.cumsum(arr, axis=0)
    out = np.full_like(csum, np.nan)
    out[window - 1:] = csum[window - 1:]                 # t == window-1: sum of first `window`
    out[window:] = csum[window:] - csum[:-window]        # t >= window:   csum[t] - csum[t-window]
    return out


def days_since_rain(precip, threshold: float):
    """Length of the dry spell ending at each day (consecutive days < threshold).

    Recovers the recency/ordering information that a rolling *sum* discards: "80
    dry days then rain" and "rain then 80 dry days" have the same 90-day sum but
    very different `days_since_rain`.

    Args:
        precip: ndarray with time as axis 0. NOTE `total_precipitation_mean` is an
            hourly-mean (mm), so `threshold` must be in those (small) units —
            calibrate it, don't assume ~1 mm.
        threshold: a day with ``precip < threshold`` counts as "dry". NaN days
            (sea) compare False, i.e. treated as not-dry (count stays 0).

    Returns:
        float64 array, same shape; value at t = number of consecutive dry days
        ending at (and including) t. 0 on a wet day.
    """
    # memory-bound: keep `precip` at its input dtype (comparison is exact), and use int16 for the whole-cube
    # dry-day cumsum/accumulators — the spell length can't exceed the series length (< 32767 days ≈ 89 yr).
    precip = np.asarray(precip)
    is_dry = precip < threshold                          # NaN < x -> False (sea -> not dry)
    csum = np.cumsum(is_dry.astype("int16"), axis=0)
    reset = np.where(~is_dry, csum, np.int16(0))         # snapshot the count at each wet day
    running = np.maximum.accumulate(reset, axis=0)       # last wet-day count seen so far
    return (csum - running).astype("float32")            # steps since that reset
