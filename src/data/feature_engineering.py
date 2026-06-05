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


def hdw_index(vpd_kpa_value, wind_speed_ms):
    """Hot-Dry-Windy index ≈ VPD × wind speed (Srock et al. 2018).

    Surface, daily-extreme proxy: pair VPD_peak with wind_speed_max. NOTE this
    over-states the true index slightly — the original maxes the VPD×wind product
    through the lowest ~500 m from sub-daily data, and peak VPD and peak wind need
    not co-occur within the day. Fine as a daily fire-danger *ordering*; the
    absolute scale is irrelevant since training normalizes the channel.
    """
    return vpd_kpa_value * wind_speed_ms
