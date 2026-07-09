"""Fire Guard Control Center — dark, map-first live view of next-day wildfire risk for Spain.

Design: the map is the hero — a full-width dark schematic base (CartoDB dark_matter: place names + borders,
no imagery) with tomorrow's risk rendered as a smooth, day-stable yellow→red glow, and today's active fire as
crisp cyan. Below it: a compact sources/freshness line and two translucent cards (NOW | TOMORROW). The only
saturated colours on the page are the prediction itself.

This app only READS the latest published prediction (HF Dataset, Ship A; local store fallback) and renders it —
no cube, no model, no rate-limited fetch. A 10 s fragment swaps in a new prediction when one lands; a 60 s
fragment refreshes the "updated" stamp.
"""
from __future__ import annotations
import base64
import io
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:                                          # local pass times for a Spanish command center (DST-correct)
    from zoneinfo import ZoneInfo
    MADRID = ZoneInfo("Europe/Madrid")
except Exception:
    MADRID = None

import folium
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageFilter
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent
STORE = Path("/data") if (Path("/data") / "grids").exists() else ROOT / "store"
ASSETS = ROOT / "display_assets.npz"
SERVING_REPO = os.getenv("FIREGUARD_SERVING_REPO", "curiousdata/fireguard-serving")
FORCE_LOCAL = os.getenv("FIREGUARD_LOCAL_STORE") == "1"
CELL_KM2 = 4.0    # 2 km cell area (was 16.0 at 4 km) — drives exp_cells / per-area aggregates
CELL_KM = CELL_KM2 ** 0.5    # grid cell size (km). The spatial kernels below are PHYSICAL (km) and converted to
                             # cells at use, so they hold their real-world extent across resolutions (they were
                             # hardcoded in 4 km pixels: gaussian 1.2, maximum_filter 5, GaussianBlur 0.8).
CLUSTER_SMOOTH_KM = 4.8      # danger-area pre-peak smoothing (1.2 cells × 4 km)
CLUSTER_PEAKSEP_KM = 20.0    # min separation between danger-area peaks (maximum_filter size 5 × 4 km)
RISK_BLUR_KM = 3.2           # cosmetic risk-glow blur (GaussianBlur 0.8 × 4 km)
# Colour + alert anchoring is PREVALENCE-DERIVED: everything scales off BASE_RATE = the measured per-cell next-day
# fire prevalence (mean calibrated prob over land). MEASURED for the 2 km 3-bird KBDI-fixed cube on a 30-day
# 6-pass replay to 2026-06-27 = 1.276e-3. ⚠️ RE-MEASURE + update BASE_RATE on any resolution / label / calibrator
# change (the 4 km value was ≈7e-4). Multiples below are preserved from the 4 km tuning, so semantics carry over.
BASE_RATE = 1.276e-3
# ALERT tiers as prevalence multiples: moderate ≈7×, elevated ≈28×, high ≈100× base. (Calibrated per-cell probs are
# tiny + heavy-tailed, so the old fixed 0.5/0.2/0.05 almost never triggered.)
HIGH, ELEV, MOD = 100 * BASE_RATE, 28 * BASE_RATE, 7 * BASE_RATE
# Map colour ramp = LOG scale over the calibrated-prob dynamic range. Log-ramp from RAMP_FLOOR (≈1.3× base ≈ land
# p85; below → transparent so quiet days/noise floor stay dark) to RAMP_CAP (≈19× base ≈ land p99–p99.9 → full red).
# Absolute (day-comparable), reveals the whole heavy-tailed field instead of clipping to black.
RAMP_FLOOR, RAMP_CAP = 1.3 * BASE_RATE, 19 * BASE_RATE   # lower floor → more low-risk pixels visible (as dark red)
_LF = float(np.log(RAMP_FLOOR)); _LS = float(np.log(RAMP_CAP) - np.log(RAMP_FLOOR))
CLUSTER_THR = 8.6 * BASE_RATE          # cells above this (≈p99) seed/join the danger-area watershed
CLUSTER_MIN_P = 0.10                   # surface every area with ≥ this aggregate P(≥1 ignition) — the (only) knob
# feature → human cause phrase, in the RISK-RAISING direction (for the danger-area hover; never show raw names)
CAUSE_MAP = [
    (("dist_to_fire", "fire_upwind_exposure"), "near active fire"),   # CURRENT proximity only (time_since = history)
    (("kbdi", "spi_90d", "precip_sum_90d", "precip_sum_180d"), "very dry ground"),   # low spi/precip, high kbdi = dry
    (("t2m_max", "t2m_mean"), "extreme heat"),
    (("ffwi", "wind_speed_max", "RH_min"), "hot-dry-windy weather"),   # RH_min low = dry (was 'rh_min' — never matched)
    (("emc_peak", "ndvi_anomaly", "fvc", "lai_anomaly"), "dry, abundant fuel"),   # (NDVI dropped — ambiguous direction)
    (("time_since_last_fire",), "long-unburned fuel buildup"),   # HIGH time-since = fuel accumulation = ↑risk
    (("popdens", "dist_to_roads_mean", "dist_to_roads_stdev", "dist_to_urban"), "close to people & roads"),
    (("burn_frequency_365d",), "recent fire history"),
]
# VIIRS overpass schedule over Spain (~0° lon → overpass clock ≈ UTC), as UTC minutes-of-day per bird: a night
# (descending) + an afternoon (ascending) pass. Three co-planar birds within ~1 h: NOAA-21 leads, then NOAA-20,
# then Suomi-NPP (last → it sets the completeness gate). A pass becomes FETCHABLE ~FIRMS_SETTLE_H after overpass
# (FIRMS NRT latency) — that's when the serve can fold it in. NOAA-21 leads, so its data is ready before the others'.
SAT_PASSES = [
    ("Suomi-NPP", [("overnight", 1 * 60 + 30), ("afternoon", 13 * 60 + 30)]),
    ("NOAA-20",   [("overnight", 0 * 60 + 40), ("afternoon", 12 * 60 + 40)]),
    ("NOAA-21",   [("overnight", 0 * 60 + 0),  ("afternoon", 12 * 60 + 0)]),
]
FIRMS_SETTLE_H = 3.0
BAND_COLOR = {"High": "#ef4444", "Elevated": "#f59e0b", "Moderate": "#eab308", "Low": "#4ade80"}
CCAA = {1: "Andalucía", 2: "Aragón", 3: "Asturias", 4: "Baleares", 6: "Cantabria",
        7: "Castilla y León", 8: "Castilla-La Mancha", 9: "Cataluña", 10: "C. Valenciana",
        11: "Extremadura", 12: "Galicia", 13: "Madrid", 14: "Murcia", 15: "Navarra",
        16: "País Vasco", 17: "La Rioja"}

PRETTY = {
    "dist_to_fire": "proximity to active fire", "time_since_last_fire": "time since last fire",
    "burn_frequency_365d": "recent burn frequency", "fire_upwind_exposure": "upwind fire exposure",
    "dist_to_roads_stdev": "road-network variability", "dist_to_roads_mean": "distance to roads",
    "precip_sum_7d": "7-day rainfall", "precip_sum_30d": "30-day rainfall", "precip_sum_90d": "90-day rainfall",
    "days_since_rain": "days since rain", "spi_90d": "standardized precip (SPI-90)", "kbdi": "drought index (KBDI)",
    "total_precipitation_mean": "rainfall", "t2m_max": "max temperature", "t2m_mean": "mean temperature",
    "t2m_min": "min temperature", "t2m_range": "temperature range", "wind_speed_max": "max wind",
    "wind_speed_mean": "mean wind", "ffwi": "Fosberg fire-weather index", "ffwi_max": "Fosberg FWI",
    "fwi": "Fire Weather Index", "hdw": "hot-dry-windy index", "vpd_peak": "vapour-pressure deficit",
    "vpd_mean": "vapour-pressure deficit", "emc_peak": "fuel moisture (EMC)", "rh_min": "min humidity",
    "rh_mean": "mean humidity", "rh_max": "max humidity", "ndvi": "greenness (NDVI)",
    "ndvi_anomaly": "greenness anomaly", "lai": "leaf-area index", "lai_anomaly": "leaf-area anomaly",
    "fapar": "canopy light absorption", "fvc": "vegetation cover", "lst": "land-surface temperature",
    "swi_001": "soil moisture", "swi_005": "soil moisture", "swi_010": "soil moisture",
    "surface_pressure_mean": "surface pressure", "elevation": "elevation", "slope": "slope",
}


def pretty(f):
    return PRETTY.get(f, PRETTY.get(f.lower(), f.replace("_", " ")))


def risk_band(p):
    return "High" if p >= HIGH else "Elevated" if p >= ELEV else "Moderate" if p >= MOD else "Low"


# ---------- store reading (HF Dataset first, local fallback) ----------
@st.cache_resource(show_spinner=False)
def load_ccaa():
    return np.load(ASSETS)["ccaa"].astype(int)


def _unpack_grid(d, *, issue=None, target=None, source=None, mtime=None, prelim=False):
    def _j(key):
        try:
            return json.loads(str(d[key])) if key in d else ([] if key == "refreshed" else {})
        except Exception:
            return [] if key == "refreshed" else {}
    return dict(prob=d["prob"], regime=d["regime"], today_fire=d["today_fire"],
                issue=issue or str(d["issue_date"]), target=target or str(d["target_date"]),
                source=source or (str(d["source"]) if "source" in d else "?"),
                refreshed=_j("refreshed"),
                fetched_at=str(d["fetched_at"]) if "fetched_at" in d else None,
                drivers=_j("drivers"), prelim=prelim, mtime=mtime)


@st.cache_data(ttl=8, show_spinner=False)
def _hf_manifest():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(SERVING_REPO, "latest.json", repo_type="dataset")
    return json.loads(Path(p).read_text())


@st.cache_data(show_spinner=False)
def _grid_view(grid_path, version, issue, target, source, prelim):
    from huggingface_hub import hf_hub_download
    g = hf_hub_download(SERVING_REPO, grid_path, repo_type="dataset")
    return _unpack_grid(np.load(g, allow_pickle=True), issue=issue, target=target,
                        source=source, mtime=version, prelim=prelim)


def _hf_latest(man):
    return _grid_view(man["grid_path"], man.get("pushed_at"), man.get("issue"), man.get("target"),
                      man.get("source", "?"), man.get("prelim", False))


def _local_latest():
    grids = sorted((STORE / "grids").glob("*.npz"))
    if not grids:
        return None
    g = grids[-1]
    return _unpack_grid(np.load(g, allow_pickle=True), mtime=g.stat().st_mtime)


def latest_version():
    if not FORCE_LOCAL:
        try:
            m = _hf_manifest()
            if m:
                return m.get("pushed_at") or m.get("issue")
        except Exception:
            pass
    grids = sorted((STORE / "grids").glob("*.npz"))
    return grids[-1].stat().st_mtime if grids else None


def latest_store():
    if not FORCE_LOCAL:
        try:
            m = _hf_manifest()
            if m:
                return _hf_latest(m)
        except Exception:
            pass
    return _local_latest()


# ---------- map rendering ----------
@st.cache_resource(show_spinner=False)
def reproj_index():
    a = np.load(ASSETS)
    x = a["x"].astype(float); y = a["y"].astype(float)
    H, W = len(y), len(x); dx = (x[-1] - x[0]) / (W - 1); dy = (y[-1] - y[0]) / (H - 1)
    fwd = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    # ImageOverlay bounds at cell EDGES (centre ± half-cell).
    elon, elat = fwd.transform([x[0] - dx / 2, x[0] - dx / 2, x[-1] + dx / 2, x[-1] + dx / 2],
                               [y[0] - dy / 2, y[-1] + dy / 2, y[0] - dy / 2, y[-1] + dy / 2])
    bla0, bla1 = float(min(elat)), float(max(elat))   # south, north edges
    blo0, blo1 = float(min(elon)), float(max(elon))   # west, east edges
    # Display pixel CENTRES. Web-Mercator x is linear in lon, but y is NOT linear in lat — Leaflet stretches the
    # ImageOverlay linearly in Mercator screen-space, so a lat-even image is pushed NORTH (Mercator convex). Sample
    # each row at the latitude whose Mercator-y matches its screen position → the overlay seats exactly on the base.
    def my(d): return np.log(np.tan(np.pi / 4 + np.radians(d) / 2))
    def imy(v): return np.degrees(2 * np.arctan(np.exp(v)) - np.pi / 2)
    lon_c = blo0 + ((np.arange(W) + 0.5) / W) * (blo1 - blo0)
    lat_c = imy(my(bla1) + ((np.arange(H) + 0.5) / H) * (my(bla0) - my(bla1)))   # Mercator-even latitudes
    LON, LAT = np.meshgrid(lon_c, lat_c)
    inv = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    SX, SY = inv.transform(LON.ravel(), LAT.ravel())
    col = np.rint((np.asarray(SX) - x[0]) / dx).astype(np.int64)
    row = np.rint((np.asarray(SY) - y[0]) / dy).astype(np.int64)
    ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    idxmap = np.where(ok, row * W + col, -1).reshape(H, W)
    return idxmap, [[bla0, blo0], [bla1, blo1]]


def gather(a, idx):
    f = np.asarray(a, np.float32).ravel()
    return np.where(idx >= 0, f[np.clip(idx, 0, f.size - 1)], np.nan).astype(np.float32)


def risk_rgba(prob):
    """LOG-scaled, prevalence-anchored HEAT ramp (blackbody: brightest = hottest = most dangerous). Colour position
    is log(p): below RAMP_FLOOR → transparent (quiet days & noise floor stay dark); RAMP_FLOOR→RAMP_CAP ramps
    dark-red → orange → BRIGHT YELLOW, with a CONTRAST alpha (low risk dim/receding, high risk bright/popping).
    Absolute (day-comparable) but reveals the whole heavy-tailed field instead of clipping to black."""
    p = np.clip(np.nan_to_num(prob), 0, 1)
    with np.errstate(divide="ignore"):
        t = np.clip((np.log(np.maximum(p, 1e-12)) - _LF) / _LS, 0.0, 1.0)   # 0 at FLOOR, 1 at CAP (log position)
    DR = np.array([150, 22, 16.]); OR = np.array([255, 130, 34.]); YL = np.array([255, 246, 190.])  # DARK-red → orange → white-yellow
    lo = (t < 0.5)[..., None]; f = np.where(t < 0.5, t * 2, (t - 0.5) * 2)[..., None]
    rgb = np.where(lo, DR, OR) * (1 - f) + np.where(lo, OR, YL) * f
    a = np.where(p > RAMP_FLOOR, np.clip(90 + 165 * (t ** 1.4), 0, 255), 0.0)   # steeper, more-opaque top: low dark-red ~90 → top fully opaque 255
    rgba = np.zeros((*p.shape, 4), np.uint8)
    rgba[..., :3] = rgb.astype(np.uint8); rgba[..., 3] = a.astype(np.uint8)
    return rgba


def fire_rgba(mask):
    """Active-fire cells → crisp bright cyan (distinct from the yellow-red risk on a dark base)."""
    m = np.asarray(mask) > 0.5
    rgba = np.zeros((*m.shape, 4), np.uint8)
    rgba[..., 0] = np.where(m, 150, 0); rgba[..., 1] = np.where(m, 245, 0)
    rgba[..., 2] = np.where(m, 255, 0); rgba[..., 3] = np.where(m, 255, 0)
    return rgba


def _causes(feature_list):
    """Top ≤2 human-readable cause phrases from a regime's driver feature names (risk-raising direction)."""
    out = []
    for feats, phrase in CAUSE_MAP:
        if any(f in feature_list for f in feats) and phrase not in out:
            out.append(phrase)
    return out[:2]


def danger_clusters(prob, regime, ccaa, drivers, x, y, fwd):
    """WATERSHED the risk surface into hotspot-centred 'danger areas' — basins around local maxima with boundaries
    in the low-risk valleys, so adjacent hotspots stay SEPARATE (unlike a threshold+dilation blob). Per area →
    (lat, lon, radius_m, tooltip) with aggregate P(≥1 ignition)=1−Π(1−pᵢ) + 1-2 plain-language causes."""
    from scipy.ndimage import label, maximum_filter, gaussian_filter, watershed_ift
    land = regime > 0
    mask = (prob > CLUSTER_THR) & land
    if not mask.any():
        return []
    ps = gaussian_filter(prob.astype(np.float64), CLUSTER_SMOOTH_KM / CELL_KM)   # smooth → stable, meaningful peaks
    sep = max(3, round(CLUSTER_PEAKSEP_KM / CELL_KM))                        # peak-separation window in CELLS (res-scaled)
    lbl, n = label((maximum_filter(ps, size=sep) <= ps) & mask)             # each hotspot (local max) seeds a basin
    if n == 0:
        return []
    cost = (255 * (1.0 - np.clip(ps / max(ps.max(), 1e-9), 0, 1))).astype(np.uint8)   # high risk = low cost
    ws = watershed_ift(cost, lbl.astype(np.int32))                          # hotspot assignment over the whole plane
    blobs, nb = label(mask)                                                 # contiguous elevated regions → LOCAL extent
    # a danger area = one CONTIGUOUS blob split by hotspot basin: local (bounded by the blob) AND peak-separated.
    # (watershed_ift alone assigns scattered far cells to one basin → a peninsula-sized "cluster".)
    key = np.where(mask, blobs.astype(np.int64) * (int(n) + 1) + ws.astype(np.int64), 0)
    ig = _causes([d.get("feature") for d in (drivers.get("ignition") or [])])
    sp = _causes([d.get("feature") for d in (drivers.get("spread") or [])])
    dx = abs(x[1] - x[0])
    out = []
    for uid in np.unique(key):
        if uid == 0:
            continue
        comp, ncomp = label(key == uid)               # a uid can be spatially DISCONNECTED → one area per piece
        for ci in range(1, ncomp + 1):
            cells = comp == ci
            pc = prob[cells]
            pagg = 1.0 - float(np.prod(1.0 - np.clip(pc, 0, 0.99)))
            if pagg < CLUSTER_MIN_P:
                continue
            rr, cc = np.where(cells)
            j = int(np.argmax(pc))                    # PEAK cell → marker on the brightest spot, not a saddle/gap
            lon, lat = fwd.transform(float(x[cc[j]]), float(y[rr[j]]))
            rad = max((x[cc.max()] - x[cc.min()]) / 2, abs(y[rr.max()] - y[rr.min()]) / 2, dx) + dx / 2
            reg_spread = float((regime[cells] == 2).mean()) > 0.5
            rc = ccaa[cells]; rc = rc[rc > 0]
            region = CCAA.get(int(np.bincount(rc).argmax()), "") if rc.size else ""
            causes = list(sp if reg_spread else ig)
            if not reg_spread:                        # ignition areas are >6 km from fire (regime def) — not "near" it
                causes = [c for c in causes if c != "near active fire"]
            what = "fire spread" if reg_spread else "a new fire"
            tip = (f"{region + ' — ' if region else ''}up to ~{pagg * 100:.0f}% chance of {what} here tomorrow"
                   + (" · " + " · ".join(causes) if causes else ""))
            out.append((pagg, float(lat), float(lon), float(rad), tip))
    out.sort(reverse=True)                        # most dangerous first (no count cap — MIN_P governs)
    return [(lat, lon, rad, tip) for _, lat, lon, rad, tip in out]


def alpha_over(b, t):
    b = b.astype(np.float32) / 255; t = t.astype(np.float32) / 255
    ta, ba = t[..., 3:4], b[..., 3:4]; oa = ta + ba * (1 - ta)
    rgb = (t[..., :3] * ta + b[..., :3] * ba * (1 - ta)) / np.clip(oa, 1e-8, 1)
    out = np.zeros_like(b); out[..., :3] = rgb; out[..., 3:4] = oa
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def png(rgba):
    bf = io.BytesIO(); Image.fromarray(rgba, "RGBA").save(bf, "PNG"); return bf.getvalue()


def _legend(issue, target):
    return f'''<div style="position:absolute;bottom:18px;left:18px;z-index:9999;
      background:rgba(13,17,23,.80);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);
      border:1px solid rgba(255,255,255,.09);border-radius:11px;padding:11px 13px;
      font-family:'Inter',system-ui,-apple-system,sans-serif;color:#c9d4e0;">
      <div style="font-size:10.5px;letter-spacing:.07em;color:#8b98a8;margin-bottom:6px;text-transform:uppercase;">
        Risk for {target}</div>
      <div style="height:8px;width:156px;border-radius:4px;
        background:linear-gradient(90deg,rgba(245,68,28,.45),#f5441c 12%,#ff9626 55%,#fff6be);"></div>
      <div style="display:flex;justify-content:space-between;width:156px;font-size:10px;color:#7d8aa0;margin-top:3px;">
        <span>low</span><span>high</span></div>
      <div style="margin-top:8px;font-size:11px;color:#8b98a8;">
        <span style="color:#96f5ff;font-size:13px;">■</span> burning now · issued {issue}</div>
    </div>'''


@st.cache_data(show_spinner=False)
def build_map_html(prob_bytes, fire_bytes, regime_bytes, shape, issue, target, drivers_json, _idxmap, _bounds):
    """Dark base + red-tinted risk-glow raster, hover-able danger-area circles, and pulsing active-fire markers."""
    prob = np.frombuffer(prob_bytes, np.float32).reshape(shape)
    fire = np.frombuffer(fire_bytes, np.float32).reshape(shape)
    regime = np.frombuffer(regime_bytes, np.int8).reshape(shape)
    (la0, lo0), (la1, lo1) = _bounds
    m = folium.Map(location=[(la0 + la1) / 2, (lo0 + lo1) / 2], zoom_start=6, tiles="cartodbdark_matter",
                   zoom_control=True, control_scale=False)
    # risk glow (dark base shows through where risk≈0 — no land tint), softened; crisp cyan active fire on top
    pr = gather(prob, _idxmap); fr = np.nan_to_num(gather(fire, _idxmap))
    risk = np.array(Image.fromarray(risk_rgba(pr)).filter(ImageFilter.GaussianBlur(RISK_BLUR_KM / CELL_KM)))
    layer = alpha_over(risk, fire_rgba(fr > 0.5))                        # cyan fire crisp on top
    url = "data:image/png;base64," + base64.b64encode(png(layer)).decode()
    folium.raster_layers.ImageOverlay(image=url, bounds=[[la0, lo0], [la1, lo1]], opacity=0.96, zindex=2).add_to(m)
    a = np.load(ASSETS); x = a["x"].astype(float); y = a["y"].astype(float); ccaa = a["ccaa"].astype(int)
    fwd = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    # hover-able danger areas (aggregate probability + plain-language causes)
    for lat, lon, rad, tip in danger_clusters(prob, regime, ccaa, json.loads(drivers_json or "{}"), x, y, fwd):
        folium.Circle([lat, lon], radius=max(rad, 5000), color="#ffd9b0", weight=1, opacity=0.4,
                      fill=True, fill_opacity=0.04, tooltip=folium.Tooltip(tip, sticky=True)).add_to(m)
    return m._repr_html_()   # legend is rendered as a Streamlit strip below the map (reliable vs iframe overlay)


def regions_of(mask, ccaa):
    codes = np.unique(ccaa[(mask > 0.5) & (ccaa > 0)])
    return [CCAA.get(int(c), f"R{int(c)}") for c in codes]


def _to_dt(iso_or_mtime):
    """Parse an ISO string (naive → UTC) or an epoch float into a tz-aware UTC datetime (None on failure)."""
    if iso_or_mtime is None:
        return None
    try:
        if isinstance(iso_or_mtime, str):
            t = datetime.fromisoformat(iso_or_mtime)
            return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
        return datetime.fromtimestamp(float(iso_or_mtime), timezone.utc)
    except Exception:
        return None


def ago(iso_or_mtime):
    """Human 'time ago' at MINUTE granularity (no ticking seconds)."""
    t = _to_dt(iso_or_mtime)
    if t is None:
        return "—"
    s = (datetime.now(timezone.utc) - t).total_seconds()
    if s < 120:
        return "just now"
    if s < 5400:
        return f"{int(s // 60)}m ago"
    if s < 172800:
        return f"{int(s // 3600)}h ago"
    return f"{int(s // 86400)}d ago"


def _local_hm(dt_utc):
    """UTC datetime → 'HH:MM' in Europe/Madrid (falls back to UTC if zoneinfo/tzdata is unavailable)."""
    return (dt_utc.astimezone(MADRID) if MADRID else dt_utc).strftime("%H:%M")


def satellite_status(fetched, issue, now=None):
    """Time-expectation model of VIIRS active-fire coverage for the issue day — the state behind the LIVE/PARTIAL
    pill. Each of the four passes (2 birds × night/afternoon) becomes FETCHABLE at overpass + FIRMS_SETTLE_H:
      • 'in'    — settled AND the prediction was built at/after that → we hold it (✓)
      • 'miss'  — settled but the prediction predates it → we're BEHIND (⚠); the ONLY thing that makes us PARTIAL
      • 'later' — not fetchable yet (hasn't happened/settled) → expected, neutral (◦), no strike against us
    LIVE = the prediction already includes every pass fetchable as of *now* (the freshest that can exist — the
    normal all-day state). PARTIAL = a settled pass isn't in yet (e.g. the evening run didn't fire). Pure
    time-derived from the manifest's `fetched_at` + `issue`; no serve-side signal needed."""
    now = now or datetime.now(timezone.utc)
    fdt = _to_dt(fetched)
    day = _to_dt(issue) or now
    day0 = day.replace(hour=0, minute=0, second=0, microsecond=0)
    sats, any_miss, later_settles = [], False, []
    for name, passes in SAT_PASSES:
        rows = []
        for label, mod in passes:
            over = day0 + timedelta(minutes=mod)
            settle = over + timedelta(hours=FIRMS_SETTLE_H)
            if now < settle:
                state = "later"; later_settles.append(settle)
            elif fdt is None or fdt >= settle:      # None (replay/no stamp) → assume held, so history reads LIVE
                state = "in"
            else:
                state = "miss"; any_miss = True
            rows.append({"label": label, "time": _local_hm(settle), "state": state})   # show DATA-AVAILABILITY time
                                                                                        # (overpass + FIRMS lag), Madrid
        sats.append({"name": name, "passes": rows})
    npass = sum(len(p) for _, p in SAT_PASSES)               # 3 birds × 2 = 6
    if any_miss:
        cap = "newer passes have landed — refresh pending"
    elif later_settles:
        cap = f"latest passes in · next update ~{_local_hm(min(later_settles))}"
    else:
        cap = f"all {npass} passes in · final for today"
    return {"live": not any_miss, "status": "LIVE" if not any_miss else "PARTIAL", "sats": sats, "caption": cap}


SAT_TIP = ("The three VIIRS satellites — Suomi-NPP, NOAA-20, NOAA-21 — each scan Spain twice a day on staggered "
           "orbits, so together they catch fires the others would miss. We issue a first forecast from the earliest "
           "pass and refine it as each later pass arrives. Times shown are when the data from that pass becomes "
           "available in Madrid — about 3 hours after the overpass (NASA FIRMS processing lag).")


def sat_strip_html(sat, updated):
    """Render the satellite-coverage panel on ONE line: lead label · per-bird units (name + overnight/afternoon
    chips, each chip's time = when that pass's data is available) · LIVE/PARTIAL pill · a '?' help tooltip."""
    ic = {"in": "✓", "miss": "⚠", "later": "◦"}
    units = []
    for s in sat["sats"]:
        chips = "".join(
            f"<span class='cell {p['state']}'><span class='cell-l'>{p['label']}</span>"
            f"<span class='cell-i'>{ic[p['state']]}</span><span class='cell-t'>{p['time']}</span></span>"
            for p in s["passes"])
        units.append(f"<span class='sat-u'><span class='sat-name'>{s['name']}</span>{chips}</span>")
    cls = "on" if sat["live"] else "off"
    dot = "●" if sat["live"] else "◐"
    return (f"<div class='sat-panel'><span class='sat-lead'>🛰 VIIRS</span>"
            f"{''.join(units)}"
            f"<span class='sat-pill {cls}'>{dot} {sat['status']}<small>{updated}</small></span>"
            f"<span class='sat-info' tabindex='0' title='{SAT_TIP}'>?<span class='sat-tip'>{SAT_TIP}</span></span>"
            f"</div>")


# ---------------------------- UI ----------------------------
st.set_page_config(page_title="Fire Guard Control Center", page_icon="🛡️", layout="wide")
st.markdown("""<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
  html, body, [class*="css"] { font-family:'Inter',system-ui,-apple-system,sans-serif; }
  .stApp { background:#0d1117; }
  [data-testid="stHeader"], header[data-testid="stHeader"] { display:none !important; height:0 !important; }
  [data-testid="stToolbar"] { display:none !important; }
  .block-container { padding-top:1.7rem; padding-bottom:1rem; max-width:1400px; }
  .fg-legend { display:flex; align-items:center; gap:14px; margin-top:8px; font-size:.8rem; color:#8b98a8; flex-wrap:wrap; }
  .fg-legend b { color:#c9d4e0; }
  iframe { border-radius:14px; }
  .fg-title { font-size:1.7rem; font-weight:800; color:#eef2f7; letter-spacing:-.01em; line-height:1.05; }
  .fg-sub { font-size:.82rem; color:#7d8aa0; margin-top:3px; letter-spacing:.03em; }
  .fg-status { text-align:right; font-size:1.05rem; font-weight:700; line-height:1.1; }
  .fg-status-age { display:block; font-size:.8rem; color:#7d8aa0; font-weight:400; margin-top:3px; }
  /* satellite-coverage panel — ONE line: lead · per-bird units · pill · ? help */
  .sat-panel { display:flex; align-items:center; gap:8px 18px; flex-wrap:wrap; margin:14px 0 2px;
    background:rgba(22,27,34,.55); backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px);
    border:1px solid rgba(255,255,255,.06); border-radius:14px; padding:10px 16px; }
  .sat-lead { display:flex; align-items:center; gap:7px; font-size:.68rem; font-weight:700;
    text-transform:uppercase; letter-spacing:.07em; color:#8b98a8; }
  .sat-u { display:flex; align-items:center; gap:6px; }
  .sat-name { font-size:.8rem; font-weight:700; color:#c9d4e0; }
  .cell { display:inline-flex; align-items:center; gap:5px; font-size:.77rem; padding:3px 9px; border-radius:999px;
    border:1px solid rgba(255,255,255,.07); background:rgba(255,255,255,.02); }
  .cell-l { font-size:.58rem; text-transform:uppercase; letter-spacing:.03em; color:#7d8aa0; }
  .cell-i { font-weight:800; font-size:.8rem; }
  .cell-t { color:#c9d4e0; font-weight:600; font-variant-numeric:tabular-nums; }
  .cell.in    { border-color:rgba(74,222,128,.28); background:rgba(74,222,128,.06); }
  .cell.in .cell-i { color:#4ade80; }
  .cell.miss  { border-color:rgba(240,160,32,.34); background:rgba(240,160,32,.08); }
  .cell.miss .cell-i, .cell.miss .cell-t { color:#f0a020; }
  .cell.later { opacity:.5; }
  .cell.later .cell-i { color:#7d8aa0; }
  .sat-pill { margin-left:auto; display:flex; align-items:center; gap:9px; white-space:nowrap;
    font-size:1rem; font-weight:800; letter-spacing:.02em; padding:6px 15px; border-radius:999px; }
  .sat-pill small { font-size:.72rem; font-weight:500; color:#8b98a8; letter-spacing:0; }
  .sat-pill.on  { color:#4ade80; background:rgba(74,222,128,.10); border:1px solid rgba(74,222,128,.30); }
  .sat-pill.off { color:#f0a020; background:rgba(240,160,32,.10); border:1px solid rgba(240,160,32,.30); }
  .sat-info { position:relative; display:inline-flex; align-items:center; justify-content:center; flex-shrink:0;
    width:20px; height:20px; border-radius:50%; border:1px solid rgba(255,255,255,.2); color:#8b98a8;
    font-size:.72rem; font-weight:700; cursor:help; }
  .sat-info:hover, .sat-info:focus { color:#eef2f7; border-color:rgba(255,255,255,.45); outline:none; }
  .sat-tip { position:absolute; bottom:calc(100% + 10px); right:0; width:320px; padding:12px 14px;
    background:#161b22; border:1px solid rgba(255,255,255,.13); border-radius:11px; text-align:left;
    font-size:.77rem; font-weight:400; line-height:1.5; color:#c9d4e0; letter-spacing:0; text-transform:none;
    box-shadow:0 10px 32px rgba(0,0,0,.55); opacity:0; visibility:hidden; transform:translateY(4px);
    transition:opacity .12s ease, transform .12s ease; z-index:1000; pointer-events:none; }
  .sat-info:hover .sat-tip, .sat-info:focus .sat-tip { opacity:1; visibility:visible; transform:translateY(0); }
  .src-k { font-size:.68rem; text-transform:uppercase; letter-spacing:.06em; color:#6f7d90; }
  .src-v { font-size:.9rem; color:#c9d4e0; font-weight:600; }
  .card { background:rgba(22,27,34,.55); backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px);
          border:1px solid rgba(255,255,255,.06); border-radius:16px; padding:18px 22px; height:100%; }
  .card-h { font-size:.74rem; text-transform:uppercase; letter-spacing:.09em; color:#8b98a8; margin-bottom:10px; }
  .metric { font-size:2.9rem; font-weight:800; line-height:1; color:#eef2f7; }
  .metric-band { font-size:2.2rem; font-weight:800; line-height:1; }
  .metric-sub { font-size:.92rem; color:#9fb3c8; margin-top:6px; }
  .rowline { font-size:.9rem; color:#c9d4e0; margin-top:8px; }
  .muted { color:#7d8aa0; }
  .drivers { font-size:.86rem; color:#9fb3c8; margin-top:12px; padding-top:11px; border-top:1px solid rgba(255,255,255,.06); }
  .foot { color:#5f6b7d; font-size:.78rem; }
</style>""", unsafe_allow_html=True)

ccaa = load_ccaa()
idxmap, bounds = reproj_index()
S = latest_store()
if not S:
    st.markdown("<div class='fg-title'>Fire Guard Control Center</div>", unsafe_allow_html=True)
    st.warning("No prediction published yet — the engine hasn't produced one. Check back shortly.")
    st.stop()
st.session_state["seen_version"] = latest_version()

prob, regime, fire = S["prob"], S["regime"], S["today_fire"]
land = regime > 0
fire = np.where(land, fire, 0.0)   # FIRMS covers the whole Iberian bbox; only show fire within our (Spain-only)
                                   # prediction coverage — Portugal has fire but no forecast (adding it = backlog)
n_fire = int((fire > 0.5).sum())


# ---------- title + satellite-coverage strip (60 s fragment: re-derives pass state & LIVE/PARTIAL as time moves) ----------
@st.fragment(run_every=60)
def topbar():
    st.markdown("<div class='fg-title'>Fire Guard <span style='color:#9fb3c8;font-weight:700'>Control Center</span></div>"
                "<div class='fg-sub'>next-day wildfire risk · Spain</div>", unsafe_allow_html=True)
    sat = satellite_status(S["fetched_at"] or S["mtime"], S["issue"])
    st.markdown(sat_strip_html(sat, ago(S["fetched_at"] or S["mtime"])), unsafe_allow_html=True)


topbar()

# ---------- MAP: full-width hero ----------
html = build_map_html(np.ascontiguousarray(prob, np.float32).tobytes(),
                      np.ascontiguousarray(fire, np.float32).tobytes(),
                      np.ascontiguousarray(regime, np.int8).tobytes(), prob.shape,
                      S["issue"], S["target"], json.dumps(S.get("drivers") or {}), idxmap, bounds)
components.html(html, height=620)
st.markdown(
    f"<div class='fg-legend'><span>Risk for <b>{S['target']}</b></span>"
    "<span style='height:8px;width:120px;border-radius:4px;display:inline-block;"
    "background:linear-gradient(90deg,rgba(245,68,28,.45),#f5441c 12%,#ff9626 55%,#fff6be);'></span>"
    "<span>low → high</span>"
    "<span><span style='color:#96f5ff;font-size:13px;'>■</span> burning now</span>"
    f"<span>· hover a circle for the danger-area chance &amp; causes · issued {S['issue']}</span></div>",
    unsafe_allow_html=True)


# ---------- store-watcher: full rerun only when a NEW prediction lands ----------
@st.fragment(run_every=10)
def store_watcher():
    v = latest_version()
    if v is not None and v != st.session_state.get("seen_version"):
        st.cache_data.clear()
        st.rerun()


store_watcher()

# ---------- sources & freshness (compact, below the map) ----------
asof = ago(S["fetched_at"] or S["mtime"])
srcs = [("Weather", "Open-Meteo ERA5"), ("Active fire", f"FIRMS VIIRS · {n_fire} cells"),
        ("Dryness", "live KBDI / SPI"), ("Vegetation", "MODIS composite")]
scols = st.columns([1, 1, 1, 1, 1])
for col, (k, v) in zip(scols, srcs):
    col.markdown(f"<div class='src-k'>{k}</div><div class='src-v'>{v}</div>", unsafe_allow_html=True)
scols[-1].markdown(f"<div class='src-k' style='text-align:right'>freshness</div>"
                   f"<div class='src-v' style='text-align:right'>updated {asof}</div>", unsafe_allow_html=True)
st.write("")

# ---------- NOW | TOMORROW ----------
cN, cT = st.columns(2)

with cN:
    now_regions = regions_of(fire, ccaa)
    if n_fire:
        reg = " · ".join(now_regions[:6]) + (" …" if len(now_regions) > 6 else "")
        body = (f"<div class='metric' style='color:#96f5ff'>{n_fire}</div>"
                f"<div class='metric-sub'>active-fire cells · ≤ {n_fire * CELL_KM2:,.0f} km²</div>"
                f"<div class='rowline muted'>{reg}</div>")
    else:
        body = "<div class='metric' style='color:#4ade80'>0</div><div class='metric-sub'>no active-fire detections</div>"
    st.markdown(f"<div class='card'><div class='card-h'>Now — burning</div>{body}</div>", unsafe_allow_html=True)

with cT:
    pk = float(prob[land].max()) if land.any() else 0.0
    band = risk_band(pk); bcol = BAND_COLOR[band]
    rmax = {n: float(prob[(ccaa == c) & land].max()) for c, n in CCAA.items() if ((ccaa == c) & land).any()}
    tops = sorted(rmax.items(), key=lambda kv: -kv[1])[:4]
    top_line = " · ".join(f"{n} {v:.0%}" for n, v in tops if v >= MOD) or "no elevated regions"
    drv = S.get("drivers") or {}
    ig = [pretty(d["feature"]) for d in drv.get("ignition", [])][:3]
    sp = [pretty(d["feature"]) for d in drv.get("spread", [])][:3]
    drv_html = ""
    if ig:
        drv_html += f"<div class='drivers'><span class='muted'>ignition drivers</span> · {' · '.join(ig)}</div>"
    if sp:
        drv_html += f"<div class='drivers' style='border-top:none;padding-top:2px'><span class='muted'>spread drivers</span> · {' · '.join(sp)}</div>"
    st.markdown(
        f"<div class='card'><div class='card-h'>Tomorrow · {S['target']}</div>"
        f"<div class='metric-band' style='color:{bcol}'>{band}</div>"
        f"<div class='metric-sub'>peak cell risk {pk:.0%}</div>"
        f"<div class='rowline'>top regions · {top_line}</div>{drv_html}</div>", unsafe_allow_html=True)

st.write("")
st.markdown("<div class='foot'>Calibrated gradient-boosted trees on the Fire Guard Datacube v2 · three-satellite "
            "VIIRS active fire (Suomi-NPP + NOAA-20 + NOAA-21), refreshed as each pass lands · "
            "base © CARTO / OpenStreetMap.</div>", unsafe_allow_html=True)
