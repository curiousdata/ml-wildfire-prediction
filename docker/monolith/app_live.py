"""Fire Guard — wildfire control center (operational live view).

The main application: a live forest-fire control center for Spain. Not a date-picker demo — it shows the
LATEST prediction the engine produced and is built to feel operational.

Architecture: the engine is `scripts/daily_job.py --mode live` (fetch Open-Meteo + FIRMS → assemble slice →
GBT → write date-partitioned store under data/serving_store/). THIS app READS the latest stored prediction
(fast, no in-app rate-limited fetch) and renders it. Liveness is handled by st.fragment:
  • a 1s fragment redraws the clock + per-feed "age" counters WITHOUT re-rendering the heavy map;
  • a 10s fragment watches the store and triggers a full rerun only when a NEW prediction lands
    (so no manual page reload — and no repeated st_folium re-render, which was crashing the process).

Panels: a fresh daily-satellite map (NASA GIBS true-color, dated) with today's fires + tomorrow's risk;
LIVE FEEDS strip (each feed + its own freshness timer); NOW burning; TOMORROW risk; TOP DRIVERS per regime.

Run: streamlit run docker/monolith/app_live.py
"""
from __future__ import annotations
import base64
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import folium  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402
import xarray as xr  # noqa: E402

STORE = ROOT / "data" / "serving_store"
CUBE = ROOT / "data" / "gold" / "FireGuard_coarse4.zarr"   # FGDC v2 working cube (v1 IberFire cube deleted)
CELL_HA = 4.0 * 4.0 * 100   # 4 km cell = 16 km² = 1600 ha
CELL_KM2 = 16.0
HIGH, ELEV, MOD = 0.50, 0.20, 0.05    # calibrated-probability risk tiers
CCAA = {1: "Andalucía", 2: "Aragón", 3: "Asturias", 4: "Baleares", 6: "Cantabria",
        7: "Castilla y León", 8: "Castilla-La Mancha", 9: "Cataluña", 10: "C. Valenciana",
        11: "Extremadura", 12: "Galicia", 13: "Madrid", 14: "Murcia", 15: "Navarra",
        16: "País Vasco", 17: "La Rioja"}

# human-readable feature names for the "top drivers" panel
PRETTY = {
    "dist_to_fire": "Proximity to active fire", "time_since_last_fire": "Time since last fire",
    "burn_frequency_365d": "Recent burn frequency", "fire_upwind_exposure": "Upwind fire exposure",
    "dist_to_roads_stdev": "Road-network variability", "dist_to_roads_mean": "Distance to roads",
    "precip_sum_7d": "7-day rainfall", "precip_sum_30d": "30-day rainfall", "precip_sum_90d": "90-day rainfall",
    "days_since_rain": "Days since rain", "spi_90d": "Standardized precip (SPI-90)", "kbdi": "Drought index (KBDI)",
    "total_precipitation_mean": "Rainfall", "t2m_max": "Max temperature", "t2m_mean": "Mean temperature",
    "t2m_min": "Min temperature", "t2m_range": "Temperature range", "wind_speed_max": "Max wind",
    "wind_speed_mean": "Mean wind", "ffwi": "Fosberg fire-weather index", "ffwi_max": "Fosberg FWI",
    "fwi": "Fire Weather Index", "hdw": "Hot-Dry-Windy index", "vpd_peak": "Vapour-pressure deficit (peak)",
    "vpd_mean": "Vapour-pressure deficit", "emc_peak": "Fuel moisture (EMC)", "rh_min": "Min humidity",
    "rh_mean": "Mean humidity", "rh_max": "Max humidity", "ndvi": "Greenness (NDVI)",
    "ndvi_anomaly": "Greenness anomaly", "lai": "Leaf-area index", "lai_anomaly": "Leaf-area anomaly",
    "fapar": "Canopy light absorption", "fvc": "Vegetation cover", "lst": "Land-surface temperature",
    "swi_001": "Soil moisture (0–1cm)", "swi_005": "Soil moisture", "swi_010": "Soil moisture",
    "surface_pressure_mean": "Surface pressure", "elevation": "Elevation", "slope": "Slope",
}


def pretty(f):
    return PRETTY.get(f, PRETTY.get(f.lower(), f.replace("_", " ").capitalize()))


def risk_band(p):
    return ("High", "🔴") if p >= HIGH else ("Elevated", "🟠") if p >= ELEV else \
           ("Moderate", "🟡") if p >= MOD else ("Low", "🟢")


@st.cache_resource(show_spinner=False)
def load_ccaa():
    return np.rint(np.nan_to_num(xr.open_zarr(str(CUBE), consolidated=True)["AutonomousCommunities"].values)).astype(int)


@st.cache_data(show_spinner=False)
def load_importance():
    p = ROOT / "models" / "gbt_fireguard.importance.json"   # optional; per-day occlusion drivers (in each grid) are primary
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {r: [t["feature"] for t in v["top"]] for r, v in d.get("regimes", {}).items()}


def latest_store():
    grids = sorted((STORE / "grids").glob("*.npz"))
    if not grids:
        return None
    g = grids[-1]
    d = np.load(g, allow_pickle=True)
    refreshed = []
    if "refreshed" in d:
        try:
            refreshed = json.loads(str(d["refreshed"]))
        except Exception:
            refreshed = []
    fetched = str(d["fetched_at"]) if "fetched_at" in d else None
    drivers = {}
    if "drivers" in d:
        try:
            drivers = json.loads(str(d["drivers"]))
        except Exception:
            drivers = {}
    return dict(prob=d["prob"], regime=d["regime"], today_fire=d["today_fire"],
               issue=str(d["issue_date"]), target=str(d["target_date"]),
               source=str(d["source"]) if "source" in d else "?", refreshed=refreshed,
               fetched_at=fetched, drivers=drivers, mtime=g.stat().st_mtime)


# ---------- map rendering ----------
@st.cache_resource(show_spinner=False)
def reproj_index():
    z = xr.open_zarr(str(CUBE), consolidated=True)
    x = z["x"].values.astype(float); y = z["y"].values.astype(float)
    H, W = len(y), len(x); dx = (x[-1] - x[0]) / (W - 1); dy = (y[-1] - y[0]) / (H - 1)
    fwd = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    clon, clat = fwd.transform([x[0], x[0], x[-1], x[-1]], [y[0], y[-1], y[0], y[-1]])
    lo0, lo1, la0, la1 = min(clon), max(clon), min(clat), max(clat)
    LON, LAT = np.meshgrid(np.linspace(lo0, lo1, W), np.linspace(la1, la0, H))
    inv = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    SX, SY = inv.transform(LON.ravel(), LAT.ravel())
    col = np.rint((np.asarray(SX) - x[0]) / dx).astype(np.int64)
    row = np.rint((np.asarray(SY) - y[0]) / dy).astype(np.int64)
    ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    return np.where(ok, row * W + col, -1).reshape(H, W), [[float(la0), float(lo0)], [float(la1), float(lo1)]]


def gather(a, idx):
    f = np.asarray(a, np.float32).ravel()
    return np.where(idx >= 0, f[np.clip(idx, 0, f.size - 1)], np.nan).astype(np.float32)


def risk_rgba(prob, amax=210, eps=1e-4):
    p = np.clip(np.nan_to_num(prob), 0, 1)
    d = float(np.percentile(p[p > 0], 99)) if (p > 0).any() else 0.0
    ps = np.clip(p / d, 0, 1) if d > 0 else p
    rgba = np.zeros((*p.shape, 4), np.uint8)
    rgba[..., 0] = 255; rgba[..., 1] = ((1 - ps) * 200).astype(np.uint8)
    rgba[..., 3] = np.where(p > eps, np.clip(ps * amax, 25, amax), 0).astype(np.uint8)
    return rgba


def fire_rgba(mask):
    m = np.asarray(mask) > 0.5; rgba = np.zeros((*m.shape, 4), np.uint8)
    rgba[..., 0] = np.where(m, 255, 0); rgba[..., 2] = np.where(m, 255, 0)
    rgba[..., 3] = np.where(m, 255, 0).astype(np.uint8)
    return rgba


def alpha_over(b, t):
    b = b.astype(np.float32) / 255; t = t.astype(np.float32) / 255
    ta, ba = t[..., 3:4], b[..., 3:4]; oa = ta + ba * (1 - ta)
    rgb = (t[..., :3] * ta + b[..., :3] * ba * (1 - ta)) / np.clip(oa, 1e-8, 1)
    out = np.zeros_like(b); out[..., :3] = rgb; out[..., 3:4] = oa
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def png(rgba):
    bf = io.BytesIO(); Image.fromarray(rgba, "RGBA").save(bf, "PNG"); return bf.getvalue()


@st.cache_data(show_spinner=False)
def build_map_html(prob_bytes, fire_bytes, shape, gibs_date, _idxmap, _bounds):
    """Build the folium map once and cache its HTML (keyed by the prediction arrays + date)."""
    prob = np.frombuffer(prob_bytes, np.float32).reshape(shape)
    fire = np.frombuffer(fire_bytes, np.float32).reshape(shape)
    (la0, lo0), (la1, lo1) = _bounds
    m = folium.Map(location=[(la0 + la1) / 2, (lo0 + lo1) / 2], zoom_start=6, tiles=None,
                   control_scale=True)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri", name="Satellite (static)").add_to(m)
    gibs = (f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_SNPP_CorrectedReflectance_TrueColor/"
            f"default/{gibs_date}/GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.jpg")
    folium.TileLayer(gibs, attr="NASA GIBS / VIIRS", name=f"🛰 Fresh imagery {gibs_date}", overlay=False).add_to(m)
    pr = gather(prob, _idxmap); fr = np.nan_to_num(gather(fire, _idxmap))
    layer = alpha_over(risk_rgba(pr), fire_rgba(fr > 0.5))
    url = "data:image/png;base64," + base64.b64encode(png(layer)).decode()
    folium.raster_layers.ImageOverlay(image=url, bounds=[[la0, lo0], [la1, lo1]], opacity=0.85, zindex=2).add_to(m)
    return m._repr_html_()


def regions_of(mask, ccaa):
    codes = np.unique(ccaa[(mask > 0.5) & (ccaa > 0)])
    return [CCAA.get(int(c), f"R{int(c)}") for c in codes]


def ago(iso_or_mtime):
    """Human 'time ago' from an ISO-UTC string or a unix mtime float."""
    if iso_or_mtime is None:
        return "—"
    try:
        if isinstance(iso_or_mtime, str):
            t = datetime.fromisoformat(iso_or_mtime)
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
        else:
            t = datetime.fromtimestamp(iso_or_mtime, timezone.utc)
    except Exception:
        return "—"
    s = (datetime.now(timezone.utc) - t).total_seconds()
    if s < 90:
        return f"{int(s)}s ago"
    if s < 5400:
        return f"{int(s // 60)}m ago"
    if s < 172800:
        return f"{int(s // 3600)}h ago"
    return f"{int(s // 86400)}d ago"


# ---------------------------- UI ----------------------------
st.set_page_config(page_title="Fire Guard — wildfire control center", page_icon="🛡️", layout="wide")
st.markdown("""<style>
  .feed-card{background:#11151c;border:1px solid #2a3340;border-radius:12px;padding:12px 16px;height:100%;}
  .feed-name{font-size:0.95rem;color:#9fb3c8;font-weight:600;letter-spacing:.02em;}
  .feed-val{font-size:1.45rem;font-weight:700;color:#e8eef5;line-height:1.25;margin:2px 0;}
  .feed-age{font-size:0.9rem;color:#7d8aa0;}
  .big-num{font-size:2.6rem;font-weight:800;line-height:1;}
  .sub{font-size:0.95rem;color:#9fb3c8;}
</style>""", unsafe_allow_html=True)

ccaa = load_ccaa()
idxmap, bounds = reproj_index()
importance = load_importance()
S = latest_store()
if not S:
    st.markdown("## 🛡️ Fire Guard")
    st.warning("No prediction in the store yet. Run:  `python scripts/daily_job.py --mode live`  to populate it.")
    st.stop()
st.session_state["seen_mtime"] = S["mtime"]   # baseline for the store-watcher fragment

prob, regime, fire = S["prob"], S["regime"], S["today_fire"]
land = regime > 0
rfr = S["refreshed"]
weather_live = any(k in rfr for k in ("t2m_mean", "t2m_max", "t2m_min"))
dryness_live = "antecedent-dryness" in rfr
firms_disp = "display:FIRMS" in rfr
effis_live = "fire:EFFIS-live" in rfr
effis_persist = next((str(x).split(":")[-1] for x in rfr if str(x).startswith("fire:EFFIS-persist")), None)
effis_none = any(str(x).startswith("fire:none") for x in rfr)
# legacy stores: "fire:EFFIS-down→warm-cube" was the old (phantom) seasonal warm-start
effis_legacy_warm = any(str(x).startswith("fire:EFFIS-down") for x in rfr)


# ---------- header: branding + LIVE + clock (auto-ticking) ----------
@st.fragment(run_every=1)
def header_live():
    c1, c2, c3 = st.columns([3, 2, 2])
    c1.markdown("## 🛡️ Fire Guard\n###### wildfire control center · Spain")
    c2.markdown(f"<div class='feed-name'>STATUS</div><div class='feed-val' style='color:#ff4b4b'>● LIVE</div>"
                f"<div class='feed-age'>data updated {ago(S['fetched_at'] or S['mtime'])}</div>",
                unsafe_allow_html=True)
    now = datetime.now()
    c3.markdown(f"<div class='feed-name'>NOW (LOCAL)</div>"
                f"<div class='feed-val' style='font-variant-numeric:tabular-nums'>{now.strftime('%H:%M:%S')}</div>"
                f"<div class='feed-age'>{now.strftime('%A %d %b %Y')}</div>", unsafe_allow_html=True)


header_live()


# ---------- LIVE FEEDS strip: each feed + its own freshness timer (auto-ticking) ----------
@st.fragment(run_every=1)
def live_feeds():
    n_fire = int((fire > 0.5).sum())
    feeds = [
        ("🛰 Satellite", "NASA GIBS true-color", f"imagery {S['issue']} · {ago(S['issue'] + 'T12:00:00+00:00')}"),
        ("🌡 Weather", "Open-Meteo (ERA5)" if weather_live else "warm-started",
         (f"as of {ago(S['fetched_at'] or S['mtime'])}") if weather_live else "seasonal"),
        ("🔥 Active fire", f"FIRMS · {n_fire} cells" if firms_disp else f"{n_fire} cells",
         f"as of {ago(S['fetched_at'] or S['mtime'])}" if firms_disp else "—"),
        ("🟥 Burned area",
         "EFFIS (live)" if effis_live else f"EFFIS (persisted)" if effis_persist else
         "no live fire" if effis_none else "seasonal (legacy)",
         f"as of {S['issue']}" if effis_live else f"as of {effis_persist}" if effis_persist else
         "all-ignition" if effis_none else "stale"),
        ("🌿 Dryness/veg", "Open-Meteo 90-day" if dryness_live else "warm-started",
         "live antecedents" if dryness_live else "seasonal"),
    ]
    cols = st.columns(len(feeds))
    for col, (name, val, age) in zip(cols, feeds):
        col.markdown(f"<div class='feed-card'><div class='feed-name'>{name}</div>"
                     f"<div class='feed-val'>{val}</div><div class='feed-age'>{age}</div></div>",
                     unsafe_allow_html=True)


live_feeds()
st.write("")


# ---------- store-watcher: full rerun only when a NEW prediction lands (no manual reload) ----------
@st.fragment(run_every=10)
def store_watcher():
    grids = sorted((STORE / "grids").glob("*.npz"))
    if grids and grids[-1].stat().st_mtime != st.session_state.get("seen_mtime"):
        st.cache_data.clear()
        st.rerun()  # full rerun: reload store + rebuild map


store_watcher()

# ---------- degraded-input banner (never show phantom/stale risk uncaveated) ----------
degraded = []
if effis_persist:
    degraded.append(f"**Model fire is persisted from {effis_persist}** (live EFFIS unavailable) — fire geography "
                    "is the most recent KNOWN state, not necessarily today's. Spread-risk assumes persistence.")
elif effis_none:
    degraded.append("**No live burned-area** (EFFIS down, no cache yet) — the model assumes **no known fire** "
                    "(all-ignition); any active fire on the map (FIRMS) is not yet fed to spread-risk.")
elif effis_legacy_warm:
    degraded.append("**Model fire is seasonally warm-started (legacy)** — may import prior-year fire locations "
                    "and fabricate spread-risk. Re-run the engine to use the persistence cascade.")
if not dryness_live:
    degraded.append("**Antecedent dryness is warm-started** (live daily precip unavailable for the most recent "
                    "days) — drought/rainfall features are seasonal, not current.")
if degraded:
    st.warning("⚠️ **Degraded inputs — read risk with care:**\n\n" + "\n".join(f"- {x}" for x in degraded))


# ---------- map (rendered ONCE; cached; not redrawn by the 1s fragments) ----------
left, right = st.columns([3, 2])
with left:
    st.caption(f"Issued **{S['issue']}** → risk for **{S['target']}** · 🟪 burning now · 🟧 predicted risk "
               f"(intensity ∝ calibrated probability) · 🛰 fresh VIIRS true-color base")
    html = build_map_html(np.ascontiguousarray(prob, np.float32).tobytes(),
                          np.ascontiguousarray(fire, np.float32).tobytes(), prob.shape,
                          S["issue"], idxmap, bounds)
    components.html(html, height=600)

with right:
    # ---- NOW burning ----
    n_fire = int((fire > 0.5).sum())
    now_regions = regions_of(fire, ccaa)
    st.subheader("🔥 Burning now")
    if n_fire:
        st.markdown(f"<span class='big-num' style='color:#c026d3'>{n_fire}</span> "
                    f"<span class='sub'>cells with active fire</span>", unsafe_allow_html=True)
        st.markdown(f"<span class='sub'>footprint ≤ {n_fire * CELL_KM2:,.0f} km² "
                    f"(~{n_fire * CELL_HA:,.0f} ha) · {', '.join(now_regions)}</span>", unsafe_allow_html=True)
        st.caption("FIRMS flags each 4 km cell that *contains* an active-fire detection — an upper bound on "
                   "footprint, not a measured burned-area total.")
    else:
        st.success("No active-fire detections.")

    # ---- TOMORROW ----
    st.subheader(f"📈 Tomorrow ({S['target']})")
    pk = float(prob[land].max()) if land.any() else 0.0
    band, dot = risk_band(pk)
    exp_area = float(prob[land].sum()) * CELL_HA
    n_contrib = int((prob[land] > 1e-4).sum())
    st.markdown(f"<span class='big-num'>{dot} {band}</span> "
                f"<span class='sub'>peak cell risk {pk:.1%}</span>", unsafe_allow_html=True)
    st.markdown(f"**Expected new-fire area ≈ {exp_area:,.0f} ha** "
                f"<span class='sub'>(Σ calibrated risk × cell area, over {n_contrib:,} cells)</span>",
                unsafe_allow_html=True)
    rmax = {n: float(prob[(ccaa == c) & land].max()) for c, n in CCAA.items() if ((ccaa == c) & land).any()}
    high = [n for n, v in rmax.items() if v >= HIGH]
    elev = [n for n, v in rmax.items() if ELEV <= v < HIGH]
    st.markdown(f"🔴 **HIGH (≥{HIGH:.0%})** in {len(high)}: " + (", ".join(high) if high else "_none_"))
    st.markdown(f"🟠 **ELEVATED (≥{ELEV:.0%})** in {len(elev)}: " + (", ".join(elev) if elev else "_none_"))
    # reconcile the two numbers (the "9k ha but 0 elevated — what's burning?" question)
    if not high and not elev:
        st.info(f"**How to read this:** “Burning now” counts fire cells observed *today*. “Expected area” is a "
                f"*probabilistic sum for tomorrow* — not a count. Today the peak cell risk is only **{pk:.1%}**, so "
                f"risk is **diffuse and low**: ~{exp_area:,.0f} ha of expected area spread thinly across {n_contrib:,} "
                f"cells, with **no** single area reaching the Elevated tier (≥{ELEV:.0%}). Nothing is predicted to "
                f"concentrate into a hotspot.")
    st.markdown("**Top regions by risk tomorrow:**")
    for name, v in sorted(rmax.items(), key=lambda kv: -kv[1])[:5]:
        b, d = risk_band(v)
        st.markdown(f"&nbsp;&nbsp;{d} {name} — {v:.1%}")

    # ---- BIGGEST FACTORS driving TOMORROW's prediction (per-day attribution) ----
    st.subheader("🧭 Biggest factors driving tomorrow")
    drv = S.get("drivers") or {}
    if drv:
        ig = [d["feature"] for d in drv.get("ignition", [])][:4]
        sp = [d["feature"] for d in drv.get("spread", [])][:4]
        if ig:
            st.markdown("**🔥 New ignition:** " + " · ".join(pretty(f) for f in ig))
        if sp:
            st.markdown("**🌬 Spread (existing fire):** " + " · ".join(pretty(f) for f in sp))
        if not ig and not sp:
            st.caption("Risk is flat today — no feature is pushing the highest-risk cells up meaningfully.")
        st.caption("What actually pushed *this day's* highest-risk cells up: each feature zeroed to its "
                   "baseline → mean predicted-risk drop (occlusion attribution, per regime).")
    elif importance:  # fallback: model-wide general drivers
        ig = importance.get("ignition", [])[:4]; sp = importance.get("spread", [])[:4]
        if ig:
            st.markdown("**🔥 New ignition:** " + " · ".join(pretty(f) for f in ig))
        if sp:
            st.markdown("**🌬 Spread (existing fire):** " + " · ".join(pretty(f) for f in sp))
        st.caption("⚠️ Showing model-wide *general* drivers (this prediction predates per-day attribution — "
                   "re-run the engine for today-specific factors).")
    else:
        st.caption("Driver attribution not available yet.")

# ---- engine control (scheduling is the real path; this is on-demand) ----
with st.expander("⚙️ Run engine now (on-demand live fetch)"):
    st.caption("Normally the engine runs on a schedule and this app auto-updates when a new prediction lands. "
               "Use this only to force a fresh fetch (Open-Meteo + FIRMS → GBT, ~30–60s).")
    if st.button("↻ Run live engine"):
        with st.spinner("Running live engine…"):
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "daily_job.py"), "--mode", "live", "--overwrite"],
                               capture_output=True, text=True, cwd=str(ROOT))
            st.caption((r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else "done")
        st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

st.caption("Engine: point-wise isotonic-calibrated GBT (gbt_fireguard) on the FGDC v2 4 km grid — all features "
           "self-sourced from the operational feeds it serves from (no train/serve gap). The no-cold-start daily "
           "loop (append → recompute engineered over a trailing window → predict) replaces v1's seasonal warm-start.")
