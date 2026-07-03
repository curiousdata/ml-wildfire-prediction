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
import os
from datetime import datetime, timezone
from pathlib import Path

import folium
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from pyproj import Transformer

# Deployable Space build: reads predictions from a local store + precomputed display assets (CCAA grid +
# x/y coords) — NO zarr cube, NO torch, NO model at runtime. The engine (scripts/daily_job.py, run by a
# scheduler) writes new grids into store/; this app only renders the latest. See space/README.md.
ROOT = Path(__file__).resolve().parent
# Read predictions from the persistent bucket (/data, written by the scheduled engine) once it has any;
# until then fall back to the seed prediction bundled in the repo so the Space renders immediately.
STORE = Path("/data") if (Path("/data") / "grids").exists() else ROOT / "store"
ASSETS = ROOT / "display_assets.npz"
# Ship A: the LOCAL engine (scripts/daily_job.py) publishes each prediction to an HF Dataset via
# scripts/push_serving.py; this Space READS that Dataset — poll the tiny latest.json manifest, and pull the
# referenced grid only when it changes. Falls back to the local STORE if HF is unreachable. Set
# FIREGUARD_LOCAL_STORE=1 to force the local store (local dev). See the `hf-deploy-plan` memory.
SERVING_REPO = os.getenv("FIREGUARD_SERVING_REPO", "curiousdata/fireguard-serving")
FORCE_LOCAL = os.getenv("FIREGUARD_LOCAL_STORE") == "1"
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
    return np.load(ASSETS)["ccaa"].astype(int)


@st.cache_data(show_spinner=False)
def load_importance():
    p = ROOT / "gbt_coarse4.importance.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {r: [t["feature"] for t in v["top"]] for r, v in d.get("regimes", {}).items()}


def _unpack_grid(d, *, issue=None, target=None, source=None, mtime=None, prelim=False):
    """Common npz → view dict, tolerating older grids that lack some keys."""
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
    """Poll the tiny latest.json manifest (cheap — cached 8s so the 1s/10s fragments don't hammer HF)."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(SERVING_REPO, "latest.json", repo_type="dataset")
    return json.loads(Path(p).read_text())


@st.cache_data(show_spinner=False)
def _grid_view(grid_path, version, issue, target, source, prelim):
    """Download + unpack the grid, cached by (path, version) so full reruns don't re-fetch or re-load until the
    published prediction actually changes (store_watcher clears the cache when pushed_at moves)."""
    from huggingface_hub import hf_hub_download
    g = hf_hub_download(SERVING_REPO, grid_path, repo_type="dataset")
    return _unpack_grid(np.load(g, allow_pickle=True), issue=issue, target=target,
                        source=source, mtime=version, prelim=prelim)


def _hf_latest(man):
    """Pull the grid the manifest points at (cached by pushed_at → only re-downloads when it changes)."""
    return _grid_view(man["grid_path"], man.get("pushed_at"), man.get("issue"), man.get("target"),
                      man.get("source", "?"), man.get("prelim", False))


def _local_latest():
    grids = sorted((STORE / "grids").glob("*.npz"))
    if not grids:
        return None
    g = grids[-1]
    return _unpack_grid(np.load(g, allow_pickle=True), mtime=g.stat().st_mtime)


def latest_version():
    """Cheap 'has anything changed?' token for the store-watcher: the manifest's pushed_at (HF) or grid mtime."""
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
    """Latest published prediction — HF Dataset first (Ship A), local store as fallback."""
    if not FORCE_LOCAL:
        try:
            m = _hf_manifest()
            if m:
                return _hf_latest(m)
        except Exception:
            pass                                          # HF unreachable / empty → local
    return _local_latest()


# ---------- map rendering ----------
@st.cache_resource(show_spinner=False)
def reproj_index():
    a = np.load(ASSETS)
    x = a["x"].astype(float); y = a["y"].astype(float)
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
    st.warning("No prediction in the store yet — the scheduled engine hasn't published one. Check back shortly.")
    st.stop()
st.session_state["seen_version"] = latest_version()   # baseline for the store-watcher fragment

prob, regime, fire = S["prob"], S["regime"], S["today_fire"]
land = regime > 0
# FGDC v2 has NO warm-start and NO EFFIS: every prediction is built from live sources (Open-Meteo ERA5 weather +
# FIRMS VIIRS fire + live-derived dryness; veg carried from the latest MODIS composite). `source` is the only real
# state — a live forecast edge vs a replayed settled cube day; both are real data. (The old v1 `refreshed`-token
# warm-start/EFFIS/legacy chrome is retired here.)
src = S.get("source", "?")
is_live = src.startswith("live")                       # live / live-prelim / live-settled (vs "replay")


# ---------- header: branding + LIVE + clock (auto-ticking) ----------
@st.fragment(run_every=1)
def header_live():
    c1, c2, c3 = st.columns([3, 2, 2])
    c1.markdown("## 🛡️ Fire Guard\n###### wildfire control center · Spain")
    prelim = S.get("prelim")
    dot = "◐ PRELIMINARY" if prelim else "● LIVE"
    color = "#f0a020" if prelim else "#ff4b4b"
    c2.markdown(f"<div class='feed-name'>STATUS</div>"
                f"<div class='feed-val' style='color:{color}'>{dot}</div>"
                f"<div class='feed-age'>updated {ago(S['fetched_at'] or S['mtime'])}"
                f"{' · refines after ~17 UTC' if prelim else ''}</div>",
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
    asof = f"as of {ago(S['fetched_at'] or S['mtime'])}"
    feeds = [
        ("🛰 Satellite", "NASA GIBS true-color", f"imagery {S['issue']}"),
        ("🌡 Weather", "Open-Meteo (ERA5)" if is_live else "ERA5 reanalysis", asof),
        ("🔥 Active fire", f"FIRMS VIIRS · {n_fire} cells", asof),
        ("🌵 Dryness", "live antecedents · KBDI/SPI/precip", asof),
        ("🌿 Vegetation", "MODIS · latest composite", "weekly cadence"),
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
    v = latest_version()
    if v is not None and v != st.session_state.get("seen_version"):
        st.cache_data.clear()   # drop the cached manifest + map so the rerun pulls the new grid
        st.rerun()  # full rerun: reload store + rebuild map


store_watcher()

# (v2 has no warm-start / EFFIS inputs to caveat — the one genuine caveat, morning-preliminary vs settled, is the
# ◐ PRELIMINARY status badge in the header. The old v1 "degraded inputs" banner is retired.)


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

st.caption("Engine: point-wise calibrated gradient-boosted trees (Fire Guard Datacube v2) on the 4 km grid. "
           "Live inputs — Open-Meteo ERA5 weather · FIRMS VIIRS active fire · live-derived drought/dryness "
           "(KBDI/SPI/precip); vegetation from the latest MODIS composite (weekly cadence). Predictions refine "
           "through the day: preliminary in the morning, final after the afternoon satellite pass settles (~17 UTC).")
