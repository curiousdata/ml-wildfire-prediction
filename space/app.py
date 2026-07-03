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
from datetime import datetime, timezone
from pathlib import Path

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
CELL_KM2 = 16.0
HIGH, ELEV, MOD = 0.50, 0.20, 0.05    # calibrated-probability ALERT tiers (text panel bands)
# Map colour ramp = LOG scale over the calibrated-probability dynamic range. Per-cell next-day probs are tiny and
# heavy-tailed (prevalence ≈ 7e-4; median ~1e-4, p99 ~4e-3, rare cells → 0.6), so a LINEAR ramp anchored to the
# alert tiers shows almost nothing. Log-ramp from RAMP_FLOOR (≈3× prevalence; below → transparent so quiet days
# stay dark) to RAMP_CAP (≈100× prevalence → full red). Absolute → comparable across days.
RAMP_FLOOR, RAMP_CAP = 0.0013, 0.02   # low FLOOR = more visible cells; low CAP = field spans into red (dramatic)
_LF = float(np.log(RAMP_FLOOR)); _LS = float(np.log(RAMP_CAP) - np.log(RAMP_FLOOR))
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


@st.cache_data(show_spinner=False)
def load_importance():
    p = ROOT / "gbt_coarse4.importance.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {r: [t["feature"] for t in v["top"]] for r, v in d.get("regimes", {}).items()}


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


def risk_rgba(prob):
    """LOG-scaled, prevalence-anchored ramp. Per-cell probs span ~1e-4→0.6, so colour position is log(p): below
    RAMP_FLOOR (≈3× prevalence) → transparent (quiet days & the noise floor stay dark); RAMP_FLOOR→RAMP_CAP
    log-ramps yellow→orange→red. Absolute (comparable across days) but reveals the whole heavy-tailed field, so a
    genuinely-elevated day lights up instead of clipping to black (the linear tier-anchored ramp did the latter)."""
    p = np.clip(np.nan_to_num(prob), 0, 1)
    with np.errstate(divide="ignore"):
        t = np.clip((np.log(np.maximum(p, 1e-12)) - _LF) / _LS, 0.0, 1.0)   # 0 at FLOOR, 1 at CAP (log position)
    Y = np.array([255, 176, 32.]); O = np.array([255, 88, 8.]); R = np.array([244, 33, 28.])  # amber→orange-red→red
    lo = (t < 0.5)[..., None]; f = np.where(t < 0.5, t * 2, (t - 0.5) * 2)[..., None]
    rgb = np.where(lo, Y, O) * (1 - f) + np.where(lo, O, R) * f
    a = np.where(p > RAMP_FLOOR, np.clip(95 + 145 * (t ** 0.45), 0, 240), 0.0)   # opaque field (drama); dark below floor
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
        background:linear-gradient(90deg,rgba(255,232,74,.12),#ffe84a 18%,#ff8c00 60%,#dc1a1a);"></div>
      <div style="display:flex;justify-content:space-between;width:156px;font-size:10px;color:#7d8aa0;margin-top:3px;">
        <span>low</span><span>high</span></div>
      <div style="margin-top:8px;font-size:11px;color:#8b98a8;">
        <span style="color:#96f5ff;font-size:13px;">■</span> burning now · issued {issue}</div>
    </div>'''


@st.cache_data(show_spinner=False)
def build_map_html(prob_bytes, fire_bytes, shape, issue, target, _idxmap, _bounds):
    """Build the folium map once and cache its HTML. Dark schematic base + tier-anchored risk glow + cyan fire."""
    prob = np.frombuffer(prob_bytes, np.float32).reshape(shape)
    fire = np.frombuffer(fire_bytes, np.float32).reshape(shape)
    (la0, lo0), (la1, lo1) = _bounds
    m = folium.Map(location=[(la0 + la1) / 2, (lo0 + lo1) / 2], zoom_start=6, tiles="cartodbdark_matter",
                   zoom_control=True, control_scale=False)
    pr = gather(prob, _idxmap); fr = np.nan_to_num(gather(fire, _idxmap))
    risk = np.array(Image.fromarray(risk_rgba(pr)).filter(ImageFilter.GaussianBlur(0.8)))   # tight glow, keeps punch
    layer = alpha_over(risk, fire_rgba(fr > 0.5))                                            # fire stays crisp on top
    url = "data:image/png;base64," + base64.b64encode(png(layer)).decode()
    folium.raster_layers.ImageOverlay(image=url, bounds=[[la0, lo0], [la1, lo1]], opacity=0.96, zindex=2).add_to(m)
    m.get_root().html.add_child(folium.Element(_legend(issue, target)))
    return m._repr_html_()


def regions_of(mask, ccaa):
    codes = np.unique(ccaa[(mask > 0.5) & (ccaa > 0)])
    return [CCAA.get(int(c), f"R{int(c)}") for c in codes]


def ago(iso_or_mtime):
    """Human 'time ago' at MINUTE granularity (no ticking seconds)."""
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
    if s < 120:
        return "just now"
    if s < 5400:
        return f"{int(s // 60)}m ago"
    if s < 172800:
        return f"{int(s // 3600)}h ago"
    return f"{int(s // 86400)}d ago"


# ---------------------------- UI ----------------------------
st.set_page_config(page_title="Fire Guard Control Center", page_icon="🛡️", layout="wide")
st.markdown("""<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
  html, body, [class*="css"] { font-family:'Inter',system-ui,-apple-system,sans-serif; }
  .stApp { background:#0d1117; }
  .block-container { padding-top:1.1rem; padding-bottom:1rem; max-width:1400px; }
  iframe { border-radius:14px; }
  .fg-title { font-size:1.7rem; font-weight:800; color:#eef2f7; letter-spacing:-.01em; line-height:1.05; }
  .fg-sub { font-size:.82rem; color:#7d8aa0; margin-top:3px; letter-spacing:.03em; }
  .fg-status { text-align:right; font-size:1.05rem; font-weight:700; line-height:1.1; }
  .fg-status-age { display:block; font-size:.8rem; color:#7d8aa0; font-weight:400; margin-top:3px; }
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
importance = load_importance()
S = latest_store()
if not S:
    st.markdown("<div class='fg-title'>Fire Guard Control Center</div>", unsafe_allow_html=True)
    st.warning("No prediction published yet — the engine hasn't produced one. Check back shortly.")
    st.stop()
st.session_state["seen_version"] = latest_version()

prob, regime, fire = S["prob"], S["regime"], S["today_fire"]
land = regime > 0
prelim = S.get("prelim")
n_fire = int((fire > 0.5).sum())


# ---------- title + status (60 s refresh for the 'updated' stamp; no ticking clock) ----------
@st.fragment(run_every=60)
def topbar():
    c1, c2 = st.columns([4, 2])
    c1.markdown("<div class='fg-title'>Fire Guard <span style='color:#9fb3c8;font-weight:700'>Control Center</span></div>"
                "<div class='fg-sub'>next-day wildfire risk · Spain</div>", unsafe_allow_html=True)
    dot, col = ("◐ PRELIMINARY", "#f0a020") if prelim else ("● LIVE", "#4ade80")
    c2.markdown(f"<div class='fg-status' style='color:{col}'>{dot}"
                f"<span class='fg-status-age'>updated {ago(S['fetched_at'] or S['mtime'])}"
                f"{' · refines ~17 UTC' if prelim else ''}</span></div>", unsafe_allow_html=True)


topbar()

# ---------- MAP: full-width hero ----------
html = build_map_html(np.ascontiguousarray(prob, np.float32).tobytes(),
                      np.ascontiguousarray(fire, np.float32).tobytes(), prob.shape,
                      S["issue"], S["target"], idxmap, bounds)
components.html(html, height=620)


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
    if not (ig or sp) and importance:
        ig = [pretty(f) for f in importance.get("ignition", [])[:3]]
        sp = [pretty(f) for f in importance.get("spread", [])[:3]]
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
st.markdown("<div class='foot'>Calibrated gradient-boosted trees on the Fire Guard Datacube v2 · "
            "preliminary in the morning, final after the afternoon satellite pass (~17 UTC) · "
            "base © CARTO / OpenStreetMap.</div>", unsafe_allow_html=True)
