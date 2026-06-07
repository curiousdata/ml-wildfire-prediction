"""IberFire — wildfire control center (operational live view).

The main application: a live forest-fire control center for Spain. Not a date-picker demo — it shows the
LATEST prediction the engine produced and is built to feel operational.

Architecture: the engine is `scripts/daily_job.py --mode live` (fetch Open-Meteo + FIRMS → assemble slice →
GBT → write date-partitioned store under data/serving_store/). THIS app READS the latest stored prediction
(fast, no in-app rate-limited fetch) and renders:
  • header: ● LIVE + clock + "latest data change" (store freshness);
  • a fresh daily-satellite map (NASA GIBS VIIRS true-color, dated) with today's fires + tomorrow's risk;
  • NOW burning — area + which regions are alight (today's FIRMS fires);
  • TOMORROW — expected burn area + HIGH / ELEVATED risk regions (lists), from the calibrated GBT.
A "Refresh live now" button runs the engine for today on demand.

Run: streamlit run docker/monolith/app_live.py
"""
from __future__ import annotations
import base64
import io
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import folium  # noqa: E402
import xarray as xr  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

STORE = ROOT / "data" / "serving_store"
CUBE = ROOT / "data" / "gold" / "IberFire_coarse4.zarr"
CELL_HA = 4.0 * 4.0 * 100  # 4 km cell = 16 km² = 1600 ha (whole-cell assumption)
SOURCE_EPSG = 3035
HIGH, ELEV = 0.50, 0.20    # calibrated-probability risk tiers
CCAA = {1: "Andalucía", 2: "Aragón", 3: "Asturias", 4: "Baleares", 6: "Cantabria",
        7: "Castilla y León", 8: "Castilla-La Mancha", 9: "Cataluña", 10: "C. Valenciana",
        11: "Extremadura", 12: "Galicia", 13: "Madrid", 14: "Murcia", 15: "Navarra",
        16: "País Vasco", 17: "La Rioja"}


@st.cache_resource(show_spinner=False)
def load_ccaa():
    return np.rint(np.nan_to_num(xr.open_zarr(str(CUBE), consolidated=True)["AutonomousCommunities"].values)).astype(int)


def latest_store():
    grids = sorted((STORE / "grids").glob("*.npz"))
    if not grids:
        return None
    g = grids[-1]
    d = np.load(g, allow_pickle=True)
    issue = str(d["issue_date"]); target = str(d["target_date"])
    src = str(d["source"]) if "source" in d else "?"
    updated = datetime.fromtimestamp(g.stat().st_mtime)
    return dict(prob=d["prob"], regime=d["regime"], today_fire=d["today_fire"],
               issue=issue, target=target, source=src, updated=updated)


# ---------- rendering ----------
@st.cache_resource(show_spinner=False)
def reproj_index():
    z = xr.open_zarr(str(CUBE), consolidated=True)
    x = z["x"].values.astype(float); y = z["y"].values.astype(float)
    H, W = len(y), len(x); dx = (x[-1] - x[0]) / (W - 1); dy = (y[-1] - y[0]) / (H - 1)
    fwd = Transformer.from_crs(f"EPSG:{SOURCE_EPSG}", "EPSG:4326", always_xy=True)
    clon, clat = fwd.transform([x[0], x[0], x[-1], x[-1]], [y[0], y[-1], y[0], y[-1]])
    lo0, lo1, la0, la1 = min(clon), max(clon), min(clat), max(clat)
    LON, LAT = np.meshgrid(np.linspace(lo0, lo1, W), np.linspace(la1, la0, H))
    inv = Transformer.from_crs("EPSG:4326", f"EPSG:{SOURCE_EPSG}", always_xy=True)
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
    ps = np.clip(p / d, 0, 1) if d > 0 else p  # stretch so low-but-real risk is visible
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


def control_map(prob, fire, idxmap, bounds, gibs_date):
    (la0, lo0), (la1, lo1) = bounds
    m = folium.Map(location=[(la0 + la1) / 2, (lo0 + lo1) / 2], zoom_start=6, tiles=None)
    # fresh daily true-color satellite (NASA GIBS, dated); Esri as static fallback base under it
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri", name="Satellite (static)").add_to(m)
    gibs = (f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_SNPP_CorrectedReflectance_TrueColor/"
            f"default/{gibs_date}/GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.jpg")
    folium.TileLayer(gibs, attr="NASA GIBS / VIIRS", name=f"🛰 Fresh imagery {gibs_date}", overlay=False).add_to(m)
    pr = gather(prob, idxmap); fr = np.nan_to_num(gather(fire, idxmap))
    layer = alpha_over(risk_rgba(pr), fire_rgba(fr > 0.5))
    url = "data:image/png;base64," + base64.b64encode(png(layer)).decode()
    folium.raster_layers.ImageOverlay(image=url, bounds=[[la0, lo0], [la1, lo1]], opacity=0.8, zindex=2).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def regions_of(mask, ccaa):
    codes = np.unique(ccaa[(mask > 0.5) & (ccaa > 0)])
    return [CCAA.get(int(c), f"R{int(c)}") for c in codes]


# ---------------------------- UI ----------------------------
st.set_page_config(page_title="IberFire — wildfire control center", layout="wide")
ccaa = load_ccaa()
idxmap, bounds = reproj_index()
S = latest_store()

# header
c1, c2, c3, c4 = st.columns([2, 2, 3, 2])
c1.markdown("## 🔥 IberFire control center")
if S:
    c2.markdown(f"### :red[● LIVE]")
    c3.metric("Latest data change", S["updated"].strftime("%Y-%m-%d %H:%M:%S"), help=f"engine source: {S['source']}")
    c4.metric("Now (local)", datetime.now().strftime("%H:%M:%S"))
else:
    c2.markdown("### :grey[● NO DATA]")

if c4.button("↻ Refresh live now", use_container_width=True):
    with st.spinner("Running live engine (Open-Meteo + FIRMS → GBT)… ~30–60s"):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "daily_job.py"), "--mode", "live", "--overwrite"],
                           capture_output=True, text=True, cwd=str(ROOT))
        st.caption((r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else "done")
    st.cache_resource.clear(); st.rerun()

if not S:
    st.warning("No prediction in the store yet. Run:  `python scripts/daily_job.py --mode live`  (or press Refresh).")
    st.stop()

prob, regime, fire = S["prob"], S["regime"], S["today_fire"]
land = regime > 0
left, right = st.columns([3, 2])
with left:
    st.caption(f"Issued for **{S['issue']}** → risk for **{S['target']}** · 🟪 burning now · 🟧 predicted risk (intensity ∝ calibrated probability)")
    st_folium(control_map(prob, fire, idxmap, bounds, S["issue"]), width=950, height=600)

with right:
    # NOW burning
    n_fire = int((fire > 0.5).sum())
    now_regions = regions_of(fire, ccaa)
    st.subheader("🔥 Burning now")
    if n_fire:
        st.markdown(f"**~{n_fire * CELL_HA:,.0f} ha** active ({n_fire} cells @ 4 km · whole-cell est.)")
        st.markdown("Affected: " + ", ".join(now_regions))
    else:
        st.success("No active fire cells detected.")

    # TOMORROW prediction
    st.subheader(f"📈 Tomorrow ({S['target']})")
    exp_area = float(prob[land].sum()) * CELL_HA
    st.markdown(f"**Expected burn ≈ {exp_area:,.0f} ha** (Σ calibrated risk × cell area)")
    rmax = {}
    for code, name in CCAA.items():
        rm = (ccaa == code) & land
        if rm.any():
            rmax[name] = float(prob[rm].max())
    high = [n for n, v in rmax.items() if v >= HIGH]
    elev = [n for n, v in rmax.items() if ELEV <= v < HIGH]
    st.markdown(f"🔴 **HIGH risk (≥{HIGH:.0%})** in {len(high)}: " + (", ".join(high) if high else "_none_"))
    st.markdown(f"🟠 **ELEVATED (≥{ELEV:.0%})** in {len(elev)}: " + (", ".join(elev) if elev else "_none_"))
    st.markdown("**Top regions by risk tomorrow:**")
    for name, v in sorted(rmax.items(), key=lambda kv: -kv[1])[:5]:
        st.markdown(f"&nbsp;&nbsp;{name} — {v:.1%}")

st.caption("Engine: GBT (point-wise, calibrated) on the IberFire coarse4 grid. ⚠️ Live feeds: temperature "
           "(Open-Meteo) + fire (FIRMS) are live; antecedent dryness/vegetation are seasonally warm-started "
           "(see CHANGES.md) — absolute risk values will tighten as the full live feature pipeline lands.")
