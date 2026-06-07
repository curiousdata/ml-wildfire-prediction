"""IberFire live-risk dashboard (replay mode) — the operational map vision on existing data.

ONE map, THREE layers for a chosen "today" (t): (1) today's fires [is_fire(t)], (2) tomorrow's SPREAD risk
[GBT prob on spread-regime cells, near fire], (3) tomorrow's IGNITION risk [GBT prob on ignition-regime
cells, far from fire]. Plus a region ALERT panel (per Autonomous Community). Runs in REPLAY: a date
stepper plays the historical cube as if it were arriving daily — clearly labelled, no faked liveness.
True real-time swaps the replay clock for live feeds (AEMET weather + FIRMS fire + CLMS vegetation) — see
scripts/fetch_aemet.py and the ROADMAP §F real-time track.

Toggles stripped per design: reprojection + p99 scaling are always on.
Run:  streamlit run docker/monolith/app_live.py
"""
from __future__ import annotations
import base64, io, os, sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import joblib  # noqa: E402
import xarray as xr  # noqa: E402
import folium  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402
from src.data.datasets import RegimeIberFireDataset  # noqa: E402
from src.data.features import build_segmentation_features  # noqa: E402

CCAA = {1: "Andalucía", 2: "Aragón", 3: "Asturias", 4: "Baleares", 6: "Cantabria",
        7: "Castilla y León", 8: "Castilla-La Mancha", 9: "Cataluña", 10: "C. Valenciana",
        11: "Extremadura", 12: "Galicia", 13: "Madrid", 14: "Murcia", 15: "Navarra",
        16: "País Vasco", 17: "La Rioja"}


@dataclass
class Cfg:
    zarr_path: str; stats_path: str; gbt_path: str; calib_path: str
    source_epsg: int; regime_dist_cells: float; time_start: str; time_end: str


def get_cfg() -> Cfg:
    r = str(Path(__file__).resolve().parents[2])
    return Cfg(os.getenv("IBERFIRE_ZARR_PATH", f"{r}/data/gold/IberFire_coarse4.zarr"),
               os.getenv("NORM_STATS_PATH", f"{r}/stats/coarse4_norm_stats_train.json"),
               os.getenv("GBT_PATH", f"{r}/models/gbt_coarse4.joblib"),
               os.getenv("CALIBRATOR_PATH", f"{r}/models/gbt_coarse4.calibrator.joblib"),
               int(os.getenv("SOURCE_EPSG", "3035")), float(os.getenv("REGIME_DIST_CELLS", "1.5")),
               os.getenv("TIME_START", "2019-01-01"), os.getenv("TIME_END", "2024-12-31"))


@st.cache_resource(show_spinner=False)
def load_feats(p): return build_segmentation_features(xr.open_zarr(p, consolidated=True).data_vars)


@st.cache_resource(show_spinner=False)
def load_dataset(cfg, feats):
    return RegimeIberFireDataset(zarr_path=Path(cfg.zarr_path), time_start=cfg.time_start, time_end=cfg.time_end,
        feature_vars=list(feats), label_var="is_fire", lead_time=1, compute_stats=False,
        stats_path=Path(cfg.stats_path), mode="all", regime_dist_cells=cfg.regime_dist_cells)


@st.cache_resource(show_spinner=False)
def load_gbt(cfg):
    art = joblib.load(cfg.gbt_path)
    calib = joblib.load(cfg.calib_path) if Path(cfg.calib_path).exists() else None
    return art["model"], calib


@st.cache_resource(show_spinner=False)
def load_ccaa(zarr_path):
    z = xr.open_zarr(zarr_path, consolidated=True)
    return np.rint(np.nan_to_num(z["AutonomousCommunities"].values)).astype(int)


@st.cache_data(show_spinner=False, max_entries=128)
def predict_day(_gbt, _calib, _ds, gbt_path, idx):
    """Return (prob_tomorrow[H,W] calibrated, regime[H,W], today_fire[H,W])."""
    X, y, reg = _ds[idx]
    C, H, W = X.shape
    regf = reg[0].numpy().ravel(); land = regf > 0
    p = _gbt.predict_proba(X.numpy().reshape(C, -1).T[land])[:, 1]
    if _calib is not None:
        p = _calib.predict(p)
    prob = np.zeros(H * W, np.float32); prob[land] = p
    t = _ds.get_time_value(idx)
    today_fire = (_ds.ds["is_fire"].sel(time=t).values > 0.5).astype(np.float32)
    return prob.reshape(H, W), reg[0].numpy(), today_fire


# ---------- rendering ----------
def _stretch(p):
    p = np.clip(np.nan_to_num(np.asarray(p, np.float32)), 0, 1)
    d = float(np.percentile(p[p > 0], 99)) if (p > 0).any() else 0.0
    return np.clip(p / d, 0, 1) if d > 0 else p


def warm_rgba(prob, mask, amax=215, eps=1e-4):  # ignition: orange->red
    p = _stretch(np.where(mask, prob, 0.0)); rgba = np.zeros((*p.shape, 4), np.uint8)
    rgba[..., 0] = 255; rgba[..., 1] = ((1 - p) * 150).astype(np.uint8)
    rgba[..., 3] = np.where((p > eps) & mask, np.clip(p * amax, 0, amax), 0).astype(np.uint8)
    return rgba


def cool_rgba(prob, mask, amax=215, eps=1e-4):  # spread: cyan->blue
    p = _stretch(np.where(mask, prob, 0.0)); rgba = np.zeros((*p.shape, 4), np.uint8)
    rgba[..., 2] = 255; rgba[..., 1] = np.where(mask, (60 + (1 - p) * 150).astype(np.uint8), 0)
    rgba[..., 3] = np.where((p > eps) & mask, np.clip(p * amax, 0, amax), 0).astype(np.uint8)
    return rgba


def fire_rgba(mask):  # today's fire: solid magenta (R+B)
    m = (np.asarray(mask) > 0.5); rgba = np.zeros((*m.shape, 4), np.uint8)
    rgba[..., 0] = np.where(m, 255, 0)
    rgba[..., 2] = np.where(m, 255, 0)
    rgba[..., 3] = np.where(m, 255, 0).astype(np.uint8)
    return rgba


def alpha_over(bot, top):
    b = bot.astype(np.float32) / 255; t = top.astype(np.float32) / 255
    ta, ba = t[..., 3:4], b[..., 3:4]; oa = ta + ba * (1 - ta)
    rgb = (t[..., :3] * ta + b[..., :3] * ba * (1 - ta)) / np.clip(oa, 1e-8, 1)
    out = np.zeros_like(b); out[..., :3] = rgb; out[..., 3:4] = oa
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


@st.cache_resource(show_spinner=False)
def reproj_index(zarr_path, epsg):
    try:
        z = xr.open_zarr(zarr_path, consolidated=True)
        x = z["x"].values.astype(float); y = z["y"].values.astype(float)
        H, W = len(y), len(x); dx = (x[-1] - x[0]) / (W - 1); dy = (y[-1] - y[0]) / (H - 1)
        fwd = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        clon, clat = fwd.transform([x[0], x[0], x[-1], x[-1]], [y[0], y[-1], y[0], y[-1]])
        lo0, lo1, la0, la1 = min(clon), max(clon), min(clat), max(clat)
        LON, LAT = np.meshgrid(np.linspace(lo0, lo1, W), np.linspace(la1, la0, H))
        inv = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        SX, SY = inv.transform(LON.ravel(), LAT.ravel())
        col = np.rint((np.asarray(SX) - x[0]) / dx).astype(np.int64)
        row = np.rint((np.asarray(SY) - y[0]) / dy).astype(np.int64)
        ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        return np.where(ok, row * W + col, -1).reshape(H, W), [[float(la0), float(lo0)], [float(la1), float(lo1)]]
    except Exception:
        return None


def gather(a, idx):
    f = np.asarray(a, np.float32).ravel()
    return np.where(idx >= 0, f[np.clip(idx, 0, f.size - 1)], np.nan).astype(np.float32)


def png(rgba):
    b = io.BytesIO(); Image.fromarray(rgba, "RGBA").save(b, "PNG"); return b.getvalue()


def fmap(rgba, bounds):
    (la0, lo0), (la1, lo1) = bounds
    m = folium.Map(location=[(la0 + la1) / 2, (lo0 + lo1) / 2], zoom_start=6,
                   tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                   attr="Esri World Imagery")
    url = "data:image/png;base64," + base64.b64encode(png(rgba)).decode()
    folium.raster_layers.ImageOverlay(image=url, bounds=[[la0, lo0], [la1, lo1]], opacity=1.0, zindex=1).add_to(m)
    return m


# ---------------------------- UI ----------------------------
st.set_page_config(page_title="IberFire live risk (replay)", layout="wide")
cfg = get_cfg()
feats = load_feats(cfg.zarr_path)
gbt, calib = load_gbt(cfg)
ds = load_dataset(cfg, tuple(feats))
ccaa = load_ccaa(cfg.zarr_path)
mapping = reproj_index(cfg.zarr_path, cfg.source_epsg)
dates = [str(ds.get_time_value(i))[:10] for i in range(len(ds))]

st.title("🔥 IberFire — next-day wildfire risk")
left, right = st.columns([3, 1])
with right:
    st.subheader("Replay clock")
    idx = st.slider("'Today' (t)", 0, len(dates) - 1, len(dates) // 2, help="Replays the cube as if arriving daily")
    today = dates[idx]
    st.metric("Prediction issued (replay)", f"{today}  18:00")
    st.caption(f"→ risk for **tomorrow**, {dates[min(idx+1, len(dates)-1)]}")
    st.caption("⚠️ Replay on historical data. Live mode swaps this clock for AEMET+FIRMS+CLMS feeds (§F).")
    alert_thr = st.slider("Alert threshold (calibrated prob)", 0.05, 0.9, 0.25, 0.05)

prob, reg, today_fire = predict_day(gbt, calib, ds, cfg.gbt_path, idx)
ign_m, spr_m = reg == 1, reg == 2

# reproject all fields, build + composite the 3 layers
if mapping is not None:
    idxm, bounds = mapping
    probR = gather(prob, idxm); regR = np.rint(np.nan_to_num(gather(reg.astype(np.float32), idxm))).astype(int)
    fireR = np.nan_to_num(gather(today_fire, idxm))
    layer = alpha_over(alpha_over(cool_rgba(probR, regR == 2), warm_rgba(probR, regR == 1)), fire_rgba(fireR > 0.5))
else:
    bounds = [[float(ds.ds["y"].min()), float(ds.ds["x"].min())], [float(ds.ds["y"].max()), float(ds.ds["x"].max())]]
    layer = alpha_over(alpha_over(cool_rgba(prob, spr_m), warm_rgba(prob, ign_m)), fire_rgba(today_fire > 0.5))

with left:
    st.caption(f"🟪 fires today ({int((today_fire>0.5).sum())} cells) · 🟧 ignition risk tomorrow (far from fire) · "
               f"🟦 spread risk tomorrow (near fire). Intensity ∝ calibrated probability.")
    st_folium(fmap(layer, bounds), width=1000, height=620)

# ---- ALERTS (per Autonomous Community) ----
with right:
    st.subheader("⚠️ Alerts — tomorrow")
    alerts = []
    for code, name in CCAA.items():
        rm = ccaa == code
        if not rm.any():
            continue
        ign = prob[rm & ign_m]; spr = prob[rm & spr_m]
        imax = float(ign.max()) if ign.size else 0.0
        smax = float(spr.max()) if spr.size else 0.0
        if imax >= alert_thr:
            alerts.append((imax, "🟧 ignition", name, int((ign >= alert_thr).sum())))
        if smax >= alert_thr:
            alerts.append((smax, "🟦 spread", name, int((spr >= alert_thr).sum())))
    alerts.sort(reverse=True)
    if not alerts:
        st.success("No regions above threshold for tomorrow.")
    for score, kind, name, n in alerts[:12]:
        st.markdown(f"**{kind} risk — {name}**  ·  max {score:.2f} · {n} cell(s)")
