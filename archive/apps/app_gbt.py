"""Streamlit map app for the PRODUCTION point-wise GBT (post-2026-06-07 pivot — the live model).

Serves models/gbt_coarse4.joblib (+ optional isotonic calibrator) on the coarse4 (4 km) grid: per land
cell, predict_proba on the 146 features, optionally calibrate, render the risk surface. GBT inference is
instant (no GPU), so a date renders in well under a second. Views: probability (yellow→red), Ignition-vs-
Spread (warm/cool regime colours), label. Reprojection via pyproj (no rasterio), precomputed once.

Run:  streamlit run docker/monolith/app_gbt.py
Env:  IBERFIRE_ZARR_PATH, NORM_STATS_PATH, GBT_PATH, CALIBRATOR_PATH, SOURCE_EPSG
"""
from __future__ import annotations
import base64
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

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


@dataclass
class Cfg:
    zarr_path: str; stats_path: str; gbt_path: str; calib_path: str
    source_epsg: int; regime_dist_cells: float; time_start: str; time_end: str


def get_cfg() -> Cfg:
    root = str(Path(__file__).resolve().parents[2])
    return Cfg(
        zarr_path=os.getenv("IBERFIRE_ZARR_PATH", f"{root}/data/gold/IberFire_coarse4.zarr"),
        stats_path=os.getenv("NORM_STATS_PATH", f"{root}/stats/coarse4_norm_stats_train.json"),
        gbt_path=os.getenv("GBT_PATH", f"{root}/models/gbt_coarse4.joblib"),
        calib_path=os.getenv("CALIBRATOR_PATH", f"{root}/models/gbt_coarse4.calibrator.joblib"),
        source_epsg=int(os.getenv("SOURCE_EPSG", "3035")),
        regime_dist_cells=float(os.getenv("REGIME_DIST_CELLS", "1.5")),
        time_start=os.getenv("TIME_START", "2019-01-01"), time_end=os.getenv("TIME_END", "2024-12-31"))


@st.cache_resource(show_spinner=False)
def load_feats(zarr_path: str):
    return build_segmentation_features(xr.open_zarr(zarr_path, consolidated=True).data_vars)


@st.cache_resource(show_spinner=False)
def load_dataset(cfg: Cfg, feats: tuple):
    return RegimeIberFireDataset(zarr_path=Path(cfg.zarr_path), time_start=cfg.time_start, time_end=cfg.time_end,
        feature_vars=list(feats), label_var="is_fire", lead_time=1, compute_stats=False,
        stats_path=Path(cfg.stats_path), mode="all", regime_dist_cells=cfg.regime_dist_cells)


@st.cache_resource(show_spinner=False)
def load_gbt(cfg: Cfg):
    art = joblib.load(cfg.gbt_path)
    calib = joblib.load(cfg.calib_path) if Path(cfg.calib_path).exists() else None
    return art["model"], art["features"], calib


@st.cache_data(show_spinner=False, max_entries=128)
def predict_cached(_gbt, _calib, _ds, gbt_path, idx, calibrate):
    X, y, reg = _ds[idx]
    C, H, W = X.shape
    Xf = X.numpy().reshape(C, -1).T
    regf = reg[0].numpy().ravel(); land = regf > 0
    p = _gbt.predict_proba(Xf[land])[:, 1]
    if calibrate and _calib is not None:
        p = _calib.predict(p)
    prob = np.zeros(H * W, np.float32); prob[land] = p
    return prob.reshape(H, W), y[0].numpy(), reg[0].numpy()


# ---------- rendering (pyproj reproject-once + folium; ported from app_seg) ----------
def _stretch(p, mode):
    p = np.clip(np.nan_to_num(np.asarray(p, np.float32)), 0, 1)
    if mode == "p99_stretch":
        d = float(np.percentile(p[p > 0], 99)) if (p > 0).any() else 0.0
        if d > 0:
            p = np.clip(p / d, 0, 1)
    return p


def probs_to_rgba(prob, mode, amax=200, eps=1e-4):
    p = _stretch(prob, mode); rgba = np.zeros((*p.shape, 4), np.uint8)
    rgba[..., 0] = 255; rgba[..., 1] = ((1 - p) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(p > eps, np.clip(p * amax, 0, amax), 0).astype(np.uint8)
    return rgba


def regime_prob_to_rgba(prob, regime, mode, amax=210, eps=1e-4):
    p = _stretch(prob, mode); rgba = np.zeros((*p.shape, 4), np.uint8)
    ign = regime == 1; spr = regime == 2
    rgba[..., 0] = np.where(ign, 255, 0)
    rgba[..., 1] = np.where(ign, ((1 - p) * 165).astype(np.uint8), 0)
    rgba[..., 2] = np.where(spr, 255, 0)
    rgba[..., 1] = np.where(spr, (60 + (1 - p) * 140).astype(np.uint8), rgba[..., 1])
    rgba[..., 3] = np.where((p > eps) & (regime > 0), np.clip(p * amax, 0, amax), 0).astype(np.uint8)
    return rgba


def mask_to_rgba(m01, rgb=(255, 0, 255)):
    m = (np.asarray(m01) > 0.5).astype(np.uint8); rgba = np.zeros((*m.shape, 4), np.uint8)
    for k in range(3):
        rgba[..., k] = m * rgb[k]
    rgba[..., 3] = m * 255; return rgba


@st.cache_resource(show_spinner=False)
def reproj_index(zarr_path: str, source_epsg: int):
    try:
        z = xr.open_zarr(zarr_path, consolidated=True)
        x = z["x"].values.astype(float); y = z["y"].values.astype(float)
        H, W = len(y), len(x); dx = (x[-1] - x[0]) / (W - 1); dy = (y[-1] - y[0]) / (H - 1)
        fwd = Transformer.from_crs(f"EPSG:{source_epsg}", "EPSG:4326", always_xy=True)
        clon, clat = fwd.transform([x[0], x[0], x[-1], x[-1]], [y[0], y[-1], y[0], y[-1]])
        lon0, lon1, lat0, lat1 = min(clon), max(clon), min(clat), max(clat)
        lons = np.linspace(lon0, lon1, W); lats = np.linspace(lat1, lat0, H)
        LON, LAT = np.meshgrid(lons, lats)
        inv = Transformer.from_crs("EPSG:4326", f"EPSG:{source_epsg}", always_xy=True)
        SX, SY = inv.transform(LON.ravel(), LAT.ravel())
        col = np.rint((np.asarray(SX) - x[0]) / dx).astype(np.int64)
        row = np.rint((np.asarray(SY) - y[0]) / dy).astype(np.int64)
        valid = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        idx = np.where(valid, row * W + col, -1).reshape(H, W)
        return idx, [[float(lat0), float(lon0)], [float(lat1), float(lon1)]]
    except Exception:
        return None


def apply_reproj(a, idx):
    f = np.asarray(a, np.float32).ravel()
    return np.where(idx >= 0, f[np.clip(idx, 0, f.size - 1)], np.nan).astype(np.float32)


def compute_bounds(ds, cfg):
    x = np.asarray(ds.ds["x"].values); y = np.asarray(ds.ds["y"].values)
    xmin, xmax, ymin, ymax = float(x.min()), float(x.max()), float(y.min()), float(y.max())
    if abs(xmin) <= 360 and abs(ymax) <= 180:
        return [[ymin, xmin], [ymax, xmax]]
    t = Transformer.from_crs(f"EPSG:{cfg.source_epsg}", "EPSG:4326", always_xy=True)
    lons, lats = [], []
    for xx, yy in [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]:
        lo, la = t.transform(xx, yy); lons.append(lo); lats.append(la)
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


def rgba_png(rgba):
    b = io.BytesIO(); Image.fromarray(rgba, "RGBA").save(b, "PNG"); return b.getvalue()


def folium_map(png, bounds):
    (la0, lo0), (la1, lo1) = bounds
    m = folium.Map(location=[(la0 + la1) / 2, (lo0 + lo1) / 2], zoom_start=6,
                   tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                   attr="Esri World Imagery")
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    folium.raster_layers.ImageOverlay(image=url, bounds=[[la0, lo0], [la1, lo1]], opacity=1.0, zindex=1).add_to(m)
    return m


# ---------------------------- UI ----------------------------
st.set_page_config(page_title="IberFire coarse4 — GBT (live model)", layout="wide")
st.title("Next-day wildfire risk — coarse4 (4 km) point-wise GBT")
cfg = get_cfg()
with st.sidebar:
    st.subheader("Model: gradient-boosted trees (point-wise)")
    st.caption(f"{Path(cfg.gbt_path).name}  |  cube {Path(cfg.zarr_path).name}")
    view = st.radio("View", ["Probability (yellow→red)", "Ignition vs Spread (warm/cool)", "Label (fire t+1)"])
    calibrate = st.checkbox("Calibrated probability", value=True, help="isotonic-corrected (true-prevalence) vs raw GBT")
    stretch = st.radio("Scaling", ["p99_stretch", "raw"], index=0)
    use_reproj = st.checkbox("Accurate reprojection (EPSG:3035→WGS84)", value=True)

with st.spinner("Loading model + dataset (cached)..."):
    feats = load_feats(cfg.zarr_path)
    gbt, gbt_feats, calib = load_gbt(cfg)
    ds = load_dataset(cfg, tuple(feats))
    bounds0 = compute_bounds(ds, cfg)
if calib is None:
    st.sidebar.warning("No calibrator found — showing raw GBT probabilities.")

dates = [str(ds.get_time_value(i))[:10] for i in range(len(ds))]
c1, c2 = st.columns([3, 1])
date = c1.selectbox("Date (features at t → fire risk at t+1)", dates, index=min(len(dates) - 1, len(dates) // 2))
go = c2.button("Render map", use_container_width=True)

if go:
    idx = dates.index(date)
    prob, y, reg = predict_cached(gbt, calib, ds, cfg.gbt_path, idx, calibrate and calib is not None)
    st.caption(f"land cells {int((reg>0).sum())} | ignition-regime {int((reg==1).sum())}, spread-regime "
               f"{int((reg==2).sum())} | actual fire@t+1 {int((y==1).sum())} | max prob {prob[reg>0].max():.3f}"
               f"{' (calibrated)' if (calibrate and calib is not None) else ' (raw)'}")
    bounds = bounds0
    mapping = reproj_index(cfg.zarr_path, cfg.source_epsg) if use_reproj else None
    if mapping is not None:
        idxmap, bounds = mapping
        prob = apply_reproj(prob, idxmap)
        reg = np.rint(np.nan_to_num(apply_reproj(reg.astype(np.float32), idxmap))).astype(int)
        y = (np.nan_to_num(apply_reproj((y > 0.5).astype(np.float32), idxmap)) > 0.5).astype(np.float32)
    if view.startswith("Ignition"):
        rgba = regime_prob_to_rgba(prob, reg, stretch)
    elif view.startswith("Label"):
        rgba = mask_to_rgba(y > 0.5)
    else:
        rgba = probs_to_rgba(prob, stretch)
    st_folium(folium_map(rgba_png(rgba), bounds), width=1100, height=650)
    if view.startswith("Ignition"):
        st.caption("🟧 warm = new-ignition risk (far from active fire) · 🟦 cool = spread risk (near active fire). Intensity ∝ probability.")
