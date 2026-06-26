"""Streamlit map app for the coarse4 wide-and-deep segmentation model (v6+).

Distinct from app.py (the legacy coarse32 resnet34_v9 path). This serves the NEW pipeline:
  * coarse4 cube (4 km) + the 146-feature segmentation set (build_segmentation_features)
  * WideDeepUNet (build_wide_deep_unet), GroupNorm
  * the PER-REGIME logit adjustment applied at inference (REQUIRED — raw logits rank cross-regime
    garbage; regime is known from dist_to_fire(t), so applying it is legitimate)
  * two views: a yellow→red probability surface, and an IGNITION-vs-SPREAD regime-coloured surface
    (warm = new-ignition risk far from fire; cool = spread/continuation risk near active fire).

Runs the model on CPU by default so it doesn't contend with a training job on the GPU/MPS.

Run:  streamlit run docker/monolith/app_seg.py
Env:  IBERFIRE_ZARR_PATH, NORM_STATS_PATH, MODEL_PATH, MODEL_FILE, SEG_DEVICE, ADJ_IGNITION, ADJ_SPREAD
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
import torch
from PIL import Image
from pyproj import Transformer

# repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import xarray as xr  # noqa: E402
from src.data.datasets import RegimeIberFireDataset  # noqa: E402
from src.data.features import build_segmentation_features  # noqa: E402
from src.models.cnn import build_wide_deep_unet  # noqa: E402

import folium  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402


@dataclass
class Cfg:
    zarr_path: str
    stats_path: str
    model_path: str
    model_file: str
    device: str
    source_epsg: int
    adj_ignition: float
    adj_spread: float
    regime_dist_cells: float
    time_start: str
    time_end: str


def get_cfg() -> Cfg:
    root = str(Path(__file__).resolve().parents[2])
    return Cfg(
        zarr_path=os.getenv("IBERFIRE_ZARR_PATH", f"{root}/data/gold/IberFire_coarse4.zarr"),
        stats_path=os.getenv("NORM_STATS_PATH", f"{root}/stats/coarse4_norm_stats_train.json"),
        model_path=os.getenv("MODEL_PATH", f"{root}/models"),
        model_file=os.getenv("MODEL_FILE", "seg_coarse4_widedeep_v6_reg.pth"),
        device=os.getenv("SEG_DEVICE", "cpu"),
        source_epsg=int(os.getenv("SOURCE_EPSG", "3035")),
        adj_ignition=float(os.getenv("ADJ_IGNITION", "-10.90")),
        adj_spread=float(os.getenv("ADJ_SPREAD", "-2.67")),
        regime_dist_cells=float(os.getenv("REGIME_DIST_CELLS", "1.5")),
        time_start=os.getenv("TIME_START", "2019-01-01"),
        time_end=os.getenv("TIME_END", "2024-12-31"),
    )


@st.cache_resource(show_spinner=False)
def load_feats(zarr_path: str) -> List[str]:
    return build_segmentation_features(xr.open_zarr(zarr_path, consolidated=True).data_vars)


@st.cache_resource(show_spinner=False)
def load_dataset(cfg: Cfg, feats: tuple) -> RegimeIberFireDataset:
    return RegimeIberFireDataset(
        zarr_path=Path(cfg.zarr_path), time_start=cfg.time_start, time_end=cfg.time_end,
        feature_vars=list(feats), label_var="is_fire", lead_time=1,
        compute_stats=False, stats_path=Path(cfg.stats_path), mode="all",
        regime_dist_cells=cfg.regime_dist_cells,
    )


@st.cache_resource(show_spinner=False)
def load_model(cfg: Cfg, n_feats: int) -> torch.nn.Module:
    mf = Path(cfg.model_path) / cfg.model_file
    if not mf.exists():
        raise FileNotFoundError(f"Model not found: {mf}")
    m = build_wide_deep_unet(in_channels=n_feats, encoder_name="resnet34",
                             encoder_weights=None, norm="group").to(cfg.device)
    state = torch.load(str(mf), map_location=cfg.device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    m.load_state_dict(state, strict=True)
    m.eval()
    return m


@torch.no_grad()
def predict(model, ds, cfg, idx):
    """Return (prob[H,W], y[H,W], regime[H,W]) with the per-regime adjustment applied."""
    X, y, reg = ds[idx]
    logit = model(X.unsqueeze(0).to(cfg.device).float())[0, 0]
    r = reg[0].to(cfg.device)
    adj = torch.where(r == 1, logit.new_tensor(cfg.adj_ignition),
          torch.where(r == 2, logit.new_tensor(cfg.adj_spread), logit.new_tensor(0.0)))
    prob = torch.sigmoid(logit + adj).cpu().numpy()
    return prob, y[0].numpy(), reg[0].numpy()


@st.cache_data(show_spinner=False, max_entries=64)
def predict_cached(_model, _ds, _cfg, model_file: str, idx: int, adj_ign: float, adj_spr: float):
    """Cache (prob, y, regime) per date — a date's prediction never changes for a fixed model, so
    re-viewing a date is instant and only NEW dates pay the (CPU) forward pass. (_model/_ds/_cfg are
    underscore-prefixed so Streamlit skips hashing them; the cache key is model_file+idx+adj.)"""
    return predict(_model, _ds, _cfg, idx)


# ---------- rendering (ported from app.py; pure functions) ----------
def _stretch(p, mode):
    p = np.nan_to_num(np.asarray(p, np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    p = np.clip(p, 0.0, 1.0)
    if mode == "p99_stretch":
        d = float(np.percentile(p[p > 0], 99)) if (p > 0).any() else 0.0
        if d > 0:
            p = np.clip(p / d, 0.0, 1.0)
    return p


def probs_to_rgba(prob, mode="p99_stretch", alpha_max=200, eps=1e-4):
    """Yellow(low)→Red(high) probability surface."""
    p = _stretch(prob, mode)
    rgba = np.zeros((*p.shape, 4), np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = ((1.0 - p) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(p > eps, np.clip(p * alpha_max, 0, alpha_max), 0).astype(np.uint8)
    return rgba


def regime_prob_to_rgba(prob, regime, mode="p99_stretch", alpha_max=210, eps=1e-4):
    """Ignition risk (regime 1) = WARM (orange→red); spread risk (regime 2) = COOL (cyan→blue)."""
    p = _stretch(prob, mode)
    rgba = np.zeros((*p.shape, 4), np.uint8)
    ign = regime == 1
    spr = regime == 2
    # warm: R=255, G fades from ~165 (orange) to 0 (red) with prob
    rgba[..., 0] = np.where(ign, 255, 0)
    rgba[..., 1] = np.where(ign, ((1.0 - p) * 165).astype(np.uint8), 0)
    # cool: B=255, G fades from ~200 (cyan) to 60 (blue) with prob
    rgba[..., 2] = np.where(spr, 255, 0)
    rgba[..., 1] = np.where(spr, (60 + (1.0 - p) * 140).astype(np.uint8), rgba[..., 1])
    a = np.clip(p * alpha_max, 0, alpha_max).astype(np.uint8)
    rgba[..., 3] = np.where((p > eps) & (regime > 0), a, 0).astype(np.uint8)
    return rgba


def mask_to_rgba(mask01, rgb=(255, 0, 255)):
    m = (np.asarray(mask01) > 0.5).astype(np.uint8)
    rgba = np.zeros((*m.shape, 4), np.uint8)
    for k in range(3):
        rgba[..., k] = m * rgb[k]
    rgba[..., 3] = m * 255
    return rgba


def source_bounds_xy(ds):
    x = np.asarray(ds.ds["x"].values); y = np.asarray(ds.ds["y"].values)
    return float(x.min()), float(y.min()), float(x.max()), float(y.max())


def compute_bounds(ds, cfg):
    xmin, ymin, xmax, ymax = source_bounds_xy(ds)
    if abs(xmin) <= 360 and abs(ymax) <= 180:
        return [[ymin, xmin], [ymax, xmax]]
    t = Transformer.from_crs(f"EPSG:{cfg.source_epsg}", "EPSG:4326", always_xy=True)
    lons, lats = [], []
    for xx, yy in [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]:
        lo, la = t.transform(xx, yy); lons.append(lo); lats.append(la)
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


@st.cache_resource(show_spinner=False)
def reproj_index(zarr_path: str, source_epsg: int):
    """Precompute the EPSG:3035→WGS84 nearest-neighbour mapping ONCE (grid geometry is fixed), via pyproj.

    Builds a regular WGS84 dest grid, inverse-transforms each dest pixel to source CRS, and snaps to the
    nearest source pixel (the cube is a regular grid, so this is a vectorised linear index). Each render
    then just GATHERS the new values through this map (numpy fancy-index, ~µs) — no per-render warp, and
    no rasterio/GDAL (pyproj's PROJ resolves EPSG cleanly where rasterio's bundled GDAL did not).
    Returns (idx[h,w] int with -1 = no source, dst_shape, bounds_latlon) or None on failure."""
    try:
        z = xr.open_zarr(zarr_path, consolidated=True)
        x = z["x"].values.astype(float); y = z["y"].values.astype(float)
        H, W = len(y), len(x)
        dx = (x[-1] - x[0]) / (W - 1); dy = (y[-1] - y[0]) / (H - 1)  # signed steps (orientation-safe)
        fwd = Transformer.from_crs(f"EPSG:{source_epsg}", "EPSG:4326", always_xy=True)
        clon, clat = fwd.transform([x[0], x[0], x[-1], x[-1]], [y[0], y[-1], y[0], y[-1]])
        lon_min, lon_max = min(clon), max(clon); lat_min, lat_max = min(clat), max(clat)
        out_h, out_w = H, W
        lons = np.linspace(lon_min, lon_max, out_w)
        lats = np.linspace(lat_max, lat_min, out_h)  # north-up (row 0 = lat_max)
        LON, LAT = np.meshgrid(lons, lats)
        inv = Transformer.from_crs("EPSG:4326", f"EPSG:{source_epsg}", always_xy=True)
        SX, SY = inv.transform(LON.ravel(), LAT.ravel())
        col = np.rint((np.asarray(SX) - x[0]) / dx).astype(np.int64)
        row = np.rint((np.asarray(SY) - y[0]) / dy).astype(np.int64)
        valid = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        idx = np.where(valid, row * W + col, -1).reshape(out_h, out_w)
        bounds = [[float(lat_min), float(lon_min)], [float(lat_max), float(lon_max)]]
        return idx, (out_h, out_w), bounds
    except Exception:
        return None


def apply_reproj(arr2d, idx):
    """Gather a per-date value array through the precomputed index map (fast; no rasterio call)."""
    flat = np.asarray(arr2d, np.float32).ravel()
    out = np.where(idx >= 0, flat[np.clip(idx, 0, flat.size - 1)], np.nan)
    return out.astype(np.float32)


def rgba_png(rgba):
    buf = io.BytesIO(); Image.fromarray(rgba, "RGBA").save(buf, "PNG"); return buf.getvalue()


def folium_map(png, bounds):
    (la0, lo0), (la1, lo1) = bounds
    m = folium.Map(location=[(la0 + la1) / 2, (lo0 + lo1) / 2], zoom_start=6, tiles="CartoDB Voyager")
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    folium.raster_layers.ImageOverlay(image=url, bounds=[[la0, lo0], [la1, lo1]], opacity=1.0,
                                      interactive=False, zindex=1).add_to(m)
    return m


# ---------------------------- UI ----------------------------
st.set_page_config(page_title="IberFire coarse4 — wide-deep", layout="wide")
st.title("Next-day wildfire risk — coarse4 (4 km) wide-and-deep model")
cfg = get_cfg()

with st.sidebar:
    st.subheader("Runtime")
    st.caption(f"model: {cfg.model_file}")
    st.caption(f"cube: {Path(cfg.zarr_path).name}  |  device: {cfg.device}")
    st.caption(f"regime adj: ign {cfg.adj_ignition}, spr {cfg.adj_spread}")
    view = st.radio("View", ["Probability (yellow→red)", "Ignition vs Spread (warm/cool)", "Label (fire t+1)"])
    stretch = st.radio("Scaling", ["p99_stretch", "raw"], index=0)
    use_reproj = st.checkbox("Accurate reprojection (EPSG:3035→WGS84)", value=True)

with st.spinner("Loading dataset + model (cached)..."):
    feats = load_feats(cfg.zarr_path)
    ds = load_dataset(cfg, tuple(feats))
    model = load_model(cfg, len(feats))
    bounds0 = compute_bounds(ds, cfg)

dates = [str(ds.get_time_value(i))[:10] for i in range(len(ds))]
c1, c2 = st.columns([3, 1])
date = c1.selectbox("Date (features at t → fire risk at t+1)", dates, index=min(len(dates) - 1, len(dates) // 2))
go = c2.button("Render map", use_container_width=True)

if go:
    idx = dates.index(date)
    prob, y, reg = predict_cached(model, ds, cfg, cfg.model_file, idx, cfg.adj_ignition, cfg.adj_spread)
    n_ign = int((reg == 1).sum()); n_spr = int((reg == 2).sum()); n_fire = int((y == 1).sum())
    st.caption(f"land cells: {int((reg>0).sum())} | ignition-regime {n_ign}, spread-regime {n_spr} | actual fire@t+1: {n_fire} | max prob {prob[reg>0].max():.3f}")

    # Reproject by gathering through the ONCE-precomputed index map (no per-render rasterio warp).
    bounds = bounds0
    mapping = reproj_index(cfg.zarr_path, cfg.source_epsg) if use_reproj else None
    if mapping is not None:
        idxmap, _, bounds = mapping
        prob = apply_reproj(prob, idxmap)
        reg = np.rint(np.nan_to_num(apply_reproj(reg.astype(np.float32), idxmap), nan=0.0)).astype(int)
        y = (np.nan_to_num(apply_reproj((y > 0.5).astype(np.float32), idxmap), nan=0.0) > 0.5).astype(np.float32)

    if view.startswith("Ignition"):
        rgba = regime_prob_to_rgba(prob, reg, stretch)
    elif view.startswith("Label"):
        rgba = mask_to_rgba((y > 0.5))
    else:
        rgba = probs_to_rgba(prob, stretch)

    st_folium(folium_map(rgba_png(rgba), bounds), width=1100, height=650)
    if view.startswith("Ignition"):
        st.caption("🟧 warm = new-ignition risk (far from active fire) · 🟦 cool = spread risk (near active fire). Intensity ∝ predicted probability.")
