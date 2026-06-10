"""FGDC ablation harness — rigorous, reproducible with/without studies for the ablation log (ABLATIONS.md).

Principle: toggle ONE thing, hold everything else fixed (same days, same split, same model), and report the
metric delta. Two modes:
  * --groups   leave-one-feature-GROUP-out: train full, then retrain dropping each group → ΔAP, ΔROC.
  * --horizons sweep the next-{1,3,7}-day target at full features (target-definition ablation).
Both print a markdown table ready to paste into ABLATIONS.md (+ optional --json out).

Eval: per-cell next-day(or within-N-day) fire from features(t); HistGBT (class_weight balanced); chronological
80/20 split; metric = average precision (the rare-event metric) + ROC-AUC. HistGBT handles NaN natively.

NOTE: on the tiny dev slice results are DIRECTIONAL; the same harness runs on the full backfill for final
numbers. This is a methodology tool — every major feature/source/target change gets an entry in ABLATIONS.md.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.ingest import grid

GOLD = grid.ROOT / "data" / "gold" / "FireGuard_coarse4.zarr"
DYN = ["t2m_mean", "t2m_max", "t2m_min", "t2m_range", "RH_mean", "RH_min", "RH_max",
       "surface_pressure_mean", "wind_speed_mean", "wind_speed_max", "wind_u_mean", "wind_v_mean",
       "total_precipitation_mean", "soil_moisture_mean", "soil_temperature_mean",
       "NDVI", "EVI", "LAI", "FAPAR", "LST", "popdens", "built_s"]   # popdens/built_s = GHS, interpolated daily
STAT = ["elevation_mean", "slope_mean", "dist_to_roads_mean",
        "CLC_2018_forest_and_semi_natural_proportion", "CLC_2018_scrub_proportion",
        "CLC_2018_artificial_proportion"]

# feature → group (for leave-one-group-out). dist_to_fire is computed inline.
def _group(name):
    if name in ("NDVI", "EVI", "LAI", "FAPAR", "LST"):
        return "vegetation"
    if name == "dist_to_fire":
        return "fire_context"
    if name.startswith(("t2m", "RH", "surface_pressure", "wind", "total_precip")):
        return "weather"
    if name.startswith("soil_"):
        return "soil_moisture"
    if name in ("popdens", "built_s", "dist_to_roads_mean") or name.endswith("artificial_proportion"):
        return "human"
    if name.startswith("CLC_") or name.endswith("_proportion"):
        return "fuel_cover"
    if name.startswith(("elevation", "slope", "roughness", "aspect")):
        return "terrain"
    return "other"


def _build(z, horizon, dyn, stat):
    isf = z["is_fire"].values; T, H, W = isf.shape
    land = (np.nan_to_num(z["is_spain"].values) > 0.5) if "is_spain" in z else np.ones((H, W), bool)
    Sblk = np.stack([z[v].values.astype(np.float32) for v in stat], -1) if stat else np.zeros((H, W, 0), np.float32)
    tmax = T - 1 - horizon
    X, y = [], []
    for t in range(tmax + 1):
        ft = isf[t] > 0.5
        d2f = (distance_transform_edt(~ft) * 4.0) if ft.any() else np.full((H, W), 1e3, np.float32)
        dyn_t = np.stack([z[v].isel(time=t).values.astype(np.float32) for v in dyn], -1)
        feat = np.concatenate([dyn_t, d2f[..., None], Sblk], -1)[land]
        X.append(feat)
        y.append((isf[t + 1:t + 1 + horizon] > 0.5).any(0).astype(np.int8)[land])
    names = dyn + ["dist_to_fire"] + stat
    return np.concatenate(X), np.concatenate(y), names, (tmax + 1)


def _fit_eval(X, y, n_pairs, cols=None):
    Xc = X if cols is None else X[:, cols]
    n_per = X.shape[0] // n_pairs; cut = int(n_pairs * 0.8) * n_per
    gbt = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=63,
                                         l2_regularization=1.0, class_weight="balanced", random_state=0)
    gbt.fit(Xc[:cut], y[:cut])
    Xv, yv = Xc[cut:], y[cut:]
    if yv.sum() == 0 or yv.min() == yv.max():
        return None, None
    p = gbt.predict_proba(Xv)[:, 1]
    return float(average_precision_score(yv, p)), float(roc_auc_score(yv, p))


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("fgdc_ablation")
    a = sys.argv
    cube = a[a.index("--cube") + 1] if "--cube" in a else GOLD
    horizon = int(a[a.index("--horizon") + 1]) if "--horizon" in a else 3
    z = xr.open_zarr(str(cube), consolidated=True)
    dyn = [v for v in DYN if v in z]; stat = [v for v in STAT if v in z]
    X, y, names, n_pairs = _build(z, horizon, dyn, stat)
    grp = np.array([_group(n) for n in names])
    log.info(f"cube={Path(cube).name} horizon={horizon}d rows={X.shape[0]:,} feats={len(names)} pos={int(y.sum())}")

    out = {"cube": Path(cube).name, "horizon": horizon, "rows": int(X.shape[0]), "pos": int(y.sum())}
    if "--horizons" in a:
        rows = []
        for h in (1, 3, 7):
            Xh, yh, _, nph = _build(z, h, dyn, stat)
            ap, roc = _fit_eval(Xh, yh, nph)
            rows.append((h, float(yh.mean()), ap, roc))
        out["horizons"] = rows
        print("\n### Target-horizon ablation (full features)\n")
        print("| target | pos-rate | val AP | val ROC-AUC |\n|---|---|---|---|")
        for h, pr, ap, roc in rows:
            print(f"| within {h}d | {pr*100:.3f}% | {ap:.3f} | {roc:.3f} |")

    if "--groups" in a or "--horizons" not in a:
        ap_full, roc_full = _fit_eval(X, y, n_pairs)
        groups = sorted(set(grp) - {"other"})
        res = []
        for g in groups:
            keep = np.where(grp != g)[0]
            ap, roc = _fit_eval(X, y, n_pairs, cols=keep)
            res.append((g, int((grp == g).sum()), ap, roc, ap_full - (ap or 0), roc_full - (roc or 0)))
        res.sort(key=lambda r: -r[4])
        out["full"] = {"AP": ap_full, "ROC": roc_full}
        out["groups"] = res
        print(f"\n### Leave-one-group-out ablation (within-{horizon}d target)\n")
        print(f"Full model: **AP {ap_full:.3f}, ROC-AUC {roc_full:.3f}**\n")
        print("| dropped group | #feats | AP without | ΔAP | ROC without | ΔROC |\n|---|---|---|---|---|---|")
        for g, n, ap, roc, dap, droc in res:
            print(f"| {g} | {n} | {ap:.3f} | **{dap:+.3f}** | {roc:.3f} | {droc:+.3f} |")
        print("\n(ΔAP = full − without; larger positive = the group contributes more.)")

    if "--json" in a:
        Path(a[a.index("--json") + 1]).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
