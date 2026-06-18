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


def _build(z, horizon, dyn, stat, neg_ratio=30, seed=0):
    """Chronological 80/20 split built per-timestep. TRAIN negatives are subsampled to neg_ratio:1 (all
    positives kept) so a multi-year cube doesn't materialize ~50M rows (which OOMs); VAL keeps FULL
    prevalence so val AP stays honest and comparable to prior full-prevalence ABLATIONS entries.
    Returns (Xtr, ytr, Xval, yval, names)."""
    isf = z["is_fire"].values; T, H, W = isf.shape
    land = (np.nan_to_num(z["is_spain"].values) > 0.5) if "is_spain" in z else np.ones((H, W), bool)
    Sblk = np.stack([z[v].values.astype(np.float32) for v in stat], -1) if stat else np.zeros((H, W, 0), np.float32)
    tmax = T - 1 - horizon
    cut_day = int((tmax + 1) * 0.8)
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xval, yval, regval = [], [], [], [], []
    for t in range(tmax + 1):
        ft = isf[t] > 0.5
        d2f = (distance_transform_edt(~ft) * 4.0) if ft.any() else np.full((H, W), 1e3, np.float32)
        dyn_t = np.stack([z[v].isel(time=t).values.astype(np.float32) for v in dyn], -1)
        feat = np.concatenate([dyn_t, d2f[..., None], Sblk], -1)[land]
        yt = (isf[t + 1:t + 1 + horizon] > 0.5).any(0).astype(np.int8)[land]
        if t < cut_day:                              # TRAIN — keep all positives, subsample negatives
            pos = np.where(yt == 1)[0]; neg = np.where(yt == 0)[0]
            if neg.size > neg_ratio * pos.size:
                neg = rng.choice(neg, neg_ratio * max(pos.size, 1), replace=False)
            keep = np.concatenate([pos, neg])
            Xtr.append(feat[keep]); ytr.append(yt[keep])
        else:                                        # VAL — full prevalence (honest AP)
            Xval.append(feat); yval.append(yt)
            regval.append(np.where(d2f[land] < 6.0, 2, 1).astype(np.int8))  # 1=far, 2=near fire
    names = dyn + ["dist_to_fire"] + stat
    return (np.concatenate(Xtr), np.concatenate(ytr),
            np.concatenate(Xval), np.concatenate(yval), np.concatenate(regval), names)


def _fit_eval(Xtr, ytr, Xval, yval, cols=None, return_pred=False):
    Xt = Xtr if cols is None else Xtr[:, cols]
    Xv = Xval if cols is None else Xval[:, cols]
    if yval.sum() == 0 or yval.min() == yval.max():
        return None, None
    gbt = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=63,
                                         l2_regularization=1.0, class_weight="balanced", random_state=0)
    gbt.fit(Xt, ytr)
    p = gbt.predict_proba(Xv)[:, 1]
    ap, roc = average_precision_score(yval, p), roc_auc_score(yval, p)
    return (ap, roc, p) if return_pred else (ap, roc)

def _regime_eval(prob, yval, regval, neg_ratio=15, seed=0):
    rng = np.random.default_rng(seed); neg_p = prob[yval==0]
    def matched_ap(mask):
        k = min(neg_p.size, neg_ratio * int(mask.sum()))
        sel = rng.choice(neg_p.size, k, replace=False
                         )
        return average_precision_score(np.r_[np.ones(int(mask.sum())), np.zeros(k)], np.r_[prob[mask], neg_p[sel]])
    return {"new_ign_ap": matched_ap((yval==1) & (regval==1)),
            "spread_ap": matched_ap((yval==1) & (regval==2)),
            "roc": roc_auc_score(yval, prob)
            }

def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("fgdc_ablation")
    a = sys.argv
    cube = a[a.index("--cube") + 1] if "--cube" in a else GOLD
    horizon = int(a[a.index("--horizon") + 1]) if "--horizon" in a else 3
    regime = "--regime" in a

    z = xr.open_zarr(str(cube), consolidated=True)
    dyn = [v for v in DYN if v in z]; stat = [v for v in STAT if v in z]
    Xtr, ytr, Xval, yval, regval, names = _build(z, horizon, dyn, stat)
    grp = np.array([_group(n) for n in names])
    log.info(f"cube={Path(cube).name} horizon={horizon}d train_rows={Xtr.shape[0]:,} val_rows={Xval.shape[0]:,} "
             f"feats={len(names)} pos_val={int(yval.sum())}")

    out = {"cube": Path(cube).name, "horizon": horizon, "train_rows": int(Xtr.shape[0]),
           "val_rows": int(Xval.shape[0]), "pos_val": int(yval.sum())}
    if "--horizons" in a:
        rows = []
        for h in (1, 3, 7):
            Xt2, yt2, Xv2, yv2, _, _ = _build(z, h, dyn, stat)
            ap, roc = _fit_eval(Xt2, yt2, Xv2, yv2)
            rows.append((h, float(yv2.mean()), ap, roc))
        out["horizons"] = rows
        print("\n### Target-horizon ablation (full features)\n")
        print("| target | pos-rate | val AP | val ROC-AUC |\n|---|---|---|---|")
        for h, pr, ap, roc in rows:
            print(f"| within {h}d | {pr*100:.3f}% | {ap:.3f} | {roc:.3f} |")

    if "--groups" in a or "--horizons" not in a:
        ap_full, roc_full = _fit_eval(Xtr, ytr, Xval, yval)
        groups = sorted(set(grp) - {"other"})
        res = []
        for g in groups:
            keep = np.where(grp != g)[0]
            ap, roc = _fit_eval(Xtr, ytr, Xval, yval, cols=keep)
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

    if regime:
        _, _, prob = _fit_eval(Xtr, ytr, Xval, yval, return_pred=True)
        r = _regime_eval(prob, yval, regval)
        print(f"\n### Regime split (held-out val, matched prevalence)\n"
          f"new-ignition AP = {r['new_ign_ap']:.3f}  (v1 bar ≈ 0.63)\n"
          f"spread AP       = {r['spread_ap']:.3f}\n"
          f"ROC             = {r['roc']:.3f}")

    if "--json" in a:
        Path(a[a.index("--json") + 1]).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
