"""Measurement floor + predictive-potential analysis on the coarse cube.

The master unblocker (ROADMAP §A / sequence step 4). Pixel-level next-day fire
prediction on a 3-way TEMPORAL split with a touched-once test set:

  train 2008-2018 | val 2019-2021 | test 2022-2024   (features at t -> is_fire at t+1)

Baselines (the U-Net must eventually beat these): FWI-alone, logistic regression,
HistGradientBoosting. Metrics: PR-AUC + ROC-AUC, reported OVERALL and split into
NEW-IGNITION vs CONTINUATION positives (continuation = fire within ~1 cell at t,
via dist_to_fire). Plus permutation feature-importance -> the pruning shortlist.

NOT a model experiment — a grounded baseline + feature ranking. Writes a JSON report
to reports/. Read-only on the cube.

Usage:  python scripts/measurement_floor.py [--smoke] [--neg-ratio 15] [--max-train 300000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

project_root = Path(__file__).resolve().parents[1]
CUBE = project_root / "data" / "gold" / "IberFire_coarse4.zarr"
REPORTS = project_root / "reports"

SPLITS = {
    "train": ("2008-01-01", "2018-12-31"),
    "val": ("2019-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2024-12-31"),
}


def select_features(c: xr.Dataset) -> list[str]:
    """Curated numeric feature list: drop label/masks/coords/categorical, keep one CLC year."""
    drop_exact = {"is_fire", "is_sea", "is_spain", "AutonomousCommunities"}
    feats = []
    for v in c.data_vars:
        if v in drop_exact:
            continue
        if v.startswith("CLC_2006") or v.startswith("CLC_2012"):
            continue  # keep only CLC_2018_* to avoid 3x year redundancy
        if v.startswith("popdens_") and v != "popdens_2020":
            continue  # keep one recent year
        feats.append(v)
    return sorted(feats)


def build_samples(c, is_fire, dist_to_fire, t0, t1, neg_ratio, max_rows, rng, cell_km, smoke):
    """Return (t_idx, i_idx, j_idx, y, continuation) sample arrays for a split."""
    time = c["time"].values
    in_split = np.where((time >= np.datetime64(t0)) & (time <= np.datetime64(t1)))[0]
    in_split = in_split[in_split < len(time) - 1]  # need t+1
    if smoke:
        in_split = in_split[::20]
    land = np.isfinite(c["t2m_mean"].isel(time=int(in_split[0])).values)  # (y,x) land mask

    pos_t, pos_i, pos_j, neg_t, neg_i, neg_j = [], [], [], [], [], []
    for t in in_split:
        lbl = is_fire[t + 1]
        pi, pj = np.where((lbl > 0) & land)
        pos_t.append(np.full(pi.size, t)); pos_i.append(pi); pos_j.append(pj)
        ni, nj = np.where((lbl == 0) & land)
        if ni.size:
            k = min(ni.size, max(1, neg_ratio * max(pi.size, 1)))
            sel = rng.choice(ni.size, size=k, replace=False)
            neg_t.append(np.full(k, t)); neg_i.append(ni[sel]); neg_j.append(nj[sel])

    pt, pi_, pj_ = np.concatenate(pos_t), np.concatenate(pos_i), np.concatenate(pos_j)
    nt, ni_, nj_ = np.concatenate(neg_t), np.concatenate(neg_i), np.concatenate(neg_j)

    def cap(a_t, a_i, a_j, n):
        if a_t.size <= n:
            return a_t, a_i, a_j
        s = rng.choice(a_t.size, size=n, replace=False)
        return a_t[s], a_i[s], a_j[s]

    n_pos_cap = max_rows // (neg_ratio + 1)
    pt, pi_, pj_ = cap(pt, pi_, pj_, n_pos_cap)
    nt, ni_, nj_ = cap(nt, ni_, nj_, max_rows - pt.size)

    t_idx = np.concatenate([pt, nt]); i_idx = np.concatenate([pi_, ni_]); j_idx = np.concatenate([pj_, nj_])
    y = np.concatenate([np.ones(pt.size), np.zeros(nt.size)]).astype("int8")
    d_at_t = dist_to_fire[t_idx, i_idx, j_idx]
    continuation = (d_at_t <= 1.5 * cell_km)
    return t_idx.astype("int64"), i_idx.astype("int64"), j_idx.astype("int64"), y, continuation


def extract_matrix(c, feats, t_idx, i_idx, j_idx):
    """Vectorized point extraction -> (n_samples, n_feats) float32."""
    X = np.empty((t_idx.size, len(feats)), dtype="float32")
    sel_t = xr.DataArray(t_idx, dims="s"); sel_i = xr.DataArray(i_idx, dims="s"); sel_j = xr.DataArray(j_idx, dims="s")
    for k, v in enumerate(feats):
        da = c[v]
        if set(da.dims) >= {"time", "y", "x"}:
            col = da.isel(time=sel_t, y=sel_i, x=sel_j).values
        elif set(da.dims) == {"y", "x"}:
            col = da.isel(y=sel_i, x=sel_j).values
        elif set(da.dims) == {"time"}:
            col = da.isel(time=sel_t).values
        else:
            col = np.full(t_idx.size, np.nan)
        X[:, k] = col
        if k % 25 == 0:
            print(f"    extracted {k}/{len(feats)}", flush=True)
    return X


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--neg-ratio", type=int, default=15)
    ap.add_argument("--max-train", type=int, default=300_000)
    args = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)

    c = xr.open_zarr(CUBE, consolidated=True)
    cell_km = abs(float(c["x"].values[1] - c["x"].values[0])) / 1000.0
    feats = select_features(c)
    if args.smoke:
        feats = ["FWI", "VPD_peak", "kbdi", "ffwi", "dist_to_fire", "NDVI", "t2m_max",
                 "RH_min", "days_since_rain", "elevation_mean"]
    print(f"[floor] {len(feats)} features, cube {dict(c.sizes)}, cell {cell_km:.0f} km, smoke={args.smoke}", flush=True)

    is_fire = c["is_fire"].values
    dist_to_fire = c["dist_to_fire"].values
    max_rows = {"train": args.max_train, "val": args.max_train // 3, "test": args.max_train // 3}

    data = {}
    for split, (t0, t1) in SPLITS.items():
        ti, ii, jj, y, cont = build_samples(c, is_fire, dist_to_fire, t0, t1,
                                             args.neg_ratio, max_rows[split], rng, cell_km, args.smoke)
        X = extract_matrix(c, feats, ti, ii, jj)
        data[split] = dict(X=X, y=y, cont=cont,
                           fwi=X[:, feats.index("FWI")] if "FWI" in feats else np.zeros_like(y, float))
        print(f"[floor] {split}: {y.size} rows, {int(y.sum())} pos ({y.mean():.3%}), "
              f"continuation {cont[y == 1].mean():.0%} of pos", flush=True)

    Xtr, ytr = data["train"]["X"], data["train"]["y"]
    Xte, yte, cont_te = data["test"]["X"], data["test"]["y"], data["test"]["cont"]

    def split_ap(y, score, cont):
        pos = y == 1
        out = {"overall": float(average_precision_score(y, score)),
               "roc_auc": float(roc_auc_score(y, score)),
               "base_rate": float(y.mean())}
        for name, mask_pos in [("new_ignition", pos & ~cont), ("continuation", pos & cont)]:
            keep = mask_pos | (y == 0)
            out[name + "_ap"] = (float(average_precision_score(y[keep], score[keep]))
                                 if mask_pos.sum() > 0 else None)
            out[name + "_n"] = int(mask_pos.sum())
        return out

    report = {"split_dates": SPLITS, "n_features": len(feats), "features": feats,
              "test_rows": int(yte.size), "test_pos": int(yte.sum()),
              "test_continuation_frac": float(cont_te[yte == 1].mean()), "baselines": {}}

    report["baselines"]["fwi_alone"] = split_ap(yte, np.nan_to_num(data["test"]["fwi"]), cont_te)

    mean = np.nanmean(Xtr, axis=0)
    Xtr_i = np.where(np.isfinite(Xtr), Xtr, mean); Xte_i = np.where(np.isfinite(Xte), Xte, mean)
    sc = StandardScaler().fit(Xtr_i)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    lr.fit(sc.transform(Xtr_i), ytr)
    report["baselines"]["logreg"] = split_ap(yte, lr.predict_proba(sc.transform(Xte_i))[:, 1], cont_te)

    gbt = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         class_weight="balanced", random_state=0,
                                         validation_fraction=0.1, early_stopping=True)
    gbt.fit(Xtr, ytr)
    report["baselines"]["hist_gbt"] = split_ap(yte, gbt.predict_proba(Xte)[:, 1], cont_te)

    Xv, yv = data["val"]["X"], data["val"]["y"]
    n_imp = min(20000, yv.size)
    si = rng.choice(yv.size, size=n_imp, replace=False)
    pi = permutation_importance(gbt, Xv[si], yv[si], scoring="average_precision",
                                n_repeats=3, random_state=0, n_jobs=-1)
    order = np.argsort(pi.importances_mean)[::-1]
    report["feature_importance"] = [
        {"feature": feats[k], "importance": float(pi.importances_mean[k]),
         "std": float(pi.importances_std[k])} for k in order
    ]

    out = REPORTS / ("measurement_floor_smoke.json" if args.smoke else "measurement_floor.json")
    out.write_text(json.dumps(report, indent=2))
    print("\n===== BASELINE PR-AUC (test) =====", flush=True)
    for name, m in report["baselines"].items():
        print(f"  {name:10} overall={m['overall']:.4f}  new-ign={m['new_ignition_ap']}  "
              f"contin={m['continuation_ap']}  roc={m['roc_auc']:.3f}", flush=True)
    print(f"  (test base rate = {report['baselines']['fwi_alone']['base_rate']:.4%})", flush=True)
    print("\n===== TOP 20 FEATURES (permutation importance) =====", flush=True)
    for r in report["feature_importance"][:20]:
        print(f"  {r['importance']:+.4f} ± {r['std']:.4f}   {r['feature']}", flush=True)
    print(f"\n[floor] report -> {out}", flush=True)


if __name__ == "__main__":
    main()
