"""Per-regime permutation importance for the FGDC v2 GBT → models/gbt_fireguard[_TAG].importance.json.

What most drives tomorrow's risk, split by regime (ignition = no fire within REGIME_KM today; spread = fire
within). HistGBT exposes no native feature_importances_, so we use permutation importance = average-precision
DROP when a feature is shuffled, computed SEPARATELY on the ignition- and spread-regime rows of the held-out
val split. AP is rank-based → the monotonic isotonic calibrator wouldn't change ranking; use the raw GBT.

NOTE: the Space no longer reads this — danger-area causes come from LIVE per-cell drivers (serve.day_drivers +
CAUSE_MAP; the stale v1 fallback was removed in the 2026-07-07 audit). This is an ANALYSIS artifact (sanity /
paper / scorecard — e.g. it independently surfaces whether kbdi is a real driver). Factor-aware; auto-tags like
train_gbt/calibrate; recompute on retrain.

  python scripts/gbt_importance.py --factor 2
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import xarray as xr
from sklearn.inspection import permutation_importance

from src.data import metrics as M

PROJECT = M.project_root
CUBE = PROJECT / "data" / "gold" / "FireGuard_coarse4_t200.zarr"
REGIME_KM = 6.0
HORIZON = 1
TOPK = 8
NEG_PER_POS = 30       # keep ALL (rare) positives, subsample negatives → AP stays non-degenerate
N_DAYS = 45            # strided held-out val days (bounds the collected matrix on the 16 GB box)


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("gbt_importance")
    factor = int(sys.argv[sys.argv.index("--factor") + 1]) if "--factor" in sys.argv else 4
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
    if factor != 4 and not tag:
        tag = f"{factor}km"
        log.info(f"--factor {factor} without --tag → auto-tagging '{tag}'")
    cube_path = CUBE if factor == 4 else PROJECT / "data" / "gold" / f"FireGuard_coarse{factor}.zarr"
    slug = f"gbt_fireguard_{tag}" if tag else "gbt_fireguard"
    BLK = max(12, int(200 * (factor / 4) ** 2))            # area-scaled block reads (matches train/calibrate)

    art = joblib.load(PROJECT / "models" / f"{slug}.joblib")
    gbt, feats = art["model"], art["features"]             # model's exact feature list + order
    z = xr.open_zarr(str(cube_path), consolidated=True)
    isf = z["is_fire"].values
    Tn = isf.shape[0]
    land = np.nan_to_num(z["is_spain"].values) > 0.5
    dynamic = [f for f in feats if "time" in z[f].dims]; dyn_set = set(dynamic)
    stat = [f for f in feats if "time" not in z[f].dims]
    stat_vals = {f: z[f].values.astype(np.float32)[land] for f in stat}

    tmax = Tn - 1 - HORIZON
    cut = int((tmax + 1) * 0.8)                            # IDENTICAL split to train_gbt/calibrate → held-out
    val_days = list(range(cut, tmax + 1))
    stride = max(1, len(val_days) // N_DAYS)
    days = val_days[::stride]
    log.info(f"{slug} @ factor {factor}: {len(days)} val days (stride {stride}), {int(land.sum())} land cells")

    def build_feat(block, lt):                             # raw features, exactly as the model was trained
        dv = {f: block[f].isel(time=lt).values.astype(np.float32)[land] for f in dynamic}
        return np.stack([dv[f] if f in dyn_set else stat_vals[f] for f in feats], -1)

    by_block = {}
    for t in days:
        by_block.setdefault((t // BLK) * BLK, []).append(t)
    Xs, ys, rs = [], [], []
    for b0 in sorted(by_block):
        block = z[dynamic].isel(time=slice(b0, b0 + BLK + HORIZON)).load()
        for t in by_block[b0]:
            lt = t - b0
            Xs.append(build_feat(block, lt))
            ys.append(isf[t + HORIZON][land].astype(np.int8))
            rs.append(np.where(block["dist_to_fire"].isel(time=lt).values[land] <= REGIME_KM, 2, 1).astype(np.int8))
    X = np.concatenate(Xs); y = np.concatenate(ys); reg = np.concatenate(rs)
    log.info(f"collected {X.shape[0]} rows ({X.shape[1]} feats), pos rate {y.mean():.4f}")

    rng = np.random.default_rng(0)
    out = {"model": slug, "factor": factor, "features": list(feats), "regimes": {}}
    for name, code in [("ignition", 1), ("spread", 2)]:
        m = reg == code
        Xr, yr = X[m], y[m]
        if yr.sum() < 5:
            log.warning(f"{name}: only {int(yr.sum())} positives — skipping"); continue
        pos = np.where(yr == 1)[0]; neg = np.where(yr == 0)[0]
        k = min(neg.size, NEG_PER_POS * pos.size)
        sel = np.concatenate([pos, rng.choice(neg, k, replace=False)])
        Xr, yr = Xr[sel], yr[sel]
        pi = permutation_importance(gbt, Xr, yr, scoring="average_precision", n_repeats=4, random_state=0, n_jobs=-1)
        order = np.argsort(pi.importances_mean)[::-1][:TOPK]
        top = [{"feature": feats[j], "drop": float(pi.importances_mean[j]), "std": float(pi.importances_std[j])}
               for j in order]
        out["regimes"][name] = {"n_rows": int(Xr.shape[0]), "n_pos": int(yr.sum()), "top": top}
        log.info(f"{name} ({int(yr.sum())} pos / {Xr.shape[0]} rows) top: "
                 + ", ".join(f"{t['feature']}({t['drop']:.3f})" for t in top[:5]))
    p = PROJECT / "models" / f"{slug}.importance.json"
    p.write_text(json.dumps(out, indent=2))
    log.info(f"wrote {p}")


if __name__ == "__main__":
    main()
