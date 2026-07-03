"""Calibrate the production FGDC GBT's probabilities for true-prevalence risk maps.

The GBT (`models/gbt_fireguard.joblib`) is trained on NEG-subsampled cells (~3% positive), so `predict_proba`
is calibrated to that inflated prevalence — far above the true rate (~0.005% new-ign / ~6% spread over all
land cells). For a risk product "0.3" must mean ~30% burn frequency. We fit **isotonic regression** on the
TRUE-prevalence held-out VAL set (raw GBT prob → observed label): it learns the correction directly, is
monotonic (so ranking / AP / ROC are unchanged), and needs no parametric form.

v2-native: builds the eval set the way the model was actually trained — **raw features, block-read, the model's
own chronological held-out split** (no U-Net `scripts/train.py` / torch, no normalization, no `_dyn` cube; the
v1 `calibrate_gbt.py` coupled to all three and was wrong for the GBT). The held-out val is split chronologically
into a calibration half (fit isotonic) and an eval half (honest before/after ECE).

Output: models/gbt_fireguard.calibrator.joblib + models/gbt_fireguard.calibration.json.  Use --smoke for a fast check.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import joblib
import numpy as np
import xarray as xr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

CUBE = PROJECT / "data" / "gold" / "FireGuard_coarse4_t200.zarr"   # the model's training cube (block-read friendly)
MODEL = PROJECT / "models" / "gbt_fireguard.joblib"
HORIZON = 1


def reliability(prob, y, bins=12):
    """Binned reliability + ECE + Brier. Log-spaced low-end bins (signal lives at low prob)."""
    edges = np.concatenate([[0], np.geomspace(1e-4, 1.0, bins)])
    idx = np.clip(np.digitize(prob, edges) - 1, 0, len(edges) - 2)
    rows, ece, N = [], 0.0, len(prob)
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        mp, fp = float(prob[m].mean()), float(y[m].mean())
        ece += n / N * abs(mp - fp)
        rows.append((edges[b], edges[b + 1], n, mp, fp))
    brier = float(np.mean((prob - y) ** 2))
    return rows, ece, brier


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("calibrate")
    smoke = "--smoke" in sys.argv
    rng = np.random.default_rng(0)

    art = joblib.load(MODEL)
    gbt, feats = art["model"], art["features"]            # the model's exact feature list + order
    z = xr.open_zarr(str(CUBE), consolidated=True)
    isf = z["is_fire"].values
    Tn = isf.shape[0]
    land = np.nan_to_num(z["is_spain"].values) > 0.5
    dynamic = [f for f in feats if "time" in z[f].dims]; dyn_set = set(dynamic)
    stat = [f for f in feats if "time" not in z[f].dims]
    stat_vals = {f: z[f].values.astype(np.float32)[land] for f in stat}

    tmax = Tn - 1 - HORIZON
    cut = int((tmax + 1) * 0.8)                            # IDENTICAL split to train_gbt → genuine held-out
    val_days = list(range(cut, tmax + 1))
    stride = max(1, len(val_days) // (40 if smoke else 320))   # true-prevalence DAY sample (bounds runtime; full prevalence per day)
    val_days = val_days[::stride]
    n_cal = int(len(val_days) * 0.6)
    cal_days, test_days = val_days[:n_cal], val_days[n_cal:]    # chronological cal / eval within the held-out val
    log.info(f"{Tn} days; held-out val > day {cut}; fit on {len(cal_days)} days, eval on {len(test_days)} (stride {stride})")

    def build_feat(block, lt):                            # raw features, exactly as the model was trained
        dv = {f: block[f].isel(time=lt).values.astype(np.float32)[land] for f in dynamic}
        return np.stack([dv[f] if f in dyn_set else stat_vals[f] for f in feats], -1)

    def collect(days):                                    # per-day predict over ALL land cells (true prevalence), block-read
        by_block = {}
        for t in days:
            by_block.setdefault((t // 200) * 200, []).append(t)
        ps, ys = [], []
        for b0 in sorted(by_block):
            block = z[dynamic].isel(time=slice(b0, b0 + 200)).load()
            for t in by_block[b0]:
                ps.append(gbt.predict_proba(build_feat(block, t - b0))[:, 1])
                ys.append((isf[t + 1] > 0.5).astype(np.int8)[land])
        return np.concatenate(ps), np.concatenate(ys)

    log.info("collecting calibration probs (true prevalence)...")
    pc, yc = collect(cal_days)
    log.info(f"cal cells {pc.size}, pos rate {yc.mean():.6f}")
    pos = np.where(yc == 1)[0]; neg = np.where(yc == 0)[0]
    keep = np.concatenate([pos, rng.choice(neg, min(neg.size, 4_000_000), replace=False)])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(pc[keep], yc[keep])

    log.info("collecting held-out eval probs...")
    pt, yt = collect(test_days)
    pt_cal = iso.predict(pt)
    for nm, p in (("RAW", pt), ("CALIBRATED", pt_cal)):
        rows, ece, brier = reliability(p, yt)
        log.info(f"--- eval reliability [{nm}] ECE={ece:.5f} Brier={brier:.6f} ---")
        for lo, hi, n, mp, fp in rows:
            log.info(f"   p[{lo:.4f},{hi:.4f}) n={n:>8} mean_pred={mp:.4f} obs_freq={fp:.4f}")
    log.info(f"ranking preserved: AP raw={average_precision_score(yt, pt):.4f} cal={average_precision_score(yt, pt_cal):.4f} "
             f"| ROC raw={roc_auc_score(yt, pt):.4f} cal={roc_auc_score(yt, pt_cal):.4f}")

    joblib.dump(iso, PROJECT / "models" / "gbt_fireguard.calibrator.joblib")
    _, ece_r, brier_r = reliability(pt, yt); _, ece_c, brier_c = reliability(pt_cal, yt)
    (PROJECT / "models" / "gbt_fireguard.calibration.json").write_text(json.dumps({
        "method": "isotonic on true-prevalence held-out VAL (corrects neg-subsample bias); v2 raw-feature block-read",
        "model": "gbt_fireguard", "n_features": len(feats), "cube": str(CUBE),
        "eval_ece_raw": ece_r, "eval_ece_calibrated": ece_c,
        "eval_brier_raw": brier_r, "eval_brier_calibrated": brier_c,
        "eval_ap_raw": float(average_precision_score(yt, pt)), "eval_ap_calibrated": float(average_precision_score(yt, pt_cal)),
    }, indent=2))
    log.info(f"saved gbt_fireguard.calibrator.joblib + .calibration.json (ECE {ece_r:.5f} → {ece_c:.5f})")


if __name__ == "__main__":
    main()
