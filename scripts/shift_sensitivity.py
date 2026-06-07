"""Would the AEMET live feed break the model? Key-free sensitivity test.

We measured the ERA5↔AEMET shift per variable (temperature LOW ~1-2°C, precip HIGH ~0.7mm/h). This perturbs
the TEST meteo features by that measured shift (Gaussian noise, std = 1.25*MAE_physical / feature_std, applied
in the model's normalized space) and re-scores the GBT's new-ignition AP. It answers, without any API key,
which feeds are safe to swap and whether precip-driven degradation actually hurts predictions.

Caveat: antecedent precip features (precip_sum_*, kbdi) are DERIVED from a precip history; we perturb them by
the same relative shift as a first-order estimate. A true number needs the live backtest (AEMET-fed pipeline).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import xarray as xr
import scripts.train as T
from src.data.features import build_segmentation_features

# measured ERA5↔AEMET physical MAE (from the 758-station upstream validation)
AEMET_MAE = {"t2m_mean": 1.21, "t2m_min": 1.62, "t2m_max": 1.85,
             "wind_speed": 0.89, "surface_pressure_max": 9.7, "surface_pressure_min": 9.5, "tp": 0.74}
PRECIP_REL = 0.23  # precip normalized-MAE (fraction of range) — applied to precip-derived antecedents


def feat_group(name):
    n = name.lower()
    if n.startswith("t2m"):
        return "temperature"
    if "wind" in n:
        return "wind"
    if "surface_pressure" in n or n == "sp":
        return "pressure"
    if n == "tp" or n.startswith("precip") or n == "kbdi" or "rain" in n:
        return "precip"
    return None


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("shift_sens")
    smoke = "--smoke" in sys.argv
    rng = np.random.default_rng(0)
    art = joblib.load(T.project_root / "models" / "gbt_coarse4.joblib")
    gbt, feats = art["model"], art["features"]
    stats = json.loads(Path(T.STATS).read_text())
    z = xr.open_zarr(str(T.CUBE), consolidated=True)
    _ = build_segmentation_features(z.data_vars)
    test_ds = T.make_dataset(*T.SPLITS["test"], feats, use_stack=True)

    # per-feature normalized noise std for each perturbation group
    def noise_std(j):
        name = feats[j]; g = feat_group(name)
        if g is None:
            return 0.0, None
        sd = float(stats.get(name, {}).get("std", 0.0)) or 1.0
        if name in AEMET_MAE:                      # direct meteo: physical MAE / std
            return 1.25 * AEMET_MAE[name] / sd, g
        if g == "precip":                          # derived precip features: relative shift (approx)
            return 1.25 * PRECIP_REL, g            # ~fraction-of-std proxy
        if g == "temperature":
            return 1.25 * 1.5 / sd, g
        if g == "wind":
            return 1.25 * (43 * np.pi / 180), g    # dir error dominant; rough
        if g == "pressure":
            return 1.25 * 9.6 / sd, g
        return 0.0, g

    nstd = np.zeros(len(feats)); groups = {}
    for j in range(len(feats)):
        s, g = noise_std(j)
        nstd[j] = s
        if g:
            groups.setdefault(g, []).append(j)
    for g, js in groups.items():
        log.info(f"group {g}: {len(js)} features, mean noise-std(norm)={np.mean([nstd[j] for j in js]):.3f}")

    days = list(range(0, len(test_ds), max(1, len(test_ds) // (10 if smoke else 365))))
    # cache test X/y/reg once
    cache = []
    for i in days:
        X, y, reg = test_ds[i]
        cache.append((X.numpy(), y[0].numpy().ravel(), reg[0].numpy().ravel()))

    def evaluate(perturb_groups):
        ps, ys, rs = [], [], []
        for X, y, reg in cache:
            Xp = X.copy()
            if perturb_groups:
                for g in perturb_groups:
                    for j in groups.get(g, []):
                        Xp[j] += rng.normal(0, nstd[j], Xp[j].shape).astype(Xp.dtype)
            land = reg > 0
            Xf = Xp.reshape(Xp.shape[0], -1).T[land]
            ps.append(gbt.predict_proba(Xf)[:, 1]); ys.append(y[land]); rs.append(reg[land])
        return T.regime_metrics(np.concatenate(ps), np.concatenate(ys), np.concatenate(rs))

    base = evaluate(None)
    log.info(f"BASELINE (ERA5, no shift): new-ign AP={base['new_ignition_ap']:.4f} spread={base['spread_ap']:.4f}")
    for scn in (["temperature"], ["wind"], ["pressure"], ["precip"], list(groups.keys())):
        m = evaluate(scn)
        d = m["new_ignition_ap"] - base["new_ignition_ap"]
        log.info(f"  +shift[{'+'.join(scn):<28}] new-ign AP={m['new_ignition_ap']:.4f} ({d:+.4f})  spread={m['spread_ap']:.4f}")


if __name__ == "__main__":
    main()
