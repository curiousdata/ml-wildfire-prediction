"""Optuna hyperparameter tuning for the PRODUCTION point-wise GBT (Track A, v1 cube).

The shipped `gbt_coarse4.joblib` uses hand-picked defaults (lr 0.05, max_leaf_nodes 63, l2 1.0). This
searches the HistGBT hyperparameter space with Optuna, optimizing the HEADLINE operational metric —
**new-ignition AP on the VAL split** (2019-2021) — and never touches the 2022-2024 TEST set during search
(test is touched-once; it's evaluated only by the final refit, for an honest tuned-vs-baseline number).

Method (identical data path to scripts/train_gbt.py + scripts/gbt_compare.py, so the result is comparable):
  * Same chronological split (T.SPLITS), same 146 features (build_segmentation_features), same per-cell
    sampling (G.collect_cells: positives + NEG_PER_POS negatives/day, subsampled to MAX_TRAIN_ROWS).
  * Train cells and VAL-eval cells are materialized ONCE and reused across every trial → each trial is just
    a fit + predict_proba + regime_metrics (fast). Each fit uses early_stopping, so max_iter is a ceiling,
    not a tuned knob (lr trades off against it).
  * Objective = T.regime_metrics(...)['new_ignition_ap'] on the held VAL eval cells.

Output (does NOT overwrite the production model — promotion is a manual rename if it wins):
  * reports/gbt_optuna.json      — all trials, best params, baseline-vs-tuned table
  * models/gbt_coarse4.tuned.joblib + .meta.json  — best config refit on full train, full val+test metrics

CLI: --trials N (default 40) | --eval-days N (val eval days held for the objective, default 120) | --smoke
"""
from __future__ import annotations
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import optuna
import xarray as xr
from sklearn.ensemble import HistGradientBoostingClassifier

import scripts.train as T
import scripts.gbt_compare as G
from src.data.features import build_segmentation_features

log = logging.getLogger("tune_gbt")


def eval_full(gbt, ds, smoke=False):
    """Honest eval on the FULL strided eval days (all land cells), per-day to bound memory — mirrors
    gbt_compare/train_gbt so the tuned numbers are directly comparable to the production meta."""
    stride = max(1, len(ds) // (20 if smoke else 365))
    probs, ys, rs = [], [], []
    for i in range(0, len(ds), stride):
        X, y, reg = ds[i]
        Xf = X.numpy().reshape(X.shape[0], -1).T
        land = reg.numpy().ravel() > 0
        probs.append(gbt.predict_proba(Xf[land])[:, 1])
        ys.append(y.numpy().ravel()[land]); rs.append(reg.numpy().ravel()[land])
    return T.regime_metrics(np.concatenate(probs), np.concatenate(ys), np.concatenate(rs))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    smoke = "--smoke" in sys.argv
    a = sys.argv
    n_trials = int(a[a.index("--trials") + 1]) if "--trials" in a else (8 if smoke else 40)
    eval_days = int(a[a.index("--eval-days") + 1]) if "--eval-days" in a else (30 if smoke else 120)
    rng = np.random.default_rng(0)

    feats = build_segmentation_features(xr.open_zarr(str(T.CUBE), consolidated=True).data_vars)
    train_ds = T.make_dataset(*T.SPLITS["train"], feats, use_stack=True)
    val_ds = T.make_dataset(*T.SPLITS["val"], feats, use_stack=True)
    test_ds = T.make_dataset(*T.SPLITS["test"], feats, use_stack=True)
    log.info(f"{len(feats)} features | train {len(train_ds)}d val {len(val_ds)}d test {len(test_ds)}d")

    # --- materialize train cells ONCE (same sampling as production trainer) ---
    tdays = list(range(0, len(train_ds), 40 if smoke else G.TRAIN_DAY_STRIDE))
    log.info(f"collecting train cells from {len(tdays)} days...")
    Xtr, ytr, _ = G.collect_cells(train_ds, tdays, all_land=False, rng=rng)
    if Xtr.shape[0] > G.MAX_TRAIN_ROWS:
        s = rng.choice(Xtr.shape[0], G.MAX_TRAIN_ROWS, replace=False); Xtr, ytr = Xtr[s], ytr[s]
    log.info(f"train matrix {Xtr.shape}, pos rate {ytr.mean():.4f}")

    # --- materialize a VAL eval slice ONCE (all land cells over eval_days strided days) for the objective ---
    vstride = max(1, len(val_ds) // eval_days)
    vdays = list(range(0, len(val_ds), vstride))
    log.info(f"holding {len(vdays)} val eval days (all land cells) for the objective...")
    Xva, yva, rva = G.collect_cells(val_ds, vdays, all_land=True, rng=rng)
    log.info(f"val eval matrix {Xva.shape} ({Xva.nbytes/1e9:.1f} GB), pos {int(yva.sum())} "
             f"new-ign {int(((yva == 1) & (rva == 1)).sum())}")

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 255),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 300),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            max_features=trial.suggest_float("max_features", 0.5, 1.0),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
            max_iter=(80 if smoke else 600), validation_fraction=0.1, early_stopping=True,
            n_iter_no_change=20, random_state=0)
        gbt = HistGradientBoostingClassifier(**params)
        gbt.fit(Xtr, ytr)
        m = T.regime_metrics(gbt.predict_proba(Xva)[:, 1], yva, rva)
        trial.set_user_attr("n_iter", int(gbt.n_iter_))
        trial.set_user_attr("spread_ap", m["spread_ap"]); trial.set_user_attr("roc", m["roc"])
        ap = m["new_ignition_ap"]
        return ap if np.isfinite(ap) else 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info(f"{n_trials} trials in {time.time()-t0:.0f}s | best val new-ign AP "
             f"(held {len(vdays)}d) = {study.best_value:.4f}")
    log.info(f"best params: {study.best_params}")

    # --- refit best config on full train, evaluate on FULL val + test (honest, comparable to prod meta) ---
    best = dict(study.best_params, max_iter=(80 if smoke else 600), validation_fraction=0.1,
                early_stopping=True, n_iter_no_change=20, random_state=0)
    gbt = HistGradientBoostingClassifier(**best)
    gbt.fit(Xtr, ytr)
    val = eval_full(gbt, val_ds, smoke); test = eval_full(gbt, test_ds, smoke)
    for nm, m in (("VAL", val), ("TEST", test)):
        log.info(f"TUNED {nm}: new-ign AP={m['new_ignition_ap']:.4f} spread={m['spread_ap']:.4f} "
                 f"overall={m['overall_ap']:.4f} prec@K={m['prec_at_k']:.4f} roc={m['roc']:.4f}")

    # --- baseline (production meta) for the comparison table ---
    base_meta_p = T.project_root / "models" / "gbt_coarse4.meta.json"
    base = json.loads(base_meta_p.read_text()) if base_meta_p.exists() else {}
    bval, btest = base.get("val", {}), base.get("test", {})
    log.info(f"BASELINE VAL new-ign AP={bval.get('new_ignition_ap')} | TEST={btest.get('new_ignition_ap')}")

    reports = T.project_root / "reports"; reports.mkdir(exist_ok=True)
    (reports / "gbt_optuna.json").write_text(json.dumps({
        "metric": "val new_ignition_ap (held eval slice during search; full eval in refit)",
        "n_trials": n_trials, "eval_days_held": len(vdays), "smoke": smoke,
        "best_params": study.best_params, "best_val_newign_ap_held": study.best_value,
        "tuned": {"val": val, "test": test}, "baseline": {"val": bval, "test": btest},
        "delta_test_newign_ap": (test["new_ignition_ap"] - btest.get("new_ignition_ap", float("nan"))
                                 if btest else None),
        "trials": [{"number": t.number, "value": t.value, "params": t.params,
                    "n_iter": t.user_attrs.get("n_iter"), "roc": t.user_attrs.get("roc"),
                    "spread_ap": t.user_attrs.get("spread_ap")}
                   for t in study.trials],
    }, indent=2))

    out = T.project_root / "models" / "gbt_coarse4.tuned.joblib"
    joblib.dump({"model": gbt, "features": list(feats)}, out)
    out.with_suffix(".meta.json").write_text(json.dumps({
        "model": "HistGradientBoostingClassifier (Optuna-tuned; Track A v1 cube)",
        "tuned_from": "models/gbt_coarse4.joblib (hand-picked defaults)",
        "cube": str(T.CUBE), "n_features": len(feats), "features": list(feats),
        "params": {k: v for k, v in best.items()}, "n_iter": int(gbt.n_iter_),
        "train_rows": int(Xtr.shape[0]), "train_pos_rate": float(ytr.mean()),
        "splits": {k: f"{v[0]}..{v[1]}" for k, v in T.SPLITS.items()},
        "val": val, "test": test,
        "note": "NOT auto-promoted. To ship: rename over gbt_coarse4.joblib (+ re-run calibrate_gbt.py).",
    }, indent=2))
    log.info(f"saved {out.name} + report reports/gbt_optuna.json (production model untouched)")


if __name__ == "__main__":
    main()
