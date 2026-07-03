"""Dig into WHY the point-wise GBT (new-ign 0.63) crushes the U-Net (0.19-0.22) on identical eval.

Two diagnostics, both cheap (no training):
  (1) GBT trustworthy? — new-ignition permutation importance: which features drive the 0.63? Legitimate
      point-wise physics (fuel/weather/dist_to_fire/human-access) → trustworthy; a single dominant
      suspicious feature → possible leakage.
  (2) Why does the U-Net fail? — ablate the wide-deep branches on the v5 checkpoint over the TEST eval:
      deep-only (zero wide) vs wide-only (zero deep) vs full. Tells us whether the point-wise signal lives
      in the (shallow) wide branch and whether the 24.9M-param deep branch drowned it in the additive fusion.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch, xarray as xr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
import scripts.train as T
import scripts.gbt_compare as G
from src.data.features import build_segmentation_features
from src.models.cnn import build_wide_deep_unet

ADJ = {1: -10.90, 2: -2.67}


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("dig")
    smoke = "--smoke" in sys.argv
    rng = np.random.default_rng(0)
    feats = build_segmentation_features(xr.open_zarr(str(T.CUBE), consolidated=True).data_vars)
    train_ds = T.make_dataset(*T.SPLITS["train"], feats, use_stack=True)
    val_ds = T.make_dataset(*T.SPLITS["val"], feats, use_stack=True)
    test_ds = T.make_dataset(*T.SPLITS["test"], feats, use_stack=True)

    # ---------- GBT fit (same as gbt_compare) ----------
    tdays = list(range(0, len(train_ds), 40 if smoke else G.TRAIN_DAY_STRIDE))
    Xtr, ytr, _ = G.collect_cells(train_ds, tdays, all_land=False, rng=rng)
    if Xtr.shape[0] > G.MAX_TRAIN_ROWS:
        s = rng.choice(Xtr.shape[0], G.MAX_TRAIN_ROWS, replace=False); Xtr, ytr = Xtr[s], ytr[s]
    gbt = HistGradientBoostingClassifier(max_iter=50 if smoke else 400, learning_rate=0.05,
        max_leaf_nodes=63, l2_regularization=1.0, validation_fraction=0.1, early_stopping=True, random_state=0)
    gbt.fit(Xtr, ytr); log.info(f"GBT fit {gbt.n_iter_} iters")

    # ---------- (1) new-ignition permutation importance on VAL ----------
    vdays = list(range(0, len(val_ds), max(1, len(val_ds) // (10 if smoke else 60))))
    Xv, yv, rv = G.collect_cells(val_ds, vdays, all_land=True)
    ign_pos = (yv == 1) & (rv == 1); neg = (yv == 0) & (rv > 0)
    negidx = np.where(neg)[0]; k = min(negidx.size, 15 * int(ign_pos.sum()))
    keep = np.concatenate([np.where(ign_pos)[0], rng.choice(negidx, k, replace=False)])
    Xe, ye = Xv[keep], yv[keep]
    base = average_precision_score(ye, gbt.predict_proba(Xe)[:, 1])
    log.info(f"GBT new-ign AP on this val subset = {base:.4f} (n_pos={int(ye.sum())}, n={ye.size})")
    drops = []
    for j in range(Xe.shape[1]):
        col = Xe[:, j].copy(); Xe[:, j] = rng.permutation(Xe[:, j])
        ap = average_precision_score(ye, gbt.predict_proba(Xe)[:, 1]); Xe[:, j] = col
        drops.append(base - ap)
    order = np.argsort(drops)[::-1]
    log.info("TOP 15 new-ignition drivers (GBT permutation importance = AP drop):")
    for j in order[:15]:
        log.info(f"   {feats[j]:<34} {drops[j]:+.4f}")

    # ---------- (2) U-Net branch ablation on TEST ----------
    ckpt = T.project_root / "models" / "seg_coarse4_widedeep_v5.pth"
    if not ckpt.exists():
        log.warning(f"{ckpt} missing; skip ablation"); return
    dev = T.get_device()
    m = build_wide_deep_unet(in_channels=len(feats), encoder_name="resnet34", encoder_weights=None, norm="group").to(dev)
    m.load_state_dict(torch.load(str(ckpt), map_location=dev)); m.eval()
    tdays2 = list(range(0, len(test_ds), max(1, len(test_ds) // (10 if smoke else 365))))
    acc = {"deep": [], "wide": [], "full": []}
    ys, rs = [], []
    with torch.no_grad():
        for i in tdays2:
            X, y, reg = test_ds[i]
            xb = X.unsqueeze(0).to(dev).float()
            ld = m.deep(xb)[0, 0]; lw = m.wide(xb)[0, 0]
            r = reg[0].to(dev)
            adj = torch.where(r == 1, ld.new_tensor(ADJ[1]), torch.where(r == 2, ld.new_tensor(ADJ[2]), ld.new_tensor(0.0)))
            acc["deep"].append(torch.sigmoid(ld + adj).cpu().numpy().ravel())
            acc["wide"].append(torch.sigmoid(lw + adj).cpu().numpy().ravel())
            acc["full"].append(torch.sigmoid(ld + lw + adj).cpu().numpy().ravel())
            ys.append(y[0].numpy().ravel()); rs.append(reg[0].numpy().ravel())
    yt = np.concatenate(ys); rt = np.concatenate(rs)
    log.info("U-Net v5 branch ablation on TEST (new-ign AP @ matched 15:1):")
    for name in ("deep", "wide", "full"):
        mtr = T.regime_metrics(np.concatenate(acc[name]), yt, rt)
        log.info(f"   {name:<5} new-ign={mtr['new_ignition_ap']:.4f} spread={mtr['spread_ap']:.4f} prec@K={mtr['prec_at_k']:.4f}")


if __name__ == "__main__":
    main()
