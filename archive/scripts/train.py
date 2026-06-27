"""Train the single-head, regime-aware U-Net for next-day fire segmentation (coarse4).

Pipeline (the deep rewrite):
  * 3-way TEMPORAL split — train 2008-18 / val 2019-21 / test 2022-24 (test touched once).
  * RegimeIberFireDataset (146 features incl. is_fire(t); regime code per pixel) + fire-day
    resampling (WeightedRandomSampler); full-image batches.
  * GroupNorm U-Net (small-batch-safe on ~14 GB) via build_unet(norm="group").
  * Regime-aware FOCAL logit-adjusted BCE (per-regime priors from train; alpha leans to ignition;
    focal_gamma=2 by default — the load-bearing fix for rare-positive gradient dilution).
  * MLflow: PARAMS + METRICS only (no duplicate model artifact). Single .pth checkpoint.
  * MPS-aware; num_workers=0 by default (Mac/MPS DataLoader-worker overhead — benchmarked).

Selection/early-stopping is on a VAL **blend** = alpha*new-ignition AP + (1-alpha)*spread AP
(same weights as the loss). The GBT new-ignition floor ≈ 0.50 is the bar to beat. Run with
--smoke for a fast end-to-end sanity pass.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # CPU fallback for any non-MPS op

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.datasets import RegimeIberFireDataset, StackedRegimeIberFireDataset
from src.data.features import build_segmentation_features
from src.models.cnn import build_unet, build_wide_deep_unet
from src.models.losses import RegimeLogitAdjustedBCE, compute_regime_priors

CUBE = project_root / "data" / "gold" / "FireGuard_coarse4.zarr"
DYN = project_root / "data" / "gold" / "FireGuard_coarse4_dyn.zarr"
STATS = project_root / "stats" / "coarse4_norm_stats_train.json"
SPLITS = {"train": ("2008-01-01", "2018-12-31"),
          "val": ("2019-01-01", "2021-12-31"),
          "test": ("2022-01-01", "2024-12-31")}


def prefetch(iterable, depth: int = 3):
    """Background-thread prefetcher: overlaps loading with GPU compute without multiprocess
    workers (zarr/NumPy release the GIL during I/O, so the thread loads while MPS computes)."""
    import queue
    import threading
    q: "queue.Queue" = queue.Queue(maxsize=depth)
    sentinel = object()

    def worker():
        try:
            for item in iterable:
                q.put(item)
        finally:
            q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def regime_metrics(prob: np.ndarray, tgt: np.ndarray, reg: np.ndarray,
                   neg_ratio: int = 15, seed: int = 0) -> dict:
    """Interpretable, GBT-comparable metrics over land cells.

    The earlier `regime_ap` scored AP against ALL negatives (true ~0.005 % prevalence),
    which is honest but (a) uninterpretable and (b) NOT comparable to the GBT measurement
    floor, which was computed on negatives subsampled to neg_ratio:1 (~6 % prevalence). AP is
    acutely prevalence-dependent, so the two numbers differed ~1000× for no real reason. Here:
      * roc                : full-prevalence ROC-AUC (prevalence-INDEPENDENT — the honest ranker).
      * {regime}_ap        : AP at MATCHED prevalence (negatives subsampled to neg_ratio:1, same
                             seed each call → comparable across epochs AND to the GBT floor:
                             new-ign ≈ 0.50, spread/continuation ≈ 0.98).
      * prec_at_k          : overall precision@K with K = #fire cells (R-precision) — operational
                             ("of the K cells we'd flag, how many actually burn").
    """
    rng = np.random.default_rng(seed)
    land = reg > 0
    p, t, r = prob[land], tgt[land], reg[land]
    out = {"roc": float("nan"), "overall_ap": float("nan"),
           "new_ignition_ap": float("nan"), "spread_ap": float("nan"),
           "prec_at_k": float("nan"), "n_pos": int(t.sum())}
    if t.sum() == 0 or t.size == 0:
        return out
    try:
        out["roc"] = float(roc_auc_score(t, p))
    except ValueError:
        pass
    neg_p = p[t == 0]

    def matched_ap(pos_mask):
        n_pos = int(pos_mask.sum())
        if n_pos == 0 or neg_p.size == 0:
            return float("nan")
        k = min(neg_p.size, neg_ratio * n_pos)
        sel = rng.choice(neg_p.size, size=k, replace=False)
        pp = np.concatenate([p[pos_mask], neg_p[sel]])
        tt = np.concatenate([np.ones(n_pos), np.zeros(k)])
        return float(average_precision_score(tt, pp))

    out["overall_ap"] = matched_ap(t == 1)
    out["new_ignition_ap"] = matched_ap((t == 1) & (r == 1))
    out["spread_ap"] = matched_ap((t == 1) & (r == 2))
    k = int(t.sum())  # R-precision: precision in the top-K most-confident cells
    topk = np.argpartition(p, -k)[-k:]
    out["prec_at_k"] = float(t[topk].sum() / k)
    return out


@torch.no_grad()
def evaluate(model, loader, device, adj_ign: float = 0.0, adj_spr: float = 0.0) -> dict:
    """Apply the SAME per-regime logit adjustment used in the loss before sigmoid.

    The model is trained to produce logits that are correct *after* the adjustment; scoring raw
    logits cross-regime is wrong (it tanked spread AP 0.99→0.07 and ROC 0.90→0.71). Regime is
    known at inference (from dist_to_fire(t)), so applying it here — and at serving — is legitimate.
    """
    model.eval()
    probs, tgts, regs = [], [], []
    for X, y, reg in loader:
        logit = model(X.to(device).float())
        reg_d = reg.to(device)
        adj = torch.where(reg_d == 1, logit.new_tensor(adj_ign),
              torch.where(reg_d == 2, logit.new_tensor(adj_spr), logit.new_tensor(0.0)))
        probs.append(torch.sigmoid(logit + adj).float().cpu().numpy().ravel())
        tgts.append(y.numpy().ravel())
        regs.append(reg.numpy().ravel())
    return regime_metrics(np.concatenate(probs), np.concatenate(tgts), np.concatenate(regs))


def make_dataset(t0, t1, feature_vars, use_stack=False):
    common = dict(zarr_path=CUBE, time_start=t0, time_end=t1, feature_vars=feature_vars,
                  label_var="is_fire", lead_time=1, compute_stats=False, stats_path=STATS, mode="all")
    if use_stack:
        return StackedRegimeIberFireDataset(dyn_zarr_path=DYN, **common)
    return RegimeIberFireDataset(**common)


def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("train")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="seg_coarse4_v1")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=2e-3)
    ap.add_argument("--alpha", type=float, default=0.6, help="ignition-regime loss weight")
    ap.add_argument("--focal-gamma", type=float, default=2.0,
                    help="focal down-weighting of easy cells; 0=plain BCE. >0 is the load-bearing "
                         "fix for rare-positive gradient dilution (diag_overfit.py)")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--decoder-dropout", type=float, default=0.10)
    ap.add_argument("--wide", action="store_true",
                    help="wide-and-deep: add a point-wise 1x1 branch (GBT-style pathway) to the U-Net")
    ap.add_argument("--wide-dropout", type=float, default=0.10, help="Dropout2d in the wide branch")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--steps-per-epoch", type=int, default=0, help="0 = full epoch")
    ap.add_argument("--fire-oversample", type=float, default=10.0)
    ap.add_argument("--use-stack", action="store_true", help="use the pre-stacked dynamic array (fast loading)")
    ap.add_argument("--compile", action="store_true", help="try torch.compile on the model (MPS-experimental)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = get_device()
    import xarray as xr
    feature_vars = build_segmentation_features(xr.open_zarr(CUBE, consolidated=True).data_vars)
    C = len(feature_vars)
    log.info(f"device={device} | C={C} features | batch={args.batch_size} | alpha={args.alpha}")

    splits = SPLITS
    if args.smoke:  # tiny ranges + cheap loop for an end-to-end sanity pass
        splits = {"train": ("2015-06-01", "2015-09-30"), "val": ("2016-06-01", "2016-08-31"),
                  "test": ("2017-06-01", "2017-08-31")}
        args.epochs, args.steps_per_epoch, args.batch_size = 2, 15, 2

    train_ds = make_dataset(*splits["train"], feature_vars, args.use_stack)
    val_ds = make_dataset(*splits["val"], feature_vars, args.use_stack)
    test_ds = make_dataset(*splits["test"], feature_vars, args.use_stack)

    # --- per-regime priors from a strided train subset (cheap, stable) ---
    stride = max(1, len(train_ds) // (40 if args.smoke else 800))
    prior_loader = DataLoader(Subset(train_ds, list(range(0, len(train_ds), stride))),
                              batch_size=args.batch_size, num_workers=args.num_workers)
    adj_ign, adj_spr, prior_info = compute_regime_priors(prior_loader)
    log.info(f"priors: ignition pos_rate={prior_info['ignition_pos_rate']:.5f} (adj={adj_ign:.2f}) | "
             f"spread pos_rate={prior_info['spread_pos_rate']:.5f} (adj={adj_spr:.2f})")

    loss_fn = RegimeLogitAdjustedBCE(args.alpha, adj_ign, adj_spr, focal_gamma=args.focal_gamma).to(device)
    if args.wide:
        model = build_wide_deep_unet(in_channels=C, encoder_name=args.encoder, encoder_weights="imagenet",
                                     decoder_dropout=args.decoder_dropout, norm="group",
                                     wide_dropout=args.wide_dropout).to(device)
        log.info(f"wide-and-deep: deep U-Net + point-wise 1x1 branch (wide_dropout={args.wide_dropout}, "
                 f"zero-init head → starts at deep baseline)")
    else:
        model = build_unet(in_channels=C, encoder_name=args.encoder, encoder_weights="imagenet",
                           decoder_dropout=args.decoder_dropout, norm="group").to(device)
    if args.compile:
        try:
            model = torch.compile(model)
            log.info("torch.compile enabled")
        except Exception as e:  # MPS support is experimental — fall back gracefully
            log.warning(f"torch.compile failed ({e}); continuing uncompiled")
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_ds.make_weighted_sampler(args.fire_oversample),
                              num_workers=args.num_workers, drop_last=True,
                              persistent_workers=(args.num_workers > 0))
    val_stride = max(1, len(val_ds) // (20 if args.smoke else 365))  # span seasons, cheap
    val_eval = DataLoader(Subset(val_ds, list(range(0, len(val_ds), val_stride))),
                          batch_size=args.batch_size, num_workers=args.num_workers)

    import mlflow
    mlflow.set_experiment("iberfire_seg_coarse4")
    ckpt = project_root / "models" / f"{args.model_name}.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=args.model_name):
        mlflow.log_params({
            "resolution_km": 4, "in_channels": C, "encoder": args.encoder, "norm": "group",
            "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
            "alpha": args.alpha, "decoder_dropout": args.decoder_dropout,
            "focal_gamma": args.focal_gamma, "wide": args.wide, "wide_dropout": args.wide_dropout,
            "fire_oversample": args.fire_oversample, "device": str(device),
            "loss": "RegimeFocalLogitAdjustedBCE", "adj_ignition": adj_ign, "adj_spread": adj_spr,
            "lead_time": 1, **{f"split_{k}": f"{v[0]}..{v[1]}" for k, v in splits.items()},
        })

        best_ap, best_state, no_improve = -1.0, None, 0
        steps = args.steps_per_epoch or len(train_loader)
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0 = time.time()
            run_loss = run_ign = run_spr = 0.0
            for i, (X, y, reg) in enumerate(prefetch(train_loader)):
                if i >= steps:
                    break
                X, y, reg = X.to(device).float(), y.to(device).float(), reg.to(device)
                optim.zero_grad()
                logit = model(X)
                loss, comp = loss_fn(logit, y, reg)
                loss.backward()
                optim.step()
                run_loss += loss.item(); run_ign += comp["L_ignition"].item(); run_spr += comp["L_spread"].item()
            n = min(steps, len(train_loader))
            val = evaluate(model, val_eval, device, adj_ign, adj_spr)
            dt = time.time() - t0
            ni = val["new_ignition_ap"]; sp = val["spread_ap"]
            # blended selection metric — same weights as the loss (alpha to ignition)
            val_blend = (args.alpha * (ni if np.isfinite(ni) else 0.0)
                         + (1.0 - args.alpha) * (sp if np.isfinite(sp) else 0.0))
            log.info(f"epoch {epoch}/{args.epochs} [{dt:.0f}s] loss={run_loss/n:.4f} "
                     f"(ign={run_ign/n:.4f} spr={run_spr/n:.4f}) | val blend={val_blend:.4f} "
                     f"(new-ign AP={ni:.4f}[bar~0.50] spread AP={sp:.4f}[bar~0.98]) "
                     f"prec@K={val['prec_at_k']:.4f} roc={val['roc']:.3f}  (AP @matched 15:1 prevalence)")
            mlflow.log_metrics({
                "train_loss": run_loss / n, "train_L_ignition": run_ign / n, "train_L_spread": run_spr / n,
                "val_blend": val_blend, "val_new_ignition_ap": ni, "val_spread_ap": sp,
                "val_overall_ap": val["overall_ap"], "val_prec_at_k": val["prec_at_k"],
                "val_roc": val["roc"], "epoch_seconds": dt,
            }, step=epoch)

            if val_blend > best_ap + 1e-5:
                best_ap = val_blend; no_improve = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, ckpt)  # single checkpoint; no mlflow.pytorch.log_model
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    log.info(f"early stop @ epoch {epoch} (best val blend={best_ap:.4f})")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        # touched-once test (full), reported once
        test = evaluate(model, DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers),
                        device, adj_ign, adj_spr)
        log.info(f"TEST new-ign AP={test['new_ignition_ap']:.4f} (GBT floor 0.50) "
                 f"spread AP={test['spread_ap']:.4f} (GBT ~0.98) overall={test['overall_ap']:.4f} "
                 f"prec@K={test['prec_at_k']:.4f} roc={test['roc']:.3f}  (AP @matched 15:1 prevalence)")
        mlflow.log_metrics({f"test_{k}": v for k, v in test.items()})
        mlflow.log_param("best_val_blend", best_ap)
        log.info(f"saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
