import argparse
import os
import sys
import time
from pathlib import Path

# Project root is the parent directory of /scripts
project_root = Path(__file__).resolve().parents[1]

# Ensure project root is on PYTHONPATH so `import src...` works
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
import math
import logging

from src.data.datasets import BaseIberFireDataset
from src.data.features import FEATURE_VARS
from src.models.cnn import build_unet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model file inside models/")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--encoder_name", type=str, default="resnet34", help="Encoder architecture to use")
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-5, help="Learning rate for optimizer")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    model_name = args.model_name
    # Ensure we have a consistent local filename with .pth extension
    if model_name.endswith(".pth"):
        model_file_name = model_name
        model_name = model_name[:-4]  # logical name without extension
    else:
        model_file_name = model_name + ".pth"

    mlflow.set_experiment("iberfire_unet_experiments")
    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("model_name", model_name)

        ZARR_PATH = project_root / "data" / "gold" / "IberFire_coarse32.zarr"

        mlflow.log_param("zarr_path", str(ZARR_PATH))
        mlflow.log_param("coarsen_factor", 32)

        train_time_start = "2008-01-01"
        train_time_end = "2022-12-31"
        val_time_start = "2023-01-01"
        val_time_end = "2024-12-31"
        lead_time = 1
        batch_size = 1
        mlflow.log_param("train_time_start", train_time_start)
        mlflow.log_param("train_time_end", train_time_end)
        mlflow.log_param("val_time_start", val_time_start)
        mlflow.log_param("val_time_end", val_time_end)
        mlflow.log_param("lead_time", lead_time)
        mlflow.log_param("batch_size", batch_size)

        # Canonical feature set lives in src/data/features.py (single source of
        # truth, shared with the serving app). Channel order is load-bearing for
        # loaded checkpoints, so edit it there, not here.
        feature_vars = FEATURE_VARS

        in_channels = len(feature_vars)

        # Model / optimizer hyperparameters
        encoder_name = args.encoder_name  # e.g., "resnet34" 
        lr = args.learning_rate
        weight_decay = 2e-3
        decoder_dropout = 0.10  # try 0.20 next if still overfitting

        mlflow.log_param("encoder_name", encoder_name)
        mlflow.log_param("architecture", f"Unet({encoder_name},imagenet,in={in_channels})")
        mlflow.log_param("in_channels", in_channels)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("feature_vars", ",".join(feature_vars))
        mlflow.log_param("lr", lr)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("decoder_dropout", decoder_dropout)

        TRAIN_STATS_PATH = project_root / "stats" / "simple_iberfire_stats_train.json"
        FIRE_DAY_INDICES_PATH = project_root / "stats" / "fire_day_indices.json"
        train_ds = BaseIberFireDataset(
            zarr_path=ZARR_PATH,
            time_start=train_time_start,
            time_end=train_time_end,
            feature_vars=feature_vars,
            label_var="is_fire",
            lead_time=lead_time,
            compute_stats=False,
            stats_path=TRAIN_STATS_PATH,
            mode="balanced_days",
            day_indices_path=FIRE_DAY_INDICES_PATH,
            balanced_ratio=1.0,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        # All-days training dataset and loader for curriculum
        train_all_ds = BaseIberFireDataset(
            zarr_path=ZARR_PATH,
            time_start=train_time_start,
            time_end=train_time_end,
            feature_vars=feature_vars,
            label_var="is_fire",
            lead_time=lead_time,
            compute_stats=False,
            stats_path=TRAIN_STATS_PATH,
            mode="all",
            day_indices_path=FIRE_DAY_INDICES_PATH,
        )

        train_all_loader = DataLoader(
            train_all_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        # Lightweight dataset sanity check (single sample access)
        sample_X, sample_y = train_ds[0]
        assert len(train_ds) > 0, "Training dataset is empty!"
        assert sample_X.shape[0] == in_channels, f"Expected {in_channels} input channels, got {sample_X.shape[0]}"
        assert sample_y.shape[0] == 1, f"Expected 1 output channel, got {sample_y.shape[0]}"
        assert sample_X.shape[1:] == sample_y.shape[1:], "Input and output spatial dimensions do not match"

        # test dataset
        test_ds = BaseIberFireDataset(
            zarr_path=ZARR_PATH,
            time_start=val_time_start,
            time_end=val_time_end,
            feature_vars=feature_vars,
            label_var="is_fire",
            lead_time=lead_time,
            compute_stats=False,
            stats_path=TRAIN_STATS_PATH,
            mode="all",
            day_indices_path=FIRE_DAY_INDICES_PATH,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        def compute_pos_ratio(loader):
            total_pos = 0
            total_pixels = 0
            for _, yb in loader:
                total_pos += yb.sum().item()
                total_pixels += yb.numel()
            return total_pos, total_pixels, (total_pos / total_pixels if total_pixels > 0 else 0.0)

        def evaluate_loader(model, criterion, loader, device, desc: str):
            """Evaluate loss-per-pixel and threshold-free metrics on a dataloader."""
            test_loss_sum = 0.0
            val_pixels = 0
            all_probs = []
            all_targets = []

            pbar = tqdm.tqdm(
                loader,
                desc=desc,
                ncols=100,
                file=sys.stdout,
                dynamic_ncols=False,
            )
            for X_val, y_val in pbar:
                X_val = X_val.to(device).float()
                y_val = y_val.to(device).float()

                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)

                test_loss_sum += val_loss.item() * y_val.numel()
                val_pixels += y_val.numel()
                pbar.set_postfix({"loss": f"{val_loss.item():.4f}"})

                probs_batch = torch.sigmoid(val_outputs).detach().cpu().view(-1)
                targets_batch = y_val.detach().cpu().view(-1)
                all_probs.append(probs_batch)
                all_targets.append(targets_batch)

            loss_per_pixel = test_loss_sum / max(val_pixels, 1)
            all_probs_np = torch.cat(all_probs).numpy() if all_probs else np.array([])
            all_targets_np = torch.cat(all_targets).numpy() if all_targets else np.array([])

            try:
                roc_auc = roc_auc_score(all_targets_np, all_probs_np) if all_targets_np.size else float("nan")
            except ValueError:
                roc_auc = float("nan")
            try:
                pr_auc = average_precision_score(all_targets_np, all_probs_np) if all_targets_np.size else float("nan")
            except ValueError:
                pr_auc = float("nan")

            return loss_per_pixel, roc_auc, pr_auc

        # Compute priors from all-days loader (for logit adjustment)
        train_pos, train_pix, train_ratio = compute_pos_ratio(train_all_loader)

        logger.info(f"Train positives (all days): {train_pos} out of {train_pix} pixels (ratio={train_ratio:.8f})")

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        model = build_unet(
            in_channels=in_channels,
            encoder_name=encoder_name,
            classes=1,
            encoder_weights="imagenet",
            decoder_dropout=decoder_dropout,
            activation=None,
        )

        model = model.to(device)
        # --------------------------
        # Logit-adjusted BCE loss
        # --------------------------
        # Based on: Menon et al., "Long-Tailed Classification via Logit Adjustment"
        # https://arxiv.org/abs/2007.07314
        #
        # This approach addresses class imbalance by adjusting the model's logits during
        # training to counteract the bias towards the majority class. The key insight is
        # that the optimal classifier under class imbalance should not just maximize
        # accuracy on the training distribution, but should be adjusted by the ratio of
        # class priors to achieve better generalization.
        #
        # For binary classification, the adjustment is:
        #   adjusted_logits = logits + log(pi_pos / pi_neg)
        # where pi_pos and pi_neg are the empirical class priors (proportions) from training data.
        #
        # Why log ratio?
        # 1. The log ratio acts as a "correction factor" in logit space that rebalances the
        #    decision boundary. Since logits are log-odds, adding log(pi_pos/pi_neg) shifts
        #    the decision boundary to account for class imbalance.
        # 2. When pi_pos < pi_neg (minority positive class), log(pi_pos/pi_neg) < 0, which
        #    decreases the logit values for positive class predictions. This makes the model
        #    less likely to predict positive unless there's strong evidence, compensating for
        #    the fact that the model sees fewer positive examples during training.
        # 3. This is mathematically equivalent to adjusting the posterior probabilities to
        #    reflect a balanced test distribution, making the model less biased towards the
        #    majority class while still learning from the imbalanced training data.

        # Empirical class priors from training data (pixel-wise)
        pi_pos = max(train_pos / max(train_pix, 1), 1e-8)
        pi_neg = max(1.0 - pi_pos, 1e-8)

        logit_adjustment = math.log(pi_pos / pi_neg)
        logit_adjustment = torch.tensor(
            logit_adjustment, device=device, dtype=torch.float32
        )

        class LogitAdjustedBCE(torch.nn.Module):
            """
            BCEWithLogitsLoss with logit adjustment for class imbalance as per
            Menon et al., 'Long-Tailed Classification via Logit Adjustment'"""
            def __init__(self, logit_adjustment: torch.Tensor):
                super().__init__()
                self.register_buffer("logit_adjustment", logit_adjustment)
                self.base_bce = torch.nn.BCEWithLogitsLoss(reduction="mean")

            def forward(self, logits, targets):
                adjusted_logits = logits + self.logit_adjustment
                return self.base_bce(adjusted_logits, targets)

        criterion = LogitAdjustedBCE(logit_adjustment=logit_adjustment)

        mlflow.log_param("criterion", "LogitAdjustedBCE")
        mlflow.log_param("pi_pos", float(pi_pos))
        mlflow.log_param("pi_neg", float(pi_neg))
        mlflow.log_param("logit_adjustment", float(logit_adjustment.item()))
        mlflow.log_param("train_pos_ratio", float(train_ratio))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        checkpoint_path = project_root / "models" / model_file_name

        if checkpoint_path.exists():
            logger.info(f"Loading existing model from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
        else:
            logger.info(f"No model found at {checkpoint_path}. Initializing new model.")
            os.makedirs(checkpoint_path.parent, exist_ok=True)

        # Curriculum switch parameter
        curriculum_epochs = 10
        mlflow.log_param("curriculum_epochs", curriculum_epochs)

        NUM_EPOCHS = args.epochs
        overall_start = time.time()
        model.train()
        # Early stopping based on val_all_pr_auc (deployment distribution)
        best_val_all_pr_auc = -float("inf")
        best_model_state = None
        epochs_no_improve = 0
        patience = 10
        for epoch in range(NUM_EPOCHS):
            train_loss_sum = 0.0
            train_pixels = 0

            if epoch < curriculum_epochs:
                active_loader = train_loader  # balanced_days
                train_mode = "balanced_days"
            else:
                active_loader = train_all_loader  # all days
                train_mode = "all"

            pbar = tqdm.tqdm(
                active_loader,
                desc=f"Epoch: {epoch + 1}/{NUM_EPOCHS} [train_mode={train_mode}]",
                ncols=100,
                file=sys.stdout,  # force stdout
                dynamic_ncols=False,  # fixed width
            )
            for X_batch, y_batch in pbar:
                X_batch = X_batch.to(device).float()
                y_batch = y_batch.to(device).float()

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * y_batch.numel()
                train_pixels += y_batch.numel()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss_per_pixel = train_loss_sum / max(train_pixels, 1)
            mlflow.log_metric(
                "train_logit_adjusted_bce_per_pixel",
                train_loss_per_pixel,
                step=epoch + 1,
            )

            model.eval()
            with torch.no_grad():
                val_all_loss, val_all_roc, val_all_pr = evaluate_loader(
                    model=model,
                    criterion=criterion,
                    loader=test_loader,
                    device=device,
                    desc="Validation (all)",
                )

            mlflow.log_metric(
                "val_all_logit_adjusted_bce_per_pixel",
                val_all_loss,
                step=epoch + 1,
            )
            mlflow.log_metric("val_all_roc_auc", val_all_roc, step=epoch + 1)
            mlflow.log_metric("val_all_pr_auc", val_all_pr, step=epoch + 1)

            # Track best checkpoint by val_all_pr_auc and early stop when it stops improving
            if np.isfinite(val_all_pr) and (val_all_pr > best_val_all_pr_auc + 1e-6):
                best_val_all_pr_auc = float(val_all_pr)
                epochs_no_improve = 0
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                # Save best weights immediately
                torch.save(best_model_state, checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(
                        f"Early stopping: val_all_pr_auc did not improve for {patience} epochs. Best={best_val_all_pr_auc:.6f}"
                    )
                    break

            model.train()

        # Always restore best model (by val_all_pr_auc) after training
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Move model to CPU for MLflow logging and signature inference
        model_cpu = model.to("cpu").eval()

        # Log the model to MLflow for traceability and later loading
        input_example = sample_X.unsqueeze(0).cpu().numpy().astype("float32")
        mlflow.pytorch.log_model(
            model_cpu,
            name=model_name,
            input_example=input_example,
        )
        total_duration = time.time() - overall_start
        epochs_ran = (epoch + 1) if "epoch" in locals() else 0
        avg_time = total_duration / epochs_ran if epochs_ran > 0 else 0.0
        logger.info(f"Total training time: {total_duration:.2f} seconds")
        logger.info(f"Average time per epoch: {avg_time:.2f} seconds over {epochs_ran} epochs")
        logger.info(f"Model saved to {checkpoint_path}")