"""
Model Trainer and Evaluator for Anti-Collapse Dual-Stream TCN.

Features:
- Mini-batch gradient optimization with AdamW & CosineAnnealingLR
- Tripartite Compound Loss optimization (Huber + Pearson + Variance)
- Validation tracking: SBP/DBP MAE, RMSE, Error SD, Pearson r, and Variance Ratio
- Diagnostic Template Collapse detector during training and evaluation
- Checkpoint persistence and early stopping
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.ds_tcn import DualStreamTCN
from src.models.compound_loss import TripartiteCompoundLoss


@dataclass
class TrainingConfig:
    """Hyperparameter and runtime configuration for BPTrainer."""
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 15
    patience: int = 5
    delta: float = 1.0
    lambda_1: float = 0.40
    lambda_2: float = 0.25
    sigma_thresh: float = 8.0
    device: str = "cpu"


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for SBP and DBP predictions."""
    loss: float
    sbp_mae: float
    dbp_mae: float
    sbp_rmse: float
    dbp_rmse: float
    sbp_sd: float
    dbp_sd: float
    sbp_pearson_r: float
    dbp_pearson_r: float
    sbp_variance_ratio: float
    dbp_variance_ratio: float
    aami_pass: bool
    template_collapse_detected: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "loss": self.loss,
            "sbp_mae": self.sbp_mae,
            "dbp_mae": self.dbp_mae,
            "sbp_rmse": self.sbp_rmse,
            "dbp_rmse": self.dbp_rmse,
            "sbp_sd": self.sbp_sd,
            "dbp_sd": self.dbp_sd,
            "sbp_pearson_r": self.sbp_pearson_r,
            "dbp_pearson_r": self.dbp_pearson_r,
            "sbp_variance_ratio": self.sbp_variance_ratio,
            "dbp_variance_ratio": self.dbp_variance_ratio,
            "aami_pass": self.aami_pass,
            "template_collapse_detected": self.template_collapse_detected
        }


class BPTrainer:
    """
    End-to-end trainer and evaluator for DualStreamTCN blood pressure estimation.
    """
    def __init__(
        self,
        model: DualStreamTCN,
        config: Optional[TrainingConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None
    ) -> None:
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)

        self.loss_fn = loss_fn or TripartiteCompoundLoss(
            delta=self.config.delta,
            lambda_1=self.config.lambda_1,
            lambda_2=self.config.lambda_2,
            sigma_thresh=self.config.sigma_thresh
        )

        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        self.scheduler = scheduler or torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.config.epochs),
            eta_min=1e-6
        )

        self.best_val_loss = float("inf")
        self.best_model_state: Optional[Dict[str, Any]] = None

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Execute one epoch of model training.
        
        Returns:
            Dictionary containing average training loss and component losses.
        """
        self.model.train()
        total_loss = 0.0
        huber_loss = 0.0
        pearson_loss = 0.0
        variance_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            morph = batch["morph"].to(self.device)
            timing = batch["timing"].to(self.device)
            anchor = batch["anchor"].to(self.device)
            targets = batch["targets"].to(self.device)  # [B, 2]

            self.optimizer.zero_grad()

            sbp_pred, dbp_pred = self.model(morph, timing, anchor)
            preds = torch.cat([sbp_pred, dbp_pred], dim=-1)  # [B, 2]

            loss, metrics = self.loss_fn(preds, targets)
            loss.backward()

            # Gradient clipping for stable training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += float(loss.item())
            huber_loss += float(metrics.get("loss_huber", 0.0))
            pearson_loss += float(metrics.get("loss_pearson", 0.0))
            variance_loss += float(metrics.get("loss_variance", 0.0))
            num_batches += 1

        if num_batches > 0:
            return {
                "train_loss": total_loss / num_batches,
                "train_loss_huber": huber_loss / num_batches,
                "train_loss_pearson": pearson_loss / num_batches,
                "train_loss_variance": variance_loss / num_batches,
            }
        return {"train_loss": 0.0}

    def evaluate(self, dataloader: DataLoader) -> EvaluationMetrics:
        """
        Perform complete validation/test evaluation across the dataset.
        
        Computes clinical metrics (MAE, RMSE, SD, Pearson r, Variance Ratio).
        """
        self.model.eval()
        all_sbp_preds = []
        all_dbp_preds = []
        all_sbp_targets = []
        all_dbp_targets = []
        total_eval_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                morph = batch["morph"].to(self.device)
                timing = batch["timing"].to(self.device)
                anchor = batch["anchor"].to(self.device)
                targets = batch["targets"].to(self.device)

                sbp_pred, dbp_pred = self.model(morph, timing, anchor)
                preds = torch.cat([sbp_pred, dbp_pred], dim=-1)

                loss, _ = self.loss_fn(preds, targets)
                total_eval_loss += float(loss.item())
                num_batches += 1

                all_sbp_preds.extend(sbp_pred.cpu().view(-1).numpy().tolist())
                all_dbp_preds.extend(dbp_pred.cpu().view(-1).numpy().tolist())
                all_sbp_targets.extend(targets[:, 0].cpu().view(-1).numpy().tolist())
                all_dbp_targets.extend(targets[:, 1].cpu().view(-1).numpy().tolist())

        sbp_pred_arr = np.array(all_sbp_preds, dtype=np.float64)
        dbp_pred_arr = np.array(all_dbp_preds, dtype=np.float64)
        sbp_true_arr = np.array(all_sbp_targets, dtype=np.float64)
        dbp_true_arr = np.array(all_dbp_targets, dtype=np.float64)

        # MAE and RMSE
        sbp_errors = sbp_pred_arr - sbp_true_arr
        dbp_errors = dbp_pred_arr - dbp_true_arr
        sbp_mae = float(np.mean(np.abs(sbp_errors)))
        dbp_mae = float(np.mean(np.abs(dbp_errors)))
        sbp_rmse = float(np.sqrt(np.mean(sbp_errors ** 2)))
        dbp_rmse = float(np.sqrt(np.mean(dbp_errors ** 2)))
        sbp_sd = float(np.std(sbp_errors))
        dbp_sd = float(np.std(dbp_errors))

        # Pearson correlation
        def calc_r(y_true, y_pred):
            vx = y_pred - np.mean(y_pred)
            vy = y_true - np.mean(y_true)
            denom = np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)) + 1e-8
            return float(np.sum(vx * vy) / denom)

        sbp_r = calc_r(sbp_true_arr, sbp_pred_arr)
        dbp_r = calc_r(dbp_true_arr, dbp_pred_arr)

        # Variance ratio
        sbp_var_ratio = float((np.var(sbp_pred_arr) + 1e-8) / (np.var(sbp_true_arr) + 1e-8))
        dbp_var_ratio = float((np.var(dbp_pred_arr) + 1e-8) / (np.var(dbp_true_arr) + 1e-8))

        # AAMI Criteria check: MAE <= 5.0, SD <= 7.0 for SBP and DBP
        aami_pass = (sbp_mae <= 5.0 and dbp_mae <= 3.5 and sbp_sd <= 7.0 and dbp_sd <= 7.0)

        # Template collapse check: r < 0.82 or variance_ratio < 0.70
        template_collapse = (sbp_r < 0.82 or sbp_var_ratio < 0.70 or dbp_r < 0.82 or dbp_var_ratio < 0.70)

        avg_loss = total_eval_loss / max(1, num_batches)

        return EvaluationMetrics(
            loss=avg_loss,
            sbp_mae=sbp_mae,
            dbp_mae=dbp_mae,
            sbp_rmse=sbp_rmse,
            dbp_rmse=dbp_rmse,
            sbp_sd=sbp_sd,
            dbp_sd=dbp_sd,
            sbp_pearson_r=sbp_r,
            dbp_pearson_r=dbp_r,
            sbp_variance_ratio=sbp_var_ratio,
            dbp_variance_ratio=dbp_var_ratio,
            aami_pass=aami_pass,
            template_collapse_detected=template_collapse
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
        verbose: bool = False
    ) -> Dict[str, List[float]]:
        """
        Train the model over specified epochs with early stopping and metrics history.
        """
        num_epochs = epochs or self.config.epochs
        patience = self.config.patience
        patience_counter = 0

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "sbp_mae": [],
            "dbp_mae": [],
            "sbp_pearson_r": [],
            "dbp_pearson_r": []
        }

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_metrics["train_loss"])

            if self.scheduler is not None:
                self.scheduler.step()

            if val_loader is not None:
                val_eval = self.evaluate(val_loader)
                val_loss = val_eval.loss
                history["val_loss"].append(val_loss)
                history["sbp_mae"].append(val_eval.sbp_mae)
                history["dbp_mae"].append(val_eval.dbp_mae)
                history["sbp_pearson_r"].append(val_eval.sbp_pearson_r)
                history["dbp_pearson_r"].append(val_eval.dbp_pearson_r)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping triggered at epoch {epoch}")
                        break

        # Restore best weights if available
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return history

    def save_checkpoint(self, path: str) -> None:
        """Save model and training state dictionary."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        config_dict = {
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "epochs": self.config.epochs,
            "patience": self.config.patience,
            "delta": self.config.delta,
            "lambda_1": self.config.lambda_1,
            "lambda_2": self.config.lambda_2,
            "sigma_thresh": self.config.sigma_thresh,
            "device": str(self.config.device)
        }
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": config_dict
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model state from saved checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
