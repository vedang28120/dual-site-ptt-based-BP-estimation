"""
Anti-Collapse Tripartite Compound Loss Function.

Mathematical Formulation (Milestone 3 & Requirement R3):
    L_total = L_Huber(delta=1.0) + lambda_1 * (1 - Pearson_r) + lambda_2 * L_Variance

Where:
    1. L_Huber: Robust base error penalty with smooth quadratic convergence for |e| <= delta
       and linear gradient for outliers (|e| > delta).
    2. L_Pearson: Dynamic correlation penalty (1 - r) forcing dynamic tracking across subjects
       and preventing static mean predictions.
    3. L_Variance: Distribution spread enforcement:
       L_Variance = |1 - sigma(y_hat)/sigma(y)| + max(0, 1 - sigma^2(y_hat)/sigma_thresh^2)
       Actively penalizes variance collapse / regression-to-the-mean, producing steep gradients
       that repel degenerate constant predictions.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripartiteCompoundLoss(nn.Module):
    """
    PyTorch implementation of the Tripartite Compound Anti-Collapse Loss.
    
    Operates on both single-target [B, 1] / [B] and dual-target [B, 2] (SBP, DBP).
    """
    def __init__(
        self,
        delta: float = 1.0,
        lambda_1: float = 0.40,
        lambda_2: float = 0.25,
        sigma_thresh: float = 8.0,
        eps: float = 1e-8
    ) -> None:
        super().__init__()
        self.delta = delta
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.sigma_thresh = sigma_thresh
        self.eps = eps
        self.huber = nn.HuberLoss(delta=delta, reduction="mean")

    def _compute_single_channel_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute tripartite loss for a 1D tensor [B]."""
        pred = pred.view(-1)
        target = target.view(-1)

        # 1. Huber Loss
        l_huber = self.huber(pred, target)

        # 2. Pearson Correlation Loss
        vx = pred - torch.mean(pred)
        vy = target - torch.mean(target)
        norm_x = torch.sqrt(torch.sum(vx ** 2) + self.eps)
        norm_y = torch.sqrt(torch.sum(vy ** 2) + self.eps)
        r = torch.sum(vx * vy) / (norm_x * norm_y)
        l_pearson = 1.0 - r

        # 3. Variance Regularization Loss
        if pred.numel() <= 1:
            std_pred = torch.tensor(self.eps, device=pred.device, dtype=pred.dtype)
            std_target = torch.tensor(self.eps, device=target.device, dtype=target.dtype)
        else:
            std_pred = torch.std(pred) + self.eps
            std_target = torch.std(target) + self.eps
        spread_penalty = torch.abs(1.0 - (std_pred / std_target))
        floor_penalty = torch.clamp(1.0 - ((std_pred ** 2) / (self.sigma_thresh ** 2)), min=0.0)
        l_variance = spread_penalty + floor_penalty

        # Total combined loss
        total_loss = l_huber + self.lambda_1 * l_pearson + self.lambda_2 * l_variance

        metrics = {
            "loss_total": float(total_loss.detach().item()),
            "loss_huber": float(l_huber.detach().item()),
            "loss_pearson": float(l_pearson.detach().item()),
            "loss_variance": float(l_variance.detach().item()),
            "pearson_r": float(r.detach().item()),
            "std_pred": float(std_pred.detach().item()),
            "std_target": float(std_target.detach().item()),
            "variance_ratio": float(((std_pred / std_target) ** 2).detach().item())
        }
        return total_loss, metrics

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass for compound loss.
        
        Args:
            pred: Predicted blood pressure [B, 1] or [B, 2] or [B]
            target: Ground truth blood pressure [B, 1] or [B, 2] or [B]
            
        Returns:
            Tuple of (total_loss_tensor, metrics_dict)
        """
        if pred.ndim == 2 and pred.shape[-1] == 2:
            # Dual target: SBP (col 0) and DBP (col 1)
            loss_sbp, met_sbp = self._compute_single_channel_loss(pred[:, 0], target[:, 0])
            loss_dbp, met_dbp = self._compute_single_channel_loss(pred[:, 1], target[:, 1])
            total_loss = 0.5 * (loss_sbp + loss_dbp)

            metrics = {
                "loss_total": float(total_loss.detach().item()),
                "loss_huber": 0.5 * (met_sbp["loss_huber"] + met_dbp["loss_huber"]),
                "loss_pearson": 0.5 * (met_sbp["loss_pearson"] + met_dbp["loss_pearson"]),
                "loss_variance": 0.5 * (met_sbp["loss_variance"] + met_dbp["loss_variance"]),
                "sbp_pearson_r": met_sbp["pearson_r"],
                "dbp_pearson_r": met_dbp["pearson_r"],
                "sbp_variance_ratio": met_sbp["variance_ratio"],
                "dbp_variance_ratio": met_dbp["variance_ratio"],
            }
            return total_loss, metrics
        else:
            return self._compute_single_channel_loss(pred, target)


# Aliased for backward compatibility with E2E references
PyTorchTripartiteLoss = TripartiteCompoundLoss


class NumpyTripartiteLoss:
    """
    NumPy analytical reference for tripartite compound loss.
    
    Provides offline evaluation and unit testing parity against numpy arrays.
    """
    def __init__(
        self,
        delta: float = 1.0,
        lambda_1: float = 0.40,
        lambda_2: float = 0.25,
        sigma_thresh: float = 8.0,
        eps: float = 1e-8
    ) -> None:
        self.delta = delta
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.sigma_thresh = sigma_thresh
        self.eps = eps

    def compute_loss(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list]
    ) -> Dict[str, float]:
        """
        Compute loss components across ground truth and predictions.
        
        Args:
            y_true: Ground truth blood pressure array
            y_pred: Predicted blood pressure array
            
        Returns:
            Dictionary containing loss components and diagnostics.
        """
        y_true = np.asarray(y_true, dtype=np.float64).flatten()
        y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

        # 1. Huber Loss
        diff = np.abs(y_true - y_pred)
        huber_elem = np.where(
            diff <= self.delta,
            0.5 * (diff ** 2),
            self.delta * (diff - 0.5 * self.delta)
        )
        loss_huber = float(np.mean(huber_elem))

        # 2. Pearson Correlation Loss
        vy_true = y_true - np.mean(y_true)
        vy_pred = y_pred - np.mean(y_pred)
        denom = (np.sqrt(np.sum(vy_true ** 2)) * np.sqrt(np.sum(vy_pred ** 2))) + self.eps
        r = float(np.sum(vy_true * vy_pred) / denom)
        loss_pearson = float(1.0 - r)

        # 3. Variance Regularization Loss
        std_true = float(np.std(y_true) + self.eps)
        std_pred = float(np.std(y_pred) + self.eps)
        variance_ratio = (std_pred ** 2) / (std_true ** 2)
        spread_penalty = abs(1.0 - (std_pred / std_true))
        floor_penalty = max(0.0, 1.0 - ((std_pred ** 2) / (self.sigma_thresh ** 2)))
        loss_variance = float(spread_penalty + floor_penalty)

        total_loss = float(loss_huber + self.lambda_1 * loss_pearson + self.lambda_2 * loss_variance)

        return {
            "total_loss": total_loss,
            "loss_huber": loss_huber,
            "loss_pearson": loss_pearson,
            "loss_variance": loss_variance,
            "pearson_r": r,
            "variance_ratio": variance_ratio,
            "std_true": std_true,
            "std_pred": std_pred,
        }


# Aliased for backward compatibility with E2E references
ReferenceTripartiteLoss = NumpyTripartiteLoss
