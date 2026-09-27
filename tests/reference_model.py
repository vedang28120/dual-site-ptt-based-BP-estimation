"""
Reference Neural Architecture: Dual-Stream Causal TCN with Causal Cross-Attention (DS-TCN)
Author: test_writer_1 (E2E Testing Architect)

Implements the exact neural specification from PROJECT.md and Explorer Survey 3:
- Stream A: Contact Finger PPG (Morphology, High SNR)
- Stream B: Facial rPPG (Macro-Foot Timing, POS filtered)
- Causal Cross-Attention Kernel aligning morphology with transit delay
- HRVP Physiological Anchor Embedding (z_HRVP in R^6)
- Decoupled SBP and DBP regression heads
- PyTorch Tripartite Compound Loss (Huber + Pearson + Variance)
"""

import math
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D Causal Convolution with left-padding to ensure no future information leaks."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class DilatedCausalBlock(nn.Module):
    """Residual block with dilated causal convolution, batchnorm, and GELU."""
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.GELU()
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + res)


class CausalCrossAttention(nn.Module):
    """
    Causal Cross-Attention module:
    Query (Q) from Morphology Stream (Finger PPG)
    Key (K) and Value (V) from Timing Stream (Face rPPG)
    Enforces causal masking so that finger arrivals attend only to prior facial wave feet.
    """
    def __init__(self, query_dim: int = 64, key_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

    def forward(self, q_tensor: torch.Tensor, kv_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q_tensor: [B, T, query_dim], kv_tensor: [B, T, key_dim]
        B, T, _ = q_tensor.shape

        Q = self.q_proj(q_tensor).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv_tensor).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv_tensor).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale # [B, num_heads, T, T]

        # Enforce causal mask: position i can only attend to j <= i
        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=q_tensor.device), diagonal=1)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V) # [B, num_heads, T, head_dim]
        context = context.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.out_proj(context)

        return out, attn_weights


class DualStreamTCN(nn.Module):
    """
    Dual-Stream Temporal Convolutional Network with Causal Cross-Attention and HRVP Anchor Fusion.
    Total parameter count: ~190,000.
    """
    def __init__(self):
        super().__init__()
        # Stream A (Finger Contact PPG - Morphology)
        self.in_conv_a = CausalConv1d(1, 64, kernel_size=5, dilation=1)
        self.tcn_a_blocks = nn.ModuleList([
            DilatedCausalBlock(64, kernel_size=5, dilation=1),
            DilatedCausalBlock(64, kernel_size=5, dilation=2),
            DilatedCausalBlock(64, kernel_size=5, dilation=4),
            DilatedCausalBlock(64, kernel_size=5, dilation=8)
        ])

        # Stream B (Face rPPG - Macro-Foot Timing)
        self.in_conv_b = CausalConv1d(1, 32, kernel_size=5, dilation=1)
        self.tcn_b_blocks = nn.ModuleList([
            DilatedCausalBlock(32, kernel_size=5, dilation=1),
            DilatedCausalBlock(32, kernel_size=5, dilation=2),
            DilatedCausalBlock(32, kernel_size=5, dilation=4),
            DilatedCausalBlock(32, kernel_size=5, dilation=8)
        ])

        # Cross Attention
        self.cross_attn = CausalCrossAttention(query_dim=64, key_dim=32, num_heads=4)

        # HRVP Anchor MLP (Ingests [ePWV, dPTT, dP_hydro, compliance, E0, alpha])
        self.anchor_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU()
        )

        # Decoupled SBP and DBP heads
        # Bottleneck: 64 (pooled optical context) + 32 (anchor) = 96
        self.sbp_head = nn.Sequential(
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 1)
        )
        self.dbp_head = nn.Sequential(
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 1)
        )

    def forward(
        self,
        morph_input: torch.Tensor,   # [B, 1, 300]
        timing_input: torch.Tensor,  # [B, 1, 300]
        anchor_input: torch.Tensor   # [B, 6]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Stream A
        ha = self.in_conv_a(morph_input)
        for block in self.tcn_a_blocks:
            ha = block(ha) # [B, 64, 300]

        # Stream B
        hb = self.in_conv_b(timing_input)
        for block in self.tcn_b_blocks:
            hb = block(hb) # [B, 32, 300]

        # Transpose to [B, T, C] for attention
        qa = ha.transpose(1, 2) # [B, 300, 64]
        kvb = hb.transpose(1, 2) # [B, 300, 32]

        attn_out, _ = self.cross_attn(qa, kvb) # [B, 300, 64]

        # Global temporal average pooling
        pooled_optical = torch.mean(attn_out, dim=1) # [B, 64]

        # Anchor projection
        anchor_features = self.anchor_mlp(anchor_input) # [B, 32]

        # Fused bottleneck
        fused = torch.cat([pooled_optical, anchor_features], dim=-1) # [B, 96]

        sbp = self.sbp_head(fused) # [B, 1]
        dbp = self.dbp_head(fused) # [B, 1]

        return sbp, dbp


class PyTorchTripartiteLoss(nn.Module):
    """
    Tripartite Compound Loss in PyTorch:
    L = L_Huber(delta=1.0) + lambda_1 * (1 - Pearson_r) + lambda_2 * L_Variance
    """
    def __init__(self, delta: float = 1.0, lambda_1: float = 0.40, lambda_2: float = 0.25, sigma_thresh: float = 8.0):
        super().__init__()
        self.delta = delta
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.sigma_thresh = sigma_thresh
        self.huber = nn.HuberLoss(delta=delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Huber component
        l_huber = self.huber(pred, target)

        # Pearson correlation component
        vx = pred - torch.mean(pred)
        vy = target - torch.mean(target)
        denom = torch.sqrt(torch.sum(vx ** 2) + 1e-8) * torch.sqrt(torch.sum(vy ** 2) + 1e-8)
        r = torch.sum(vx * vy) / denom
        l_pearson = 1.0 - r

        # Variance component
        std_pred = torch.std(pred) + 1e-8
        std_target = torch.std(target) + 1e-8
        spread_penalty = torch.abs(1.0 - (std_pred / std_target))
        floor_penalty = torch.clamp(1.0 - ((std_pred ** 2) / (self.sigma_thresh ** 2)), min=0.0)
        l_variance = spread_penalty + floor_penalty

        total_loss = l_huber + self.lambda_1 * l_pearson + self.lambda_2 * l_variance

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_huber": float(l_huber.item()),
            "loss_pearson": float(l_pearson.item()),
            "loss_variance": float(l_variance.item()),
            "pearson_r": float(r.item()),
            "std_pred": float(std_pred.item()),
            "std_target": float(std_target.item())
        }

        return total_loss, metrics
