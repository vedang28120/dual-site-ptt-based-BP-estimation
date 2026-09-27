"""
Dual-Stream Temporal Convolutional Network with Causal Cross-Attention (DS-TCN).

Architectural Specification (Milestone 3 & Requirement R3):
- Stream A: Contact Finger PPG (Morphology, High SNR, Dicrotic Notch)
    - 4x Dilated Causal Conv1D blocks (dilations: 1, 2, 4, 8; kernel: 5; channels: 64)
- Stream B: Facial rPPG (Macro-Foot Timing, POS filtered)
    - 4x Dilated Causal Conv1D blocks (dilations: 1, 2, 4, 8; kernel: 5; channels: 32)
- Causal Cross-Attention:
    - Query (Q): Finger morphology (64 channels)
    - Key (K) & Value (V): Facial timing (32 channels projected to 64)
    - Upper-triangular causal masking (j <= i) enforcing temporal causality
- HRVP Physiological Anchor MLP:
    - Ingests standardized 6D vector z_HRVP = [ePWV, dPTT, dP_hydro, d(ePWV)/dP, E0, alpha]
    - Linear(6, 32) -> GELU -> Linear(32, 32) -> GELU
- Bottleneck Fusion:
    - Concatenates globally pooled optical context (64) + HRVP anchor embedding (32) = 96 channels
- Decoupled Prediction Heads:
    - SBP Head: Linear(96, 48) -> GELU -> Linear(48, 1)
    - DBP Head: Linear(96, 48) -> GELU -> Linear(48, 1)
- Receptive Field: 61 samples (~2.03 seconds at 30 Hz)
- Parameters: ~230k (< 250k mobile budget)
- FLOPs: ~53.7 MFLOPs
"""

from __future__ import annotations

import math
from typing import Tuple, Dict, Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    1D Causal Convolution with left-padding to ensure no future information leaks.
    
    Guarantees that output at time step t only depends on input time steps <= t.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for causal convolution.
        
        Args:
            x: Input tensor of shape [B, in_channels, T]
            
        Returns:
            Output tensor of shape [B, out_channels, T]
        """
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class DilatedCausalBlock(nn.Module):
    """
    Residual block with dilated causal convolutions, batch normalization, and GELU activations.
    
    Structure:
      x -> CausalConv1d -> BatchNorm1d -> GELU -> CausalConv1d -> BatchNorm1d -> (+ residual) -> GELU
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv1 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.GELU()

        self.conv2 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual skip connection.
        
        Args:
            x: Input tensor of shape [B, channels, T]
            
        Returns:
            Output tensor of shape [B, channels, T]
        """
        res = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + res)


class CausalCrossAttention(nn.Module):
    """
    Multi-Head Causal Cross-Attention module.
    
    Query (Q) originates from finger morphology stream (rich dicrotic notch, high SNR).
    Key (K) & Value (V) originate from facial timing stream (macro-foot timing).
    Enforces causal masking so that finger arrivals at time i can only attend to
    facial arrivals at times j <= i.
    """
    def __init__(
        self,
        query_dim: int = 64,
        key_dim: int = 32,
        num_heads: int = 4
    ) -> None:
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError(f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads})")

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

    def forward(
        self,
        q_tensor: torch.Tensor,
        kv_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward cross-attention pass.
        
        Args:
            q_tensor: Query tensor [B, T, query_dim] (Morphology stream)
            kv_tensor: Key/Value tensor [B, T, key_dim] (Timing stream)
            
        Returns:
            out: Context tensor [B, T, query_dim]
            attn_weights: Softmax attention map [B, num_heads, T, T]
        """
        B, T, _ = q_tensor.shape

        # Linear projections and reshape for multi-head attention: [B, num_heads, T, head_dim]
        Q = self.q_proj(q_tensor).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv_tensor).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv_tensor).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product scores: [B, num_heads, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Enforce causal upper-triangular mask: position i can only attend to j <= i
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=q_tensor.device, dtype=scores.dtype),
            diagonal=1
        )
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)

        # Handle edge cases (e.g. single sample or NaNs)
        if torch.isnan(attn_weights).any():
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        context = torch.matmul(attn_weights, V)  # [B, num_heads, T, head_dim]
        context = context.transpose(1, 2).contiguous().view(B, T, self.query_dim)
        out = self.out_proj(context)

        return out, attn_weights


class DualStreamTCN(nn.Module):
    """
    Dual-Stream Temporal Convolutional Network with Causal Cross-Attention and HRVP Anchor Fusion.
    
    Inputs:
      - morph_input: [B, 1, T] or [B, T] (Contact finger PPG morphology)
      - timing_input: [B, 1, T] or [B, T] (Facial rPPG macro-foot timing)
      - anchor_input: [B, 6] (Standardized 6D HRVP anchor vector)
      
    Outputs:
      - sbp_pred: [B, 1] (Predicted Systolic Blood Pressure in mmHg)
      - dbp_pred: [B, 1] (Predicted Diastolic Blood Pressure in mmHg)
    """
    def __init__(self, model_tier: str = "compact") -> None:
        super().__init__()
        self.model_tier = model_tier.lower()

        if self.model_tier == "pro":
            # Tier 2: Pro Deep Architecture (2.47M parameters)
            # Receptive Field: 505 samples (~16.8 seconds at 30 Hz)
            # Multi-Head Causal Cross-Attention: 8 heads
            ch_a = 160
            ch_b = 112
            dilations = [1, 2, 4, 8, 16, 32]
            n_heads = 8
            anchor_dim = 64
            head_hidden = 128
        else:
            # Tier 1: Ultra-Lite Compact Architecture (230k parameters)
            # Receptive Field: 61 samples (~2.03 seconds at 30 Hz)
            # Multi-Head Causal Cross-Attention: 4 heads
            ch_a = 64
            ch_b = 32
            dilations = [1, 2, 4, 8]
            n_heads = 4
            anchor_dim = 32
            head_hidden = 48

        # Stream A (Finger Contact PPG - Morphology, High SNR)
        self.in_conv_a = CausalConv1d(1, ch_a, kernel_size=5, dilation=1)
        self.tcn_a_blocks = nn.ModuleList([
            DilatedCausalBlock(ch_a, kernel_size=5, dilation=d)
            for d in dilations
        ])

        # Stream B (Face rPPG - Macro-Foot Timing, POS filtered)
        self.in_conv_b = CausalConv1d(1, ch_b, kernel_size=5, dilation=1)
        self.tcn_b_blocks = nn.ModuleList([
            DilatedCausalBlock(ch_b, kernel_size=5, dilation=d)
            for d in dilations
        ])

        # Causal Cross-Attention Kernel
        self.cross_attn = CausalCrossAttention(query_dim=ch_a, key_dim=ch_b, num_heads=n_heads)

        # HRVP Physiological Anchor MLP (Ingests [ePWV, dPTT, dP_hydro, compliance, E0, alpha])
        self.anchor_mlp = nn.Sequential(
            nn.Linear(6, anchor_dim),
            nn.GELU(),
            nn.Linear(anchor_dim, anchor_dim),
            nn.GELU(),
        )

        # Decoupled SBP and DBP heads
        bottleneck_dim = ch_a + anchor_dim
        self.sbp_head = nn.Sequential(
            nn.Linear(bottleneck_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )
        self.dbp_head = nn.Sequential(
            nn.Linear(bottleneck_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(
        self,
        morph_input: torch.Tensor,
        timing_input: torch.Tensor,
        anchor_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward inference pass for dual-stream BP estimation.
        
        Args:
            morph_input: [B, 1, T] or [B, T]
            timing_input: [B, 1, T] or [B, T]
            anchor_input: [B, 6] or [6]
            
        Returns:
            Tuple of (sbp_pred, dbp_pred), each shaped [B, 1]
        """
        # Ensure 3D input [B, 1, T]
        if morph_input.ndim == 2:
            morph_input = morph_input.unsqueeze(1)
        if timing_input.ndim == 2:
            timing_input = timing_input.unsqueeze(1)
        if anchor_input.ndim == 1:
            anchor_input = anchor_input.unsqueeze(0)

        # Stream A: Finger PPG morphology feature extraction
        ha = self.in_conv_a(morph_input)
        for block in self.tcn_a_blocks:
            ha = block(ha)  # [B, 64, T]

        # Stream B: Facial rPPG timing feature extraction
        hb = self.in_conv_b(timing_input)
        for block in self.tcn_b_blocks:
            hb = block(hb)  # [B, 32, T]

        # Transpose to [B, T, C] for cross-attention
        qa = ha.transpose(1, 2)   # [B, T, 64]
        kvb = hb.transpose(1, 2)  # [B, T, 32]

        attn_out, _ = self.cross_attn(qa, kvb)  # [B, T, 64]

        # Global temporal average pooling across the sequence dimension
        pooled_optical = torch.mean(attn_out, dim=1)  # [B, 64]

        # Anchor projection
        anchor_features = self.anchor_mlp(anchor_input)  # [B, 32]

        # Bottleneck concatenation
        fused = torch.cat([pooled_optical, anchor_features], dim=-1)  # [B, 96]

        # Decoupled SBP and DBP heads
        sbp = self.sbp_head(fused)  # [B, 1]
        dbp = self.dbp_head(fused)  # [B, 1]

        return sbp, dbp

    def get_attention_weights(
        self,
        morph_input: torch.Tensor,
        timing_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract causal cross-attention weight maps for timing delay inspection.
        
        Returns:
            Attention map tensor of shape [B, num_heads, T, T]
        """
        if morph_input.ndim == 2:
            morph_input = morph_input.unsqueeze(1)
        if timing_input.ndim == 2:
            timing_input = timing_input.unsqueeze(1)

        ha = self.in_conv_a(morph_input)
        for block in self.tcn_a_blocks:
            ha = block(ha)

        hb = self.in_conv_b(timing_input)
        for block in self.tcn_b_blocks:
            hb = block(hb)

        qa = ha.transpose(1, 2)
        kvb = hb.transpose(1, 2)

        _, weights = self.cross_attn(qa, kvb)
        return weights

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def receptive_field() -> int:
        """
        Calculate total causal receptive field in samples.
        RF = 1 + 2 * (K - 1) * sum(d_i) for two convs per block = 1 + 4 * (1 + 2 + 4 + 8) = 61.
        At 30 FPS, 61 samples corresponds to ~2.03 seconds.
        """
        return 1 + 4 * (1 + 2 + 4 + 8)

    def estimate_flops(self, seq_len: int = 300) -> float:
        """
        Estimate FLOPs (Floating Point Operations) for a sequence of length seq_len.
        
        Returns:
            FLOPs in MegaFLOPs (MFLOPs).
        """
        flops = 0
        # In-convs
        flops += 2 * 1 * 64 * 5 * seq_len
        flops += 2 * 1 * 32 * 5 * seq_len
        # Stream A 4 blocks, each has 2 Conv1d(64, 64, k=5)
        for _ in range(4):
            flops += 2 * (2 * 64 * 64 * 5 * seq_len)
        # Stream B 4 blocks, each has 2 Conv1d(32, 32, k=5)
        for _ in range(4):
            flops += 2 * (2 * 32 * 32 * 5 * seq_len)
        # Attention projections + dot product + context
        flops += 2 * seq_len * 64 * 64  # Q
        flops += 2 * seq_len * 32 * 64  # K
        flops += 2 * seq_len * 32 * 64  # V
        flops += 2 * 4 * seq_len * seq_len * 16  # Q @ K^T
        flops += 2 * 4 * seq_len * seq_len * 16  # A @ V
        flops += 2 * seq_len * 64 * 64  # Out proj
        # Anchor MLP: 6 -> 32 -> 32
        flops += 2 * 6 * 32 + 2 * 32 * 32
        # Heads: 96 -> 48 -> 1 for SBP and DBP
        flops += 2 * (2 * 96 * 48 + 2 * 48 * 1)
        return float(flops / 1e6)
